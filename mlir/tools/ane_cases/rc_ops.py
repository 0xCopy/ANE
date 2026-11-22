"""rc_ (recurrent, random, detection, control-flow) cases and program-shape limits.

A: lstm / gru / rnn (seq 8,32 x hidden 64,128 x forward/bidirectional), random_* , non_maximum_suppression,
   cond, while_loop, make_list/list_*, shape, fill_like, squeeze.
B: program-shape limits: many inputs, many outputs, long elementwise chains, many constants, big tensors.
Random ops are not deterministic against the CPU: status 'wrong' is expected there; 'ok'/'wrong' both mean it compiled and ran.
"""

_UN16 = ["relu", "sigmoid", "tanh", "abs", "square", "floor", "ceil", "sign",
         "sin", "cos", "softplus", "softsign", "exp", "silu", "erf", "atan"]


def register(env):
    case, mb, np, f16, rs, SHAPE, X, XY = env.case, env.mb, env.np, env.f16, env.rs, env.SHAPE, env.X, env.XY

    # ------------------------------------------------------------------ A1: recurrent
    # MIL layouts: x [seq, batch, I]; initial_h [batch, H]; lstm initial_c [batch, H];
    # weights [G*H, I] / [G*H, H] (G = 4 lstm, 3 gru, 1 rnn), bias [G*H]; bidirectional adds *_back weights (lstm) or stacked (gru/rnn).
    I = 32

    def rec(kind, seq, H, bidir):
        G = {"lstm": 4, "gru": 3, "rnn": 1}[kind]
        ins = [("x", (seq, 1, I)), ("h", (1, H))]
        if kind == "lstm":
            ins.append(("c", (1, H)))

        def fn(*v):
            r = np.random.RandomState(7)
            w = lambda *s: f16(r.uniform(-0.3, 0.3, s))
            x, h = v[0], v[1]
            if kind == "lstm":
                c = v[2]
                kw = dict(x=x, initial_h=h, initial_c=c, weight_ih=w(G * H, I), weight_hh=w(G * H, H), bias=w(G * H),
                          direction="bidirectional" if bidir else "forward", output_sequence=True)
                if bidir:
                    kw.update(weight_ih_back=w(G * H, I), weight_hh_back=w(G * H, H), bias_back=w(G * H))
                if bidir == "stack":
                    kw = dict(x=x, initial_h=h, initial_c=c, weight_ih=w(2, G * H, I), weight_hh=w(2, G * H, H),
                              bias=w(2, G * H), direction="bidirectional", output_sequence=True)
                o = mb.lstm(**kw)
                return tuple(o) if isinstance(o, (list, tuple)) else (o,)
            wi = w(G * H, I)
            wh = w(G * H, H)
            b = w(G * H)
            kw = dict(x=x, initial_h=h, weight_ih=wi, weight_hh=wh, bias=b,
                      direction="reverse" if bidir else "forward", output_sequence=True)
            o = (mb.gru if kind == "gru" else mb.rnn)(**kw)
            return tuple(o) if isinstance(o, (list, tuple)) else (o,)

        return ins, fn

    for kind in ("lstm", "gru", "rnn"):
        for seq in (8, 32):
            for H in (64, 128):
                for bidir in (False, True):
                    ins, fn = rec(kind, seq, H, bidir)
                    nm = f"rc_{kind}_s{seq}_h{H}_{'rev' if bidir and kind != 'lstm' else ('bi' if bidir else 'fwd')}"
                    case(nm, "recurrent", ins, fn, kinds={"x": "small"}, tol=5e-2)
    # variants: smaller recurrence (seq 2, H 16) to see if ANE accepts any recurrent op at all
    for kind in ("lstm", "gru", "rnn"):
        ins, fn = rec(kind, 2, 16, False)
        case(f"rc_{kind}_s2_h16_fwd", "recurrent", ins, fn, kinds={"x": "small"}, tol=5e-2)
    ins, fn = rec("lstm", 2, 16, "stack")
    case("rc_lstm_s2_h16_bistack", "recurrent", ins, fn, kinds={"x": "small"}, tol=5e-2)

    # ------------------------------------------------------------------ A2: random (only 'compiled and ran' matters)
    SH = np.array([1, 64, 1, 32], np.int32)
    case("rc_random_bernoulli", "random", X,
         lambda x: mb.add(x=mb.mul(x=x, y=f16(0.0)), y=mb.random_bernoulli(shape=SH, prob=f16(0.5), seed=1)))
    case("rc_random_normal", "random", X,
         lambda x: mb.add(x=mb.mul(x=x, y=f16(0.0)), y=mb.random_normal(shape=SH, mean=f16(0.0), stddev=f16(1.0), seed=1)))
    case("rc_random_uniform", "random", X,
         lambda x: mb.add(x=mb.mul(x=x, y=f16(0.0)), y=mb.random_uniform(shape=SH, low=f16(0.0), high=f16(1.0), seed=1)))

    case("rc_random_normal_bare", "random", X,
         lambda x: mb.random_normal(shape=SH, mean=f16(0.0), stddev=f16(1.0), seed=1))
    case("rc_random_uniform_bare", "random", X,
         lambda x: mb.random_uniform(shape=SH, low=f16(0.0), high=f16(1.0), seed=1))

    def rcat(x):
        try:
            o = mb.random_categorical(x=x, mode="logits", size=8, seed=1)
        except TypeError:
            o = mb.random_categorical(logits=x, mode="logits", size=8, seed=1)
        return mb.cast(x=o, dtype="fp16")
    case("rc_random_categorical", "random", [("x", (4, 16))], rcat)

    # ------------------------------------------------------------------ A3: detection
    def nms(boxes, scores):
        o = mb.non_maximum_suppression(boxes=boxes, scores=scores, iou_threshold=np.float32(0.5),
                                       max_boxes=np.int32(4), per_class_suppression=False)
        o = list(o) if isinstance(o, (list, tuple)) else [o]
        return tuple(mb.cast(x=t, dtype="fp16") for t in o)
    case("rc_non_maximum_suppression", "detection", [("boxes", (1, 4, 8), "fp32"), ("scores", (1, 2, 8), "fp32")], nms,
         kinds={"boxes": "pos", "scores": "small"})

    # ------------------------------------------------------------------ A4: control flow
    def cond_fn(x):
        s = mb.reduce_sum(x=x, axes=[0, 1, 2, 3], keep_dims=False)
        pred = mb.less(x=s, y=f16(0.0))
        o = mb.cond(pred=pred, _true_fn=lambda: mb.relu(x=x), _false_fn=lambda: mb.mul(x=x, y=f16(-1.0)))
        return o[0] if isinstance(o, (list, tuple)) else o
    case("rc_cond", "control_flow", X, cond_fn)
    # variant: constant predicate (one change: no data-dependent pred)
    case("rc_cond_constpred", "control_flow", X,
         lambda x: (lambda o: o[0] if isinstance(o, (list, tuple)) else o)(
             mb.cond(pred=mb.const(val=np.array(True)), _true_fn=lambda: mb.relu(x=x), _false_fn=lambda: mb.mul(x=x, y=f16(-1.0)))))

    def wl(x):
        i0 = np.int32(0)
        def c(i, v):
            return mb.less(x=i, y=mb.const(val=np.int32(4)))
        def b(i, v):
            return (mb.add(x=i, y=mb.const(val=np.int32(1))), mb.mul(x=v, y=f16(1.1)))
        r = mb.while_loop(_cond=c, _body=b, loop_vars=(mb.const(val=i0), x))
        return r[1]
    case("rc_while_loop", "control_flow", X, wl)
    # variant: one iteration instead of four
    def wl1(x):
        def c(i, v):
            return mb.less(x=i, y=mb.const(val=np.int32(1)))
        def b(i, v):
            return (mb.add(x=i, y=mb.const(val=np.int32(1))), mb.mul(x=v, y=f16(1.1)))
        return mb.while_loop(_cond=c, _body=b, loop_vars=(mb.const(val=np.int32(0)), x))[1]
    case("rc_while_loop_1iter", "control_flow", X, wl1)

    def ml_wr(x):
        ls = mb.make_list(init_length=2, elem_shape=(1, 64, 1, 32), dtype="fp16")
        ls = mb.list_write(ls=ls, index=np.int32(0), value=x)
        return mb.list_read(ls=ls, index=np.int32(0))
    case("rc_make_list_write_read", "control_flow", X, ml_wr)
    # variant: no list at all, just the write value through a cast/relu (one change: no make_list/list_write)
    case("rc_make_list_none", "control_flow", X, lambda x: mb.relu(x=x))

    def ml_len(x):
        ls = mb.make_list(init_length=2, elem_shape=(1, 64, 1, 32), dtype="fp16")
        ls = mb.list_write(ls=ls, index=np.int32(0), value=x)
        return mb.cast(x=mb.list_length(ls=ls), dtype="fp16")
    case("rc_list_length", "control_flow", X, ml_len)

    # ------------------------------------------------------------------ A5: shape / fill / squeeze
    case("rc_shape", "shape", X, lambda x: mb.cast(x=mb.shape(x=x), dtype="fp16"))
    case("rc_shape_int32_out", "shape", X, lambda x: mb.shape(x=x))
    case("rc_fill_like", "shape", X, lambda x: mb.fill_like(ref_tensor=x, value=f16(1.5)))
    case("rc_squeeze", "shape", X, lambda x: mb.squeeze(x=x, axes=[0, 2]))

    # ------------------------------------------------------------------ B1: many inputs
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
        ins = [(f"x{i}", (1, 16, 1, 16)) for i in range(n)]
        def fin(*xs):
            if len(xs) == 1:
                return mb.relu(x=xs[0])
            acc = xs[0]
            for v in xs[1:]:
                acc = mb.add(x=acc, y=v)
            return acc
        case(f"rc_lim_inputs_{n}", "limit_inputs", ins, fin)

    # ------------------------------------------------------------------ B2: many outputs
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
        def fout(x, n=n):
            return tuple(getattr(mb, op)(x=x) for op in _UN16[:n])
        case(f"rc_lim_outputs_{n}", "limit_outputs", [("x", (1, 16, 1, 16))], fout, kinds={"x": "small"})

    # ------------------------------------------------------------------ B3: long chains, x = relu(x*0.99+0.01)
    # (0.99 not 1.01: with 1.01, 1.01^2048 overflows fp16 and the reference becomes inf)
    for N in (8, 16, 32, 64, 128, 256, 512, 768, 1024, 2048):
        def chain(x, N=N):
            for _ in range(N):
                x = mb.relu(x=mb.add(x=mb.mul(x=x, y=f16(0.99)), y=f16(0.01)))
            return x
        case(f"rc_lim_chain_{N}", "limit_chain", X, chain)

    # ------------------------------------------------------------------ B4: many constants
    for N in (1, 8, 32, 128, 512):
        def consts(x, N=N):
            r = np.random.RandomState(11)
            for _ in range(N):
                x = mb.add(x=x, y=f16(r.uniform(-0.01, 0.01, (1, 64, 1, 32))))
            return x
        case(f"rc_lim_consts_{N}", "limit_consts", X, consts)

    # ------------------------------------------------------------------ B5: big tensors, relu on (1,C,1,W), W=32
    for mbytes in (1, 4, 8, 16, 64, 256):
        W = 32
        C = mbytes * 1024 * 1024 // 2 // W
        case(f"rc_lim_big_{mbytes}mb", "limit_big", [("x", (1, C, 1, W))], lambda x: mb.relu(x=x))

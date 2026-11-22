"""lg_: legalization recipes.  Ops the ANE compiler rejects, rebuilt from ops it accepts (argsort, equal, cast, matmul with a dynamic or constant
operand, reduce_sum, slice, mul).  Each case is one recipe; `ok` means the recipe compiles on the ANE and equals the numpy meaning of the
rejected op.  Index inputs are integer-valued fp16 (kind idxf:N): the ANE reports int32 tensors as fp16 anyway."""


def register(env):
    case, mb, np, f16, rs, SHAPE, X, XY = env.case, env.mb, env.np, env.f16, env.rs, env.SHAPE, env.X, env.XY
    W, C = 32, 64
    iota_w = lambda: f16(np.arange(W).reshape(1, 1, 1, W))
    stable_desc = lambda x, ax: np.argsort(-x, axis=ax, kind="stable")
    stable_asc = lambda x, ax: np.argsort(x, axis=ax, kind="stable")

    # ---- argmax / argmin / topk indices = a slice of argsort
    case("lg_argmax_w", "order", X, lambda x: mb.slice_by_index(x=mb.argsort(x=x, axis=3, ascending=False), begin=(0, 0, 0, 0), end=(1, C, 1, 1)),
         tol=0, ref=lambda x: stable_desc(x, 3)[..., :1])
    case("lg_argmax_w_f16out", "order", X, lambda x: mb.cast(x=mb.slice_by_index(x=mb.argsort(x=x, axis=3, ascending=False), begin=(0, 0, 0, 0), end=(1, C, 1, 1)), dtype="fp16"),
         tol=0, ref=lambda x: stable_desc(x, 3)[..., :1])
    case("lg_argmin_c", "order", X, lambda x: mb.slice_by_index(x=mb.argsort(x=x, axis=1, ascending=True), begin=(0, 0, 0, 0), end=(1, 1, 1, W)),
         tol=0, ref=lambda x: stable_asc(x, 1)[:, :1])
    case("lg_topk_idx_w4", "order", X, lambda x: mb.slice_by_index(x=mb.argsort(x=x, axis=3, ascending=False), begin=(0, 0, 0, 0), end=(1, C, 1, 4)),
         tol=0, ref=lambda x: stable_desc(x, 3)[..., :4])

    # ---- topk values: one-hot of each selected index, multiplied with x and summed (no gather)
    def topk_vals(x, k=4):
        idx = mb.slice_by_index(x=mb.argsort(x=x, axis=3, ascending=False), begin=(0, 0, 0, 0), end=(1, C, 1, k))
        idxf = mb.cast(x=idx, dtype="fp16")
        cols = []
        for j in range(k):
            sel = mb.slice_by_index(x=idxf, begin=(0, 0, 0, j), end=(1, C, 1, j + 1))                   # [1, C, 1, 1]
            hot = mb.cast(x=mb.equal(x=sel, y=iota_w()), dtype="fp16")                                    # [1, C, 1, W]
            cols.append(mb.reduce_sum(x=mb.mul(x=hot, y=x), axes=[3], keep_dims=True))
        return mb.concat(values=tuple(cols), axis=3)
    case("lg_topk_val_w4", "order", X, topk_vals, tol=1e-3, ref=lambda x: -np.sort(-x, axis=3)[..., :4])

    # ---- gather along W with dynamic indices: x[..., idx] = x2d @ onehot(idx)
    J = 8
    def gather_w(x, idx):
        M = mb.cast(x=mb.equal(x=f16(np.arange(W).reshape(W, 1)), y=mb.reshape(x=idx, shape=(1, J))), dtype="fp16")        # [W, J]
        y = mb.matmul(x=mb.reshape(x=x, shape=(C, W)), y=M)                                                                   # [C, J]
        return mb.reshape(x=y, shape=(1, C, 1, J))
    case("lg_gather_w_dyn", "gather", [("x", SHAPE), ("idx", (1, 1, 1, J))], gather_w, kinds={"idx": "idxf:32"}, tol=2e-3,
         ref=lambda x, idx: np.take_along_axis(x, np.broadcast_to(idx.astype(np.int64), (1, C, 1, J)), axis=3))
    def gather_c(x, idx):                                                                                                   # rows (channels): onehot^T @ x
        M = mb.cast(x=mb.equal(x=mb.reshape(x=idx, shape=(J, 1)), y=f16(np.arange(C).reshape(1, C))), dtype="fp16")        # [J, C]
        y = mb.matmul(x=M, y=mb.reshape(x=x, shape=(C, W)))                                                                   # [J, W]
        return mb.reshape(x=y, shape=(1, J, 1, W))
    case("lg_gather_c_dyn", "gather", [("x", SHAPE), ("idx", (1, 1, 1, J))], gather_c, kinds={"idx": "idxf:64"}, tol=2e-3,
         ref=lambda x, idx: x[:, idx.reshape(-1).astype(np.int64)])


    # ---- what exactly breaks after argsort?  (int32 result consumed by another op)
    srt = lambda x: mb.argsort(x=x, axis=3, ascending=False)
    case("lg_argsort_alone", "order", X, srt, tol=0, ref=lambda x: stable_desc(x, 3))
    case("lg_argsort_cast16", "order", X, lambda x: mb.cast(x=srt(x), dtype="fp16"), tol=0, ref=lambda x: stable_desc(x, 3))
    case("lg_argsort_reshape", "order", X, lambda x: mb.reshape(x=srt(x), shape=(1, C, 1, W)), tol=0, ref=lambda x: stable_desc(x, 3))
    case("lg_argsort_slice", "order", X, lambda x: mb.slice_by_index(x=srt(x), begin=(0, 0, 0, 0), end=(1, C, 1, 4)), tol=0, ref=lambda x: stable_desc(x, 3)[..., :4])
    case("lg_argsort_identity", "order", X, lambda x: mb.identity(x=srt(x)), tol=0, ref=lambda x: stable_desc(x, 3))

    # ---- argmax / argmin / topk without an int32 intermediate: reduce_max, equal, an iota, reduce_min
    BIG = 1024.0
    def first_idx(hot, iota):                              # smallest index where hot == 1  (hot: fp16 0/1, iota broadcastable)
        return mb.reduce_min(x=mb.add(x=mb.mul(x=iota, y=hot), y=mb.mul(x=mb.sub(x=f16(1.0), y=hot), y=f16(BIG))), axes=[3], keep_dims=True)
    def argmax_w(x):
        hot = mb.cast(x=mb.equal(x=x, y=mb.reduce_max(x=x, axes=[3], keep_dims=True)), dtype="fp16")
        return first_idx(hot, iota_w())
    case("lg_argmax_w_iota", "order", X, argmax_w, tol=0, ref=lambda x: np.argmax(x, axis=3).reshape(1, C, 1, 1).astype(np.float64))
    def argmin_c(x):
        iota_c = f16(np.arange(C).reshape(1, C, 1, 1))
        hot = mb.cast(x=mb.equal(x=x, y=mb.reduce_min(x=x, axes=[1], keep_dims=True)), dtype="fp16")
        return mb.reduce_min(x=mb.add(x=mb.mul(x=iota_c, y=hot), y=mb.mul(x=mb.sub(x=f16(1.0), y=hot), y=f16(BIG))), axes=[1], keep_dims=True)
    case("lg_argmin_c_iota", "order", X, argmin_c, tol=0, ref=lambda x: np.argmin(x, axis=1).reshape(1, 1, 1, W).astype(np.float64))
    # topk without dead code: a round's index is computed only if it is an output or feeds the mask of the next round
    def topk_var(k, outs, sentinel, full=False):
        io = iota_full if full else iota_w
        def fn(x):
            vals, idxs = [], []
            for r in range(k):
                m = mb.reduce_max(x=x, axes=[3], keep_dims=True); vals.append(m)
                if outs in ("i", "vi") or r < k - 1:
                    hot = mb.cast(x=mb.equal(x=x, y=m), dtype="fp16"); i = first_idx(hot, iota_w()); idxs.append(i)
                    if r < k - 1: x = mb.add(x=x, y=mb.mul(x=mb.cast(x=mb.equal(x=io(), y=i), dtype="fp16"), y=f16(sentinel)))
            v = vals[0] if k == 1 else mb.concat(values=tuple(vals), axis=3)
            if outs == "v": return v
            ix = idxs[0] if k == 1 else mb.concat(values=tuple(idxs), axis=3)
            return ix if outs == "i" else (v, ix)
        return fn
    vref = lambda k: (lambda x: -np.sort(-x, axis=3)[..., :k]); iref = lambda k: (lambda x: stable_desc(x, 3)[..., :k].astype(np.float64))
    both = lambda k: (lambda x: [-np.sort(-x, axis=3)[..., :k], stable_desc(x, 3)[..., :k].astype(np.float64)])
    iota_full = lambda: f16(np.broadcast_to(np.arange(W).reshape(1, 1, 1, W), (1, C, 1, W)).copy())
    for k in (1, 2, 4, 8):
        case(f"lg_topk{k}_vals", "order", X, topk_var(k, "v", -60000.0), tol=1e-3, ref=vref(k))
        case(f"lg_topk{k}_idx", "order", X, topk_var(k, "i", -60000.0), tol=0, ref=iref(k))
        case(f"lg_topk{k}_both", "order", X, topk_var(k, "vi", -60000.0), tol=1e-3, ref=both(k))
    case("lg_topk4_vals_sentinel_m1000", "order", X, topk_var(4, "v", -1000.0), tol=1e-3, ref=vref(4))
    case("lg_topk4_vals_fullshape_iota", "order", X, topk_var(4, "v", -60000.0, full=True), tol=1e-3, ref=vref(4))

    # bisect the k=1 recipe stage by stage (each stage is a live output)
    def stage(upto, sentinel=-60000.0):
        def fn(x):
            m = mb.reduce_max(x=x, axes=[3], keep_dims=True); hot = mb.cast(x=mb.equal(x=x, y=m), dtype="fp16"); i = first_idx(hot, iota_w())
            if upto == "i": return i
            one = mb.cast(x=mb.equal(x=iota_full(), y=i), dtype="fp16")
            if upto == "one": return one
            sub = mb.mul(x=one, y=f16(sentinel))
            if upto == "mulc": return sub
            return mb.add(x=x, y=sub)
        return fn
    case("lg_bis_i", "order", X, stage("i"), tol=0, ref=lambda x: np.argmax(x, axis=3).reshape(1, C, 1, 1).astype(np.float64))
    case("lg_bis_one", "order", X, stage("one"), tol=0, ref=lambda x: (np.arange(W).reshape(1, 1, 1, W) == np.argmax(x, axis=3)[..., None]).astype(np.float64))
    case("lg_bis_mulc", "order", X, stage("mulc"), tol=1e-3, ref=lambda x: (np.arange(W).reshape(1, 1, 1, W) == np.argmax(x, axis=3)[..., None]) * -60000.0)
    case("lg_bis_mulc_m1000", "order", X, stage("mulc", -1000.0), tol=1e-3, ref=lambda x: (np.arange(W).reshape(1, 1, 1, W) == np.argmax(x, axis=3)[..., None]) * -1000.0)
    case("lg_bis_add_m1000", "order", X, stage("add", -1000.0), tol=1e-3, ref=lambda x: x + (np.arange(W).reshape(1, 1, 1, W) == np.argmax(x, axis=3)[..., None]) * -1000.0)
    case("lg_bis_add_m60000", "order", X, stage("add", -60000.0), tol=5, ref=lambda x: x + (np.arange(W).reshape(1, 1, 1, W) == np.argmax(x, axis=3)[..., None]) * -60000.0)
    # fp16 spacing near 60000 is 32, so the product is only reproduced to a relative 1e-3
    case("lg_mul_const_m60000", "unary", X, lambda x: mb.mul(x=x, y=f16(-60000.0)), kinds={"x": "unit"}, tol=1e-3, ref=lambda x: x * -60000.0)
    case("lg_mul_const_m30000", "unary", X, lambda x: mb.mul(x=x, y=f16(-30000.0)), kinds={"x": "unit"}, tol=1e-3, ref=lambda x: x * -30000.0)
    # dead code: ops whose results nothing uses (the converter runs with no passes, so they stay in model.mil)
    case("lg_dead_none", "unary", X, lambda x: mb.relu(x=x), tol=0, ref=lambda x: np.maximum(x, 0))
    def dead1(x):
        mb.exp(x=x); return mb.relu(x=x)
    case("lg_dead_exp_after_input", "unary", X, dead1, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead2(x):
        r = mb.relu(x=x); mb.exp(x=r); return r
    case("lg_dead_exp_after_live", "unary", X, dead2, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead3(x):
        r = mb.relu(x=x); d = mb.mul(x=mb.cast(x=mb.equal(x=r, y=f16(1.0)), dtype="fp16"), y=f16(-1000.0)); mb.add(x=x, y=d); return r
    case("lg_dead_chain3", "unary", X, dead3, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead4(x):
        m = mb.reduce_max(x=x, axes=[3], keep_dims=True); mb.exp(x=x); return m
    case("lg_dead_exp_beside_reduce", "reduce", X, dead4, tol=1e-3, ref=lambda x: x.max(axis=3, keepdims=True))
    def dead_eq(x):
        r = mb.relu(x=x); mb.equal(x=r, y=f16(1.0)); return r
    case("lg_dead_equal", "unary", X, dead_eq, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_eq_cast(x):
        r = mb.relu(x=x); mb.cast(x=mb.equal(x=r, y=f16(1.0)), dtype="fp16"); return r
    case("lg_dead_equal_cast", "unary", X, dead_eq_cast, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_mul_add(x):
        r = mb.relu(x=x); mb.add(x=x, y=mb.mul(x=r, y=f16(-1000.0))); return r
    case("lg_dead_mul_add", "unary", X, dead_mul_add, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_two(x):
        r = mb.relu(x=x); mb.exp(x=mb.exp(x=x)); return r
    case("lg_dead_exp_exp", "unary", X, dead_two, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_cmp_select(x):
        r = mb.relu(x=x); mb.select(cond=mb.greater(x=r, y=f16(0.0)), a=x, b=r); return r
    case("lg_dead_greater_select", "unary", X, dead_cmp_select, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_add_noconst(x):
        r = mb.relu(x=x); mb.add(x=x, y=r); return r
    case("lg_dead_add_noconst", "unary", X, dead_add_noconst, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_mul_noconst(x):
        r = mb.relu(x=x); mb.mul(x=x, y=x); return r
    case("lg_dead_mul_self", "unary", X, dead_mul_noconst, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_add_tensorconst(x):
        r = mb.relu(x=x); mb.add(x=x, y=f16(np.ones((1, C, 1, W)))); return r
    case("lg_dead_add_tensor_const", "unary", X, dead_add_tensorconst, tol=0, ref=lambda x: np.maximum(x, 0))
    def dead_sub_scalar_live_const(x):                                  # the same scalar const is also used by a live op
        r = mb.relu(x=x); c = f16(2.0); mb.add(x=x, y=c); return mb.mul(x=r, y=c)
    case("lg_dead_add_shared_const", "unary", X, dead_sub_scalar_live_const, tol=3e-3, ref=lambda x: np.maximum(x, 0) * 2.0)
    def dead_reduce(x):
        r = mb.relu(x=x); mb.reduce_sum(x=x, axes=[3], keep_dims=True); return r
    case("lg_dead_reduce", "unary", X, dead_reduce, tol=0, ref=lambda x: np.maximum(x, 0))
    # the broadcast-both suspicion on its own
    case("lg_equal_bcast_both", "compare", [("a", (1, 1, 1, W)), ("b", (1, C, 1, 1))], lambda a, b: mb.cast(x=mb.equal(x=a, y=b), dtype="fp16"), kinds={"a": "round", "b": "round"}, tol=0)
    case("lg_add_bcast_both", "binary", [("a", (1, 1, 1, W)), ("b", (1, C, 1, 1))], lambda a, b: mb.add(x=a, y=b), tol=3e-3)
    case("lg_mul_bcast_both", "binary", [("a", (1, 1, 1, W)), ("b", (1, C, 1, 1))], lambda a, b: mb.mul(x=a, y=b), tol=3e-3)

    # ---- one_hot, cumsum, band_part, scatter-add
    case("lg_one_hot", "gather", [("idx", (1, 1, 1, W))], lambda idx: mb.cast(x=mb.equal(x=mb.reshape(x=idx, shape=(1, 1, W, 1)), y=f16(np.arange(16).reshape(1, 1, 1, 16))), dtype="fp16"),
         kinds={"idx": "idxf:16"}, tol=0, ref=lambda idx: (idx.reshape(1, 1, W, 1) == np.arange(16).reshape(1, 1, 1, 16)).astype(np.float64))
    tri = np.triu(np.ones((W, W)))                                                                                          # T[i, j] = 1 for i <= j
    case("lg_cumsum_w", "reduce", X, lambda x: mb.reshape(x=mb.matmul(x=mb.reshape(x=x, shape=(C, W)), y=f16(tri)), shape=(1, C, 1, W)), kinds={"x": "small"}, tol=5e-3,
         ref=lambda x: np.cumsum(x, axis=3))
    case("lg_cumsum_c", "reduce", X, lambda x: mb.reshape(x=mb.matmul(x=f16(np.tril(np.ones((C, C)))), y=mb.reshape(x=x, shape=(C, W))), shape=(1, C, 1, W)), kinds={"x": "small"}, tol=5e-3,
         ref=lambda x: np.cumsum(x, axis=1))
    mask = np.tril(np.ones((W, W)))
    case("lg_band_part_lower", "shape", [("x", (1, 1, W, W))], lambda x: mb.mul(x=x, y=f16(mask.reshape(1, 1, W, W))), tol=0, ref=lambda x: x * mask.reshape(1, 1, W, W))
    def scatter_add(x, idx):                                                                                                # out[c] = sum over j with idx[j]==c of x[j]  (rows)
        M = mb.cast(x=mb.equal(x=f16(np.arange(C).reshape(C, 1)), y=mb.reshape(x=idx, shape=(1, J))), dtype="fp16")        # [C, J]
        src = mb.reshape(x=mb.slice_by_index(x=x, begin=(0, 0, 0, 0), end=(1, J, 1, W)), shape=(J, W))
        return mb.reshape(x=mb.matmul(x=M, y=src), shape=(1, C, 1, W))
    def scatter_ref(x, idx):
        out = np.zeros((1, C, 1, W)); src = x[:, :J]
        for j, i in enumerate(idx.reshape(-1).astype(np.int64)): out[0, i, 0] += src[0, j, 0]
        return out
    case("lg_scatter_add_c", "gather", [("x", SHAPE), ("idx", (1, 1, 1, J))], scatter_add, kinds={"idx": "idxf:64"}, tol=3e-3, ref=scatter_ref)

    # ---- the rejected elementwise math, recomposed (already in rd_comp_*; one line each here so the table is in one place)
    case("lg_tan", "comp", X, lambda x: mb.real_div(x=mb.sin(x=x), y=mb.cos(x=x)), kinds={"x": "small"}, tol=2e-2)
    case("lg_sinh", "comp", X, lambda x: mb.mul(x=mb.sub(x=mb.exp(x=x), y=mb.exp(x=mb.mul(x=x, y=f16(-1.0)))), y=f16(0.5)), kinds={"x": "small"}, tol=1e-2)
    case("lg_cosh", "comp", X, lambda x: mb.mul(x=mb.add(x=mb.exp(x=x), y=mb.exp(x=mb.mul(x=x, y=f16(-1.0)))), y=f16(0.5)), kinds={"x": "small"}, tol=1e-2)
    case("lg_logical_or", "logical", XY, lambda x, y: mb.cast(x=mb.logical_not(x=mb.logical_and(x=mb.logical_not(x=mb.cast(x=x, dtype="bool")), y=mb.logical_not(x=mb.cast(x=y, dtype="bool")))), dtype="fp16"),
         kinds={"x": "round", "y": "round"}, tol=0)
    case("lg_reduce_prod_w", "reduce", X, lambda x: mb.exp(x=mb.reduce_sum(x=mb.log(x=x), axes=[3], keep_dims=True)), kinds={"x": "near1"}, tol=0.03,
         ref=lambda x: np.prod(x, axis=3, keepdims=True))                       # exp(sum(log x)): about 1% relative error in fp16, and the range of fp16 applies

"""rd_: reductions, ordering (topk/argsort/cumsum/argmax), elementwise math at other ranks and as compositions, rounding family."""


def register(env):
    case, mb, np, f16, rs, SHAPE, X, XY = env.case, env.mb, env.np, env.f16, env.rs, env.SHAPE, env.X, env.XY
    RS = (2, 8, 4, 16)
    RX = [("x", RS)]

    # ---- reductions: every op, every axis set, keep_dims True/False, on a shape where every axis is > 1
    RED = {"sum": ("reduce_sum", "any"), "mean": ("reduce_mean", "any"), "prod": ("reduce_prod", "unit"),
           "min": ("reduce_min", "any"), "max": ("reduce_max", "any"), "l1_norm": ("reduce_l1_norm", "any"),
           "l2_norm": ("reduce_l2_norm", "any"), "log_sum": ("reduce_log_sum", "pos"),
           "log_sum_exp": ("reduce_log_sum_exp", "any"), "sum_square": ("reduce_sum_square", "any")}
    AX = {"ax0": [0], "ax1": [1], "ax2": [2], "ax3": [3], "ax12": [1, 2], "ax23": [2, 3], "ax123": [1, 2, 3]}
    for op, (fname, kind) in RED.items():
        for axn, ax in AX.items():
            for keep in (True, False):
                case(f"rd_red_{op}_{axn}_{'k' if keep else 'nk'}", "reduce", RX,
                     lambda x, f=fname, ax=ax, keep=keep: getattr(mb, f)(x=x, axes=ax, keep_dims=keep), kinds={"x": kind})

    # ---- the failing elementwise ops at ranks 2, 3, 5 (native op), plus reduce_prod and mod / logical at those ranks
    RANKS = {"r2": (64, 32), "r3": (4, 8, 16), "r5": (1, 16, 4, 8, 16)}
    UN = {"tan": ("tan", "small"), "asin": ("asin", "unit"), "acos": ("acos", "unit"),
          "sinh": ("sinh", "small"), "cosh": ("cosh", "small"), "atanh": ("atanh", "unit")}
    for rn, shp in RANKS.items():
        for nm, (op, kind) in UN.items():
            case(f"rd_{rn}_{nm}", "elem_rank", [("x", shp)], lambda x, op=op: getattr(mb, op)(x=x), kinds={"x": kind})
        case(f"rd_{rn}_mod", "elem_rank", [("x", shp), ("y", shp)], lambda x, y: mb.mod(x=x, y=y), kinds={"y": "pos"})
        for op in ("logical_or", "logical_xor"):
            case(f"rd_{rn}_{op}", "elem_rank", [("x", shp), ("y", shp)],
                 (lambda op: lambda x, y: mb.cast(x=getattr(mb, op)(x=mb.cast(x=x, dtype="bool"), y=mb.cast(x=y, dtype="bool")), dtype="fp16"))(op),
                 kinds={"x": "round", "y": "round"})
        ax = len(shp) - 1
        case(f"rd_{rn}_reduce_prod", "elem_rank", [("x", shp)], lambda x, ax=ax: mb.reduce_prod(x=x, axes=[ax], keep_dims=True), kinds={"x": "unit"})

    # ---- compositions of the failing ops from ops that compile (SHAPE)
    SF = X
    case("rd_comp_tan", "comp", SF, lambda x: mb.real_div(x=mb.sin(x=x), y=mb.cos(x=x)), kinds={"x": "small"})
    def _exp_pm(x):
        return mb.exp(x=x), mb.exp(x=mb.mul(x=x, y=f16(-1.0)))
    case("rd_comp_sinh", "comp", SF, lambda x: (lambda a, b: mb.mul(x=mb.sub(x=a, y=b), y=f16(0.5)))(*_exp_pm(x)), kinds={"x": "small"})
    case("rd_comp_cosh", "comp", SF, lambda x: (lambda a, b: mb.mul(x=mb.add(x=a, y=b), y=f16(0.5)))(*_exp_pm(x)), kinds={"x": "small"})
    def _asin(x):  # asin(x) = atan(x / sqrt(1 - x^2))
        return mb.atan(x=mb.real_div(x=x, y=mb.sqrt(x=mb.add(x=mb.mul(x=mb.square(x=x), y=f16(-1.0)), y=f16(1.0)))))
    case("rd_comp_asin", "comp", SF, lambda x: _asin(x), kinds={"x": "unit"})
    case("rd_comp_acos", "comp", SF, lambda x: mb.add(x=mb.mul(x=_asin(x), y=f16(-1.0)), y=f16(np.pi / 2)), kinds={"x": "unit"})
    case("rd_comp_atanh", "comp", SF,
         lambda x: mb.mul(x=mb.log(x=mb.real_div(x=mb.add(x=x, y=f16(1.0)), y=mb.add(x=mb.mul(x=x, y=f16(-1.0)), y=f16(1.0)))), y=f16(0.5)),
         kinds={"x": "unit"})
    case("rd_comp_mod_floor", "comp", XY,
         lambda x, y: mb.sub(x=x, y=mb.mul(x=mb.floor(x=mb.real_div(x=x, y=y)), y=y)), kinds={"y": "pos"})
    # x - floor(x / y) * y: where the quotient is within fp16 rounding of an integer, floor() may land on either side, so those elements (about
    # 0.1% of random inputs) are left out of the comparison; the rest must match numpy's mod.
    case("rd_comp_mod_fdiv", "comp", XY,
         lambda x, y: mb.sub(x=x, y=mb.mul(x=mb.floor_div(x=x, y=y), y=y)), kinds={"y": "pos"},
         ref=lambda x, y: np.mod(x, y), skip=lambda x, y: np.abs(x / y - np.rint(x / y)) < 0.02)
    def _lor(a, b):  # a or b  ==  not(not a and not b)
        return mb.logical_not(x=mb.logical_and(x=mb.logical_not(x=a), y=mb.logical_not(x=b)))
    case("rd_comp_logical_or", "comp", XY,
         lambda x, y: mb.cast(x=_lor(mb.cast(x=x, dtype="bool"), mb.cast(x=y, dtype="bool")), dtype="fp16"),
         kinds={"x": "round", "y": "round"})
    case("rd_comp_logical_xor", "comp", XY,
         lambda x, y: (lambda a, b: mb.cast(x=mb.logical_and(x=_lor(a, b), y=mb.logical_not(x=mb.logical_and(x=a, y=b))), dtype="fp16"))(
             mb.cast(x=x, dtype="bool"), mb.cast(x=y, dtype="bool")),
         kinds={"x": "round", "y": "round"})
    case("rd_comp_reduce_prod_w", "comp", SF, lambda x: mb.exp(x=mb.reduce_sum(x=mb.log(x=x), axes=[3], keep_dims=True)), kinds={"x": "pos"})
    case("rd_comp_reduce_prod_c", "comp", SF, lambda x: mb.exp(x=mb.reduce_sum(x=mb.log(x=x), axes=[1], keep_dims=True)), kinds={"x": "pos"})

    # ---- rounding family: exact .5 values and negatives.  The device rounds half away from zero (Core ML on the CPU: half to even), so
    # the round cases are checked against a numpy half-away reference: `ok` pins that behaviour.
    away = lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5)
    for op in ("round", "floor", "ceil", "sign"):
        case(f"rd_rnd_half_{op}", "round", X, lambda x, op=op: getattr(mb, op)(x=mb.mul(x=x, y=f16(0.5))), kinds={"x": "round"},
             **({"ref": lambda x: away(x * 0.5)} if op == "round" else {}))
    case("rd_rnd_nhalf_round", "round", X, lambda x: mb.round(x=mb.mul(x=x, y=f16(-0.5))), kinds={"x": "round"}, ref=lambda x: away(x * -0.5))
    case("rd_rnd_quarter_round", "round", X, lambda x: mb.round(x=mb.mul(x=x, y=f16(0.25))), kinds={"x": "round"}, ref=lambda x: away(x * 0.25))
    case("rd_rnd_raw_round", "round", X, lambda x: mb.round(x=x), kinds={"x": "round"}, ref=lambda x: away(x))
    case("rd_rnd_plus_quarter_round", "round", X, lambda x: mb.round(x=mb.add(x=x, y=f16(0.25))), kinds={"x": "round"}, ref=lambda x: away(x + 0.25))
    case("rd_rnd_p15_round", "round", X, lambda x: mb.round(x=mb.mul(x=x, y=f16(1.5))), kinds={"x": "round"}, ref=lambda x: away(x * 1.5))
    case("rd_rnd_p25_round", "round", X, lambda x: mb.round(x=mb.mul(x=x, y=f16(2.5))), kinds={"x": "round"}, ref=lambda x: away(x * 2.5))
    case("rd_rnd_any_round", "round", X, lambda x: mb.round(x=x), ref=lambda x: away(x))

    # ---- ordering: argsort (indices), sorted values via topk(k=N), topk, cumsum, argmax/argmin
    for ax in (1, 3):
        for asc in (True, False):
            tag = "asc" if asc else "desc"
            case(f"rd_argsort_ax{ax}_{tag}", "order", X, lambda x, ax=ax, asc=asc: mb.argsort(x=x, axis=ax, ascending=asc), tol=0, ref=lambda x, ax=ax, asc=asc: np.argsort(x if asc else -x, axis=ax, kind="stable"))
            case(f"rd_argsort_ties_ax{ax}_{tag}", "order", X, lambda x, ax=ax, asc=asc: mb.argsort(x=x, axis=ax, ascending=asc), tol=0, kinds={"x": "round"}, ref=lambda x, ax=ax, asc=asc: np.argsort(x if asc else -x, axis=ax, kind="stable"))
            case(f"rd_sortvals_ax{ax}_{tag}", "order", X, lambda x, ax=ax, asc=asc: mb.topk(x=x, k=(64 if ax == 1 else 32), axis=ax, ascending=asc)[0], tol=0)
    for k in (1, 4, 8):
        for ax in (1, 3):
            for asc in (True, False):
                case(f"rd_topk_k{k}_ax{ax}_{'asc' if asc else 'desc'}", "order", X,
                     lambda x, k=k, ax=ax, asc=asc: mb.topk(x=x, k=k, axis=ax, ascending=asc), tol=0)
    for ax in (1, 3):
        for ex in (False, True):
            for rev in (False, True):
                case(f"rd_cumsum_ax{ax}_ex{int(ex)}_rev{int(rev)}", "order", X,
                     lambda x, ax=ax, ex=ex, rev=rev: mb.cumsum(x=x, axis=ax, exclusive=ex, reverse=rev), kinds={"x": "small"})
    # ---- one-change variants of the failing/wrong ones
    case("rd_var_argmax_castout", "var", X, lambda x: mb.cast(x=mb.reduce_argmax(x=x, axis=3, keep_dims=True), dtype="fp16"), tol=0)
    case("rd_var_cumsum_r2", "var", [("x", (64, 32))], lambda x: mb.cumsum(x=x, axis=0), kinds={"x": "small"})
    case("rd_var_topk_k1_vals", "var", X, lambda x: mb.topk(x=x, k=1, axis=3)[0], tol=0)
    case("rd_var_topk_r2", "var", [("x", (64, 32))], lambda x: mb.topk(x=x, k=4, axis=0)[0], tol=0)
    for op in ("reduce_argmax", "reduce_argmin"):
        for ax in range(4):
            for keep in (True, False):
                case(f"rd_{op[7:]}_ax{ax}_{'k' if keep else 'nk'}", "order", RX,
                     lambda x, op=op, ax=ax, keep=keep: getattr(mb, op)(x=x, axis=ax, keep_dims=keep), tol=0)

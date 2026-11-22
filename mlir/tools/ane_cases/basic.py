"""Basic coverage: activations and elementwise math, comparison, reductions, normalisation, matmul, conv, pooling, shape ops,
resizing, attention, dtype round trips, ranks.  One case = one tiny program; see tools/ane_ops.py for how it is run and judged."""

def register(env):
    case, mb, np, f16, rs, SHAPE, X, XY = env.case, env.mb, env.np, env.f16, env.rs, env.SHAPE, env.X, env.XY
    # ---- unary: activations and elementwise math
    for nm, (op, kw, kind) in {
        "relu": ("relu", {}, "any"), "relu6": ("relu6", {}, "any"), "leaky_relu": ("leaky_relu", {"alpha": 0.1}, "any"), "elu": ("elu", {"alpha": 1.0}, "any"),
        "sigmoid": ("sigmoid", {}, "any"), "sigmoid_hard": ("sigmoid_hard", {"alpha": 0.2, "beta": 0.5}, "any"), "softplus": ("softplus", {}, "any"),
        "softsign": ("softsign", {}, "any"), "tanh": ("tanh", {}, "any"), "scaled_tanh": ("scaled_tanh", {"alpha": 2.0, "beta": 0.5}, "any"),
        "thresholded_relu": ("thresholded_relu", {"alpha": 1.0}, "any"), "clamped_relu": ("clamped_relu", {"alpha": 1.0, "beta": 0.5}, "any"),
        "linear_activation": ("linear_activation", {"alpha": 2.0, "beta": 1.0}, "any"),
        "gelu_exact": ("gelu", {"mode": "EXACT"}, "any"), "gelu_tanh": ("gelu", {"mode": "TANH_APPROXIMATION"}, "any"), "gelu_sigmoid": ("gelu", {"mode": "SIGMOID_APPROXIMATION"}, "any"),
        "silu": ("silu", {}, "any"), "exp": ("exp", {}, "small"), "exp2": ("exp2", {}, "small"), "log": ("log", {}, "pos"), "sqrt": ("sqrt", {}, "pos"), "rsqrt": ("rsqrt", {}, "pos"),
        "inverse": ("inverse", {}, "nz"), "abs": ("abs", {}, "any"), "sign": ("sign", {}, "any"), "floor": ("floor", {}, "any"), "ceil": ("ceil", {}, "any"), "round": ("round", {}, "any"),
        "sin": ("sin", {}, "any"), "cos": ("cos", {}, "any"), "tan": ("tan", {}, "small"), "asin": ("asin", {}, "unit"), "acos": ("acos", {}, "unit"), "atan": ("atan", {}, "any"),
        "sinh": ("sinh", {}, "small"), "cosh": ("cosh", {}, "small"), "atanh": ("atanh", {}, "unit"),
        "erf": ("erf", {}, "any"), "square": ("square", {}, "any"), "threshold": ("threshold", {"alpha": f16(0.5)}, "any"), "clip": ("clip", {"alpha": f16(-1.0), "beta": f16(1.0)}, "any"),
    }.items():
        case(nm, "unary", X, (lambda op, kw: lambda x: getattr(mb, op)(x=x, **kw))(op, kw), kinds={"x": kind},
             **({"ref": lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5)} if nm == "round" else {}))     # the device rounds half away from zero
    case("prelu", "unary", X, lambda x: mb.prelu(x=x, alpha=f16(np.full(64, 0.25))))
    # ---- binary, comparison, logical, select
    for nm, kinds in {"add": {}, "sub": {}, "mul": {}, "real_div": {"y": "nz"}, "floor_div": {"y": "nz"}, "pow": {"x": "pos", "y": "small"}, "maximum": {}, "minimum": {}, "mod": {"y": "pos"}}.items():
        case(nm, "binary", XY, (lambda nm: lambda x, y: getattr(mb, nm)(x=x, y=y))(nm), kinds=kinds)
    for nm in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"):
        case(nm, "compare", XY, (lambda nm: lambda x, y: mb.cast(x=getattr(mb, nm)(x=x, y=y), dtype="fp16"))(nm), kinds={"y": "round"})
    for nm in ("logical_and", "logical_or", "logical_xor"):
        case(nm, "logical", XY, (lambda nm: lambda x, y: mb.cast(x=getattr(mb, nm)(x=mb.cast(x=x, dtype="bool"), y=mb.cast(x=y, dtype="bool")), dtype="fp16"))(nm), kinds={"x": "round", "y": "round"})
    case("logical_not", "logical", X, lambda x: mb.cast(x=mb.logical_not(x=mb.cast(x=x, dtype="bool")), dtype="fp16"), kinds={"x": "round"})
    case("select", "logical", XY, lambda x, y: mb.select(cond=mb.greater(x=x, y=f16(0.0)), a=x, b=y))
    case("add_bcast_channel", "binary", [("x", SHAPE), ("y", (1, 64, 1, 1))], lambda x, y: mb.add(x=x, y=y))
    case("mul_bcast_width", "binary", [("x", SHAPE), ("y", (1, 1, 1, 32))], lambda x, y: mb.mul(x=x, y=y))
    case("add_scalar_const", "binary", X, lambda x: mb.add(x=x, y=f16(1.5)))
    case("mul_const_tensor", "binary", X, lambda x: mb.mul(x=x, y=f16(rs.uniform(-1, 1, (1, 64, 1, 1)))))
    # ---- reductions
    for nm, kind in {"reduce_sum": "any", "reduce_mean": "any", "reduce_prod": "unit", "reduce_min": "any", "reduce_max": "any", "reduce_l1_norm": "any", "reduce_l2_norm": "any",
                     "reduce_log_sum": "pos", "reduce_log_sum_exp": "any", "reduce_sum_square": "any"}.items():
        for ax, tag in (([3], "w"), ([1], "c"), ([2, 3], "hw")):
            case(f"{nm}_{tag}", "reduce", X, (lambda nm, ax: lambda x: getattr(mb, nm)(x=x, axes=ax, keep_dims=True))(nm, ax), kinds={"x": kind})
    case("reduce_sum_nokeep", "reduce", X, lambda x: mb.reduce_sum(x=x, axes=[3], keep_dims=False))
    case("argmax_c", "reduce", X, lambda x: mb.reduce_argmax(x=x, axis=1, keep_dims=True), tol=0)
    case("argmin_w", "reduce", X, lambda x: mb.reduce_argmin(x=x, axis=3, keep_dims=True), tol=0)
    case("topk_w4", "reduce", X, lambda x: mb.topk(x=x, k=4, axis=3), tol=0)
    case("cumsum_w", "reduce", X, lambda x: mb.cumsum(x=x, axis=3), kinds={"x": "small"})
    case("softmax_w", "norm", X, lambda x: mb.softmax(x=x, axis=3))
    case("softmax_c", "norm", X, lambda x: mb.softmax(x=x, axis=1))
    # ---- normalisation
    g, b = f16(rs.uniform(0.5, 1.5, 64)), f16(rs.uniform(-0.5, 0.5, 64))
    case("layer_norm_c", "norm", X, lambda x: mb.layer_norm(x=x, axes=[1], gamma=g, beta=b, epsilon=f16(1e-5)))
    case("layer_norm_w", "norm", X, lambda x: mb.layer_norm(x=x, axes=[3], gamma=f16(rs.uniform(0.5, 1.5, 32)), beta=f16(rs.uniform(-.5, .5, 32)), epsilon=f16(1e-5)))
    case("layer_norm_nogamma", "norm", X, lambda x: mb.layer_norm(x=x, axes=[1], epsilon=f16(1e-5)))
    case("instance_norm", "norm", X, lambda x: mb.instance_norm(x=x, gamma=g, beta=b, epsilon=f16(1e-5)))
    case("batch_norm", "norm", X, lambda x: mb.batch_norm(x=x, mean=f16(rs.uniform(-1, 1, 64)), variance=f16(rs.uniform(0.5, 2, 64)), gamma=g, beta=b, epsilon=f16(1e-5)))
    case("l2_norm", "norm", X, lambda x: mb.l2_norm(x=x, epsilon=f16(1e-6)))
    case("local_response_norm", "norm", X, lambda x: mb.local_response_norm(x=x, size=5, alpha=f16(1e-4), beta=f16(0.75), k=f16(1.0)))
    # ---- linear algebra
    w128 = f16(rs.uniform(-.3, .3, (128, 64))); b128 = f16(rs.uniform(-.3, .3, 128))
    case("linear", "matmul", [("x", (1, 32, 64))], lambda x: mb.linear(x=x, weight=w128, bias=b128))
    case("linear_nobias", "matmul", [("x", (1, 32, 64))], lambda x: mb.linear(x=x, weight=w128))
    case("matmul_const_w", "matmul", [("x", (1, 32, 64))], lambda x: mb.matmul(x=x, y=f16(rs.uniform(-.3, .3, (64, 48)))))
    case("matmul_dynamic", "matmul", [("x", (1, 32, 64)), ("y", (1, 64, 48))], lambda x, y: mb.matmul(x=x, y=y))
    case("matmul_dynamic_ty", "matmul", [("x", (1, 32, 64)), ("y", (1, 48, 64))], lambda x, y: mb.matmul(x=x, y=y, transpose_y=True))
    case("matmul_batched4", "matmul", [("x", (4, 32, 64)), ("y", (4, 64, 32))], lambda x, y: mb.matmul(x=x, y=y))
    case("matmul_rank4", "matmul", [("x", (1, 8, 32, 64)), ("y", (1, 8, 64, 32))], lambda x, y: mb.matmul(x=x, y=y))
    case("einsum_nchw_nwhu", "matmul", [("x", (1, 8, 32, 64)), ("y", (1, 64, 32, 48))], lambda x, y: mb.einsum(values=(x, y), equation="nchw,nwhu->nchu"),
         ref=lambda x, y: np.einsum("nchw,nwhu->nchu", x, y), tol=5e-3)
    # ---- convolution
    def cw(oc, ic, k, groups=1): return f16(rs.uniform(-.2, .2, (oc, ic // groups, *k)))
    CS = (1, 64, 8, 32)
    case("conv1x1", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(128, 64, (1, 1)), pad_type="valid"))
    case("conv1x1_bias", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(128, 64, (1, 1)), bias=f16(rs.uniform(-.3, .3, 128)), pad_type="valid"))
    case("conv3x3_same", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (3, 3)), pad_type="same"))
    case("conv3x3_valid", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (3, 3)), pad_type="valid"))
    case("conv3x3_stride2", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (3, 3)), strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1]))
    case("conv3x3_dilated2", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (3, 3)), dilations=[2, 2], pad_type="same"))
    case("conv_depthwise3x3", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (3, 3), 64), groups=64, pad_type="same"))
    case("conv_groups2", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (1, 1), 2), groups=2, pad_type="valid"))
    case("conv1x7", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=cw(64, 64, (1, 7)), pad_type="same"))
    case("conv1d_k3", "conv", [("x", (1, 64, 32))], lambda x: mb.conv(x=x, weight=f16(rs.uniform(-.2, .2, (64, 64, 3))), pad_type="same"))
    case("conv3d_k3", "conv", [("x", (1, 16, 4, 8, 16))], lambda x: mb.conv(x=x, weight=f16(rs.uniform(-.2, .2, (16, 16, 3, 3, 3))), pad_type="same"))
    case("conv_transpose2x2_s2", "conv", [("x", CS)], lambda x: mb.conv_transpose(x=x, weight=f16(rs.uniform(-.2, .2, (64, 32, 2, 2))), strides=[2, 2], pad_type="valid"))
    # ---- pooling
    case("avg_pool2x2", "pool", [("x", CS)], lambda x: mb.avg_pool(x=x, kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid"))
    case("max_pool2x2", "pool", [("x", CS)], lambda x: mb.max_pool(x=x, kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid"))
    case("max_pool3x3_same", "pool", [("x", CS)], lambda x: mb.max_pool(x=x, kernel_sizes=[3, 3], strides=[1, 1], pad_type="same"))
    case("l2_pool2x2", "pool", [("x", CS)], lambda x: mb.l2_pool(x=x, kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid"))
    case("global_avg_pool", "pool", [("x", CS)], lambda x: mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True))
    # ---- shape manipulation
    case("reshape", "shape", X, lambda x: mb.reshape(x=x, shape=(1, 32, 1, 64)))
    case("reshape_rank3", "shape", X, lambda x: mb.reshape(x=x, shape=(1, 64, 32)))
    case("transpose_0321", "shape", X, lambda x: mb.transpose(x=x, perm=(0, 3, 2, 1)))
    case("transpose_0231", "shape", [("x", CS)], lambda x: mb.transpose(x=x, perm=(0, 2, 3, 1)))
    case("concat_c", "shape", XY, lambda x, y: mb.concat(values=(x, y), axis=1))
    case("concat_w", "shape", XY, lambda x, y: mb.concat(values=(x, y), axis=3))
    case("split_c2", "shape", X, lambda x: mb.split(x=x, num_splits=2, axis=1))
    case("split_w_sizes", "shape", X, lambda x: mb.split(x=x, split_sizes=[8, 24], axis=3))
    case("stack", "shape", XY, lambda x, y: mb.stack(values=(x, y), axis=0))
    case("expand_dims", "shape", X, lambda x: mb.expand_dims(x=x, axes=[0]))
    case("squeeze", "shape", X, lambda x: mb.squeeze(x=x, axes=[2]))
    case("tile_c2", "shape", X, lambda x: mb.tile(x=x, reps=(1, 2, 1, 1)))
    case("pad_const", "shape", [("x", CS)], lambda x: mb.pad(x=x, pad=(1, 1, 2, 2), mode="constant", constant_val=f16(0.0)))
    case("pad_reflect", "shape", [("x", CS)], lambda x: mb.pad(x=x, pad=(1, 1, 2, 2), mode="reflect", constant_val=f16(0.0)))
    case("pad_replicate", "shape", [("x", CS)], lambda x: mb.pad(x=x, pad=(1, 1, 2, 2), mode="replicate", constant_val=f16(0.0)))
    case("slice_by_index", "shape", X, lambda x: mb.slice_by_index(x=x, begin=(0, 8, 0, 4), end=(1, 40, 1, 28)))
    case("slice_by_size", "shape", X, lambda x: mb.slice_by_size(x=x, begin=(0, 8, 0, 4), size=(1, 32, 1, 24)))
    case("slice_stride2", "shape", X, lambda x: mb.slice_by_index(x=x, begin=(0, 0, 0, 0), end=(1, 64, 1, 32), stride=(1, 2, 1, 1)))
    case("gather_c", "shape", X, lambda x: mb.gather(x=x, indices=np.array([5, 3, 60, 0], np.int32), axis=1))
    case("gather_dynamic", "shape", [("x", SHAPE), ("idx", (4,), "int32")], lambda x, idx: mb.gather(x=x, indices=idx, axis=1), kinds={"idx": "idx:64"}, tol=0)
    case("gather_along_axis", "shape", [("x", SHAPE), ("idx", SHAPE, "int32")], lambda x, idx: mb.gather_along_axis(x=x, indices=idx, axis=3), kinds={"idx": "idx:32"})
    case("flatten2d", "shape", X, lambda x: mb.flatten2d(x=x, axis=1))
    case("space_to_depth", "shape", [("x", (1, 16, 8, 32))], lambda x: mb.space_to_depth(x=x, block_size=2))
    case("depth_to_space", "shape", [("x", (1, 16, 8, 32))], lambda x: mb.depth_to_space(x=x, block_size=2))
    case("pixel_shuffle", "shape", [("x", (1, 16, 8, 32))], lambda x: mb.pixel_shuffle(x=x, upscale_factor=2))
    case("pixel_unshuffle", "shape", [("x", (1, 4, 16, 32))], lambda x: mb.pixel_unshuffle(x=x, downscale_factor=np.uint32(2)))
    case("reverse_w", "shape", X, lambda x: mb.reverse(x=x, axes=[3]))
    case("identity", "shape", X, lambda x: mb.identity(x=x))
    case("argsort_w", "reduce", X, lambda x: mb.argsort(x=x, axis=3, ascending=True), tol=0, ref=lambda x: np.argsort(x, axis=3, kind="stable"))
    case("one_hot", "shape", [("idx", (1, 32), "int32")], lambda idx: mb.cast(x=mb.one_hot(indices=idx, one_hot_vector_size=16), dtype="fp16"), kinds={"idx": "idx:16"})
    case("band_part", "shape", [("x", (1, 1, 32, 32))], lambda x: mb.band_part(x=x, lower=-1, upper=0))
    # ---- resizing
    RS = (1, 16, 8, 16)
    case("resize_bilinear", "resize", [("x", RS)], lambda x: mb.resize_bilinear(x=x, target_size_height=16, target_size_width=32))
    case("resize_nearest", "resize", [("x", RS)], lambda x: mb.resize_nearest_neighbor(x=x, target_size_height=16, target_size_width=32))
    case("upsample_bilinear", "resize", [("x", RS)], lambda x: mb.upsample_bilinear(x=x, scale_factor_height=2, scale_factor_width=2))
    case("upsample_nearest", "resize", [("x", RS)], lambda x: mb.upsample_nearest_neighbor(x=x, scale_factor_height=2, scale_factor_width=2))
    # ---- attention
    AQ = (1, 8, 32, 64)
    case("sdpa", "attention", [("q", AQ), ("k", AQ), ("v", AQ)], lambda q, k, v: mb.scaled_dot_product_attention(query=q, key=k, value=v), kinds={"q": "small", "k": "small", "v": "small"})
    case("sdpa_mask", "attention", [("q", AQ), ("k", AQ), ("v", AQ), ("m", (1, 1, 32, 32))], lambda q, k, v, m: mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=m),
         kinds={"q": "small", "k": "small", "v": "small", "m": "small"})
    # The same causal masks as a runtime INPUT, and a smooth random mask as a constant: is it the constant, or the large negative values, that the device ignores?
    for tag, kd in (("m1e4", "causal"), ("m65504", "causal_max"), ("minf", "causal_inf")):
        case(f"sdpa_in_causal_{tag}", "attention", [("q", AQ), ("k", AQ), ("v", AQ), ("m", (1, 1, 32, 32))],
             lambda q, k, v, m: mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=m), kinds={"q": "small", "k": "small", "v": "small", "m": kd})
    smooth = f16(np.random.RandomState(7).uniform(-1, 1, (1, 1, 32, 32)))
    case("sdpa_const_smooth", "attention", [("q", AQ), ("k", AQ), ("v", AQ)],
         lambda q, k, v: mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=smooth), kinds={"q": "small", "k": "small", "v": "small"})
    # Production shape (Stories110M: 12 heads, 256 tokens, head dim 64): the fused op with a causal mask input, the same attention decomposed into
    # matmul / mul / add / softmax / matmul (what the upstream pipeline does), and the unmasked fused op as the timing baseline.
    AQ2 = (1, 12, 256, 64); qkv2 = {"q": "small", "k": "small", "v": "small"}
    case("sdpa_s256", "attention", [("q", AQ2), ("k", AQ2), ("v", AQ2)], lambda q, k, v: mb.scaled_dot_product_attention(query=q, key=k, value=v), kinds=qkv2)
    case("sdpa_in_causal_s256", "attention", [("q", AQ2), ("k", AQ2), ("v", AQ2), ("m", (1, 1, 256, 256))],
         lambda q, k, v, m: mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=m), kinds=dict(qkv2, m="causal"))
    def sdpa_decomp(q, k, v, m):
        s = mb.mul(x=mb.matmul(x=q, y=k, transpose_y=True), y=f16(0.125))
        return mb.matmul(x=mb.softmax(x=mb.add(x=s, y=m), axis=-1), y=v)
    case("sdpa_decomp_causal_s256", "attention", [("q", AQ2), ("k", AQ2), ("v", AQ2), ("m", (1, 1, 256, 256))], sdpa_decomp, kinds=dict(qkv2, m="causal"))
    # The configuration of upstream's own debug test (training/test_ane_sdpa5.m): 12 heads, 8 tokens, head dim 64, causal mask as a runtime input.
    AQ8 = (1, 12, 8, 64)
    case("sdpa_in_causal_s8", "attention", [("q", AQ8), ("k", AQ8), ("v", AQ8), ("m", (1, 1, 8, 8))],
         lambda q, k, v, m: mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=m), kinds={"q": "small", "k": "small", "v": "small", "m": "causal_max"})
    # Causal masks as constants (the upstream README says the ANE hardware ignores attn_mask in SDPA): additive 0 / -1e4, -65504, -inf, and boolean.
    tri = np.tril(np.ones((32, 32), bool)).reshape(1, 1, 32, 32)
    for tag, mk in (("m1e4", f16(np.where(tri, 0.0, -1e4))), ("m65504", f16(np.where(tri, 0.0, -65504.0))), ("minf", f16(np.where(tri, 0.0, -np.inf))), ("bool", tri)):
        case(f"sdpa_causal_{tag}", "attention", [("q", AQ), ("k", AQ), ("v", AQ)],
             (lambda mk: lambda q, k, v: mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mk))(mk), kinds={"q": "small", "k": "small", "v": "small"})
    # ---- dtype conversions through the engine
    # fp16 -> integer -> fp16.  The device rounds to nearest with ties away from zero (shown by int8 and uint16, where ties-to-even differs); an
    # unsigned type gives 0 for negative input.  The cast_sat_* cases below push values past the range of the integer type.  Core ML on the CPU rounds
    # differently, so the integer types are pinned to a numpy model of the device.  int32 and fp32 as the intermediate type are refused.
    cast_lim = {"int8": (-128, 127), "uint8": (0, 255), "int16": (-32768, 32767), "uint16": (0, 65535)}
    away_ = lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5)
    f16_ = lambda v: np.asarray(v, np.float64).astype(np.float16).astype(np.float64)
    for dt in ("int8", "uint8", "int16", "uint16", "int32", "fp32"):
        kw = {"ref": (lambda lo, hi: lambda x: np.clip(away_(x * 8.0), lo, hi))(*cast_lim[dt])} if dt in cast_lim else {}
        case(f"cast_roundtrip_{dt}", "dtype", X, (lambda dt: lambda x: mb.cast(x=mb.cast(x=mb.mul(x=x, y=f16(8.0)), dtype=dt), dtype="fp16"))(dt), kinds={"x": "small"}, **kw)
    # beyond the range: x in [-3, 3] times k.  int8: up to +-300 (range -128..127); uint8: up to 300 (0..255); int16: up to +-60000 (-32768..32767).
    # uint16 cannot be pushed past 65535: fp16 tops out at 65504.
    for dt, k in (("int8", 100.0), ("uint8", 100.0), ("int16", 20000.0)):
        lo, hi = cast_lim[dt]
        case(f"cast_sat_{dt}", "dtype", X, (lambda dt, k: lambda x: mb.cast(x=mb.cast(x=mb.mul(x=x, y=f16(k)), dtype=dt), dtype="fp16"))(dt, k), kinds={"x": "any"},
             ref=(lambda k, lo, hi: lambda x: np.clip(away_(f16_(x * k)), lo, hi))(k, lo, hi), tol=1e-3)
    # ---- rank coverage: relu through ranks 1..5
    for r, shp in enumerate([(2048,), (64, 32), (64, 1, 32), (1, 64, 1, 32), (1, 16, 4, 8, 16)], 1):
        case(f"rank{r}_add", "rank", [("x", shp), ("y", shp)], lambda x, y: mb.add(x=x, y=y))

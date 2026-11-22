"""Compressed weights and quantization ops feeding conv 1x1 / linear on the ANE (case prefix wq_).

Weight forms: constexpr_affine_dequantize (int8/uint8, per-tensor / per-axis scale, zero_point 0 / nonzero),
constexpr_lut_to_dense (nbits 1,2,4,6,8; per-tensor and per-output-row palettes), constexpr_blockwise_shift_scale
(block 8/16/32/64 along input or output axis, int8 and int4 data, with/without offset), constexpr_sparse_to_dense (50% mask).
Activation ops: quantize -> dequantize round trips.
Speed cases: 512x512 1x1 conv at SP 64 (input (1,512,1,64)).  Functional: 64x64 conv at (1,64,1,32), 128x64 linear at (1,32,64).
"""


def register(env):
    case, mb, np, f16 = env.case, env.mb, env.np, env.f16
    from coremltools.converters.mil.mil.types import (np_uint1_dtype, np_uint2_dtype, np_uint4_dtype,
                                                      np_uint6_dtype, np_int4_dtype)
    UINT = {1: np_uint1_dtype, 2: np_uint2_dtype, 4: np_uint4_dtype, 6: np_uint6_dtype, 8: np.uint8}
    rs = np.random.RandomState(7)
    TOL = 5e-3
    FX = (1, 64, 1, 32)   # functional conv input
    SX = (1, 512, 1, 64)  # speed input
    LX = (1, 32, 64)      # linear input

    # Every generator works on a 4-D (oc, ic, 1, 1) weight; linear squeezes to 2-D inside the thunk.
    def wf(oc, ic, s=0.2):
        return rs.uniform(-s, s, (oc, ic, 1, 1)).astype(np.float32)

    def fit(a, lin):
        return a.reshape(a.shape[:2]) if lin else a

    def q_int(w, sb, zp, dtype):
        lo, hi = (-128, 127) if dtype == np.int8 else (0, 255)
        return np.clip(np.round(w / sb) + zp, lo, hi).astype(dtype)

    def affine(w, axis, dtype=np.int8, zp=0, per_channel=True, lin=False):
        if per_channel:
            red = tuple(i for i in range(4) if i != axis)
            sc = (np.abs(w).max(axis=red) / 100.0 + 1e-4).astype(np.float32)
            sb = sc.reshape([-1 if i == axis else 1 for i in range(4)])
            scale = sc.astype(np.float16)
            zpv = np.full(sc.shape, zp, dtype)
        else:
            s = float(np.abs(w).max() / 100.0 + 1e-4)
            sb, scale, zpv = s, np.float16(s), dtype(zp)
        q = q_int(w, sb, zp, dtype)
        if per_channel:
            zpv = zpv
        return lambda: mb.constexpr_affine_dequantize(quantized_data=fit(q, lin), zero_point=zpv if np.ndim(zpv) == 0 else fit(zpv.reshape(sc.shape + (1,) * 0), lin) if False else zpv,
                                                      scale=scale, axis=axis)

    def lut(w, nbits, per_channel=False, lin=False):
        """nbits palette; per_channel = one palette per output row. Indices unpacked uint<nbits>; lut rank = idx rank + 2."""
        oc, ic = w.shape[0], w.shape[1]
        npal = 2 ** nbits
        if per_channel:
            pal = np.stack([np.linspace(-0.25, 0.25, npal) * (1 + 0.1 * r) for r in range(oc)]).astype(np.float16)
            flat = w.reshape(oc, ic)
            idx = np.abs(flat[:, :, None] - pal[:, None, :].astype(np.float32)).argmin(axis=2)
            lutv = pal.reshape((oc,) + (1,) * (2 if not lin else 1) + (1,) * 0 + (npal, 1)) if not lin else pal.reshape(oc, npal, 1)
            if not lin:
                lutv = pal.reshape(oc, 1, 1, 1, npal, 1)
        else:
            pal = np.linspace(-0.25, 0.25, npal).astype(np.float16)
            idx = np.abs(w.reshape(-1, 1) - pal.astype(np.float32)).argmin(axis=1).reshape(oc, ic)
            lutv = pal.reshape((1,) * (2 if lin else 4) + (npal, 1))
        ind = idx.reshape(fit(np.zeros((oc, ic, 1, 1)), lin).shape).astype(UINT[nbits])
        return lambda: mb.constexpr_lut_to_dense(indices=ind, lut=lutv)

    def blockwise(w, bs, axis=1, offset=False, dtype=np.int8, lin=False):
        oc, ic = w.shape[0], w.shape[1]
        w2 = w.reshape(oc, ic)
        if axis == 1:
            nb = ic // bs
            blocks = w2.reshape(oc, nb, bs)
            sc = (np.abs(blocks).max(axis=2, keepdims=True) / 100.0 + 1e-4).astype(np.float32)   # (oc,nb,1)
            zp = np.full(sc.shape, 3, np.int8) if offset else np.zeros(sc.shape, np.int8)
            q2 = np.clip(np.round(blocks / sc) + zp, -128, 127).astype(np.int8).reshape(oc, ic)
            scale2 = sc.reshape(oc, nb)
            off2 = zp.reshape(oc, nb)
        else:
            nb = oc // bs
            blocks = w2.reshape(nb, bs, ic)
            sc = (np.abs(blocks).max(axis=1, keepdims=True) / 100.0 + 1e-4).astype(np.float32)   # (nb,1,ic)
            zp = np.full(sc.shape, 3, np.int8) if offset else np.zeros(sc.shape, np.int8)
            q2 = np.clip(np.round(blocks / sc) + zp, -128, 127).astype(np.int8).reshape(oc, ic)
            scale2 = sc.reshape(nb, ic)
            off2 = zp.reshape(nb, ic)
        if dtype == np_int4_dtype:
            q2 = np.clip(np.round(q2.astype(np.float32) / 16), -8, 7).astype(np_int4_dtype)
            scale2 = (scale2 * 16).astype(np.float16)
            off2 = np.zeros_like(off2, dtype=np_int4_dtype) if offset else None
        q = q2.reshape(oc, ic, 1, 1) if not lin else q2
        def ex(a):   # scale/offset rank matches data rank (4-D for conv, 2-D for linear)
            return a.astype(np.float16).reshape(a.shape + (1, 1)) if not lin else a.astype(np.float16)
        scale = ex(scale2)
        if offset:
            off = off2.reshape(off2.shape + (1, 1)) if not lin else off2
            return lambda: mb.constexpr_blockwise_shift_scale(data=q, scale=scale, offset=off)
        return lambda: mb.constexpr_blockwise_shift_scale(data=q, scale=scale)

    def sparse(w, lin=False):
        oc, ic = w.shape[0], w.shape[1]
        keep = rs.uniform(size=(oc, ic, 1, 1)) < 0.5
        vals = w.reshape(-1)[keep.reshape(-1)].astype(np.float16)
        mask = fit(keep.astype(np_uint1_dtype), lin)
        return lambda: mb.constexpr_sparse_to_dense(nonzero_data=vals, mask=mask)

    def conv_case(name, family, mk, inp=FX, tol=TOL, kind="small"):
        case(name, family, [("x", inp)], lambda x: mb.conv(x=x, weight=mk(), pad_type="valid"), kinds={"x": kind}, tol=tol)

    def lin_case(name, family, mk, tol=TOL):
        case(name, family, [("x", LX)], lambda x: mb.linear(x=x, weight=mk()), kinds={"x": "small"}, tol=tol)

    # ------------------------------------------------------------------ speed set: conv1x1 512->512 at SP 64
    w512 = wf(512, 512)
    case("wq_speed_fp16_plain", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=f16(w512), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_affine_i8_pc0", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=affine(w512, 0)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_affine_u8_pc0_zp128", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=affine(w512, 0, np.uint8, 128)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_lut4_tensor", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=lut(w512, 4)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_lut8_tensor", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=lut(w512, 8)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_blockwise_i8_b32", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=blockwise(w512, 32)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_blockwise_i8_b64", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=blockwise(w512, 64)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)
    case("wq_speed_sparse50", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=sparse(w512)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)

    # ------------------------------------------------------------------ affine
    w64 = wf(64, 64)
    conv_case("wq_affine_i8_pc0_zp0", "affine", affine(w64, 0))
    conv_case("wq_affine_i8_pc1_zp0", "affine", affine(w64, 1))
    conv_case("wq_affine_i8_tensor_zp0", "affine", affine(w64, 0, per_channel=False))
    conv_case("wq_affine_i8_pc0_zp5", "affine", affine(w64, 0, zp=5))
    conv_case("wq_affine_u8_pc0_zp128", "affine", affine(w64, 0, np.uint8, 128))
    conv_case("wq_affine_u8_tensor_zp100", "affine", affine(w64, 0, np.uint8, 100, per_channel=False))
    conv_case("wq_affine_u8_pc1_zp128", "affine", affine(w64, 1, np.uint8, 128))
    conv_case("wq_affine_i8_pc1_zp5", "affine", affine(w64, 1, zp=5))
    lin_case("wq_linear_affine_i8_pc0", "affine", affine(wf(128, 64), 0, lin=True))
    # variants of the failing axis-1 per-channel case (one change each)
    conv_case("wq_affine_i8_tensor_axis1", "affine", affine(w64, 1, per_channel=False))
    lin_case("wq_linear_affine_i8_pc1", "affine", affine(wf(128, 64), 1, lin=True))
    case("wq_speed_affine_i8_pc1", "speed", [("x", SX)],
         lambda x: mb.conv(x=x, weight=affine(w512, 1)(), pad_type="valid"), kinds={"x": "small"}, tol=TOL)

    # ------------------------------------------------------------------ lut nbits sweep
    for nb in (1, 2, 4, 6, 8):
        conv_case(f"wq_lut{nb}_tensor", "lut", lut(w64, nb))
    conv_case("wq_lut4_perchannel", "lut", lut(w64, 4, per_channel=True))
    lin_case("wq_linear_lut4_tensor", "lut", lut(wf(128, 64), 4, lin=True))

    # ------------------------------------------------------------------ blockwise
    conv_case("wq_blockwise_i8_b8", "blockwise", blockwise(w64, 8))
    conv_case("wq_blockwise_i8_b16", "blockwise", blockwise(w64, 16))
    conv_case("wq_blockwise_i8_b32", "blockwise", blockwise(w64, 32))
    conv_case("wq_blockwise_i8_b64", "blockwise", blockwise(w64, 64))
    conv_case("wq_blockwise_i8_b32_axis0", "blockwise", blockwise(w64, 32, axis=0))
    conv_case("wq_blockwise_i8_b32_offset3", "blockwise", blockwise(w64, 32, offset=True))
    conv_case("wq_blockwise_i4_b32", "blockwise", blockwise(w64, 32, dtype=np_int4_dtype))
    conv_case("wq_blockwise_i4_b64", "blockwise", blockwise(w64, 64, dtype=np_int4_dtype))
    lin_case("wq_linear_blockwise_i8_b32", "blockwise", blockwise(wf(128, 64), 32, lin=True))

    # ------------------------------------------------------------------ sparse
    conv_case("wq_sparse50", "sparse", sparse(w64))
    lin_case("wq_linear_sparse50", "sparse", sparse(wf(128, 64), lin=True))

    # ------------------------------------------------------------------ activation quantize / dequantize
    def qdq(out_dt, zp, scale, axis=None):
        def fn(x):
            qk = dict(input=x, scale=scale, zero_point=zp, output_dtype=("int8" if out_dt == np.int8 else "uint8"))
            if axis is not None: qk["axis"] = axis
            q = mb.quantize(**qk)
            dk = dict(input=q, scale=scale, zero_point=zp)
            if axis is not None: dk["axis"] = axis
            return mb.dequantize(**dk)
        return fn

    # The device model (README "Sharp corners", macOS 27.2.1), reproduced exactly (0 of 2048 values differ) for per-tensor and per-channel scales
    # and for zero points 0, 7, 100, 128:   b = f16(z * s);  inv = f16(1 / s);  t = f16(f16(x + b) * inv);  q = clip(round_half_away(t));  y = f16((q - z) * s)
    # b and inv round to nearest even (they are constants, folded by the compiler); the per-element f16() of the add, the multiply and the final
    # (q - z) * s round exact ties away from zero.  The zero point is added in the x domain (as z*s) before the reciprocal multiply.
    # Core ML on the CPU divides and rounds half to even, so these cases are checked against this numpy model: `ok` pins it.
    away = lambda v: np.sign(v) * np.floor(np.abs(v) + 0.5)
    f16_rne = lambda v: np.asarray(v, np.float64).astype(np.float16).astype(np.float64)
    def f16_away(v):
        v = np.asarray(v, np.float64); r = v.astype(np.float16); rr = r.astype(np.float64)
        nb = np.where(v > rr, np.nextafter(r, np.float16(np.inf)), np.nextafter(r, np.float16(-np.inf))).astype(np.float64)
        tie = (v != rr) & (np.abs(v - rr) == np.abs(nb - v))
        return np.where(tie & (np.abs(nb) > np.abs(rr)), nb, rr)
    def qdq_ref(dt, zp, scale, axis=None):
        lo, hi = (-128, 127) if dt == np.int8 else (0, 255)
        def ref(x):
            sc = np.asarray(scale, np.float64); z = np.asarray(zp, np.float64)
            if axis is not None: sc, z = sc.reshape(1, -1, 1, 1), z.reshape(1, -1, 1, 1)
            b, inv = f16_rne(z * sc), f16_rne(1.0 / sc)                       # folded into constants by the compiler: ordinary round-to-nearest-even
            t = f16_away(f16_away(x + b) * inv)                               # the per-element add and multiply tie away from zero
            return f16_away((np.clip(away(t), lo, hi) - z) * sc)
        return ref
    case("wq_act_qdq_i8_tensor", "activation", [("x", FX)], qdq(np.int8, np.int8(0), np.float16(1 / 64)), kinds={"x": "small"}, tol=TOL, ref=qdq_ref(np.int8, 0, np.float16(1 / 64)))
    case("wq_act_qdq_u8_zp128", "activation", [("x", FX)], qdq(np.uint8, np.uint8(128), np.float16(1 / 64)), kinds={"x": "small"}, tol=TOL, ref=qdq_ref(np.uint8, 128, np.float16(1 / 64)))
    case("wq_act_qdq_i8_zp7", "activation", [("x", FX)], qdq(np.int8, np.int8(7), np.float16(1 / 64)), kinds={"x": "small"}, tol=TOL, ref=qdq_ref(np.int8, 7, np.float16(1 / 64)))
    sc_c = np.linspace(0.01, 0.03, 64).astype(np.float16)
    case("wq_act_qdq_i8_axis1", "activation", [("x", FX)], qdq(np.int8, np.zeros(64, np.int8), sc_c, axis=1), kinds={"x": "small"}, tol=TOL, ref=qdq_ref(np.int8, np.zeros(64), sc_c, axis=1))
    case("wq_act_qdq_u8_axis1_zp", "activation", [("x", FX)], qdq(np.uint8, np.full(64, 100, np.uint8), sc_c, axis=1), kinds={"x": "small"}, tol=TOL, ref=qdq_ref(np.uint8, np.full(64, 100), sc_c, axis=1))

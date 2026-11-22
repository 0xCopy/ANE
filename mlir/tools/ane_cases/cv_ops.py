"""Convolution / pooling / resize variants and size-limit sweeps for the Apple M3 Pro ANE (prefix cv_).

Variant cases: cv_<name>.  Sweep cases: cv_sw_<sweep>_<value>; bisection refinements: cv_bs_<sweep>_<value>.
Run subsets with --only '^cv_sw_' or --only '^cv_(?!sw_)' to keep one invocation <= ~60 cases.
"""

# coarse sweep values (powers of two; refined later by bisection)
IC_VALS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
OC_VALS = IC_VALS
SP_VALS = [8, 16, 32, 64, 128, 256, 512]
RELU_W = [1, 16, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
RELU_C = [1, 16, 256, 1024, 4096, 16384, 65536, 131072]
RELU_N = [1, 2, 4, 8, 16, 32, 64]
RELU_H = [1, 16, 64, 256, 1024, 2048, 4096]


def register(env):
    case, mb, np, f16, rs, SHAPE, X, XY = env.case, env.mb, env.np, env.f16, env.rs, env.SHAPE, env.X, env.XY

    def W(*shape):
        return f16(rs.uniform(-0.2, 0.2, shape))

    CS = (1, 64, 8, 32)
    BIG = (1, 64, 32, 32)

    # ================================================================ (A) convolution variants
    for k in [(1, 3), (3, 1), (5, 5), (7, 7), (9, 9), (11, 11)]:
        case(f"cv_k{k[0]}x{k[1]}", "conv", [("x", BIG)],
             (lambda k: lambda x: mb.conv(x=x, weight=W(64, 64, *k), pad_type="same"))(k))
    case("cv_k16x16_valid", "conv", [("x", (1, 64, 32, 32))], lambda x: mb.conv(x=x, weight=W(64, 64, 16, 16), pad_type="valid"))
    # one-thing changes from the failing k16x16 case
    case("cv_k16x16_valid_ic16", "conv", [("x", (1, 16, 32, 32))], lambda x: mb.conv(x=x, weight=W(64, 16, 16, 16), pad_type="valid"))
    case("cv_k16x16_valid_oc16", "conv", [("x", (1, 64, 32, 32))], lambda x: mb.conv(x=x, weight=W(16, 64, 16, 16), pad_type="valid"))
    case("cv_k8x8_valid", "conv", [("x", (1, 64, 32, 32))], lambda x: mb.conv(x=x, weight=W(64, 64, 8, 8), pad_type="valid"))
    for s in (3, 4):
        case(f"cv_stride{s}", "conv", [("x", BIG)],
             lambda x, s=s: mb.conv(x=x, weight=W(64, 64, 3, 3), strides=[s, s], pad_type="custom", pad=[1, 1, 1, 1]))
    for d in (4, 8):
        case(f"cv_dil{d}", "conv", [("x", BIG)], lambda x, d=d: mb.conv(x=x, weight=W(64, 64, 3, 3), dilations=[d, d], pad_type="same"))
    case("cv_groups4", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=W(64, 16, 3, 3), groups=4, pad_type="same"))
    case("cv_dw5x5", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=W(64, 1, 5, 5), groups=64, pad_type="same"))
    case("cv_dw_mult2", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=W(128, 1, 3, 3), groups=64, pad_type="same"))
    case("cv_pad_asym", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=W(64, 64, 3, 3), pad_type="custom", pad=[0, 2, 1, 3]))
    case("cv_same_lower", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=W(64, 64, 3, 3), pad_type="same_lower"))
    case("cv_k3_bias", "conv", [("x", CS)], lambda x: mb.conv(x=x, weight=W(64, 64, 3, 3), bias=f16(rs.uniform(-.3, .3, 64)), pad_type="same"))
    case("cv_conv1d_k5_s2", "conv", [("x", (1, 64, 32))],
         lambda x: mb.conv(x=x, weight=W(64, 64, 5), strides=[2], pad_type="custom", pad=[2, 2]))
    case("cv_conv1d_valid_k1", "conv", [("x", (1, 64, 32))], lambda x: mb.conv(x=x, weight=W(64, 64, 1), pad_type="valid"))
    case("cv_conv3d_k1", "conv", [("x", (1, 16, 4, 8, 16))], lambda x: mb.conv(x=x, weight=W(32, 16, 1, 1, 1), pad_type="valid"))
    case("cv_conv3d_s2", "conv", [("x", (1, 16, 4, 8, 16))],
         lambda x: mb.conv(x=x, weight=W(16, 16, 3, 3, 3), strides=[2, 2, 2], pad_type="same"))
    for nm, k, s in [("k2_s1", 2, 1), ("k3_s2", 3, 2), ("k4_s2", 4, 2), ("k4_s4", 4, 4)]:
        case(f"cv_ct_{nm}", "conv", [("x", CS)],
             (lambda k, s: lambda x: mb.conv_transpose(x=x, weight=W(64, 32, k, k), strides=[s, s], pad_type="valid"))(k, s))
    case("cv_ct_k3_s1_bias", "conv", [("x", CS)],
         lambda x: mb.conv_transpose(x=x, weight=W(64, 64, 3, 3), bias=f16(rs.uniform(-.3, .3, 64)), strides=[1, 1], pad_type="same"))

    # ================================================================ (A) pooling variants
    P = CS
    case("cv_avg_k3_s1_same", "pool", [("x", P)], lambda x: mb.avg_pool(x=x, kernel_sizes=[3, 3], strides=[1, 1], pad_type="same"))
    case("cv_avg_k5_s2_custom", "pool", [("x", P)],
         lambda x: mb.avg_pool(x=x, kernel_sizes=[5, 5], strides=[2, 2], pad_type="custom", pad=[2, 2, 2, 2]))
    case("cv_avg_k7_s1_noexcl", "pool", [("x", P)],
         lambda x: mb.avg_pool(x=x, kernel_sizes=[7, 7], strides=[1, 1], pad_type="same", exclude_padding_from_average=False))
    case("cv_avg_k7_s1_excl", "pool", [("x", P)],
         lambda x: mb.avg_pool(x=x, kernel_sizes=[7, 7], strides=[1, 1], pad_type="same", exclude_padding_from_average=True))
    case("cv_avg_k2_s2_ceil", "pool", [("x", (1, 64, 9, 31))],
         lambda x: mb.avg_pool(x=x, kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid", ceil_mode=True))
    case("cv_max_k3_s2_same", "pool", [("x", P)], lambda x: mb.max_pool(x=x, kernel_sizes=[3, 3], strides=[2, 2], pad_type="same"))
    case("cv_max_k5_s1_valid", "pool", [("x", P)], lambda x: mb.max_pool(x=x, kernel_sizes=[5, 5], strides=[1, 1], pad_type="valid"))
    case("cv_max_k7_s2_custom", "pool", [("x", P)],
         lambda x: mb.max_pool(x=x, kernel_sizes=[7, 7], strides=[2, 2], pad_type="custom", pad=[3, 3, 3, 3]))
    case("cv_max_k2_s2_ceil", "pool", [("x", (1, 64, 9, 31))],
         lambda x: mb.max_pool(x=x, kernel_sizes=[2, 2], strides=[2, 2], pad_type="valid", ceil_mode=True))
    case("cv_l2_k3_s1_same", "pool", [("x", P)], lambda x: mb.l2_pool(x=x, kernel_sizes=[3, 3], strides=[1, 1], pad_type="same"))
    case("cv_l2_k5_s2_valid", "pool", [("x", P)], lambda x: mb.l2_pool(x=x, kernel_sizes=[5, 5], strides=[2, 2], pad_type="valid"))

    # ================================================================ (A) resize variants
    RS = (1, 16, 8, 16)
    for mode in ("DEFAULT", "STRICT_ALIGN_CORNERS", "ALIGN_CORNERS", "OFFSET_CORNERS"):
        case(f"cv_rbil_{mode.lower()}", "resize", [("x", RS)],
             (lambda m: lambda x: mb.resize_bilinear(x=x, target_size_height=16, target_size_width=32, sampling_mode=m))(mode))
    case("cv_rbil_down", "resize", [("x", RS)],
         lambda x: mb.resize_bilinear(x=x, target_size_height=4, target_size_width=8, sampling_mode="DEFAULT"))
    case("cv_rnn_up2x", "resize", [("x", RS)],
         lambda x: mb.resize_nearest_neighbor(x=x, target_size_height=16, target_size_width=32))
    case("cv_rnn_default_down", "resize", [("x", RS)],
         lambda x: mb.resize_nearest_neighbor(x=x, target_size_height=4, target_size_width=8))
    case("cv_upbil_align", "resize", [("x", RS)],
         lambda x: mb.upsample_bilinear(x=x, scale_factor_height=2, scale_factor_width=2, align_corners=True))
    case("cv_upbil_noalign", "resize", [("x", RS)],
         lambda x: mb.upsample_bilinear(x=x, scale_factor_height=3, scale_factor_width=2, align_corners=False))
    case("cv_upnn_s3", "resize", [("x", RS)],
         lambda x: mb.upsample_nearest_neighbor(x=x, scale_factor_height=3, scale_factor_width=3))
    case("cv_crop_resize", "resize", [("x", (1, 16, 16, 16)), ("roi", (2, 4)), ("bidx", (2,), "int32")],
         lambda x, roi, bidx: mb.crop_resize(x=x, boxes=roi, box_indices=bidx, target_height=8, target_width=8, normalized_coordinates=True, spatial_scale=1.0, pad_value=f16(0.0)),
         kinds={"roi": "unit", "bidx": "idx:1"}, tol=5e-2)
    # resample, bilinear, padding "constant" (value 0), pixel coordinates.  Core ML on the CPU blends the in-image corners of a sample point that is
    # a little outside the image with the padding value; the device returns the padding value for every sample point outside [0, W-1] x [0, H-1]
    # (measured, see sh_resample_*).  The case is checked against that device model, so `ok` pins it.
    def resample_unnorm_dev(x, c):
        _, C, H, Wd = x.shape; out = np.zeros((1, C, c.shape[1], c.shape[2]))
        for i in range(c.shape[1]):
            for j in range(c.shape[2]):
                fx, fy = c[0, i, j]
                if fx < 0 or fy < 0 or fx > Wd - 1 or fy > H - 1: continue
                x0, y0 = int(np.floor(fx)), int(np.floor(fy)); ax, ay = fx - x0, fy - y0; x1, y1 = min(x0 + 1, Wd - 1), min(y0 + 1, H - 1)
                out[0, :, i, j] = (1 - ay) * (1 - ax) * x[0, :, y0, x0] + (1 - ay) * ax * x[0, :, y0, x1] + ay * (1 - ax) * x[0, :, y1, x0] + ay * ax * x[0, :, y1, x1]
        return out
    case("cv_resample", "resize", [("x", RS), ("c", (1, 8, 16, 2))],
         lambda x, c: mb.resample(x=x, coordinates=c, sampling_mode="bilinear", padding_mode="constant", padding_value=f16(0.0),
                                  coordinates_mode="unnormalized", align_corners=True),
         kinds={"c": "unit"}, tol=5e-2, ref=resample_unnorm_dev)

    # ================================================================ (B) size-limit sweeps (coarse, powers of two)
    for ic in IC_VALS:
        case(f"cv_sw_conv1x1_ic_{ic}", "sweep", [("x", (1, ic, 8, 8))],
             (lambda ic: lambda x: mb.conv(x=x, weight=W(64, ic, 1, 1), pad_type="valid"))(ic))
    for oc in OC_VALS:
        case(f"cv_sw_conv1x1_oc_{oc}", "sweep", [("x", (1, 64, 8, 8))],
             (lambda oc: lambda x: mb.conv(x=x, weight=W(oc, 64, 1, 1), pad_type="valid"))(oc))
    for s in SP_VALS:
        case(f"cv_sw_conv3x3_sp_{s}", "sweep", [("x", (1, 16, s, s))],
             (lambda: lambda x: mb.conv(x=x, weight=W(16, 16, 3, 3), pad_type="same"))())
    for w in RELU_W:
        case(f"cv_sw_relu_w_{w}", "sweep", [("x", (1, 1, 1, w))], lambda x: mb.relu(x=x))
    for c in RELU_C:
        case(f"cv_sw_relu_c_{c}", "sweep", [("x", (1, c, 1, 1))], lambda x: mb.relu(x=x))
    for n in RELU_N:
        case(f"cv_sw_relu_n_{n}", "sweep", [("x", (n, 16, 1, 16))], lambda x: mb.relu(x=x))
    for h in RELU_H:
        case(f"cv_sw_relu_h_{h}", "sweep", [("x", (1, 16, h, 16))], lambda x: mb.relu(x=x))

    # ---- one-thing variants for the fail/wrong variant cases
    case("cv_k16x16_same", "conv", [("x", (1, 64, 32, 32))], lambda x: mb.conv(x=x, weight=W(64, 64, 16, 16), pad_type="same"))
    case("cv_resample_nearest", "resize", [("x", RS), ("c", (1, 8, 16, 2))],
         lambda x, c: mb.resample(x=x, coordinates=c, sampling_mode="nearest", padding_mode="constant", padding_value=f16(0.0),
                                  coordinates_mode="unnormalized", align_corners=True), kinds={"c": "unit"}, tol=5e-2)
    case("cv_resample_border", "resize", [("x", RS), ("c", (1, 8, 16, 2))],
         lambda x, c: mb.resample(x=x, coordinates=c, sampling_mode="bilinear", padding_mode="border", padding_value=f16(0.0),
                                  coordinates_mode="unnormalized", align_corners=True), kinds={"c": "unit"}, tol=5e-2)
    case("cv_crop_resize_unnorm", "resize", [("x", (1, 16, 16, 16)), ("roi", (2, 4)), ("bidx", (2,), "int32")],
         lambda x, roi, bidx: mb.crop_resize(x=x, boxes=roi, box_indices=bidx, target_height=8, target_width=8, normalized_coordinates=False, spatial_scale=1.0, pad_value=f16(0.0)),
         kinds={"roi": "pos", "bidx": "idx:1"}, tol=5e-2)
    case("cv_crop_resize_1box", "resize", [("x", (1, 16, 16, 16)), ("roi", (1, 4)), ("bidx", (1,), "int32")],
         lambda x, roi, bidx: mb.crop_resize(x=x, boxes=roi, box_indices=bidx, target_height=8, target_width=8, normalized_coordinates=True, spatial_scale=1.0, pad_value=f16(0.0)),
         kinds={"roi": "unit", "bidx": "idx:1"}, tol=5e-2)

    # ---- bisection refinements (cv_bs_*): values between last pass and first fail, plus IC-1/IC+1 around 16384
    for ic in (9216, 10240, 12288, 14336, 16383, 16385):
        case(f"cv_bs_conv1x1_ic_{ic}", "sweep", [("x", (1, ic, 8, 8))],
             (lambda ic: lambda x: mb.conv(x=x, weight=W(64, ic, 1, 1), pad_type="valid"))(ic))
    for w in (24576, 32768, 49152, 57344, 65535):
        case(f"cv_bs_relu_w_{w}", "sweep", [("x", (1, 1, 1, w))], lambda x: mb.relu(x=x))
    for c in (81920, 98304, 114688, 122880, 131071):
        case(f"cv_bs_relu_c_{c}", "sweep", [("x", (1, c, 1, 1))], lambda x: mb.relu(x=x))
    for v in (65537,):
        case(f"cv_bs_relu_w_{v}", "sweep", [("x", (1, 1, 1, v))], lambda x: mb.relu(x=x))

    # ---- round 2 bisection: relu_w between 16384 (pass) and 24576 (fail); relu_c between 65536 (pass) and 81920 (fail); conv1x1 gaps
    for w in (16385, 17408, 18432, 20480, 22528, 24575):
        case(f"cv_bs2_relu_w_{w}", "sweep", [("x", (1, 1, 1, w))], lambda x: mb.relu(x=x))
    for c in (65537, 69632, 73728, 77824, 81919):
        case(f"cv_bs2_relu_c_{c}", "sweep", [("x", (1, c, 1, 1))], lambda x: mb.relu(x=x))
    for ic in (11264, 13312, 15360):
        case(f"cv_bs2_conv1x1_ic_{ic}", "sweep", [("x", (1, ic, 8, 8))],
             (lambda ic: lambda x: mb.conv(x=x, weight=W(64, ic, 1, 1), pad_type="valid"))(ic))

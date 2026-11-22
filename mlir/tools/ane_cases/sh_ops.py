"""sh_: do the size limits depend on how the same elements are laid out?  Same op, same element counts, different (H, W) split; and the silent-wrong
16x16 convolution mapped over IC and OC.  Checked against an fp32 reference (Core ML CPU in fp32), so accumulation error is not mistaken for a device error."""


def register(env):
    case, mb, np, f16, rs = env.case, env.mb, env.np, env.f16, env.rs
    W = lambda oc, ic, kh=1, kw=1: f16(rs.uniform(-.2, .2, (oc, ic, kh, kw)) * (64.0 / max(ic, 64)) ** 0.5)

    # ---- conv 1x1, OC=64, 64 spatial positions, four layouts of those 64 positions
    for ic in (32512, 32513, 65536):
        for h, w in ((1, 64), (8, 8), (64, 1), (2, 32)):
            case(f"sh_conv1x1_ic{ic}_h{h}w{w}", "limit", [("x", (1, ic, h, w))], lambda x, ic=ic: mb.conv(x=x, weight=W(64, ic), pad_type="valid"), tol=2e-3)
    # ---- the 'big tensor' confound: same bytes, different arrangement.  relu is elementwise, so the numbers are exact.
    for c, w, label in ((1024, 4096, "8mb_c1024_w4096"), (4096, 8192, "64mb_c4096_w8192"), (2048, 4096, "16mb_c2048_w4096"), (16384, 16384, "512mb_c16384_w16384"),
                        (131072, 32, "8mb_c131072_w32"), (65536, 64, "8mb_c65536_w64"), (8192, 8192, "128mb_c8192_w8192")):
        case(f"sh_relu_{label}", "limit", [("x", (1, c, 1, w))], lambda x: mb.relu(x=x), tol=0)
    for h, w, label in ((256, 256, "8mb_c64_h256_w256"), (1024, 1024, "128mb_c64_h1024_w1024"), (4096, 16, "8mb_c64_h4096_w16")):
        case(f"sh_relu_{label}", "limit", [("x", (1, 64, h, w))], lambda x: mb.relu(x=x), tol=0)
    # ---- the 16x16 kernel: where does a result stop being a convolution?
    for oc in (16, 64):
        for ic in (1, 2, 4, 8, 16, 32, 48, 64):
            case(f"sh_k16_oc{oc}_ic{ic}", "conv", [("x", (1, ic, 32, 32))], lambda x, oc=oc, ic=ic: mb.conv(x=x, weight=W(oc, ic, 16, 16), pad_type="valid"), tol=5e-3)
    for k in (9, 10, 12, 13, 14, 15):
        case(f"sh_k{k}_oc64_ic16", "conv", [("x", (1, 16, 32, 32))], lambda x, k=k: mb.conv(x=x, weight=W(64, 16, k, k), pad_type="valid"), tol=5e-3)

    # ---- where exactly does a 16x16 kernel turn from correct to garbage to refused?  output channels at IC=4 and IC=1; kernels beyond 16
    for oc in (24, 32, 40, 48, 56):
        case(f"sh_k16_oc{oc}_ic4", "conv", [("x", (1, 4, 32, 32))], lambda x, oc=oc: mb.conv(x=x, weight=W(oc, 4, 16, 16), pad_type="valid"), tol=5e-3)
    for k in (17, 20, 24, 32):
        case(f"sh_k{k}_oc16_ic1", "conv", [("x", (1, 1, 64, 64))], lambda x, k=k: mb.conv(x=x, weight=W(16, 1, k, k), pad_type="valid"), tol=5e-3)
    for kh, kw in ((16, 4), (4, 16), (16, 1), (1, 16), (8, 16), (16, 8)):
        case(f"sh_k{kh}x{kw}_oc64_ic4", "conv", [("x", (1, 4, 32, 32))], lambda x, kh=kh, kw=kw: mb.conv(x=x, weight=W(64, 4, kh, kw), pad_type="valid"), tol=5e-3)

    # ---- random_uniform.  Core ML on the CPU gives U[low, high).  The device compiles it and returns a FIXED array: the same values at every evaluation
    #      and for every seed, all inside [low, low + (high - low) / 16) (measured on macOS 27.2.1: see README "Sharp corners").  These cases compare with
    #      the CPU's distribution, so their status is `wrong` and the baseline pins that; randomness has to be fed to the program as an input.
    for lo, hi in ((0.0, 1.0), (0.0, 16.0), (-1.0, 1.0), (2.0, 3.0)):
        for seed in (1, 7):
            case(f"sh_randu_{lo:g}_{hi:g}_s{seed}".replace("-", "m").replace(".", "p"), "random", [("x", env.SHAPE)],
                 lambda x, lo=lo, hi=hi, seed=seed: mb.add(x=mb.mul(x=x, y=f16(0.0)), y=mb.random_uniform(shape=(1, 64, 1, 32), low=f16(lo), high=f16(hi), seed=seed)),
                 tol=None, check=lambda outs, ins, lo=lo, hi=hi: None if (outs[0].max() - outs[0].min()) >= 0.5 * (hi - lo) else
                 f"output spans [{outs[0].min():.4g}, {outs[0].max():.4g}] = {(outs[0].max() - outs[0].min()) / (hi - lo):.3f} of the requested range [{lo:g}, {hi:g})")

    # ---- resample with constant padding: is it only the border that differs?  coordinates drawn inside the image (normalized -1..1, 'unit' kind = +-0.9)
    RS = (1, 8, 16, 16)
    for pm in ("constant", "border", "reflection", "symmetric"):
        for inside in (True, False):
            if pm == "constant" and not inside: continue          # the device differs from Core ML here; pinned by sh_resample_constant_any_dev below
            case(f"sh_resample_{pm}_{'inside' if inside else 'any'}", "resize", [("x", RS), ("c", (1, 8, 16, 2))],
                 lambda x, c, pm=pm: mb.resample(x=x, coordinates=c, sampling_mode="bilinear", padding_mode=pm, padding_value=f16(0.0), coordinates_mode="normalized_minus_one_to_one", align_corners=True),
                 kinds={"c": "unit" if inside else "any"}, tol=3e-2)

    # ---- kernel extent sweep: width (kw) and height (kh) 12..34, OC 16 and 32, IC 1.  Each case: is it refused, garbage, or a convolution?
    for kw in range(12, 35):
        for oc in (16, 32):
            case(f"sh_kw{kw}_oc{oc}", "conv", [("x", (1, 1, 8, 64))], lambda x, kw=kw, oc=oc: mb.conv(x=x, weight=W(oc, 1, 1, kw), pad_type="valid"), tol=5e-3)
    for kh in range(12, 35, 2):
        case(f"sh_kh{kh}_oc16", "conv", [("x", (1, 1, 64, 8))], lambda x, kh=kh: mb.conv(x=x, weight=W(16, 1, kh, 1), pad_type="valid"), tol=5e-3)

    # ---- resample with constant padding, pinned to the device behaviour: a sample point outside the image returns the padding value
    #      (no per-corner blending with the border pixels, which is what Core ML on the CPU does)
    def resample_dev(x, c):
        _, C, H, Wd = x.shape; out = np.zeros((1, C, c.shape[1], c.shape[2]))
        for i in range(c.shape[1]):
            for j in range(c.shape[2]):
                cx, cy = c[0, i, j]
                if abs(cx) > 1 or abs(cy) > 1: continue
                fx = (cx + 1) * 0.5 * (Wd - 1); fy = (cy + 1) * 0.5 * (H - 1); x0, y0 = int(np.floor(fx)), int(np.floor(fy)); ax, ay = fx - x0, fy - y0
                x1, y1 = min(x0 + 1, Wd - 1), min(y0 + 1, H - 1)
                out[0, :, i, j] = (1 - ay) * (1 - ax) * x[0, :, y0, x0] + (1 - ay) * ax * x[0, :, y0, x1] + ay * (1 - ax) * x[0, :, y1, x0] + ay * ax * x[0, :, y1, x1]
        return out
    for inside in (True, False):
        case(f"sh_resample_constant_{'inside' if inside else 'any'}_dev", "resize", [("x", RS), ("c", (1, 8, 16, 2))],
             lambda x, c: mb.resample(x=x, coordinates=c, sampling_mode="bilinear", padding_mode="constant", padding_value=f16(0.0), coordinates_mode="normalized_minus_one_to_one", align_corners=True),
             kinds={"c": "unit" if inside else "any"}, tol=3e-2, ref=resample_dev)

    # ---- program length: N x (mul, add, relu) chained.  Long programs crash ANECompilerService (stack overflow in the compiler); these probe the boundary.
    #      They are in a separate module group (chain) so the default sweep leaves them out: run with --only '^sh_chain_640$' one at a time.
    def chain(N):
        def fn(x):
            for _ in range(N): x = mb.relu(x=mb.add(x=mb.mul(x=x, y=f16(0.99)), y=f16(0.01)))
            return x
        return fn
    for N in (576, 608, 640, 672, 704, 736, 768, 832, 896, 1024):
        case(f"sh_chain_{N}", "limit_chain", [("x", env.SHAPE)], chain(N))

    # ---- the same question for a wide, shallow program: N parallel branches relu(x * c_i + d_i), summed by a balanced add tree (depth 3 + log2 N)
    def wide(N):
        def fn(x):
            r = np.random.RandomState(5); outs = []
            for i in range(N): outs.append(mb.relu(x=mb.add(x=mb.mul(x=x, y=f16(r.uniform(0.5, 1.5))), y=f16(r.uniform(-0.1, 0.1)))))
            while len(outs) > 1:
                outs = [mb.add(x=outs[k], y=outs[k + 1]) if k + 1 < len(outs) else outs[k] for k in range(0, len(outs), 2)]
            return mb.mul(x=outs[0], y=f16(1.0 / N))
        return fn
    for N in (128, 512, 1024):
        case(f"sh_wide_{N}", "limit_chain", [("x", env.SHAPE)], wide(N), tol=2e-2)

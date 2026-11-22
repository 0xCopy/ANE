#!/usr/bin/env python3
"""Did the device apply or ignore the attn_mask of scaled_dot_product_attention?  Compares each case's device output with numpy attention computed
with the mask applied (as an additive bias) and with the mask ignored.  The case discriminates only if the two numpy results differ.

usage: ane_sdpa_mask_check.py WORKDIR   (the --work directory of a tools/ane_ops.py run that included the sdpa_* cases)

A mask that is a runtime input has a file m.bin in its case directory; a mask that is a constant in the MIL does not, so its pattern is rebuilt here
(lower-triangular 0 / large negative for sdpa_causal_*, the seed-7 uniform(-1, 1) matrix for sdpa_const_smooth).
"""
import os, sys
import numpy as np

W = sys.argv[1]
TRI = lambda L: np.tril(np.ones((L, L), bool))
NEG = {"m1e4": -1e4, "m65504": -65504.0, "minf": -np.inf}


def case(name, H, L):
    D = f"{W}/{name}"
    if not os.path.exists(f"{D}/out_0.bin"):
        return
    rd = lambda f, shp: np.fromfile(f"{D}/{f}.bin", "<f2").reshape(shp).astype(np.float64)
    q, k, v, dev = (rd(n, (1, H, L, 64)) for n in ("q", "k", "v", "out_0"))
    if os.path.exists(f"{D}/m.bin"):
        m, kind = rd("m", (1, 1, L, L)), "runtime input"
    elif name == "sdpa_const_smooth":
        m, kind = np.random.RandomState(7).uniform(-1, 1, (1, 1, L, L)).astype(np.float16).astype(np.float64), "constant"
    elif name.startswith("sdpa_causal_"):
        m, kind = np.where(TRI(L), 0.0, NEG.get(name.split("sdpa_causal_")[1], -np.inf)).reshape(1, 1, L, L), "constant"
    else:
        return

    def sdpa(bias):
        s = q @ k.transpose(0, 1, 3, 2) / 8.0
        if bias is not None:
            s = s + bias
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        return (p / p.sum(-1, keepdims=True)) @ v
    masked, plain = sdpa(m), sdpa(None)
    a, b, gap = np.abs(dev - masked).max(), np.abs(dev - plain).max(), np.abs(masked - plain).max()
    verdict = "mask APPLIED" if a < 0.01 and b > 10 * a else "mask IGNORED" if b < 0.01 and a > 10 * b else "unclear"
    print(f"{name:26s} {kind:14s} device vs masked {a:.5f} | vs unmasked {b:.5f} | numpy masked-vs-unmasked gap {gap:.4f} -> {verdict}")


for n, H, L in (("sdpa_mask", 8, 32), ("sdpa_in_causal_m1e4", 8, 32), ("sdpa_in_causal_m65504", 8, 32), ("sdpa_in_causal_minf", 8, 32), ("sdpa_in_causal_s8", 12, 8),
                ("sdpa_in_causal_s256", 12, 256), ("sdpa_decomp_causal_s256", 12, 256), ("sdpa_const_smooth", 8, 32), ("sdpa_causal_m1e4", 8, 32),
                ("sdpa_causal_m65504", 8, 32), ("sdpa_causal_minf", 8, 32)):
    case(n, H, L)

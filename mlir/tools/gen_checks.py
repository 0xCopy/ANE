#!/usr/bin/env python3
"""Writes mlir/test/check_*.mlir: drivers that run a literal ANE kernel and its normal form on the same data."""
import os, sys

OUT = sys.argv[1]
M = "tensor<?x?xf16>"
RUN = """// RUN: cat %S/harness.mlir %S/../normal/ane_normal.mlir %S/small/{lit}.mlir %s | mlir-opt \\
// RUN:   --transform-preload-library=transform-library-paths=%S/decompose_softmax.mlir --transform-interpreter \\
// RUN:   --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops \\
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \\
// RUN:   --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \\
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \\
// RUN: | mlir-runner -e main -entry-point-result=void \\
// RUN:   -shared-libs=%llvmlib/libmlir_runner_utils.dylib,%llvmlib/libmlir_c_runner_utils.dylib | FileCheck %s
"""


class T:
    def __init__(self, lit, title, note):
        self.lit, self.l, self.checks, self.n, self.seed = lit, [], [], 0, 0.1
        self.head = f"// {title}\n//\n" + "".join(f"// {x}\n" for x in note.strip().split("\n")) + "//\n"
        self.consts = set()

    def e(self, s):
        self.l.append("  " + s)

    def idx(self, v):
        if v not in self.consts:
            self.consts.add(v); self.e(f"%i{v} = arith.constant {v} : index")
        return f"%i{v}"

    def f16(self, name, v):
        self.e(f"%{name} = arith.constant {v} : f16")

    def gen(self, name, r, c, amp=0.5):
        self.seed += 0.9
        self.e(f"%{name}.s = arith.constant {self.seed:.1f} : f32")
        self.e(f"%{name}.a = arith.constant {amp} : f32")
        self.e(f"%{name} = call @gen({self.idx(r)}, {self.idx(c)}, %{name}.s, %{name}.a) : (index, index, f32, f32) -> {M}")

    def gen1(self, name, n):
        self.gen(name + ".m", n, 1, 1.0)
        self.e(f"%{name} = tensor.collapse_shape %{name}.m [[0, 1]] : {M} into tensor<?xf16>")

    def call(self, res, fn, args, tys, rty):
        self.e(f"{res} = call @{fn}({', '.join(args)}) : ({', '.join(tys)}) -> {rty}")

    def cat(self, name, parts, axis="v"):
        cur = parts[0]
        for k, p in enumerate(parts[1:]):
            nxt = f"{name}.{k}" if k < len(parts) - 2 else name
            self.call(f"%{nxt}", axis + "cat", [f"%{cur}", f"%{p}"], [M, M], M); cur = nxt

    def act4(self, name, src, c, s):
        """[C,S] -> 1xCx1xS"""
        self.e(f"%{name}.c = tensor.cast %{src} : {M} to tensor<{c}x{s}xf16>")
        self.e(f"%{name} = tensor.expand_shape %{name}.c [[0, 1], [2, 3]] output_shape [1, {c}, 1, {s}] : tensor<{c}x{s}xf16> into tensor<1x{c}x1x{s}xf16>")
        return f"tensor<1x{c}x1x{s}xf16>"

    def w4(self, name, src, o, i):
        """[O,I] -> OxIx1x1"""
        self.e(f"%{name}.c = tensor.cast %{src} : {M} to tensor<{o}x{i}xf16>")
        self.e(f"%{name} = tensor.expand_shape %{name}.c [[0], [1, 2, 3]] output_shape [{o}, {i}, 1, 1] : tensor<{o}x{i}xf16> into tensor<{o}x{i}x1x1xf16>")
        return f"tensor<{o}x{i}x1x1xf16>"

    def lead(self, name, src, a, b, lead):
        """[A,B] -> 1x..x1xAxB (lead unit dims)"""
        st = "tensor<" + "1x" * lead + f"{a}x{b}xf16>"
        self.e(f"%{name}.c = tensor.cast %{src} : {M} to tensor<{a}x{b}xf16>")
        self.e(f"%{name} = tensor.expand_shape %{name}.c [{list(range(lead + 1))}, [{lead + 1}]] output_shape {[1] * lead + [a, b]} : tensor<{a}x{b}xf16> into {st}")
        return st

    def vec4(self, name, src, c):
        """[C] -> 1xCx1x1"""
        self.e(f"%{name}.c = tensor.cast %{src} : tensor<?xf16> to tensor<{c}xf16>")
        self.e(f"%{name} = tensor.expand_shape %{name}.c [[0, 1, 2, 3]] output_shape [1, {c}, 1, 1] : tensor<{c}xf16> into tensor<1x{c}x1x1xf16>")
        return f"tensor<1x{c}x1x1xf16>"

    def from4(self, name, src, c, s):
        self.e(f"%{name}.c = tensor.collapse_shape %{src} [[0, 1], [2, 3]] : tensor<1x{c}x1x{s}xf16> into tensor<{c}x{s}xf16>")
        self.e(f"%{name} = tensor.cast %{name}.c : tensor<{c}x{s}xf16> to {M}")

    def rows(self, name, src, off, n):
        self.call(f"%{name}", "rows", [f"%{src}", self.idx(off), self.idx(n)], [M, "index", "index"], M)

    def report(self, label, lit, nf, exact=True):
        self.e(f'vector.print str "{label}: "')
        self.call("", "report", [f"%{lit}", f"%{nf}"], [M, M], "()")
        self.l[-1] = self.l[-1].replace(" = call", "call").replace("  call", "  call")
        self.checks.append(f"// CHECK: {label}: maxabs {{{{[0-9.e+-]+}}}} maxdiff " + ("0{{$}}" if exact else "{{0(\\.00[0-9]+)?$}}"))

    def taps(self, lit, nfs, sizes, labels, prefix, exact=True):
        off = 0
        for nf, n, lab in zip(nfs, sizes, labels):
            self.rows(f"{prefix}.{lab}", lit, off, n); off += n
            self.report(f"{prefix} {lab}", f"{prefix}.{lab}", nf, exact)

    def write(self, fname):
        body = "\n".join(self.l).replace("   = call", "  call").replace("  = call", "call")
        with open(os.path.join(OUT, fname), "w") as fh:
            fh.write(self.head + RUN.format(lit=self.lit) + "\n" + "\n".join(self.checks) + "\n\nfunc.func @main() {\n" + body + "\n  return\n}\n")


# ----------------------------------------------------------------------------------------------------------------------
D, HID, H, S, HD, V = 8, 12, 2, 6, 4, 10
t = T("static", "Static-weight kernels (training/stories_mil.h, ane_rmsnorm_bwd.h, ane_classifier.h) against their normal forms.", f"""
Sizes: DIM={D} HIDDEN={HID} HEADS={H} HD={HD} SEQ={S} VOCAB={V} (the generators re-run with small #defines).
maxdiff 0 means bit-identical f16 results.  Where the normal form changes the order of the f16 arithmetic
(rsqrt for pow(-0.5); one fused loop for a chain of mul/sub) the check is a bound, and the measured value is printed.""")
t.f16("eps", "1.0e-05"); t.f16("scale", "0.5")
t.e(f"%mask = call @causal_mask({t.idx(S)}) : (index) -> {M}")
mk4 = t.lead("mask4", "mask", S, S, 2)
t.gen("X", D, S, 1.0); t.gen1("rw", D)
for w, (o, i) in {"Wq": (D, D), "Wk": (D, D), "Wv": (D, D), "Wo": (D, D), "W1": (HID, D), "W3": (HID, D), "W2": (D, HID), "We": (V, D)}.items():
    t.gen(w, o, i)
x4 = t.act4("X4", "X", D, S); rw4 = t.vec4("rw4", "rw", D)
wt = {w: t.w4(w + "4", w, o, i) for w, (o, i) in {"Wq": (D, D), "Wk": (D, D), "Wv": (D, D), "Wo": (D, D), "W1": (HID, D), "W3": (HID, D), "W2": (D, HID), "We": (V, D)}.items()}

# sdpa_fwd_taps
t.call("%a.l4", "ane_sdpa_fwd_taps", ["%X4", "%rw4", "%Wq4", "%Wk4", "%Wv4", "%Wo4", "%mask4"], [x4, rw4, wt["Wq"], wt["Wk"], wt["Wv"], wt["Wo"], mk4], f"tensor<1x{6*D}x1x{S}xf16>")
t.from4("a.l", "a.l4", 6 * D, S)
t.call("%a.o, %a.q, %a.k, %a.v, %a.a, %a.xn", "nf_sdpa_fwd", ["%X", "%rw", "%Wq", "%Wk", "%Wv", "%Wo", "%mask", t.idx(H), "%eps", "%scale"],
       [M, "tensor<?xf16>", M, M, M, M, M, "index", "f16", "f16"], "(" + ", ".join([M] * 6) + ")")
t.taps("a.l", ["a.o", "a.q", "a.k", "a.v", "a.a", "a.xn"], [D] * 6, ["o", "q", "k", "v", "attn", "xnorm"], "sdpa_fwd", exact=False)

# ffn_fwd_taps
t.call("%b.l4", "ane_ffn_fwd_taps", ["%X4", "%rw4", "%W14", "%W34", "%W24"], [x4, rw4, wt["W1"], wt["W3"], wt["W2"]], f"tensor<1x{2*D+3*HID}x1x{S}xf16>")
t.from4("b.l", "b.l4", 2 * D + 3 * HID, S)
t.call("%b.y, %b.h1, %b.h3, %b.g, %b.xn", "nf_ffn_fwd", ["%X", "%rw", "%W1", "%W3", "%W2", "%eps"], [M, "tensor<?xf16>", M, M, M, "f16"], "(" + ", ".join([M] * 5) + ")")
t.taps("b.l", ["b.y", "b.h1", "b.h3", "b.g", "b.xn"], [D, HID, HID, HID, D], ["y", "h1", "h3", "gate", "xnorm"], "ffn_fwd", exact=False)

# ffn_bwd: literal takes baked transposes
t.gen("dy", D, S)
for w in ("W1", "W2", "W3", "Wq", "Wk", "Wv", "Wo"):
    t.call(f"%{w}t", "transpose", [f"%{w}"], [M], M)
tw = {"W2t": t.w4("W2t4", "W2t", HID, D), "W1t": t.w4("W1t4", "W1t", D, HID), "W3t": t.w4("W3t4", "W3t", D, HID)}
t.cat("c.in", ["dy", "b.h1", "b.h3"])
ci = t.act4("c.in4", "c.in", D + 2 * HID, S)
t.call("%c.l4", "ane_ffn_bwd", ["%c.in4", "%W2t4", "%W1t4", "%W3t4"], [ci, tw["W2t"], tw["W1t"], tw["W3t"]], ci)
t.from4("c.l", "c.l4", D + 2 * HID, S)
t.call("%c.dx, %c.dh1, %c.dh3", "nf_ffn_bwd", ["%dy", "%b.h1", "%b.h3", "%W1", "%W3", "%W2"], [M] * 6, "(" + ", ".join([M] * 3) + ")")
t.taps("c.l", ["c.dx", "c.dh1", "c.dh3"], [D, HID, HID], ["dx", "dh1", "dh3"], "ffn_bwd")

# sdpa_bwd1 -> sdpa_bwd2
t.gen("do", D, S)
t.cat("d.in", ["a.q", "a.k", "a.v", "do"])
di = t.act4("d.in4", "d.in", 4 * D, S)
wot = t.w4("Wot4", "Wot", D, D)
SC = H * S
t.call("%d.l4", "ane_sdpa_bwd1", ["%d.in4", "%Wot4", "%mask4"], [di, wot, mk4], f"tensor<1x{D+2*SC}x1x{S}xf16>")
t.from4("d.l", "d.l4", D + 2 * SC, S)
t.rows("d.dv", "d.l", 0, D); t.rows("d.pdp", "d.l", D, 2 * SC)
t.cat("e.in", ["d.pdp", "a.q", "a.k"])
ei = t.act4("e.in4", "e.in", 2 * SC + 2 * D, S)
t.call("%e.l4", "ane_sdpa_bwd2", ["%e.in4"], [ei], f"tensor<1x{2*D}x1x{S}xf16>")
t.from4("e.l", "e.l4", 2 * D, S)
t.call("%e.dq, %e.dk, %e.dv", "nf_sdpa_bwd", ["%a.q", "%a.k", "%a.v", "%do", "%Wo", "%mask", t.idx(H), "%scale"], [M] * 6 + ["index", "f16"], "(" + ", ".join([M] * 3) + ")")
t.report("sdpa_bwd dv", "d.dv", "e.dv")
t.taps("e.l", ["e.dq", "e.dk"], [D, D], ["dq", "dk"], "sdpa_bwd")

# qkvb
t.cat("f.in", ["e.dq", "e.dk", "e.dv"])
fi = t.act4("f.in4", "f.in", 3 * D, S)
t.call("%f.l4", "ane_qkvb", ["%f.in4", "%Wqt4", "%Wkt4", "%Wvt4"], [fi, t.w4("Wqt4", "Wqt", D, D), t.w4("Wkt4", "Wkt", D, D), t.w4("Wvt4", "Wvt", D, D)], x4)
t.from4("f.l", "f.l4", D, S)
t.call("%f.n", "nf_qkv_bwd", ["%e.dq", "%e.dk", "%e.dv", "%Wq", "%Wk", "%Wv"], [M] * 6, M)
t.report("qkv_bwd dx", "f.l", "f.n")

# rmsnorm_bwd, final_rmsnorm
t.cat("g.in", ["dy", "X"])
gi = t.act4("g.in4", "g.in", 2 * D, S)
t.call("%g.l4", "ane_rmsnorm_bwd", ["%g.in4", "%rw4"], [gi, rw4], x4)
t.from4("g.l", "g.l4", D, S)
t.call("%g.n", "rmsnorm_bwd", ["%dy", "%X", "%rw", "%eps"], [M, M, "tensor<?xf16>", "f16"], M)
t.report("rmsnorm_bwd dx", "g.l", "g.n", exact=False)
t.call("%h.l4", "ane_final_rmsnorm", ["%X4", "%rw4"], [x4, rw4], x4)
t.from4("h.l", "h.l4", D, S)
t.call("%h.n", "rmsnorm", ["%X", "%rw", "%eps"], [M, "tensor<?xf16>", "f16"], M)
t.report("final_rmsnorm", "h.l", "h.n", exact=False)

# classifier fwd/bwd, softmax
t.call("%k.l4", "ane_classifier_fwd", ["%X4", "%We4"], [x4, wt["We"]], f"tensor<1x{V}x1x{S}xf16>")
t.from4("k.l", "k.l4", V, S)
t.call("%k.n", "linear", ["%We", "%X"], [M, M], M)
t.report("classifier_fwd", "k.l", "k.n")
t.call("%Wet", "transpose", ["%We"], [M], M)
wet3 = t.lead("Wet3", "Wet", D, V, 1)
t.call("%m.l4", "ane_classifier_bwd", ["%k.l4", "%Wet3"], [f"tensor<1x{V}x1x{S}xf16>", wet3], x4)
t.from4("m.l", "m.l4", D, S)
t.call("%m.n", "linear_t", ["%We", "%k.l"], [M, M], M)
t.report("classifier_bwd", "m.l", "m.n")
t.call("%n.l4", "ane_softmax_vocab", ["%k.l4"], [f"tensor<1x{V}x1x{S}xf16>"], f"tensor<1x{V}x1x{S}xf16>")
t.from4("n.l", "n.l4", V, S)
t.call("%n.n", "softmax_ch", ["%k.l"], [M], M)
t.report("softmax_vocab", "n.l", "n.n")
t.write("check_static.mlir")

# ----------------------------------------------------------------------------------------------------------------------
D, HID, H, KVH, HD, S = 8, 12, 4, 2, 4, 6
QD, KD = H * HD, KVH * HD
t = T("dynamic", "Dynamic-weight kernels (training/training_dynamic/mil_dynamic.h) against their normal forms.", f"""
Sizes: DIM={D} HIDDEN={HID} HEADS={H} KV_HEADS={KVH} HD={HD} SEQ={S} NLAYERS=2, so Q_DIM={QD}, KV_DIM={KD}, GQA_RATIO=2.
A dynamic kernel has one input [1, IC, 1, SEQ + sum(OC)]: activations in the first SEQ columns, weights (stored
[IC, OC], i.e. transposed) in the rest.  The drivers pack that input with hcat and unpack the outputs with rows.

The last two lines show the head-tiling mismatch: sdpaFwd tiles K/V with concat(k, k) (head h <- kv head h mod KVH),
the CPU helpers used by the backward pass (gqa_tile_kv, gqa_reduce_kv) use h div GQA_RATIO.""")
t.f16("scale", "0.5"); t.f16("alpha", "0.5")
t.e(f"%mask = call @causal_mask({t.idx(S)}) : (index) -> {M}")
mk4 = t.lead("mask4", "mask", S, S, 2)
t.e(f"%cos, %sin = call @rope_tables({t.idx(S)}, {t.idx(HD)}) : (index, index) -> ({M}, {M})")
cs4 = t.lead("cos4", "cos", S, HD, 2); t.lead("sin4", "sin", S, HD, 2)

# every plain dyn_matmul kernel
for fn, ic, oc in [("dyn_matmul_mil", D, HID), ("wo_fwd_dynamic", QD, D), ("ffn_bwd_w2t_dynamic", D, HID), ("wot_dynamic", D, QD), ("q_bwd_dynamic", QD, D)]:
    p = fn.split("_dyn")[0] if fn != "dyn_matmul_mil" else "mm"
    t.gen(f"{p}.x", ic, S); t.gen(f"{p}.w", ic, oc)
    t.cat(f"{p}.in", [f"{p}.x", f"{p}.w"], "h")
    it = t.act4(f"{p}.in4", f"{p}.in", ic, S + oc)
    t.call(f"%{p}.l4", "ane_" + fn, [f"%{p}.in4"], [it], f"tensor<1x{oc}x1x{S}xf16>")
    t.from4(f"{p}.l", f"{p}.l4", oc, S)
    t.call(f"%{p}.n", "linear_t", [f"%{p}.w", f"%{p}.x"], [M, M], M)
    t.report(fn, f"{p}.l", f"{p}.n")

# sdpa_fwd_dynamic
t.gen("xn", D, S, 1.0); t.gen("Wqt", D, QD); t.gen("Wkt", D, KD); t.gen("Wvt", D, KD)
t.cat("a.in", ["xn", "Wqt", "Wkt", "Wvt"], "h")
ai = t.act4("a.in4", "a.in", D, S + QD + 2 * KD)
OC = 2 * QD + 2 * KD + D
t.call("%a.l4", "ane_sdpa_fwd_dynamic", ["%a.in4", "%cos4", "%sin4", "%mask4"], [ai, cs4, cs4, mk4], f"tensor<1x{OC}x1x{S}xf16>")
t.from4("a.l", "a.l4", OC, S)
t.call("%a.a, %a.q, %a.k, %a.v", "nf_sdpa_fwd_gqa", ["%xn", "%Wqt", "%Wkt", "%Wvt", "%cos", "%sin", "%mask", t.idx(H), t.idx(KVH), "%scale"],
       [M] * 7 + ["index", "index", "f16"], "(" + ", ".join([M] * 4) + ")")
t.taps("a.l", ["a.a", "a.q", "a.k", "a.v", "xn"], [QD, QD, KD, KD, D], ["attn", "q_rope", "k_rope", "v", "xnorm"], "sdpa_fwd_gqa")

# ffn_fused_dynamic
t.gen("x2", D, S, 1.0); t.gen("W1t", D, HID); t.gen("W3t", D, HID); t.gen("W2", D, HID)
t.cat("b.in", ["xn", "x2", "W1t", "W3t", "W2"], "h")
bi = t.act4("b.in4", "b.in", D, 2 * S + 3 * HID)
t.call("%b.l4", "ane_ffn_fused_dynamic", ["%b.in4"], [bi], f"tensor<1x{D+3*HID}x1x{S}xf16>")
t.from4("b.l", "b.l4", D + 3 * HID, S)
t.call("%b.x, %b.h1, %b.h3, %b.g", "nf_ffn_fused", ["%xn", "%x2", "%W1t", "%W3t", "%W2", "%alpha"], [M] * 5 + ["f16"], "(" + ", ".join([M] * 4) + ")")
t.taps("b.l", ["b.x", "b.h1", "b.h3", "b.g"], [D, HID, HID, HID], ["x_next", "h1", "h3", "gate"], "ffn_fused")

# two-input backward kernels: dx = A^T da + B^T db
for fn, ic, lab in [("ffn_bwd_w13t_dynamic", HID, "w13t"), ("kv_bwd_dynamic", KD, "kvb")]:
    for n in ("da", "db"):
        t.gen(f"{lab}.{n}", ic, S)
    for n in ("A", "B"):
        t.gen(f"{lab}.{n}", ic, D)
    t.cat(f"{lab}.in", [f"{lab}.da", f"{lab}.db", f"{lab}.A", f"{lab}.B"], "h")
    it = t.act4(f"{lab}.in4", f"{lab}.in", ic, 2 * S + 2 * D)
    t.call(f"%{lab}.l4", "ane_" + fn, [f"%{lab}.in4"], [it], f"tensor<1x{D}x1x{S}xf16>")
    t.from4(f"{lab}.l", f"{lab}.l4", D, S)
    t.call(f"%{lab}.n1", "linear_t", [f"%{lab}.A", f"%{lab}.da"], [M, M], M)
    t.call(f"%{lab}.n2", "linear_t", [f"%{lab}.B", f"%{lab}.db"], [M, M], M)
    t.call(f"%{lab}.n", "add", [f"%{lab}.n1", f"%{lab}.n2"], [M, M], M)
    t.report(fn, f"{lab}.l", f"{lab}.n")

# sdpa_bwd1_noweight -> sdpa_bwd2, on already-tiled K and V
for n in ("q", "k", "v", "da"):
    t.gen(f"s.{n}", QD, S)
t.cat("s.in", ["s.q", "s.k", "s.v", "s.da"])
si = t.act4("s.in4", "s.in", 4 * QD, S)
SC = H * S
t.call("%s.l4", "ane_sdpa_bwd1_noweight", ["%s.in4", "%mask4"], [si, mk4], f"tensor<1x{QD+2*SC}x1x{S}xf16>")
t.from4("s.l", "s.l4", QD + 2 * SC, S)
t.rows("s.dv", "s.l", 0, QD); t.rows("s.pdp", "s.l", QD, 2 * SC)
t.cat("u.in", ["s.pdp", "s.q", "s.k"])
ui = t.act4("u.in4", "u.in", 2 * SC + 2 * QD, S)
t.call("%u.l4", "ane_sdpa_bwd2", ["%u.in4"], [ui], f"tensor<1x{2*QD}x1x{S}xf16>")
t.from4("u.l", "u.l4", 2 * QD, S)
H3 = "tensor<?x?x?xf16>"
for n in ("q", "k", "v", "da"):
    t.call(f"%s.{n}h", "split_heads", [f"%s.{n}", t.idx(H)], [M, "index"], H3)
t.call("%u.dq3, %u.dk3, %u.dv3", "sdpa_bwd", ["%s.qh", "%s.kh", "%s.vh", "%mask", "%scale", "%s.dah"], [H3, H3, H3, M, "f16", H3], f"({H3}, {H3}, {H3})")
for n in ("dq", "dk", "dv"):
    t.call(f"%u.{n}", "merge_heads", [f"%u.{n}3"], [H3], M)
t.report("sdpa_bwd dv", "s.dv", "u.dv")
t.taps("u.l", ["u.dq", "u.dk"], [QD, QD], ["dq", "dk"], "sdpa_bwd")

# tiling conventions
t.gen("g.k", KD, S)
t.call("%g.kh", "split_heads", ["%g.k", t.idx(KVH)], [M, "index"], H3)
for c in ("mod", "div"):
    t.call(f"%g.{c}3", f"gqa_tile_{c}", ["%g.kh", t.idx(H)], [H3, "index"], H3)
    t.call(f"%g.{c}", "merge_heads", [f"%g.{c}3"], [H3], M)
t.cat("g.cat", ["g.k", "g.k"])
t.report("gqa concat(k,k) vs tile_mod", "g.cat", "g.mod")
t.e('vector.print str "gqa tile_mod vs tile_div: "')
t.e(f"call @report(%g.mod, %g.div) : ({M}, {M}) -> ()")
t.checks.append("// CHECK: gqa tile_mod vs tile_div: maxabs {{[0-9.e+-]+}} maxdiff 0.{{[0-9]*[1-9]}}")
t.write("check_dynamic.mlir")

# ----------------------------------------------------------------------------------------------------------------------
C, O, SP = 4, 6, 5
t = T("primitives", "Single-op generators (training/ane_mil_gen.h) and the W8A8 chain (ane_int8_bench.m) against their normal forms.", f"""
Sizes: in_ch={C} out_ch={O} spatial={SP}; the int8 chain is 3 layers of {C}x{C} on a {SP}x{SP} image.
ane_mil_gen.h kernels take and return fp32 and cast at both ends: the normal form is to_f32(linear(to_f16(.))).""")
M32 = "tensor<?x?xf32>"
t.e("%s0 = arith.constant 0.4 : f32"); t.e("%s1 = arith.constant 2.2 : f32"); t.e("%amp = arith.constant 0.9 : f32")
t.call("%x32", "gen32", [t.idx(C), t.idx(SP), "%s0", "%amp"], ["index", "index", "f32", "f32"], M32)
t.call("%w32", "gen32", [t.idx(O), t.idx(C), "%s1", "%amp"], ["index", "index", "f32", "f32"], M32)
t.call("%x16", "to_f16", ["%x32"], [M32], M); t.call("%w16", "to_f16", ["%w32"], [M32], M)
t.call("%n16", "linear", ["%w16", "%x16"], [M, M], M)
# matmul: fp32 x [1,C,SP], W [1,O,C]
t.e(f"%x3c = tensor.cast %x32 : {M32} to tensor<{C}x{SP}xf32>")
t.e(f"%x3 = tensor.expand_shape %x3c [[0, 1], [2]] output_shape [1, {C}, {SP}] : tensor<{C}x{SP}xf32> into tensor<1x{C}x{SP}xf32>")
t.e(f"%w3c = tensor.cast %w32 : {M32} to tensor<{O}x{C}xf32>")
t.e(f"%w3 = tensor.expand_shape %w3c [[0, 1], [2]] output_shape [1, {O}, {C}] : tensor<{O}x{C}xf32> into tensor<1x{O}x{C}xf32>")
t.call("%mm3", "ane_matmul", ["%x3", "%w3"], [f"tensor<1x{C}x{SP}xf32>", f"tensor<1x{O}x{C}xf32>"], f"tensor<1x{O}x{SP}xf32>")
t.e(f"%mmc = tensor.collapse_shape %mm3 [[0, 1], [2]] : tensor<1x{O}x{SP}xf32> into tensor<{O}x{SP}xf32>")
t.e(f"%mmd = tensor.cast %mmc : tensor<{O}x{SP}xf32> to {M32}")
t.call("%mm", "to_f16", ["%mmd"], [M32], M)
t.report("matmul (fp32 in/out)", "mm", "n16")
# conv: fp32 x [1,C,1,SP], baked fp16 W
t.e(f"%x4 = tensor.expand_shape %x3c [[0, 1], [2, 3]] output_shape [1, {C}, 1, {SP}] : tensor<{C}x{SP}xf32> into tensor<1x{C}x1x{SP}xf32>")
wt = t.w4("w4", "w16", O, C)
t.call("%cv4", "ane_conv", ["%x4", "%w4"], [f"tensor<1x{C}x1x{SP}xf32>", wt], f"tensor<1x{O}x1x{SP}xf32>")
t.e(f"%cvc = tensor.collapse_shape %cv4 [[0, 1], [2, 3]] : tensor<1x{O}x1x{SP}xf32> into tensor<{O}x{SP}xf32>")
t.e(f"%cvd = tensor.cast %cvc : tensor<{O}x{SP}xf32> to {M32}")
t.call("%cv", "to_f16", ["%cvd"], [M32], M)
t.report("conv (fp32 in/out)", "cv", "n16")
# W8A8 chain
M8 = "tensor<?x?xi8>"
t.f16("qs", "0.125")
t.e("%a8 = arith.constant 100.0 : f32"); t.e("%ax = arith.constant 3.0 : f32")
t.call("%qx32", "gen32", [t.idx(C), t.idx(SP * SP), "%s0", "%ax"], ["index", "index", "f32", "f32"], M32)
t.call("%qx", "to_f16", ["%qx32"], [M32], M)
t.e(f"%qxc = tensor.cast %qx : {M} to tensor<{C}x{SP*SP}xf16>")
t.e(f"%qx4 = tensor.expand_shape %qxc [[0, 1], [2, 3]] output_shape [1, {C}, {SP}, {SP}] : tensor<{C}x{SP*SP}xf16> into tensor<1x{C}x{SP}x{SP}xf16>")
prev = "qx"
for i in range(3):
    t.e(f"%ws{i} = arith.constant {0.7 + i} : f32")
    t.call(f"%wq{i}", "gen8", [t.idx(C), t.idx(C), f"%ws{i}", "%a8"], ["index", "index", "f32", "f32"], M8)
    t.e(f"%wq{i}c = tensor.cast %wq{i} : {M8} to tensor<{C}x{C}xi8>")
    t.e(f"%wq{i}4 = tensor.expand_shape %wq{i}c [[0], [1, 2, 3]] output_shape [{C}, {C}, 1, 1] : tensor<{C}x{C}xi8> into tensor<{C}x{C}x1x1xi8>")
    t.call(f"%wd{i}", "dequantize", [f"%wq{i}", "%qs"], [M8, "f16"], M)
    t.call(f"%c{i}", "linear", [f"%wd{i}", f"%{prev}"], [M, M], M)
    if i < 2:
        t.call(f"%q{i}", "quantize", [f"%c{i}", "%qs"], [M, "f16"], M8)
        t.call(f"%dq{i}", "dequantize", [f"%q{i}", "%qs"], [M8, "f16"], M)
        prev = f"dq{i}"
I4 = f"tensor<1x{C}x{SP}x{SP}xf16>"; W8 = f"tensor<{C}x{C}x1x1xi8>"
t.call("%i8l4", "ane_genMILInt8", ["%qx4", "%wq04", "%wq14", "%wq24"], [I4, W8, W8, W8], I4)
t.e(f"%i8lc = tensor.collapse_shape %i8l4 [[0, 1], [2, 3]] : {I4} into tensor<{C}x{SP*SP}xf16>")
t.e(f"%i8l = tensor.cast %i8lc : tensor<{C}x{SP*SP}xf16> to {M}")
t.report("W8A8 chain (3 layers)", "i8l", "c2")
t.write("check_primitives.mlir")

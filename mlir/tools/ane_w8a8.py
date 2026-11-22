#!/usr/bin/env python3
"""The 3-layer W8A8 chain of ane_int8_bench.m (genMILInt8, 64 channels, 32x32 image): on the device, through the ane dialect, on the CPU.

usage: ane_w8a8.py [WORKDIR]    needs numpy, xcrun clang and LLVM with MLIR (LLVM_PREFIX, default /opt/homebrew/opt/llvm); a few seconds

  1. the generator's own MIL (tools/dump_gen.m) runs on the device; the MIL printed back from the dialect (ane_dialect.py) runs on the
     device too, and the two outputs must be byte-identical;
  2. the same chain through mil2mlir.py (fp16 accumulation, and --wide: f32 accumulation) runs on the CPU (mlir-runner); the
     comparison with the device is in fp16 ulps;
  3. a numpy model of the chain (exact sums, one fp16 rounding per tensor, the quantize / dequantize rules of README "Sharp corners")
     is compared in the same way.
The weights are random int8 in [-QMAX, QMAX]; the run repeats for QMAX 9 (a few values saturate at 127) and 24 (many do).
"""
import os, re, struct, subprocess, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); T = f"{ROOT}/mlir/tools"
W = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ane_w8a8_"); os.makedirs(W, exist_ok=True)
LP = os.environ.get("LLVM_PREFIX", "/opt/homebrew/opt/llvm"); LB, LL = f"{LP}/bin", f"{LP}/lib"; P = sys.executable
CH, SP, DEPTH = 64, 32, 3; NB = CH * SP * SP * 2
IRDL = f"{ROOT}/mlir/dialect/ane.irdl.mlir"
XSDK = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
# macOS 27.2.1 + Xcode 26.3: `xcrun clang` defaults to the CommandLineTools 27.2 SDK, whose libobjc.tbd (arch arm64e.x1-macos)
# the Xcode linker cannot read.  Build against Xcode's own SDK when it is there.
BUILD_ENV = dict(os.environ, **({} if "SDKROOT" in os.environ or not os.path.isdir(XSDK) else {"SDKROOT": XSDK}))

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=BUILD_ENV)
    if r.returncode: sys.exit(f"FAILED: {cmd[:220]}\n{r.stderr[:1500]}")
    return r.stdout

sh(f"cd {ROOT} && xcrun clang -w -fobjc-arc -framework Foundation -framework IOSurface -Itraining -o {W}/dg mlir/tools/dump_gen.m")
sh(f"cd {ROOT} && xcrun clang -w -fobjc-arc -framework Foundation -framework IOSurface -framework Accelerate -framework CoreML -DACCELERATE_NEW_LAPACK "
   f"-Itraining/training_dynamic -include mlir/tools/mid_dynamic.h -o {W}/ane_try mlir/tools/ane_try.m")
mil = re.split(r"^=== genMILInt8\n", sh(f"{W}/dg {CH} 96 {SP}"), flags=re.M)[1].split("\n=== ")[0]
open(f"{W}/w8.mil", "w").write(mil)
print(f"generator MIL: {len(mil)} bytes, {mil.count('constexpr_affine_dequantize')} constexpr_affine_dequantize, {mil.count('quantize(input')} quantize")

# the dialect round trip: MIL -> ane ops (checked against the IRDL declaration) -> MIL
ane = sh(f"printf '=== genMILInt8\\n' | cat - {W}/w8.mil | {P} {T}/ane_dialect.py from-mil | {LB}/mlir-opt --irdl-file={IRDL}")
open(f"{W}/w8.ane.mlir", "w").write(ane)
re_mil = sh(f"{LB}/mlir-opt --irdl-file={IRDL} {W}/w8.ane.mlir | {P} {T}/ane_dialect.py to-mil").split("=== ", 1)[1].split("\n", 1)[1]
open(f"{W}/w8.re.mil", "w").write(re_mil)
print(f"ane ops: {len(re.findall(r'"ane[.][a-z_]+"', ane))} (verified against {os.path.basename(IRDL)}); the dialect prints its own layout of the MIL, so the device comparison below is the check")

def f16(v): return np.asarray(v, np.float64).astype(np.float16).astype(np.float64)
def away(v): return np.sign(v) * np.floor(np.abs(v) + 0.5)
def f16_away(v):
    """the fp16 nearest to v; an exact tie goes to the neighbour of larger magnitude"""
    v = np.asarray(v, np.float64); r = v.astype(np.float16); rr = r.astype(np.float64)
    nb = np.where(v > rr, np.nextafter(r, np.float16(np.inf)), np.nextafter(r, np.float16(-np.inf))).astype(np.float64)
    tie = (v != rr) & (np.abs(v - rr) == np.abs(nb - v))
    return np.where(tie & (np.abs(nb) > np.abs(rr)), nb, rr)
def ulp(v): v = np.abs(v); return 2.0 ** (np.floor(np.log2(np.maximum(v, 2.0 ** -14))) - 10)
def report(label, a, y):
    d = np.abs(a - y) / ulp(np.maximum(np.abs(a), np.abs(y)))
    print(f"  {label:34s} identical {100 * np.mean(d == 0):5.1f}%  within 1 ulp {100 * np.mean(d <= 1):5.1f}%  p99 {np.percentile(d, 99):4.0f} ulp  max {d.max():5.0f} ulp  maxabsdiff {np.abs(a - y).max():.3g}")

def run(QMAX):
    rng = np.random.default_rng(5)
    q = [rng.integers(-QMAX, QMAX + 1, (CH, CH)).astype(np.int8) for _ in range(DEPTH)]
    cs = 64 + CH * CH; blob = bytearray(64 + cs * DEPTH); blob[0] = 1; blob[4] = 2
    for i in range(DEPTH):
        c = 64 + i * cs; blob[c:c + 4] = b"\xef\xbe\xad\xde"; blob[c + 4] = 1
        struct.pack_into("<I", blob, c + 8, CH * CH); struct.pack_into("<I", blob, c + 16, c + 64); blob[c + 64:c + 64 + CH * CH] = q[i].tobytes()
    open(f"{W}/w8.bin", "wb").write(blob)
    x = rng.uniform(-.5, .5, (1, CH, SP, SP)).astype(np.float16); open(f"{W}/w8.inb", "wb").write(x.astype("<f2").tobytes())

    def dev(tag, milfile):
        r = subprocess.run(["perl", "-e", "alarm 120; exec @ARGV", f"{W}/ane_try", milfile, str(NB), str(NB), "f16", "f16", f"{W}/w8.bin"],
                           capture_output=True, text=True, env=dict(os.environ, ANE_IN=f"{W}/w8.inb", ANE_DUMP=f"{W}/{tag}"))
        assert "eval=1" in r.stdout + r.stderr, (r.stdout + r.stderr)[:400]
        return np.frombuffer(open(f"{W}/{tag}.out", "rb").read(), "<f2").copy()
    a, b = dev("w8_gen", f"{W}/w8.mil"), dev("w8_re", f"{W}/w8.re.mil")
    ad = a.astype(np.float64)
    print(f"QMAX {QMAX}: device output, generator MIL vs MIL printed from the dialect: byte-identical {a.tobytes() == b.tobytes()}; "
          f"{a.size} values, max|y| {np.abs(ad).max():g}, zeros {int((a == 0).sum())}")

    def cpu(wide):
        lin = sh(f"printf '=== genMILInt8\\n' | cat - {W}/w8.re.mil | {P} {T}/mil2mlir.py {'--wide' if wide else ''}")
        fn = re.search(r"func\.func @(\w+)\(", lin).group(1); sig = re.search(r"func\.func @\w+\((.*?)\)\s*\n?\s*-> \((.*?)\)", lin, re.S)
        cst = lambda arr, ty: f"dense<{arr.astype(np.float64).tolist() if ty == 'f16' else arr.tolist()}> : tensor<{'x'.join(map(str, arr.shape))}x{ty}>"
        T4 = f"tensor<1x{CH}x{SP}x{SP}xf"; imap = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"
        main = ["func.func private @printMemrefF32(%ptr : tensor<*xf32>)", "func.func @main() {", f"  %x = arith.constant {cst(x, 'f16')}"]
        main += [f"  %w{i} = arith.constant {cst(q[i].reshape(CH, CH, 1, 1), 'i8')}" for i in range(DEPTH)]
        main += [f"  %y = call @{fn}(%x, " + ", ".join(f"%w{i}" for i in range(DEPTH)) + f") : ({', '.join(re.findall(r'%\w+: ([^,)]+)', sig.group(1)))}) -> ({sig.group(2)})",
                 f"  %e = tensor.empty() : {T4}32>",
                 f'  %f = linalg.generic {{indexing_maps = [{imap}, {imap}], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}} '
                 f"ins(%y : {T4}16>) outs(%e : {T4}32>) {{", "    ^bb0(%a : f16, %o : f32):", "      %c = arith.extf %a : f16 to f32", "      linalg.yield %c : f32",
                 f"  }} -> {T4}32>", f"  %u = tensor.cast %f : {T4}32> to tensor<*xf32>", "  call @printMemrefF32(%u) : (tensor<*xf32>) -> ()", "  return", "}"]
        open(f"{W}/w8_main.mlir", "w").write(lin + "\n" + "\n".join(main) + "\n")
        pipe = (f"--transform-preload-library=transform-library-paths={ROOT}/mlir/test/decompose_softmax.mlir --transform-interpreter --linalg-generalize-named-ops "
                "--one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops --expand-strided-metadata --lower-affine --convert-scf-to-cf "
                "--convert-math-to-llvm --convert-math-to-libm --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm "
                "--convert-cf-to-llvm --reconcile-unrealized-casts")
        out = sh(f"cat {W}/w8_main.mlir | {LB}/mlir-opt {pipe} | {LB}/mlir-runner -e main -entry-point-result=void "
                 f"-shared-libs={LL}/libmlir_runner_utils.dylib,{LL}/libmlir_c_runner_utils.dylib")
        vals = [float(v) for l in out.splitlines() if not l.startswith("Unranked") for v in re.findall(r"-?(?:nan|inf|\d+\.?\d*(?:e[-+]?\d+)?)", l)]
        assert len(vals) == a.size, (len(vals), a.size)
        return f16(vals)
    for wide in (False, True): report(f"CPU, {'f32' if wide else 'fp16'} accumulation vs device", ad, cpu(wide))

    h = x.astype(np.float64).reshape(CH, SP * SP); sat = []
    for i in range(DEPTH):
        h = f16((q[i].astype(np.float64) * 0.125) @ h)                       # 1x1 conv: exact sum, one fp16 rounding
        if i < DEPTH - 1:
            t = np.clip(away(f16(h * f16(1 / 0.125))), -128, 127); sat.append(float(np.mean(np.abs(t) >= 127)))
            h = f16_away(t * 0.125)                                           # quantize, dequantize as the device does them
    report("numpy model vs device", ad, h.reshape(-1))
    print(f"  (share of int8 activations at the clamp after layers 1 and 2: {sat[0]:.3f}, {sat[1]:.3f})")

for qmax in (9, 24): run(qmax)

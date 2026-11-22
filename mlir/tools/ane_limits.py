#!/usr/bin/env python3
"""Re-measure the device facts that README.md and dialect/ane.irdl.mlir state, on whatever Mac runs this.

usage: ane_limits.py [WORKDIR]   needs numpy and xcrun clang; about a minute; writes only to WORKDIR (default: a temp dir)

One line per check: what ran, what the device did, and `same` or `DIFFERS` against the last run (M3 Pro, macOS 27.2.1).
  ok    compiled, evaluated, and matched a CPU reference to 1% of its largest value
  wrong evaluated without an error but did not match the reference
  fail  compile or eval error
The exit status is the number of checks that differ.  After a macOS or chip change, read the DIFFERS lines and update
the README table "Device facts" and the comments in dialect/ane.irdl.mlir.
"""
import math, os, random, re, shutil, struct, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ane_limits_")
os.makedirs(W, exist_ok=True)
TRY = f"{W}/ane_try"
XSDK = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
# macOS 27.2.1 + Xcode 26.3: `xcrun clang` defaults to the CommandLineTools 27.2 SDK, whose libobjc.tbd (arch arm64e.x1-macos)
# the Xcode linker cannot read.  Build against Xcode's own SDK when it is there.
BUILD_ENV = dict(os.environ, **({} if "SDKROOT" in os.environ or not os.path.isdir(XSDK) else {"SDKROOT": XSDK}))
RNG = np.random.default_rng(7)

r = subprocess.run(["xcrun", "clang", "-w", "-fobjc-arc", "-framework", "Foundation", "-framework", "IOSurface", "-framework", "Accelerate",
                    "-framework", "CoreML", "-DACCELERATE_NEW_LAPACK", f"-I{ROOT}/training/training_dynamic", "-include",
                    f"{ROOT}/mlir/tools/mid_dynamic.h", "-o", TRY, f"{ROOT}/mlir/tools/ane_try.m"], capture_output=True, text=True, env=BUILD_ENV)
if r.returncode: sys.exit("building ane_try failed:\n" + r.stderr[:2000])

def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout.strip()

def ane(tag, prog, insz, outsz, io="f16", weights=None, inb=None, attr=False):
    """prog: a .mil path or dynmm:IC,OC,SEQ.  Returns ane_try's output; the surfaces are dumped to W/tag.in and W/tag.out."""
    env = dict(os.environ, ANE_DUMP=f"{W}/{tag}")
    if inb is not None:
        open(f"{W}/{tag}.inb", "wb").write(inb); env["ANE_IN"] = f"{W}/{tag}.inb"
    if attr: env["ANE_ATTR"] = "1"
    a = [TRY, prog, str(insz), str(outsz), io, io] + ([weights] if weights else [])
    r = subprocess.run(["perl", "-e", "alarm 280; exec @ARGV"] + a, capture_output=True, text=True, env=env)
    return r.stdout + r.stderr

def status(txt, ref):
    """(ok|wrong|fail, detail).  ref() -> (max|reference|, max|device - reference|), called only after a clean eval."""
    if "compile=OK" not in txt:
        return "fail", " ".join(sorted(set(re.findall(r"InvalidMILProgram|CompilationFailure", txt)))) or "compile error"
    if "eval=1" not in txt:
        return "fail", "eval: " + (re.search(r"err=(.{0,90})", txt) or [0, "?"])[1].replace("\n", " ")
    mr, md = ref()
    return ("ok" if md <= 0.01 * mr else "wrong"), f"max|ref|={mr:.4g} max|diff|={md:.3g} ({100 * md / mr:.2f}%)"

DIFFS = []; N = [0]
def check(name, expect, st, detail=""):
    N[0] += 1; same = st == expect
    if not same: DIFFS.append(name)
    print(f"  {'same' if same else 'DIFFERS':7s} {st:5s} {name:56s} {detail}" + ("" if same else f"   (27.2.1: {expect})"), flush=True)

# ---- dynamic matmul: contracted dimension, and the 64-byte row stride -----------------------------------------------
def stride64(nbytes): return math.ceil(nbytes / 64) * 64

def dynmm(IC, OC, SEQ, staged=True, tag="mm"):
    """[1,IC,1,SEQ+OC] -> [1,OC,1,SEQ], y[o,s] = sum_d x[d,SEQ+o] x[d,s].  staged: rows written at the model's stride."""
    Wd = SEQ + OC; inS, outS = stride64(Wd * 2), stride64(SEQ * 2)
    x = RNG.uniform(-.5, .5, (IC, Wd)).astype(np.float16)
    if staged: buf = np.zeros((IC, inS // 2), "<f2"); buf[:, :Wd] = x; inb = buf.tobytes()
    else: inb = x.tobytes()
    txt = ane(tag, f"dynmm:{IC},{OC},{SEQ}", IC * inS, OC * outS, inb=inb)
    def ref():
        y = np.frombuffer(open(f"{W}/{tag}.out", "rb").read(), "<f2").reshape(OC, outS // 2)[:, :SEQ].astype(np.float64)
        rows = sorted({0, OC // 2, OC - 1}); r = x[:, SEQ:].astype(np.float64).T[rows] @ x[:, :SEQ].astype(np.float64)
        return float(np.abs(r).max()), float(np.abs(y[rows] - r).max())
    return status(txt, ref)

# ---- conv with baked weights ----------------------------------------------------------------------------------------
HDR = '''program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<%(T)s, [1, %(IC)d, 1, %(SP)d]> x) {
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
'''
BLOB = 'BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64))'
S = "[%(OC)d, %(IC)d, 1, 1]"
W16 = f'        tensor<fp16, {S}> W = const()[name = string("W"), val = tensor<fp16, {S}>({BLOB})];\n'
W8CAST = (f'        tensor<int8, {S}> W8 = const()[name = string("W8"), val = tensor<int8, {S}>({BLOB})];\n'
          '        string to16 = const()[name = string("to16"), val = string("fp16")];\n'
          f'        tensor<fp16, {S}> Wc = cast(dtype = to16, x = W8)[name = string("Wc")];\n'
          '        fp16 w_scale = const()[name = string("w_scale"), val = fp16(0x1p-3)];\n'
          f'        tensor<fp16, {S}> W = mul(x = Wc, y = w_scale)[name = string("W")];\n')
def cad(dt, zp):
    return (f'        tensor<fp16, {S}> W = constexpr_affine_dequantize()[axis = int32(0), name = string("W"), quantized_data = '
            f'tensor<{dt}, {S}>({BLOB}), scale = fp16(0x1p-3), zero_point = {dt}({zp})];\n')
def cadarg(dt, zp):     # the same op with its parameters as call arguments instead of in the [...] attribute list
    return (f'        tensor<fp16, {S}> W = constexpr_affine_dequantize(quantized_data = tensor<{dt}, {S}>({BLOB}), scale = fp16(0x1p-3), '
            f'zero_point = {dt}({zp}), axis = int32(0))[name = string("W")];\n')

def blob(payload, size=None, off=128, byte10=None):
    """64-byte file header, then a chunk: magic at +0, payload size at +8, data offset (from file start) at +16.
    byte10: what ane_int8_bench.m, inmem_peak.m and inmem_bench.m write instead (one byte of the size, no offset)."""
    b = bytearray(128 + len(payload)); b[0] = 1; b[4] = 2; c = 64
    b[c:c+4] = b"\xef\xbe\xad\xde"; b[c+4] = 1
    if byte10 is not None: b[c+10] = byte10
    else: struct.pack_into("<I", b, c+8, len(payload) if size is None else size)
    struct.pack_into("<I", b, c+16, off); b[128:] = payload; return bytes(b)

def weights(form, OC, IC, dt="int8", zp=0):
    """(MIL text of the weight, blob payload, rows -> effective float64 weights of those output channels)"""
    if form == "fp16":
        w = (RNG.uniform(-.5, .5, (OC, IC)) * 0.2).astype("<f2"); return W16, w.tobytes(), lambda r: w[r].astype(np.float64)
    q = RNG.integers(-100, 100, (OC, IC)).astype(np.int8) if dt == "int8" else RNG.integers(0, 255, (OC, IC)).astype(np.uint8)
    if form == "i8cast": return W8CAST, q.tobytes(), lambda r: q[r].astype(np.float64) * 0.125
    return (cadarg if form == "cadarg" else cad)(dt, zp), q.tobytes(), lambda r: (q[r].astype(np.float64) - zp) * 0.125

def conv(tag, IC, OC, SP, form="fp16", io="f16", dt="int8", zp=0, **blobkw):
    wpart, payload, weff = weights(form, OC, IC, dt, zp)
    p = dict(IC=IC, OC=OC, SP=SP, T="fp32" if io == "f32" else "fp16"); xin = "x"; m = HDR % p
    if io == "f32":
        m += ('        string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];\n'
              '        tensor<fp16, [1, %(IC)d, 1, %(SP)d]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];\n' % p); xin = "x16"
    m += wpart % p
    m += (f'        tensor<fp16, [1, %(OC)d, 1, %(SP)d]> y{"16" if io == "f32" else ""} = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, '
          f'pad_type = c_pad_type, strides = c_strides, weight = W, x = {xin})[name = string("conv")];\n') % p
    if io == "f32":
        m += ('        string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];\n'
              '        tensor<fp32, [1, %(OC)d, 1, %(SP)d]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];\n' % p)
    m += "    } -> (y);\n}\n"
    open(f"{W}/{tag}.mil", "w").write(m); open(f"{W}/{tag}.bin", "wb").write(blob(payload, **blobkw))
    e = 4 if io == "f32" else 2
    txt = ane(tag, f"{W}/{tag}.mil", IC * SP * e, OC * SP * e, io, weights=f"{W}/{tag}.bin")
    def ref():
        d = "<f4" if io == "f32" else "<f2"
        x = np.frombuffer(open(f"{W}/{tag}.in", "rb").read(), d).astype(np.float64).reshape(IC, SP)
        y = np.frombuffer(open(f"{W}/{tag}.out", "rb").read(), d).astype(np.float64).reshape(OC, SP)
        rows = sorted({0, OC // 3, OC - 1}); r = weff(rows) @ x
        return float(np.abs(r).max()), float(np.abs(y[rows] - r).max())
    return status(txt, ref)

t0 = time.time()
print(f"{sh('sw_vers', '-productName')} {sh('sw_vers', '-productVersion')} ({sh('sw_vers', '-buildVersion')}), {sh('sysctl', '-n', 'machdep.cpu.brand_string')}")
print(f"work dir {W}\n")

print("matmul: contracted dimension (dynamic matmul, OC=128 SEQ=64)")
for IC, ex in ((16384, "ok"), (16385, "fail")): check(f"matmul contracted dim {IC}", ex, *dynmm(IC, 128, 64))

print("conv: input channels, output channels, spatial")
check("conv 32000->64  SP=64", "ok", *conv("c1", 32000, 64, 64))
check("conv 32000->128 SP=64", "ok", *conv("c2", 32000, 128, 64))
check("conv 32000->768 SP=64", "fail", *conv("c3", 32000, 768, 64))
check("conv 32000->768 SP=256, fp32 I/O + casts", "ok", *conv("c4", 32000, 768, 256, io="f32"))
lo, hi = 1024, 65536
assert conv("b", lo, 64, 64)[0] == "ok" and conv("b", hi, 64, 64)[0] == "fail", "bisect bounds"
while hi - lo > 1:
    mid = (lo + hi) // 2
    if conv("b", mid, 64, 64)[0] == "ok": lo = mid
    else: hi = mid
check("conv input-channel ceiling (OC=64 SP=64)", "ok" if lo == 32512 else "fail", "ok" if lo == 32512 else "fail", f"largest that compiles: {lo}")

print("quantized weights (256 -> 256, SP=64; reference (q - zero_point) * 0.125)")
check("int8 blob -> cast fp16 -> mul scale", "ok", *conv("q1", 256, 256, 64, form="i8cast"))
check("constexpr_affine_dequantize int8, zero_point 0", "ok", *conv("q2", 256, 256, 64, form="cad"))
check("constexpr_affine_dequantize int8, zero_point 3", "ok", *conv("q3", 256, 256, 64, form="cad", zp=3))
check("constexpr_affine_dequantize uint8, zero_point 128", "ok", *conv("q4", 256, 256, 64, form="cad", dt="uint8", zp=128))
check("constexpr_affine_dequantize, parameters as call arguments", "fail", *conv("q5", 256, 256, 64, form="cadarg"))

print("weight blob chunk header (fp16 conv 256 -> 256, SP=64)")
check("size at +8 and offset at +16", "ok", *conv("h1", 256, 256, 64))
check("size field wrong (524288)", "ok", *conv("h2", 256, 256, 64, size=524288))
check("size field 0", "ok", *conv("h3", 256, 256, 64, size=0))
check("offset field 0 (data read from the file start)", "wrong", *conv("h4", 256, 256, 64, off=0))
check("what the benches write: byte +10 only, offset 0", "wrong", *conv("h5", 256, 256, 64, byte10=0x08, off=0))

print("channel counts with a correct header (the old '512 only' wall)")
for IC, OC in ((64, 64), (96, 160), (1024, 1024), (2048, 512)): check(f"conv {IC}->{OC} SP=64", "ok", *conv(f"w{IC}", IC, OC, 64))

print("IOSurface rows: stride = ceil(bytes per row / 64) * 64, inputs and outputs")
rows = []
for SEQ, OC in ((32, 1), (32, 8), (32, 16), (32, 17), (32, 31), (32, 32), (32, 33), (32, 48), (32, 96), (8, 32), (16, 32), (24, 32), (40, 32), (48, 48), (60, 36), (31, 33)):
    t = ane("st", f"dynmm:64,{OC},{SEQ}", 4000000, 4000000, attr=True)
    if "compile=OK" not in t: continue
    i, o = t.index("LiveInputList"), t.index("LiveOutputList")
    g = lambda s, k: int(re.findall(k + r" = (\d+);", s)[0])
    rows.append((g(t[i:o], "Width"), g(t[i:o], "RowStride"), g(t[o:], "Width"), g(t[o:], "RowStride")))
good = rows and all(rs == stride64(w * 2) and rs2 == stride64(w2 * 2) for w, rs, w2, rs2 in rows)
check(f"row stride rule over {len(rows)} kernels", "ok", "ok" if good and len(rows) == 16 else "fail")
check("tiny dynmm 8x12x6, rows staged at the stride", "ok", *dynmm(8, 12, 6, staged=True))
check("tiny dynmm 8x12x6, rows packed densely", "wrong", *dynmm(8, 12, 6, staged=False))
check("dynmm 64x96x32 (row 256 B, already aligned), packed", "ok", *dynmm(64, 96, 32, staged=False))

# ---- quantize / dequantize: what the device computes ---------------------------------------------------------------------
def f16(a): return np.asarray(a, np.float64).astype(np.float16).astype(np.float64)
def away(v): return np.sign(v) * np.floor(np.abs(v) + 0.5)
def f16_away(v):
    """the fp16 nearest to v; an exact tie goes to the neighbour of larger magnitude"""
    v = np.asarray(v, np.float64); r = v.astype(np.float16); rr = r.astype(np.float64)
    nb = np.where(v > rr, np.nextafter(r, np.float16(np.inf)), np.nextafter(r, np.float16(-np.inf))).astype(np.float64)
    tie = (v != rr) & (np.abs(v - rr) == np.abs(nb - v))
    return np.where(tie & (np.abs(nb) > np.abs(rr)), nb, rr)

def quant_model(x, scale):
    """q = sat_i8(round_half_away(f16(x * f16(1/s)))),  y = f16_ties_away(q * s)   -- mil2mlir.py's translation of quantize/dequantize"""
    s = float(np.float16(scale)); q = np.clip(away(f16(np.asarray(x, np.float64) * f16(1.0 / s))), -128, 127)
    return q, f16_away(q * s)

def quant_dev(tag, scale, x):
    mil = ('program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, '
           '{"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]\n{\n    func main<ios18>(tensor<fp16, [1,64,1,32]> x) {\n'
           f'        fp16 s = const()[name = string("s"), val = fp16({float(np.float16(scale)).hex()})];\n'
           '        tensor<int8, [1,64,1,32]> q = quantize(input = x, output_dtype = string("int8"), scale = s)[name = string("q")];\n'
           '        tensor<fp16, [1,64,1,32]> y = dequantize(input = q, scale = s)[name = string("y")];\n    } -> (y);\n}\n')
    open(f"{W}/{tag}.mil", "w").write(mil)
    txt = ane(tag, f"{W}/{tag}.mil", 4096, 4096, inb=np.asarray(x, "<f2").tobytes())
    if "eval=1" not in txt: return None
    return np.frombuffer(open(f"{W}/{tag}.out", "rb").read(), "<f2").astype(np.float64)

print("quantize -> dequantize (fp16 [1,64,1,32] in, int8 between, fp16 out; model: q = sat_i8(round_half_away(f16(x * f16(1/s)))), y = f16_ties_away(q * s))")
kk = np.arange(-64, 64); ties = (kk + 0.5) * 0.125                      # exact ties of x / 0.125
x = np.concatenate([ties, RNG.uniform(-16, 16, 2048 - 128)]).astype(np.float16)
y = quant_dev("qt", 0.125, x)
if y is None: check("quantize at exact ties (scale 0.125)", "ok", "fail")
else:
    ya = quant_model(x, 0.125)[1][:128]; ye = np.clip(np.rint(x[:128].astype(np.float64) / 0.125), -128, 127) * 0.125
    check("quantize at exact ties (scale 0.125): half away from zero", "ok", "ok" if np.array_equal(y[:128], ya) and np.array_equal(y, quant_model(x, 0.125)[1]) else "wrong",
          f"128 ties: device equals ties-away {int((y[:128] == ya).sum())}, ties-to-even {int((y[:128] == ye).sum())}")
bad = tot = 0
for sc in (0.1, 0.03, 0.3, 0.33333, 0.007, 0.02, 0.5, 0.125):
    xs = RNG.uniform(-12, 12, 2048).astype(np.float16); yd = quant_dev("qr", sc, xs)
    if yd is None: bad += 2048; tot += 2048; continue
    bad += int((yd != quant_model(xs, sc)[1]).sum()); tot += 2048
check("quantize -> dequantize, 8 scales x 2048 random inputs, equals the model", "ok", "ok" if bad == 0 else "wrong", f"{bad} mismatches of {tot}")

# ---- fp16 multiply and add: results that fall exactly between two fp16 values -----------------------------------------------
def tie_cases(op, n):
    """n pairs of fp16 values (a, b) whose exact result p lies exactly midway between two neighbouring fp16 values; p, lo, hi as float64"""
    g = np.random.default_rng(1); out = []
    while len(out) < n:
        m, k = g.integers(1024, 2048, (2, 400000)); e1, e2 = g.integers(-4, 4, (2, 400000)); s1, s2 = g.choice([-1, 1], (2, 400000))
        a = s1 * m * 2.0 ** (e1 - 10); b = s2 * k * 2.0 ** (e2 - 10)
        if op == "add": b = s2 * np.where(k % 8 == 0, 1, k % 8) * 2.0 ** (e1 - 11)     # b a multiple of half an ulp of the sum
        p = a * b if op == "mul" else a + b
        h = np.float16(p); lo = np.where(h.astype(np.float64) <= p, h, np.nextafter(h, np.float16(-np.inf))); hi = np.nextafter(lo, np.float16(np.inf))
        lo, hi = lo.astype(np.float64), hi.astype(np.float64); ok = (p == (lo + hi) / 2) & (np.abs(p) < 60000) & (np.abs(p) > 2 ** -13)
        out.extend(zip(a[ok], b[ok], p[ok], lo[ok], hi[ok]))
    return [np.array(c) for c in zip(*out[:n])]

def tie_dev(tag, op, a, b):
    sl = ''.join(f'        tensor<int32, [4]> {n}0 = const()[name = string("{n}0"), val = tensor<int32, [4]>([0,0,0,{o}])];\n' for n, o in (("a", 0), ("b", 32)))
    m = ('program(1.3)\n[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, '
         '{"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]\n{\n    func main<ios18>(tensor<fp16, [1,64,1,64]> x) {\n' + sl +
         '        tensor<int32, [4]> sz = const()[name = string("sz"), val = tensor<int32, [4]>([1,64,1,32])];\n'
         '        tensor<fp16, [1,64,1,32]> a = slice_by_size(x = x, begin = a0, size = sz)[name = string("a")];\n'
         '        tensor<fp16, [1,64,1,32]> b = slice_by_size(x = x, begin = b0, size = sz)[name = string("b")];\n'
         f'        tensor<fp16, [1,64,1,32]> y = {op}(x = a, y = b)[name = string("y")];\n    }} -> (y);\n}}\n')
    open(f"{W}/{tag}.mil", "w").write(m)
    x = np.concatenate([a.reshape(64, 32), b.reshape(64, 32)], axis=1).astype("<f2")           # [64][64]: a, then b; rows are 128 B (a multiple of 64)
    txt = ane(tag, f"{W}/{tag}.mil", 64 * 128, 64 * 64, inb=x.tobytes())
    return np.frombuffer(open(f"{W}/{tag}.out", "rb").read(), "<f2").astype(np.float64) if "eval=1" in txt else None

print("fp16 multiply and add: 2048 results exactly midway between two fp16 values (a one-op program each)")
for op in ("mul", "add"):
    a, b, p, lo, hi = tie_cases(op, 2048); y = tie_dev(f"tie_{op}", op, a, b)
    if y is None: check(f"fp16 {op}, exact ties: half away from zero", "ok", "fail"); continue
    away = np.where(p > 0, hi, lo); even = np.where((lo.astype(np.float16).view(np.uint16) & 1) == 0, lo, hi)
    check(f"fp16 {op}, exact ties: half away from zero", "ok", "ok" if np.array_equal(y, away) else "wrong",
          f"device equals ties-away {int((y == away).sum())}, ties-to-even {int((y == even).sum())} of {len(p)}")

# ---- programs recorded on macOS 15.8.3: what the device wrote then, against now -------------------------------------------
print("conv programs recorded on macOS 15.8.3 (test/device/macos15.8.3: MIL, weights, input, and the output surface the device wrote)")
G = f"{ROOT}/mlir/test/device/macos15.8.3"
for name, wb, xin, io, nb in (("V1_fp16W_castIO", "weight_fp16.bin", "x_f32.bin", "f32", 8192), ("V2_fp16W_fp16IO", "weight_fp16.bin", "x_f16.bin", "f16", 4096),
                              ("N1_int8_cast_mul", "weight_int8.bin", "x_f16.bin", "f16", 4096), ("N2_uint8_cast_sub_mul", "weight_uint8.bin", "x_f16.bin", "f16", 4096)):
    txt = ane(f"g_{name}", f"{G}/{name}.mil", nb, nb, io=io, weights=f"{G}/{wb}", inb=open(f"{G}/{xin}", "rb").read())
    old = open(f"{G}/{name}.out", "rb").read(); new = open(f"{W}/g_{name}.out", "rb").read() if "eval=1" in txt else b""
    bad = sum(a != b for a, b in zip(new, old)) + abs(len(new) - len(old))
    check(f"{name}: output surface equals the one the 15.8.3 device wrote", "ok", "ok" if bad == 0 else "wrong", f"{len(old)} bytes, {bad} differ")

# ---- a bundle of several weight files --------------------------------------------------------------------------------------
print("weight bundle of several files (two 1x1 convs 64 -> 64 -> 64, SP=64, one file per weight)")
def twoconv(tag, n1, n2, IC=64, SP=64):
    ws = [(RNG.uniform(-.5, .5, (IC, IC)) * 0.2).astype("<f2") for _ in (0, 1)]
    d = f"{W}/{tag}.w"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    for n, w in zip((n1, n2), ws): open(f"{d}/{n}.bin", "wb").write(blob(w.tobytes()))
    S2 = f"[{IC}, {IC}, 1, 1]"; m = HDR % dict(T="fp16", IC=IC, SP=SP)
    for i, n in enumerate((n1, n2), 1):
        m += f'        tensor<fp16, {S2}> W{i} = const()[name = string("W{i}"), val = tensor<fp16, {S2}>(BLOBFILE(path = string("@model_path/weights/{n}.bin"), offset = uint64(64)))];\n'
    cv = "conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W%d, x = %s)"
    m += f'        tensor<fp16, [1, {IC}, 1, {SP}]> y1 = ' + cv % (1, "x") + '[name = string("conv1")];\n'
    m += f'        tensor<fp16, [1, {IC}, 1, {SP}]> y = ' + cv % (2, "y1") + '[name = string("conv2")];\n    } -> (y);\n}\n'
    open(f"{W}/{tag}.mil", "w").write(m)
    txt = ane(tag, f"{W}/{tag}.mil", IC * SP * 2, IC * SP * 2, weights=d)
    enum = (re.search(r"WEIGHTS enum=(\S+)", txt) or [0, ""])[1].split(",")
    def ref():
        xi = np.frombuffer(open(f"{W}/{tag}.in", "rb").read(), "<f2").astype(np.float64).reshape(IC, SP)
        yo = np.frombuffer(open(f"{W}/{tag}.out", "rb").read(), "<f2").astype(np.float64).reshape(IC, SP)
        r = ws[1].astype(np.float64) @ (ws[0].astype(np.float64) @ xi); return float(np.abs(r).max()), float(np.abs(yo - r).max())
    return status(txt, ref), enum, "verifyBundleAtPath" in txt

R = random.Random(11); res = []
for t in range(80):
    n1, n2 = ("".join(R.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(R.randint(2, 6))) for _ in (0, 1))
    if n1 == n2: continue
    (st, det), enum, vb = twoconv(f"b{t}", n1, n2)
    res.append((len(enum) == 2 and enum[-1] == sorted(enum)[-1], st, vb))
    if sum(p for p, _, _ in res) >= 6 and sum(not p for p, _, _ in res) >= 6: break
agree = sum(((st == "ok") if p else (st == "fail" and vb)) for p, st, vb in res)
check(f"bundle passes iff its last-enumerated file is the one whose name sorts last ({len(res)} name sets)", "ok", "ok" if agree == len(res) else "wrong",
      f"predicted pass {sum(p for p, _, _ in res)} / fail {sum(not p for p, _, _ in res)}; rule right in {agree}; failures name verifyBundleAtPath in {sum(vb for _, _, vb in res)}")

print(f"\n{N[0]} checks, {len(DIFFS)} differ from the last run, {time.time() - t0:.0f} s")
for d in DIFFS: print("  DIFFERS:", d)
sys.exit(len(DIFFS))

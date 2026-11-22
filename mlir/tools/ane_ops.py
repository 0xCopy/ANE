#!/usr/bin/env python3
"""Which MIL operations does this Mac's Neural Engine compile and compute correctly?

usage: ane_ops.py [--only REGEX_ON_CASE_NAME] [--modules REGEX_ON_FILE] [--jobs N] [--work DIR] [--out FILE.jsonl] [--keep-mil DIR]
                  [--write-baseline FILE.json | --baseline FILE.json] [--skip REGEX | --all]
  --write-baseline  records every case's status (ok | wrong | fail | build) as the expected result on this OS and chip
  --baseline        compares this run with such a file: prints `DIFFERS` per case whose status changed, and exits with the number of them
  env ANE_OPS_WEIGHT_SALT=STR   changes every case's random weights (to ask whether an outcome depends on the weight values)
  --via-dialect     for every case that is ok: MIL -> `ane` ops (tools/ane_dialect.py) -> mlir-opt verifies them against dialect/ane.irdl.mlir ->
                    MIL again -> the device, and the output bytes must equal those of the original MIL.  Prints `dialect:` lines for the cases that differ.
  --skip / --all    cases left out by default (SKIP below: they kill the compiler service or take minutes); --all runs them too
needs the coremltools venv (~/.local/share/mlir-ane/ct-venv: pip install coremltools numpy), Xcode's coremlcompiler, tools/ane_run.m

For every case in CASES: build the program with coremltools' MIL builder, convert it (no optimisation passes, iOS18 opset, fp16),
compile it with coremlcompiler (that writes the model.mil the ANE compiler is fed in real use), run Core ML on the CPU for the
reference, then run the same model.mil on the ANE through the private in-memory path (tools/ane_run.m) and compare.

One JSON line per case: status ok | wrong | fail | build, the error class the ANE compiler gave (InvalidMILProgram, ...), the
largest reference value and the largest difference, and the time per evaluation.
"""
import argparse, glob, json, multiprocessing as mp, os, re, shutil, subprocess, sys, time, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
XSDK = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
COREMLC = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/coremlcompiler"
SHAPE = (1, 64, 1, 32)          # channel-first with a 64-byte row: no padding question for fp16


# ---------------------------------------------------------------------------------------------------------------- cases
class Case:
    def __init__(self, name, family, inputs, fn, kinds=None, tol=3e-2, note="", ref=None, skip=None, check=None):
        """ref: optional numpy reference, called with the input arrays (float64 for fp16/fp32 inputs, int32 for index inputs) in input order;
        returns an array or a list of arrays, one per output.  Without it the references are Core ML on the CPU in fp32 and in fp16 (see prepare).
        skip: optional, called like ref; returns a boolean array (output 0's shape) of elements to leave out of the comparison (inputs where the
        program's own rounding legitimately decides, e.g. a quotient that sits on an integer).
        check: optional, called with the device outputs (a list of flat float64 arrays) and the inputs; returns None, or a message when a property of the
        output fails.  With tol=None the numeric comparison is skipped and the check alone decides (for ops whose output is not a function of the inputs)."""
        self.name, self.family, self.fn, self.kinds, self.tol, self.note, self.ref, self.skip, self.check = name, family, fn, kinds or {}, tol, note, ref, skip, check
        self.inputs = [(i[0], tuple(i[1]), i[2] if len(i) > 2 else "fp16") for i in inputs]

CASES, CASES32 = [], []
RNGS = []          # the shared RandomState of each registration (fp16 program, fp32 twin): reseeded from the case name before every build, see _prepare
def case(name, family, inputs, fn, **kw): CASES.append(Case(name, family, inputs, fn, **kw))
def case32(name, family, inputs, fn, **kw): CASES32.append(Case(name, family, inputs, fn, **kw))
X = [("x", SHAPE)]; XY = [("x", SHAPE), ("y", SHAPE)]

def define(mods=None):
    """(re)build CASES from tools/ane_cases/*.py (those whose file name matches the regex `mods`, if given), in file-name order;
    pool workers call this too, so the indices agree.  A module that raises is dropped whole with a warning on stderr.
    Every module is registered twice: once as written (fp16 tensors: the program the ANE runs) and once through a shim that widens every
    fp16 array or scalar handed to a builder op to fp32 (CASES32: the same program in fp32, for the Core ML CPU reference)."""
    import glob, importlib.util, types as _t
    from coremltools.converters.mil import Builder as mb
    class MB32:
        def __getattr__(self, op):
            f = getattr(mb, op)
            def wide(v):
                if isinstance(v, np.ndarray) and v.dtype == np.float16: return v.astype(np.float32)
                if isinstance(v, np.float16): return np.float32(v)
                if isinstance(v, (list, tuple)): return type(v)(wide(i) for i in v)
                return v
            return lambda *a, **k: f(*[wide(x) for x in a], **{kk: wide(v) for kk, v in k.items()})
    f16 = lambda a: np.asarray(a, np.float16)
    CASES.clear(); CASES32.clear(); RNGS.clear(); rs16, rs32 = np.random.RandomState(0), np.random.RandomState(0); RNGS.extend([rs16, rs32])
    for path in sorted(glob.glob(f"{HERE}/ane_cases/*.py")):
        base = os.path.basename(path)[:-3]
        if mods and not re.search(mods, base): continue
        n0, n1 = len(CASES), len(CASES32)
        try:
            sp = importlib.util.spec_from_file_location("ane_cases_" + base, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
            m.register(_t.SimpleNamespace(case=case, mb=mb, np=np, f16=f16, SHAPE=SHAPE, X=X, XY=XY, rs=rs16))
            try:
                m.register(_t.SimpleNamespace(case=case32, mb=MB32(), np=np, f16=lambda a: f16(a).astype(np.float32), SHAPE=SHAPE, X=X, XY=XY, rs=rs32))
            except Exception as e:                      # the fp32 twin is a convenience: without it the fp16 reference is used
                del CASES32[n1:]; print(f"warning: fp32 twin of module {base} dropped: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        except Exception as e:
            del CASES[n0:]; del CASES32[n1:]; print(f"warning: module {base} dropped: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)


# ----------------------------------------------------------------------------------------------------------------- data
def gen(rng, kind, shape, dtype):
    if kind.startswith("idxf:"):                                        # integer-valued fp16 indices (the ANE holds int32 tensors as fp16 anyway)
        return rng.randint(0, int(kind.split(":")[1]), shape).astype(np.float16 if dtype == "fp16" else np.float32)
    if dtype == "int32":
        n = int(kind.split(":")[1]) if kind.startswith("idx:") else 8
        return rng.randint(0, n, shape).astype(np.int32)
    if kind == "pos": a = rng.uniform(0.2, 4, shape)
    elif kind == "unit": a = rng.uniform(-0.9, 0.9, shape)
    elif kind == "gt1": a = rng.uniform(1.1, 4, shape)
    elif kind == "small": a = rng.uniform(-1, 1, shape)
    elif kind == "near1": a = rng.uniform(0.8, 1.25, shape)               # products of ~32 of these stay inside fp16
    elif kind in ("causal", "causal_inf", "causal_max"):                  # additive causal mask over the last two dims: 0 on and below the diagonal, a large negative above
        neg = {"causal": -1e4, "causal_inf": -np.inf, "causal_max": -65504.0}[kind]
        a = np.broadcast_to(np.where(np.tril(np.ones(shape[-2:], bool)), 0.0, neg), shape).copy()
    elif kind == "nz": a = rng.uniform(0.5, 3, shape) * rng.choice([-1, 1], shape)
    elif kind == "round": a = np.round(rng.uniform(-2, 2, shape))
    else: a = rng.uniform(-3, 3, shape)
    return a.astype(np.float16 if dtype == "fp16" else np.float32)


# -------------------------------------------------------------------------------------------- build / compile / reference
def drop_package(m):
    """coremltools keeps every converted model in a tmpXXXX.mlpackage under the system temp dir and removes it in MLModel.__del__, which does
    not run when a pool worker is terminated: 8000 of them (1.2 GB) piled up over a few sweeps.  Remove it as soon as the reference is computed."""
    pp = getattr(m, "package_path", None)
    if pp and os.path.isdir(pp) and os.path.basename(pp).startswith("tmp") and pp.endswith(".mlpackage"): shutil.rmtree(pp, ignore_errors=True)


def prepare(args):
    """worker: case index -> dict (build, compile, reference) ; runs in a pool"""
    made = []
    try: return _prepare(args, made)
    finally:
        for m in made: drop_package(m)


def _prepare(args, made):
    i, work = args
    c = CASES[i]; d = f"{work}/{c.name}"; r = {"name": c.name, "family": c.family, "dir": d}
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    try:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import Function, Program, types
        def build(cs, wide):
            for g in RNGS: g.seed(zlib.crc32(("weights:" + c.name + os.environ.get("ANE_OPS_WEIGHT_SALT", "")).encode()) % (2 ** 31))   # weights: a function of the case name (and the optional salt), not of the worker's history
            mt = {"fp16": types.fp32 if wide else types.fp16, "fp32": types.fp32, "int32": types.int32}
            prog = Program()
            with Function({n: mb.TensorSpec(shape=s, dtype=mt[dt]) for n, s, dt in cs.inputs}, opset_version=ct.target.iOS18) as fn:
                o = cs.fn(*[fn.inputs[n] for n, _, _ in cs.inputs]); fn.set_outputs(list(o) if isinstance(o, (list, tuple)) else [o])
            prog.add_function("main", fn)
            return ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT32 if wide else ct.precision.FLOAT16,
                              compute_units=ct.ComputeUnit.CPU_ONLY, pass_pipeline=ct.PassPipeline.EMPTY)
        m = build(c, False); made.append(m)
        m.save(f"{d}/m.mlpackage")
    except Exception as e:
        r["build"] = f"{type(e).__name__}: {str(e)[:240]}"; return r
    p = subprocess.run([COREMLC, "compile", f"{d}/m.mlpackage", f"{d}/c"], capture_output=True, text=True)
    mc = f"{d}/c/m.mlmodelc"
    if p.returncode or not os.path.exists(f"{mc}/model.mil"):
        r["build"] = "coremlcompiler: " + (p.stdout + p.stderr)[:240]; return r
    shutil.copy(f"{mc}/model.mil", f"{d}/model.mil")
    if os.path.isdir(f"{mc}/weights"): shutil.copytree(f"{mc}/weights", f"{d}/weights")
    meta = json.load(open(f"{mc}/metadata.json"))[0]
    dt_of = {"Float16": "fp16", "Float32": "fp32", "Int32": "int32"}
    rng = np.random.RandomState(zlib.crc32(c.name.encode()) % (2 ** 31))          # the same inputs in every run (str hash() is reseeded per process)
    feed, spec_in = {}, []
    for (n, s, dt), sch in zip(c.inputs, meta["inputSchema"]):
        a = gen(rng, c.kinds.get(n, "any"), s, dt); feed[sch["name"]] = a
        a.astype({"fp16": "<f2", "fp32": "<f4", "int32": "<i4"}[dt]).tofile(f"{d}/{n}.bin")
        spec_in.append({"name": sch["name"], "dtype": dt, "shape": list(s), "file": f"{n}.bin"})
    spec_out = [{"name": sch["name"], "dtype": dt_of[sch["dataType"]], "shape": [int(v) for v in re.findall(r"\d+", sch["shape"])], "file": f"out_{k}.bin"} for k, sch in enumerate(meta["outputSchema"])]
    spec = {"mil": "model.mil", "inputs": spec_in, "outputs": spec_out}
    if os.path.isdir(f"{d}/weights"): spec["weights"] = "weights"
    json.dump(spec, open(f"{d}/spec.json", "w"))
    try:
        ref = m.predict(feed)
        for k, so in enumerate(spec_out): np.save(f"{d}/ref_{k}.npy", np.asarray(ref[so["name"]]))
    except Exception as e:
        r["ref_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    c32 = next((x for x in CASES32 if x.name == c.name), None)
    if c32 is not None:                                                         # same program in fp32: accumulation and chain rounding are not fp16's
        try:
            m32 = build(c32, True); made.append(m32); feed32 = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in feed.items()}
            ref32 = m32.predict(feed32)
            for k, so in enumerate(spec_out): np.save(f"{d}/ref32_{k}.npy", np.asarray(ref32[so["name"]]))
        except Exception as e:
            r["ref32_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    if c.skip is not None:
        try: np.save(f"{d}/skip_0.npy", np.asarray(c.skip(*[feed[sch["name"]].astype(np.float64) if feed[sch["name"]].dtype != np.int32 else feed[sch["name"]] for sch in meta["inputSchema"]])).astype(bool))
        except Exception as e: r["skip_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    if c.ref is not None:
        try:
            refs = c.ref(*[feed[sch["name"]].astype(np.float64) if feed[sch["name"]].dtype != np.int32 else feed[sch["name"]] for sch in meta["inputSchema"]])
            refs = refs if isinstance(refs, (list, tuple)) else [refs]
            for k, a in enumerate(refs): np.save(f"{d}/refnp_{k}.npy", np.asarray(a))
        except Exception as e:
            r["refnp_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    shutil.rmtree(f"{d}/m.mlpackage", ignore_errors=True); shutil.rmtree(f"{d}/c", ignore_errors=True)
    r["ready"] = True; return r


def short_err(e):
    if not e: return ""
    m = re.search(r"err=\(\s*(\w+)\s*\)", e) or re.search(r"Code=\d+ \"([^\"]{0,90})", e)
    return m.group(1) if m else e[:100]


# Compile failures that say nothing about the program: the service connection broke (it crashed, usually because another client sent it a very long
# program), or the compiler could not allocate memory for its files (intermittent on big kernels: the same bundle gives it 4 times in 16 compiles).
TRANSIENT = re.compile(r"connection to service|Failed to allocate memory|Code=4097|XPC_ERROR")
RETRIES = 3


def run_one(r, runner, c):
    d = r["dir"]; spec = json.load(open(f"{d}/spec.json"))
    for attempt in range(RETRIES + 1):
        p = subprocess.run(["perl", "-e", "alarm 120; exec @ARGV", runner, d, "10"], capture_output=True, text=True)
        try: j = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception: r.update(status="fail", err=(p.stdout + p.stderr)[:160]); return r
        if j.get("compile") == "ok" or not TRANSIENT.search(j.get("err", "")) or attempt == RETRIES: break
        r["retries"] = attempt + 1; time.sleep(2.0 + 3.0 * attempt)             # give a restarted service time to come up
    r["compile"], r["eval"] = j.get("compile"), j.get("eval")
    if j.get("type_mismatch"): r["status"] = "fail"; r["err"] = j.get("err", "")[:120]; return r
    if j.get("compile") != "ok" or j.get("eval") != "ok":
        r["status"] = "fail"; r["err"] = short_err(j.get("err", "")); r["err_full"] = j.get("err", "")[:600]; return r
    r["ms_eval"] = round(j.get("ms_eval", 0), 4)
    r["layout"] = {"in": [(a["Batches"], a["Channels"], a["Depth"], a["Height"], a["Width"], a["RowStride"], a["Type"]) for a in j["inputs"]],
                   "out": [(a["Batches"], a["Channels"], a["Depth"], a["Height"], a["Width"], a["RowStride"], a["Type"]) for a in j["outputs"]]}
    if j.get("layout_err"): r["status"] = "layout"; r["err"] = j.get("err", "")[:160]; return r
    if c.tol is None:
        r["status"] = "ok"
        return finish_check(r, c, d, spec, j)
    names = ["np"] if os.path.exists(f"{d}/refnp_0.npy") else [n for n, f in (("fp32", "ref32_0.npy"), ("fp16", "ref_0.npy")) if os.path.exists(f"{d}/{f}")]
    if not names: r["status"] = "ok-unchecked"; return r
    fname = {"np": "refnp_{k}.npy", "fp32": "ref32_{k}.npy", "fp16": "ref_{k}.npy"}
    best = None
    for nm in names:
        worst_ref = worst_diff = 0.0; bad = False; size_bad = None
        for k, so in enumerate(spec["outputs"]):
            ref = np.load(f"{d}/" + fname[nm].format(k=k)).astype(np.float64).reshape(-1)
            ot = (j.get("out_types") or [so["dtype"]] * len(spec["outputs"]))[k]
            out = np.fromfile(f"{d}/{so['file']}", {"fp16": "<f2", "fp32": "<f4", "int32": "<i4", "uint32": "<u4", "int16": "<i2", "uint16": "<u2", "int8": "i1", "uint8": "u1"}[ot]).astype(np.float64).reshape(-1)
            if nm == names[0] and ot != so["dtype"]: r.setdefault("out_type_note", []).append(f"{so['dtype']}->{ot}")
            if out.size != ref.size: size_bad = f"output {k}: {out.size} values, reference {ref.size}"; break
            if k == 0 and os.path.exists(f"{d}/skip_0.npy"):
                keep = ~np.load(f"{d}/skip_0.npy").reshape(-1)
                if keep.size == ref.size: ref, out = np.where(keep, ref, 0.0), np.where(keep, out, 0.0)
            same_nf = (~np.isfinite(ref)) & (ref == out)                      # +-inf in both places
            fin = np.isfinite(ref) & np.isfinite(out)
            if np.any(np.isnan(ref) != np.isnan(out)) or np.any((~np.isfinite(ref) & ~same_nf & ~np.isnan(ref)) | (~np.isfinite(out) & ~np.isnan(out) & (ref != out))): bad = True
            mr = float(np.max(np.abs(ref[fin]))) if fin.any() else 0.0; md = float(np.max(np.abs(out[fin] - ref[fin]))) if fin.any() else 0.0
            worst_ref, worst_diff = max(worst_ref, mr), max(worst_diff, md)
        if size_bad: r["status"] = "wrong"; r["err"] = size_bad; return r
        ok = (not bad) and worst_diff <= c.tol * max(1.0, worst_ref)
        cand = (not ok, worst_diff / max(1.0, worst_ref), nm, worst_ref, worst_diff)
        if best is None or cand < best: best = cand
    notok, _, nm, mr, md = best
    r["maxref"], r["maxdiff"], r["ref"] = round(mr, 5), round(md, 5), nm
    r["status"] = "wrong" if notok else "ok"
    return finish_check(r, c, d, spec, j) if r["status"] == "ok" else r


MLIR_OPT = os.environ.get("MLIR_OPT", "/opt/homebrew/opt/llvm/bin/mlir-opt")
IRDL = os.path.join(HERE, "..", "dialect", "ane.irdl.mlir")


def split_top_commas(t):
    out, depth, cur = [], 0, ""
    for ch in t:
        depth += (ch in "([<{") - (ch in ")]>}")
        if ch == "," and depth == 0: out.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out


def via_dialect(r, runner, c):
    """MIL -> ane ops -> (mlir-opt checks them against the IRDL declaration) -> MIL -> device: same output bytes as the original MIL?  -> (verdict, detail)"""
    import ane_dialect
    d = r["dir"]; text = open(f"{d}/model.mil").read()
    try: mod = ane_dialect.mil_to_ane(c.name, text)
    except Exception as e: return "convert", f"{type(e).__name__}: {str(e)[:160]}"
    p = subprocess.run([MLIR_OPT, f"--irdl-file={IRDL}"], input=mod, capture_output=True, text=True)
    if p.returncode: return "verify", p.stderr.strip().splitlines()[0][:160] if p.stderr.strip() else "mlir-opt failed"
    try: back = ane_dialect.ane_to_mil(p.stdout)
    except Exception as e: return "to-mil", f"{type(e).__name__}: {str(e)[:160]}"
    if len(back) != 1: return "to-mil", f"{len(back)} programs"
    d2 = d + "_ane"; shutil.rmtree(d2, ignore_errors=True); os.makedirs(d2)
    spec = json.load(open(f"{d}/spec.json"))
    for f in os.listdir(d):
        if f == "weights" or (f.endswith(".bin") and not f.startswith("out_")): os.symlink(f"{d}/{f}", f"{d2}/{f}")
    # the converted program names its inputs a0, a1, ... (by position in the MIL's argument list) and its outputs v<N>; the spec names the original ones
    m0 = re.search(r"func main<\w+>\((.*?)\)\s*\{", text, re.S); orig_in = [x.rsplit(" ", 1)[-1] for x in split_top_commas(m0.group(1))]
    orig_out = [x.strip() for x in re.search(r"\}\s*->\s*\((.*?)\);", text).group(1).split(",")]
    new_out = [x.strip() for x in re.search(r"\}\s*->\s*\((.*?)\);", back[0][1]).group(1).split(",")]
    spec2 = dict(spec, inputs=[dict(i, name=f"a{orig_in.index(i['name'])}") for i in spec["inputs"]],
                 outputs=[dict(o, name=new_out[orig_out.index(o["name"])]) for o in spec["outputs"]])
    json.dump(spec2, open(f"{d2}/spec.json", "w"))
    open(f"{d2}/model.mil", "w").write(back[0][1])
    for attempt in range(RETRIES + 1):
        q = subprocess.run(["perl", "-e", "alarm 120; exec @ARGV", runner, d2, "10"], capture_output=True, text=True)
        try: j = json.loads(q.stdout.strip().splitlines()[-1])
        except Exception: return "device", (q.stdout + q.stderr)[:160]
        if j.get("compile") == "ok" or not TRANSIENT.search(j.get("err", "")) or attempt == RETRIES: break
        time.sleep(2.0 + 3.0 * attempt)
    if j.get("compile") != "ok" or j.get("eval") != "ok": return "device", short_err(j.get("err", ""))
    spec = json.load(open(f"{d}/spec.json")); bad = []
    for so in spec["outputs"]:
        a, b = open(f"{d}/{so['file']}", "rb").read(), open(f"{d2}/{so['file']}", "rb").read()
        if a != b: bad.append(f"{so['file']}: {sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))} of {len(a)} bytes differ")
    shutil.rmtree(d2, ignore_errors=True)
    return ("differs", "; ".join(bad)) if bad else ("same", "")


def finish_check(r, c, d, spec, j):
    """apply the case's output-property check, if any, to the device outputs"""
    if c.check is None: return r
    outs = []
    for k, so in enumerate(spec["outputs"]):
        ot = (j.get("out_types") or [so["dtype"]] * len(spec["outputs"]))[k]
        outs.append(np.fromfile(f"{d}/{so['file']}", {"fp16": "<f2", "fp32": "<f4", "int32": "<i4", "uint32": "<u4", "int16": "<i2", "uint16": "<u2", "int8": "i1", "uint8": "u1"}[ot]).astype(np.float64).reshape(-1))
    ins = [np.fromfile(f"{d}/{i['file']}", {"fp16": "<f2", "fp32": "<f4", "int32": "<i4"}[i["dtype"]]).astype(np.float64) for i in spec["inputs"]]
    try: msg = c.check(outs, ins)
    except Exception as e: msg = f"check raised {type(e).__name__}: {str(e)[:80]}"
    if msg: r["status"], r["err"] = "wrong", msg[:160]
    return r


# Cases that are left out of a default run (--all runs them).
#   * A sequential chain of ~2300 ops overflows the stack of ANECompilerService (the service dies and every compile in flight on the Mac fails):
#     chains of 736 steps (2208 ops) compile, 768 steps crash it.  The 1024-wide program is slow (minutes).
#   * Tensors of 64 MB and more: their inputs, outputs and references take 0.2 - 2 GB per case on disk (a full disk kills the whole run).
SKIP = r"^(rc_lim_chain_(768|1024|2048)|sh_chain_(768|832|896|1024)|sh_wide_1024|rc_lim_big_(64|256)mb|sh_relu_(64|128|512)mb_\w+)$"
MIN_FREE_GB = 3.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--skip", default=SKIP); ap.add_argument("--all", action="store_true"); ap.add_argument("--via-dialect", action="store_true"); ap.add_argument("--baseline"); ap.add_argument("--write-baseline")
    ap.add_argument("--work", default=os.path.join(__import__("tempfile").gettempdir(), "ane-ops-work")); ap.add_argument("--out", default=None)
    ap.add_argument("--keep-mil", default=None, help="copy every model.mil (and small weights) to this directory")
    ap.add_argument("--modules", default=None, help="regex on the file names (no .py) in tools/ane_cases to load; default all")
    a = ap.parse_args(); define(a.modules)
    os.makedirs(a.work, exist_ok=True); runner = f"{a.work}/ane_run"
    free = shutil.disk_usage(a.work).free / 2 ** 30
    if free < MIN_FREE_GB: sys.exit(f"only {free:.1f} GB free under {a.work}: need {MIN_FREE_GB:g} GB (a default run writes about 1 GB; --all several GB)")
    env = dict(os.environ, **({} if "SDKROOT" in os.environ else {"SDKROOT": XSDK}))
    q = subprocess.run(["xcrun", "clang", "-w", "-fobjc-arc", "-framework", "Foundation", "-framework", "IOSurface", "-o", runner, f"{HERE}/ane_run.m"], capture_output=True, text=True, env=env)
    if q.returncode: sys.exit("building ane_run failed:\n" + q.stderr[:1500])
    idx = [i for i, c in enumerate(CASES) if (not a.only or re.search(a.only, c.name)) and (a.all or not re.search(a.skip, c.name))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs, initializer=define, initargs=(a.modules,)) as pool: prepared = pool.map(prepare, [(i, a.work) for i in idx], chunksize=1)
    print(f"built {sum(1 for r in prepared if r.get('ready'))}/{len(prepared)} in {time.time() - t0:.0f} s", flush=True)
    out = open(a.out, "w") if a.out else None; res = []
    for i, r in zip(idx, prepared):
        c = CASES[i]
        if not r.get("ready"): r["status"] = "build"
        else:
            run_one(r, runner, c)
            if a.via_dialect and r["status"] == "ok": r["dialect"], r["dialect_detail"] = via_dialect(r, runner, c)
        if a.keep_mil and r.get("ready"):
            os.makedirs(a.keep_mil, exist_ok=True); shutil.copy(f"{r['dir']}/model.mil", f"{a.keep_mil}/{c.name}.mil")
        r.pop("dir", None); res.append(r)
        if out: out.write(json.dumps(r) + "\n"); out.flush()
        print(f"  {r['status']:12s} {c.family:9s} {c.name:26s} {r.get('err') or r.get('build') or ''} {('diff %.3g / %.3g' % (r['maxdiff'], r['maxref'])) if 'maxdiff' in r else ''}", flush=True)
    from collections import Counter
    print(dict(Counter(r["status"] for r in res)), f"{time.time() - t0:.0f} s")
    nd = 0
    if a.via_dialect:
        tally = Counter(r["dialect"] for r in res if "dialect" in r); nd = sum(v for k, v in tally.items() if k != "same")
        for r in res:
            if r.get("dialect") not in (None, "same"): print(f"  dialect: {r['name']}: {r['dialect']}: {r['dialect_detail']}")
        print(f"via the dialect: {dict(tally)} of {sum(tally.values())} ok programs")
    status = {r["name"]: r["status"] for r in res}
    if a.write_baseline:
        old = json.load(open(a.write_baseline)) if os.path.exists(a.write_baseline) and a.only else {}
        for n, v in status.items():
            if old.get(n) == "flaky": continue                      # a case marked flaky stays flaky until someone edits the file
            old[n] = v
        json.dump(dict(sorted(old.items())), open(a.write_baseline, "w"), indent=0); print(f"baseline of {len(old)} cases written to {a.write_baseline}")
    if a.baseline:
        base = json.load(open(a.baseline)); diffs = [n for n in sorted(status) if n in base and base[n] != status[n]]; new = [n for n in sorted(status) if n not in base]
        gone = [n for n in sorted(base) if n not in status and not a.only] if not a.only else []
        flaky = [n for n in diffs if base[n] == "flaky"]; diffs = [n for n in diffs if base[n] != "flaky"]
        for n in diffs: print(f"  DIFFERS {n}: baseline {base[n]}, now {status[n]}")
        for n in new: print(f"  new     {n}: {status[n]} (not in the baseline)")
        print(f"{len(status)} cases, {len(diffs)} differ from the baseline, {len(new)} new, {len(gone)} in the baseline but not run"
              + (f"; {sum(1 for n in status if base.get(n) == 'flaky')} are `flaky` in the baseline and not compared (their status has changed between runs, see README)" if any(v == "flaky" for v in base.values()) else ""))
        sys.exit(len(diffs) + nd)

if __name__ == "__main__":
    main()

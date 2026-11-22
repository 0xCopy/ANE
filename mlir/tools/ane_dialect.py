#!/usr/bin/env python3
"""The `ane` dialect (dialect/ane.irdl.mlir): MIL text <-> MLIR.

usage: ane_dialect.py irdl                 print the dialect declaration (IRDL)
       ane_dialect.py from-mil [PREFIX]    `=== name` separated MIL programs on stdin -> one module of ane ops
       ane_dialect.py to-mil               a module of ane ops, as mlir-opt prints it, on stdin -> `=== name` MIL programs

One MIL statement is one ane op.  MIL `const` statements of int, bool, string and shape type become attributes of the
op that uses them; scalars used as operands become ane.const, BLOBFILE constants ane.blob.  to-mil writes them back as
`const` statements, so MIL -> ane -> MIL -> ane is a fixed point (test/check_ane_dialect.mlir).

Two kinds of op.  The ones in OPS below are written by hand (the kernels of the repo, with notes from the device).  The
rest come from ops/ane_ops.json, the catalog tools/ane_catalog.py derives from a tools/ane_ops.py run: every MIL op that
compiled and ran correctly on the device, with the arguments it was seen with (operand or attribute, optional or not).
Ops the compiler refused are listed in the declaration, not declared.  Regenerate dialect/ane.irdl.mlir after the catalog
changes:  python3 tools/ane_dialect.py irdl > dialect/ane.irdl.mlir
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil2mlir import DT, split_top, parse_type, parse_lit, tt, fnum

# op: (operands in order; `name*` is variadic, `name?` optional; [(attribute, kind[, optional])]; what the declaration says about it)
OPS = {
    "mul": (["x", "y"], [], "elementwise, numpy broadcast; a scalar operand is an ane.const"),
    "add": (["x", "y"], [], "elementwise, numpy broadcast"),
    "sub": (["x", "y"], [], "elementwise, numpy broadcast"),
    "pow": (["x", "y"], [], "elementwise x^y, numpy broadcast"),
    "sigmoid": (["x"], [], "1 / (1 + exp(-x)); result type = operand type"),
    "softmax": (["x"], [("axis", "int")], "along `axis`; result type = operand type"),
    "reduce_sum": (["x"], [("axes", "ints"), ("keep_dims", "bool")], "sum over `axes`"),
    "reshape": (["x"], [("shape", "ints")], "row-major; `shape` equals the result shape"),
    "transpose": (["x"], [("perm", "ints")], "result dim i = operand dim perm[i]"),
    "slice_by_size": (["x"], [("begin", "ints"), ("size", "ints")],
                      "x[begin : begin + size]; how a dynamic kernel unpacks activations and weights from its one input"),
    "concat": (["values*"], [("axis", "int"), ("interleave", "bool")],
               "along `axis`; how a kernel packs several outputs into its one output.\n"
               "  // IRDL gives variadic operands one type, so each value is passed as tensor<*xT> (tensor.cast)"),
    "matmul": (["x", "y"], [("transpose_x", "bool"), ("transpose_y", "bool")],
               "batched over the leading dims.  Device (M3 Pro, README \"Device facts\"): a contracted dim of 16384 compiles, 16385 does not"),
    "conv": (["x", "weight", "bias?"], [("strides", "ints"), ("dilations", "ints"), ("pad", "ints"), ("pad_type", "str"), ("groups", "int")],
             "NCHW x, OIHW weight.  Every use in the repo is 1x1, valid, groups 1: a linear map over channels.\n"
             "  // Device (M3 Pro): 32000 input channels compile, 32512 is the most that do (macOS 27.2.1); 32000 -> 768 fails at SP 64, compiles at SP 256"),
    "cast": (["x"], [("dtype", "str")], "fp32 <-> fp16; fp32 -> fp16 rounds to nearest even and overflows to inf at 65520"),
    "quantize": (["input", "scale", "zero_point?"], [("output_dtype", "str"), ("axis", "int", True)],
                 "input * fp16(1 / scale), rounded half away from zero, saturated to [-128, 127]  (device, macOS 27.2.1: README \"Sharp corners\")"),
    "dequantize": (["input", "scale", "zero_point?"], [("axis", "int", True)],
                   "int8 * scale, the product rounded to fp16 with ties away from zero (device, macOS 27.2.1)"),
}
SAME = ("sigmoid", "softmax")  # result type = operand type
KIND = {"int": "#builtin.integer", "bool": "#builtin.integer", "ints": "#builtin.array", "str": "#builtin.string"}
# every element type the sweep programs use (mil2mlir's DT knows the few the repo's own kernels need)
XDT = dict(DT, uint8="ui8", int16="i16", uint16="ui16", uint32="ui32", int4="si4", uint4="ui4", uint1="ui1", uint2="ui2", uint6="ui6")
MILDT = {v: k for k, v in XDT.items()}
KINDX = {"float": "#builtin.float", "int": "#builtin.integer", "bool": "#builtin.integer", "str": "#builtin.string", "ints": "#builtin.array",
         "bools": "#builtin.array", "list2d": "#builtin.array"}


def xtype(t):
    t = t.strip()
    m = re.match(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>", t)
    if m:
        return XDT[m.group(1)], [int(x) for x in m.group(2).split(",") if x.strip()]
    if t in XDT:
        return XDT[t], None  # scalar
    if t == "string":
        return "string", None
    raise ValueError(t)


NUM = re.compile(r"-?0x[0-9a-fA-F]+(?:\.[0-9a-fA-F]*)?p[+-]?\d+|-?\d+\.?\d*(?:[eE][+-]?\d+)?|true|false|\[|\]")


def nested(body):
    """`[[1, 2], [3, 4]]` / `[[-0x1p-2]]` / `[true, false]` / a bare number (a splat) -> nested python lists of numbers"""
    toks = NUM.findall(body)
    pos = [0]

    def item():
        t = toks[pos[0]]
        pos[0] += 1
        if t == "[":
            out = []
            while toks[pos[0]] != "]":
                out.append(item())
            pos[0] += 1
            return out
        if t in ("true", "false"):
            return t == "true"
        if "0x" in t:
            return float.fromhex(t)
        return float(t) if any(c in t for c in ".eE") else int(t)
    return item()


def nested_text(v):
    """nested python lists -> MLIR dense text: exact decimal floats, integers, bools"""
    if isinstance(v, list):
        return "[" + ", ".join(nested_text(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return fnum(repr(float(v)))


def xlit(v):
    """MIL literal -> ('blob', dt, shape, path, off) | ('list', dt, nested list, shape) | ('scalar', dt, value) | ('bool', b) | ('str', s)"""
    v = v.strip()
    m = re.match(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\s*\((.*)\)$", v, re.S)
    if m:
        dt, shape, body = XDT[m.group(1)], [int(x) for x in m.group(2).split(",") if x.strip()], m.group(3).strip()
        b = re.match(r'BLOBFILE\(path\s*=\s*string\("([^"]*)"\)\s*,\s*offset\s*=\s*uint64\((\d+)\)\)$', body)
        if b:
            return ("blob", dt, shape, b.group(1), int(b.group(2)))
        return ("list", dt, nested(body), shape)
    m = re.match(r"(\w+)\((.*)\)$", v, re.S)
    k, body = m.group(1), m.group(2)
    if k == "string":
        return ("str", body.strip().strip('"'))
    if k == "bool":
        return ("bool", body.strip() == "true")
    if k in ("fp16", "fp32"):
        return ("scalar", XDT[k], fnum(body))
    return ("scalar", XDT[k], int(body))


CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ops", "ane_ops.json")
SPECIAL = ("const", "blob", "constexpr_affine_dequantize")
LEAD = ("x", "input", "values", "query")


def load_gen():
    """-> (GEN, rejected, deferred).  GEN: op -> schema for the catalog's working ops that OPS does not declare:
    operands [(name, optional, variadic)], attrs [(name, kind, optional)], segments (operandSegmentSizes needed), variadic_results, doc."""
    gen, rejected, deferred = {}, [], []
    if not os.path.exists(CATALOG):
        return gen, rejected, deferred
    for op, v in sorted(json.load(open(CATALOG))["ops"].items()):
        if op in OPS:
            continue
        if op in SPECIAL:
            deferred.append(op)
            continue
        if v["status"] not in ("supported", "sensitive"):
            rejected.append((op, v["status"]))
            continue
        operands, attrs = [], []
        for a, kinds in v["args"].items():
            ks, opt = set(kinds), v["present"].get(a, v["uses"]) < v["uses"]
            if ks & {"tensor", "blob", "qblob"}:
                operands.append((a, opt, False))
            elif "tuple" in ks:
                operands.append((a, opt, True))
            else:
                kind = ("bool" if ks <= {"bool"} else "bools" if ks <= {"bools"} else "str" if ks <= {"str"} else "ints" if ks <= {"ints"}
                        else "list2d" if ks <= {"tconst"} else "float" if ks <= {"float"} else "int" if ks <= {"int"} else "scalar")
                attrs.append((a, kind, opt))
        operands.sort(key=lambda o: (o[1] or o[2], o[0] not in LEAD, o[0]))
        attrs.sort()
        doc = "%s on the M3 Pro, macOS 27.2.1: %d ok, %d wrong, %d refused programs in tools/ane_ops.py" % (v["status"], v["ok"], v["wrong"], v["fail"])
        if v["failing_variants"]:
            doc += "; refused in " + ", ".join(v["failing_variants"][:4])
        gen[op] = {"operands": operands, "attrs": attrs, "segments": sum(1 for _, o, vr in operands if o or vr) >= 2,
                   "variadic_results": max(v["results"]) > 1, "doc": doc}
    return gen, rejected, deferred


GEN, REJECTED, DEFERRED = load_gen()


def hand_schema():
    out = {}
    for op, (operands, attrs, doc) in OPS.items():
        ops_ = [(o.rstrip("*?"), o.endswith("?"), o.endswith("*")) for o in operands]
        out[op] = {"operands": ops_, "attrs": [(a[0], a[1], len(a) > 2 and a[2]) for a in attrs],
                   "segments": sum(1 for _, o, v in ops_ if o or v) >= 2, "variadic_results": False}
    return out


SCHEMA = dict(hand_schema(), **GEN)

IRDL_HEAD = '''// The `ane` dialect: the MIL operations this repo sends to the Apple Neural Engine, one MLIR op per MIL op.
//
// Declared in IRDL, so the stock tools load it with no C++:  mlir-opt --irdl-file=ane.irdl.mlir file.mlir
// What IRDL checks: operand and result counts, that they are tensors, attribute names and kinds, and the type
// equalities noted per op.  What it cannot check (shapes, element types inside a tensor, device limits) is in the
// comments and is checked by running: tools/ane_dialect.py to-mil prints a module of these ops as MIL for the device,
// and the same MIL goes through tools/mil2mlir.py to linalg for the CPU (NEON through LLVM).
//
// A kernel is a func.func of ane ops.  Attributes hold what MIL writes as `const` statements of int, bool, string
// and shape type.  Operand order is fixed (MIL passes arguments by name; the names are in each op below).
// An optional operand may be absent; when an op has two or more optional or variadic operands, the attribute
// `operandSegmentSizes = array<i32: ...>` gives the size of each group, in declaration order.  Variadic operands and
// results share one type in IRDL, so they are passed and returned as tensor<*xT> (tensor.cast).  Attributes the
// declaration does not list are optional in MIL and are checked only by the round trip.
//
// Generated by `tools/ane_dialect.py irdl`; edit the table there.

irdl.dialect @ane {
  // A scalar constant (float or integer), as a rank-0 tensor.  MIL: `fp16 n = const()[val = fp16(v)]`; the attribute's type
  // is the MIL type (f16, f32, i32, i8, ui32, ...).
  irdl.operation @const {
    %vf = irdl.base "#builtin.float"
    %vi = irdl.base "#builtin.integer"
    %v = irdl.any_of(%vf, %vi)
    %r = irdl.base "!builtin.tensor"
    irdl.attributes { "value" = %v }
    irdl.results(r: %r)
  }

  // A small tensor constant written inline in the MIL text (a lookup table, ...): `dense<[...]> : tensor<..>`.  IRDL has no
  // constraint for dense attributes except `any`; the round trip checks the rest.
  irdl.operation @dense {
    %v = irdl.any
    %r = irdl.base "!builtin.tensor"
    irdl.attributes { "value" = %v }
    irdl.results(r: %r)
  }

  // Constant data in a weight file.  MIL: `const()[val = tensor<..>(BLOBFILE(path, offset))]`.
  // `offset` is the byte offset of the chunk header.  The device reads the data from the absolute data offset in the chunk
  // header (at +16).  Measured on M3 Pro: with 0 there, macOS 15.8.3 reported InvalidMILProgram (the benches' header, which
  // also lacks the payload size), and macOS 27.2.1 compiles and evaluates on the wrong bytes.  The payload size (at +8) is
  // not checked on 27.2.1: any value, 0 included.
  irdl.operation @blob {
    %p = irdl.base "#builtin.string"
    %o = irdl.base "#builtin.integer"
    %r = irdl.base "!builtin.tensor"
    irdl.attributes { "path" = %p, "offset" = %o }
    irdl.results(r: %r)
  }

  // int8 weights dequantized at compile time: (q - zero_point) * scale; the blob has the result's shape.
  // macOS 27.2.1, M3 Pro: compiles, and runs at fp16 speed, with the parameters in the [...] attribute list (how
  // ane_int8_bench.m and to-mil write it; int8 with zero_point 0 and 3, uint8 with 128); with the parameters as call
  // arguments it fails with InvalidMILProgram.  The 15.8.3 note "the compiler rejects this op" was measured with the argument
  // form only.  ane.blob (int8) + ane.cast + ane.mul also compiles, but runs 5.7x (256 channels) to 20x (384, 512) slower than
  // the fp16 conv on 27.2.1 (5.1-5.4x at every size on 15.8.3); constexpr_affine_dequantize runs at 0.97-1.13x of fp16.
  irdl.operation @constexpr_affine_dequantize {
    %p = irdl.base "#builtin.string"
    %o = irdl.base "#builtin.integer"
    %a = irdl.base "#builtin.integer"
    %z = irdl.base "#builtin.integer"
    %s = irdl.base "#builtin.float"
    %r = irdl.base "!builtin.tensor"
    irdl.attributes { "path" = %p, "offset" = %o, "axis" = %a, "zero_point" = %z, "scale" = %s }
    irdl.results(r: %r)
  }

  // The same MIL op in general form: the quantized data (int8 or uint8, an ane.blob), the scale and the zero point (an ane.blob
  // per channel, or an ane.const for one value) are operands, `axis` says which dimension the per-channel tensors run along.
  // The attribute form above is the int8, per-tensor case with the blob folded in.  Both write the MIL the device accepts
  // (parameters in the [...] list).  Measured on 27.2.1: axis = 0 compiles and matches the CPU (per-channel or per-tensor scale,
  // zero point 0, 5 or 128); axis = 1 is refused, also with a per-tensor scale.
  irdl.operation @affine_dequantize {
    %t0 = irdl.base "!builtin.tensor"
    %t1 = irdl.base "!builtin.tensor"
    %t2 = irdl.base "!builtin.tensor"
    %a = irdl.base "#builtin.integer"
    %r = irdl.base "!builtin.tensor"
    irdl.operands(quantized_data: %t0, scale: %t1, zero_point: %t2)
    irdl.attributes { "axis" = %a }
    irdl.results(r: %r)
  }
'''


def irdl():
    L = [IRDL_HEAD]
    for op, (operands, attrs, doc) in OPS.items():
        L.append(f"  // {doc}")
        L.append(f"  irdl.operation @{op} {{")
        names = []
        for i, o in enumerate(operands):
            c = "r" if (op in SAME and i == 0) else f"t{i}"  # one constraint value used twice means equal types
            if c != "r":
                L.append(f'    %{c} = irdl.base "!builtin.{"unranked_" if o.endswith("*") else ""}tensor"')
            names.append(f"{o.rstrip('*?')}: {'variadic ' if o.endswith('*') else 'optional ' if o.endswith('?') else ''}%{c}")
        L.append('    %r = irdl.base "!builtin.tensor"')
        decl = [(a[0], a[1]) for a in attrs if not (len(a) > 2 and a[2])]          # optional attributes are not declared (IRDL has no optional ones)
        for j, (a, k) in enumerate(decl):
            L.append(f'    %a{j} = irdl.base "{KIND[k]}"')
        L.append(f"    irdl.operands({', '.join(names)})")
        if decl:
            L.append("    irdl.attributes { " + ", ".join(f'"{a}" = %a{j}' for j, (a, k) in enumerate(decl)) + " }")
        L.append("    irdl.results(r: %r)\n  }\n")
    gen_irdl(L)
    print("\n".join(L) + "}")


def gen_irdl(L):
    L.append("  // ---- generated from ops/ane_ops.json: the MIL ops that compiled and ran correctly on the device ----\n")
    for op, g in GEN.items():
        L.append("  // " + g["doc"])
        L.append("  irdl.operation @%s {" % op)
        names = []
        for i, (a, opt, var) in enumerate(g["operands"]):
            L.append('    %%t%d = irdl.base "!builtin.%stensor"' % (i, "unranked_" if var else ""))
            names.append("%s: %s%%t%d" % (a, "variadic " if var else "optional " if opt else "", i))
        L.append('    %%r = irdl.base "!builtin.%stensor"' % ("unranked_" if g["variadic_results"] else ""))
        decl = [(a, k) for a, k, opt in g["attrs"] if not opt]
        for j, (a, k) in enumerate(decl):
            if k == "scalar":
                L.append('    %%a%df = irdl.base "#builtin.float"' % j)
                L.append('    %%a%di = irdl.base "#builtin.integer"' % j)
                L.append("    %%a%d = irdl.any_of(%%a%df, %%a%di)" % (j, j, j))
            else:
                L.append('    %%a%d = irdl.base "%s"' % (j, KINDX[k]))
        if names:
            L.append("    irdl.operands(%s)" % ", ".join(names))
        if decl:
            L.append("    irdl.attributes { " + ", ".join('"%s" = %%a%d' % (a, j) for j, (a, k) in enumerate(decl)) + " }")
        L.append("    irdl.results(r: %s%%r)\n  }\n" % ("variadic " if g["variadic_results"] else ""))
    if GEN or REJECTED:
        L.append("  // ---- not declared ----")
        L.append("  // Refused by the device compiler in every program tried (ops/ane_ops.md), so there is no op for them:")
        L.append("  //   " + " ".join(op for op, st in REJECTED if st == "unsupported"))
        dev = [op for op, st in REJECTED if st == "deviates"]
        if dev:
            L.append("  // Compiles but the output is not the op's: " + " ".join(dev) + " (random_uniform returns one fixed array for every seed, all values")
            L.append("  // within a sixteenth of the requested range; feed randomness in as an input).")
        L.append("  // Declared above by hand with the parameters in the [...] list: " + " ".join(DEFERRED))
        L.append("")


def mtype(dt, shape):
    return tt(dt, shape if shape is not None else [])


def mlir_float(txt, dt):
    """an MLIR float attribute's text -> a number: decimal, or the IEEE bit pattern in hex that the printer falls back to"""
    import struct
    txt = txt.strip()
    if txt.startswith(("0x", "-0x")):
        bits = int(txt, 16)
        return struct.unpack("<e", struct.pack("<H", bits))[0] if dt == "f16" else struct.unpack("<f", struct.pack("<I", bits))[0]
    return float(txt)


def attr_text(kind, lit):
    """MIL literal (xlit) -> MLIR attribute text"""
    if kind == "bool":
        return "true" if lit[1] else "false"
    if kind == "str":
        return f'"{lit[1]}"'
    if kind in ("int", "float", "scalar"):
        return f"{lit[2]} : {lit[1]}"
    return nested_text(lit[2])                          # ints, bools, list2d: nested lists


def mil_attr(kind, x, const):
    """MLIR attribute text -> the named MIL const that holds it (const(typ, val) emits the statement and returns its name)"""
    if kind == "bool":
        return const("bool", f"bool({x})")
    if kind == "str":
        return const("string", f"string({x})")
    if kind in ("int", "float", "scalar"):
        val, mt = x.rsplit(" : ", 1)
        mt = mt.strip()
        mil = MILDT[mt]
        if mt in ("f16", "f32"):
            return const(mil, f"{mil}({fnum(repr(mlir_float(val, mt)))})")
        return const(mil, f"{mil}({int(val)})")
    items = json.loads(x)
    if kind == "list2d":
        return const(f"tensor<int32, [{len(items)}, {len(items[0])}]>", f"tensor<int32, [{len(items)}, {len(items[0])}]>({json.dumps(items)})")
    dt = "bool" if kind == "bools" else "int32"
    return const(f"tensor<{dt}, [{len(items)}]>", f"tensor<{dt}, [{len(items)}]>({json.dumps(items)})")


def parse_stmt(line):
    """`T a, T b = op(k = v, ...)[k = v, ...];` -> ([(type, name)], op, {arg: text}, {attr: text}), or None for lines that are not statements"""
    line = line.strip()
    if " = " not in line or not line.endswith("];"):
        return None
    lhs, rhs = line.split(" = ", 1)
    m = re.match(r"(\w+)\(", rhs)
    if not m:
        return None
    i, depth, q = m.end(), 1, False
    while i < len(rhs) and depth:
        ch = rhs[i]
        if ch == '"':
            q = not q
        elif not q:
            depth += (ch == "(") - (ch == ")")
        i += 1
    args, rest = rhs[m.end():i - 1], rhs[i:]
    if not (rest.startswith("[") and rest.endswith("];")):
        return None

    def kv(txt):
        return dict((k.strip(), v.strip()) for k, v in (x.split("=", 1) for x in split_top(txt)))

    return [tuple(x.rsplit(" ", 1)) for x in split_top(lhs)], m.group(1), kv(args), kv(rest[1:-2])


def elem(t):
    return re.match(r"tensor<(?:[\d?]+x)*(\w+)>$", t).group(1)


def mil_to_ane(name, text):
    """one MIL program -> the text of one ane func.func"""
    m = re.search(r"func main<(\w+)>\((.*?)\)\s*\{", text)
    ver = re.search(r"program\(([\d.]+)\)", text).group(1)
    ty, env, body, args, n = {}, {}, [], [], [0]
    for a in split_top(m.group(2)):
        t, an = a.rsplit(" ", 1)
        ty[an] = xtype(t)
        args.append(an)
    rets = [r.strip() for r in re.search(r"\}\s*->\s*\((.*?)\);", text).group(1).split(",")]

    def blob_op(lit, res=None):
        if res is None:
            n[0] += 1
            res = f"b.{n[0]}"
        body.append(f'  %{res} = "ane.blob"() {{offset = {lit[4]} : i64, path = "{lit[3]}"}} : () -> {tt(lit[1], lit[2])}')
        ty[res] = (lit[1], lit[2])
        return f"%{res}", tt(lit[1], lit[2])

    def use(a):  # MIL argument -> SSA operand (scalar constants and inline blobs are materialised at their use)
        a = a.strip()
        if a in ty:
            return f"%{a}", mtype(*ty[a])
        lit = env[a] if a in env else xlit(a)
        if lit[0] == "scalar":
            n[0] += 1
            s = f"%k.{n[0]}"
            body.append(f'  {s} = "ane.const"() {{value = {lit[2]} : {lit[1]}}} : () -> tensor<{lit[1]}>')
            return s, f"tensor<{lit[1]}>"
        if lit[0] == "blob":
            return blob_op(lit)
        if lit[0] == "list":
            n[0] += 1
            s, T = f"%d.{n[0]}", tt(lit[1], lit[3])
            body.append(f'  {s} = "ane.dense"() {{value = dense<{nested_text(lit[2])}> : {T}}} : () -> {T}')
            return s, T
        raise AssertionError((a, lit))

    def static(a):
        a = a.strip()
        return env[a] if a in env else xlit(a)

    for line in text[m.end():].split("\n"):
        st = parse_stmt(line)
        if not st:
            continue
        results, op, a, at = st
        res = results[0][1]
        if op == "const":
            lit = xlit(at["val"])
            if lit[0] == "blob":
                blob_op(lit, res)
            else:
                env[res] = lit
            continue
        if op == "constexpr_affine_dequantize":                  # parameters in the [...] list, no operands
            dt, shape = xtype(results[0][0])
            q, sc, zp = xlit(at["quantized_data"]), xlit(at["scale"]), xlit(at["zero_point"])
            if q[1] == "i8" and sc[0] == "scalar" and zp[0] == "scalar":        # the int8, per-tensor case: the blob is folded into the op
                body.append(f'  %{res} = "ane.constexpr_affine_dequantize"() {{axis = {xlit(at["axis"])[2]} : i32, '
                            f'offset = {q[4]} : i64, path = "{q[3]}", scale = {sc[2]} : {dt}, '
                            f'zero_point = {zp[2]} : i8}} : () -> {mtype(dt, shape)}')
            else:                                                # general form: data, scale, zero point are operands
                ops3 = [use(at["quantized_data"]), use(at["scale"]), use(at["zero_point"])]
                body.append(f'  %{res} = "ane.affine_dequantize"({", ".join(x for x, _ in ops3)}) {{axis = {xlit(at["axis"])[2]} : i32}} : '
                            f'({", ".join(t for _, t in ops3)}) -> {mtype(dt, shape)}')
            ty[res] = (dt, shape)
            continue
        assert op in SCHEMA, f"MIL op `{op}` is not in the dialect (refused by the device compiler, or never tried): ops/ane_ops.md"
        sch = SCHEMA[op]
        known = {o for o, _, _ in sch["operands"]} | {x for x, _, _ in sch["attrs"]}
        assert set(a) <= known, f"{op}: arguments {sorted(set(a) - known)} are not in the dialect (the sweep never used them)"
        miss = {o for o, opt, v in sch["operands"] if not opt} | {x for x, k, opt in sch["attrs"] if not opt}
        assert miss <= set(a), f"{op}: required arguments missing: {sorted(miss - set(a))}"
        us, seg = [], []
        for o, opt, var in sch["operands"]:
            if o not in a:
                seg.append(0)
                continue
            if var:  # IRDL: variadic operands share one type, so each goes through an unranked cast
                vals = [use(v) for v in split_top(a[o].strip()[1:-1])]
                for i, (sv, t) in enumerate(vals):
                    body.append(f"  %{res}.{i} = tensor.cast {sv} : {t} to tensor<*x{elem(t)}>")
                    us.append((f"%{res}.{i}", f"tensor<*x{elem(t)}>"))
                seg.append(len(vals))
            else:
                us.append(use(a[o]))
                seg.append(1)
        ats = [f"{x} = {attr_text(k, static(a[x]))}" for x, k, opt in sorted(sch["attrs"]) if x in a]
        if sch["segments"]:
            ats.append("operandSegmentSizes = array<i32: " + ", ".join(map(str, seg)) + ">")
        ats = "{" + ", ".join(ats) + "} " if ats else ""
        operand_txt, type_txt = ", ".join(sv for sv, _ in us), ", ".join(t for _, t in us)
        types = [xtype(t) for t, _ in results]
        if len(results) == 1 and not sch["variadic_results"]:
            body.append(f'  %{res} = "ane.{op}"({operand_txt}) {ats}: ({type_txt}) -> {mtype(*types[0])}')
        else:                                                      # results share one unranked type; each is cast back to its own
            g = f"grp.{res}"
            u = "tensor<*x%s>" % types[0][0]
            body.append(f'  %{g}:{len(results)} = "ane.{op}"({operand_txt}) {ats}: ({type_txt}) -> ({", ".join([u] * len(results))})')
            for k, (_, rn) in enumerate(results):
                body.append(f"  %{rn} = tensor.cast %{g}#{k} : {u} to {mtype(*types[k])}")
        for (_, rn), t in zip(results, types):
            ty[rn] = t
    sig = ", ".join(f"%{x}: {mtype(*ty[x])}" for x in args)
    rt = ", ".join(mtype(*ty[r]) for r in rets)
    return "\n".join([f'func.func @{name}({sig}) -> ({rt}) attributes {{ane.program = "{ver}", ane.target = "{m.group(1)}"}} {{'] + body
                     + [f"  return {', '.join('%' + r for r in rets)} : {rt}\n}}\n"])


def from_mil(prefix):
    out = []
    for chunk in re.split(r"^=== ", sys.stdin.read(), flags=re.M)[1:]:
        name, text = chunk.split("\n", 1)
        out.append(mil_to_ane(prefix + name.strip().replace("gen_", "").replace("mil_", ""), text))
    sys.stdout.write("\n".join(out))


BUILD_INFO = ('[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, '
              '{"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]')


def mil_type(t):
    body = re.match(r"tensor<(.*)>$", t.strip()).group(1).split("x")
    dt, shape = MILDT[body[-1]], body[:-1]
    return f"tensor<{dt}, [{', '.join(shape)}]>" if shape else dt


INLINE = {"constexpr_affine_dequantize", "constexpr_lut_to_dense", "constexpr_sparse_to_dense", "constexpr_blockwise_shift_scale"}
MIL_OF = {"affine_dequantize": "constexpr_affine_dequantize"}          # the dialect's op name -> the MIL op it writes


def dense_value(txt):
    """`dense<...> : tensor<2x1xf16>` -> nested python lists (a splat is expanded to the whole shape)"""
    m = re.match(r"dense<(.*)> : tensor<([\dx]+)x\w+>$", txt.strip(), re.S)
    val, dims = nested(m.group(1)), [int(d) for d in m.group(2).split("x")]
    if not isinstance(val, list):
        for d in reversed(dims):
            val = [val] * d
    return val, dims


def ane_to_mil(text):
    """a module of ane ops, as mlir-opt prints it -> [(name, MIL program text)]"""
    progs = []
    for fm in re.finditer(r'func\.func @(\w+)\((.*?)\) -> (.*?) attributes \{(.*?)\} \{\n(.*?)\n  \}', text, re.S):
        name, sig, _, fattrs, body = fm.groups()
        fa = dict(re.findall(r'(ane\.\w+) = "([^"]*)"', fattrs))
        rtypes = dict(((g, int(i)), t) for g, i, t in re.findall(r"%[\w.]+ = tensor\.cast (%[\w.]+)#(\d+) : tensor<\*x\w+> to (tensor<.*?>)\s*$", body, re.M))
        # ---- pass 1: statements, and who uses each constant
        stmts, users = [], {}
        for line in body.split("\n"):
            line = line.strip()
            r = re.match(r"return (.*?) :", line)
            if r:
                stmts.append(("return", [x.strip() for x in r.group(1).split(",")]))
                for x in stmts[-1][1]:
                    users.setdefault(x, []).append("return")
                continue
            cm = re.match(r"(%[\w.]+) = tensor\.cast (%[\w.#]+) :", line)
            if cm:
                stmts.append(("cast", cm.group(1), cm.group(2).strip()))
                users.setdefault(cm.group(2).strip(), []).append("cast")
                continue
            om = re.match(r'(%[\w.]+)(?::(\d+))? = "ane\.(\w+)"\((.*?)\) (?:\{(.*)\} )?: \((.*?)\) -> (.*)$', line)
            if not om:
                continue
            res, nres, op, opnds, atxt, _, rty = om.groups()
            at = dict((x.split(" = ", 1)[0].strip(), x.split(" = ", 1)[1].strip()) for x in split_top(atxt or ""))
            ins = [x.strip() for x in opnds.split(",")] if opnds.strip() else []
            stmts.append(("op", res, nres, op, ins, at, rty))
            for x in ins:
                users.setdefault(x, []).append(MIL_OF.get(op, op))
        inline_only = {r for r, us in users.items() if us and all(u in INLINE for u in us)}
        # ---- pass 2: MIL text
        nm, lits, L, k = {}, {}, [], [0]

        def v(sv):
            return nm[sv.strip()]

        def const(typ, val):
            k[0] += 1
            L.append(f'        {typ} k{k[0]} = const()[name = string("k{k[0]}"), val = {val}];')
            return f"k{k[0]}"

        def operand(sv, inline):
            sv = sv.strip()
            return lits[sv] if inline and sv in lits else v(sv)

        params = []
        for a in split_top(sig):
            sv, t = a.split(":", 1)
            nm[sv.strip()] = "a" + sv.strip().lstrip("%arg")
            params.append(f"{mil_type(t)} {nm[sv.strip()]}")
        rets = []
        for st in stmts:
            if st[0] == "return":
                rets = [v(x) for x in st[1]]
                continue
            if st[0] == "cast":
                nm[st[1]] = v(st[2])
                continue
            _, res, nres, op, ins, at, rty = st
            num = lambda x: x.rsplit(" : ", 1)[0]
            if nres:                                               # several results: one MIL name each, typed by the casts that follow
                names = [f"v{res.lstrip('%')}_{i}" for i in range(int(nres))]
                for i, nme in enumerate(names):
                    nm[f"{res}#{i}"] = nme
                lhs = ", ".join(f"{mil_type(rtypes[(res, i)])} {nme}" for i, nme in enumerate(names))
                T, n = mil_type(rtypes[(res, 0)]), names[0]
            else:
                T, n = mil_type(rty), "v" + res.lstrip("%")
                lhs = f"{T} {n}"
                nm[res] = n
            emit = res not in inline_only                          # a constant only the constexpr_* ops use is written inline, not as a statement
            if op == "const":
                x, mt = at["value"].rsplit(" : ", 1)
                val = fnum(repr(mlir_float(x, mt))) if mt in ("f16", "f32") else str(int(x))
                lits[res] = f"{T}({val})"
                if emit:
                    L.append(f'        {lhs} = const()[name = string("{n}"), val = {lits[res]}];')
            elif op == "dense":
                dv, dims = dense_value(at["value"])
                lits[res] = f"{T}({nested_text(dv)})"
                if emit:
                    L.append(f'        {lhs} = const()[name = string("{n}"), val = {lits[res]}];')
            elif op == "blob":
                lits[res] = f"{T}(BLOBFILE(path = string({at['path']}), offset = uint64({num(at['offset'])})))"
                if emit:
                    L.append(f'        {lhs} = const()[name = string("{n}"), val = {lits[res]}];')
            elif op == "constexpr_affine_dequantize":              # the attribute form: an int8 blob and two scalars, folded into the op
                L.append(f'        {lhs} = constexpr_affine_dequantize()[axis = int32({num(at["axis"])}), name = string("{n}"), '
                         f'quantized_data = {T.replace("fp16", "int8")}(BLOBFILE(path = string({at["path"]}), '
                         f'offset = uint64({num(at["offset"])}))), scale = fp16({fnum(repr(mlir_float(num(at["scale"]), "f16")))}), '
                         f'zero_point = int8({num(at["zero_point"])})];')
            elif op == "affine_dequantize":                        # the general form: parameters in the [...] list, as the device wants them
                d, sc, zp = (operand(x, True) for x in ins)
                L.append(f'        {lhs} = constexpr_affine_dequantize()[axis = int32({num(at["axis"])}), name = string("{n}"), '
                         f'quantized_data = {d}, scale = {sc}, zero_point = {zp}];')
            else:
                sch = SCHEMA[op]
                if sch["segments"]:
                    sizes = [int(x) for x in re.search(r"array<i32: ([\d, ]+)>", at["operandSegmentSizes"]).group(1).split(",")]
                else:
                    extra = len(ins) - sum(1 for _, o, vr in sch["operands"] if not (o or vr))
                    sizes = [extra if (o or vr) else 1 for _, o, vr in sch["operands"]]
                args, pos, inl = {}, 0, op in INLINE
                for (o, opt, var), cnt in zip(sch["operands"], sizes):
                    vals = ins[pos:pos + cnt]
                    pos += cnt
                    if not cnt:
                        continue
                    args[o] = "(" + ", ".join(operand(x, inl) for x in vals) + ")" if var else operand(vals[0], inl)
                for a, kind, opt in sch["attrs"]:
                    if a in at:
                        args[a] = mil_attr(kind, at[a], const)
                L.append(f'        {lhs} = {op}({", ".join(f"{a} = {args[a]}" for a in sorted(args))})[name = string("{n}")];')
        progs.append((name, f"program({fa['ane.program']})\n{BUILD_INFO}\n{{\n    func main<{fa['ane.target']}>({', '.join(params)}) {{\n"
                      + "\n".join(L) + f"\n    }} -> ({', '.join(rets)});\n}}"))
    return progs


def to_mil():
    for name, prog in ane_to_mil(sys.stdin.read()):
        print(f"=== {name}\n{prog}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "irdl":
        irdl()
    elif cmd == "from-mil":
        from_mil(sys.argv[2] if len(sys.argv) > 2 else "")
    elif cmd == "to-mil":
        to_mil()
    else:
        sys.exit(__doc__)

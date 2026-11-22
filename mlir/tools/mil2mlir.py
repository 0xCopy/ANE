#!/usr/bin/env python3
"""MIL text (as emitted by the ANE repo's generators) -> literal MLIR (linalg/tensor on tensors).

One MIL statement becomes one MLIR op group, in the same order, with the same value names.
Reads `=== name` separated MIL programs on stdin, writes one module on stdout.

usage: mil2mlir.py [--wide] [PREFIX]
  --wide   sums of products (conv, matmul) and reduce_sum accumulate in f32 and round to f16 once at the end.
           Without it every intermediate is rounded to f16, which is what the MIL types say.
"""
import re, sys

WIDE = False
DT = {"fp16": "f16", "fp32": "f32", "int8": "i8", "int32": "i32", "bool": "i1"}


def split_top(s, sep=","):
    out, depth, cur, q = [], 0, "", False
    for ch in s:
        if ch == '"':
            q = not q
        if not q:
            if ch in "([{<":
                depth += 1
            elif ch in ")]}>":
                depth -= 1
            elif ch == sep and depth == 0:
                out.append(cur.strip()); cur = ""; continue
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_type(t):
    t = t.strip()
    m = re.match(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>", t)
    if m:
        return DT[m.group(1)], [int(x) for x in m.group(2).split(",") if x.strip()]
    if t in DT:
        return DT[t], None  # scalar
    if t == "string":
        return "string", None
    raise ValueError(t)


def tt(dt, shape):
    return "tensor<" + "".join(f"{d}x" for d in shape) + dt + ">"


def fnum(txt):
    txt = txt.strip()
    v = float.fromhex(txt) if "0x" in txt else float(txt)
    r = repr(v)
    if "e" in r and "." not in r:
        r = r.replace("e", ".0e")
    return r if v != int(v) or abs(v) > 1e15 else f"{v:.1f}"


def parse_lit(v):
    """Literal value -> ('blob', dt, shape, path, off) | python value."""
    v = v.strip()
    m = re.match(r"tensor<\s*(\w+)\s*,\s*\[([^\]]*)\]\s*>\s*\((.*)\)$", v, re.S)
    if m:
        dt, shape, body = DT[m.group(1)], [int(x) for x in m.group(2).split(",") if x.strip()], m.group(3).strip()
        b = re.match(r'BLOBFILE\(path\s*=\s*string\("([^"]*)"\)\s*,\s*offset\s*=\s*uint64\((\d+)\)\)', body)
        if b:
            return ("blob", dt, shape, b.group(1), int(b.group(2)))
        return [int(x) for x in body.strip("[]").split(",")]
    m = re.match(r"(\w+)\((.*)\)$", v, re.S)
    k, body = m.group(1), m.group(2)
    if k in ("int32", "int8", "uint64"):
        return int(body)
    if k == "bool":
        return body == "true"
    if k == "string":
        return body.strip('"')
    if k in ("fp16", "fp32"):
        return ("fscalar", DT[k], fnum(body))
    raise ValueError(v)


class Fn:
    def __init__(self, name):
        self.name, self.args, self.body, self.env, self.ty, self.n = name, [], [], {}, {}, 0
        self.blobs = []

    def tmp(self, base, suf):
        return f"%{base}.{suf}"

    def emit(self, s):
        self.body.append("  " + s)

    def wide_begin(self, res, dt, shape):
        """Accumulator type and name for a sum of products: f32 under --wide (the named op widens its f16 inputs)."""
        if WIDE and dt == "f16":
            return "f32", res + ".w", tt("f32", shape)
        return dt, res, tt(dt, shape)

    def wide_end(self, res, dt, shape):
        if WIDE and dt == "f16":
            self.elementwise(res, dt, shape, [(f"%{res}.w", shape)], ["%r = arith.truncf %a0 : f32 to f16"], in_dts=["f32"])

    def val(self, a):
        """MIL arg -> static python value (looked up or inline literal)."""
        a = a.strip()
        if a in self.env:
            return self.env[a]
        return parse_lit(a)

    def empty(self, name, dt, shape, zero=False):
        e = self.tmp(name, "e")
        self.emit(f"{e} = tensor.empty() : {tt(dt, shape)}")
        if not zero:
            return e
        z, f = self.tmp(name, "z"), self.tmp(name, "f")
        self.emit(f"{z} = arith.constant 0.0 : {dt}")
        self.emit(f"{f} = linalg.fill ins({z} : {dt}) outs({e} : {tt(dt, shape)}) -> {tt(dt, shape)}")
        return f

    def operand(self, a, name, i, dt):
        """-> (ssa, shape|None). Float scalar consts are materialised next to their use."""
        a = a.strip()
        if a in self.ty:
            return f"%{a}", self.ty[a][1]
        v = self.val(a)
        assert isinstance(v, tuple) and v[0] == "fscalar", (a, v)
        c = f"%{a}" if a in self.env else self.tmp(name, f"c{i}")
        if c not in self.defined_scalars:
            self.emit(f"{c} = arith.constant {v[2]} : {v[1]}")
            self.defined_scalars.add(c)
        return c, None

    def elementwise(self, name, dt, shape, ins, body_ops, in_dts=None):
        """ins: [(ssa, shape|None)], broadcast numpy-style into `shape`."""
        r = len(shape)
        dims = ", ".join(f"d{i}" for i in range(r))
        maps, tys = [], []
        for k, (s, sh) in enumerate(ins):
            idt = in_dts[k] if in_dts else dt
            if sh is None:
                maps.append(f"affine_map<({dims}) -> ()>"); tys.append(idt); continue
            off = r - len(sh)
            exprs = []
            for j, d in enumerate(sh):
                assert d == shape[off + j] or d == 1, (name, sh, shape)
                exprs.append("0" if (d == 1 and shape[off + j] != 1) else f"d{off + j}")
            maps.append(f"affine_map<({dims}) -> ({', '.join(exprs)})>"); tys.append(tt(idt, sh))
        maps.append(f"affine_map<({dims}) -> ({dims})>")
        e = self.empty(name, dt, shape)
        its = ", ".join('"parallel"' for _ in range(r))
        bargs = ", ".join(f"%_a{k} : {in_dts[k] if in_dts else dt}" for k in range(len(ins)))
        self.emit(f"%{name} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = [{its}]}}")
        self.emit(f"    ins({', '.join(s for s, _ in ins)} : {', '.join(tys)}) outs({e} : {tt(dt, shape)}) {{")
        self.emit(f"  ^bb0({bargs}, %_o : {dt}):")
        for l in body_ops:
            self.emit("    " + re.sub(r"%(\w+)", r"%_\1", l))
        self.emit(f"    linalg.yield %_r : {dt}")
        self.emit(f"  }} -> {tt(dt, shape)}")


def translate(name, text, out):
    f = Fn(name)
    f.defined_scalars = set()
    m = re.search(r"func main<\w+>\((.*?)\)\s*\{", text)
    for a in split_top(m.group(1)):
        t, n = a.rsplit(" ", 1)
        f.ty[n] = parse_type(t); f.args.append(n)
    rets = [r.strip() for r in re.search(r"\}\s*->\s*\((.*?)\);", text).group(1).split(",")]
    body = text[m.end():]
    for line in body.split("\n"):
        line = line.strip()
        sm = re.match(r"(.+?)\s+(\w+)\s*=\s*(\w+)\((.*)\)\[(.*)\];$", line)
        if not sm:
            continue
        typ, res, op, argtxt, attrtxt = sm.groups()
        dt, shape = parse_type(typ)
        args = dict((k.strip(), v.strip()) for k, v in (x.split("=", 1) for x in split_top(argtxt)))
        attrs = dict((k.strip(), v.strip()) for k, v in (x.split("=", 1) for x in split_top(attrtxt)))
        f.emit(f"// {res} = {op}({argtxt})")
        T = tt(dt, shape) if shape is not None else dt

        if op == "const":
            v = parse_lit(attrs["val"])
            f.body.pop()
            if isinstance(v, tuple) and v[0] == "blob":
                f.ty[res] = (v[1], v[2]); f.blobs.append((res, v))
            else:
                f.env[res] = v
            continue
        if op == "constexpr_affine_dequantize":
            q = parse_lit(attrs["quantized_data"])
            f.ty[res + "_q"] = (q[1], q[2]); f.blobs.append((res + "_q", q))
            sc, zp = parse_lit(attrs["scale"]), parse_lit(attrs["zero_point"])
            f.emit(f"%{res}.s = arith.constant {sc[2]} : {dt}")
            f.emit(f"%{res}.zp = arith.constant {zp} : i8")
            f.elementwise(res, dt, shape, [(f"%{res}_q", q[2]), (f"%{res}.zp", None), (f"%{res}.s", None)],
                          ["%d = arith.subi %a0, %a1 : i8", f"%c = arith.sitofp %d : i8 to {dt}",
                           f"%r = arith.mulf %c, %a2 : {dt}"], in_dts=["i8", "i8", dt])
        elif op in ("mul", "add", "sub", "pow"):
            a, b = f.operand(args["x"], res, 0, dt), f.operand(args["y"], res, 1, dt)
            if a[1] == shape and b[1] == shape:
                e = f.empty(res, dt, shape)
                nm = {"mul": "mul", "add": "add", "sub": "sub", "pow": "powf"}[op]
                f.emit(f"%{res} = linalg.{nm} ins({a[0]}, {b[0]} : {T}, {T}) outs({e} : {T}) -> {T}")
            else:
                ar = {"mul": "arith.mulf", "add": "arith.addf", "sub": "arith.subf", "pow": "math.powf"}[op]
                f.elementwise(res, dt, shape, [a, b], [f"%r = {ar} %a0, %a1 : {dt}"])
        elif op == "sigmoid":
            x = f.operand(args["x"], res, 0, dt)
            f.elementwise(res, dt, shape, [x], [f"%one = arith.constant 1.0 : {dt}", f"%n = arith.negf %a0 : {dt}",
                                                f"%ex = math.exp %n : {dt}", f"%den = arith.addf %one, %ex : {dt}",
                                                f"%r = arith.divf %one, %den : {dt}"])
        elif op == "cast":
            x = args["x"]; sdt, ssh = f.ty[x]
            assert f.val(args["dtype"]) in ("fp16", "fp32")
            c = "arith.truncf" if (sdt, dt) == ("f32", "f16") else "arith.extf"
            f.elementwise(res, dt, shape, [(f"%{x}", ssh)], [f"%r = {c} %a0 : {sdt} to {dt}"], in_dts=[sdt])
        elif op == "quantize":
            x = args["input"]; sdt, ssh = f.ty[x]
            assert f.val(args["output_dtype"]) == "int8" and "zero_point" not in args
            s = f.operand(args["scale"], res, 1, sdt)
            # Measured on the device (M3 Pro, macOS 27.2.1; README "Sharp corners"): x is multiplied by the fp16
            # reciprocal of the scale, not divided by it, and the result is rounded half AWAY from zero, then saturated.
            f.elementwise(res, dt, shape, [(f"%{x}", ssh), s],
                          [f"%one = arith.constant 1.0 : {sdt}", f"%inv = arith.divf %one, %a1 : {sdt}",
                           f"%q = arith.mulf %a0, %inv : {sdt}", f"%rn = math.round %q : {sdt}",
                           f"%lo = arith.constant -128.0 : {sdt}", f"%hi = arith.constant 127.0 : {sdt}",
                           f"%c0 = arith.maximumf %rn, %lo : {sdt}", f"%c1 = arith.minimumf %c0, %hi : {sdt}",
                           f"%r = arith.fptosi %c1 : {sdt} to i8"], in_dts=[sdt, sdt])
        elif op == "dequantize":
            x = args["input"]; sdt, ssh = f.ty[x]
            assert "zero_point" not in args and dt == "f16"
            s = f.operand(args["scale"], res, 1, dt)
            # q * scale is exact in f32 (8 + 11 significant bits); the device then rounds it to fp16 with ties away from
            # zero.  arith.truncf has a `to_nearest_away` mode but the stock lowering ignores it and rounds to even, so
            # the tie rule is done on the bits: add half a unit of the dropped 13 mantissa bits to the magnitude, clear
            # them, and the truncation is exact.  Below 2^-14 (fp16 subnormals) the grid is 2^-24: round(p * 2^24) / 2^24.
            f.elementwise(res, dt, shape, [(f"%{x}", ssh), s],
                          ["%c = arith.sitofp %a0 : i8 to f32", "%s = arith.extf %a1 : f16 to f32", "%p = arith.mulf %c, %s : f32",
                           "%b = arith.bitcast %p : f32 to i32",
                           "%kmag = arith.constant 2147483647 : i32", "%ksgn = arith.constant -2147483648 : i32",
                           "%khalf = arith.constant 4096 : i32", "%kclr = arith.constant -8192 : i32", "%knorm = arith.constant 947912704 : i32",
                           "%m = arith.andi %b, %kmag : i32", "%sg = arith.andi %b, %ksgn : i32",
                           "%inc = arith.addi %m, %khalf : i32", "%tr = arith.andi %inc, %kclr : i32", "%ob = arith.ori %tr, %sg : i32",
                           "%ra = arith.bitcast %ob : i32 to f32",
                           "%up = arith.constant 16777216.0 : f32", "%dn = arith.constant 5.9604644775390625e-08 : f32",
                           "%pk = arith.mulf %p, %up : f32", "%rk = math.round %pk : f32", "%rs = arith.mulf %rk, %dn : f32",
                           "%small = arith.cmpi ult, %m, %knorm : i32", "%sel = arith.select %small, %rs, %ra : f32",
                           "%r = arith.truncf %sel : f32 to f16"], in_dts=["i8", dt])
        elif op == "reduce_sum":
            x = args["x"]; _, ssh = f.ty[x]
            axes = [a % len(ssh) for a in f.val(args["axes"])]
            assert f.val(args["keep_dims"]) is True
            rsh = [d for i, d in enumerate(ssh) if i not in axes]
            src = f"%{x}"
            adt = "f32" if (WIDE and dt == "f16") else dt
            if adt != dt:
                f.elementwise(res + ".x", adt, ssh, [(src, ssh)], [f"%r = arith.extf %a0 : {dt} to {adt}"], in_dts=[dt])
                src = f"%{res}.x"
            init = f.empty(res, adt, rsh, zero=True)
            rr = f"%{res}.r" if adt == dt else f"%{res}.w"
            f.emit(f"{rr} = linalg.reduce {{ arith.addf }} ins({src} : {tt(adt, ssh)}) outs({init} : {tt(adt, rsh)}) "
                   f"dimensions = [{', '.join(map(str, axes))}]")
            if adt != dt:
                f.elementwise(res + ".r", dt, rsh, [(rr, rsh)], [f"%r = arith.truncf %a0 : {adt} to {dt}"], in_dts=[adt])
            # keep_dims: re-insert the unit dims
            groups, cur = [], []
            for i in range(len(ssh)):
                cur.append(i)
                if i not in axes:
                    groups.append(cur); cur = []
            if cur:
                groups[-1] += cur
            # leading reduced dims attach to the first kept dim (handled by the loop); trailing to the last
            f.emit(f"%{res} = tensor.expand_shape %{res}.r {groups} output_shape {shape} : {tt(dt, rsh)} into {T}")
        elif op == "softmax":
            x = args["x"]
            ax = f.val(args["axis"]) % len(shape)
            e = f.empty(res, dt, shape)
            f.emit(f"%{res} = linalg.softmax dimension({ax}) ins(%{x} : {T}) outs({e} : {T}) -> {T}")
        elif op == "reshape":
            x = args["x"]; sdt, ssh = f.ty[x]
            assert f.val(args["shape"]) == shape
            f.emit(f"%{res}.s = arith.constant dense<{shape}> : tensor<{len(shape)}xi64>")
            f.emit(f"%{res} = tensor.reshape %{x}(%{res}.s) : ({tt(sdt, ssh)}, tensor<{len(shape)}xi64>) -> {T}")
        elif op == "transpose":
            x = args["x"]; sdt, ssh = f.ty[x]
            e = f.empty(res, dt, shape)
            f.emit(f"%{res} = linalg.transpose ins(%{x} : {tt(sdt, ssh)}) outs({e} : {T}) permutation = {f.val(args['perm'])}")
        elif op == "slice_by_size":
            x = args["x"]; sdt, ssh = f.ty[x]
            b, s = f.val(args["begin"]), f.val(args["size"])
            assert s == shape
            f.emit(f"%{res} = tensor.extract_slice %{x}{b} {s} {[1] * len(s)} : {tt(sdt, ssh)} to {T}")
        elif op == "concat":
            vals = [v.strip() for v in args["values"].strip("()").split(",")]
            assert f.val(args["interleave"]) is False
            ax = f.val(args["axis"]) % len(shape)
            f.emit(f"%{res} = tensor.concat dim({ax}) {', '.join('%' + v for v in vals)} : "
                   f"({', '.join(tt(*f.ty[v]) for v in vals)}) -> {T}")
        elif op == "conv":
            x, w = args["x"], args["weight"]
            assert f.val(args["pad_type"]) == "valid" and f.val(args["groups"]) == 1 and f.val(args["pad"]) == [0, 0, 0, 0]
            st, dl = f.val(args["strides"]), f.val(args["dilations"])
            adt, acc, AT = f.wide_begin(res, dt, shape)
            init = f.empty(acc, adt, shape, zero=True)
            f.emit(f"%{acc} = linalg.conv_2d_nchw_fchw {{dilations = dense<{dl}> : tensor<2xi64>, strides = dense<{st}> : tensor<2xi64>}}")
            f.emit(f"    ins(%{x}, %{w} : {tt(*f.ty[x])}, {tt(*f.ty[w])}) outs({init} : {AT}) -> {AT}")
            f.wide_end(res, dt, shape)
        elif op == "matmul":
            x, y = args["x"], args["y"]
            tx, ty = f.val(args["transpose_x"]), f.val(args["transpose_y"])
            xs, ys = f.ty[x][1], f.ty[y][1]
            r = len(shape); nb = r - 2
            assert len(xs) == r and len(ys) == r and xs[:nb] == ys[:nb]
            dims = [f"b{i}" for i in range(nb)] + ["i", "j", "k"]
            B = dims[:nb]
            mx = B + (["k", "i"] if tx else ["i", "k"])
            my = B + (["j", "k"] if ty else ["k", "j"])
            mo = B + ["i", "j"]
            am = lambda m: f"affine_map<({', '.join(dims)}) -> ({', '.join(m)})>"
            adt, acc, AT = f.wide_begin(res, dt, shape)
            init = f.empty(acc, adt, shape, zero=True)
            f.emit(f"%{acc} = linalg.contract indexing_maps = [{am(mx)}, {am(my)}, {am(mo)}]")
            f.emit(f"    ins(%{x}, %{y} : {tt(*f.ty[x])}, {tt(*f.ty[y])}) outs({init} : {AT}) -> {AT}")
            f.wide_end(res, dt, shape)
        else:
            raise NotImplementedError(op)
        f.ty[res] = (dt, shape)

    sig = [f"%{a}: {tt(*f.ty[a])}" for a in f.args] + [f"%{n}: {tt(v[1], v[2])}" for n, v in f.blobs]
    out.append(f"// MIL: {name}")
    for n, v in f.blobs:
        out.append(f"//   %{n}: BLOBFILE {v[3]} @ offset {v[4]}")
    out.append(f"func.func @{name}({', '.join(sig)})")
    out.append(f"    -> ({', '.join(tt(*f.ty[r]) for r in rets)}) {{")
    out.extend(f.body)
    out.append(f"  return {', '.join('%' + r for r in rets)} : {', '.join(tt(*f.ty[r]) for r in rets)}")
    out.append("}\n")


def main():
    global WIDE
    argv = [a for a in sys.argv[1:] if a != "--wide"]
    WIDE = len(argv) != len(sys.argv) - 1
    prefix = argv[0] if argv else ""
    out = []
    for chunk in re.split(r"^=== ", sys.stdin.read(), flags=re.M)[1:]:
        name, text = chunk.split("\n", 1)
        translate(prefix + name.strip().replace("gen_", "").replace("mil_", ""), text, out)
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()

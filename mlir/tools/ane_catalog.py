#!/usr/bin/env python3
"""Per-op device support catalog, derived from a tools/ane_ops.py run.

usage: ane_catalog.py WORKDIR RESULTS.jsonl [--out ops/ane_ops.json] [--md ops/ane_ops.md]

Every program (case) is a unit.  For each MIL op that appears in a program:
  supported   the op appears in at least one ok program (it compiled, ran, and matched its reference) and no failing program is blamed on it
  unsupported it appears in no ok program and in no program that ran (every program with it failed to compile)
  deviates    it appears in no ok program, but in programs that compiled and ran with output that differs from the reference
  sensitive   it appears in an ok program, but also in a failing program in which every op appears in some ok program: the failure is a
              variant (shape, dtype, attribute, combination), not the op; `failing_variants` lists those programs
For a failing program the *culprits* are the ops that occur in no ok program.  Argument kinds are read off the generated model.mil:
a tensor (SSA value), or a const of string / int / bool / float / int-list / blob / inline tensor type.
"""
import argparse, collections, json, os, re, sys

KIND_OF = [(r'^string\(', "str"), (r'^(int32|int8|uint8|int16|uint16|uint32)\(', "int"), (r'^bool\(', "bool"), (r'^(fp16|fp32)\(', "float"), (r'^tensor<int32, \[\d+\]>\(', "ints"),
           (r'^tensor<bool, \[\d+\]>\(', "bools"), (r'^tensor<(int8|uint8|int4|uint4|uint1|uint2|uint3|uint6)', "qblob"), (r'^tensor<(fp16|fp32), .*BLOBFILE', "blob"),
           (r'^tensor<(fp16|fp32), ', "tconst"), (r'^tensor<(int32|uint32)', "tconst")]

def split_top(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([<{": depth += 1
        if ch in ")]>}": depth -= 1
        if ch == "," and depth == 0: out.append(cur.strip()); cur = ""
        else: cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

def parse(text):
    """-> (ops: [(results, op, {arg: value_text})], consts: {name: kind}, inputs: [names], outputs: [names])"""
    m = re.search(r"func main<\w+>\((.*?)\)\s*\{(.*)\}\s*->\s*\((.*?)\);", text, re.S)
    if not m: return None
    inputs = [a.rsplit(" ", 1)[-1] for a in split_top(m.group(1))]; outputs = [o.strip() for o in m.group(3).split(",")]
    ops, consts = [], {}
    for line in m.group(2).split("\n"):
        line = line.strip()
        lm = re.match(r"(.+?)\s*=\s*(\w+)\((.*)\)\[(.*)\];$", line)
        if not lm: continue
        lhs, op, args, tail = lm.groups()
        results = [r.strip().rsplit(" ", 1)[-1] for r in split_top(lhs)]
        if op == "const":
            vm = re.search(r"val = (.*)\]$", "[" + tail) or re.search(r"val = (.*)$", tail)
            val = vm.group(1).rstrip("]") if vm else ""
            kind = next((k for pat, k in KIND_OF if re.match(pat, val)), "other")
            consts[results[0]] = (kind, val)
            continue
        a = {}
        for x in split_top(args):
            if "=" in x: k, v = x.split("=", 1); a[k.strip()] = v.strip()
        ops.append((results, op, a))
    return ops, consts, inputs, outputs

def dead_ops(ops, outputs):
    """names of the results that nothing uses: not an output, and not read (directly or through other ops) by anything that is.
    The converter runs with no passes, so a program can carry ops whose results are dropped, and the device refuses some of those programs
    for that reason alone (README, \"Which MIL ops the M3 Pro accepts\": a dropped op with an fp16 constant operand): a failing program with dead ops is weak evidence against the ops it contains."""
    defined = {}
    for results, op, args in ops:
        reads = set()
        for v in args.values(): reads.update(re.findall(r"[A-Za-z_][\w.]*", v))
        for r in results: defined[r] = reads
    live, stack = set(), list(outputs)
    while stack:
        n = stack.pop()
        if n in live or n not in defined: continue
        live.add(n); stack.extend(defined[n])
    return [r for r in defined if r not in live]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("work"); ap.add_argument("results"); ap.add_argument("--out"); ap.add_argument("--md")
    a = ap.parse_args()
    status = {}
    for l in open(a.results):
        r = json.loads(l); status[r["name"]] = r
    ops_seen = collections.defaultdict(lambda: {"ok": [], "wrong": [], "fail": [], "sig": collections.defaultdict(collections.Counter), "results": collections.Counter(), "errs": collections.Counter(), "uses": 0, "present": collections.Counter(), "clean_fail": set()})
    prog_ops = {}
    for name, r in status.items():
        p = f"{a.work}/{name}/model.mil"
        if not os.path.exists(p): continue
        parsed = parse(open(p).read())
        if not parsed: continue
        ops, consts, _, outs = parsed; prog_ops[name] = set(); has_dead = bool(dead_ops(ops, outs))
        for results, op, args in ops:
            prog_ops[name].add(op); e = ops_seen[op]; e["uses"] += 1
            for k in args: e["present"][k] += 1
            if r["status"] == "fail" and not has_dead: e["clean_fail"].add(name)
            e[r["status"] if r["status"] in ("ok", "wrong", "fail") else "fail"].append(name)
            e["results"][len(results)] += 1
            for k, v in args.items():
                kind = consts[v][0] if v in consts else ("tensor" if re.match(r"^[A-Za-z_][\w.]*$", v) else ("tuple" if v.startswith("(") else next((k for pat, k in KIND_OF if re.match(pat, v)), "other")))
                e["sig"][k][kind] += 1
            if r["status"] == "fail": e["errs"][r.get("err", "?")[:40]] += 1
    ok_ops = {op for op, e in ops_seen.items() if e["ok"]}
    culprits, variant_fail = {}, collections.defaultdict(list)
    for name, ops in prog_ops.items():
        if status[name]["status"] == "fail":
            c = sorted(o for o in ops if o not in ok_ops)
            culprits[name] = c
            if not c:
                for o in ops: variant_fail[o].append(name)
    cat = {}
    for op, e in sorted(ops_seen.items()):
        st = ("deviates" if e["wrong"] else "unsupported") if not e["ok"] else ("sensitive" if variant_fail.get(op) else "supported")
        cat[op] = {"status": st, "ok": len(e["ok"]), "wrong": len(e["wrong"]), "fail": len(e["fail"]), "results": sorted(e["results"]), "uses": e["uses"],
                   "present": dict(sorted(e["present"].items())), "failing_without_dead_ops": len(e["clean_fail"]),
                   "args": {k: dict(c) for k, c in sorted(e["sig"].items())}, "error_classes": dict(e["errs"]),
                   "ok_programs": sorted(e["ok"])[:6], "failing_programs": sorted(e["fail"] + e["wrong"])[:6], "failing_variants": sorted(variant_fail.get(op, []))[:8]}
    out = {"ops": cat, "failing_program_culprits": culprits,
           "programs": {"ok": sum(1 for r in status.values() if r["status"] == "ok"), "wrong": sum(1 for r in status.values() if r["status"] == "wrong"),
                        "fail": sum(1 for r in status.values() if r["status"] == "fail"), "build": sum(1 for r in status.values() if r["status"] == "build")}}
    if a.out: json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    cnt = collections.Counter(v["status"] for v in cat.values())
    print(f"{len(cat)} MIL ops seen; {dict(cnt)}; programs {out['programs']}")
    if a.md:
        with open(a.md, "w") as f:
            f.write("| MIL op | status | ok | wrong | fail | fail, no dead ops | arguments (kinds seen) | failing error classes |\n|---|---|---:|---:|---:|---:|---|---|\n")
            for op, v in cat.items():
                args = "; ".join(f"{k}:{'/'.join(c)}" for k, c in v["args"].items())
                f.write(f"| `{op}` | {v['status']} | {v['ok']} | {v['wrong']} | {v['fail']} | {v['failing_without_dead_ops']} | {args} | {', '.join(v['error_classes'])} |\n")
    weak = [op for op, v in cat.items() if v["status"] == "unsupported" and not v["failing_without_dead_ops"]]
    if weak: print("unsupported only in programs that carry dead ops (weak evidence):", " ".join(weak))

if __name__ == "__main__":
    main()

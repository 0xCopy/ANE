import re, math, sys, subprocess, os
D,HID,H,KVH,HD,S=[int(x) for x in sys.argv[1:7]]; lit=sys.argv[7]; aneout=sys.argv[8]
QD,KD=H*HD,KVH*HD; SC=H*S
ks=[("dyn_matmul_mil",D,S+HID,HID,[]),("sdpa_fwd_dynamic",D,S+QD+2*KD,2*QD+2*KD+D,["cos4","sin4","mask4"]),("ffn_fused_dynamic",D,2*S+3*HID,D+3*HID,[]),
("ffn_bwd_w13t_dynamic",HID,2*S+2*D,D,[]),("sdpa_bwd1_noweight",4*QD,S,QD+2*SC,["mask4"]),("sdpa_bwd2",2*SC+2*QD,S,2*QD,[])]
M="tensor<?x?xf16>"
o=["func.func private @printMemrefF32(%ptr : tensor<*xf32>)","func.func @main() {","  %seed = arith.constant 0.3 : f32","  %amp = arith.constant 0.5 : f32",
f"  %s = arith.constant {S} : index",f"  %hd = arith.constant {HD} : index",
f"  %mask = call @causal_mask(%s) : (index) -> {M}",f"  %cos, %sin = call @rope_tables(%s, %hd) : (index, index) -> ({M}, {M})"]
ty={}
for n,a,b in [("mask",S,S),("cos",S,HD),("sin",S,HD)]:
    o.append(f"  %{n}c = tensor.cast %{n} : {M} to tensor<{a}x{b}xf16>")
    o.append(f"  %{n}4 = tensor.expand_shape %{n}c [[0, 1, 2], [3]] output_shape [1, 1, {a}, {b}] : tensor<{a}x{b}xf16> into tensor<1x1x{a}x{b}xf16>")
    ty[n+"4"]=f"tensor<1x1x{a}x{b}xf16>"
for i,(n,ic,sp,oc,ex) in enumerate(ks):
    o+= [f"  %r{i} = arith.constant {ic} : index", f"  %c{i} = arith.constant {sp} : index",
     f"  %g{i} = call @gen(%r{i}, %c{i}, %seed, %amp) : (index, index, f32, f32) -> {M}",
     f"  %gc{i} = tensor.cast %g{i} : {M} to tensor<{ic}x{sp}xf16>",
     f"  %g4{i} = tensor.expand_shape %gc{i} [[0, 1], [2, 3]] output_shape [1, {ic}, 1, {sp}] : tensor<{ic}x{sp}xf16> into tensor<1x{ic}x1x{sp}xf16>",
     f"  %o{i} = call @ane_{n}({', '.join(['%g4'+str(i)]+['%'+e for e in ex])}) : ({', '.join([f'tensor<1x{ic}x1x{sp}xf16>']+[ty[e] for e in ex])}) -> tensor<1x{oc}x1x{S}xf16>",
     f"  %oc{i} = tensor.collapse_shape %o{i} [[0, 1], [2, 3]] : tensor<1x{oc}x1x{S}xf16> into tensor<{oc}x{S}xf16>",
     f"  %od{i} = tensor.cast %oc{i} : tensor<{oc}x{S}xf16> to {M}",
     f"  %of{i} = call @to_f32(%od{i}) : ({M}) -> tensor<?x?xf32>",
     f"  %ou{i} = tensor.cast %of{i} : tensor<?x?xf32> to tensor<*xf32>",
     f'  vector.print str "KERNEL {n}\\n"',
     f"  call @printMemrefF32(%ou{i}) : (tensor<*xf32>) -> ()"]
o+=["  return","}"]
open("probe_main.mlir","w").write("\n".join(o)+"\n")
T=os.path.expanduser("~/work/mlir-ane/mlir/test"); B="/opt/homebrew/opt/llvm/bin"; L="/opt/homebrew/opt/llvm/lib"
PIPE=f"--transform-preload-library=transform-library-paths={T}/decompose_softmax.mlir --transform-interpreter --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts"
out=subprocess.run(f"cat {T}/harness.mlir {T}/../normal/ane_normal.mlir {lit} probe_main.mlir | {B}/mlir-opt {PIPE} | {B}/mlir-runner -e main -entry-point-result=void -shared-libs={L}/libmlir_runner_utils.dylib,{L}/libmlir_c_runner_utils.dylib",shell=True,capture_output=True,text=True)
if out.returncode: print(out.stderr[:2000]); sys.exit(1)
a={};cur=None
for l in open(aneout):
    if l.startswith('KERNEL'): cur=l.split()[1]; a[cur]=[]
    else: a[cur].append(float(l))
m={};cur=None
for l in out.stdout.split("\n"):
    if l.startswith('KERNEL'): cur=l.split()[1]; m[cur]=[]
    elif l.startswith('Unranked'): continue
    elif cur: m[cur]+= [float(x) for x in re.findall(r'-?(?:nan|inf|\d+\.?\d*(?:e[-+]?\d+)?)',l)]
import struct
f16=lambda v: struct.unpack('<e',struct.pack('<e',v))[0]
def ulp(v):
    v=abs(v)
    if v < 2.0**-14: return 2.0**-24
    return 2.0**(math.floor(math.log2(v))-10)
for k in a:
    x,y=a[k],[f16(v) for v in m[k]]
    assert len(x)==len(y),(k,len(x),len(y))
    pr=[(p,q) for p,q in zip(x,y) if math.isfinite(p) and math.isfinite(q)]
    nf=len(x)-len(pr)
    du=sorted(abs(p-q)/ulp(max(abs(p),abs(q))) for p,q in pr)
    mx=max(abs(p) for p,_ in pr)
    print(f"{k:22s} n={len(x):5d} max|ane|={mx:7.4g} maxabsdiff={max(abs(p-q) for p,q in pr):8.3g} identical={sum(1 for v in du if v==0)*100//len(x):3d}% within1ulp={sum(1 for v in du if v<=1)*100//len(x):3d}% p99={du[int(len(du)*.99)]:.0f}ulp max={du[-1]:.0f}ulp nonfinite={nf}")

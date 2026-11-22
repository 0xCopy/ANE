# MLIR translations of the ANE kernels

The MIL programs this repo sends to the Neural Engine, the same computations with the ANE packaging removed, and the
CPU-side NEON code, all as MLIR that the stock LLVM 23 tools parse, lower and run.

    literal/      one MLIR function per MIL program, op for op (linalg/tensor on static fp16 tensors)
      stories110m/static.mlir     training/stories_mil.h, ane_rmsnorm_bwd.h, ane_classifier.h   (11 kernels)
      stories110m/dynamic.mlir    training/training_dynamic/mil_dynamic.h, MODEL=stories110m    (11 kernels)
      qwen3_06b/dynamic.mlir      the same, MODEL=qwen3_06b (GQA 16/8 heads)
      qwen3_06b/dynamic_wide.mlir the same with f32 accumulation (see "fp16 accumulation")
      primitives.mlir             training/ane_mil_gen.h and the W8A8 / FP16 chains of ane_int8_bench.m (6 kernels)
    normal/ane_normal.mlir        the generalized, normalized forms: dynamic shapes, [C, S] matrices, named primitives
    aarch64/      neon_fp16.mlir (io.h conversions, f16 arithmetic, quantize), neon_i8.mlir (sdot), run.mlir, asm.test
    test/         small/ = the literal kernels at tiny sizes; check_*.mlir run literal and normal form on the same data
    tools/        mil2mlir.py (the translator), dump_*.m (print the MIL the generators emit), gen_checks.py,
                  ane_probe.m + ane_compare.py (run kernels on the real ANE and diff against the MLIR)

Run everything: `mlir/run-tests.sh` (needs LLVM 23 with MLIR at /opt/homebrew/opt/llvm, or set LLVM_PREFIX).

## MIL -> MLIR (literal/)

| MIL                              | MLIR                                                                          |
|----------------------------------|-------------------------------------------------------------------------------|
| `tensor<fp16, [..]>`             | `tensor<..xf16>`                                                              |
| `const` (int/bool/string/shape)  | folded into the op that uses it (attribute, permutation, slice bounds)        |
| `const` (fp16 scalar)            | `arith.constant`                                                              |
| `const` with `BLOBFILE`          | a function argument; path and offset are in the comment above the function    |
| `conv` (1x1, valid, groups 1)    | `linalg.conv_2d_nchw_fchw`                                                    |
| `matmul` (+ transpose_x/y)       | `linalg.contract` with the transposes in the indexing maps                    |
| `mul` `add` `sub` `pow`          | `linalg.mul/add/sub/powf`, or `linalg.generic` when one side broadcasts       |
| `sigmoid`                        | `linalg.generic`: 1 / (1 + exp(-x))                                           |
| `softmax`                        | `linalg.softmax`                                                              |
| `reduce_sum` (keep_dims)         | `linalg.reduce` + `tensor.expand_shape`                                       |
| `reshape`                        | `tensor.reshape`                                                              |
| `transpose`                      | `linalg.transpose`                                                            |
| `slice_by_size`                  | `tensor.extract_slice`                                                        |
| `concat`                         | `tensor.concat`                                                               |
| `cast` fp32 <-> fp16             | `arith.truncf` / `arith.extf`                                                 |
| `quantize`                       | `x * (1 / scale)`, `math.round` (half away from zero), clamp to [-128, 127], `arith.fptosi` |
| `dequantize`                     | `sitofp`, mul in f32, the fp16 rounding with ties away from zero done on the bits (Sharp corners) |
| `constexpr_affine_dequantize`    | the int8 blob is the argument; the dequantize is an ordinary op               |

Value names are the MIL names. Regenerate after changing a generator:

    xcrun clang -w -fobjc-arc -framework Foundation -framework IOSurface -framework Accelerate -DACCELERATE_NEW_LAPACK \
        -Itraining/training_dynamic -include training/training_dynamic/models/qwen3_06b.h -o /tmp/dd mlir/tools/dump_dyn.m
    /tmp/dd | python3 mlir/tools/mil2mlir.py ane_ > mlir/literal/qwen3_06b/dynamic.mlir

## The `ane` dialect (dialect/, ane/)

`dialect/ane.irdl.mlir` declares the MIL operations as MLIR ops, one for one, in IRDL: the stock `mlir-opt` loads it
with `--irdl-file`, no C++ and no build. 117 ops. 21 are written by hand in `tools/ane_dialect.py` (the 16 MIL operations
the generators use, with what the device does to them; `ane.const` for scalars, `ane.dense` for small inline tensors,
`ane.blob` for BLOBFILE constants, and the two forms of `constexpr_affine_dequantize`). The other 96 are generated from
`ops/ane_ops.json` (next section): every MIL op that compiled and ran correctly on the device, with the arguments the
sweep programs gave it. The 38 ops the compiler refused are not declared; the declaration lists them in a comment.
A kernel is a `func.func` of ane ops; MIL's int, bool, string and shape `const` statements are attributes.
`ane/` holds the 39 kernels in this form.

    tools/ane_dialect.py from-mil   MIL text -> ane ops
    tools/ane_dialect.py to-mil     ane ops -> MIL text, for the device
    tools/ane_to_ll.sh [--wide]     ane ops -> MIL -> linalg (mil2mlir.py) -> LLVM IR, for the CPU

One kernel text therefore has two targets: the Neural Engine, and AArch64 through LLVM.

What is checked (test/check_ane_dialect.mlir, aarch64/ane_asm.test):

- All five files in ane/ verify against the declaration; ane -> MIL -> ane is a fixed point, and `ane/primitives.mlir`
  regenerates byte for byte from the generators.
- On the device, for every program of the sweep that is `ok` (663 of the 1008 in the last full pass), the MIL printed back from the
  dialect (`ane_ops.py --via-dialect`: MIL -> `ane` ops -> `mlir-opt` verifies them against the declaration -> MIL -> device) gives
  output byte-identical to the original MIL's. The 345 programs that are not `ok` were not round-tripped.
- The small dynamic kernels, taken ane -> MIL -> linalg, pass check_dynamic's expectations unchanged.
- Wrong operand count, a missing attribute, an attribute of the wrong kind and a softmax that changes its type are
  rejected.
- On the device (manual, M3 Pro, macOS 15.8.3 and again on 27.2.1): for the six kernels of tools/ane_probe.m at the mid
  sizes, the MIL printed from the dialect compiles, evaluates, and returns byte-identical output to the generators' own
  MIL (38912 values, max magnitude 251); the same holds for the W8A8 chain of ane_int8_bench.m (`tools/ane_w8a8.py`).
  `ane_probe <dir>` reads `<dir>/<kernel>.mil` in place of the generator.
- On the CPU, an fp16 `ane.matmul` vectorizes its products (`fmul.8h`) but not its sum: float addition keeps its
  written order, so the sum is a chain of scalar `fadd`. Without FEAT_FP16 everything also converts through f32.

Limits of the declaration. IRDL checks counts, tensor-ness, attribute names and kinds. It does not check shapes or
element types, and it gives variadic operands one type, so `ane.concat` takes its values as `tensor<*xT>` through
`tensor.cast`; an op with several results (`ane.split`) returns `tensor<*xT>` and each result is cast back. Three more
facts about this IRDL (LLVM 23.1.6), all checked: it has no optional attributes (a name ending in `?` is taken literally), so
an attribute the sweep programs sometimes left out is not declared and is accepted unchecked, the round trip being its
check; an optional operand may be absent, and with two or more optional or variadic operands the op needs
`operandSegmentSizes = array<i32: ...>`, which the converters write; `irdl.any` is the only constraint that takes a dense
attribute. The device limits measured on the M3 Pro are comments on the ops and the table "Device facts" below, not
verifier rules: `matmul` fails to compile above a contracted dim of 16384, `conv` above 32512 input channels, a blob
chunk header must carry the data offset, and a model with several weight files must pass the bundle rule. The CPU path
goes through MIL text, not a C++ lowering pass.

## Which MIL ops the M3 Pro accepts (ops/, tools/ane_ops.py)

`tools/ane_ops.py` builds each case as a Core ML program (coremltools, `ct-venv`: `python3.12 -m venv`, `pip install coremltools
numpy`), converts it to MIL with no optimization passes, compiles and runs that MIL on the ANE through `tools/ane_run.m`, and
compares the output with a reference: Core ML on the CPU in fp32 and in fp16, or a numpy model written for the case when the
device is known to differ from Core ML (rounding, `resample` outside the image). A case is `ok` (compiled, ran, matched),
`wrong` (ran, differs), `fail` (the compiler refused it: `InvalidMILProgram` or `CompilationFailure`, nothing finer) or `build`
(the converter could not build it: no verdict). Cases
live in `tools/ane_cases/*_ops.py`, one module per family (`basic`, `ix` indexing, `rd` reductions and math, `cv` conv, pool,
resize, `wq` weight and activation quantization, `rc` recurrent, random, control flow and program shape, `lg` compositions of
accepted ops, `sh` shape-dependence); 1020 cases run by default, 1034 are defined (`--all` adds the 14 that are slow or too
big for the disk: tensors of 64 MB and more, chains of 768 steps and more, a 1024-wide program).

    python3.12 -m venv ~/.local/share/mlir-ane/ct-venv && ~/.local/share/mlir-ane/ct-venv/bin/pip install coremltools numpy
    ~/.local/share/mlir-ane/ct-venv/bin/python tools/ane_ops.py --jobs 6 --baseline ops/ane_baseline.json     # about 5 minutes, about 1 GB of disk
    ~/.local/share/mlir-ane/ct-venv/bin/python tools/ane_ops.py --via-dialect --baseline ops/ane_baseline.json

`ops/ane_baseline.json` is each case's status on macOS 27.2.1 / M3 Pro; `--baseline` prints `DIFFERS` per case that changed and
exits with their number. `ops/ane_ops.json` and `ops/ane_ops.md` are the per-op catalog `tools/ane_catalog.py` derives from a
run (`ane_catalog.py WORKDIR RESULTS.jsonl --out ops/ane_ops.json --md ops/ane_ops.md`): `supported` (appears in an ok program;
93 ops), `sensitive` (also in failing programs that contain no op without an ok program, so a variant fails, not the op; 20),
`unsupported` (in no ok program; 38), `deviates` (compiles, output is not the op's; 1). The dialect's generated ops come from
this file.

Results of the last full pass (1008 cases; 12 attention-mask cases were added after it, all listed below): 663 ok, 17 wrong, 323 fail, 5 build. Of the 38 refused ops, 32 have two or more
failing programs without dead ops (see "Dead ops" below) and six have one (`list_length`, `list_read`,
`non_maximum_suppression`, `random_bernoulli`, `random_categorical`, `random_normal`): those six are the weakest entries.

- Refused (every program that contains the op failed to compile): `tan asin acos sinh cosh atanh`, `mod`, `logical_or
  logical_xor`, `reduce_prod reduce_argmax reduce_argmin`, `topk`, `cumsum`, `gather gather_nd gather_along_axis`, `scatter
  scatter_nd scatter_along_axis`, `one_hot`, `band_part`, `non_zero`, `sliding_windows`, `reverse_sequence`, `shape`,
  `crop_resize`, `non_maximum_suppression`, `lstm gru rnn`, `make_list list_read list_write list_length`, `random_bernoulli
  random_categorical random_normal`. Refused in the variants tried (the op is accepted in others): `cast` fp16 -> int32 -> fp16 and
  fp16 -> fp32 -> fp16 (int8, uint8, int16, uint16 work; fp32 at the program boundary, as the repo's kernels use it, is a
  different case and works); `resample` with `symmetric` padding; `constexpr_blockwise_shift_scale` with blocks of 8, 16 or 32 (64
  works, int8 and int4 data; larger blocks were not tried); `constexpr_affine_dequantize` with `axis = 1` (per-channel or
  per-tensor scale; `axis = 0` works); a program of 768 or more sequential `mul`+`add`+`relu` steps (the compiler service
  crashes, below). Kernel extents of `conv` (IC 1 unless noted): 1xKW refused for KW 16 to 34 and compiles for KW 12 to 15
  (OC 16 and 32); KHx1 compiles for KH 12 to 30 and is refused at 32 and 34; square kernels on a 64x64 input, OC 16: 17 and 20
  refused, 24 and 32 compile; at 4 input channels 16x1, 16x4, 16x8 compile and 4x16, 1x16 are refused. No rule on the
  width alone fits these, so none is claimed. Per-op program counts: `ops/ane_ops.md`.
- Compositions of accepted ops give most of what is refused, and the device runs them correctly (`lg_*`, in the baseline):
  `tan sinh cosh` from `sin cos exp`; `logical_or` from `logical_not` and `logical_and`; `reduce_prod` as `exp(reduce_sum(log))`
  (positive inputs, tested near 1); `cumsum` as a matmul with a triangular matrix; `band_part` as a multiplication with a mask;
  `one_hot` as `equal` with an iota; `gather` along W or C with dynamic indices as a matmul with the one-hot of the indices;
  `scatter` with add as a matmul with the one-hot of the indices (`lg_scatter_add_c`); `argmax`, `argmin` and the indices of
  `topk` as `reduce_max`, `equal` and an iota (the smallest iota value where the elements are equal), repeated k times with the
  chosen element masked out for `topk` (k up to 8 tested; `lg_topk*_vals` also ok). `argsort` itself is accepted, but its
  int32 result cannot feed another op (a `cast`, `reshape`, `slice` or `identity` after it is refused), so the argmax recipe is
  the usable one. Not recomposed here: `gather_nd`, `scatter_nd`, `non_zero`, `lstm gru rnn`, `crop_resize`, `random_*`.
- Dead ops. The converter runs with no passes, so a program may carry ops whose results nothing reads, and the device refuses
  some such programs by that alone. In `lg_dead_*`: a dropped op with an fp16 constant operand makes the compile fail
  (`equal` with 1.0, `mul` by -1000, `greater` with 0.0, `add` of a tensor of ones, also when the constant is shared with a live
  op); dropped ops without one compile (`exp`, `exp(exp)`, `reduce_sum`, `add(x, r)`, `mul(x, x)`). A refusal is therefore counted
  against an op only when a failing program without any dead op contains it (`ane_catalog.py` prints the count per op;
  `ops/ane_ops.md`, column "fail, no dead ops").
- Differs from Core ML on the CPU, same inputs. `round` and the casts fp16 -> int8, uint8, int16, uint16 round ties away from
  zero and saturate (`cast_roundtrip_*`, `cast_sat_*`: a numpy model of that agrees; ties-to-even does not for int8 and uint16).
  `quantize` followed by `dequantize` with a zero point z and scale s: `b = f16(z * s)` and `inv = f16(1 / s)` round to nearest
  even; `t = f16(f16(x + b) * inv)` (the sum rounds ties away from zero, nearest-even gives mismatches; whether the product does
  is not decided by this data); `q = clip(round_half_away(t))`; `y = f16((q - z) * s)` with ties away. 0 of 2048 values differ
  for a per-tensor and a per-channel scale and zero points 0, 7, 100 and 128 (`wq_act_qdq_*`, five cases). `random_uniform`
  compiles and returns one fixed array: identical at every evaluation (4, 5, 9 and 44 evaluations compared) and for seeds 1 and 7,
  all inside the first sixteenth of the requested range (`[0, 1)` gives values up to 0.0624, and the array for `[0, 16)` is 16
  times the one for `[0, 1)`); `sh_randu_*` is `wrong` in the baseline for that reason. `resample` with constant padding
  returns the padding value for every sample point outside the image, where Core ML blends the in-image corners of a point that is
  a little outside; `cv_resample` and `sh_resample_constant_*_dev` check the device's rule.
- `scaled_dot_product_attention` with `attn_mask`. The upstream README says "ANE hardware ignores `attn_mask` in SDPA ops". Measured
  here (`sdpa_*`): a mask that is a constant in the MIL (an inline `const`, a BLOBFILE `const`; float or boolean; smooth values,
  0 / -1e4, 0 / -65504, 0 / -inf) is ignored: the output equals the unmasked attention to 0.0006 and misses the masked one by
  1.1 to 1.2 (a smooth constant mask by 0.26); upstream's own inline-mask test (12 heads, 8 tokens) fails to compile here, with its bare integer literals and with decimal ones
  alike, while the 32x32 inline constants that coremltools writes compile and are ignored; the cause of that difference is not
  isolated. A mask that
  is a runtime input is applied: the output matches the masked attention to 0.001 and misses the unmasked one by 1.0 to 1.3, at
  [1, 8, 32, 64], at upstream's own test shape [1, 12, 8, 64], and at [1, 12, 256, 64] (Stories110M), for masks 0 / -1e4, 0 / -65504,
  0 / -inf and a smooth random one (`tools/ane_sdpa_mask_check.py WORKDIR` compares each output with numpy attention with and
  without the mask). Upstream's
  `training/test_ane_sdpa5.m` reported the runtime mask as failed because of how that test binds its surfaces, not because of the
  device: the model indexes its input symbols alphabetically (`k, m, q, v`), the test passes `q, k, v, mask`; and the mask surface
  has dense 16-byte rows where the model expects 64-byte rows (512 bytes). Either fix alone leaves Test 4 failing with "IOSurface
  smaller than the model expects"; both together make it report CAUSAL (diff 0.0004 against the causal CPU reference), and the
  order fix alone changes its no-mask Test 1 from 0.0556 to 0.0004 from the non-causal reference, so the original binding had q and
  k swapped there too. Timing at [1, 12, 256, 64], median of three serial runs, ms per eval: fused op with a mask input 0.313, the
  same attention as matmul + add + softmax + matmul 0.300, unmasked fused op 0.260. The fused op saves no time over the decomposed
  form, which also runs entirely on the ANE; it is one op instead of four. Whether this differs on an M4 or macOS 15 was not
  measured: the test's own bug is enough to explain upstream's conclusion.
- Sizes, from the baseline. Plain `relu`: the last dimension of a [1, 1, 1, W] tensor compiles up to 16384 and is refused at
  16385 and beyond; the channel dimension of a [1, C, 1, 1] tensor compiles up to 65536 and is refused at 65537. The byte size
  is not the limit: [1, 16384, 1, 16384] (512 MB) compiles, and 8 MB as 131072 x 32 does not (131072 channels), while the same 8 MB as
  1024 x 4096, 65536 x 64 and 64 x 256 x 256 does (the 64 MB to 512 MB cases ran once, before the first service crash, and are
  skipped by default). A 1x1 `conv` with 64 outputs on 64 positions: 32512 input channels compile in every layout of the 64
  positions (1x64, 2x32, 8x8, 64x1); 32513 is refused as 1x64 and 2x32 and compiles as 8x8 and 64x1; 65536 compiled only as
  8x8. The 32512 of the Device facts table below is therefore the limit for the 1x64 layout, not for the op. A wide program of
  3073 ops (512 parallel branches) compiles; 6145 ops (1024 branches) compiles in 131 s; a chain of 736 steps (2208 ops)
  compiles and 768 do not.
- Unstable (`flaky` in the baseline: recorded, not compared). A 16x16 kernel with 64 output channels compiles and returns garbage
  (no output channel correlates with the reference) at IC 1 to 16, is refused at IC 32, and the 16x16 kernel with 16 or 24
  output channels is correct, except `sh_k16_oc32_ic4`, again garbage. Between the first sweep (21:05) and the later ones these cases
  changed: `sh_k16_oc16_ic32`, `ic48`, `ic64` and `sh_k16_oc64_ic8`, `ic16` were refused and are now correct or garbage; the
  service had crashed once in between (21:22). `sh_k10_oc64_ic16`, `k12`, `k14`, `k15` and `sh_k16_oc24_ic4` failed in the 6-job
  sweep of 21:34 with `Code=22 ... Failed to allocate memory for file` and compiled in every later run (16 compiles in a row
  of `k15`, `k10`: same outcome; 12 and 16 different weight draws: same outcome). Of the 20 flaky cases, 14 gave the same status in
  the 21:34 sweep and in the last full run, 6 differed. The cause is not isolated: the state of the compiler service process and
  concurrent compiles are candidates, a changed weight draw is not. Do not read a size limit off these cases.

The compiler service's crash report (`/Library/Logs/DiagnosticReports/ANECompilerService-*.ips`): SIGBUS in a recursion
`ZinIrParallelExecutionOpportunityFinder::FindAllAttentionBranchesInRegion` -> `TopologicalSortImpl`, 425 frames, consistent with a
stack overflow on a long chain. Every compile in flight when it dies fails with `connection to service named
com.apple.ANECompilerService`, which is not a verdict on the case; `ane_ops.py` retries that error and the allocation failure
three times. The error from the compiler is only a class (`InvalidMILProgram`, `CompilationFailure`); `ANE_DEBUG=capture_folder=...;
capture_bytecode=1;capture_anec` is accepted by the client and wrote nothing in the compiles tried.

## Generalizations and normalizations (normal/)

- Sizes are `?`; heads, eps and the attention scale are arguments. One function covers Stories110M and Qwen3.
- An activation is a matrix `[C, S]` (the IOSurface contents). A per-head tensor is `[H, S, HD]`.
- A 1x1 conv with a baked weight and the dynamic reshape/transpose/matmul/transpose/reshape pattern are the same
  thing: `@linear` (weight `[OC, IC]`) and `@linear_t` (weight `[IC, OC]`, as the dynamic kernels stage it).
- Backward kernels take the forward weight; the transposed blobs (`W1t`, `Wot`, ...) are `@linear_t` on it.
- The single packed input (`slice_by_size`) becomes arguments; the single packed output (`concat`) becomes results.
- `pow(x, -0.5)` is `math.rsqrt`; RoPE (reshape to pairs, slice, negate, concat, reshape) is one `linalg.generic`;
  GQA tiling is an index map; SDPA and its backward are `@sdpa` / `@sdpa_bwd` (bwd1 + bwd2 in one function).
- Tables that the C code bakes into blobs are functions: `@causal_mask`, `@rope_tables`.

`test/check_*.mlir` execute each literal kernel and its normal form on the same inputs. 33 of the 46 compared outputs
are bit-identical. The other 13 are everything downstream of an RMSNorm (rsqrt instead of pow, and one fused loop
instead of a chain of fp16 ops); they differ by at most 0.004 on values of magnitude 1-2.

## Sharp corners

**fp16 accumulation.** The MIL types say every intermediate is fp16. The hardware does not round a sum of products
after each term. Measured on an M3 Pro (`tools/ane_probe.m`, DIM=64 HIDDEN=96 HEADS=4 KV_HEADS=2 HD=16 SEQ=32),
ANE output against the MLIR translation, in fp16 ulps:

| kernel               | fp16 accumulate: identical / within 1 ulp | f32 accumulate (`--wide`): identical / within 1 ulp |
|----------------------|-------------------------------------------|-----------------------------------------------------|
| dyn_matmul           | 20% / 61%                                 | 78% / 99%                                           |
| sdpa_fwd_dynamic     | 18% / 68%                                 | 59% / 97%                                           |
| ffn_fused_dynamic    | 16% / 49%                                 | 60% / 84%                                           |
| ffn_bwd_w13t_dynamic | 20% / 41%                                 | 93% / 99%                                           |
| sdpa_bwd1_noweight   | 25% / 66%                                 | 29% / 78%                                           |
| sdpa_bwd2            |  1% / 37%                                 |  1% / 51%                                           |

So `mil2mlir.py --wide` (conv, matmul and reduce_sum accumulate in f32, one rounding at the end) is the closer model
of the device; neither is bit-exact. The remaining difference is concentrated in the kernels with softmax and long
elementwise chains, which were not modelled further.

**GQA head tiling disagrees between forward and backward.** `gen_sdpa_fwd_dynamic` tiles K and V with
`concat(k, k)`: query head h reads KV head `h mod KV_HEADS`. `gqa_tile_kv` / `gqa_reduce_kv` in io.h, which feed and
reduce the backward kernels, use `h div GQA_RATIO`. They agree only when HEADS == KV_HEADS (Stories110M). For
Qwen3 (16/8) the backward pass differentiates a different head assignment than the forward pass computed. Both are in
normal/ as `@gqa_tile_mod` and `@gqa_tile_div`; the last line of check_dynamic shows they differ.

**fp32 -> fp16 is not a truncation.** Round to nearest even, and anything of magnitude >= 65520 becomes inf
(aarch64/run.mlir). The causal mask is -65504, the largest finite fp16, not -inf.

**f16 arithmetic needs FEAT_FP16.** With `+fullfp16` an 8-lane multiply-add is `fmul.8h`/`fadd.8h`; without it LLVM
widens every operation to f32 and narrows again (aarch64/asm.test). The conversions (`fcvtn`/`fcvtl`) are base NEON.
`arith.mulf` + `arith.addf` is not fused; `math.fma` is (`fmla.8h`).

**Float to int.** `arith.fptosi` is poison out of range, the AArch64 instruction (`fcvtzs`) saturates. The quantize
translation clamps first so the IR does not depend on the target.

**W8A8 in integers.** Dequantize -> fp16 conv -> quantize loses bits that an int8 dot product keeps: 4 x (127 x 127)
x 1/64 is 1008.0625 exactly (`arm_neon.intr.sdot`, then one scale) and 1008 through fp16 (aarch64/neon_i8.mlir).

**`quantize` / `dequantize` are not divide-and-round-to-even.** Measured on the device (macOS 27.2.1, `tools/ane_limits.py`):
`quantize` computes `f16(x * f16(1 / scale))`, rounds half away from zero and saturates to [-128, 127]; `dequantize`
computes `q * scale` and rounds that to fp16, again with ties away from zero. That model equals the device on 16384 of
16384 random values (8 scales) and on all 128 exact ties of scale 0.125; round-half-even, which this translation used
before and which `math.roundeven` does, gets half of the ties wrong, and dividing by the scale instead of multiplying by
its fp16 reciprocal differs on 305 (fp16 divide) to 894 (f32 divide) of 24576 values. `arith.truncf` has a `to_nearest_away` mode, but the stock lowering ignores it and
rounds to even, so mil2mlir.py does the dequantize rounding on the bits: add half of the dropped 13 mantissa bits to the
magnitude, clear them, truncate (below 2^-14: round(p * 2^24) / 2^24). On AArch64 the quantize becomes `fdiv` once,
`fmul.8h`, `frinta.8h`, `fmaxnm`/`fminnm`, `fcvtzs.8h`, `xtn` (aarch64/asm.test).

**fp16 multiply and add round ties away from zero too.** A result exactly midway between two fp16 values goes to the one
of larger magnitude: 2048 of 2048 constructed ties for `mul` and for `add` (round-to-even would get about half).

**The W8A8 chain is close, not exact.** `tools/ane_w8a8.py`: three int8 convs (64 channels, 32x32), quantize between them,
against the device. With f32 accumulation 81% of the outputs are identical and 90% are within 1 ulp (weights |q| <= 9 and
<= 24; p99 73-108 ulp, worst 3700 ulp); with fp16 accumulation 4-22% and 12-37%. A numpy model of the chain agrees with the
f32-accumulation run to 0.1 percentage points. The 10% beyond 1 ulp is not explained.

## Device facts (M3 Pro)

`tools/ane_limits.py` re-measures these on whatever Mac runs it (about ten seconds; it prints `same` or `DIFFERS` against
the 27.2.1 column and exits with the number of differences). The 15.8.3 column is what was recorded there before; "not measured", "not tried" and "not recorded" claim nothing
about that OS, so those rows are not differences.

| fact                                                      | macOS 15.8.3 (recorded)                    | macOS 27.2.1 (measured)                              |
|-----------------------------------------------------------|--------------------------------------------|------------------------------------------------------|
| dynamic `matmul`, contracted dim 16384 / 16385            | compiles / fails                           | compiles / fails                                     |
| 1x1 `conv`, input channels (OC 64, SP 64)                 | 32000 compiles                             | 32512 compiles, 32513 fails                          |
| `conv` 32000 -> 768                                       | compiles at SP 256                         | fails at SP 64, compiles at SP 256                   |
| `constexpr_affine_dequantize`, parameters in `[...]`      | not tried with a correct header            | compiles and evaluates (int8 zp 0 and 3, uint8 zp 128) |
| same op, parameters as call arguments                     | InvalidMILProgram                          | InvalidMILProgram                                    |
| int8 weights as `blob` + `cast` + `mul` instead           | 5.1-5.4x slower than fp16                  | 5.7x (256 ch) to 20x (384, 512 ch) slower than fp16; the op above runs at 0.97-1.13x |
| blob chunk header as the benches write it (no size at +8, offset 0 at +16) | InvalidMILProgram, except at 512 channels | compiles, evaluates the wrong bytes |
| blob chunk header, payload size (+8) 0 or wrong           | not tested alone                           | accepted                                             |
| `conv` channels 64 ... 2048, correct header               | not recorded                               | all compile                                          |
| IOSurface rows (inputs and outputs)                       | not measured                               | stride = ceil(bytes per row / 64) * 64; dense rows evaluate wrongly or are refused |
| several weight files in one model                         | compiles                                   | compiles only if the bundle rule below holds         |
| `quantize` / `dequantize`, fp16 `mul` / `add` at ties     | rounding assumed, not measured             | half away from zero (above)                          |
| `test_weight_reload`: new weights without a recompile     | output unchanged                           | output unchanged                                     |
| classifier backward as a 32000-deep `matmul`              | fails to compile                           | fails to compile                                     |

The 15.8.3 "512 channels only" wall and "`constexpr_affine_dequantize` is rejected" came from the benches' blob header
(no data offset) and from one placement of the op's parameters (all 11 files that 15.8.3 rejected put them in the call
arguments; the form `ane_int8_bench.m` writes, in `[...]`, was only tried with the bad header). What differs on 27.2.1 is that
the benches' header no longer fails to compile: it runs on the wrong bytes. `inmem_bench.m`, `inmem_peak.m` and
`ane_int8_bench.m` therefore time kernels with garbage weights; the timings are unaffected (the FP16 line of `ane_int8_bench`,
128x conv 512ch 64x64, is 16.2 ms on both OSes), but the outputs are meaningless and none of the three looks at its output. The root project
knows the full header: all 15 other files on its `main` that write the blob magic (`bridge/ane_bridge.m` and 14 under
`training/`, found by grep) also write the data size at +8 and the data offset (128) at +16 of the chunk; only the five root-level
benchmarks that build weights themselves (`ane_int8_bench.m`, `inmem_bench.m`, `inmem_peak.m`, `sram_bench.m`, `sram_probe.m`)
write the short header, and merged PR #27, which rewrote the ones it touched for macOS 26, kept it. No issue or PR comment I
searched ties that header to the 512-channel reports.

**What did not change.** Same program, weights and input on both OSes (`test/device/macos15.8.3`, replayed by `ane_limits.py`):
the four conv programs there (fp16 weights with fp32 and with fp16 I/O, int8 weights as cast + mul, uint8 as cast, sub, mul)
write output surfaces byte-identical to the ones the 15.8.3 device wrote (4 x 2048 values), also with the conv op renamed to a
name never compiled before (the model identifier contains a hash of the MIL text, so that model is new to the compiler).
Also unchanged: the 24 cells of the accumulation table under Sharp corners; the compile/eval verdicts of the 27 saved int8,
LUT, quantize and relu programs that have a recorded 15.8.3 verdict (the recorded quantize runs used zero input, so those
compare verdicts only); and the losses of `training_dynamic/train` (9.1106 -> 7.4495).

**What differs, besides the bundle rule and the blob header.** The int8-as-cast+mul chain is 3.6-3.7x slower at 384 and 512
channels (128x conv 512ch: 84 ms recorded on 15.8.3, 310-314 ms in three runs on 27.2.1; the fp16 column is 15-16 ms on both)
and 1.09x slower at 256. The `train_large` losses agree to the printed 6 decimals for steps 0-9 (forward pass, one set of
weights) and differ by at most 0.013 after the first weight update; three runs on 27.2.1 are identical to each other. The
cause is not isolated: candidates are the static backward kernels, which were not compared on their own, and the CPU
gradient code (Accelerate).

**Not comparable.** Quantize and fp16 tie rounding were never measured on 15.8.3.

**The bundle rule.** A model built from several weight files (`weights = @{ "@model_path/weights/a.bin": ..., ... }`)
compiles on 27.2.1 only if the weight that ends up as the bundle's root file `data` is the weight whose name sorts last.
`data` is the last one in the dictionary's enumeration order, which is arbitrary and changes with the names. When the
two differ, ANECompilerService logs `verifyBundleAtPath: hash mismatch ... (TOCTOU?)` and the caller sees
`Code=10 "verifyBundleAtPath: invalid model"`. The rule was fitted on 41 name sets and then predicted, before compiling,
the outcome of 77 more (60 random name sets of the 6-weight attention kernel, the five static Stories110M kernels that have
weights, 12 two-conv models) with no miss; the mechanism is inferred, not read from Apple's code. Consequences: with the repo's file names the
attention forward and the FFN forward kernels break the rule, so `train_large`, `train_large_ane` and `train_opt` fail at
layer 0 on 27.2.1 and restart in a loop (the other four static kernels happen to satisfy it), while the dynamic trainer and
the kernels with one weight file are unaffected. Renaming the weight files
with a numeric suffix until the two coincide (a loop around `compile_kern_mil_w`, tried on a scratch copy of `stories_io.h`)
makes all three run.

## Not covered

- `scaled_dot_product_attention` and the other MIL emitted only by the `training/test_*.m` probes, `tiny_train*.m`,
  `inmem_*.m`, `sram_*.m` and bridge/ are not translated.
- The static (baked-weight) kernels run on the device only inside the training/ programs; they are not compared against
  the CPU translation. `train_opt` prints `loss=nan` from step 10 (right after its `exec()` restart) on 15.8.3 and on
  27.2.1; this is not an OS effect and is not investigated here.
- The device comparison is a separate, manual step, at the mid sizes: build tools/ane_probe.m with tools/mid_dynamic.h,
  then tools/ane_compare.py. The tiny sizes of test/small are not refused for their size: with the IOSurface rows staged at
  the 64-byte stride the 8x12x6 dynamic matmul evaluates correctly (`tools/ane_limits.py`); the probe's dense rows are
  what failed.
- CPU-side Accelerate code (vDSP RMSNorm, cross-entropy, Adam, cblas_sgemm gradients) is not translated.
- Nothing here is wired into training; the ObjC still builds MIL text itself.

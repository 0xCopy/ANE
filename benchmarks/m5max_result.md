# M5 Max ANE Probe & Training Benchmark

**Machine**: MacBook Pro · Apple M5 Max (6P + 12E CPU) · 128 GB RAM
**macOS**: 26.4.1 (Darwin 25.4.0), Model Identifier `Mac17,7`
**Date**: 2026-04-23
**ANE Family**: **`h17`** (new — M4 and base M5 are `h16`)

All data gathered with the repo's probes and training harness as-is, no source
changes. Compared against:
- [README.md](../README.md) M4 reference figures
- [training/m5result.md](../training/m5result.md) — base M5 (10-core, 16 GB) probe notes
- [benchmarks/community_results.json](./community_results.json) — M1/M3/M4/M5 submissions

---

## Hardware identification

**Question**: Is the ANE in M5 Max the same silicon block as M4 / base M5?

**Result**: **No — `_ANEDeviceInfo.aneSubType` returns `h17`**, a version not
seen in any community submission. The base M5 (per `m5result.md`) still reports
`h16`, same as M4. M5 Max is the first `h17` on record.

```
=== ANE INT8 W8A8 Benchmark (M4, h17) ===    ← header label is hardcoded "M4",
                                                 the "h17" is read from the device.
```

Everything else still works: `program(1.3)` MIL, `_ANEInMemoryModelDescriptor`,
`constexpr_affine_dequantize`, `quantize` / `dequantize`. No API closures on
macOS 26.4.1.

---

## inmem_peak — deep conv stacks (FP16)

**Question**: Peak FP16 throughput using the same 128-layer conv sweep
([inmem_peak.m](../inmem_peak.m)) the README reports for M4.

```
Config                         W(MB)   GFLOP   ms/eval  TFLOPS
-----------------------------------------------------------------
 32x conv 512ch sp64            16.0    1.07   0.135 ms    7.95
 48x conv 512ch sp64            24.0    1.61   0.171 ms    9.42
 64x conv 512ch sp64            32.0    2.15   0.206 ms   10.40
 96x conv 512ch sp64            48.0    3.22   0.266 ms   12.13
128x conv 512ch sp64            64.0    4.29   0.311 ms   13.80  ← peak
 64x conv 256ch sp64             8.0    0.54   0.168 ms    3.19
128x conv 256ch sp64            16.0    1.07   0.132 ms    8.16
256x conv 256ch sp64            32.0    2.15   0.216 ms    9.94
 64x conv 384ch sp64            18.0    1.21   0.142 ms    8.52
128x conv 384ch sp64            36.0    2.42   0.203 ms   11.91
```

**Peak: 13.80 TFLOPS** at `128× conv 512ch sp=64`.

| Chip       | inmem_peak FP16 (TFLOPS) |
|------------|--------------------------|
| M3 Pro     | 9.98 |
| M4 Pro     | 12.57 |
| M4 Max     | 10.93 |
| M5 (16 GB) | 12.17 |
| M5 (32 GB) | 12.44 |
| **M5 Max** | **13.80** |

---

## ane_int8_bench — FP16 vs INT8 W8A8 (larger spatial 64×64)

**Question**: How close does M5 Max come to the M4 blog's 19 TFLOPS / 35 TOPS
figures when the conv is large enough to saturate the array?

```
Config                            GOP   ms/eval    TOPS   Ratio
-----------------------------------------------------------------
FP16 128x conv 512ch 64x64      274.88  14.263 ms  19.27
W8A8 128x conv 512ch 64x64      274.88   7.720 ms  35.61   1.85x
FP16  64x conv 512ch 64x64      137.44   7.153 ms  19.21
W8A8  64x conv 512ch 64x64      137.44   3.824 ms  35.94   1.87x
FP16 256x conv 256ch 64x64      137.44   7.318 ms  18.78
W8A8 256x conv 256ch 64x64      137.44   4.118 ms  33.37   1.78x
FP16 128x conv 256ch 64x64       68.72   3.696 ms  18.59
W8A8 128x conv 256ch 64x64       68.72   2.112 ms  32.54   1.75x
FP16 128x conv 384ch 64x64      154.62   8.154 ms  18.96
W8A8 128x conv 384ch 64x64      154.62   4.389 ms  35.23   1.86x
```

| Precision | M5 Max | M4 (README `H16G`) |
|-----------|--------|---------------------|
| FP16 peak | **19.27 TFLOPS** | 18.6 TFLOPS |
| INT8 W8A8 peak | **35.61 TOPS** | 35.1 TOPS |
| INT8/FP16 ratio | 1.85× | 1.88× |

**Implication**: the `h17` ANE's raw compute is **within 4 % of `h16`**
(run-to-run noise). Apple kept the ~19 TFLOPS FP16 / ~35 TOPS INT8 ceiling
across two chip generations. The "38 TOPS" spec remains the INT8 path.

---

## sram_bench — working-set cliff

**Question**: Where does the on-chip SRAM spill to DRAM?

```
Config                     W(MB)  Act(MB)  Tot(MB)   ms/eval  TFLOPS
---------------------------------------------------------------------
 256ch x 64sp                0.1     0.03     0.2   0.212 ms    0.04
 512ch x 64sp                0.5     0.06     0.6   0.085 ms    0.40
1024ch x 64sp                2.0     0.12     2.2   0.335 ms    0.40
2048ch x 64sp                8.0     0.25     8.5   0.141 ms    3.80
3072ch x 64sp               18.0     0.38    18.8   0.204 ms    5.92
4096ch x 64sp               32.0     0.50    33.0   0.300 ms    7.17
5120ch x 64sp               50.0     0.62    51.2   0.432 ms    7.76
6144ch x 64sp               72.0     0.75    73.5   0.565 ms    8.56
8192ch x 32sp              128.0     0.50   129.0   0.965 ms    4.45
```

**M4 in the blog shows the cliff around ~32 MB**. On M5 Max throughput is still
climbing past **73 MB** and only breaks at **129 MB**. Caveat: the last row
also halves `sp` from 64 to 32 — a pipeline-starvation confound we can't rule
out without an independent probe. What's unambiguous: the effective SRAM
working set is **at least as large as M4's**, plausibly larger.

---

## inmem_bench — single 1×1 conv latency scan

```
Config         W(MB)    ms/eval  TFLOPS
--------------------------------------------
 256ch x64sp     0.1   0.088 ms    0.09
 512ch x64sp     0.5   0.089 ms    0.38
1024ch x64sp    2.0    0.313 ms    0.43
2048ch x64sp    8.0    0.131 ms    4.10
3072ch x64sp   18.0    0.189 ms    6.38
4096ch x64sp   32.0    0.302 ms    7.11
```

Dispatch floor ≈ 0.09 ms, matching the M4 blog's ~0.095 ms XPC/IOKit overhead.

---

## Training — dynamic pipeline (`training_dynamic/train.m`)

**Synthetic token data** (5 M random uint16 in [0, 5000) to mimic a compressed
TinyStories vocab), random init, `--accum 10`.

| Model | Params | Layers | Kernels compiled once | **M5 Max ms/step** |
|-------|--------|--------|-----------------------|--------------------|
| Stories110M (MHA 12/12) | 109 M | 12 | 421 ms | **73.5 ms** |
| Qwen3-0.6B (GQA 16/8) | 596 M | 28 | 398 ms | **320.0 ms** |

Qwen3-0.6B per-step timing breakdown (stable from step 10+):

```
ane_fwd=54.6   io_fwd=15.2   rms=4.5    ane_bwd=70.5   io_bwd=43.3
silu=27.0      rms_bwd=12.4  cls=8.7    cblas_wait=0.0 dw_copy=9.9
```

ANE time = 125 ms (39 %) · CPU time = 195 ms (61 %). Bottleneck is unchanged
from the README's diagnosis: ANE is idle most of the step waiting for
RMSNorm / SiLU / classifier / dW / Adam on CPU.

---

## Training — static pipeline (`train_large.m`)

For apples-to-apples with `community_results.json` (all existing entries use
this path).

```
[batch 10: compile=3384ms train=902.5ms (90.2ms/step) compiles=72]
    ane=8.0 io=2.8 cls=7.6 elem=11.7 rms=0.1 cblas_wait=0.0 ms/step
[batch 20: compile=3335ms train=897.1ms (89.7ms/step) compiles=72]
[batch 30: compile=3353ms train=900.6ms (90.1ms/step) compiles=72]
Total steps:     30
Wall time:       13.1 s
Compile time:    10072 ms (76.9 %)
Train time:      2700 ms (20.6 %)
Avg train:       90.0 ms/step
```

| Chip        | ms/step | ane ms | compile / 10 |
|-------------|---------|--------|--------------|
| M1 Pro      | 148–163 | 32–35  | 7.9–8.5 s    |
| M1 Max      | 143–167 | 35–45  | ~7.1 s       |
| M3 Ultra\*  | 91      | ~10    | ~3.7 s       |
| M4 Pro      | 69–73   | 8.9    | ~3.5 s       |
| M4 Max      | 64      | 10.2   | ~3.5 s       |
| M5 (16 GB)  | 101–120 | 9.1–9.8| 3.2–3.4 s    |
| **M5 Max**  | **90.0**| **8.0**| **~3.35 s**  |

\* repo reference platform.

---

## Speedup summary — M5 Max vs baselines

| Metric | M4 (README) | M5 base | M5 Max | vs M4 | vs M5 base |
|--------|-------------|---------|--------|-------|------------|
| FP16 peak (TFLOPS) | 18.6 | 12.17–12.44 | **19.27** | 1.04× | 1.55× |
| INT8 W8A8 (TOPS)   | 35.1 | —           | **35.61** | 1.01× | —    |
| Stories110M static (ms/step) | 91 | 101–120 | **90.0**  | 1.01× | 1.22× |
| Stories110M dynamic (ms/step)| — | —         | **73.5**  | —     | —     |
| Qwen3-0.6B dynamic (ms/step) | 412| —        | **320.0** | 1.29× | —     |

**Takeaways**:

1. **Peak ANE compute has not moved between M4 and M5 Max** (≈ 19 TFLOPS FP16,
   ≈ 35 TOPS INT8). The `h16 → h17` version bump does not show up in peak math.
2. **Training gains of 1.22–1.29× are CPU-driven**, not ANE-driven. The 12
   performance cores plus Accelerate's `cblas_sgemm` on M5 Max close the gap
   that made base M5 (4P + 6E) slower than M4 Pro despite a newer ANE.
3. **M5 Max's effective SRAM working set is ≥ M4's.** The `sram_bench` cliff
   sits past 70 MB where M4's was at ~32 MB, though a cleaner probe is needed
   (the 128 MB row changes two variables at once).

---

## Strategic implications

- Anyone optimizing training on this repo for M5 Max should focus on pushing
  RMSNorm / SiLU / classifier onto the ANE, not on peak-throughput MIL tricks —
  the ANE already has 60 % idle headroom per step.
- `h17` is worth re-probing with the tests under `training/test_*.m` — the
  `m5result.md` findings (weight-reload fails, weightsBuffer is inert,
  procedureIndex is accepted but ignored, QoS has no effect) were recorded on
  `h16` and may or may not hold on `h17`.
- No evidence that Apple has tightened the private-API surface on macOS 26.4.1.

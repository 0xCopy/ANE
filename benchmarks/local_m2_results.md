# Local Apple M2 ANE Benchmark Results

Date: 2026-06-05

Host:

- Mac mini (Mac14,3), Apple M2, 8-core CPU (4 performance + 4 efficiency), 8 GB memory
- macOS 26.5, build 25F71
- Apple clang 21.0.0 (clang-2100.1.1.101)
- ANE subtype reported by the private benchmark API: h14
- Repository commit tested: d91c9845c0784dec7753048954fc6d0e8411fe29 (`origin/main`)

Runtime notes:

- Results were measured from a clean worktree at `/private/tmp/ANE-m2-clean` to avoid local code changes affecting benchmark numbers.
- Benchmarks were run sequentially on the same machine because parallel ANE workloads contend for the accelerator and produce outliers.
- Tables below use the median of three runs where repeated. Raw logs were kept locally under `/private/tmp/ane_m2_2026-06-05_clean/`.
- Qwen3-0.6B dynamic training was not run on this 8 GB M2 machine; resident fp32 weights, gradients, Adam state, activations, transposed buffers, and IOSurfaces are expected to be memory-heavy.
- Two upstream output quirks were observed but not fixed in this results-only report: `inmem_peak` prints an invalid `%peak` value, and `ane_int8_bench` labels the h14 run as `M4`. The tables below use the raw TFLOPS/TOPS and host metadata instead.

## In-Memory Baseline

Command:

```bash
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -o inmem_bench inmem_bench.m
./inmem_bench
```

Median of three runs:

| Config | Weight MB | ms/eval | TFLOPS |
|---|---:|---:|---:|
| 256ch x 64sp | 0.1 | 0.136 | 0.06 |
| 512ch x 64sp | 0.5 | 0.140 | 0.24 |
| 1024ch x 64sp | 2.0 | 0.204 | 0.66 |
| 2048ch x 64sp | 8.0 | 0.358 | 1.50 |
| 3072ch x 64sp | 18.0 | 0.552 | 2.19 |
| 4096ch x 64sp | 32.0 | 0.909 | 2.36 |

## Peak-Style Conv Chain

Command:

```bash
xcrun clang -O2 -fobjc-arc -framework Foundation -framework CoreML \
  -framework IOSurface -ldl -o inmem_peak inmem_peak.m
./inmem_peak
```

Median of three runs. `% peak` below is computed against the M2 15.8 TFLOPS FP16 reference; the current program output prints an invalid `%peak` column.

| Config | Weight MB | GFLOP | ms/eval | TFLOPS | % peak |
|---|---:|---:|---:|---:|---:|
| 32x conv 512ch sp64 | 16.0 | 1.07 | 0.239 | 4.50 | 28.5 |
| 48x conv 512ch sp64 | 24.0 | 1.61 | 0.280 | 5.74 | 36.3 |
| 64x conv 512ch sp64 | 32.0 | 2.15 | 0.301 | 7.13 | 45.1 |
| 96x conv 512ch sp64 | 48.0 | 3.22 | 0.404 | 7.98 | 50.5 |
| 128x conv 512ch sp64 | 64.0 | 4.29 | 0.537 | 7.99 | 50.6 |
| 64x conv 256ch sp64 | 8.0 | 0.54 | 0.160 | 3.35 | 21.2 |
| 128x conv 256ch sp64 | 16.0 | 1.07 | 0.222 | 4.84 | 30.6 |
| 256x conv 256ch sp64 | 32.0 | 2.15 | 0.340 | 6.32 | 40.0 |
| 64x conv 384ch sp64 | 18.0 | 1.21 | 0.245 | 4.94 | 31.3 |
| 128x conv 384ch sp64 | 36.0 | 2.42 | 0.345 | 7.00 | 44.3 |

## INT8 W8A8

Command:

```bash
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench
```

Median of three runs:

| Config | Precision | Weight MB | GOP | ms/eval | TOPS | Ratio |
|---|---|---:|---:|---:|---:|---:|
| 128x conv 512ch 64x64 | FP16 | 64.0 | 274.88 | 22.847 | 12.03 | - |
| 128x conv 512ch 64x64 | W8A8 | 32.0 | 274.88 | 23.046 | 11.93 | 1.01x |
| 64x conv 512ch 64x64 | FP16 | 32.0 | 137.44 | 12.442 | 11.05 | - |
| 64x conv 512ch 64x64 | W8A8 | 16.0 | 137.44 | 11.861 | 11.59 | 1.06x |
| 256x conv 256ch 64x64 | FP16 | 32.0 | 137.44 | 12.984 | 10.59 | - |
| 256x conv 256ch 64x64 | W8A8 | 16.0 | 137.44 | 13.272 | 10.36 | 1.05x |
| 128x conv 256ch 64x64 | FP16 | 16.0 | 68.72 | 6.801 | 10.10 | - |
| 128x conv 256ch 64x64 | W8A8 | 8.0 | 68.72 | 6.348 | 10.83 | 1.11x |
| 128x conv 384ch 64x64 | FP16 | 36.0 | 154.62 | 14.220 | 10.87 | - |
| 128x conv 384ch 64x64 | W8A8 | 18.0 | 154.62 | 13.770 | 11.23 | 1.08x |

On this M2, W8A8 is approximately parity to a modest improvement for these kernels, not the larger M4 speedup reported upstream.

## Dynamic Matmul

Command:

```bash
cd training
xcrun clang -O2 -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc \
  -o test_dynamic_matmul test_dynamic_matmul.m \
  -framework Foundation -framework CoreML -framework IOSurface -ldl -framework Accelerate
./test_dynamic_matmul
```

Result:

- 64x64 identity correctness: PASS, max error 0.001938
- 64x64 scale-by-2 correctness: PASS, ratio 2.000
- 768x768x256 single dynamic matmul: 1.012 ms/eval, 298.4 GFLOP/s
- With weight IO: 0.871 ms/eval, 346.8 GFLOP/s
- vs `cblas_sgemm`: PASS, max error 0.014646

Tiled 768x768 matmul:

| tile_oc | tiles | compile ms | eval ms | GFLOP/s |
|---:|---:|---:|---:|---:|
| 64 | 12 | 543 | 4.318 | 69.9 |
| 128 | 6 | 260 | 1.752 | 172.4 |
| 256 | 3 | 110 | 1.041 | 290.1 |
| 384 | 2 | 69 | 0.871 | 346.8 |
| 768 | 1 | 47 | 0.652 | 463.0 |

## Dynamic Training

Data:

```bash
cd training
bash download_data.sh
```

The local run used `tinystories_data00.bin`: 20,658,981 tokens, 41.3 MB.

Build and run:

```bash
cd training/training_dynamic
make MODEL=stories110m
./train --scratch --steps 20 --accum 10 --warmup 2 --data ../tinystories_data00.bin
```

Result:

- Model: Stories110M, 109.5M parameters
- Active compact vocab: 9,205 tokens from 32,000
- One-time compile: 1,196 ms for 10 kernels
- Train time: 31,081 ms total
- Average train: 1,554.0 ms/step
- Wall time: 68.0 s
- Step 0 loss: 9.1105
- Step 10 loss: 8.6389

Notes:

- This is the dynamic weight pipeline, not the static pipeline used by the main cross-generation training table.
- The measured dynamic training run was IO-dominated on this 8 GB M2 machine.

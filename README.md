# Running Transformers on Apple's Neural Engine

You might be asking, "why the FUCK would you pick GPT2?"

## Project Scope & Intent

I'm genuinely grateful for all the attention this project has received — I never expected a weekend research hack to blow up like this. Thank you to everyone who starred, forked, ran benchmarks on their own hardware, and shared the work. It means a lot.

That said, I want to set clear expectations about what this project is and isn't.

This is a **research project**, not a production framework.

The goal was to demonstrate that **training on the Apple Neural Engine — and potentially other NPUs — is possible**, and that the barrier has always been software support, not hardware capability. The ANE is a remarkably capable piece of silicon that Apple restricts to inference-only use through CoreML. This project bypasses that restriction using reverse-engineered private APIs to show what's possible when you give the hardware a chance.

### What This Project Is

- A proof of concept for ANE training via `_ANEClient` and `_ANECompiler` private APIs
- A set of benchmarks documenting real ANE performance characteristics (throughput, power, SRAM behavior)
- A reference for anyone exploring direct ANE access outside CoreML
- Research code that I update when I find something interesting

### What This Project Is Not

- A maintained framework or library
- A replacement for CoreML, MLX, llama.cpp, or any production inference stack
- A path to training large models on consumer hardware (yet)

### On The Hype

Some coverage of this project has overstated its implications. To be clear:

- Training works, but utilization is low (~5-9% of peak) with significant engineering challenges remaining
- Many element-wise operations still fall back to CPU
- This does **not** replace GPU training for anything beyond small research models today

The honest results — including all limitations — are documented in the accompanying articles:
- [Part 1: Reverse Engineering](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Part 2: Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Part 3: Training](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-c8b)

### On Maintenance

---

That said:
- I'll keep pushing updates when I discover something interesting
- Bug fixes and benchmark contributions (especially on hardware I don't own) are welcome
- Feature requests will likely go unaddressed — but feel free to fork
- PRs will be merged at a relatively slow pace, otherwise I become the bottleneck for community growth around this tech

This fork extends the original project with:

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it. If in future, community decides to maintain one source of truth repo, I'm in full support of that.

---

## What is this?

A from-scratch implementation of transformer training (forward + backward pass) running on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS FP16 (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Current results:**

| Model | Params | ms/step | Pipeline |
|-------|--------|---------|----------|
| Stories110M (12L, dim=768, MHA 12/12) | 109M | **91 ms** | Dynamic (no recompile) |
| Qwen3-0.6B (28L, dim=1024, GQA 16/8) | 596M | **412 ms** | Dynamic (no recompile) |

- All forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume via exec() restart
- GQA (Grouped-Query Attention) support with per-head tiling/reduction
- GPU↔ANE zero-copy pipeline via shared IOSurface (GPU prefill → ANE decode)

**INT8 W8A8 quantization — 1.88x throughput (M4, H16G):**

| Config | FP16 | INT8 W8A8 | Speedup |
|--------|------|-----------|---------|
| 128x conv 512ch 64x64 | 18.6 TOPS, 14.8ms | 35.1 TOPS, 7.8ms | **1.88x** |
| 64x conv 512ch 64x64 | 18.4 TOPS, 7.5ms | 34.1 TOPS, 4.0ms | **1.85x** |

INT8 activations halve L2 SRAM bandwidth between tiles via MIL `quantize`/`dequantize` ops. Weights use `constexpr_affine_dequantize` (int8 stored, fp16 at compile time).
Have you read the art bro? Have you? Nah. I doubt it.

GPT2 had more soul in its theoretical pinky finger than all of us combined.

But I digress..

Running transformer inference and training directly on Apple's Neural Engine via reverse-engineered private APIs. No CoreML, no Metal, no GPU — pure ANE compute through `_ANEInMemoryModel` and MIL programs compiled at runtime.

The dynamic pipeline uses shared ANE kernels with weights packed into spatial dimensions (no recompilation when weights change):

**MHA models (Stories110M) — 6 kernels per layer:**

| Kernel | Function |
|--------|----------|
| `sdpaFwd` | QKV projection + SDPA + output projection |
| `ffnFused` | SwiGLU FFN (W1, W3, SiLU, W2) |
| `ffnBwdW2t` / `ffnBwdW13t` | FFN backward (split for memory) |
| `sdpaBwd1` / `sdpaBwd2` | SDPA backward |

**GQA models (Qwen3-0.6B) — 10 kernels per layer:**
Adds separate `woFwd`, `qBwd`, `kvBwd` kernels for grouped-query attention (Q_DIM ≠ DIM).

CPU handles: RMSNorm forward/backward, residual connections (DeepNet α scaling), loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer updates.
Forked from [maderix/ANEtransformers](https://github.com/maderix/ANEtransformers) which demonstrated ANE training (Stories110M, 12-layer forward+backward on ANE). This fork extends the project with GPT-2 inference on ANE, systematic M5 hardware investigation, and fused kernel optimization.

## What's Here

### GPT-2 Inference on ANE (`training/gpt2.m`)

Complete GPT-2 (124M) inference engine. Two-phase architecture:

1. **ANE prefill** — Full sequence processed on ANE using fused attention and FFN kernels. One fused attention kernel (QKV + multi-head SDPA + causal mask + softmax + output projection) and one fused FFN kernel (W1 + GELU + W2) per layer, compiled per sequence-length bucket (32, 64, 128, 256, 512, 1024). Embedding and LayerNorm on CPU with Accelerate.

2. **CPU decode with KV cache** — Single-token generation runs entirely on CPU using NEON fp16 matmul (4-row unrolled), bypassing ANE dispatch overhead. LM head via GCD-parallel NEON fp16 over 50,257 vocab rows.

Includes a from-scratch BPE tokenizer (`gpt2_tokenizer.h`) and weight converter (`gpt2_convert.py`) that pulls weights from HuggingFace with no PyTorch dependency.

```bash
cd training

# Download and convert weights
pip install safetensors huggingface_hub
python3 gpt2_convert.py

# Build and run
make gpt2
./gpt2 --prompt "The meaning of life is" --tokens 100 --temp 0.8
```

### ANE Training (upstream)

Key optimizations:
- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms to 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **Process restart** — bypasses ~119 ANE compile limit per process via checkpoint and re-launch

## File Structure

```
├── api_exploration.m           # Initial ANE API discovery
├── inmem_basic.m               # In-memory MIL compilation proof-of-concept
├── inmem_bench.m               # ANE dispatch latency benchmarks
├── inmem_peak.m                # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m                # ANE SRAM bandwidth probing
├── sram_probe.m                # SRAM size/layout exploration
├── quant_probe.m               # Quantization probe: int8/int4 affine, blockwise and LUT weights
├── inference/                  # LLM inference on the ANE (Qwen), with Metal GPU and Q4 paths
├── mlir/                       # MLIR translations of the kernels, the `ane` dialect, the device op sweep
├── contrib/kmp-ane/            # Kotlin Multiplatform ANE scaffold (MNIST demo)
├── community_benchmarks/       # Standardized community benchmark output
├── docs/                       # Security audit reports, development diaries, plans
├── benchmarks/
│   ├── ANE_BENCHMARK_REPORT.md # Cross-chip benchmark report
│   └── community_results.json  # Community-submitted benchmark data
├── bridge/
│   ├── Makefile
│   ├── ane_bridge.h            # ANE bridge header
│   └── ane_bridge.m            # ANE bridge implementation
└── training/
    ├── Makefile
    ├── README.md               # Training pipeline documentation
    ├── ane_runtime.h           # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_mil_gen.h           # MIL program generation helpers
    ├── ane_classifier.h        # ANE-offloaded classifier
    ├── ane_rmsnorm_bwd.h       # ANE RMSNorm backward pass
    ├── model.h                 # Model weight initialization and blob builders
    ├── forward.h               # Forward pass MIL generators
    ├── backward.h              # Backward pass MIL generators
    ├── stories_config.h        # Stories model configuration
    ├── stories_cpu_ops.h       # CPU-side operations for Stories model
    ├── stories_io.h            # Stories data I/O (TinyStories loading)
    ├── stories_mil.h           # Stories MIL program generators
    ├── train.m                 # Minimal training loop (early prototype)
    ├── tiny_train.m            # 2-layer tiny model training
    ├── train_large.m           # Main: Stories110M training (static pipeline)
    ├── train_large_ane.m       # Stories110M training (ANE classifier)
    ├── test_*.m                # Unit tests for individual kernels
    ├── dashboard.py            # Live training dashboard (power, throughput)
    ├── tokenize.py             # TinyStories pretokenization script
    ├── download_data.sh        # Training data download script
    └── training_dynamic/       # Dynamic pipeline (no recompilation)
        ├── Makefile
        ├── config.h
        ├── cpu_ops.h
        ├── io.h
        ├── mil_dynamic.h
        └── train.m
```

## Training Data

Training requires pretokenized TinyStories data. To download:
```bash
cd training && bash download_data.sh
```
See [training/README.md](training/README.md) for detailed training instructions.

## Building

Requires macOS 15+ on Apple Silicon (tested on M1 through M5).

```bash
# Dynamic pipeline (recommended) — model selected at build time
cd training/training_dynamic
make MODEL=stories110m    # Stories110M (12L, MHA, 109M params)
make MODEL=qwen3_06b      # Qwen3-0.6B (28L, GQA, 596M params)
./train --scratch          # train from random init
./train --resume           # resume from checkpoint

# Static pipeline (legacy — recompiles weights each step)
cd training && make train_large
./train_large ane_stories110M_ckpt.bin 256 100 1e-4

# INT8 benchmark
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
  -o ane_int8_bench ane_int8_bench.m
./ane_int8_bench

# Bridge library (C-callable ANE API)
cd bridge && make
cd training

# GPT-2 inference
make gpt2
./gpt2 --prompt "Once upon a time" --tokens 200

# Stories110M training (upstream)
make train_large
./train_large

# Benchmarks
make bench_fused
./bench_fused
```

## Known ANE Constraints

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk mlmodelc needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16 or fp32; fp16 direct I/O is ~37% faster)
4. **Dynamic weights** — Activations and weights packed into a single spatial input dimension, sliced apart inside the MIL kernel. Weights change without recompilation.
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas
6. **INT8 quantization** — `constexpr_affine_dequantize` for int8 weights, `quantize`/`dequantize` between layers for int8 activation caching in L2 SRAM (1.88x throughput)

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T (ANE) → mask+softmax (CPU) → scores@V (ANE)
- **~119 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart with checkpoint
- **FP16 gradient underflow** — backward matmuls underflow in fp16; fixed with global loss scaling (`256 * NLAYERS`)
- **Single-input constraint** — multi-input ANE requests cause 0x1d error; inputs packed into spatial dimension instead

## Performance

**Training throughput (M4):**

| Model | Params | ms/step | Layers | Kernels/layer |
|-------|--------|---------|--------|---------------|
| Stories110M | 109M | 91 ms | 12 | 6 (MHA) |
| Qwen3-0.6B | 596M | 412 ms | 28 | 10 (GQA) |

**ANE peak throughput (M4, H16G):**

| Precision | Peak TOPS | Config |
|-----------|-----------|--------|
| FP16 | 18.6 | 128x conv 512ch 64x64 |
| INT8 W8A8 | 35.1 | 128x conv 512ch 64x64 |

**GPU↔ANE inference pipeline (M4, seq=256):**

| Model | GPU Prefill | ANE Decode | Total |
|-------|------------|------------|-------|
| Stories110M | 6.7ms | 1.9ms | 8.8ms |
| Qwen3-0.6B | 9.7ms | 2.3ms | 12.0ms |
- **Weights are immutable after compile** — no hot-swap, no `weightsBuffer` override, no file-swap reload
- **SDPA causal masking ignored by hardware** — must decompose into Q@K^T + mask add + softmax + scores@V
- **~119 compile limit per process** — ANE compiler leaks resources; training uses `exec()` restart
- **IOSurface minimum 49152 bytes** — even for tiny tensors (seq=1 decode)
- **Spatial stride padded to 32** — `[1, C, 1, W]` surfaces have stride `max(W, 32)`
- **Chaining API inaccessible** — `_ANEChainingRequest` validates but driver rejects (Error Code=15)

*Note: Above numbers are for single-layer training. Full 12-layer training runs at 62-72 ms/step.*

## Community Benchmarks

Community hardware benchmark submissions live in [`benchmarks/submissions/`](benchmarks/submissions/).

- [Mac Studio (Apple M3 Ultra, 256 GB) — 2026-03-03](benchmarks/submissions/m3-ultra-mac-studio-2026-03-03/README.md)


## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA section 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## Hardware Characterization: Apple M5 (2026)

The M5 (Apple 10 family) introduces specific ANE behavioral constraints that differ from earlier M-series chips. This section documents the key findings from reverse-engineering efforts.

### Benchmark Methodology

**Hardware Configuration:**
- **Chip**: Apple M5 (base model, 16 NE cores)
- **macOS Version**: 26.3 (25D125) (Darwin 25.3.0)
- **Date Measured**: 2026-03-01
- **ANE Family**: H16 (same as M4)

**Measurement Approach:**
- Peak throughput measured using 4096×4096 dynamic matmul operations via the [`m5_performance_suite.m`](training/m5_performance_suite.m) benchmark tool
- Weight update latency measured as `memcpy` to IOSurface + ANE evaluation
- All IOSurface buffers use 128-byte alignment (required for M5 ANE compatibility)
- 1000 iterations per measurement after 10-iteration warmup
- FLOPS calculated as `2 × dim × dim` (multiply-add per output element)

**Important Notes:**
- M5 Pro and M5 Max variants have **not yet been benchmarked** — results may differ
- The Fusion Architecture in Pro/Max models may change ANE behavior

### Key M5 ANE Constraints

| Constraint | Value | Notes |
|:---|:---|:---|
| **IOSurface Alignment** | 128 bytes | All input, output, and weight surfaces must be 128-byte aligned. Failure results in silent evaluation errors or compiler rejection. |
| **MIL Version** | program(1.5) | M5 is optimized for MIL 1.5 using static `BLOBFILE` weights. However, **any dynamic weight injection via input tensors must use `program(1.3)` and `<ios17>`** to bypass strict AST compiler validations. |
| **Max Dynamic Dimension** | 4096 × 4096 | Maximum dimension for dynamic weight tensors passed as inputs. |
| **Peak Throughput** | ~1.7 TFLOPS | Pure ANE compute for 4096-dim matmul operations (measured: 1.66-1.76 TFLOPS). |
| **Update Latency** | ~1.27 ms | CPU-to-IOSurface `memcpy` + ANE eval for weight updates at 4096 dims. |

### Dynamic Weight Injection

On M5, the traditional approach of baking weights into the compiled model (via `BLOBFILE`) does not support runtime updates—the ANE snapshots weights into private memory at load time. The only viable path for real-time weight updates is:

**Treat weights as Input Tensors using the `matmul` operator.**

```objc
// MIL pattern for dynamic weights (M5 compatible)
// Input 0: activations [1, 1, SEQ, IC]
// Input 1: weights [1, 1, IC, OC]  ← dynamic!
// Output:  [1, 1, SEQ, OC]

NSString *mil = [NSString stringWithFormat:
    @"program(1.3)\n"
    "{\n"
    "    func main<ios17>(tensor<fp32, [1, 1, %d, %d]> x, tensor<fp32, [1, 1, %d, %d]> weights) {\n"
    "        // Cast to fp16, matmul, cast back to fp32\n"
    "    } -> (y);\n"
    "}\n", seq, ic, ic, oc];
```

This approach enables:
- **Zero-copy weight swapping**: Update weights via `memcpy` into the input IOSurface
- **~100x faster updates** vs. recompile-and-load cycle (1.8ms vs 40-170ms)
- **On-device training**: Foundation for gradient descent on ANE

### M5 Performance Benchmarks

Run the benchmark suite:

```bash
cd training
make m5_performance_suite
./m5_performance_suite
```

Expected output on M5 (measured on base M5, macOS 26.3):

```
Max Dynamic Dimension:     4096 x 4096
Peak Throughput:           1.02 TFLOPS
Weight Update Latency:     1.78 ms
Max Weight Tensor Size:    67.11 MB
```

> **Note**: These values are from actual M5 hardware measurements. M5 Pro/Max variants have not yet been tested — results may differ.

### Implementation Notes

1. **Alignment Helper**: Use `ane_create_surface()` which automatically applies 128-byte alignment—backward compatible with M3/M4.

2. **MIL Generation**: Use `mil_gen_dynamic_matmul()` from `ane_mil_gen.h` for M5-compatible dynamic weight layers.

3. **Weight Surface**: For large weights (>16MB), use `ane_create_weights_surface()` which adds `kIOSurfaceIsGlobal` for ANE hardware access.

4. **Matmul vs Conv**: For dynamic weights, `matmul` is more stable than `conv` on M5 due to flexible hardware tiling on the NCE (Neural Compute Engine).

---

## License

MIT — see [LICENSE](LICENSE)

---

*Built by a human + Claude, one weekend at a time.*



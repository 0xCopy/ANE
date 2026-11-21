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

I don't intend to grow this into a large community project. My focus is on original research (compiler infrastructure for edge AI optimization), and maintaining an open-source framework takes time away from that.

That said:
- I'll keep pushing updates when I discover something interesting
- Bug fixes and benchmark contributions (especially on hardware I don't own) are welcome
- Feature requests will likely go unaddressed — but feel free to fork
- PRs will be merged at a relatively slow pace, otherwise I become the bottleneck for community growth around this tech

### Fork it, build on it

This is MIT licensed for a reason. Everyone now has access to AI-assisted development tools that can adapt and extend code in hours. If this project is useful to you — take it, modify it, build something better. If you do something cool with it, I'd love to hear about it.If in future, community decides to maintain one source of truth repo, I'm in full support of that.

---

## What This Is

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

The original [maderix](https://github.com/maderix/ANEtransformers) work: training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on ANE. 12-layer forward+backward pass, 6 ANE kernel types per layer (72 kernels/step), 107 ms/step on M4. Adam optimizer, gradient accumulation, checkpoint/resume via `exec()` restart. See `training/train_large.m` and the [training README](training/README.md).

### M5 Hardware Investigation

Systematic probing of ANE behavior on Apple M5 (H16 family, same as M4). Key findings documented in [`training/m5result.md`](training/m5result.md):

| Question | Result |
|----------|--------|
| Can weights be swapped without recompile? | **No.** Weights baked at compile time. File overwrite + reload ignored. |
| Does `weightsBuffer` IOSurface override compiled weights? | **No.** Same output regardless. |
| Does QoS affect ANE frequency? | **No.** All QoS 0-63 work. Fixed frequency, no latency difference. |
| Can `_ANEChainingRequest` chain kernels without CPU round-trips? | **Validates but rejected by driver** (Error Code=15). Likely requires entitlements only CoreML holds. |
| Can `_ANEPerformanceStats` expose hardware counters? | Class exists with `hwExecutionTime` but requires factory construction via model `perfStatsMask`. |
| Real-time eval path? | `beginRealTimeTask` returns NO (needs entitlement). `evaluateRealTimeWithModel` works but no perf gain. |

### Fused Kernel Benchmarks (`training/bench_fused.m`)

Quantifies the value of operation fusion on ANE:

- **Dispatch overhead**: ~60-80 us per sequential ANE dispatch
- **Fused vs separate**: 1.5-3x speedup from fusing multiple convolutions into single MIL programs
- **Conclusion**: Fused MIL is the only viable path to high ANE utilization. Chaining API is inaccessible without Apple-internal entitlements. Intermediates stay in ANE SRAM when fused.

Full investigation: [`training/CHAINING_INVESTIGATION.md`](training/CHAINING_INVESTIGATION.md)

## How It Works

1. **MIL generation** — Objective-C constructs MIL program text at runtime: convolutions (linear layers), matmul (attention), softmax, element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + fp16 weight blobs directly to ANE programs, no disk `.mlmodelc` needed
3. **IOSurface I/O** — Tensors via IOSurface shared memory in `[1, C, 1, S]` channel-first fp16 format. Spatial dimension padded to minimum stride of 32. Minimum surface size 49152 bytes.
4. **Weight baking** — Weights compiled as BLOBFILE constants into MIL programs. No runtime weight update possible — must recompile to change weights.
5. **NEON fp16** — ARM NEON intrinsics for fast fp32-fp16 conversion at IOSurface boundaries and for CPU-side matmul during decode

## File Structure

```
├── api_exploration.m           # Initial ANE API discovery
├── inmem_basic.m               # In-memory MIL compilation proof-of-concept
├── inmem_bench.m               # ANE dispatch latency benchmarks
├── inmem_peak.m                # Peak TFLOPS measurement (2048x2048 matmul)
├── ane_int8_bench.m            # INT8 W8A8 vs FP16 throughput benchmark
├── sram_bench.m                # ANE SRAM bandwidth probing
├── sram_probe.m                # SRAM size/layout exploration
├── gpu_ane_share.m             # GPU↔ANE zero-copy IOSurface demo
├── gpu_prefill_ane_decode.m    # GPU prefill → ANE decode pipeline
├── bridge/
│   ├── ane_bridge.h            # C-callable ANE API (compile, eval, I/O)
│   ├── ane_bridge.m            # Bridge implementation (int8 + fp16 weight blobs)
│   └── Makefile
└── training/
    ├── ane_runtime.h           # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_classifier.h        # Classifier fwd (32K conv), softmax, rmsnorm on ANE
    ├── train_large.m           # Static pipeline (weights as constants, recompiles)
    ├── training_dynamic/
    │   ├── train.m             # Dynamic training loop (model-agnostic)
    │   ├── config.h            # Derived sizes, structs, alloc helpers
    │   ├── mil_dynamic.h       # MIL generators for dynamic weight kernels (GQA-aware)
    │   ├── io.h                # IOSurface I/O, weight staging, GQA tile/reduce
    │   ├── models/
    │   │   ├── stories110m.h   # Stories110M config (12L, MHA)
    │   │   └── qwen3_06b.h    # Qwen3-0.6B config (28L, GQA)
    │   └── Makefile
    ├── dashboard.py            # Live training dashboard (blessed TUI)
├── api_exploration.m               # Initial ANE private API discovery
├── inmem_basic.m                   # In-memory MIL compilation proof-of-concept
├── inmem_bench.m                   # ANE dispatch latency benchmarks
├── inmem_peak.m                    # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m                    # ANE SRAM bandwidth probing
├── sram_probe.m                    # SRAM size/layout exploration
└── training/
    ├── gpt2.m                      # GPT-2 124M inference: ANE prefill + CPU KV-cache decode
    ├── gpt2_convert.py             # HuggingFace → ANE weight converter (no PyTorch)
    ├── gpt2_tokenizer.h            # Self-contained BPE tokenizer (header-only C)
    ├── bench_fused.m               # Fused vs separate kernel benchmarks
    ├── CHAINING_INVESTIGATION.md   # Full chaining API reverse-engineering writeup
    ├── m5result.md                 # M5 ANE probe results
    ├── train_large.m               # 12-layer Stories110M ANE training (upstream)
    ├── stories_config.h            # Training model config and structs
    ├── stories_io.h                # Training IOSurface I/O and kernel compile/eval
    ├── stories_mil.h               # Training MIL generators (6 kernel types)
    ├── stories_cpu_ops.h           # vDSP RMSNorm, cross-entropy, Adam, embeddings
    ├── dashboard.py                # Training TUI: loss curves, power, text generation
    ├── ane_runtime.h               # ANE private API wrapper
    ├── ane_mil_gen.h               # MIL generation helpers
    ├── test_chaining.m             # Chaining API experiments
    ├── test_weight_reload.m        # Weight swap without recompile test
    ├── test_ane_advanced.m         # weightsBuffer, procedureIndex, shared events probe
    ├── test_qos_sweep.m            # QoS 0-63 latency sweep
    ├── test_perf_stats.m           # _ANEPerformanceStats introspection
    ├── test_decode_attn.m          # Multi-input decode attention kernel validation
    ├── test_multi_input.m          # Multi-input IOSurface size constraints
    ├── test_ffn_seq1.m             # FFN at seq=1 (decode mode) validation
    ├── test_lm_head_ane.m          # LM head on ANE feasibility
    ├── test_lm_head_fast.m         # LM head CPU benchmark (6 approaches)
    ├── test_lm_head_neon.m         # LM head NEON fp16 benchmark
    ├── docs/                       # Roadmap: GPT-2 XL, streaming, sampling, interactive
    └── Makefile
```

## Training Data

Training requires pretokenized TinyStories data. To download:
```bash
cd training && bash download_data.sh
```
See [training/README.md](training/README.md) for detailed training instructions.

## Building

Requires macOS 15+ on Apple Silicon (tested on M4, M5). No external dependencies — system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

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

## Disclaimer

This project uses Apple's private, undocumented APIs (`_ANEClient`, `_ANECompiler`, `_ANEInMemoryModelDescriptor`). These APIs are not covered by any public stability guarantee and may change or break with any macOS update. This is independent research into Apple Neural Engine architecture, using APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA §1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)

---

*Built by a human + Claude, one weekend at a time.*



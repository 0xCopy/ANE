<<<<<<< HEAD
# ANE Training — On-Device Training on Apple Neural Engine

Training transformer models directly on Apple's Neural Engine using private ANE APIs. Supports multiple architectures including GQA (Grouped-Query Attention).

![Dashboard](dashboard.gif)

## Supported Models

| Model | Layers | Heads (Q/KV) | Dim | Hidden | Params | ms/step |
|-------|--------|--------------|-----|--------|--------|---------|
| Stories110M | 12 | 12/12 (MHA) | 768 | 2048 | 109M | ~115 |
| Qwen3-0.6B | 28 | 16/8 (GQA) | 1024 | 3072 | 596M | ~412 |

Model configs live in `training_dynamic/models/*.h`. To add a new model, create a header with the architecture defines (see below).

## Architecture

- **SDPA causal mask workaround**: ANE hardware ignores attn_mask — decompose into Q@K^T (ANE conv) + mask+softmax (CPU) + scores@V (ANE conv)
- **GQA support**: K/V heads tiled to match Q heads for SDPA, reduced back after backward pass

## Three Training Pipelines

### 1. Static Baseline (`train_large`)
Original pipeline. Weights baked as constants in MIL kernels — recompile every 10 steps via `exec()` restart.

- 60 weight-bearing + 12 weight-free kernels = 72 per compile batch
- Classifier + softmax + RMSNorm backward on CPU
- **106.7 ms/step**, 7.6s compile per restart

### 2. Static + ANE Extras (`train_large_ane`) — PR#19
Offloads classifier forward (32K conv), softmax, final RMSNorm, and RMSNorm backward to ANE. Bridge API for C-callable ANE access.

- 86 kernels per compile batch (+24 rmsnorm_bwd, +1 classifier, +1 finalRms)
- **91.8 ms/step** (14% faster), 9.6s compile per restart
- Use `--no-ane-extras` to disable and fall back to CPU (for debugging)

### 3. Dynamic Weight Pipeline (`training_dynamic/`)
Weights passed via IOSurface spatial dimension — compile 10 kernels once at startup, no recompilation needed. Supports multiple models via `make MODEL=xxx`.

- 10 shared kernels across all layers (GQA-aware: split sdpaFwd/woFwd, split qBwd/kvBwd)
- **~115 ms/step** (Stories110M) / **~412 ms/step** (Qwen3-0.6B), 0.4s one-time compile
- No exec() restart, no compile limit issues

## Performance Comparison (20 Steps)

| | Static Baseline | PR#19 + ANE extras | PR#19 no extras | Dynamic |
|---|---|---|---|---|
| **Wall time** | **10.1s** | **11.7s** | **10.7s** | **~2.6s** |
| Compile | 7.6s (75.7%) | 9.6s (81.6%) | 7.5s (69.7%) | 0.4s (15%) |
| Train | 2.1s (21.2%) | 1.8s (15.6%) | 2.9s (27.4%) | 2.2s (85%) |
| **ms/step** | **106.7** | **91.8** | **147.0** | **111** |
| Kernels/restart | 72 | 86 | 60 | 9 (once) |
| ANE TFLOPS | 0.87 | 1.15 | 0.72 | — |
| Total TFLOPS | 1.63 | 1.90 | 1.19 | — |

**Key insights:**
- Dynamic wins on wall time for any practical run length (3.9x faster at 20 steps)
- PR#19 has the best per-step throughput (92ms) but compile overhead dominates short runs
- Static restarts every 10 steps, so dynamic's zero-recompile advantage compounds
=======
# ANE Training & SDK — General-Purpose Neural Engine Platform
Training a 109M-parameter Llama2-architecture transformer (Stories110M) directly on Apple's Neural Engine. This repository has evolved into a fully-featured **ANE SDK** for developing and training arbitrary neural network architectures on Apple Silicon.

![Dashboard](dashboard.gif)

## 🚀 The ANE SDK
The ANE SDK provides a high-level API for defining, training, and benchmarking models on the Neural Engine without manual MIL (Model Intermediate Language) string concatenation.

### Key Features
- **Modular Layer Library**: High-level builders for NLP and Vision (`Linear`, `Conv2D`, `LayerNorm`, `Softmax`, etc.).
- **Graph Orchestration**: Automatic activation chaining and IOSurface management via a `Sequential` model container.
- **Weights-as-Tensors**: Every layer utilizes a zero-recompile optimization pattern, allowing dynamic weight updates for training.
- **Native Performance**: Sustained throughput of **>90 TFLOPS** across modular components.

### Architecture Comparison

| Specialized (Legacy) | ANE SDK (General-Purpose) |
|----------------------|---------------------------|
| **Fixed Topology**: Transformer only | **Dynamic Topology**: Arbitrary layers |
| **Manual I/O**: Manual surface pointers | **Automated Chaining**: Sequential runner |
| **Hardcoded MIL**: `stories_mil.h` | **Modular MIL**: `layers/core.h`, `layers/cnn.h` |
| **Optimized Path**: Hand-tuned SDPA | **Ease of Use**: PyTorch-like API |

## Performance (Optimized)

| Metric | Value |
|-----------|---------------|
| **Training Latency** | **~79.6 ms/step** |
| **Inference Latency (SEQ=256)** | **0.60 ms** |
| **Sustained ANE Throughput** | **~94.4 TFLOPS** |
| **Theoretical Inference TPS** | **~429,000 Tokens/sec** |
| **Weight Sync** | ~3.4 ms per layer (NEON-accelerated) |
| **Compile Budget** | **0 restarts** (Dynamic weight updates) |
>>>>>>> e113fae6832b497b79e1b32f0861b1797e007f53

## Configuration Variables

<<<<<<< HEAD
| File | Description |
|------|-------------|
| `train_large.m` | Static baseline — 72 kernels, classifier/softmax on CPU |
| `train_large_ane.m` | PR#19 — 86 kernels, classifier/softmax/rmsnorm_bwd on ANE |
| `training_dynamic/train.m` | Dynamic pipeline — 10 kernels, weights via IOSurface |
| `training_dynamic/mil_dynamic.h` | MIL generators for dynamic weight kernels (GQA-aware) |
| `training_dynamic/config.h` | Derived sizes, structs, alloc helpers (model-agnostic) |
| `training_dynamic/models/*.h` | Per-model configs (stories110m.h, qwen3_06b.h) |
| `training_dynamic/io.h` | IOSurface I/O, weight staging, GQA tile/reduce |
| `training_dynamic/cpu_ops.h` | CPU ops (SiLU backward, cross-entropy, Adam) |
| `stories_config.h` | Static pipeline config, structs, alloc helpers |
| `stories_io.h` | IOSurface I/O, NEON fp16 conversion, kernel compile/eval |
| `stories_mil.h` | MIL generators for static pipeline (6 kernel types) |
| `stories_cpu_ops.h` | vDSP-vectorized RMSNorm, cross-entropy, Adam |
| `ane_classifier.h` | ANE classifier fwd (32K conv), softmax kernels |
| `ane_rmsnorm_bwd.h` | ANE rmsnorm backward kernel |
| `dashboard.py` | TUI dashboard — loss curve, power/CPU/memory graphs |
| `Makefile` | Build targets |
| `training_dynamic/generate.m` | Text generation (inference) — forward-only, 3 kernels |

## Usage

### 1. Download Training Data
=======
Most configuration is handled in [stories_config.h](stories_config.h) and [train_large.m](train_large.m).

### Model Hyperparameters (`stories_config.h`)
- `DIM`: Model dimension (default: 768)
- `HIDDEN`: FFN hidden dimension (default: 2048)
- `NLAYERS`: Number of transformer layers (default: 12)
- `VOCAB`: Vocabulary size (default: 5000)
- `SEQ`: Sequence length / context window (default: 256)

### Training Paths (`train_large.m`)
- `DATA_PATH`: Path to the tokenized binary dataset (default: `tinystories_data00.bin`)
- `MODEL_PATH`: Path to the initial pretrained weights in llama2.c format.
- `CKPT_PATH`: Output path for training checkpoints.

## Compiling & Running

### 1. Prerequisites
Ensure you have a modern Mac with Apple Silicon (M1/M2/M3/M4). 
You will need `xcrun` (Xcode Command Line Tools) and various Python dependencies for data prep and monitoring.
>>>>>>> e113fae6832b497b79e1b32f0861b1797e007f53

### 2. Prepare Data
The trainer expects a flat binary file of `uint16_t` token IDs.
```bash
<<<<<<< HEAD
bash download_data.sh
```

Downloads pretokenized TinyStories (Llama 2 BPE, 32K vocab) from HuggingFace. Produces `tinystories_data00.bin` (~41 MB, ~20M tokens).

All training pipelines perform token-data validation at startup:
- token file byte length must align to 16-bit token boundaries
- token file must contain at least `SEQ+1` tokens
- every token id must be within `[0, vocab_size)`

### 2. Build & Train

```bash
# Static baseline (classifier + softmax on CPU)
make train_large
./train_large stories110M.bin 256 100 1e-4
./train_large --model stories110M.bin --steps 100 --lr 1e-4
./train_large --data ./tinystories_data00.bin --steps 100 --lr 1e-4

# PR#19: ANE-offloaded classifier + softmax + rmsnorm_bwd
make train_large_ane
./train_large_ane stories110M.bin 256 100 1e-4
./train_large_ane --no-ane-extras --steps 100    # disable ANE extras
./train_large_ane --data ./tinystories_data00.bin --steps 100 --lr 1e-4

# Dynamic pipeline (model selected at build time)
cd training_dynamic
make MODEL=qwen3_06b           # default — Qwen3-0.6B (28L, GQA, 596M)
make MODEL=stories110m         # Stories110M (12L, MHA, 109M)
./train --scratch              # train from random init
./train --resume               # resume from checkpoint
./train --scratch --steps 200 --lr 1e-4  # custom steps/lr from random init
```

**CLI flags (`train_large` / `train_large_ane`):**
- `--steps N` (default 10000)
- `--lr F` (default 3e-4)
- `--model PATH` — pretrained weights file
- `--data PATH` — tokenized TinyStories `.bin` file (default: `tinystories_data00.bin`)
- `--ckpt PATH` — checkpoint file (preserved across exec() restarts)
- `--resume` — resume from checkpoint
- `--no-ane-extras` — (train_large_ane only) disable ANE classifier/softmax/rmsnorm_bwd

**CLI flags (`training_dynamic/train`):**
- `--scratch` — start from random initialization
- `--resume` — resume from the model-specific checkpoint, if present
- `--steps N`, `--lr F`, `--accum N`, `--warmup N`, `--clip F`
- `--data PATH` — tokenized TinyStories `.bin` file
- `--help` — print usage

The dynamic pipeline does not load pretrained weights directly yet; use `--scratch` for a new run or `--resume` after a checkpoint exists.

### 3. Generate Text (Inference)

After training, generate text from a checkpoint using the dynamic pipeline's forward kernels:

```bash
cd training_dynamic
make MODEL=stories110m generate    # build for Stories110M
make MODEL=qwen3_06b generate      # build for Qwen3-0.6B

# Greedy decoding (deterministic)
./generate --ckpt ane_stories110M_dyn_ckpt.bin --tokens 100

# Temperature sampling with top-k
./generate --ckpt ane_stories110M_dyn_ckpt.bin --tokens 200 --temp 0.8 --topk 40

# Use a different data file for prompt context
./generate --ckpt ane_stories110M_dyn_ckpt.bin --data ../tinystories_data00.bin --tokens 100
```

**CLI flags (`training_dynamic/generate`):**
- `--ckpt PATH` — checkpoint file (required, from `train --resume`)
- `--tokens N` — number of tokens to generate (default: 100)
- `--temp F` — sampling temperature, 0 = greedy (default: 0.0)
- `--topk K` — top-K sampling, 0 = disabled (default: 0)
- `--data PATH` — tokenized data for prompt context (default: `../tinystories_data00.bin`)
- `--seed N` — random seed for sampling (default: 42)

The generator loads checkpoint weights (skipping Adam state), compiles only the 3 forward kernels (sdpaFwd, woFwd, ffnFused), and runs a sliding-window forward pass. Prompt context is taken from the first `SEQ` tokens of the data file. A llama2.c tokenizer (`tokenizer.bin`) placed in the working directory enables text output; without it, token IDs are printed.

For detokenization, place a llama2.c format `tokenizer.bin` in the working directory. It can be obtained from the [llama2.c tokenizer.bin](https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin) (Llama 2, 32K vocab) or the Qwen3 tokenizer for the 152K vocab model.

### 4. Monitor with Dashboard

```bash
pip install blessed psutil numpy
sudo python3 dashboard.py          # static pipeline
sudo python3 dashboard.py --dynamic # dynamic pipeline
```

### 5. Benchmarking

All programs print an **Efficiency Report** at completion:

```
=== Efficiency Report ===
Total steps:     20
Wall time:       11738 ms (11.7 s)
Compile time:    9583 ms (81.6%)
Train time:      1835 ms (15.6%)
Avg train:       91.8 ms/step
ANE TFLOPS:      1.15 sustained
```

## Adding a New Model

Create `training_dynamic/models/mymodel.h`:

```c
#pragma once
#define MODEL_NAME "MyModel-1B"

#define DIM 2048        // model hidden dim
#define HIDDEN 5504     // FFN intermediate dim
#define HEADS 32        // number of query heads
#define KV_HEADS 8      // number of KV heads (= HEADS for MHA)
#define HD 64           // head dim (can differ from DIM/HEADS)
#define SEQ 256         // sequence length
#define NLAYERS 22      // number of transformer layers
#define VOCAB 32000     // vocabulary size

#define CKPT_PATH "ane_mymodel_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "../tinystories_data00.bin"
```

Everything else is derived automatically: `GQA_RATIO`, `Q_DIM`, `KV_DIM`, weight sizes, IOSurface layouts, MIL kernels.

Build with: `make MODEL=mymodel`

**Constraints:**
- `HEADS` must be divisible by `KV_HEADS`
- `HD` is explicit (not necessarily `DIM/HEADS` — Qwen3 uses HD=128 with DIM/HEADS=64)
- For MHA (no GQA), set `KV_HEADS = HEADS`

## Key Techniques

- **NEON vectorized fp16↔fp32**: ARM NEON intrinsics for fast IOSurface data transfer
- **vDSP cross-entropy**: `vDSP_mtrans` + `vvexpf` + `vDSP_sve` — 8x faster than scalar
- **Async weight gradients**: cblas_sgemm dispatched to background queue, overlapped with ANE
- **Vocab compaction** (dynamic): 32K–152K → 9.2K active tokens, up to 16.5x reduction in classifier work
- **Dynamic weight packing**: Activations + weights concatenated in IOSurface spatial dimension — one kernel serves all layers
- **GQA tile/reduce**: K/V tiled from KV_HEADS→HEADS on CPU before SDPA backward, gradients reduced HEADS→KV_HEADS after
- **exec() restart**: Workaround for ANE ~119 compile limit per process
=======
# Tokenize raw text into the expected format
python3 tokenize.py
```

### 3. Build and Train
```bash
# Compile the training binary
make train_large

# Start training (fresh start or default steps)
./train_large

# Resume with custom steps and learning rate
./train_large --resume --steps 1000 --lr 1e-4
```

## Dataset Adaptation

To adapt this trainer to any custom text dataset:
1. **Tokenize**: Use a tokenizer to convert your text corpus into a sequence of IDs.
2. **Export**: Save the IDs as a raw binary file of `uint16_t` values.
3. **Configure**: Update `VOCAB`, `SEQ`, and `DATA_PATH` in the config files to match your dataset.
4. **Compile**: Re-run `make train_large`. The ANE kernels will automatically adjust to your new shapes.

## Monitoring with Dashboard

The TUI dashboard provides real-time telemetry on loss, power usage, and model generation.
```bash
pip install blessed psutil numpy
# Dashboard may require sudo for powermetrics access
python3 dashboard.py --resume
```

## Testing the Model

You can test the trained model using the standalone inference script. It uses standard vanilla NumPy to perform the forward pass on the CPU, making it easy to inspect.

### Generate Text
```bash
# Test with a custom prompt and checkpoint
python3 sample.py --prompt "Once upon a time" --ckpt ane_stories110M_ckpt.bin --steps 100
```

### Parameters
- `--prompt`: The starting text for generation.
- `--ckpt`: Path to the training checkpoint (`.bin`).
- `--vocab`: Path to the BPE vocabulary (`vocab.json`).
- `--steps`: Maximum number of tokens to generate.
- `--temp`: Sampling temperature (default 0.8).

## ANE SDK Usage

You can build arbitrary models using the modular layer library in `layers/`.

### 1. Define Model Architecture
```objectivec
#import "layers/anesdk.h"

// Define layers
ANESDKLayer l1 = anesdk_linear_create("fc1", 768, 2048, 256);
ANESDKLayer l2 = anesdk_relu_create("relu1", 2048, 1, 256);
ANESDKLayer l3 = anesdk_layernorm_create("ln1", 2048, 256);

// Assemble into Sequential model
ANESDKLayer layers[] = { l1, l2, l3 };
ANESDKModel model = anesdk_model_sequential_create(layers, 3);
```

### 2. Run Forward Pass
The SDK automatically manages IOSurface chaining between layers.
```objectivec
// Write input to the first layer
io_write_fp16(model.layers[0].kern->inputs[0], input_data, 768, 256);

// Run the whole graph on ANE
anesdk_model_forward(&model);

// Read result from the last layer
io_read_fp16(model.layers[2].kern->ioOut, output_data, 0, 2048, 256);
```

### 3. Automated Verification
The repository includes a regression suite that verifies both the legacy Transformer and your new SDK layers.
```bash
# Build and run all tests (Fast SDK tests -> Training -> Inference)
make regression
```

---

## Performance Utilities

### ANE Hardware Benchmark
To measure raw hardware throughput and verify the **Weights-as-Tensors** optimization, use the native C-based benchmark:
```bash
make benchmark_ane
./benchmark_ane
```
Average Forward Pass (SEQ=256): **0.60 ms** | Throughput: **~94.4 TFLOPS**.

### Model Inference Utility (`sample.py`)
Verify trained checkpoints on the CPU using vanilla NumPy.
```bash
python3 sample.py --prompt "Once upon a time" --ckpt ane_stories110M_ckpt.bin
```

---

## Key Optimization: Weights as Tensors

Previously, ANE training required recompiling kernels every time weights changed, hitting an OS-enforced 119-compile limit. 

The current implementation defines weights as formal function parameters (`tensor<fp16, [dim, dim]>`) in the MIL program. This allows us to:
1. Compile the kernel logic **once**.
2. Update weights between batches by writing directly to **IOSurfaces** via NEON-accelerated loops (`io_write_fp16_t`).
3. Maintain resident memory for the model, eliminating the need for `exec()` restarts.
>>>>>>> e113fae6832b497b79e1b32f0861b1797e007f53

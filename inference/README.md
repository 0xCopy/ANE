# ANE Inference — Full LLM on Apple Neural Engine

<<<<<<< HEAD
First complete LLM inference running directly on Apple's Neural Engine via reverse-engineered `_ANEClient` APIs. No CoreML. No Xcode compiler dependency at runtime. Token-for-token match with PyTorch.
=======
First complete LLM inference running directly on Apple's Neural Engine via reverse-engineered `_ANEClient` APIs. No CoreML. No Xcode compiler dependency at runtime.
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf

Built on top of the [maderix/ANE](https://github.com/maderix/ANE) training runtime.

## What This Does

<<<<<<< HEAD
Runs **Qwen2.5-0.5B-Instruct** (24 transformer layers, 494M parameters) entirely on the ANE:

- **169 ANE kernels** compiled at startup via `_ANEInMemoryModel`
- **82 tokens/sec** decode on M4 Pro
- **Zero GPU usage** — runs on 16 dedicated neural cores
- **Correct output** — matches PyTorch reference token-for-token

All linear projections (Q, K, V, O, gate, up, down × 24 layers + chunked LM head) compile as baked-weight 1×1 convolution kernels on ANE. Element-wise ops (RMSNorm, RoPE, softmax, SiLU, attention scores) run on CPU via Accelerate BLAS.
=======
Runs **Qwen2.5-0.5B-Instruct** (24 transformer layers, 494M parameters) on ANE:

- **169 ANE kernels** compiled at startup via `_ANEInMemoryModel`
- **~60 tokens/sec** decode on M4 Max
- **Pure C HTTP API** — no Python needed for serving
- **BPE tokenizer in C** — send plain text, get plain text back
- **~6s cold start**, then instant responses in server mode

## Quick Start (One Command)

```bash
cd inference
./setup.sh
```

This automatically:
1. Creates a Python venv and installs dependencies
2. Downloads [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) from HuggingFace (~953 MB)
3. Converts BF16 safetensors to f32 binary format (~1.9 GB)
4. Builds the `qwen_ane` binary
5. Runs a smoke test

After setup, you're ready to go.

## HTTP API (Recommended)

The fastest way to use inference. Single process, zero Python overhead.

```bash
# Start server (compiles 169 ANE kernels on first launch, ~6s)
./qwen_ane qwen05b.bin --http 8000

# Query with plain text — tokenization happens in C
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_tokens": 50}'
```

Response:
```json
{
  "text": "2+2 equals 4.",
  "prompt_tokens": 29,
  "gen_tokens": 8,
  "prefill_tps": 66.2,
  "decode_tps": 57.3,
  "elapsed_s": 0.608
}
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/completions` | Generate text from a prompt |
| GET | `/health` | Server status check |

### POST /v1/completions

```json
{
  "prompt": "Your question here",
  "max_tokens": 50,
  "system": "You are a helpful assistant."
}
```

- `prompt` (required): The user message
- `max_tokens` (optional, default 50, max 512): Maximum tokens to generate
- `system` (optional): System prompt override

### Options

```bash
# Custom port
./qwen_ane qwen05b.bin --http 9000

# Custom model directory (for tokenizer files)
./qwen_ane qwen05b.bin --http 8000 --model-dir /path/to/Qwen2.5-0.5B-Instruct
```

Default model directory: `~/models/Qwen2.5-0.5B-Instruct`

## Other Modes

### Socket server (for programmatic access)

```bash
# Terminal 1: start server
./qwen_ane qwen05b.bin --server /tmp/qwen_ane.sock

# Terminal 2: query with run.py (auto-detects socket)
python3 run.py "What is 2+2?"

# Or query directly with nc
echo '{"tokens": [151644, 8948, 198], "max_tokens": 50}' | nc -U /tmp/qwen_ane.sock
```

### Stdin server (for piping/scripting)

```bash
./qwen_ane qwen05b.bin --server
# Send space-separated token IDs, pipe char separates max_tokens:
# 151644 8948 198 2610 525|20
```

### Single-shot (no server)

```bash
# Raw token IDs
./qwen_ane qwen05b.bin "151644 8948 198 2610 525 264 10950 17847 13" 20

# With Python tokenizer
python3 run.py "Say hello in one word."
```

### Python API server (alternative)

If you prefer Python for the HTTP layer:

```bash
./qwen_ane qwen05b.bin --server /tmp/qwen_ane.sock
python3 api_server.py --port 8000
```

## Throughput Benchmark

Run the standardized benchmark to measure your hardware's performance:

```bash
./benchmark.sh
```

This runs 5 prompts of varying length, measures prefill and decode tokens/sec in server mode, tests cold start latency, and checks decode speed consistency.

Sample output (M4 Max, 128 GB):
```
Prompt        Input Output Prefill(t/s)  Decode(t/s)  Latency(ms)
──────────────────────────────────────────────────────────────────
tiny             23     10         53.7         53.6          632
short            29      8         66.2         49.5          628
medium           33     84         63.4         55.3         2064
long             36    200         66.4         54.5         4235
stress          122     11         58.6         58.5         2303
──────────────────────────────────────────────────────────────────
Average                            61.7         54.3

Cold start (single-shot): ~6.2s (includes ANE kernel compilation)
```

Results are saved to `benchmark_results.json` for programmatic use.

### Compare with LM Studio

The benchmark script prints instructions for running the same prompts in LM Studio:

1. Download [LM Studio](https://lmstudio.ai)
2. Search for and download **Qwen2.5-0.5B-Instruct** (GGUF Q4_K_M or Q8_0)
3. Load the model, start the server (Developer tab, port 1234)
4. Run the same prompts and compare tokens/sec:

```bash
curl http://localhost:1234/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen2.5-0.5b-instruct","system_prompt":"You are a helpful assistant.","input":"What is 2+2?"}'
```

Note: LM Studio uses quantized GGUF weights (CPU/GPU) while we use full BF16 precision on the Neural Engine.

## Performance

| Mode | First prompt | Subsequent prompts |
|------|-------------|-------------------|
| Single-shot | ~6s | ~6s (recompiles each time) |
| Server (socket/HTTP) | ~6s (startup) | ~0.5s |
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf

## Architecture

```
<<<<<<< HEAD
Token → Embedding (CPU) → 24× Transformer Layer → LM Head (CPU) → Next Token
                              │
                              ├── RMSNorm (CPU)
                              ├── Q/K/V Projection (ANE conv kernel)
                              ├── RoPE (CPU, rotate_half)
                              ├── GQA Attention (CPU, 14 heads / 2 KV heads)
                              ├── O Projection (ANE conv kernel)
                              ├── Residual (CPU)
                              ├── RMSNorm (CPU)
                              ├── Gate/Up Projection (ANE conv kernel)
                              ├── SiLU + elementwise mul (CPU)
                              ├── Down Projection (ANE conv kernel)
                              └── Residual (CPU)
```

## Quick Start

```bash
# 1. Convert weights from HuggingFace safetensors to flat binary
pip install safetensors torch transformers
python3 convert_weights.py /path/to/Qwen2.5-0.5B-Instruct qwen05b.bin

# 2. Build
xcrun clang -O2 -framework Foundation -framework IOSurface \
  -framework CoreML -framework Accelerate -ldl -lobjc \
  -o qwen_ane main.m

# 3. Run (pass space-separated token IDs)
./qwen_ane qwen05b.bin "151644 8948 198 2610 525 264 10950 17847 13" 20

# 4. With tokenizer (requires transformers)
python3 run.py "Say hello in one word."
```

## Output

```
=== Qwen2.5-0.5B ANE Inference ===

Loading weights...
Config: dim=896 hidden=4864 layers=24 heads=14 kv_heads=2 vocab=151936
Compiling ANE kernels (169 total)...
Compile time: 5.1s

Prompt: 28 tokens, generating up to 10
Prefill: 64.2 t/s (28 tokens)
OUT: 9707 13 151645
Decode: 82.4 t/s (2 tokens)

→ "Hello."  (matches PyTorch exactly)
=======
Token -> Embedding (CPU) -> 24x Transformer Layer -> LM Head (CPU) -> Next Token
                              |
                              +-- RMSNorm (CPU)
                              +-- Q/K/V Projection (ANE conv kernel)
                              +-- RoPE (CPU, rotate_half)
                              +-- GQA Attention (CPU, 14 heads / 2 KV heads)
                              +-- O Projection (ANE conv kernel)
                              +-- Residual (CPU)
                              +-- RMSNorm (CPU)
                              +-- Gate/Up Projection (ANE conv kernel)
                              +-- SiLU + elementwise mul (CPU)
                              +-- Down Projection (ANE conv kernel)
                              +-- Residual (CPU)
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf
```

## Files

| File | What |
|------|------|
<<<<<<< HEAD
| `qwen_ane_infer.h` | Full 24-layer transformer forward pass, ANE kernel compilation, KV cache |
| `main.m` | Weight loader, token I/O, main generation loop |
| `convert_weights.py` | HuggingFace safetensors → flat f32 binary (includes Q/K/V biases) |
| `run.py` | Python wrapper with HuggingFace tokenizer |

## Model Support

Currently implements **Qwen2.5** architecture:
- GQA attention (grouped-query, `n_heads` ≠ `n_kv_heads`)
- `rotate_half` RoPE (not interleaved pairs)
- SwiGLU FFN (gate + up + silu + down)
- Q/K/V bias (Qwen-specific)
- Tied word embeddings (lm_head = embed)
- Chunked LM head (vocab > 65536 exceeds ANE max dim)

Adapting to other architectures (LLaMA, Gemma, Mistral) requires:
1. Adjusting the config constants in `qwen_ane_infer.h`
2. Updating `convert_weights.py` for the weight naming scheme
3. Removing Q/K/V bias handling if the model doesn't have them
4. Switching RoPE to interleaved pairs if needed
=======
| `setup.sh` | One-command setup: downloads model, converts weights, builds binary |
| `benchmark.sh` | Throughput benchmark with LM Studio comparison |
| `main.m` | Entry point: weight loader, server modes, HTTP API |
| `qwen_ane_infer.h` | Full 24-layer transformer forward pass, ANE kernel compilation, KV cache |
| `tokenizer.h` | BPE tokenizer in C: vocab/merge loading, encode/decode, chat template |
| `http_server.h` | Minimal HTTP/1.1 server: TCP, request parsing, JSON responses |
| `convert_weights.py` | HuggingFace safetensors to flat f32 binary |
| `run.py` | Python wrapper with HuggingFace tokenizer (auto-connects to socket server) |
| `api_server.py` | Python HTTP API bridge to socket server (alternative to C HTTP) |

## Model

**[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)**

- 494M parameters, BFloat16
- 24 layers, 896 dim, 4864 hidden
- 14 attention heads, 2 KV heads (GQA)
- 151,936 vocab size
- Download: `setup.sh` handles this automatically
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf

## Requirements

- macOS 15+ on Apple Silicon (M1/M2/M3/M4)
<<<<<<< HEAD
- Xcode Command Line Tools (for `xcrun clang`)
- Python 3.9+ with `safetensors`, `torch`, `transformers` (for weight conversion)

## Known Limitations

- **CPU projections only** — ANE baked-weight conv kernels compile successfully but produce incorrect output (FP16 weight blob format mismatch). The `USE_ANE_PROJECTIONS` toggle exists but defaults to 0 (CPU via Accelerate BLAS). Fixing this would push decode speed from 82 t/s to 120+ t/s.
- **No persistent server** — each invocation recompiles 169 kernels (~5s). A server mode that compiles once and serves via HTTP would eliminate this overhead.
- **Single model** — hardcoded for Qwen2.5-0.5B. Needs parameterization for other sizes.
- **f32 weights** — 1.9GB on disk. FP16 or quantized weight support would halve this.

## How It Works

The key insight from maderix's reverse engineering: the ANE executes compiled MIL (Machine Learning Intermediate Language) programs as atomic graph operations. Each linear projection becomes a MIL program with baked FP16 weights, compiled in-memory via `_ANEInMemoryModel`, and executed through IOSurface-based zero-copy I/O.

We chain 169 of these atomic operations (7 per transformer layer + 16 LM head chunks) with CPU-side element-wise ops in between. The ANE handles the compute-heavy matmuls; the CPU handles the memory-bound operations (attention scores, softmax, RoPE).
=======
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.11+ (for weight conversion only, not needed for serving)

## Known Limitations

- **CPU projections only** — ANE baked-weight conv kernels compile but produce incorrect output (FP16 weight blob format mismatch). `USE_ANE_PROJECTIONS` defaults to 0 (CPU via Accelerate BLAS). Fixing this would increase decode speed significantly.
- **Single model** — hardcoded for Qwen2.5-0.5B. Other sizes need config changes.
- **f32 weights** — 1.9GB on disk. FP16 weight support would halve this.
- **Single-threaded HTTP** — handles one request at a time. Sufficient for local use.
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf

## License

Same as maderix/ANE — research and educational use.

#!/usr/bin/env python3
"""Run Qwen2.5-0.5B on ANE with proper tokenization.

<<<<<<< HEAD
Usage:
    python3 run.py "Your prompt here" [--max-tokens 50]
"""
import argparse
import ctypes
import struct
=======
Auto-connects to a running socket server for instant responses (~0ms startup).
Falls back to subprocess mode if no server is running (~6s startup per call).

Usage:
    python3 run.py "Your prompt here" [--max-tokens 50]

Server mode (start server first in another terminal):
    ./qwen_ane qwen05b.bin --server /tmp/qwen_ane.sock
    python3 run.py "Your prompt here"
"""
import argparse
import json
import os
import socket
import subprocess
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf
import sys
import time
from pathlib import Path

INFERENCE_DIR = Path(__file__).parent
WEIGHTS_PATH = INFERENCE_DIR / "qwen05b.bin"
MODEL_DIR = Path.home() / "models" / "Qwen2.5-0.5B-Instruct"
<<<<<<< HEAD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--max-tokens", type=int, default=50)
=======
DEFAULT_SOCK = "/tmp/qwen_ane.sock"


def query_socket(token_ids: list[int], max_tokens: int, sock_path: str = DEFAULT_SOCK) -> dict | None:
    """Send a request to the socket server. Returns parsed JSON or None on failure."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(120)
        s.connect(sock_path)
        req = json.dumps({"tokens": token_ids, "max_tokens": max_tokens}) + "\n"
        s.sendall(req.encode())

        data = b""
        while True:
            chunk = s.recv(131072)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        s.close()
        return json.loads(data.decode().strip())
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return None


def query_subprocess(token_ids: list[int], max_tokens: int) -> dict | None:
    """Fall back to spawning the binary as a subprocess."""
    binary = str(INFERENCE_DIR / "qwen_ane")
    if not os.path.exists(binary):
        print(f"Binary not found: {binary}", file=sys.stderr)
        return None

    result = subprocess.run(
        [binary, str(WEIGHTS_PATH),
         " ".join(str(t) for t in token_ids),
         str(max_tokens)],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500], file=sys.stderr)

    output_ids = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUT:"):
            ids = [int(x) for x in line[4:].split() if x.lstrip("-").isdigit()]
            output_ids.extend(ids)

    return {"output": output_ids} if output_ids else None


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B ANE inference")
    parser.add_argument("prompt", type=str)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--no-server", action="store_true",
                        help="Force subprocess mode even if server is running")
    parser.add_argument("--sock", type=str, default=DEFAULT_SOCK,
                        help="Socket path for server mode")
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

<<<<<<< HEAD
    # Build chat template
=======
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": args.prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(text)
<<<<<<< HEAD
    print(f"Prompt tokens: {len(input_ids)}")

    # Run the C binary — pass token IDs as arguments
    import subprocess
    binary = str(INFERENCE_DIR / "qwen_ane")

    # We need to modify the binary to accept token IDs as input
    # For now, print the token IDs so we can verify tokenization
    print(f"First 10 tokens: {input_ids[:10]}")
    print(f"Token text: {[tok.decode([t]) for t in input_ids[:10]]}")
    print(f"\nRunning ANE inference with {len(input_ids)} prompt tokens + {args.max_tokens} generation...")

    # Call binary with token IDs piped via stdin
    result = subprocess.run(
        [binary, str(WEIGHTS_PATH), " ".join(str(t) for t in input_ids),
         str(args.max_tokens)],
        capture_output=True, text=True, timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr[:500], file=sys.stderr)

    # Parse output token IDs from binary stdout
    output_ids = []
    for line in result.stdout.split("\n"):
        if line.startswith("OUT:"):
            ids = [int(x) for x in line[4:].split() if x.isdigit()]
            output_ids.extend(ids)

    if output_ids:
        decoded = tok.decode(output_ids, skip_special_tokens=True)
        print(f"\n=== Response ===\n{decoded}")
    else:
        print("\n(No output tokens parsed — binary may need token ID input mode)")
=======
    print(f"Prompt: {len(input_ids)} tokens")

    # Try socket server first (instant response)
    result = None
    if not args.no_server and os.path.exists(args.sock):
        print(f"Connecting to server at {args.sock}...")
        t0 = time.time()
        result = query_socket(input_ids, args.max_tokens, args.sock)
        elapsed = time.time() - t0
        if result:
            print(f"Server responded in {elapsed:.3f}s")
        else:
            print("Server not responding, falling back to subprocess...")

    # Fall back to subprocess
    if result is None:
        print("Running inference (subprocess mode, ~6s startup)...")
        result = query_subprocess(input_ids, args.max_tokens)

    if not result or "output" not in result:
        print("(No output received)", file=sys.stderr)
        return

    output_ids = result["output"]
    if output_ids:
        decoded = tok.decode(output_ids, skip_special_tokens=True)
        print(f"\n=== Response ===\n{decoded}")

    if "prefill_tps" in result:
        print(f"\nPrefill: {result['prefill_tps']:.1f} t/s | "
              f"Decode: {result['decode_tps']:.1f} t/s")
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf


if __name__ == "__main__":
    main()

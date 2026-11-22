#!/usr/bin/env python3
"""Extract pretokenized TinyStories data from zip.
Data format: flat uint16 token IDs (llama2.c BPE, 32K vocab).
Source: ~/tiny_stories_data_pretokenized.zip"""

import os, sys, struct, zipfile
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Look for zip in local directory
ZIP_PATH = os.path.join(BASE_DIR, 'tiny_stories_data_pretokenized.zip')
if not os.path.exists(ZIP_PATH):
    # Fallback to local name if not in base dir
    ZIP_PATH = 'tiny_stories_data_pretokenized.zip'
OUTPUT_PATH = str(Path(__file__).resolve().parent / 'tinystories_data00.bin')
VOCAB_SIZE = 32000
MAX_ZIP_SIZE = int(os.environ.get('MAX_ZIP_BYTES', str(10 * 1024 * 1024 * 1024)))

def main():
    # --- Input validation (ref: docs/reports/security-audit-2026-03-02.md LOW-03) ---

    # 1. Validate ZIP exists before attempting to open
    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: ZIP-Datei nicht gefunden: {ZIP_PATH}", file=sys.stderr)
        print(f"  Erwartet: ~/tiny_stories_data_pretokenized.zip", file=sys.stderr)
        sys.exit(1)

    # 2. Validate ZIP size (guard against unexpectedly large or corrupted files)
    zip_size = os.path.getsize(ZIP_PATH)
    if zip_size > MAX_ZIP_SIZE:
        print(f"ERROR: ZIP-Datei zu gross ({zip_size/1e9:.1f} GB > {MAX_ZIP_SIZE/1e9:.0f} GB Limit).",
              file=sys.stderr)
        print(f"  Setze MAX_ZIP_BYTES=<bytes> um das Limit anzupassen.", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(OUTPUT_PATH):
        n = os.path.getsize(OUTPUT_PATH) // 2
        print(f"{OUTPUT_PATH} already exists ({n} tokens, {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"ERROR: ZIP file not found: {ZIP_PATH}", file=sys.stderr)
        print(f"  Expected: ~/tiny_stories_data_pretokenized.zip", file=sys.stderr)
        sys.exit(1)

    zip_size = os.path.getsize(ZIP_PATH)
    if zip_size > MAX_ZIP_SIZE:
        print(f"ERROR: ZIP file too large ({zip_size/1e9:.1f} GB > {MAX_ZIP_SIZE/1e9:.0f} GB limit).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Extracting data00.bin from {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        names = z.namelist()
        if 'data00.bin' not in names:
            print(f"ERROR: data00.bin not found in ZIP. Contents: {names[:10]}", file=sys.stderr)
            sys.exit(1)

        with z.open('data00.bin') as src, open(OUTPUT_PATH, 'wb') as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)

    n = os.path.getsize(OUTPUT_PATH) // 2
    print(f"Written {OUTPUT_PATH} ({n} tokens, {os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")

    with open(OUTPUT_PATH, 'rb') as f:
        raw = f.read(20)
        if len(raw) < 20:
            print(f"WARNING: output file is very small ({len(raw)} bytes), possibly empty.", file=sys.stderr)
        else:
            tokens = struct.unpack('<10H', raw)
            print(f"First 10 tokens: {tokens}")
            oob = [t for t in tokens if t >= VOCAB_SIZE]
            if oob:
                print(f"WARNING: out-of-vocab tokens found: {oob} (vocab_size={VOCAB_SIZE})",
                      file=sys.stderr)

if __name__ == '__main__':
    main()

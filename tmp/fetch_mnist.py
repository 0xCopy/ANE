#!/usr/bin/env python3
"""Download and extract MNIST dataset into data/ directory."""
import os, gzip, struct, urllib.request

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FALLBACK = "https://ossci-datasets.s3.amazonaws.com/mnist/"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

def main():
    os.makedirs("data", exist_ok=True)
    for f in FILES:
        gz_path  = os.path.join("data", f)
        raw_path = gz_path[:-3]
        if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
            print(f"  ✓ {raw_path} ({os.path.getsize(raw_path):,} bytes)")
            continue
        print(f"  Downloading {f}...")
        for base in [BASE_URL, FALLBACK]:
            try:
                urllib.request.urlretrieve(base + f, gz_path)
                break
            except Exception:
                continue
        with gzip.open(gz_path, 'rb') as gz:
            with open(raw_path, 'wb') as out:
                out.write(gz.read())
        os.remove(gz_path)
        print(f"    → {raw_path} ({os.path.getsize(raw_path):,} bytes)")

    with open("data/train-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"\n  Train: {n} images, {rows}x{cols}")
    with open("data/t10k-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"  Test:  {n} images, {rows}x{cols}")
    print("\nReady: make run")

if __name__ == "__main__":
    main()

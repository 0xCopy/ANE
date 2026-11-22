#!/usr/bin/env python3
"""Download and extract MNIST dataset into data/ directory."""
import os, gzip, struct, urllib.request

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# Fallback: "http://yann.lecun.com/exdb/mnist/"

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
        raw_path = gz_path[:-3]  # strip .gz

        if os.path.exists(raw_path):
            # Verify file isn't truncated
            sz = os.path.getsize(raw_path)
            if sz > 1000:
                print(f"  ✓ {raw_path} ({sz:,} bytes)")
                continue

        print(f"  Downloading {f}...")
        try:
            urllib.request.urlretrieve(BASE_URL + f, gz_path)
        except Exception as e:
            # Try fallback URL
            print(f"    Primary failed, trying fallback...")
            urllib.request.urlretrieve(
                "http://yann.lecun.com/exdb/mnist/" + f, gz_path)

        with gzip.open(gz_path, 'rb') as gz:
            with open(raw_path, 'wb') as out:
                out.write(gz.read())
        os.remove(gz_path)
        print(f"    → {raw_path} ({os.path.getsize(raw_path):,} bytes)")

    # Quick sanity check
    with open("data/train-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"\n  Train: {n} images, {rows}x{cols}")
    with open("data/t10k-images-idx3-ubyte", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        print(f"  Test:  {n} images, {rows}x{cols}")

    print("\nDone. Ready to train: make mnist && ./mnist_ane")

if __name__ == "__main__":
    main()

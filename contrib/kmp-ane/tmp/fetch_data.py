#!/usr/bin/env python3
"""Download and extract raw MNIST IDX files into data/."""
import os, gzip, struct, urllib.request

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
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
        raw = os.path.join("data", f[:-3])
        if os.path.exists(raw) and os.path.getsize(raw) > 1000:
            print(f"  ✓ {raw} ({os.path.getsize(raw):,} bytes)")
            continue
        gz = os.path.join("data", f)
        for base in [MIRROR, FALLBACK]:
            try:
                print(f"  GET {base}{f}")
                urllib.request.urlretrieve(base + f, gz)
                break
            except Exception:
                continue
        with gzip.open(gz, 'rb') as gi:
            with open(raw, 'wb') as out:
                out.write(gi.read())
        os.remove(gz)
        print(f"    → {raw} ({os.path.getsize(raw):,} bytes)")

    with open("data/train-images-idx3-ubyte", "rb") as fi:
        _, n, r, c = struct.unpack(">IIII", fi.read(16))
        print(f"\n  Train: {n} images, {r}x{c}")
    with open("data/t10k-images-idx3-ubyte", "rb") as fi:
        _, n, r, c = struct.unpack(">IIII", fi.read(16))
        print(f"  Test:  {n} images, {r}x{c}")
    print("\nReady. Run: make && ./mnist_demo")

if __name__ == "__main__":
    main()

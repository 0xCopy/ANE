#!/usr/bin/env python3
"""
MNIST Data Prep — multi-source fetch for the KMP-ANE trainer.

Tries, in order: torchvision → tensorflow/keras → sklearn → raw HTTP.
Override with --source torch|tf|sklearn|raw.

Output (flat binary, ready for Kotlin/Native mmap):
  data/mnist_train_images.bin  — 60000 × 784 float32 [0,1]
  data/mnist_train_labels.bin  — 60000 × uint8
  data/mnist_test_images.bin   — 10000 × 784 float32 [0,1]
  data/mnist_test_labels.bin   — 10000 × uint8
"""
import argparse, gzip, os, struct, sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── helpers ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

def _save_np(images, labels, prefix):
    """Save with numpy (fast path)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    imgs = images.reshape(len(images), -1).astype(np.float32)
    if imgs.max() > 1.1:          # still 0-255
        imgs = imgs / 255.0
    lbls = labels.astype(np.uint8)
    img_p = DATA_DIR / f"mnist_{prefix}_images.bin"
    lbl_p = DATA_DIR / f"mnist_{prefix}_labels.bin"
    imgs.tofile(str(img_p))
    lbls.tofile(str(lbl_p))
    print(f"  {img_p.name}: {imgs.shape} f32 ({img_p.stat().st_size/1e6:.1f} MB)")
    print(f"  {lbl_p.name}: {lbls.shape} u8  ({lbl_p.stat().st_size/1e3:.1f} KB)")

def _save_pure(images, labels, prefix):
    """Save with struct (no numpy)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    img_p = DATA_DIR / f"mnist_{prefix}_images.bin"
    lbl_p = DATA_DIR / f"mnist_{prefix}_labels.bin"
    with open(img_p, "wb") as f:
        for img in images:
            f.write(struct.pack(f"<{len(img)}f", *img))
    with open(lbl_p, "wb") as f:
        f.write(bytes(labels))
    print(f"  {img_p.name}: {len(images)}×784 f32 ({img_p.stat().st_size/1e6:.1f} MB)")
    print(f"  {lbl_p.name}: {len(labels)} u8")

# ── source: torchvision ────────────────────────────────────────────────────
def fetch_torch():
    print("📦 [torchvision] Downloading MNIST...")
    import torch
    import torchvision
    import torchvision.transforms as T
    train = torchvision.datasets.MNIST("/tmp/mnist_tv", train=True,  download=True)
    test  = torchvision.datasets.MNIST("/tmp/mnist_tv", train=False, download=True)
    _save_np(train.data.numpy(), train.targets.numpy(), "train")
    _save_np(test.data.numpy(),  test.targets.numpy(),  "test")

# ── source: tensorflow ─────────────────────────────────────────────────────
def fetch_tf():
    print("📦 [tensorflow] Downloading MNIST...")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    _save_np(x_tr, y_tr, "train")
    _save_np(x_te, y_te, "test")

# ── source: sklearn (fetch_openml) ─────────────────────────────────────────
def fetch_sklearn():
    print("📦 [sklearn] Downloading MNIST via fetch_openml('mnist_784')...")
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    imgs  = mnist.data.astype(np.float32)   # (70000, 784)  0-255
    lbls  = mnist.target.astype(np.uint8)
    _save_np(imgs[:60000], lbls[:60000], "train")
    _save_np(imgs[60000:], lbls[60000:], "test")

# ── source: raw HTTP (zero-dep beyond stdlib) ──────────────────────────────
def fetch_raw():
    MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
    PRIMARY = "http://yann.lecun.com/exdb/mnist"
    NAMES = {
        "train_images": ("train-images-idx3-ubyte.gz", 16, 60000, 784),
        "train_labels": ("train-labels-idx1-ubyte.gz", 8,  60000, 1),
        "test_images":  ("t10k-images-idx3-ubyte.gz",  16, 10000, 784),
        "test_labels":  ("t10k-labels-idx1-ubyte.gz",  8,  10000, 1),
    }
    cache = Path("/tmp/mnist_raw"); cache.mkdir(exist_ok=True)
    print("📦 [raw HTTP] Downloading MNIST from ossci mirror...")

    def dl(fname):
        p = cache / fname
        if p.exists(): return p.read_bytes()
        for base in [MIRROR, PRIMARY]:
            try:
                print(f"  GET {base}/{fname}")
                data = urllib.request.urlopen(f"{base}/{fname}", timeout=30).read()
                p.write_bytes(data); return data
            except Exception:
                continue
        raise RuntimeError(f"Cannot download {fname}")

    arrays = {}
    for key, (fname, offset, n, dim) in NAMES.items():
        raw = gzip.decompress(dl(fname))
        arrays[key] = raw[offset:]

    if HAS_NP:
        for prefix in ["train", "test"]:
            n = 60000 if prefix == "train" else 10000
            imgs = np.frombuffer(arrays[f"{prefix}_images"], dtype=np.uint8).reshape(n, 784)
            lbls = np.frombuffer(arrays[f"{prefix}_labels"], dtype=np.uint8)[:n]
            _save_np(imgs, lbls, prefix)
    else:
        # Pure-Python fallback
        for prefix in ["train", "test"]:
            n = 60000 if prefix == "train" else 10000
            raw_img = arrays[f"{prefix}_images"]
            raw_lbl = arrays[f"{prefix}_labels"]
            images = []
            for i in range(n):
                off = i * 784
                images.append([raw_img[off+j] / 255.0 for j in range(784)])
            _save_pure(images, list(raw_lbl[:n]), prefix)

# ── auto-detect ────────────────────────────────────────────────────────────
SOURCES = {"torch": fetch_torch, "tf": fetch_tf, "sklearn": fetch_sklearn, "raw": fetch_raw}

def auto_detect():
    for name in ["torch", "tf", "sklearn", "raw"]:
        try:
            if name == "torch":    __import__("torchvision"); return name
            elif name == "tf":     __import__("tensorflow");  return name
            elif name == "sklearn": __import__("sklearn");    return name
            else:                  return name
        except ImportError:
            continue
    return "raw"

# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST for KMP-ANE trainer")
    parser.add_argument("--source", choices=list(SOURCES), default=None)
    args = parser.parse_args()

    # Already done?
    expected = [f"mnist_{s}_{t}.bin" for s in ["train","test"] for t in ["images","labels"]]
    if all((DATA_DIR / f).exists() for f in expected):
        n_tr = os.path.getsize(DATA_DIR / "mnist_train_images.bin") // (784*4)
        n_te = os.path.getsize(DATA_DIR / "mnist_test_images.bin")  // (784*4)
        print(f"✅ MNIST data already prepared: {n_tr} train + {n_te} test")
        print(f"   {DATA_DIR}")
        return

    src = args.source or auto_detect()
    print(f"Source: {src}")
    SOURCES[src]()

    # Verify
    print("\n✅ Verification:")
    if HAS_NP:
        for prefix in ["train", "test"]:
            imgs = np.fromfile(str(DATA_DIR / f"mnist_{prefix}_images.bin"), dtype=np.float32)
            lbls = np.fromfile(str(DATA_DIR / f"mnist_{prefix}_labels.bin"), dtype=np.uint8)
            n = len(lbls); imgs = imgs.reshape(n, 784)
            print(f"  {prefix}: {n} samples, px [{imgs.min():.3f}..{imgs.max():.3f}], "
                  f"labels [{lbls.min()}..{lbls.max()}]")
    else:
        for prefix in ["train", "test"]:
            img_sz = os.path.getsize(DATA_DIR / f"mnist_{prefix}_images.bin")
            lbl_sz = os.path.getsize(DATA_DIR / f"mnist_{prefix}_labels.bin")
            print(f"  {prefix}: {lbl_sz} samples, {img_sz/1e6:.1f} MB images")

    print(f"\n📁 {DATA_DIR}")
    print("Ready for Kotlin/Native mmap.")

if __name__ == "__main__":
    main()

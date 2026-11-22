#!/bin/bash
# run.sh — Build and run the KMP MNIST trainer on ANE
# Usage: ./run.sh [--cpu-only]
set -e

cd "$(dirname "$0")"

echo "=== KMP-ANE MNIST Trainer ==="
echo ""

# Step 1: Prepare MNIST data
if [ ! -f data/mnist_train_images.bin ]; then
    echo "Step 1: Downloading MNIST data..."
    python3 prepare_mnist.py
else
    echo "Step 1: MNIST data already prepared ✓"
fi
echo ""

# Step 2: Build native bridge library
echo "Step 2: Building libanebridge.a..."
make clean all
echo ""

# Step 3: Build KMP binary
echo "Step 3: Building Kotlin/Native MNIST trainer..."
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    TARGET="macosArm64"
else
    TARGET="macosX64"
fi

# Use gradle wrapper if available, otherwise use system gradle
if [ -f gradlew ]; then
    ./gradlew ${TARGET}Binaries --console=plain
elif command -v gradle &>/dev/null; then
    gradle ${TARGET}Binaries --console=plain
else
    echo "ERROR: gradle not found. Install with: brew install gradle"
    echo "Or create a Gradle wrapper: gradle wrapper"
    exit 1
fi
echo ""

# Step 4: Run
echo "Step 4: Running MNIST trainer..."
echo "═══════════════════════════════════════════════"
BINARY="build/bin/${TARGET}/mnist_trainerDebugExecutable/mnist_trainer.kexe"
if [ ! -f "$BINARY" ]; then
    BINARY="build/bin/${TARGET}/mnist_trainerReleaseExecutable/mnist_trainer.kexe"
fi
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY"
    echo "Check build output above for errors."
    exit 1
fi

exec "$BINARY" "$@"

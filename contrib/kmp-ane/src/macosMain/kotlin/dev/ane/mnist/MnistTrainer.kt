// MnistTrainer.kt — End-to-end MNIST training on Apple Neural Engine
//
// Uses the KotlinGrad-style Autograd.kt tape for automatic differentiation,
// with ANELinear.kt mapping forward matmuls to ANE conv kernels.
//
// Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10
// Forward: linear layers compiled to ANE 1×1 conv kernels (baked weights)
// Backward: automatic differentiation via Autograd tape, dW on CPU
//
// Data: raw binary files from prepare_mnist.py
//   mnist_train_images.bin — 60000 × 784 × float32
//   mnist_train_labels.bin — 60000 × uint8
//   mnist_test_images.bin  — 10000 × 784 × float32
//   mnist_test_labels.bin  — 10000 × uint8
package dev.ane.mnist

import dev.ane.grad.*
import dev.ane.runtime.ANEDevice
import dev.ane.runtime.ANEStatus
import platform.posix.*
import kotlinx.cinterop.*
import kotlin.math.*

// ---------------------------------------------------------------------------
//  Data loading (Kotlin/Native file I/O)
// ---------------------------------------------------------------------------
@OptIn(ExperimentalForeignApi::class)
fun loadFloatBinary(path: String, count: Int): FloatArray {
    val f = fopen(path, "rb") ?: error("Cannot open $path")
    val buf = FloatArray(count)
    buf.usePinned { pinned ->
        fread(pinned.addressOf(0), 4u, count.toULong(), f)
    }
    fclose(f)
    return buf
}

@OptIn(ExperimentalForeignApi::class)
fun loadUInt8Binary(path: String, count: Int): IntArray {
    val f = fopen(path, "rb") ?: error("Cannot open $path")
    val buf = ByteArray(count)
    buf.usePinned { pinned ->
        fread(pinned.addressOf(0), 1u, count.toULong(), f)
    }
    fclose(f)
    return IntArray(count) { buf[it].toInt() and 0xFF }
}

// ---------------------------------------------------------------------------
//  Config (use TrainConfig from commonMain)
// ---------------------------------------------------------------------------
// Local aliases for readability:
private val BATCH_SIZE  get() = TrainConfig.BATCH_SIZE
private val EPOCHS      get() = TrainConfig.EPOCHS
private val HIDDEN1     get() = TrainConfig.HIDDEN1
private val HIDDEN2     get() = TrainConfig.HIDDEN2
private val INPUT_DIM   get() = TrainConfig.INPUT_DIM
private val NUM_CLASSES get() = TrainConfig.NUM_CLASSES
private val TRAIN_N     get() = TrainConfig.TRAIN_N
private val TEST_N      get() = TrainConfig.TEST_N
private val LR          get() = TrainConfig.LR

// computeAccuracy is defined in commonMain/MnistTrainer.kt

// ---------------------------------------------------------------------------
//  Main entry point
// ---------------------------------------------------------------------------
fun main() {
    println("╔══════════════════════════════════════════════╗")
    println("║  MNIST Trainer — KMP → Swift → ObjC → ANE   ║")
    println("║                                              ║")
    println("║  KotlinGrad expressions → ANE conv kernels   ║")
    println("╚══════════════════════════════════════════════╝")
    println()

    // --- Check ANE availability ---
    val aneStatus = ANEDevice.init()
    val useANE = aneStatus == ANEStatus.OK && ANEDevice.isAvailable()
    println("[Runtime] ANE: ${if (useANE) "AVAILABLE ✓" else "not available, using CPU"}")

    // --- Locate data ---
    val dataDirs = listOf("data", "contrib/kmp-ane/data", "../data", ".")
    val dataDir = dataDirs.firstOrNull { dir ->
        access("$dir/mnist_train_images.bin", F_OK) == 0
    }

    if (dataDir == null) {
        println()
        println("ERROR: MNIST data not found. Run prepare_mnist.py first:")
        println("  cd contrib/kmp-ane && python3 prepare_mnist.py")
        return
    }
    println("[Data]    Loading from $dataDir/")

    val trainImages = loadFloatBinary("$dataDir/mnist_train_images.bin", TRAIN_N * INPUT_DIM)
    val trainLabels = loadUInt8Binary("$dataDir/mnist_train_labels.bin", TRAIN_N)
    val testImages  = loadFloatBinary("$dataDir/mnist_test_images.bin",  TEST_N  * INPUT_DIM)
    val testLabels  = loadUInt8Binary("$dataDir/mnist_test_labels.bin",  TEST_N)
    println("[Data]    Train: $TRAIN_N images, Test: $TEST_N images")

    // --- Create trainable parameters (Kaiming init) ---
    val W1 = Parameter(Tensor.kaiming(HIDDEN1, INPUT_DIM))   // [128, 784]
    val b1 = Parameter(Tensor.zeros(1, HIDDEN1))              // [1, 128]
    val W2 = Parameter(Tensor.kaiming(HIDDEN2, HIDDEN1))      // [64, 128]
    val b2 = Parameter(Tensor.zeros(1, HIDDEN2))              // [1, 64]
    val W3 = Parameter(Tensor.kaiming(NUM_CLASSES, HIDDEN2))  // [10, 64]
    val b3 = Parameter(Tensor.zeros(1, NUM_CLASSES))          // [1, 10]

    val params = listOf(W1, b1, W2, b2, W3, b3)
    val totalParams = params.sumOf { it.value.size }
    println("[Model]   $totalParams parameters")
    println("[Model]   $INPUT_DIM → $HIDDEN1 (ReLU) → $HIDDEN2 (ReLU) → $NUM_CLASSES")
    println("[Backend] ${if (useANE) "ANE forward, CPU backward" else "CPU only"}")

    // --- Optimizer ---
    val optimizer = Adam(params, lr = LR)
    val stepsPerEpoch = TRAIN_N / BATCH_SIZE

    println()
    println("Training: $EPOCHS epochs, batch=$BATCH_SIZE, lr=$LR, steps/epoch=$stepsPerEpoch")
    println("─".repeat(70))

    // --- Training loop ---
    for (epoch in 1..EPOCHS) {
        var epochLoss = 0f
        var epochCorrect = 0
        var epochTotal = 0

        for (step in 0 until stepsPerEpoch) {
            val offset = step * BATCH_SIZE

            // --- Extract batch ---
            val batchX = FloatArray(BATCH_SIZE * INPUT_DIM)
            val batchY = IntArray(BATCH_SIZE)
            for (i in 0 until BATCH_SIZE) {
                val idx = (offset + i) % TRAIN_N
                trainImages.copyInto(batchX, i * INPUT_DIM, idx * INPUT_DIM, idx * INPUT_DIM + INPUT_DIM)
                batchY[i] = trainLabels[idx]
            }

            // --- Build KotlinGrad expression graph (rebuilt each step) ---
            // This is the KotlinGrad pattern: build graph → forward → backward
            optimizer.zeroGrad()

            val x = Variable(Tensor(batchX, intArrayOf(BATCH_SIZE, INPUT_DIM)))

            // Layer 1: x @ W1^T + b1 → ReLU
            //   ANELinear does x @ W^T via ANE conv when available
            val h1 = x.aneLinear(W1, useANE = useANE).addBias(b1).relu()

            // Layer 2: h1 @ W2^T + b2 → ReLU
            val h2 = h1.aneLinear(W2, useANE = useANE).addBias(b2).relu()

            // Layer 3: h2 @ W3^T + b3 → logSoftmax → NLL loss
            val logits = h2.aneLinear(W3, useANE = useANE).addBias(b3)
            val logProbs = logits.logSoftmax()
            val loss = logProbs.nllLoss(batchY)

            // --- Backward (automatic differentiation) ---
            loss.backward()

            // --- Optimizer step ---
            optimizer.step()

            // --- Track metrics ---
            epochLoss += loss.value[0]
            epochCorrect += (computeAccuracy(logits.value, batchY) * BATCH_SIZE).toInt()
            epochTotal += BATCH_SIZE
        }

        val avgLoss = epochLoss / stepsPerEpoch
        val trainAcc = epochCorrect.toFloat() / epochTotal * 100f

        // --- Test evaluation ---
        var testCorrect = 0
        val testSteps = TEST_N / BATCH_SIZE
        for (step in 0 until testSteps) {
            val offset = step * BATCH_SIZE
            val batchX = FloatArray(BATCH_SIZE * INPUT_DIM)
            val batchY = IntArray(BATCH_SIZE)
            for (i in 0 until BATCH_SIZE) {
                val idx = (offset + i)
                if (idx >= TEST_N) break
                testImages.copyInto(batchX, i * INPUT_DIM, idx * INPUT_DIM, idx * INPUT_DIM + INPUT_DIM)
                batchY[i] = testLabels[idx]
            }
            // Inference only — no grad needed
            val x = Variable(Tensor(batchX, intArrayOf(BATCH_SIZE, INPUT_DIM)))
            val h1 = x.aneLinear(W1, useANE = useANE).addBias(b1).relu()
            val h2 = h1.aneLinear(W2, useANE = useANE).addBias(b2).relu()
            val logits = h2.aneLinear(W3, useANE = useANE).addBias(b3)
            testCorrect += (computeAccuracy(logits.value, batchY) * BATCH_SIZE).toInt()
        }
        val testAcc = testCorrect.toFloat() / (testSteps * BATCH_SIZE) * 100f

        val backend = if (useANE) "ANE" else "CPU"
        println("Epoch %2d/%d │ loss=%.4f │ train=%.1f%% │ test=%.1f%% │ [%s]".format(
            epoch, EPOCHS, avgLoss, trainAcc, testAcc, backend))
    }

    println("─".repeat(70))
    println("Training complete.")
    println()
    println("Stack: Kotlin (KMP) → aneLinear() → ANEDevice (cinterop)")
    println("     → ANEBridge.m (ObjC) → ane_runtime.h → AppleNeuralEngine.framework")
}

// KMP entry point for the macOS/Apple MNIST demo binary.
package com.ane.mnist

fun main() {
    println("╔══════════════════════════════════════════════════╗")
    println("║  MNIST Training on Apple Neural Engine (Kotlin)  ║")
    println("╚══════════════════════════════════════════════════╝")
    println()

    val config = MnistConfig(
        hiddenDim = 128,
        batchSize = 128,
        learningRate = 0.01f,
        momentum = 0.9f,
    )

    val trainer = MnistTrainer(config)

    println("Loading MNIST data...")
    if (!trainer.loadData("data")) {
        println("ERROR: Failed to load data. Run: python3 mnist/fetch_mnist.py")
        return
    }
    println("  Train: ${trainer.trainCount} samples")
    println("  Test:  ${trainer.testCount} samples")
    println()

    println("Compiling ANE kernels...")
    if (!trainer.compile()) {
        println("ERROR: ANE compile failed. Is this macOS 15+ on Apple Silicon?")
        return
    }
    println("  2 matmul kernels ready (0 recompiles needed)")
    println()

    val epochs = 10
    println("Training for $epochs epochs (${trainer.batchesPerEpoch} batches/epoch)...")
    println()

    trainer.trainEpochs(
        epochs = epochs,
        logEvery = 100,
        onStep = { r ->
            println("  Epoch ${r.epoch + 1}  step=${r.step}  loss=${String.format("%.4f", r.loss)}  acc=${String.format("%.1f", r.accuracy)}%")
        },
        onEpochEnd = { epoch, testAcc ->
            println("═══ Epoch $epoch  test_acc=${String.format("%.1f", testAcc)}% ═══")
        },
    )

    val finalAcc = trainer.evaluate()
    println()
    println("Final test accuracy: ${String.format("%.1f", finalAcc)}%")

    trainer.close()
    println("Done.")
}

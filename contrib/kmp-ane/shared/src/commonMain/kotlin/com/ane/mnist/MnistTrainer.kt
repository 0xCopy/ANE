// MnistTrainer.kt — KMP common declarations for MNIST training
// Platform-specific implementations:
//   appleMain → ANE via cinterop to ANEMnist.h
//   jvmMain   → Pure-Kotlin CPU fallback
package com.ane.mnist

/**
 * Configuration for the MNIST trainer.
 */
data class MnistConfig(
    val inputDim: Int = 784,
    val hiddenDim: Int = 128,
    val outputDim: Int = 10,
    val batchSize: Int = 128,
    val learningRate: Float = 0.01f,
    val momentum: Float = 0.9f,
    val accumSteps: Int = 50,
    val maxCompiles: Int = 118,
)

/**
 * Result of a single training step.
 */
data class TrainStepResult(
    val loss: Float,
    val accuracy: Float,
    val step: Int,
    val epoch: Int,
    val epochDone: Boolean,
    val compileCount: Int = 0,
    val budgetExhausted: Boolean = false,
)

/**
 * Cross-platform MNIST trainer.
 * On Apple Silicon: runs linear layers on ANE via conv kernels with baked weights.
 * On JVM: falls back to CPU implementation.
 */
expect class MnistTrainer(config: MnistConfig = MnistConfig()) {
    /**
     * Load MNIST data from a directory containing IDX files.
     * Expected files: train-images-idx3-ubyte, train-labels-idx1-ubyte,
     *                 t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
     * @return true on success
     */
    fun loadData(dataDir: String): Boolean

    /**
     * Compile ANE kernels (on Apple) or initialize CPU engine (on JVM).
     * Must be called after loadData, before training.
     * @return true on success
     */
    fun compile(): Boolean

    /**
     * Run one training step: forward → loss → backward → SGD update.
     * Automatically handles epoch transitions and data shuffling.
     */
    fun trainStep(): TrainStepResult

    /**
     * Evaluate accuracy on the test set.
     * @return accuracy percentage [0, 100]
     */
    fun evaluate(): Float

    /** Number of batches per training epoch. */
    val batchesPerEpoch: Int

    /** Number of training samples. */
    val trainCount: Int

    /** Number of test samples. */
    val testCount: Int

    /** Current global step. */
    val currentStep: Int

    /** Release all resources. */
    fun close()
}

/**
 * Convenience: train for N epochs with callbacks.
 */
fun MnistTrainer.trainEpochs(
    epochs: Int,
    logEvery: Int = 100,
    onStep: ((TrainStepResult) -> Unit)? = null,
    onEpochEnd: ((epoch: Int, testAccuracy: Float) -> Unit)? = null,
) {
    repeat(epochs) {
        val bpe = batchesPerEpoch
        for (step in 0 until bpe) {
            val result = trainStep()
            if ((step + 1) % logEvery == 0) {
                onStep?.invoke(result)
            }
            if (result.budgetExhausted) return
            if (result.epochDone) {
                val testAcc = evaluate()
                onEpochEnd?.invoke(result.epoch, testAcc)
            }
        }
    }
}

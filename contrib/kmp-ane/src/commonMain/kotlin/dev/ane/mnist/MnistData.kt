// MnistData.kt — Load MNIST binary data files produced by prepare_mnist.py
// Binary format:
//   images: N × 784 float32 (row-major, values in [0,1])
//   labels: N × uint8
package dev.ane.mnist

import dev.ane.grad.Tensor

/**
 * MNIST dataset: images and labels loaded from flat binary files.
 */
class MnistDataset(
    val images: FloatArray,   // [N * 784] f32
    val labels: ByteArray,    // [N] u8
    val count: Int
) {
    /** Get a mini-batch starting at index [start] with size [batchSize]. */
    fun getBatch(start: Int, batchSize: Int): Pair<Tensor, IntArray> {
        val actualSize = minOf(batchSize, count - start)
        val imgBatch = FloatArray(actualSize * 784)
        val lblBatch = IntArray(actualSize)
        for (i in 0 until actualSize) {
            val srcOff = (start + i) * 784
            val dstOff = i * 784
            images.copyInto(imgBatch, dstOff, srcOff, srcOff + 784)
            lblBatch[i] = labels[start + i].toInt() and 0xFF
        }
        return Tensor(imgBatch, intArrayOf(actualSize, 784)) to lblBatch
    }

    /** Shuffle indices and return a permuted copy. */
    fun shuffled(seed: Long = System.currentTimeMillis()): MnistDataset {
        val indices = IntArray(count) { it }
        // Fisher-Yates shuffle with simple LCG
        var s = seed
        for (i in count - 1 downTo 1) {
            s = (s * 6364136223846793005L + 1442695040888963407L)
            val j = ((s ushr 33) % (i + 1)).toInt().let { if (it < 0) it + i + 1 else it }
            indices[i] = indices[j].also { indices[j] = indices[i] }
        }
        val newImages = FloatArray(count * 784)
        val newLabels = ByteArray(count)
        for (i in 0 until count) {
            val src = indices[i]
            images.copyInto(newImages, i * 784, src * 784, src * 784 + 784)
            newLabels[i] = labels[src]
        }
        return MnistDataset(newImages, newLabels, count)
    }

    companion object
}

/**
 * Platform-specific file loading.
 * In commonMain this is an expect; actual uses Kotlin/Native file I/O.
 */
expect fun MnistDataset.Companion.loadFromFiles(
    imagesPath: String,
    labelsPath: String
): MnistDataset

/**
 * Helper: get the current time in milliseconds (for shuffling seed).
 * Using a simple expect/actual since Clock isn't in common stdlib.
 */
expect fun System.currentTimeMillis(): Long

/**
 * Singleton "System" for the currentTimeMillis expect.
 */
object System

// MnistDataNative.kt — Kotlin/Native actual file-loading for MNIST binary data
package dev.ane.mnist

import dev.ane.grad.Tensor
import kotlinx.cinterop.*
import platform.posix.*

@OptIn(ExperimentalForeignApi::class)
actual fun MnistDataset.Companion.loadFromFiles(
    imagesPath: String,
    labelsPath: String
): MnistDataset {
    // Load labels (uint8)
    val lblFile = fopen(labelsPath, "rb")
        ?: error("Cannot open labels file: $labelsPath")
    fseek(lblFile, 0, SEEK_END)
    val lblSize = ftell(lblFile).toInt()
    fseek(lblFile, 0, SEEK_SET)
    val labels = ByteArray(lblSize)
    labels.usePinned { pinned ->
        fread(pinned.addressOf(0), 1u, lblSize.toULong(), lblFile)
    }
    fclose(lblFile)

    val count = lblSize

    // Load images (float32, N × 784)
    val imgFile = fopen(imagesPath, "rb")
        ?: error("Cannot open images file: $imagesPath")
    fseek(imgFile, 0, SEEK_END)
    val imgBytes = ftell(imgFile).toInt()
    fseek(imgFile, 0, SEEK_SET)

    val expectedBytes = count * 784 * 4  // float32
    require(imgBytes == expectedBytes) {
        "Image file size mismatch: expected $expectedBytes bytes for $count samples, got $imgBytes"
    }

    val images = FloatArray(count * 784)
    images.usePinned { pinned ->
        fread(pinned.addressOf(0), 1u, imgBytes.toULong(), imgFile)
    }
    fclose(imgFile)

    println("Loaded $count samples from $labelsPath")
    return MnistDataset(images, labels, count)
}

actual fun System.currentTimeMillis(): Long {
    return memScoped {
        val tv = alloc<timeval>()
        gettimeofday(tv.ptr, null)
        tv.tv_sec * 1000L + tv.tv_usec / 1000L
    }
}

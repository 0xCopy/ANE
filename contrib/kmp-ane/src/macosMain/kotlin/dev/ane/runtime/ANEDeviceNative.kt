// ANEDeviceNative.kt — Kotlin/Native actual implementation via cinterop
// Maps to libanebridge.a C functions through the generated cinterop bindings.
package dev.ane.runtime

import dev.ane.cinterop.*
import kotlinx.cinterop.*

actual typealias ANEKernelHandle = CPointer<ANEBridgeKernel>

@OptIn(ExperimentalForeignApi::class)
actual object ANEDevice {

    actual fun init(): ANEStatus {
        val code = ane_bridge_init()
        return ANEStatus.fromCode(code)
    }

    actual fun isAvailable(): Boolean = ane_bridge_is_available()

    actual fun compileConv(
        weights: FloatArray,
        inCh: Int,
        outCh: Int,
        spatial: Int
    ): ANEKernelHandle? = memScoped {
        val outPtr = alloc<CPointerVar<ANEBridgeKernel>>()
        weights.usePinned { pinned ->
            val status = ane_bridge_compile_conv(
                pinned.addressOf(0),
                inCh, outCh, spatial,
                outPtr.ptr
            )
            if (status == ANE_OK) outPtr.value else null
        }
    }

    actual fun convEval(
        kernel: ANEKernelHandle,
        x: FloatArray,
        S: Int,
        inDim: Int,
        outDim: Int
    ): FloatArray? {
        val y = FloatArray(S * outDim)
        val status = x.usePinned { xPin ->
            y.usePinned { yPin ->
                ane_bridge_conv_eval(
                    kernel, xPin.addressOf(0), yPin.addressOf(0),
                    S, inDim, outDim
                )
            }
        }
        return if (status == ANE_OK) y else null
    }

    actual fun freeKernel(kernel: ANEKernelHandle) {
        ane_bridge_free_kernel(kernel)
    }

    actual fun compile(
        milText: String,
        weightData: ByteArray?,
        nInputs: Int,
        inputBytes: LongArray,
        nOutputs: Int,
        outputBytes: LongArray
    ): ANEKernelHandle? = memScoped {
        val outPtr = alloc<CPointerVar<ANEBridgeKernel>>()
        val milBytes = milText.encodeToByteArray()

        val inSizes = allocArray<size_tVar>(nInputs)
        for (i in 0 until nInputs) inSizes[i] = inputBytes[i].toULong()

        val outSizes = allocArray<size_tVar>(nOutputs)
        for (i in 0 until nOutputs) outSizes[i] = outputBytes[i].toULong()

        val status = milBytes.usePinned { milPin ->
            if (weightData != null) {
                weightData.usePinned { wPin ->
                    ane_bridge_compile(
                        milPin.addressOf(0).reinterpret(),
                        milBytes.size.toULong(),
                        wPin.addressOf(0),
                        weightData.size.toULong(),
                        nInputs, inSizes,
                        nOutputs, outSizes,
                        outPtr.ptr
                    )
                }
            } else {
                ane_bridge_compile(
                    milPin.addressOf(0).reinterpret(),
                    milBytes.size.toULong(),
                    null, 0u,
                    nInputs, inSizes,
                    nOutputs, outSizes,
                    outPtr.ptr
                )
            }
        }
        if (status == ANE_OK) outPtr.value else null
    }

    actual fun writeInput(kernel: ANEKernelHandle, index: Int, data: FloatArray): ANEStatus {
        val code = data.usePinned { pinned ->
            ane_bridge_write_input(
                kernel, index,
                pinned.addressOf(0),
                (data.size * 4).toULong()
            )
        }
        return ANEStatus.fromCode(code)
    }

    actual fun readOutput(kernel: ANEKernelHandle, index: Int, size: Int): FloatArray? {
        val result = FloatArray(size)
        val code = result.usePinned { pinned ->
            ane_bridge_read_output(
                kernel, index,
                pinned.addressOf(0),
                (size * 4).toULong()
            )
        }
        return if (code == ANE_OK) result else null
    }

    actual fun eval(kernel: ANEKernelHandle): ANEStatus {
        return ANEStatus.fromCode(ane_bridge_eval(kernel))
    }
}

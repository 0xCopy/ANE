// ANEDevice.kt — Platform-agnostic ANE device interface (commonMain)
// Expect declarations resolved by native actual implementations.
package dev.ane.runtime

/**
 * Status codes from ANE operations.
 */
enum class ANEStatus(val code: Int) {
    OK(0),
    INIT_FAILED(-1),
    COMPILE_FAILED(-2),
    LOAD_FAILED(-3),
    EVAL_FAILED(-4),
    NULL_KERNEL(-5),
    INDEX_OOB(-6),
    ALLOC_FAILED(-7);

    companion object {
        fun fromCode(code: Int): ANEStatus =
            entries.find { it.code == code } ?: EVAL_FAILED
    }
}

/**
 * Opaque handle to a compiled ANE kernel.
 */
expect class ANEKernelHandle

/**
 * Platform-specific ANE device operations.
 */
expect object ANEDevice {
    /** Initialize the ANE runtime. */
    fun init(): ANEStatus

    /** Check if ANE hardware is available. */
    fun isAvailable(): Boolean

    /**
     * Compile a baked-weight 1×1 conv kernel.
     * @param weights f32 array [outCh × inCh] row-major
     * @param inCh input channels  
     * @param outCh output channels
     * @param spatial spatial dimension (batch size)
     * @return kernel handle or null on failure
     */
    fun compileConv(weights: FloatArray, inCh: Int, outCh: Int, spatial: Int): ANEKernelHandle?

    /**
     * Evaluate a conv kernel: x[S, inDim] → y[S, outDim]  
     * Handles row-major ↔ channel-first transpose internally.
     */
    fun convEval(kernel: ANEKernelHandle, x: FloatArray, S: Int, inDim: Int, outDim: Int): FloatArray?

    /** Free a kernel. */
    fun freeKernel(kernel: ANEKernelHandle)

    /**
     * Compile arbitrary MIL with optional weights.
     */
    fun compile(milText: String, weightData: ByteArray?,
                nInputs: Int, inputBytes: LongArray,
                nOutputs: Int, outputBytes: LongArray): ANEKernelHandle?

    /** Write data to kernel input IOSurface. */
    fun writeInput(kernel: ANEKernelHandle, index: Int, data: FloatArray): ANEStatus

    /** Read data from kernel output IOSurface. */  
    fun readOutput(kernel: ANEKernelHandle, index: Int, size: Int): FloatArray?

    /** Execute kernel on ANE. */
    fun eval(kernel: ANEKernelHandle): ANEStatus
}

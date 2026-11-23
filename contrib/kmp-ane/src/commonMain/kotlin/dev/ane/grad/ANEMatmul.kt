// ANEMatmul.kt — matrix products on the Apple Neural Engine with weights passed as data.
//
// The baked-weight conv kernels compile the weights INTO the kernel, so a trainer had to
// recompile every layer on every step. Here both operands are kernel inputs: a kernel is
// compiled once per (shapes, transposes) and reused for the whole run, whatever the weights.
// That covers a linear layer's forward and both of its backward products:
//
//   forward   y  = x  @ W^T    product(x,  M,K,  W, N,K,  transposeB = true)
//   backward  dx = dy @ W      product(dy, M,N,  W, N,K)
//   backward  dW = dy^T @ x    product(dy, M,N,  x, M,K,  transposeA = true)
//
// A 2-layer MLP needs 5-6 kernels in total instead of 2 compiles per step.
package dev.ane.grad

import dev.ane.runtime.ANEDevice
import dev.ane.runtime.ANEKernelHandle
import dev.ane.runtime.ANEStatus
import kotlin.time.TimeSource

object ANEMatmul {
    /** Distinct kernel shapes kept loaded; the oldest is freed beyond this. */
    private const val MAX_KERNELS = 32

    private val kernels = LinkedHashMap<String, ANEKernelHandle>()
    private val unusable = HashSet<String>()      // shapes whose compile failed; not retried every step
    private var available: Boolean? = null
    private val clock = TimeSource.Monotonic.markNow()

    /** Kernels compiled since start (stays small: one per shape, not one per step). */
    var compiles = 0
        private set
    /** Products that ran on the ANE. */
    var aneEvals = 0L
        private set
    /** Products the caller had to compute on the CPU because [product] returned null. */
    var cpuFallbacks = 0L
        private set
    /** Time spent in ANE products, including the copies in and out (nanoseconds). */
    var aneNs = 0L
        private set
    /** Time spent compiling kernels (nanoseconds). */
    var compileNs = 0L
        private set

    val isAvailable: Boolean
        get() = available ?: (ANEDevice.init() == ANEStatus.OK && ANEDevice.isAvailable()).also { available = it }

    /**
     * Y = op(A) @ op(B) on the ANE, row-major. A is stored [aRows, aCols], B [bRows, bCols].
     * Returns null when the ANE cannot do it (unavailable, compile or eval failure);
     * the caller then computes the product on the CPU.
     */
    fun product(
        a: FloatArray, aRows: Int, aCols: Int,
        b: FloatArray, bRows: Int, bCols: Int,
        transposeA: Boolean = false, transposeB: Boolean = false
    ): FloatArray? {
        require(a.size == aRows * aCols && b.size == bRows * bCols) { "operand size does not match its shape" }
        val outRows = if (transposeA) aCols else aRows
        val outCols = if (transposeB) bRows else bCols
        require((if (transposeA) aRows else aCols) == (if (transposeB) bCols else bRows)) { "inner dimensions differ" }

        val kernel = kernelFor(aRows, aCols, bRows, bCols, transposeA, transposeB)
        if (kernel == null) { cpuFallbacks++; return null }

        val t0 = clock.elapsedNow().inWholeNanoseconds
        val y = ANEDevice.matmulEval(kernel, a, b, outRows * outCols)
        if (y == null) { cpuFallbacks++; return null }
        aneNs += clock.elapsedNow().inWholeNanoseconds - t0
        aneEvals++
        return y
    }

    private fun kernelFor(
        aRows: Int, aCols: Int, bRows: Int, bCols: Int, tA: Boolean, tB: Boolean
    ): ANEKernelHandle? {
        if (!isAvailable) return null
        val key = "$aRows,$aCols,$bRows,$bCols,$tA,$tB"
        kernels[key]?.let { return it }
        if (key in unusable) return null

        val t0 = clock.elapsedNow().inWholeNanoseconds
        val kernel = ANEDevice.compileMatmul(aRows, aCols, bRows, bCols, tA, tB)
        compileNs += clock.elapsedNow().inWholeNanoseconds - t0
        if (kernel == null) {
            unusable.add(key)
            println("[ANE] matmul kernel [$key] did not compile, using CPU for it: ${ANEDevice.lastError()}")
            return null
        }
        compiles++
        if (kernels.size >= MAX_KERNELS) {
            val oldest = kernels.keys.first()
            kernels.remove(oldest)?.let { ANEDevice.freeKernel(it) }
        }
        kernels[key] = kernel
        return kernel
    }

    /** Free every cached kernel (end of training). */
    fun releaseAll() {
        for (k in kernels.values) ANEDevice.freeKernel(k)
        kernels.clear()
        unusable.clear()
    }
}

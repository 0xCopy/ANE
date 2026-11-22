// ANELinear.kt — ANE-accelerated linear layer expression for the autograd graph
// When ANE is available, the forward matmul runs on Apple Neural Engine via
// ane_bridge_conv_eval. Backward always runs on CPU (matching the main project's
// approach: forward on ANE, dW on CPU).
package dev.ane.grad

import dev.ane.runtime.ANEDevice
import dev.ane.runtime.ANEKernelHandle
import dev.ane.runtime.ANEStatus

/**
 * An autograd expression that does y = x @ W^T + bias, where the matmul
 * can be offloaded to ANE as a 1×1 convolution.
 *
 * The ANE kernel bakes weights, so we recompile whenever weights are updated.
 * For MNIST this is fine — the compile cost is ~2ms per kernel.
 */
class ANELinearOp(
    val x: Expr,
    val weight: Parameter,
    val bias: Parameter?,
    val useANE: Boolean
) : Expr() {

    private var aneKernel: ANEKernelHandle? = null

    override val value: Tensor = computeForward()

    init {
        parents = if (bias != null) listOf(x, weight, bias) else listOf(x, weight)
        backwardFn = { dy ->
            val (batch, outDim) = dy.shape
            val inDim = weight.value.shape[1]

            // dx = dy @ W   (dy:[B, outDim] @ W:[outDim, inDim] → dx:[B, inDim])
            x.accumGrad(cpuMM(dy, weight.value))

            // dW = dy^T @ x  (dy^T:[outDim, B] @ x:[B, inDim] → dW:[outDim, inDim])
            weight.accumGrad(cpuMM(transposeT(dy), x.value))

            // dBias = sum(dy, axis=0)
            if (bias != null) {
                val db = FloatArray(outDim)
                for (b in 0 until batch)
                    for (j in 0 until outDim)
                        db[j] += dy[b * outDim + j]
                bias.accumGrad(Tensor(db, bias.value.shape.copyOf()))
            }
        }
    }

    private fun computeForward(): Tensor {
        val (batch, inDim) = x.value.shape
        val outDim = weight.value.shape[0]

        val y: FloatArray

        if (useANE && ANEDevice.isAvailable() && batch > 0) {
            // ANE path: compile baked-weight kernel, eval
            val kernel = ANEDevice.compileConv(
                weight.value.data, inDim, outDim, batch
            )
            if (kernel != null) {
                aneKernel = kernel
                val result = ANEDevice.convEval(kernel, x.value.data, batch, inDim, outDim)
                if (result != null) {
                    y = result
                } else {
                    // Fallback to CPU
                    y = cpuForward(batch, inDim, outDim)
                }
                // Free kernel immediately (weights will change next step)
                ANEDevice.freeKernel(kernel)
                aneKernel = null
            } else {
                y = cpuForward(batch, inDim, outDim)
            }
        } else {
            y = cpuForward(batch, inDim, outDim)
        }

        // Add bias
        if (bias != null) {
            for (b in 0 until batch)
                for (j in 0 until outDim)
                    y[b * outDim + j] += bias.value.data[j]
        }

        return Tensor(y, intArrayOf(batch, outDim))
    }

    private fun cpuForward(batch: Int, inDim: Int, outDim: Int): FloatArray {
        // y = x @ W^T
        val y = FloatArray(batch * outDim)
        val w = weight.value.data
        val xd = x.value.data
        for (b in 0 until batch)
            for (j in 0 until outDim) {
                var sum = 0f
                for (k in 0 until inDim)
                    sum += xd[b * inDim + k] * w[j * inDim + k]
                y[b * outDim + j] = sum
            }
        return y
    }

    companion object {
        private fun cpuMM(a: Tensor, b: Tensor): Tensor {
            val m = a.shape[0]; val k = a.shape[1]; val n = b.shape[1]
            require(b.shape[0] == k)
            val out = FloatArray(m * n)
            for (i in 0 until m)
                for (j in 0 until n) {
                    var s = 0f
                    for (p in 0 until k) s += a[i * k + p] * b[p * n + j]
                    out[i * n + j] = s
                }
            return Tensor(out, intArrayOf(m, n))
        }

        private fun transposeT(t: Tensor): Tensor {
            val (m, n) = t.shape
            val out = FloatArray(m * n)
            for (i in 0 until m)
                for (j in 0 until n)
                    out[j * m + i] = t[i * n + j]
            return Tensor(out, intArrayOf(n, m))
        }
    }
}

// ─── Convenience builder ────────────────────────────────────────────────

/**
 * Build an ANE-accelerated linear expression: y = x @ W^T + b
 */
fun Expr.aneLinear(weight: Parameter, bias: Parameter? = null, useANE: Boolean = true): Expr =
    ANELinearOp(this, weight, bias, useANE)

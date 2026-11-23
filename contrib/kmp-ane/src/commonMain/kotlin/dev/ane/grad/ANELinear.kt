// ANELinear.kt — ANE-accelerated linear layer expression for the autograd graph
// The forward matmul and both backward matmuls run on the Apple Neural Engine through
// dynamic-weight kernels (see ANEMatmul): the weights are kernel inputs, so nothing is
// recompiled when they change. CPU is the fallback when the ANE is unavailable.
package dev.ane.grad

/**
 * An autograd expression that does y = x @ W^T + bias.
 */
class ANELinearOp(
    val x: Expr,
    val weight: Parameter,
    val bias: Parameter?,
    val useANE: Boolean
) : Expr() {

    override val value: Tensor = computeForward()

    init {
        parents = if (bias != null) listOf(x, weight, bias) else listOf(x, weight)
        backwardFn = { dy ->
            val (batch, outDim) = dy.shape
            val inDim = weight.value.shape[1]

            // dx = dy @ W   (dy:[B, outDim] @ W:[outDim, inDim] → dx:[B, inDim])
            // Skipped for data inputs: nobody reads their gradient.
            if (x !is Variable && x !is Constant) {
                val dx = ane { ANEMatmul.product(dy.data, batch, outDim, weight.value.data, outDim, inDim) }
                    ?.let { Tensor(it, intArrayOf(batch, inDim)) }
                    ?: cpuMM(dy, weight.value)
                x.accumGrad(dx)
            }

            // dW = dy^T @ x  (dy^T:[outDim, B] @ x:[B, inDim] → dW:[outDim, inDim])
            val dW = ane { ANEMatmul.product(dy.data, batch, outDim, x.value.data, batch, inDim, transposeA = true) }
                ?.let { Tensor(it, intArrayOf(outDim, inDim)) }
                ?: cpuMM(transposeT(dy), x.value)
            weight.accumGrad(dW)

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

    private inline fun ane(block: () -> FloatArray?): FloatArray? = if (useANE) block() else null

    private fun computeForward(): Tensor {
        val (batch, inDim) = x.value.shape
        val outDim = weight.value.shape[0]

        // y = x @ W^T
        val y = (if (batch > 0) ane {
            ANEMatmul.product(x.value.data, batch, inDim, weight.value.data, outDim, inDim, transposeB = true)
        } else null) ?: cpuForward(batch, inDim, outDim)

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

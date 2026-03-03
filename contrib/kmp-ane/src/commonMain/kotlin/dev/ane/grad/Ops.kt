// Ops.kt — Standalone CPU operations for manual forward/backward passes.
// These are used by the macosMain training loop where ops are called
// explicitly (not through the KotlinGrad DSL or Autograd tape).
//
// Namespaced to avoid clashes with internal functions in KotlinGrad.kt
// and Autograd.kt.
package dev.ane.grad

import kotlin.math.*

// ---------------------------------------------------------------------------
//  Matrix multiply: y = x @ W^T
//  x: [M, K]  W: [N, K]  → y: [M, N]
// ---------------------------------------------------------------------------
object CpuOps {

fun matmul(x: Tensor, w: Tensor): Tensor {
    val M = x.rows
    val K = x.cols
    val N = w.rows
    require(w.cols == K) { "Matmul shape mismatch: x=[${M},${K}] w=[${N},${w.cols}]" }
    val y = Tensor.zeros(M, N)
    for (m in 0 until M)
        for (n in 0 until N) {
            var sum = 0f
            for (k in 0 until K)
                sum += x[m * K + k] * w[n * K + k]
            y[m * N + n] = sum
        }
    return y
}

/** Backward for matmul: dL/dx = dL/dy @ W, dL/dW += dL/dy^T @ x */
fun cpuMatmulBackward(
    dy: Tensor, x: Tensor, w: Tensor, dw: Tensor
): Tensor {
    val M = dy.rows
    val N = dy.cols
    val K = x.cols
    // dx = dy @ W
    val dx = Tensor.zeros(M, K)
    for (m in 0 until M)
        for (k in 0 until K) {
            var sum = 0f
            for (n in 0 until N)
                sum += dy[m * N + n] * w[n * K + k]
            dx[m * K + k] = sum
        }
    // dW += dy^T @ x
    for (n in 0 until N)
        for (k in 0 until K) {
            var sum = 0f
            for (m in 0 until M)
                sum += dy[m * N + n] * x[m * K + k]
            dw[n * K + k] += sum
        }
    return dx
}

// ---------------------------------------------------------------------------
//  Bias add / backward
// ---------------------------------------------------------------------------
fun addBias(y: Tensor, bias: Tensor): Tensor {
    val M = y.rows; val N = y.cols
    require(bias.size == N)
    val out = y.copy()
    for (m in 0 until M)
        for (n in 0 until N)
            out[m * N + n] += bias[n]
    return out
}

fun biasBackward(dy: Tensor, dbias: Tensor) {
    val M = dy.rows; val N = dy.cols
    for (m in 0 until M)
        for (n in 0 until N)
            dbias[n] += dy[m * N + n]
}

// ---------------------------------------------------------------------------
//  ReLU / backward
// ---------------------------------------------------------------------------
fun relu(x: Tensor): Tensor {
    return Tensor(FloatArray(x.size) { maxOf(0f, x[it]) }, x.shape.copyOf())
}

fun reluBackward(dy: Tensor, x: Tensor): Tensor {
    return Tensor(FloatArray(dy.size) { if (x[it] > 0f) dy[it] else 0f }, dy.shape.copyOf())
}

// ---------------------------------------------------------------------------
//  Softmax cross-entropy loss + backward
// ---------------------------------------------------------------------------
/** Softmax cross-entropy loss. Returns (loss, dLogits). */
fun softmaxCrossEntropy(logits: Tensor, labels: IntArray): Pair<Float, Tensor> {
    val M = logits.rows
    val C = logits.cols
    require(labels.size == M)

    val dlogits = Tensor.zeros(M, C)
    var totalLoss = 0f

    for (m in 0 until M) {
        // Numerically stable softmax
        var mx = Float.NEGATIVE_INFINITY
        for (c in 0 until C) {
            val v = logits[m * C + c]
            if (v > mx) mx = v
        }
        var sumExp = 0f
        for (c in 0 until C) {
            val e = exp((logits[m * C + c] - mx).toDouble()).toFloat()
            dlogits[m * C + c] = e
            sumExp += e
        }
        // Normalize to get probabilities
        for (c in 0 until C) dlogits[m * C + c] /= sumExp

        // Loss: -log(prob[label])
        val prob = dlogits[m * C + labels[m]]
        totalLoss -= ln(maxOf(prob, 1e-10f))

        // Gradient: prob - one_hot(label)
        dlogits[m * C + labels[m]] -= 1f

        // Average over batch
        for (c in 0 until C) dlogits[m * C + c] /= M.toFloat()
    }

    return Pair(totalLoss / M, dlogits)
}

// ---------------------------------------------------------------------------
//  Accuracy
// ---------------------------------------------------------------------------
fun accuracy(logits: Tensor, labels: IntArray): Float {
    val M = logits.rows
    val C = logits.cols
    var correct = 0
    for (m in 0 until M) {
        var best = 0
        var bestVal = logits[m * C]
        for (c in 1 until C) {
            val v = logits[m * C + c]
            if (v > bestVal) { bestVal = v; best = c }
        }
        if (best == labels[m]) correct++
    }
    return correct.toFloat() / M
}

} // end CpuOps

// ---------------------------------------------------------------------------
//  Adam optimizer (standalone, name-based — used by manual training loops)
// ---------------------------------------------------------------------------
class AdamOptimizer(
    private val lr: Float = 0.001f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val eps: Float = 1e-8f
) {
    private var step = 0
    private val m = mutableMapOf<String, FloatArray>()
    private val v = mutableMapOf<String, FloatArray>()

    fun update(name: String, param: Tensor, grad: Tensor) {
        step++
        val n = param.size
        val mArr = m.getOrPut(name) { FloatArray(n) }
        val vArr = v.getOrPut(name) { FloatArray(n) }

        val bc1 = 1f - beta1.pow(step)
        val bc2 = 1f - beta2.pow(step)

        for (i in 0 until n) {
            mArr[i] = beta1 * mArr[i] + (1 - beta1) * grad[i]
            vArr[i] = beta2 * vArr[i] + (1 - beta2) * grad[i] * grad[i]
            val mHat = mArr[i] / bc1
            val vHat = vArr[i] / bc2
            param[i] -= lr * mHat / (sqrt(vHat) + eps)
        }
    }

    private fun Float.pow(n: Int): Float {
        var r = 1f; var b = this
        var exp = n
        while (exp > 0) { if (exp % 2 == 1) r *= b; b *= b; exp /= 2 }
        return r
    }
}

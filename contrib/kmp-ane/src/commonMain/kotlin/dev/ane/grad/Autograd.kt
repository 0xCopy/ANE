// Autograd.kt — KotlinGrad-style reverse-mode automatic differentiation tape
// Maps computation graph nodes to ANE kernels at training time.
//
// Usage (notebook style):
//   val x = Variable(inputTensor)
//   val W = Parameter(weightTensor)
//   val y = x.matmul(W).relu()
//   val loss = y.crossEntropyLoss(labels)
//   loss.backward()
//   W.grad  // ← accumulated gradient
//
package dev.ane.grad
import kotlin.math.*

// ─── Tape node ──────────────────────────────────────────────────────────
/**
 * A node in the computation graph. Each op builds a new Expr that
 * records its parents + a backward lambda.
 */
sealed class Expr {
    abstract val value: Tensor
    var grad: Tensor? = null

    /** Backward lambda: receives upstream grad, propagates to children. */
    internal var backwardFn: ((Tensor) -> Unit)? = null

    /** Accumulated list of parent→child edges for topological sort. */
    internal var parents: List<Expr> = emptyList()

    // ── Operators (build the graph) ─────────────────────────────────
    fun matmul(other: Expr): Expr = MatMulOp(this, other)
    fun relu(): Expr = ReLUOp(this)
    fun sigmoid(): Expr = SigmoidOp(this)
    fun logSoftmax(): Expr = LogSoftmaxOp(this)
    fun add(other: Expr): Expr = AddOp(this, other)
    operator fun plus(other: Expr) = add(other)
}

/** Leaf: a trainable parameter. */
class Parameter(override val value: Tensor) : Expr() {
    /** Zero the gradient before a new backward pass. */
    fun zeroGrad() { grad?.zero() }
}

/** Leaf: an input (not trainable). */
class Variable(override val value: Tensor) : Expr()

/** Constant — no gradient flows through. */
class Constant(override val value: Tensor) : Expr()

// ─── Ops ────────────────────────────────────────────────────────────────

/** y = A @ B  where A is [M,K], B is [K,N] → y is [M,N]. */
private class MatMulOp(val a: Expr, val b: Expr) : Expr() {
    override val value: Tensor = cpuMatmul(a.value, b.value)
    init {
        parents = listOf(a, b)
        backwardFn = { dy ->
            // dA = dy @ B^T
            a.accumGrad(cpuMatmul(dy, transpose(b.value)))
            // dB = A^T @ dy
            b.accumGrad(cpuMatmul(transpose(a.value), dy))
        }
    }
}

/** y = relu(x) */
private class ReLUOp(val x: Expr) : Expr() {
    override val value: Tensor = Tensor(
        FloatArray(x.value.size) { maxOf(0f, x.value[it]) },
        x.value.shape.copyOf()
    )
    init {
        parents = listOf(x)
        backwardFn = { dy ->
            val dx = Tensor(
                FloatArray(dy.size) { if (x.value[it] > 0f) dy[it] else 0f },
                dy.shape.copyOf()
            )
            x.accumGrad(dx)
        }
    }
}

/** y = sigmoid(x) */
private class SigmoidOp(val x: Expr) : Expr() {
    override val value: Tensor = Tensor(
        FloatArray(x.value.size) { 1f / (1f + exp(-x.value[it])) },
        x.value.shape.copyOf()
    )
    init {
        parents = listOf(x)
        backwardFn = { dy ->
            val dx = Tensor(FloatArray(dy.size) {
                val s = value[it]
                dy[it] * s * (1f - s)
            }, dy.shape.copyOf())
            x.accumGrad(dx)
        }
    }
}

/** y = log_softmax(x) along last dim. Shape: [batch, classes]. */
private class LogSoftmaxOp(val x: Expr) : Expr() {
    override val value: Tensor = computeLogSoftmax(x.value)
    init {
        parents = listOf(x)
        backwardFn = { dy ->
            // d log_softmax / dx_i = dy_i - softmax(x_i) * sum(dy)
            val batch = x.value.shape[0]
            val cls = x.value.shape[1]
            val dx = Tensor.zeros(batch, cls)
            for (b in 0 until batch) {
                val off = b * cls
                var sumDy = 0f
                for (c in 0 until cls) sumDy += dy[off + c]
                for (c in 0 until cls) {
                    val softmax_c = exp(value[off + c])
                    dx[off + c] = dy[off + c] - softmax_c * sumDy
                }
            }
            x.accumGrad(dx)
        }
    }
}

/** y = a + b (element-wise, same shape). */
private class AddOp(val a: Expr, val b: Expr) : Expr() {
    override val value: Tensor = run {
        require(a.value.size == b.value.size)
        Tensor(FloatArray(a.value.size) { a.value[it] + b.value[it] }, a.value.shape.copyOf())
    }
    init {
        parents = listOf(a, b)
        backwardFn = { dy ->
            // Broadcast: if b is bias [1, N], sum dy along batch dim
            if (b.value.shape[0] == 1 && a.value.shape[0] > 1) {
                a.accumGrad(dy)
                val cols = b.value.shape[1]
                val batchGrad = FloatArray(cols)
                for (i in dy.data.indices) batchGrad[i % cols] += dy[i]
                b.accumGrad(Tensor(batchGrad, b.value.shape.copyOf()))
            } else {
                a.accumGrad(dy)
                b.accumGrad(dy)
            }
        }
    }
}

// ─── Add with broadcast (for bias) ─────────────────────────────────────

/** Broadcast-add a bias [1, N] to x [B, N]. */
fun Expr.addBias(bias: Expr): Expr = BiasAddOp(this, bias)

private class BiasAddOp(val x: Expr, val bias: Expr) : Expr() {
    override val value: Tensor = run {
        val (b, n) = x.value.shape
        val biasData = bias.value.data
        require(biasData.size == n) { "Bias size ${biasData.size} != cols $n" }
        val out = FloatArray(b * n)
        for (i in 0 until b)
            for (j in 0 until n)
                out[i * n + j] = x.value[i * n + j] + biasData[j]
        Tensor(out, intArrayOf(b, n))
    }
    init {
        parents = listOf(x, bias)
        backwardFn = { dy ->
            x.accumGrad(dy)
            // Sum gradient over batch for bias
            val n = bias.value.size
            val bGrad = FloatArray(n)
            val batch = dy.size / n
            for (i in 0 until batch)
                for (j in 0 until n)
                    bGrad[j] += dy[i * n + j]
            bias.accumGrad(Tensor(bGrad, bias.value.shape.copyOf()))
        }
    }
}

// ─── Loss functions ─────────────────────────────────────────────────────

/**
 * NLL loss: -mean(logProbs[i, labels[i]])
 * logProbs: [batch, classes] (output of logSoftmax)
 * labels: IntArray of target class indices
 * Returns a scalar Expr — call .backward() on it.
 */
fun Expr.nllLoss(labels: IntArray): NLLLossOp = NLLLossOp(this, labels)

class NLLLossOp(val logProbs: Expr, val labels: IntArray) : Expr() {
    override val value: Tensor = run {
        val batch = logProbs.value.shape[0]
        val cls = logProbs.value.shape[1]
        var loss = 0f
        for (b in 0 until batch) {
            loss -= logProbs.value[b * cls + labels[b]]
        }
        Tensor(floatArrayOf(loss / batch), intArrayOf(1))
    }
    init {
        parents = listOf(logProbs)
        backwardFn = { _ ->
            val batch = logProbs.value.shape[0]
            val cls = logProbs.value.shape[1]
            val dx = Tensor.zeros(batch, cls)
            for (b in 0 until batch) {
                dx[b * cls + labels[b]] = -1f / batch
            }
            logProbs.accumGrad(dx)
        }
    }
}

// ─── Backward pass ──────────────────────────────────────────────────────

/** Accumulate gradient into this node. */
internal fun Expr.accumGrad(upstream: Tensor) {
    val g = grad
    if (g == null) {
        grad = upstream.copy()
    } else {
        g.addInPlace(upstream)
    }
}

/**
 * Run backward pass from this (loss) node.
 * Topologically sorts the graph, then calls backwardFn in reverse.
 */
fun Expr.backward() {
    // Seed with 1.0
    this.grad = Tensor.ones(*value.shape)

    // Topological sort
    val order = mutableListOf<Expr>()
    val visited = mutableSetOf<Expr>()
    fun topo(node: Expr) {
        if (node in visited) return
        visited.add(node)
        for (p in node.parents) topo(p)
        order.add(node)
    }
    topo(this)

    // Backward in reverse topological order
    for (node in order.reversed()) {
        val g = node.grad ?: continue
        node.backwardFn?.invoke(g)
    }
}

// ─── CPU helpers ────────────────────────────────────────────────────────

/** CPU matmul: A[M,K] @ B[K,N] → C[M,N]. */
private fun cpuMatmul(a: Tensor, b: Tensor): Tensor {
    val m = a.shape[0]; val k = a.shape[1]
    val n = b.shape[1]
    require(b.shape[0] == k) { "Inner dim mismatch: ${a.shape.toList()} @ ${b.shape.toList()}" }
    val out = FloatArray(m * n)
    for (i in 0 until m)
        for (j in 0 until n) {
            var s = 0f
            for (p in 0 until k) s += a[i * k + p] * b[p * n + j]
            out[i * n + j] = s
        }
    return Tensor(out, intArrayOf(m, n))
}

/** Transpose [M,N] → [N,M]. */
private fun transpose(t: Tensor): Tensor {
    val (m, n) = t.shape
    val out = FloatArray(m * n)
    for (i in 0 until m)
        for (j in 0 until n)
            out[j * m + i] = t[i * n + j]
    return Tensor(out, intArrayOf(n, m))
}

/** Compute log-softmax along last dim on [batch, classes]. */
private fun computeLogSoftmax(x: Tensor): Tensor {
    val (batch, cls) = x.shape
    val out = FloatArray(batch * cls)
    for (b in 0 until batch) {
        val off = b * cls
        var mx = Float.NEGATIVE_INFINITY
        for (c in 0 until cls) if (x[off + c] > mx) mx = x[off + c]
        var sumExp = 0f
        for (c in 0 until cls) sumExp += exp(x[off + c] - mx)
        val logSumExp = mx + ln(sumExp)
        for (c in 0 until cls) out[off + c] = x[off + c] - logSumExp
    }
    return Tensor(out, x.shape.copyOf())
}

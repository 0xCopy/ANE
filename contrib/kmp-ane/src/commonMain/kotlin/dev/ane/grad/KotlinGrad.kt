// KotlinGrad.kt — KotlinGrad-style differentiable DSL with scoped ANE driver
//
// TWO LAYERS:
//   1. Public KotlinGrad DSL   — user-facing symbolic AD (inspired by github.com/breandan/kotlingrad)
//   2. Private ANE driver ops  — hardware dispatch hidden behind scoped receivers
//
// Usage (mimics KotlinGrad scalarFun / vectorFun style):
//
//   val loss = differentiable {          // ← opens GradScope
//       val x = input(batchImages)       // leaf variable
//       val h = (x matmul W1).relu()     // symbolic graph + ANE dispatch
//       val logits = h matmul W2
//       logits.crossEntropy(labels)       // returns scalar loss Var
//   }
//   loss.backward()                      // reverse-mode AD
//   W1.grad, W2.grad                     // accumulated gradients
//
package dev.ane.grad

import dev.ane.runtime.ANEDevice
import dev.ane.runtime.ANEKernelHandle
import dev.ane.runtime.ANEStatus
import kotlin.math.*

// Platform-agnostic monotonic nanosecond timer
private val timerOrigin = kotlin.time.TimeSource.Monotonic.markNow()
private fun nanoTime(): Long = timerOrigin.elapsedNow().inWholeNanoseconds

// ═══════════════════════════════════════════════════════════════════════
//  § 1. PUBLIC DSL — KotlinGrad-style differentiable expressions
// ═══════════════════════════════════════════════════════════════════════

/**
 * A differentiable variable in the computation graph.
 * Immutable value, lazily-accumulated gradient.
 */
class Var(
    val data: Tensor,
    /** Parent edges for topological sort */
    internal val parents: List<Var> = emptyList(),
    /** Human-readable label for debugging */
    val label: String = "",
    /** Backward closure: receives upstream gradient, propagates to parents.
     *  Last so call sites can pass it as a trailing lambda: Var(y, parents, "op") { dy -> ... } */
    internal val backwardFn: ((Tensor) -> Unit)? = null
) {
    /** Accumulated gradient (same shape as data) */
    var grad: Tensor? = null
        internal set

    internal fun accumGrad(upstream: Tensor) {
        val g = grad
        if (g == null) grad = upstream.copy()
        else g.addInPlace(upstream)
    }

    /** Convenience accessors */
    val shape: IntArray get() = data.shape
    val rows: Int get() = data.rows
    val cols: Int get() = data.cols

    override fun toString(): String =
        "Var(${shape.toList()}, label=$label, hasGrad=${grad != null})"
}

/**
 * A trainable parameter — wraps a Var with optimizer metadata.
 */
class Param(
    val weight: Var,
    val name: String
) {
    val data: Tensor get() = weight.data
    val grad: Tensor? get() = weight.grad
    fun zeroGrad() { weight.grad?.zero() }

    companion object {
        fun kaiming(name: String, outDim: Int, inDim: Int): Param {
            val t = Tensor.kaiming(outDim, inDim)
            return Param(Var(t, label = name), name)
        }

        fun zeros(name: String, vararg shape: Int): Param {
            val t = Tensor.zeros(*shape)
            return Param(Var(t, label = name), name)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  § 2. GRAD SCOPE — scoped receiver for building the graph
// ═══════════════════════════════════════════════════════════════════════

/**
 * Scoped DSL receiver. All differentiable ops execute inside this scope.
 * The scope owns the ANE kernel cache and dispatches matmuls to hardware.
 */
class GradScope internal constructor() {

    // ── Private ANE driver state (not visible to DSL users) ─────────
    private val _aneKernels = mutableMapOf<String, ANEKernelHandle>()
    private var _aneAvailable = false
    private var _batchSize = -1

    /** Timing accumulators for ANE vs CPU (nanoseconds) */
    internal var aneTimeNs: Long = 0L
        private set
    internal var cpuTimeNs: Long = 0L
        private set

    // ── Lifecycle ───────────────────────────────────────────────────

    /** Try to initialize ANE. Called automatically. */
    internal fun initANE(): Boolean {
        if (_aneAvailable) return true
        _aneAvailable = (ANEDevice.init() == ANEStatus.OK && ANEDevice.isAvailable())
        return _aneAvailable
    }

    /** Release all ANE kernels. Called when scope closes. */
    internal fun close() {
        for ((_, k) in _aneKernels) ANEDevice.freeKernel(k)
        _aneKernels.clear()
    }

    // ── Private ANE dispatch (hidden from public DSL) ───────────────

    /**
     * Compile or retrieve a cached ANE conv kernel for the given
     * weight shape. Returns null if ANE unavailable.
     */
    private fun aneCompileConv(
        key: String, weights: FloatArray,
        inCh: Int, outCh: Int, spatial: Int
    ): ANEKernelHandle? {
        if (!_aneAvailable) return null

        // Recompile if batch size changed
        if (spatial != _batchSize) {
            for ((_, k) in _aneKernels) ANEDevice.freeKernel(k)
            _aneKernels.clear()
            _batchSize = spatial
        }

        // remove() not get(): if the compile below fails (ANE per-process compile limit) the map must not keep
        // the freed handle, or the next call frees it a second time and crashes in objc_msgSend.
        _aneKernels.remove(key)?.let { ANEDevice.freeKernel(it) }
        val kernel = ANEDevice.compileConv(weights, inCh, outCh, spatial)
        if (kernel != null) _aneKernels[key] = kernel
        return kernel
    }

    /**
     * Execute a matmul on ANE: y = x @ W^T
     * x: [M, K],  W: [N, K] → y: [M, N]
     * Returns null if ANE is not available or eval fails.
     */
    private fun aneMatmul(
        key: String, x: Tensor, w: Tensor
    ): Tensor? {
        if (!_aneAvailable) return null
        val (M, K) = x.shape
        val N = w.rows
        val kernel = aneCompileConv(key, w.data, K, N, M) ?: return null

        val t0 = nanoTime()
        val result = ANEDevice.convEval(kernel, x.data, M, K, N)
        aneTimeNs += nanoTime() - t0

        return result?.let { Tensor(it, intArrayOf(M, N)) }
    }

    // ── Public DSL operators ────────────────────────────────────────

    /** Wrap raw tensor data as a non-trainable input variable. */
    fun input(data: Tensor, label: String = "input"): Var =
        Var(data, label = label)

    /** Wrap a Param as a Var in this scope. */
    fun param(p: Param): Var = p.weight

    /**
     * Matrix multiply: y = a @ b^T
     * a: [M, K],  b: [N, K]  →  y: [M, N]
     *
     * Dispatches to ANE when available; falls back to CPU.
     * Backward: dA = dY @ B,  dB = dY^T @ A
     */
    infix fun Var.matmul(w: Var): Var {
        val a = this
        val M = a.rows; val K = a.cols
        val N = w.rows

        // Forward: try ANE, fall back to CPU
        val aneKey = "${w.label}_${M}x${K}x${N}"
        val y = aneMatmul(aneKey, a.data, w.data) ?: run {
            val t0 = nanoTime()
            val result = gradScopeMatmul(a.data, w.data)
            cpuTimeNs += nanoTime() - t0
            result
        }

        return Var(y, parents = listOf(a, w), label = "matmul") { dy ->
            // dA = dY @ W  (not transposed: [M,N] @ [N,K] → [M,K])
            val dA = gradScopeMatmulUntransposed(dy, w.data)
            a.accumGrad(dA)
            // dW = dY^T @ A  → [N,M] @ [M,K] → [N,K]
            val dW = gradScopeTransposeMatmul(dy, a.data)
            w.accumGrad(dW)
        }
    }

    /**
     * Bias add: y = x + bias (broadcast along batch dim)
     * x: [M, N],  bias: [1, N] or [N]  → y: [M, N]
     */
    fun Var.addBias(bias: Var): Var {
        val x = this
        val (M, N) = x.data.shape
        val out = FloatArray(M * N)
        val bd = bias.data.data
        for (i in 0 until M)
            for (j in 0 until N)
                out[i * N + j] = x.data[i * N + j] + bd[j]
        return Var(Tensor(out, intArrayOf(M, N)), listOf(x, bias), "bias_add") { dy ->
            // dx = dy (pass through)
            x.accumGrad(dy)
            // dbias = sum over batch dim
            val db = FloatArray(N)
            for (i in 0 until M)
                for (j in 0 until N)
                    db[j] += dy[i * N + j]
            bias.accumGrad(Tensor(db, bias.data.shape.copyOf()))
        }
    }

    /** ReLU activation */
    fun Var.relu(): Var {
        val x = this
        val out = Tensor(
            FloatArray(x.data.size) { maxOf(0f, x.data[it]) },
            x.data.shape.copyOf()
        )
        return Var(out, listOf(x), "relu") { dy ->
            val dx = Tensor(
                FloatArray(dy.size) { if (x.data[it] > 0f) dy[it] else 0f },
                dy.shape.copyOf()
            )
            x.accumGrad(dx)
        }
    }

    /** Log-softmax along last dimension */
    fun Var.logSoftmax(): Var {
        val x = this
        val (batch, cls) = x.data.shape
        val out = FloatArray(batch * cls)
        for (b in 0 until batch) {
            val off = b * cls
            var mx = Float.NEGATIVE_INFINITY
            for (c in 0 until cls) if (x.data[off + c] > mx) mx = x.data[off + c]
            var sumExp = 0f
            for (c in 0 until cls) sumExp += exp(x.data[off + c] - mx)
            val logSumExp = mx + ln(sumExp)
            for (c in 0 until cls) out[off + c] = x.data[off + c] - logSumExp
        }
        val logSm = Tensor(out, intArrayOf(batch, cls))

        return Var(logSm, listOf(x), "log_softmax") { dy ->
            val dx = Tensor.zeros(batch, cls)
            for (b in 0 until batch) {
                val off = b * cls
                var sumDy = 0f
                for (c in 0 until cls) sumDy += dy[off + c]
                for (c in 0 until cls) {
                    val sm = exp(logSm[off + c])
                    dx[off + c] = dy[off + c] - sm * sumDy
                }
            }
            x.accumGrad(dx)
        }
    }

    /**
     * Softmax cross-entropy loss (fused for numerical stability).
     * logits: [batch, classes],  labels: IntArray of target indices.
     * Returns a (scalar) Var whose backward produces dLogits.
     */
    fun Var.crossEntropy(labels: IntArray): Var {
        val logits = this
        val (batch, cls) = logits.data.shape
        val probs = FloatArray(batch * cls)
        var totalLoss = 0f

        for (b in 0 until batch) {
            val off = b * cls
            var mx = Float.NEGATIVE_INFINITY
            for (c in 0 until cls) if (logits.data[off + c] > mx) mx = logits.data[off + c]
            var sumExp = 0f
            for (c in 0 until cls) {
                probs[off + c] = exp(logits.data[off + c] - mx)
                sumExp += probs[off + c]
            }
            for (c in 0 until cls) probs[off + c] /= sumExp
            totalLoss -= ln(maxOf(probs[off + labels[b]], 1e-10f))
        }
        val loss = totalLoss / batch

        return Var(
            Tensor(floatArrayOf(loss), intArrayOf(1)),
            listOf(logits),
            "cross_entropy"
        ) { _ ->
            // dLogits = (softmax - one_hot) / batch
            val dL = Tensor(probs.copyOf(), intArrayOf(batch, cls))
            for (b in 0 until batch) dL[b * cls + labels[b]] -= 1f
            dL.scaleInPlace(1f / batch)
            logits.accumGrad(dL)
        }
    }

    /** Element-wise add (same shape) */
    operator fun Var.plus(other: Var): Var {
        val a = this
        require(a.data.size == other.data.size)
        val out = Tensor(
            FloatArray(a.data.size) { a.data[it] + other.data[it] },
            a.data.shape.copyOf()
        )
        return Var(out, listOf(a, other), "add") { dy ->
            a.accumGrad(dy)
            other.accumGrad(dy)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  § 3. DSL ENTRY POINTS
// ═══════════════════════════════════════════════════════════════════════

/**
 * Open a differentiable scope. All ops inside enjoy:
 *   - ANE-accelerated matmul dispatch (private, auto-cached)
 *   - Automatic reverse-mode differentiation
 *
 * ```
 * val loss = differentiable {
 *     val x = input(images)
 *     val h = (x matmul W1).relu()
 *     val logits = h matmul W2
 *     logits.crossEntropy(labels)
 * }
 * ```
 */
fun differentiable(block: GradScope.() -> Var): Var {
    val scope = GradScope()
    scope.initANE()
    try {
        return scope.block()
    } finally {
        scope.close()
    }
}

/**
 * Persistent scope — keeps ANE kernels alive across training steps.
 * Call [TrainingScope.close] when done.
 */
class TrainingScope {
    val scope = GradScope()
    init { scope.initANE() }

    /** Run one forward pass inside this scope. */
    inline fun forward(block: GradScope.() -> Var): Var = scope.block()

    /** ANE time this scope has accumulated (nanoseconds). */
    val aneTimeNs: Long get() = scope.aneTimeNs

    /** CPU time this scope has accumulated (nanoseconds). */
    val cpuTimeNs: Long get() = scope.cpuTimeNs

    fun close() = scope.close()
}

// ═══════════════════════════════════════════════════════════════════════
//  § 4. BACKWARD — reverse-mode AD
// ═══════════════════════════════════════════════════════════════════════

/**
 * Run reverse-mode automatic differentiation from this node.
 * Seeds gradient with 1.0 for scalar loss, or identity for vector outputs.
 */
fun Var.backward() {
    // Seed
    this.grad = Tensor.ones(*data.shape)

    // Topological sort
    val order = mutableListOf<Var>()
    val visited = mutableSetOf<Var>()
    fun topo(node: Var) {
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

// ═══════════════════════════════════════════════════════════════════════
//  § 5. PRIVATE CPU MATMUL VARIANTS (for backward pass)
// ═══════════════════════════════════════════════════════════════════════

/**
 * y = A @ B where A:[M,K], B:[N,K] → y:[M,N]
 * B is stored row-major as [N,K] and we treat it as B^T implicitly:
 * y[m,n] = Σ_k A[m,k] * B[n,k]
 *
 * (This matches the forward convention: y = x @ W^T)
 */
private fun gradScopeMatmul(x: Tensor, w: Tensor): Tensor {
    val M = x.rows; val K = x.cols
    val N = w.rows
    val y = FloatArray(M * N)
    for (m in 0 until M)
        for (n in 0 until N) {
            var s = 0f
            for (k in 0 until K)
                s += x[m * K + k] * w[n * K + k]
            y[m * N + n] = s
        }
    return Tensor(y, intArrayOf(M, N))
}

/**
 * dA = dY @ W  (no transpose on W)
 * dY:[M,N], W:[N,K] → dA:[M,K]
 * dA[m,k] = Σ_n dY[m,n] * W[n,k]
 */
private fun gradScopeMatmulUntransposed(dy: Tensor, w: Tensor): Tensor {
    val M = dy.rows; val N = dy.cols
    val K = w.cols
    val dA = FloatArray(M * K)
    for (m in 0 until M)
        for (k in 0 until K) {
            var s = 0f
            for (n in 0 until N)
                s += dy[m * N + n] * w[n * K + k]
            dA[m * K + k] = s
        }
    return Tensor(dA, intArrayOf(M, K))
}

/**
 * dW = dY^T @ A
 * dY:[M,N], A:[M,K] → dW:[N,K]
 * dW[n,k] = Σ_m dY[m,n] * A[m,k]
 */
private fun gradScopeTransposeMatmul(dy: Tensor, a: Tensor): Tensor {
    val M = dy.rows; val N = dy.cols
    val K = a.cols
    val dW = FloatArray(N * K)
    for (n in 0 until N)
        for (k in 0 until K) {
            var s = 0f
            for (m in 0 until M)
                s += dy[m * N + n] * a[m * K + k]
            dW[n * K + k] = s
        }
    return Tensor(dW, intArrayOf(N, K))
}

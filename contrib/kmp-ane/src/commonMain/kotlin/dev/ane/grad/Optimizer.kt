// Optimizer.kt — SGD and Adam optimizers for Parameter nodes
package dev.ane.grad

import kotlin.math.*

/**
 * Optimizer interface — call step() after backward() to update parameters.
 */
interface Optimizer {
    fun step()
    fun zeroGrad()
}

/**
 * SGD with optional momentum and weight decay.
 */
class SGD(
    private val params: List<Parameter>,
    private val lr: Float = 0.01f,
    private val momentum: Float = 0.0f,
    private val weightDecay: Float = 0.0f
) : Optimizer {
    private val velocities: List<Tensor?> = if (momentum > 0f) {
        params.map { Tensor.zeros(*it.value.shape) }
    } else {
        params.map { null }
    }

    override fun step() {
        for ((i, p) in params.withIndex()) {
            val g = p.grad ?: continue

            // Weight decay
            if (weightDecay > 0f) {
                for (j in g.data.indices) g.data[j] += weightDecay * p.value.data[j]
            }

            if (momentum > 0f) {
                val v = velocities[i]!!
                for (j in v.data.indices) {
                    v.data[j] = momentum * v.data[j] + g.data[j]
                    p.value.data[j] -= lr * v.data[j]
                }
            } else {
                for (j in p.value.data.indices) {
                    p.value.data[j] -= lr * g.data[j]
                }
            }
        }
    }

    override fun zeroGrad() = params.forEach { it.zeroGrad() }
}

/**
 * Adam optimizer (Kingma & Ba, 2014).
 */
class Adam(
    private val params: List<Parameter>,
    private val lr: Float = 0.001f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val eps: Float = 1e-8f,
    private val weightDecay: Float = 0.0f
) : Optimizer {
    private var t = 0
    private val m: List<FloatArray> = params.map { FloatArray(it.value.size) }
    private val v: List<FloatArray> = params.map { FloatArray(it.value.size) }

    override fun step() {
        t++
        val bc1 = 1f - beta1.pow(t)
        val bc2 = 1f - beta2.pow(t)

        for ((i, p) in params.withIndex()) {
            val g = p.grad ?: continue
            val mi = m[i]; val vi = v[i]

            for (j in p.value.data.indices) {
                var gj = g.data[j]
                if (weightDecay > 0f) gj += weightDecay * p.value.data[j]

                mi[j] = beta1 * mi[j] + (1f - beta1) * gj
                vi[j] = beta2 * vi[j] + (1f - beta2) * gj * gj

                val mHat = mi[j] / bc1
                val vHat = vi[j] / bc2
                p.value.data[j] -= lr * mHat / (sqrt(vHat) + eps)
            }
        }
    }

    override fun zeroGrad() = params.forEach { it.zeroGrad() }
}

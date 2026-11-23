// MnistModel.kt — A 3-layer MLP for MNIST on ANE
// Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (LogSoftmax)
// Forward and backward matmuls dispatch to the ANE (see grad/ANEMatmul.kt); the rest is CPU.
package dev.ane.mnist

import dev.ane.grad.*

/**
 * MNIST MLP with ANE-accelerated linear layers.
 *
 * ```
 * val model = MnistModel(useANE = true)
 * val (loss, logits) = model.forward(xBatch, labels)
 * loss.backward()
 * optimizer.step()
 * ```
 */
class MnistModel(
    val useANE: Boolean = true,
    hiddenSize1: Int = 128,
    hiddenSize2: Int = 64
) {
    // Layer 1: 784 → hidden1
    val w1 = Parameter(Tensor.kaiming(hiddenSize1, 784))
    val b1 = Parameter(Tensor.zeros(1, hiddenSize1))

    // Layer 2: hidden1 → hidden2
    val w2 = Parameter(Tensor.kaiming(hiddenSize2, hiddenSize1))
    val b2 = Parameter(Tensor.zeros(1, hiddenSize2))

    // Layer 3: hidden2 → 10
    val w3 = Parameter(Tensor.kaiming(10, hiddenSize2))
    val b3 = Parameter(Tensor.zeros(1, 10))

    /** All trainable parameters (for the optimizer). */
    val parameters: List<Parameter> = listOf(w1, b1, w2, b2, w3, b3)

    /**
     * Forward pass: returns (loss, logits_expr).
     *
     * @param x   Input expression [batch, 784]
     * @param labels  Target class indices [batch]
     * @return Pair of NLL loss expression and logits expression
     */
    fun forward(x: Expr, labels: IntArray): Pair<NLLLossOp, Expr> {
        // Layer 1: linear + ReLU
        val h1 = x.aneLinear(w1, b1, useANE).relu()

        // Layer 2: linear + ReLU
        val h2 = h1.aneLinear(w2, b2, useANE).relu()

        // Layer 3: linear → logSoftmax
        val logits = h2.aneLinear(w3, b3, useANE)
        val logProbs = logits.logSoftmax()

        // NLL Loss
        val loss = logProbs.nllLoss(labels)

        return loss to logits
    }

    /** Count total parameters. */
    fun paramCount(): Int = parameters.sumOf { it.value.size }

    /**
     * Compute accuracy on a batch: compare argmax(logits) vs labels.
     */
    fun accuracy(logits: Expr, labels: IntArray): Float {
        val batch = logits.value.shape[0]
        val cls = logits.value.shape[1]
        var correct = 0
        for (b in 0 until batch) {
            val off = b * cls
            var maxIdx = 0; var maxVal = logits.value[off]
            for (c in 1 until cls) {
                if (logits.value[off + c] > maxVal) {
                    maxVal = logits.value[off + c]; maxIdx = c
                }
            }
            if (maxIdx == labels[b]) correct++
        }
        return correct.toFloat() / batch
    }
}

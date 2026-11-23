// MnistTrainer.kt — Shared configuration for MNIST training on ANE
//
// The actual training entry point is in macosMain/MnistTrainer.kt
// (Kotlin/Native with ANE dispatch).
//
// This file provides common configuration and utility functions
// shared across platform-specific implementations.
//
// Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10
// Forward:  ANE matmul kernels (weights passed as kernel inputs, compiled once per shape)
// Backward: reverse-mode AD via Autograd tape; its matmuls run on the ANE as well
// Optimizer: Adam (lr=0.001)
//
package dev.ane.mnist

import dev.ane.grad.*

// ─────────────────────────────────────────────────────────────────
//  Training hyperparameters — shared across all platform targets
// ─────────────────────────────────────────────────────────────────

object TrainConfig {
    const val INPUT_DIM   = 784
    const val HIDDEN1     = 128
    const val HIDDEN2     = 64
    const val NUM_CLASSES = 10
    const val BATCH_SIZE  = 64
    const val EPOCHS      = 10
    const val LR          = 0.001f
    const val TRAIN_N     = 60000
    const val TEST_N      = 10000
}

// ─────────────────────────────────────────────────────────────────
//  Common accuracy computation
// ─────────────────────────────────────────────────────────────────

/** Argmax accuracy: compare predicted class vs target labels. */
fun computeAccuracy(logits: Tensor, labels: IntArray): Float {
    val batch = logits.shape[0]
    val cls = logits.shape[1]
    var correct = 0
    for (b in 0 until batch) {
        var best = 0; var bestVal = logits[b * cls]
        for (c in 1 until cls) {
            val v = logits[b * cls + c]
            if (v > bestVal) { bestVal = v; best = c }
        }
        if (best == labels[b]) correct++
    }
    return correct.toFloat() / batch
}

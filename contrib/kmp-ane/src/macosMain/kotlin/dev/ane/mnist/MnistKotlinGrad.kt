// MnistKotlinGrad.kt — MNIST trained with the KotlinGrad scoped DSL + ANE
//
// This is the "notebook-style" trainer:
//   - Each section is a logical cell (marked with box headers)
//   - Stdout is CSV-parseable for the Jupyter orchestrator
//   - Uses KotlinGrad.kt scoped DSL (differentiable {}, TrainingScope)
//   - ANE dispatch is fully private inside GradScope
//
// Compared to the original MnistTrainer.kt (Autograd tape + aneLinear):
//   - Uses `GradScope.matmul` instead of `Expr.aneLinear()`
//   - Weights are `Param` (not `Parameter`), gradients come from `Var.backward()`
//   - TrainingScope keeps ANE kernels alive across steps (zero recompiles)
//   - ParamAdam works directly on Param.grad (no adapter layer)
//
// Data: reads flat binary from prepare_mnist.py:
//   data/mnist_train_images.bin — 60000 × 784 × float32
//   data/mnist_train_labels.bin — 60000 × uint8
//   data/mnist_test_images.bin  — 10000 × 784 × float32
//   data/mnist_test_labels.bin  — 10000 × uint8
//
package dev.ane.mnist

import dev.ane.grad.*
import dev.ane.runtime.ANEDevice
import dev.ane.runtime.ANEStatus
import kotlinx.cinterop.*
import platform.posix.*
import kotlin.math.*

// ╔═══════════════════════════════════════════════════════════════╗
// ║  Cell 1: Configuration                                       ║
// ╚═══════════════════════════════════════════════════════════════╝

object KG {
    const val INPUT_DIM   = 784
    const val HIDDEN_DIM  = 128
    const val OUTPUT_DIM  = 10
    const val BATCH_SIZE  = 128
    const val N_EPOCHS    = 10
    const val LR          = 0.001f
    const val TRAIN_N     = 60000
    const val TEST_N      = 10000
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║  Cell 2: Data Loading (Kotlin/Native POSIX)                  ║
// ╚═══════════════════════════════════════════════════════════════╝

@OptIn(ExperimentalForeignApi::class)
private fun loadF32(path: String, count: Int): FloatArray {
    val f = fopen(path, "rb") ?: error("Cannot open $path\nRun: python3 prepare_mnist.py")
    val buf = FloatArray(count)
    buf.usePinned { pinned ->
        fread(pinned.addressOf(0), 4u, count.toULong(), f)
    }
    fclose(f)
    return buf
}

@OptIn(ExperimentalForeignApi::class)
private fun loadU8(path: String, count: Int): IntArray {
    val f = fopen(path, "rb") ?: error("Cannot open $path")
    val raw = ByteArray(count)
    raw.usePinned { pinned ->
        fread(pinned.addressOf(0), 1u, count.toULong(), f)
    }
    fclose(f)
    return IntArray(count) { raw[it].toInt() and 0xFF }
}

/** Find data directory (searching several candidate paths). */
private fun findDataDir(): String {
    val candidates = listOf(
        "data",
        "contrib/kmp-ane/data",
        "../data",
        "mnist/data",
        "."
    )
    return candidates.firstOrNull { dir ->
        access("$dir/mnist_train_images.bin", F_OK) == 0
    } ?: error(
        "MNIST data not found. Run:\n" +
        "  cd contrib/kmp-ane && python3 prepare_mnist.py"
    )
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║  Cell 3: Adam Optimizer for KotlinGrad Params                ║
// ╚═══════════════════════════════════════════════════════════════╝

/**
 * Adam (Kingma & Ba, 2014) that works directly on KotlinGrad [Param].
 * Reads gradients from `Param.weight.grad` after `Var.backward()`.
 */
class ParamAdam(
    private val params: List<Param>,
    private val lr: Float = 0.001f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val eps: Float = 1e-8f
) {
    private var t = 0
    private val m = params.map { FloatArray(it.data.size) }
    private val v = params.map { FloatArray(it.data.size) }

    fun step() {
        t++
        val bc1 = 1f - beta1.pow(t)
        val bc2 = 1f - beta2.pow(t)

        for ((i, p) in params.withIndex()) {
            val g = p.grad ?: continue
            val mi = m[i]; val vi = v[i]
            val w = p.data.data
            val gd = g.data

            for (j in w.indices) {
                mi[j] = beta1 * mi[j] + (1f - beta1) * gd[j]
                vi[j] = beta2 * vi[j] + (1f - beta2) * gd[j] * gd[j]
                val mHat = mi[j] / bc1
                val vHat = vi[j] / bc2
                w[j] -= lr * mHat / (sqrt(vHat) + eps)
            }
        }
    }
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║  Cell 4: Model (KotlinGrad Params + scoped forward)          ║
// ╚═══════════════════════════════════════════════════════════════╝

/**
 * Two-layer MNIST MLP using KotlinGrad [Param] weights.
 *
 * The forward pass is defined as a GradScope extension:
 * matmul dispatch to ANE is handled privately by the scope.
 * User code is pure symbolic math — no hardware references.
 */
class KGModel {
    val W1 = Param.kaiming("W1", KG.HIDDEN_DIM, KG.INPUT_DIM)  // [128, 784]
    val b1 = Param.zeros("b1", 1, KG.HIDDEN_DIM)               // [1, 128]
    val W2 = Param.kaiming("W2", KG.OUTPUT_DIM, KG.HIDDEN_DIM) // [10, 128]
    val b2 = Param.zeros("b2", 1, KG.OUTPUT_DIM)               // [1, 10]

    val params = listOf(W1, b1, W2, b2)

    fun paramCount(): Int = params.sumOf { it.data.size }
    fun zeroGrad() = params.forEach { it.zeroGrad() }

    /**
     * Forward → loss (used for training).
     *
     * Inside the GradScope:
     *   - `x matmul param(W1)` dispatches to ANE (private) or CPU
     *   - `.addBias()` / `.relu()` / `.crossEntropy()` are always CPU
     *   - backward closures are captured automatically
     */
    fun GradScope.forward(x: Tensor, labels: IntArray): Var {
        val inp = input(x, "images")                // [B, 784]

        // Layer 1: Linear + ReLU
        val h = (inp matmul param(W1))              // [B, 128] ← ANE
            .addBias(param(b1))
            .relu()

        // Layer 2: Linear + softmax CE loss
        val logits = (h matmul param(W2))           // [B, 10]  ← ANE
            .addBias(param(b2))

        return logits.crossEntropy(labels)           // scalar
    }

    /** Forward → raw logits (used for eval; no loss). */
    fun GradScope.logits(x: Tensor): Var {
        val inp = input(x, "eval")
        val h = (inp matmul param(W1)).addBias(param(b1)).relu()
        return (h matmul param(W2)).addBias(param(b2))
    }
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║  Cell 5: Evaluation                                          ║
// ╚═══════════════════════════════════════════════════════════════╝

/**
 * Compute accuracy over a dataset in batches.
 * Uses the persistent TrainingScope (ANE kernels stay warm).
 */
fun evalAccuracy(
    model: KGModel,
    images: FloatArray, labels: IntArray, count: Int,
    scope: TrainingScope
): Float {
    var correct = 0
    var total = 0
    val bs = KG.BATCH_SIZE

    for (start in 0 until count - bs + 1 step bs) {
        val bx = FloatArray(bs * KG.INPUT_DIM)
        val bl = IntArray(bs)
        for (i in 0 until bs) {
            images.copyInto(bx, i * KG.INPUT_DIM, (start + i) * KG.INPUT_DIM, (start + i + 1) * KG.INPUT_DIM)
            bl[i] = labels[start + i]
        }
        val x = Tensor(bx, intArrayOf(bs, KG.INPUT_DIM))

        val logits = scope.forward {
            with(model) { logits(x) }
        }

        // Argmax
        for (b in 0 until bs) {
            var best = 0; var bestVal = logits.data[b * KG.OUTPUT_DIM]
            for (c in 1 until KG.OUTPUT_DIM) {
                val v = logits.data[b * KG.OUTPUT_DIM + c]
                if (v > bestVal) { bestVal = v; best = c }
            }
            if (best == bl[b]) correct++
            total++
        }
    }
    return if (total > 0) correct.toFloat() / total * 100f else 0f
}

// ╔═══════════════════════════════════════════════════════════════╗
// ║  Cell 6: Training Loop                                       ║
// ╚═══════════════════════════════════════════════════════════════╝

/** Deterministic xorshift64 PRNG. */
private var _rng = 0x5DEECE66DL
private fun rng(): Long { _rng = _rng xor (_rng shl 13); _rng = _rng xor (_rng shr 7); _rng = _rng xor (_rng shl 17); return _rng }
private fun shuffle(a: IntArray) { for (i in a.size - 1 downTo 1) { val j = ((rng() ushr 1) % (i + 1)).toInt(); val t = a[i]; a[i] = a[j]; a[j] = t } }

@OptIn(ExperimentalForeignApi::class)
fun kgMain() {
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  MNIST — KotlinGrad Scoped DSL + Apple Neural Engine       ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    println("║  Network:  ${KG.INPUT_DIM} → ${KG.HIDDEN_DIM} (ReLU) → ${KG.OUTPUT_DIM}")
    println("║  Training: Adam(lr=${KG.LR}), batch=${KG.BATCH_SIZE}, epochs=${KG.N_EPOCHS}")
    println("║  DSL:      KotlinGrad scoped AD → private ANE dispatch     ║")
    println("╚══════════════════════════════════════════════════════════════╝")
    println()

    // ── ANE status ──────────────────────────────────────────────
    val aneOK = ANEDevice.init() == ANEStatus.OK && ANEDevice.isAvailable()
    println("[ANE] ${if (aneOK) "✓ available — matmuls dispatched to Neural Engine" else "✗ not available — CPU fallback"}")

    // ── Load data ───────────────────────────────────────────────
    val dir = findDataDir()
    println("[Data] Loading from $dir/")
    val trainImages = loadF32("$dir/mnist_train_images.bin", KG.TRAIN_N * KG.INPUT_DIM)
    val trainLabels = loadU8("$dir/mnist_train_labels.bin", KG.TRAIN_N)
    val testImages  = loadF32("$dir/mnist_test_images.bin",  KG.TEST_N  * KG.INPUT_DIM)
    val testLabels  = loadU8("$dir/mnist_test_labels.bin",   KG.TEST_N)
    println("[Data] Train: ${KG.TRAIN_N}, Test: ${KG.TEST_N}")

    // ── Model + optimizer ───────────────────────────────────────
    val model = KGModel()
    println("[Model] ${model.paramCount()} parameters")
    val optimizer = ParamAdam(model.params, lr = KG.LR)

    // ── Persistent scope (ANE kernels cached across steps) ──────
    val scope = TrainingScope()

    val stepsPerEpoch = KG.TRAIN_N / KG.BATCH_SIZE
    val idx = IntArray(KG.TRAIN_N) { it }

    println()
    println("Training ${KG.N_EPOCHS} epochs × $stepsPerEpoch steps...")
    println()

    // CSV header for Jupyter parsing
    println("# CSV: epoch,step,loss,ane_ms,cpu_ms,test_acc")

    // ── Epoch loop ──────────────────────────────────────────────
    for (epoch in 1..KG.N_EPOCHS) {
        shuffle(idx)
        var epochLoss = 0.0

        @OptIn(ExperimentalForeignApi::class)
        val t0 = memScoped {
            val tv = alloc<timeval>()
            gettimeofday(tv.ptr, null)
            tv.tv_sec * 1000L + tv.tv_usec / 1000L
        }

        for (step in 0 until stepsPerEpoch) {
            // ── Gather shuffled batch ───────────────────────────
            val bx = FloatArray(KG.BATCH_SIZE * KG.INPUT_DIM)
            val bl = IntArray(KG.BATCH_SIZE)
            for (i in 0 until KG.BATCH_SIZE) {
                val s = idx[step * KG.BATCH_SIZE + i]
                trainImages.copyInto(bx, i * KG.INPUT_DIM, s * KG.INPUT_DIM, (s + 1) * KG.INPUT_DIM)
                bl[i] = trainLabels[s]
            }
            val x = Tensor(bx, intArrayOf(KG.BATCH_SIZE, KG.INPUT_DIM))

            // ── Forward (GradScope dispatches matmuls privately) ─
            model.zeroGrad()
            val loss = scope.forward {
                with(model) { forward(x, bl) }
            }

            // ── Backward (reverse-mode AD) ──────────────────────
            loss.backward()

            // ── Adam step ───────────────────────────────────────
            optimizer.step()

            epochLoss += loss.data[0]

            // Log every 100 steps
            if ((step + 1) % 100 == 0) {
                val aneMs = scope.aneTimeNs / 1_000_000.0
                val cpuMs = scope.cpuTimeNs / 1_000_000.0
                println("$epoch,${step + 1},${fixed(epochLoss / (step + 1), 6)},${fixed(aneMs, 1)},${fixed(cpuMs, 1)},")
            }
        }

        @OptIn(ExperimentalForeignApi::class)
        val elapsed = memScoped {
            val tv = alloc<timeval>()
            gettimeofday(tv.ptr, null)
            (tv.tv_sec * 1000L + tv.tv_usec / 1000L) - t0
        }

        // ── Test accuracy ───────────────────────────────────────
        val testAcc = evalAccuracy(model, testImages, testLabels, KG.TEST_N, scope)

        val avgLoss = epochLoss / stepsPerEpoch
        val aneMs = scope.aneTimeNs / 1_000_000.0
        val cpuMs = scope.cpuTimeNs / 1_000_000.0
        println("$epoch,$stepsPerEpoch,${fixed(avgLoss, 6)},${fixed(aneMs, 1)},${fixed(cpuMs, 1)},${fixed(testAcc, 2)}")
        println("# Epoch $epoch/${ KG.N_EPOCHS}  loss=${fixed(avgLoss, 4)}  test=${fixed(testAcc, 1)}%  (${elapsed}ms)")
        println()
    }

    // ── Final ───────────────────────────────────────────────────
    val finalAcc = evalAccuracy(model, testImages, testLabels, KG.TEST_N, scope)
    val totalAne = scope.aneTimeNs / 1_000_000.0
    val totalCpu = scope.cpuTimeNs / 1_000_000.0

    println()
    println("════════════════════════════════════════════════════")
    println("  Final test accuracy: ${fixed(finalAcc, 1)}%")
    println()
    println("  Matmuls (forward + backward):")
    println("    on ANE:          ${ANEMatmul.aneEvals} in ${fixed(totalAne, 0)} ms")
    println("    CPU fallback:    ${ANEMatmul.cpuFallbacks} in ${fixed(totalCpu, 0)} ms")
    println("    ANE kernels compiled: ${ANEMatmul.compiles} in ${fixed(ANEMatmul.compileNs / 1e6, 0)} ms (once per shape, weights are inputs)")
    println("  Everything else (bias, ReLU, loss, Adam, batching) runs on the CPU.")
    println()
    println("  Stack: KotlinGrad DSL → GradScope (private ANE) → cinterop")
    println("       → ANEBridge.m → ane_runtime.h → AppleNeuralEngine.framework")
    println("════════════════════════════════════════════════════")

    scope.close()
}

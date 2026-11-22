// MnistCpuFallback.kt — JVM actual implementation (pure-Kotlin CPU fallback)
// Implements the same 784→128→10 network without ANE, for non-Apple platforms.
package com.ane.mnist

import java.io.DataInputStream
import java.io.FileInputStream
import kotlin.math.*
import kotlin.random.Random

actual class MnistTrainer actual constructor(config: MnistConfig) {

    private val cfg = config
    private val rng = Random(System.nanoTime())

    // Weights [out, in] row-major
    private var w1 = FloatArray(cfg.hiddenDim * cfg.inputDim)
    private var w2 = FloatArray(cfg.outputDim * cfg.hiddenDim)
    private var vW1 = FloatArray(w1.size)   // momentum
    private var vW2 = FloatArray(w2.size)

    // Data
    private var trainImages = FloatArray(0)
    private var trainLabels = IntArray(0)
    private var testImages = FloatArray(0)
    private var testLabels = IntArray(0)
    private var nTrain = 0
    private var nTest = 0

    // State
    private var globalStep = 0
    private var stepInEpoch = 0
    private var currentEpochVal = 0
    private var shuffleIdx = IntArray(0)

    init {
        // He initialization
        val s1 = sqrt(2f / cfg.inputDim)
        val s2 = sqrt(2f / cfg.hiddenDim)
        for (i in w1.indices) w1[i] = randn() * s1
        for (i in w2.indices) w2[i] = randn() * s2
    }

    private fun randn(): Float {
        val u1 = rng.nextFloat().coerceIn(1e-7f, 1f - 1e-7f)
        val u2 = rng.nextFloat().coerceIn(1e-7f, 1f - 1e-7f)
        return sqrt(-2f * ln(u1)) * cos(2f * PI.toFloat() * u2)
    }

    actual fun loadData(dataDir: String): Boolean {
        return try {
            val (ti, tl) = loadIdx("$dataDir/train-images-idx3-ubyte",
                                   "$dataDir/train-labels-idx1-ubyte")
            trainImages = ti.first; trainLabels = ti.second; nTrain = tl
            val (xi, xl) = loadIdx("$dataDir/t10k-images-idx3-ubyte",
                                   "$dataDir/t10k-labels-idx1-ubyte")
            testImages = xi.first; testLabels = xi.second; nTest = xl
            shuffleIdx = IntArray(nTrain) { it }
            true
        } catch (e: Exception) {
            println("Data load error: ${e.message}")
            false
        }
    }

    private fun loadIdx(imgPath: String, lblPath: String): Pair<Pair<FloatArray, IntArray>, Int> {
        val imgs = DataInputStream(FileInputStream(imgPath))
        imgs.readInt() // magic
        val n = imgs.readInt()
        imgs.readInt(); imgs.readInt() // rows, cols
        val pixels = ByteArray(n * cfg.inputDim)
        imgs.readFully(pixels)
        imgs.close()
        val fImages = FloatArray(n * cfg.inputDim) { (pixels[it].toInt() and 0xFF) / 255f }

        val lbls = DataInputStream(FileInputStream(lblPath))
        lbls.readInt(); lbls.readInt()
        val rawLbls = ByteArray(n)
        lbls.readFully(rawLbls)
        lbls.close()
        val iLabels = IntArray(n) { rawLbls[it].toInt() and 0xFF }

        return Pair(Pair(fImages, iLabels), n)
    }

    actual fun compile(): Boolean = true  // No-op on CPU

    actual fun trainStep(): TrainStepResult {
        val bs = cfg.batchSize
        val inD = cfg.inputDim
        val hidD = cfg.hiddenDim
        val outD = cfg.outputDim
        val bpe = nTrain / bs

        if (stepInEpoch == 0) shuffleIdx.shuffle(rng)

        // Gather batch
        val batchImg = FloatArray(bs * inD)
        val batchLbl = IntArray(bs)
        for (b in 0 until bs) {
            val idx = shuffleIdx[stepInEpoch * bs + b]
            System.arraycopy(trainImages, idx * inD, batchImg, b * inD, inD)
            batchLbl[b] = trainLabels[idx]
        }

        // Forward: transpose to channel-first [inD, bs]
        val xCf = FloatArray(inD * bs)
        for (r in 0 until bs) for (c in 0 until inD) xCf[c * bs + r] = batchImg[r * inD + c]

        // Layer 1: w1[hidD, inD] @ xCf[inD, bs] → hPre[hidD, bs]
        val hPre = FloatArray(hidD * bs)
        for (h in 0 until hidD)
            for (b in 0 until bs) {
                var s = 0f
                for (i in 0 until inD) s += w1[h * inD + i] * xCf[i * bs + b]
                hPre[h * bs + b] = s
            }

        // ReLU
        val hPost = FloatArray(hidD * bs) { maxOf(0f, hPre[it]) }

        // Layer 2: w2[outD, hidD] @ hPost[hidD, bs] → logits[outD, bs]
        val logits = FloatArray(outD * bs)
        for (o in 0 until outD)
            for (b in 0 until bs) {
                var s = 0f
                for (h in 0 until hidD) s += w2[o * hidD + h] * hPost[h * bs + b]
                logits[o * bs + b] = s
            }

        // Loss + accuracy
        var loss = 0f; var correct = 0
        for (b in 0 until bs) {
            var mx = Float.NEGATIVE_INFINITY
            for (c in 0 until outD) mx = maxOf(mx, logits[c * bs + b])
            var se = 0f
            for (c in 0 until outD) se += exp(logits[c * bs + b] - mx)
            loss -= (logits[batchLbl[b] * bs + b] - mx - ln(se))
            var best = 0; var bv = logits[b]
            for (c in 1 until outD) {
                val v = logits[c * bs + b]; if (v > bv) { bv = v; best = c }
            }
            if (best == batchLbl[b]) correct++
        }
        loss /= bs

        // Backward
        val dL = FloatArray(outD * bs)
        for (b in 0 until bs) {
            var mx = Float.NEGATIVE_INFINITY
            for (c in 0 until outD) mx = maxOf(mx, logits[c * bs + b])
            var se = 0f
            for (c in 0 until outD) { dL[c * bs + b] = exp(logits[c * bs + b] - mx); se += dL[c * bs + b] }
            for (c in 0 until outD) dL[c * bs + b] /= se
            dL[batchLbl[b] * bs + b] -= 1f
            for (c in 0 until outD) dL[c * bs + b] /= bs
        }

        // dW2 = dL @ hPost^T
        val dW2 = FloatArray(outD * hidD)
        for (o in 0 until outD)
            for (h in 0 until hidD)
                for (b in 0 until bs)
                    dW2[o * hidD + h] += dL[o * bs + b] * hPost[h * bs + b]

        // d_h_post = W2^T @ dL then ReLU bwd → dW1
        val dW1 = FloatArray(hidD * inD)
        for (h in 0 until hidD) {
            for (b in 0 until bs) {
                var dhp = 0f
                for (o in 0 until outD) dhp += w2[o * hidD + h] * dL[o * bs + b]
                val dh = if (hPre[h * bs + b] > 0) dhp else 0f
                for (i in 0 until inD) dW1[h * inD + i] += dh * xCf[i * bs + b]
            }
        }

        // SGD + momentum
        val lr = cfg.learningRate; val mom = cfg.momentum
        for (i in w1.indices) { vW1[i] = mom * vW1[i] - lr * dW1[i]; w1[i] += vW1[i] }
        for (i in w2.indices) { vW2[i] = mom * vW2[i] - lr * dW2[i]; w2[i] += vW2[i] }

        // Advance
        globalStep++; stepInEpoch++
        var epochDone = false
        if (stepInEpoch >= bpe) { stepInEpoch = 0; currentEpochVal++; epochDone = true }

        return TrainStepResult(loss, correct.toFloat() / bs * 100f, globalStep, currentEpochVal, epochDone,
            compileCount = 0, budgetExhausted = false)
    }

    actual fun evaluate(): Float {
        val bs = cfg.batchSize; val inD = cfg.inputDim; val hidD = cfg.hiddenDim; val outD = cfg.outputDim
        var correct = 0; var total = 0
        var start = 0
        while (start + bs <= nTest) {
            val xCf = FloatArray(inD * bs)
            for (r in 0 until bs) for (c in 0 until inD)
                xCf[c * bs + r] = testImages[(start + r) * inD + c]

            val hp = FloatArray(hidD * bs)
            for (h in 0 until hidD) for (b in 0 until bs) {
                var s = 0f; for (i in 0 until inD) s += w1[h * inD + i] * xCf[i * bs + b]
                hp[h * bs + b] = maxOf(0f, s)
            }
            val lg = FloatArray(outD * bs)
            for (o in 0 until outD) for (b in 0 until bs) {
                var s = 0f; for (h in 0 until hidD) s += w2[o * hidD + h] * hp[h * bs + b]
                lg[o * bs + b] = s
            }
            for (b in 0 until bs) {
                var best = 0; var bv = lg[b]
                for (c in 1 until outD) { val v = lg[c * bs + b]; if (v > bv) { bv = v; best = c } }
                if (best == testLabels[start + b]) correct++
            }
            total += bs; start += bs
        }
        return if (total > 0) correct.toFloat() / total * 100f else 0f
    }

    actual val batchesPerEpoch: Int get() = nTrain / cfg.batchSize
    actual val trainCount: Int get() = nTrain
    actual val testCount: Int get() = nTest
    actual val currentStep: Int get() = globalStep

    actual fun close() { /* GC handles it */ }
}

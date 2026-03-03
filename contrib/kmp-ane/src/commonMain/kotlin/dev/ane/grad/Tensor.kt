// Tensor.kt — Simple row-major float tensor for the autograd engine.
package dev.ane.grad

/**
 * A simple dense tensor: row-major float data with shape metadata.
 * Not a full tensor library — just enough for MNIST MLP training.
 */
class Tensor(
    val data: FloatArray,
    val shape: IntArray
) {
    val size: Int get() = data.size
    val rows: Int get() = if (shape.size >= 2) shape[0] else 1
    val cols: Int get() = if (shape.size >= 2) shape[1] else shape[0]

    operator fun get(i: Int): Float = data[i]
    operator fun set(i: Int, v: Float) { data[i] = v }

    fun copy(): Tensor = Tensor(data.copyOf(), shape.copyOf())

    fun fill(v: Float) { data.fill(v) }
    fun zero() = fill(0f)

    companion object {
        fun zeros(vararg shape: Int) = Tensor(FloatArray(shape.reduce(Int::times)), shape)
        fun ones(vararg shape: Int) = Tensor(FloatArray(shape.reduce(Int::times)) { 1f }, shape)

        fun rand(vararg shape: Int, scale: Float = 1f): Tensor {
            val n = shape.reduce(Int::times)
            // Simple xorshift PRNG for reproducibility
            var seed = 42L
            val data = FloatArray(n) {
                seed = seed xor (seed shl 13)
                seed = seed xor (seed shr 7)
                seed = seed xor (seed shl 17)
                ((seed and 0x7FFFFFFF).toFloat() / Int.MAX_VALUE.toFloat()) * scale
            }
            return Tensor(data, shape)
        }

        /** Kaiming / He initialization for ReLU networks */
        fun kaiming(outCh: Int, inCh: Int): Tensor {
            val scale = kotlin.math.sqrt(2f / inCh)
            return rand(outCh, inCh, scale = scale)
        }
    }
}

/** Element-wise add (in place, a += b) */
fun Tensor.addInPlace(other: Tensor) {
    require(size == other.size) { "Size mismatch: $size vs ${other.size}" }
    for (i in data.indices) data[i] += other.data[i]
}

/** Element-wise scaled add (in place, a += scale * b) */
fun Tensor.addScaledInPlace(other: Tensor, scale: Float) {
    require(size == other.size) { "Size mismatch" }
    for (i in data.indices) data[i] += scale * other.data[i]
}

/** Scalar multiply in place */
fun Tensor.scaleInPlace(s: Float) {
    for (i in data.indices) data[i] *= s
}

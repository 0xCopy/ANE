package dev.ane.mnist

// Fixed-decimal formatting. String.format is not available in Kotlin/Native common code.
fun fixed(x: Double, decimals: Int): String {
    var scale = 1L
    repeat(decimals) { scale *= 10 }
    val neg = x < 0.0
    val scaled = ((if (neg) -x else x) * scale + 0.5).toLong()
    val frac = (scaled % scale).toString().padStart(decimals, '0')
    val whole = (scaled / scale).toString()
    return (if (neg) "-" else "") + whole + (if (decimals > 0) ".$frac" else "")
}

fun fixed(x: Float, decimals: Int): String = fixed(x.toDouble(), decimals)

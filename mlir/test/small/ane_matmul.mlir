// One ane.matmul at a size the device accepts, for aarch64/ane_asm.test: activations [32, 64] times a weight [64, 96].
func.func @mm(%a: tensor<1x1x32x64xf16>, %w: tensor<1x1x64x96xf16>) -> tensor<1x1x32x96xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
  %y = "ane.matmul"(%a, %w) {transpose_x = false, transpose_y = false} : (tensor<1x1x32x64xf16>, tensor<1x1x64x96xf16>) -> tensor<1x1x32x96xf16>
  return %y : tensor<1x1x32x96xf16>
}

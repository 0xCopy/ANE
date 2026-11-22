// W8A8 (ane_int8_bench.m) in the integer domain: what NEON offers once weights and activations are both int8.
//
// The MIL program dequantizes (int8 -> fp16), convolves in fp16 and quantizes again.  With one scale per tensor the
// same 1x1 conv is   y[o] = (sw * sx) * sum_i wq[o,i] * xq[i]   and the sum is exact in i32.
//
//   arm_neon.intr.sdot (FEAT_DotProd, ARMv8.4-A; every Apple Silicon Mac):
//       acc[l] += sum_{k<4} b[4l+k] * c[4l+k]          one instruction = four length-4 int8 dot products
//   arm_neon.intr.smmla (FEAT_I8MM): acc(2x2 of i32) += b(2x8 i8) * c(2x8 i8)^T
//
// @conv1x1_i8_4x4: IC = 4, OC = 4.  wq holds the four weight rows back to back; xq is the input column, repeated per row.
func.func @conv1x1_i8_4x4(%wq: vector<16xi8>, %x: vector<4xi8>) -> vector<4xi32> {
  %xq = vector.shuffle %x, %x [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3] : vector<4xi8>, vector<4xi8>
  %z = arith.constant dense<0> : vector<4xi32>
  %r = arm_neon.intr.sdot %z, %wq, %xq : vector<16xi8>, vector<16xi8> to vector<4xi32>
  return %r : vector<4xi32>
}

// One output of the same conv the way the MIL program states it: dequantize both sides, multiply and add in f16, in order.
func.func @dot_f16(%wq: vector<4xi8>, %xq: vector<4xi8>, %sw: f16, %sx: f16) -> f16 {
  %swv = vector.broadcast %sw : f16 to vector<4xf16>
  %sxv = vector.broadcast %sx : f16 to vector<4xf16>
  %wf0 = arith.sitofp %wq : vector<4xi8> to vector<4xf16>
  %xf0 = arith.sitofp %xq : vector<4xi8> to vector<4xf16>
  %wf = arith.mulf %wf0, %swv : vector<4xf16>
  %xf = arith.mulf %xf0, %sxv : vector<4xf16>
  %p = arith.mulf %wf, %xf : vector<4xf16>
  %z = arith.constant 0.0 : f16
  %r = vector.reduction <add>, %p, %z : vector<4xf16> into f16
  return %r : f16
}

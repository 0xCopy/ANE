// Single-op generators (training/ane_mil_gen.h) and the W8A8 chain (ane_int8_bench.m) against their normal forms.
//
// Sizes: in_ch=4 out_ch=6 spatial=5; the int8 chain is 3 layers of 4x4 on a 5x5 image.
// ane_mil_gen.h kernels take and return fp32 and cast at both ends: the normal form is to_f32(linear(to_f16(.))).
//
// RUN: cat %S/harness.mlir %S/../normal/ane_normal.mlir %S/small/primitives.mlir %s | mlir-opt \
// RUN:   --transform-preload-library=transform-library-paths=%S/decompose_softmax.mlir --transform-interpreter \
// RUN:   --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
// RUN:   --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%llvmlib/libmlir_runner_utils.dylib,%llvmlib/libmlir_c_runner_utils.dylib | FileCheck %s

// CHECK: matmul (fp32 in/out): maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: conv (fp32 in/out): maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: W8A8 chain (3 layers): maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}

func.func @main() {
  %s0 = arith.constant 0.4 : f32
  %s1 = arith.constant 2.2 : f32
  %amp = arith.constant 0.9 : f32
  %i4 = arith.constant 4 : index
  %i5 = arith.constant 5 : index
  %x32 = call @gen32(%i4, %i5, %s0, %amp) : (index, index, f32, f32) -> tensor<?x?xf32>
  %i6 = arith.constant 6 : index
  %w32 = call @gen32(%i6, %i4, %s1, %amp) : (index, index, f32, f32) -> tensor<?x?xf32>
  %x16 = call @to_f16(%x32) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %w16 = call @to_f16(%w32) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %n16 = call @linear(%w16, %x16) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %x3c = tensor.cast %x32 : tensor<?x?xf32> to tensor<4x5xf32>
  %x3 = tensor.expand_shape %x3c [[0, 1], [2]] output_shape [1, 4, 5] : tensor<4x5xf32> into tensor<1x4x5xf32>
  %w3c = tensor.cast %w32 : tensor<?x?xf32> to tensor<6x4xf32>
  %w3 = tensor.expand_shape %w3c [[0, 1], [2]] output_shape [1, 6, 4] : tensor<6x4xf32> into tensor<1x6x4xf32>
  %mm3 = call @ane_matmul(%x3, %w3) : (tensor<1x4x5xf32>, tensor<1x6x4xf32>) -> tensor<1x6x5xf32>
  %mmc = tensor.collapse_shape %mm3 [[0, 1], [2]] : tensor<1x6x5xf32> into tensor<6x5xf32>
  %mmd = tensor.cast %mmc : tensor<6x5xf32> to tensor<?x?xf32>
  %mm = call @to_f16(%mmd) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  vector.print str "matmul (fp32 in/out): "
  call @report(%mm, %n16) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %x4 = tensor.expand_shape %x3c [[0, 1], [2, 3]] output_shape [1, 4, 1, 5] : tensor<4x5xf32> into tensor<1x4x1x5xf32>
  %w4.c = tensor.cast %w16 : tensor<?x?xf16> to tensor<6x4xf16>
  %w4 = tensor.expand_shape %w4.c [[0], [1, 2, 3]] output_shape [6, 4, 1, 1] : tensor<6x4xf16> into tensor<6x4x1x1xf16>
  %cv4 = call @ane_conv(%x4, %w4) : (tensor<1x4x1x5xf32>, tensor<6x4x1x1xf16>) -> tensor<1x6x1x5xf32>
  %cvc = tensor.collapse_shape %cv4 [[0, 1], [2, 3]] : tensor<1x6x1x5xf32> into tensor<6x5xf32>
  %cvd = tensor.cast %cvc : tensor<6x5xf32> to tensor<?x?xf32>
  %cv = call @to_f16(%cvd) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  vector.print str "conv (fp32 in/out): "
  call @report(%cv, %n16) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %qs = arith.constant 0.125 : f16
  %a8 = arith.constant 100.0 : f32
  %ax = arith.constant 3.0 : f32
  %i25 = arith.constant 25 : index
  %qx32 = call @gen32(%i4, %i25, %s0, %ax) : (index, index, f32, f32) -> tensor<?x?xf32>
  %qx = call @to_f16(%qx32) : (tensor<?x?xf32>) -> tensor<?x?xf16>
  %qxc = tensor.cast %qx : tensor<?x?xf16> to tensor<4x25xf16>
  %qx4 = tensor.expand_shape %qxc [[0, 1], [2, 3]] output_shape [1, 4, 5, 5] : tensor<4x25xf16> into tensor<1x4x5x5xf16>
  %ws0 = arith.constant 0.7 : f32
  %wq0 = call @gen8(%i4, %i4, %ws0, %a8) : (index, index, f32, f32) -> tensor<?x?xi8>
  %wq0c = tensor.cast %wq0 : tensor<?x?xi8> to tensor<4x4xi8>
  %wq04 = tensor.expand_shape %wq0c [[0], [1, 2, 3]] output_shape [4, 4, 1, 1] : tensor<4x4xi8> into tensor<4x4x1x1xi8>
  %wd0 = call @dequantize(%wq0, %qs) : (tensor<?x?xi8>, f16) -> tensor<?x?xf16>
  %c0 = call @linear(%wd0, %qx) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %q0 = call @quantize(%c0, %qs) : (tensor<?x?xf16>, f16) -> tensor<?x?xi8>
  %dq0 = call @dequantize(%q0, %qs) : (tensor<?x?xi8>, f16) -> tensor<?x?xf16>
  %ws1 = arith.constant 1.7 : f32
  %wq1 = call @gen8(%i4, %i4, %ws1, %a8) : (index, index, f32, f32) -> tensor<?x?xi8>
  %wq1c = tensor.cast %wq1 : tensor<?x?xi8> to tensor<4x4xi8>
  %wq14 = tensor.expand_shape %wq1c [[0], [1, 2, 3]] output_shape [4, 4, 1, 1] : tensor<4x4xi8> into tensor<4x4x1x1xi8>
  %wd1 = call @dequantize(%wq1, %qs) : (tensor<?x?xi8>, f16) -> tensor<?x?xf16>
  %c1 = call @linear(%wd1, %dq0) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %q1 = call @quantize(%c1, %qs) : (tensor<?x?xf16>, f16) -> tensor<?x?xi8>
  %dq1 = call @dequantize(%q1, %qs) : (tensor<?x?xi8>, f16) -> tensor<?x?xf16>
  %ws2 = arith.constant 2.7 : f32
  %wq2 = call @gen8(%i4, %i4, %ws2, %a8) : (index, index, f32, f32) -> tensor<?x?xi8>
  %wq2c = tensor.cast %wq2 : tensor<?x?xi8> to tensor<4x4xi8>
  %wq24 = tensor.expand_shape %wq2c [[0], [1, 2, 3]] output_shape [4, 4, 1, 1] : tensor<4x4xi8> into tensor<4x4x1x1xi8>
  %wd2 = call @dequantize(%wq2, %qs) : (tensor<?x?xi8>, f16) -> tensor<?x?xf16>
  %c2 = call @linear(%wd2, %dq1) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %i8l4 = call @ane_genMILInt8(%qx4, %wq04, %wq14, %wq24) : (tensor<1x4x5x5xf16>, tensor<4x4x1x1xi8>, tensor<4x4x1x1xi8>, tensor<4x4x1x1xi8>) -> tensor<1x4x5x5xf16>
  %i8lc = tensor.collapse_shape %i8l4 [[0, 1], [2, 3]] : tensor<1x4x5x5xf16> into tensor<4x25xf16>
  %i8l = tensor.cast %i8lc : tensor<4x25xf16> to tensor<?x?xf16>
  vector.print str "W8A8 chain (3 layers): "
  call @report(%i8l, %c2) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  return
}

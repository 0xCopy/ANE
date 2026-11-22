// The ane dialect (dialect/ane.irdl.mlir) and its two printers.
//
// 1. The declaration is what tools/ane_dialect.py generates, and every kernel file in ane/ verifies against it.
// RUN: python3 %S/../tools/ane_dialect.py irdl | diff - %S/../dialect/ane.irdl.mlir
// RUN: mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %S/../ane/stories110m/static.mlir -o /dev/null
// RUN: mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %S/../ane/stories110m/dynamic.mlir -o /dev/null
// RUN: mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %S/../ane/primitives.mlir -o /dev/null
// RUN: mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %S/../ane/small/dynamic.mlir -o /dev/null
//
// 2. ane -> MIL -> ane is a fixed point (the Qwen3 kernels: 11 programs, 209 ops).
// RUN: mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %S/../ane/qwen3_06b/dynamic.mlir -o %t.a
// RUN: python3 %S/../tools/ane_dialect.py to-mil < %t.a | python3 %S/../tools/ane_dialect.py from-mil \
// RUN:   | mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir | diff - %t.a
//
// 3. The CPU reading of the dialect computes what the kernels compute: the small dynamic kernels go
//    ane -> MIL -> linalg and take the place of small/dynamic.mlir in check_dynamic, with its expectations.
// RUN: mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %S/../ane/small/dynamic.mlir \
// RUN:   | python3 %S/../tools/ane_dialect.py to-mil | python3 %S/../tools/mil2mlir.py > %t.lin
// RUN: cat %S/harness.mlir %S/../normal/ane_normal.mlir %t.lin %S/check_dynamic.mlir | mlir-opt \
// RUN:   --transform-preload-library=transform-library-paths=%S/decompose_softmax.mlir --transform-interpreter \
// RUN:   --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
// RUN:   --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%llvmlib/libmlir_runner_utils.dylib,%llvmlib/libmlir_c_runner_utils.dylib \
// RUN: | FileCheck %S/check_dynamic.mlir
//
// 4. What the declaration rejects (this file's own functions).
// RUN: not mlir-opt --irdl-file=%S/../dialect/ane.irdl.mlir %s 2>&1 | FileCheck %s

// CHECK: op expects exactly 2 operands, but got 1
func.func @one_operand(%x: tensor<1x2x3xf16>) -> tensor<1x2x3xf16> {
  %r = "ane.matmul"(%x) {transpose_x = false, transpose_y = false} : (tensor<1x2x3xf16>) -> tensor<1x2x3xf16>
  return %r : tensor<1x2x3xf16>
}

// CHECK: attribute "axis" is expected but not provided
func.func @no_axis(%x: tensor<1x2x3xf16>) -> tensor<1x2x3xf16> {
  %r = "ane.softmax"(%x) : (tensor<1x2x3xf16>) -> tensor<1x2x3xf16>
  return %r : tensor<1x2x3xf16>
}

// A softmax keeps its operand's type.
// CHECK: expected 'tensor<1x2x3xf16>' but got 'tensor<1x2x4xf16>'
func.func @softmax_shape(%x: tensor<1x2x3xf16>) -> tensor<1x2x4xf16> {
  %r = "ane.softmax"(%x) {axis = 1 : i32} : (tensor<1x2x3xf16>) -> tensor<1x2x4xf16>
  return %r : tensor<1x2x4xf16>
}

// CHECK: expected base attribute 'builtin.array'
func.func @perm_kind(%x: tensor<1x2x3xf16>) -> tensor<1x3x2xf16> {
  %r = "ane.transpose"(%x) {perm = 3 : i32} : (tensor<1x2x3xf16>) -> tensor<1x3x2xf16>
  return %r : tensor<1x3x2xf16>
}

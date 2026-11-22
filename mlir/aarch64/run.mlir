// Executed checks for neon_fp16.mlir and neon_i8.mlir (arm64 host with FEAT_FP16 and FEAT_DotProd).
//
// RUN: cat %S/neon_fp16.mlir %S/neon_i8.mlir %s | mlir-opt --convert-vector-to-scf --convert-scf-to-cf --convert-math-to-llvm \
// RUN:   --convert-vector-to-llvm=enable-arm-neon --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%llvmlib/libmlir_runner_utils.dylib,%llvmlib/libmlir_c_runner_utils.dylib | FileCheck %s

func.func private @printMemrefF32(%ptr : memref<*xf32>)
memref.global "private" constant @vals : memref<11xf32> =
    dense<[0.1, 65504.0, 65519.0, 65520.0, -65520.0, 2049.0, 2051.0, 1.0e-08, 0.33333334, 2049.0, 65520.0]>

func.func @main() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c11 = arith.constant 11 : index

  // f32 -> f16 -> f32 over 11 elements (one vector block of 8, a scalar tail of 3).
  //   65504 is the largest finite f16; 65519 rounds down to it, 65520 rounds up to +inf (the causal mask uses -65504,
  //   not -inf).  Above 2048 the spacing is 2: 2049 and 2051 are ties and go to the even neighbour.  1e-8 underflows.
  %src = memref.get_global @vals : memref<11xf32>
  %h = memref.alloc() : memref<11xf16>
  %back = memref.alloc() : memref<11xf32>
  %srcd = memref.cast %src : memref<11xf32> to memref<?xf32>
  %hd = memref.cast %h : memref<11xf16> to memref<?xf16>
  %backd = memref.cast %back : memref<11xf32> to memref<?xf32>
  call @cvt_f32_f16(%hd, %srcd, %c11) : (memref<?xf16>, memref<?xf32>, index) -> ()
  call @cvt_f16_f32(%backd, %hd, %c11) : (memref<?xf32>, memref<?xf16>, index) -> ()
  %bu = memref.cast %back : memref<11xf32> to memref<*xf32>
  call @printMemrefF32(%bu) : (memref<*xf32>) -> ()
  // CHECK: [0.0999756,  65504,  65504,  inf,  -inf,  2048,  2052,  0,  0.333252,  2048,  inf]

  // quantize, scale 0.125: ties away from zero (0.0625/0.125 = 0.5 -> 1, 0.1875/0.125 = 1.5 -> 2; ties to even would give 0
  // and 2: the device gives 1 and 2, see README "Sharp corners"), saturation at both ends.
  %scale = arith.constant 0.125 : f16
  %qx = arith.constant dense<[0.0625, 0.1875, -0.1875, 1.0, 15.875, 16.0, -16.0, -1000.0]> : vector<8xf16>
  %q = call @quantize8(%qx, %scale) : (vector<8xf16>, f16) -> vector<8xi8>
  vector.print %q : vector<8xi8>
  // CHECK: ( 1, 2, -2, 8, 127, 127, -128, -128 )

  // W8A8 1x1 conv, IC = OC = 4, both scales 0.125.  Row 0 is all 127 against x = 127: the exact sum is 4 * 16129.
  %wq = arith.constant dense<[127, 127, 127, 127,  1, 2, 3, 4,  -128, -128, -128, -128,  100, -100, 50, -50]> : vector<16xi8>
  %x = arith.constant dense<[127, 127, 127, 127]> : vector<4xi8>
  %acc = call @conv1x1_i8_4x4(%wq, %x) : (vector<16xi8>, vector<4xi8>) -> vector<4xi32>
  vector.print %acc : vector<4xi32>
  // CHECK: ( 64516, 1270, -65024, 0 )
  // Scaled by sw * sx = 1/64 in f32: exact.
  %accf = arith.sitofp %acc : vector<4xi32> to vector<4xf32>
  %k = arith.constant dense<0.015625> : vector<4xf32>
  %exact = arith.mulf %accf, %k : vector<4xf32>
  vector.print %exact : vector<4xf32>
  // CHECK: ( 1008.06, 19.8438, -1016, 0 )
  // The fp16 route for rows 0 and 2: each product 127 * 127 / 64 = 252.02 is rounded to 252 before the sum.
  %w0 = arith.constant dense<[127, 127, 127, 127]> : vector<4xi8>
  %w2 = arith.constant dense<[-128, -128, -128, -128]> : vector<4xi8>
  %d0 = call @dot_f16(%w0, %x, %scale, %scale) : (vector<4xi8>, vector<4xi8>, f16, f16) -> f16
  %d2 = call @dot_f16(%w2, %x, %scale, %scale) : (vector<4xi8>, vector<4xi8>, f16, f16) -> f16
  vector.print %d0 : f16
  vector.print %d2 : f16
  // CHECK: 1008{{$}}
  // CHECK: -1016{{$}}
  return
}

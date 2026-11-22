module {
  func.func @ane_matmul(%arg0: tensor<1x64x32xf32>, %arg1: tensor<1x96x64xf32>) -> tensor<1x96x32xf32> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.cast"(%arg0) {dtype = "fp16"} : (tensor<1x64x32xf32>) -> tensor<1x64x32xf16>
    %1 = "ane.cast"(%arg1) {dtype = "fp16"} : (tensor<1x96x64xf32>) -> tensor<1x96x64xf16>
    %2 = "ane.matmul"(%1, %0) {transpose_x = false, transpose_y = false} : (tensor<1x96x64xf16>, tensor<1x64x32xf16>) -> tensor<1x96x32xf16>
    %3 = "ane.cast"(%2) {dtype = "fp32"} : (tensor<1x96x32xf16>) -> tensor<1x96x32xf32>
    return %3 : tensor<1x96x32xf32>
  }
  func.func @ane_conv(%arg0: tensor<1x64x1x32xf32>) -> tensor<1x96x1x32xf32> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.cast"(%arg0) {dtype = "fp16"} : (tensor<1x64x1x32xf32>) -> tensor<1x64x1x32xf16>
    %1 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<96x64x1x1xf16>
    %2 = "ane.conv"(%0, %1) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x1x32xf16>, tensor<96x64x1x1xf16>) -> tensor<1x96x1x32xf16>
    %3 = "ane.cast"(%2) {dtype = "fp32"} : (tensor<1x96x1x32xf16>) -> tensor<1x96x1x32xf32>
    return %3 : tensor<1x96x1x32xf32>
  }
  func.func @ane_qkv(%arg0: tensor<1x64x1x32xf32>) -> (tensor<1x64x1x32xf32>, tensor<1x64x1x32xf32>, tensor<1x64x1x32xf32>) attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.cast"(%arg0) {dtype = "fp16"} : (tensor<1x64x1x32xf32>) -> tensor<1x64x1x32xf16>
    %1 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<64x64x1x1xf16>
    %2 = "ane.blob"() {offset = 8320 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<64x64x1x1xf16>
    %3 = "ane.blob"() {offset = 16576 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<64x64x1x1xf16>
    %4 = "ane.conv"(%0, %1) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x1x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x1x32xf16>
    %5 = "ane.conv"(%0, %2) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x1x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x1x32xf16>
    %6 = "ane.conv"(%0, %3) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x1x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x1x32xf16>
    %7 = "ane.cast"(%4) {dtype = "fp32"} : (tensor<1x64x1x32xf16>) -> tensor<1x64x1x32xf32>
    %8 = "ane.cast"(%5) {dtype = "fp32"} : (tensor<1x64x1x32xf16>) -> tensor<1x64x1x32xf32>
    %9 = "ane.cast"(%6) {dtype = "fp32"} : (tensor<1x64x1x32xf16>) -> tensor<1x64x1x32xf32>
    return %7, %8, %9 : tensor<1x64x1x32xf32>, tensor<1x64x1x32xf32>, tensor<1x64x1x32xf32>
  }
  func.func @ane_ffn_up(%arg0: tensor<1x64x1x32xf32>) -> (tensor<1x96x1x32xf32>, tensor<1x96x1x32xf32>) attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.cast"(%arg0) {dtype = "fp16"} : (tensor<1x64x1x32xf32>) -> tensor<1x64x1x32xf16>
    %1 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<96x64x1x1xf16>
    %2 = "ane.blob"() {offset = 12416 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<96x64x1x1xf16>
    %3 = "ane.conv"(%0, %1) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x1x32xf16>, tensor<96x64x1x1xf16>) -> tensor<1x96x1x32xf16>
    %4 = "ane.conv"(%0, %2) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x1x32xf16>, tensor<96x64x1x1xf16>) -> tensor<1x96x1x32xf16>
    %5 = "ane.cast"(%3) {dtype = "fp32"} : (tensor<1x96x1x32xf16>) -> tensor<1x96x1x32xf32>
    %6 = "ane.cast"(%4) {dtype = "fp32"} : (tensor<1x96x1x32xf16>) -> tensor<1x96x1x32xf32>
    return %5, %6 : tensor<1x96x1x32xf32>, tensor<1x96x1x32xf32>
  }
  func.func @ane_genMILInt8(%arg0: tensor<1x64x32x32xf16>) -> tensor<1x64x32x32xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.constexpr_affine_dequantize"() {axis = 0 : i32, offset = 64 : i64, path = "@model_path/weights/weight.bin", scale = 1.250000e-01 : f16, zero_point = 0 : i8} : () -> tensor<64x64x1x1xf16>
    %1 = "ane.conv"(%arg0, %0) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x32x32xf16>
    %2 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %3 = "ane.quantize"(%1, %2) {output_dtype = "int8"} : (tensor<1x64x32x32xf16>, tensor<f16>) -> tensor<1x64x32x32xi8>
    %4 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %5 = "ane.dequantize"(%3, %4) : (tensor<1x64x32x32xi8>, tensor<f16>) -> tensor<1x64x32x32xf16>
    %6 = "ane.constexpr_affine_dequantize"() {axis = 0 : i32, offset = 4224 : i64, path = "@model_path/weights/weight.bin", scale = 1.250000e-01 : f16, zero_point = 0 : i8} : () -> tensor<64x64x1x1xf16>
    %7 = "ane.conv"(%5, %6) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x32x32xf16>
    %8 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %9 = "ane.quantize"(%7, %8) {output_dtype = "int8"} : (tensor<1x64x32x32xf16>, tensor<f16>) -> tensor<1x64x32x32xi8>
    %10 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %11 = "ane.dequantize"(%9, %10) : (tensor<1x64x32x32xi8>, tensor<f16>) -> tensor<1x64x32x32xf16>
    %12 = "ane.constexpr_affine_dequantize"() {axis = 0 : i32, offset = 8384 : i64, path = "@model_path/weights/weight.bin", scale = 1.250000e-01 : f16, zero_point = 0 : i8} : () -> tensor<64x64x1x1xf16>
    %13 = "ane.conv"(%11, %12) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x32x32xf16>
    return %13 : tensor<1x64x32x32xf16>
  }
  func.func @ane_genMILFP16(%arg0: tensor<1x64x32x32xf16>) -> tensor<1x64x32x32xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<64x64x1x1xf16>
    %1 = "ane.conv"(%arg0, %0) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x32x32xf16>
    %2 = "ane.blob"() {offset = 8320 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<64x64x1x1xf16>
    %3 = "ane.conv"(%1, %2) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x32x32xf16>
    %4 = "ane.blob"() {offset = 16576 : i64, path = "@model_path/weights/weight.bin"} : () -> tensor<64x64x1x1xf16>
    %5 = "ane.conv"(%3, %4) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x64x32x32xf16>, tensor<64x64x1x1xf16>) -> tensor<1x64x32x32xf16>
    return %5 : tensor<1x64x32x32xf16>
  }
}


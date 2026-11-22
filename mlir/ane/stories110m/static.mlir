module {
  func.func @ane_sdpa_fwd_taps(%arg0: tensor<1x768x1x256xf16>) -> tensor<1x4608x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.mul"(%arg0, %arg0) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.reduce_sum"(%0) {axes = [1], keep_dims = true} : (tensor<1x768x1x256xf16>) -> tensor<1x1x1x256xf16>
    %2 = "ane.const"() {value = 1.301770e-03 : f16} : () -> tensor<f16>
    %3 = "ane.mul"(%1, %2) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %4 = "ane.const"() {value = 1.001360e-05 : f16} : () -> tensor<f16>
    %5 = "ane.add"(%3, %4) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %6 = "ane.const"() {value = -5.000000e-01 : f16} : () -> tensor<f16>
    %7 = "ane.pow"(%5, %6) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %8 = "ane.mul"(%arg0, %7) : (tensor<1x768x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x768x1x256xf16>
    %9 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/rms1.bin"} : () -> tensor<1x768x1x1xf16>
    %10 = "ane.mul"(%8, %9) : (tensor<1x768x1x256xf16>, tensor<1x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %11 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wq.bin"} : () -> tensor<768x768x1x1xf16>
    %12 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wk.bin"} : () -> tensor<768x768x1x1xf16>
    %13 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wv.bin"} : () -> tensor<768x768x1x1xf16>
    %14 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wo.bin"} : () -> tensor<768x768x1x1xf16>
    %15 = "ane.conv"(%10, %11) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %16 = "ane.conv"(%10, %12) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %17 = "ane.conv"(%10, %13) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %18 = "ane.reshape"(%15) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %19 = "ane.transpose"(%18) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %20 = "ane.reshape"(%16) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %21 = "ane.transpose"(%20) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %22 = "ane.reshape"(%17) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %23 = "ane.transpose"(%22) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %24 = "ane.matmul"(%19, %21) {transpose_x = false, transpose_y = true} : (tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x256xf16>
    %25 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %26 = "ane.mul"(%24, %25) : (tensor<1x12x256x256xf16>, tensor<f16>) -> tensor<1x12x256x256xf16>
    %27 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/mask.bin"} : () -> tensor<1x1x256x256xf16>
    %28 = "ane.add"(%26, %27) : (tensor<1x12x256x256xf16>, tensor<1x1x256x256xf16>) -> tensor<1x12x256x256xf16>
    %29 = "ane.softmax"(%28) {axis = -1 : i32} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
    %30 = "ane.matmul"(%29, %23) {transpose_x = false, transpose_y = false} : (tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
    %31 = "ane.transpose"(%30) {perm = [0, 1, 3, 2]} : (tensor<1x12x256x64xf16>) -> tensor<1x12x64x256xf16>
    %32 = "ane.reshape"(%31) {shape = [1, 768, 1, 256]} : (tensor<1x12x64x256xf16>) -> tensor<1x768x1x256xf16>
    %33 = "ane.conv"(%32, %14) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %cast = tensor.cast %33 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %15 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_1 = tensor.cast %16 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_2 = tensor.cast %17 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_3 = tensor.cast %32 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_4 = tensor.cast %10 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %34 = "ane.concat"(%cast, %cast_0, %cast_1, %cast_2, %cast_3, %cast_4) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x4608x1x256xf16>
    return %34 : tensor<1x4608x1x256xf16>
  }
  func.func @ane_ffn_fwd_taps(%arg0: tensor<1x768x1x256xf16>) -> tensor<1x7680x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.mul"(%arg0, %arg0) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.reduce_sum"(%0) {axes = [1], keep_dims = true} : (tensor<1x768x1x256xf16>) -> tensor<1x1x1x256xf16>
    %2 = "ane.const"() {value = 1.301770e-03 : f16} : () -> tensor<f16>
    %3 = "ane.mul"(%1, %2) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %4 = "ane.const"() {value = 1.001360e-05 : f16} : () -> tensor<f16>
    %5 = "ane.add"(%3, %4) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %6 = "ane.const"() {value = -5.000000e-01 : f16} : () -> tensor<f16>
    %7 = "ane.pow"(%5, %6) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %8 = "ane.mul"(%arg0, %7) : (tensor<1x768x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x768x1x256xf16>
    %9 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/rms2.bin"} : () -> tensor<1x768x1x1xf16>
    %10 = "ane.mul"(%8, %9) : (tensor<1x768x1x256xf16>, tensor<1x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %11 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/w1.bin"} : () -> tensor<2048x768x1x1xf16>
    %12 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/w3.bin"} : () -> tensor<2048x768x1x1xf16>
    %13 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/w2.bin"} : () -> tensor<768x2048x1x1xf16>
    %14 = "ane.conv"(%10, %11) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<2048x768x1x1xf16>) -> tensor<1x2048x1x256xf16>
    %15 = "ane.conv"(%10, %12) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<2048x768x1x1xf16>) -> tensor<1x2048x1x256xf16>
    %16 = "ane.sigmoid"(%14) : (tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %17 = "ane.mul"(%14, %16) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %18 = "ane.mul"(%17, %15) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %19 = "ane.conv"(%18, %13) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x2048x1x256xf16>, tensor<768x2048x1x1xf16>) -> tensor<1x768x1x256xf16>
    %cast = tensor.cast %19 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %14 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_1 = tensor.cast %15 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_2 = tensor.cast %18 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_3 = tensor.cast %10 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %20 = "ane.concat"(%cast, %cast_0, %cast_1, %cast_2, %cast_3) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x7680x1x256xf16>
    return %20 : tensor<1x7680x1x256xf16>
  }
  func.func @ane_ffn_bwd(%arg0: tensor<1x4864x1x256xf16>) -> tensor<1x4864x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x4864x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 768, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x4864x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 2816, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x4864x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %3 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/w2t.bin"} : () -> tensor<2048x768x1x1xf16>
    %4 = "ane.conv"(%0, %3) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<2048x768x1x1xf16>) -> tensor<1x2048x1x256xf16>
    %5 = "ane.sigmoid"(%1) : (tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %6 = "ane.const"() {value = 1.000000e+00 : f16} : () -> tensor<f16>
    %7 = "ane.sub"(%6, %5) : (tensor<f16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %8 = "ane.mul"(%1, %7) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %9 = "ane.const"() {value = 1.000000e+00 : f16} : () -> tensor<f16>
    %10 = "ane.add"(%9, %8) : (tensor<f16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %11 = "ane.mul"(%5, %10) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %12 = "ane.mul"(%4, %2) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %13 = "ane.mul"(%12, %11) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %14 = "ane.mul"(%1, %5) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %15 = "ane.mul"(%4, %14) : (tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %16 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/w1t.bin"} : () -> tensor<768x2048x1x1xf16>
    %17 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/w3t.bin"} : () -> tensor<768x2048x1x1xf16>
    %18 = "ane.conv"(%13, %16) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x2048x1x256xf16>, tensor<768x2048x1x1xf16>) -> tensor<1x768x1x256xf16>
    %19 = "ane.conv"(%15, %17) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x2048x1x256xf16>, tensor<768x2048x1x1xf16>) -> tensor<1x768x1x256xf16>
    %20 = "ane.add"(%18, %19) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %cast = tensor.cast %20 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %13 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_1 = tensor.cast %15 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %21 = "ane.concat"(%cast, %cast_0, %cast_1) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x4864x1x256xf16>
    return %21 : tensor<1x4864x1x256xf16>
  }
  func.func @ane_qkvb(%arg0: tensor<1x2304x1x256xf16>) -> tensor<1x768x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x2304x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 768, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x2304x1x256xf16>) -> tensor<1x768x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 1536, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x2304x1x256xf16>) -> tensor<1x768x1x256xf16>
    %3 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wqt.bin"} : () -> tensor<768x768x1x1xf16>
    %4 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wkt.bin"} : () -> tensor<768x768x1x1xf16>
    %5 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wvt.bin"} : () -> tensor<768x768x1x1xf16>
    %6 = "ane.conv"(%0, %3) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %7 = "ane.conv"(%1, %4) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %8 = "ane.conv"(%2, %5) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %9 = "ane.add"(%6, %7) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %10 = "ane.add"(%9, %8) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    return %10 : tensor<1x768x1x256xf16>
  }
  func.func @ane_sdpa_bwd1(%arg0: tensor<1x3072x1x256xf16>) -> tensor<1x6912x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 768, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x768x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 1536, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x768x1x256xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 2304, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x768x1x256xf16>
    %4 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/wot.bin"} : () -> tensor<768x768x1x1xf16>
    %5 = "ane.conv"(%3, %4) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<768x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %6 = "ane.reshape"(%0) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %7 = "ane.transpose"(%6) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %8 = "ane.reshape"(%1) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %9 = "ane.transpose"(%8) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %10 = "ane.reshape"(%2) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %11 = "ane.transpose"(%10) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %12 = "ane.reshape"(%5) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %13 = "ane.transpose"(%12) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %14 = "ane.matmul"(%7, %9) {transpose_x = false, transpose_y = true} : (tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x256xf16>
    %15 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %16 = "ane.mul"(%14, %15) : (tensor<1x12x256x256xf16>, tensor<f16>) -> tensor<1x12x256x256xf16>
    %17 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/mask.bin"} : () -> tensor<1x1x256x256xf16>
    %18 = "ane.add"(%16, %17) : (tensor<1x12x256x256xf16>, tensor<1x1x256x256xf16>) -> tensor<1x12x256x256xf16>
    %19 = "ane.softmax"(%18) {axis = -1 : i32} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
    %20 = "ane.matmul"(%19, %13) {transpose_x = true, transpose_y = false} : (tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
    %21 = "ane.matmul"(%13, %11) {transpose_x = false, transpose_y = true} : (tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x256xf16>
    %22 = "ane.transpose"(%20) {perm = [0, 1, 3, 2]} : (tensor<1x12x256x64xf16>) -> tensor<1x12x64x256xf16>
    %23 = "ane.reshape"(%22) {shape = [1, 768, 1, 256]} : (tensor<1x12x64x256xf16>) -> tensor<1x768x1x256xf16>
    %24 = "ane.reshape"(%19) {shape = [1, 3072, 1, 256]} : (tensor<1x12x256x256xf16>) -> tensor<1x3072x1x256xf16>
    %25 = "ane.reshape"(%21) {shape = [1, 3072, 1, 256]} : (tensor<1x12x256x256xf16>) -> tensor<1x3072x1x256xf16>
    %cast = tensor.cast %23 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %24 : tensor<1x3072x1x256xf16> to tensor<*xf16>
    %cast_1 = tensor.cast %25 : tensor<1x3072x1x256xf16> to tensor<*xf16>
    %26 = "ane.concat"(%cast, %cast_0, %cast_1) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x6912x1x256xf16>
    return %26 : tensor<1x6912x1x256xf16>
  }
  func.func @ane_sdpa_bwd2(%arg0: tensor<1x7680x1x256xf16>) -> tensor<1x1536x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 3072, 1, 256]} : (tensor<1x7680x1x256xf16>) -> tensor<1x3072x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 3072, 0, 0], size = [1, 3072, 1, 256]} : (tensor<1x7680x1x256xf16>) -> tensor<1x3072x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 6144, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x7680x1x256xf16>) -> tensor<1x768x1x256xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 6912, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x7680x1x256xf16>) -> tensor<1x768x1x256xf16>
    %4 = "ane.reshape"(%0) {shape = [1, 12, 256, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x12x256x256xf16>
    %5 = "ane.reshape"(%1) {shape = [1, 12, 256, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x12x256x256xf16>
    %6 = "ane.reshape"(%2) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %7 = "ane.transpose"(%6) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %8 = "ane.reshape"(%3) {shape = [1, 12, 64, 256]} : (tensor<1x768x1x256xf16>) -> tensor<1x12x64x256xf16>
    %9 = "ane.transpose"(%8) {perm = [0, 1, 3, 2]} : (tensor<1x12x64x256xf16>) -> tensor<1x12x256x64xf16>
    %10 = "ane.mul"(%4, %5) : (tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
    %11 = "ane.reduce_sum"(%10) {axes = [-1], keep_dims = true} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x1xf16>
    %12 = "ane.sub"(%5, %11) : (tensor<1x12x256x256xf16>, tensor<1x12x256x1xf16>) -> tensor<1x12x256x256xf16>
    %13 = "ane.mul"(%4, %12) : (tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
    %14 = "ane.const"() {value = 1.250000e-01 : f16} : () -> tensor<f16>
    %15 = "ane.mul"(%13, %14) : (tensor<1x12x256x256xf16>, tensor<f16>) -> tensor<1x12x256x256xf16>
    %16 = "ane.matmul"(%15, %9) {transpose_x = false, transpose_y = false} : (tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
    %17 = "ane.matmul"(%15, %7) {transpose_x = true, transpose_y = false} : (tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
    %18 = "ane.transpose"(%16) {perm = [0, 1, 3, 2]} : (tensor<1x12x256x64xf16>) -> tensor<1x12x64x256xf16>
    %19 = "ane.transpose"(%17) {perm = [0, 1, 3, 2]} : (tensor<1x12x256x64xf16>) -> tensor<1x12x64x256xf16>
    %20 = "ane.reshape"(%18) {shape = [1, 768, 1, 256]} : (tensor<1x12x64x256xf16>) -> tensor<1x768x1x256xf16>
    %21 = "ane.reshape"(%19) {shape = [1, 768, 1, 256]} : (tensor<1x12x64x256xf16>) -> tensor<1x768x1x256xf16>
    %cast = tensor.cast %20 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %21 : tensor<1x768x1x256xf16> to tensor<*xf16>
    %22 = "ane.concat"(%cast, %cast_0) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>) -> tensor<1x1536x1x256xf16>
    return %22 : tensor<1x1536x1x256xf16>
  }
  func.func @ane_rmsnorm_bwd(%arg0: tensor<1x1536x1x256xf16>) -> tensor<1x768x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x1536x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 768, 0, 0], size = [1, 768, 1, 256]} : (tensor<1x1536x1x256xf16>) -> tensor<1x768x1x256xf16>
    %2 = "ane.mul"(%1, %1) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %3 = "ane.reduce_sum"(%2) {axes = [1], keep_dims = true} : (tensor<1x768x1x256xf16>) -> tensor<1x1x1x256xf16>
    %4 = "ane.const"() {value = 1.301770e-03 : f16} : () -> tensor<f16>
    %5 = "ane.mul"(%3, %4) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %6 = "ane.const"() {value = 1.001360e-05 : f16} : () -> tensor<f16>
    %7 = "ane.add"(%5, %6) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %8 = "ane.const"() {value = -5.000000e-01 : f16} : () -> tensor<f16>
    %9 = "ane.pow"(%7, %8) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %10 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/rms_w.bin"} : () -> tensor<1x768x1x1xf16>
    %11 = "ane.mul"(%0, %10) : (tensor<1x768x1x256xf16>, tensor<1x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    %12 = "ane.mul"(%11, %1) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %13 = "ane.reduce_sum"(%12) {axes = [1], keep_dims = true} : (tensor<1x768x1x256xf16>) -> tensor<1x1x1x256xf16>
    %14 = "ane.const"() {value = 1.301770e-03 : f16} : () -> tensor<f16>
    %15 = "ane.mul"(%13, %14) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %16 = "ane.mul"(%9, %9) : (tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16>
    %17 = "ane.mul"(%15, %16) : (tensor<1x1x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x1x1x256xf16>
    %18 = "ane.mul"(%1, %17) : (tensor<1x768x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x768x1x256xf16>
    %19 = "ane.sub"(%11, %18) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %20 = "ane.mul"(%19, %9) : (tensor<1x768x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x768x1x256xf16>
    return %20 : tensor<1x768x1x256xf16>
  }
  func.func @ane_classifier_fwd(%arg0: tensor<1x768x1x256xf16>) -> tensor<1x32000x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/embed.bin"} : () -> tensor<32000x768x1x1xf16>
    %1 = "ane.conv"(%arg0, %0) {dilations = [1, 1], groups = 1 : i32, pad = [0, 0, 0, 0], pad_type = "valid", strides = [1, 1]} : (tensor<1x768x1x256xf16>, tensor<32000x768x1x1xf16>) -> tensor<1x32000x1x256xf16>
    return %1 : tensor<1x32000x1x256xf16>
  }
  func.func @ane_classifier_bwd(%arg0: tensor<1x32000x1x256xf16>) -> tensor<1x768x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.reshape"(%arg0) {shape = [1, 32000, 256]} : (tensor<1x32000x1x256xf16>) -> tensor<1x32000x256xf16>
    %1 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/embed_t.bin"} : () -> tensor<1x768x32000xf16>
    %2 = "ane.matmul"(%1, %0) {transpose_x = false, transpose_y = false} : (tensor<1x768x32000xf16>, tensor<1x32000x256xf16>) -> tensor<1x768x256xf16>
    %3 = "ane.reshape"(%2) {shape = [1, 768, 1, 256]} : (tensor<1x768x256xf16>) -> tensor<1x768x1x256xf16>
    return %3 : tensor<1x768x1x256xf16>
  }
  func.func @ane_softmax_vocab(%arg0: tensor<1x32000x1x256xf16>) -> tensor<1x32000x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.softmax"(%arg0) {axis = 1 : i32} : (tensor<1x32000x1x256xf16>) -> tensor<1x32000x1x256xf16>
    return %0 : tensor<1x32000x1x256xf16>
  }
  func.func @ane_final_rmsnorm(%arg0: tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.mul"(%arg0, %arg0) : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
    %1 = "ane.reduce_sum"(%0) {axes = [1], keep_dims = true} : (tensor<1x768x1x256xf16>) -> tensor<1x1x1x256xf16>
    %2 = "ane.const"() {value = 1.301770e-03 : f16} : () -> tensor<f16>
    %3 = "ane.mul"(%1, %2) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %4 = "ane.const"() {value = 1.001360e-05 : f16} : () -> tensor<f16>
    %5 = "ane.add"(%3, %4) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %6 = "ane.const"() {value = -5.000000e-01 : f16} : () -> tensor<f16>
    %7 = "ane.pow"(%5, %6) : (tensor<1x1x1x256xf16>, tensor<f16>) -> tensor<1x1x1x256xf16>
    %8 = "ane.mul"(%arg0, %7) : (tensor<1x768x1x256xf16>, tensor<1x1x1x256xf16>) -> tensor<1x768x1x256xf16>
    %9 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/rms_w.bin"} : () -> tensor<1x768x1x1xf16>
    %10 = "ane.mul"(%8, %9) : (tensor<1x768x1x256xf16>, tensor<1x768x1x1xf16>) -> tensor<1x768x1x256xf16>
    return %10 : tensor<1x768x1x256xf16>
  }
}


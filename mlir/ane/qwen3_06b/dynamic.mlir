module {
  func.func @ane_dyn_matmul_mil(%arg0: tensor<1x1024x1x3328xf16>) -> tensor<1x3072x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x3328xf16>) -> tensor<1x1024x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 1024, 1, 3072]} : (tensor<1x1024x1x3328xf16>) -> tensor<1x1024x1x3072xf16>
    %2 = "ane.reshape"(%0) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %3 = "ane.transpose"(%2) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %4 = "ane.reshape"(%1) {shape = [1, 1, 1024, 3072]} : (tensor<1x1024x1x3072xf16>) -> tensor<1x1x1024x3072xf16>
    %5 = "ane.matmul"(%3, %4) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x3072xf16>) -> tensor<1x1x256x3072xf16>
    %6 = "ane.transpose"(%5) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x3072xf16>) -> tensor<1x1x3072x256xf16>
    %7 = "ane.reshape"(%6) {shape = [1, 3072, 1, 256]} : (tensor<1x1x3072x256xf16>) -> tensor<1x3072x1x256xf16>
    return %7 : tensor<1x3072x1x256xf16>
  }
  func.func @ane_sdpa_fwd_dynamic(%arg0: tensor<1x1024x1x4352xf16>) -> tensor<1x7168x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x4352xf16>) -> tensor<1x1024x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 1024, 1, 2048]} : (tensor<1x1024x1x4352xf16>) -> tensor<1x1024x1x2048xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 2304], size = [1, 1024, 1, 1024]} : (tensor<1x1024x1x4352xf16>) -> tensor<1x1024x1x1024xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 3328], size = [1, 1024, 1, 1024]} : (tensor<1x1024x1x4352xf16>) -> tensor<1x1024x1x1024xf16>
    %4 = "ane.reshape"(%0) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %5 = "ane.transpose"(%4) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %6 = "ane.reshape"(%1) {shape = [1, 1, 1024, 2048]} : (tensor<1x1024x1x2048xf16>) -> tensor<1x1x1024x2048xf16>
    %7 = "ane.reshape"(%2) {shape = [1, 1, 1024, 1024]} : (tensor<1x1024x1x1024xf16>) -> tensor<1x1x1024x1024xf16>
    %8 = "ane.reshape"(%3) {shape = [1, 1, 1024, 1024]} : (tensor<1x1024x1x1024xf16>) -> tensor<1x1x1024x1024xf16>
    %9 = "ane.matmul"(%5, %6) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<1x1x256x2048xf16>
    %10 = "ane.matmul"(%5, %7) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x1024xf16>) -> tensor<1x1x256x1024xf16>
    %11 = "ane.matmul"(%5, %8) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x1024xf16>) -> tensor<1x1x256x1024xf16>
    %12 = "ane.transpose"(%9) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x2048xf16>) -> tensor<1x1x2048x256xf16>
    %13 = "ane.transpose"(%10) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %14 = "ane.transpose"(%11) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %15 = "ane.reshape"(%12) {shape = [1, 2048, 1, 256]} : (tensor<1x1x2048x256xf16>) -> tensor<1x2048x1x256xf16>
    %16 = "ane.reshape"(%13) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    %17 = "ane.reshape"(%14) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    %18 = "ane.reshape"(%15) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %19 = "ane.transpose"(%18) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %20 = "ane.reshape"(%16) {shape = [1, 8, 128, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x8x128x256xf16>
    %21 = "ane.transpose"(%20) {perm = [0, 1, 3, 2]} : (tensor<1x8x128x256xf16>) -> tensor<1x8x256x128xf16>
    %22 = "ane.reshape"(%17) {shape = [1, 8, 128, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x8x128x256xf16>
    %23 = "ane.transpose"(%22) {perm = [0, 1, 3, 2]} : (tensor<1x8x128x256xf16>) -> tensor<1x8x256x128xf16>
    %24 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/rope_cos.bin"} : () -> tensor<1x1x256x128xf16>
    %25 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/rope_sin.bin"} : () -> tensor<1x1x256x128xf16>
    %26 = "ane.reshape"(%19) {shape = [1, 16, 16384, 2]} : (tensor<1x16x256x128xf16>) -> tensor<1x16x16384x2xf16>
    %27 = "ane.slice_by_size"(%26) {begin = [0, 0, 0, 0], size = [1, 16, 16384, 1]} : (tensor<1x16x16384x2xf16>) -> tensor<1x16x16384x1xf16>
    %28 = "ane.slice_by_size"(%26) {begin = [0, 0, 0, 1], size = [1, 16, 16384, 1]} : (tensor<1x16x16384x2xf16>) -> tensor<1x16x16384x1xf16>
    %29 = "ane.const"() {value = -1.000000e+00 : f16} : () -> tensor<f16>
    %30 = "ane.mul"(%28, %29) : (tensor<1x16x16384x1xf16>, tensor<f16>) -> tensor<1x16x16384x1xf16>
    %cast = tensor.cast %30 : tensor<1x16x16384x1xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %27 : tensor<1x16x16384x1xf16> to tensor<*xf16>
    %31 = "ane.concat"(%cast, %cast_0) {axis = 3 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>) -> tensor<1x16x16384x2xf16>
    %32 = "ane.reshape"(%31) {shape = [1, 16, 256, 128]} : (tensor<1x16x16384x2xf16>) -> tensor<1x16x256x128xf16>
    %33 = "ane.mul"(%19, %24) : (tensor<1x16x256x128xf16>, tensor<1x1x256x128xf16>) -> tensor<1x16x256x128xf16>
    %34 = "ane.mul"(%32, %25) : (tensor<1x16x256x128xf16>, tensor<1x1x256x128xf16>) -> tensor<1x16x256x128xf16>
    %35 = "ane.add"(%33, %34) : (tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x128xf16>
    %36 = "ane.reshape"(%21) {shape = [1, 8, 16384, 2]} : (tensor<1x8x256x128xf16>) -> tensor<1x8x16384x2xf16>
    %37 = "ane.slice_by_size"(%36) {begin = [0, 0, 0, 0], size = [1, 8, 16384, 1]} : (tensor<1x8x16384x2xf16>) -> tensor<1x8x16384x1xf16>
    %38 = "ane.slice_by_size"(%36) {begin = [0, 0, 0, 1], size = [1, 8, 16384, 1]} : (tensor<1x8x16384x2xf16>) -> tensor<1x8x16384x1xf16>
    %39 = "ane.const"() {value = -1.000000e+00 : f16} : () -> tensor<f16>
    %40 = "ane.mul"(%38, %39) : (tensor<1x8x16384x1xf16>, tensor<f16>) -> tensor<1x8x16384x1xf16>
    %cast_1 = tensor.cast %40 : tensor<1x8x16384x1xf16> to tensor<*xf16>
    %cast_2 = tensor.cast %37 : tensor<1x8x16384x1xf16> to tensor<*xf16>
    %41 = "ane.concat"(%cast_1, %cast_2) {axis = 3 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>) -> tensor<1x8x16384x2xf16>
    %42 = "ane.reshape"(%41) {shape = [1, 8, 256, 128]} : (tensor<1x8x16384x2xf16>) -> tensor<1x8x256x128xf16>
    %43 = "ane.mul"(%21, %24) : (tensor<1x8x256x128xf16>, tensor<1x1x256x128xf16>) -> tensor<1x8x256x128xf16>
    %44 = "ane.mul"(%42, %25) : (tensor<1x8x256x128xf16>, tensor<1x1x256x128xf16>) -> tensor<1x8x256x128xf16>
    %45 = "ane.add"(%43, %44) : (tensor<1x8x256x128xf16>, tensor<1x8x256x128xf16>) -> tensor<1x8x256x128xf16>
    %cast_3 = tensor.cast %45 : tensor<1x8x256x128xf16> to tensor<*xf16>
    %cast_4 = tensor.cast %45 : tensor<1x8x256x128xf16> to tensor<*xf16>
    %46 = "ane.concat"(%cast_3, %cast_4) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>) -> tensor<1x16x256x128xf16>
    %cast_5 = tensor.cast %23 : tensor<1x8x256x128xf16> to tensor<*xf16>
    %cast_6 = tensor.cast %23 : tensor<1x8x256x128xf16> to tensor<*xf16>
    %47 = "ane.concat"(%cast_5, %cast_6) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>) -> tensor<1x16x256x128xf16>
    %48 = "ane.matmul"(%35, %46) {transpose_x = false, transpose_y = true} : (tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x256xf16>
    %49 = "ane.const"() {value = 8.837890e-02 : f16} : () -> tensor<f16>
    %50 = "ane.mul"(%48, %49) : (tensor<1x16x256x256xf16>, tensor<f16>) -> tensor<1x16x256x256xf16>
    %51 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/mask.bin"} : () -> tensor<1x1x256x256xf16>
    %52 = "ane.add"(%50, %51) : (tensor<1x16x256x256xf16>, tensor<1x1x256x256xf16>) -> tensor<1x16x256x256xf16>
    %53 = "ane.softmax"(%52) {axis = -1 : i32} : (tensor<1x16x256x256xf16>) -> tensor<1x16x256x256xf16>
    %54 = "ane.matmul"(%53, %47) {transpose_x = false, transpose_y = false} : (tensor<1x16x256x256xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x128xf16>
    %55 = "ane.transpose"(%54) {perm = [0, 1, 3, 2]} : (tensor<1x16x256x128xf16>) -> tensor<1x16x128x256xf16>
    %56 = "ane.reshape"(%55) {shape = [1, 2048, 1, 256]} : (tensor<1x16x128x256xf16>) -> tensor<1x2048x1x256xf16>
    %57 = "ane.transpose"(%35) {perm = [0, 1, 3, 2]} : (tensor<1x16x256x128xf16>) -> tensor<1x16x128x256xf16>
    %58 = "ane.reshape"(%57) {shape = [1, 2048, 1, 256]} : (tensor<1x16x128x256xf16>) -> tensor<1x2048x1x256xf16>
    %59 = "ane.transpose"(%45) {perm = [0, 1, 3, 2]} : (tensor<1x8x256x128xf16>) -> tensor<1x8x128x256xf16>
    %60 = "ane.reshape"(%59) {shape = [1, 1024, 1, 256]} : (tensor<1x8x128x256xf16>) -> tensor<1x1024x1x256xf16>
    %cast_7 = tensor.cast %56 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_8 = tensor.cast %58 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_9 = tensor.cast %60 : tensor<1x1024x1x256xf16> to tensor<*xf16>
    %cast_10 = tensor.cast %17 : tensor<1x1024x1x256xf16> to tensor<*xf16>
    %cast_11 = tensor.cast %0 : tensor<1x1024x1x256xf16> to tensor<*xf16>
    %61 = "ane.concat"(%cast_7, %cast_8, %cast_9, %cast_10, %cast_11) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x7168x1x256xf16>
    return %61 : tensor<1x7168x1x256xf16>
  }
  func.func @ane_wo_fwd_dynamic(%arg0: tensor<1x2048x1x1280xf16>) -> tensor<1x1024x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x2048x1x1280xf16>) -> tensor<1x2048x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 2048, 1, 1024]} : (tensor<1x2048x1x1280xf16>) -> tensor<1x2048x1x1024xf16>
    %2 = "ane.reshape"(%0) {shape = [1, 1, 2048, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x1x2048x256xf16>
    %3 = "ane.transpose"(%2) {perm = [0, 1, 3, 2]} : (tensor<1x1x2048x256xf16>) -> tensor<1x1x256x2048xf16>
    %4 = "ane.reshape"(%1) {shape = [1, 1, 2048, 1024]} : (tensor<1x2048x1x1024xf16>) -> tensor<1x1x2048x1024xf16>
    %5 = "ane.matmul"(%3, %4) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x2048xf16>, tensor<1x1x2048x1024xf16>) -> tensor<1x1x256x1024xf16>
    %6 = "ane.transpose"(%5) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %7 = "ane.reshape"(%6) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    return %7 : tensor<1x1024x1x256xf16>
  }
  func.func @ane_ffn_fused_dynamic(%arg0: tensor<1x1024x1x9728xf16>) -> tensor<1x10240x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x9728xf16>) -> tensor<1x1024x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x9728xf16>) -> tensor<1x1024x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 512], size = [1, 1024, 1, 3072]} : (tensor<1x1024x1x9728xf16>) -> tensor<1x1024x1x3072xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 3584], size = [1, 1024, 1, 3072]} : (tensor<1x1024x1x9728xf16>) -> tensor<1x1024x1x3072xf16>
    %4 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 6656], size = [1, 1024, 1, 3072]} : (tensor<1x1024x1x9728xf16>) -> tensor<1x1024x1x3072xf16>
    %5 = "ane.reshape"(%0) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %6 = "ane.transpose"(%5) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %7 = "ane.reshape"(%2) {shape = [1, 1, 1024, 3072]} : (tensor<1x1024x1x3072xf16>) -> tensor<1x1x1024x3072xf16>
    %8 = "ane.reshape"(%3) {shape = [1, 1, 1024, 3072]} : (tensor<1x1024x1x3072xf16>) -> tensor<1x1x1024x3072xf16>
    %9 = "ane.matmul"(%6, %7) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x3072xf16>) -> tensor<1x1x256x3072xf16>
    %10 = "ane.matmul"(%6, %8) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x3072xf16>) -> tensor<1x1x256x3072xf16>
    %11 = "ane.transpose"(%9) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x3072xf16>) -> tensor<1x1x3072x256xf16>
    %12 = "ane.transpose"(%10) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x3072xf16>) -> tensor<1x1x3072x256xf16>
    %13 = "ane.reshape"(%11) {shape = [1, 3072, 1, 256]} : (tensor<1x1x3072x256xf16>) -> tensor<1x3072x1x256xf16>
    %14 = "ane.reshape"(%12) {shape = [1, 3072, 1, 256]} : (tensor<1x1x3072x256xf16>) -> tensor<1x3072x1x256xf16>
    %15 = "ane.sigmoid"(%13) : (tensor<1x3072x1x256xf16>) -> tensor<1x3072x1x256xf16>
    %16 = "ane.mul"(%13, %15) : (tensor<1x3072x1x256xf16>, tensor<1x3072x1x256xf16>) -> tensor<1x3072x1x256xf16>
    %17 = "ane.mul"(%16, %14) : (tensor<1x3072x1x256xf16>, tensor<1x3072x1x256xf16>) -> tensor<1x3072x1x256xf16>
    %18 = "ane.reshape"(%17) {shape = [1, 1, 3072, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x1x3072x256xf16>
    %19 = "ane.transpose"(%18) {perm = [0, 1, 3, 2]} : (tensor<1x1x3072x256xf16>) -> tensor<1x1x256x3072xf16>
    %20 = "ane.reshape"(%4) {shape = [1, 1, 1024, 3072]} : (tensor<1x1024x1x3072xf16>) -> tensor<1x1x1024x3072xf16>
    %21 = "ane.transpose"(%20) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x3072xf16>) -> tensor<1x1x3072x1024xf16>
    %22 = "ane.matmul"(%19, %21) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x3072xf16>, tensor<1x1x3072x1024xf16>) -> tensor<1x1x256x1024xf16>
    %23 = "ane.transpose"(%22) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %24 = "ane.reshape"(%23) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    %25 = "ane.const"() {value = 1.336670e-01 : f16} : () -> tensor<f16>
    %26 = "ane.mul"(%24, %25) : (tensor<1x1024x1x256xf16>, tensor<f16>) -> tensor<1x1024x1x256xf16>
    %27 = "ane.add"(%1, %26) : (tensor<1x1024x1x256xf16>, tensor<1x1024x1x256xf16>) -> tensor<1x1024x1x256xf16>
    %cast = tensor.cast %27 : tensor<1x1024x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %13 : tensor<1x3072x1x256xf16> to tensor<*xf16>
    %cast_1 = tensor.cast %14 : tensor<1x3072x1x256xf16> to tensor<*xf16>
    %cast_2 = tensor.cast %17 : tensor<1x3072x1x256xf16> to tensor<*xf16>
    %28 = "ane.concat"(%cast, %cast_0, %cast_1, %cast_2) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x10240x1x256xf16>
    return %28 : tensor<1x10240x1x256xf16>
  }
  func.func @ane_ffn_bwd_w2t_dynamic(%arg0: tensor<1x1024x1x3328xf16>) -> tensor<1x3072x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x3328xf16>) -> tensor<1x1024x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 1024, 1, 3072]} : (tensor<1x1024x1x3328xf16>) -> tensor<1x1024x1x3072xf16>
    %2 = "ane.reshape"(%0) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %3 = "ane.transpose"(%2) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %4 = "ane.reshape"(%1) {shape = [1, 1, 1024, 3072]} : (tensor<1x1024x1x3072xf16>) -> tensor<1x1x1024x3072xf16>
    %5 = "ane.matmul"(%3, %4) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x3072xf16>) -> tensor<1x1x256x3072xf16>
    %6 = "ane.transpose"(%5) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x3072xf16>) -> tensor<1x1x3072x256xf16>
    %7 = "ane.reshape"(%6) {shape = [1, 3072, 1, 256]} : (tensor<1x1x3072x256xf16>) -> tensor<1x3072x1x256xf16>
    return %7 : tensor<1x3072x1x256xf16>
  }
  func.func @ane_ffn_bwd_w13t_dynamic(%arg0: tensor<1x3072x1x2560xf16>) -> tensor<1x1024x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 3072, 1, 256]} : (tensor<1x3072x1x2560xf16>) -> tensor<1x3072x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 3072, 1, 256]} : (tensor<1x3072x1x2560xf16>) -> tensor<1x3072x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 512], size = [1, 3072, 1, 1024]} : (tensor<1x3072x1x2560xf16>) -> tensor<1x3072x1x1024xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 1536], size = [1, 3072, 1, 1024]} : (tensor<1x3072x1x2560xf16>) -> tensor<1x3072x1x1024xf16>
    %4 = "ane.reshape"(%0) {shape = [1, 1, 3072, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x1x3072x256xf16>
    %5 = "ane.transpose"(%4) {perm = [0, 1, 3, 2]} : (tensor<1x1x3072x256xf16>) -> tensor<1x1x256x3072xf16>
    %6 = "ane.reshape"(%1) {shape = [1, 1, 3072, 256]} : (tensor<1x3072x1x256xf16>) -> tensor<1x1x3072x256xf16>
    %7 = "ane.transpose"(%6) {perm = [0, 1, 3, 2]} : (tensor<1x1x3072x256xf16>) -> tensor<1x1x256x3072xf16>
    %8 = "ane.reshape"(%2) {shape = [1, 1, 3072, 1024]} : (tensor<1x3072x1x1024xf16>) -> tensor<1x1x3072x1024xf16>
    %9 = "ane.reshape"(%3) {shape = [1, 1, 3072, 1024]} : (tensor<1x3072x1x1024xf16>) -> tensor<1x1x3072x1024xf16>
    %10 = "ane.matmul"(%5, %8) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x3072xf16>, tensor<1x1x3072x1024xf16>) -> tensor<1x1x256x1024xf16>
    %11 = "ane.matmul"(%7, %9) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x3072xf16>, tensor<1x1x3072x1024xf16>) -> tensor<1x1x256x1024xf16>
    %12 = "ane.add"(%10, %11) : (tensor<1x1x256x1024xf16>, tensor<1x1x256x1024xf16>) -> tensor<1x1x256x1024xf16>
    %13 = "ane.transpose"(%12) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %14 = "ane.reshape"(%13) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    return %14 : tensor<1x1024x1x256xf16>
  }
  func.func @ane_wot_dynamic(%arg0: tensor<1x1024x1x2304xf16>) -> tensor<1x2048x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x2304xf16>) -> tensor<1x1024x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 1024, 1, 2048]} : (tensor<1x1024x1x2304xf16>) -> tensor<1x1024x1x2048xf16>
    %2 = "ane.reshape"(%0) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %3 = "ane.transpose"(%2) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %4 = "ane.reshape"(%1) {shape = [1, 1, 1024, 2048]} : (tensor<1x1024x1x2048xf16>) -> tensor<1x1x1024x2048xf16>
    %5 = "ane.matmul"(%3, %4) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x2048xf16>) -> tensor<1x1x256x2048xf16>
    %6 = "ane.transpose"(%5) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x2048xf16>) -> tensor<1x1x2048x256xf16>
    %7 = "ane.reshape"(%6) {shape = [1, 2048, 1, 256]} : (tensor<1x1x2048x256xf16>) -> tensor<1x2048x1x256xf16>
    return %7 : tensor<1x2048x1x256xf16>
  }
  func.func @ane_q_bwd_dynamic(%arg0: tensor<1x2048x1x1280xf16>) -> tensor<1x1024x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x2048x1x1280xf16>) -> tensor<1x2048x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 2048, 1, 1024]} : (tensor<1x2048x1x1280xf16>) -> tensor<1x2048x1x1024xf16>
    %2 = "ane.reshape"(%0) {shape = [1, 1, 2048, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x1x2048x256xf16>
    %3 = "ane.transpose"(%2) {perm = [0, 1, 3, 2]} : (tensor<1x1x2048x256xf16>) -> tensor<1x1x256x2048xf16>
    %4 = "ane.reshape"(%1) {shape = [1, 1, 2048, 1024]} : (tensor<1x2048x1x1024xf16>) -> tensor<1x1x2048x1024xf16>
    %5 = "ane.matmul"(%3, %4) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x2048xf16>, tensor<1x1x2048x1024xf16>) -> tensor<1x1x256x1024xf16>
    %6 = "ane.transpose"(%5) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %7 = "ane.reshape"(%6) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    return %7 : tensor<1x1024x1x256xf16>
  }
  func.func @ane_kv_bwd_dynamic(%arg0: tensor<1x1024x1x2560xf16>) -> tensor<1x1024x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x2560xf16>) -> tensor<1x1024x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 256], size = [1, 1024, 1, 256]} : (tensor<1x1024x1x2560xf16>) -> tensor<1x1024x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 512], size = [1, 1024, 1, 1024]} : (tensor<1x1024x1x2560xf16>) -> tensor<1x1024x1x1024xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 1536], size = [1, 1024, 1, 1024]} : (tensor<1x1024x1x2560xf16>) -> tensor<1x1024x1x1024xf16>
    %4 = "ane.reshape"(%0) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %5 = "ane.transpose"(%4) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %6 = "ane.reshape"(%1) {shape = [1, 1, 1024, 256]} : (tensor<1x1024x1x256xf16>) -> tensor<1x1x1024x256xf16>
    %7 = "ane.transpose"(%6) {perm = [0, 1, 3, 2]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1x256x1024xf16>
    %8 = "ane.reshape"(%2) {shape = [1, 1, 1024, 1024]} : (tensor<1x1024x1x1024xf16>) -> tensor<1x1x1024x1024xf16>
    %9 = "ane.reshape"(%3) {shape = [1, 1, 1024, 1024]} : (tensor<1x1024x1x1024xf16>) -> tensor<1x1x1024x1024xf16>
    %10 = "ane.matmul"(%5, %8) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x1024xf16>) -> tensor<1x1x256x1024xf16>
    %11 = "ane.matmul"(%7, %9) {transpose_x = false, transpose_y = false} : (tensor<1x1x256x1024xf16>, tensor<1x1x1024x1024xf16>) -> tensor<1x1x256x1024xf16>
    %12 = "ane.add"(%10, %11) : (tensor<1x1x256x1024xf16>, tensor<1x1x256x1024xf16>) -> tensor<1x1x256x1024xf16>
    %13 = "ane.transpose"(%12) {perm = [0, 1, 3, 2]} : (tensor<1x1x256x1024xf16>) -> tensor<1x1x1024x256xf16>
    %14 = "ane.reshape"(%13) {shape = [1, 1024, 1, 256]} : (tensor<1x1x1024x256xf16>) -> tensor<1x1024x1x256xf16>
    return %14 : tensor<1x1024x1x256xf16>
  }
  func.func @ane_sdpa_bwd1_noweight(%arg0: tensor<1x8192x1x256xf16>) -> tensor<1x10240x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x8192x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 2048, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x8192x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 4096, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x8192x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 6144, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x8192x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %4 = "ane.reshape"(%0) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %5 = "ane.transpose"(%4) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %6 = "ane.reshape"(%1) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %7 = "ane.transpose"(%6) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %8 = "ane.reshape"(%2) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %9 = "ane.transpose"(%8) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %10 = "ane.reshape"(%3) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %11 = "ane.transpose"(%10) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %12 = "ane.matmul"(%5, %7) {transpose_x = false, transpose_y = true} : (tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x256xf16>
    %13 = "ane.const"() {value = 8.837890e-02 : f16} : () -> tensor<f16>
    %14 = "ane.mul"(%12, %13) : (tensor<1x16x256x256xf16>, tensor<f16>) -> tensor<1x16x256x256xf16>
    %15 = "ane.blob"() {offset = 64 : i64, path = "@model_path/weights/mask.bin"} : () -> tensor<1x1x256x256xf16>
    %16 = "ane.add"(%14, %15) : (tensor<1x16x256x256xf16>, tensor<1x1x256x256xf16>) -> tensor<1x16x256x256xf16>
    %17 = "ane.softmax"(%16) {axis = -1 : i32} : (tensor<1x16x256x256xf16>) -> tensor<1x16x256x256xf16>
    %18 = "ane.matmul"(%17, %11) {transpose_x = true, transpose_y = false} : (tensor<1x16x256x256xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x128xf16>
    %19 = "ane.matmul"(%11, %9) {transpose_x = false, transpose_y = true} : (tensor<1x16x256x128xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x256xf16>
    %20 = "ane.transpose"(%18) {perm = [0, 1, 3, 2]} : (tensor<1x16x256x128xf16>) -> tensor<1x16x128x256xf16>
    %21 = "ane.reshape"(%20) {shape = [1, 2048, 1, 256]} : (tensor<1x16x128x256xf16>) -> tensor<1x2048x1x256xf16>
    %22 = "ane.reshape"(%17) {shape = [1, 4096, 1, 256]} : (tensor<1x16x256x256xf16>) -> tensor<1x4096x1x256xf16>
    %23 = "ane.reshape"(%19) {shape = [1, 4096, 1, 256]} : (tensor<1x16x256x256xf16>) -> tensor<1x4096x1x256xf16>
    %cast = tensor.cast %21 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %22 : tensor<1x4096x1x256xf16> to tensor<*xf16>
    %cast_1 = tensor.cast %23 : tensor<1x4096x1x256xf16> to tensor<*xf16>
    %24 = "ane.concat"(%cast, %cast_0, %cast_1) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) -> tensor<1x10240x1x256xf16>
    return %24 : tensor<1x10240x1x256xf16>
  }
  func.func @ane_sdpa_bwd2(%arg0: tensor<1x12288x1x256xf16>) -> tensor<1x4096x1x256xf16> attributes {ane.program = "1.3", ane.target = "ios18"} {
    %0 = "ane.slice_by_size"(%arg0) {begin = [0, 0, 0, 0], size = [1, 4096, 1, 256]} : (tensor<1x12288x1x256xf16>) -> tensor<1x4096x1x256xf16>
    %1 = "ane.slice_by_size"(%arg0) {begin = [0, 4096, 0, 0], size = [1, 4096, 1, 256]} : (tensor<1x12288x1x256xf16>) -> tensor<1x4096x1x256xf16>
    %2 = "ane.slice_by_size"(%arg0) {begin = [0, 8192, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x12288x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %3 = "ane.slice_by_size"(%arg0) {begin = [0, 10240, 0, 0], size = [1, 2048, 1, 256]} : (tensor<1x12288x1x256xf16>) -> tensor<1x2048x1x256xf16>
    %4 = "ane.reshape"(%0) {shape = [1, 16, 256, 256]} : (tensor<1x4096x1x256xf16>) -> tensor<1x16x256x256xf16>
    %5 = "ane.reshape"(%1) {shape = [1, 16, 256, 256]} : (tensor<1x4096x1x256xf16>) -> tensor<1x16x256x256xf16>
    %6 = "ane.reshape"(%2) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %7 = "ane.transpose"(%6) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %8 = "ane.reshape"(%3) {shape = [1, 16, 128, 256]} : (tensor<1x2048x1x256xf16>) -> tensor<1x16x128x256xf16>
    %9 = "ane.transpose"(%8) {perm = [0, 1, 3, 2]} : (tensor<1x16x128x256xf16>) -> tensor<1x16x256x128xf16>
    %10 = "ane.mul"(%4, %5) : (tensor<1x16x256x256xf16>, tensor<1x16x256x256xf16>) -> tensor<1x16x256x256xf16>
    %11 = "ane.reduce_sum"(%10) {axes = [-1], keep_dims = true} : (tensor<1x16x256x256xf16>) -> tensor<1x16x256x1xf16>
    %12 = "ane.sub"(%5, %11) : (tensor<1x16x256x256xf16>, tensor<1x16x256x1xf16>) -> tensor<1x16x256x256xf16>
    %13 = "ane.mul"(%4, %12) : (tensor<1x16x256x256xf16>, tensor<1x16x256x256xf16>) -> tensor<1x16x256x256xf16>
    %14 = "ane.const"() {value = 8.837890e-02 : f16} : () -> tensor<f16>
    %15 = "ane.mul"(%13, %14) : (tensor<1x16x256x256xf16>, tensor<f16>) -> tensor<1x16x256x256xf16>
    %16 = "ane.matmul"(%15, %9) {transpose_x = false, transpose_y = false} : (tensor<1x16x256x256xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x128xf16>
    %17 = "ane.matmul"(%15, %7) {transpose_x = true, transpose_y = false} : (tensor<1x16x256x256xf16>, tensor<1x16x256x128xf16>) -> tensor<1x16x256x128xf16>
    %18 = "ane.transpose"(%16) {perm = [0, 1, 3, 2]} : (tensor<1x16x256x128xf16>) -> tensor<1x16x128x256xf16>
    %19 = "ane.transpose"(%17) {perm = [0, 1, 3, 2]} : (tensor<1x16x256x128xf16>) -> tensor<1x16x128x256xf16>
    %20 = "ane.reshape"(%18) {shape = [1, 2048, 1, 256]} : (tensor<1x16x128x256xf16>) -> tensor<1x2048x1x256xf16>
    %21 = "ane.reshape"(%19) {shape = [1, 2048, 1, 256]} : (tensor<1x16x128x256xf16>) -> tensor<1x2048x1x256xf16>
    %cast = tensor.cast %20 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %21 : tensor<1x2048x1x256xf16> to tensor<*xf16>
    %22 = "ane.concat"(%cast, %cast_0) {axis = 1 : i32, interleave = false} : (tensor<*xf16>, tensor<*xf16>) -> tensor<1x4096x1x256xf16>
    return %22 : tensor<1x4096x1x256xf16>
  }
}


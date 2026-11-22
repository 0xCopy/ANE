// MIL: ane_matmul
func.func @ane_matmul(%x: tensor<1x4x5xf32>, %W: tensor<1x6x4xf32>)
    -> (tensor<1x6x5xf32>) {
  // x16 = cast(dtype = to_fp16, x = x)
  %x16.e = tensor.empty() : tensor<1x4x5xf16>
  %x16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x4x5xf32>) outs(%x16.e : tensor<1x4x5xf16>) {
    ^bb0(%_a0 : f32, %_o : f16):
      %_r = arith.truncf %_a0 : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x4x5xf16>
  // W16 = cast(dtype = to_fp16, x = W)
  %W16.e = tensor.empty() : tensor<1x6x4xf16>
  %W16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%W : tensor<1x6x4xf32>) outs(%W16.e : tensor<1x6x4xf16>) {
    ^bb0(%_a0 : f32, %_o : f16):
      %_r = arith.truncf %_a0 : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x6x4xf16>
  // y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)
  %y16.e = tensor.empty() : tensor<1x6x5xf16>
  %y16.z = arith.constant 0.0 : f16
  %y16.f = linalg.fill ins(%y16.z : f16) outs(%y16.e : tensor<1x6x5xf16>) -> tensor<1x6x5xf16>
  %y16 = linalg.contract indexing_maps = [affine_map<(b0, i, j, k) -> (b0, i, k)>, affine_map<(b0, i, j, k) -> (b0, k, j)>, affine_map<(b0, i, j, k) -> (b0, i, j)>]
      ins(%W16, %x16 : tensor<1x6x4xf16>, tensor<1x4x5xf16>) outs(%y16.f : tensor<1x6x5xf16>) -> tensor<1x6x5xf16>
  // y = cast(dtype = to_fp32, x = y16)
  %y.e = tensor.empty() : tensor<1x6x5xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%y16 : tensor<1x6x5xf16>) outs(%y.e : tensor<1x6x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x6x5xf32>
  return %y : tensor<1x6x5xf32>
}

// MIL: ane_conv
//   %W: BLOBFILE @model_path/weights/weight.bin @ offset 64
func.func @ane_conv(%x: tensor<1x4x1x5xf32>, %W: tensor<6x4x1x1xf16>)
    -> (tensor<1x6x1x5xf32>) {
  // x16 = cast(dtype = to_fp16, x = x)
  %x16.e = tensor.empty() : tensor<1x4x1x5xf16>
  %x16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x4x1x5xf32>) outs(%x16.e : tensor<1x4x1x5xf16>) {
    ^bb0(%_a0 : f32, %_o : f16):
      %_r = arith.truncf %_a0 : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x4x1x5xf16>
  // y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)
  %y16.e = tensor.empty() : tensor<1x6x1x5xf16>
  %y16.z = arith.constant 0.0 : f16
  %y16.f = linalg.fill ins(%y16.z : f16) outs(%y16.e : tensor<1x6x1x5xf16>) -> tensor<1x6x1x5xf16>
  %y16 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x16, %W : tensor<1x4x1x5xf16>, tensor<6x4x1x1xf16>) outs(%y16.f : tensor<1x6x1x5xf16>) -> tensor<1x6x1x5xf16>
  // y = cast(dtype = to_fp32, x = y16)
  %y.e = tensor.empty() : tensor<1x6x1x5xf32>
  %y = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%y16 : tensor<1x6x1x5xf16>) outs(%y.e : tensor<1x6x1x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x6x1x5xf32>
  return %y : tensor<1x6x1x5xf32>
}

// MIL: ane_qkv
//   %Wq: BLOBFILE @model_path/weights/weight.bin @ offset 64
//   %Wk: BLOBFILE @model_path/weights/weight.bin @ offset 160
//   %Wv: BLOBFILE @model_path/weights/weight.bin @ offset 256
func.func @ane_qkv(%x: tensor<1x4x1x5xf32>, %Wq: tensor<4x4x1x1xf16>, %Wk: tensor<4x4x1x1xf16>, %Wv: tensor<4x4x1x1xf16>)
    -> (tensor<1x4x1x5xf32>, tensor<1x4x1x5xf32>, tensor<1x4x1x5xf32>) {
  // x16 = cast(dtype = to_fp16, x = x)
  %x16.e = tensor.empty() : tensor<1x4x1x5xf16>
  %x16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x4x1x5xf32>) outs(%x16.e : tensor<1x4x1x5xf16>) {
    ^bb0(%_a0 : f32, %_o : f16):
      %_r = arith.truncf %_a0 : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x4x1x5xf16>
  // q16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x16)
  %q16.e = tensor.empty() : tensor<1x4x1x5xf16>
  %q16.z = arith.constant 0.0 : f16
  %q16.f = linalg.fill ins(%q16.z : f16) outs(%q16.e : tensor<1x4x1x5xf16>) -> tensor<1x4x1x5xf16>
  %q16 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x16, %Wq : tensor<1x4x1x5xf16>, tensor<4x4x1x1xf16>) outs(%q16.f : tensor<1x4x1x5xf16>) -> tensor<1x4x1x5xf16>
  // k16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)
  %k16.e = tensor.empty() : tensor<1x4x1x5xf16>
  %k16.z = arith.constant 0.0 : f16
  %k16.f = linalg.fill ins(%k16.z : f16) outs(%k16.e : tensor<1x4x1x5xf16>) -> tensor<1x4x1x5xf16>
  %k16 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x16, %Wk : tensor<1x4x1x5xf16>, tensor<4x4x1x1xf16>) outs(%k16.f : tensor<1x4x1x5xf16>) -> tensor<1x4x1x5xf16>
  // v16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)
  %v16.e = tensor.empty() : tensor<1x4x1x5xf16>
  %v16.z = arith.constant 0.0 : f16
  %v16.f = linalg.fill ins(%v16.z : f16) outs(%v16.e : tensor<1x4x1x5xf16>) -> tensor<1x4x1x5xf16>
  %v16 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x16, %Wv : tensor<1x4x1x5xf16>, tensor<4x4x1x1xf16>) outs(%v16.f : tensor<1x4x1x5xf16>) -> tensor<1x4x1x5xf16>
  // q = cast(dtype = to_fp32, x = q16)
  %q.e = tensor.empty() : tensor<1x4x1x5xf32>
  %q = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q16 : tensor<1x4x1x5xf16>) outs(%q.e : tensor<1x4x1x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x4x1x5xf32>
  // k = cast(dtype = to_fp32, x = k16)
  %k.e = tensor.empty() : tensor<1x4x1x5xf32>
  %k = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%k16 : tensor<1x4x1x5xf16>) outs(%k.e : tensor<1x4x1x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x4x1x5xf32>
  // v = cast(dtype = to_fp32, x = v16)
  %v.e = tensor.empty() : tensor<1x4x1x5xf32>
  %v = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%v16 : tensor<1x4x1x5xf16>) outs(%v.e : tensor<1x4x1x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x4x1x5xf32>
  return %q, %k, %v : tensor<1x4x1x5xf32>, tensor<1x4x1x5xf32>, tensor<1x4x1x5xf32>
}

// MIL: ane_ffn_up
//   %W1: BLOBFILE @model_path/weights/weight.bin @ offset 64
//   %W3: BLOBFILE @model_path/weights/weight.bin @ offset 176
func.func @ane_ffn_up(%x: tensor<1x4x1x5xf32>, %W1: tensor<6x4x1x1xf16>, %W3: tensor<6x4x1x1xf16>)
    -> (tensor<1x6x1x5xf32>, tensor<1x6x1x5xf32>) {
  // x16 = cast(dtype = to_fp16, x = x)
  %x16.e = tensor.empty() : tensor<1x4x1x5xf16>
  %x16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x4x1x5xf32>) outs(%x16.e : tensor<1x4x1x5xf16>) {
    ^bb0(%_a0 : f32, %_o : f16):
      %_r = arith.truncf %_a0 : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x4x1x5xf16>
  // h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)
  %h1.e = tensor.empty() : tensor<1x6x1x5xf16>
  %h1.z = arith.constant 0.0 : f16
  %h1.f = linalg.fill ins(%h1.z : f16) outs(%h1.e : tensor<1x6x1x5xf16>) -> tensor<1x6x1x5xf16>
  %h1 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x16, %W1 : tensor<1x4x1x5xf16>, tensor<6x4x1x1xf16>) outs(%h1.f : tensor<1x6x1x5xf16>) -> tensor<1x6x1x5xf16>
  // h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)
  %h3.e = tensor.empty() : tensor<1x6x1x5xf16>
  %h3.z = arith.constant 0.0 : f16
  %h3.f = linalg.fill ins(%h3.z : f16) outs(%h3.e : tensor<1x6x1x5xf16>) -> tensor<1x6x1x5xf16>
  %h3 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x16, %W3 : tensor<1x4x1x5xf16>, tensor<6x4x1x1xf16>) outs(%h3.f : tensor<1x6x1x5xf16>) -> tensor<1x6x1x5xf16>
  // out1 = cast(dtype = to_fp32, x = h1)
  %out1.e = tensor.empty() : tensor<1x6x1x5xf32>
  %out1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%h1 : tensor<1x6x1x5xf16>) outs(%out1.e : tensor<1x6x1x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x6x1x5xf32>
  // out3 = cast(dtype = to_fp32, x = h3)
  %out3.e = tensor.empty() : tensor<1x6x1x5xf32>
  %out3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%h3 : tensor<1x6x1x5xf16>) outs(%out3.e : tensor<1x6x1x5xf32>) {
    ^bb0(%_a0 : f16, %_o : f32):
      %_r = arith.extf %_a0 : f16 to f32
      linalg.yield %_r : f32
    } -> tensor<1x6x1x5xf32>
  return %out1, %out3 : tensor<1x6x1x5xf32>, tensor<1x6x1x5xf32>
}

// MIL: ane_genMILInt8
//   %W0_q: BLOBFILE @model_path/weights/weight.bin @ offset 64
//   %W1_q: BLOBFILE @model_path/weights/weight.bin @ offset 144
//   %W2_q: BLOBFILE @model_path/weights/weight.bin @ offset 224
func.func @ane_genMILInt8(%x: tensor<1x4x5x5xf16>, %W0_q: tensor<4x4x1x1xi8>, %W1_q: tensor<4x4x1x1xi8>, %W2_q: tensor<4x4x1x1xi8>)
    -> (tensor<1x4x5x5xf16>) {
  // W0 = constexpr_affine_dequantize()
  %W0.s = arith.constant 0.125 : f16
  %W0.zp = arith.constant 0 : i8
  %W0.e = tensor.empty() : tensor<4x4x1x1xf16>
  %W0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%W0_q, %W0.zp, %W0.s : tensor<4x4x1x1xi8>, i8, f16) outs(%W0.e : tensor<4x4x1x1xf16>) {
    ^bb0(%_a0 : i8, %_a1 : i8, %_a2 : f16, %_o : f16):
      %_d = arith.subi %_a0, %_a1 : i8
      %_c = arith.sitofp %_d : i8 to f16
      %_r = arith.mulf %_c, %_a2 : f16
      linalg.yield %_r : f16
    } -> tensor<4x4x1x1xf16>
  // c0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W0, x = x)
  %c0.e = tensor.empty() : tensor<1x4x5x5xf16>
  %c0.z = arith.constant 0.0 : f16
  %c0.f = linalg.fill ins(%c0.z : f16) outs(%c0.e : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  %c0 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x, %W0 : tensor<1x4x5x5xf16>, tensor<4x4x1x1xf16>) outs(%c0.f : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  // q0 = quantize(input = c0, output_dtype = q_dtype, scale = q_scale)
  %q_scale = arith.constant 0.125 : f16
  %q0.e = tensor.empty() : tensor<1x4x5x5xi8>
  %q0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%c0, %q_scale : tensor<1x4x5x5xf16>, f16) outs(%q0.e : tensor<1x4x5x5xi8>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : i8):
      %_one = arith.constant 1.0 : f16
      %_inv = arith.divf %_one, %_a1 : f16
      %_q = arith.mulf %_a0, %_inv : f16
      %_rn = math.round %_q : f16
      %_lo = arith.constant -128.0 : f16
      %_hi = arith.constant 127.0 : f16
      %_c0 = arith.maximumf %_rn, %_lo : f16
      %_c1 = arith.minimumf %_c0, %_hi : f16
      %_r = arith.fptosi %_c1 : f16 to i8
      linalg.yield %_r : i8
    } -> tensor<1x4x5x5xi8>
  // dq0 = dequantize(input = q0, scale = dq_scale)
  %dq_scale = arith.constant 0.125 : f16
  %dq0.e = tensor.empty() : tensor<1x4x5x5xf16>
  %dq0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q0, %dq_scale : tensor<1x4x5x5xi8>, f16) outs(%dq0.e : tensor<1x4x5x5xf16>) {
    ^bb0(%_a0 : i8, %_a1 : f16, %_o : f16):
      %_c = arith.sitofp %_a0 : i8 to f32
      %_s = arith.extf %_a1 : f16 to f32
      %_p = arith.mulf %_c, %_s : f32
      %_b = arith.bitcast %_p : f32 to i32
      %_kmag = arith.constant 2147483647 : i32
      %_ksgn = arith.constant -2147483648 : i32
      %_khalf = arith.constant 4096 : i32
      %_kclr = arith.constant -8192 : i32
      %_knorm = arith.constant 947912704 : i32
      %_m = arith.andi %_b, %_kmag : i32
      %_sg = arith.andi %_b, %_ksgn : i32
      %_inc = arith.addi %_m, %_khalf : i32
      %_tr = arith.andi %_inc, %_kclr : i32
      %_ob = arith.ori %_tr, %_sg : i32
      %_ra = arith.bitcast %_ob : i32 to f32
      %_up = arith.constant 16777216.0 : f32
      %_dn = arith.constant 5.9604644775390625e-08 : f32
      %_pk = arith.mulf %_p, %_up : f32
      %_rk = math.round %_pk : f32
      %_rs = arith.mulf %_rk, %_dn : f32
      %_small = arith.cmpi ult, %_m, %_knorm : i32
      %_sel = arith.select %_small, %_rs, %_ra : f32
      %_r = arith.truncf %_sel : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x4x5x5xf16>
  // W1 = constexpr_affine_dequantize()
  %W1.s = arith.constant 0.125 : f16
  %W1.zp = arith.constant 0 : i8
  %W1.e = tensor.empty() : tensor<4x4x1x1xf16>
  %W1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%W1_q, %W1.zp, %W1.s : tensor<4x4x1x1xi8>, i8, f16) outs(%W1.e : tensor<4x4x1x1xf16>) {
    ^bb0(%_a0 : i8, %_a1 : i8, %_a2 : f16, %_o : f16):
      %_d = arith.subi %_a0, %_a1 : i8
      %_c = arith.sitofp %_d : i8 to f16
      %_r = arith.mulf %_c, %_a2 : f16
      linalg.yield %_r : f16
    } -> tensor<4x4x1x1xf16>
  // c1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = dq0)
  %c1.e = tensor.empty() : tensor<1x4x5x5xf16>
  %c1.z = arith.constant 0.0 : f16
  %c1.f = linalg.fill ins(%c1.z : f16) outs(%c1.e : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  %c1 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dq0, %W1 : tensor<1x4x5x5xf16>, tensor<4x4x1x1xf16>) outs(%c1.f : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  // q1 = quantize(input = c1, output_dtype = q_dtype, scale = q_scale)
  %q1.e = tensor.empty() : tensor<1x4x5x5xi8>
  %q1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%c1, %q_scale : tensor<1x4x5x5xf16>, f16) outs(%q1.e : tensor<1x4x5x5xi8>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : i8):
      %_one = arith.constant 1.0 : f16
      %_inv = arith.divf %_one, %_a1 : f16
      %_q = arith.mulf %_a0, %_inv : f16
      %_rn = math.round %_q : f16
      %_lo = arith.constant -128.0 : f16
      %_hi = arith.constant 127.0 : f16
      %_c0 = arith.maximumf %_rn, %_lo : f16
      %_c1 = arith.minimumf %_c0, %_hi : f16
      %_r = arith.fptosi %_c1 : f16 to i8
      linalg.yield %_r : i8
    } -> tensor<1x4x5x5xi8>
  // dq1 = dequantize(input = q1, scale = dq_scale)
  %dq1.e = tensor.empty() : tensor<1x4x5x5xf16>
  %dq1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q1, %dq_scale : tensor<1x4x5x5xi8>, f16) outs(%dq1.e : tensor<1x4x5x5xf16>) {
    ^bb0(%_a0 : i8, %_a1 : f16, %_o : f16):
      %_c = arith.sitofp %_a0 : i8 to f32
      %_s = arith.extf %_a1 : f16 to f32
      %_p = arith.mulf %_c, %_s : f32
      %_b = arith.bitcast %_p : f32 to i32
      %_kmag = arith.constant 2147483647 : i32
      %_ksgn = arith.constant -2147483648 : i32
      %_khalf = arith.constant 4096 : i32
      %_kclr = arith.constant -8192 : i32
      %_knorm = arith.constant 947912704 : i32
      %_m = arith.andi %_b, %_kmag : i32
      %_sg = arith.andi %_b, %_ksgn : i32
      %_inc = arith.addi %_m, %_khalf : i32
      %_tr = arith.andi %_inc, %_kclr : i32
      %_ob = arith.ori %_tr, %_sg : i32
      %_ra = arith.bitcast %_ob : i32 to f32
      %_up = arith.constant 16777216.0 : f32
      %_dn = arith.constant 5.9604644775390625e-08 : f32
      %_pk = arith.mulf %_p, %_up : f32
      %_rk = math.round %_pk : f32
      %_rs = arith.mulf %_rk, %_dn : f32
      %_small = arith.cmpi ult, %_m, %_knorm : i32
      %_sel = arith.select %_small, %_rs, %_ra : f32
      %_r = arith.truncf %_sel : f32 to f16
      linalg.yield %_r : f16
    } -> tensor<1x4x5x5xf16>
  // W2 = constexpr_affine_dequantize()
  %W2.s = arith.constant 0.125 : f16
  %W2.zp = arith.constant 0 : i8
  %W2.e = tensor.empty() : tensor<4x4x1x1xf16>
  %W2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%W2_q, %W2.zp, %W2.s : tensor<4x4x1x1xi8>, i8, f16) outs(%W2.e : tensor<4x4x1x1xf16>) {
    ^bb0(%_a0 : i8, %_a1 : i8, %_a2 : f16, %_o : f16):
      %_d = arith.subi %_a0, %_a1 : i8
      %_c = arith.sitofp %_d : i8 to f16
      %_r = arith.mulf %_c, %_a2 : f16
      linalg.yield %_r : f16
    } -> tensor<4x4x1x1xf16>
  // c2 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2, x = dq1)
  %c2.e = tensor.empty() : tensor<1x4x5x5xf16>
  %c2.z = arith.constant 0.0 : f16
  %c2.f = linalg.fill ins(%c2.z : f16) outs(%c2.e : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  %c2 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dq1, %W2 : tensor<1x4x5x5xf16>, tensor<4x4x1x1xf16>) outs(%c2.f : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  return %c2 : tensor<1x4x5x5xf16>
}

// MIL: ane_genMILFP16
//   %W0: BLOBFILE @model_path/weights/weight.bin @ offset 64
//   %W1: BLOBFILE @model_path/weights/weight.bin @ offset 160
//   %W2: BLOBFILE @model_path/weights/weight.bin @ offset 256
func.func @ane_genMILFP16(%x: tensor<1x4x5x5xf16>, %W0: tensor<4x4x1x1xf16>, %W1: tensor<4x4x1x1xf16>, %W2: tensor<4x4x1x1xf16>)
    -> (tensor<1x4x5x5xf16>) {
  // c0 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W0, x = x)
  %c0.e = tensor.empty() : tensor<1x4x5x5xf16>
  %c0.z = arith.constant 0.0 : f16
  %c0.f = linalg.fill ins(%c0.z : f16) outs(%c0.e : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  %c0 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x, %W0 : tensor<1x4x5x5xf16>, tensor<4x4x1x1xf16>) outs(%c0.f : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  // c1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = c0)
  %c1.e = tensor.empty() : tensor<1x4x5x5xf16>
  %c1.z = arith.constant 0.0 : f16
  %c1.f = linalg.fill ins(%c1.z : f16) outs(%c1.e : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  %c1 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%c0, %W1 : tensor<1x4x5x5xf16>, tensor<4x4x1x1xf16>) outs(%c1.f : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  // c2 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W2, x = c1)
  %c2.e = tensor.empty() : tensor<1x4x5x5xf16>
  %c2.z = arith.constant 0.0 : f16
  %c2.f = linalg.fill ins(%c2.z : f16) outs(%c2.e : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  %c2 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%c1, %W2 : tensor<1x4x5x5xf16>, tensor<4x4x1x1xf16>) outs(%c2.f : tensor<1x4x5x5xf16>) -> tensor<1x4x5x5xf16>
  return %c2 : tensor<1x4x5x5xf16>
}

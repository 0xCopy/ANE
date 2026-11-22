// MIL: ane_sdpa_fwd_taps
//   %rw: BLOBFILE @model_path/weights/rms1.bin @ offset 64
//   %Wq: BLOBFILE @model_path/weights/wq.bin @ offset 64
//   %Wk: BLOBFILE @model_path/weights/wk.bin @ offset 64
//   %Wv: BLOBFILE @model_path/weights/wv.bin @ offset 64
//   %Wo: BLOBFILE @model_path/weights/wo.bin @ offset 64
//   %cm: BLOBFILE @model_path/weights/mask.bin @ offset 64
func.func @ane_sdpa_fwd_taps(%x: tensor<1x8x1x6xf16>, %rw: tensor<1x8x1x1xf16>, %Wq: tensor<8x8x1x1xf16>, %Wk: tensor<8x8x1x1xf16>, %Wv: tensor<8x8x1x1xf16>, %Wo: tensor<8x8x1x1xf16>, %cm: tensor<1x1x6x6xf16>)
    -> (tensor<1x48x1x6xf16>) {
  // sq = mul(x=x,y=x)
  %sq.e = tensor.empty() : tensor<1x8x1x6xf16>
  %sq = linalg.mul ins(%x, %x : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%sq.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)
  %ss.e = tensor.empty() : tensor<1x1x6xf16>
  %ss.z = arith.constant 0.0 : f16
  %ss.f = linalg.fill ins(%ss.z : f16) outs(%ss.e : tensor<1x1x6xf16>) -> tensor<1x1x6xf16>
  %ss.r = linalg.reduce { arith.addf } ins(%sq : tensor<1x8x1x6xf16>) outs(%ss.f : tensor<1x1x6xf16>) dimensions = [1]
  %ss = tensor.expand_shape %ss.r [[0], [1, 2], [3]] output_shape [1, 1, 1, 6] : tensor<1x1x6xf16> into tensor<1x1x1x6xf16>
  // ss2 = mul(x=ss,y=invd)
  %invd = arith.constant 0.125 : f16
  %ss2.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss, %invd : tensor<1x1x1x6xf16>, f16) outs(%ss2.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // ss3 = add(x=ss2,y=eps)
  %eps = arith.constant 1.0e-05 : f16
  %ss3.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss2, %eps : tensor<1x1x1x6xf16>, f16) outs(%ss3.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // rrms = pow(x=ss3,y=nhalf)
  %nhalf = arith.constant -0.5 : f16
  %rrms.e = tensor.empty() : tensor<1x1x1x6xf16>
  %rrms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss3, %nhalf : tensor<1x1x1x6xf16>, f16) outs(%rrms.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = math.powf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // xr = mul(x=x,y=rrms)
  %xr.e = tensor.empty() : tensor<1x8x1x6xf16>
  %xr = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x, %rrms : tensor<1x8x1x6xf16>, tensor<1x1x1x6xf16>) outs(%xr.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // xn = mul(x=xr,y=rw)
  %xn.e = tensor.empty() : tensor<1x8x1x6xf16>
  %xn = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%xr, %rw : tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>) outs(%xn.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)
  %qf.e = tensor.empty() : tensor<1x8x1x6xf16>
  %qf.z = arith.constant 0.0 : f16
  %qf.f = linalg.fill ins(%qf.z : f16) outs(%qf.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %qf = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%xn, %Wq : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%qf.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)
  %kf.e = tensor.empty() : tensor<1x8x1x6xf16>
  %kf.z = arith.constant 0.0 : f16
  %kf.f = linalg.fill ins(%kf.z : f16) outs(%kf.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %kf = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%xn, %Wk : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%kf.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)
  %vf.e = tensor.empty() : tensor<1x8x1x6xf16>
  %vf.z = arith.constant 0.0 : f16
  %vf.f = linalg.fill ins(%vf.z : f16) outs(%vf.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %vf = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%xn, %Wv : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%vf.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // q4 = reshape(shape=qsh,x=qf)
  %q4.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %q4 = tensor.reshape %qf(%q4.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // q = transpose(perm=pm,x=q4)
  %q.e = tensor.empty() : tensor<1x2x6x4xf16>
  %q = linalg.transpose ins(%q4 : tensor<1x2x4x6xf16>) outs(%q.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // k4 = reshape(shape=qsh,x=kf)
  %k4.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %k4 = tensor.reshape %kf(%k4.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // k = transpose(perm=pm,x=k4)
  %k.e = tensor.empty() : tensor<1x2x6x4xf16>
  %k = linalg.transpose ins(%k4 : tensor<1x2x4x6xf16>) outs(%k.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // v4 = reshape(shape=qsh,x=vf)
  %v4.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %v4 = tensor.reshape %vf(%v4.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // v = transpose(perm=pm,x=v4)
  %v.e = tensor.empty() : tensor<1x2x6x4xf16>
  %v = linalg.transpose ins(%v4 : tensor<1x2x4x6xf16>) outs(%v.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)
  %sc1.e = tensor.empty() : tensor<1x2x6x6xf16>
  %sc1.z = arith.constant 0.0 : f16
  %sc1.f = linalg.fill ins(%sc1.z : f16) outs(%sc1.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  %sc1 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, j, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%q, %k : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16>) outs(%sc1.f : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // sc2 = mul(x=sc1,y=scv)
  %scv = arith.constant 0.5 : f16
  %sc2.e = tensor.empty() : tensor<1x2x6x6xf16>
  %sc2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc1, %scv : tensor<1x2x6x6xf16>, f16) outs(%sc2.e : tensor<1x2x6x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x2x6x6xf16>
  // ms = add(x=sc2,y=cm)
  %ms.e = tensor.empty() : tensor<1x2x6x6xf16>
  %ms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc2, %cm : tensor<1x2x6x6xf16>, tensor<1x1x6x6xf16>) outs(%ms.e : tensor<1x2x6x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x2x6x6xf16>
  // aw = softmax(axis=sax,x=ms)
  %aw.e = tensor.empty() : tensor<1x2x6x6xf16>
  %aw = linalg.softmax dimension(3) ins(%ms : tensor<1x2x6x6xf16>) outs(%aw.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)
  %a4.e = tensor.empty() : tensor<1x2x6x4xf16>
  %a4.z = arith.constant 0.0 : f16
  %a4.f = linalg.fill ins(%a4.z : f16) outs(%a4.e : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  %a4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%aw, %v : tensor<1x2x6x6xf16>, tensor<1x2x6x4xf16>) outs(%a4.f : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  // at = transpose(perm=pm,x=a4)
  %at.e = tensor.empty() : tensor<1x2x4x6xf16>
  %at = linalg.transpose ins(%a4 : tensor<1x2x6x4xf16>) outs(%at.e : tensor<1x2x4x6xf16>) permutation = [0, 1, 3, 2]
  // af = reshape(shape=os,x=at)
  %af.s = arith.constant dense<[1, 8, 1, 6]> : tensor<4xi64>
  %af = tensor.reshape %at(%af.s) : (tensor<1x2x4x6xf16>, tensor<4xi64>) -> tensor<1x8x1x6xf16>
  // oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)
  %oo.e = tensor.empty() : tensor<1x8x1x6xf16>
  %oo.z = arith.constant 0.0 : f16
  %oo.f = linalg.fill ins(%oo.z : f16) outs(%oo.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %oo = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%af, %Wo : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%oo.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))
  %out = tensor.concat dim(1) %oo, %qf, %kf, %vf, %af, %xn : (tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) -> tensor<1x48x1x6xf16>
  return %out : tensor<1x48x1x6xf16>
}

// MIL: ane_ffn_fwd_taps
//   %rw: BLOBFILE @model_path/weights/rms2.bin @ offset 64
//   %W1: BLOBFILE @model_path/weights/w1.bin @ offset 64
//   %W3: BLOBFILE @model_path/weights/w3.bin @ offset 64
//   %W2: BLOBFILE @model_path/weights/w2.bin @ offset 64
func.func @ane_ffn_fwd_taps(%x: tensor<1x8x1x6xf16>, %rw: tensor<1x8x1x1xf16>, %W1: tensor<12x8x1x1xf16>, %W3: tensor<12x8x1x1xf16>, %W2: tensor<8x12x1x1xf16>)
    -> (tensor<1x52x1x6xf16>) {
  // sq = mul(x=x,y=x)
  %sq.e = tensor.empty() : tensor<1x8x1x6xf16>
  %sq = linalg.mul ins(%x, %x : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%sq.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)
  %ss.e = tensor.empty() : tensor<1x1x6xf16>
  %ss.z = arith.constant 0.0 : f16
  %ss.f = linalg.fill ins(%ss.z : f16) outs(%ss.e : tensor<1x1x6xf16>) -> tensor<1x1x6xf16>
  %ss.r = linalg.reduce { arith.addf } ins(%sq : tensor<1x8x1x6xf16>) outs(%ss.f : tensor<1x1x6xf16>) dimensions = [1]
  %ss = tensor.expand_shape %ss.r [[0], [1, 2], [3]] output_shape [1, 1, 1, 6] : tensor<1x1x6xf16> into tensor<1x1x1x6xf16>
  // ss2 = mul(x=ss,y=invd)
  %invd = arith.constant 0.125 : f16
  %ss2.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss, %invd : tensor<1x1x1x6xf16>, f16) outs(%ss2.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // ss3 = add(x=ss2,y=eps)
  %eps = arith.constant 1.0e-05 : f16
  %ss3.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss2, %eps : tensor<1x1x1x6xf16>, f16) outs(%ss3.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // rrms = pow(x=ss3,y=nhalf)
  %nhalf = arith.constant -0.5 : f16
  %rrms.e = tensor.empty() : tensor<1x1x1x6xf16>
  %rrms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss3, %nhalf : tensor<1x1x1x6xf16>, f16) outs(%rrms.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = math.powf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // xr = mul(x=x,y=rrms)
  %xr.e = tensor.empty() : tensor<1x8x1x6xf16>
  %xr = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x, %rrms : tensor<1x8x1x6xf16>, tensor<1x1x1x6xf16>) outs(%xr.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // xn = mul(x=xr,y=rw)
  %xn.e = tensor.empty() : tensor<1x8x1x6xf16>
  %xn = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%xr, %rw : tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>) outs(%xn.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)
  %h1.e = tensor.empty() : tensor<1x12x1x6xf16>
  %h1.z = arith.constant 0.0 : f16
  %h1.f = linalg.fill ins(%h1.z : f16) outs(%h1.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  %h1 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%xn, %W1 : tensor<1x8x1x6xf16>, tensor<12x8x1x1xf16>) outs(%h1.f : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)
  %h3.e = tensor.empty() : tensor<1x12x1x6xf16>
  %h3.z = arith.constant 0.0 : f16
  %h3.f = linalg.fill ins(%h3.z : f16) outs(%h3.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  %h3 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%xn, %W3 : tensor<1x8x1x6xf16>, tensor<12x8x1x1xf16>) outs(%h3.f : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // sig = sigmoid(x=h1)
  %sig.e = tensor.empty() : tensor<1x12x1x6xf16>
  %sig = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%h1 : tensor<1x12x1x6xf16>) outs(%sig.e : tensor<1x12x1x6xf16>) {
    ^bb0(%_a0 : f16, %_o : f16):
      %_one = arith.constant 1.0 : f16
      %_n = arith.negf %_a0 : f16
      %_ex = math.exp %_n : f16
      %_den = arith.addf %_one, %_ex : f16
      %_r = arith.divf %_one, %_den : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x1x6xf16>
  // silu = mul(x=h1,y=sig)
  %silu.e = tensor.empty() : tensor<1x12x1x6xf16>
  %silu = linalg.mul ins(%h1, %sig : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%silu.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // gate = mul(x=silu,y=h3)
  %gate.e = tensor.empty() : tensor<1x12x1x6xf16>
  %gate = linalg.mul ins(%silu, %h3 : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%gate.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)
  %y.e = tensor.empty() : tensor<1x8x1x6xf16>
  %y.z = arith.constant 0.0 : f16
  %y.f = linalg.fill ins(%y.z : f16) outs(%y.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %y = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%gate, %W2 : tensor<1x12x1x6xf16>, tensor<8x12x1x1xf16>) outs(%y.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))
  %out = tensor.concat dim(1) %y, %h1, %h3, %gate, %xn : (tensor<1x8x1x6xf16>, tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>, tensor<1x8x1x6xf16>) -> tensor<1x52x1x6xf16>
  return %out : tensor<1x52x1x6xf16>
}

// MIL: ane_ffn_bwd
//   %W2t: BLOBFILE @model_path/weights/w2t.bin @ offset 64
//   %W1t: BLOBFILE @model_path/weights/w1t.bin @ offset 64
//   %W3t: BLOBFILE @model_path/weights/w3t.bin @ offset 64
func.func @ane_ffn_bwd(%x: tensor<1x32x1x6xf16>, %W2t: tensor<12x8x1x1xf16>, %W1t: tensor<8x12x1x1xf16>, %W3t: tensor<8x12x1x1xf16>)
    -> (tensor<1x32x1x6xf16>) {
  // dffn = slice_by_size(x=x,begin=bd,size=sd)
  %dffn = tensor.extract_slice %x[0, 0, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x8x1x6xf16>
  // h1 = slice_by_size(x=x,begin=b1,size=s1)
  %h1 = tensor.extract_slice %x[0, 8, 0, 0] [1, 12, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x12x1x6xf16>
  // h3 = slice_by_size(x=x,begin=b3,size=s1)
  %h3 = tensor.extract_slice %x[0, 20, 0, 0] [1, 12, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x12x1x6xf16>
  // dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)
  %dsilu.e = tensor.empty() : tensor<1x12x1x6xf16>
  %dsilu.z = arith.constant 0.0 : f16
  %dsilu.f = linalg.fill ins(%dsilu.z : f16) outs(%dsilu.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  %dsilu = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dffn, %W2t : tensor<1x8x1x6xf16>, tensor<12x8x1x1xf16>) outs(%dsilu.f : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // sig = sigmoid(x=h1)
  %sig.e = tensor.empty() : tensor<1x12x1x6xf16>
  %sig = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%h1 : tensor<1x12x1x6xf16>) outs(%sig.e : tensor<1x12x1x6xf16>) {
    ^bb0(%_a0 : f16, %_o : f16):
      %_one = arith.constant 1.0 : f16
      %_n = arith.negf %_a0 : f16
      %_ex = math.exp %_n : f16
      %_den = arith.addf %_one, %_ex : f16
      %_r = arith.divf %_one, %_den : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x1x6xf16>
  // oms = sub(x=one,y=sig)
  %one = arith.constant 1.0 : f16
  %oms.e = tensor.empty() : tensor<1x12x1x6xf16>
  %oms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%one, %sig : f16, tensor<1x12x1x6xf16>) outs(%oms.e : tensor<1x12x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.subf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x1x6xf16>
  // homs = mul(x=h1,y=oms)
  %homs.e = tensor.empty() : tensor<1x12x1x6xf16>
  %homs = linalg.mul ins(%h1, %oms : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%homs.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // brk = add(x=one,y=homs)
  %brk.e = tensor.empty() : tensor<1x12x1x6xf16>
  %brk = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%one, %homs : f16, tensor<1x12x1x6xf16>) outs(%brk.e : tensor<1x12x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x1x6xf16>
  // dsd = mul(x=sig,y=brk)
  %dsd.e = tensor.empty() : tensor<1x12x1x6xf16>
  %dsd = linalg.mul ins(%sig, %brk : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%dsd.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // t1 = mul(x=dsilu,y=h3)
  %t1.e = tensor.empty() : tensor<1x12x1x6xf16>
  %t1 = linalg.mul ins(%dsilu, %h3 : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%t1.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // dh1 = mul(x=t1,y=dsd)
  %dh1.e = tensor.empty() : tensor<1x12x1x6xf16>
  %dh1 = linalg.mul ins(%t1, %dsd : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%dh1.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // slh = mul(x=h1,y=sig)
  %slh.e = tensor.empty() : tensor<1x12x1x6xf16>
  %slh = linalg.mul ins(%h1, %sig : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%slh.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // dh3 = mul(x=dsilu,y=slh)
  %dh3.e = tensor.empty() : tensor<1x12x1x6xf16>
  %dh3 = linalg.mul ins(%dsilu, %slh : tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) outs(%dh3.e : tensor<1x12x1x6xf16>) -> tensor<1x12x1x6xf16>
  // dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)
  %dx1.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dx1.z = arith.constant 0.0 : f16
  %dx1.f = linalg.fill ins(%dx1.z : f16) outs(%dx1.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %dx1 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dh1, %W1t : tensor<1x12x1x6xf16>, tensor<8x12x1x1xf16>) outs(%dx1.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)
  %dx3.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dx3.z = arith.constant 0.0 : f16
  %dx3.f = linalg.fill ins(%dx3.z : f16) outs(%dx3.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %dx3 = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dh3, %W3t : tensor<1x12x1x6xf16>, tensor<8x12x1x1xf16>) outs(%dx3.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // dx = add(x=dx1,y=dx3)
  %dx.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dx = linalg.add ins(%dx1, %dx3 : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%dx.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))
  %out = tensor.concat dim(1) %dx, %dh1, %dh3 : (tensor<1x8x1x6xf16>, tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) -> tensor<1x32x1x6xf16>
  return %out : tensor<1x32x1x6xf16>
}

// MIL: ane_qkvb
//   %Wqt: BLOBFILE @model_path/weights/wqt.bin @ offset 64
//   %Wkt: BLOBFILE @model_path/weights/wkt.bin @ offset 64
//   %Wvt: BLOBFILE @model_path/weights/wvt.bin @ offset 64
func.func @ane_qkvb(%x: tensor<1x24x1x6xf16>, %Wqt: tensor<8x8x1x1xf16>, %Wkt: tensor<8x8x1x1xf16>, %Wvt: tensor<8x8x1x1xf16>)
    -> (tensor<1x8x1x6xf16>) {
  // dq = slice_by_size(x=x,begin=b0,size=sz)
  %dq = tensor.extract_slice %x[0, 0, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x24x1x6xf16> to tensor<1x8x1x6xf16>
  // dk = slice_by_size(x=x,begin=b1,size=sz)
  %dk = tensor.extract_slice %x[0, 8, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x24x1x6xf16> to tensor<1x8x1x6xf16>
  // dv = slice_by_size(x=x,begin=b2,size=sz)
  %dv = tensor.extract_slice %x[0, 16, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x24x1x6xf16> to tensor<1x8x1x6xf16>
  // dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq)
  %dxq.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dxq.z = arith.constant 0.0 : f16
  %dxq.f = linalg.fill ins(%dxq.z : f16) outs(%dxq.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %dxq = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dq, %Wqt : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%dxq.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk)
  %dxk.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dxk.z = arith.constant 0.0 : f16
  %dxk.f = linalg.fill ins(%dxk.z : f16) outs(%dxk.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %dxk = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dk, %Wkt : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%dxk.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv)
  %dxv.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dxv.z = arith.constant 0.0 : f16
  %dxv.f = linalg.fill ins(%dxv.z : f16) outs(%dxv.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %dxv = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dv, %Wvt : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%dxv.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // dxqk = add(x=dxq,y=dxk)
  %dxqk.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dxqk = linalg.add ins(%dxq, %dxk : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%dxqk.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // out = add(x=dxqk,y=dxv)
  %out.e = tensor.empty() : tensor<1x8x1x6xf16>
  %out = linalg.add ins(%dxqk, %dxv : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%out.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  return %out : tensor<1x8x1x6xf16>
}

// MIL: ane_sdpa_bwd1
//   %Wot: BLOBFILE @model_path/weights/wot.bin @ offset 64
//   %cm: BLOBFILE @model_path/weights/mask.bin @ offset 64
func.func @ane_sdpa_bwd1(%x: tensor<1x32x1x6xf16>, %Wot: tensor<8x8x1x1xf16>, %cm: tensor<1x1x6x6xf16>)
    -> (tensor<1x32x1x6xf16>) {
  // qf = slice_by_size(x=x,begin=b0,size=sz)
  %qf = tensor.extract_slice %x[0, 0, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x8x1x6xf16>
  // kf = slice_by_size(x=x,begin=b1,size=sz)
  %kf = tensor.extract_slice %x[0, 8, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x8x1x6xf16>
  // vf = slice_by_size(x=x,begin=b2,size=sz)
  %vf = tensor.extract_slice %x[0, 16, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x8x1x6xf16>
  // dx2f = slice_by_size(x=x,begin=b3,size=sz)
  %dx2f = tensor.extract_slice %x[0, 24, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x32x1x6xf16> to tensor<1x8x1x6xf16>
  // df = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=dx2f)
  %df.e = tensor.empty() : tensor<1x8x1x6xf16>
  %df.z = arith.constant 0.0 : f16
  %df.f = linalg.fill ins(%df.z : f16) outs(%df.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  %df = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%dx2f, %Wot : tensor<1x8x1x6xf16>, tensor<8x8x1x1xf16>) outs(%df.f : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // qr = reshape(shape=rsh,x=qf)
  %qr.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %qr = tensor.reshape %qf(%qr.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // q = transpose(perm=pm,x=qr)
  %q.e = tensor.empty() : tensor<1x2x6x4xf16>
  %q = linalg.transpose ins(%qr : tensor<1x2x4x6xf16>) outs(%q.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // kr = reshape(shape=rsh,x=kf)
  %kr.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %kr = tensor.reshape %kf(%kr.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // k = transpose(perm=pm,x=kr)
  %k.e = tensor.empty() : tensor<1x2x6x4xf16>
  %k = linalg.transpose ins(%kr : tensor<1x2x4x6xf16>) outs(%k.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // vr = reshape(shape=rsh,x=vf)
  %vr.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %vr = tensor.reshape %vf(%vr.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // v = transpose(perm=pm,x=vr)
  %v.e = tensor.empty() : tensor<1x2x6x4xf16>
  %v = linalg.transpose ins(%vr : tensor<1x2x4x6xf16>) outs(%v.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // dr = reshape(shape=rsh,x=df)
  %dr.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %dr = tensor.reshape %df(%dr.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // da = transpose(perm=pm,x=dr)
  %da.e = tensor.empty() : tensor<1x2x6x4xf16>
  %da = linalg.transpose ins(%dr : tensor<1x2x4x6xf16>) outs(%da.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)
  %sc1.e = tensor.empty() : tensor<1x2x6x6xf16>
  %sc1.z = arith.constant 0.0 : f16
  %sc1.f = linalg.fill ins(%sc1.z : f16) outs(%sc1.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  %sc1 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, j, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%q, %k : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16>) outs(%sc1.f : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // sc2 = mul(x=sc1,y=scv)
  %scv = arith.constant 0.5 : f16
  %sc2.e = tensor.empty() : tensor<1x2x6x6xf16>
  %sc2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc1, %scv : tensor<1x2x6x6xf16>, f16) outs(%sc2.e : tensor<1x2x6x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x2x6x6xf16>
  // ms = add(x=sc2,y=cm)
  %ms.e = tensor.empty() : tensor<1x2x6x6xf16>
  %ms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc2, %cm : tensor<1x2x6x6xf16>, tensor<1x1x6x6xf16>) outs(%ms.e : tensor<1x2x6x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x2x6x6xf16>
  // probs = softmax(axis=sax,x=ms)
  %probs.e = tensor.empty() : tensor<1x2x6x6xf16>
  %probs = linalg.softmax dimension(3) ins(%ms : tensor<1x2x6x6xf16>) outs(%probs.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da)
  %dv4.e = tensor.empty() : tensor<1x2x6x4xf16>
  %dv4.z = arith.constant 0.0 : f16
  %dv4.f = linalg.fill ins(%dv4.z : f16) outs(%dv4.e : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  %dv4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, k, i)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%probs, %da : tensor<1x2x6x6xf16>, tensor<1x2x6x4xf16>) outs(%dv4.f : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  // dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)
  %dp4.e = tensor.empty() : tensor<1x2x6x6xf16>
  %dp4.z = arith.constant 0.0 : f16
  %dp4.f = linalg.fill ins(%dp4.z : f16) outs(%dp4.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  %dp4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, j, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%da, %v : tensor<1x2x6x4xf16>, tensor<1x2x6x4xf16>) outs(%dp4.f : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // dvt = transpose(perm=pm,x=dv4)
  %dvt.e = tensor.empty() : tensor<1x2x4x6xf16>
  %dvt = linalg.transpose ins(%dv4 : tensor<1x2x6x4xf16>) outs(%dvt.e : tensor<1x2x4x6xf16>) permutation = [0, 1, 3, 2]
  // dvf = reshape(shape=dvs,x=dvt)
  %dvf.s = arith.constant dense<[1, 8, 1, 6]> : tensor<4xi64>
  %dvf = tensor.reshape %dvt(%dvf.s) : (tensor<1x2x4x6xf16>, tensor<4xi64>) -> tensor<1x8x1x6xf16>
  // pf = reshape(shape=scs,x=probs)
  %pf.s = arith.constant dense<[1, 12, 1, 6]> : tensor<4xi64>
  %pf = tensor.reshape %probs(%pf.s) : (tensor<1x2x6x6xf16>, tensor<4xi64>) -> tensor<1x12x1x6xf16>
  // dpf = reshape(shape=scs,x=dp4)
  %dpf.s = arith.constant dense<[1, 12, 1, 6]> : tensor<4xi64>
  %dpf = tensor.reshape %dp4(%dpf.s) : (tensor<1x2x6x6xf16>, tensor<4xi64>) -> tensor<1x12x1x6xf16>
  // out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))
  %out = tensor.concat dim(1) %dvf, %pf, %dpf : (tensor<1x8x1x6xf16>, tensor<1x12x1x6xf16>, tensor<1x12x1x6xf16>) -> tensor<1x32x1x6xf16>
  return %out : tensor<1x32x1x6xf16>
}

// MIL: ane_sdpa_bwd2
func.func @ane_sdpa_bwd2(%x: tensor<1x40x1x6xf16>)
    -> (tensor<1x16x1x6xf16>) {
  // pf = slice_by_size(x=x,begin=b0,size=sz_sc)
  %pf = tensor.extract_slice %x[0, 0, 0, 0] [1, 12, 1, 6] [1, 1, 1, 1] : tensor<1x40x1x6xf16> to tensor<1x12x1x6xf16>
  // dpf = slice_by_size(x=x,begin=b1,size=sz_sc)
  %dpf = tensor.extract_slice %x[0, 12, 0, 0] [1, 12, 1, 6] [1, 1, 1, 1] : tensor<1x40x1x6xf16> to tensor<1x12x1x6xf16>
  // qf = slice_by_size(x=x,begin=b2,size=sz_d)
  %qf = tensor.extract_slice %x[0, 24, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x40x1x6xf16> to tensor<1x8x1x6xf16>
  // kf = slice_by_size(x=x,begin=b3,size=sz_d)
  %kf = tensor.extract_slice %x[0, 32, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x40x1x6xf16> to tensor<1x8x1x6xf16>
  // probs = reshape(shape=ssh,x=pf)
  %probs.s = arith.constant dense<[1, 2, 6, 6]> : tensor<4xi64>
  %probs = tensor.reshape %pf(%probs.s) : (tensor<1x12x1x6xf16>, tensor<4xi64>) -> tensor<1x2x6x6xf16>
  // dp = reshape(shape=ssh,x=dpf)
  %dp.s = arith.constant dense<[1, 2, 6, 6]> : tensor<4xi64>
  %dp = tensor.reshape %dpf(%dp.s) : (tensor<1x12x1x6xf16>, tensor<4xi64>) -> tensor<1x2x6x6xf16>
  // qr = reshape(shape=rsh,x=qf)
  %qr.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %qr = tensor.reshape %qf(%qr.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // q = transpose(perm=pm,x=qr)
  %q.e = tensor.empty() : tensor<1x2x6x4xf16>
  %q = linalg.transpose ins(%qr : tensor<1x2x4x6xf16>) outs(%q.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // kr = reshape(shape=rsh,x=kf)
  %kr.s = arith.constant dense<[1, 2, 4, 6]> : tensor<4xi64>
  %kr = tensor.reshape %kf(%kr.s) : (tensor<1x8x1x6xf16>, tensor<4xi64>) -> tensor<1x2x4x6xf16>
  // k = transpose(perm=pm,x=kr)
  %k.e = tensor.empty() : tensor<1x2x6x4xf16>
  %k = linalg.transpose ins(%kr : tensor<1x2x4x6xf16>) outs(%k.e : tensor<1x2x6x4xf16>) permutation = [0, 1, 3, 2]
  // pdp = mul(x=probs,y=dp)
  %pdp.e = tensor.empty() : tensor<1x2x6x6xf16>
  %pdp = linalg.mul ins(%probs, %dp : tensor<1x2x6x6xf16>, tensor<1x2x6x6xf16>) outs(%pdp.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)
  %spdp.e = tensor.empty() : tensor<1x2x6xf16>
  %spdp.z = arith.constant 0.0 : f16
  %spdp.f = linalg.fill ins(%spdp.z : f16) outs(%spdp.e : tensor<1x2x6xf16>) -> tensor<1x2x6xf16>
  %spdp.r = linalg.reduce { arith.addf } ins(%pdp : tensor<1x2x6x6xf16>) outs(%spdp.f : tensor<1x2x6xf16>) dimensions = [3]
  %spdp = tensor.expand_shape %spdp.r [[0], [1], [2, 3]] output_shape [1, 2, 6, 1] : tensor<1x2x6xf16> into tensor<1x2x6x1xf16>
  // dps = sub(x=dp,y=spdp)
  %dps.e = tensor.empty() : tensor<1x2x6x6xf16>
  %dps = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%dp, %spdp : tensor<1x2x6x6xf16>, tensor<1x2x6x1xf16>) outs(%dps.e : tensor<1x2x6x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.subf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x2x6x6xf16>
  // ds0 = mul(x=probs,y=dps)
  %ds0.e = tensor.empty() : tensor<1x2x6x6xf16>
  %ds0 = linalg.mul ins(%probs, %dps : tensor<1x2x6x6xf16>, tensor<1x2x6x6xf16>) outs(%ds0.e : tensor<1x2x6x6xf16>) -> tensor<1x2x6x6xf16>
  // ds = mul(x=ds0,y=scv)
  %scv = arith.constant 0.5 : f16
  %ds.e = tensor.empty() : tensor<1x2x6x6xf16>
  %ds = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ds0, %scv : tensor<1x2x6x6xf16>, f16) outs(%ds.e : tensor<1x2x6x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x2x6x6xf16>
  // dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)
  %dq4.e = tensor.empty() : tensor<1x2x6x4xf16>
  %dq4.z = arith.constant 0.0 : f16
  %dq4.f = linalg.fill ins(%dq4.z : f16) outs(%dq4.e : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  %dq4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%ds, %k : tensor<1x2x6x6xf16>, tensor<1x2x6x4xf16>) outs(%dq4.f : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  // dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)
  %dk4.e = tensor.empty() : tensor<1x2x6x4xf16>
  %dk4.z = arith.constant 0.0 : f16
  %dk4.f = linalg.fill ins(%dk4.z : f16) outs(%dk4.e : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  %dk4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, k, i)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%ds, %q : tensor<1x2x6x6xf16>, tensor<1x2x6x4xf16>) outs(%dk4.f : tensor<1x2x6x4xf16>) -> tensor<1x2x6x4xf16>
  // dqt = transpose(perm=pm,x=dq4)
  %dqt.e = tensor.empty() : tensor<1x2x4x6xf16>
  %dqt = linalg.transpose ins(%dq4 : tensor<1x2x6x4xf16>) outs(%dqt.e : tensor<1x2x4x6xf16>) permutation = [0, 1, 3, 2]
  // dkt = transpose(perm=pm,x=dk4)
  %dkt.e = tensor.empty() : tensor<1x2x4x6xf16>
  %dkt = linalg.transpose ins(%dk4 : tensor<1x2x6x4xf16>) outs(%dkt.e : tensor<1x2x4x6xf16>) permutation = [0, 1, 3, 2]
  // dqf = reshape(shape=fs,x=dqt)
  %dqf.s = arith.constant dense<[1, 8, 1, 6]> : tensor<4xi64>
  %dqf = tensor.reshape %dqt(%dqf.s) : (tensor<1x2x4x6xf16>, tensor<4xi64>) -> tensor<1x8x1x6xf16>
  // dkf = reshape(shape=fs,x=dkt)
  %dkf.s = arith.constant dense<[1, 8, 1, 6]> : tensor<4xi64>
  %dkf = tensor.reshape %dkt(%dkf.s) : (tensor<1x2x4x6xf16>, tensor<4xi64>) -> tensor<1x8x1x6xf16>
  // out = concat(axis=cax,interleave=cid,values=(dqf,dkf))
  %out = tensor.concat dim(1) %dqf, %dkf : (tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) -> tensor<1x16x1x6xf16>
  return %out : tensor<1x16x1x6xf16>
}

// MIL: ane_rmsnorm_bwd
//   %w: BLOBFILE @model_path/weights/rms_w.bin @ offset 64
func.func @ane_rmsnorm_bwd(%inp: tensor<1x16x1x6xf16>, %w: tensor<1x8x1x1xf16>)
    -> (tensor<1x8x1x6xf16>) {
  // dy = slice_by_size(x=inp,begin=b0,size=sz)
  %dy = tensor.extract_slice %inp[0, 0, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x16x1x6xf16> to tensor<1x8x1x6xf16>
  // x = slice_by_size(x=inp,begin=b1,size=sz)
  %x = tensor.extract_slice %inp[0, 8, 0, 0] [1, 8, 1, 6] [1, 1, 1, 1] : tensor<1x16x1x6xf16> to tensor<1x8x1x6xf16>
  // sq = mul(x=x,y=x)
  %sq.e = tensor.empty() : tensor<1x8x1x6xf16>
  %sq = linalg.mul ins(%x, %x : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%sq.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)
  %ss.e = tensor.empty() : tensor<1x1x6xf16>
  %ss.z = arith.constant 0.0 : f16
  %ss.f = linalg.fill ins(%ss.z : f16) outs(%ss.e : tensor<1x1x6xf16>) -> tensor<1x1x6xf16>
  %ss.r = linalg.reduce { arith.addf } ins(%sq : tensor<1x8x1x6xf16>) outs(%ss.f : tensor<1x1x6xf16>) dimensions = [1]
  %ss = tensor.expand_shape %ss.r [[0], [1, 2], [3]] output_shape [1, 1, 1, 6] : tensor<1x1x6xf16> into tensor<1x1x1x6xf16>
  // ss2 = mul(x=ss,y=invd)
  %invd = arith.constant 0.125 : f16
  %ss2.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss, %invd : tensor<1x1x1x6xf16>, f16) outs(%ss2.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // ss3 = add(x=ss2,y=eps)
  %eps = arith.constant 1.0e-05 : f16
  %ss3.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss2, %eps : tensor<1x1x1x6xf16>, f16) outs(%ss3.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // rrms = pow(x=ss3,y=nhalf)
  %nhalf = arith.constant -0.5 : f16
  %rrms.e = tensor.empty() : tensor<1x1x1x6xf16>
  %rrms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss3, %nhalf : tensor<1x1x1x6xf16>, f16) outs(%rrms.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = math.powf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // dyw = mul(x=dy,y=w)
  %dyw.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dyw = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%dy, %w : tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>) outs(%dyw.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // dywx = mul(x=dyw,y=x)
  %dywx.e = tensor.empty() : tensor<1x8x1x6xf16>
  %dywx = linalg.mul ins(%dyw, %x : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%dywx.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // dot_sum = reduce_sum(x=dywx,axes=rax,keep_dims=kd)
  %dot_sum.e = tensor.empty() : tensor<1x1x6xf16>
  %dot_sum.z = arith.constant 0.0 : f16
  %dot_sum.f = linalg.fill ins(%dot_sum.z : f16) outs(%dot_sum.e : tensor<1x1x6xf16>) -> tensor<1x1x6xf16>
  %dot_sum.r = linalg.reduce { arith.addf } ins(%dywx : tensor<1x8x1x6xf16>) outs(%dot_sum.f : tensor<1x1x6xf16>) dimensions = [1]
  %dot_sum = tensor.expand_shape %dot_sum.r [[0], [1, 2], [3]] output_shape [1, 1, 1, 6] : tensor<1x1x6xf16> into tensor<1x1x1x6xf16>
  // dot_sc = mul(x=dot_sum,y=invd)
  %dot_sc.e = tensor.empty() : tensor<1x1x1x6xf16>
  %dot_sc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%dot_sum, %invd : tensor<1x1x1x6xf16>, f16) outs(%dot_sc.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // rrms2 = mul(x=rrms,y=rrms)
  %rrms2.e = tensor.empty() : tensor<1x1x1x6xf16>
  %rrms2 = linalg.mul ins(%rrms, %rrms : tensor<1x1x1x6xf16>, tensor<1x1x1x6xf16>) outs(%rrms2.e : tensor<1x1x1x6xf16>) -> tensor<1x1x1x6xf16>
  // coeff = mul(x=dot_sc,y=rrms2)
  %coeff.e = tensor.empty() : tensor<1x1x1x6xf16>
  %coeff = linalg.mul ins(%dot_sc, %rrms2 : tensor<1x1x1x6xf16>, tensor<1x1x1x6xf16>) outs(%coeff.e : tensor<1x1x1x6xf16>) -> tensor<1x1x1x6xf16>
  // xc = mul(x=x,y=coeff)
  %xc.e = tensor.empty() : tensor<1x8x1x6xf16>
  %xc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x, %coeff : tensor<1x8x1x6xf16>, tensor<1x1x1x6xf16>) outs(%xc.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // diff = sub(x=dyw,y=xc)
  %diff.e = tensor.empty() : tensor<1x8x1x6xf16>
  %diff = linalg.sub ins(%dyw, %xc : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%diff.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // out = mul(x=diff,y=rrms)
  %out.e = tensor.empty() : tensor<1x8x1x6xf16>
  %out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%diff, %rrms : tensor<1x8x1x6xf16>, tensor<1x1x1x6xf16>) outs(%out.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  return %out : tensor<1x8x1x6xf16>
}

// MIL: ane_classifier_fwd
//   %We: BLOBFILE @model_path/weights/embed.bin @ offset 64
func.func @ane_classifier_fwd(%x: tensor<1x8x1x6xf16>, %We: tensor<10x8x1x1xf16>)
    -> (tensor<1x10x1x6xf16>) {
  // out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=We,x=x)
  %out.e = tensor.empty() : tensor<1x10x1x6xf16>
  %out.z = arith.constant 0.0 : f16
  %out.f = linalg.fill ins(%out.z : f16) outs(%out.e : tensor<1x10x1x6xf16>) -> tensor<1x10x1x6xf16>
  %out = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : tensor<2xi64>, strides = dense<[1, 1]> : tensor<2xi64>}
      ins(%x, %We : tensor<1x8x1x6xf16>, tensor<10x8x1x1xf16>) outs(%out.f : tensor<1x10x1x6xf16>) -> tensor<1x10x1x6xf16>
  return %out : tensor<1x10x1x6xf16>
}

// MIL: ane_classifier_bwd
//   %Wet: BLOBFILE @model_path/weights/embed_t.bin @ offset 64
func.func @ane_classifier_bwd(%dl: tensor<1x10x1x6xf16>, %Wet: tensor<1x8x10xf16>)
    -> (tensor<1x8x1x6xf16>) {
  // dl3 = reshape(shape=sh3,x=dl)
  %dl3.s = arith.constant dense<[1, 10, 6]> : tensor<3xi64>
  %dl3 = tensor.reshape %dl(%dl3.s) : (tensor<1x10x1x6xf16>, tensor<3xi64>) -> tensor<1x10x6xf16>
  // dx3 = matmul(transpose_x=bF,transpose_y=bF,x=Wet,y=dl3)
  %dx3.e = tensor.empty() : tensor<1x8x6xf16>
  %dx3.z = arith.constant 0.0 : f16
  %dx3.f = linalg.fill ins(%dx3.z : f16) outs(%dx3.e : tensor<1x8x6xf16>) -> tensor<1x8x6xf16>
  %dx3 = linalg.contract indexing_maps = [affine_map<(b0, i, j, k) -> (b0, i, k)>, affine_map<(b0, i, j, k) -> (b0, k, j)>, affine_map<(b0, i, j, k) -> (b0, i, j)>]
      ins(%Wet, %dl3 : tensor<1x8x10xf16>, tensor<1x10x6xf16>) outs(%dx3.f : tensor<1x8x6xf16>) -> tensor<1x8x6xf16>
  // out = reshape(shape=sh4,x=dx3)
  %out.s = arith.constant dense<[1, 8, 1, 6]> : tensor<4xi64>
  %out = tensor.reshape %dx3(%out.s) : (tensor<1x8x6xf16>, tensor<4xi64>) -> tensor<1x8x1x6xf16>
  return %out : tensor<1x8x1x6xf16>
}

// MIL: ane_softmax_vocab
func.func @ane_softmax_vocab(%x: tensor<1x10x1x6xf16>)
    -> (tensor<1x10x1x6xf16>) {
  // out = softmax(axis=ax,x=x)
  %out.e = tensor.empty() : tensor<1x10x1x6xf16>
  %out = linalg.softmax dimension(1) ins(%x : tensor<1x10x1x6xf16>) outs(%out.e : tensor<1x10x1x6xf16>) -> tensor<1x10x1x6xf16>
  return %out : tensor<1x10x1x6xf16>
}

// MIL: ane_final_rmsnorm
//   %rw: BLOBFILE @model_path/weights/rms_w.bin @ offset 64
func.func @ane_final_rmsnorm(%x: tensor<1x8x1x6xf16>, %rw: tensor<1x8x1x1xf16>)
    -> (tensor<1x8x1x6xf16>) {
  // sq = mul(x=x,y=x)
  %sq.e = tensor.empty() : tensor<1x8x1x6xf16>
  %sq = linalg.mul ins(%x, %x : tensor<1x8x1x6xf16>, tensor<1x8x1x6xf16>) outs(%sq.e : tensor<1x8x1x6xf16>) -> tensor<1x8x1x6xf16>
  // ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)
  %ss.e = tensor.empty() : tensor<1x1x6xf16>
  %ss.z = arith.constant 0.0 : f16
  %ss.f = linalg.fill ins(%ss.z : f16) outs(%ss.e : tensor<1x1x6xf16>) -> tensor<1x1x6xf16>
  %ss.r = linalg.reduce { arith.addf } ins(%sq : tensor<1x8x1x6xf16>) outs(%ss.f : tensor<1x1x6xf16>) dimensions = [1]
  %ss = tensor.expand_shape %ss.r [[0], [1, 2], [3]] output_shape [1, 1, 1, 6] : tensor<1x1x6xf16> into tensor<1x1x1x6xf16>
  // ss2 = mul(x=ss,y=invd)
  %invd = arith.constant 0.125 : f16
  %ss2.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss, %invd : tensor<1x1x1x6xf16>, f16) outs(%ss2.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // ss3 = add(x=ss2,y=eps)
  %eps = arith.constant 1.0e-05 : f16
  %ss3.e = tensor.empty() : tensor<1x1x1x6xf16>
  %ss3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss2, %eps : tensor<1x1x1x6xf16>, f16) outs(%ss3.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // rrms = pow(x=ss3,y=nhalf)
  %nhalf = arith.constant -0.5 : f16
  %rrms.e = tensor.empty() : tensor<1x1x1x6xf16>
  %rrms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ss3, %nhalf : tensor<1x1x1x6xf16>, f16) outs(%rrms.e : tensor<1x1x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = math.powf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x1x1x6xf16>
  // xr = mul(x=x,y=rrms)
  %xr.e = tensor.empty() : tensor<1x8x1x6xf16>
  %xr = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%x, %rrms : tensor<1x8x1x6xf16>, tensor<1x1x1x6xf16>) outs(%xr.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  // out = mul(x=xr,y=rw)
  %out.e = tensor.empty() : tensor<1x8x1x6xf16>
  %out = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%xr, %rw : tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>) outs(%out.e : tensor<1x8x1x6xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x8x1x6xf16>
  return %out : tensor<1x8x1x6xf16>
}

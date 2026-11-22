// MIL: ane_dyn_matmul_mil
func.func @ane_dyn_matmul_mil(%x: tensor<1x768x1x2304xf16>)
    -> (tensor<1x2048x1x256xf16>) {
  // mm_act = slice_by_size(x=x,begin=mm_ba,size=mm_sa)
  %mm_act = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x2304xf16> to tensor<1x768x1x256xf16>
  // mm_wt = slice_by_size(x=x,begin=mm_bw,size=mm_sw)
  %mm_wt = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 2048] [1, 1, 1, 1] : tensor<1x768x1x2304xf16> to tensor<1x768x1x2048xf16>
  // mm_a2 = reshape(shape=mm_ra,x=mm_act)
  %mm_a2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %mm_a2 = tensor.reshape %mm_act(%mm_a2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // mm_a3 = transpose(perm=mm_pm,x=mm_a2)
  %mm_a3.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_a3 = linalg.transpose ins(%mm_a2 : tensor<1x1x768x256xf16>) outs(%mm_a3.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // mm_W = reshape(shape=mm_rw,x=mm_wt)
  %mm_W.s = arith.constant dense<[1, 1, 768, 2048]> : tensor<4xi64>
  %mm_W = tensor.reshape %mm_wt(%mm_W.s) : (tensor<1x768x1x2048xf16>, tensor<4xi64>) -> tensor<1x1x768x2048xf16>
  // mm_yh = matmul(transpose_x=bF,transpose_y=bF,x=mm_a3,y=mm_W)
  %mm_yh.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %mm_yh.z = arith.constant 0.0 : f16
  %mm_yh.f = linalg.fill ins(%mm_yh.z : f16) outs(%mm_yh.e : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  %mm_yh = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%mm_a3, %mm_W : tensor<1x1x256x768xf16>, tensor<1x1x768x2048xf16>) outs(%mm_yh.f : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  // mm_yt = transpose(perm=mm_pm,x=mm_yh)
  %mm_yt.e = tensor.empty() : tensor<1x1x2048x256xf16>
  %mm_yt = linalg.transpose ins(%mm_yh : tensor<1x1x256x2048xf16>) outs(%mm_yt.e : tensor<1x1x2048x256xf16>) permutation = [0, 1, 3, 2]
  // mm_y = reshape(shape=mm_ro,x=mm_yt)
  %mm_y.s = arith.constant dense<[1, 2048, 1, 256]> : tensor<4xi64>
  %mm_y = tensor.reshape %mm_yt(%mm_y.s) : (tensor<1x1x2048x256xf16>, tensor<4xi64>) -> tensor<1x2048x1x256xf16>
  return %mm_y : tensor<1x2048x1x256xf16>
}

// MIL: ane_sdpa_fwd_dynamic
//   %rope_cos: BLOBFILE @model_path/weights/rope_cos.bin @ offset 64
//   %rope_sin: BLOBFILE @model_path/weights/rope_sin.bin @ offset 64
//   %cm: BLOBFILE @model_path/weights/mask.bin @ offset 64
func.func @ane_sdpa_fwd_dynamic(%x: tensor<1x768x1x2560xf16>, %rope_cos: tensor<1x1x256x64xf16>, %rope_sin: tensor<1x1x256x64xf16>, %cm: tensor<1x1x256x256xf16>)
    -> (tensor<1x3840x1x256xf16>) {
  // xn = slice_by_size(x=x,begin=bx,size=sx)
  %xn = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x2560xf16> to tensor<1x768x1x256xf16>
  // Wq = slice_by_size(x=x,begin=bq,size=swq)
  %Wq = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x2560xf16> to tensor<1x768x1x768xf16>
  // Wk = slice_by_size(x=x,begin=bk,size=swk)
  %Wk = tensor.extract_slice %x[0, 0, 0, 1024] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x2560xf16> to tensor<1x768x1x768xf16>
  // Wv = slice_by_size(x=x,begin=bv,size=swk)
  %Wv = tensor.extract_slice %x[0, 0, 0, 1792] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x2560xf16> to tensor<1x768x1x768xf16>
  // xn2 = reshape(shape=r2,x=xn)
  %xn2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %xn2 = tensor.reshape %xn(%xn2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // xnt = transpose(perm=pm,x=xn2)
  %xnt.e = tensor.empty() : tensor<1x1x256x768xf16>
  %xnt = linalg.transpose ins(%xn2 : tensor<1x1x768x256xf16>) outs(%xnt.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // Wq2 = reshape(shape=rwq,x=Wq)
  %Wq2.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %Wq2 = tensor.reshape %Wq(%Wq2.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // Wk2 = reshape(shape=rwk,x=Wk)
  %Wk2.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %Wk2 = tensor.reshape %Wk(%Wk2.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // Wv2 = reshape(shape=rwk,x=Wv)
  %Wv2.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %Wv2 = tensor.reshape %Wv(%Wv2.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)
  %qm.e = tensor.empty() : tensor<1x1x256x768xf16>
  %qm.z = arith.constant 0.0 : f16
  %qm.f = linalg.fill ins(%qm.z : f16) outs(%qm.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %qm = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%xnt, %Wq2 : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%qm.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)
  %km.e = tensor.empty() : tensor<1x1x256x768xf16>
  %km.z = arith.constant 0.0 : f16
  %km.f = linalg.fill ins(%km.z : f16) outs(%km.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %km = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%xnt, %Wk2 : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%km.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)
  %vm.e = tensor.empty() : tensor<1x1x256x768xf16>
  %vm.z = arith.constant 0.0 : f16
  %vm.f = linalg.fill ins(%vm.z : f16) outs(%vm.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %vm = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%xnt, %Wv2 : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%vm.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // qt = transpose(perm=pm,x=qm)
  %qt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %qt = linalg.transpose ins(%qm : tensor<1x1x256x768xf16>) outs(%qt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // kt = transpose(perm=pm,x=km)
  %kt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %kt = linalg.transpose ins(%km : tensor<1x1x256x768xf16>) outs(%kt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // vt = transpose(perm=pm,x=vm)
  %vt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %vt = linalg.transpose ins(%vm : tensor<1x1x256x768xf16>) outs(%vt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // qf = reshape(shape=qsh,x=qt)
  %qf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %qf = tensor.reshape %qt(%qf.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // kf = reshape(shape=kvsh,x=kt)
  %kf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %kf = tensor.reshape %kt(%kf.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // vf = reshape(shape=kvsh,x=vt)
  %vf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %vf = tensor.reshape %vt(%vf.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // q4 = reshape(shape=qhsh,x=qf)
  %q4.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %q4 = tensor.reshape %qf(%q4.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // q = transpose(perm=pm,x=q4)
  %q.e = tensor.empty() : tensor<1x12x256x64xf16>
  %q = linalg.transpose ins(%q4 : tensor<1x12x64x256xf16>) outs(%q.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // k4 = reshape(shape=khsh,x=kf)
  %k4.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %k4 = tensor.reshape %kf(%k4.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // k = transpose(perm=pm,x=k4)
  %k.e = tensor.empty() : tensor<1x12x256x64xf16>
  %k = linalg.transpose ins(%k4 : tensor<1x12x64x256xf16>) outs(%k.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // v4 = reshape(shape=khsh,x=vf)
  %v4.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %v4 = tensor.reshape %vf(%v4.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // v = transpose(perm=pm,x=v4)
  %v.e = tensor.empty() : tensor<1x12x256x64xf16>
  %v = linalg.transpose ins(%v4 : tensor<1x12x64x256xf16>) outs(%v.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // q_p = reshape(shape=rp_sh,x=q)
  %q_p.s = arith.constant dense<[1, 12, 8192, 2]> : tensor<4xi64>
  %q_p = tensor.reshape %q(%q_p.s) : (tensor<1x12x256x64xf16>, tensor<4xi64>) -> tensor<1x12x8192x2xf16>
  // q_e = slice_by_size(x=q_p,begin=rp_b0,size=rp_s1)
  %q_e = tensor.extract_slice %q_p[0, 0, 0, 0] [1, 12, 8192, 1] [1, 1, 1, 1] : tensor<1x12x8192x2xf16> to tensor<1x12x8192x1xf16>
  // q_o = slice_by_size(x=q_p,begin=rp_b1,size=rp_s1)
  %q_o = tensor.extract_slice %q_p[0, 0, 0, 1] [1, 12, 8192, 1] [1, 1, 1, 1] : tensor<1x12x8192x2xf16> to tensor<1x12x8192x1xf16>
  // nq = mul(x=q_o,y=neg1)
  %neg1 = arith.constant -1.0 : f16
  %nq.e = tensor.empty() : tensor<1x12x8192x1xf16>
  %nq = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q_o, %neg1 : tensor<1x12x8192x1xf16>, f16) outs(%nq.e : tensor<1x12x8192x1xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x8192x1xf16>
  // qrp = concat(axis=rpax,interleave=rpil,values=(nq,q_e))
  %qrp = tensor.concat dim(3) %nq, %q_e : (tensor<1x12x8192x1xf16>, tensor<1x12x8192x1xf16>) -> tensor<1x12x8192x2xf16>
  // q_rot = reshape(shape=rp_bk_q,x=qrp)
  %q_rot.s = arith.constant dense<[1, 12, 256, 64]> : tensor<4xi64>
  %q_rot = tensor.reshape %qrp(%q_rot.s) : (tensor<1x12x8192x2xf16>, tensor<4xi64>) -> tensor<1x12x256x64xf16>
  // qc = mul(x=q,y=rope_cos)
  %qc.e = tensor.empty() : tensor<1x12x256x64xf16>
  %qc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q, %rope_cos : tensor<1x12x256x64xf16>, tensor<1x1x256x64xf16>) outs(%qc.e : tensor<1x12x256x64xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x64xf16>
  // qrs = mul(x=q_rot,y=rope_sin)
  %qrs.e = tensor.empty() : tensor<1x12x256x64xf16>
  %qrs = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%q_rot, %rope_sin : tensor<1x12x256x64xf16>, tensor<1x1x256x64xf16>) outs(%qrs.e : tensor<1x12x256x64xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x64xf16>
  // q_rope = add(x=qc,y=qrs)
  %q_rope.e = tensor.empty() : tensor<1x12x256x64xf16>
  %q_rope = linalg.add ins(%qc, %qrs : tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) outs(%q_rope.e : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // k_p = reshape(shape=rp_sh_k,x=k)
  %k_p.s = arith.constant dense<[1, 12, 8192, 2]> : tensor<4xi64>
  %k_p = tensor.reshape %k(%k_p.s) : (tensor<1x12x256x64xf16>, tensor<4xi64>) -> tensor<1x12x8192x2xf16>
  // k_e = slice_by_size(x=k_p,begin=rp_b0,size=rp_s1_k)
  %k_e = tensor.extract_slice %k_p[0, 0, 0, 0] [1, 12, 8192, 1] [1, 1, 1, 1] : tensor<1x12x8192x2xf16> to tensor<1x12x8192x1xf16>
  // k_o = slice_by_size(x=k_p,begin=rp_b1,size=rp_s1_k)
  %k_o = tensor.extract_slice %k_p[0, 0, 0, 1] [1, 12, 8192, 1] [1, 1, 1, 1] : tensor<1x12x8192x2xf16> to tensor<1x12x8192x1xf16>
  // nk = mul(x=k_o,y=neg1)
  %nk.e = tensor.empty() : tensor<1x12x8192x1xf16>
  %nk = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%k_o, %neg1 : tensor<1x12x8192x1xf16>, f16) outs(%nk.e : tensor<1x12x8192x1xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x8192x1xf16>
  // krp = concat(axis=rpax,interleave=rpil,values=(nk,k_e))
  %krp = tensor.concat dim(3) %nk, %k_e : (tensor<1x12x8192x1xf16>, tensor<1x12x8192x1xf16>) -> tensor<1x12x8192x2xf16>
  // k_rot = reshape(shape=rp_bk_k,x=krp)
  %k_rot.s = arith.constant dense<[1, 12, 256, 64]> : tensor<4xi64>
  %k_rot = tensor.reshape %krp(%k_rot.s) : (tensor<1x12x8192x2xf16>, tensor<4xi64>) -> tensor<1x12x256x64xf16>
  // kc = mul(x=k,y=rope_cos)
  %kc.e = tensor.empty() : tensor<1x12x256x64xf16>
  %kc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%k, %rope_cos : tensor<1x12x256x64xf16>, tensor<1x1x256x64xf16>) outs(%kc.e : tensor<1x12x256x64xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x64xf16>
  // krs = mul(x=k_rot,y=rope_sin)
  %krs.e = tensor.empty() : tensor<1x12x256x64xf16>
  %krs = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%k_rot, %rope_sin : tensor<1x12x256x64xf16>, tensor<1x1x256x64xf16>) outs(%krs.e : tensor<1x12x256x64xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x64xf16>
  // k_rope = add(x=kc,y=krs)
  %k_rope.e = tensor.empty() : tensor<1x12x256x64xf16>
  %k_rope = linalg.add ins(%kc, %krs : tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) outs(%k_rope.e : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // k_tiled = concat(axis=cax,interleave=cid,values=(k_rope))
  %k_tiled = tensor.concat dim(1) %k_rope : (tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // v_tiled = concat(axis=cax,interleave=cid,values=(v))
  %v_tiled = tensor.concat dim(1) %v : (tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q_rope,y=k_tiled)
  %sc1.e = tensor.empty() : tensor<1x12x256x256xf16>
  %sc1.z = arith.constant 0.0 : f16
  %sc1.f = linalg.fill ins(%sc1.z : f16) outs(%sc1.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %sc1 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, j, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%q_rope, %k_tiled : tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) outs(%sc1.f : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // sc2 = mul(x=sc1,y=scv)
  %scv = arith.constant 0.125 : f16
  %sc2.e = tensor.empty() : tensor<1x12x256x256xf16>
  %sc2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc1, %scv : tensor<1x12x256x256xf16>, f16) outs(%sc2.e : tensor<1x12x256x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x256xf16>
  // ms = add(x=sc2,y=cm)
  %ms.e = tensor.empty() : tensor<1x12x256x256xf16>
  %ms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc2, %cm : tensor<1x12x256x256xf16>, tensor<1x1x256x256xf16>) outs(%ms.e : tensor<1x12x256x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x256xf16>
  // aw = softmax(axis=sax,x=ms)
  %aw.e = tensor.empty() : tensor<1x12x256x256xf16>
  %aw = linalg.softmax dimension(3) ins(%ms : tensor<1x12x256x256xf16>) outs(%aw.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // a4 = matmul(transpose_x=bF,transpose_y=bF,x=aw,y=v_tiled)
  %a4.e = tensor.empty() : tensor<1x12x256x64xf16>
  %a4.z = arith.constant 0.0 : f16
  %a4.f = linalg.fill ins(%a4.z : f16) outs(%a4.e : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  %a4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%aw, %v_tiled : tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) outs(%a4.f : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // at = transpose(perm=pm,x=a4)
  %at.e = tensor.empty() : tensor<1x12x64x256xf16>
  %at = linalg.transpose ins(%a4 : tensor<1x12x256x64xf16>) outs(%at.e : tensor<1x12x64x256xf16>) permutation = [0, 1, 3, 2]
  // af = reshape(shape=qsh,x=at)
  %af.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %af = tensor.reshape %at(%af.s) : (tensor<1x12x64x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // qrt = transpose(perm=pm,x=q_rope)
  %qrt.e = tensor.empty() : tensor<1x12x64x256xf16>
  %qrt = linalg.transpose ins(%q_rope : tensor<1x12x256x64xf16>) outs(%qrt.e : tensor<1x12x64x256xf16>) permutation = [0, 1, 3, 2]
  // qrf = reshape(shape=qsh,x=qrt)
  %qrf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %qrf = tensor.reshape %qrt(%qrf.s) : (tensor<1x12x64x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // krt = transpose(perm=pm,x=k_rope)
  %krt.e = tensor.empty() : tensor<1x12x64x256xf16>
  %krt = linalg.transpose ins(%k_rope : tensor<1x12x256x64xf16>) outs(%krt.e : tensor<1x12x64x256xf16>) permutation = [0, 1, 3, 2]
  // krf = reshape(shape=kvsh,x=krt)
  %krf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %krf = tensor.reshape %krt(%krf.s) : (tensor<1x12x64x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // out = concat(axis=cax,interleave=cid,values=(af,qrf,krf,vf,xn))
  %out = tensor.concat dim(1) %af, %qrf, %krf, %vf, %xn : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x3840x1x256xf16>
  return %out : tensor<1x3840x1x256xf16>
}

// MIL: ane_wo_fwd_dynamic
func.func @ane_wo_fwd_dynamic(%x: tensor<1x768x1x1024xf16>)
    -> (tensor<1x768x1x256xf16>) {
  // mm_act = slice_by_size(x=x,begin=mm_ba,size=mm_sa)
  %mm_act = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x1024xf16> to tensor<1x768x1x256xf16>
  // mm_wt = slice_by_size(x=x,begin=mm_bw,size=mm_sw)
  %mm_wt = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x1024xf16> to tensor<1x768x1x768xf16>
  // mm_a2 = reshape(shape=mm_ra,x=mm_act)
  %mm_a2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %mm_a2 = tensor.reshape %mm_act(%mm_a2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // mm_a3 = transpose(perm=mm_pm,x=mm_a2)
  %mm_a3.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_a3 = linalg.transpose ins(%mm_a2 : tensor<1x1x768x256xf16>) outs(%mm_a3.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // mm_W = reshape(shape=mm_rw,x=mm_wt)
  %mm_W.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %mm_W = tensor.reshape %mm_wt(%mm_W.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // mm_yh = matmul(transpose_x=bF,transpose_y=bF,x=mm_a3,y=mm_W)
  %mm_yh.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_yh.z = arith.constant 0.0 : f16
  %mm_yh.f = linalg.fill ins(%mm_yh.z : f16) outs(%mm_yh.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %mm_yh = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%mm_a3, %mm_W : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%mm_yh.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // mm_yt = transpose(perm=mm_pm,x=mm_yh)
  %mm_yt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %mm_yt = linalg.transpose ins(%mm_yh : tensor<1x1x256x768xf16>) outs(%mm_yt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // mm_y = reshape(shape=mm_ro,x=mm_yt)
  %mm_y.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %mm_y = tensor.reshape %mm_yt(%mm_y.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  return %mm_y : tensor<1x768x1x256xf16>
}

// MIL: ane_ffn_fused_dynamic
func.func @ane_ffn_fused_dynamic(%x: tensor<1x768x1x6656xf16>)
    -> (tensor<1x6912x1x256xf16>) {
  // x2norm = slice_by_size(x=x,begin=b_xn,size=s_ds)
  %x2norm = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x6656xf16> to tensor<1x768x1x256xf16>
  // x2 = slice_by_size(x=x,begin=b_x2,size=s_ds)
  %x2 = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x6656xf16> to tensor<1x768x1x256xf16>
  // W1 = slice_by_size(x=x,begin=b_w1,size=s_wh)
  %W1 = tensor.extract_slice %x[0, 0, 0, 512] [1, 768, 1, 2048] [1, 1, 1, 1] : tensor<1x768x1x6656xf16> to tensor<1x768x1x2048xf16>
  // W3 = slice_by_size(x=x,begin=b_w3,size=s_wh)
  %W3 = tensor.extract_slice %x[0, 0, 0, 2560] [1, 768, 1, 2048] [1, 1, 1, 1] : tensor<1x768x1x6656xf16> to tensor<1x768x1x2048xf16>
  // W2r = slice_by_size(x=x,begin=b_w2,size=s_wh)
  %W2r = tensor.extract_slice %x[0, 0, 0, 4608] [1, 768, 1, 2048] [1, 1, 1, 1] : tensor<1x768x1x6656xf16> to tensor<1x768x1x2048xf16>
  // xn2 = reshape(shape=rd,x=x2norm)
  %xn2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %xn2 = tensor.reshape %x2norm(%xn2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // xnt = transpose(perm=pm,x=xn2)
  %xnt.e = tensor.empty() : tensor<1x1x256x768xf16>
  %xnt = linalg.transpose ins(%xn2 : tensor<1x1x768x256xf16>) outs(%xnt.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // W12 = reshape(shape=rw,x=W1)
  %W12.s = arith.constant dense<[1, 1, 768, 2048]> : tensor<4xi64>
  %W12 = tensor.reshape %W1(%W12.s) : (tensor<1x768x1x2048xf16>, tensor<4xi64>) -> tensor<1x1x768x2048xf16>
  // W32 = reshape(shape=rw,x=W3)
  %W32.s = arith.constant dense<[1, 1, 768, 2048]> : tensor<4xi64>
  %W32 = tensor.reshape %W3(%W32.s) : (tensor<1x768x1x2048xf16>, tensor<4xi64>) -> tensor<1x1x768x2048xf16>
  // h1m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W12)
  %h1m.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %h1m.z = arith.constant 0.0 : f16
  %h1m.f = linalg.fill ins(%h1m.z : f16) outs(%h1m.e : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  %h1m = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%xnt, %W12 : tensor<1x1x256x768xf16>, tensor<1x1x768x2048xf16>) outs(%h1m.f : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  // h3m = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=W32)
  %h3m.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %h3m.z = arith.constant 0.0 : f16
  %h3m.f = linalg.fill ins(%h3m.z : f16) outs(%h3m.e : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  %h3m = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%xnt, %W32 : tensor<1x1x256x768xf16>, tensor<1x1x768x2048xf16>) outs(%h3m.f : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  // h1t = transpose(perm=pm,x=h1m)
  %h1t.e = tensor.empty() : tensor<1x1x2048x256xf16>
  %h1t = linalg.transpose ins(%h1m : tensor<1x1x256x2048xf16>) outs(%h1t.e : tensor<1x1x2048x256xf16>) permutation = [0, 1, 3, 2]
  // h3t = transpose(perm=pm,x=h3m)
  %h3t.e = tensor.empty() : tensor<1x1x2048x256xf16>
  %h3t = linalg.transpose ins(%h3m : tensor<1x1x256x2048xf16>) outs(%h3t.e : tensor<1x1x2048x256xf16>) permutation = [0, 1, 3, 2]
  // h1 = reshape(shape=rh,x=h1t)
  %h1.s = arith.constant dense<[1, 2048, 1, 256]> : tensor<4xi64>
  %h1 = tensor.reshape %h1t(%h1.s) : (tensor<1x1x2048x256xf16>, tensor<4xi64>) -> tensor<1x2048x1x256xf16>
  // h3 = reshape(shape=rh,x=h3t)
  %h3.s = arith.constant dense<[1, 2048, 1, 256]> : tensor<4xi64>
  %h3 = tensor.reshape %h3t(%h3.s) : (tensor<1x1x2048x256xf16>, tensor<4xi64>) -> tensor<1x2048x1x256xf16>
  // sig = sigmoid(x=h1)
  %sig.e = tensor.empty() : tensor<1x2048x1x256xf16>
  %sig = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%h1 : tensor<1x2048x1x256xf16>) outs(%sig.e : tensor<1x2048x1x256xf16>) {
    ^bb0(%_a0 : f16, %_o : f16):
      %_one = arith.constant 1.0 : f16
      %_n = arith.negf %_a0 : f16
      %_ex = math.exp %_n : f16
      %_den = arith.addf %_one, %_ex : f16
      %_r = arith.divf %_one, %_den : f16
      linalg.yield %_r : f16
    } -> tensor<1x2048x1x256xf16>
  // silu = mul(x=h1,y=sig)
  %silu.e = tensor.empty() : tensor<1x2048x1x256xf16>
  %silu = linalg.mul ins(%h1, %sig : tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) outs(%silu.e : tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
  // gate = mul(x=silu,y=h3)
  %gate.e = tensor.empty() : tensor<1x2048x1x256xf16>
  %gate = linalg.mul ins(%silu, %h3 : tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) outs(%gate.e : tensor<1x2048x1x256xf16>) -> tensor<1x2048x1x256xf16>
  // g2 = reshape(shape=rg,x=gate)
  %g2.s = arith.constant dense<[1, 1, 2048, 256]> : tensor<4xi64>
  %g2 = tensor.reshape %gate(%g2.s) : (tensor<1x2048x1x256xf16>, tensor<4xi64>) -> tensor<1x1x2048x256xf16>
  // gt = transpose(perm=pm,x=g2)
  %gt.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %gt = linalg.transpose ins(%g2 : tensor<1x1x2048x256xf16>) outs(%gt.e : tensor<1x1x256x2048xf16>) permutation = [0, 1, 3, 2]
  // W22 = reshape(shape=rw,x=W2r)
  %W22.s = arith.constant dense<[1, 1, 768, 2048]> : tensor<4xi64>
  %W22 = tensor.reshape %W2r(%W22.s) : (tensor<1x768x1x2048xf16>, tensor<4xi64>) -> tensor<1x1x768x2048xf16>
  // W2t = transpose(perm=pm,x=W22)
  %W2t.e = tensor.empty() : tensor<1x1x2048x768xf16>
  %W2t = linalg.transpose ins(%W22 : tensor<1x1x768x2048xf16>) outs(%W2t.e : tensor<1x1x2048x768xf16>) permutation = [0, 1, 3, 2]
  // fm = matmul(transpose_x=bF,transpose_y=bF,x=gt,y=W2t)
  %fm.e = tensor.empty() : tensor<1x1x256x768xf16>
  %fm.z = arith.constant 0.0 : f16
  %fm.f = linalg.fill ins(%fm.z : f16) outs(%fm.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %fm = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%gt, %W2t : tensor<1x1x256x2048xf16>, tensor<1x1x2048x768xf16>) outs(%fm.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // ft = transpose(perm=pm,x=fm)
  %ft.e = tensor.empty() : tensor<1x1x768x256xf16>
  %ft = linalg.transpose ins(%fm : tensor<1x1x256x768xf16>) outs(%ft.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // ffn_out = reshape(shape=rd2,x=ft)
  %ffn_out.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %ffn_out = tensor.reshape %ft(%ffn_out.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // ffn_scaled = mul(x=ffn_out,y=res_alpha)
  %res_alpha = arith.constant 0.204124 : f16
  %ffn_scaled.e = tensor.empty() : tensor<1x768x1x256xf16>
  %ffn_scaled = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ffn_out, %res_alpha : tensor<1x768x1x256xf16>, f16) outs(%ffn_scaled.e : tensor<1x768x1x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x768x1x256xf16>
  // x_next = add(x=x2,y=ffn_scaled)
  %x_next.e = tensor.empty() : tensor<1x768x1x256xf16>
  %x_next = linalg.add ins(%x2, %ffn_scaled : tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) outs(%x_next.e : tensor<1x768x1x256xf16>) -> tensor<1x768x1x256xf16>
  // out = concat(axis=cax,interleave=cid,values=(x_next,h1,h3,gate))
  %out = tensor.concat dim(1) %x_next, %h1, %h3, %gate : (tensor<1x768x1x256xf16>, tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>, tensor<1x2048x1x256xf16>) -> tensor<1x6912x1x256xf16>
  return %out : tensor<1x6912x1x256xf16>
}

// MIL: ane_ffn_bwd_w2t_dynamic
func.func @ane_ffn_bwd_w2t_dynamic(%x: tensor<1x768x1x2304xf16>)
    -> (tensor<1x2048x1x256xf16>) {
  // mm_act = slice_by_size(x=x,begin=mm_ba,size=mm_sa)
  %mm_act = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x2304xf16> to tensor<1x768x1x256xf16>
  // mm_wt = slice_by_size(x=x,begin=mm_bw,size=mm_sw)
  %mm_wt = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 2048] [1, 1, 1, 1] : tensor<1x768x1x2304xf16> to tensor<1x768x1x2048xf16>
  // mm_a2 = reshape(shape=mm_ra,x=mm_act)
  %mm_a2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %mm_a2 = tensor.reshape %mm_act(%mm_a2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // mm_a3 = transpose(perm=mm_pm,x=mm_a2)
  %mm_a3.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_a3 = linalg.transpose ins(%mm_a2 : tensor<1x1x768x256xf16>) outs(%mm_a3.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // mm_W = reshape(shape=mm_rw,x=mm_wt)
  %mm_W.s = arith.constant dense<[1, 1, 768, 2048]> : tensor<4xi64>
  %mm_W = tensor.reshape %mm_wt(%mm_W.s) : (tensor<1x768x1x2048xf16>, tensor<4xi64>) -> tensor<1x1x768x2048xf16>
  // mm_yh = matmul(transpose_x=bF,transpose_y=bF,x=mm_a3,y=mm_W)
  %mm_yh.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %mm_yh.z = arith.constant 0.0 : f16
  %mm_yh.f = linalg.fill ins(%mm_yh.z : f16) outs(%mm_yh.e : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  %mm_yh = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%mm_a3, %mm_W : tensor<1x1x256x768xf16>, tensor<1x1x768x2048xf16>) outs(%mm_yh.f : tensor<1x1x256x2048xf16>) -> tensor<1x1x256x2048xf16>
  // mm_yt = transpose(perm=mm_pm,x=mm_yh)
  %mm_yt.e = tensor.empty() : tensor<1x1x2048x256xf16>
  %mm_yt = linalg.transpose ins(%mm_yh : tensor<1x1x256x2048xf16>) outs(%mm_yt.e : tensor<1x1x2048x256xf16>) permutation = [0, 1, 3, 2]
  // mm_y = reshape(shape=mm_ro,x=mm_yt)
  %mm_y.s = arith.constant dense<[1, 2048, 1, 256]> : tensor<4xi64>
  %mm_y = tensor.reshape %mm_yt(%mm_y.s) : (tensor<1x1x2048x256xf16>, tensor<4xi64>) -> tensor<1x2048x1x256xf16>
  return %mm_y : tensor<1x2048x1x256xf16>
}

// MIL: ane_ffn_bwd_w13t_dynamic
func.func @ane_ffn_bwd_w13t_dynamic(%x: tensor<1x2048x1x2048xf16>)
    -> (tensor<1x768x1x256xf16>) {
  // dh1 = slice_by_size(x=x,begin=b0,size=sh)
  %dh1 = tensor.extract_slice %x[0, 0, 0, 0] [1, 2048, 1, 256] [1, 1, 1, 1] : tensor<1x2048x1x2048xf16> to tensor<1x2048x1x256xf16>
  // dh3 = slice_by_size(x=x,begin=b1,size=sh)
  %dh3 = tensor.extract_slice %x[0, 0, 0, 256] [1, 2048, 1, 256] [1, 1, 1, 1] : tensor<1x2048x1x2048xf16> to tensor<1x2048x1x256xf16>
  // W1t = slice_by_size(x=x,begin=b2,size=sw)
  %W1t = tensor.extract_slice %x[0, 0, 0, 512] [1, 2048, 1, 768] [1, 1, 1, 1] : tensor<1x2048x1x2048xf16> to tensor<1x2048x1x768xf16>
  // W3t = slice_by_size(x=x,begin=b3,size=sw)
  %W3t = tensor.extract_slice %x[0, 0, 0, 1280] [1, 2048, 1, 768] [1, 1, 1, 1] : tensor<1x2048x1x2048xf16> to tensor<1x2048x1x768xf16>
  // dh12 = reshape(shape=ra,x=dh1)
  %dh12.s = arith.constant dense<[1, 1, 2048, 256]> : tensor<4xi64>
  %dh12 = tensor.reshape %dh1(%dh12.s) : (tensor<1x2048x1x256xf16>, tensor<4xi64>) -> tensor<1x1x2048x256xf16>
  // dh1t = transpose(perm=pm,x=dh12)
  %dh1t.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %dh1t = linalg.transpose ins(%dh12 : tensor<1x1x2048x256xf16>) outs(%dh1t.e : tensor<1x1x256x2048xf16>) permutation = [0, 1, 3, 2]
  // dh32 = reshape(shape=ra,x=dh3)
  %dh32.s = arith.constant dense<[1, 1, 2048, 256]> : tensor<4xi64>
  %dh32 = tensor.reshape %dh3(%dh32.s) : (tensor<1x2048x1x256xf16>, tensor<4xi64>) -> tensor<1x1x2048x256xf16>
  // dh3t = transpose(perm=pm,x=dh32)
  %dh3t.e = tensor.empty() : tensor<1x1x256x2048xf16>
  %dh3t = linalg.transpose ins(%dh32 : tensor<1x1x2048x256xf16>) outs(%dh3t.e : tensor<1x1x256x2048xf16>) permutation = [0, 1, 3, 2]
  // W1t2 = reshape(shape=rw,x=W1t)
  %W1t2.s = arith.constant dense<[1, 1, 2048, 768]> : tensor<4xi64>
  %W1t2 = tensor.reshape %W1t(%W1t2.s) : (tensor<1x2048x1x768xf16>, tensor<4xi64>) -> tensor<1x1x2048x768xf16>
  // W3t2 = reshape(shape=rw,x=W3t)
  %W3t2.s = arith.constant dense<[1, 1, 2048, 768]> : tensor<4xi64>
  %W3t2 = tensor.reshape %W3t(%W3t2.s) : (tensor<1x2048x1x768xf16>, tensor<4xi64>) -> tensor<1x1x2048x768xf16>
  // dx1m = matmul(transpose_x=bF,transpose_y=bF,x=dh1t,y=W1t2)
  %dx1m.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dx1m.z = arith.constant 0.0 : f16
  %dx1m.f = linalg.fill ins(%dx1m.z : f16) outs(%dx1m.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %dx1m = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%dh1t, %W1t2 : tensor<1x1x256x2048xf16>, tensor<1x1x2048x768xf16>) outs(%dx1m.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // dx3m = matmul(transpose_x=bF,transpose_y=bF,x=dh3t,y=W3t2)
  %dx3m.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dx3m.z = arith.constant 0.0 : f16
  %dx3m.f = linalg.fill ins(%dx3m.z : f16) outs(%dx3m.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %dx3m = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%dh3t, %W3t2 : tensor<1x1x256x2048xf16>, tensor<1x1x2048x768xf16>) outs(%dx3m.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // dxm = add(x=dx1m,y=dx3m)
  %dxm.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dxm = linalg.add ins(%dx1m, %dx3m : tensor<1x1x256x768xf16>, tensor<1x1x256x768xf16>) outs(%dxm.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // dxt = transpose(perm=pm,x=dxm)
  %dxt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %dxt = linalg.transpose ins(%dxm : tensor<1x1x256x768xf16>) outs(%dxt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // dx = reshape(shape=ro,x=dxt)
  %dx.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %dx = tensor.reshape %dxt(%dx.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  return %dx : tensor<1x768x1x256xf16>
}

// MIL: ane_wot_dynamic
func.func @ane_wot_dynamic(%x: tensor<1x768x1x1024xf16>)
    -> (tensor<1x768x1x256xf16>) {
  // mm_act = slice_by_size(x=x,begin=mm_ba,size=mm_sa)
  %mm_act = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x1024xf16> to tensor<1x768x1x256xf16>
  // mm_wt = slice_by_size(x=x,begin=mm_bw,size=mm_sw)
  %mm_wt = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x1024xf16> to tensor<1x768x1x768xf16>
  // mm_a2 = reshape(shape=mm_ra,x=mm_act)
  %mm_a2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %mm_a2 = tensor.reshape %mm_act(%mm_a2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // mm_a3 = transpose(perm=mm_pm,x=mm_a2)
  %mm_a3.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_a3 = linalg.transpose ins(%mm_a2 : tensor<1x1x768x256xf16>) outs(%mm_a3.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // mm_W = reshape(shape=mm_rw,x=mm_wt)
  %mm_W.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %mm_W = tensor.reshape %mm_wt(%mm_W.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // mm_yh = matmul(transpose_x=bF,transpose_y=bF,x=mm_a3,y=mm_W)
  %mm_yh.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_yh.z = arith.constant 0.0 : f16
  %mm_yh.f = linalg.fill ins(%mm_yh.z : f16) outs(%mm_yh.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %mm_yh = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%mm_a3, %mm_W : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%mm_yh.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // mm_yt = transpose(perm=mm_pm,x=mm_yh)
  %mm_yt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %mm_yt = linalg.transpose ins(%mm_yh : tensor<1x1x256x768xf16>) outs(%mm_yt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // mm_y = reshape(shape=mm_ro,x=mm_yt)
  %mm_y.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %mm_y = tensor.reshape %mm_yt(%mm_y.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  return %mm_y : tensor<1x768x1x256xf16>
}

// MIL: ane_q_bwd_dynamic
func.func @ane_q_bwd_dynamic(%x: tensor<1x768x1x1024xf16>)
    -> (tensor<1x768x1x256xf16>) {
  // mm_act = slice_by_size(x=x,begin=mm_ba,size=mm_sa)
  %mm_act = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x1024xf16> to tensor<1x768x1x256xf16>
  // mm_wt = slice_by_size(x=x,begin=mm_bw,size=mm_sw)
  %mm_wt = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x1024xf16> to tensor<1x768x1x768xf16>
  // mm_a2 = reshape(shape=mm_ra,x=mm_act)
  %mm_a2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %mm_a2 = tensor.reshape %mm_act(%mm_a2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // mm_a3 = transpose(perm=mm_pm,x=mm_a2)
  %mm_a3.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_a3 = linalg.transpose ins(%mm_a2 : tensor<1x1x768x256xf16>) outs(%mm_a3.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // mm_W = reshape(shape=mm_rw,x=mm_wt)
  %mm_W.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %mm_W = tensor.reshape %mm_wt(%mm_W.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // mm_yh = matmul(transpose_x=bF,transpose_y=bF,x=mm_a3,y=mm_W)
  %mm_yh.e = tensor.empty() : tensor<1x1x256x768xf16>
  %mm_yh.z = arith.constant 0.0 : f16
  %mm_yh.f = linalg.fill ins(%mm_yh.z : f16) outs(%mm_yh.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %mm_yh = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%mm_a3, %mm_W : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%mm_yh.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // mm_yt = transpose(perm=mm_pm,x=mm_yh)
  %mm_yt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %mm_yt = linalg.transpose ins(%mm_yh : tensor<1x1x256x768xf16>) outs(%mm_yt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // mm_y = reshape(shape=mm_ro,x=mm_yt)
  %mm_y.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %mm_y = tensor.reshape %mm_yt(%mm_y.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  return %mm_y : tensor<1x768x1x256xf16>
}

// MIL: ane_kv_bwd_dynamic
func.func @ane_kv_bwd_dynamic(%x: tensor<1x768x1x2048xf16>)
    -> (tensor<1x768x1x256xf16>) {
  // dk = slice_by_size(x=x,begin=b0,size=sh)
  %dk = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x2048xf16> to tensor<1x768x1x256xf16>
  // dv = slice_by_size(x=x,begin=b1,size=sh)
  %dv = tensor.extract_slice %x[0, 0, 0, 256] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x768x1x2048xf16> to tensor<1x768x1x256xf16>
  // Wkt = slice_by_size(x=x,begin=b2,size=sw)
  %Wkt = tensor.extract_slice %x[0, 0, 0, 512] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x2048xf16> to tensor<1x768x1x768xf16>
  // Wvt = slice_by_size(x=x,begin=b3,size=sw)
  %Wvt = tensor.extract_slice %x[0, 0, 0, 1280] [1, 768, 1, 768] [1, 1, 1, 1] : tensor<1x768x1x2048xf16> to tensor<1x768x1x768xf16>
  // dk2 = reshape(shape=ra,x=dk)
  %dk2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %dk2 = tensor.reshape %dk(%dk2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // dkt = transpose(perm=pm,x=dk2)
  %dkt.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dkt = linalg.transpose ins(%dk2 : tensor<1x1x768x256xf16>) outs(%dkt.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // dv2 = reshape(shape=ra,x=dv)
  %dv2.s = arith.constant dense<[1, 1, 768, 256]> : tensor<4xi64>
  %dv2 = tensor.reshape %dv(%dv2.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x1x768x256xf16>
  // dvt = transpose(perm=pm,x=dv2)
  %dvt.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dvt = linalg.transpose ins(%dv2 : tensor<1x1x768x256xf16>) outs(%dvt.e : tensor<1x1x256x768xf16>) permutation = [0, 1, 3, 2]
  // Wkt2 = reshape(shape=rw,x=Wkt)
  %Wkt2.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %Wkt2 = tensor.reshape %Wkt(%Wkt2.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // Wvt2 = reshape(shape=rw,x=Wvt)
  %Wvt2.s = arith.constant dense<[1, 1, 768, 768]> : tensor<4xi64>
  %Wvt2 = tensor.reshape %Wvt(%Wvt2.s) : (tensor<1x768x1x768xf16>, tensor<4xi64>) -> tensor<1x1x768x768xf16>
  // dxk = matmul(transpose_x=bF,transpose_y=bF,x=dkt,y=Wkt2)
  %dxk.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dxk.z = arith.constant 0.0 : f16
  %dxk.f = linalg.fill ins(%dxk.z : f16) outs(%dxk.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %dxk = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%dkt, %Wkt2 : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%dxk.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // dxv = matmul(transpose_x=bF,transpose_y=bF,x=dvt,y=Wvt2)
  %dxv.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dxv.z = arith.constant 0.0 : f16
  %dxv.f = linalg.fill ins(%dxv.z : f16) outs(%dxv.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  %dxv = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%dvt, %Wvt2 : tensor<1x1x256x768xf16>, tensor<1x1x768x768xf16>) outs(%dxv.f : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // dxm = add(x=dxk,y=dxv)
  %dxm.e = tensor.empty() : tensor<1x1x256x768xf16>
  %dxm = linalg.add ins(%dxk, %dxv : tensor<1x1x256x768xf16>, tensor<1x1x256x768xf16>) outs(%dxm.e : tensor<1x1x256x768xf16>) -> tensor<1x1x256x768xf16>
  // dxt = transpose(perm=pm,x=dxm)
  %dxt.e = tensor.empty() : tensor<1x1x768x256xf16>
  %dxt = linalg.transpose ins(%dxm : tensor<1x1x256x768xf16>) outs(%dxt.e : tensor<1x1x768x256xf16>) permutation = [0, 1, 3, 2]
  // dx = reshape(shape=ro,x=dxt)
  %dx.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %dx = tensor.reshape %dxt(%dx.s) : (tensor<1x1x768x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  return %dx : tensor<1x768x1x256xf16>
}

// MIL: ane_sdpa_bwd1_noweight
//   %cm: BLOBFILE @model_path/weights/mask.bin @ offset 64
func.func @ane_sdpa_bwd1_noweight(%x: tensor<1x3072x1x256xf16>, %cm: tensor<1x1x256x256xf16>)
    -> (tensor<1x6912x1x256xf16>) {
  // qf = slice_by_size(x=x,begin=b0,size=sz)
  %qf = tensor.extract_slice %x[0, 0, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x3072x1x256xf16> to tensor<1x768x1x256xf16>
  // kf = slice_by_size(x=x,begin=b1,size=sz)
  %kf = tensor.extract_slice %x[0, 768, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x3072x1x256xf16> to tensor<1x768x1x256xf16>
  // vf = slice_by_size(x=x,begin=b2,size=sz)
  %vf = tensor.extract_slice %x[0, 1536, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x3072x1x256xf16> to tensor<1x768x1x256xf16>
  // da = slice_by_size(x=x,begin=b3,size=sz)
  %da = tensor.extract_slice %x[0, 2304, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x3072x1x256xf16> to tensor<1x768x1x256xf16>
  // qr = reshape(shape=rsh,x=qf)
  %qr.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %qr = tensor.reshape %qf(%qr.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // q = transpose(perm=pm,x=qr)
  %q.e = tensor.empty() : tensor<1x12x256x64xf16>
  %q = linalg.transpose ins(%qr : tensor<1x12x64x256xf16>) outs(%q.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // kr = reshape(shape=rsh,x=kf)
  %kr.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %kr = tensor.reshape %kf(%kr.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // k = transpose(perm=pm,x=kr)
  %k.e = tensor.empty() : tensor<1x12x256x64xf16>
  %k = linalg.transpose ins(%kr : tensor<1x12x64x256xf16>) outs(%k.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // vr = reshape(shape=rsh,x=vf)
  %vr.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %vr = tensor.reshape %vf(%vr.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // v = transpose(perm=pm,x=vr)
  %v.e = tensor.empty() : tensor<1x12x256x64xf16>
  %v = linalg.transpose ins(%vr : tensor<1x12x64x256xf16>) outs(%v.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // dr = reshape(shape=rsh,x=da)
  %dr.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %dr = tensor.reshape %da(%dr.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // dat = transpose(perm=pm,x=dr)
  %dat.e = tensor.empty() : tensor<1x12x256x64xf16>
  %dat = linalg.transpose ins(%dr : tensor<1x12x64x256xf16>) outs(%dat.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)
  %sc1.e = tensor.empty() : tensor<1x12x256x256xf16>
  %sc1.z = arith.constant 0.0 : f16
  %sc1.f = linalg.fill ins(%sc1.z : f16) outs(%sc1.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %sc1 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, j, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%q, %k : tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) outs(%sc1.f : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // sc2 = mul(x=sc1,y=scv)
  %scv = arith.constant 0.125 : f16
  %sc2.e = tensor.empty() : tensor<1x12x256x256xf16>
  %sc2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc1, %scv : tensor<1x12x256x256xf16>, f16) outs(%sc2.e : tensor<1x12x256x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x256xf16>
  // ms = add(x=sc2,y=cm)
  %ms.e = tensor.empty() : tensor<1x12x256x256xf16>
  %ms = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%sc2, %cm : tensor<1x12x256x256xf16>, tensor<1x1x256x256xf16>) outs(%ms.e : tensor<1x12x256x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.addf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x256xf16>
  // probs = softmax(axis=sax,x=ms)
  %probs.e = tensor.empty() : tensor<1x12x256x256xf16>
  %probs = linalg.softmax dimension(3) ins(%ms : tensor<1x12x256x256xf16>) outs(%probs.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=dat)
  %dv4.e = tensor.empty() : tensor<1x12x256x64xf16>
  %dv4.z = arith.constant 0.0 : f16
  %dv4.f = linalg.fill ins(%dv4.z : f16) outs(%dv4.e : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  %dv4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, k, i)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%probs, %dat : tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) outs(%dv4.f : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // dp4 = matmul(transpose_x=bF,transpose_y=bT,x=dat,y=v)
  %dp4.e = tensor.empty() : tensor<1x12x256x256xf16>
  %dp4.z = arith.constant 0.0 : f16
  %dp4.f = linalg.fill ins(%dp4.z : f16) outs(%dp4.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %dp4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, j, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%dat, %v : tensor<1x12x256x64xf16>, tensor<1x12x256x64xf16>) outs(%dp4.f : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // dvt = transpose(perm=pm,x=dv4)
  %dvt.e = tensor.empty() : tensor<1x12x64x256xf16>
  %dvt = linalg.transpose ins(%dv4 : tensor<1x12x256x64xf16>) outs(%dvt.e : tensor<1x12x64x256xf16>) permutation = [0, 1, 3, 2]
  // dvf = reshape(shape=dvs,x=dvt)
  %dvf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %dvf = tensor.reshape %dvt(%dvf.s) : (tensor<1x12x64x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // pf = reshape(shape=scs,x=probs)
  %pf.s = arith.constant dense<[1, 3072, 1, 256]> : tensor<4xi64>
  %pf = tensor.reshape %probs(%pf.s) : (tensor<1x12x256x256xf16>, tensor<4xi64>) -> tensor<1x3072x1x256xf16>
  // dpf = reshape(shape=scs,x=dp4)
  %dpf.s = arith.constant dense<[1, 3072, 1, 256]> : tensor<4xi64>
  %dpf = tensor.reshape %dp4(%dpf.s) : (tensor<1x12x256x256xf16>, tensor<4xi64>) -> tensor<1x3072x1x256xf16>
  // out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))
  %out = tensor.concat dim(1) %dvf, %pf, %dpf : (tensor<1x768x1x256xf16>, tensor<1x3072x1x256xf16>, tensor<1x3072x1x256xf16>) -> tensor<1x6912x1x256xf16>
  return %out : tensor<1x6912x1x256xf16>
}

// MIL: ane_sdpa_bwd2
func.func @ane_sdpa_bwd2(%x: tensor<1x7680x1x256xf16>)
    -> (tensor<1x1536x1x256xf16>) {
  // pf = slice_by_size(x=x,begin=b0,size=sz_sc)
  %pf = tensor.extract_slice %x[0, 0, 0, 0] [1, 3072, 1, 256] [1, 1, 1, 1] : tensor<1x7680x1x256xf16> to tensor<1x3072x1x256xf16>
  // dpf = slice_by_size(x=x,begin=b1,size=sz_sc)
  %dpf = tensor.extract_slice %x[0, 3072, 0, 0] [1, 3072, 1, 256] [1, 1, 1, 1] : tensor<1x7680x1x256xf16> to tensor<1x3072x1x256xf16>
  // qf = slice_by_size(x=x,begin=b2,size=sz_q)
  %qf = tensor.extract_slice %x[0, 6144, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x7680x1x256xf16> to tensor<1x768x1x256xf16>
  // kf = slice_by_size(x=x,begin=b3,size=sz_q)
  %kf = tensor.extract_slice %x[0, 6912, 0, 0] [1, 768, 1, 256] [1, 1, 1, 1] : tensor<1x7680x1x256xf16> to tensor<1x768x1x256xf16>
  // probs = reshape(shape=ssh,x=pf)
  %probs.s = arith.constant dense<[1, 12, 256, 256]> : tensor<4xi64>
  %probs = tensor.reshape %pf(%probs.s) : (tensor<1x3072x1x256xf16>, tensor<4xi64>) -> tensor<1x12x256x256xf16>
  // dp = reshape(shape=ssh,x=dpf)
  %dp.s = arith.constant dense<[1, 12, 256, 256]> : tensor<4xi64>
  %dp = tensor.reshape %dpf(%dp.s) : (tensor<1x3072x1x256xf16>, tensor<4xi64>) -> tensor<1x12x256x256xf16>
  // qr = reshape(shape=rsh,x=qf)
  %qr.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %qr = tensor.reshape %qf(%qr.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // q = transpose(perm=pm,x=qr)
  %q.e = tensor.empty() : tensor<1x12x256x64xf16>
  %q = linalg.transpose ins(%qr : tensor<1x12x64x256xf16>) outs(%q.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // kr = reshape(shape=rsh,x=kf)
  %kr.s = arith.constant dense<[1, 12, 64, 256]> : tensor<4xi64>
  %kr = tensor.reshape %kf(%kr.s) : (tensor<1x768x1x256xf16>, tensor<4xi64>) -> tensor<1x12x64x256xf16>
  // k = transpose(perm=pm,x=kr)
  %k.e = tensor.empty() : tensor<1x12x256x64xf16>
  %k = linalg.transpose ins(%kr : tensor<1x12x64x256xf16>) outs(%k.e : tensor<1x12x256x64xf16>) permutation = [0, 1, 3, 2]
  // pdp = mul(x=probs,y=dp)
  %pdp.e = tensor.empty() : tensor<1x12x256x256xf16>
  %pdp = linalg.mul ins(%probs, %dp : tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) outs(%pdp.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)
  %spdp.e = tensor.empty() : tensor<1x12x256xf16>
  %spdp.z = arith.constant 0.0 : f16
  %spdp.f = linalg.fill ins(%spdp.z : f16) outs(%spdp.e : tensor<1x12x256xf16>) -> tensor<1x12x256xf16>
  %spdp.r = linalg.reduce { arith.addf } ins(%pdp : tensor<1x12x256x256xf16>) outs(%spdp.f : tensor<1x12x256xf16>) dimensions = [3]
  %spdp = tensor.expand_shape %spdp.r [[0], [1], [2, 3]] output_shape [1, 12, 256, 1] : tensor<1x12x256xf16> into tensor<1x12x256x1xf16>
  // dps = sub(x=dp,y=spdp)
  %dps.e = tensor.empty() : tensor<1x12x256x256xf16>
  %dps = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%dp, %spdp : tensor<1x12x256x256xf16>, tensor<1x12x256x1xf16>) outs(%dps.e : tensor<1x12x256x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.subf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x256xf16>
  // ds0 = mul(x=probs,y=dps)
  %ds0.e = tensor.empty() : tensor<1x12x256x256xf16>
  %ds0 = linalg.mul ins(%probs, %dps : tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) outs(%ds0.e : tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  // ds = mul(x=ds0,y=scv)
  %scv = arith.constant 0.125 : f16
  %ds.e = tensor.empty() : tensor<1x12x256x256xf16>
  %ds = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%ds0, %scv : tensor<1x12x256x256xf16>, f16) outs(%ds.e : tensor<1x12x256x256xf16>) {
    ^bb0(%_a0 : f16, %_a1 : f16, %_o : f16):
      %_r = arith.mulf %_a0, %_a1 : f16
      linalg.yield %_r : f16
    } -> tensor<1x12x256x256xf16>
  // dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)
  %dq4.e = tensor.empty() : tensor<1x12x256x64xf16>
  %dq4.z = arith.constant 0.0 : f16
  %dq4.f = linalg.fill ins(%dq4.z : f16) outs(%dq4.e : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  %dq4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, i, k)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%ds, %k : tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) outs(%dq4.f : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)
  %dk4.e = tensor.empty() : tensor<1x12x256x64xf16>
  %dk4.z = arith.constant 0.0 : f16
  %dk4.f = linalg.fill ins(%dk4.z : f16) outs(%dk4.e : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  %dk4 = linalg.contract indexing_maps = [affine_map<(b0, b1, i, j, k) -> (b0, b1, k, i)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, k, j)>, affine_map<(b0, b1, i, j, k) -> (b0, b1, i, j)>]
      ins(%ds, %q : tensor<1x12x256x256xf16>, tensor<1x12x256x64xf16>) outs(%dk4.f : tensor<1x12x256x64xf16>) -> tensor<1x12x256x64xf16>
  // dqt = transpose(perm=pm,x=dq4)
  %dqt.e = tensor.empty() : tensor<1x12x64x256xf16>
  %dqt = linalg.transpose ins(%dq4 : tensor<1x12x256x64xf16>) outs(%dqt.e : tensor<1x12x64x256xf16>) permutation = [0, 1, 3, 2]
  // dkt = transpose(perm=pm,x=dk4)
  %dkt.e = tensor.empty() : tensor<1x12x64x256xf16>
  %dkt = linalg.transpose ins(%dk4 : tensor<1x12x256x64xf16>) outs(%dkt.e : tensor<1x12x64x256xf16>) permutation = [0, 1, 3, 2]
  // dqf = reshape(shape=fs,x=dqt)
  %dqf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %dqf = tensor.reshape %dqt(%dqf.s) : (tensor<1x12x64x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // dkf = reshape(shape=fs,x=dkt)
  %dkf.s = arith.constant dense<[1, 768, 1, 256]> : tensor<4xi64>
  %dkf = tensor.reshape %dkt(%dkf.s) : (tensor<1x12x64x256xf16>, tensor<4xi64>) -> tensor<1x768x1x256xf16>
  // out = concat(axis=cax,interleave=cid,values=(dqf,dkf))
  %out = tensor.concat dim(1) %dqf, %dkf : (tensor<1x768x1x256xf16>, tensor<1x768x1x256xf16>) -> tensor<1x1536x1x256xf16>
  return %out : tensor<1x1536x1x256xf16>
}

// Normal forms of the ANE kernels.
//
// The MIL programs in training/*.h are shaped by what the ANE compiler accepts: one input port, one output port, rank-4
// [1, C, 1, S] fp16, sizes fixed at compile time.  The functions here state what those programs compute once that
// packaging is removed.
//
//   generalization            every size is a `?`; H (heads), eps and the attention scale are arguments
//   normalization             one spelling per meaning:
//     layout                  activation = matrix [C, S] (the IOSurface bytes, unit dims dropped);
//                             per-head tensor = [H, S, HD]; forward weight = [OC, IC]
//     conv 1x1 [OC,IC,1,1]    -> @linear
//     reshape/transpose/matmul/transpose/reshape around a weight sliced out of the input -> @linear_t
//     baked W^T blob          -> @linear_t on the forward weight (the transpose belongs to the op, not to the data)
//     pow(x, -0.5)            -> math.rsqrt
//     concat(taps) on axis 1  -> several results;  slice_by_size on one packed input -> several arguments
//     concat(k, k) on heads   -> @gqa_tile_mod (an index map, no copy in the IR)
//     reshape to pairs/slice/negate/concat/reshape -> @rope (one generic)
//
// Each kernel-level function (@nf_*) is checked against the literal translation in test/*.mlir.

// ---------------------------------------------------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------------------------------------------------

// Y[o,s] = sum_i W[o,i] * X[i,s]            W: [OC, IC], X: [IC, S]
func.func @linear(%W: tensor<?x?xf16>, %X: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %o = tensor.dim %W, %c0 : tensor<?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?xf16>
  %z = arith.constant 0.0 : f16
  %e = tensor.empty(%o, %s) : tensor<?x?xf16>
  %f = linalg.fill ins(%z : f16) outs(%e : tensor<?x?xf16>) -> tensor<?x?xf16>
  %Y = linalg.matmul ins(%W, %X : tensor<?x?xf16>, tensor<?x?xf16>) outs(%f : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %Y : tensor<?x?xf16>
}

// Y[o,s] = sum_i Wt[i,o] * X[i,s]           Wt: [IC, OC] (the weight as the dynamic kernels stage it), X: [IC, S]
func.func @linear_t(%Wt: tensor<?x?xf16>, %X: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c1 = arith.constant 1 : index
  %o = tensor.dim %Wt, %c1 : tensor<?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?xf16>
  %z = arith.constant 0.0 : f16
  %e = tensor.empty(%o, %s) : tensor<?x?xf16>
  %f = linalg.fill ins(%z : f16) outs(%e : tensor<?x?xf16>) -> tensor<?x?xf16>
  %Y = linalg.contract
      indexing_maps = [affine_map<(o, s, i) -> (i, o)>, affine_map<(o, s, i) -> (i, s)>, affine_map<(o, s, i) -> (o, s)>]
      ins(%Wt, %X : tensor<?x?xf16>, tensor<?x?xf16>) outs(%f : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %Y : tensor<?x?xf16>
}

func.func @add(%A: tensor<?x?xf16>, %B: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %A, %c1 : tensor<?x?xf16>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %R = linalg.add ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>) outs(%e : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

// rrms[s] = rsqrt(mean_c(X[c,s]^2) + eps)
func.func @rrms(%X: tensor<?x?xf16>, %eps: f16) -> tensor<?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c = tensor.dim %X, %c0 : tensor<?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?xf16>
  %z = arith.constant 0.0 : f16
  %one = arith.constant 1.0 : f16
  %ci = arith.index_cast %c : index to i32
  %cf = arith.uitofp %ci : i32 to f16
  %invd = arith.divf %one, %cf : f16
  %e = tensor.empty(%s) : tensor<?xf16>
  %f = linalg.fill ins(%z : f16) outs(%e : tensor<?xf16>) -> tensor<?xf16>
  %ss = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (s)>],
                        iterator_types = ["reduction", "parallel"]}
      ins(%X : tensor<?x?xf16>) outs(%f : tensor<?xf16>) {
    ^bb0(%x : f16, %acc : f16):
      %sq = arith.mulf %x, %x : f16
      %r = arith.addf %acc, %sq : f16
      linalg.yield %r : f16
    } -> tensor<?xf16>
  %e2 = tensor.empty(%s) : tensor<?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(s) -> (s)>, affine_map<(s) -> (s)>], iterator_types = ["parallel"]}
      ins(%ss : tensor<?xf16>) outs(%e2 : tensor<?xf16>) {
    ^bb0(%a : f16, %o : f16):
      %m = arith.mulf %a, %invd : f16
      %me = arith.addf %m, %eps : f16
      %r = math.rsqrt %me : f16
      linalg.yield %r : f16
    } -> tensor<?xf16>
  return %R : tensor<?xf16>
}

// Y[c,s] = X[c,s] * rrms[s] * w[c]
func.func @rmsnorm(%X: tensor<?x?xf16>, %w: tensor<?xf16>, %eps: f16) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c = tensor.dim %X, %c0 : tensor<?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?xf16>
  %r = call @rrms(%X, %eps) : (tensor<?x?xf16>, f16) -> tensor<?xf16>
  %e = tensor.empty(%c, %s) : tensor<?x?xf16>
  %Y = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (s)>, affine_map<(c, s) -> (c)>,
                                        affine_map<(c, s) -> (c, s)>], iterator_types = ["parallel", "parallel"]}
      ins(%X, %r, %w : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%e : tensor<?x?xf16>) {
    ^bb0(%x : f16, %rr : f16, %ww : f16, %o : f16):
      %xr = arith.mulf %x, %rr : f16
      %y = arith.mulf %xr, %ww : f16
      linalg.yield %y : f16
    } -> tensor<?x?xf16>
  return %Y : tensor<?x?xf16>
}

// dX[c,s] = (dY[c,s]*w[c] - X[c,s] * coeff[s]) * rrms[s],   coeff[s] = sum_c(dY*w*X)[s] / C * rrms[s]^2
func.func @rmsnorm_bwd(%dY: tensor<?x?xf16>, %X: tensor<?x?xf16>, %w: tensor<?xf16>, %eps: f16) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c = tensor.dim %X, %c0 : tensor<?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?xf16>
  %z = arith.constant 0.0 : f16
  %one = arith.constant 1.0 : f16
  %ci = arith.index_cast %c : index to i32
  %cf = arith.uitofp %ci : i32 to f16
  %invd = arith.divf %one, %cf : f16
  %r = call @rrms(%X, %eps) : (tensor<?x?xf16>, f16) -> tensor<?xf16>
  %e = tensor.empty(%s) : tensor<?xf16>
  %f = linalg.fill ins(%z : f16) outs(%e : tensor<?xf16>) -> tensor<?xf16>
  %dot = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c)>,
                                          affine_map<(c, s) -> (s)>], iterator_types = ["reduction", "parallel"]}
      ins(%dY, %X, %w : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?xf16>) outs(%f : tensor<?xf16>) {
    ^bb0(%dy : f16, %x : f16, %ww : f16, %acc : f16):
      %dyw = arith.mulf %dy, %ww : f16
      %p = arith.mulf %dyw, %x : f16
      %a = arith.addf %acc, %p : f16
      linalg.yield %a : f16
    } -> tensor<?xf16>
  %e2 = tensor.empty(%c, %s) : tensor<?x?xf16>
  %dX = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c)>,
                                         affine_map<(c, s) -> (s)>, affine_map<(c, s) -> (s)>, affine_map<(c, s) -> (c, s)>],
                        iterator_types = ["parallel", "parallel"]}
      ins(%dY, %X, %w, %dot, %r : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>)
      outs(%e2 : tensor<?x?xf16>) {
    ^bb0(%dy : f16, %x : f16, %ww : f16, %d : f16, %rr : f16, %o : f16):
      %dyw = arith.mulf %dy, %ww : f16
      %dsc = arith.mulf %d, %invd : f16
      %rr2 = arith.mulf %rr, %rr : f16
      %cof = arith.mulf %dsc, %rr2 : f16
      %xc = arith.mulf %x, %cof : f16
      %dif = arith.subf %dyw, %xc : f16
      %out = arith.mulf %dif, %rr : f16
      linalg.yield %out : f16
    } -> tensor<?x?xf16>
  return %dX : tensor<?x?xf16>
}

// gate = silu(h1) * h3
func.func @swiglu_gate(%h1: tensor<?x?xf16>, %h3: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %h1, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %h1, %c1 : tensor<?x?xf16>
  %one = arith.constant 1.0 : f16
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %G = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%h1, %h3 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%e : tensor<?x?xf16>) {
    ^bb0(%a : f16, %b : f16, %o : f16):
      %n = arith.negf %a : f16
      %ex = math.exp %n : f16
      %den = arith.addf %one, %ex : f16
      %sig = arith.divf %one, %den : f16
      %silu = arith.mulf %a, %sig : f16
      %g = arith.mulf %silu, %b : f16
      linalg.yield %g : f16
    } -> tensor<?x?xf16>
  return %G : tensor<?x?xf16>
}

// Backward of the gate: (dh1, dh3) from dsilu = d(gate).
//   dh1 = dsilu * h3 * sig * (1 + h1 * (1 - sig)),   dh3 = dsilu * h1 * sig
func.func @swiglu_gate_bwd(%ds: tensor<?x?xf16>, %h1: tensor<?x?xf16>, %h3: tensor<?x?xf16>)
    -> (tensor<?x?xf16>, tensor<?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %h1, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %h1, %c1 : tensor<?x?xf16>
  %one = arith.constant 1.0 : f16
  %e1 = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %e3 = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %dh1, %dh3 = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>,
                                                affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>],
                               iterator_types = ["parallel", "parallel"]}
      ins(%ds, %h1, %h3 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%e1, %e3 : tensor<?x?xf16>, tensor<?x?xf16>) {
    ^bb0(%d : f16, %a : f16, %b : f16, %o1 : f16, %o3 : f16):
      %n = arith.negf %a : f16
      %ex = math.exp %n : f16
      %den = arith.addf %one, %ex : f16
      %sig = arith.divf %one, %den : f16
      %oms = arith.subf %one, %sig : f16
      %homs = arith.mulf %a, %oms : f16
      %brk = arith.addf %one, %homs : f16
      %dsd = arith.mulf %sig, %brk : f16
      %t1 = arith.mulf %d, %b : f16
      %r1 = arith.mulf %t1, %dsd : f16
      %slh = arith.mulf %a, %sig : f16
      %r3 = arith.mulf %d, %slh : f16
      linalg.yield %r1, %r3 : f16, f16
    } -> (tensor<?x?xf16>, tensor<?x?xf16>)
  return %dh1, %dh3 : tensor<?x?xf16>, tensor<?x?xf16>
}

// [H*HD, S] -> [H, S, HD]
func.func @split_heads(%X: tensor<?x?xf16>, %H: index) -> tensor<?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c = tensor.dim %X, %c0 : tensor<?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?xf16>
  %hd = arith.divui %c, %H : index
  %x3 = tensor.expand_shape %X [[0, 1], [2]] output_shape [%H, %hd, %s] : tensor<?x?xf16> into tensor<?x?x?xf16>
  %e = tensor.empty(%H, %s, %hd) : tensor<?x?x?xf16>
  %R = linalg.transpose ins(%x3 : tensor<?x?x?xf16>) outs(%e : tensor<?x?x?xf16>) permutation = [0, 2, 1]
  return %R : tensor<?x?x?xf16>
}

// [H, S, HD] -> [H*HD, S]
func.func @merge_heads(%X: tensor<?x?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h = tensor.dim %X, %c0 : tensor<?x?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?x?xf16>
  %hd = tensor.dim %X, %c2 : tensor<?x?x?xf16>
  %e = tensor.empty(%h, %hd, %s) : tensor<?x?x?xf16>
  %t = linalg.transpose ins(%X : tensor<?x?x?xf16>) outs(%e : tensor<?x?x?xf16>) permutation = [0, 2, 1]
  %R = tensor.collapse_shape %t [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

// Rotary embedding on [H, S, HD] with tables cos/sin [S, HD] (each value stored twice, at 2i and 2i+1):
//   y[2i]   = x[2i]   * cos - x[2i+1] * sin
//   y[2i+1] = x[2i+1] * cos + x[2i]   * sin
func.func @rope(%X: tensor<?x?x?xf16>, %cos: tensor<?x?xf16>, %sin: tensor<?x?xf16>) -> tensor<?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h = tensor.dim %X, %c0 : tensor<?x?x?xf16>
  %s = tensor.dim %X, %c1 : tensor<?x?x?xf16>
  %hd = tensor.dim %X, %c2 : tensor<?x?x?xf16>
  %e = tensor.empty(%h, %s, %hd) : tensor<?x?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(h, s, d) -> (h, s, d)>, affine_map<(h, s, d) -> (s, d)>,
                                        affine_map<(h, s, d) -> (s, d)>, affine_map<(h, s, d) -> (h, s, d)>],
                       iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%X, %cos, %sin : tensor<?x?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%e : tensor<?x?x?xf16>) {
    ^bb0(%x : f16, %cv : f16, %sv : f16, %o : f16):
      %ih = linalg.index 0 : index
      %is = linalg.index 1 : index
      %id = linalg.index 2 : index
      %partner = arith.xori %id, %c1 : index
      %xp = tensor.extract %X[%ih, %is, %partner] : tensor<?x?x?xf16>
      %bit = arith.andi %id, %c1 : index
      %odd = arith.cmpi eq, %bit, %c1 : index
      %nxp = arith.negf %xp : f16
      %rot = arith.select %odd, %xp, %nxp : f16
      %t0 = arith.mulf %x, %cv : f16
      %t1 = arith.mulf %rot, %sv : f16
      %y = arith.addf %t0, %t1 : f16
      linalg.yield %y : f16
    } -> tensor<?x?x?xf16>
  return %R : tensor<?x?x?xf16>
}

// Grouped-query attention: give each of the H query heads a K/V head.
//   @gqa_tile_mod: head h reads kv head (h mod KVH).  This is what concat(k, k) in gen_sdpa_fwd_dynamic computes.
//   @gqa_tile_div: head h reads kv head (h div (H/KVH)).  This is what gqa_tile_kv()/gqa_reduce_kv() in io.h compute on
//                  the CPU for the backward pass (and the usual repeat_interleave convention).
// The two agree only when H == KVH.
func.func @gqa_tile_mod(%K: tensor<?x?x?xf16>, %H: index) -> tensor<?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %kvh = tensor.dim %K, %c0 : tensor<?x?x?xf16>
  %s = tensor.dim %K, %c1 : tensor<?x?x?xf16>
  %hd = tensor.dim %K, %c2 : tensor<?x?x?xf16>
  %e = tensor.empty(%H, %s, %hd) : tensor<?x?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(h, s, d) -> (h, s, d)>], iterator_types = ["parallel", "parallel", "parallel"]}
      outs(%e : tensor<?x?x?xf16>) {
    ^bb0(%o : f16):
      %ih = linalg.index 0 : index
      %is = linalg.index 1 : index
      %id = linalg.index 2 : index
      %g = arith.remui %ih, %kvh : index
      %x = tensor.extract %K[%g, %is, %id] : tensor<?x?x?xf16>
      linalg.yield %x : f16
    } -> tensor<?x?x?xf16>
  return %R : tensor<?x?x?xf16>
}

func.func @gqa_tile_div(%K: tensor<?x?x?xf16>, %H: index) -> tensor<?x?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %kvh = tensor.dim %K, %c0 : tensor<?x?x?xf16>
  %s = tensor.dim %K, %c1 : tensor<?x?x?xf16>
  %hd = tensor.dim %K, %c2 : tensor<?x?x?xf16>
  %ratio = arith.divui %H, %kvh : index
  %e = tensor.empty(%H, %s, %hd) : tensor<?x?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(h, s, d) -> (h, s, d)>], iterator_types = ["parallel", "parallel", "parallel"]}
      outs(%e : tensor<?x?x?xf16>) {
    ^bb0(%o : f16):
      %ih = linalg.index 0 : index
      %is = linalg.index 1 : index
      %id = linalg.index 2 : index
      %g = arith.divui %ih, %ratio : index
      %x = tensor.extract %K[%g, %is, %id] : tensor<?x?x?xf16>
      linalg.yield %x : f16
    } -> tensor<?x?x?xf16>
  return %R : tensor<?x?x?xf16>
}

// Scaled dot-product attention on [H, S, HD] with an additive mask [S, S].
//   P[h,i,j] = softmax_j(scale * sum_d Q[h,i,d] K[h,j,d] + mask[i,j]),   A[h,i,d] = sum_j P[h,i,j] V[h,j,d]
func.func @sdpa(%Q: tensor<?x?x?xf16>, %K: tensor<?x?x?xf16>, %V: tensor<?x?x?xf16>, %mask: tensor<?x?xf16>, %scale: f16)
    -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h = tensor.dim %Q, %c0 : tensor<?x?x?xf16>
  %s = tensor.dim %Q, %c1 : tensor<?x?x?xf16>
  %hd = tensor.dim %Q, %c2 : tensor<?x?x?xf16>
  %z = arith.constant 0.0 : f16
  %se = tensor.empty(%h, %s, %s) : tensor<?x?x?xf16>
  %sf = linalg.fill ins(%z : f16) outs(%se : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %sc = linalg.contract
      indexing_maps = [affine_map<(h, i, j, d) -> (h, i, d)>, affine_map<(h, i, j, d) -> (h, j, d)>, affine_map<(h, i, j, d) -> (h, i, j)>]
      ins(%Q, %K : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%sf : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %me = tensor.empty(%h, %s, %s) : tensor<?x?x?xf16>
  %ms = linalg.generic {indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>, affine_map<(h, i, j) -> (i, j)>,
                                         affine_map<(h, i, j) -> (h, i, j)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%sc, %mask : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%me : tensor<?x?x?xf16>) {
    ^bb0(%a : f16, %m : f16, %o : f16):
      %t = arith.mulf %a, %scale : f16
      %r = arith.addf %t, %m : f16
      linalg.yield %r : f16
    } -> tensor<?x?x?xf16>
  %pe = tensor.empty(%h, %s, %s) : tensor<?x?x?xf16>
  %P = linalg.softmax dimension(2) ins(%ms : tensor<?x?x?xf16>) outs(%pe : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %ae = tensor.empty(%h, %s, %hd) : tensor<?x?x?xf16>
  %af = linalg.fill ins(%z : f16) outs(%ae : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %A = linalg.contract
      indexing_maps = [affine_map<(h, i, d, j) -> (h, i, j)>, affine_map<(h, i, d, j) -> (h, j, d)>, affine_map<(h, i, d, j) -> (h, i, d)>]
      ins(%P, %V : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%af : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  return %A, %P : tensor<?x?x?xf16>, tensor<?x?x?xf16>
}

// Backward of @sdpa: (dQ, dK, dV) from dA.  P is recomputed from Q, K as the ANE kernel does.
//   dV[h,j,d] = sum_i P[h,i,j] dA[h,i,d]           dP[h,i,j] = sum_d dA[h,i,d] V[h,j,d]
//   dS = P * (dP - sum_j P*dP) * scale
//   dQ[h,i,d] = sum_j dS[h,i,j] K[h,j,d]           dK[h,j,d] = sum_i dS[h,i,j] Q[h,i,d]
func.func @sdpa_bwd(%Q: tensor<?x?x?xf16>, %K: tensor<?x?x?xf16>, %V: tensor<?x?x?xf16>, %mask: tensor<?x?xf16>, %scale: f16,
                    %dA: tensor<?x?x?xf16>) -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %h = tensor.dim %Q, %c0 : tensor<?x?x?xf16>
  %s = tensor.dim %Q, %c1 : tensor<?x?x?xf16>
  %hd = tensor.dim %Q, %c2 : tensor<?x?x?xf16>
  %z = arith.constant 0.0 : f16
  %A, %P = call @sdpa(%Q, %K, %V, %mask, %scale)
      : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xf16>, f16) -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %ve = tensor.empty(%h, %s, %hd) : tensor<?x?x?xf16>
  %vf = linalg.fill ins(%z : f16) outs(%ve : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %dV = linalg.contract
      indexing_maps = [affine_map<(h, j, d, i) -> (h, i, j)>, affine_map<(h, j, d, i) -> (h, i, d)>, affine_map<(h, j, d, i) -> (h, j, d)>]
      ins(%P, %dA : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%vf : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %pe = tensor.empty(%h, %s, %s) : tensor<?x?x?xf16>
  %pf = linalg.fill ins(%z : f16) outs(%pe : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %dP = linalg.contract
      indexing_maps = [affine_map<(h, i, j, d) -> (h, i, d)>, affine_map<(h, i, j, d) -> (h, j, d)>, affine_map<(h, i, j, d) -> (h, i, j)>]
      ins(%dA, %V : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%pf : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %re = tensor.empty(%h, %s) : tensor<?x?xf16>
  %rf = linalg.fill ins(%z : f16) outs(%re : tensor<?x?xf16>) -> tensor<?x?xf16>
  %spdp = linalg.generic {indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>, affine_map<(h, i, j) -> (h, i, j)>,
                                           affine_map<(h, i, j) -> (h, i)>], iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%P, %dP : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%rf : tensor<?x?xf16>) {
    ^bb0(%p : f16, %dp : f16, %acc : f16):
      %m = arith.mulf %p, %dp : f16
      %r = arith.addf %acc, %m : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>
  %de = tensor.empty(%h, %s, %s) : tensor<?x?x?xf16>
  %dS = linalg.generic {indexing_maps = [affine_map<(h, i, j) -> (h, i, j)>, affine_map<(h, i, j) -> (h, i, j)>,
                                         affine_map<(h, i, j) -> (h, i)>, affine_map<(h, i, j) -> (h, i, j)>],
                        iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%P, %dP, %spdp : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%de : tensor<?x?x?xf16>) {
    ^bb0(%p : f16, %dp : f16, %sm : f16, %o : f16):
      %dps = arith.subf %dp, %sm : f16
      %ds0 = arith.mulf %p, %dps : f16
      %r = arith.mulf %ds0, %scale : f16
      linalg.yield %r : f16
    } -> tensor<?x?x?xf16>
  %qe = tensor.empty(%h, %s, %hd) : tensor<?x?x?xf16>
  %qf = linalg.fill ins(%z : f16) outs(%qe : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %dQ = linalg.contract
      indexing_maps = [affine_map<(h, i, d, j) -> (h, i, j)>, affine_map<(h, i, d, j) -> (h, j, d)>, affine_map<(h, i, d, j) -> (h, i, d)>]
      ins(%dS, %K : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%qf : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %ke = tensor.empty(%h, %s, %hd) : tensor<?x?x?xf16>
  %kf = linalg.fill ins(%z : f16) outs(%ke : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  %dK = linalg.contract
      indexing_maps = [affine_map<(h, j, d, i) -> (h, i, j)>, affine_map<(h, j, d, i) -> (h, i, d)>, affine_map<(h, j, d, i) -> (h, j, d)>]
      ins(%dS, %Q : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%kf : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
  return %dQ, %dK, %dV : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>
}

// Softmax over channels of [C, S] (gen_softmax_vocab).
func.func @softmax_ch(%X: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %X, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %X, %c1 : tensor<?x?xf16>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %R = linalg.softmax dimension(0) ins(%X : tensor<?x?xf16>) outs(%e : tensor<?x?xf16>) -> tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

// W8A8 activation caching (ane_int8_bench.m).  What the device does, measured on an M3 Pro (README "Sharp corners"):
//   q  = sat_i8(round_half_away(x * f16(1 / scale)))      the reciprocal is taken in fp16 and multiplied, not divided by
//   x' = fp16_ties_away(q * scale)                        q * scale is exact in f32; the fp16 rounding sends ties away from 0
// The clamp comes before the float->int conversion: arith.fptosi is poison out of range, AArch64 fcvt* saturates.
func.func @quantize(%X: tensor<?x?xf16>, %scale: f16) -> tensor<?x?xi8> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %X, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %X, %c1 : tensor<?x?xf16>
  %lo = arith.constant -128.0 : f16
  %hi = arith.constant 127.0 : f16
  %one = arith.constant 1.0 : f16
  %inv = arith.divf %one, %scale : f16
  %e = tensor.empty(%d0, %d1) : tensor<?x?xi8>
  %R = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>], iterator_types = ["parallel", "parallel"]}
      ins(%X : tensor<?x?xf16>) outs(%e : tensor<?x?xi8>) {
    ^bb0(%x : f16, %o : i8):
      %q = arith.mulf %x, %inv : f16
      %rn = math.round %q : f16
      %a = arith.maximumf %rn, %lo : f16
      %b = arith.minimumf %a, %hi : f16
      %r = arith.fptosi %b : f16 to i8
      linalg.yield %r : i8
    } -> tensor<?x?xi8>
  return %R : tensor<?x?xi8>
}

// arith.truncf has a `to_nearest_away` mode, but the stock lowering ignores it (mlir-opt 23: the result is rounded to
// even), so the tie rule is done on the bits.  For an f32 whose fp16 image is normal, the 13 dropped mantissa bits are
// the low 13 bits: add half a unit (0x1000) to the magnitude, clear them, and the truncation is exact; a carry into the
// exponent is the correct rounding up.  Below 2^-14 the fp16 grid is 2^-24: round(p * 2^24) / 2^24.
func.func @dequantize(%Q: tensor<?x?xi8>, %scale: f16) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %Q, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %Q, %c1 : tensor<?x?xi8>
  %kmag = arith.constant 2147483647 : i32
  %ksgn = arith.constant -2147483648 : i32
  %khalf = arith.constant 4096 : i32
  %kclr = arith.constant -8192 : i32
  %knorm = arith.constant 947912704 : i32
  %up = arith.constant 16777216.0 : f32
  %dn = arith.constant 5.9604644775390625e-08 : f32
  %s32 = arith.extf %scale : f16 to f32
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>], iterator_types = ["parallel", "parallel"]}
      ins(%Q : tensor<?x?xi8>) outs(%e : tensor<?x?xf16>) {
    ^bb0(%q : i8, %o : f16):
      %f = arith.sitofp %q : i8 to f32
      %p = arith.mulf %f, %s32 : f32
      %b = arith.bitcast %p : f32 to i32
      %m = arith.andi %b, %kmag : i32
      %sg = arith.andi %b, %ksgn : i32
      %inc = arith.addi %m, %khalf : i32
      %tr = arith.andi %inc, %kclr : i32
      %ob = arith.ori %tr, %sg : i32
      %ra = arith.bitcast %ob : i32 to f32
      %pk = arith.mulf %p, %up : f32
      %rk = math.round %pk : f32
      %rs = arith.mulf %rk, %dn : f32
      %small = arith.cmpi ult, %m, %knorm : i32
      %sel = arith.select %small, %rs, %ra : f32
      %r = arith.truncf %sel : f32 to f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

// ---------------------------------------------------------------------------------------------------------------------
// Kernels: one normal form per family of MIL programs
// ---------------------------------------------------------------------------------------------------------------------

// gen_sdpa_fwd_taps: (o, q, k, v, a, xn).  X: [D, S]; Wq, Wk, Wv: [D, D]; Wo: [D, D]; mask: [S, S].
func.func @nf_sdpa_fwd(%X: tensor<?x?xf16>, %rw: tensor<?xf16>, %Wq: tensor<?x?xf16>, %Wk: tensor<?x?xf16>, %Wv: tensor<?x?xf16>,
                       %Wo: tensor<?x?xf16>, %mask: tensor<?x?xf16>, %H: index, %eps: f16, %scale: f16)
    -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) {
  %xn = call @rmsnorm(%X, %rw, %eps) : (tensor<?x?xf16>, tensor<?xf16>, f16) -> tensor<?x?xf16>
  %qf = call @linear(%Wq, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kf = call @linear(%Wk, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %vf = call @linear(%Wv, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %q = call @split_heads(%qf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %k = call @split_heads(%kf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %v = call @split_heads(%vf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %a, %p = call @sdpa(%q, %k, %v, %mask, %scale)
      : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xf16>, f16) -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %af = call @merge_heads(%a) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %o = call @linear(%Wo, %af) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %o, %qf, %kf, %vf, %af, %xn : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
}

// gen_ffn_fwd_taps: (y, h1, h3, gate, xn).  W1, W3: [HID, D]; W2: [D, HID].
func.func @nf_ffn_fwd(%X: tensor<?x?xf16>, %rw: tensor<?xf16>, %W1: tensor<?x?xf16>, %W3: tensor<?x?xf16>, %W2: tensor<?x?xf16>,
                      %eps: f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) {
  %xn = call @rmsnorm(%X, %rw, %eps) : (tensor<?x?xf16>, tensor<?xf16>, f16) -> tensor<?x?xf16>
  %h1 = call @linear(%W1, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %h3 = call @linear(%W3, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %g = call @swiglu_gate(%h1, %h3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %y = call @linear(%W2, %g) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %y, %h1, %h3, %g, %xn : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
}

// gen_ffn_bwd: (dx, dh1, dh3) from (dy, h1, h3).  Takes the forward weights; the ANE kernel bakes W1^T, W2^T, W3^T.
func.func @nf_ffn_bwd(%dy: tensor<?x?xf16>, %h1: tensor<?x?xf16>, %h3: tensor<?x?xf16>, %W1: tensor<?x?xf16>, %W3: tensor<?x?xf16>,
                      %W2: tensor<?x?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) {
  %ds = call @linear_t(%W2, %dy) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %dh1, %dh3 = call @swiglu_gate_bwd(%ds, %h1, %h3)
      : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>)
  %dx1 = call @linear_t(%W1, %dh1) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %dx3 = call @linear_t(%W3, %dh3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %dx = call @add(%dx1, %dx3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %dx, %dh1, %dh3 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
}

// gen_qkvb: dx = Wq^T dq + Wk^T dk + Wv^T dv
func.func @nf_qkv_bwd(%dq: tensor<?x?xf16>, %dk: tensor<?x?xf16>, %dv: tensor<?x?xf16>, %Wq: tensor<?x?xf16>, %Wk: tensor<?x?xf16>,
                      %Wv: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %a = call @linear_t(%Wq, %dq) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %b = call @linear_t(%Wk, %dk) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %c = call @linear_t(%Wv, %dv) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %ab = call @add(%a, %b) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %r = call @add(%ab, %c) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %r : tensor<?x?xf16>
}

// gen_sdpa_bwd1 followed by gen_sdpa_bwd2: (dq, dk, dv) in [D, S] from the taps (q, k, v) and the gradient of the block output.
func.func @nf_sdpa_bwd(%qf: tensor<?x?xf16>, %kf: tensor<?x?xf16>, %vf: tensor<?x?xf16>, %do: tensor<?x?xf16>, %Wo: tensor<?x?xf16>,
                       %mask: tensor<?x?xf16>, %H: index, %scale: f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) {
  %daf = call @linear_t(%Wo, %do) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %q = call @split_heads(%qf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %k = call @split_heads(%kf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %v = call @split_heads(%vf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %da = call @split_heads(%daf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %dq, %dk, %dv = call @sdpa_bwd(%q, %k, %v, %mask, %scale, %da)
      : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xf16>, f16, tensor<?x?x?xf16>)
        -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %dqf = call @merge_heads(%dq) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %dkf = call @merge_heads(%dk) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %dvf = call @merge_heads(%dv) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  return %dqf, %dkf, %dvf : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
}

// gen_sdpa_fwd_dynamic (GQA + RoPE, weights staged transposed, no Wo): (a, q_rope, k_rope, v).
//   xn: [D, S]; Wqt: [D, H*HD]; Wkt, Wvt: [D, KVH*HD]; cos, sin: [S, HD]
func.func @nf_sdpa_fwd_gqa(%xn: tensor<?x?xf16>, %Wqt: tensor<?x?xf16>, %Wkt: tensor<?x?xf16>, %Wvt: tensor<?x?xf16>,
                           %cos: tensor<?x?xf16>, %sin: tensor<?x?xf16>, %mask: tensor<?x?xf16>, %H: index, %KVH: index, %scale: f16)
    -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) {
  %qf = call @linear_t(%Wqt, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kf = call @linear_t(%Wkt, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %vf = call @linear_t(%Wvt, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %q0 = call @split_heads(%qf, %H) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %k0 = call @split_heads(%kf, %KVH) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %v0 = call @split_heads(%vf, %KVH) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %q = call @rope(%q0, %cos, %sin) : (tensor<?x?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?x?xf16>
  %k = call @rope(%k0, %cos, %sin) : (tensor<?x?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?x?xf16>
  %kt = call @gqa_tile_mod(%k, %H) : (tensor<?x?x?xf16>, index) -> tensor<?x?x?xf16>
  %vt = call @gqa_tile_mod(%v0, %H) : (tensor<?x?x?xf16>, index) -> tensor<?x?x?xf16>
  %a, %p = call @sdpa(%q, %kt, %vt, %mask, %scale)
      : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xf16>, f16) -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %af = call @merge_heads(%a) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %qrf = call @merge_heads(%q) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %krf = call @merge_heads(%k) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  return %af, %qrf, %krf, %vf : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
}

// gen_ffn_fused_dynamic: (x_next, h1, h3, gate) with x_next = x2 + alpha * W2 * gate.
//   W1t, W3t: [D, HID] (transposed, as staged); W2: [D, HID] (forward weight, as staged)
func.func @nf_ffn_fused(%xn: tensor<?x?xf16>, %x2: tensor<?x?xf16>, %W1t: tensor<?x?xf16>, %W3t: tensor<?x?xf16>, %W2: tensor<?x?xf16>,
                        %alpha: f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) {
  %h1 = call @linear_t(%W1t, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %h3 = call @linear_t(%W3t, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %g = call @swiglu_gate(%h1, %h3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %y = call @linear(%W2, %g) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %xnext = call @axpy(%alpha, %y, %x2) : (f16, tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  return %xnext, %h1, %h3, %g : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>
}

// R = B + alpha * A   (the residual connection inside gen_ffn_fused_dynamic)
func.func @axpy(%alpha: f16, %A: tensor<?x?xf16>, %B: tensor<?x?xf16>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %A, %c1 : tensor<?x?xf16>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%A, %B : tensor<?x?xf16>, tensor<?x?xf16>) outs(%e : tensor<?x?xf16>) {
    ^bb0(%a : f16, %b : f16, %o : f16):
      %t = arith.mulf %a, %alpha : f16
      %r = arith.addf %b, %t : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

// ---------------------------------------------------------------------------------------------------------------------
// The fp32 <-> fp16 boundary (MIL `cast`; io_write_fp16 / io_read_fp16 on the CPU side) and the baked tables
// ---------------------------------------------------------------------------------------------------------------------

// Round to nearest even; |x| > 65519.99 becomes +-inf.  See aarch64/neon_fp16.mlir for the vector form.
func.func @to_f16(%X: tensor<?x?xf32>) -> tensor<?x?xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %X, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %X, %c1 : tensor<?x?xf32>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>], iterator_types = ["parallel", "parallel"]}
      ins(%X : tensor<?x?xf32>) outs(%e : tensor<?x?xf16>) {
    ^bb0(%x : f32, %o : f16):
      %r = arith.truncf %x : f32 to f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

func.func @to_f32(%X: tensor<?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %X, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %X, %c1 : tensor<?x?xf16>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %R = linalg.generic {indexing_maps = [affine_map<(c, s) -> (c, s)>, affine_map<(c, s) -> (c, s)>], iterator_types = ["parallel", "parallel"]}
      ins(%X : tensor<?x?xf16>) outs(%e : tensor<?x?xf32>) {
    ^bb0(%x : f16, %o : f32):
      %r = arith.extf %x : f16 to f32
      linalg.yield %r : f32
    } -> tensor<?x?xf32>
  return %R : tensor<?x?xf32>
}

// get_mask_blob: mask[t, t2] = 0 if t2 <= t else -65504 (the most negative finite fp16, not -inf).
func.func @causal_mask(%s: index) -> tensor<?x?xf16> {
  %zero = arith.constant 0.0 : f16
  %neg = arith.constant -65504.0 : f16
  %e = tensor.empty(%s, %s) : tensor<?x?xf16>
  %R = linalg.generic {indexing_maps = [affine_map<(t, u) -> (t, u)>], iterator_types = ["parallel", "parallel"]}
      outs(%e : tensor<?x?xf16>) {
    ^bb0(%o : f16):
      %t = linalg.index 0 : index
      %u = linalg.index 1 : index
      %keep = arith.cmpi ule, %u, %t : index
      %r = arith.select %keep, %zero, %neg : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>
  return %R : tensor<?x?xf16>
}

// get_rope_cos_blob / get_rope_sin_blob: tables [S, HD], theta(p, i) = p / 10000^(2i/HD), stored at 2i and 2i+1.
// Computed in f32 and rounded to f16, as the C code does.
func.func @rope_tables(%s: index, %hd: index) -> (tensor<?x?xf16>, tensor<?x?xf16>) {
  %c2 = arith.constant 2 : index
  %base = arith.constant 10000.0 : f32
  %two = arith.constant 2.0 : f32
  %hdi = arith.index_cast %hd : index to i32
  %hdf = arith.sitofp %hdi : i32 to f32
  %ec = tensor.empty(%s, %hd) : tensor<?x?xf16>
  %es = tensor.empty(%s, %hd) : tensor<?x?xf16>
  %C, %S = linalg.generic {indexing_maps = [affine_map<(p, d) -> (p, d)>, affine_map<(p, d) -> (p, d)>], iterator_types = ["parallel", "parallel"]}
      outs(%ec, %es : tensor<?x?xf16>, tensor<?x?xf16>) {
    ^bb0(%o0 : f16, %o1 : f16):
      %p = linalg.index 0 : index
      %d = linalg.index 1 : index
      %i = arith.divui %d, %c2 : index
      %ii = arith.index_cast %i : index to i32
      %pi = arith.index_cast %p : index to i32
      %fi = arith.sitofp %ii : i32 to f32
      %fp = arith.sitofp %pi : i32 to f32
      %n = arith.mulf %two, %fi : f32
      %ex = arith.divf %n, %hdf : f32
      %den = math.powf %base, %ex : f32
      %theta = arith.divf %fp, %den : f32
      %cv = math.cos %theta : f32
      %sv = math.sin %theta : f32
      %ch = arith.truncf %cv : f32 to f16
      %sh = arith.truncf %sv : f32 to f16
      linalg.yield %ch, %sh : f16, f16
    } -> (tensor<?x?xf16>, tensor<?x?xf16>)
  return %C, %S : tensor<?x?xf16>, tensor<?x?xf16>
}

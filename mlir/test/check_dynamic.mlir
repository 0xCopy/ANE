// Dynamic-weight kernels (training/training_dynamic/mil_dynamic.h) against their normal forms.
//
// Sizes: DIM=8 HIDDEN=12 HEADS=4 KV_HEADS=2 HD=4 SEQ=6 NLAYERS=2, so Q_DIM=16, KV_DIM=8, GQA_RATIO=2.
// A dynamic kernel has one input [1, IC, 1, SEQ + sum(OC)]: activations in the first SEQ columns, weights (stored
// [IC, OC], i.e. transposed) in the rest.  The drivers pack that input with hcat and unpack the outputs with rows.
// 
// The last two lines show the head-tiling mismatch: sdpaFwd tiles K/V with concat(k, k) (head h <- kv head h mod KVH),
// the CPU helpers used by the backward pass (gqa_tile_kv, gqa_reduce_kv) use h div GQA_RATIO.
//
// RUN: cat %S/harness.mlir %S/../normal/ane_normal.mlir %S/small/dynamic.mlir %s | mlir-opt \
// RUN:   --transform-preload-library=transform-library-paths=%S/decompose_softmax.mlir --transform-interpreter \
// RUN:   --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
// RUN:   --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%llvmlib/libmlir_runner_utils.dylib,%llvmlib/libmlir_c_runner_utils.dylib | FileCheck %s

// CHECK: dyn_matmul_mil: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: wo_fwd_dynamic: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_bwd_w2t_dynamic: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: wot_dynamic: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: q_bwd_dynamic: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_fwd_gqa attn: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_fwd_gqa q_rope: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_fwd_gqa k_rope: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_fwd_gqa v: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_fwd_gqa xnorm: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_fused x_next: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_fused h1: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_fused h3: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_fused gate: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_bwd_w13t_dynamic: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: kv_bwd_dynamic: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_bwd dv: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_bwd dq: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_bwd dk: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: gqa concat(k,k) vs tile_mod: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: gqa tile_mod vs tile_div: maxabs {{[0-9.e+-]+}} maxdiff 0.{{[0-9]*[1-9]}}

func.func @main() {
  %scale = arith.constant 0.5 : f16
  %alpha = arith.constant 0.5 : f16
  %i6 = arith.constant 6 : index
  %mask = call @causal_mask(%i6) : (index) -> tensor<?x?xf16>
  %mask4.c = tensor.cast %mask : tensor<?x?xf16> to tensor<6x6xf16>
  %mask4 = tensor.expand_shape %mask4.c [[0, 1, 2], [3]] output_shape [1, 1, 6, 6] : tensor<6x6xf16> into tensor<1x1x6x6xf16>
  %i4 = arith.constant 4 : index
  %cos, %sin = call @rope_tables(%i6, %i4) : (index, index) -> (tensor<?x?xf16>, tensor<?x?xf16>)
  %cos4.c = tensor.cast %cos : tensor<?x?xf16> to tensor<6x4xf16>
  %cos4 = tensor.expand_shape %cos4.c [[0, 1, 2], [3]] output_shape [1, 1, 6, 4] : tensor<6x4xf16> into tensor<1x1x6x4xf16>
  %sin4.c = tensor.cast %sin : tensor<?x?xf16> to tensor<6x4xf16>
  %sin4 = tensor.expand_shape %sin4.c [[0, 1, 2], [3]] output_shape [1, 1, 6, 4] : tensor<6x4xf16> into tensor<1x1x6x4xf16>
  %mm.x.s = arith.constant 1.0 : f32
  %mm.x.a = arith.constant 0.5 : f32
  %i8 = arith.constant 8 : index
  %mm.x = call @gen(%i8, %i6, %mm.x.s, %mm.x.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %mm.w.s = arith.constant 1.9 : f32
  %mm.w.a = arith.constant 0.5 : f32
  %i12 = arith.constant 12 : index
  %mm.w = call @gen(%i8, %i12, %mm.w.s, %mm.w.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %mm.in = call @hcat(%mm.x, %mm.w) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %mm.in4.c = tensor.cast %mm.in : tensor<?x?xf16> to tensor<8x18xf16>
  %mm.in4 = tensor.expand_shape %mm.in4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 18] : tensor<8x18xf16> into tensor<1x8x1x18xf16>
  %mm.l4 = call @ane_dyn_matmul_mil(%mm.in4) : (tensor<1x8x1x18xf16>) -> tensor<1x12x1x6xf16>
  %mm.l.c = tensor.collapse_shape %mm.l4 [[0, 1], [2, 3]] : tensor<1x12x1x6xf16> into tensor<12x6xf16>
  %mm.l = tensor.cast %mm.l.c : tensor<12x6xf16> to tensor<?x?xf16>
  %mm.n = call @linear_t(%mm.w, %mm.x) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "dyn_matmul_mil: "
  call @report(%mm.l, %mm.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %wo_fwd.x.s = arith.constant 2.8 : f32
  %wo_fwd.x.a = arith.constant 0.5 : f32
  %i16 = arith.constant 16 : index
  %wo_fwd.x = call @gen(%i16, %i6, %wo_fwd.x.s, %wo_fwd.x.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %wo_fwd.w.s = arith.constant 3.7 : f32
  %wo_fwd.w.a = arith.constant 0.5 : f32
  %wo_fwd.w = call @gen(%i16, %i8, %wo_fwd.w.s, %wo_fwd.w.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %wo_fwd.in = call @hcat(%wo_fwd.x, %wo_fwd.w) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %wo_fwd.in4.c = tensor.cast %wo_fwd.in : tensor<?x?xf16> to tensor<16x14xf16>
  %wo_fwd.in4 = tensor.expand_shape %wo_fwd.in4.c [[0, 1], [2, 3]] output_shape [1, 16, 1, 14] : tensor<16x14xf16> into tensor<1x16x1x14xf16>
  %wo_fwd.l4 = call @ane_wo_fwd_dynamic(%wo_fwd.in4) : (tensor<1x16x1x14xf16>) -> tensor<1x8x1x6xf16>
  %wo_fwd.l.c = tensor.collapse_shape %wo_fwd.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %wo_fwd.l = tensor.cast %wo_fwd.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %wo_fwd.n = call @linear_t(%wo_fwd.w, %wo_fwd.x) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "wo_fwd_dynamic: "
  call @report(%wo_fwd.l, %wo_fwd.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_bwd_w2t.x.s = arith.constant 4.6 : f32
  %ffn_bwd_w2t.x.a = arith.constant 0.5 : f32
  %ffn_bwd_w2t.x = call @gen(%i8, %i6, %ffn_bwd_w2t.x.s, %ffn_bwd_w2t.x.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %ffn_bwd_w2t.w.s = arith.constant 5.5 : f32
  %ffn_bwd_w2t.w.a = arith.constant 0.5 : f32
  %ffn_bwd_w2t.w = call @gen(%i8, %i12, %ffn_bwd_w2t.w.s, %ffn_bwd_w2t.w.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %ffn_bwd_w2t.in = call @hcat(%ffn_bwd_w2t.x, %ffn_bwd_w2t.w) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %ffn_bwd_w2t.in4.c = tensor.cast %ffn_bwd_w2t.in : tensor<?x?xf16> to tensor<8x18xf16>
  %ffn_bwd_w2t.in4 = tensor.expand_shape %ffn_bwd_w2t.in4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 18] : tensor<8x18xf16> into tensor<1x8x1x18xf16>
  %ffn_bwd_w2t.l4 = call @ane_ffn_bwd_w2t_dynamic(%ffn_bwd_w2t.in4) : (tensor<1x8x1x18xf16>) -> tensor<1x12x1x6xf16>
  %ffn_bwd_w2t.l.c = tensor.collapse_shape %ffn_bwd_w2t.l4 [[0, 1], [2, 3]] : tensor<1x12x1x6xf16> into tensor<12x6xf16>
  %ffn_bwd_w2t.l = tensor.cast %ffn_bwd_w2t.l.c : tensor<12x6xf16> to tensor<?x?xf16>
  %ffn_bwd_w2t.n = call @linear_t(%ffn_bwd_w2t.w, %ffn_bwd_w2t.x) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "ffn_bwd_w2t_dynamic: "
  call @report(%ffn_bwd_w2t.l, %ffn_bwd_w2t.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %wot.x.s = arith.constant 6.4 : f32
  %wot.x.a = arith.constant 0.5 : f32
  %wot.x = call @gen(%i8, %i6, %wot.x.s, %wot.x.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %wot.w.s = arith.constant 7.3 : f32
  %wot.w.a = arith.constant 0.5 : f32
  %wot.w = call @gen(%i8, %i16, %wot.w.s, %wot.w.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %wot.in = call @hcat(%wot.x, %wot.w) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %wot.in4.c = tensor.cast %wot.in : tensor<?x?xf16> to tensor<8x22xf16>
  %wot.in4 = tensor.expand_shape %wot.in4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 22] : tensor<8x22xf16> into tensor<1x8x1x22xf16>
  %wot.l4 = call @ane_wot_dynamic(%wot.in4) : (tensor<1x8x1x22xf16>) -> tensor<1x16x1x6xf16>
  %wot.l.c = tensor.collapse_shape %wot.l4 [[0, 1], [2, 3]] : tensor<1x16x1x6xf16> into tensor<16x6xf16>
  %wot.l = tensor.cast %wot.l.c : tensor<16x6xf16> to tensor<?x?xf16>
  %wot.n = call @linear_t(%wot.w, %wot.x) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "wot_dynamic: "
  call @report(%wot.l, %wot.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %q_bwd.x.s = arith.constant 8.2 : f32
  %q_bwd.x.a = arith.constant 0.5 : f32
  %q_bwd.x = call @gen(%i16, %i6, %q_bwd.x.s, %q_bwd.x.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %q_bwd.w.s = arith.constant 9.1 : f32
  %q_bwd.w.a = arith.constant 0.5 : f32
  %q_bwd.w = call @gen(%i16, %i8, %q_bwd.w.s, %q_bwd.w.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %q_bwd.in = call @hcat(%q_bwd.x, %q_bwd.w) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %q_bwd.in4.c = tensor.cast %q_bwd.in : tensor<?x?xf16> to tensor<16x14xf16>
  %q_bwd.in4 = tensor.expand_shape %q_bwd.in4.c [[0, 1], [2, 3]] output_shape [1, 16, 1, 14] : tensor<16x14xf16> into tensor<1x16x1x14xf16>
  %q_bwd.l4 = call @ane_q_bwd_dynamic(%q_bwd.in4) : (tensor<1x16x1x14xf16>) -> tensor<1x8x1x6xf16>
  %q_bwd.l.c = tensor.collapse_shape %q_bwd.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %q_bwd.l = tensor.cast %q_bwd.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %q_bwd.n = call @linear_t(%q_bwd.w, %q_bwd.x) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "q_bwd_dynamic: "
  call @report(%q_bwd.l, %q_bwd.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %xn.s = arith.constant 10.0 : f32
  %xn.a = arith.constant 1.0 : f32
  %xn = call @gen(%i8, %i6, %xn.s, %xn.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %Wqt.s = arith.constant 10.9 : f32
  %Wqt.a = arith.constant 0.5 : f32
  %Wqt = call @gen(%i8, %i16, %Wqt.s, %Wqt.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %Wkt.s = arith.constant 11.8 : f32
  %Wkt.a = arith.constant 0.5 : f32
  %Wkt = call @gen(%i8, %i8, %Wkt.s, %Wkt.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %Wvt.s = arith.constant 12.7 : f32
  %Wvt.a = arith.constant 0.5 : f32
  %Wvt = call @gen(%i8, %i8, %Wvt.s, %Wvt.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %a.in.0 = call @hcat(%xn, %Wqt) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %a.in.1 = call @hcat(%a.in.0, %Wkt) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %a.in = call @hcat(%a.in.1, %Wvt) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %a.in4.c = tensor.cast %a.in : tensor<?x?xf16> to tensor<8x38xf16>
  %a.in4 = tensor.expand_shape %a.in4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 38] : tensor<8x38xf16> into tensor<1x8x1x38xf16>
  %a.l4 = call @ane_sdpa_fwd_dynamic(%a.in4, %cos4, %sin4, %mask4) : (tensor<1x8x1x38xf16>, tensor<1x1x6x4xf16>, tensor<1x1x6x4xf16>, tensor<1x1x6x6xf16>) -> tensor<1x56x1x6xf16>
  %a.l.c = tensor.collapse_shape %a.l4 [[0, 1], [2, 3]] : tensor<1x56x1x6xf16> into tensor<56x6xf16>
  %a.l = tensor.cast %a.l.c : tensor<56x6xf16> to tensor<?x?xf16>
  %i2 = arith.constant 2 : index
  %a.a, %a.q, %a.k, %a.v = call @nf_sdpa_fwd_gqa(%xn, %Wqt, %Wkt, %Wvt, %cos, %sin, %mask, %i4, %i2, %scale) : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, index, index, f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
  %i0 = arith.constant 0 : index
  %sdpa_fwd_gqa.attn = call @rows(%a.l, %i0, %i16) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd_gqa attn: "
  call @report(%sdpa_fwd_gqa.attn, %a.a) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %sdpa_fwd_gqa.q_rope = call @rows(%a.l, %i16, %i16) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd_gqa q_rope: "
  call @report(%sdpa_fwd_gqa.q_rope, %a.q) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i32 = arith.constant 32 : index
  %sdpa_fwd_gqa.k_rope = call @rows(%a.l, %i32, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd_gqa k_rope: "
  call @report(%sdpa_fwd_gqa.k_rope, %a.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i40 = arith.constant 40 : index
  %sdpa_fwd_gqa.v = call @rows(%a.l, %i40, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd_gqa v: "
  call @report(%sdpa_fwd_gqa.v, %a.v) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i48 = arith.constant 48 : index
  %sdpa_fwd_gqa.xnorm = call @rows(%a.l, %i48, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd_gqa xnorm: "
  call @report(%sdpa_fwd_gqa.xnorm, %xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %x2.s = arith.constant 13.6 : f32
  %x2.a = arith.constant 1.0 : f32
  %x2 = call @gen(%i8, %i6, %x2.s, %x2.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W1t.s = arith.constant 14.5 : f32
  %W1t.a = arith.constant 0.5 : f32
  %W1t = call @gen(%i8, %i12, %W1t.s, %W1t.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W3t.s = arith.constant 15.4 : f32
  %W3t.a = arith.constant 0.5 : f32
  %W3t = call @gen(%i8, %i12, %W3t.s, %W3t.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W2.s = arith.constant 16.3 : f32
  %W2.a = arith.constant 0.5 : f32
  %W2 = call @gen(%i8, %i12, %W2.s, %W2.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %b.in.0 = call @hcat(%xn, %x2) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %b.in.1 = call @hcat(%b.in.0, %W1t) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %b.in.2 = call @hcat(%b.in.1, %W3t) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %b.in = call @hcat(%b.in.2, %W2) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %b.in4.c = tensor.cast %b.in : tensor<?x?xf16> to tensor<8x48xf16>
  %b.in4 = tensor.expand_shape %b.in4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 48] : tensor<8x48xf16> into tensor<1x8x1x48xf16>
  %b.l4 = call @ane_ffn_fused_dynamic(%b.in4) : (tensor<1x8x1x48xf16>) -> tensor<1x44x1x6xf16>
  %b.l.c = tensor.collapse_shape %b.l4 [[0, 1], [2, 3]] : tensor<1x44x1x6xf16> into tensor<44x6xf16>
  %b.l = tensor.cast %b.l.c : tensor<44x6xf16> to tensor<?x?xf16>
  %b.x, %b.h1, %b.h3, %b.g = call @nf_ffn_fused(%xn, %x2, %W1t, %W3t, %W2, %alpha) : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
  %ffn_fused.x_next = call @rows(%b.l, %i0, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fused x_next: "
  call @report(%ffn_fused.x_next, %b.x) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_fused.h1 = call @rows(%b.l, %i8, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fused h1: "
  call @report(%ffn_fused.h1, %b.h1) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i20 = arith.constant 20 : index
  %ffn_fused.h3 = call @rows(%b.l, %i20, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fused h3: "
  call @report(%ffn_fused.h3, %b.h3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_fused.gate = call @rows(%b.l, %i32, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fused gate: "
  call @report(%ffn_fused.gate, %b.g) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %w13t.da.s = arith.constant 17.2 : f32
  %w13t.da.a = arith.constant 0.5 : f32
  %w13t.da = call @gen(%i12, %i6, %w13t.da.s, %w13t.da.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %w13t.db.s = arith.constant 18.1 : f32
  %w13t.db.a = arith.constant 0.5 : f32
  %w13t.db = call @gen(%i12, %i6, %w13t.db.s, %w13t.db.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %w13t.A.s = arith.constant 19.0 : f32
  %w13t.A.a = arith.constant 0.5 : f32
  %w13t.A = call @gen(%i12, %i8, %w13t.A.s, %w13t.A.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %w13t.B.s = arith.constant 19.9 : f32
  %w13t.B.a = arith.constant 0.5 : f32
  %w13t.B = call @gen(%i12, %i8, %w13t.B.s, %w13t.B.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %w13t.in.0 = call @hcat(%w13t.da, %w13t.db) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %w13t.in.1 = call @hcat(%w13t.in.0, %w13t.A) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %w13t.in = call @hcat(%w13t.in.1, %w13t.B) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %w13t.in4.c = tensor.cast %w13t.in : tensor<?x?xf16> to tensor<12x28xf16>
  %w13t.in4 = tensor.expand_shape %w13t.in4.c [[0, 1], [2, 3]] output_shape [1, 12, 1, 28] : tensor<12x28xf16> into tensor<1x12x1x28xf16>
  %w13t.l4 = call @ane_ffn_bwd_w13t_dynamic(%w13t.in4) : (tensor<1x12x1x28xf16>) -> tensor<1x8x1x6xf16>
  %w13t.l.c = tensor.collapse_shape %w13t.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %w13t.l = tensor.cast %w13t.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %w13t.n1 = call @linear_t(%w13t.A, %w13t.da) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %w13t.n2 = call @linear_t(%w13t.B, %w13t.db) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %w13t.n = call @add(%w13t.n1, %w13t.n2) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "ffn_bwd_w13t_dynamic: "
  call @report(%w13t.l, %w13t.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %kvb.da.s = arith.constant 20.8 : f32
  %kvb.da.a = arith.constant 0.5 : f32
  %kvb.da = call @gen(%i8, %i6, %kvb.da.s, %kvb.da.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %kvb.db.s = arith.constant 21.7 : f32
  %kvb.db.a = arith.constant 0.5 : f32
  %kvb.db = call @gen(%i8, %i6, %kvb.db.s, %kvb.db.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %kvb.A.s = arith.constant 22.6 : f32
  %kvb.A.a = arith.constant 0.5 : f32
  %kvb.A = call @gen(%i8, %i8, %kvb.A.s, %kvb.A.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %kvb.B.s = arith.constant 23.5 : f32
  %kvb.B.a = arith.constant 0.5 : f32
  %kvb.B = call @gen(%i8, %i8, %kvb.B.s, %kvb.B.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %kvb.in.0 = call @hcat(%kvb.da, %kvb.db) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kvb.in.1 = call @hcat(%kvb.in.0, %kvb.A) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kvb.in = call @hcat(%kvb.in.1, %kvb.B) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kvb.in4.c = tensor.cast %kvb.in : tensor<?x?xf16> to tensor<8x28xf16>
  %kvb.in4 = tensor.expand_shape %kvb.in4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 28] : tensor<8x28xf16> into tensor<1x8x1x28xf16>
  %kvb.l4 = call @ane_kv_bwd_dynamic(%kvb.in4) : (tensor<1x8x1x28xf16>) -> tensor<1x8x1x6xf16>
  %kvb.l.c = tensor.collapse_shape %kvb.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %kvb.l = tensor.cast %kvb.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %kvb.n1 = call @linear_t(%kvb.A, %kvb.da) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kvb.n2 = call @linear_t(%kvb.B, %kvb.db) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %kvb.n = call @add(%kvb.n1, %kvb.n2) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "kv_bwd_dynamic: "
  call @report(%kvb.l, %kvb.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %s.q.s = arith.constant 24.4 : f32
  %s.q.a = arith.constant 0.5 : f32
  %s.q = call @gen(%i16, %i6, %s.q.s, %s.q.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %s.k.s = arith.constant 25.3 : f32
  %s.k.a = arith.constant 0.5 : f32
  %s.k = call @gen(%i16, %i6, %s.k.s, %s.k.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %s.v.s = arith.constant 26.2 : f32
  %s.v.a = arith.constant 0.5 : f32
  %s.v = call @gen(%i16, %i6, %s.v.s, %s.v.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %s.da.s = arith.constant 27.1 : f32
  %s.da.a = arith.constant 0.5 : f32
  %s.da = call @gen(%i16, %i6, %s.da.s, %s.da.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %s.in.0 = call @vcat(%s.q, %s.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %s.in.1 = call @vcat(%s.in.0, %s.v) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %s.in = call @vcat(%s.in.1, %s.da) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %s.in4.c = tensor.cast %s.in : tensor<?x?xf16> to tensor<64x6xf16>
  %s.in4 = tensor.expand_shape %s.in4.c [[0, 1], [2, 3]] output_shape [1, 64, 1, 6] : tensor<64x6xf16> into tensor<1x64x1x6xf16>
  %s.l4 = call @ane_sdpa_bwd1_noweight(%s.in4, %mask4) : (tensor<1x64x1x6xf16>, tensor<1x1x6x6xf16>) -> tensor<1x64x1x6xf16>
  %s.l.c = tensor.collapse_shape %s.l4 [[0, 1], [2, 3]] : tensor<1x64x1x6xf16> into tensor<64x6xf16>
  %s.l = tensor.cast %s.l.c : tensor<64x6xf16> to tensor<?x?xf16>
  %s.dv = call @rows(%s.l, %i0, %i16) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  %s.pdp = call @rows(%s.l, %i16, %i48) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  %u.in.0 = call @vcat(%s.pdp, %s.q) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %u.in = call @vcat(%u.in.0, %s.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %u.in4.c = tensor.cast %u.in : tensor<?x?xf16> to tensor<80x6xf16>
  %u.in4 = tensor.expand_shape %u.in4.c [[0, 1], [2, 3]] output_shape [1, 80, 1, 6] : tensor<80x6xf16> into tensor<1x80x1x6xf16>
  %u.l4 = call @ane_sdpa_bwd2(%u.in4) : (tensor<1x80x1x6xf16>) -> tensor<1x32x1x6xf16>
  %u.l.c = tensor.collapse_shape %u.l4 [[0, 1], [2, 3]] : tensor<1x32x1x6xf16> into tensor<32x6xf16>
  %u.l = tensor.cast %u.l.c : tensor<32x6xf16> to tensor<?x?xf16>
  %s.qh = call @split_heads(%s.q, %i4) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %s.kh = call @split_heads(%s.k, %i4) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %s.vh = call @split_heads(%s.v, %i4) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %s.dah = call @split_heads(%s.da, %i4) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %u.dq3, %u.dk3, %u.dv3 = call @sdpa_bwd(%s.qh, %s.kh, %s.vh, %mask, %scale, %s.dah) : (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?xf16>, f16, tensor<?x?x?xf16>) -> (tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>)
  %u.dq = call @merge_heads(%u.dq3) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %u.dk = call @merge_heads(%u.dk3) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %u.dv = call @merge_heads(%u.dv3) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  vector.print str "sdpa_bwd dv: "
  call @report(%s.dv, %u.dv) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %sdpa_bwd.dq = call @rows(%u.l, %i0, %i16) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_bwd dq: "
  call @report(%sdpa_bwd.dq, %u.dq) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %sdpa_bwd.dk = call @rows(%u.l, %i16, %i16) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_bwd dk: "
  call @report(%sdpa_bwd.dk, %u.dk) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %g.k.s = arith.constant 28.0 : f32
  %g.k.a = arith.constant 0.5 : f32
  %g.k = call @gen(%i8, %i6, %g.k.s, %g.k.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %g.kh = call @split_heads(%g.k, %i2) : (tensor<?x?xf16>, index) -> tensor<?x?x?xf16>
  %g.mod3 = call @gqa_tile_mod(%g.kh, %i4) : (tensor<?x?x?xf16>, index) -> tensor<?x?x?xf16>
  %g.mod = call @merge_heads(%g.mod3) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %g.div3 = call @gqa_tile_div(%g.kh, %i4) : (tensor<?x?x?xf16>, index) -> tensor<?x?x?xf16>
  %g.div = call @merge_heads(%g.div3) : (tensor<?x?x?xf16>) -> tensor<?x?xf16>
  %g.cat = call @vcat(%g.k, %g.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "gqa concat(k,k) vs tile_mod: "
  call @report(%g.cat, %g.mod) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  vector.print str "gqa tile_mod vs tile_div: "
  call @report(%g.mod, %g.div) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  return
}

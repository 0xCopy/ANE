// Static-weight kernels (training/stories_mil.h, ane_rmsnorm_bwd.h, ane_classifier.h) against their normal forms.
//
// Sizes: DIM=8 HIDDEN=12 HEADS=2 HD=4 SEQ=6 VOCAB=10 (the generators re-run with small #defines).
// maxdiff 0 means bit-identical f16 results.  Where the normal form changes the order of the f16 arithmetic
// (rsqrt for pow(-0.5); one fused loop for a chain of mul/sub) the check is a bound, and the measured value is printed.
//
// RUN: cat %S/harness.mlir %S/../normal/ane_normal.mlir %S/small/static.mlir %s | mlir-opt \
// RUN:   --transform-preload-library=transform-library-paths=%S/decompose_softmax.mlir --transform-interpreter \
// RUN:   --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops \
// RUN:   --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
// RUN:   --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:   --convert-cf-to-llvm --reconcile-unrealized-casts \
// RUN: | mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%llvmlib/libmlir_runner_utils.dylib,%llvmlib/libmlir_c_runner_utils.dylib | FileCheck %s

// CHECK: sdpa_fwd o: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: sdpa_fwd q: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: sdpa_fwd k: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: sdpa_fwd v: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: sdpa_fwd attn: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: sdpa_fwd xnorm: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: ffn_fwd y: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: ffn_fwd h1: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: ffn_fwd h3: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: ffn_fwd gate: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: ffn_fwd xnorm: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: ffn_bwd dx: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_bwd dh1: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: ffn_bwd dh3: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_bwd dv: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_bwd dq: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: sdpa_bwd dk: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: qkv_bwd dx: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: rmsnorm_bwd dx: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: final_rmsnorm: maxabs {{[0-9.e+-]+}} maxdiff {{0(\.00[0-9]+)?$}}
// CHECK: classifier_fwd: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: classifier_bwd: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}
// CHECK: softmax_vocab: maxabs {{[0-9.e+-]+}} maxdiff 0{{$}}

func.func @main() {
  %eps = arith.constant 1.0e-05 : f16
  %scale = arith.constant 0.5 : f16
  %i6 = arith.constant 6 : index
  %mask = call @causal_mask(%i6) : (index) -> tensor<?x?xf16>
  %mask4.c = tensor.cast %mask : tensor<?x?xf16> to tensor<6x6xf16>
  %mask4 = tensor.expand_shape %mask4.c [[0, 1, 2], [3]] output_shape [1, 1, 6, 6] : tensor<6x6xf16> into tensor<1x1x6x6xf16>
  %X.s = arith.constant 1.0 : f32
  %X.a = arith.constant 1.0 : f32
  %i8 = arith.constant 8 : index
  %X = call @gen(%i8, %i6, %X.s, %X.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %rw.m.s = arith.constant 1.9 : f32
  %rw.m.a = arith.constant 1.0 : f32
  %i1 = arith.constant 1 : index
  %rw.m = call @gen(%i8, %i1, %rw.m.s, %rw.m.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %rw = tensor.collapse_shape %rw.m [[0, 1]] : tensor<?x?xf16> into tensor<?xf16>
  %Wq.s = arith.constant 2.8 : f32
  %Wq.a = arith.constant 0.5 : f32
  %Wq = call @gen(%i8, %i8, %Wq.s, %Wq.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %Wk.s = arith.constant 3.7 : f32
  %Wk.a = arith.constant 0.5 : f32
  %Wk = call @gen(%i8, %i8, %Wk.s, %Wk.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %Wv.s = arith.constant 4.6 : f32
  %Wv.a = arith.constant 0.5 : f32
  %Wv = call @gen(%i8, %i8, %Wv.s, %Wv.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %Wo.s = arith.constant 5.5 : f32
  %Wo.a = arith.constant 0.5 : f32
  %Wo = call @gen(%i8, %i8, %Wo.s, %Wo.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W1.s = arith.constant 6.4 : f32
  %W1.a = arith.constant 0.5 : f32
  %i12 = arith.constant 12 : index
  %W1 = call @gen(%i12, %i8, %W1.s, %W1.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W3.s = arith.constant 7.3 : f32
  %W3.a = arith.constant 0.5 : f32
  %W3 = call @gen(%i12, %i8, %W3.s, %W3.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W2.s = arith.constant 8.2 : f32
  %W2.a = arith.constant 0.5 : f32
  %W2 = call @gen(%i8, %i12, %W2.s, %W2.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %We.s = arith.constant 9.1 : f32
  %We.a = arith.constant 0.5 : f32
  %i10 = arith.constant 10 : index
  %We = call @gen(%i10, %i8, %We.s, %We.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %X4.c = tensor.cast %X : tensor<?x?xf16> to tensor<8x6xf16>
  %X4 = tensor.expand_shape %X4.c [[0, 1], [2, 3]] output_shape [1, 8, 1, 6] : tensor<8x6xf16> into tensor<1x8x1x6xf16>
  %rw4.c = tensor.cast %rw : tensor<?xf16> to tensor<8xf16>
  %rw4 = tensor.expand_shape %rw4.c [[0, 1, 2, 3]] output_shape [1, 8, 1, 1] : tensor<8xf16> into tensor<1x8x1x1xf16>
  %Wq4.c = tensor.cast %Wq : tensor<?x?xf16> to tensor<8x8xf16>
  %Wq4 = tensor.expand_shape %Wq4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %Wk4.c = tensor.cast %Wk : tensor<?x?xf16> to tensor<8x8xf16>
  %Wk4 = tensor.expand_shape %Wk4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %Wv4.c = tensor.cast %Wv : tensor<?x?xf16> to tensor<8x8xf16>
  %Wv4 = tensor.expand_shape %Wv4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %Wo4.c = tensor.cast %Wo : tensor<?x?xf16> to tensor<8x8xf16>
  %Wo4 = tensor.expand_shape %Wo4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %W14.c = tensor.cast %W1 : tensor<?x?xf16> to tensor<12x8xf16>
  %W14 = tensor.expand_shape %W14.c [[0], [1, 2, 3]] output_shape [12, 8, 1, 1] : tensor<12x8xf16> into tensor<12x8x1x1xf16>
  %W34.c = tensor.cast %W3 : tensor<?x?xf16> to tensor<12x8xf16>
  %W34 = tensor.expand_shape %W34.c [[0], [1, 2, 3]] output_shape [12, 8, 1, 1] : tensor<12x8xf16> into tensor<12x8x1x1xf16>
  %W24.c = tensor.cast %W2 : tensor<?x?xf16> to tensor<8x12xf16>
  %W24 = tensor.expand_shape %W24.c [[0], [1, 2, 3]] output_shape [8, 12, 1, 1] : tensor<8x12xf16> into tensor<8x12x1x1xf16>
  %We4.c = tensor.cast %We : tensor<?x?xf16> to tensor<10x8xf16>
  %We4 = tensor.expand_shape %We4.c [[0], [1, 2, 3]] output_shape [10, 8, 1, 1] : tensor<10x8xf16> into tensor<10x8x1x1xf16>
  %a.l4 = call @ane_sdpa_fwd_taps(%X4, %rw4, %Wq4, %Wk4, %Wv4, %Wo4, %mask4) : (tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>, tensor<8x8x1x1xf16>, tensor<8x8x1x1xf16>, tensor<8x8x1x1xf16>, tensor<8x8x1x1xf16>, tensor<1x1x6x6xf16>) -> tensor<1x48x1x6xf16>
  %a.l.c = tensor.collapse_shape %a.l4 [[0, 1], [2, 3]] : tensor<1x48x1x6xf16> into tensor<48x6xf16>
  %a.l = tensor.cast %a.l.c : tensor<48x6xf16> to tensor<?x?xf16>
  %i2 = arith.constant 2 : index
  %a.o, %a.q, %a.k, %a.v, %a.a, %a.xn = call @nf_sdpa_fwd(%X, %rw, %Wq, %Wk, %Wv, %Wo, %mask, %i2, %eps, %scale) : (tensor<?x?xf16>, tensor<?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, index, f16, f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
  %i0 = arith.constant 0 : index
  %sdpa_fwd.o = call @rows(%a.l, %i0, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd o: "
  call @report(%sdpa_fwd.o, %a.o) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %sdpa_fwd.q = call @rows(%a.l, %i8, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd q: "
  call @report(%sdpa_fwd.q, %a.q) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i16 = arith.constant 16 : index
  %sdpa_fwd.k = call @rows(%a.l, %i16, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd k: "
  call @report(%sdpa_fwd.k, %a.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i24 = arith.constant 24 : index
  %sdpa_fwd.v = call @rows(%a.l, %i24, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd v: "
  call @report(%sdpa_fwd.v, %a.v) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i32 = arith.constant 32 : index
  %sdpa_fwd.attn = call @rows(%a.l, %i32, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd attn: "
  call @report(%sdpa_fwd.attn, %a.a) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i40 = arith.constant 40 : index
  %sdpa_fwd.xnorm = call @rows(%a.l, %i40, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_fwd xnorm: "
  call @report(%sdpa_fwd.xnorm, %a.xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %b.l4 = call @ane_ffn_fwd_taps(%X4, %rw4, %W14, %W34, %W24) : (tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>, tensor<12x8x1x1xf16>, tensor<12x8x1x1xf16>, tensor<8x12x1x1xf16>) -> tensor<1x52x1x6xf16>
  %b.l.c = tensor.collapse_shape %b.l4 [[0, 1], [2, 3]] : tensor<1x52x1x6xf16> into tensor<52x6xf16>
  %b.l = tensor.cast %b.l.c : tensor<52x6xf16> to tensor<?x?xf16>
  %b.y, %b.h1, %b.h3, %b.g, %b.xn = call @nf_ffn_fwd(%X, %rw, %W1, %W3, %W2, %eps) : (tensor<?x?xf16>, tensor<?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
  %ffn_fwd.y = call @rows(%b.l, %i0, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fwd y: "
  call @report(%ffn_fwd.y, %b.y) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_fwd.h1 = call @rows(%b.l, %i8, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fwd h1: "
  call @report(%ffn_fwd.h1, %b.h1) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i20 = arith.constant 20 : index
  %ffn_fwd.h3 = call @rows(%b.l, %i20, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fwd h3: "
  call @report(%ffn_fwd.h3, %b.h3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_fwd.gate = call @rows(%b.l, %i32, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fwd gate: "
  call @report(%ffn_fwd.gate, %b.g) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %i44 = arith.constant 44 : index
  %ffn_fwd.xnorm = call @rows(%b.l, %i44, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_fwd xnorm: "
  call @report(%ffn_fwd.xnorm, %b.xn) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %dy.s = arith.constant 10.0 : f32
  %dy.a = arith.constant 0.5 : f32
  %dy = call @gen(%i8, %i6, %dy.s, %dy.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %W1t = call @transpose(%W1) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %W2t = call @transpose(%W2) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %W3t = call @transpose(%W3) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %Wqt = call @transpose(%Wq) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %Wkt = call @transpose(%Wk) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %Wvt = call @transpose(%Wv) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %Wot = call @transpose(%Wo) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %W2t4.c = tensor.cast %W2t : tensor<?x?xf16> to tensor<12x8xf16>
  %W2t4 = tensor.expand_shape %W2t4.c [[0], [1, 2, 3]] output_shape [12, 8, 1, 1] : tensor<12x8xf16> into tensor<12x8x1x1xf16>
  %W1t4.c = tensor.cast %W1t : tensor<?x?xf16> to tensor<8x12xf16>
  %W1t4 = tensor.expand_shape %W1t4.c [[0], [1, 2, 3]] output_shape [8, 12, 1, 1] : tensor<8x12xf16> into tensor<8x12x1x1xf16>
  %W3t4.c = tensor.cast %W3t : tensor<?x?xf16> to tensor<8x12xf16>
  %W3t4 = tensor.expand_shape %W3t4.c [[0], [1, 2, 3]] output_shape [8, 12, 1, 1] : tensor<8x12xf16> into tensor<8x12x1x1xf16>
  %c.in.0 = call @vcat(%dy, %b.h1) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %c.in = call @vcat(%c.in.0, %b.h3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %c.in4.c = tensor.cast %c.in : tensor<?x?xf16> to tensor<32x6xf16>
  %c.in4 = tensor.expand_shape %c.in4.c [[0, 1], [2, 3]] output_shape [1, 32, 1, 6] : tensor<32x6xf16> into tensor<1x32x1x6xf16>
  %c.l4 = call @ane_ffn_bwd(%c.in4, %W2t4, %W1t4, %W3t4) : (tensor<1x32x1x6xf16>, tensor<12x8x1x1xf16>, tensor<8x12x1x1xf16>, tensor<8x12x1x1xf16>) -> tensor<1x32x1x6xf16>
  %c.l.c = tensor.collapse_shape %c.l4 [[0, 1], [2, 3]] : tensor<1x32x1x6xf16> into tensor<32x6xf16>
  %c.l = tensor.cast %c.l.c : tensor<32x6xf16> to tensor<?x?xf16>
  %c.dx, %c.dh1, %c.dh3 = call @nf_ffn_bwd(%dy, %b.h1, %b.h3, %W1, %W3, %W2) : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
  %ffn_bwd.dx = call @rows(%c.l, %i0, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_bwd dx: "
  call @report(%ffn_bwd.dx, %c.dx) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_bwd.dh1 = call @rows(%c.l, %i8, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_bwd dh1: "
  call @report(%ffn_bwd.dh1, %c.dh1) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %ffn_bwd.dh3 = call @rows(%c.l, %i20, %i12) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "ffn_bwd dh3: "
  call @report(%ffn_bwd.dh3, %c.dh3) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %do.s = arith.constant 10.9 : f32
  %do.a = arith.constant 0.5 : f32
  %do = call @gen(%i8, %i6, %do.s, %do.a) : (index, index, f32, f32) -> tensor<?x?xf16>
  %d.in.0 = call @vcat(%a.q, %a.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %d.in.1 = call @vcat(%d.in.0, %a.v) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %d.in = call @vcat(%d.in.1, %do) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %d.in4.c = tensor.cast %d.in : tensor<?x?xf16> to tensor<32x6xf16>
  %d.in4 = tensor.expand_shape %d.in4.c [[0, 1], [2, 3]] output_shape [1, 32, 1, 6] : tensor<32x6xf16> into tensor<1x32x1x6xf16>
  %Wot4.c = tensor.cast %Wot : tensor<?x?xf16> to tensor<8x8xf16>
  %Wot4 = tensor.expand_shape %Wot4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %d.l4 = call @ane_sdpa_bwd1(%d.in4, %Wot4, %mask4) : (tensor<1x32x1x6xf16>, tensor<8x8x1x1xf16>, tensor<1x1x6x6xf16>) -> tensor<1x32x1x6xf16>
  %d.l.c = tensor.collapse_shape %d.l4 [[0, 1], [2, 3]] : tensor<1x32x1x6xf16> into tensor<32x6xf16>
  %d.l = tensor.cast %d.l.c : tensor<32x6xf16> to tensor<?x?xf16>
  %d.dv = call @rows(%d.l, %i0, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  %d.pdp = call @rows(%d.l, %i8, %i24) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  %e.in.0 = call @vcat(%d.pdp, %a.q) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %e.in = call @vcat(%e.in.0, %a.k) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %e.in4.c = tensor.cast %e.in : tensor<?x?xf16> to tensor<40x6xf16>
  %e.in4 = tensor.expand_shape %e.in4.c [[0, 1], [2, 3]] output_shape [1, 40, 1, 6] : tensor<40x6xf16> into tensor<1x40x1x6xf16>
  %e.l4 = call @ane_sdpa_bwd2(%e.in4) : (tensor<1x40x1x6xf16>) -> tensor<1x16x1x6xf16>
  %e.l.c = tensor.collapse_shape %e.l4 [[0, 1], [2, 3]] : tensor<1x16x1x6xf16> into tensor<16x6xf16>
  %e.l = tensor.cast %e.l.c : tensor<16x6xf16> to tensor<?x?xf16>
  %e.dq, %e.dk, %e.dv = call @nf_sdpa_bwd(%a.q, %a.k, %a.v, %do, %Wo, %mask, %i2, %scale) : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, index, f16) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>)
  vector.print str "sdpa_bwd dv: "
  call @report(%d.dv, %e.dv) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %sdpa_bwd.dq = call @rows(%e.l, %i0, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_bwd dq: "
  call @report(%sdpa_bwd.dq, %e.dq) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %sdpa_bwd.dk = call @rows(%e.l, %i8, %i8) : (tensor<?x?xf16>, index, index) -> tensor<?x?xf16>
  vector.print str "sdpa_bwd dk: "
  call @report(%sdpa_bwd.dk, %e.dk) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %f.in.0 = call @vcat(%e.dq, %e.dk) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %f.in = call @vcat(%f.in.0, %e.dv) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %f.in4.c = tensor.cast %f.in : tensor<?x?xf16> to tensor<24x6xf16>
  %f.in4 = tensor.expand_shape %f.in4.c [[0, 1], [2, 3]] output_shape [1, 24, 1, 6] : tensor<24x6xf16> into tensor<1x24x1x6xf16>
  %Wqt4.c = tensor.cast %Wqt : tensor<?x?xf16> to tensor<8x8xf16>
  %Wqt4 = tensor.expand_shape %Wqt4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %Wkt4.c = tensor.cast %Wkt : tensor<?x?xf16> to tensor<8x8xf16>
  %Wkt4 = tensor.expand_shape %Wkt4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %Wvt4.c = tensor.cast %Wvt : tensor<?x?xf16> to tensor<8x8xf16>
  %Wvt4 = tensor.expand_shape %Wvt4.c [[0], [1, 2, 3]] output_shape [8, 8, 1, 1] : tensor<8x8xf16> into tensor<8x8x1x1xf16>
  %f.l4 = call @ane_qkvb(%f.in4, %Wqt4, %Wkt4, %Wvt4) : (tensor<1x24x1x6xf16>, tensor<8x8x1x1xf16>, tensor<8x8x1x1xf16>, tensor<8x8x1x1xf16>) -> tensor<1x8x1x6xf16>
  %f.l.c = tensor.collapse_shape %f.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %f.l = tensor.cast %f.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %f.n = call @nf_qkv_bwd(%e.dq, %e.dk, %e.dv, %Wq, %Wk, %Wv) : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "qkv_bwd dx: "
  call @report(%f.l, %f.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %g.in = call @vcat(%dy, %X) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  %g.in4.c = tensor.cast %g.in : tensor<?x?xf16> to tensor<16x6xf16>
  %g.in4 = tensor.expand_shape %g.in4.c [[0, 1], [2, 3]] output_shape [1, 16, 1, 6] : tensor<16x6xf16> into tensor<1x16x1x6xf16>
  %g.l4 = call @ane_rmsnorm_bwd(%g.in4, %rw4) : (tensor<1x16x1x6xf16>, tensor<1x8x1x1xf16>) -> tensor<1x8x1x6xf16>
  %g.l.c = tensor.collapse_shape %g.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %g.l = tensor.cast %g.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %g.n = call @rmsnorm_bwd(%dy, %X, %rw, %eps) : (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?xf16>, f16) -> tensor<?x?xf16>
  vector.print str "rmsnorm_bwd dx: "
  call @report(%g.l, %g.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %h.l4 = call @ane_final_rmsnorm(%X4, %rw4) : (tensor<1x8x1x6xf16>, tensor<1x8x1x1xf16>) -> tensor<1x8x1x6xf16>
  %h.l.c = tensor.collapse_shape %h.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %h.l = tensor.cast %h.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %h.n = call @rmsnorm(%X, %rw, %eps) : (tensor<?x?xf16>, tensor<?xf16>, f16) -> tensor<?x?xf16>
  vector.print str "final_rmsnorm: "
  call @report(%h.l, %h.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %k.l4 = call @ane_classifier_fwd(%X4, %We4) : (tensor<1x8x1x6xf16>, tensor<10x8x1x1xf16>) -> tensor<1x10x1x6xf16>
  %k.l.c = tensor.collapse_shape %k.l4 [[0, 1], [2, 3]] : tensor<1x10x1x6xf16> into tensor<10x6xf16>
  %k.l = tensor.cast %k.l.c : tensor<10x6xf16> to tensor<?x?xf16>
  %k.n = call @linear(%We, %X) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "classifier_fwd: "
  call @report(%k.l, %k.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %Wet = call @transpose(%We) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  %Wet3.c = tensor.cast %Wet : tensor<?x?xf16> to tensor<8x10xf16>
  %Wet3 = tensor.expand_shape %Wet3.c [[0, 1], [2]] output_shape [1, 8, 10] : tensor<8x10xf16> into tensor<1x8x10xf16>
  %m.l4 = call @ane_classifier_bwd(%k.l4, %Wet3) : (tensor<1x10x1x6xf16>, tensor<1x8x10xf16>) -> tensor<1x8x1x6xf16>
  %m.l.c = tensor.collapse_shape %m.l4 [[0, 1], [2, 3]] : tensor<1x8x1x6xf16> into tensor<8x6xf16>
  %m.l = tensor.cast %m.l.c : tensor<8x6xf16> to tensor<?x?xf16>
  %m.n = call @linear_t(%We, %k.l) : (tensor<?x?xf16>, tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "classifier_bwd: "
  call @report(%m.l, %m.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  %n.l4 = call @ane_softmax_vocab(%k.l4) : (tensor<1x10x1x6xf16>) -> tensor<1x10x1x6xf16>
  %n.l.c = tensor.collapse_shape %n.l4 [[0, 1], [2, 3]] : tensor<1x10x1x6xf16> into tensor<10x6xf16>
  %n.l = tensor.cast %n.l.c : tensor<10x6xf16> to tensor<?x?xf16>
  %n.n = call @softmax_ch(%k.l) : (tensor<?x?xf16>) -> tensor<?x?xf16>
  vector.print str "softmax_vocab: "
  call @report(%n.l, %n.n) : (tensor<?x?xf16>, tensor<?x?xf16>) -> ()
  return
}

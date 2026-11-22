// io.h: cvt_f32_f16 / cvt_f16_f32 (the only hand-written NEON in the repo) in the vector dialect, and the f16
// arithmetic the ANE kernels become on the CPU.
//
//   C (arm_neon.h)                                   MLIR                                      AArch64
//   vld1q_f32 x2, vcvt_f16_f32 x2, vcombine_f16,     vector.load <8 x f32>, arith.truncf,      fcvtn + fcvtn2, one store
//   vst1q_f16                                        vector.store <8 x f16>
//   vld1q_f16, vget_low/high, vcvt_f32_f16 x2,       vector.load <8 x f16>, arith.extf,        fcvtl + fcvtl2, two stores
//   vst1q_f32 x2                                     vector.store <8 x f32>
//
// The 4+4 split in the C source is register width, not meaning: <8 x f32> is two q registers and the backend splits it.
// The scalar tail loop is kept; `(_Float16)src[i]` is the same arith.truncf on a scalar (fcvt h, s).

func.func @cvt_f32_f16(%dst: memref<?xf16>, %src: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %rem = arith.remui %n, %c8 : index
  %n8 = arith.subi %n, %rem : index
  scf.for %i = %c0 to %n8 step %c8 {
    %v = vector.load %src[%i] : memref<?xf32>, vector<8xf32>
    %h = arith.truncf %v : vector<8xf32> to vector<8xf16>
    vector.store %h, %dst[%i] : memref<?xf16>, vector<8xf16>
  }
  scf.for %i = %n8 to %n step %c1 {
    %x = memref.load %src[%i] : memref<?xf32>
    %h = arith.truncf %x : f32 to f16
    memref.store %h, %dst[%i] : memref<?xf16>
  }
  return
}

func.func @cvt_f16_f32(%dst: memref<?xf32>, %src: memref<?xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %rem = arith.remui %n, %c8 : index
  %n8 = arith.subi %n, %rem : index
  scf.for %i = %c0 to %n8 step %c8 {
    %h = vector.load %src[%i] : memref<?xf16>, vector<8xf16>
    %v = arith.extf %h : vector<8xf16> to vector<8xf32>
    vector.store %v, %dst[%i] : memref<?xf32>, vector<8xf32>
  }
  scf.for %i = %n8 to %n step %c1 {
    %x = memref.load %src[%i] : memref<?xf16>
    %v = arith.extf %x : f16 to f32
    memref.store %v, %dst[%i] : memref<?xf32>
  }
  return
}

// One step of an f16 inner product, 8 lanes: acc + a * b.
//   +fullfp16 (ARMv8.2-A, every Apple Silicon Mac): fmul/fadd (or fmla) on v.8h
//   -fullfp16 (baseline ARMv8.0 has f16 only as a storage format): each lane is widened, computed in f32, narrowed
// arith.mulf + arith.addf is not a fused multiply-add: fusing needs the `contract` fastmath flag (see @fma_step).
func.func @mla_step(%acc: vector<8xf16>, %a: vector<8xf16>, %b: vector<8xf16>) -> vector<8xf16> {
  %m = arith.mulf %a, %b : vector<8xf16>
  %r = arith.addf %acc, %m : vector<8xf16>
  return %r : vector<8xf16>
}

func.func @fma_step(%acc: vector<8xf16>, %a: vector<8xf16>, %b: vector<8xf16>) -> vector<8xf16> {
  %r = math.fma %a, %b, %acc : vector<8xf16>
  return %r : vector<8xf16>
}

// quantize (W8A8), 8 lanes, as the device does it (README "Sharp corners"): sat_i8(round_half_away(x * f16(1 / scale))).
//   The reciprocal is taken once in fp16 and multiplied; math.round (ties away from zero) -> frinta.  arith.fptosi is
//   poison out of range, so the clamp comes first (fmaxnm/fminnm); the AArch64 instruction it becomes (fcvtzs) would
//   have saturated on its own, but the IR may not rely on that.
func.func @quantize8(%x: vector<8xf16>, %scale: f16) -> vector<8xi8> {
  %one = arith.constant 1.0 : f16
  %inv = arith.divf %one, %scale : f16
  %s = vector.broadcast %inv : f16 to vector<8xf16>
  %lo = arith.constant dense<-128.0> : vector<8xf16>
  %hi = arith.constant dense<127.0> : vector<8xf16>
  %q = arith.mulf %x, %s : vector<8xf16>
  %rn = math.round %q : vector<8xf16>
  %a = arith.maxnumf %rn, %lo : vector<8xf16>
  %b = arith.minnumf %a, %hi : vector<8xf16>
  %r = arith.fptosi %b : vector<8xf16> to vector<8xi8>
  return %r : vector<8xi8>
}

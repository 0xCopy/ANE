#include "mil_dynamic.h"
// With a directory argument, <dir>/<name>.mil replaces the generator's text (MIL printed from the ane dialect).
static const char *g_dir;
static void run(const char *name, NSString *mil, NSDictionary *w, int ic, int sp, int oc, int osp, float seed) {
  if (g_dir) { mil = [NSString stringWithContentsOfFile:[NSString stringWithFormat:@"%s/%s.mil", g_dir, name] encoding:NSUTF8StringEncoding error:nil];
    if (!mil) { printf("KERNEL %s NO_FILE\n", name); return; } }
  Kern *k = compile_kern_mil_w(mil, w, ic*sp*2, oc*osp*2);
  if (!k) { printf("KERNEL %s COMPILE_FAIL\n", name); return; }
  float *in = malloc(ic*sp*4);
  for (int i=0;i<ic;i++) for (int j=0;j<sp;j++) in[i*sp+j] = sinf(0.7f*i + 1.3f*j + seed) * 0.5f;
  io_write_fp16(k->ioIn, in, ic, sp);
  { id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    fprintf(stderr, "%s eval ok=%d %s\n", name, ok, e ? [[e description] UTF8String] : ""); }
  float *out = malloc(oc*osp*4);
  io_read_fp16(k->ioOut, out, 0, oc, osp);
  printf("KERNEL %s %d %d\n", name, oc, osp);
  for (int i=0;i<oc*osp;i++) printf("%.9g\n", out[i]);
  free_kern(k);
}
int main(int argc, char **argv){@autoreleasepool{
  g_dir = argc > 1 ? argv[1] : NULL;
  ane_init();
  NSDictionary *mask = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()}};
  NSDictionary *sd = @{@"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
                       @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
                       @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}};
  run("dyn_matmul_mil", gen_dyn_matmul_mil(DIM,HIDDEN,SEQ), @{}, DIM, SEQ+HIDDEN, HIDDEN, SEQ, 0.3f);
  run("sdpa_fwd_dynamic", gen_sdpa_fwd_dynamic(), sd, DIM, SEQ+Q_DIM+2*KV_DIM, 2*Q_DIM+2*KV_DIM+DIM, SEQ, 0.3f);
  run("ffn_fused_dynamic", gen_ffn_fused_dynamic(), @{}, DIM, 2*SEQ+3*HIDDEN, DIM+3*HIDDEN, SEQ, 0.3f);
  run("ffn_bwd_w13t_dynamic", gen_ffn_bwd_w13t_dynamic(), @{}, HIDDEN, 2*SEQ+2*DIM, DIM, SEQ, 0.3f);
  run("sdpa_bwd1_noweight", gen_sdpa_bwd1_noweight(), mask, 4*Q_DIM, SEQ, Q_DIM+2*SCORE_CH, SEQ, 0.3f);
  run("sdpa_bwd2", gen_sdpa_bwd2(), @{}, 2*SCORE_CH+2*Q_DIM, SEQ, 2*Q_DIM, SEQ, 0.3f);
}return 0;}

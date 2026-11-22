// ane_try.m: compile and evaluate one MIL program on the ANE through the repo's own path (training_dynamic/io.h).
// usage: ane_try FILE.mil IN_BYTES OUT_BYTES f16|f32 f16|f32 [WEIGHT.BIN | WEIGHT_DIR]
//        ane_try dynmm:IC,OC,SEQ IN_BYTES ...     the program is gen_dyn_matmul_mil(IC, OC, SEQ), not a file
//   IN_BYTES / OUT_BYTES size the two IOSurfaces; f16|f32 is the element type written to / read from them.
//   WEIGHT.BIN becomes @model_path/weights/weight.bin; a WEIGHT_DIR gives every file in it its own @model_path/weights/NAME
//   and prints `WEIGHTS enum=a.bin,b.bin` (the dictionary's enumeration order) before compiling.
// env:   ANE_IN=FILE     bytes copied to the input surface (default: a sine pattern, dense)
//        ANE_DUMP=PFX    writes the input surface to PFX.in and the output surface to PFX.out
//        ANE_ATTR=1      prints the compiled model's attributes (symbols, strides, sizes)
// build: xcrun clang -w -fobjc-arc -framework Foundation -framework IOSurface -framework Accelerate -framework CoreML \
//          -DACCELERATE_NEW_LAPACK -Itraining/training_dynamic -include mlir/tools/mid_dynamic.h -o ane_try mlir/tools/ane_try.m
// stdout: RESULT compile=OK|FAIL eval=0|1 err=... nonfinite=N maxabs=X nonzero=N/M fnv=HEX   (the framework's compile
//         error is printed by compile_kern_mil_w before it)
// tools/ane_limits.py drives this to re-measure the device facts in README.md on a new OS or chip.
#include "mil_dynamic.h"
int main(int argc, char **argv) { @autoreleasepool {
  if (argc < 6) { fprintf(stderr, "usage: ane_try FILE.mil|dynmm:IC,OC,SEQ in_bytes out_bytes f16|f32 f16|f32 [weight.bin]\n"); return 64; }
  ane_init();
  NSString *mil;
  if (!strncmp(argv[1], "dynmm:", 6)) { int a = 0, b = 0, c = 0; sscanf(argv[1] + 6, "%d,%d,%d", &a, &b, &c); mil = gen_dyn_matmul_mil(a, b, c); }
  else mil = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:nil];
  if (!mil) { printf("RESULT compile=NOFILE\n"); return 65; }
  size_t ib = strtoull(argv[2], 0, 10), ob = strtoull(argv[3], 0, 10);
  int in32 = !strcmp(argv[4], "f32"), out32 = !strcmp(argv[5], "f32");
  NSDictionary *w = @{};
  if (argc > 6) { BOOL dir = NO; [[NSFileManager defaultManager] fileExistsAtPath:@(argv[6]) isDirectory:&dir];
    if (dir) { NSMutableDictionary *m = [NSMutableDictionary dictionary]; NSString *d0 = @(argv[6]);
      for (NSString *f in [[NSFileManager defaultManager] contentsOfDirectoryAtPath:d0 error:nil]) {
        NSData *d = [NSData dataWithContentsOfFile:[d0 stringByAppendingPathComponent:f]];
        if (d) m[[@"@model_path/weights/" stringByAppendingString:f]] = @{@"offset": @0, @"data": d}; }
      w = m; NSMutableArray *order = [NSMutableArray array]; for (NSString *k in w) [order addObject:k.lastPathComponent];
      printf("WEIGHTS enum=%s\n", [[order componentsJoinedByString:@","] UTF8String]); fflush(stdout); }
    else { NSData *d = [NSData dataWithContentsOfFile:@(argv[6])];
      if (!d) { printf("RESULT compile=NOWEIGHTFILE\n"); return 66; }
      w = @{@"@model_path/weights/weight.bin": @{@"offset": @0, @"data": d}}; } }
  Kern *k = compile_kern_mil_w(mil, w, (int)ib, (int)ob);
  if (!k) { printf("RESULT compile=FAIL\n"); return 2; }
  if (getenv("ANE_ATTR")) { id a = ((id(*)(id,SEL))objc_msgSend)((__bridge id)k->model, @selector(modelAttributes));
    printf("MODEL_ATTRIBUTES %s\n", [[a description] UTF8String]); }
  IOSurfaceLock(k->ioIn, 0, NULL);
  void *p = IOSurfaceGetBaseAddress(k->ioIn);
  size_t n = ib / (in32 ? 4 : 2);
  for (size_t i = 0; i < n; i++) { float v = sinf(0.7f * (float)(i % 977) + 0.3f) * 0.5f;
    if (in32) ((float *)p)[i] = v; else ((_Float16 *)p)[i] = (_Float16)v; }
  if (getenv("ANE_IN")) { NSData *d = [NSData dataWithContentsOfFile:@(getenv("ANE_IN"))]; if (d) memcpy(p, d.bytes, MIN(d.length, ib)); }
  IOSurfaceUnlock(k->ioIn, 0, NULL);
  id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
  BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
  IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
  void *q = IOSurfaceGetBaseAddress(k->ioOut);
  size_t m = ob / (out32 ? 4 : 2), nonfinite = 0, nonzero = 0; double maxabs = 0; unsigned cs = 2166136261u;
  for (size_t i = 0; i < m; i++) { float v = out32 ? ((float *)q)[i] : (float)((_Float16 *)q)[i];
    if (!isfinite(v)) nonfinite++; else { if (v != 0) nonzero++; if (fabs(v) > maxabs) maxabs = fabs(v); }
    cs = (cs ^ (out32 ? ((unsigned *)q)[i] : (unsigned)((unsigned short *)q)[i])) * 16777619u; }
  const char *dump = getenv("ANE_DUMP");
  if (dump) { NSString *b = @(dump);
    [[NSData dataWithBytes:p length:ib] writeToFile:[b stringByAppendingString:@".in"] atomically:YES];
    [[NSData dataWithBytes:q length:ob] writeToFile:[b stringByAppendingString:@".out"] atomically:YES]; }
  IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
  printf("RESULT compile=OK eval=%d err=%s nonfinite=%zu maxabs=%.6g nonzero=%zu/%zu fnv=%08x\n", ok, e ? [[e description] UTF8String] : "-", nonfinite, maxabs, nonzero, m, cs);
  free_kern(k);
} return 0; }

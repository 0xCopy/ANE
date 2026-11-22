// ane_run.m: compile and evaluate an arbitrary MIL program on the ANE (private in-memory path, the same calls as io.h's
// compile_kern_mil_w); any number of inputs and outputs; every tensor is staged at the strides the compiled model reports.
//
// usage: ane_run DIR [ITERS]
//   DIR/spec.json  {"mil": "model.mil", "weights": "weights",              (weights: optional directory, every file under it
//                                                                            becomes @model_path/<path relative to DIR>)
//                   "inputs":  [{"name": "x", "dtype": "fp16", "shape": [1,64,1,32], "file": "x.bin"}],
//                   "outputs": [{"name": "y", "dtype": "fp16", "shape": [1,64,1,32], "file": "y.bin"}]}
//   Tensor files hold the logical tensor, row-major, little endian, no padding.  Output files are written, in the element type
//   the compiled model reports for that output (listed as "out_types"), which need not be the type the spec asked for.
// env:  ANE_COMPILE_OPTS / ANE_LOAD_OPTS / ANE_EVAL_OPTS = JSON object   the options dictionary of compileWithQoS: / loadWithQoS: / evaluateWithQoS:
// env:  ANE_PERF=MASK   sets the model's perfStatsMask to MASK (e.g. 0x1), passes an _ANEPerformanceStats object in the request, and reports
//                       what it holds after the timed evaluations as "perf": {hwExecutionTimeNS, counters, perfCounterDataBytes, rawBytes, description}
// stdout: one JSON line {"compile","load","eval" : "ok"|"fail"|"skip", "err", "ms_compile", "ms_eval", "inputs": [...], "outputs": [...]}
//   where each input/output carries what the compiler reported: Batches Channels Depth Height Width BatchStride DepthStride
//   PlaneStride RowStride Interleave Type.
// build: SDKROOT=<Xcode MacOSX.sdk> xcrun clang -w -fobjc-arc -framework Foundation -framework IOSurface -o ane_run ane_run.m
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

static NSMutableDictionary *R;
static void emit(void) { NSData *d = [NSJSONSerialization dataWithJSONObject:R options:0 error:nil]; printf("%s\n", [[NSString alloc] initWithData:d encoding:NSUTF8StringEncoding].UTF8String); }
static void fail(NSString *stage, NSString *err) { R[stage] = @"fail"; R[@"err"] = err ?: @"no error"; emit(); exit(0); }
static NSDictionary *envDict(const char *name) {            // ANE_LOAD_OPTS / ANE_EVAL_OPTS / ANE_COMPILE_OPTS: a JSON object, passed as the options dictionary of that call
  const char *v = getenv(name); if (!v) return @{};
  id j = [NSJSONSerialization JSONObjectWithData:[@(v) dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil]; return [j isKindOfClass:[NSDictionary class]] ? j : @{}; }
static double now_ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }
static size_t tsize(NSString *t) {        // the Type string of the compiled model's input/output list
  if ([t isEqual:@"Float32"] || [t isEqual:@"Int32"] || [t isEqual:@"UInt32"]) return 4;
  if ([t isEqual:@"Float16"] || [t isEqual:@"Int16"] || [t isEqual:@"UInt16"] || [t isEqual:@"BFloat16"]) return 2;
  return 1; }
static NSString *dname(NSString *t) {
  NSDictionary *m = @{@"Float16": @"fp16", @"Float32": @"fp32", @"Int32": @"int32", @"UInt32": @"uint32", @"Int16": @"int16", @"UInt16": @"uint16", @"Int8": @"int8", @"UInt8": @"uint8", @"BFloat16": @"bf16"};
  return m[t] ?: t; }
static size_t esize(NSString *dt) {
  if ([dt isEqual:@"fp32"] || [dt isEqual:@"int32"] || [dt isEqual:@"uint32"]) return 4;
  if ([dt isEqual:@"fp16"] || [dt isEqual:@"int16"] || [dt isEqual:@"uint16"] || [dt isEqual:@"bf16"]) return 2;
  return 1; }
static IOSurfaceRef surf(size_t bytes) {
  return IOSurfaceCreate((__bridge CFDictionaryRef)@{ (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1, (id)kIOSurfaceBytesPerElement: @1,
                                                       (id)kIOSurfaceBytesPerRow: @(bytes), (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0 }); }
#define N(k) [a[@k] longLongValue]
// The file side is the MIL tensor, row-major.  The surface side is described by what the compiled model reports (Batches Channels Depth Height
// Width and the strides).  Rank <= 4: the model labels the dims (N, C, 1, H, W); only unit dims are inserted, so the logical order is the reported
// order.  Rank 5: a 3-D convolution keeps (N, C, D, H, W); every other program (stack, expand_dims, elementwise at rank 5, ...) reports MIL
// [m0, m1, m2, m3, m4] as (B, C, D, H, W) = (m0, m2, m1, m3, m4), and its strides then make the surface the plain MIL row-major order.  The
// mapping is chosen from the reported dims; if dims 1 and 2 are equal and larger than 1 the two are indistinguishable and the case is refused.
static BOOL move(uint8_t *surface, uint8_t *logical, NSDictionary *a, NSArray *shape, size_t es, BOOL toSurface, NSString **err) {
  long B = N("Batches"), C = N("Channels"), D = N("Depth"), H = N("Height"), W = N("Width"), bs = N("BatchStride"), ds = N("DepthStride"), ps = N("PlaneStride"), rs = N("RowStride");
  if (shape.count == 5) {
    long s0 = [shape[0] longLongValue], s1 = [shape[1] longLongValue], s2 = [shape[2] longLongValue], s3 = [shape[3] longLongValue], s4 = [shape[4] longLongValue];
    BOOL ident = (B == s0 && C == s1 && D == s2 && H == s3 && W == s4), swap = (B == s0 && C == s2 && D == s1 && H == s3 && W == s4);
    if (ident && swap && s1 != 1) { *err = @"ambiguous rank-5 layout (dims 1 and 2 are equal)"; return NO; }
    if (!ident && !swap) { *err = [NSString stringWithFormat:@"rank-5 shape %ld,%ld,%ld,%ld,%ld vs reported %ld,%ld,%ld,%ld,%ld", s0, s1, s2, s3, s4, B, C, D, H, W]; return NO; }
    for (long i0 = 0; i0 < s0; i0++) for (long i1 = 0; i1 < s1; i1++) for (long i2 = 0; i2 < s2; i2++) for (long i3 = 0; i3 < s3; i3++) {
      uint8_t *sp = surface + i0 * bs + (ident ? i2 * ds + i1 * ps : i1 * ds + i2 * ps) + i3 * rs; uint8_t *l = logical + ((((i0 * s1 + i1) * s2 + i2) * s3 + i3) * s4) * es;
      if (toSurface) memcpy(sp, l, s4 * es); else memcpy(l, sp, s4 * es); }
    return YES; }
  for (long b = 0; b < B; b++) for (long c = 0; c < C; c++) for (long d = 0; d < D; d++) for (long h = 0; h < H; h++) {
    uint8_t *sp = surface + b * bs + d * ds + c * ps + h * rs; uint8_t *l = logical + ((((b * C + c) * D + d) * H + h) * W) * es;
    if (toSurface) memcpy(sp, l, W * es); else memcpy(l, sp, W * es); }
  return YES; }
static long elems(NSDictionary *a) { return N("Batches") * N("Channels") * N("Depth") * N("Height") * N("Width"); }

int main(int argc, char **argv) { @autoreleasepool {
  R = [NSMutableDictionary dictionaryWithDictionary:@{@"compile": @"skip", @"load": @"skip", @"eval": @"skip"}];
  if (argc < 2) { fprintf(stderr, "usage: ane_run DIR [ITERS]\n"); return 64; }
  NSString *dir = @(argv[1]); int iters = argc > 2 ? atoi(argv[2]) : 10;
  dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
  Class D = NSClassFromString(@"_ANEInMemoryModelDescriptor"), I = NSClassFromString(@"_ANEInMemoryModel"), AR = NSClassFromString(@"_ANERequest"), AIO = NSClassFromString(@"_ANEIOSurfaceObject");
  NSDictionary *spec = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"spec.json"]] options:0 error:nil];
  if (!spec) fail(@"compile", @"no spec.json");
  NSData *mil = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:spec[@"mil"]]];
  if (!mil) fail(@"compile", @"no MIL file");
  NSMutableDictionary *weights = [NSMutableDictionary dictionary]; NSFileManager *fm = [NSFileManager defaultManager];
  if (spec[@"weights"]) { NSString *wd = [dir stringByAppendingPathComponent:spec[@"weights"]];
    for (NSString *rel in [fm subpathsAtPath:wd]) { BOOL isdir = NO; NSString *full = [wd stringByAppendingPathComponent:rel]; [fm fileExistsAtPath:full isDirectory:&isdir]; if (isdir) continue;
      weights[[NSString stringWithFormat:@"@model_path/%@/%@", spec[@"weights"], rel]] = @{@"offset": @0, @"data": [NSData dataWithContentsOfFile:full]}; } }
  id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(D, @selector(modelWithMILText:weights:optionsPlist:), mil, weights, nil);
  if (!desc) fail(@"compile", @"descriptor is nil");
  id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(I, @selector(inMemoryModelWithDescriptor:), desc);
  id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
  NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
  [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
  [mil writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
  for (NSString *path in weights) { NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""]; NSString *full = [td stringByAppendingPathComponent:rel];
    [fm createDirectoryAtPath:[full stringByDeletingLastPathComponent] withIntermediateDirectories:YES attributes:nil error:nil]; [weights[path][@"data"] writeToFile:full atomically:YES]; }
  NSError *e = nil; double t0 = now_ms();
  const char *perfEnv = getenv("ANE_PERF"); id perf = nil;
  IOSurfaceRef perfSurf = NULL; size_t perfBytes = 0;
  if (perfEnv) { ((void(*)(id,SEL,unsigned))objc_msgSend)(mdl, @selector(setPerfStatsMask:), (unsigned)strtoul(perfEnv, 0, 0));
    perfBytes = getenv("ANE_PERF_BYTES") ? strtoull(getenv("ANE_PERF_BYTES"), 0, 0) : 65536; perfSurf = surf(perfBytes);
    perf = ((id(*)(Class,SEL,IOSurfaceRef,long long))objc_msgSend)(NSClassFromString(@"_ANEPerformanceStatsIOSurface"), @selector(objectWithIOSurface:statType:), perfSurf, getenv("ANE_PERF_TYPE") ? atoll(getenv("ANE_PERF_TYPE")) : 0); }
  if (!((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, envDict("ANE_COMPILE_OPTS"), &e)) { [fm removeItemAtPath:td error:nil]; fail(@"compile", e.description); }
  R[@"compile"] = @"ok"; R[@"ms_compile"] = @(now_ms() - t0); t0 = now_ms();
  if (!((BOOL(*)(id,SEL,unsigned,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, envDict("ANE_LOAD_OPTS"), &e)) { [fm removeItemAtPath:td error:nil]; fail(@"load", e.description); }
  R[@"load"] = @"ok"; R[@"ms_load"] = @(now_ms() - t0);
  NSDictionary *attrs = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(modelAttributes));
  NSArray *inSyms = attrs[@"ANEFModelDescription"][@"kANEFModelInputSymbolsArrayKey"], *outSyms = attrs[@"ANEFModelDescription"][@"kANEFModelOutputSymbolsArrayKey"];
  NSDictionary *net = [attrs[@"NetworkStatusList"] firstObject]; NSArray *liveIn = net[@"LiveInputList"], *liveOut = net[@"LiveOutputList"];
  R[@"inputs"] = liveIn ?: @[]; R[@"outputs"] = liveOut ?: @[];
  if (inSyms.count != [spec[@"inputs"] count] || outSyms.count != [spec[@"outputs"] count] || liveIn.count != inSyms.count || liveOut.count != outSyms.count) {
    R[@"err"] = [NSString stringWithFormat:@"io count: model %lu/%lu, spec %lu/%lu", (unsigned long)inSyms.count, (unsigned long)outSyms.count, (unsigned long)[spec[@"inputs"] count], (unsigned long)[spec[@"outputs"] count]]; emit(); return 0; }
  NSMutableArray *wIn = [NSMutableArray array], *wOut = [NSMutableArray array], *iIdx = [NSMutableArray array], *oIdx = [NSMutableArray array];
  NSMutableArray<NSValue *> *inSurf = [NSMutableArray array], *outSurf = [NSMutableArray array]; NSMutableArray *inSpec = [NSMutableArray array], *outSpec = [NSMutableArray array];
  for (NSUInteger i = 0; i < inSyms.count; i++) { NSDictionary *s = nil; for (NSDictionary *c in spec[@"inputs"]) if ([c[@"name"] isEqual:inSyms[i]]) s = c;
    if (!s) { R[@"err"] = [@"no spec for input " stringByAppendingString:inSyms[i]]; emit(); return 0; }
    NSDictionary *a = liveIn[i]; size_t bytes = N("Batches") * N("BatchStride"); IOSurfaceRef sf = surf(bytes);
    NSData *d = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:s[@"file"]]]; size_t es = esize(s[@"dtype"]);
    if (![dname(a[@"Type"]) isEqual:s[@"dtype"]]) { R[@"err"] = [NSString stringWithFormat:@"input %@: model type %@, spec %@", s[@"name"], a[@"Type"], s[@"dtype"]]; R[@"type_mismatch"] = @"in"; emit(); return 0; }
    if (!d || d.length != elems(a) * es) { R[@"err"] = [NSString stringWithFormat:@"input %@: file %lu bytes, model wants %ld elements x %zu", s[@"name"], (unsigned long)d.length, elems(a), es]; emit(); return 0; }
    NSString *merr = nil; IOSurfaceLock(sf, 0, NULL); memset(IOSurfaceGetBaseAddress(sf), 0, bytes); BOOL mok = move(IOSurfaceGetBaseAddress(sf), (uint8_t *)d.bytes, a, s[@"shape"], es, YES, &merr); IOSurfaceUnlock(sf, 0, NULL);
    if (!mok) { R[@"err"] = [NSString stringWithFormat:@"input %@: %@", s[@"name"], merr]; R[@"layout_err"] = @"in"; emit(); return 0; }
    [wIn addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), sf)]; [iIdx addObject:@(i)]; [inSurf addObject:[NSValue valueWithPointer:sf]]; [inSpec addObject:s]; }
  for (NSUInteger i = 0; i < outSyms.count; i++) { NSString *nm = [outSyms[i] stringByReplacingOccurrencesOfString:@"@output" withString:@""]; NSDictionary *s = nil; for (NSDictionary *c in spec[@"outputs"]) if ([c[@"name"] isEqual:nm]) s = c;
    if (!s) { R[@"err"] = [@"no spec for output " stringByAppendingString:nm]; emit(); return 0; }
    NSDictionary *a = liveOut[i]; size_t bytes = N("Batches") * N("BatchStride"); IOSurfaceRef sf = surf(bytes);
    [wOut addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), sf)]; [oIdx addObject:@(i)]; [outSurf addObject:[NSValue valueWithPointer:sf]]; [outSpec addObject:s]; }
  id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:), wIn, iIdx, wOut, oIdx, nil, perf ? @[perf] : nil, @0);
  t0 = now_ms();
  BOOL ok = ((BOOL(*)(id,SEL,unsigned,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, envDict("ANE_EVAL_OPTS"), req, &e);
  if (!ok) { [fm removeItemAtPath:td error:nil]; fail(@"eval", e.description); }
  R[@"eval"] = @"ok"; R[@"ms_first_eval"] = @(now_ms() - t0);
  for (int i = 0; i < 3; i++) ((BOOL(*)(id,SEL,unsigned,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, envDict("ANE_EVAL_OPTS"), req, &e);
  t0 = now_ms(); for (int i = 0; i < iters; i++) ((BOOL(*)(id,SEL,unsigned,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, envDict("ANE_EVAL_OPTS"), req, &e);
  R[@"ms_eval"] = @((now_ms() - t0) / (iters ? iters : 1));
  for (NSUInteger i = 0; i < outSyms.count; i++) { NSDictionary *a = liveOut[i], *s = outSpec[i]; IOSurfaceRef sf = (IOSurfaceRef)[outSurf[i] pointerValue]; size_t es = tsize(a[@"Type"]);
    NSMutableData *buf = [NSMutableData dataWithLength:elems(a) * es]; IOSurfaceLock(sf, kIOSurfaceLockReadOnly, NULL); NSString *merr = nil; BOOL mok = move(IOSurfaceGetBaseAddress(sf), buf.mutableBytes, a, s[@"shape"], es, NO, &merr); IOSurfaceUnlock(sf, kIOSurfaceLockReadOnly, NULL);
    if (!mok) { R[@"err"] = [NSString stringWithFormat:@"output %@: %@", s[@"name"], merr]; R[@"layout_err"] = @"out"; continue; }
    [buf writeToFile:[dir stringByAppendingPathComponent:s[@"file"]] atomically:YES]; }
  NSMutableArray *ot = [NSMutableArray array]; for (NSDictionary *a in liveOut) [ot addObject:dname(a[@"Type"])]; R[@"out_types"] = ot;
  if (perf) { NSMutableDictionary *pd = [NSMutableDictionary dictionary];
    // the buffer the hardware / driver filled: first bytes as hex, count of non-zero bytes, and the framework's own decoding of it
    IOSurfaceLock(perfSurf, kIOSurfaceLockReadOnly, NULL); const uint8_t *pb = IOSurfaceGetBaseAddress(perfSurf); size_t nz = 0, last = 0; for (size_t i = 0; i < perfBytes; i++) if (pb[i]) { nz++; last = i + 1; }
    pd[@"bufferBytes"] = @(perfBytes); pd[@"nonzeroBytes"] = @(nz); pd[@"usedBytes"] = @(last);
    NSMutableString *hx = [NSMutableString string]; for (size_t i = 0; i < MIN(last, (size_t)96); i++) [hx appendFormat:@"%02x", pb[i]]; pd[@"head"] = hx;
    NSData *raw = [NSData dataWithBytes:pb length:last ? last : 0]; IOSurfaceUnlock(perfSurf, kIOSurfaceLockReadOnly, NULL);
    @try { id st = ((id(*)(id,SEL))objc_msgSend)(perf, @selector(stats)); pd[@"statsClass"] = st ? NSStringFromClass([st class]) : @"nil"; pd[@"stats"] = st ? [[st description] substringToIndex:MIN((NSUInteger)600, [[st description] length])] : @"nil"; } @catch (id x) { pd[@"statsErr"] = [x description]; }
    @try { id d = ((id(*)(Class,SEL,id))objc_msgSend)(NSClassFromString(@"_ANEPerformanceStats"), @selector(decodeRawStatsData:), raw); if (d) pd[@"decoded"] = [[d description] substringToIndex:MIN((NSUInteger)900, [[d description] length])]; } @catch (id x) { pd[@"decodeErr"] = [x description]; }
    @try { NSArray *pa = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray)); pd[@"perfStatsArrayCount"] = @(pa.count); } @catch (id x) {}
    R[@"perf"] = pd; }
  ((BOOL(*)(id,SEL,unsigned,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
  [fm removeItemAtPath:td error:nil];
  emit(); } return 0; }

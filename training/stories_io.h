// stories_io.h — IOSurface helpers, blob builders, NEON conversion
#pragma once
#include "stories_config.h"
#include <arm_neon.h>

// IOSurface pool — reuse freed surfaces of the same size
#define IOSURF_POOL_MAX 128
static struct {
    IOSurfaceRef surfaces[IOSURF_POOL_MAX];
    size_t sizes[IOSURF_POOL_MAX];
    int count;
} g_iosurf_pool = { .count = 0 };

static IOSurfaceRef make_surface(size_t bytes) {
    // Check pool for matching size
    for (int i = 0; i < g_iosurf_pool.count; i++) {
        if (g_iosurf_pool.sizes[i] == bytes) {
            IOSurfaceRef s = g_iosurf_pool.surfaces[i];
            // Swap-remove
            g_iosurf_pool.surfaces[i] = g_iosurf_pool.surfaces[--g_iosurf_pool.count];
            g_iosurf_pool.sizes[i] = g_iosurf_pool.sizes[g_iosurf_pool.count];
            return s;
        }
    }
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}

static void pool_return_surface(IOSurfaceRef s, size_t bytes) {
    if (g_iosurf_pool.count < IOSURF_POOL_MAX) {
        g_iosurf_pool.surfaces[g_iosurf_pool.count] = s;
        g_iosurf_pool.sizes[g_iosurf_pool.count] = bytes;
        g_iosurf_pool.count++;
    } else {
        CFRelease(s);
    }
}

static NSData *build_blob(const float *w, int rows, int cols) {
    size_t ws=(size_t)rows*cols*2, tot=128+ws;  // size_t prevents int overflow (CRIT-04)
    uint8_t *b=(uint8_t*)calloc(tot,1);
    if (!b) { fprintf(stderr, "build_blob: calloc(%zu) failed\n", tot); return nil; }
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=(uint32_t)ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(size_t i=0;i<(size_t)rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_t(const float *w, int rows, int cols) {
    size_t ws=(size_t)cols*rows*2, tot=128+ws;  // size_t prevents int overflow (CRIT-04)
    uint8_t *b=(uint8_t*)calloc(tot,1);
    if (!b) { fprintf(stderr, "build_blob_t: calloc(%zu) failed\n", tot); return nil; }
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=(uint32_t)ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) fp16[j*rows+i]=(_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    size_t ws=(size_t)cnt*2, tot=128+ws;  // size_t prevents int overflow (CRIT-04)
    uint8_t *b=(uint8_t*)calloc(tot,1);
    if (!b) { fprintf(stderr, "build_blob_fp16: calloc(%zu) failed\n", tot); return nil; }
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=(uint32_t)ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// MED-05: NEON alignment guarantee.
// IOSurface base address is page-aligned (≥4096 bytes). Offset = ch_off*SEQ*sizeof(_Float16).
// With SEQ%8==0, all offsets are multiples of 16 bytes → aligned for vld1q_f16/vst1q_f32.
// Additionally, ARM64 handles unaligned NEON loads in hardware (unlike ARM32).
_Static_assert(SEQ % 8 == 0,
    "SEQ must be multiple of 8 to guarantee 16-byte alignment for NEON (MED-05)");

// NEON vectorized conversion
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// IOSurface I/O (channel-first [C,S] layout)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    if (IOSurfaceLock(s, 0, NULL) != kIOReturnSuccess) {  // MED-01
        fprintf(stderr, "IOSurfaceLock(write) failed — surface write skipped\n");
        return;
    }
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s), data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    if (IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {  // MED-01
        fprintf(stderr, "IOSurfaceLock(read) failed — output read skipped\n");
        return;
    }
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) {  // MED-01
        fprintf(stderr, "IOSurfaceLock(copy dst) failed — copy skipped\n");
        return;
    }
    if (IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {  // MED-01
        fprintf(stderr, "IOSurfaceLock(copy src) failed — copy skipped\n");
        IOSurfaceUnlock(dst, 0, NULL);
        return;
    }
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    if (IOSurfaceLock(s, 0, NULL) != kIOReturnSuccess) {  // MED-01
        fprintf(stderr, "IOSurfaceLock(write_at) failed — surface write skipped\n");
        return;
    }
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
// Read raw fp16 from IOSurface without conversion (for fp16 activation cache)
static void io_read_raw_fp16(IOSurfaceRef s, _Float16 *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp, channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

static void io_write_fp16_t(IOSurfaceRef s, const float *w, int rows, int cols) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *f16 = (_Float16*)IOSurfaceGetBaseAddress(s);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) f16[j*rows+i]=(_Float16)w[i*cols+j];
    IOSurfaceUnlock(s, 0, NULL);
}

// Kernel compile/eval
static Kern *compile_kern_mil_multi(NSString *mil, NSDictionary *weights, int *in_sizes, int n_in, int oc_bytes) {
    @autoreleasepool {
    if (!g_ane_ok_large) { printf("  [compile] ANE not available\n"); return NULL; }  // CRIT-01/02
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    if (!mdl) { printf("  [compile] mdl=NULL\n"); return NULL; }  // CRIT-02
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    // MED-02: pid + atomic sequence counter make the directory unique per process and
    // per call, preventing TOCTOU conflicts when two instances compile the same model.
    int seq = __sync_fetch_and_add(&g_compile_seq, 1);
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"ANE_%d_%d_%@", getpid(), seq, hx]];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n"); return NULL;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = (Kern*)calloc(1, sizeof(Kern));
    if (!k) { fprintf(stderr, "OOM: calloc(Kern)\n"); abort(); }  // HIGH-04
    k->model = (void*)CFBridgingRetain(mdl);
    k->n_inputs = n_in;
    k->inputs = (IOSurfaceRef*)calloc(n_in, sizeof(IOSurfaceRef));
    NSMutableArray *inObs = [NSMutableArray array];
    NSMutableArray *inIdx = [NSMutableArray array];
    for(int i=0; i<n_in; i++) {
        k->inputs[i] = make_surface(in_sizes[i]);
        [inObs addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->inputs[i])];
        [inIdx addObject:@(i)];
    }
    k->ioOut = make_surface(oc_bytes);
    k->ioIn = k->inputs[0];   // single-input code (the trainers, the tests) keeps using ioIn: it is the first of inputs[]
    k->inBytes = in_sizes[0];
    k->outBytes = oc_bytes;
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = (void*)CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        inObs, inIdx, @[wO], @[@0], nil, nil, @0));
    k->tmpDir = (void*)CFBridgingRetain(td);
    return k;
    }
}
// Single-input kernel, the form the trainers and the tests call (48 call sites): ic_bytes in, oc_bytes out.
static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    int sizes[1] = { ic_bytes };
    return compile_kern_mil_multi(mil, weights, sizes, 1, oc_bytes);
}

static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    // inputs[0] is the same surface as ioIn and goes back to the pool with the output; an SDK kernel with several inputs has more, which are released
    pool_return_surface(k->inputs[0], k->inBytes);
    for (int i = 1; i < k->n_inputs; i++) CFRelease(k->inputs[i]);
    free(k->inputs);
    pool_return_surface(k->ioOut, k->outBytes);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}
static bool ane_eval(Kern *k) {  // HIGH-05: was void
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) fprintf(stderr, "  [ane_eval] FAILED: %s\n",
                     e ? [[e description] UTF8String] : "unknown error");
    return (bool)ok;
}

// mnist_demo.m — Self-contained MNIST training on Apple Neural Engine
//
// This is a standalone sandbox demo: no external #includes from training/,
// everything needed is embedded here with memory-leak fixes applied.
//
// Architecture: Linear(784→128) → ReLU → Linear(128→10) → Softmax+CE
// Forward:  ANE 1×1 conv kernels (baked fp16 weights, channel-first I/O)
// Backward: CPU (naive but correct — fast enough for MNIST)
// Optimizer: SGD with momentum + gradient accumulation
//
// Build:  make
// Run:    python3 fetch_mnist.py && ./mnist_demo
//
// Requires macOS 15+ on Apple Silicon.
// See KNOWN_BUGS.md for upstream issues — fixes are applied in this file.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach_time.h>

// ═════════════════════════════════════════════════════════════════════════════
// § 1  ANE Runtime (embedded, with ARC leak fixes)
// ═════════════════════════════════════════════════════════════════════════════

typedef struct {
    id model;               // _ANEInMemoryModel  (__strong under ARC)
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;             // _ANERequest         (__strong under ARC)
    NSString *tmpDir;       //                     (__strong under ARC)
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
} ANEKernel;

static Class g_ANEDesc, g_ANEInMem, g_ANEReq, g_ANEIO;
static bool g_ane_loaded = false;

static void ane_init(void) {
    if (g_ane_loaded) return;
    void *h = dlopen("/System/Library/PrivateFrameworks/"
                     "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) {
        fprintf(stderr, "WARN: Could not load AppleNeuralEngine.framework\n");
    }
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    g_ane_loaded = (g_ANEDesc && g_ANEInMem && g_ANEReq && g_ANEIO);
    if (!g_ane_loaded) {
        fprintf(stderr, "WARN: ANE classes not found — is this Apple Silicon + macOS 15+?\n");
    }
}

static IOSurfaceRef ane_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static ANEKernel *ane_compile(NSData *milText, NSData *weightData,
                               int nInputs, size_t *inputSizes,
                               int nOutputs, size_t *outputSizes) {
    ane_init();
    if (!g_ane_loaded) return NULL;
    NSError *e = nil;

    NSDictionary *wdict = nil;
    if (weightData) {
        wdict = @{@"@model_path/weights/weight.bin":
                      @{@"offset": @0, @"data": weightData}};
    }
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
        milText, wdict, nil);
    if (!desc) return NULL;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);

    // Pre-populate temp dir with MIL + weights
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milText writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
              atomically:YES];
    if (weightData)
        [weightData writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"]
                     atomically:YES];

    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE compile failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        fprintf(stderr, "ANE load failed: %s\n", [[e description] UTF8String]);
        [fm removeItemAtPath:td error:nil];
        return NULL;
    }

    ANEKernel *k = calloc(1, sizeof(ANEKernel));
    k->model = mdl;
    k->tmpDir = td;
    k->nInputs = nInputs;
    k->nOutputs = nOutputs;
    k->inputBytes  = malloc(nInputs  * sizeof(size_t));
    k->outputBytes = malloc(nOutputs * sizeof(size_t));
    memcpy(k->inputBytes,  inputSizes,  nInputs  * sizeof(size_t));
    memcpy(k->outputBytes, outputSizes, nOutputs * sizeof(size_t));

    // Create IOSurfaces
    k->ioInputs  = malloc(nInputs  * sizeof(IOSurfaceRef));
    k->ioOutputs = malloc(nOutputs * sizeof(IOSurfaceRef));
    for (int i = 0; i < nInputs;  i++) k->ioInputs[i]  = ane_create_surface(inputSizes[i]);
    for (int i = 0; i < nOutputs; i++) k->ioOutputs[i] = ane_create_surface(outputSizes[i]);

    // Build request
    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:nInputs];
    for (int i = 0; i < nInputs; i++) {
        [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
        [iIdx addObject:@(i)];
    }
    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:nOutputs];
    NSMutableArray *oIdx  = [NSMutableArray arrayWithCapacity:nOutputs];
    for (int i = 0; i < nOutputs; i++) {
        [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
        [oIdx addObject:@(i)];
    }
    k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
        g_ANEReq,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        wIns, iIdx, wOuts, oIdx, nil, nil, @0);

    return k;
}

static void ane_write_input(ANEKernel *k, int idx, const void *data, size_t bytes) {
    IOSurfaceLock(k->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(k->ioInputs[idx], 0, NULL);
}

static void ane_read_output(ANEKernel *k, int idx, void *data, size_t bytes) {
    IOSurfaceLock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

static bool ane_eval(ANEKernel *k) {
    NSError *e = nil;
    return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        k->model, @selector(evaluateWithQoS:options:request:error:),
        21, @{}, k->request, &e);
}

// FIX for KNOWN_BUGS.md #1: nil strong ObjC refs before free()
static void ane_free(ANEKernel *k) {
    if (!k) return;

    // Unload from ANE device
    NSError *e = nil;
    if (k->model) {
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            k->model, @selector(unloadWithQoS:error:), 21, &e);
    }

    // Release IOSurfaces
    for (int i = 0; i < k->nInputs;  i++) if (k->ioInputs[i])  CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) if (k->ioOutputs[i]) CFRelease(k->ioOutputs[i]);

    // Clean up temp directory
    if (k->tmpDir) {
        [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
    }

    // ── BUG-1 FIX: ARC won't release strong refs on free() ──
    // Must nil them explicitly so ARC inserts the release calls.
    k->model   = nil;
    k->request = nil;
    k->tmpDir  = nil;
    // ── end fix ──

    free(k->ioInputs);  free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    free(k);
}

// ═════════════════════════════════════════════════════════════════════════════
// § 2  MIL Generation (baked-weight 1×1 conv)
// ═════════════════════════════════════════════════════════════════════════════

// Build fp16 weight blob: 64-byte global header + 64-byte chunk header + data
// weights_f32: row-major [out_ch, in_ch]
static NSData *mil_build_weight_blob(const float *weights_f32, int out_ch, int in_ch) {
    NSUInteger wsize = (NSUInteger)out_ch * in_ch * 2;  // fp16
    NSUInteger total = 64 + 64 + wsize;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t *)(chunk + 8)  = (uint32_t)wsize;
    *(uint32_t *)(chunk + 16) = 128;  // data offset from file start
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (NSUInteger i = 0; i < (NSUInteger)out_ch * in_ch; i++)
        fp16[i] = (_Float16)weights_f32[i];
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Generate MIL for 1×1 conv with baked weights:
//   Input:  [1, in_ch, 1, spatial] fp32
//   Weight: [out_ch, in_ch, 1, 1]  fp16 from BLOBFILE
//   Output: [1, out_ch, 1, spatial] fp32
static NSString *mil_gen_conv(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        "        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        "        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        "        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)"
        "[name = string(\"cast_in\")];\n"
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
        "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), "
        "offset = uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
        "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)"
        "[name = string(\"conv\")];\n"
        "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)"
        "[name = string(\"cast_out\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_ch, spatial, in_ch, spatial,
        out_ch, in_ch, out_ch, in_ch,
        out_ch, spatial, out_ch, spatial];
}

// ═════════════════════════════════════════════════════════════════════════════
// § 3  Configuration
// ═════════════════════════════════════════════════════════════════════════════

#define IN_DIM      784
#define HID_DIM     128
#define OUT_DIM     10
#define BATCH       128
#define LR          0.01f
#define MOM         0.9f
#define N_EPOCHS    6
#define ACCUM_STEPS 50
#define MAX_COMPILES 118

// ═════════════════════════════════════════════════════════════════════════════
// § 4  MNIST Data Loader
// ═════════════════════════════════════════════════════════════════════════════

static uint32_t read_be32(FILE *f) {
    uint8_t b[4]; fread(b, 1, 4, f);
    return ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|(uint32_t)b[3];
}

typedef struct {
    float *images;  // [n, 784] row-major, [0,1]
    int   *labels;  // [n]
    int    n;
} MnistSet;

static MnistSet mnist_load(const char *img_path, const char *lbl_path) {
    MnistSet s = {0};
    FILE *fi = fopen(img_path, "rb");
    FILE *fl = fopen(lbl_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Cannot open: %s / %s\nRun: python3 fetch_mnist.py\n",
                img_path, lbl_path);
        if (fi) fclose(fi);
        if (fl) fclose(fl);
        return s;
    }

    read_be32(fi); s.n = (int)read_be32(fi);
    read_be32(fi); read_be32(fi);
    read_be32(fl); read_be32(fl);

    s.images = malloc((size_t)s.n * IN_DIM * sizeof(float));
    s.labels = malloc((size_t)s.n * sizeof(int));

    uint8_t *raw = malloc((size_t)s.n * IN_DIM);
    fread(raw, 1, (size_t)s.n * IN_DIM, fi);
    for (int i = 0; i < s.n * IN_DIM; i++) s.images[i] = raw[i] / 255.0f;
    free(raw);

    uint8_t *rlbl = malloc((size_t)s.n);
    fread(rlbl, 1, (size_t)s.n, fl);
    for (int i = 0; i < s.n; i++) s.labels[i] = rlbl[i];
    free(rlbl);

    fclose(fi); fclose(fl);
    printf("  Loaded %d samples from %s\n", s.n, img_path);
    return s;
}

static void mnist_free(MnistSet *s) {
    free(s->images); s->images = NULL;
    free(s->labels); s->labels = NULL;
    s->n = 0;
}

// ═════════════════════════════════════════════════════════════════════════════
// § 5  Utilities
// ═════════════════════════════════════════════════════════════════════════════

static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// Row-major [rows, cols] → channel-first [cols, rows]
static void transpose(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

static void shuffle_indices(int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

static double now_ms(void) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// ═════════════════════════════════════════════════════════════════════════════
// § 6  Network
// ═════════════════════════════════════════════════════════════════════════════

// FIX for KNOWN_BUGS.md #5: wrap each compile in its own @autoreleasepool
static ANEKernel *compile_conv(const float *weights, int in_ch, int out_ch, int spatial) {
    ANEKernel *result = NULL;
    @autoreleasepool {
        NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
        NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
        size_t inBytes  = (size_t)in_ch  * spatial * 4;
        size_t outBytes = (size_t)out_ch * spatial * 4;
        result = ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding],
                             wb, 1, &inBytes, 1, &outBytes);
    }  // autoreleasepool drains here — frees MIL string + weight NSData temps
    return result;
}

typedef struct {
    float *W1, *W2;        // [out, in] row-major
    float *dW1, *dW2;      // accumulated gradients
    float *vW1, *vW2;      // SGD momentum buffers

    ANEKernel *kern1;      // conv: 784→128
    ANEKernel *kern2;      // conv: 128→10
    int compile_count;

    // Activation buffers (channel-first: [channels, batch])
    float *x_cf;           // [IN_DIM, BATCH]
    float *h_pre;          // [HID_DIM, BATCH] before ReLU
    float *h_post;         // [HID_DIM, BATCH] after ReLU
    float *logits;         // [OUT_DIM, BATCH]
} Net;

static void net_init(Net *net) {
    memset(net, 0, sizeof(Net));

    net->W1  = calloc(HID_DIM * IN_DIM,  sizeof(float));
    net->W2  = calloc(OUT_DIM * HID_DIM, sizeof(float));
    net->dW1 = calloc(HID_DIM * IN_DIM,  sizeof(float));
    net->dW2 = calloc(OUT_DIM * HID_DIM, sizeof(float));
    net->vW1 = calloc(HID_DIM * IN_DIM,  sizeof(float));
    net->vW2 = calloc(OUT_DIM * HID_DIM, sizeof(float));

    float s1 = sqrtf(2.0f / IN_DIM);
    float s2 = sqrtf(2.0f / HID_DIM);
    for (int i = 0; i < HID_DIM * IN_DIM;  i++) net->W1[i] = randn() * s1;
    for (int i = 0; i < OUT_DIM * HID_DIM; i++) net->W2[i] = randn() * s2;

    net->x_cf   = calloc(IN_DIM  * BATCH, sizeof(float));
    net->h_pre  = calloc(HID_DIM * BATCH, sizeof(float));
    net->h_post = calloc(HID_DIM * BATCH, sizeof(float));
    net->logits = calloc(OUT_DIM * BATCH, sizeof(float));
}

static int net_compile(Net *net) {
    @autoreleasepool {
        if (net->kern1) { ane_free(net->kern1); net->kern1 = NULL; }
        if (net->kern2) { ane_free(net->kern2); net->kern2 = NULL; }

        net->kern1 = compile_conv(net->W1, IN_DIM, HID_DIM, BATCH);
        if (!net->kern1) return -1;
        net->compile_count++;

        net->kern2 = compile_conv(net->W2, HID_DIM, OUT_DIM, BATCH);
        if (!net->kern2) return -2;
        net->compile_count++;
    }
    return 0;
}

static float net_forward(Net *net, const float *batch_images, const int *batch_labels) {
    // Row-major [BATCH, 784] → channel-first [784, BATCH]
    transpose(net->x_cf, batch_images, BATCH, IN_DIM);

    // Layer 1: ANE conv (W1 @ x)
    ane_write_input(net->kern1, 0, net->x_cf, (size_t)IN_DIM * BATCH * 4);
    ane_eval(net->kern1);
    ane_read_output(net->kern1, 0, net->h_pre, (size_t)HID_DIM * BATCH * 4);

    // ReLU (CPU)
    for (int i = 0; i < HID_DIM * BATCH; i++)
        net->h_post[i] = net->h_pre[i] > 0 ? net->h_pre[i] : 0;

    // Layer 2: ANE conv (W2 @ h)
    ane_write_input(net->kern2, 0, net->h_post, (size_t)HID_DIM * BATCH * 4);
    ane_eval(net->kern2);
    ane_read_output(net->kern2, 0, net->logits, (size_t)OUT_DIM * BATCH * 4);

    // Cross-entropy loss (CPU) — logits: [OUT_DIM, BATCH] channel-first
    float loss = 0;
    for (int b = 0; b < BATCH; b++) {
        float mx = -1e9f;
        for (int c = 0; c < OUT_DIM; c++) {
            float v = net->logits[c * BATCH + b];
            if (v > mx) mx = v;
        }
        float sum_exp = 0;
        for (int c = 0; c < OUT_DIM; c++)
            sum_exp += expf(net->logits[c * BATCH + b] - mx);
        loss -= (net->logits[batch_labels[b] * BATCH + b] - mx - logf(sum_exp));
    }
    return loss / BATCH;
}

static void net_backward(Net *net, const int *batch_labels) {
    // dLogits = softmax(logits) − one_hot(labels)
    float *dL = calloc(OUT_DIM * BATCH, sizeof(float));
    for (int b = 0; b < BATCH; b++) {
        float mx = -1e9f;
        for (int c = 0; c < OUT_DIM; c++) {
            float v = net->logits[c * BATCH + b];
            if (v > mx) mx = v;
        }
        float se = 0;
        for (int c = 0; c < OUT_DIM; c++) {
            dL[c * BATCH + b] = expf(net->logits[c * BATCH + b] - mx);
            se += dL[c * BATCH + b];
        }
        for (int c = 0; c < OUT_DIM; c++) dL[c * BATCH + b] /= se;
        dL[batch_labels[b] * BATCH + b] -= 1.0f;
        for (int c = 0; c < OUT_DIM; c++) dL[c * BATCH + b] /= BATCH;
    }

    // dW2 += dL @ h_post^T
    for (int o = 0; o < OUT_DIM; o++)
        for (int h = 0; h < HID_DIM; h++)
            for (int b = 0; b < BATCH; b++)
                net->dW2[o * HID_DIM + h] += dL[o * BATCH + b] * net->h_post[h * BATCH + b];

    // d_h_post = W2^T @ dL → ReLU backward → dW1
    float *d_hp = calloc(HID_DIM * BATCH, sizeof(float));
    for (int h = 0; h < HID_DIM; h++)
        for (int b = 0; b < BATCH; b++)
            for (int o = 0; o < OUT_DIM; o++)
                d_hp[h * BATCH + b] += net->W2[o * HID_DIM + h] * dL[o * BATCH + b];

    for (int h = 0; h < HID_DIM; h++)
        for (int i = 0; i < IN_DIM; i++)
            for (int b = 0; b < BATCH; b++) {
                float dh = net->h_pre[h * BATCH + b] > 0 ? d_hp[h * BATCH + b] : 0;
                net->dW1[h * IN_DIM + i] += dh * net->x_cf[i * BATCH + b];
            }

    free(dL);
    free(d_hp);
}

static void net_sgd_step(Net *net, int accum_steps) {
    float scale = 1.0f / accum_steps;
    for (int i = 0; i < HID_DIM * IN_DIM; i++) {
        net->vW1[i] = MOM * net->vW1[i] - LR * net->dW1[i] * scale;
        net->W1[i] += net->vW1[i];
    }
    for (int i = 0; i < OUT_DIM * HID_DIM; i++) {
        net->vW2[i] = MOM * net->vW2[i] - LR * net->dW2[i] * scale;
        net->W2[i] += net->vW2[i];
    }
    memset(net->dW1, 0, (size_t)HID_DIM * IN_DIM  * sizeof(float));
    memset(net->dW2, 0, (size_t)OUT_DIM * HID_DIM * sizeof(float));
}

static int net_count_correct(Net *net, const int *batch_labels) {
    int correct = 0;
    for (int b = 0; b < BATCH; b++) {
        int best = 0;
        float best_v = net->logits[b];
        for (int c = 1; c < OUT_DIM; c++) {
            float v = net->logits[c * BATCH + b];
            if (v > best_v) { best_v = v; best = c; }
        }
        if (best == batch_labels[b]) correct++;
    }
    return correct;
}

static float net_eval_accuracy(Net *net, const float *images, const int *labels, int n) {
    int correct = 0, total = 0;
    float *bimg = malloc((size_t)BATCH * IN_DIM * sizeof(float));
    int   *blbl = malloc((size_t)BATCH * sizeof(int));

    for (int start = 0; start + BATCH <= n; start += BATCH) {
        memcpy(bimg, images + (size_t)start * IN_DIM, (size_t)BATCH * IN_DIM * sizeof(float));
        memcpy(blbl, labels + start, (size_t)BATCH * sizeof(int));
        net_forward(net, bimg, blbl);
        correct += net_count_correct(net, blbl);
        total += BATCH;
    }

    free(bimg);
    free(blbl);
    return total > 0 ? (float)correct / total * 100.0f : 0;
}

static void net_free(Net *net) {
    if (net->kern1) { ane_free(net->kern1); net->kern1 = NULL; }
    if (net->kern2) { ane_free(net->kern2); net->kern2 = NULL; }
    free(net->W1);  free(net->W2);
    free(net->dW1); free(net->dW2);
    free(net->vW1); free(net->vW2);
    free(net->x_cf); free(net->h_pre); free(net->h_post); free(net->logits);
}

// ═════════════════════════════════════════════════════════════════════════════
// § 7  Main — Training Loop
// ═════════════════════════════════════════════════════════════════════════════

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        srand((unsigned)time(NULL));

        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║   MNIST Demo — Apple Neural Engine (self-contained)     ║\n");
        printf("╠══════════════════════════════════════════════════════════╣\n");
        printf("║  Network:  %d → %d (ReLU) → %d (softmax+CE)       ║\n", IN_DIM, HID_DIM, OUT_DIM);
        printf("║  Batch:    %-4d  LR: %.4f  Momentum: %.2f            ║\n", BATCH, LR, MOM);
        printf("║  Epochs:   %-2d    Grad accum: %d steps                  ║\n", N_EPOCHS, ACCUM_STEPS);
        printf("║  Mode:     ANE conv (baked fp16 weights, recompile)     ║\n");
        printf("║  Fixes:    BUG-1 (ARC leak), BUG-5 (autorelease pool)  ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");

        // ── Load MNIST ──────────────────────────────────────────────────
        MnistSet train = mnist_load("data/train-images-idx3-ubyte",
                                    "data/train-labels-idx1-ubyte");
        MnistSet test  = mnist_load("data/t10k-images-idx3-ubyte",
                                    "data/t10k-labels-idx1-ubyte");
        if (!train.images || !test.images) {
            fprintf(stderr, "Data not found. Run: python3 fetch_mnist.py\n");
            return 1;
        }
        printf("\n");

        // ── Init network ────────────────────────────────────────────────
        Net net;
        net_init(&net);

        printf("Parameters: W1[%d×%d] + W2[%d×%d] = %d floats\n",
               HID_DIM, IN_DIM, OUT_DIM, HID_DIM,
               HID_DIM * IN_DIM + OUT_DIM * HID_DIM);

        // ── Initial compile ─────────────────────────────────────────────
        printf("Compiling ANE conv kernels...\n");
        if (net_compile(&net) != 0) {
            fprintf(stderr, "ANE compilation failed.\n");
            fprintf(stderr, "Requirements: macOS 15+, Apple Silicon, SIP check.\n");
            net_free(&net);
            mnist_free(&train); mnist_free(&test);
            return 1;
        }
        int max_updates = (MAX_COMPILES - net.compile_count) / 2;
        printf("  ✓ 2 kernels compiled (%d/%d budget)\n", net.compile_count, MAX_COMPILES);
        printf("  Max weight updates: %d (×%d accum = %d forward batches)\n\n",
               max_updates, ACCUM_STEPS, max_updates * ACCUM_STEPS);

        // ── Training loop ───────────────────────────────────────────────
        int batches_per_epoch = train.n / BATCH;
        int *indices = malloc((size_t)train.n * sizeof(int));
        for (int i = 0; i < train.n; i++) indices[i] = i;

        float *batch_img = malloc((size_t)BATCH * IN_DIM * sizeof(float));
        int   *batch_lbl = malloc((size_t)BATCH * sizeof(int));

        int global_step = 0, accum_count = 0, accum_correct = 0;
        float accum_loss = 0;
        double t_start = now_ms();
        double t_ane_compile = 0, t_ane_eval = 0, t_cpu_backward = 0;

        for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
            shuffle_indices(indices, train.n);
            float epoch_loss = 0;
            int epoch_correct = 0, epoch_steps = 0;

            for (int step = 0; step < batches_per_epoch; step++) {
                if (net.compile_count >= MAX_COMPILES) {
                    printf("\n  ⚠ ANE compile budget exhausted (%d/%d).\n",
                           net.compile_count, MAX_COMPILES);
                    goto done;
                }

                // Gather batch
                for (int b = 0; b < BATCH; b++) {
                    int idx = indices[step * BATCH + b];
                    memcpy(batch_img + (size_t)b * IN_DIM,
                           train.images + (size_t)idx * IN_DIM,
                           IN_DIM * sizeof(float));
                    batch_lbl[b] = train.labels[idx];
                }

                // Forward (ANE)
                double t0 = now_ms();
                float loss = net_forward(&net, batch_img, batch_lbl);
                t_ane_eval += now_ms() - t0;

                // Backward (CPU)
                t0 = now_ms();
                net_backward(&net, batch_lbl);
                t_cpu_backward += now_ms() - t0;

                accum_loss += loss;
                accum_correct += net_count_correct(&net, batch_lbl);
                accum_count++;
                epoch_loss += loss;
                epoch_correct += net_count_correct(&net, batch_lbl);
                epoch_steps++;
                global_step++;

                // Weight update after ACCUM_STEPS
                if (accum_count >= ACCUM_STEPS) {
                    net_sgd_step(&net, accum_count);

                    float avg_loss = accum_loss / accum_count;
                    float avg_acc  = (float)accum_correct / (accum_count * BATCH) * 100.0f;
                    printf("  E%d step=%4d  loss=%.4f  acc=%.1f%%  [%d compiles]\n",
                           epoch + 1, epoch_steps, avg_loss, avg_acc, net.compile_count);

                    accum_loss = 0; accum_correct = 0; accum_count = 0;

                    // Recompile with updated weights
                    t0 = now_ms();
                    if (net_compile(&net) != 0) {
                        fprintf(stderr, "Recompile failed at step %d\n", global_step);
                        goto done;
                    }
                    t_ane_compile += now_ms() - t0;
                }
            }

            // Flush remaining accumulated gradients
            if (accum_count > 0) {
                net_sgd_step(&net, accum_count);
                accum_loss = 0; accum_correct = 0; accum_count = 0;
                if (net.compile_count < MAX_COMPILES) {
                    double t0 = now_ms();
                    if (net_compile(&net) != 0) {
                        fprintf(stderr, "Recompile failed at epoch end\n");
                        goto done;
                    }
                    t_ane_compile += now_ms() - t0;
                }
            }

            float test_acc = net_eval_accuracy(&net, test.images, test.labels, test.n);
            printf("─ Epoch %d/%d  loss=%.4f  train=%.1f%%  test=%.1f%%  [%d/%d]\n\n",
                   epoch + 1, N_EPOCHS,
                   epoch_loss / epoch_steps,
                   (float)epoch_correct / (epoch_steps * BATCH) * 100.0f,
                   test_acc, net.compile_count, MAX_COMPILES);
        }

    done: ;
        double elapsed = now_ms() - t_start;

        // ── Final eval ──────────────────────────────────────────────────
        float final_acc = net_eval_accuracy(&net, test.images, test.labels, test.n);

        printf("══════════════════════════════════════════════════════\n");
        printf("  Final test accuracy:  %.1f%%\n", final_acc);
        printf("  Training steps:       %d\n", global_step);
        printf("  ANE compiles:         %d / %d\n", net.compile_count, MAX_COMPILES);
        printf("  Wall time:            %.1fs\n", elapsed / 1000.0);
        printf("  ├─ ANE eval:          %.1fs\n", t_ane_eval / 1000.0);
        printf("  ├─ ANE compile:       %.1fs\n", t_ane_compile / 1000.0);
        printf("  └─ CPU backward:      %.1fs\n", t_cpu_backward / 1000.0);
        printf("══════════════════════════════════════════════════════\n");

        // ── Cleanup (all allocations freed, no leaks) ───────────────────
        net_free(&net);
        mnist_free(&train);
        mnist_free(&test);
        free(indices);
        free(batch_img);
        free(batch_lbl);
    }
    return 0;
}

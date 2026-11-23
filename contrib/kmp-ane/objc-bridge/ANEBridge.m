// ANEBridge.m — C-callable implementation wrapping ane_runtime.h / ane_mil_gen.h
// Compile: xcrun clang -O2 -c ANEBridge.m -o ANEBridge.o -fobjc-arc \
//          -framework Foundation -framework IOSurface
#import "ANEBridge.h"
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <os/lock.h>

// ---------------------------------------------------------------------------
// Private ANE class references
// ---------------------------------------------------------------------------
static Class g_ANEDesc  = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq   = nil;
static Class g_ANEIO    = nil;
static bool  g_inited   = false;
static os_unfair_lock g_lock = OS_UNFAIR_LOCK_INIT;

// ---------------------------------------------------------------------------
// Buffer layout of one input/output.
//
// The ANE does not take tightly packed tensors: every row (the last tensor
// dimension) starts on a 64-byte boundary, so a [128, 10] fp32 tensor needs
// 128 * 64 bytes, not 128 * 40. A surface of the packed size is rejected at
// eval ("IOSurface smaller than the model expects"). Packed and padded sizes
// only coincide when the row is a multiple of 16 floats, which is why
// [1, C, 1, 16k] kernels worked without this. The loaded model reports the
// real layout in modelAttributes; we size the surfaces from it and copy row
// by row.
// ---------------------------------------------------------------------------
typedef struct {
    size_t rows;          // channels * height
    size_t planeRows;     // height (rows per channel plane)
    size_t rowBytes;      // payload per row: width * element size
    size_t rowStride;     // distance between row starts in the surface
    size_t planeStride;   // distance between channel planes
    size_t surfaceBytes;  // allocation the ANE expects
    size_t elemSize;      // 4 (Float32) or 2 (Float16)
    bool   known;         // false: layout not reported, treat as one packed row
} ANEPort;

// ---------------------------------------------------------------------------
// Internal kernel struct (matches opaque ANEBridgeKernel)
// ---------------------------------------------------------------------------
struct ANEBridgeKernel {
    id           model;       // _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id           request;     // _ANERequest
    NSString     *tmpDir;
    int          nInputs;
    int          nOutputs;
    size_t       *inputBytes;   // caller-visible (tightly packed) size of each input
    size_t       *outputBytes;
    ANEPort      *inPorts;      // layout of each buffer as the loaded model reports it
    ANEPort      *outPorts;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_init(void) {
    os_unfair_lock_lock(&g_lock);
    if (g_inited) { os_unfair_lock_unlock(&g_lock); return ANE_OK; }

    void *h = dlopen("/System/Library/PrivateFrameworks/"
                      "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) { os_unfair_lock_unlock(&g_lock); return ANE_ERR_INIT_FAILED; }

    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        os_unfair_lock_unlock(&g_lock);
        return ANE_ERR_INIT_FAILED;
    }
    g_inited = true;
    os_unfair_lock_unlock(&g_lock);
    return ANE_OK;
}

bool ane_bridge_is_available(void) {
    return g_inited;
}

// Last framework error, kept as text so callers (C, Kotlin) can report why a status came back.
static char g_last_error[1024] = "";

static void set_last_error(const char *stage, NSError *e) {
    NSString *d = e ? [e description] : @"(no NSError returned)";
    snprintf(g_last_error, sizeof(g_last_error), "%s: %s", stage, [d UTF8String] ?: "");
}

const char *ane_bridge_last_error(void) {
    return g_last_error;
}

// ---------------------------------------------------------------------------
// IOSurface helper
// ---------------------------------------------------------------------------
static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:           @(bytes),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow:     @(bytes),
        (id)kIOSurfaceAllocSize:       @(bytes),
        (id)kIOSurfacePixelFormat:     @0
    });
}

// Fill `port` for a buffer the caller declared as `declared` packed bytes.
// `info` is the model's LiveInputList/LiveOutputList entry, or nil if unavailable.
static void port_init(ANEPort *port, NSDictionary *info, size_t declared) {
    memset(port, 0, sizeof(*port));
    port->rows = port->planeRows = 1;
    port->rowBytes = port->rowStride = port->planeStride = port->surfaceBytes = declared;
    port->elemSize = 4;
    if (![info isKindOfClass:[NSDictionary class]]) return;

    NSString *type = info[@"Type"];
    size_t elem = [type isEqual:@"Float32"] ? 4 : [type isEqual:@"Float16"] ? 2 : 0;
    size_t w  = [info[@"Width"] unsignedLongValue],    h  = [info[@"Height"] unsignedLongValue];
    size_t c  = [info[@"Channels"] unsignedLongValue], rs = [info[@"RowStride"] unsignedLongValue];
    size_t ps = [info[@"PlaneStride"] unsignedLongValue], bs = [info[@"BatchStride"] unsignedLongValue];
    // Only trust a layout that describes exactly the tensor the caller declared.
    if (!elem || !w || !h || !c || rs < w * elem || ps < h * rs || bs < c * ps) return;
    if (w * h * c * elem != declared) return;

    port->rows = c * h;      port->planeRows = h;
    port->rowBytes = w * elem; port->rowStride = rs;
    port->planeStride = ps;  port->surfaceBytes = bs;
    port->elemSize = elem;   port->known = true;
}

static inline size_t port_row_offset(const ANEPort *p, size_t row) {
    return (row / p->planeRows) * p->planeStride + (row % p->planeRows) * p->rowStride;
}

// Copy packed caller data into the padded surface layout (scale != 1 only for Float32 ports).
static void port_write(IOSurfaceRef surf, const ANEPort *p, const void *data, size_t bytes, float scale) {
    IOSurfaceLock(surf, 0, NULL);
    uint8_t *base = IOSurfaceGetBaseAddress(surf);
    const uint8_t *src = data;
    for (size_t r = 0; r < p->rows && bytes > 0; r++) {
        size_t n = bytes < p->rowBytes ? bytes : p->rowBytes;
        if (scale != 1.0f && p->elemSize == 4)
            vDSP_vsmul((const float *)(src + r * p->rowBytes), 1, &scale,
                       (float *)(base + port_row_offset(p, r)), 1, n / 4);
        else
            memcpy(base + port_row_offset(p, r), src + r * p->rowBytes, n);
        bytes -= n;
    }
    IOSurfaceUnlock(surf, 0, NULL);
}

static void port_read(IOSurfaceRef surf, const ANEPort *p, void *data, size_t bytes, float scale) {
    IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *base = IOSurfaceGetBaseAddress(surf);
    uint8_t *dst = data;
    for (size_t r = 0; r < p->rows && bytes > 0; r++) {
        size_t n = bytes < p->rowBytes ? bytes : p->rowBytes;
        if (scale != 1.0f && p->elemSize == 4)
            vDSP_vsmul((const float *)(base + port_row_offset(p, r)), 1, &scale,
                       (float *)(dst + r * p->rowBytes), 1, n / 4);
        else
            memcpy(dst + r * p->rowBytes, base + port_row_offset(p, r), n);
        bytes -= n;
    }
    IOSurfaceUnlock(surf, kIOSurfaceLockReadOnly, NULL);
}

// ---------------------------------------------------------------------------
// Kernel compile
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_compile(const char *mil_text,
                                    size_t mil_text_len,
                                    const void *weight_data,
                                    size_t weight_len,
                                    int n_inputs,
                                    const size_t *input_bytes,
                                    int n_outputs,
                                    const size_t *output_bytes,
                                    ANEBridgeKernel **out_kernel) {
    @autoreleasepool {
        if (!g_inited) return ANE_ERR_INIT_FAILED;
        if (!mil_text || !out_kernel) return ANE_ERR_NULL_KERNEL;

        NSData *milData = [NSData dataWithBytes:mil_text length:mil_text_len];
        NSError *e = nil;

        // Weight dictionary. Must be an empty dictionary, not nil, for kernels without baked
        // weights (dynamic-weight kernels): the descriptor factory returns nil for nil weights.
        NSDictionary *wdict = @{};
        if (weight_data && weight_len > 0) {
            NSData *wd = [NSData dataWithBytes:weight_data length:weight_len];
            wdict = @{@"@model_path/weights/weight.bin":
                          @{@"offset": @0, @"data": wd}};
        }

        // Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc,
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) { set_last_error("descriptor (modelWithMILText returned nil)", nil); return ANE_ERR_COMPILE_FAILED; }

        // Create in-memory model
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem,
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { set_last_error("model (inMemoryModelWithDescriptor returned nil)", nil); return ANE_ERR_COMPILE_FAILED; }

        // Set up temp directory
        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];
        if (weight_data && weight_len > 0) {
            NSData *wd = [NSData dataWithBytes:weight_data length:weight_len];
            [wd writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"]
                 atomically:YES];
        }

        // Compile
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            set_last_error("compile", e);
            [fm removeItemAtPath:td error:nil];
            return ANE_ERR_COMPILE_FAILED;
        }

        // Load
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            set_last_error("load", e);
            [fm removeItemAtPath:td error:nil];
            return ANE_ERR_LOAD_FAILED;
        }

        // Allocate kernel struct
        ANEBridgeKernel *k = calloc(1, sizeof(ANEBridgeKernel));
        if (!k) return ANE_ERR_ALLOC_FAILED;

        k->model    = mdl;
        k->tmpDir   = td;
        k->nInputs  = n_inputs;
        k->nOutputs = n_outputs;

        k->inputBytes  = malloc(n_inputs  * sizeof(size_t));
        k->outputBytes = malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes,  input_bytes,  n_inputs  * sizeof(size_t));
        memcpy(k->outputBytes, output_bytes, n_outputs * sizeof(size_t));

        // Buffer layouts, as the loaded model reports them (see ANEPort).
        NSArray *liveIn = nil, *liveOut = nil;
        @try {
            NSDictionary *attrs = [mdl valueForKey:@"modelAttributes"];
            NSArray *nets = [attrs isKindOfClass:[NSDictionary class]] ? attrs[@"NetworkStatusList"] : nil;
            NSDictionary *net = ([nets isKindOfClass:[NSArray class]] && nets.count) ? nets[0] : nil;
            if ([net isKindOfClass:[NSDictionary class]]) {
                liveIn  = net[@"LiveInputList"];
                liveOut = net[@"LiveOutputList"];
            }
        } @catch (NSException *ex) { liveIn = liveOut = nil; }
        if (![liveIn  isKindOfClass:[NSArray class]] || (int)liveIn.count  != n_inputs)  liveIn  = nil;
        if (![liveOut isKindOfClass:[NSArray class]] || (int)liveOut.count != n_outputs) liveOut = nil;

        k->inPorts  = calloc(n_inputs,  sizeof(ANEPort));
        k->outPorts = calloc(n_outputs, sizeof(ANEPort));
        for (int i = 0; i < n_inputs;  i++) port_init(&k->inPorts[i],  liveIn[i],  input_bytes[i]);
        for (int i = 0; i < n_outputs; i++) port_init(&k->outPorts[i], liveOut[i], output_bytes[i]);

        // Create IOSurfaces
        k->ioInputs  = malloc(n_inputs  * sizeof(IOSurfaceRef));
        k->ioOutputs = malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs;  i++)
            k->ioInputs[i]  = create_surface(k->inPorts[i].surfaceBytes);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(k->outPorts[i].surfaceBytes);

        // Build ANE request
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);

        *out_kernel = k;
        return ANE_OK;
    }
}

// ---------------------------------------------------------------------------
// I/O
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_write_input(ANEBridgeKernel *kernel,
                                        int index,
                                        const void *data,
                                        size_t bytes) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;
    if (index < 0 || index >= kernel->nInputs) return ANE_ERR_INDEX_OOB;
    if (bytes > kernel->inputBytes[index]) return ANE_ERR_INDEX_OOB;   // would overrun the surface
    port_write(kernel->ioInputs[index], &kernel->inPorts[index], data, bytes, 1.0f);
    return ANE_OK;
}

ANEBridgeStatus ane_bridge_read_output(ANEBridgeKernel *kernel,
                                        int index,
                                        void *data,
                                        size_t bytes) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;
    if (index < 0 || index >= kernel->nOutputs) return ANE_ERR_INDEX_OOB;
    if (bytes > kernel->outputBytes[index]) return ANE_ERR_INDEX_OOB;  // would read past the surface
    port_read(kernel->ioOutputs[index], &kernel->outPorts[index], data, bytes, 1.0f);
    return ANE_OK;
}

// ---------------------------------------------------------------------------
// Eval
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_eval(ANEBridgeKernel *kernel) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        kernel->model,
        @selector(evaluateWithQoS:options:request:error:),
        21, @{}, kernel->request, &e);
    if (!ok) set_last_error("eval", e);
    return ok ? ANE_OK : ANE_ERR_EVAL_FAILED;
}

// ---------------------------------------------------------------------------
// Free
// ---------------------------------------------------------------------------
void ane_bridge_free_kernel(ANEBridgeKernel *kernel) {
    if (!kernel) return;
    @autoreleasepool {
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            kernel->model, @selector(unloadWithQoS:error:), 21, &e);
        for (int i = 0; i < kernel->nInputs;  i++) CFRelease(kernel->ioInputs[i]);
        for (int i = 0; i < kernel->nOutputs; i++) CFRelease(kernel->ioOutputs[i]);
        [[NSFileManager defaultManager] removeItemAtPath:kernel->tmpDir error:nil];
        // The kernel is calloc'd, so ARC never releases its strong ObjC fields on free().
        // Nil them first (assignment releases) or every kernel leaks its model/request/tmpDir.
        kernel->model = nil;
        kernel->request = nil;
        kernel->tmpDir = nil;
        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->inputBytes);
        free(kernel->outputBytes);
        free(kernel->inPorts);
        free(kernel->outPorts);
        free(kernel);
    }
}

// ---------------------------------------------------------------------------
// MIL generators — return heap-allocated C strings (caller frees)
// ---------------------------------------------------------------------------
static char *nsstring_to_cstr(NSString *s) {
    const char *utf8 = [s UTF8String];
    size_t len = strlen(utf8);
    char *c = malloc(len + 1);
    memcpy(c, utf8, len + 1);
    return c;
}

char *ane_bridge_mil_gen_conv(int in_ch, int out_ch, int spatial) {
    @autoreleasepool {
        NSString *mil = [NSString stringWithFormat:
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
            "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
            "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), "
            "val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n"
            "        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, "
            "pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n",
            in_ch, spatial, in_ch, spatial,
            out_ch, in_ch, out_ch, in_ch,
            out_ch, spatial, out_ch, spatial];
        return nsstring_to_cstr(mil);
    }
}


char *ane_bridge_mil_gen_qkv(int dim, int spatial) {
    // Simplified: return NULL for now (not needed for MNIST)
    return NULL;
}

char *ane_bridge_mil_gen_ffn_up(int dim, int hidden_dim, int spatial) {
    // Simplified: return NULL for now (not needed for MNIST)
    return NULL;
}

// ---------------------------------------------------------------------------
// Weight blob builders
// ---------------------------------------------------------------------------
void *ane_bridge_build_weight_blob(const float *weights_f32,
                                    int out_ch, int in_ch,
                                    size_t *out_len) {
    size_t wsize = (size_t)out_ch * in_ch * 2; // FP16
    size_t total = 64 + 64 + wsize; // global header + chunk header + data
    uint8_t *buf = calloc(total, 1);
    if (!buf) { *out_len = 0; return NULL; }

    buf[0] = 0x01; buf[4] = 0x02;
    uint8_t *chunk = buf + 64;
    chunk[0] = 0xEF; chunk[1] = 0xBE; chunk[2] = 0xAD; chunk[3] = 0xDE;
    chunk[4] = 0x01;
    *(uint32_t*)(chunk + 8)  = (uint32_t)wsize;
    *(uint32_t*)(chunk + 16) = 128; // data offset from file start

    _Float16 *fp16 = (_Float16*)(buf + 128);
    for (size_t i = 0; i < (size_t)out_ch * in_ch; i++)
        fp16[i] = (_Float16)weights_f32[i];

    *out_len = total;
    return buf;
}

void *ane_bridge_build_qkv_weight_blob(const float *wq, const float *wk,
                                        const float *wv, int dim,
                                        size_t *out_len) {
    *out_len = 0;
    return NULL; // Not needed for MNIST
}

void *ane_bridge_build_ffn_up_weight_blob(const float *w1, const float *w3,
                                           int hidden_dim, int dim,
                                           size_t *out_len) {
    *out_len = 0;
    return NULL; // Not needed for MNIST
}

// ---------------------------------------------------------------------------
// High-level: compile baked-weight conv kernel
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_compile_conv(const float *weights,
                                         int in_ch, int out_ch, int spatial,
                                         ANEBridgeKernel **out_kernel) {
    if (!g_inited) { ANEBridgeStatus s = ane_bridge_init(); if (s) return s; }

    // Build weight blob
    size_t wb_len = 0;
    void *wb = ane_bridge_build_weight_blob(weights, out_ch, in_ch, &wb_len);
    if (!wb) return ANE_ERR_ALLOC_FAILED;

    // Generate MIL
    char *mil = ane_bridge_mil_gen_conv(in_ch, out_ch, spatial);
    if (!mil) { free(wb); return ANE_ERR_COMPILE_FAILED; }

    size_t mil_len = strlen(mil);
    size_t in_bytes  = (size_t)in_ch  * spatial * 4; // fp32
    size_t out_bytes = (size_t)out_ch * spatial * 4;

    ANEBridgeStatus status = ane_bridge_compile(
        mil, mil_len, wb, wb_len,
        1, &in_bytes, 1, &out_bytes, out_kernel);

    free(mil);
    free(wb);
    return status;
}

// ---------------------------------------------------------------------------
// High-level: conv eval with transpose
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_conv_eval(ANEBridgeKernel *kernel,
                                      const float *x, float *y,
                                      int S, int in_dim, int out_dim) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;

    // Transpose row-major [S, in_dim] → channel-first [in_dim, S]
    float *x_t = malloc((size_t)S * in_dim * sizeof(float));
    if (!x_t) return ANE_ERR_ALLOC_FAILED;
    for (int t = 0; t < S; t++)
        for (int i = 0; i < in_dim; i++)
            x_t[i * S + t] = x[t * in_dim + i];

    ANEBridgeStatus s = ane_bridge_write_input(kernel, 0, x_t,
                                                (size_t)S * in_dim * sizeof(float));
    free(x_t);
    if (s != ANE_OK) return s;

    s = ane_bridge_eval(kernel);
    if (s != ANE_OK) return s;

    // Read and transpose back [out_dim, S] → [S, out_dim]
    float *y_t = malloc((size_t)S * out_dim * sizeof(float));
    if (!y_t) return ANE_ERR_ALLOC_FAILED;

    s = ane_bridge_read_output(kernel, 0, y_t,
                                (size_t)S * out_dim * sizeof(float));
    if (s != ANE_OK) { free(y_t); return s; }

    for (int t = 0; t < S; t++)
        for (int i = 0; i < out_dim; i++)
            y[t * out_dim + i] = y_t[i * S + t];

    free(y_t);
    return ANE_OK;
}

// ---------------------------------------------------------------------------
// Matmul compile (2 inputs: x and W, no baked weights)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Dynamic matmul: Y = op(A) @ op(B), both operands are kernel inputs
// ---------------------------------------------------------------------------
// MIL for Y = op(A) @ op(B). The inputs are named "a" and "b" on purpose: the ANE
// numbers a kernel's inputs by NAME (byte order), not by declaration order, so a
// kernel declared (x, W) gets W as input 0. Names that sort in declaration order
// keep "input 0 = A, input 1 = B" true.
static char *mil_gen_mm(int a_rows, int a_cols, int b_rows, int b_cols,
                        bool transpose_a, bool transpose_b) {
    int m = transpose_a ? a_cols : a_rows;   // output rows
    int n = transpose_b ? b_rows : b_cols;   // output cols
    @autoreleasepool {
        NSString *t = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, 1, %d, %d]> a, tensor<fp32, [1, 1, %d, %d]> b) {\n"
            "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, 1, %d, %d]> a16 = cast(dtype = to_fp16, x = a)[name = string(\"cast_a\")];\n"
            "        tensor<fp16, [1, 1, %d, %d]> b16 = cast(dtype = to_fp16, x = b)[name = string(\"cast_b\")];\n"
            "        bool tx = const()[name = string(\"tx\"), val = bool(%s)];\n"
            "        bool ty = const()[name = string(\"ty\"), val = bool(%s)];\n"
            "        tensor<fp16, [1, 1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = a16, y = b16)[name = string(\"mm\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, 1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n",
            a_rows, a_cols, b_rows, b_cols,
            a_rows, a_cols, b_rows, b_cols,
            transpose_a ? "true" : "false", transpose_b ? "true" : "false",
            m, n, m, n];
        return nsstring_to_cstr(t);
    }
}

ANEBridgeStatus ane_bridge_compile_mm(int a_rows, int a_cols,
                                       int b_rows, int b_cols,
                                       bool transpose_a, bool transpose_b,
                                       ANEBridgeKernel **out_kernel) {
    if (!g_inited) { ANEBridgeStatus s = ane_bridge_init(); if (s) return s; }
    if (!out_kernel || a_rows <= 0 || a_cols <= 0 || b_rows <= 0 || b_cols <= 0) return ANE_ERR_NULL_KERNEL;

    int m  = transpose_a ? a_cols : a_rows;   // output rows
    int ka = transpose_a ? a_rows : a_cols;   // inner dimension seen from A
    int kb = transpose_b ? b_cols : b_rows;   // inner dimension seen from B
    int n  = transpose_b ? b_rows : b_cols;   // output cols
    if (ka != kb) {
        snprintf(g_last_error, sizeof(g_last_error), "compile_mm: inner dimensions differ (%d vs %d)", ka, kb);
        return ANE_ERR_COMPILE_FAILED;
    }

    char *mil = mil_gen_mm(a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b);
    if (!mil) return ANE_ERR_ALLOC_FAILED;

    size_t inputSizes[2] = { (size_t)a_rows * a_cols * 4, (size_t)b_rows * b_cols * 4 };
    size_t outSize = (size_t)m * n * 4;
    ANEBridgeStatus status = ane_bridge_compile(mil, strlen(mil), NULL, 0,
                                                2, inputSizes, 1, &outSize, out_kernel);
    free(mil);
    return status;
}

ANEBridgeStatus ane_bridge_mm_eval(ANEBridgeKernel *kernel,
                                    const float *a, const float *b, float *y) {
    if (!kernel || !a || !b || !y) return ANE_ERR_NULL_KERNEL;
    if (kernel->nInputs != 2 || kernel->nOutputs != 1) return ANE_ERR_INDEX_OOB;
    size_t na = kernel->inputBytes[0] / 4, nb = kernel->inputBytes[1] / 4, ny = kernel->outputBytes[0] / 4;

    // Normalise both operands for the fp16 kernel (see header).
    float sa = 0, sb = 0;
    vDSP_maxmgv(a, 1, &sa, na);
    vDSP_maxmgv(b, 1, &sb, nb);
    if (sa == 0 || sb == 0) { memset(y, 0, ny * sizeof(float)); return ANE_OK; }
    if (!isfinite(sa) || !isfinite(sb)) sa = sb = 1.0f;   // let NaN/Inf propagate unscaled

    port_write(kernel->ioInputs[0], &kernel->inPorts[0], a, na * 4, 1.0f / sa);
    port_write(kernel->ioInputs[1], &kernel->inPorts[1], b, nb * 4, 1.0f / sb);
    ANEBridgeStatus s = ane_bridge_eval(kernel);
    if (s != ANE_OK) return s;
    port_read(kernel->ioOutputs[0], &kernel->outPorts[0], y, ny * 4, sa * sb);
    return ANE_OK;
}

// ---------------------------------------------------------------------------
// Channel-first matmul helpers (y = W @ x), built on the dynamic matmul above.
// Input 0 is W[out_ch, in_ch], input 1 is x[in_ch, spatial].
// ---------------------------------------------------------------------------
char *ane_bridge_mil_gen_matmul(int in_ch, int out_ch, int spatial) {
    return mil_gen_mm(out_ch, in_ch, in_ch, spatial, false, false);
}

ANEBridgeStatus ane_bridge_compile_matmul(int in_ch, int out_ch, int spatial,
                                           ANEBridgeKernel **out_kernel) {
    return ane_bridge_compile_mm(out_ch, in_ch, in_ch, spatial, false, false, out_kernel);
}

ANEBridgeStatus ane_bridge_matmul_eval(ANEBridgeKernel *kernel,
                                        const float *W, const float *x, float *y,
                                        int in_ch, int out_ch, int spatial) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;
    if (kernel->nInputs != 2 || kernel->nOutputs != 1 ||
        kernel->inputBytes[0]  != (size_t)out_ch * in_ch * 4 ||
        kernel->inputBytes[1]  != (size_t)in_ch * spatial * 4 ||
        kernel->outputBytes[0] != (size_t)out_ch * spatial * 4) return ANE_ERR_INDEX_OOB;
    return ane_bridge_mm_eval(kernel, W, x, y);
}

ANEBridgeStatus ane_bridge_matmul_eval_rowmajor(ANEBridgeKernel *kernel,
                                                 const float *W, const float *x, float *y,
                                                 int in_ch, int out_ch, int spatial) {
    if (!kernel || !W || !x || !y) return ANE_ERR_NULL_KERNEL;

    // x: [spatial, in_ch] → [in_ch, spatial];  y: [out_ch, spatial] → [spatial, out_ch]
    float *x_cf = (float *)malloc((size_t)in_ch * spatial * sizeof(float));
    float *y_cf = (float *)malloc((size_t)out_ch * spatial * sizeof(float));
    if (!x_cf || !y_cf) { free(x_cf); free(y_cf); return ANE_ERR_ALLOC_FAILED; }
    for (int t = 0; t < spatial; t++)
        for (int i = 0; i < in_ch; i++)
            x_cf[i * spatial + t] = x[t * in_ch + i];

    ANEBridgeStatus s = ane_bridge_matmul_eval(kernel, W, x_cf, y_cf, in_ch, out_ch, spatial);
    if (s == ANE_OK)
        for (int t = 0; t < spatial; t++)
            for (int i = 0; i < out_ch; i++)
                y[t * out_ch + i] = y_cf[i * spatial + t];
    free(x_cf); free(y_cf);
    return s;
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
#import <mach/mach_time.h>
uint64_t ane_bridge_time_ns(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    return mach_absolute_time() * info.numer / info.denom;
}

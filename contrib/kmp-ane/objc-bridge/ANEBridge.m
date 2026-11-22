// ANEBridge.m — C-callable implementation wrapping ane_runtime.h / ane_mil_gen.h
// Compile: xcrun clang -O2 -c ANEBridge.m -o ANEBridge.o -fobjc-arc \
//          -framework Foundation -framework IOSurface
#import "ANEBridge.h"
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
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
    size_t       *inputBytes;
    size_t       *outputBytes;
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

        // Weight dictionary
        NSDictionary *wdict = nil;
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
        if (!desc) return ANE_ERR_COMPILE_FAILED;

        // Create in-memory model
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem,
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) return ANE_ERR_COMPILE_FAILED;

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
            [fm removeItemAtPath:td error:nil];
            return ANE_ERR_COMPILE_FAILED;
        }

        // Load
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
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

        // Create IOSurfaces
        k->ioInputs  = malloc(n_inputs  * sizeof(IOSurfaceRef));
        k->ioOutputs = malloc(n_outputs * sizeof(IOSurfaceRef));
        for (int i = 0; i < n_inputs;  i++)
            k->ioInputs[i]  = create_surface(input_bytes[i]);
        for (int i = 0; i < n_outputs; i++)
            k->ioOutputs[i] = create_surface(output_bytes[i]);

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
    IOSurfaceLock(kernel->ioInputs[index], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(kernel->ioInputs[index]), data, bytes);
    IOSurfaceUnlock(kernel->ioInputs[index], 0, NULL);
    return ANE_OK;
}

ANEBridgeStatus ane_bridge_read_output(ANEBridgeKernel *kernel,
                                        int index,
                                        void *data,
                                        size_t bytes) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;
    if (index < 0 || index >= kernel->nOutputs) return ANE_ERR_INDEX_OOB;
    IOSurfaceLock(kernel->ioOutputs[index], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(kernel->ioOutputs[index]), bytes);
    IOSurfaceUnlock(kernel->ioOutputs[index], kIOSurfaceLockReadOnly, NULL);
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
        free(kernel->ioInputs);
        free(kernel->ioOutputs);
        free(kernel->inputBytes);
        free(kernel->outputBytes);
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

char *ane_bridge_mil_gen_matmul(int in_ch, int out_ch, int spatial) {
    @autoreleasepool {
        NSString *mil = [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, %d, %d]> x, tensor<fp32, [1, %d, %d]> W) {\n"
            "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, %d, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"
            "        tensor<fp16, [1, %d, %d]> W16 = cast(dtype = to_fp16, x = W)[name = string(\"cast_W\")];\n"
            "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
            "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
            "        tensor<fp16, [1, %d, %d]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string(\"mm\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, %d, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n",
            in_ch, spatial, out_ch, in_ch,
            in_ch, spatial, out_ch, in_ch,
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
ANEBridgeStatus ane_bridge_compile_matmul(int in_ch, int out_ch, int spatial,
                                           ANEBridgeKernel **out_kernel) {
    if (!g_inited) { ANEBridgeStatus s = ane_bridge_init(); if (s) return s; }

    char *mil = ane_bridge_mil_gen_matmul(in_ch, out_ch, spatial);
    if (!mil) return ANE_ERR_COMPILE_FAILED;

    size_t mil_len = strlen(mil);
    // 2 inputs: x[1, in_ch, spatial], W[1, out_ch, in_ch]
    size_t inputSizes[2] = {
        (size_t)in_ch  * spatial * 4,  // x
        (size_t)out_ch * in_ch   * 4   // W
    };
    size_t outSize = (size_t)out_ch * spatial * 4;

    ANEBridgeStatus status = ane_bridge_compile(
        mil, mil_len, NULL, 0,
        2, inputSizes, 1, &outSize, out_kernel);
    free(mil);
    return status;
}

// ---------------------------------------------------------------------------
// Matmul eval — channel-first layout
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_matmul_eval(ANEBridgeKernel *kernel,
                                        const float *W, const float *x, float *y,
                                        int in_ch, int out_ch, int spatial) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;
    ANEBridgeStatus s;
    s = ane_bridge_write_input(kernel, 0, x, (size_t)in_ch * spatial * sizeof(float));
    if (s != ANE_OK) return s;
    s = ane_bridge_write_input(kernel, 1, W, (size_t)out_ch * in_ch * sizeof(float));
    if (s != ANE_OK) return s;
    s = ane_bridge_eval(kernel);
    if (s != ANE_OK) return s;
    if (!y) return ANE_OK;  // caller reads the output itself (see ane_bridge_matmul_eval_rowmajor)
    return ane_bridge_read_output(kernel, 0, y, (size_t)out_ch * spatial * sizeof(float));
}

// ---------------------------------------------------------------------------
// Matmul eval — row-major convenience (transposes automatically)
// W[out_ch, in_ch], x[spatial, in_ch] → y[spatial, out_ch]
// ---------------------------------------------------------------------------
ANEBridgeStatus ane_bridge_matmul_eval_rowmajor(ANEBridgeKernel *kernel,
                                                 const float *W, const float *x, float *y,
                                                 int in_ch, int out_ch, int spatial) {
    if (!kernel) return ANE_ERR_NULL_KERNEL;

    // Transpose x: [spatial, in_ch] → [in_ch, spatial]
    float *x_cf = (float *)malloc((size_t)in_ch * spatial * sizeof(float));
    if (!x_cf) return ANE_ERR_ALLOC_FAILED;
    for (int t = 0; t < spatial; t++)
        for (int i = 0; i < in_ch; i++)
            x_cf[i * spatial + t] = x[t * in_ch + i];

    ANEBridgeStatus s = ane_bridge_matmul_eval(kernel, W, x_cf, NULL, in_ch, out_ch, spatial);
    if (s != ANE_OK) { free(x_cf); return s; }

    // Read + transpose: [out_ch, spatial] → [spatial, out_ch]
    float *y_cf = (float *)malloc((size_t)out_ch * spatial * sizeof(float));
    if (!y_cf) { free(x_cf); return ANE_ERR_ALLOC_FAILED; }
    s = ane_bridge_read_output(kernel, 0, y_cf, (size_t)out_ch * spatial * sizeof(float));
    if (s == ANE_OK) {
        for (int t = 0; t < spatial; t++)
            for (int i = 0; i < out_ch; i++)
                y[t * out_ch + i] = y_cf[i * spatial + t];
    }
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

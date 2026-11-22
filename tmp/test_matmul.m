// test_matmul.m — Minimal test: can ANE compile a 2-input matmul kernel?
// Build: clang -O2 -fobjc-arc -fmodules -o test_matmul test_matmul.m \
//          -framework Foundation -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include <stdio.h>
#include <string.h>

static Class g_Desc, g_InMem, g_Req, g_IO;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/"
           "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_InMem = NSClassFromString(@"_ANEInMemoryModel");
    g_Req   = NSClassFromString(@"_ANERequest");
    g_IO    = NSClassFromString(@"_ANEIOSurfaceObject");
    printf("Classes: desc=%p inMem=%p req=%p io=%p\n",
           g_Desc, g_InMem, g_Req, g_IO);
}

static NSString *mil_gen_matmul(int in_ch, int out_ch, int spatial) {
    return [NSString stringWithFormat:
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
}

static int try_compile(int in_ch, int out_ch, int spatial) {
    printf("\n--- Trying matmul(%d, %d, spatial=%d) ---\n", in_ch, out_ch, spatial);

    NSString *mil = mil_gen_matmul(in_ch, out_ch, spatial);
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    printf("MIL text: %zu bytes\n", [milData length]);

    // Print first 200 chars of MIL for inspection
    NSString *preview = [mil substringToIndex:MIN(200, [mil length])];
    printf("MIL preview:\n%s...\n", [preview UTF8String]);

    NSError *e = nil;

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, nil, nil);
    printf("Descriptor: %s\n", desc ? "OK" : "FAILED");
    if (!desc) return -1;

    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
        g_InMem, @selector(inMemoryModelWithDescriptor:), desc);
    printf("Model: %s\n", mdl ? "OK" : "FAILED");
    if (!mdl) return -2;

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager]
        createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    printf("Compile: %s", ok ? "OK" : "FAILED");
    if (e) printf(" (%s)", [[e description] UTF8String]);
    printf("\n");

    if (ok) {
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("Load: %s", ok ? "OK" : "FAILED");
        if (e) printf(" (%s)", [[e description] UTF8String]);
        printf("\n");

        // Unload+clean
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
    }

    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
    return ok ? 0 : -3;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        ane_init();

        // Test progressively larger dimensions
        try_compile(4, 4, 4);       // Tiny
        try_compile(16, 8, 16);     // Small
        try_compile(128, 64, 32);   // Medium
        try_compile(128, 10, 128);  // Layer 2 size (should work)
        try_compile(784, 128, 128); // Layer 1 size (MNIST)
        try_compile(784, 128, 1);   // Single sample
    }
    return 0;
}

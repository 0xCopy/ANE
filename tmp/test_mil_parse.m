// test_mil_parse.m — Compare matmul vs conv MIL parsing
// Build: clang -O2 -fobjc-arc -fmodules -o test_mil_parse test_mil_parse.m \
//          -framework Foundation -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <stdio.h>

static Class g_Desc, g_InMem;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/"
           "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_InMem = NSClassFromString(@"_ANEInMemoryModel");
}

static void test_mil(const char *label, NSString *mil) {
    NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
    printf("\n=== %s (%zu bytes) ===\n", label, [milData length]);
    printf("%s\n", [mil UTF8String]);

    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
        g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
        milData, nil, nil);
    printf("Result: %s\n", desc ? "OK" : "FAILED");
    if (desc) printf("  desc: %s\n", [[desc description] UTF8String]);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        ane_init();

        // Test 1: Known-working conv MIL (1 input, no weights — will fail at compile but desc should work)
        test_mil("Conv 4→4 (no weight blob)", [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x) {\n"
            "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, 4, 1, 4]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to_fp32, x = x16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n"]);

        // Test 2: Simplest 2-input passthrough (no matmul op)
        test_mil("2-input passthrough", [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, 4, 4]> x, tensor<fp32, [1, 4, 4]> W) {\n"
            "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, 4, 4]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, 4, 4]> y = cast(dtype = to_fp32, x = x16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n"]);

        // Test 3: 3D tensor with matmul
        test_mil("3D matmul 4x4x4", [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, 4, 4]> x, tensor<fp32, [1, 4, 4]> W) {\n"
            "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, 4, 4]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_x\")];\n"
            "        tensor<fp16, [1, 4, 4]> W16 = cast(dtype = to_fp16, x = W)[name = string(\"cast_W\")];\n"
            "        bool tx = const()[name = string(\"tx\"), val = bool(false)];\n"
            "        bool ty = const()[name = string(\"ty\"), val = bool(false)];\n"
            "        tensor<fp16, [1, 4, 4]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string(\"mm\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, 4, 4]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n"]);

        // Test 4: 4D tensor [1, C, 1, S] (conv-compatible shape) with cast only, 2 inputs
        test_mil("4D 2-input passthrough", [NSString stringWithFormat:
            @"program(1.3)\n"
            "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
            "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
            "{\"coremltools-version\", \"9.0\"}})]\n"
            "{\n"
            "    func main<ios18>(tensor<fp32, [1, 4, 1, 4]> x, tensor<fp32, [1, 4, 1, 4]> W) {\n"
            "        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"
            "        tensor<fp16, [1, 4, 1, 4]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n"
            "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
            "        tensor<fp32, [1, 4, 1, 4]> y = cast(dtype = to_fp32, x = x16)[name = string(\"cast_out\")];\n"
            "    } -> (y);\n"
            "}\n"]);
    }
    return 0;
}

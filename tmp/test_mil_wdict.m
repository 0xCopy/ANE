// test_mil_wdict.m — Test if descriptor needs a weights dictionary even with no weights
// Build: clang -O2 -fobjc-arc -fmodules -o test_mil_wdict test_mil_wdict.m \
//          -framework Foundation -framework IOSurface -ldl
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#include <stdio.h>

static Class g_Desc;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/"
           "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
}

static NSString *simple_mil(void) {
    return @"program(1.3)\n"
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
           "}\n";
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        ane_init();

        NSString *mil = simple_mil();
        NSData *milData = [mil dataUsingEncoding:NSUTF8StringEncoding];
        printf("MIL text: %zu bytes\n\n", [milData length]);

        // Test 1: nil weights
        printf("Test 1: weights=nil\n");
        id desc1 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, nil, nil);
        printf("  Result: %s\n", desc1 ? "OK" : "FAILED");

        // Test 2: empty dictionary
        printf("Test 2: weights=@{}\n");
        id desc2 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        printf("  Result: %s\n", desc2 ? "OK" : "FAILED");

        // Test 3: empty bytes (byte-sized empty data at the expected path)
        printf("Test 3: weights with empty data at expected key\n");
        NSData *emptyData = [NSData data];
        NSDictionary *wdict3 = @{@"@model_path/weights/weight.bin":
                                    @{@"offset": @0, @"data": emptyData}};
        id desc3 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict3, nil);
        printf("  Result: %s\n", desc3 ? "OK" : "FAILED");

        // Test 4: options=nil vs options=@{}
        printf("Test 4: options=@{} (instead of nil)\n");
        id desc4 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, nil, @{});
        printf("  Result: %s\n", desc4 ? "OK" : "FAILED");

        // Test 5: Use NSString directly instead of NSData for milText?
        printf("Test 5: milText as NSString (not NSData)\n");
        id desc5 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, nil, nil);
        printf("  Result: %s\n", desc5 ? "OK" : "FAILED");

        // Test 6: Both NSString and empty weights
        printf("Test 6: milText=NSString, weights=@{}\n");
        id desc6 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, @{}, nil);
        printf("  Result: %s\n", desc6 ? "OK" : "FAILED");

        // Test 7: NSString + empty weight data
        printf("Test 7: milText=NSString, weights with empty data\n");
        id desc7 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            mil, wdict3, nil);
        printf("  Result: %s\n", desc7 ? "OK" : "FAILED");
    }
    return 0;
}

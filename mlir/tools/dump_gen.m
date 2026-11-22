#include "ane_mil_gen.h"
#define main bench_main
#include "../ane_int8_bench.m"
#undef main
int main(int c,char**v){@autoreleasepool{ int ch=atoi(v[1]), o=atoi(v[2]), sp=atoi(v[3]);
 printf("=== mil_gen_matmul\n%s", [mil_gen_matmul(ch,o,sp) UTF8String]);
 printf("=== mil_gen_conv\n%s", [mil_gen_conv(ch,o,sp) UTF8String]);
 printf("=== mil_gen_qkv\n%s", [mil_gen_qkv(ch,sp) UTF8String]);
 printf("=== mil_gen_ffn_up\n%s", [mil_gen_ffn_up(ch,o,sp) UTF8String]);
 printf("=== genMILInt8\n%s", [genMILInt8(ch,sp,3) UTF8String]);
 printf("=== genMILFP16\n%s", [genMILFP16(ch,sp,3) UTF8String]);
}return 0;}

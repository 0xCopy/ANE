#include "mil_dynamic.h"
#define D(n) printf("=== %s\n%s", #n, [n() UTF8String])
int main(void){@autoreleasepool{
 printf("=== gen_dyn_matmul_mil\n%s", [gen_dyn_matmul_mil(DIM, HIDDEN, SEQ) UTF8String]);
 D(gen_sdpa_fwd_dynamic); D(gen_wo_fwd_dynamic); D(gen_ffn_fused_dynamic); D(gen_ffn_bwd_w2t_dynamic); D(gen_ffn_bwd_w13t_dynamic);
 D(gen_wot_dynamic); D(gen_q_bwd_dynamic); D(gen_kv_bwd_dynamic); D(gen_sdpa_bwd1_noweight); D(gen_sdpa_bwd2);
}return 0;}

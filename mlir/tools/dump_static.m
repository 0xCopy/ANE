#include "stories_mil.h"
#include "ane_rmsnorm_bwd.h"
#include "ane_classifier.h"
#define D(n) printf("=== %s\n%s", #n, [n() UTF8String])
int main(void){@autoreleasepool{
 D(gen_sdpa_fwd_taps); D(gen_ffn_fwd_taps); D(gen_ffn_bwd); D(gen_qkvb); D(gen_sdpa_bwd1); D(gen_sdpa_bwd2);
 D(gen_rmsnorm_bwd); D(gen_classifier_fwd); D(gen_classifier_bwd); D(gen_softmax_vocab); D(gen_final_rmsnorm);
}return 0;}

// generate.m — Text generation (inference) on ANE
// Uses the dynamic pipeline's forward kernels to generate text from a checkpoint.
// Usage: ./generate --ckpt <path> [options]
//
// Forward-only: compiles 3 ANE kernels (sdpaFwd, woFwd, ffnFused) once,
// loads checkpoint weights, then runs forward pass + sampling per generated token.
// No backward, no optimizer — pure inference.
#include "mil_dynamic.h"
#include "cpu_ops.h"

// Transpose W[rows,cols] -> W^T[cols,rows] stored as [cols channels, rows spatial]
static void transpose_weight(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

// We only need the 3 forward kernels for generation
typedef struct {
    Kern *sdpaFwd;
    Kern *woFwd;
    Kern *ffnFused;
} GenKernels;

static void print_usage(const char *prog) {
    printf("Usage: %s --ckpt PATH [options]\n", prog);
    printf("\n");
    printf("Text generation on ANE using the dynamic forward pipeline.\n");
    printf("Model: %s (compiled at build time)\n", MODEL_NAME);
    printf("\n");
    printf("Required:\n");
    printf("  --ckpt PATH      Checkpoint file from training (--resume)\n");
    printf("\n");
    printf("Options:\n");
    printf("  --tokens N       Number of new tokens to generate (default: 100)\n");
    printf("  --temp F         Sampling temperature, 0 = greedy (default: 0.0)\n");
    printf("  --topk K         Top-K sampling, 0 = disabled (default: 0)\n");
    printf("  --data PATH      Tokenized data for prompt lookup (default: %s)\n", DEFAULT_DATA_PATH);
    printf("  --seed N         Random seed for sampling (default: 42)\n");
    printf("  --help, -h       Show this help\n");
    printf("\n");
    printf("Prompt is taken from the first SEQ tokens of the data file.\n");
    printf("Use --temp 0 for greedy decoding, or --temp 0.8 --topk 40 for sampling.\n");
}

// Minimal llama2.c tokenizer for detokenization only.
// Format: max_token_length (i32), then vocab_size entries.
// Each entry: score (f32), then token string (max_token_len bytes, padded with nulls).
typedef struct {
    char **vocab;     // [vocab_size] string pointers
    int vocab_size;
} Tokenizer;

static Tokenizer *load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int max_token_len;
    if (fread(&max_token_len, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    Tokenizer *tk = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    tk->vocab_size = VOCAB;
    tk->vocab = (char**)calloc(tk->vocab_size, sizeof(char*));

    for (int i = 0; i < tk->vocab_size; i++) {
        float score;
        if (fread(&score, sizeof(float), 1, f) != 1) {
            tk->vocab_size = i;
            break;
        }
        char buf[512];
        int j = 0;
        while (j < max_token_len) {
            int c = fgetc(f);
            if (c == EOF) break;
            buf[j] = (char)c;
            j++;
        }
        buf[j] = '\0';
        tk->vocab[i] = strdup(buf);
    }
    fclose(f);
    return tk;
}

static void free_tokenizer(Tokenizer *tk) {
    if (!tk) return;
    for (int i = 0; i < tk->vocab_size; i++) free(tk->vocab[i]);
    free(tk->vocab);
    free(tk);
}

static void detokenize(Tokenizer *tk, uint16_t token, bool is_start) {
    if (!tk || token >= (uint16_t)tk->vocab_size) {
        printf("[%d]", token);
        return;
    }
    const char *s = tk->vocab[token];
    if (is_start && s[0] == ' ') s++;
    printf("%s", s);
}

// Load checkpoint — same format as train.m, but we only need weights (skip Adam state)
static bool load_checkpoint_weights(const char *path,
                                     LayerWeights *lw, float *rms_final, float *embed) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54 || h.version != 4) {
        printf("ERROR: invalid checkpoint (magic=0x%X version=%d, expected 0x424C5A54 v4)\n",
               h.magic, h.version);
        fclose(f); return false;
    }
    printf("  Checkpoint: step %d, loss=%.4f, model=%s (%d layers)\n",
           h.step, h.loss, MODEL_NAME, h.n_layers);

    for (int L = 0; L < NLAYERS; L++) {
        fread(lw[L].Wq, 4, WQ_SZ, f); fread(lw[L].Wk, 4, WK_SZ, f);
        fread(lw[L].Wv, 4, WV_SZ, f); fread(lw[L].Wo, 4, WO_SZ, f);
        fread(lw[L].W1, 4, W1_SZ, f); fread(lw[L].W2, 4, W2_SZ, f); fread(lw[L].W3, 4, W3_SZ, f);
        fread(lw[L].rms_att, 4, DIM, f); fread(lw[L].rms_ffn, 4, DIM, f);
        // Skip Adam state for this layer (m and v, same sizes as weights)
        fseek(f, (long)(2*(WQ_SZ + WK_SZ + WV_SZ + WO_SZ + W1_SZ + W2_SZ + W3_SZ + 2*DIM)) * 4, SEEK_CUR);
    }
    fread(rms_final, 4, DIM, f);
    fseek(f, (long)(2*DIM) * 4, SEEK_CUR);
    fread(embed, 4, VOCAB*DIM, f);
    fseek(f, (long)(2*(size_t)VOCAB*DIM) * 4, SEEK_CUR);

    fclose(f);
    return true;
}

static bool compile_gen_kernels(GenKernels *gk) {
    NSDictionary *sdpa_fwd_w = @{
        @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
        @"@model_path/weights/rope_cos.bin": @{@"offset":@0, @"data":get_rope_cos_blob()},
        @"@model_path/weights/rope_sin.bin": @{@"offset":@0, @"data":get_rope_sin_blob()}
    };

    int sdpa_out_ch = Q_DIM + Q_DIM + KV_DIM + KV_DIM + DIM;
    printf("  Compiling sdpaFwd...\n");
    gk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), sdpa_fwd_w,
        DIM*SDPA_FWD_SP*2, sdpa_out_ch*SEQ*2);
    if (!gk->sdpaFwd) return false;

    printf("  Compiling woFwd...\n");
    gk->woFwd = compile_kern_mil_w(gen_wo_fwd_dynamic(), @{},
        Q_DIM*WO_FWD_SP*2, DIM*SEQ*2);
    if (!gk->woFwd) return false;

    printf("  Compiling ffnFused...\n");
    int ffn_och = DIM + 3*HIDDEN;
    gk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, ffn_och*SEQ*2);
    if (!gk->ffnFused) return false;

    return true;
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        const char *ckpt_path = NULL;
        const char *data_path = DEFAULT_DATA_PATH;
        int n_gen = 100;
        float temperature = 0.0f;
        int top_k = 0;
        uint32_t seed = 42;

        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--ckpt") == 0 && i+1 < argc)
                ckpt_path = argv[++i];
            else if (strcmp(argv[i], "--data") == 0 && i+1 < argc)
                data_path = argv[++i];
            else if (strcmp(argv[i], "--tokens") == 0 && i+1 < argc)
                n_gen = atoi(argv[++i]);
            else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc)
                temperature = atof(argv[++i]);
            else if (strcmp(argv[i], "--topk") == 0 && i+1 < argc)
                top_k = atoi(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
                seed = (uint32_t)strtoul(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            }
        }
        if (!ckpt_path) {
            printf("ERROR: --ckpt is required\n\n");
            print_usage(argv[0]);
            return 2;
        }

        srand(seed);
        srand48(seed);

        LayerWeights lw[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) lw[L] = layer_weights_alloc();
        float *rms_final = (float*)malloc(DIM*4);
        float *embed = (float*)malloc(VOCAB*DIM*4);

        printf("Loading checkpoint: %s\n", ckpt_path);
        if (!load_checkpoint_weights(ckpt_path, lw, rms_final, embed)) {
            printf("Failed to load checkpoint\n");
            return 1;
        }
        float res_alpha = 1.0f / sqrtf(2.0f * NLAYERS);

        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
        float *W1t_buf[NLAYERS], *W2t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            Wqt_buf[L] = (float*)malloc(WQ_SZ*4); Wkt_buf[L] = (float*)malloc(WK_SZ*4);
            Wvt_buf[L] = (float*)malloc(WV_SZ*4); Wot_buf[L] = (float*)malloc(WO_SZ*4);
            W1t_buf[L] = (float*)malloc(W1_SZ*4); W2t_buf[L] = (float*)malloc(W2_SZ*4);
            W3t_buf[L] = (float*)malloc(W3_SZ*4);
            transpose_weight(Wqt_buf[L], lw[L].Wq, Q_DIM, DIM);
            transpose_weight(Wkt_buf[L], lw[L].Wk, KV_DIM, DIM);
            transpose_weight(Wvt_buf[L], lw[L].Wv, KV_DIM, DIM);
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, Q_DIM);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W2t_buf[L], lw[L].W2, DIM, HIDDEN);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }

        Tokenizer *tk = load_tokenizer("tokenizer.bin");
        if (!tk) tk = load_tokenizer("../tokenizer.bin");
        if (!tk) tk = load_tokenizer("../../assets/models/tokenizer.bin");
        if (!tk) {
            printf("Warning: tokenizer.bin not found. Output will show token IDs only.\n");
            printf("  Place tokenizer.bin in the working directory or assets/models/.\n\n");
        }

        uint16_t prompt_tokens[SEQ];
        int n_prompt = 0;

        int data_fd = open(data_path, O_RDONLY);
        if (data_fd >= 0) {
            struct stat st; fstat(data_fd, &st);
            size_t data_len = st.st_size;
            uint16_t *token_data = (uint16_t*)mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, data_fd, 0);
            if (token_data != MAP_FAILED) {
                size_t n_tokens = data_len / 2;
                n_prompt = n_tokens >= (size_t)SEQ ? SEQ : (int)n_tokens;
                for (int t = 0; t < n_prompt; t++) prompt_tokens[t] = token_data[t];
                munmap(token_data, data_len);
            }
            close(data_fd);
        }
        if (n_prompt == 0) {
            prompt_tokens[0] = 1;
            n_prompt = 1;
            printf("No data file, using BOS token as prompt.\n");
        }

        printf("Compiling 3 forward kernels...\n");
        uint64_t tc = mach_absolute_time();
        GenKernels gk;
        if (!compile_gen_kernels(&gk)) {
            printf("Compilation failed!\n");
            return 1;
        }
        double compile_ms = tb_ms(mach_absolute_time() - tc);
        printf("Compiled in %.0fms\n", compile_ms);

        PerLayerSurfaces pls[NLAYERS];
        PerLayerRequests plr[NLAYERS];
        for (int L = 0; L < NLAYERS; L++) {
            pls[L].sdpaFwd_in = make_surface(DIM*SDPA_FWD_SP*2);
            pls[L].woFwd_in = make_surface(Q_DIM*WO_FWD_SP*2);
            pls[L].ffnFused_in = make_surface(DIM*FFN_FUSED_SP*2);
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
            plr[L].sdpaFwd = make_request(gk.sdpaFwd, pls[L].sdpaFwd_in);
            plr[L].woFwd = make_request(gk.woFwd, pls[L].woFwd_in);
            plr[L].ffnFused = make_request(gk.ffnFused, pls[L].ffnFused_in);
        }

        float *x_cur = (float*)malloc(SEQ*DIM*4);
        float *xnorm_buf = (float*)malloc(SEQ*DIM*4);
        float *x_final = (float*)malloc(SEQ*DIM*4);
        float *attn_out = (float*)malloc(SEQ*Q_DIM*4);
        float *o_out = (float*)malloc(SEQ*DIM*4);
        float *x2 = (float*)malloc(SEQ*DIM*4);
        float *x2norm = (float*)malloc(SEQ*DIM*4);
        float *logits = (float*)malloc(VOCAB*4);

        uint16_t tokens[SEQ];
        for (int t = 0; t < n_prompt; t++) tokens[t] = prompt_tokens[t];
        for (int t = n_prompt; t < SEQ; t++) tokens[t] = 1;

        printf("\n=== Generation (temp=%.1f, topk=%d, %d tokens) ===\n", temperature, top_k, n_gen);
        printf("Prompt: ");
        for (int t = 0; t < n_prompt; t++) detokenize(tk, tokens[t], t == 0);
        printf("\n\n");

        uint64_t t_start = mach_absolute_time();

        for (int gen = 0; gen < n_gen; gen++) {
            embed_lookup(x_cur, embed, tokens, DIM, SEQ);

            for (int L = 0; L < NLAYERS; L++) {
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);

                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
                ane_eval_req(gk.sdpaFwd, plr[L].sdpaFwd);

                IOSurfaceLock(gk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(gk.sdpaFwd->ioOut);
                int off = 0;
                cvt_f16_f32(attn_out, fwd_out + off, Q_DIM*SEQ);
                off += Q_DIM*SEQ;
                off += Q_DIM*SEQ + KV_DIM*SEQ + KV_DIM*SEQ + DIM*SEQ;
                IOSurfaceUnlock(gk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);

                write_wo_fwd_acts(pls[L].woFwd_in, attn_out);
                ane_eval_req(gk.woFwd, plr[L].woFwd);
                io_read_dyn(gk.woFwd->ioOut, o_out, DIM, SEQ);

                vDSP_vsma(o_out, 1, &res_alpha, x_cur, 1, x2, 1, (vDSP_Length)(SEQ*DIM));
                rmsnorm(x2norm, x2, lw[L].rms_ffn, DIM, SEQ);

                write_ffn_fused_acts(pls[L].ffnFused_in, x2norm, x2);
                ane_eval_req(gk.ffnFused, plr[L].ffnFused);

                IOSurfaceLock(gk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(gk.ffnFused->ioOut);
                cvt_f16_f32(x_cur, ffn_out, DIM*SEQ);
                IOSurfaceUnlock(gk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
            }

            rmsnorm(x_final, x_cur, rms_final, DIM, SEQ);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        VOCAB, 1, DIM, 1.0f, embed, DIM,
                        x_final + (SEQ-1)*DIM, 1, 0.0f, logits, 1);

            int next_token;
            if (temperature == 0.0f) {
                next_token = 0;
                float max_logit = logits[0];
                for (int v = 1; v < VOCAB; v++) {
                    if (logits[v] > max_logit) { max_logit = logits[v]; next_token = v; }
                }
            } else {
                float *scaled = (float*)malloc(VOCAB*4);
                float inv_temp = 1.0f / temperature;
                for (int v = 0; v < VOCAB; v++) scaled[v] = logits[v] * inv_temp;

                if (top_k > 0 && top_k < VOCAB) {
                    float *tmp = (float*)malloc(VOCAB*4);
                    memcpy(tmp, scaled, VOCAB*4);
                    for (int k = 0; k < top_k; k++) {
                        int max_i = k;
                        for (int j = k+1; j < VOCAB; j++)
                            if (tmp[j] > tmp[max_i]) max_i = j;
                        float t = tmp[k]; tmp[k] = tmp[max_i]; tmp[max_i] = t;
                    }
                    float threshold = tmp[top_k-1];
                    free(tmp);
                    for (int v = 0; v < VOCAB; v++)
                        if (scaled[v] < threshold) scaled[v] = -INFINITY;
                }

                float max_l = -INFINITY;
                for (int v = 0; v < VOCAB; v++)
                    if (scaled[v] > max_l) max_l = scaled[v];
                float sum_exp = 0;
                for (int v = 0; v < VOCAB; v++) {
                    scaled[v] = expf(scaled[v] - max_l);
                    sum_exp += scaled[v];
                }
                float r = (float)drand48();
                float cum = 0;
                next_token = VOCAB - 1;
                for (int v = 0; v < VOCAB; v++) {
                    cum += scaled[v] / sum_exp;
                    if (r < cum) { next_token = v; break; }
                }
                free(scaled);
            }

            detokenize(tk, (uint16_t)next_token, false);
            fflush(stdout);

            for (int t = 0; t < SEQ - 1; t++) tokens[t] = tokens[t+1];
            tokens[SEQ-1] = (uint16_t)next_token;
        }

        double total_gen_ms = tb_ms(mach_absolute_time() - t_start);
        printf("\n\n=== Done: %d tokens in %.0fms (%.1fms/token) ===\n",
               n_gen, total_gen_ms, total_gen_ms / n_gen);

        for (int L = 0; L < NLAYERS; L++) {
            layer_weights_free(&lw[L]);
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
            free(W1t_buf[L]); free(W2t_buf[L]); free(W3t_buf[L]);
            CFRelease(pls[L].sdpaFwd_in); CFRelease(pls[L].woFwd_in); CFRelease(pls[L].ffnFused_in);
            CFRelease(plr[L].sdpaFwd); CFRelease(plr[L].woFwd); CFRelease(plr[L].ffnFused);
        }
        free(rms_final); free(embed);
        free_kern(gk.sdpaFwd); free_kern(gk.woFwd); free_kern(gk.ffnFused);
        free(x_cur); free(xnorm_buf); free(x_final); free(attn_out); free(o_out);
        free(x2); free(x2norm); free(logits);
        free_tokenizer(tk);
    }
    return 0;
}

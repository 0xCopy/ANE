// mnist_ane.m — End-to-end MNIST training on Apple Neural Engine
//
// Architecture: Linear(784→128) → ReLU → Linear(128→10) → Softmax+CE
// Forward:  ANE 1×1 conv kernels for both linear layers (baked fp16 weights)
// Backward: CPU (naive loops — fast enough for MNIST-sized matrices)
// Optimizer: SGD with momentum
//
// Strategy: use gradient accumulation to amortize the recompile cost.
// Each weight update requires recompiling 2 kernels (~119 compile limit/process).
// With ACCUM_STEPS=50 and batch=128: ~6 full epochs before hitting the limit.
//
// Build:  make mnist
// Usage:  python3 fetch_mnist.py && ./mnist_ane
//
// Requires macOS 15+ on Apple Silicon.

#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mach/mach_time.h>

// ANE runtime from the main training directory
#include "../../../training/ane_runtime.h"
#include "../../../training/ane_mil_gen.h"

int g_fp16_io = 0;  // M1/M2: set to 1 when the cast op is unsupported (declared extern in ane_mil_gen.h)


// ─── Configuration ──────────────────────────────────────────────────────────
#define IN_DIM      784
#define HID_DIM     128
#define OUT_DIM     10
#define BATCH       128
#define LR          0.01f
#define MOM         0.9f
#define N_EPOCHS    6
#define ACCUM_STEPS 50    // gradient accumulation steps per weight update
#define MAX_COMPILES 118  // leave 1 spare of the ~119 ANE limit

// ─── MNIST IDX file parser ──────────────────────────────────────────────────
static uint32_t read_be32(FILE *f) {
    uint8_t b[4];
    fread(b, 1, 4, f);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

typedef struct {
    float *images;   // [n, 784] row-major, normalized [0,1]
    int   *labels;   // [n]
    int    n;
} MnistSet;

static MnistSet mnist_load(const char *img_path, const char *lbl_path) {
    MnistSet s = {0};
    FILE *fi = fopen(img_path, "rb");
    FILE *fl = fopen(lbl_path, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "ERROR: Cannot open MNIST files.\n");
        fprintf(stderr, "  Expected: %s\n           %s\n", img_path, lbl_path);
        fprintf(stderr, "  Run: python3 fetch_mnist.py\n");
        if (fi) fclose(fi);
        if (fl) fclose(fl);
        return s;
    }

    read_be32(fi); s.n = (int)read_be32(fi);
    read_be32(fi); read_be32(fi); // rows, cols
    read_be32(fl); read_be32(fl);

    s.images = (float *)malloc((size_t)s.n * IN_DIM * sizeof(float));
    s.labels = (int *)malloc((size_t)s.n * sizeof(int));

    uint8_t *raw = (uint8_t *)malloc((size_t)s.n * IN_DIM);
    fread(raw, 1, (size_t)s.n * IN_DIM, fi);
    for (int i = 0; i < s.n * IN_DIM; i++)
        s.images[i] = raw[i] / 255.0f;
    free(raw);

    uint8_t *rlbl = (uint8_t *)malloc((size_t)s.n);
    fread(rlbl, 1, (size_t)s.n, fl);
    for (int i = 0; i < s.n; i++)
        s.labels[i] = rlbl[i];
    free(rlbl);

    fclose(fi); fclose(fl);
    printf("Loaded %d samples from %s\n", s.n, img_path);
    return s;
}

// ─── Utilities ──────────────────────────────────────────────────────────────
static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

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

// ─── Compile a baked-weight conv kernel ─────────────────────────────────────
// Input:  [1, in_ch, 1, spatial] fp32
// Weight: [out_ch, in_ch, 1, 1] fp16 baked into BLOBFILE
// Output: [1, out_ch, 1, spatial] fp32
static ANEKernel *compile_conv(const float *weights, int in_ch, int out_ch, int spatial) {
    @autoreleasepool {
        NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
        NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
        size_t inBytes  = (size_t)in_ch  * spatial * 4;
        size_t outBytes = (size_t)out_ch * spatial * 4;
        return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding],
                           wb, 1, &inBytes, 1, &outBytes);
    }
}

// ─── Network ────────────────────────────────────────────────────────────────
typedef struct {
    float *W1, *W2;        // [out, in] row-major weights
    float *dW1, *dW2;      // accumulated gradients
    float *vW1, *vW2;      // SGD momentum

    ANEKernel *kern1;      // conv: 784→128 (baked W1)
    ANEKernel *kern2;      // conv: 128→10  (baked W2)
    int compile_count;     // track total ANE compiles

    // Activations (channel-first: [channels, batch])
    float *x_cf;           // [IN_DIM, BATCH]  input
    float *h_pre;          // [HID_DIM, BATCH] before ReLU
    float *h_post;         // [HID_DIM, BATCH] after ReLU
    float *logits;         // [OUT_DIM, BATCH]
} Net;

static void net_init(Net *net) {
    memset(net, 0, sizeof(Net));

    net->W1  = (float *)calloc(HID_DIM * IN_DIM,  sizeof(float));
    net->W2  = (float *)calloc(OUT_DIM * HID_DIM, sizeof(float));
    net->dW1 = (float *)calloc(HID_DIM * IN_DIM,  sizeof(float));
    net->dW2 = (float *)calloc(OUT_DIM * HID_DIM, sizeof(float));
    net->vW1 = (float *)calloc(HID_DIM * IN_DIM,  sizeof(float));
    net->vW2 = (float *)calloc(OUT_DIM * HID_DIM, sizeof(float));

    // He initialization
    float s1 = sqrtf(2.0f / IN_DIM);
    float s2 = sqrtf(2.0f / HID_DIM);
    for (int i = 0; i < HID_DIM * IN_DIM; i++)  net->W1[i] = randn() * s1;
    for (int i = 0; i < OUT_DIM * HID_DIM; i++) net->W2[i] = randn() * s2;

    net->x_cf   = (float *)calloc(IN_DIM  * BATCH, sizeof(float));
    net->h_pre  = (float *)calloc(HID_DIM * BATCH, sizeof(float));
    net->h_post = (float *)calloc(HID_DIM * BATCH, sizeof(float));
    net->logits = (float *)calloc(OUT_DIM * BATCH, sizeof(float));
}

static int net_compile(Net *net) {
    // Free old kernels if any
    if (net->kern1) { ane_free(net->kern1); net->kern1 = NULL; }
    if (net->kern2) { ane_free(net->kern2); net->kern2 = NULL; }

    net->kern1 = compile_conv(net->W1, IN_DIM, HID_DIM, BATCH);
    if (!net->kern1) return -1;
    net->compile_count++;

    net->kern2 = compile_conv(net->W2, HID_DIM, OUT_DIM, BATCH);
    if (!net->kern2) return -2;
    net->compile_count++;

    return 0;
}

// Forward pass — returns cross-entropy loss
static float net_forward(Net *net, const float *batch_images, const int *batch_labels) {
    // Transpose images to channel-first [IN_DIM, BATCH]
    transpose(net->x_cf, batch_images, BATCH, IN_DIM);

    // ── Layer 1: ANE conv ───────────────────────────────────────────────
    ane_write_input(net->kern1, 0, net->x_cf, (size_t)IN_DIM * BATCH * 4);
    ane_eval(net->kern1);
    ane_read_output(net->kern1, 0, net->h_pre, (size_t)HID_DIM * BATCH * 4);

    // ── ReLU (CPU) ──────────────────────────────────────────────────────
    for (int i = 0; i < HID_DIM * BATCH; i++)
        net->h_post[i] = net->h_pre[i] > 0 ? net->h_pre[i] : 0;

    // ── Layer 2: ANE conv ───────────────────────────────────────────────
    ane_write_input(net->kern2, 0, net->h_post, (size_t)HID_DIM * BATCH * 4);
    ane_eval(net->kern2);
    ane_read_output(net->kern2, 0, net->logits, (size_t)OUT_DIM * BATCH * 4);

    // ── Cross-entropy loss (CPU) ────────────────────────────────────────
    // logits: [OUT_DIM, BATCH] channel-first → logits[c * BATCH + b]
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

// Accumulate gradients into net->dW1, net->dW2 (does NOT zero them)
static void net_accumulate_gradients(Net *net, const int *batch_labels) {
    // dLogits = softmax(logits) − one_hot(labels)
    float *dL = (float *)calloc(OUT_DIM * BATCH, sizeof(float));
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

    // d_h_post = W2^T @ dL → ReLU backward → dW1 +=
    float *d_hp = (float *)calloc(HID_DIM * BATCH, sizeof(float));
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

// Apply accumulated gradients with SGD+momentum, then zero grad buffers
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
    memset(net->dW1, 0, (size_t)HID_DIM * IN_DIM * sizeof(float));
    memset(net->dW2, 0, (size_t)OUT_DIM * HID_DIM * sizeof(float));
}

// Count correct predictions from the last forward pass
static int net_count_correct(Net *net, const int *batch_labels) {
    int correct = 0;
    for (int b = 0; b < BATCH; b++) {
        int best = 0;
        float best_v = net->logits[b]; // logits[0 * BATCH + b]
        for (int c = 1; c < OUT_DIM; c++) {
            float v = net->logits[c * BATCH + b];
            if (v > best_v) { best_v = v; best = c; }
        }
        if (best == batch_labels[b]) correct++;
    }
    return correct;
}

// Evaluate accuracy on a dataset
static float net_eval_accuracy(Net *net, const float *images, const int *labels, int n) {
    int correct = 0, total = 0;
    float *bimg = (float *)malloc((size_t)BATCH * IN_DIM * sizeof(float));
    int   *blbl = (int *)malloc((size_t)BATCH * sizeof(int));

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
    if (net->kern1) ane_free(net->kern1);
    if (net->kern2) ane_free(net->kern2);
    free(net->W1);  free(net->W2);
    free(net->dW1); free(net->dW2);
    free(net->vW1); free(net->vW2);
    free(net->x_cf); free(net->h_pre); free(net->h_post); free(net->logits);
}

// ─── Main ───────────────────────────────────────────────────────────────────
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        srand((unsigned)time(NULL));

        printf("╔══════════════════════════════════════════════════════╗\n");
        printf("║      MNIST Training on Apple Neural Engine          ║\n");
        printf("╠══════════════════════════════════════════════════════╣\n");
        printf("║  Network:    %d → %d (ReLU) → %d                 ║\n", IN_DIM, HID_DIM, OUT_DIM);
        printf("║  Batch:      %-4d  LR: %.4f  Mom: %.2f           ║\n", BATCH, LR, MOM);
        printf("║  Epochs:     %-2d    Accum: %d steps                 ║\n", N_EPOCHS, ACCUM_STEPS);
        printf("║  ANE mode:   conv (baked fp16 weights, recompile)   ║\n");
        printf("╚══════════════════════════════════════════════════════╝\n\n");

        // ── Load MNIST data ─────────────────────────────────────────────
        MnistSet train = mnist_load("data/train-images-idx3-ubyte",
                                    "data/train-labels-idx1-ubyte");
        MnistSet test  = mnist_load("data/t10k-images-idx3-ubyte",
                                    "data/t10k-labels-idx1-ubyte");
        if (!train.images || !test.images) {
            fprintf(stderr, "Failed to load data. Aborting.\n");
            return 1;
        }
        printf("\n");

        // ── Initialize network ──────────────────────────────────────────
        Net net;
        net_init(&net);

        // ── Initial kernel compilation ──────────────────────────────────
        printf("Compiling initial ANE conv kernels...\n");
        if (net_compile(&net) != 0) {
            fprintf(stderr, "ANE compilation failed. Is this macOS 15+ on Apple Silicon?\n");
            return 1;
        }
        printf("  2 conv kernels compiled (%d/%d budget)\n", net.compile_count, MAX_COMPILES);
        printf("  Max weight updates: %d (x%d accum = %d batches)\n",
               (MAX_COMPILES - net.compile_count) / 2,
               ACCUM_STEPS,
               ((MAX_COMPILES - net.compile_count) / 2) * ACCUM_STEPS);
        printf("\n");

        // ── Training loop with gradient accumulation ────────────────────
        int batches_per_epoch = train.n / BATCH;
        int *indices = (int *)malloc((size_t)train.n * sizeof(int));
        for (int i = 0; i < train.n; i++) indices[i] = i;

        float *batch_img = (float *)malloc((size_t)BATCH * IN_DIM * sizeof(float));
        int   *batch_lbl = (int *)malloc((size_t)BATCH * sizeof(int));

        int global_step = 0;
        int accum_count = 0;
        float accum_loss = 0;
        int accum_correct = 0;
        double train_start = now_ms();

        for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
            shuffle_indices(indices, train.n);
            float epoch_loss = 0;
            int   epoch_correct = 0;
            int   epoch_steps = 0;

            for (int step = 0; step < batches_per_epoch; step++) {
                // Check compile budget
                if (net.compile_count >= MAX_COMPILES) {
                    printf("\n  ANE compile budget exhausted (%d/%d). Stopping.\n",
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

                float loss = net_forward(&net, batch_img, batch_lbl);
                net_accumulate_gradients(&net, batch_lbl);

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
                    printf("  Epoch %d/%d  step=%4d  loss=%.4f  acc=%.1f%%  "
                           "[%d compiles]\n",
                           epoch + 1, N_EPOCHS, epoch_steps,
                           avg_loss, avg_acc, net.compile_count);

                    accum_loss = 0;
                    accum_correct = 0;
                    accum_count = 0;

                    // Recompile kernels with updated weights
                    if (net_compile(&net) != 0) {
                        fprintf(stderr, "Recompilation failed at step %d\n", global_step);
                        goto done;
                    }
                }
            }

            // Flush any remaining accumulated gradients
            if (accum_count > 0) {
                net_sgd_step(&net, accum_count);
                accum_loss = 0;
                accum_correct = 0;
                accum_count = 0;

                if (net.compile_count < MAX_COMPILES) {
                    if (net_compile(&net) != 0) {
                        fprintf(stderr, "Recompilation failed at epoch end\n");
                        goto done;
                    }
                }
            }

            float test_acc = net_eval_accuracy(&net, test.images, test.labels, test.n);
            printf("== Epoch %d/%d  avg_loss=%.4f  train_acc=%.1f%%  test_acc=%.1f%%  "
                   "[%d/%d compiles]\n",
                   epoch + 1, N_EPOCHS,
                   epoch_loss / epoch_steps,
                   (float)epoch_correct / (epoch_steps * BATCH) * 100.0f,
                   test_acc, net.compile_count, MAX_COMPILES);
        }

    done:;
        double elapsed = now_ms() - train_start;

        // ── Final evaluation ────────────────────────────────────────────
        float final_acc = net_eval_accuracy(&net, test.images, test.labels, test.n);
        printf("\n========================================\n");
        printf("  Final test accuracy:  %.1f%%\n", final_acc);
        printf("  Total training steps: %d\n", global_step);
        printf("  ANE compiles used:    %d/%d\n", net.compile_count, MAX_COMPILES);
        printf("  Wall time:            %.1fs\n", elapsed / 1000.0);
        printf("========================================\n");

        // ── Cleanup ─────────────────────────────────────────────────────
        net_free(&net);
        free(train.images); free(train.labels);
        free(test.images);  free(test.labels);
        free(indices); free(batch_img); free(batch_lbl);
    }
    return 0;
}

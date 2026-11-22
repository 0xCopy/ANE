// ANEMnist.m — MNIST-on-ANE training bridge (conv-based, baked weights)
//
// Uses 1×1 conv kernels with fp16 weights baked into BLOBFILE constants.
// Gradient accumulation over accum_steps batches amortizes recompile cost.
// Each weight update requires recompiling 2 kernels (784→128, 128→10).

#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ANEMnist.h"

// ANE runtime from training directory
#include "../../../training/ane_runtime.h"
#include "../../../training/ane_mil_gen.h"

// Declared extern in ane_mil_gen.h; one definition per link. 1 = fp16 I/O for M1/M2 (no cast op).
int g_fp16_io = 0;

// ─── Internal state ─────────────────────────────────────────────────────────
struct ANEMnistTrainer {
    ANEMnistConfig cfg;

    // MNIST data
    float *train_images;   // [n_train, input_dim] row-major
    int   *train_labels;
    int    n_train;
    float *test_images;
    int   *test_labels;
    int    n_test;

    // Weights [out, in] row-major
    float *W1, *W2;
    // Accumulated gradients
    float *dW1, *dW2;
    // Momentum
    float *vW1, *vW2;

    // ANE conv kernels (baked weights, single input)
    ANEKernel *kern1, *kern2;
    int compile_count;

    // Activations (channel-first: [channels, batch])
    float *x_cf, *h_pre, *h_post, *logits;

    // Training state
    int  global_step;
    int  step_in_epoch;
    int  current_epoch;
    int  accum_count;     // steps accumulated since last weight update
    float accum_loss;
    int   accum_correct;
    float last_loss;
    float last_accuracy;
    int  *shuffle_idx;
    bool  seeded;
};

// ─── Helpers ────────────────────────────────────────────────────────────────
static float _randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void _transpose(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

static void _shuffle(int *idx, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

static uint32_t _read_be32(FILE *f) {
    uint8_t b[4];
    fread(b, 1, 4, f);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |  (uint32_t)b[3];
}

// Compile a baked-weight conv kernel on ANE
static ANEKernel *_compile_conv(const float *weights, int in_ch, int out_ch, int spatial) {
    @autoreleasepool {
        NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
        NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
        size_t inBytes  = (size_t)in_ch  * spatial * 4;
        size_t outBytes = (size_t)out_ch * spatial * 4;
        return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding],
                           wb, 1, &inBytes, 1, &outBytes);
    }
}

// Compile both kernels with current weights
static int _compile_kernels(ANEMnistTrainer *t) {
    if (t->kern1) { ane_free(t->kern1); t->kern1 = NULL; }
    if (t->kern2) { ane_free(t->kern2); t->kern2 = NULL; }

    int bs = t->cfg.batch_size;
    t->kern1 = _compile_conv(t->W1, t->cfg.input_dim, t->cfg.hidden_dim, bs);
    if (!t->kern1) return -1;
    t->compile_count++;

    t->kern2 = _compile_conv(t->W2, t->cfg.hidden_dim, t->cfg.output_dim, bs);
    if (!t->kern2) return -2;
    t->compile_count++;

    return 0;
}

// ─── API implementation ─────────────────────────────────────────────────────

ANEMnistConfig ane_mnist_default_config(void) {
    return (ANEMnistConfig){
        .input_dim     = 784,
        .hidden_dim    = 128,
        .output_dim    = 10,
        .batch_size    = 128,
        .learning_rate = 0.01f,
        .momentum      = 0.9f,
        .accum_steps   = 50,
        .max_compiles  = 118,
    };
}

ANEMnistTrainer *ane_mnist_create(ANEMnistConfig config) {
    ANEMnistTrainer *t = (ANEMnistTrainer *)calloc(1, sizeof(ANEMnistTrainer));
    if (!t) return NULL;
    t->cfg = config;

    int in = config.input_dim;
    int hid = config.hidden_dim;
    int out = config.output_dim;

    t->W1  = (float *)calloc((size_t)hid * in,  sizeof(float));
    t->W2  = (float *)calloc((size_t)out * hid, sizeof(float));
    t->dW1 = (float *)calloc((size_t)hid * in,  sizeof(float));
    t->dW2 = (float *)calloc((size_t)out * hid, sizeof(float));
    t->vW1 = (float *)calloc((size_t)hid * in,  sizeof(float));
    t->vW2 = (float *)calloc((size_t)out * hid, sizeof(float));

    // He init
    if (!t->seeded) { srand((unsigned)time(NULL)); t->seeded = true; }
    float s1 = sqrtf(2.0f / in);
    float s2 = sqrtf(2.0f / hid);
    for (int i = 0; i < hid * in; i++)  t->W1[i] = _randn() * s1;
    for (int i = 0; i < out * hid; i++) t->W2[i] = _randn() * s2;

    int bs = config.batch_size;
    t->x_cf   = (float *)calloc((size_t)in  * bs, sizeof(float));
    t->h_pre  = (float *)calloc((size_t)hid * bs, sizeof(float));
    t->h_post = (float *)calloc((size_t)hid * bs, sizeof(float));
    t->logits = (float *)calloc((size_t)out * bs, sizeof(float));

    return t;
}

int ane_mnist_load_data(ANEMnistTrainer *t,
                        const char *train_img, const char *train_lbl,
                        const char *test_img,  const char *test_lbl) {
    // Load training images
    FILE *fi = fopen(train_img, "rb");
    FILE *fl = fopen(train_lbl, "rb");
    if (!fi || !fl) {
        if (fi) fclose(fi); if (fl) fclose(fl);
        return -1;
    }
    _read_be32(fi); t->n_train = (int)_read_be32(fi);
    _read_be32(fi); _read_be32(fi);
    _read_be32(fl); _read_be32(fl);

    int idim = t->cfg.input_dim;
    t->train_images = (float *)malloc((size_t)t->n_train * idim * sizeof(float));
    t->train_labels = (int *)malloc((size_t)t->n_train * sizeof(int));

    uint8_t *raw = (uint8_t *)malloc((size_t)t->n_train * idim);
    fread(raw, 1, (size_t)t->n_train * idim, fi);
    for (int i = 0; i < t->n_train * idim; i++)
        t->train_images[i] = raw[i] / 255.0f;
    free(raw);

    uint8_t *rlbl = (uint8_t *)malloc((size_t)t->n_train);
    fread(rlbl, 1, (size_t)t->n_train, fl);
    for (int i = 0; i < t->n_train; i++) t->train_labels[i] = rlbl[i];
    free(rlbl);
    fclose(fi); fclose(fl);

    // Load test images
    fi = fopen(test_img, "rb");
    fl = fopen(test_lbl, "rb");
    if (!fi || !fl) {
        if (fi) fclose(fi); if (fl) fclose(fl);
        return -2;
    }
    _read_be32(fi); t->n_test = (int)_read_be32(fi);
    _read_be32(fi); _read_be32(fi);
    _read_be32(fl); _read_be32(fl);

    t->test_images = (float *)malloc((size_t)t->n_test * idim * sizeof(float));
    t->test_labels = (int *)malloc((size_t)t->n_test * sizeof(int));

    raw = (uint8_t *)malloc((size_t)t->n_test * idim);
    fread(raw, 1, (size_t)t->n_test * idim, fi);
    for (int i = 0; i < t->n_test * idim; i++)
        t->test_images[i] = raw[i] / 255.0f;
    free(raw);

    rlbl = (uint8_t *)malloc((size_t)t->n_test);
    fread(rlbl, 1, (size_t)t->n_test, fl);
    for (int i = 0; i < t->n_test; i++) t->test_labels[i] = rlbl[i];
    free(rlbl);
    fclose(fi); fclose(fl);

    // Shuffle indices
    t->shuffle_idx = (int *)malloc((size_t)t->n_train * sizeof(int));
    for (int i = 0; i < t->n_train; i++) t->shuffle_idx[i] = i;

    return 0;
}

int ane_mnist_compile(ANEMnistTrainer *t) {
    return _compile_kernels(t);
}

ANEMnistResult ane_mnist_train_step(ANEMnistTrainer *t) {
    @autoreleasepool {
        int in  = t->cfg.input_dim;
        int hid = t->cfg.hidden_dim;
        int out = t->cfg.output_dim;
        int bs  = t->cfg.batch_size;
        int bpe = t->n_train / bs;
        bool epoch_done = false;
        bool budget_exhausted = (t->compile_count >= t->cfg.max_compiles);

        if (budget_exhausted) {
            return (ANEMnistResult){
                .loss = t->last_loss,
                .accuracy = t->last_accuracy,
                .step = t->global_step,
                .epoch = t->current_epoch,
                .epoch_done = false,
                .compile_count = t->compile_count,
                .budget_exhausted = true,
            };
        }

        // Reshuffle at epoch boundary
        if (t->step_in_epoch == 0) {
            _shuffle(t->shuffle_idx, t->n_train);
        }

        // ── Gather batch ────────────────────────────────────────────────
        float *bimg = (float *)malloc((size_t)bs * in * sizeof(float));
        int   *blbl = (int *)malloc((size_t)bs * sizeof(int));
        for (int b = 0; b < bs; b++) {
            int idx = t->shuffle_idx[t->step_in_epoch * bs + b];
            memcpy(bimg + (size_t)b * in,
                   t->train_images + (size_t)idx * in,
                   (size_t)in * sizeof(float));
            blbl[b] = t->train_labels[idx];
        }

        // ── Forward on ANE (conv kernels, single input) ─────────────────
        _transpose(t->x_cf, bimg, bs, in);

        ane_write_input(t->kern1, 0, t->x_cf, (size_t)in * bs * 4);
        ane_eval(t->kern1);
        ane_read_output(t->kern1, 0, t->h_pre, (size_t)hid * bs * 4);

        for (int i = 0; i < hid * bs; i++)
            t->h_post[i] = t->h_pre[i] > 0 ? t->h_pre[i] : 0;

        ane_write_input(t->kern2, 0, t->h_post, (size_t)hid * bs * 4);
        ane_eval(t->kern2);
        ane_read_output(t->kern2, 0, t->logits, (size_t)out * bs * 4);

        // ── Loss + accuracy ─────────────────────────────────────────────
        float loss = 0;
        int correct = 0;
        for (int b = 0; b < bs; b++) {
            float mx = -1e9f;
            for (int c = 0; c < out; c++) {
                float v = t->logits[c * bs + b];
                if (v > mx) mx = v;
            }
            float se = 0;
            for (int c = 0; c < out; c++)
                se += expf(t->logits[c * bs + b] - mx);
            loss -= (t->logits[blbl[b] * bs + b] - mx - logf(se));

            int best = 0; float bv = t->logits[b];
            for (int c = 1; c < out; c++) {
                float v = t->logits[c * bs + b];
                if (v > bv) { bv = v; best = c; }
            }
            if (best == blbl[b]) correct++;
        }
        loss /= bs;

        // ── Backward (CPU) — accumulate gradients ───────────────────────
        float *dL = (float *)calloc((size_t)out * bs, sizeof(float));
        for (int b = 0; b < bs; b++) {
            float mx = -1e9f;
            for (int c = 0; c < out; c++) {
                float v = t->logits[c * bs + b];
                if (v > mx) mx = v;
            }
            float se = 0;
            for (int c = 0; c < out; c++) {
                dL[c * bs + b] = expf(t->logits[c * bs + b] - mx);
                se += dL[c * bs + b];
            }
            for (int c = 0; c < out; c++) dL[c * bs + b] /= se;
            dL[blbl[b] * bs + b] -= 1.0f;
            for (int c = 0; c < out; c++) dL[c * bs + b] /= bs;
        }

        // Accumulate dW2
        for (int o = 0; o < out; o++)
            for (int h = 0; h < hid; h++)
                for (int b = 0; b < bs; b++)
                    t->dW2[o * hid + h] += dL[o * bs + b] * t->h_post[h * bs + b];

        // Accumulate dW1
        float *d_hp = (float *)calloc((size_t)hid * bs, sizeof(float));
        for (int h = 0; h < hid; h++)
            for (int b = 0; b < bs; b++)
                for (int o = 0; o < out; o++)
                    d_hp[h * bs + b] += t->W2[o * hid + h] * dL[o * bs + b];

        for (int h = 0; h < hid; h++)
            for (int i = 0; i < in; i++)
                for (int b = 0; b < bs; b++) {
                    float dh = t->h_pre[h * bs + b] > 0 ? d_hp[h * bs + b] : 0;
                    t->dW1[h * in + i] += dh * t->x_cf[i * bs + b];
                }

        free(dL); free(d_hp); free(bimg); free(blbl);

        // ── Accumulate tracking ─────────────────────────────────────────
        t->accum_count++;
        t->accum_loss += loss;
        t->accum_correct += correct;

        // ── Weight update + recompile after accum_steps ─────────────────
        if (t->accum_count >= t->cfg.accum_steps) {
            float scale = 1.0f / t->accum_count;
            float lr = t->cfg.learning_rate, mom = t->cfg.momentum;

            for (int i = 0; i < hid * in; i++) {
                t->vW1[i] = mom * t->vW1[i] - lr * t->dW1[i] * scale;
                t->W1[i] += t->vW1[i];
            }
            for (int i = 0; i < out * hid; i++) {
                t->vW2[i] = mom * t->vW2[i] - lr * t->dW2[i] * scale;
                t->W2[i] += t->vW2[i];
            }

            // Zero grad buffers
            memset(t->dW1, 0, (size_t)hid * in * sizeof(float));
            memset(t->dW2, 0, (size_t)out * hid * sizeof(float));

            // Report averaged metrics from this accumulation window
            t->last_loss = t->accum_loss / t->accum_count;
            t->last_accuracy = (float)t->accum_correct / (t->accum_count * bs) * 100.0f;

            t->accum_loss = 0;
            t->accum_correct = 0;
            t->accum_count = 0;

            // Recompile kernels with updated weights
            if (t->compile_count < t->cfg.max_compiles) {
                _compile_kernels(t);
            }
        } else {
            t->last_loss = loss;
            t->last_accuracy = (float)correct / bs * 100.0f;
        }

        // ── Advance counters ────────────────────────────────────────────
        t->global_step++;
        t->step_in_epoch++;
        if (t->step_in_epoch >= bpe) {
            // Flush remaining accumulated gradients at epoch end
            if (t->accum_count > 0) {
                float scale = 1.0f / t->accum_count;
                float lr = t->cfg.learning_rate, mom = t->cfg.momentum;
                for (int i = 0; i < hid * in; i++) {
                    t->vW1[i] = mom * t->vW1[i] - lr * t->dW1[i] * scale;
                    t->W1[i] += t->vW1[i];
                }
                for (int i = 0; i < out * hid; i++) {
                    t->vW2[i] = mom * t->vW2[i] - lr * t->dW2[i] * scale;
                    t->W2[i] += t->vW2[i];
                }
                memset(t->dW1, 0, (size_t)hid * in * sizeof(float));
                memset(t->dW2, 0, (size_t)out * hid * sizeof(float));
                t->accum_count = 0;
                t->accum_loss = 0;
                t->accum_correct = 0;

                if (t->compile_count < t->cfg.max_compiles) {
                    _compile_kernels(t);
                }
            }

            t->step_in_epoch = 0;
            t->current_epoch++;
            epoch_done = true;
        }

        return (ANEMnistResult){
            .loss       = t->last_loss,
            .accuracy   = t->last_accuracy,
            .step       = t->global_step,
            .epoch      = t->current_epoch,
            .epoch_done = epoch_done,
            .compile_count = t->compile_count,
            .budget_exhausted = (t->compile_count >= t->cfg.max_compiles),
        };
    }
}

float ane_mnist_evaluate(ANEMnistTrainer *t) {
    @autoreleasepool {
        int in  = t->cfg.input_dim;
        int hid = t->cfg.hidden_dim;
        int out = t->cfg.output_dim;
        int bs  = t->cfg.batch_size;

        float *xcf = (float *)malloc((size_t)in  * bs * sizeof(float));
        float *hp  = (float *)malloc((size_t)hid * bs * sizeof(float));
        float *hpp = (float *)malloc((size_t)hid * bs * sizeof(float));
        float *lg  = (float *)malloc((size_t)out * bs * sizeof(float));

        int correct = 0, total = 0;
        for (int start = 0; start + bs <= t->n_test; start += bs) {
            _transpose(xcf, t->test_images + (size_t)start * in, bs, in);

            ane_write_input(t->kern1, 0, xcf, (size_t)in * bs * 4);
            ane_eval(t->kern1);
            ane_read_output(t->kern1, 0, hp, (size_t)hid * bs * 4);

            for (int i = 0; i < hid * bs; i++) hpp[i] = hp[i] > 0 ? hp[i] : 0;

            ane_write_input(t->kern2, 0, hpp, (size_t)hid * bs * 4);
            ane_eval(t->kern2);
            ane_read_output(t->kern2, 0, lg, (size_t)out * bs * 4);

            for (int b = 0; b < bs; b++) {
                int best = 0; float bv = lg[b];
                for (int c = 1; c < out; c++) {
                    float v = lg[c * bs + b];
                    if (v > bv) { bv = v; best = c; }
                }
                if (best == t->test_labels[start + b]) correct++;
            }
            total += bs;
        }

        free(xcf); free(hp); free(hpp); free(lg);
        return total > 0 ? (float)correct / total * 100.0f : 0;
    }
}

float ane_mnist_current_loss(const ANEMnistTrainer *t) { return t->last_loss; }
int   ane_mnist_current_step(const ANEMnistTrainer *t) { return t->global_step; }
int   ane_mnist_train_count(const ANEMnistTrainer *t)  { return t->n_train; }
int   ane_mnist_test_count(const ANEMnistTrainer *t)   { return t->n_test; }
int   ane_mnist_batches_per_epoch(const ANEMnistTrainer *t) {
    return t->n_train / t->cfg.batch_size;
}
int   ane_mnist_compile_count(const ANEMnistTrainer *t) { return t->compile_count; }

void ane_mnist_free(ANEMnistTrainer *t) {
    if (!t) return;
    if (t->kern1) ane_free(t->kern1);
    if (t->kern2) ane_free(t->kern2);
    free(t->W1);  free(t->W2);
    free(t->dW1); free(t->dW2);
    free(t->vW1); free(t->vW2);
    free(t->x_cf); free(t->h_pre); free(t->h_post); free(t->logits);
    free(t->train_images); free(t->train_labels);
    free(t->test_images);  free(t->test_labels);
    free(t->shuffle_idx);
    free(t);
}

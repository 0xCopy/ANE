// test_bridge.c — regression test for libanebridge on real ANE hardware.
//   make test            (from contrib/kmp-ane)
//
// Each check exists because the behaviour was once broken:
//   1. sustain   — compile/eval/free past the ~128 mark (freed kernels used to leak ANE resources)
//   2. conv      — baked-weight conv with a batch that is not a multiple of 16 (64-byte row stride)
//   3. mm        — dynamic matmul: one compile, operands rewritten every eval, checked against a
//                  CPU reference, including transposes, odd shapes and gradient-sized operands
//                  (fp16 flushes anything below ~6e-5 to zero unless operands are normalised)
//   4. matmul    — the channel-first W @ x helpers (input order is by name, not declaration)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#include "ANEBridge.h"

static int g_failures = 0;
#define MAX_REL_ERR 0.005   /* fp16 kernel vs double reference, relative to max |reference| */

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static void fail(const char *what, ANEBridgeStatus s) {
    printf("  FAIL %s (status %d) %.200s\n", what, (int)s, ane_bridge_last_error());
    g_failures++;
}

static void fill(float *v, int n, int seed, float scale) {
    for (int i = 0; i < n; i++) v[i] = (float)(((i * 37 + seed * 11) % 201) - 100) / 100.0f * scale;
}

static double rel_err(const float *got, const float *ref, int n) {
    double e = 0, mx = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)got[i] - ref[i]);
        if (d > e) e = d;
        if (fabs(ref[i]) > mx) mx = fabs(ref[i]);
    }
    return mx > 0 ? e / mx : e;
}

// 1. More compile/eval/free cycles than the ANE allows live kernels per process.
static void test_sustain(int cycles) {
    int in = 64, out = 32, S = 16;
    float *W = malloc(sizeof(float) * in * out), *x = malloc(sizeof(float) * in * S), *y = malloc(sizeof(float) * out * S);
    fill(x, in * S, 1, 1.0f);
    int done = 0;
    for (; done < cycles; done++) {
        fill(W, in * out, done, 0.1f);   // new weights: a real recompile every cycle
        ANEBridgeKernel *k = NULL;
        ANEBridgeStatus s = ane_bridge_compile_conv(W, in, out, S, &k);
        if (s != ANE_OK || !k) { fail("sustain: compile", s); break; }
        s = ane_bridge_conv_eval(k, x, y, S, in, out);
        ane_bridge_free_kernel(k);
        if (s != ANE_OK) { fail("sustain: eval", s); break; }
    }
    printf("sustain: %d/%d compile/eval/free cycles %s\n", done, cycles, done == cycles ? "ok" : "FAILED");
    free(W); free(x); free(y);
}

// 2. Baked-weight conv, any batch size.
static void test_conv(int S) {
    int in = 64, out = 32;
    float *W = malloc(sizeof(float) * in * out), *x = malloc(sizeof(float) * in * S);
    float *y = malloc(sizeof(float) * out * S), *r = malloc(sizeof(float) * out * S);
    fill(W, in * out, 3, 0.1f); fill(x, in * S, 5, 1.0f);
    ANEBridgeKernel *k = NULL;
    ANEBridgeStatus s = ane_bridge_compile_conv(W, in, out, S, &k);
    if (s != ANE_OK) { fail("conv: compile", s); goto done; }
    s = ane_bridge_conv_eval(k, x, y, S, in, out);
    if (s != ANE_OK) { fail("conv: eval", s); goto done; }
    for (int t = 0; t < S; t++)
        for (int o = 0; o < out; o++) {
            double a = 0;
            for (int i = 0; i < in; i++) a += (double)x[t * in + i] * W[o * in + i];
            r[t * out + o] = (float)a;
        }
    double e = rel_err(y, r, out * S);
    printf("conv: batch %3d  err %.4f%% %s\n", S, 100 * e, e < MAX_REL_ERR ? "ok" : "FAILED");
    if (e >= MAX_REL_ERR) g_failures++;
done:
    ane_bridge_free_kernel(k);
    free(W); free(x); free(y); free(r);
}

// 3. Dynamic matmul Y = op(A) @ op(B).
static void test_mm(int ar, int ac, int br, int bc, int ta, int tb, float a_scale, const char *what) {
    int m = ta ? ac : ar, kk = ta ? ar : ac, n = tb ? br : bc, evals = 50;
    float *A = malloc(sizeof(float) * ar * ac), *B = malloc(sizeof(float) * br * bc);
    float *y = malloc(sizeof(float) * m * n), *r = malloc(sizeof(float) * m * n);
    ANEBridgeKernel *k = NULL;
    double t0 = now_ms(), worst = 0, eval_ms = 0;
    ANEBridgeStatus s = ane_bridge_compile_mm(ar, ac, br, bc, ta, tb, &k);
    double compile_ms = now_ms() - t0;
    if (s != ANE_OK) { fail(what, s); goto done; }
    for (int it = 0; it < evals; it++) {
        fill(A, ar * ac, it, a_scale); fill(B, br * bc, it + 7, 0.05f);   // both operands change every eval
        t0 = now_ms();
        s = ane_bridge_mm_eval(k, A, B, y);
        eval_ms += now_ms() - t0;
        if (s != ANE_OK) { fail(what, s); goto done; }
        if (it % 10) continue;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double acc = 0;
                for (int p = 0; p < kk; p++)
                    acc += (double)(ta ? A[p * ac + i] : A[i * ac + p]) * (tb ? B[j * bc + p] : B[p * bc + j]);
                r[i * n + j] = (float)acc;
            }
        double e = rel_err(y, r, m * n);
        if (e > worst) worst = e;
    }
    printf("mm: %-34s [%d,%d]%s @ [%d,%d]%s  max|A|=%-6g 1 compile (%.0f ms), %.2f ms/eval, err %.4f%% %s\n",
           what, ar, ac, ta ? "^T" : "", br, bc, tb ? "^T" : "", a_scale, compile_ms, eval_ms / evals,
           100 * worst, worst < MAX_REL_ERR ? "ok" : "FAILED");
    if (worst >= MAX_REL_ERR) g_failures++;
done:
    ane_bridge_free_kernel(k);
    free(A); free(B); free(y); free(r);
}

// 4. Channel-first helpers: y = W @ x, and the row-major convenience wrapper.
static void test_matmul_helpers(int in, int out, int S) {
    float *W = malloc(sizeof(float) * out * in), *x = malloc(sizeof(float) * S * in);
    float *y = malloc(sizeof(float) * S * out), *r = malloc(sizeof(float) * S * out);
    fill(W, out * in, 2, 0.05f); fill(x, S * in, 9, 1.0f);
    ANEBridgeKernel *k = NULL;
    ANEBridgeStatus s = ane_bridge_compile_matmul(in, out, S, &k);
    if (s != ANE_OK) { fail("matmul helpers: compile", s); goto done; }
    s = ane_bridge_matmul_eval_rowmajor(k, W, x, y, in, out, S);
    if (s != ANE_OK) { fail("matmul helpers: eval", s); goto done; }
    for (int t = 0; t < S; t++)
        for (int o = 0; o < out; o++) {
            double a = 0;
            for (int i = 0; i < in; i++) a += (double)x[t * in + i] * W[o * in + i];
            r[t * out + o] = (float)a;
        }
    double e = rel_err(y, r, S * out);
    printf("matmul helpers: in %d out %d batch %d  err %.4f%% %s\n", in, out, S, 100 * e, e < MAX_REL_ERR ? "ok" : "FAILED");
    if (e >= MAX_REL_ERR) g_failures++;
done:
    ane_bridge_free_kernel(k);
    free(W); free(x); free(y); free(r);
}

int main(void) {
    ANEBridgeStatus s = ane_bridge_init();
    if (s != ANE_OK || !ane_bridge_is_available()) {
        printf("ANE not available (status %d); nothing tested.\n", (int)s);
        return 2;
    }

    test_sustain(300);

    test_conv(16);
    test_conv(10);
    test_conv(100);

    // The six products of a 784 → 128 → 10 MLP at batch 128 (see ANEBridge.h), with
    // gradient-sized dy for the backward ones.
    test_mm(128, 784, 128, 784, 0, 1, 1.0f,  "layer 1 forward  x @ W1^T");
    test_mm(128, 128,  10, 128, 0, 1, 1.0f,  "layer 2 forward  h @ W2^T");
    test_mm(128,  10,  10, 128, 0, 0, 1e-3f, "layer 2 backward dy @ W2");
    test_mm(128,  10, 128, 128, 1, 0, 1e-3f, "layer 2 backward dy^T @ h");
    test_mm(128, 128, 128, 784, 0, 0, 1e-4f, "layer 1 backward dy @ W1");
    test_mm(128, 128, 128, 784, 1, 0, 1e-4f, "layer 1 backward dy^T @ x");
    // Shapes with rows that are not multiples of 16 floats, operands far below fp16 range.
    test_mm(  4,   3,   3,   5, 0, 0, 1.0f,  "tiny");
    test_mm(  7,  13,   5,  13, 0, 1, 1e-6f, "odd, max|A| = 1e-6");
    test_mm( 33,  17,  33,   9, 1, 0, 1e-7f, "odd, max|A| = 1e-7");

    test_matmul_helpers(784, 128, 128);
    test_matmul_helpers(50, 7, 13);

    printf(g_failures ? "\n%d FAILED\n" : "\nall bridge tests passed\n", g_failures);
    return g_failures ? 1 : 0;
}

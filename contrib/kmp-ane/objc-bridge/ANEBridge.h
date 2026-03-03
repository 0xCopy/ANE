// ANEBridge.h — Clean C-callable ObjC bridge for the ANE research runtime
// Wraps ane_runtime.h / ane_mil_gen.h into a stable ABI for Swift / Kotlin/Native cinterop
//
// NOTE: This header is intentionally pure C (no Foundation/ObjC imports) so that
// Kotlin/Native cinterop can parse it without SDK version issues.
// The .m implementation imports Foundation separately.
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------
typedef struct ANEBridgeKernel ANEBridgeKernel;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
typedef struct {
    int dim;
    int hidden_dim;
    int n_heads;
    int n_layers;
    int vocab_size;
    int seq_len;
} ANEBridgeConfig;

// ---------------------------------------------------------------------------
// Status / error codes
// ---------------------------------------------------------------------------
typedef enum {
    ANE_OK = 0,
    ANE_ERR_INIT_FAILED    = -1,
    ANE_ERR_COMPILE_FAILED = -2,
    ANE_ERR_LOAD_FAILED    = -3,
    ANE_ERR_EVAL_FAILED    = -4,
    ANE_ERR_NULL_KERNEL    = -5,
    ANE_ERR_INDEX_OOB      = -6,
    ANE_ERR_ALLOC_FAILED   = -7,
} ANEBridgeStatus;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Initialise the ANE private framework (idempotent, thread-safe).
ANEBridgeStatus ane_bridge_init(void);

/// Returns true after a successful ane_bridge_init().
bool ane_bridge_is_available(void);

// ---------------------------------------------------------------------------
// Kernel — compile / eval / free
// ---------------------------------------------------------------------------

/// Compile a MIL program (UTF-8 text) + optional weight blob into an ANE kernel.
/// @param mil_text      MIL program source (UTF-8, null-terminated)
/// @param mil_text_len  byte length excluding null terminator
/// @param weight_data   raw weight blob (may be NULL)
/// @param weight_len    byte length of weight blob
/// @param n_inputs      number of input tensors
/// @param input_bytes   array[n_inputs] of byte sizes per input IOSurface
/// @param n_outputs     number of output tensors
/// @param output_bytes  array[n_outputs] of byte sizes per output IOSurface
/// @param out_kernel    receives the kernel handle on success
ANEBridgeStatus ane_bridge_compile(const char *mil_text,
                                    size_t mil_text_len,
                                    const void *weight_data,
                                    size_t weight_len,
                                    int n_inputs,
                                    const size_t *input_bytes,
                                    int n_outputs,
                                    const size_t *output_bytes,
                                    ANEBridgeKernel **out_kernel);

/// Write data into an input IOSurface.
ANEBridgeStatus ane_bridge_write_input(ANEBridgeKernel *kernel,
                                        int index,
                                        const void *data,
                                        size_t bytes);

/// Read data from an output IOSurface.
ANEBridgeStatus ane_bridge_read_output(ANEBridgeKernel *kernel,
                                        int index,
                                        void *data,
                                        size_t bytes);

/// Execute the kernel on the ANE.
ANEBridgeStatus ane_bridge_eval(ANEBridgeKernel *kernel);

/// Unload + free all resources associated with a kernel.
void ane_bridge_free_kernel(ANEBridgeKernel *kernel);

// ---------------------------------------------------------------------------
// MIL generation helpers  (return heap-allocated C strings, caller must free)
// ---------------------------------------------------------------------------

/// Generate MIL for a 1×1 conv linear: x[1,in_ch,1,S] → y[1,out_ch,1,S]
char *ane_bridge_mil_gen_conv(int in_ch, int out_ch, int spatial);

/// Generate MIL for a matmul: W[1,out,in] @ x[1,in,S] → y[1,out,S]
char *ane_bridge_mil_gen_matmul(int in_ch, int out_ch, int spatial);

/// Generate MIL for fused QKV (3 parallel convs from same input)
char *ane_bridge_mil_gen_qkv(int dim, int spatial);

/// Generate MIL for fused FFN up (W1 + W3 parallel convs)
char *ane_bridge_mil_gen_ffn_up(int dim, int hidden_dim, int spatial);

// ---------------------------------------------------------------------------
// Weight blob builders (return malloc'd buffer + set *out_len)
// ---------------------------------------------------------------------------

/// Build FP16 weight blob from f32 weights.
void *ane_bridge_build_weight_blob(const float *weights_f32,
                                    int out_ch, int in_ch,
                                    size_t *out_len);

/// Build fused QKV weight blob (3 matrices).
void *ane_bridge_build_qkv_weight_blob(const float *wq,
                                        const float *wk,
                                        const float *wv,
                                        int dim,
                                        size_t *out_len);

/// Build fused FFN up weight blob (W1 + W3).
void *ane_bridge_build_ffn_up_weight_blob(const float *w1,
                                           const float *w3,
                                           int hidden_dim,
                                           int dim,
                                           size_t *out_len);

// ---------------------------------------------------------------------------
// High-level model helpers
// ---------------------------------------------------------------------------

/// Convenience: compile a baked-weight conv kernel from f32 weights.
/// Generates MIL + weight blob internally.
ANEBridgeStatus ane_bridge_compile_conv(const float *weights,
                                         int in_ch, int out_ch, int spatial,
                                         ANEBridgeKernel **out_kernel);

/// Run a conv eval: x[S, in_dim] → y[S, out_dim]
/// Handles the row-major ↔ channel-first transpose internally.
ANEBridgeStatus ane_bridge_conv_eval(ANEBridgeKernel *kernel,
                                      const float *x, float *y,
                                      int S, int in_dim, int out_dim);

// ---------------------------------------------------------------------------
// Matmul helpers (2-input kernels — weights passed at eval time, no recompile)
// ---------------------------------------------------------------------------

/// Compile a matmul kernel: y = W @ x
/// W[1, out_ch, in_ch], x[1, in_ch, spatial] → y[1, out_ch, spatial]
/// The kernel takes 2 inputs (x at index 0, W at index 1).
ANEBridgeStatus ane_bridge_compile_matmul(int in_ch, int out_ch, int spatial,
                                           ANEBridgeKernel **out_kernel);

/// Evaluate a matmul kernel. Writes W and x, evals, reads y.
/// All data is in channel-first layout: [1, channels, spatial] as flat f32.
ANEBridgeStatus ane_bridge_matmul_eval(ANEBridgeKernel *kernel,
                                        const float *W, const float *x, float *y,
                                        int in_ch, int out_ch, int spatial);

/// Convenience: row-major matmul. Transposes to/from channel-first internally.
/// W[out_ch, in_ch], x[spatial, in_ch] → y[spatial, out_ch]
ANEBridgeStatus ane_bridge_matmul_eval_rowmajor(ANEBridgeKernel *kernel,
                                                 const float *W, const float *x, float *y,
                                                 int in_ch, int out_ch, int spatial);

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// High-resolution monotonic time in nanoseconds.
uint64_t ane_bridge_time_ns(void);

#ifdef __cplusplus
}
#endif

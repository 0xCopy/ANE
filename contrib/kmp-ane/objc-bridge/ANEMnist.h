// ANEMnist.h — C-callable bridge for MNIST training on ANE
// Used by Swift wrapper and Kotlin/Native cinterop
#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Opaque trainer handle ──────────────────────────────────────────────────
typedef struct ANEMnistTrainer ANEMnistTrainer;

// ─── Configuration ──────────────────────────────────────────────────────────
typedef struct {
    int input_dim;      // default 784
    int hidden_dim;     // default 128
    int output_dim;     // default 10
    int batch_size;     // default 128
    float learning_rate; // default 0.01
    float momentum;      // default 0.9
    int accum_steps;     // gradient accumulation steps before weight update (default 50)
    int max_compiles;    // ANE compile budget limit (default 118)
} ANEMnistConfig;

/// Returns a config with sensible defaults.
ANEMnistConfig ane_mnist_default_config(void);

// ─── Training step result ───────────────────────────────────────────────────
typedef struct {
    float loss;
    float accuracy;     // percentage [0, 100]
    int   step;         // global step counter
    int   epoch;        // current epoch (0-based)
    bool  epoch_done;   // true when this step completed an epoch
    int   compile_count; // ANE compiles used so far
    bool  budget_exhausted; // true if compile budget is exhausted
} ANEMnistResult;

// ─── Lifecycle ──────────────────────────────────────────────────────────────

/// Create a trainer with the given config. Returns NULL on failure.
ANEMnistTrainer *ane_mnist_create(ANEMnistConfig config);

/// Load MNIST data files (IDX format). Returns 0 on success.
int ane_mnist_load_data(ANEMnistTrainer *t,
                        const char *train_images_path,
                        const char *train_labels_path,
                        const char *test_images_path,
                        const char *test_labels_path);

/// Compile ANE kernels. Call after create, before training. Returns 0 on success.
int ane_mnist_compile(ANEMnistTrainer *t);

// ─── Training & evaluation ──────────────────────────────────────────────────

/// Run one training step (forward + backward + SGD on one batch).
/// Automatically handles epoch transitions and shuffling.
ANEMnistResult ane_mnist_train_step(ANEMnistTrainer *t);

/// Evaluate accuracy on the test set. Returns percentage [0, 100].
float ane_mnist_evaluate(ANEMnistTrainer *t);

/// Get current loss (from last train_step).
float ane_mnist_current_loss(const ANEMnistTrainer *t);

/// Get current global step count.
int ane_mnist_current_step(const ANEMnistTrainer *t);

/// Get number of training samples loaded.
int ane_mnist_train_count(const ANEMnistTrainer *t);

/// Get number of test samples loaded.
int ane_mnist_test_count(const ANEMnistTrainer *t);

/// Get number of batches per epoch.
int ane_mnist_batches_per_epoch(const ANEMnistTrainer *t);

/// Get number of ANE compiles used so far.
int ane_mnist_compile_count(const ANEMnistTrainer *t);

// ─── Cleanup ────────────────────────────────────────────────────────────────

/// Free all resources. Safe to call with NULL.
void ane_mnist_free(ANEMnistTrainer *t);

#ifdef __cplusplus
}
#endif

# Known Bugs — ANE Runtime & Bridge Code

Tracked here; fixes deferred until after the tmp/ demo is validated.

---

## BUG-1: ARC object leak in `ane_free()` (CRITICAL)

**File:** `training/ane_runtime.h` → `ane_free()`

**Problem:** The `ANEKernel` struct contains `__strong` ObjC fields (`id model`,
`id request`, `NSString *tmpDir`) allocated inside a `calloc()`'d block. Under
ARC, `free()` does NOT release strong references — it just frees raw memory.
The retained ObjC objects (the compiled ANE model, the eval request, the temp
directory string) are **permanently leaked** on every `ane_free()` call.

**Impact:** Every kernel compile → free cycle leaks ~3 ObjC objects plus their
backing ANE resources. In the MNIST trainer, each weight-update recompile
leaks the *previous* model. Over 58 recompiles, that's 174+ leaked objects
plus unreleased ANE device handles.

**Fix:**
```c
static void ane_free(ANEKernel *k) {
    if (!k) return;
    NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
        k->model, @selector(unloadWithQoS:error:), 21, &e);
    for (int i = 0; i < k->nInputs; i++) CFRelease(k->ioInputs[i]);
    for (int i = 0; i < k->nOutputs; i++) CFRelease(k->ioOutputs[i]);
    [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
+   // ARC: must nil strong refs before free() — free() won't release them
+   k->model = nil;
+   k->request = nil;
+   k->tmpDir = nil;
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    free(k);
}
```

---

## BUG-2: Same ARC leak in `ANEBridge.m:ane_bridge_free_kernel()`

**File:** `contrib/kmp-ane/objc-bridge/ANEBridge.m`

**Problem:** Identical pattern — the `ANEBridgeKernel` opaque struct wraps
`ANEKernel` which has the same calloc/free lifecycle without ARC nilling.
`ane_bridge_free_kernel()` calls `ane_free()` which has BUG-1.

**Fix:** Fix BUG-1 and this is automatically resolved (both use `ane_free()`).

---

## BUG-3: NULL pointer dereference in `ane_bridge_matmul_eval_rowmajor()`

**File:** `contrib/kmp-ane/objc-bridge/ANEBridge.m`

**Problem:** The function signature is:
```c
ANEBridgeStatus ane_bridge_matmul_eval_rowmajor(
    ANEBridgeKernel *kernel,
    const float *x_rowmajor, int x_rows, int x_cols,
    const float *w_rowmajor, int w_rows, int w_cols,
    float *y_rowmajor)
```
But the implementation transpositions write into local buffers, calls
`ane_bridge_matmul_eval()` which expects to write output to
`y_rowmajor` via `ane_read_output()` into a temp buffer. The actual
`y_rowmajor` parameter is used as the final destination for an
un-transpose step, but if `ane_bridge_matmul_eval()` internally
allocates and discards its own buffer, the output may be lost or
written to the wrong address.

**Severity:** Depends on call path — potentially a crash or silent
data corruption.

**Fix:** Audit the full data flow in `ane_bridge_matmul_eval_rowmajor()`
to ensure the output buffer is correctly populated through the
transpose → eval → un-transpose chain.

---

## BUG-4: Cascading leak in `ANEMnist.m`

**File:** `contrib/kmp-ane/objc-bridge/ANEMnist.m`

**Problem:** `ane_mnist_free()` calls `ane_free()` for its kernels,
inheriting BUG-1. Additionally, the `ANEMnistState` struct may hold
its own strong ObjC refs that need nilling before its `free()`.

**Fix:** Fix BUG-1, then audit ANEMnistState for any additional
ObjC strong refs that need nilling.

---

## BUG-5: Autorelease pool pressure during training loop

**File:** `training/ane_runtime.h`, `contrib/kmp-ane/mnist/mnist_ane.m`

**Problem:** The functions `ane_compile()`, `mil_build_weight_blob()`,
`mil_gen_conv()` create many temporary ObjC objects (NSData, NSString,
NSArray, NSDictionary). In the training loop, these are only released
when the outermost `@autoreleasepool` drains — which is at process
exit. Each recompilation accumulates ~dozens of intermediate ObjC
allocations.

**Impact:** Memory grows linearly with number of recompiles. Not a
crash, but adds ~MB of waste over a full training run.

**Fix:** Wrap each `compile_conv()` call (and the `net_compile()`
function) in its own `@autoreleasepool {}` block. The standalone
`mnist_ane.m` demo already wraps `compile_conv()` — verify the
inner functions also benefit.

---

## Summary

| # | Severity | File | Root Cause |
|---|----------|------|------------|
| 1 | CRITICAL | ane_runtime.h | ARC strong refs in calloc'd struct, freed without nilling |
| 2 | CRITICAL | ANEBridge.m | Same as #1 (calls ane_free) |
| 3 | HIGH | ANEBridge.m | Potential NULL write in matmul eval |
| 4 | CRITICAL | ANEMnist.m | Cascading from #1 |
| 5 | MODERATE | ane_runtime.h | Autorelease pressure in tight recompile loop |

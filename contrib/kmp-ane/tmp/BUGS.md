# Known Memory Leaks & Bugs

Tracked here, to be fixed post-demo.

## Bug 1 — `ane_free()` leaks ARC-retained ObjC objects (CRITICAL)
**File:** `training/ane_runtime.h` line ~155 (`ane_free`)  
**Root cause:** `ANEKernel` is `calloc`'d C struct with strong ObjC fields (`id model`,
`id request`, `NSString *tmpDir`). Under ARC, assigning to these fields increments
the retain count. But `free()` on a C struct does NOT invoke ARC release — the
runtime only releases fields in ObjC `@interface` objects or stack-scope variables.  
**Impact:** Every `ane_free()` call leaks the `_ANEInMemoryModel`, `_ANERequest`,
and temp dir `NSString`. With 118 kernel compiles, that's 354 leaked objects per run.  
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
    // *** FIX: nil out ARC fields before free ***
    k->model = nil;
    k->request = nil;
    k->tmpDir = nil;
    free(k->ioInputs); free(k->ioOutputs);
    free(k->inputBytes); free(k->outputBytes);
    free(k);
}
```

## Bug 2 — `ANEBridge.m:ane_bridge_free_kernel()` same pattern (CRITICAL)
**File:** `contrib/kmp-ane/objc-bridge/ANEBridge.m`  
**Same root cause:** wraps `ane_free()`, inherits the leak.

## Bug 3 — `ane_bridge_matmul_eval_rowmajor()` passes NULL to memcpy (CRASH)
**File:** `contrib/kmp-ane/objc-bridge/ANEBridge.m`  
**Root cause:** `ane_bridge_matmul_eval()` is called with `y = NULL`, then 
`memcpy(y, ...)` segfaults.  
**Fix:** Check `if (y)` before memcpy, or don't allow NULL callers.

## Bug 4 — `ANEMnist.m:ane_mnist_free()` inherits Bug 1
**File:** `contrib/kmp-ane/objc-bridge/ANEMnist.m`  
**Impact:** Calls `ane_free()` which has the leak.

---

## Status
- [x] Fix Bug 1 in `training/ane_runtime.h` (nil the ARC fields before free; MNIST demo went from dying at 106/118 compiles with ANE Code=54 to completing 118/118)
- [x] Bug 2 in `objc-bridge/ANEBridge.m` (inherits the Bug 1 fix; wrapper unchanged)
- [x] Fix Bug 3 in `objc-bridge/ANEBridge.m` (`ane_bridge_matmul_eval` returns before the read when `y` is NULL)
- [x] Bug 4 inherits from Bug 1 fix

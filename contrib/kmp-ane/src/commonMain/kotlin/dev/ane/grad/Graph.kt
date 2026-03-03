// Graph.kt — Retained for API compatibility but the actual expression graph
// is built via the Autograd.kt tape (KotlinGrad-style).
//
// The Autograd.kt system provides:
//   - Expr (sealed class with Variable, Parameter, Constant)
//   - .matmul(), .relu(), .logSoftmax(), .addBias(), .nllLoss()
//   - .backward() for reverse-mode autodiff
//   - ANELinear.kt: .aneLinear() for ANE-accelerated forward matmul
//
// The Optimizer.kt provides SGD and Adam.
//
// This file is intentionally empty to avoid duplicate Expr definitions.
package dev.ane.grad


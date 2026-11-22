#!/bin/bash
# A module of ane ops -> LLVM IR for the CPU.  usage: ane_to_ll.sh [--wide] file.mlir > out.ll
#   --wide   sums of products accumulate in f32 and round to f16 once (closer to what the device returns)
here=$(cd "$(dirname "$0")" && pwd)
wide=; [ "$1" = "--wide" ] && { wide=--wide; shift; }
mlir-opt --irdl-file="$here/../dialect/ane.irdl.mlir" "$1" | python3 "$here/ane_dialect.py" to-mil \
  | python3 "$here/mil2mlir.py" $wide \
  | mlir-opt --transform-preload-library=transform-library-paths="$here/../test/decompose_softmax.mlir" --transform-interpreter \
      --linalg-generalize-named-ops --one-shot-bufferize=bufferize-function-boundaries --convert-linalg-to-loops \
      --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-math-to-llvm --convert-math-to-libm \
      --convert-vector-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
      --convert-cf-to-llvm --reconcile-unrealized-casts \
  | mlir-translate --mlir-to-llvmir

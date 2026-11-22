#!/bin/bash
# Runs the `// RUN:` lines of mlir/test/check_*.mlir and mlir/aarch64/{run.mlir,asm.test,ane_asm.test}.
#   %s the test file, %S its directory, %t a temp file, %llvmlib the LLVM lib dir.
# usage: mlir/run-tests.sh [-v]     (LLVM_PREFIX defaults to /opt/homebrew/opt/llvm)
LLVM_PREFIX=${LLVM_PREFIX:-/opt/homebrew/opt/llvm}
export PATH="$LLVM_PREFIX/bin:$PATH"
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fail=0; n=0
for f in "$here"/test/check_*.mlir "$here"/aarch64/run.mlir "$here"/aarch64/asm.test "$here"/aarch64/ane_asm.test; do
  cmds=$(sed -n 's|^// RUN: \{0,1\}||p' "$f" | sed -e ':a' -e '/\\$/N; s/\\\n//; ta')
  ok=1
  while IFS= read -r c; do
    [ -z "$c" ] && continue
    c=${c//%llvmlib/$LLVM_PREFIX/lib}; c=${c//%S/$(dirname "$f")}; c=${c//%s/$f}; c=${c//%t/$tmp/$(basename "$f")}
    if ! out=$(bash -o pipefail -c "$c" 2>&1); then ok=0; echo "$out" | head -20; fi
    [ "$1" = "-v" ] && echo "$out"
  done <<< "$cmds"
  n=$((n+1))
  if [ $ok = 1 ]; then echo "PASS ${f#$here/}"; else echo "FAIL ${f#$here/}"; fail=$((fail+1)); fi
done
echo "$((n-fail))/$n passed"
exit $fail

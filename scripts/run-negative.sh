#!/usr/bin/env bash
# Each tests/neg/*.ptx must make krun fail (non-zero exit, or a stuck configuration reporting a fault).
# Optional first-line comment `// expect: text` must occur in the output.
set -u
export K_OPTS="${K_OPTS:--Xss512m}"
cd "$(dirname "$0")/.."
DEF=${PTX_DEF:-build/ptx-kompiled}
pass=0; fail=0
for t in tests/neg/*.ptx; do
  out=$(krun -d "$DEF" "$t" --output pretty 2>&1); code=$?
  want=$(sed -n '1s|^// expect: ||p' "$t")
  if [ $code -ne 0 ] && { [ -z "$want" ] || printf '%s' "$out" | grep -q -- "$want"; }; then echo "PASS  $t"; pass=$((pass+1)); else echo "FAIL  $t (exit $code)"; fail=$((fail+1)); fi
done
echo "---- $pass passed, $fail failed"; [ $fail -eq 0 ]

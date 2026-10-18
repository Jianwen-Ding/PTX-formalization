#!/usr/bin/env bash
# Run the AGX assembly tests (agx/tests/*.agx): a test passes when krun exits 0 (all .expect* directives held).
set -u
cd "$(dirname "$0")/.."
export K_OPTS="${K_OPTS:--Xss512m}"
pass=0; fail=0
for t in agx/tests/*.agx; do
  out=$(krun -d build/agx-kompiled "$t" --output pretty 2>&1); code=$?
  msg=$(printf '%s' "$out" | grep -A1 '<msg>' | tail -1 | sed 's/^ *//')
  st=$(printf '%s' "$out" | grep -A1 '<status>' | tail -1 | tr -d ' "')
  if [ $code -eq 0 ] && [ "$st" = "stopped" ]; then echo "PASS  $t"; pass=$((pass+1)); else echo "FAIL  $t (exit $code, status $st) $msg"; fail=$((fail+1)); fi
done
echo "---- $pass passed, $fail failed"; [ $fail -eq 0 ]

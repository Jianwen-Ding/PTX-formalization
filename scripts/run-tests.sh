#!/usr/bin/env bash
# Run every tests/*.ptx through the K interpreter and compare stdout with the
# golden tests/*.out file; a test passes when the output matches and krun
# exits with 0 (all `.expect` checks held).
#   scripts/run-tests.sh [--update] [test.ptx ...]
set -u
cd "$(dirname "$0")/.."
DEF=${PTX_DEF:-build/ptx-kompiled}
update=0
if [ "${1:-}" = "--update" ]; then update=1; shift; fi
tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=(tests/*.ptx)

pass=0; fail=0
for t in "${tests[@]}"; do
  golden="${t%.ptx}.out"
  actual=$(krun -d "$DEF" "$t" --output none 2>/dev/null)
  code=$?
  if [ $update -eq 1 ]; then printf '%s\n' "$actual" > "$golden"; fi
  if [ $code -eq 0 ] && [ -f "$golden" ] && [ "$actual" = "$(cat "$golden")" ]; then
    echo "PASS  $t"; pass=$((pass + 1))
  else
    echo "FAIL  $t (exit $code)"; fail=$((fail + 1))
    [ -f "$golden" ] && diff <(printf '%s\n' "$actual") "$golden" | head -20
  fi
done
echo "---- $pass passed, $fail failed"
[ $fail -eq 0 ]

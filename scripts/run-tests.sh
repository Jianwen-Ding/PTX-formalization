#!/usr/bin/env bash
# Run every tests/*.ptx through the K interpreter and compare stdout with the
# golden tests/*.out file; a test passes when the output matches and krun
# exits with 0 (all `.expect` checks held).
#   scripts/run-tests.sh [--update] [test.ptx ...]
# JOBS=N runs N tests in parallel (default 2; the large wgmma/mma tests need several GB each).
set -u
export K_OPTS="${K_OPTS:--Xss512m}"
cd "$(dirname "$0")/.."
export DEF=${PTX_DEF:-build/ptx-kompiled}
update=0
if [ "${1:-}" = "--update" ]; then update=1; shift; fi
export update
tests=("$@")
[ ${#tests[@]} -eq 0 ] && tests=(tests/*.ptx)

one() {
  t=$1; golden="${t%.ptx}.out"
  actual=$(krun -d "$DEF" "$t" --output none 2>/dev/null)
  code=$?
  if [ "$update" -eq 1 ]; then printf '%s\n' "$actual" > "$golden"; fi
  if [ $code -eq 0 ] && [ -f "$golden" ] && [ "$actual" = "$(cat "$golden")" ]; then
    echo "PASS  $t"
  else
    echo "FAIL  $t (exit $code)"
    [ -f "$golden" ] && diff <(printf '%s\n' "$actual") "$golden" | head -20
  fi
}
export -f one
out=$(printf '%s\n' "${tests[@]}" | xargs -P "${JOBS:-2}" -I{} bash -c 'one {}')
# tests killed for lack of memory (exit 137) when run in parallel are retried one at a time
retry=$(printf '%s\n' "$out" | grep '^FAIL.*(exit 137)' | awk '{print $2}')
if [ -n "$retry" ]; then
  out=$(printf '%s\n' "$out" | grep -v '^FAIL.*(exit 137)'; for t in $retry; do one "$t"; done)
fi
printf '%s\n' "$out"
pass=$(printf '%s\n' "$out" | grep -c '^PASS'); fail=$(printf '%s\n' "$out" | grep -c '^FAIL')
echo "---- $pass passed, $fail failed"
[ "$fail" -eq 0 ]

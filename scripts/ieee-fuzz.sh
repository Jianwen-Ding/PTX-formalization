#!/usr/bin/env bash
# Differential IEEE-754 fuzzing: for each seed, generate random binary32 and
# binary64 operands (incl. subnormals, infinities, signed zeros), run them
# through the K semantics and compare bit-for-bit with the host FPU.
#   scripts/ieee-fuzz.sh N SEED...
set -u
cd "$(dirname "$0")/.."
n=${1:-256}; shift || true
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(1 2 3 4)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
status=0
for s in "${seeds[@]}"; do
  python3 scripts/gen_ieee_diff.py "$n" "$s" > "$tmp/fuzz-$s.ptx" || exit 2
  out=$(krun -d build/ptx-kompiled "$tmp/fuzz-$s.ptx" --output none 2>/dev/null)
  code=$?
  np=$(grep -c '^PASS' <<< "$out"); nf=$(grep -c '^FAIL' <<< "$out")
  echo "seed $s: $np ops passed, $nf failed ($((n * (np + nf))) checks)"
  if [ $code -ne 0 ]; then status=1; grep -A5 '^FAIL' <<< "$out" | head -20; fi
done
exit $status

#!/usr/bin/env bash
# Differential check of rounding modes / .ftz / .sat against the exact-rational
# reference (scripts/ieee_ref.py).  usage: round-fuzz.sh N SEED...
set -u
cd "$(dirname "$0")/.."
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Xss512m"
n=${1:-40}; shift || true
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(11 12)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
status=0
for s in "${seeds[@]}"; do
  for w in 16 32 64; do
    python3 scripts/gen_round_diff.py "$n" "$s" "$w" > "$tmp/r.ptx" || exit 2
    out=$(krun -d build/ptx-kompiled "$tmp/r.ptx" --output none 2>/dev/null); code=$?
    np=$(grep -c '^PASS' <<< "$out"); nf=$(grep -c '^FAIL' <<< "$out")
    echo "seed $s f$w: $np configs passed, $nf failed"
    if [ $code -ne 0 ]; then status=1; grep -A4 '^FAIL' <<< "$out" | head -12; fi
  done
done
exit $status

#!/usr/bin/env bash
# Demonstrates exhaustive interleaving exploration with `krun --search`:
# for each example, list every distinct exit code reachable over all thread
# schedules (0 = all `.expect` checks hold, 1 = some check fails).
set -u
cd "$(dirname "$0")/.."
DEF=build/ptx-kompiled

outcomes() {
  krun -d $DEF "$1" --search --pattern '<exit-code> X:Int </exit-code>' 2>/dev/null \
    | grep -A2 'X:Int' | grep -E '^\s+[0-9]+$' | sort -u | tr -d ' ' | paste -sd, -
}

for ex in race race-atomic reduce-search reduce-nobarrier; do
  printf '%-34s reachable exit codes: {%s}\n' "examples/$ex.ptx" "$(outcomes examples/$ex.ptx)"
done
echo "examples/deadlock.ptx              final thread states:"
krun -d $DEF examples/deadlock.ptx 2>/dev/null | grep -A1 '<status>' \
  | grep -v -e '<status>' -e '--' | sed 's/^ */    /'

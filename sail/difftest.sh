#!/usr/bin/env bash
# Differential tests generated from the Sail models (see README.md, "Differential tests"):
#   AGX: LLVM MC layer vs encdec (mccheck.py), agx_run vs the Sail model (simcheck.py)
#   PTX: generated programs with Sail-model .expect values on ZLUDA + agx_run (ptxgen.py, ptxdiff.py)
# usage: sail/difftest.sh [SEED] [OUTDIR]     (environment: LLVM_BIN, AGX_RUN, ZLUDA, ZOC, APPLEGPU, JOBS)
set -u
cd "$(dirname "$0")"
SEED=${1:-1}
OUT=${2:-${TMPDIR:-/tmp}/ptx-difftest-$SEED}
JOBS=${JOBS:-2}
eval "$(opam env 2>/dev/null)" || true
make -s agx ptx JOBS="$JOBS" || exit 2
LOG=$(mktemp)
status=0
run() { echo "== $*"; "$@" 2>&1 | tee -a "$LOG" | tail -n 25; [ "${PIPESTATUS[0]}" -eq 0 ] || status=1; }
run python3 agx/gen/mccheck.py -s "$SEED" -n 3000 -w 3000 --reps 2 -j "$JOBS"
run python3 agx/gen/simcheck.py -s "$SEED" -n 400 -l 40 -j "$JOBS" --tests agx/tests/hw-semantics.agx agx/tests/binary-alu.agx agx/tests/atomic-variants.agx agx/tests/funop.agx agx/tests/formats.agx agx/tests/convert.agx agx/tests/stack.agx
rm -rf "$OUT"
run python3 ptx/gen/ptxgen.py "$OUT" -s "$SEED"
run python3 ptx/gen/ptxdiff.py "$OUT" --mode both -j "$JOBS" --json "$OUT/ptxdiff.json"
echo "==== summary (seed $SEED)"
grep -E "^(AGX MC|  (sail|applegpu|decode|reencode|text|asm) |AGX simulator|PASS|FAIL|---- [0-9]+ kernels|PTX (strict|default))" "$LOG"
rm -f "$LOG"
exit $status

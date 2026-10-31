#!/bin/sh
# Regenerate tests/stress/: the data-driven generators of scripts/ (expected values from Python transcriptions of
# the PTX ISA: bit-by-bit pseudo-code, big integers, the exact-rational IEEE reference ieee_ref.py) with seeds other
# than the ones of tests/ and more cases per launch.  The programs check only the ISA's results (no hardware
# polarity), so a backend can be run on them with tools like ZLUDA's agx_ptx_formalization_check.py.
#   usage: scripts/gen_stress.sh [N] [SEEDS...]      (default N 128, seeds 101 202 303; the matrix generators
#   with the first seed)
set -e
cd "$(dirname "$0")/.."
N=${1:-128}; [ $# -gt 0 ] && shift
SEEDS=${*:-101 202 303}
OUT=tests/stress
mkdir -p "$OUT"; rm -f "$OUT"/*.ptx   # (keeping its README.md)
g() { # name, generator, args...
    name=$1; shift
    python3 "scripts/$@" > "$OUT/$name.ptx" || { echo "generator failed: $*" >&2; rm -f "$OUT/$name.ptx"; }
}
for s in $SEEDS; do
    g bits-s$s gen_bits.py "$N" "$s"
    g carry-s$s gen_carry.py "$N" "$s"
    g intx-s$s gen_intx.py "$N" "$s"
    g fmisc-s$s gen_fmisc.py "$N" "$s"
    g lop3-s$s gen_lop3.py "$N" "$s"
    g video-s$s gen_video.py "$N" "$s"
    g ieee-diff-s$s gen_ieee_diff.py "$N" "$s"
    for w in 16 32 64; do g rounding-diff-f$w-s$s gen_round_diff.py "$N" "$s" "$w"; done
    for p in 0 1 2 3 4 5 6 7 8 9 10 11; do g cvt-$p-s$s gen_cvt.py "$N" "$s" "$p" 12; done
    for p in 0 1 2 3; do g half-$p-s$s gen_half.py "$N" "$s" "$p" 4; done
    for p in 0 1 2 3 4 5; do g pred-$p-s$s gen_pred.py "$N" "$s" "$p" 6; done
    for p in 0 1 2 3; do g mini-$p-s$s gen_mini.py "$N" "$s" "$p" 4; done
    for p in 0 1 2 3; do g warp-$p-s$s gen_warp.py "$s" "$p" 4; done
done
# The matrix generators (mma, mma.sp, wgmma, wgmma.sp, tcgen05.mma, block-scaled mma) with the first seed: they
# write tests/<name>.ptx under the working directory, so they run in a temporary one.
MS=${SEEDS%% *}
ROOT=$(pwd)
TMP=$(mktemp -d)
mkdir "$TMP/tests"
for p in 0 1 2 3 4 5 6 7; do python3 scripts/gen_mma.py "$MS" "$p" 8 > "$TMP/tests/mma-$p.ptx"; done
for gen in gen_mmasp gen_wgmma gen_wgmma_sp gen_tcgen05 gen_mma_bs; do
    (cd "$TMP" && python3 "$ROOT/scripts/$gen.py" "$MS" > /dev/null) || echo "generator failed: $gen" >&2
done
for f in "$TMP"/tests/*.ptx; do mv "$f" "$OUT/$(basename "$f" .ptx)-s$MS.ptx"; done
rmdir "$TMP/tests" "$TMP"
ls "$OUT" | wc -l

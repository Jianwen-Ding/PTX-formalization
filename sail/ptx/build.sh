#!/usr/bin/env bash
# Build the PTX Sail model with the C backend: sail/ptx/build/ptx (see sail/Makefile)
set -eu
cd "$(dirname "$0")"
eval "$(opam env 2>/dev/null)" || true
mkdir -p build
COMMON="../common/prelude.sail ../common/ints.sail ../common/lists.sail ../common/ieee.sail"
PTX="ptx_types.sail ptx_state.sail ptx_common.sail ptx_decls.sail ptx_alu.sail ptx_core.sail ptx_exec.sail ptx_coll.sail ptx_int.sail ptx_sched.sail ptx_half.sail ptx_mini.sail ptx_math.sail ptx_mma.sail ptx_ldm.sail ptx_async.sail ptx_dispatch.sail ptx_ext.sail ptx_main.sail"
sail --no-warn -c -c_include io.h -O $COMMON $PTX -o build/ptx
L=$(sail --dir)/lib
gcc -O1 -I"$L" -I../common build/ptx.c ../common/io.c "$L"/sail.c "$L"/rts.c "$L"/sail_failure.c "$L"/elf.c "$L"/sail_config.c "$L"/cJSON.c -lgmp -lz -o build/ptx
rm -f sail_smt_cache

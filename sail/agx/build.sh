#!/usr/bin/env bash
# Build the AGX Sail model with the C backend: sail/agx/build/agx (see sail/Makefile)
set -eu
cd "$(dirname "$0")"
eval "$(opam env 2>/dev/null)" || true
mkdir -p build
SAIL_SRCS="../common/prelude.sail ../common/ints.sail ../common/lists.sail ../common/ieee.sail agx_types.sail agx_state.sail agx_exec.sail agx_encdec.sail agx_asm.sail agx_main.sail"
sail --no-warn -c -c_include io.h $SAIL_SRCS -o build/agx
L=$(sail --dir)/lib
gcc -O2 -I"$L" -I../common build/agx.c ../common/io.c "$L"/sail.c "$L"/rts.c "$L"/sail_failure.c "$L"/elf.c "$L"/sail_config.c "$L"/cJSON.c -lgmp -lz -o build/agx

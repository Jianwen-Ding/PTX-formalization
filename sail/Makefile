# Sail models of AGX and PTX (C backend).  Needs Sail (opam: `eval $(opam env)`), gcc, libgmp, zlib.
#   make -C sail            build sail/agx/build/agx and sail/ptx/build/ptx
#   make -C sail test       run the AGX and PTX tests on the Sail models
AGX_SRCS := common/prelude.sail common/ints.sail common/lists.sail common/ieee.sail \
            agx/agx_types.sail agx/agx_state.sail agx/agx_exec.sail agx/agx_encdec.sail agx/agx_asm.sail agx/agx_main.sail

.PHONY: all agx test test-agx xcheck-agx clean

all: agx

agx: agx/build/agx

agx/build/agx: $(AGX_SRCS) common/io.c common/io.h
	bash agx/build.sh

test: test-agx

test-agx: agx
	python3 agx/agxtest.py --roundtrip
	python3 agx/agxtest.py --roundtrip agx/tests/*.agx
	python3 agx/agxtest.py --binary ../agx/tests/*.agx agx/tests/*.agx

# bit layouts against dougallj/applegpu (APPLEGPU=/path/to/applegpu)
xcheck-agx: agx
	PYTHONDONTWRITEBYTECODE=1 python3 agx/xcheck_applegpu.py -n 3000

clean:
	rm -rf agx/build ptx/build

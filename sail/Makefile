# Sail models of AGX and PTX (C backend).  Needs Sail (opam: `eval $(opam env)`), gcc, libgmp, zlib.
#   make -C sail            build sail/agx/build/agx and sail/ptx/build/ptx
#   make -C sail test       run the AGX and PTX tests on the Sail models (agx/tests, tests/*.ptx)
#   make -C sail test-ptx-extra   also tests/neg (expected failures) and tests/zluda (EXPECTED_PASS)
#   make -C sail report     rewrite ptx/TEST-REPORT.txt
AGX_SRCS := common/prelude.sail common/ints.sail common/lists.sail common/ieee.sail \
            agx/agx_types.sail agx/agx_state.sail agx/agx_exec.sail agx/agx_encdec.sail agx/agx_asm.sail agx/agx_main.sail
PTX_SRCS := common/prelude.sail common/ints.sail common/lists.sail common/ieee.sail $(wildcard ptx/*.sail)
JOBS ?= 2

.PHONY: all agx ptx isa-table test test-agx test-ptx test-ptx-extra report xcheck-agx difftest clean

all: agx ptx

agx: agx/build/agx

ptx: ptx/build/ptx

ptx/build/ptx: $(PTX_SRCS) common/io.c common/io.h
	bash ptx/build.sh

# ptx/ptx_isa_table.sail (the `.isacheck ;` rows, committed) is generated from the text of the PTX ISA 9.3
# specification, which is not in the repository (`pdftotext -layout ptx_isa_9.3.pdf isa.txt`):
#   make -C sail isa-table ISA_TXT=/path/to/isa.txt
isa-table:
	@test -n "$(ISA_TXT)" || { echo "usage: make -C sail isa-table ISA_TXT=/path/to/isa.txt" >&2; exit 2; }
	python3 ptx/gen_isa_table.py "$(ISA_TXT)" ptx/ptx_isa_table.sail

agx/build/agx: $(AGX_SRCS) common/io.c common/io.h
	bash agx/build.sh

test: test-agx test-ptx

test-ptx: ptx
	python3 ptx/ptxtest.py -j $(JOBS)

test-ptx-extra: ptx
	python3 ptx/ptxtest.py -j $(JOBS) ../tests/neg/*.ptx
	python3 ptx/ptxtest.py -j $(JOBS) ../tests/zluda/*.ptx

report: ptx
	python3 ptx/ptxtest.py -j $(JOBS) --report ptx/TEST-REPORT.txt ../tests/*.ptx ../tests/neg/*.ptx ../tests/zluda/*.ptx

test-agx: agx
	python3 agx/agxtest.py --roundtrip
	python3 agx/agxtest.py --roundtrip agx/tests/*.agx
	python3 agx/agxtest.py --binary ../agx/tests/*.agx agx/tests/*.agx

# bit layouts against dougallj/applegpu (APPLEGPU=/path/to/applegpu)
xcheck-agx: agx
	PYTHONDONTWRITEBYTECODE=1 python3 agx/xcheck_applegpu.py -n 3000

# differential tests generated from the models (LLVM MC, agx_run, ZLUDA): SEED=n
difftest:
	bash difftest.sh $(or $(SEED),1)

clean:
	rm -rf agx/build ptx/build

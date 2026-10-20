# Executable K semantics of NVIDIA PTX
#
#   make            build the interpreter (build/ptx-kompiled)
#   make test       run tests/*.ptx and compare against tests/*.out
#   make fuzz       differential IEEE-754 check against the host FPU
#   make build-search   build the exhaustive-search interpreter (build/ptx-search-kompiled)
#   make examples   run the examples (race detection with --search, deadlock)
#   make run FILE=prog.ptx

KOMPILE  ?= kompile
# A small young generation makes the JVM collect (and close) the source-file handles that kompile's
# parser leaks for every rule; without it a full parse runs out of file descriptors (ulimit -n).
export K_OPTS ?= -Xmn64m -Xmx10g
KRUN     ?= krun
BUILD    := build/ptx-kompiled
SEARCH   := build/ptx-search-kompiled
MM       := build/ptx-mm-kompiled
SOURCES  := $(wildcard semantics/*.k)
SEEDS    ?= 1 2 3 4 5 6 7 8
FUZZ_N   ?= 256

# Sail models (sail/): make sail builds them, make sail-test runs agx/tests and tests/*.ptx on them
.PHONY: sail sail-test difftest validate-agx properties build-agx test-agx isa-examples approx all build build-search build-mm memmodel litmus negative test fuzz examples zluda run clean

all: build

build: $(BUILD)/timestamp

$(BUILD)/timestamp: $(SOURCES)
	# two stages: the JVM frontend and llvm-kompile together exceed the sandbox's memory limit
	$(KOMPILE) semantics/ptx.k --main-module PTX --syntax-module PTX-SYNTAX \
	    --no-llvm-kompile --output-definition $(BUILD)
	cd $(BUILD) && llvm-kompile definition.kore dt main -- -o interpreter
	touch $(BUILD)/timestamp

# The exhaustive-search interpreter (krun --search, used by the litmus tests and examples) needs a second, much
# larger decision tree, so it is built separately (build/ptx-search-kompiled).
build-search: $(SEARCH)/timestamp

$(SEARCH)/timestamp: $(SOURCES)
	$(KOMPILE) semantics/ptx.k --main-module PTX --syntax-module PTX-SYNTAX \
	    --enable-search --no-llvm-kompile --output-definition $(SEARCH)
	cd $(SEARCH) && llvm-kompile definition.kore dt main -- -o interpreter
	cd $(SEARCH) && llvm-kompile definition.kore dt search -- -o search
	touch $(SEARCH)/timestamp

litmus: build-mm
	PTX_DEF=$(MM) bash scripts/run-litmus.sh

memmodel: build-mm
	python3 scripts/ptxmm.py && PTX_DEF=$(MM) python3 scripts/run-memmodel.py

# The reduced definition (semantics/ptx-mm.k: no tensor cores, TMA, textures, bulk copies) with the search interpreter:
# used for the litmus tests, the memory-model cross-check and the examples that explore all schedules.
build-mm: $(MM)/timestamp

$(MM)/timestamp: $(SOURCES)
	$(KOMPILE) semantics/ptx-mm.k --main-module PTX-MM --syntax-module PTX-SYNTAX \
	    --enable-search --no-llvm-kompile --output-definition $(MM)
	cd $(MM) && llvm-kompile definition.kore dt main -- -o interpreter
	cd $(MM) && llvm-kompile definition.kore dt search -- -o search
	touch $(MM)/timestamp

negative: build
	bash scripts/run-negative.sh

test: build
	@scripts/run-tests.sh

fuzz: build
	@scripts/ieee-fuzz.sh $(FUZZ_N) $(SEEDS)
	@scripts/round-fuzz.sh 40 $(SEEDS)

zluda: build
	@scripts/run-zluda.py

examples: build-mm
	@PTX_DEF=$(MM) scripts/run-examples.sh

run: build
	$(KRUN) -d $(BUILD) $(FILE)

clean:
	rm -rf build

# the approximate instructions (rcp/sqrt/rsqrt/sin/cos/ex2/lg2/tanh .approx) against the error bounds of the ISA
approx: build
	python3 scripts/check_approx.py

# parse every example of the PTX ISA document (fetch the HTML first: scripts/extract_isa_examples.py explains)
isa-examples: build
	python3 scripts/extract_isa_examples.py $(ISA_HTML) /tmp/isa_examples.json
	python3 scripts/isa_examples.py /tmp/isa_examples.json

# the Apple AGX (G13) semantics, built separately from the PTX definition (it reuses the PTX float core)
AGX := build/agx-kompiled
build-agx: $(AGX)/timestamp
$(AGX)/timestamp: agx/agx.k semantics/ptx-float.k semantics/ptx-common.k semantics/ptx-syntax.k
	$(KOMPILE) agx/agx.k --main-module AGX --syntax-module AGX-SYNTAX --no-llvm-kompile --output-definition $(AGX)
	cd $(AGX) && llvm-kompile definition.kore dt main -- -o interpreter
	touch $(AGX)/timestamp
test-agx: build-agx
	bash scripts/run-agx.sh

# schedule-independence properties of the example programs (exhaustive search with the reduced definition)
properties: build-mm
	python3 scripts/properties.py examples/race.ptx examples/race-atomic.ptx examples/reduce-nobarrier.ptx examples/reduce-search.ptx examples/deadlock.ptx examples/collective-mismatch.ptx || true

# translation validation of the PTX -> AGX lowering (needs build and build-agx)
validate-agx: build build-agx
	python3 tools/validate_agx.py

sail:
	$(MAKE) -C sail all

sail-test:
	$(MAKE) -C sail test

difftest:
	$(MAKE) -C sail difftest

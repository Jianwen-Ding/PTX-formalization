# Executable K semantics of NVIDIA PTX
#
#   make            build the interpreter (build/ptx-kompiled)
#   make test       run tests/*.ptx and compare against tests/*.out
#   make fuzz       differential IEEE-754 check against the host FPU
#   make examples   run the examples (race detection with --search, deadlock)
#   make run FILE=prog.ptx

KOMPILE  ?= kompile
# A small young generation makes the JVM collect (and close) the source-file handles that kompile's
# parser leaks for every rule; without it a full parse runs out of file descriptors (ulimit -n).
export K_OPTS ?= -Xmn64m
KRUN     ?= krun
BUILD    := build/ptx-kompiled
SOURCES  := $(wildcard semantics/*.k)
SEEDS    ?= 1 2 3 4 5 6 7 8
FUZZ_N   ?= 256

.PHONY: all build test fuzz examples zluda run clean

all: build

build: $(BUILD)/timestamp

$(BUILD)/timestamp: $(SOURCES)
	$(KOMPILE) semantics/ptx.k --main-module PTX --syntax-module PTX-SYNTAX \
	    --enable-search --output-definition $(BUILD)

test: build
	@scripts/run-tests.sh

fuzz: build
	@scripts/ieee-fuzz.sh $(FUZZ_N) $(SEEDS)
	@scripts/round-fuzz.sh 40 $(SEEDS)

zluda: build
	@scripts/run-zluda.py

examples: build
	@scripts/run-examples.sh

run: build
	$(KRUN) -d $(BUILD) $(FILE)

clean:
	rm -rf build

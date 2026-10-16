# Prior work survey (task 41)

Rule for this project: reuse trusted, widely used formalizations and tools
instead of re-deriving them. Findings as of 2026-10-16.

## Existing PTX formalizations

| Work | What it is | Reusable here? |
|---|---|---|
| Lustig, Sahasrabuddhe, Giroux, *A Formal Analysis of the NVIDIA PTX Memory Consistency Model* (ASPLOS 2019) | Axiomatic PTX memory model (Alloy, mapped to Coq with a soundness proof of the C++ mapping) | Model is the right spec for task 36. No public source repository found; implement from the paper and PTX ISA memory-model chapter, validate on litmus tests |
| Lustig et al., *Mixed-Proxy Extensions for the NVIDIA PTX Memory Consistency Model* (ISCA 2022) | Adds proxies (generic/texture/surface/constant/async) | Same: spec for the proxy part of task 36 |
| Ferrell, Duan, Hamlen, *CUDA au Coq* (DATE 2019) | Coq operational semantics of PTX (warps, blocks, grids) | Not importable into K (Coq, no public repo found). Useful as a cross-check of the execution model |
| Hathhorn et al., CUDA-C in K | K semantics of CUDA-C, not PTX | Different language level; confirms K suits this domain |
| herdtools7 (`herd/libdir`) | Memory-model tooling; has no PTX cat model (PTX/GPU support deprecated) | Cloned and checked: nothing to import. Its litmus format is a reference for task 36 |

No K, Isabelle or Lean semantics of PTX itself was found, so the K definition
in `semantics/` remains the first executable one.

## Existing K semantics (machine-level building blocks)

* **K's own builtins** (`FLOAT` over GNU MPFR, `BYTES`, `INT`/`MInt`) are used
  directly; this is what KWasm and KEVM rely on too.
* **KWasm** (runtimeverification/wasm-semantics) also builds on K's `FLOAT`
  (`signFloat`, `==Float`, ...) and implements Wasm's IEEE-754 behaviour with
  the same builtins, so there is no separate float library to import.
* **KEVM** has 256-bit word and `Bytes` memory helpers; PTX needs only the
  builtin `Bytes` functions already used.

## Trusted test oracles (imported)

* **Berkeley SoftFloat 3e / TestFloat 3e** (J. Hauser, BSD licence; the
  reference implementation of IEEE-754 used to validate hardware and
  libraries). `scripts/gen_testfloat.py` drives `testfloat_gen` to produce
  `tests/testfloat.ptx` (f32/f64 add, sub, mul, div, sqrt, fma, round to
  nearest even, 150 vectors each, spread across TestFloat's level-1 cases).
  All 1800 vectors pass. The same generator also produced the directed-rounding
  files `tests/testfloat-{rz,rm,rp}.ptx` (TestFloat modes `minMag`, `min`,
  `max`; 60 vectors per op and format), which pass too. f16, bf16, conversions
  and comparisons follow with tasks 11-14.
  Regenerate:
  ```sh
  git clone --depth 1 https://github.com/ucb-bar/berkeley-softfloat-3
  git clone --depth 1 https://github.com/ucb-bar/berkeley-testfloat-3
  make -C berkeley-softfloat-3/build/Linux-x86_64-GCC
  make -C berkeley-testfloat-3/build/Linux-x86_64-GCC testfloat_gen \
       SOFTFLOAT_DIR=../../../berkeley-softfloat-3
  python3 scripts/gen_testfloat.py path/to/testfloat_gen 1 150 > tests/testfloat.ptx
  ```
  The generated file is checked in, so CI needs neither tool.
* The host FPU and exact rational arithmetic (`scripts/gen_ieee_diff.py`)
  stay as independent second oracles.

## Consequences for the roadmap

* Task 36: implement from the ASPLOS'19 / ISCA'22 axioms, check against the
  litmus outcomes in those papers; no code to import.
* Tasks 4, 11-14: extend `gen_testfloat.py` to each rounding mode and format
  TestFloat supports (f16, bf16, directed rounding, int/float conversions).
* Task 39: the TestFloat test takes ~55 s because every store copies the
  whole `Bytes` memory; this is the first performance target.

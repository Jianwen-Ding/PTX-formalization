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

## Imported test corpora (added during task 6)

* **ZLUDA** (https://github.com/vosen/ZLUDA, MIT OR Apache-2.0): ~190 small PTX
  programs with inputs and outputs recorded from real NVIDIA hardware.
  `scripts/import_zluda.py` converts them into self-contained programs for this
  semantics (`tests/zluda/`), `scripts/run-zluda.py` runs them and keeps
  `tests/zluda/EXPECTED_PASS` as a regression list. Importing them already
  exposed missing grammar (registers without `%`, `U`-suffixed literals), a
  wrong `setp.num` with NaNs, missing sign extension for narrow `cvt`/`ld`
  results, and the carry polarity of `sub.cc`.
* **gpgpu-sim** (BSD-style licence): its functional PTX simulator was used as an
  independent reading of `prmt` modes, `bfe`, `bfi` and `bfind`.

## Sparse mma.sp metadata layout (assumption)

The PTX ISA text specifies the register fragments of the compressed matrix A with formulas, but
gives the metadata (`e`) layout only in figures (Figures 122-142), which are not available as
text.  `semantics/ptx-exec-sp.k` therefore uses the assignment implied by the formulas: a row of
`nc` chunks needs `4*nc` bits; 16 bits -> lane `4g+f` holds rows g/g+8 in its low/high half;
32 bits -> lane `4g+2f+rowhalf`; 64 bits -> lane `4g+2*(chunk div 8)+rowhalf`.  Nibbles are
stored little-endian in chunk order.  Tests `tests/sp-*.ptx` are generated with the same
assumption, so they check the arithmetic and the A/B/C fragment formulas, not this layout; it
should be validated against hardware (e.g. CUTLASS' sparse metadata reorder tables).

## wgmma (sm_90a) modelling choices

`wgmma.mma_async` is executed as a 128-thread rendezvous; the last arriver snapshots shared
memory and the whole product is computed at that point (one legal schedule of the asynchronous
operation), so `wgmma.fence/commit_group/wait_group` are no-ops.  The swizzled shared-memory
layouts are transcribed from the CuTe canonical-layout table of the ISA (swizzle functors are
applied to byte addresses); the D fragment is the m16n8 accumulator layout repeated every 8
columns per warp (the ISA gives it only as a figure).  The tests build shared memory from
swizzle atoms (independently of the address formula) but share the D/A fragment assumption.
`wgmma.mma_async.sp` is not modelled.

## tcgen05 (sm_100a) modelling choices

Tensor memory is a per-CTA map (lane, column) -> 32-bit word.  `tcgen05.alloc/dealloc` are warp
collectives that allocate first-fit (aligned to their size) and fault when memory is exhausted
(hardware blocks).  `tcgen05.ld/st` are exact for `.32x32b`; the 16-lane shapes use the thread
mapping documented in `semantics/ptx-exec-tc.k`, which the ISA gives only as figures, and the
tests only check that each shape round-trips.  `tcgen05.mma` (dense, `.cta_group::1`,
`.kind::{f16,tf32,i8,f8f6f4}`, A from shared or tensor memory) is executed atomically at issue
and writes D with row r in lane r, column j in column j (assumption).  Instruction-descriptor
fields and the sm_100 shared-memory descriptor follow the ISA tables (text); swizzled layouts
reuse the wgmma canonical layouts.

Further tcgen05 modelling (layouts the ISA gives only as figures are *assumptions*, each checked only
against an independent Python model in `scripts/gen_tcgen05_*.py`):

* `.sp`: row r's metadata is in TMEM lane r, 4 bits per chunk, in the word at `meta_col + sparsity selector`
  (16-bit/tf32 kinds) or two consecutive words (8-bit kinds).
* `.ws`: the zero-column mask descriptor is a periodic pattern (period skip+use, offset `sc`, initial value `fs`),
  one sub-mask per 128/M row group, each covering N/(128/M) columns; the four examples in the ISA text are reproduced
  by the generator's model.  The `shift` field moves B columns.  Collector qualifiers are accepted and have no effect
  (no collector buffer is modelled); `.ashift` shifts A rows by one.
* fp6 / fp4 operands of `kind::f8f6f4` live in 8-bit containers; `mxf4`/`mxf4nvf4` use packed 4-bit elements.
* Block scaling (`kind::mxf8f6f4`, `mxf4`, `mxf4nvf4`): scale A for row r is byte `SFA_ID+blk` of the word at lane r,
  column `[scale-A]`; scale B for column n is byte `SFB_ID+blk` of the word at lane n mod 32, column `[scale-B] + n/32`.
  Scale types UE8M0 and UE4M3 (idesc bit 23); block 32 (`scale_vec::1X`/`2X`) or 16 (`4X`).
* `tcgen05.cp` reads source rows as K-major 32-bit elements; `shift` moves rows down by one lane; `commit` signals
  its mbarrier.

Not modelled: `.cta_group::2`, `tcgen05.cp` decompression, `ld.red`/`spcompress`, commit multicast.

## Tensor maps / TMA

The 128-byte tensor-map object is produced by the host driver API on real hardware and is opaque
to PTX.  The harness directive `.tensormap NAME .type BUFFER .dims {..} .strides {..} .box {..}
.swizzle N;` builds one in an encoding private to this semantics (see `tmBlob` in
`semantics/ptx-harness.k`); tile/gather4 loads, tile/scatter4 stores and reductions, prefetch,
`tensormap.replace` and `tensormap.cp_fenceproxy` interpret it.  Interleaved layouts, im2col
modes and `.override` forms are not modelled.

## Relaxed memory mode (`.memmodel relaxed;`)
Operational under-approximation of the PTX memory model: global/shared stores enter a per-thread store buffer
(`<sb>`), loads forward from it, and an idle thread may retire its oldest buffered store or a younger one that
may overtake the older ones (different location, no `st.release` involved; reorder window of three).  Fences,
barriers, atomics, collectives, mbarrier/cp/wgmma/tcgen05 and thread exit drain the buffer first.  Memory is
multi-copy atomic; load-buffering reorderings and scope hierarchy are not modelled.  The litmus suite
(`make litmus`: SB, SB+fence.sc, SB under SC, MP, MP+release/acquire) checks the allowed/forbidden outcomes.
A design constraint of the LLVM backend: rules that match across `<ctas>/<cta>/<thread>` with list patterns or
high-priority catch-alls on `exec(...)` made kompile run out of memory, hence the thread-local formulation.

## Version / target legality (`.isacheck ;`)
`semantics/ptx-isa-table.k` (generated by `scripts/gen_isa_table.py` from the PTX ISA document) lists for each base
instruction the minimum PTX version, minimum sm and whether an arch-specific (`a`/`f`) target is required.  With
`.isacheck ;` the launch faults if any instruction of the module exceeds the declared `.version`/`.target`.
Qualifier-level requirements are not checked.  `make negative` runs `tests/neg/`.

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
  its mbarrier (multicast: one arrive per CTA in the mask).  `.cta_group::2` makes both CTAs of the pair copy from their own
  shared memory (same descriptor / taddr) or shift their own tensor memory.
* `tcgen05.cp` decompression (`.b8x16.b6x16_p32`, `.b8x16.b4x16_p64`): groups of four source words (16 elements of 6 / 4 bits
  followed by padding) become four words of 8-bit elements; an element's value is placed in the low bits of its byte
  (assumption: the bit mapping is in figures).
* `tcgen05.ld.red` reduces the values a thread loaded (f32 min/max with `.abs`/`.NaN`, or u32/s32).
  `tcgen05.ld[.red].spcompress` keeps, in each group of four loaded f32 values, two (by `.min`/`.max`, optionally `.abs`),
  writes them in index order to `cdata` and their 2-bit indices (packed LSB first) to `mdata`.  Ties keep the lowest index,
  NaNs are always kept (the ISA says these are implementation-specific).  `spcompress`/`spdecompress` (PTX ISA 9.4) use the
  same selection; vectors are bit-packed little-endian across their registers and `spdecompress`'s `.numN` counts
  source groups.

Not modelled: `.cta_group::2` for sparse / block-scaled MMA, `.decompress::lut::b` (needs the lookup-table figures), `kind::ti16`.

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

## mma.sync with fp6 / fp4 and block scaling (sm_120a, PTX ISA 8.7)

`kind::f8f6f4` takes fp6 / fp4 elements in 8-bit containers (ISA text: `e2m1` in the central 4 bits, `e3m2` / `e2m3` in the low 6 bits;
the other container bits are ignored).  `kind::mxf4` / `mxf4nvf4` use two packed 4-bit elements per byte, with the `m16n8k64` 4-bit
fragment layouts.  For `.block_scale` the ISA gives the placement of the scale factors only as figures; the model assumes: the
factors of row r of scale_A come from the thread of quad (r mod 8) whose lane % 4 is `2 * thread-id-a + r div 8`, the factors of column
n of scale_B from the thread of quad n whose lane % 4 is `thread-id-b`; chunk j (`scale_vec::NX` has N chunks per row / column)
is byte `byte-id + j` of the 32-bit scale register; selectors are read from lane 0.  The sparse forms (`mma.sp::ordered_metadata`
with `kind::f8f6f4` or block scaling, `m16n8k64` / `m16n8k128`) reuse the sparse A fragment and metadata model.
Test generators: `scripts/gen_mma_bs.py`, `scripts/gen_mmasp_bs.py`.

tcgen05 kinds `f8f6f4` / `mxf8f6f4`: fp6 / fp4 elements are bit-packed in shared memory (ISA 9.7.18.10.4.4): with Dense-K 32 every 16
elements occupy a 16-byte chunk (padding after the 12 / 8 data bytes), with Dense-K 64 rows are contiguous.  K-major tiles only.

## ldmatrix / stmatrix sm_100 forms

`ldmatrix.m16n16` (`.trans` mandatory; 16 row addresses of 16 bytes per matrix, `.x1`/`.x2`) gives thread (g, t) register 2i = bytes M[4t+b][g] and
register 2i+1 = M[4t+b][g+8] (b = 0..3).  `ldmatrix.m8n16` gives thread (g, t) the elements 4t..4t+3 of row g.  With `.b8x16.b6x16_p32` /
`.b4x16_p64` / `.s8.s4` the row's 16 elements are expanded to bytes (fp6: low 6 bits, fp4: central 4 bits as for mma.sync containers, s4:
sign-extended).  `stmatrix.m16n8.b8` stores the 16 x 8 matrix transposed: memory row n (addressed by thread 8i+n, 16 bytes) holds X[0..15][n] with
thread (g, t) holding X[g][2t..2t+1] in bytes 0-1 and X[g+8][2t..2t+1] in bytes 2-3.  These per-thread layouts are figures in the ISA (assumed);
tests: `scripts/gen_ldm2.py`.

## TMA im2col modes

`cp.async.bulk.tensor ... .im2col` (3D/4D/5D), `.im2col::w`, `.im2col::w::128` and the store/reduce form `.im2col_no_offs::w`.  The ISA text
defines the parameters (bounding-box corners, pixels per column, channels per pixel, traversal strides, im2col offsets, `wHalo`, `wOffset`) but
the exact pixel enumeration only in figures.  Model: tensor dim 0 = channels, dims 1..S = W, H, D, last = N.  Along spatial dim k the filter-base
positions are `L_k, L_k + st_k, ...` up to `size_k - 1 + U_k` (so `L = -pad`, `U = pad - (filter - 1) * dilation` gives the usual convolution
output grid, matching the ISA's examples); the tensor coordinates give the first pixel's position (its box index is `(c_k - L_k) div st_k`); successive
pixels run W-fastest, wrapping W into H into D into N; a pixel reads at base position + im2col offset, zero / NaN fill outside the tensor.  The `::w` modes vary
only W (`c_W + q * st_W + wOffset`), load `pixels-per-column + wHalo` pixels (`::128`: 128 pixels with `wHalo` halo pixels after every 32);
`im2col_no_offs::w` writes pixels only up to the first one outside the tensor.  Tensor-map encoding (harness `.tensormap ... .im2col {lower} {upper} .estrides {..}`):
corners offset-binary in words 1 / 2, flag in word 14.  Tests: `scripts/gen_tma_im2col.py`.

## TMA swizzle sub-modes and out-of-bounds fill

Tensor-map swizzle codes follow the driver enum: 0 none, 1 32B, 2 64B, 3 128B, 4 128B with 32B atomicity (chunk pairs move together, line mod 4),
5 the same plus an 8-byte flip of the two halves of every 16-byte chunk on odd 128-byte lines, 6 128B with 64B atomicity (groups of four chunks, line mod 2).
The sm_100 shared-memory matrix descriptor value 1 ("128B with 32B atomicity") is the same pattern as mode 4.  Not modelled: the 96B swizzle
(the ISA gives its pattern only as a figure).  OOB-NaN fill writes a quiet NaN of the element type (f16 0x7e00, bf16 0x7fc0, f32/tf32 0x7fc00000, f64).
L2 promotion and cache hints have no architectural effect.  Interleaved layouts (NC/8HWC8, NC/16HWC16) are plain tiled copies of the dims as seen in
global memory.  Tests: `scripts/gen_tma_swz.py`.

## Tensor maps: encoding, tensormap.replace, overrides, packed sub-byte types

The tensor-map object is our own 128-byte encoding (the real one is opaque).  Element types use the numbering of `tensormap.replace.elemtype` (ISA table: 0 u8 ... 6 f16,
7 f32, 8 f32.ftz, 9 f64, 10 bf16, 11 tf32, 12 tf32.ftz, 13 b4x16, 14 b4x16_p64, 15 b6x16_p32), which differs from the driver's `CUtensorMapDataType`.  Swizzle is stored as
`swizzle_mode` (0 none, 1 32B, 2 64B, 3 128B) plus `swizzle_atomicity` (0 16B, 1 32B, 2 32B + 8B flip, 3 64B); the harness `.swizzle Z` takes the driver enum (4 / 5 / 6 =
128B with 32B / 32B+flip / 64B atomicity) and fill mode in bits 8...  Every field of `tensormap.replace` (incl. `swizzle_atomicity`, `interleave_layout`, `rank`) is modelled.
`.override::global_address`, `.override::global_dim` and `.override::global_dim_stride` patch a copy of the tensor map before use (stride = (lower + (upper << 32)) << 4).
Packed sub-byte types (elemtype 13-15) load groups of 16 elements (8 or 12 global bytes) into 8 or 16-byte shared-memory groups (padding written as zeros), i.e. the
format `tcgen05.mma` kinds f8f6f4 / mxf8f6f4 read; stores of `.b6p2x16` are not modelled.  Tests: `scripts/gen_tma_rep.py`.

## cp.async.bulk.tensor: multicast, cta_group::2, reduce

`.multicast::cluster` / `::16b` / `::32b` copy to the same CTA-relative offset in every CTA of the mask and signal the same-offset mbarrier there.  With `.cta_group::2` the
signal for destination CTA d goes to the CTA of d's pair whose rank parity equals that of the CTA holding `mbar` (so one barrier of the pair collects the bytes of both halves).
`cp.reduce.async.bulk.tensor` reduces element-wise (atomically per element) for the full table of the ISA: add (u32 s32 u64 f32 f16 bf16), min / max (u32 s32 u64 s64 f16 bf16),
inc / dec (u32), and / or / xor (b32 b64), in tile, `.im2col_no_offs` and `.im2col_no_offs::w` modes.  `prefetch.tensor` is a no-op.  Tests: `scripts/gen_tma_red.py`.

## mbarrier .multicast::cluster::32b (PTX ISA 9.4)

`mbarrier.expect_tx`, `mbarrier.complete_tx`, `mbarrier.arrive` and `mbarrier.arrive.expect_tx` with `.multicast::cluster::32b` apply the operation to the mbarrier at the same
CTA-relative offset in every CTA whose rank bit is set in the 32-bit `ctaMask` (arrive: optional count, then the mask).  Test: `scripts/gen_mbar_mc.py`.
Not modelled: the conditional phase / payload report of `.layout::v1` mbarriers (`report-on`) used by `.mbarrier::report::*` of the async copies and by fabric operations.

## Textures: mipmaps, cubemaps, depth compare (own model; the ISA gives no numeric rules)
* Texture objects gain levels (word 12), flags (13: bit 0 depth, bit 1 cube) and a mip filter (15); samplers gain mip filter, minLOD, maxLOD (1/256 units).  Levels are stored one after another, each holding all layers.
* lod: `.base` 0, `.level` the operand, `.grad` log2 of the larger scaled gradient length (cube: divided by 2*major axis); clamped to [minLOD, min(maxLOD, levels-1)]; point mip = floor(lod+1/2), linear mip = lerp of two levels, weight quantised to 1/256.
* Cube faces follow OpenGL (+X -X +Y -Y +Z -Z); filtering clamps inside the face (no seam blending).  Depth compare is LEQUAL on each texel before filtering; offsets add to texel indices; the residency predicate is always true.
* Divisions in the texture unit are computed at double precision: `mpfr_div` at the 4400-bit working precision crashed (SIGFPE) or returned wrong values.
* Build note: the JVM frontend plus llvm-kompile exceed the sandbox memory limit, so the Makefile runs kompile with `--no-llvm-kompile` and then llvm-kompile.

# Review checklist: the Sail PTX model (`sail/ptx`)

Reference: the public **PTX ISA, Release 9.3** (NVIDIA, Jun 25 2026), cited by section, table, figure or page.
The K semantics in `semantics/` is the model `sail/ptx` was written from and is kept in step with it, but it is not
the reference for correctness.

## Fixes in commit "fixes from the Sail review", checked against the ISA

| Fix | ISA reference | Verdict |
|---|---|---|
| `mapa` with no `.space` takes and returns generic addresses | §9.7.9.24: "When the optional qualifier .space is not specified, both a and d are registers containing generic addresses pointing to shared memory." | confirmed |
| `getctarank` with no `.space` takes a generic address | §9.7.9.25: "When the optional qualifier .space is not specified, a is a register containing a generic addresses pointing to shared memory." | confirmed |
| `tanh.approx.f16` is ±1 instead of NaN for 11.09 ≤ \|x\| < 20 | §9.7.4.9: "The maximum absolute error for .f16 type is 2^-10.987" (NaN violates it) | confirmed |
| `lg2.approx.f32` rounded once | §9.7.3.20: max absolute error 2^-22 on (0.5, 2), max relative error 2^-22 elsewhere. The old three-rounding result (relative error ≈ 2^-24 at 2^-126) was **within** that bound | not an ISA defect; kept because the model's stated choice is to return the correctly rounded value for `.approx` |
| ue8m0 scale 0xFF is NaN (block-scaled `mma`, `tcgen05.mma`) | §5.2 (alternate formats, line "ue8m0 data format"): "NaN value is limited to 0xff" | confirmed |
| `mma.sync.m8n8k4` with non-`.f64` inputs is unsupported instead of running on the f64 fragments | §9.7.15.5.1 and Figures 46–52: `.f16` m8n8k4 is a separate form (4 MMA operations, its own fragments) | confirmed defect; the form itself is still to be implemented |
| tcgen05 Negate A/B flips bit 5 of fp6 / fp4 elements | Table 56 (negate supported for `.kind::f8f6f4`, `.kind::mxf8f6f4`); fp6 `.b8` elements have "2 MSB bits padded" (§5.2); Figure 195: fp4 in an 8-bit container is `0 0 S E2 M1 0 0` (sign at bit 5); Figure 197: fp6 is `0 0 S …` | confirmed |
| `tcgen05.mma.cta_group::2` with a non-zero disable-output-lane stops as not modelled | §9.7.17.10 (tcgen05.mma): a set bit means the lane "will not be updated"; vector size 8 for `::2`. The split of the 256 bits between the CTA pair is not stated | confirmed defect (the mask was ignored); bit-to-CTA mapping unspecified, so not modelled |
| `.ashift` with `.sp`: rows of the expanded A shift | §9.7.17.10: ".ashift shifts the rows of the A matrix down by one row, except for the last row in the Tensor Memory", also allowed on `tcgen05.mma.sp`. Whether the compressed data or the logical matrix shifts (and whether the metadata moves) is not stated | **open**: the ISA does not decide between the old and new behaviour |
| textures: wrap / mirror with unnormalized coordinates clamp | §5.3.2 defers address modes to the CUDA C++ Programming Guide ("See the CUDA C++ Programming Guide for more details") | not in the PTX ISA; follows the CUDA guide |
| a call whose return list does not match the declaration faults (Sail looped forever) | §9.7.13 call: "Input arguments and return values are optional"; operands are type-checked against flist / fproto | the loop was a defect; that such a call must fault is not stated by the ISA |
| `d\|p` operands reject a literal after the bar | §9.7.14.15 elect.sync: `d` is a register or the sink `_`, `p` a predicate | confirmed |
| thread exit rechecks every barrier with arrivals (Sail checked ids 0–15) | §9.7.14 barrier: "barriers numbered 0..15" | matches K; only reachable with ids the ISA disallows |

## Found, not fixed yet

- [x] ue4m3 scale 0x7F is not NaN in block-scaled `mma` / `tcgen05.mma`, and bit 7 is not masked. §5.2: ue4m3 "NaN value is
      limited to 0x7f", "MSB bit padded with zero". Also: the tcgen05 idesc scale type is the one bit 23 (UE4M3 = 0,
      UE8M0 = 1; bits 24-26 reserved); the model read bits 23-24 and had a bias-15 format the ISA does not define.
      Fixed; `tests/mma-bs-nan.ptx`.
- [ ] Implement `mma.sync.m8n8k4` with `.f16` / `.f32` accumulators from §9.7.15.5.1 (Figures 46–52).
- [ ] Barrier ids ≥ 16 should be rejected (§9.7.14: 0..15) rather than accepted.
- [ ] `clusterlaunchcontrol.try_cancel … .multicast::cluster::all` writes the response and signals the mbarrier only in
      the issuing CTA. To check against the ISA section on clusterlaunchcontrol.
- [ ] `sail/common/io.c`: addresses wrap at 2^64 (`ad + i`), keys outside [0, 2^64) are dropped; `calloc`/`realloc`
      results unchecked.
- [ ] `docs/prior-work.md` (tcgen05.cp decompression) says 4-bit values land in the low bits; Figure 195 puts them in
      bits 2–5, as the code does. The header of `semantics/ptx-exec-tc.k` lists implemented features as not modelled.
- [ ] `.ashift` with `.sp`: decide the reading (see table) and document it.

## Integer review against the ISA (fixed)

- [x] `bmsk`: the Semantics pseudo-code (§9.7.1, bmsk) on the full 32-bit operands; the model and `scripts/gen_bits.py`
      masked a and b to 8 bits first (`bmsk.clamp 256, 4` gave 0xF, the ISA gives 0).
- [x] `vshl` / `vshr` `.clamp`: "if ( mode == .clamp && tb > 32 ) tb = 32" on the full value (was masked to 8 bits).
- [x] `vset d.dsel, a, b, c`: "d = optMerge( dsel, tmp, c )" (the merge form wrote nothing).
- [x] `lop3.BoolOp _|p`: "The sink symbol '_' may be used in place of the destination operand d" (was unsupported).
- [x] `setp` / `set` / `slct`: "Modifier .ftz applies only to .f32 comparisons" (f64 subnormals were flushed).
- [ ] **ISA vs hardware, not changed:** `sub.cc` "borrow-out written to CC.CF" and `subc` "d = a - (b + CC.CF)". The model
      keeps CC.CF = carry of a + ~b + 1 (1 = no borrow), because ZLUDA's hardware-recorded `subc_cc_s32` reads
      CC.CF = 0 after `sub.cc` with a < b, which the ISA text would make 1. Needs a decision.

## Sync / async / textures review against the ISA (fixed, Sail only)

- [x] cp.async: with `.L2::cache_hint` a single trailing operand is the cache_policy (§9.7.9.26.3.1, ~15824-15880)
- [x] warp collectives (shfl, vote, match, redux, bar.warp.sync) wait only for non-exited lanes of the membermask
      (~12949, 21672-21767); match.all returns the participating lanes; elect.sync keeps the full mask ("all threads")
- [x] redux.sync `.abs` (~21878)
- [x] activemask: exited, inactive and predicated-off lanes contribute 0 (~21812-21818): the lanes executing the same
      dynamic instance together under the round-robin schedule
- [x] clusterlaunchcontrol.try_cancel `.multicast::cluster::all`: response and complete-tx in every CTA (~23576-23579)
- [x] mbarrier test_wait / try_wait reportPredicate / reportValue operands (zero for `.layout::v0`, ~23064-23067)
- [x] txq `.array_size` / `.num_mipmap_levels` 0 when not array / mipmapped, sampler queries, `txq.level` (~19255-19264)
- [x] counted CTA barriers mark warp arrivals (~20252-20261)
- [x] 96B swizzle (Table 33, Figure 31)

## Matrix review against the ISA (fixed, Sail only)

- [x] `mma.m8n8k4` with `.f16` inputs (§9.7.15.5.1, Figures 46-52); `.ctype .f32` requires `.dtype .f32` (~26898)
- [x] f64 `mma` `.rn/.rz/.rm/.rp` (~26937-26942)
- [x] sparse metadata of 16/32-bit `mma.sp` and `wgmma.sp` (Figures 122, 125, 176, 178)
- [x] tcgen05.ld / st `.16x64b`, `.16x128b`, `.16x256b` cells (Figures 184-186)
- [x] tcgen05 block-scale factor placement incl. K = 96 column groups (Figures 231, 234-251)
- [x] tcgen05.mma.sp `kind::f16` / `tf32` metadata (Figures 262-265)
- [x] matrix descriptor bit 52 (absolute leading-dimension mode): the ISA gives no layout, so the model stops with an error
- [ ] open: f64 `mma` "precision ... identical to that of .f64 precision fused multiply-add" (the model rounds the exact
      sum once); Figure 187 (`.16x32bx2`) draws threads 16-31 at lanes 16-31 while the text and the model use t % 16;
      M = 64 "Layout F" lane placement is not modelled

## Where the Sail model now differs from the K semantics

The K semantics in `semantics/` is no longer kept in step for the fixes above (it cannot be built here). Differences:
cp.async operand parsing; redux `.abs`; exit-tolerant collectives and match.all's mask; activemask; try_cancel multicast
and generic addresses; mbarrier report operands; txq; per-warp barrier arrival; 96B swizzle; m8n8k4 `.f16` and the
ctype/dtype check; f64 mma rounding modifiers (also wmma f64); sparse metadata layouts (`ptx-exec-sp.k`,
`ptx-exec-wg.k`); tcgen05 `tcCell`, `tcScalesA`, `tcNib`, descriptor bit 52 (`ptx-exec-tc.k`).

## Next: review the Sail model against the ISA (not against K)

- [ ] Integer and bit instructions (§9.7.1, §9.7.2)
- [ ] Floating point, half precision, conversions (§9.7.3, §9.7.4, §9.7.9 cvt), including every `.approx` error bound
- [ ] Comparison, selection, logic (§9.7.5–§9.7.8)
- [ ] Data movement: ld / st / atom / red, state spaces, generic addressing (§9.7.9, §9.7.13)
- [ ] Control flow and calls (§9.7.13)
- [ ] Synchronisation and warp collectives (§9.7.14)
- [ ] Matrix: mma / wmma / ldmatrix (§9.7.15), wgmma (§9.7.16), tcgen05 (§9.7.17)
- [ ] Async copies, mbarrier, TMA, clusters
- [ ] Textures and surfaces (§9.7.10, §9.7.11)
- [ ] Version / target legality table (`ptx_isa_table.sail`) against each section's "PTX ISA Notes" / "Target ISA Notes"

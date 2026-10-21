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

- [ ] ue4m3 scale 0x7F is not NaN in block-scaled `mma` / `tcgen05.mma`, and bit 7 is not masked. §5.2: ue4m3 "NaN value is
      limited to 0x7f", "MSB bit padded with zero".
- [ ] Implement `mma.sync.m8n8k4` with `.f16` / `.f32` accumulators from §9.7.15.5.1 (Figures 46–52).
- [ ] Barrier ids ≥ 16 should be rejected (§9.7.14: 0..15) rather than accepted.
- [ ] `clusterlaunchcontrol.try_cancel … .multicast::cluster::all` writes the response and signals the mbarrier only in
      the issuing CTA. To check against the ISA section on clusterlaunchcontrol.
- [ ] `sail/common/io.c`: addresses wrap at 2^64 (`ad + i`), keys outside [0, 2^64) are dropped; `calloc`/`realloc`
      results unchecked.
- [ ] `docs/prior-work.md` (tcgen05.cp decompression) says 4-bit values land in the low bits; Figure 195 puts them in
      bits 2–5, as the code does. The header of `semantics/ptx-exec-tc.k` lists implemented features as not modelled.
- [ ] `.ashift` with `.sp`: decide the reading (see table) and document it.

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

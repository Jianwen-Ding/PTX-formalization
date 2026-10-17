# Roadmap: full modern PTX in K

Goal: cover the current PTX ISA, excluding long-deprecated legacy features
(e.g. non-`.sync` `shfl`/`vote`, pre-sm_70 warp semantics). Prefer importing
existing trusted formalizations over re-deriving them (task 41).
Every task is done when its tests pass (`make test`, plus `make fuzz` where
relevant), and the work is committed and pushed.

Order: 41 → 1 → {2, 3, 4} → feature work. `(after N)` marks other dependencies.

## 0. Prior work
- [x] 41 Survey and reuse existing trusted formalizations (K semantics, PTX memory-model formalizations, IEEE test vectors) → `docs/prior-work.md`

## 1. Infrastructure
- [x] 1 Modularize semantics for parallel extension
- [x] 2 Complete PTX grammar and type coverage (`::` opcodes, b128/f16x2/bf16/tf32/fp8/fp6/fp4 types, vector decls, prototypes, branch targets, debug sections, ...)
- [x] 3 Warp-collective rendezvous infrastructure (+ `%lanemask_*`)
- [x] 4 Directed-rounding and FTZ float infrastructure (+ exact-rational fuzz reference)

## 2. Integer
- [x] 5 Extended precision: add.cc / addc / sub.cc / subc / mad.cc / madc
- [x] 6 Bit-field and permute: bfe, bfi, bfind, fns, bmsk, szext, prmt
- [x] 7 lop3 and funnel shifts (shf)
- [x] 8 mul24/mad24, sad, dp4a/dp2a, .relu min/max, 16x2 SIMD integer ops
- [x] 9 set, slct, testp
- [x] 10 Video instructions (scalar and SIMD; lower priority)

## 3. Floating point
- [x] 11 f32/f64 rounding modes and modifiers (.rz/.rm/.rp, .ftz, .sat, .relu, .NaN, .xorsign.abs, approx variants)
- [x] 12 f16 / f16x2 / bf16 / bf16x2 arithmetic
- [x] 13 Complete cvt matrix (tf32, .rna, .satfinite, packing, cvt.pack, cvt.rs) (after 12)
- [x] 14 FP8 / FP6 / FP4 / UE8M0 conversions (after 12)

## 4. Warp-level and synchronisation (after 3)
- [x] 15 shfl.sync
- [x] 16 vote.sync, activemask, match.sync, elect.sync, redux.sync
- [x] 17 Barrier variants (bar.warp.sync, bar.red, barrier.cta.*)

## 5. Matrix (after 3, 12)
- [x] 18 wmma
- [ ] 19 mma.sync with specified fragment layouts
- [x] 20 ldmatrix / stmatrix / movmatrix (after 19)
- [x] 21 mma.sp (after 19)
- [x] 22 wgmma, sm_90a (after 19, 27)
- [x] 23 tcgen05, sm_100a (after 22, 27)

## 6. Memory, async and clusters
- [x] 24 Memory-operation qualifiers, cache controls, createpolicy, isspacep
- [x] 25 Extended atomics (f16/bf16/vector/b128, .sem/.scope)
- [x] 26 Thread-block clusters and distributed shared memory
- [x] 27 mbarrier
- [x] 28 cp.async (after 27)
- [x] 29 Bulk async copies (after 26, 27)
- [x] 30 Tensor maps and TMA (after 29)
- [x] 31 st.async, red.async, fence.proxy, griddepcontrol, setmaxnreg, ... (after 27)
- [x] 32 Remaining special registers

## 7. Control flow and resources
- [x] 33 Indirect branches and calls (brx.idx, .callprototype)
- [ ] 34 alloca / stacksave / stackrestore
- [ ] 35 Textures, samplers and surfaces

## 8. Memory model
- [ ] 36 PTX weak memory consistency model (reuse existing formalization; litmus suite)

## 9. Found while importing ZLUDA's conformance programs
- [ ] 42 Vector registers and `.reg`-parameter functions
- [ ] 43 Scope-aware register shadowing

## 10. Quality
- [ ] 37 Version / target legality checking
- [ ] 38 Realistic kernel corpus (after 15, 19, 22, 28, 30)
- [ ] 39 Interpreter performance
- [ ] 40 Coverage documentation against the PTX ISA (after 38)

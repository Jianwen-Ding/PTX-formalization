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
- [x] 19 mma.sync with specified fragment layouts
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
- [x] 34 alloca / stacksave / stackrestore
- [x] 35 Textures, samplers and surfaces

## 8. Memory model
- [x] 36 PTX weak memory consistency model (reuse existing formalization; litmus suite)

## 9. Found while importing ZLUDA's conformance programs
- [x] 42 Vector registers and `.reg`-parameter functions
- [x] 43 Scope-aware register shadowing

## 10. Quality
- [x] 37 Version / target legality checking
- [x] 38 Realistic kernel corpus (after 15, 19, 22, 28, 30)
- [x] 39 Interpreter performance
- [x] 40 Coverage documentation against the PTX ISA (after 38)

# Part 2: remaining differences from the full PTX stack (and the AGX goal)
Items are numbered in task order (44 onward).  Sources: `docs/coverage.md`, `docs/prior-work.md`, the ISA text.

## Instructions
- [x] 44 vmad and istypep (built next; tests `vmad`, `istypep`)
- [x] 45 video: `.mask` destinations, permuting SIMD selectors
- [x] 46 wgmma.sp f16/bf16 k32 (A from smem and registers, metadata, sparsity selector)
- [x] 47 wgmma.sp tf32 k16, e4m3/e5m2 k64, s8/u8 k64
- [x] 48 wgmma audit: b1 (and/xor.popc), scale-d, imm-scale-a/b, imm-trans-a/b, all N, f16 accumulators, fp8->f16
- [x] 49 tcgen05.mma sparse (.sp)
- [x] 50 tcgen05.mma block-scaled kinds (mxf8f6f4, mxf4, mxf4nvf4)
- [x] 51 tcgen05.mma .cta_group::2 (CTA pairs, peer TMEM, operand sharing)
- [x] 52 tcgen05.mma .ws, collector::a modes, ashift; kind::i8/f8f6f4/tf32 audit
- [x] 53 tcgen05.cp (all shapes, decompression), shift, ld/st 16x32bx2 and .red, commit multicast
- [x] 54 mma.sync kind::f8f6f4 / block-scaled mx forms, sm_100 shapes
- [x] 55 ldmatrix/stmatrix sm_100 m16n8 b8/b6/b4 forms
- [x] 56 TMA im2col, im2col::w, w::128 modes
- [x] 57 TMA interleave layouts, OOB fill, L2 promotion, all swizzles audit
- [x] 58 TMA hardware 128-byte tensormap encoding, tensormap.replace all fields, .override
- [x] 59 cp.async.bulk.tensor multicast / cta_group::2 / prefetch.tensor / reduce audit
- [ ] 60 mbarrier .multicast::cluster::32b, cluster-scope arrive/expect_tx, try_wait timeouts
- [x] 61 textures: cubemaps (+arrays), tex.level/.grad, mipmaps
- [x] 62 textures: depth compare, tld4 on a2d/cube/acube, offsets, residency predicate, array layer clamping (sampler border colours other than zero not modelled)
- [x] 63 surfaces: formatted accesses (.p), clamp modes, sured all types, suq
- [ ] 64 multimem.ld_reduce / st / red
- [ ] 65 fabric.*
- [ ] 66 generic-window cluster addresses (cvta.shared::cluster), exited-thread DSMEM semantics

## Memory model and execution model
- [ ] 67 scopes (.cta/.cluster/.gpu/.sys), morally-strong, scope-aware fences and release/acquire
- [ ] 68 load-buffering and other load/store reorderings, causality order (PTX model)
- [ ] 69 proxies (generic/async/texture/surface/tensormap) and fence.proxy.*
- [ ] 70 asynchronous operations as concurrent agents (cp.async, bulk, TMA, wgmma, tcgen05) instead of atomic at issue
- [ ] 71 validate against the official PTX memory model (Alloy/herd litmus catalogue), data-race UB check
- [ ] 72 UB detection: out-of-bounds, misalignment, uninitialised reads, `.aligned` violations, bad masks
- [ ] 73 independent thread scheduling: convergence, activemask under divergence, bar.warp.sync, reconvergence
- [ ] 74 grid level: CTA scheduling fairness, cooperative launch grid sync, streams/multiple kernels, griddepcontrol (PDL), device-side launch

## Language, linking, legality
- [ ] 75 `.address_size 32` mode and 32-bit windows
- [ ] 76 linking: .extern/.weak/.common, .alias, dynamic shared sizing, .managed, .const banks, symbol-address initialisers, .section
- [ ] 77 kernel/function directives as launch constraints (.maxnreg, .maxntid, .reqntid, .minnctapersm, .maxclusterrank, .noreturn), .pragma, .callprototype call-site checks
- [ ] 78 param space: byval structs, .ptr/.align, size limits, cvta on param, ld.param::entry
- [ ] 79 packed/vector types audit (.b128, f16x2, bf16x2, e4m3x2, e2m1x2, ...)
- [ ] 80 float audit: every op x rounding x ftz x sat x NaN against references; model transcendental .approx error bounds
- [ ] 81 run every example in the ISA document as a test (auto-extracted)
- [ ] 82 qualifier-level version/target legality (generated), deprecated-feature errors, .target options
- [ ] 83 opcode sweep: diff every instruction in the ISA document against the rules, implement stragglers (brkpt, pmevent, nanosleep, trap, prefetchu, ...)

## Validation and scale
- [ ] 84 parse corpora: LLVM NVPTX test expectations, CUTLASS/Triton/PyTorch PTX
- [ ] 85 hardware differential-testing harness (needs a GPU)
- [ ] 86 large kernels (CUTLASS-style GEMM, flash-attention) and interpreter scaling; symbolic execution for small kernels
- [ ] 87 formal properties layer (race-free determinism, barrier lemmas), scope TBD

## Apple AGX target (the original goal)
- [ ] 88 survey AGX ISA documentation (Asahi, applegpu, MSL spec) and write the mapping doc
- [ ] 89 formalize the AGX ISA subset in K
- [ ] 90 PTX->AGX lowering for scalar ALU/float/int/memory with translation-validation tests
- [ ] 91 warp(32)<->SIMD-group mapping: shuffles, votes, match, redux, elect, barriers, divergence
- [ ] 92 shared/local memory, generic windows, atomics and memory orders on the Metal model
- [ ] 93 tensor-core emulation (mma/wmma/wgmma/tcgen05 -> simdgroup_matrix) and numerics differences
- [ ] 94 async copy/TMA/mbarrier/cluster emulation strategies
- [ ] 95 texture/surface mapping to Metal and precision differences
- [ ] 96 end-to-end: lower real PTX kernels, run on AGX, compare to the K semantics

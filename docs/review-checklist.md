# Review checklist: the Sail PTX model (`sail/ptx`)

Reference: the public **PTX ISA, Release 9.4**, from the 9.4 migration below on; the earlier entries were checked
against **Release 9.3** (NVIDIA, Jun 25 2026) and cite its section, table, figure, page and line numbers (`Lnnnn`,
`~nnnn`). Messages of the model (`[PTX ISA 9.4 <section>]`) and the legality table use 9.4 section numbers.
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
| `.ashift` with `.sp`: rows of the expanded A shift | §9.7.17.10: ".ashift shifts the rows of the A matrix down by one row, except for the last row in the Tensor Memory", also allowed on `tcgen05.mma.sp`. Whether the compressed data or the logical matrix shifts (and whether the metadata moves) is not stated | decided (9.4 text unchanged, §9.7.18.10.10.2): the model shifts the expanded A, so each row keeps its own metadata; shifting the compressed rows alone would pair every row with the previous row's metadata, which has no sensible reading |
| textures: wrap / mirror with unnormalized coordinates clamp | §5.3.2 defers address modes to the CUDA C++ Programming Guide ("See the CUDA C++ Programming Guide for more details") | not in the PTX ISA; follows the CUDA guide |
| a call whose return list does not match the declaration faults (Sail looped forever) | §9.7.13 call: "Input arguments and return values are optional"; operands are type-checked against flist / fproto | the loop was a defect; that such a call must fault is not stated by the ISA |
| `d\|p` operands reject a literal after the bar | §9.7.14.15 elect.sync: `d` is a register or the sink `_`, `p` a predicate | confirmed |
| thread exit rechecks every barrier with arrivals (Sail checked ids 0–15) | §9.7.14 barrier: "barriers numbered 0..15" | matches K; only reachable with ids the ISA disallows |

## Found, not fixed yet

- [x] ue4m3 scale 0x7F is not NaN in block-scaled `mma` / `tcgen05.mma`, and bit 7 is not masked. §5.2: ue4m3 "NaN value is
      limited to 0x7f", "MSB bit padded with zero". Also: the tcgen05 idesc scale type is the one bit 23 (UE4M3 = 0,
      UE8M0 = 1; bits 24-26 reserved); the model read bits 23-24 and had a bias-15 format the ISA does not define.
      Fixed; `tests/mma-bs-nan.ptx`.
- [x] Implement `mma.sync.m8n8k4` with `.f16` / `.f32` accumulators from §9.7.15.5.1 (Figures 46–52) (tests/mma-m8n8k4-*.ptx).
- [x] Barrier ids ≥ 16 should be rejected (§9.7.14.1: 0..15) rather than accepted. Now an error ("barrier id outside
      0..15"); a thread count that is not a multiple of 32 counts whole warps and is reported under `.ub strict`
      (tests/neg/bar-id.ptx, tests/neg/bar-count-ub.ptx, tests/bar-count-round.ptx).
- [x] `clusterlaunchcontrol.try_cancel … .multicast::cluster::all` writes the response and signals the mbarrier only in
      the issuing CTA. Checked against §9.7.14.18: the model writes addr and signals mbar in every CTA of the cluster
      (tests/clc-multicast.ptx).
- [x] `sail/common/io.c`: `calloc` / `realloc` / `strdup` results are checked (an allocation failure stops the run with
      "out of memory" instead of crashing or losing data).  Kept by design: byte addresses wrap at 2^64 (`ad + i`), as
      64-bit PTX addresses do; keys outside [0, 2^64) read 0 and are not written (the model checks its addresses first).
- [x] `docs/prior-work.md` (tcgen05.cp decompression) said 4-bit values land in the low bits; Figure 195 puts them in
      bits 2–5, as the code does (doc corrected; the K header is obsolete with the K semantics).
- [x] `.ashift` with `.sp`: decided and documented (see table): the expanded A shifts, rows keep their metadata.

## Integer review against the ISA (fixed)

- [x] `bmsk`: the Semantics pseudo-code (§9.7.1, bmsk) on the full 32-bit operands; the model and `scripts/gen_bits.py`
      masked a and b to 8 bits first (`bmsk.clamp 256, 4` gave 0xF, the ISA gives 0).
- [x] `vshl` / `vshr` `.clamp`: "if ( mode == .clamp && tb > 32 ) tb = 32" on the full value (was masked to 8 bits).
- [x] `vset d.dsel, a, b, c`: "d = optMerge( dsel, tmp, c )" (the merge form wrote nothing).
- [x] `lop3.BoolOp _|p`: "The sink symbol '_' may be used in place of the destination operand d" (was unsupported).
- [x] `setp` / `set` / `slct`: "Modifier .ftz applies only to .f32 comparisons" (f64 subnormals were flushed).
- [x] **ISA vs hardware, decided for the ISA:** `sub.cc` "borrow-out written to CC.CF" and `subc` "d = a - (b + CC.CF)"
      (§9.7.2.3, §9.7.2.4): CC.CF = 1 when the subtraction borrows (tests/sub-borrow.ptx). ZLUDA's hardware-recorded
      `subc_cc_s32` reads the opposite polarity (carry of a + ~b + 1), so it is no longer in tests/zluda/EXPECTED_PASS.

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
- [x] f64 `mma` "precision ... identical to that of .f64 precision fused multiply-add" (§9.7.15.5.14): each product is
      now added by an f64 fma rounded with the modifier, starting from c, in k order (the order is the model's choice);
      it rounded the exact sum once. tests/mma-f64-fma.ptx; scripts/gen_mma.py / gen_mma_m8n8k4.py model the chain.
- [ ] open: Figure 187 (`.16x32bx2`; Figure 190 in 9.4, unchanged) draws threads 16-31 at lanes 16-31 of the
      second access while the text ("A warp executing tcgen05{.ld,.st}.16x32bx2 will access 16 lanes of the Tensor
      Memory", 9.4 §9.7.18.2.3.6; the second access is ".16x32b" at taddr+immHalfSplitoff) and the model use lane
      t % 16 for both accesses; M = 64 "Layout F" lane placement is not modelled

## Floating-point review against the ISA (fixed, Sail only)

- [x] `div.approx.f32` with 2^126 < |b| < 2^128: NaN for infinite a, else a sign-preserving zero (~8581)
- [x] `rcp.approx.ftz.f64`, `rsqrt.approx.ftz.f64`: computed on a[63:32], d[31:0] = 0, NaN -> 0x7fffffff00000000
      (~9150-9187, 9403-9437). Choice: the high word is rounded once to nearest even (the ISA names no rounding)
- [x] f64 NaN payloads (~7859): arithmetic returns the first NaN operand, quieted (which NaN and quieting are model
      choices); `abs.f64` passes a NaN through unchanged (~8674); `neg.f64` NaN stays unspecified (~8738)
- [x] cvt float -> int NaN: 0, or 1 << (BitWidth(dst) - 1) for an .f64 source or 64-bit destination (~15168-15170)
- [x] `tanh.approx.f16` / `.bf16` correctly rounded (bounds 2^-10.987 / 2^-8 at ~10848 were exceeded); checked on all
      65536 inputs of each type
- [x] three-input `min` / `max` (~8773-8975); two-operand `cvt.pack.sat.{u16,s16}.s32` (~15506)
- [x] `.relu` / `.satfinite` on fp8/fp6/fp4/s2f6 up-conversions; ue8m0x2 `.satfinite` of |x| > MAX_NORM is 0xfe (~15249-15257)
- [x] `cvt.rs` stochastic rounding (~14809, Figures 38-41). Choices where the text is ambiguous: with more discarded bits
      than random bits, the random bits are added at the most significant discarded bits; the x4 forms' rbits split
- [x] `atom` / `red` `.add.f32` on global memory and `cp.reduce.async.bulk.add.f32` flush subnormals (~20917, 16478)
- [x] test generators `gen_cvt.py`, `gen_mini.py`, `gen_testfloat.py` follow the same rules

## Memory, calls and launch review against the ISA (fixed, Sail only)

- [x] declarations: unsized first dimension from the initializer (~2675); `.v2` / `.v4` variables sized and aligned to the
      whole vector (~2650); incomplete nested initializers padded per sub-array (~2693)
- [x] address windows: `.const` window, `.param` window inside `.global` for `isspacep` (~3963-3966, 14661-14664);
      windows under `.address_size 32`; a missing `.address_size` means 32 (~39194)
- [x] `alloca` / `stacksave` / `stackrestore` return `.local` addresses (~35770-35960). Choice: the stack top (16 MiB)
- [x] `mov` of a device-function `.param` address copies the parameter to the stack (`.local`) (~1812, 2128, 12608)
- [x] `.param` variables as arguments of `.reg` formals and `.reg` returns into `.param` variables (~4307-4330)
- [x] `.maxntid` bounds the total thread count (~39897); `%dynamic_smem_size`, `%total_smem_size`, `%aggr_smem_size`
      (choices: bytes rounded up to the allocation unit; module-wide static `.shared`); `%is_explicit_cluster`
- [x] `generic(v)+off` and `mask()` initializers (~2743-2760); `mov r, A[k]` (choice: k is a byte offset, constant
      only); an omitted unsized last argument (~39381)
- [x] register-indexed operands such as `ld [array[r1]]` (tests/array-regindex.ptx); `tests/launch-ok.ptx` combined
      `.maxntid` with `.reqntid`, which the ISA forbids (11.4.3): split into two kernels, and `.isacheck` reports the
      combination, and `.maxclusterrank` with `.reqnctapercluster` (11.7.3) (tests/neg/isa-reqntid-maxntid.ptx)

## Second review pass (fixed, Sail only)

Syntax and operands:
- [x] constant expressions (4.5.4 - 4.5.6) in operands, address offsets (`[%rd - 8]`), initializers and array sizes, with
      the .s64 / .u64 typing rules; `& | ^` keep the converted type as the text says (Table 5 says .u64: the same bits).
      Loose: 0f / 0d literals in an expression stand for their value
- [x] octal literals and the U suffix (4.5.1); non-zero integer constants are True as predicates (4.5.3)
- [x] colour fields `.r .g .b .a` of vector registers (6.4.2)
- [x] array elements without brackets as memory operands (`ld.global.u32 s, a[0]`, 6.4.2); index = byte offset (choice)
- [x] `.pragma` string lists and `.pragma` between a kernel header and its body; `.func .attribute(...)`
- [x] an undeclared register is an error (it read as 0); writing a special register is an error (chapter 10)
- [x] loose: `.pred %p;` without `.reg` declares a register (it was a .global variable named %p)
- [x] the test generators `gen_cvt.py` / `gen_mma.py` declared too few registers; regenerated

Instructions and resources:
- [x] `wmma.load` / `wmma.store` without stride: the leading dimension (9.7.15.3)
- [x] tcgen05 `.pack::16b` / `.unpack::16b` cells for the 16-lane shapes (Figure 193)
- [x] `tcgen05.alloc` blocks until the columns are free (9.7.17.5); a never-satisfiable alloc ends in deadlock
- [x] matrix descriptor base offset (bits 49-51) for wgmma and tcgen05 (9.7.16.5.1, Table 45)
- [x] per-kernel layout and naming of a kernel's own `.shared` variables (5.1.5): no 2^18 wrap of descriptor addresses in
      multi-kernel modules, no clash of equal names, per-kernel `%total_smem_size`
- [x] the reserved shared-memory region (10.29) placed after the user's shared memory (it overlapped offset 0)
- [x] `isspacep.shared::cluster` only for ranks of the executing cluster
- [x] textures / TMA / bulk / multimem (merged from the agent branch): 2dms / a2dms, lod type, f16 destinations,
      force_unnormalized_coords, txq enums, TMA element strides, generic addressing of st.async / red.async / st.bulk,
      op x type tables reported under `.isacheck`
- [x] `.isacheck` legality data generated from the ISA 9.3 text, exact / family target rules, `compute_xx` targets

Open:
- [x] `sub.cc` / `subc` borrow vs carry: the model follows the ISA text (borrow)
- [x] register-indexed array operands of ld / st / atom / red (`a[%r1]`, `a[%r1+4]`); in `mov` still unsupported (error)
- [x] qualifier rules under `.isacheck` (default mode accepts): ld / st `.weak` `.volatile` `.relaxed` `.acquire` / `.release`
      exclusive, no cache ops with them, scope required, `.mmio` only `.relaxed.sys`; `fma` needs a rounding modifier;
      `setmaxnreg` 24 .. 256 step 8; `pmevent` 0 .. 15; `alloca` alignment a power of 2 (tests/neg/isa-rule-*.ptx)
- [x] the operand type rules (§9.4, Tables 26-28) under `.isacheck`: register operands of the same-type instructions
      (add, sub, mul, mad, min, max, abs, neg, div, rem, logic, fma, transcendentals, copysign, mov, setp, selp) must
      agree with the instruction type (bit-size types with any same-size type, signed / unsigned with each other, floats
      exactly); the data operand of ld / st and both cvt operands follow the relaxed Tables 27 / 28 (wider registers,
      no float with an integer type). Checked in the front end, which knows declared register types; no false positive
      over tests/ and tests/zluda (tests/neg/isa-ty-*.ptx, tests/isa-ty-relaxed.ptx). tests/sm1x-ftz.ptx used
      `selp.u32` into an `.f32` register and was corrected. Also the `.u32` shift amount of shl / shr, the doubled
      d (and c) of mul.wide / mad.wide, and the `.u32` result of popc / clz (tests/neg/isa-ty-shamt.ptx, isa-ty-wide.ptx).
- [x] hard errors audited (loose syntax, solid semantics): `mma.m8n8k4` `.dtype .f16` with `.ctype .f32` now runs (f32
      accumulation rounded to f16, checked against the exact model) and is reported under `.isacheck`; a call without a
      return list discards the results; an undefined `wgmma` shape stops with an error instead of deadlocking.  Kept as
      errors (no meaning to give): undeclared registers, special-register writes, `d|literal`, partial return lists,
      tcgen05 descriptor bit 52 and `cta_group::2` disable-output-lane (layouts not specified), inc / dec on floats
- [x] tcgen05 allocation patterns (alloc after relinquish, growing size, exit without dealloc) under `.ub strict`
      (§9.7.17.7.1; tests/neg/tc-ub-relinq.ptx, tc-ub-grow.ptx, tc-ub-leak.ptx)
- [x] sm_1x single-precision default: on an sm_1x target every `.f32` instruction that takes `.ftz` flushes subnormal
      inputs and results (§9.7.3); `.f64` keeps them (tests/sm1x-ftz.ptx)
- [x] sm_1x `mad.f32` mantissa truncation of the product (§9.7.3.5): without a rounding modifier on an sm_1x target the
      exact product is truncated to 24 significant bits, then c is added with .rn; with c = ±0 it is mul then add
      (tests/sm1x-mad.ptx)
- [x] `brkpt` "Suspends execution" (§9.7.20.1): the run stops there (it was a no-op) (tests/neg/brkpt.ptx)
- [x] function-scope variables (`.shared`, `.global`, `.const`) are bound per function: equal names in two functions no
      longer share a symbol (tests/func-scope-vars.ptx)
- [x] `.ashift` with `.sp` decided (see table); Figure 187 lane placement stays open (see the matrix section)
- [x] `tcgen05.alloc` without `.shared::cta` takes a generic address (§9.7.17.7.1: "If no state space is specified then
      Generic Addressing is used"); the model read it as a shared offset (tests/tc-alloc-generic.ptx)
- [x] `.multimem` fabric operations with counted completion bump the counter on every destination device, each counting
      the accesses at that device (§9.7.10.4); the model bumped only the first device's (tests/fabric-mm-counted.ptx)
- [x] `fabric.try_atom` and the `.tensor` fabric forms are not in the 9.3 text: they run in the default mode, `.isacheck`
      reported them (superseded by the 9.4 migration: they are 9.4 forms, checked by the table rows)
- [x] packed `.f16x2` / `.bf16x2` `setp` / `set`: `.ftz` flushes lane inputs; the `.BoolOp` form gives
      p = BoolOp(t[0], c), q = BoolOp(t[1], c) and d[i] = BoolOp(t[i], c); a sink `_` in either destination keeps the
      packed meaning (§9.7.7.1, §9.7.7.2). The model compared the whole word as an integer, ignored `.ftz`, and sent
      `_|q` to the scalar path (tests/half-cmp-pk.ptx, half-cmp-pk2.ptx, half-misc.ptx)
- [x] `fma.oob` gives +0 when an operand is the OOB-NaN (§9.7.4.4). The ISA gives no OOB-NaN bits; the model uses the
      pattern its own tensor copies fill with (f16 0x7E00, bf16 0x7FC0) (tests/half-oob-set.ptx)
- [x] `.sat` of −0 returns −0: −0.0 lies in [0.0, 1.0], and the text only forces +0 for NaN results; kept per the ISA
- [x] `.isacheck` reports half-precision forms outside the §9.7.4 syntax: rounding other than `.rn`, `.ftz` / `.sat` on
      `.bf16`, `.relu` with `.sat`, `.oob` with `.ftz` / `.sat`, `ex2.approx` with / without `.ftz` on the wrong type
      (tests/neg/isa-half-*.ptx)
- [x] `testp.normal` is true for ±0 ("As a special case, positive and negative zero are considered normal numbers",
      §9.7.3.1); scripts/gen_pred.py had the same mistake (tests/pred-4.ptx corrected)
- [x] `.rp` / `.rm` to the small float formats (e4m3, e5m2, e2m3, e3m2, e2m1, s2f6) round the signed value toward
      ±infinity; the model rounded the magnitude up (`.rp`) or to nearest (`.rm`). The ISA allows only `.rn` (`.rs` for
      x4) there, so `.isacheck` reports these forms (tests/cvt-mini-dir.ptx, tests/neg/isa-cvt-mini-rnd.ptx)
- [x] `cvt.rn.satfinite.ue8m0x2.f32` is listed at isa.txt 15298 while the syntax has `.frnd3 = { .rz, .rp }` (14836) and
      15302 lists `{rz/rp}` for the same form: the syntax is taken as authoritative, `.isacheck` reports `.rn`, and the
      default mode rounds it to the nearest power of two, ties to the even code (tests/cvt-ue8m0-rn.ptx)
- [x] deprecated non-`.sync` `vote` / `shfl` (targets below sm_70): the same exchange among the active lanes, found as
      for `activemask` (tests/warp-nosync.ptx)
- [x] `elect.sync` with exited lanes in the membermask waits for them: "wait until all threads in the membermask execute
      the elect instruction" (§9.7.13 elect, isa.txt 22011) has no "non-exited" exception, unlike vote / match / redux /
      shfl, so the model follows the text and reports the resulting deadlock
- [x] `discard` (§9.7.9.18) "behaves like a weak write of an unstable indeterminate value": the model keeps the old
      bytes (one such value); an address outside the `.global` window or not 128-byte aligned is reported under
      `.ub strict`, a size other than 128 under `.isacheck`; the same for `applypriority` (§9.7.9.17)
      (tests/neg/discard-*.ptx, applypriority-align.ptx, isa-discard-size.ptx)
- [x] `%aggr_smem_size` is "the size of user shared memory allocated ... and the size of shared memory region which is
      reserved" (§10.31): the allocated user size (as `%total_smem_size`, in allocation units) plus the reserved region;
      the model added the unrounded byte count (tests/smem-sregs.ptx)
- [x] `%reserved_smem_offset_cap` is the "Total size of the reserved shared memory region" (§10.29): 1024 (it held the end
      offset); a user shared access into the reserved region is reported under `.ub strict` ("accessing this region from
      user code results in undefined behavior"), and so is an access past the CTA's shared memory
      (tests/smem-reserved-cap.ptx, tests/neg/smem-reserved-access.ptx, tests/neg/smem-oob.ptx)
- [x] ld / st `.v8` only with 32-bit types, and `.v8` / `.v4` with 64-bit types only with `.global` or generic
      addressing (§9.7.9.8, §9.7.9.11): reported under `.isacheck` (tests/neg/isa-ld-v8-*.ptx); ld.global.nc with sink
      elements (tests/ld-nc-vec-sink.ptx)
- [x] mbarrier (§9.7.14.16) review:
      - `try_wait` resumes "following a system-dependent time limit": when no thread can otherwise progress, suspended
        try_waits return False (it deadlocked) (tests/mbar-trywait-timeout.ptx)
      - `cp.async.mbarrier.arrive.noinc` without a state space uses generic addressing (tests/mbar-cpasync-noinc-generic.ptx)
      - an operation on an object that is not initialised (or after inval), or at an address outside the shared windows,
        stops with one message for every operation (it fell through to "unsupported instruction" or deadlocked)
      - `.ub strict`: init on a valid object, init count outside Table 37, tx-count outside ±(2^20 - 1), an arrive count
        of 0 or beyond the pending count, a `.noComplete` arrive that completes the phase, test_wait with a state older
        than the preceding phase, an address not 8-byte aligned (tests/neg/mbar-ub-*.ptx)
      - `.isacheck`: .sem / .scope together, .scope only .cta / .cluster, arrive not .acquire, waits not .release,
        expect_tx / complete_tx only .relaxed, waits only on .shared::cta, noComplete only .release.cta, check_layout needs
        .layout, init count and parity immediates in range, the `.multicast::cluster::32b` forms (not in 9.3; 9.4 rows since)
        (tests/neg/isa-mbar-*.ptx)
- [x] mbarrier: a shared ld / st / atom on an initialised object is reported under `.ub strict` (tests/neg/mbar-ub-st.ptx)
- [x] mbarrier, `.ub strict`: an arrive-on before a wait returned True for the previous phase (§9.7.14.16.5.1), and
      cp.async.mbarrier.arrive pushing the pending count past its limit (§9.7.14.16.18) (tests/neg/mbar-ub-nowait.ptx,
      mbar-ub-cpasync-limit.ptx)
- [ ] mbarrier, not reported: reading the v1 report operands when waitComplete is False;
      whether expect_tx alone may complete a phase (22188-22192 vs 22227-22231) is left as is (it can)
- [x] `setmaxnreg` (§9.7.20.5): `.dec` above / `.inc` below the warp's current count is reported under `.ub strict`
      once the count is known (after the warp's first setmaxnreg) (tests/neg/setmaxnreg-ub.ptx); not checked: a
      setmaxnreg not executed by every warp of the warpgroup, the kernel's initial count
- [x] `st.async` / `red.async` mbarrier forms (§9.7.9.12, §9.7.14.7) under `.ub strict`: a and mbar in the
      `.shared::cluster` window and in one CTA; st.async needs `%cluster_nctarank > 1`; red.async's CTA differs from the
      executing one (tests/neg/red-async-own.ptx, tests/misc-async-ub.ptx)
- [x] `st.bulk` (§9.7.9.14) states no alignment for a but was written as one size-byte access, which `.ub strict` took
      for a 16-byte naturally aligned access; it now writes 8-byte pieces (or bytes); size not a multiple of 8 and an
      address outside `.shared` are reported under `.ub strict` (tests/neg/st-bulk-size.ptx)
- [x] `cp.async.bulk` / `cp.reduce.async.bulk` (§9.7.9.26.4.1-2) under `.ub strict`: size not a multiple of 16, src / dst
      not 16-byte aligned, a `.shared::cta` → `.shared::cluster` copy into the executing CTA (tests/neg/bulk-ub-*.ptx);
      the existing bulk and TMA tests pass under `.ub strict`
- [x] a plain ld / st / atom on a multimem address is reported under `.ub strict` (§9.7.9.15) (tests/neg/multimem-plain-ld.ptx);
      multimem.* on a non-multimem address already stops
- [x] fabric try_get / try_put / try_red (§9.7.10.5.1-3, §9.7.10.4.3) under `.ub strict`: size a multiple of 16, 16-byte
      aligned data offset and shared pointer, a `.layout::v1` mbarrier, a unicast endpoint (multicast with `.multimem`),
      a 256-byte aligned counter not overlapping the data (tests/neg/fabric-ub-unicast.ptx); tests/fabric-mm-counted.ptx
      initialised its mbarrier without `.layout::v1` and was corrected
- [x] sparse `mma.sp` (§9.7.15.6.1) under `.ub strict`: a sparsity selector outside the shape's values ({0..3}, {0, 1}
      or 0) and metadata index values that are not meaningful (tf32: 0b1110 / 0b0100; ordered_metadata and the f8f6f4 /
      mx kinds: 0b0100 0b1000 0b1001 0b1100 0b1101 0b1110; otherwise not 0b0000 0b0101 0b1010 0b1111)
      (tests/neg/mma-sp-ub-*.ptx); the generated sp / sp-bs / wgsp / tcsp tests pass under `.ub strict`
- [x] wgmma ordering (§9.7.16.7.1, §9.7.16.7.3) under `.ub strict`: a `wgmma.mma_async` before any `wgmma.fence`, and an
      instruction other than wgmma naming an accumulator or A register of an mma_async before a `wait_group` covers its
      group (wait_group N leaves the N most recent groups pending) (tests/neg/wgmma-ub-*.ptx); the 37 generated wgmma
      tests pass under `.ub strict`.  Also the fence required between another register access and a later mma_async
      on the same register (tests/neg/wgmma-ub-fence-between.ptx), and accumulators shared by mma_async instructions of
      different shapes without a fence (the default ordering covers only the same shape) (tests/neg/wgmma-ub-shape.ptx)
- [x] wmma.load / wmma.store (§9.7.15.4.2-4) under `.ub strict`: a stride below the shape's default, and an address or
      stride (in bytes) that is not a multiple of the fragment size (tests/neg/wmma-ub-*.ptx); the existing wmma
      programs pass under `.ub strict`
- [x] under `.ub strict`: `cvta.to.<space>` of a generic address outside that window (§9.7.9.21), `cp.async` with
      src-size larger than cp-size, and two `cp.async` of one cp.async-group copying to the same location
      (§9.7.9.26.3.1) (tests/neg/cvta-ub-window.ptx, cpasync-ub-*.ptx, tests/cpasync-groups-ub.ptx)
- [x] `ld.param::entry` naming something other than a parameter of the executing kernel, and `ld.param::func` naming a
      kernel parameter (§9.7.9.8), under `.ub strict` for named addresses (tests/neg/ld-param-entry-ub.ptx); an access
      after `stackrestore` to memory it freed (§9.7.18.3) was already reported ("local access above the stack top")
- [x] `tcgen05.mma.sp` metadata (§9.7.17.10.8) under `.ub strict`: kind::tf32 0b1110 / 0b0100; the 2:4 and pair-wise
      4:8 kinds 0b0100 0b1000 0b1100 0b1001 0b1101 0b0110 0b1110 (this set includes 0b0110, which mma.sp's
      ordered_metadata set does not) (tests/neg/tcsp-ub-meta.ptx); mma.sp::ordered_metadata's "sorted in an
      increasing order" is the meaningful-value set already checked
- [x] tcgen05.ld / st "All the threads in the warp must specify the same value of taddr" (§9.7.17.8.3-4) under
      `.ub strict`: lanes of one instance (function, pc, per-thread visit count) compare their taddr
      (tests/neg/tcgen05-ub-taddr.ptx); the 35 tcgen05 ld / st tests pass under `.ub strict`
- [x] wmma.load / wmma.store "the same values of p and stride" in every thread (§9.7.15.4.3-4) under `.ub strict`, with
      one warp-uniform-operand helper (warpUniform, ptx_coll.sail) shared with tcgen05.ld / st (tests/neg/wmma-ub-uniform.ptx)
- [x] tcgen05: one `.cta_group` value for all tcgen05 instructions of a kernel (§9.7.17.7.1, checked per CTA) under
      `.isacheck` (tests/neg/isa-tc-ctagroup.ptx); `tcgen05.alloc` nCols warp-uniform under `.ub strict`
      (tests/neg/tc-alloc-ub-ncols.ptx).  The tcgen05 tests now conform under `.isacheck` and `.ub strict`:
      scripts/gen_tcgen05.py emits `.version 8.8`, tests/tc-red.ptx targets sm_103a (tcgen05.ld.red is not on sm_100a),
      tests/tc-alloc-generic.ptx deallocates its tensor memory
- [x] sweep: every test without its own `.isacheck` / `.ub` run with both.  No model false positive was found; the reports
      are the tests' own: an older `.version` / `.target` than the forms need (mostly generated mma.sp / wgmma.sp / mma
      tests), `sm_107f` (not in 9.3), and tests of forms outside the syntax on purpose.  tests/tex.ptx and
      tests/surf.ptx did an 8-byte suld at a 4-byte aligned x ("must be naturally aligned to a multiple of the access
      size", §9.7.12.1); scripts/gen_tex.py and gen_surf.py now use x = 8
- [x] ZLUDA programs under `.isacheck` + `.ub strict`: no model false positive.  Harness fixes (scripts/import_zluda.py):
      `.extern .shared` programs get 128 bytes of dynamic shared memory (they had none), the 64-thread 2x mma test a
      2048-byte output buffer.  Reports kept as the programs' own: ldmatrix stores past its 256-byte shared array
      (threads 32-63 at +128), vector8_extract uses `ld.v8.u8` (not a 9.3 form), vote_ballot_nosync non-`.sync` vote on
      sm_70+
- [x] a vector ld / st is one access for the alignment rule ("the access size of ld.v4.b32 is 16 bytes ... must be
      naturally aligned to a multiple of the access size", §6.4.1); the model only checked each element.  Now under
      `.ub strict` (tests/neg/vec-ub-align.ptx).  Seven tests stored vectors at offsets that are not multiples of the
      vector size (or loaded from a 4-byte aligned array); they now use aligned pieces or `.align 8` with unchanged
      results (alloca-local, carry, const-expr, decl-init-shape, ld-nc-vec-sink, shared-per-kernel, sub-borrow);
      no ZLUDA program is affected.  The same for vector atom / red (§9.7.14.5-6) (tests/atom-vec.ptx,
      tests/neg/atom-vec-ub-align.ptx)
- [x] ldmatrix / stmatrix with a generic address outside `.shared` (§9.7.15.5.15-16) under `.ub strict`
      (tests/neg/ldmatrix-ub-global.ptx); the 23 ldmatrix / stmatrix programs pass, the ZLUDA ldmatrix program keeps its
      own out-of-bounds store
- [x] address-window rules ("does not fall within the address window of ... the behavior is undefined") under
      `.ub strict` with one helper (winUb): tcgen05.alloc dst (.shared::cta), tcgen05.commit mbar (.shared::cluster),
      clusterlaunchcontrol.try_cancel addr / mbar (.shared::cta), tensormap.replace addr (.global or .shared::cta)
      (tests/neg/tc-alloc-ub-window.ptx, clc-ub-window.ptx); with the earlier ones (mbarrier, st / red.async, st.bulk,
      discard, applypriority, cvta, ldmatrix / stmatrix, multimem) every window rule of §9.7 is checked except
      createpolicy's (an opaque policy with no memory access in the model) and cp.async's prefetch hint
- [x] operand ranges under `.isacheck`: createpolicy fraction in (0.0, 1.0] (§9.7.9.19), tensormap.replace ord in
      [0..4] (§9.7.9.27), tcgen05.mma scale-input-d in [0, 15] and only with .kind::tf32 / .kind::f16
      (§9.7.17.10.9.1) (tests/neg/isa-createpolicy-fraction.ptx, isa-tensormap-ord.ptx); the existing uses pass
- [x] tensor-copy restrictions (§9.7.9.26.5.1) under `.ub strict`: OOB-NaN fill with sub-byte types; for .b4x16_p64 /
      .b6x16_p32 the first coordinate a multiple of 128, a 32B aligned global address and strides; 96B swizzle with 16B
      atomicity only, Box-Size[0] <= 96B, no padded sub-byte types, no .im2col::w::128; 128B swizzle with 32B + 8B flip
      only for copies into shared memory and not with .im2col::w modes; non-negative starting coordinates for
      .global.shared::cta (tests/neg/tma-ub-*.ptx).  tests/tma-rep-sub14 / sub15 had 16 / 24-byte strides; their rows
      are now 32 bytes apart (scripts/gen_tma_rep.py)
- [ ] open: ".global.shared::cta ... The bounding box along the D, W and H dimensions must stay within the tensor
      boundaries" names the im2col dimensions; tile-mode stores past the boundary keep their clipping (tests/tma.ptx);
      "Box-Size[0] must be exactly 64B / 96B" for the padded types does not say in which layout the bytes are counted
- [x] mandatory `.sync` / `.aligned` qualifiers (ldmatrix, stmatrix, movmatrix, mma, wmma (.aligned from 6.3), wgmma,
      setmaxnreg, tensormap.cp_fenceproxy, tcgen05 alloc / dealloc / relinquish / ld / st / wait, elect) under
      `.isacheck` (tests/neg/isa-ldmatrix-sync.ptx); no report over tests/ and ZLUDA
- [x] declarations / directives review (chapters 5-7, 11): parameterized names in every state space (`.global .u32
      g<3>` declared one variable `g`; §5.4.6) (tests/decl-param-names.ptx); a one-element call argument list holding a
      constant, `(21)`, was read as a constant expression (§9.7.13.5) (tests/call-const-arg.ptx); `.branchtargets N<3>`
      label ranges (§11.3.1) (tests/brx-label-range.ptx); a `.noreturn` function that returns, under `.ub strict`
      (§11.4.6) (tests/neg/noreturn-ub.ptx)
- [x] declarations, now reported under `.isacheck` by the front end (9.4 §5.4.1-5.4.6): a declaration without a state
      space, `.pred` outside `.reg`, `.pred` vectors, `.v8` variables, vectors over 128 bits, `.align` not a power of
      two, an initializer on a parameterized name, outside `.global` / `.const`, on `.extern`, on `.f16` / `.f16x2` /
      `.pred`, a label or a non-`.global` / `.const` variable in an initializer, an address held in a type other than
      `.u8` / `.u32` / `.u64`, in a `.u8` without `mask()`, and a `mask()` value outside 0xFF << 8k.  All of them
      still run in the default mode.  tests/cluster.ptx, cluster-mapa-generic.ptx and int-spec.ptx declared `.pred`
      without `.reg`; fixed (tests/neg/isa-decl-*.ptx)
- [x] a label as a `mov` operand (§9.7.10.3) and a `0f` literal in a constant expression (§4.5.2: such values "may not be
      used in constant expressions") are reported under `.isacheck` (tests/neg/isa-mov-label.ptx, isa-cexpr-0f.ptx)
- [ ] `tcgen05.cp … .warpx2::02_13` / `::01_23`: which warp pairs share a 32-lane slice is not spelled out beyond the
      qualifier name; the model's pairing follows the name and is unconfirmed

## Where the Sail model now differs from the K semantics

The K semantics in `semantics/` is no longer kept in step for the fixes above (it cannot be built here). Differences:
cp.async operand parsing; redux `.abs`; exit-tolerant collectives and match.all's mask; activemask; try_cancel multicast
and generic addresses; mbarrier report operands; txq; per-warp barrier arrival; 96B swizzle; m8n8k4 `.f16` and the
ctype/dtype check; f64 mma rounding modifiers (also wmma f64); sparse metadata layouts (`ptx-exec-sp.k`,
`ptx-exec-wg.k`); tcgen05 `tcCell`, `tcScalesA`, `tcNib`, descriptor bit 52 (`ptx-exec-tc.k`); every floating-point
fix above (and the test expectations that changed with them: testfloat f64 NaNs, cvt NaN -> int, mini-1 ue8m0,
transc-edge tanh); declarations, address map, `isspacep`, alloca, parameter passing, launch checks and the
shared-memory registers (`ptx-load.k`, `ptx-configuration.k`, `ptx-exec-pred.k`, `ptx-exec-misc.k`,
`ptx-exec-alu.k`, `ptx-exec-ctrl.k`, `ptx-harness.k`).  `tests/*.out` files are therefore no longer K output.

## Next: review the Sail model against the ISA (not against K)

- [x] Integer and bit instructions (§9.7.1, §9.7.2)
- [x] Floating point, half precision, conversions (§9.7.3, §9.7.4, §9.7.9 cvt), including every `.approx` error bound
- [x] Comparison, selection, logic (§9.7.5–§9.7.8)
- [x] Data movement: ld / st / atom / red, state spaces, generic addressing (§9.7.9, §9.7.13)
- [x] Control flow and calls (§9.7.13)
- [x] Synchronisation and warp collectives (§9.7.14)
- [x] Matrix: mma / wmma / ldmatrix (§9.7.15), wgmma (§9.7.16), tcgen05 (§9.7.17)
- [x] Async copies, mbarrier, TMA, clusters
- [x] Textures and surfaces (§9.7.10, §9.7.11)
- [x] Version / target legality table (`ptx_isa_table.sail`) against each section's "PTX ISA Notes" / "Target ISA Notes"

Each area above has had at least one review pass against the 9.3 text (sections "… review against the ISA" and "Second
review pass"); the third pass (half precision and compares, warp and miscellaneous, conversions, fabric / tcgen05)
is recorded in the list under "Second review pass".  The legality table is generated from the 9.3 text by
`sail/ptx/gen_isa_table.py`.

## Migration to PTX ISA 9.4

- [x] `ptx_isa_table.sail` regenerated from the 9.4 text (`gen_isa_table.py`: 9.4 section numbers for the OVERRIDES /
      GUARDS keys, the 9.4 page footer, the new whole-section forms "Supported on the following family-specific
      architectures:" and "Requires sm_Nf or higher in the same family", 0 unhandled statements).  New rows for the 9.4
      additions: mixed-precision 3-type add / sub / fma and mul (§9.7.5, 9.4 / sm_107f), alternate-format x4 add / sub /
      mul / fma (§9.7.6, 9.4 / sm_100a, sm_103a), set with .u8x4 / .s8x4 / .u16x2 / .s16x2 (§9.7.7.1, sm_107f),
      applypriority.async.bulk(.tensor), cvt .rz to f8 / f6 / f4 x2 and .ue5m3x2, .noftz with .f32 (cp.reduce.async.bulk,
      multimem.cp.reduce.async.bulk, atom, red), fabric.*.async.tensor (sm_107f), fence .alias with .acquire / .release,
      tcgen05.mma .decompress::lut::b and .collector::b::* (sm_107f), sm_107f added to the existing family lists
      (e.g. add.s8x4).  sm_107 is in the sm_10x family (9.4 Table 76), which the family rule already gives.
- [x] `isaQual` (9.4 §9.7.10.24 cvt syntax): .frnd2 = {.rn, .rz} for the f8 / f6 / f4 / s2f6 x2 destinations (9.3
      allowed only .rn), the x4 destinations only .rs, .ue8m0x2 .frnd3 = {.rz, .rp}, .ue5m3x2 .frnd4 = {.rn, .rz, .rp}
      (.frnd2 with .scaled::n1::ue8m0).  The "not in PTX ISA 9.3" rules for fabric .try_atom / .tensor and mbarrier
      .multicast::cluster::32b are gone (9.4 defines them; the table rows check version and target):
      tests/neg/isa-fabric-atom-sm90.ptx, tests/neg/isa-mbar-multicast.ptx
- [x] message citations `[PTX ISA 9.3 <section>]` renumbered to the 9.4 sections (the 9.3 -> 9.4 section map, checked
      by title); comments keep their 9.3 section / line citations.  Fixed on the way: the cvt rounding messages cited
      9.7.9.21 (cvta in 9.3) instead of cvt
- [x] `.isacheck` sweep of tests/*.ptx with the 9.4 table: the new reports are the tests' own (9.4 forms under an older
      `.version`, sm_107f forms on other targets, .rm / .rp / .rn cvt to f8 / ue8m0x2 tested on purpose)
- [x] alternate floating point x4 add / sub / mul / fma (9.4 §9.7.6) were not modelled ("unsupported instruction"): the
      four elements are extracted per the Semantics (f8 / ue8m0 a[8i..8i+7], f6 a[8i..8i+5], .e2m1p4x4 a[8i..8i+3],
      .e2m1x4 a[4i..4i+3] of a .b16), operated on exactly and rounded to nearest even into .e4m3 / .e5m2; .satfinite gives
      the sign-preserved MAX_NORM; without it an .e4m3 overflow is NaN ("converted to NaN") and an .e5m2 overflow
      infinity (IEEE overflow; the ISA states only the .e4m3 case).  NaN results are 0x7f (bits not given by the ISA).
      The front end knows .ue8m0x4 and .e2m1p4x4 (tests/altfp-x4.ptx)
- [x] per-CTA global memory (9.4 §10.34, §10.35, §11.4.10): `%perctamemoryoffset` / `%perctamemorysize` were not
      predefined (front-end error).  A kernel with `.minperctamemory N` gets one zero-initialised slice per CTA of
      max(alignUp(N, 1024), 1024) bytes ("a multiple of 1024 bytes", at least N), 1024-aligned, in linear CTA order;
      without the directive both registers read 0 (the "Driver that doesn't support allocation" value).  `.isacheck`
      reports their use outside an entry with the directive, the directive on a non-entry, and PTX ISA < 9.4 / sm < 80
      (tests/perctamemory.ptx, tests/neg/isa-perctamemory.ptx)
- [x] mixed precision (9.4 §9.7.5) 3-type forms were already modelled (tests/mixed-prec.ptx); the table row for the
      fma form checks four types.  Spec inconsistency: `fma.dtype.atype.btype.ctype` gives `.btype = .f32x2` in the
      syntax but `convert(b[0:15])` in the semantics; the model reads b as a pair of `.atype` (also accepted when
      spelled `.atype`).  `.ftz` on `add.rz.ftz.f16x2.f32x2.f32x2` is not described; the model flushes the f32 inputs
      and the f32 intermediate
- [x] `.layout::v1` payload report and conditional phase (9.4 §9.7.15.16.7, .8, .10) and the `.report_mechanism` of
      cp.async.bulk / cp.async.bulk.tensor (§9.7.10.28.1.3, Table 33) were not modelled (the report was always zero
      and `.phase_type::conditional` waited on the primary phase).  Now: a matching inspection issues a report-on before
      the complete-tx (report predicate set, report value OR'ed with 0, so 0); on phase completion the conditional phase
      advances only with a zero report, the report of the completed phase is kept for the waits, and the next phase's
      report is zero; `.phase_type::primary` waits return the completed phase's report predicate; `.phase_type::conditional`
      waits test the conditional phase's parity.  Model choices where Table 33 is open: `.per_16bytes` samples the first
      element of each aligned 16-byte chunk (the "sampled element" is not specified; for tensor loads, of each aligned
      global chunk holding a read element), `.per_element::ff` matches a 0xff byte (the table's 4-bit "0xF" column is
      not read as a nibble match); sub-byte tensor types are not inspected.  A report mechanism on a `.layout::v0` object
      is reported under `.ub strict` ("otherwise the behavior is undefined"); without it the object has no payload
      report.  Fabric operations never fail in the model, so `.mbarrier::report::fabric` never reports
      (tests/mbar-report-v1.ptx, tests/tma-report.ptx, tests/neg/mbar-ub-report-v0.ptx)
- [x] 9.4 memory hints: `ld.proxy::readonly` outside the global / const window and `prefetch.valid_addr` on an invalid
      location (or `.L1::32B` outside global) are reported under `.ub strict`; `.L1::32B` without `.valid_addr` or with
      `.local` under `.isacheck` (§9.7.10.8, §9.7.10.16).  Not reported: a write, during the kernel, to a location read
      by `ld.proxy::readonly` ("treated as read-only during the lifetime of the kernel")
      (tests/hint94.ptx, tests/neg/ld-readonly-shared.ptx, prefetch-valid-null.ptx, isa-prefetch-l1-32b.ptx)
- [x] already modelled before the migration: ldmatrix `.s8.s4` `.m8n16` (sign extension of the 4-bit elements),
      `.override::global_address` / `.override_attribute` tensor operands, cvt `.pzo` / `.scaled::n1::ue8m0` /
      `.ue5m3x2`, set with packed integers, spcompress / spdecompress
- [x] tcgen05.mma `.kind::ti16` (9.4 §9.7.18.4.2 Table 53, §9.7.18.2.1 Table 50, §9.7.18.10.9): A / B `.s1z4m11`, D `.s32`,
      K = 16 dense / 32 sparse, for mma, mma.sp, mma.ws and mma.ws.sp.  The products are summed exactly and the `.s32`
      result wraps (the tcgen05 text gives only the Saturate bit, "0 (NA)" for ti16; the mma / wgmma wording "on overflow
      ... limited to MIN_INT32..MAX_INT32 ... If .satfinite is not specified, the accumulated value is wrapped" is
      followed, as for `.kind::i8`).  Negate A / B negate the integers (Table 64: ti16 Yes, i8 No).  The sparse
      metadata uses the `.kind::f16` layout (Figures 287 / 288 name ti16; sparsity selector honoured), as the model
      already did for M = 64 (lanes 16 (r / 16) .. as A and D).  `.isacheck` reports an idesc with dtype != S32,
      atype / btype != S1Z4M11, bit 3 or bit 29 set (the default mode ignores the type fields and gives bit 3 its
      `.kind::i8` saturation meaning); `.ub strict` reports an element outside [-2047, 2047] (tests/tc-94-0..3.ptx,
      tests/neg/tc-94-ti16-ub-range.ptx, tests/neg/isa-tc-94-ti16-idesc.ptx).  Spec gaps: the bit encoding of `.s1z4m11`
      is only "a 16-bit signed integer with the range [-2^11 + 1, 2^11 - 1]"; the model reads two's complement (the
      name suggests sign / 4 zero bits / 11-bit magnitude, which differs for negative values).  Table 53 says ti16
      "No Transpose = 0", Table 64 says transpose "Yes": the bits are honoured and not reported.  The 9.4 syntax of
      `tcgen05.mma.sp ... .kind::ti16.collector_b_usage` lists no `[sp-meta-tmem]` operand (taken as a typo: the model
      needs the metadata operand)
- [x] tcgen05.mma block scaling with UE5M3 scales (9.4 Table 55 bits 23-24: UE4M3 = 0, UE8M0 = 1, UE5M3 = 2; §5.2.3,
      Table 71): `tcScaleTy` reads the 2-bit field for `.kind::mxf4nvf4` (and `.kind::mxf4`, whose only listed value is
      UE8M0: other values reported under `.isacheck`, honoured in the default mode; `.kind::mxf8f6f4` keeps UE8M0 and
      `.isacheck` reports bit 23 = 0); UE5M3 decodes as the model's `.ue5m3` (5 exponent bits, bias 15, 3 mantissa
      bits, NaN 0xff, no infinity).  UE5M3 is valid with every `.scale_vectorsize` of `.kind::mxf4nvf4`.  Value 3 is an
      error.  "requires target architecture sm_107f or higher in the same family, otherwise behavior is undefined" (UE5M3,
      and UE4M3 with `.block32`) under `.ub strict` against the `.target` (tests/tc-94-4.ptx, tc-94-5.ptx,
      tests/neg/tc-94-ue5m3-ub-target.ptx).  Gap: the bias of ue5m3 is not stated in §5.2.3 (15 = 2^(5-1) - 1 as cvt)
- [x] tcgen05.mma `.decompress::lut::b` (9.4 §9.7.18.10.8, §9.7.18.4.1 bit 53, §9.7.18.10.10.1 form 7, Figures 279-282):
      `[b-decompress-metadata]` operand, for `.kind::f8f6f4` and `.kind::mxf8f6f4.block_scale`.  Per B row n the 48 bytes
      of a K = 128 index block are read as a K-major row of 8-bit elements of b-compressed-desc, descriptor bit 53 selects
      the 24-byte chunk (K = 64), index k is bits 3k..3k+2 of the chunk, and B(n, k) is entry j of the 8-byte table of N
      slice n / 8 in TMEM lane n / 8, column metadata + j / 4, byte j % 4 (Figure 279: one lane per 8-column slice, 8
      one-byte entries; Figure 282: the K[0:63] / K[64:127] tables side by side).  Negate B applies.  `.isacheck`: btype
      other than E4M3 ("the only type allowed is E4M3"), Layout B / C / F (M = 64 with cta_group::1, M = 128 with
      cta_group::2), and statically a `.sp` / `.ws` / other kind / other `.scale_vectorsize` form; `.ub strict`: a
      metadata column that is not two-column aligned.  Errors: transposed B ("doesn't support transpose": no layout for
      the indices), K != 64, `.cta_group::2` (tests/tc-94-6..8.ptx, tests/neg/isa-tc-94-lut-btype.ptx,
      tests/neg/isa-tc-94-lut-sp.ptx).  Spec gaps: the bit order of the 3-bit indices in the 24 bytes (the model packs
      LSB first, as its fp6 / fp4 rows), the byte order of the table entries in the two 32-bit columns (entry j at byte
      j % 4 of column j / 4; Figure 279 labels the 8 bytes "1 TMEM col"), which lane holds the tables (rows 0-31 of
      Figures 279 / 282; the address's lane field is not used, as for the scale factors and the sparsity metadata),
      whether the table for K[64:127] is chosen by bit 53 or by the address (the model: by the address only), the
      meaning of K = 32 and of the CTA pair (which CTA's tables serve which N)
- [x] `.collector::b::*` (and `::a::*`, `::bN::*` of `.ws`) have no effect on the result ("reuse is opportunistic ... the
      source memory ... must not be modified while the MMA instruction ... has not completed - regardless of collector
      qualifier permissions", §9.7.18.10).  `.ub strict` tracks one collector state per CTA and buffer: ::fill
      validates, ::lastuse / ::discard (and the default ::discard of a missing `.collector::a` / `::b`, or of `::b0` for
      `.ws` without a collector qualifier) end it, and ::use / ::lastuse without a valid fill ("This requires a previous
      fill to the collector buffer to be still valid") is reported.  `.isacheck` (isaQual): `.collector::a::use` /
      `::fill` with `.ashift` ("It is illegal"), `::a` / `::b` on `.ws`, `::bN` off `.ws`, `::b` with `.kind::i8` (only the
      ti16 integer forms have it) (tests/tc-94-0.ptx, tests/neg/tc-94-coll-ub-use.ptx,
      tests/neg/isa-tc-94-coll-ashift.ptx).  Gaps: the ISA does not say whose buffer it is (CTA, SM, issuing thread,
      CTA pair) or whether the filled matrix must be the one later used; the model checks per CTA and only the fill
- [x] tcgen05.alloc / dealloc `.exclusive` (9.4 §9.7.18.7.1, Table 60) was not modelled (the qualifier was ignored and
      non-power-of-two sizes failed).  An exclusive allocation waits until the CTA has no live allocation, starts at
      column 0, takes any multiple of 32 up to 512 (576 on sm_107), and blocks every other allocation until its
      `dealloc.exclusive`; a dealloc whose `.exclusive` does not match the allocation is reported under `.ub strict`.
      Model scope: tensor memory is per CTA in the model, so "no other CTA may allocate" has no counterpart; non-exclusive
      allocations stay within 512 columns (tests/tc-alloc-exclusive.ptx, tests/neg/tc-ub-exclusive-dealloc.ptx)

## Review of the 9.3 -> 9.4 text changes (section-by-section diff, four reviewers)

- [x] atom / red `.add.noftz.f32` on global memory flushed subnormals (the model flushed every global `.add.f32`):
      9.4 §9.7.15.5 / §9.7.15.6 "The atom.add.f32 operation without .noftz, flushes ..."; also the vector forms
      (tests/atom-noftz-f32.ptx)
- [x] cp.reduce.async.bulk `.add.f32` keeps subnormals in 9.4 ("The default behavior ... is equivalent to specifying
      optional .noftz", §9.7.10.28.4.2; 9.3 flushed) and `.noftz` is legal with `.f32` (it was reported under
      `.isacheck`).  tests/atom-f32-ftz.ptx expected the 9.3 flushing; updated
- [x] mbarrier.arrive_drop `.multicast::cluster::32b` stopped as unsupported; now drops on every CTA of ctaMask
      (tests/mbar-drop-mc.ptx).  New 9.4 UB "If the decrement causes the expected arrivals count to be zero" under
      `.ub strict` (multicast and local)
- [x] test_wait / try_wait `.parity` immediates outside 0 / 1: 9.4 drops "So the valid values of phaseParity operand
      are 0 and 1"; the `.isacheck` report now applies below PTX ISA 9.4 only
- [x] the generated row for fence `.alias` with `.acquire` / `.release` (9.4) no longer matches the 9.3 fabric proxy
      kinds (`generic::fabric` ...), which were reported as needing 9.4
- [x] cvt `.scaled::n1::ue8m0` from `.bf16x2` flushed inputs below 2^-14 (the f16 threshold); now below 2^-126
- [x] the ctaMask of `.multicast::cluster` without `::32b` uses 16 bits ("By default, ctaMask is assumed to be of 16-bit
      width"): cp.async.bulk, cp.async.bulk.tensor, tcgen05.commit
- [x] tensor stores: "For sm_107f, starting co-ordinates ... can be negative for .tile::scatter4 mode" (no `.ub strict`
      report there)
- [x] padded sub-byte tensor types (§9.7.10.28.5.1, 9.4 rules 2, 3, 8-13): `.b4x16_p64` stores, sub-byte
      `.shared::cluster.global` copies on sm_120a, 16B alignment for `.b6x16_p32` / `.b6p2x16` stores on sm_107f (rule 10,
      taken over rule 8), the swizzle modes per type and target, under `.ub strict` (tests/neg/tma-ub-swz-b6.ptx).
      Box-Size[0] / Tensor-Size[0] sizes stay unchecked (the byte layout question above)
- [x] isaTables messages cited 9.3 section numbers under "PTX ISA 9.4"; renumbered
- [x] cvt `.pzo` applied only to the f8 / f6 / f4 x2 destinations; now also `.f16`, `.bf16`, `.tf32`, `.f16x2`,
      `.bf16x2` (also `.rs`) and the x4 `.rs` forms ("if the result of the conversion produces -0.0, it will be forced to
      +0.0", §9.7.10.24); `.pzo` outside a float-to-float down-conversion is reported under `.isacheck`
      (tests/cvt-pzo.ptx, tests/neg/isa-cvt-pzo-up.ptx)
- [x] `.isacheck`: applypriority.async.bulk takes any multiple of 16 as size (the plain applypriority rule "only 128" was
      applied to it); `.ignore_oob` with a report mechanism other than `.mbarrier::report::disabled`; fabric.try_pullred
      imm-membermask other than 0xFFFFFFFF (9.4 makes it a rule, not UB)
- [x] "The state space specified by memory instructions must match the state space of the address operand, otherwise the
      behavior is undefined" (9.4 §6.4.1): ld / ldu / st / atom / red with an explicit state space on a variable of
      another space are reported under `.ub strict` (a `.shared` variable matches `.shared::cluster`; a device-function
      parameter moved to `.local` by taking its address stays a `.param` operand; register addresses are not tracked)
      (tests/neg/ub-space-mismatch.ptx)
- [x] `.nv_intermediate_source_section`: the raw text between `.source_begin` and `.source_end` was lexed as PTX (a `#`
      or an unbalanced brace broke the module); it is now dropped before lexing.  `.loc_intermediate` not directly after
      a `.loc` is reported under `.isacheck` (§11.5.5) (tests/syntax-intermediate-ir.ptx, tests/neg/isa-loc-intermediate.ptx).
      Not checked: that its file index names an intermediate IR file
- [x] `im2col_no_offs::w` (9.4 §5.5.5, §5.5.5.1): an out-of-bounds access "raises a runtime error" after the pixels
      before it (the model silently stopped); under `.ub strict` the Lower-Corner-W >= 0 / Upper-Corner-W <= 0 rule and
      "swizzling mode must not be set to (i) no swizzle" for the three `im2col::w` modes.  scripts/gen_tma_im2col.py's
      store test went out of bounds on purpose; it now stays inside (tests/tma-i2c-store.ptx) and the old program is
      tests/neg/tma-i2c-store-oob.ptx

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
- [x] Implement `mma.sync.m8n8k4` with `.f16` / `.f32` accumulators from §9.7.15.5.1 (Figures 46–52) (tests/mma-m8n8k4-*.ptx).
- [x] Barrier ids ≥ 16 should be rejected (§9.7.14.1: 0..15) rather than accepted. Now an error ("barrier id outside
      0..15"); a thread count that is not a multiple of 32 counts whole warps and is reported under `.ub strict`
      (tests/neg/bar-id.ptx, tests/neg/bar-count-ub.ptx, tests/bar-count-round.ptx).
- [x] `clusterlaunchcontrol.try_cancel … .multicast::cluster::all` writes the response and signals the mbarrier only in
      the issuing CTA. Checked against §9.7.14.18: the model writes addr and signals mbar in every CTA of the cluster
      (tests/clc-multicast.ptx).
- [ ] `sail/common/io.c`: addresses wrap at 2^64 (`ad + i`), keys outside [0, 2^64) are dropped; `calloc`/`realloc`
      results unchecked.
- [x] `docs/prior-work.md` (tcgen05.cp decompression) said 4-bit values land in the low bits; Figure 195 puts them in
      bits 2–5, as the code does (doc corrected; the K header is obsolete with the K semantics).
- [ ] `.ashift` with `.sp`: decide the reading (see table) and document it.

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
- [ ] open: Figure 187 (`.16x32bx2`) draws threads 16-31 at lanes 16-31 while the text and the model use t % 16;
      M = 64 "Layout F" lane placement is not modelled

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
- [ ] `.ashift` with `.sp`, Figure 187 lane placement (see the matrix section)
- [x] `tcgen05.alloc` without `.shared::cta` takes a generic address (§9.7.17.7.1: "If no state space is specified then
      Generic Addressing is used"); the model read it as a shared offset (tests/tc-alloc-generic.ptx)
- [x] `.multimem` fabric operations with counted completion bump the counter on every destination device, each counting
      the accesses at that device (§9.7.10.4); the model bumped only the first device's (tests/fabric-mm-counted.ptx)
- [x] `fabric.try_atom` and the `.tensor` fabric forms are not in the 9.3 text: they run in the default mode, `.isacheck`
      reports them (tests/neg/isa-rule-fabric94.ptx)
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
        .layout, init count and parity immediates in range, the `.multicast::cluster::32b` forms (not in 9.3)
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
- [ ] not checked: tcgen05.ld / st "All the threads in the warp must specify the same value of taddr" (the model runs
      them per thread)
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

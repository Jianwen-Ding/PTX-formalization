# Review checklist: double-checking the PTX formalization

A task list for an independent pass over the K semantics (`semantics/`), its tests and its documentation.
Each item says what to check and how; tick it when the check has been done and any finding is fixed or filed.
Items under "Known findings" were spotted while writing this list and should be resolved first.

## 0. Known findings (from a first read)

- [ ] **Tests without golden files.** `tests/isa-q-ok.ptx`, `tests/isspacep-runtime.ptx` and `tests/packed8.ptx`
      have no `.out`; `scripts/run-tests.sh` reports a missing golden as FAIL, so `make test` (and CI) cannot be
      green. Generate and *review* the goldens (`scripts/run-tests.sh --update <file>`), don't just accept them.
- [ ] **README "Not (yet) modelled" is stale.** It lists `bfe/bfi/prmt`, `addc/subc`, `f16x2/bf16`, tensor cores,
      clusters, async copies and the weak memory model, all of which the instruction table above it, `ROADMAP.md`
      and `docs/coverage.md` say are implemented.
- [ ] **`docs/coverage.md` "Not modelled" contradicts `ROADMAP.md`.** It lists `vmad`, `istypep`, `.mask`
      destinations, `wgmma.mma_async.sp`, TMA im2col / interleave / `.override`, the hardware tensor-map encoding,
      generic-window cluster addresses, proxies and asynchronous agents as missing; ROADMAP items 44–48, 56–58, 66,
      69 and 70 are ticked. Decide which is true for each and fix the other.
- [ ] **README memory-model text is out of date.** "Sequentially consistent", "`membar`/`fence` are no-ops", and
      "every memory instruction is a single atomic step" predate `.memmodel relaxed;`, load buffering (#71) and async
      agents (#70). State which mode each statement applies to.
- [ ] **ROADMAP numbering.** Task 74 is used twice (memory-model validation, grid level); 72 is `[~]`.
- [ ] **`make properties` ignores failures** (`|| true` in the Makefile). Either drop it or document why.
- [ ] **CI coverage.** CI runs only `test`, `fuzz` and `examples`. `negative`, `zluda`, `litmus`, `memmodel`,
      `approx`, `isa-examples`, `test-agx`, `validate-agx` and `sail-test` never run automatically, so their
      claimed results are unverified. Run each locally and record the outcome; add the cheap ones to CI.
- [ ] **Stray artefacts in the repo root.** `kore-match-disjunction.tar.gz` holds only `error.log` and
      `kore-match-disjunction.log`; `sail/ptx/TEST-REPORT.txt` is a committed run log. Remove or move to `docs/`,
      and make sure no claim depends on them.

## 1. Reproduce the claimed results

- [ ] Install K 7.1.337 exactly as CI does; `make build` from a clean tree.
- [ ] `make test`: all `tests/*.ptx` pass; record count (README's "9 passed" sample output is stale).
- [ ] `make negative`: every `tests/neg/*.ptx` fails *for the stated reason* (check the `// expect:` text exists
      in each file; a file without one passes on any failure, including a parse error).
- [ ] `make zluda`: compare against `tests/zluda/EXPECTED_PASS`; list every ZLUDA program not in it and why.
- [ ] `make fuzz` with more seeds than CI (e.g. `SEEDS="1 … 32" FUZZ_N=1024`).
- [ ] `make examples`, `make litmus`, `make memmodel`: reachable outcome sets match the documented ones.
- [ ] `make approx`, `make isa-examples` (needs the ISA HTML), `make test-agx`, `make validate-agx`, `make sail-test`,
      `make difftest`.
- [ ] Re-run with `JOBS=1` to rule out results that depend on the OOM-retry path in `run-tests.sh`.

## 2. Test-oracle independence

The biggest risk is goldens produced by the semantics itself (`--update`) and then trusted.

- [ ] For each `tests/*.out`, classify the oracle: `.expect` with independently computed values, external vectors
      (TestFloat, ZLUDA, hardware), a `scripts/gen_*.py` model, or "self-generated". List the self-generated ones.
- [ ] For tests whose only oracle is a `scripts/gen_*.py` model, review that model against the ISA text — it was
      written alongside the semantics and may share its misreadings (especially `gen_mma*`, `gen_wgmma`,
      `gen_tcgen05_*`, `gen_ldm`, `gen_tma_red`, `gen_tex`).
- [ ] Check that every `.expect` actually constrains the result (not just printed) and that `.print`-only tests have
      reviewed goldens.
- [ ] Mutation check: perturb a handful of rules (rounding mode, sign extension, carry flag, lane mapping) and
      confirm at least one test fails for each.

## 3. Front end and loading (`ptx-syntax.k`, `ptx-common.k`, `ptx-load.k`)

- [ ] `decodeOpc` splits every opcode form in the ISA correctly, including `::` qualifiers and ambiguous suffixes.
- [ ] Unknown opcodes/qualifiers reach `#unsupported(...)` rather than being silently accepted (README promise).
      Grep for catch-all rules that drop modifiers.
- [ ] Register-range declarations, vector registers, scope shadowing (`name@id`), labels in nested blocks.
- [ ] Static allocation: alignment, `.extern` dynamic shared size, `.const`/`.global` initialisers.
- [ ] Run the clang/nvcc corpora (`corpora/`, `scripts/parse-corpus.sh`) and record parse failures.

## 4. Integer and bit operations (`ptx-alu.k`, `ptx-exec-int.k`, `-bits`, `-intx`)

- [ ] Signed vs unsigned, width truncation and sign extension for every width (8/16/32/64), `.hi/.wide/.sat`.
- [ ] Carry chains: CC.CF set/consumed per ISA; behaviour of `addc` without a preceding `.cc`.
- [ ] Shift amounts ≥ width, `bfe`/`bfi` position/length clamping, `prmt` modes, `shf` clamp vs wrap.
- [ ] Documented choice "integer division by zero yields all ones": confirm PTX leaves it unspecified and that
      `rem` by zero and `INT_MIN / -1` are handled consistently.

## 5. Floating point (`ptx-float.k`, `ptx-exec-half.k`, cvt)

- [ ] Review the custom binary16/32/64 encode/decode and the 4400-bit FMA width argument.
- [ ] Rounding modes × `.ftz` × `.sat` × NaN for every op (ROADMAP #80 is still open — scope what is untested).
- [ ] NaN policy: canonical `0x7FFFFFFF` everywhere? Check `min/max` NaN propagation, `.NaN`, `abs/neg` (bit ops,
      must not canonicalise), `copysign`, and f16/bf16 canonical NaNs.
- [ ] Float→int conversions: saturation and NaN→0 match the ISA for every rounding variant.
- [ ] `.approx` returns the correctly rounded value: legitimate as one allowed behaviour, but note that programs
      relying on approximation error are not explored; check `rsqrt`, `ex2`, `lg2` subnormal/`.ftz` behaviour.
- [ ] fp8/fp6/fp4/ue8m0: `.satfinite`, NaN/Inf encodings (e4m3 has no Inf), rounding, packed element order.
- [ ] tf32 and `cvt.rna`/`cvt.rs` (random-bits operand).

## 6. Memory and state spaces (`ptx-exec-mem.k`, `-memq`, `ptx-configuration.k`)

- [ ] Little-endian byte layout, vector `ld/st` element order, `.b128`.
- [ ] Generic-address windows: disjoint, `cvta` round-trips, `isspacep`, the "below 4096 faults" rule.
- [ ] Bounds and misalignment UB checks (#72): which spaces are checked; document what is not.
- [ ] Atomics: every op × type, return value, `cas` semantics, `inc/dec` wrap rules, `.sem/.scope` accepted.

## 7. Concurrency and memory model

- [ ] Partial-order reduction soundness: audit `isVisible` in `ptx-exec-core.k` against every instruction that
      touches shared state (generic loads, atomics, async copies, mbarrier, cluster/DSMEM, textures, fabric,
      multimem). A missing entry silently prunes schedules.
- [ ] Barriers: count semantics, exited threads, `bar.red`, `.aligned`, cluster barriers; deadlock detection.
- [ ] Warp collectives: membermask handling, mismatched masks, lanes that exited (README's `vote_all_sub` gap).
- [ ] Relaxed mode vs the official PTX model: re-run `scripts/ptxmm.py` / `run-memmodel.py`; list litmus
      shapes not covered (scopes #67, causality #68 are open — make sure docs don't overclaim).
- [ ] Async agents (#70): cp.async / bulk / TMA / mbarrier tx-count completion ordering.

## 8. Matrix, async, textures (largest modules: `-tc`, `-mma`, `-tma`, `-tex`)

- [ ] For each layout listed as an *assumption* in `docs/prior-work.md` (mma.sp metadata, tcgen05, block-scale,
      ldmatrix sm_100 forms, …), check whether the ISA now specifies it in text and whether hardware data exists.
- [ ] Spot-check a few fragment mappings by hand against the ISA figures (one per shape family).
- [ ] TMA: swizzle, OOB fill, im2col, interleave; texture addressing/filtering and the zero-border limitation.

## 9. Legality (`ptx-isa-table.k`, `.isacheck`)

- [ ] Regenerate the table (`scripts/gen_isa_table.py`) and diff; check version/target minima for a sample.
- [ ] Confirm the K and Sail copies of the table agree (`sail/ptx/gen_isa_table.py`).

## 10. AGX and Sail side

- [ ] `agx/agx.k` and `sail/agx`: check encodings against applegpu (`xcheck_applegpu.py`) and note what is
      documented-only vs hardware-observed.
- [ ] `tools/ptx2agx.py` / `validate_agx.py`: what translation validation actually proves (per-test equivalence on
      concrete inputs, not general correctness) — make sure the docs say so.
- [ ] K vs Sail PTX models: run `sail/difftest.sh` and list disagreements.

## 11. Documentation consistency (final pass)

- [ ] One source of truth for coverage: reconcile README, `docs/coverage.md`, `ROADMAP.md`, `sail/README.md`.
- [ ] Every modelling choice (canonical NaN, division by zero, `.approx`, fabric always succeeds, converged-warp
      `activemask`, assumed layouts) is listed in one place.
- [ ] Every Makefile target named in the docs exists and works.

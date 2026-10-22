# Sail models of AGX and PTX

Executable [Sail](https://github.com/rems-project/sail) re-expressions of the two K semantics in this
repository, built with Sail's C backend and checked against the same tests.

```
eval $(opam env)          # Sail 0.20.3
make sail                 # sail/agx/build/agx, sail/ptx/build/ptx
make sail-test            # agx/tests/*.agx and tests/*.ptx on the Sail models
make -C sail xcheck-agx   # AGX bit layouts against dougallj/applegpu (APPLEGPU=/home/user/applegpu)
```

| Directory | Contents |
|---|---|
| `common/` | shared Sail: integer helpers (`ints.sail`: K's `&Int`, `|Int`, `xorInt`, shifts on unbounded integers), list helpers, the IEEE-754 core of `semantics/ptx-float.k` (`ieee.sail`), and the C runtime (`io.c`: token reader of the drivers, sparse byte memories) |
| `agx/` | the AGX (Apple G13, with the G14C device-atomic forms) model of `agx/agx.k`: AST, execution, `encdec` binary encodings, `assembly` mapping, driver |
| `ptx/` | the PTX model of `semantics/*.k` |

## AGX (`sail/agx`)

Files: `agx_types.sail` (operands, `union ast`), `agx_state.sail` (machine state), `agx_exec.sail`
(`function clause execute`), `agx_encdec.sail` (`mapping clause encdec`), `agx_asm.sail`
(`mapping clause assembly`), `agx_main.sail` (driver), `agxtest.py` (test driver), `xcheck_applegpu.py`
(encoding cross-check).

**State** (as in `agx/agx.k`): one SIMD-group of 32 lanes in lock step; pc; 32-bit execution mask;
general registers r0..r255 per lane with 16-bit halves (`rNl`, `rNh`) as views; the execution-mask
stack counter in r0l; 256 uniform registers; special registers per lane; byte-addressed device and
threadgroup memories (sparse, kept by the C runtime); the list of pending loads (results become
visible at the next `wait`); status / exit code / message.

**Instructions**: `mov get_sr iadd isub imadd bitop bfi bfeil asr popcount bitrev ffs fadd fmul fmadd`
(f32, results flushed to zero) and `fadd16 fmul16 fmadd16`, `rcp`, `icmpsel fcmpsel`,
`if/else/while_{i,f}cmp`, `pop_exec`, `jmp_exec_any/none`, `icmp_ballot fcmp_ballot`,
`simd_shuffle{,_xor,_up,_down}`, `device_load/store` (with the `signed|unsigned` index operand),
`threadgroup_load/store`, `device_atomic threadgroup_atomic`, `wait`, `threadgroup_barrier`, `memory_barrier`,
`stop`, `trap` (see "G14C additions" below).
Every instruction follows the corresponding K rule (lane loop over the pre-instruction state, `opv`
operand views, the exact `cmpv` condition semantics including its treatment of integer conditions
on float compares, `stkNew` stack arithmetic over all lanes, loads captured at issue and written at
`wait`, atomics serialised in lane order).  Float arithmetic uses `common/ieee.sail` (the same
semantics as `PTX-FLOAT`).

The AST also carries the fields of the binary encodings that the K dialect has no syntax for, with
the K model's behaviour as default: register-cache hints, source modifiers (`.sx`, `.abs`, `.neg`),
`.sat`, `lsl` shifts and `imsub`/`isub`, push counts of the stack compares, 64-bit register pairs,
non-contiguous element masks, wait groups, threadgroup base registers, atomics without destination
and the further atomic operations (`sub imin imax`), and the hardware condition codes without a K
name (`seq sne fltn fgtn fnequ fnlt fngt fnle fnge fnltn fngtn`).  Their semantics follow the
pseudo-code of applegpu.

**Encodings** (`encdec : ast <-> bits(80)`, little-endian instruction word zero-extended to the
longest 10-byte form; `agx_size` gives the length of the instruction at the start of a window) for
every instruction above: `mov_imm` (16/32-bit), `mov` (bitop alias), `get_sr`, `iadd/isub`,
`imadd/imsub`, `bitop`, `bfi`, `bfeil`, `asr`, `popcount`, `bitrev`, `ffs`, `fadd32/16`, `fmul32/16`,
`fmadd32/16`, `rcp`, `icmpsel`, `fcmpsel`, `if/else/while_icmp/fcmp`, `pop_exec`,
`jmp_exec_any/none`, `icmp_ballot`, `fcmp_ballot`, the four shuffles, `device_load/store`,
`threadgroup_load/store`, `atomic`, `threadgroup_atomic`, `wait`, `threadgroup_barrier`, `memory_barrier`, `stop`,
`trap`.
Short / long forms are chosen like applegpu (long iff a bit beyond the short form is set).  Operand
forms that applegpu's pseudo-code calls UNDEFINED (odd register numbers for 32-bit operands, ...) or
that are ambiguous are rejected by the decoder, so `encdec` is a bijection on what it accepts.

Checks:
* `agxtest.py` runs `agx/tests/*.agx` (all pass) and `sail/agx/tests/*.agx`; `--roundtrip` prints
  every instruction with the `assembly` mapping, parses the text again and compares, and encodes
  each encodable instruction, decodes the encoding (from a window with garbage after it) and
  compares; `--binary` reruns the fully encodable programs from their binary encoding.  A test passes when the run
  ends in `stopped`, or in the status named by `.expectstatus` (`trapped`, or `fault` for a form the model leaves
  undefined, with `.expectmsg TEXT` naming the error).
* `xcheck_applegpu.py` encodes the test instructions (`agx/tests`, `sail/agx/tests`) and 3000 random instructions with the Sail
  model and disassembles them with applegpu (all agree), and decodes 3000 random words with both
  (every word that the Sail model decodes disassembles identically in applegpu).  One applegpu
  disassembler issue is accounted for: `CmpselSrcDesc.decode` drops bit 8 of a uniform register
  index (its encoder and pseudo-code keep it, as the Sail model does).

Not encodable (the K dialect allows them, the hardware forms do not): integer immediates beyond
8 bits in ALU sources (16/32 bits in `mov`), float immediates that are not one of the 256 8-bit
float codes, integer immediates in float sources, 32-bit registers as threadgroup indices, a
threadgroup `shift`, immediate atomic values, non-adjacent `uA_uB` pairs, K's `fne` (ordered
not-equal; the hardware's inverted `eq` is `fnequ`), and conditions without a hardware code on
`icmpsel`/`fcmpsel` (no inversion bit there).  Decoder rejections: the r0l cache hint of the stack
instructions, the `kill` bit of `mov_imm`, values of unidentified atomic fields other than those
applegpu observed, bits 52 and 53 of the 8-byte `fmadd32`/`fmadd16` (below), and the device-access shift value 3 (applegpu's notes suggest it acts as 2,
unconfirmed; agx.k adds the shift as given, so the value is rejected in both directions and every
encodable access means the same in agx.k, this model and the hardware notes).

Semantics follow the hardware where applegpu's descriptions and the hardware differ, as recorded by the
AGX-Performance-Simulator test suite (M2 Max; its functional simulator `agx_run` matches all 2670 recorded cases) and
checked by the differential test below; `agx/agx.k` was changed in the same way: sources keep their own size (`asr`
sign-extends A at its size, integer compares extend each operand at its size, also for `seq`/`sne`), truth tables
`0011`/`1100` give A, float sources of 16-bit registers are binary16 and fp32 subnormal sources read as zero in the
32-bit forms and compares, binary32 results below 2^-126 become zeros before rounding, a 16-bit destination of a 32-bit
form receives the exact result rounded to binary16, NaNs are canonical (0x7fc00000 / 0x7e00), `.sat` maps NaN and -0
to +0, shuffles move values between quads (applegpu's formulation), memory register tuples of 16-bit registers are
consecutive halves, device stores keep the lowest lane's value on conflicts, threadgroup accesses address byte
`base + 4 * size * index + size * element` and threadgroup atomics `base + 16 * index` with their result written
without a wait. The LLVM AGX backend relies on this indexing since 2026-10-20 (llvm-project 56d7e6bbc). It puts
constant threadgroup offsets that are multiples of 4 * size in the immediate index, a form `agxgen.py` already
generates (`idx` imm) and `mccheck.py` / `simcheck.py` cover.

Remaining, documented differences: `rcp` is the correctly rounded reciprocal (the hardware's approximation is within
one ulp; `simcheck.py` runs `agx_run` without its recorded tables); the model's threadgroup memory has no size (the
hardware reads 0 / drops stores beyond the allocation); reading a load destination before its `wait` gives the old
value in the model, while `agx_run` completes wait-group-0 loads at once (not observed on the hardware).

### G14C additions

Measured on an Apple M2 Max (driver-reported G14C B1) in Apple-M2-Playground; the model has no chip parameter, so
**bits 45-46 of the device atomic mean what they mean on G14C** (on G13X Mesa sets bit 45 as a multi-die coherency
bit; whether `sub` / `cmpxchg` change meaning there is open, there was no G13X device).

* **Device atomic bits 45-46** (`novel-results/2026-10-17-atomic-variants`, 74 dispatch pairs, 0 faults).  The AST's
  `mode` is the LLVM backend's operand (`AGXAtomicMode`): bit 0 = bit 45, bit 1 = bit 46 clear, so 0 is Mesa's G14
  encoding (b45 = 0, b46 = 1).  Assembly: `device_atomic OP D, base, index, S[, signed][, g1][, mode=N]`, `mode=` only
  when it differs from the op's default (1 for `fadd` and `umax64`, 3 for `umin64`, 0 otherwise).

  | op field | b45 = 0 | b45 = 1, b46 = 0 | b45 = 1, b46 = 1 |
  |---|---|---|---|
  | 0, 2, 4-10 | the 32-bit op | same op | same op |
  | 1 | `sub` | `fadd` | `fadd` (the backend's form, mode 1) |
  | 3 | `cmpxchg` | `umin64` (mode 3) | `umax64` (mode 1) |
  | 11-15 | `end11`..`end15` (mode 0 only) | rejected (not run) | rejected (not run) |

  * `cmpxchg`: 32-bit compare-and-swap; S is the pair `rN_rN+1`, rN the new value, rN+1 the compare value; D receives
    the old word.
  * `fadd`: float32 add, round to nearest even, denormal inputs and results flushed to sign-preserving zero, every NaN
    result 0x7fc00000 (2,048 edge and rounding pairs; this is PTX `atom.add.f32`); D receives the old word.
  * `umin64` / `umax64`: unsigned min / max of the 64-bit word at base + 4 * index (the index counts 32-bit words) with
    the pair S (lo), S+1 (hi); with a destination D receives 1 if memory changed, else 0 (not the old value).
  * `end11`..`end15`: no memory effect; the SIMD group ends, inactive lanes included, and earlier stores have landed.
    In this one-SIMD-group model that is the model's `stop`.
  * As for the other atomics, active lanes are serialised in lane order and returned values become visible at `wait`
    (the K model's convention; on the hardware the order varies between runs, 1,024-lane contention tests found one
    serial chain).
* **`memory_barrier a, b, c`** (2 bytes, `0xf5 | b << 8 | a << 10 | c << 12`, applegpu `MemoryBarrierInstructionDesc`
  and the LLVM backend's `MEMORY_BARRIER_g13`).  Every field value encodes; with one SIMD group and one sequential
  memory the model executes it as a no-op.  Only the seven encodings the backend emits ran on the hardware
  (`2026-10-18-fence-cost`, `2026-10-11-memory-ordering`: f596, f533, f59a, f59e, f503, f599, f59d); the model stops
  with an error on the others.
* **`trap`** (0x0008, applegpu and the backend's `TRAP_g13`).  **Provisional**: what it does when executed is not
  measured (AGX-Performance-Simulator task B21; `agx_run` treats it as `stop`).  The model ends execution with its own
  status `trapped` and the message `trap`, so tests can tell it from `stop` (`.expectstatus trapped`).
* **fmadd bits 52 and 53** (`2026-10-11-fmadd-lane-offset`, `2026-10-18-fmadd-followups`, `2026-10-18-fma-b52`): bit 53
  of the 8-byte form selects a 10-byte form whose bits 75-79 are a lane offset for A, and bit 52 changes the fmadd32
  operation.  The decoder rejects both bits (it used to accept and ignore them).  The lane form is not modelled: which
  value lanes 16..16+k-1 read depends on the following instruction (`wait` or ALU) and on the distance to the write of
  A, and is not deterministic at any tested distance; with a `wait` after it, lanes 0..k-1 occasionally read non-zero
  values; the grouped forms (bits 72 / 73) of fmadd32 read undefined values outside the group and were measured only
  with k = 1; bit 74, bits 72 + 73 together and fmadd16 lane forms under a mask are not established.

Not modelled, or left undefined (the model stops with an error rather than guess):
* `fadd` without a destination (`red.add.f32`): not run on the hardware.
* `umin64` / `umax64` with a destination when the source equals the memory word (no tested lane had equal values), and
  at an address that is not 8-byte aligned (the hardware tests used even indices with aligned bases).  64-bit min / max
  under contention and with a register base were not measured either; the model treats them like the other atomics.
* Threadgroup atomics keep the ten 32-bit operations (no threadgroup float add was found among 19 gap bits; its
  `cmpxchg` and ops 11-15 are not modelled).
* The 31 undecoded opcodes with low bits `0b11` (`2026-10-11-opcode-descriptions`): 6-byte true no-ops, but only for
  the tested operand bytes, so the decoder does not accept them (a disassembler must still skip 6 bytes); likewise
  the 16 "ends" opcodes, 0x18 and 0x78.  The 0x78 barrier (4 aligned SIMD groups) is not modelled.
* The ordering memory barriers and fences provide between cores, SIMD groups and threadgroups: the model has one SIMD
  group and one sequential memory.

### Differential tests (`sail/agx/gen`)

* `agxgen.py`: instruction generator over the Sail AST: every instruction schema, each dimension (operand kinds
  r32/r16l/r16h/r64/u32/u16l/u16h/immediates/float immediates, hints, `.sx`/`.abs`/`.neg`, conditions, opcodes,
  saturation, shifts, formats, masks, signedness, wait groups, push counts) one factor at a time, plus random forms.
* `mccheck.py`: the LLVM AGX MC layer against `encdec`: applegpu disassembly of the Sail bytes, `llvm-mc --disassemble`
  (length, text, re-encoding) and `llvm-mc` assembly of the expected text, for the generated forms, random
  instructions and random words the Sail model decodes.
* `simcheck.py`: random programs (ALU, nested if/else/while with exec-mask jumps, device/threadgroup memory and
  atomics), encoded by the Sail model, run on the Sail model and on `agx_run`; final registers and memory compared.
  `--tests` also runs Sail-dialect tests (`sail/agx/tests/hw-semantics.agx`, `atomic-variants.agx`) on `agx_run`.
  Device atomics cover every op but `end11`..`end15` and every mode; `umin64` / `umax64` without destination at even
  indices, `fadd` with a destination (the forms the model defines).

### PTX differential tests (`sail/ptx/gen`)

* `ptxgen.py OUT -s SEED`: one kernel per instruction form of the Sail PTX model (arithmetic, conversions, half and
  packed types, integer, video, atomics, compares, carry chains), with edge-case and random inputs and `.expect`
  values computed by the model.
* `ptxdiff.py OUT --mode strict|default|both`: runs them through ZLUDA (`zoc`) and `agx_run`, and classes each form as
  pass, fail, approx (an `.approx` instruction within its tolerance), nan-unspecified, crash or rejected.
* `sail/difftest.sh SEED` runs both halves (AGX and PTX).

Seed 1, 2026-10-20 (1578 forms, simulator main b3bbae1). In strict mode, 1535 pass, 17 fail and 2 crash; 7 are
rejected. The default mode is not exact by design (bf16 NaN payloads, denormals). ZLUDA then fixed:
* `cvt.sat` from unsigned to wider signed types;
* `cnot`;
* `ex2.approx` and `tanh.approx` on f16x2 / bf16 / bf16x2;
* f16 <-> bf16 `cvt` flushing bf16 subnormals.

On the failing files that leaves 7 strict failures:
* the NaN payload of `atom.add.f32` (a hardware question, AGX-Performance-Simulator `tasks/ptx-re-blockers.md` B3);
* `.ftz` + `.sat` on negative subnormal results (ZLUDA flushes before saturating, giving -0; the model gives +0);
* `min.ftz.f32` of a negative and a positive subnormal (+0 instead of -0; B11 asks how the compares treat
  denormals);
* `div.approx.f32` special cases;
* 1-ulp `tanh.approx.f16` results that are outside the tolerance.

## PTX (`sail/ptx`)

```
make -C sail ptx              # sail/ptx/build/ptx
make -C sail test-ptx         # tests/*.ptx against tests/*.out (or their .expect checks)
make -C sail test-ptx-extra   # tests/neg (must fail with the expected message), tests/zluda (EXPECTED_PASS)
make -C sail report           # all three suites -> sail/ptx/TEST-REPORT.txt
python3 sail/ptx/ptxtest.py -v tests/mma-0.ptx    # single tests, -v shows output diffs
```

**Structure.**  `ptxfront.py` is the front end: the lexer and parser of `ptx-syntax.k` and the
purely syntactic loading pass of `ptx-load.k` (flattening function bodies into code lists, nested-scope
register renaming, vector-register expansion, label and register-width tables, static declarations,
`pp`/`ppDecl`).  It serializes the module and the harness directives as a token stream; everything
with run-time meaning (linking, static allocation, buffers, launches, `.print` / `.expect` / `.patch`,
every instruction) is done by the Sail model (`ptx_main.sail` reads the stream).  `ptxtest.py`
runs a test and compares stdout with the `.out` file (exit code from the model's `EXIT n` line).

**Reference.**  The K semantics (`semantics/`, `agx/agx.k`) is obsolete for now; these Sail models are the maintained
ones.  The PTX model was written as a re-expression of the K semantics in `semantics/`, and has since been
corrected against the public PTX ISA (Release 9.3, and 9.4 since) wherever the two differ; `docs/review-checklist.md` lists each
correction with its ISA section and every place where this model and the K rules now disagree.

Instructions keep K's representation: `opc(base, modifiers, types)` from `decodeOpc` and operands as
in `ptx-syntax.k`, layered because Sail has no recursive types (simple operands, addresses, bracket
items, instruction operands).  Each K module is a Sail file with the same rules, in K's order of
priority (`ptx_dispatch.sail`: the `priority(40)` rules, then the opcode-specific `exec` rules, then
the generic `#gx` rules, then `#unsupported`).  K's maps are string-keyed maps and sparse byte
memories of the C runtime (`common/io.c`); per-thread and per-CTA cells are register vectors.

| K module (`semantics/`) | Sail file | Contents |
|---|---|---|
| `ptx-common.k`, `ptx-float.k` | `ptx_common.sail`, `common/ieee.sail` | types, literals, `fmtFloat`, IEEE-754 encode/decode and rounding (exact integer arithmetic, all modes, ftz / sat / relu / satfinite) |
| `ptx-alu.k`, `ptx-exec-alu.k` | `ptx_alu.sail`, `ptx_exec.sail` | integer / float / compare / convert / atomic operations, setp, mov (pack/unpack), cvt, cvta |
| `ptx-exec-core.k`, `ptx-exec-mem.k`, `ptx-configuration.k` | `ptx_state.sail`, `ptx_core.sail`, `ptx_exec.sail` | operands, effective addresses, address windows, `#mem` (with `.ub strict` checks), ld / st / atom / red, vectors |
| `ptx-exec-ctrl.k`, `ptx-exec-sync.k` | `ptx_exec.sail`, `ptx_sched.sail` | bra / brx / call / ret / exit / trap, ABI of `.param` / `.reg` arguments, bar / barrier / bar.red, exit and barrier bookkeeping |
| `ptx-exec-coll.k`, `ptx-exec-warp.k` | `ptx_coll.sail` | rendezvous of warp(group) collectives, shfl, vote, match, elect, redux, bar.warp, activemask (also `.convergence its`) |
| `ptx-exec-int.k`, `ptx-exec-bits.k`, `ptx-exec-intx.k`, `ptx-exec-pred.k`, `ptx-exec-misc.k` | `ptx_int.sail`, `ptx_async.sail` | carry chains, bit operations, packed integers, set / slct / testp / isspacep / ldu, clocks, alloca, st.async / red.async / st.bulk / clusterlaunchcontrol |
| `ptx-exec-half.k`, `ptx-exec-memq.k`, `ptx-exec-video.k` | `ptx_half.sail` | f16 / bf16 / packed / mixed precision, vector atomics, cache hints, video instructions |
| `ptx-exec-mini.k` | `ptx_mini.sail` | fp8 / fp6 / fp4 / ue8m0 / ue5m3 conversions |
| transcendentals of `ptx-float.k` | `ptx_math.sail` | correctly rounded sqrt, rcp, rsqrt, sin, cos, lg2, ex2, tanh with K's intermediate roundings (fixed-point series with a Ziv rounding test) |
| `ptx-exec-mma.k`, `ptx-exec-sp.k` | `ptx_mma.sail` | mma.sync (all shapes / types), block-scaled mma, wmma.mma / load / store, mma.sp (dense and block-scaled) |
| `ptx-exec-ldm.k` | `ptx_ldm.sail` | ldmatrix / stmatrix / movmatrix (m8n8, m16n16, m8n16, m16n8) |
| `ptx-exec-cluster.k`, `ptx-exec-mbar.k`, `ptx-exec-async.k`, `ptx-exec-bulk.k` | `ptx_async.sail` | DSMEM, mapa / getctarank, barrier.cluster, mbarrier (incl. multicast, try_wait), cp.async, cp.async.bulk / cp.reduce.async.bulk |
| `ptx-exec-wg.k` | `ptx_wg.sail` | wgmma (register / descriptor A, sparse), matrix descriptors and swizzles |
| `ptx-exec-tma.k` | `ptx_tma.sail` | tensor maps (`.tensormap`), TMA loads / stores / reductions (tile, gather4 / scatter4, sub-byte types, im2col modes), tensormap.replace / cp_fenceproxy |
| `ptx-exec-spc.k` | `ptx_spc.sail` | spcompress / spdecompress |
| `ptx-exec-tc.k` | `ptx_tc.sail` | tcgen05: tensor memory alloc / ld / st (red, spcompress), mma (dense, sparse, block-scaled, `.ws`, `.ashift`, cta_group::2), commit, cp, shift |
| `ptx-exec-tex.k` | `ptx_tex.sail` | textures (filtering, mipmaps, cubemaps, depth compare, tld4, txq), surfaces, samplers, istypep |
| `ptx-exec-mm.k`, `ptx-exec-fabric.k` | `ptx_mm.sail` | multimem.ld_reduce / st / red, fabric.* on logical endpoints |
| `ptx-exec-isa.k`, `ptx-isa-table.k` | `ptx_isa.sail`, `ptx_isa_table.sail` (generated by `gen_isa_table.py`) | `.isacheck` legality check against `.version` / `.target` |
| `ptx-harness.k`, `ptx-load.k` (run-time part) | `ptx_main.sail` | linking, statics, buffers, launches (limits, cooperative, occupancy), print / expect / patch / append, harness directives of the subsystems |
| `ptx-exec-weak.k` (`.memmodel relaxed`) | -- | not modelled (the relaxed store-buffer mode needs K's exhaustive search); such programs stop with an error |

**Schedule.**  K runs the threads interleaved at instruction granularity under the turn of
`ptx-configuration.k`.  The Sail model executes one deterministic schedule of that transition system:
threads take turns round-robin in creation order (CTA order, then thread index); a turn executes the
thread's next instruction and then its thread-local instructions up to the next visible one (memory
outside `.param` / `.local`, barriers, warp collectives, the visible extension instructions, exit).
A blocked thread (barrier, collective rendezvous, `mbarrier.try_wait`, `barrier.cluster.wait`) is
skipped until its condition holds; when no thread can move the run stops with a deadlock error.
Programs whose result depends on the schedule can differ from a particular K run.

**Numerics.**  Floats are exact dyadic values (sign, integer mantissa, exponent) rounded once into
the destination format, which is what K's 4400-bit (and, in the tensor-core rules, 300-bit) MPFR
floats do for all operations whose exact result fits; in the texture rules, where K's intermediate
results do not fit, every operation is rounded to 4400 bits as in K, divisions at double precision
(`divD`) and logarithms correctly rounded at double precision.  A difference from K could only arise
for mma / wgmma f64 sums whose terms span more than 300 bits of exponent.

**Order details that K leaves to its map implementation** (they do not affect the tests): the lanes
of a collective are combined in ascending order; `.isacheck` reports the first offending instruction
in function definition order.

### PTX test results

Run with `make -C sail report` (the full list is in `sail/ptx/TEST-REPORT.txt`).  A test of
`tests/*.ptx` passes when the model's output equals the `.out` file produced by the K semantics and
every `.expect` holds; `tests/neg` programs must fail with the message of their `// expect:` line;
`tests/zluda` programs must print PASS when they are listed in `tests/zluda/EXPECTED_PASS` (the list
of programs the K semantics passes).

| Suite | Tests | Pass | Fail |
|---|---|---|---|
| `tests/*.ptx` | 285 | 281 | 4 (`ldm2-1` .. `ldm2-4`, see below) |
| `tests/neg/*.ptx` | 21 | 21 | 0 |
| `tests/zluda/*.ptx` | 184 (182 expected to pass) | 182 of 182 | 0 (`reg_local` also fails in K; `subc_cc_s32` records the hardware's carry polarity of `sub.cc`, the model follows the ISA's borrow) |

`tests/*.ptx` by subsystem:

| Subsystem | Tests | Pass |
|---|---|---|
| core: ALU, floats and conversions (incl. narrow formats, transcendentals), memory, control flow, calls, barriers, warp collectives, video, harness | 101 | 101 |
| mma.sync, block-scaled mma, wmma, mma.sp | 52 | 52 |
| wgmma (dense, sparse) | 34 | 34 |
| tcgen05 | 32 | 32 |
| TMA and tensor maps | 28 | 28 |
| ldmatrix / stmatrix / movmatrix | 22 | 18 |
| mbarrier, cp.async, bulk copies, clusters, st.async | 9 | 9 |
| textures, surfaces, istypep | 4 | 4 |
| multimem, fabric | 2 | 2 |
| spcompress | 1 | 1 |

The four failures `ldm2-1` .. `ldm2-4` use `ldmatrix` without a state space on a register that holds
a raw `.shared` offset (`mov.u64 %rd2, buf`, no `cvta`).  `ptx-exec-ldm.k` resolves the address with
`ea(A, "generic")`, which maps an address below the shared window to `.global`, and the read of global
address 0 .. 511 then faults (`#mem` owise rule: "invalid global address").  The model follows these
rules and stops with that fault; the golden `.out` files expect the shared-memory reading.  They were
added in a commit marked "build pending", so the expected output does not seem to come from a run of
the current K rules (the K toolchain is not available here to confirm).  The `.shared` variants of the
same instructions (`ldm2-0`, `ldm2-5`) pass.

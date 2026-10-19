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
| `agx/` | the AGX (Apple G13) model of `agx/agx.k`: AST, execution, `encdec` binary encodings, `assembly` mapping, driver |
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
`threadgroup_load/store`, `device_atomic threadgroup_atomic`, `wait`, `threadgroup_barrier`, `stop`.
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
`threadgroup_load/store`, `atomic`, `threadgroup_atomic`, `wait`, `threadgroup_barrier`, `stop`.
Short / long forms are chosen like applegpu (long iff a bit beyond the short form is set).  Operand
forms that applegpu's pseudo-code calls UNDEFINED (odd register numbers for 32-bit operands, ...) or
that are ambiguous are rejected by the decoder, so `encdec` is a bijection on what it accepts.

Checks:
* `agxtest.py` runs `agx/tests/*.agx` (all pass) and `sail/agx/tests/*.agx`; `--roundtrip` prints
  every instruction with the `assembly` mapping, parses the text again and compares, and encodes
  each encodable instruction, decodes the encoding (from a window with garbage after it) and
  compares; `--binary` reruns the fully encodable programs from their binary encoding.
* `xcheck_applegpu.py` encodes the test instructions and 3000 random instructions with the Sail
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
applegpu observed, and the device-access shift value 3 (applegpu's notes suggest it acts as 2,
unconfirmed; agx.k adds the shift as given, so the value is rejected in both directions and every
encodable access means the same in agx.k, this model and the hardware notes).

Known semantic differences from the hardware descriptions (the model follows `agx.k`): shuffles are
plain lane reads (applegpu describes a quad-based formulation), a 16-bit destination of an f32
operation takes the low half of the f32 result (applegpu rounds to f16), and `fcmpsel`/ballot
compares read registers without flushing f32 subnormals.

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
| `tests/zluda/*.ptx` | 184 (181 expected to pass) | 181 of 181 | 0 (the 3 others also fail in K: `reg_local`, `vote_all_sub`, `vote_ballot_nosync`) |

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

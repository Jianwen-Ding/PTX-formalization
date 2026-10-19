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

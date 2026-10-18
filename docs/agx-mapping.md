# Apple AGX (G13 / M1 and later) as a target for PTX

Groundwork for implementing PTX on Apple GPUs.  Everything about AGX below comes from the public reverse-engineering
project *dougallj/applegpu* (documentation `docs.html`, instruction descriptors and a hardware-validated emulator in
`applegpu.py`) and from Metal's documented programming model; Apple publishes no ISA.  The G13 documents state that the text is
"based on reverse engineering and is likely to have mistakes"; semantics marked **doc** below are written out in pseudo-code
there, **desc** means an instruction descriptor exists (encoding known) but the behaviour is `TODO()`, **mtl** means only Metal
documents the behaviour.  Later generations (G14, G15, G16/M3 with dynamic caching and ray tracing, M4) are not covered.

## 1. Machine model

| AGX (G13) | PTX counterpart |
|---|---|
| SIMD-group of **32 threads** (Metal "SIMD-group"); each thread is a Metal "thread" | warp of 32 threads / thread |
| per SIMD-group: `pc`, stack pointer `sp`, 32-bit `exec_mask`, link register `r1` | per-thread pc (independent thread scheduling) |
| up to **128 32-bit GPRs** per thread, each also addressable as low/high **16-bit halves** (`r0l`, `r0h`); pairs for 64-bit values; 4 consecutive regs for vector memory ops | `.reg` (virtual, unlimited) |
| **256 uniform registers** `u0..u255` (also 16-bit halves), same value in all threads: buffer addresses, grid sizes | kernel `.param`, `ld.param` of uniform data |
| `r0l` is the **execution-mask stack counter**; `if_icmp`, `else_icmp`, `while_icmp`, `pop_exec`, `jmp_exec_any/none` manage it and `exec_mask` | predicated execution / `bra` with divergence |
| register cache with `cache` / `discard` hints (performance only; `discard` makes a later read undefined) | none |
| memory: **device** (global, 64-bit base from uniform or GPR + offset), **threadgroup** (CTA shared), **stack** (per-thread), **tile** / image-block, textures + samplers, uniform-register stores | `.global`, `.shared`, `.local`, `.const`, `.param`, tex/surf |
| loads are asynchronous; `wait n` synchronises on outstanding loads | in-order register semantics (the lowering inserts `wait`) |
| `threadgroup_barrier`, `memory_barrier` (desc), `simd_*` ops | `bar.sync`, `fence`, `shfl`, `vote` |
| `get_sr` reads special registers (thread position, threadgroup id, ...) | `%tid`, `%ctaid`, ... |
| no clusters, no TMA/bulk copies, no mbarrier, no tensor-memory, no cp.async | PTX-only: emulated or rejected (section 4) |
| `simdgroup_matrix` multiply-add: `SimdMatrixFMadd{16,32}` descriptors | wmma / mma.sync (partial) |

Thread blocks map to Metal **threadgroups** (max 1024 threads; sizes are multiples of 32 as SIMD-groups), grids to the Metal
dispatch grid.  Metal has no warp-specialised divergence control like PTX's `convergence barrier` but the exec-mask stack
gives structured reconvergence at the end of each `if`/`while`.

## 2. Instruction inventory (applegpu.py descriptors)

* **Move**: `mov` (16/32-bit immediate, **doc**), `get_sr` (**doc**), `convert` (desc), `uniform_store`.
* **Integer** (**doc**): `iadd` (also subtract, shift-add by up to 4, saturating, 16/32/64-bit), `imadd` (multiply-add with optional shift, signed/unsigned operands via source flags), `bfi`, `bfeil`, `extr`, `shlhi`, `shrhi`, `asr`, `asrh`, `bitop` (any 2-input boolean function by truth table), `bitrev`, `popcount`, `ffs`, `interleave` (desc).
* **Float** (**doc** for 32/16-bit `fmadd`, `fadd`, `fmul`; round-to-nearest-even; 32-bit results flush to zero (`flush_to_zero=True`), 16-bit do not; optional saturate, source negate/abs modifiers): `fmadd`, `fadd`, `fmul` (+16-bit forms); unary: `floor`, `ceil`, `trunc`, `rint`, `rcp`, `rsqrt`, `rsqrt_special`, `sin_pt_1/2` (two-part sine), `log2`, `exp2`, `dfdx`, `dfdy` (desc, accuracy unknown).
* **Compare / select**: `icmpsel`, `fcmpsel` (**doc**), conditions `eq ne lt ... (signed/unsigned, ordered/unordered)`, `icmp_mask`, `fcmp_mask`.
* **Flow control**: `jmp_exec_any`, `jmp_exec_none`, `call`, `ret`, `stop`, `trap` (**doc**: jumps; `call` sets `r1 = pc + 6`), execution-mask stack `if_icmp/if_fcmp/else_*/while_*/pop_exec` (**doc**, exact stack arithmetic).
* **SIMD**: `icmp_ballot`, `fcmp_ballot` (**doc**), quad ballots (desc), `simd_shuffle` (**doc**), `simd_shuffle_down/up/rotate_up/xor` (desc), `simdgroup_matrix` madd (desc).
* **Memory**: `device_load/store` (up to 4 values per thread, formats 8/16/32 and packed unorm/snorm/rgb10a2/rg11b10f/rgb9e5, base+shifted index, unaligned rounded down; **desc**, semantics in prose), `threadgroup_load/store`, `stack_load/store/get_ptr/adjust`, `wait`, `ld/st_tile`, `ld_var`, `texture_sample/load`, `image_write_block`, `memory_barrier`, `threadgroup_barrier` (desc).
* **Not in the documents**: device/threadgroup **atomics** and their ordering (Metal exposes relaxed atomics only; the instruction forms were not reverse-engineered in this source), 64-bit integer divide / float division (done in software by the compiler), fp64 (no hardware), `cvt` details (`convert` is `TODO()` in the doc), memory ordering (Metal's model: relaxed atomics, `threadgroup_barrier` with a mem-flags argument, `atomic_thread_fence`).

## 3. Mapping of the PTX subset

**Direct**: integer/float ALU (`add sub mul mad`, shifts, bit ops, `popc`, `clz`~`ffs`, `brev`=`bitrev`, `bfe/bfi`), `mov`, `selp`/`slct`=`icmpsel`/`fcmpsel`, `setp` (+ predicate in a register = `icmp_mask`), `bra` with predicates (exec-mask stack), `call/ret`, `ld/st` for global/shared/local (`device_load/store`, `threadgroup_*`, `stack_*`), `bar.sync` (`threadgroup_barrier` for the whole threadgroup), `vote` / `shfl` (`*_ballot`, `simd_shuffle*`), `rcp/rsqrt/sin/cos/ex2/lg2` approximate forms (`rcp`, `rsqrt`, `sin_pt_*`, `exp2`, `log2`; accuracy to be validated on hardware, `cos` via a phase shift), 16-bit float arithmetic (`fadd16`...).

**Needs a sequence**: `div`, `sqrt` (Newton iterations from `rcp`/`rsqrt`, as Metal does), `rem`, IEEE rounding modes other than `.rn` (software), `f64` (software emulation or reject), `.ftz`-less float arithmetic (hardware flushes f32 subnormals: PTX default keeps them, so denormal-sensitive code needs a software path), 64-bit integer multiply/mad (`imadd` gives 64-bit destinations from 32-bit operands), `mulhi`, `madc`/`addc` carry chains (via compares), predicated `atom`, `vote.sync` with a membermask other than full (the exec mask is the only participation control: masks are emulated by predicating the ballot), `shfl.sync` with `.bfly/.idx/.up/.down` (`simd_shuffle_xor/_/up/down`) plus clamp semantics (PTX `c` operand) in software.

**Different semantics, must be modelled explicitly**: independent thread scheduling (AGX reconverges by structured stack: a PTX kernel with unstructured divergent control flow, `bra.uni` misuse, or warp-level synchronisation inside divergent code needs a transformation or rejection), `wait` insertion for asynchronous loads, 32-bit f32 flush-to-zero, the memory model (PTX scoped acquire/release vs Metal relaxed atomics + fences).

## 4. PTX features without AGX hardware

Reject at lowering, or emulate: clusters and distributed shared memory (emulate as one threadgroup if the cluster fits; otherwise reject), `cp.async` and `cp.async.bulk` (synchronous copy + barrier semantics), TMA / tensor maps (software gather), `mbarrier` (threadgroup counter in threadgroup memory + `threadgroup_barrier` polling), `wgmma` / `tcgen05` (emulate through `simdgroup_matrix` or scalar FMAs), `multimem`, `fabric`, FP8/FP6/FP4 conversions (software), `tex`/`surf` (map to `texture_sample/load`; PTX samplers, border modes and filtering rules differ: tasks 96-98), `griddepcontrol`, cooperative launch (Metal has no grid-wide barrier: reject).

## 5. Plan (tasks 92-99)

1. `semantics/agx.k`: an executable K semantics of the **doc**-level subset (SIMD-group state with exec mask and `r0l` stack, GPR/uniform files with 16/32/64-bit views, ALU, float, compare/select, flow control, ballot / shuffle, device / threadgroup / stack memory, `wait` modelled as completion at the next `wait`).
2. A PTX -> AGX lowering (Python, `tools/ptx2agx.py`) for scalar ALU / float / int / memory with **translation validation**: both programs run on the K semantics with the same inputs and results are compared.
3. Warp mapping (shuffle, vote, barriers, divergence) validated the same way.
4. Memory: shared / local / generic windows, atomics and the Metal memory model; tensor-core emulation; async-copy emulation; texture mapping; end-to-end kernels.
5. Hardware cross-check (needs an Apple machine): *applegpu*'s `hwtest.py` compares its emulator with the device; our K semantics should be cross-checked against its emulator (`applegpu.py`) on the same instruction sequences, since no Apple hardware is available in this environment.

## 6. Caveats

* The reverse-engineered text is incomplete (many `TODO()`), possibly wrong, and specific to G13; the lowering must be parameterised by generation.
* Float conversion, transcendental accuracy and denormal behaviour need hardware measurements before any bit-exactness claim is made.
* Nothing here has been run on Apple hardware.

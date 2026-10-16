# PTX-formalization

An executable formal semantics of **NVIDIA PTX** (Parallel Thread Execution,
the virtual ISA that CUDA compiles to), written in the
[K Framework](https://kframework.org).

From the definition, K gives you:

* an **interpreter**: run real PTX (as emitted by `nvcc`) on a modelled GPU
  grid with `krun`;
* **exhaustive interleaving exploration**: `krun --search` enumerates every
  thread schedule, which finds data races, missing barriers and deadlocks;
* a basis for **deductive verification** of kernels with K's symbolic
  backends (`kprove`).

```
$ make && make test
PASS  tests/atomics.ptx
PASS  tests/calls.ptx
...
---- 9 passed, 0 failed

$ make examples
examples/race.ptx                  reachable exit codes: {0,1}
examples/race-atomic.ptx           reachable exit codes: {0}
examples/reduce-search.ptx         reachable exit codes: {0}
examples/reduce-nobarrier.ptx      reachable exit codes: {0,1}
...
```

## Quick start

Requires K 7.1.x (tested with v7.1.337). On Ubuntu 24.04:

```sh
curl -LO https://github.com/runtimeverification/k/releases/download/v7.1.337/kframework_7.1.337_amd64_ubuntu_noble.deb
sudo apt-get install ./kframework_7.1.337_amd64_ubuntu_noble.deb

make                                  # kompile -> build/ptx-kompiled
make run FILE=tests/vecadd.ptx        # print the final configuration
krun -d build/ptx-kompiled tests/vecadd.ptx --output none   # just program output
```

`krun` exits with 0 when every `.expect` check holds and with 1 otherwise,
or when execution gets stuck (fault, deadlock, unsupported instruction). In the
stuck case the final configuration shows where.

## Writing programs

A program is a PTX module followed by a small **host harness** (not part of
PTX) that plays the role of the CUDA runtime:

```ptx
.version 8.0
.target sm_80
.address_size 64

.visible .entry scale(.param .u64 p_x, .param .f32 p_k) { ... }

.buffer X .f32 = { 1.0, 2.0, 3.0 };      // cudaMalloc + cudaMemcpy
.buffer Y .s32 [128];                    // zero-initialised
.launch scale <<< 2, 64 >>> (X, 0.5);    // grid, block (ints or (x,y[,z]))
.launch k2 <<< (2,2), (8,8), 256 >>> (Y);// optional dynamic shared bytes
.print X .f32;                           // X = [0.5, 1.0, 1.5]
.expect X .f32 = { 0.5, 1.0, 0f3FC00000 };// bit-exact check (PASS/FAIL)
```

Launches are synchronous. Kernel arguments are buffer names (passed as global
pointers) or literals encoded at the parameter's declared type. `.print` and
`.expect` also accept module-scope `.global` variables. Literals may be
decimal, `0x` hex, or PTX bit patterns (`0f3F800000`, `0d3FF0000000000000`).

## Layout

| File | Contents |
|---|---|
| `semantics/ptx-syntax.k` | concrete syntax: module/function/declaration grammar, operands, harness directives |
| `semantics/ptx-common.k` | types, bit-vector helpers, IEEE-754 encode/decode, opcode decoding (`ld.global.nc.f32` → base op + modifier set + type list) |
| `semantics/ptx-configuration.k` | machine state, state-space address map, byte memories |
| `semantics/ptx-load.k` | flattening function bodies to code maps, labels, register widths, static allocation |
| `semantics/ptx-alu.k` | pure semantics of arithmetic, logic, comparisons, conversions and atomics |
| `semantics/ptx-exec*.k` | per-thread execution, one module per area: `-core` (operands, scheduling, fetch), `-alu`, `-mem`, `-ctrl` (branches, calls, exit), `-sync` (barriers), `-coll` (warp-collective rendezvous); `ptx-exec.k` assembles them |
| `semantics/ptx-harness.k` | host side: module loading, buffers, grid launch, `.print`/`.expect` |
| `tests/` | regression programs and golden outputs; `tests/zluda/` holds [ZLUDA](https://github.com/vosen/ZLUDA)'s hardware-validated conformance programs (MIT/Apache-2.0), run with `make zluda` |
| `examples/` | `--search` demonstrations (races, barriers, deadlock) |
| `scripts/` | test runner, IEEE-754 differential fuzzer, examples runner |

## Extending

Features are added as new modules, so independent areas do not conflict:

1. **Syntax**: new tokens/productions go in `ptx-syntax.k`. Opcodes need none,
   since any dotted name already parses and is split by `decodeOpc`.
2. **Pure operations** (value computations): add `rule bin/un/tern/cvt(...)`
   cases in `ptx-alu.k` (or a new `ptx-alu-<area>.k` imported by
   `PTX-EXEC-ALU`'s module chain), extending the dispatch sets (`BINOPS`, ...).
3. **Instructions**: create `semantics/ptx-exec-<area>.k` with
   `module PTX-EXEC-<AREA>` importing `PTX-EXEC-CORE`, write
   `rule <tk> exec(opc("name", M, Tys), ops(...), C) => ... </tk>` rules, then
   `requires`/`imports` it in `ptx-exec.k`.
4. **Visible steps**: an instruction touching shared state must be listed in
   `isVisible` (ptx-exec-core.k) so the partial-order reduction yields before it.
5. **Extra per-thread state**: store in `<regs>` under a key starting with
   `%%` (e.g. `"%%cc.cf"`) so it cannot clash with PTX registers, or add a
   cell to `ptx-configuration.k`.
6. **Host directives**: `ptx-syntax.k` (`HarnessDir`) and `ptx-harness.k`.
7. **Warp-collective (`.sync`) instructions** use the rendezvous in
   `ptx-exec-coll.k`: add `rule isCollective("op") => true`, rewrite the
   instruction to `#coll(kind, mask, data, contribution, dests)`, and define
   `collCompute(kind, data, arrivals)`. See the header of that file.
8. Add a program to `tests/` (generate its golden file with
   `scripts/run-tests.sh --update tests/x.ptx`) and tick `ROADMAP.md`.

## What is modelled

**Execution model.** A launch creates the full grid: one `<cta>` per block,
each with its own shared memory, barrier state and `<thread>`s. Each thread has
a register file, `.param` space, `.local` memory and a call stack. The special
registers are `%tid`, `%ntid`, `%ctaid`, `%nctaid` (`.x/.y/.z`), `%laneid`,
`%warpid`, `%nwarpid` and `%warpsize`. Threads interleave at instruction
granularity under a **sequentially consistent** memory model. Each memory
instruction (and so each `atom`/`red`) is a single atomic step.

**State spaces.** `.global`, `.const`, `.shared` (static and `.extern`
dynamic), `.local` and `.param` are separate byte-addressed little-endian
memories. Generic addressing is supported: shared and local memory are
mapped to disjoint windows of the generic space, and `cvta` converts between
generic and state-space addresses. Accesses to global addresses below 4096
(e.g. null) fault.

**Instructions**

| Group | Instructions |
|---|---|
| integer | `add sub mul{.lo,.hi,.wide} mad{.lo,.hi,.wide} div rem min max abs neg` with `.sat`; carry chains `add.cc addc sub.cc subc mad.cc madc` (CC.CF); bit-field `bfe bfi bfind fns bmsk szext prmt` (all modes), `lop3` (incl. predicate form), funnel shifts `shf`, `mul24 mad24 sad dp4a dp2a`, `.relu` min/max, packed `.u16x2/.s16x2` add/sub/min/max/neg/abs; `and or xor not cnot shl shr popc clz brev` |
| floating point (f16/bf16/f32/f64, packed f16x2/bf16x2) | `add sub mul div fma mad min max abs neg sqrt rsqrt rcp sin cos lg2 ex2 tanh copysign` with `.rn/.rz/.rm/.rp`, `.ftz`, `.sat`, and for min/max `.NaN` and `.xorsign.abs`, `fma.relu`, `setp/set` on packed pairs |
| comparison / selection | `set`, `slct`, `testp`, `setp.{eq,ne,lt,le,gt,ge,lo,ls,hi,hs,equ,neu,ltu,leu,gtu,geu,num,nan}[.and/.or/.xor]` with `p\|q` destinations, `selp`, predicated execution `@p` / `@!p`, `not/and/or/xor.pred` |
| data movement | `mov` (incl. symbol addresses, special registers, vector pack/unpack), `cvt` (all int↔int, int↔float, float↔float, `.rni/.rzi/.rmi/.rpi`, `.sat`), `cvta[.to]` |
| memory | `ld`, `st` (all state spaces, generic, `.v2/.v4`, sub-word sign/zero extension; cache qualifiers accepted), `atom` / `red` (`add exch cas min max and or xor inc dec`, integer and float `add`) |
| control | `bra`, `call` (nvcc `.param` calling convention, recursion), `ret`, `exit`, `trap` |
| synchronisation | `bar.sync` / `barrier.sync` (with optional thread count), `bar.arrive`; `membar`/`fence` are no-ops under SC; `bar.warp.sync` is a no-op |

Syntax accepted includes `.visible/.extern/.weak` linkage, `.align`,
`.ptr`, array and initialiser declarations, `%r<N>` register ranges, nested
`{ }` scopes, labels, `.maxntid`-style performance directives, `.pragma`,
`.loc` and `.file`, and `//` and `/* */` comments.

Video instructions (`vadd vsub vabsdiff vmin vmax vshl vshr vset`, SIMD `v*2`/`v*4`) are modelled best-effort from the ISA text (only `vshr` is hardware-checked); `vmad`, `.mask` destinations and permuting SIMD selectors are not modelled.

**Not (yet) modelled:** warp-synchronous operations (`shfl`, `vote`,
`match`, `activemask`, `redux`), textures and surfaces, `bfe/bfi/prmt`,
`addc/subc` carry chains, `f16x2`/`bf16`, tensor-core instructions, clusters,
asynchronous copies, and the PTX weak memory model (scopes and orders are
accepted but behave as SC). An instruction outside the subset gets stuck as
`#unsupported(...)` rather than being silently mis-executed.

## Floating point

Arithmetic and rounding are not hand-written. They use K's builtin `FLOAT`
module, whose LLVM-backend hooks are implemented with
[GNU MPFR](https://www.mpfr.org/), the standard correctly-rounded
multiple-precision library. Byte-level memory uses K's builtin `BYTES`
module (`Int2Bytes`/`Bytes2Int`/`replaceAtBytes`). The only custom
floating-point code is the bit-level encode/decode of binary16/32/64. K has no
builtin for this, so it is built from exact MPFR operations: every value scaled
by 2^(emin−p+1) is an integer, so all rounding happens in a single MPFR
`roundFloat`. `fma` is computed exactly in a 4400-bit format and rounded
once.

Because that layer is custom, it is checked **differentially against the
host FPU**. `scripts/gen_ieee_diff.py` generates random binary32/binary64
operands (signed zeros, subnormals, infinities, boundary values) and compares
the semantics bit-for-bit with hardware results obtained through Python's
`struct`. It covers `add sub mul div sqrt fma`, widening and narrowing. The f32
FMA reference uses exact rational arithmetic, and the f64 FMA reference is
cross-checked against C `fma()`. This fuzzing found that K's `rootFloat` hook
returns +0 for `sqrt(-0)`, which IEEE-754 says must be −0, and the semantics now
handles that case.

Directed rounding (`.rz .rm .rp`), `.ftz` and `.sat` are modelled for
`add sub mul div fma sqrt rcp` on f16/f32/f64 (`semantics/ptx-float.k`). The
directed modes round an exact (add/sub/mul/fma) or sticky-corrected
(div/sqrt) wide result to the neighbouring floats, and are checked three ways:
against Berkeley TestFloat vectors for `rz/rm/rp` (`tests/testfloat-r*.ptx`),
and against an exact-rational reference that also covers `.ftz`, `.sat` and
signed-zero rules (`scripts/ieee_ref.py`, `tests/rounding-diff-f*.ptx`,
`make fuzz`).

Modelling choices: `.approx`
instructions return the correctly rounded value, which is within PTX's error
bounds. NaN results are PTX's canonical NaN (`0x7FFFFFFF`).
Float-to-integer conversions saturate and map NaN to 0. Integer division by
zero yields all ones (unspecified in PTX).

## Interleaving exploration

`krun --search` explores every schedule. To keep this tractable the semantics
uses a sound partial-order reduction. A thread holding the scheduling `<turn>`
runs its thread-local instructions (registers, control flow, calls, `.param`
and `.local` accesses), which commute with other threads' steps, without
interruption. It yields only before instructions that touch shared state:
global, shared or generic memory, barriers and `exit`. Exhaustively checking
the 4-thread barrier reduction in `examples/reduce-search.ptx` takes about 13
seconds.

```sh
# all reachable outcomes of a racy increment
krun -d build/ptx-kompiled examples/race.ptx --search \
     --pattern '<exit-code> X:Int </exit-code> <out> O </out>'
```

Barrier semantics: a barrier completes when the expected number of threads
has arrived, which is the operand count, or else all *live* threads of the CTA.
Threads that have exited no longer count, as on current GPUs. Mismatched
barriers deadlock (`examples/deadlock.ptx`), and the final configuration shows
the waiting threads.

## Testing

```sh
make test                      # tests/*.ptx vs golden tests/*.out
scripts/run-tests.sh --update  # regenerate golden outputs
make fuzz SEEDS="1 2 3" FUZZ_N=512
make examples
```

CI (`.github/workflows/ci.yml`) installs K from the official release and runs
all three.

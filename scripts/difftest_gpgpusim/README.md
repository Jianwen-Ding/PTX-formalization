# Differential testing: Sail model vs. GPGPU-Sim's PTX functional simulator

This directory builds the PTX front end and functional simulator of
[GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution) (BSD-style
licence) as a stand-alone program and compares it with the Sail model
(`sail/ptx`) on the harness tests and on random programs.

**No NVIDIA software is used.** Nothing from the CUDA toolkit, driver, nvcc,
ptxas or cuobjdump is downloaded, installed or compiled. The few CUDA type
names that GPGPU-Sim's headers mention (`dim3`, `cudaError_t`,
`textureReference`, ...) come from small headers written for this purpose
(`stubs/include/`).

## Files

| file | purpose |
|---|---|
| `build.sh` | checks out GPGPU-Sim at the pinned commit, copies the files listed below, applies `patches/`, runs bison/flex, compiles with `make -j2` (`JOBS=` to change) into `WORKDIR/gpgpusim-ptx` |
| `stubs/include/*.h` | clean-room `vector_types.h`, `driver_types.h`, `texture_types.h`, `device_types.h`, `builtin_types.h` with only the declarations GPGPU-Sim's core uses |
| `stubs/cuda-sim/cuda-math.h` | clean-room replacement of GPGPU-Sim's `src/cuda-sim/cuda-math.h` (which is CUDA-derived and includes NVIDIA's `math_functions.h`): the 24 `cuda_math::` helpers that `instructions.cc` calls, written with `<cmath>`/`<fenv.h>` and the same signatures, so GPGPU-Sim behaves as upstream |
| `stubs/decuda_pred_table.h` | replaces the decuda-derived predicate table (ptxplus only) |
| `stubs/sim_stubs.cc` | link-time stand-ins for the timing model, stream manager, statistics and CUDA runtime front end: a `gpgpu_sim` object that only carries the functional configuration (warp size 32, 2048 threads/SM, POST_DOMINATOR reconvergence), `LOGB2`, CTA-id iteration; everything else is a no-op or aborts if reached |
| `patches/0001-arg_buffer_t-copy-constructor.patch` | fixes a double free in upstream `ptx_ir.h` (the implicit copy of `arg_buffer_t` shared its `.param` buffer), which crashed every `call` with `.param` arguments |
| `driver/gpgpusim_driver.cc` | the driver: parses a PTX file, writes the initialisers of module-scope `.global`/`.const` variables, allocates global buffers with given bytes, launches kernels (grid, block, raw parameter bytes or buffer pointers) on the functional simulator (`gpgpu_cuda_ptx_sim_main_func`) and dumps the buffers |
| `run_compare.py` | runs `tests/*.ptx`-style programs through both simulators and compares the final buffers byte by byte |
| `gen_random.py` | seeded random program generator, fuzz loop and delta minimiser |
| `show_case.py` | prints minimised fuzz cases with the inputs of the differing thread |
| `known_deviations.json` | analysed corpus differences (all GPGPU-Sim deviations); `run_compare.py` reports them as `DIFF-KNOWN` |

### GPGPU-Sim files compiled

Pinned commit `c3eeb6d19016960ab66789efdd8a3200cc30222a` (branch `dev`).

* `src/cuda-sim/`: `cuda-sim.cc`, `cuda_device_printf.cc`, `cuda_device_runtime.cc`,
  `instructions.cc`, `memory.cc`, `ptx-stats.cc`, `ptx_ir.cc`, `ptx_loader.cc`,
  `ptx_parser.cc`, `ptx_sim.cc`, and the bison/flex output of `ptx.y`, `ptx.l`,
  `ptxinfo.y`, `ptxinfo.l`
* `src/abstract_hardware_model.cc`, `src/option_parser.cc`
* headers only: `src/*.h`, `src/gpgpu-sim/*.h`, `libcuda/gpgpu_context.h`,
  `libcuda/cuda_api_object.h`, `libcuda/cuobjdump.h`

Not compiled, not copied: `libcuda/cuda_runtime_api.cc` and
`src/cuda-sim/cuda-math.h` (the two CUDA-derived files named in GPGPU-Sim's
COPYRIGHT), `libcuda/cuda_api.h` (an NVIDIA `cuda.h`),
`src/cuda-sim/decuda_pred_table/` (decuda-derived), and the whole timing model
(`src/gpgpu-sim/*.cc`, `src/intersim2`, `src/accelwattch`).

## Usage

Tools: `git bison flex g++ make patch python3` (Ubuntu: `apt-get install flex bison`).

```sh
scripts/difftest_gpgpusim/build.sh /tmp/gs            # or GPGPUSIM_SRC=/path/to/checkout ... (read-only, must be at the pin)
S=/path/to/PTX-formalization/sail/ptx                 # ptxfront.py and build/ptx (the Sail model is not built here)
scripts/difftest_gpgpusim/run_compare.py --sail-dir $S --gs /tmp/gs/gpgpusim-ptx tests/*.ptx tests/zluda/*.ptx
scripts/difftest_gpgpusim/gen_random.py fuzz --minimise --cats int,float,cvt,mem,ctrl,warp \
    --seeds 0:500 --out /tmp/fz --sail-dir $S --gs /tmp/gs/gpgpusim-ptx
scripts/difftest_gpgpusim/show_case.py /tmp/fz/fuzz.jsonl
```

`run_compare.py` strips the harness lines for GPGPU-Sim, runs the Sail model
once on the `.buffer` lines alone to obtain the initial memory image (so both
start from identical bytes), performs the `.launch` lines in order in both,
and compares every buffer at the end (`.expect` lines are ignored). Programs
with harness features the driver does not model (`.patch`, `.append`,
`.tensormap`, textures, `.ub`, `.memmodel`, dynamic shared memory, ...) are
`SKIP`. Three rewrites only work around GPGPU-Sim's older grammar: entries are
renamed (`dt_k_NAME`, ZLUDA names kernels after opcodes), registers named like
GPGPU-Sim special registers (`%tid`, ...) are renamed, and `.reqntid` is
dropped. Statuses: `MATCH`, `DIFF`, `DIFF-KNOWN`, `SKIP`, `GS-PARSE` (GPGPU-Sim
rejects the module), `GS-FAIL` (GPGPU-Sim aborts at run time), `SAIL-FAIL`.

`gen_random.py` programs: one kernel, 1-2 CTAs of 32 or 64 threads; typed
registers (8 `.b32`, 4 `.b64`, 6 `.f32`, 3 `.f64`, 2 `.b16`, 3 `.pred`) loaded
from per-thread inputs mixing random and special values, 12-40 random
statements, every register stored at the end; a 64-byte private global scratch
area and a shared array are compared too. Categories:

* `int`: add/sub (.sat), mul.lo/hi/wide, mad.lo/hi/wide, mul24/mad24, sad,
  div/rem (non-zero divisor), min/max, abs/neg, and/or/xor/not/cnot, shl/shr,
  bfe, bfi, prmt (all modes), brev, clz, popc, bfind(.shiftamt), setp, selp,
  set, slct, shf, predicate logic; 16-bit and 64-bit variants;
* `float`: add/sub/mul/fma/mad with .rn/.rz/.rm/.rp (.ftz, .sat), div, sqrt,
  rcp (correctly rounded forms), min/max, abs/neg, setp/set with all 14
  comparisons, selp, f64 variants;
* `cvt`: int<->int (narrow destination types, .sat), float->int with
  .rni/.rzi/.rmi/.rpi, int->float with .rn/.rz/.rm/.rp, f32<->f64, integral
  rounding f->f, f16<->f32;
* `mem`: ld/st b8..b64 and v2/v4 on private global scratch, atom (add, min,
  max, inc, dec, and, or, xor, exch, cas, add.f32; 64-bit) on private global and
  shared words, commutative atomics on common shared words (one operation per
  word, result discarded), shared exchange between `bar.sync`;
* `ctrl`: predicated instructions, if / if-else, bounded (possibly divergent)
  loops, nested;
* `warp`: shfl.sync (up/down/bfly/idx, random b and c, optional predicate
  output), vote.sync (all/any/ballot; uni with `--raw`), match.any.sync, at
  nesting depth 0.

Only programs whose result the ISA fully specifies are produced (see the
generator's docstring). Comparison treats any two single-precision NaNs as
equal (ISA 9.7.3: "Single-precision instructions return an unspecified NaN"),
reports NaN-vs-NaN `.f64` differences separately, and by default ignores
+0/-0-only differences (`--zero-sign` to count them). By default the generator
also steers clear of the GPGPU-Sim deviations listed below (operands inside
GPGPU-Sim's correct domain, or the form left out; `--raw` disables this, and
`--all-forms` also emits forms GPGPU-Sim cannot execute at all); every program
carries `.isacheck`, so the Sail front end checks the operand types.

## GPGPU-Sim deviations found (not Sail model bugs)

Integer: `add/sub.sat.s32` do not saturate; `bfe` does not mask/clamp `pos`
and `len` (C shifts by >= 32); `bfi.b64`, `brev.b64` only handle 32 bits;
`bfi.b32` with `len` >= 32; `shf` with shift 0 or 32 (`.clamp`) is wrong;
`shr.s64` by >= 64; `selp` with an immediate predicate; `vote.sync.uni` is
inverted (`or ^ and`); `vote.sync.all` counts exited lanes; `activemask` in
divergent code returns the whole warp; an `atom` whose destination register is
also a source operand (e.g. `atom.cas %r0, [a], %r7, %r0`) reads the
overwritten register; `ld.s8/.s16` into a 32-bit register zero-extends
(ISA 9.4.1 Table 28: sign-extend), and `cvt` to an `.s8/.u8/.s16/.u16`
destination in a 32-bit register neither truncates nor extends the result;
vector `ld` with sink elements; not
implemented: `red`, `addc/subc`, `rem.u16/.s16`, `setp` with a boolean operator, 64-bit
`atom.and/or/xor/min/max`, `testp`, `copysign`, nested register scopes.

Floating point: `.ftz` ignored on most instructions; `.sat` ignored on
`sub/fma` and `cvt`; directed rounding ignored on `rcp`, `sqrt`, `cvt.f32.f64`,
`cvt.f16.f32` (which also rounds ties away), overflow under directed rounding
gives NaN/inf instead of the largest finite value; `neg`/`abs` of +-0 and
`min`/`max` of +0/-0 get the sign of zero wrong; subnormal inputs flushed in
`cvt` (`cvt.rmi.s32.f32 -tiny` gives 0, not -1); float->int `cvt` goes through
a 32-bit signed C conversion (no saturation, NaN not 0, 64-bit results
truncated to 32 bits); `.sat` ignored on integer->integer `cvt`;
`rsqrt/rcp.approx.f64` computed exactly; `div.approx.f32` is an exact division;
f16x2 `cvt`/`fma` only handle one half.

Other: only the first of several `.weak` definitions should win (GPGPU-Sim
takes the last); no `sm_1x` semantics.

# fporacle: an independent floating-point oracle for the PTX Sail model

A seeded differential fuzzer for the floating-point instructions of the Sail model in `sail/ptx`.
It generates PTX programs, runs them on the model, and compares every result bit pattern with a
reference written independently of the model:

* **MPFR** (via the `gmpy2` Python package) for the IEEE 754 formats f64, f32, f16 and bf16:
  correctly rounded `add/sub/mul/fma/div/sqrt` and conversions with the destination precision,
  exponent range and subnormalisation (`subnormalize=True`) in all four directed/nearest modes.
* **Exact rational arithmetic** (`fractions.Fraction`, integer square root) and a rounding routine
  of its own (`formats.round_mag`). Every IEEE result is computed both ways and the two must agree
  (an assertion), so an oracle mistake cannot pass as a model bug. This path also covers `.rna`,
  `.rs`, tf32 and the small formats.
* The small formats are written directly from the **OCP Microscaling Formats (MX) v1.0**
  specification (section 5.3, FP8 E4M3/E5M2 and FP6 E2M3/E3M2 / FP4 E2M1; section 5.4.1, E8M0).
  Where PTX defines the behaviour (`.satfinite`, `.relu`, `.pzo`, NaN handling, rounding modes,
  float-to-int saturation and NaN results), the reference follows the **PTX ISA 9.4** text instead
  (sections 9.7.3, 9.7.4, 9.7.5, 9.7.6 and 9.7.10.24). `ue5m3` is not an OCP format: it is taken
  from PTX ISA 9.4 5.2.3, which does not give its bias; 15 is assumed.

No NVIDIA software, driver, compiler or hardware is used. The only inputs are the two public
specifications and the model itself (its front end `ptxfront.py` and the built `build/ptx`).

## Install

```sh
python3 -m venv fpvenv
fpvenv/bin/pip install gmpy2          # bundles GMP / MPFR wheels on Linux x86-64
```

(If no wheel exists for the platform: install `libgmp-dev libmpfr-dev libmpc-dev` first.)

## Run

```sh
fpvenv/bin/python scripts/fporacle/fuzz.py --list                    # the groups of variants
fpvenv/bin/python scripts/fporacle/fuzz.py --seed 1 --cases 1024 --jobs 2 --json out.json
fpvenv/bin/python scripts/fporacle/fuzz.py --only '^cvt-narrow' --cases 4096
```

By default the model of the main checkout is used: `/home/user/PTX-formalization/sail/ptx/ptxfront.py`
and `.../build/ptx`. Override with `PTX_MODEL_ROOT=<dir containing ptxfront.py and build/ptx>`
(or `PTX_FRONT` / `PTX_MODEL` individually). The fuzzer never builds the model.

Each group (`ops.py`) is a list of instruction variants sharing an operand signature. A program runs
one thread per case; each thread loads its operands from `.buffer`s and executes every variant of
the group, storing each result in its own slot of `OUT`, which is dumped with `.print OUT .u64`.
`--batch` sets the number of cases (threads) per program. The model needs about 0.1 ms per thread
and variant (instruction + store), so a program of 1024 cases x 30 variants runs in about 4 s; the
Python oracle costs about as much again. A full run (`--cases 2048 --jobs 3`, 1589 variants,
3.2 million variant-cases) takes about an hour on four (shared) cores.

For every lane of every result the oracle gives an expectation:

| status | meaning |
|---|---|
| `ok`   | the model's bits equal the reference bits (NaNs: same class where the ISA says "unspecified NaN") |
| `alt`  | the ISA text is ambiguous and the model chose an accepted alternative (the tag names the ambiguity) |
| `soft` | NaN results: a NaN, but not the preferred one (f64 payload propagation, the usual canonical NaN) |
| `any`  | the ISA does not define the result (tag says why); the model's choice is recorded only |
| `FAIL` | disagreement |

Every failing variant is re-run alone, as a one-thread program, and the inputs are simplified byte
by byte while the failure persists; the minimal case is printed and saved in the JSON output.

## Files

* `formats.py` - format descriptions, decoding/encoding, rational rounding (all modes incl. `.rna`, `.rs`)
* `oracle.py`  - exact arithmetic, MPFR cross-check, PTX modifiers (`.ftz .sat .relu .satfinite .pzo`),
  float-to-int, min/max
* `gen.py`     - seeded edge-biased generators (signed zeros, subnormals, min/max normals, ties,
  overflow boundaries, infinities, quiet/signalling NaNs, cancellation)
* `ops.py`     - the instruction variants and their reference semantics, with ISA citations
* `runner.py`  - PTX program builder and model runner
* `fuzz.py`    - driver, comparison, minimisation, report

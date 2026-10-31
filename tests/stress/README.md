# Generated stress tests

`scripts/gen_stress.sh [N] [SEEDS...]` writes these programs (default: 128 cases per launch, seeds 101, 202, 303).
They come from the same generators as the corresponding tests in `tests/` (bits, carry, intx, fmisc, lop3, video,
ieee-diff, rounding-diff, cvt, half, pred, mini, warp) with other seeds and more cases, and, with the first seed,
the matrix generators (mma, mma.sp, wgmma, wgmma.sp, tcgen05.mma, block-scaled mma: `mma-*`, `sp-*`, `wg-*`,
`wgsp-*`, `tc-*`, `mma-bs-*`), whose expected values are exact dense products of the ISA's fragment layouts. Every expected value is
computed by the generator's Python transcription of the PTX ISA (bit-by-bit pseudo-code, big integers, the
exact-rational IEEE reference `scripts/ieee_ref.py`), not recorded from any GPU or vendor tool.

The programs carry `.launch` / `.expect` directives like the rest of `tests/`, so a backend can be checked against
them (for example ZLUDA's `ptx/tools/agx_ptx_formalization_check.py`). They have no `.out` files: they are not run
on the K semantics by `make test`.

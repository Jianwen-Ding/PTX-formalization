# Coverage against the PTX ISA

Status of this K semantics (`semantics/`) relative to the PTX ISA (version 8.x, sm_70 … sm_100a).  Every
implemented family has tests in `tests/` (golden `.out` files; `make test`), most cross-checked against independent
Python models (`scripts/gen_*.py`), ZLUDA hardware-validated programs, or imported IEEE test vectors.  Machine-readable
per-instruction version/target minima are in `semantics/ptx-isa-table.k` (`.isacheck ;` enforces them).

## Implemented (executable)
| Area | Notes |
|---|---|
| Integer arithmetic | add/sub/mul/mad (+ .cc/.hi/.wide/.sat), div/rem, abs/neg/min/max (+ .relu), mul24/mad24, sad, dp4a/dp2a, 16x2 SIMD, popc/clz/brev/bfe/bfi/bfind/fns/bmsk/szext/prmt, lop3, shf, set/setp/selp/slct |
| Floating point | f16/bf16/f32/f64, all rounding modes, .ftz/.sat/.NaN/.xorsign.abs, fma/rcp/sqrt/div/rsqrt, approx transcendentals, testp, cvt matrix incl. tf32/fp8/fp6/fp4/ue8m0 |
| Video | scalar and SIMD `vadd/vsub/vabsdiff/vmin/vmax/vshl/vshr/vset` families |
| Memory | ld/st/atom/red (all spaces, vector, packed, `.b128`), cache hints, `ldu`, `isspacep`, cvta, mapa/getctarank, prefetch/discard/applypriority/createpolicy, alloca |
| Synchronisation | bar/barrier (.sync/.arrive/.red/.aligned), barrier.cluster, mbarrier (full), fence/membar, elect/vote/match/redux/shfl/activemask |
| Matrix | wmma, mma.sync (dense + sparse metadata), ldmatrix/stmatrix/movmatrix, wgmma (dense), tcgen05 (alloc, ld/st, mma, fence/wait/commit) |
| Async | cp.async (+ groups), cp.async.bulk(.tensor) incl. multicast, TMA tile/gather4 loads, stores and reductions, tensormap.*, st.async/red.async/st.bulk, griddepcontrol, clusterlaunchcontrol |
| Control | bra/brx.idx/call (direct, indirect, prototypes)/ret/exit, predication, dynamic stack |
| Textures | tex/tld4/txq/suld/sust/sured/suq on descriptor objects |
| Special registers | all of `%tid … %globaltimer`, cluster registers, `%warpid`, `%smid`-style registers |
| Memory model | default sequentially consistent interleaving (`--search` for all schedules); `.memmodel relaxed;` store-buffer mode with litmus suite |
| Legality | `.version`/`.target` minima for 103 base instructions (`.isacheck ;`) |

## Not modelled
* `multimem.*`, `fabric.*` (need multi-device objects).
* Textures: cubemaps, mipmaps/`tex.level/.grad` LOD hardware, depth compare, formatted surface accesses (`.p`).
* `wgmma.mma_async.sp`; tcgen05 `.cta_group::2` for sparse/block-scaled kinds, `.decompress::lut::b`, `kind::ti16`.
* TMA: interleaved layouts, `im2col` modes, `.override` forms.  (The tensor-map encoding is private to this semantics, not the hardware 128-byte format.)
* `vmad`, video `.mask` destinations, permuting SIMD selectors; `istypep`.
* `.multicast::cluster::32b` for mbarrier; generic-window cluster addresses.
* Relaxed memory mode: load-buffering reorderings, scope hierarchy, proxies; asynchronous operations execute atomically at issue.
* Hardware-defined layouts that are not specified in the ISA document are assumptions, listed in `docs/prior-work.md`.

## Language front end
Vector registers (`.reg .v2/.v4`, also arrays), `.reg` formal parameters and return values (by value, vectors flattened) and
register shadowing in nested blocks (renaming `name@id`) are supported.  Known gaps: expression operands such as
`st [x], temp + 1` (a ZLUDA extension, not PTX), old non-`.sync` `vote`, and `.v8` component names (`s0`..`s7`, assumed).

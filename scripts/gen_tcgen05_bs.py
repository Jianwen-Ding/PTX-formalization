#!/usr/bin/env python3
"""Generate tests/tcbs-*.ptx: tcgen05.mma.cta_group::1.kind::{mxf8f6f4,mxf4,mxf4nvf4}.block_scale against an exact model.
Scale-factor placement in tensor memory follows the layout assumed in semantics/ptx-exec-tc.k (docs/prior-work.md).
usage: gen_tcgen05_bs.py"""
import random, sys
from fractions import Fraction
here = __file__.rsplit('/', 1)[0]
s = open(here + '/gen_tcgen05_x.py').read()
exec(compile(s[:s.index('def zmask')], 'gen_x_prefix', 'exec'))
random.seed(7)
def ue8m0(x): return Fraction(2) ** (x - 127)
def ue4m3(x):
    e, m = x >> 3, x & 7
    return Fraction(m, 8) * Fraction(2) ** (1 - 7) if e == 0 else Fraction(8 + m, 8) * Fraction(2) ** (e - 7)
SC = {'ue8m0': (ue8m0, lambda: random.randint(122, 131)), 'ue4m3': (ue4m3, lambda: random.randint(0x30, 0x48))}
# name, kind, qualifiers, atype, btype, N, K, block, scale type, SFA id, SFB id, enable-input-d, K-major z
CFG = [
 ('mxf8', 'kind::mxf8f6f4', '', 'e4m3', 'e2m3', 8, 32, 32, 'ue8m0', 2, 1, 1, 1),
 ('mxf4', 'kind::mxf4', '.scale_vec::2X', 'e2m1', 'e2m1', 8, 64, 32, 'ue8m0', 0, 2, 0, 0),
 ('nvf4', 'kind::mxf4nvf4', '.scale_vec::4X', 'e2m1', 'e2m1', 8, 64, 16, 'ue4m3', 0, 0, 0, 0),
]
M = 128
for kk, (nm, kind, extra, at, bt, N, K, blk, sty, sfa, sfb, en, z) in enumerate(CFG):
    w = 4 if at == 'e2m1' and kind != 'kind::mxf8f6f4' else 8
    sf, sg = SC[sty]; nb = K // blk
    A = [[rand_float_bits(at) for _ in range(K)] for _ in range(M)]
    B = [[rand_float_bits(bt) for _ in range(K)] for _ in range(N)]
    SA = [[sg() for _ in range(nb)] for _ in range(M)]
    SB = [[sg() for _ in range(nb)] for _ in range(N)]
    Din = {(r, c): rand_float_bits('f32') for r in range(M) for c in range(N)}
    Dout = {}
    for r in range(M):
        for c in range(N):
            acc = fval('f32', Din[(r, c)]) if en else Fraction(0)
            acc += sum(fval(at, A[r][k]) * fval(bt, B[c][k]) * sf(SA[r][k // blk]) * sf(SB[c][k // blk]) for k in range(K))
            Dout[(r, c)] = to_bits('f32', acc)
    a_img, la, sa_ = build_tile(A, M, K, w, True, z, 0, at if kind == 'kind::mxf8f6f4' else None)
    descA = desc100(0, la, sa_, z)
    off = (max(a_img) + 1 + 1023) // 1024 * 1024
    b_img, lb, sb_ = build_tile(B, N, K, w, True, z, off, bt if kind == 'kind::mxf8f6f4' else None)
    img = dict(a_img); img.update(b_img); descB = desc100(off, lb, sb_, z)
    size = (max(img) + 1 + 2047) // 2048 * 2048
    data = bytes(img.get(i, 0) for i in range(size))
    TC = {'e4m3': 0, 'e5m2': 1, 'e2m3': 3, 'e3m2': 4, 'e2m1': 1 if w == 4 else 5}
    idesc = (sfb << 4) | (TC[at] << 7) | (TC[bt] << 10) | ((N >> 3) << 17) | ((1 if sty == 'ue8m0' else 0) << 23) | ((M >> 7) << 27) | (sfa << 29)
    if kind == 'kind::mxf4nvf4': idesc &= ~(3 << 23)
    # scale words (little endian bytes SFA_ID + block)
    wa = [sum(SA[r][j] << (8 * (sfa + j)) for j in range(nb)) for r in range(M)]
    wb = [sum(SB[c][j] << (8 * (sfb + j)) for j in range(nb)) for c in range(N)] + [0] * (128 - N)
    nd = N; SAC, SBC = 300, 310
    L = HDR[:]
    L.append(f""".visible .entry k{kk}(.param .u64 pimg, .param .u64 pd, .param .u64 psa, .param .u64 psb, .param .u64 po)
{{
    .reg .b32 %r<8>, %x<4>, %D<{nd}>, %S<2>; .reg .b64 %rd<8>; .reg .b64 %dA, %dB; .reg .b32 %idesc, %tl, %tw;
    .reg .pred %p, %q;
    .shared .align 1024 .b8 buf[{size}];
    .shared .align 16 .b32 slot[2];
    mov.u32 %tl, %tid.x;
    mov.u32 %r3, 0; mov.u32 %r4, 0;
    mov.u64 %rd5, slot;
    setp.lt.u32 %p, %tl, 32;
    setp.eq.u32 %q, %tl, 0;
    @%p tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd5], 512;
    ld.param.u64 %rd0, [pimg]; mov.u64 %rd2, buf; mul.wide.u32 %rd1, %tl, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;""")
    for it in range(size // 2048):
        o = it * 2048
        L.append(f"    ld.global.b32 %x0, [%rd0+{o}]; ld.global.b32 %x1, [%rd0+{o + 4}]; ld.global.b32 %x2, [%rd0+{o + 8}]; ld.global.b32 %x3, [%rd0+{o + 12}];")
        L.append(f"    st.shared.b32 [%rd2+{o}], %x0; st.shared.b32 [%rd2+{o + 4}], %x1; st.shared.b32 [%rd2+{o + 8}], %x2; st.shared.b32 [%rd2+{o + 12}], %x3;")
    L.append("    bar.sync 0;")
    L.append("    shr.u32 %tw, %tl, 5; shl.b32 %tw, %tw, 21;")
    L.append(f"    ld.param.u64 %rd0, [pd]; mul.wide.u32 %rd1, %tl, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    ld.global.b32 %D{i}, [%rd0+{i * 4}];")
    L.append(f"    tcgen05.st.sync.aligned.32x32b.x{nd}.b32 [%tw], {vec('D', nd)};")
    L.append(f"    ld.param.u64 %rd0, [psa]; mul.wide.u32 %rd1, %tl, 4; add.u64 %rd0, %rd0, %rd1; ld.global.b32 %S0, [%rd0];")
    L.append(f"    add.u32 %r1, %tw, {SAC};\n    tcgen05.st.sync.aligned.32x32b.x1.b32 [%r1], {{%S0}};")
    L.append(f"    ld.param.u64 %rd0, [psb]; add.u64 %rd0, %rd0, %rd1; ld.global.b32 %S1, [%rd0];")
    L.append(f"    add.u32 %r1, %tw, {SBC};\n    tcgen05.st.sync.aligned.32x32b.x1.b32 [%r1], {{%S1}};")
    L.append("    tcgen05.wait::st.sync.aligned;\n    tcgen05.fence::before_thread_sync;\n    bar.sync 0;")
    L.append(f"    mov.b64 %dB, {descB};\n    mov.b64 %dA, {descA};\n    mov.b32 %idesc, {idesc};\n    mov.u32 %r5, {SAC};\n    mov.u32 %r6, {SBC};")
    L.append(f"    @%q tcgen05.mma.cta_group::1.{kind}.block_scale{extra} [%r3], %dA, %dB, %idesc, [%r5], [%r6], {en};")
    L.append("    tcgen05.fence::after_thread_sync;\n    bar.sync 0;")
    L.append(f"    tcgen05.ld.sync.aligned.32x32b.x{nd}.b32 {vec('D', nd)}, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
    L.append(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %tl, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    st.global.b32 [%rd0+{i * 4}], %D{i};")
    L.append("    bar.sync 0;\n    @%p tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 512;\n    ret;\n}\n")
    code = "\n".join(L)
    flat = lambda rows: [x for r in rows for x in r]
    code += f"\n.buffer IMG .u8 [{len(data)}];\n"
    i = 0
    while i < len(data):
        if data[i] == 0: i += 1; continue
        j = i
        while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
        code += f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};\n"
        i = j
    code += buf('DIN', 'u32', flat([[Din[(u, c)] for c in range(N)] for u in range(128)])) + "\n"
    code += buf('SAI', 'u32', wa) + "\n" + buf('SBI', 'u32', wb) + "\n"
    code += f".buffer O .u32 [{128 * nd}];\n.launch k{kk} <<< 1, 128 >>> (IMG, DIN, SAI, SBI, O);\n" + expect('O', 'u32', flat([[Dout[(u, c)] for c in range(N)] for u in range(128)])) + "\n"
    open(f'tests/tcbs-{kk}.ptx', 'w').write(code)
print(len(CFG), "tests")

#!/usr/bin/env python3
"""Generate tests/tcg2-*.ptx: tcgen05.mma.cta_group::2 on a CTA pair (cluster of 2) against an exact model.
Each CTA supplies M/2 rows of A and N/2 columns of B and receives M/2 rows of D (see semantics/ptx-exec-tc.k).
usage: gen_tcgen05_g2.py"""
import random, sys
from fractions import Fraction
here = __file__.rsplit('/', 1)[0]
s = open(here + '/gen_tcgen05_x.py').read()
exec(compile(s[:s.index('def zmask')], 'gen_x_prefix', 'exec'))
random.seed(11)
# name, M, N, K, dtype, atype, btype, A form
CFG = [('g2a', 128, 8, 16, 'f32', 'f16', 'f16', 'desc'),
       ('g2b', 256, 8, 16, 'f32', 'bf16', 'bf16', 'desc'),
       ('g2c', 128, 8, 16, 'f32', 'f16', 'f16', 'tmem')]
TC = {'f16': 0, 'bf16': 1}
for kk, (nm, M, N, K, dt, at, bt, aform) in enumerate(CFG):
    Mh, Nh = M // 2, N // 2
    A = [[rand_float_bits(at) for _ in range(K)] for _ in range(M)]
    B = [[rand_float_bits(bt) for _ in range(K)] for _ in range(N)]
    Din = {(r, c): 0 for r in range(M) for c in range(N)}
    Dout = {(r, c): to_bits(dt, fval(dt, Din[(r, c)]) + sum(fval(at, A[r][k]) * fval(bt, B[c][k]) for k in range(K))) for r in range(M) for c in range(N)}
    imgs = []; descA = descB = 0
    for c in range(2):
        img = {}; off = 0
        if aform == 'desc':
            a_img, la, sa_ = build_tile(A[c * Mh:(c + 1) * Mh], Mh, K, 16, True, 1, 0)
            img.update(a_img); descA = desc100(0, la, sa_, 1); off = (max(a_img) + 1 + 1023) // 1024 * 1024
        b_img, lb, sb_ = build_tile(B[c * Nh:(c + 1) * Nh], Nh, K, 16, True, 1, off)
        img.update(b_img); descB = desc100(off, lb, sb_, 1)
        imgs.append(img)
    size = (max(max(i) for i in imgs) + 1 + 2047) // 2048 * 2048
    data = b"".join(bytes(i.get(j, 0) for j in range(size)) for i in imgs)
    idesc = ({'f16': 0, 'f32': 1}[dt] << 4) | (TC[at] << 7) | (TC[bt] << 10) | ((N >> 3) << 17) | ((M >> 4) << 24)
    per = 2; na = K // per
    awords = [[sum(A[r][w * per + j] << (16 * j) for j in range(per)) for w in range(na)] for r in range(M)]
    nd = N
    L = HDR[:]
    L.append(f""".visible .entry k{kk}(.param .u64 pimg, .param .u64 pd, .param .u64 pa, .param .u64 po) .reqnctapercluster 2, 1, 1
{{
    .reg .b32 %r<8>, %x<4>, %D<{nd}>, %A<{na}>, %cta; .reg .b64 %rd<8>; .reg .b64 %dA, %dB; .reg .b32 %idesc, %tl, %tw;
    .reg .pred %p, %q;
    .shared .align 1024 .b8 buf[{size}];
    .shared .align 16 .b32 slot[2];
    mov.u32 %tl, %tid.x;
    mov.u32 %cta, %ctaid.x;
    mov.u32 %r3, 0; mov.u32 %r4, 0;
    mov.u64 %rd5, slot;
    setp.lt.u32 %p, %tl, 32;
    setp.eq.u32 %q, %tl, 0;
    setp.eq.u32 %q, %tl, 0;
    @%p tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%rd5], 512;
    ld.param.u64 %rd0, [pimg]; mul.wide.u32 %rd6, %cta, {size}; add.u64 %rd0, %rd0, %rd6;
    mov.u64 %rd2, buf; mul.wide.u32 %rd1, %tl, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;""")
    for it in range(size // 2048):
        o = it * 2048
        L.append(f"    ld.global.b32 %x0, [%rd0+{o}]; ld.global.b32 %x1, [%rd0+{o + 4}]; ld.global.b32 %x2, [%rd0+{o + 8}]; ld.global.b32 %x3, [%rd0+{o + 12}];")
        L.append(f"    st.shared.b32 [%rd2+{o}], %x0; st.shared.b32 [%rd2+{o + 4}], %x1; st.shared.b32 [%rd2+{o + 8}], %x2; st.shared.b32 [%rd2+{o + 12}], %x3;")
    L.append("    bar.sync 0;")
    L.append(f"    shr.u32 %tw, %tl, 5; shl.b32 %tw, %tw, 21;")
    L.append(f"    ld.param.u64 %rd0, [pd]; mul.wide.u32 %rd6, %cta, {128 * nd * 4}; add.u64 %rd0, %rd0, %rd6; mul.wide.u32 %rd1, %tl, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    ld.global.b32 %D{i}, [%rd0+{i * 4}];")
    L.append(f"    tcgen05.st.sync.aligned.32x32b.x{nd}.b32 [%tw], {vec('D', nd)};")
    if aform == 'tmem':
        L.append(f"    ld.param.u64 %rd0, [pa]; mul.wide.u32 %rd6, %cta, {128 * na * 4}; add.u64 %rd0, %rd0, %rd6; mul.wide.u32 %rd1, %tl, {na * 4}; add.u64 %rd0, %rd0, %rd1;")
        for i in range(na): L.append(f"    ld.global.b32 %A{i}, [%rd0+{i * 4}];")
        L.append(f"    add.u32 %r1, %tw, 256;\n    tcgen05.st.sync.aligned.32x32b.x{na}.b32 [%r1], {vec('A', na)};")
    L.append("    tcgen05.wait::st.sync.aligned;\n    tcgen05.fence::before_thread_sync;\n    bar.sync 0;")
    L.append("    barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;")
    L.append(f"    mov.b64 %dB, {descB};\n    mov.b32 %idesc, {idesc};")
    if aform == 'desc': L.append(f"    mov.b64 %dA, {descA};")
    else: L.append("    mov.u32 %r2, 256;")
    a_op = "%dA" if aform == 'desc' else "[%r2]"
    L.append("    setp.eq.u32 %p, %cta, 0;\n    and.pred %q, %q, %p;")
    L.append(f"    @%q tcgen05.mma.cta_group::2.kind::{'f16'} [%r3], {a_op}, %dB, %idesc, 1;")
    L.append("    tcgen05.fence::after_thread_sync;\n    bar.sync 0;\n    barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;")
    L.append(f"    tcgen05.ld.sync.aligned.32x32b.x{nd}.b32 {vec('D', nd)}, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
    L.append(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd6, %cta, {128 * nd * 4}; add.u64 %rd0, %rd0, %rd6; mul.wide.u32 %rd1, %tl, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    st.global.b32 [%rd0+{i * 4}], %D{i};")
    L.append("    bar.sync 0;\n    barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;\n    @%p tcgen05.dealloc.cta_group::2.sync.aligned.b32 %r4, 512;\n    ret;\n}\n")
    code = "\n".join(L)
    flat = lambda rows: [x for r in rows for x in r]
    ain = []; dout = []
    for c in range(2):
        for u in range(128):
            if u < Mh: dout.append([Dout[(c * Mh + u, j)] for j in range(N)])
            ain.append(awords[c * Mh + u] if u < Mh else [0] * na)
    code += f"\n.buffer IMG .u8 [{len(data)}];\n"
    i = 0
    while i < len(data):
        if data[i] == 0: i += 1; continue
        j = i
        while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
        code += f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};\n"
        i = j
    code += f".buffer DIN .u32 [{256 * nd}];" + "\n" + buf('AIN', 'u32', flat(ain) if aform == 'tmem' else [0]) + "\n"
    code += f".buffer O .u32 [{256 * nd}];\n.launch k{kk} <<< 2, 128 >>> (IMG, DIN, AIN, O);\n"
    for c in range(2):
        rows = flat([[Dout[(c * Mh + u, j)] for j in range(N)] for u in range(Mh)])
        code += expect('O', 'u32', rows).replace('@ ', '@ ', 1) if False else "\n".join(f".expect O .u32 @ {c * 128 * nd + i} = {{ {h(rows[i:i + CH])} }};" for i in range(0, len(rows), CH)) + "\n"
    open(f'tests/tcg2-{kk}.ptx', 'w').write(code)
print(len(CFG), "tests")

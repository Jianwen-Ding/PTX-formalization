#!/usr/bin/env python3
"""Generate tests/tcsp-*.ptx: tcgen05.mma.sp against a dense model.  The packed A matrix and the metadata words
(row r in TMEM lane r, 4 bits per chunk; one word selected by the sparsity selector for 16-bit/tf32 kinds, two words
for 8-bit kinds -- the layout assumption of semantics/ptx-exec-tc.k) are built from a randomly pruned dense A.
usage: gen_tcgen05_sp.py [SEED]"""
import random, sys, struct
from fractions import Fraction
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_wgmma.py').read().split('# name: (N, K')[0]
sys.argv = [sys.argv[0]] + sys.argv[1:2]
exec(compile(src, 'gen_wgmma_prefix', 'exec'))
t5 = open(here + '/gen_tcgen05.py').read()
exec(compile(t5[t5.index('h = lambda xs'):t5.index('# ---------------------------------------------------------------- alloc')], 'gen_tcgen05_prefix', 'exec'))
ssrc = open(here + '/gen_mmasp.py').read()
exec(compile(ssrc[ssrc.index('def chunking'):ssrc.index('def regions')], 'gen_mmasp_a', 'exec'))
exec(compile(ssrc[ssrc.index('def nz_choice'):ssrc.index('def build_sp')], 'gen_mmasp_b', 'exec'))
KIND = {'f16': 'kind::f16', 'tf32': 'kind::tf32', 'i8': 'kind::i8', 'f8f6f4': 'kind::f8f6f4'}
TCODE = {'f16': 0, 'bf16': 1, 'tf32': 2, 'u8': 0, 's8': 1, 'e4m3': 0, 'e5m2': 1}
# (N, Ksparse, kind, dtype, atype, btype, aform, akm, az, bkm, bz, enable, scale, negA, negB, M, sel, sat)
CFG = [
 (8, 32, 'f16', 'f32', 'f16', 'f16', 'desc', True, 1, True, 1, 1, 0, 0, 0, 128, 0, 0),
 (8, 32, 'f16', 'f32', 'bf16', 'bf16', 'tmem', True, 0, False, 2, 1, 0, 0, 1, 128, 2, 0),
 (16, 16, 'tf32', 'f32', 'tf32', 'tf32', 'desc', True, 3, True, 2, 1, 0, 1, 0, 64, 3, 0),
 (8, 64, 'i8', 's32', 'u8', 's8', 'desc', True, 1, True, 1, 1, 0, 0, 0, 128, 0, 1),
 (8, 64, 'f8f6f4', 'f32', 'e4m3', 'e5m2', 'tmem', True, 0, True, 1, 1, 0, 0, 0, 64, 0, 0),
 (8, 32, 'f16', 'f16', 'f16', 'f16', 'desc', False, 2, False, 3, 1, 1, 0, 0, 128, 1, 0),
]
for kk, (N, K, kind, dt, at, bt, aform, akm, az, bkm, bz, en, sc, na, nb, M, sel, sat) in enumerate(CFG):
    wa, wb = bw(at), bw(bt)
    small = (dt == 'f16')
    cw, nz = chunking(at)
    nch = K // cw
    def rnd(ty):
        if small and isflt(ty):
            v = random.choice([-2, -1, -0.5, 0, 0.5, 1, 2, 3]); return struct.unpack('<H', struct.pack('<e', v))[0]
        if isflt(ty): return rand_float_bits(ty)
        return random.getrandbits(bw(ty))
    choice = {(r, c): random.choice(nz_choice(at)) for r in range(M) for c in range(nch)}
    A = {}
    packed = [[] for _ in range(M)]
    for (r, c), ch in sorted(choice.items()):
        for col in range(cw): A[(r, cw * c + col)] = 0
        cols = sorted(dense_cols(at, ch))
        for col in cols:
            v = rnd(at); A[(r, cw * c + col)] = v; packed[r].append(v)
    B = [[rnd(bt) for _ in range(K)] for _ in range(N)]
    Din = {(r, c): (struct.unpack('<H', struct.pack('<e', random.choice([-1, 0, 1, 2, 4])))[0] if small else rand_float_bits('f32')) if dt != 's32'
                    else random.choice([0, 1, 100, 0xFFFFFFF0, 0x7FFFFFF0, random.getrandbits(32)]) for r in range(M) for c in range(N)}
    neg = lambda ty, bits, s: bits ^ (1 << (bw(ty) - 1)) if (s and isflt(ty)) else bits
    Dout = {}
    for r in range(M):
        for c in range(N):
            if isflt(at):
                acc = fval(dt, Din[(r, c)]) / (2 ** sc) if en else Fraction(0)
                s = acc + sum(fval(at, neg(at, A[(r, k)], na)) * fval(bt, neg(bt, B[c][k], nb)) for k in range(K))
                Dout[(r, c)] = to_bits(dt, s)
            else:
                va = (lambda x: sx(x, 8)) if at[0] == 's' else (lambda x: x)
                vb = (lambda x: sx(x, 8)) if bt[0] == 's' else (lambda x: x)
                s = (sx(Din[(r, c)], 32) if en else 0) + sum(va(A[(r, k)]) * vb(B[c][k]) for k in range(K))
                if sat: s = max(-(1 << 31), min((1 << 31) - 1, s))
                Dout[(r, c)] = s & 0xFFFFFFFF
    # metadata words per row
    nw = 1 if nch <= 8 else 2
    meta = []
    for r in range(M):
        words = [sum(nibble(at, choice[(r, c)]) << (4 * (c - 8 * w)) for c in range(8 * w, min(nch, 8 * w + 8))) for w in range(nw)]
        meta.append(words)
    MC = 384
    img = {}; off = 0; descA = 0
    if aform == 'desc':
        a_img, la, sa_ = build_tile(packed, M, K // 2, wa, akm, az, 0)
        img.update(a_img); descA = desc100(0, la, sa_, az)
        off = (max(a_img) + 1 + 1023) // 1024 * 1024
    b_img, lb, sb_ = build_tile(B, N, K, wb, bkm, bz, off)
    img.update(b_img); descB = desc100(off, lb, sb_, bz)
    size = (max(img) + 1 + 2047) // 2048 * 2048
    data = bytes(img.get(i, 0) for i in range(size))
    idesc = (sel & 3) | (1 << 2) | (sat << 3) | ({'f16': 0, 'f32': 1, 's32': 2}[dt] << 4) | (TCODE[at] << 7) | (TCODE[bt] << 10) | (na << 13) | (nb << 14) \
          | ((0 if akm else 1) << 15) | ((0 if bkm else 1) << 16) | ((N >> 3) << 17) | ((M >> 4) << 24)
    per = 32 // wa
    kp = K // 2
    awords = [[sum(packed[r][w * per + j] << (wa * j) for j in range(per)) for w in range(kp // per)] for r in range(M)]
    na_cols = kp // per
    nd = N
    ncol_meta = 4 if nw == 1 else 2
    mwords = []
    for r in range(128):
        row = [random.getrandbits(32) for _ in range(ncol_meta)] if r < M else [0] * ncol_meta     # unselected columns hold garbage
        if r < M:
            for w in range(nw): row[(sel if nw == 1 else w)] = meta[r][w]
        mwords.append(row)
    L = HDR[:]
    L.append(f""".visible .entry k{kk}(.param .u64 pimg, .param .u64 pd, .param .u64 pa, .param .u64 pm, .param .u64 po)
{{
    .reg .b32 %r<8>, %x<4>, %D<{nd}>, %A<{max(na_cols, 1)}>, %E<{ncol_meta}>; .reg .b64 %rd<8>; .reg .b64 %dA, %dB; .reg .b32 %idesc, %tl, %tw, %m0, %m1, %m2, %m3;
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
    L.append(f"    shr.u32 %tw, %tl, 5; shl.b32 %tw, %tw, 21;")
    L.append(f"    ld.param.u64 %rd0, [pd]; mul.wide.u32 %rd1, %tl, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    ld.global.b32 %D{i}, [%rd0+{i * 4}];")
    L.append(f"    tcgen05.st.sync.aligned.32x32b.x{nd}.b32 [%tw], {vec('D', nd)};")
    if aform == 'tmem':
        L.append(f"    ld.param.u64 %rd0, [pa]; mul.wide.u32 %rd1, %tl, {na_cols * 4}; add.u64 %rd0, %rd0, %rd1;")
        for i in range(na_cols): L.append(f"    ld.global.b32 %A{i}, [%rd0+{i * 4}];")
        L.append(f"    add.u32 %r1, %tw, 256;\n    tcgen05.st.sync.aligned.32x32b.x{na_cols}.b32 [%r1], {vec('A', na_cols)};")
    L.append(f"    ld.param.u64 %rd0, [pm]; mul.wide.u32 %rd1, %tl, {ncol_meta * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(ncol_meta): L.append(f"    ld.global.b32 %E{i}, [%rd0+{i * 4}];")
    L.append(f"    add.u32 %r5, %tw, {MC};\n    tcgen05.st.sync.aligned.32x32b.x{ncol_meta}.b32 [%r5], {vec('E', ncol_meta)};")
    L.append("    tcgen05.wait::st.sync.aligned;\n    tcgen05.fence::before_thread_sync;\n    bar.sync 0;")
    L.append(f"    mov.b64 %dB, {descB};\n    mov.b32 %idesc, {idesc};")
    for i in range(4): L.append(f"    mov.b32 %m{i}, 0;")
    a_op = "%dA" if aform == 'desc' else "[%r2]"
    if aform == 'desc': L.append(f"    mov.b64 %dA, {descA};")
    else: L.append("    mov.u32 %r2, 256;")
    L.append(f"    mov.u32 %r6, {MC};")
    scale_arg = f", {sc}" if kind in ('f16', 'tf32') else ""
    L.append(f"    @%q tcgen05.mma.sp.cta_group::1.{KIND[kind]} [%r3], {a_op}, %dB, [%r6], %idesc, {{%m0, %m1, %m2, %m3}}, {en}{scale_arg};")
    L.append("    tcgen05.fence::after_thread_sync;\n    bar.sync 0;")
    L.append(f"    tcgen05.ld.sync.aligned.32x32b.x{nd}.b32 {vec('D', nd)}, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
    L.append(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %tl, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    st.global.b32 [%rd0+{i * 4}], %D{i};")
    L.append("    bar.sync 0;\n    @%p tcgen05.dealloc.cta_group::1.sync.aligned.b32 %r4, 512;\n    ret;\n}\n")
    code = "\n".join(L)
    flat = lambda rows: [x for r in rows for x in r]
    din = [[Din[(u, c)] if u < M else 0 for c in range(N)] for u in range(128)]
    dout = [[Dout[(u, c)] if u < M else 0 for c in range(N)] for u in range(128)]
    code += f"\n.buffer IMG .u8 [{len(data)}];\n"
    i = 0
    while i < len(data):
        if data[i] == 0: i += 1; continue
        j = i
        while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
        code += f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};\n"
        i = j
    code += buf('DIN', 'u32', flat(din)) + "\n"
    code += buf('AIN', 'u32', flat(awords + [[0] * na_cols for _ in range(128 - M)]) if aform == 'tmem' else [0]) + "\n"
    code += buf('MIN', 'u32', flat(mwords)) + "\n"
    code += f".buffer O .u32 [{128 * nd}];\n.launch k{kk} <<< 1, 128 >>> (IMG, DIN, AIN, MIN, O);\n" + expect('O', 'u32', flat(dout)) + "\n"
    open(f'tests/tcsp-{kk}.ptx', 'w').write(code)
print(len(CFG), "tests")

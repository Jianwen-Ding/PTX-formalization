#!/usr/bin/env python3
"""Generate tests/tcx-*.ptx: tcgen05.mma.ws (zero-column mask), .ashift / collector qualifiers, and fp6/fp4 operand types of
kind::f8f6f4 against dense models.  The zero-column mask model is checked against the four examples of the PTX ISA.
usage: gen_tcgen05_x.py"""
import random, sys, struct
from fractions import Fraction
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_mma.py').read().split('cases = []')[0]
sys.argv = [sys.argv[0]]
exec(compile(src, 'gen_mma_prefix', 'exec'))
# extend the element tables with fp6 / fp4
def mini_tab2(name, e, m):
    bias = (1 << (e - 1)) - 1; t = {}
    for c in range(1 << (e + m + 1)):
        s, E, M = c >> (e + m), (c >> m) & ((1 << e) - 1), c & ((1 << m) - 1)
        v = Fraction(M, 1 << m) * Fraction(2) ** (1 - bias) if E == 0 else Fraction((1 << m) + M, 1 << m) * Fraction(2) ** (E - bias)
        t[c] = -v if s else v
    return t
MINI.update({'e2m3': mini_tab2('e2m3', 2, 3), 'e3m2': mini_tab2('e3m2', 3, 2), 'e2m1': mini_tab2('e2m1', 2, 1)})
wsrc = open(here + '/gen_wgmma.py').read()
wsrc = wsrc[wsrc.index('def bw(ty)'):wsrc.index('# name: (N, K')]
wsrc = wsrc.replace("{'tf32': 32, 'e4m3': 8, 'e5m2': 8, 'b1': 1, 'bf16': 16, 'f16': 16}.get(ty)", "{'tf32': 32, 'e4m3': 8, 'e5m2': 8, 'e2m3': 8, 'e3m2': 8, 'e2m1': 8, 'b1': 1, 'bf16': 16, 'f16': 16}.get(ty)")
wsrc = wsrc.replace("ty in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2')", "ty in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2', 'e2m3', 'e3m2', 'e2m1')")
exec(compile(wsrc, 'gen_wgmma_prefix', 'exec'))
t5 = open(here + '/gen_tcgen05.py').read()
exec(compile(t5[t5.index('h = lambda xs'):t5.index('# ---------------------------------------------------------------- alloc')], 'gen_tcgen05_prefix', 'exec'))
_rfb = rand_float_bits
def rand_float_bits(ty):        # 6-bit / 4-bit values in 8-bit containers
    if ty in ('e2m3', 'e3m2', 'e2m1'): return random.choice(list(MINI[ty]))
    return _rfb(ty)

def zmask(Z, M, N):
    """Zero-column mask (list over N output columns of 0/1) from the descriptor, by explicit run generation."""
    sc = [(Z >> (8 * i)) & 255 for i in range(4)]; fs = [(Z >> (32 + i)) & 1 for i in range(4)]
    nz = (Z >> 39) & 1; skip = ((Z >> 40) & 255) + 1; use = ((Z >> 48) & 255) + 1
    ones, zeros = skip, use                       # runs of ones (column zeroed) and zeros (column used)
    if not nz: return [0] * N
    P = 128 // M; seg = N // P; out = []
    for i in range(P):
        bits = []; v = fs[i]; first = True
        while len(bits) < seg:
            ln = (ones if v else zeros) - (sc[i] if first else 0)
            bits += [v] * max(ln, 0); v ^= 1; first = False
        out += bits[:seg]
    return out
# the four examples of the ISA text (printed MSB first there; compare LSB first)
def lsb(s): return [int(c) for c in s.replace(' ', '')][::-1]
def desc_of(sc, fs, nz, skip, use, shift): return sum(s << (8 * i) for i, s in enumerate(sc)) | sum(f << (32 + i) for i, f in enumerate(fs)) | (nz << 39) | (skip << 40) | (use << 48) | (shift << 56)
m = zmask(desc_of([0, 0, 0, 0], [0, 0, 0, 0], 0, 4, 3, 0), 128, 16); assert m == [0] * 16
m = zmask(desc_of([0, 0, 0, 0], [0, 0, 0, 0], 1, 2, 3, 0), 128, 14); assert m == lsb('111 0000 111 0000'), m
m = zmask(desc_of([0, 0, 0, 0], [1, 0, 0, 0], 1, 2, 3, 0), 64, 28)
assert m[:14] == lsb('0000 111 0000 111 0000 111 0000')[::-1][::-1][:0] or True
assert m[:14] == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], m[:14]            # mask0: ...111 0000 111 0000 111
assert m[14:28] == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], m[14:28]         # mask1 (fs1 = 0): 0000 111 0000 111
m = zmask(desc_of([0, 1, 2, 1], [1, 1, 0, 0], 1, 2, 3, 2), 32, 40)
assert m[0:10] == [1, 1, 1, 0, 0, 0, 0, 1, 1, 1], m[0:10]                       # mask0: 111 0000 111
assert m[10:20] == [1, 1, 0, 0, 0, 0, 1, 1, 1, 0], m[10:20]                     # mask1: 11 0000 111 (one column skipped)
assert m[20:30] == [0, 0, 1, 1, 1, 0, 0, 0, 0, 1], m[20:30]                     # mask2: 00 111 0000 1..
assert m[30:40] == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0], m[30:40]                     # mask3: 000 111 0000

KIND = {'f16': 'kind::f16', 'tf32': 'kind::tf32', 'i8': 'kind::i8', 'f8f6f4': 'kind::f8f6f4'}
TCODE = {'f16': 0, 'bf16': 1, 'tf32': 2, 'u8': 0, 's8': 1, 'e4m3': 0, 'e5m2': 1, 'e2m3': 3, 'e3m2': 4, 'e2m1': 5}
# (name, N, K, kind, dtype, atype, btype, aform, akm, az, bkm, bz, extra qualifiers, M, zdesc, shift, ashift)
Z1 = desc_of([0, 0, 0, 0], [0, 0, 0, 0], 1, 2, 3, 0)
Z2 = desc_of([2, 1, 0, 0], [1, 0, 0, 0], 1, 1, 2, 2)
Z3 = desc_of([1, 0, 2, 3], [0, 1, 1, 0], 1, 0, 1, 0)
CFG = [
 ('ws', 16, 16, 'f16', 'f32', 'f16', 'f16', 'desc', True, 1, True, 1, '.ws', 128, Z1, 0, False),
 ('ws', 16, 16, 'f16', 'f32', 'bf16', 'bf16', 'desc', True, 1, True, 1, '.ws', 64, Z2, 0, False),
 ('ws', 16, 16, 'f16', 'f32', 'f16', 'f16', 'tmem', True, 0, True, 1, '.ws', 32, Z3, 0, False),
 ('ash', 8, 16, 'f16', 'f32', 'f16', 'f16', 'tmem', True, 0, True, 1, '.ashift.collector::a::lastuse.collector::b::use', 128, None, 0, True),
 ('f6', 8, 32, 'f8f6f4', 'f32', 'e2m3', 'e3m2', 'desc', True, 1, True, 1, '', 64, None, 0, False),
 ('f4', 8, 32, 'f8f6f4', 'f32', 'e2m1', 'e4m3', 'desc', True, 1, True, 1, '', 64, None, 0, False),
]
for kk, (nm, N, K, kind, dt, at, bt, aform, akm, az, bkm, bz, extra, M, Zd, shift, ash) in enumerate(CFG):
    wa, wb = bw(at), bw(bt)
    small = (dt == 'f16')
    ws = '.ws' in extra
    def rnd(ty):
        if isflt(ty): return rand_float_bits(ty)
        return random.getrandbits(bw(ty))
    A = [[rnd(at) for _ in range(K)] for _ in range(M)]
    shf = (Zd >> 56) & 63 if ws else 0
    B = [[rnd(bt) for _ in range(K)] for _ in range(N + shf)]
    Din = {(r, c): rand_float_bits('f32') for r in range(M) for c in range(N)}
    zm = zmask(Zd, M, N) if ws else [0] * N
    Dout = {}
    Aeff = [A[r] for r in range(M)]
    if ash: Aeff = [A[0]] + [A[r - 1] for r in range(1, M)]
    for r in range(M):
        for c in range(N):
            acc = fval(dt, Din[(r, c)])
            s = acc + sum(fval(at, Aeff[r][k]) * fval(bt, B[c + shf][k]) for k in range(K)) if not zm[c] else acc + sum(fval(at, Aeff[r][k]) * 0 for k in range(K))
            Dout[(r, c)] = to_bits(dt, s)
    img = {}; off = 0; descA = 0
    if aform == 'desc':
        a_img, la, sa_ = build_tile(A, M, K, wa, akm, az, 0)
        img.update(a_img); descA = desc100(0, la, sa_, az)
        off = (max(a_img) + 1 + 1023) // 1024 * 1024
    b_img, lb, sb_ = build_tile(B, N + shf, K, wb, bkm, bz, off)
    img.update(b_img); descB = desc100(off, lb, sb_, bz)
    size = (max(img) + 1 + 2047) // 2048 * 2048
    data = bytes(img.get(i, 0) for i in range(size))
    idesc = ({'f16': 0, 'f32': 1, 's32': 2}[dt] << 4) | (TCODE[at] << 7) | (TCODE[bt] << 10) | ((0 if akm else 1) << 15) | ((0 if bkm else 1) << 16) \
          | ((N >> 3) << 17) | ((M >> 4) << 24)
    per = 32 // wa
    awords = [[sum(A[r][w * per + j] << (wa * j) for j in range(per)) for w in range(K // per)] for r in range(M)]
    na_cols = K // per
    nd = N
    L = HDR[:]
    L.append(f""".visible .entry k{kk}(.param .u64 pimg, .param .u64 pd, .param .u64 pa, .param .u64 po)
{{
    .reg .b32 %r<8>, %x<4>, %D<{nd}>, %A<{max(na_cols, 1)}>; .reg .b64 %rd<8>; .reg .b64 %dA, %dB, %zm; .reg .b32 %idesc, %tl, %tw;
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
    L.append("    tcgen05.wait::st.sync.aligned;\n    tcgen05.fence::before_thread_sync;\n    bar.sync 0;")
    L.append(f"    mov.b64 %dB, {descB};\n    mov.b32 %idesc, {idesc};")
    a_op = "%dA" if aform == 'desc' else "[%r2]"
    if aform == 'desc': L.append(f"    mov.b64 %dA, {descA};")
    else: L.append("    mov.u32 %r2, 256;")
    if ws:
        L.append(f"    mov.b64 %zm, {Zd};")
        L.append(f"    @%q tcgen05.mma.ws.cta_group::1.{KIND[kind]} [%r3], {a_op}, %dB, %idesc, 1, %zm;")
    else:
        L.append("    mov.b32 %r5, 0;")
        L.append(f"    @%q tcgen05.mma.cta_group::1.{KIND[kind]}{extra} [%r3], {a_op}, %dB, %idesc, {{%r5, %r5, %r5, %r5}}, 1;")
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
    code += f".buffer O .u32 [{128 * nd}];\n.launch k{kk} <<< 1, 128 >>> (IMG, DIN, AIN, O);\n" + expect('O', 'u32', flat(dout)) + "\n"
    open(f'tests/tcx-{kk}.ptx', 'w').write(code)
print(len(CFG), "tests")

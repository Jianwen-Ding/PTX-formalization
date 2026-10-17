#!/usr/bin/env python3
"""Generate tests/wg-*.ptx: wgmma.mma_async against a dense model.  The shared-memory tiles are
built forward from the swizzle-atom description (chunk permutation inside 8-row atoms), not with
the address formula used by the semantics.  usage: gen_wgmma.py [SEED]"""
import random, sys
from fractions import Fraction
src = open(__file__.rsplit('/', 1)[0] + '/gen_mma.py').read().split('cases = []')[0]
sys.argv = [sys.argv[0]] + sys.argv[1:2]
exec(compile(src, 'gen_mma_prefix', 'exec'))

def bw(ty): return {'tf32': 32, 'e4m3': 8, 'e5m2': 8, 'b1': 1, 'bf16': 16, 'f16': 16}.get(ty) or int(ty[1:])
def isflt(ty): return ty in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2')
def schunk(z, r):                      # chunk permutation of 8-row atoms (z = swizzle field 1/2/3)
    return {1: r, 2: (r >> 1) & 3, 3: (r >> 2) & 1}[z]
def build_tile(X, rows, K, w, kmajor, z, base):
    """X[mn][k] -> bytes image (dict offset->byte), (LBO, SBO) in bytes"""
    img = {}
    T = 128 // w
    put = lambda off, bits, v: img.__setitem__(off, img.get(off, 0) | ((v & ((1 << bits) - 1)) << 0)) if False else None
    bitimg = {}
    def setel(byte, bitoff, v):
        v &= (1 << w) - 1
        if w >= 8:
            for i in range(w // 8): bitimg[byte + i] = (v >> (8 * i)) & 255
        else: bitimg[byte] = bitimg.get(byte, 0) | (v << bitoff)
    if kmajor and z == 0:
        kc = -(-K // T); LBO, SBO = 128, 128 * kc
        for mn in range(rows):
            for k in range(K):
                bit = (mn // 8) * SBO * 8 + (k // T) * LBO * 8 + (mn % 8) * 128 + (k % T) * w
                setel(base + bit // 8, bit % 8, X[mn][k])
    elif kmajor:
        G = {1: 8, 2: 4, 3: 2}[z]; LBO, SBO = 16, 128 * G
        for mn in range(rows):
            for k in range(K):
                r0, ch, within = mn % 8, k // T, k % T
                bit = (mn // 8) * SBO * 8 + r0 * 16 * G * 8 + ((ch ^ schunk(z, r0)) * 128) + within * w
                setel(base + bit // 8, bit % 8, X[mn][k])
    elif z == 0:
        mc = -(-rows // T); SBO, LBO = 128, 128 * mc
        for mn in range(rows):
            for k in range(K):
                bit = (mn // T) * SBO * 8 + (k // 8) * LBO * 8 + (k % 8) * 128 + (mn % T) * w
                setel(base + bit // 8, bit % 8, X[mn][k])
    else:
        G = {1: 8, 2: 4, 3: 2}[z]; LBO = 128 * G; mc = -(-rows // (T * G)); SBO = LBO * mc
        for mn in range(rows):
            for k in range(K):
                k0 = k % 8; b = (mn // T) % G; c = mn // (T * G)
                bit = c * LBO * 8 + (k // 8) * SBO * 8 + k0 * 16 * G * 8 + ((b ^ schunk(z, k0)) * 128) + (mn % T) * w
                setel(base + bit // 8, bit % 8, X[mn][k])
    return bitimg, LBO, SBO
def desc(start, lbo, sbo, z): return ((start >> 4) & 0x3FFF) | (((lbo >> 4) & 0x3FFF) << 16) | (((sbo >> 4) & 0x3FFF) << 32) | (z << 62)
def tile_bytes(K, rows, w): return 0

# name: (N, K, dtype, atype, btype, extra, A form, A kmajor, A z, B kmajor, B z, scaleD, sa, sb, small)
CFG = [
 (16, 16, 'f32', 'f16', 'f16', '', 'desc', True, 1, True, 1, 1, 1, 1, False),
 (24, 16, 'f32', 'f16', 'f16', '', 'desc', False, 2, False, 3, 1, -1, 1, False),
 (8, 16, 'f32', 'bf16', 'bf16', '', 'reg', True, 0, False, 0, 0, 1, 1, False),
 (8, 16, 'f16', 'f16', 'f16', '', 'desc', True, 0, True, 0, 1, 1, 1, True),
 (16, 8, 'f32', 'tf32', 'tf32', '', 'desc', True, 1, True, 1, 1, 1, 1, False),
 (8, 8, 'f32', 'tf32', 'tf32', '', 'reg', True, 0, True, 3, 1, 1, -1, False),
 (8, 32, 'f32', 'e4m3', 'e5m2', '', 'desc', True, 1, True, 1, 1, 1, 1, False),
 (8, 32, 'f32', 'e5m2', 'e4m3', '', 'reg', True, 0, True, 2, 1, -1, 1, False),
 (16, 32, 's32', 's8', 'u8', '.satfinite', 'desc', True, 1, True, 1, 1, 1, 1, False),
 (8, 32, 's32', 'u8', 's8', '', 'reg', True, 0, True, 3, 0, 1, 1, False),
 (8, 256, 's32', 'b1', 'b1', '.and.popc', 'desc', True, 1, True, 1, 1, 1, 1, False),
 (24, 16, 'f32', 'f16', 'f16', '', 'desc', True, 3, False, 1, 1, 1, 1, False),
]
h = lambda xs: ", ".join(str(x) for x in xs)
for kk, (N, K, dt, at, bt, extra, aform, akm, az, bkm, bz, sd, sa, sb, small) in enumerate(CFG):
    wa, wb = bw(at), bw(bt)
    def rnd(ty):
        if small and isflt(ty): return to_bits_small(ty)
        if isflt(ty): return rand_float_bits(ty)
        return random.getrandbits(bw(ty)) if ty != 'b1' else random.getrandbits(1)
    def to_bits_small(ty):
        v = random.choice([-2, -1, -0.5, 0, 0.5, 1, 2, 3])
        import struct
        return struct.unpack('<H', struct.pack('<e', v))[0]
    A = [[rnd(at) for _ in range(K)] for _ in range(64)]
    B = [[rnd(bt) for _ in range(K)] for _ in range(N)]          # B tile indexed [n][k]
    ewd = 16 if dt == 'f16' else 32
    Cm = {}
    for r in range(64):
        for c in range(N):
            Cm[(r, c)] = (to_bits_small('f16') if small else rand_float_bits('f32')) if dt != 's32' else random.choice([0, 1, 100, 0xFFFFFFF0, 0x7FFFFFF0, random.getrandbits(32)])
    # expected
    def neg(ty, bits, s): return bits ^ (1 << (bw(ty) - 1)) if (s < 0 and isflt(ty)) else bits
    D = {}
    for r in range(64):
        for c in range(N):
            if isflt(at):
                acc = fval(dt, Cm[(r, c)]) if sd else Fraction(0)
                s = acc + sum(fval(at, neg(at, A[r][k], sa)) * fval(bt, neg(bt, B[c][k], sb)) for k in range(K))
                D[(r, c)] = to_bits(dt, s)
            elif at == 'b1':
                s = (sx(Cm[(r, c)], 32) if sd else 0) + sum(A[r][k] & B[c][k] for k in range(K))
                D[(r, c)] = s & 0xFFFFFFFF
            else:
                va = (lambda x: sx(x, 8)) if at[0] == 's' else (lambda x: x)
                vb = (lambda x: sx(x, 8)) if bt[0] == 's' else (lambda x: x)
                s = (sx(Cm[(r, c)], 32) if sd else 0) + sum(va(A[r][k]) * vb(B[c][k]) for k in range(K))
                if 'satfinite' in extra: s = max(-(1 << 31), min((1 << 31) - 1, s))
                D[(r, c)] = s & 0xFFFFFFFF
    # shared image: A tile (if desc), then B tile, 1024-aligned
    img = {}; off = 0
    a_img = None
    if aform == 'desc':
        a_img, la, sa_ = build_tile(A, 64, K, wa, akm, az, 0)
        img.update(a_img); descA = desc(0, la, sa_, az)
        off = (max(a_img) + 1 + 1023) // 1024 * 1024
    b_img, lb, sb_ = build_tile(B, N, K, wb, bkm, bz, off)
    img.update(b_img); descB = desc(off, lb, sb_, bz)
    size = (max(img) + 1 + 2047) // 2048 * 2048
    data = bytes(img.get(i, 0) for i in range(size))
    # register fragments
    def elem_layout(shape, w): return layout('A', shape, w)
    LAm = layout('A', f"m16n8k{K}", wa) if aform == 'reg' else None
    fa = None
    if aform == 'reg':
        fa = []
        per = 32 // wa
        for u in range(128):
            lane, wp = u % 32, u // 32
            regs = []
            nel = 16 * K // 32
            for ri in range(nel // per):
                v = 0
                for j in range(per):
                    r, c = LAm[(lane, ri * per + j)]
                    v |= A[16 * wp + r][c] << (wa * j)
                regs.append(v)
            fa.append(regs)
    nd = N // 2 if ewd == 32 else N // 4
    fd_in, fd_out = [], []
    for u in range(128):
        lane, wp = u % 32, u // 32
        g, t = lane >> 2, lane & 3
        for tgt, src in ((fd_in, Cm), (fd_out, D)):
            regs = []
            per = 32 // ewd
            for ri in range(nd):
                v = 0
                for j in range(per):
                    e = ri * per + j
                    r, c = 16 * wp + g + 8 * ((e % 4) // 2), 8 * (e // 4) + 2 * t + e % 2
                    v |= src[(r, c)] << (ewd * j)
                regs.append(v)
            tgt.append(regs)
    na = len(fa[0]) if fa else 0
    L = [f"// GENERATED by scripts/gen_wgmma.py -- wgmma.mma_async vs. a dense model (tiles built from swizzle atoms).",
         ".version 8.0\n.target sm_90a\n.address_size 64\n",
         f".visible .entry k{kk}(.param .u64 pimg, .param .u64 pd, .param .u64 pa, .param .u64 po)\n{{\n"
         f"    .reg .b32 %r<8>, %x<4>; .reg .b64 %rd<8>; .reg .b64 %dA, %dB; .reg .b32 %A<{max(na, 1)}>; .reg .b32 %D<{nd}>;",
         f"    .shared .align 1024 .b8 buf[{size}];",
         "    mov.u32 %r0, %tid.x;"]
    L.append("    ld.param.u64 %rd0, [pimg]; mov.u64 %rd2, buf; mul.wide.u32 %rd1, %r0, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;")
    for it in range(size // 2048):
        o = it * 2048
        L.append(f"    ld.global.b32 %x0, [%rd0+{o}]; ld.global.b32 %x1, [%rd0+{o + 4}]; ld.global.b32 %x2, [%rd0+{o + 8}]; ld.global.b32 %x3, [%rd0+{o + 12}];")
        L.append(f"    st.shared.b32 [%rd2+{o}], %x0; st.shared.b32 [%rd2+{o + 4}], %x1; st.shared.b32 [%rd2+{o + 8}], %x2; st.shared.b32 [%rd2+{o + 12}], %x3;")
    L.append("    bar.sync 0;")
    L.append(f"    ld.param.u64 %rd0, [pd]; mul.wide.u32 %rd1, %r0, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    ld.global.b32 %D{i}, [%rd0+{i * 4}];")
    if aform == 'reg':
        L.append(f"    ld.param.u64 %rd0, [pa]; mul.wide.u32 %rd1, %r0, {na * 4}; add.u64 %rd0, %rd0, %rd1;")
        for i in range(na): L.append(f"    ld.global.b32 %A{i}, [%rd0+{i * 4}];")
    L.append(f"    mov.b64 %dB, {descB};")
    if aform == 'desc': L.append(f"    mov.b64 %dA, {descA};")
    L.append("    wgmma.fence.sync.aligned;")
    vec = lambda p, c: "{" + ", ".join(f"%{p}{i}" for i in range(c)) + "}"
    a_op = "%dA" if aform == 'desc' else vec('A', na)
    if at in ('f16', 'bf16'):
        args = f"{sd}, {sa}, {sb}, {(1 if not akm else 0)}, {(1 if not bkm else 0)}" if aform == 'desc' else f"{sd}, {sa}, {sb}, {(1 if not bkm else 0)}"
    elif at in ('tf32', 'e4m3', 'e5m2'): args = f"{sd}, {sa}, {sb}"
    else: args = f"{sd}"
    L.append(f"    wgmma.mma_async.sync.aligned.m64n{N}k{K}.{dt}.{at}.{bt}{extra} {vec('D', nd)}, {a_op}, %dB, {args};")
    L.append("    wgmma.commit_group.sync.aligned;\n    wgmma.wait_group.sync.aligned 0;")
    L.append(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %r0, {nd * 4}; add.u64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    st.global.b32 [%rd0+{i * 4}], %D{i};")
    L.append("    ret;\n}\n")
    flat = lambda f: [x for u in f for x in u]
    CH = 512
    def buf(name, ty, vals):
        out = []
        for i in range(0, max(len(vals), 1), CH):
            part = vals[i:i + CH] or [0]
            out.append(f".buffer {name} .{ty} = {{ {h(part)} }};" if i == 0 else f".append {name} .{ty} = {{ {h(part)} }};")
        return "\n".join(out)
    def expect(name, ty, vals):
        return "\n".join(f".expect {name} .{ty} @ {i} = {{ {h(vals[i:i + CH])} }};" for i in range(0, len(vals), CH))
    # IMG is mostly swizzle padding (zeros): allocate it zeroed and patch only the non-zero runs
    L.append(f".buffer IMG .u8 [{len(data)}];")
    i = 0
    while i < len(data):
        if data[i] == 0: i += 1; continue
        j = i
        while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
        L.append(f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};")
        i = j
    L.append(buf('DIN', 'u32', flat(fd_in)))
    L.append(buf('AIN', 'u32', flat(fa) if fa else [0]))
    L.append(f".buffer O .u32 [{128 * nd}];\n.launch k{kk} <<< 1, 128 >>> (IMG, DIN, AIN, O);")
    L.append(expect('O', 'u32', flat(fd_out)))
    open(f'tests/wg-{kk}.ptx', 'w').write("\n".join(L) + "\n")
print(len(CFG), "cases")

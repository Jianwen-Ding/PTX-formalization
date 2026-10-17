#!/usr/bin/env python3
"""Generate tests/wgsp-*.ptx: sparse wgmma.mma_async against a dense model.  Per warp the sparse A fragment and
metadata are built exactly like gen_mmasp.py builds mma.sp m16n8kK operands (the layout assumption documented in
docs/prior-work.md); B and the packed A tile (descriptor form) are laid out from swizzle atoms (gen_wgmma.py).
usage: gen_wgmma_sp.py [SEED]"""
import random, sys
from fractions import Fraction
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_mma.py').read().split('cases = []')[0]
sys.argv = [sys.argv[0]] + sys.argv[1:2]
exec(compile(src, 'gen_mma_prefix', 'exec'))
wsrc = open(here + '/gen_wgmma.py').read()
wsrc = wsrc[wsrc.index('def bw(ty)'):wsrc.index('# name: (N, K')]
exec(compile(wsrc, 'gen_wgmma_prefix', 'exec'))
ssrc = open(here + '/gen_mmasp.py').read()
ssrc = ssrc[ssrc.index('def chunking'):ssrc.index('h = lambda xs')]
ssrc = ssrc.replace('return shape, dt, at, ct, mods, sel, fa, fb, fc, fd, meta', 'return shape, dt, at, ct, mods, sel, fa, fb, fc, fd, meta, A, choice')
ssrc = ssrc.replace('sel = random.randrange(4) if bits == 16 else (random.randrange(2) if bits == 32 else 0)',
                    'sel = FORCE_SEL if bits != 64 else 0')
exec(compile(ssrc, 'gen_mmasp_prefix', 'exec'))

# (N, K, dtype, atype, btype, A form, A kmajor, A z, B kmajor, B z, sel, scaleD, sa, sb)
CFG = [
 (8, 32, 'f32', 'f16', 'f16', 'reg', True, 0, True, 1, 0, 1, 1, 1),
 (16, 32, 'f32', 'bf16', 'bf16', 'reg', True, 0, False, 2, 1, 1, -1, 1),
 (8, 32, 'f16', 'f16', 'f16', 'desc', True, 1, True, 0, 1, 1, 1, 1),
 (8, 16, 'f32', 'tf32', 'tf32', 'reg', True, 0, True, 2, 0, 0, 1, 1),
 (8, 16, 'f32', 'tf32', 'tf32', 'desc', True, 1, True, 1, 1, 1, 1, -1),
 (8, 64, 'f32', 'e4m3', 'e5m2', 'reg', True, 0, True, 1, 0, 1, 1, 1),
 (16, 64, 'f16', 'e5m2', 'e4m3', 'desc', True, 1, True, 1, 0, 1, -1, 1),
 (8, 64, 's32', 's8', 'u8', 'reg', True, 0, True, 1, 0, 1, 1, 1),
 (16, 64, 's32', 'u8', 's8', 'desc', True, 1, True, 1, 0, 0, 1, 1),
 (8, 32, 'f32', 'f16', 'f16', 'desc', False, 2, False, 3, 0, 1, 1, 1),
 (8, 16, 'f32', 'tf32', 'tf32', 'reg', True, 0, True, 1, 0, 1, 1, 1),
 (8, 16, 'f32', 'tf32', 'tf32', 'reg', True, 0, True, 1, 1, 1, 1, 1),
 (8, 32, 'f32', 'bf16', 'bf16', 'reg', True, 0, True, 1, 1, 1, 1, 1),
 (8, 16, 'f32', 'tf32', 'tf32', 'reg', True, 0, True, 1, 0, 0, 1, 1),
 (8, 16, 'f32', 'tf32', 'tf32', 'reg', True, 0, True, 2, 0, 1, 1, 1),
]
h = lambda xs: ", ".join(str(x) for x in xs)
for kk, (N, K, dt, at, bt, aform, akm, az, bkm, bz, sel, sd, sa, sb) in enumerate(CFG):
    FORCE_SEL = sel
    wa, wb = bw(at), bw(bt)
    Adense = {}; frags = []; metas = []; packed = [[] for _ in range(64)]
    cw, nz = chunking(at)
    for wp in range(4):
        shape, _, _, _, _, s, fa, fb, fc, fd, meta, A, choice = build_sp((f"m16n8k{K}", 'f32', at, 'f32', ''))
        for (r, c), v in A.items(): Adense[(16 * wp + r, c)] = v
        frags += fa; metas += meta
        for r in range(16):
            for c in range(K // cw):
                packed[16 * wp + r] += [A[(r, cw * c + x)] for x in sorted(dense_cols(at, choice[(r, c)]))]
    isf = at in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2')
    def rnd(ty): return rand_float_bits(ty) if isflt(ty) else random.getrandbits(bw(ty))
    B = [[rnd(bt) for _ in range(K)] for _ in range(N)]
    ewd = 16 if dt == 'f16' else 32
    Cm = {(r, c): (rand_float_bits(dt) if dt != 's32' else random.choice([0, 1, 100, 0xFFFFFFF0, random.getrandbits(32)])) for r in range(64) for c in range(N)}
    def neg(ty, bits, s): return bits ^ (1 << (bw(ty) - 1)) if (s < 0 and isflt(ty)) else bits
    D = {}
    for r in range(64):
        for c in range(N):
            if isflt(at):
                acc = fval(dt, Cm[(r, c)]) if sd else Fraction(0)
                s_ = acc + sum(fval(at, neg(at, Adense[(r, k)], sa)) * fval(bt, neg(bt, B[c][k], sb)) for k in range(K))
                D[(r, c)] = to_bits(dt, s_)
            else:
                va = (lambda x: sx(x, 8)) if at[0] == 's' else (lambda x: x)
                vb = (lambda x: sx(x, 8)) if bt[0] == 's' else (lambda x: x)
                s_ = (sx(Cm[(r, c)], 32) if sd else 0) + sum(va(Adense[(r, k)]) * vb(B[c][k]) for k in range(K))
                D[(r, c)] = s_ & 0xFFFFFFFF
    img = {}; off = 0
    if aform == 'desc':
        a_img, la, sa_ = build_tile(packed, 64, K // 2, wa, akm, az, 0)
        img.update(a_img); descA = desc(0, la, sa_, az)
        off = (max(a_img) + 1 + 1023) // 1024 * 1024
    b_img, lb, sb_ = build_tile(B, N, K, wb, bkm, bz, off)
    img.update(b_img); descB = desc(off, lb, sb_, bz)
    size = (max(img) + 1 + 2047) // 2048 * 2048
    data = bytes(img.get(i, 0) for i in range(size))
    nd = N // 2 if ewd == 32 else N // 4
    fd_in, fd_out = [], []
    for u in range(128):
        lane, wp = u % 32, u // 32
        g, t = lane >> 2, lane & 3
        for tgt, src_ in ((fd_in, Cm), (fd_out, D)):
            regs = []; per = 32 // ewd
            for ri in range(nd):
                v = 0
                for j in range(per):
                    e = ri * per + j
                    r, c = 16 * wp + g + 8 * ((e % 4) // 2), 8 * (e // 4) + 2 * t + e % 2
                    v |= src_[(r, c)] << (ewd * j)
                regs.append(v)
            tgt.append(regs)
    fa_ = frags if aform == 'reg' else None
    na = len(fa_[0]) if fa_ else 0
    L = ["// GENERATED by scripts/gen_wgmma_sp.py -- sparse wgmma.mma_async vs. a dense model.",
         ".version 8.0\n.target sm_90a\n.address_size 64\n",
         f".visible .entry k{kk}(.param .u64 pimg, .param .u64 pd, .param .u64 pa, .param .u64 pm, .param .u64 po)\n{{\n"
         f"    .reg .b32 %r<8>, %x<4>, %meta; .reg .b64 %rd<8>; .reg .b64 %dA, %dB; .reg .b32 %A<{max(na, 1)}>; .reg .b32 %D<{nd}>;",
         f"    .shared .align 1024 .b8 buf[{size}];",
         "    mov.u32 %r0, %tid.x;",
         "    ld.param.u64 %rd0, [pimg]; mov.u64 %rd2, buf; mul.wide.u32 %rd1, %r0, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;"]
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
    L.append("    ld.param.u64 %rd0, [pm]; mul.wide.u32 %rd1, %r0, 4; add.u64 %rd0, %rd0, %rd1; ld.global.b32 %meta, [%rd0];")
    L.append(f"    mov.b64 %dB, {descB};")
    if aform == 'desc': L.append(f"    mov.b64 %dA, {descA};")
    L.append("    wgmma.fence.sync.aligned;")
    vec = lambda p, c: "{" + ", ".join(f"%{p}{i}" for i in range(c)) + "}"
    a_op = "%dA" if aform == 'desc' else vec('A', na)
    if at in ('f16', 'bf16'):
        args = f"{sd}, {sa}, {sb}, {(1 if not akm else 0)}, {(1 if not bkm else 0)}" if aform == 'desc' else f"{sd}, {sa}, {sb}, {(1 if not bkm else 0)}"
    elif at in ('tf32', 'e4m3', 'e5m2'): args = f"{sd}, {sa}, {sb}"
    else: args = f"{sd}"
    L.append(f"    wgmma.mma_async.sp.sync.aligned.m64n{N}k{K}.{dt}.{at}.{bt} {vec('D', nd)}, {a_op}, %dB, %meta, {sel}, {args};")
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
    L.append(f".buffer IMG .u8 [{len(data)}];")
    i = 0
    while i < len(data):
        if data[i] == 0: i += 1; continue
        j = i
        while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
        L.append(f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};")
        i = j
    L.append(buf('DIN', 'u32', flat(fd_in)))
    L.append(buf('AIN', 'u32', flat(fa_) if fa_ else [0]))
    L.append(buf('MIN', 'u32', metas))
    L.append(f".buffer O .u32 [{128 * nd}];\n.launch k{kk} <<< 1, 128 >>> (IMG, DIN, AIN, MIN, O);")
    L.append(expect('O', 'u32', flat(fd_out)))
    open(f'tests/wgsp-{kk}.ptx', 'w').write("\n".join(L) + "\n")
print(len(CFG), "cases")

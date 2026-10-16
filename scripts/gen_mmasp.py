#!/usr/bin/env python3
"""Generate tests/sp-*.ptx: mma.sp against a dense model.  A is built dense with the required
2:4 / 1:2 / pair-wise 4:8 structure, then compressed per the ISA fragment formulas (regions of
a row held by a lane, non-zeros in column order) and metadata nibbles placed per the lane
assignment documented in semantics/ptx-exec-sp.k.  usage: gen_mmasp.py [SEED]"""
import random, sys
from fractions import Fraction
src = open(__file__.rsplit('/', 1)[0] + '/gen_mma.py').read().split('cases = []')[0]
sys.argv = [sys.argv[0]] + sys.argv[1:2]
exec(compile(src, 'gen_mma_prefix', 'exec'))

CASES = []
for sh in ('m16n8k16', 'm16n8k32'):
    CASES += [(sh, 'f32', 'f16', 'f32', ''), (sh, 'f16', 'f16', 'f16', ''), (sh, 'f32', 'bf16', 'f32', '')]
for sh in ('m16n8k8', 'm16n8k16'): CASES.append((sh, 'f32', 'tf32', 'f32', ''))
for sh in ('m16n8k32', 'm16n8k64'):
    for a in ('s8', 'u8'): CASES += [(sh, 's32', a, 's32', ''), (sh, 's32', a, 's32', '.satfinite')]
for sh in ('m16n8k64', 'm16n8k128'):
    for a in ('s4', 'u4'): CASES.append((sh, 's32', a, 's32', ''))
for a in ('e4m3', 'e5m2'): CASES.append(('m16n8k64', 'f32', a, 'f32', ''))

def chunking(at):
    if at == 'tf32': return 2, 1          # chunk width, non-zeros per chunk
    if at in ('s4', 'u4'): return 8, 4
    return 4, 2
def regions(at, k, g, t):
    """(row, first col, ncols) for the compressed regions of lane (g, t), in fragment order"""
    w = {'f16': 4, 'bf16': 4, 'tf32': 2, 's8': 8, 'u8': 8, 'e4m3': 8, 'e5m2': 8, 's4': 16, 'u4': 16}[at]
    second = {'f16': 16, 'bf16': 16, 'tf32': 8, 's8': 32, 'u8': 32, 'e4m3': 32, 'e5m2': 32, 's4': 64, 'u4': 64}[at]
    base = [(g, w * t, w), (g + 8, w * t, w)]
    if k > {'f16': 16, 'bf16': 16, 'tf32': 8, 's8': 32, 'u8': 32, 'e4m3': 32, 'e5m2': 32, 's4': 64, 'u4': 64}[at]:
        base += [(g, w * t + second, w), (g + 8, w * t + second, w)]
    return base
def nz_choice(at):
    cw, nz = chunking(at)
    if at == 'tf32': return [(0,), (1,)]
    if cw == 8: return [(a, b) for a in range(4) for b in range(a + 1, 4)]       # sub-chunk pairs
    return [(a, b) for a in range(4) for b in range(a + 1, 4)]
def nibble(at, ch):
    if at == 'tf32': return 0b0100 if ch[0] == 0 else 0b1110
    return ch[0] | (ch[1] << 2)
def dense_cols(at, ch):
    cw, nz = chunking(at)
    if at == 'tf32': return [ch[0]]
    if cw == 8: return [2 * ch[0], 2 * ch[0] + 1, 2 * ch[1], 2 * ch[1] + 1]
    return list(ch)

def build_sp(case):
    shape, dt, at, ct, mods = case
    m, n, k = [int(x) for x in shape[1:].replace('n', ' ').replace('k', ' ').split()]
    ewa = ew_of(at)
    isf = at in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2')
    cw, nz = chunking(at)
    nch = k // cw
    choice = {(r, c): random.choice(nz_choice(at)) for r in range(m) for c in range(nch)}
    rnd = (lambda: rand_float_bits(at)) if isf else (lambda: random.getrandbits(ewa))
    A = {}                                        # dense (row, col) -> bits (0 where pruned)
    for (r, c), ch in choice.items():
        for col in range(cw * c, cw * c + cw): A[(r, col)] = 0
        for col in dense_cols(at, ch): A[(r, cw * c + col)] = rnd()
    # B and C with the dense layouts (B f16 k32 extends the k16 formula to 32 rows)
    btype = at
    ewb = ewa
    LB = {}
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        nb = k * n // 32
        for i in range(nb):
            if ewb == 16: LB[(lane, i)] = (2 * t + (i & 1) + 8 * (i >> 1), g)
            elif ewb == 32: LB[(lane, i)] = (t + 4 * i, g)
            else:
                ek = 32 // ewb; reg, sub = i // ek, i % ek
                LB[(lane, i)] = (ek * t + sub + 4 * ek * reg, g)
    LC = layout('C', shape, ewa)
    B = {pos: (rand_float_bits(at) if isf else random.getrandbits(ewa)) for pos in LB.values()}
    C = {pos: (rand_float_bits(ct) if ct in ('f16', 'f32') else random.choice([0, 1, 100, 0xFFFFFFF0, 0x7FFFFFF0, random.getrandbits(32)])) for pos in LC.values()}
    D = {}
    for r in range(m):
        for c in range(n):
            if isf:
                s = fval(ct, C[(r, c)]) + sum(fval(at, A[(r, kk)]) * fval(at, B[(kk, c)]) for kk in range(k))
                D[(r, c)] = to_bits(dt, s)
            else:
                va = (lambda x: sx(x, ewa)) if at[0] == 's' else (lambda x: x)
                s = sx(C[(r, c)], 32) + sum(va(A[(r, kk)]) * va(B[(kk, c)]) for kk in range(k))
                if 'satfinite' in mods: s = max(-(1 << 31), min((1 << 31) - 1, s))
                D[(r, c)] = s & 0xFFFFFFFF
    # compressed A fragments
    per = 32 // ewa
    fa = []
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        elems = []
        for (r, c0, w) in regions(at, k, g, t):
            for c in range(c0 // cw, (c0 + w) // cw):
                for col in sorted(dense_cols(at, choice[(r, c)])) if cw != 8 else [x for x in dense_cols(at, choice[(r, c)])]:
                    pass
                cols = dense_cols(at, choice[(r, c)])
                if cw == 8:     # pair-wise: the 4 kept elements in column order
                    cols = sorted(cols)
                else: cols = sorted(cols)
                elems += [A[(r, cw * c + x)] for x in cols]
        regs = []
        for ri in range(len(elems) // per):
            wv = 0
            for j in range(per): wv |= elems[ri * per + j] << (ewa * j)
            regs.append(wv)
        fa.append(regs)
    # metadata
    nc = nch; bits = 4 * nc
    sel = random.randrange(4) if bits == 16 else (random.randrange(2) if bits == 32 else 0)
    meta = [random.getrandbits(32) for _ in range(32)]          # lanes that do not supply metadata hold garbage
    for g in range(8):
        def row_bits(r, lo, hi):
            return sum(nibble(at, choice[(r, c)]) << (4 * (c - lo)) for c in range(lo, hi))
        if bits == 16: meta[4 * g + sel] = row_bits(g, 0, 4) | (row_bits(g + 8, 0, 4) << 16)
        elif bits == 32:
            for rh in range(2): meta[4 * g + 2 * sel + rh] = row_bits(g + 8 * rh, 0, 8)
        else:
            for h in range(2):
                for rh in range(2): meta[4 * g + 2 * h + rh] = row_bits(g + 8 * rh, 8 * h, 8 * h + 8)
    fb = frag_regs(LB, B, ewa, k * n // 32, 32)
    fc = frag_regs(LC, C, 16 if ct == 'f16' else 32, m * n // 32, 32)
    fd = frag_regs(LC, D, 16 if dt == 'f16' else 32, m * n // 32, 32)
    return shape, dt, at, ct, mods, sel, fa, fb, fc, fd, meta

h = lambda xs: ", ".join(str(x) for x in xs)
for kk, case in enumerate(CASES):
    shape, dt, at, ct, mods, sel, fa, fb, fc, fd, meta = build_sp(case)
    na, nb, nc_, nd = len(fa[0]), len(fb[0]), len(fc[0]), len(fd[0])
    rg = lambda p, i: f"%{p}{i}"
    vec = lambda p, c: "{" + ", ".join(rg(p, i) for i in range(c)) + "}"
    L = [f"// GENERATED by scripts/gen_mmasp.py -- mma.sp vs. a dense model.", ".version 8.0\n.target sm_90\n.address_size 64\n",
         f".visible .entry k{kk}(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pe, .param .u64 po)\n{{\n    .reg .b32 %r<4>; .reg .b64 %rd<4>; .reg .b32 %A<{na}>; .reg .b32 %B<{nb}>; .reg .b32 %C<{nc_}>; .reg .b32 %D<{nd}>; .reg .b32 %E;",
         "    mov.u32 %r0, %tid.x;"]
    for nm, cnt, par in (('A', na, 'pa'), ('B', nb, 'pb'), ('C', nc_, 'pc'), ('E', 1, 'pe')):
        L.append(f"    ld.param.u64 %rd0, [{par}]; mul.wide.u32 %rd1, %r0, {cnt * 4}; add.s64 %rd0, %rd0, %rd1;")
        for i in range(cnt): L.append(f"    ld.global.b32 {rg(nm, i) if nm != 'E' else '%E'}, [%rd0+{i * 4}];")
    L.append(f"    mma.sp::ordered_metadata.sync.aligned.{shape}.row.col{mods}.{dt}.{at}.{at}.{ct} {vec('D', nd)}, {vec('A', na)}, {vec('B', nb)}, {vec('C', nc_)}, %E, {sel};")
    L.append(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %r0, {nd * 4}; add.s64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    st.global.b32 [%rd0+{i * 4}], %D{i};")
    L.append("    ret;\n}\n")
    flat = lambda f: h(x for lane in f for x in lane)
    L.append(f".buffer A .u32 = {{ {flat(fa)} }};\n.buffer B .u32 = {{ {flat(fb)} }};\n.buffer C .u32 = {{ {flat(fc)} }};\n.buffer E .u32 = {{ {h(meta)} }};")
    L.append(f".buffer O .u32 [{32 * nd}];\n.launch k{kk} <<< 1, 32 >>> (A, B, C, E, O);\n.expect O .u32 = {{ {flat(fd)} }};")
    open(f'tests/sp-{kk}.ptx', 'w').write("\n".join(L) + "\n")
print(len(CASES), "cases")

#!/usr/bin/env python3
"""Generate tests/tc-red.ptx (tcgen05.ld.red, tcgen05.ld[.red].spcompress) and tests/tc-cpd.ptx (tcgen05.cp decompression)
and tests/spc.ptx (spcompress / spdecompress) and tests/tc-ldmap.ptx (the lane / column cells of the 16-lane
tcgen05.ld / st shapes, ISA Figures 184-186) against direct models.  usage: gen_tcgen05_ld.py"""
import random, sys, struct, math
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_wgmma.py').read().split('# name: (N, K')[0]
sys.argv = [sys.argv[0]]
exec(compile(src, 'gen_wgmma_prefix', 'exec'))
t5 = open(here + '/gen_tcgen05.py').read()
exec(compile(t5[t5.index('h = lambda xs'):t5.index('# ---------------------------------------------------------------- alloc')], 'gen_tcgen05_prefix', 'exec'))
random.seed(21)
f2b = lambda f: struct.unpack('<I', struct.pack('<f', f))[0]
b2f = lambda b: struct.unpack('<f', struct.pack('<I', b))[0]
s32 = lambda x: x - (1 << 32) if x >= 1 << 31 else x
NAN = 0x7fffffff
def rnd_f():
    return f2b(random.choice([-1, 1]) * random.choice([0.25, 0.5, 1.5, 3.0, 7.0, 100.0, 1e-3]) * random.randint(1, 9))
# ------------------------------------------------------------------ ld.red / spcompress
NCOL = 8; T = 32
V = [[rnd_f() for _ in range(NCOL)] for _ in range(T)]
V[5][3] = 0x7fc00000          # a NaN in thread 5
V[9][0] = V[9][1] = f2b(3.0)  # ties
def isnan(b): return (b & 0x7f800000) == 0x7f800000 and (b & 0x7fffff) != 0
def red_f(vals, mn, ab, nan):
    xs = [(v & 0x7fffffff) if ab else v for v in vals]
    ns = [x for x in xs if isnan(x)]
    if ns and nan: return NAN
    fs = [x for x in xs if not isnan(x)]
    if not fs: return NAN
    key = lambda x: b2f(x)
    return min(fs, key=key) if mn else max(fs, key=key)
def red_i(vals, mn, signed):
    k = (lambda x: s32(x)) if signed else (lambda x: x)
    return (min if mn else max)(vals, key=k)
def spsel(vals, op):
    """two kept indices of a group of four (op 0 max, 1 maxabs, 2 min, 3 minabs); NaNs first, ties lowest index"""
    def before(j, i):
        nj, ni = isnan(vals[j]), isnan(vals[i])
        if nj: return (not ni) or j < i
        if ni: return False
        kj, ki = b2f(vals[j]), b2f(vals[i])
        if op % 2: kj, ki = abs(kj), abs(ki)
        return (kj > ki if op < 2 else kj < ki) or (kj == ki and j < i)
    return [i for i in range(4) if sum(1 for j in range(4) if j != i and before(j, i)) < 2]
def spc(vals, op):
    idx, cd = [], []
    for g in range(len(vals) // 4):
        s = spsel(vals[4 * g:4 * g + 4], op)
        idx += s; cd += [vals[4 * g + i] for i in s]
    m = sum(i << (2 * k) for k, i in enumerate(idx))
    return [m & 0xffffffff], cd
outs = []   # per thread: list of words
tests = [('red.min.f32', 'min.f32', lambda v: [red_f(v, True, False, False)]),
         ('red.max.abs.f32', 'max.abs.f32', lambda v: [red_f(v, False, True, False)]),
         ('red.max.NaN.f32', 'max.NaN.f32', lambda v: [red_f(v, False, False, True)]),
         ('red.min.u32', 'min.u32', lambda v: [red_i(v, True, False)]),
         ('red.max.s32', 'max.s32', lambda v: [red_i(v, False, True)])]
exp = [[] for _ in range(T)]
for t in range(T):
    for _, _, f in tests: exp[t] += f(V[t])
    m, c = spc(V[t], 2); exp[t] += m + c                                   # min, no abs
    m, c = spc(V[t], 1); exp[t] += m + c + [red_f(V[t], False, True, False)]    # red.spcompress max.abs
NO = len(exp[0])
L = HDR[:]
regs = vec('r', NCOL)
L.append(f""".visible .entry k(.param .u64 pd, .param .u64 po)
{{
    .reg .b32 %r<{NCOL}>, %v<{NCOL}>, %rv, %md, %c<4>, %tl, %tw; .reg .b64 %rd<4>; .reg .pred %p;
    .shared .align 16 .b32 slot[2];
    mov.u32 %tl, %tid.x;
    mov.u64 %rd1, slot;
    mov.u32 %tw, 0;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd1], 32;
    ld.param.u64 %rd0, [pd]; mul.wide.u32 %rd2, %tl, {NCOL * 4}; add.u64 %rd0, %rd0, %rd2;""")
for i in range(NCOL): L.append(f"    ld.global.b32 %v{i}, [%rd0+{i * 4}];")
L.append(f"    tcgen05.st.sync.aligned.32x32b.x{NCOL}.b32 [%tw], {vec('v', NCOL)};\n    tcgen05.wait::st.sync.aligned;\n    tcgen05.fence::before_thread_sync;\n    tcgen05.fence::after_thread_sync;")
L.append("    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd2, %tl, %d; add.u64 %rd0, %rd0, %rd2;".replace('%d', str(NO * 4)))
k = 0
def out(reg):
    global k
    L.append(f"    st.global.b32 [%rd0+{k * 4}], {reg};"); k += 1
for _, q, _ in tests:
    L.append(f"    tcgen05.ld.red.sync.aligned.32x32b.x{NCOL}.{q} {regs}, %rv, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
    out('%rv')
L.append(f"    tcgen05.ld.spcompress.sync.aligned.32x32b.x{NCOL}.min.sp::2:4.f32.b2 {{%md}}, {{%c0, %c1, %c2, %c3}}, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
out('%md'); [out(f'%c{i}') for i in range(4)]
L.append(f"    tcgen05.ld.red.spcompress.sync.aligned.32x32b.x{NCOL}.max.sp::2:4.abs.f32.b2 {{%md}}, {{%c0, %c1, %c2, %c3}}, %rv, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
out('%md'); [out(f'%c{i}') for i in range(4)]; out('%rv')
L.append("    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tw, 32;\n    ret;\n}\n")
assert k == NO, (k, NO)
code = "\n".join(L) + "\n" + buf('DIN', 'u32', [x for r in V for x in r]) + f"\n.buffer O .u32 [{T * NO}];\n.launch k <<< 1, {T} >>> (DIN, O);\n" + expect('O', 'u32', [x for r in exp for x in r]) + "\n"
open('tests/tc-red.ptx', 'w').write(code)

# ------------------------------------------------------------------ cp decompression
R = 128; W = 4
SRC = [[random.getrandbits(32) for _ in range(W)] for _ in range(R)]
img, lbo, sbo = build_tile(SRC, R, W, 32, True, 0, 0)
size = (max(img) + 1 + 2047) // 2048 * 2048
data = bytes(img.get(i, 0) for i in range(size))
D = desc100(0, lbo, sbo, 0)
def dec(row, nb):
    big = sum(w << (32 * i) for i, w in enumerate(row))
    els = [(big >> (nb * i)) & ((1 << nb) - 1) for i in range(16)]
    v = sum((e << (2 if nb == 4 else 0)) << (8 * i) for i, e in enumerate(els))
    return [(v >> (32 * i)) & 0xffffffff for i in range(4)]
NC = 12
words = [[0] * NC for _ in range(128)]
for r in range(128):
    words[r][0:4] = dec(SRC[r], 6); words[r][4:8] = dec(SRC[r], 4); words[r][8:12] = SRC[r]
L = HDR[:]
L.append(f""".visible .entry k(.param .u64 pimg, .param .u64 po)
{{
    .reg .b32 %r<{NC}>, %x<4>, %tl, %tw, %t0; .reg .b64 %rd<8>, %sd; .reg .pred %p, %q;
    .shared .align 1024 .b8 buf[{size}];
    .shared .align 16 .b32 slot[2];
    mov.u32 %tl, %tid.x;
    mov.u64 %rd5, slot;
    setp.lt.u32 %p, %tl, 32;
    setp.eq.u32 %q, %tl, 0;
    @%p tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd5], 32;
    ld.param.u64 %rd0, [pimg]; mov.u64 %rd2, buf; mul.wide.u32 %rd1, %tl, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;""")
for it in range(size // 2048):
    o = it * 2048
    L.append(f"    ld.global.b32 %x0, [%rd0+{o}]; ld.global.b32 %x1, [%rd0+{o + 4}]; ld.global.b32 %x2, [%rd0+{o + 8}]; ld.global.b32 %x3, [%rd0+{o + 12}];")
    L.append(f"    st.shared.b32 [%rd2+{o}], %x0; st.shared.b32 [%rd2+{o + 4}], %x1; st.shared.b32 [%rd2+{o + 8}], %x2; st.shared.b32 [%rd2+{o + 12}], %x3;")
L.append(f"""    bar.sync 0;
    mov.b64 %sd, {D};
    shr.u32 %tw, %tl, 5; shl.b32 %tw, %tw, 21;
    mov.u32 %t0, 0;
    @%q tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [%t0], %sd;
    @%q tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%t0+4], %sd;
    @%q tcgen05.cp.cta_group::1.128x128b [%t0+8], %sd;
    tcgen05.fence::after_thread_sync;
    bar.sync 0;
    tcgen05.ld.sync.aligned.32x32b.x{NC}.b32 {vec('r', NC)}, [%tw];
    tcgen05.wait::ld.sync.aligned;
    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %tl, {NC * 4}; add.u64 %rd0, %rd0, %rd1;""")
for i in range(NC): L.append(f"    st.global.b32 [%rd0+{i * 4}], %r{i};")
L.append("    bar.sync 0;\n    @%p tcgen05.dealloc.cta_group::1.sync.aligned.b32 %t0, 32;\n    ret;\n}\n")
code = "\n".join(L) + f"\n.buffer IMG .u8 [{len(data)}];\n"
i = 0
while i < len(data):
    if data[i] == 0: i += 1; continue
    j = i
    while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
    code += f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};\n"
    i = j
code += f".buffer O .u32 [{128 * NC}];\n.launch k <<< 1, 128 >>> (IMG, O);\n" + expect('O', 'u32', [x for r in words for x in r]) + "\n"
open('tests/tc-cpd.ptx', 'w').write(code)

# ------------------------------------------------------------------ spcompress / spdecompress
def f16(x): return struct.unpack('<H', struct.pack('<e', x))[0]
def val_s8(b): return b - 256 if b >= 128 else b
def sp_ref(words, es, op, tcode):
    mask = (1 << es) - 1
    big = sum(w << (32 * i) for i, w in enumerate(words))
    els = [(big >> (es * i)) & mask for i in range(len(words) * 32 // es)]
    def key(b):
        v = val_s8(b) if (es == 8 and tcode == 1) else (b if es == 8 and tcode == 0 else struct.unpack('<e', struct.pack('<H', b))[0])
        return abs(v) if op % 2 else v
    idx, cd = [], []
    for g in range(len(els) // 4):
        e = els[4 * g:4 * g + 4]
        rank = lambda i: sum(1 for j in range(4) if j != i and ((key(e[j]) > key(e[i]) if op < 2 else key(e[j]) < key(e[i])) or (key(e[j]) == key(e[i]) and j < i)))
        s = [i for i in range(4) if rank(i) < 2]
        idx += s; cd += [e[i] for i in s]
    return idx, cd
def pack(vals, bits, nwords):
    big = sum(v << (bits * i) for i, v in enumerate(vals))
    return [(big >> (32 * i)) & 0xffffffff for i in range(nwords)]
cases = []   # (es, ib, op, tcode, num)
cases = [(8, 2, 1, 1, 4), (8, 2, 2, 0, 2), (16, 2, 0, 0, 2), (8, 4, 3, 1, 2), (16, 4, 2, 0, 4)]
body = []; exp_o = []; din = []; k = 0
for n, (es, ib, op, tc, num) in enumerate(cases):
    data = []
    for _ in range(2 * num):
        if es == 16: ws = [f16(random.choice([-1, 1]) * random.choice([0.5, 1, 2, 3, 5, 8, 0.25]) * random.randint(1, 5)) | (f16(random.choice([-1, 1]) * random.choice([0.5, 1, 2, 3, 5, 8, 0.25])) << 16)]
        else: ws = [random.getrandbits(32)]
        data += ws
    idx, cd = sp_ref(data, es, op, tc)
    nm = max(1, -(-num * ib // es)); nc = num
    md = pack(idx, ib, nm); cw = pack(cd, es, nc)
    din += data
    qual = f"{'b8' if es == 8 else 'b16'}.{'b2' if ib == 2 else 'b4'}.sp::2:4.x{num}"
    ins = ",".join(f"%d{i}" for i in range(2 * num)); cs = ",".join(f"%c{i}" for i in range(nc)); ms = ",".join(f"%m{i}" for i in range(nm))
    body.append((f"    mov.b32 %sd, {op | (tc << 2)};", f"    spcompress.{qual} {{{ms}}}, {{{cs}}}, {{{ins}}}, %sd;", n, 2 * num, nm, nc))
    exp_o += md + cw
    # round trip with spdecompress.sp::2:4 (same index size): scattered back, zeros elsewhere
    dd = [0] * (len(data))
    ntarget = 4
    bigd = 0
    ng = len(idx) // 2
    for g in range(ng):
        for s in range(2):
            bigd |= cd[2 * g + s] << (es * (4 * g + idx[2 * g + s]))
    qd = qual.rsplit('.x', 1)[0] + f".x{len(idx) // 2}"      # spdecompress: .numN counts the groups of four
    body[-1] += (f"    spdecompress.{qd} {{{','.join(f'%e{i}' for i in range(2 * num))}}}, {{{ms}}}, {{{cs}}};",)
    exp_o += [(bigd >> (32 * i)) & 0xffffffff for i in range(2 * num)]
NO = len(exp_o)
maxd = 2 * max(c[4] for c in cases)
L = HDR[:]
L.append(f""".visible .entry k(.param .u64 pd, .param .u64 po)
{{
    .reg .b32 %d<{maxd}>, %e<{maxd}>, %m<{maxd}>, %c<{maxd}>, %sd; .reg .b64 %rd<2>;
    ld.param.u64 %rd0, [pd]; ld.param.u64 %rd1, [po];""")
off = 0; ko = 0
for (sdm, ins, n, nd, nm, nc, *dec_) in body:
    for i in range(nd): L.append(f"    ld.global.b32 %d{i}, [%rd0+{(off + i) * 4}];")
    off += nd
    L += [sdm, ins]
    for i in range(nm): L.append(f"    st.global.b32 [%rd1+{(ko + i) * 4}], %m{i};")
    ko += nm
    for i in range(nc): L.append(f"    st.global.b32 [%rd1+{(ko + i) * 4}], %c{i};")
    ko += nc
    L.append(dec_[0])
    for i in range(nd): L.append(f"    st.global.b32 [%rd1+{(ko + i) * 4}], %e{i};")
    ko += nd
L.append("    ret;\n}\n")
assert ko == NO, (ko, NO)
code = "\n".join(L) + "\n" + buf('DIN', 'u32', din) + f"\n.buffer O .u32 [{NO}];\n.launch k <<< 1, 1 >>> (DIN, O);\n" + expect('O', 'u32', exp_o) + "\n"
open('tests/spc.ptx', 'w').write(code)
# ------------------------------------------------------------------ 16xNb shapes: cells of each (thread, register)
# Transcribed row by row from ISA Figures 184-186 (one .x1 repetition; repetition x adds x * width columns):
#   .16x64b  (2 columns): lane L < 8: T4L r0 col 0, T4L+2 r0 col 1; lane L + 8: T4L+1 r0 col 0, T4L+3 r0 col 1
#   .16x128b (4 columns): lane L: T4L+j r0 col j; lane L + 8: T4L+j r1 col j
#   .16x256b (8 columns): lane L: T4L+j r0 col 2j, r1 col 2j+1; lane L + 8: T4L+j r2 col 2j, r3 col 2j+1
def cells16(shape, num):
    width, nreg = {'16x64b': (2, 1), '16x128b': (4, 2), '16x256b': (8, 4)}[shape]
    rep = {}                     # (thread, register in the repetition) -> (lane, column) for one repetition
    for L in range(8):
        for j in range(4):
            t = 4 * L + j
            if shape == '16x64b':
                rep[(4 * L, 0)], rep[(4 * L + 2, 0)], rep[(4 * L + 1, 0)], rep[(4 * L + 3, 0)] = (L, 0), (L, 1), (L + 8, 0), (L + 8, 1)
            elif shape == '16x128b':
                rep[(t, 0)], rep[(t, 1)] = (L, j), (L + 8, j)
            else:
                rep[(t, 0)], rep[(t, 1)], rep[(t, 2)], rep[(t, 3)] = (L, 2 * j), (L, 2 * j + 1), (L + 8, 2 * j), (L + 8, 2 * j + 1)
    out = {}
    for t in range(32):
        for x in range(num):
            for i in range(nreg):
                ln, c = rep[(t, i)]
                out[(t, x * nreg + i)] = (ln, x * width + c)
    assert sorted(out.values()) == [(ln, c) for ln in range(16) for c in range(width * num)]
    return out
MAPS = [('16x64b', 4), ('16x128b', 2), ('16x256b', 1)]
NC = 8                           # every shape above covers 8 columns of lanes 0-15
val = lambda salt, t, c: (salt << 24) | (t << 8) | c
L = HDR[:]
L.append(""".visible .entry k(.param .u64 po)
{
    .reg .b32 %v<8>, %w<8>, %tl, %tw; .reg .b64 %rd<4>;
    .shared .align 16 .b32 slot[1];
    mov.u32 %tl, %tid.x;
    mov.u64 %rd1, slot;
    mov.u32 %tw, 0;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd1], 32;
    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd2, %tl, 4; add.u64 %rd0, %rd0, %rd2;""")
exp = {t: [] for t in range(32)}
k = 0
def put(reg):
    global k
    L.append(f"    st.global.b32 [%rd0+{k * 128}], {reg};"); k += 1
for si, (shape, num) in enumerate(MAPS):
    cl = cells16(shape, num); n = len(cl) // 32
    # 1. lane t, column c := val(si, t, c) by .32x32b; read with the shape: register i of thread t gets its cell
    for c in range(NC): L.append(f"    mad.lo.u32 %v{c}, %tl, 256, {(si << 24) | c};")
    L.append(f"    tcgen05.st.sync.aligned.32x32b.x{NC}.b32 [%tw], {vec('v', NC)};\n    tcgen05.wait::st.sync.aligned;")
    L.append(f"    tcgen05.ld.sync.aligned.{shape}.x{num}.b32 {vec('w', n)}, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
    for i in range(n): put(f"%w{i}")
    for t in range(32): exp[t] += [val(si, *cl[(t, i)]) for i in range(n)]
    # 2. write register i of thread t := val(si + 8, t, i) with the shape; read every cell back by .32x32b
    for i in range(n): L.append(f"    mad.lo.u32 %w{i}, %tl, 256, {((si + 8) << 24) | i};")
    L.append(f"    tcgen05.st.sync.aligned.{shape}.x{num}.b32 [%tw], {vec('w', n)};\n    tcgen05.wait::st.sync.aligned;")
    L.append(f"    tcgen05.ld.sync.aligned.32x32b.x{NC}.b32 {vec('v', NC)}, [%tw];\n    tcgen05.wait::ld.sync.aligned;")
    for c in range(NC): put(f"%v{c}")
    inv = {cell: ti for ti, cell in cl.items()}
    for t in range(32): exp[t] += [val(si + 8, *inv[(t, c)]) if (t, c) in inv else val(si, t, c) for c in range(NC)]
L.append("    tcgen05.dealloc.cta_group::1.sync.aligned.b32 %tw, 32;\n    ret;\n}\n")
code = "\n".join(L) + f"\n.buffer O .u32 [{32 * k}];\n.launch k <<< 1, 32 >>> (O);\n" + expect('O', 'u32', [exp[t][j] for j in range(k) for t in range(32)]) + "\n"
open('tests/tc-ldmap.ptx', 'w').write(code)
print("ok")

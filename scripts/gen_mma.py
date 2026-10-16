#!/usr/bin/env python3
"""Generate tests/mma-*.ptx: mma.sync for every supported shape/type against an exact
Python model.  The fragment layouts here are a second, independently written
transcription of the PTX ISA tables (as per-element lists of (row, col)); the
generator first checks that each layout is a bijection onto the matrix.
usage: gen_mma.py [SEED] [PART PARTS]"""
import random, sys
from fractions import Fraction
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import ieee_ref as R
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 61)
PART, PARTS = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)

def layout(mat, shape, ew):
    """list over (lane, i) of (row, col) for matrix A, B or C"""
    m, n, k = [int(x) for x in shape[1:].replace('n', ' ').replace('k', ' ').split()]
    out = {}
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        if mat == 'C':
            n_el = m * n // 32
            for i in range(n_el):
                out[(lane, i)] = (g + (8 if i >= 2 and m == 16 else 0), 2 * t + (i & 1)) if m == 16 else (g, 2 * t + i)
            continue
        if mat == 'A':
            n_el = m * k // 32
            for i in range(n_el):
                if ew == 16:
                    if k == 16: out[(lane, i)] = (g + 8 * ((i >> 1) & 1), 2 * t + (i & 1) + 8 * (i >> 2))
                    else:        out[(lane, i)] = (g + 8 * (i >> 1), 2 * t + (i & 1))
                elif ew in (32, 64) and not (ew == 64 and shape == 'm8n8k4'):
                    if k == 4: out[(lane, i)] = (g + 8 * i, t)
                    else: out[(lane, i)] = (g + 8 * (i & 1), t + 4 * (i >> 1))
                elif ew == 64: out[(lane, i)] = (g, t)
                else:
                    ek = 32 // ew                           # elements per 32-bit register (k contiguous)
                    per = 4 * ek                            # k covered by one register column set
                    if m == 8: out[(lane, i)] = (g, ek * t + i)
                    else:
                        ec = ek if ew == 1 else ek
                        nblk = n_el // ek                   # registers
                        reg, sub = i // ek, i % ek
                        out[(lane, i)] = (g + 8 * (reg % 2), ek * t + sub + 4 * ek * (reg // 2))
        else:
            n_el = k * n // 32
            for i in range(n_el):
                if ew == 16:
                    if k == 16: out[(lane, i)] = (2 * t + (i & 1) + 8 * (i >> 1), g)
                    else: out[(lane, i)] = (2 * t + i, g)
                elif ew in (32, 64):
                    out[(lane, i)] = (t + 4 * i, g)
                else:
                    ek = 32 // ew
                    reg, sub = i // ek, i % ek
                    out[(lane, i)] = (ek * t + sub + 4 * ek * reg, g)
    return out
def check_bij(mat, shape, ew):
    L = layout(mat, shape, ew)
    m, n, k = [int(x) for x in shape[1:].replace('n', ' ').replace('k', ' ').split()]
    rows, cols = {'A': (m, k), 'B': (k, n), 'C': (m, n)}[mat]
    vals = list(L.values())
    assert len(set(vals)) == len(vals) == rows * cols, (mat, shape, ew, len(set(vals)), len(vals), rows * cols)
    assert all(0 <= r < rows and 0 <= c < cols for r, c in vals), (mat, shape, ew)
    return L

# ---------------------------------------------------------------- element models
def mini_tab(name):
    e, m = {'e4m3': (4, 3), 'e5m2': (5, 2)}[name]; bias = (1 << (e - 1)) - 1
    t = {}
    for c in range(1 << (e + m + 1)):
        s, E, M = c >> (e + m), (c >> m) & ((1 << e) - 1), c & ((1 << m) - 1)
        if name == 'e4m3' and (c & 0x7F) == 0x7F: continue
        if name == 'e5m2' and E == 31: continue
        v = Fraction(M, 1 << m) * Fraction(2) ** (1 - bias) if E == 0 else Fraction((1 << m) + M, 1 << m) * Fraction(2) ** (E - bias)
        t[c] = -v if s else v
    return t
MINI = {n: mini_tab(n) for n in ('e4m3', 'e5m2')}
def fval(ty, bits):
    if ty == 'f16': k, v, s = R.classify(bits, 16)
    elif ty == 'bf16': k, v, s = R.classify(bits, 116)
    elif ty == 'tf32': k, v, s = R.classify(bits & ~0x1FFF, 32)
    elif ty == 'f32': k, v, s = R.classify(bits, 32)
    elif ty == 'f64': k, v, s = R.classify(bits, 64)
    else: return MINI[ty][bits]
    assert k in ('fin', 'zero')
    return Fraction(0) if k == 'zero' else (-v if s else v)
def rand_float_bits(ty):
    w = {'f16': 16, 'bf16': 116, 'tf32': 32, 'f32': 32, 'f64': 64}.get(ty)
    if ty in MINI: return random.choice(list(MINI[ty]))
    p, emin, emax, eb = R.FMT[w]; cw = R.CW[w]; bias = (1 << (eb - 1)) - 1
    b = (random.getrandbits(1) << (cw - 1)) | (random.randint(bias - 5, bias + 5) << (p - 1)) | random.getrandbits(p - 1)
    if random.random() < .08: b = 0
    if ty == 'tf32': b &= ~0x1FFF
    return b
def to_bits(ty, v):                       # Fraction -> bits of accumulator type (round to nearest even, no overflow expected)
    w = {'f16': 16, 'f32': 32, 'f64': 64}[ty]
    if v == 0: return 0
    sg = 1 if v < 0 else 0
    return (sg << (R.CW[w] - 1)) | R.round_mag(abs(v), 0, 'rn', sg, w)
def sx(x, w): x &= (1 << w) - 1; return x - (1 << w) if x >> (w - 1) else x

# ---------------------------------------------------------------- configurations
CFGS = []   # (shape, dtype, atype, btype, ctype, mods)
for sh in ('m16n8k8', 'm16n8k16'):
    CFGS += [(sh, 'f32', 'f16', 'f16', 'f32', ''), (sh, 'f16', 'f16', 'f16', 'f16', ''), (sh, 'f32', 'bf16', 'bf16', 'f32', '')]
CFGS += [('m16n8k4', 'f32', 'tf32', 'tf32', 'f32', ''), ('m16n8k8', 'f32', 'tf32', 'tf32', 'f32', '')]
CFGS += [(sh, 'f64', 'f64', 'f64', 'f64', '') for sh in ('m8n8k4', 'm16n8k4', 'm16n8k8', 'm16n8k16')]
for sh in ('m8n8k16', 'm16n8k16', 'm16n8k32'):
    for a in ('s8', 'u8'):
        for b in ('s8', 'u8'):
            CFGS += [(sh, 's32', a, b, 's32', ''), (sh, 's32', a, b, 's32', '.satfinite')]
for sh in ('m8n8k32', 'm16n8k32', 'm16n8k64'):
    for a in ('s4', 'u4'):
        for b in ('s4', 'u4'):
            CFGS.append((sh, 's32', a, b, 's32', ''))
for sh in ('m8n8k128', 'm16n8k128', 'm16n8k256'):
    CFGS += [(sh, 's32', 'b1', 'b1', 's32', '.xor.popc'), (sh, 's32', 'b1', 'b1', 's32', '.and.popc')]
for sh in ('m16n8k16', 'm16n8k32'):
    for a in ('e4m3', 'e5m2'):
        for b in ('e4m3', 'e5m2'):
            CFGS.append((sh, 'f32', a, b, 'f32', ''))

def ewof(ty): return {'e4m3': 8, 'e5m2': 8}.get(ty) or int(''.join(ch for ch in ty if ch.isdigit()) or 0) if ty != 'tf32' else 32
def ew_of(ty):
    if ty in ('e4m3', 'e5m2'): return 8
    if ty == 'tf32': return 32
    return int(ty[1:]) if ty[0] in 'sufb' and ty[1:].isdigit() else 16
def build(cfg):
    shape, dt, at, bt, ct, mods = cfg
    m, n, k = [int(x) for x in shape[1:].replace('n', ' ').replace('k', ' ').split()]
    ewa = ew_of(at)
    LA, LB, LC = check_bij('A', shape, ewa), check_bij('B', shape, ewa), check_bij('C', shape, ewa)
    isf = at in ('f16', 'bf16', 'tf32', 'f64', 'e4m3', 'e5m2')
    A = {}; B = {}; C = {}
    for pos in LA.values():
        A[pos] = rand_float_bits(at) if isf else (random.getrandbits(1) if at == 'b1' else random.getrandbits(ewa))
    for pos in LB.values():
        B[pos] = rand_float_bits(bt) if isf else (random.getrandbits(1) if bt == 'b1' else random.getrandbits(ewa))
    for pos in LC.values():
        C[pos] = rand_float_bits(ct) if ct in ('f16', 'f32', 'f64') else random.choice([0, 1, 100, 0xFFFFFFF0, 0x7FFFFFF0, random.getrandbits(32)])
    # model
    D = {}
    for r in range(m):
        for c in range(n):
            if isf:
                s = fval(ct, C[(r, c)]) + sum(fval(at, A[(r, kk)]) * fval(bt, B[(kk, c)]) for kk in range(k))
                D[(r, c)] = to_bits(dt, s)
            elif at == 'b1':
                op = (lambda x, y: x ^ y) if 'xor' in mods else (lambda x, y: x & y)
                s = sx(C[(r, c)], 32) + sum(op(A[(r, kk)], B[(kk, c)]) for kk in range(k))
                D[(r, c)] = s & 0xFFFFFFFF
            else:
                va = (lambda x: sx(x, ewa)) if at[0] == 's' else (lambda x: x)
                vb = (lambda x: sx(x, ewa)) if bt[0] == 's' else (lambda x: x)
                s = sx(C[(r, c)], 32) + sum(va(A[(r, kk)]) * vb(B[(kk, c)]) for kk in range(k))
                if 'satfinite' in mods: s = max(-(1 << 31), min((1 << 31) - 1, s))
                D[(r, c)] = s & 0xFFFFFFFF
    return shape, dt, at, bt, ct, mods, m, n, k, ewa, LA, LB, LC, A, B, C, D

def frag_regs(L, vals, ew, ncount, regbits):
    """per-lane register words (each reg packs regbits/ew elements, element 0 lowest)"""
    per = regbits // ew; words = []
    for lane in range(32):
        regs = []
        for r in range((ncount + per - 1) // per):
            w = 0
            for j in range(per):
                i = r * per + j
                if i < ncount: w |= vals[L[(lane, i)]] << (ew * j)
            regs.append(w)
        words.append(regs)
    return words

cases = []
for cfg in CFGS:
    cases.append(build(cfg))
cases = cases[PART::PARTS]
print("// GENERATED by scripts/gen_mma.py -- mma.sync vs. an exact model with independently transcribed layouts.")
print(".version 8.0\n.target sm_90\n.address_size 64\n")
def regtype(ty): return 'b64' if ty == 'f64' else 'b32'
out_bufs = []
for kk, (shape, dt, at, bt, ct, mods, m, n, k, ewa, LA, LB, LC, A, B, C, D) in enumerate(cases):
    rb = 64 if at == 'f64' else 32
    ewc = 16 if ct == 'f16' else (64 if ct == 'f64' else 32)
    fa = frag_regs(LA, A, ewa, m * k // 32, rb)
    fb = frag_regs(LB, B, ewa if bt == at or True else ew_of(bt), k * n // 32, rb)
    fc = frag_regs(LC, C, ewc, m * n // 32, 64 if ct == 'f64' else 32)
    fd = frag_regs(LC, D, 16 if dt == 'f16' else (64 if dt == 'f64' else 32), m * n // 32, 64 if dt == 'f64' else 32)
    na, nb, nc, nd = len(fa[0]), len(fb[0]), len(fc[0]), len(fd[0])
    cb = 'b64' if ct == 'f64' else 'b32'; db = 'b64' if dt == 'f64' else 'b32'; ab = 'b64' if at == 'f64' else 'b32'
    rg = lambda pre, wide, i: f"%{pre.upper()}{i}"
    print(f".visible .entry k{kk}(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 po)\n{{\n    .reg .b32 %r<130>; .reg .b64 %rd<130>;")
    print(f"    mov.u32 %r0, %tid.x;")
    for nm, nreg, bt_, wide, par in (('a', na, ab, at == 'f64', 'pa'), ('b', nb, ab, at == 'f64', 'pb'), ('c', nc, cb, ct == 'f64', 'pc')):
        sz = 8 if wide else 4
        print(f"    ld.param.u64 %rd0, [{par}]; mul.wide.u32 %rd1, %r0, {nreg * sz}; add.s64 %rd0, %rd0, %rd1;")
        for i in range(nreg):
            print(f"    ld.global.{bt_} {rg(nm, wide, i)}, [%rd0+{i * sz}];")
    def vec(nm, n_, wide): return "{" + ", ".join(rg(nm, wide, i) for i in range(n_)) + "}"
    # destination registers use their own prefix 'd'
    print(f"    mma.sync.aligned.{shape}.row.col{mods}.{dt}.{at}.{bt}.{ct} {vec('d', nd, dt == 'f64')}, {vec('a', na, at == 'f64')}, {vec('b', nb, at == 'f64')}, {vec('c', nc, ct == 'f64')};")
    sz = 8 if dt == 'f64' else 4
    print(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %r0, {nd * sz}; add.s64 %rd0, %rd0, %rd1;")
    for i in range(nd): print(f"    st.global.{db} [%rd0+{i * sz}], {rg('d', dt == 'f64', i)};")
    print("    ret;\n}\n")
    out_bufs.append((na, nb, nc, nd, fa, fb, fc, fd))
for kk, (cfg, ob) in enumerate(zip(cases, out_bufs)):
    shape, dt, at, bt, ct, mods = cfg[:6]
    na, nb, nc, nd, fa, fb, fc, fd = ob
    wa = 'u64' if at == 'f64' else 'u32'; wc = 'u64' if ct == 'f64' else 'u32'; wd = 'u64' if dt == 'f64' else 'u32'
    flat = lambda f: ", ".join(str(x) for lane in f for x in lane)
    print(f".buffer A{kk} .{wa} = {{ {flat(fa)} }};\n.buffer B{kk} .{wa} = {{ {flat(fb)} }};\n.buffer C{kk} .{wc} = {{ {flat(fc)} }};")
    print(f".buffer O{kk} .{wd} [{32 * nd}];\n.launch k{kk} <<< 1, 32 >>> (A{kk}, B{kk}, C{kk}, O{kk});\n.expect O{kk} .{wd} = {{ {flat(fd)} }};")

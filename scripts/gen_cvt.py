#!/usr/bin/env python3
"""Generate tests/cvt-*.ptx: the cvt conversion matrix (float<->float, int<->float,
tf32) with every rounding mode and .ftz/.relu/.sat/.satfinite, against the
exact-rational reference (scripts/ieee_ref.py).
usage: gen_cvt.py [N] [SEED] [PART PARTS]"""
import random, sys
from fractions import Fraction
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import ieee_ref as R
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 31)
PART, PARTS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
R.FMT[132] = (11, -126, 127, 8); R.CW[132] = 32
FN = {16: 'f16', 116: 'bf16', 32: 'f32', 64: 'f64'}
CWID = {16: 16, 116: 16, 32: 32, 64: 64}
def rnd_float(w):
    p, emin, emax, eb = R.FMT[w]; cw = R.CW[w]; bias = (1 << (eb - 1)) - 1
    k = random.random()
    if k < .12: return random.choice([0, 1 << (cw - 1), R.inf_bits(0, w), R.inf_bits(1, w), R.canon_nan(w), 1, R.one_bits(w),
                                      R.one_bits(w) | (1 << (cw - 1))])
    if k < .22: return random.getrandbits(p - 1) | (random.getrandbits(1) << (cw - 1))
    if k < .35:
        v = random.randint(-70000, 70000) / random.choice([1, 2, 4, 8, 3])
        try: return R.mag_to_bits(Fraction(abs(v)).limit_denominator(1 << 20), w) | ((v < 0) << (cw - 1)) if False else \
            random.getrandbits(cw)
        except Exception: pass
    return (random.getrandbits(1) << (cw - 1)) | (random.randint(max(1, bias - 22), min((1 << eb) - 2, bias + 22)) << (p - 1)) | random.getrandbits(p - 1)

def maxfin(d): return R.inf_bits(0, d) - 1
def conv_ff(b, s, d, mode, ftz, relu, sat, satfin):
    if ftz: b = R.flush(b, s)
    k, v, sg = R.classify(b, s)
    cw = R.CW[d]
    if k == 'nan': out = R.canon_nan(d)
    elif k == 'inf': out = R.inf_bits(sg, d)
    elif k == 'zero': out = R.zero_bits(sg, d)
    else: out = (sg << (cw - 1)) | R.round_mag(v, 0, mode, sg, d)
    if ftz: out = R.flush(out, d)
    kk, vv, ss = R.classify(out, d)
    if satfin and kk == 'inf': out = (ss << (cw - 1)) | maxfin(d)
    if relu and kk != 'nan' and ss: out = 0
    return R.post(out, d, False, sat), (sat and kk == 'zero')

def to_tf32(b32mag_bits_p11):   # bits in (p=11) layout -> f32 container layout
    e = b32mag_bits_p11 >> 10; m = b32mag_bits_p11 & 0x3FF
    return (e << 23) | (m << 13)
def conv_tf32(b, mode, relu, satfin):
    k, v, sg = R.classify(b, 32)
    if k == 'nan': return R.canon_nan(32)
    if k == 'inf': out = R.inf_bits(sg, 32)
    elif k == 'zero': out = R.zero_bits(sg, 32)
    else: out = (sg << 31) | to_tf32(R.round_mag(v, 0, mode, sg, 132)) if mode != 'rna' else (sg << 31) | to_tf32(rna_mag(v))
    kk, vv, ss = R.classify(out, 32)
    if satfin and kk == 'inf': out = (ss << 31) | to_tf32(R.inf_bits(0, 132) - 1) if False else (ss << 31) | ((127 + 127) << 23) | (0x3FF << 13)
    if relu and kk != 'nan' and ss: out = 0
    return out
def rna_mag(v):
    # nearest, ties away from zero, in the p=11 / emin=-126 format
    lo = R.round_mag(v, 0, 'rz', 0, 132); hi = R.round_mag(v, 0, 'rp', 0, 132)
    if lo == hi: return lo
    def val(bits):
        e = bits >> 10; m = bits & 0x3FF
        return Fraction(m, 1 << 10) * Fraction(2) ** -126 if e == 0 else Fraction(1024 + m, 1 << 10) * Fraction(2) ** (e - 127)
    dl, dh = v - val(lo), (val(hi) - v) if hi < R.inf_bits(0, 132) else None
    if dh is None: return hi
    return lo if dl < dh else hi if dh < dl else hi

def int_range(t):
    w = int(t[1:]); return (-(1 << (w - 1)), (1 << (w - 1)) - 1) if t[0] == 's' else (0, (1 << w) - 1)
def conv_fi(b, s, t, rm):
    k, v, sg = R.classify(b, s)
    lo, hi = int_range(t)
    if k == 'nan': return 0
    if k == 'inf': return hi if not sg else lo
    x = Fraction(0) if k == 'zero' else (-v if sg else v)
    fl = x.numerator // x.denominator
    if rm == 'rzi': n = int(x)
    elif rm == 'rmi': n = fl
    elif rm == 'rpi': n = fl if x == fl else fl + 1
    else:
        fr = x - fl; n = fl if fr < Fraction(1, 2) else fl + 1 if fr > Fraction(1, 2) else (fl if fl % 2 == 0 else fl + 1)
    return max(lo, min(hi, n))
def conv_if(i, t, d, mode):
    if i == 0: return 0
    sg = 1 if i < 0 else 0
    return (sg << (R.CW[d] - 1)) | R.round_mag(Fraction(abs(i)), 0, mode, sg, d)

cases = []   # (opcode text, src kind, src width bits, dst reg kind, dst width bits, rows)
def addc(text, st, sw, dw, rows): cases.append((text, st, sw, dw, rows))
FF = [(64, 32), (64, 16), (64, 116), (32, 16), (32, 116), (16, 116), (116, 32), (16, 32), (32, 64), (16, 64), (116, 64), (116, 16)]
for s, d in FF:
    narrowing = R.FMT[d][0] < R.FMT[s][0] or R.FMT[d][1] > R.FMT[s][1] or R.FMT[d][2] < R.FMT[s][2]
    modes = ['rn', 'rz', 'rm', 'rp'] if narrowing else ['']
    for mode in modes:
        for mods in (['', '.ftz', '.relu', '.sat', '.satfinite'] if mode in ('rn', '') else ['']):
            if '.ftz' in mods and s in (116,): continue
            A = [rnd_float(s) for _ in range(N)]
            rows = []
            for a in A:
                out, skip = conv_ff(a, s, d, mode or 'rn', mods == '.ftz', mods == '.relu', mods == '.sat', mods == '.satfinite')
                if not skip: rows.append((a, out))
            op = 'cvt' + ('.' + mode if mode else '') + mods + f'.{FN[d]}.{FN[s]}'
            addc(op, 'f', CWID[s], CWID[d], rows)
INTS = ['s8', 'u8', 's16', 'u16', 's32', 'u32', 's64', 'u64']
def rint(t):
    lo, hi = int_range(t)
    return random.choice([lo, hi, 0, 1, -1 if lo < 0 else 2, hi - 1, lo + 1]) if random.random() < .35 else random.randint(lo, hi) if random.random() < .5 else int(random.getrandbits(4) * 0 + random.randint(lo, hi))
for t in INTS:
    for d in (16, 116, 32, 64):
        if t in ('s8', 'u8', 's16', 'u16') and d == 64: continue
        for mode in ('rn', 'rz', 'rm', 'rp'):
            vals = [rint(t) for _ in range(N)]
            w = int(t[1:])
            addc(f'cvt.{mode}.{FN[d]}.{t}', 'i' + t, w, CWID[d], [(v & ((1 << w) - 1), conv_if(v, t, d, mode)) for v in vals])
for s in (16, 116, 32, 64):
    for t in INTS:
        if s in (16, 116) and t in ('s64', 'u64'): pass
        for rm in ('rni', 'rzi', 'rmi', 'rpi'):
            A = [rnd_float(s) for _ in range(N)]
            addc(f'cvt.{rm}.{t}.{FN[s]}', 'f', CWID[s], int(t[1:]), [(a, conv_fi(a, s, t, rm) & ((1 << int(t[1:])) - 1)) for a in A])
for mode, mods in (('rna', ''), ('rna', '.satfinite'), ('rn', ''), ('rn', '.relu'), ('rz', ''), ('rm', '')):
    A = [rnd_float(32) for _ in range(N)]
    addc(f'cvt.{mode}{mods}.tf32.f32', 'f', 32, 32, [(a, conv_tf32(a, mode, '.relu' in mods, '.satfinite' in mods)) for a in A])
cases = cases[PART::PARTS]

print("// GENERATED by scripts/gen_cvt.py -- cvt conversion matrix vs. the exact-rational reference.")
print(".version 8.0\n.target sm_90\n.address_size 64\n")
SZ = {8: 'b8', 16: 'b16', 32: 'b32', 64: 'b64'}
for k, (text, st, sw, dw, rows) in enumerate(cases):
    n = len(rows)
    srcty = text.split('.')[-1]
    if st == 'f':
        ldty = {16: 'b16', 32: 'b32', 64: 'b64'}[sw]
    else:
        ldty = srcty                                   # sign/zero-extending load for integer sources
    sreg = {8: '%r', 16: '%r', 32: '%r', 64: '%rd'}[sw] if st != 'f' else {16: '%h', 32: '%r', 64: '%rd'}[sw]
    dreg = {8: '%r', 16: '%h', 32: '%r', 64: '%rd'}[dw] if dw != 16 or True else '%h'
    dstty = text.split('.')[-2] if len(text.split('.')) > 2 else ''
    isfloatdst = dstty in ('f16', 'bf16', 'f32', 'f64', 'tf32')
    if isfloatdst: dreg = {16: '%h', 32: '%r', 64: '%rd'}[dw]
    elif dw < 32: dreg = '%r'
    print(f".visible .entry k{k}(.param .u64 pi, .param .u64 po)\n{{\n    .reg .b16 %h<4>; .reg .b32 %r<8>; .reg .b64 %rd<8>;\n    mov.u32 %r0, %tid.x;")
    print(f"    mul.wide.u32 %rd1, %r0, {max(sw // 8, 1)}; ld.param.u64 %rd2, [pi]; add.s64 %rd2, %rd2, %rd1; ld.global.{ldty} {sreg}4, [%rd2];")
    print(f"    {text} {dreg}5, {sreg}4;")
    stty = {8: 'b8', 16: 'b16', 32: 'b32', 64: 'b64'}[dw]
    print(f"    mul.wide.u32 %rd3, %r0, {max(dw // 8, 1)}; ld.param.u64 %rd4, [po]; add.s64 %rd4, %rd4, %rd3; st.global.{stty} [%rd4], {dreg}5;\n    ret;\n}}\n")
for k, (text, st, sw, dw, rows) in enumerate(cases):
    n = len(rows)
    it = {8: 'u8', 16: 'u16', 32: 'u32', 64: 'u64'}[sw]; ot = {8: 'u8', 16: 'u16', 32: 'u32', 64: 'u64'}[dw]
    print(f".buffer I{k} .{it} = {{ {', '.join(str(r[0]) for r in rows)} }};")
    print(f".buffer O{k} .{ot} [{n}];\n.launch k{k} <<< 1, {n} >>> (I{k}, O{k});")
    print(f".expect O{k} .{ot} = {{ {', '.join(str(r[1]) for r in rows)} }};")

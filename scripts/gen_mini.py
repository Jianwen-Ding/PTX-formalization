#!/usr/bin/env python3
"""Generate tests/mini-*.ptx: fp8/fp6/fp4 and ue8m0 conversions against an
independent reference that enumerates every code of each format and picks the
nearest representable value (ties to the even code).
usage: gen_mini.py [N] [SEED] [PART PARTS]"""
import math, random, struct, sys
from fractions import Fraction
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 41)
# name: (exponent bits, mantissa bits, has_inf)
FMTS = {'e4m3': (4, 3, False), 'e5m2': (5, 2, True), 'e2m3': (2, 3, False), 'e3m2': (3, 2, False), 'e2m1': (2, 1, False)}
def table(name):
    e, m, inf = FMTS[name]; bias = (1 << (e - 1)) - 1
    vals = {}
    for c in range(1 << (e + m)):
        E, M = c >> m, c & ((1 << m) - 1)
        if name == 'e4m3' and c == 127: vals[c] = 'nan'; continue
        if name == 'e5m2' and E == 31: vals[c] = 'inf' if M == 0 else 'nan'; continue
        vals[c] = Fraction(M, 1 << m) * Fraction(2) ** (1 - bias) if E == 0 else Fraction((1 << m) + M, 1 << m) * Fraction(2) ** (E - bias)
    return vals
TAB = {n: table(n) for n in FMTS}
def finite_codes(n): return sorted((c for c, v in TAB[n].items() if isinstance(v, Fraction)), key=lambda c: TAB[n][c])
def dec_bits_f16(name, code, signbit):
    v = TAB[name][code]
    if v == 'nan': return 0x7FFF
    if v == 'inf': return 0x7C00 | (signbit << 15)
    return f16bits(float(v)) | (signbit << 15) if v != 0 else (signbit << 15)
def f16bits(x): return struct.unpack('<H', struct.pack('<e', x))[0]
def f32(b): return struct.unpack('<f', struct.pack('<I', b))[0]
def f16(b): return struct.unpack('<e', struct.pack('<H', b))[0]
def enc(name, x, relu):
    e, m, inf = FMTS[name]; nb = 1 + e + m
    if math.isnan(x): return 127 if name in ('e4m3', 'e5m2') else max(finite_codes(name), key=lambda c: TAB[name][c])
    sign = 1 if math.copysign(1, x) < 0 else 0
    if relu and sign and not math.isnan(x): return 0
    codes = finite_codes(name); top = codes[-1]
    if math.isinf(x): return (sign << (nb - 1)) | top
    a = abs(Fraction(x))
    best = min(codes, key=lambda c: (abs(TAB[name][c] - a), c & 1))     # nearest, ties -> even code
    if a > TAB[name][top]: best = top
    return (sign << (nb - 1)) | best

def rnd_f32():
    k = random.random()
    if k < .12: return random.choice([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x3F800000, 0xBF800000])
    if k < .5:   # near grid points and midpoints of some format
        n = random.choice(list(FMTS)); codes = finite_codes(n)
        c = random.choice(codes); v = TAB[n][c]
        v2 = TAB[n][random.choice(codes)]
        x = float((v + v2) / 2) if random.random() < .5 else float(v)
        x = x * random.choice([1, 1, -1, 1.0000001, 0.9999999])
        return struct.unpack('<I', struct.pack('<f', x))[0]
    return struct.unpack('<I', struct.pack('<f', random.uniform(-1, 1) * 10 ** random.randint(-6, 6)))[0]

cases = []   # (opcode, kind, rows)  kind: 'enc32' (2 f32 -> packed), 'enc16' (f16x2 -> packed), 'dec' (packed -> f16x2)
for name in FMTS:
    e, m, inf = FMTS[name]; nb = 1 + e + m
    for relu in ('', '.relu'):
        rows = []
        for _ in range(N):
            a, b = rnd_f32(), rnd_f32()
            ha, hb = enc(name, f32(a), relu != ''), enc(name, f32(b), relu != '')
            out = ((ha << 4) | hb) if nb <= 4 else ((ha << 8) | hb)
            rows.append(((a, b), out))
        cases.append((f'cvt.rn{relu}.satfinite.{name}x2.f32', 'enc32', nb, rows))
        rows = []
        for _ in range(N):
            ha16 = random.getrandbits(16); hb16 = random.getrandbits(16)
            if random.random() < .3: ha16 = f16bits(random.choice([0.0, -0.0, 1.0, 448.0, 6e4, -6e4, 0.00001, 6.0, 7.5]))
            vin = ha16 | (hb16 << 16)       # lane0 = ha16 (low), lane1 = hb16 (high)
            hi = enc(name, f16(hb16), relu != ''); lo = enc(name, f16(ha16), relu != '')
            out = ((hi << 4) | lo) if nb <= 4 else ((hi << 8) | lo)
            rows.append(((vin,), out))
        cases.append((f'cvt.rn{relu}.satfinite.{name}x2.f16x2', 'enc16', nb, rows))
    # decode: every element code (with sign) in both lanes
    allc = list(range(1 << nb)); rows = []
    for i in range(0, max(len(allc), N)):
        lo, hi = allc[i % len(allc)], allc[(i * 7 + 3) % len(allc)]
        packed = ((hi << 4) | lo) if nb <= 4 else ((hi << 8) | lo)
        def d(code):
            sbit = code >> (nb - 1); c = code & ((1 << (nb - 1)) - 1)
            return dec_bits_f16(name, c, sbit)
        rows.append(((packed,), d(lo) | (d(hi) << 16)))
    cases.append((f'cvt.rn.f16x2.{name}x2', 'dec', nb, rows))
# ue8m0
def ue(x, mode):
    if math.isnan(x): return 255
    # .satfinite: |x| > MAX_NORM (2^127), -Inf and large negatives included, gives MAX_NORM 0xfe (PTX ISA 9.3 cvt)
    if math.isinf(x) or abs(x) > 2.0 ** 127: return 254
    if x <= 0: return 0
    a = Fraction(x); e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a: e -= 1
    if mode == 'rp' and Fraction(2) ** e != a: e += 1
    return max(0, min(254, e + 127))
for mode in ('rz', 'rp'):
    rows = []
    for _ in range(N):
        a, b = rnd_f32(), rnd_f32()
        rows.append(((a, b), (ue(f32(a), mode) << 8) | ue(f32(b), mode)))
    cases.append((f'cvt.{mode}.satfinite.ue8m0x2.f32', 'enc32', 8, rows))
rows = []
for i in range(256):
    lo, hi = i, (i * 37 + 5) % 256
    def bf(c): return 0x7FFF if c == 255 else (0x0040 if c == 0 else c << 7)
    rows.append(((lo | (hi << 8),), bf(lo) | (bf(hi) << 16)))
cases.append(('cvt.rn.bf16x2.ue8m0x2', 'dec', 8, rows))

PART, PARTS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
cases = cases[PART::PARTS]
print("// GENERATED by scripts/gen_mini.py -- fp8/fp6/fp4/ue8m0 conversions vs. an enumerating reference.")
print(".version 8.0\n.target sm_100a\n.address_size 64\n")
for k, (op, kind, nb, rows) in enumerate(cases):
    nin = len(rows[0][0]); n = len(rows)
    ps = ", ".join(f".param .u64 p{j}" for j in range(nin + 1))
    inw = 16 if kind == 'dec' and nb > 4 else 8 if kind == 'dec' else 32
    outw = 32 if kind == 'dec' else (8 if nb <= 4 else 16)
    ildt = {8: 'b8', 16: 'b16', 32: 'b32'}[inw]; ostt = {8: 'b8', 16: 'b16', 32: 'b32'}[outw]
    print(f".visible .entry k{k}({ps})\n{{\n    .reg .b16 %h<6>; .reg .b32 %r<10>; .reg .b64 %rd<8>;\n    mov.u32 %r0, %tid.x;")
    for j in range(nin):
        print(f"    mul.wide.u32 %rd1, %r0, {inw // 8}; ld.param.u64 %rd2, [p{j}]; add.s64 %rd2, %rd2, %rd1; ld.global.{ildt} %r{j+4}, [%rd2];")
    srcs = ", ".join(f"%r{j+4}" for j in range(nin))
    # destination register class
    if kind == 'dec': print(f"    {op} %r8, {srcs};")
    else: print(f"    {op} %r8, {srcs};")
    print(f"    mul.wide.u32 %rd3, %r0, {outw // 8}; ld.param.u64 %rd4, [p{nin}]; add.s64 %rd4, %rd4, %rd3; st.global.{ostt} [%rd4], %r8;\n    ret;\n}}\n")
for k, (op, kind, nb, rows) in enumerate(cases):
    nin = len(rows[0][0]); n = len(rows)
    inw = 16 if kind == 'dec' and nb > 4 else 8 if kind == 'dec' else 32
    outw = 32 if kind == 'dec' else (8 if nb <= 4 else 16)
    T = {8: 'u8', 16: 'u16', 32: 'u32'}
    for j in range(nin): print(f".buffer I{k}_{j} .{T[inw]} = {{ {', '.join(str(r[0][j]) for r in rows)} }};")
    print(f".buffer O{k} .{T[outw]} [{n}];\n.launch k{k} <<< 1, {n} >>> ({', '.join([f'I{k}_{j}' for j in range(nin)] + [f'O{k}'])});")
    print(f".expect O{k} .{T[outw]} = {{ {', '.join(str(r[1]) for r in rows)} }};")

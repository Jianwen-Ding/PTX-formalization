#!/usr/bin/env python3
"""Generate a differential IEEE-754 test for the K semantics of PTX.

Reference results come from the host's hardware floating point via Python's
`struct` module (binary32/binary64 encode/decode) and native double arithmetic.
For binary32 add/sub/mul/div/sqrt, computing in binary64 and rounding once to
binary32 is exact (binary64 has more than 2p+2 bits), so the reference is the
correctly rounded IEEE result.  NaN results are compared against PTX's
canonical NaN (0x7FFFFFFF / 0x7FFFFFFFFFFFFFFF).

usage: gen_ieee_diff.py [N] [SEED] > tests/ieee-diff.ptx
"""
import math
from fractions import Fraction
import random
import struct
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 2026)

def f32(bits):  return struct.unpack('<f', struct.pack('<I', bits))[0]
def b32(x):     return struct.unpack('<I', struct.pack('<f', x))[0]
def f64(bits):  return struct.unpack('<d', struct.pack('<Q', bits))[0]
def b64(x):     return struct.unpack('<Q', struct.pack('<d', x))[0]

CN32, CN64 = 0x7FFFFFFF, 0x7FFFFFFFFFFFFFFF

def r32(x):
    """round a double to binary32 bits (overflow -> inf), canonical NaN"""
    if math.isnan(x): return CN32
    try:
        return b32(x)
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000

def rand32():
    k = random.random()
    if k < 0.08:   # specials and boundaries
        return random.choice([0, 0x80000000, 0x7F800000, 0xFF800000, 0x00000001,
                              0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x3F800000, 0xBF800000])
    if k < 0.20:   # subnormals
        return random.getrandbits(23) | (random.getrandbits(1) << 31)
    if k < 0.50:   # moderate exponents (interesting cancellations)
        return (random.getrandbits(1) << 31) | (random.randint(110, 145) << 23) | random.getrandbits(23)
    b = random.getrandbits(32)
    return b if (b >> 23) & 0xFF != 0xFF else b & ~(1 << 30)   # avoid NaN inputs

def rand64():
    if random.random() < 0.1:
        return random.getrandbits(52) | (random.getrandbits(1) << 63)   # subnormal
    b = random.getrandbits(64)
    return b if (b >> 52) & 0x7FF != 0x7FF else b & ~(1 << 62)

A = [rand32() for _ in range(N)]
B = [rand32() for _ in range(N)]
C = [rand32() for _ in range(N)]
DA = [rand64() for _ in range(N)]
DB = [rand64() for _ in range(N)]
DC = [rand64() for _ in range(N)]

def op32(f, a, b):
    x, y = f32(a), f32(b)
    try:
        return r32(f(x, y))
    except ZeroDivisionError:
        if math.isnan(x) or x == 0: return CN32
        neg = (math.copysign(1, x) * math.copysign(1, y)) < 0
        return 0xFF800000 if neg else 0x7F800000

def div(x, y): return x / y
def sqrt32(a):
    x = f32(a)
    if math.isnan(x) or x < 0: return CN32
    return r32(math.sqrt(x))

def op64(f, a, b):
    x, y = f64(a), f64(b)
    try:
        r = f(x, y)
    except ZeroDivisionError:
        if x == 0: return CN64
        neg = (math.copysign(1, x) * math.copysign(1, y)) < 0
        return 0xFFF0000000000000 if neg else 0x7FF0000000000000
    except OverflowError:
        return 0x7FF0000000000000 if (x * y) > 0 else 0xFFF0000000000000
    return CN64 if math.isnan(r) else b64(r)

def round_frac(q, p, emin, emax):
    """correctly round an exact rational to binary(p) (RN-even); returns bits"""
    W = {24: 32, 53: 64}[p]
    sign = 1 if q < 0 else 0
    q = abs(q)
    if q == 0:
        return sign << (W - 1)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if Fraction(2) ** e > q: e -= 1          # 2^e <= q < 2^(e+1)
    e = max(e, emin)                           # subnormal range
    scaled = q / Fraction(2) ** (e - (p - 1))  # integer part has <= p bits
    n, rem = divmod(scaled.numerator, scaled.denominator)
    if 2 * rem > scaled.denominator or (2 * rem == scaled.denominator and n % 2 == 1):
        n += 1
    if n == 1 << p:                            # carry into next binade
        n >>= 1; e += 1
    ebits = {24: 8, 53: 11}[p]
    if e > emax:
        return (sign << (W - 1)) | (((1 << ebits) - 1) << (p - 1))
    if n < 1 << (p - 1):                       # subnormal
        return (sign << (W - 1)) | n
    return (sign << (W - 1)) | ((e - emin + 1) << (p - 1)) | (n - (1 << (p - 1)))

def fma_ref(x, y, z, p, emin, emax, canon):
    if any(math.isnan(v) or math.isinf(v) for v in (x, y, z)):
        try:
            r = x * y + z if not (math.isinf(x * y) and math.isinf(z)) else x * y + z
        except OverflowError:
            r = math.nan
        if math.isnan(r): return canon
        return b64(r) if p == 53 else r32(r)
    q = Fraction(x) * Fraction(y) + Fraction(z)
    if q == 0:  # exact zero: sign per IEEE (RN): -0 only if both addends are -0
        prod_neg = (math.copysign(1, x) * math.copysign(1, y)) < 0
        neg = prod_neg and math.copysign(1, z) < 0
        W = 64 if p == 53 else 32
        return (1 << (W - 1)) if neg else 0
    return round_frac(q, p, emin, emax)

def mul64(x, y):
    r = x * y
    return r

def narrow(d):   # cvt.rn.f32.f64
    x = f64(d)
    return CN32 if math.isnan(x) else r32(x)

def widen(a):    # cvt.f64.f32
    x = f32(a)
    return CN64 if math.isnan(x) else b64(x)

res = {
    'ADD': [op32(lambda x, y: x + y, a, b) for a, b in zip(A, B)],
    'SUB': [op32(lambda x, y: x - y, a, b) for a, b in zip(A, B)],
    'MUL': [op32(lambda x, y: x * y, a, b) for a, b in zip(A, B)],
    'DIV': [op32(div, a, b) for a, b in zip(A, B)],
    'SQRT': [sqrt32(a) for a in A],
    'WIDE': [widen(a) for a in A],
    'DADD': [op64(lambda x, y: x + y, a, b) for a, b in zip(DA, DB)],
    'DMUL': [op64(mul64, a, b) for a, b in zip(DA, DB)],
    'DDIV': [op64(div, a, b) for a, b in zip(DA, DB)],
    'NARROW': [narrow(d) for d in DA],
    'FMA': [fma_ref(f32(a), f32(b), f32(c), 24, -126, 127, CN32) for a, b, c in zip(A, B, C)],
    'DFMA': [fma_ref(f64(a), f64(b), f64(c), 53, -1022, 1023, CN64) for a, b, c in zip(DA, DB, DC)],
}
# cross-check the exact-rational f64 FMA reference against the C library fma()
for a, b, c, r in zip(DA, DB, DC, res['DFMA']):
    try:
        m = math.fma(f64(a), f64(b), f64(c))
    except (ValueError, OverflowError):
        continue
    assert (CN64 if math.isnan(m) else b64(m)) == r, (hex(a), hex(b), hex(c))

def h32(v): return '0f%08X' % v
def h64(v): return '0d%016X' % v
def lst(vals, h): return '{ ' + ', '.join(h(v) for v in vals) + ' }'

print(f"""// GENERATED by scripts/gen_ieee_diff.py -- differential IEEE-754 test.
// Expected values are bit patterns computed with the host's hardware FPU.
.version 8.0
.target sm_80
.address_size 64

.visible .entry ieee(
    .param .u64 p_a, .param .u64 p_b, .param .u64 p_da, .param .u64 p_db,
    .param .u64 p_add, .param .u64 p_sub, .param .u64 p_mul, .param .u64 p_div,
    .param .u64 p_sqrt, .param .u64 p_wide, .param .u64 p_dadd, .param .u64 p_dmul,
    .param .u64 p_ddiv, .param .u64 p_narrow, .param .u64 p_c, .param .u64 p_dc,
    .param .u64 p_fma, .param .u64 p_dfma)
{{
    .reg .b32   %r<2>;
    .reg .f32   %f<10>;
    .reg .f64   %fd<10>;
    .reg .b64   %rd<40>;

    mov.u32         %r1, %tid.x;
    mul.wide.u32    %rd1, %r1, 4;
    mul.wide.u32    %rd2, %r1, 8;
    ld.param.u64    %rd3, [p_a];     add.s64 %rd3, %rd3, %rd1;  ld.global.f32 %f1, [%rd3];
    ld.param.u64    %rd4, [p_b];     add.s64 %rd4, %rd4, %rd1;  ld.global.f32 %f2, [%rd4];
    ld.param.u64    %rd5, [p_da];    add.s64 %rd5, %rd5, %rd2;  ld.global.f64 %fd1, [%rd5];
    ld.param.u64    %rd6, [p_db];    add.s64 %rd6, %rd6, %rd2;  ld.global.f64 %fd2, [%rd6];

    add.rn.f32 %f3, %f1, %f2;  ld.param.u64 %rd7, [p_add];  add.s64 %rd7, %rd7, %rd1;  st.global.f32 [%rd7], %f3;
    sub.rn.f32 %f4, %f1, %f2;  ld.param.u64 %rd8, [p_sub];  add.s64 %rd8, %rd8, %rd1;  st.global.f32 [%rd8], %f4;
    mul.rn.f32 %f5, %f1, %f2;  ld.param.u64 %rd9, [p_mul];  add.s64 %rd9, %rd9, %rd1;  st.global.f32 [%rd9], %f5;
    div.rn.f32 %f6, %f1, %f2;  ld.param.u64 %rd10, [p_div]; add.s64 %rd10, %rd10, %rd1; st.global.f32 [%rd10], %f6;
    sqrt.rn.f32 %f7, %f1;      ld.param.u64 %rd11, [p_sqrt]; add.s64 %rd11, %rd11, %rd1; st.global.f32 [%rd11], %f7;
    cvt.f64.f32 %fd3, %f1;     ld.param.u64 %rd12, [p_wide]; add.s64 %rd12, %rd12, %rd2; st.global.f64 [%rd12], %fd3;
    add.rn.f64 %fd4, %fd1, %fd2; ld.param.u64 %rd13, [p_dadd]; add.s64 %rd13, %rd13, %rd2; st.global.f64 [%rd13], %fd4;
    mul.rn.f64 %fd5, %fd1, %fd2; ld.param.u64 %rd14, [p_dmul]; add.s64 %rd14, %rd14, %rd2; st.global.f64 [%rd14], %fd5;
    div.rn.f64 %fd6, %fd1, %fd2; ld.param.u64 %rd15, [p_ddiv]; add.s64 %rd15, %rd15, %rd2; st.global.f64 [%rd15], %fd6;
    ld.param.u64    %rd17, [p_c];    add.s64 %rd17, %rd17, %rd1; ld.global.f32 %f9, [%rd17];
    ld.param.u64    %rd18, [p_dc];   add.s64 %rd18, %rd18, %rd2; ld.global.f64 %fd7, [%rd18];
    fma.rn.f32 %f9, %f1, %f2, %f9;   ld.param.u64 %rd19, [p_fma];  add.s64 %rd19, %rd19, %rd1; st.global.f32 [%rd19], %f9;
    fma.rn.f64 %fd7, %fd1, %fd2, %fd7; ld.param.u64 %rd20, [p_dfma]; add.s64 %rd20, %rd20, %rd2; st.global.f64 [%rd20], %fd7;
    cvt.rn.f32.f64 %f8, %fd1;  ld.param.u64 %rd16, [p_narrow]; add.s64 %rd16, %rd16, %rd1; st.global.f32 [%rd16], %f8;
    ret;
}}

.buffer A .f32 = {lst(A, h32)};
.buffer B .f32 = {lst(B, h32)};
.buffer DA .f64 = {lst(DA, h64)};
.buffer DB .f64 = {lst(DB, h64)};
.buffer C .f32 = {lst(C, h32)};
.buffer DC .f64 = {lst(DC, h64)};""")
for name in res:
    t = '.f64' if name in ('WIDE', 'DADD', 'DMUL', 'DDIV', 'DFMA') else '.f32'
    print(f".buffer {name} {t} [{N}];")
print(f".launch ieee <<< 1, {N} >>> (A, B, DA, DB, ADD, SUB, MUL, DIV, SQRT, WIDE, DADD, DMUL, DDIV, NARROW, C, DC, FMA, DFMA);")
for name, vals in res.items():
    t, h = ('.f64', h64) if name in ('WIDE', 'DADD', 'DMUL', 'DDIV', 'DFMA') else ('.f32', h32)
    print(f".expect {name} {t} = {lst(vals, h)};")

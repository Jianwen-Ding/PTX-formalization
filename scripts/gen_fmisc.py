#!/usr/bin/env python3
"""Generate tests/fmisc-*.ptx: float min/max (.ftz .NaN .xorsign.abs), abs/neg .ftz,
selp on floats, and special-value behaviour of the approximate functions.
usage: gen_fmisc.py [N] [SEED]"""
import math, random, struct, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
def f(b, w): return struct.unpack('<f' if w == 32 else '<d', struct.pack('<I' if w == 32 else '<Q', b))[0]
def bits(x, w):
    try:
        return struct.unpack('<I' if w == 32 else '<Q', struct.pack('<f' if w == 32 else '<d', x))[0]
    except OverflowError:       # beyond f32 range -> infinity of the same sign
        return bits(math.copysign(math.inf, x), w)
SIGN = {32: 1 << 31, 64: 1 << 63}
CN = {32: 0x7FFFFFFF, 64: 0x7FFFFFFFFFFFFFFF}
def isnan(b, w): return math.isnan(f(b, w))
def subn(b, w):
    p, eb = (23, 8) if w == 32 else (52, 11)
    return (b >> p) & ((1 << eb) - 1) == 0 and b & ((1 << p) - 1) != 0
def flush(b, w): return b & SIGN[w] if subn(b, w) else b
def rnd(w):
    sp = [0, SIGN[w], 1, SIGN[w] | 1, bits(1.0, w), bits(-1.0, w), bits(math.inf, w), bits(-math.inf, w),
          CN[w], bits(2.5, w), bits(-2.5, w), (1 << (23 if w == 32 else 52)) - 1, 1 << (23 if w == 32 else 52)]
    if random.random() < .6: return random.choice(sp)
    return random.getrandbits(w) if random.random() < .5 else bits(random.uniform(-4, 4), w)

def minmax(op, a, b, w, ftz, nan, xs):
    if ftz: a, b = flush(a, w), flush(b, w)
    s = ((a >> (w - 1)) ^ (b >> (w - 1))) & 1
    if xs: a, b = a & ~SIGN[w], b & ~SIGN[w]
    na, nb = isnan(a, w), isnan(b, w)
    if na or nb:
        if nan or (na and nb): return CN[w]
        r = b if na else a
    else:
        x, y = f(a, w), f(b, w)
        if x == y:     # equal values: only differently-signed zeros differ; -0 < +0
            r = (a | b) if op == 'min' else (a & b)
            if x != 0: r = a
        else:
            r = (a if x < y else b) if op == 'min' else (a if x > y else b)
    if xs: r = (r & ~SIGN[w]) | (s << (w - 1))
    return r

cases = []   # (ptx text, width, inputs, outputs)
for w, ty in ((32, 'f32'), (64, 'f64')):
    mods_list = ['', '.NaN', '.xorsign.abs', '.NaN.xorsign.abs'] + (['.ftz', '.ftz.NaN'] if w == 32 else [])
    for op in ('min', 'max'):
        for mods in mods_list:
            A = [rnd(w) for _ in range(N)]; B = [rnd(w) for _ in range(N)]
            cases.append((f'{op}{mods}.{ty}', w, [A, B],
                          [minmax(op, A[i], B[i], w, '.ftz' in mods, '.NaN' in mods, 'xorsign' in mods) for i in range(N)]))
    X = [rnd(w) for _ in range(N)]
    cases.append((f'abs.{ty}', w, [X], [x & ~SIGN[w] for x in X]))
    cases.append((f'neg.{ty}', w, [X], [x ^ SIGN[w] for x in X]))
    if w == 32:
        cases.append(('abs.ftz.f32', w, [X], [flush(x, w) & ~SIGN[w] for x in X]))
        cases.append(('neg.ftz.f32', w, [X], [flush(x, w) ^ SIGN[w] for x in X]))
# special values of approximate functions (finite normal results are not bit-exact vs hardware, so avoid them)
SPECIAL = [0, SIGN[32], bits(math.inf, 32), bits(-math.inf, 32), CN[32], 1, SIGN[32] | 1, bits(1.0, 32), bits(-1.0, 32)]
def sp(fun):
    out = []
    for b in SPECIAL:
        x = f(b, 32)
        try: r = fun(x)
        except Exception: r = None
        out.append(r)
    return out
def b32r(r, x):
    if r is None or (isinstance(r, float) and math.isnan(r)): return CN[32]
    return bits(r, 32)
X = SPECIAL
def rsq(x): return math.inf if x == 0 and math.copysign(1, x) > 0 else -math.inf if x == 0 else (math.nan if x < 0 else (0.0 if math.isinf(x) else 1 / math.sqrt(x)))
def ex2(x): return 0.0 if x == -math.inf else (math.inf if x == math.inf else (math.nan if math.isnan(x) else 2.0 ** x))
def lg2(x): return -math.inf if x == 0 else (math.nan if x < 0 or math.isnan(x) else (math.inf if math.isinf(x) else math.log2(x)))
def rcp(x): return math.copysign(math.inf, x) if x == 0 else (math.copysign(0.0, x) if math.isinf(x) else 1.0 / x)
def sin_(x): return math.nan if math.isinf(x) or math.isnan(x) else math.sin(x)
def cos_(x): return math.nan if math.isinf(x) or math.isnan(x) else math.cos(x)
for name, fun, ok in (('rsqrt.approx.f32', rsq, lambda x: abs(x) in (0.0, math.inf) or x < 0 or x == 1.0 or math.isnan(x)),
                      ('ex2.approx.f32', ex2, lambda x: x in (-math.inf, math.inf, 0.0, 1.0, -1.0) or math.isnan(x) or abs(x) < 1e-30),
                      ('lg2.approx.f32', lg2, lambda x: x in (0.0, math.inf, 1.0) or x < 0 or math.isnan(x)),
                      ('rcp.approx.ftz.f32', rcp, lambda x: x in (0.0, -0.0, math.inf, -math.inf, 1.0, -1.0) or math.isnan(x) or abs(x) < 1e-30),
                      ('sin.approx.f32', sin_, lambda x: math.isinf(x) or math.isnan(x) or abs(x) < 1e-30),
                      ('cos.approx.f32', cos_, lambda x: math.isinf(x) or math.isnan(x) or x == 0 or abs(x) < 1e-30)):
    rows = [(b,) for b in SPECIAL if ok(f(b, 32))]
    ins = [r[0] for r in rows]
    outs = []
    for b in ins:
        x = f(b, 32)
        if '.ftz' in name: x = f(flush(b, 32), 32)
        r = fun(x)
        outs.append(CN[32] if (isinstance(r, float) and math.isnan(r)) else bits(r, 32))
    cases.append((name, 32, [ins], outs))
# selp on floats (regression: used to be ambiguous with fma)
A = [rnd(32) for _ in range(N)]; B = [rnd(32) for _ in range(N)]; P = [random.getrandbits(1) for _ in range(N)]
cases.append(('selp.f32', 32, [A, B, P], [A[i] if P[i] else B[i] for i in range(N)]))

PART, PARTS = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
cases = cases[PART::PARTS]
print("// GENERATED by scripts/gen_fmisc.py -- float min/max, abs/neg, approximate functions at special values.")
print(".version 8.0\n.target sm_80\n.address_size 64\n")
for k, (text, w, ins, outs) in enumerate(cases):
    n = len(ins); rt = 'f32' if w == 32 else 'f64'; rg = '%f' if w == 32 else '%fd'; sz = w // 8
    ps = ", ".join(f".param .u64 p{j}" for j in range(n + 1))
    print(f".visible .entry k{k}({ps})\n{{\n    .reg .b32 %r<4>; .reg .f32 %f<8>; .reg .f64 %fd<8>; .reg .b64 %rd<6>; .reg .pred %p<2>;\n    mov.u32 %r0, %tid.x; mul.wide.u32 %rd1, %r0, {sz};")
    for j in range(n):
        if text.startswith('selp') and j == 2:
            print(f"    ld.param.u64 %rd2, [p{j}]; mul.wide.u32 %rd5, %r0, 4; add.s64 %rd2, %rd2, %rd5; ld.global.b32 %r2, [%rd2]; setp.ne.u32 %p1, %r2, 0;")
        else:
            print(f"    ld.param.u64 %rd2, [p{j}]; add.s64 %rd2, %rd2, %rd1; ld.global.{rt} {rg}{1+j}, [%rd2];")
    src = ", ".join([f"{rg}{1+j}" for j in range(n)])
    if text.startswith('selp'): src = f"{rg}1, {rg}2, %p1"
    print(f"    {text} {rg}7, {src};")
    print(f"    ld.param.u64 %rd3, [p{n}]; add.s64 %rd3, %rd3, %rd1; st.global.{rt} [%rd3], {rg}7;\n    ret;\n}}\n")
for k, (text, w, ins, outs) in enumerate(cases):
    T = 'u32' if w == 32 else 'u64'
    for j, col in enumerate(ins):
        tj = 'u32' if (text.startswith('selp') and j == 2) else T
        print(f".buffer I{k}_{j} .{tj} = {{ {', '.join(map(str, col))} }};")
    print(f".buffer O{k} .{T} [{len(outs)}];\n.launch k{k} <<< 1, {len(outs)} >>> ({', '.join([f'I{k}_{j}' for j in range(len(ins))] + [f'O{k}'])});")
    print(f".expect O{k} .{T} = {{ {', '.join(map(str, outs))} }};")

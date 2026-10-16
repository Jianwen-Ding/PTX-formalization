#!/usr/bin/env python3
"""Import ZLUDA's hardware-validated PTX conformance programs as K tests.

ZLUDA (https://github.com/vosen/ZLUDA, MIT/Apache-2.0) ships ~190 small PTX
kernels together with the inputs and outputs they produce on real hardware
(ptx/src/test/spirv_run/*.ptx and the `test_ptx!` table in mod.rs).  This
script turns each of them into a self-contained program for this semantics by
appending host-harness directives:

    .buffer IN .u8 = { <input bytes> };  .buffer OUT .u8 [n];
    .launch kernel <<< 1, T >>> (IN, OUT);  .expect OUT .u8 = { <bytes> };

usage: import_zluda.py PATH/TO/ZLUDA OUTDIR
"""
import math, os, re, struct, sys

root, outdir = sys.argv[1], sys.argv[2]
spirv = os.path.join(root, 'ptx/src/test/spirv_run')
src = open(os.path.join(spirv, 'mod.rs')).read()

# strip // comments (keeps strings; there are none with //)
src = re.sub(r'//[^\n]*', '', src)

def find_calls(text):
    for m in re.finditer(r'^test_ptx(_with_32|_warp)?!\(', text, re.M):
        i = m.end(); depth = 1; j = i
        while depth:
            c = text[j]
            depth += (c in '([{') - (c in ')]}')
            j += 1
        yield m.group(1) or '', text[i:j - 1]

def split_top(s):
    parts, depth, cur = [], 0, ''
    for c in s:
        if c in '([{': depth += 1
        if c in ')]}': depth -= 1
        if c == ',' and depth == 0:
            parts.append(cur); cur = ''
        else:
            cur += c
    if cur.strip(): parts.append(cur)
    return [p.strip() for p in parts]

TYPES = {'u8': ('B', 1), 'u16': ('H', 2), 'u32': ('I', 4), 'u64': ('Q', 8),
         'i8': ('b', 1), 'i16': ('h', 2), 'i32': ('i', 4), 'i64': ('q', 8),
         'f32': ('f', 4), 'f64': ('d', 8)}

def eval_elem(tok, ty):
    tok = tok.strip()
    num = lambda t: t.replace('_', '')
    m = re.fullmatch(r'(-?)(0x[0-9a-fA-F_]+|0b[01_]+|[0-9][0-9_]*)(u8|u16|u32|u64|i8|i16|i32|i64)?', tok)
    if m and not re.fullmatch(r'\d+\.\d*', tok):
        sign, lit, t = m.groups()
        v = int(num(lit), 0) * (-1 if sign else 1)
        return v, t
    m = re.fullmatch(r'(-?)([0-9][0-9_]*\.[0-9_]*(?:e-?\d+)?|[0-9][0-9_]*e-?\d+|[0-9][0-9_]*(?=f(?:32|64)))(f32|f64)?', tok)
    if m:
        return float(num(m.group(2))) * (-1 if m.group(1) else 1), m.group(3)
    m = re.fullmatch(r'f(32|64)::(?:from_?bits)\((.*)\)', tok)
    if m:
        bits = int(num(m.group(2)), 0)
        return (struct.unpack('<f', struct.pack('<I', bits))[0] if m.group(1) == '32'
                else struct.unpack('<d', struct.pack('<Q', bits))[0]), 'f' + m.group(1)
    m = re.fullmatch(r'(u8|u16|u32|u64|i8|i16|i32|i64)::(MAX|MIN|max_?value\(\)|min_?value\(\))', tok)
    if m:
        t, w = m.group(1), TYPES[m.group(1)][1] * 8
        mx = m.group(2).startswith('MAX') or m.group(2).startswith('max')
        if t[0] == 'u': return ((1 << w) - 1 if mx else 0), t
        return ((1 << (w - 1)) - 1 if mx else -(1 << (w - 1))), t
    m = re.fullmatch(r'f(32|64)::(NAN|INFINITY|NEG_?INFINITY)', tok)
    if m:
        return {'NAN': math.nan, 'INFINITY': math.inf, 'NEGINFINITY': -math.inf, 'NEG_INFINITY': -math.inf}[m.group(2)], 'f' + m.group(1)
    m = re.fullmatch(r'std::f(32|64)::consts::PI(?:\s*/\s*([0-9.]+)(?:f32|f64)?)?', tok)
    if m:
        v = math.pi / (float(m.group(2)) if m.group(2) else 1.0)
        return v, 'f' + m.group(1)
    raise ValueError(tok)

def to_bytes(expr):
    expr = expr.strip()
    m = re.fullmatch(r'\[(.*)\];?\s*', expr, re.S)
    if not m: raise ValueError('not an array: ' + expr)
    body = m.group(1)
    rep = re.fullmatch(r'(.*);\s*(\d+)', body, re.S)
    if rep:
        elems = [rep.group(1)] * int(rep.group(2))
    else:
        elems = split_top(body)
    vals = [eval_elem(e, None) for e in elems]
    ty = next((t for _, t in vals if t), None)
    if ty is None:
        ty = 'f32' if any(isinstance(v, float) for v, _ in vals) else 'u32'    # untyped: assume 32-bit
    code, size = TYPES[ty]
    out = bytearray()
    for v, _ in vals:
        if ty[0] == 'f':
            out += struct.pack('<' + code, v)
        else:
            w = size * 8
            out += (int(v) & ((1 << w) - 1)).to_bytes(size, 'little')
    return bytes(out)

os.makedirs(outdir, exist_ok=True)
index = []
for kind, body in find_calls(src):
    parts = split_top(body)
    name = parts[0]
    path = os.path.join(spirv, name + '.ptx')
    if not os.path.exists(path):
        continue
    try:
        if kind == '_warp':
            inp, out, threads = b'', to_bytes(parts[1]), 64
        else:
            inp, out, threads = to_bytes(parts[1]), to_bytes(parts[2]), 1
    except IndexError:
        continue        # llvm-only test with no recorded inputs/outputs
    except Exception as e:
        print(f'skip {name}: {e}', file=sys.stderr)
        continue
    ptx = open(path).read()
    kernel = re.search(r'\.entry\s+(\w+)', ptx).group(1)
    nparams = len(re.search(r'\.entry\s+\w+\s*\((.*?)\)', ptx, re.S).group(1).split('.param')) - 1
    hx = lambda b: ', '.join(str(x) for x in b) if b else None
    harness = [f'\n// ---- harness (generated by scripts/import_zluda.py) ----']
    if inp:
        harness.append(f'.buffer IN .u8 = {{ {hx(inp)} }};')
    harness.append(f'.buffer OUT .u8 [{len(out)}];')
    args = ('IN, OUT' if inp else 'OUT') if nparams == (2 if inp else 1) else ', '.join(['IN'] * (nparams - 1) + ['OUT'])
    harness.append(f'.launch {kernel} <<< 1, {threads} >>> ({args});')
    harness.append(f'.expect OUT .u8 = {{ {hx(out)} }};')
    header = ('// From ZLUDA (https://github.com/vosen/ZLUDA, MIT OR Apache-2.0), '
              f'ptx/src/test/spirv_run/{name}.ptx;\n// inputs/outputs from its mod.rs test table.\n')
    open(os.path.join(outdir, name + '.ptx'), 'w').write(header + ptx + '\n'.join(harness) + '\n')
    index.append(name)
print(len(index), 'tests written to', outdir)

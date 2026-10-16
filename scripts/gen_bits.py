#!/usr/bin/env python3
"""Generate tests/bits.ptx: bit-field instructions checked against literal
transcriptions of the PTX ISA pseudo-code (bit-by-bit loops, no tricks).
usage: gen_bits.py [N] [SEED]"""
import random, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
M32, M64 = (1 << 32) - 1, (1 << 64) - 1

def bit(v, i): return (v >> i) & 1 if i >= 0 else 0

def bfe(a, b, c, w, signed):
    msb = w - 1; pos = b & 0xff; ln = c & 0xff
    sbit = 0 if (not signed or ln == 0) else bit(a, min(pos + ln - 1, msb))
    d = 0
    for i in range(w):
        d |= (bit(a, pos + i) if (i < ln and pos + i <= msb) else sbit) << i
    return d

def bfi(a, b, c, d, w):
    pos = c & 0xff; ln = d & 0xff; f = b
    for i in range(ln):
        if pos + i <= w - 1:
            f = (f & ~(1 << (pos + i))) | (bit(a, i) << (pos + i))
    return f & ((1 << w) - 1)

def bfind(a, w, signed, shiftamt):
    if signed and bit(a, w - 1): a = ~a & ((1 << w) - 1)
    d = 0xFFFFFFFF
    for i in range(w - 1, -1, -1):
        if bit(a, i): d = i; break
    if shiftamt and d != 0xFFFFFFFF: d = w - 1 - d
    return d

def fns(mask, base, offset):          # offset is signed
    d = 0xFFFFFFFF
    if offset == 0:
        if bit(mask, base): d = base
    else:
        pos = base; count = abs(offset) - 1; inc = 1 if offset > 0 else -1
        while 0 <= pos < 32:
            if bit(mask, pos):
                if count == 0: d = pos; break
                count -= 1
            pos += inc
    return d

def bmsk(a, b, mode):
    a1 = a & 0xff; b1 = b & 0xff
    if mode == 'clamp': pos = min(a1, 32); ln = min(b1, 32)
    else: pos = a1 & 31; ln = b1 & 31
    m = 0
    for i in range(ln):
        if pos + i < 32: m |= 1 << (pos + i)
    return m

def szext(a, b, mode, signed):
    if mode == 'clamp' and b >= 32: return a
    n = b if mode == 'clamp' else b & 31
    if n == 0: return 0
    low = a & ((1 << n) - 1)
    if signed and bit(a, n - 1): low |= M32 & ~((1 << n) - 1)
    return low & M32

# prmt tables (rows: c[1:0]; columns: result byte) from the PTX ISA / gpgpu-sim
T = {
 'f4e': [[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]],
 'b4e': [[0,7,6,5],[1,0,7,6],[2,1,0,7],[3,2,1,0]],
 'rc8': [[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]],
 'ecl': [[0,1,2,3],[1,1,2,3],[2,2,2,3],[3,3,3,3]],
 'ecr': [[0,0,0,0],[0,1,1,1],[0,1,2,2],[0,1,2,3]],
 'rc16':[[0,1,0,1],[2,3,2,3],[0,1,0,1],[2,3,2,3]],
}
def prmt(a, b, c, mode):
    v = a | (b << 32); d = 0
    for i in range(4):
        if mode is None:
            sel = (c >> (4 * i)) & 0xf
            byte = (v >> (8 * (sel & 7))) & 0xff
            if sel & 8: byte = 0xff if byte & 0x80 else 0
        else:
            byte = (v >> (8 * T[mode][c & 3][i])) & 0xff
        d |= byte << (8 * i)
    return d

def r32():
    return random.choice([0, M32, 0x80000000, 0x7FFFFFFF, 1, 0xF0F0F0F0]) if random.random() < 0.25 else random.getrandbits(32)
def r64():
    return random.choice([0, M64, 1 << 63, (1 << 63) - 1, 1]) if random.random() < 0.25 else random.getrandbits(64)
def rpos(w): return random.choice([0, 1, w - 1, w, w + 1, 255, 256 + 3]) if random.random() < 0.3 else random.randint(0, w + 4)
def s32(x): return x - (1 << 32) if x >> 31 else x

cases = []   # (name, [inputs as u64 words], result bits, result width)
def case(op, args, res, rw): cases.append((op, args, res, rw))

for _ in range(N):
    a, a64 = r32(), r64(); p, l = rpos(32), rpos(32); p6, l6 = rpos(64), rpos(64)
    case('bfe.u32', [a, p, l], bfe(a, p, l, 32, False), 32)
    case('bfe.s32', [a, p, l], bfe(a, p, l, 32, True), 32)
    case('bfe.u64', [a64, p6, l6], bfe(a64, p6, l6, 64, False), 64)
    case('bfe.s64', [a64, p6, l6], bfe(a64, p6, l6, 64, True), 64)
    b, b64 = r32(), r64()
    case('bfi.b32', [a, b, p, l], bfi(a, b, p, l, 32), 32)
    case('bfi.b64', [a64, b64, p6, l6], bfi(a64, b64, p6, l6, 64), 64)
    for ty, w, v in (('u32', 32, a), ('s32', 32, a), ('u64', 64, a64), ('s64', 64, a64)):
        case(f'bfind.{ty}', [v], bfind(v, w, ty[0] == 's', False), 32)
        case(f'bfind.shiftamt.{ty}', [v], bfind(v, w, ty[0] == 's', True), 32)
    m, base, off = r32(), random.randint(0, 31), random.randint(-6, 6)
    case('fns.b32', [m, base, off & M32], fns(m, base, off), 32)
    for mode in ('clamp', 'wrap'):
        pa, pb = rpos(32), rpos(32)
        case(f'bmsk.{mode}.b32', [pa, pb], bmsk(pa, pb, mode), 32)
        nz = random.choice([0, 1, 5, 31, 32, 33, 63, 200, random.randint(0, 40)])
        case(f'szext.{mode}.u32', [a, nz], szext(a, nz, mode, False), 32)
        case(f'szext.{mode}.s32', [a, nz], szext(a, nz, mode, True), 32)
    c = random.getrandbits(16)
    case('prmt.b32', [a, b, c], prmt(a, b, c, None), 32)
    for mode in T:
        case(f'prmt.b32.{mode}', [a, b, random.getrandbits(8)], None, 32)
        cases[-1] = (cases[-1][0], cases[-1][1], prmt(a, b, cases[-1][1][2], mode), 32)

# group by opcode: one kernel per opcode, one thread per case
by_op = {}
for op, args, res, rw in cases: by_op.setdefault(op, []).append((args, res, rw))
print("// GENERATED by scripts/gen_bits.py -- bit-field instructions vs. literal ISA pseudo-code.")
print(".version 8.0\n.target sm_80\n.address_size 64\n")
bufs = []
for k, (op, rows) in enumerate(by_op.items()):
    nargs = len(rows[0][0]); rw = rows[0][2]
    is64 = rw == 64
    in64 = op.endswith('64') and not op.startswith('bfind')  # first operand width
    # bfe/bfi 64-bit take a 64-bit value operand(s) and 32-bit position/length operands
    ld = []
    sizes = []
    for j in range(nargs):
        wide = in64 and ((op.startswith('bfe') and j == 0) or (op.startswith('bfi') and j <= 1))
        wide = wide or (op.startswith('bfind') and op.endswith('64') and j == 0)
        sizes.append(8 if wide else 4)
    ps = ", ".join(f".param .u64 p{j}" for j in range(nargs + 1))
    print(f".visible .entry k{k}({ps})\n{{\n    .reg .b32 %r<16>;\n    .reg .b64 %rd<16>;\n    mov.u32 %r0, %tid.x;")
    for j in range(nargs):
        print(f"    mul.wide.u32 %rd1, %r0, {sizes[j]}; ld.param.u64 %rd2, [p{j}]; add.s64 %rd2, %rd2, %rd1;")
        if sizes[j] == 8: print(f"    ld.global.b64 %rd{4+j}, [%rd2];")
        else: print(f"    ld.global.b32 %r{4+j}, [%rd2];")
    ops = ", ".join((f"%rd{4+j}" if sizes[j] == 8 else f"%r{4+j}") for j in range(nargs))
    dst = "%rd12" if is64 else "%r12"
    print(f"    {op} {dst}, {ops};")
    print(f"    mul.wide.u32 %rd1, %r0, {8 if is64 else 4}; ld.param.u64 %rd3, [p{nargs}]; add.s64 %rd3, %rd3, %rd1;")
    print(f"    st.global.{'b64' if is64 else 'b32'} [%rd3], {dst};\n    ret;\n}}\n")
    bufs.append((k, op, rows, sizes, is64))
for k, op, rows, sizes, is64 in bufs:
    nargs = len(sizes)
    for j in range(nargs):
        t = 'u64' if sizes[j] == 8 else 'u32'
        print(f".buffer I{k}_{j} .{t} = {{ " + ", ".join(str(r[0][j] & (M64 if sizes[j] == 8 else M32)) for r in rows) + " };")
    t = 'u64' if is64 else 'u32'
    print(f".buffer O{k} .{t} [{len(rows)}];")
    print(f".launch k{k} <<< 1, {len(rows)} >>> (" + ", ".join([f"I{k}_{j}" for j in range(nargs)] + [f"O{k}"]) + ");")
    print(f".expect O{k} .{t} = {{ " + ", ".join(str(r[1]) for r in rows) + " };")

#!/usr/bin/env python3
"""Generate tests/multimem.ptx: multimem.ld_reduce / st / red over three device buffers."""
import struct, math
from fractions import Fraction as Fr
M32 = (1 << 32) - 1; M64 = (1 << 64) - 1
ND = 3
dev = [bytearray(64) for _ in range(ND)]
def w(d, off, n, v): dev[d][off:off + n] = (v & ((1 << 8 * n) - 1)).to_bytes(n, 'little')
def r(d, off, n): return int.from_bytes(dev[d][off:off + n], 'little')
def f32b(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def f16b(x): return struct.unpack('<H', struct.pack('<e', x))[0]
def f16v(b): return struct.unpack('<e', struct.pack('<H', b))[0]
def f32v(b): return struct.unpack('<f', struct.pack('<I', b))[0]
# fp8 tables
def e4m3v(c):
    s = -1 if c & 0x80 else 1; e = (c >> 3) & 15; m = c & 7
    if e == 15 and m == 7: return float('nan')
    return s * (m / 8 if e == 0 and False else 0) if False else s * ((m / 8) * 2 ** -6 if e == 0 else (1 + m / 8) * 2 ** (e - 7))
def e5m2v(c):
    s = -1 if c & 0x80 else 1; e = (c >> 2) & 31; m = c & 3
    if e == 31: return float('inf') if m == 0 else float('nan')
    return s * ((m / 4) * 2 ** -14 if e == 0 else (1 + m / 4) * 2 ** (e - 15))
def enc(vf, x):
    best = None
    for c in range(256):
        v = vf(c)
        if v != v or abs(v) == float('inf'): continue
        d = abs(v - x)
        if best is None or d < best[0] or (d == best[0] and c % 2 == 0 and best[1] % 2 == 1): best = (d, c)
    return best[1]
E4 = (e4m3v, lambda x: enc(e4m3v, x)); E5 = (e5m2v, lambda x: enc(e5m2v, x))
# device contents
for d in range(ND):
    for i in range(16): w(d, 4 * i, 4, (d + 1) * 1000 + i * 17 + 3)
w(0, 0, 4, 0xF0F0F0F0); w(1, 0, 4, 0xFF00FF00); w(2, 0, 4, 0x0FFF0FFF)
w(0, 4, 4, (-5) & M32); w(1, 4, 4, 7); w(2, 4, 4, (-9) & M32)
w(0, 8, 8, 1 << 40); w(1, 8, 8, (1 << 40) + 5); w(2, 8, 8, 3)
w(0, 16, 4, f32b(1.5)); w(1, 16, 4, f32b(2.25)); w(2, 16, 4, f32b(-0.75))
w(0, 24, 8, struct.unpack('<Q', struct.pack('<d', 1.0))[0]); w(1, 24, 8, struct.unpack('<Q', struct.pack('<d', 0.5))[0]); w(2, 24, 8, struct.unpack('<Q', struct.pack('<d', 0.25))[0])
for d, (a, b) in enumerate([(1.5, 2.0), (0.25, -1.0), (3.0, 0.5)]): w(d, 32, 4, f16b(a) | (f16b(b) << 16))     # f16x2
for d, c in enumerate([[0x38, 0x40, 0x30, 0x3C], [0x40, 0x38, 0x38, 0x44], [0x30, 0x30, 0x40, 0x38]]):
    for j, x in enumerate(c): w(d, 40 + j, 1, x)                                                        # e4m3 x4 (bytes 40..43)
for d in range(ND):
    for j in range(4): w(d, 48 + 4 * j, 4, f32b(d + 1 + j * 0.5))                                         # 4 x f32 at 48
init = [bytes(d) for d in dev]
body, exp = [], []
def emit(reg, val): body.append(f"    st.global.b32 [%o+{4 * len(exp)}], {reg};"); exp.append(val & M32)
sg = lambda x, b: x - (1 << b) if x >> (b - 1) else x
# integer ld_reduce
body.append("    multimem.ld_reduce.add.u32 %r0, [%mm+12];"); emit("%r0", sum(r(d, 12, 4) for d in range(ND)))
body.append("    multimem.ld_reduce.min.s32 %r0, [%mm+4];"); emit("%r0", min(sg(r(d, 4, 4), 32) for d in range(ND)))
body.append("    multimem.ld_reduce.max.u64 %q0, [%mm+8];  mov.b64 {%r0, %r1}, %q0;")
v = max(r(d, 8, 8) for d in range(ND)); emit("%r0", v & M32); emit("%r1", v >> 32)
body.append("    multimem.ld_reduce.relaxed.gpu.global.and.b32 %r0, [%mm+0];"); emit("%r0", r(0, 0, 4) & r(1, 0, 4) & r(2, 0, 4))
body.append("    multimem.ld_reduce.or.b64 %q0, [%mm+8];  mov.b64 {%r0, %r1}, %q0;")
v = r(0, 8, 8) | r(1, 8, 8) | r(2, 8, 8); emit("%r0", v & M32); emit("%r1", v >> 32)
body.append("    multimem.ld_reduce.acquire.sys.xor.b32 %r0, [%mm+0];"); emit("%r0", r(0, 0, 4) ^ r(1, 0, 4) ^ r(2, 0, 4))
# floating point
body.append("    multimem.ld_reduce.add.f32 %r0, [%mm+16];"); emit("%r0", f32b(f32v(r(0, 16, 4)) + f32v(r(1, 16, 4)) + f32v(r(2, 16, 4))))
body.append("    multimem.ld_reduce.add.f64 %q0, [%mm+24]; mov.b64 {%r0, %r1}, %q0;")
v = struct.unpack('<Q', struct.pack('<d', 1.0 + 0.5 + 0.25))[0]; emit("%r0", v & M32); emit("%r1", v >> 32)
# f16x2, default accumulation and acc::f32
def lane16(d, j): return f16v((r(d, 32, 4) >> (16 * j)) & 0xFFFF)
body.append("    multimem.ld_reduce.add.f16x2 %r0, [%mm+32];")
emit("%r0", f16b(sum(lane16(d, 0) for d in range(ND))) | (f16b(sum(lane16(d, 1) for d in range(ND))) << 16))
body.append("    multimem.ld_reduce.min.f16x2 %r0, [%mm+32];")
emit("%r0", f16b(min(lane16(d, 0) for d in range(ND))) | (f16b(min(lane16(d, 1) for d in range(ND))) << 16))
body.append("    multimem.ld_reduce.add.acc::f32.f16x2 %r0, [%mm+32];")
emit("%r0", f16b(sum(lane16(d, 0) for d in range(ND))) | (f16b(sum(lane16(d, 1) for d in range(ND))) << 16))
# vectors: v2.f16x2 reads 8 bytes
body.append("    multimem.ld_reduce.add.v2.f16x2 {%r0, %r1}, [%mm+32];")
def lane16o(d, off, j): return f16v((r(d, off, 4) >> (16 * j)) & 0xFFFF)
for off in (32, 36):
    emit("%r0" if off == 32 else "%r1", f16b(sum(lane16o(d, off, 0) for d in range(ND))) | (f16b(sum(lane16o(d, off, 1) for d in range(ND))) << 16))
# v4.f32
body.append("    multimem.ld_reduce.add.v4.f32 {%r0, %r1, %r2, %r3}, [%mm+48];")
for j in range(4): emit(f"%r{j}", f32b(sum(f32v(r(d, 48 + 4 * j, 4)) for d in range(ND))))
# e4m3 scalar lanes: x4 default accumulation (e4m3 each step) and acc::f16
body.append("    multimem.ld_reduce.add.e4m3x4 %r0, [%mm+40];")
def fp8red(vf, ef, op, lanes, acc16=False):
    out = 0
    for j in range(lanes):
        vals = [vf(r(d, 40 + j, 1)) for d in range(ND)]
        a = vals[0]
        for x in vals[1:]:
            a = op(a, x)
            if not acc16: a = vf(ef(a))        # default accumulation: round to the element type after every step
        out |= ef(a) << (8 * j)
    return out
emit("%r0", fp8red(*E4, lambda a, b: a + b, 4))
body.append("    multimem.ld_reduce.add.acc::f16.e4m3x4 %r0, [%mm+40];")
emit("%r0", fp8red(*E4, lambda a, b: a + b, 4, True))
body.append("    multimem.ld_reduce.max.e4m3x2 %r0, [%mm+40];")
emit("%r0", fp8red(*E4, max, 2))
# st / red
body.append("    mov.b32 %r4, 0x01020304; multimem.st.relaxed.gpu.global.b32 [%mm+56], %r4;")
for d in range(ND): w(d, 56, 4, 0x01020304)
body.append("    mov.b32 %r4, 5; multimem.red.relaxed.gpu.add.u32 [%mm+12], %r4;")
for d in range(ND): w(d, 12, 4, r(d, 12, 4) + 5)
body.append("    mov.s32 %r4, 4; multimem.red.max.s32 [%mm+4], %r4;")
for d in range(ND): w(d, 4, 4, max(sg(r(d, 4, 4), 32), 4))
body.append("    mov.b32 %r4, 0f40000000; mov.b32 %r5, 0f3F800000; mov.b32 %r6, 0f40400000; mov.b32 %r7, 0f40800000;")
body.append("    multimem.red.add.v4.f32 [%mm+48], {%r4, %r5, %r6, %r7};")
for d in range(ND):
    for j, x in enumerate([2.0, 1.0, 3.0, 4.0]): w(d, 48 + 4 * j, 4, f32b(f32v(r(d, 48 + 4 * j, 4)) + x))
body.append("    mov.b32 %r4, 0x00110022; mov.b32 %r5, 0x00330044; multimem.st.v2.b32 [%mm+40], {%r4, %r5};")
for d in range(ND): w(d, 40, 4, 0x00110022); w(d, 44, 4, 0x00330044)
body.append("    multimem.ld_reduce.max.u32 %r0, [%mm+56];"); emit("%r0", 0x01020304)
ptx = f'''// multimem.ld_reduce / st / red over three device buffers.  GENERATED by scripts/gen_mm.py
.version 8.6
.target sm_100a
.address_size 64

.visible .entry k(.param .u64 p_mm, .param .u64 p_o)
{{
    .reg .b64 %mm, %o, %q0; .reg .b32 %r0, %r1, %r2, %r3, %r4, %r5, %r6, %r7;
    ld.param.u64 %mm, [p_mm]; ld.param.u64 %o, [p_o];
''' + "\n".join(body) + """
    ret;
}
"""
for d in range(ND):
    ptx += f".buffer D{d} .u8 = {{ {', '.join(map(str, init[d]))} }};\n"
ptx += ".multimem MM .size 64 { D0, D1, D2 } ;\n"
ptx += f".buffer O .u32 [{len(exp)}];\n.launch k <<< 1, 1 >>> (MM, O);\n"
ptx += f".expect O .u32 = {{ {', '.join(map(str, exp))} }};\n"
for d in range(ND):
    ptx += f".expect D{d} .u8 = {{ {', '.join(map(str, dev[d]))} }};\n"
open('tests/multimem.ptx', 'w').write(ptx)
print(len(exp), "values")

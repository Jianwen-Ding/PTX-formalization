#!/usr/bin/env python3
"""Generate tests/surf.ptx: unformatted suld/sust/sured on 1d/2d/3d/a1d/a2d surfaces (b8..b64, vectors, clamp
modes, 64-bit reductions), formatted sust.p / sured.p for every sample format, and suq."""
import struct, math
from fractions import Fraction as Fr
f32b = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]
hexf = lambda v: "0f%08X" % f32b(v)
def rne(x):
    f = math.floor(x); d = x - f
    return f + (1 if d > Fr(1, 2) or (d == Fr(1, 2) and f % 2 == 1) else 0)
class Surf:
    def __init__(s, name, dims, tb, layers=1, fmt=11, ch=1, cdt=0, cord=0, layout=1, arr=None, init=None):
        s.name, s.dims, s.tb, s.layers, s.fmt, s.ch, s.cdt, s.cord, s.layout = name, dims, tb, layers, fmt, ch, cdt, cord, layout
        s.arr = (layers > 1) if arr is None else arr
        n = dims[0] * dims[1] * dims[2] * layers * tb
        s.mem = bytearray(init if init is not None else [(i * 37 + 11) & 255 for i in range(n)])
        s.init = bytes(s.mem)
    def off(s, x, y=0, z=0, l=0):
        return (((l * s.dims[2] + z) * s.dims[1] + y) * s.dims[0]) * s.tb + x
    def rd(s, o, n): return int.from_bytes(s.mem[o:o + n], 'little')
    def wr(s, o, n, v): s.mem[o:o + n] = (v & ((1 << 8 * n) - 1)).to_bytes(n, 'little')
    def bytes_total(s): return len(s.mem)
body, exp = [], []
def emit(reg, val): body.append(f"    st.global.b32 [%o+{4 * len(exp)}], {reg};"); exp.append(val & 0xFFFFFFFF)
surfs = []
def S(*a, **k):
    s = Surf(*a, **k); surfs.append(s); return s
# ---- unformatted -------------------------------------------------------------------------------
s2 = S("S2", (4, 3, 1), 4)
s3 = S("S3", (2, 2, 2), 4)
sa1 = S("SA1", (4, 1, 1), 4, layers=3)
sa2 = S("SA2", (2, 2, 1), 4, layers=2)
sb = S("SB", (4, 1, 1), 8)
# suld.b.2d.b8 at byte x=5,y=1
body.append("    mov.s32 %x, 5; mov.s32 %y, 1;")
body.append("    suld.b.2d.b8.trap %r0, [%S2, {%x, %y}];"); emit("%r0", s2.rd(s2.off(5, 1), 1))
body.append("    mov.s32 %x, 4; mov.s32 %y, 2;")
body.append("    suld.b.2d.v2.b16.trap {%r0, %r1}, [%S2, {%x, %y}];"); emit("%r0", s2.rd(s2.off(4, 2), 2)); emit("%r1", s2.rd(s2.off(6, 2), 2))
body.append("    mov.s32 %x, 8; mov.s32 %y, 0;")
body.append("    suld.b.2d.b64.cg.trap %d0, [%S2, {%x, %y}];  mov.b64 {%r0, %r1}, %d0;")
v = s2.rd(s2.off(8, 0), 8); emit("%r0", v & 0xFFFFFFFF); emit("%r1", v >> 32)
body.append("    mov.s32 %x, 100; mov.s32 %y, 1;")
body.append("    suld.b.2d.b32.clamp %r0, [%S2, {%x, %y}];"); emit("%r0", s2.rd(s2.off(12, 1), 4))
body.append("    mov.s32 %x, -3; mov.s32 %y, 100;")
body.append("    suld.b.2d.v4.b8.clamp {%r0, %r1, %r2, %r3}, [%S2, {%x, %y}];")
for i, r in enumerate(["%r0", "%r1", "%r2", "%r3"]): emit(r, s2.rd(s2.off(0 + i, 2), 1))
body.append("    mov.s32 %x, 4; mov.s32 %y, 1; mov.s32 %z, 1; mov.s32 %w, 0;")
body.append("    suld.b.3d.b32.trap %r0, [%S3, {%x, %y, %z, %w}];"); emit("%r0", s3.rd(s3.off(4, 1, 1), 4))
body.append("    mov.u32 %l, 2; mov.s32 %x, 4;")
body.append("    suld.b.a1d.v2.b32.trap {%r0, %r1}, [%SA1, {%l, %x}];"); emit("%r0", sa1.rd(sa1.off(4, 0, 0, 2), 4)); emit("%r1", sa1.rd(sa1.off(8, 0, 0, 2), 4))
body.append("    mov.u32 %l, 1; mov.s32 %x, 4; mov.s32 %y, 1; mov.s32 %z, 0;")
body.append("    suld.b.a2d.b32.zero %r0, [%SA2, {%l, %x, %y, %z}];"); emit("%r0", sa2.rd(sa2.off(4, 1, 0, 1), 4))
body.append("    mov.u32 %l, 1; mov.s32 %x, 40;")
body.append("    suld.b.a2d.b32.zero %r0, [%SA2, {%l, %x, %y, %z}];"); emit("%r0", 0)
body.append("    mov.s32 %x, 16;")
body.append("    suld.b.1d.v2.b64.trap {%d0, %d1}, [%SB, {%x}];  mov.b64 {%r0, %r1}, %d1;")
v = sb.rd(16 + 8, 8); emit("%r0", v & 0xFFFFFFFF); emit("%r1", v >> 32)
# sust.b: various, with cache ops and zero mode
body.append("    mov.s32 %x, 8; mov.s32 %y, 2; mov.b32 %v0, 0xAABBCCDD; mov.b32 %v1, 0x11223344;")
body.append("    sust.b.2d.v2.b32.wb.trap [%S2, {%x, %y}], {%v0, %v1};")
s2.wr(s2.off(8, 2), 4, 0xAABBCCDD); s2.wr(s2.off(12, 2) if False else s2.off(8, 2) + 4, 4, 0x11223344)
body.append("    mov.s32 %x, 1; mov.b32 %v0, 0x5A;")
body.append("    sust.b.2d.b8.cs.trap [%S2, {%x, %y}], %v0;"); s2.wr(s2.off(1, 2), 1, 0x5A)
body.append("    mov.s32 %x, 99;")
body.append("    sust.b.2d.b8.zero [%S2, {%x, %y}], %v0;")
body.append("    mov.s32 %x, 99;")
body.append("    sust.b.2d.b16.clamp [%S2, {%x, %y}], %v0;"); s2.wr(s2.off(14, 2), 2, 0x5A)
body.append("    mov.s32 %x, 0; mov.s32 %y, 0; mov.b64 %d0, 0x0123456789ABCDEF;")
body.append("    sust.b.3d.b64.trap [%S3, {%x, %y, %z, %w}], %d0;"); s3.wr(s3.off(0, 0, 0), 8, 0x0123456789ABCDEF)
# sured.b: 32/64-bit
def red(s, o, n, op, v, signed):
    cur = s.rd(o, n); bits = 8 * n
    sg = lambda x: x - (1 << bits) if x >> (bits - 1) else x
    a, b = (sg(cur), sg(v)) if signed else (cur, v)
    r = {"add": cur + v, "min": min(a, b), "max": max(a, b), "and": cur & v, "or": cur | v}[op]
    s.wr(o, n, r)
body.append("    mov.s32 %x, 0; mov.b64 %d0, 5;")
body.append("    sured.b.add.1d.u64.trap [%SB, {%x}], %d0;"); red(sb, 0, 8, "add", 5, False)
body.append("    mov.s32 %x, 8; mov.b64 %d0, -7;")
body.append("    sured.b.min.1d.s64.trap [%SB, {%x}], %d0;"); red(sb, 8, 8, "min", (1 << 64) - 7, True)
body.append("    mov.s32 %x, 16; mov.b64 %d0, 0xFFFFFFFFFFFFFFF0;")
body.append("    sured.b.max.1d.u64.trap [%SB, {%x}], %d0;"); red(sb, 16, 8, "max", 0xFFFFFFFFFFFFFFF0, False)
body.append("    mov.s32 %x, 4; mov.s32 %y, 0; mov.b32 %v0, 0xF0F0F0F0;")
body.append("    sured.b.and.2d.b32.trap [%S2, {%x, %y}], %v0;"); red(s2, s2.off(4, 0), 4, "and", 0xF0F0F0F0, False)
body.append("    mov.s32 %x, 100; sured.b.or.2d.b32.zero [%S2, {%x, %y}], %v0;")
body.append("    mov.s32 %x, 100; mov.s32 %v0, -9; sured.b.min.2d.s32.clamp [%S2, {%x, %y}], %v0;")
red(s2, s2.off(12, 0), 4, "min", (1 << 32) - 9, True)
# ---- formatted -----------------------------------------------------------------------------------
def hf(v): return struct.unpack('<I', struct.pack('<f', v))[0]
fU8 = S("FU8", (4, 1, 1), 4, fmt=0, ch=4, cdt=1, cord=2, init=[0] * 16)
fS8 = S("FS8", (4, 1, 1), 4, fmt=1, ch=4, init=[0] * 16)
fUI8 = S("FUI8", (4, 1, 1), 4, fmt=2, ch=4, init=[0] * 16)
fSI8 = S("FSI8", (4, 1, 1), 4, fmt=3, ch=4, init=[0] * 16)
fU16 = S("FU16", (4, 1, 1), 4, fmt=4, ch=2, init=[0] * 16)
fS16 = S("FS16", (4, 1, 1), 4, fmt=5, ch=2, init=[0] * 16)
fF16 = S("FF16", (4, 1, 1), 4, fmt=8, ch=2, init=[0] * 16)
fUI32 = S("FUI32", (4, 1, 1), 4, fmt=9, ch=1, init=[0] * 16)
fSI32 = S("FSI32", (4, 1, 1), 4, fmt=10, ch=1, init=[0] * 16)
fF32 = S("FF32", (4, 1, 1), 4, fmt=11, ch=1, init=[0] * 16)
fU8b = S("FU8B", (4, 1, 1), 4, fmt=0, ch=4, init=[0x77] * 16)
fU64 = S("FU64", (4, 1, 1), 8, fmt=9, ch=2, init=[0] * 32)
def conv(fmt, x):
    cb = 1 if fmt <= 3 else 2 if fmt <= 8 else 4
    if fmt in (0, 4): return rne(max(Fr(0), min(Fr(1), Fr(x))) * ((1 << 8 * cb) - 1))
    if fmt in (1, 5): return rne(max(Fr(-1), min(Fr(1), Fr(x))) * ((1 << 8 * cb - 1) - 1)) & ((1 << 8 * cb) - 1)
    if fmt in (2, 6, 9): return min(x, (1 << 8 * cb) - 1)
    if fmt in (3, 7, 10): return max(-(1 << 8 * cb - 1), min((1 << 8 * cb - 1) - 1, x)) & ((1 << 8 * cb) - 1)
    if fmt == 8: return struct.unpack('<H', struct.pack('<e', x))[0]
    return hf(x)
def sustp(s, vec, srcs, x, ints=False):
    regs = ", ".join(f"%v{i}" for i in range(len(srcs)))
    for i, v in enumerate(srcs):
        body.append(f"    mov.b32 %v{i}, {v if ints else hexf(v)};" if not ints else f"    mov.b32 %v{i}, {v & 0xFFFFFFFF};")
    body.append(f"    mov.s32 %x, {x};")
    vs = "" if vec == 1 else f".v{vec}"
    body.append(f"    sust.p.1d{vs}.b32.trap [%{s.name}, {{%x}}], {{{regs}}};" if vec > 1 else f"    sust.p.1d.b32.trap [%{s.name}, {{%x}}], %v0;")
    cb = 1 if s.fmt <= 3 else 2 if s.fmt <= 8 else 4
    for c in range(min(len(srcs), s.ch)):
        x_ = srcs[c]
        s.wr(x * s.tb + c * cb, cb, conv(s.fmt, x_))
sustp(fU8, 4, [0.5, 1.0, 0.0, 2.0], 1)
sustp(fS8, 4, [-1.0, 0.5, -0.5, 3.0], 2)
sustp(fUI8, 4, [300, 7, 0, 255], 1, ints=True)
sustp(fSI8, 4, [-300, 5, -5, 200], 3, ints=True)
sustp(fU16, 2, [0.25, 1.0], 0)
sustp(fS16, 2, [-0.5, 2.0], 1)
sustp(fF16, 2, [1.5, -2.0], 2)
sustp(fUI32, 1, [123456], 0, ints=True)
sustp(fSI32, 1, [-77], 1, ints=True)
sustp(fF32, 1, [3.25], 2)
sustp(fUI32, 1, [15], 2, ints=True)
sustp(fU8b, 2, [1.0, 0.0], 0)      # only the first two channels are written
# sured.p
def redp(s, x, op, v, n):
    signed = s.fmt in (3, 7, 10)
    red(s, x * n, n, op, v, signed)
body.append("    mov.s32 %x, 3; mov.b32 %v0, 10;")
body.append("    sured.p.add.1d.b32.trap [%FUI32, {%x}], %v0;"); redp(fUI32, 3, "add", 10, 4)
body.append("    mov.s32 %x, 1; mov.b32 %v0, 3;")
body.append("    sured.p.min.1d.b32.trap [%FSI32, {%x}], %v0;"); redp(fSI32, 1, "min", 3, 4)
body.append("    mov.s32 %x, 0; mov.b32 %v0, 3;")
body.append("    sured.p.min.1d.b32.trap [%FUI32, {%x}], %v0;"); redp(fUI32, 0, "min", 3, 4)
body.append("    mov.s32 %x, 1; mov.b64 %d0, 1000000000000;")
body.append("    sured.p.max.1d.b64.trap [%FU64, {%x}], %d0;"); redp(fU64, 1, "max", 1000000000000, 8)
body.append("    mov.s32 %x, 2; mov.b32 %v0, 6;")
body.append("    sured.p.and.2d.b32.trap [%FUI32, {%x, %y}], %v0;")
fUI32.wr(8, 4, fUI32.rd(8, 4) & 6)
# suq
body.append("    suq.width.b32 %r0, [%S3]; suq.height.b32 %r1, [%S3]; suq.depth.b32 %r2, [%S3];")
emit("%r0", 2); emit("%r1", 2); emit("%r2", 2)
body.append("    suq.array_size.b32 %r0, [%SA1]; suq.array_size.b32 %r1, [%S2]; suq.channel_data_type.b32 %r2, [%FU8]; suq.channel_order.b32 %r3, [%FU8]; suq.memory_layout.b32 %r4, [%S2];")
emit("%r0", 3); emit("%r1", 0); emit("%r2", 1); emit("%r3", 2); emit("%r4", 1)
# state checks of surfaces done via .expect; the kernel only exports query/load results above
names = [s.name for s in surfs]
ty = {4: ".u32", 8: ".u64"}
def attrs(s): return f"{s.layers}, {s.fmt}, {s.ch}, {s.cdt}, {s.cord}, {s.layout}, {int(s.arr)}"
ptx = f'''// Surfaces: unformatted suld/sust/sured (1d/2d/3d/a1d/a2d, b8..b64, vectors, trap/clamp/zero, cache ops, 64-bit
// reductions), formatted sust.p / sured.p on every sample format, suq.  GENERATED by scripts/gen_surf.py
.version 8.6
.target sm_90
.address_size 64

.visible .entry k({", ".join(f".param .u64 p_{n}" for n in names)}, .param .u64 p_o)
{{
    .reg .b64 {", ".join("%" + n for n in names)}, %o, %d0, %d1; .reg .b32 %r0, %r1, %r2, %r3, %r4, %v0, %v1, %v2, %v3; .reg .u32 %l; .reg .s32 %x, %y, %z, %w;
''' + "\n".join(f"    ld.param.u64 %{n}, [p_{n}];" for n in names) + "\n    ld.param.u64 %o, [p_o];\n    mov.s32 %y, 0; mov.s32 %z, 0; mov.s32 %w, 0;\n" + "\n".join(body) + "\n    ret;\n}\n"
for sf in surfs:
    ptx += f".buffer B{sf.name} .u8 = {{ {', '.join(map(str, sf.init))} }};\n"
    ptx += f".surface {sf.name} {ty[sf.tb]} B{sf.name} .dims {{ {sf.dims[0]}, {sf.dims[1]}, {sf.dims[2]} }} .attrs {{ {attrs(sf)} }} ;\n"
ptx += f".buffer O .u32 [{len(exp)}];\n.launch k <<< 1, 1 >>> ({', '.join(names)}, O);\n"
ptx += f".expect O .u32 = {{ {', '.join(map(str, exp))} }};\n"
for sf in surfs:
    ptx += f".expect B{sf.name} .u8 = {{ {', '.join(map(str, sf.mem))} }};\n"
open('tests/surf.ptx', 'w').write(ptx)
print(len(exp), "values", len(ptx), "bytes")

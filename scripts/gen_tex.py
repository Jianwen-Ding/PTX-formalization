#!/usr/bin/env python3
"""Generate tests/tex.ptx: textures (point/linear, address modes, read-normalized, tld4, txq) and surfaces."""
import struct, math
from fractions import Fraction
fx = lambda v: struct.unpack('<I', struct.pack('<f', v))[0]
f32 = lambda x: struct.unpack('<I', struct.pack('<f', float(x)))[0]
hexf = lambda v: "0f%08X" % fx(v)
W, H = 4, 3
data = [float(1 + i * 3 + (i % 2) * 0.5) for i in range(W * H)]
dbits = [fx(v) for v in data]
def T(i, j, mode):
    def ad(i, n):
        if mode == 'clamp': return min(max(i, 0), n - 1)
        if mode == 'wrap': return i % n
        if mode == 'border': return i if 0 <= i < n else None
    ii, jj = ad(i, W), ad(j, H)
    return Fraction(0) if ii is None or jj is None else Fraction(data[jj * W + ii])
def qw(a): return Fraction(math.floor(a * 256 + Fraction(1, 2)), 256)
def lin(x, y, mode):
    xb, yb = x - Fraction(1, 2), y - Fraction(1, 2)
    i0, j0 = math.floor(xb), math.floor(yb)
    ax, ay = qw(xb - i0), qw(yb - j0)
    r0 = (1 - ax) * T(i0, j0, mode) + ax * T(i0 + 1, j0, mode)
    r1 = (1 - ax) * T(i0, j0 + 1, mode) + ax * T(i0 + 1, j0 + 1, mode)
    return (1 - ay) * r0 + ay * r1
U = [(10, 200), (255, 0), (128, 64), (1, 2), (3, 4), (5, 6)]
ubytes = [x for p in U for x in p]
S0 = [100 + i for i in range(12)]
out = []
body = []
def tex4(kind, obj, coords, dt, ct, regs, extra=""):
    global body
    body.append(f"    tex.2d.v4.{dt}.{ct} {{{regs}}}, [{obj}, {{{coords}}}];")
def store(vals_regs):
    for r in vals_regs:
        i = len(out)
        body.append(f"    st.global.b32 [%o+{4 * i}], {r};")
        out.append(None)
exp = []
def emit(regs, vals):
    for r, v in zip(regs, vals):
        body.append(f"    st.global.b32 [%o+{4 * len(exp)}], {r};"); exp.append(v)
R4 = ["%fr0", "%fr1", "%fr2", "%fr3"]; U4 = ["%ur0", "%ur1", "%ur2", "%ur3"]
# A: integer coordinates, point, clamp
body.append("    mov.s32 %ix, 1; mov.s32 %iy, 2;")
body.append("    tex.2d.v4.f32.s32 {%fr0, %fr1, %fr2, %fr3}, [%tA, {%ix, %iy}];")
emit(R4, [dbits[2 * W + 1], 0, 0, fx(1.0)])
# B: normalized (1.25, 0.1), wrap, point
body.append(f"    mov.f32 %x, {hexf(1.25)}; mov.f32 %y, {hexf(0.1)};")
body.append("    tex.2d.v4.f32.f32 {%fr0, %fr1, %fr2, %fr3}, [%tB, {%x, %y}];")
emit(R4, [f32(T(math.floor(Fraction(5, 4) * W), math.floor(Fraction(fx(0.1) and struct.unpack('<f', struct.pack('<f', 0.1))[0]) * H), 'wrap')), 0, 0, fx(1.0)])
# C: linear, clamp, unnormalized (1.75, 0.6)
body.append(f"    mov.f32 %x, {hexf(1.75)}; mov.f32 %y, {hexf(0.6)};")
body.append("    tex.2d.v4.f32.f32 {%fr0, %fr1, %fr2, %fr3}, [%tC, {%x, %y}];")
yf = Fraction(struct.unpack('<f', struct.pack('<f', 0.6))[0])
emit(R4, [f32(lin(Fraction(7, 4), yf, 'clamp')), 0, 0, fx(1.0)])
# D: border mode, integer coordinates outside
body.append("    mov.s32 %ix, 7; mov.s32 %iy, 7;")
body.append("    tex.2d.v4.f32.s32 {%fr0, %fr1, %fr2, %fr3}, [%tD, {%ix, %iy}];")
emit(R4, [0, 0, 0, 0])
# E: u8x2, read-normalized float (texel (1,0))
body.append("    mov.s32 %ix, 1; mov.s32 %iy, 0;")
body.append("    tex.2d.v4.f32.s32 {%fr0, %fr1, %fr2, %fr3}, [%tE, {%ix, %iy}];")
emit(R4, [fx(1.0), fx(0.0), 0, fx(1.0)])
# F: u8x2, element-type read into u32 (texel (2,1))
body.append("    mov.s32 %ix, 2; mov.s32 %iy, 1;")
body.append("    tex.2d.v4.u32.s32 {%ur0, %ur1, %ur2, %ur3}, [%tU, {%ix, %iy}];")
emit(U4, [5, 6, 0, 1])
# G: tld4 red component at (1.5, 1.5)
body.append(f"    mov.f32 %x, {hexf(1.5)}; mov.f32 %y, {hexf(1.5)};")
body.append("    tld4.r.2d.v4.f32.f32 {%fr0, %fr1, %fr2, %fr3}, [%tA, {%x, %y}];")
emit(R4, [dbits[2 * W + 1], dbits[2 * W + 2], dbits[1 * W + 2], dbits[1 * W + 1]])
# txq
body.append("    txq.width.b32 %q0, [%tA]; txq.height.b32 %q1, [%tA]; txq.depth.b32 %q2, [%tA];")
emit(["%q0", "%q1", "%q2"], [W, H, 1])
# surfaces
body.append("    mov.s32 %ix, 8; mov.s32 %iy, 1;")
body.append("    suld.b.2d.v1.b32.trap {%s0}, [%sf, {%ix, %iy}];")
emit(["%s0"], [S0[1 * 4 + 2]])
# "the address given by the coordinate vector must be naturally aligned to a multiple of the access size" (9.7.12.1):
# an 8-byte .v2.b32 access at byte x = 8
body.append("    mov.s32 %ix, 8; mov.s32 %iy, 2;")
body.append("    suld.b.2d.v2.b32.trap {%s0, %s1}, [%sf, {%ix, %iy}];")
emit(["%s0", "%s1"], [S0[2 * 4 + 2], S0[2 * 4 + 3]])
body.append("    mov.s32 %ix, 100; mov.s32 %iy, 100;")
body.append("    suld.b.2d.v1.b32.zero {%s0}, [%sf, {%ix, %iy}];")
emit(["%s0"], [0])
body.append("    suld.b.2d.v1.b32.clamp {%s0}, [%sf, {%ix, %iy}];")
emit(["%s0"], [S0[11]])
body.append("    mov.s32 %ix, 0; mov.s32 %iy, 0; mov.b32 %v, 777;")
body.append("    sust.b.2d.v1.b32.trap [%sf, {%ix, %iy}], {%v};")
body.append("    mov.s32 %ix, 100; sust.b.2d.v1.b32.zero [%sf, {%ix, %iy}], {%v};")
body.append("    mov.s32 %ix, 4; mov.s32 %iy, 0; mov.b32 %v, 5;")
body.append("    sured.b.add.2d.u32.trap [%sf, {%ix, %iy}], %v;")
body.append("    suq.width.b32 %q0, [%sf]; suq.height.b32 %q1, [%sf];")
emit(["%q0", "%q1"], [4, 3])
S1 = list(S0); S1[0] = 777; S1[1] += 5
ptx = f'''// Textures (point / linear filtering, address modes, read-normalized integers, tld4, txq) and surfaces
// (suld / sust / sured / suq with .trap, .clamp, .zero).  GENERATED by scripts/gen_tex.py
.version 8.6
.target sm_90
.address_size 64

.visible .entry k(.param .u64 pA, .param .u64 pB, .param .u64 pC, .param .u64 pD, .param .u64 pE, .param .u64 pU, .param .u64 psf, .param .u64 po)
{{
    .reg .b64 %tA, %tB, %tC, %tD, %tE, %tU, %sf, %o; .reg .f32 %x, %y, %fr0, %fr1, %fr2, %fr3;
    .reg .u32 %ur0, %ur1, %ur2, %ur3; .reg .b32 %s0, %s1, %q0, %q1, %q2, %v; .reg .s32 %ix, %iy;
    ld.param.u64 %tA, [pA]; ld.param.u64 %tB, [pB]; ld.param.u64 %tC, [pC]; ld.param.u64 %tD, [pD];
    ld.param.u64 %tE, [pE]; ld.param.u64 %tU, [pU]; ld.param.u64 %sf, [psf]; ld.param.u64 %o, [po];
''' + "\n".join(body) + f'''
    ret;
}}
.buffer TEXBUF .u32 = {{ {", ".join(repr(v) if False else str(fx(v)) for v in data)} }};
.buffer U8BUF .u8 = {{ {", ".join(map(str, ubytes))} }};
.buffer SURF .u32 = {{ {", ".join(map(str, S0))} }};
.texture TA .f32 TEXBUF .dims {{ 4, 3, 1 }} .attrs {{ 1, 0, 1, 0, 0, 1 }} ;
.texture TB .f32 TEXBUF .dims {{ 4, 3, 1 }} .attrs {{ 1, 0, 0, 1, 0, 1 }} ;
.texture TC .f32 TEXBUF .dims {{ 4, 3, 1 }} .attrs {{ 1, 1, 1, 0, 0, 1 }} ;
.texture TD .f32 TEXBUF .dims {{ 4, 3, 1 }} .attrs {{ 1, 0, 2, 0, 0, 1 }} ;
.texture TE .u8 U8BUF .dims {{ 3, 2, 1 }} .attrs {{ 2, 0, 1, 0, 1, 1 }} ;
.texture TU .u8 U8BUF .dims {{ 3, 2, 1 }} .attrs {{ 2, 0, 1, 0, 0, 1 }} ;
.surface SF .b32 SURF .dims {{ 4, 3, 1 }} .attrs {{ 1 }} ;
.buffer O .u32 [{len(exp)}];
.launch k <<< 1, 1 >>> (TA, TB, TC, TD, TE, TU, SF, O);
.expect O .u32 = {{ {", ".join(map(str, exp))} }};
.expect SURF .u32 = {{ {", ".join(map(str, S1))} }};
'''
open('tests/tex.ptx', 'w').write(ptx)
print(len(exp), "values")

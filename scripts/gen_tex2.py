#!/usr/bin/env python3
"""Generate tests/tex-mip.ptx: mipmaps (tex.level/.grad/.base, mip filters, lod clamps), cubemaps (+arrays),
texel offsets, depth compare, residency predicate, tld4 on a2d/cube.  Independent Python model (exact Fractions)."""
import struct, math
from fractions import Fraction as Fr
f32b = lambda v: struct.unpack('<I', struct.pack('<f', float(v)))[0]
hexf = lambda v: "0f%08X" % f32b(v)
def rnd(x): return f32b(float(x))   # all model values are small dyadic rationals: exact in f32

# ---- textures --------------------------------------------------------------
class Tex:
    def __init__(s, w, h, layers, levels, filt, addr, norm, mipf, data=None):
        s.w, s.h, s.layers, s.levels, s.filt, s.addr, s.norm, s.mipf = w, h, layers, levels, filt, addr, norm, mipf
        s.lv = []   # per level: [layer][j][i]
        s.flat = []
        seed = 1
        for l in range(levels):
            lw, lh = max(1, w >> l), max(1, h >> l)
            L = []
            for ly in range(layers):
                P = []
                for j in range(lh):
                    row = []
                    for i in range(lw):
                        v = Fr(seed * 3 % 17 + l * 20 + ly * 100, 2); seed += 1
                        row.append(v); s.flat.append(v)
                    P.append(row)
                L.append(P)
            s.lv.append(L)
def adr(mode, i, n):
    if mode == 1: return min(max(i, 0), n - 1)
    if mode == 0: return i % n
    if mode == 2: return i if 0 <= i < n else None
    if mode == 3:
        m = i % (2 * n); return m if m < n else 2 * n - m - 1
def texel(t, lv, i, j, ly, am, ref=None):
    lw, lh = max(1, t.w >> lv), max(1, t.h >> lv)
    ii, jj = adr(am, i, lw), adr(am, j, lh)
    v = Fr(0) if ii is None or jj is None else t.lv[lv][ly][jj][ii]
    if ref is not None: v = Fr(1) if ref <= v else Fr(0)
    return v
def qw(a): return Fr(math.floor(a * 256 + Fr(1, 2)), 256)
def samp2(t, lv, x, y, ly, am, norm, filt, off=(0, 0), ref=None):
    lw, lh = max(1, t.w >> lv), max(1, t.h >> lv)
    px, py = (x * lw, y * lh) if norm else (x, y)
    if filt == 0:
        return texel(t, lv, math.floor(px) + off[0], math.floor(py) + off[1], ly, am, ref)
    xb, yb = px - Fr(1, 2), py - Fr(1, 2)
    i0, j0 = math.floor(xb), math.floor(yb)
    ax, ay = qw(xb - i0), qw(yb - j0)
    T = lambda i, j: texel(t, lv, i + off[0], j + off[1], ly, am, ref)
    r0 = (1 - ax) * T(i0, j0) + ax * T(i0 + 1, j0)
    r1 = (1 - ax) * T(i0, j0 + 1) + ax * T(i0 + 1, j0 + 1)
    return (1 - ay) * r0 + ay * r1
def lam_clamp(t, lam, mn, mx):
    return min(max(max(lam, mn), 0), min(mx, t.levels - 1))
def mipsamp(t, lam, mn, mx, mipf, x, y, ly, am, norm, filt, **kw):
    lc = lam_clamp(t, lam, mn, mx)
    if mipf == 1:
        l0 = math.floor(lc); l1 = min(l0 + 1, t.levels - 1); w = qw(lc - l0)
        a, b = samp2(t, l0, x, y, ly, am, norm, filt, **kw), samp2(t, l1, x, y, ly, am, norm, filt, **kw)
        return (1 - w) * a + w * b
    lv = min(math.floor(lc + Fr(1, 2)), t.levels - 1)
    return samp2(t, lv, x, y, ly, am, norm, filt, **kw)
def cube(x, y, z):
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az: f, sc, tc, ma = (0 if x >= 0 else 1), (-z if x >= 0 else z), -y, ax
    elif ay > ax and ay >= az: f, sc, tc, ma = (2 if y >= 0 else 3), x, (z if y >= 0 else -z), ay
    else: f, sc, tc, ma = (4 if z >= 0 else 5), (x if z >= 0 else -x), -y, az
    return f, (sc / ma + 1) / 2, (tc / ma + 1) / 2
def lam2(S):   # log2(sqrt(S)) for S a power of two
    return Fr(math.log2(float(S)) / 2).limit_denominator(1000) if S > 0 else Fr(-1000)

# ---- test program -----------------------------------------------------------
TM  = Tex(4, 4, 1, 3, 0, 1, 1, 0)   # point, clamp, normalized, point mip
TL  = Tex(4, 4, 1, 3, 1, 1, 1, 1)   # linear, clamp, normalized, linear mip
TD  = Tex(4, 4, 1, 1, 0, 1, 0, 0)   # depth-like, point, unnormalized
TDL = Tex(4, 4, 1, 1, 1, 1, 0, 0)   # linear (PCF)
TA  = Tex(4, 4, 3, 2, 0, 1, 1, 0)   # a2d, 3 layers, 2 levels
TC  = Tex(2, 2, 6, 1, 0, 1, 1, 0)   # cube
TCA = Tex(2, 2, 12, 1, 0, 1, 1, 0)  # cube array
TCL = Tex(2, 2, 6, 1, 1, 1, 1, 0)   # cube, linear
body, exp = [], []
def emit(reg, val): body.append(f"    st.global.b32 [%o+{4 * len(exp)}], {reg};"); exp.append(val)
def setf(r, v): body.append(f"    mov.f32 {r}, {hexf(v)};")
R4 = "{%fr0, %fr1, %fr2, %fr3}"
def fl(x): return float(x)
# 1: .base
setf("%x", 0.625); setf("%y", 0.375)
body.append(f"    tex.base.2d.v4.f32.f32 {R4}, [%tM, {{%x, %y}}];")
emit("%fr0", rnd(samp2(TM, 0, Fr(5, 8), Fr(3, 8), 0, 1, 1, 0))); emit("%fr3", f32b(1.0))
# 2-5: .level point mip, lod 0, 1, 1.4, 1.6, 2, 7 (clamped to levels-1)
for lod in (0.0, 1.0, 1.375, 1.625, 2.0, 7.0):
    setf("%l", lod)
    body.append(f"    tex.level.2d.v4.f32.f32 {R4}, [%tM, {{%x, %y}}], %l;")
    emit("%fr0", rnd(mipsamp(TM, Fr(lod), Fr(0), Fr(10**6), 0, Fr(5, 8), Fr(3, 8), 0, 1, 1, 0)))
# 6: linear mip + linear filter
for lod in (0.5, 1.25, 1.9375):
    setf("%l", lod)
    body.append(f"    tex.level.2d.v4.f32.f32 {R4}, [%tL, {{%x, %y}}], %l;")
    emit("%fr0", rnd(mipsamp(TL, Fr(lod), Fr(0), Fr(10**6), 1, Fr(5, 8), Fr(3, 8), 0, 1, 1, 1)))
# 7: grad (norm: scale by 4): (0.25,0.25) -> rho^2 = 2 -> lod 0.5 ; (0.5,0) -> lod 1 ; (1,0)/(0,2) -> lod 3 -> clamp
for (dx, dy) in (((0.25, 0.25), (0.0, 0.0)), ((0.5, 0.0), (0.0, 0.25)), ((0.125, 0.0), (0.0, 1.0))):
    setf("%dx0", dx[0]); setf("%dx1", dx[1]); setf("%dy0", dy[0]); setf("%dy1", dy[1])
    body.append(f"    tex.grad.2d.v4.f32.f32 {R4}, [%tL, {{%x, %y}}], {{%dx0, %dx1}}, {{%dy0, %dy1}};")
    S = max(sum((Fr(d) * 4) ** 2 for d in dx), sum((Fr(d) * 4) ** 2 for d in dy))
    emit("%fr0", rnd(mipsamp(TL, lam2(S), Fr(0), Fr(10**6), 1, Fr(5, 8), Fr(3, 8), 0, 1, 1, 1)))
# 8: sampler with minLOD = 1.0 (256) and maxLOD = 1.0 clamps lod 0 and 2 to 1; mip filter point
for lod in (0.0, 2.0):
    setf("%l", lod)
    body.append(f"    tex.level.2d.v4.f32.f32 {R4}, [%tM, %smp, {{%x, %y}}], %l;")
    emit("%fr0", rnd(mipsamp(TM, Fr(lod), Fr(1), Fr(1), 0, Fr(5, 8), Fr(3, 8), 0, 1, 1, 0)))
# 9: offsets (point, unnormalized texture TD) and depth compare
body.append("    mov.f32 %x, 0f3FC00000; mov.f32 %y, 0f40100000;")      # 1.5, 2.25
body.append(f"    tex.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], {{1, -1}};")
emit("%fr0", rnd(samp2(TD, 0, Fr(3, 2), Fr(9, 4), 0, 1, 0, 0, off=(1, -1))))
body.append(f"    tex.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], {{7, 7}};")
emit("%fr0", rnd(samp2(TD, 0, Fr(3, 2), Fr(9, 4), 0, 1, 0, 0, off=(7, 7))))
ref = Fr(TD.lv[0][0][2][1])            # compare against the texel itself: 1.0
setf("%ref", float(ref))
body.append(f"    tex.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], %ref;")
emit("%fr0", rnd(samp2(TD, 0, Fr(3, 2), Fr(9, 4), 0, 1, 0, 0, ref=ref)))
setf("%ref", float(ref) + 0.5)
body.append(f"    tex.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], %ref;")
emit("%fr0", rnd(samp2(TD, 0, Fr(3, 2), Fr(9, 4), 0, 1, 0, 0, ref=ref + Fr(1, 2))))
setf("%ref", float(ref) - 0.5)
body.append(f"    tex.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], %ref;")
emit("%fr0", rnd(samp2(TD, 0, Fr(3, 2), Fr(9, 4), 0, 1, 0, 0, ref=ref - Fr(1, 2))))
# PCF: linear filter + offset + reference
setf("%ref", 10.0)
body.append(f"    tex.2d.v4.f32.f32 {R4}, [%tDL, {{%x, %y}}], {{0, 1}}, %ref;")
emit("%fr0", rnd(samp2(TDL, 0, Fr(3, 2), Fr(9, 4), 0, 1, 0, 1, off=(0, 1), ref=Fr(10))))
# 10: a2d with mip level
setf("%x", 0.75); setf("%y", 0.25); setf("%l", 1.0)
body.append("    mov.u32 %ly, 2;")
body.append(f"    tex.level.a2d.v4.f32.f32 {R4}, [%tA, {{%ly, %x, %y}}], %l;")
emit("%fr0", rnd(mipsamp(TA, Fr(1), Fr(0), Fr(10**6), 0, Fr(3, 4), Fr(1, 4), 2, 1, 1, 0)))
body.append("    mov.u32 %ly, 9;")
body.append(f"    tex.base.a2d.v4.f32.f32 {R4}, [%tA, {{%ly, %x, %y}}];")
emit("%fr0", rnd(samp2(TA, 0, Fr(3, 4), Fr(1, 4), 2, 1, 1, 0)))     # layer clamped to 2
# 11: cubemaps
dirs = [(Fr(3, 2), Fr(1, 2), Fr(-1, 4)), (Fr(-2), Fr(1, 2), Fr(1, 2)), (Fr(1, 4), Fr(2), Fr(-1, 2)), (Fr(1, 4), Fr(-3, 2), Fr(1, 2)),
        (Fr(1, 2), Fr(-1, 4), Fr(2)), (Fr(-1, 2), Fr(1, 4), Fr(-3))]
for d in dirs:
    setf("%x", float(d[0])); setf("%y", float(d[1])); setf("%z", float(d[2]))
    body.append(f"    tex.cube.v4.f32.f32 {R4}, [%tC, {{%x, %y, %z, %z}}];")
    f, s, t = cube(*d)
    emit("%fr0", rnd(samp2(TC, 0, s, t, f, 1, 1, 0)))
d = dirs[3]; setf("%x", float(d[0])); setf("%y", float(d[1])); setf("%z", float(d[2]))
body.append("    mov.u32 %ly, 1;")
body.append(f"    tex.acube.v4.f32.f32 {R4}, [%tCA, {{%ly, %x, %y, %z}}];")
f, s, t = cube(*d); emit("%fr0", rnd(samp2(TCA, 0, s, t, 6 + f, 1, 1, 0)))
body.append(f"    tex.cube.v4.f32.f32 {R4}, [%tCL, {{%x, %y, %z, %z}}];")
emit("%fr0", rnd(samp2(TCL, 0, s, t, f, 1, 1, 1)))
# 12: residency predicate
body.append(f"    tex.2d.v4.f32.f32 {{%fr0, %fr1, %fr2, %fr3}}|%rp, [%tD, {{%x, %y}}];")
body.append("    selp.u32 %q, 5, 6, %rp;"); emit("%q", 5)
# 13: tld4 on a2d and cube (component r, counter-clockwise from lower-left), with offset
setf("%x", 1.5); setf("%y", 1.5)
body.append("    mov.u32 %ly, 1;")
body.append(f"    tld4.r.a2d.v4.f32.f32 {R4}, [%tA, {{%ly, %x, %y}}];")
# tld4 uses level 0, normalized per texture -> TA is normalized: coords 1.5 -> wraps via clamp
def gather(t, ly, x, y, am, norm, off=(0, 0)):
    px, py = (x * t.w, y * t.h) if norm else (x, y)
    i0, j0 = math.floor(px - Fr(1, 2)) + off[0], math.floor(py - Fr(1, 2)) + off[1]
    return [texel(t, 0, i0, j0 + 1, ly, am), texel(t, 0, i0 + 1, j0 + 1, ly, am), texel(t, 0, i0 + 1, j0, ly, am), texel(t, 0, i0, j0, ly, am)]
for r, v in zip(["%fr0", "%fr1", "%fr2", "%fr3"], gather(TA, 1, Fr(3, 2), Fr(3, 2), 1, 1)): emit(r, rnd(v))
d = dirs[0]; setf("%x", float(d[0])); setf("%y", float(d[1])); setf("%z", float(d[2]))
body.append(f"    tld4.r.cube.v4.f32.f32 {R4}, [%tC, {{%x, %y, %z, %z}}];")
f, s, t = cube(*d)
for r, v in zip(["%fr0", "%fr1", "%fr2", "%fr3"], gather(TC, f, s, t, 1, 1)): emit(r, rnd(v))
body.append(f"    tld4.r.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], {{1, 1}};")
setf("%x", 1.5); setf("%y", 1.5)
body[-1] = f"    tld4.r.2d.v4.f32.f32 {R4}, [%tD, {{%x, %y}}], {{1, 1}};"
body.insert(len(body) - 1, f"    mov.f32 %x, {hexf(1.5)}; mov.f32 %y, {hexf(1.5)};")
for r, v in zip(["%fr0", "%fr1", "%fr2", "%fr3"], gather(TD, 0, Fr(3, 2), Fr(3, 2), 1, 0, off=(1, 1))): emit(r, rnd(v))
# txq
body.append("    txq.num_mipmap_levels.b32 %q, [%tM];"); emit("%q", 3)
body.append("    txq.array_size.b32 %q, [%tCA];"); emit("%q", 2)

def buf(t): return ", ".join(str(rnd(v)) for v in t.flat)
def tx(name, buf_, t, flags=0):
    return (f".texture {name} .f32 {buf_} .dims {{ {t.w}, {t.h}, 1 }} .attrs {{ 1, {t.filt}, {t.addr}, {t.norm}, 0, {t.layers}, {t.levels}, {flags}, {t.mipf} }} ;")
params = ["tM", "tL", "tD", "tDL", "tA", "tC", "tCA", "tCL", "smp", "o"]
ptx = f'''// Mipmapped textures (tex.base/.level/.grad, point / linear mip filters, sampler lod clamps), cubemaps and cube
// arrays, texel offsets, depth compare (incl. PCF), residency predicate, tld4 on a2d / cube / offsets.
// GENERATED by scripts/gen_tex2.py
.version 8.6
.target sm_90
.address_size 64

.visible .entry k({", ".join(f".param .u64 p_{p}" for p in params)})
{{
    .reg .b64 {", ".join("%" + p for p in params)}; .reg .f32 %x, %y, %z, %l, %ref, %dx0, %dx1, %dy0, %dy1, %fr0, %fr1, %fr2, %fr3;
    .reg .u32 %ly, %q; .reg .pred %rp;
''' + "\n".join(f"    ld.param.u64 %{p}, [p_{p}];" for p in params) + "\n" + "\n".join(body) + f'''
    ret;
}}
.buffer BM .u32 = {{ {buf(TM)} }};
.buffer BL .u32 = {{ {buf(TL)} }};
.buffer BD .u32 = {{ {buf(TD)} }};
.buffer BDL .u32 = {{ {buf(TDL)} }};
.buffer BA .u32 = {{ {buf(TA)} }};
.buffer BC .u32 = {{ {buf(TC)} }};
.buffer BCA .u32 = {{ {buf(TCA)} }};
.buffer BCL .u32 = {{ {buf(TCL)} }};
{tx("XM", "BM", TM)}
{tx("XL", "BL", TL)}
{tx("XD", "BD", TD)}
{tx("XDL", "BDL", TDL)}
{tx("XA", "BA", TA)}
{tx("XC", "BC", TC, 2)}
{tx("XCA", "BCA", TCA, 2)}
{tx("XCL", "BCL", TCL, 2)}
.sampler SMP .attrs {{ 0, 1, 1, 0, 256, 256 }} ;
.buffer O .u32 [{len(exp)}];
.launch k <<< 1, 1 >>> (XM, XL, XD, XDL, XA, XC, XCA, XCL, SMP, O);
.expect O .u32 = {{ {", ".join(map(str, exp))} }};
'''
open('tests/tex-mip.ptx', 'w').write(ptx)
print(len(exp), "values", len(ptx), "bytes")

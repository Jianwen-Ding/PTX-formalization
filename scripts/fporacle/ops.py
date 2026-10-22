"""The instruction variants under test and their reference semantics (PTX ISA 9.4)."""
from fractions import Fraction

from formats import (F64, F32, F16, BF16, TF32, E4M3, E5M2, E2M3, E3M2, E2M1, UE5M3, UE8M0,
                     RN, RNA, RZ, RM, RP, tf32_to_f32bits, _ilog2)
from oracle import (Exp, NAN, flush, neg, ieee_compute, finish, cvt_ieee, cvt_mini, cvt_ue8m0,
                    cvt_f2i, int_val, minmax, x_add, x_mul, rnd)
import gen

RNDS = [RN, RZ, RM, RP]
IRNDS = ["rni", "rzi", "rmi", "rpi"]
CANON = {F32: 0x7FFFFFFF, F16: 0x7FFF, BF16: 0x7FFF, F64: 0x7FFFFFFFFFFFFFFF}


class IntFmt:
    name = "int"

    @staticmethod
    def is_nan(b):
        return False


INT = IntFmt()


class Variant:
    def __init__(self, name, tmpl, ow, expect):
        self.name, self.tmpl, self.ow, self.expect = name, tmpl, ow, expect


class Group:
    def __init__(self, name, target, in_widths, gen, variants, area, exhaustive=None):
        self.name, self.target, self.in_widths, self.gen = name, target, in_widths, gen
        self.variants, self.area, self.exhaustive = variants, area, exhaustive


def lane(x, i, w):
    return (x >> (i * w)) & ((1 << w) - 1)


def m(*mods):
    return "".join("." + x for x in mods if x)


def dv(fmt, b):
    return fmt.decode(b)


def quiet(fmt, b):
    return b | (1 << (fmt.mbits - 1))


def nan_soft_of(fmt, bits):
    """f64: 'NaN payloads are supported' (9.7.3): prefer one of the input NaNs, quietened."""
    if fmt is not F64:
        return None
    ns = tuple(quiet(fmt, b) for b in bits if fmt.is_nan(b))
    return ns or None


def arith(fmt, op, rnd_, bits, *, ftz=False, sat=False, relu=False, ftzfmt=None):
    vs = [dv(fmt, b) for b in bits]
    if ftz:
        vs = [flush(fmt, v) for v in vs]
    if op == "sub":
        vs[1] = neg(vs[1])
        op = "add"
    if op == "rcp":
        vs = [("fin", 0, Fraction(1)), vs[0]]
        op = "div"
    r = ieee_compute(op, fmt, rnd_, vs)
    return finish(fmt, r, ftz=ftz, sat=sat, relu=relu,
                  nan_soft=(CANON.get(fmt) if relu else nan_soft_of(fmt, bits)))


# ======================================================================================
# generators
def g_fmt(*fmts, related=True):
    def g(rng):
        if related and len(fmts) >= 2 and all(f is fmts[0] for f in fmts):
            return gen.operands(rng, list(fmts))
        return [gen.rand_bits(rng, f) for f in fmts]
    return g


def g_fma(fmt):
    def g(rng):
        a = gen.rand_bits(rng, fmt)
        b = gen.related(rng, fmt, a) if rng.random() < 0.3 else gen.rand_bits(rng, fmt)
        c = gen.rand_bits(rng, fmt)
        if rng.random() < 0.4:                       # c close to -a*b: cancellation / ties
            p = x_mul(dv(fmt, a), dv(fmt, b))
            if p[0] == "fin" and p[2] != 0:
                r = rnd(fmt, p, RN)
                if r[0] == "fin":
                    c = fmt.enc(1 - p[1], r[2])
                    c = (c + rng.randrange(-2, 3)) & ((1 << fmt.width) - 1)
        return [a, b, c]
    return g


def g_packed(fmt, nl, nops, w=None):
    w = w or fmt.width

    def g(rng):
        cols = [gen.operands(rng, [fmt] * nops) for _ in range(nl)]
        return [sum(cols[l][j] << (l * w) for l in range(nl)) for j in range(nops)]
    return g


def g_packed_fma(fmt, nl):
    gf = g_fma(fmt)

    def g(rng):
        cols = [gf(rng) for _ in range(nl)]
        return [sum(cols[l][j] << (l * fmt.width) for l in range(nl)) for j in range(3)]
    return g


# ======================================================================================
def groups():
    G = []

    # ---------------- 9.7.3 f32 / f64 ---------------------------------------------------
    for fmt, w, tag in ((F32, 32, "f32"), (F64, 64, "f64")):
        V = []
        for op in ("add", "sub", "mul", "div"):
            for r in RNDS:
                for ftz in ([False, True] if fmt is F32 else [False]):
                    for sat in ([False, True] if (fmt is F32 and op != "div") else [False]):
                        V.append(Variant("%s%s.%s" % (op, m(r, "ftz" if ftz else "", "sat" if sat else ""), tag),
                                         "%s%s.%s {d}, {0}, {1};" % (op, m(r, "ftz" if ftz else "", "sat" if sat else ""), tag), w,
                                         (lambda op, r, ftz, sat, fmt, w: lambda x: [(0, w, fmt, arith(fmt, op, r, x, ftz=ftz, sat=sat))])(op, r, ftz, sat, fmt, w)))
        G.append(Group("%s-binary" % tag, "sm_100a", [w, w], g_fmt(fmt, fmt), V, "%s add/sub/mul/div" % tag))
        V = []
        for r in RNDS:
            for ftz in ([False, True] if fmt is F32 else [False]):
                for sat in ([False, True] if fmt is F32 else [False]):
                    mm = m(r, "ftz" if ftz else "", "sat" if sat else "")
                    V.append(Variant("fma%s.%s" % (mm, tag), "fma%s.%s {d}, {0}, {1}, {2};" % (mm, tag), w,
                                     (lambda r, ftz, sat, fmt, w: lambda x: [(0, w, fmt, arith(fmt, "fma", r, x, ftz=ftz, sat=sat))])(r, ftz, sat, fmt, w)))
        for r in RNDS:
            mm = m(r)
            V.append(Variant("mad%s.%s" % (mm, tag), "mad%s.%s {d}, {0}, {1}, {2};" % (mm, tag), w,
                             (lambda r, fmt, w: lambda x: [(0, w, fmt, arith(fmt, "fma", r, x))])(r, fmt, w)))
        G.append(Group("%s-fma" % tag, "sm_100a", [w, w, w], g_fma(fmt), V, "%s fma/mad" % tag))
        V = []
        for op in ("rcp", "sqrt"):
            for r in RNDS:
                for ftz in ([False, True] if fmt is F32 else [False]):
                    mm = m(r, "ftz" if ftz else "")
                    V.append(Variant("%s%s.%s" % (op, mm, tag), "%s%s.%s {d}, {0};" % (op, mm, tag), w,
                                     (lambda op, r, ftz, fmt, w: lambda x: [(0, w, fmt, arith(fmt, op, r, x, ftz=ftz))])(op, r, ftz, fmt, w)))
        G.append(Group("%s-unary" % tag, "sm_100a", [w], g_fmt(fmt), V, "%s rcp/sqrt" % tag))

    # f32x2 (9.7.3.3-6, sm_100)
    V = []
    for op in ("add", "sub", "mul"):
        for r in RNDS:
            for ftz in (False, True):
                mm = m(r, "ftz" if ftz else "")
                V.append(Variant("%s%s.f32x2" % (op, mm), "%s%s.f32x2 {d}, {0}, {1};" % (op, mm), 64,
                                 (lambda op, r, ftz: lambda x: [(32 * i, 32, F32, arith(F32, op, r, [lane(x[0], i, 32), lane(x[1], i, 32)], ftz=ftz)) for i in range(2)])(op, r, ftz)))
    G.append(Group("f32x2-binary", "sm_100a", [64, 64], g_packed(F32, 2, 2), V, "f32x2 add/sub/mul"))
    V = []
    for r in RNDS:
        for ftz in (False, True):
            mm = m(r, "ftz" if ftz else "")
            V.append(Variant("fma%s.f32x2" % mm, "fma%s.f32x2 {d}, {0}, {1}, {2};" % mm, 64,
                             (lambda r, ftz: lambda x: [(32 * i, 32, F32, arith(F32, "fma", r, [lane(v, i, 32) for v in x], ftz=ftz)) for i in range(2)])(r, ftz)))
    G.append(Group("f32x2-fma", "sm_100a", [64, 64, 64], g_packed_fma(F32, 2), V, "f32x2 fma"))

    # min / max f32, f64 (9.7.3.11/12)
    V = []
    for op in ("min", "max"):
        for ftz in (False, True):
            for nan in (False, True):
                for xa in (False, True):
                    mm = m("ftz" if ftz else "", "NaN" if nan else "", "xorsign.abs" if xa else "")
                    V.append(Variant("%s%s.f32" % (op, mm), "%s%s.f32 {d}, {0}, {1};" % (op, mm), 32,
                                     (lambda op, ftz, nan, xa: lambda x: [(0, 32, F32, minmax(F32, op, x[:2], nan_mode=nan, abs_=xa, xorsign=xa, ftz=ftz, canon=CANON[F32]))])(op, ftz, nan, xa)))
                mm = m("ftz" if ftz else "", "NaN" if nan else "")
                for ab in (False, True):
                    mm2 = mm + m("abs" if ab else "")
                    V.append(Variant("%s%s.f32(3)" % (op, mm2), "%s%s.f32 {d}, {0}, {1}, {2};" % (op, mm2), 32,
                                     (lambda op, ftz, nan, ab: lambda x: [(0, 32, F32, minmax(F32, op, x, nan_mode=nan, abs_=ab, ftz=ftz, canon=CANON[F32]))])(op, ftz, nan, ab)))
    G.append(Group("f32-minmax", "sm_100a", [32, 32, 32], g_fmt(F32, F32, F32), V, "f32 min/max"))
    V = [Variant("%s.f64" % op, "%s.f64 {d}, {0}, {1};" % op, 64,
                 (lambda op: lambda x: [(0, 64, F64, minmax(F64, op, x))])(op)) for op in ("min", "max")]
    G.append(Group("f64-minmax", "sm_100a", [64, 64], g_fmt(F64, F64), V, "f64 min/max"))

    # ---------------- 9.7.4 half precision --------------------------------------------------
    for fmt, base in ((F16, "f16"), (BF16, "bf16")):
        for nl in (1, 2):
            t = base + ("x2" if nl == 2 else "")
            w = 16 * nl
            V = []
            for op in ("add", "sub", "mul"):
                for ftz in ([False, True] if fmt is F16 else [False]):
                    for sat in ([False, True] if fmt is F16 else [False]):
                        mm = m("rn", "ftz" if ftz else "", "sat" if sat else "")
                        V.append(Variant("%s%s.%s" % (op, mm, t), "%s%s.%s {d}, {0}, {1};" % (op, mm, t), w,
                                         (lambda op, ftz, sat, fmt, nl: lambda x: [(16 * i, 16, fmt, arith(fmt, op, RN, [lane(x[0], i, 16), lane(x[1], i, 16)], ftz=ftz, sat=sat)) for i in range(nl)])(op, ftz, sat, fmt, nl)))
            G.append(Group("%s-binary" % t, "sm_100a", [w, w], g_packed(fmt, nl, 2), V, "%s add/sub/mul" % t))
            V = []
            for ftz in ([False, True] if fmt is F16 else [False]):
                for sr in (["", "sat", "relu"] if fmt is F16 else ["", "relu"]):
                    mm = m("rn", "ftz" if ftz else "", sr)
                    V.append(Variant("fma%s.%s" % (mm, t), "fma%s.%s {d}, {0}, {1}, {2};" % (mm, t), w,
                                     (lambda ftz, sr, fmt, nl: lambda x: [(16 * i, 16, fmt, arith(fmt, "fma", RN, [lane(v, i, 16) for v in x], ftz=ftz, sat=sr == "sat", relu=sr == "relu")) for i in range(nl)])(ftz, sr, fmt, nl)))
            G.append(Group("%s-fma" % t, "sm_100a", [w, w, w], g_packed_fma(fmt, nl), V, "%s fma" % t))
            V = []
            for op in ("min", "max"):
                for ftz in ([False, True] if fmt is F16 else [False]):
                    for nan in (False, True):
                        for xa in (False, True):
                            mm = m("ftz" if ftz else "", "NaN" if nan else "", "xorsign.abs" if xa else "")
                            V.append(Variant("%s%s.%s" % (op, mm, t), "%s%s.%s {d}, {0}, {1};" % (op, mm, t), w,
                                             (lambda op, ftz, nan, xa, fmt, nl: lambda x: [(16 * i, 16, fmt, minmax(fmt, op, [lane(x[0], i, 16), lane(x[1], i, 16)], nan_mode=nan, abs_=xa, xorsign=xa, ftz=ftz, canon=CANON[fmt])) for i in range(nl)])(op, ftz, nan, xa, fmt, nl)))
            G.append(Group("%s-minmax" % t, "sm_100a", [w, w], g_packed(fmt, nl, 2), V, "%s min/max" % t))

    # ---------------- 9.7.5 mixed precision -------------------------------------------------
    for fmt, t in ((F16, "f16"), (BF16, "bf16")):
        V = []
        for op in ("add", "sub"):
            for r in RNDS:
                for sat in (False, True):
                    mm = m(r, "sat" if sat else "")
                    V.append(Variant("%s%s.f32.%s" % (op, mm, t), "%s%s.f32.%s {d}, {0}, {1};" % (op, mm, t), 32,
                                     (lambda op, r, sat, fmt: lambda x: [(0, 32, F32, mixed(op, r, [(fmt, x[0]), (F32, x[1])], sat=sat))])(op, r, sat, fmt)))
        G.append(Group("mixed-%s-f32" % t, "sm_100a", [16, 32], g_fmt(fmt, F32, related=False), V, "mixed add/sub .f32.%s" % t))
        V = []
        for r in RNDS:
            for sat in (False, True):
                mm = m(r, "sat" if sat else "")
                V.append(Variant("fma%s.f32.%s" % (mm, t), "fma%s.f32.%s {d}, {0}, {1}, {2};" % (mm, t), 32,
                                 (lambda r, sat, fmt: lambda x: [(0, 32, F32, mixed("fma", r, [(fmt, x[0]), (fmt, x[1]), (F32, x[2])], sat=sat))])(r, sat, fmt)))
        G.append(Group("mixed-fma-%s-f32" % t, "sm_100a", [16, 16, 32], g_mixed_fma(fmt), V, "mixed fma .f32.%s" % t))
        # 9.4 packed up-converting forms (sm_107f)
        V = []
        for op in ("add", "sub"):
            for r in RNDS:
                mm = m(r)
                V.append(Variant("%s%s.f32x2.%sx2.f32x2" % (op, mm, t), "%s%s.f32x2.%sx2.f32x2 {d}, {0}, {1};" % (op, mm, t), 64,
                                 (lambda op, r, fmt: lambda x: [(32 * i, 32, F32, mixed(op, r, [(fmt, lane(x[0], i, 16)), (F32, lane(x[1], i, 32))])) for i in range(2)])(op, r, fmt)))
        G.append(Group("mixed-%sx2-f32x2" % t, "sm_107a", [32, 64], g_mixed_pair(fmt), V, "mixed add/sub .f32x2.%sx2.f32x2" % t))
        V = []
        for r in RNDS:
            mm = m(r)
            V.append(Variant("fma%s.f32x2.%sx2.%sx2.f32x2" % (mm, t, t), "fma%s.f32x2.%sx2.%sx2.f32x2 {d}, {0}, {1}, {2};" % (mm, t, t), 64,
                             (lambda r, fmt: lambda x: [(32 * i, 32, F32, mixed("fma", r, [(fmt, lane(x[0], i, 16)), (fmt, lane(x[1], i, 16)), (F32, lane(x[2], i, 32))])) for i in range(2)])(r, fmt)))
        G.append(Group("mixed-fma-%sx2-f32x2" % t, "sm_107a", [32, 32, 64], g_mixed_fma_pair(fmt), V,
                       "mixed fma .f32x2.%sx2 (b read as %sx2)" % (t, t)))
    # down-converting forms
    V = []
    for op, mods, d in (("add", "rz.ftz", "f16x2"), ("sub", "rz.ftz", "f16x2"), ("add", "rz", "bf16x2"),
                        ("sub", "rz", "bf16x2"), ("mul", "ftz.rz", "f16x2"), ("mul", "rz", "bf16x2")):
        # 9.7.5.1/2/4 syntax: .ftz is not optional for the .f16x2 destination, absent for .bf16x2
        fmt = F16 if d == "f16x2" else BF16
        V.append(Variant("%s.%s.%s.f32x2.f32x2" % (op, mods, d), "%s.%s.%s.f32x2.f32x2 {d}, {0}, {1};" % (op, mods, d), 32,
                         (lambda op, ftz, fmt: lambda x: [(16 * i, 16, fmt, mixed_down(op, fmt, lane(x[0], i, 32), lane(x[1], i, 32), ftz)) for i in range(2)])(op, "ftz" in mods, fmt)))
    G.append(Group("mixed-down-f32x2", "sm_107a", [64, 64], g_down(), V, "mixed add/sub/mul .{f16x2,bf16x2}.f32x2.f32x2"))
    V = []
    for d, c in (("bf16x2", "f16x2"), ("f16x2", "bf16x2")):
        fd, fc = (BF16, F16) if d == "bf16x2" else (F16, BF16)
        V.append(Variant("mul.%s.%s.%s" % (d, d, c), "mul.%s.%s.%s {d}, {0}, {1};" % (d, d, c), 32,
                         (lambda fd, fc: lambda x: [(16 * i, 16, fd, mixed_mul_conv(fd, fc, lane(x[0], i, 16), lane(x[1], i, 16))) for i in range(2)])(fd, fc)))
    G.append(Group("mixed-mul-half", "sm_107a", [32, 32], g_halfmix(), V, "mixed mul .bf16x2.bf16x2.f16x2 / .f16x2.f16x2.bf16x2"))

    # ---------------- 9.7.6 alternate-format x4 arithmetic ----------------------------------
    G.extend(altfp_groups())

    # ---------------- 9.7.10.24 cvt --------------------------------------------------------
    G.extend(cvt_groups())
    return G


# ---- mixed precision helpers -----------------------------------------------------------------
def mixed(op, r, ins, sat=False):
    """Inputs are converted (exactly) to f32, then the f32 operation is applied."""
    vs = [dv(f, b) for f, b in ins]
    if op == "sub":
        vs[1] = neg(vs[1])
        op = "add"
    res = ieee_compute(op, F32, r, vs)
    return finish(F32, res, sat=sat)


def mixed_down(op, fmt, a, c, ftz):
    """add/sub/mul .rz{.ftz}.{f16x2,bf16x2}.f32x2.f32x2: exact f32 op, rounded once with .rz
    into the half format (two .rz roundings give the same result)."""
    va, vc = dv(F32, a), dv(F32, c)
    e_noflush = _down(op, fmt, va, vc, ftz)
    if ftz:
        e_flush = _down(op, fmt, flush(F32, va), flush(F32, vc), ftz)
        return merge(e_flush, e_noflush, "mixed .ftz on f32 inputs")
    return e_noflush


def _down(op, fmt, va, vc, ftz):
    if op == "sub":
        vc = neg(vc)
        op = "add"
    if op == "add":
        x = x_add(va, vc, RZ)
    else:
        x = x_mul(va, vc)
    return cvt_ieee(fmt, x, RZ, ftz=ftz)


def mixed_mul_conv(fd, fc, a, c):
    """mul.bf16x2.bf16x2.f16x2 (d = a * convert(c)): no rounding modifier given; the convert and
    the multiply both round to nearest even (alternative: one rounding of the exact product)."""
    vc = dv(fc, c)
    cc = cvt_ieee(fd, vc, RN)
    va = dv(fd, a)
    if cc.nan:
        two = arith(fd, "mul", RN, [a, fd.qnan()])
    else:
        two = arith(fd, "mul", RN, [a, cc.bits])
    one = finish(fd, ieee_compute("mul", fd, RN, [va, vc]))
    return merge(two, one, "mixed mul: convert-then-multiply vs one rounding")


def merge(e1, e2, tag):
    """Accept both readings: e1 is primary."""
    if e1.any or e2.any:
        return e1
    if e1.nan or e2.nan:
        if e1.nan and e2.nan:
            return e1
        return Exp(any_=True, tag=tag + " (NaN in one reading)")
    if e1.bits == e2.bits and set(e1.alts) == set(e2.alts):
        return e1
    alts = tuple(set(e1.alts) | {e2.bits} | set(e2.alts) - {e1.bits})
    return Exp(e1.bits, alts=alts, tag=tag if e2.bits != e1.bits else e1.tag)


def g_mixed_fma(fmt):
    def g(rng):
        return [gen.rand_bits(rng, fmt), gen.rand_bits(rng, fmt), gen.rand_bits(rng, F32)]
    return g


def g_mixed_pair(fmt):
    def g(rng):
        return [gen.rand_bits(rng, fmt) | (gen.rand_bits(rng, fmt) << 16),
                gen.rand_bits(rng, F32) | (gen.rand_bits(rng, F32) << 32)]
    return g


def g_mixed_fma_pair(fmt):
    def g(rng):
        return [gen.rand_bits(rng, fmt) | (gen.rand_bits(rng, fmt) << 16),
                gen.rand_bits(rng, fmt) | (gen.rand_bits(rng, fmt) << 16),
                gen.rand_bits(rng, F32) | (gen.rand_bits(rng, F32) << 32)]
    return g


def g_down():
    def one(rng):
        # f32 values around the f16 / bf16 range so that rounding, overflow and subnormals matter
        c = rng.random()
        if c < 0.5:
            return gen.rand_bits(rng, F32, emid=127 + rng.choice([0, -14, -24, 15, 16, -126, 127]), espan=3)
        return gen.rand_bits(rng, F32)
    def g(rng):
        a = [one(rng), one(rng)]
        b = [gen.related(rng, F32, a[0]) if rng.random() < 0.5 else one(rng),
             gen.related(rng, F32, a[1]) if rng.random() < 0.5 else one(rng)]
        return [a[0] | (a[1] << 32), b[0] | (b[1] << 32)]
    return g


def g_halfmix():
    def g(rng):
        return [gen.rand_bits(rng, F16) | (gen.rand_bits(rng, BF16) << 16),
                gen.rand_bits(rng, F16) | (gen.rand_bits(rng, BF16) << 16)] if rng.random() < 0.5 else \
               [gen.rand_bits(rng, BF16) | (gen.rand_bits(rng, F16) << 16),
                gen.rand_bits(rng, BF16) | (gen.rand_bits(rng, F16) << 16)]
    return g


# ---- 9.7.6 alternate-format x4 -----------------------------------------------------------------
ALT = {  # name: (format, lane width in register, lane stride, register width)
    "e5m2x4": (E5M2, 8, 8, 32), "e4m3x4": (E4M3, 8, 8, 32), "e3m2x4": (E3M2, 6, 8, 32),
    "e2m3x4": (E2M3, 6, 8, 32), "e2m1x4": (E2M1, 4, 4, 16), "e2m1p4x4": (E2M1, 4, 8, 32),
    "ue8m0x4": (UE8M0, 8, 8, 32),
}


def alt_lanes(t, x):
    f, w, st, _ = ALT[t]
    return [dv(f, (x >> (i * st)) & ((1 << w) - 1)) for i in range(4)]


def alt_gen_lane(rng, f):
    if f is UE8M0:
        c = rng.random()
        if c < 0.2:
            return rng.choice([0, 1, 0x7F, 0x80, 0x7E, 0xFE, 0xFF, 0x70, 0x8F])
        return 127 + rng.randrange(-20, 21) if c < 0.8 else rng.randrange(256)
    return gen.rand_bits(rng, f, p_special=0.3)


def alt_gen(rng, t):
    f, w, st, rw = ALT[t]
    return sum(alt_gen_lane(rng, f) << (i * st) for i in range(4))


def alt_one(op, ins, dt, sat):
    vs = list(ins)
    if op == "sub":
        vs[1] = neg(vs[1])
        op = "add"
    if op == "add":
        x = x_add(vs[0], vs[1], RN)
    elif op == "mul":
        x = x_mul(vs[0], vs[1])
    else:
        x = x_add(x_mul(vs[0], vs[1]), vs[2], RN)
    f = E5M2 if dt == "e5m2x4" else E4M3
    return cvt_mini(f, x, RN, satfinite=sat)


def altfp_groups():
    G = []
    types = list(ALT)
    for op in ("add", "sub", "mul", "fma"):
        for at in types:
            V = []
            atw = ALT[at][3]
            for dt in ("e5m2x4", "e4m3x4"):
                for sat in (False, True):
                    mm = m("rn", "satfinite" if sat else "")
                    if op in ("add", "sub"):
                        V.append(Variant("%s%s.%s.%s" % (op, mm, dt, at), "%s%s.%s.%s {d}, {0}, {2};" % (op, mm, dt, at), 32,
                                         (lambda dt, sat, at, op: lambda x: [(8 * i, 8, E5M2 if dt == "e5m2x4" else E4M3,
                                                                              alt_one(op, [alt_lanes(at, x[0])[i], alt_lanes(dt, x[2])[i]], dt, sat)) for i in range(4)])(dt, sat, at, op)))
                    else:
                        for bt in types:
                            if op == "mul":
                                tm = "%s%s.%s.%s.%s {d}, {0}, {1};" % (op, mm, dt, at, bt)
                            else:
                                tm = "%s%s.%s.%s.%s {d}, {0}, {1}, {2};" % (op, mm, dt, at, bt)
                            bsel = 1 if ALT[bt][3] == 32 else 3
                            V.append(Variant(tm.split(" ")[0], tm.replace("{1}", "{%d}" % bsel), 32,
                                             (lambda dt, sat, at, bt, op, bsel: lambda x: [(8 * i, 8, E5M2 if dt == "e5m2x4" else E4M3,
                                                                                            alt_one(op, [alt_lanes(at, x[0])[i], alt_lanes(bt, x[bsel])[i]] + ([alt_lanes(dt, x[2])[i]] if op == "fma" else []), dt, sat)) for i in range(4)])(dt, sat, at, bt, op, bsel)))

            def g(rng, at=at):
                # operand 0: a (.atype); 1: a 32-bit b, 3: a 16-bit b (e2m1x4); 2: c (e5m2 or e4m3 lanes)
                b32 = sum(alt_gen_lane(rng, rng.choice([E5M2, E4M3, E3M2, E2M3, UE8M0, E2M1])) << (8 * i) for i in range(4))
                # keep the padding of 6-/4-bit lanes zero only where the lane type says so: the
                # variant reads b with its own .btype, so draw a value valid for every type
                b32 = sum(((b32 >> (8 * i)) & (0x3F if rng.random() < 0.5 else 0xFF)) << (8 * i) for i in range(4))
                c = sum(alt_gen_lane(rng, rng.choice([E5M2, E4M3])) << (8 * i) for i in range(4))
                return [alt_gen(rng, at), b32, c, alt_gen(rng, "e2m1x4")]
            G.append(Group("alt-%s-%s" % (op, at), "sm_100a", [atw, 32, 32, 16], g, V, "x4 %s .%s" % (op, at)))
    return G


# ---- 9.7.10.24 cvt ------------------------------------------------------------------------
def int_fmt_bits(t):
    return t[0] == "s", int(t[1:])


INTS = ["s8", "s16", "s32", "s64", "u8", "u16", "u32", "u64"]
FT = {"f16": F16, "bf16": BF16, "f32": F32, "f64": F64}
FW = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


def regw(t):
    if t in FW:
        return FW[t]
    w = int(t[1:])
    return max(w, 16)


def f2i_exp(srct, dt, irnd, ftz):
    sg, w = int_fmt_bits(dt)
    sf = FT[srct]
    rw = regw(dt)

    def e(x):
        v = dv(sf, x[0])
        if ftz:
            v = flush(sf, v)
        r = cvt_f2i(v, irnd, sg, w, srct == "f64")
        b = r.bits
        if sg and w < rw and b >> (w - 1):       # sign-extended to the register (9.7.10.24 Notes)
            b |= ((1 << rw) - 1) ^ ((1 << w) - 1)
        return [(0, rw, INT, Exp(b))]
    return e


def i2f_exp(st, dt, r):
    sg, w = int_fmt_bits(st)
    df = FT[dt]

    def e(x):
        v = int_val(x[0], sg, w)
        return [(0, FW[dt], df, cvt_ieee(df, v, r))]
    return e


def f2f_exp(srct, dt, r, *, ftz=False, sat=False, relu=False, satfinite=False, pzo=False, irnd=None):
    sf, df = FT[srct], (TF32 if dt == "tf32" else FT[dt])

    def e(x):
        v = dv(sf, x[0])
        ftz_in = sf if (ftz and srct == "f32") else None
        ftz_out = ftz and dt == "f32"
        if irnd:
            if ftz_in:
                v = flush(sf, v)
            if v[0] == "fin":
                q = -v[2] if v[1] else v[2]
                from oracle import round_int
                k = round_int(q, irnd)
                v = ("fin", v[1], Fraction(abs(k)))      # integral value keeps the sign (-0.4 -> -0)
            res = finish(df, v, sat=sat)
            return [(0, FW[dt], df, res)]
        if dt == "tf32":
            ex = cvt_ieee(TF32, v, r, ftz_in=ftz_in, relu=relu, satfinite=satfinite, pzo=pzo,
                          nan_soft=CANON[F32] if relu else None,
                          enc=lambda vv: tf32_to_f32bits(TF32.enc(vv[1], "inf" if vv[0] == "inf" else vv[2])))
            return [(0, 32, F32, ex)]
        ex = cvt_ieee(df, v, r, ftz_in=ftz_in, ftz=ftz_out, sat=sat, relu=relu, satfinite=satfinite, pzo=pzo,
                      nan_soft=CANON.get(df) if relu else None)
        return [(0, FW[dt], df, ex)]
    return e


MINI = {"e4m3x2": E4M3, "e5m2x2": E5M2, "e2m3x2": E2M3, "e3m2x2": E3M2, "e2m1x2": E2M1, "ue5m3x2": UE5M3}


def to_mini(df, v, r, relu, satfinite, pzo, rbits=None, nrbits=0):
    if df is E5M2:
        if v[0] == "nan":
            return Exp(nan=True, note="canonical-nan") if relu else Exp(nan=True)
        return cvt_mini(E5M2, v, r, relu=relu, satfinite=satfinite, pzo=pzo)
    return cvt_mini(df, v, r, relu=relu, satfinite=satfinite, pzo=pzo)


def down_pair_exp(srct, dt, r, relu, satfinite, pzo):
    """cvt to an x2 format from (a, b) f32 or from one f16x2/bf16x2: a / upper half -> upper lane."""
    if dt in ("f16x2", "bf16x2"):
        df = F16 if dt == "f16x2" else BF16
        lw = 16
    elif dt == "ue8m0x2":
        df, lw = UE8M0, 8
    else:
        df = MINI[dt]
        lw = 4 if df is E2M1 else 8

    def conv(v):
        if df in (F16, BF16):
            return cvt_ieee(df, v, r, relu=relu, satfinite=satfinite, pzo=pzo, nan_soft=CANON[df] if relu else None)
        if df is UE8M0:
            return cvt_ue8m0(v, r, satfinite)
        return to_mini(df, v, r, relu, satfinite, pzo)

    def e(x):
        if srct == "f32":
            hi, lo = dv(F32, x[0]), dv(F32, x[1])
        else:
            sf = F16 if srct == "f16x2" else BF16
            hi, lo = dv(sf, lane(x[0], 1, 16)), dv(sf, lane(x[0], 0, 16))
        return [(lw, lw, df, conv(hi)), (0, lw, df, conv(lo))]
    return e


def rs_exp(dt, relu, satfinite, pzo):
    """cvt.rs{.relu}{.satfinite}{.pzo}.{f16x2,bf16x2}.f32 d, a, b, rbits (9.7.10.24):
    f16: 13 LSBs of the upper/lower 16 bits of rbits for a/b (13 = 23-10 discarded bits);
    bf16: the full upper/lower 16 bits (16 discarded bits)."""
    df = F16 if dt == "f16x2" else BF16
    nb = 13 if df is F16 else 16

    def e(x):
        out = []
        for (shift, v, rb) in ((16, dv(F32, x[0]), lane(x[2], 1, 16)), (0, dv(F32, x[1]), lane(x[2], 0, 16))):
            rb &= (1 << nb) - 1
            if v[0] == "fin" and v[2] != 0 and (df is F16 and v[2] < F16.min_norm):
                # more than 13 bits are discarded for a subnormal f16 result and the ISA does not say which 13 the
                # random bits are added to: the result must still be one of the two neighbours (toward / away from 0)
                ra = RM if v[1] else RP
                out.append((shift, 16, df, merge(cvt_ieee(df, v, RZ, relu=relu, satfinite=satfinite, pzo=pzo),
                                                  cvt_ieee(df, v, ra, relu=relu, satfinite=satfinite, pzo=pzo),
                                                  ".rs to a subnormal f16: toward or away from zero")))
                continue
            out.append((shift, 16, df, cvt_ieee(df, v, "rs", rbits=rb, nrbits=nb, relu=relu, satfinite=satfinite, pzo=pzo,
                                                 nan_soft=CANON[df] if relu else None)))
        return out
    return e


def up_pair_exp(st, dt, satfinite=False, relu=False, scale=None):
    """cvt.rn{.relu}{.satfinite}.{f16x2,bf16x2}.<x2 format>: exact up-conversion."""
    df = F16 if dt == "f16x2" else BF16
    if st == "ue8m0x2":
        sf, lw = UE8M0, 8
    else:
        sf = MINI[st]
        lw = 4 if sf is E2M1 else 8

    def e(x):
        out = []
        for i in (1, 0):
            b = lane(x[0], i, lw)
            if lw == 8 and sf in (E2M3, E3M2):
                b &= 0x3F
            v = dv(sf, b)
            out.append((16 * i, 16, df, cvt_ieee(df, v, RN, relu=relu, satfinite=satfinite, nan_soft=CANON[df] if relu else None)))
        return out
    return e


def g_ints(st):
    sg, w = int_fmt_bits(st)

    def g(rng):
        c = rng.random()
        lo = -(1 << (w - 1)) if sg else 0
        hi = (1 << (w - 1)) - 1 if sg else (1 << w) - 1
        if c < 0.25:
            x = rng.choice([0, 1, -1, lo, hi, lo + 1, hi - 1, 2, 3])
        elif c < 0.7:
            # around powers of two with low bits forming ties for 11/24/53-bit significands
            k = rng.randrange(0, w)
            x = (1 << k) + rng.randrange(-4, 5)
            p = rng.choice([8, 11, 24, 53])
            if k > p:
                sh = k - p
                x = (1 << k) | (rng.getrandbits(p) << sh) | rng.choice([0, 1 << (sh - 1), (1 << (sh - 1)) + 1, (1 << (sh - 1)) - 1 if sh > 1 else 0])
            if sg and rng.random() < 0.5:
                x = -x
        else:
            x = rng.randrange(lo, hi + 1)
        x = max(lo, min(hi, x))
        return [x & ((1 << max(w, 16)) - 1) if w >= 16 else (x & 0xFFFF)]
    return g


def g_f2i(sf):
    def g(rng):
        if rng.random() < 0.6:
            # values near integers, half-integers and integer type boundaries
            k = rng.choice([0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 3, 10, 23, 24, 52, 53])
            k = min(k, sf.emax)
            base = Fraction(2) ** k
            q = base + Fraction(rng.randrange(-8, 9), rng.choice([1, 2, 4, 8]))
            if q <= 0:
                q = Fraction(rng.randrange(1, 9), 8)
            from oracle import rnd as _r
            r = _r(sf, ("fin", 0, q), RN)
            if r[0] == "fin":
                return [sf.enc(rng.getrandbits(1), r[2])]
        return [gen.rand_bits(rng, sf)]
    return g


def g_x2bytes(fmts, lw=8):
    def g(rng):
        if lw == 4:
            return [rng.getrandbits(8)]
        return [alt_gen_lane(rng, rng.choice(fmts)) << 8 | alt_gen_lane(rng, rng.choice(fmts))]
    return g


def g_f32_for(fmts):
    """f32 inputs scaled to the range of one of the destination formats."""
    def one(rng):
        f = rng.choice(fmts)
        c = rng.random()
        if c < 0.55 and f is not UE8M0:
            # random value on a grid 3 bits finer than f, in f's range
            e = rng.randrange(f.emin - f.mbits - 1, f.emax + 2)
            q = Fraction(rng.randrange(1 << (f.mbits + 3), 1 << (f.mbits + 4)), 1 << (f.mbits + 3)) * Fraction(2) ** e
            if rng.random() < 0.3:
                q += Fraction(1, 1 << 40) * Fraction(2) ** e * rng.choice([1, -1])
            r = rnd(F32, ("fin", 0, q), RZ)
            if r[0] == "fin":
                return F32.enc(rng.getrandbits(1), r[2])
        if c < 0.7 and f is UE8M0:
            e = rng.randrange(-130, 130)
            q = Fraction(2) ** e * rng.choice([1, 1, Fraction(3, 2), Fraction(5, 4), Fraction(255, 256)])
            r = rnd(F32, ("fin", 0, q), RN)
            if r[0] == "fin":
                return F32.enc(0 if rng.random() < 0.85 else 1, r[2])
        if c < 0.8:
            mx = f.max_norm
            r = rnd(F32, ("fin", 0, mx * rng.choice([1, Fraction(17, 16), Fraction(31, 32), Fraction(33, 32), 2, Fraction(1, 2)])), RN)
            if r[0] == "fin":
                return F32.enc(rng.getrandbits(1), r[2])
        return gen.rand_bits(rng, F32)
    return one


def g1(one):
    return lambda rng: [one(rng)]


def cvt_groups():
    G = []
    T = "sm_107a"
    # ---- float -> float (scalar) ----
    V = []
    for r in RNDS:
        for ftz in (False, True):
            for sat in (False, True):
                mm = m(r, "ftz" if ftz else "", "sat" if sat else "")
                V.append(Variant("cvt%s.f16.f32" % mm, "cvt%s.f16.f32 {d}, {0};" % mm, 16, f2f_exp("f32", "f16", r, ftz=ftz, sat=sat)))
        V.append(Variant("cvt.%s.bf16.f32" % r, "cvt.%s.bf16.f32 {d}, {0};" % r, 16, f2f_exp("f32", "bf16", r)))
    for t in ("f16", "bf16"):
        for r in (RN, RZ):
            for relu in (False, True):
                for sf in (False, True):
                    for pzo in (False, True):
                        if not (relu or sf or pzo):
                            continue
                        mm = m(r, "relu" if relu else "", "satfinite" if sf else "", "pzo" if pzo else "")
                        V.append(Variant("cvt%s.%s.f32" % (mm, t), "cvt%s.%s.f32 {d}, {0};" % (mm, t), 16, f2f_exp("f32", t, r, relu=relu, satfinite=sf, pzo=pzo)))
    G.append(Group("cvt-f32-to-half", T, [32], g1(g_f32_for([F16, BF16, F32])), V, "cvt f32 -> f16/bf16"))
    V = []
    for ftz in (False, True):
        for sat in (False, True):
            mm = m("ftz" if ftz else "", "sat" if sat else "")
            V.append(Variant("cvt%s.f64.f32" % mm, "cvt%s.f64.f32 {d}, {0};" % mm, 64, f2f_up("f32", "f64", ftz, sat)))
            if ftz or sat:
                V.append(Variant("cvt%s.f32.f32" % mm, "cvt%s.f32.f32 {d}, {0};" % mm, 32, f2f_up("f32", "f32", ftz, sat)))
            for ir in IRNDS:
                mm2 = m(ir, "ftz" if ftz else "", "sat" if sat else "")
                V.append(Variant("cvt%s.f32.f32" % mm2, "cvt%s.f32.f32 {d}, {0};" % mm2, 32, f2f_exp("f32", "f32", None, ftz=ftz, sat=sat, irnd=ir)))
    V.append(Variant("cvt.rna.tf32.f32", "cvt.rna.tf32.f32 {d}, {0};", 32, f2f_exp("f32", "tf32", RNA)))
    V.append(Variant("cvt.rna.satfinite.tf32.f32", "cvt.rna.satfinite.tf32.f32 {d}, {0};", 32, f2f_exp("f32", "tf32", RNA, satfinite=True)))
    for r in (RN, RZ):
        for relu in (False, True):
            for sf in (False, True):
                for pzo in (False, True):
                    mm = m(r, "satfinite" if sf else "", "relu" if relu else "", "pzo" if pzo else "")
                    V.append(Variant("cvt%s.tf32.f32" % mm, "cvt%s.tf32.f32 {d}, {0};" % mm, 32, f2f_exp("f32", "tf32", r, relu=relu, satfinite=sf, pzo=pzo)))
    G.append(Group("cvt-f32-misc", T, [32], g1(g_f32_for([F32, F16])), V, "cvt f32 -> f32(int)/f64/tf32"))

    # f64 / f16 / bf16 sources
    V = []
    for r in RNDS:
        for ftz in (False, True):
            for sat in (False, True):
                mm = m(r, "ftz" if ftz else "", "sat" if sat else "")
                V.append(Variant("cvt%s.f32.f64" % mm, "cvt%s.f32.f64 {d}, {0};" % mm, 32, f2f_exp("f64", "f32", r, ftz=ftz, sat=sat)))
        for sat in (False, True):
            mm = m(r, "sat" if sat else "")
            V.append(Variant("cvt%s.f16.f64" % mm, "cvt%s.f16.f64 {d}, {0};" % mm, 16, f2f_exp("f64", "f16", r, sat=sat)))
        V.append(Variant("cvt.%s.bf16.f64" % r, "cvt.%s.bf16.f64 {d}, {0};" % r, 16, f2f_exp("f64", "bf16", r)))
    for ir in IRNDS:
        for sat in (False, True):
            mm = m(ir, "sat" if sat else "")
            V.append(Variant("cvt%s.f64.f64" % mm, "cvt%s.f64.f64 {d}, {0};" % mm, 64, f2f_exp("f64", "f64", None, sat=sat, irnd=ir)))
    G.append(Group("cvt-f64-src", T, [64], g_f64_for(), V, "cvt f64 -> f32/f16/bf16/f64(int)"))
    for st, sf in (("f16", F16), ("bf16", BF16)):
        V = []
        other = "bf16" if st == "f16" else "f16"
        for r in RNDS:
            if st == "bf16":
                for sat in (False, True):
                    mm = m(r, "sat" if sat else "")
                    V.append(Variant("cvt%s.f16.bf16" % mm, "cvt%s.f16.bf16 {d}, {0};" % mm, 16, f2f_exp("bf16", "f16", r, sat=sat)))
            else:
                V.append(Variant("cvt.%s.bf16.f16" % r, "cvt.%s.bf16.f16 {d}, {0};" % r, 16, f2f_exp("f16", "bf16", r)))
        for ftz in (False, True):
            for sat in ((False, True) if st == "f16" else (False,)):
                mm = m("ftz" if ftz else "", "sat" if sat else "")
                V.append(Variant("cvt%s.f32.%s" % (mm, st), "cvt%s.f32.%s {d}, {0};" % (mm, st), 32, f2f_up(st, "f32", ftz, sat)))
        V.append(Variant("cvt.f64.%s" % st, "cvt.f64.%s {d}, {0};" % st, 64, f2f_up(st, "f64", False, False)))
        for ir in IRNDS:
            mm = m(ir)
            V.append(Variant("cvt%s.%s.%s" % (mm, st, st), "cvt%s.%s.%s {d}, {0};" % (mm, st, st), 16, f2f_exp(st, st, None, irnd=ir)))
        G.append(Group("cvt-%s-src" % st, T, [16], g_fmt(sf), V, "cvt %s -> float" % st))

    # ---- float <-> int ----
    for st in ("f16", "bf16", "f32", "f64"):
        V = []
        for dt in INTS:
            for ir in IRNDS:
                for ftz in ((False, True) if st == "f32" else (False,)):
                    mm = m(ir, "ftz" if ftz else "")
                    V.append(Variant("cvt%s.%s.%s" % (mm, dt, st), "cvt%s.%s.%s {d}, {0};" % (mm, dt, st), regw(dt), f2i_exp(st, dt, ir, ftz)))
        G.append(Group("cvt-%s-to-int" % st, T, [FW[st]], g_f2i(FT[st]), V, "cvt %s -> int" % st))
    for st in INTS:
        V = []
        for dt in ("f16", "bf16", "f32", "f64"):
            for r in RNDS:
                V.append(Variant("cvt.%s.%s.%s" % (r, dt, st), "cvt.%s.%s.%s {d}, {0};" % (r, dt, st), FW[dt], i2f_exp(st, dt, r)))
        G.append(Group("cvt-%s-to-float" % st, T, [regw(st)], g_ints(st), V, "cvt int -> float"))

    # ---- packed narrow conversions ----
    downs = ["e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2"]
    for src in ("f32", "f16x2", "bf16x2"):
        V = []
        for dt in downs:
            for r in (RN, RZ):
                for relu in (False, True):
                    for pzo in (False, True):
                        mm = m(r, "satfinite", "relu" if relu else "", "pzo" if pzo else "")
                        ops_ = "{d}, {0}, {1};" if src == "f32" else "{d}, {0};"
                        V.append(Variant("cvt%s.%s.%s" % (mm, dt, src), "cvt%s.%s.%s %s" % (mm, dt, src, ops_),
                                         8 if dt == "e2m1x2" else 16, down_pair_exp(src, dt, r, relu, True, pzo)))
        for r in (RN, RZ, RP):
            for sf in (False, True):
                mm = m(r, "satfinite" if sf else "")
                ops_ = "{d}, {0}, {1};" if src == "f32" else "{d}, {0};"
                V.append(Variant("cvt%s.ue5m3x2.%s" % (mm, src), "cvt%s.ue5m3x2.%s %s" % (mm, src, ops_), 16, down_pair_exp(src, "ue5m3x2", r, False, sf, False)))
        if src in ("f32", "bf16x2"):
            for r in (RZ, RP):
                for sf in (False, True):
                    mm = m(r, "satfinite" if sf else "")
                    ops_ = "{d}, {0}, {1};" if src == "f32" else "{d}, {0};"
                    V.append(Variant("cvt%s.ue8m0x2.%s" % (mm, src), "cvt%s.ue8m0x2.%s %s" % (mm, src, ops_), 16, down_pair_exp(src, "ue8m0x2", r, False, sf, False)))
        if src == "f32":
            for dt in ("f16x2", "bf16x2"):
                for r in (RN, RZ):
                    for relu in (False, True):
                        for sf in (False, True):
                            for pzo in (False, True):
                                mm = m(r, "relu" if relu else "", "satfinite" if sf else "", "pzo" if pzo else "")
                                V.append(Variant("cvt%s.%s.f32" % (mm, dt), "cvt%s.%s.f32 {d}, {0}, {1};" % (mm, dt), 32, down_pair_exp("f32", dt, r, relu, sf, pzo)))
            gsrc = g_f32_for([E4M3, E5M2, E2M3, E3M2, E2M1, UE5M3, F16, BF16, UE8M0])
            g = (lambda gs: lambda rng: [gs(rng), gs(rng)])(gsrc)
            iw = [32, 32]
        else:
            hf = F16 if src == "f16x2" else BF16
            gsrc = g_f32_for([E4M3, E5M2, E2M3, E3M2, E2M1, UE5M3] + ([UE8M0] if hf is BF16 else []))

            def g(rng, gs=gsrc, hf=hf):
                out = 0
                for i in range(2):
                    if rng.random() < 0.75:
                        v = dv(F32, gs(rng))
                        rr = rnd(hf, v, RZ) if v[0] == "fin" else v
                        b = gen.rand_bits(rng, hf) if rr[0] != "fin" else hf.enc(rr[1], rr[2])
                    else:
                        b = gen.rand_bits(rng, hf)
                    out |= b << (16 * i)
                return [out]
            iw = [32]
        G.append(Group("cvt-narrow-from-%s" % src, T, iw, g, V, "cvt %s -> fp8/fp6/fp4/ue5m3/ue8m0" % src))

    # .scaled::n1::ue8m0 down-conversions and .scaled::n2::ue8m0 up-conversions (9.4, sm_107f)
    V = []
    for dt in downs + ["ue5m3x2"]:
        for r in (RN, RZ):
            for relu in ((False,) if dt == "ue5m3x2" else (False, True)):
                mm = m(r, "satfinite", "relu" if relu else "", "scaled::n1::ue8m0")
                V.append(Variant("cvt%s.%s.f32" % (mm, dt), "cvt%s.%s.f32 {d}, {0}, {1}, {2};" % (mm, dt),
                                 8 if dt == "e2m1x2" else 16, scaled_down_exp(dt, r, relu)))
    gs = g_f32_for([E4M3, E5M2, E2M3, E3M2, E2M1, UE5M3])
    G.append(Group("cvt-scaled-from-f32", T, [32, 32, 8], lambda rng: [gs(rng), gs(rng), g_scale(rng)], V,
                   "cvt.scaled::n1::ue8m0 f32 -> fp8/fp6/fp4/ue5m3"))
    for src, hf in (("f16x2", F16), ("bf16x2", BF16)):
        V = []
        for dt in downs + ["ue5m3x2"]:
            for r in (RN, RZ):
                for relu in ((False,) if dt == "ue5m3x2" else (False, True)):
                    mm = m(r, "satfinite", "relu" if relu else "", "scaled::n1::ue8m0")
                    V.append(Variant("cvt%s.%s.%s" % (mm, dt, src), "cvt%s.%s.%s {d}, {0}, {2};" % (mm, dt, src),
                                     8 if dt == "e2m1x2" else 16, scaled_down_exp(dt, r, relu, src)))

        def gh(rng, hf=hf):
            out = 0
            for i in range(2):
                v = dv(F32, gs(rng))
                rr = rnd(hf, v, RZ) if v[0] == "fin" else v
                b = gen.rand_bits(rng, hf) if (rr[0] != "fin" or rng.random() < 0.3) else hf.enc(rr[1], rr[2])
                out |= b << (16 * i)
            return [out, 0, g_scale(rng)]
        G.append(Group("cvt-scaled-from-%s" % src, T, [32, 32, 8], gh, V, "cvt.scaled::n1::ue8m0 %s -> fp8/fp6/fp4/ue5m3" % src))
    V = []
    for st in ("e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2", "ue5m3x2"):
        for relu in ((False,) if st == "ue5m3x2" else (False, True)):
            for sf in (False, True):
                mm = m("rn", "relu" if relu else "", "satfinite" if sf else "", "scaled::n2::ue8m0")
                V.append(Variant("cvt%s.bf16x2.%s" % (mm, st), "cvt%s.bf16x2.%s {d}, {%d}, {1};" % (mm, st, 2 if st == "e2m1x2" else 0), 32,
                                 scaled_up_exp(st, relu, sf)))

    def gup(rng):
        a = 0
        for i in range(2):
            f = rng.choice([E4M3, E5M2, E2M3, E3M2, UE5M3, E2M3, E3M2])
            a |= alt_gen_lane(rng, f) << (8 * i)
        return [a, g_scale(rng) | (g_scale(rng) << 8), rng.getrandbits(8)]
    G.append(Group("cvt-scaled-to-bf16x2", T, [16, 16, 8], gup, V, "cvt.scaled::n2::ue8m0 x2 -> bf16x2"))

    # .rs (stochastic) to f16x2 / bf16x2
    V = []
    for dt in ("f16x2", "bf16x2"):
        for relu in (False, True):
            for sf in (False, True):
                for pzo in (False, True):
                    mm = m("rs", "relu" if relu else "", "satfinite" if sf else "", "pzo" if pzo else "")
                    V.append(Variant("cvt%s.%s.f32" % (mm, dt), "cvt%s.%s.f32 {d}, {0}, {1}, {2};" % (mm, dt), 32, rs_exp(dt, relu, sf, pzo)))
    gs = g_f32_for([F16, BF16, F32])
    G.append(Group("cvt-rs-half", "sm_100a", [32, 32, 32], lambda rng: [gs(rng), gs(rng), rng.getrandbits(32)], V, "cvt.rs f32 -> f16x2/bf16x2"))

    # .rs x4 forms: the bit layout of rbits is given only in figures (not in the text), so the check is that each
    # result is one of the two neighbours of the input (rounded toward or away from zero); exact inputs are strict
    V = []
    for dt in ("e4m3x4", "e5m2x4", "e2m3x4", "e3m2x4", "e2m1x4"):
        for relu in (False, True):
            for pzo in (False, True):
                mm = m("rs", "relu" if relu else "", "satfinite", "pzo" if pzo else "")
                V.append(Variant("cvt%s.%s.f32" % (mm, dt), "cvt%s.%s.f32 {d}, {{{0}, {1}, {2}, {3}}}, {4};" % (mm, dt),
                                 16 if dt == "e2m1x4" else 32, rs_x4_exp(dt, relu, pzo)))
    gs4 = g_f32_for([E4M3, E5M2, E2M3, E3M2, E2M1])
    G.append(Group("cvt-rs-x4", "sm_100a", [32, 32, 32, 32, 32], lambda rng: [gs4(rng) for _ in range(4)] + [rng.getrandbits(32)], V,
                   "cvt.rs f32 -> e4m3x4/e5m2x4/e2m3x4/e3m2x4/e2m1x4 (neighbour check)"))

    # up-conversions from the x2 formats (exhaustive per lane)
    for sts, gname, ex in ((("e4m3x2", "e5m2x2"), "cvt-up-from-f8x2", lambda: [[(x << 8) | ((x * 37 + 11) & 0xFF)] for x in range(256)]),
                           (("e2m3x2", "e3m2x2"), "cvt-up-from-f6x2", lambda: [[(x << 8) | y] for x in range(64) for y in range(0, 64, 3)])):
        V = []
        for st in sts:
            V.extend(up_variants(st))
        G.append(Group(gname, T, [16], None, V, "cvt %s -> f16x2/bf16x2" % "/".join(sts), exhaustive=ex))
    V = []
    for st in ():
        for relu in (False, True):
            mm = m("rn", "relu" if relu else "")
            V.append(Variant("cvt%s.f16x2.%s" % (mm, st), "cvt%s.f16x2.%s {d}, {0};" % (mm, st), 32, up_pair_exp(st, "f16x2", relu=relu)))
            for sf in (False, True):
                mm = m("rn", "relu" if relu else "", "satfinite" if sf else "")
                V.append(Variant("cvt%s.bf16x2.%s" % (mm, st), "cvt%s.bf16x2.%s {d}, {0};" % (mm, st), 32, up_pair_exp(st, "bf16x2", relu=relu, satfinite=sf)))
    V.append(Variant("cvt.rn.bf16x2.ue8m0x2", "cvt.rn.bf16x2.ue8m0x2 {d}, {0};", 32, up_pair_exp("ue8m0x2", "bf16x2")))
    V.append(Variant("cvt.rn.f16x2.ue5m3x2", "cvt.rn.f16x2.ue5m3x2 {d}, {0};", 32, up_pair_exp("ue5m3x2", "f16x2")))
    for sf in (False, True):
        mm = m("rn", "satfinite" if sf else "")
        V.append(Variant("cvt%s.bf16x2.ue5m3x2" % mm, "cvt%s.bf16x2.ue5m3x2 {d}, {0};" % mm, 32, up_pair_exp("ue5m3x2", "bf16x2", satfinite=sf)))
    G.append(Group("cvt-up-from-ue-x2", T, [16], None, V, "cvt ue8m0x2/ue5m3x2 -> f16x2/bf16x2",
                   exhaustive=lambda: [[(x << 8) | ((x * 37 + 11) & 0xFF)] for x in range(256)]))
    V = []
    for relu in (False, True):
        mm = m("rn", "relu" if relu else "")
        V.append(Variant("cvt%s.f16x2.e2m1x2" % mm, "cvt%s.f16x2.e2m1x2 {d}, {0};" % mm, 32, up_pair_exp("e2m1x2", "f16x2", relu=relu)))
        for sf in (False, True):
            mm = m("rn", "relu" if relu else "", "satfinite" if sf else "")
            V.append(Variant("cvt%s.bf16x2.e2m1x2" % mm, "cvt%s.bf16x2.e2m1x2 {d}, {0};" % mm, 32, up_pair_exp("e2m1x2", "bf16x2", relu=relu, satfinite=sf)))
    G.append(Group("cvt-up-from-e2m1x2", T, [8], None, V, "cvt e2m1x2 -> f16x2/bf16x2",
                   exhaustive=lambda: [[x] for x in range(256)]))
    return G


def g_scale(rng):
    c = rng.random()
    if c < 0.7:
        return 127 + rng.randrange(-12, 13)
    if c < 0.9:
        return rng.randrange(0, 255)
    return rng.choice([0, 1, 254, 255, 127 - 126, 127 + 127])


def scaled_down_exp(dt, r, relu, src="f32"):
    """d = convert(x / 2^(sf - 127)); 9.7.10.24: with .scaled::n1::ue8m0 "subnormal values for
    inputs a and b are flushed to zero before conversion"; scale-factor[7:0] for both inputs."""
    df = MINI[dt]
    lw = 4 if df is E2M1 else 8

    def e(x):
        out = []
        if src == "f32":
            sf, sfmt, pairs = x[2] & 0xFF, F32, ((lw, x[0]), (0, x[1]))
        else:
            sfmt = F16 if src == "f16x2" else BF16
            sf, pairs = x[2] & 0xFF, ((lw, lane(x[0], 1, 16)), (0, lane(x[0], 0, 16)))
        for sh, b in pairs:
            v0 = dv(sfmt, b)
            if sf == 255:
                out.append((sh, lw, df, Exp(any_=True, tag="scale factor 0xFF (NaN)")))
                continue
            v = flush(sfmt, v0)
            if v[0] == "fin":
                v = ("fin", v[1], v[2] * Fraction(2) ** (127 - sf))
            e1 = to_mini(df, v, r, relu, True, False)
            if v0[0] == "fin" and 0 < v0[2] < sfmt.min_norm and v0[1]:
                # "flushed to zero" (not "sign-preserving zero" as elsewhere): accept +0 too
                e1 = merge(e1, to_mini(df, ("fin", 0, Fraction(0)), r, relu, True, False), "scaled::n1 flush of a negative subnormal: +0 or -0")
            out.append((sh, lw, df, e1))
        return out
    return e


def scaled_up_exp(st, relu, satfinite):
    """d = convert(a) * 2^(sf - 127) into bf16, rounded to nearest even."""
    sf_ = MINI[st]
    lw = 4 if sf_ is E2M1 else 8

    def e(x):
        a = x[2] if lw == 4 else x[0]
        out = []
        for i in (1, 0):
            b = lane(a, i, lw)
            if sf_ in (E2M3, E3M2) and b & 0xC0:
                out.append((16 * i, 16, BF16, Exp(any_=True, tag="6-bit input with nonzero padding (outside the ISA)")))
                continue
            s = lane(x[1], i, 8)
            if s == 255:
                out.append((16 * i, 16, BF16, Exp(any_=True, tag="scale factor 0xFF (NaN)")))
                continue
            v = dv(sf_, b)
            if v[0] == "fin":
                v = ("fin", v[1], v[2] * Fraction(2) ** (s - 127))
            out.append((16 * i, 16, BF16, cvt_ieee(BF16, v, RN, relu=relu, satfinite=satfinite, nan_soft=CANON[BF16] if relu else None)))
        return out
    return e


def rs_x4_exp(dt, relu, pzo):
    df = {"e4m3x4": E4M3, "e5m2x4": E5M2, "e2m3x4": E2M3, "e3m2x4": E3M2, "e2m1x4": E2M1}[dt]
    lw = 4 if df is E2M1 else 8

    def e(x):
        out = []
        for k in range(4):                       # a, b, e, f from the most significant lane down
            v = dv(F32, x[k])
            ra = RM if (v[0] != "nan" and v[1]) else RP
            e1 = merge(to_mini(df, v, RZ, relu, True, pzo), to_mini(df, v, ra, relu, True, pzo), ".rs x4: toward or away from zero")
            out.append(((3 - k) * lw, lw, df, e1))
        return out
    return e


def up_variants(st):
    V = []
    for relu in (False, True):
        mm = m("rn", "relu" if relu else "")
        V.append(Variant("cvt%s.f16x2.%s" % (mm, st), "cvt%s.f16x2.%s {d}, {0};" % (mm, st), 32, up_pair_exp(st, "f16x2", relu=relu)))
        for sf in (False, True):
            mm = m("rn", "relu" if relu else "", "satfinite" if sf else "")
            V.append(Variant("cvt%s.bf16x2.%s" % (mm, st), "cvt%s.bf16x2.%s {d}, {0};" % (mm, st), 32, up_pair_exp(st, "bf16x2", relu=relu, satfinite=sf)))
    return V


def f2f_up(st, dt, ftz, sat):
    """Exact up-conversion (or f32->f32 with .ftz/.sat): no rounding modifier."""
    sf, df = FT[st], FT[dt]

    def e(x):
        v = dv(sf, x[0])
        if ftz and st == "f32":
            v = flush(sf, v)
        ex = finish(df, v, ftz=ftz and dt == "f32", sat=sat)
        return [(0, FW[dt], df, ex)]
    return e


def g_f64_for():
    gs = g_f32_for([F32, F16, BF16])

    def g(rng):
        c = rng.random()
        if c < 0.5:
            v = dv(F32, gs(rng))
            if v[0] == "fin":
                q = v[2]
                if rng.random() < 0.6 and q != 0:
                    q = q * (1 + Fraction(rng.choice([1, -1, 3]), 1 << rng.choice([25, 26, 30, 40, 52])))
                rr = rnd(F64, ("fin", v[1], q), RN)
                return [F64.enc(rr[1], rr[2])]
        if c < 0.65:
            # around the f32 subnormal / overflow range
            e = rng.choice([-150, -149, -127, -126, -125, 127, 128, -25, -24, -15, -14, 15, 16])
            q = Fraction(2) ** e * Fraction(rng.randrange(1 << 20, 1 << 21), 1 << 20)
            rr = rnd(F64, ("fin", 0, q), RN)
            return [F64.enc(rng.getrandbits(1), rr[2])]
        return [gen.rand_bits(rng, F64)]
    return g

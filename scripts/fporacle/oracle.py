"""Reference semantics for the PTX floating-point instructions under test.

Two independent computations are made for every IEEE-format (f64/f32/f16/bf16) result rounded
with rn/rz/rm/rp:
  * exact rational arithmetic (fractions.Fraction, integer square root) + formats.round_mag,
  * MPFR through gmpy2 with the target precision, exponent range and subnormalisation.
They must agree bit for bit (an assertion, so an oracle bug cannot silently pass as a model bug).

Values are tuples: ('nan',) | ('inf', s) | ('fin', s, Fraction magnitude) (s = sign bit).

An expectation (Exp) for one result lane says which bit patterns are acceptable:
  bits     the expected pattern (PTX ISA text pins it down)
  nan      any NaN of the destination format is accepted (PTX: "unspecified NaN")
  alts     further acceptable patterns where the ISA text is ambiguous, with a tag naming the
           ambiguity (counted separately in the report)
  any      the ISA does not define the result (tag says why); the model's choice is recorded
  soft     for NaN results: a preferred pattern (f64 payload propagation, canonical NaN);
           a different NaN is reported as an observation, not a failure
"""
from fractions import Fraction
from math import isqrt

import gmpy2

from formats import (F64, F32, F16, BF16, TF32, E4M3, E5M2, E2M3, E3M2, E2M1, UE5M3, UE8M0,
                     RN, RNA, RZ, RM, RP, round_mag, overflow_result, _ilog2)

NAN = ("nan",)


class Exp:
    __slots__ = ("bits", "nan", "alts", "tag", "any", "soft", "note")

    def __init__(self, bits=None, nan=False, alts=(), tag=None, any_=False, soft=None, note=None):
        self.bits, self.nan, self.alts, self.tag = bits, nan, tuple(alts), tag
        self.any, self.soft, self.note = any_, soft, note

    def __repr__(self):
        if self.any:
            return "any[%s]" % self.tag
        if self.nan:
            return "NaN" + ("(pref %#x)" % self.soft if self.soft is not None else "")
        s = "%#x" % self.bits
        if self.alts:
            s += " or " + "/".join("%#x" % a for a in self.alts) + " [%s]" % self.tag
        return s


def check(fmt, e, got):
    """-> (status, tag) with status in ok / alt / fail / soft / any."""
    if e.any:
        return "any", e.tag
    if e.nan:
        if not fmt.is_nan(got):
            return "fail", None
        if e.soft is not None and got not in (e.soft if isinstance(e.soft, tuple) else (e.soft,)):
            return "soft", e.note or "nan-bits"
        return "ok", None
    if got == e.bits:
        return "ok", None
    if got in e.alts:
        return "alt", e.tag
    return "fail", None


# ---- value helpers ------------------------------------------------------------------------
def dec(fmt, b):
    return fmt.decode(b)


def flush(fmt, v):
    """ftz on an input: subnormal -> sign-preserving zero."""
    if v[0] == "fin" and 0 < v[2] < fmt.min_norm:
        return ("fin", v[1], Fraction(0))
    return v


def neg(v):
    if v[0] == "nan":
        return v
    return (v[0], 1 - v[1]) + v[2:]


def absv(v):
    if v[0] == "nan":
        return v
    return (v[0], 0) + v[2:]


def signed(v):
    """Comparable key (Fraction or +-inf float) of a non-NaN value."""
    if v[0] == "inf":
        return float("-inf") if v[1] else float("inf")
    return -v[2] if v[1] else v[2]


# ---- exact arithmetic ---------------------------------------------------------------------
def x_add(a, b, mode):
    if a[0] == "nan" or b[0] == "nan":
        return NAN
    if a[0] == "inf" and b[0] == "inf":
        return a if a[1] == b[1] else NAN
    if a[0] == "inf":
        return a
    if b[0] == "inf":
        return b
    q = (-a[2] if a[1] else a[2]) + (-b[2] if b[1] else b[2])
    if q == 0:
        s = a[1] if (a[1] == b[1]) else (1 if mode == RM else 0)
        return ("fin", s, Fraction(0))
    return ("fin", 1 if q < 0 else 0, abs(q))


def x_mul(a, b):
    if a[0] == "nan" or b[0] == "nan":
        return NAN
    s = a[1] ^ b[1]
    if a[0] == "inf" or b[0] == "inf":
        if (a[0] == "fin" and a[2] == 0) or (b[0] == "fin" and b[2] == 0):
            return NAN
        return ("inf", s)
    return ("fin", s, a[2] * b[2])


def x_div(a, b):
    if a[0] == "nan" or b[0] == "nan":
        return NAN
    s = a[1] ^ b[1]
    if a[0] == "inf":
        return NAN if b[0] == "inf" else ("inf", s)
    if b[0] == "inf":
        return ("fin", s, Fraction(0))
    if b[2] == 0:
        return NAN if a[2] == 0 else ("inf", s)
    return ("fin", s, a[2] / b[2])


def sqrt_sticky(q, bits=200):
    """A Fraction r with the same rounding as sqrt(q) on any grid coarser than 2^-bits relative:
    truncated square root plus a half unit of a much finer place when inexact."""
    if q == 0:
        return Fraction(0)
    e = _ilog2(q)
    k = bits - e // 2
    k += k & 1 or 0
    scaled = q * Fraction(4) ** k
    n = scaled.numerator // scaled.denominator
    r = isqrt(n)
    exact = (r * r == n) and scaled.denominator == 1 and n * scaled.denominator == scaled.numerator
    return (Fraction(r) + (0 if exact else Fraction(1, 2))) / Fraction(2) ** k


def x_sqrt(a):
    if a[0] == "nan":
        return NAN
    if a[0] == "inf":
        return NAN if a[1] else a
    if a[2] == 0:
        return a                                   # sqrt(+-0) = +-0
    if a[1]:
        return NAN
    return ("fin", 0, sqrt_sticky(a[2]))


# ---- rounding to a destination format -----------------------------------------------------
def rnd(fmt, v, mode, rbits=None, nrbits=0):
    """Round an exact value: returns ('nan',) | ('inf', s) | ('ovf', s) | ('fin', s, q)."""
    if v[0] != "fin":
        return v
    s, q = v[1], v[2]
    r = round_mag(fmt, s, q, mode, rbits, nrbits)
    if r > fmt.max_norm:
        return ("ovf", s)
    return ("fin", s, r)


def ieee_finish(fmt, r, mode):
    """IEEE 754 result of a rounded value (overflow per 7.4) -> value with no 'ovf'."""
    if r[0] == "ovf":
        m = overflow_result(fmt, r[1], mode if mode != "rs" else RZ)
        return ("inf", r[1]) if m == "inf" else ("fin", r[1], m)
    return r


# ---- MPFR cross-check ---------------------------------------------------------------------
_MODE = {RN: gmpy2.RoundToNearest, RZ: gmpy2.RoundToZero, RM: gmpy2.RoundDown, RP: gmpy2.RoundUp}


def _ctx(fmt, mode):
    # MPFR exponents: x = 0.1b... * 2^e, so emax = bias + 1, emin = 2 - bias - mbits
    return gmpy2.context(precision=fmt.mbits + 1, emin=2 - fmt.bias - fmt.mbits, emax=fmt.bias + 1,
                         subnormalize=True, round=_MODE[mode])


def to_mpfr(v):
    if v[0] == "nan":
        return gmpy2.mpfr("nan")
    if v[0] == "inf":
        return gmpy2.mpfr("-inf") if v[1] else gmpy2.mpfr("inf")
    n, d = v[2].numerator, v[2].denominator
    assert d & (d - 1) == 0, "inputs are dyadic rationals"
    if n == 0:
        return gmpy2.mpfr("-0") if v[1] else gmpy2.mpfr(0)
    sn = -n if v[1] else n
    x = gmpy2.mpfr(gmpy2.mpq(sn, d), max(n.bit_length(), 1) + 2)
    assert x.as_integer_ratio() == (sn, d), "inexact conversion to MPFR"
    return x


def from_mpfr(x):
    if gmpy2.is_nan(x):
        return NAN
    s = 1 if gmpy2.is_signed(x) else 0
    if gmpy2.is_infinite(x):
        return ("inf", s)
    n, d = x.as_integer_ratio()
    return ("fin", s, abs(Fraction(int(n), int(d))))


def mpfr_op(op, fmt, mode, ins):
    c = _ctx(fmt, mode)
    xs = [to_mpfr(v) for v in ins]
    if op == "add":
        r = c.add(xs[0], xs[1])
    elif op == "mul":
        r = c.mul(xs[0], xs[1])
    elif op == "fma":
        r = c.fma(xs[0], xs[1], xs[2])
    elif op == "div":
        r = c.div(xs[0], xs[1])
    elif op == "sqrt":
        r = c.sqrt(xs[0])
    elif op == "cvt":
        r = c.plus(xs[0])
    else:
        raise ValueError(op)
    return from_mpfr(r)


def same(a, b):
    if a[0] == "nan" or b[0] == "nan":
        return a[0] == b[0]
    return a == b


def ieee_compute(op, fmt, mode, ins):
    """Correctly rounded IEEE result (no PTX modifiers): rational path, cross-checked by MPFR."""
    if op == "add":
        x = x_add(ins[0], ins[1], mode)
    elif op == "mul":
        x = x_mul(ins[0], ins[1])
    elif op == "fma":
        p = x_mul(ins[0], ins[1])
        x = x_add(p, ins[2], mode)
    elif op == "div":
        x = x_div(ins[0], ins[1])
    elif op == "sqrt":
        x = x_sqrt(ins[0])
    elif op == "cvt":
        x = ins[0]
    else:
        raise ValueError(op)
    r = ieee_finish(fmt, rnd(fmt, x, mode), mode)
    if fmt in (F64, F32, F16, BF16) and mode in _MODE:
        m = mpfr_op(op, fmt, mode, ins)
        assert same(r, m), ("oracle disagreement", op, fmt, mode, ins, r, m)
    return r


# ---- encoding results with PTX modifiers --------------------------------------------------
def enc_val(fmt, v):
    if v[0] == "inf":
        return fmt.enc(v[1], "inf")
    return fmt.enc(v[1], v[2])


def finish(fmt, r, *, ftz=False, sat=False, relu=False, pzo=False, nan_soft=None, enc=None):
    """PTX post-processing of an IEEE-rounded value r for an IEEE-style destination
    (f64/f32/f16/bf16/tf32): .ftz (results), .sat, .relu, .pzo.  -> Exp"""
    enc = enc or (lambda v: enc_val(fmt, v))
    if r[0] == "nan":
        if sat:
            return Exp(enc(("fin", 0, Fraction(0))))
        if relu:
            return Exp(nan=True, soft=nan_soft, note="canonical-nan")
        return Exp(nan=True, soft=nan_soft)
    if ftz and r[0] == "fin" and 0 < r[2] < fmt.min_norm:
        r = ("fin", r[1], Fraction(0))
    alts, tag = [], None
    if sat:
        if r[1]:
            if r[0] == "fin" and r[2] == 0:
                # "clamps the result to [0.0, 1.0]": -0.0 is inside the range numerically
                alts, tag = [enc(r)], "sat(-0)"
            r = ("fin", 0, Fraction(0))
        elif r[0] == "inf" or r[2] > 1:
            r = ("fin", 0, Fraction(1))
    if relu and r[1]:
        if r[0] == "fin" and r[2] == 0:
            alts, tag = [enc(r)], "relu(-0)"
        r = ("fin", 0, Fraction(0))
    if pzo and r[1] and r[0] == "fin" and r[2] == 0:
        r = ("fin", 0, Fraction(0))
        alts, tag = [], None
    return Exp(enc(r), alts=alts, tag=tag)


def cvt_ieee(fmt, v, mode, *, ftz_in=None, ftz=False, sat=False, relu=False, satfinite=False,
             pzo=False, nan_soft=None, enc=None, rbits=None, nrbits=0):
    """Float-to-float conversion into an IEEE-style destination (f64 f32 f16 bf16 tf32 e5m2)."""
    if ftz_in is not None:
        v = flush(ftz_in, v)
    s = v[1] if v[0] != "nan" else 0
    if satfinite and (v[0] == "inf" or (v[0] == "fin" and v[2] > fmt.max_norm)):
        # 9.7.10.24 .satfinite: "If the absolute value of input (ignoring sign) is greater than
        # MAX_NORM of the specified destination format, then the result is sign-preserved MAX_NORM"
        r = ("fin", s, fmt.max_norm)
    else:
        r = rnd(fmt, v, mode, rbits, nrbits)
        if r[0] == "ovf":
            if mode == "rs":
                # the two neighbours: MAX_NORM (toward zero) or infinity (away from zero)
                e = finish(fmt, ("fin", r[1], fmt.max_norm), relu=relu, pzo=pzo, enc=enc)
                i = finish(fmt, ("inf", r[1]), relu=relu, pzo=pzo, enc=enc)
                return Exp(i.bits, alts=(e.bits,) if e.bits != i.bits else (), tag=".rs overflow without .satfinite: MAX_NORM or inf")
            r = ieee_finish(fmt, r, mode)
        if fmt in (F64, F32, F16, BF16) and mode in _MODE:
            m = mpfr_op("cvt", fmt, mode, [v])
            assert same(r, m), ("oracle disagreement", "cvt", fmt, mode, v, r, m)
    return finish(fmt, r, ftz=ftz, sat=sat, relu=relu, pzo=pzo, nan_soft=nan_soft, enc=enc)


def cvt_mini(fmt, v, mode, *, relu=False, satfinite=True, pzo=False, rbits=None, nrbits=0):
    """Conversion into E4M3 / E2M3 / E3M2 / E2M1 / UE5M3 (no infinity) and E5M2 non-IEEE use.
    PTX ISA 9.4 9.7.10.24 .satfinite / .relu rules, OCP MX v1.0 5.3 encodings."""
    if v[0] == "nan":
        if fmt.nan == "none":                       # FP6/FP4: NaN -> +MAX_NORM
            return Exp(fmt.max_bits(0))
        if relu:
            return Exp(nan=True, note="canonical-nan")
        return Exp(nan=True)
    s = v[1]
    unsigned = not fmt.signed
    # round first: a value above MAX_NORM that rounds to MAX_NORM is not an overflow (with
    # .satfinite both readings of "absolute value of input/output > MAX_NORM" agree)
    if v[0] == "inf":
        r = ("ovf", s)
    else:
        r = rnd(fmt, v, mode, rbits, nrbits)
    if r[0] == "ovf":
        if satfinite:
            if relu and s:
                return Exp(fmt.enc(0, Fraction(0)))
            # ue5m3 has no sign: "a positive MAX_NORM in .ue8m0x2 and .ue5m3x2"
            return Exp(fmt.max_bits(0 if unsigned else s))
        if fmt.has_inf:
            m = overflow_result(fmt, s, mode if mode != "rs" else RZ)
            return Exp(fmt.enc(s, m))
        if fmt is E4M3:
            # 9.7.6 x4 arithmetic: "In absence of .satfinite for .e4m3x4 destination format,
            # results beyond the largest representable ... are converted to NaN"
            return Exp(nan=True)
        return Exp(any_=True, tag="%s overflow without .satfinite" % fmt.name)
    q = r[2]
    if unsigned and s:
        if q == 0:
            return Exp(fmt.enc(0, Fraction(0)), tag=None)
        if relu:
            return Exp(fmt.enc(0, Fraction(0)))
        return Exp(any_=True, tag="negative input to unsigned %s" % fmt.name)
    alts, tag = [], None
    if relu and s:
        if q == 0:
            alts, tag = [fmt.enc(1, Fraction(0))], "relu(-0)"
        s, q = 0, Fraction(0)
    if pzo and s and q == 0:
        s = 0
        alts, tag = [], None
    return Exp(fmt.enc(s, q), alts=alts, tag=tag)


def cvt_ue8m0(v, mode, satfinite):
    """f32/bf16 -> ue8m0 with .rz or .rp.  OCP MX v1.0 5.4.1; PTX ISA 9.4 9.7.10.24."""
    if v[0] == "nan":
        return Exp(0xFF)
    s = v[1]
    if v[0] == "inf" or (v[0] == "fin" and v[2] > UE8M0.max_norm):
        if satfinite:
            return Exp(0xFE)                         # positive MAX_NORM
        return Exp(any_=True, tag="ue8m0 overflow without .satfinite")
    q = v[2]
    if q == 0:
        return Exp(any_=True, tag="ue8m0 of zero (no zero encoding)")
    if s:
        return Exp(any_=True, tag="ue8m0 of negative value")
    e = _ilog2(q)
    if mode == RP and q != Fraction(2) ** e:
        e += 1
    if e > 127:
        if satfinite:
            return Exp(0xFE)
        return Exp(any_=True, tag="ue8m0 overflow without .satfinite")
    if e < -127:
        return Exp(any_=True, tag="ue8m0 underflow below 2^-127")
    return Exp(e + 127)


# ---- float to integer ---------------------------------------------------------------------
def round_int(q_signed, irnd):
    """q_signed: Fraction. Integer rounding of PTX ISA 9.7.10.24 Integer Notes."""
    fl = q_signed.numerator // q_signed.denominator
    rem = q_signed - fl
    if rem == 0:
        return fl
    if irnd == "rzi":
        return fl if q_signed > 0 else fl + 1
    if irnd == "rmi":
        return fl
    if irnd == "rpi":
        return fl + 1
    if irnd == "rni":
        if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl & 1):
            return fl + 1
        return fl
    raise ValueError(irnd)


def cvt_f2i(v, irnd, signed_, width, src_is_f64):
    """-> integer bit pattern (width bits). Float-to-int saturates by default; NaN rules from
    the 9.7.10.24 Integer Notes."""
    lo = -(1 << (width - 1)) if signed_ else 0
    hi = (1 << (width - 1)) - 1 if signed_ else (1 << width) - 1
    if v[0] == "nan":
        if (not src_is_f64) and width != 64:
            return Exp(0)
        return Exp(1 << (width - 1))
    if v[0] == "inf":
        x = lo if v[1] else hi
    else:
        q = -v[2] if v[1] else v[2]
        x = max(lo, min(hi, round_int(q, irnd)))
    return Exp(x & ((1 << width) - 1))


def int_val(bits, signed_, width):
    x = bits & ((1 << width) - 1)
    if signed_ and x >> (width - 1):
        x -= 1 << width
    return ("fin", 1 if x < 0 else 0, Fraction(abs(x)))


# ---- min / max ----------------------------------------------------------------------------
def minmax(fmt, op, vals, *, nan_mode=False, abs_=False, xorsign=False, ftz=False, canon=None):
    """PTX ISA 9.4 9.7.3.11/12 and 9.7.4.7/8 semantics on raw bit patterns."""
    bits = list(vals)
    if ftz:
        bits = [(b & fmt.sign_bit) if fmt.is_sub(b) else b for b in bits]
    if xorsign:
        xs = ((bits[0] ^ bits[1]) & fmt.sign_bit)
    if abs_:
        bits = [b & ~fmt.sign_bit & ((1 << fmt.width) - 1) for b in bits]

    def two(x, y):
        nx, ny = fmt.is_nan(x), fmt.is_nan(y)
        if nan_mode and (nx or ny):
            return None
        if nx and ny:
            return None
        if nx:
            return y
        if ny:
            return x
        kx, ky = key(fmt, x), key(fmt, y)
        if op == "min":
            return x if kx < ky else y
        return x if kx > ky else y

    d = two(bits[0], bits[1])
    if len(bits) == 3 and d is not None:
        d = two(d, bits[2])
    elif len(bits) == 3 and d is None:
        # min_num(NaN, c) = c when not .NaN
        d = None if (nan_mode or fmt.is_nan(bits[2])) else bits[2]
    if d is None:
        if nan_mode:
            return Exp(nan=True, soft=canon, note="canonical-nan")
        return Exp(nan=True)
    if xorsign:
        d = (d & ~fmt.sign_bit) | xs
    return Exp(d)


def key(fmt, b):
    """Total order key for non-NaN values with -0 < +0."""
    d = fmt.decode(b)
    v = signed(d)
    return (v, 1 if (d[0] == "fin" and d[2] == 0 and d[1] == 0) else 0)

"""Seeded input generators biased towards floating-point edge cases."""
import random
from fractions import Fraction

from formats import Fmt, _ilog2


def specials(f):
    """Bit patterns of the edge values of format f."""
    out = []
    w = f.width
    sb = f.sign_bit
    for s in ([0, sb] if f.signed else [0]):
        out.append(s)                                         # zero
        out.append(s | 1)                                     # min subnormal
        out.append(s | ((1 << f.mbits) - 1))                  # max subnormal
        out.append(s | (1 << f.mbits))                        # min normal
        out.append(s | ((1 << f.mbits) + 1))                  # min normal + ulp
        out.append(s | f.max_bits(0))                         # max normal
        out.append(s | (f.max_bits(0) - 1))                   # max normal - ulp
        out.append(s | f.enc(0, Fraction(1)))                 # 1.0
        out.append(s | (f.enc(0, Fraction(1)) + 1))           # 1 + ulp
        out.append(s | (f.enc(0, Fraction(1)) - 1))           # 1 - ulp/2
        out.append(s | f.enc(0, Fraction(1, 2)))
        if f.has_inf:
            out.append(s | f.enc(0, "inf"))
        if f.nan == "ieee":
            top = ((1 << f.ebits) - 1) << f.mbits
            out.append(s | top | (1 << (f.mbits - 1)))        # quiet NaN
            out.append(s | top | 1)                           # signalling NaN
            out.append(s | top | ((1 << f.mbits) - 1))        # all-ones NaN
        elif f.nan == "allones":
            out.append(s | f.mag_mask)
    return [b & ((1 << w) - 1) for b in out]


def rand_bits(rng, f, p_special=0.25, emid=None, espan=None):
    """One edge-biased bit pattern of f."""
    r = rng.random()
    if r < p_special:
        return rng.choice(specials(f))
    s = rng.getrandbits(1) if f.signed else 0
    sb = f.sign_bit if s else 0
    if r < 0.35:                                              # fully random pattern
        return rng.getrandbits(f.width) & (((1 << f.width) - 1) if f.signed else 0xFF)
    if r < 0.45:                                              # subnormal
        return sb | rng.randrange(1, 1 << f.mbits)
    # normal number with an exponent near emid, mantissa with structured low bits
    top = f.emax + f.bias
    if emid is None:
        e = rng.choice([rng.randrange(1, top + 1), f.bias + rng.randrange(-3, 4),
                        rng.randrange(1, min(top, 1 + f.mbits + 3) + 1),
                        rng.randrange(max(1, top - f.mbits - 3), top + 1)])
    else:
        e = max(1, min(top, emid + rng.randrange(-espan, espan + 1)))
    m = rng.getrandbits(f.mbits)
    k = rng.random()
    if k < 0.3:                                               # few significant bits (ties likely)
        z = rng.randrange(0, f.mbits + 1)
        m = (m >> z) << z
    elif k < 0.4:
        m = (1 << f.mbits) - 1 - rng.randrange(0, 3)
    elif k < 0.5:
        m = rng.randrange(0, 3)
    b = sb | (e << f.mbits) | m
    if f.nan == "allones" and (b & f.mag_mask) == f.mag_mask and rng.random() < 0.7:
        b -= 1
    return b


def related(rng, f, a):
    """A second operand related to a (same/near exponent, ulp offsets, tie makers)."""
    d = f.decode(a)
    if d[0] != "fin" or d[2] == 0:
        return rand_bits(rng, f)
    q = d[2]
    e = _ilog2(q) if q >= f.min_norm else f.emin
    c = rng.random()
    s = rng.getrandbits(1) if f.signed else 0
    if c < 0.3:                                               # -a +- a few ulps (cancellation)
        bb = (a ^ f.sign_bit) if f.signed else a
        bb += rng.randrange(-3, 4)
        return bb & ((1 << f.width) - 1)
    if c < 0.65:                                              # half-ulp-ish value: ties for add
        sh = rng.choice([1, 1, 2, 3, f.mbits + 1, f.mbits + 2])
        t = Fraction(2) ** (e - f.mbits - sh) * rng.choice([1, 1, 3, 5, (1 << rng.randrange(1, 4)) + 1])
        if t < f.min_sub or t > f.max_norm:
            return rand_bits(rng, f)
        try:
            return f.enc(s, t)
        except AssertionError:
            return rand_bits(rng, f)
    # value with exponent near a's
    top = f.emax + f.bias
    ee = max(1, min(top, e + f.bias + rng.randrange(-f.mbits - 2, f.mbits + 3)))
    return rand_bits(rng, f, p_special=0.0, emid=ee, espan=0)


def operands(rng, fmts, kind=None):
    """A tuple of bit patterns, one per format in fmts."""
    a = rand_bits(rng, fmts[0])
    out = [a]
    for f in fmts[1:]:
        if f is fmts[0] and rng.random() < 0.5:
            out.append(related(rng, f, a))
        else:
            out.append(rand_bits(rng, f))
    return out

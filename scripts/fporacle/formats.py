"""Floating-point formats and exact (rational) rounding.

Independent of the Sail model: every value is decoded to a Python Fraction and rounded with
plain integer arithmetic.  The IEEE-style formats are also rounded through MPFR (gmpy2) in
oracle.py and the two roundings are cross-checked on every case.

Format parameters
  IEEE 754 binary64/32/16 and bfloat16 (8-bit exponent, 7-bit fraction).
  tf32: PTX ISA 9.4 5.2.3 leaves the layout implementation defined; it is modelled as an .f32
        bit pattern whose 13 low fraction bits are zero (10 fraction bits, f32 exponent range).
  OCP Microscaling Formats (MX) v1.0, section 5.3 (Table 2 "FP8", Table 3 "FP6 and FP4") and
  section 5.4.1 (E8M0):
     E4M3  bias 7,  no infinity, NaN S.1111.111, max normal 448
     E5M2  bias 15, IEEE-like (infinity S.11111.00, NaN S.11111.{01,10,11}), max normal 57344
     E2M3  bias 1,  no inf / NaN, max normal 7.5
     E3M2  bias 3,  no inf / NaN, max normal 28
     E2M1  bias 1,  no inf / NaN, max normal 6
     E8M0  bias 127, unsigned, no zero, no infinity, NaN 0xFF, values 2^-127 .. 2^127
  PTX ISA 9.4 5.2.3: ue5m3 is an unsigned 8-bit format, 5 exponent and 3 mantissa bits, no
  infinity, NaN 0xFF.  The bias is not stated; 15 (as for the 5-bit exponent of E5M2) is assumed.
  All the small formats have subnormals (OCP MX 5.3: "subnormals ... are supported").
"""
from fractions import Fraction

RN, RNA, RZ, RM, RP = "rn", "rna", "rz", "rm", "rp"


class Fmt:
    def __init__(self, name, ebits, mbits, bias, width, signed=True, inf=True, nan="ieee",
                 maxcode=None):
        self.name, self.ebits, self.mbits, self.bias = name, ebits, mbits, bias
        self.width, self.signed, self.has_inf, self.nan = width, signed, inf, nan
        self.emin = 1 - bias                       # exponent of the smallest normal
        top = (1 << ebits) - 1                     # largest exponent field
        if nan == "ieee":                          # all-ones exponent reserved
            self.emax, maxfrac = top - 1 - bias, (1 << mbits) - 1
        elif nan == "allones":                     # only S.1..1.1..1 is NaN (E4M3, ue5m3)
            self.emax, maxfrac = top - bias, (1 << mbits) - 2
        else:                                      # no NaN encoding (E2M3, E3M2, E2M1)
            self.emax, maxfrac = top - bias, (1 << mbits) - 1
        self.max_norm = Fraction((1 << mbits) + maxfrac, 1 << mbits) * Fraction(2) ** self.emax
        self.min_norm = Fraction(2) ** self.emin
        self.min_sub = Fraction(2) ** (self.emin - mbits)
        self.sign_bit = (1 << (width - 1)) if signed else 0
        self.mag_mask = (1 << (ebits + mbits)) - 1

    def __repr__(self):
        return self.name

    # ---- decoding -------------------------------------------------------------------------
    def decode(self, b):
        """-> ('nan',) | ('inf', s) | ('fin', s, Fraction magnitude)"""
        s = 1 if (self.signed and b & self.sign_bit) else 0
        e = (b >> self.mbits) & ((1 << self.ebits) - 1)
        f = b & ((1 << self.mbits) - 1)
        top = (1 << self.ebits) - 1
        if self.nan == "ieee" and e == top:
            return ("nan",) if f else ("inf", s)
        if self.nan == "allones" and e == top and f == (1 << self.mbits) - 1:
            return ("nan",)
        if e == 0:
            return ("fin", s, Fraction(f, 1 << self.mbits) * Fraction(2) ** self.emin)
        return ("fin", s, Fraction((1 << self.mbits) + f, 1 << self.mbits) * Fraction(2) ** (e - self.bias))

    def is_nan(self, b):
        return self.decode(b)[0] == "nan"

    def is_sub(self, b):
        d = self.decode(b)
        return d[0] == "fin" and 0 < d[2] < self.min_norm

    # ---- encoding of exactly representable magnitudes ------------------------------------
    def enc(self, s, mag):
        """mag: Fraction exactly representable (or 'inf')."""
        sb = self.sign_bit if (s and self.signed) else 0
        if mag == "inf":
            assert self.has_inf
            return sb | (((1 << self.ebits) - 1) << self.mbits)
        if mag == 0:
            return sb
        if mag < self.min_norm:
            q = mag / self.min_sub
            assert q.denominator == 1, (self, mag)
            return sb | int(q)
        e = _ilog2(mag)
        q = mag / Fraction(2) ** (e - self.mbits)
        assert q.denominator == 1 and (1 << self.mbits) <= q < (2 << self.mbits), (self, mag)
        assert e <= self.emax and mag <= self.max_norm, (self, mag)
        return sb | ((e + self.bias) << self.mbits) | (int(q) - (1 << self.mbits))

    def qnan(self):
        """A representative NaN (only used as a placeholder; NaNs are compared by class)."""
        if self.nan == "ieee":
            return (((1 << self.ebits) - 1) << self.mbits) | (1 << (self.mbits - 1)) | ((1 << (self.mbits - 1)) - 1)
        if self.nan == "allones":
            return self.mag_mask
        return None

    def max_bits(self, s):
        return self.enc(s, self.max_norm)


def _ilog2(q):
    """floor(log2(q)) for a positive Fraction."""
    n, d = q.numerator, q.denominator
    e = n.bit_length() - d.bit_length()
    if e >= 0:
        if n < (d << e):
            e -= 1
    else:
        if (n << -e) < d:
            e -= 1
    return e


def round_mag(fmt, s, q, mode, rbits=None, nrbits=0):
    """Round the exact magnitude q >= 0 (sign s) to fmt's grid with an unbounded exponent above.
    Returns a Fraction (possibly > max_norm: the caller decides overflow).
    mode: rn rna rz rm rp, or 'rs' with rbits/nrbits (stochastic: add rbits to the top nrbits
    discarded bits, round away from zero on carry out)."""
    if q == 0:
        return Fraction(0)
    e = max(_ilog2(q), fmt.emin)
    ulp = Fraction(2) ** (e - fmt.mbits)
    t = q / ulp
    k, rem = divmod(t.numerator, t.denominator)
    rem = Fraction(rem, t.denominator)            # fraction of an ulp discarded, in [0, 1)
    up = False
    if rem:
        if mode == RN:
            up = rem > Fraction(1, 2) or (rem == Fraction(1, 2) and k & 1)
        elif mode == RNA:
            up = rem >= Fraction(1, 2)
        elif mode == RZ:
            up = False
        elif mode == RP:
            up = not s
        elif mode == RM:
            up = bool(s)
        elif mode == "rs":
            # discarded bits truncated to nrbits bits, plus the random bits; carry out -> away.
            disc = int(rem * (1 << nrbits))
            up = disc + rbits >= (1 << nrbits)
        else:
            raise ValueError(mode)
    return (k + (1 if up else 0)) * ulp


def overflow_result(fmt, s, mode):
    """IEEE 754 7.4 overflow result for a format with infinities."""
    if mode in (RN, RNA):
        return "inf"
    if mode == RZ:
        return fmt.max_norm
    if mode == RP:
        return fmt.max_norm if s else "inf"
    if mode == RM:
        return "inf" if s else fmt.max_norm
    raise ValueError(mode)


F64 = Fmt("f64", 11, 52, 1023, 64)
F32 = Fmt("f32", 8, 23, 127, 32)
F16 = Fmt("f16", 5, 10, 15, 16)
BF16 = Fmt("bf16", 8, 7, 127, 16)
TF32 = Fmt("tf32", 8, 10, 127, 19)            # encoded into f32 bits by tf32_to_f32bits
E4M3 = Fmt("e4m3", 4, 3, 7, 8, inf=False, nan="allones")
E5M2 = Fmt("e5m2", 5, 2, 15, 8)
E2M3 = Fmt("e2m3", 2, 3, 1, 6, inf=False, nan="none")
E3M2 = Fmt("e3m2", 3, 2, 3, 6, inf=False, nan="none")
E2M1 = Fmt("e2m1", 2, 1, 1, 4, inf=False, nan="none")
UE5M3 = Fmt("ue5m3", 5, 3, 15, 8, signed=False, inf=False, nan="allones")

assert E4M3.max_norm == 448 and E5M2.max_norm == 57344 and E2M3.max_norm == Fraction(15, 2)
assert E3M2.max_norm == 28 and E2M1.max_norm == 6 and F16.max_norm == 65504
assert UE5M3.max_norm == Fraction(7, 4) * 2 ** 16


class UE8M0:
    """OCP MX v1.0 5.4.1: E8M0, value 2^(E-127), E in 0..254, 0xFF = NaN, no zero/sign/inf."""
    name = "ue8m0"
    width = 8
    max_norm = Fraction(2) ** 127
    min_val = Fraction(2) ** -127

    @staticmethod
    def decode(b):
        if b == 0xFF:
            return ("nan",)
        return ("fin", 0, Fraction(2) ** (b - 127))

    @staticmethod
    def is_nan(b):
        return b == 0xFF


FMTS = {f.name: f for f in (F64, F32, F16, BF16, E4M3, E5M2, E2M3, E3M2, E2M1, UE5M3)}


def tf32_to_f32bits(b19):
    return b19 << 13


def value(fmt, b):
    """Decoded value as Fraction / float('inf') / None(NaN), sign folded in (zero sign lost)."""
    d = fmt.decode(b)
    if d[0] == "nan":
        return None
    if d[0] == "inf":
        return float("-inf") if d[1] else float("inf")
    return -d[2] if d[1] else d[2]

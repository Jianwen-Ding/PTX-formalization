"""Exact-rational IEEE-754 reference for binary16/32/64 with all rounding modes.

Independent of K and of the host FPU: values are Python Fractions, rounding is
done on bit patterns.  Used by gen_round_diff.py to cross-check the K semantics
of PTX (.rn/.rz/.rm/.rp, .ftz, .sat).
"""
from fractions import Fraction
from math import isqrt

FMT = {16: (11, -14, 15, 5), 32: (24, -126, 127, 8), 64: (53, -1022, 1023, 11)}  # p, emin, emax, ebits

def canon_nan(w): return (1 << (w - 1)) - 1

def classify(b, w):
    p, emin, emax, eb = FMT[w]
    s = b >> (w - 1)
    e = (b >> (p - 1)) & ((1 << eb) - 1)
    m = b & ((1 << (p - 1)) - 1)
    if e == (1 << eb) - 1:
        return ('nan', None, s) if m else ('inf', None, s)
    if e == 0:
        v = Fraction(m) * Fraction(2) ** (emin - p + 1)
    else:
        v = Fraction(m + (1 << (p - 1))) * Fraction(2) ** (e - 1 + emin - p + 1)
    return ('zero' if v == 0 else 'fin', v, s)

def is_subnormal(b, w):
    p, _, _, eb = FMT[w]
    return ((b >> (p - 1)) & ((1 << eb) - 1)) == 0 and (b & ((1 << (p - 1)) - 1)) != 0

def flush(b, w):
    return b & (1 << (w - 1)) if is_subnormal(b, w) else b

def mag_to_bits(val, w):
    """bits of an exactly representable non-negative magnitude"""
    p, emin, emax, eb = FMT[w]
    if val == 0:
        return 0
    e = val.numerator.bit_length() - val.denominator.bit_length()
    if Fraction(2) ** e > val:
        e -= 1
    if e < emin:
        n = val / Fraction(2) ** (emin - p + 1)
        assert n.denominator == 1
        return int(n)
    n = val / Fraction(2) ** (e - p + 1)
    assert n.denominator == 1 and (1 << (p - 1)) <= n < (1 << p), (val, e)
    return ((e - emin + 1) << (p - 1)) | (int(n) - (1 << (p - 1)))

def round_mag(q, sticky, mode, neg, w):
    """round the true magnitude q + sticky*eps (q >= 0) -> magnitude bits"""
    p, emin, emax, eb = FMT[w]
    inf_bits = ((1 << eb) - 1) << (p - 1)
    maxfin = inf_bits - 1
    if q >= Fraction(2) ** (emax + 1):
        lo, hi, frac_cmp = maxfin, inf_bits, 1   # far above the grid: nearest is inf
        exact = False
    else:
        if q == 0:
            e = emin
        else:
            e = q.numerator.bit_length() - q.denominator.bit_length()
            if Fraction(2) ** e > q:
                e -= 1
            e = max(e, emin)
        quantum = Fraction(2) ** (e - p + 1)
        n = q / quantum
        lo_n = n.numerator // n.denominator
        frac = n - lo_n
        if frac == 0:
            b = mag_to_bits(q, w)
            exact = True
            if sticky > 0: lo, hi = b, b + 1
            elif sticky < 0: lo, hi = b - 1, b
            else: lo = hi = b
            frac_cmp = 0
        else:
            exact = False
            lo = mag_to_bits(lo_n * quantum, w)
            hi = lo + 1
            frac_cmp = (frac * 2 > 1) - (frac * 2 < 1)   # 1: above midpoint, 0: tie, -1: below
            if frac_cmp == 0: frac_cmp = 0
    if lo == hi:
        return lo
    if mode == 'rz': return lo
    if mode == 'rm': return hi if neg else lo
    if mode == 'rp': return lo if neg else hi
    # nearest (rn); only reached for inexact / sticky cases
    if exact:
        # q is a grid point; true value is q +- eps: nearest is q
        return lo if sticky < 0 else hi if sticky > 0 else lo
    if q >= Fraction(2) ** (emax + 1):
        return hi
    if frac_cmp < 0: return lo
    if frac_cmp > 0: return hi
    if sticky > 0: return hi
    if sticky < 0: return lo
    return lo if lo % 2 == 0 else hi

def finish(q, sticky, neg, mode, w, ftz, sat):
    """q: exact signed value (Fraction) or None for special handled by caller"""
    mag = round_mag(abs(q), sticky, mode, neg, w)
    return post((neg << (w - 1)) | mag, w, ftz, sat)

def post(b, w, ftz, sat):
    if ftz:
        b = flush(b, w)
    if sat:
        k, v, s = classify(b, w)
        if k == 'nan': return 0
        if k in ('zero',): return b          # sign of zero: callers avoid vectors where it matters
        if k == 'inf': return (1 << (FMT[w][0] - 1)) * 0 + (0 if s else one_bits(w))
        if s: return 0
        if v > 1: return one_bits(w)
    return b

def one_bits(w):
    return mag_to_bits(Fraction(1), w)

def inf_bits(s, w):
    p, _, _, eb = FMT[w]
    return (s << (w - 1)) | (((1 << eb) - 1) << (p - 1))

def zero_bits(s, w): return s << (w - 1)

def ref(op, args, mode, w, ftz=False, sat=False):
    args = [flush(a, w) if ftz else a for a in args]
    cl = [classify(a, w) for a in args]
    nan = canon_nan(w)
    if any(k == 'nan' for k, _, _ in cl):
        return post(nan, w, ftz, sat)
    if op in ('add', 'sub'):
        (ka, va, sa), (kb, vb, sb) = cl
        if op == 'sub': sb ^= 1
        if ka == 'inf' and kb == 'inf':
            return post(nan if sa != sb else inf_bits(sa, w), w, ftz, sat)
        if ka == 'inf': return post(inf_bits(sa, w), w, ftz, sat)
        if kb == 'inf': return post(inf_bits(sb, w), w, ftz, sat)
        xa = (-cl[0][1] if cl[0][2] else cl[0][1])
        xb0 = (-cl[1][1] if cl[1][2] else cl[1][1])
        xb = -xb0 if op == 'sub' else xb0
        q = xa + xb
        if q == 0:
            if ka == 'zero' and kb == 'zero' and sa == sb:
                return post(zero_bits(sa, w), w, ftz, sat)
            return post(zero_bits(1 if mode == 'rm' else 0, w), w, ftz, sat)
        return finish(q, 0, q < 0, mode, w, ftz, sat)
    if op == 'mul':
        (ka, va, sa), (kb, vb, sb) = cl
        s = sa ^ sb
        if (ka == 'inf' and kb == 'zero') or (ka == 'zero' and kb == 'inf'):
            return post(nan, w, ftz, sat)
        if ka == 'inf' or kb == 'inf': return post(inf_bits(s, w), w, ftz, sat)
        if ka == 'zero' or kb == 'zero': return post(zero_bits(s, w), w, ftz, sat)
        return finish(va * vb, 0, s, mode, w, ftz, sat)
    if op == 'div':
        (ka, va, sa), (kb, vb, sb) = cl
        s = sa ^ sb
        if (ka == 'zero' and kb == 'zero') or (ka == 'inf' and kb == 'inf'):
            return post(nan, w, ftz, sat)
        if ka == 'inf': return post(inf_bits(s, w), w, ftz, sat)
        if kb == 'inf': return post(zero_bits(s, w), w, ftz, sat)
        if kb == 'zero': return post(inf_bits(s, w), w, ftz, sat)
        if ka == 'zero': return post(zero_bits(s, w), w, ftz, sat)
        return finish(va / vb, 0, s, mode, w, ftz, sat)
    if op == 'sqrt':
        (ka, va, sa), = cl
        if ka == 'zero': return post(zero_bits(sa, w), w, ftz, sat)
        if sa: return post(nan, w, ftz, sat)
        if ka == 'inf': return post(inf_bits(0, w), w, ftz, sat)
        K = 1300
        nn = va.numerator * (1 << (2 * K)) // va.denominator
        r = isqrt(nn)
        exact = (Fraction(r) ** 2 == va * (1 << (2 * K)))
        q = Fraction(r, 1 << K)
        return finish(q, 0 if exact else 1, 0, mode, w, ftz, sat)
    if op == 'fma':
        (ka, va, sa), (kb, vb, sb), (kc, vc, sc) = cl
        ps = sa ^ sb
        if (ka == 'inf' and kb == 'zero') or (ka == 'zero' and kb == 'inf'):
            return post(nan, w, ftz, sat)
        pinf = ka == 'inf' or kb == 'inf'
        if pinf and kc == 'inf' and sc != ps: return post(nan, w, ftz, sat)
        if pinf: return post(inf_bits(ps, w), w, ftz, sat)
        if kc == 'inf': return post(inf_bits(sc, w), w, ftz, sat)
        prod = (va * vb if ka != 'zero' and kb != 'zero' else Fraction(0))
        prod = -prod if ps else prod
        cval = -vc if sc else vc
        if kc == 'zero': cval = Fraction(0)
        q = prod + cval
        if q == 0:
            pz = ka == 'zero' or kb == 'zero'
            if pz and kc == 'zero' and ps == sc:
                return post(zero_bits(ps, w), w, ftz, sat)
            return post(zero_bits(1 if mode == 'rm' else 0, w), w, ftz, sat)
        return finish(q, 0, q < 0, mode, w, ftz, sat)
    raise ValueError(op)

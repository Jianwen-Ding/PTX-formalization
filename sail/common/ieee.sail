/* IEEE-754 arithmetic as modelled by PTX-FLOAT / PTX-COMMON (semantics/ptx-float.k, ptx-common.k).
 *
 * The K model computes round-to-nearest results with MPFR at the operand format and the directed
 * modes (.rz .rm .rp .rna) from exact (add, sub, mul, fma) or sticky-corrected (div, sqrt) wide
 * values.  Both give the correctly rounded IEEE result, so this file computes every operation
 * exactly on integers (value = (-1)^s * m * 2^e) and rounds once, with the K model's conventions:
 *   - formats are identified by K's format codes: 16 (binary16), 32, 64, 116 (bfloat16 in a 16-bit
 *     container), 132 (tf32: 11-bit precision in a binary32 container);
 *   - every NaN result is the canonical NaN mask(cw - 1) (positive, all payload bits set);
 *   - .ftz flushes subnormal inputs and outputs to a zero of the same sign;
 *   - .sat clamps to [0, 1] (NaN -> +0), .relu clamps negatives and -0 to +0, .satfinite maps
 *     infinities to the largest finite value of the same sign.
 */

enum rmode = { RN, RZ, RM, RP, RNA }

struct fmods = {
  rm : rmode,
  ftz : bool,
  sat : bool,
  relu : bool,
  satfinite : bool,
}

let fm_rn : fmods = struct { rm = RN, ftz = false, sat = false, relu = false, satfinite = false }
let fm_ftz : fmods = struct { rm = RN, ftz = true, sat = false, relu = false, satfinite = false }

/* a float value: NaN, signed infinity, or finite (sign, m >= 0, e) = (-1)^sign * m * 2^e */
union fv = {
  FV_NaN : unit,
  FV_Inf : bool,
  FV_Fin : (bool, int, int),
}

/* ------------------------------------------------------------------ formats */
val prec : int -> int
function prec(w) = match w { 116 => 8, 132 => 11, 16 => 11, 32 => 24, 64 => 53, _ => 53 }
val ebits : int -> int
function ebits(w) = match w { 116 => 8, 132 => 8, 16 => 5, 32 => 8, 64 => 11, _ => 11 }
val cw : int -> int
function cw(w) = match w { 116 => 16, 132 => 32, _ => w }

/* ------------------------------------------------------------------ rounding */
val fv_max : (bool, int, int) -> fv
function fv_max(s, p, eb) = {
  let bias = pow2(eb - 1) - 1;
  FV_Fin(s, pow2(p) - 1, bias - (p - 1))
}

/* Round (-1)^s * (m + d) * 2^e to precision p / eb exponent bits, where d = 0 if not sticky and
 * 0 < d < 1 if sticky (callers make m wide enough that the sticky part lies below the rounding point). */
val fround : (bool, int, int, bool, int, int, rmode) -> fv
function fround(s, m, e, sticky, p, eb, mode) = {
  if m == 0 & not_bool(sticky) then return FV_Fin(s, 0, 0);
  let bias = pow2(eb - 1) - 1;
  let emin = 1 - bias;
  let emax = bias;
  let nb = bitlen(m);
  let lead = nb - 1 + e;
  var lsb : int = if lead < emin then emin - (p - 1) else lead - (p - 1);
  let shift = lsb - e;
  var q : int = 0;
  var r : int = 0;
  var half : int = 0;
  if shift <= 0 then {
    q = m * pow2(negate(shift));
    r = 0
  } else {
    q = shr_int(m, shift);
    r = m - q * pow2(shift);
    half = pow2(shift - 1)
  };
  let inexact = r != 0 | sticky;
  let up : bool = match mode {
    RN => shift > 0 & (r > half | (r == half & (sticky | fmod_int(q, 2) == 1))),
    RNA => shift > 0 & r >= half,
    RZ => false,
    RP => inexact & not_bool(s),
    RM => inexact & s
  };
  if up then q = q + 1;
  if q == pow2(p) then { q = pow2(p - 1); lsb = lsb + 1 };
  if q >= pow2(p - 1) & lsb + p - 1 > emax then {
    match mode {
      RN => FV_Inf(s),
      RNA => FV_Inf(s),
      RZ => fv_max(s, p, eb),
      RP => if s then fv_max(s, p, eb) else FV_Inf(s),
      RM => if s then FV_Inf(s) else fv_max(s, p, eb)
    }
  } else FV_Fin(s, q, lsb)
}

/* round a value to a (precision, exponent bits) format */
val round_fv : (fv, int, int, rmode) -> fv
function round_fv(v, p, eb, mode) = match v {
  FV_Fin(s, m, e) => fround(s, m, e, false, p, eb, mode),
  _ => v
}

/* ------------------------------------------------------------------ bits <-> values */
val decodeAux : (int, int, int, int) -> fv
function decodeAux(s, e, m, w) = {
  let p = prec(w);
  let eb = ebits(w);
  let bias = pow2(eb - 1) - 1;
  if e == mask_int(eb) then {
    if m == 0 then FV_Inf(s == 1) else FV_NaN()
  } else if e == 0 then FV_Fin(s == 1, m, 1 - bias - (p - 1))
  else FV_Fin(s == 1, m + pow2(p - 1), e - bias - (p - 1))
}

/* decodeF(B, W) of ptx-common.k */
val decodeF : (int, int) -> fv
function decodeF(b, w) = {
  if w == 132 then round_fv(decodeF(b, 32), 11, 8, RN)
  else decodeAux(land_int(asr_int(b, cw(w) - 1), 1), land_int(asr_int(b, prec(w) - 1), mask_int(ebits(w))),
                 land_int(b, mask_int(prec(w) - 1)), w)
}

/* exact encoding of a value that is representable in the format */
val encExact : (fv, int) -> int
function encExact(v, w) = {
  let p = prec(w);
  let eb = ebits(w);
  let c = cw(w);
  let bias = pow2(eb - 1) - 1;
  match v {
    FV_NaN() => mask_int(c - 1),
    FV_Inf(s) => (if s then pow2(c - 1) else 0) + mask_int(eb) * pow2(p - 1),
    FV_Fin(s, q, lsb) => {
      let sb = if s then pow2(c - 1) else 0;
      if q == 0 then sb
      else if q < pow2(p - 1) then sb + q
      else sb + (lsb + (p - 1) + bias) * pow2(p - 1) + (q - pow2(p - 1))
    }
  }
}

/* encodeF(F, W) of ptx-common.k: round to nearest even at the format, then encode */
val encodeF : (fv, int) -> int
function encodeF(v, w) = {
  if w == 132 then encodeF(round_fv(v, 11, 8, RN), 32)
  else encExact(round_fv(v, prec(w), ebits(w), RN), w)
}

/* the f64 bit pattern of a K Float literal (double precision) as a value */
val f64v : int -> fv
function f64v(b) = decodeF(b, 64)

val fv_is_nan : fv -> bool
function fv_is_nan(v) = match v { FV_NaN() => true, _ => false }
val fv_is_inf : fv -> bool
function fv_is_inf(v) = match v { FV_Inf(_) => true, _ => false }
val fv_is_zero : fv -> bool
function fv_is_zero(v) = match v { FV_Fin(_, m, _) => m == 0, _ => false }
val fv_sign : fv -> bool
function fv_sign(v) = match v { FV_NaN() => false, FV_Inf(s) => s, FV_Fin(s, _, _) => s }
val fv_neg : fv -> fv
function fv_neg(v) = match v { FV_NaN() => FV_NaN(), FV_Inf(s) => FV_Inf(not_bool(s)), FV_Fin(s, m, e) => FV_Fin(not_bool(s), m, e) }
val fv_abs : fv -> fv
function fv_abs(v) = match v { FV_NaN() => FV_NaN(), FV_Inf(_) => FV_Inf(false), FV_Fin(_, m, e) => FV_Fin(false, m, e) }

/* signed integer numerator at a common exponent: value = n * 2^e */
val fv_scaled : (fv, int) -> int
function fv_scaled(v, e) = match v {
  FV_Fin(s, m, e1) => { let n = m * pow2(e1 - e); if s then negate(n) else n },
  _ => 0
}
val fv_exp : fv -> int
function fv_exp(v) = match v { FV_Fin(_, _, e) => e, _ => 0 }

/* three-way comparison of two non-NaN values: -1, 0, 1 (-0 == +0) */
val fv_cmp : (fv, fv) -> int
function fv_cmp(a, b) = {
  match (a, b) {
    (FV_Inf(sa), FV_Inf(sb)) => if sa == sb then 0 else if sa then -1 else 1,
    (FV_Inf(sa), _) => if sa then -1 else 1,
    (_, FV_Inf(sb)) => if sb then 1 else -1,
    _ => {
      let e = min_int(fv_exp(a), fv_exp(b));
      let x = fv_scaled(a, e);
      let y = fv_scaled(b, e);
      if x < y then -1 else if x > y then 1 else 0
    }
  }
}

/* K's ordered comparisons on Float (<Float, ==Float, ...): false when either is NaN */
val fv_lt : (fv, fv) -> bool
function fv_lt(a, b) = not_bool(fv_is_nan(a) | fv_is_nan(b)) & fv_cmp(a, b) < 0
val fv_eq : (fv, fv) -> bool
function fv_eq(a, b) = not_bool(fv_is_nan(a) | fv_is_nan(b)) & fv_cmp(a, b) == 0

/* ------------------------------------------------------------------ ftz, clamps, results */
val isSubnB : (int, int) -> bool
function isSubnB(b, w) =
  land_int(asr_int(b, prec(w) - 1), mask_int(ebits(w))) == 0 & land_int(b, mask_int(prec(w) - 1)) != 0
val flushB : (int, int) -> int
function flushB(b, w) = if isSubnB(b, w) then land_int(b, pow2(cw(w) - 1)) else b
val ftzIn : (fmods, int, int) -> int
function ftzIn(m, w, b) = if m.ftz then flushB(b, w) else b
val ftzOut : (fmods, int, int) -> int
function ftzOut(m, w, b) = if m.ftz then flushB(b, w) else b

val fv_one : unit -> fv
function fv_one() = FV_Fin(false, 1, 0)

val satF : fv -> fv
function satF(f) = {
  if fv_is_nan(f) | fv_lt(f, FV_Fin(false, 0, 0)) then FV_Fin(false, 0, 0)
  else if fv_lt(fv_one(), f) then fv_one()
  else f
}
val reluF : fv -> fv
function reluF(f) = {
  if not_bool(fv_is_nan(f)) & (fv_lt(f, FV_Fin(false, 0, 0)) | (fv_is_zero(f) & fv_sign(f))) then FV_Fin(false, 0, 0)
  else f
}
val clampF : (fmods, fv, int) -> fv
function clampF(m, f, w) = {
  if m.sat then satF(f)
  else if m.relu then reluF(f)
  else if m.satfinite & fv_is_inf(f) then fv_max(fv_sign(f), prec(w), ebits(w))
  else f
}
/* post(M, W, F) of ptx-float.k for a value already rounded to the format */
val post : (fmods, int, fv) -> int
function post(m, w, f) = ftzOut(m, w, encodeF(clampF(m, f, w), w))

/* round an exact result (or a NaN / infinity) to the format with the instruction's mode */
val fin : (fmods, int, fv, bool) -> int
function fin(m, w, v, sticky) = {
  let r : fv = match v {
    FV_Fin(s, q, e) => if w == 132 then fround(s, q, e, sticky, 11, 8, m.rm) else fround(s, q, e, sticky, prec(w), ebits(w), m.rm),
    _ => v
  };
  post(m, w, r)
}

/* ------------------------------------------------------------------ operations */
/* exact sum of two values (NaN / infinity rules of IEEE); zero sign: same-sign zeros keep the
 * sign, otherwise +0, or -0 under round-down */
val fv_add_exact : (fv, fv, rmode) -> fv
function fv_add_exact(a, b, mode) = {
  match (a, b) {
    (FV_NaN(), _) => FV_NaN(),
    (_, FV_NaN()) => FV_NaN(),
    (FV_Inf(sa), FV_Inf(sb)) => if sa == sb then FV_Inf(sa) else FV_NaN(),
    (FV_Inf(sa), _) => FV_Inf(sa),
    (_, FV_Inf(sb)) => FV_Inf(sb),
    (FV_Fin(sa, ma, ea), FV_Fin(sb, mb, eb)) => {
      let e = min_int(ea, eb);
      let n = fv_scaled(a, e) + fv_scaled(b, e);
      if n == 0 then {
        if sa == sb & ma == 0 & mb == 0 then FV_Fin(sa, 0, 0)
        else FV_Fin(mode == RM, 0, 0)
      } else if n < 0 then FV_Fin(true, negate(n), e) else FV_Fin(false, n, e)
    }
  }
}

val fv_mul_exact : (fv, fv) -> fv
function fv_mul_exact(a, b) = {
  match (a, b) {
    (FV_NaN(), _) => FV_NaN(),
    (_, FV_NaN()) => FV_NaN(),
    (FV_Inf(sa), FV_Inf(sb)) => FV_Inf(sa != sb),
    (FV_Inf(sa), FV_Fin(sb, mb, _)) => if mb == 0 then FV_NaN() else FV_Inf(sa != sb),
    (FV_Fin(sa, ma, _), FV_Inf(sb)) => if ma == 0 then FV_NaN() else FV_Inf(sa != sb),
    (FV_Fin(sa, ma, ea), FV_Fin(sb, mb, eb)) => FV_Fin(sa != sb, ma * mb, ea + eb)
  }
}

val fpAdd : (fmods, int, int, int) -> int
function fpAdd(m, w, a, b) =
  fin(m, w, fv_add_exact(decodeF(ftzIn(m, w, a), w), decodeF(ftzIn(m, w, b), w), m.rm), false)

val fpSub : (fmods, int, int, int) -> int
function fpSub(m, w, a, b) = fpAdd(m, w, a, lxor_int(b, pow2(cw(w) - 1)))

val fpMul : (fmods, int, int, int) -> int
function fpMul(m, w, a, b) =
  fin(m, w, fv_mul_exact(decodeF(ftzIn(m, w, a), w), decodeF(ftzIn(m, w, b), w)), false)

val fv_fma_exact : (fv, fv, fv, rmode) -> fv
function fv_fma_exact(a, b, c, mode) = {
  let p = fv_mul_exact(a, b);
  match (p, c) {
    (FV_Fin(sp, mp, _), FV_Fin(sc, mc, _)) =>
      if mp == 0 & mc == 0 then { if sp == sc then FV_Fin(sp, 0, 0) else FV_Fin(mode == RM, 0, 0) }
      else fv_add_exact(p, c, mode),
    _ => fv_add_exact(p, c, mode)
  }
}

val fpFma : (fmods, int, int, int, int) -> int
function fpFma(m, w, a, b, c) =
  fin(m, w, fv_fma_exact(decodeF(ftzIn(m, w, a), w), decodeF(ftzIn(m, w, b), w), decodeF(ftzIn(m, w, c), w), m.rm), false)

/* quotient with enough bits for rounding plus a sticky flag */
val fv_div : (fmods, int, fv, fv) -> int
function fv_div(m, w, x, y) = {
  match (x, y) {
    (FV_NaN(), _) => fin(m, w, FV_NaN(), false),
    (_, FV_NaN()) => fin(m, w, FV_NaN(), false),
    (FV_Inf(_), FV_Inf(_)) => fin(m, w, FV_NaN(), false),
    (FV_Inf(sx), FV_Fin(sy, _, _)) => fin(m, w, FV_Inf(sx != sy), false),
    (FV_Fin(sx, _, _), FV_Inf(sy)) => fin(m, w, FV_Fin(sx != sy, 0, 0), false),
    (FV_Fin(sx, mx, ex), FV_Fin(sy, my, ey)) => {
      if my == 0 then { if mx == 0 then fin(m, w, FV_NaN(), false) else fin(m, w, FV_Inf(sx != sy), false) }
      else if mx == 0 then fin(m, w, FV_Fin(sx != sy, 0, 0), false)
      else {
        let k = max_int(0, prec(w) + 4 + bitlen(my) - bitlen(mx));
        let num = mx * pow2(k);
        let q = tdiv_int(num, my);
        let r = num - q * my;
        fin(m, w, FV_Fin(sx != sy, q, ex - ey - k), r != 0)
      }
    }
  }
}

val fpDiv : (fmods, int, int, int) -> int
function fpDiv(m, w, a, b) = fv_div(m, w, decodeF(ftzIn(m, w, a), w), decodeF(ftzIn(m, w, b), w))

/* integer square root (floor) */
val isqrt : int -> int
function isqrt(n) = {
  if n <= 0 then 0 else {
    var x : int = pow2(tdiv_int(bitlen(n) + 1, 2));
    var y : int = tdiv_int(x + tdiv_int(n, x), 2);
    while y < x do { x = y; y = tdiv_int(x + tdiv_int(n, x), 2) };
    x
  }
}

val fpSqrt : (fmods, int, int) -> int
function fpSqrt(m, w, a) = {
  let x = decodeF(ftzIn(m, w, a), w);
  match x {
    FV_NaN() => fin(m, w, FV_NaN(), false),
    FV_Inf(s) => if s then fin(m, w, FV_NaN(), false) else fin(m, w, x, false),
    FV_Fin(s, mx, ex) => {
      if mx == 0 then fin(m, w, x, false)
      else if s then fin(m, w, FV_NaN(), false)
      else {
        /* value = mx * 2^ex; scale mx by an even power so that ex' is even and the root has >= p + 4 bits */
        var k : int = max_int(0, 2 * (prec(w) + 4) - bitlen(mx) + 2);
        if fmod_int(ex - k, 2) != 0 then k = k + 1;
        let n = mx * pow2(k);
        let r = isqrt(n);
        fin(m, w, FV_Fin(false, r, tdiv_int(ex - k, 2)), r * r != n)
      }
    }
  }
}

val fpRcp : (fmods, int, int) -> int
function fpRcp(m, w, a) = fpDiv(m, w, encodeF(fv_one(), w), a)

/* ------------------------------------------------------------------ min / max (ptx-float.k fpMinMax) */
val signB : (int, int) -> int
function signB(b, w) = land_int(asr_int(b, cw(w) - 1), 1)
val isNaNB : (int, int) -> bool
function isNaNB(b, w) =
  land_int(asr_int(b, prec(w) - 1), mask_int(ebits(w))) == mask_int(ebits(w)) & land_int(b, mask_int(prec(w) - 1)) != 0

val mmLess : (bool, int, int, int) -> bool
function mmLess(xorsign, w, a, b) =
  if xorsign then fv_lt(decodeF(land_int(a, mask_int(cw(w) - 1)), w), decodeF(land_int(b, mask_int(cw(w) - 1)), w))
  else fv_lt(decodeF(a, w), decodeF(b, w))

val mmPick : (bool, int, int, int, bool) -> int
function mmPick(xorsign, w, a, b, isMin) = {
  if xorsign | land_int(a, mask_int(cw(w) - 1)) != 0 | land_int(b, mask_int(cw(w) - 1)) != 0 then {
    if mmLess(xorsign, w, a, b) then (if isMin then a else b)
    else if mmLess(xorsign, w, b, a) then (if isMin then b else a)
    else (if isMin then lor_int(a, b) else land_int(a, b))
  } else (if isMin then lor_int(a, b) else land_int(a, b))
}

val mmSign : (bool, int, int, int) -> int
function mmSign(xorsign, w, r, xs) =
  if xorsign then lor_int(land_int(r, mask_int(cw(w) - 1)), xs * pow2(cw(w) - 1)) else r

/* fpMinMax(M, W, A, B, IsMin) with M given by its ftz / NaN / xorsign flags */
val fpMinMax : (bool, bool, bool, int, int, int, bool) -> int
function fpMinMax(ftz, nanf, xorsign, w, a0, b0, isMin) = {
  let mods : fmods = { fm_rn with ftz = ftz };
  let a = ftzIn(mods, w, a0);
  let b = ftzIn(mods, w, b0);
  let xs = lxor_int(signB(a, w), signB(b, w));
  if isNaNB(a, w) | isNaNB(b, w) then {
    if nanf | (isNaNB(a, w) & isNaNB(b, w)) then mask_int(cw(w) - 1)
    else mmSign(xorsign, w, if isNaNB(a, w) then b else a, xs)
  } else mmSign(xorsign, w, mmPick(xorsign, w, a, b, isMin), xs)
}

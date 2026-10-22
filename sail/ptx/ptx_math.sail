/* =============================================================================
 * The transcendental functions of PTX-ALU (rsqrt sin cos lg2 ex2 tanh): the K model evaluates
 * them with K's builtin FLOAT (MPFR, correctly rounded to nearest at the operand's precision):
 *   rsqrt(x) = 1.0 / sqrt(x)                 two roundings
 *   sin(x), cos(x), 2.0 ^ x                  one rounding
 *   lg2(x) = log2(x)                         one rounding (log computed at 300 bits, then rounded)
 *   tanh(x) = (e^x - e^-x) / (e^x + e^-x)    for |x| < 20 (each operation rounded) unless e^x overflows, else +-1
 * Here each correctly rounded elementary function is computed with integer arithmetic: an
 * approximation with an error bound, refined until both ends of the error interval round to the
 * same value (Ziv's strategy).
 * ============================================================================= */

/* rounding at a format: precision and exponent bits */
val fmt_p : int -> int
function fmt_p(w) = if w == 132 then 11 else prec(w)
val fmt_eb : int -> int
function fmt_eb(w) = if w == 132 then 8 else ebits(w)

/* round the integer v * 2^e to format w (nearest even) */
val rnd_ie : (int, int, int) -> fv
function rnd_ie(v, e, w) = if v == 0 then FV_Fin(false, 0, 0) else fround(v < 0, abs_int(v), e, false, fmt_p(w), fmt_eb(w), RN)

val fv_same : (fv, fv, int) -> bool
function fv_same(a, b, w) = encExact(a, if w == 132 then 32 else w) == encExact(b, if w == 132 then 32 else w) & fv_sign(a) == fv_sign(b)

/* round the interval [v - err, v + err] * 2^e; None if its ends round differently */
val round_iv : (int, int, int, int) -> option(fv)
function round_iv(v, err, e, w) = {
  let a = rnd_ie(v - err, e, w);
  let b = rnd_ie(v + err, e, w);
  if fv_same(a, b, w) then Some(a) else None()
}

/* floor(x * 2^q) of a finite value */
val fixed : (fv, int) -> int
function fixed(x, q) = match x {
  FV_Fin(s, m, e) => { let n = fdiv_int(m * pow2(max_int(0, e + q)), pow2(max_int(0, negate(e + q)))); if s then negate(n) else n },
  _ => 0
}

val fexp : fv -> int
/* exponent of the leading bit of a nonzero finite value */
function fexp(x) = match x { FV_Fin(_, m, e) => bitlen(m) - 1 + e, _ => 0 }

/* ln 2 * 2^q, error < 2: 2 atanh(1/3) = sum 2 / ((2k+1) 3^(2k+1)) */
val ln2_fixed : int -> int
function ln2_fixed(q) = {
  let g = q + 16;
  var s : int = 0;
  var t : int = tdiv_int(pow2(g + 1), 3);      /* 2 / 3 */
  var k : int = 0;
  while t > 0 do { s = s + tdiv_int(t, 2 * k + 1); t = tdiv_int(t, 9); k = k + 1 };
  shr_int(s, 16)
}

/* atan(1/n) * 2^g */
val atan_inv : (int, int) -> int
function atan_inv(n, g) = {
  var s : int = 0;
  var t : int = tdiv_int(pow2(g), n);
  var k : int = 0;
  while t > 0 do {
    if fmod_int(k, 2) == 0 then s = s + tdiv_int(t, 2 * k + 1) else s = s - tdiv_int(t, 2 * k + 1);
    t = tdiv_int(t, n * n);
    k = k + 1
  };
  s
}

/* pi / 2 * 2^q, error < 2 (Machin) */
val halfpi_fixed : int -> int
function halfpi_fixed(q) = {
  let g = q + 16;
  shr_int(8 * atan_inv(5, g) - 2 * atan_inv(239, g), 16)
}

/* exp(r) * 2^g for a fixed-point r = R * 2^-g with |r| < 1; returns (value, number of terms) */
val exp_series : (int, int) -> (int, int)
function exp_series(r, g) = {
  let one = pow2(g);
  var s : int = one;
  var t : int = one;
  var i : int = 1;
  while t != 0 do { t = tdiv_int(t * r, one * i); s = s + t; i = i + 1 };
  (s, i)
}

/* sin(r) / cos(r) * 2^g for |r| <= pi/4 */
val sincos_series : (int, int, bool) -> (int, int)
function sincos_series(r, g, issin) = {
  let one = pow2(g);
  var t : int = if issin then r else one;
  var s : int = t;
  var i : int = if issin then 1 else 0;
  var n : int = 0;
  while t != 0 do { t = negate(tdiv_int(t * r * r, one * one * (i + 1) * (i + 2))); s = s + t; i = i + 2; n = n + 1 };
  (s, n)
}

/* exp(x) for a finite x with |x| <= 20000 at working precision p: (t, err, sh) with e^x in [t - err, t + err] * 2^sh */
val exp_fx : (fv, int) -> (int, int, int)
function exp_fx(x, p) = {
  let b = 16;
  let g = p + b + 8;
  let xf = fixed(x, g);
  let l = ln2_fixed(g);
  let k = fdiv_int(2 * xf + l, 2 * l);
  let rr = xf - k * l;
  let rg = asr_int(rr, b + 8);
  let (t, n) = exp_series(rg, p);
  (t, n + 8 + 4 * abs_int(tdiv_int(abs_int(k) * 2, pow2(b + 8)) + 1), k - p)
}

/* correctly rounded exp(x) at format w */
val exp_r : (fv, int) -> fv
function exp_r(x, w) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(s) => if s then FV_Fin(false, 0, 0) else FV_Inf(false),
  FV_Fin(s, m, _) => {
    if m == 0 then fv_one()
    else if fv_cmp(x, FV_Fin(false, 20000, 0)) > 0 then FV_Inf(false)
    else if fv_cmp(x, FV_Fin(true, 20000, 0)) < 0 then FV_Fin(false, 0, 0)
    else {
      var p : int = fmt_p(w) + 64;
      var r : option(fv) = None();
      while is_none(r) do {
        let (t, err, sh) = exp_fx(x, p);
        r = round_iv(t, err, sh, w);
        p = p + 64
      };
      match r { Some(v) => v, None() => FV_NaN() }
    }
  }
}

/* correctly rounded log(x) at format w */
val log_r : (fv, int) -> fv
function log_r(x, w) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(s) => if s then FV_NaN() else FV_Inf(false),
  FV_Fin(s, m, e) => {
    if m == 0 then FV_Inf(true)
    else if s then FV_NaN()
    else if fv_cmp(x, fv_one()) == 0 then FV_Fin(false, 0, 0)
    else {
      var p : int = fmt_p(w) + 64;
      var r : option(fv) = None();
      while is_none(r) do {
        let g = p + 16;
        let nb = bitlen(m);
        let k = nb - 1 + e;
        let one = pow2(g);
        let y = if g >= nb - 1 then m * pow2(g - (nb - 1)) else shr_int(m, (nb - 1) - g);
        let z = tdiv_int((y - one) * one, y + one);
        let z2 = tdiv_int(z * z, one);
        var sum : int = 0;
        var term : int = z;
        var j : int = 0;
        while term != 0 do { sum = sum + tdiv_int(term, 2 * j + 1); term = tdiv_int(term * z2, one); j = j + 1 };
        let v = k * ln2_fixed(g) + 2 * sum;
        let err = 2 * abs_int(k) + 2 * (j + 4);
        r = round_iv(v, err, negate(g), w);
        p = p + 64
      };
      match r { Some(v) => v, None() => FV_NaN() }
    }
  }
}

/* correctly rounded log2(x) at format w: the exponent k exactly plus ln(m) / ln 2 for the mantissa m in [1, 2) */
val log2_r : (fv, int) -> fv
function log2_r(x, w) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(s) => if s then FV_NaN() else FV_Inf(false),
  FV_Fin(s, m, e) => {
    if m == 0 then FV_Inf(true)
    else if s then FV_NaN()
    else if fv_cmp(x, fv_one()) == 0 then FV_Fin(false, 0, 0)
    else if land_int(m, m - 1) == 0 then rnd_ie(bitlen(m) - 1 + e, 0, w)
    else {
      var p : int = fmt_p(w) + 64;
      var r : option(fv) = None();
      while is_none(r) do {
        let g = p + 16;
        let nb = bitlen(m);
        let k = nb - 1 + e;
        let one = pow2(g);
        let y = if g >= nb - 1 then m * pow2(g - (nb - 1)) else shr_int(m, (nb - 1) - g);
        let z = tdiv_int((y - one) * one, y + one);
        let z2 = tdiv_int(z * z, one);
        var sum : int = 0;
        var term : int = z;
        var j : int = 0;
        while term != 0 do { sum = sum + tdiv_int(term, 2 * j + 1); term = tdiv_int(term * z2, one); j = j + 1 };
        /* 2 sum = ln(m) * 2^g within 2 (j + 4), below 2^g; dividing by ln 2 (error < 2) adds < 3 * (that + 2) + 1 */
        let v = k * one + tdiv_int(2 * sum * one, ln2_fixed(g));
        let err = 3 * (2 * (j + 4) + 2) + 2;
        r = round_iv(v, err, negate(g), w);
        p = p + 64
      };
      match r { Some(v) => v, None() => FV_NaN() }
    }
  }
}

/* correctly rounded 2^x at format w (MPFR pow(2, x)) */
val exp2_r : (fv, int) -> fv
function exp2_r(x, w) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(s) => if s then FV_Fin(false, 0, 0) else FV_Inf(false),
  FV_Fin(s, m, e) => {
    if m == 0 then fv_one()
    else if fv_cmp(x, FV_Fin(false, 20000, 0)) > 0 then FV_Inf(false)
    else if fv_cmp(x, FV_Fin(true, 20000, 0)) < 0 then FV_Fin(false, 0, 0)
    else if e >= 0 then {
      /* integral exponent: exact */
      let n = if s then negate(m * pow2(e)) else m * pow2(e);
      rnd_ie(1, n, w)
    } else {
      /* x = n + f, 0 < f < 1: 2^x = 2^n exp(f ln 2) */
      let num = if s then negate(m) else m;
      let n = fdiv_int(num, pow2(negate(e)));
      let f = num - n * pow2(negate(e));          /* f * 2^-(-e) */
      var p : int = fmt_p(w) + 64;
      var r : option(fv) = None();
      while is_none(r) do {
        let g = p + 8;
        let fg = if g >= negate(e) then f * pow2(g + e) else shr_int(f, negate(e) - g);
        let t = tdiv_int(fg * ln2_fixed(g), pow2(g));
        let (v, nt) = exp_series(t, g);
        r = round_iv(v, nt + 8, n - g, w);
        p = p + 64
      };
      match r { Some(v) => v, None() => FV_NaN() }
    }
  }
}

/* correctly rounded sin / cos at format w */
val sincos_r : (fv, int, bool) -> fv
function sincos_r(x, w, issin) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(_) => FV_NaN(),
  FV_Fin(s, m, _) => {
    if m == 0 then (if issin then x else fv_one())
    else {
      var p : int = fmt_p(w) + 64;
      var r : option(fv) = None();
      while is_none(r) do {
        let b = max_int(0, fexp(x) + 2);
        let g = p + 16;
        let gg = g + b + 12;
        let hp = halfpi_fixed(gg);
        let xf = fixed(x, gg);
        let k = fdiv_int(2 * xf + hp, 2 * hp);
        let rr = asr_int(xf - k * hp, b + 12);
        let q = fmod_int(k, 4);
        let usesin = if issin then q == 0 | q == 2 else q == 1 | q == 3;
        let neg = if issin then q == 2 | q == 3 else q == 1 | q == 2;
        let (v0, n) = sincos_series(rr, g, usesin);
        let v = if neg then negate(v0) else v0;
        r = round_iv(v, n + 16, negate(g), w);
        p = p + 64
      };
      match r { Some(v) => v, None() => FV_NaN() }
    }
  }
}

/* correctly rounded quotient and square root of values at format w */
val div_r : (fv, fv, int) -> fv
function div_r(x, y, w) = match (x, y) {
  (FV_NaN(), _) => FV_NaN(),
  (_, FV_NaN()) => FV_NaN(),
  (FV_Inf(_), FV_Inf(_)) => FV_NaN(),
  (FV_Inf(sx), FV_Fin(sy, _, _)) => FV_Inf(sx != sy),
  (FV_Fin(sx, _, _), FV_Inf(sy)) => FV_Fin(sx != sy, 0, 0),
  (FV_Fin(sx, mx, ex), FV_Fin(sy, my, ey)) => {
    if my == 0 then (if mx == 0 then FV_NaN() else FV_Inf(sx != sy))
    else if mx == 0 then FV_Fin(sx != sy, 0, 0)
    else {
      let k = max_int(0, fmt_p(w) + 4 + bitlen(my) - bitlen(mx));
      let num = mx * pow2(k);
      let q = tdiv_int(num, my);
      fround(sx != sy, q, ex - ey - k, num != q * my, fmt_p(w), fmt_eb(w), RN)
    }
  }
}

val sqrt_r : (fv, int) -> fv
function sqrt_r(x, w) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(s) => if s then FV_NaN() else x,
  FV_Fin(s, m, e) => {
    if m == 0 then x
    else if s then FV_NaN()
    else {
      var k : int = max_int(0, 2 * (fmt_p(w) + 4) - bitlen(m) + 2);
      if fmod_int(e - k, 2) != 0 then k = k + 1;
      let n = m * pow2(k);
      let r = isqrt(n);
      fround(false, r, tdiv_int(e - k, 2), r * r != n, fmt_p(w), fmt_eb(w), RN)
    }
  }
}

val addsub_r : (fv, fv, int) -> fv
function addsub_r(a, b, w) = match fv_add_exact(a, b, RN) {
  FV_Fin(s, m, e) => fround(s, m, e, false, fmt_p(w), fmt_eb(w), RN),
  v => v
}

/* correctly rounded tanh(x) at format w for finite x != 0: tanh|x| = (E - 1) / (E + 1), E = e^(2|x|), increasing in E,
 * so the bounds of E bound tanh|x|; both ends rounded until they agree */
val tanh_r : (fv, int) -> fv
function tanh_r(x, w) = match x {
  FV_Fin(s, m, e) => {
    if fv_cmp(fv_abs(x), FV_Fin(false, 20, 0)) >= 0 then FV_Fin(s, 1, 0)
    else {
      var p : int = fmt_p(w) + 64;
      var r : option(fv) = None();
      while is_none(r) do {
        let (t, err, sh) = exp_fx(FV_Fin(false, m, e + 1), p);
        let a = if sh >= 0 then (t - err) * pow2(sh) else t - err;
        let b = if sh >= 0 then (t + err) * pow2(sh) else t + err;
        let one = if sh >= 0 then 1 else pow2(negate(sh));
        if a > one then {
          let lo = div_r(FV_Fin(s, a - one, 0), FV_Fin(false, a + one, 0), w);
          let hi = div_r(FV_Fin(s, b - one, 0), FV_Fin(false, b + one, 0), w);
          if fv_same(lo, hi, w) then r = Some(lo)
        };
        p = p + 64
      };
      match r { Some(v) => v, None() => FV_NaN() }
    }
  },
  _ => x
}

/* the transcendental instructions on the (ftz'd) source bits a at format w */
function transcendental(op, w, a, raw) = {
  let x = decodeF(a, w);
  match op {
    "rsqrt" => div_r(fv_one(), sqrt_r(x, w), w),
    "sin" => sincos_r(x, w, true),
    "cos" => sincos_r(x, w, false),
    "lg2" => log2_r(x, w),
    "ex2" => exp2_r(x, w),
    "tanh" => {
      /* tanh(NaN) = NaN; below 2^-12 tanh(x) = x within 2^-24 (the difference of exponentials cancels) */
      if fv_is_nan(x) then FV_NaN()
      /* tanh.approx.{f16,bf16} (ISA 9.7.4.9): maximum absolute error 2^-10.987 / 2^-8; the correctly rounded tanh */
      else if (w == 16 | w == 116) & fv_is_inf(x) then FV_Fin(fv_sign(x), 1, 0)
      else if (w == 16 | w == 116) & not_bool(fv_is_zero(x)) then tanh_r(x, w)
      else if fv_cmp(fv_abs(x), FV_Fin(false, 1, -12)) < 0 then x
      else if fv_cmp(fv_abs(x), FV_Fin(false, 20, 0)) < 0 then {
        let e1 = exp_r(x, w);
        let e2 = exp_r(fv_neg(x), w);
        /* f16: e^|x| overflows above ln 65520 ~ 11.09, where tanh(x) rounds to +-1 */
        if fv_is_inf(e1) | fv_is_inf(e2) then (if fv_sign(x) then FV_Fin(true, 1, 0) else fv_one())
        else div_r(addsub_r(e1, fv_neg(e2), w), addsub_r(e1, e2, w), w)
      } else if fv_sign(x) then FV_Fin(true, 1, 0) else fv_one()
    },
    _ => { fault(concat_str("no semantics for ", op), 0); FV_NaN() }
  }
}

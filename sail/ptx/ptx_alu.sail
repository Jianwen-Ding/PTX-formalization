/* =============================================================================
 * PTX-ALU (semantics/ptx-alu.k): pure semantics of arithmetic, logic, comparison, conversion and
 * atomic read-modify-write operations on bit patterns.
 * ============================================================================= */

/* the float modifiers of an instruction: rndOf(M), .ftz, .sat, .relu, .satfinite */
val fmods_of : list(string) -> fmods
function fmods_of(m) = {
  let r : rmode = match pick(m, [|"rn", "rz", "rm", "rp", "rna"|], "rn") {
    "rz" => RZ, "rm" => RM, "rp" => RP, "rna" => RNA, _ => RN
  };
  struct { rm = r, ftz = has(m, "ftz"), sat = has(m, "sat"), relu = has(m, "relu"), satfinite = has(m, "satfinite") }
}

let BINOPS : list(string) = [|"add", "sub", "mul", "div", "rem", "min", "max", "and", "or", "xor", "shl", "shr", "copysign"|]
let UNOPS : list(string) = [|"neg", "abs", "not", "cnot", "popc", "clz", "brev", "sqrt", "rsqrt", "rcp", "sin", "cos", "lg2", "ex2", "tanh"|]
let TERNOPS : list(string) = [|"mad", "fma", "selp"|]

val src2Ty : (string, ty) -> ty
function src2Ty(op, t) = if op == "shl" | op == "shr" then mkty("u", 32) else t

val relu : (list(string), ty, int) -> int
function relu(m, t, v) = if has(m, "relu") & isS(t) & ival(v, t) < 0 then 0 else v

/* "NaN payloads are supported for double-precision instructions" (ISA 9.7.3): an f64 operation with a NaN operand
 * returns the first NaN operand, quieted (the ISA does not say which of several NaNs; the first is chosen) */
val nanPay : (int, list(int)) -> option(int)
function nanPay(w, l) = match l {
  [||] => None(),
  x :: r => if w == 64 & isNaNB(x, 64) then Some(lor_int(x, pow2(51))) else nanPay(w, r)
}

/* div.approx.f32 (ISA 9.7.3.8): d = a * (1 / b); for 2^126 < |b| < 2^128 a NaN for an infinite a, otherwise a
 * sign-preserving zero (a NaN a gives NaN, as a * (1 / b) does) */
val divApprox : (fmods, int, int) -> option(int)
function divApprox(fm, a, b) = {
  let x = decodeF(ftzIn(fm, 32, a), 32);
  let y = decodeF(b, 32);
  if fv_is_nan(y) | fv_is_inf(y) | fv_cmp(fv_abs(y), FV_Fin(false, 1, 126)) <= 0 then None()
  else if fv_is_nan(x) | fv_is_inf(x) then Some(mask(31))
  else Some(lxor_int(land_int(a, pow2(31)), land_int(b, pow2(31))))
}

val bin : (string, list(string), ty, int, int) -> int
function bin(op, m, t, a, b) = {
  if isF(t) & is_some(nanPay(fw(t), [|a, b|])) & (op == "add" | op == "sub" | op == "mul" | op == "div") then
    match nanPay(fw(t), [|a, b|]) { Some(v) => v, None() => 0 }
  else if isF(t) then {
    let fm = fmods_of(m);
    match op {
      "add" => fpAdd(fm, fw(t), a, b),
      "sub" => fpSub(fm, fw(t), a, b),
      "mul" => fpMul(fm, fw(t), a, b),
      "div" => match (if has(m, "approx") & fw(t) == 32 then divApprox(fm, a, b) else None()) {
        Some(v) => v,
        None() => fpDiv(fm, fw(t), a, b)
      },
      "min" => fpMinMax(fm.ftz, has(m, "NaN"), has(m, "xorsign"), fw(t), a, b, true),
      "max" => fpMinMax(fm.ftz, has(m, "NaN"), has(m, "xorsign"), fw(t), a, b, false),
      "copysign" => lor_int(land_int(b, mask(wid(t) - 1)), land_int(a, pow2(wid(t) - 1))),
      "and" => land_int(a, b),
      "or" => lor_int(a, b),
      "xor" => lxor_int(a, b),
      "shl" => if b >= wid(t) then 0 else trunc(lsl_int(a, b), wid(t)),
      "shr" => trunc(asr_int(ival(a, t), min_int(b, wid(t))), wid(t)),
      _ => { fault(concat_str("no float semantics for ", op), 0); 0 }
    }
  } else match op {
    "add" => trunc(if has(m, "sat") then clamp(ival(a, t) + ival(b, t), t) else a + b, wid(t)),
    "sub" => trunc(if has(m, "sat") then clamp(ival(a, t) - ival(b, t), t) else a - b, wid(t)),
    "mul" => {
      if has(m, "wide") then trunc(ival(a, t) * ival(b, t), 2 * wid(t))
      else if has(m, "hi") then trunc(asr_int(ival(a, t) * ival(b, t), wid(t)), wid(t))
      else trunc(ival(a, t) * ival(b, t), wid(t))
    },
    "div" => if ival(b, t) == 0 then mask(wid(t)) else trunc(tdiv_int(ival(a, t), ival(b, t)), wid(t)),
    "rem" => if ival(b, t) == 0 then a else trunc(tmod_int(ival(a, t), ival(b, t)), wid(t)),
    "min" => relu(m, t, if ival(a, t) <= ival(b, t) then a else b),
    "max" => relu(m, t, if ival(a, t) >= ival(b, t) then a else b),
    "and" => land_int(a, b),
    "or" => lor_int(a, b),
    "xor" => lxor_int(a, b),
    "shl" => if b >= wid(t) then 0 else trunc(lsl_int(a, b), wid(t)),
    "shr" => trunc(asr_int(ival(a, t), min_int(b, wid(t))), wid(t)),
    _ => { fault(concat_str("no integer semantics for ", op), 0); 0 }
  }
}

/* rcp / rsqrt .approx.ftz.f64 (ISA 9.7.3.14, 9.7.3.17): 1 / tmp, 1 / sqrt(tmp) of tmp = a[63:32] (1.11.20), the
 * result in 1.11.20 in d[63:32] and d[31:0] = 0; subnormal inputs and results flushed to sign-preserving zero, NaN ->
 * 0x7fffffff00000000.  The ISA does not name the rounding to 1.11.20: rounded once to nearest even here. */
val apx64 : (bool, int) -> int
function apx64(isr, a) = {
  let hi = pow2(64) - pow2(32);
  let r : fv = match decodeF(flushB(land_int(a, hi), 64), 64) {
    FV_NaN() => FV_NaN(),
    FV_Inf(s) => if isr & s then FV_NaN() else FV_Fin(s, 0, 0),
    FV_Fin(s, m, e) => {
      if m == 0 then FV_Inf(s)
      else if isr & s then FV_NaN()
      else if isr then {
        let j = if tmod_int(e, 2) == 0 then 0 else 1;
        let q = isqrt(tdiv_int(pow2(256), m * pow2(j)));
        fround(false, q, -128 - tdiv_int(e - j, 2), q * q * m * pow2(j) != pow2(256), 21, 11, RN)
      } else {
        let q = tdiv_int(pow2(128), m);
        fround(s, q, -128 - e, q * m != pow2(128), 21, 11, RN)
      }
    }
  };
  if fv_is_nan(r) then (pow2(31) - 1) * pow2(32) else land_int(flushB(encodeF(r, 64), 64), hi)
}

val un : (string, list(string), ty, int) -> int
function un(op, m, t, a) = {
  if isF(t) & fw(t) == 64 & has(m, "approx") & (op == "rcp" | op == "rsqrt") then apx64(op == "rsqrt", a)
  /* abs.f64: "NaN input is passed through unchanged" (ISA 9.7.3.9) */
  else if isF(t) & fw(t) == 64 & op == "abs" & isNaNB(a, 64) then a
  else if isF(t) & is_some(nanPay(fw(t), [|a|])) & (op == "sqrt" | op == "rcp") then
    match nanPay(fw(t), [|a|]) { Some(v) => v, None() => 0 }
  else if isF(t) & (op == "neg" | op == "abs" | op == "sqrt" | op == "rsqrt" | op == "rcp" | op == "sin" | op == "cos"
               | op == "lg2" | op == "ex2" | op == "tanh") then {
    let fm = fmods_of(m);
    let w = fw(t);
    match op {
      "neg" => lxor_int(ftzIn(fm, w, a), pow2(cw(w) - 1)),
      "abs" => land_int(ftzIn(fm, w, a), mask(cw(w) - 1)),
      "sqrt" => fpSqrt(fm, w, a),
      "rcp" => fpRcp(fm, w, a),
      _ => post(fm, w, transcendental(op, w, ftzIn(fm, w, a), a))
    }
  } else match op {
    "neg" => trunc(0 - a, wid(t)),
    "abs" => trunc(abs_int(ival(a, t)), wid(t)),
    "not" => trunc(lnot_int(a), wid(t)),
    "cnot" => b2i(a == 0),
    "popc" => popc_int(a),
    "clz" => clz(a, wid(t)),
    "brev" => brev(a, wid(t)),
    _ => { fault(concat_str("no integer semantics for ", op), 0); 0 }
  }
}

val src3Ty : (string, list(string), ty) -> ty
function src3Ty(op, m, t) = if op == "selp" then mkty("p", 1) else if op == "mad" & has(m, "wide") then wide(t) else t

/* the OOB-NaN a tensor copy writes in OOB-NaN fill mode (5.5; the ISA names it but gives no bits: the model's
 * pattern is the quiet NaN with an empty payload) for f16 / bf16; no other type has one */
val oobNaN : int -> int
function oobNaN(w) = match w { 16 => 32256, 116 => 32704, _ => -1 }

/* mad.f32 on .target sm_1x (9.7.3.5): "computes the product of a and b at double precision, and then the mantissa is
 * truncated to 23 bits, but the exponent is preserved ... when c = +/-0.0, mad.f32 is identical to the result computed
 * using separate mul and add instructions"; inputs and results flush (sm_1x single precision) */
val isaAny0 : (list(string), list(string)) -> bool
function isaAny0(names, alts) = match alts { [||] => false, a :: rest => has(names, a) | isaAny0(names, rest) }

val madSm1x : (fmods, int, int, int) -> int
function madSm1x(fm, a, b, c) = {
  let f = { fm with ftz = true };
  let cf = decodeF(ftzIn(f, 32, c), 32);
  if fv_is_zero(cf) then fpAdd(f, 32, fpMul(f, 32, a, b), c)
  else {
    let p : fv = match fv_mul_exact(decodeF(ftzIn(f, 32, a), 32), decodeF(ftzIn(f, 32, b), 32)) {
      FV_Fin(s, m, e) => { let k = max_int(0, bitlen(m) - 24); FV_Fin(s, asr_int(m, k), e + k) },
      v => v
    };
    fin(f, 32, fv_add_exact(p, cf, f.rm), false)
  }
}

val tern : (string, list(string), ty, int, int, int) -> int
function tern(op, m, t, a, b, c) = {
  if op == "selp" then (if c != 0 then a else b)
  /* fma.oob: "clamps the result to 0 if either of the operands is OOB NaN" (9.7.4.4) */
  else if has(m, "oob") & isF(t) & (a == oobNaN(fw(t)) | b == oobNaN(fw(t)) | c == oobNaN(fw(t))) then 0
  else if op == "mad" & isasm > 0 & isasm < 20 & isF(t) & fw(t) == 32 & not_bool(isaAny0(m, [|"rn", "rz", "rm", "rp"|])) then madSm1x(fmods_of(m), a, b, c)
  else if isF(t) & is_some(nanPay(fw(t), [|a, b, c|])) then match nanPay(fw(t), [|a, b, c|]) { Some(v) => v, None() => 0 }
  else if isF(t) then fpFma(fmods_of(m), fw(t), a, b, c)
  else if has(m, "wide") then trunc(ival(a, t) * ival(b, t) + c, 2 * wid(t))
  else if has(m, "hi") & has(m, "sat") then trunc(clamp(asr_int(ival(a, t) * ival(b, t), wid(t)) + ival(c, t), t), wid(t))
  else if has(m, "hi") then trunc(asr_int(ival(a, t) * ival(b, t), wid(t)) + c, wid(t))
  else trunc(ival(a, t) * ival(b, t) + c, wid(t))
}

/* min / max{.ftz}{.NaN}{.abs}.f32 d, a, b, c (ISA 9.7.3.11, 9.7.3.12): .abs on every input, then
 * d = two_inputs_min(two_inputs_min(a, b), c) */
val absIf : (bool, int, int) -> int
function absIf(b, w, x) = if b then land_int(x, mask(cw(w) - 1)) else x
val minmax3 : (string, list(string), ty, int, int, int) -> int
function minmax3(op, m, t, a, b, c) = {
  let w = fw(t);
  let fz = has(m, "ftz");
  let f : fmods = { fm_rn with ftz = fz };
  let x = absIf(has(m, "abs"), w, ftzIn(f, w, a));
  let y = absIf(has(m, "abs"), w, ftzIn(f, w, b));
  let z = absIf(has(m, "abs"), w, ftzIn(f, w, c));
  fpMinMax(fz, has(m, "NaN"), false, w, fpMinMax(fz, has(m, "NaN"), false, w, x, y, op == "min"), z, op == "min")
}

/* ---------------------------------------------------------------- comparison */
val cmpOp : list(string) -> string
function cmpOp(m) = pick(m, [|"eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu",
                              "gtu", "geu", "num", "nan"|], "eq")
val boolOp : list(string) -> string
function boolOp(m) = pick(m, [|"and", "or", "xor"|], "none")

val cmpOrd : (string, fv, fv) -> bool
function cmpOrd(op, x, y) = {
  let c = fv_cmp(x, y);
  match op { "eq" => c == 0, "ne" => c != 0, "lt" => c < 0, "le" => c <= 0, "gt" => c > 0, "ge" => c >= 0,
             _ => { fault(concat_str("no float comparison ", op), 0); false } }
}

val cmpF : (string, fv, fv) -> bool
function cmpF(op, x, y) = {
  let n = fv_is_nan(x) | fv_is_nan(y);
  if op == "num" then not_bool(n)
  else if op == "nan" then n
  else if n then string_length(op) == 3
  else cmpOrd(stake(op, 2), x, y)
}

val cmpI : (string, int, int, int, int) -> bool
function cmpI(op, x, y, rx, ry) = match op {
  "eq" => x == y, "ne" => x != y, "lt" => x < y, "le" => x <= y, "gt" => x > y, "ge" => x >= y,
  "lo" => rx < ry, "ls" => rx <= ry, "hi" => rx > ry, "hs" => rx >= ry,
  _ => { fault(concat_str("no integer comparison ", op), 0); false }
}

val cmp : (string, ty, int, int) -> int
function cmp(op, t, a, b) =
  if isF(t) then b2i(cmpF(op, fv(a, t), fv(b, t)))
  else b2i(cmpI(op, ival(a, t), ival(b, t), trunc(a, wid(t)), trunc(b, wid(t))))

val bop : (int, string, int) -> int
function bop(v, o, c) = match o { "and" => land_int(v, c), "or" => lor_int(v, c), "xor" => lxor_int(v, c), _ => v }

/* ---------------------------------------------------------------- conversion */
val intRnd : (list(string), string) -> string
function intRnd(m, d) = pick(m, [|"rni", "rzi", "rmi", "rpi"|], d)

/* .pzo: "if the result of the conversion produces -0.0, it will be forced to +0.0" (PTX ISA 9.4 9.7.10.24) */
val pzoBits : (list(string), int, int) -> int
function pzoBits(m, w, r) = if has(m, "pzo") & r == pow2(w - 1) then 0 else r

val cvt : (list(string), ty, ty, int) -> int
function cvt(m, d, s, v) = {
  let fm = fmods_of(m);
  /* .ftz applies only to .f32 inputs and results (PTX ISA, cvt: "Modifier .ftz can only be specified when either
   * .dtype or .atype is .f32 and applies only to single precision (.f32) inputs and results") */
  let fs = if fw(s) == 32 then fm else { fm with ftz = false };
  let fd = if fw(d) == 32 then fm else { fm with ftz = false };
  if isF(d) & isF(s) & intRnd(m, "") != "" then
    post(fd, fw(d), roundW(rndInt(fv(ftzIn(fs, fw(s), v), s), intRnd(m, "")), fw(d)))
  else if isF(d) & isF(s) then pzoBits(m, wid(d), fin(fd, fw(d), fv(ftzIn(fs, fw(s), v), s), false))
  else if isF(d) then fin(fd, fw(d), fv_of_int(ival(v, s)), false)
  else if isF(s) then f2i(fv(ftzIn(fs, fw(s), v), s), intRnd(m, "rzi"), d, fw(s) == 64)
  else trunc(if has(m, "sat") then clamp(ival(v, s), d) else ival(v, s), wid(d))
}

/* .rs (ISA 9.7.9.22, Floating Point Notes): the result is rounded toward zero, or away from zero when adding the
 * k random bits r to the discarded bits gives a carry out; r is aligned to the most significant discarded bits */
val fround_rs : (fv, int, int, int, int) -> fv
function fround_rs(v, p, eb, r, k) = match v {
  FV_Fin(s, m, e) => match fround(s, m, e, false, p, eb, RZ) {
    FV_Fin(_, q, lsb) => {
      let e0 = min_int(e, lsb - k);
      let t = shr_int(m * pow2(e - e0) - q * pow2(lsb - e0), lsb - k - e0);
      if m != 0 & t + r >= pow2(k) then fround(s, m, e, false, p, eb, if s then RM else RP) else FV_Fin(s, q, lsb)
    },
    z => z
  },
  _ => v
}

/* ---------------------------------------------------------------- atomics */
val atomOp : list(string) -> string
function atomOp(m) = pick(m, [|"add", "exch", "cas", "min", "max", "and", "or", "xor", "inc", "dec"|], "add")

val atomNew1 : (string, ty, int, int, int) -> int
function atomNew1(op, t, o, b, c) = match op {
  "exch" => b,
  "add.ftz" => bin("add", [|"ftz"|], t, o, b),
  "cas" => if o == b then c else o,
  "inc" => if o >= b then 0 else trunc(o + 1, wid(t)),
  "dec" => if o == 0 | o > b then b else trunc(o - 1, wid(t)),
  _ => bin(op, [||], t, o, b)
}

/* packed operands (PTX-EXEC-MEMQ, priority 40): lane-wise on the two 16-bit halves */
val atomNew : (string, ty, int, int, int) -> int
function atomNew(op, t, o, b, c) = {
  if t.k == "f16x2" | t.k == "bf16x2" | t.k == "s16x2" | t.k == "u16x2" then {
    let lt : ty = match t.k { "f16x2" => mkty("f", 16), "bf16x2" => mkty("bf", 16), "s16x2" => mkty("s", 16), _ => mkty("u", 16) };
    lor_int(atomNew1(op, lt, land_int(o, 65535), land_int(b, 65535), land_int(c, 65535)),
            shl_int(atomNew1(op, lt, land_int(asr_int(o, 16), 65535), land_int(asr_int(b, 16), 65535), land_int(asr_int(c, 16), 65535)), 16))
  } else atomNew1(op, t, o, b, c)
}

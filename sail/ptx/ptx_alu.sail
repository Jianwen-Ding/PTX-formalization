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

val bin : (string, list(string), ty, int, int) -> int
function bin(op, m, t, a, b) = {
  if isF(t) then {
    let fm = fmods_of(m);
    match op {
      "add" => fpAdd(fm, fw(t), a, b),
      "sub" => fpSub(fm, fw(t), a, b),
      "mul" => fpMul(fm, fw(t), a, b),
      "div" => fpDiv(fm, fw(t), a, b),
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

val un : (string, list(string), ty, int) -> int
function un(op, m, t, a) = {
  if isF(t) & (op == "neg" | op == "abs" | op == "sqrt" | op == "rsqrt" | op == "rcp" | op == "sin" | op == "cos"
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

val tern : (string, list(string), ty, int, int, int) -> int
function tern(op, m, t, a, b, c) = {
  if op == "selp" then (if c != 0 then a else b)
  else if isF(t) then fpFma(fmods_of(m), fw(t), a, b, c)
  else if has(m, "wide") then trunc(ival(a, t) * ival(b, t) + c, 2 * wid(t))
  else if has(m, "hi") & has(m, "sat") then trunc(clamp(asr_int(ival(a, t) * ival(b, t), wid(t)) + ival(c, t), t), wid(t))
  else if has(m, "hi") then trunc(asr_int(ival(a, t) * ival(b, t), wid(t)) + c, wid(t))
  else trunc(ival(a, t) * ival(b, t) + c, wid(t))
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

val cvt : (list(string), ty, ty, int) -> int
function cvt(m, d, s, v) = {
  let fm = fmods_of(m);
  /* .ftz applies only to .f32 inputs and results (PTX ISA, cvt: "Modifier .ftz can only be specified when either
   * .dtype or .atype is .f32 and applies only to single precision (.f32) inputs and results") */
  let fs = if fw(s) == 32 then fm else { fm with ftz = false };
  let fd = if fw(d) == 32 then fm else { fm with ftz = false };
  if isF(d) & isF(s) & intRnd(m, "") != "" then
    post(fd, fw(d), roundW(rndInt(fv(ftzIn(fs, fw(s), v), s), intRnd(m, "")), fw(d)))
  else if isF(d) & isF(s) then fin(fd, fw(d), fv(ftzIn(fs, fw(s), v), s), false)
  else if isF(d) then fin(fd, fw(d), fv_of_int(ival(v, s)), false)
  else if isF(s) then f2i(fv(ftzIn(fs, fw(s), v), s), intRnd(m, "rzi"), d)
  else trunc(if has(m, "sat") then clamp(ival(v, s), d) else ival(v, s), wid(d))
}

/* ---------------------------------------------------------------- atomics */
val atomOp : list(string) -> string
function atomOp(m) = pick(m, [|"add", "exch", "cas", "min", "max", "and", "or", "xor", "inc", "dec"|], "add")

val atomNew1 : (string, ty, int, int, int) -> int
function atomNew1(op, t, o, b, c) = match op {
  "exch" => b,
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

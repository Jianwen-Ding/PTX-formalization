/* =============================================================================
 * PTX-EXEC-INT (carry chains), PTX-EXEC-BITS (bit fields, prmt, lop3, shf), PTX-EXEC-INTX
 * (mul24, mad24, clmad, sad, dp4a, dp2a, packed 16x2 / 8x4), PTX-EXEC-PRED (set, slct, testp,
 * isspacep, ldu) and PTX-EXEC-MISC (asynchronous stores, clocks, alloca, ...).
 * ============================================================================= */

/* ---------------------------------------------------------------- carry chains */
let CCKEY : string = "%%cc.cf"

val madHalf : (list(string), ty, int, int) -> int
function madHalf(m, t, a, b) =
  if has(m, "hi") then trunc(asr_int(ival(a, t) * ival(b, t), wid(t)), wid(t)) else trunc(ival(a, t) * ival(b, t), wid(t))

/* ccOp: (result, carry out) */
val ccOp : (string, list(string), ty, int, int, int, int) -> (int, int)
function ccOp(op, m, t, cin, a, b, c) = {
  let w = wid(t);
  let s : int = match op {
    "add" => trunc(a, w) + trunc(b, w) + cin,
    "addc" => trunc(a, w) + trunc(b, w) + cin,
    "sub" => trunc(a, w) + trunc(lnot_int(b), w) + 1,
    "subc" => trunc(a, w) + trunc(lnot_int(b), w) + cin,
    _ => madHalf(m, t, a, b) + trunc(c, w) + cin
  };
  (trunc(s, w), asr_int(s, w))
}

val isCCOp : (string, list(string)) -> bool
function isCCOp(op, m) = ((op == "add" | op == "sub" | op == "mad") & has(m, "cc")) | op == "addc" | op == "subc" | op == "madc"

val gx_int : (opcode, list(xop)) -> bool
function gx_int(o, ops) = {
  let op = o.base;
  let m = o.mods;
  if ntys(o) != 1 | not_bool(isCCOp(op, m)) then false
  else {
    let t = ty1(o);
    let cin = if op == "addc" | op == "subc" | op == "madc" then rreg(CCKEY) else 0;
    match ops {
      [|d, a, b|] => {
        if op == "mad" | op == "madc" then false
        else {
          let (v, cf) = ccOp(op, m, t, cin, rd(a, t), rd(b, t), 0);
          wr(d, v); if has(m, "cc") then wreg(CCKEY, cf); true
        }
      },
      [|d, a, b, c|] => {
        if op == "mad" | op == "madc" then {
          let (v, cf) = ccOp(op, m, t, cin, rd(a, t), rd(b, t), rd(c, t));
          wr(d, v); if has(m, "cc") then wreg(CCKEY, cf); true
        } else false
      },
      _ => false
    }
  }
}

/* ---------------------------------------------------------------- bit fields */
val bitAt : (int, int) -> int
function bitAt(v, i) = if i >= 0 then land_int(asr_int(v, i), 1) else 0

val bfeLen : (ty, int, int) -> int
function bfeLen(t, p, l) = if p > wid(t) - 1 then 0 else min_int(l, wid(t) - p)

val bfe : (ty, int, int, int) -> int
function bfe(t, a, b, c) = {
  let p = land_int(b, 255);
  let l = land_int(c, 255);
  let fill = if isS(t) & l > 0 & bitAt(a, min_int(p + l - 1, wid(t) - 1)) == 1
             then land_int(mask(wid(t)), lnot_int(mask(bfeLen(t, p, l)))) else 0;
  land_int(lor_int(land_int(asr_int(a, p), mask(bfeLen(t, p, l))), fill), mask(wid(t)))
}

val fieldMask : (ty, int, int) -> int
function fieldMask(t, p, l) = if p > wid(t) - 1 then 0 else shl_int(mask(min_int(l, wid(t) - p)), p)

val bfi : (ty, int, int, int, int) -> int
function bfi(t, a, b, p0, l0) = {
  let p = land_int(p0, 255);
  let l = land_int(l0, 255);
  lor_int(land_int(b, lnot_int(fieldMask(t, p, l))), land_int(shl_int(a, p), fieldMask(t, p, l)))
}

val bfind : (ty, bool, int) -> int
function bfind(t, shift, a0) = {
  let a = if isS(t) & bitAt(a0, wid(t) - 1) == 1 then trunc(lnot_int(a0), wid(t)) else trunc(a0, wid(t));
  if a <= 0 then 4294967295 else if shift then wid(t) - 1 - log2_int(a) else log2_int(a)
}

val fns : (int, int, int) -> int
function fns(m, b, o) = {
  if b < 0 | b >= 32 then 4294967295
  else if o == 0 then (if bitAt(m, b) == 1 then b else 4294967295)
  else {
    var p : int = b;
    var n : int = abs_int(o) - 1;
    let stp = if o > 0 then 1 else -1;
    var r : int = -1;
    while r == -1 do {
      if p < 0 | p >= 32 then r = 4294967295
      else if bitAt(m, p) == 1 then { if n == 0 then r = p else { n = n - 1; p = p + stp } }
      else p = p + stp
    };
    r
  }
}

val bmsk : (bool, int, int) -> int
function bmsk(clampm, a, b) = {
  /* the Semantics pseudo-code of bmsk (PTX ISA 9.7.1) on the 32-bit operands */
  let a = land_int(a, 4294967295);
  let b = land_int(b, 4294967295);
  let a1 = land_int(a, 31);
  let b1 = land_int(b, 31);
  let mask0 = if clampm & a >= 32 then 0 else land_int(shl_int(4294967295, a1), 4294967295);
  let ovf = a1 + b1 >= 32 | (clampm & (a >= 32 | b >= 32));
  let mask1 = if ovf then 0 else if b1 == 0 then 4294967295 else land_int(shl_int(4294967295, a1 + b1), 4294967295);
  land_int(mask0, lnot_int(mask1))
}

val szext : (bool, ty, int, int) -> int
function szext(clampm, t, a, b) = {
  if clampm & b >= 32 then a
  else {
    let n = if clampm then b else land_int(b, 31);
    if n == 0 then 0
    else if n > 0 & n <= 32 then {
      if isS(t) & bitAt(a, n - 1) == 1 then trunc(lor_int(land_int(a, mask(n)), land_int(mask(32), lnot_int(mask(n)))), 32)
      else land_int(a, mask(n))
    } else a
  }
}

val prmtByte : (int, int) -> int
function prmtByte(v, s) =
  if land_int(s, 8) == 8 then (if bitAt(v, 8 * land_int(s, 7) + 7) == 1 then 255 else 0)
  else land_int(asr_int(v, 8 * land_int(s, 7)), 255)

val prmtSel : (string, int, int) -> int
function prmtSel(mode, c, i) = match mode {
  "f4e" => land_int(c, 3) + i,
  "b4e" => fmod_int(land_int(c, 3) - i, 8),
  "rc8" => land_int(c, 3),
  "ecl" => max_int(land_int(c, 3), i),
  "ecr" => min_int(land_int(c, 3), i),
  "rc16" => land_int(c, 1) * 2 + tmod_int(i, 2),
  _ => 0
}

val prmt : (string, int, int, int) -> int
function prmt(mode, a, b, c) = {
  let v = lor_int(a, shl_int(b, 32));
  var r : int = 0;
  foreach (i from 0 to 3) {
    let sel = if mode == "" then land_int(asr_int(c, 4 * i), 15) else prmtSel(mode, c, i);
    r = lor_int(r, shl_int(prmtByte(v, sel), 8 * i))
  };
  r
}

val lop3 : (int, int, int, int) -> int
function lop3(a, b, c, l) = {
  var acc : int = 0;
  foreach (i from 0 to 31) acc = lor_int(acc, shl_int(bitAt(l, bitAt(a, i) * 4 + bitAt(b, i) * 2 + bitAt(c, i)), i));
  acc
}

val bopB : (bool, list(string), int) -> int
function bopB(x, m, q) = if has(m, "and") then b2i(x & q != 0) else if has(m, "or") then b2i(x | q != 0) else b2i(x)

val shf : (bool, bool, int, int, int) -> int
function shf(left, clampm, a, b, c) = {
  let v = lor_int(a, shl_int(b, 32));
  let s = if clampm then min_int(c, 32) else land_int(c, 31);
  if left then land_int(asr_int(land_int(shl_int(v, s), 18446744073709551615), 32), 4294967295)
  else land_int(asr_int(v, s), 4294967295)
}

let B32 : ty = struct { k = "b", w = 32 }
let U32 : ty = struct { k = "u", w = 32 }
let S32 : ty = struct { k = "s", w = 32 }
let U64 : ty = struct { k = "u", w = 64 }
let PRED : ty = struct { k = "p", w = 1 }

val exec_bits : (opcode, list(xop)) -> bool
function exec_bits(o, ops) = {
  let m = o.mods;
  let t = ty1(o);
  match (o.base, ops) {
    ("bfe", [|d, a, b, c|]) => if ntys(o) == 1 then { wr(d, bfe(t, rd(a, t), rd(b, U32), rd(c, U32))); true } else false,
    ("bfi", [|d, a, b, p, l|]) => if ntys(o) == 1 then { wr(d, bfi(t, rd(a, t), rd(b, t), rd(p, U32), rd(l, U32))); true } else false,
    ("bfind", [|d, a|]) => if ntys(o) == 1 then { wr(d, bfind(t, has(m, "shiftamt"), rd(a, t))); true } else false,
    ("fns", [|d, mk, b, off|]) => { wr(d, fns(rd(mk, B32), rd(b, B32), ival(rd(off, S32), S32))); true },
    ("bmsk", [|d, a, b|]) => { wr(d, bmsk(has(m, "clamp"), rd(a, U32), rd(b, U32))); true },
    ("szext", [|d, a, b|]) => if ntys(o) == 1 then { wr(d, szext(has(m, "clamp"), t, rd(a, t), rd(b, U32))); true } else false,
    ("prmt", [|d, a, b, c|]) => {
      wr(d, prmt(pick(m, [|"f4e", "b4e", "rc8", "ecl", "ecr", "rc16"|], ""), rd(a, B32), rd(b, B32), rd(c, B32))); true
    },
    ("lop3", [|d, a, b, c, l|]) => match dual_regs(d) {
      None() => { wr(d, lop3(rd(a, B32), rd(b, B32), rd(c, B32), rd(l, B32))); true },
      Some(_) => false
    },
    /* d|p, where the ISA allows the sink _ for d when .BoolOp is given */
    ("lop3", [|dp, a, b, c, l, q|]) => match dual_regs(dp) {
      Some((dd, pp)) => {
        let v = lop3(rd(a, B32), rd(b, B32), rd(c, B32), rd(l, B32));
        let qv = rd(q, PRED);
        wr_sop(dd, v);
        wr_sop(pp, bopB(v != 0, m, qv));
        true
      },
      None() => false
    },
    ("shf", [|d, a, b, c|]) => { wr(d, shf(has(m, "l"), has(m, "clamp"), rd(a, B32), rd(b, B32), rd(c, U32))); true },
    _ => false
  }
}

/* ---------------------------------------------------------------- PTX-EXEC-INTX */
val v24 : (ty, int) -> int
function v24(t, a) = if isS(t) then sext(a, 24) else land_int(a, 16777215)
val mul24 : (list(string), ty, int, int) -> int
function mul24(m, t, a, b) = if has(m, "hi") then trunc(asr_int(v24(t, a) * v24(t, b), 16), 32) else trunc(v24(t, a) * v24(t, b), 32)
val mad24 : (list(string), ty, int, int, int) -> int
function mad24(m, t, a, b, c) =
  if has(m, "hi") & has(m, "sat") then trunc(clamp(asr_int(v24(t, a) * v24(t, b), 16) + ival(c, t), t), 32)
  else trunc(mul24(m, t, a, b) + c, 32)

val clmul : (int, int) -> int
function clmul(a, b) = {
  var r : int = 0;
  foreach (i from 0 to 63) { if bitAt(a, i) == 1 then r = lxor_int(r, shl_int(b, i)) };
  r
}

val field : (ty, int, int, int) -> int
function field(t, v, o, w) = if isS(t) then sext(asr_int(v, o), w) else land_int(asr_int(v, o), mask(w))

val laneTy : ty -> ty
function laneTy(t) = match t.k { "s16x2" => mkty("s", 16), "u16x2" => mkty("u", 16), "s8x4" => mkty("s", 8), "u8x4" => mkty("u", 8), _ => t }

val lane16 : (int, int) -> int
function lane16(v, i) = land_int(asr_int(v, 16 * i), 65535)
val pack16 : (int, int) -> int
function pack16(l0, l1) = lor_int(land_int(l0, 65535), shl_int(land_int(l1, 65535), 16))
val lane8 : (int, int) -> int
function lane8(v, i) = land_int(asr_int(v, 8 * i), 255)

val exec_intx : (opcode, list(xop)) -> bool
function exec_intx(o, ops) = {
  let m = o.mods;
  let t = ty1(o);
  match (o.base, ops) {
    ("mul24", [|d, a, b|]) => if ntys(o) == 1 then { wr(d, mul24(m, t, rd(a, t), rd(b, t))); true } else false,
    ("mad24", [|d, a, b, c|]) => if ntys(o) == 1 then { wr(d, mad24(m, t, rd(a, t), rd(b, t), rd(c, t))); true } else false,
    ("clmad", [|d, a, b, c|]) => {
      let p = clmul(rd(a, U64), rd(b, U64));
      wr(d, lxor_int(if has(m, "hi") then asr_int(p, 64) else land_int(p, 18446744073709551615), rd(c, U64)));
      true
    },
    ("sad", [|d, a, b, c|]) => if ntys(o) == 1 then { wr(d, trunc(rd(c, t) + abs_int(ival(rd(a, t), t) - ival(rd(b, t), t)), wid(t))); true } else false,
    ("dp4a", [|d, a, b, c|]) => {
      if ntys(o) != 2 then false
      else {
        let ta = ty_nth(o.tys, 0);
        let tb = ty_nth(o.tys, 1);
        let x = rd(a, ta);
        let y = rd(b, tb);
        var s : int = rd(c, U32);
        foreach (i from 0 to 3) s = s + field(ta, x, 8 * i, 8) * field(tb, y, 8 * i, 8);
        wr(d, trunc(s, 32));
        true
      }
    },
    ("dp2a", [|d, a, b, c|]) => {
      if ntys(o) != 2 then false
      else {
        let ta = ty_nth(o.tys, 0);
        let tb = ty_nth(o.tys, 1);
        let x = rd(a, ta);
        let y = rd(b, tb);
        let hi = has(m, "hi");
        let s = rd(c, U32) + field(ta, x, 0, 16) * field(tb, y, if hi then 16 else 0, 8) + field(ta, x, 16, 16) * field(tb, y, if hi then 24 else 8, 8);
        wr(d, trunc(s, 32));
        true
      }
    },
    _ => false
  }
}

val gx_intx : (opcode, list(xop)) -> bool
function gx_intx(o, ops) = {
  let op = o.base;
  let m = o.mods;
  if ntys(o) != 1 then false
  else {
    let t = ty1(o);
    let is16 = t.k == "s16x2" | t.k == "u16x2";
    let is8 = t.k == "s8x4" | t.k == "u8x4";
    let lt = laneTy(t);
    match ops {
      [|d, a, b|] => {
        if (is16 | is8) & (op == "add" | op == "sub" | op == "min" | op == "max") then {
          let x = rd(a, t);
          let y = rd(b, t);
          if is16 then wr(d, pack16(bin(op, m, lt, lane16(x, 0), lane16(y, 0)), bin(op, m, lt, lane16(x, 1), lane16(y, 1))))
          else {
            var r : int = 0;
            foreach (i from 0 to 3) r = lor_int(r, shl_int(land_int(bin(op, m, lt, lane8(x, i), lane8(y, i)), 255), 8 * i));
            wr(d, r)
          };
          true
        } else false
      },
      [|d, a|] => {
        if (is16 | is8) & (op == "neg" | op == "abs") then {
          let x = rd(a, t);
          if is16 then wr(d, pack16(un(op, m, lt, lane16(x, 0)), un(op, m, lt, lane16(x, 1))))
          else {
            var r : int = 0;
            foreach (i from 0 to 3) r = lor_int(r, shl_int(land_int(un(op, m, lt, lane8(x, i)), 255), 8 * i));
            wr(d, r)
          };
          true
        } else false
      },
      _ => false
    }
  }
}

/* ---------------------------------------------------------------- PTX-EXEC-PRED */
val ftzSrc : (list(string), ty, int) -> int
function ftzSrc(m, t, v) = if has(m, "ftz") & isF(t) & fw(t) != 64 then flushB(v, fw(t)) else v   /* .ftz: not .f64 */

val setRes : (bool, ty) -> int
function setRes(b, t) = if b then (if isF(t) then encodeF(fv_one(), fw(t)) else mask(wid(t))) else 0

val setLane : ty -> ty
function setLane(t) = match t.k { "u8x4" => mkty("u", 8), "s8x4" => mkty("s", 8), "u16x2" => mkty("u", 16), "s16x2" => mkty("s", 16), _ => t }
val isSetPk : ty -> bool
function isSetPk(t) = t.k == "u8x4" | t.k == "s8x4" | t.k == "u16x2" | t.k == "s16x2"

val nonNeg : (ty, int) -> bool
function nonNeg(t, v) = if isF(t) then { let f = fv(v, t); not_bool(fv_is_nan(f)) & fv_cmp(f, FV_Fin(false, 0, 0)) >= 0 } else ival(v, t) >= 0

val expField : (int, int) -> int
function expField(v, w) = land_int(asr_int(v, prec(w) - 1), mask(ebits(w)))
val manField : (int, int) -> int
function manField(v, w) = land_int(v, mask(prec(w) - 1))

val testp : (string, int, int) -> bool
function testp(op, v, w) = match op {
  "finite" => expField(v, w) != mask(ebits(w)),
  "infinite" => expField(v, w) == mask(ebits(w)) & manField(v, w) == 0,
  "number" => not_bool(expField(v, w) == mask(ebits(w)) & manField(v, w) != 0),
  "notanumber" => expField(v, w) == mask(ebits(w)) & manField(v, w) != 0,
  "normal" => expField(v, w) != 0 & expField(v, w) != mask(ebits(w)),
  "subnormal" => expField(v, w) == 0 & manField(v, w) != 0,
  _ => false
}

val inSpace : (list(string), int, int) -> bool
function inSpace(m, a, own) =
  ((has(m, "global") | has(m, "const") | has(m, "param")) & a < SHBASE)
  | ((has(m, "shared") | has(m, "shared::cta")) & ((a >= SHBASE & a < LOCBASE) | (a >= CLBASE & asr_int(a - CLBASE, 24) == own)))
  | (has(m, "shared::cluster") & ((a >= SHBASE & a < LOCBASE) | a >= CLBASE))
  | (has(m, "local") & a >= LOCBASE & a < CLBASE)

val exec_pred : (opcode, list(xop)) -> bool
function exec_pred(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    ("set", [|d, a, b|]) => {
      if ntys(o) == 2 then {
        let dt = ty_nth(o.tys, 0);
        let st = ty_nth(o.tys, 1);
        wr(d, setRes(cmp(cmpOp(m), st, ftzSrc(m, st, rd(a, st)), ftzSrc(m, st, rd(b, st))) != 0, dt)); true
      } else if ntys(o) == 1 & isSetPk(ty1(o)) then {
        let lt = setLane(ty1(o));
        let x = rd(a, B32);
        let y = rd(b, B32);
        var r : int = 0;
        foreach (i from 0 to (tdiv_int(32, wid(lt)) - 1)) {
          if cmp(cmpOp(m), lt, land_int(asr_int(x, wid(lt) * i), mask(wid(lt))), land_int(asr_int(y, wid(lt) * i), mask(wid(lt)))) != 0 then
            r = lor_int(r, shl_int(mask(wid(lt)), wid(lt) * i))
        };
        wr(d, r);
        true
      } else false
    },
    ("set", [|d, a, b, c|]) => {
      if ntys(o) == 2 then {
        let dt = ty_nth(o.tys, 0);
        let st = ty_nth(o.tys, 1);
        wr(d, setRes(bop(cmp(cmpOp(m), st, ftzSrc(m, st, rd(a, st)), ftzSrc(m, st, rd(b, st))), boolOp(m), rd(c, PRED)) != 0, dt)); true
      } else false
    },
    ("slct", [|d, a, b, c|]) => {
      if ntys(o) == 2 then {
        let dt = ty_nth(o.tys, 0);
        let st = ty_nth(o.tys, 1);
        wr(d, if nonNeg(st, ftzSrc(m, st, rd(c, st))) then rd(a, dt) else rd(b, dt)); true
      } else false
    },
    ("testp", [|d, a|]) => {
      let t = ty1(o);
      wr(d, b2i(testp(pick(m, [|"finite", "infinite", "number", "notanumber", "normal", "subnormal"|], ""), rd(a, t), wid(t))));
      true
    },
    ("isspacep", [|d, a|]) => { wr(d, b2i(inSpace(m, rd(a, U64), rreg("%cluster_ctarank")))); true },
    ("ldu", _) => exec_mem({ o with base = "ld" }, ops),
    _ => false
  }
}

/* ---------------------------------------------------------------- PTX-EXEC-MISC */
val gx_misc : (opcode, list(xop)) -> bool
function gx_misc(o, ops) = o.base == "griddepcontrol" | o.base == "setmaxnreg"

val isClockReg : string -> bool
function isClockReg(s) = has([|"%clock", "%clock64", "%globaltimer", "%globaltimer_lo", "%globaltimer_hi"|], s)
val isVecSreg : string -> bool
function isVecSreg(s) = has([|"%tid", "%ntid", "%ctaid", "%nctaid", "%clusterid", "%nclusterid", "%cluster_ctaid", "%cluster_nctaid"|], s)

val exec_misc_prio : (opcode, list(xop)) -> bool
function exec_misc_prio(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    ("mov", [|d, XS(SR(s))|]) => {
      if ntys(o) == 1 & isClockReg(s) & not_bool(isVec(d)) then {
        wr(d, trunc(if s == "%globaltimer_hi" then asr_int(tick, 32) else tick, wid(ty1(o))));
        tick = tick + 25;
        true
      } else if ntys(o) == 1 & isVecSreg(s) & (has(m, "v4") | has(m, "v2")) then match d {
        XV(ds) => {
          var i : int = 0;
          var x = ds;
          while is_cons(x) do {
            match x {
              dd :: rest => {
                let sfx = if i == 0 then ".x" else if i == 1 then ".y" else ".z";
                wr_sop(dd, if i < 3 then rreg(concat_str(s, sfx)) else 0);
                i = i + 1; x = rest
              },
              [||] => ()
            }
          };
          true
        },
        _ => false
      } else false
    },
    ("activemask", [|_|]) => if its then exec_warp(o, ops) else false,
    _ => false
  }
}

val allocaAlign : list(xop) -> int
function allocaAlign(l) = match l { x :: _ => max_int(8, rd(x, U32)), [||] => 8 }

val exec_misc : (opcode, list(xop)) -> bool
function exec_misc(o, ops) = {
  let t = tchk(cur);
  match (o.base, ops) {
    ("alloca", d :: sz :: rest) => {
      let ty = ty1(o);
      let al = allocaAlign(rest);
      let a = alignUp(t_ltop[t], al);
      wr(d, trunc(a + (if wid(ty) == 64 then LOCBASE else 0), wid(ty)));
      t_ltop[t] = a + rd(sz, U32);
      true
    },
    ("stacksave", [|d|]) => { let ty = ty1(o); wr(d, trunc(t_ltop[t] + (if wid(ty) == 64 then LOCBASE else 0), wid(ty))); true },
    ("stackrestore", [|a|]) => { let ty = ty1(o); t_ltop[t] = rd(a, ty) - (if wid(ty) == 64 then LOCBASE else 0); true },
    _ => false
  }
}

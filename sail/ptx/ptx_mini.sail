/* =============================================================================
 * PTX-EXEC-MINI (semantics/ptx-exec-mini.k): conversions to and from the 8-, 6- and 4-bit float
 * formats (e4m3 e5m2 e2m3 e3m2 e2m1 ue5m3), s2f6 and the e8m0 scale format.
 * ============================================================================= */

val isMini : string -> bool
function isMini(k) = has([|"e4m3", "e5m2", "e2m3", "e3m2", "e2m1", "ue5m3"|], k)
val mE : string -> int
function mE(k) = match k { "e4m3" => 4, "e5m2" => 5, "e2m3" => 2, "e3m2" => 3, "e2m1" => 2, "ue5m3" => 5, _ => 0 }
val mM : string -> int
function mM(k) = match k { "e4m3" => 3, "e5m2" => 2, "e2m3" => 3, "e3m2" => 2, "e2m1" => 1, "ue5m3" => 3, _ => 0 }
val mTop : string -> int
function mTop(k) = match k { "e4m3" => 126, "e5m2" => 123, "e2m3" => 31, "e3m2" => 31, "e2m1" => 7, "ue5m3" => 254, _ => 0 }
val mBias : string -> int
function mBias(k) = pow2(mE(k) - 1) - 1
val mBits : string -> int
function mBits(k) = if k == "ue5m3" then 8 else 1 + mE(k) + mM(k)

val miniNaN : (string, int) -> bool
function miniNaN(k, c) = match k { "e4m3" => c == 127, "e5m2" => c > 124, "ue5m3" => c == 255, _ => false }
val miniInf : (string, int) -> bool
function miniInf(k, c) = k == "e5m2" & c == 124

/* F * 2^n, exact */
val scaleP2 : (fv, int) -> fv
function scaleP2(f, n) = match f { FV_Fin(s, m, e) => FV_Fin(s, m, e + n), _ => f }

val miniMag : (string, int) -> fv
function miniMag(k, c) = {
  let hi = asr_int(c, mM(k));
  if hi == 0 then FV_Fin(false, land_int(c, mask(mM(k))), 1 - mBias(k) - mM(k))
  else FV_Fin(false, land_int(c, mask(mM(k))) + pow2(mM(k)), hi - mBias(k) - mM(k))
}

val signedF : (int, fv) -> fv
function signedF(s, f) = if s == 0 then f else fv_neg(f)

val miniToF16 : (string, int) -> int
function miniToF16(k, b) = {
  let c = land_int(b, mask(mBits(k) - 1));
  if miniNaN(k, c) then 32767
  else if miniInf(k, c) then lor_int(31744, shl_int(asr_int(b, mBits(k) - 1), 15))
  else encodeF(signedF(asr_int(b, mBits(k) - 1), miniMag(k, c)), 16)
}

/* floor / ceil / nearest-even integer of a nonnegative value */
val roundGrid : (fv, int) -> int
function roundGrid(f, rdm) = match f {
  FV_Fin(_, m, e) => {
    if e >= 0 then m * pow2(e)
    else {
      let d = pow2(negate(e));
      let q = tdiv_int(m, d);
      let r = m - q * d;
      if rdm == 1 then q
      else if rdm == 2 then (if r > 0 then q + 1 else q)
      else if 2 * r < d then q else if 2 * r > d then q + 1 else if fmod_int(q, 2) == 0 then q else q + 1
    }
  },
  _ => 0
}

val miniCode : (string, int, int) -> int
function miniCode(k, e, n) = {
  let raw = if n >= pow2(mM(k) + 1) then shl_int(e + 1 + mBias(k), mM(k))
            else if n >= pow2(mM(k)) then shl_int(e + mBias(k), mM(k)) + (n - pow2(mM(k)))
            else n;
  min_int(mTop(k), raw)
}

/* magnitude code of |x| with rounding 0 nearest-even, 1 toward zero, 2 up; saturating */
val miniEncR : (string, fv, int) -> int
function miniEncR(k, a, rdm) = {
  if fv_is_zero(a) then 0
  else {
    var e : int = 1 - mBias(k);
    while fv_cmp(a, FV_Fin(false, 1, e + 1)) >= 0 & e < 40 do e = e + 1;
    miniCode(k, e, roundGrid(scaleP2(a, mM(k) - e), rdm))
  }
}

val miniSign : (string, fv) -> int
function miniSign(k, x) = if fv_sign(x) & k != "ue5m3" then pow2(mBits(k) - 1) else 0

val miniFromR : (string, fv, bool, int) -> int
function miniFromR(k, x, rl, rdm) = {
  if fv_is_nan(x) then (if k == "e4m3" | k == "e5m2" then 127 else if k == "ue5m3" then 255 else mTop(k))
  else if (rl | k == "ue5m3") & fv_sign(x) then 0
  else if fv_is_inf(x) then lor_int(miniSign(k, x), mTop(k))
  else lor_int(miniSign(k, x), miniEncR(k, fv_abs(x), rdm))
}

val isMiniX2 : ty -> bool
function isMiniX2(t) = has([|"e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2", "ue5m3x2"|], t.k)
val miniKind : ty -> string
function miniKind(t) = if t.k == "ue5m3x2" then "ue5m3" else stake(t.k, 4)
val miniElem : (ty, int, int) -> int
function miniElem(t, v, i) = if mBits(miniKind(t)) <= 4 then land_int(asr_int(v, 4 * i), 15) else land_int(asr_int(v, 8 * i), 255)
val miniPack : (ty, int, int) -> int
function miniPack(t, h, l) = if mBits(miniKind(t)) <= 4 then lor_int(shl_int(h, 4), l) else lor_int(shl_int(h, 8), l)

val nanF : unit -> fv
function nanF() = FV_NaN()
val scDiv : (fv, int) -> fv
function scDiv(x, s) = if s == 255 then FV_NaN() else scaleP2(x, 127 - s)
val scMul : (fv, int) -> fv
function scMul(x, s) = if s == 255 then FV_NaN() else scaleP2(x, s - 127)
val flushIn : (bool, fv, int) -> fv
function flushIn(b, x, mn) =
  if b & not_bool(fv_is_nan(x)) & not_bool(fv_is_inf(x)) & fv_cmp(fv_abs(x), FV_Fin(false, 1, mn)) < 0 then FV_Fin(false, 0, 0) else x
val mrnd : list(string) -> int
function mrnd(m) = if has(m, "rz") then 1 else if has(m, "rp") then 2 else 0
val pzoFix : (bool, string, int) -> int
function pzoFix(p, k, c) = if p & k != "ue5m3" & c == pow2(mBits(k) - 1) then 0 else c
val miniCvt : (list(string), string, fv, int) -> int
function miniCvt(m, k, x, s) = pzoFix(has(m, "pzo"), k, miniFromR(k, scDiv(x, s), has(m, "relu"), mrnd(m)))
val scHi : (list(string), int) -> int
function scHi(m, s) = if has(m, "scaled::n2::ue8m0") then land_int(asr_int(s, 8), 255) else land_int(s, 255)
val scLo : (list(string), int) -> int
function scLo(m, s) = land_int(s, 255)
val isN1 : list(string) -> bool
function isN1(m) = has(m, "scaled::n1::ue8m0")

val miniDecF : (string, int) -> fv
function miniDecF(k, b) = {
  let c = land_int(b, mask(mBits(k) - (if k == "ue5m3" then 0 else 1)));
  if miniNaN(k, c) then FV_NaN()
  else if miniInf(k, land_int(b, mask(mBits(k) - 1))) then signedF(asr_int(b, mBits(k) - 1), FV_Inf(false))
  else signedF(if k == "ue5m3" then 0 else asr_int(b, mBits(k) - 1), miniMag(k, land_int(b, mask(if k == "ue5m3" then 8 else mBits(k) - 1))))
}

val miniTo16 : (string, int, int) -> int
function miniTo16(k, b, f) = if f == 16 then miniToF16(k, b) else encodeF(decodeF(miniToF16(k, b), 16), f)

/* s2f6 */
val isS2F6 : ty -> bool
function isS2F6(t) = t.k == "s2f6x2"
val s2f6Code : (list(string), fv) -> int
function s2f6Code(m, x) = {
  if fv_is_nan(x) then 127
  else if has(m, "relu") & fv_sign(x) then 0
  else if fv_is_inf(x) then (if fv_sign(x) then 128 else 127)
  else if fv_sign(x) then land_int(0 - min_int(128, roundGrid(scaleP2(fv_abs(x), 6), mrnd(m))), 255)
  else min_int(127, roundGrid(scaleP2(x, 6), mrnd(m)))
}
val s2f6Enc : (list(string), fv, int) -> int
function s2f6Enc(m, x, s) = s2f6Code(m, scDiv(x, s))
val s2f6Val : int -> fv
function s2f6Val(b) = scaleP2(fv_of_int(sext(land_int(b, 255), 8)), -6)

/* ue8m0 */
val isUe : ty -> bool
function isUe(t) = t.k == "ue8m0x2"
val ueEnc : (fv, bool) -> int
function ueEnc(x, up) = {
  if fv_is_nan(x) then 255
  else if fv_sign(x) | fv_is_zero(x) then 0
  else if fv_is_inf(x) then 254
  else {
    var e : int = -160;
    var r : int = -1000;
    while r == -1000 do {
      if e >= 130 then r = e
      else if fv_cmp(x, FV_Fin(false, 1, e + 1)) < 0 then r = (if up & fv_cmp(x, FV_Fin(false, 1, e)) > 0 then e + 1 else e)
      else e = e + 1
    };
    max_int(0, min_int(254, r + 127))
  }
}
val ueDec : int -> int
function ueDec(c) = if c == 255 then 32767 else if c == 0 then 64 else shl_int(c, 7)

let B16 : ty = struct { k = "b", w = 16 }

val exec_mini : (opcode, list(xop)) -> bool
function exec_mini(o, ops) = {
  let m = o.mods;
  if o.base != "cvt" | ntys(o) != 2 then false
  else {
    let dt = ty_nth(o.tys, 0);
    let st = ty_nth(o.tys, 1);
    match ops {
      [|d, a, b|] => {
        if isMiniX2(dt) & isF(st) then {
          wr(d, miniPack(dt, miniCvt(m, miniKind(dt), flushIn(isN1(m), fv(rd(a, st), st), -126), 127),
                             miniCvt(m, miniKind(dt), flushIn(isN1(m), fv(rd(b, st), st), -126), 127))); true
        } else if isS2F6(dt) & isF(st) then {
          wr(d, lor_int(shl_int(s2f6Enc(m, fv(rd(a, st), st), 127), 8), s2f6Enc(m, fv(rd(b, st), st), 127))); true
        } else if isUe(dt) & isF(st) then {
          wr(d, lor_int(shl_int(ueEnc(fv(rd(a, st), st), has(m, "rp")), 8), ueEnc(fv(rd(b, st), st), has(m, "rp")))); true
        } else if isMiniX2(dt) & isHalfPair(st) then {
          let s = rd(b, B16);
          let v = rd(a, st);
          wr(d, miniPack(dt, miniCvt(m, miniKind(dt), flushIn(isN1(m), decodeF(lane16(v, 1), fw(laneTyH(st))), -14), scHi(m, s)),
                             miniCvt(m, miniKind(dt), flushIn(isN1(m), decodeF(lane16(v, 0), fw(laneTyH(st))), -14), scLo(m, s)))); true
        } else if isHalfPair(dt) & isMiniX2(st) then {
          let s = rd(b, B16);
          let v = rd(a, st);
          wr(d, pack16(encodeF(scMul(miniDecF(miniKind(st), miniElem(st, v, 0)), scLo(m, s)), fw(laneTyH(dt))),
                       encodeF(scMul(miniDecF(miniKind(st), miniElem(st, v, 1)), scHi(m, s)), fw(laneTyH(dt))))); true
        } else if isS2F6(dt) & isHalfPair(st) then {
          let s = rd(b, B16);
          let v = rd(a, st);
          wr(d, lor_int(shl_int(s2f6Enc(m, decodeF(lane16(v, 1), fw(laneTyH(st))), scHi(m, s)), 8), s2f6Enc(m, decodeF(lane16(v, 0), fw(laneTyH(st))), scLo(m, s)))); true
        } else if isHalfPair(dt) & isS2F6(st) then {
          let s = rd(b, B16);
          let v = rd(a, st);
          wr(d, pack16(encodeF(scMul(s2f6Val(v), scLo(m, s)), fw(laneTyH(dt))), encodeF(scMul(s2f6Val(asr_int(v, 8)), scHi(m, s)), fw(laneTyH(dt))))); true
        } else false
      },
      [|d, a, b, sc|] => {
        if isMiniX2(dt) & isF(st) then {
          let s = rd(sc, B16);
          wr(d, miniPack(dt, miniCvt(m, miniKind(dt), flushIn(isN1(m), fv(rd(a, st), st), -126), scHi(m, s)),
                             miniCvt(m, miniKind(dt), flushIn(isN1(m), fv(rd(b, st), st), -126), scLo(m, s)))); true
        } else if isS2F6(dt) & isF(st) then {
          let s = rd(sc, B16);
          wr(d, lor_int(shl_int(s2f6Enc(m, fv(rd(a, st), st), scHi(m, s)), 8), s2f6Enc(m, fv(rd(b, st), st), scLo(m, s)))); true
        } else false
      },
      [|d, a|] => {
        if isHalfPair(dt) & ty_eq(st, mkty("ue5m3x2", 16)) then {
          let v = rd(a, st);
          wr(d, pack16(encodeF(miniDecF("ue5m3", miniElem(st, v, 0)), fw(laneTyH(dt))), encodeF(miniDecF("ue5m3", miniElem(st, v, 1)), fw(laneTyH(dt))))); true
        } else if isMiniX2(dt) & isHalfPair(st) then {
          let v = rd(a, st);
          wr(d, miniPack(dt, miniCvt(m, miniKind(dt), flushIn(isN1(m), decodeF(lane16(v, 1), fw(laneTyH(st))), -14), 127),
                             miniCvt(m, miniKind(dt), flushIn(isN1(m), decodeF(lane16(v, 0), fw(laneTyH(st))), -14), 127))); true
        } else if isS2F6(dt) & isHalfPair(st) then {
          let v = rd(a, st);
          wr(d, lor_int(shl_int(s2f6Enc(m, decodeF(lane16(v, 1), fw(laneTyH(st))), 127), 8), s2f6Enc(m, decodeF(lane16(v, 0), fw(laneTyH(st))), 127))); true
        } else if isHalfPair(dt) & isS2F6(st) then {
          let v = rd(a, st);
          wr(d, pack16(encodeF(s2f6Val(v), fw(laneTyH(dt))), encodeF(s2f6Val(asr_int(v, 8)), fw(laneTyH(dt))))); true
        } else if isHalfPair(dt) & isMiniX2(st) then {
          let v = rd(a, st);
          wr(d, pack16(miniTo16(miniKind(st), miniElem(st, v, 0), fw(laneTyH(dt))), miniTo16(miniKind(st), miniElem(st, v, 1), fw(laneTyH(dt))))); true
        } else if isUe(dt) & isHalfPair(st) then {
          let v = rd(a, st);
          wr(d, lor_int(shl_int(ueEnc(decodeF(lane16(v, 1), 116), has(m, "rp")), 8), ueEnc(decodeF(lane16(v, 0), 116), has(m, "rp")))); true
        } else if isHalfPair(dt) & isUe(st) then {
          let v = rd(a, st);
          wr(d, pack16(ueDec(land_int(v, 255)), ueDec(land_int(asr_int(v, 8), 255)))); true
        } else false
      },
      _ => false
    }
  }
}

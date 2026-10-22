/* =============================================================================
 * PTX-EXEC-MMA (semantics/ptx-exec-mma.k): warp-level matrix multiply-accumulate: mma.sync,
 * mma.sync with block scaling, wmma.mma / wmma.load / wmma.store.
 *
 * The warp's fragments are gathered into matrices through the fragment layouts, the products and
 * sums are exact and rounded once into the accumulator format (the ISA leaves accumulation order and
 * rounding unspecified for these types), except .f64, which is a chain of f64 fused multiply-adds
 * (9.7.15.5.14), and the D fragments are scattered back.  Matrices are sparse maps (C runtime string maps keyed by
 * the position r * 1024 + c, absent = 0, as `orDefault 0` in K).
 * ============================================================================= */

let SP_MA : int = 20
let SP_MB : int = 21
let SP_MC : int = 22
let SP_MD : int = 23
let SP_SA : int = 24
let SP_SB : int = 25

val mget : (int, int) -> int
function mget(sp, p) = smap_get(sp, dec_str(p))
val mset : (int, int, int) -> unit
function mset(sp, p, v) = smap_set(sp, dec_str(p), v)

val rc : (int, int) -> int
function rc(r, c) = r * 1024 + c

/* ---------------------------------------------------------------- shapes */
val shM : string -> int
val shN : string -> int
val shK : string -> int
val str_index : (string, string) -> int
function str_index(s, c) = {
  var i : int = 0;
  var r : int = -1;
  while r < 0 & i < string_length(s) do { if stake(sdrop(s, i), 1) == c then r = i; i = i + 1 };
  r
}
val str_int : string -> int
function str_int(s) = str2base(s, 10)
function shM(s) = str_int(stake(sdrop(s, 1), str_index(s, "n") - 1))
function shN(s) = str_int(stake(sdrop(s, str_index(s, "n") + 1), str_index(s, "k") - str_index(s, "n") - 1))
function shK(s) = str_int(sdrop(s, str_index(s, "k") + 1))

val shapeOf : list(string) -> string
function shapeOf(m) = pick(m, [|"m8n8k4", "m8n8k16", "m8n8k32", "m8n8k128", "m16n8k4", "m16n8k8", "m16n8k16", "m16n8k32",
                               "m16n8k64", "m16n8k128", "m16n8k256", "m16n16k16", "m8n32k16", "m32n8k16", "m16n16k8"|], "")

/* ---------------------------------------------------------------- element decode */
val isFloatElem : ty -> bool
function isFloatElem(t) = has([|"f", "bf", "tf", "e4m3", "e5m2", "e2m3", "e3m2", "e2m1", "e2m1p"|], t.k)

val elemF : (ty, int) -> fv
function elemF(t, b) = match t.k {
  "f" => decodeF(b, t.w),
  "bf" => decodeF(b, 116),
  "tf" => decodeF(land_int(b, lnot_int(8191)), 32),
  "e4m3" => decodeF(miniToF16("e4m3", b), 16),
  "e5m2" => decodeF(miniToF16("e5m2", b), 16),
  "e2m3" => decodeF(miniToF16("e2m3", land_int(b, 63)), 16),
  "e3m2" => decodeF(miniToF16("e3m2", land_int(b, 63)), 16),
  "e2m1" => decodeF(miniToF16("e2m1", land_int(asr_int(b, 2), 15)), 16),
  "e2m1p" => decodeF(miniToF16("e2m1", land_int(b, 15)), 16),
  _ => FV_Fin(false, 0, 0)
}

val elemI : (ty, int) -> int
function elemI(t, b) = ival(b, t)

/* element width of the A/B operands in bits (fp6 / fp4 in 8-bit containers) */
val ew : ty -> int
function ew(t) = if has([|"e4m3", "e5m2", "e2m3", "e3m2", "e2m1"|], t.k) then 8 else t.w

/* ---------------------------------------------------------------- fragment layouts */
val posA : (string, int, int, int) -> int
function posA(s, b, l, i) = {
  let g = tdiv_int(l, 4);
  let q = fmod_int(l, 4);
  match b {
    16 => if s == "m16n8k16" then rc(g + 8 * fmod_int(tdiv_int(i, 2), 2), 2 * q + fmod_int(i, 2) + 8 * tdiv_int(i, 4))
          else if s == "m16n8k8" then rc(g + 8 * tdiv_int(i, 2), 2 * q + fmod_int(i, 2)) else -1,
    32 => if s == "m16n8k8" then rc(g + 8 * fmod_int(i, 2), q + 4 * tdiv_int(i, 2))
          else if s == "m16n8k4" then rc(g + 8 * i, q) else -1,
    64 => if s == "m8n8k4" then rc(g, q)
          else if s == "m16n8k4" then rc(g + 8 * i, q)
          else if s == "m16n8k8" | s == "m16n8k16" then rc(g + 8 * fmod_int(i, 2), q + 4 * tdiv_int(i, 2)) else -1,
    8 => if s == "m8n8k16" then rc(g, 4 * q + i)
         else if s == "m16n8k16" then rc(g + 8 * tdiv_int(i, 4), 4 * q + fmod_int(i, 4))
         else if s == "m16n8k32" then rc(g + 8 * fmod_int(tdiv_int(i, 4), 2), 4 * q + fmod_int(i, 4) + 16 * tdiv_int(i, 8)) else -1,
    4 => if s == "m8n8k32" then rc(g, 8 * q + i)
         else if s == "m16n8k32" then rc(g + 8 * tdiv_int(i, 8), 8 * q + fmod_int(i, 8))
         else if s == "m16n8k64" then rc(g + 8 * fmod_int(tdiv_int(i, 8), 2), 8 * q + fmod_int(i, 8) + 32 * tdiv_int(i, 16)) else -1,
    1 => if s == "m8n8k128" then rc(g, 32 * q + i)
         else if s == "m16n8k128" then rc(g + 8 * tdiv_int(i, 32), 32 * q + fmod_int(i, 32))
         else if s == "m16n8k256" then rc(g + 8 * fmod_int(tdiv_int(i, 32), 2), 32 * q + fmod_int(i, 32) + 128 * tdiv_int(i, 64)) else -1,
    _ => -1
  }
}

val posB : (string, int, int, int) -> int
function posB(s, b, l, i) = {
  let g = tdiv_int(l, 4);
  let q = fmod_int(l, 4);
  match b {
    16 => if s == "m16n8k16" | s == "m16n8k32" then rc(2 * q + fmod_int(i, 2) + 8 * tdiv_int(i, 2), g)
          else if s == "m16n8k8" then rc(2 * q + i, g) else -1,
    32 => if s == "m16n8k8" | s == "m16n8k4" | s == "m16n8k16" then rc(q + 4 * i, g) else -1,
    64 => if s == "m8n8k4" | s == "m16n8k4" | s == "m16n8k8" | s == "m16n8k16" then rc(q + 4 * i, g) else -1,
    8 => if s == "m8n8k16" | s == "m16n8k16" then rc(4 * q + i, g)
         else if s == "m16n8k32" | s == "m16n8k64" then rc(4 * q + fmod_int(i, 4) + 16 * tdiv_int(i, 4), g) else -1,
    4 => if s == "m8n8k32" | s == "m16n8k32" then rc(8 * q + i, g)
         else if s == "m16n8k64" | s == "m16n8k128" then rc(8 * q + fmod_int(i, 8) + 32 * tdiv_int(i, 8), g) else -1,
    1 => if s == "m8n8k128" | s == "m16n8k128" then rc(32 * q + i, g)
         else if s == "m16n8k256" then rc(32 * q + fmod_int(i, 32) + 128 * tdiv_int(i, 32), g) else -1,
    _ => -1
  }
}

val posC : (string, int, int) -> int
function posC(s, l, i) = {
  if shM(s) == 8 then rc(tdiv_int(l, 4), 2 * fmod_int(l, 4) + i)
  else rc(tdiv_int(l, 4) + 8 * tdiv_int(i, 2), 2 * fmod_int(l, 4) + fmod_int(i, 2))
}

/* ---------------------------------------------------------------- fragments <-> matrices */
val regBits : ty -> int
function regBits(t) = if t.k == "f" & t.w == 64 then 64 else 32
val perReg : ty -> int
function perReg(t) = tdiv_int(regBits(t), ew(t))
val nElA : string -> int
function nElA(s) = tdiv_int(shM(s) * shK(s), 32)
val nElB : string -> int
function nElB(s) = tdiv_int(shK(s) * shN(s), 32)
val nElC : string -> int
function nElC(s) = tdiv_int(shM(s) * shN(s), 32)
val regsOf : (int, ty) -> int
function regsOf(n, t) = tdiv_int(n + perReg(t) - 1, perReg(t))

/* element I of a lane's register list (fragment at register offset Off) */
val fragBits : (list(int), int, ty, int) -> int
function fragBits(rs, off, t, i) =
  land_int(asr_int(at(rs, off + tdiv_int(i, perReg(t))), ew(t) * fmod_int(i, perReg(t))), mask(ew(t)))

/* fillAll(which (0 = A, 1 = B), shape, type, arrivals, count, register offset) into space sp */
val fillAll : (int, int, string, ty, list((int, list(int))), int, int) -> unit
function fillAll(sp, wh, s, t, arr, n, off) = {
  smap_clear(sp);
  foreach (l from 0 to 31) {
    match lane_get(arr, l) {
      Some(rs) => foreach (i from 0 to (n - 1)) mset(sp, if wh == 0 then posA(s, ew(t), l, i) else posB(s, ew(t), l, i), fragBits(rs, off, t, i)),
      None() => ()
    }
  }
}

val fillC : (int, string, ty, list((int, list(int))), int) -> unit
function fillC(sp, s, t, arr, off) = {
  smap_clear(sp);
  foreach (l from 0 to 31) {
    match lane_get(arr, l) {
      Some(rs) => foreach (i from 0 to (nElC(s) - 1)) mset(sp, posC(s, l, i), fragBits(rs, off, t, i)),
      None() => ()
    }
  }
}

/* ---------------------------------------------------------------- the product */
val fv_zero : unit -> fv
function fv_zero() = FV_Fin(false, 0, 0)

val dotF : (ty, ty, int, int, int, fv, rmode) -> fv
function dotF(at_, bt, r, c, k, acc0, md) = {
  var acc : fv = acc0;
  foreach (i from 0 to (k - 1))
    acc = fv_add_exact(acc, fv_mul_exact(elemF(at_, mget(SP_MA, rc(r, i))), elemF(bt, mget(SP_MB, rc(i, c)))), md);
  acc
}

val dotI : (ty, ty, int, int, int, int, string) -> int
function dotI(at_, bt, r, c, k, acc0, op) = {
  var acc : int = acc0;
  foreach (i from 0 to (k - 1)) {
    let a = mget(SP_MA, rc(r, i));
    let b = mget(SP_MB, rc(i, c));
    acc = acc + (if op == "xor" then lxor_int(a, b) else if op == "and" then land_int(a, b) else elemI(at_, a) * elemI(bt, b))
  };
  acc
}

val noMods : unit -> fmods
function noMods() = fmods_of([||])

/* one result element, as bits of the D type; the rounding modifier (.rn / .rz / .rm / .rp, ISA 9.7.15.5: f64 only,
   .rn by default) rounds the exact sum */
val resElem : (ty, ty, ty, ty, int, int, int, list(string)) -> int
function resElem(dt, at_, bt, ct, r, c, k, m) = {
  let cv = mget(SP_MC, rc(r, c));
  let md = fmods_of(m).rm;
  /* .f64: "Precision of the element-wise multiplication and addition operation is identical to that of .f64 precision
     fused multiply-add" (9.7.15.5.14): each product is added by an f64 fma rounded with the modifier, starting from c,
     in k order (the order is the model's choice) */
  if isFloatElem(at_) & fw(at_) == 64 then {
    let fm = { noMods() with rm = md };
    var acc : int = cv;
    foreach (i from 0 to (k - 1)) acc = fpFma(fm, 64, mget(SP_MA, rc(r, i)), mget(SP_MB, rc(i, c)), acc);
    acc
  } else if isFloatElem(at_) then
    fin({ noMods() with satfinite = has(m, "satfinite"), rm = md }, fw(dt), dotF(at_, bt, r, c, k, elemF(ct, cv), md), false)
  else if ew(at_) > 1 then {
    let v = dotI(at_, bt, r, c, k, ival(cv, ct), "");
    trunc(if has(m, "satfinite") then clamp(v, dt) else v, 32)
  } else trunc(dotI(at_, bt, r, c, k, ival(cv, ct), if has(m, "and") then "and" else "xor"), 32)
}

/* all result elements into SP_MD */
val resRows : (string, ty, ty, ty, ty, list(string)) -> unit
function resRows(s, dt, at_, bt, ct, m) = {
  smap_clear(SP_MD);
  foreach (r from 0 to (shM(s) - 1))
    foreach (c from 0 to (shN(s) - 1))
      mset(SP_MD, rc(r, c), resElem(dt, at_, bt, ct, r, c, shK(s), m))
}

/* scattering D: pack each lane's D elements into registers */
val laneRegs : (string, ty, int) -> list(int)
function laneRegs(s, dt, l) = {
  let per = perReg(dt);
  var out : list(int) = [||];
  var i : int = 0;
  while i < nElC(s) do {
    var v : int = 0;
    foreach (j from 0 to (per - 1)) v = lor_int(v, shl_int(mget(SP_MD, posC(s, l, i + j)), ew(dt) * j));
    out = pend_append(out, [|v|]);
    i = i + per
  };
  out
}

val scatterAll : (string, ty, list((int, list(int)))) -> list((int, list(int)))
function scatterAll(s, dt, arr) = {
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) if is_some(lane_get(arr, l)) then out = (l, laneRegs(s, dt, l)) :: out;
  out
}

/* collective data: [shape, DT, AT, BT, CT, M (, ST)] */
val cv_str : (list(cval), int) -> string
function cv_str(d, i) = match d { [||] => "", h :: t => if i == 0 then (match h { CVS(x) => x, _ => "" }) else cv_str(t, i - 1) }

/* ---------------------------------------------------------------- mma.m8n8k4 with .f16 A / B (ISA 9.7.15.5.1)
 * Four independent 8x8x4 products: quad pair Q = (%laneid % 16) / 4 (lanes 4Q..4Q+3 low, 4Q+16..4Q+19 high group).
 * Product Q is kept at rows / columns 8Q..8Q+7 of D and C, rows 8Q.. of A and columns 8Q.. of B, so the
 * dense dot product above is reused (it reads only the 4 columns of A / rows of B of its own product). */
val layCol : (list(string), int) -> bool
function layCol(m, n) = {
  /* the N-th (0 = .alayout, 1 = .blayout) .row / .col qualifier is .col */
  var l = m;
  var k : int = 0;
  var r : bool = false;
  while is_cons(l) do {
    match l { x :: rest => { if x == "row" | x == "col" then { if k == n then r = x == "col"; k = k + 1 }; l = rest }, [||] => () }
  };
  r
}

val m884Q : int -> int
function m884Q(l) = tdiv_int(fmod_int(l, 16), 4)
val m884H : int -> int
function m884H(l) = 4 * tdiv_int(l, 16)

/* (row, col) of element i of lane l of the accumulator: .f16 (Figure 50) or .f32 (Figures 51, 52) */
val posC884 : (ty, int, int) -> int
function posC884(t, l, i) = {
  let o = 8 * m884Q(l);
  if ew(t) == 16 then rc(o + fmod_int(l, 4) + m884H(l), o + i)
  else rc(o + land_int(l, 1) + land_int(i, 2) + m884H(l), o + land_int(i, 4) + land_int(l, 2) + land_int(i, 1))
}

val m884_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function m884_compute(d, arr) = {
  let dt = cv_ty(d, 1);
  let at_ = cv_ty(d, 2);
  let bt = cv_ty(d, 3);
  let ct = cv_ty(d, 4);
  let m = cv_mods(d, 5);
  let acol = layCol(m, 0);
  let bcol = layCol(m, 1);
  smap_clear(SP_MA);
  smap_clear(SP_MB);
  smap_clear(SP_MC);
  foreach (l from 0 to 31) {
    match lane_get(arr, l) {
      Some(rs) => {
        let o = 8 * m884Q(l);
        let t = fmod_int(l, 4);
        let h = m884H(l);
        foreach (i from 0 to 3) {
          /* A: Figures 46 / 47; B: Figures 48 / 49 */
          mset(SP_MA, if acol then rc(o + i + h, t) else rc(o + t + h, i), fragBits(rs, 0, at_, i));
          mset(SP_MB, if bcol then rc(i, o + t + h) else rc(t, o + i + h), fragBits(rs, 2, bt, i))
        };
        foreach (i from 0 to 7) mset(SP_MC, posC884(ct, l, i), fragBits(rs, 4, ct, i))
      },
      None() => ()
    }
  };
  smap_clear(SP_MD);
  foreach (q from 0 to 3)
    foreach (r from 0 to 7)
      foreach (c from 0 to 7)
        mset(SP_MD, rc(8 * q + r, 8 * q + c), resElem(dt, at_, bt, ct, 8 * q + r, 8 * q + c, 4, m));
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) {
    if is_some(lane_get(arr, l)) then {
      let per = perReg(dt);
      var regs : list(int) = [||];
      foreach (r from 0 to (tdiv_int(8, per) - 1)) {
        var v : int = 0;
        foreach (j from 0 to (per - 1)) v = lor_int(v, shl_int(mget(SP_MD, posC884(dt, l, r * per + j)), ew(dt) * j));
        regs = pend_append(regs, [|v|])
      };
      out = (l, regs) :: out
    }
  };
  out
}

val mma_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function mma_compute(d, arr) = if cv_str(d, 0) == "m8n8k4" & ew(cv_ty(d, 2)) == 16 then m884_compute(d, arr) else {
  let s = cv_str(d, 0);
  let dt = cv_ty(d, 1);
  let at_ = cv_ty(d, 2);
  let bt = cv_ty(d, 3);
  let ct = cv_ty(d, 4);
  let ra = regsOf(nElA(s), at_);
  let rb = regsOf(nElB(s), bt);
  fillAll(SP_MA, 0, s, at_, arr, nElA(s), 0);
  fillAll(SP_MB, 1, s, bt, arr, nElB(s), ra);
  fillC(SP_MC, s, ct, arr, ra + rb);
  resRows(s, dt, at_, bt, ct, cv_mods(d, 5));
  scatterAll(s, dt, arr)
}

/* ---------------------------------------------------------------- block scaling */
val mxTy : (list(string), ty) -> ty
function mxTy(m, t) = if has(m, "kind::mxf4") | has(m, "kind::mxf4nvf4") then mkty("e2m1p", 4) else t

val bsNB : list(string) -> int
function bsNB(m) =
  if has(m, "scale_vec::4X") then 4
  else if has(m, "scale_vec::2X") | (has(m, "kind::mxf4") & not_bool(has(m, "scale_vec::1X"))) then 2
  else 1

val bsScale : (bool, int) -> fv
function bsScale(e43, x) =
  if not_bool(e43) then (if x == 255 then FV_NaN() else FV_Fin(false, 1, x - 127))   /* ue8m0: 0xFF is NaN */
  /* ue4m3 (ISA 5.2): 7 bits with the MSB padded, NaN is 0x7f */
  else if land_int(x, 127) == 127 then FV_NaN()
  else if asr_int(land_int(x, 127), 3) == 0 then FV_Fin(false, land_int(x, 7), -9)
  else FV_Fin(false, land_int(x, 7) + 8, asr_int(land_int(x, 127), 3) - 10)

val bsReg : (list((int, list(int))), int, int) -> int
function bsReg(arr, l, i) = at(listAt(arr, l), i)

/* the block-scaled product once A is in SP_MA; ra = registers of the A fragment */
val mmabs_core : (list(cval), list((int, list(int))), int) -> list((int, list(int)))
function mmabs_core(d, arr, ra) = {
  let s = cv_str(d, 0);
  let dt = cv_ty(d, 1);
  let at_ = cv_ty(d, 2);
  let bt = cv_ty(d, 3);
  let ct = cv_ty(d, 4);
  let m = cv_mods(d, 5);
  let st = cv_ty(d, 6);
  let rb = regsOf(nElB(s), bt);
  let off = ra + rb + regsOf(nElC(s), ct);
  let nb = bsNB(m);
  fillAll(SP_MB, 1, s, bt, arr, nElB(s), ra);
  fillC(SP_MC, s, ct, arr, ra + rb);
  smap_clear(SP_SA);
  smap_clear(SP_SB);
  foreach (r from 0 to 15)
    foreach (j from 0 to (nb - 1))
      mset(SP_SA, rc(r, j), land_int(asr_int(bsReg(arr, fmod_int(r, 8) * 4 + 2 * bsReg(arr, 0, off + 2) + tdiv_int(r, 8), off),
                                             8 * (bsReg(arr, 0, off + 1) + j)), 255));
  foreach (c from 0 to 7)
    foreach (j from 0 to (nb - 1))
      mset(SP_SB, rc(j, c), land_int(asr_int(bsReg(arr, c * 4 + bsReg(arr, 0, off + 5), off + 3),
                                             8 * (bsReg(arr, 0, off + 4) + j)), 255));
  let e43 = ty_eq(st, mkty("ue4m3", 8));
  let blk = tdiv_int(shK(s), nb);
  smap_clear(SP_MD);
  foreach (r from 0 to (shM(s) - 1))
    foreach (c from 0 to (shN(s) - 1)) {
      var acc = elemF(ct, mget(SP_MC, rc(r, c)));
      foreach (i from 0 to (shK(s) - 1)) {
        let p = fv_mul_exact(fv_mul_exact(fv_mul_exact(elemF(at_, mget(SP_MA, rc(r, i))), elemF(bt, mget(SP_MB, rc(i, c)))),
                                          bsScale(e43, mget(SP_SA, rc(r, tdiv_int(i, blk))))),
                             bsScale(e43, mget(SP_SB, rc(tdiv_int(i, blk), c))));
        acc = fv_add_exact(acc, p, RN)
      };
      mset(SP_MD, rc(r, c), fin(noMods(), 32, acc, false))
    };
  scatterAll(s, dt, arr)
}

val mmabs_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function mmabs_compute(d, arr) = {
  let s = cv_str(d, 0);
  let at_ = cv_ty(d, 2);
  fillAll(SP_MA, 0, s, at_, arr, nElA(s), 0);
  mmabs_core(d, arr, regsOf(nElA(s), at_))
}

/* ---------------------------------------------------------------- wmma fragments */
/* element I of lane L holds matrix element ((L * NEL + I) mod size) in row-major order */
val wmRegs : (string, string, ty) -> int
function wmRegs(mat0, s, t) = {
  let mat = if mat0 == "d" then "c" else mat0;
  let f16 = ty_eq(t, mkty("f", 16));
  if mat == "c" then {
    if f16 then 4
    else if shM(s) == 8 & shN(s) == 8 then 2
    else 8
  } else if mat == "a" | mat == "b" then {
    if f16 then 8
    else if t.k == "bf" then {
      if mat == "a" then (if s == "m16n16k16" then 4 else if s == "m8n32k16" then 2 else 8)
      else (if s == "m16n16k16" then 4 else if s == "m8n32k16" then 8 else 2)
    } else if t.k == "tf" then 4
    else if ty_eq(t, mkty("f", 64)) then 1
    else if t.w == 8 & (t.k == "s" | t.k == "u") then {
      if mat == "a" then (if s == "m16n16k16" then 2 else if s == "m8n32k16" then 1 else 4)
      else (if s == "m16n16k16" then 2 else if s == "m8n32k16" then 4 else 1)
    } else 1
  } else 1
}

val wmNel : (string, string, ty) -> int
function wmNel(mat, s, t) = wmRegs(mat, s, t) * perReg(t)
val wmRows : (string, string) -> int
function wmRows(mat, s) = if mat == "b" then shK(s) else shM(s)
val wmCols : (string, string) -> int
function wmCols(mat, s) = if mat == "a" then shK(s) else shN(s)

val wmPos : (string, string, ty, int, int) -> int
function wmPos(mat, s, t, l, i) = {
  let x = fmod_int(l * wmNel(mat, s, t) + i, wmRows(mat, s) * wmCols(mat, s));
  rc(tdiv_int(x, wmCols(mat, s)), fmod_int(x, wmCols(mat, s)))
}

val wmFill : (int, string, string, ty, list((int, list(int))), int) -> unit
function wmFill(sp, mat, s, t, arr, off) = {
  smap_clear(sp);
  foreach (l from 0 to 31) {
    match lane_get(arr, l) {
      Some(rs) => foreach (i from 0 to (wmNel(mat, s, t) - 1)) mset(sp, wmPos(mat, s, t, l, i), fragBits(rs, off, t, i)),
      None() => ()
    }
  }
}

val wmScatter : (string, ty, list((int, list(int)))) -> list((int, list(int)))
function wmScatter(s, dt, arr) = {
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) {
    if is_some(lane_get(arr, l)) then {
      var regs : list(int) = [||];
      foreach (r from 0 to (wmRegs("c", s, dt) - 1)) {
        var v : int = 0;
        foreach (j from 0 to (perReg(dt) - 1))
          v = lor_int(v, shl_int(mget(SP_MD, wmPos("c", s, dt, l, r * perReg(dt) + j)), ew(dt) * j));
        regs = pend_append(regs, [|v|])
      };
      out = (l, regs) :: out
    }
  };
  out
}

val wmma_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function wmma_compute(d, arr) = {
  let s = cv_str(d, 0);
  let dt = cv_ty(d, 1);
  let at_ = cv_ty(d, 2);
  let bt = cv_ty(d, 3);
  let ct = cv_ty(d, 4);
  wmFill(SP_MA, "a", s, at_, arr, 0);
  wmFill(SP_MB, "b", s, bt, arr, wmRegs("a", s, at_));
  wmFill(SP_MC, "c", s, ct, arr, wmRegs("a", s, at_) + wmRegs("b", s, bt));
  resRows(s, dt, at_, bt, ct, cv_mods(d, 5));
  wmScatter(s, dt, arr)
}

/* ---------------------------------------------------------------- operands */
val vecRegs : (xop, ty) -> list(int)
function vecRegs(o, t) = match o { XV(l) => rdList(l, mkty("b", regBits(t))), _ => [||] }
val vecOps : xop -> list(xop)
function vecOps(o) = {
  var out : list(xop) = [||];
  var l = vec_items(o);
  while is_cons(l) do { match l { s :: rest => { out = pend_append(out, [|XS(s)|]); l = rest }, [||] => () } };
  out
}

/* ---------------------------------------------------------------- wmma.load / wmma.store */
val elemIndex : (bool, int, int, int) -> int
function elemIndex(col, r, c, st) = if col then c * st + r else r * st + c

val wmBytes : ty -> int
function wmBytes(t) = if ew(t) < 8 then 1 else tdiv_int(ew(t), 8)

/* element I of the lane's fragment in memory: (location, bit offset) */
val wmLoc : (string, string, ty, int, bool, int, loc, int) -> (loc, int)
function wmLoc(mat, s, t, l, col, st, base, i) = {
  let p = wmPos(mat, s, t, l, i);
  let x = elemIndex(col, tdiv_int(p, 1024), fmod_int(p, 1024), st) * ew(t);
  (loc_add(base, tdiv_int(x, 8)), fmod_int(x, 8))
}

/* the default wmma stride: the leading dimension (A: K row-major / M column-major, B: N / K, C and D: N / M) */
val wmDefStride : (string, string, bool) -> int
function wmDefStride(mat, s, col) = match mat {
  "a" => if col then shM(s) else shK(s),
  "b" => if col then shK(s) else shN(s),
  _ => if col then shM(s) else shN(s)
}

val wm_load : (opcode, xop, addr, int) -> unit
function wm_load(o, dv, a, st) = {
  let m = o.mods;
  let t = ty1(o);
  let mat = pick(m, [|"a", "b", "c"|], "a");
  let s = shapeOf(m);
  let base = ea(a, spaceOf(m));
  let l = fmod_int(t_tid[tchk(cur)], 32);
  let n = wmNel(mat, s, t);
  var els : list(int) = [||];
  var i : int = 0;
  while i < n & not_bool(stuck) do {
    let (lc, off) = wmLoc(mat, s, t, l, has(m, "col"), st, base, i);
    let v = memop_do(lc, RdOp(wmBytes(t)));
    els = pend_append(els, [|land_int(asr_int(v, off), mask(ew(t)))|]);
    i = i + 1
  };
  if not_bool(stuck) then {
    var ds = vecOps(dv);
    var j : int = 0;
    while is_cons(ds) & j < n do {
      match ds { d :: rest => {
        var v : int = 0;
        foreach (k from 0 to (perReg(t) - 1)) if j + k < n then v = lor_int(v, shl_int(land_int(at(els, j + k), mask(ew(t))), ew(t) * k));
        wr(d, v);
        ds = rest
      }, [||] => () };
      j = j + perReg(t)
    }
  }
}

val wm_store : (opcode, addr, xop, int) -> unit
function wm_store(o, a, vs, st) = {
  let m = o.mods;
  let t = ty1(o);
  let s = shapeOf(m);
  let base = ea(a, spaceOf(m));
  let l = fmod_int(t_tid[tchk(cur)], 32);
  let n = wmNel("c", s, t);
  let rs = vecRegs(vs, t);
  var i : int = 0;
  while i < n & not_bool(stuck) do {
    let (lc, off) = wmLoc("c", s, t, l, has(m, "col"), st, base, i);
    let v = fragBits(rs, 0, t, i);
    let _ = memop_do(lc, if ew(t) < 8 then InsOp(off, ew(t), v) else WrOp(tdiv_int(ew(t), 8), v));
    i = i + 1
  }
}

val exec_mma : (opcode, list(xop)) -> bool
function exec_mma(o, ops) = {
  let m = o.mods;
  let full : int = 4294967295;
  match (o.base, o.tys, ops) {
    ("mma", [|dt, at_, bt, ct|], [|dv, av, bv, cv|]) => {
      if has(m, "sync") & not_bool(has(m, "sp")) & not_bool(has(m, "sp::ordered_metadata")) then {
        /* ".m8n8k4 : When .ctype is .f32, .dtype must also be .f32" (ISA 9.7.15.5.14) is reported under .isacheck;
           otherwise the f32 accumulation is rounded to the .f16 destination */
        do_coll("mma", full, [|CVS(shapeOf(m)), CVT(dt), CVT(at_), CVT(bt), CVT(ct), CVM(m)|],
                pend_append(pend_append(vecRegs(av, at_), vecRegs(bv, bt)), vecRegs(cv, ct)), vecOps(dv));
        true
      } else false
    },
    ("mma", [|dt, at0, bt0, ct, st|], [|dv, av, bv, cv, sa, sela, sb, selb|]) => {
      if has(m, "sync") & has(m, "block_scale") then {
        let at_ = mxTy(m, at0);
        let bt = mxTy(m, bt0);
        do_coll("mmabs", full, [|CVS(shapeOf(m)), CVT(dt), CVT(at_), CVT(bt), CVT(ct), CVM(m), CVT(st)|],
                pend_append(pend_append(pend_append(vecRegs(av, at_), vecRegs(bv, bt)), vecRegs(cv, ct)), pend_append(pend_append([|rd(sa, mkty("b", 32))|], vecRegs(sela, mkty("b", 32))), pend_append([|rd(sb, mkty("b", 32))|], vecRegs(selb, mkty("b", 32))))), vecOps(dv));
        true
      } else false
    },
    ("wmma", [|dt, ct|], [|dv, av, bv, cv|]) => {
      if has(m, "mma") then {
        let h = mkty("f", 16);
        do_coll("wmma", full, [|CVS(shapeOf(m)), CVT(dt), CVT(h), CVT(h), CVT(ct), CVM(m)|],
                pend_append(pend_append(vecRegs(av, h), vecRegs(bv, h)), vecRegs(cv, ct)), vecOps(dv));
        true
      } else false
    },
    ("wmma", [|dt, at_, bt, ct|], [|dv, av, bv, cv|]) => {
      if has(m, "mma") then {
        do_coll("wmma", full, [|CVS(shapeOf(m)), CVT(dt), CVT(at_), CVT(bt), CVT(ct), CVM(m)|],
                pend_append(pend_append(vecRegs(av, at_), vecRegs(bv, bt)), vecRegs(cv, ct)), vecOps(dv));
        true
      } else false
    },
    ("wmma", [|_|], [|dv, XB([|a|], [||]), sto|]) => if has(m, "load") then { wm_load(o, dv, a, rdu32(sto)); true } else false,
    ("wmma", [|_|], [|XB([|a|], [||]), vs, sto|]) => if has(m, "store") then { wm_store(o, a, vs, rdu32(sto)); true } else false,
    /* stride is optional: the default is the leading dimension of the matrix (ISA 9.7.15.3 Matrix Storage for WMMA) */
    ("wmma", [|_|], [|dv, XB([|a|], [||])|]) =>
      if has(m, "load") then { wm_load(o, dv, a, wmDefStride(pick(m, [|"a", "b", "c"|], "a"), shapeOf(m), has(m, "col"))); true } else false,
    ("wmma", [|_|], [|XB([|a|], [||]), vs|]) =>
      if has(m, "store") then { wm_store(o, a, vs, wmDefStride("c", shapeOf(m), has(m, "col"))); true } else false,
    _ => false
  }
}

/* =============================================================================
 * PTX-EXEC-SP (semantics/ptx-exec-sp.k): sparse mma.sp: A is expanded from its compressed values
 * and metadata (chunking 2:4, 1:2 for tf32, pair-wise 4:8 for 4-bit elements), then the dense
 * arithmetic above is reused.
 * ============================================================================= */
val spCw : ty -> int
function spCw(t) = if t.k == "tf" then 2 else if ew(t) == 4 then 8 else 4
val spNz : ty -> int
function spNz(t) = if t.k == "tf" then 1 else if ew(t) == 4 then 4 else 2
val spNel : string -> int
function spNel(s) = tdiv_int(shM(s) * shK(s), 64)
val spRh : (ty, int) -> int
function spRh(t, u) = if ew(t) >= 16 then fmod_int(u, 2) else fmod_int(tdiv_int(u, 2), 2)
val spChunk : (ty, int, int) -> int
function spChunk(t, u, tg) = if ew(t) >= 16 then tg + 4 * tdiv_int(u, 2) else 2 * tg + fmod_int(u, 2) + 8 * tdiv_int(u, 4)

val spMeta : (list((int, list(int))), int) -> int
function spMeta(arr, l) = { let c = listAt(arr, l); at(c, list_length(c) - 1) }

/* (lane in the group of four, bit offset) of the nibble of chunk c of row g + 8 rh, from NC chunks per row and the
   selector: one thread (NC = 4), a thread pair (NC = 8) or all four threads (NC = 16).  With eight chunks the
   16- and 32-bit types (Figures 122, 125; wgmma Figures 176, 178) give thread 2 sel + c / 4 chunks 4 (c / 4) .. + 3
   of both rows (row g + 8 in the upper half), the 8-bit types (Figure 129) give thread 2 sel + rh all chunks of
   one row */
val spNibAt : (ty, int, int, int, int) -> (int, int)
function spNibAt(t, rh, c, nc, sel) =
  if nc == 4 then (sel, 16 * rh + 4 * c)
  else if nc == 8 & ew(t) >= 16 then (2 * sel + tdiv_int(c, 4), 16 * rh + 4 * fmod_int(c, 4))
  else if nc == 8 then (2 * sel + rh, 4 * c)
  else (2 * tdiv_int(c, 8) + rh, 4 * fmod_int(c, 8))

val spNib : (ty, list((int, list(int))), int, int, int, int, int) -> int
function spNib(t, arr, g, rh, c, nc, sel) =
  if nc == 4 | nc == 8 | nc == 16 then { let (ln, b) = spNibAt(t, rh, c, nc, sel); land_int(asr_int(spMeta(arr, 4 * g + ln), b), 15) }
  else 0

val spPos : (ty, int, int) -> int
function spPos(t, nib, j) =
  if t.k == "tf" then tdiv_int(land_int(nib, 3), 2)
  else if ew(t) == 4 then 2 * land_int(asr_int(nib, 2 * tdiv_int(j, 2)), 3) + fmod_int(j, 2)
  else land_int(asr_int(nib, 2 * j), 3)

/* 9.7.15.6.1, under .ub strict: the sparsity selector is in {0..3} when one thread holds the metadata, {0, 1} for a
 * thread pair, 0 when all four threads do; metadata values: .tf32 "0b1110 and 0b0100 are the only meaningful index
 * values"; for mma.sp::ordered_metadata and the f8f6f4 / mx kinds "0b0100, 0b1000, 0b1001, 0b1100, 0b1101, 0b1110 are
 * the meaningful values"; otherwise "0b0000, 0b0101, 0b1010, 0b1111 are invalid values for metadata" */
val spSelOk : (int, int) -> bool
function spSelOk(nc, sel) = if nc == 4 then sel >= 0 & sel <= 3 else if nc == 8 then sel == 0 | sel == 1 else sel == 0
val spNibOk : (ty, bool, int) -> bool
function spNibOk(t, ord, nib) =
  if t.k == "tf" then nib == 14 | nib == 4
  else if ord then has([|"4", "8", "9", "12", "13", "14"|], dec_str(nib))
  else not_bool(nib == 0 | nib == 5 | nib == 10 | nib == 15)

val spFillAll : (string, ty, list((int, list(int))), int, bool) -> unit
function spFillAll(s, t, arr, sel, ord) = {
  smap_clear(SP_MA);
  if ub & not_bool(spSelOk(tdiv_int(shK(s), spCw(t)), sel)) then fault("mma.sp: sparsity selector outside the values of the shape", sel);
  foreach (l from 0 to 31) {
    match lane_get(arr, l) {
      Some(rs) => foreach (i from 0 to (spNel(s) - 1)) {
        let u = tdiv_int(i, spNz(t));
        let rh = spRh(t, u);
        let ch = spChunk(t, u, fmod_int(l, 4));
        let nib = spNib(t, arr, tdiv_int(l, 4), rh, ch, tdiv_int(shK(s), spCw(t)), sel);
        if ub & not_bool(stuck) & not_bool(spNibOk(t, ord, nib)) then fault("mma.sp: metadata index value that is not meaningful", nib);
        mset(SP_MA, rc(tdiv_int(l, 4) + 8 * rh, spCw(t) * ch + spPos(t, nib, fmod_int(i, spNz(t)))), fragBits(rs, 0, t, i))
      },
      None() => ()
    }
  }
}

val spOrd : list(string) -> bool
function spOrd(m) = has(m, "sp::ordered_metadata") | has(m, "kind::f8f6f4") | has(m, "kind::mxf8f6f4") | has(m, "kind::mxf4") | has(m, "kind::mxf4nvf4")

val mmasp_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function mmasp_compute(d, arr) = {
  let s = cv_str(d, 0);
  let dt = cv_ty(d, 1);
  let at_ = cv_ty(d, 2);
  let bt = cv_ty(d, 3);
  let ct = cv_ty(d, 4);
  let ra = regsOf(spNel(s), at_);
  let rb = regsOf(nElB(s), bt);
  spFillAll(s, at_, arr, cv_int(d, 6), spOrd(cv_mods(d, 5)));
  fillAll(SP_MB, 1, s, bt, arr, nElB(s), ra);
  fillC(SP_MC, s, ct, arr, ra + rb);
  resRows(s, dt, at_, bt, ct, cv_mods(d, 5));
  scatterAll(s, dt, arr)
}

val mmabssp_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function mmabssp_compute(d, arr) = {
  let s = cv_str(d, 0);
  let at_ = cv_ty(d, 2);
  spFillAll(s, at_, arr, cv_int(d, 7), spOrd(cv_mods(d, 5)));
  mmabs_core(d, arr, regsOf(spNel(s), at_))
}

val exec_sp : (opcode, list(xop)) -> bool
function exec_sp(o, ops) = {
  let m = o.mods;
  let full : int = 4294967295;
  let b32 = mkty("b", 32);
  if o.base != "mma" | not_bool(has(m, "sync") & has(m, "sp")) then false
  else match (o.tys, ops) {
    ([|dt, at_, bt, ct|], [|dv, av, bv, cv, e, f|]) => {
      do_coll("mma.sp", full, [|CVS(shapeOf(m)), CVT(dt), CVT(at_), CVT(bt), CVT(ct), CVM(m), CVI(rdu32(f))|],
              pend_append(pend_append(pend_append(vecRegs(av, at_), vecRegs(bv, bt)), vecRegs(cv, ct)), [|rd(e, b32)|]), vecOps(dv));
      true
    },
    ([|dt, at0, bt0, ct, st|], [|dv, av, bv, cv, e, f, sa, sela, sb, selb|]) => {
      if has(m, "block_scale") then {
        let at_ = mxTy(m, at0);
        let bt = mxTy(m, bt0);
        do_coll("mmabs.sp", full, [|CVS(shapeOf(m)), CVT(dt), CVT(at_), CVT(bt), CVT(ct), CVM(m), CVT(st), CVI(rdu32(f))|],
                pend_append(pend_append(pend_append(pend_append(vecRegs(av, at_), vecRegs(bv, bt)), vecRegs(cv, ct)),
                                        pend_append(pend_append([|rd(sa, b32)|], vecRegs(sela, b32)), pend_append([|rd(sb, b32)|], vecRegs(selb, b32)))),
                            [|rd(e, b32)|]), vecOps(dv));
        true
      } else false
    },
    _ => false
  }
}

val coll_mma : (string, list(cval), list((int, list(int)))) -> option(list((int, list(int))))
function coll_mma(k, d, arr) = match k {
  "mma" => Some(mma_compute(d, arr)),
  "mmabs" => Some(mmabs_compute(d, arr)),
  "wmma" => Some(wmma_compute(d, arr)),
  "mma.sp" => Some(mmasp_compute(d, arr)),
  "mmabs.sp" => Some(mmabssp_compute(d, arr)),
  _ => None()
}

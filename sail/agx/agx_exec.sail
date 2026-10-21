/* =============================================================================
 * AGX execution: `execute` clauses, one per instruction (the #exec rules of agx/agx.k).
 * ============================================================================= */

/* ---------------------------------------------------------------- register views */
val gidx : (int, int) -> range(0, 8191)
function gidx(t, n) = {
  let i = t * 256 + n;
  assert(0 <= i & i < 8192, "register out of range");
  i
}

val reg_raw : (int, int) -> int
function reg_raw(t, n) = unsigned(gpr[gidx(t, n)])

/* raw(G, U, T, R): the 32-bit container.  A 32-bit uniform at an odd 16-bit index (view 4) is not measured: the
 * model stops with an error (applegpu and agx_run read uN) */
val raw : (int, areg) -> int
function raw(t, r) = {
  if r.view == 4 then fatal("a 32-bit uniform at an odd 16-bit index (uN.odd) is not measured");
  if r.file == UNI then {
    unsigned(get_upr(r.num))
  } else reg_raw(t, r.num)
}

val raw_n : (int, rfile, int) -> int
function raw_n(t, f, n) = raw(t, { gpr32(n) with file = f })

val view_bits : areg -> int
function view_bits(r) = match r.view { 0 => 32, 3 => 64, 4 => 32, _ => 16 }

/* val(G, U, T, R) with the encoding's sign-extension modifier */
val rval : (int, areg) -> int
function rval(t, r) = {
  let v : int = match r.view {
    0 => raw(t, r),
    4 => raw(t, r),
    1 => land_int(raw(t, r), 65535),
    2 => land_int(shr_int(raw(t, r), 16), 65535),
    _ => raw(t, r) + shl_int(raw_n(t, r.file, r.num + 1), 32)
  };
  if r.sx then sext_int(v, view_bits(r)) else v
}

/* put(G, T, R, V): write lane t's destination, truncated to the view (always the general registers) */
val put : (int, areg, int) -> unit
function put(t, r, v) = {
  let i = gidx(t, r.num);
  match r.view {
    0 => gpr[i] = get_slice_int(32, v, 0),
    1 => gpr[i] = gpr[i][31 .. 16] @ get_slice_int(16, v, 0),
    2 => gpr[i] = get_slice_int(16, v, 0) @ gpr[i][15 .. 0],
    _ => {
      gpr[i] = get_slice_int(32, v, 0);
      gpr[gidx(t, r.num + 1)] = get_slice_int(32, v, 32)
    }
  }
}

val db : areg -> int
function db(r) = view_bits(r)

val amask : int -> int
function amask(b) = mask_int(b)

/* the abs / neg modifiers of a float immediate */
val fimm_mods : (fv, bool, bool) -> fv
function fimm_mods(v, ab, ng) = { let v1 = if ab then fv_abs(v) else v in if ng then fv_neg(v1) else v1 }

/* opv(G, U, T, A, B): registers read their view, integer immediates are broadcast (truncated to
 * B bits), float immediates are encoded at B bits */
val opv : (int, aop, int) -> int
function opv(t, a, b) = match a {
  OReg(r) => rval(t, r),
  OImm(i) => land_int(i, amask(b)),
  OFImm(f) => encodeF(f64v(f), b),
  OFMod(f, ab, ng) => encodeF(fimm_mods(f64v(f), ab, ng), b)
}

/* float source: the operand bits at width w with the abs / neg modifiers applied to the sign bit */
val fopv : (int, aop, int) -> int
function fopv(t, a, w) = {
  let v = opv(t, a, w);
  match a {
    OReg(r) => {
      let v1 = if r.fabs then land_int(v, lnot_int(pow2(w - 1))) else v;
      if r.fneg then lxor_int(v1, pow2(w - 1)) else v1
    },
    _ => v
  }
}

val is_active : int -> bool
function is_active(t) = bit_int(unsigned(em), t)

/* ---------------------------------------------------------------- comparisons (cmpv of agx.k) */
/* "nan loses" less / greater than: true when only the second operand is NaN */
val nanloses : (fv, fv, int) -> bool
function nanloses(x, y, dir) = {
  if fv_is_nan(x) | fv_is_nan(y) then { let r : bool = fv_is_nan(y) & not_bool(fv_is_nan(x)); r }
  else { let c = fv_cmp(x, y); if dir < 0 then c < 0 else c > 0 }
}

val fcmpv : (acc, fv, fv) -> bool
function fcmpv(cc, x, y) = {
  let n = fv_is_nan(x) | fv_is_nan(y);
  match cc {
    CC_feq => not_bool(n) & fv_cmp(x, y) == 0,
    CC_fne => not_bool(n) & fv_cmp(x, y) != 0,
    CC_flt => not_bool(n) & fv_cmp(x, y) < 0,
    CC_fle => not_bool(n) & fv_cmp(x, y) <= 0,
    CC_fgt => not_bool(n) & fv_cmp(x, y) > 0,
    CC_fge => not_bool(n) & fv_cmp(x, y) >= 0,
    CC_fltn => nanloses(x, y, -1),
    CC_fgtn => nanloses(x, y, 1),
    CC_fnequ => not_bool(not_bool(n) & fv_cmp(x, y) == 0),
    CC_fnlt => not_bool(not_bool(n) & fv_cmp(x, y) < 0),
    CC_fngt => not_bool(not_bool(n) & fv_cmp(x, y) > 0),
    CC_fnle => not_bool(not_bool(n) & fv_cmp(x, y) <= 0),
    CC_fnge => not_bool(not_bool(n) & fv_cmp(x, y) >= 0),
    CC_fnltn => not_bool(nanloses(x, y, -1)),
    CC_fngtn => not_bool(nanloses(x, y, 1)),
    _ => false
  }
}

/* integer value of a source: its view zero-extended (sign-extended with `.sx`); immediates 32-bit */
val op_size : aop -> int
function op_size(a) = match a { OReg(r) => view_bits(r), _ => 32 }

/* float value of a source (applegpu / hardware): 16-bit registers are binary16, 32-bit registers
 * binary32 (subnormals read as zero when daz, i.e. outside the 16-bit forms), float literals rounded
 * to the operation's format (binary32 when daz, else binary16; exact for the 8-bit float immediates of
 * the encodings), integer immediates (K dialect only) the bits at that format; then abs / neg */
val fval : (int, aop, bool) -> fv
function fval(t, a, daz) = match a {
  OReg(r) => {
    let b = rval(t, { r with sx = false });
    let v : fv = match r.view {
      0 => decodeF(if daz then flushB(b, 32) else b, 32),
      4 => decodeF(if daz then flushB(b, 32) else b, 32),
      3 => decodeF(b, 64),
      _ => decodeF(b, 16)
    };
    let v1 = if r.fabs then fv_abs(v) else v;
    if r.fneg then fv_neg(v1) else v1
  },
  OFImm(f) => { let w : int = if daz then 32 else 16 in decodeF(encodeF(f64v(f), w), w) },
  OFMod(f, ab, ng) => { let w : int = if daz then 32 else 16 in fimm_mods(decodeF(encodeF(f64v(f), w), w), ab, ng) },
  OImm(i) => { let w : int = if daz then 32 else 16 in decodeF(land_int(i, mask_int(w)), w) }
}

/* condition codes on integers: unsigned / signed (sign-extended at each operand's size) */
val icmpv : (acc, int, int, int, int) -> bool
function icmpv(cc, a, asz, b, bsz) = match cc {
  CC_eq => a == b, CC_ne => a != b,
  CC_seq => sext_int(a, asz) == sext_int(b, bsz), CC_sne => sext_int(a, asz) != sext_int(b, bsz),
  CC_ult => a < b, CC_ule => a <= b, CC_ugt => a > b, CC_uge => a >= b,
  CC_slt => sext_int(a, asz) < sext_int(b, bsz),
  CC_sle => sext_int(a, asz) <= sext_int(b, bsz),
  CC_sgt => sext_int(a, asz) > sext_int(b, bsz),
  CC_sge => sext_int(a, asz) >= sext_int(b, bsz),
  _ => false
}

val is_fcc : acc -> bool
function is_fcc(cc) = match cc {
  CC_feq => true, CC_fne => true, CC_flt => true, CC_fle => true, CC_fgt => true, CC_fge => true,
  CC_fltn => true, CC_fgtn => true, CC_fnequ => true, CC_fnlt => true, CC_fngt => true, CC_fnle => true,
  CC_fnge => true, CC_fnltn => true, CC_fngtn => true, _ => false
}

/* the comparison of lane t (cmpv of agx.k).  Integer conditions compare the sources at their own
 * sizes (applegpu IConditionDesc: unsigned values, or sign-extended for the signed conditions), for
 * both instruction kinds; float conditions on float instructions compare the float values (fval:
 * fp32 subnormals read as zero, as on the hardware).  K dialect only: `eq` / `ne` never hold on
 * float instructions, and `feq` / `fne` on integer instructions compare the 32-bit views. */
val cmp_lane : (int, bool, acc, aop, aop) -> bool
function cmp_lane(t, isf, cc, a, b) = {
  if is_fcc(cc) then {
    if isf then fcmpv(cc, fval(t, a, true), fval(t, b, true))
    else match cc { CC_feq => opv(t, a, 32) == opv(t, b, 32), CC_fne => opv(t, a, 32) != opv(t, b, 32), _ => false }
  } else if isf & (cc == CC_eq | cc == CC_ne) then false
  else icmpv(cc, opv(t, a, 32), op_size(a), opv(t, b, 32), op_size(b))
}

/* ---------------------------------------------------------------- integer helpers */
val asext : (int, int) -> int
function asext(v, w) = sext_int(v, w)

/* truth tables 0011 and 1100 (B and not B by the table) give A on the hardware (applegpu: UNDEFINED,
 * result = a) */
val bitopv : (int, int, int, int) -> int
function bitopv(tt, a, b, w) = if tt == 12 | tt == 3 then land_int(a, amask(w)) else {
  let r0 = if land_int(tt, 1) != 0 then land_int(lnot_int(a), lnot_int(b)) else 0;
  let r1 = if land_int(tt, 2) != 0 then land_int(a, lnot_int(b)) else 0;
  let r2 = if land_int(tt, 4) != 0 then land_int(lnot_int(a), b) else 0;
  let r3 = if land_int(tt, 8) != 0 then land_int(a, b) else 0;
  land_int(lor_int(lor_int(r0, r1), lor_int(r2, r3)), amask(w))
}

val fmask : int -> int
function fmask(m) = if m == 0 then 4294967295 else amask(m)

val bfiv : (int, int, int, int) -> int
function bfiv(m, a, b, c) = {
  let sh = land_int(c, 127);
  land_int(lor_int(land_int(a, lnot_int(shl_int(fmask(m), sh))), shl_int(land_int(b, fmask(m)), sh)), 4294967295)
}

val bfeilv : (int, int, int, int) -> int
function bfeilv(m, a, b, c) =
  land_int(lor_int(land_int(a, lnot_int(fmask(m))), land_int(asr_int(b, land_int(c, 127)), fmask(m))), 4294967295)

val popc32 : int -> int
function popc32(v) = popc_int(trunc_int(v, 32))

val brev32 : int -> int
function brev32(v) = {
  var r : int = 0;
  foreach (i from 0 to 31) { if bit_int(v, i) then r = r + pow2(31 - i) };
  r
}

/* ffsv(V, 31): index of the highest set bit among bits 31..0, 4294967295 if none */
val ffsv : int -> int
function ffsv(v) = {
  var r : int = 4294967295;
  var i : int = 31;
  while i >= 0 & r == 4294967295 do { if bit_int(v, i) then r = i; i = i - 1 };
  r
}

val saturate : (int, int, bool) -> int
function saturate(v, w, sgn) = {
  let lo = if sgn then negate(pow2(w - 1)) else 0;
  let hi = if sgn then pow2(w - 1) - 1 else pow2(w) - 1;
  if v < lo then lo else if v > hi then hi else v
}

val op_sx : aop -> bool
function op_sx(a) = match a { OReg(r) => r.sx, _ => false }
val op_bits : aop -> int
function op_bits(a) = match a { OReg(r) => view_bits(r), _ => 16 }

/* ---------------------------------------------------------------- float helpers */
/* .sat: clamp to [0, 1]; NaN and zeros give +0 */
val agx_sat : fv -> fv
function agx_sat(v) = {
  if fv_is_nan(v) | fv_is_zero(v) | fv_sign(v) then FV_Fin(false, 0, 0)
  else if fv_lt(fv_one(), v) then fv_one()
  else v
}

/* a result (exact value) written to D: binary16 for a 16-bit destination (round to nearest even,
 * subnormals kept, NaN 0x7e00); binary32 otherwise, where a non-zero value of magnitude below 2^-126
 * becomes a zero of its sign before rounding, and NaN is 0x7fc00000 */
val fres : (fv, bool, areg) -> int
function fres(v, sat, d) = {
  let x = if sat then agx_sat(v) else v;
  if view_bits(d) == 16 then { if fv_is_nan(x) then 32256 else encodeF(x, 16) }
  else if fv_is_nan(x) then 2143289344
  else match x {
    FV_Fin(sg, m, _) => if m != 0 & fv_lt(fv_abs(x), FV_Fin(false, 1, -126)) then (if sg then 2147483648 else 0) else encodeF(x, 32),
    _ => encodeF(x, 32)
  }
}

/* ---------------------------------------------------------------- execute */
val execute : ast -> unit
scattered function execute

/* #lanes(OP): every active lane computes its result from the state before the instruction; the
 * results are then written (the per-lane vectors r below) */
function clause execute(MOV(d, a)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then r[t] = opv(t, a, db(d)) };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(GET_SR(d, n)) = {
  foreach (t from 0 to 31) {
    if is_active(t) then {
      put(t, d, unsigned(get_sr_v(n * 32 + t)))
    }
  }
}

function clause execute(IADD(sub, sat, d, a, b, sh)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let x = opv(t, a, db(d));
      let y0 = opv(t, b, db(d));
      let y1 = if sub then negate(y0) else y0;
      let y = if sh < 5 then shl_int(y1, sh) else 0;
      let s = x + y;
      let s2 = if sat & sh == 0 & op_bits(a) <= 32 & op_bits(b) <= 32 & db(d) <= 32
               then saturate(s, db(d), op_sx(a) | op_sx(b)) else s;
      r[t] = land_int(s2, amask(db(d)))
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

val is_reg64 : aop -> bool
function is_reg64(a) = match a { OReg(r) => r.view == 3, _ => false }

/* a 64-bit multiplicand (A or B a register pair: an encoding Mesa's packer accepts, applegpu and agx_run read the
 * low register, the LLVM backend rejects it) is not measured (Apple-M2-Playground 2026-10-11-int64 measured 32-bit
 * multiplicands only): the model stops with an error */
function clause execute(IMADD(sub, sat, d, a, b, c, sh)) = {
  if is_reg64(a) | is_reg64(b) then fatal("imadd / imsub with a 64-bit multiplicand is not measured");
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let z0 = opv(t, c, db(d));
      let z1 = if sub then negate(z0) else z0;
      let z = if sh < 5 then shl_int(z1, sh) else 0;
      let s = opv(t, a, db(d)) * opv(t, b, db(d)) + z;
      let s2 = if sat & sh == 0 & op_bits(c) <= 32 & db(d) <= 32
               then saturate(s, db(d), op_sx(a) | op_sx(b) | op_sx(c)) else s;
      r[t] = land_int(s2, amask(db(d)))
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(BITOP(tt, d, a, b)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then r[t] = bitopv(tt, opv(t, a, db(d)), opv(t, b, db(d)), db(d)) };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(BFI(m, d, a, b, c)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then r[t] = bfiv(m, opv(t, a, 32), opv(t, b, 32), opv(t, c, 32)) };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(BFEIL(m, d, a, b, c)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then r[t] = bfeilv(m, opv(t, a, 32), opv(t, b, 32), opv(t, c, 32)) };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(ASR(d, a, b)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then
      r[t] = land_int(asr_int(asext(opv(t, a, op_size(a)), op_size(a)), land_int(opv(t, b, 32), 127)), amask(db(d)))
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(IUNOP(op, d, a)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let v = opv(t, a, 32);
      r[t] = match op { U_popcount => popc32(v), U_bitrev => brev32(v), U_ffs => ffsv(v) }
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

/* fadd / fmul / fmadd: the exact result of the source values (fval; the 32-bit forms read fp32
 * subnormals as zero), rounded once into the destination (fres) */
function clause execute(FBIN(op, h, sat, d, a, b)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let x = fval(t, a, not_bool(h));
      let y = fval(t, b, not_bool(h));
      r[t] = fres(match op { FB_add => fv_add_exact(x, y, RN), FB_mul => fv_mul_exact(x, y) }, sat, d)
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(FMADD(h, sat, d, a, b, c)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then
      r[t] = fres(fv_fma_exact(fval(t, a, not_bool(h)), fval(t, b, not_bool(h)), fval(t, c, not_bool(h)), RN), sat, d)
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

/* rcp: the binary32 reciprocal (correctly rounded; fp32 subnormal sources read as zero, results of
 * magnitude below 2^-126 flushed), then .sat; a 16-bit destination receives that binary32 value
 * rounded to binary16.  The hardware computes an approximation within one unit in the last place of
 * this value (recorded tables of AGX-Performance-Simulator, testsuite/transcendental). */
val rcpv : fv -> fv
function rcpv(x) = match x {
  FV_NaN() => FV_NaN(),
  FV_Inf(s) => FV_Fin(s, 0, 0),
  FV_Fin(s, m, _) =>
    if m == 0 then FV_Inf(s)
    else if fv_lt(FV_Fin(false, 1, 126), fv_abs(x)) then FV_Fin(s, 0, 0)
    else decodeF(fv_div(fm_rn, 32, fv_one(), x), 32)
}

function clause execute(RCP(sat, d, a)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let y = rcpv(fval(t, a, true));
      r[t] = fres(if sat then agx_sat(y) else y, false, d)
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(CMPSEL(isf, cc, d, a, b, x, y)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then r[t] = if cmp_lane(t, isf, cc, a, b) then opv(t, x, db(d)) else opv(t, y, db(d))
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

/* shuffles (applegpu simd_shuffle*): values move between quads.  Every quad takes the OR of the low
 * two index bits of its four lanes (active or not), q, and permutes its lanes by q (idx: lane q of the
 * quad for every element; down: (i + q) & 3; up: (i - q) & 3; xor: i ^ q).  An active lane t with
 * index i reads element t & 3 of quad (t + i) >> 2 (down), (t - i) >> 2 (up), (t ^ i) >> 2 (xor), or the
 * value of quad i >> 2 (idx), and its own A when that lane is outside the SIMD-group. */
val lidx : int -> range(0, 31)
function lidx(i) = { assert(0 <= i & i < 32, "lane out of range"); i }

function clause execute(SHUFFLE(k, d, a, b)) = {
  var av : vector(32, int) = vector_init(0);
  var bv : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { av[t] = opv(t, a, 32); bv[t] = opv(t, b, 32) };
  var qv : vector(32, int) = vector_init(0);   /* element i of quad q at 4q + i */
  foreach (q from 0 to 7) {
    var qi : int = 0;
    foreach (i from 0 to 3) { qi = lor_int(qi, land_int(bv[lidx(4 * q + i)], 3)) };
    foreach (i from 0 to 3) {
      let l : int = match k {
        SH_idx => qi,
        SH_down => land_int(i + qi, 3),
        SH_up => land_int(i - qi + 4, 3),
        SH_xor => lxor_int(i, qi)
      };
      qv[lidx(4 * q + i)] = av[lidx(4 * q + l)]
    }
  };
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let i = bv[t];
      r[t] = match k {
        SH_idx => if i < 32 then qv[lidx(4 * shr_int(i, 2))] else av[t],
        SH_down => if t + i < 32 then qv[lidx(4 * shr_int(t + i, 2) + land_int(t, 3))] else av[t],
        SH_up => if t - i >= 0 then qv[lidx(4 * shr_int(t - i, 2) + land_int(t, 3))] else av[t],
        SH_xor => if lxor_int(t, i) < 32 then qv[lidx(4 * shr_int(lxor_int(t, i), 2) + land_int(t, 3))] else av[t]
      }
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

/* ---------------------------------------------------------------- execution mask stack (r0l) */
val r0l : int -> int
function r0l(t) = land_int(reg_raw(t, 0), 65535)

/* stkNew(v, kind, comparison result, n) */
val stkNew : (int, int, bool, int) -> int
function stkNew(v, kind, c, n) = match kind {
  0 => if v != 0 then v + n else if c then 0 else 1,                       /* if */
  1 => if v == 0 then n else if v == 1 then (if c then 0 else 1) else v,     /* else */
  2 => if v < n then (if c then 0 else n) else v,                           /* while */
  _ => if v - n < 0 then 0 else v - n                                       /* pop */
}

/* every lane (active or not) updates r0l; a lane is active afterwards iff its r0l is 0 */
val stack_update : (int, bool, acc, aop, aop, int) -> unit
function stack_update(kind, isf, cc, a, b, n) = {
  var nv : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    let c = kind == 3 | cmp_lane(t, isf, cc, a, b);
    nv[t] = stkNew(r0l(t), kind, c, n)
  };
  var e : bits(32) = 0x00000000;
  foreach (t from 0 to 31) {
    let i = gidx(t, 0);
    gpr[i] = gpr[i][31 .. 16] @ get_slice_int(16, nv[t], 0);
    if nv[t] == 0 then e[t] = bitone
  };
  em = e
}

function clause execute(STACKOP(k, isf, cc, a, b, n)) = {
  let kind : int = match k { SK_if => 0, SK_else => 1, SK_while => 2 };
  stack_update(kind, isf, cc, a, b, n)
}

function clause execute(POP_EXEC(n)) = stack_update(3, false, CC_eq, OImm(0), OImm(0), n)

/* jumps: the target is resolved to a byte offset relative to the jump when the program is loaded */
val pc_of_addr : int -> option(int)
function pc_of_addr(a) = {
  var r : option(int) = None();
  foreach (i from 0 to (ncode - 1)) { if get_addr(i) == a then r = Some(i) };
  r
}

function clause execute(JMP_EXEC(any, tgt)) = {
  let take = if any then em != 0x00000000 else em == 0x00000000;
  match tgt {
    JLabel(l) => fatal(concat_str("unresolved label ", l)),
    JRel(off) => {
      if take then {
        /* pc was already advanced: the jump is instruction pc - 1 */
        match pc_of_addr(get_addr(pc - 1) + off) {
          Some(p) => pc = p,
          None() => pc = ncode    /* outside the program: falls off the end */
        }
      }
    }
  }
}

function clause execute(STOP()) = status = ST_stopped

function clause execute(TG_BARRIER()) = ()

/* memory_barrier: one SIMD group in lock step with one sequential memory, so the barriers order nothing that is
 * not already ordered (the cross-core and cross-group ordering they provide is not modelled).  Only the seven
 * encodings the LLVM backend emits ran on the hardware (G14C, Apple-M2-Playground 2026-10-18-fence-cost and
 * 2026-10-11-memory-ordering: f596, f533, f59a, f59e, f503, f599, f59d); the model stops with an error on the others. */
val barrier_measured : (int, int, int) -> bool
function barrier_measured(a, b, c) = {
  /* the second byte b | a << 2 | c << 4 of the encodings: 96 33 9a 9e 03 99 9d */
  let x = b + 4 * a + 16 * c;
  x == 150 | x == 51 | x == 154 | x == 158 | x == 3 | x == 153 | x == 157
}

function clause execute(MEMORY_BARRIER(a, b, c)) =
  if not_bool(barrier_measured(a, b, c)) then fatal("memory_barrier with an unmeasured encoding")

/* trap (0x0008): PROVISIONAL.  What the hardware does when it executes `trap` has not been measured
 * (AGX-Performance-Simulator task B21); the model ends execution with its own status `trapped`, so that a test
 * can tell it from `stop`. */
function clause execute(TRAP()) = { status = ST_trapped; msg = "trap" }

/* ---------------------------------------------------------------- ballots */
function clause execute(BALLOT(isf, cc, d, a, b)) = {
  var r : int = 0;
  foreach (t from 0 to 31) { if is_active(t) & cmp_lane(t, isf, cc, a, b) then r = r + pow2(t) };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r) }
}

/* ---------------------------------------------------------------- memory */
val fsize : afmt -> int
function fsize(f) = match f { F_U8 => 1, F_U16 => 2, F_U32 => 4 }
val flog : afmt -> int
function flog(f) = match f { F_U8 => 0, F_U16 => 1, F_U32 => 2 }

/* ubase(U, P): the 64-bit base of lane t */
val ubase : (int, apair) -> int
function ubase(t, p) = raw_n(t, p.pfile, p.lo) + shl_int(raw_n(t, p.pfile, p.hi), 32)

/* idxv: the 32-bit index sign- or zero-extended according to the signedness field */
val idxv : (int, aop, bool) -> int
function idxv(t, i, sgn) = if sgn then asext(opv(t, i, 32), 32) else opv(t, i, 32)

/* ea: base + (index << (log2 size + shift)) modulo 2^64, rounded down to the element size */
val ea : (int, afmt, aop, bool, int, int) -> int
function ea(t, f, i, sgn, s, b) =
  shl_int(shr_int(trunc_int(b + shl_int(idxv(t, i, sgn), flog(f) + s), 64), flog(f)), flog(f))

/* the memory elements of an access: mask bit k selects element k (at k * size bytes); the j-th selected
 * element uses register j of the tuple that starts at R: the 32-bit registers R + j, or for a 16-bit R
 * the 16-bit registers that follow it (r5h, r6l, r6h, ...).  A load writes the element zero-extended to a
 * 32-bit register, or truncated to a 16-bit one; a store writes the register's low `size` bytes.  The
 * hardware performs the elements in order, each over all lanes; a device store's lanes go from the
 * highest down (so the lowest lane's value stays when lanes store to the same bytes), a threadgroup
 * store's from the lowest up (AGX-Performance-Simulator hardware test suite, mem_store / threadgroup). */
val mreg_code : (areg, int) -> int
function mreg_code(d, j) = if d.view == 0 then d.num + j else 1024 + 2 * d.num + d.view - 1 + j

val mreg_get : (int, int) -> int
function mreg_get(t, c) =
  if c < 1024 then reg_raw(t, c)
  else { let h = c - 1024 in land_int(shr_int(reg_raw(t, shr_int(h, 1)), 16 * land_int(h, 1)), 65535) }

val mreg_put : (int, int, int) -> unit
function mreg_put(t, c, v) =
  if c < 1024 then { let i = gidx(t, c) in gpr[i] = get_slice_int(32, v, 0) }
  else {
    let h = c - 1024 in
    let i = gidx(t, shr_int(h, 1)) in
    if land_int(h, 1) == 0 then gpr[i] = gpr[i][31 .. 16] @ get_slice_int(16, v, 0)
    else gpr[i] = get_slice_int(16, v, 0) @ gpr[i][15 .. 0]
  }

function clause execute(DEVMEM(st, f, mask, d, p, i, sgn, s, _, _)) = {
  var newp : list((int, int, int)) = [||];
  var addrs : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then addrs[t] = ea(t, f, i, sgn, s, ubase(t, p)) };
  var j : int = 0;
  foreach (k from 0 to 3) {
    if bit_int(mask, k) then {
      foreach (t from 31 downto 0) {
        if is_active(t) then {
          let a = addrs[t] + k * fsize(f);
          if st then wrBytes(SP_DEV, a, mreg_get(t, mreg_code(d, j)), fsize(f))
          else newp = (t, mreg_code(d, j), rdBytes(SP_DEV, a, fsize(f))) :: newp
        }
      };
      j = j + 1
    }
  };
  if not_bool(st) then pend = pend_append(pend, reverse_list(newp))
}

/* threadgroup accesses: byte base + 4 * size * index + size * k (the index counts four-element
 * vectors; base and index are 16-bit values: registers, uniforms or immediates; hardware test suite
 * finding of AGX-Performance-Simulator).  The model's threadgroup memory has no size: the hardware
 * reads 0 and drops stores beyond the threadgroup allocation. */
val tg_addr : (int, afmt, aop, aop) -> int
function tg_addr(t, f, base, i) = opv(t, base, 16) + 4 * fsize(f) * opv(t, i, 16)

function clause execute(TGMEM(st, f, mask, d, base, i, _)) = {
  var newp : list((int, int, int)) = [||];
  var addrs : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then addrs[t] = tg_addr(t, f, base, i) };
  var j : int = 0;
  foreach (k from 0 to 3) {
    if bit_int(mask, k) then {
      foreach (t from 0 to 31) {
        if is_active(t) then {
          let a = addrs[t] + k * fsize(f);
          if st then wrBytes(SP_TG, a, mreg_get(t, mreg_code(d, j)), fsize(f))
          else newp = (t, mreg_code(d, j), rdBytes(SP_TG, a, fsize(f))) :: newp
        }
      };
      j = j + 1
    }
  };
  if not_bool(st) then pend = pend_append(pend, reverse_list(newp))
}

/* aop(OP, O, V) */
val aopv : (aaop, int, int) -> int
function aopv(op, o, v) = match op {
  A_add => land_int(o + v, 4294967295),
  A_sub => land_int(o - v, 4294967295),
  A_umin => if o < v then o else v,
  A_umax => if o > v then o else v,
  A_imin => if sext_int(o, 32) < sext_int(v, 32) then o else trunc_int(v, 32),
  A_imax => if sext_int(o, 32) > sext_int(v, 32) then o else trunc_int(v, 32),
  A_and => land_int(o, v),
  A_or => lor_int(o, v),
  A_xor => lxor_int(o, v),
  A_xchg => v,
  /* the device-only forms are computed in DEVATOM */
  _ => o
}

/* the float32 atomic add (device atomic op 1 with bit 45, G14C): round to nearest even; denormal inputs and
 * results become zeros of their sign; every NaN result is 0x7fc00000 (2,048 measured edge and rounding pairs) */
val atom_fadd : (int, int) -> int
function atom_fadd(o, v) = fres(fv_add_exact(decodeF(flushB(o, 32), 32), decodeF(flushB(v, 32), 32), RN), false, gpr32(0))

val is_end_op : aaop -> bool
function is_end_op(op) = match op { A_end11 => true, A_end12 => true, A_end13 => true, A_end14 => true, A_end15 => true, _ => false }

/* the value operand of the pair forms: its register number (S, S + 1) */
val pair_reg : aop -> option(int)
function pair_reg(v) = match v { OReg(r) => if r.file == GPR & r.view == 3 then Some(r.num) else None(), _ => None() }

/* device atomics: the active lanes perform their read-modify-write one after the other in lane order; the
 * returned value (the old word, or the umin64 / umax64 flag) is written like a load (visible at the next wait).
 *   cmpxchg   32-bit compare-and-swap: S is the new value, S + 1 the compare value; returns the old word
 *   fadd      float32 add (atom_fadd); returns the old word.  Without a destination (red.add.f32) it was not
 *             run on the hardware: the model stops with an error
 *   umin64 / umax64   unsigned min / max of the 64-bit word at base + 4 * index with the pair S (lo), S + 1 (hi);
 *             the destination receives 1 if memory changed, else 0.  Not measured, so the model stops with an
 *             error: a word that is not 8-byte aligned, and a destination when the source equals the memory word
 *   end11 .. end15   no memory effect; the SIMD group ends, inactive lanes included (as `stop` in this
 *             one-SIMD-group model); earlier stores have landed
 * mode (bits 45-46) has no further effect. */
function clause execute(DEVATOM(op, d, p, i, sgn, v, _, _)) = {
  if is_end_op(op) then status = ST_stopped
  else if op == A_fadd & is_none(d) then fatal("device_atomic fadd without a destination (red.add.f32) is not measured")
  else {
    let wide = op == A_umin64 | op == A_umax64;
    let pairop = wide | op == A_cmpxchg;
    let s : int = match pair_reg(v) { Some(n) => n, None() => 0 };
    if pairop & is_none(pair_reg(v)) then fatal("device_atomic cmpxchg / umin64 / umax64 needs a register pair")
    else {
      var newp : list((int, int, int)) = [||];
      var vals : vector(32, int) = vector_init(0);
      var cmps : vector(32, int) = vector_init(0);
      var addrs : vector(32, int) = vector_init(0);
      foreach (t from 0 to 31) {
        if is_active(t) then {
          addrs[t] = ea(t, F_U32, i, sgn, 0, ubase(t, p));
          if wide then vals[t] = reg_raw(t, s) + shl_int(reg_raw(t, s + 1), 32)
          else if op == A_cmpxchg then { vals[t] = reg_raw(t, s); cmps[t] = reg_raw(t, s + 1) }
          else vals[t] = opv(t, v, 32)
        }
      };
      foreach (t from 0 to 31) {
        if is_active(t) & status != ST_fault then {
          if wide then {
            if land_int(addrs[t], 7) != 0 then fatal("device_atomic umin64 / umax64 at an address that is not 8-byte aligned (not measured)")
            else {
              let o = rdBytes(SP_DEV, addrs[t], 8);
              let n = if op == A_umin64 then (if vals[t] < o then vals[t] else o) else (if vals[t] > o then vals[t] else o);
              if is_some(d) & vals[t] == o then fatal("device_atomic umin64 / umax64: the changed flag when the source equals memory is not measured")
              else {
                wrBytes(SP_DEV, addrs[t], n, 8);
                match d { Some(r) => newp = (t, r.num, if n != o then 1 else 0) :: newp, None() => () }
              }
            }
          } else {
            let o = rdBytes(SP_DEV, addrs[t], 4);
            let n : int = match op {
              A_cmpxchg => if o == cmps[t] then vals[t] else o,
              A_fadd => atom_fadd(o, vals[t]),
              _ => aopv(op, o, vals[t])
            };
            wrBytes(SP_DEV, addrs[t], n, 4);
            match d { Some(r) => newp = (t, r.num, o) :: newp, None() => () }
          }
        }
      };
      if status != ST_fault then pend = pend_append(pend, reverse_list(newp))
    }
  }
}

/* threadgroup atomics: the word at byte base + 16 * index (as a threadgroup access with 32-bit
 * elements); the old value is written to the 32-bit register that holds D when the instruction
 * completes (no wait: applegpu notes that threadgroup atomics have no wait group) */
function clause execute(TGATOM(op, d, base, i, v)) = {
  match op {
    A_add => (), A_sub => (), A_xchg => (), A_umin => (), A_imin => (), A_umax => (), A_imax => (), A_and => (), A_or => (), A_xor => (),
    _ => fatal("threadgroup_atomic: only the ten 32-bit operations are modelled")
  };
  var olds : vector(32, int) = vector_init(0);
  var vals : vector(32, int) = vector_init(0);
  var addrs : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then { addrs[t] = opv(t, base, 16) + 16 * opv(t, i, 16); vals[t] = opv(t, v, 32) }
  };
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let o = rdBytes(SP_TG, addrs[t], 4);
      wrBytes(SP_TG, addrs[t], aopv(op, o, vals[t]), 4);
      olds[t] = o
    }
  };
  match d {
    Some(r) => foreach (t from 0 to 31) { if is_active(t) then mreg_put(t, r.num, olds[t]) },
    None() => ()
  }
}

function clause execute(WAIT(_)) = {
  var l = pend;
  while is_cons(l) do {
    match l {
      (t, r, v) :: rest => { mreg_put(t, r, v); l = rest },
      [||] => ()
    }
  };
  pend = [||]
}

end execute

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

/* raw(G, U, T, R): the 32-bit container */
val raw : (int, areg) -> int
function raw(t, r) = {
  if r.file == UNI then {
    unsigned(get_upr(r.num))
  } else reg_raw(t, r.num)
}

val raw_n : (int, rfile, int) -> int
function raw_n(t, f, n) = raw(t, { gpr32(n) with file = f })

val view_bits : areg -> int
function view_bits(r) = match r.view { 0 => 32, 3 => 64, _ => 16 }

/* val(G, U, T, R) with the encoding's sign-extension modifier */
val rval : (int, areg) -> int
function rval(t, r) = {
  let v : int = match r.view {
    0 => raw(t, r),
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

/* opv(G, U, T, A, B): registers read their view, integer immediates are broadcast (truncated to
 * B bits), float immediates are encoded at B bits */
val opv : (int, aop, int) -> int
function opv(t, a, b) = match a {
  OReg(r) => rval(t, r),
  OImm(i) => land_int(i, amask(b)),
  OFImm(f) => encodeF(f64v(f), b)
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

/* cmpv(S, A, B) of agx.k, where S is "icmpsel <cc>" or "fcmpsel <cc>".  The K rules select on the
 * suffix of S: integer conditions compare the unsigned / signed 32-bit views (for both
 * instruction kinds), `eq` / `ne` hold only for integer instructions, the f-conditions compare
 * the binary32 values (false on NaN) for float instructions; for integer instructions `feq` and
 * `fne` end in "eq" / "ne" and compare as integers, the other f-conditions are false. */
val cmpv : (bool, acc, int, int) -> bool
function cmpv(isf, cc, a, b) = match cc {
  CC_eq => not_bool(isf) & a == b,
  CC_ne => not_bool(isf) & a != b,
  CC_seq => a == b,
  CC_sne => a != b,
  CC_ult => a < b,
  CC_ule => a <= b,
  CC_ugt => a > b,
  CC_uge => a >= b,
  CC_slt => sext_int(a, 32) < sext_int(b, 32),
  CC_sle => sext_int(a, 32) <= sext_int(b, 32),
  CC_sgt => sext_int(a, 32) > sext_int(b, 32),
  CC_sge => sext_int(a, 32) >= sext_int(b, 32),
  _ => {
    if isf then fcmpv(cc, decodeF(a, 32), decodeF(b, 32))
    else match cc { CC_feq => a == b, CC_fne => a != b, _ => false }
  }
}

/* the compared values of lane t: 32-bit views (float operands with their modifiers) */
val cmp_lane : (int, bool, acc, aop, aop) -> bool
function cmp_lane(t, isf, cc, a, b) =
  if isf then cmpv(isf, cc, fopv(t, a, 32), fopv(t, b, 32)) else cmpv(isf, cc, opv(t, a, 32), opv(t, b, 32))

/* ---------------------------------------------------------------- integer helpers */
val asext : (int, int) -> int
function asext(v, w) = sext_int(v, w)

val bitopv : (int, int, int, int) -> int
function bitopv(tt, a, b, w) = {
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
val fz : bool -> fmods
function fz(sat) = { fm_ftz with sat = sat }
val nofz : bool -> fmods
function nofz(sat) = { fm_rn with sat = sat }

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

function clause execute(IMADD(sub, sat, d, a, b, c, sh)) = {
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
      r[t] = land_int(asr_int(asext(opv(t, a, db(d)), db(d)), land_int(opv(t, b, db(d)), 127)), amask(db(d)))
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

/* f32 results flush subnormals to zero (FZ() = SetItem("ftz")), f16 results do not */
function clause execute(FBIN(op, h, sat, d, a, b)) = {
  var r : vector(32, int) = vector_init(0);
  let w = if h then 16 else 32;
  let m = if h then nofz(sat) else fz(sat);
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let x = fopv(t, a, w);
      let y = fopv(t, b, w);
      r[t] = match op { FB_add => fpAdd(m, w, x, y), FB_mul => fpMul(m, w, x, y) }
    }
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(FMADD(h, sat, d, a, b, c)) = {
  var r : vector(32, int) = vector_init(0);
  let w = if h then 16 else 32;
  let m = if h then nofz(sat) else fz(sat);
  foreach (t from 0 to 31) {
    if is_active(t) then r[t] = fpFma(m, w, fopv(t, a, w), fopv(t, b, w), fopv(t, c, w))
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

/* rcp: fpDiv(FZ(), 32, 1.0, A) */
function clause execute(RCP(sat, d, a)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) { if is_active(t) then r[t] = fpDiv(fz(sat), 32, 1065353216, fopv(t, a, 32)) };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(CMPSEL(isf, cc, d, a, b, x, y)) = {
  var r : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then r[t] = if cmp_lane(t, isf, cc, a, b) then opv(t, x, db(d)) else opv(t, y, db(d))
  };
  foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
}

function clause execute(SHUFFLE(k, d, a, b)) = {
  var r : vector(32, int) = vector_init(0);
  match a {
    OReg(ra) => {
      foreach (t from 0 to 31) {
        if is_active(t) then {
          let i = opv(t, b, 32);
          r[t] = match k {
            SH_idx => if i < 32 then rval(i, ra) else 0,
            SH_xor => rval(lxor_int(t, land_int(i, 31)), ra),
            SH_up => if t - i >= 0 then rval(t - i, ra) else rval(t, ra),
            SH_down => if t + i < 32 then rval(t + i, ra) else rval(t, ra)
          }
        }
      };
      foreach (t from 0 to 31) { if is_active(t) then put(t, d, r[t]) }
    },
    _ => fatal("simd_shuffle of an immediate")
  }
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

/* the memory elements of an access: mask bit k selects element k, the j-th selected element uses
 * register R + j (a mask 2^N - 1 is the K model's N consecutive elements) */
function clause execute(DEVMEM(st, f, mask, d, p, i, sgn, s, _, _)) = {
  var newp : list((int, int, int)) = [||];
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let a = ea(t, f, i, sgn, s, ubase(t, p));
      var j : int = 0;
      foreach (k from 0 to 3) {
        if bit_int(mask, k) then {
          if st then wrBytes(SP_DEV, a + k * fsize(f), reg_raw(t, d.num + j), fsize(f))
          else newp = (t, d.num + j, rdBytes(SP_DEV, a + k * fsize(f), fsize(f))) :: newp;
          j = j + 1
        }
      }
    }
  };
  if not_bool(st) then pend = pend_append(pend, reverse_list(newp))
}

function clause execute(TGMEM(st, f, mask, d, base, i, s)) = {
  var newp : list((int, int, int)) = [||];
  /* stores: the K model writes all lanes' values (lane order) before any load of the same
   * instruction; one instruction is either a load or a store */
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let a = ea(t, f, i, false, s, opv(t, base, 32));
      var j : int = 0;
      foreach (k from 0 to 3) {
        if bit_int(mask, k) then {
          if st then wrBytes(SP_TG, a + k * fsize(f), reg_raw(t, d.num + j), fsize(f))
          else newp = (t, d.num + j, rdBytes(SP_TG, a + k * fsize(f), fsize(f))) :: newp;
          j = j + 1
        }
      }
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
  A_xchg => v
}

/* atomics: the active lanes perform their read-modify-write one after the other in lane order;
 * the old value is returned like a load (visible at the next wait) */
function clause execute(DEVATOM(op, d, p, i, sgn, v, _)) = {
  var newp : list((int, int, int)) = [||];
  var vals : vector(32, int) = vector_init(0);
  var addrs : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then { addrs[t] = ea(t, F_U32, i, sgn, 0, ubase(t, p)); vals[t] = opv(t, v, 32) }
  };
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let o = rdBytes(SP_DEV, addrs[t], 4);
      wrBytes(SP_DEV, addrs[t], aopv(op, o, vals[t]), 4);
      match d { Some(r) => newp = (t, r.num, o) :: newp, None() => () }
    }
  };
  pend = pend_append(pend, reverse_list(newp))
}

function clause execute(TGATOM(op, d, base, i, v)) = {
  var newp : list((int, int, int)) = [||];
  var vals : vector(32, int) = vector_init(0);
  var addrs : vector(32, int) = vector_init(0);
  foreach (t from 0 to 31) {
    if is_active(t) then { addrs[t] = ea(t, F_U32, i, false, 0, opv(t, base, 32)); vals[t] = opv(t, v, 32) }
  };
  foreach (t from 0 to 31) {
    if is_active(t) then {
      let o = rdBytes(SP_TG, addrs[t], 4);
      wrBytes(SP_TG, addrs[t], aopv(op, o, vals[t]), 4);
      match d { Some(r) => newp = (t, r.num, o) :: newp, None() => () }
    }
  };
  pend = pend_append(pend, reverse_list(newp))
}

/* wait: apply the pending loads in issue order */
function clause execute(WAIT(_)) = {
  var l = pend;
  while is_cons(l) do {
    match l {
      (t, r, v) :: rest => { gpr[gidx(t, r)] = get_slice_int(32, v, 0); l = rest },
      [||] => ()
    }
  };
  pend = [||]
}

end execute

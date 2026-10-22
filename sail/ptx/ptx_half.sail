/* =============================================================================
 * PTX-EXEC-HALF (packed half pairs, f32x2, mixed precision, cvt to pairs, cvt.pack),
 * PTX-EXEC-MEMQ (cache qualifiers, packed and vector atomics) and PTX-EXEC-VIDEO.
 * ============================================================================= */

val isHalfPair : ty -> bool
function isHalfPair(t) = t.k == "f16x2" | t.k == "bf16x2"

/* laneTy of PTX-EXEC-INTX / -HALF / -MEMQ */
val laneTyH : ty -> ty
function laneTyH(t) = match t.k { "f16x2" => mkty("f", 16), "bf16x2" => mkty("bf", 16), _ => laneTy(t) }

val lane32 : (int, int) -> int
function lane32(v, i) = land_int(asr_int(v, 32 * i), 4294967295)
val pack32 : (int, int) -> int
function pack32(l0, l1) = lor_int(land_int(l0, 4294967295), shl_int(land_int(l1, 4294967295), 32))

val isNarrowF : ty -> bool
function isNarrowF(t) = ty_eq(t, mkty("f", 16)) | ty_eq(t, mkty("bf", 16))
val isNarrowPair : ty -> bool
function isNarrowPair(t) = ty_eq(t, mkty("f16x2", 32)) | ty_eq(t, mkty("bf16x2", 32))

val withRz : list(string) -> list(string)
function withRz(m) = match m { [||] => [|"rz"|], h :: t => if h == "rn" | h == "rm" | h == "rp" then withRz(t) else h :: withRz(t) }

let F32 : ty = struct { k = "f", w = 32 }
val up32 : (ty, int) -> int
function up32(t, v) = cvt([||], F32, t, v)

val gx_half : (opcode, list(xop)) -> bool
function gx_half(o, ops) = {
  let op = o.base;
  let m = o.mods;
  let n = ntys(o);
  if n == 1 then {
    let t = ty1(o);
    match ops {
      [|d, a, b|] => {
        if isHalfPair(t) & has([|"add", "sub", "mul", "min", "max"|], op) then {
          let x = rd(a, t); let y = rd(b, t); let lt = laneTyH(t);
          wr(d, pack16(bin(op, m, lt, lane16(x, 0), lane16(y, 0)), bin(op, m, lt, lane16(x, 1), lane16(y, 1)))); true
        } else if ty_eq(t, mkty("f32x2", 64)) & has([|"add", "sub", "mul"|], op) then {
          let x = rd(a, t); let y = rd(b, t);
          wr(d, pack32(bin(op, m, F32, lane32(x, 0), lane32(y, 0)), bin(op, m, F32, lane32(x, 1), lane32(y, 1)))); true
        } else false
      },
      [|d, a, b, c|] => {
        if isHalfPair(t) & (op == "fma" | op == "mad") then {
          let x = rd(a, t); let y = rd(b, t); let z = rd(c, t); let lt = laneTyH(t);
          wr(d, pack16(tern(op, m, lt, lane16(x, 0), lane16(y, 0), lane16(z, 0)), tern(op, m, lt, lane16(x, 1), lane16(y, 1), lane16(z, 1)))); true
        } else if ty_eq(t, mkty("f32x2", 64)) & op == "fma" then {
          let x = rd(a, t); let y = rd(b, t); let z = rd(c, t);
          wr(d, pack32(tern("fma", m, F32, lane32(x, 0), lane32(y, 0), lane32(z, 0)), tern("fma", m, F32, lane32(x, 1), lane32(y, 1), lane32(z, 1)))); true
        } else false
      },
      [|d, a|] => {
        if isHalfPair(t) & has([|"neg", "abs", "ex2", "tanh"|], op) then {
          let x = rd(a, t); let lt = laneTyH(t);
          wr(d, pack16(un(op, m, lt, lane16(x, 0)), un(op, m, lt, lane16(x, 1)))); true
        } else false
      },
      _ => false
    }
  } else if n == 2 then {
    let dt = ty_nth(o.tys, 0);
    let aty = ty_nth(o.tys, 1);
    match ops {
      [|d, a, x|] => {
        if ty_eq(dt, F32) & isNarrowF(aty) & has([|"add", "sub", "mul"|], op) then {
          wr(d, bin(op, m, F32, up32(aty, rd(a, aty)), rd(x, F32))); true
        } else false
      },
      [|d, a, b, x|] => {
        if op == "fma" & ty_eq(dt, F32) & isNarrowF(aty) then {
          wr(d, tern("fma", m, F32, up32(aty, rd(a, aty)), up32(aty, rd(b, aty)), rd(x, F32))); true
        } else false
      },
      _ => false
    }
  } else if n == 3 then {
    let dt = ty_nth(o.tys, 0);
    let aty = ty_nth(o.tys, 1);
    let ct = ty_nth(o.tys, 2);
    match ops {
      [|d, a, x|] => {
        let f32x2 = mkty("f32x2", 64);
        if ty_eq(dt, f32x2) & isNarrowPair(aty) & ty_eq(ct, f32x2) & has([|"add", "sub", "mul"|], op) then {
          let va = rd(a, aty); let vx = rd(x, ct); let lt = laneTyH(aty);
          wr(d, pack32(bin(op, m, F32, up32(lt, lane16(va, 0)), lane32(vx, 0)), bin(op, m, F32, up32(lt, lane16(va, 1)), lane32(vx, 1)))); true
        } else if isNarrowPair(dt) & ty_eq(aty, f32x2) & ty_eq(ct, f32x2) & has([|"add", "sub", "mul"|], op) then {
          let va = rd(a, aty); let vx = rd(x, ct); let lt = laneTyH(dt);
          wr(d, pack16(cvt([|"rz"|], lt, F32, bin(op, withRz(m), F32, lane32(va, 0), lane32(vx, 0))),
                       cvt([|"rz"|], lt, F32, bin(op, withRz(m), F32, lane32(va, 1), lane32(vx, 1))))); true
        } else if op == "mul" & isNarrowPair(dt) & ty_eq(dt, aty) & isNarrowPair(ct) & not_bool(ty_eq(dt, ct)) then {
          let va = rd(a, aty); let vx = rd(x, ct); let lt = laneTyH(dt); let lc = laneTyH(ct);
          wr(d, pack16(bin("mul", m, lt, lane16(va, 0), cvt([||], lt, lc, lane16(vx, 0))),
                       bin("mul", m, lt, lane16(va, 1), cvt([||], lt, lc, lane16(vx, 1))))); true
        } else false
      },
      _ => false
    }
  } else if n == 4 then {
    let dt = ty_nth(o.tys, 0);
    let aty = ty_nth(o.tys, 1);
    let bt = ty_nth(o.tys, 2);
    let ct = ty_nth(o.tys, 3);
    let f32x2 = mkty("f32x2", 64);
    match ops {
      [|d, a, b, x|] => {
        if op == "fma" & ty_eq(dt, f32x2) & isNarrowPair(aty) & ty_eq(bt, f32x2) & ty_eq(ct, f32x2) then {
          let va = rd(a, aty); let vb = rd(b, aty); let vx = rd(x, ct); let lt = laneTyH(aty);
          wr(d, pack32(tern("fma", m, F32, up32(lt, lane16(va, 0)), up32(lt, lane16(vb, 0)), lane32(vx, 0)),
                       tern("fma", m, F32, up32(lt, lane16(va, 1)), up32(lt, lane16(vb, 1)), lane32(vx, 1)))); true
        } else false
      },
      _ => false
    }
  } else false
}

val laneTrue : ty -> int
function laneTrue(t) = match t.k { "f16x2" => 15360, "bf16x2" => 16256, _ => 65535 }

/* the lane results of a packed half compare: .ftz flushes subnormal lane inputs, then BoolOp with c when present */
val halfCmp : (list(string), ty, int, int, list(xop)) -> (int, int)
function halfCmp(m, t, x, y, rest) = {
  let lt = laneTyH(t);
  let c0 = cmp(cmpOp(m), lt, ftzSrc(m, lt, lane16(x, 0)), ftzSrc(m, lt, lane16(y, 0)));
  let c1 = cmp(cmpOp(m), lt, ftzSrc(m, lt, lane16(x, 1)), ftzSrc(m, lt, lane16(y, 1)));
  match rest {
    [|c|] => (bop(c0, boolOp(m), rd(c, PRED)), bop(c1, boolOp(m), rd(c, PRED))),
    _ => (c0, c1)
  }
}

/* opcode-specific rules of PTX-EXEC-HALF (setp / set on pairs, cvt to pairs, cvt.pack) */
val exec_half : (opcode, list(xop)) -> bool
function exec_half(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    /* setp.CmpOp{.BoolOp}{.ftz}.{f16x2,bf16x2} p|q, a, b{, {!}c}: p = BoolOp(t[0], c), q = BoolOp(t[1], c); either
     * destination may be the sink '_' (9.7.7.2) */
    ("setp", XS(SPIPE(k1, p, k2, q)) :: a :: b :: rest) => {
      let t = ty1(o);
      if ntys(o) == 1 & isHalfPair(t) & xop_count(rest) <= 1 then {
        let (r0, r1) = halfCmp(m, t, rd(a, t), rd(b, t), rest);
        wr_sop(if k1 then SR(p) else SN(p), r0);
        wr_sop(if k2 then SR(q) else SN(q), r1);
        true
      } else false
    },
    /* set.CmpOp{.BoolOp}{.ftz}.dtype.{f16x2,bf16x2} d, a, b{, {!}c}: d[i] = BoolOp(t[i], c) (9.7.7.1) */
    ("set", d :: a :: b :: rest) => {
      if ntys(o) == 2 & isHalfPair(ty_nth(o.tys, 1)) & xop_count(rest) <= 1 then {
        let dt = ty_nth(o.tys, 0); let st = ty_nth(o.tys, 1);
        let (r0, r1) = halfCmp(m, st, rd(a, st), rd(b, st), rest);
        wr(d, pack16(r0 * laneTrue(dt), r1 * laneTrue(dt)));
        true
      } else false
    },
    ("cvt", [|d, a, b|]) => {
      if ntys(o) == 2 & isHalfPair(ty_nth(o.tys, 0)) & isF(ty_nth(o.tys, 1)) then {
        let dt = ty_nth(o.tys, 0); let st = ty_nth(o.tys, 1);
        wr(d, pack16(cvt(m, laneTyH(dt), st, rd(b, st)), cvt(m, laneTyH(dt), st, rd(a, st))));
        true
      } else if ntys(o) == 2 & has(m, "pack") then {
        /* cvt.pack.sat.{u16,s16}.s32 d, a, b (ISA 9.7.9.23): d = sat(a) << 16 | sat(b) */
        let pt = ty_nth(o.tys, 0); let st = ty_nth(o.tys, 1);
        wr(d, lor_int(shl_int(land_int(clamp(ival(rd(a, st), st), pt), mask(wid(pt))), wid(pt)), land_int(clamp(ival(rd(b, st), st), pt), mask(wid(pt)))));
        true
      } else false
    },
    ("cvt", [|d, a, b, x|]) => {
      if ntys(o) == 2 & has(m, "rs") & isHalfPair(ty_nth(o.tys, 0)) & ty_eq(ty_nth(o.tys, 1), F32) then {
        /* cvt.rs{.relu}{.satfinite}.{f16x2,bf16x2}.f32 d, a, b, rbits (ISA 9.7.9.22, Figures 38 / 39): random bits
         * rbits[28:16] / rbits[12:0] (13 bits, .f16) or rbits[31:16] / rbits[15:0] (.bf16) for a / b */
        let lt = laneTyH(ty_nth(o.tys, 0)); let w = fw(lt); let r = rd(x, U32);
        let k = if w == 16 then 13 else 16;
        let fm = fmods_of(m);
        wr(d, pack16(post(fm, w, fround_rs(fv(rd(b, F32), F32), prec(w), ebits(w), land_int(r, mask(k)), k)),
                     post(fm, w, fround_rs(fv(rd(a, F32), F32), prec(w), ebits(w), land_int(asr_int(r, 16), mask(k)), k))));
        true
      } else if ntys(o) == 3 & has(m, "pack") then {
        let pt = ty_nth(o.tys, 0); let st = ty_nth(o.tys, 1);
        wr(d, trunc(lor_int(lor_int(shl_int(rd(x, U32), 2 * wid(pt)), shl_int(land_int(clamp(ival(rd(a, st), st), pt), mask(wid(pt))), wid(pt))),
                            land_int(clamp(ival(rd(b, st), st), pt), mask(wid(pt)))), 32));
        true
      } else false
    },
    _ => false
  }
}

/* ---------------------------------------------------------------- PTX-EXEC-MEMQ */
val drop_mod : (list(string), string) -> list(string)
function drop_mod(m, s) = match m { [||] => [||], h :: t => if h == s then drop_mod(t, s) else h :: drop_mod(t, s) }

val drop_last : list(xop) -> list(xop)
function drop_last(l) = match l { [||] => [||], [|_|] => [||], h :: t => h :: drop_last(t) }

/* atom / red on a vector of elements: each element in order of increasing address */
val atv : (list(sop), loc, string, ty, list(int)) -> unit
function atv(ds, l, op, t, vs) = match l {
  BadLoc(e) => fault(e, 0),
  _ => match vs {
    [||] => (),
    v :: vr => {
      let old = memop_do(l, AtOp(atOpG(l, op, t), t, v, 0));
      if not_bool(stuck) then match ds {
        d :: dr => { ldwb(XS(d), t, old); atv(dr, loc_add(l, bytes(t)), op, t, vr) },
        [||] => atv([||], loc_add(l, bytes(t)), op, t, vr)
      }
    }
  }
}

val rdList : (list(sop), ty) -> list(int)
function rdList(l, t) = match l { [||] => [||], h :: r => rd_sop(h, t) :: rdList(r, t) }

val exec_memq_prio : (opcode, list(xop)) -> bool
function exec_memq_prio(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    /* a vector atom / red is one access of (elements x element size) bytes for the alignment rule (6.4.1) */
    ("atom", [|XV(ds), XB([|a|], [||]), XV(bs)|]) => {
      let t = ty1(o); let l = ea(a, spaceOf(m));
      if vecAligned(l, sop_count(bs) * bytes(t)) then atv(ds, l, atomOp(m), t, rdList(bs, t)); true
    },
    ("red", [|XB([|a|], [||]), XV(bs)|]) => {
      let t = ty1(o); let l = ea(a, spaceOf(m));
      if vecAligned(l, sop_count(bs) * bytes(t)) then atv([||], l, atomOp(m), t, rdList(bs, t)); true
    },
    _ => false
  }
}

val exec_memq : (opcode, list(xop)) -> bool
function exec_memq(o, ops) = match (o.base, ops) {
  /* "The valid range of values for the operand fraction is (0.0,.., 1.0]" (9.7.9.19): reported under .isacheck */
  ("createpolicy", [|d, fr|]) if isachk & has(o.mods, "fractional") & not_bool(fv_lt(FV_Fin(false, 0, 0), decodeF(rd(fr, F32), 32)) & not_bool(fv_lt(fv_one(), decodeF(rd(fr, F32), 32)))) => {
    fault("createpolicy: fraction outside (0.0, 1.0] [PTX ISA 9.4 9.7.10.21]", 0); true
  },
  ("createpolicy", d :: _) => { wr(d, 0); true },
  _ => false
}

val gx_memq : (opcode, list(xop)) -> bool
function gx_memq(o, ops) = {
  let m = o.mods;
  if has(m, "L2::cache_hint") & not_bool(has(m, "bulk") | has(m, "tensor")) & has([|"ld", "st", "atom", "red", "cp"|], o.base) then {
    exec_dispatch({ o with mods = drop_mod(m, "L2::cache_hint") }, drop_last(ops)); true
  } else if o.base == "discard" | o.base == "applypriority" then {
    /* discard "behaves like a weak write of an unstable indeterminate value" (9.7.9.18): the old bytes are one such
       value, so memory is left as it is; applypriority only sets a cache eviction priority (9.7.9.17).  For both, "If
       the specified address does not fall within the address window of .global state space then the behavior is
       undefined" and "a must be aligned to 128 bytes": reported under .ub strict */
    match ops {
      XB([|a|], [||]) :: _ => if ub then match ea(a, if has(m, "global") then "global" else "generic") {
        Loc(sp, off) => if sp != "global" then fault(concat_str(o.base, ": address outside the .global window"), off)
                        else if tmod_int(off, 128) != 0 then fault(concat_str(o.base, ": address not aligned to 128 bytes"), off),
        _ => fault(concat_str(o.base, ": address outside the .global window"), 0)
      },
      _ => ()
    };
    true
  } else false
}

/* ---------------------------------------------------------------- PTX-EXEC-VIDEO */
val opName : xop -> string
function opName(o) = opStr(o)

val selOf : string -> string
function selOf(s) = {
  let n = string_length(s);
  if n > 3 then { let x = sdrop(s, n - 3); if has([|".b0", ".b1", ".b2", ".b3", ".h0", ".h1"|], x) then x else "" } else ""
}

val ext : (ty, int, int, int) -> int
function ext(t, v, o, w) = if isS(t) then sext(asr_int(v, o), w) else land_int(asr_int(v, o), mask(w))

val vpart : (string, ty, int) -> int
function vpart(s, t, v) = match s {
  ".b0" => ext(t, v, 0, 8), ".b1" => ext(t, v, 8, 8), ".b2" => ext(t, v, 16, 8), ".b3" => ext(t, v, 24, 8),
  ".h0" => ext(t, v, 0, 16), ".h1" => ext(t, v, 16, 16),
  "" => ext(t, v, 0, min_int(wid(t), 32)),
  _ => ext(t, v, 0, 32)
}

val vrd : (xop, ty) -> int
function vrd(o, t) = {
  let n = opName(o);
  let s = selOf(n);
  if s != "" then vpart(s, t, rreg(stake(n, string_length(n) - 3))) else vpart("", t, rd(o, U32))
}

val secOp : list(string) -> string
function secOp(m) = pick(m, [|"add", "min", "max"|], "")
val secApply : (string, ty, int, int) -> int
function secApply(op, t, v, c) = match op { "add" => v + ival(c, t), "min" => min_int(v, ival(c, t)), "max" => max_int(v, ival(c, t)), _ => v }
val vfinish : (list(string), ty, int, int) -> int
function vfinish(m, dt, v, c) = trunc(secApply(secOp(m), dt, if has(m, "sat") then clamp(v, dt) else v, c), wid(dt))

val vcore : (string, int, int) -> int
function vcore(op, a, b) = match op { "vadd" => a + b, "vsub" => a - b, "vabsdiff" => abs_int(a - b), "vmin" => min_int(a, b), "vmax" => max_int(a, b), _ => a }

val vshift : (bool, bool, int, int) -> int
function vshift(left, clampm, a, b) = {
  /* ISA vshl/vshr: if (mode == .clamp && tb > 32) tb = 32, on the full selected value */
  let s = if clampm then min_int(land_int(b, 4294967295), 32) else land_int(b, 31);
  if left then shl_int(a, s) else asr_int(a, s)
}

val cmpV : (string, int, int) -> int
function cmpV(op, a, b) = b2i(cmpI(op, a, b, a, b))
val vsetRes : (list(string), int, int, int) -> int
function vsetRes(m, a, b, c) = trunc(secApply(secOp(m), U32, cmpV(cmpOp(m), a, b), c), 32)

val allDig : string -> bool
function allDig(s) = { var ok : bool = true; var x = s; while string_length(x) > 0 do { if not_bool(is_dec_digit(stake(x, 1))) then ok = false; x = sdrop(x, 1) }; ok }

/* selector suffix .h10 / .b7654 (after the last '.') */
val selSfx : string -> string
function selSfx(s) = {
  var i : int = string_length(s) - 1;
  while i >= 0 & stake(sdrop(s, i), 1) != "." do i = i - 1;
  if i >= 0 & string_length(s) - i > 2 then {
    let c = stake(sdrop(s, i + 1), 1);
    if (c == "h" | c == "b") & allDig(sdrop(s, i + 2)) then sdrop(s, i + 1) else ""
  } else ""
}
val baseStr : string -> string
function baseStr(s) = if selSfx(s) == "" then s else stake(s, string_length(s) - string_length(selSfx(s)) - 1)
val rawV : xop -> int
function rawV(o) = if selSfx(opName(o)) != "" then rreg(baseStr(opName(o))) else rd(o, U32)
val selOr : (xop, string) -> string
function selOr(o, d) = if selSfx(opName(o)) != "" then selSfx(opName(o)) else d

val isScalarVop : string -> bool
function isScalarVop(op) = has([|"vadd", "vsub", "vabsdiff", "vmin", "vmax", "vshl", "vshr"|], op)

val satLane : (bool, ty, int, int) -> int
function satLane(sat, dt, w, v) =
  if not_bool(sat) then v
  else if isS(dt) then min_int(max_int(v, negate(pow2(w - 1))), pow2(w - 1) - 1)
  else min_int(max_int(v, 0), mask(w))

val vmergeW : string -> int
function vmergeW(sel) = if stake(sel, 1) == "h" then 16 else 8
val vmergePos : string -> int
function vmergePos(sel) = hex_digit(stake(sdrop(sel, 1), 1)) * vmergeW(sel)
val vmerge : (ty, bool, string, int, int) -> int
function vmerge(dt, sat, sel, v, cr) =
  lor_int(land_int(land_int(cr, mask(32)), lnot_int(shl_int(mask(vmergeW(sel)), vmergePos(sel)))),
          shl_int(land_int(satLane(sat, dt, vmergeW(sel), v), mask(vmergeW(sel))), vmergePos(sel)))

val isSimd : string -> bool
function isSimd(op) = has([|"vadd2", "vsub2", "vavrg2", "vabsdiff2", "vmin2", "vmax2", "vset2", "vadd4", "vsub4", "vavrg4", "vabsdiff4", "vmin4", "vmax4", "vset4"|], op)
val nlanes : string -> int
function nlanes(op) = if sdrop(op, string_length(op) - 1) == "2" then 2 else 4
val defSel : (string, int) -> string
function defSel(w, n) = match (w, n) {
  ("a", 2) => "h10", ("b", 2) => "h32", ("m", 2) => "h10", ("a", 4) => "b3210", ("b", 4) => "b7654", ("m", 4) => "b3210", _ => ""
}
val simdOp : (string, int, int, list(string)) -> int
function simdOp(op, a, b, m) = match op {
  "vadd" => a + b, "vsub" => a - b,
  "vavrg" => if a + b >= 0 then asr_int(a + b + 1, 1) else asr_int(a + b, 1),
  "vabsdiff" => abs_int(a - b), "vmin" => min_int(a, b), "vmax" => max_int(a, b),
  "vset" => cmpV(cmpOp(m), a, b),
  _ => a
}
val elemOf : (ty, int, int, int, int) -> int
function elemOf(t, a, b, k, w) = if k < tdiv_int(32, w) then ext(t, a, k * w, w) else ext(t, b, (k - tdiv_int(32, w)) * w, w)
val selDigit : (string, int, int) -> int
function selDigit(s, l, n) = if string_length(s) >= 1 + n then hex_digit(stake(sdrop(s, 1 + (n - 1 - l)), 1)) else l
val maskHas : (string, int) -> bool
function maskHas(mk, l) = {
  var x = sdrop(mk, 1);
  var r : bool = false;
  while string_length(x) > 0 do { if stake(x, 1) == dec_str(l) then r = true; x = sdrop(x, 1) };
  r
}
val tLane : (string, list(string), ty, ty, ty, int, int, string, string, int, int) -> int
function tLane(op, m, dt, aty, bt, a, b, as_, bs, l, n) =
  satLane(has(m, "sat") & op != "vset", dt, tdiv_int(32, n),
          simdOp(op, elemOf(aty, a, b, selDigit(as_, l, n), tdiv_int(32, n)), elemOf(bt, a, b, selDigit(bs, l, n), tdiv_int(32, n)), m))

val simdRes : (string, list(string), ty, ty, ty, int, int, int, string, string, string) -> int
function simdRes(op0, m, dt, aty, bt, a, b, cv, as_, bs, mk) = {
  let n = nlanes(op0);
  let op = stake(op0, string_length(op0) - 1);
  let w = tdiv_int(32, n);
  if has(m, "add") then {
    var acc : int = 0;
    foreach (l from 0 to (n - 1)) { if maskHas(mk, l) then acc = acc + tLane(op, m, dt, aty, bt, a, b, as_, bs, l, n) };
    trunc(cv + acc, 32)
  } else {
    var r : int = 0;
    foreach (l from 0 to (n - 1)) {
      let v = if maskHas(mk, l) then land_int(tLane(op, m, dt, aty, bt, a, b, as_, bs, l, n), mask(w)) else land_int(asr_int(cv, l * w), mask(w));
      r = lor_int(r, shl_int(v, l * w))
    };
    r
  }
}

val isNegOp : xop -> bool
function isNegOp(o) = match o { XS(SNEG(_, _)) => true, _ => false }
val unNeg : xop -> xop
function unNeg(o) = match o { XS(SNEG(r, n)) => XS(if r then SR(n) else SN(n)), _ => o }

val vmadRes : (list(string), ty, ty, int, int, int, bool, bool, bool) -> int
function vmadRes(m, aty, bt, ta, tb, cr, na, nb, nc) = {
  let sf = isS(aty) | isS(bt) | (na != nb) | nc;
  let p = if has(m, "po") then ta * tb else if na != nb then lnot_int(ta * tb) else ta * tb;
  let c = if has(m, "po") | na != nb then cr else if nc then lnot_int(cr) else cr;
  let l = if has(m, "po") | na != nb | nc then 1 else 0;
  let v0 = p + (if sf then sext(land_int(c, mask(32)), 32) else land_int(c, mask(32))) + l;
  let v = if has(m, "shr7") then asr_int(v0, 7) else if has(m, "shr15") then asr_int(v0, 15) else v0;
  if has(m, "sat") then (if sf then min_int(max_int(v, negate(pow2(31))), pow2(31) - 1) else min_int(max_int(v, 0), mask(32)))
  else trunc(v, 32)
}

val exec_video : (opcode, list(xop)) -> bool
function exec_video(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    ("vset", [|d, a, b|]) => if ntys(o) == 2 then { let aty = ty_nth(o.tys, 0); let bt = ty_nth(o.tys, 1); wr(d, vsetRes(m, vrd(a, aty), vrd(b, bt), 0)); true } else false,
    ("vset", [|d, a, b, x|]) => if ntys(o) == 2 then {
      let aty = ty_nth(o.tys, 0); let bt = ty_nth(o.tys, 1);
      /* ISA vset: d = optMerge(dsel, tmp, c) when d carries a selector, else the secondary operation */
      if selSfx(opName(d)) == "" then wr(d, vsetRes(m, vrd(a, aty), vrd(b, bt), rd(x, U32)))
      else wreg(baseStr(opName(d)), vmerge(U32, false, selSfx(opName(d)), cmpV(cmpOp(m), vrd(a, aty), vrd(b, bt)), rd(x, U32)));
      true
    } else false,
    ("vmad", [|d, a, b, x|]) => {
      if ntys(o) == 3 then {
        let aty = ty_nth(o.tys, 1); let bt = ty_nth(o.tys, 2);
        wr(d, vmadRes(m, aty, bt, vrd(unNeg(a), aty), vrd(unNeg(b), bt), rd(unNeg(x), U32), isNegOp(a), isNegOp(b), isNegOp(x))); true
      } else false
    },
    _ => false
  }
}

val gx_video : (opcode, list(xop)) -> bool
function gx_video(o, ops) = {
  let op = o.base;
  let m = o.mods;
  if ntys(o) == 3 then {
    let dt = ty_nth(o.tys, 0); let aty = ty_nth(o.tys, 1); let bt = ty_nth(o.tys, 2);
    match ops {
      [|d, a, b|] => {
        if has([|"vadd", "vsub", "vabsdiff", "vmin", "vmax"|], op) then { wr(d, vfinish(m, dt, vcore(op, vrd(a, aty), vrd(b, bt)), 0)); true }
        else if op == "vshl" | op == "vshr" then { wr(d, vfinish(m, dt, vshift(op == "vshl", not_bool(has(m, "wrap")), vrd(a, aty), vrd(b, bt)), 0)); true }
        else false
      },
      [|d, a, b, x|] => {
        if isScalarVop(op) & selSfx(opName(d)) == "" then {
          let v = if op == "vshl" | op == "vshr" then vshift(op == "vshl", not_bool(has(m, "wrap")), vrd(a, aty), vrd(b, bt)) else vcore(op, vrd(a, aty), vrd(b, bt));
          wr(d, vfinish(m, dt, v, rd(x, dt))); true
        } else if isScalarVop(op) then {
          let v = if op == "vshl" | op == "vshr" then vshift(op == "vshl", not_bool(has(m, "wrap")), vrd(a, aty), vrd(b, bt)) else vcore(op, vrd(a, aty), vrd(b, bt));
          wreg(baseStr(opName(d)), vmerge(dt, has(m, "sat"), selSfx(opName(d)), v, rd(x, U32))); true
        } else if isSimd(op) & op != "vset2" & op != "vset4" then {
          wreg(baseStr(opName(d)), simdRes(op, m, dt, aty, bt, rawV(a), rawV(b), rd(x, U32), selOr(a, defSel("a", nlanes(op))),
                                           selOr(b, defSel("b", nlanes(op))), selOr(d, defSel("m", nlanes(op)))));
          true
        } else false
      },
      _ => false
    }
  } else if ntys(o) == 2 & (op == "vset2" | op == "vset4") then match ops {
    [|d, a, b, x|] => {
      let aty = ty_nth(o.tys, 0); let bt = ty_nth(o.tys, 1);
      wreg(baseStr(opName(d)), simdRes(op, m, U32, aty, bt, rawV(a), rawV(b), rd(x, U32), selOr(a, defSel("a", nlanes(op))),
                                       selOr(b, defSel("b", nlanes(op))), selOr(d, defSel("m", nlanes(op)))));
      true
    },
    _ => false
  } else false
}

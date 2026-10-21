/* =============================================================================
 * PTX-EXEC-WG (semantics/ptx-exec-wg.k): warpgroup-level wgmma.mma_async (dense and sparse).
 *
 * The 128 threads of a warpgroup rendezvous (kind "wgmma"); when the last one arrives the A / B
 * tiles are read from the CTA's shared memory as it is at that point (K's snapshot) through the
 * canonical swizzled layouts of the matrix descriptors, and the product is computed with the mma
 * arithmetic of ptx_mma.sail.  fence / commit_group / wait_group are no-ops.
 * ============================================================================= */

val wgShape : list(string) -> string
function wgShape(m) = {
  var r : string = "";
  var l = m;
  while is_cons(l) & r == "" do {
    match l { x :: rest => {
      if string_startswith(x, "m64n") & str_index(x, "k") > 4 then {
        let k = sdrop(x, str_index(x, "k") + 1);
        if k == "8" | k == "16" | k == "32" | k == "64" | k == "256" then {
          let n = str_int(stake(sdrop(x, 4), str_index(x, "k") - 4));
          if n >= 8 & n <= 256 & fmod_int(n, 8) == 0 then r = x
        }
      };
      l = rest
    }, [||] => () }
  };
  r
}

/* ---------------------------------------------------------------- shared-memory tiles */
val dStart : int -> int
function dStart(d) = shl_int(land_int(d, 16383), 4)
val dLbo : int -> int
function dLbo(d) = shl_int(land_int(asr_int(d, 16), 16383), 4)
val dSbo : int -> int
function dSbo(d) = shl_int(land_int(asr_int(d, 32), 16383), 4)
val dSwz : int -> int
function dSwz(d) = land_int(asr_int(d, 62), 3) + 4 * land_int(asr_int(d, 56), 1)

val swG : int -> int
function swG(z) = match z { 1 => 8, 2 => 4, 3 => 2, 4 => 8, 5 => 8, 6 => 8, _ => 1 }
val swB : int -> int
function swB(z) = match z { 1 => 3, 2 => 2, 3 => 1, _ => 0 }

val swizzle : (int, int) -> int
function swizzle(z, a) =
  if z <= 3 then lxor_int(a, asr_int(land_int(a, shl_int(mask(swB(z)), 7)), 3))
  else if z == 4 then lxor_int(a, shl_int(land_int(asr_int(a, 7), 3), 5))
  else if z == 5 then lxor_int(lxor_int(a, shl_int(land_int(asr_int(a, 7), 3), 5)), shl_int(land_int(asr_int(a, 7), 1), 3))
  else if z == 6 then lxor_int(a, shl_int(land_int(asr_int(a, 7), 1), 6))
  else a

/* element offset of (mn, k) in the canonical layout; t = elements per 16 bytes */
val wgEO : (int, bool, int, int, int, int, int) -> int
function wgEO(z, km, t, mn, k, l, s) = {
  if km then {
    if z == 0 then fmod_int(mn, 8) * t + tdiv_int(mn, 8) * s + fmod_int(k, t) + tdiv_int(k, t) * l
    else fmod_int(mn, 8) * swG(z) * t + tdiv_int(mn, 8) * s + fmod_int(k, t) + tdiv_int(k, t) * t
  } else {
    if z == 0 then fmod_int(mn, t) + tdiv_int(mn, t) * s + fmod_int(k, 8) * t + tdiv_int(k, 8) * l
    else fmod_int(mn, t) + fmod_int(tdiv_int(mn, t), swG(z)) * t + tdiv_int(mn, t * swG(z)) * l + fmod_int(k, 8) * swG(z) * t + tdiv_int(k, 8) * s
  }
}

val wgBitOff : (int, bool, int, int, int) -> int
function wgBitOff(d, km, w, mn, k) =
  wgEO(dSwz(d), km, tdiv_int(128, w), mn, k, tdiv_int(dLbo(d) * 8, w), tdiv_int(dSbo(d) * 8, w)) * w

/* element (mn, k) of the tile of descriptor d in the executing CTA's shared memory */
val wgElem : (int, bool, int, int, int) -> int
function wgElem(d, km, w, mn, k) = {
  let bo = wgBitOff(d, km, w, mn, k);
  let a = swizzle(dSwz(d), dStart(d) + tdiv_int(bo, 8));
  land_int(asr_int(mem_rdn(cta_smem(), a, max_int(1, tdiv_int(w, 8))), fmod_int(bo, 8)), mask(w))
}

/* negation of float inputs (imm-scale = -1): the sign bit, bit 5 for fp6 and for fp4 in the centre of its 8-bit container */
val negBit : ty -> int
function negBit(t) = if t.k == "e2m3" | t.k == "e3m2" | t.k == "e2m1" then 5 else ew(t) - 1
val wgNeg : (ty, int, int) -> int
function wgNeg(t, sc, v) = if sc < 0 & isFloatElem(t) then lxor_int(v, pow2(negBit(t))) else v

/* ---------------------------------------------------------------- the accumulator layout */
val wgND : (string, ty) -> int
function wgND(s, t) = regsOf(tdiv_int(shN(s), 2), t)
val posD : (int, int) -> int
function posD(l, e) = rc(tdiv_int(l, 4) + 8 * tdiv_int(fmod_int(e, 4), 2), 8 * tdiv_int(e, 4) + 2 * fmod_int(l, 4) + fmod_int(e, 2))
/* (row, col) of element e of thread u */
val posDU : (int, int) -> int
function posDU(u, e) = { let p = posD(fmod_int(u, 32), e); rc(16 * tdiv_int(u, 32) + tdiv_int(p, 1024), fmod_int(p, 1024)) }

/* the metadata nibble for warp w (sparse descriptor form) */
val wgNibW : (list((int, list(int))), int, int, int, int, int, int) -> int
function wgNibW(arr, w, g, rh, c, nc, sel) =
  if nc == 4 then land_int(asr_int(spMeta(arr, 32 * w + 4 * g + sel), 16 * rh + 4 * c), 15)
  else if nc == 8 then land_int(asr_int(spMeta(arr, 32 * w + 4 * g + 2 * sel + rh), 4 * c), 15)
  else if nc == 16 then land_int(asr_int(spMeta(arr, 32 * w + 4 * g + 2 * tdiv_int(c, 8) + rh), 4 * fmod_int(c, 8)), 15)
  else 0

/* the 32 threads of warp w as lanes 0..31 */
val wgSlice : (list((int, list(int))), int) -> list((int, list(int)))
function wgSlice(arr, w) = {
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) out = (l, listAt(arr, 32 * w + l)) :: out;
  out
}

/* Data = [shape, dtype, atype, btype, mods, descA, descB, scaleA, scaleB, transA, transB, scaleD, aInRegs (, selector)] */
val wg_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function wg_compute(d, arr) = {
  let s = cv_str(d, 0);
  let dt = cv_ty(d, 1);
  let at_ = cv_ty(d, 2);
  let bt = cv_ty(d, 3);
  let m = cv_mods(d, 4);
  let descA = cv_int(d, 5);
  let descB = cv_int(d, 6);
  let scA = cv_int(d, 7);
  let scB = cv_int(d, 8);
  let trA = cv_int(d, 9);
  let trB = cv_int(d, 10);
  let scD = cv_int(d, 11);
  let aregs = cv_int(d, 12) == 1;
  let sel = cv_int(d, 13);
  let kk = shK(s);
  let nd = wgND(s, dt);
  let sub = concat_str("m16n8k", dec_str(kk));
  /* A */
  smap_clear(SP_MA);
  if not_bool(has(m, "sp")) then {
    if aregs then {
      foreach (u from 0 to 127) {
        let rs = listAt(arr, u);
        foreach (e from 0 to (tdiv_int(16 * kk, 32) - 1)) {
          let p = posA(sub, ew(at_), fmod_int(u, 32), e);
          mset(SP_MA, rc(16 * tdiv_int(u, 32) + tdiv_int(p, 1024), fmod_int(p, 1024)), wgNeg(at_, scA, fragBits(rs, nd, at_, e)))
        }
      }
    } else {
      foreach (r from 0 to 63)
        foreach (k from 0 to (kk - 1))
          mset(SP_MA, rc(r, k), wgNeg(at_, scA, wgElem(descA, trA == 0, ew(at_), r, k)))
    }
  } else {
    if aregs then {
      foreach (w from 0 to 3) {
        let aw = wgSlice(arr, w);
        foreach (l from 0 to 31) {
          let rs = ldrop(listAt(aw, l), nd);
          foreach (i from 0 to (spNel(sub) - 1)) {
            let u = tdiv_int(i, spNz(at_));
            let rh = spRh(at_, u);
            let ch = spChunk(at_, u, fmod_int(l, 4));
            let nib = spNib(aw, tdiv_int(l, 4), rh, ch, tdiv_int(kk, spCw(at_)), sel);
            mset(SP_MA, rc(16 * w + tdiv_int(l, 4) + 8 * rh, spCw(at_) * ch + spPos(at_, nib, fmod_int(i, spNz(at_)))),
                 wgNeg(at_, scA, fragBits(rs, 0, at_, i)))
          }
        }
      }
    } else {
      foreach (r from 0 to 63)
        foreach (kp from 0 to (tdiv_int(kk, 2) - 1)) {
          let nib = wgNibW(arr, tdiv_int(r, 16), fmod_int(fmod_int(r, 16), 8), tdiv_int(fmod_int(r, 16), 8), tdiv_int(kp, spNz(at_)),
                           tdiv_int(kk, spCw(at_)), sel);
          mset(SP_MA, rc(r, spCw(at_) * tdiv_int(kp, spNz(at_)) + spPos(at_, nib, fmod_int(kp, spNz(at_)))),
               wgNeg(at_, scA, wgElem(descA, trA == 0, ew(at_), r, kp)))
        }
    }
  };
  /* B */
  smap_clear(SP_MB);
  foreach (k from 0 to (kk - 1))
    foreach (n from 0 to (shN(s) - 1))
      mset(SP_MB, rc(k, n), wgNeg(bt, scB, wgElem(descB, trB == 0, ew(bt), n, k)));
  /* C = D */
  smap_clear(SP_MC);
  if scD != 0 then
    foreach (u from 0 to 127) {
      let rs = listAt(arr, u);
      foreach (e from 0 to (tdiv_int(shN(s), 2) - 1)) mset(SP_MC, posDU(u, e), fragBits(rs, 0, dt, e))
    };
  resRows(s, dt, at_, bt, dt, m);
  /* scatter */
  var out : list((int, list(int))) = [||];
  foreach (u from 127 downto 0) {
    if is_some(lane_get(arr, u)) then {
      var regs : list(int) = [||];
      foreach (r from 0 to (nd - 1)) {
        var v : int = 0;
        foreach (j from 0 to (perReg(dt) - 1)) v = lor_int(v, shl_int(mget(SP_MD, posDU(u, r * perReg(dt) + j)), ew(dt) * j));
        regs = pend_append(regs, [|v|])
      };
      out = (u, regs) :: out
    }
  };
  out
}

val ldropx : (list(xop), int) -> list(xop)
function ldropx(l, n) = if n <= 0 then l else match l { [||] => [||], _ :: t => ldropx(t, n - 1) }

/* optional trailing immediates */
val wgArg : (list(xop), int, int) -> int
function wgArg(l, i, dflt) = if i >= 0 & i < xop_count(l) then sext(rd(xop_nth(l, i), mkty("s", 32)), 32) else dflt

val exec_wg : (opcode, list(xop)) -> bool
function exec_wg(o, ops) = {
  let m = o.mods;
  if o.base != "wgmma" then false
  else if has(m, "fence") | has(m, "commit_group") then true
  else if has(m, "wait_group") then xop_count(ops) == 1
  else if not_bool(has(m, "mma_async")) then false
  else match (o.tys, ops) {
    ([|dt, at_, bt|], dv :: av :: bv :: rest0) => {
      let sp = has(m, "sp");
      let rest : list(xop) = if sp then ldropx(rest0, 2) else rest0;
      let nr = xop_count(rest);
      let data = [|CVS(wgShape(m)), CVT(dt), CVT(at_), CVT(bt), CVM(m),
                   CVI(if isVec(av) then 0 else rd(av, mkty("b", 64))), CVI(rd(bv, mkty("b", 64))),
                   CVI(wgArg(rest, 1, 1)), CVI(wgArg(rest, 2, 1)), CVI(wgArg(rest, 3, 0)),
                   CVI(wgArg(rest, if nr >= 4 then nr - 1 else -1, 0)), CVI(wgArg(rest, 0, 1)), CVI(b2i(isVec(av))),
                   CVI(if sp then rdu32(xop_nth(rest0, 1)) else 0)|];
      let contrib = pend_append(vecRegs(dv, dt), vecRegs(av, at_));
      do_coll("wgmma", pow2(128) - 1, data, if sp then pend_append(contrib, [|rd(xop_nth(rest0, 0), mkty("b", 32))|]) else contrib, vecOps(dv));
      true
    },
    _ => false
  }
}

/* =============================================================================
 * PTX-EXEC-LDM (semantics/ptx-exec-ldm.k): warp-level matrix moves ldmatrix / stmatrix /
 * movmatrix (8x8 b16, and the sm_100 m16n16 / m8n16 / m16n8 8-bit forms).
 *
 * ldmatrix: a rendezvous exchanges the row addresses, then each lane reads the elements it needs
 * (the #ldmFetch continuation, cpost P_ldm / P_ldm2); stmatrix: a rendezvous builds the rows, then
 * each addressing lane writes its row (#stmGo, cpost P_stm).
 * ============================================================================= */

val numOf : list(string) -> int
function numOf(m) = if has(m, "x4") then 4 else if has(m, "x2") then 2 else 1

val dlist : xop -> list(xop)
function dlist(o) = match o { XV(_) => vecOps(o), _ => [|o|] }

val ldrop : (list(int), int) -> list(int)
function ldrop(l, n) = if n <= 0 then l else match l { [||] => [||], _ :: t => ldrop(t, n - 1) }

val half16 : (int, int) -> int
function half16(v, h) = land_int(asr_int(v, 16 * h), 65535)

/* locSp / locOff: the space and offset of a location (an invalid one: offset 0) */
val locSp : (loc, string) -> string
function locSp(l, s) = match l { Loc(sp, _) => sp, _ => if s == "generic" then "shared" else s }
val locOff : loc -> int
function locOff(l) = match l { Loc(_, a) => a, _ => 0 }

/* ---------------------------------------------------------------- ldmatrix m8n8 */
val ldmAddr : (list((int, list(int))), int, int, int) -> int
function ldmAddr(arr, i, a, b) = at(listAt(arr, 8 * i + a), 0) + 2 * b

val ldmHalf : (list((int, list(int))), int, int, int, bool) -> int
function ldmHalf(arr, l, i, h, tr) =
  if tr then ldmAddr(arr, i, 2 * fmod_int(l, 4) + h, tdiv_int(l, 4)) else ldmAddr(arr, i, tdiv_int(l, 4), 2 * fmod_int(l, 4) + h)

val ldm_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function ldm_compute(d, arr) = {
  let n = cv_int(d, 0);
  let tr = cv_bool(d, 1);
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) {
    if is_some(lane_get(arr, l)) then {
      var rs : list(int) = [||];
      foreach (i from (n - 1) downto 0) rs = ldmHalf(arr, l, i, 0, tr) :: ldmHalf(arr, l, i, 1, tr) :: rs;
      out = (l, rs) :: out
    }
  };
  out
}

/* ---------------------------------------------------------------- ldmatrix m8n16 / m16n16 */
val ldmFmt : (list(string), list(ty)) -> int
function ldmFmt(m, ts) = {
  var s4 : bool = false;
  var l = ts;
  while is_cons(l) do { match l { t :: rest => { if ty_eq(t, mkty("s", 4)) then s4 = true; l = rest }, [||] => () } };
  if has(m, "b6x16_p32") then 6 else if has(m, "b4x16_p64") then 4 else if s4 then 5 else 0
}

val ldmEl : (int, int, int) -> int
function ldmEl(f, r, c) = match f {
  0 => land_int(asr_int(r, 8 * c), 255),
  6 => land_int(asr_int(r, 6 * c), 63),
  4 => shl_int(land_int(asr_int(r, 4 * c), 15), 2),
  5 => { let x = land_int(asr_int(r, 4 * c), 15); land_int(if land_int(x, 8) == 8 then lor_int(x, 240) else x, 255) },
  _ => 0
}

val ldm2_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function ldm2_compute(d, arr) = {
  let k = cv_int(d, 0);
  let n = cv_int(d, 1);
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) {
    if is_some(lane_get(arr, l)) then {
      var rs : list(int) = [||];
      foreach (i from 0 to (n - 1)) {
        if k == 0 then rs = pend_append(rs, [|at(listAt(arr, 8 * i + tdiv_int(l, 4)), 0)|])
        else foreach (j from 0 to 3) rs = pend_append(rs, [|at(listAt(arr, 16 * i + 4 * fmod_int(l, 4) + j), 0)|])
      };
      out = (l, rs) :: out
    }
  };
  out
}

val ldm2Out : (int, int, int, list(int)) -> list(int)
function ldm2Out(k, f, t, rows) = {
  var out : list(int) = [||];
  var rs = rows;
  if k == 0 then {
    while is_cons(rs) do { match rs { r :: rest => {
      var v : int = 0;
      foreach (j from 0 to 3) v = lor_int(v, shl_int(ldmEl(f, r, 4 * fmod_int(t, 4) + j), 8 * j));
      out = pend_append(out, [|v|]);
      rs = rest
    }, [||] => () } }
  } else {
    while list_length(rs) >= 4 do {
      var v0 : int = 0;
      var v1 : int = 0;
      foreach (j from 0 to 3) {
        v0 = lor_int(v0, shl_int(ldmEl(f, at(rs, j), tdiv_int(t, 4)), 8 * j));
        v1 = lor_int(v1, shl_int(ldmEl(f, at(rs, j), tdiv_int(t, 4) + 8), 8 * j))
      };
      out = pend_append(out, [|v0, v1|]);
      rs = ldrop(rs, 4)
    }
  };
  out
}

/* ---------------------------------------------------------------- stmatrix */
val stmSel : (list((int, list(int))), int, int, int, bool) -> int
function stmSel(arr, i, a, b, tr) =
  if tr then half16(at(listAt(arr, b * 4 + tdiv_int(a, 2)), i), fmod_int(a, 2))
  else half16(at(listAt(arr, a * 4 + tdiv_int(b, 2)), i), fmod_int(b, 2))

val stm_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function stm_compute(d, arr) = {
  let n = cv_int(d, 0);
  let tr = cv_bool(d, 1);
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) {
    if is_some(lane_get(arr, l)) then {
      var v : int = 0;
      if l < 8 * n then foreach (c from 0 to 7) v = lor_int(v, shl_int(stmSel(arr, tdiv_int(l, 8), fmod_int(l, 8), c, tr), 16 * c));
      out = (l, [|v|]) :: out
    }
  };
  out
}

val stmX : (list((int, list(int))), int, int, int) -> int
function stmX(arr, i, r, c) =
  land_int(asr_int(at(listAt(arr, 4 * fmod_int(r, 8) + tdiv_int(c, 2)), i), 8 * (2 * tdiv_int(r, 8) + fmod_int(c, 2))), 255)

val stm2_compute : (list(cval), list((int, list(int)))) -> list((int, list(int)))
function stm2_compute(d, arr) = {
  let n = cv_int(d, 0);
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0) {
    if is_some(lane_get(arr, l)) then {
      var v : int = 0;
      if l < 8 * n then foreach (r from 0 to 15) v = lor_int(v, shl_int(stmX(arr, tdiv_int(l, 8), r, fmod_int(l, 8)), 8 * r));
      out = (l, [|v|]) :: out
    }
  };
  out
}

/* ---------------------------------------------------------------- movmatrix */
val mvElem : (list((int, list(int))), int, int) -> int
function mvElem(arr, r, c) = half16(at(listAt(arr, r * 4 + tdiv_int(c, 2)), 0), fmod_int(c, 2))

val mv_compute : list((int, list(int))) -> list((int, list(int)))
function mv_compute(arr) = {
  var out : list((int, list(int))) = [||];
  foreach (l from 31 downto 0)
    if is_some(lane_get(arr, l)) then
      out = (l, [|lor_int(mvElem(arr, 2 * fmod_int(l, 4), tdiv_int(l, 4)), shl_int(mvElem(arr, 2 * fmod_int(l, 4) + 1, tdiv_int(l, 4)), 16))|]) :: out;
  out
}

/* ---------------------------------------------------------------- instructions */
/* "If the generic address doesn't fall in .shared state space, then the behavior is undefined" (ldmatrix 9.7.15.5.15,
   stmatrix 9.7.15.5.16): under .ub strict */
val ldmShared : (string, loc) -> bool
function ldmShared(op, l) = match l {
  Loc(sp, a) => if ub & sp != "shared" then { fault(concat_str(op, ": the address is not in the .shared window"), a); false } else true,
  _ => true
}

val exec_ldm : (opcode, list(xop)) -> bool
function exec_ldm(o, ops) = {
  let m = o.mods;
  let full : int = 4294967295;
  let t = tchk(cur);
  match (o.base, ops) {
    ("ldmatrix", [|dv, XB([|a|], [||])|]) => {
      let l = ea(a, spaceOf(m));
      if not_bool(ldmShared("ldmatrix", l)) then return true;
      if has(m, "m16n16") | has(m, "m8n16") then {
        let k = if has(m, "m16n16") then 1 else 0;
        t_post[t] = P_ldm2(locSp(l, spaceOf(m)), if ldmFmt(m, o.tys) == 5 then 8 else 16, k, ldmFmt(m, o.tys), dlist(dv));
        do_coll("ldmatrix2", full, [|CVI(k), CVI(numOf(m))|], [|locOff(l)|], [||])
      } else {
        t_post[t] = P_ldm(locSp(l, spaceOf(m)), dlist(dv));
        do_coll("ldmatrix", full, [|CVI(numOf(m)), CVB(has(m, "trans"))|], [|locOff(l)|], [||])
      };
      true
    },
    ("stmatrix", [|XB([|a|], [||]), vs|]) => {
      let l = ea(a, spaceOf(m));
      if not_bool(ldmShared("stmatrix", l)) then return true;
      t_post[t] = P_stm(l, numOf(m));
      if has(m, "m16n8") then do_coll("stmatrix2", full, [|CVI(numOf(m))|], vecRegs(vs, mkty("b", 32)), [||])
      else do_coll("stmatrix", full, [|CVI(numOf(m)), CVB(has(m, "trans"))|], vecRegs(vs, mkty("b", 32)), [||]);
      true
    },
    ("movmatrix", [|d, a|]) => { do_coll("movmatrix", full, [||], [|rd(a, mkty("b", 32))|], [|d|]); true },
    _ => false
  }
}

val coll_ldm : (string, list(cval), list((int, list(int)))) -> option(list((int, list(int))))
function coll_ldm(k, d, arr) = match k {
  "ldmatrix" => Some(ldm_compute(d, arr)),
  "ldmatrix2" => Some(ldm2_compute(d, arr)),
  "stmatrix" => Some(stm_compute(d, arr)),
  "stmatrix2" => Some(stm2_compute(d, arr)),
  "movmatrix" => Some(mv_compute(arr)),
  _ => None()
}

/* the continuations */
val post_ldm : (cpost, list(int)) -> unit
function post_ldm(p, vs) = match p {
  P_ldm(sp, ds) => {
    var as_ = vs;
    var dl = ds;
    while is_cons(dl) & list_length(as_) >= 2 & not_bool(stuck) do {
      match dl { d :: rest => {
        let v = memop_do(mkLoc(sp, at(as_, 0)), RdOp(2));
        let w = memop_do(mkLoc(sp, at(as_, 1)), RdOp(2));
        if not_bool(stuck) then wr(d, lor_int(v, shl_int(w, 16)));
        as_ = ldrop(as_, 2);
        dl = rest
      }, [||] => () }
    }
  },
  P_ldm2(sp, sz, k, f, ds) => {
    var rows : list(int) = [||];
    var as_ = vs;
    while is_cons(as_) & not_bool(stuck) do {
      match as_ { a :: rest => { rows = pend_append(rows, [|memop_do(mkLoc(sp, a), RdOp(sz))|]); as_ = rest }, [||] => () }
    };
    if not_bool(stuck) then collWrite(ds, ldm2Out(k, f, fmod_int(t_tid[tchk(cur)], 32), rows))
  },
  P_stm(l, n) => {
    if fmod_int(t_tid[tchk(cur)], 32) < 8 * n then { let _ = memop_do(l, WrOp(16, at(vs, 0))); () }
  },
  _ => ()
}

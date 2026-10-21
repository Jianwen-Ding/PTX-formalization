/* =============================================================================
 * PTX-EXEC-COLL (semantics/ptx-exec-coll.k): rendezvous of warp-collective `.sync` instructions,
 * and PTX-EXEC-WARP (ptx-exec-warp.k): shfl, vote, match, elect, redux, bar.warp.sync, activemask.
 *
 * A collective with member mask M completes once every lane of M has executed it; the i-th
 * collective of a (kind, mask) executed by each lane belongs to the same instance, keyed by
 * (warp, kind, mask, i).  Arriving lanes wait (#collWait) until the results exist.
 * ============================================================================= */

/* instruction-wide data of a collective (K Lists of Ints, types, modifier sets, ...) */
union cval = {
  CVI : int,
  CVT : ty,
  CVM : list(string),
  CVB : bool,
  CVS : string,
}

struct coll = {
  ckey : string,
  ckind : string,
  cdata : list(cval),
  cmask : int,
  carr : list((int, list(int))),      /* lane |-> contribution (in arrival order) */
  cres : list((int, list(int))),      /* lane |-> result; empty until complete */
  cdone : bool,
  ctaken : int,
}

register colls : list(coll) = [||]

/* collectives whose kind starts with "wg" rendezvous the 128 threads of a warpgroup */
val collW : string -> int
function collW(k) = if string_startswith(k, "wg") then 128 else 32

val lane_get : (list((int, list(int))), int) -> option(list(int))
function lane_get(l, x) = match l { [||] => None(), (k, v) :: rest => if k == x then Some(v) else lane_get(rest, x) }

/* listAt(Arr, L) / at(L, I) */
val listAt : (list((int, list(int))), int) -> list(int)
function listAt(a, x) = match lane_get(a, x) { Some(v) => v, None() => [||] }
val at : (list(int), int) -> int
function at(l, i) = match l { [||] => 0, h :: t => if i == 0 then h else at(t, i - 1) }

/* the lanes of the arrivals in ascending order */
val arr_lanes : list((int, list(int))) -> list(int)
function arr_lanes(a) = {
  var r : list(int) = [||];
  foreach (i from 127 downto 0) { if is_some(lane_get(a, i)) then r = i :: r };
  r
}

val sameAll : (list(int), list(int)) -> list((int, list(int)))
function sameAll(ls, v) = match ls { [||] => [||], l :: rest => (l, v) :: sameAll(rest, v) }

val minLane : list(int) -> int
function minLane(ls) = match ls { [||] => 32, l :: _ => l }

/* the module hook: results of the collectives of the other modules (mma, ldmatrix, ...) */
val coll_compute_ext : (string, list(cval), list((int, list(int)))) -> option(list((int, list(int))))

/* ---------------------------------------------------------------- warp collectives */
val shflJ : (string, int, int, int, int, int) -> (int, int)
function shflJ(mode, l, bv, sm, maxl, minl) = match mode {
  "shfl.up" => (l - bv, b2i(l - bv >= maxl)),
  "shfl.down" => (l + bv, b2i(l + bv <= maxl)),
  "shfl.bfly" => (lxor_int(l, bv), b2i(lxor_int(l, bv) <= maxl)),
  _ => {
    let j = lor_int(minl, land_int(land_int(bv, lnot_int(sm)), 31));
    (j, b2i(j <= maxl))
  }
}

val shflLane : (string, int, list((int, list(int)))) -> list(int)
function shflLane(mode, l, arr) = {
  let c = listAt(arr, l);
  let own = at(c, 0);
  let bv = land_int(at(c, 1), 31);
  let cv = land_int(at(c, 2), 31);
  let sm = land_int(asr_int(at(c, 2), 8), 31);
  let maxl = lor_int(land_int(l, sm), land_int(land_int(cv, lnot_int(sm)), 31));
  let minl = land_int(l, sm);
  let (j, pv) = shflJ(mode, l, bv, sm, maxl, minl);
  if pv == 1 then [|match lane_get(arr, j) { Some(v) => at(v, 0), None() => own }, 1|] else [|own, 0|]
}

val shflAll : (string, list(int), list((int, list(int)))) -> list((int, list(int)))
function shflAll(mode, ls, arr) = match ls { [||] => [||], l :: rest => (l, shflLane(mode, l, arr)) :: shflAll(mode, rest, arr) }

val preds : (list((int, list(int))), list(int)) -> list(int)
function preds(arr, ls) = match ls { [||] => [||], l :: rest => at(listAt(arr, l), 0) :: preds(arr, rest) }

val allTrue : list(int) -> bool
function allTrue(l) = match l { [||] => true, v :: rest => v != 0 & allTrue(rest) }
val anyTrue : list(int) -> bool
function anyTrue(l) = match l { [||] => false, v :: rest => v != 0 | anyTrue(rest) }

val ballotv : (list((int, list(int))), list(int)) -> int
function ballotv(arr, ls) = match ls { [||] => 0, l :: rest => lor_int(if at(listAt(arr, l), 0) != 0 then pow2(l) else 0, ballotv(arr, rest)) }

val sameValueMask : (list((int, list(int))), list(int), int) -> int
function sameValueMask(arr, ls, v) = match ls {
  [||] => 0,
  l :: rest => lor_int(if at(listAt(arr, l), 0) == v then pow2(l) else 0, sameValueMask(arr, rest, v))
}

val cv_int : (list(cval), int) -> int
function cv_int(d, i) = match d { [||] => 0, h :: t => if i == 0 then (match h { CVI(n) => n, CVB(b) => b2i(b), _ => 0 }) else cv_int(t, i - 1) }
val cv_ty : (list(cval), int) -> ty
function cv_ty(d, i) = match d { [||] => mkty("b", 32), h :: t => if i == 0 then (match h { CVT(x) => x, _ => mkty("b", 32) }) else cv_ty(t, i - 1) }
val cv_mods : (list(cval), int) -> list(string)
function cv_mods(d, i) = match d { [||] => [||], h :: t => if i == 0 then (match h { CVM(x) => x, _ => [||] }) else cv_mods(t, i - 1) }
val cv_bool : (list(cval), int) -> bool
function cv_bool(d, i) = match d { [||] => false, h :: t => if i == 0 then (match h { CVB(x) => x, CVI(n) => n != 0, _ => false }) else cv_bool(t, i - 1) }

val reduxStep : (string, ty, list(string), int, int) -> int
function reduxStep(op, t, m, a, b) = match op {
  "redux.add" => trunc(a + b, 32),
  "redux.and" => land_int(a, b),
  "redux.or" => lor_int(a, b),
  "redux.xor" => lxor_int(a, b),
  "redux.min" => if isF(t) then fpMinMax(has(m, "ftz"), has(m, "NaN"), has(m, "xorsign"), 32, a, b, true) else if ival(a, t) <= ival(b, t) then a else b,
  "redux.max" => if isF(t) then fpMinMax(has(m, "ftz"), has(m, "NaN"), has(m, "xorsign"), 32, a, b, false) else if ival(a, t) >= ival(b, t) then a else b,
  _ => a
}

/* redux.sync .abs: "the absolute value of the input is considered for the reduction operation" (.f32 only) */
val reduxAbs : (ty, list(string), list(int)) -> list(int)
function reduxAbs(t, m, l) = match l {
  [||] => [||],
  v :: rest => (if isF(t) & has(m, "abs") then land_int(v, 2147483647) else v) :: reduxAbs(t, m, rest)
}

val reduxFold : (string, ty, list(string), list(int)) -> int
function reduxFold(op, t, m, l) = match l {
  [||] => 0,
  [|v|] => v,
  v :: rest => reduxStep(op, t, m, v, reduxFold(op, t, m, rest))
}

val matchAny : (list(int), list((int, list(int))), list(int)) -> list((int, list(int)))
val electAll : (list(int), int) -> list((int, list(int)))
val sum_contrib : (list((int, list(int))), list(int)) -> int
val sum_list : list(int) -> int
/* the lanes of the arrivals as a mask */
val arrMask : list((int, list(int))) -> int

val collCompute : (string, list(cval), list((int, list(int)))) -> list((int, list(int)))
function collCompute(k, d, arr) = {
  let ls = arr_lanes(arr);
  match k {
    "bar.warp" => sameAll(ls, [|0|]),
    "shfl.up" => shflAll(k, ls, arr),
    "shfl.down" => shflAll(k, ls, arr),
    "shfl.bfly" => shflAll(k, ls, arr),
    "shfl.idx" => shflAll(k, ls, arr),
    "vote.all" => sameAll(ls, [|b2i(allTrue(preds(arr, ls)))|]),
    "vote.any" => sameAll(ls, [|b2i(anyTrue(preds(arr, ls)))|]),
    "vote.uni" => sameAll(ls, [|b2i(allTrue(preds(arr, ls)) | not_bool(anyTrue(preds(arr, ls))))|]),
    "vote.ballot" => sameAll(ls, [|ballotv(arr, ls)|]),
    "match.any" => matchAny(ls, arr, ls),
    "match.all" => {
      /* "d is set to mask corresponding to non-exited threads in membermask" (the lanes that took part) */
      if sameValueMask(arr, ls, at(listAt(arr, minLane(ls)), 0)) == arrMask(arr) then sameAll(ls, [|arrMask(arr), 1|])
      else sameAll(ls, [|0, 0|])
    },
    "elect" => electAll(ls, minLane(ls)),
    "coll_test.sum" => sameAll(ls, [|sum_contrib(arr, ls)|]),
    "coll_test.first" => sameAll(ls, listAt(arr, minLane(ls))),
    _ => {
      if k == "redux.add" | k == "redux.min" | k == "redux.max" | k == "redux.and" | k == "redux.or" | k == "redux.xor" then
        sameAll(ls, [|reduxFold(k, cv_ty(d, 0), cv_mods(d, 1), reduxAbs(cv_ty(d, 0), cv_mods(d, 1), preds(arr, ls)))|])
      else match coll_compute_ext(k, d, arr) { Some(r) => r, None() => arr }
    }
  }
}

function matchAny(ls, arr, all) = match ls {
  [||] => [||],
  l :: rest => (l, [|sameValueMask(arr, all, at(listAt(arr, l), 0)), 1|]) :: matchAny(rest, arr, all)
}

function electAll(ls, leader) = match ls { [||] => [||], l :: rest => (l, [|leader, b2i(l == leader)|]) :: electAll(rest, leader) }

function sum_contrib(arr, ls) = match ls { [||] => 0, l :: rest => sum_list(listAt(arr, l)) + sum_contrib(arr, rest) }
function sum_list(l) = match l { [||] => 0, h :: t => h + sum_list(t) }

/* ---------------------------------------------------------------- arrival and completion */
val seqKey : (string, int) -> string
function seqKey(k, m) = concat_str(k, concat_str(":", dec_str(m)))

/* mkKey(T, K, M, CS): (warp, kind, mask, sequence number) within the CTA */
val mkKey : (int, string, int, int) -> string
function mkKey(t, k, m, sq) =
  concat_str(dec_str(cur_cta()), concat_str("#", concat_str(dec_str(tdiv_int(t, collW(k))), concat_str("#", concat_str(k, concat_str("#", concat_str(dec_str(m), concat_str("#", dec_str(sq)))))))))

val find_coll : string -> option(coll)
function find_coll(key) = {
  var l = colls;
  var r : option(coll) = None();
  while is_cons(l) & is_none(r) do { match l { c :: rest => { if c.ckey == key then r = Some(c); l = rest }, [||] => () } };
  r
}

val put_coll : coll -> unit
function put_coll(c) = {
  var l = colls;
  var out : list(coll) = [||];
  while is_cons(l) do { match l { x :: rest => { if x.ckey != c.ckey then out = x :: out; l = rest }, [||] => () } };
  colls = c :: out
}

val drop_coll : string -> unit
function drop_coll(key) = {
  var l = colls;
  var out : list(coll) = [||];
  while is_cons(l) do { match l { x :: rest => { if x.ckey != key then out = x :: out; l = rest }, [||] => () } };
  colls = out
}

val list_len_pairs : list((int, list(int))) -> int
function list_len_pairs(l) = match l { [||] => 0, _ :: t => 1 + list_len_pairs(t) }

/* shfl.sync, vote.sync, match.sync and redux.sync wait for "all non-exited threads corresponding to
 * membermask" (PTX ISA 9.3, shfl.sync, vote.sync, match.sync, redux.sync), and bar.warp.sync is a barrier,
 * and "Barriers exclusively waiting on arrivals from exited threads are always released" (exit).  For these
 * kinds the lanes of the mask that have exited do not take part; elect.sync waits for "all threads in the
 * membermask" and the warp-wide matrix instructions are undefined after an exit, so they keep the full mask. */
val collExitOk : string -> bool
function collExitOk(k) =
  string_startswith(k, "shfl.") | string_startswith(k, "vote.") | string_startswith(k, "match.")
  | string_startswith(k, "redux.") | k == "bar.warp"

function arrMask(a) = match a { [||] => 0, (l, _) :: rest => lor_int(pow2(l), arrMask(rest)) }

/* the lanes an instance of kind K, mask M of warp W waits for */
val collNeed : (int, int, string, int) -> int
function collNeed(c, w, k, m) = if collExitOk(k) then land_int(m, smap_get(c_wlive[cchk(c)], dec_str(w))) else m

val collComplete : (int, int, coll) -> coll
function collComplete(c, w, x) =
  if not_bool(x.cdone) & land_int(collNeed(c, w, x.ckind, x.cmask), lnot_int(arrMask(x.carr))) == 0
  then { x with cres = collCompute(x.ckind, x.cdata, x.carr), cdone = true }
  else x

/* a lane of warp W of CTA C exited: the instances that only waited for exited lanes complete */
val collRecheck : (int, int) -> unit
function collRecheck(c, w) = {
  let pre = concat_str(dec_str(c), concat_str("#", concat_str(dec_str(w), "#")));
  var l = colls;
  var out : list(coll) = [||];
  while is_cons(l) do {
    match l {
      x :: rest => {
        out = (if string_startswith(x.ckey, pre) & collExitOk(x.ckind) then collComplete(c, w, x) else x) :: out;
        l = rest
      },
      [||] => ()
    }
  };
  colls = out
}

/* #coll(K, M, D, Cn, Ds) */
val do_coll : (string, int, list(cval), list(int), list(xop)) -> unit
function do_coll(k, m, d, cn, ds) = {
  let t = tchk(cur);
  let tid = t_tid[t];
  let lane = fmod_int(tid, collW(k));
  if not_bool(bit_int(m, lane)) then fault("lane not in membermask", m)
  else {
    let cs = t_cseq[t];
    let sk = seqKey(k, m);
    let sq = smap_get(cs, sk);
    smap_set(cs, sk, sq + 1);
    let key = mkKey(tid, k, m, sq);
    let c0 : coll = match find_coll(key) {
      Some(c) => c,
      None() => struct { ckey = key, ckind = k, cdata = d, cmask = m, carr = [||], cres = [||], cdone = false, ctaken = 0 }
    };
    let arr = pend_append(c0.carr, [|(lane, cn)|]);
    put_coll(collComplete(cur_cta(), tdiv_int(tid, collW(k)), { c0 with ckind = k, cdata = d, cmask = m, carr = arr }));
    t_cont[t] = K_collwait(key, ds);
    yielded = true
  }
}

/* #collWait(Key, Ds): once the instance is complete, each lane takes its own results */
val coll_ready : int -> bool
function coll_ready(t) = match t_cont[tchk(t)] {
  K_collwait(key, _) => match find_coll(key) {
    Some(c) => c.cdone & is_cons(listAt(c.cres, fmod_int(t_tid[tchk(t)], collW(c.ckind)))),
    None() => false
  },
  _ => false
}

val collWrite : (list(xop), list(int)) -> unit
function collWrite(ds, vs) = match (ds, vs) { (d :: dr, v :: vr) => { wr(d, v); collWrite(dr, vr) }, _ => () }

/* the continuation of a collective after its rendezvous (the K items following #coll, e.g.
 * #ldmFetch / #stmGo): the lane's results are handed to it instead of being written */
union cpost = {
  P_none : unit,
  P_ldm : (string, list(xop)),                    /* space, destinations */
  P_ldm2 : (string, int, int, int, list(xop)),    /* space, row bytes, kind, format, destinations */
  P_stm : (loc, int),                              /* row location, matrices */
  P_tcalloc : (loc, int),                          /* #tcAlloc(L, N) */
  P_tcdealloc : (int, int),                        /* #tcDealloc(B, N) */
}
register t_post : vector(16384, cpost) = vector_init(P_none())
val coll_post_ext : (cpost, list(int)) -> unit

val coll_take : unit -> unit
function coll_take() = {
  let t = tchk(cur);
  match t_cont[t] {
    K_collwait(key, ds) => match find_coll(key) {
      Some(c) => {
        let vs = listAt(c.cres, fmod_int(t_tid[t], collW(c.ckind)));
        match t_post[t] {
          P_none() => collWrite(ds, vs),
          p => { t_post[t] = P_none(); coll_post_ext(p, vs) }
        };
        if c.ctaken + 1 == list_len_pairs(c.carr) then drop_coll(key) else put_coll({ c with ctaken = c.ctaken + 1 });
        t_cont[t] = K_none()
      },
      None() => ()
    },
    _ => ()
  }
}

/* ---------------------------------------------------------------- instructions */
val dests : xop -> list(xop)
function dests(dp) = match dual_regs(dp) { Some((d, p)) => [|XS(d), XS(p)|], None() => [|dp|] }

/* activemask without `.convergence its`: "An active thread will contribute 1 for its entry in the result and exited
 * or inactive or predicated-off thread will contribute 0" (PTX ISA 9.3, activemask).  The model reports the lanes that
 * execute this activemask together with the executing lane: activemask ends a turn (isVisible), so the first lane of
 * the warp to execute it finds the lanes that will execute the same instruction next (same function and pc, not
 * blocked, guard true; an unguarded lane whose collective just completed counts too) and records the mask for all of
 * them.  Exited lanes, lanes elsewhere (divergent paths) and predicated-off lanes contribute 0.  The visit count of the
 * instruction keys the recorded mask (t_misc "act:<site>:<visit>"). */
val actKey : (int, string) -> string
function actKey(s, site) = concat_str("act:", concat_str(site, concat_str(":", dec_str(smap_get(t_cseq[tchk(s)], seqKey(site, 0))))))

val actJoins : (int, int, string) -> bool
function actJoins(t, s, site) = {
  let me = tchk(t);
  let o = tchk(s);
  let pc = t_pc[me] - 1;
  if not_bool(t_alive[o]) | t_status[o] != 0 | t_cta[o] != t_cta[me] | tdiv_int(t_tid[o], 32) != tdiv_int(t_tid[me], 32)
     | t_fn[o] != t_fn[me] | t_pc[o] != pc | smap_get(t_misc[o], actKey(s, site)) != 0 then false
  else match get_item(f_base[fchk(t_fn[o])] + pc) {
    INS(None(), _, _) => t_cont[o] == K_none() | coll_ready(s),
    INS(Some((p, neg)), _, _) => {
      if t_cont[o] != K_none() then false
      else { let sv = cur; cur = s; let v = rreg(p); cur = sv; (v != 0) != neg }
    },
    _ => false
  }
}

val actGroup : int -> int
function actGroup(t) = {
  let s = tchk(t);
  let site = concat_str("act@", concat_str(f_name[fchk(t_fn[s])], concat_str(":", dec_str(t_pc[s]))));
  let key = actKey(t, site);
  let cs = t_cseq[s];
  let sk = seqKey(site, 0);
  let mine = smap_get(t_misc[s], key);
  var msk : int = mine;
  if mine == 0 then {
    let lane = fmod_int(t_tid[s], 32);
    msk = pow2(lane);
    var grp : list(int) = [||];
    foreach (j from 0 to 31) {
      let o = t - lane + j;
      if j != lane & o >= 0 & o < nthreads then { if actJoins(t, o, site) then { msk = lor_int(msk, pow2(j)); grp = o :: grp } }
    };
    while is_cons(grp) do match grp { o :: rest => { smap_set(t_misc[tchk(o)], actKey(o, site), msk); grp = rest }, [||] => () }
  } else smap_del(t_misc[s], key);
  smap_set(cs, sk, smap_get(cs, sk) + 1);
  msk
}

val exec_warp : (opcode, list(xop)) -> bool
function exec_warp(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    ("bar", [|mm|]) => if has(m, "warp") then { do_coll("bar.warp", rdu32(mm), [||], [||], [||]); true } else false,
    ("shfl", [|dp, a, b, c, mm|]) => {
      do_coll(concat_str("shfl.", pick(m, [|"up", "down", "bfly", "idx"|], "idx")), rdu32(mm), [||],
              [|rdb32(a), rdu32(b), rdu32(c)|], dests(dp));
      true
    },
    ("vote", [|d, q, mm|]) => {
      do_coll(concat_str("vote.", pick(m, [|"all", "any", "uni", "ballot"|], "all")), rdu32(mm), [||], [|rdp(q)|], [|d|]);
      true
    },
    ("match", [|dp, a, mm|]) => {
      do_coll(concat_str("match.", pick(m, [|"any", "all"|], "any")), rdu32(mm), [|CVI(rdu32(mm))|], [|rd(a, ty1(o))|], dests(dp));
      true
    },
    ("elect", [|dp, mm|]) => { do_coll("elect", rdu32(mm), [||], [||], dests(dp)); true },
    ("redux", [|d, a, mm|]) => {
      do_coll(concat_str("redux.", pick(m, [|"add", "min", "max", "and", "or", "xor"|], "add")), rdu32(mm),
              [|CVT(ty1(o)), CVM(m)|], [|rd(a, ty1(o))|], [|d|]);
      true
    },
    ("coll_test", [|d, a, mk|]) => {
      do_coll(concat_str("coll_test.", pick(m, [|"sum", "first"|], "sum")), rdu32(mk), [||], [|rd(a, ty1(o))|], [|d|]);
      true
    },
    ("activemask", [|d|]) => {
      let t = tchk(cur);
      if its then {
        /* the lanes that have executed this instruction (same function, pc, visit count) so far */
        let site = concat_str("act@", concat_str(f_name[fchk(t_fn[t])], concat_str(":", dec_str(t_pc[t]))));
        let cs = t_cseq[t];
        let sk = seqKey(site, 0);
        let sq = smap_get(cs, sk);
        smap_set(cs, sk, sq + 1);
        let key = mkKey(t_tid[t], site, 0, sq);
        let lane = fmod_int(t_tid[t], 32);
        let c0 : coll = match find_coll(key) {
          Some(c) => c,
          None() => struct { ckey = key, ckind = "act", cdata = [||], cmask = 0, carr = [||], cres = [||], cdone = false, ctaken = 0 }
        };
        let arr = if is_some(lane_get(c0.carr, lane)) then c0.carr else pend_append(c0.carr, [|(lane, [|0|])|]);
        put_coll({ c0 with carr = arr });
        var msk : int = 0;
        var l = arr_lanes(arr);
        while is_cons(l) do { match l { x :: rest => { msk = lor_int(msk, pow2(x)); l = rest }, [||] => () } };
        wr(d, msk)
      } else wr(d, actGroup(t));
      true
    },
    _ => false
  }
}

val isCollective_core : string -> bool
function isCollective_core(op) = has([|"shfl", "vote", "match", "elect", "redux", "coll_test"|], op)

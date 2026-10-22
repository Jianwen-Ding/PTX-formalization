/* =============================================================================
 * PTX-EXEC-CLUSTER (semantics/ptx-exec-cluster.k): clusters, distributed shared memory, mapa /
 * getctarank, barrier.cluster;
 * PTX-EXEC-MBAR (ptx-exec-mbar.k): mbarrier objects;
 * PTX-EXEC-ASYNC (ptx-exec-async.k): cp.async;
 * PTX-EXEC-BULK (ptx-exec-bulk.k): cp.async.bulk / cp.reduce.async.bulk (non-tensor).
 *
 * Copies are performed at issue, as in K.  Blocked waits (#clWait, #mbTry) are the thread
 * continuation K_wait(kind, _) with their data in t_wd / t_wx; they are retried at the start of a
 * turn (K's `<turn> -1`).
 * ============================================================================= */

register t_wd : vector(16384, list(int)) = vector_init([||])
register t_wx : vector(16384, list(xop)) = vector_init([||])

/* ---------------------------------------------------------------- geometry */
val clTarget : (int, int, dim3, dim3) -> int
function clTarget(c, r, g, cd) =
  (tdiv_int(fmod_int(c, g.x), cd.x) * cd.x + fmod_int(r, cd.x))
  + (tdiv_int(fmod_int(tdiv_int(c, g.x), g.y), cd.y) * cd.y + fmod_int(tdiv_int(r, cd.x), cd.y)) * g.x
  + (tdiv_int(tdiv_int(c, g.x * g.y), cd.z) * cd.z + tdiv_int(r, cd.x * cd.y)) * g.x * g.y

val clId : (int, dim3, dim3) -> int
function clId(c, g, cd) =
  tdiv_int(fmod_int(c, g.x), cd.x) + tdiv_int(fmod_int(tdiv_int(c, g.x), g.y), cd.y) * tdiv_int(g.x, cd.x)
  + tdiv_int(tdiv_int(c, g.x * g.y), cd.z) * tdiv_int(g.x, cd.x) * tdiv_int(g.y, cd.y)

val clSize : dim3 -> int
function clSize(d) = d.x * d.y * d.z

/* ---------------------------------------------------------------- distributed shared memory */
function cluster_mem(a, op) = {
  let c = cur_cta();
  let r = asr_int(a, 24);
  let off = land_int(a, 16777215);
  if r == 0 | clTarget(c, r - 1, gridDim, clusterDim) == c then Some(memop_do(Loc("shared", off), op))
  else {
    let tc = clTarget(c, r - 1, gridDim, clusterDim);
    if r - 1 < clSize(clusterDim) & 0 <= tc & tc < ncta_slots & c_alive[cchk(tc)] then Some(mem_apply(c_smem[cchk(tc)], off, op))
    else { fault("shared::cluster access to a CTA that has exited or is not in the cluster", a); Some(0) }
  }
}

/* ---------------------------------------------------------------- barrier.cluster: cl(arrived, generation, exited) */
val clGet : (int, string) -> int
function clGet(id, f) = smap_get(SP_CLBAR, concat_str(f, dec_str(id)))
val clPut : (int, int, int, int) -> unit
function clPut(id, n, g, x) = {
  smap_set(SP_CLBAR, concat_str("n", dec_str(id)), n);
  smap_set(SP_CLBAR, concat_str("g", dec_str(id)), g);
  smap_set(SP_CLBAR, concat_str("x", dec_str(id)), x)
}

function cluster_exit() = {
  let id = clId(cur_cta(), gridDim, clusterDim);
  let n = clGet(id, "n");
  let g = clGet(id, "g");
  let x = clGet(id, "x");
  let tot = clSize(clusterDim) * vol(blockDim);
  let arrived = b2i(rreg("%%clgen") > g);
  if n - arrived >= tot - (x + 1) & n - arrived > 0 then clPut(id, 0, g + 1, x + 1)
  else clPut(id, n - arrived, g, x + 1)
}

val exec_cluster : (opcode, list(xop)) -> bool
function exec_cluster(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    // without .shared::cluster, a and d are generic addresses (in the shared::cta or cluster window)
    ("mapa", [|d, a, r|]) => {
      let own = rreg("%cluster_ctarank");
      let gen = not_bool(has(m, "shared::cluster"));
      let v = rd(a, ty1(o));
      let x = lor_int(land_int(if gen then clFromGeneric(v, own) else v, 16777215), shl_int(rdu32(r) + 1, 24));
      wr(d, if gen then trunc(clToGeneric(x, own), wid(ty1(o))) else x);
      true
    },
    ("getctarank", [|d, a|]) => {
      let v = if has(m, "shared::cluster") then rd(a, ty1(o)) else clFromGeneric(rd(a, ty1(o)), rreg("%cluster_ctarank"));
      wr(d, if asr_int(v, 24) == 0 then rreg("%cluster_ctarank") else asr_int(v, 24) - 1);
      true
    },
    ("barrier", [||]) => {
      if has(m, "cluster") & has(m, "arrive") then {
        let id = clId(cur_cta(), gridDim, clusterDim);
        let n = clGet(id, "n");
        let g = clGet(id, "g");
        let x = clGet(id, "x");
        let tot = clSize(clusterDim) * vol(blockDim);
        if n + 1 >= tot - x then clPut(id, 0, g + 1, x) else clPut(id, n + 1, g, x);
        wreg("%%clgen", g + 1);
        true
      } else if has(m, "cluster") & has(m, "wait") then {
        let t = tchk(cur);
        t_wd[t] = [|clId(cur_cta(), gridDim, clusterDim), rreg("%%clgen")|];
        t_cont[t] = K_wait("clwait", 0);
        yielded = true;
        true
      } else false
    },
    _ => false
  }
}

/* ---------------------------------------------------------------- mbarrier objects mb(phase, pending, expected, tx, layout) */
val mbKey : loc -> int
function mbKey(l) = match l {
  Loc("shared", a) => cur_cta() * 4294967296 + a,
  Loc("cluster", a) => {
    let c = cur_cta();
    (if asr_int(a, 24) == 0 then c else clTarget(c, asr_int(a, 24) - 1, gridDim, clusterDim)) * 4294967296 + land_int(a, 16777215)
  },
  _ => -1
}

val mbHas : int -> bool
function mbHas(k) = smap_has(SP_MBARS, concat_str("p", dec_str(k)))
val mbF : (int, string) -> int
function mbF(k, f) = smap_get(SP_MBARS, concat_str(f, dec_str(k)))
/* mbGet: an uninitialised object is mb(-1, 0, 1, 0, 0) */
val mbGet : int -> (int, int, int, int, int)
function mbGet(k) = if mbHas(k) then (mbF(k, "p"), mbF(k, "q"), mbF(k, "e"), mbF(k, "t"), mbF(k, "l")) else (-1, 0, 1, 0, 0)
val mbPut : (int, (int, int, int, int, int)) -> unit
function mbPut(k, s) = {
  let (p, q, e, t, l) = s;
  let ks = dec_str(k);
  /* "w": a wait has returned True for the last completed phase (cleared when a phase completes) */
  if not_bool(smap_has(SP_MBARS, concat_str("p", ks))) | smap_get(SP_MBARS, concat_str("p", ks)) != p then
    smap_set(SP_MBARS, concat_str("w", ks), if p == 0 then 1 else 0);
  smap_set(SP_MBARS, concat_str("p", ks), p);
  smap_set(SP_MBARS, concat_str("q", ks), q);
  smap_set(SP_MBARS, concat_str("e", ks), e);
  smap_set(SP_MBARS, concat_str("t", ks), t);
  smap_set(SP_MBARS, concat_str("l", ks), l)
}
val mbDel : int -> unit
function mbDel(k) = {
  let ks = dec_str(k);
  smap_del(SP_MBARS, concat_str("p", ks)); smap_del(SP_MBARS, concat_str("q", ks)); smap_del(SP_MBARS, concat_str("e", ks));
  smap_del(SP_MBARS, concat_str("t", ks)); smap_del(SP_MBARS, concat_str("l", ks)); smap_del(SP_MBARS, concat_str("w", ks))
}

/* "Performing any mbarrier operation except mbarrier.init on a memory location that does not contain a valid mbarrier
 * object, results in undefined behaviour"; an address outside the .shared::cta (or cluster shared) window is undefined
 * too (9.7.14.16).  Neither has a meaning the model could give, so both stop the run. */
/* an mbarrier object is a .b64 with "Alignment (bytes) 8" in .shared (9.7.14.16.1): a misaligned
   address is reported under .ub strict */
val mbAligned : (string, int) -> bool
function mbAligned(op, k) =
  if ub & tmod_int(k, 8) != 0 then { fault(concat_str("mbarrier.", concat_str(op, ": address not aligned to 8 bytes")), k); false }
  else true

val mbOk : (string, int) -> bool
function mbOk(op, k) =
  if k == -1 then { fault(concat_str("mbarrier.", concat_str(op, ": address outside the shared memory windows")), 0); false }
  else if not_bool(mbHas(k)) then { fault(concat_str("mbarrier.", concat_str(op, " on an object that is not initialised")), 0); false }
  else mbAligned(op, k)

/* Table 37 (9.7.14.16.3): the counts' ranges, checked under .ub strict */
val mbMaxCount : int -> int
function mbMaxCount(l) = if l == 1 then 511 else 1048575
val mbUbState : (string, (int, int, int, int, int)) -> bool
function mbUbState(op, s) = {
  let (_, q, e, t, l) = s;
  if ub & (t > 1048575 | t < negate(1048575)) then { fault(concat_str("mbarrier.", concat_str(op, ": tx-count outside -(2^20 - 1) .. 2^20 - 1")), t); false }
  else if ub & (q < 0 | q > mbMaxCount(l)) then { fault(concat_str("mbarrier.", concat_str(op, ": arrive count larger than the pending count")), q); false }
  else true
}

/* "For each primary phase of the mbarrier object, at least one test_wait or try_wait operation must be performed which
 * returns True for waitComplete before an arrive-on operation in the subsequent primary phase" (9.7.14.16.5.1): an
 * arrive-on before such a wait is reported under .ub strict */
val mbWaited : int -> unit
function mbWaited(k) = smap_set(SP_MBARS, concat_str("w", dec_str(k)), 1)
val mbArriveOk : (string, int) -> bool
function mbArriveOk(op, k) =
  if ub & smap_get(SP_MBARS, concat_str("w", dec_str(k))) == 0 then {
    fault(concat_str("mbarrier.", concat_str(op, ": arrive-on before a wait returned True for the previous phase")), 0); false
  } else true

val mbCheck : ((int, int, int, int, int)) -> (int, int, int, int, int)
function mbCheck(s) = { let (p, q, e, t, l) = s; if q == 0 & t == 0 then (p + 1, e, e, 0, l) else s }
val mbArrive : ((int, int, int, int, int), int) -> (int, int, int, int, int)
function mbArrive(s, n) = { let (p, q, e, t, l) = s; mbCheck((p, q - n, e, t, l)) }
val mbTx : ((int, int, int, int, int), int) -> (int, int, int, int, int)
function mbTx(s, n) = { let (p, q, e, t, l) = s; mbCheck((p, q, e, t + n, l)) }
val mbDrop : ((int, int, int, int, int), int) -> (int, int, int, int, int)
function mbDrop(s, n) = { let (p, q, e, t, l) = s; mbCheck((p, q - n, e - n, t, l)) }
val mbState : ((int, int, int, int, int)) -> int
function mbState(s) = { let (p, q, _, _, _) = s; lor_int(p, shl_int(q, 32)) }
val mbPhase : int -> int
function mbPhase(k) = { let (p, _, _, _, _) = mbGet(k); p }

val mbDone : (int, list(string), int) -> bool
function mbDone(k, m, v) =
  if has(m, "parity") then tmod_int(mbPhase(k), 2) != land_int(v, 1) else mbPhase(k) > land_int(v, 4294967295)

val mbMcKeys : (int, int) -> list(int)
function mbMcKeys(off, mk) = {
  var out : list(int) = [||];
  foreach (r from 0 to 31)
    if bit_int(mk, r) & asr_int(mk, r) > 0 then out = pend_append(out, [|mbKey(Loc("cluster", lor_int(shl_int(r + 1, 24), off)))|]);
  out
}

val locOffA : loc -> int
function locOffA(l) = match l { Loc(_, a) => a, _ => 0 }

/* the operands after the address: tx-count / count */
val mbTxArg : (list(string), list(xop)) -> int
function mbTxArg(m, l) = match l { x :: _ => if has(m, "expect_tx") then rdu32(x) else 0, [||] => 0 }
val mbCountArg : (list(string), list(xop)) -> int
function mbCountArg(m, l) = if has(m, "expect_tx") then 1 else match l { x :: _ => rdu32(x), [||] => 1 }

val xdrop_last : list(xop) -> list(xop)
function xdrop_last(l) = match l { [||] => [||], [|_|] => [||], x :: rest => x :: xdrop_last(rest) }
val xlast : list(xop) -> option(xop)
function xlast(l) = match l { [||] => None(), [|x|] => Some(x), _ :: rest => xlast(rest) }

val wrFirst : (list(xop), int) -> unit
function wrFirst(ds, v) = match ds { d :: _ => wr(d, v), [||] => () }

/* test_wait / try_wait: waitComplete, then the payload report of .phase_type::primary (reportPredicate, reportValue).
 * "For .layout::v0, reportPredicate and reportValue are always zero" (PTX ISA 9.3, mbarrier.test_wait); for
 * .layout::v1 the report is zero unless a tracked operation reported a failure, which no operation of the model does. */
val wrZeros : list(xop) -> unit
function wrZeros(ds) = match ds { d :: rest => { wr(d, 0); wrZeros(rest) }, [||] => () }
val wrWait : (list(xop), int) -> unit
function wrWait(ds, v) = match ds { d :: rest => { wr(d, v); wrZeros(rest) }, [||] => () }

val mbWaitOp : (list(string), string, list(xop), addr, xop) -> unit
function mbWaitOp(m, sp, ds, a, s) = {
  let k = mbKey(ea(a, sp));
  let v = rd(s, mkty("b", 64));
  let t = tchk(cur);
  if has(m, "try_wait") & not_bool(mbOk("try_wait", k)) then ()
  else if has(m, "try_wait") then {
    t_wd[t] = [|k, b2i(has(m, "parity")), v|];
    t_wx[t] = ds;
    t_cont[t] = K_wait("mbtry", 0);
    yielded = true
  } else if mbOk("test_wait", k) then {
    /* waitPhase "must be ... the current phase or the immediately preceding phase" (9.7.14.16.19): an older state is
       reported under .ub strict */
    if ub & not_bool(has(m, "parity")) & mbPhase(k) > land_int(v, 4294967295) + 1 then
      fault("mbarrier.test_wait: the state is older than the immediately preceding phase", v)
    else { let r = mbDone(k, m, v); if r then mbWaited(k); wrWait(ds, b2i(r)) }
  }
}

val exec_mbar : (opcode, list(xop)) -> bool
function exec_mbar(o, ops) = {
  let m = o.mods;
  let sp = spaceOf(m);
  let mc = has(m, "multicast::cluster::32b");
  if o.base != "mbarrier" then false
  else match ops {
    [|XB([|a|], [||]), n|] if has(m, "init") => {
      let c = rdu32(n);
      let k = mbKey(ea(a, sp));
      let l = if has(m, "layout::v1") then 1 else 0;
      if k == -1 then fault("mbarrier.init: address outside the shared memory windows", 0)
      /* "The behavior of performing an mbarrier.init operation on a memory location containing a valid mbarrier object
         is undefined"; count in [1, 2^20 - 1] (v0) or [1, 2^9 - 1] (v1) (9.7.14.16.12): under .ub strict */
      else if not_bool(mbAligned("init", k)) then ()
      else if ub & mbHas(k) then fault("mbarrier.init on a valid mbarrier object", 0)
      else if ub & (c < 1 | c > mbMaxCount(l)) then fault("mbarrier.init: count outside the range of the layout", c)
      else mbPut(k, (0, c, c, 0, l));
      true
    },
    [|XB([|a|], [||])|] if has(m, "inval") => { let k = mbKey(ea(a, sp)); if mbOk("inval", k) then mbDel(k); true },
    [|XB([|a|], [||]), n, mk|] if (has(m, "expect_tx") | has(m, "complete_tx")) & not_bool(has(m, "arrive")) & mc => {
      let v = if has(m, "complete_tx") then negate(rdu32(n)) else rdu32(n);
      var ks = mbMcKeys(land_int(locOffA(ea(a, sp)), 16777215), rdu32(mk));
      while is_cons(ks) do { match ks { k :: rest => { if mbHas(k) then mbPut(k, mbTx(mbGet(k), v)); ks = rest }, [||] => () } };
      true
    },
    XB([|a|], [||]) :: n :: _ if (has(m, "expect_tx") | has(m, "complete_tx")) & not_bool(has(m, "arrive") | has(m, "arrive_drop")) & not_bool(mc) => {
      let k = mbKey(ea(a, sp));
      if mbOk(if has(m, "complete_tx") then "complete_tx" else "expect_tx", k) then {
        let (p, q, e, t, l) = mbGet(k);
        let t1 = t + (if has(m, "complete_tx") then negate(rdu32(n)) else rdu32(n));
        if mbUbState("expect_tx", (p, q, e, t1, l)) then mbPut(k, mbCheck((p, q, e, t1, l)))
      };
      true
    },
    _ :: XB([|a|], [||]) :: rest if has(m, "arrive") & mc => {
      let mk : int = match xlast(rest) { Some(x) => rdu32(x), None() => 0 };
      let args = xdrop_last(rest);
      let tx = mbTxArg(m, args);
      let cnt = mbCountArg(m, args);
      var ks = mbMcKeys(land_int(locOffA(ea(a, sp)), 16777215), mk);
      while is_cons(ks) do { match ks { k :: rr => { if mbHas(k) then mbPut(k, mbArrive(mbTx(mbGet(k), tx), cnt)); ks = rr }, [||] => () } };
      true
    },
    d :: XB([|a|], [||]) :: rest if (has(m, "arrive") | has(m, "arrive_drop")) & not_bool(mc) => {
      let k = mbKey(ea(a, sp));
      if mbOk(if has(m, "arrive_drop") then "arrive_drop" else "arrive", k) & mbArriveOk(if has(m, "arrive_drop") then "arrive_drop" else "arrive", k) then {
        let s = mbGet(k);
        let (p0, q0, e0, t0, l0) = s;
        let n = mbCountArg(m, rest);
        let t1 = t0 + mbTxArg(m, rest);
        /* count "must be in the range" of Table 37, so at most the pending count; a .noComplete arrive "must not
           cause the mbarrier to complete its current phase" (9.7.14.16.16): under .ub strict */
        if ub & (n < 1 | n > q0) then fault("mbarrier.arrive: arrive count larger than the pending count", n)
        else if mbUbState("arrive", (p0, q0, e0, t1, l0)) then {
          let s2 = if has(m, "arrive_drop") then mbDrop((p0, q0, e0, t1, l0), n) else mbArrive((p0, q0, e0, t1, l0), n);
          let (p2, _, _, _, _) = s2;
          if ub & has(m, "noComplete") & p2 != p0 then fault("mbarrier.arrive.noComplete completes the current phase", 0)
          else { mbPut(k, s2); wr(d, mbState(s)) }
        }
      };
      true
    },
    d :: XB([|a|], [||]) :: s :: _ if has(m, "test_wait") | has(m, "try_wait") => { mbWaitOp(m, sp, dests(d), a, s); true },
    d :: rv :: XB([|a|], [||]) :: s :: _ if (has(m, "test_wait") | has(m, "try_wait")) & has(m, "phase_type::primary") => {
      mbWaitOp(m, sp, pend_append(dests(d), [|rv|]), a, s);
      true
    },
    [|d, s|] if has(m, "pending_count") => { wr(d, land_int(asr_int(rd(s, mkty("b", 64)), 32), 1048575)); true },
    [|d, XB([|a|], [||])|] if has(m, "check_layout") => {
      let k = mbKey(ea(a, sp));
      if mbOk("check_layout", k) then { let (_, _, _, _, l) = mbGet(k); wr(d, b2i(l == (if has(m, "layout::v1") then 1 else 0))) };
      true
    },
    _ => false
  }
}

/* try_wait: "Suspended thread resumes execution when the specified phase completes OR before the phase completes
 * following a system-dependent time limit" (9.7.14.16.19).  The model's time limit expires when no thread can otherwise
 * make progress: every suspended try_wait then returns False. */
function ext_timeout() = {
  var any : bool = false;
  foreach (t from 0 to (nthreads - 1)) {
    let s = tchk(t);
    if t_alive[s] then match t_cont[s] {
      K_wait("mbtry", _) => { let sv = cur; cur = t; wrWait(t_wx[s], 0); cur = sv; t_cont[s] = K_none(); any = true },
      _ => ()
    }
  };
  any
}

/* the blocked waits */
function ext_wait_ready() = {
  let t = tchk(cur);
  match t_cont[t] {
    K_wait("clwait", _) => {
      let id = at(t_wd[t], 0);
      if clGet(id, "g") >= at(t_wd[t], 1) then { t_cont[t] = K_none(); true } else false
    },
    K_wait("mbtry", _) => {
      let k = at(t_wd[t], 0);
      let m : list(string) = if at(t_wd[t], 1) == 1 then [|"parity"|] else [||];
      if mbHas(k) & mbDone(k, m, at(t_wd[t], 2)) then { mbWaited(k); wrWait(t_wx[t], 1); t_cont[t] = K_none(); true } else false
    },
    _ => ext_wait_ready2()
  }
}

/* ---------------------------------------------------------------- cp.async */
/* the bytes read from src: the optional src-size / ignore-src operand, which comes before the
 * cache_policy operand; with .L2::cache_hint the last operand is cache_policy (PTX ISA 9.3, cp.async
 * syntax: cp-size{, src-size}{, cache_policy}, and "When the optional argument cache_policy is specified,
 * the qualifier .level::cache_hint is required") */
val cpSrcN : (int, list(xop), bool) -> int
function cpSrcN(n, rest, hint) = match rest {
  [||] => n,
  [|_|] if hint => n,
  XS(SI(x)) :: _ => min_int(x, n),
  x :: _ => if regW(x) == 1 then (if rdu32(x) == 0 then n else 0) else min_int(rdu32(x), n)
}

/* #bulkTx(Mb, N): complete-tx of N bytes on an existing mbarrier */
val bulkTx : (loc, int) -> unit
function bulkTx(mb, n) = {
  let k = mbKey(mb);
  if mbHas(k) then mbPut(k, mbTx(mbGet(k), negate(n)))
  else fault("complete-tx on an mbarrier that is not initialised", 0)
}

/* ---------------------------------------------------------------- bulk copies */
/* cp.async.bulk / cp.reduce.async.bulk (9.7.9.26.4.1, 9.7.9.26.4.2): "size must be a multiple of 16. If the value is not
 * a multiple of 16, then the behavior is undefined", "The addresses dstMem and srcMem must be aligned to 16 bytes";
 * from .shared::cta to .shared::cluster "the destination has to be in the shared memory of a different CTA".  Under
 * .ub strict. */
val bulkUb : (string, int, loc, loc, bool) -> bool
function bulkUb(op, n, sl, dl, remote) =
  if not_bool(ub) then true
  else if tmod_int(n, 16) != 0 then { fault(concat_str(op, ": size is not a multiple of 16"), n); false }
  else match (sl, dl) {
    (Loc(_, sa), Loc(ds, da)) =>
      if tmod_int(sa, 16) != 0 | tmod_int(da, 16) != 0 then { fault(concat_str(op, ": addresses not aligned to 16 bytes"), 0); false }
      else if remote & (ds == "shared" | (ds == "cluster" & tdiv_int(mbKey(dl), 4294967296) == cur_cta())) then {
        fault(concat_str(op, ": the .shared::cluster destination is in the executing CTA"), 0); false
      } else true,
    _ => true
  }


val bulkMbar : list(xop) -> option(addr)
function bulkMbar(l) = match l { [||] => None(), XB(a :: _, _) :: _ => Some(a), _ :: rest => bulkMbar(rest) }
val bulkMask : list(xop) -> int
function bulkMask(l) = match l { XB(_, _) :: x :: _ => rdu32(x), XB(_, _) :: [||] => 0, [||] => 0, _ :: rest => bulkMask(rest) }
val lastArg : list(xop) -> int
function lastArg(l) = match xlast(l) { Some(x) => rdu32(x), None() => 0 }

val isMc : list(string) -> bool
function isMc(m) = has(m, "multicast::cluster") | has(m, "multicast::cluster::16b") | has(m, "multicast::cluster::32b")

val rankLocs : (int, int) -> list(loc)
function rankLocs(off, mk) = {
  var out : list(loc) = [||];
  foreach (r from 0 to 31)
    if bit_int(mk, r) & asr_int(mk, r) > 0 then out = pend_append(out, [|Loc("cluster", lor_int(shl_int(r + 1, 24), off))|]);
  out
}
val bulkLocs : (loc, int, bool) -> list(loc)
function bulkLocs(l, mk, mc) = if not_bool(mc) then [|l|] else match l { Loc(_, a) => rankLocs(land_int(a, 16777215), mk), _ => [|l|] }

val expandMask : (int, int) -> int
function expandMask(mk, chunks) = {
  var r : int = 0;
  foreach (i from 0 to (chunks * 16 - 1)) if bit_int(mk, fmod_int(i, 16)) then r = lor_int(r, shl_int(255, 8 * i));
  r
}

/* "The current implementation of cp.reduce.async.bulk.add.f32 flushes subnormal inputs and results to
 * sign-preserving zero" (ISA 9.7.9.26.4.2) */
val bulkOp : (string, ty) -> string
function bulkOp(op, t) = if op == "add" & ty_eq(t, mkty("f", 32)) then "add.ftz" else op

/* the PTX name of a type (.u32, .bf16, .f16x2, .e4m3x2, ...) */
val tnm : ty -> string
function tnm(t) =
  if t.k == "bf" & t.w == 16 then "bf16" else if t.k == "tf" & t.w == 32 then "tf32"
  else if string_length(t.k) > 2 then t.k else concat_str(t.k, dec_str(t.w))

/* cp.reduce.async.bulk: "the valid combinations of .redOp and element type" (PTX ISA 9.3 ~16477-16486) and the .noftz
 * form: ".add.f16 and .add.bf16 operations require .noftz qualifier" (~16471), which is valid only for them (~16426-16437).
 * "" when legal, else the error.  Reported by `.isacheck ;` (isaTables, ptx_mm.sail).  Without it the other combinations
 * run with the natural meaning of the operation on the type: integer operations on .b32 / .b64 as unsigned, .and / .or /
 * .xor on the bits of any type, .add / .min / .max on floats as in atom (.add.f16 / .bf16 without .noftz as with it, the
 * only behaviour the ISA defines; .noftz is ignored elsewhere); .inc / .dec on a floating-point type have no meaning and
 * stop with an error (bulkRedRun). */
val bulkRedErr : (bool, string, ty, bool) -> string
function bulkRedErr(glob, op, t, noftz) = {
  let n = tnm(t);
  let ok : bool = match op {
    "add" => if glob then has([|"u32", "s32", "u64", "f32", "f64", "f16", "bf16"|], n) else has([|"u32", "s32", "u64"|], n),
    "min" => if glob then has([|"u32", "s32", "u64", "s64", "f16", "bf16"|], n) else has([|"u32", "s32"|], n),
    "max" => if glob then has([|"u32", "s32", "u64", "s64", "f16", "bf16"|], n) else has([|"u32", "s32"|], n),
    "inc" => n == "u32",
    "dec" => n == "u32",
    "and" => if glob then has([|"b32", "b64"|], n) else n == "b32",
    "or" => if glob then has([|"b32", "b64"|], n) else n == "b32",
    "xor" => if glob then has([|"b32", "b64"|], n) else n == "b32",
    _ => false
  };
  let half = op == "add" & (n == "f16" | n == "bf16");
  if not_bool(ok) then concat_str("cp.reduce.async.bulk: .", concat_str(op, concat_str(" with .", concat_str(n,
    if glob then " is not a valid .redOp / type combination for .global" else " is not a valid .redOp / type combination for .shared::cluster"))))
  else if half & not_bool(noftz) then concat_str("cp.reduce.async.bulk: .add.", concat_str(n, " requires .noftz"))
  else if noftz & not_bool(half) then "cp.reduce.async.bulk: .noftz is only valid with .add.f16 / .add.bf16"
  else ""
}

/* the combinations that have no meaning even outside the ISA tables: .inc / .dec on floating-point elements */
val isFloatTy : ty -> bool
function isFloatTy(t) = t.k == "f" | t.k == "bf" | t.k == "tf" | t.k == "f16x2" | t.k == "bf16x2" | t.k == "e4m3" | t.k == "e5m2"
val bulkRedRun : (string, ty) -> string
function bulkRedRun(op, t) =
  if (op == "inc" | op == "dec") & isFloatTy(t) then concat_str("cp.reduce.async.bulk: .", concat_str(op, concat_str(" on floating-point type .", tnm(t))))
  else ""

val exec_async : (opcode, list(xop)) -> bool
function exec_async(o, ops) = {
  let m = o.mods;
  if o.base != "cp" | not_bool(has(m, "async")) | has(m, "tensor") then false
  else if not_bool(has(m, "bulk")) then {
    if has(m, "commit_group") | has(m, "wait_all") then {
      /* a new cp.async-group starts (wait_all commits too) */
      if ub then {
        let mp = t_misc[tchk(cur)];
        var i : int = 0;
        while i < smap_size(mp) do { let k = smap_key(mp, i); if stake(k, 4) == "cpa:" then smap_del(mp, k) else i = i + 1 }
      };
      true
    } else if has(m, "wait_group") then true
    else if has(m, "mbarrier") then match ops {
      [|XB([|a|], [||])|] => {
        /* "If no state space is specified then Generic Addressing is used" (9.7.14.16.18) */
        let k = mbKey(ea(a, if has(m, "shared") then "shared" else "generic"));
        if mbOk("arrive", k) & mbArriveOk("arrive", k) then {
          let (p, q, e, t, l) = mbGet(k);
          if has(m, "noinc") then mbPut(k, mbArrive((p, q, e, t, l), 1))
          /* without .noinc the pending count is incremented first and "should not exceed the limit ... Otherwise, the
             behavior is undefined" (9.7.14.16.18): under .ub strict */
          else if ub & q + 1 > mbMaxCount(l) then fault("cp.async.mbarrier.arrive: pending count beyond the limit", q + 1)
        };
        true
      },
      _ => false
    } else match ops {
      XB([|d|], [||]) :: XB([|s|], [||]) :: XS(SI(n)) :: rest => {
        let k = cpSrcN(n, rest, has(m, "L2::cache_hint"));
        let dl = ea(d, "shared");
        /* "Specifying src-size larger than cp-size results in undefined behavior"; "two or more cp.async operations
           within a cp.async-group copying data to the same location results in undefined behavior" (9.7.9.26.3.1):
           under .ub strict, with "cpa:<byte>" in the thread's misc space for the bytes of the open group */
        if ub then {
          let sz : int = match rest { XS(SI(x)) :: _ => x, x :: _ if regW(x) != 1 & not_bool(has(m, "L2::cache_hint") & xop_count(rest) == 1) => rdu32(x), _ => 0 };
          if sz > n then { fault("cp.async: src-size larger than cp-size", sz); return true };
          match dl { Loc(_, off) => {
            let mp = t_misc[tchk(cur)];
            foreach (b from 0 to (n - 1)) {
              if smap_has(mp, concat_str("cpa:", dec_str(off + b))) & not_bool(stuck) then
                fault("cp.async: two cp.async of one cp.async-group copy to the same location", off + b);
              smap_set(mp, concat_str("cpa:", dec_str(off + b)), 1)
            }
          }, _ => () };
          if stuck then return true
        };
        let v = if k > 0 then memop_do(ea(s, "generic"), RdOp(k)) else 0;
        if not_bool(stuck) then { let _ = memop_do(dl, WrOp(n, v)); () };
        true
      },
      _ => false
    }
  } else if has(m, "prefetch") | has(m, "commit_group") | has(m, "wait_group") then true
  else match ops {
    XB([|d|], [||]) :: XB([|s|], [||]) :: sz :: rest => {
      let n = rdu32(sz);
      let rerr = if has(m, "reduce") then bulkRedRun(atomOp(m), ty1(o)) else "";
      if rerr != "" then { fault(rerr, 0); true }
      else if has(m, "reduce") & not_bool(bulkUb("cp.reduce.async.bulk", n, ea(s, "shared"), ea(d, if has(m, "global") then "generic" else "shared_cluster"), false)) then true
      else if has(m, "reduce") then {
        let v = memop_do(ea(s, "shared"), RdOp(n));
        let dl = ea(d, if has(m, "global") then "generic" else "shared_cluster");
        let mb : option(loc) = match bulkMbar(rest) { Some(a) => Some(ea(a, "shared_cluster")), None() => None() };
        if not_bool(stuck) then { let _ = memop_do(dl, BulkRedOp(bulkOp(atomOp(m), ty1(o)), ty1(o), n, v)); () };
        if not_bool(stuck) then match mb { Some(l) => bulkTx(l, n), None() => () };
        true
      } else if has(m, "mbarrier::complete_tx::bytes") & is_some(bulkMbar(rest)) then {
        let cl = if has(m, "shared::cluster") then "shared_cluster" else "shared";
        if not_bool(bulkUb("cp.async.bulk", n, ea(s, if has(m, "global") then "generic" else "shared"), ea(d, cl),
                           not_bool(has(m, "global")) & has(m, "shared::cluster"))) then return true;
        let v = memop_do(ea(s, if has(m, "global") then "generic" else "shared"), RdOp(n));
        let mk = bulkMask(rest);
        var ds = bulkLocs(ea(d, cl), mk, isMc(m));
        var ms : list(loc) = match bulkMbar(rest) { Some(a) => bulkLocs(ea(a, cl), mk, isMc(m)), None() => [||] };
        while is_cons(ds) & is_cons(ms) & not_bool(stuck) do {
          match (ds, ms) {
            (dl :: dr, ml :: mr) => { let _ = memop_do(dl, WrOp(n, v)); if not_bool(stuck) then bulkTx(ml, n); ds = dr; ms = mr },
            _ => ()
          }
        };
        true
      } else if has(m, "bulk_group") then {
        if not_bool(bulkUb("cp.async.bulk", n, ea(s, "shared"), ea(d, "generic"), false)) then return true;
        let v = memop_do(ea(s, "shared"), RdOp(n));
        let dl = ea(d, "generic");
        if not_bool(stuck) then {
          if not_bool(has(m, "cp_mask")) then { let _ = memop_do(dl, WrOp(n, v)); () }
          else {
            let old = memop_do(dl, RdOp(n));
            let em = expandMask(lastArg(rest), tdiv_int(n, 16));
            if not_bool(stuck) then { let _ = memop_do(dl, WrOp(n, lor_int(land_int(v, em), land_int(old, lnot_int(em))))); () }
          }
        };
        true
      } else false
    },
    _ => false
  }
}

/* ---------------------------------------------------------------- PTX-EXEC-MISC (ptx-exec-misc.k): st.async, red.async, st.bulk,
 * clusterlaunchcontrol */
/* the conditions of the mbarrier forms of st.async (9.7.9.12) and red.async (9.7.14.7), under .ub strict: a and mbar in
 * the .shared::cluster window and in the same CTA; st.async: "%cluster_nctarank > 1 is true"; red.async: "The CTA to
 * which they belong is different from the CTA of the executing thread" */
val asyncUb : (string, loc, loc) -> bool
function asyncUb(op, d, mb) =
  if not_bool(ub) then true
  else {
    let kd = mbKey(d);
    let km = mbKey(mb);
    if kd == -1 | km == -1 then { fault(concat_str(op, ": address outside the .shared::cluster window"), 0); false }
    else if tdiv_int(kd, 4294967296) != tdiv_int(km, 4294967296) then { fault(concat_str(op, ": a and mbar belong to different CTAs"), 0); false }
    else if op == "st.async" & rreg("%cluster_nctarank") <= 1 then { fault("st.async: the cluster has a single CTA", 0); false }
    else if op == "red.async" & tdiv_int(kd, 4294967296) == cur_cta() then { fault("red.async: a and mbar belong to the executing CTA", 0); false }
    else true
  }

val exec_amisc : (opcode, list(xop)) -> bool
function exec_amisc(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    ("st", [|XB([|a|], [||]), v, XB([|mb|], [||])|]) => {
      if has(m, "async") then {
        let t = ty1(o);
        let vs : list(int) = match v { XV(l) => rdList(l, t), _ => [|rd(v, t)|] };
        /* "If .ss is not specified, Generic Addressing is used" for a and mbar (PTX ISA 9.3 st.async ~13917) */
        let sp = if has(m, "shared::cluster") then "shared_cluster" else "generic";
        var l = ea(a, sp);
        if not_bool(asyncUb("st.async", l, ea(mb, sp))) then return true;
        var n : int = 0;
        var r = vs;
        while is_cons(r) & not_bool(stuck) do {
          match r { x :: rest => { let _ = memop_do(l, WrOp(bytes(t), x)); l = loc_add(l, bytes(t)); n = n + bytes(t); r = rest }, [||] => () }
        };
        if not_bool(stuck) then bulkTx(ea(mb, sp), n);
        true
      } else false
    },
    ("red", [|XB([|a|], [||]), v, XB([|mb|], [||])|]) => {
      if has(m, "async") then {
        let t = ty1(o);
        /* "If .ss is not specified, Generic Addressing is used" (PTX ISA 9.3 red.async ~21414) */
        let sp = if has(m, "shared::cluster") then "shared_cluster" else "generic";
        if not_bool(asyncUb("red.async", ea(a, sp), ea(mb, sp))) then return true;
        let _ = memop_do(ea(a, sp), AtOp(atomOp(m), t, rd(v, t), 0));
        if not_bool(stuck) then bulkTx(ea(mb, sp), bytes(t));
        true
      } else false
    },
    ("st", [|XB([|a|], [||]), sz, _|]) => {
      /* "If no state space is specified then Generic Addressing is used"; "size must be a multiple of 8. If the value is
         not a multiple of 8, then the behavior is undefined"; an address outside the .shared window is undefined
         (9.7.9.14).  st.bulk states no alignment for a, so the region is written in 8-byte pieces when a is 8-byte
         aligned and byte by byte otherwise (not as one access of size bytes, which the alignment check would treat as
         a naturally aligned 16-byte access) */
      if has(m, "bulk") then {
        let n = rdu32(sz);
        let l0 = ea(a, if has(m, "shared::cta") then "shared" else "generic");
        match l0 {
          Loc(sp, off) => {
            if ub & sp != "shared" then fault("st.bulk: address outside the .shared window", off)
            else if ub & tmod_int(n, 8) != 0 then fault("st.bulk: size is not a multiple of 8", n)
            else {
              let step = if tmod_int(off, 8) == 0 & tmod_int(n, 8) == 0 then 8 else 1;
              var i : int = 0;
              while i < n & not_bool(stuck) do { let _ = memop_do(Loc(sp, off + i), WrOp(step, 0)); i = i + step }
            }
          },
          _ => { let _ = memop_do(l0, WrOp(n, 0)); () }
        };
        true
      } else false
    },
    ("clusterlaunchcontrol", [|XB([|a|], [||]), XB([|mb|], [||])|]) => {
      if has(m, "try_cancel") then {
        /* "If the .space qualifier is specified, both operands addr and mbar must be in the .shared::cta state space.
         * Otherwise, generic addressing will be assumed for both"; with .multicast::cluster::all "the response is
         * asynchronously written ... to the corresponding local shared memory addr of each CTA in the requesting
         * cluster", signalled by a complete-tx on the mbarrier "on the shared memory of that CTA" (PTX ISA 9.3,
         * clusterlaunchcontrol.try_cancel).  The response is always the failure response (0). */
        let sp = if has(m, "shared") then "shared" else "generic";
        /* "The result is undefined if any of address operands do not fall within the address window of .shared::cta" */
        if not_bool(winUb("clusterlaunchcontrol.try_cancel", "addr", ea(a, sp), [|"shared"|]) & winUb("clusterlaunchcontrol.try_cancel", "mbar", ea(mb, sp), [|"shared"|])) then return true;
        let mc = has(m, "multicast::cluster::all");
        let all = pow2(vol(clusterDim)) - 1;
        var ds = if mc then rankLocs(land_int(locOffA(ea(a, sp)), 16777215), all) else [|ea(a, sp)|];
        var ms = if mc then rankLocs(land_int(locOffA(ea(mb, sp)), 16777215), all) else [|ea(mb, sp)|];
        while is_cons(ds) & is_cons(ms) & not_bool(stuck) do {
          match (ds, ms) {
            (dl :: dr, ml :: mr) => { let _ = memop_do(dl, WrOp(16, 0)); if not_bool(stuck) then bulkTx(ml, 16); ds = dr; ms = mr },
            _ => ()
          }
        };
        true
      } else false
    },
    ("clusterlaunchcontrol", [|d, _|]) => if has(m, "query_cancel") then { wr(d, 0); true } else false,
    _ => false
  }
}

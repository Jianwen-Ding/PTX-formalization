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
  smap_del(SP_MBARS, concat_str("t", ks)); smap_del(SP_MBARS, concat_str("l", ks))
}

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
  if has(m, "try_wait") then {
    t_wd[t] = [|k, b2i(has(m, "parity")), v|];
    t_wx[t] = ds;
    t_cont[t] = K_wait("mbtry", 0);
    yielded = true
  } else if mbHas(k) then wrWait(ds, b2i(mbDone(k, m, v)))
  else fault("mbarrier.test_wait on an object that is not initialised", 0)
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
      mbPut(mbKey(ea(a, sp)), (0, c, c, 0, if has(m, "layout::v1") then 1 else 0));
      true
    },
    [|XB([|a|], [||])|] if has(m, "inval") => { mbDel(mbKey(ea(a, sp))); true },
    [|XB([|a|], [||]), n, mk|] if (has(m, "expect_tx") | has(m, "complete_tx")) & not_bool(has(m, "arrive")) & mc => {
      let v = if has(m, "complete_tx") then negate(rdu32(n)) else rdu32(n);
      var ks = mbMcKeys(land_int(locOffA(ea(a, sp)), 16777215), rdu32(mk));
      while is_cons(ks) do { match ks { k :: rest => { if mbHas(k) then mbPut(k, mbTx(mbGet(k), v)); ks = rest }, [||] => () } };
      true
    },
    XB([|a|], [||]) :: n :: _ if (has(m, "expect_tx") | has(m, "complete_tx")) & not_bool(has(m, "arrive") | has(m, "arrive_drop")) & not_bool(mc)
                                  & mbHas(mbKey(ea(a, sp))) => {
      let k = mbKey(ea(a, sp));
      mbPut(k, mbTx(mbGet(k), if has(m, "complete_tx") then negate(rdu32(n)) else rdu32(n)));
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
    d :: XB([|a|], [||]) :: rest if (has(m, "arrive") | has(m, "arrive_drop")) & not_bool(mc) & mbHas(mbKey(ea(a, sp))) => {
      let k = mbKey(ea(a, sp));
      let s = mbGet(k);
      let s1 = mbTx(s, mbTxArg(m, rest));
      mbPut(k, if has(m, "arrive_drop") then mbDrop(s1, mbCountArg(m, rest)) else mbArrive(s1, mbCountArg(m, rest)));
      wr(d, mbState(s));
      true
    },
    d :: XB([|a|], [||]) :: s :: _ if has(m, "test_wait") | has(m, "try_wait") => { mbWaitOp(m, sp, dests(d), a, s); true },
    d :: rv :: XB([|a|], [||]) :: s :: _ if (has(m, "test_wait") | has(m, "try_wait")) & has(m, "phase_type::primary") => {
      mbWaitOp(m, sp, pend_append(dests(d), [|rv|]), a, s);
      true
    },
    [|d, s|] if has(m, "pending_count") => { wr(d, land_int(asr_int(rd(s, mkty("b", 64)), 32), 1048575)); true },
    [|d, XB([|a|], [||])|] if has(m, "check_layout") => {
      let (_, _, _, _, l) = mbGet(mbKey(ea(a, sp)));
      wr(d, b2i(l == (if has(m, "layout::v1") then 1 else 0)));
      true
    },
    _ => false
  }
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
      if mbHas(k) & mbDone(k, m, at(t_wd[t], 2)) then { wrWait(t_wx[t], 1); t_cont[t] = K_none(); true } else false
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

val exec_async : (opcode, list(xop)) -> bool
function exec_async(o, ops) = {
  let m = o.mods;
  if o.base != "cp" | not_bool(has(m, "async")) | has(m, "tensor") then false
  else if not_bool(has(m, "bulk")) then {
    if has(m, "commit_group") | has(m, "wait_group") | has(m, "wait_all") then true
    else if has(m, "mbarrier") then match ops {
      [|XB([|a|], [||])|] => {
        if has(m, "noinc") then {
          let k = mbKey(ea(a, "shared"));
          if mbHas(k) then { mbPut(k, mbArrive(mbGet(k), 1)); true } else false
        } else true
      },
      _ => false
    } else match ops {
      XB([|d|], [||]) :: XB([|s|], [||]) :: XS(SI(n)) :: rest => {
        let k = cpSrcN(n, rest, has(m, "L2::cache_hint"));
        let dl = ea(d, "shared");
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
      if has(m, "reduce") then {
        let v = memop_do(ea(s, "shared"), RdOp(n));
        let dl = ea(d, if has(m, "global") then "generic" else "shared_cluster");
        let mb : option(loc) = match bulkMbar(rest) { Some(a) => Some(ea(a, "shared_cluster")), None() => None() };
        if not_bool(stuck) then { let _ = memop_do(dl, BulkRedOp(bulkOp(atomOp(m), ty1(o)), ty1(o), n, v)); () };
        if not_bool(stuck) then match mb { Some(l) => bulkTx(l, n), None() => () };
        true
      } else if has(m, "mbarrier::complete_tx::bytes") & is_some(bulkMbar(rest)) then {
        let cl = if has(m, "shared::cluster") then "shared_cluster" else "shared";
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
val exec_amisc : (opcode, list(xop)) -> bool
function exec_amisc(o, ops) = {
  let m = o.mods;
  match (o.base, ops) {
    ("st", [|XB([|a|], [||]), v, XB([|mb|], [||])|]) => {
      if has(m, "async") then {
        let t = ty1(o);
        let vs : list(int) = match v { XV(l) => rdList(l, t), _ => [|rd(v, t)|] };
        var l = ea(a, "shared_cluster");
        var n : int = 0;
        var r = vs;
        while is_cons(r) & not_bool(stuck) do {
          match r { x :: rest => { let _ = memop_do(l, WrOp(bytes(t), x)); l = loc_add(l, bytes(t)); n = n + bytes(t); r = rest }, [||] => () }
        };
        if not_bool(stuck) then bulkTx(ea(mb, "shared_cluster"), n);
        true
      } else false
    },
    ("red", [|XB([|a|], [||]), v, XB([|mb|], [||])|]) => {
      if has(m, "async") then {
        let t = ty1(o);
        let _ = memop_do(ea(a, "shared_cluster"), AtOp(atomOp(m), t, rd(v, t), 0));
        if not_bool(stuck) then bulkTx(ea(mb, "shared_cluster"), bytes(t));
        true
      } else false
    },
    ("st", [|XB([|a|], [||]), sz, _|]) => {
      if has(m, "bulk") then { let _ = memop_do(ea(a, "shared"), WrOp(rdu32(sz), 0)); true } else false
    },
    ("clusterlaunchcontrol", [|XB([|a|], [||]), XB([|mb|], [||])|]) => {
      if has(m, "try_cancel") then {
        /* "If the .space qualifier is specified, both operands addr and mbar must be in the .shared::cta state space.
         * Otherwise, generic addressing will be assumed for both"; with .multicast::cluster::all "the response is
         * asynchronously written ... to the corresponding local shared memory addr of each CTA in the requesting
         * cluster", signalled by a complete-tx on the mbarrier "on the shared memory of that CTA" (PTX ISA 9.3,
         * clusterlaunchcontrol.try_cancel).  The response is always the failure response (0). */
        let sp = if has(m, "shared") then "shared" else "generic";
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

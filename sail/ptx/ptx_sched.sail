/* =============================================================================
 * Instruction fetch, thread exit, the grid launch and the scheduler
 * (ptx-exec-core.k fetch rules, ptx-exec-sync.k exit rules, ptx-harness.k launch).
 *
 * Schedule.  K explores the interleavings of the threads at instruction granularity with the turn
 * of ptx-configuration.k: a thread holding the turn runs its thread-local instructions and gives
 * the turn up right before a visible instruction (memory outside .param/.local, barriers,
 * collectives, exit).  This model runs one deterministic schedule of that system: the threads take
 * the turn round-robin, in creation order (CTA order, then thread index); each turn executes the
 * thread's next instruction followed by its thread-local instructions up to the next visible one.
 * ============================================================================= */

/* the module hooks for collectives and visible instructions of the extension modules */
val isCollective : string -> bool
val exec_dispatch : (opcode, list(xop)) -> unit

/* other modules' blocking continuations */
val ext_wait_ready : unit -> bool
val ext_wait_ready2 : unit -> bool
val mkCtas : (int, int) -> unit

val isVisible : item -> bool
function isVisible(it) = match it {
  INS(_, o, _) => {
    let op = o.base;
    let m = o.mods;
    (has([|"ld", "st", "atom", "red", "multimem", "fabric"|], op) & not_bool(spaceOf(m) == "param" | spaceOf(m) == "local"))
    | isBar(op, m) | isBarRed(op, m) | isBarWarp(op, m) | isBarCluster(op, m) | op == "exit" | isCollective(op)
  },
  _ => false
}

/* fetch(ins(G, O, Os), C): guards are evaluated at fetch time */
val fetch_exec : item -> unit
function fetch_exec(it) = match it {
  INS(g, o, ops) => match g {
    None() => exec_dispatch(o, ops),
    Some((p, neg)) => if (rreg(p) != 0) != neg then exec_dispatch(o, ops)
  },
  DECLVAR(sp, nm, sz, al) => declVar(sp, nm, sz, al)
}

/* ---------------------------------------------------------------- thread exit (#exit ~> #barCheckAll ~> #clExit ~> #die) */
val cluster_exit : unit -> unit

function do_exit() = {
  let t = tchk(cur);
  let c = cchk(t_cta[t]);
  let w = dec_str(tdiv_int(t_tid[t], 32));
  let wl = c_wlive[c];
  smap_set(wl, w, land_int(smap_get(wl, w), lnot_int(pow2(fmod_int(t_tid[t], 32)))));
  c_live[c] = c_live[c] - 1;
  /* #barCheckAll(keys_list(BC)): every barrier with arrivals (ids beyond 15 included), collected before checking */
  let bs = c_bars[cchk(t_cta[t])];
  var ids : list(int) = [||];
  foreach (i from 0 to (smap_size(bs) - 1)) {
    let k = smap_key(bs, i);
    if string_startswith(k, "cnt:") then ids = str2base(sdrop(k, 4), 10) :: ids
  };
  while is_cons(ids) do match ids { b :: rest => { barCheck(t_cta[t], b); ids = rest }, [||] => () };
  cluster_exit();
  t_alive[t] = false;
  t_status[t] = 2;
  yielded = true
}

/* ---------------------------------------------------------------- one turn */
val runnable : int -> bool
function runnable(t) = {
  let s = tchk(t);
  if not_bool(t_alive[s]) then false
  else {
    /* waiting(B, G) => running once the barrier's generation moved on */
    if t_status[s] == 1 then {
      cur = t;
      if bar_get("gen:", t_wbar[s], 0) > t_wgen[s] then t_status[s] = 0
    };
    if t_status[s] != 0 then false
    else match t_cont[s] {
      K_none() => true,
      K_barwaitred(b, g, d) => {
        cur = t;
        if bar_get("gen:", b, 0) > g then { wr(d, bar_get("res:", b, 0)); t_cont[s] = K_none(); true } else false
      },
      K_collwait(_, _) => {
        cur = t;
        if coll_ready(t) then { coll_take(); true } else false
      },
      K_wait(_, _) => { cur = t; ext_wait_ready() }
    }
  }
}

register steps : int = 0

val run_turn : int -> bool
/* returns true if the thread made progress */
function run_turn(t) = {
  let s = tchk(t);
  cur = t;
  yielded = false;
  var first : bool = true;
  var progress : bool = false;
  var go : bool = true;
  while go do {
    if stuck | not_bool(t_alive[s]) | t_status[s] != 0 | t_cont[s] != K_none() then go = false
    else {
      let f = t_fn[s];
      let pc = t_pc[s];
      if pc < 0 | pc >= f_len[fchk(f)] then {
        /* falling off the end of a function body is an implicit ret */
        do_ret();
        first = false;
        progress = true
      } else {
        let it = get_item(f_base[fchk(f)] + pc);
        if not_bool(first) & isVisible(it) then go = false
        else {
          first = false;
          t_pc[s] = pc + 1;
          fetch_exec(it);
          progress = true;
          steps = steps + 1
        }
      }
    };
    if yielded then go = false
  };
  progress
}

/* ---------------------------------------------------------------- the grid */
val any_alive : unit -> bool
function any_alive() = {
  var r : bool = false;
  foreach (t from 0 to (nthreads - 1)) { if t_alive[tchk(t)] then r = true };
  r
}

/* run the grid until every CTA has finished (#startGrid ~> #waitGrid) */
val run_grid : unit -> unit
function run_grid() = {
  var go : bool = true;
  while go & not_bool(stuck) do {
    var progress : bool = false;
    foreach (t from 0 to (nthreads - 1)) {
      if not_bool(stuck) & runnable(t) then { if run_turn(t) then progress = true }
    };
    /* finished CTAs leave; with an occupancy limit the next CTA starts */
    foreach (c from 0 to (ncta_slots - 1)) {
      if c_alive[cchk(c)] & c_live[cchk(c)] == 0 then { c_alive[cchk(c)] = false; resident = resident - 1 }
    };
    if not_bool(any_alive()) then {
      if nextcta >= vol(gridDim) then go = false
      else if occ > 0 & resident < occ then { mkCtas(nextcta, nextcta + 1); progress = true }
    } else if occ > 0 & resident < occ & nextcta < vol(gridDim) then { mkCtas(nextcta, nextcta + 1); progress = true };
    if go & not_bool(progress) then { fault("deadlock: no thread can make progress", 0); go = false }
  };
  turn = -2
}

/* ---------------------------------------------------------------- special registers */
val sreg_set : (int, string, int) -> unit
function sreg_set(sp, n, v) = smap_set(sp, n, v)

val dimRegs : (int, string, int, dim3) -> unit
function dimRegs(sp, p, i, d) = {
  sreg_set(sp, concat_str(p, ".x"), fmod_int(i, d.x));
  sreg_set(sp, concat_str(p, ".y"), fmod_int(tdiv_int(i, d.x), d.y));
  sreg_set(sp, concat_str(p, ".z"), tdiv_int(i, d.x * d.y))
}
val sizeRegs : (int, string, dim3) -> unit
function sizeRegs(sp, p, d) = {
  sreg_set(sp, concat_str(p, ".x"), d.x); sreg_set(sp, concat_str(p, ".y"), d.y); sreg_set(sp, concat_str(p, ".z"), d.z)
}

val clRank : (int, dim3, dim3) -> int
function clRank(c, g, cd) =
  fmod_int(fmod_int(c, g.x), cd.x) + fmod_int(fmod_int(tdiv_int(c, g.x), g.y), cd.y) * cd.x + fmod_int(tdiv_int(c, g.x * g.y), cd.z) * cd.x * cd.y

/* sregs(C, T, GD, BD, CD) */
val mk_sregs : (int, int, int, dim3, dim3, dim3) -> unit
function mk_sregs(sp, c, t, g, b, cd) = {
  dimRegs(sp, "%tid", t, b);
  dimRegs(sp, "%ctaid", c, g);
  sizeRegs(sp, "%ntid", b);
  sizeRegs(sp, "%nctaid", g);
  let l = fmod_int(t, 32);
  sreg_set(sp, "%lanemask_eq", pow2(l));
  sreg_set(sp, "%lanemask_lt", pow2(l) - 1);
  sreg_set(sp, "%lanemask_le", lor_int(pow2(l) - 1, pow2(l)));
  sreg_set(sp, "%lanemask_gt", land_int(4294967295, lnot_int(lor_int(pow2(l) - 1, pow2(l)))));
  sreg_set(sp, "%lanemask_ge", land_int(4294967295, lnot_int(pow2(l) - 1)));
  sreg_set(sp, "%laneid", l);
  sreg_set(sp, "%warpid", tdiv_int(t, 32));
  sreg_set(sp, "%nwarpid", tdiv_int(vol(b) + 31, 32));
  sreg_set(sp, "%clusterid.x", tdiv_int(fmod_int(c, g.x), cd.x));
  sreg_set(sp, "%clusterid.y", tdiv_int(fmod_int(tdiv_int(c, g.x), g.y), cd.y));
  sreg_set(sp, "%clusterid.z", tdiv_int(tdiv_int(c, g.x * g.y), cd.z));
  sreg_set(sp, "%nclusterid.x", tdiv_int(g.x, cd.x));
  sreg_set(sp, "%nclusterid.y", tdiv_int(g.y, cd.y));
  sreg_set(sp, "%nclusterid.z", tdiv_int(g.z, cd.z));
  sreg_set(sp, "%cluster_ctaid.x", fmod_int(fmod_int(c, g.x), cd.x));
  sreg_set(sp, "%cluster_ctaid.y", fmod_int(fmod_int(tdiv_int(c, g.x), g.y), cd.y));
  sreg_set(sp, "%cluster_ctaid.z", fmod_int(tdiv_int(c, g.x * g.y), cd.z));
  sreg_set(sp, "%cluster_nctaid.x", cd.x);
  sreg_set(sp, "%cluster_nctaid.y", cd.y);
  sreg_set(sp, "%cluster_nctaid.z", cd.z);
  sreg_set(sp, "%cluster_ctarank", clRank(c, g, cd));
  sreg_set(sp, "%cluster_nctarank", cd.x * cd.y * cd.z);
  sreg_set(sp, "%is_explicit_cluster", 0);
  sreg_set(sp, "%smid", 0); sreg_set(sp, "%nsmid", 1); sreg_set(sp, "%gridid", 0);
  sreg_set(sp, "%clock", 0); sreg_set(sp, "%clock64", 0); sreg_set(sp, "%warpsize", 32);
  sreg_set(sp, "%total_smem_size", 233472); sreg_set(sp, "%aggr_smem_size", 233472);
  sreg_set(sp, "%dynamic_smem_size", 0); sreg_set(sp, "%current_graph_exec", 0);
  sreg_set(sp, "%reserved_smem_offset_begin", 0); sreg_set(sp, "%reserved_smem_offset_end", 1024);
  sreg_set(sp, "%reserved_smem_offset_cap", 1024);
  sreg_set(sp, "%reserved_smem_offset_0", 0); sreg_set(sp, "%reserved_smem_offset_1", 0);
  foreach (i from 0 to 7) {
    sreg_set(sp, concat_str("%pm", dec_str(i)), 0);
    sreg_set(sp, concat_str("%pm", concat_str(dec_str(i), "_64")), 0)
  };
  foreach (i from 0 to 31) sreg_set(sp, concat_str("%envreg", dec_str(i)), 0)
}

/* #mkCtas(C, N) / #mkThreads(C, T, N) */
function mkCtas(c0, n) = {
  var c : int = c0;
  while c < n do {
    let cs = cchk(c);
    c_alive[cs] = true;
    c_live[cs] = vol(blockDim);
    c_smem[cs] = new_space();
    c_bars[cs] = new_space();
    c_colls[cs] = new_space();
    c_wlive[cs] = new_space();
    c_misc[cs] = new_space();
    if c >= ncta_slots then ncta_slots = c + 1;
    foreach (t from 0 to (vol(blockDim) - 1)) {
      let s = tchk(nthreads);
      nthreads = nthreads + 1;
      let w = dec_str(tdiv_int(t, 32));
      smap_set(c_wlive[cs], w, lor_int(smap_get(c_wlive[cs], w), pow2(fmod_int(t, 32))));
      t_alive[s] = true;
      t_cta[s] = c;
      t_tid[s] = t;
      t_status[s] = 0;
      t_fn[s] = kfid;
      t_pc[s] = 0;
      let sr = new_space();
      mk_sregs(sr, c, t, gridDim, blockDim, clusterDim);
      t_sregs[s] = sr;
      let rg = new_space();
      smap_copy(rg, sr);
      t_regs[s] = rg;
      let env = new_space();
      smap_copy(env, SP_KENV);
      t_env[s] = env;
      let pm = new_space();
      mem_copy(pm, 0, SP_KPARAMS, 0, kptop);
      t_pmem[s] = pm;
      t_ptop[s] = kptop;
      t_lmem[s] = new_space();
      t_ltop[s] = 0;
      t_cseq[s] = new_space();
      t_misc[s] = new_space();
      t_cont[s] = K_none();
      t_stack[s] = [||]
    };
    nextcta = c + 1;
    resident = resident + 1;
    c = c + 1
  }
}

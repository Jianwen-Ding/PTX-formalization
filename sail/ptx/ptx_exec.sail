/* =============================================================================
 * Thread-level execution of the core instructions: PTX-EXEC-ALU, -MEM, -CTRL, -SYNC, -COLL
 * (semantics/ptx-exec-*.k).  Each exec_* function corresponds to the `exec` / `#gx` rules of one K
 * module and returns true when one of its rules applies.
 * ============================================================================= */

val movv : (list(sop), list(sop), ty) -> unit
val ldv : (list(sop), loc, ty) -> unit
val stv : (list(sop), loc, ty) -> unit
val icall : (int, list(sop), list(sop)) -> unit

val ty1 : opcode -> ty
function ty1(o) = ty_nth(o.tys, 0)
val ntys : opcode -> int
function ntys(o) = ty_count(o.tys)

val unsupported : opcode -> unit
function unsupported(o) = fault(concat_str("unsupported instruction ", o.base), 0)

/* ---------------------------------------------------------------- ALU (PTX-EXEC-ALU) */
val dual_regs : xop -> option((sop, sop))
function dual_regs(d) = match d {
  XS(SPIPE(k1, a, k2, b)) => if k1 == k2 then Some((if k1 then SR(a) else SN(a), if k2 then SR(b) else SN(b))) else None(),
  _ => None()
}

/* wrP(R, D, V, BO, X): setp destination p or p|q */
val wrP : (xop, int, string, int) -> unit
function wrP(d, v, bo, x) = match dual_regs(d) {
  Some((p, q)) => { wr_sop(p, bop(v, bo, x)); wr_sop(q, bop(1 - v, bo, x)) },
  None() => wr(d, bop(v, bo, x))
}

/* pack / unpack of mov with vector operands */
val pack : (list(sop), int, int) -> int
function pack(l, w, sh) = match l { [||] => 0, v :: rest => lor_int(shl_int(rd_sop(v, mkty("b", w)), sh), pack(rest, w, sh + w)) }

val unpack : (list(sop), int, int) -> unit
function unpack(l, w, v) = match l { [||] => (), d :: rest => { wr_sop(d, trunc(v, w)); unpack(rest, w, asr_int(v, w)) } }

val clToGeneric : (int, int) -> int
function clToGeneric(a, own) =
  winBase("cluster") + shl_int(if asr_int(a, 24) == 0 then own else asr_int(a, 24) - 1, 24) + land_int(a, 16777215)
val clFromGeneric : (int, int) -> int
function clFromGeneric(a, own) = {
  if genWin(a) == "cluster" then lor_int(shl_int(asr_int(a - winBase("cluster"), 24) + 1, 24), land_int(a - winBase("cluster"), 16777215))
  else if genWin(a) == "shared" then lor_int(shl_int(own + 1, 24), a - winBase("shared"))
  else 0
}

/* size and alignment of the formal parameter N of a parameter list */
val pdFind : (list(pdecl), string) -> option((int, int))
function pdFind(l, n) = match l { [||] => None(), p :: rest => if p.pname == n then Some((p.psize, p.palign)) else pdFind(rest, n) }

/* mov of the address of a device-function parameter: the parameter is copied onto the stack frame and its
 * address is in the .local state space (PTX ISA 5.1 note 3, mov).  The name is bound to the copy, so that
 * ld.param / st.param see the same bytes; ret copies it back to the argument ("@cpy:" entries of the frame). */
val paramToLocal : string -> unit
function paramToLocal(n) = {
  let t = tchk(cur);
  let f = fchk(t_fn[t]);
  let s = smap_get(t_env[t], n);
  let (sz, al) : (int, int) = match pdFind(f_params[f], n) {
    Some(x) => x,
    None() => match pdFind(f_rets[f], n) { Some(x) => x, None() => (0, 1) }
  };
  let a = alignUp(t_ltop[t], max_int(al, 1));
  t_ltop[t] = a + sz;
  mem_copy(t_lmem[t], a, t_pmem[t], sym_addr(s), sz);
  smap_set(t_env[t], n, mksym("local", a));
  smap_set(t_env[t], concat_str("@cpy:", n), s);
  smap_set(t_env[t], concat_str("@cpysz:", n), sz)
}

/* is N a .param formal of the executing device function (not yet copied to the stack) */
val isFuncParam : string -> bool
function isFuncParam(n) = {
  let t = tchk(cur);
  let e = t_env[t];
  f_kind[fchk(t_fn[t])] == "func" & not_bool(isRegName(n)) & smap_has(e, n) & not_bool(smap_has(e, concat_str("@cpy:", n)))
  & sym_space(smap_get(e, n)) == "param"
  & (is_some(pdFind(f_params[fchk(t_fn[t])], n)) | is_some(pdFind(f_rets[fchk(t_fn[t])], n)))
}

val exec_alu : (opcode, list(xop)) -> bool
function exec_alu(o, ops) = {
  let m = o.mods;
  let n = xop_count(ops);
  match o.base {
    "setp" => {
      if ntys(o) != 1 then false
      else {
        let t = ty1(o);
        /* .ftz: subnormal float inputs compare as sign-preserving zeros (ISA: .ftz applies only to .f32 comparisons) */
        let fz = has(m, "ftz") & isF(t) & fw(t) != 64;
        let x = if fz then flushB(rd(xop_nth(ops, 1), t), fw(t)) else rd(xop_nth(ops, 1), t);
        let y = if fz then flushB(rd(xop_nth(ops, 2), t), fw(t)) else rd(xop_nth(ops, 2), t);
        if n == 3 then { wrP(xop_nth(ops, 0), cmp(cmpOp(m), t, x, y), "none", 0); true }
        else if n == 4 then {
          wrP(xop_nth(ops, 0), cmp(cmpOp(m), t, x, y), boolOp(m), rdp(xop_nth(ops, 3)));
          true
        } else false
      }
    },
    "mov" => {
      if ntys(o) != 1 | n != 2 then false
      else {
        let t = ty1(o);
        let d = xop_nth(ops, 0);
        let a = xop_nth(ops, 1);
        match a {
          XS(SN(nm)) => if isFuncParam(nm) then paramToLocal(nm),
          XS(SNP(nm, _)) => if isFuncParam(nm) then paramToLocal(nm),
          _ => ()
        };
        match (d, a) {
          (XV(ds), XV(vs)) => { movv(ds, vs, t); true },
          (XV(ds), _) => { let v = rd(a, t); unpack(ds, tdiv_int(wid(t), sop_count(ds)), v); true },
          (_, XV(vs)) => { wr(d, pack(vs, tdiv_int(wid(t), sop_count(vs)), 0)); true },
          _ => { wr(d, rd(a, t)); true }
        }
      }
    },
    "cvt" => {
      if ntys(o) != 2 | n != 2 then false
      else {
        let dt = ty_nth(o.tys, 0);
        let st = ty_nth(o.tys, 1);
        let d = xop_nth(ops, 0);
        wr(d, extReg(cvt(m, dt, st, rd(xop_nth(ops, 1), st)), dt, d));
        true
      }
    },
    "cvta" => {
      if ntys(o) != 1 | n != 2 then false
      else {
        let t = ty1(o);
        let a = rd(xop_nth(ops, 1), t);
        if has(m, "shared::cluster") then {
          let own = rreg("%cluster_ctarank");
          wr(xop_nth(ops, 0), trunc(if has(m, "to") then clFromGeneric(a, own) else clToGeneric(a, own), wid(t)))
        } else
          wr(xop_nth(ops, 0), trunc(if has(m, "to") then a - winBase(spaceOf(m)) else a + winBase(spaceOf(m)), wid(t)));
        true
      }
    },
    _ => false
  }
}

/* elementwise; every source is read in the instruction's context (the registers before it), as K's
 * movv(R, Ds, Vs, T, C) reads through C */
val movw : (list(sop), list(int)) -> unit
function movw(ds, vs) = match (ds, vs) { (d :: dr, v :: vr) => { wr_sop(d, v); movw(dr, vr) }, _ => () }
function movv(ds, vs, t) = {
  var vals : list(int) = [||];
  var x = vs;
  var n : int = sop_count(ds);
  while is_cons(x) & n > 0 do { match x { v :: rest => { vals = pend_append(vals, [|rd_sop(v, t)|]); x = rest; n = n - 1 }, [||] => () } };
  movw(ds, vals)
}

/* the #gx ALU rules: BINOPS / UNOPS / TERNOPS on one type */
val gx_alu : (opcode, list(xop)) -> bool
function gx_alu(o, ops) = {
  let m = o.mods;
  let n = xop_count(ops);
  if ntys(o) != 1 then false
  else {
    let t = ty1(o);
    let op = o.base;
    if has(BINOPS, op) & not_bool(has(m, "cc")) & not_bool(isPacked(t)) & n == 3 then {
      wr(xop_nth(ops, 0), bin(op, m, t, rd(xop_nth(ops, 1), t), rd(xop_nth(ops, 2), src2Ty(op, t)))); true
    } else if has(UNOPS, op) & not_bool(isPacked(t)) & n == 2 then {
      wr(xop_nth(ops, 0), un(op, m, t, rd(xop_nth(ops, 1), t))); true
    } else if has(TERNOPS, op) & not_bool(has(m, "cc")) & not_bool(isPacked(t)) & n == 4 then {
      wr(xop_nth(ops, 0), tern(op, m, t, rd(xop_nth(ops, 1), t), rd(xop_nth(ops, 2), t), rd(xop_nth(ops, 3), src3Ty(op, m, t)))); true
    } else false
  }
}

/* ---------------------------------------------------------------- memory (PTX-EXEC-MEM) */
val exec_mem : (opcode, list(xop)) -> bool
function exec_mem(o, ops) = {
  let m = o.mods;
  let n = xop_count(ops);
  if ntys(o) != 1 then false
  else {
    let t = ty1(o);
    match o.base {
      "ld" => {
        if n != 2 | not_bool(is_mem1(xop_nth(ops, 1))) then false
        else match mem_addr(xop_nth(ops, 1)) {
          Some(a) => {
            let l = ea(a, spaceOf(m));
            match xop_nth(ops, 0) {
              XV(ds) => ldv(ds, l, t),
              d => { let v = memop_do(l, RdOp(bytes(t))); if not_bool(stuck) then ldwb(d, t, v) }
            };
            true
          },
          None() => false
        }
      },
      "st" => {
        if n != 2 | not_bool(is_mem1(xop_nth(ops, 0))) then false
        else match mem_addr(xop_nth(ops, 0)) {
          Some(a) => {
            let l = ea(a, spaceOf(m));
            match xop_nth(ops, 1) {
              XV(vs) => stv(vs, l, t),
              v => { let _ = memop_do(l, WrOp(bytes(t), rd(v, t))) in () }
            };
            true
          },
          None() => false
        }
      },
      "atom" => {
        if (n != 3 & n != 4) | not_bool(is_mem1(xop_nth(ops, 1))) then false
        else match mem_addr(xop_nth(ops, 1)) {
          Some(a) => {
            let c = if n == 4 then rd(xop_nth(ops, 3), t) else 0;
            let v = memop_do(ea(a, spaceOf(m)), AtOp(atomOp(m), t, rd(xop_nth(ops, 2), t), c));
            if not_bool(stuck) then ldwb(xop_nth(ops, 0), t, v);
            true
          },
          None() => false
        }
      },
      "red" => {
        if n != 2 | not_bool(is_mem1(xop_nth(ops, 0))) then false
        else match mem_addr(xop_nth(ops, 0)) {
          Some(a) => { let _ = memop_do(ea(a, spaceOf(m)), AtOp(atomOp(m), t, rd(xop_nth(ops, 1), t), 0)) in (); true },
          None() => false
        }
      },
      _ => false
    }
  }
}

function ldv(ds, l, t) = match l {
  BadLoc(s) => fault(s, 0),
  _ => match ds {
    [||] => (),
    d :: rest => { let v = memop_do(l, RdOp(bytes(t))); if not_bool(stuck) then { ldwb(XS(d), t, v); ldv(rest, loc_add(l, bytes(t)), t) } }
  }
}

function stv(vs, l, t) = match l {
  BadLoc(s) => fault(s, 0),
  _ => match vs {
    [||] => (),
    v :: rest => { let _ = memop_do(l, WrOp(bytes(t), rd_sop(v, t))); if not_bool(stuck) then stv(rest, loc_add(l, bytes(t)), t) }
  }
}

/* declVar: frame-allocated .local / .param variables */
val declVar : (string, string, int, int) -> unit
function declVar(sp, nm, sz, al) = {
  let t = tchk(cur);
  if sp == "local" then {
    let a = alignUp(t_ltop[t], al);
    t_ltop[t] = a + sz;
    smap_set(t_env[t], nm, mksym("local", a))
  } else {
    let a = alignUp(t_ptop[t], al);
    t_ptop[t] = a + sz;
    smap_set(t_env[t], nm, mksym("param", a))
  }
}

/* ---------------------------------------------------------------- calls (PTX-EXEC-CTRL) */
val lookup_func : string -> option(int)
function lookup_func(f) = if smap_has(SP_FUNCS, f) then Some(smap_get(SP_FUNCS, f) - 1) else None()

val enter_func : (int, int, list((string, string))) -> unit
/* #enter(F, E): new frame; the callee's registers are the special registers plus the by-value (.reg) arguments */
function enter_func(f, env, outs) = {
  let t = tchk(cur);
  let regs = new_space();
  smap_copy(regs, t_sregs[t]);
  /* regIns: "@in:P" |-> sym("regin", V) */
  let n = smap_size(env);
  foreach (i from 0 to (n - 1)) {
    let k = smap_key(env, i);
    if string_startswith(k, "@in:") then smap_set(regs, sdrop(k, 4), sym_addr(smap_get(env, k)))
  };
  t_stack[t] = (t_fn[t], t_pc[t], t_regs[t], t_env[t], t_ltop[t], t_lsp[t], outs) :: t_stack[t];
  t_fn[t] = f;
  t_pc[t] = 0;
  t_regs[t] = regs;
  t_env[t] = env
}

val do_call : (string, list(sop), list(sop)) -> unit
function do_call(fname, rs, args) = match lookup_func(fname) {
  None() => fault(concat_str("call of undefined function ", fname), 0),
  Some(f) => {
    let t = tchk(cur);
    let env = new_space();
    /* #bindArgs */
    var ps = f_params[fchk(f)];
    var aa = args;
    var ok : bool = true;
    while ok & is_cons(ps) do {
      match (ps, aa) {
        (p :: pr, a :: ar) => {
          if p.palign < 0 then smap_set(env, concat_str("@in:", p.pname), mksym("regin", rd_sop(a, p.pty)))
          else match a {
            SN(nm) => { let s = lookupSym(nm); smap_set(env, p.pname, if s < 0 then mksym("nosym", 0) else s) },
            _ => {
              let addr = alignUp(t_ptop[t], p.palign);
              mem_wrn(t_pmem[t], addr, p.psize, rd_sop(a, p.pty));
              t_ptop[t] = addr + p.psize;
              smap_set(env, p.pname, mksym("param", addr))
            }
          };
          ps = pr; aa = ar
        },
        _ => { ok = false; fault(concat_str("call to ", concat_str(fname, ": too few arguments")), 0) }
      }
    };
    if ok & is_cons(aa) then { ok = false; fault(concat_str("call to ", concat_str(fname, ": too many arguments")), 0) };
    /* #bindRets */
    var outs : list((string, string)) = [||];
    var rps = f_rets[fchk(f)];
    var rr = rs;
    var rok : bool = true;
    while ok & rok & is_cons(rps) do {
      match (rps, rr) {
        (p :: pr, d :: dr) => {
          if p.palign < 0 then outs = (p.pname, sop_name(d)) :: outs
          else match d {
            SN(nm) => { let s = lookupSym(nm); smap_set(env, p.pname, if s < 0 then mksym("nosym", 0) else s) },
            _ => { rok = false }
          };
          rps = pr; rr = dr
        },
        _ => { rok = false }
      }
    };
    /* K has no #bindRets rule for a count mismatch or a .param return that is not a name: the call is stuck there */
    if ok & (not_bool(rok) | is_cons(rr)) then { ok = false; fault(concat_str("call to ", concat_str(fname, ": return values do not match the declaration")), 0) };
    if ok then enter_func(f, env, outs)
  }
}

/* #ret: pop the frame (regOuts copies the by-value results); with no frame, the thread exits */
val do_exit : unit -> unit

val do_ret : unit -> unit
function do_ret() = {
  let t = tchk(cur);
  match t_stack[t] {
    [||] => do_exit(),
    (f, pc, regs, env, lt, sp, outs) :: rest => {
      let cr = t_regs[t];
      /* parameters whose address was taken (paramToLocal): copy them back to the arguments */
      let e = t_env[t];
      foreach (i from 0 to (smap_size(e) - 1)) {
        let k = smap_key(e, i);
        if string_startswith(k, "@cpy:") then {
          let n = sdrop(k, 5);
          let s = smap_get(e, k);
          mem_copy(t_pmem[t], sym_addr(s), t_lmem[t], sym_addr(smap_get(e, n)), smap_get(e, concat_str("@cpysz:", n)))
        }
      };
      var l = outs;
      while is_cons(l) do {
        match l { (p, d) :: lr => { smap_set(regs, d, smap_get(cr, p)); l = lr }, [||] => () }
      };
      smap_clear(cr);
      smap_clear(t_env[t]);
      t_stack[t] = rest;
      t_fn[t] = f;
      t_pc[t] = pc;
      t_regs[t] = regs;
      t_env[t] = env;
      t_ltop[t] = lt;
      t_lsp[t] = sp
    }
  }
}

val label_pc : (string) -> option(int)
function label_pc(l) = if smap_has(cur_lab(), l) then Some(smap_get(cur_lab(), l)) else None()

val find_tgts : (int, string) -> list(string)
function find_tgts(f, l) = {
  var x = tgts;
  var r : list(string) = [||];
  while is_cons(x) do { match x { (g, k, ts) :: rest => { if g == f & k == l then r = ts; x = rest }, [||] => () } };
  r
}

val str_nth : (list(string), int) -> option(string)
function str_nth(l, i) = match l { [||] => None(), h :: t => if i == 0 then Some(h) else str_nth(t, i - 1) }

val paren_items : xop -> option(list(sop))
function paren_items(o) = match o { XP(l) => Some(l), _ => None() }

val exec_ctrl : (opcode, list(xop)) -> bool
function exec_ctrl(o, ops) = {
  let n = xop_count(ops);
  match o.base {
    "bra" => match ops {
      [|XS(SN(l))|] => match label_pc(l) { Some(p) => { t_pc[tchk(cur)] = p; true }, None() => false },
      _ => false
    },
    "ret" => if n == 0 then { do_ret(); true } else false,
    "exit" => if n == 0 then { do_exit(); true } else false,
    "trap" => { fault("trap", 0); true },
    "call" => match ops {
      [|XP(rs), XS(SN(f)), XP(xs)|] => { do_call(f, rs, xs); true },
      [|XP(rs), XS(SN(f))|] => { do_call(f, rs, [||]); true },
      [|XS(SN(f)), XP(xs)|] => { do_call(f, [||], xs); true },
      [|XS(SN(f))|] => { do_call(f, [||], [||]); true },
      [|XP(rs), XS(SR(fp)), XP(xs), XS(SN(_))|] => { icall(rd_sop(SR(fp), mkty("u", 64)), rs, xs); true },
      [|XP(rs), XS(SR(fp)), XS(SN(_))|] => { icall(rd_sop(SR(fp), mkty("u", 64)), rs, [||]); true },
      [|XS(SR(fp)), XP(xs), XS(SN(_))|] => { icall(rd_sop(SR(fp), mkty("u", 64)), [||], xs); true },
      [|XS(SR(fp)), XS(SN(_))|] => { icall(rd_sop(SR(fp), mkty("u", 64)), [||], [||]); true },
      _ => false
    },
    "brx" => match ops {
      [|i, XS(SN(tb))|] => {
        let idx = rdu32(i);
        let ts = find_tgts(cur_fn(), concat_str("@branchtargets:", tb));
        match str_nth(ts, idx) {
          Some(l) => match label_pc(l) { Some(p) => t_pc[tchk(cur)] = p, None() => fault("brx.idx: index outside the target list", idx) },
          None() => fault("brx.idx: index outside the target list", idx)
        };
        true
      },
      _ => false
    },
    _ => false
  }
}

function icall(a, rs, xs) = {
  let k = dec_str(a);
  if smap_has(SP_FADDRS, k) then do_call(f_name[fchk(smap_get(SP_FADDRS, k) - 1)], rs, xs)
  else fault("indirect call through an invalid function pointer", a)
}

/* fences and hints: no-ops under sequential consistency */
val gx_ctrl : (opcode, list(xop)) -> bool
function gx_ctrl(o, ops) = has([|"membar", "fence", "nanosleep", "brkpt", "pmevent", "prefetch", "prefetchu"|], o.base)

/* ---------------------------------------------------------------- barriers (PTX-EXEC-SYNC) */
val isBar : (string, list(string)) -> bool
function isBar(op, m) = (op == "bar" | op == "barrier") & not_bool(has(m, "warp")) & not_bool(has(m, "red")) & not_bool(has(m, "cluster"))
val isBarCluster : (string, list(string)) -> bool
function isBarCluster(op, m) = op == "barrier" & has(m, "cluster")
val isBarRed : (string, list(string)) -> bool
function isBarRed(op, m) = (op == "bar" | op == "barrier") & has(m, "red") & not_bool(has(m, "warp"))
val isBarWarp : (string, list(string)) -> bool
function isBarWarp(op, m) = op == "bar" & has(m, "warp")

val bkey : (string, int) -> string
function bkey(k, b) = concat_str(k, dec_str(b))
val bar_get : (string, int, int) -> int
function bar_get(k, b, d) = { let sp = c_bars[cchk(cur_cta())]; if smap_has(sp, bkey(k, b)) then smap_get(sp, bkey(k, b)) else d }
val bar_set : (string, int, int) -> unit
function bar_set(k, b, v) = smap_set(c_bars[cchk(cur_cta())], bkey(k, b), v)

/* #barCheck(B): complete the barrier when the expected number of threads have arrived */
val barCheck : (int, int) -> unit
function barCheck(c, b) = {
  let sp = c_bars[cchk(c)];
  if smap_has(sp, bkey("cnt:", b)) then {
    let cnt = smap_get(sp, bkey("cnt:", b));
    let x = if smap_has(sp, bkey("exp:", b)) then smap_get(sp, bkey("exp:", b)) else -1;
    let need = if x < 0 then c_live[cchk(c)] else x;
    if cnt >= need then {
      smap_del(sp, bkey("cnt:", b));
      let v = if smap_has(sp, bkey("val:", b)) then smap_get(sp, bkey("val:", b)) else 0;
      smap_del(sp, bkey("val:", b));
      smap_set(sp, bkey("res:", b), v);
      smap_set(sp, bkey("gen:", b), (if smap_has(sp, bkey("gen:", b)) then smap_get(sp, bkey("gen:", b)) else 0) + 1)
    }
  }
}

register yielded : bool = false

/* #bar(B, X, W) */
val do_bar : (int, int, bool) -> unit
function do_bar(b, x, w) = {
  let t = tchk(cur);
  let g = bar_get("gen:", b, 0);
  if w then { t_status[t] = 1; t_wbar[t] = b; t_wgen[t] = g };
  bar_set("cnt:", b, bar_get("cnt:", b, 0) + 1);
  bar_set("exp:", b, x);
  barCheck(cur_cta(), b);
  if w then yielded = true
}

val redKind : list(string) -> string
function redKind(m) = pick(m, [|"popc", "and", "or"|], "popc")
val redStep : (string, int, int) -> int
function redStep(k, a, v) =
  if a < 0 then v
  else match k { "popc" => a + v, "and" => land_int(a, v), "or" => lor_int(a, v), _ => v }

/* #barRed(B, X, K, V, D) */
val do_barred : (int, int, string, int, xop) -> unit
function do_barred(b, x, k, v, d) = {
  let t = tchk(cur);
  let g = bar_get("gen:", b, 0);
  t_status[t] = 1; t_wbar[t] = b; t_wgen[t] = g;
  bar_set("cnt:", b, bar_get("cnt:", b, 0) + 1);
  bar_set("exp:", b, x);
  bar_set("val:", b, redStep(k, bar_get("val:", b, -1), v));
  t_cont[t] = K_barwaitred(b, g, d);
  barCheck(cur_cta(), b);
  yielded = true
}

val gx_sync : (opcode, list(xop)) -> bool
function gx_sync(o, ops) = {
  let m = o.mods;
  if isBar(o.base, m) then match ops {
    [|a|] => { do_bar(rdu32(a), -1, not_bool(has(m, "arrive"))); true },
    [|a, nn|] => { do_bar(rdu32(a), rdu32(nn), not_bool(has(m, "arrive"))); true },
    _ => false
  } else if isBarRed(o.base, m) then match ops {
    [|d, a, pc|] => { do_barred(rdu32(a), -1, redKind(m), b2i(rdp(pc) != 0), d); true },
    [|d, a, nn, pc|] => { do_barred(rdu32(a), rdu32(nn), redKind(m), b2i(rdp(pc) != 0), d); true },
    _ => false
  } else false
}

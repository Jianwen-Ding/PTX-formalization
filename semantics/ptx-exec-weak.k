// =============================================================================
// PTX-EXEC-WEAK: an operational weak-memory mode (`.memmodel relaxed;`)
//
// By default the machine is sequentially consistent: every memory instruction takes effect when
// it executes.  In relaxed mode, stores to .global / .shared go through a per-thread store buffer
// (cell <sb>) and reach memory through spontaneous flush steps, so that `krun --search` explores
// the reorderings that the PTX memory model permits for weak and relaxed accesses:
//
//   * a thread reads its own buffered stores (forwarding); other threads see only memory;
//   * stores to different addresses may reach memory out of order (PSO-like);
//   * st.release is a barrier entry: it leaves the buffer after every earlier store and before
//     every later one;
//   * atom / red / bar / fence / membar / mbarrier / cp.* / vector accesses / exit first drain the
//     buffer (a sound strengthening: it only removes behaviours);
//   * memory is multi-copy atomic: a flushed store becomes visible to all threads at once.
//
// This is an under-approximation of the PTX model (every behaviour produced is allowed by PTX, and
// the fences and release/acquire patterns of the standard litmus tests behave as the model requires)
// but not a complete one: load-buffering style reorderings (a load overtaking a later store) and the
// scope hierarchy are not modelled.  In relaxed mode a plain `krun` follows one schedule; use `--search`.
// =============================================================================
requires "ptx-exec-mem.k"

module PTX-EXEC-WEAK
  imports PTX-EXEC-MEM

  syntax SbE ::= sbe(String, Int, Int, Int, Bool)            // space, address, bytes, value, release barrier?
  
  // ---------------------------------------------------------------- stores and loads (at the #mem level)
  // global / shared stores enter the buffer; loads forward from it; atomics wait for it to drain.
  rule <tk> #rm(loc(S, A), wrOp(N, V)) => #res(0) ... </tk>
       <sb> SB => SB ListItem(sbe(S, A, N, V, false)) </sb>
  rule <tk> #rm(loc(S, A), wrRelOp(N, V)) => #res(0) ... </tk>
       <sb> SB => SB ListItem(sbe(S, A, N, V, true)) </sb>
  rule <tk> #rm(L, rdOp(N)) => #mem1(L, rdOp(N)) ... </tk>
       <sb> SB </sb> requires notBool sbOverlap(SB, L, N)
  rule <tk> #rm(L, rdOp(N)) => #res(sbFwd(SB, L, N)) ... </tk>
       <sb> SB </sb> requires sbOverlap(SB, L, N) andBool sbExact(SB, L, N)
  // a partially overlapping buffered store: wait until the flush rules have drained the buffer
  rule <tk> #rm(L, rdOp(N)) => #sbWait ~> #rm(L, rdOp(N)) ... </tk>
       <sb> SB </sb> requires sbOverlap(SB, L, N) andBool notBool sbExact(SB, L, N)
  rule <tk> #rm(L, atOp(OP, T, B, X)) => #mem1(L, atOp(OP, T, B, X)) ... </tk> <sb> .List </sb>
  rule <tk> #rm(L, atOp(OP, T, B, X)) => #sbWait ~> #rm(L, atOp(OP, T, B, X)) ... </tk> <sb> ListItem(_) _ </sb>
  rule <tk> #rm(L, insOp(O, W, V)) => #mem1(L, insOp(O, W, V)) ... </tk> <sb> .List </sb>
  rule <tk> #rm(L, insOp(O, W, V)) => #sbWait ~> #rm(L, insOp(O, W, V)) ... </tk> <sb> ListItem(_) _ </sb>

  // a partially overlapping buffered store: the load is stuck until the flush rules have drained it
  rule <tk> #mem(L, rdOp(N)) => #sbWait ~> #mem(L, rdOp(N)) ... </tk>
       <sb> SB </sb> <mm> "relaxed" </mm>
    requires sbOverlap(SB, L, N) andBool notBool sbExact(SB, L, N) [priority(40)]
  rule <tk> #mem(L, atOp(OP, T, B, X)) => #sbWait ~> #mem(L, atOp(OP, T, B, X)) ... </tk>
       <sb> ListItem(_) _ </sb> <mm> "relaxed" </mm> [priority(40)]
  rule <tk> #sbWait => .K ... </tk> <sb> .List </sb>
  // a partially overlapping buffered store: the load waits until the flush rules have drained it
  syntax Bool ::= sbOverlap(List, Loc, Int) [function, total]
  rule sbOverlap(ListItem(sbe(S, A, N, _, _)) Rest, loc(S2, A2), N2) =>
       (S ==String S2 andBool A <Int A2 +Int N2 andBool A2 <Int A +Int N) orBool sbOverlap(Rest, loc(S2, A2), N2)
  rule sbOverlap(_, _, _) => false [owise]
  // the youngest overlapping entry has exactly the address and size of the load
  syntax Bool ::= sbExact(List, Loc, Int) [function, total]
  rule sbExact(SB, L, N) => isRelExact(sbYoungest(SB, L, N), L, N)
  syntax Bool ::= isRelExact(SbE, Loc, Int) [function, total]
  rule isRelExact(sbe(S, A, N, _, _), loc(S, A), N) => true
  rule isRelExact(_, _, _) => false [owise]
  syntax SbE ::= sbYoungest(List, Loc, Int) [function, total]
  rule sbYoungest(SB ListItem(E:SbE), L, N) => E requires sbOverlap(ListItem(E), L, N)
  rule sbYoungest(SB ListItem(_), L, N) => sbYoungest(SB, L, N) [owise]
  rule sbYoungest(.List, _, _) => sbe("", 0 -Int 1, 0, 0, false)
  syntax Int ::= sbFwd(List, Loc, Int) [function, total]
  rule sbFwd(SB, L, N) => sbVal(sbYoungest(SB, L, N))
  syntax Int ::= sbVal(SbE) [function, total]
  rule sbVal(sbe(_, _, _, V, _)) => V


  // ---------------------------------------------------------------- load buffering
  // A weak or relaxed scalar load from .global / .shared is issued but performed later: the thread goes on with the
  // following instructions (which may include stores that reach memory first) and the load takes its value from
  // memory at a spontaneous step.  Instructions that use the destination register, memory-ordering instructions,
  // later accesses to an overlapping location and release stores wait for the pending loads.  This allows the
  // load-buffering outcomes of the PTX model (a load observing a later store of another thread that itself
  // observed a store of this thread).
  syntax KItem ::= #plPush(Operand, Ty, Loc, Bool) | #plDo(PendLoad)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops(D, [ A ], .Operands), C) => #plPush(D, T, ea(A, spaceOf(M), C), "relaxed" in M) ... </tk>
       <mm> "relaxed" </mm>
    requires notBool isVec(D) andBool notBool ("acquire" in M orBool "volatile" in M orBool "mmio" in M) [priority(40)]
  rule <tk> #plPush(D, T, loc(S, A), St) => .K ... </tk>
       <pl> PL => PL ListItem(pl(D, T, S, A, bytes(T), St)) </pl>
    requires S ==String "global" orBool S ==String "shared"
  rule <tk> #plPush(D, T, loc(S, A), _) => #mem(loc(S, A), rdOp(bytes(T))) ~> #ldwb(D, T) ... </tk>
    requires S =/=String "global" andBool S =/=String "shared"
  rule <tk> #plPush(D, T, badLoc(E), _) => #fault(E, 0) ... </tk>
  rule <tk> #plDo(pl(D, T, S, A, N, _)) => #rm(loc(S, A), rdOp(N)) ~> #ldwb(D, T) ... </tk>
  rule <tk> #plDrain => .K ... </tk> <pl> .List </pl>
  rule <tk> #plDrain => #plDo(E) ~> #plDrain ... </tk> <pl> ListItem(E) Post => Post </pl>
  // a spontaneous step: the turn is free and the thread idle
  rule <tk> .K => #plDo(E) </tk> <turn> -1 </turn> <pl> ListItem(E) Post => Post </pl>
  // loads to different locations, or weak loads, may also be performed out of order
  rule <tk> .K => #plDo(E2) </tk> <turn> -1 </turn> <pl> ListItem(E1) ListItem(E2) Post => ListItem(E1) Post </pl> requires plCanSwap(E1, E2)
  syntax Bool ::= plCanSwap(PendLoad, PendLoad) [function, total]
  rule plCanSwap(pl(_, _, S1, A1, N1, St1), pl(_, _, S2, A2, N2, St2)) =>
       notBool (S1 ==String S2 andBool A1 <Int A2 +Int N2 andBool A2 <Int A1 +Int N1) orBool (notBool St1 andBool notBool St2)
  rule <tk> .K </tk> <turn> U => -1 </turn> <ctaid> C </ctaid> <tid> T </tid> <status> running </status>
       <pl> ListItem(_) _ </pl> requires U ==Int uid(C, T)
  // later accesses to a location with a pending load wait for it; so does a release store
  syntax Bool ::= plOverlap(List, Loc) [function, total]
  rule plOverlap(ListItem(pl(_, _, S, A, N, _)) L, loc(S2, A2)) => (S ==String S2 andBool A <Int A2 +Int 8 andBool A2 <Int A +Int N) orBool plOverlap(L, loc(S2, A2))
  rule plOverlap(_, _) => false [owise]
  rule <tk> #mem(L, OP) => #plDrain ~> #mem(L, OP) ... </tk> <pl> PL </pl> <mm> "relaxed" </mm> requires plOverlap(PL, L) [priority(30)]
  rule <tk> #mem(L, wrRelOp(N, V)) => #plDrain ~> #mem(L, wrRelOp(N, V)) ... </tk> <pl> ListItem(_) _ </pl> <mm> "relaxed" </mm> [priority(30)]


  // ---------------------------------------------------------------- asynchronous operations as concurrent agents
  // In relaxed mode the data movement of the asynchronous instructions (cp.async, cp.async.bulk[.tensor], cp.reduce.async.bulk,
  // st.async, red.async, fabric.*) is not performed at issue: the instruction, with the register values of the issue, is queued
  // in the issuing thread's <ag> list and performed at a spontaneous step.  Kinds: 0 cp.async (waited by commit_group /
  // wait_group / wait_all), 1 completed through an mbarrier, 2 cp.async.bulk to global (waited by the bulk groups).  A thread
  // that exits performs what it still has queued.
  syntax AgE ::= ag(Int, Int, K)
  syntax KItem ::= #async(Int, K) | #agWait(Int, Int) | "#agDrain"
  syntax Int ::= agKind(String, Set) [function, total]
  rule agKind("cp", M) => #if "async" in M andBool notBool ("commit_group" in M orBool "wait_group" in M orBool "wait_all" in M orBool "prefetch" in M orBool ("mbarrier" in M andBool notBool "bulk" in M))
                           #then (#if "bulk_group" in M #then 2 #else (#if "bulk" in M #then 1 #else 0 #fi) #fi) #else 0 -Int 1 #fi
  rule agKind("st", M) => #if "async" in M #then 1 #else 0 -Int 1 #fi
  rule agKind("red", M) => #if "async" in M #then 1 #else 0 -Int 1 #fi
  rule agKind("fabric", M) => #if "submit" in M orBool "wait" in M #then 0 -Int 1 #else 1 #fi
  rule agKind(_, _) => 0 -Int 1 [owise]
  rule <tk> exec(opc(OP, M, Ts), Os, C) => #async(agKind(OP, M), exec(opc(OP, M SetItem("#now"), Ts), Os, C)) ... </tk>
       <mm> "relaxed" </mm>
    requires agKind(OP, M) >=Int 0 andBool notBool ("#now" in M) [priority(30)]
  syntax Int ::= agGid(Int, Int, Int) [function, total]
  rule agGid(0, N0, _) => N0
  rule agGid(2, _, N2) => N2
  rule agGid(_, _, _) => 0 -Int 1 [owise]
  rule <tk> #async(Kd, A) => .K ... </tk> <ag> L => L ListItem(ag(Kd, agGid(Kd, N0, N2), A)) </ag> <agn> N0 </agn> <abn> N2 </abn>
  rule <tk> .K => A </tk> <turn> -1 </turn> <ag> ListItem(ag(_, _, A)) Rest => Rest </ag>
  rule <tk> .K => A </tk> <turn> -1 </turn> <ag> ListItem(E1) ListItem(ag(_, _, A)) Rest => ListItem(E1) Rest </ag>
  rule <tk> .K </tk> <turn> U => -1 </turn> <ctaid> C </ctaid> <tid> T </tid> <status> running </status>
       <ag> ListItem(_) _ </ag> requires U ==Int uid(C, T)
  // commit / wait
  rule <tk> exec(opc("cp", M, _), ops(_), _) => .K ... </tk> <agn> G => G +Int 1 </agn> <mm> "relaxed" </mm>
    requires "async" in M andBool "commit_group" in M andBool notBool "bulk" in M [priority(30)]
  rule <tk> exec(opc("cp", M, _), ops(_), _) => .K ... </tk> <abn> G => G +Int 1 </abn> <mm> "relaxed" </mm>
    requires "async" in M andBool "commit_group" in M andBool "bulk" in M [priority(30)]
  rule <tk> exec(opc("cp", M, _), ops(N, .Operands), C) => #agWait(#if "bulk" in M #then 2 #else 0 #fi, #if "bulk" in M #then G2 #else G0 #fi -Int 1 -Int rd(N, ty("u", 32), C)) ... </tk>
       <agn> G0 </agn> <abn> G2 </abn> <mm> "relaxed" </mm>
    requires "async" in M andBool "wait_group" in M [priority(30)]
  rule <tk> exec(opc("cp", M, _), ops(.Operands), _) => #agWait(0, G0) ... </tk> <agn> G0 => G0 +Int 1 </agn> <mm> "relaxed" </mm>
    requires "async" in M andBool "wait_all" in M andBool notBool "bulk" in M [priority(30)]
  syntax Int ::= agIdx(List, Int, Int, Int) [function, total]        // first queued entry of the kind whose group is not newer than the limit
  rule agIdx(ListItem(ag(Kd, G, _)) _, Kd, Lim, I) => I requires G <=Int Lim
  rule agIdx(ListItem(_) L, Kd, Lim, I) => agIdx(L, Kd, Lim, I +Int 1) [owise]
  rule agIdx(.List, _, _, _) => 0 -Int 1
  syntax K ::= agAct(List, Int) [function, total]
  rule agAct(ListItem(ag(_, _, A)) _, 0) => A
  rule agAct(ListItem(_) L, I) => agAct(L, I -Int 1) requires I >Int 0
  rule agAct(_, _) => .K [owise]
  syntax List ::= agDel(List, Int) [function, total]
  rule agDel(ListItem(_) L, 0) => L
  rule agDel(ListItem(E) L, I) => ListItem(E) agDel(L, I -Int 1) requires I >Int 0
  rule agDel(L, _) => L [owise]
  rule <tk> #agWait(Kd, Lim) => agAct(L, agIdx(L, Kd, Lim, 0)) ~> #agWait(Kd, Lim) ... </tk>
       <ag> L => agDel(L, agIdx(L, Kd, Lim, 0)) </ag> requires agIdx(L, Kd, Lim, 0) >=Int 0
  rule <tk> #agWait(Kd, Lim) => .K ... </tk> <ag> L </ag> requires agIdx(L, Kd, Lim, 0) <Int 0
  rule <tk> #agDrain => A ~> #agDrain ... </tk> <ag> ListItem(ag(_, _, A)) Rest => Rest </ag>
  rule <tk> #agDrain => .K ... </tk> <ag> .List </ag>
  rule <tk> #exit => #agDrain ~> #exit ... </tk> <ag> ListItem(_) _ </ag> [priority(30)]

  // ---------------------------------------------------------------- flushing (spontaneous steps)
  // A thread whose computation is idle while the turn is free may retire a buffered store: the oldest one, or a
  // younger one that is allowed to overtake all older ones (different locations, no release entries involved).
  // The reorder window is three entries.  A thread draining before a fence/atomic/exit retires oldest-first.
  // a thread retires only once its buffer is empty
  rule <tk> #exit => #sbWait ~> #exit ... </tk> <sb> ListItem(_) _ </sb>
  // while stores are buffered the turn is given up before every instruction, so that flushes (and the
  // reordering of the buffered stores) can interleave with it, also right before the thread retires
  rule <tk> .K </tk> <turn> U => -1 </turn> <ctaid> C </ctaid> <tid> T </tid> <status> running </status>
       <sb> ListItem(_) _ </sb> requires U ==Int uid(C, T)
  syntax KItem ::= #flushE(SbE)
  rule <tk> #flushE(sbe(S, A, N, V, _)) => #mem1(loc(S, A), wrOp(N, V)) ~> #drop ... </tk>
  rule <tk> #sbWait => #flushE(E) ~> #sbWait ... </tk> <sb> ListItem(E) Post => Post </sb>
  rule <tk> .K => #flushE(E1) </tk> <turn> -1 </turn> <sb> ListItem(E1) Post => Post </sb>
  rule <tk> .K => #flushE(E2) </tk> <turn> -1 </turn> <sb> ListItem(E1) ListItem(E2) Post => ListItem(E1) Post </sb>
    requires canSwap(E1, E2)
  rule <tk> .K => #flushE(E3) </tk> <turn> -1 </turn> <sb> ListItem(E1) ListItem(E2) ListItem(E3) Post => ListItem(E1) ListItem(E2) Post </sb>
    requires canSwap(E1, E3) andBool canSwap(E2, E3)
  syntax Bool ::= sbIsBar(SbE) [function, total]
  rule sbIsBar(sbe(_, _, _, _, B)) => B
  syntax Bool ::= canSwap(SbE, SbE) [function, total]
  rule canSwap(E1, E2) => notBool sbIsBar(E1) andBool notBool sbIsBar(E2) andBool notBool sbOverlap(ListItem(E1), sbLoc(E2), sbN(E2))
  syntax Loc ::= sbLoc(SbE) [function, total]
  rule sbLoc(sbe(S, A, _, _, _)) => loc(S, A)
  syntax Int ::= sbN(SbE) [function, total]
  rule sbN(sbe(_, _, N, _, _)) => N
endmodule

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
  syntax KItem ::= #plPush(Operand, Ty, Loc) | #plDo(PendLoad)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops(D, [ A ], .Operands), C) => #plPush(D, T, ea(A, spaceOf(M), C)) ... </tk>
       <mm> "relaxed" </mm>
    requires notBool isVec(D) andBool notBool ("acquire" in M orBool "volatile" in M orBool "mmio" in M) [priority(40)]
  rule <tk> #plPush(D, T, loc(S, A)) => .K ... </tk>
       <pl> PL => PL ListItem(pl(D, T, S, A, bytes(T))) </pl>
    requires S ==String "global" orBool S ==String "shared"
  rule <tk> #plPush(D, T, loc(S, A)) => #mem(loc(S, A), rdOp(bytes(T))) ~> #ldwb(D, T) ... </tk>
    requires S =/=String "global" andBool S =/=String "shared"
  rule <tk> #plPush(D, T, badLoc(E)) => #fault(E, 0) ... </tk>
  rule <tk> #plDo(pl(D, T, S, A, N)) => #mem(loc(S, A), rdOp(N)) ~> #ldwb(D, T) ... </tk>
  rule <tk> #plDrain => .K ... </tk> <pl> .List </pl>
  rule <tk> #plDrain => #plDo(E) ~> #plDrain ... </tk> <pl> ListItem(E) Post => Post </pl>
  // a spontaneous step: the turn is free and the thread idle
  rule <tk> .K => #plDo(E) </tk> <turn> -1 </turn> <pl> ListItem(E) Post => Post </pl>
  rule <tk> .K </tk> <turn> U => -1 </turn> <ctaid> C </ctaid> <tid> T </tid> <status> running </status>
       <pl> ListItem(_) _ </pl> requires U ==Int uid(C, T)
  // later accesses to a location with a pending load wait for it; so does a release store
  syntax Bool ::= plOverlap(List, Loc) [function, total]
  rule plOverlap(ListItem(pl(_, _, S, A, N)) L, loc(S2, A2)) => (S ==String S2 andBool A <Int A2 +Int 8 andBool A2 <Int A +Int N) orBool plOverlap(L, loc(S2, A2))
  rule plOverlap(_, _) => false [owise]
  rule <tk> #rm(L, OP) => #plDrain ~> #rm(L, OP) ... </tk> <pl> PL </pl> requires plOverlap(PL, L) [priority(40)]
  rule <tk> #rm(L, wrRelOp(N, V)) => #plDrain ~> #rm(L, wrRelOp(N, V)) ... </tk> <pl> ListItem(_) _ </pl> [priority(40)]

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

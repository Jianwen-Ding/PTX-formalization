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
  syntax KItem ::= #sbPush(Loc, Int, Int, Bool) | "#sbWait" | #sbLd(Loc, Int, Operand, Ty)

  // ---------------------------------------------------------------- stores
  rule <tk> exec(opc("st", M, ListItem(T)), ops([ A ], V, .Operands), C) =>
            #sbPush(ea(A, spaceOf(M), C), bytes(T), rd(V, T, C), "release" in M) ... </tk>
       <mm> "relaxed" </mm>
    requires notBool isVec(V) andBool notBool vecMod(M) [priority(40)]
  rule <tk> #sbPush(loc(S, A), N, V, B) => .K ... </tk>
       <sb> SB => SB ListItem(sbe(S, A, N, V, B)) </sb>
    requires S ==String "global" orBool S ==String "shared"
  rule <tk> #sbPush(L, N, V, _) => #mem(L, wrOp(N, V)) ~> #drop ... </tk> [owise]
  syntax Bool ::= vecMod(Set) [function, total]
  rule vecMod(M) => "v2" in M orBool "v4" in M orBool "v8" in M

  // ---------------------------------------------------------------- loads (with store-to-load forwarding)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops(D, [ A ], .Operands), C) => #sbLd(ea(A, spaceOf(M), C), bytes(T), D, T) ... </tk>
       <mm> "relaxed" </mm>
    requires notBool isVec(D) andBool notBool vecMod(M) [priority(40)]
  rule <tk> #sbLd(L, N, D, T) => #mem(L, rdOp(N)) ~> #ldwb(D, T) ... </tk>
       <sb> SB </sb>
    requires notBool sbOverlap(SB, L, N)
  rule <tk> #sbLd(L, N, D, T) => #res(sbFwd(SB, L, N)) ~> #ldwb(D, T) ... </tk>
       <sb> SB </sb>
    requires sbOverlap(SB, L, N) andBool sbExact(SB, L, N)
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

  // ---------------------------------------------------------------- draining
  syntax Bool ::= strongOp(String, Set) [function, total]
  rule strongOp(OP, M) => OP in (SetItem("atom") SetItem("red") SetItem("bar") SetItem("barrier") SetItem("fence") SetItem("membar")
                                 SetItem("mbarrier") SetItem("cp") SetItem("wgmma") SetItem("tcgen05") SetItem("exit"))
                          orBool ((OP ==String "st" orBool OP ==String "ld") andBool vecMod(M))
  rule <tk> exec(opc(OP, M, T), Ops, C) => #sbWait ~> exec(opc(OP, M, T), Ops, C) ... </tk>
       <sb> ListItem(_) _ </sb> <mm> "relaxed" </mm>
    requires strongOp(OP, M) [priority(30)]
  rule <tk> #exit => #sbWait ~> #exit ... </tk>
       <sb> ListItem(_) _ </sb> <mm> "relaxed" </mm> [priority(30)]
  rule <tk> #sbWait => .K ... </tk> <sb> .List </sb>

  // ---------------------------------------------------------------- flushing (spontaneous steps)
  syntax Bool ::= flushable(List, String, Int, Int, Bool) [function, total]
  rule flushable(Pre, _, _, _, true) => Pre ==K .List
  rule flushable(Pre, S, A, N, false) => noBarrier(Pre) andBool notBool sbOverlap(Pre, loc(S, A), N)
  syntax Bool ::= noBarrier(List) [function, total]
  rule noBarrier(ListItem(sbe(_, _, _, _, B)) Rest) => notBool B andBool noBarrier(Rest)
  rule noBarrier(_) => true [owise]
  rule <ctas> ... <cta> ... <threads> ... <thread> ... <sb> Pre ListItem(sbe("global", A, N, V, B)) Post => Pre Post </sb> ... </thread> ... </threads> ... </cta> ... </ctas>
       <gmem> M => writeBytes(M, A, N, V) </gmem>
    requires flushable(Pre, "global", A, N, B)
  rule <cta> ... <smem> M => writeBytes(M, A, N, V) </smem> <threads> ... <thread> ... <sb> Pre ListItem(sbe("shared", A, N, V, B)) Post => Pre Post </sb> ... </thread> ... </threads> ... </cta>
    requires flushable(Pre, "shared", A, N, B)
endmodule

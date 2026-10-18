// =============================================================================
// PTX-EXEC-WARP: warp-synchronous instructions, built on the rendezvous in
// ptx-exec-coll.k.
//
//   shfl.sync.{up,down,bfly,idx}.b32 d[|p], a, b, c, membermask
//   vote.sync.{all,any,uni}.pred p, {!}q, membermask      vote.sync.ballot.b32 d, {!}q, membermask
//   match.sync.{any,all}.{b32,b64} d[|p], a, membermask
//   elect.sync d|p, membermask
//   redux.sync.{add,min,max,and,or,xor}.{u32,s32,b32} d, a, membermask
//   redux.sync.{min,max}[.abs][.NaN].f32 d, a, membermask
//   activemask.b32 d
//
// Only the `.sync` forms exist (the pre-sm_70 forms without a membermask are
// deprecated).  Lanes that name inconsistent masks never complete (see
// examples/collective-mismatch.ptx).  activemask assumes a converged warp: it
// returns the lanes of the warp that have not exited.
// =============================================================================
requires "ptx-exec-coll.k"

module PTX-EXEC-WARP
  imports PTX-EXEC-COLL

  rule isCollective("shfl") => true
  rule isCollective("vote") => true
  rule isCollective("match") => true
  rule isCollective("elect") => true
  rule isCollective("redux") => true

  // bar.warp.sync membermask : all named lanes rendezvous (no data)
  rule <tk> exec(opc("bar", M, _), ops(MM, .Operands), C) => #coll("bar.warp", rd(MM, ty("u", 32), C), .List, .List, .List) ... </tk>
    requires "warp" in M
  rule collCompute("bar.warp", _, Arr) => sameAll(keys_list(Arr), ListItem(0))      // no data; a non-empty result marks completion

  // d|p -> [d, p];  d -> [d]
  syntax List ::= dests(Operand) [function, total]
  rule dests(D:Reg | P:Reg) => ListItem(D) ListItem(P)
  rule dests(D:Name | P:Name) => ListItem(D) ListItem(P)
  rule dests(D) => ListItem(D) [owise]

  syntax Int ::= at(List, Int) [function, total]
  rule at(L, I) => {L[I]}:>Int requires I >=Int 0 andBool I <Int size(L)
  rule at(_, _) => 0 [owise]
  syntax List ::= listAt(Map, Int) [function, total]          // contribution of a lane
  rule listAt(Arr, L) => {Arr[L]}:>List requires L in_keys(Arr)
  rule listAt(_, _) => .List [owise]

  // ---------------------------------------------------------------- shfl.sync
  rule <tk> exec(opc("shfl", M, _), ops(DP, A, B, Cn, MM, .Operands), C) =>
            #coll("shfl." +String pick(M, ListItem("up") ListItem("down") ListItem("bfly") ListItem("idx"), "idx"),
                  rd(MM, ty("u", 32), C), .List,
                  ListItem(rd(A, ty("b", 32), C)) ListItem(rd(B, ty("u", 32), C)) ListItem(rd(Cn, ty("u", 32), C)),
                  dests(DP)) ... </tk>

  // result list [value, predicate] of lane L for mode `Mode`
  syntax List ::= shflLane(String, Int, Map) [function, total]
  rule shflLane(Mode, L, Arr) => shflPick(L, at(listAt(Arr, L), 0),
                                          shflSrc(Mode, L, at(listAt(Arr, L), 1) &Int 31, at(listAt(Arr, L), 2) &Int 31,
                                                  (at(listAt(Arr, L), 2) >>Int 8) &Int 31), Arr)
  // the source lane j and whether it is valid (pval)
  syntax List ::= shflSrc(String, Int, Int, Int, Int) [function, total]       // mode, lane, bval, cval, segmask -> j, pval
  rule shflSrc(Mode, L, BV, CV, SM) => shflJ(Mode, L, BV, SM, (L &Int SM) |Int (CV &Int (~Int SM) &Int 31), L &Int SM)
  syntax List ::= shflJ(String, Int, Int, Int, Int, Int) [function, total]    // mode, lane, bval, segmask, maxLane, minLane
  rule shflJ("shfl.up", L, BV, _, MAXL, _) => ListItem(L -Int BV) ListItem(b2i(L -Int BV >=Int MAXL))
  rule shflJ("shfl.down", L, BV, _, MAXL, _) => ListItem(L +Int BV) ListItem(b2i(L +Int BV <=Int MAXL))
  rule shflJ("shfl.bfly", L, BV, _, MAXL, _) => ListItem(L xorInt BV) ListItem(b2i((L xorInt BV) <=Int MAXL))
  rule shflJ(_, _, BV, SM, MAXL, MINL) => ListItem(MINL |Int (BV &Int (~Int SM) &Int 31)) ListItem(b2i((MINL |Int (BV &Int (~Int SM) &Int 31)) <=Int MAXL)) [owise]
  syntax List ::= shflPick(Int, Int, List, Map) [function, total]             // lane, own value, [j, pval], arrivals
  rule shflPick(L, OWN, ListItem(J:Int) ListItem(1), Arr) =>
       ListItem(#if J in_keys(Arr) #then at(listAt(Arr, J), 0) #else OWN #fi) ListItem(1)
  rule shflPick(_, OWN, _, _) => ListItem(OWN) ListItem(0) [owise]

  syntax Map ::= shflAll(String, List, Map) [function, total]
  rule shflAll(Mode, ListItem(L:Int) Ls, Arr) => (L |-> shflLane(Mode, L, Arr)) shflAll(Mode, Ls, Arr)
  rule shflAll(_, _, _) => .Map [owise]
  rule collCompute("shfl.up", _, Arr)   => shflAll("shfl.up", keys_list(Arr), Arr)
  rule collCompute("shfl.down", _, Arr) => shflAll("shfl.down", keys_list(Arr), Arr)
  rule collCompute("shfl.bfly", _, Arr) => shflAll("shfl.bfly", keys_list(Arr), Arr)
  rule collCompute("shfl.idx", _, Arr)  => shflAll("shfl.idx", keys_list(Arr), Arr)

  // ---------------------------------------------------------------- vote.sync
  rule <tk> exec(opc("vote", M, _), ops(D, Q, MM, .Operands), C) =>
            #coll("vote." +String pick(M, ListItem("all") ListItem("any") ListItem("uni") ListItem("ballot"), "all"),
                  rd(MM, ty("u", 32), C), .List, ListItem(rd(Q, ty("p", 1), C)), ListItem(D)) ... </tk>
  syntax List ::= preds(Map, List) [function, total]          // predicate values in lane order
  rule preds(Arr, ListItem(L:Int) Ls) => ListItem(at(listAt(Arr, L), 0)) preds(Arr, Ls)
  rule preds(_, _) => .List [owise]
  syntax Bool ::= allTrue(List) [function, total]
  rule allTrue(ListItem(V:Int) Vs) => V =/=Int 0 andBool allTrue(Vs)
  rule allTrue(_) => true [owise]
  syntax Bool ::= anyTrue(List) [function, total]
  rule anyTrue(ListItem(V:Int) Vs) => V =/=Int 0 orBool anyTrue(Vs)
  rule anyTrue(_) => false [owise]
  syntax Int ::= ballotBit(Int, Int) [function, total]
  rule ballotBit(V, L) => #if V =/=Int 0 #then laneBit(L) #else 0 #fi
  syntax Int ::= ballot(Map, List) [function, total]
  rule ballot(Arr, ListItem(L:Int) Ls) => ballotBit(at(listAt(Arr, L), 0), L) |Int ballot(Arr, Ls)
  rule ballot(_, _) => 0 [owise]
  syntax Map ::= sameAll(List, List) [function, total]        // lanes, value list -> every lane maps to the value list
  rule sameAll(ListItem(L:Int) Ls, V) => (L |-> V) sameAll(Ls, V)
  rule sameAll(_, _) => .Map [owise]
  rule collCompute("vote.all", _, Arr)    => sameAll(keys_list(Arr), ListItem(b2i(allTrue(preds(Arr, keys_list(Arr))))))
  rule collCompute("vote.any", _, Arr)    => sameAll(keys_list(Arr), ListItem(b2i(anyTrue(preds(Arr, keys_list(Arr))))))
  rule collCompute("vote.uni", _, Arr)    => sameAll(keys_list(Arr), ListItem(b2i(allTrue(preds(Arr, keys_list(Arr))) orBool notBool anyTrue(preds(Arr, keys_list(Arr))))))
  rule collCompute("vote.ballot", _, Arr) => sameAll(keys_list(Arr), ListItem(ballot(Arr, keys_list(Arr))))

  // ---------------------------------------------------------------- match.sync
  rule <tk> exec(opc("match", M, ListItem(T)), ops(DP, A, MM, .Operands), C) =>
            #coll("match." +String pick(M, ListItem("any") ListItem("all"), "any"),
                  rd(MM, ty("u", 32), C), ListItem(rd(MM, ty("u", 32), C)), ListItem(rd(A, T, C)), dests(DP)) ... </tk>
  syntax Int ::= sameValueMask(Map, List, Int) [function, total]      // lanes with value V
  rule sameValueMask(Arr, ListItem(L:Int) Ls, V) => ballotBit(b2i(at(listAt(Arr, L), 0) ==Int V), L) |Int sameValueMask(Arr, Ls, V)
  rule sameValueMask(_, _, _) => 0 [owise]
  syntax Map ::= matchAny(List, Map) [function, total]
  rule matchAny(ListItem(L:Int) Ls, Arr) =>
       (L |-> (ListItem(sameValueMask(Arr, keys_list(Arr), at(listAt(Arr, L), 0))) ListItem(1))) matchAny(Ls, Arr)
  rule matchAny(_, _) => .Map [owise]
  rule collCompute("match.any", _, Arr) => matchAny(keys_list(Arr), Arr)
  rule collCompute("match.all", D, Arr) =>
       sameAll(keys_list(Arr), #if sameValueMask(Arr, keys_list(Arr), at(listAt(Arr, minLane(keys_list(Arr), 32)), 0)) ==Int at(D, 0)
                               #then ListItem(at(D, 0)) ListItem(1) #else ListItem(0) ListItem(0) #fi)

  // ---------------------------------------------------------------- elect.sync
  rule <tk> exec(opc("elect", _, _), ops(DP, MM, .Operands), C) =>
            #coll("elect", rd(MM, ty("u", 32), C), .List, .List, dests(DP)) ... </tk>
  syntax Map ::= electAll(List, Int) [function, total]
  rule electAll(ListItem(L:Int) Ls, Leader) => (L |-> (ListItem(Leader) ListItem(b2i(L ==Int Leader)))) electAll(Ls, Leader)
  rule electAll(_, _) => .Map [owise]
  rule collCompute("elect", _, Arr) => electAll(keys_list(Arr), minLane(keys_list(Arr), 32))

  // ---------------------------------------------------------------- redux.sync
  rule <tk> exec(opc("redux", M, ListItem(T)), ops(D, A, MM, .Operands), C) =>
            #coll("redux." +String pick(M, ListItem("add") ListItem("min") ListItem("max") ListItem("and") ListItem("or") ListItem("xor"), "add"),
                  rd(MM, ty("u", 32), C), ListItem(T) ListItem(M), ListItem(rd(A, T, C)), ListItem(D)) ... </tk>
  syntax Int ::= reduxFold(String, Ty, Set, List) [function, total]   // op, type, modifiers, values
  rule reduxFold(Op, T, M, ListItem(V:Int)) => V
  rule reduxFold(Op, T, M, ListItem(V:Int) Vs) => reduxStep(Op, T, M, V, reduxFold(Op, T, M, Vs)) requires size(Vs) >Int 0
  rule reduxFold(_, _, _, _) => 0 [owise]
  syntax Int ::= reduxStep(String, Ty, Set, Int, Int) [function, total]
  rule reduxStep("redux.add", T, _, A, B) => trunc(A +Int B, 32)
  rule reduxStep("redux.and", _, _, A, B) => A &Int B
  rule reduxStep("redux.or",  _, _, A, B) => A |Int B
  rule reduxStep("redux.xor", _, _, A, B) => A xorInt B
  rule reduxStep("redux.min", T, M, A, B) => fpMinMax(M, 32, A, B, true)  requires isF(T)
  rule reduxStep("redux.max", T, M, A, B) => fpMinMax(M, 32, A, B, false) requires isF(T)
  rule reduxStep("redux.min", T, _, A, B) => #if ival(A, T) <=Int ival(B, T) #then A #else B #fi requires notBool isF(T)
  rule reduxStep("redux.max", T, _, A, B) => #if ival(A, T) >=Int ival(B, T) #then A #else B #fi requires notBool isF(T)
  rule reduxStep(_, _, _, A, _) => A [owise]
  rule collCompute(Op, D, Arr) =>
       sameAll(keys_list(Arr), ListItem(reduxFold(Op, {D[0]}:>Ty, {D[1]}:>Set, preds(Arr, keys_list(Arr)))))
    requires Op in (SetItem("redux.add") SetItem("redux.min") SetItem("redux.max") SetItem("redux.and") SetItem("redux.or") SetItem("redux.xor"))

  // ---------------------------------------------------------------- activemask
  // `.convergence its;`: independent thread scheduling.  The lanes of a warp are not assumed to be converged: activemask
  // returns the lanes that have executed this same instruction (same function and pc, same visit count) up to now,
  // the executing lane included.  Over all schedules (krun --search) every prefix of the arrival order is observable.
  syntax Int ::= arrMask(Set) [function, total]
  rule arrMask(SetItem(L:Int) S) => (1 <<Int L) |Int arrMask(S)
  rule arrMask(.Set) => 0
  syntax String ::= actSite(String, Int) [function, total]
  rule actSite(F, PC) => "act@" +String F +String ":" +String Int2String(PC)
  syntax Map ::= itsArr(Map, Int, String, Int, Map) [function, total]      // arrivals so far, plus this lane
  rule itsArr(Cs, T, S, _, CS) => arrivals(Cs, mkKey(T, S, 0, CS))[T %Int 32 <- 0]
  rule <tk> exec(opc("activemask", _, _), ops(D, .Operands), _) => .K ... </tk>
       <its> true </its> <tid> T </tid> <fn> F </fn> <pc> PC </pc>
       <regs> R => wr(R, D, arrMask(keys(itsArr(Cs, T, actSite(F, PC), 0, CS)))) </regs>
       <cseq> CS => CS[seqKey(actSite(F, PC), 0) <- {CS[seqKey(actSite(F, PC), 0)] orDefault 0}:>Int +Int 1] </cseq>
       <colls> Cs => Cs[mkKey(T, actSite(F, PC), 0, CS) <- coll("act", .List, 0, itsArr(Cs, T, actSite(F, PC), 0, CS), .Map, 0)] </colls>
    [priority(40)]
  rule <tk> exec(opc("activemask", _, _), ops(D, .Operands), _) => .K ... </tk>
       <tid> T </tid>
       <wlive> WL </wlive>
       <regs> R => wr(R, D, {WL[T /Int 32] orDefault 0}:>Int) </regs>
endmodule

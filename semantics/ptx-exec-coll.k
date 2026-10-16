// =============================================================================
// PTX-EXEC-COLL: rendezvous infrastructure for `.sync` warp collectives
// (shfl.sync, vote.sync, match.sync, redux.sync, ...).
//
// Under independent thread scheduling, a collective with membermask M
// completes once every lane in M has executed it.  The i-th collective of a
// given (kind, mask) executed by each lane belongs to the same instance, so
// instances are keyed by (warp, kind, mask, i).
//
// To implement a collective instruction in another module:
//   1. `rule isCollective("opname") => true`   (makes it a visible step)
//   2. rewrite its `exec(...)` to
//        #coll(Kind, Mask, Data, Contribution, Dests)
//      where Contribution is this lane's input (a List of Ints), Data is
//      instruction-wide immutable information (List) and Dests the list of
//      destination registers;
//   3. add `rule collCompute(Kind, Data, Arrivals) => Results`, mapping
//      lane |-> contribution to lane |-> result (a List with one Int per
//      destination).
//
// A lane waits (holding no turn) until the instance is complete; if lanes use
// inconsistent masks, the instances never complete and the threads are stuck
// at #collWait in the final configuration -- the same deadlock real hardware
// exhibits.
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-COLL
  imports PTX-EXEC-CORE

  syntax KItem ::= #coll(String, Int, List, List, List)
                 | #collWait(CollKey, List)

  syntax Map ::= collCompute(String, List, Map) [function]
  // default: every lane gets back its own contribution
  rule collCompute(_, _, Arr) => Arr [owise]

  // collectives whose kind starts with "wg" rendezvous the 128 threads of a warpgroup
  syntax Int ::= collW(String) [function, total]
  syntax String ::= ckKind(CollKey) [function, total]
  rule collW(K) => 128 requires findString(K, "wg", 0) ==Int 0
  rule collW(_) => 32 [owise]
  rule ckKind(ck(_, K, _, _)) => K

  syntax Coll ::= collArrive(Map, CollKey, String, List, Int, Int, List) [function, total]
  rule collArrive(Cs, Key, K, D, M, L, Cn) =>
       collFinish(K, D, M, arrivals(Cs, Key)[L <- Cn])
  syntax Map ::= arrivals(Map, CollKey) [function, total]
  rule arrivals(Cs, Key) => collArr({Cs[Key]}:>Coll) requires Key in_keys(Cs)
  rule arrivals(_, _) => .Map [owise]
  syntax Map ::= collArr(Coll) [function, total]
  rule collArr(coll(_, _, _, A, _, _)) => A
  syntax Coll ::= collFinish(String, List, Int, Map) [function, total]
  rule collFinish(K, D, M, A) => coll(K, D, M, A, collCompute(K, D, A), 0)
    requires size(A) ==Int popc(M)
  rule collFinish(K, D, M, A) => coll(K, D, M, A, .Map, 0) [owise]

  // arrival: executed atomically with respect to the other lanes
  syntax String  ::= seqKey(String, Int) [function, total]
  syntax CollKey ::= mkKey(Int, String, Int, Map) [function, total]
  rule seqKey(K, M) => K +String ":" +String Int2String(M)
  rule mkKey(T, K, M, CS) => ck(T /Int collW(K), K, M, {CS[seqKey(K, M)] orDefault 0}:>Int)
  rule <tk> #coll(K, M, D, Cn, Ds) => #yield ~> #collWait(mkKey(T, K, M, CS), Ds) ... </tk>
       <tid> T </tid>
       <cseq> CS => CS[seqKey(K, M) <- {CS[seqKey(K, M)] orDefault 0}:>Int +Int 1] </cseq>
       <colls> Cs => Cs[mkKey(T, K, M, CS) <- collArrive(Cs, mkKey(T, K, M, CS), K, D, M, T %Int collW(K), Cn)] </colls>
    requires ((M >>Int (T %Int collW(K))) &Int 1) ==Int 1
  rule <tk> #coll(K, M, _, _, _) => #fault("lane not in membermask", M) ... </tk>
       <tid> T </tid>
    requires ((M >>Int (T %Int collW(K))) &Int 1) ==Int 0

  // the instance is complete once results exist; each lane takes its own
  syntax Map ::= collWrite(Map, List, List) [function, total]
  rule collWrite(R, ListItem(D:Operand) Ds, ListItem(V:Int) Vs) => collWrite(wr(R, D, V), Ds, Vs)
  rule collWrite(R, _, _) => R [owise]

  syntax List ::= laneRes(Map, Int) [function, total]
  rule laneRes(Res, L) => {Res[L]}:>List requires L in_keys(Res)
  rule laneRes(_, _) => .List [owise]

  rule <tk> #collWait(Key, Ds) => .K ... </tk>
       <tid> T </tid>
       <turn> -1 </turn>
       <regs> R => collWrite(R, Ds, laneRes(Res, T %Int collW(ckKind(Key)))) </regs>
       <colls> ... Key |-> coll(_, _, M, _, Res, Cn => Cn +Int 1) ... </colls>
    requires laneRes(Res, T %Int collW(ckKind(Key))) =/=K .List andBool Cn +Int 1 <Int popc(M)
  rule <tk> #collWait(Key, Ds) => .K ... </tk>
       <tid> T </tid>
       <turn> -1 </turn>
       <regs> R => collWrite(R, Ds, laneRes(Res, T %Int collW(ckKind(Key)))) </regs>
       <colls> ... Key |-> coll(_, _, M, _, Res, Cn) => .Map ... </colls>
    requires laneRes(Res, T %Int collW(ckKind(Key))) =/=K .List andBool Cn +Int 1 ==Int popc(M)

  // ---------------------------------------------------------------- self-test
  // coll_test.sum.u32 d, a, mask   : d = sum of a over the lanes in mask
  // coll_test.first.u32 d, a, mask : d = a of the lowest lane in mask
  // (not PTX instructions; they exercise the rendezvous machinery)
  rule isCollective("coll_test") => true
  rule <tk> exec(opc("coll_test", M, ListItem(T)), ops(D, A, Mk, .Operands), C) =>
            #coll(pick(M, ListItem("sum") ListItem("first"), "sum"), rd(Mk, ty("u", 32), C),
                  .List, ListItem(rd(A, T, C)), ListItem(D)) ... </tk>
  rule collCompute("sum", _, Arr) => mapAll(keys_list(Arr), ListItem(sumList(values(Arr))))
  rule collCompute("first", _, Arr) => mapAll(keys_list(Arr), {Arr[minLane(keys_list(Arr), 32)]}:>List)
  syntax Map ::= mapAll(List, List) [function, total]
  rule mapAll(ListItem(K:Int) Ks, V) => (K |-> V) mapAll(Ks, V)
  rule mapAll(_, _) => .Map [owise]
  syntax Int ::= sumList(List) [function, total]
  rule sumList(ListItem(L:List) Vs) => sumList(L) +Int sumList(Vs)
  rule sumList(ListItem(V:Int) Vs) => V +Int sumList(Vs)
  rule sumList(_) => 0 [owise]
  syntax Int ::= minLane(List, Int) [function, total]
  rule minLane(ListItem(K:Int) Ks, A) => minLane(Ks, minInt(K, A))
  rule minLane(_, A) => A [owise]
endmodule

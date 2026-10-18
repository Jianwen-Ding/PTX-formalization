// =============================================================================
// PTX-EXEC-MBAR: mbarrier objects in shared memory
//
//   mbarrier.init / inval / arrive / arrive_drop / expect_tx / complete_tx /
//   test_wait / try_wait / pending_count        (both .layout::v0 and ::v1; the v1 payload
//                                                 report is always zero, as for v0)
//
// An object is the state mb(phase, pending arrivals, expected arrivals, tx-count, layout 0/1) stored in
// cell <mbars> under (CTA, shared offset); the 64-bit memory itself is opaque.  A phase
// completes when pending arrivals and tx-count are both zero; the object then advances to
// the next phase with pending := expected.  mbarrier.arrive returns the phase it arrived in
// as its `state`; test_wait/try_wait with state compare against that phase, with .parity
// against the current phase's parity.  try_wait blocks until the phase completes.
// Remote operations on shared::cluster addresses use the DSMEM addressing of ptx-exec-cluster.k.
// .multicast::cluster::32b (expect_tx / complete_tx / arrive{.expect_tx}) applies the operation to the mbarrier at the same CTA-relative offset in every CTA of the 32-bit ctaMask.
// =============================================================================
requires "ptx-exec-cluster.k"
requires "ptx-exec-warp.k"

module PTX-EXEC-MBAR
  imports PTX-EXEC-CLUSTER
  imports PTX-EXEC-WARP

  rule isCollective("mbarrier") => true      // visible steps: they touch shared state

  syntax Int ::= mbKey(Loc, Int, Dim3, Dim3) [function, total]
  rule mbKey(loc("shared", A), C, _, _) => C *Int 4294967296 +Int A
  rule mbKey(loc("cluster", A), C, GD, CD) =>
       #if A >>Int 24 ==Int 0 #then C #else clTarget(C, (A >>Int 24) -Int 1, GD, CD) #fi *Int 4294967296 +Int (A &Int 16777215)
  rule mbKey(_, _, _, _) => 0 -Int 1 [owise]

  syntax MbState ::= mbCheck(MbState) [function, total]
  rule mbCheck(mb(P, 0, E, 0, L)) => mb(P +Int 1, E, E, 0, L)
  rule mbCheck(S) => S [owise]
  syntax MbState ::= mbArrive(MbState, Int) [function, total]       // pending -= n
  rule mbArrive(mb(P, Pe, E, T, L), N) => mbCheck(mb(P, Pe -Int N, E, T, L))
  syntax MbState ::= mbTx(MbState, Int) [function, total]           // tx-count += n
  rule mbTx(mb(P, Pe, E, T, L), N) => mbCheck(mb(P, Pe, E, T +Int N, L))
  syntax MbState ::= mbDrop(MbState, Int) [function, total]         // expected -= n, pending -= n
  rule mbDrop(mb(P, Pe, E, T, L), N) => mbCheck(mb(P, Pe -Int N, E -Int N, T, L))
  syntax Int ::= mbState(MbState) [function, total]          // opaque arrive state: phase | pending-before << 32
  rule mbState(mb(P, Pe, _, _, _)) => P |Int (Pe <<Int 32)
  syntax Int ::= mbPhase(MbState) [function, total] | mbPend(MbState) [function, total]
  rule mbPhase(mb(P, _, _, _, _)) => P
  rule mbPend(mb(_, Pe, _, _, _)) => Pe
  syntax MbState ::= mbGet(Map, Int) [function, total]
  rule mbGet(MB, K) => {MB[K]}:>MbState requires K in_keys(MB)
  rule mbGet(_, _) => mb(0 -Int 1, 0, 1, 0, 0) [owise]                 // uninitialised: phase -1
  syntax Bool ::= mbOk(Map, Int, Int) [function, total]             // object exists, pending >= n
  rule mbOk(MB, K, N) => K in_keys(MB) andBool mbPend(mbGet(MB, K)) >=Int N

  rule <tk> exec(opc("mbarrier", M, _), ops([ A ], N, .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => MB[mbKey(ea(A, spaceOf(M), C), CT, GD, CD) <- mb(0, rd(N, ty("u", 32), C), rd(N, ty("u", 32), C), 0, #if "layout::v1" in M #then 1 #else 0 #fi)] </mbars>
    requires "init" in M
  rule <tk> exec(opc("mbarrier", M, _), ops([ A ], .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => MB[mbKey(ea(A, spaceOf(M), C), CT, GD, CD) <- undef] </mbars>
    requires "inval" in M

  // expect_tx / complete_tx
  rule <tk> exec(opc("mbarrier", M, _), ops([ A ], N, Rest:Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => MB[mbKey(ea(A, spaceOf(M), C), CT, GD, CD) <- mbTx(mbGet(MB, mbKey(ea(A, spaceOf(M), C), CT, GD, CD)),
                          #if "complete_tx" in M #then 0 -Int rd(N, ty("u", 32), C) #else rd(N, ty("u", 32), C) #fi)] </mbars>
    requires ("expect_tx" in M orBool "complete_tx" in M) andBool notBool "arrive" in M andBool notBool "arrive_drop" in M
             andBool mbKey(ea(A, spaceOf(M), C), CT, GD, CD) in_keys(MB) andBool notBool "multicast::cluster::32b" in M

  // arrive / arrive_drop [.expect_tx]  state|_ , [addr] {, count | txCount}
  rule <tk> exec(opc("mbarrier", M, _), ops(D, [ A ], Rest:Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => MB[mbKey(ea(A, spaceOf(M), C), CT, GD, CD) <-
            #if "arrive_drop" in M
            #then mbDrop(mbTx(mbGet(MB, mbKey(ea(A, spaceOf(M), C), CT, GD, CD)), mbTxArg(M, opsList(Rest), C)), mbCountArg(M, opsList(Rest), C))
            #else mbArrive(mbTx(mbGet(MB, mbKey(ea(A, spaceOf(M), C), CT, GD, CD)), mbTxArg(M, opsList(Rest), C)), mbCountArg(M, opsList(Rest), C)) #fi] </mbars>
       <regs> R => wr(R, D, mbState(mbGet(MB, mbKey(ea(A, spaceOf(M), C), CT, GD, CD)))) </regs>
    requires ("arrive" in M orBool "arrive_drop" in M) andBool mbKey(ea(A, spaceOf(M), C), CT, GD, CD) in_keys(MB) andBool notBool "multicast::cluster::32b" in M
  syntax Int ::= mbTxArg(Set, List, Ctx) [function, total] | mbCountArg(Set, List, Ctx) [function, total]
  rule mbTxArg(M, L, C) => rd({L[0]}:>Operand, ty("u", 32), C) requires "expect_tx" in M andBool size(L) >Int 0
  rule mbTxArg(_, _, _) => 0 [owise]
  rule mbCountArg(M, _, _) => 1 requires "expect_tx" in M
  rule mbCountArg(M, L, C) => rd({L[0]}:>Operand, ty("u", 32), C) requires notBool "expect_tx" in M andBool size(L) >Int 0
  rule mbCountArg(_, _, _) => 1 [owise]

  // .multicast::cluster::32b
  syntax List ::= mbMcKeys(Int, Int, Int, Int, Dim3, Dim3) [function, total]      // offset, mask, rank, own CTA, grid, cluster
  rule mbMcKeys(Off, Mk, R, CT, GD, CD) =>
       #if (Mk >>Int R) &Int 1 ==Int 1 #then ListItem(mbKey(loc("cluster", ((R +Int 1) <<Int 24) |Int Off), CT, GD, CD)) #else .List #fi mbMcKeys(Off, Mk, R +Int 1, CT, GD, CD)
    requires R <Int 32 andBool (Mk >>Int R) >Int 0
  rule mbMcKeys(_, _, _, _, _, _) => .List [owise]
  syntax Map ::= mbTxKeys(Map, List, Int) [function, total]                       // tx-count += n at every key that exists
  rule mbTxKeys(MB, ListItem(K:Int) Ks, N) => mbTxKeys(#if K in_keys(MB) #then MB[K <- mbTx(mbGet(MB, K), N)] #else MB #fi, Ks, N)
  rule mbTxKeys(MB, _, _) => MB [owise]
  syntax Map ::= mbArrKeys(Map, List, Int, Int) [function, total]                  // expect-tx then arrive (count)
  rule mbArrKeys(MB, ListItem(K:Int) Ks, Tx, N) => mbArrKeys(#if K in_keys(MB) #then MB[K <- mbArrive(mbTx(mbGet(MB, K), Tx), N)] #else MB #fi, Ks, Tx, N)
  rule mbArrKeys(MB, _, _, _) => MB [owise]
  syntax Int ::= mbOff(Loc) [function, total]
  rule mbOff(loc(_, A)) => A
  rule mbOff(_) => 0 [owise]
  syntax Int ::= mbLast(List, Ctx) [function, total]
  rule mbLast(L, C) => rd({L[size(L) -Int 1]}:>Operand, ty("u", 32), C) requires size(L) >Int 0
  rule mbLast(_, _) => 0 [owise]
  syntax List ::= mbDropLast(List) [function, total]
  rule mbDropLast(ListItem(X) ListItem(Y) L) => ListItem(X) mbDropLast(ListItem(Y) L)
  rule mbDropLast(_) => .List [owise]
  rule <tk> exec(opc("mbarrier", M, _), ops([ A ], N, Mk, .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => mbTxKeys(MB, mbMcKeys(mbOff(ea(A, spaceOf(M), C)) &Int 16777215, rd(Mk, ty("u", 32), C), 0, CT, GD, CD),
                              #if "complete_tx" in M #then 0 -Int rd(N, ty("u", 32), C) #else rd(N, ty("u", 32), C) #fi) </mbars>
    requires ("expect_tx" in M orBool "complete_tx" in M) andBool notBool "arrive" in M andBool "multicast::cluster::32b" in M
  rule <tk> exec(opc("mbarrier", M, _), ops(_, [ A ], Rest:Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => mbArrKeys(MB, mbMcKeys(mbOff(ea(A, spaceOf(M), C)) &Int 16777215, mbLast(opsList(Rest), C), 0, CT, GD, CD),
                               mbTxArg(M, mbDropLast(opsList(Rest)), C), mbCountArg(M, mbDropLast(opsList(Rest)), C)) </mbars>
    requires "arrive" in M andBool "multicast::cluster::32b" in M

  // test_wait / try_wait  waitComplete , [addr], state|parity {, timeHint}
  syntax Bool ::= mbDone(Map, Int, Set, Int) [function, total]       // objects, key, mods, state-or-parity operand value
  rule mbDone(MB, K, M, V) => (mbPhase(mbGet(MB, K)) %Int 2) =/=Int (V &Int 1) requires "parity" in M
  rule mbDone(MB, K, M, V) => mbPhase(mbGet(MB, K)) >Int (V &Int 4294967295) requires notBool "parity" in M
  rule <tk> exec(opc("mbarrier", M, _), ops(D, [ A ], S, _:Operands), C) =>
            #if "try_wait" in M #then #yield ~> #mbTry(mbKey(ea(A, spaceOf(M), C), CT, GD, CD), M, rd(S, ty("b", 64), C), dests(D))
            #else #mbTest(mbKey(ea(A, spaceOf(M), C), CT, GD, CD), M, rd(S, ty("b", 64), C), dests(D)) #fi ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
    requires "test_wait" in M orBool "try_wait" in M
  syntax KItem ::= #mbTest(Int, Set, Int, List) | #mbTry(Int, Set, Int, List)
  syntax Map ::= wrFirst(Map, List, Int) [function, total]
  rule wrFirst(R, ListItem(D:Operand) _, V) => wr(R, D, V)
  rule wrFirst(R, _, _) => R [owise]
  rule <tk> #mbTest(K, M, V, D) => .K ... </tk>
       <mbars> MB </mbars> <regs> R => wrFirst(R, D, #if mbDone(MB, K, M, V) #then 1 #else 0 #fi) </regs>
    requires K in_keys(MB)
  rule <tk> #mbTry(K, M, V, D) => .K ... </tk>
       <turn> -1 </turn>
       <mbars> MB </mbars> <regs> R => wrFirst(R, D, 1) </regs>
    requires K in_keys(MB) andBool mbDone(MB, K, M, V)

  // mbarrier.pending_count.b64 count, state   (state from an arrive: phase | pending-before << 32)
  rule <tk> exec(opc("mbarrier", M, _), ops(D, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, (rd(S, ty("b", 64), C) >>Int 32) &Int 1048575) </regs>
    requires "pending_count" in M
  // mbarrier.check_layout.layout::vN p, [addr]
  rule <tk> exec(opc("mbarrier", M, _), ops(D, [ A ], .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB </mbars> <regs> R => wr(R, D, #if layoutOf(mbGet(MB, mbKey(ea(A, spaceOf(M), C), CT, GD, CD))) ==Int (#if "layout::v1" in M #then 1 #else 0 #fi) #then 1 #else 0 #fi) </regs>
    requires "check_layout" in M
  syntax Int ::= layoutOf(MbState) [function, total]
  rule layoutOf(mb(_, _, _, _, L)) => L
endmodule

// =============================================================================
// PTX-EXEC-FABRIC: fabric.* operations on CFT logical endpoints (PTX ISA 9.3 / 9.4, sm_100)
//
// A logical endpoint is a numbered object whose resource is a range of bytes on one device (unicast) or on every member
// device (multicast).  The harness defines one with
//     .fabric ID .size N { BUF0, BUF1, ... } ;
// (one buffer per member device; the first buffer is the issuing device's own memory, which also holds the counters
// of counted completion).  All operations execute atomically when issued and always succeed, so the report predicate
// of mbarrier layout::v1 stays clear; out-of-range accesses fault (the ISA calls them undefined behaviour).  The
// completion count follows the ISA: bytes for try_get / try_pullred, 16-byte units for try_put / try_red / try_atom.
//   try_get      endpoint -> shared            try_put    shared -> endpoint (all members with .multimem, .cp_mask, counted)
//   try_red      shared -> endpoint, elementwise reduction                    try_atom   16-byte atomic with shared in/out
//   try_pullred  multicast endpoint -> shared, reduction across members (warp-collective: lane 0 performs the access)
//   *.async.tensor  the tensor-map copy of cp.async.bulk.tensor against the endpoint memory (override::global_address)
//   submit / wait   no effect (operations are already complete)
// =============================================================================
requires "ptx-exec-mm.k"
requires "ptx-exec-tma.k"

module PTX-EXEC-FABRIC
  imports PTX-EXEC-MM
  imports PTX-EXEC-TMA

  // device addresses of [leId, off .. off+n)
  syntax List ::= fbBases(Map, Int, Int, Int, Bool) [function, total]             // table, endpoint, offset, bytes, all members?
  rule fbBases(F, Le, Off, N, All) => fbSel(mmBases({F[Le]}:>MmEnt), Off, All) requires Le in_keys(F) andBool Off >=Int 0 andBool Off +Int N <=Int mmSize({F[Le]}:>MmEnt)
  rule fbBases(_, _, _, _, _) => .List [owise]
  syntax List ::= fbSel(List, Int, Bool) [function, total]
  rule fbSel(Bs, Off, true) => mmAddAll(Bs, Off)
  rule fbSel(ListItem(B:Int) _, Off, false) => ListItem(B +Int Off)
  rule fbSel(_, _, _) => .List [owise]
  syntax Loc ::= fbAdd(Loc, Int) [function, total]
  rule fbAdd(loc(S, A), K) => loc(S, A +Int K)
  rule fbAdd(L, _) => L [owise]
  syntax Int ::= fbHead(List) [function, total]
  rule fbHead(ListItem(A:Int) _) => A
  rule fbHead(_) => 0 -Int 1 [owise]
  syntax Int ::= fbMask(Set, List, Ctx) [function, total]
  rule fbMask(M, ListItem(O:Operand) _, C) => rd(O, ty("u", 32), C) requires "cp_mask" in M
  rule fbMask(_, _, _) => 0 -Int 1 [owise]

  // ---- try_get
  syntax KItem ::= #fbGet(List, Loc, Loc, Int) | #fbWr(Loc, Loc, Int, Int)
  rule <tk> exec(opc("fabric", M, _), ops([ D:Addr ], [ L:Addr , O:Operand ], Sz, [ B:Addr ], _:Operands), C) =>
            #fbGet(fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), rd(Sz, ty("u", 32), C), false), ea(D, "shared", C), ea(B, "shared", C), rd(Sz, ty("u", 32), C)) ... </tk>
       <fabtab> F </fabtab>
    requires "try_get" in M andBool notBool "tensor" in M
  rule <tk> #fbGet(ListItem(A:Int) _, D, B, N) => #amem(loc("global", A), rdOp(N)) ~> #fbWr(D, B, N, N) ... </tk>
  rule <tk> #fbGet(.List, _, _, _) => #fault("fabric access outside the logical endpoint", 0) ... </tk>
  rule <tk> #res(V) ~> #fbWr(D, B, N, Tx) => #amem(D, wrOp(N, V)) ~> #drop ~> #bulkTx(B, Tx) ... </tk>

  // ---- try_put (plain, masked, counted)
  syntax KItem ::= #fbPut(List, Loc, Loc, Int, Int, Int) | #fbPutW(List, Loc, Int, Int, Int)
  rule <tk> exec(opc("fabric", M, _), ops([ L:Addr , O:Operand ], [ S:Addr ], Sz, [ B:Addr ], Rest:Operands), C) =>
            #fbPut(fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), rd(Sz, ty("u", 32), C), "multimem" in M), ea(S, "shared", C), ea(B, "shared", C),
                   rd(Sz, ty("u", 32), C), fbMask(M, opsList(Rest), C), 0 -Int 1) ... </tk>
       <fabtab> F </fabtab>
    requires "try_put" in M andBool notBool "tensor" in M
  rule <tk> exec(opc("fabric", M, _), ops([ L:Addr , O:Addr , Cn:Operand ], [ S:Addr ], Sz, [ B:Addr ], _:Operands), C) =>
            #fbPut(fbBases(F, hAddr(L, C), hAddr(O, C), rd(Sz, ty("u", 32), C), "multimem" in M), ea(S, "shared", C), ea(B, "shared", C),
                   rd(Sz, ty("u", 32), C), 0 -Int 1, fbHead(fbBases(F, hAddr(L, C), rd(Cn, ty("u", 64), C), 8, false))) ... </tk>
       <fabtab> F </fabtab>
    requires "try_put" in M andBool notBool "tensor" in M
  rule <tk> #fbPut(.List, _, _, _, _, _) => #fault("fabric access outside the logical endpoint", 0) ... </tk>
  rule <tk> #fbPut(Ds, S, B, N, Mask, Cn) => #amem(S, rdOp(N)) ~> #fbPutW(Ds, B, N, Mask, Cn) ... </tk> requires Ds =/=K .List
  rule <tk> #res(V) ~> #fbPutW(Ds, B, N, Mask, Cn) => #fbWrDev(Ds, N, V, Mask) ~> #fbCnt(Cn, N) ~> #bulkTx(B, N /Int 16) ... </tk>
  syntax KItem ::= #fbWrDev(List, Int, Int, Int) | #fbMerge(Int, List, Int, Int, Int) | #fbCnt(Int, Int)
  rule <tk> #fbWrDev(ListItem(A:Int) As, N, V, Mask) => #amem(loc("global", A), wrOp(N, V)) ~> #drop ~> #fbWrDev(As, N, V, Mask) ... </tk> requires Mask <Int 0
  rule <tk> #fbWrDev(ListItem(A:Int) As, N, V, Mask) => #amem(loc("global", A), rdOp(N)) ~> #fbMerge(A, As, N, V, Mask) ... </tk> requires Mask >=Int 0
  rule <tk> #res(Old) ~> #fbMerge(A, As, N, V, Mask) =>
            #amem(loc("global", A), wrOp(N, (V &Int expandMask(Mask, N /Int 16, 0)) |Int (Old &Int (~Int expandMask(Mask, N /Int 16, 0))))) ~> #drop ~> #fbWrDev(As, N, V, Mask) ... </tk>
  rule <tk> #fbWrDev(.List, _, _, _) => .K ... </tk>
  rule <tk> #fbCnt(Cn, N) => #amem(loc("global", Cn), atOp("add", ty("u", 64), N, 0)) ~> #drop ... </tk> requires Cn >=Int 0
  rule <tk> #fbCnt(Cn, _) => .K ... </tk> requires Cn <Int 0

  // ---- try_red: elementwise reduction into every target location
  syntax KItem ::= #fbRedW(List, Loc, Int, Ty, String, Int)
  rule <tk> exec(opc("fabric", M, ListItem(T)), ops([ L:Addr , O:Operand ], [ S:Addr ], Sz, [ B:Addr ], _:Operands), C) =>
            #fbRed(fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), rd(Sz, ty("u", 32), C), "multimem" in M), ea(S, "shared", C), ea(B, "shared", C),
                   rd(Sz, ty("u", 32), C), T, atomOp(M), 0 -Int 1) ... </tk>
       <fabtab> F </fabtab>
    requires "try_red" in M andBool notBool "tensor" in M
  rule <tk> exec(opc("fabric", M, ListItem(T)), ops([ L:Addr , O:Addr , Cn:Operand ], [ S:Addr ], Sz, [ B:Addr ], _:Operands), C) =>
            #fbRed(fbBases(F, hAddr(L, C), hAddr(O, C), rd(Sz, ty("u", 32), C), "multimem" in M), ea(S, "shared", C), ea(B, "shared", C),
                   rd(Sz, ty("u", 32), C), T, atomOp(M), fbHead(fbBases(F, hAddr(L, C), rd(Cn, ty("u", 64), C), 8, false))) ... </tk>
       <fabtab> F </fabtab>
    requires "try_red" in M andBool notBool "tensor" in M
  syntax KItem ::= #fbRed(List, Loc, Loc, Int, Ty, String, Int) | #fbRedV(List, Loc, Int, Ty, String, Int)
  rule <tk> #fbRed(.List, _, _, _, _, _, _) => #fault("fabric access outside the logical endpoint", 0) ... </tk>
  rule <tk> #fbRed(Ds, S, B, N, T, OP, Cn) => #amem(S, rdOp(N)) ~> #fbRedV(Ds, B, N, T, OP, Cn) ... </tk> requires Ds =/=K .List
  rule <tk> #res(V) ~> #fbRedV(Ds, B, N, T, OP, Cn) => #seq(fbRedOps(Ds, V, N /Int bytes(T), T, OP)) ~> #fbCnt(Cn, N) ~> #bulkTx(B, N /Int 16) ... </tk>
  syntax List ::= fbRedOps(List, Int, Int, Ty, String) [function, total]
  rule fbRedOps(ListItem(A:Int) As, V, NE, T, OP) => fbRedDev(A, V, 0, NE, T, OP) fbRedOps(As, V, NE, T, OP)
  rule fbRedOps(_, _, _, _, _) => .List [owise]
  syntax KItem ::= #memDrop(Loc, MemOp)
  rule <tk> #memDrop(L, O) => #amem(L, O) ~> #drop ... </tk>
  syntax List ::= fbRedDev(Int, Int, Int, Int, Ty, String) [function, total]
  rule fbRedDev(A, V, I, NE, T, OP) =>
       ListItem(#memDrop(loc("global", A +Int I *Int bytes(T)), atOp(OP, T, (V >>Int (8 *Int bytes(T) *Int I)) &Int mask(8 *Int bytes(T)), 0))) fbRedDev(A, V, I +Int 1, NE, T, OP)
    requires I <Int NE
  rule fbRedDev(_, _, _, _, _, _) => .List [owise]

  // ---- try_pullred: warp-collective; lane 0 reads every member and reduces
  syntax KItem ::= #fbPull(List, Loc, Loc, Int, Ty, Set, List)
  rule <tk> exec(opc("fabric", M, ListItem(T)), ops([ D:Addr ], [ L:Addr , O:Operand ], Sz, [ B:Addr ], _:Operands), C) =>
            #fbPull(fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), rd(Sz, ty("u", 32), C), true), ea(D, "shared", C), ea(B, "shared", C), rd(Sz, ty("u", 32), C), T, M, .List) ... </tk>
       <fabtab> F </fabtab>
    requires "try_pullred" in M andBool rreg("%laneid", C) ==Int 0
  rule <tk> exec(opc("fabric", M, _), ops([ _:Addr ], [ _:Addr , _:Operand ], _, [ _:Addr ], _:Operands), C) => .K ... </tk>
    requires "try_pullred" in M andBool rreg("%laneid", C) =/=Int 0
  syntax KItem ::= #fbPullG(List, Loc, Loc, Int, Ty, Set, List)
  rule <tk> #fbPull(.List, D, B, N, T, M, Vs) => #fault("fabric access outside the logical endpoint", 0) ... </tk> requires Vs ==K .List
  rule <tk> #fbPull(ListItem(A:Int) As, D, B, N, T, M, Vs) => #amem(loc("global", A), rdOp(N)) ~> #fbPullG(As, D, B, N, T, M, Vs) ... </tk>
  rule <tk> #res(V) ~> #fbPullG(As, D, B, N, T, M, Vs) => #fbPull(As, D, B, N, T, M, Vs ListItem(V)) ... </tk>
  rule <tk> #fbPull(.List, D, B, N, T, M, Vs) =>
            #amem(D, wrOp(N, fbPack(mmOp(M), T, mmAcc(M, T), Vs, 0, N /Int bytes(T)))) ~> #drop ~> #bulkTx(B, N) ... </tk> requires Vs =/=K .List
  syntax Int ::= fbPack(String, Ty, Ty, List, Int, Int) [function, total]
  rule fbPack(OP, T, A, Vs, I, NE) =>
       (mmRed(OP, T, A, fbElem(Vs, bytes(T), I)) <<Int (8 *Int bytes(T) *Int I)) |Int fbPack(OP, T, A, Vs, I +Int 1, NE) requires I <Int NE
  rule fbPack(_, _, _, _, _, _) => 0 [owise]
  syntax List ::= fbElem(List, Int, Int) [function, total]
  rule fbElem(ListItem(V:Int) Vs, W, I) => ListItem((V >>Int (8 *Int W *Int I)) &Int mask(8 *Int W)) fbElem(Vs, W, I)
  rule fbElem(_, _, _) => .List [owise]

  // ---- try_atom: 16-byte atomic; the old element goes to smem_dst + (off & 15)
  syntax KItem ::= #fbAt0(Int, Int, Loc, Loc, Loc, Ty, String) | #fbAt1(Int, Loc, Loc, Ty, String) | #fbAt2(Int, Loc, Ty, String, Int) | #fbAtOut(Loc, Int)
  rule <tk> exec(opc("fabric", M, ListItem(T)), ops([ L:Addr , O:Operand ], [ DS:Addr ], [ SS:Addr ], [ B:Addr ], _:Operands), C) =>
            #fbAt0(fbHead(fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), bytes(T), false)), rd(O, ty("u", 64), C) %Int 16,
                   ea(DS, "shared", C), ea(SS, "shared", C), ea(B, "shared", C), T, atomOp(M)) ... </tk>
       <fabtab> F </fabtab>
    requires "try_atom" in M
  rule <tk> #fbAt0(A, _, _, _, _, _, _) => #fault("fabric access outside the logical endpoint", 0) ... </tk> requires A <Int 0
  rule <tk> #fbAt0(A, So, D, S, B, T, OP) => #amem(fbAdd(S, So), rdOp(bytes(T))) ~> #fbAt1(A, fbAdd(D, So), fbAdd(S, 16 +Int So), T, OP) ~> #bulkTx(B, 1) ... </tk> requires A >=Int 0
  // operand b (and, for cas, the swap value 16 bytes further on)
  rule <tk> #res(V) ~> #fbAt1(A, DE, SW, T, "cas") => #amem(SW, rdOp(bytes(T))) ~> #fbAt2(A, DE, T, "cas", V) ... </tk>
  rule <tk> #res(V) ~> #fbAt1(A, DE, _, T, OP) => #amem(loc("global", A), atOp(OP, T, V, 0)) ~> #fbAtOut(DE, bytes(T)) ... </tk> requires OP =/=String "cas"
  rule <tk> #res(X) ~> #fbAt2(A, DE, T, OP, V) => #amem(loc("global", A), atOp(OP, T, V, X)) ~> #fbAtOut(DE, bytes(T)) ... </tk>
  rule <tk> #res(Old) ~> #fbAtOut(DE, W) => #amem(DE, wrOp(W, Old)) ~> #drop ... </tk>

  // ---- tensor forms: the tensor-map copy against endpoint memory
  syntax KItem ::= #fbTx(Loc, Int)
  rule <tk> #fbTx(B, N) => #bulkTx(B, N) ... </tk>
  // try_get.async.tensor [dstMem], [tensorMap, leId, off, override, coords], [mbar]
  rule <tk> exec(opc("fabric", M, _), ops([ D:Addr ], [ T:Addr , L:Addr , O:Operand , Ov:Operand , Cs:Operand ], [ B:Addr ], _:Operands), C) =>
            #tmLd(M SetItem("bulk"), D, tmOvDims(tmOvAddr(tmObj(GM, ea(T, "generic", C)), fbHead(fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), 0, false))), tmCoords(Ov, C), 0), Cs, B, .Operands, C) ... </tk>
       <gmem> GM </gmem> <fabtab> F </fabtab>
    requires "try_get" in M andBool "tensor" in M
  // try_put / try_red .async.tensor [tensorMap, leId, off{, counterOff}, override, coords], [src], sizeBytes, [mbar]
  rule <tk> exec(opc("fabric", M, _), ops([ T:Addr , L:Addr , O:Operand , Ov:Operand , Cs:Operand ], [ S:Addr ], Sz, [ B:Addr ], _:Operands), C) =>
            #seq(fbStores(fbBlobs(tmObj(GM, ea(T, "generic", C)), fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), 0, "multimem" in M), tmCoords(Ov, C)), fbTM(M), Cs, S, C))
            ~> #fbTx(ea(B, "shared", C), rd(Sz, ty("u", 32), C) /Int 16) ... </tk>
       <gmem> GM </gmem> <fabtab> F </fabtab>
    requires ("try_put" in M orBool "try_red" in M) andBool "tensor" in M
  rule <tk> exec(opc("fabric", M, _), ops([ T:Addr , L:Addr , O:Operand , Cn:Operand , Ov:Operand , Cs:Operand ], [ S:Addr ], Sz, [ B:Addr ], _:Operands), C) =>
            #seq(fbStores(fbBlobs(tmObj(GM, ea(T, "generic", C)), fbBases(F, hAddr(L, C), rd(O, ty("u", 64), C), 0, "multimem" in M), tmCoords(Ov, C)), fbTM(M), Cs, S, C))
            ~> #fbCnt(fbHead(fbBases(F, hAddr(L, C), rd(Cn, ty("u", 64), C), 8, false)), rd(Sz, ty("u", 32), C))
            ~> #fbTx(ea(B, "shared", C), rd(Sz, ty("u", 32), C) /Int 16) ... </tk>
       <gmem> GM </gmem> <fabtab> F </fabtab>
    requires ("try_put" in M orBool "try_red" in M) andBool "tensor" in M
  syntax List ::= fbBlobs(Int, List, List) [function, total] | fbStores(List, Set, Operand, Addr, Ctx) [function, total]
  rule fbBlobs(Bl, ListItem(A:Int) As, Ds) => ListItem(tmOvDims(tmOvAddr(Bl, A), Ds, 0)) fbBlobs(Bl, As, Ds)
  rule fbBlobs(_, _, _) => .List [owise]
  rule fbStores(ListItem(Bl:Int) Bls, M, Cs, S, C) => ListItem(#tmSt(M, Bl, Cs, S, C)) fbStores(Bls, M, Cs, S, C)
  rule fbStores(_, _, _, _, _) => .List [owise]
  rule <tk> exec(opc("fabric", M, _), ops(.Operands), _) => .K ... </tk> requires "submit" in M orBool "wait" in M
  syntax Set ::= fbTM(Set) [function, total]
  rule fbTM(M) => M SetItem("bulk") SetItem("bulk_group") SetItem("reduce") requires "try_red" in M
  rule fbTM(M) => M SetItem("bulk") SetItem("bulk_group") [owise]
endmodule

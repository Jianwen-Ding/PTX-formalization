// =============================================================================
// PTX-EXEC-MM: multimem.ld_reduce / multimem.st / multimem.red (sm_90, PTX ISA 8.1)
//
// A multimem address designates the same offset in several memories, one per participating device.
// The harness builds such an object with
//     .multimem MM .size N { BUF0, BUF1, ... } ;
// which allocates an N-byte window in global memory (the multimem address range; it must only be used with multimem.*
// instructions) and records, in <mmtab>, the buffers that stand for the devices' memories.  An address a inside the
// window designates offset (a - window start) of every device buffer.
//   multimem.st       writes the value to every device location (as an ordinary st on each, in device order)
//   multimem.red      reduces the operand into every device location (an ordinary red on each)
//   multimem.ld_reduce loads from every device and reduces the values in device order: integers exactly; floating
//                     point step by step in the accumulation precision (the element type by default, or .acc::f32 /
//                     .acc::f16); 8-bit floats combine through f16 and round back to the element type (rn, satfinite).
// =============================================================================
requires "ptx-exec-memq.k"
requires "ptx-exec-mma.k"
requires "ptx-exec-tex.k"

module PTX-EXEC-MM
  imports PTX-EXEC-MEMQ
  imports PTX-EXEC-MMA
  imports PTX-EXEC-TEX

  syntax MmEnt ::= mm(Int, List)
  // window containing an address -> (start, entry)
  syntax Int ::= mmStart(Map, List, Int) [function, total]
  rule mmStart(T, ListItem(K:Int) Ks, A) => K requires A >=Int K andBool A <Int K +Int mmSize({T[K]}:>MmEnt)
  rule mmStart(T, ListItem(_) Ks, A) => mmStart(T, Ks, A) [owise]
  rule mmStart(_, .List, _) => 0 -Int 1
  syntax Int ::= mmSize(MmEnt) [function, total]
  rule mmSize(mm(S, _)) => S
  syntax List ::= mmBases(MmEnt) [function, total]
  rule mmBases(mm(_, B)) => B
  // device addresses of an address inside a window
  syntax List ::= mmAddrs(Map, Int) [function, total]
  rule mmAddrs(T, A) => mmAddAll(mmBases({T[mmStart(T, keys_list(T), A)]}:>MmEnt), A -Int mmStart(T, keys_list(T), A))
    requires mmStart(T, keys_list(T), A) >=Int 0
  rule mmAddrs(_, _) => .List [owise]
  syntax List ::= mmAddAll(List, Int) [function, total]
  rule mmAddAll(ListItem(B:Int) Bs, O) => ListItem(B +Int O) mmAddAll(Bs, O)
  rule mmAddAll(_, _) => .List [owise]
  syntax List ::= mmAddElem(List, Int) [function, total]
  rule mmAddElem(ListItem(B:Int) Bs, O) => ListItem(B +Int O) mmAddElem(Bs, O)
  rule mmAddElem(_, _) => .List [owise]

  syntax KItem ::= #seq(List)
  rule <tk> #seq(ListItem(I:KItem) L) => I ~> #seq(L) ... </tk>
  rule <tk> #seq(.List) => .K ... </tk>

  syntax Operand ::= mmAddrOp(Int) [function, total]
  rule mmAddrOp(A) => [ A ]
  // ---- st / red: the same instruction on every device location
  syntax List ::= mmEach(String, Set, List, Operand, List, Ctx) [function, total]
  rule mmEach(N, M, Ts, B, ListItem(A:Int) As, C) =>
       ListItem(exec(opc(N, M SetItem("global"), Ts), ops(mmAddrOp(A), B, .Operands), C)) mmEach(N, M, Ts, B, As, C)
  rule mmEach(_, _, _, _, _, _) => .List [owise]
  rule <tk> exec(opc("multimem", M, Ts), ops([ A ], B, .Operands), C) => #seq(mmEach("st", M, Ts, B, mmAddrs(T, hAddr(A, C)), C)) ... </tk>
       <mmtab> T </mmtab>
    requires "st" in M andBool mmStart(T, keys_list(T), hAddr(A, C)) >=Int 0
  rule <tk> exec(opc("multimem", M, Ts), ops([ A ], B, .Operands), C) => #seq(mmEach("red", M, Ts, B, mmAddrs(T, hAddr(A, C)), C)) ... </tk>
       <mmtab> T </mmtab>
    requires "red" in M andBool mmStart(T, keys_list(T), hAddr(A, C)) >=Int 0
  rule <tk> exec(opc("multimem", M, _), ops([ A ], _, .Operands), C) => #fault("multimem address expected", hAddr(A, C)) ... </tk>
       <mmtab> T </mmtab>
    requires ("st" in M orBool "red" in M) andBool mmStart(T, keys_list(T), hAddr(A, C)) <Int 0

  // ---- ld_reduce
  syntax KItem ::= #mmRd(List, Ty, List, Set, Int, Int, Operand) | #mmGot(List, Ty, List, Set, Int, Int, Operand)
  syntax Int ::= mmVec(Set) [function, total]
  rule mmVec(M) => 8 requires "v8" in M
  rule mmVec(M) => 4 requires "v4" in M andBool notBool "v8" in M
  rule mmVec(M) => 2 requires "v2" in M andBool notBool ("v4" in M orBool "v8" in M)
  rule mmVec(_) => 1 [owise]
  // device-major addresses for each element: element e at offset e*bytes
  syntax List ::= mmElems(List, Int, Int, Int) [function, total]          // device bases, offset, element bytes, element count
  rule mmElems(Bs, O, W, N) => mmAddElem(Bs, O) mmElems(Bs, O +Int W, W, N -Int 1) requires N >Int 0
  rule mmElems(_, _, _, _) => .List [owise]
  rule <tk> exec(opc("multimem", M, ListItem(T)), ops(D, [ A ], .Operands), C) =>
            #mmRd(mmElems(mmBases({Tb[mmStart(Tb, keys_list(Tb), hAddr(A, C))]}:>MmEnt), hAddr(A, C) -Int mmStart(Tb, keys_list(Tb), hAddr(A, C)), bytes(T), mmVec(M)),
                  T, .List, M, size(mmBases({Tb[mmStart(Tb, keys_list(Tb), hAddr(A, C))]}:>MmEnt)), mmVec(M), D) ... </tk>
       <mmtab> Tb </mmtab>
    requires "ld_reduce" in M andBool mmStart(Tb, keys_list(Tb), hAddr(A, C)) >=Int 0
  rule <tk> exec(opc("multimem", M, _), ops(_, [ A ], .Operands), C) => #fault("multimem address expected", hAddr(A, C)) ... </tk>
       <mmtab> Tb </mmtab>
    requires "ld_reduce" in M andBool mmStart(Tb, keys_list(Tb), hAddr(A, C)) <Int 0
  rule <tk> #mmRd(ListItem(L:Int) Ls, T, Vs, M, ND, NE, D) => #mem(loc("global", L), rdOp(bytes(T))) ~> #mmGot(Ls, T, Vs, M, ND, NE, D) ... </tk>
  rule <tk> #res(V) ~> #mmGot(Ls, T, Vs, M, ND, NE, D) => #mmRd(Ls, T, Vs ListItem(V), M, ND, NE, D) ... </tk>
  rule <tk> #mmRd(.List, T, Vs, M, ND, NE, D) => #wmWr(mmDst(D), mmElemsRed(Vs, T, M, ND, NE)) ... </tk>
  syntax List ::= mmDst(Operand) [function, total]
  rule mmDst(D) => vecOps(D) requires vecOps(D) =/=K .List
  rule mmDst(D) => ListItem(D) [owise]
  // the values arrive element-major: for each element, ND device values
  syntax List ::= mmElemsRed(List, Ty, Set, Int, Int) [function, total]
  rule mmElemsRed(Vs, T, M, ND, NE) => ListItem(mmRed(mmOp(M), T, mmAcc(M, T), mmTake(Vs, ND))) mmElemsRed(mmDrop(Vs, ND), T, M, ND, NE -Int 1) requires NE >Int 0
  rule mmElemsRed(_, _, _, _, _) => .List [owise]
  syntax List ::= mmTake(List, Int) [function, total] | mmDrop(List, Int) [function, total]
  rule mmTake(ListItem(X) L, N) => ListItem(X) mmTake(L, N -Int 1) requires N >Int 0
  rule mmTake(_, _) => .List [owise]
  rule mmDrop(ListItem(_) L, N) => mmDrop(L, N -Int 1) requires N >Int 0
  rule mmDrop(L, _) => L [owise]
  syntax String ::= mmOp(Set) [function, total]
  rule mmOp(M) => pick(M, ListItem("add") ListItem("min") ListItem("max") ListItem("and") ListItem("or") ListItem("xor"), "add")
  // accumulation type: the element's lane type unless .acc::f32 / .acc::f16
  syntax Ty ::= mmAcc(Set, Ty) [function, total]
  rule mmAcc(M, _) => ty("f", 32) requires "acc::f32" in M
  rule mmAcc(M, _) => ty("f", 16) requires "acc::f16" in M
  rule mmAcc(_, T) => mmLaneTy(T) [owise]
  syntax Ty ::= mmLaneTy(Ty) [function, total]
  rule mmLaneTy(ty("f16x2", _)) => ty("f", 16)      rule mmLaneTy(ty("bf16x2", _)) => ty("bf", 16)
  rule mmLaneTy(ty("e4m3x2", _)) => ty("e4m3", 8)   rule mmLaneTy(ty("e4m3x4", _)) => ty("e4m3", 8)
  rule mmLaneTy(ty("e5m2x2", _)) => ty("e5m2", 8)   rule mmLaneTy(ty("e5m2x4", _)) => ty("e5m2", 8)
  rule mmLaneTy(T) => T [owise]
  syntax Int ::= mmLaneN(Ty) [function, total]
  rule mmLaneN(ty("f16x2", _)) => 2   rule mmLaneN(ty("bf16x2", _)) => 2   rule mmLaneN(ty("e4m3x2", _)) => 2   rule mmLaneN(ty("e5m2x2", _)) => 2
  rule mmLaneN(ty("e4m3x4", _)) => 4  rule mmLaneN(ty("e5m2x4", _)) => 4
  rule mmLaneN(_) => 1 [owise]
  syntax Bool ::= mmIsFp8(Ty) [function, total]
  rule mmIsFp8(ty(K, _)) => K ==String "e4m3" orBool K ==String "e5m2"
  // reduce the device values of one element
  syntax Int ::= mmRed(String, Ty, Ty, List) [function, total]
  rule mmRed(OP, T, A, Vs) => mmFoldPlain(OP, T, Vs) requires A ==K T andBool notBool mmIsFp8(T)
  rule mmRed(OP, T, A, Vs) => mmLanes(OP, T, A, Vs, 0) [owise]
  // plain: atomic-update semantics fold
  syntax Int ::= mmFoldPlain(String, Ty, List) [function, total]
  rule mmFoldPlain(OP, T, ListItem(V:Int) Vs) => mmFoldPlainN(OP, T, V, Vs)
  rule mmFoldPlain(_, _, _) => 0 [owise]
  syntax Int ::= mmFoldPlainN(String, Ty, Int, List) [function, total]
  rule mmFoldPlainN(OP, T, Acc, ListItem(V:Int) Vs) => mmFoldPlainN(OP, T, atomNew(OP, T, Acc, V, 0), Vs)
  rule mmFoldPlainN(_, _, Acc, _) => Acc [owise]
  // lane-wise: each lane accumulates in the accumulation type
  syntax Int ::= mmLanes(String, Ty, Ty, List, Int) [function, total]
  rule mmLanes(OP, T, A, Vs, J) => (mmLane(OP, mmLaneTy(T), A, mmLanesOf(Vs, mmLaneTy(T), J)) <<Int (8 *Int bytes(mmLaneTy(T)) *Int J)) |Int mmLanes(OP, T, A, Vs, J +Int 1)
    requires J <Int mmLaneN(T)
  rule mmLanes(_, _, _, _, _) => 0 [owise]
  syntax List ::= mmLanesOf(List, Ty, Int) [function, total]
  rule mmLanesOf(ListItem(V:Int) Vs, L, J) => ListItem((V >>Int (8 *Int bytes(L) *Int J)) &Int mask(8 *Int bytes(L))) mmLanesOf(Vs, L, J)
  rule mmLanesOf(_, _, _) => .List [owise]
  syntax Int ::= mmLane(String, Ty, Ty, List) [function, total]
  rule mmLane(OP, L, A, ListItem(V:Int) Vs) => mmBack(L, A, mmStep(OP, L, A, mmTo(L, A, V), Vs))
  rule mmLane(_, _, _, _) => 0 [owise]
  syntax Int ::= mmStep(String, Ty, Ty, Int, List) [function, total]
  rule mmStep(OP, L, A, Acc, ListItem(V:Int) Vs) => mmStep(OP, L, A, mmComb(OP, A, Acc, mmTo(L, A, V)), Vs)
  rule mmStep(_, _, _, Acc, _) => Acc [owise]
  syntax Int ::= mmTo(Ty, Ty, Int) [function, total] | mmBack(Ty, Ty, Int) [function, total] | mmComb(String, Ty, Int, Int) [function, total]
  syntax String ::= mmKind(Ty) [function, total]
  rule mmKind(ty(K, _)) => K
  syntax Float ::= mmFlt(Ty, Int) [function, total]
  rule mmFlt(ty(_, W), V) => roundFloat(decodeF(V, W), WP, WEXP)
  rule mmTo(L, A, V) => V requires L ==K A
  rule mmTo(L, A, V) => miniToF16(mmKind(L), V) requires L =/=K A andBool mmIsFp8(L) andBool A ==K ty("f", 16)
  rule mmTo(L, A, V) => cvt(.Set, A, ty("f", 16), miniToF16(mmKind(L), V)) requires L =/=K A andBool mmIsFp8(L) andBool A =/=K ty("f", 16)
  rule mmTo(L, A, V) => cvt(.Set, A, L, V) [owise]
  rule mmBack(L, A, V) => V requires L ==K A
  rule mmBack(L, A, V) => miniFrom(mmKind(L), mmFlt(A, V), false) requires L =/=K A andBool mmIsFp8(L)
  rule mmBack(L, A, V) => cvt(SetItem("rn"), L, A, V) [owise]
  // 8-bit floats accumulate through f16
  rule mmComb(OP, A, X, Y) => miniFrom(mmKind(A), mmFlt(ty("f", 16), bin(OP, .Set, ty("f", 16), miniToF16(mmKind(A), X), miniToF16(mmKind(A), Y))), false) requires mmIsFp8(A)
  rule mmComb(OP, A, X, Y) => bin(OP, .Set, A, X, Y) [owise]
endmodule

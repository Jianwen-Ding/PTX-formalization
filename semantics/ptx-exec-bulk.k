// =============================================================================
// PTX-EXEC-BULK: bulk asynchronous copies (sm_90)
//
//   cp.async.bulk.{shared::cta,shared::cluster}.global.mbarrier::complete_tx::bytes[...]  [dst], [src], size, [mbar]{, ctaMask}
//   cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes               [dst], [src], size, [mbar]
//   cp.async.bulk.global.shared::cta.bulk_group{.cp_mask}                                [dst], [src], size{, policy}{, byteMask}
//   cp.reduce.async.bulk.<dst>.<src>.<completion>.<redOp>.<type>                         [dst], [src], size{, [mbar] | policy}
//   cp.async.bulk.prefetch.L2.global  /  cp.async.bulk.{commit_group,wait_group}
//
// As for cp.async, the copy is performed at issue; the complete-tx on the mbarrier follows
// the data transfer in the same thread.  .multicast::cluster writes the same CTA-relative
// destination (and signals the same mbarrier offset) in every CTA of the mask.
// =============================================================================
requires "ptx-exec-async.k"

module PTX-EXEC-BULK
  imports PTX-EXEC-ASYNC

  // ---------------------------------------------------------------- reducing store
  syntax MemOp ::= bulkRed(String, Ty, Int, Int)               // op, element type, bytes, source array
  syntax Int ::= redArr(String, Ty, Int, Int, Int, Int) [function, total]      // op, type, old, src, i, n
  rule memNew(bulkRed(OP, T, N, V), M, A) => writeBytes(M, A, N, redArr(OP, T, readBytes(M, A, N), V, 0, N /Int bytes(T)))
  rule memVal(bulkRed(_, _, _, _), _, _) => 0
  rule redArr(OP, T, O, V, I, N) =>
       (atomNew(OP, T, (O >>Int (8 *Int bytes(T) *Int I)) &Int mask(8 *Int bytes(T)), (V >>Int (8 *Int bytes(T) *Int I)) &Int mask(8 *Int bytes(T)), 0)
        <<Int (8 *Int bytes(T) *Int I)) |Int redArr(OP, T, O, V, I +Int 1, N)
    requires I <Int N
  rule redArr(_, _, _, _, _, _) => 0 [owise]

  // ---------------------------------------------------------------- operand helpers
  syntax Addr ::= bulkMbar(List) [function]
  rule bulkMbar(ListItem([ A ]) _) => A
  rule bulkMbar(ListItem(_) L) => bulkMbar(L) [owise]
  syntax Bool ::= hasMbar(List) [function, total]
  rule hasMbar(ListItem([ _ ]) _) => true
  rule hasMbar(ListItem(_) L) => hasMbar(L) [owise]
  rule hasMbar(.List) => false
  // the operand following the first address operand (ctaMask)
  syntax Int ::= bulkMask(List, Ctx) [function, total]
  rule bulkMask(ListItem([ _ ]) ListItem(X:Operand) _, C) => rd(X, ty("u", 32), C)
  rule bulkMask(ListItem(_) L, C) => bulkMask(L, C) [owise]
  rule bulkMask(_, _) => 0 [owise]
  // the byteMask is the last operand of a .cp_mask copy
  syntax Int ::= lastArg(List, Ctx) [function, total]
  rule lastArg(L, C) => rd({L[size(L) -Int 1]}:>Operand, ty("u", 32), C) requires size(L) >Int 0
  rule lastArg(_, _) => 0 [owise]

  // destination / mbarrier locations (one per target CTA)
  syntax List ::= bulkLocs(Loc, Int, Bool) [function, total]                   // base location, ctaMask, multicast?
  rule bulkLocs(L, _, false) => ListItem(L)
  rule bulkLocs(loc(_, A), Mask, true) => rankLocs(A &Int 16777215, Mask, 0)
  rule bulkLocs(L, Mask, true) => ListItem(L) [owise]
  syntax List ::= rankLocs(Int, Int, Int) [function, total]
  rule rankLocs(Off, Mask, R) => #if (Mask >>Int R) &Int 1 ==Int 1 #then ListItem(loc("cluster", ((R +Int 1) <<Int 24) |Int Off)) #else .List #fi rankLocs(Off, Mask, R +Int 1)
    requires R <Int 32 andBool (Mask >>Int R) >Int 0
  rule rankLocs(_, _, _) => .List [owise]

  syntax Bool ::= isMc(Set) [function, total]
  rule isMc(M) => "multicast::cluster" in M orBool "multicast::cluster::16b" in M orBool "multicast::cluster::32b" in M

  // ---------------------------------------------------------------- mbarrier-completed copies (to shared)
  syntax KItem ::= #bulkRd(Loc, List, List, Int) | #bulkWr(List, List, Int) | #bulkEach(List, List, Int, Int) | #bulkTx(Loc, Int)
  // data movement towards shared memory, completed on an mbarrier
  rule <tk> exec(opc("cp", M, _), ops([ D ], [ S ], Sz, Rest:Operands), C) =>
            #mem(ea(S, #if "global" in M #then "generic" #else "shared" #fi, C), rdOp(rd(Sz, ty("u", 32), C)))
            ~> #bulkWr(bulkLocs(ea(D, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(opsList(Rest), C), isMc(M)),
                       bulkLocs(ea(bulkMbar(opsList(Rest)), #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(opsList(Rest), C), isMc(M)),
                       rd(Sz, ty("u", 32), C)) ... </tk>
    requires "async" in M andBool "bulk" in M andBool "mbarrier::complete_tx::bytes" in M andBool hasMbar(opsList(Rest))
             andBool notBool "reduce" in M andBool notBool "tensor" in M
  rule <tk> #res(V) ~> #bulkWr(Ds, Ms, N) => #bulkEach(Ds, Ms, N, V) ... </tk>
  rule <tk> #bulkEach(ListItem(D) Ds, ListItem(Mb) Ms, N, V) =>
            #mem(D, wrOp(N, V)) ~> #drop ~> #bulkTx(Mb, N) ~> #bulkEach(Ds, Ms, N, V) ... </tk>
  rule <tk> #bulkEach(.List, _, _, _) => .K ... </tk>
  rule <tk> #bulkTx(Mb, N) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => MB[mbKey(Mb, CT, GD, CD) <- mbTx(mbGet(MB, mbKey(Mb, CT, GD, CD)), 0 -Int N)] </mbars>
    requires mbKey(Mb, CT, GD, CD) in_keys(MB)

  // ---------------------------------------------------------------- shared::cta -> global (bulk_group)
  syntax KItem ::= #bulkG(Loc, Int, Int) | #bulkMerge(Loc, Int, Int, Int)
  rule <tk> exec(opc("cp", M, _), ops([ D ], [ S ], Sz, Rest:Operands), C) =>
            #mem(ea(S, "shared", C), rdOp(rd(Sz, ty("u", 32), C)))
            ~> #bulkG(ea(D, "generic", C), rd(Sz, ty("u", 32), C), #if "cp_mask" in M #then lastArg(opsList(Rest), C) #else 0 -Int 1 #fi) ... </tk>
    requires "async" in M andBool "bulk" in M andBool "bulk_group" in M andBool notBool "reduce" in M andBool notBool "tensor" in M
  rule <tk> #res(V) ~> #bulkG(D, N, Mask) => #mem(D, wrOp(N, V)) ~> #drop ... </tk> requires Mask <Int 0
  rule <tk> #res(V) ~> #bulkG(D, N, Mask) => #mem(D, rdOp(N)) ~> #bulkMerge(D, N, V, Mask) ... </tk> requires Mask >=Int 0
  rule <tk> #res(Old) ~> #bulkMerge(D, N, V, Mask) =>
            #mem(D, wrOp(N, (V &Int expandMask(Mask, N /Int 16, 0)) |Int (Old &Int (~Int expandMask(Mask, N /Int 16, 0))))) ~> #drop ... </tk>
  // the i-th bit of byteMask selects byte i of every 16-byte chunk
  syntax Int ::= expandMask(Int, Int, Int) [function, total]
  rule expandMask(Mask, Chunks, I) => byteSel(Mask, I %Int 16) <<Int (8 *Int I) |Int expandMask(Mask, Chunks, I +Int 1) requires I <Int Chunks *Int 16
  rule expandMask(_, _, _) => 0 [owise]
  syntax Int ::= byteSel(Int, Int) [function, total]
  rule byteSel(Mask, J) => #if (Mask >>Int J) &Int 1 ==Int 1 #then 255 #else 0 #fi

  // ---------------------------------------------------------------- cp.reduce.async.bulk
  syntax KItem ::= #bulkR(Loc, String, Ty, Int, List)
  rule <tk> exec(opc("cp", M, ListItem(T)), ops([ D ], [ S ], Sz, Rest:Operands), C) =>
            #mem(ea(S, "shared", C), rdOp(rd(Sz, ty("u", 32), C)))
            ~> #bulkR(ea(D, #if "global" in M #then "generic" #else "shared_cluster" #fi, C), atomOp(M), T, rd(Sz, ty("u", 32), C),
                      #if hasMbar(opsList(Rest)) #then ListItem(ea(bulkMbar(opsList(Rest)), "shared_cluster", C)) #else .List #fi) ... </tk>
    requires "reduce" in M andBool "async" in M andBool "bulk" in M andBool notBool "tensor" in M
  rule <tk> #res(V) ~> #bulkR(D, OP, T, N, Mb) => #mem(D, bulkRed(OP, T, N, V)) ~> #drop ~> #bulkR2(Mb, N) ... </tk>
  syntax KItem ::= #bulkR2(List, Int)
  rule <tk> #bulkR2(ListItem(Mb) _, N) => #bulkTx(Mb, N) ... </tk>
  rule <tk> #bulkR2(.List, _) => .K ... </tk>

  // ---------------------------------------------------------------- hints and group management
  rule <tk> exec(opc("cp", M, _), ops(_), _) => .K ... </tk>
    requires "async" in M andBool "bulk" in M andBool ("prefetch" in M orBool "commit_group" in M orBool "wait_group" in M)
endmodule

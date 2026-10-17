// =============================================================================
// PTX-EXEC-ASYNC: cp.async and its completion mechanisms
//
//   cp.async.{ca,cg}.shared.global [dst], [src], cp-size{, src-size | , ignore-src}
//   cp.async.commit_group / wait_group N / wait_all / mbarrier.arrive{.noinc} [mbar]
//
// A copy is performed at issue (a legal schedule for programs that synchronise with
// wait_group / wait_all / an mbarrier before using the data): commit_group and the waits are
// therefore no-ops, and cp.async.mbarrier.arrive performs its arrive-on immediately (without
// .noinc the pending-count increment and the asynchronous decrement cancel).
// =============================================================================
requires "ptx-exec-mbar.k"
requires "ptx-exec-memq.k"

module PTX-EXEC-ASYNC
  imports PTX-EXEC-MBAR
  imports PTX-EXEC-MEMQ

  rule isCollective("cp") => true      // visible steps: the copies touch shared/global memory

  syntax KItem ::= #cpRd(Loc, Loc, Int, Int) | #cpWr(Loc, Int)

  // number of bytes taken from the source: cp-size, src-size, or 0 for a true ignore-src predicate
  syntax Int ::= cpSrcN(Int, List, Ctx, Map) [function, total]
  rule cpSrcN(N, .List, _, _) => N
  rule cpSrcN(N, ListItem(X:Int) _, _, _) => minInt(X, N)
  rule cpSrcN(N, ListItem(X:Operand) _, C, RW) => #if rd(X, ty("u", 32), C) ==Int 0 #then N #else 0 #fi
    requires notBool isInt(X) andBool regW(RW, X) ==Int 1
  rule cpSrcN(N, ListItem(X:Operand) _, C, RW) => minInt(rd(X, ty("u", 32), C), N)
    requires notBool isInt(X) andBool regW(RW, X) =/=Int 1

  rule <tk> exec(opc("cp", M, _), ops([ D ], [ S ], N:Int, Rest:Operands), C) =>
            #cpRd(ea(S, "generic", C), ea(D, "shared", C), N, cpSrcN(N, opsList(Rest), C, RW)) ... </tk>
       <rw> RW </rw>
    requires "async" in M andBool notBool ("bulk" in M orBool "mbarrier" in M orBool "commit_group" in M orBool "wait_group" in M orBool "wait_all" in M)
  rule <tk> #cpRd(L, Dl, N, K) => #mem(L, rdOp(K)) ~> #cpWr(Dl, N) ... </tk> requires K >Int 0
  rule <tk> #cpRd(_, Dl, N, 0) => #res(0) ~> #cpWr(Dl, N) ... </tk>
  rule <tk> #res(V) ~> #cpWr(Dl, N) => #mem(Dl, wrOp(N, V)) ~> #drop ... </tk>

  rule <tk> exec(opc("cp", M, _), ops(_), _) => .K ... </tk>
    requires "async" in M andBool ("commit_group" in M orBool "wait_group" in M orBool "wait_all" in M) andBool notBool "bulk" in M

  // cp.async.mbarrier.arrive{.noinc} [addr]
  rule <tk> exec(opc("cp", M, _), ops([ A ], .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => MB[mbKey(ea(A, "shared", C), CT, GD, CD) <- mbArrive(mbGet(MB, mbKey(ea(A, "shared", C), CT, GD, CD)), 1)] </mbars>
    requires "async" in M andBool "mbarrier" in M andBool "noinc" in M andBool mbKey(ea(A, "shared", C), CT, GD, CD) in_keys(MB)
  rule <tk> exec(opc("cp", M, _), ops([ _ ], .Operands), _) => .K ... </tk>
    requires "async" in M andBool "mbarrier" in M andBool notBool "noinc" in M
endmodule

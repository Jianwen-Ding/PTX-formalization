// =============================================================================
// PTX-EXEC-ALU: ALU, setp, mov, cvt and cvta instructions
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-ALU
  imports PTX-EXEC-CORE

  // ------------------------------------------------------------ ALU
  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bin(OP, M, T, rd(A, T, C), rd(B, src2Ty(OP, T), C))) </regs>
    requires OP in BINOPS andBool notBool "cc" in M andBool notBool isPacked(T)

  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, un(OP, M, T, rd(A, T, C))) </regs>
    requires OP in UNOPS andBool notBool isPacked(T)

  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, tern(OP, M, T, rd(A, T, C), rd(B, T, C), rd(X, src3Ty(OP, M, T), C))) </regs>
    requires OP in TERNOPS andBool notBool "cc" in M andBool notBool isPacked(T)

  // setp.cmp[.bop].type  p[|q], a, b[, c]
  syntax Map ::= wrP(Map, Operand, Int, String, Int) [function, total]
  rule wrP(R, P:Reg | Q:Reg, V, BO, X) =>
       R[Reg2String(P) <- bop(V, BO, X)][Reg2String(Q) <- bop(1 -Int V, BO, X)]
  rule wrP(R, P:Name | Q:Name, V, BO, X) =>
       R[Name2String(P) <- bop(V, BO, X)][Name2String(Q) <- bop(1 -Int V, BO, X)]
  rule wrP(R, D, V, BO, X) => wr(R, D, bop(V, BO, X)) [owise]
  // .ftz: subnormal float inputs compare as sign-preserving zeros
  syntax Int ::= setpIn(Set, Ty, Int) [function, total]
  rule setpIn(M, T, V) => ftzIn(M, fw(T), V) requires isF(T)
  rule setpIn(_, _, V) => V [owise]
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wrP(R, D, cmp(cmpOp(M), T, setpIn(M, T, rd(A, T, C)), setpIn(M, T, rd(B, T, C))), "none", 0) </regs>
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wrP(R, D, cmp(cmpOp(M), T, setpIn(M, T, rd(A, T, C)), setpIn(M, T, rd(B, T, C))), boolOp(M), rd(X, ty("p", 1), C)) </regs>

  // mov, including vector pack/unpack: mov.b64 %rd, {%r0, %r1}
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, rd(A, T, C)) </regs>
    requires notBool isVec(D) andBool notBool isVec(A)
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, { Vs }, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack(opsList(Vs), wid(T) /Int size(opsList(Vs)), 0, C)) </regs>
    requires notBool isVec(D)
  rule <tk> exec(opc("mov", _, ListItem(T)), ops({ Ds }, A, .Operands), C) => .K ... </tk>
       <regs> R => unpack(R, opsList(Ds), wid(T) /Int size(opsList(Ds)), rd(A, T, C)) </regs>
    requires notBool isVec(A)
  // vector to vector (mov.v2.u32 {a, b}, {c, d}), elementwise
  rule <tk> exec(opc("mov", _, ListItem(T)), ops({ Ds }, { Vs }, .Operands), C) => .K ... </tk>
       <regs> R => movv(R, opsList(Ds), opsList(Vs), T, C) </regs>
  syntax Map ::= movv(Map, List, List, Ty, Ctx) [function, total]
  rule movv(R, ListItem(D:Operand) Ds, ListItem(V:Operand) Vs, T, C) => movv(wr(R, D, rd(V, T, C)), Ds, Vs, T, C)
  rule movv(R, _, _, _, _) => R [owise]

  syntax Int ::= pack(List, Int, Int, Ctx) [function]
  rule pack(ListItem(V:Operand) L, W, Sh, C) => (rd(V, ty("b", W), C) <<Int Sh) |Int pack(L, W, Sh +Int W, C)
  rule pack(.List, _, _, _) => 0
  syntax Map ::= unpack(Map, List, Int, Int) [function]
  rule unpack(R, ListItem(D:Operand) L, W, V) => unpack(wr(R, D, trunc(V, W)), L, W, V >>Int W)
  rule unpack(R, .List, _, _) => R

  // cvt.[mods].dtype.stype
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, extReg(cvt(M, DT, ST, rd(A, ST, C)), DT, RW, D)) </regs> <rw> RW </rw>

  // cvta[.to].space.size: move between a state space and the generic space
  rule <tk> exec(opc("cvta", M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(#if "to" in M #then rd(A, T, C) -Int winBase(spaceOf(M))
                                  #else rd(A, T, C) +Int winBase(spaceOf(M)) #fi, wid(T))) </regs>
    requires notBool "shared::cluster" in M
  // shared::cluster addresses are (rank + 1) << 24 | offset (no rank bits: the executing CTA); the generic window of the cluster is at CLBASE
  rule <tk> exec(opc("cvta", M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(#if "to" in M #then clFromGeneric(rd(A, T, C), rreg("%cluster_ctarank", C))
                                  #else clToGeneric(rd(A, T, C), rreg("%cluster_ctarank", C)) #fi, wid(T))) </regs>
    requires "shared::cluster" in M
  syntax Int ::= clToGeneric(Int, Int) [function, total] | clFromGeneric(Int, Int) [function, total]
  rule clToGeneric(A, Own) => CLBASE +Int ((#if A >>Int 24 ==Int 0 #then Own #else (A >>Int 24) -Int 1 #fi) <<Int 24) +Int (A &Int 16777215)
  rule clFromGeneric(A, _) => ((((A -Int CLBASE) >>Int 24) +Int 1) <<Int 24) |Int ((A -Int CLBASE) &Int 16777215) requires A >=Int CLBASE
  rule clFromGeneric(A, Own) => ((Own +Int 1) <<Int 24) |Int (A -Int SHBASE) requires A >=Int SHBASE andBool A <Int LOCBASE
  rule clFromGeneric(_, _) => 0 [owise]
endmodule

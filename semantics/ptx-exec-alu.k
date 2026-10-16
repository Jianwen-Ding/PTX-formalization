// =============================================================================
// PTX-EXEC-ALU: ALU, setp, mov, cvt and cvta instructions
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-ALU
  imports PTX-EXEC-CORE

  // ------------------------------------------------------------ ALU
  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bin(OP, M, T, rd(A, T, C), rd(B, src2Ty(OP, T), C))) </regs>
    requires OP in BINOPS

  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, un(OP, M, T, rd(A, T, C))) </regs>
    requires OP in UNOPS

  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, tern(OP, M, T, rd(A, T, C), rd(B, T, C), rd(X, src3Ty(OP, M, T), C))) </regs>
    requires OP in TERNOPS

  // setp.cmp[.bop].type  p[|q], a, b[, c]
  syntax Map ::= wrP(Map, Operand, Int, String, Int) [function, total]
  rule wrP(R, P:Reg | Q:Reg, V, BO, X) =>
       R[Reg2String(P) <- bop(V, BO, X)][Reg2String(Q) <- bop(1 -Int V, BO, X)]
  rule wrP(R, D, V, BO, X) => wr(R, D, bop(V, BO, X)) [owise]
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wrP(R, D, cmp(cmpOp(M), T, rd(A, T, C), rd(B, T, C)), "none", 0) </regs>
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wrP(R, D, cmp(cmpOp(M), T, rd(A, T, C), rd(B, T, C)), boolOp(M), rd(X, ty("p", 1), C)) </regs>

  // mov, including vector pack/unpack: mov.b64 %rd, {%r0, %r1}
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, rd(A, T, C)) </regs>
    requires notBool isVec(D) andBool notBool isVec(A)
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, { Vs }, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack(opsList(Vs), wid(T) /Int size(opsList(Vs)), 0, C)) </regs>
  rule <tk> exec(opc("mov", _, ListItem(T)), ops({ Ds }, A, .Operands), C) => .K ... </tk>
       <regs> R => unpack(R, opsList(Ds), wid(T) /Int size(opsList(Ds)), rd(A, T, C)) </regs>

  syntax Int ::= pack(List, Int, Int, Ctx) [function]
  rule pack(ListItem(V:Operand) L, W, Sh, C) => (rd(V, ty("b", W), C) <<Int Sh) |Int pack(L, W, Sh +Int W, C)
  rule pack(.List, _, _, _) => 0
  syntax Map ::= unpack(Map, List, Int, Int) [function]
  rule unpack(R, ListItem(D:Operand) L, W, V) => unpack(wr(R, D, trunc(V, W)), L, W, V >>Int W)
  rule unpack(R, .List, _, _) => R

  // cvt.[mods].dtype.stype
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, cvt(M, DT, ST, rd(A, ST, C))) </regs>

  // cvta[.to].space.size: move between a state space and the generic space
  rule <tk> exec(opc("cvta", M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(#if "to" in M #then rd(A, T, C) -Int winBase(spaceOf(M))
                                  #else rd(A, T, C) +Int winBase(spaceOf(M)) #fi, wid(T))) </regs>
endmodule

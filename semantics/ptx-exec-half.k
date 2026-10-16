// =============================================================================
// PTX-EXEC-HALF: lane-wise arithmetic on packed half-precision pairs
// (.f16x2 and .bf16x2).  Scalar .f16 and .bf16 arithmetic needs no extra
// rules: the generic float rules work on any format (see fw/prec/ebits in
// ptx-common.k).
//
//   add sub mul min max (binary)   fma mad (ternary)   neg abs ex2 tanh (unary)
//   setp.CmpOp.f16x2 p|q, a, b     set.CmpOp.f16x2.f16x2 d, a, b
// =============================================================================
requires "ptx-exec-intx.k"

module PTX-EXEC-HALF
  imports PTX-EXEC-INTX

  syntax Bool ::= isHalfPair(Ty) [function, total]
  rule isHalfPair(ty(K, _)) => K in (SetItem("f16x2") SetItem("bf16x2"))
  rule laneTy(ty("f16x2", _)) => ty("f", 16)
  rule laneTy(ty("bf16x2", _)) => ty("bf", 16)

  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(bin(OP, M, laneTy(T), lane16(rd(A, T, C), 0), lane16(rd(B, T, C), 0)),
                                   bin(OP, M, laneTy(T), lane16(rd(A, T, C), 1), lane16(rd(B, T, C), 1)))) </regs>
    requires isHalfPair(T) andBool OP in (SetItem("add") SetItem("sub") SetItem("mul") SetItem("min") SetItem("max"))
  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(tern(OP, M, laneTy(T), lane16(rd(A, T, C), 0), lane16(rd(B, T, C), 0), lane16(rd(X, T, C), 0)),
                                   tern(OP, M, laneTy(T), lane16(rd(A, T, C), 1), lane16(rd(B, T, C), 1), lane16(rd(X, T, C), 1)))) </regs>
    requires isHalfPair(T) andBool OP in (SetItem("fma") SetItem("mad"))
  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(un(OP, M, laneTy(T), lane16(rd(A, T, C), 0)),
                                   un(OP, M, laneTy(T), lane16(rd(A, T, C), 1)))) </regs>
    requires isHalfPair(T) andBool OP in (SetItem("neg") SetItem("abs") SetItem("ex2") SetItem("tanh"))

  // setp.CmpOp.f16x2 p|q, a, b : p = lane 0 compare, q = lane 1 compare
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(P:Reg | Q:Reg, A, B, .Operands), C) => .K ... </tk>
       <regs> R => R[Reg2String(P) <- cmp(cmpOp(M), laneTy(T), lane16(rd(A, T, C), 0), lane16(rd(B, T, C), 0))]
                    [Reg2String(Q) <- cmp(cmpOp(M), laneTy(T), lane16(rd(A, T, C), 1), lane16(rd(B, T, C), 1))] </regs>
    requires isHalfPair(T)
  // set.CmpOp.dtype.f16x2 d, a, b : per lane, true is 1.0 (dtype .f16x2/.bf16x2) or 0xFFFF (integer dtype)
  // (observed on hardware: ZLUDA set_gt_f16x2 and set_f16)
  syntax Int ::= laneTrue(Ty) [function, total]
  rule laneTrue(ty("f16x2", _))  => 15360     // 0x3C00 = 1.0 in binary16
  rule laneTrue(ty("bf16x2", _)) => 16256     // 0x3F80 = 1.0 in bfloat16
  rule laneTrue(_) => 65535 [owise]
  rule <tk> exec(opc("set", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(cmp(cmpOp(M), laneTy(ST), lane16(rd(A, ST, C), 0), lane16(rd(B, ST, C), 0)) *Int laneTrue(DT),
                                   cmp(cmpOp(M), laneTy(ST), lane16(rd(A, ST, C), 1), lane16(rd(B, ST, C), 1)) *Int laneTrue(DT))) </regs>
    requires isHalfPair(ST)
endmodule

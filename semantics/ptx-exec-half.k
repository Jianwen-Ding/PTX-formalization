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

  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(bin(OP, M, laneTy(T), lane16(rd(A, T, C), 0), lane16(rd(B, T, C), 0)),
                                   bin(OP, M, laneTy(T), lane16(rd(A, T, C), 1), lane16(rd(B, T, C), 1)))) </regs>
    requires isHalfPair(T) andBool OP in (SetItem("add") SetItem("sub") SetItem("mul") SetItem("min") SetItem("max"))
  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(tern(OP, M, laneTy(T), lane16(rd(A, T, C), 0), lane16(rd(B, T, C), 0), lane16(rd(X, T, C), 0)),
                                   tern(OP, M, laneTy(T), lane16(rd(A, T, C), 1), lane16(rd(B, T, C), 1), lane16(rd(X, T, C), 1)))) </regs>
    requires isHalfPair(T) andBool OP in (SetItem("fma") SetItem("mad"))
  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(un(OP, M, laneTy(T), lane16(rd(A, T, C), 0)),
                                   un(OP, M, laneTy(T), lane16(rd(A, T, C), 1)))) </regs>
    requires isHalfPair(T) andBool OP in (SetItem("neg") SetItem("abs") SetItem("ex2") SetItem("tanh"))

  // add / sub / mul / fma .f32x2 (sm_100): two f32 lanes in a 64-bit register, each rounded as the scalar instruction
  syntax Int ::= lane32(Int, Int) [function, total] | pack32(Int, Int) [function, total]
  rule lane32(V, I) => (V >>Int (32 *Int I)) &Int 4294967295
  rule pack32(L0, L1) => (L0 &Int 4294967295) |Int ((L1 &Int 4294967295) <<Int 32)
  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack32(bin(OP, M, ty("f", 32), lane32(rd(A, T, C), 0), lane32(rd(B, T, C), 0)),
                                   bin(OP, M, ty("f", 32), lane32(rd(A, T, C), 1), lane32(rd(B, T, C), 1)))) </regs>
    requires T ==K ty("f32x2", 64) andBool OP in (SetItem("add") SetItem("sub") SetItem("mul"))
  rule <tk> #gx(opc("fma", M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack32(tern("fma", M, ty("f", 32), lane32(rd(A, T, C), 0), lane32(rd(B, T, C), 0), lane32(rd(X, T, C), 0)),
                                   tern("fma", M, ty("f", 32), lane32(rd(A, T, C), 1), lane32(rd(B, T, C), 1), lane32(rd(X, T, C), 1)))) </regs>
    requires T ==K ty("f32x2", 64)

  // ---------------------------------------------------------------- mixed precision add / sub / mul / fma (PTX ISA 8.6 / 9.4)
  //   .f32 <- .f16/.bf16 operand a (up-convert), .f32x2 <- .f16x2/.bf16x2 a, .f16x2/.bf16x2 <- .f32x2 a and c (down-convert:
  //   computed in rz and rounded to the narrow type with rz), and mul .bf16x2/.f16x2 with the c operand converted to the type of a
  syntax Bool ::= isNarrowF(Ty) [function, total] | isNarrowPair(Ty) [function, total]
  rule isNarrowF(T) => T ==K ty("f", 16) orBool T ==K ty("bf", 16)
  rule isNarrowPair(T) => T ==K ty("f16x2", 32) orBool T ==K ty("bf16x2", 32)
  syntax Set ::= withRz(Set) [function, total]
  rule withRz(M) => (M -Set (SetItem("rn") SetItem("rm") SetItem("rp"))) |Set SetItem("rz")
  syntax Int ::= up32(Ty, Int) [function, total]                      // narrow scalar -> f32 bits
  rule up32(T, V) => cvt(.Set, ty("f", 32), T, V)
  syntax Int ::= mixLane(String, Set, Ty, Ty, Int, Int, Int) [function, total]   // op, modifiers, result lane type, a lane type, a, c (b), lane
  // f32 <- f16/bf16 scalar
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT)), ops(D, A, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bin(OP, M, ty("f", 32), up32(AT, rd(A, AT, C)), rd(X, ty("f", 32), C))) </regs>
    requires DT ==K ty("f", 32) andBool isNarrowF(AT) andBool OP in (SetItem("add") SetItem("sub") SetItem("mul"))
  rule <tk> #gx(opc("fma", M, ListItem(DT) ListItem(AT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, tern("fma", M, ty("f", 32), up32(AT, rd(A, AT, C)), up32(AT, rd(B, AT, C)), rd(X, ty("f", 32), C))) </regs>
    requires DT ==K ty("f", 32) andBool isNarrowF(AT)
  // f32x2 <- f16x2/bf16x2 a, f32x2 c
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(CT)), ops(D, A, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack32(bin(OP, M, ty("f", 32), up32(laneTy(AT), lane16(rd(A, AT, C), 0)), lane32(rd(X, CT, C), 0)),
                                   bin(OP, M, ty("f", 32), up32(laneTy(AT), lane16(rd(A, AT, C), 1)), lane32(rd(X, CT, C), 1)))) </regs>
    requires DT ==K ty("f32x2", 64) andBool isNarrowPair(AT) andBool CT ==K ty("f32x2", 64) andBool OP in (SetItem("add") SetItem("sub") SetItem("mul"))
  rule <tk> #gx(opc("fma", M, ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack32(tern("fma", M, ty("f", 32), up32(laneTy(AT), lane16(rd(A, AT, C), 0)), up32(laneTy(AT), lane16(rd(B, AT, C), 0)), lane32(rd(X, CT, C), 0)),
                                   tern("fma", M, ty("f", 32), up32(laneTy(AT), lane16(rd(A, AT, C), 1)), up32(laneTy(AT), lane16(rd(B, AT, C), 1)), lane32(rd(X, CT, C), 1)))) </regs>
    requires DT ==K ty("f32x2", 64) andBool isNarrowPair(AT) andBool BT ==K ty("f32x2", 64) andBool CT ==K ty("f32x2", 64)
  // f16x2/bf16x2 <- f32x2 a, f32x2 c (down-convert, rz)
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(CT)), ops(D, A, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(cvt(SetItem("rz"), laneTy(DT), ty("f", 32), bin(OP, withRz(M), ty("f", 32), lane32(rd(A, AT, C), 0), lane32(rd(X, CT, C), 0))),
                                   cvt(SetItem("rz"), laneTy(DT), ty("f", 32), bin(OP, withRz(M), ty("f", 32), lane32(rd(A, AT, C), 1), lane32(rd(X, CT, C), 1))))) </regs>
    requires isNarrowPair(DT) andBool AT ==K ty("f32x2", 64) andBool CT ==K ty("f32x2", 64) andBool OP in (SetItem("add") SetItem("sub") SetItem("mul"))
  // mul .bf16x2 <- .bf16x2, .f16x2 (and the converse): c is converted to the type of a, the product rounds in that type
  rule <tk> #gx(opc("mul", M, ListItem(DT) ListItem(AT) ListItem(CT)), ops(D, A, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(bin("mul", M, laneTy(DT), lane16(rd(A, AT, C), 0), cvt(.Set, laneTy(DT), laneTy(CT), lane16(rd(X, CT, C), 0))),
                                   bin("mul", M, laneTy(DT), lane16(rd(A, AT, C), 1), cvt(.Set, laneTy(DT), laneTy(CT), lane16(rd(X, CT, C), 1))))) </regs>
    requires isNarrowPair(DT) andBool DT ==K AT andBool isNarrowPair(CT) andBool DT =/=K CT

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

  // cvt.frnd2{.relu}{.satfinite}.f16x2.f32 d, a, b  (also bf16x2): d.hi = cvt(a), d.lo = cvt(b)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(cvt(M, laneTy(DT), ST, rd(B, ST, C)), cvt(M, laneTy(DT), ST, rd(A, ST, C)))) </regs>
    requires isHalfPair(DT) andBool isF(ST)

  // cvt.pack.sat.type.stype.b32 d, a, b, c : c shifted up, then sat(a), sat(b) packed below it
  rule <tk> exec(opc("cvt", M, ListItem(PT) ListItem(ST) ListItem(_)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc((rd(X, ty("u", 32), C) <<Int (2 *Int wid(PT)))
                                   |Int ((clamp(ival(rd(A, ST, C), ST), PT) &Int mask(wid(PT))) <<Int wid(PT))
                                   |Int (clamp(ival(rd(B, ST, C), ST), PT) &Int mask(wid(PT))), 32)) </regs>
    requires "pack" in M
endmodule

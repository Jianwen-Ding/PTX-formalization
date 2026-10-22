// =============================================================================
// PTX-EXEC-BITS: bit-field extract/insert/find and byte permutation.
//
//   bfe  bfi  bfind[.shiftamt]  fns  bmsk  szext  prmt[.f4e .b4e .rc8 .ecl .ecr .rc16]
//
// Sources: the PTX ISA instruction descriptions; prmt's mode tables and bfind
// cross-checked against gpgpu-sim's functional simulator and the ZLUDA
// conformance programs (tests/zluda: bfe, bfi, bfind_shiftamt, bmsk_clamp_b32,
// prmt, prmt_signext).
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-BITS
  imports PTX-EXEC-CORE

  syntax Int ::= bitAt(Int, Int) [function, total]
  rule bitAt(V, I) => (V >>Int I) &Int 1 requires I >=Int 0
  rule bitAt(_, _) => 0 [owise]

  // ---------------------------------------------------------------- bfe
  // bfe.type d, a, b, c : b = start bit, c = length (low 8 bits of each)
  syntax Int ::= bfe(Ty, Int, Int, Int) [function, total]
  rule bfe(T, A, B, C) => bfeAux(T, A, B &Int 255, C &Int 255)
  syntax Int ::= bfeAux(Ty, Int, Int, Int) [function, total]
  rule bfeAux(T, A, P, L) =>
       (((A >>Int P) &Int mask(bfeLen(T, P, L))) |Int bfeFill(T, A, P, L)) &Int mask(wid(T))
  syntax Int ::= bfeLen(Ty, Int, Int) [function, total]
               | bfeFill(Ty, Int, Int, Int) [function, total]
  rule bfeLen(T, P, L) => #if P >Int wid(T) -Int 1 #then 0 #else minInt(L, wid(T) -Int P) #fi
  // sign bit replicated into the bits above the field (zero for unsigned types or empty fields)
  rule bfeFill(T, A, P, L) =>
       #if isS(T) andBool L >Int 0 andBool bitAt(A, minInt(P +Int L -Int 1, wid(T) -Int 1)) ==Int 1
       #then mask(wid(T)) &Int (~Int mask(bfeLen(T, P, L))) #else 0 #fi

  rule <tk> exec(opc("bfe", _, ListItem(T)), ops(D, A, B, Cn, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bfe(T, rd(A, T, C), rd(B, ty("u", 32), C), rd(Cn, ty("u", 32), C))) </regs>

  // ---------------------------------------------------------------- bfi
  // bfi.type f, a, b, c, d : insert the low d bits of a into b at bit c
  syntax Int ::= bfi(Ty, Int, Int, Int, Int) [function, total]
  rule bfi(T, A, B, P, L) => bfiAux(T, A, B, P &Int 255, L &Int 255)
  syntax Int ::= bfiAux(Ty, Int, Int, Int, Int) [function, total]
  rule bfiAux(T, A, B, P, L) =>
       (B &Int (~Int fieldMask(T, P, L))) |Int ((A <<Int P) &Int fieldMask(T, P, L))
  syntax Int ::= fieldMask(Ty, Int, Int) [function, total]
  rule fieldMask(T, P, L) => #if P >Int wid(T) -Int 1 #then 0 #else mask(minInt(L, wid(T) -Int P)) <<Int P #fi

  rule <tk> exec(opc("bfi", _, ListItem(T)), ops(D, A, B, P, L, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bfi(T, rd(A, T, C), rd(B, T, C), rd(P, ty("u", 32), C), rd(L, ty("u", 32), C))) </regs>

  // ---------------------------------------------------------------- bfind
  // position of the most significant non-sign bit, or 0xFFFFFFFF
  syntax Int ::= bfind(Ty, Bool, Int) [function, total]
  rule bfind(T, Shift, A) => bfindAux(T, Shift,
         #if isS(T) andBool bitAt(A, wid(T) -Int 1) ==Int 1 #then trunc(~Int A, wid(T)) #else trunc(A, wid(T)) #fi)
  syntax Int ::= bfindAux(Ty, Bool, Int) [function, total]
  rule bfindAux(_, _, 0) => 4294967295
  rule bfindAux(T, Shift, A) => #if Shift #then wid(T) -Int 1 -Int log2Int(A) #else log2Int(A) #fi
    requires A >Int 0
  rule bfindAux(_, _, _) => 4294967295 [owise]

  rule <tk> exec(opc("bfind", M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bfind(T, "shiftamt" in M, rd(A, T, C))) </regs>

  // ---------------------------------------------------------------- fns
  // fns.b32 d, mask, base, offset : position of the |offset|-th set bit of
  // mask from base, searching up (offset > 0) or down (offset < 0)
  syntax Int ::= fns(Int, Int, Int) [function, total]
  rule fns(M, B, 0) => #if bitAt(M, B) ==Int 1 #then B #else 4294967295 #fi requires B >=Int 0 andBool B <Int 32
  rule fns(M, B, O) => fnsLoop(M, B, absInt(O) -Int 1, #if O >Int 0 #then 1 #else -1 #fi)
    requires O =/=Int 0 andBool B >=Int 0 andBool B <Int 32
  rule fns(_, _, _) => 4294967295 [owise]
  syntax Int ::= fnsLoop(Int, Int, Int, Int) [function, total]     // mask, pos, remaining, increment
  rule fnsLoop(M, P, N, I) => #if N ==Int 0 #then P #else fnsLoop(M, P +Int I, N -Int 1, I) #fi
    requires P >=Int 0 andBool P <Int 32 andBool bitAt(M, P) ==Int 1
  rule fnsLoop(M, P, N, I) => fnsLoop(M, P +Int I, N, I)
    requires P >=Int 0 andBool P <Int 32 andBool bitAt(M, P) ==Int 0
  rule fnsLoop(_, _, _, _) => 4294967295 [owise]

  rule <tk> exec(opc("fns", _, _), ops(D, M, B, O, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, fns(rd(M, ty("b", 32), C), rd(B, ty("b", 32), C), ival(rd(O, ty("s", 32), C), ty("s", 32)))) </regs>

  // ---------------------------------------------------------------- bmsk
  // bmsk.mode.b32 d, a, b : b ones starting at bit a
  syntax Int ::= bmsk(Bool, Int, Int) [function, total]            // clamp?, a, b
  // the Semantics pseudo-code of bmsk (PTX ISA 9.7.1) on the 32-bit operands
  rule bmsk(CL, A, B) => bmskAux(CL, A &Int 4294967295, B &Int 4294967295)
  syntax Int ::= bmskAux(Bool, Int, Int) [function, total]
  rule bmskAux(CL, A, B) =>
       (#if CL andBool A >=Int 32 #then 0 #else (4294967295 <<Int (A &Int 31)) &Int 4294967295 #fi)
       &Int ~Int (#if (A &Int 31) +Int (B &Int 31) >=Int 32 orBool (CL andBool (A >=Int 32 orBool B >=Int 32)) #then 0
                  #else #if B &Int 31 ==Int 0 #then 4294967295 #else (4294967295 <<Int ((A &Int 31) +Int (B &Int 31))) &Int 4294967295 #fi #fi)

  rule <tk> exec(opc("bmsk", M, _), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bmsk("clamp" in M, rd(A, ty("u", 32), C), rd(B, ty("u", 32), C))) </regs>

  // ---------------------------------------------------------------- szext
  // szext.mode.type d, a, b : sign/zero-extend the low b bits of a
  syntax Int ::= szext(Bool, Ty, Int, Int) [function, total]       // clamp?, type, a, b
  rule szext(true, _, A, B) => A requires B >=Int 32
  rule szext(Clamp, T, A, B) => szextAux(T, A, #if Clamp #then B #else B &Int 31 #fi) [owise]
  syntax Int ::= szextAux(Ty, Int, Int) [function, total]
  rule szextAux(_, _, 0) => 0
  rule szextAux(T, A, N) =>
       #if isS(T) andBool bitAt(A, N -Int 1) ==Int 1
       #then trunc((A &Int mask(N)) |Int (mask(32) &Int (~Int mask(N))), 32) #else A &Int mask(N) #fi
    requires N >Int 0 andBool N <=Int 32
  rule szextAux(_, A, _) => A [owise]

  rule <tk> exec(opc("szext", M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, szext("clamp" in M, T, rd(A, T, C), rd(B, ty("u", 32), C))) </regs>

  // ---------------------------------------------------------------- prmt
  // prmt.b32{.mode} d, a, b, c : select bytes from the 64-bit value {b, a}
  syntax Int ::= prmtByte(Int, Int) [function, total]               // 64-bit value, 4-bit selector
  rule prmtByte(V, S) =>
       #if (S &Int 8) ==Int 8
       #then #if bitAt(V, 8 *Int (S &Int 7) +Int 7) ==Int 1 #then 255 #else 0 #fi
       #else (V >>Int (8 *Int (S &Int 7))) &Int 255 #fi
  syntax Int ::= prmtSel(String, Int, Int) [function, total]        // mode, selector control c, result byte i -> source byte index
  // tables from the PTX ISA, as columns: result byte i takes source byte prmtSel(mode, c, i)
  rule prmtSel("f4e", C, I)  => (C &Int 3) +Int I
  rule prmtSel("b4e", C, I)  => ((C &Int 3) -Int I) modInt 8
  rule prmtSel("rc8", C, _)  => C &Int 3
  rule prmtSel("ecl", C, I)  => maxInt(C &Int 3, I)
  rule prmtSel("ecr", C, I)  => minInt(C &Int 3, I)
  rule prmtSel("rc16", C, I) => (C &Int 1) *Int 2 +Int (I %Int 2)
  rule prmtSel(_, _, _) => 0 [owise]

  syntax Int ::= prmt(String, Int, Int, Int) [function, total]      // mode, a, b, c
  rule prmt("", A, B, C) =>
       prmtByte(A |Int (B <<Int 32), C &Int 15)
       |Int (prmtByte(A |Int (B <<Int 32), (C >>Int 4) &Int 15) <<Int 8)
       |Int (prmtByte(A |Int (B <<Int 32), (C >>Int 8) &Int 15) <<Int 16)
       |Int (prmtByte(A |Int (B <<Int 32), (C >>Int 12) &Int 15) <<Int 24)
  rule prmt(Mode, A, B, C) =>
       prmtByte(A |Int (B <<Int 32), prmtSel(Mode, C, 0))
       |Int (prmtByte(A |Int (B <<Int 32), prmtSel(Mode, C, 1)) <<Int 8)
       |Int (prmtByte(A |Int (B <<Int 32), prmtSel(Mode, C, 2)) <<Int 16)
       |Int (prmtByte(A |Int (B <<Int 32), prmtSel(Mode, C, 3)) <<Int 24)
    requires Mode =/=String ""

  rule <tk> exec(opc("prmt", M, _), ops(D, A, B, Cn, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, prmt(pick(M, ListItem("f4e") ListItem("b4e") ListItem("rc8") ListItem("ecl")
                                         ListItem("ecr") ListItem("rc16"), ""),
                                 rd(A, ty("b", 32), C), rd(B, ty("b", 32), C), rd(Cn, ty("b", 32), C))) </regs>

  // ---------------------------------------------------------------- lop3
  // lop3.b32 d, a, b, c, immLut : d[i] = immLut[a[i]*4 + b[i]*2 + c[i]]
  // lop3.BoolOp.b32 d|p, a, b, c, immLut, q : additionally p = (d != 0) BoolOp q
  syntax Int ::= lop3(Int, Int, Int, Int) [function, total]
  rule lop3(A, B, C, L) => lop3Bit(A, B, C, L, 0, 0)
  syntax Int ::= lop3Bit(Int, Int, Int, Int, Int, Int) [function, total]     // ..., bit index, accumulator
  rule lop3Bit(A, B, C, L, I, ACC) =>
       lop3Bit(A, B, C, L, I +Int 1,
               ACC |Int (bitAt(L, bitAt(A, I) *Int 4 +Int bitAt(B, I) *Int 2 +Int bitAt(C, I)) <<Int I))
    requires I <Int 32
  rule lop3Bit(_, _, _, _, _, ACC) => ACC [owise]

  rule <tk> exec(opc("lop3", _, _), ops(D, A, B, Cn, L, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, lop3(rd(A, ty("b", 32), C), rd(B, ty("b", 32), C), rd(Cn, ty("b", 32), C),
                                 rd(L, ty("b", 32), C))) </regs>
    requires notBool isDual(D)
  rule <tk> exec(opc("lop3", M, _), ops(D:Reg | P:Reg, A, B, Cn, L, Q, .Operands), C) => .K ... </tk>
       <regs> R => wr(wr(R, D, lop3(rd(A, ty("b", 32), C), rd(B, ty("b", 32), C), rd(Cn, ty("b", 32), C), rd(L, ty("b", 32), C))),
                      P, bopB(lop3(rd(A, ty("b", 32), C), rd(B, ty("b", 32), C), rd(Cn, ty("b", 32), C), rd(L, ty("b", 32), C)) =/=Int 0,
                              M, rd(Q, ty("p", 1), C))) </regs>
  syntax Bool ::= isDual(Operand) [function, total]
  rule isDual(_:Reg | _:Reg) => true
  rule isDual(_) => false [owise]
  syntax Int ::= bopB(Bool, Set, Int) [function, total]
  rule bopB(X, M, Q) => b2i(X andBool Q =/=Int 0) requires "and" in M
  rule bopB(X, M, Q) => b2i(X orBool Q =/=Int 0)  requires "or" in M
  rule bopB(X, _, _) => b2i(X) [owise]

  // ---------------------------------------------------------------- shf
  // shf.l/r.mode.b32 d, a, b, c : funnel shift of the 64-bit value {b:a} (b high)
  syntax Int ::= shf(Bool, Bool, Int, Int, Int) [function, total]        // left?, clamp?, a, b, c
  rule shf(Left, Clamp, A, B, C) => shfAux(Left, A |Int (B <<Int 32), #if Clamp #then minInt(C, 32) #else C &Int 31 #fi)
  syntax Int ::= shfAux(Bool, Int, Int) [function, total]
  rule shfAux(true,  V, S) => (((V <<Int S) &Int 18446744073709551615) >>Int 32) &Int 4294967295
  rule shfAux(false, V, S) => (V >>Int S) &Int 4294967295

  rule <tk> exec(opc("shf", M, _), ops(D, A, B, Cn, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, shf("l" in M, "clamp" in M, rd(A, ty("b", 32), C), rd(B, ty("b", 32), C),
                                rd(Cn, ty("u", 32), C))) </regs>
endmodule

// =============================================================================
// PTX-EXEC-INTX: remaining integer arithmetic.
//
//   mul24 mad24{.sat}   sad   dp4a dp2a   packed 16x2: add sub min max neg abs
//   (.u16x2 .s16x2, with .relu on min/max)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-INTX
  imports PTX-EXEC-CORE

  // ---------------------------------------------------------------- mul24 / mad24
  // 24-bit operands (sign- or zero-extended per type), 48-bit product; .lo = low 32 bits, .hi = bits 47:16
  syntax Int ::= v24(Ty, Int) [function, total]
  rule v24(T, A) => #if isS(T) #then sext(A, 24) #else A &Int 16777215 #fi
  syntax Int ::= mul24(Set, Ty, Int, Int) [function, total]
  rule mul24(M, T, A, B) => trunc((v24(T, A) *Int v24(T, B)) >>Int 16, 32) requires "hi" in M
  rule mul24(_, T, A, B) => trunc(v24(T, A) *Int v24(T, B), 32) [owise]
  syntax Int ::= mad24(Set, Ty, Int, Int, Int) [function, total]
  rule mad24(M, T, A, B, C) => trunc(clamp(((v24(T, A) *Int v24(T, B)) >>Int 16) +Int ival(C, T), T), 32)
    requires "hi" in M andBool "sat" in M
  rule mad24(M, T, A, B, C) => trunc(mul24(M, T, A, B) +Int C, 32) [owise]

  rule <tk> exec(opc("mul24", M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, mul24(M, T, rd(A, T, C), rd(B, T, C))) </regs>
  rule <tk> exec(opc("mad24", M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, mad24(M, T, rd(A, T, C), rd(B, T, C), rd(X, T, C))) </regs>

  // ---------------------------------------------------------------- sad
  // d = c + |a - b|
  syntax Int ::= sad(Ty, Int, Int, Int) [function, total]
  rule sad(T, A, B, C) => trunc(C +Int absInt(ival(A, T) -Int ival(B, T)), wid(T))
  rule <tk> exec(opc("sad", _, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, sad(T, rd(A, T, C), rd(B, T, C), rd(X, T, C))) </regs>

  // ---------------------------------------------------------------- dp4a / dp2a
  // dp4a.atype.btype d, a, b, c : c + sum of four byte products
  // dp2a.mode.atype.btype d, a, b, c : c + two 16-bit x 8-bit products (b bytes 0,1 for .lo; 2,3 for .hi)
  syntax Int ::= field(Ty, Int, Int, Int) [function, total]          // type, value, bit offset, width -> mathematical value
  rule field(T, V, O, W) => #if isS(T) #then sext(V >>Int O, W) #else (V >>Int O) &Int mask(W) #fi
  syntax Int ::= dp4a(Ty, Ty, Int, Int, Int) [function, total]
  rule dp4a(TA, TB, A, B, C) => trunc(C +Int field(TA, A, 0, 8) *Int field(TB, B, 0, 8)
                                          +Int field(TA, A, 8, 8) *Int field(TB, B, 8, 8)
                                          +Int field(TA, A, 16, 8) *Int field(TB, B, 16, 8)
                                          +Int field(TA, A, 24, 8) *Int field(TB, B, 24, 8), 32)
  syntax Int ::= dp2a(Bool, Ty, Ty, Int, Int, Int) [function, total]
  rule dp2a(Hi, TA, TB, A, B, C) => trunc(C +Int field(TA, A, 0, 16) *Int field(TB, B, #if Hi #then 16 #else 0 #fi, 8)
                                           +Int field(TA, A, 16, 16) *Int field(TB, B, #if Hi #then 24 #else 8 #fi, 8), 32)
  rule <tk> exec(opc("dp4a", _, ListItem(TA) ListItem(TB)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, dp4a(TA, TB, rd(A, TA, C), rd(B, TB, C), rd(X, ty("u", 32), C))) </regs>
  rule <tk> exec(opc("dp2a", M, ListItem(TA) ListItem(TB)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, dp2a("hi" in M, TA, TB, rd(A, TA, C), rd(B, TB, C), rd(X, ty("u", 32), C))) </regs>

  // ---------------------------------------------------------------- packed 16x2
  syntax Ty ::= laneTy(Ty) [function, total]
  rule laneTy(ty("s16x2", _)) => ty("s", 16)
  rule laneTy(ty("u16x2", _)) => ty("u", 16)
  rule laneTy(T) => T [owise]
  syntax Int ::= lane16(Int, Int) [function, total]
  rule lane16(V, I) => (V >>Int (16 *Int I)) &Int 65535
  syntax Int ::= pack16(Int, Int) [function, total]
  rule pack16(L0, L1) => (L0 &Int 65535) |Int ((L1 &Int 65535) <<Int 16)

  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(bin(OP, M, laneTy(T), lane16(rd(A, T, C), 0), lane16(rd(B, T, C), 0)),
                                   bin(OP, M, laneTy(T), lane16(rd(A, T, C), 1), lane16(rd(B, T, C), 1)))) </regs>
    requires isPacked(T) andBool notBool isF(T) andBool OP in (SetItem("add") SetItem("sub") SetItem("min") SetItem("max"))
     andBool (T ==K ty("s16x2", 32) orBool T ==K ty("u16x2", 32))
  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(un(OP, M, laneTy(T), lane16(rd(A, T, C), 0)),
                                   un(OP, M, laneTy(T), lane16(rd(A, T, C), 1)))) </regs>
    requires OP in (SetItem("neg") SetItem("abs")) andBool (T ==K ty("s16x2", 32) orBool T ==K ty("u16x2", 32))
endmodule

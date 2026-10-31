// =============================================================================
// PTX-EXEC-INT: extended-precision integer arithmetic.
//
//   add.cc  addc  addc.cc      sub.cc  subc  subc.cc
//   mad{.lo,.hi}.cc  madc{.lo,.hi}  madc{.lo,.hi}.cc        (u32 s32 u64 s64)
//
// Each thread has a one-bit carry flag CC.CF, kept in its register file under
// the reserved key "%%cc.cf".  Instructions without `.cc` leave it untouched.
// Carry-out of an addition is the unsigned carry out of the type's width.
// Subtraction follows the ISA text (PTX ISA 9.7.2.3, 9.7.2.4): sub.cc writes
// the borrow-out into CC.CF, and subc computes d = a - (b + CC.CF) with the
// flag as borrow-in, so CC.CF is 1 exactly when the subtraction borrows (as
// the Sail model, sail/ptx/ptx_int.sail, and tests/sub-borrow.ptx).  (NVIDIA
// hardware keeps the opposite polarity, which ZLUDA's subc_cc_s32 recorded;
// the expected value of tests/zluda/subc_cc_s32.ptx is the ISA's.)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-INT
  imports PTX-EXEC-CORE

  syntax String ::= "CCKEY" [macro]
  rule CCKEY => "%%cc.cf"

  // (result, carry/borrow out)
  syntax List ::= ccOp(String, Set, Ty, Int, Int, Int, Int) [function, total]   // op, mods, type, cin, a, b, c
  rule ccOp(OP, _, T, CIN, A, B, _) => ccSum(trunc(A, wid(T)) +Int trunc(B, wid(T)) +Int CIN, wid(T))
    requires OP ==String "add" orBool OP ==String "addc"
  rule ccOp("sub", _, T, _, A, B, _) =>
       ccDiff(trunc(A, wid(T)) -Int trunc(B, wid(T)), wid(T))
  rule ccOp("subc", _, T, CIN, A, B, _) =>
       ccDiff(trunc(A, wid(T)) -Int trunc(B, wid(T)) -Int CIN, wid(T))
  rule ccOp(OP, M, T, CIN, A, B, C) =>
       ccSum(madHalf(M, T, A, B) +Int trunc(C, wid(T)) +Int CIN, wid(T))
    requires OP ==String "mad" orBool OP ==String "madc"
  rule ccOp(_, _, _, _, _, _, _) => ListItem(0) ListItem(0) [owise]

  syntax List ::= ccSum(Int, Int) [function, total]
  rule ccSum(S, W) => ListItem(trunc(S, W)) ListItem(S >>Int W)
  // (result, borrow out): the difference modulo 2^W, and 1 when it is negative
  syntax List ::= ccDiff(Int, Int) [function, total]
  rule ccDiff(S, W) => ListItem(trunc(S, W)) ListItem(#if S <Int 0 #then 1 #else 0 #fi)
  syntax Int ::= madHalf(Set, Ty, Int, Int) [function, total]     // .lo / .hi half of the full product
  rule madHalf(M, T, A, B) => trunc((ival(A, T) *Int ival(B, T)) >>Int wid(T), wid(T)) requires "hi" in M
  rule madHalf(_, T, A, B) => trunc(ival(A, T) *Int ival(B, T), wid(T)) [owise]

  syntax Map ::= ccWrite(Map, Operand, List, Bool) [function, total]
  rule ccWrite(R, D, ListItem(V:Int) ListItem(CF:Int), true)  => wr(R, D, V)[CCKEY <- CF]
  rule ccWrite(R, D, ListItem(V:Int) ListItem(_), false)      => wr(R, D, V)
  rule ccWrite(R, _, _, _) => R [owise]

  syntax Bool ::= isCCOp(String, Set) [function, total]
  rule isCCOp(OP, M) => (OP in (SetItem("add") SetItem("sub") SetItem("mad"))
                         andBool "cc" in M)
                        orBool OP in (SetItem("addc") SetItem("subc") SetItem("madc"))
  // carry-in comes from CC.CF only for the "...c" forms
  syntax Int ::= cinOf(String, Map) [function, total]
  rule cinOf(OP, R) => {R[CCKEY] orDefault 0}:>Int
    requires OP in (SetItem("addc") SetItem("subc") SetItem("madc"))
  rule cinOf(_, _) => 0 [owise]

  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => ccWrite(R, D, ccOp(OP, M, T, cinOf(OP, R), rd(A, T, C), rd(B, T, C), 0), "cc" in M) </regs>
    requires isCCOp(OP, M) andBool OP =/=String "mad" andBool OP =/=String "madc"
  rule <tk> #gx(opc(OP, M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => ccWrite(R, D, ccOp(OP, M, T, cinOf(OP, R), rd(A, T, C), rd(B, T, C), rd(X, T, C)), "cc" in M) </regs>
    requires isCCOp(OP, M) andBool (OP ==String "mad" orBool OP ==String "madc")
endmodule

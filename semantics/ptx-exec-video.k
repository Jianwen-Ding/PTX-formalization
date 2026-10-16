// =============================================================================
// PTX-EXEC-VIDEO: video instructions (best effort; legacy but still emitted for
// the CUDA SIMD intrinsics such as __vadd2, __vabsdiffu4, __vcmpeq2).
//
//   scalar: vadd vsub vabsdiff vmin vmax vshl vshr vset
//           op.dtype.atype.btype{.sat}{.secop} d, a{.asel}, b{.bsel}[, c]
//           (.asel/.bsel: .b0-.b3, .h0, .h1; secop: .add .min .max;
//            vshl/vshr take .clamp/.wrap; vset takes a comparison)
//   SIMD:   vadd2 vsub2 vavrg2 vabsdiff2 vmin2 vmax2 vset2 and the 4-way
//           versions, default selectors, with .sat and .add
//
// Not modelled: vmad, .mask destinations, permuting SIMD selectors.  Only the
// vshr/.clamp/.add path is checked against hardware (ZLUDA's vshr_clamp); the
// rest follows the ISA text.
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-VIDEO
  imports PTX-EXEC-CORE

  // ---------------------------------------------------------------- operand parts
  syntax String ::= opName(Operand) [function, total]
  rule opName(R:Reg)  => Reg2String(R)
  rule opName(N:Name) => Name2String(N)
  rule opName(_) => "" [owise]

  // selector suffix of a register name: ".b0".. ".b3", ".h0", ".h1" (or "")
  syntax String ::= selOf(String) [function, total]
  rule selOf(S) => substrString(S, lengthString(S) -Int 3, lengthString(S))
    requires lengthString(S) >Int 3
     andBool substrString(S, lengthString(S) -Int 3, lengthString(S)) in
             (SetItem(".b0") SetItem(".b1") SetItem(".b2") SetItem(".b3") SetItem(".h0") SetItem(".h1"))
  rule selOf(_) => "" [owise]

  // mathematical value of operand `O` read as type T with its (optional) part selector
  syntax Int ::= vrd(Operand, Ty, Ctx) [function, total]
  rule vrd(O, T, C) => vpart(selOf(opName(O)), T, rreg(substrString(opName(O), 0, lengthString(opName(O)) -Int 3), C))
    requires selOf(opName(O)) =/=String ""
  rule vrd(O, T, C) => vpart("", T, rd(O, ty("u", 32), C)) [owise]
  syntax Int ::= vpart(String, Ty, Int) [function, total]
  rule vpart(".b0", T, V) => ext(T, V, 0, 8)    rule vpart(".b1", T, V) => ext(T, V, 8, 8)
  rule vpart(".b2", T, V) => ext(T, V, 16, 8)   rule vpart(".b3", T, V) => ext(T, V, 24, 8)
  rule vpart(".h0", T, V) => ext(T, V, 0, 16)   rule vpart(".h1", T, V) => ext(T, V, 16, 16)
  rule vpart("", T, V) => ext(T, V, 0, minInt(wid(T), 32))
  rule vpart(_, T, V) => ext(T, V, 0, 32) [owise]
  syntax Int ::= ext(Ty, Int, Int, Int) [function, total]
  rule ext(T, V, O, W) => #if isS(T) #then sext(V >>Int O, W) #else (V >>Int O) &Int mask(W) #fi

  // ---------------------------------------------------------------- scalar
  syntax String ::= secOp(Set) [function, total]
  rule secOp(M) => pick(M, ListItem("add") ListItem("min") ListItem("max"), "")
  syntax Int ::= satTo(Bool, Ty, Int) [function, total]
  rule satTo(true, T, V) => clamp(V, T)
  rule satTo(false, _, V) => V
  syntax Int ::= secApply(String, Ty, Int, Int) [function, total]            // op, dtype, tmp, c (raw)
  rule secApply("add", T, V, C) => V +Int ival(C, T)
  rule secApply("min", T, V, C) => minInt(V, ival(C, T))
  rule secApply("max", T, V, C) => maxInt(V, ival(C, T))
  rule secApply(_, _, V, _) => V [owise]
  syntax Int ::= vfinish(Set, Ty, Int, Int) [function, total]                // mods, dtype, primary result, c
  rule vfinish(M, DT, V, C) => trunc(secApply(secOp(M), DT, satTo("sat" in M, DT, V), C), wid(DT))

  syntax Int ::= vcore(String, Int, Int) [function, total]
  rule vcore("vadd", A, B) => A +Int B
  rule vcore("vsub", A, B) => A -Int B
  rule vcore("vabsdiff", A, B) => absInt(A -Int B)
  rule vcore("vmin", A, B) => minInt(A, B)
  rule vcore("vmax", A, B) => maxInt(A, B)
  rule vcore(_, A, _) => A [owise]

  rule <tk> exec(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vcore(OP, vrd(A, AT, C), vrd(B, BT, C)), 0)) </regs>
    requires OP in (SetItem("vadd") SetItem("vsub") SetItem("vabsdiff") SetItem("vmin") SetItem("vmax"))
  rule <tk> exec(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vcore(OP, vrd(A, AT, C), vrd(B, BT, C)), rd(X, DT, C))) </regs>
    requires OP in (SetItem("vadd") SetItem("vsub") SetItem("vabsdiff") SetItem("vmin") SetItem("vmax"))

  // vshl / vshr : shift amount from b's low bits, clamped to 32 or wrapped mod 32
  syntax Int ::= vshift(Bool, Bool, Int, Int) [function, total]              // left?, clamp?, a, b
  rule vshift(Left, Clamp, A, B) =>
       #if Left #then A <<Int #if Clamp #then minInt(B &Int 255, 32) #else B &Int 31 #fi
       #else A >>Int #if Clamp #then minInt(B &Int 255, 32) #else B &Int 31 #fi #fi
  rule <tk> exec(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vshift(OP ==String "vshl", notBool "wrap" in M, vrd(A, AT, C), vrd(B, BT, C)), 0)) </regs>
    requires OP in (SetItem("vshl") SetItem("vshr"))
  rule <tk> exec(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vshift(OP ==String "vshl", notBool "wrap" in M, vrd(A, AT, C), vrd(B, BT, C)), rd(X, DT, C))) </regs>
    requires OP in (SetItem("vshl") SetItem("vshr"))

  // vset.atype.btype.cmp[.secop] d, a, b[, c]
  rule <tk> exec(opc("vset", M, ListItem(AT) ListItem(BT)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vsetRes(M, vrd(A, AT, C), vrd(B, BT, C), 0)) </regs>
  rule <tk> exec(opc("vset", M, ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vsetRes(M, vrd(A, AT, C), vrd(B, BT, C), rd(X, ty("u", 32), C))) </regs>
  syntax Int ::= vsetRes(Set, Int, Int, Int) [function, total]
  rule vsetRes(M, A, B, C) => trunc(secApply(secOp(M), ty("u", 32), cmpV(cmpOp(M), A, B), C), 32)
  syntax Int ::= cmpV(String, Int, Int) [function, total]
  rule cmpV(OP, A, B) => b2i(cmpI(OP, A, B, A, B))

  // ---------------------------------------------------------------- SIMD (2 x 16, 4 x 8)
  syntax Bool ::= isSimd(String) [function, total]
  rule isSimd(OP) => OP in (SetItem("vadd2") SetItem("vsub2") SetItem("vavrg2") SetItem("vabsdiff2") SetItem("vmin2")
                            SetItem("vmax2") SetItem("vset2") SetItem("vadd4") SetItem("vsub4") SetItem("vavrg4")
                            SetItem("vabsdiff4") SetItem("vmin4") SetItem("vmax4") SetItem("vset4"))
  syntax Int ::= lanes(String) [function, total]
  rule lanes(OP) => #if substrString(OP, lengthString(OP) -Int 1, lengthString(OP)) ==String "2" #then 2 #else 4 #fi
  syntax Int ::= simdOp(String, Int, Int, Set) [function, total]             // op (without count), ta, tb, mods
  rule simdOp("vadd", A, B, _) => A +Int B
  rule simdOp("vsub", A, B, _) => A -Int B
  rule simdOp("vavrg", A, B, _) => (A +Int B +Int 1) >>Int 1
  rule simdOp("vabsdiff", A, B, _) => absInt(A -Int B)
  rule simdOp("vmin", A, B, _) => minInt(A, B)
  rule simdOp("vmax", A, B, _) => maxInt(A, B)
  rule simdOp("vset", A, B, M) => cmpV(cmpOp(M), A, B)
  rule simdOp(_, A, _, _) => A [owise]
  // lane L of width W: value of dtype-clamped result
  syntax Int ::= simdLane(String, Set, Ty, Ty, Ty, Int, Int, Int, Int) [function, total]   // op, mods, dt, at, bt, a, b, lane, width
  rule simdLane(OP, M, DT, AT, BT, A, B, L, W) =>
       satLane("sat" in M andBool OP =/=String "vset", DT, W,
               simdOp(OP, ext(AT, A, L *Int W, W), ext(BT, B, L *Int W, W), M))
  syntax Int ::= satLane(Bool, Ty, Int, Int) [function, total]
  rule satLane(true, DT, W, V) => #if isS(DT) #then minInt(maxInt(V, 0 -Int (1 <<Int (W -Int 1))), (1 <<Int (W -Int 1)) -Int 1)
                                  #else minInt(maxInt(V, 0), mask(W)) #fi
  rule satLane(false, _, _, V) => V
  syntax Int ::= simdAll(String, Set, Ty, Ty, Ty, Int, Int, Int, Int, Bool) [function, total]
  // accumulate lanes: either packed (no .add) or summed (.add)
  rule simdAll(OP, M, DT, AT, BT, A, B, L, W, Add) =>
       #if L <Int 0 #then 0
       #else #if Add #then simdLane(OP, M, DT, AT, BT, A, B, L, W) +Int simdAll(OP, M, DT, AT, BT, A, B, L -Int 1, W, Add)
             #else ((simdLane(OP, M, DT, AT, BT, A, B, L, W) &Int mask(W)) <<Int (L *Int W))
                   |Int simdAll(OP, M, DT, AT, BT, A, B, L -Int 1, W, Add) #fi #fi

  rule <tk> exec(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(simdAll(substrString(OP, 0, lengthString(OP) -Int 1), M, DT, AT, BT,
                                          rd(A, ty("u", 32), C), rd(B, ty("u", 32), C),
                                          lanes(OP) -Int 1, 32 /Int lanes(OP), "add" in M)
                                   +Int #if "add" in M #then rd(X, DT, C) #else 0 #fi, 32)) </regs>
    requires isSimd(OP)

  // vset2 / vset4 have no destination type suffix (d is .u32)
  rule <tk> exec(opc(OP, M, ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(simdAll(substrString(OP, 0, lengthString(OP) -Int 1), M, ty("u", 32), AT, BT,
                                          rd(A, ty("u", 32), C), rd(B, ty("u", 32), C),
                                          lanes(OP) -Int 1, 32 /Int lanes(OP), "add" in M)
                                   +Int #if "add" in M #then rd(X, ty("u", 32), C) #else 0 #fi, 32)) </regs>
    requires isSimd(OP)
endmodule

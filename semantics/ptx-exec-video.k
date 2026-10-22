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
// Not modelled: .mask destinations, permuting SIMD selectors.  Only the
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

  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vcore(OP, vrd(A, AT, C), vrd(B, BT, C)), 0)) </regs>
    requires OP in (SetItem("vadd") SetItem("vsub") SetItem("vabsdiff") SetItem("vmin") SetItem("vmax"))
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vcore(OP, vrd(A, AT, C), vrd(B, BT, C)), rd(X, DT, C))) </regs>
    requires OP in (SetItem("vadd") SetItem("vsub") SetItem("vabsdiff") SetItem("vmin") SetItem("vmax")) andBool selSfx(opName(D)) ==String ""

  // vshl / vshr : shift amount from b's low bits, clamped to 32 or wrapped mod 32
  syntax Int ::= vshift(Bool, Bool, Int, Int) [function, total]              // left?, clamp?, a, b
  rule vshift(Left, Clamp, A, B) =>
       #if Left #then A <<Int #if Clamp #then minInt(B &Int 4294967295, 32) #else B &Int 31 #fi
       #else A >>Int #if Clamp #then minInt(B &Int 4294967295, 32) #else B &Int 31 #fi #fi
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vshift(OP ==String "vshl", notBool "wrap" in M, vrd(A, AT, C), vrd(B, BT, C)), 0)) </regs>
    requires OP in (SetItem("vshl") SetItem("vshr"))
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vfinish(M, DT, vshift(OP ==String "vshl", notBool "wrap" in M, vrd(A, AT, C), vrd(B, BT, C)), rd(X, DT, C))) </regs>
    requires OP in (SetItem("vshl") SetItem("vshr")) andBool selSfx(opName(D)) ==String ""

  // vset.atype.btype.cmp[.secop] d, a, b[, c]
  rule <tk> exec(opc("vset", M, ListItem(AT) ListItem(BT)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vsetRes(M, vrd(A, AT, C), vrd(B, BT, C), 0)) </regs>
  rule <tk> exec(opc("vset", M, ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vsetRes(M, vrd(A, AT, C), vrd(B, BT, C), rd(X, ty("u", 32), C))) </regs>
    requires selSfx(opName(D)) ==String ""
  // ISA vset: d = optMerge(dsel, tmp, c) when d carries a selector
  rule <tk> exec(opc("vset", M, ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => R[baseStr(opName(D)) <- vmerge(ty("u", 32), false, selSfx(opName(D)), cmpV(cmpOp(M), vrd(A, AT, C), vrd(B, BT, C)),
                                                   rd(X, ty("u", 32), C))] </regs>
    requires selSfx(opName(D)) =/=String ""
  syntax Int ::= vsetRes(Set, Int, Int, Int) [function, total]
  rule vsetRes(M, A, B, C) => trunc(secApply(secOp(M), ty("u", 32), cmpV(cmpOp(M), A, B), C), 32)
  syntax Int ::= cmpV(String, Int, Int) [function, total]
  rule cmpV(OP, A, B) => b2i(cmpI(OP, A, B, A, B))

  // ---------------------------------------------------------------- selectors and masks
  // Operand names may carry a trailing selector: r2.h10, r3.b7654 (SIMD source/destination) or r1.b2 / r1.h0 (scalar).
  syntax Bool ::= isDg(String) [function, total] | allDig(String) [function, total]
  rule isDg(C) => C >=String "0" andBool C <=String "9"
  rule allDig(S) => true requires lengthString(S) ==Int 0
  rule allDig(S) => isDg(substrString(S, 0, 1)) andBool allDig(substrString(S, 1, lengthString(S))) requires lengthString(S) >Int 0
  syntax String ::= selSfx(String) [function, total] | selSfxAt(String, Int) [function, total]
  rule selSfx(S) => selSfxAt(S, rfindString(S, ".", lengthString(S)))
  rule selSfxAt(S, I) => substrString(S, I +Int 1, lengthString(S))
    requires I >=Int 0 andBool lengthString(S) -Int I >Int 2
     andBool (substrString(S, I +Int 1, I +Int 2) ==String "h" orBool substrString(S, I +Int 1, I +Int 2) ==String "b")
     andBool allDig(substrString(S, I +Int 2, lengthString(S)))
  rule selSfxAt(_, _) => "" [owise]
  syntax String ::= baseStr(String) [function, total]
  rule baseStr(S) => S requires selSfx(S) ==String ""
  rule baseStr(S) => substrString(S, 0, lengthString(S) -Int lengthString(selSfx(S)) -Int 1) requires selSfx(S) =/=String ""
  syntax Int ::= rawV(Operand, Ctx) [function, total]       // the whole 32-bit value of a register operand, ignoring a selector suffix
  rule rawV(O, C) => rreg(baseStr(opName(O)), C) requires selSfx(opName(O)) =/=String ""
  rule rawV(O, C) => rd(O, ty("u", 32), C) requires selSfx(opName(O)) ==String ""
  syntax String ::= selOr(Operand, String) [function, total]
  rule selOr(O, Dflt) => selSfx(opName(O)) requires selSfx(opName(O)) =/=String ""
  rule selOr(O, Dflt) => Dflt requires selSfx(opName(O)) ==String ""

  // ---------------------------------------------------------------- scalar data merge   vop.dtype.atype.btype{.sat} d.dsel, a, b, c
  syntax Bool ::= isScalarVop(String) [function, total]
  rule isScalarVop(OP) => OP in (SetItem("vadd") SetItem("vsub") SetItem("vabsdiff") SetItem("vmin") SetItem("vmax") SetItem("vshl") SetItem("vshr"))
  syntax Int ::= vmergeW(String) [function, total] | vmergePos(String) [function, total]
  rule vmergeW(Sel) => #if substrString(Sel, 0, 1) ==String "h" #then 16 #else 8 #fi
  rule vmergePos(Sel) => String2Int(substrString(Sel, 1, 2)) *Int vmergeW(Sel)
  syntax Int ::= vmerge(Ty, Bool, String, Int, Int) [function, total]     // dtype, sat, dsel, tmp, c
  rule vmerge(DT, Sat, Sel, V, Cr) =>
       ((Cr &Int mask(32)) &Int (~Int (mask(vmergeW(Sel)) <<Int vmergePos(Sel))))
       |Int ((satLane(Sat, DT, vmergeW(Sel), V) &Int mask(vmergeW(Sel))) <<Int vmergePos(Sel))
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => R[baseStr(opName(D)) <- vmerge(DT, "sat" in M, selSfx(opName(D)),
                       #if OP ==String "vshl" orBool OP ==String "vshr"
                       #then vshift(OP ==String "vshl", notBool "wrap" in M, vrd(A, AT, C), vrd(B, BT, C))
                       #else vcore(OP, vrd(A, AT, C), vrd(B, BT, C)) #fi,
                       rd(X, ty("u", 32), C))] </regs>
    requires isScalarVop(OP) andBool selSfx(opName(D)) =/=String ""

  // ---------------------------------------------------------------- SIMD (2 x 16, 4 x 8), ISA semantics
  syntax Bool ::= isSimd(String) [function, total]
  rule isSimd(OP) => OP in (SetItem("vadd2") SetItem("vsub2") SetItem("vavrg2") SetItem("vabsdiff2") SetItem("vmin2")
                            SetItem("vmax2") SetItem("vset2") SetItem("vadd4") SetItem("vsub4") SetItem("vavrg4")
                            SetItem("vabsdiff4") SetItem("vmin4") SetItem("vmax4") SetItem("vset4"))
  syntax Int ::= lanes(String) [function, total]
  rule lanes(OP) => #if substrString(OP, lengthString(OP) -Int 1, lengthString(OP)) ==String "2" #then 2 #else 4 #fi
  syntax String ::= defSel(String, Int) [function, total]       // which ("a", "b", "m"), lanes
  rule defSel("a", 2) => "h10"
  rule defSel("b", 2) => "h32"
  rule defSel("m", 2) => "h10"
  rule defSel("a", 4) => "b3210"
  rule defSel("b", 4) => "b7654"
  rule defSel("m", 4) => "b3210"
  rule defSel(_, _) => "" [owise]
  syntax Int ::= simdOp(String, Int, Int, Set) [function, total]             // op (without count), ta, tb, mods
  rule simdOp("vadd", A, B, _) => A +Int B
  rule simdOp("vsub", A, B, _) => A -Int B
  rule simdOp("vavrg", A, B, _) => #if A +Int B >=Int 0 #then (A +Int B +Int 1) >>Int 1 #else (A +Int B) >>Int 1 #fi
  rule simdOp("vabsdiff", A, B, _) => absInt(A -Int B)
  rule simdOp("vmin", A, B, _) => minInt(A, B)
  rule simdOp("vmax", A, B, _) => maxInt(A, B)
  rule simdOp("vset", A, B, M) => cmpV(cmpOp(M), A, B)
  rule simdOp(_, A, _, _) => A [owise]
  syntax Int ::= satLane(Bool, Ty, Int, Int) [function, total]
  rule satLane(true, DT, W, V) => #if isS(DT) #then minInt(maxInt(V, 0 -Int (1 <<Int (W -Int 1))), (1 <<Int (W -Int 1)) -Int 1)
                                  #else minInt(maxInt(V, 0), mask(W)) #fi
  rule satLane(false, _, _, V) => V
  // element number K of the 2n-element concatenation (a, b), sign- or zero-extended according to type T
  syntax Int ::= elemOf(Ty, Int, Int, Int, Int) [function, total]            // type, a, b, k, lane width
  rule elemOf(T, A, B, K, W) => #if K <Int 32 /Int W #then ext(T, A, K *Int W, W) #else ext(T, B, (K -Int 32 /Int W) *Int W, W) #fi
  syntax Int ::= selDigit(String, Int, Int) [function, total]                // selector string, lane, lanes
  rule selDigit(S, L, N) => String2Int(substrString(S, 1 +Int (N -Int 1 -Int L), 2 +Int (N -Int 1 -Int L))) requires lengthString(S) >=Int 1 +Int N
  rule selDigit(_, L, _) => L [owise]
  syntax Bool ::= maskHas(String, Int) [function, total]
  rule maskHas(Mk, L) => findString(substrString(Mk, 1, lengthString(Mk)), Int2String(L), 0) >=Int 0
  syntax Int ::= tLane(String, Set, Ty, Ty, Ty, Int, Int, String, String, Int, Int) [function, total]
              // op, mods, dtype, atype, btype, a, b, asel, bsel, lane, lanes
  rule tLane(OP, M, DT, AT, BT, A, B, AS, BS, L, N) =>
       satLane("sat" in M andBool OP =/=String "vset", DT, 32 /Int N,
               simdOp(OP, elemOf(AT, A, B, selDigit(AS, L, N), 32 /Int N), elemOf(BT, A, B, selDigit(BS, L, N), 32 /Int N), M))
  syntax Int ::= simdAcc(String, Set, Ty, Ty, Ty, Int, Int, String, String, String, Int, Int) [function, total]
  rule simdAcc(OP, M, DT, AT, BT, A, B, AS, BS, Mk, L, N) =>
       #if L <Int 0 #then 0
       #else (#if maskHas(Mk, L) #then tLane(OP, M, DT, AT, BT, A, B, AS, BS, L, N) #else 0 #fi)
             +Int simdAcc(OP, M, DT, AT, BT, A, B, AS, BS, Mk, L -Int 1, N) #fi
  syntax Int ::= simdMrg(String, Set, Ty, Ty, Ty, Int, Int, Int, String, String, String, Int, Int) [function, total]
  rule simdMrg(OP, M, DT, AT, BT, A, B, Cv, AS, BS, Mk, L, N) =>
       #if L <Int 0 #then 0
       #else (((#if maskHas(Mk, L) #then tLane(OP, M, DT, AT, BT, A, B, AS, BS, L, N) &Int mask(32 /Int N)
               #else (Cv >>Int (L *Int (32 /Int N))) &Int mask(32 /Int N) #fi) <<Int (L *Int (32 /Int N))))
             |Int simdMrg(OP, M, DT, AT, BT, A, B, Cv, AS, BS, Mk, L -Int 1, N) #fi
  syntax Int ::= simdRes(String, Set, Ty, Ty, Ty, Int, Int, Int, String, String, String) [function, total]
  rule simdRes(OP, M, DT, AT, BT, A, B, Cv, AS, BS, Mk) =>
       #if "add" in M
       #then trunc(Cv +Int simdAcc(substrString(OP, 0, lengthString(OP) -Int 1), M, DT, AT, BT, A, B, AS, BS, Mk, lanes(OP) -Int 1, lanes(OP)), 32)
       #else simdMrg(substrString(OP, 0, lengthString(OP) -Int 1), M, DT, AT, BT, A, B, Cv, AS, BS, Mk, lanes(OP) -Int 1, lanes(OP)) #fi
  rule <tk> #gx(opc(OP, M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => R[baseStr(opName(D)) <- simdRes(OP, M, DT, AT, BT, rawV(A, C), rawV(B, C), rd(X, ty("u", 32), C),
                                                   selOr(A, defSel("a", lanes(OP))), selOr(B, defSel("b", lanes(OP))),
                                                   selOr(D, defSel("m", lanes(OP))))] </regs>
    requires isSimd(OP) andBool OP =/=String "vset2" andBool OP =/=String "vset4"
  // vset2 / vset4 have no destination type suffix (d is .u32)
  rule <tk> #gx(opc(OP, M, ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => R[baseStr(opName(D)) <- simdRes(OP, M, ty("u", 32), AT, BT, rawV(A, C), rawV(B, C), rd(X, ty("u", 32), C),
                                                   selOr(A, defSel("a", lanes(OP))), selOr(B, defSel("b", lanes(OP))),
                                                   selOr(D, defSel("m", lanes(OP))))] </regs>
    requires OP ==String "vset2" orBool OP ==String "vset4"

  // ---------------------------------------------------------------- vmad
  //   vmad.dtype.atype.btype{.po}{.sat}{.scale} d, {-}a{.asel}, {-}b{.bsel}, {-}c     (see the ISA semantics)
  syntax Bool ::= isNegOp(Operand) [function, total]
  rule isNegOp(- _:Reg) => true
  rule isNegOp(- _:Name) => true
  rule isNegOp(_) => false [owise]
  syntax Operand ::= unNeg(Operand) [function, total]
  rule unNeg(- R:Reg) => R
  rule unNeg(- N:Name) => N
  rule unNeg(O) => O [owise]
  rule <tk> exec(opc("vmad", M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, vmadRes(M, AT, BT, vrd(unNeg(A), AT, C), vrd(unNeg(B), BT, C), rd(unNeg(X), ty("u", 32), C),
                                    isNegOp(A), isNegOp(B), isNegOp(X))) </regs>
  syntax Int ::= vmadRes(Set, Ty, Ty, Int, Int, Int, Bool, Bool, Bool) [function, total]
  rule vmadRes(M, AT, BT, TA, TB, CR, NA, NB, NC) =>
       vmadFin(M, vmadSigned(AT, BT, NA, NB, NC),
               #if "po" in M #then TA *Int TB
               #else #if NA =/=Bool NB #then ~Int (TA *Int TB) #else TA *Int TB #fi #fi,
               #if "po" in M orBool NA =/=Bool NB #then CR #else #if NC #then ~Int CR #else CR #fi #fi,
               #if "po" in M orBool NA =/=Bool NB orBool NC #then 1 #else 0 #fi)
  syntax Bool ::= vmadSigned(Ty, Ty, Bool, Bool, Bool) [function, total]
  rule vmadSigned(AT, BT, NA, NB, NC) => isS(AT) orBool isS(BT) orBool (NA =/=Bool NB) orBool NC
  syntax Int ::= vmadFin(Set, Bool, Int, Int, Int) [function, total]          // mods, signed final?, product, c (32-bit pattern), lsb
  rule vmadFin(M, SF, P, CR, L) => vmadSat(M, SF,
       vmadScale(M, P +Int #if SF #then sext(CR &Int mask(32), 32) #else CR &Int mask(32) #fi +Int L))
  syntax Int ::= vmadScale(Set, Int) [function, total]
  rule vmadScale(M, V) => #if "shr7" in M #then V >>Int 7 #else #if "shr15" in M #then V >>Int 15 #else V #fi #fi
  syntax Int ::= vmadSat(Set, Bool, Int) [function, total]
  rule vmadSat(M, SF, V) => #if "sat" in M
       #then #if SF #then minInt(maxInt(V, 0 -Int (1 <<Int 31)), (1 <<Int 31) -Int 1) #else minInt(maxInt(V, 0), mask(32)) #fi
       #else trunc(V, 32) #fi
endmodule

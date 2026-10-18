// =============================================================================
// PTX-EXEC-MINI: conversions to and from the 8-, 6- and 4-bit floating point
// formats and the e8m0 scale format.
//
//   fp8 : .e4m3 (no inf; NaN = S.1111.111; max 448)      .e5m2 (IEEE-like; max 57344)
//   fp6 : .e2m3 (max 7.5)   .e3m2 (max 28)                no inf, no NaN
//   fp4 : .e2m1 (max 6)                                   no inf, no NaN
//   e8m0: .ue8m0 unsigned exponent-only, value 2^(e-127), 0xFF = NaN
//
// Packed containers: "x2" holds two elements (fp8: two bytes; fp6: two bytes with
// the 6-bit code in the low bits; fp4: two nibbles, element 1 in the high nibble;
// ue8m0: two bytes).  The first source operand of the packing conversions goes
// into the high element, matching cvt.rn.f16x2.f32 (hardware-checked with
// ZLUDA's cvt_rn_f16x2_e4m3x2 and cvt_rn_satfinite_e4m3x2_f32).
//
// Instructions (all exact or round-to-nearest-even; .satfinite is mandatory in
// PTX for the narrowing direction and always applied here):
//   cvt.rn[.relu].satfinite.<fmt>x2.f32 d, a, b        cvt.rn[.relu].satfinite.<fmt>x2.f16x2 d, a
//   cvt.rn.f16x2.<fmt>x2 d, a                           cvt.rn.bf16x2.ue8m0x2 d, a
//   cvt.rz[.satfinite].ue8m0x2.f32 d, a, b  (also .rp, and .bf16x2 sources)
//
// Assumptions where the ISA text was not available offline: a NaN input to a
// format without NaN encodings yields +max-finite; a non-positive input to
// ue8m0 yields 0x00 (2^-127) and NaN yields 0xFF.
// =============================================================================
requires "ptx-exec-half.k"

module PTX-EXEC-MINI
  imports PTX-EXEC-HALF

  // kind -> (exponent bits, mantissa bits, bias)
  syntax Bool ::= isMini(String) [function, total]
  rule isMini(K) => K in (SetItem("e4m3") SetItem("e5m2") SetItem("e2m3") SetItem("e3m2") SetItem("e2m1") SetItem("ue5m3"))
  syntax Int ::= mE(String) [function, total] | mM(String) [function, total] | mBias(String) [function, total]
                | mTop(String) [function, total]           // largest finite code (without sign)
  rule mE("e4m3") => 4   rule mM("e4m3") => 3   rule mTop("e4m3") => 126          // 0x7E = 448
  rule mE("e5m2") => 5   rule mM("e5m2") => 2   rule mTop("e5m2") => 123          // 0x7B = 57344
  rule mE("e2m3") => 2   rule mM("e2m3") => 3   rule mTop("e2m3") => 31
  rule mE("e3m2") => 3   rule mM("e3m2") => 2   rule mTop("e3m2") => 31
  rule mE("e2m1") => 2   rule mM("e2m1") => 1   rule mTop("e2m1") => 7
  rule mE("ue5m3") => 5  rule mM("ue5m3") => 3  rule mTop("ue5m3") => 254         // unsigned, no infinity, NaN = 0xFF
  rule mE(_) => 0 [owise]    rule mM(_) => 0 [owise]    rule mTop(_) => 0 [owise]
  rule mBias(K) => (1 <<Int (mE(K) -Int 1)) -Int 1
  syntax Int ::= mBits(String) [function, total]                  // element width in bits (with sign)
  rule mBits("ue5m3") => 8
  rule mBits(K) => 1 +Int mE(K) +Int mM(K) [owise]

  // ---------------------------------------------------------------- element decode / encode
  syntax Bool ::= miniNaN(String, Int) [function, total]          // code without sign
  rule miniNaN("e4m3", C) => C ==Int 127
  rule miniNaN("e5m2", C) => C >Int 124
  rule miniNaN("ue5m3", C) => C ==Int 255
  rule miniNaN(_, _) => false [owise]
  syntax Bool ::= miniInf(String, Int) [function, total]
  rule miniInf("e5m2", C) => C ==Int 124
  rule miniInf(_, _) => false [owise]

  // exact value of the magnitude code (wide float)
  syntax Float ::= miniMag(String, Int) [function, total]
  rule miniMag(K, C) => scaleP2(Int2Float(#if (C >>Int mM(K)) ==Int 0 #then C &Int mask(mM(K))
                                          #else (C &Int mask(mM(K))) +Int (1 <<Int mM(K)) #fi, 64, WEXP),
                                #if (C >>Int mM(K)) ==Int 0 #then 1 -Int mBias(K) -Int mM(K)
                                #else (C >>Int mM(K)) -Int mBias(K) -Int mM(K) #fi)
  syntax Float ::= scaleP2(Float, Int) [function, total]           // F * 2^n, exact
  rule scaleP2(F, N) => F *Float (Int2Float(1, 64, WEXP) *Float Int2Float(1 <<Int N, 64, WEXP)) requires N >=Int 0
  rule scaleP2(F, N) => F /Float Int2Float(1 <<Int (0 -Int N), 64, WEXP) [owise]

  // element code (sign included) -> value as binary16 bits
  syntax Int ::= miniToF16(String, Int) [function, total]
  rule miniToF16(K, B) => #if miniNaN(K, B &Int mask(mBits(K) -Int 1)) #then 32767
                          #else #if miniInf(K, B &Int mask(mBits(K) -Int 1)) #then 31744 |Int ((B >>Int (mBits(K) -Int 1)) <<Int 15)
                          #else encodeF(signedF(B >>Int (mBits(K) -Int 1), miniMag(K, B &Int mask(mBits(K) -Int 1))), 16) #fi #fi
  syntax Float ::= signedF(Int, Float) [function, total]
  rule signedF(0, F) => F
  rule signedF(_, F) => --Float F [owise]

  // round |x| (wide, finite, nonzero or zero) to the nearest code, ties to even, saturating at mTop
  syntax Int ::= miniEnc(String, Float) [function, total] | miniEncR(String, Float, Int) [function, total]   // magnitude code; rounding 0 rn, 1 rz, 2 rp
  rule miniEnc(K, A) => miniEncR(K, A, 0)
  rule miniEncR(K, A, Rd) => #if A ==Float Int2Float(0, 64, WEXP) #then 0 #else miniEncE(K, A, mBias(K), minExp(K), Rd) #fi
  syntax Int ::= minExp(String) [function, total]
  rule minExp(K) => 1 -Int mBias(K)
  // find the binade: largest E in [minExp, ..] with 2^E <= A
  syntax Int ::= miniEncE(String, Float, Int, Int, Int) [function, total]
  rule miniEncE(K, A, Bias, E, Rd) => #if A >=Float scaleP2(Int2Float(1, 64, WEXP), E +Int 1) andBool E <Int 40
                                  #then miniEncE(K, A, Bias, E +Int 1, Rd) #else miniRound(K, A, E, Rd) #fi
  syntax Int ::= miniRound(String, Float, Int, Int) [function, total]
  rule miniRound(K, A, E, Rd) => miniCode(K, E, roundGrid(scaleP2(A, mM(K) -Int E), Rd))
  // nearest-even (0), toward zero (1), toward +infinity (2) integer of a wide nonnegative float
  syntax Int ::= roundGrid(Float, Int) [function, total]
  rule roundGrid(F, 1) => Float2Int(floorFloat(F))
  rule roundGrid(F, 2) => Float2Int(ceilFloat(F))
  rule roundGrid(F, _) => nearestEven(F) [owise]
  // nearest integer, ties to even, of a wide nonnegative float
  syntax Int ::= nearestEven(Float) [function, total]
  rule nearestEven(F) => #if F -Float Int2Float(Float2Int(floorFloat(F)), 64, WEXP) <Float Int2Float(1, 64, WEXP) /Float Int2Float(2, 64, WEXP) #then Float2Int(floorFloat(F))
                         #else #if F -Float Int2Float(Float2Int(floorFloat(F)), 64, WEXP) >Float Int2Float(1, 64, WEXP) /Float Int2Float(2, 64, WEXP) #then Float2Int(floorFloat(F)) +Int 1
                         #else #if Float2Int(floorFloat(F)) %Int 2 ==Int 0 #then Float2Int(floorFloat(F)) #else Float2Int(floorFloat(F)) +Int 1 #fi #fi #fi
  // assemble: binade exponent E (>= minExp), rounded significand N (in [2^m, 2^(m+1)] for normals; < 2^m subnormal)
  syntax Int ::= miniCode(String, Int, Int) [function, total]
  rule miniCode(K, E, N) => minInt(mTop(K), miniCodeRaw(K, E, N))
  syntax Int ::= miniCodeRaw(String, Int, Int) [function, total]
  rule miniCodeRaw(K, E, N) => #if N >=Int (1 <<Int (mM(K) +Int 1)) #then ((E +Int 1 +Int mBias(K)) <<Int mM(K))     // carry into next binade
                               #else #if N >=Int (1 <<Int mM(K)) #then ((E +Int mBias(K)) <<Int mM(K)) +Int (N -Int (1 <<Int mM(K)))
                               #else N #fi #fi                                                                   // subnormal
    requires E >=Int minExp(K)

  // full element encode of a value: NaN, relu, infinity, sign, rounding with saturation
  syntax Int ::= miniFrom(String, Float, Bool) [function, total] | miniFromR(String, Float, Bool, Int) [function, total]    // kind, value, relu?, rounding
  rule miniFrom(K, X, Relu) => miniFromR(K, X, Relu, 0)
  rule miniFromR(K, X, Relu, Rd) =>
       #if isNaN(X) #then (#if K ==String "e4m3" orBool K ==String "e5m2" #then 127 #else #if K ==String "ue5m3" #then 255 #else mTop(K) #fi #fi)
       #else #if (Relu orBool K ==String "ue5m3") andBool signFloat(X) #then 0
       #else #if isInfinite(X) #then miniSign(K, X) |Int mTop(K)
       #else miniSign(K, X) |Int miniEncR(K, roundFloat(absFloat(X), 300, WEXP), Rd) #fi #fi #fi
  syntax Int ::= miniSign(String, Float) [function, total]
  rule miniSign(K, X) => #if signFloat(X) andBool K =/=String "ue5m3" #then 1 <<Int (mBits(K) -Int 1) #else 0 #fi

  // ---------------------------------------------------------------- packed element access
  syntax Bool ::= isMiniX2(Ty) [function, total]
  rule isMiniX2(ty(K, _)) => K in (SetItem("e4m3x2") SetItem("e5m2x2") SetItem("e2m3x2") SetItem("e3m2x2") SetItem("e2m1x2") SetItem("ue5m3x2"))
  syntax String ::= miniKind(Ty) [function, total]
  rule miniKind(ty(K, _)) => substrString(K, 0, 4)
  rule miniKind(ty("ue5m3x2", _)) => "ue5m3"
  // element i (1 = high) of a packed pair, packed result
  syntax Int ::= miniElem(Ty, Int, Int) [function, total]          // container type, bits, i
  rule miniElem(T, V, I) => #if mBits(miniKind(T)) <=Int 4 #then (V >>Int (4 *Int I)) &Int 15 #else (V >>Int (8 *Int I)) &Int 255 #fi
  syntax Int ::= miniPack(Ty, Int, Int) [function, total]          // hi, lo element codes
  rule miniPack(T, H, L) => #if mBits(miniKind(T)) <=Int 4 #then (H <<Int 4) |Int L #else (H <<Int 8) |Int L #fi

  // ---------------------------------------------------------------- cvt to the narrow formats, with rounding, .pzo and scale factors
  //   cvt.{rn,rz}[.relu].satfinite[.pzo][.scaled::n1::ue8m0].<fmt>x2.f32 d, a, b{, scale}      (.rp also for ue5m3x2)
  //   cvt.{rn,rz}...<fmt>x2.{f16x2,bf16x2} d, a{, scale}
  syntax Float ::= nanF() [function, total] | scDiv(Float, Int) [function, total] | scMul(Float, Int) [function, total]
                 | flushIn(Bool, Float, Int) [function, total]
  rule nanF() => decodeF(2143289344, 32)
  rule scDiv(_, 255) => nanF()
  rule scDiv(X, S) => scaleP2(X, 127 -Int S) [owise]
  rule scMul(_, 255) => nanF()
  rule scMul(X, S) => scaleP2(X, S -Int 127) [owise]
  rule flushIn(true, X, MN) => Int2Float(0, 64, WEXP) requires notBool isNaN(X) andBool notBool isInfinite(X) andBool absFloat(X) <Float scaleP2(Int2Float(1, 64, WEXP), MN)
  rule flushIn(_, X, _) => X [owise]
  syntax Int ::= rndOf(Set) [function, total] | pzoFix(Bool, String, Int) [function, total]
  rule rndOf(M) => 1 requires "rz" in M
  rule rndOf(M) => 2 requires "rp" in M andBool notBool "rz" in M
  rule rndOf(_) => 0 [owise]
  rule pzoFix(true, K, C) => 0 requires K =/=String "ue5m3" andBool C ==Int (1 <<Int (mBits(K) -Int 1))
  rule pzoFix(_, _, C) => C [owise]
  syntax Int ::= miniCvt(Set, String, Float, Int) [function, total]       // modifiers, kind, value, scale code (127 = none)
  rule miniCvt(M, K, X, S) => pzoFix("pzo" in M, K, miniFromR(K, scDiv(X, S), "relu" in M, rndOf(M)))
  // scale operand bytes: n1 -> bits 7:0 for both elements, n2 -> hi element 15:8, lo element 7:0
  syntax Int ::= scHi(Set, Int) [function, total] | scLo(Set, Int) [function, total]
  rule scHi(M, S) => #if "scaled::n2::ue8m0" in M #then (S >>Int 8) &Int 255 #else S &Int 255 #fi
  rule scLo(_, S) => S &Int 255
  syntax Int ::= defScale(Set) [function, total]
  rule defScale(M) => #if "scaled::n2::ue8m0" in M #then 32639 #else 127 #fi          // 0x7f7f / 0x7f
  syntax Bool ::= isN1(Set) [function, total]
  rule isN1(M) => "scaled::n1::ue8m0" in M

  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, miniPack(DT, miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(rd(A, ST, C), fw(ST)), 0 -Int 126), 127),
                                          miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(rd(B, ST, C), fw(ST)), 0 -Int 126), 127))) </regs>
    requires isMiniX2(DT) andBool isF(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, miniPack(DT, miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(rd(A, ST, C), fw(ST)), 0 -Int 126), scHi(M, rd(S, ty("b", 16), C))),
                                          miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(rd(B, ST, C), fw(ST)), 0 -Int 126), scLo(M, rd(S, ty("b", 16), C))))) </regs>
    requires isMiniX2(DT) andBool isF(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, miniPack(DT, miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(lane16(rd(A, ST, C), 1), fw(laneTy(ST))), 0 -Int 14), 127),
                                          miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(lane16(rd(A, ST, C), 0), fw(laneTy(ST))), 0 -Int 14), 127))) </regs>
    requires isMiniX2(DT) andBool isHalfPair(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, miniPack(DT, miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(lane16(rd(A, ST, C), 1), fw(laneTy(ST))), 0 -Int 14), scHi(M, rd(S, ty("b", 16), C))),
                                          miniCvt(M, miniKind(DT), flushIn(isN1(M), decodeF(lane16(rd(A, ST, C), 0), fw(laneTy(ST))), 0 -Int 14), scLo(M, rd(S, ty("b", 16), C))))) </regs>
    requires isMiniX2(DT) andBool isHalfPair(ST)

  // element -> exact float (NaN / infinity per format)
  syntax Float ::= miniDecF(String, Int) [function, total]
  rule miniDecF(K, B) => nanF() requires miniNaN(K, B &Int mask(mBits(K) -Int (#if K ==String "ue5m3" #then 0 #else 1 #fi)))
  rule miniDecF(K, B) => signedF(B >>Int (mBits(K) -Int 1), decodeF(2139095040, 32))
    requires notBool miniNaN(K, B &Int mask(mBits(K) -Int (#if K ==String "ue5m3" #then 0 #else 1 #fi)))
     andBool miniInf(K, B &Int mask(mBits(K) -Int 1))
  rule miniDecF(K, B) => signedF(#if K ==String "ue5m3" #then 0 #else B >>Int (mBits(K) -Int 1) #fi,
                                 miniMag(K, B &Int mask(#if K ==String "ue5m3" #then 8 #else mBits(K) -Int 1 #fi))) [owise]
  // cvt.rn.{f16x2,bf16x2}.<fmt>x2 d, a, scale   (scale only with .scaled::n2::ue8m0 and a bf16x2 destination)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(encodeF(scMul(miniDecF(miniKind(ST), miniElem(ST, rd(A, ST, C), 0)), scLo(M, rd(S, ty("b", 16), C))), fw(laneTy(DT))),
                                   encodeF(scMul(miniDecF(miniKind(ST), miniElem(ST, rd(A, ST, C), 1)), scHi(M, rd(S, ty("b", 16), C))), fw(laneTy(DT))))) </regs>
    requires isHalfPair(DT) andBool isMiniX2(ST)
  // ue5m3x2 up-conversion without scale (cvt.rn.f16x2.ue5m3x2 overflows f16: encodeF saturates to infinity)
  rule <tk> exec(opc("cvt", _, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(encodeF(miniDecF("ue5m3", miniElem(ST, rd(A, ST, C), 0)), fw(laneTy(DT))),
                                   encodeF(miniDecF("ue5m3", miniElem(ST, rd(A, ST, C), 1)), fw(laneTy(DT))))) </regs>
    requires isHalfPair(DT) andBool ST ==K ty("ue5m3x2", 16) [priority(40)]

  // ---------------------------------------------------------------- s2f6x2 (signed 2.6 fixed point, saturating; PTX ISA 9.1)
  syntax Bool ::= isS2F6(Ty) [function, total]
  rule isS2F6(T) => T ==K ty("s2f6x2", 16)
  syntax Int ::= s2f6Enc(Set, Float, Int) [function, total]            // modifiers, value (already divided by the scale), scale code
  rule s2f6Enc(M, X, S) => s2f6Code(M, scDiv(X, S))
  syntax Int ::= s2f6Code(Set, Float) [function, total]
  rule s2f6Code(_, X) => 127 requires isNaN(X)
  rule s2f6Code(M, X) => 0 requires notBool isNaN(X) andBool "relu" in M andBool signFloat(X)
  rule s2f6Code(_, X) => #if signFloat(X) #then 128 #else 127 #fi requires notBool isNaN(X) andBool isInfinite(X)
  rule s2f6Code(M, X) => #if signFloat(X) #then (0 -Int minInt(128, roundGrid(scaleP2(absFloat(X), 6), rndOf(M)))) &Int 255
                         #else minInt(127, roundGrid(scaleP2(X, 6), rndOf(M))) #fi
    requires notBool isNaN(X) andBool notBool isInfinite(X) andBool notBool ("relu" in M andBool signFloat(X))
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, (s2f6Enc(M, decodeF(rd(A, ST, C), fw(ST)), 127) <<Int 8) |Int s2f6Enc(M, decodeF(rd(B, ST, C), fw(ST)), 127)) </regs>
    requires isS2F6(DT) andBool isF(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, (s2f6Enc(M, decodeF(rd(A, ST, C), fw(ST)), scHi(M, rd(S, ty("b", 16), C))) <<Int 8) |Int s2f6Enc(M, decodeF(rd(B, ST, C), fw(ST)), scLo(M, rd(S, ty("b", 16), C)))) </regs>
    requires isS2F6(DT) andBool isF(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, (s2f6Enc(M, decodeF(lane16(rd(A, ST, C), 1), fw(laneTy(ST))), 127) <<Int 8) |Int s2f6Enc(M, decodeF(lane16(rd(A, ST, C), 0), fw(laneTy(ST))), 127)) </regs>
    requires isS2F6(DT) andBool isHalfPair(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, (s2f6Enc(M, decodeF(lane16(rd(A, ST, C), 1), fw(laneTy(ST))), scHi(M, rd(S, ty("b", 16), C))) <<Int 8) |Int s2f6Enc(M, decodeF(lane16(rd(A, ST, C), 0), fw(laneTy(ST))), scLo(M, rd(S, ty("b", 16), C)))) </regs>
    requires isS2F6(DT) andBool isHalfPair(ST)
  // s2f6x2 -> bf16x2: each byte is a two's complement integer times 2^-6, times the optional scale
  syntax Float ::= s2f6Val(Int) [function, total]
  rule s2f6Val(B) => scaleP2(Int2Float(sext(B &Int 255, 8), 64, WEXP), 0 -Int 6)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(encodeF(s2f6Val(rd(A, ST, C)), fw(laneTy(DT))), encodeF(s2f6Val(rd(A, ST, C) >>Int 8), fw(laneTy(DT))))) </regs>
    requires isHalfPair(DT) andBool isS2F6(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, S, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(encodeF(scMul(s2f6Val(rd(A, ST, C)), scLo(M, rd(S, ty("b", 16), C))), fw(laneTy(DT))),
                                   encodeF(scMul(s2f6Val(rd(A, ST, C) >>Int 8), scHi(M, rd(S, ty("b", 16), C))), fw(laneTy(DT))))) </regs>
    requires isHalfPair(DT) andBool isS2F6(ST)

  // cvt.rn.f16x2.<fmt>x2 d, a  (exact; also into bf16x2)
  rule <tk> exec(opc("cvt", _, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(miniTo16(miniKind(ST), miniElem(ST, rd(A, ST, C), 0), fw(laneTy(DT))),
                                   miniTo16(miniKind(ST), miniElem(ST, rd(A, ST, C), 1), fw(laneTy(DT))))) </regs>
    requires isHalfPair(DT) andBool isMiniX2(ST)
  syntax Int ::= miniTo16(String, Int, Int) [function, total]       // kind, element code, destination format
  rule miniTo16(K, B, 16) => miniToF16(K, B)
  rule miniTo16(K, B, F) => encodeF(decodeF(miniToF16(K, B), 16), F) [owise]

  // ---------------------------------------------------------------- ue8m0 (unsigned 8-bit exponent, 2^(e-127), 0xFF = NaN)
  syntax Bool ::= isUe(Ty) [function, total]
  rule isUe(ty("ue8m0x2", _)) => true
  rule isUe(_) => false [owise]
  // exponent code of |x| rounded toward zero (.rz) or up (.rp); x <= 0 -> 0, inf -> 254, NaN -> 255
  syntax Int ::= ueEnc(Float, Bool) [function, total]
  rule ueEnc(X, _) => 255 requires isNaN(X)
  rule ueEnc(X, _) => 0 requires notBool isNaN(X) andBool (signFloat(X) orBool X ==Float Int2Float(0, 24, 8))
  rule ueEnc(X, _) => 254 requires notBool isNaN(X) andBool notBool signFloat(X) andBool isInfinite(X)
  rule ueEnc(X, Up) => maxInt(0, minInt(254, ueExp(roundFloat(X, 300, WEXP), Up) +Int 127))
    requires notBool isNaN(X) andBool notBool signFloat(X) andBool notBool isInfinite(X) andBool X >Float Int2Float(0, 24, 8)
  // floor(log2 a) (or ceil when Up and a is not a power of two) by search
  syntax Int ::= ueExp(Float, Bool) [function, total]
  rule ueExp(A, Up) => ueSearch(A, Up, 0 -Int 160)
  syntax Int ::= ueSearch(Float, Bool, Int) [function, total]
  rule ueSearch(A, Up, E) => #if E >=Int 130 #then E
                             #else #if A <Float scaleP2(Int2Float(1, 64, WEXP), E +Int 1) #then (#if Up andBool A >Float scaleP2(Int2Float(1, 64, WEXP), E) #then E +Int 1 #else E #fi)
                             #else ueSearch(A, Up, E +Int 1) #fi #fi
  syntax Int ::= ueDec(Int) [function, total]                      // code -> bfloat16 bits
  rule ueDec(255) => 32767
  rule ueDec(0) => 64
  rule ueDec(C) => C <<Int 7 [owise]

  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, ueEnc(decodeF(rd(A, ST, C), fw(ST)), "rp" in M) <<Int 8 |Int ueEnc(decodeF(rd(B, ST, C), fw(ST)), "rp" in M)) </regs>
    requires isUe(DT) andBool isF(ST)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, ueEnc(decodeF(lane16(rd(A, ST, C), 1), 116), "rp" in M) <<Int 8 |Int ueEnc(decodeF(lane16(rd(A, ST, C), 0), 116), "rp" in M)) </regs>
    requires isUe(DT) andBool isHalfPair(ST)
  rule <tk> exec(opc("cvt", _, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack16(ueDec(rd(A, ST, C) &Int 255), ueDec((rd(A, ST, C) >>Int 8) &Int 255))) </regs>
    requires isHalfPair(DT) andBool isUe(ST)
endmodule

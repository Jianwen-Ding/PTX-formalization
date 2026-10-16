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
  rule isMini(K) => K in (SetItem("e4m3") SetItem("e5m2") SetItem("e2m3") SetItem("e3m2") SetItem("e2m1"))
  syntax Int ::= mE(String) [function, total] | mM(String) [function, total] | mBias(String) [function, total]
                | mTop(String) [function, total]           // largest finite code (without sign)
  rule mE("e4m3") => 4   rule mM("e4m3") => 3   rule mTop("e4m3") => 126          // 0x7E = 448
  rule mE("e5m2") => 5   rule mM("e5m2") => 2   rule mTop("e5m2") => 123          // 0x7B = 57344
  rule mE("e2m3") => 2   rule mM("e2m3") => 3   rule mTop("e2m3") => 31
  rule mE("e3m2") => 3   rule mM("e3m2") => 2   rule mTop("e3m2") => 31
  rule mE("e2m1") => 2   rule mM("e2m1") => 1   rule mTop("e2m1") => 7
  rule mE(_) => 0 [owise]    rule mM(_) => 0 [owise]    rule mTop(_) => 0 [owise]
  rule mBias(K) => (1 <<Int (mE(K) -Int 1)) -Int 1
  syntax Int ::= mBits(String) [function, total]                  // element width in bits (with sign)
  rule mBits(K) => 1 +Int mE(K) +Int mM(K)

  // ---------------------------------------------------------------- element decode / encode
  syntax Bool ::= miniNaN(String, Int) [function, total]          // code without sign
  rule miniNaN("e4m3", C) => C ==Int 127
  rule miniNaN("e5m2", C) => C >Int 124
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
  syntax Int ::= miniEnc(String, Float) [function, total]          // magnitude code
  rule miniEnc(K, A) => #if A ==Float Int2Float(0, 64, WEXP) #then 0 #else miniEncE(K, A, mBias(K), minExp(K)) #fi
  syntax Int ::= minExp(String) [function, total]
  rule minExp(K) => 1 -Int mBias(K)
  // find the binade: largest E in [minExp, ..] with 2^E <= A
  syntax Int ::= miniEncE(String, Float, Int, Int) [function, total]
  rule miniEncE(K, A, Bias, E) => #if A >=Float scaleP2(Int2Float(1, 64, WEXP), E +Int 1) andBool E <Int 40
                                  #then miniEncE(K, A, Bias, E +Int 1) #else miniRound(K, A, E) #fi
  syntax Int ::= miniRound(String, Float, Int) [function, total]
  rule miniRound(K, A, E) => miniCode(K, E, nearestEven(scaleP2(A, mM(K) -Int E)))
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
  syntax Int ::= miniFrom(String, Float, Bool) [function, total]    // kind, value, relu?
  rule miniFrom(K, X, Relu) =>
       #if isNaN(X) #then (#if K ==String "e4m3" orBool K ==String "e5m2" #then 127 #else mTop(K) #fi)
       #else #if Relu andBool signFloat(X) #then 0
       #else #if isInfinite(X) #then miniSign(K, X) |Int mTop(K)
       #else miniSign(K, X) |Int miniEnc(K, roundFloat(absFloat(X), 300, WEXP)) #fi #fi #fi
  syntax Int ::= miniSign(String, Float) [function, total]
  rule miniSign(K, X) => #if signFloat(X) #then 1 <<Int (mBits(K) -Int 1) #else 0 #fi

  // ---------------------------------------------------------------- packed element access
  syntax Bool ::= isMiniX2(Ty) [function, total]
  rule isMiniX2(ty(K, _)) => K in (SetItem("e4m3x2") SetItem("e5m2x2") SetItem("e2m3x2") SetItem("e3m2x2") SetItem("e2m1x2"))
  syntax String ::= miniKind(Ty) [function, total]
  rule miniKind(ty(K, _)) => substrString(K, 0, 4)
  // element i (1 = high) of a packed pair, packed result
  syntax Int ::= miniElem(Ty, Int, Int) [function, total]          // container type, bits, i
  rule miniElem(T, V, I) => #if mBits(miniKind(T)) <=Int 4 #then (V >>Int (4 *Int I)) &Int 15 #else (V >>Int (8 *Int I)) &Int 255 #fi
  syntax Int ::= miniPack(Ty, Int, Int) [function, total]          // hi, lo element codes
  rule miniPack(T, H, L) => #if mBits(miniKind(T)) <=Int 4 #then (H <<Int 4) |Int L #else (H <<Int 8) |Int L #fi

  // cvt.rn[.relu].satfinite.<fmt>x2.f32 d, a, b
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, miniPack(DT, miniFrom(miniKind(DT), decodeF(rd(A, ST, C), fw(ST)), "relu" in M),
                                          miniFrom(miniKind(DT), decodeF(rd(B, ST, C), fw(ST)), "relu" in M))) </regs>
    requires isMiniX2(DT) andBool isF(ST)
  // cvt.rn[.relu].satfinite.<fmt>x2.f16x2 d, a   (and .bf16x2)
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, miniPack(DT, miniFrom(miniKind(DT), decodeF(lane16(rd(A, ST, C), 1), fw(laneTy(ST))), "relu" in M),
                                          miniFrom(miniKind(DT), decodeF(lane16(rd(A, ST, C), 0), fw(laneTy(ST))), "relu" in M))) </regs>
    requires isMiniX2(DT) andBool isHalfPair(ST)
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

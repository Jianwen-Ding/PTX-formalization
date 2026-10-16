// =============================================================================
// Shared helpers: PTX types, bit-vector arithmetic, IEEE-754 encode/decode,
// token conversion and opcode decoding.
//
// Register and memory values are represented uniformly as *unsigned bit
// patterns* (K Int in [0, 2^W)).  Instructions interpret those bits according
// to their type suffix: two's complement for .sN, unsigned for .uN/.bN, and
// IEEE-754 binary16/32/64 for .f16/.f32/.f64.
// =============================================================================
requires "ptx-syntax.k"

module PTX-COMMON
  imports PTX-SYNTAX
  imports DOMAINS
  imports FLOAT
  imports BYTES
  imports STRING
  imports K-EQUAL

  // ------------------------------------------------------------ token -> string
  syntax String ::= Name2String(Name)     [function, total, hook(STRING.token2string)]
                  | Reg2String(Reg)       [function, total, hook(STRING.token2string)]
                  | Hex2String(HexLit)    [function, total, hook(STRING.token2string)]

  // ------------------------------------------------------------ types
  // kind is one of "u", "s", "b", "f", "p" (predicate)
  syntax Ty ::= ty(String, Int)

  syntax Ty ::= typeOf(Type) [function, total]
  rule typeOf(.u8)  => ty("u", 8)    rule typeOf(.u16) => ty("u", 16)
  rule typeOf(.u32) => ty("u", 32)   rule typeOf(.u64) => ty("u", 64)
  rule typeOf(.s8)  => ty("s", 8)    rule typeOf(.s16) => ty("s", 16)
  rule typeOf(.s32) => ty("s", 32)   rule typeOf(.s64) => ty("s", 64)
  rule typeOf(.b8)  => ty("b", 8)    rule typeOf(.b16) => ty("b", 16)
  rule typeOf(.b32) => ty("b", 32)   rule typeOf(.b64) => ty("b", 64)
  rule typeOf(.f16) => ty("f", 16)   rule typeOf(.f32) => ty("f", 32)
  rule typeOf(.f64) => ty("f", 64)   rule typeOf(.pred) => ty("p", 1)

  syntax Bool ::= isTyStr(String) [function, total]
  rule isTyStr(S) => S in tyStrs
  syntax Set ::= "tyStrs" [macro]
  rule tyStrs => SetItem("u8") SetItem("u16") SetItem("u32") SetItem("u64")
                 SetItem("s8") SetItem("s16") SetItem("s32") SetItem("s64")
                 SetItem("b8") SetItem("b16") SetItem("b32") SetItem("b64")
                 SetItem("f16") SetItem("f32") SetItem("f64") SetItem("pred")

  syntax Ty ::= strTy(String) [function]
  rule strTy("pred") => ty("p", 1)
  rule strTy(S) => ty(substrString(S, 0, 1), String2Int(substrString(S, 1, lengthString(S))))
    requires S =/=String "pred"

  syntax Int ::= wid(Ty)   [function, total]
               | bytes(Ty) [function, total]
  rule wid(ty(_, W)) => W
  rule bytes(ty(_, W)) => maxInt(W /Int 8, 1)

  syntax Bool ::= isF(Ty) [function, total] | isS(Ty) [function, total]
  rule isF(ty(K, _)) => K ==String "f"
  rule isS(ty(K, _)) => K ==String "s"

  syntax Ty ::= wide(Ty) [function, total]    // double-width version of a type
  rule wide(ty(K, W)) => ty(K, 2 *Int W)

  // ------------------------------------------------------------ bit vectors
  syntax Int ::= mask(Int)            [function, total]
               | trunc(Int, Int)      [function, total]   // value, width
               | sext(Int, Int)       [function, total]   // signed view
               | ival(Int, Ty)        [function, total]   // mathematical value
               | minOf(Ty)            [function, total]
               | maxOf(Ty)            [function, total]
               | clamp(Int, Ty)       [function, total]
  rule mask(W) => (1 <<Int W) -Int 1                       requires W >=Int 0
  rule mask(_) => 0                                         [owise]
  rule trunc(V, W) => V &Int mask(W)
  rule sext(V, W) => #if trunc(V, W) >=Int (1 <<Int (W -Int 1))
                     #then trunc(V, W) -Int (1 <<Int W)
                     #else trunc(V, W) #fi
  rule ival(V, T) => #if isS(T) #then sext(V, wid(T)) #else trunc(V, wid(T)) #fi
  rule minOf(T) => #if isS(T) #then 0 -Int (1 <<Int (wid(T) -Int 1)) #else 0 #fi
  rule maxOf(T) => #if isS(T) #then (1 <<Int (wid(T) -Int 1)) -Int 1 #else mask(wid(T)) #fi
  rule clamp(V, T) => minInt(maxInt(V, minOf(T)), maxOf(T))

  syntax Int ::= popc(Int)          [function, total]
               | brev(Int, Int)     [function, total]
               | brevAux(Int, Int, Int) [function, total]
               | clz(Int, Int)      [function, total]
  rule popc(0) => 0
  rule popc(V) => (V &Int 1) +Int popc(V >>Int 1) requires V >Int 0
  rule popc(_) => 0 [owise]
  rule brev(V, W) => brevAux(V, W, 0)
  rule brevAux(_, 0, A) => A
  rule brevAux(V, N, A) => brevAux(V >>Int 1, N -Int 1, (A <<Int 1) |Int (V &Int 1)) requires N >Int 0
  rule brevAux(_, _, A) => A [owise]
  rule clz(0, W) => W
  rule clz(V, W) => W -Int (log2Int(V) +Int 1) requires V >Int 0
  rule clz(_, W) => W [owise]

  // ------------------------------------------------------------ IEEE-754
  // precision (incl. hidden bit) and exponent width for binary16/32/64
  syntax Int ::= prec(Int)  [function, total]
               | ebits(Int) [function, total]
               | fscale(Int) [function, total]  // log2 of 1/(smallest subnormal)
  rule prec(16) => 11   rule ebits(16) => 5
  rule prec(32) => 24   rule ebits(32) => 8
  rule prec(64) => 53   rule ebits(64) => 11
  rule prec(_)  => 53   [owise]
  rule ebits(_) => 11   [owise]
  rule fscale(W) => prec(W) -Int 2 +Int ((1 <<Int (ebits(W) -Int 1)) -Int 1)

  // Wide-exponent working format: every binary16/32/64 value, scaled by
  // 2^fscale, is an integer that is exactly representable in it.
  syntax Int ::= "WEXP" [macro]
  rule WEXP => 20

  // bits -> Float (of the matching K precision)
  syntax Float ::= decodeF(Int, Int)                      [function]  // bits, width
                 | decodeAux(Int, Int, Int, Int)          [function]  // sign, exp, mant, width
                 | pinf(Int)                              [function]
                 | qnan(Int)                              [function]
  rule pinf(W) => Int2Float(1, prec(W), ebits(W)) /Float Int2Float(0, prec(W), ebits(W))
  rule qnan(W) => Int2Float(0, prec(W), ebits(W)) /Float Int2Float(0, prec(W), ebits(W))

  rule decodeF(B, W) => decodeAux((B >>Int (W -Int 1)) &Int 1,
                                  (B >>Int (prec(W) -Int 1)) &Int mask(ebits(W)),
                                  B &Int mask(prec(W) -Int 1), W)
  rule decodeAux(S, E, M, W) => #if M ==Int 0 #then signed(S, pinf(W)) #else qnan(W) #fi
    requires E ==Int mask(ebits(W))
  rule decodeAux(S, E, M, W) =>
       signed(S, roundFloat(
         Int2Float(#if E ==Int 0 #then M #else (M +Int (1 <<Int (prec(W) -Int 1))) <<Int (E -Int 1) #fi,
                   prec(W), WEXP)
         /Float Int2Float(1 <<Int fscale(W), prec(W), WEXP),
         prec(W), ebits(W)))
    requires E =/=Int mask(ebits(W))

  syntax Float ::= signed(Int, Float) [function]
  rule signed(0, F) => F
  rule signed(_, F) => --Float F [owise]

  // Float -> bits (after rounding to the target format, RN-even)
  syntax Int ::= encodeF(Float, Int)        [function]   // value, width
               | encodeAux(Float, Int)      [function]
               | encFin(Int, Int, Int)      [function]   // sign, scaled magnitude, width
  rule encodeF(F, W) => encodeAux(roundFloat(F, prec(W), ebits(W)), W)
  rule encodeAux(F, W) => mask(W -Int 1) requires isNaN(F)            // canonical NaN
  rule encodeAux(F, W) => (#if signFloat(F) #then 1 <<Int (W -Int 1) #else 0 #fi)
                          |Int (mask(ebits(W)) <<Int (prec(W) -Int 1))
    requires notBool isNaN(F) andBool isInfinite(F)
  rule encodeAux(F, W) =>
       encFin(#if signFloat(F) #then 1 #else 0 #fi,
              Float2Int(roundFloat(absFloat(F), prec(W), WEXP)
                        *Float Int2Float(1 <<Int fscale(W), prec(W), WEXP)), W)
    requires notBool isNaN(F) andBool notBool isInfinite(F)
  rule encFin(S, N, W) => (S <<Int (W -Int 1)) |Int N
    requires N <Int (1 <<Int (prec(W) -Int 1))
  rule encFin(S, N, W) =>
         (S <<Int (W -Int 1))
    |Int ((log2Int(N) -Int prec(W) +Int 2) <<Int (prec(W) -Int 1))
    |Int ((N >>Int (log2Int(N) -Int (prec(W) -Int 1))) -Int (1 <<Int (prec(W) -Int 1)))
    [owise]

  // constant of the same format as a reference value
  syntax Float ::= fc(Float, Float) [function]
  rule fc(C, Ref) => roundFloat(C, precisionFloat(Ref), exponentBitsFloat(Ref))

  // fused multiply-add: compute exactly in a very wide format, round once
  syntax Float ::= fmaF(Float, Float, Float) [function]
  rule fmaF(A, B, C) => roundFloat(
         roundFloat(A, 4400, WEXP) *Float roundFloat(B, 4400, WEXP)
         +Float roundFloat(C, 4400, WEXP), precisionFloat(A), exponentBitsFloat(A))

  // round-to-integral in a given mode ("rni", "rzi", "rmi", "rpi")
  syntax Float ::= rndInt(Float, String) [function]
  rule rndInt(F, "rzi") => truncFloat(F)
  rule rndInt(F, "rmi") => floorFloat(F)
  rule rndInt(F, "rpi") => ceilFloat(F)
  rule rndInt(F, _)     => F requires isNaN(F) orBool isInfinite(F)
  rule rndInt(F, "rni") => rni(F, floorFloat(F), F -Float floorFloat(F))
    requires notBool (isNaN(F) orBool isInfinite(F))
  syntax Float ::= rni(Float, Float, Float) [function]
  rule rni(_, Fl, D) => Fl requires D <Float fc(0.5, Fl)
  rule rni(_, Fl, D) => Fl +Float fc(1.0, Fl) requires D >Float fc(0.5, Fl)
  rule rni(_, Fl, D) => #if Float2Int(Fl) %Int 2 ==Int 0 #then Fl #else Fl +Float fc(1.0, Fl) #fi
    requires D ==Float fc(0.5, Fl)

  // float -> integer conversion with PTX saturation semantics
  syntax Int ::= f2i(Float, String, Ty) [function]
  rule f2i(F, _, _) => 0 requires isNaN(F)
  rule f2i(F, M, T) => trunc(clamp(Float2Int(rndInt(F, M)), T), wid(T))
    requires notBool isNaN(F) andBool notBool isInfinite(F)
  rule f2i(F, _, T) => trunc(#if signFloat(F) #then minOf(T) #else maxOf(T) #fi, wid(T))
    requires isInfinite(F)

  syntax Float ::= satF(Float) [function]      // clamp to [0.0, 1.0], NaN -> 0
  rule satF(F) => fc(0.0, F) requires isNaN(F) orBool F <Float fc(0.0, F)
  rule satF(F) => fc(1.0, F) requires F >Float fc(1.0, F)
  rule satF(F) => F [owise]

  // ------------------------------------------------------------ literals
  syntax Int ::= hexBits(HexLit, Ty) [function]
               | hexBitsAux(String, String, Ty) [function]
  rule hexBits(H, T) => hexBitsAux(substrString(Hex2String(H), 1, 2),
                                   substrString(Hex2String(H), 2, lengthString(Hex2String(H))), T)
  rule hexBitsAux(P, D, T) => encodeF(decodeF(String2Base(D, 16), 32), wid(T))
    requires (P ==String "f" orBool P ==String "F") andBool isF(T) andBool wid(T) =/=Int 32
  rule hexBitsAux(P, D, T) => encodeF(decodeF(String2Base(D, 16), 64), wid(T))
    requires (P ==String "d" orBool P ==String "D") andBool isF(T) andBool wid(T) =/=Int 64
  rule hexBitsAux(_, D, T) => trunc(String2Base(D, 16), wid(T)) [owise]

  // integer / float literal interpreted at a type
  syntax Int ::= intLit(Int, Ty)   [function]
               | fltLit(Float, Ty) [function]
  rule intLit(I, T) => encodeF(Int2Float(I, prec(wid(T)), ebits(wid(T))), wid(T)) requires isF(T)
  rule intLit(I, T) => trunc(I, wid(T)) [owise]
  rule fltLit(F, T) => encodeF(F, wid(T)) requires isF(T)
  rule fltLit(F, T) => trunc(Float2Int(F), wid(T)) [owise]

  syntax Int ::= initBits(InitVal, Ty) [function]
  rule initBits(I:Int, T) => intLit(I, T)
  rule initBits(F:Float, T) => fltLit(F, T)
  rule initBits(H:HexLit, T) => hexBits(H, T)

  // ------------------------------------------------------------ formatting
  syntax String ::= fmtVal(Int, Ty) [function]
  rule fmtVal(V, T) => fmtFloat(decodeF(V, wid(T))) requires isF(T)
  rule fmtVal(V, T) => Int2String(ival(V, T)) [owise]

  // Human-readable rendering of K's "0.DDDDeXf" Float2String output,
  // e.g. 0.150000000e1f ~> 1.5,  0.100000000e11f ~> 1.0e10
  syntax String ::= fmtFloat(Float)                    [function]
                  | fmtSci(Bool, String, Int)          [function]  // neg, digits, exponent
                  | stripZeros(String)                 [function, total]
                  | zeros(Int)                         [function, total]
  rule fmtFloat(F) => "nan" requires isNaN(F)
  rule fmtFloat(F) => #if signFloat(F) #then "-inf" #else "inf" #fi requires isInfinite(F)
  rule fmtFloat(F) => #if signFloat(F) #then "-0.0" #else "0.0" #fi
    requires notBool isNaN(F) andBool F ==Float fc(0.0, F)
  rule fmtFloat(F) => fmtSci(signFloat(F),
                             stripZeros(substrString(Float2String(absFloat(F)), 2, findString(Float2String(absFloat(F)), "e", 0))),
                             String2Int(digitsAfterE(Float2String(absFloat(F)))))
    [owise]
  syntax String ::= digitsAfterE(String) [function]
                  | takeInt(String, Int) [function, total]
  rule digitsAfterE(S) => takeInt(substrString(S, findString(S, "e", 0) +Int 1, lengthString(S)), 0)
  rule takeInt(S, I) => takeInt(S, I +Int 1)
    requires I <Int lengthString(S)
     andBool (findChar("-+0123456789", substrString(S, I, I +Int 1), 0) >=Int 0)
  rule takeInt(S, I) => substrString(S, 0, I) [owise]
  rule stripZeros(S) => stripZeros(substrString(S, 0, lengthString(S) -Int 1))
    requires lengthString(S) >Int 1 andBool substrString(S, lengthString(S) -Int 1, lengthString(S)) ==String "0"
  rule stripZeros(S) => S [owise]
  rule zeros(N) => "0" +String zeros(N -Int 1) requires N >Int 0
  rule zeros(_) => "" [owise]
  // value = 0.D1D2...Dn * 10^E
  rule fmtSci(N, D, E) => #if N #then "-" #else "" #fi +String fmtPos(D, E)
  syntax String ::= fmtPos(String, Int) [function]
  rule fmtPos(D, E) => "0." +String zeros(0 -Int E) +String D
    requires E <=Int 0 andBool E >Int -5
  rule fmtPos(D, E) => D +String zeros(E -Int lengthString(D)) +String ".0"
    requires E >Int 0 andBool E <=Int 9 andBool lengthString(D) <=Int E
  rule fmtPos(D, E) => substrString(D, 0, E) +String "." +String substrString(D, E, lengthString(D))
    requires E >Int 0 andBool E <=Int 9 andBool lengthString(D) >Int E
  rule fmtPos(D, E) => substrString(D, 0, 1) +String "." +String
                       #if lengthString(D) >Int 1 #then substrString(D, 1, lengthString(D)) #else "0" #fi
                       +String "e" +String Int2String(E -Int 1)
    [owise]

  // ------------------------------------------------------------ opcodes
  // `ld.global.nc.f32` ~> opc("ld", {global, nc}, [f32])
  syntax Opc ::= opc(String, Set, List)
  syntax Opc ::= decodeOpc(String) [function]
               | decOpc(List, Opc) [function]
  syntax List ::= splitDot(String) [function]
  rule splitDot(S) => ListItem(S) requires findString(S, ".", 0) <Int 0
  rule splitDot(S) => ListItem(substrString(S, 0, findString(S, ".", 0)))
                      splitDot(substrString(S, findString(S, ".", 0) +Int 1, lengthString(S)))
    [owise]
  rule decodeOpc(S) => decOpc(splitDot(S), opc("", .Set, .List))
  rule decOpc(ListItem(B) L, opc("", .Set, .List)) => decOpc(L, opc(B, .Set, .List))
  rule decOpc(ListItem(T:String) L, opc(B, M, TS)) => decOpc(L, opc(B, M, TS ListItem(strTy(T))))
    requires B =/=String "" andBool isTyStr(T)
  rule decOpc(ListItem(X:String) L, opc(B, M, TS)) => decOpc(L, opc(B, M SetItem(X), TS))
    requires B =/=String "" andBool notBool isTyStr(X)
  rule decOpc(.List, O) => O

  // first element of a candidate list that occurs in a modifier set
  syntax String ::= pick(Set, List, String) [function, total]
  rule pick(M, ListItem(X:String) _, _) => X requires X in M
  rule pick(M, ListItem(_) L, D) => pick(M, L, D) [owise]
  rule pick(_, .List, D) => D

  syntax String ::= spaceOf(Set) [function, total]
  rule spaceOf(M) => pick(M, ListItem("global") ListItem("shared") ListItem("local")
                             ListItem("param") ListItem("const"), "generic")
endmodule

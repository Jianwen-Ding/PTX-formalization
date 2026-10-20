// =============================================================================
// PTX-ALU: pure semantics of arithmetic, logic, comparison, conversion and
// atomic read-modify-write operations on bit patterns.
//
// Floating point: all IEEE-754 arithmetic is delegated to K's builtin FLOAT
// module (GNU MPFR, correctly rounded, round-to-nearest-even).  `.approx`
// instructions are computed with the correctly-rounded MPFR result, which is
// within the error bounds the PTX ISA allows.  Directed rounding modifiers
// (.rz/.rm/.rp) on float arithmetic and `.ftz` are accepted but not modelled.
// =============================================================================
requires "ptx-configuration.k"
requires "ptx-float.k"

module PTX-ALU
  imports PTX-CONFIGURATION
  imports PTX-FLOAT

  syntax Int ::= b2i(Bool) [function, total]
  rule b2i(true) => 1
  rule b2i(false) => 0

  syntax Int ::= fres(Float, Set, Ty) [function]       // encode, applying .sat
  rule fres(F, M, T) => encodeF(#if "sat" in M #then satF(F) #else F #fi, fw(T))

  syntax Float ::= fv(Int, Ty) [function]              // decode at a type
  rule fv(V, T) => decodeF(V, fw(T))

  // ------------------------------------------------------------ binary ops
  syntax Set ::= "BINOPS" [macro]
  rule BINOPS => SetItem("add") SetItem("sub") SetItem("mul") SetItem("div") SetItem("rem")
                 SetItem("min") SetItem("max") SetItem("and") SetItem("or") SetItem("xor")
                 SetItem("shl") SetItem("shr") SetItem("copysign")

  syntax Ty ::= src2Ty(String, Ty) [function, total]   // type of 2nd source
  rule src2Ty("shl", _) => ty("u", 32)
  rule src2Ty("shr", _) => ty("u", 32)
  rule src2Ty(_, T) => T [owise]

  // .relu on integer min/max: negative results become 0
  syntax Int ::= relu(Set, Ty, Int) [function, total]
  rule relu(M, T, V) => 0 requires "relu" in M andBool isS(T) andBool ival(V, T) <Int 0
  rule relu(_, _, V) => V [owise]

  syntax Int ::= bin(String, Set, Ty, Int, Int) [function]
  // floating point
  rule bin("add", M, T, A, B) => fpAdd(M, fw(T), A, B) requires isF(T)
  rule bin("sub", M, T, A, B) => fpSub(M, fw(T), A, B) requires isF(T)
  rule bin("mul", M, T, A, B) => fpMul(M, fw(T), A, B) requires isF(T)
  rule bin("div", M, T, A, B) => fpDiv(M, fw(T), A, B) requires isF(T)
  rule bin("min", M, T, A, B) => fpMinMax(M, fw(T), A, B, true) requires isF(T)
  rule bin("max", M, T, A, B) => fpMinMax(M, fw(T), A, B, false) requires isF(T)
  rule bin("copysign", _, T, A, B) =>      // copysign.f32 d, a, b: |b| with sign of a
       (B &Int mask(wid(T) -Int 1)) |Int (A &Int (1 <<Int (wid(T) -Int 1))) requires isF(T)
  // integers (two's complement, wrap-around unless .sat)
  rule bin("add", M, T, A, B) => trunc(#if "sat" in M #then clamp(ival(A, T) +Int ival(B, T), T)
                                       #else A +Int B #fi, wid(T)) requires notBool isF(T)
  rule bin("sub", M, T, A, B) => trunc(#if "sat" in M #then clamp(ival(A, T) -Int ival(B, T), T)
                                       #else A -Int B #fi, wid(T)) requires notBool isF(T)
  rule bin("mul", M, T, A, B) => trunc(ival(A, T) *Int ival(B, T), 2 *Int wid(T))
    requires notBool isF(T) andBool "wide" in M
  rule bin("mul", M, T, A, B) => trunc((ival(A, T) *Int ival(B, T)) >>Int wid(T), wid(T))
    requires notBool isF(T) andBool "hi" in M
  rule bin("mul", M, T, A, B) => trunc(ival(A, T) *Int ival(B, T), wid(T))
    requires notBool isF(T) andBool notBool "wide" in M andBool notBool "hi" in M
  // division by zero is unspecified in PTX; we return all ones (as hardware does)
  rule bin("div", _, T, A, B) => #if ival(B, T) ==Int 0 #then mask(wid(T))
                                 #else trunc(ival(A, T) /Int ival(B, T), wid(T)) #fi
    requires notBool isF(T)
  rule bin("rem", _, T, A, B) => #if ival(B, T) ==Int 0 #then A
                                 #else trunc(ival(A, T) %Int ival(B, T), wid(T)) #fi
    requires notBool isF(T)
  rule bin("min", M, T, A, B) => relu(M, T, #if ival(A, T) <=Int ival(B, T) #then A #else B #fi) requires notBool isF(T)
  rule bin("max", M, T, A, B) => relu(M, T, #if ival(A, T) >=Int ival(B, T) #then A #else B #fi) requires notBool isF(T)
  rule bin("and", _, _, A, B) => A &Int B
  rule bin("or",  _, _, A, B) => A |Int B
  rule bin("xor", _, _, A, B) => A xorInt B
  rule bin("shl", _, T, A, B) => #if B >=Int wid(T) #then 0 #else trunc(A <<Int B, wid(T)) #fi
  rule bin("shr", _, T, A, B) => trunc(ival(A, T) >>Int minInt(B, wid(T)), wid(T))

  // ------------------------------------------------------------ unary ops
  syntax Set ::= "UNOPS" [macro]
  rule UNOPS => SetItem("neg") SetItem("abs") SetItem("not") SetItem("cnot") SetItem("popc")
                SetItem("clz") SetItem("brev") SetItem("sqrt") SetItem("rsqrt") SetItem("rcp")
                SetItem("sin") SetItem("cos") SetItem("lg2") SetItem("ex2") SetItem("tanh")

  syntax Int ::= un(String, Set, Ty, Int) [function]
  rule un("neg", M, T, A) => ftzIn(M, fw(T), A) xorInt (1 <<Int (cw(fw(T)) -Int 1)) requires isF(T)
  rule un("abs", M, T, A) => ftzIn(M, fw(T), A) &Int mask(cw(fw(T)) -Int 1)         requires isF(T)
  rule un("sqrt",  M, T, A) => fpSqrt(M, fw(T), A) requires isF(T)
  rule un("rsqrt", M, T, A) => post(M, fw(T), fc(1.0, fv(ftzIn(M, fw(T), A), T)) /Float sqrtF(fv(ftzIn(M, fw(T), A), T))) requires isF(T)
  rule un("rcp",   M, T, A) => fpRcp(M, fw(T), A) requires isF(T)
  rule un("sin",   M, T, A) => post(M, fw(T), sinFloat(fv(ftzIn(M, fw(T), A), T))) requires isF(T)
  rule un("cos",   M, T, A) => post(M, fw(T), cosFloat(fv(ftzIn(M, fw(T), A), T))) requires isF(T)
  rule un("lg2",   M, T, A) => post(M, fw(T), logFloat(fv(ftzIn(M, fw(T), A), T)) /Float logFloat(fc(2.0, fv(A, T)))) requires isF(T)
  rule un("ex2",   M, T, A) => post(M, fw(T), fc(2.0, fv(A, T)) ^Float fv(ftzIn(M, fw(T), A), T)) requires isF(T)
  rule un("tanh",  M, T, A) => post(M, fw(T), tanhF(fv(ftzIn(M, fw(T), A), T))) requires isF(T)
  rule un("neg",  _, T, A) => trunc(0 -Int A, wid(T))            requires notBool isF(T)
  rule un("abs",  _, T, A) => trunc(absInt(ival(A, T)), wid(T))  requires notBool isF(T)
  rule un("not",  _, T, A) => trunc(~Int A, wid(T))
  rule un("cnot", _, _, A) => b2i(A ==Int 0)
  rule un("popc", _, _, A) => popc(A)
  rule un("clz",  _, T, A) => clz(A, wid(T))
  rule un("brev", _, T, A) => brev(A, wid(T))

  // tanh(NaN) = NaN; below 2^-12 tanh(x) = x within 2^-24 (the difference of exponentials cancels there)
  syntax Float ::= tanhF(Float) [function]
  rule tanhF(F) => F requires isNaN(F) orBool absFloat(F) <Float fc(0.000244140625, F)
  rule tanhF(F) => (expFloat(F) -Float expFloat(--Float F)) /Float (expFloat(F) +Float expFloat(--Float F))
    requires notBool isNaN(F) andBool absFloat(F) >=Float fc(0.000244140625, F) andBool absFloat(F) <Float fc(20.0, F)
  rule tanhF(F) => #if signFloat(F) #then fc(-1.0, F) #else fc(1.0, F) #fi [owise]

  // ------------------------------------------------------------ ternary ops
  syntax Set ::= "TERNOPS" [macro]
  rule TERNOPS => SetItem("mad") SetItem("fma") SetItem("selp")

  syntax Ty ::= src3Ty(String, Set, Ty) [function, total]
  rule src3Ty("selp", _, _) => ty("p", 1)
  rule src3Ty("mad", M, T) => wide(T) requires "wide" in M
  rule src3Ty(_, _, T) => T [owise]

  syntax Int ::= tern(String, Set, Ty, Int, Int, Int) [function]
  rule tern("selp", _, _, A, B, P) => #if P =/=Int 0 #then A #else B #fi
  rule tern(OP, M, T, A, B, C) => fpFma(M, fw(T), A, B, C) requires isF(T) andBool OP in (SetItem("fma") SetItem("mad"))
  rule tern("mad", M, T, A, B, C) => trunc(ival(A, T) *Int ival(B, T) +Int C, 2 *Int wid(T))
    requires notBool isF(T) andBool "wide" in M
  rule tern("mad", M, T, A, B, C) => trunc(((ival(A, T) *Int ival(B, T)) >>Int wid(T)) +Int C, wid(T))
    requires notBool isF(T) andBool "hi" in M andBool notBool "sat" in M
  // mad.hi.sat.s32: the sum saturates to the signed 32-bit range
  rule tern("mad", M, T, A, B, C) => trunc(clamp(((ival(A, T) *Int ival(B, T)) >>Int wid(T)) +Int ival(C, T), T), wid(T))
    requires notBool isF(T) andBool "hi" in M andBool "sat" in M
  rule tern("mad", M, T, A, B, C) => trunc(ival(A, T) *Int ival(B, T) +Int C, wid(T))
    requires notBool isF(T) andBool notBool "wide" in M andBool notBool "hi" in M

  // ------------------------------------------------------------ comparison
  syntax String ::= cmpOp(Set) [function, total]
  rule cmpOp(M) => pick(M, ListItem("eq") ListItem("ne") ListItem("lt") ListItem("le")
                           ListItem("gt") ListItem("ge") ListItem("lo") ListItem("ls")
                           ListItem("hi") ListItem("hs") ListItem("equ") ListItem("neu")
                           ListItem("ltu") ListItem("leu") ListItem("gtu") ListItem("geu")
                           ListItem("num") ListItem("nan"), "eq")
  syntax String ::= boolOp(Set) [function, total]
  rule boolOp(M) => pick(M, ListItem("and") ListItem("or") ListItem("xor"), "none")

  syntax Int ::= cmp(String, Ty, Int, Int) [function]
  syntax Bool ::= cmpI(String, Int, Int, Int, Int) [function]  // op, signed-view a b, raw a b
                | cmpF(String, Float, Float) [function]
  rule cmp(OP, T, A, B) => b2i(cmpF(OP, fv(A, T), fv(B, T))) requires isF(T)
  rule cmp(OP, T, A, B) => b2i(cmpI(OP, ival(A, T), ival(B, T), trunc(A, wid(T)), trunc(B, wid(T)))) [owise]
  rule cmpI("eq", X, Y, _, _) => X ==Int Y
  rule cmpI("ne", X, Y, _, _) => X =/=Int Y
  rule cmpI("lt", X, Y, _, _) => X <Int Y
  rule cmpI("le", X, Y, _, _) => X <=Int Y
  rule cmpI("gt", X, Y, _, _) => X >Int Y
  rule cmpI("ge", X, Y, _, _) => X >=Int Y
  rule cmpI("lo", _, _, X, Y) => X <Int Y
  rule cmpI("ls", _, _, X, Y) => X <=Int Y
  rule cmpI("hi", _, _, X, Y) => X >Int Y
  rule cmpI("hs", _, _, X, Y) => X >=Int Y
  rule cmpF("num", X, Y) => notBool (isNaN(X) orBool isNaN(Y))
  rule cmpF("nan", X, Y) => isNaN(X) orBool isNaN(Y)
  rule cmpF(OP, X, Y) => false
    requires (isNaN(X) orBool isNaN(Y)) andBool lengthString(OP) ==Int 2
  rule cmpF(OP, X, Y) => true
    requires (isNaN(X) orBool isNaN(Y)) andBool lengthString(OP) ==Int 3
     andBool OP =/=String "num" andBool OP =/=String "nan"
  rule cmpF(OP, X, Y) => cmpOrd(substrString(OP, 0, 2), X, Y)
    requires notBool (isNaN(X) orBool isNaN(Y)) andBool OP =/=String "num" andBool OP =/=String "nan"
  syntax Bool ::= cmpOrd(String, Float, Float) [function]
  rule cmpOrd("eq", X, Y) => X ==Float Y
  rule cmpOrd("ne", X, Y) => notBool X ==Float Y
  rule cmpOrd("lt", X, Y) => X <Float Y
  rule cmpOrd("le", X, Y) => X <=Float Y
  rule cmpOrd("gt", X, Y) => X >Float Y
  rule cmpOrd("ge", X, Y) => X >=Float Y

  syntax Int ::= bop(Int, String, Int) [function, total]
  rule bop(V, "and", C) => V &Int C
  rule bop(V, "or",  C) => V |Int C
  rule bop(V, "xor", C) => V xorInt C
  rule bop(V, _, _) => V [owise]

  // ------------------------------------------------------------ conversion
  syntax String ::= intRnd(Set, String) [function, total]
  rule intRnd(M, D) => pick(M, ListItem("rni") ListItem("rzi") ListItem("rmi") ListItem("rpi"), D)

  // cvt.[rnd][.ftz][.sat|.relu|.satfinite].dtype.stype
  //  float -> float: exact wide value, rounded by fpFin (any of .rn .rz .rm .rp .rna), or
  //                  rounded to an integral value in the same format with .rni/.rzi/.rmi/.rpi
  //  int   -> float: exact, then rounded;  float -> int: rounded to integral (default .rzi),
  //                  NaN -> 0, saturating;  int -> int: sign/zero extension, optional .sat
  syntax Int ::= cvt(Set, Ty, Ty, Int) [function]          // modifiers, dst, src, bits
  // .ftz applies only to .f32 inputs and results (PTX ISA, cvt: "Modifier .ftz can only be specified when either .dtype or
  // .atype is .f32 and applies only to single precision (.f32) inputs and results")
  syntax Set ::= ftz32(Set, Ty) [function, total]
  rule ftz32(M, T) => M requires fw(T) ==Int 32
  rule ftz32(M, _) => M -Set SetItem("ftz") [owise]
  rule cvt(M, D, S, V) => post(ftz32(M, D), fw(D), roundFloat(rndInt(fv(ftzIn(ftz32(M, S), fw(S), V), S), intRnd(M, "")), prec(fw(D)), ebits(fw(D))))
    requires isF(D) andBool isF(S) andBool intRnd(M, "") =/=String ""
  rule cvt(M, D, S, V) => fpFin(ftz32(M, D), fw(D), roundFloat(fv(ftzIn(ftz32(M, S), fw(S), V), S), WP, WEXP), 0)
    requires isF(D) andBool isF(S) andBool intRnd(M, "") ==String ""
  rule cvt(M, D, S, V) => fpFin(ftz32(M, D), fw(D), Int2Float(ival(V, S), WP, WEXP), 0)
    requires isF(D) andBool notBool isF(S)
  rule cvt(M, D, S, V) => f2i(fv(ftzIn(ftz32(M, S), fw(S), V), S), intRnd(M, "rzi"), D)
    requires notBool isF(D) andBool isF(S)
  rule cvt(M, D, S, V) => trunc(#if "sat" in M #then clamp(ival(V, S), D) #else ival(V, S) #fi, wid(D))
    requires notBool isF(D) andBool notBool isF(S)

  // ------------------------------------------------------------ atomics
  syntax String ::= atomOp(Set) [function, total]
  rule atomOp(M) => pick(M, ListItem("add") ListItem("exch") ListItem("cas") ListItem("min")
                            ListItem("max") ListItem("and") ListItem("or") ListItem("xor")
                            ListItem("inc") ListItem("dec"), "add")

  syntax Int ::= atomNew(String, Ty, Int, Int, Int) [function]   // op, type, old, b, c
  rule atomNew("add", T, O, B, _) => bin("add", .Set, T, O, B)
  rule atomNew("min", T, O, B, _) => bin("min", .Set, T, O, B)
  rule atomNew("max", T, O, B, _) => bin("max", .Set, T, O, B)
  rule atomNew("and", T, O, B, _) => bin("and", .Set, T, O, B)
  rule atomNew("or",  T, O, B, _) => bin("or",  .Set, T, O, B)
  rule atomNew("xor", T, O, B, _) => bin("xor", .Set, T, O, B)
  rule atomNew("exch", _, _, B, _) => B
  rule atomNew("cas", _, O, B, C) => #if O ==Int B #then C #else O #fi
  rule atomNew("inc", T, O, B, _) => #if O >=Int B #then 0 #else trunc(O +Int 1, wid(T)) #fi
  rule atomNew("dec", T, O, B, _) => #if O ==Int 0 orBool O >Int B #then B #else trunc(O -Int 1, wid(T)) #fi
endmodule

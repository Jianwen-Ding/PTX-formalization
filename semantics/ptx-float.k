// =============================================================================
// PTX-FLOAT: IEEE-754 binary16/32/64 arithmetic with all PTX rounding modes
// (.rn .rz .rm .rp), .ftz and .sat.
//
// Round-to-nearest-even is computed directly with K's builtin FLOAT (MPFR) at
// the operand format.  The directed modes cannot be obtained that way, so they
// are computed from an *exact* (or sticky-corrected) wide result:
//
//   * add, sub, mul, fma: computed exactly in a 4400-bit, 20-bit-exponent
//     format (every f16/f32/f64 sum/product fits), error sign 0;
//   * div, sqrt: quotient / root in 300 bits (round to nearest) plus the sign
//     `s` of (true value - computed value), obtained exactly by multiplying
//     back (a - q*b, a - q*q);
//
// and then rounded to the target format by `roundDirF`, which picks between
// the two neighbouring floats using only exact comparisons.
// =============================================================================
requires "ptx-common.k"

module PTX-FLOAT
  imports PTX-COMMON

  // ------------------------------------------------------------ modifiers
  syntax String ::= rndOf(Set) [function, total]
  rule rndOf(M) => pick(M, ListItem("rn") ListItem("rz") ListItem("rm") ListItem("rp") ListItem("rna"), "rn")

  syntax Int ::= "WP" [macro]       // exact working precision
  rule WP => 4400
  syntax Int ::= "QP" [macro]       // quotient / root precision
  rule QP => 300

  syntax Float ::= wf(Int, Int, Int) [function, total]     // bits, width, precision
  rule wf(B, W, P) => roundFloat(decodeF(B, W), P, WEXP)

  // ------------------------------------------------------------ ftz
  syntax Bool ::= isSubnB(Int, Int) [function, total]       // nonzero subnormal bit pattern
  rule isSubnB(B, W) => ((B >>Int (prec(W) -Int 1)) &Int mask(ebits(W))) ==Int 0
                        andBool (B &Int mask(prec(W) -Int 1)) =/=Int 0
  syntax Int ::= flushB(Int, Int) [function, total]          // sign-preserving flush
  rule flushB(B, W) => B &Int (1 <<Int (W -Int 1)) requires isSubnB(B, W)
  rule flushB(B, _) => B [owise]
  syntax Int ::= ftzIn(Set, Int, Int)  [function, total]
               | ftzOut(Set, Int, Int) [function, total]
  rule ftzIn(M, W, B)  => flushB(B, W) requires "ftz" in M
  rule ftzIn(_, _, B)  => B [owise]
  rule ftzOut(M, W, B) => flushB(B, W) requires "ftz" in M
  rule ftzOut(_, _, B) => B [owise]

  // result of an operation: apply .sat, encode, apply .ftz to the result
  syntax Int ::= post(Set, Int, Float) [function, total]
  rule post(M, W, F) => ftzOut(M, W, encodeF(#if "sat" in M #then satF(F) #else F #fi, W))

  // ------------------------------------------------------------ neighbours
  syntax Float ::= nextUpF(Float, Int)   [function, total]
                 | nextDownF(Float, Int) [function, total]
  rule nextUpF(F, _) => F requires isNaN(F) orBool (isInfinite(F) andBool notBool signFloat(F))
  rule nextUpF(F, W) => decodeF(1, W) requires notBool isNaN(F) andBool F ==Float fc(0.0, F)
  rule nextUpF(F, W) => decodeF(encodeF(F, W) +Int 1, W)
    requires notBool isNaN(F) andBool notBool isInfinite(F) andBool F >Float fc(0.0, F)
  rule nextUpF(F, W) => decodeF(encodeF(F, W) -Int 1, W)
    requires notBool isNaN(F) andBool F <Float fc(0.0, F)
  rule nextDownF(F, W) => --Float nextUpF(--Float F, W)

  // ------------------------------------------------------------ directed rounding
  // v: wide value, s: sign of (true value - v) in {-1, 0, 1}
  syntax Float ::= roundDirF(Float, Int, String, Int) [function, total]
                 | rdPick(Float, Int, String, Float, Float, Int) [function, total]   // v, s, mode, down, up, width
  rule roundDirF(V, S, Mode, W) => rdFind(V, S, Mode, W, roundFloat(V, prec(W), ebits(W)))
  syntax Float ::= rdFind(Float, Int, String, Int, Float) [function, total]
  rule rdFind(V, S, Mode, W, R) => #if S ==Int 0 #then R
                                   #else rdPick(V, S, Mode, #if S >Int 0 #then R #else nextDownF(R, W) #fi,
                                                           #if S >Int 0 #then nextUpF(R, W) #else R #fi, W) #fi
    requires R ==Float V
  rule rdFind(V, S, Mode, W, R) => rdPick(V, S, Mode, R, nextUpF(R, W), W)
    requires R <Float V
  rule rdFind(V, S, Mode, W, R) => rdPick(V, S, Mode, nextDownF(R, W), R, W)
    requires R >Float V
  rule rdFind(V, _, _, _, R) => R requires isNaN(V) orBool isInfinite(V)

  rule rdPick(_, _, "rm", D, _, _) => D
  rule rdPick(_, _, "rp", _, U, _) => U
  rule rdPick(V, S, "rz", D, U, _) => #if V >Float fc(0.0, V) orBool (V ==Float fc(0.0, V) andBool S >Int 0)
                                    #then D #else U #fi
  // nearest: exact gap comparison; ties go by the sticky sign, else away ("rna") / even ("rn")
  rule rdPick(V, S, Mode, D, U, W) => #if V -Float D <Float U -Float V #then D
                                    #else #if V -Float D >Float U -Float V #then U
                                    #else rdTie(V, S, Mode, D, U, W) #fi #fi
    requires Mode ==String "rna" orBool Mode ==String "rn"
  syntax Float ::= rdTie(Float, Int, String, Float, Float, Int) [function, total]
  rule rdTie(_, S, _, D, U, _) => #if S >Int 0 #then U #else D #fi requires S =/=Int 0
  rule rdTie(V, 0, "rna", D, U, _) => #if V >Float fc(0.0, V) #then U #else D #fi
  rule rdTie(_, 0, _, D, U, W) => #if encodeF(D, W) %Int 2 ==Int 0 #then D #else U #fi [owise]

  // ------------------------------------------------------------ operations
  // Each takes the instruction modifiers, the format width and operand bit patterns.
  syntax Int ::= fpAdd(Set, Int, Int, Int)      [function, total]
               | fpSub(Set, Int, Int, Int)      [function, total]
               | fpMul(Set, Int, Int, Int)      [function, total]
               | fpDiv(Set, Int, Int, Int)      [function, total]
               | fpSqrt(Set, Int, Int)          [function, total]
               | fpFma(Set, Int, Int, Int, Int) [function, total]
               | fpRcp(Set, Int, Int)           [function, total]
               | fpFin(Set, Int, Float, Int)    [function, total]   // wide value + sticky -> bits
               | fpAddW(Set, Int, Float, Float, Bool) [function, total]
               | fpDivW(Set, Int, Float, Float) [function, total]

  // finishing a wide result v (already exact or with sticky sign s)
  rule fpFin(_, W, V, _) => encodeF(qnan(W), W)  requires isNaN(V)
  rule fpFin(M, W, V, _) => post(M, W, roundFloat(V, prec(W), ebits(W))) requires isInfinite(V)
  rule fpFin(M, W, V, S) => post(M, W, roundDirF(V, S, rndOf(M), W))
    requires notBool isNaN(V) andBool notBool isInfinite(V)

  // add: -0 appears only as (-0)+(-0), or as an exact cancellation under .rm
  rule fpAdd(M, W, A, B) => post(M, W, decodeF(ftzIn(M, W, A), W) +Float decodeF(ftzIn(M, W, B), W))
    requires rndOf(M) ==String "rn"
  rule fpAdd(M, W, A, B) => fpAddW(M, W, wf(ftzIn(M, W, A), W, WP), wf(ftzIn(M, W, B), W, WP), false)
    requires rndOf(M) =/=String "rn"
  rule fpSub(M, W, A, B) => fpAdd(M, W, A, B xorInt (1 <<Int (W -Int 1)))
  rule fpAddW(M, W, X, Y, _) => fpFin(M, W, #if (signFloat(X) =/=Bool signFloat(Y)) andBool X +Float Y ==Float fc(0.0, X)
                                              andBool rndOf(M) ==String "rm" #then --Float fc(0.0, X)
                                              #else X +Float Y #fi, 0)

  rule fpMul(M, W, A, B) => post(M, W, decodeF(ftzIn(M, W, A), W) *Float decodeF(ftzIn(M, W, B), W))
    requires rndOf(M) ==String "rn"
  rule fpMul(M, W, A, B) => fpFin(M, W, wf(ftzIn(M, W, A), W, WP) *Float wf(ftzIn(M, W, B), W, WP), 0)
    requires rndOf(M) =/=String "rn"

  // fma: a*b exact, then + c exact
  rule fpFma(M, W, A, B, C) => post(M, W, fmaF(decodeF(ftzIn(M, W, A), W), decodeF(ftzIn(M, W, B), W),
                                              decodeF(ftzIn(M, W, C), W)))
    requires rndOf(M) ==String "rn"
  rule fpFma(M, W, A, B, C) => fpFin(M, W, fmaZero(M, wf(ftzIn(M, W, A), W, WP), wf(ftzIn(M, W, B), W, WP),
                                                 wf(ftzIn(M, W, C), W, WP)), 0)
    requires rndOf(M) =/=String "rn"
  syntax Float ::= fmaZero(Set, Float, Float, Float) [function, total]
  rule fmaZero(M, X, Y, Z) =>
       #if X *Float Y +Float Z ==Float fc(0.0, X) andBool rndOf(M) ==String "rm"
           andBool (signFloat(X *Float Y) =/=Bool signFloat(Z)) andBool notBool isNaN(X *Float Y +Float Z)
       #then --Float fc(0.0, X) #else X *Float Y +Float Z #fi

  // div: q to 300 bits, sticky from the exact residual a - q*b
  rule fpDiv(M, W, A, B) => post(M, W, decodeF(ftzIn(M, W, A), W) /Float decodeF(ftzIn(M, W, B), W))
    requires rndOf(M) ==String "rn"
  rule fpDiv(M, W, A, B) => fpDivW(M, W, wf(ftzIn(M, W, A), W, QP), wf(ftzIn(M, W, B), W, QP))
    requires rndOf(M) =/=String "rn"
  rule fpDivW(M, W, X, Y) => fpFin(M, W, X /Float Y, divSticky(X, Y, X /Float Y))
  syntax Int ::= divSticky(Float, Float, Float) [function, total]
  rule divSticky(X, Y, Q) => sgn(roundFloat(X, WP, WEXP) -Float roundFloat(Q, WP, WEXP) *Float roundFloat(Y, WP, WEXP))
                              *Int #if signFloat(Y) #then -1 #else 1 #fi
    requires notBool isNaN(Q) andBool notBool isInfinite(Q) andBool notBool isNaN(X) andBool notBool isInfinite(X)
     andBool notBool isNaN(Y) andBool notBool isInfinite(Y) andBool Q =/=Float fc(0.0, Q)
  rule divSticky(_, _, _) => 0 [owise]
  syntax Int ::= sgn(Float) [function, total]
  rule sgn(F) => 0 requires isNaN(F) orBool F ==Float fc(0.0, F)
  rule sgn(F) => 1 requires notBool isNaN(F) andBool F >Float fc(0.0, F)
  rule sgn(F) => -1 requires notBool isNaN(F) andBool F <Float fc(0.0, F)

  // sqrt
  rule fpSqrt(M, W, A) => post(M, W, sqrtF(decodeF(ftzIn(M, W, A), W)))
    requires rndOf(M) ==String "rn"
  rule fpSqrt(M, W, A) => fpFin(M, W, sqrtF(wf(ftzIn(M, W, A), W, QP)), sqrtSticky(wf(ftzIn(M, W, A), W, QP), sqrtF(wf(ftzIn(M, W, A), W, QP))))
    requires rndOf(M) =/=String "rn"
  syntax Int ::= sqrtSticky(Float, Float) [function, total]
  rule sqrtSticky(X, Q) => sgn(roundFloat(X, WP, WEXP) -Float roundFloat(Q, WP, WEXP) *Float roundFloat(Q, WP, WEXP))
    requires notBool isNaN(Q) andBool notBool isInfinite(Q) andBool Q >Float fc(0.0, Q)
  rule sqrtSticky(_, _) => 0 [owise]
  // IEEE: sqrt(-0) = -0 (MPFR's rootn returns +0)
  syntax Float ::= sqrtF(Float) [function]
  rule sqrtF(F) => F requires notBool isNaN(F) andBool F ==Float fc(0.0, F)
  rule sqrtF(F) => rootFloat(F, 2) [owise]

  rule fpRcp(M, W, A) => fpDiv(M, W, encodeF(Int2Float(1, prec(W), ebits(W)), W), A)
endmodule

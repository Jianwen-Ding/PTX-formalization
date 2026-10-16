// =============================================================================
// PTX-EXEC-PRED: set, slct, testp.
//
//   set.CmpOp[.BoolOp][.ftz].dtype.stype d, a, b[, {!}c]
//        d = all-ones / 0 (integer dtype) or 1.0 / 0.0 (float dtype)
//   slct[.ftz].dtype.stype d, a, b, c        d = (c >= 0) ? a : b
//   testp.{finite,infinite,number,notanumber,normal,subnormal}.type p, a
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-PRED
  imports PTX-EXEC-CORE

  // flush a subnormal float input when .ftz
  syntax Int ::= ftzSrc(Set, Ty, Int) [function, total]
  rule ftzSrc(M, T, V) => flushB(V, wid(T)) requires "ftz" in M andBool isF(T)
  rule ftzSrc(_, _, V) => V [owise]

  // ---------------------------------------------------------------- set
  syntax Int ::= setRes(Bool, Ty) [function, total]               // truth, destination type
  rule setRes(B, T) => #if B #then #if isF(T) #then encodeF(Int2Float(1, prec(wid(T)), ebits(wid(T))), wid(T))
                                   #else mask(wid(T)) #fi
                       #else 0 #fi

  rule <tk> exec(opc("set", M, ListItem(DT) ListItem(ST)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, setRes(cmp(cmpOp(M), ST, ftzSrc(M, ST, rd(A, ST, C)), ftzSrc(M, ST, rd(B, ST, C))) =/=Int 0, DT)) </regs>
  rule <tk> exec(opc("set", M, ListItem(DT) ListItem(ST)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, setRes(bop(cmp(cmpOp(M), ST, ftzSrc(M, ST, rd(A, ST, C)), ftzSrc(M, ST, rd(B, ST, C))),
                                      boolOp(M), rd(X, ty("p", 1), C)) =/=Int 0, DT)) </regs>

  // ---------------------------------------------------------------- slct
  syntax Bool ::= nonNeg(Ty, Int) [function, total]
  rule nonNeg(T, V) => notBool isNaN(decodeF(V, wid(T))) andBool decodeF(V, wid(T)) >=Float fc(0.0, decodeF(V, wid(T)))
    requires isF(T)
  rule nonNeg(T, V) => ival(V, T) >=Int 0 [owise]

  rule <tk> exec(opc("slct", M, ListItem(DT) ListItem(ST)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, #if nonNeg(ST, ftzSrc(M, ST, rd(X, ST, C))) #then rd(A, DT, C) #else rd(B, DT, C) #fi) </regs>

  // ---------------------------------------------------------------- testp
  syntax Bool ::= testp(String, Int, Int) [function, total]       // op, bits, width
  rule testp("finite", V, W)      => expField(V, W) =/=Int mask(ebits(W))
  rule testp("infinite", V, W)    => expField(V, W) ==Int mask(ebits(W)) andBool manField(V, W) ==Int 0
  rule testp("number", V, W)      => notBool (expField(V, W) ==Int mask(ebits(W)) andBool manField(V, W) =/=Int 0)
  rule testp("notanumber", V, W)  => expField(V, W) ==Int mask(ebits(W)) andBool manField(V, W) =/=Int 0
  rule testp("normal", V, W)      => expField(V, W) =/=Int 0 andBool expField(V, W) =/=Int mask(ebits(W))
  rule testp("subnormal", V, W)   => expField(V, W) ==Int 0 andBool manField(V, W) =/=Int 0
  rule testp(_, _, _) => false [owise]
  syntax Int ::= expField(Int, Int) [function, total] | manField(Int, Int) [function, total]
  rule expField(V, W) => (V >>Int (prec(W) -Int 1)) &Int mask(ebits(W))
  rule manField(V, W) => V &Int mask(prec(W) -Int 1)

  rule <tk> exec(opc("testp", M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, b2i(testp(pick(M, ListItem("finite") ListItem("infinite") ListItem("number")
                                              ListItem("notanumber") ListItem("normal") ListItem("subnormal"), ""),
                                      rd(A, T, C), wid(T)))) </regs>
endmodule

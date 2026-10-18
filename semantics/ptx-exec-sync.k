// =============================================================================
// PTX-EXEC-SYNC: CTA barriers
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-SYNC
  imports PTX-EXEC-CORE

  // ------------------------------------------------------------ barriers
  // bar{.cta}.sync / barrier{.cta}.sync{.aligned}  a{, b}
  // A thread arriving at barrier `a` records the barrier's current generation;
  // once the expected number of threads (b, or all live threads of the CTA)
  // has arrived the generation is bumped, releasing every waiter.
  syntax Int ::= clearBit(Int, Int) [function, total]
  rule clearBit(M, L) => M &Int ~Int laneBit(L)
  syntax KItem ::= #bar(Int, Int, Bool)            // id, expected (-1: all), wait?
                 | #barCheck(Int) | #barCheckAll(List)

  rule <tk> #gx(opc(OP, M, _), ops(A, .Operands), C) => #bar(rd(A, ty("u", 32), C), -1, notBool "arrive" in M) ... </tk>
    requires isBar(OP, M)
  rule <tk> #gx(opc(OP, M, _), ops(A, N, .Operands), C) =>
            #bar(rd(A, ty("u", 32), C), rd(N, ty("u", 32), C), notBool "arrive" in M) ... </tk>
    requires isBar(OP, M)
  // bar.warp.sync is a warp collective with no data (ptx-exec-warp.k)

  // bar.red.popc.u32 d, a{, b}, {!}c     bar.red.{and,or}.pred p, a{, b}, {!}c
  // Every arriving thread contributes its predicate; once the barrier completes all
  // participants receive the count of true predicates (popc) or their and / or.
  syntax KItem ::= #barRed(Int, Int, String, Int, Operand) | #barWaitRed(Int, Int, Operand)
  syntax String ::= redKind(Set) [function, total]
  rule redKind(M) => pick(M, ListItem("popc") ListItem("and") ListItem("or"), "popc")
  syntax Int ::= redStep(String, Int, Int) [function, total]      // kind, accumulator (or -1 = none yet), predicate value
  rule redStep("popc", -1, V) => V
  rule redStep("popc", A, V) => A +Int V requires A >=Int 0
  rule redStep("and", -1, V) => V
  rule redStep("and", A, V) => A &Int V requires A >=Int 0
  rule redStep("or", -1, V) => V
  rule redStep("or", A, V) => A |Int V requires A >=Int 0
  rule redStep(_, _, V) => V [owise]
  rule <tk> #gx(opc(OP, M, _), ops(D, A, Pc, .Operands), C) =>
            #barRed(rd(A, ty("u", 32), C), -1, redKind(M), b2i(rd(Pc, ty("p", 1), C) =/=Int 0), D) ... </tk>
    requires isBarRed(OP, M)
  rule <tk> #gx(opc(OP, M, _), ops(D, A, N, Pc, .Operands), C) =>
            #barRed(rd(A, ty("u", 32), C), rd(N, ty("u", 32), C), redKind(M), b2i(rd(Pc, ty("p", 1), C) =/=Int 0), D) ... </tk>
    requires isBarRed(OP, M)
  rule <tk> #barRed(B, X, K, V, D) => #barCheck(B) ~> #yield ~> #barWaitRed(B, {BG[B] orDefault 0}:>Int, D) ... </tk>
       <status> running => waiting(B, {BG[B] orDefault 0}:>Int) </status>
       <barGen> BG </barGen>
       <barCnt> BC => BC[B <- {BC[B] orDefault 0}:>Int +Int 1] </barCnt>
       <barExp> BX => BX[B <- X] </barExp>
       <barVal> BV => BV[B <- redStep(K, {BV[B] orDefault -1}:>Int, V)] </barVal>
  rule <tk> #barWaitRed(B, G, D) => .K ... </tk>
       <status> running </status>
       <barGen> BG </barGen>
       <barRes> BR </barRes>
       <regs> R => wr(R, D, {BR[B] orDefault 0}:>Int) </regs>
    requires {BG[B] orDefault 0}:>Int >Int G

  rule <tk> #bar(B, X, W) => #barCheck(B) ~> #if W #then #yield #else .K #fi ... </tk>
       <status> running => #if W #then waiting(B, {BG[B] orDefault 0}:>Int) #else running #fi </status>
       <barGen> BG </barGen>
       <barCnt> BC => BC[B <- {BC[B] orDefault 0}:>Int +Int 1] </barCnt>
       <barExp> BX => BX[B <- X] </barExp>
  rule <tk> #barCheck(B) => .K ... </tk>
       <barCnt> BC => BC[B <- undef] </barCnt>
       <barVal> BV => BV[B <- undef] </barVal>
       <barRes> BR => BR[B <- {BV[B] orDefault 0}:>Int] </barRes>
       <barGen> BG => BG[B <- {BG[B] orDefault 0}:>Int +Int 1] </barGen>
       <barExp> BX </barExp>
       <live> L </live>
    requires B in_keys(BC)
     andBool {BC[B]}:>Int >=Int #if {BX[B] orDefault -1}:>Int <Int 0 #then L #else {BX[B]}:>Int #fi
  rule <tk> #barCheck(_) => .K ... </tk> [owise]
  rule <status> waiting(B, G) => running </status>
       <barGen> BG </barGen>
       <turn> -1 </turn>
    requires {BG[B] orDefault 0}:>Int >Int G

  // ------------------------------------------------------------ thread exit
  rule <tk> #exit => #barCheckAll(keys_list(BC)) ~> #clExit ~> #die ... </tk>
       <tid> T </tid>
       <wlive> WL => WL[T /Int 32 <- clearBit({WL[T /Int 32] orDefault 0}:>Int, T %Int 32)] </wlive>
       <live> L => L -Int 1 </live>
       <barCnt> BC </barCnt> <sb> .List </sb>
  rule <tk> #barCheckAll(ListItem(B:Int) Bs) => #barCheck(B) ~> #barCheckAll(Bs) ... </tk>
  rule <tk> #barCheckAll(.List) => .K ... </tk>
  rule <threads> ... (<thread> <tid> _ </tid> <tk> #die ... </tk> ... </thread> => .Bag) ... </threads>
       <turn> _ => -1 </turn>
  rule <ctas> ... (<cta> <ctaid> _ </ctaid> <live> 0 </live> <threads> .Bag </threads> ... </cta> => .Bag) ... </ctas>

  // anything else is outside the modelled subset
endmodule

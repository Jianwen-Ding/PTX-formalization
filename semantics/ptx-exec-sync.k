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
  syntax KItem ::= #bar(Int, Int, Bool)            // id, expected (-1: all), wait?
                 | #barCheck(Int) | #barCheckAll(List)

  rule <tk> exec(opc(OP, M, _), ops(A, .Operands), C) => #bar(rd(A, ty("u", 32), C), -1, notBool "arrive" in M) ... </tk>
    requires isBar(OP, M)
  rule <tk> exec(opc(OP, M, _), ops(A, N, .Operands), C) =>
            #bar(rd(A, ty("u", 32), C), rd(N, ty("u", 32), C), notBool "arrive" in M) ... </tk>
    requires isBar(OP, M)
  // bar.warp.sync: warps are not modelled as lock-step units; no-op
  rule <tk> exec(opc("bar", M, _), _, _) => .K ... </tk> requires "warp" in M

  rule <tk> #bar(B, X, W) => #barCheck(B) ~> #if W #then #yield #else .K #fi ... </tk>
       <status> running => #if W #then waiting(B, {BG[B] orDefault 0}:>Int) #else running #fi </status>
       <barGen> BG </barGen>
       <barCnt> BC => BC[B <- {BC[B] orDefault 0}:>Int +Int 1] </barCnt>
       <barExp> BX => BX[B <- X] </barExp>
  rule <tk> #barCheck(B) => .K ... </tk>
       <barCnt> BC => BC[B <- undef] </barCnt>
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
  rule <tk> #exit => #barCheckAll(keys_list(BC)) ~> #die ... </tk>
       <live> L => L -Int 1 </live>
       <barCnt> BC </barCnt>
  rule <tk> #barCheckAll(ListItem(B:Int) Bs) => #barCheck(B) ~> #barCheckAll(Bs) ... </tk>
  rule <tk> #barCheckAll(.List) => .K ... </tk>
  rule <threads> ... (<thread> <tid> _ </tid> <tk> #die ... </tk> ... </thread> => .Bag) ... </threads>
       <turn> _ => -1 </turn>
  rule <ctas> ... (<cta> <ctaid> _ </ctaid> <live> 0 </live> <threads> .Bag </threads> ... </cta> => .Bag) ... </ctas>

  // anything else is outside the modelled subset
endmodule

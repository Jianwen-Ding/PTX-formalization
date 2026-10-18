// =============================================================================
// PTX-EXEC-CORE: execution context, operand access, scheduling turn and instruction fetch
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-load.k"
requires "ptx-alu.k"

module PTX-EXEC-CORE
  imports PTX-LOAD
  imports PTX-ALU

  syntax Ctx ::= ctx(Map, Map, Map, Map, Map)   // regs, env, syms, labels, declared register widths
  syntax Ops ::= ops(Operands)
  syntax KItem ::= exec(Opc, Ops, Ctx)
                 | #fault(String, Int)            // stuck: run-time error
                 | #unsupported(Opc)              // stuck: not modelled

  // ------------------------------------------------------------ registers
  syntax Int ::= rreg(String, Ctx) [function, total]
  rule rreg(R, ctx(RS, _, _, _, _)) => {RS[R] orDefault 0}:>Int

  syntax Sym ::= lookupSym(String, Ctx) [function, total]
  rule lookupSym(N, ctx(_, E, _, _, _)) => {E[N]}:>Sym requires N in_keys(E)
  rule lookupSym(N, ctx(_, E, S, _, _)) => {S[N]}:>Sym requires notBool N in_keys(E) andBool N in_keys(S)
  rule lookupSym(N, _) => noSym(N) [owise]

  syntax Int ::= symAddr(Sym) [function, total]
  rule symAddr(sym(_, A)) => A
  rule symAddr(noSym(_)) => 0

  // operand value at a type
  syntax Int ::= rd(Operand, Ty, Ctx) [function]
  rule rd(R:Reg, T, C)    => trunc(rreg(Reg2String(R), C), wid(T))
  rule rd(! R:Reg, _, C)  => b2i(rreg(Reg2String(R), C) ==Int 0)
  rule rd(I:Int, T, _)    => intLit(I, T)
  rule rd(F:Float, T, _)  => fltLit(F, T)
  rule rd(H:HexLit, T, _) => hexBits(H, T)
  // a Name is a register if declared with .reg (names need no % prefix), else a symbol
  rule rd(N:Name, T, C)   => trunc(rreg(Name2String(N), C), wid(T)) requires isRegName(N, C)
  rule rd(! N:Name, _, C) => b2i(rreg(Name2String(N), C) ==Int 0) requires isRegName(N, C)
  rule rd(N:Name, T, C)   => trunc(symAddr(lookupSym(Name2String(N), C)), wid(T))  // &var
    requires Name2String(N) =/=String "WARP_SZ" andBool notBool isRegName(N, C)
  rule rd(N:Name, T, _)   => trunc(32, wid(T)) requires Name2String(N) ==String "WARP_SZ"

  // width of a destination register (declared with .reg), default 32
  syntax Int ::= regW(Map, Operand) [function, total]
  rule regW(RW, D:Reg)  => {RW[Reg2String(D)] orDefault 32}:>Int
  rule regW(RW, D:Name) => {RW[Name2String(D)] orDefault 32}:>Int
  rule regW(_, _) => 32 [owise]
  // a narrow signed value is sign-extended to the register that holds it
  syntax Int ::= extReg(Int, Ty, Map, Operand) [function, total]
  rule extReg(V, T, RW, D) => trunc(sext(V, wid(T)), regW(RW, D))
    requires isS(T) andBool wid(T) <Int regW(RW, D)
  rule extReg(V, _, _, _) => V [owise]

  // packed 2x16-bit types are handled lane-wise by dedicated rules
  syntax Bool ::= isPacked(Ty) [function, total]
  rule isPacked(ty(K, _)) => K in (SetItem("s16x2") SetItem("u16x2") SetItem("f16x2") SetItem("bf16x2"))

  syntax Bool ::= isRegName(Name, Ctx) [function, total]
  rule isRegName(N, ctx(_, _, _, _, RW)) => Name2String(N) in_keys(RW)

  syntax Map ::= wr(Map, Operand, Int) [function, total]
  rule wr(R, D:Reg, V) => R[Reg2String(D) <- V]
  rule wr(R, D:Name, V) => R[Name2String(D) <- V] requires Name2String(D) =/=String "_"
  rule wr(R, _, _) => R [owise]                                   // sink `_`

  syntax Bool ::= isVec(Operand) [function, total]
  rule isVec({ _ }) => true
  rule isVec(_) => false [owise]

  // ------------------------------------------------------------ fetch
  syntax K ::= fetch(KItem, Ctx) [function]
  rule fetch(ins(noG, O, Os), C) => drainPre(O) ~> exec(O, ops(Os), C)
  rule fetch(ins(g(P, N), O, Os), C) =>
       #if (rreg(P, C) =/=Int 0) =/=Bool N #then drainPre(O) ~> exec(O, ops(Os), C) #else .K #fi
  rule fetch(X, _) => X [owise]

  syntax Int ::= uid(Int, Int) [function, total]
  rule uid(C, T) => C *Int 1048576 +Int T

  // Instructions whose effect is visible to other threads.  Everything else
  // (ALU, control flow, calls, .param/.local accesses) is thread-local.
  // In the relaxed memory model (ptx-exec-weak.k) fences, barriers, exit and asynchronous operations first wait
  // for the thread's store buffer to drain; the buffer is always empty in the default sequentially consistent mode.
  syntax KItem ::= "#sbWait"
  syntax K ::= drainPre(Opc) [function, total]
  rule drainPre(opc(OP, M, _)) => #sbWait
    requires OP ==String "fence" orBool OP ==String "membar" orBool OP ==String "exit"
      orBool isBar(OP, M) orBool isBarRed(OP, M) orBool isBarWarp(OP, M) orBool isBarCluster(OP, M) orBool isCollective(OP)
  rule drainPre(_) => .K [owise]
  rule <tk> #sbWait => .K ... </tk> <sb> .List </sb>

  syntax Bool ::= isVisible(KItem) [function, total]
  rule isVisible(ins(_, opc(OP, M, _), _)) =>
       ((OP in (SetItem("ld") SetItem("st") SetItem("atom") SetItem("red") SetItem("multimem") SetItem("fabric")))
         andBool notBool spaceOf(M) in (SetItem("param") SetItem("local")))
       orBool isBar(OP, M) orBool isBarRed(OP, M) orBool isBarWarp(OP, M) orBool isBarCluster(OP, M) orBool OP ==String "exit" orBool isCollective(OP)
  rule isVisible(_) => false [owise]

  // Warp-collective instructions rendezvous through the CTA (ptx-exec-coll.k);
  // later modules extend this set with shfl, vote, match, ...
  // Each such module adds `rule isCollective("name") => true`.
  syntax Bool ::= isCollective(String) [function, total]
  rule isCollective(_) => false [owise]

  // fetch: allowed when holding the turn (local instruction) or when the turn
  // is free (any instruction; the thread then takes the turn)
  rule <tk> .K => fetch({CODE[PC]}:>KItem, ctx(R, E, S, L, RW)) </tk>
       <turn> U => uid(C, T) </turn>
       <ctaid> C </ctaid> <tid> T </tid>
       <status> running </status>
       <fn> F </fn> <pc> PC => PC +Int 1 </pc>
       <regs> R </regs> <env> E </env> <syms> S </syms> <rw> RW </rw>
       <funcs> ... F |-> fdef(_, _, _, CODE, L, _) ... </funcs>
    requires PC in_keys(CODE)
     andBool (U ==Int -1 orBool (U ==Int uid(C, T) andBool notBool isVisible({CODE[PC]}:>KItem)))
  // yield before a visible instruction
  rule <tk> .K </tk>
       <turn> U => -1 </turn>
       <ctaid> C </ctaid> <tid> T </tid>
       <status> running </status>
       <fn> F </fn> <pc> PC </pc>
       <funcs> ... F |-> fdef(_, _, _, CODE, _, _) ... </funcs>
    requires U ==Int uid(C, T) andBool PC in_keys(CODE) andBool isVisible({CODE[PC]}:>KItem)
  // falling off the end of a function body is an implicit `ret`
  rule <tk> .K => #ret </tk>
       <turn> U => uid(C, T) </turn>
       <ctaid> C </ctaid> <tid> T </tid>
       <status> running </status>
       <fn> F </fn> <pc> PC </pc>
       <funcs> ... F |-> fdef(_, _, _, CODE, _, _) ... </funcs>
    requires notBool PC in_keys(CODE) andBool (U ==Int -1 orBool U ==Int uid(C, T))

  syntax KItem ::= "#yield" | "#ret" | "#exit" | "#die"
  rule <tk> #yield => .K ... </tk> <turn> _ => -1 </turn>

  syntax Bool ::= isBar(String, Set) [function, total]
  rule isBar(OP, M) => (OP ==String "bar" orBool OP ==String "barrier")
                       andBool notBool "warp" in M andBool notBool "red" in M andBool notBool "cluster" in M
  syntax Bool ::= isBarCluster(String, Set) [function, total]
  rule isBarCluster(OP, M) => OP ==String "barrier" andBool "cluster" in M
  syntax Bool ::= isBarRed(String, Set) [function, total]
  rule isBarRed(OP, M) => (OP ==String "bar" orBool OP ==String "barrier") andBool "red" in M andBool notBool "warp" in M
  syntax Bool ::= isBarWarp(String, Set) [function, total]
  rule isBarWarp(OP, M) => OP ==String "bar" andBool "warp" in M
endmodule

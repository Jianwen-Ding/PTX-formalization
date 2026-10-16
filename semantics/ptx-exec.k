// =============================================================================
// PTX-EXEC: thread-level execution.
//
// A thread whose <tk> is empty and whose status is `running` fetches the
// instruction at its pc.  Guards are evaluated at fetch time.  The fetched
// instruction becomes `exec(opcode, ops(operands), ctx)` where the context
// carries the (thread-local, hence stable) register file and symbol tables.
//
// Every access to shared state (global/shared memory, barriers) happens in a
// single rewrite step, so each memory instruction -- and in particular each
// atom/red -- is atomic, and different threads interleave arbitrarily.
// =============================================================================
requires "ptx-load.k"
requires "ptx-alu.k"

module PTX-EXEC
  imports PTX-LOAD
  imports PTX-ALU

  syntax Ctx ::= ctx(Map, Map, Map, Map)        // regs, env, syms, labels
  syntax Ops ::= ops(Operands)
  syntax KItem ::= exec(Opc, Ops, Ctx)
                 | #fault(String, Int)            // stuck: run-time error
                 | #unsupported(Opc)              // stuck: not modelled

  // ------------------------------------------------------------ registers
  syntax Int ::= rreg(String, Ctx) [function, total]
  rule rreg(R, ctx(RS, _, _, _)) => {RS[R] orDefault 0}:>Int

  syntax Sym ::= lookupSym(String, Ctx) [function, total]
  rule lookupSym(N, ctx(_, E, _, _)) => {E[N]}:>Sym requires N in_keys(E)
  rule lookupSym(N, ctx(_, E, S, _)) => {S[N]}:>Sym requires notBool N in_keys(E) andBool N in_keys(S)
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
  rule rd(N:Name, T, C)   => trunc(symAddr(lookupSym(Name2String(N), C)), wid(T))  // &var

  syntax Map ::= wr(Map, Operand, Int) [function, total]
  rule wr(R, D:Reg, V) => R[Reg2String(D) <- V]
  rule wr(R, _, _) => R [owise]                                   // sink `_`

  syntax Bool ::= isVec(Operand) [function, total]
  rule isVec({ _ }) => true
  rule isVec(_) => false [owise]

  // ------------------------------------------------------------ fetch
  syntax K ::= fetch(KItem, Ctx) [function]
  rule fetch(ins(noG, O, Os), C) => exec(O, ops(Os), C)
  rule fetch(ins(g(P, N), O, Os), C) =>
       #if (rreg(P, C) =/=Int 0) =/=Bool N #then exec(O, ops(Os), C) #else .K #fi
  rule fetch(X, _) => X [owise]

  syntax Int ::= uid(Int, Int) [function, total]
  rule uid(C, T) => C *Int 1048576 +Int T

  // Instructions whose effect is visible to other threads.  Everything else
  // (ALU, control flow, calls, .param/.local accesses) is thread-local.
  syntax Bool ::= isVisible(KItem) [function, total]
  rule isVisible(ins(_, opc(OP, M, _), _)) =>
       ((OP in (SetItem("ld") SetItem("st") SetItem("atom") SetItem("red")))
         andBool notBool spaceOf(M) in (SetItem("param") SetItem("local")))
       orBool isBar(OP, M) orBool OP ==String "exit"
  rule isVisible(_) => false [owise]

  // fetch: allowed when holding the turn (local instruction) or when the turn
  // is free (any instruction; the thread then takes the turn)
  rule <tk> .K => fetch({CODE[PC]}:>KItem, ctx(R, E, S, L)) </tk>
       <turn> U => uid(C, T) </turn>
       <ctaid> C </ctaid> <tid> T </tid>
       <status> running </status>
       <fn> F </fn> <pc> PC => PC +Int 1 </pc>
       <regs> R </regs> <env> E </env> <syms> S </syms>
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

  syntax KItem ::= "#yield"
  rule <tk> #yield => .K ... </tk> <turn> _ => -1 </turn>

  // ------------------------------------------------------------ ALU
  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, bin(OP, M, T, rd(A, T, C), rd(B, src2Ty(OP, T), C))) </regs>
    requires OP in BINOPS

  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, un(OP, M, T, rd(A, T, C))) </regs>
    requires OP in UNOPS

  rule <tk> exec(opc(OP, M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, tern(OP, M, T, rd(A, T, C), rd(B, T, C), rd(X, src3Ty(OP, M, T), C))) </regs>
    requires OP in TERNOPS

  // setp.cmp[.bop].type  p[|q], a, b[, c]
  syntax Map ::= wrP(Map, Operand, Int, String, Int) [function, total]
  rule wrP(R, P:Reg | Q:Reg, V, BO, X) =>
       R[Reg2String(P) <- bop(V, BO, X)][Reg2String(Q) <- bop(1 -Int V, BO, X)]
  rule wrP(R, D, V, BO, X) => wr(R, D, bop(V, BO, X)) [owise]
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(D, A, B, .Operands), C) => .K ... </tk>
       <regs> R => wrP(R, D, cmp(cmpOp(M), T, rd(A, T, C), rd(B, T, C)), "none", 0) </regs>
  rule <tk> exec(opc("setp", M, ListItem(T)), ops(D, A, B, X, .Operands), C) => .K ... </tk>
       <regs> R => wrP(R, D, cmp(cmpOp(M), T, rd(A, T, C), rd(B, T, C)), boolOp(M), rd(X, ty("p", 1), C)) </regs>

  // mov, including vector pack/unpack: mov.b64 %rd, {%r0, %r1}
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, rd(A, T, C)) </regs>
    requires notBool isVec(D) andBool notBool isVec(A)
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, { Vs }, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, pack(opsList(Vs), wid(T) /Int size(opsList(Vs)), 0, C)) </regs>
  rule <tk> exec(opc("mov", _, ListItem(T)), ops({ Ds }, A, .Operands), C) => .K ... </tk>
       <regs> R => unpack(R, opsList(Ds), wid(T) /Int size(opsList(Ds)), rd(A, T, C)) </regs>

  syntax Int ::= pack(List, Int, Int, Ctx) [function]
  rule pack(ListItem(V:Operand) L, W, Sh, C) => (rd(V, ty("b", W), C) <<Int Sh) |Int pack(L, W, Sh +Int W, C)
  rule pack(.List, _, _, _) => 0
  syntax Map ::= unpack(Map, List, Int, Int) [function]
  rule unpack(R, ListItem(D:Operand) L, W, V) => unpack(wr(R, D, trunc(V, W)), L, W, V >>Int W)
  rule unpack(R, .List, _, _) => R

  // cvt.[mods].dtype.stype
  rule <tk> exec(opc("cvt", M, ListItem(DT) ListItem(ST)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, cvt(M, DT, ST, rd(A, ST, C))) </regs>

  // cvta[.to].space.size: move between a state space and the generic space
  rule <tk> exec(opc("cvta", M, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(#if "to" in M #then rd(A, T, C) -Int winBase(spaceOf(M))
                                  #else rd(A, T, C) +Int winBase(spaceOf(M)) #fi, wid(T))) </regs>

  // ------------------------------------------------------------ memory
  syntax Loc ::= loc(String, Int) | badLoc(String)
  syntax Loc ::= ea(Addr, String, Ctx)  [function]
               | mkLoc(String, Int)     [function, total]
               | symLoc(Sym, Int)       [function, total]
  rule ea(R:Reg, S, C)         => mkLoc(S, rreg(Reg2String(R), C))
  rule ea(R:Reg I:Int, S, C)   => mkLoc(S, rreg(Reg2String(R), C) +Int I)
  rule ea(R:Reg + I:Int, S, C) => mkLoc(S, rreg(Reg2String(R), C) +Int I)
  rule ea(N:Name, _, C)         => symLoc(lookupSym(Name2String(N), C), 0)
  rule ea(N:Name I:Int, _, C)   => symLoc(lookupSym(Name2String(N), C), I)
  rule ea(N:Name + I:Int, _, C) => symLoc(lookupSym(Name2String(N), C), I)
  rule ea(I:Int, S, _)          => mkLoc(S, I)

  rule mkLoc("generic", A) => loc("local", A -Int LOCBASE) requires A >=Int LOCBASE
  rule mkLoc("generic", A) => loc("shared", A -Int SHBASE) requires A >=Int SHBASE andBool A <Int LOCBASE
  rule mkLoc("generic", A) => loc("global", A) requires A <Int SHBASE
  rule mkLoc("const", A) => loc("global", A)
  rule mkLoc(S, A) => loc(S, A) [owise]
  rule symLoc(sym("const", A), I) => loc("global", A +Int I)
  rule symLoc(sym(S, A), I) => loc(S, A +Int I) [owise]
  rule symLoc(noSym(N), _) => badLoc("undefined symbol " +String N)

  syntax MemOp ::= rdOp(Int)                       // bytes
                 | wrOp(Int, Int)                  // bytes, value
                 | atOp(String, Ty, Int, Int)      // op, type, b, c
  syntax KItem ::= #mem(Loc, MemOp) | #res(Int)
                 | #ldwb(Operand, Ty) | "#drop"
  syntax Bytes ::= memNew(MemOp, Bytes, Int) [function]
  syntax Int   ::= memVal(MemOp, Bytes, Int) [function]
  rule memNew(rdOp(_), M, _) => M
  rule memNew(wrOp(N, V), M, A) => writeBytes(M, A, N, V)
  rule memNew(atOp(OP, T, B, X), M, A) => writeBytes(M, A, bytes(T), atomNew(OP, T, readBytes(M, A, bytes(T)), B, X))
  rule memVal(rdOp(N), M, A) => readBytes(M, A, N)
  rule memVal(wrOp(_, _), _, _) => 0
  rule memVal(atOp(_, T, _, _), M, A) => readBytes(M, A, bytes(T))

  // one step per access: the memory operation is atomic
  rule <tk> #mem(loc("global", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <gmem> M => memNew(OP, M, A) </gmem>
    requires A >=Int 4096
  rule <tk> #mem(loc("shared", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <smem> M => memNew(OP, M, A) </smem>
    requires A >=Int 0
  rule <tk> #mem(loc("local", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <lmem> M => memNew(OP, M, A) </lmem>
    requires A >=Int 0
  rule <tk> #mem(loc("param", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <pmem> M => memNew(OP, M, A) </pmem>
    requires A >=Int 0
  rule <tk> #mem(badLoc(S), _) => #fault(S, 0) ... </tk>
  rule <tk> #mem(loc(S, A), _) => #fault("invalid " +String S +String " address", A) ... </tk> [owise]

  // write-back of a loaded value; sub-word signed loads sign-extend to the
  // declared register width
  rule <tk> #res(V) ~> #ldwb(D, T) => .K ... </tk>
       <regs> R => wr(R, D, ldExt(V, T, RW, D)) </regs> <rw> RW </rw>
  rule <tk> #res(_) ~> #drop => .K ... </tk>
  syntax Int ::= ldExt(Int, Ty, Map, Operand) [function, total]
  rule ldExt(V, T, RW, D:Reg) => trunc(sext(V, wid(T)), {RW[Reg2String(D)] orDefault 32}:>Int)
    requires isS(T) andBool wid(T) <Int 32
  rule ldExt(V, _, _, _) => V [owise]

  syntax KItem ::= #ldv(List, Loc, Ty) | #stv(List, Loc, Ty, Ctx)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops(D, [ A ], .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), rdOp(bytes(T))) ~> #ldwb(D, T) ... </tk>
    requires notBool isVec(D)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops({ Ds }, [ A ], .Operands), C) =>
            #ldv(opsList(Ds), ea(A, spaceOf(M), C), T) ... </tk>
  rule <tk> exec(opc("st", M, ListItem(T)), ops([ A ], V, .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), wrOp(bytes(T), rd(V, T, C))) ~> #drop ... </tk>
    requires notBool isVec(V)
  rule <tk> exec(opc("st", M, ListItem(T)), ops([ A ], { Vs }, .Operands), C) =>
            #stv(opsList(Vs), ea(A, spaceOf(M), C), T, C) ... </tk>
  rule <tk> #ldv(ListItem(D:Operand) Ds, loc(S, A), T) =>
            #mem(loc(S, A), rdOp(bytes(T))) ~> #ldwb(D, T) ~> #ldv(Ds, loc(S, A +Int bytes(T)), T) ... </tk>
  rule <tk> #stv(ListItem(V:Operand) Vs, loc(S, A), T, C) =>
            #mem(loc(S, A), wrOp(bytes(T), rd(V, T, C))) ~> #drop ~> #stv(Vs, loc(S, A +Int bytes(T)), T, C) ... </tk>
  rule <tk> #ldv(.List, _, _) => .K ... </tk>
  rule <tk> #stv(.List, _, _, _) => .K ... </tk>
  rule <tk> #ldv(_, badLoc(S), _) => #fault(S, 0) ... </tk>
  rule <tk> #stv(_, badLoc(S), _, _) => #fault(S, 0) ... </tk>

  // atom.space.op.type d, [a], b[, c]     red.space.op.type [a], b
  rule <tk> exec(opc("atom", M, ListItem(T)), ops(D, [ A ], B, .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), atOp(atomOp(M), T, rd(B, T, C), 0)) ~> #ldwb(D, T) ... </tk>
  rule <tk> exec(opc("atom", M, ListItem(T)), ops(D, [ A ], B, X, .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), atOp(atomOp(M), T, rd(B, T, C), rd(X, T, C))) ~> #ldwb(D, T) ... </tk>
  rule <tk> exec(opc("red", M, ListItem(T)), ops([ A ], B, .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), atOp(atomOp(M), T, rd(B, T, C), 0)) ~> #drop ... </tk>

  // frame-allocated .local / .param variables
  rule <tk> declVar("local", N, SZ, AL) => .K ... </tk>
       <ltop> LT => alignUp(LT, AL) +Int SZ </ltop>
       <env> E => E[N <- sym("local", alignUp(LT, AL))] </env>
  rule <tk> declVar("param", N, SZ, AL) => .K ... </tk>
       <ptop> PT => alignUp(PT, AL) +Int SZ </ptop>
       <env> E => E[N <- sym("param", alignUp(PT, AL))] </env>

  // ------------------------------------------------------------ control flow
  syntax KItem ::= "#ret" | "#exit" | "#die"
  rule <tk> exec(opc("bra", _, _), ops(N:Name, .Operands), ctx(_, _, _, L)) => .K ... </tk>
       <pc> _ => {L[Name2String(N)]}:>Int </pc>
    requires Name2String(N) in_keys(L)
  rule <tk> exec(opc("ret", _, _), ops(.Operands), _) => #ret ... </tk>
  rule <tk> exec(opc("exit", _, _), ops(.Operands), _) => #exit ... </tk>
  rule <tk> exec(opc("trap", _, _), _, _) => #fault("trap", 0) ... </tk>
  // memory ordering is sequentially consistent here, so fences are no-ops
  rule <tk> exec(opc(OP, _, _), _, _) => .K ... </tk>
    requires OP in (SetItem("membar") SetItem("fence") SetItem("nanosleep") SetItem("brkpt")
                    SetItem("pmevent") SetItem("prefetch") SetItem("prefetchu"))

  // ------------------------------------------------------------ calls
  syntax KItem ::= #call(String, List, List, Ctx)
                 | #bindArgs(List, List, Ctx, Map, String, List)
                 | #bindRets(List, List, Ctx, Map, String)
                 | #enter(String, Map)
  rule <tk> exec(opc("call", _, _), ops(( RS ), F:Name, ( AS ), .Operands), C) =>
            #call(Name2String(F), opsList(RS), opsList(AS), C) ... </tk>
  rule <tk> exec(opc("call", _, _), ops(( RS ), F:Name, .Operands), C) =>
            #call(Name2String(F), opsList(RS), .List, C) ... </tk>
  rule <tk> exec(opc("call", _, _), ops(F:Name, ( AS ), .Operands), C) =>
            #call(Name2String(F), .List, opsList(AS), C) ... </tk>
  rule <tk> exec(opc("call", _, _), ops(F:Name, .Operands), C) =>
            #call(Name2String(F), .List, .List, C) ... </tk>

  rule <tk> #call(F, RS, AS, C) => #bindArgs(PS, AS, C, .Map, F, RS) ... </tk>
       <funcs> ... F |-> fdef(_, _, PS, _, _, _) ... </funcs>
  // a .param variable argument is passed by binding the callee's formal
  // parameter to the caller's variable; other arguments get a fresh slot
  rule <tk> #bindArgs(ListItem(pdecl(P, _, _, _)) PS, ListItem(N:Name) AS, C, E, F, RS) =>
            #bindArgs(PS, AS, C, E[P <- lookupSym(Name2String(N), C)], F, RS) ... </tk>
  rule <tk> #bindArgs(ListItem(pdecl(P, T, SZ, AL)) PS, ListItem(V:Operand) AS, C, E, F, RS) =>
            #bindArgs(PS, AS, C, E[P <- sym("param", alignUp(PT, AL))], F, RS) ... </tk>
       <pmem> PM => writeBytes(PM, alignUp(PT, AL), SZ, rd(V, T, C)) </pmem>
       <ptop> PT => alignUp(PT, AL) +Int SZ </ptop>
    requires notBool isName(V)
  rule <tk> #bindArgs(.List, .List, C, E, F, RS) => #bindRets(RPS, RS, C, E, F) ... </tk>
       <funcs> ... F |-> fdef(_, RPS, _, _, _, _) ... </funcs>
  rule <tk> #bindRets(ListItem(pdecl(P, _, _, _)) RPS, ListItem(N:Name) RS, C, E, F) =>
            #bindRets(RPS, RS, C, E[P <- lookupSym(Name2String(N), C)], F) ... </tk>
  rule <tk> #bindRets(.List, .List, _, E, F) => #enter(F, E) ... </tk>
  rule <tk> #enter(F, E) => .K ... </tk>
       <fn> CF => F </fn> <pc> PC => 0 </pc>
       <regs> R => SR </regs> <rw> RW => FRW </rw> <env> CE => E </env>
       <ltop> LT </ltop> <sregs> SR </sregs>
       <stack> .List => ListItem(frame(CF, PC, R, RW, CE, LT)) ... </stack>
       <funcs> ... F |-> fdef(_, _, _, _, _, FRW) ... </funcs>

  rule <tk> #ret => .K ... </tk>
       <stack> ListItem(frame(F, PC, R, RW, E, LT)) => .List ... </stack>
       <fn> _ => F </fn> <pc> _ => PC </pc> <regs> _ => R </regs> <rw> _ => RW </rw>
       <env> _ => E </env> <ltop> _ => LT </ltop>
  rule <tk> #ret => #exit ... </tk> <stack> .List </stack>

  // ------------------------------------------------------------ barriers
  // bar{.cta}.sync / barrier{.cta}.sync{.aligned}  a{, b}
  // A thread arriving at barrier `a` records the barrier's current generation;
  // once the expected number of threads (b, or all live threads of the CTA)
  // has arrived the generation is bumped, releasing every waiter.
  syntax KItem ::= #bar(Int, Int, Bool)            // id, expected (-1: all), wait?
                 | #barCheck(Int) | #barCheckAll(List)
  syntax Bool ::= isBar(String, Set) [function, total]
  rule isBar(OP, M) => (OP ==String "bar" orBool OP ==String "barrier")
                       andBool notBool "warp" in M andBool notBool "red" in M

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
  rule <tk> exec(O, _, _) => #unsupported(O) ... </tk> [owise]
endmodule

// =============================================================================
// PTX-EXEC-CTRL: branches, calls and returns, thread exit
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-CTRL
  imports PTX-EXEC-CORE

  // ------------------------------------------------------------ control flow
  rule <tk> exec(opc("bra", _, _), ops(N:Name, .Operands), ctx(_, _, _, L, _)) => .K ... </tk>
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

  // indirect calls: call (ret), %fptr, (args), proto_or_targets ;   the target is checked against the address table
  syntax KItem ::= #icall(Int, List, List, Ctx)
  rule <tk> exec(opc("call", _, _), ops(( RS ), F:Reg, ( AS ), T:Name, .Operands), C) =>
            #icall(rd(F, ty("u", 64), C), opsList(RS), opsList(AS), C) ... </tk>
  rule <tk> exec(opc("call", _, _), ops(( RS ), F:Reg, T:Name, .Operands), C) =>
            #icall(rd(F, ty("u", 64), C), opsList(RS), .List, C) ... </tk>
  rule <tk> exec(opc("call", _, _), ops(F:Reg, ( AS ), T:Name, .Operands), C) =>
            #icall(rd(F, ty("u", 64), C), .List, opsList(AS), C) ... </tk>
  rule <tk> exec(opc("call", _, _), ops(F:Reg, T:Name, .Operands), C) =>
            #icall(rd(F, ty("u", 64), C), .List, .List, C) ... </tk>
  rule <tk> #icall(A, RS, AS, C) => #call(F, RS, AS, C) ... </tk>
       <faddrs> ... A |-> F ... </faddrs>
  rule <tk> #icall(A, _, _, _) => #fault("indirect call through an invalid function pointer", A) ... </tk> [owise]

  // brx.idx i, table ;   (table: label : .branchtargets L0, L1, ... ;)
  rule <tk> exec(opc("brx", _, _), ops(I, T:Name, .Operands), ctx(R, E, S, L, RW)) => #brx(rd(I, ty("u", 32), ctx(R, E, S, L, RW)), {L["@branchtargets:" +String Name2String(T)] orDefault .List}:>List, L) ... </tk>
  syntax KItem ::= #brx(Int, List, Map)
  rule <tk> #brx(I, Ts, L) => .K ... </tk>
       <pc> _ => {L[{Ts[I]}:>String]}:>Int </pc>
    requires I >=Int 0 andBool I <Int size(Ts) andBool {Ts[I]}:>String in_keys(L)
  rule <tk> #brx(I, _, _) => #fault("brx.idx: index outside the target list", I) ... </tk> [owise]

  rule <tk> #call(F, RS, AS, C) => #bindArgs(PS, AS, C, .Map, F, RS) ... </tk>
       <funcs> ... F |-> fdef(_, _, PS, _, _, _) ... </funcs>
  // a .param variable argument is passed by binding the callee's formal
  // parameter to the caller's variable; other arguments get a fresh slot
  // `.reg` formals (alignment < 0) are passed by value: the value is parked in the environment as "@in:P" and
  // written into the callee's register P on entry; "@out:P" names the caller register receiving P on return
  rule <tk> #bindArgs(ListItem(pdecl(P, T, _, AL)) PS, ListItem(V:Operand) AS, C, E, F, RS) =>
            #bindArgs(PS, AS, C, E[("@in:" +String P) <- sym("regin", rd(V, T, C))], F, RS) ... </tk>
    requires AL <Int 0
  rule <tk> #bindArgs(ListItem(pdecl(P, _, _, AL)) PS, ListItem(N:Name) AS, C, E, F, RS) =>
            #bindArgs(PS, AS, C, E[P <- lookupSym(Name2String(N), C)], F, RS) ... </tk>
    requires AL >=Int 0
  rule <tk> #bindArgs(ListItem(pdecl(P, T, SZ, AL)) PS, ListItem(V:Operand) AS, C, E, F, RS) =>
            #bindArgs(PS, AS, C, E[P <- sym("param", alignUp(PT, AL))], F, RS) ... </tk>
       <pmem> PM => writeBytes(PM, alignUp(PT, AL), SZ, rd(V, T, C)) </pmem>
       <ptop> PT => alignUp(PT, AL) +Int SZ </ptop>
    requires notBool isName(V) andBool AL >=Int 0
  rule <tk> #bindArgs(.List, .List, C, E, F, RS) => #bindRets(RPS, RS, C, E, F) ... </tk>
       <funcs> ... F |-> fdef(_, RPS, _, _, _, _) ... </funcs>
  rule <tk> #bindRets(ListItem(pdecl(P, _, _, AL)) RPS, ListItem(D:Operand) RS, C, E, F) =>
            #bindRets(RPS, RS, C, E[("@out:" +String P) <- noSym(opStr(D))], F) ... </tk>
    requires AL <Int 0
  rule <tk> #bindRets(ListItem(pdecl(P, _, _, AL)) RPS, ListItem(N:Name) RS, C, E, F) =>
            #bindRets(RPS, RS, C, E[P <- lookupSym(Name2String(N), C)], F) ... </tk>
    requires AL >=Int 0
  rule <tk> #bindRets(.List, .List, _, E, F) => #enter(F, E) ... </tk>
  rule <tk> #enter(F, E) => .K ... </tk>
       <fn> CF => F </fn> <pc> PC => 0 </pc>
       <regs> R => regIns(SR, keys_list(E), E) </regs> <rw> RW => FRW </rw> <env> CE => E </env>
       <ltop> LT </ltop> <sregs> SR </sregs>
       <stack> .List => ListItem(frame(CF, PC, R, RW, CE, LT)) ... </stack>
       <funcs> ... F |-> fdef(_, _, _, _, _, FRW) ... </funcs>

  rule <tk> #ret => .K ... </tk>
       <stack> ListItem(frame(F, PC, R, RW, E, LT)) => .List ... </stack>
       <fn> _ => F </fn> <pc> _ => PC </pc> <regs> CR => regOuts(R, keys_list(CE), CE, CR) </regs> <rw> _ => RW </rw>
       <env> CE => E </env> <ltop> _ => LT </ltop>
  syntax Map ::= regIns(Map, List, Map) [function, total] | regOuts(Map, List, Map, Map) [function, total]
  rule regIns(R, .List, _) => R
  rule regIns(R, ListItem(K:String) Ks, E) => regIns(#if pfx(K, "@in:")
                                                       #then R[substrString(K, 4, lengthString(K)) <- symAddr({E[K]}:>Sym)] #else R #fi, Ks, E)
  rule regIns(R, _, _) => R [owise]
  rule regOuts(R, .List, _, _) => R
  rule regOuts(R, ListItem(K:String) Ks, E, CR) => regOuts(#if pfx(K, "@out:")
                                                       #then R[symName({E[K]}:>Sym) <- {CR[substrString(K, 5, lengthString(K))] orDefault 0}:>Int] #else R #fi, Ks, E, CR)
  rule regOuts(R, _, _, _) => R [owise]
  syntax String ::= symName(Sym) [function, total]
  rule symName(noSym(S)) => S
  rule symName(sym(S, _)) => S
  rule <tk> #ret => #exit ... </tk> <stack> .List </stack>
endmodule

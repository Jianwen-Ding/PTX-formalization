// =============================================================================
// PTX-EXEC-CTRL: branches, calls and returns, thread exit
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-CTRL
  imports PTX-EXEC-CORE

  // ------------------------------------------------------------ control flow
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
endmodule

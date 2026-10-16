// =============================================================================
// PTX-EXEC-MEM: ld/st/atom/red, effective addresses, memory accesses, frame variables
// (part of the thread-level execution semantics; see ptx-exec.k)
// =============================================================================
requires "ptx-exec-core.k"

module PTX-EXEC-MEM
  imports PTX-EXEC-CORE

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
endmodule

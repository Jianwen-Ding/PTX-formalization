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
  rule ea(R:Reg, S, C)         => mkLocA(S, rreg(Reg2String(R), C), Reg2String(R), C)
  rule ea(R:Reg I:Int, S, C)   => mkLocA(S, rreg(Reg2String(R), C) +Int I, Reg2String(R), C)
  rule ea(R:Reg + I:Int, S, C) => mkLocA(S, rreg(Reg2String(R), C) +Int I, Reg2String(R), C)
  rule ea(N:Name, S, C)         => mkLocA(S, rreg(Name2String(N), C), Name2String(N), C) requires isRegName(N, C)
  rule ea(N:Name I:Int, S, C)   => mkLocA(S, rreg(Name2String(N), C) +Int I, Name2String(N), C) requires isRegName(N, C)
  rule ea(N:Name + I:Int, S, C) => mkLocA(S, rreg(Name2String(N), C) +Int I, Name2String(N), C) requires isRegName(N, C)
  rule ea(N:Name, _, C)         => symLoc(lookupSym(Name2String(N), C), 0) requires notBool isRegName(N, C)
  rule ea(N:Name I:Int, _, C)   => symLoc(lookupSym(Name2String(N), C), I) requires notBool isRegName(N, C)
  rule ea(N:Name + I:Int, _, C) => symLoc(lookupSym(Name2String(N), C), I) requires notBool isRegName(N, C)
  rule ea(I:Int, S, _)          => mkLoc(S, I)

  // `.address_size 32`: address registers must be 32 bits wide and addresses wrap modulo 2^32
  syntax Loc ::= mkLocA(String, Int, String, Ctx) [function, total]
  syntax Bool ::= addr32(Ctx) [function, total]
  rule addr32(ctx(_, _, S, _, _)) => "%%asz" in_keys(S)
  rule mkLocA(S, V, _, C) => mkLoc(S, V) requires notBool addr32(C)
  rule mkLocA(_, _, R, ctx(_, _, S, _, RW)) => badLoc("address_size 32: 64-bit register " +String R +String " used as an address")
    requires "%%asz" in_keys(S) andBool {RW[R] orDefault 32}:>Int ==Int 64
  rule mkLocA(S, V, R, ctx(_, _, SY, _, RW)) => mkLoc(S, V modInt 4294967296)
    requires "%%asz" in_keys(SY) andBool {RW[R] orDefault 32}:>Int =/=Int 64

  rule mkLoc("generic", A) => loc("cluster", (((A -Int CLBASE) >>Int 24) +Int 1) <<Int 24 |Int ((A -Int CLBASE) &Int 16777215)) requires A >=Int CLBASE andBool A <Int PARBASE
  rule mkLoc("generic", A) => loc("param", A -Int PARBASE) requires A >=Int PARBASE
  rule mkLoc("generic", A) => loc("local", A -Int LOCBASE) requires A >=Int LOCBASE andBool A <Int CLBASE
  rule mkLoc("generic", A) => loc("shared", A -Int SHBASE) requires A >=Int SHBASE andBool A <Int LOCBASE
  rule mkLoc("generic", A) => loc("global", A) requires A <Int SHBASE
  rule mkLoc("const", A) => loc("global", A)
  rule mkLoc(S, A) => loc(S, A) [owise]
  rule symLoc(sym("const", A), I) => loc("global", A +Int I)
  rule symLoc(sym(S, A), I) => loc(S, A +Int I) [owise]
  rule symLoc(noSym(N), _) => badLoc("undefined symbol " +String N)

  syntax MemOp ::= rdOp(Int)                       // bytes
                 | insOp(Int, Int, Int)            // bit offset within the byte, width, value (sub-byte element store)
                 | wrOp(Int, Int)                  // bytes, value
                 | wrRelOp(Int, Int)               // st.release: like wrOp, but an ordering point in the relaxed memory model
                 | atOp(String, Ty, Int, Int)      // op, type, b, c
  syntax KItem ::= #mem(Loc, MemOp) | #res(Int)
                 | #ldwb(Operand, Ty) | "#drop"
  syntax Bytes ::= memNew(MemOp, Bytes, Int) [function]
  syntax Int   ::= memVal(MemOp, Bytes, Int) [function]
  rule memNew(rdOp(_), M, _) => M
  rule memNew(wrOp(N, V), M, A) => writeBytes(M, A, N, V)
  rule memNew(wrRelOp(N, V), M, A) => writeBytes(M, A, N, V)
  rule memNew(atOp(OP, T, B, X), M, A) => writeBytes(M, A, bytes(T), atomNew(OP, T, readBytes(M, A, bytes(T)), B, X))
  rule memNew(insOp(O, W, V), M, A) => writeBytes(M, A, 1, (readBytes(M, A, 1) &Int (~Int (mask(W) <<Int O))) |Int ((V &Int mask(W)) <<Int O))
  rule memVal(insOp(_, _, _), _, _) => 0
  rule memVal(rdOp(N), M, A) => readBytes(M, A, N)
  rule memVal(wrOp(_, _), _, _) => 0
  rule memVal(wrRelOp(_, _), _, _) => 0
  rule memVal(atOp(_, T, _, _), M, A) => readBytes(M, A, bytes(T))

  // one step per access: the memory operation is atomic
  rule <tk> #mem(loc("global", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <gmem> M => memNew(OP, M, A) </gmem> <mm> "sc" </mm>
    requires A >=Int 4096
  rule <tk> #mem(loc("shared", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <smem> M => memNew(OP, M, A) </smem> <mm> "sc" </mm>
    requires A >=Int 0
  // relaxed memory model (ptx-exec-weak.k): global and shared accesses go through the store buffer
  syntax KItem ::= #rm(Loc, MemOp) | #mem1(Loc, MemOp)
  // accesses of the asynchronous proxy (bulk copies, fabric operations): they do not see the issuing thread's store buffer, so a
  // generic-proxy store must be made visible with fence.proxy.async (or a fence) before it
  syntax KItem ::= #amem(Loc, MemOp)
  rule <tk> #amem(L, OP) => #mem(L, OP) ... </tk> <mm> M </mm> requires M =/=String "relaxed"
  rule <tk> #amem(L, OP) => #mem1(L, OP) ... </tk> <mm> "relaxed" </mm>
  rule <tk> #mem(loc(S, A), OP) => #rm(loc(S, A), OP) ... </tk> <mm> "relaxed" </mm>
    requires (S ==String "global" andBool A >=Int 4096) orBool (S ==String "shared" andBool A >=Int 0)
  rule <tk> #mem1(loc("global", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <gmem> M => memNew(OP, M, A) </gmem>
  rule <tk> #mem1(loc("shared", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <smem> M => memNew(OP, M, A) </smem>
  rule <tk> #mem(loc("local", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <lmem> M => memNew(OP, M, A) </lmem>
    requires A >=Int 0
  rule <tk> #mem(loc("param", A), OP) => #res(memVal(OP, M, A)) ... </tk>
       <pmem> M => memNew(OP, M, A) </pmem>
    requires A >=Int 0
  rule <tk> #mem(badLoc(S), _) => #fault(S, 0) ... </tk>
  // strict undefined-behaviour mode (`.ub strict;`): a naturally misaligned access, an access outside the allocated global
  // data (below the first symbol or beyond the allocation top) and an access above the thread's stack top are errors
  syntax Int ::= memBytes(MemOp) [function, total]
  rule memBytes(rdOp(N)) => N
  rule memBytes(wrOp(N, _)) => N
  rule memBytes(wrRelOp(N, _)) => N
  rule memBytes(atOp(_, T, _, _)) => bytes(T)
  rule memBytes(_) => 0 [owise]
  rule <tk> #mem(loc(S, A), OP) => #fault("misaligned " +String S +String " access of " +String Int2String(memBytes(OP)) +String " bytes", A) ... </tk>
       <ub> true </ub>
    requires (memBytes(OP) ==Int 2 orBool memBytes(OP) ==Int 4 orBool memBytes(OP) ==Int 8 orBool memBytes(OP) ==Int 16)
     andBool A %Int memBytes(OP) =/=Int 0 [priority(20)]
  rule <tk> #mem(loc("global", A), OP) => #fault("global access outside the allocated data", A) ... </tk>
       <ub> true </ub> <gtop> G </gtop>
    requires memBytes(OP) >Int 0 andBool (A <Int 4096 orBool A +Int memBytes(OP) >Int G) [priority(20)]
  rule <tk> #mem(loc("local", A), OP) => #fault("local access above the stack top", A) ... </tk>
       <ub> true </ub> <ltop> LT </ltop>
    requires memBytes(OP) >Int 0 andBool A +Int memBytes(OP) >Int LT [priority(20)]
  rule <tk> #mem(loc(S, A), _) => #fault("invalid " +String S +String " address", A) ... </tk> [owise]

  // write-back of a loaded value; sub-word signed loads sign-extend to the
  // declared register width
  rule <tk> #res(V) ~> #ldwb(D, T) => .K ... </tk>
       <regs> R => wr(R, D, ldExt(V, T, RW, D)) </regs> <rw> RW </rw>
  rule <tk> #res(_) ~> #drop => .K ... </tk>
  syntax Int ::= ldExt(Int, Ty, Map, Operand) [function, total]
  rule ldExt(V, T, RW, D) => extReg(V, T, RW, D)

  syntax MemOp ::= stOp(Set, Int, Int) [function, total]
  rule stOp(M, N, V) => wrRelOp(N, V) requires "release" in M
  rule stOp(M, N, V) => wrOp(N, V) requires notBool "release" in M
  syntax KItem ::= #ldv(List, Loc, Ty) | #stv(List, Loc, Ty, Ctx)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops(D, [ A ], .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), rdOp(bytes(T))) ~> #ldwb(D, T) ... </tk>
    requires notBool isVec(D)
  rule <tk> exec(opc("ld", M, ListItem(T)), ops({ Ds }, [ A ], .Operands), C) =>
            #ldv(opsList(Ds), ea(A, spaceOf(M), C), T) ... </tk>
  rule <tk> exec(opc("st", M, ListItem(T)), ops([ A ], V, .Operands), C) =>
            #mem(ea(A, spaceOf(M), C), stOp(M, bytes(T), rd(V, T, C))) ~> #drop ... </tk>
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

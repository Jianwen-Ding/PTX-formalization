// =============================================================================
// PTX-EXEC-MEMQ: memory-operation qualifiers and cache controls.
//
// The machine is sequentially consistent at instruction granularity, so cache operators
// (.ca .cg .cs .lu .cv .wb .wt), eviction priorities (L1::evict_*), prefetch sizes,
// .nc, .volatile, .relaxed/.acquire/.release/.mmio and scopes are accepted and have no effect
// beyond the access itself (the weak memory model of task 36 refines this).  The cache-policy
// operand of L2::cache_hint accesses is dropped; createpolicy produces an opaque 64-bit value.
// =============================================================================
requires "ptx-exec-core.k"
requires "ptx-exec-mem.k"

module PTX-EXEC-MEMQ
  imports PTX-EXEC-CORE
  imports PTX-EXEC-MEM

  syntax Operands ::= dropLast(Operands) [function, total]
  rule dropLast(_:Operand, .Operands) => .Operands
  rule dropLast(O:Operand, Os:Operands) => O, dropLast(Os) requires Os =/=K .Operands
  rule dropLast(.Operands) => .Operands

  // ld/st/atom/red with a trailing L2 cache-policy operand
  rule <tk> exec(opc(OP, M, T), ops(Os), C) => exec(opc(OP, M -Set SetItem("L2::cache_hint"), T), ops(dropLast(Os)), C) ... </tk>
    requires "L2::cache_hint" in M andBool OP in (SetItem("ld") SetItem("st") SetItem("atom") SetItem("red"))

  // hints without observable effect
  rule <tk> exec(opc(OP, _, _), ops(_), _) => .K ... </tk>
    requires OP in (SetItem("discard") SetItem("applypriority"))

  // createpolicy.*  d, ... : an opaque cache policy
  rule <tk> exec(opc("createpolicy", _, _), ops(D, _:Operands), _) => .K ... </tk>
       <regs> R => wr(R, D, 0) </regs>

  // ---------------------------------------------------------------- packed atomics
  // atom/red on .f16x2 / .bf16x2 (and 16-bit integer pairs) operate lane-wise
  syntax Ty ::= laneTy(Ty) [function, total]
  rule laneTy(ty("f16x2", _)) => ty("f", 16)
  rule laneTy(ty("bf16x2", _)) => ty("bf", 16)
  rule laneTy(ty("s16x2", _)) => ty("s", 16)
  rule laneTy(ty("u16x2", _)) => ty("u", 16)
  rule laneTy(T) => T [owise]
  rule atomNew(OP, T, O, B, C) =>
       atomNew(OP, laneTy(T), O &Int 65535, B &Int 65535, C &Int 65535)
       |Int (atomNew(OP, laneTy(T), (O >>Int 16) &Int 65535, (B >>Int 16) &Int 65535, (C >>Int 16) &Int 65535) <<Int 16)
    requires isPacked(T) [priority(40)]
  // ---------------------------------------------------------------- vector atomics (v2/v4/v8)
  // each element is updated atomically, in order of increasing address
  syntax KItem ::= #atv(List, Loc, String, Ty, List)
  syntax List ::= rdList(List, Ty, Ctx) [function, total]
  rule rdList(ListItem(O:Operand) Os, T, C) => ListItem(rd(O, T, C)) rdList(Os, T, C)
  rule rdList(_, _, _) => .List [owise]
  rule <tk> exec(opc("atom", M, ListItem(T)), ops({ Ds }, [ A ], { Bs }, .Operands), C) =>
            #atv(opsList(Ds), ea(A, spaceOf(M), C), atomOp(M), T, rdList(opsList(Bs), T, C)) ... </tk> [priority(40)]
  rule <tk> exec(opc("red", M, ListItem(T)), ops([ A ], { Bs }, .Operands), C) =>
            #atv(.List, ea(A, spaceOf(M), C), atomOp(M), T, rdList(opsList(Bs), T, C)) ... </tk> [priority(40)]
  rule <tk> #atv(ListItem(D:Operand) Ds, loc(S, Ad), OP, T, ListItem(V:Int) Vs) =>
            #mem(loc(S, Ad), atOp(OP, T, V, 0)) ~> #ldwb(D, T) ~> #atv(Ds, loc(S, Ad +Int bytes(T)), OP, T, Vs) ... </tk>
  rule <tk> #atv(.List, loc(S, Ad), OP, T, ListItem(V:Int) Vs) =>
            #mem(loc(S, Ad), atOp(OP, T, V, 0)) ~> #drop ~> #atv(.List, loc(S, Ad +Int bytes(T)), OP, T, Vs) ... </tk>
  rule <tk> #atv(_, _, _, _, .List) => .K ... </tk>
  rule <tk> #atv(_, badLoc(E), _, _, _) => #fault(E, 0) ... </tk> [priority(40)]
endmodule

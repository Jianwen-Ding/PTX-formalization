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
endmodule

// =============================================================================
// PTX-EXEC: thread-level execution, assembled from per-area modules.
//
// A thread whose <tk> is empty and whose status is `running` fetches the
// instruction at its pc.  Guards are evaluated at fetch time.  The fetched
// instruction becomes `exec(opcode, ops(operands), ctx)` where the context
// carries the (thread-local, hence stable) register file and symbol tables.
//
// Every access to shared state (global/shared memory, barriers) happens in a
// single rewrite step, so each memory instruction -- and in particular each
// atom/red -- is atomic, and different threads interleave arbitrarily.
//
// To add a feature, create `ptx-exec-<area>.k` with a module importing
// PTX-EXEC-CORE, add `rule <tk> exec(opc("name", M, Tys), ops(...), C) => ...`
// rules, and import the module below.  See README.md, "Extending".
// =============================================================================
requires "ptx-exec-core.k"
requires "ptx-exec-alu.k"
requires "ptx-exec-mem.k"
requires "ptx-exec-ctrl.k"
requires "ptx-exec-sync.k"
requires "ptx-exec-coll.k"
requires "ptx-exec-int.k"
requires "ptx-exec-bits.k"
requires "ptx-exec-intx.k"
requires "ptx-exec-pred.k"
requires "ptx-exec-video.k"
requires "ptx-exec-half.k"
requires "ptx-exec-mini.k"
requires "ptx-exec-warp.k"
requires "ptx-exec-mma.k"
requires "ptx-exec-ldm.k"
requires "ptx-exec-sp.k"
requires "ptx-exec-wg.k"
requires "ptx-exec-tc.k"
requires "ptx-exec-memq.k"
requires "ptx-exec-cluster.k"
requires "ptx-exec-mbar.k"
requires "ptx-exec-async.k"
requires "ptx-exec-bulk.k"
requires "ptx-exec-tma.k"
requires "ptx-exec-misc.k"
requires "ptx-exec-tex.k"
requires "ptx-exec-weak.k"
requires "ptx-exec-isa.k"

module PTX-EXEC
  imports PTX-EXEC-CORE
  imports PTX-EXEC-ALU
  imports PTX-EXEC-MEM
  imports PTX-EXEC-CTRL
  imports PTX-EXEC-SYNC
  imports PTX-EXEC-COLL
  imports PTX-EXEC-INT
  imports PTX-EXEC-BITS
  imports PTX-EXEC-INTX
  imports PTX-EXEC-PRED
  imports PTX-EXEC-VIDEO
  imports PTX-EXEC-HALF
  imports PTX-EXEC-MINI
  imports PTX-EXEC-WARP
  imports PTX-EXEC-MMA
  imports PTX-EXEC-LDM
  imports PTX-EXEC-SP
  imports PTX-EXEC-WG
  imports PTX-EXEC-TC
  imports PTX-EXEC-MEMQ
  imports PTX-EXEC-CLUSTER
  imports PTX-EXEC-MBAR
  imports PTX-EXEC-ASYNC
  imports PTX-EXEC-BULK
  imports PTX-EXEC-TMA
  imports PTX-EXEC-MISC
  imports PTX-EXEC-TEX
  imports PTX-EXEC-WEAK
  imports PTX-EXEC-ISA

  // anything else is outside the modelled subset
  rule <tk> exec(O, _, _) => #unsupported(O) ... </tk> [owise]
endmodule

// =============================================================================
// PTX-MM: a reduced definition for the concurrency / memory-model work: the core machine, ALU, memory, control flow,
// barriers, warp collectives, clusters, mbarrier and the relaxed memory mode, without the tensor-core, TMA, texture and
// bulk-copy subsystems.  It builds in a fraction of the time (including the exhaustive-search interpreter) and runs the
// litmus tests:   make build-mm
// =============================================================================
requires "ptx-syntax.k"
requires "ptx-harness.k"

module PTX-MM
  imports PTX-HARNESS-BASE
endmodule

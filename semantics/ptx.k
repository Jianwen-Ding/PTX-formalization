// =============================================================================
// PTX: an executable semantics of NVIDIA PTX in the K Framework.
//
//   kompile ptx.k --main-module PTX --syntax-module PTX-SYNTAX --enable-search
//   krun program.ptx
// =============================================================================
requires "ptx-syntax.k"
requires "ptx-harness.k"

module PTX
  imports PTX-HARNESS
endmodule

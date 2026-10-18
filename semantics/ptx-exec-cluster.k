// =============================================================================
// PTX-EXEC-CLUSTER: thread block clusters and distributed shared memory (sm_90)
//
// Cluster dimensions come from the kernel's .reqnctapercluster (default 1x1x1); the grid is
// tiled by clusters and a CTA's rank inside its cluster is x + y*cx + z*cx*cy.
// A shared::cluster address is  (rank + 1) << 24 | offset ; an address with no rank bits
// (a plain shared::cta offset) designates the executing CTA, as the shared::cta window is
// a sub-window of shared::cluster.  mapa builds such addresses, getctarank decodes them.
// barrier.cluster.arrive/wait use a per-cluster generation counter (cell <clbar>).
// The generic address space carries the whole cluster's shared memory in the window starting at CLBASE (rank r at CLBASE + r * 2^24,
// see cvta.shared::cluster); the executing CTA's own shared::cta window at SHBASE is a sub-window of it.  A thread that exits is
// dropped from the cluster barrier, and an access to the shared memory of a CTA that has exited faults (undefined behaviour in the ISA).
// =============================================================================
requires "ptx-exec-mem.k"

module PTX-EXEC-CLUSTER
  imports PTX-EXEC-MEM

  // ---------------------------------------------------------------- geometry
  syntax Int ::= clTarget(Int, Int, Dim3, Dim3) [function, total]      // ctaid, rank -> ctaid of that cluster mate
  rule clTarget(C, R, dim3(GX, GY, _), dim3(CX, CY, CZ)) =>
       (((C %Int GX) /Int CX) *Int CX +Int R %Int CX)
       +Int ((((C /Int GX) %Int GY) /Int CY) *Int CY +Int (R /Int CX) %Int CY) *Int GX
       +Int (((C /Int (GX *Int GY)) /Int CZ) *Int CZ +Int R /Int (CX *Int CY)) *Int GX *Int GY
  syntax Int ::= clId(Int, Dim3, Dim3) [function, total]               // linear cluster id of a CTA
  rule clId(C, dim3(GX, GY, _), dim3(CX, CY, CZ)) =>
       ((C %Int GX) /Int CX) +Int (((C /Int GX) %Int GY) /Int CY) *Int (GX /Int CX)
       +Int ((C /Int (GX *Int GY)) /Int CZ) *Int (GX /Int CX) *Int (GY /Int CY)
  syntax Int ::= clSize(Dim3) [function, total]
  rule clSize(dim3(X, Y, Z)) => X *Int Y *Int Z

  // ---------------------------------------------------------------- distributed shared memory accesses
  rule mkLoc("shared_cluster", A) => loc("cluster", A)
  // own CTA
  rule <tk> #mem(loc("cluster", A), OP) => #mem(loc("shared", A &Int 16777215), OP) ... </tk>
       <ctaid> C </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
    requires A >>Int 24 ==Int 0 orBool clTarget(C, (A >>Int 24) -Int 1, GD, CD) ==Int C
  // another CTA of the cluster
  rule <cta> <ctaid> C </ctaid> <threads> ... <thread> <tk> #mem(loc("cluster", A), OP) => #res(memVal(OP, M, A &Int 16777215)) ... </tk> ... </thread> ... </threads> ... </cta>
       <cta> <ctaid> TC </ctaid> <smem> M => memNew(OP, M, A &Int 16777215) </smem> ... </cta>
       <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
    requires A >>Int 24 >Int 0 andBool (A >>Int 24) -Int 1 <Int clSize(CD) andBool TC ==Int clTarget(C, (A >>Int 24) -Int 1, GD, CD) andBool TC =/=Int C
  rule <tk> #mem(loc("cluster", A), _) => #fault("shared::cluster access to a CTA that has exited or is not in the cluster", A) ... </tk> [owise]

  // mapa.shared::cluster.type d, a, rank ;  getctarank.shared::cluster.type d, a
  rule <tk> exec(opc("mapa", _, ListItem(T)), ops(D, A, R, .Operands), C) => .K ... </tk>
       <regs> Rg => wr(Rg, D, (rd(A, T, C) &Int 16777215) |Int ((rd(R, ty("u", 32), C) +Int 1) <<Int 24)) </regs>
  rule <tk> exec(opc("getctarank", _, ListItem(T)), ops(D, A, .Operands), C) => .K ... </tk>
       <regs> Rg => wr(Rg, D, #if (rd(A, T, C) >>Int 24) ==Int 0 #then rreg("%cluster_ctarank", C) #else (rd(A, T, C) >>Int 24) -Int 1 #fi) </regs>

  // ---------------------------------------------------------------- barrier.cluster
  syntax KItem ::= #clWait(Int, Int)
  syntax ClBar ::= clState(Map, Int) [function, total]
  rule clState(CB, Id) => {CB[Id] orDefault cl(0, 0, 0)}:>ClBar
  syntax ClBar ::= clNext(ClBar, Int) [function, total]
  rule clNext(cl(N, G, X), T) => #if N +Int 1 >=Int T -Int X #then cl(0, G +Int 1, X) #else cl(N +Int 1, G, X) #fi
  syntax Int ::= clGen(ClBar) [function, total]
  rule clGen(cl(_, G, _)) => G
  // a thread that exits no longer takes part in the barrier: if it had already arrived in the open phase its arrival is withdrawn,
  // and the phase completes once every remaining thread has arrived
  syntax ClBar ::= clExit(ClBar, Bool, Int) [function, total]
  rule clExit(cl(N, G, X), Arrived, T) =>
       #if (N -Int b2i(Arrived)) >=Int T -Int (X +Int 1) andBool (N -Int b2i(Arrived)) >Int 0
       #then cl(0, G +Int 1, X +Int 1) #else cl(N -Int b2i(Arrived), G, X +Int 1) #fi
  rule <tk> #clExit => .K ... </tk>
       <ctaid> C </ctaid> <gridDim> GD </gridDim> <blockDim> BD </blockDim> <clusterDim> CD </clusterDim>
       <regs> R </regs>
       <clbar> CB => CB[clId(C, GD, CD) <- clExit(clState(CB, clId(C, GD, CD)), {R["%%clgen"] orDefault 0}:>Int >Int clGen(clState(CB, clId(C, GD, CD))), clSize(CD) *Int clSize(BD))] </clbar>
  rule <tk> exec(opc("barrier", M, _), ops(.Operands), _) => .K ... </tk>
       <ctaid> C </ctaid> <gridDim> GD </gridDim> <blockDim> BD </blockDim> <clusterDim> CD </clusterDim>
       <clbar> CB => CB[clId(C, GD, CD) <- clNext(clState(CB, clId(C, GD, CD)), clSize(CD) *Int clSize(BD))] </clbar>
       <regs> R => R["%%clgen" <- clGen(clState(CB, clId(C, GD, CD))) +Int 1] </regs>
    requires "cluster" in M andBool "arrive" in M
  rule <tk> exec(opc("barrier", M, _), ops(.Operands), _) => #yield ~> #clWait(clId(C, GD, CD), {R["%%clgen"] orDefault 0}:>Int) ... </tk>
       <ctaid> C </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim> <regs> R </regs>
    requires "cluster" in M andBool "wait" in M
  rule <tk> #clWait(Id, G) => .K ... </tk>
       <turn> -1 </turn>
       <clbar> CB </clbar>
    requires clGen(clState(CB, Id)) >=Int G
endmodule

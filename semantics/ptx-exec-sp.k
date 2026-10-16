// =============================================================================
// PTX-EXEC-SP: sparse matrix multiply-accumulate
//
//   mma.sp[::ordered_metadata].sync.aligned.<shape>.row.col[.satfinite]
//       dtype.atype.btype.ctype  d, a, b, c, e, f
//
// A (MxK) is structured sparse: each chunk of the row keeps only its non-zero
// elements, the compressed values live in the `a` registers and their positions in
// the metadata `e`; `f` is the sparsity selector (which lanes of a group of four
// supply the metadata).  The warp rendezvous expands A to the dense matrix and then
// reuses the dense mma.sync arithmetic of ptx-exec-mma.k.
//
// Chunking: 2:4 for 16-bit and 8-bit elements, 1:2 for tf32, pair-wise 4:8 for 4-bit
// elements.  Each chunk has a 4-bit metadata nibble; for 2:4 it holds two 2-bit
// positions, for 1:2 the position is bit 1 of the low index (0b0100 -> 0, 0b1110 -> 1),
// for 4:8 the two 2-bit values select the two non-zero 2-wide sub-chunks.
//
// Fragment layouts of A follow the register/column formulas of the PTX ISA.  The ISA
// gives the metadata layout only in figures, which could not be extracted as text; the
// assignment used here (documented in docs/prior-work.md) is the one the formulas
// imply: a row of nc chunks needs 4*nc metadata bits, i.e.
//   16 bits: one lane (index f) of the group holds rows g (bits 15:0) and g+8 (31:16);
//   32 bits: lane 2*f + rowhalf holds the whole row;
//   64 bits: lane 2*(chunk div 8) + rowhalf holds chunks 8*h .. 8*h+7 of the row.
// Not modelled: the .kind::*f8f6f4 / .kind::mx* forms and block scaling.
// =============================================================================
requires "ptx-exec-mma.k"

module PTX-EXEC-SP
  imports PTX-EXEC-MMA

  // chunk width (dense elements) and compressed elements per chunk
  syntax Int ::= spCw(Ty) [function, total] | spNz(Ty) [function, total]
  rule spCw(ty("tf", _)) => 2
  rule spCw(T) => 8 requires ew(T) ==Int 4
  rule spCw(_) => 4 [owise]
  rule spNz(ty("tf", _)) => 1
  rule spNz(T) => 4 requires ew(T) ==Int 4
  rule spNz(_) => 2 [owise]

  // compressed elements held by a lane: M*K/2/32
  syntax Int ::= spNel(String) [function, total]
  rule spNel(S) => shM(S) *Int shK(S) /Int 64

  // (row half, chunk) of compressed unit u (a chunk's worth of elements) of lane (g, t)
  syntax Int ::= spRh(Ty, Int) [function, total] | spChunk(Ty, Int, Int) [function, total]
  rule spRh(T, U) => U %Int 2 requires ew(T) >=Int 16
  rule spRh(T, U) => (U /Int 2) %Int 2 requires ew(T) <Int 16
  rule spChunk(T, U, Tg) => Tg +Int 4 *Int (U /Int 2) requires ew(T) >=Int 16
  rule spChunk(T, U, Tg) => 2 *Int Tg +Int U %Int 2 +Int 8 *Int (U /Int 4) requires ew(T) <Int 16

  // metadata nibble of (row half, chunk) of group g
  syntax Int ::= spNib(Map, Int, Int, Int, Int, Int) [function, total]     // arrivals, g, rowhalf, chunk, nc, selector
  rule spNib(Arr, G, Rh, C, Nc, Sel) => (spMeta(Arr, 4 *Int G +Int Sel) >>Int (16 *Int Rh +Int 4 *Int C)) &Int 15 requires Nc ==Int 4
  rule spNib(Arr, G, Rh, C, Nc, Sel) => (spMeta(Arr, 4 *Int G +Int 2 *Int Sel +Int Rh) >>Int (4 *Int C)) &Int 15 requires Nc ==Int 8
  rule spNib(Arr, G, Rh, C, Nc, Sel) => (spMeta(Arr, 4 *Int G +Int 2 *Int (C /Int 8) +Int Rh) >>Int (4 *Int (C %Int 8))) &Int 15 requires Nc ==Int 16
  rule spNib(_, _, _, _, _, _) => 0 [owise]
  syntax Int ::= spMeta(Map, Int) [function, total]
  rule spMeta(Arr, L) => at(listAt(Arr, L), size(listAt(Arr, L)) -Int 1)

  // dense column (within the chunk) of the j-th non-zero element
  syntax Int ::= spPos(Ty, Int, Int) [function, total]
  rule spPos(ty("tf", _), Nib, _) => (Nib &Int 3) /Int 2
  rule spPos(T, Nib, J) => 2 *Int ((Nib >>Int (2 *Int (J /Int 2))) &Int 3) +Int J %Int 2 requires ew(T) ==Int 4
  rule spPos(_, Nib, J) => (Nib >>Int (2 *Int J)) &Int 3 [owise]

  // dense A element map
  syntax Map ::= spFillLane(String, Ty, List, Map, Int, Int, Int, Map) [function, total]   // shape, atype, lane regs, arrivals, lane, i, selector, acc
  rule spFillLane(S, T, Rs, Arr, L, I, Sel, Acc) =>
       spFillLane(S, T, Rs, Arr, L, I +Int 1, Sel,
         Acc[rc(L /Int 4 +Int 8 *Int spRh(T, I /Int spNz(T)),
                spCw(T) *Int spChunk(T, I /Int spNz(T), L %Int 4)
                  +Int spPos(T, spNib(Arr, L /Int 4, spRh(T, I /Int spNz(T)), spChunk(T, I /Int spNz(T), L %Int 4), shK(S) /Int spCw(T), Sel),
                             I %Int spNz(T))) <- fragBits(Rs, 0, T, I)])
    requires I <Int spNel(S)
  rule spFillLane(_, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= spFillAll(String, Ty, Map, List, Int, Map) [function, total]
  rule spFillAll(S, T, Arr, ListItem(L:Int) Ls, Sel, Acc) => spFillAll(S, T, Arr, Ls, Sel, spFillLane(S, T, listAt(Arr, L), Arr, L, 0, Sel, Acc))
  rule spFillAll(_, _, _, _, _, Acc) => Acc [owise]

  // Data = [shape, dtype, atype, btype, ctype, mods, selector]
  rule collCompute("mma.sp", Data, Arr) =>
       scatterAll(shapeD(Data), dTy(Data),
                  resRows(shapeD(Data), dTy(Data), aTy(Data), bTy(Data), cTy(Data),
                          spFillAll(shapeD(Data), aTy(Data), Arr, keys_list(Arr), {Data[6]}:>Int, .Map),
                          fillAll(1, shapeD(Data), bTy(Data), Arr, keys_list(Arr), nElB(shapeD(Data)), regsOf(spNel(shapeD(Data)), aTy(Data)), .Map),
                          fillC(shapeD(Data), cTy(Data), Arr, keys_list(Arr), regsOf(spNel(shapeD(Data)), aTy(Data)) +Int regsOf(nElB(shapeD(Data)), bTy(Data)), .Map),
                          0, 0, modsD(Data), .Map),
                  keys_list(Arr))

  rule isCollective("mma") => true
  rule <tk> exec(opc("mma", M, ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT)), ops(Dv, Av, Bv, Cv, E, F, .Operands), C) =>
            #coll("mma.sp", 4294967295, ListItem(shapeOf(M)) ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT) ListItem(M) ListItem(rd(F, ty("u", 32), C)),
                  vecRegs(Av, AT, C) vecRegs(Bv, BT, C) vecRegs(Cv, CT, C) ListItem(rd(E, ty("b", 32), C)), vecOps(Dv)) ... </tk>
    requires "sync" in M andBool "sp" in M
endmodule

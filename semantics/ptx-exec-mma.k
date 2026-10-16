// =============================================================================
// PTX-EXEC-MMA: warp-level matrix multiply-accumulate.
//
//   mma.sync.aligned.<shape>.row.col[.satfinite].dtype.atype.btype.ctype  D, A, B, C
//   (and, in the wmma section below, wmma.load / wmma.store / wmma.mma)
//
// A warp executes D = A*B + C as a rendezvous (ptx-exec-coll.k): the 32 lanes
// contribute their register fragments, the full matrices are reassembled
// through the documented fragment layouts (groupID = lane/4, threadID_in_group =
// lane%4), multiplied exactly, rounded once into the accumulator format, and the
// D fragments are scattered back.
//
// Numerics: float products and sums are exact and rounded once (round to nearest
// even) into the accumulator type -- real tensor cores use implementation-defined
// intermediate rounding, which PTX does not specify.  Integer results wrap to 32
// bits (or saturate with .satfinite); .b1 uses xor.popc / and.popc.
// Layouts are transcribed from the PTX ISA; those of f16/bf16 m16n8k16 and s8
// m16n8k32 are checked against hardware by ZLUDA's mma_* programs, the rest
// follow the documentation.
// =============================================================================
requires "ptx-exec-mini.k"
requires "ptx-exec-warp.k"

module PTX-EXEC-MMA
  imports PTX-EXEC-MINI
  imports PTX-EXEC-WARP

  rule isCollective("mma") => true
  rule isCollective("wmma") => false

  // ---------------------------------------------------------------- shapes
  syntax Int ::= shM(String) [function, total] | shN(String) [function, total] | shK(String) [function, total]
  rule shM(S) => String2Int(substrString(S, 1, findString(S, "n", 0)))
  rule shN(S) => String2Int(substrString(S, findString(S, "n", 0) +Int 1, findString(S, "k", 0)))
  rule shK(S) => String2Int(substrString(S, findString(S, "k", 0) +Int 1, lengthString(S)))
  syntax String ::= shapeOf(Set) [function, total]
  rule shapeOf(M) => pick(M, ListItem("m8n8k4") ListItem("m8n8k16") ListItem("m8n8k32") ListItem("m8n8k128")
                             ListItem("m16n8k4") ListItem("m16n8k8") ListItem("m16n8k16") ListItem("m16n8k32")
                             ListItem("m16n8k64") ListItem("m16n8k128") ListItem("m16n8k256")
                             ListItem("m16n16k16") ListItem("m8n32k16") ListItem("m32n8k16") ListItem("m16n16k8"), "")

  // ---------------------------------------------------------------- element decode / encode
  syntax Bool ::= isFloatElem(Ty) [function, total]
  rule isFloatElem(ty(K, _)) => K in (SetItem("f") SetItem("bf") SetItem("tf") SetItem("e4m3") SetItem("e5m2"))
  syntax Float ::= elemF(Ty, Int) [function, total]
  rule elemF(ty("f", W), B) => roundFloat(decodeF(B, W), 300, WEXP)
  rule elemF(ty("bf", _), B) => roundFloat(decodeF(B, 116), 300, WEXP)
  rule elemF(ty("tf", _), B) => roundFloat(decodeF(B &Int (~Int 8191), 32), 300, WEXP)
  rule elemF(ty("e4m3", _), B) => roundFloat(decodeF(miniToF16("e4m3", B), 16), 300, WEXP)
  rule elemF(ty("e5m2", _), B) => roundFloat(decodeF(miniToF16("e5m2", B), 16), 300, WEXP)
  rule elemF(_, _) => Int2Float(0, 300, WEXP) [owise]
  syntax Int ::= elemI(Ty, Int) [function, total]
  rule elemI(T, B) => ival(B, T)
  // element width of the A/B operands in bits
  syntax Int ::= ew(Ty) [function, total]
  rule ew(ty("e4m3", _)) => 8   rule ew(ty("e5m2", _)) => 8
  rule ew(ty(_, W)) => W [owise]

  // ---------------------------------------------------------------- fragment layouts
  // (row, col) encoded as row * 1024 + col
  syntax Int ::= posA(String, Int, Int, Int) [function, total]     // shape, element bits, lane, index
               | posB(String, Int, Int, Int) [function, total]
               | posC(String, Int, Int) [function, total]
  rule posA(S, 16, L, I) => rc(L /Int 4 +Int 8 *Int ((I /Int 2) %Int 2), 2 *Int (L %Int 4) +Int I %Int 2 +Int 8 *Int (I /Int 4)) requires S ==String "m16n8k16"
  rule posA(S, 16, L, I) => rc(L /Int 4 +Int 8 *Int (I /Int 2), 2 *Int (L %Int 4) +Int I %Int 2) requires S ==String "m16n8k8"
  rule posA(S, 32, L, I) => rc(L /Int 4 +Int 8 *Int (I %Int 2), L %Int 4 +Int 4 *Int (I /Int 2)) requires S ==String "m16n8k8"
  rule posA(S, 32, L, I) => rc(L /Int 4 +Int 8 *Int I, L %Int 4) requires S ==String "m16n8k4"
  rule posA(S, 64, L, I) => rc(L /Int 4, L %Int 4) requires S ==String "m8n8k4"
  rule posA(S, 64, L, I) => rc(L /Int 4 +Int 8 *Int I, L %Int 4) requires S ==String "m16n8k4"
  rule posA(S, 64, L, I) => rc(L /Int 4 +Int 8 *Int (I %Int 2), L %Int 4 +Int 4 *Int (I /Int 2)) requires S ==String "m16n8k8" orBool S ==String "m16n8k16"
  rule posA(S, 8, L, I) => rc(L /Int 4, 4 *Int (L %Int 4) +Int I) requires S ==String "m8n8k16"
  rule posA(S, 8, L, I) => rc(L /Int 4 +Int 8 *Int (I /Int 4), 4 *Int (L %Int 4) +Int I %Int 4) requires S ==String "m16n8k16"
  rule posA(S, 8, L, I) => rc(L /Int 4 +Int 8 *Int ((I /Int 4) %Int 2), 4 *Int (L %Int 4) +Int I %Int 4 +Int 16 *Int (I /Int 8)) requires S ==String "m16n8k32"
  rule posA(S, 4, L, I) => rc(L /Int 4, 8 *Int (L %Int 4) +Int I) requires S ==String "m8n8k32"
  rule posA(S, 4, L, I) => rc(L /Int 4 +Int 8 *Int (I /Int 8), 8 *Int (L %Int 4) +Int I %Int 8) requires S ==String "m16n8k32"
  rule posA(S, 4, L, I) => rc(L /Int 4 +Int 8 *Int ((I /Int 8) %Int 2), 8 *Int (L %Int 4) +Int I %Int 8 +Int 32 *Int (I /Int 16)) requires S ==String "m16n8k64"
  rule posA(S, 1, L, I) => rc(L /Int 4, 32 *Int (L %Int 4) +Int I) requires S ==String "m8n8k128"
  rule posA(S, 1, L, I) => rc(L /Int 4 +Int 8 *Int (I /Int 32), 32 *Int (L %Int 4) +Int I %Int 32) requires S ==String "m16n8k128"
  rule posA(S, 1, L, I) => rc(L /Int 4 +Int 8 *Int ((I /Int 32) %Int 2), 32 *Int (L %Int 4) +Int I %Int 32 +Int 128 *Int (I /Int 64)) requires S ==String "m16n8k256"
  rule posA(_, _, _, _) => 0 -Int 1 [owise]

  rule posB(S, 16, L, I) => rc(2 *Int (L %Int 4) +Int I %Int 2 +Int 8 *Int (I /Int 2), L /Int 4) requires S ==String "m16n8k16"
  rule posB(S, 16, L, I) => rc(2 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m16n8k8"
  rule posB(S, 32, L, I) => rc(L %Int 4 +Int 4 *Int I, L /Int 4) requires S ==String "m16n8k8" orBool S ==String "m16n8k4"
  rule posB(S, 64, L, I) => rc(L %Int 4 +Int 4 *Int I, L /Int 4) requires S ==String "m8n8k4" orBool S ==String "m16n8k4" orBool S ==String "m16n8k8" orBool S ==String "m16n8k16"
  rule posB(S, 8, L, I) => rc(4 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m8n8k16" orBool S ==String "m16n8k16"
  rule posB(S, 8, L, I) => rc(4 *Int (L %Int 4) +Int I %Int 4 +Int 16 *Int (I /Int 4), L /Int 4) requires S ==String "m16n8k32"
  rule posB(S, 4, L, I) => rc(8 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m8n8k32" orBool S ==String "m16n8k32"
  rule posB(S, 4, L, I) => rc(8 *Int (L %Int 4) +Int I %Int 8 +Int 32 *Int (I /Int 8), L /Int 4) requires S ==String "m16n8k64"
  rule posB(S, 1, L, I) => rc(32 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m8n8k128" orBool S ==String "m16n8k128"
  rule posB(S, 1, L, I) => rc(32 *Int (L %Int 4) +Int I %Int 32 +Int 128 *Int (I /Int 32), L /Int 4) requires S ==String "m16n8k256"
  rule posB(_, _, _, _) => 0 -Int 1 [owise]

  rule posC(S, L, I) => rc(L /Int 4, 2 *Int (L %Int 4) +Int I) requires shM(S) ==Int 8
  rule posC(S, L, I) => rc(L /Int 4 +Int 8 *Int (I /Int 2), 2 *Int (L %Int 4) +Int I %Int 2) requires shM(S) ==Int 16
  syntax Int ::= rc(Int, Int) [function, total]
  rule rc(R, C) => R *Int 1024 +Int C

  // ---------------------------------------------------------------- fragments <-> matrices
  // register width (bits) of a fragment element type
  syntax Int ::= regBits(Ty) [function, total]
  rule regBits(ty("f", 64)) => 64
  rule regBits(_) => 32 [owise]
  syntax Int ::= perReg(Ty) [function, total]                       // elements per register
  rule perReg(T) => regBits(T) /Int ew(T)
  syntax Int ::= nElA(String) [function, total] | nElB(String) [function, total] | nElC(String) [function, total]
  rule nElA(S) => shM(S) *Int shK(S) /Int 32
  rule nElB(S) => shK(S) *Int shN(S) /Int 32
  rule nElC(S) => shM(S) *Int shN(S) /Int 32
  syntax Int ::= regsOf(Int, Ty) [function, total]                  // number of registers for n elements
  rule regsOf(N, T) => (N +Int perReg(T) -Int 1) /Int perReg(T)

  // element I of a lane's register list
  syntax Int ::= fragBits(List, Int, Ty, Int) [function, total]     // regs, offset of the fragment in the list, type, element index
  rule fragBits(Rs, Off, T, I) => (at(Rs, Off +Int I /Int perReg(T)) >>Int (ew(T) *Int (I %Int perReg(T)))) &Int mask(ew(T))

  // ---------------------------------------------------------------- gathering the matrices
  syntax Map ::= fillLane(Int, String, Ty, List, Int, Int, Int, Int, Map) [function, total]
  // which (0 = A, 1 = B), shape, type, regs of the lane, lane, element, count, register offset, accumulator
  rule fillLane(Wh, S, T, Rs, L, I, N, Off, Acc) =>
       fillLane(Wh, S, T, Rs, L, I +Int 1, N, Off,
                Acc[#if Wh ==Int 0 #then posA(S, ew(T), L, I) #else posB(S, ew(T), L, I) #fi <- fragBits(Rs, Off, T, I)])
    requires I <Int N
  rule fillLane(_, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= fillAll(Int, String, Ty, Map, List, Int, Int, Map) [function, total]
  rule fillAll(Wh, S, T, Arr, ListItem(L:Int) Ls, N, Off, Acc) =>
       fillAll(Wh, S, T, Arr, Ls, N, Off, fillLane(Wh, S, T, listAt(Arr, L), L, 0, N, Off, Acc))
  rule fillAll(_, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= fillC(String, Ty, Map, List, Int, Map) [function, total]            // C accumulator fragments
  rule fillC(S, T, Arr, ListItem(L:Int) Ls, Off, Acc) => fillC(S, T, Arr, Ls, Off, fillCLane(S, T, listAt(Arr, L), L, 0, Off, Acc))
  rule fillC(_, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= fillCLane(String, Ty, List, Int, Int, Int, Map) [function, total]
  rule fillCLane(S, T, Rs, L, I, Off, Acc) => fillCLane(S, T, Rs, L, I +Int 1, Off, Acc[posC(S, L, I) <- fragBits(Rs, Off, T, I)])
    requires I <Int nElC(S)
  rule fillCLane(_, _, _, _, _, _, Acc) => Acc [owise]

  // ---------------------------------------------------------------- the product
  // dot product of row R of A and column C of B (value maps keyed by position)
  syntax Float ::= dotF(Ty, Ty, Map, Map, Int, Int, Int, Int, Float) [function, total]    // ..., r, c, k, K, acc
  rule dotF(AT, BT, A, B, R, Cn, I, K, Acc) =>
       dotF(AT, BT, A, B, R, Cn, I +Int 1, K,
            Acc +Float elemF(AT, {A[rc(R, I)] orDefault 0}:>Int) *Float elemF(BT, {B[rc(I, Cn)] orDefault 0}:>Int))
    requires I <Int K
  rule dotF(_, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= dotI(Ty, Ty, Map, Map, Int, Int, Int, Int, Int, String) [function, total]    // ..., acc, b1 op ("xor"/"and"/"")
  rule dotI(AT, BT, A, B, R, Cn, I, K, Acc, Op) =>
       dotI(AT, BT, A, B, R, Cn, I +Int 1, K,
            Acc +Int #if Op ==String "xor" #then ({A[rc(R, I)] orDefault 0}:>Int xorInt {B[rc(I, Cn)] orDefault 0}:>Int)
                     #else #if Op ==String "and" #then ({A[rc(R, I)] orDefault 0}:>Int &Int {B[rc(I, Cn)] orDefault 0}:>Int)
                     #else elemI(AT, {A[rc(R, I)] orDefault 0}:>Int) *Int elemI(BT, {B[rc(I, Cn)] orDefault 0}:>Int) #fi #fi, Op)
    requires I <Int K
  rule dotI(_, _, _, _, _, _, _, _, Acc, _) => Acc [owise]

  // one result element, as bits of the D type
  syntax Int ::= resElem(Ty, Ty, Ty, Ty, Map, Map, Map, Int, Int, Int, Set) [function, total]
  // dtype, atype, btype, ctype, A, B, C, r, c, K, mods
  rule resElem(DT, AT, BT, CT, A, B, C, R, Cn, K, M) =>
       fpFin(#if "satfinite" in M #then SetItem("satfinite") #else .Set #fi, fw(DT),
             dotF(AT, BT, A, B, R, Cn, 0, K, elemF(CT, {C[rc(R, Cn)] orDefault 0}:>Int)), 0)
    requires isFloatElem(AT)
  rule resElem(DT, AT, BT, CT, A, B, C, R, Cn, K, M) =>
       #if "satfinite" in M #then trunc(clamp(dotI(AT, BT, A, B, R, Cn, 0, K, ival({C[rc(R, Cn)] orDefault 0}:>Int, CT), ""), DT), 32)
       #else trunc(dotI(AT, BT, A, B, R, Cn, 0, K, ival({C[rc(R, Cn)] orDefault 0}:>Int, CT), ""), 32) #fi
    requires notBool isFloatElem(AT) andBool ew(AT) >Int 1
  rule resElem(DT, AT, BT, CT, A, B, C, R, Cn, K, M) =>
       trunc(dotI(AT, BT, A, B, R, Cn, 0, K, ival({C[rc(R, Cn)] orDefault 0}:>Int, CT), #if "and" in M #then "and" #else "xor" #fi) , 32)
    requires ew(AT) ==Int 1
  // note: for .b1, dotI above adds raw xor/and bits; the popcount is the sum over k, as intended

  // ---------------------------------------------------------------- scattering D
  syntax List ::= laneRegs(String, Ty, Map, Int, Int, Int, Int, Int, List) [function, total]    // shape, DT, D values, lane, element, count, per-reg, (unused), acc
  // pack the lane's D elements into registers
  rule laneRegs(S, DT, DV, L, I, N, Per, Cur, Acc) =>
       laneRegs(S, DT, DV, L, I +Int Per, N, Per, 0,
                Acc ListItem(packEl(DT, DV, S, L, I, Per, 0)))
    requires I <Int N
  rule laneRegs(_, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= packEl(Ty, Map, String, Int, Int, Int, Int) [function, total]
  rule packEl(DT, DV, S, L, I, Per, J) =>
       ({DV[posC(S, L, I +Int J)] orDefault 0}:>Int <<Int (ew(DT) *Int J)) |Int packEl(DT, DV, S, L, I, Per, J +Int 1)
    requires J <Int Per
  rule packEl(_, _, _, _, _, _, _) => 0 [owise]
  syntax Map ::= scatterAll(String, Ty, Map, List) [function, total]
  rule scatterAll(S, DT, DV, ListItem(L:Int) Ls) =>
       (L |-> laneRegs(S, DT, DV, L, 0, nElC(S), perReg(DT), 0, .List)) scatterAll(S, DT, DV, Ls)
  rule scatterAll(_, _, _, _) => .Map [owise]

  // all result elements: position -> bits
  syntax Map ::= resRows(String, Ty, Ty, Ty, Ty, Map, Map, Map, Int, Int, Set, Map) [function, total]
  rule resRows(S, DT, AT, BT, CT, A, B, C, R, Cn, M, Acc) =>
       resRows(S, DT, AT, BT, CT, A, B, C, #if Cn +Int 1 ==Int shN(S) #then R +Int 1 #else R #fi,
               #if Cn +Int 1 ==Int shN(S) #then 0 #else Cn +Int 1 #fi, M,
               Acc[rc(R, Cn) <- resElem(DT, AT, BT, CT, A, B, C, R, Cn, shK(S), M)])
    requires R <Int shM(S)
  rule resRows(_, _, _, _, _, _, _, _, _, _, _, Acc) => Acc [owise]

  // ---------------------------------------------------------------- mma.sync
  // Data = [shape, dtype-name...] is passed through the types encoded in the kind string
  rule collCompute(Kind, Data, Arr) =>
       scatterAll(shapeD(Data), dTy(Data),
                  resRows(shapeD(Data), dTy(Data), aTy(Data), bTy(Data), cTy(Data),
                          fillAll(0, shapeD(Data), aTy(Data), Arr, keys_list(Arr), nElA(shapeD(Data)), 0, .Map),
                          fillAll(1, shapeD(Data), bTy(Data), Arr, keys_list(Arr), nElB(shapeD(Data)), regsOf(nElA(shapeD(Data)), aTy(Data)), .Map),
                          fillC(shapeD(Data), cTy(Data), Arr, keys_list(Arr), regsOf(nElA(shapeD(Data)), aTy(Data)) +Int regsOf(nElB(shapeD(Data)), bTy(Data)), .Map),
                          0, 0, modsD(Data), .Map),
                  keys_list(Arr))
    requires Kind ==String "mma"
  syntax String ::= shapeD(List) [function, total]
  rule shapeD(L) => {L[0]}:>String
  syntax Ty ::= dTy(List) [function, total] | aTy(List) [function, total] | bTy(List) [function, total] | cTy(List) [function, total]
  rule dTy(L) => {L[1]}:>Ty   rule aTy(L) => {L[2]}:>Ty   rule bTy(L) => {L[3]}:>Ty   rule cTy(L) => {L[4]}:>Ty
  syntax Set ::= modsD(List) [function, total]
  rule modsD(L) => {L[5]}:>Set

  syntax List ::= vecRegs(Operand, Ty, Ctx) [function, total]
  rule vecRegs({ Os }, T, C) => regList(opsList(Os), T, C)
  rule vecRegs(_, _, _) => .List [owise]
  syntax List ::= regList(List, Ty, Ctx) [function, total]
  rule regList(ListItem(O:Operand) Os, T, C) => ListItem(rd(O, ty("b", regBits(T)), C)) regList(Os, T, C)
  rule regList(_, _, _) => .List [owise]
  syntax List ::= vecOps(Operand) [function, total]
  rule vecOps({ Os }) => opsList(Os)
  rule vecOps(_) => .List [owise]

  rule <tk> exec(opc("mma", M, ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT)), ops(Dv, Av, Bv, Cv, .Operands), C) =>
            #coll("mma", 4294967295, ListItem(shapeOf(M)) ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT) ListItem(M),
                  vecRegs(Av, AT, C) vecRegs(Bv, BT, C) vecRegs(Cv, CT, C), vecOps(Dv)) ... </tk>
    requires "sync" in M
endmodule

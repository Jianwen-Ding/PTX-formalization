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
requires "ptx-exec-mem.k"

module PTX-EXEC-MMA
  imports PTX-EXEC-MINI
  imports PTX-EXEC-WARP
  imports PTX-EXEC-MEM

  rule isCollective("mma") => true

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
  rule isFloatElem(ty(K, _)) => K in (SetItem("f") SetItem("bf") SetItem("tf") SetItem("e4m3") SetItem("e5m2") SetItem("e2m3") SetItem("e3m2") SetItem("e2m1"))
  syntax Float ::= elemF(Ty, Int) [function, total]
  rule elemF(ty("f", W), B) => roundFloat(decodeF(B, W), 300, WEXP)
  rule elemF(ty("bf", _), B) => roundFloat(decodeF(B, 116), 300, WEXP)
  rule elemF(ty("tf", _), B) => roundFloat(decodeF(B &Int (~Int 8191), 32), 300, WEXP)
  rule elemF(ty("e4m3", _), B) => roundFloat(decodeF(miniToF16("e4m3", B), 16), 300, WEXP)
  rule elemF(ty("e5m2", _), B) => roundFloat(decodeF(miniToF16("e5m2", B), 16), 300, WEXP)
  rule elemF(ty("e2m3", _), B) => roundFloat(decodeF(miniToF16("e2m3", B &Int 63), 16), 300, WEXP)
  rule elemF(ty("e3m2", _), B) => roundFloat(decodeF(miniToF16("e3m2", B &Int 63), 16), 300, WEXP)
  rule elemF(ty("e2m1", _), B) => roundFloat(decodeF(miniToF16("e2m1", B &Int 15), 16), 300, WEXP)
  rule elemF(_, _) => Int2Float(0, 300, WEXP) [owise]
  syntax Int ::= elemI(Ty, Int) [function, total]
  rule elemI(T, B) => ival(B, T)
  // element width of the A/B operands in bits
  syntax Int ::= ew(Ty) [function, total]
  rule ew(ty("e4m3", _)) => 8   rule ew(ty("e5m2", _)) => 8   rule ew(ty("e2m3", _)) => 8   rule ew(ty("e3m2", _)) => 8   rule ew(ty("e2m1", _)) => 8   // fp6/fp4 in 8-bit containers
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

  rule posB(S, 16, L, I) => rc(2 *Int (L %Int 4) +Int I %Int 2 +Int 8 *Int (I /Int 2), L /Int 4) requires S ==String "m16n8k16" orBool S ==String "m16n8k32"
  rule posB(S, 16, L, I) => rc(2 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m16n8k8"
  rule posB(S, 32, L, I) => rc(L %Int 4 +Int 4 *Int I, L /Int 4) requires S ==String "m16n8k8" orBool S ==String "m16n8k4" orBool S ==String "m16n8k16"
  rule posB(S, 64, L, I) => rc(L %Int 4 +Int 4 *Int I, L /Int 4) requires S ==String "m8n8k4" orBool S ==String "m16n8k4" orBool S ==String "m16n8k8" orBool S ==String "m16n8k16"
  rule posB(S, 8, L, I) => rc(4 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m8n8k16" orBool S ==String "m16n8k16"
  rule posB(S, 8, L, I) => rc(4 *Int (L %Int 4) +Int I %Int 4 +Int 16 *Int (I /Int 4), L /Int 4) requires S ==String "m16n8k32" orBool S ==String "m16n8k64"
  rule posB(S, 4, L, I) => rc(8 *Int (L %Int 4) +Int I, L /Int 4) requires S ==String "m8n8k32" orBool S ==String "m16n8k32"
  rule posB(S, 4, L, I) => rc(8 *Int (L %Int 4) +Int I %Int 8 +Int 32 *Int (I /Int 8), L /Int 4) requires S ==String "m16n8k64" orBool S ==String "m16n8k128"
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
    requires "sync" in M andBool notBool "sp" in M

  // ================================================================ wmma
  // Fragment layouts of wmma are unspecified in PTX, so any consistent assignment is
  // valid: element I of lane L holds matrix element ((L * NEL + I) mod size) in row-major
  // order, which covers the matrix and replicates elements when the fragments are larger
  // than the matrix (as the register counts in the ISA require).
  syntax Int ::= wmRegs(String, String, Ty) [function, total]          // matrix (a/b/c/d), shape, element type
  rule wmRegs("c", S, T) => 4 requires T ==K ty("f", 16)
  rule wmRegs("c", S, T) => 2 requires shM(S) ==Int 8 andBool shN(S) ==Int 8 andBool notBool T ==K ty("f", 16)
  rule wmRegs("c", S, T) => 8 requires shM(S) ==Int 16 andBool shN(S) ==Int 16 andBool notBool T ==K ty("f", 16)
  rule wmRegs("c", _, _) => 8 [owise]
  rule wmRegs("d", S, T) => wmRegs("c", S, T)
  rule wmRegs(Mat, S, T) => 8 requires (Mat ==String "a" orBool Mat ==String "b") andBool T ==K ty("f", 16)
  rule wmRegs(Mat, S, ty("bf", _)) => #if Mat ==String "a" #then (#if S ==String "m16n16k16" #then 4 #else #if S ==String "m8n32k16" #then 2 #else 8 #fi #fi)
                                       #else (#if S ==String "m16n16k16" #then 4 #else #if S ==String "m8n32k16" #then 8 #else 2 #fi #fi) #fi
    requires Mat ==String "a" orBool Mat ==String "b"
  rule wmRegs(Mat, _, ty("tf", _)) => 4 requires Mat ==String "a" orBool Mat ==String "b"
  rule wmRegs(Mat, _, ty("f", 64)) => 1 requires Mat ==String "a" orBool Mat ==String "b"
  rule wmRegs(Mat, S, ty(K, 8)) => #if Mat ==String "a" #then (#if S ==String "m16n16k16" #then 2 #else #if S ==String "m8n32k16" #then 1 #else 4 #fi #fi)
                                    #else (#if S ==String "m16n16k16" #then 2 #else #if S ==String "m8n32k16" #then 4 #else 1 #fi #fi) #fi
    requires (Mat ==String "a" orBool Mat ==String "b") andBool (K ==String "s" orBool K ==String "u")
  rule wmRegs(Mat, _, ty(_, 4)) => 1 requires Mat ==String "a" orBool Mat ==String "b"
  rule wmRegs(Mat, _, ty("b", 1)) => 1 requires Mat ==String "a" orBool Mat ==String "b"
  rule wmRegs(_, _, _) => 1 [owise]

  syntax Int ::= wmNel(String, String, Ty) [function, total]            // elements per lane
  rule wmNel(Mat, S, T) => wmRegs(Mat, S, T) *Int perReg(T)
  syntax Int ::= wmRows(String, String) [function, total] | wmCols(String, String) [function, total]
  rule wmRows("a", S) => shM(S)   rule wmCols("a", S) => shK(S)
  rule wmRows("b", S) => shK(S)   rule wmCols("b", S) => shN(S)
  rule wmRows(_, S) => shM(S) [owise]   rule wmCols(_, S) => shN(S) [owise]
  // matrix position of element I of lane L (fragment of `Mat`, element type T)
  syntax Int ::= wmPos(String, String, Ty, Int, Int) [function, total]
  rule wmPos(Mat, S, T, L, I) => rc(((L *Int wmNel(Mat, S, T) +Int I) %Int (wmRows(Mat, S) *Int wmCols(Mat, S))) /Int wmCols(Mat, S),
                                    ((L *Int wmNel(Mat, S, T) +Int I) %Int (wmRows(Mat, S) *Int wmCols(Mat, S))) %Int wmCols(Mat, S))

  // ---------------------------------------------------------------- wmma.mma
  syntax Map ::= wmFill(String, String, Ty, Map, List, Int, Map) [function, total]       // matrix, shape, type, arrivals, lanes, reg offset, acc
  rule wmFill(Mat, S, T, Arr, ListItem(L:Int) Ls, Off, Acc) =>
       wmFill(Mat, S, T, Arr, Ls, Off, wmFillLane(Mat, S, T, listAt(Arr, L), L, 0, Off, Acc))
  rule wmFill(_, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= wmFillLane(String, String, Ty, List, Int, Int, Int, Map) [function, total]
  rule wmFillLane(Mat, S, T, Rs, L, I, Off, Acc) =>
       wmFillLane(Mat, S, T, Rs, L, I +Int 1, Off, Acc[wmPos(Mat, S, T, L, I) <- fragBits(Rs, Off, T, I)])
    requires I <Int wmNel(Mat, S, T)
  rule wmFillLane(_, _, _, _, _, _, _, Acc) => Acc [owise]

  syntax List ::= wmLaneRegs(String, Ty, Map, Int, Int, List) [function, total]          // shape, type, D values, lane, reg, acc
  rule wmLaneRegs(S, T, DV, L, R, Acc) => wmLaneRegs(S, T, DV, L, R +Int 1, Acc ListItem(wmPack(S, T, DV, L, R *Int perReg(T), 0)))
    requires R <Int wmRegs("c", S, T)
  rule wmLaneRegs(_, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= wmPack(String, Ty, Map, Int, Int, Int) [function, total]
  rule wmPack(S, T, DV, L, I, J) => ({DV[wmPos("c", S, T, L, I +Int J)] orDefault 0}:>Int <<Int (ew(T) *Int J)) |Int wmPack(S, T, DV, L, I, J +Int 1)
    requires J <Int perReg(T)
  rule wmPack(_, _, _, _, _, _) => 0 [owise]
  syntax Map ::= wmScatter(String, Ty, Map, List) [function, total]
  rule wmScatter(S, T, DV, ListItem(L:Int) Ls) => (L |-> wmLaneRegs(S, T, DV, L, 0, .List)) wmScatter(S, T, DV, Ls)
  rule wmScatter(_, _, _, _) => .Map [owise]

  // positions in D follow the layout of the accumulator fragment: re-key the result matrix
  rule collCompute("wmma", Data, Arr) =>
       wmScatter(shapeD(Data), dTy(Data),
                 resRows(shapeD(Data), dTy(Data), aTy(Data), bTy(Data), cTy(Data),
                         wmFill("a", shapeD(Data), aTy(Data), Arr, keys_list(Arr), 0, .Map),
                         wmFill("b", shapeD(Data), bTy(Data), Arr, keys_list(Arr), wmRegs("a", shapeD(Data), aTy(Data)), .Map),
                         wmFill("c", shapeD(Data), cTy(Data), Arr, keys_list(Arr), wmRegs("a", shapeD(Data), aTy(Data)) +Int wmRegs("b", shapeD(Data), bTy(Data)), .Map),
                         0, 0, modsD(Data), .Map),
                 keys_list(Arr))

  // wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype d, a, b, c           (f16)
  // wmma.mma.sync.aligned.alayout.blayout.shape.dtype.atype.btype.ctype d, a, b, c
  rule isCollective("wmma") => true
  rule <tk> exec(opc("wmma", M, ListItem(DT) ListItem(CT)), ops(Dv, Av, Bv, Cv, .Operands), C) =>
            #coll("wmma", 4294967295, ListItem(shapeOf(M)) ListItem(DT) ListItem(ty("f", 16)) ListItem(ty("f", 16)) ListItem(CT) ListItem(M),
                  vecRegs(Av, ty("f", 16), C) vecRegs(Bv, ty("f", 16), C) vecRegs(Cv, CT, C), vecOps(Dv)) ... </tk>
    requires "mma" in M
  rule <tk> exec(opc("wmma", M, ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT)), ops(Dv, Av, Bv, Cv, .Operands), C) =>
            #coll("wmma", 4294967295, ListItem(shapeOf(M)) ListItem(DT) ListItem(AT) ListItem(BT) ListItem(CT) ListItem(M),
                  vecRegs(Av, AT, C) vecRegs(Bv, BT, C) vecRegs(Cv, CT, C), vecOps(Dv)) ... </tk>
    requires "mma" in M

  // ---------------------------------------------------------------- wmma.load / wmma.store
  // element (r, c) of a matrix in memory: row layout p + (r*stride + c) elements, col layout p + (c*stride + r)
  syntax Int ::= elemIndex(Bool, Int, Int, Int) [function, total]                 // col-major?, r, c, stride
  rule elemIndex(Col, R, Cn, St) => #if Col #then Cn *Int St +Int R #else R *Int St +Int Cn #fi
  // a fragment element in memory: (byte location, bit offset in that byte)
  syntax WmEl ::= wmEl(Loc, Int)
  syntax List ::= wmLocs(String, String, Ty, Int, Bool, Int, Loc, Int, Int) [function, total]   // mat, shape, type, lane, col?, stride, base, I, N
  rule wmLocs(Mat, S, T, L, Col, St, loc(Sp, A), I, N) =>
       ListItem(wmEl(loc(Sp, A +Int (elemIndex(Col, wmPos(Mat, S, T, L, I) /Int 1024, wmPos(Mat, S, T, L, I) %Int 1024, St) *Int ew(T)) /Int 8),
                    (elemIndex(Col, wmPos(Mat, S, T, L, I) /Int 1024, wmPos(Mat, S, T, L, I) %Int 1024, St) *Int ew(T)) %Int 8))
       wmLocs(Mat, S, T, L, Col, St, loc(Sp, A), I +Int 1, N)
    requires I <Int N
  rule wmLocs(_, _, _, _, _, _, _, _, _) => .List [owise]
  rule wmLocs(Mat, S, T, L, Col, St, badLoc(E), I, N) => ListItem(wmEl(badLoc(E), 0)) [priority(40)]
  syntax Int ::= wmBytes(Ty) [function, total]
  rule wmBytes(T) => #if ew(T) <Int 8 #then 1 #else ew(T) /Int 8 #fi

  syntax KItem ::= #wmLoad(List, List, Ty, Operand) | #wmGot(Int, Ty, List, List, Operand)
                 | #wmWr(List, List) | #wmStore(List, List, Ty)
  syntax List ::= wmPackRegs(List, Ty) [function, total]          // elements -> registers
  rule wmPackRegs(Es, T) => ListItem(wmPackOne(Es, T, 0)) wmPackRegs(dropN(Es, perReg(T)), T) requires size(Es) >Int 0
  rule wmPackRegs(_, _) => .List [owise]
  syntax Int ::= wmPackOne(List, Ty, Int) [function, total]
  rule wmPackOne(Es, T, J) => ((at(Es, J) &Int mask(ew(T))) <<Int (ew(T) *Int J)) |Int wmPackOne(Es, T, J +Int 1) requires J <Int perReg(T) andBool J <Int size(Es)
  rule wmPackOne(_, _, _) => 0 [owise]
  syntax List ::= dropN(List, Int) [function, total]
  rule dropN(ListItem(_) Ls, N) => dropN(Ls, N -Int 1) requires N >Int 0
  rule dropN(Ls, _) => Ls [owise]

  rule <tk> exec(opc("wmma", M, ListItem(T)), ops(Dv, [ A ], St, .Operands), C) =>
            #wmLoad(wmLocs(pick(M, ListItem("a") ListItem("b") ListItem("c"), "a"), shapeOf(M), T, TID %Int 32, "col" in M,
                           rd(St, ty("u", 32), C), ea(A, spaceOf(M), C), 0, wmNel(pick(M, ListItem("a") ListItem("b") ListItem("c"), "a"), shapeOf(M), T)),
                    .List, T, Dv) ... </tk>
       <tid> TID </tid>
    requires "load" in M
  rule <tk> #wmLoad(ListItem(wmEl(L, O)) Ls, Acc, T, Dv) =>
            #mem(L, rdOp(wmBytes(T))) ~> #wmGot(O, T, Ls, Acc, Dv) ... </tk>
  rule <tk> #wmLoad(ListItem(wmEl(badLoc(S), _)) _, _, _, _) => #fault(S, 0) ... </tk> [priority(40)]
  rule <tk> #res(V) ~> #wmGot(O, T, Ls, Acc, Dv) => #wmLoad(Ls, Acc ListItem((V >>Int O) &Int mask(ew(T))), T, Dv) ... </tk>
  rule <tk> #wmLoad(.List, Acc, T, Dv) => #wmWr(vecOps(Dv), wmPackRegs(Acc, T)) ... </tk>
  rule <tk> #wmWr(ListItem(D:Operand) Ds, ListItem(V:Int) Vs) => #wmWr(Ds, Vs) ... </tk>
       <regs> R => wr(R, D, V) </regs>
  rule <tk> #wmWr(_, _) => .K ... </tk> [owise]

  // wmma.store.d.sync.aligned.layout.shape.space.type [p], {r...}, stride
  rule <tk> exec(opc("wmma", M, ListItem(T)), ops([ A ], Vs, St, .Operands), C) =>
            #wmStore(wmLocs("c", shapeOf(M), T, TID %Int 32, "col" in M,
                            rd(St, ty("u", 32), C), ea(A, spaceOf(M), C), 0, wmNel("c", shapeOf(M), T)),
                     wmElems(vecRegs(Vs, T, C), T, 0, wmNel("c", shapeOf(M), T)), T) ... </tk>
       <tid> TID </tid>
    requires "store" in M
  syntax List ::= wmElems(List, Ty, Int, Int) [function, total]
  rule wmElems(Rs, T, I, N) => ListItem(fragBits(Rs, 0, T, I)) wmElems(Rs, T, I +Int 1, N) requires I <Int N
  rule wmElems(_, _, _, _) => .List [owise]
  rule <tk> #wmStore(ListItem(wmEl(L, O)) Ls, ListItem(V:Int) Vs, T) =>
            #mem(L, wmStOp(T, O, V)) ~> #drop ~> #wmStore(Ls, Vs, T) ... </tk>
  rule <tk> #wmStore(ListItem(wmEl(badLoc(S), _)) _, _, _) => #fault(S, 0) ... </tk> [priority(40)]
  rule <tk> #wmStore(.List, _, _) => .K ... </tk>
  syntax MemOp ::= wmStOp(Ty, Int, Int) [function, total]
  rule wmStOp(T, O, V) => insOp(O, ew(T), V) requires ew(T) <Int 8
  rule wmStOp(T, _, V) => wrOp(ew(T) /Int 8, V) [owise]
endmodule

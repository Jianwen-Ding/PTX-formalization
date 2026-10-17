// =============================================================================
// PTX-EXEC-WG: warpgroup-level matrix multiply-accumulate (sm_90a)
//
//   wgmma.mma_async.sync.aligned.m64nNkK.dtype.atype.btype  d, a|a-desc, b-desc, scale-d, ...
//   wgmma.fence / wgmma.commit_group / wgmma.wait_group
//
// The 128 threads of a warpgroup rendezvous (collective kind "wgmma"); the last thread to
// arrive snapshots the CTA's shared memory, from which the B (and, in the descriptor
// form, A) tiles are read through the canonical swizzled layouts of the PTX ISA
// (matrix descriptor: start address, leading/stride byte offsets, swizzle mode).
// The multiplication itself is performed at that point, i.e. the asynchronous operation
// completes before the next instruction; this is one of the legal executions for any
// program that follows the fence/commit/wait protocol, so fence, commit_group and
// wait_group are no-ops.  Registers of D are updated in the same step.
//
// Accumulator D: warp w of the warpgroup owns rows 16w..16w+15 with the m16n8
// accumulator layout repeated every 8 columns (element e: row g + 8*((e%4)/2),
// column 8*(e/4) + 2*t + e%2).  A in registers uses the mma.m16n8kK A layout per warp.
// Sparse A (wgmma.mma_async.sp): each warp's fragment and metadata follow the mma.sp m16n8kK layout of ptx-exec-sp.k
// (an assumption, see docs/prior-work.md); the descriptor form reads the packed M x K/2 tile and expands it with the metadata.
// =============================================================================
requires "ptx-exec-sp.k"

module PTX-EXEC-WG
  imports PTX-EXEC-SP

  rule isCollective("wgmma") => true
  rule <tk> exec(opc("wgmma", M, _), ops(_), _) => .K ... </tk> requires "fence" in M orBool "commit_group" in M
  rule <tk> exec(opc("wgmma", M, _), ops(_, .Operands), _) => .K ... </tk> requires "wait_group" in M

  syntax String ::= wgShape(Set) [function, total]
  rule wgShape(M) => pick(M, ListItem("m64n8k8") ListItem("m64n16k8") ListItem("m64n24k8") ListItem("m64n32k8") ListItem("m64n40k8") ListItem("m64n48k8") ListItem("m64n56k8") ListItem("m64n64k8") ListItem("m64n72k8") ListItem("m64n80k8") ListItem("m64n88k8") ListItem("m64n96k8") ListItem("m64n104k8") ListItem("m64n112k8") ListItem("m64n120k8") ListItem("m64n128k8") ListItem("m64n136k8") ListItem("m64n144k8") ListItem("m64n152k8") ListItem("m64n160k8") ListItem("m64n168k8") ListItem("m64n176k8") ListItem("m64n184k8") ListItem("m64n192k8") ListItem("m64n200k8") ListItem("m64n208k8") ListItem("m64n216k8") ListItem("m64n224k8") ListItem("m64n232k8") ListItem("m64n240k8") ListItem("m64n248k8") ListItem("m64n256k8") ListItem("m64n8k16") ListItem("m64n16k16") ListItem("m64n24k16") ListItem("m64n32k16") ListItem("m64n40k16") ListItem("m64n48k16") ListItem("m64n56k16") ListItem("m64n64k16") ListItem("m64n72k16") ListItem("m64n80k16") ListItem("m64n88k16") ListItem("m64n96k16") ListItem("m64n104k16") ListItem("m64n112k16") ListItem("m64n120k16") ListItem("m64n128k16") ListItem("m64n136k16") ListItem("m64n144k16") ListItem("m64n152k16") ListItem("m64n160k16") ListItem("m64n168k16") ListItem("m64n176k16") ListItem("m64n184k16") ListItem("m64n192k16") ListItem("m64n200k16") ListItem("m64n208k16") ListItem("m64n216k16") ListItem("m64n224k16") ListItem("m64n232k16") ListItem("m64n240k16") ListItem("m64n248k16") ListItem("m64n256k16") ListItem("m64n8k32") ListItem("m64n16k32") ListItem("m64n24k32") ListItem("m64n32k32") ListItem("m64n40k32") ListItem("m64n48k32") ListItem("m64n56k32") ListItem("m64n64k32") ListItem("m64n72k32") ListItem("m64n80k32") ListItem("m64n88k32") ListItem("m64n96k32") ListItem("m64n104k32") ListItem("m64n112k32") ListItem("m64n120k32") ListItem("m64n128k32") ListItem("m64n136k32") ListItem("m64n144k32") ListItem("m64n152k32") ListItem("m64n160k32") ListItem("m64n168k32") ListItem("m64n176k32") ListItem("m64n184k32") ListItem("m64n192k32") ListItem("m64n200k32") ListItem("m64n208k32") ListItem("m64n216k32") ListItem("m64n224k32") ListItem("m64n232k32") ListItem("m64n240k32") ListItem("m64n248k32") ListItem("m64n256k32") ListItem("m64n8k256") ListItem("m64n16k256") ListItem("m64n24k256") ListItem("m64n32k256") ListItem("m64n40k256") ListItem("m64n48k256") ListItem("m64n56k256") ListItem("m64n64k256") ListItem("m64n72k256") ListItem("m64n80k256") ListItem("m64n88k256") ListItem("m64n96k256") ListItem("m64n104k256") ListItem("m64n112k256") ListItem("m64n120k256") ListItem("m64n128k256") ListItem("m64n136k256") ListItem("m64n144k256") ListItem("m64n152k256") ListItem("m64n160k256") ListItem("m64n168k256") ListItem("m64n176k256") ListItem("m64n184k256") ListItem("m64n192k256") ListItem("m64n200k256") ListItem("m64n208k256") ListItem("m64n216k256") ListItem("m64n224k256") ListItem("m64n232k256") ListItem("m64n240k256") ListItem("m64n248k256") ListItem("m64n256k256") ListItem("m64n8k64") ListItem("m64n16k64") ListItem("m64n24k64") ListItem("m64n32k64") ListItem("m64n40k64") ListItem("m64n48k64") ListItem("m64n56k64") ListItem("m64n64k64") ListItem("m64n72k64") ListItem("m64n80k64") ListItem("m64n88k64") ListItem("m64n96k64") ListItem("m64n104k64") ListItem("m64n112k64") ListItem("m64n120k64") ListItem("m64n128k64") ListItem("m64n136k64") ListItem("m64n144k64") ListItem("m64n152k64") ListItem("m64n160k64") ListItem("m64n168k64") ListItem("m64n176k64") ListItem("m64n184k64") ListItem("m64n192k64") ListItem("m64n200k64") ListItem("m64n208k64") ListItem("m64n216k64") ListItem("m64n224k64") ListItem("m64n232k64") ListItem("m64n240k64") ListItem("m64n248k64") ListItem("m64n256k64"), "")

  // ---------------------------------------------------------------- shared-memory tiles
  // field extraction of the matrix descriptor
  syntax Int ::= dStart(Int) [function, total] | dLbo(Int) [function, total] | dSbo(Int) [function, total] | dSwz(Int) [function, total]
  rule dStart(D) => (D &Int 16383) <<Int 4
  rule dLbo(D) => ((D >>Int 16) &Int 16383) <<Int 4
  rule dSbo(D) => ((D >>Int 32) &Int 16383) <<Int 4
  rule dSwz(D) => ((D >>Int 62) &Int 3) +Int 4 *Int ((D >>Int 56) &Int 1)           // sm_90 modes 0..3; bit 56 (internal): the sm_100 / TMA sub-modes 4..6
  // atom rows per swizzle: 128B -> 8, 64B -> 4, 32B -> 2 (units of 16 bytes)
  syntax Int ::= swG(Int) [function, total]
  rule swG(1) => 8   rule swG(2) => 4   rule swG(3) => 2   rule swG(4) => 8   rule swG(5) => 8   rule swG(6) => 8   rule swG(_) => 1 [owise]
  syntax Int ::= swB(Int) [function, total]                 // swizzle bits B of Swizzle<B,4,3>
  rule swB(1) => 3   rule swB(2) => 2   rule swB(3) => 1   rule swB(_) => 0 [owise]
  syntax Int ::= swizzle(Int, Int) [function, total]        // swizzle mode, byte address
  rule swizzle(Mode, A) => A xorInt ((A &Int (mask(swB(Mode)) <<Int 7)) >>Int 3) requires Mode <=Int 3
  // 128B swizzle with 32B atomicity (4), plus an 8-byte flip on odd 128-byte lines (5), and with 64B atomicity (6): 16-byte chunk index c of a 128-byte line
  // becomes c xor 2 * (line mod 4) (pairs of chunks move together), c xor 4 * (line mod 2) (groups of four)
  rule swizzle(4, A) => A xorInt (((A >>Int 7) &Int 3) <<Int 5)
  rule swizzle(5, A) => A xorInt (((A >>Int 7) &Int 3) <<Int 5) xorInt (((A >>Int 7) &Int 1) <<Int 3)
  rule swizzle(6, A) => A xorInt (((A >>Int 7) &Int 1) <<Int 6)
  rule swizzle(Mode, A) => A [owise]

  // element offset (in elements) of (mn, k) in the canonical layout; T = elements per 16 bytes
  // (small helper functions keep every rule free of parenthesised sub-expressions)
  syntax Int ::= md8(Int) [function, total] | dv8(Int) [function, total] | mdT(Int, Int) [function, total] | dvT(Int, Int) [function, total]
  rule md8(X) => X %Int 8
  rule dv8(X) => X /Int 8
  rule mdT(X, T) => X %Int T
  rule dvT(X, T) => X /Int T
  syntax Int ::= wgEO(Int, Bool, Int, Int, Int, Int, Int) [function, total]     // swizzle, K-major?, T, mn, k, LBO, SBO (in elements)
  rule wgEO(0, true, T, MN, K, L, S) => md8(MN) *Int T +Int dv8(MN) *Int S +Int mdT(K, T) +Int dvT(K, T) *Int L
  rule wgEO(Z, true, T, MN, K, L, S) => md8(MN) *Int swG(Z) *Int T +Int dv8(MN) *Int S +Int mdT(K, T) +Int dvT(K, T) *Int T
    requires Z >Int 0
  rule wgEO(0, false, T, MN, K, L, S) => mdT(MN, T) +Int dvT(MN, T) *Int S +Int md8(K) *Int T +Int dv8(K) *Int L
  rule wgEO(Z, false, T, MN, K, L, S) =>
       mdT(MN, T) +Int mdT(dvT(MN, T), swG(Z)) *Int T +Int dvT(MN, T *Int swG(Z)) *Int L +Int md8(K) *Int swG(Z) *Int T +Int dv8(K) *Int S
    requires Z >Int 0

  // element offset of (mn, k) in elements, and in bits
  syntax Int ::= wgBitOff(Int, Bool, Int, Int, Int) [function, total]            // descriptor, K-major?, element bits, mn, k
  rule wgBitOff(D, KM, W, MN, K) => wgEO(dSwz(D), KM, 128 /Int W, MN, K, wgScaleOff(dLbo(D), W), wgScaleOff(dSbo(D), W)) *Int W
  syntax Int ::= wgScaleOff(Int, Int) [function, total]                           // byte offset -> elements
  rule wgScaleOff(B, W) => (B *Int 8) /Int W
  syntax Int ::= wgAddr(Int, Int) [function, total]                               // descriptor, bit offset -> swizzled byte address
  rule wgAddr(D, BO) => swizzle(dSwz(D), dStart(D) +Int (BO /Int 8))
  syntax Int ::= wgElem(Bytes, Int, Bool, Int, Int, Int) [function, total]        // smem, descriptor, K-major?, element bits, mn, k
  rule wgElem(SM, D, KM, W, MN, K) => wgRead(SM, wgAddr(D, wgBitOff(D, KM, W, MN, K)), wgBitOff(D, KM, W, MN, K) %Int 8, W)
  syntax Int ::= wgRead(Bytes, Int, Int, Int) [function, total]                    // smem, byte address, bit within byte, element bits
  rule wgRead(SM, A, BI, W) => (readBytes(SM, A, maxInt(1, W /Int 8)) >>Int BI) &Int mask(W)

  // negation of float inputs (imm-scale = -1)
  syntax Int ::= wgNeg(Ty, Int, Int) [function, total]
  rule wgNeg(T, Sc, V) => V xorInt (1 <<Int (ew(T) -Int 1)) requires Sc <Int 0 andBool isFloatElem(T)
  rule wgNeg(_, _, V) => V [owise]

  // ---------------------------------------------------------------- assembling the operands
  // Data = [shape, dtype, atype, btype, mods, descA, descB, scaleA, scaleB, transA, transB, scaleD, aInRegs, smem]
  syntax Bytes ::= wgSmem(List) [function, total]
  rule wgSmem(L) => {L[13]}:>Bytes
  syntax Int ::= wgI(List, Int) [function, total]
  rule wgI(L, I) => {L[I]}:>Int

  syntax Map ::= wgTileDesc(String, Ty, Bytes, Int, Bool, Int, Int, Int, Map) [function, total]   // rows/cols: R x C tile; K-major?, scale, i, acc
  // B tile (K x N), element (k, n)
  syntax Map ::= wgB(List, Int, Int, Map) [function, total]
  rule wgB(Data, K, N, Acc) =>
       wgB(Data, K, N +Int 1, Acc[rc(K, N) <- wgNeg(bTy(Data), wgI(Data, 8),
                                    wgElem(wgSmem(Data), wgI(Data, 6), wgI(Data, 10) ==Int 0, ew(bTy(Data)), N, K))])
    requires N <Int shN(shapeD(Data)) andBool K <Int shK(shapeD(Data))
  rule wgB(Data, K, N, Acc) => wgB(Data, K +Int 1, 0, Acc) requires N >=Int shN(shapeD(Data)) andBool K +Int 1 <Int shK(shapeD(Data))
  rule wgB(_, _, _, Acc) => Acc [owise]
  // A tile (M x K), element (m, k), from the descriptor
  syntax Map ::= wgAd(List, Int, Int, Map) [function, total]
  rule wgAd(Data, M, K, Acc) =>
       wgAd(Data, M, K +Int 1, Acc[rc(M, K) <- wgNeg(aTy(Data), wgI(Data, 7),
                                    wgElem(wgSmem(Data), wgI(Data, 5), wgI(Data, 9) ==Int 0, ew(aTy(Data)), M, K))])
    requires K <Int shK(shapeD(Data)) andBool M <Int 64
  rule wgAd(Data, M, K, Acc) => wgAd(Data, M +Int 1, 0, Acc) requires K >=Int shK(shapeD(Data)) andBool M +Int 1 <Int 64
  rule wgAd(_, _, _, Acc) => Acc [owise]

  // registers: thread u holds, after the D registers, the A fragment of its warp
  syntax Int ::= wgND(String, Ty) [function, total]               // D registers per thread
  rule wgND(S, T) => regsOf(shN(S) /Int 2, T)
  syntax Int ::= posD(Int, Int) [function, total]                  // lane, element
  rule posD(L, E) => rc(L /Int 4 +Int 8 *Int ((E %Int 4) /Int 2), 8 *Int (E /Int 4) +Int 2 *Int (L %Int 4) +Int E %Int 2)

  syntax Map ::= wgFillD(String, Ty, Map, List, Map) [function, total]
  rule wgFillD(S, T, Arr, ListItem(U:Int) Us, Acc) => wgFillD(S, T, Arr, Us, wgFillDU(S, T, listAt(Arr, U), U, 0, Acc))
  rule wgFillD(_, _, _, _, Acc) => Acc [owise]
  syntax Map ::= wgFillDU(String, Ty, List, Int, Int, Map) [function, total]
  rule wgFillDU(S, T, Rs, U, E, Acc) =>
       wgFillDU(S, T, Rs, U, E +Int 1, Acc[rc(16 *Int (U /Int 32) +Int posD(U %Int 32, E) /Int 1024, posD(U %Int 32, E) %Int 1024) <- fragBits(Rs, 0, T, E)])
    requires E <Int shN(S) /Int 2
  rule wgFillDU(_, _, _, _, _, Acc) => Acc [owise]

  syntax Map ::= wgFillA(List, Map, List, Map) [function, total]
  rule wgFillA(Data, Arr, ListItem(U:Int) Us, Acc) => wgFillA(Data, Arr, Us, wgFillAU(Data, listAt(Arr, U), U, 0, Acc))
  rule wgFillA(_, _, _, Acc) => Acc [owise]
  syntax Map ::= wgFillAU(List, List, Int, Int, Map) [function, total]
  rule wgFillAU(Data, Rs, U, E, Acc) =>
       wgFillAU(Data, Rs, U, E +Int 1,
                Acc[rc(16 *Int (U /Int 32) +Int posA("m16n8k" +String Int2String(shK(shapeD(Data))), ew(aTy(Data)), U %Int 32, E) /Int 1024,
                       posA("m16n8k" +String Int2String(shK(shapeD(Data))), ew(aTy(Data)), U %Int 32, E) %Int 1024)
                    <- wgNeg(aTy(Data), wgI(Data, 7), fragBits(Rs, wgND(shapeD(Data), dTy(Data)), aTy(Data), E))])
    requires E <Int 16 *Int shK(shapeD(Data)) /Int 32
  rule wgFillAU(_, _, _, _, Acc) => Acc [owise]

  // pack results
  syntax List ::= wgRegs(String, Ty, Map, Int, Int, List) [function, total]
  rule wgRegs(S, T, DV, U, R, Acc) => wgRegs(S, T, DV, U, R +Int 1, Acc ListItem(wgPack(S, T, DV, U, R *Int perReg(T), 0)))
    requires R <Int wgND(S, T)
  rule wgRegs(_, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= wgPack(String, Ty, Map, Int, Int, Int) [function, total]
  rule wgPack(S, T, DV, U, E, J) =>
       ({DV[rc(16 *Int (U /Int 32) +Int posD(U %Int 32, E +Int J) /Int 1024, posD(U %Int 32, E +Int J) %Int 1024)] orDefault 0}:>Int <<Int (ew(T) *Int J))
         |Int wgPack(S, T, DV, U, E, J +Int 1)
    requires J <Int perReg(T)
  rule wgPack(_, _, _, _, _, _) => 0 [owise]
  syntax Map ::= wgScatter(String, Ty, Map, List) [function, total]
  rule wgScatter(S, T, DV, ListItem(U:Int) Us) => (U |-> wgRegs(S, T, DV, U, 0, .List)) wgScatter(S, T, DV, Us)
  rule wgScatter(_, _, _, _) => .Map [owise]

  // ---------------------------------------------------------------- sparse A
  syntax Map ::= wgAMap(List, Map) [function, total]
  rule wgAMap(Data, Arr) => wgFillA(Data, Arr, keys_list(Arr), .Map) requires notBool ("sp" in {Data[4]}:>Set) andBool wgI(Data, 12) ==Int 1
  rule wgAMap(Data, Arr) => wgAd(Data, 0, 0, .Map) requires notBool ("sp" in {Data[4]}:>Set) andBool wgI(Data, 12) =/=Int 1
  rule wgAMap(Data, Arr) => wgNegAll(wgSpRegs(Data, Arr, 0, .Map), aTy(Data), wgI(Data, 7), keys_list(wgSpRegs(Data, Arr, 0, .Map)), .Map) requires "sp" in {Data[4]}:>Set andBool wgI(Data, 12) ==Int 1
  rule wgAMap(Data, Arr) => wgSpDesc(Data, Arr, 0, 0, .Map) requires "sp" in {Data[4]}:>Set andBool wgI(Data, 12) =/=Int 1
  syntax Map ::= wgNegAll(Map, Ty, Int, List, Map) [function, total]
  rule wgNegAll(M, T, Sc, ListItem(K:Int) Ks, Acc) => wgNegAll(M, T, Sc, Ks, Acc[K <- wgNeg(T, Sc, {M[K]}:>Int)])
  rule wgNegAll(_, _, _, _, Acc) => Acc [owise]
  syntax String ::= wgSpShape(List) [function, total]
  rule wgSpShape(Data) => "m16n8k" +String Int2String(shK(shapeD(Data)))
  // the 32 threads of warp W: lane -> registers (D, A, metadata)
  syntax Map ::= wgSlice(Map, Int, Int, Map) [function, total]
  rule wgSlice(Arr, W, L, Acc) => wgSlice(Arr, W, L +Int 1, Acc[L <- listAt(Arr, 32 *Int W +Int L)]) requires L <Int 32
  rule wgSlice(_, _, _, Acc) => Acc [owise]
  syntax Map ::= wgSpShift(Map, Int) [function, total]
  rule wgSpShift(M, Off) => wgShiftRows(M, Off, keys_list(M), .Map)
  syntax Map ::= wgShiftRows(Map, Int, List, Map) [function, total]
  rule wgShiftRows(M, Off, ListItem(K:Int) Ks, Acc) => wgShiftRows(M, Off, Ks, Acc[K +Int Off <- {M[K]}:>Int])
  rule wgShiftRows(_, _, _, Acc) => Acc [owise]
  // A from registers: the m16n8kK sparse fragment of each warp, rows offset by 16 * warp
  syntax Map ::= wgSpRegs(List, Map, Int, Map) [function, total]
  rule wgSpRegs(Data, Arr, W, Acc) =>
       wgSpRegs(Data, Arr, W +Int 1, updateMap(Acc, wgSpShift(wgSpRegW(Data, Arr, wgSlice(Arr, W, 0, .Map), W, 0, .Map), 16384 *Int W)))
    requires W <Int 4
  rule wgSpRegs(_, _, _, Acc) => Acc [owise]
  syntax Map ::= wgSpRegW(List, Map, Map, Int, Int, Map) [function, total]       // Data, Arr, warp slice, W, lane, acc
  rule wgSpRegW(Data, Arr, ArrW, W, L, Acc) =>
       wgSpRegW(Data, Arr, ArrW, W, L +Int 1,
                spFillLane(wgSpShape(Data), aTy(Data), range(listAt(ArrW, L), wgND(shapeD(Data), dTy(Data)), 0), ArrW, L, 0, wgI(Data, 14), Acc))
    requires L <Int 32
  rule wgSpRegW(_, _, _, _, _, Acc) => Acc [owise]
  // A from shared memory: the packed M x (K/2) tile through the descriptor, expanded by the metadata
  syntax Int ::= wgNibW(Map, Int, Int, Int, Int, Int, Int) [function, total]      // arrivals, warp, g, rowhalf, chunk, chunks per row, selector
  rule wgNibW(Arr, W, G, Rh, C, Nc, Sel) => (spMeta(Arr, 32 *Int W +Int 4 *Int G +Int Sel) >>Int (16 *Int Rh +Int 4 *Int C)) &Int 15 requires Nc ==Int 4
  rule wgNibW(Arr, W, G, Rh, C, Nc, Sel) => (spMeta(Arr, 32 *Int W +Int 4 *Int G +Int 2 *Int Sel +Int Rh) >>Int (4 *Int C)) &Int 15 requires Nc ==Int 8
  rule wgNibW(Arr, W, G, Rh, C, Nc, Sel) => (spMeta(Arr, 32 *Int W +Int 4 *Int G +Int 2 *Int (C /Int 8) +Int Rh) >>Int (4 *Int (C %Int 8))) &Int 15 requires Nc ==Int 16
  rule wgNibW(_, _, _, _, _, _, _) => 0 [owise]
  syntax Map ::= wgSpDesc(List, Map, Int, Int, Map) [function, total]              // Data, Arr, m, k', acc
  rule wgSpDesc(Data, Arr, M, KP, Acc) =>
       wgSpDesc(Data, Arr, M, KP +Int 1,
         Acc[rc(M, spCw(aTy(Data)) *Int (KP /Int spNz(aTy(Data)))
                   +Int spPos(aTy(Data),
                              wgNibW(Arr, M /Int 16, (M %Int 16) %Int 8, (M %Int 16) /Int 8, KP /Int spNz(aTy(Data)),
                                     shK(shapeD(Data)) /Int spCw(aTy(Data)), wgI(Data, 14)),
                              KP %Int spNz(aTy(Data))))
             <- wgNeg(aTy(Data), wgI(Data, 7), wgElem(wgSmem(Data), wgI(Data, 5), wgI(Data, 9) ==Int 0, ew(aTy(Data)), M, KP))])
    requires KP <Int shK(shapeD(Data)) /Int 2 andBool M <Int 64
  rule wgSpDesc(Data, Arr, M, KP, Acc) => wgSpDesc(Data, Arr, M +Int 1, 0, Acc) requires KP >=Int shK(shapeD(Data)) /Int 2 andBool M +Int 1 <Int 64
  rule wgSpDesc(_, _, _, _, Acc) => Acc [owise]

  rule collCompute("wgmma", Data, Arr) =>
       wgScatter(shapeD(Data), dTy(Data),
                 resRows(shapeD(Data), dTy(Data), aTy(Data), bTy(Data), dTy(Data),
                         wgAMap(Data, Arr),
                         wgB(Data, 0, 0, .Map),
                         #if wgI(Data, 11) =/=Int 0 #then wgFillD(shapeD(Data), dTy(Data), Arr, keys_list(Arr), .Map) #else .Map #fi,
                         0, 0, {Data[4]}:>Set, .Map),
                 keys_list(Arr))

  // wgmma.mma_async.sync.aligned.shape.dtype.atype.btype d, a|a-desc, b-desc, scale-d, imm...
  rule <tk> exec(opc("wgmma", M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(Dv, Av, Bv, Rest:Operands), C) =>
            #coll("wgmma", 340282366920938463463374607431768211455,
                  ListItem(wgShape(M)) ListItem(DT) ListItem(AT) ListItem(BT) ListItem(M)
                  ListItem(#if isVec(Av) #then 0 #else rd(Av, ty("b", 64), C) #fi) ListItem(rd(Bv, ty("b", 64), C))
                  ListItem(wgArg(opsList(Rest), 1, 1, C)) ListItem(wgArg(opsList(Rest), 2, 1, C))
                  ListItem(wgArg(opsList(Rest), 3, 0, C)) ListItem(wgArg(opsList(Rest), #if size(opsList(Rest)) >=Int 4 #then size(opsList(Rest)) -Int 1 #else 0 -Int 1 #fi, 0, C))
                  ListItem(wgArg(opsList(Rest), 0, 1, C)) ListItem(#if isVec(Av) #then 1 #else 0 #fi) ListItem(SM),
                  vecRegs(Dv, DT, C) vecRegs(Av, AT, C), vecOps(Dv)) ... </tk>
       <smem> SM </smem>
    requires "mma_async" in M andBool notBool "sp" in M
  // sparse: d, a|a-desc, b-desc, sp-meta, sp-sel, scale-d, ...
  rule <tk> exec(opc("wgmma", M, ListItem(DT) ListItem(AT) ListItem(BT)), ops(Dv, Av, Bv, Meta, Sel, Rest:Operands), C) =>
            #coll("wgmma", 340282366920938463463374607431768211455,
                  ListItem(wgShape(M)) ListItem(DT) ListItem(AT) ListItem(BT) ListItem(M)
                  ListItem(#if isVec(Av) #then 0 #else rd(Av, ty("b", 64), C) #fi) ListItem(rd(Bv, ty("b", 64), C))
                  ListItem(wgArg(opsList(Rest), 1, 1, C)) ListItem(wgArg(opsList(Rest), 2, 1, C))
                  ListItem(wgArg(opsList(Rest), 3, 0, C)) ListItem(wgArg(opsList(Rest), #if size(opsList(Rest)) >=Int 4 #then size(opsList(Rest)) -Int 1 #else 0 -Int 1 #fi, 0, C))
                  ListItem(wgArg(opsList(Rest), 0, 1, C)) ListItem(#if isVec(Av) #then 1 #else 0 #fi) ListItem(SM) ListItem(rd(Sel, ty("u", 32), C)),
                  vecRegs(Dv, DT, C) vecRegs(Av, AT, C) ListItem(rd(Meta, ty("b", 32), C)), vecOps(Dv)) ... </tk>
       <smem> SM </smem>
    requires "mma_async" in M andBool "sp" in M
  // optional trailing immediates; the transposes are the last operand (desc form of 16-bit types: 5 args, register form: 4)
  syntax Int ::= wgArg(List, Int, Int, Ctx) [function, total]
  rule wgArg(L, I, _, C) => ival(rd({L[I]}:>Operand, ty("s", 32), C), ty("s", 32)) requires I >=Int 0 andBool I <Int size(L)
  rule wgArg(_, _, D, _) => D [owise]
endmodule

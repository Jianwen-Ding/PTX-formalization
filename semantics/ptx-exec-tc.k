// =============================================================================
// PTX-EXEC-TC: fifth-generation tensor core (tcgen05, sm_100a)
//
//   tcgen05.alloc / dealloc / relinquish_alloc_permit     tensor memory management
//   tcgen05.ld / tcgen05.st                               registers <-> tensor memory
//   tcgen05.wait / fence                                  (no-ops: operations complete in program order)
//   tcgen05.mma.cta_group::1.kind::{f16,tf32,i8,f8f6f4}   D = A*B + D, dense, single-thread issue
//
// Tensor memory is a per-CTA array of 128 lanes x 512 32-bit columns (cell <tmem>); a
// tensor-memory address is lane << 16 | column.  Asynchronous operations are executed
// atomically at issue (one legal schedule for programs that wait/fence correctly).
//
// Layout assumptions that the ISA gives only as figures: tcgen05.ld/st shapes other than
// .32x32b (lane t of the warp reads lane t, column i) use lane t%16 and the column mapping
// documented in the rules below; the M=128 accumulator layout puts row r in lane r and column
// j of N in column j (one element per 32-bit word, low bits); A in tensor memory is K-major,
// packed little-endian inside a column.  Shared-memory operands use the sm_100 descriptor
// (swizzle codes 2/4/6 for 128B/64B/32B) with the canonical layouts of ptx-exec-wg.k.
// Sparse A (tcgen05.mma.sp) uses an assumed metadata layout (see below).  Not modelled: .cta_group::2, block scaling, .ws, collector/ashift forms,
// decompression, tcgen05.cp/shift/commit (commit follows with mbarrier), ld.red/spcompress.
// =============================================================================
requires "ptx-exec-wg.k"
requires "ptx-exec-ldm.k"
requires "ptx-exec-mbar.k"

module PTX-EXEC-TC
  imports PTX-EXEC-WG
  imports PTX-EXEC-LDM
  imports PTX-EXEC-MBAR

  rule isCollective("tcgen05") => true      // makes the instructions visible steps (they touch shared state)

  syntax Int ::= tcLane(Int) [function, total] | tcCol(Int) [function, total]
  rule tcLane(A) => (A >>Int 16) &Int 65535
  rule tcCol(A) => A &Int 65535
  syntax Int ::= tcKey(Int, Int) [function, total]
  rule tcKey(L, C) => L *Int 65536 +Int C
  syntax Int ::= tcAddr(Addr, Ctx) [function, total]
  rule tcAddr(A, C) => locOff(ea(A, "global", C))

  rule <tk> exec(opc("tcgen05", M, _), ops(_), _) => .K ... </tk>
    requires "wait" in M orBool "fence" in M orBool "relinquish_alloc_permit" in M

  // ---------------------------------------------------------------- allocation
  // first-fit, aligned to the allocation size
  syntax Int ::= tcFind(Map, Int, Int) [function, total]
  rule tcFind(Al, N, B) => B requires B +Int N <=Int 512 andBool notBool tcOverlap(Al, B, N, keys_list(Al))
  rule tcFind(Al, N, B) => tcFind(Al, N, B +Int N) requires B +Int N <=Int 512 andBool tcOverlap(Al, B, N, keys_list(Al))
  rule tcFind(_, _, _) => 0 -Int 1 [owise]
  syntax Bool ::= tcOverlap(Map, Int, Int, List) [function, total]
  rule tcOverlap(Al, B, N, ListItem(K:Int) Ks) => (B <Int K +Int {Al[K]}:>Int andBool K <Int B +Int N) orBool tcOverlap(Al, B, N, Ks)
  rule tcOverlap(_, _, _, _) => false [owise]
  syntax Bool ::= tcPow2(Int) [function, total]
  rule tcPow2(N) => N >=Int 32 andBool N <=Int 512 andBool popc(N) ==Int 1

  // alloc / dealloc are warp-collective: the lanes rendezvous, the first one through performs the
  // operation and records the result under the instance (warp, kind, sequence number)
  syntax String ::= tcInst(Int, Map, String) [function, total]
  rule tcInst(T, CS, K) => Int2String(T /Int 32) +String ":" +String K +String ":" +String Int2String({CS[seqKey(K, 4294967295)] orDefault 1}:>Int -Int 1)
  syntax KItem ::= #tcAlloc(Loc, Int) | #tcDealloc(Int, Int)
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], N, .Operands), C) =>
            #coll("tcgen05.alloc", 4294967295, .List, ListItem(0), ListItem(stmTmp())) ~> #tcAlloc(ea(D, "shared", C), rd(N, ty("u", 32), C)) ... </tk>
    requires "alloc" in M
  rule <tk> #tcAlloc(L, N) => #mem(L, wrOp(4, {R[tcInst(T, CS, "tcgen05.alloc")]}:>Int)) ~> #drop ... </tk>
       <tid> T </tid> <cseq> CS </cseq> <tmres> R </tmres>
    requires tcInst(T, CS, "tcgen05.alloc") in_keys(R)
  rule <tk> #tcAlloc(L, N) => #mem(L, wrOp(4, tcFind(Al, N, 0))) ~> #drop ... </tk>
       <tid> T </tid> <cseq> CS </cseq>
       <tmres> R => R[tcInst(T, CS, "tcgen05.alloc") <- tcFind(Al, N, 0)] </tmres>
       <tmalloc> Al => Al[tcFind(Al, N, 0) <- N] </tmalloc>
    requires notBool tcInst(T, CS, "tcgen05.alloc") in_keys(R) andBool tcPow2(N) andBool tcFind(Al, N, 0) >=Int 0
  rule <tk> #tcAlloc(_, N) => #fault("tcgen05.alloc: invalid size or tensor memory exhausted", N) ... </tk> [owise]
  rule <tk> exec(opc("tcgen05", M, _), ops(T, N, .Operands), C) =>
            #coll("tcgen05.dealloc", 4294967295, .List, ListItem(0), ListItem(stmTmp())) ~> #tcDealloc(tcCol(rd(T, ty("b", 32), C)), rd(N, ty("u", 32), C)) ... </tk>
    requires "dealloc" in M
  rule <tk> #tcDealloc(_, _) => .K ... </tk>
       <tid> T </tid> <cseq> CS </cseq> <tmres> R </tmres>
    requires tcInst(T, CS, "tcgen05.dealloc") in_keys(R)
  rule <tk> #tcDealloc(B, N) => .K ... </tk>
       <tid> T </tid> <cseq> CS </cseq>
       <tmres> R => R[tcInst(T, CS, "tcgen05.dealloc") <- 0] </tmres>
       <tmalloc> Al => Al[B <- undef] </tmalloc>
    requires notBool tcInst(T, CS, "tcgen05.dealloc") in_keys(R) andBool Al[B] ==K N
  rule <tk> #tcDealloc(B, N) => #fault("tcgen05.dealloc: not an allocation of that size", B) ... </tk> [owise]

  // ---------------------------------------------------------------- ld / st
  syntax Int ::= tcNum(Set) [function, total]
  rule tcNum(M) => pick2(M)
  syntax Int ::= pick2(Set) [function, total]
  rule pick2(M) => 128 requires "x128" in M
  rule pick2(M) => 64 requires "x64" in M
  rule pick2(M) => 32 requires "x32" in M
  rule pick2(M) => 16 requires "x16" in M
  rule pick2(M) => 8 requires "x8" in M
  rule pick2(M) => 4 requires "x4" in M
  rule pick2(M) => 2 requires "x2" in M
  rule pick2(_) => 1 [owise]
  // registers per repeat: .16x128b -> 2, .16x256b -> 4, others 1
  syntax Int ::= tcPer(Set) [function, total]
  rule tcPer(M) => 4 requires "16x256b" in M
  rule tcPer(M) => 2 requires "16x128b" in M
  rule tcPer(_) => 1 [owise]
  // (lane, column) of register i of thread t
  syntax Int ::= tcCell(Set, Int, Int, Int, Int) [function, total]          // shape mods, taddr, thread-in-warp, i, half-split offset
  rule tcCell(M, A, T, I, H) => tcKey(tcLane(A) +Int T, tcCol(A) +Int I) requires "32x32b" in M
  rule tcCell(M, A, T, I, H) => tcKey(tcLane(A) +Int T %Int 16, tcCol(A) +Int 2 *Int I +Int T /Int 16) requires "16x64b" in M
  rule tcCell(M, A, T, I, H) => tcKey(tcLane(A) +Int T %Int 16, tcCol(A) +Int 4 *Int (I /Int 2) +Int 2 *Int (T /Int 16) +Int I %Int 2) requires "16x128b" in M
  rule tcCell(M, A, T, I, H) => tcKey(tcLane(A) +Int T %Int 16, tcCol(A) +Int 8 *Int (I /Int 4) +Int 4 *Int (T /Int 16) +Int I %Int 4) requires "16x256b" in M
  rule tcCell(M, A, T, I, H) => tcKey(tcLane(A) +Int T %Int 16, tcCol(A) +Int I +Int (T /Int 16) *Int H) requires "16x32bx2" in M
  rule tcCell(_, _, _, _, _) => 0 -Int 1 [owise]
  syntax Int ::= lo16(Int) [function, total] | hi16(Int) [function, total]
  rule lo16(V) => V &Int 65535
  rule hi16(V) => (V >>Int 16) &Int 65535
  syntax Int ::= tcWord(Map, Int) [function, total]
  rule tcWord(TM, K) => {TM[K] orDefault 0}:>Int

  syntax List ::= tcLoad(Map, Set, Int, Int, Int, Int, Int, Bool) [function, total]    // tmem, mods, taddr, thread, split, i, count, pack
  rule tcLoad(TM, M, A, T, H, I, N, false) => ListItem(tcWord(TM, tcCell(M, A, T, I, H))) tcLoad(TM, M, A, T, H, I +Int 1, N, false) requires I <Int N
  rule tcLoad(TM, M, A, T, H, I, N, true) =>
       ListItem((tcWord(TM, tcCell(M, A, T, 2 *Int I, H)) &Int 65535) |Int ((tcWord(TM, tcCell(M, A, T, 2 *Int I +Int 1, H)) &Int 65535) <<Int 16))
       tcLoad(TM, M, A, T, H, I +Int 1, N, true) requires I <Int N
  rule tcLoad(_, _, _, _, _, _, _, _) => .List [owise]
  syntax Map ::= tcStore(Map, Set, Int, Int, Int, Int, List, Bool) [function, total]   // tmem, mods, taddr, thread, split, i, values, unpack
  rule tcStore(TM, M, A, T, H, I, ListItem(V:Int) Vs, false) => tcStore(TM[tcCell(M, A, T, I, H) <- V], M, A, T, H, I +Int 1, Vs, false)
  rule tcStore(TM, M, A, T, H, I, ListItem(V:Int) Vs, true) =>
       tcStore(TM[tcCell(M, A, T, 2 *Int I, H) <- lo16(V)][tcCell(M, A, T, 2 *Int I +Int 1, H) <- hi16(V)], M, A, T, H, I +Int 1, Vs, true)
  rule tcStore(TM, _, _, _, _, _, _, _) => TM [owise]

  rule <tk> exec(opc("tcgen05", M, _), ops(Dv, [ A ], .Operands), C) =>
            #wmWr(vecOps(Dv), tcLoad(TM, M, tcAddr(A, C), TID %Int 32, 0, 0, size(vecOps(Dv)), "pack::16b" in M)) ... </tk>
       <tid> TID </tid> <tmem> TM </tmem>
    requires "ld" in M
  rule <tk> exec(opc("tcgen05", M, _), ops(Dv, [ A ], Off, .Operands), C) =>
            #wmWr(vecOps(Dv), tcLoad(TM, M, tcAddr(A, C), TID %Int 32, rd(Off, ty("u", 32), C), 0, size(vecOps(Dv)), "pack::16b" in M)) ... </tk>
       <tid> TID </tid> <tmem> TM </tmem>
    requires "ld" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ A ], Vs, .Operands), C) => .K ... </tk>
       <tid> TID </tid>
       <tmem> TM => tcStore(TM, M, tcAddr(A, C), TID %Int 32, 0, 0, vecRegs(Vs, ty("b", 32), C), "unpack::16b" in M) </tmem>
    requires "st" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ A ], Off, Vs, .Operands), C) => .K ... </tk>
       <tid> TID </tid>
       <tmem> TM => tcStore(TM, M, tcAddr(A, C), TID %Int 32, rd(Off, ty("u", 32), C), 0, vecRegs(Vs, ty("b", 32), C), "unpack::16b" in M) </tmem>
    requires "st" in M

  // ---------------------------------------------------------------- tcgen05.mma
  // smem descriptor (sm_100) -> the equivalent sm_90 descriptor read by wgElem
  syntax Int ::= tcDesc(Int) [function, total]
  rule tcDesc(D) => (D &Int 16383) |Int (((D >>Int 16) &Int 16383) <<Int 16) |Int (((D >>Int 32) &Int 16383) <<Int 32) |Int (tcSwz((D >>Int 61) &Int 7) <<Int 62)
  syntax Int ::= tcSwz(Int) [function, total]
  rule tcSwz(2) => 1   rule tcSwz(4) => 2   rule tcSwz(6) => 3   rule tcSwz(_) => 0 [owise]

  // element types from the instruction descriptor
  syntax Ty ::= tcDT(Set, Int) [function, total] | tcAT(Set, Int) [function, total]
  rule tcDT(M, I) => ty("f", 32) requires "kind::mxf8f6f4" in M orBool "kind::mxf4" in M orBool "kind::mxf4nvf4" in M
  rule tcDT(M, I) => ty("s", 32) requires "kind::i8" in M
  rule tcDT(M, I) => #if ((I >>Int 4) &Int 3) ==Int 0 #then ty("f", 16) #else ty("f", 32) #fi [owise]
  rule tcAT(M, I) => ty("e2m1p", 4) requires "kind::mxf4" in M orBool "kind::mxf4nvf4" in M         // two 4-bit elements per byte
  rule tcAT(M, I) => ty("tf", 32) requires "kind::tf32" in M
  rule tcAT(M, I) => #if ((I >>Int 7) &Int 7) ==Int 0 #then ty("u", 8) #else ty("s", 8) #fi requires "kind::i8" in M
  rule tcAT(M, I) => tcF8((I >>Int 7) &Int 7) requires "kind::f8f6f4" in M orBool "kind::mxf8f6f4" in M
  rule tcAT(M, I) => #if ((I >>Int 7) &Int 7) ==Int 0 #then ty("f", 16) #else ty("bf", 16) #fi [owise]
  syntax Ty ::= tcF8(Int) [function, total]
  rule tcF8(0) => ty("e4m3", 8)   rule tcF8(1) => ty("e5m2", 8)   rule tcF8(3) => ty("e2m3", 8)   rule tcF8(4) => ty("e3m2", 8)
  rule tcF8(5) => ty("e2m1", 8)   rule tcF8(_) => ty("e4m3", 8) [owise]
  syntax Ty ::= tcBT(Set, Int) [function, total]
  rule tcBT(M, I) => tcAT(M, I >>Int 3)
  syntax Int ::= tcMD(Set, Int) [function, total]            // M: bits 24-28 (x16), or bits 27-28 (x128) for the block-scaled kinds
  rule tcMD(M, I) => ((I >>Int 27) &Int 3) *Int 128 requires "kind::mxf8f6f4" in M orBool "kind::mxf4" in M orBool "kind::mxf4nvf4" in M
  rule tcMD(M, I) => ((I >>Int 24) &Int 31) *Int 16 [owise]
  syntax Int ::= tcK(Set, Int) [function, total]
  rule tcK(M, I) => #if ((I >>Int 31) &Int 1) ==Int 1 #then 64 #else 32 #fi requires "kind::mxf8f6f4" in M
  rule tcK(M, I) => #if ((I >>Int 3) &Int 1) ==Int 1 #then 128 #else #if ((I >>Int 31) &Int 1) ==Int 1 #then 96 #else 64 #fi #fi
    requires "kind::mxf4" in M orBool "kind::mxf4nvf4" in M
  rule tcK(M, I) => 8 requires "kind::tf32" in M
  rule tcK(M, I) => 32 requires "kind::i8" in M
  rule tcK(M, I) => #if ((I >>Int 29) &Int 1) ==Int 1 #then 64 #else 32 #fi requires "kind::f8f6f4" in M
  rule tcK(_, _) => 16 [owise]

  // operand tiles
  syntax Map ::= tcTileA(Bool, Int, Map, Bytes, Ty, Int, Bool, Int, Int, Int, Int, Int, Map) [function, total]
  // A (M x K): descriptor-or-tmem?, value, tmem, smem, type, negate sign, K-major?, M, K, m, k, tmem columns... accumulate
  rule tcTileA(Desc, V, TM, SM, T, Sg, KM, M, K, R, Kk, _, Acc) =>
       tcTileA(Desc, V, TM, SM, T, Sg, KM, M, K, #if Kk +Int 1 ==Int K #then R +Int 1 #else R #fi, #if Kk +Int 1 ==Int K #then 0 #else Kk +Int 1 #fi, 0,
               Acc[rc(R, Kk) <- wgNeg(T, Sg,
                     #if Desc #then wgElem(SM, tcDesc(V), KM, ew(T), R, Kk)
                     #else (tcWord(TM, tcKey(R, tcCol(V) +Int (Kk *Int ew(T)) /Int 32)) >>Int ((Kk *Int ew(T)) %Int 32)) &Int mask(ew(T)) #fi)])
    requires R <Int M
  rule tcTileA(_, _, _, _, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= tcTileB(Int, Bytes, Ty, Int, Bool, Int, Int, Int, Int, Map, Int, Int, Int) [function, total]
  // ..., accumulator, column shift, zero-column mask descriptor (-1: none), rows of the M dimension
  rule tcTileB(V, SM, T, Sg, KM, N, K, Nn, Kk, Acc, Sh, Z, Md) =>
       tcTileB(V, SM, T, Sg, KM, N, K, #if Kk +Int 1 ==Int K #then Nn +Int 1 #else Nn #fi, #if Kk +Int 1 ==Int K #then 0 #else Kk +Int 1 #fi,
               Acc[rc(Kk, Nn) <- #if tcZero(Z, Md, N, Nn) #then 0 #else wgNeg(T, Sg, wgElem(SM, tcDesc(V), KM, ew(T), Nn +Int Sh, Kk)) #fi], Sh, Z, Md)
    requires Nn <Int N
  rule tcTileB(_, _, _, _, _, _, _, _, _, Acc, _, _, _) => Acc [owise]
  // zero-column mask descriptor (.ws): is output column C zeroed?  Sub-masks: 128/M of them, each covering N/(128/M) columns.
  syntax Bool ::= tcZero(Int, Int, Int, Int) [function, total]
  rule tcZero(Z, Md, N, C) => tcZBit(Z, C /Int (N /Int (128 /Int Md)), C %Int (N /Int (128 /Int Md))) requires Z >=Int 0 andBool Md >Int 0 andBool Md <=Int 128 andBool N /Int (128 /Int Md) >Int 0
  rule tcZero(_, _, _, _) => false [owise]
  syntax Bool ::= tcZBit(Int, Int, Int) [function, total]                 // descriptor, sub-mask, position
  rule tcZBit(Z, I, P) => ((Z >>Int 39) &Int 1) ==Int 1 andBool
       tcRun(((Z >>Int (32 +Int I)) &Int 1), ((Z >>Int (8 *Int I)) &Int 255), ((Z >>Int 40) &Int 255) +Int 1, ((Z >>Int 48) &Int 255) +Int 1, P)
  // periodic pattern of period On+Zr starting with a run of V0 (1 = ones of length On, 0 = zeros of length Zr), shifted by Sc columns
  syntax Bool ::= tcRun(Int, Int, Int, Int, Int) [function, total]
  rule tcRun(V0, Sc, On, Zr, P) => (V0 ==Int 1 andBool ((P +Int Sc) %Int (On +Int Zr)) <Int On) orBool (V0 ==Int 0 andBool ((P +Int Sc) %Int (On +Int Zr)) >=Int Zr)
  // D accumulator input: scaled by 2^-s
  syntax Map ::= tcTileD(Map, Ty, Int, Int, Int, Int, Int, Int, Int, Map) [function, total]
  rule tcTileD(TM, T, Base, M, N, R, Cn, S, Z, Acc) =>
       tcTileD(TM, T, Base, M, N, #if Cn +Int 1 ==Int N #then R +Int 1 #else R #fi, #if Cn +Int 1 ==Int N #then 0 #else Cn +Int 1 #fi, S, Z,
               Acc[rc(R, Cn) <- tcScale(T, tcWord(TM, tcKey(R, Base +Int Cn)) &Int mask(ew(T)), S)])
    requires R <Int M
  rule tcTileD(_, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= tcScale(Ty, Int, Int) [function, total]
  rule tcScale(T, B, S) => fpFin(.Set, fw(T), elemF(T, B) /Float Int2Float(1 <<Int S, 300, WEXP), 0) requires S >Int 0 andBool isFloatElem(T)
  rule tcScale(_, B, _) => B [owise]

  // write the result back (lane r, column Base + c), honouring the disable-output-lane mask
  syntax Map ::= tcPutD(Map, Map, List, Int, Int, Int, Int, Int, Ty) [function, total]
  rule tcPutD(TM, Res, ListItem(K:Int) Ks, Base, Mask, M, N, _, T) =>
       tcPutD(#if ((Mask >>Int (K /Int 1024)) &Int 1) ==Int 1 #then TM
              #else TM[tcKey(K /Int 1024, Base +Int K %Int 1024) <- {Res[K] orDefault 0}:>Int] #fi, Res, Ks, Base, Mask, M, N, 0, T)
  rule tcPutD(TM, _, _, _, _, _, _, _, _) => TM [owise]

  syntax Map ::= tcMma(Set, Map, Bytes, Bool, Int, Int, Int, Int, Bool, Int, Int, Int, Int) [function, total]
  // mods, tmem, smem, a-is-desc, a value, b desc, idesc, d-taddr, enable, scale, mask -> new tmem
  rule tcMma(M, TM, SM, AD, AV, BV, ID, DA, En, S, Mask, Meta, Z) =>
       tcPutD(TM,
              resRows("m" +String Int2String(tcMD(M, ID)) +String "n" +String Int2String(((ID >>Int 17) &Int 63) *Int 8) +String "k" +String Int2String(tcKS(M, ID, Meta)),
                      tcDT(M, ID), tcAT(M, ID), tcBT(M, ID), tcDT(M, ID),
                      tcAMat(M, TM, SM, AD, AV, ID, Meta),
                      tcTileB(BV, SM, tcBT(M, ID), #if ((ID >>Int 14) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 16) &Int 1) ==Int 0,
                              ((ID >>Int 17) &Int 63) *Int 8, tcKS(M, ID, Meta), 0, 0, .Map, tcShiftZ(Z), Z, tcMD(M, ID)),
                      #if En #then tcTileD(TM, tcDT(M, ID), tcCol(DA), tcMD(M, ID), ((ID >>Int 17) &Int 63) *Int 8, 0, 0, S, 0, .Map) #else .Map #fi,
                      0, 0, #if ((ID >>Int 3) &Int 1) ==Int 1 #then SetItem("satfinite") #else .Set #fi, .Map),
              rowKeys(tcMD(M, ID), ((ID >>Int 17) &Int 63) *Int 8, 0, 0, .List),
              tcCol(DA), Mask, 0, 0, 0, tcDT(M, ID))
  // sparse A (tcgen05.mma.sp): the packed M x K/2 matrix is read like a dense one and expanded with the metadata
  // words in tensor memory (assumed layout, see docs/prior-work.md): row r's metadata sits in lane r, 4 bits per chunk;
  // 16-bit and tf32 kinds use one word, selected by the descriptor's sparsity selector (column base + selector);
  // 8-bit kinds use two consecutive words (chunks 0-7, 8-15).
  syntax Int ::= tcKS(Set, Int, Int) [function, total]
  rule tcKS(M, ID, Meta) => 2 *Int tcK(M, ID) requires Meta >=Int 0
  rule tcKS(M, ID, Meta) => tcK(M, ID) requires Meta <Int 0
  syntax Map ::= tcAMat(Set, Map, Bytes, Bool, Int, Int, Int) [function, total]
  rule tcAMat(M, TM, SM, AD, AV, ID, Meta) => tcAshift(tcAMat0(M, TM, SM, AD, AV, ID, Meta), "ashift" in M)
  // .ashift: the rows of A move down one row (row 0 is kept, the last row is dropped)
  syntax Map ::= tcAshift(Map, Bool) [function, total]
  rule tcAshift(A, false) => A
  rule tcAshift(A, true) => tcAshiftK(A, keys_list(A), .Map)
  syntax Map ::= tcAshiftK(Map, List, Map) [function, total]
  rule tcAshiftK(A, ListItem(K:Int) Ks, Acc) => tcAshiftK(A, Ks, Acc[K <- {A[#if K >=Int 1024 #then K -Int 1024 #else K #fi] orDefault 0}:>Int])
  rule tcAshiftK(_, _, Acc) => Acc [owise]
  syntax Map ::= tcAMat0(Set, Map, Bytes, Bool, Int, Int, Int) [function, total]
  rule tcAMat0(M, TM, SM, AD, AV, ID, Meta) =>
       tcTileA(AD, AV, TM, SM, tcAT(M, ID), #if ((ID >>Int 13) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 15) &Int 1) ==Int 0,
               tcMD(M, ID), tcK(M, ID), 0, 0, 0, .Map)
    requires Meta <Int 0
  rule tcAMat0(M, TM, SM, AD, AV, ID, Meta) =>
       tcSpExp(tcTileA(AD, AV, TM, SM, tcAT(M, ID), #if ((ID >>Int 13) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 15) &Int 1) ==Int 0,
                       tcMD(M, ID), tcK(M, ID), 0, 0, 0, .Map),
               TM, tcAT(M, ID), tcMD(M, ID), tcK(M, ID), Meta, ID &Int 3, 0, 0, .Map)
    requires Meta >=Int 0
  syntax Map ::= tcSpExp(Map, Map, Ty, Int, Int, Int, Int, Int, Int, Map) [function, total]   // packed, tmem, type, rows, packed cols, meta column, selector, r, kp, acc
  rule tcSpExp(Pk, TM, T, M, KP, MC, Sel, R, K, Acc) =>
       tcSpExp(Pk, TM, T, M, KP, MC, Sel, #if K +Int 1 ==Int KP #then R +Int 1 #else R #fi, #if K +Int 1 ==Int KP #then 0 #else K +Int 1 #fi,
               Acc[rc(R, spCw(T) *Int (K /Int spNz(T)) +Int spPos(T, tcNib(TM, T, R, MC, Sel, K /Int spNz(T), KP), K %Int spNz(T)))
                   <- {Pk[rc(R, K)] orDefault 0}:>Int])
    requires R <Int M
  rule tcSpExp(_, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= tcNib(Map, Ty, Int, Int, Int, Int, Int) [function, total]                      // tmem, type, row, meta col, selector, chunk, packed cols
  rule tcNib(TM, T, R, MC, Sel, C, KP) => (tcWord(TM, tcKey(R, MC +Int Sel)) >>Int (4 *Int C)) &Int 15 requires 2 *Int KP /Int spCw(T) <=Int 8
  rule tcNib(TM, T, R, MC, Sel, C, KP) => (tcWord(TM, tcKey(R, MC +Int C /Int 8)) >>Int (4 *Int (C %Int 8))) &Int 15 requires 2 *Int KP /Int spCw(T) >Int 8

  syntax Int ::= tcShiftZ(Int) [function, total]
  rule tcShiftZ(Z) => (Z >>Int 56) &Int 63 requires Z >=Int 0
  rule tcShiftZ(_) => 0 [owise]

  // the result keys (r, c) in order
  syntax List ::= rowKeys(Int, Int, Int, Int, List) [function, total]
  rule rowKeys(M, N, R, C, Acc) => rowKeys(M, N, #if C +Int 1 ==Int N #then R +Int 1 #else R #fi, #if C +Int 1 ==Int N #then 0 #else C +Int 1 #fi, Acc ListItem(rc(R, C))) requires R <Int M
  rule rowKeys(_, _, _, _, Acc) => Acc [owise]

  // mask vector -> 128-bit lane mask
  syntax Int ::= tcMask(List, Int) [function, total]
  rule tcMask(ListItem(V:Int) Vs, I) => (V <<Int (32 *Int I)) |Int tcMask(Vs, I +Int 1)
  rule tcMask(_, _) => 0 [owise]

  // tcgen05.mma.cta_group::1.kind [d], a-desc | [a-tmem], b-desc, idesc, {mask}, enable[, scale]
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], [ A ], Bd, ID, Rest:Operands), C) =>
            #tcMma(M, false, tcAddr(A, C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), 0 -Int 1, opsList(Rest), C) ... </tk>
    requires "mma" in M andBool notBool "sp" in M andBool notBool "block_scale" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], Ad, Bd, ID, Rest:Operands), C) =>
            #tcMma(M, true, rd(Ad, ty("b", 64), C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), 0 -Int 1, opsList(Rest), C) ... </tk>
    requires "mma" in M andBool notBool "sp" in M andBool notBool "block_scale" in M [priority(60)]
  // sparse: [d], a-desc | [a-tmem], b-desc, [sp-meta-tmem], idesc, ...
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], [ A ], Bd, [ MT ], ID, Rest:Operands), C) =>
            #tcMma(M, false, tcAddr(A, C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), tcCol(tcAddr(MT, C)), opsList(Rest), C) ... </tk>
    requires "mma" in M andBool "sp" in M andBool notBool "block_scale" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], Ad, Bd, [ MT ], ID, Rest:Operands), C) =>
            #tcMma(M, true, rd(Ad, ty("b", 64), C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), tcCol(tcAddr(MT, C)), opsList(Rest), C) ... </tk>
    requires "mma" in M andBool "sp" in M andBool notBool "block_scale" in M [priority(60)]
  syntax KItem ::= #tcMma(Set, Bool, Int, Int, Int, Int, Int, List, Ctx)
  rule <tk> #tcMma(M, AD, AV, BV, ID, DA, Meta, Rest, C) => .K ... </tk>
       <smem> SM </smem>
       <tmem> TM => tcMma(M, TM, SM, AD, AV, BV, ID, DA, tcEn(Rest, C), #if "ws" in M #then 0 #else tcSc(Rest, C) #fi, tcMk(Rest, C), Meta, tcZm(M, Rest, C)) </tmem>
  // optional vector mask first, then the enable predicate, then the optional scale
  syntax List ::= tcOps(List) [function, total]
  rule tcOps(ListItem(V:Operand) Vs) => #if isVec(V) #then Vs #else ListItem(V) Vs #fi
  rule tcOps(.List) => .List
  syntax Int ::= tcMk(List, Ctx) [function, total]
  rule tcMk(ListItem(V:Operand) _, C) => tcMask(vecRegs(V, ty("b", 32), C), 0) requires isVec(V)
  rule tcMk(_, _) => 0 [owise]
  syntax Bool ::= tcEn(List, Ctx) [function, total]
  rule tcEn(L, C) => rd({tcOps(L)[0]}:>Operand, ty("u", 32), C) =/=Int 0 requires size(tcOps(L)) >Int 0
  rule tcEn(_, _) => true [owise]
  syntax Int ::= tcZm(Set, List, Ctx) [function, total]
  rule tcZm(M, L, C) => rd({tcOps(L)[1]}:>Operand, ty("b", 64), C) requires "ws" in M andBool size(tcOps(L)) >Int 1
  rule tcZm(_, _, _) => 0 -Int 1 [owise]
  syntax Int ::= tcSc(List, Ctx) [function, total]
  rule tcSc(L, C) => rd({tcOps(L)[1]}:>Operand, ty("u", 32), C) requires size(tcOps(L)) >Int 1
  rule tcSc(_, _) => 0 [owise]

  // ---------------------------------------------------------------- tcgen05.commit / cp / shift
  // Asynchronous operations complete at issue, so commit is an arrive (count 1) on the mbarrier(s) right away.
  //   tcgen05.commit.cta_group::N.mbarrier::arrive::one{.shared::cluster}{.multicast::cluster{::16b,::32b}}.b64 [mbar]{, ctaMask}
  rule <tk> exec(opc("tcgen05", M, _), ops([ A ], .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => mbArriveKeys(MB, ListItem(mbKey(ea(A, spaceOf(M), C), CT, GD, CD))) </mbars>
    requires "commit" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ A ], Mk, .Operands), C) => .K ... </tk>
       <ctaid> CT </ctaid> <gridDim> GD </gridDim> <clusterDim> CD </clusterDim>
       <mbars> MB => mbArriveKeys(MB, tcMcKeys(locOff(ea(A, spaceOf(M), C)) &Int 16777215, rd(Mk, ty("u", 32), C), 0, CT, GD, CD, .List)) </mbars>
    requires "commit" in M andBool "multicast" in M
  syntax Map ::= mbArriveKeys(Map, List) [function, total]
  rule mbArriveKeys(MB, ListItem(K:Int) Ks) => mbArriveKeys(#if K in_keys(MB) #then MB[K <- mbArrive(mbGet(MB, K), 1)] #else MB #fi, Ks)
  rule mbArriveKeys(MB, _) => MB [owise]
  syntax List ::= tcMcKeys(Int, Int, Int, Int, Dim3, Dim3, List) [function, total]          // offset, mask, bit, own CTA, grid, cluster, acc
  rule tcMcKeys(Off, Mk, B, CT, GD, CD, Acc) =>
       tcMcKeys(Off, Mk, B +Int 1, CT, GD, CD,
                #if (Mk >>Int B) &Int 1 ==Int 1 #then Acc ListItem(mbKey(loc("cluster", ((B +Int 1) <<Int 24) |Int Off), CT, GD, CD)) #else Acc #fi)
    requires B <Int 32
  rule tcMcKeys(_, _, _, _, _, _, Acc) => Acc [owise]

  //   tcgen05.cp.cta_group::N.shape{.multicast}  [taddr], s-desc    shared memory -> tensor memory
  // shape lane x size: `lane` rows of size/32 words; row i goes to tensor-memory lane (taddr.lane + i).  The source is read like
  // a K-major tile of 32-bit elements through the (sm_100) descriptor.  Not modelled: the .b8x16.b6x16_p32 / .b4x16_p64 decompression.
  syntax Int ::= tcCpRows(Set) [function, total] | tcCpWords(Set) [function, total]
  rule tcCpRows(M) => 128 requires "128x256b" in M orBool "128x128b" in M
  rule tcCpRows(M) => 4 requires "4x256b" in M
  rule tcCpRows(M) => 64 requires "64x128b" in M
  rule tcCpRows(M) => 32 requires "32x128b" in M
  rule tcCpRows(M) => 0 requires notBool ("128x256b" in M orBool "128x128b" in M orBool "4x256b" in M orBool "64x128b" in M orBool "32x128b" in M)
  rule tcCpWords(M) => 8 requires "128x256b" in M orBool "4x256b" in M
  rule tcCpWords(M) => 4 [owise]
  // destination lanes (relative to the base lane) of source row i: plain copy, or warp multicast
  syntax List ::= tcCpDst(Set, Int) [function, total]
  rule tcCpDst(M, I) => ListItem(I) requires notBool ("warpx2" in M orBool "warpx4" in M)
  rule tcCpDst(M, I) => ListItem(I) ListItem(I +Int 32) ListItem(I +Int 64) ListItem(I +Int 96) requires "warpx4" in M
  rule tcCpDst(M, I) => ListItem(I %Int 32 +Int 32 *Int (2 *Int (I /Int 32))) ListItem(I %Int 32 +Int 32 *Int (2 *Int (I /Int 32) +Int 1))
    requires "warpx2::02_13" in M
  rule tcCpDst(M, I) => ListItem(I %Int 32 +Int 32 *Int (I /Int 32)) ListItem(I %Int 32 +Int 32 *Int (I /Int 32 +Int 2))
    requires "warpx2::01_23" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ T ], Dd, .Operands), C) => .K ... </tk>
       <smem> SM </smem> <tmem> TM => tcCp(TM, SM, M, tcLane(tcAddr(T, C)), tcCol(tcAddr(T, C)), tcDesc(rd(Dd, ty("b", 64), C)), 0) </tmem>
    requires "cp" in M
  syntax Map ::= tcCp(Map, Bytes, Set, Int, Int, Int, Int) [function, total]            // tmem, smem, mods, lane, column, descriptor, source row
  rule tcCp(TM, SM, M, L, Cl, D, I) => tcCp(tcCpPut(TM, tcCpDst(M, I), L, Cl, tcCpRow(SM, D, I, 0, tcCpWords(M), .List)), SM, M, L, Cl, D, I +Int 1)
    requires I <Int tcCpRows(M)
  rule tcCp(TM, _, _, _, _, _, _) => TM [owise]
  syntax List ::= tcCpRow(Bytes, Int, Int, Int, Int, List) [function, total]
  rule tcCpRow(SM, D, I, W, N, Acc) => tcCpRow(SM, D, I, W +Int 1, N, Acc ListItem(wgElem(SM, D, true, 32, I, W))) requires W <Int N
  rule tcCpRow(_, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= tcCpPut(Map, List, Int, Int, List) [function, total]
  rule tcCpPut(TM, ListItem(R:Int) Rs, L, Cl, Ws) => tcCpPut(tcCpWr(TM, L +Int R, Cl, Ws), Rs, L, Cl, Ws)
  rule tcCpPut(TM, _, _, _, _) => TM [owise]
  syntax Map ::= tcCpWr(Map, Int, Int, List) [function, total]
  rule tcCpWr(TM, Ln, Cl, ListItem(V:Int) Ws) => tcCpWr(TM[tcKey(Ln, Cl) <- V], Ln, Cl +Int 1, Ws)
  rule tcCpWr(TM, _, _, _) => TM [owise]

  //   tcgen05.shift.cta_group::N.down [taddr]: the 32-byte (8-word) rows of the 32-lane block move down one row; the last row is dropped
  rule <tk> exec(opc("tcgen05", M, _), ops([ T ], .Operands), C) => .K ... </tk>
       <tmem> TM => tcShift(TM, tcLane(tcAddr(T, C)), tcCol(tcAddr(T, C)), 30) </tmem>
    requires "shift" in M
  syntax Map ::= tcShift(Map, Int, Int, Int) [function, total]                           // tmem, base lane, column, row (descending)
  rule tcShift(TM, L, Cl, R) => tcShift(tcCpWr(TM, L +Int R +Int 1, Cl, tcRowWords(TM, L +Int R, Cl, 0, .List)), L, Cl, R -Int 1) requires R >=Int 0
  rule tcShift(TM, _, _, _) => TM [owise]
  syntax List ::= tcRowWords(Map, Int, Int, Int, List) [function, total]
  rule tcRowWords(TM, Ln, Cl, W, Acc) => tcRowWords(TM, Ln, Cl, W +Int 1, Acc ListItem(tcWord(TM, tcKey(Ln, Cl +Int W)))) requires W <Int 8
  rule tcRowWords(_, _, _, _, Acc) => Acc [owise]

  // ---------------------------------------------------------------- block-scaled MMA  (.kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4)
  //   D = sum over K-blocks of scaleA[m][blk] * scaleB[blk][n] * (A row block . B column block) + D
  // Assumed scale placement (the ISA gives it as figures): scale A for row r is byte (SFA_ID + blk) of the tensor-memory word at lane r,
  // column [scale-A-tmem]; scale B for column n is byte (SFB_ID + blk) of the word at lane n mod 32, column [scale-B-tmem] + n div 32.
  // Scale types: UE8M0 (2^(x-127)), UE4M3, UE5M3; elements of .kind::mxf4/.mxf4nvf4 are 4-bit values packed two per byte.
  syntax Int ::= tcBlk(Set, Int, Int) [function, total]                      // block size from the qualifiers
  rule tcBlk(M, ID, K) => K requires "kind::mxf8f6f4" in M
  rule tcBlk(M, ID, K) => 16 requires notBool "kind::mxf8f6f4" in M andBool "block16" in M
  rule tcBlk(M, ID, K) => K /Int 4 requires notBool "kind::mxf8f6f4" in M andBool notBool "block16" in M andBool "scale_vec::4X" in M
  rule tcBlk(M, ID, K) => K /Int 2 requires notBool "kind::mxf8f6f4" in M andBool notBool "block16" in M andBool notBool "scale_vec::4X" in M andBool "scale_vec::2X" in M
  rule tcBlk(M, ID, K) => 32 requires notBool "kind::mxf8f6f4" in M andBool notBool "block16" in M andBool notBool "scale_vec::4X" in M andBool notBool "scale_vec::2X" in M
  syntax Float ::= tcScaleF(Int, Int) [function, total]                       // scale type code, byte
  rule tcScaleF(1, X) => scaleP2(Int2Float(1, 64, WEXP), X -Int 127)
  rule tcScaleF(0, X) => tcUE(X, 3, 7)
  rule tcScaleF(2, X) => tcUE(X, 3, 15)
  rule tcScaleF(_, X) => tcUE(X, 3, 7) [owise]
  syntax Float ::= tcUE(Int, Int, Int) [function, total]                      // unsigned float with Mn mantissa bits and bias Bi
  rule tcUE(X, Mn, Bi) =>
       #if (X >>Int Mn) ==Int 0 #then scaleP2(Int2Float(X &Int mask(Mn), 64, WEXP), 1 -Int Bi -Int Mn)
       #else scaleP2(Int2Float((X &Int mask(Mn)) +Int (1 <<Int Mn), 64, WEXP), (X >>Int Mn) -Int Bi -Int Mn) #fi
  syntax Int ::= tcScaleTy(Set, Int) [function, total]
  rule tcScaleTy(M, ID) => (ID >>Int 23) &Int 3 requires "kind::mxf4nvf4" in M
  rule tcScaleTy(M, ID) => 1 requires notBool "kind::mxf4nvf4" in M
  syntax Map ::= tcScalesA(Map, Int, Int, Int, Int, Int, Int, Map) [function, total]           // tmem, column, SFA_ID, rows, blocks, r, blk, acc
  rule tcScalesA(TM, Cl, Id, M, NB, R, J, Acc) =>
       tcScalesA(TM, Cl, Id, M, NB, #if J +Int 1 ==Int NB #then R +Int 1 #else R #fi, #if J +Int 1 ==Int NB #then 0 #else J +Int 1 #fi,
                 Acc[rc(R, J) <- (tcWord(TM, tcKey(R, Cl)) >>Int (8 *Int (Id +Int J))) &Int 255])
    requires R <Int M
  rule tcScalesA(_, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= tcScalesB(Map, Int, Int, Int, Int, Int, Int, Map) [function, total]           // tmem, column, SFB_ID, N, blocks, n, blk, acc
  rule tcScalesB(TM, Cl, Id, N, NB, Nn, J, Acc) =>
       tcScalesB(TM, Cl, Id, N, NB, #if J +Int 1 ==Int NB #then Nn +Int 1 #else Nn #fi, #if J +Int 1 ==Int NB #then 0 #else J +Int 1 #fi,
                 Acc[rc(J, Nn) <- (tcWord(TM, tcKey(Nn %Int 32, Cl +Int Nn /Int 32)) >>Int (8 *Int (Id +Int J))) &Int 255])
    requires Nn <Int N
  rule tcScalesB(_, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Float ::= dotFS(Ty, Ty, Map, Map, Map, Map, Int, Int, Int, Int, Int, Int, Float) [function, total]
  // atype, btype, A, B, scale A, scale B, scale type, r, c, k, K, block size, acc
  rule dotFS(AT, BT, A, B, SA, SB, ST, R, Cn, I, K, Bs, Acc) =>
       dotFS(AT, BT, A, B, SA, SB, ST, R, Cn, I +Int 1, K, Bs,
             Acc +Float elemF(AT, {A[rc(R, I)] orDefault 0}:>Int) *Float elemF(BT, {B[rc(I, Cn)] orDefault 0}:>Int)
                 *Float tcScaleF(ST, {SA[rc(R, I /Int Bs)] orDefault 0}:>Int) *Float tcScaleF(ST, {SB[rc(I /Int Bs, Cn)] orDefault 0}:>Int))
    requires I <Int K
  rule dotFS(_, _, _, _, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= bsRows(Ty, Ty, Map, Map, Map, Map, Map, Int, Int, Int, Int, Int, Int, Int, Map) [function, total]
  // atype, btype, A, B, D in, scale A, scale B, scale type, M, N, K, block size, r, c, acc
  rule bsRows(AT, BT, A, B, C, SA, SB, ST, M, N, K, Bs, R, Cn, Acc) =>
       bsRows(AT, BT, A, B, C, SA, SB, ST, M, N, K, Bs, #if Cn +Int 1 ==Int N #then R +Int 1 #else R #fi, #if Cn +Int 1 ==Int N #then 0 #else Cn +Int 1 #fi,
              Acc[rc(R, Cn) <- fpFin(.Set, fw(ty("f", 32)),
                                     dotFS(AT, BT, A, B, SA, SB, ST, R, Cn, 0, K, Bs, elemF(ty("f", 32), {C[rc(R, Cn)] orDefault 0}:>Int)), 0)])
    requires R <Int M
  rule bsRows(_, _, _, _, _, _, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Map ::= tcMmaBS(Set, Map, Bytes, Bool, Int, Int, Int, Int, Bool, Int, Int, Int) [function, total]
  // mods, tmem, smem, a-is-desc, a value, b desc, idesc, d-taddr, enable, scale-A column, scale-B column, meta column (-1: dense)
  rule tcMmaBS(M, TM, SM, AD, AV, BV, ID, DA, En, SAC, SBC, Meta) =>
       tcPutD(TM,
              bsRows(tcAT(M, ID), tcBT(M, ID), tcAMat(M, TM, SM, AD, AV, ID, Meta),
                     tcTileB(BV, SM, tcBT(M, ID), #if ((ID >>Int 14) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 16) &Int 1) ==Int 0,
                             ((ID >>Int 17) &Int 63) *Int 8, tcKS(M, ID, Meta), 0, 0, .Map, 0, 0 -Int 1, tcMD(M, ID)),
                     #if En #then tcTileD(TM, ty("f", 32), tcCol(DA), tcMD(M, ID), ((ID >>Int 17) &Int 63) *Int 8, 0, 0, 0, 0, .Map) #else .Map #fi,
                     tcScalesA(TM, SAC, (ID >>Int 29) &Int 3, tcMD(M, ID), tcKS(M, ID, Meta) /Int tcBlk(M, ID, tcKS(M, ID, Meta)), 0, 0, .Map),
                     tcScalesB(TM, SBC, (ID >>Int 4) &Int 3, ((ID >>Int 17) &Int 63) *Int 8, tcKS(M, ID, Meta) /Int tcBlk(M, ID, tcKS(M, ID, Meta)), 0, 0, .Map),
                     tcScaleTy(M, ID), tcMD(M, ID), ((ID >>Int 17) &Int 63) *Int 8, tcKS(M, ID, Meta), tcBlk(M, ID, tcKS(M, ID, Meta)), 0, 0, .Map),
              rowKeys(tcMD(M, ID), ((ID >>Int 17) &Int 63) *Int 8, 0, 0, .List), tcCol(DA), 0, 0, 0, 0, ty("f", 32))
  // tcgen05.mma[.sp].cta_group::1.kind.block_scale{.scale_vectorsize} [d], a-desc | [a], b-desc, {[sp-meta],} idesc, [scale-A], [scale-B], enable
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], [ A ], Bd, ID, [ SA ], [ SB ], En, .Operands), C) => .K ... </tk>
       <smem> SM </smem>
       <tmem> TM => tcMmaBS(M, TM, SM, false, tcAddr(A, C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C),
                            rd(En, ty("u", 32), C) =/=Int 0, tcCol(tcAddr(SA, C)), tcCol(tcAddr(SB, C)), 0 -Int 1) </tmem>
    requires "mma" in M andBool "block_scale" in M andBool notBool "sp" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], Ad, Bd, ID, [ SA ], [ SB ], En, .Operands), C) => .K ... </tk>
       <smem> SM </smem>
       <tmem> TM => tcMmaBS(M, TM, SM, true, rd(Ad, ty("b", 64), C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C),
                            rd(En, ty("u", 32), C) =/=Int 0, tcCol(tcAddr(SA, C)), tcCol(tcAddr(SB, C)), 0 -Int 1) </tmem>
    requires "mma" in M andBool "block_scale" in M andBool notBool "sp" in M [priority(60)]
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], [ A ], Bd, [ MT ], ID, [ SA ], [ SB ], En, .Operands), C) => .K ... </tk>
       <smem> SM </smem>
       <tmem> TM => tcMmaBS(M, TM, SM, false, tcAddr(A, C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C),
                            rd(En, ty("u", 32), C) =/=Int 0, tcCol(tcAddr(SA, C)), tcCol(tcAddr(SB, C)), tcCol(tcAddr(MT, C))) </tmem>
    requires "mma" in M andBool "block_scale" in M andBool "sp" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], Ad, Bd, [ MT ], ID, [ SA ], [ SB ], En, .Operands), C) => .K ... </tk>
       <smem> SM </smem>
       <tmem> TM => tcMmaBS(M, TM, SM, true, rd(Ad, ty("b", 64), C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C),
                            rd(En, ty("u", 32), C) =/=Int 0, tcCol(tcAddr(SA, C)), tcCol(tcAddr(SB, C)), tcCol(tcAddr(MT, C))) </tmem>
    requires "mma" in M andBool "block_scale" in M andBool "sp" in M [priority(60)]
endmodule

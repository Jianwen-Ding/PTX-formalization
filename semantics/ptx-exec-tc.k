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

module PTX-EXEC-TC
  imports PTX-EXEC-WG
  imports PTX-EXEC-LDM

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
  rule tcDT(M, I) => ty("s", 32) requires "kind::i8" in M
  rule tcDT(M, I) => #if ((I >>Int 4) &Int 3) ==Int 0 #then ty("f", 16) #else ty("f", 32) #fi [owise]
  rule tcAT(M, I) => ty("tf", 32) requires "kind::tf32" in M
  rule tcAT(M, I) => #if ((I >>Int 7) &Int 7) ==Int 0 #then ty("u", 8) #else ty("s", 8) #fi requires "kind::i8" in M
  rule tcAT(M, I) => #if ((I >>Int 7) &Int 7) ==Int 0 #then ty("e4m3", 8) #else ty("e5m2", 8) #fi requires "kind::f8f6f4" in M
  rule tcAT(M, I) => #if ((I >>Int 7) &Int 7) ==Int 0 #then ty("f", 16) #else ty("bf", 16) #fi [owise]
  syntax Ty ::= tcBT(Set, Int) [function, total]
  rule tcBT(M, I) => tcAT(M, I >>Int 3)
  syntax Int ::= tcK(Set, Int) [function, total]
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
  syntax Map ::= tcTileB(Int, Bytes, Ty, Int, Bool, Int, Int, Int, Int, Map) [function, total]
  rule tcTileB(V, SM, T, Sg, KM, N, K, Nn, Kk, Acc) =>
       tcTileB(V, SM, T, Sg, KM, N, K, #if Kk +Int 1 ==Int K #then Nn +Int 1 #else Nn #fi, #if Kk +Int 1 ==Int K #then 0 #else Kk +Int 1 #fi,
               Acc[rc(Kk, Nn) <- wgNeg(T, Sg, wgElem(SM, tcDesc(V), KM, ew(T), Nn, Kk))])
    requires Nn <Int N
  rule tcTileB(_, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
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

  syntax Map ::= tcMma(Set, Map, Bytes, Bool, Int, Int, Int, Int, Bool, Int, Int, Int) [function, total]
  // mods, tmem, smem, a-is-desc, a value, b desc, idesc, d-taddr, enable, scale, mask -> new tmem
  rule tcMma(M, TM, SM, AD, AV, BV, ID, DA, En, S, Mask, Meta) =>
       tcPutD(TM,
              resRows("m" +String Int2String(((ID >>Int 24) &Int 31) *Int 16) +String "n" +String Int2String(((ID >>Int 17) &Int 63) *Int 8) +String "k" +String Int2String(tcKS(M, ID, Meta)),
                      tcDT(M, ID), tcAT(M, ID), tcBT(M, ID), tcDT(M, ID),
                      tcAMat(M, TM, SM, AD, AV, ID, Meta),
                      tcTileB(BV, SM, tcBT(M, ID), #if ((ID >>Int 14) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 16) &Int 1) ==Int 0,
                              ((ID >>Int 17) &Int 63) *Int 8, tcKS(M, ID, Meta), 0, 0, .Map),
                      #if En #then tcTileD(TM, tcDT(M, ID), tcCol(DA), ((ID >>Int 24) &Int 31) *Int 16, ((ID >>Int 17) &Int 63) *Int 8, 0, 0, S, 0, .Map) #else .Map #fi,
                      0, 0, #if ((ID >>Int 3) &Int 1) ==Int 1 #then SetItem("satfinite") #else .Set #fi, .Map),
              rowKeys(((ID >>Int 24) &Int 31) *Int 16, ((ID >>Int 17) &Int 63) *Int 8, 0, 0, .List),
              tcCol(DA), Mask, 0, 0, 0, tcDT(M, ID))
  // sparse A (tcgen05.mma.sp): the packed M x K/2 matrix is read like a dense one and expanded with the metadata
  // words in tensor memory (assumed layout, see docs/prior-work.md): row r's metadata sits in lane r, 4 bits per chunk;
  // 16-bit and tf32 kinds use one word, selected by the descriptor's sparsity selector (column base + selector);
  // 8-bit kinds use two consecutive words (chunks 0-7, 8-15).
  syntax Int ::= tcKS(Set, Int, Int) [function, total]
  rule tcKS(M, ID, Meta) => 2 *Int tcK(M, ID) requires Meta >=Int 0
  rule tcKS(M, ID, Meta) => tcK(M, ID) requires Meta <Int 0
  syntax Map ::= tcAMat(Set, Map, Bytes, Bool, Int, Int, Int) [function, total]
  rule tcAMat(M, TM, SM, AD, AV, ID, Meta) =>
       tcTileA(AD, AV, TM, SM, tcAT(M, ID), #if ((ID >>Int 13) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 15) &Int 1) ==Int 0,
               ((ID >>Int 24) &Int 31) *Int 16, tcK(M, ID), 0, 0, 0, .Map)
    requires Meta <Int 0
  rule tcAMat(M, TM, SM, AD, AV, ID, Meta) =>
       tcSpExp(tcTileA(AD, AV, TM, SM, tcAT(M, ID), #if ((ID >>Int 13) &Int 1) ==Int 1 #then 0 -Int 1 #else 1 #fi, ((ID >>Int 15) &Int 1) ==Int 0,
                       ((ID >>Int 24) &Int 31) *Int 16, tcK(M, ID), 0, 0, 0, .Map),
               TM, tcAT(M, ID), ((ID >>Int 24) &Int 31) *Int 16, tcK(M, ID), Meta, ID &Int 3, 0, 0, .Map)
    requires Meta >=Int 0
  syntax Map ::= tcSpExp(Map, Map, Ty, Int, Int, Int, Int, Int, Int, Map) [function, total]   // packed, tmem, type, rows, packed cols, meta column, selector, r, kp, acc
  rule tcSpExp(Pk, TM, T, M, KP, MC, Sel, R, K, Acc) =>
       tcSpExp(Pk, TM, T, M, KP, MC, Sel, #if K +Int 1 ==Int KP then R +Int 1 #else R #fi, #if K +Int 1 ==Int KP then 0 #else K +Int 1 #fi,
               Acc[rc(R, spCw(T) *Int (K /Int spNz(T)) +Int spPos(T, tcNib(TM, T, R, MC, Sel, K /Int spNz(T), KP), K %Int spNz(T)))
                   <- {Pk[rc(R, K)] orDefault 0}:>Int])
    requires R <Int M
  rule tcSpExp(_, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  syntax Int ::= tcNib(Map, Ty, Int, Int, Int, Int, Int) [function, total]                      // tmem, type, row, meta col, selector, chunk, packed cols
  rule tcNib(TM, T, R, MC, Sel, C, KP) => (tcWord(TM, tcKey(R, MC +Int Sel)) >>Int (4 *Int C)) &Int 15 requires 2 *Int KP /Int spCw(T) <=Int 8
  rule tcNib(TM, T, R, MC, Sel, C, KP) => (tcWord(TM, tcKey(R, MC +Int C /Int 8)) >>Int (4 *Int (C %Int 8))) &Int 15 requires 2 *Int KP /Int spCw(T) >Int 8

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
    requires "mma" in M andBool notBool "sp" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], Ad, Bd, ID, Rest:Operands), C) =>
            #tcMma(M, true, rd(Ad, ty("b", 64), C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), 0 -Int 1, opsList(Rest), C) ... </tk>
    requires "mma" in M andBool notBool "sp" in M [priority(60)]
  // sparse: [d], a-desc | [a-tmem], b-desc, [sp-meta-tmem], idesc, ...
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], [ A ], Bd, [ MT ], ID, Rest:Operands), C) =>
            #tcMma(M, false, tcAddr(A, C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), tcCol(tcAddr(MT, C)), opsList(Rest), C) ... </tk>
    requires "mma" in M andBool "sp" in M
  rule <tk> exec(opc("tcgen05", M, _), ops([ D ], Ad, Bd, [ MT ], ID, Rest:Operands), C) =>
            #tcMma(M, true, rd(Ad, ty("b", 64), C), rd(Bd, ty("b", 64), C), rd(ID, ty("b", 32), C), tcAddr(D, C), tcCol(tcAddr(MT, C)), opsList(Rest), C) ... </tk>
    requires "mma" in M andBool "sp" in M [priority(60)]
  syntax KItem ::= #tcMma(Set, Bool, Int, Int, Int, Int, Int, List, Ctx)
  rule <tk> #tcMma(M, AD, AV, BV, ID, DA, Meta, Rest, C) => .K ... </tk>
       <smem> SM </smem>
       <tmem> TM => tcMma(M, TM, SM, AD, AV, BV, ID, DA, tcEn(Rest, C), tcSc(Rest, C), tcMk(Rest, C), Meta) </tmem>
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
  syntax Int ::= tcSc(List, Ctx) [function, total]
  rule tcSc(L, C) => rd({tcOps(L)[1]}:>Operand, ty("u", 32), C) requires size(tcOps(L)) >Int 1
  rule tcSc(_, _) => 0 [owise]
endmodule

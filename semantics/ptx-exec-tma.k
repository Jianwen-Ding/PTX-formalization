// =============================================================================
// PTX-EXEC-TMA: tensor maps and the Tensor Memory Accelerator (sm_90)
//
//   cp.async.bulk.tensor.<d>d.{shared::cta,shared::cluster}.global.{tile,tile::gather4}.mbarrier::complete_tx::bytes
//        [dst], [tmap, {coords}], [mbar]{, ctaMask}{, policy}          (loads, optional multicast)
//   cp.async.bulk.tensor.<d>d.global.shared::cta.{tile,tile::scatter4}.bulk_group   [tmap, {coords}], [src]
//   cp.reduce.async.bulk.tensor.<d>d.global.shared::cta.<redOp>.tile.bulk_group     [tmap, {coords}], [src]
//   cp.async.bulk.prefetch.tensor / tensormap.replace / tensormap.cp_fenceproxy / prefetch.tensormap
//
// A tensor map is a 128-byte object that the host API (cuTensorMapEncode*) fills with an opaque
// encoding; here the harness directive .tensormap builds it in our own encoding (see
// ptx-harness.k): element type code, rank, global address, dims, byte strides, box dims,
// element strides, swizzle mode and OOB fill.  A tile copy moves the box of Pi(box_i) elements whose
// origin is `coords`: element e (dim 0 fastest) comes from global coordinate coords_i + b_i*estride_i,
// out-of-bounds elements are filled with zeros (or a NaN pattern with fill mode 1); the shared-memory
// image is dense row-major with 16-byte chunks permuted by the swizzle mode (a function of the
// shared address bits).  Completion is a complete-tx of the number of bytes in the box.
// Not modelled: interleaved layouts, the .override forms of the instructions and
// tensor maps in .param space.
// =============================================================================
requires "ptx-exec-bulk.k"
requires "ptx-exec-wg.k"
requires "ptx-exec-ldm.k"

module PTX-EXEC-TMA
  imports PTX-EXEC-BULK
  imports PTX-EXEC-WG
  imports PTX-EXEC-LDM

  // ---------------------------------------------------------------- tensor-map fields
  syntax Int ::= tmW(Int, Int) [function, total] | tmDim(Int, Int) [function, total] | tmStride(Int, Int) [function, total]
               | tmBox(Int, Int) [function, total] | tmEst(Int, Int) [function, total] | tmSwz(Int) [function, total]
               | tmFillMode(Int) [function, total] | tmRank(Int) [function, total] | tmAddrOf(Int) [function, total] | tmEb(Int) [function, total]
  rule tmW(B, I) => (B >>Int (64 *Int I)) &Int 18446744073709551615
  rule tmRank(B) => tmW(B, 2) &Int 255
  rule tmAddrOf(B) => tmW(B, 3)
  rule tmDim(B, I) => tmW(B, 4 +Int I)
  rule tmStride(B, I) => tmW(B, 8 +Int I)
  rule tmBox(B, I) => (tmW(B, 13) >>Int (16 *Int I)) &Int 65535 requires I <Int 4
  rule tmBox(B, _) => tmW(B, 14) &Int 65535 [owise]
  rule tmEst(B, I) => (tmW(B, 15) >>Int (8 *Int I)) &Int 255
  rule tmSwz(B) => (tmW(B, 14) >>Int 16) &Int 255                 // swizzle_mode
  syntax Int ::= tmAtom(Int) [function, total] | tmEffMode(Int) [function, total]
  rule tmAtom(B) => (tmW(B, 14) >>Int 48) &Int 255                  // swizzle_atomicity
  rule tmFillMode(B) => (tmW(B, 14) >>Int 24) &Int 255
  // element size / type from the CUtensorMapDataType code
  rule tmEb(B) => tmBytesOf(tmW(B, 1) &Int 255)
  syntax Int ::= tmBytesOf(Int) [function, total]
  // element type values of tensormap.replace.elemtype: 0 u8, 1 u16, 2 u32, 3 s32, 4 u64, 5 s64, 6 f16, 7 f32, 8 f32.ftz, 9 f64, 10 bf16, 11 tf32, 12 tf32.ftz,
  // 13 b4x16, 14 b4x16_p64, 15 b6x16_p32 (packed sub-byte groups of 16 elements; one byte is used as the container size here)
  rule tmBytesOf(0) => 1  rule tmBytesOf(1) => 2  rule tmBytesOf(2) => 4  rule tmBytesOf(3) => 4  rule tmBytesOf(4) => 8  rule tmBytesOf(5) => 8
  rule tmBytesOf(6) => 2  rule tmBytesOf(7) => 4  rule tmBytesOf(8) => 4  rule tmBytesOf(9) => 8  rule tmBytesOf(10) => 2 rule tmBytesOf(11) => 4 rule tmBytesOf(12) => 4
  rule tmBytesOf(_) => 1 [owise]
  syntax Ty ::= tmTy(Int) [function, total]
  rule tmTy(B) => tmTyOf(tmW(B, 1) &Int 255)
  syntax Ty ::= tmTyOf(Int) [function, total]
  rule tmTyOf(0) => ty("u", 8)  rule tmTyOf(1) => ty("u", 16)  rule tmTyOf(2) => ty("u", 32)  rule tmTyOf(3) => ty("s", 32)
  rule tmTyOf(4) => ty("u", 64)  rule tmTyOf(5) => ty("s", 64)  rule tmTyOf(6) => ty("f", 16)  rule tmTyOf(7) => ty("f", 32)
  rule tmTyOf(8) => ty("f", 32)  rule tmTyOf(9) => ty("f", 64)  rule tmTyOf(10) => ty("bf", 16)  rule tmTyOf(11) => ty("f", 32)  rule tmTyOf(12) => ty("f", 32)
  rule tmTyOf(_) => ty("u", 8) [owise]
  // swizzle_mode / swizzle_atomicity of the tensor map -> mode of ptx-exec-wg.k's swizzle (1 = 128B, 2 = 64B, 3 = 32B, 4 = 128B/32B atomicity,
  // 5 = ... + 8-byte flip, 6 = 128B/64B atomicity); the 96B swizzle (mode 4) is not modelled
  rule tmEffMode(B) => 3 requires tmSwz(B) ==Int 1
  rule tmEffMode(B) => 2 requires tmSwz(B) ==Int 2
  rule tmEffMode(B) => 1 requires tmSwz(B) ==Int 3 andBool tmAtom(B) ==Int 0
  rule tmEffMode(B) => 4 requires tmSwz(B) ==Int 3 andBool tmAtom(B) ==Int 1
  rule tmEffMode(B) => 5 requires tmSwz(B) ==Int 3 andBool tmAtom(B) ==Int 2
  rule tmEffMode(B) => 6 requires tmSwz(B) ==Int 3 andBool tmAtom(B) ==Int 3
  rule tmEffMode(_) => 0 [owise]
  syntax Int ::= tmFill(Int) [function, total]
  rule tmFill(B) => 0 requires tmFillMode(B) ==Int 0
  rule tmFill(B) => 2143289344 requires tmFillMode(B) =/=Int 0 andBool tmEb(B) ==Int 4              // 0x7fc00000
  rule tmFill(B) => 32256 requires tmFillMode(B) =/=Int 0 andBool (tmW(B, 1) &Int 255) ==Int 6          // f16 0x7e00
  rule tmFill(B) => 32704 requires tmFillMode(B) =/=Int 0 andBool (tmW(B, 1) &Int 255) ==Int 10          // bf16 0x7fc0
  rule tmFill(B) => 9221120237041090560 requires tmFillMode(B) =/=Int 0 andBool (tmW(B, 1) &Int 255) ==Int 9        // f64 0x7ff8000000000000
  rule tmFill(_) => 0 [owise]

  // ---------------------------------------------------------------- element addresses
  // total elements of the box (tile) or of the 4 gathered rows (gather4/scatter4)
  syntax Int ::= tmCount(Int, Bool) [function, total] | tmVol(Int, Int, Int) [function, total]
  rule tmCount(B, false) => tmVol(B, 0, 1)
  rule tmCount(B, true) => 4 *Int tmBox(B, 0)
  rule tmVol(B, I, A) => tmVol(B, I +Int 1, A *Int tmBox(B, I)) requires I <Int tmRank(B)
  rule tmVol(_, _, A) => A [owise]
  // global address of element e (or -1 if out of bounds)
  syntax Int ::= tmAddr(Int, List, Int, Bool) [function, total]
  rule tmAddr(B, Cs, E, false) => tmAddrT(B, Cs, E, 0, 1, 0)
  rule tmAddr(B, Cs, E, true) => tmAddrG(B, Cs, E)
  syntax Int ::= tmAddrT(Int, List, Int, Int, Int, Int) [function, total]            // blob, coords, e, i, divisor, accumulated offset
  rule tmAddrT(B, Cs, E, I, Dv, Acc) =>
       #if gcoord(B, Cs, E, I, Dv) <Int 0 orBool gcoord(B, Cs, E, I, Dv) >=Int tmDim(B, I) #then 0 -Int 1
       #else tmAddrT(B, Cs, E, I +Int 1, Dv *Int tmBox(B, I), Acc +Int #if I ==Int 0 #then gcoord(B, Cs, E, I, Dv) *Int tmEb(B) #else gcoord(B, Cs, E, I, Dv) *Int tmStride(B, I) #fi) #fi
    requires I <Int tmRank(B)
  rule tmAddrT(B, _, _, _, _, Acc) => tmAddrOf(B) +Int Acc [owise]
  syntax Int ::= gcoord(Int, List, Int, Int, Int) [function, total]
  rule gcoord(B, Cs, E, I, Dv) => nth(Cs, I) +Int ((E /Int Dv) %Int tmBox(B, I)) *Int tmEst(B, I)
  // gather4 / scatter4: coords {col, row0..row3}; element e = q * box0 + b0
  syntax Int ::= tmAddrG(Int, List, Int) [function, total]
  rule tmAddrG(B, Cs, E) =>
       #if nth(Cs, 0) +Int (E %Int tmBox(B, 0)) <Int 0 orBool nth(Cs, 0) +Int (E %Int tmBox(B, 0)) >=Int tmDim(B, 0)
            orBool nth(Cs, 1 +Int E /Int tmBox(B, 0)) <Int 0 orBool nth(Cs, 1 +Int E /Int tmBox(B, 0)) >=Int tmDim(B, 1)
       #then 0 -Int 1
       #else tmAddrOf(B) +Int (nth(Cs, 0) +Int (E %Int tmBox(B, 0))) *Int tmEb(B) +Int nth(Cs, 1 +Int E /Int tmBox(B, 0)) *Int tmStride(B, 1) #fi


  // ---------------------------------------------------------------- operand forms: [tensorMap{, gAddrToOverride}{, attributeOverrideInfo}, tensorCoords]
  // (.override::global_address / .override::global_dim[_stride], PTX ISA 9.4): the tensor-map object is patched before use.
  // .cta_group::2 with multicast: the mbarrier signal for destination CTA d goes to the CTA of d's pair whose rank parity is that of the CTA holding `mbar`
  syntax List ::= tmBarLocs(Loc, Int, Bool, Bool, Int) [function, total]            // mbar location, ctaMask, multicast?, cta_group::2?, own rank
  rule tmBarLocs(L, Mask, Mc, false, _) => bulkLocs(L, Mask, Mc)
  rule tmBarLocs(L, Mask, false, true, _) => ListItem(L)
  rule tmBarLocs(loc(S, A), Mask, true, true, Own) => tmBarRanks(A &Int 16777215, Mask, 0, #if (A >>Int 24) >Int 0 #then ((A >>Int 24) -Int 1) %Int 2 #else Own %Int 2 #fi)
  rule tmBarLocs(L, _, _, _, _) => ListItem(L) [owise]
  syntax List ::= tmBarRanks(Int, Int, Int, Int) [function, total]
  rule tmBarRanks(Off, Mask, R, Par) =>
       #if (Mask >>Int R) &Int 1 ==Int 1 #then ListItem(loc("cluster", ((R -Int R %Int 2 +Int Par +Int 1) <<Int 24) |Int Off)) #else .List #fi tmBarRanks(Off, Mask, R +Int 1, Par)
    requires R <Int 32 andBool (Mask >>Int R) >Int 0
  rule tmBarRanks(_, _, _, _) => .List [owise]
  syntax KItem ::= #tmLd(Set, Addr, Int, Operand, Addr, Operands, Ctx) | #tmSt(Set, Int, Operand, Addr, Ctx)
  syntax Int ::= tmObj(Bytes, Loc) [function, total]
  rule tmObj(GM, L) => readBytes(GM, locOff(L), 128)
  // new base address; dims (8- or 16-bit values) and strides ((lower + (upper << 32)) << 4, upper = 4-bit fields of one register) for the dimensions given
  syntax Int ::= tmOvAddr(Int, Int) [function, total] | tmOvDims(Int, List, Int) [function, total] | tmOvStr(Int, List, Int, Int) [function, total]
  rule tmOvAddr(B, G) => setWord(B, 3, G)
  rule tmOvDims(B, ListItem(V:Int) Vs, I) => tmOvDims(setWord(B, 4 +Int I, V), Vs, I +Int 1)
  rule tmOvDims(B, _, _) => B [owise]
  rule tmOvStr(B, ListItem(V:Int) Vs, Up, I) => tmOvStr(setWord(B, 9 +Int I, (V +Int (((Up >>Int (4 *Int I)) &Int 15) <<Int 32)) <<Int 4), Vs, Up, I +Int 1)
  rule tmOvStr(B, _, _, _) => B [owise]
  syntax Int ::= tmOv(Int, Int, List, List, Int) [function, total]          // blob, address, dims, lower strides, upper strides
  rule tmOv(B, G, Ds, Ls, Up) => tmOvStr(tmOvDims(tmOvAddr(B, G), Ds, 0), Ls, Up, 0)
  rule <tk> exec(opc("cp", M, _), ops([ D:Addr ], [ T:Addr , Cs ], [ Mb:Addr ], Rest:Operands), C) => #tmLd(M, D, tmObj(GM, ea(T, "generic", C)), Cs, Mb, Rest, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M
  rule <tk> exec(opc("cp", M, _), ops([ D:Addr ], [ T:Addr , G:Addr , Cs ], [ Mb:Addr ], Rest:Operands), C) =>
            #tmLd(M, D, tmOvAddr(tmObj(GM, ea(T, "generic", C)), locOff(ea(G, "generic", C))), Cs, Mb, Rest, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M
  rule <tk> exec(opc("cp", M, _), ops([ D:Addr ], [ T:Addr , G:Addr , Sz , Cs ], [ Mb:Addr ], Rest:Operands), C) =>
            #tmLd(M, D, tmOv(tmObj(GM, ea(T, "generic", C)), locOff(ea(G, "generic", C)), tmCoords(Sz, C), .List, 0), Cs, Mb, Rest, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M
  rule <tk> exec(opc("cp", M, _), ops([ D:Addr ], [ T:Addr , G:Addr , Sz , Lo , Up , Cs ], [ Mb:Addr ], Rest:Operands), C) =>
            #tmLd(M, D, tmOv(tmObj(GM, ea(T, "generic", C)), locOff(ea(G, "generic", C)), tmCoords(Sz, C), tmCoords(Lo, C), rd(Up, ty("b", 16), C)), Cs, Mb, Rest, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M
  rule <tk> exec(opc("cp", M, _), ops([ T:Addr , Cs ], [ S:Addr ], _:Operands), C) => #tmSt(M, tmObj(GM, ea(T, "generic", C)), Cs, S, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M
  rule <tk> exec(opc("cp", M, _), ops([ T:Addr , G:Addr , Cs ], [ S:Addr ], _:Operands), C) =>
            #tmSt(M, tmOvAddr(tmObj(GM, ea(T, "generic", C)), locOff(ea(G, "generic", C))), Cs, S, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M
  rule <tk> exec(opc("cp", M, _), ops([ T:Addr , G:Addr , Sz , Cs ], [ S:Addr ], _:Operands), C) =>
            #tmSt(M, tmOv(tmObj(GM, ea(T, "generic", C)), locOff(ea(G, "generic", C)), tmCoords(Sz, C), .List, 0), Cs, S, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M
  rule <tk> exec(opc("cp", M, _), ops([ T:Addr , G:Addr , Sz , Lo , Up , Cs ], [ S:Addr ], _:Operands), C) =>
            #tmSt(M, tmOv(tmObj(GM, ea(T, "generic", C)), locOff(ea(G, "generic", C)), tmCoords(Sz, C), tmCoords(Lo, C), rd(Up, ty("b", 16), C)), Cs, S, C) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M

  // ---------------------------------------------------------------- loads: global -> shared image chunks
  syntax KItem ::= #tmWr(String, List) | #tmGWr(List, String, Ty)
  syntax TmW ::= tmw(Int, Int)                         // shared/cluster address, 16-byte value
  // mode 5 (128B swizzle, 32B atomicity, 8-byte flip): chunks move like mode 4 and, on odd 128-byte lines, their two 8-byte halves are exchanged
  syntax TmW ::= tmWrap(Int, Int, Int) [function, total]
  rule tmWrap(5, A, V) => tmw(swizzle(4, A), #if (A >>Int 7) &Int 1 ==Int 1 #then ((V &Int 18446744073709551615) <<Int 64) |Int (V >>Int 64) #else V #fi)
  rule tmWrap(M, A, V) => tmw(swizzle(M, A), V) [owise]
  syntax TmG ::= tmg(Int, Int, Int)                    // global address, bytes, value
  syntax Int ::= tmElem(Map, Int, List, Int, Bool) [function, total]            // gmem as a Bytes cell, ...
  syntax Int ::= tmElemV(Bytes, Int, List, Int, Bool) [function, total]
  rule tmElemV(GM, B, Cs, E, G) => #if tmAddr(B, Cs, E, G) <Int 0 #then tmFill(B) #else readBytes(GM, tmAddr(B, Cs, E, G), tmEb(B)) #fi
  syntax Int ::= tmChunkV(Bytes, Int, List, Int, Int, Bool) [function, total]      // first element of the chunk, j, ...
  rule tmChunkV(GM, B, Cs, E, J, G) =>
       (tmElemV(GM, B, Cs, E +Int J, G) <<Int (8 *Int tmEb(B) *Int J)) |Int tmChunkV(GM, B, Cs, E, J +Int 1, G) requires J <Int 16 /Int tmEb(B)
  rule tmChunkV(_, _, _, _, _, _) => 0 [owise]
  syntax List ::= tmLoadList(Bytes, Int, List, Int, Int, Bool) [function, total]    // gmem, blob, coords, dst base, chunk k, gather4?
  rule tmLoadList(GM, B, Cs, Base, K, G) =>
       ListItem(tmWrap(tmEffMode(B), Base +Int 16 *Int K, tmChunkV(GM, B, Cs, K *Int (16 /Int tmEb(B)), 0, G)))
       tmLoadList(GM, B, Cs, Base, K +Int 1, G)
    requires K *Int 16 <Int tmCount(B, G) *Int tmEb(B)
  rule tmLoadList(_, _, _, _, _, _) => .List [owise]
  syntax List ::= tmLoadAll(Bytes, Int, List, List, Bool) [function, total]         // over destination bases
  rule tmLoadAll(GM, B, Cs, ListItem(Base:Int) Bs, G) => tmLoadList(GM, B, Cs, Base, 0, G) tmLoadAll(GM, B, Cs, Bs, G)
  rule tmLoadAll(_, _, _, _, _) => .List [owise]
  syntax List ::= tmBases(Loc, Int, Bool) [function, total]
  rule tmBases(loc(_, A), _, false) => ListItem(A)
  rule tmBases(loc(_, A), Mask, true) => rankBases(A &Int 16777215, Mask, 0)
  rule tmBases(_, _, _) => .List [owise]
  syntax List ::= rankBases(Int, Int, Int) [function, total]
  rule rankBases(Off, Mask, R) => #if (Mask >>Int R) &Int 1 ==Int 1 #then ListItem(((R +Int 1) <<Int 24) |Int Off) #else .List #fi rankBases(Off, Mask, R +Int 1)
    requires R <Int 32 andBool (Mask >>Int R) >Int 0
  rule rankBases(_, _, _) => .List [owise]

  rule <tk> #tmLd(M, D, B, Cs, Mb, Rest, C) =>
            #tmWr(#if "shared::cluster" in M #then "cluster" #else "shared" #fi,
                  tmLoadAll(GM, B, tmCoords(Cs, C),
                            tmBases(ea(D, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) opsList(Rest), C), isMc(M)),
                            "gather4" in M orBool "tile::gather4" in M))
            ~> #tmTxAll(tmBarLocs(ea(Mb, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) opsList(Rest), C), isMc(M), "cta_group::2" in M, rreg("%cluster_ctarank", C)),
                        tmCount(B, "tile::gather4" in M) *Int tmEb(B)) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M andBool notBool isI2c(M) andBool notBool tmSub(B)
  syntax List ::= tmCoords(Operand, Ctx) [function, total]
  rule tmCoords({ Os }, C) => tmCoordList(opsList(Os), C)
  rule tmCoords(_, _) => .List [owise]
  syntax List ::= tmCoordList(List, Ctx) [function, total]
  rule tmCoordList(ListItem(O:Operand) Os, C) => ListItem(ival(rd(O, ty("s", 32), C), ty("s", 32))) tmCoordList(Os, C)
  rule tmCoordList(_, _) => .List [owise]
  syntax KItem ::= #tmTxAll(List, Int)
  rule <tk> #tmWr(S, ListItem(tmw(A, V)) Ls) => #mem(loc(S, A), wrOp(16, V)) ~> #drop ~> #tmWr(S, Ls) ... </tk>
  rule <tk> #tmWr(_, .List) => .K ... </tk>
  rule <tk> #tmTxAll(ListItem(Mb) Ms, N) => #bulkTx(Mb, N) ~> #tmTxAll(Ms, N) ... </tk>
  rule <tk> #tmTxAll(.List, _) => .K ... </tk>



  // ---------------------------------------------------------------- packed sub-byte element types (elemtype 13 b4x16, 14 b4x16_p64, 15 b6x16_p32)
  // dim 0 counts elements and is a multiple of 16; a group of 16 elements takes 8 (4-bit) or 12 (6-bit) bytes in global memory and 8 (b4x16) or 16 bytes
  // (the _p64 / _p32 formats: data followed by padding, written as zeros here) in shared memory.  Groups are laid out like the elements of a tile.
  syntax Bool ::= tmSub(Int) [function, total]
  rule tmSub(B) => (tmW(B, 1) &Int 255) >=Int 13
  syntax Int ::= tmGB(Int) [function, total] | tmSB(Int) [function, total] | tmNb(Int) [function, total]
  rule tmGB(B) => 12 requires (tmW(B, 1) &Int 255) ==Int 15
  rule tmGB(_) => 8 [owise]
  rule tmSB(B) => 8 requires (tmW(B, 1) &Int 255) ==Int 13
  rule tmSB(_) => 16 [owise]
  rule tmNb(B) => 6 requires (tmW(B, 1) &Int 255) ==Int 15
  rule tmNb(_) => 4 [owise]
  syntax Int ::= tmGroups(Int) [function, total] | tmGrpAddr(Int, List, Int) [function, total] | tmGrpK(Int, List, Int, Int, Int, Int) [function, total]
  rule tmGroups(B) => (tmBox(B, 0) /Int 16) *Int tmVol(B, 1, 1)
  rule tmGrpAddr(B, Cs, Q) =>
       #if nth(Cs, 0) +Int (Q %Int (tmBox(B, 0) /Int 16)) *Int 16 <Int 0 orBool nth(Cs, 0) +Int (Q %Int (tmBox(B, 0) /Int 16)) *Int 16 >=Int tmDim(B, 0) #then 0 -Int 1
       #else tmGrpK(B, Cs, Q /Int (tmBox(B, 0) /Int 16), 1,
                    tmAddrOf(B) +Int ((nth(Cs, 0) +Int (Q %Int (tmBox(B, 0) /Int 16)) *Int 16) *Int tmNb(B)) /Int 8, 0) #fi
  rule tmGrpK(B, Cs, Q, I, Acc, _) =>
       #if nth(Cs, I) +Int (Q %Int tmBox(B, I)) *Int tmEst(B, I) <Int 0 orBool nth(Cs, I) +Int (Q %Int tmBox(B, I)) *Int tmEst(B, I) >=Int tmDim(B, I) #then 0 -Int 1
       #else tmGrpK(B, Cs, Q /Int tmBox(B, I), I +Int 1, Acc +Int (nth(Cs, I) +Int (Q %Int tmBox(B, I)) *Int tmEst(B, I)) *Int tmStride(B, I), 0) #fi
    requires I <Int tmRank(B)
  rule tmGrpK(_, _, _, _, Acc, _) => Acc [owise]
  syntax Int ::= tmGrpVal(Bytes, Int, List, Int) [function, total]
  rule tmGrpVal(GM, B, Cs, Q) => #if Q >=Int tmGroups(B) orBool tmGrpAddr(B, Cs, Q) <Int 0 #then 0 #else readBytes(GM, tmGrpAddr(B, Cs, Q), tmGB(B)) #fi
  syntax Int ::= tmChunkS(Bytes, Int, List, Int) [function, total]
  rule tmChunkS(GM, B, Cs, K) => tmGrpVal(GM, B, Cs, K) requires tmSB(B) ==Int 16
  rule tmChunkS(GM, B, Cs, K) => tmGrpVal(GM, B, Cs, 2 *Int K) |Int (tmGrpVal(GM, B, Cs, 2 *Int K +Int 1) <<Int 64) [owise]
  syntax List ::= tmLoadListS(Bytes, Int, List, Int, Int) [function, total]
  rule tmLoadListS(GM, B, Cs, Base, K) => ListItem(tmWrap(tmEffMode(B), Base +Int 16 *Int K, tmChunkS(GM, B, Cs, K))) tmLoadListS(GM, B, Cs, Base, K +Int 1)
    requires K *Int 16 <Int tmGroups(B) *Int tmSB(B)
  rule tmLoadListS(_, _, _, _, _) => .List [owise]
  syntax List ::= tmLoadAllS(Bytes, Int, List, List) [function, total]
  rule tmLoadAllS(GM, B, Cs, ListItem(Base:Int) Bs) => tmLoadListS(GM, B, Cs, Base, 0) tmLoadAllS(GM, B, Cs, Bs)
  rule tmLoadAllS(_, _, _, _) => .List [owise]
  rule <tk> #tmLd(M, D, B, Cs, Mb, Rest, C) =>
            #tmWr(#if "shared::cluster" in M #then "cluster" #else "shared" #fi,
                  tmLoadAllS(GM, B, tmCoords(Cs, C),
                             tmBases(ea(D, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) tmNoVec(opsList(Rest)), C), isMc(M))))
            ~> #tmTxAll(tmBarLocs(ea(Mb, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) tmNoVec(opsList(Rest)), C), isMc(M), "cta_group::2" in M, rreg("%cluster_ctarank", C)),
                        tmGroups(B) *Int tmSB(B)) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M andBool notBool isI2c(M) andBool tmSub(B)

  // ---------------------------------------------------------------- im2col modes
  // The tensor (dim 0 = channels C, dims 1..S = W, H, D, last dim = N; S = rank - 2) is read as a column of `pixels` pixels x `channels` channels
  // (box dims 1 and 0).  Bounding-box corners (tensor map): the filter-base positions along spatial dim k run over
  // Lk, Lk + st_k, ... up to size_k - 1 + Uk (st = traversal stride); there are nk = (size_k - 1 + Uk - Lk) div st_k + 1 of them.  The tensor
  // coordinates give the position of the first pixel (its index in the box is (c_k - Lk) div st_k); pixel p is the p-th position in
  // (N, D, H, W) order, W fastest, wrapping W into H into D into N.  Its data is read at base position + im2col offset (positions
  // outside the tensor, or images beyond N, read as the fill value).  The ISA gives the exact pixel enumeration only as figures (assumption).
  //   .im2col::w / .im2col::w::128: only W varies: pixel q is at c_W + q * st_W + wOffset, D / H / N are the coordinates; im2col::w loads
  //   pixels-per-column + wHalo pixels; ::128 loads 128 pixels with wHalo halo pixels after every 32.   .im2col_no_offs::w (stores): offsets 0, stops at the
  //   first pixel that is outside the tensor.
  syntax Bool ::= isI2c(Set) [function, total]
  rule isI2c(M) => "im2col" in M orBool "im2col::w" in M orBool "im2col::w::128" in M orBool "im2col_no_offs::w" in M
  syntax Int ::= imMode(Set) [function, total]
  rule imMode(M) => 1 requires "im2col::w" in M
  rule imMode(M) => 2 requires "im2col::w::128" in M
  rule imMode(M) => 3 requires "im2col_no_offs::w" in M
  rule imMode(_) => 0 [owise]
  syntax Int ::= tmLo(Int, Int) [function, total] | tmUp(Int, Int) [function, total]
  rule tmLo(B, K) => ((tmW(B, 1) >>Int (8 +Int 16 *Int K)) &Int 65535) -Int 32768
  rule tmUp(B, K) => ((tmW(B, 2) >>Int (8 +Int 16 *Int K)) &Int 65535) -Int 32768
  syntax Int ::= tmNk(Int, Int) [function, total]                   // number of base positions along spatial dim K (1..S)
  rule tmNk(B, K) => maxInt(1, (tmDim(B, K) -Int 1 +Int tmUp(B, K -Int 1) -Int tmLo(B, K -Int 1)) /Int tmEst(B, K) +Int 1)
  syntax Int ::= tmStart(Int, List, Int, Int, Int) [function, total]   // blob, coords, k, multiplier, accumulated start index
  rule tmStart(B, Cs, K, Mu, Acc) =>
       tmStart(B, Cs, K +Int 1, Mu *Int tmNk(B, K), Acc +Int ((nth(Cs, K) -Int tmLo(B, K -Int 1)) divInt tmEst(B, K)) *Int Mu)
    requires K <=Int tmRank(B) -Int 2
  rule tmStart(B, Cs, K, Mu, Acc) => Acc +Int nth(Cs, tmRank(B) -Int 1) *Int Mu [owise]
  // pixels loaded
  syntax Int ::= tmPix(Int, List, Int) [function, total]
  rule tmPix(B, _, 0) => tmBox(B, 1)
  rule tmPix(B, I, 1) => tmBox(B, 1) +Int nth(I, 0)
  rule tmPix(B, I, 2) => 128 +Int 4 *Int nth(I, 0)
  rule tmPix(B, _, 3) => tmBox(B, 1)
  rule tmPix(B, _, _) => tmBox(B, 1) [owise]
  // global address of element E (pixel E div cpp, channel E mod cpp), or -1
  syntax Int ::= tmAddrI(Int, List, List, Int, Int) [function, total]
  rule tmAddrI(B, Cs, I, Md, E) =>
       #if nth(Cs, 0) +Int (E %Int tmBox(B, 0)) *Int tmEst(B, 0) <Int 0 orBool nth(Cs, 0) +Int (E %Int tmBox(B, 0)) *Int tmEst(B, 0) >=Int tmDim(B, 0) #then 0 -Int 1
       #else tmAddrIk(B, Cs, I, Md, E /Int tmBox(B, 0), 1, tmAddrOf(B) +Int (nth(Cs, 0) +Int (E %Int tmBox(B, 0)) *Int tmEst(B, 0)) *Int tmEb(B),
                      tmStart(B, Cs, 1, 1, 0) +Int E /Int tmBox(B, 0)) #fi
  // spatial dims one by one; Idx is the remaining linear pixel index (modes 0)
  syntax Int ::= tmAddrIk(Int, List, List, Int, Int, Int, Int, Int) [function, total]     // blob, coords, info, mode, pixel, k, acc, idx
  rule tmAddrIk(B, Cs, I, 0, P, K, Acc, Idx) =>
       #if tmLo(B, K -Int 1) +Int (Idx %Int tmNk(B, K)) *Int tmEst(B, K) +Int nth(I, K -Int 1) <Int 0
           orBool tmLo(B, K -Int 1) +Int (Idx %Int tmNk(B, K)) *Int tmEst(B, K) +Int nth(I, K -Int 1) >=Int tmDim(B, K) #then 0 -Int 1
       #else tmAddrIk(B, Cs, I, 0, P, K +Int 1,
                      Acc +Int (tmLo(B, K -Int 1) +Int (Idx %Int tmNk(B, K)) *Int tmEst(B, K) +Int nth(I, K -Int 1)) *Int tmStride(B, K), Idx /Int tmNk(B, K)) #fi
    requires K <=Int tmRank(B) -Int 2
  rule tmAddrIk(B, Cs, I, 0, P, K, Acc, Idx) =>
       #if Idx <Int 0 orBool Idx >=Int tmDim(B, tmRank(B) -Int 1) #then 0 -Int 1 #else Acc +Int Idx *Int tmStride(B, tmRank(B) -Int 1) #fi [owise]
  // W modes: positions are c_W + q * st_W + wOffset; D / H / N are the coordinates themselves
  rule tmAddrIk(B, Cs, I, Md, P, K, Acc, Idx) => tmAddrW(B, Cs, tmQ(Md, nth(I, 0), P) *Int tmEst(B, 1) +Int tmWOff(Md, I), 1, Acc) requires Md =/=Int 0
  syntax Int ::= tmQ(Int, Int, Int) [function, total]
  rule tmQ(2, Halo, P) => 32 *Int (P /Int (32 +Int Halo)) +Int P %Int (32 +Int Halo)
  rule tmQ(_, _, P) => P [owise]
  syntax Int ::= tmWOff(Int, List) [function, total]
  rule tmWOff(3, _) => 0
  rule tmWOff(_, I) => nth(I, 1) [owise]
  syntax Int ::= tmAddrW(Int, List, Int, Int, Int) [function, total]      // blob, coords, w shift, k, acc
  rule tmAddrW(B, Cs, Sh, K, Acc) =>
       #if nth(Cs, K) +Int #if K ==Int 1 #then Sh #else 0 #fi <Int 0 orBool nth(Cs, K) +Int #if K ==Int 1 #then Sh #else 0 #fi >=Int tmDim(B, K) #then 0 -Int 1
       #else tmAddrW(B, Cs, Sh, K +Int 1, Acc +Int (nth(Cs, K) +Int #if K ==Int 1 #then Sh #else 0 #fi) *Int tmStride(B, K)) #fi
    requires K <=Int tmRank(B) -Int 2
  rule tmAddrW(B, Cs, _, K, Acc) =>
       #if nth(Cs, tmRank(B) -Int 1) <Int 0 orBool nth(Cs, tmRank(B) -Int 1) >=Int tmDim(B, tmRank(B) -Int 1) #then 0 -Int 1
       #else Acc +Int nth(Cs, tmRank(B) -Int 1) *Int tmStride(B, tmRank(B) -Int 1) #fi [owise]
  // the W-mode helper above is called with `Acc` already holding the channel offset; its first call passes K = 1
  // loads
  syntax Int ::= tmElemI(Bytes, Int, List, List, Int, Int) [function, total]
  rule tmElemI(GM, B, Cs, I, Md, E) => #if tmAddrI(B, Cs, I, Md, E) <Int 0 #then tmFill(B) #else readBytes(GM, tmAddrI(B, Cs, I, Md, E), tmEb(B)) #fi
  syntax Int ::= tmChunkI(Bytes, Int, List, List, Int, Int, Int) [function, total]
  rule tmChunkI(GM, B, Cs, I, Md, E, J) =>
       (tmElemI(GM, B, Cs, I, Md, E +Int J) <<Int (8 *Int tmEb(B) *Int J)) |Int tmChunkI(GM, B, Cs, I, Md, E, J +Int 1) requires J <Int 16 /Int tmEb(B)
  rule tmChunkI(_, _, _, _, _, _, _) => 0 [owise]
  syntax List ::= tmLoadListI(Bytes, Int, List, List, Int, Int, Int) [function, total]
  rule tmLoadListI(GM, B, Cs, I, Md, Base, K) =>
       ListItem(tmWrap(tmEffMode(B), Base +Int 16 *Int K, tmChunkI(GM, B, Cs, I, Md, K *Int (16 /Int tmEb(B)), 0)))
       tmLoadListI(GM, B, Cs, I, Md, Base, K +Int 1)
    requires K *Int 16 <Int tmPix(B, I, Md) *Int tmBox(B, 0) *Int tmEb(B)
  rule tmLoadListI(_, _, _, _, _, _, _) => .List [owise]
  syntax List ::= tmLoadAllI(Bytes, Int, List, List, Int, List) [function, total]
  rule tmLoadAllI(GM, B, Cs, I, Md, ListItem(Base:Int) Bs) => tmLoadListI(GM, B, Cs, I, Md, Base, 0) tmLoadAllI(GM, B, Cs, I, Md, Bs)
  rule tmLoadAllI(_, _, _, _, _, _) => .List [owise]
  // the first vector operand after the mbarrier: im2col offsets / {wHalo, wOffset}
  syntax List ::= tmNoVec(List) [function, total]
  rule tmNoVec(ListItem(O:Operand) Os) => #if isVec(O) #then tmNoVec(Os) #else ListItem(O) tmNoVec(Os) #fi
  rule tmNoVec(_) => .List [owise]
  syntax List ::= tmInfo(Operands, Ctx) [function, total] | tmInfoL(List, Ctx) [function, total]
  rule tmInfo(Rest, C) => tmInfoL(opsList(Rest), C)
  rule tmInfoL(ListItem(O:Operand) Os, C) => #if isVec(O) #then tmCoords(O, C) #else tmInfoL(Os, C) #fi
  rule tmInfoL(_, _) => .List [owise]
  rule <tk> #tmLd(M, D, B, Cs, Mb, Rest, C) =>
            #tmWr(#if "shared::cluster" in M #then "cluster" #else "shared" #fi,
                  tmLoadAllI(GM, B, tmCoords(Cs, C), tmInfo(Rest, C), imMode(M),
                             tmBases(ea(D, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) tmNoVec(opsList(Rest)), C), isMc(M))))
            ~> #tmTxAll(tmBarLocs(ea(Mb, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) tmNoVec(opsList(Rest)), C), isMc(M), "cta_group::2" in M, rreg("%cluster_ctarank", C)),
                        tmPix(B, tmInfo(Rest, C), imMode(M)) *Int tmBox(B, 0)
                        *Int tmEb(B)) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M andBool isI2c(M)
  // im2col_no_offs::w stores / reductions: pixels from the first one outside the tensor on are not written
  syntax Int ::= tmBad(Int, List, Int, Int) [function, total]            // first pixel whose W position is outside the tensor
  rule tmBad(B, Cs, Q, Cnt) => Q requires Q >=Int Cnt orBool nth(Cs, 1) +Int Q *Int tmEst(B, 1) <Int 0 orBool nth(Cs, 1) +Int Q *Int tmEst(B, 1) >=Int tmDim(B, 1)
  rule tmBad(B, Cs, Q, Cnt) => tmBad(B, Cs, Q +Int 1, Cnt) [owise]
  syntax List ::= tmStoreListI(Bytes, Int, List, Int, Int, Int, Int) [function, total]     // ..., first bad pixel, mode (0 .im2col_no_offs, 3 ::w)
  rule tmStoreListI(SM, B, Cs, Base, E, Bad, Md) =>
       #if E /Int tmBox(B, 0) >=Int Bad orBool tmAddrI(B, Cs, ListItem(0) ListItem(0) ListItem(0), Md, E) <Int 0 #then .List
       #else ListItem(tmg(tmAddrI(B, Cs, ListItem(0) ListItem(0) ListItem(0), Md, E), tmEb(B), tmSrcV(SM, B, Base, E))) #fi
       tmStoreListI(SM, B, Cs, Base, E +Int 1, Bad, Md)
    requires E <Int tmBox(B, 1) *Int tmBox(B, 0)
  rule tmStoreListI(_, _, _, _, _, _, _) => .List [owise]
  rule <tk> #tmSt(M, B, Cs, S, C) =>
            #tmGWr(tmStoreListI(SM, B, tmCoords(Cs, C), locOff(ea(S, "shared", C)), 0,
                                #if "im2col_no_offs::w" in M #then tmBad(B, tmCoords(Cs, C), 0, tmBox(B, 1)) #else tmBox(B, 1) #fi,
                                #if "im2col_no_offs::w" in M #then 3 #else 0 #fi),
                   #if "reduce" in M #then atomOp(M) #else "" #fi, tmTy(B)) ... </tk>
       <smem> SM </smem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M andBool ("im2col_no_offs::w" in M orBool "im2col_no_offs" in M)

  // ---------------------------------------------------------------- stores / reductions: shared -> global
  // source element e of the shared image (dense, chunk-swizzled)
  syntax Int ::= tmSrcV(Bytes, Int, Int, Int) [function, total]                       // smem, blob, base, e
  rule tmSrcV(SM, B, Base, E) => readBytes(SM, swizzle(tmEffMode(B), Base +Int E *Int tmEb(B)), tmEb(B))
  syntax List ::= tmStoreList(Bytes, Int, List, Int, Int, Bool) [function, total]
  rule tmStoreList(SM, B, Cs, Base, E, G) =>
       #if tmAddr(B, Cs, E, G) <Int 0 #then .List #else ListItem(tmg(tmAddr(B, Cs, E, G), tmEb(B), tmSrcV(SM, B, Base, E))) #fi
       tmStoreList(SM, B, Cs, Base, E +Int 1, G)
    requires E <Int tmCount(B, G)
  rule tmStoreList(_, _, _, _, _, _) => .List [owise]
  rule <tk> #tmSt(M, B, Cs, S, C) =>
            #tmGWr(tmStoreList(SM, B, tmCoords(Cs, C), locOff(ea(S, "shared", C)), 0,
                               "scatter4" in M orBool "tile::scatter4" in M),
                   #if "reduce" in M #then atomOp(M) #else "" #fi, tmTy(B)) ... </tk>
       <gmem> GM </gmem> <smem> SM </smem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M andBool notBool ("im2col_no_offs::w" in M orBool "im2col_no_offs" in M)
  rule <tk> #tmGWr(ListItem(tmg(A, N, V)) Ls, "", T) => #mem(loc("global", A), wrOp(N, V)) ~> #drop ~> #tmGWr(Ls, "", T) ... </tk>
  rule <tk> #tmGWr(ListItem(tmg(A, N, V)) Ls, OP, T) => #mem(loc("global", A), bulkRed(OP, T, N, V)) ~> #drop ~> #tmGWr(Ls, OP, T) ... </tk>
    requires OP =/=String ""
  rule <tk> #tmGWr(.List, _, _) => .K ... </tk>

  // ---------------------------------------------------------------- tensormap.replace / cp_fenceproxy
  // tensormap.replace.tile.<field>[.global|.shared::cta].b1024.<type> [addr], {ord,} new_val
  syntax KItem ::= #tmRep(Loc, String, Int, Int)
  rule <tk> exec(opc("tensormap", M, _), ops([ A ], N, .Operands), C) =>
            #mem(ea(A, #if "shared::cta" in M #then "shared" #else "generic" #fi, C), rdOp(128)) ~> #tmRep(ea(A, #if "shared::cta" in M #then "shared" #else "generic" #fi, C), tmField(M), 0, rd(N, ty("u", 64), C)) ... </tk>
    requires "replace" in M andBool notBool tmField(M) in (SetItem("box_dim") SetItem("global_dim") SetItem("global_stride") SetItem("element_stride"))
  rule <tk> exec(opc("tensormap", M, _), ops([ A ], Ord, N, .Operands), C) =>
            #mem(ea(A, #if "shared::cta" in M #then "shared" #else "generic" #fi, C), rdOp(128)) ~> #tmRep(ea(A, #if "shared::cta" in M #then "shared" #else "generic" #fi, C), tmField(M), rd(Ord, ty("u", 32), C), rd(N, ty("u", 64), C)) ... </tk>
    requires "replace" in M andBool tmField(M) in (SetItem("box_dim") SetItem("global_dim") SetItem("global_stride") SetItem("element_stride"))
  syntax String ::= tmField(Set) [function, total]
  rule tmField(M) => pick(M, ListItem("global_address") ListItem("rank") ListItem("box_dim") ListItem("global_dim") ListItem("global_stride")
                              ListItem("element_stride") ListItem("elemtype") ListItem("interleave_layout") ListItem("swizzle_mode")
                              ListItem("swizzle_atomicity") ListItem("fill_mode"), "")
  rule <tk> #res(B) ~> #tmRep(L, F, Ord, V) => #mem(L, wrOp(128, tmReplace(B, F, Ord, V))) ~> #drop ... </tk>
  syntax Int ::= tmReplace(Int, String, Int, Int) [function, total]
  rule tmReplace(B, "global_address", _, V) => setWord(B, 3, V)
  rule tmReplace(B, "rank", _, V) => setWord(B, 2, (tmW(B, 2) &Int (~Int 255)) |Int (V +Int 1))                          // the PTX operand holds rank - 1
  rule tmReplace(B, "elemtype", _, V) => setWord(B, 1, (tmW(B, 1) &Int (~Int 255)) |Int (V &Int 255))
  rule tmReplace(B, "global_dim", O, V) => setWord(B, 4 +Int O, V)
  rule tmReplace(B, "global_stride", O, V) => setWord(B, 9 +Int O, V)                 // stride of dim O+1, in bytes
  rule tmReplace(B, "box_dim", O, V) => setBox(B, O, V)
  rule tmReplace(B, "element_stride", O, V) => setWord(B, 15, (tmW(B, 15) &Int (~Int (255 <<Int (8 *Int O)))) |Int (V <<Int (8 *Int O)))
  rule tmReplace(B, "swizzle_mode", _, V) => setWord(B, 14, (tmW(B, 14) &Int (~Int (255 <<Int 16))) |Int (V <<Int 16))
  rule tmReplace(B, "swizzle_atomicity", _, V) => setWord(B, 14, (tmW(B, 14) &Int (~Int (255 <<Int 48))) |Int (V <<Int 48))
  rule tmReplace(B, "interleave_layout", _, V) => setWord(B, 14, (tmW(B, 14) &Int (~Int (255 <<Int 32))) |Int (V <<Int 32))
  rule tmReplace(B, "fill_mode", _, V) => setWord(B, 14, (tmW(B, 14) &Int (~Int (255 <<Int 24))) |Int (V <<Int 24))
  rule tmReplace(B, _, _, _) => B [owise]
  syntax Int ::= setWord(Int, Int, Int) [function, total] | setBox(Int, Int, Int) [function, total]
  rule setWord(B, I, V) => (B &Int (~Int (18446744073709551615 <<Int (64 *Int I)))) |Int ((V &Int 18446744073709551615) <<Int (64 *Int I))
  rule setBox(B, O, V) => setWord(B, 13, (tmW(B, 13) &Int (~Int (65535 <<Int (16 *Int O)))) |Int ((V &Int 65535) <<Int (16 *Int O))) requires O <Int 4
  rule setBox(B, _, V) => setWord(B, 14, (tmW(B, 14) &Int (~Int 65535)) |Int (V &Int 65535)) [owise]

  // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.<scope>.sync.aligned [dst], [src], 128
  rule <tk> exec(opc("tensormap", M, _), ops([ D ], [ S ], _, .Operands), C) =>
            #mem(ea(S, "shared", C), rdOp(128)) ~> #tmCp(ea(D, "generic", C)) ... </tk>
    requires "cp_fenceproxy" in M
  syntax KItem ::= #tmCp(Loc)
  rule <tk> #res(V) ~> #tmCp(L) => #mem(L, wrOp(128, V)) ~> #drop ... </tk>
  rule <tk> exec(opc("cp", M, _), ops(_), _) => .K ... </tk>
    requires "async" in M andBool "bulk" in M andBool "prefetch" in M andBool "tensor" in M
  rule isCollective("tensormap") => true
endmodule

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
// Not modelled: interleaved layouts, im2col modes, the .override forms of the instructions and
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
  rule tmRank(B) => tmW(B, 2)
  rule tmAddrOf(B) => tmW(B, 3)
  rule tmDim(B, I) => tmW(B, 4 +Int I)
  rule tmStride(B, I) => tmW(B, 8 +Int I)
  rule tmBox(B, I) => (tmW(B, 13) >>Int (16 *Int I)) &Int 65535 requires I <Int 4
  rule tmBox(B, _) => tmW(B, 14) &Int 65535 [owise]
  rule tmEst(B, I) => (tmW(B, 15) >>Int (8 *Int I)) &Int 255
  rule tmSwz(B) => (tmW(B, 14) >>Int 16) &Int 255
  rule tmFillMode(B) => (tmW(B, 14) >>Int 24) &Int 255
  // element size / type from the CUtensorMapDataType code
  rule tmEb(B) => tmBytesOf(tmW(B, 1))
  syntax Int ::= tmBytesOf(Int) [function, total]
  rule tmBytesOf(0) => 1  rule tmBytesOf(1) => 2  rule tmBytesOf(2) => 4  rule tmBytesOf(3) => 4  rule tmBytesOf(4) => 8  rule tmBytesOf(5) => 8
  rule tmBytesOf(6) => 2  rule tmBytesOf(7) => 4  rule tmBytesOf(8) => 8  rule tmBytesOf(9) => 2  rule tmBytesOf(_) => 4 [owise]
  syntax Ty ::= tmTy(Int) [function, total]
  rule tmTy(B) => tmTyOf(tmW(B, 1))
  syntax Ty ::= tmTyOf(Int) [function, total]
  rule tmTyOf(0) => ty("u", 8)  rule tmTyOf(1) => ty("u", 16)  rule tmTyOf(2) => ty("u", 32)  rule tmTyOf(3) => ty("s", 32)
  rule tmTyOf(4) => ty("u", 64)  rule tmTyOf(5) => ty("s", 64)  rule tmTyOf(6) => ty("f", 16)  rule tmTyOf(7) => ty("f", 32)
  rule tmTyOf(8) => ty("f", 64)  rule tmTyOf(9) => ty("bf", 16)  rule tmTyOf(_) => ty("f", 32) [owise]
  // swizzle code of the tensor map -> mode of ptx-exec-wg.k's swizzle (1 = 128B, 2 = 64B, 3 = 32B)
  syntax Int ::= tmMode(Int) [function, total]
  rule tmMode(3) => 1  rule tmMode(2) => 2  rule tmMode(1) => 3  rule tmMode(_) => 0 [owise]
  syntax Int ::= tmFill(Int) [function, total]
  rule tmFill(B) => 0 requires tmFillMode(B) ==Int 0
  rule tmFill(B) => 2143289344 requires tmFillMode(B) =/=Int 0 andBool tmEb(B) ==Int 4              // 0x7fc00000
  rule tmFill(B) => 32256 requires tmFillMode(B) =/=Int 0 andBool tmEb(B) ==Int 2                   // 0x7e00
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

  // ---------------------------------------------------------------- loads: global -> shared image chunks
  syntax KItem ::= #tmWr(String, List) | #tmGWr(List, String, Ty)
  syntax TmW ::= tmw(Int, Int)                         // shared/cluster address, 16-byte value
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
       ListItem(tmw(swizzle(tmMode(tmSwz(B)), Base +Int 16 *Int K), tmChunkV(GM, B, Cs, K *Int (16 /Int tmEb(B)), 0, G)))
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

  rule <tk> exec(opc("cp", M, _), ops([ D ], [ T , Cs ], [ Mb ], Rest:Operands), C) =>
            #tmWr(#if "shared::cluster" in M #then "cluster" #else "shared" #fi,
                  tmLoadAll(GM, readBytes(GM, locOff(ea(T, "generic", C)), 128), tmCoords(Cs, C),
                            tmBases(ea(D, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) opsList(Rest), C), isMc(M)),
                            "gather4" in M orBool "tile::gather4" in M))
            ~> #tmTxAll(bulkLocs(ea(Mb, #if "shared::cluster" in M #then "shared_cluster" #else "shared" #fi, C), bulkMask(ListItem([ Mb ]) opsList(Rest), C), isMc(M)),
                        tmCount(readBytes(GM, locOff(ea(T, "generic", C)), 128), "tile::gather4" in M) *Int tmEb(readBytes(GM, locOff(ea(T, "generic", C)), 128))) ... </tk>
       <gmem> GM </gmem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "mbarrier::complete_tx::bytes" in M
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

  // ---------------------------------------------------------------- stores / reductions: shared -> global
  // source element e of the shared image (dense, chunk-swizzled)
  syntax Int ::= tmSrcV(Bytes, Int, Int, Int) [function, total]                       // smem, blob, base, e
  rule tmSrcV(SM, B, Base, E) =>
       readBytes(SM, swizzle(tmMode(tmSwz(B)), Base +Int ((E *Int tmEb(B)) /Int 16) *Int 16) +Int (E *Int tmEb(B)) %Int 16, tmEb(B))
  syntax List ::= tmStoreList(Bytes, Int, List, Int, Int, Bool) [function, total]
  rule tmStoreList(SM, B, Cs, Base, E, G) =>
       #if tmAddr(B, Cs, E, G) <Int 0 #then .List #else ListItem(tmg(tmAddr(B, Cs, E, G), tmEb(B), tmSrcV(SM, B, Base, E))) #fi
       tmStoreList(SM, B, Cs, Base, E +Int 1, G)
    requires E <Int tmCount(B, G)
  rule tmStoreList(_, _, _, _, _, _) => .List [owise]
  rule <tk> exec(opc("cp", M, _), ops([ T , Cs ], [ S ], _:Operands), C) =>
            #tmGWr(tmStoreList(SM, readBytes(GM, locOff(ea(T, "generic", C)), 128), tmCoords(Cs, C), locOff(ea(S, "shared", C)), 0,
                               "scatter4" in M orBool "tile::scatter4" in M),
                   #if "reduce" in M #then atomOp(M) #else "" #fi, tmTy(readBytes(GM, locOff(ea(T, "generic", C)), 128))) ... </tk>
       <gmem> GM </gmem> <smem> SM </smem>
    requires "async" in M andBool "bulk" in M andBool "tensor" in M andBool "bulk_group" in M
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
  rule tmReplace(B, "rank", _, V) => setWord(B, 2, V +Int 1)                          // the PTX operand holds rank - 1
  rule tmReplace(B, "elemtype", _, V) => setWord(B, 1, V)
  rule tmReplace(B, "global_dim", O, V) => setWord(B, 4 +Int O, V)
  rule tmReplace(B, "global_stride", O, V) => setWord(B, 9 +Int O, V)                 // stride of dim O+1, in bytes
  rule tmReplace(B, "box_dim", O, V) => setBox(B, O, V)
  rule tmReplace(B, "element_stride", O, V) => setWord(B, 15, (tmW(B, 15) &Int (~Int (255 <<Int (8 *Int O)))) |Int (V <<Int (8 *Int O)))
  rule tmReplace(B, "swizzle_mode", _, V) => setWord(B, 14, (tmW(B, 14) &Int (~Int (255 <<Int 16))) |Int (V <<Int 16))
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

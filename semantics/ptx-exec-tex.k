// =============================================================================
// PTX-EXEC-TEX: texture and surface instructions
//
//   tex.{1d,2d,3d,a1d,a2d}.v4.dtype.ctype d, [tex, {coords}]  /  [tex, sampler, {coords}]
//   tld4.{r,g,b,a}.2d.v4.dtype.f32 d, [tex, {x, y}]            txq.<field>.b32 d, [tex]
//   suld.b / sust.b / sured.b {1d,2d,3d,a1d,a2d}               suq.<field>.b32 d, [surf]
//
// Texture and surface objects are the 64-bit handles of descriptor objects that the harness builds in
// global memory (.texture / .sampler / .surface, see ptx-harness.k); module-scope .texref / .surfref
// variables and the formatted surface accesses (.p) are not modelled, nor are cubemaps, mipmap
// levels, depth compares, 2dms textures and the residency predicate.
//
// Filtering follows the CUDA programming guide: coordinates scale by the extent when normalized, linear
// filtering uses weights quantised to 1/256 (the fixed-point format of the hardware, rounding to nearest),
// products and sums are exact and rounded once to the destination format.  Address modes: wrap and mirror
// need normalized coordinates (unnormalized coordinates clamp), border yields zeros.  Missing channels read
// as 0 (alpha as 1); read-normalized integer textures convert to [0,1] / [-1,1].
// =============================================================================
requires "ptx-exec-misc.k"
requires "ptx-exec-ldm.k"

module PTX-EXEC-TEX
  imports PTX-EXEC-MISC
  imports PTX-EXEC-LDM

  // ---------------------------------------------------------------- descriptor fields
  syntax Int ::= hAddr(Addr, Ctx) [function, total]       // value of a register used as a handle
  rule hAddr(A, C) => locOff(ea(A, "global", C))
  syntax Int ::= tdW(Int, Int) [function, total]            // word I of a 128-byte descriptor
  rule tdW(B, I) => (B >>Int (64 *Int I)) &Int 18446744073709551615
  syntax Int ::= txBase(Int) [function, total] | txCode(Int) [function, total] | txCh(Int) [function, total] | txFilter(Int) [function, total]
               | txAddr(Int) [function, total] | txNorm(Int) [function, total] | txReadNorm(Int) [function, total] | txLayers(Int) [function, total]
               | txDim(Int, Int) [function, total]
  rule txBase(B) => tdW(B, 1)    rule txCode(B) => tdW(B, 2)    rule txCh(B) => tdW(B, 3)     rule txFilter(B) => tdW(B, 4)
  rule txAddr(B) => tdW(B, 5)    rule txNorm(B) => tdW(B, 6)    rule txReadNorm(B) => tdW(B, 7)    rule txLayers(B) => maxInt(1, tdW(B, 8))
  rule txDim(B, I) => maxInt(1, tdW(B, 9 +Int I))
  syntax Int ::= txEb(Int) [function, total]                // element bytes from the type code
  rule txEb(B) => txEbOf(txCode(B))
  syntax Int ::= txEbOf(Int) [function, total]
  rule txEbOf(0) => 1  rule txEbOf(20) => 1  rule txEbOf(1) => 2  rule txEbOf(21) => 2  rule txEbOf(6) => 2  rule txEbOf(_) => 4 [owise]
  // a sampler overrides filter, address mode and normalization of the texture
  syntax Int ::= effFilter(Int, Int) [function, total] | effAddr(Int, Int) [function, total] | effNorm(Int, Int) [function, total]
  rule effFilter(B, S) => tdW(S, 1) requires S >=Int 0
  rule effFilter(B, _) => txFilter(B) [owise]
  rule effAddr(B, S) => tdW(S, 2) requires S >=Int 0
  rule effAddr(B, _) => txAddr(B) [owise]
  rule effNorm(B, S) => tdW(S, 3) requires S >=Int 0
  rule effNorm(B, _) => txNorm(B) [owise]

  // ---------------------------------------------------------------- texels
  syntax Float ::= zF() [function, total] | oneF() [function, total]
  rule zF() => Int2Float(0, WP, WEXP)
  rule oneF() => Int2Float(1, WP, WEXP)
  // index with the address mode applied (-1: border / outside)
  syntax Int ::= adIdx(Int, Int, Int) [function, total]     // mode, index, extent
  rule adIdx(1, I, N) => minInt(maxInt(I, 0), N -Int 1)
  rule adIdx(0, I, N) => ((I %Int N) +Int N) %Int N
  rule adIdx(3, I, N) => mirrorIdx(((I %Int (2 *Int N)) +Int 2 *Int N) %Int (2 *Int N), N)
  rule adIdx(2, I, N) => #if I >=Int 0 andBool I <Int N #then I #else 0 -Int 1 #fi
  rule adIdx(_, I, N) => minInt(maxInt(I, 0), N -Int 1) [owise]
  syntax Int ::= mirrorIdx(Int, Int) [function, total]
  rule mirrorIdx(M, N) => #if M <Int N #then M #else 2 *Int N -Int M -Int 1 #fi
  // raw channel c of the texel at integer coordinates (already resolved), as an exact Float
  syntax Float ::= chanF(Bytes, Int, Int, Int) [function, total]            // gmem, blob, address of the texel, channel
  rule chanF(GM, B, A, C) => convF(txCode(B), readBytes(GM, A +Int C *Int txEb(B), txEb(B)), txReadNorm(B))
  syntax Float ::= convF(Int, Int, Int) [function, total]                   // type code, bits, read-normalized?
  rule convF(6, V, _) => roundFloat(decodeF(V, 16), WP, WEXP)
  rule convF(7, V, _) => roundFloat(decodeF(V, 32), WP, WEXP)
  rule convF(C, V, N) => convI(C, V, N) requires C =/=Int 6 andBool C =/=Int 7
  // integer formats: the value itself, or normalized to [0,1] (unsigned) / [-1,1] (signed)
  syntax Float ::= convI(Int, Int, Int) [function, total]
  rule convI(C, V, N) => Int2Float(codeVal(C, V), WP, WEXP) requires N ==Int 0
  rule convI(C, V, N) => maxF(Int2Float(codeVal(C, V), WP, WEXP) /Float Int2Float(codeMax(C), WP, WEXP), Int2Float(0 -Int 1, WP, WEXP)) requires N =/=Int 0
  syntax Int ::= codeVal(Int, Int) [function, total] | codeMax(Int) [function, total] | codeBits(Int) [function, total]
  rule codeBits(C) => 8 requires C ==Int 0 orBool C ==Int 20
  rule codeBits(C) => 16 requires C ==Int 1 orBool C ==Int 21
  rule codeBits(C) => 32 requires C =/=Int 0 andBool C =/=Int 20 andBool C =/=Int 1 andBool C =/=Int 21
  rule codeVal(C, V) => sext(V, codeBits(C)) requires C ==Int 20 orBool C ==Int 21 orBool C ==Int 3 orBool C ==Int 5
  rule codeVal(C, V) => V requires C =/=Int 20 andBool C =/=Int 21 andBool C =/=Int 3 andBool C =/=Int 5
  rule codeMax(C) => (1 <<Int codeBits(C)) -Int 1 requires C ==Int 0 orBool C ==Int 1 orBool C ==Int 2 orBool C ==Int 4
  rule codeMax(C) => (1 <<Int (codeBits(C) -Int 1)) -Int 1 requires C =/=Int 0 andBool C =/=Int 1 andBool C =/=Int 2 andBool C =/=Int 4
  syntax Float ::= maxF(Float, Float) [function, total]
  rule maxF(A, B) => #if A >=Float B #then A #else B #fi

  // the RGBA texel at resolved integer coordinates (a negative coordinate means border: zeros)
  syntax List ::= texel(Bytes, Int, Int, Int, Int, Int) [function, total]   // gmem, blob, i, j, k, layer
  rule texel(GM, B, I, J, K, L) => ListItem(zF()) ListItem(zF()) ListItem(zF()) ListItem(zF()) requires I <Int 0 orBool J <Int 0 orBool K <Int 0
  rule texel(GM, B, I, J, K, L) => texelAt(GM, B, txBase(B) +Int (((L *Int txDim(B, 2) +Int K) *Int txDim(B, 1) +Int J) *Int txDim(B, 0) +Int I) *Int txCh(B) *Int txEb(B)) [owise]
  syntax List ::= texelAt(Bytes, Int, Int) [function, total]
  rule texelAt(GM, B, A) => ListItem(texC(GM, B, A, 0)) ListItem(texC(GM, B, A, 1)) ListItem(texC(GM, B, A, 2)) ListItem(texC(GM, B, A, 3))
  syntax Float ::= texC(Bytes, Int, Int, Int) [function, total]
  rule texC(GM, B, A, C) => chanF(GM, B, A, C) requires C <Int txCh(B)
  rule texC(_, _, _, C) => oneF() requires C ==Int 3
  rule texC(_, _, _, _) => zF() [owise]

  // weighted combination of two texels
  syntax List ::= lerp4(List, List, Float) [function, total]
  rule lerp4(A, B, W) => ListItem(lerpF(nthF(A, 0), nthF(B, 0), W)) ListItem(lerpF(nthF(A, 1), nthF(B, 1), W))
                         ListItem(lerpF(nthF(A, 2), nthF(B, 2), W)) ListItem(lerpF(nthF(A, 3), nthF(B, 3), W))
  syntax Float ::= lerpF(Float, Float, Float) [function, total] | nthF(List, Int) [function, total]
  rule lerpF(A, B, W) => (oneF() -Float W) *Float A +Float W *Float B
  rule nthF(L, I) => {L[I]}:>Float requires I >=Int 0 andBool I <Int size(L)
  rule nthF(_, _) => zF() [owise]

  // ---------------------------------------------------------------- coordinates
  // unnormalized position of a coordinate along a dimension
  syntax Float ::= pos(Float, Int, Int) [function, total]           // coordinate, extent, normalized?
  rule pos(X, N, 1) => X *Float Int2Float(N, WP, WEXP)
  rule pos(X, _, _) => X [owise]
  syntax Int ::= flo(Float) [function, total]
  rule flo(X) => Float2Int(floorFloat(X))
  // weight of the upper texel quantised to 1/256
  syntax Float ::= qw(Float) [function, total]
  rule qw(Xb) => Int2Float(flo((Xb -Float floorFloat(Xb)) *Float Int2Float(256, WP, WEXP) +Float Int2Float(1, WP, WEXP) /Float Int2Float(2, WP, WEXP)), WP, WEXP) /Float Int2Float(256, WP, WEXP)
  syntax Float ::= half() [function, total]
  rule half() => Int2Float(1, WP, WEXP) /Float Int2Float(2, WP, WEXP)

  // sample at (x, y, z) floats with layer l
  syntax List ::= sample(Bytes, Int, Int, Int, Int, Float, Float, Float, Int, Int) [function, total]   // gmem, blob, filter, addrmode, norm, x,y,z, layer, rank
  rule sample(GM, B, 0, AM, NM, X, Y, Z, L, R) =>
       texel(GM, B, adIdx(AM, flo(pos(X, txDim(B, 0), NM)), txDim(B, 0)),
                    #if R >=Int 2 #then adIdx(AM, flo(pos(Y, txDim(B, 1), NM)), txDim(B, 1)) #else 0 #fi,
                    #if R >=Int 3 #then adIdx(AM, flo(pos(Z, txDim(B, 2), NM)), txDim(B, 2)) #else 0 #fi, L)
  rule sample(GM, B, 1, AM, NM, X, Y, Z, L, R) =>
       lerp4(lerp4(lerp4(texel(GM, B, ix(B, AM, NM, X, 0), jy(B, AM, NM, Y, 0, R), kz(B, AM, NM, Z, 0, R), L),
                         texel(GM, B, ix(B, AM, NM, X, 1), jy(B, AM, NM, Y, 0, R), kz(B, AM, NM, Z, 0, R), L), wx(B, NM, X)),
                   lerp4(texel(GM, B, ix(B, AM, NM, X, 0), jy(B, AM, NM, Y, 1, R), kz(B, AM, NM, Z, 0, R), L),
                         texel(GM, B, ix(B, AM, NM, X, 1), jy(B, AM, NM, Y, 1, R), kz(B, AM, NM, Z, 0, R), L), wx(B, NM, X)), wy(B, NM, Y, R)),
             lerp4(lerp4(texel(GM, B, ix(B, AM, NM, X, 0), jy(B, AM, NM, Y, 0, R), kz(B, AM, NM, Z, 1, R), L),
                         texel(GM, B, ix(B, AM, NM, X, 1), jy(B, AM, NM, Y, 0, R), kz(B, AM, NM, Z, 1, R), L), wx(B, NM, X)),
                   lerp4(texel(GM, B, ix(B, AM, NM, X, 0), jy(B, AM, NM, Y, 1, R), kz(B, AM, NM, Z, 1, R), L),
                         texel(GM, B, ix(B, AM, NM, X, 1), jy(B, AM, NM, Y, 1, R), kz(B, AM, NM, Z, 1, R), L), wx(B, NM, X)), wy(B, NM, Y, R)),
             wz(B, NM, Z, R))
  rule sample(GM, B, _, AM, NM, X, Y, Z, L, R) => sample(GM, B, 0, AM, NM, X, Y, Z, L, R) [owise]
  // linear-filter helpers: lower/upper texel index, and weights
  syntax Int ::= ix(Int, Int, Int, Float, Int) [function, total] | jy(Int, Int, Int, Float, Int, Int) [function, total] | kz(Int, Int, Int, Float, Int, Int) [function, total]
  rule ix(B, AM, NM, X, D) => adIdx(AM, flo(pos(X, txDim(B, 0), NM) -Float half()) +Int D, txDim(B, 0))
  rule jy(B, AM, NM, Y, D, R) => #if R >=Int 2 #then adIdx(AM, flo(pos(Y, txDim(B, 1), NM) -Float half()) +Int D, txDim(B, 1)) #else 0 #fi
  rule kz(B, AM, NM, Z, D, R) => #if R >=Int 3 #then adIdx(AM, flo(pos(Z, txDim(B, 2), NM) -Float half()) +Int D, txDim(B, 2)) #else 0 #fi
  syntax Float ::= wx(Int, Int, Float) [function, total] | wy(Int, Int, Float, Int) [function, total] | wz(Int, Int, Float, Int) [function, total]
  rule wx(B, NM, X) => qw(pos(X, txDim(B, 0), NM) -Float half())
  rule wy(B, NM, Y, R) => #if R >=Int 2 #then qw(pos(Y, txDim(B, 1), NM) -Float half()) #else zF() #fi
  rule wz(B, NM, Z, R) => #if R >=Int 3 #then qw(pos(Z, txDim(B, 2), NM) -Float half()) #else zF() #fi

  // ---------------------------------------------------------------- result conversion and the instruction
  syntax Int ::= outBits(Ty, Float) [function, total]
  rule outBits(ty("f", 32), F) => fpFin(.Set, 32, F, 0)
  rule outBits(ty("s", 32), F) => trunc(Float2Int(roundFloat(F, WP, WEXP)), 32)
  rule outBits(ty("u", 32), F) => trunc(Float2Int(roundFloat(F, WP, WEXP)), 32)
  rule outBits(_, F) => fpFin(.Set, 32, F, 0) [owise]
  syntax List ::= outList(Ty, List) [function, total]
  rule outList(T, ListItem(F:Float) Fs) => ListItem(outBits(T, F)) outList(T, Fs)
  rule outList(_, _) => .List [owise]

  syntax Int ::= rankOf(Set) [function, total]
  rule rankOf(M) => 1 requires "1d" in M orBool "a1d" in M
  rule rankOf(M) => 3 requires "3d" in M
  rule rankOf(_) => 2 [owise]
  syntax Bool ::= isArr(Set) [function, total]
  rule isArr(M) => "a1d" in M orBool "a2d" in M

  // coordinates as floats (ctype f32) or exact integers (ctype s32 -> point sampling of texel indices)
  syntax Float ::= coordF(Operand, Ty, Ctx) [function, total]
  rule coordF(O, ty("f", 32), C) => roundFloat(decodeF(rd(O, ty("b", 32), C), 32), WP, WEXP)
  rule coordF(O, T, C) => Int2Float(ival(rd(O, ty("s", 32), C), ty("s", 32)), WP, WEXP) [owise]
  syntax Operand ::= vecAt(Operand, Int) [function, total]
  rule vecAt(O, I) => {vecOps(O)[I]}:>Operand requires I >=Int 0 andBool I <Int size(vecOps(O))
  rule vecAt(_, _) => 0 [owise]

  // tex.geom.v4.dtype.ctype {r,g,b,a}, [tex, {coords}]   or   [tex, sampler, {coords}]
  syntax KItem ::= #texOut(Operand, List)
  rule <tk> exec(opc("tex", M, ListItem(DT) ListItem(CT)), ops(Dv, [ T , Cs ], .Operands), C) =>
            #texOut(Dv, texRun(GM, readBytes(GM, hAddr(T, C), 128), 0 -Int 1, M, DT, CT, Cs, C)) ... </tk>
       <gmem> GM </gmem>
  rule <tk> exec(opc("tex", M, ListItem(DT) ListItem(CT)), ops(Dv, [ T , S , Cs ], .Operands), C) =>
            #texOut(Dv, texRun(GM, readBytes(GM, hAddr(T, C), 128), readBytes(GM, hAddr(S, C), 128), M, DT, CT, Cs, C)) ... </tk>
       <gmem> GM </gmem>
  rule <tk> #texOut(Dv, Vs) => #wmWr(vecOps(Dv), Vs) ... </tk>
  syntax List ::= texRun(Bytes, Int, Int, Set, Ty, Ty, Operand, Ctx) [function, total]
  rule texRun(GM, B, S, M, DT, CT, Cs, C) =>
       outList(DT, sample(GM, B,
                          #if CT ==K ty("f", 32) #then effFilter(B, S) #else 0 #fi, effAddr(B, S), effNorm(B, S),
                          coordF(vecAt(Cs, #if isArr(M) #then 1 #else 0 #fi), CT, C),
                          coordF(vecAt(Cs, #if isArr(M) #then 2 #else 1 #fi), CT, C),
                          coordF(vecAt(Cs, #if isArr(M) #then 3 #else 2 #fi), CT, C),
                          #if isArr(M) #then minInt(rd(vecAt(Cs, 0), ty("u", 32), C), txLayers(B) -Int 1) #else 0 #fi, rankOf(M)))

  // tld4.{r,g,b,a}.2d.v4.dtype.f32 d, [tex, {x, y}] : the component of the 2x2 footprint, clockwise from the lower-left texel
  rule <tk> exec(opc("tld4", M, ListItem(DT) ListItem(_)), ops(Dv, [ T , Cs ], .Operands), C) =>
            #texOut(Dv, outList(DT, gather4(GM, readBytes(GM, hAddr(T, C), 128), gcomp(M),
                                            coordF(vecAt(Cs, 0), ty("f", 32), C), coordF(vecAt(Cs, 1), ty("f", 32), C)))) ... </tk>
       <gmem> GM </gmem>
  syntax Int ::= gcomp(Set) [function, total]
  rule gcomp(M) => 1 requires "g" in M
  rule gcomp(M) => 2 requires "b" in M
  rule gcomp(M) => 3 requires "a" in M
  rule gcomp(_) => 0 [owise]
  syntax List ::= gather4(Bytes, Int, Int, Float, Float) [function, total]
  rule gather4(GM, B, Cc, X, Y) =>
       ListItem(nthF(texel(GM, B, ix(B, txAddr(B), txNorm(B), X, 0), jy(B, txAddr(B), txNorm(B), Y, 1, 2), 0, 0), Cc))
       ListItem(nthF(texel(GM, B, ix(B, txAddr(B), txNorm(B), X, 1), jy(B, txAddr(B), txNorm(B), Y, 1, 2), 0, 0), Cc))
       ListItem(nthF(texel(GM, B, ix(B, txAddr(B), txNorm(B), X, 1), jy(B, txAddr(B), txNorm(B), Y, 0, 2), 0, 0), Cc))
       ListItem(nthF(texel(GM, B, ix(B, txAddr(B), txNorm(B), X, 0), jy(B, txAddr(B), txNorm(B), Y, 0, 2), 0, 0), Cc))

  // txq.<field>.b32 d, [tex]
  rule <tk> exec(opc("txq", M, _), ops(D, [ T ], .Operands), C) => .K ... </tk>
       <gmem> GM </gmem> <regs> R => wr(R, D, txqVal(M, readBytes(GM, hAddr(T, C), 128))) </regs>
  syntax Int ::= txqVal(Set, Int) [function, total]
  rule txqVal(M, B) => txDim(B, 0) requires "width" in M
  rule txqVal(M, B) => txDim(B, 1) requires "height" in M
  rule txqVal(M, B) => txDim(B, 2) requires "depth" in M
  rule txqVal(M, B) => tdW(B, 8) requires "array_size" in M
  rule txqVal(M, B) => txNorm(B) requires "normalized_coords" in M
  rule txqVal(M, B) => txFilter(B) requires "filter_mode" in M
  rule txqVal(M, B) => txAddr(B) requires "addr_mode_0" in M orBool "addr_mode_1" in M orBool "addr_mode_2" in M
  rule txqVal(M, B) => 1 requires "num_mipmap_levels" in M
  rule txqVal(_, _) => 0 [owise]

  // ---------------------------------------------------------------- surfaces
  syntax Int ::= sfBase(Int) [function, total] | sfTb(Int) [function, total] | sfLayers(Int) [function, total] | sfDim(Int, Int) [function, total]
  rule sfBase(B) => tdW(B, 1)   rule sfTb(B) => maxInt(1, tdW(B, 2))   rule sfLayers(B) => maxInt(1, tdW(B, 3))   rule sfDim(B, I) => maxInt(1, tdW(B, 4 +Int I))
  // byte address of (x bytes, y, z, layer); -1 when outside
  syntax Int ::= sfAddr(Int, Int, Int, Int, Int, Int, Int) [function, total]       // blob, x, y, z, layer, access bytes, rank
  rule sfAddr(B, X, Y, Z, L, N, R) =>
       #if X <Int 0 orBool X +Int N >Int sfDim(B, 0) *Int sfTb(B) orBool Y <Int 0 orBool Y >=Int sfDim(B, 1) orBool Z <Int 0 orBool Z >=Int sfDim(B, 2) orBool L <Int 0 orBool L >=Int sfLayers(B)
       #then 0 -Int 1
       #else sfBase(B) +Int (((L *Int sfDim(B, 2) +Int Z) *Int sfDim(B, 1) +Int Y) *Int sfDim(B, 0)) *Int sfTb(B) +Int X #fi
  // clamp mode: coordinates are clamped into the surface
  syntax Int ::= sfClampX(Int, Int, Int) [function, total]
  rule sfClampX(B, X, N) => minInt(maxInt(X, 0), maxInt(sfDim(B, 0) *Int sfTb(B) -Int N, 0))
  syntax Int ::= sfCo(Operand, Int, Ctx) [function, total]
  rule sfCo(Cs, I, C) => ival(rd(vecAt(Cs, I), ty("s", 32), C), ty("s", 32))
  // the access: coordinates {l?, x, y?, z?} -> (x, y, z, layer)
  syntax List ::= sfCoords(Set, Operand, Ctx) [function, total]
  rule sfCoords(M, Cs, C) => #if isArr(M)
       #then ListItem(sfCo(Cs, 1, C)) ListItem(#if rankOf(M) >=Int 2 #then sfCo(Cs, 2, C) #else 0 #fi) ListItem(0) ListItem(rd(vecAt(Cs, 0), ty("u", 32), C))
       #else ListItem(sfCo(Cs, 0, C)) ListItem(#if rankOf(M) >=Int 2 #then sfCo(Cs, 1, C) #else 0 #fi) ListItem(#if rankOf(M) >=Int 3 #then sfCo(Cs, 2, C) #else 0 #fi) ListItem(0) #fi
  syntax Int ::= sfVec(Set) [function, total]
  rule sfVec(M) => 4 requires "v4" in M
  rule sfVec(M) => 2 requires "v2" in M
  rule sfVec(_) => 1 [owise]
  syntax Int ::= sfAcc(Set, Ty) [function, total]                                   // bytes accessed
  rule sfAcc(M, T) => sfVec(M) *Int bytes(T)
  syntax Int ::= sfA(Int, Set, Ty, List) [function, total]                          // blob -> resolved address (or -1) honouring clamp/zero/trap
  rule sfA(B, M, T, Xs) => sfAddr(B, #if "clamp" in M #then sfClampX(B, nth(Xs, 0), sfAcc(M, T)) #else nth(Xs, 0) #fi,
                                  #if "clamp" in M #then minInt(maxInt(nth(Xs, 1), 0), sfDim(B, 1) -Int 1) #else nth(Xs, 1) #fi,
                                  #if "clamp" in M #then minInt(maxInt(nth(Xs, 2), 0), sfDim(B, 2) -Int 1) #else nth(Xs, 2) #fi,
                                  #if "clamp" in M #then minInt(maxInt(nth(Xs, 3), 0), sfLayers(B) -Int 1) #else nth(Xs, 3) #fi, sfAcc(M, T), rankOf(M))

  syntax KItem ::= #sfLoad(Operand, Ty, Int, Int, Set) | #sfStore(Int, Ty, List, Set) | #sfUnpack(List, Ty, Int)
  // suld.b.geom.vN.btype.mode {d..}, [surf, {coords}]     (.trap faults out of bounds, .zero returns zeros, .clamp clamps)
  rule <tk> exec(opc("suld", M, ListItem(T)), ops(Dv, [ S , Cs ], .Operands), C) =>
            #sfLoad(Dv, T, sfVec(M), sfA(readBytes(GM, hAddr(S, C), 128), M, T, sfCoords(M, Cs, C)), M) ... </tk>
       <gmem> GM </gmem>
    requires "b" in M
  rule <tk> #sfLoad(Dv, T, N, A, _) => #mem(loc("global", A), rdOp(N *Int bytes(T))) ~> #sfUnpack(vecOps(Dv), T, N) ... </tk> requires A >=Int 0
  rule <tk> #sfLoad(Dv, _, N, A, M) => #wmWr(vecOps(Dv), zeros(N)) ... </tk> requires A <Int 0 andBool "zero" in M
  rule <tk> #sfLoad(_, _, _, A, _) => #fault("surface access out of bounds", A) ... </tk> [owise]
  rule <tk> #res(V) ~> #sfUnpack(Ds, T, N) => #wmWr(Ds, split(V, T, N, 0)) ... </tk>
  syntax List ::= split(Int, Ty, Int, Int) [function, total] | zeros(Int) [function, total]
  rule split(V, T, N, I) => ListItem((V >>Int (8 *Int bytes(T) *Int I)) &Int mask(8 *Int bytes(T))) split(V, T, N, I +Int 1) requires I <Int N
  rule split(_, _, _, _) => .List [owise]
  rule zeros(N) => ListItem(0) zeros(N -Int 1) requires N >Int 0
  rule zeros(_) => .List [owise]

  // sust.b.geom.vN.btype.mode [surf, {coords}], {r..}
  rule <tk> exec(opc("sust", M, ListItem(T)), ops([ S , Cs ], Vs, .Operands), C) =>
            #sfStore(sfA(readBytes(GM, hAddr(S, C), 128), M, T, sfCoords(M, Cs, C)), T, rdList(opsList(vecItems(Vs)), T, C), M) ... </tk>
       <gmem> GM </gmem>
    requires "b" in M
  rule <tk> #sfStore(A, T, Vs, _) => #mem(loc("global", A), wrOp(size(Vs) *Int bytes(T), packList(Vs, T, 0))) ~> #drop ... </tk> requires A >=Int 0
  rule <tk> #sfStore(A, _, _, M) => .K ... </tk> requires A <Int 0 andBool "zero" in M
  rule <tk> #sfStore(A, _, _, _) => #fault("surface access out of bounds", A) ... </tk> [owise]
  syntax Int ::= packList(List, Ty, Int) [function, total]
  rule packList(ListItem(V:Int) Vs, T, I) => ((V &Int mask(8 *Int bytes(T))) <<Int (8 *Int bytes(T) *Int I)) |Int packList(Vs, T, I +Int 1)
  rule packList(_, _, _) => 0 [owise]

  // sured.b.op.geom.type [surf, {coords}], v
  rule <tk> exec(opc("sured", M, ListItem(T)), ops([ S , Cs ], V, .Operands), C) =>
            #sfRed(sfA(readBytes(GM, hAddr(S, C), 128), M, T, sfCoords(M, Cs, C)), atomOp(M), T, rd(V, T, C), M) ... </tk>
       <gmem> GM </gmem>
  syntax KItem ::= #sfRed(Int, String, Ty, Int, Set)
  rule <tk> #sfRed(A, OP, T, V, _) => #mem(loc("global", A), atOp(OP, T, V, 0)) ~> #drop ... </tk> requires A >=Int 0
  rule <tk> #sfRed(A, _, _, _, M) => .K ... </tk> requires A <Int 0 andBool "zero" in M
  rule <tk> #sfRed(A, _, _, _, _) => #fault("surface access out of bounds", A) ... </tk> [owise]

  // suq.<field>.b32 d, [surf]
  rule <tk> exec(opc("suq", M, _), ops(D, [ S ], .Operands), C) => .K ... </tk>
       <gmem> GM </gmem> <regs> R => wr(R, D, suqVal(M, readBytes(GM, hAddr(S, C), 128))) </regs>
  syntax Int ::= suqVal(Set, Int) [function, total]
  rule suqVal(M, B) => sfDim(B, 0) requires "width" in M
  rule suqVal(M, B) => sfDim(B, 1) requires "height" in M
  rule suqVal(M, B) => sfDim(B, 2) requires "depth" in M
  rule suqVal(M, B) => tdW(B, 3) requires "array_size" in M
  rule suqVal(_, _) => 0 [owise]

  rule isCollective("tex") => true   rule isCollective("tld4") => true   rule isCollective("txq") => true
  rule isCollective("suld") => true  rule isCollective("sust") => true   rule isCollective("sured") => true   rule isCollective("suq") => true

  // istypep.{texref,samplerref,surfref} p, a : is a a handle of the given object kind?  (descriptor objects carry a magic word)
  syntax Int ::= istpMagic(Set) [function, total]
  rule istpMagic(M) => 1415075924 requires "texref" in M
  rule istpMagic(M) => 1396920397 requires "samplerref" in M
  rule istpMagic(M) => 1398035526 requires "surfref" in M
  rule istpMagic(M) => 0 -Int 1 requires notBool ("texref" in M orBool "samplerref" in M orBool "surfref" in M)
  syntax KItem ::= #istp(Operand, Int)
  rule <tk> exec(opc("istypep", M, _), ops(D, A, .Operands), C) => #mem(loc("global", rd(A, ty("u", 64), C)), rdOp(8)) ~> #istp(D, istpMagic(M)) ... </tk>
       <gtop> G </gtop> requires rd(A, ty("u", 64), C) >=Int 4096 andBool rd(A, ty("u", 64), C) +Int 8 <=Int G
  rule <tk> exec(opc("istypep", _, _), ops(D, A, .Operands), C) => #res(0 -Int 3) ~> #istp(D, 0 -Int 2) ... </tk>
       <gtop> G </gtop> requires notBool (rd(A, ty("u", 64), C) >=Int 4096 andBool rd(A, ty("u", 64), C) +Int 8 <=Int G)
  rule <tk> #res(V) ~> #istp(D, Mg) => .K ... </tk> <regs> R => wr(R, D, b2i(V ==Int Mg)) </regs>
endmodule

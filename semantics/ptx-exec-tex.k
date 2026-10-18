// =============================================================================
// PTX-EXEC-TEX: texture and surface instructions
//
//   tex.{1d,2d,3d,a1d,a2d}.v4.dtype.ctype d, [tex, {coords}]  /  [tex, sampler, {coords}]
//   tld4.{r,g,b,a}.2d.v4.dtype.f32 d, [tex, {x, y}]            txq.<field>.b32 d, [tex]
//   suld.b / sust.b / sured.b {1d,2d,3d,a1d,a2d}               suq.<field>.b32 d, [surf]
//
// Texture and surface objects are the 64-bit handles of descriptor objects that the harness builds in
// global memory (.texture / .sampler / .surface, see ptx-harness.k); module-scope .texref / .surfref
// variables and the formatted surface accesses (.p) are not modelled, nor are 2dms textures.
// Mipmaps: levels are stored one after the other (each holds all layers, dims halve down to 1); lod comes from
// .base (0), .level (operand) or .grad (log2 of the larger scaled gradient); the sampler's mip filter picks the
// nearest level or blends two by a 1/256-quantised weight; lod is clamped to [minLOD, maxLOD, levels-1].
// Cubemaps are 6 consecutive layers per cube (+X -X +Y -Y +Z -Z), the face follows the OpenGL rules, filtering is
// clamped inside the face (no seam blending).  Depth compare (LEQUAL, ref <= texel) is applied per texel before
// filtering; offsets are added to the texel indices; the residency predicate is always true.
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
  // levels (word 12, default 1), flags (13: bit 0 depth, bit 1 cube), mip filter (15)
  syntax Int ::= txLevels(Int) [function, total] | txFlags(Int) [function, total] | txMipF(Int) [function, total]
               | lvD(Int, Int, Int) [function, total] | lvOff(Int, Int) [function, total] | lvTexels(Int, Int) [function, total]
  rule txLevels(B) => maxInt(1, tdW(B, 12))   rule txFlags(B) => tdW(B, 13)   rule txMipF(B) => tdW(B, 15)
  rule lvD(B, Lv, I) => maxInt(1, txDim(B, I) >>Int Lv)
  rule lvTexels(B, Lv) => txLayers(B) *Int lvD(B, Lv, 0) *Int lvD(B, Lv, 1) *Int lvD(B, Lv, 2)
  rule lvOff(B, Lv) => lvOff(B, Lv -Int 1) +Int lvTexels(B, Lv -Int 1) requires Lv >Int 0
  rule lvOff(_, _) => 0 [owise]
  // sampler words: 4 mip filter, 5 minLOD, 6 maxLOD (1/256 units, signed); a texture alone has no lod clamp
  syntax Int ::= effMip(Int, Int) [function, total] | minLodW(Int) [function, total] | maxLodW(Int) [function, total]
  rule effMip(B, S) => tdW(S, 4) requires S >=Int 0
  rule effMip(B, _) => txMipF(B) [owise]
  rule minLodW(S) => sext(tdW(S, 5), 64) requires S >=Int 0
  rule minLodW(_) => 0 [owise]
  rule maxLodW(S) => sext(tdW(S, 6), 64) requires S >=Int 0
  rule maxLodW(_) => 1048576 [owise]
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

  // the RGBA texel at resolved integer coordinates of level Lv (a negative coordinate means border: zeros)
  syntax List ::= texel0(Bytes, Int, Int, Int, Int, Int, Int) [function, total]   // gmem, blob, level, i, j, k, layer
  rule texel0(GM, B, Lv, I, J, K, L) => ListItem(zF()) ListItem(zF()) ListItem(zF()) ListItem(zF()) requires I <Int 0 orBool J <Int 0 orBool K <Int 0
  rule texel0(GM, B, Lv, I, J, K, L) =>
       texelAt(GM, B, txBase(B) +Int (lvOff(B, Lv) +Int ((L *Int lvD(B, Lv, 2) +Int K) *Int lvD(B, Lv, 1) +Int J) *Int lvD(B, Lv, 0) +Int I) *Int txCh(B) *Int txEb(B)) [owise]
  syntax List ::= texelAt(Bytes, Int, Int) [function, total]
  rule texelAt(GM, B, A) => ListItem(texC(GM, B, A, 0)) ListItem(texC(GM, B, A, 1)) ListItem(texC(GM, B, A, 2)) ListItem(texC(GM, B, A, 3))
  syntax Float ::= texC(Bytes, Int, Int, Int) [function, total]
  rule texC(GM, B, A, C) => chanF(GM, B, A, C) requires C <Int txCh(B)
  rule texC(_, _, _, C) => oneF() requires C ==Int 3
  rule texC(_, _, _, _) => zF() [owise]

  // sampling context Q = [level, offset x, y, z, has depth reference, reference (f32 bits)]
  syntax Int ::= qi(List, Int) [function, total]
  rule qi(Q, I) => {Q[I]}:>Int requires I >=Int 0 andBool I <Int size(Q)
  rule qi(_, _) => 0 [owise]
  syntax List ::= mkQ(Int, List, Bool, Int) [function, total]
  rule mkQ(Lv, Of, H, Rf) => ListItem(Lv) ListItem(nth0(Of, 0)) ListItem(nth0(Of, 1)) ListItem(nth0(Of, 2)) ListItem(b2i(H)) ListItem(Rf)
  syntax Int ::= nth0(List, Int) [function, total]
  rule nth0(L, I) => {L[I]}:>Int requires I >=Int 0 andBool I <Int size(L)
  rule nth0(_, _) => 0 [owise]
  // depth compare (LEQUAL): the reference against the texel's first channel, 1.0 / 0.0
  syntax List ::= texel(Bytes, Int, List, Int, Int, Int, Int) [function, total]
  rule texel(GM, B, Q, I, J, K, L) => cmpTex(Q, texel0(GM, B, qi(Q, 0), I, J, K, L))
  syntax List ::= cmpTex(List, List) [function, total]
  rule cmpTex(Q, T) => T requires qi(Q, 4) ==Int 0
  rule cmpTex(Q, T) => ListItem(#if roundFloat(decodeF(qi(Q, 5), 32), WP, WEXP) <=Float nthF(T, 0) #then oneF() #else zF() #fi) ListItem(zF()) ListItem(zF()) ListItem(oneF())
    requires qi(Q, 4) =/=Int 0

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
  rule qw(Xb) => Int2Float(flo((Xb -Float floorFloat(Xb)) *Float Int2Float(256, WP, WEXP) +Float half()), WP, WEXP) *Float inv256()
  syntax Float ::= half() [function, total]
  rule half() => divD(Int2Float(1, WP, WEXP), Int2Float(2, WP, WEXP))
  // quotient computed at double precision: mpfr_div at the 4400-bit working precision has been seen to crash or
  // return wrong values; texture-unit divisions (lod, cube projection) are approximate on hardware anyway
  syntax Float ::= divD(Float, Float) [function, total]
  rule divD(A, B) => roundFloat(roundFloat(A, 53, 11) /Float roundFloat(B, 53, 11), WP, WEXP)
  // 1/256 by multiplication: mpfr_div has crashed (SIGFPE in mpfr_divhigh_n) on some 4400-bit operands
  syntax Float ::= inv256() [function, total]
  rule inv256() => half() *Float half() *Float half() *Float half() *Float half() *Float half() *Float half() *Float half()

  // sample at (x, y, z) floats with layer l in the level of Q
  syntax List ::= sample(Bytes, Int, List, Int, Int, Int, Float, Float, Float, Int, Int) [function, total]   // gmem, blob, Q, filter, addrmode, norm, x,y,z, layer, rank
  rule sample(GM, B, Q, 0, AM, NM, X, Y, Z, L, R) =>
       texel(GM, B, Q, adIdx(AM, flo(pos(X, lvD(B, qi(Q, 0), 0), NM)) +Int qi(Q, 1), lvD(B, qi(Q, 0), 0)),
                    #if R >=Int 2 #then adIdx(AM, flo(pos(Y, lvD(B, qi(Q, 0), 1), NM)) +Int qi(Q, 2), lvD(B, qi(Q, 0), 1)) #else 0 #fi,
                    #if R >=Int 3 #then adIdx(AM, flo(pos(Z, lvD(B, qi(Q, 0), 2), NM)) +Int qi(Q, 3), lvD(B, qi(Q, 0), 2)) #else 0 #fi, L)
  rule sample(GM, B, Q, 1, AM, NM, X, Y, Z, L, R) =>
       lerp4(lerp4(lerp4(tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 0, 0, 0), tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 1, 0, 0), wx(B, Q, NM, X)),
                   lerp4(tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 0, 1, 0), tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 1, 1, 0), wx(B, Q, NM, X)), wy(B, Q, NM, Y, R)),
             lerp4(lerp4(tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 0, 0, 1), tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 1, 0, 1), wx(B, Q, NM, X)),
                   lerp4(tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 0, 1, 1), tl(GM, B, Q, AM, NM, X, Y, Z, L, R, 1, 1, 1), wx(B, Q, NM, X)), wy(B, Q, NM, Y, R)),
             wz(B, Q, NM, Z, R))
  syntax List ::= tl(Bytes, Int, List, Int, Int, Float, Float, Float, Int, Int, Int, Int, Int) [function, total]    // one footprint texel
  rule tl(GM, B, Q, AM, NM, X, Y, Z, L, R, DX, DY, DZ) =>
       texel(GM, B, Q, ix(B, Q, AM, NM, X, DX), jy(B, Q, AM, NM, Y, DY, R), kz(B, Q, AM, NM, Z, DZ, R), L)
  rule sample(GM, B, Q, _, AM, NM, X, Y, Z, L, R) => sample(GM, B, Q, 0, AM, NM, X, Y, Z, L, R) [owise]
  // linear-filter helpers: lower/upper texel index (with offset), and weights
  syntax Int ::= ix(Int, List, Int, Int, Float, Int) [function, total] | jy(Int, List, Int, Int, Float, Int, Int) [function, total] | kz(Int, List, Int, Int, Float, Int, Int) [function, total]
  rule ix(B, Q, AM, NM, X, D) => adIdx(AM, flo(pos(X, lvD(B, qi(Q, 0), 0), NM) -Float half()) +Int D +Int qi(Q, 1), lvD(B, qi(Q, 0), 0))
  rule jy(B, Q, AM, NM, Y, D, R) => #if R >=Int 2 #then adIdx(AM, flo(pos(Y, lvD(B, qi(Q, 0), 1), NM) -Float half()) +Int D +Int qi(Q, 2), lvD(B, qi(Q, 0), 1)) #else 0 #fi
  rule kz(B, Q, AM, NM, Z, D, R) => #if R >=Int 3 #then adIdx(AM, flo(pos(Z, lvD(B, qi(Q, 0), 2), NM) -Float half()) +Int D +Int qi(Q, 3), lvD(B, qi(Q, 0), 2)) #else 0 #fi
  syntax Float ::= wx(Int, List, Int, Float) [function, total] | wy(Int, List, Int, Float, Int) [function, total] | wz(Int, List, Int, Float, Int) [function, total]
  rule wx(B, Q, NM, X) => qw(pos(X, lvD(B, qi(Q, 0), 0), NM) -Float half())
  rule wy(B, Q, NM, Y, R) => #if R >=Int 2 #then qw(pos(Y, lvD(B, qi(Q, 0), 1), NM) -Float half()) #else zF() #fi
  rule wz(B, Q, NM, Z, R) => #if R >=Int 3 #then qw(pos(Z, lvD(B, qi(Q, 0), 2), NM) -Float half()) #else zF() #fi

  // ---------------------------------------------------------------- mipmaps
  // level of detail -> blend of levels.  `Lam` is the requested lod; levels are clamped by the sampler's [minLOD, maxLOD] and the texture's level count.
  syntax Float ::= lamClamp(Int, Int, Float) [function, total]      // blob, sampler, lod
  rule lamClamp(B, S, Lam) => minF(maxF(maxF(Lam, lodF(minLodW(S))), zF()), minF(lodF(maxLodW(S)), Int2Float(txLevels(B) -Int 1, WP, WEXP)))
  syntax Float ::= lodF(Int) [function, total] | minF(Float, Float) [function, total]
  rule lodF(V) => Int2Float(V, WP, WEXP) *Float inv256()
  rule minF(A, B) => #if A <=Float B #then A #else B #fi
  syntax List ::= mipSample(Bytes, Int, Int, Bool, Float, List, Int, Int, Int, Float, Float, Float, Int, Int) [function, total]
       // gmem, blob, sampler, use-lod?, lod, Q, filter, addrmode, norm, x, y, z, layer, rank
  rule mipSample(GM, B, S, false, _, Q, F, AM, NM, X, Y, Z, L, R) => sample(GM, B, Q, F, AM, NM, X, Y, Z, L, R)
  rule mipSample(GM, B, S, true, Lam, Q, F, AM, NM, X, Y, Z, L, R) =>
       mipSample2(GM, B, effMip(B, S), lamClamp(B, S, Lam), Q, F, AM, NM, X, Y, Z, L, R)
  syntax List ::= mipSample2(Bytes, Int, Int, Float, List, Int, Int, Int, Float, Float, Float, Int, Int) [function, total]
  rule mipSample2(GM, B, 1, Lc, Q, F, AM, NM, X, Y, Z, L, R) =>
       lerp4(sample(GM, B, setLv(Q, flo(Lc)), F, AM, NM, X, Y, Z, L, R),
             sample(GM, B, setLv(Q, minInt(flo(Lc) +Int 1, txLevels(B) -Int 1)), F, AM, NM, X, Y, Z, L, R), qw(Lc))
  rule mipSample2(GM, B, _, Lc, Q, F, AM, NM, X, Y, Z, L, R) =>
       sample(GM, B, setLv(Q, minInt(flo(Lc +Float half()), txLevels(B) -Int 1)), F, AM, NM, X, Y, Z, L, R) [owise]
  syntax List ::= setLv(List, Int) [function, total]
  rule setLv(Q, Lv) => ListItem(Lv) ListItem(qi(Q, 1)) ListItem(qi(Q, 2)) ListItem(qi(Q, 3)) ListItem(qi(Q, 4)) ListItem(qi(Q, 5))
  // lod from gradients: log2 of the larger footprint axis (in texels of level 0)
  syntax Float ::= gradLam(Int, Int, Int, List, List, Float) [function, total]    // blob, norm, rank, dPdx, dPdy, scale divisor
  rule gradLam(B, NM, R, Dx, Dy, Sd) => lamLog(maxF(gLen2(B, NM, R, Dx, 0, Sd), gLen2(B, NM, R, Dy, 0, Sd)))
  syntax Float ::= lamLog(Float) [function, total]
  rule lamLog(S) => divD(logFloat(roundFloat(S, 53, 11)), Int2Float(2, WP, WEXP) *Float logFloat(Int2Float(2, 53, 11))) requires S >Float zF()
  rule lamLog(_) => Int2Float(0 -Int 1000, WP, WEXP) [owise]
  syntax Float ::= gLen2(Int, Int, Int, List, Int, Float) [function, total]
  rule gLen2(B, NM, R, D, I, Sd) => gTerm(nthF(D, I) *Float gScale(B, NM, I, Sd)) +Float gLen2(B, NM, R, D, I +Int 1, Sd) requires I <Int R
  rule gLen2(_, _, _, _, _, _) => zF() [owise]
  syntax Float ::= gTerm(Float) [function, total] | gScale(Int, Int, Int, Float) [function, total]
  rule gTerm(X) => X *Float X
  rule gScale(B, 1, I, Sd) => divD(Int2Float(txDim(B, I), WP, WEXP), Sd)
  rule gScale(_, _, _, Sd) => divD(Int2Float(1, WP, WEXP), Sd) [owise]
  syntax Float ::= absF(Float) [function, total]
  rule absF(X) => #if X <Float zF() #then zF() -Float X #else X #fi

  // ---------------------------------------------------------------- cubemaps
  // direction (x, y, z) -> face (+X -X +Y -Y +Z -Z) and normalized face coordinates (OpenGL convention)
  syntax Int ::= cubeFace(Float, Float, Float) [function, total]
  rule cubeFace(X, Y, Z) => #if X >=Float zF() #then 0 #else 1 #fi requires absF(X) >=Float absF(Y) andBool absF(X) >=Float absF(Z)
  rule cubeFace(X, Y, Z) => #if Y >=Float zF() #then 2 #else 3 #fi requires absF(Y) >Float absF(X) andBool absF(Y) >=Float absF(Z)
  rule cubeFace(X, Y, Z) => #if Z >=Float zF() #then 4 #else 5 #fi [owise]
  syntax Float ::= cubeMa(Float, Float, Float) [function, total]
  rule cubeMa(X, Y, Z) => #if cubeFace(X, Y, Z) <Int 2 #then absF(X) #else #if cubeFace(X, Y, Z) <Int 4 #then absF(Y) #else absF(Z) #fi #fi
  syntax Float ::= cubeSC(Float, Float, Float) [function, total] | cubeTC(Float, Float, Float) [function, total] | cubeNorm(Float, Float) [function, total]
  rule cubeSC(X, Y, Z) => cscSel(cubeFace(X, Y, Z), X, Z)
  rule cubeTC(X, Y, Z) => ctcSel(cubeFace(X, Y, Z), Y, Z)
  syntax Float ::= cscSel(Int, Float, Float) [function, total] | ctcSel(Int, Float, Float) [function, total] | negF(Float) [function, total]
  rule negF(X) => zF() -Float X
  rule cscSel(0, _, Z) => negF(Z)    rule cscSel(1, _, Z) => Z    rule cscSel(5, X, _) => negF(X)    rule cscSel(_, X, _) => X [owise]
  rule ctcSel(2, _, Z) => Z          rule ctcSel(3, _, Z) => negF(Z)    rule ctcSel(_, Y, _) => negF(Y) [owise]
  rule cubeNorm(S, Ma) => (divD(S, Ma) +Float oneF()) *Float half() requires Ma >Float zF()
  rule cubeNorm(_, _) => half() [owise]

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
  syntax Bool ::= isArr(Set) [function, total] | isCube(Set) [function, total]
  rule isArr(M) => "a1d" in M orBool "a2d" in M orBool "acube" in M
  rule isCube(M) => "cube" in M orBool "acube" in M

  // coordinates as floats (ctype f32) or exact integers (ctype s32 -> point sampling of texel indices)
  syntax Float ::= coordF(Operand, Ty, Ctx) [function, total]
  rule coordF(O, ty("f", 32), C) => roundFloat(decodeF(rd(O, ty("b", 32), C), 32), WP, WEXP)
  rule coordF(O, T, C) => Int2Float(ival(rd(O, ty("s", 32), C), ty("s", 32)), WP, WEXP) [owise]
  syntax Operand ::= vecAt(Operand, Int) [function, total]
  rule vecAt(O, I) => {vecOps(O)[I]}:>Operand requires I >=Int 0 andBool I <Int size(vecOps(O))
  rule vecAt(_, _) => 0 [owise]

  // tex[.base|.level|.grad].geom.v4.dtype.ctype d[|p], [tex, {coords}] | [tex, sampler, {coords}] {, lod | , dPdx, dPdy} {, offset} {, depth-ref}
  syntax KItem ::= #texOut(Operand, List)
  rule <tk> exec(opc("tex", M, ListItem(DT) ListItem(CT)), ops(Dv, [ T , Cs ], Ex:Operands), C) =>
            #texOut(Dv, texRun(GM, readBytes(GM, hAddr(T, C), 128), 0 -Int 1, M, DT, CT, Cs, opsList(Ex), C)) ... </tk>
       <gmem> GM </gmem>
  rule <tk> exec(opc("tex", M, ListItem(DT) ListItem(CT)), ops(Dv, [ T , S , Cs ], Ex:Operands), C) =>
            #texOut(Dv, texRun(GM, readBytes(GM, hAddr(T, C), 128), readBytes(GM, hAddr(S, C), 128), M, DT, CT, Cs, opsList(Ex), C)) ... </tk>
       <gmem> GM </gmem>
  // the destination vector, optionally followed by a residency predicate (always set: every level is resident)
  rule <tk> #texOut({ Os:Operands } | P:Reg, Vs) => #wmWr(opsList(Os), Vs) ... </tk> <regs> R => wr(R, P, 1) </regs>
  rule <tk> #texOut({ Os:Operands } | P:Name, Vs) => #wmWr(opsList(Os), Vs) ... </tk> <regs> R => wr(R, P, 1) </regs>
  rule <tk> #texOut(Dv, Vs) => #wmWr(vecOps(Dv), Vs) ... </tk> [owise]

  // the operands after the coordinates: how many are the lod / gradients
  syntax Int ::= nLod(Set) [function, total]
  rule nLod(M) => 1 requires "level" in M
  rule nLod(M) => 2 requires "grad" in M
  rule nLod(_) => 0 [owise]
  syntax List ::= dropL(List, Int) [function, total]
  rule dropL(ListItem(_) L, N) => dropL(L, N -Int 1) requires N >Int 0
  rule dropL(L, _) => L [owise]
  // offset: the first remaining operand when it is a vector (or a lone scalar for 1d), depth reference: the scalar one
  syntax List ::= txOffs(List, Ctx) [function, total]
  rule txOffs(ListItem(O:Operand) _, C) => offList(O, C) requires vecOps(O) =/=K .List
  rule txOffs(_, _) => .List [owise]
  syntax List ::= offList(Operand, Ctx) [function, total]
  rule offList(O, C) => ListItem(offAt(O, 0, C)) ListItem(offAt(O, 1, C)) ListItem(offAt(O, 2, C))
  syntax Int ::= offAt(Operand, Int, Ctx) [function, total]
  rule offAt(O, I, C) => ival(rd(vecAt(O, I), ty("s", 32), C), ty("s", 32)) requires I <Int size(vecOps(O))
  rule offAt(_, _, _) => 0 [owise]
  syntax List ::= txRest(List) [function, total]           // drop the offset vector, if any
  rule txRest(ListItem(O:Operand) L) => L requires vecOps(O) =/=K .List
  rule txRest(L) => L [owise]
  syntax Bool ::= txHasRef(List) [function, total]
  rule txHasRef(ListItem(_) _) => true
  rule txHasRef(_) => false [owise]
  syntax Int ::= txRef(List, Ctx) [function, total]
  rule txRef(ListItem(O:Operand) _, C) => rd(O, ty("b", 32), C)
  rule txRef(_, _) => 0 [owise]
  syntax Float ::= lodArg(List, Ctx) [function, total]
  rule lodArg(ListItem(O:Operand) _, C) => coordF(O, ty("f", 32), C)
  rule lodArg(_, _) => zF() [owise]
  syntax List ::= gradVec(List, Int, Ctx) [function, total]
  rule gradVec(Ex, I, C) => floatsOf(vecOps(nthOp(Ex, I)), C)
  syntax Operand ::= nthOp(List, Int) [function, total]
  rule nthOp(L, I) => {L[I]}:>Operand requires I >=Int 0 andBool I <Int size(L)
  rule nthOp(_, _) => 0 [owise]
  syntax List ::= floatsOf(List, Ctx) [function, total]
  rule floatsOf(ListItem(O:Operand) L, C) => ListItem(coordF(O, ty("f", 32), C)) floatsOf(L, C)
  rule floatsOf(_, _) => .List [owise]

  syntax List ::= texRun(Bytes, Int, Int, Set, Ty, Ty, Operand, List, Ctx) [function, total]
  rule texRun(GM, B, S, M, DT, CT, Cs, Ex, C) =>
       outList(DT, texRun2(GM, B, S, M, CT, Cs, Ex, dropL(Ex, nLod(M)), C))
  syntax List ::= texRun2(Bytes, Int, Int, Set, Ty, Operand, List, List, Ctx) [function, total]       // Rest = operands after lod / gradients
  rule texRun2(GM, B, S, M, CT, Cs, Ex, Rest, C) =>
       texRun3(GM, B, S, M, CT, Cs, Ex, mkQ(0, txOffs(Rest, C), txHasRef(txRest(Rest)), txRef(txRest(Rest), C)), C)
  syntax List ::= texRun3(Bytes, Int, Int, Set, Ty, Operand, List, List, Ctx) [function, total]
  // cubemaps: pick the face, then sample the face as a clamped 2D layer
  rule texRun3(GM, B, S, M, CT, Cs, Ex, Q, C) =>
       mipSample(GM, B, S, nLod(M) >Int 0,
                 #if "level" in M #then lodArg(Ex, C) #else gradLam(B, 1, 3, gradVec(Ex, 0, C), gradVec(Ex, 1, C), cubeMa(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)) *Float Int2Float(2, WP, WEXP)) #fi,
                 Q, effFilter(B, S), 1, 1,
                 cubeNorm(cubeSC(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)), cubeMa(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C))),
                 cubeNorm(cubeTC(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)), cubeMa(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C))),
                 zF(), 6 *Int minInt(#if "acube" in M #then rd(vecAt(Cs, 0), ty("u", 32), C) #else 0 #fi, maxInt(txLayers(B) /Int 6 -Int 1, 0)) +Int cubeFace(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)), 2)
    requires isCube(M)
  rule texRun3(GM, B, S, M, CT, Cs, Ex, Q, C) =>
       mipSample(GM, B, S, nLod(M) >Int 0,
                 #if "level" in M #then lodArg(Ex, C) #else gradLam(B, effNorm(B, S), rankOf(M), gradVec(Ex, 0, C), gradVec(Ex, 1, C), oneF()) #fi,
                 Q, #if CT ==K ty("f", 32) #then effFilter(B, S) #else 0 #fi, effAddr(B, S), effNorm(B, S),
                 coordF(vecAt(Cs, #if isArr(M) #then 1 #else 0 #fi), CT, C),
                 coordF(vecAt(Cs, #if isArr(M) #then 2 #else 1 #fi), CT, C),
                 coordF(vecAt(Cs, #if isArr(M) #then 3 #else 2 #fi), CT, C),
                 #if isArr(M) #then minInt(rd(vecAt(Cs, 0), ty("u", 32), C), txLayers(B) -Int 1) #else 0 #fi, rankOf(M)) [owise]
  syntax Float ::= cx(Operand, Set, Ctx) [function, total] | cy(Operand, Set, Ctx) [function, total] | cz(Operand, Set, Ctx) [function, total]
  rule cx(Cs, M, C) => coordF(vecAt(Cs, #if "acube" in M #then 1 #else 0 #fi), ty("f", 32), C)
  rule cy(Cs, M, C) => coordF(vecAt(Cs, #if "acube" in M #then 2 #else 1 #fi), ty("f", 32), C)
  rule cz(Cs, M, C) => coordF(vecAt(Cs, #if "acube" in M #then 3 #else 2 #fi), ty("f", 32), C)

  // tld4.{r,g,b,a}.{2d,a2d,cube,acube}.v4.dtype.f32 d[|p], [tex {, sampler}, {coords}] {, offset} {, depth-ref}
  // the component of the 2x2 footprint, counter-clockwise from the lower-left texel (level 0; sampler address mode)
  rule <tk> exec(opc("tld4", M, ListItem(DT) ListItem(_)), ops(Dv, [ T , Cs ], Ex:Operands), C) =>
            #texOut(Dv, outList(DT, gatherRun(GM, readBytes(GM, hAddr(T, C), 128), 0 -Int 1, M, Cs, opsList(Ex), C))) ... </tk>
       <gmem> GM </gmem>
  rule <tk> exec(opc("tld4", M, ListItem(DT) ListItem(_)), ops(Dv, [ T , S , Cs ], Ex:Operands), C) =>
            #texOut(Dv, outList(DT, gatherRun(GM, readBytes(GM, hAddr(T, C), 128), readBytes(GM, hAddr(S, C), 128), M, Cs, opsList(Ex), C))) ... </tk>
       <gmem> GM </gmem>
  syntax Int ::= gcomp(Set) [function, total]
  rule gcomp(M) => 1 requires "g" in M
  rule gcomp(M) => 2 requires "b" in M
  rule gcomp(M) => 3 requires "a" in M
  rule gcomp(_) => 0 [owise]
  syntax List ::= gatherRun(Bytes, Int, Int, Set, Operand, List, Ctx) [function, total]
  rule gatherRun(GM, B, S, M, Cs, Ex, C) =>
       gather4(GM, B, mkQ(0, txOffs(Ex, C), txHasRef(txRest(Ex)), txRef(txRest(Ex), C)), effAddr(B, S), gcomp(M), 
               gX(M, Cs, C), gY(M, Cs, C), gLayer(B, M, Cs, C), effNormG(B, S, M))
  syntax Int ::= effNormG(Int, Int, Set) [function, total]
  rule effNormG(_, _, M) => 1 requires isCube(M)
  rule effNormG(B, S, _) => effNorm(B, S) [owise]
  syntax Float ::= gX(Set, Operand, Ctx) [function, total] | gY(Set, Operand, Ctx) [function, total]
  rule gX(M, Cs, C) => cubeNorm(cubeSC(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)), cubeMa(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C))) requires isCube(M)
  rule gY(M, Cs, C) => cubeNorm(cubeTC(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)), cubeMa(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C))) requires isCube(M)
  rule gX(M, Cs, C) => coordF(vecAt(Cs, #if isArr(M) #then 1 #else 0 #fi), ty("f", 32), C) [owise]
  rule gY(M, Cs, C) => coordF(vecAt(Cs, #if isArr(M) #then 2 #else 1 #fi), ty("f", 32), C) [owise]
  syntax Int ::= gLayer(Int, Set, Operand, Ctx) [function, total]
  rule gLayer(B, M, Cs, C) => 6 *Int minInt(#if "acube" in M #then rd(vecAt(Cs, 0), ty("u", 32), C) #else 0 #fi, maxInt(txLayers(B) /Int 6 -Int 1, 0)) +Int cubeFace(cx(Cs, M, C), cy(Cs, M, C), cz(Cs, M, C)) requires isCube(M)
  rule gLayer(B, M, Cs, C) => #if isArr(M) #then minInt(rd(vecAt(Cs, 0), ty("u", 32), C), txLayers(B) -Int 1) #else 0 #fi [owise]
  // mode, address mode, component, x, y, layer, normalized
  syntax List ::= gather4(Bytes, Int, List, Int, Int, Float, Float, Int, Int) [function, total]
  rule gather4(GM, B, Q, AM, Cc, X, Y, L, NM) =>
       ListItem(nthF(texel(GM, B, Q, ix(B, Q, AM, NM, X, 0), jy(B, Q, AM, NM, Y, 1, 2), 0, L), Cc))
       ListItem(nthF(texel(GM, B, Q, ix(B, Q, AM, NM, X, 1), jy(B, Q, AM, NM, Y, 1, 2), 0, L), Cc))
       ListItem(nthF(texel(GM, B, Q, ix(B, Q, AM, NM, X, 1), jy(B, Q, AM, NM, Y, 0, 2), 0, L), Cc))
       ListItem(nthF(texel(GM, B, Q, ix(B, Q, AM, NM, X, 0), jy(B, Q, AM, NM, Y, 0, 2), 0, L), Cc))

  // txq.<field>.b32 d, [tex]
  rule <tk> exec(opc("txq", M, _), ops(D, [ T ], .Operands), C) => .K ... </tk>
       <gmem> GM </gmem> <regs> R => wr(R, D, txqVal(M, readBytes(GM, hAddr(T, C), 128))) </regs>
  syntax Int ::= txqVal(Set, Int) [function, total]
  rule txqVal(M, B) => txDim(B, 0) requires "width" in M
  rule txqVal(M, B) => txDim(B, 1) requires "height" in M
  rule txqVal(M, B) => txDim(B, 2) requires "depth" in M
  rule txqVal(M, B) => #if (txFlags(B) &Int 2) =/=Int 0 #then tdW(B, 8) /Int 6 #else tdW(B, 8) #fi requires "array_size" in M
  rule txqVal(M, B) => txNorm(B) requires "normalized_coords" in M
  rule txqVal(M, B) => txFilter(B) requires "filter_mode" in M
  rule txqVal(M, B) => txAddr(B) requires "addr_mode_0" in M orBool "addr_mode_1" in M orBool "addr_mode_2" in M
  rule txqVal(M, B) => txLevels(B) requires "num_mipmap_levels" in M
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
  rule sfA(B, M, T, Xs) => sfAN(B, M, sfAcc(M, T), Xs)
  syntax Int ::= sfAN(Int, Set, Int, List) [function, total]
  rule sfAN(B, M, N, Xs) => sfAddr(B, #if "clamp" in M #then sfClampX(B, nth(Xs, 0), N) #else nth(Xs, 0) #fi,
                                  #if "clamp" in M #then minInt(maxInt(nth(Xs, 1), 0), sfDim(B, 1) -Int 1) #else nth(Xs, 1) #fi,
                                  #if "clamp" in M #then minInt(maxInt(nth(Xs, 2), 0), sfDim(B, 2) -Int 1) #else nth(Xs, 2) #fi,
                                  #if "clamp" in M #then minInt(maxInt(nth(Xs, 3), 0), sfLayers(B) -Int 1) #else nth(Xs, 3) #fi, N, rankOf(M))

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

  // ---- formatted (.p) accesses: the lowest coordinate is a sample offset
  syntax Int ::= sfFmt(Int) [function, total] | sfCh(Int) [function, total] | sfChB(Int) [function, total]
  rule sfFmt(B) => tdW(B, 7)    rule sfCh(B) => maxInt(1, tdW(B, 8))
  rule sfChB(B) => chBytesOf(sfFmt(B))
  syntax Int ::= chBytesOf(Int) [function, total]
  rule chBytesOf(F) => 1 requires F <=Int 3
  rule chBytesOf(F) => 2 requires F >=Int 4 andBool F <=Int 8
  rule chBytesOf(_) => 4 [owise]
  // unsigned / signed integer formats (for sured.p's reinterpretation of .b32 / .b64)
  syntax Bool ::= fmtSigned(Int) [function, total]
  rule fmtSigned(F) => F ==Int 3 orBool F ==Int 7 orBool F ==Int 10
  // f32 / u32 / s32 source bits -> channel bits, saturating (UNORM / SNORM round to nearest even)
  syntax Int ::= fmtConv(Int, Int) [function, total]
  rule fmtConv(F, V) => rneI(clampF(decodeF(V, 32), zF(), oneF()) *Float Int2Float(maskPow(F), WP, WEXP)) requires F ==Int 0 orBool F ==Int 4
  rule fmtConv(F, V) => trunc(rneI(clampF(decodeF(V, 32), Int2Float(0 -Int 1, WP, WEXP), oneF()) *Float Int2Float(maskPow(F), WP, WEXP)), 8 *Int chBytesOf(F)) requires F ==Int 1 orBool F ==Int 5
  rule fmtConv(F, V) => minInt(V, (1 <<Int (8 *Int chBytesOf(F))) -Int 1) requires F ==Int 2 orBool F ==Int 6 orBool F ==Int 9
  rule fmtConv(F, V) => trunc(minInt(maxInt(sext(V, 32), 0 -Int (1 <<Int (8 *Int chBytesOf(F) -Int 1))), (1 <<Int (8 *Int chBytesOf(F) -Int 1)) -Int 1), 8 *Int chBytesOf(F)) requires F ==Int 3 orBool F ==Int 7 orBool F ==Int 10
  rule fmtConv(8, V) => fpFin(.Set, 16, roundFloat(decodeF(V, 32), WP, WEXP), 0)
  rule fmtConv(_, V) => V [owise]
  syntax Int ::= maskPow(Int) [function, total]          // largest positive code of a normalized format
  rule maskPow(F) => (1 <<Int (8 *Int chBytesOf(F) -Int (#if F ==Int 1 orBool F ==Int 5 #then 1 #else 0 #fi))) -Int 1
  syntax Float ::= clampF(Float, Float, Float) [function, total]
  rule clampF(X, Lo, Hi) => #if isNaN(X) #then zF() #else #if X <Float Lo #then Lo #else #if X >Float Hi #then Hi #else X #fi #fi #fi
  syntax Int ::= rneI(Float) [function, total]
  rule rneI(X) => #if X -Float floorFloat(X) >Float half() orBool (X -Float floorFloat(X) ==Float half() andBool Float2Int(floorFloat(X)) %Int 2 ==Int 1)
                  #then Float2Int(floorFloat(X)) +Int 1 #else Float2Int(floorFloat(X)) #fi
  syntax List ::= fmtList(Int, List, Int) [function, total]          // format, source words, count
  rule fmtList(F, ListItem(V:Int) Vs, N) => ListItem(fmtConv(F, V)) fmtList(F, Vs, N -Int 1) requires N >Int 0
  rule fmtList(_, _, _) => .List [owise]
  syntax Int ::= packCh(List, Int, Int) [function, total]
  rule packCh(ListItem(V:Int) Vs, W, I) => ((V &Int mask(W)) <<Int (W *Int I)) |Int packCh(Vs, W, I +Int 1)
  rule packCh(_, _, _) => 0 [owise]
  syntax List ::= sfScaleX(List, Int) [function, total]
  rule sfScaleX(ListItem(X:Int) Xs, N) => ListItem(X *Int N) Xs
  rule sfScaleX(L, _) => L [owise]
  // sust.p.geom.vec.b32.clamp [surf, {coords}], {r, g, b, a}
  rule <tk> exec(opc("sust", M, ListItem(T)), ops([ S , Cs ], Vs, .Operands), C) =>
            #sfStoreP(readBytes(GM, hAddr(S, C), 128), sfCoords(M, Cs, C), rdList(opsList(vecItems(Vs)), ty("b", 32), C), M) ... </tk>
       <gmem> GM </gmem>
    requires "p" in M
  syntax KItem ::= #sfStoreP(Int, List, List, Set)
  rule #sfStoreP(B, Xs, Vs, M) => #sfStore(sfAN(B, M, sfNP(B, Vs), sfScaleX(Xs, sfTb(B))),
                                           ty("b", 8 *Int sfChB(B)), fmtList(sfFmt(B), Vs, sfCh(B)), M)
  syntax Int ::= sfNP(Int, List) [function, total]
  rule sfNP(B, Vs) => minInt(size(Vs), sfCh(B)) *Int sfChB(B)
  // sured.p.op.geom.b32|b64.clamp: the sample offset scales by the data size, signedness follows the surface format
  rule <tk> exec(opc("sured", M, ListItem(T)), ops([ S , Cs ], V, .Operands), C) =>
            #sfRed(sfA(readBytes(GM, hAddr(S, C), 128), M, T, sfScaleX(sfCoords(M, Cs, C), bytes(T))), atomOp(M),
                   ty(#if fmtSigned(sfFmt(readBytes(GM, hAddr(S, C), 128))) #then "s" #else "u" #fi, bitsOf(T)), rd(V, T, C), M) ... </tk>
       <gmem> GM </gmem>
    requires "p" in M
  syntax Int ::= bitsOf(Ty) [function, total]
  rule bitsOf(T) => 8 *Int bytes(T)

  // sured.b.op.geom.type [surf, {coords}], v
  rule <tk> exec(opc("sured", M, ListItem(T)), ops([ S , Cs ], V, .Operands), C) =>
            #sfRed(sfA(readBytes(GM, hAddr(S, C), 128), M, T, sfCoords(M, Cs, C)), atomOp(M), T, rd(V, T, C), M) ... </tk>
       <gmem> GM </gmem>
    requires notBool ("p" in M)
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
  rule suqVal(M, B) => #if tdW(B, 12) =/=Int 0 #then tdW(B, 3) #else 0 #fi requires "array_size" in M
  rule suqVal(M, B) => tdW(B, 9) requires "channel_data_type" in M
  rule suqVal(M, B) => tdW(B, 10) requires "channel_order" in M
  rule suqVal(M, B) => tdW(B, 11) requires "memory_layout" in M
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

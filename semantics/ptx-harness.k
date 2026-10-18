// =============================================================================
// PTX-HARNESS: the host side.  The <k> cell walks the module top to bottom:
// PTX directives load functions and variables; harness directives allocate
// buffers, launch grids (waiting for completion, like a synchronous
// cudaLaunchKernel + cudaDeviceSynchronize) and print / check results.
// =============================================================================
requires "ptx-exec.k"

module PTX-HARNESS
  imports PTX-EXEC

  rule <k> T:TopDecl Ts:Module => T ~> Ts ... </k>
  rule <k> .Module => .K ... </k> <fails> 0 </fails> <exit-code> _ => 0 </exit-code>
  rule <k> .Module => .K ... </k> <fails> N </fails> requires N >Int 0

  // ------------------------------------------------------------ module directives
  rule <k> .version F      => .K ... </k> <isaver> _ => Float2Int(F *Float 10.0 +Float 0.5) </isaver>
  rule <k> .target Ns      => .K ... </k> <isasm> _ => targetSm(namesL(Ns)) </isasm> <isaarch> _ => targetArch(namesL(Ns)) </isaarch>
  rule <k> .isacheck ; => .K ... </k> <isachk> _ => true </isachk>
  rule <k> .address_size _ => .K ... </k>
  rule <k> .file _ _       => .K ... </k>
  rule <k> .file _ _ , _ , _ => .K ... </k>
  rule <k> (.pragma _:String ;):TopDecl => .K ... </k>
  rule <k> .section _ { _:SecItems } => .K ... </k>          // debug info is ignored
  rule <k> _:DAttrs _:TexKind _:DItems ; => .K ... </k>       // textures: see task 35
  rule <k> .alias A , B ; => .K ... </k>
       <funcs> F => F[Name2String(A) <- {F[Name2String(B)]}:>FDef] </funcs>
    requires Name2String(B) in_keys(F)

  rule <k> A:DAttrs H:FuncHead P:PerfDirs { Ss:Stmts } =>
           #addFunc(fname(H), fkind(H), frets(H), fparams(H), pp(Ss, pps(.Map, 0, .Map, fmarks(frets(H) fparams(H)), .List))) ... </k>
       <clusterDims> CDs => CDs[fname(H) <- clusterOf(P)] </clusterDims>
    requires notBool "extern" in aset(A)
  syntax Dim3 ::= clusterOf(PerfDirs) [function, total]
  rule clusterOf(.reqnctapercluster X:Int , Y:Int , Z:Int _:PerfDirs) => dim3(X, Y, Z)
  rule clusterOf(.reqnctapercluster X:Int , Y:Int _:PerfDirs) => dim3(X, Y, 1)
  rule clusterOf(.reqnctapercluster X:Int _:PerfDirs) => dim3(X, 1, 1)
  rule clusterOf(_:PerfDir P:PerfDirs) => clusterOf(P) [owise]
  rule clusterOf(_) => dim3(1, 1, 1) [owise]
  rule <k> _:DAttrs _:FuncHead _:PerfDirs ; => .K ... </k>          // prototype

  syntax KItem ::= #addFunc(String, String, List, List, PPS) | #static(List)
  // every function also gets an address (symbol F), usable in mov, initializers and indirect calls
  rule <k> #addFunc(F, K, RS, PS, pps(C, _, L, RW, SH)) => #static(SH) ... </k>
       <funcs> FS => FS[F <- fdef(K, RS, PS, C, L, RW)] </funcs>
       <syms> S => #if F in_keys(S) #then S #else S[F <- sym("func", 536870912 +Int 64 *Int size(FA))] #fi </syms>
       <faddrs> FA => #if F in_keys(S) #then FA #else FA[536870912 +Int 64 *Int size(FA) <- F] #fi </faddrs>

  rule <k> A1:DAttrs T:Type A2:DAttrs Is:DItems ; =>
           #static(sdecls(aset(A1) |Set aset(A2), alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), .List)) ... </k>
  rule <k> A1:DAttrs T:Type A2:DAttrs Is:DItems = I:InitVal ; =>
           #static(sdecls(aset(A1) |Set aset(A2), alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), initOf(I))) ... </k>

  rule <k> #static(.List) => .K ... </k>
  // `.extern .shared` arrays: dynamically sized, placed after static shared data at launch
  rule <k> #static(ListItem(sdecl("shared", N, _, _, _, true, _)) L) => #static(L) ... </k>
       <syms> S => S[N <- sym("dynshared", 0)] </syms>
  rule <k> #static(ListItem(sdecl("shared", N, _, SZ, AL, false, _)) L) => #static(L) ... </k>
       <syms> S => S[N <- sym("shared", alignUp(ST, AL))] </syms>
       <stop> ST => alignUp(ST, AL) +Int SZ </stop>
  rule <k> #static(ListItem(sdecl(SP, N, T, SZ, AL, false, I)) L) => #static(L) ... </k>
       <syms> S => S[N <- sym(SP, alignUp(G, AL))] </syms>
       <gtop> G => alignUp(G, AL) +Int maxInt(SZ, 1) </gtop>
       <gmem> M => initMem(M, alignUp(G, AL), T, resolveL(S, I)) </gmem>
    requires SP ==String "global" orBool SP ==String "const"
  rule <k> #static(ListItem(_) L) => #static(L) ... </k> [owise]

  // ------------------------------------------------------------ buffers
  syntax KItem ::= #alloc(String, Ty, Int, List)
  rule <k> .buffer N T [ C ] ; => #alloc(Name2String(N), typeOf(T), C, .List) ... </k>
  rule <k> .buffer N T = { Vs } ; => #alloc(Name2String(N), typeOf(T), size(ivList(Vs)), ivList(Vs)) ... </k>
  rule <k> .buffer N T [ C ] = { Vs } ; => #alloc(Name2String(N), typeOf(T), C, ivList(Vs)) ... </k>
  rule <k> #alloc(N, T, C, Vs) => .K ... </k>
       <gtop> G => alignUp(G, 256) +Int maxInt(C *Int bytes(T), 1) </gtop>
       <syms> S => S[N <- sym("global", alignUp(G, 256))] </syms>
       <bufs> B => B[N <- buf(alignUp(G, 256), T, C)] </bufs>
       <gmem> M => initMem(M, alignUp(G, 256), T, Vs) </gmem>

  // .multimem MM .size N { BUF0, BUF1, ... } ;   a multimem window of N bytes over the buffers (see ptx-exec-mm.k)
  syntax KItem ::= #mmReg(String, Int, List)
  rule <k> .multimem N .size S { Bs } ; => #alloc(Name2String(N), ty("b", 8), S, .List) ~> #mmReg(Name2String(N), S, namesL(Bs)) ... </k>
  rule <k> #mmReg(N, S, Bs) => .K ... </k>
       <syms> Sy </syms>
       <mmtab> T => T[symA(Sy, N) <- mm(S, symAs(Sy, Bs))] </mmtab>
  syntax List ::= symAs(Map, List) [function, total]
  syntax Int ::= symA(Map, String) [function, total]
  rule symA(Sy, N) => addrOfSym({Sy[N]}:>Sym) requires N in_keys(Sy)
  rule symA(_, _) => 0 [owise]
  syntax Int ::= addrOfSym(Sym) [function, total]
  rule addrOfSym(sym(_, A)) => A
  rule symAs(Sy, ListItem(N:String) Ns) => ListItem(symA(Sy, N)) symAs(Sy, Ns)
  rule symAs(_, _) => .List [owise]

  syntax KItem ::= #patch(String, Ty, Int, List) | #appendTo(String, Ty, List)
  // .tensormap TM .f16 BUF .dims { d0, d1, ... } .strides { s1, ... } .box { b0, b1, ... } .swizzle 0 ;
  // builds a 128-byte tensor-map object (our own encoding, see ptx-exec-tma.k) in global memory
  syntax KItem ::= #tensormap(String, Ty, String, List, List, List, Int) | #tensormapI(String, Ty, String, List, List, List, Int, List, List, List)
  // ... .swizzle Z .im2col { lower corners (W, H, D) } { upper corners } .estrides { traversal strides of dims 0.. } ;  (im2col tensor map; box = {channels per pixel, pixels per column})
  rule <k> .tensormap N T B .dims { Ds } .strides { Ss } .box { Bs } .swizzle Z .im2col { Ls } { Us } .estrides { Es } ; =>
           #tensormapI(Name2String(N), typeOf(T), Name2String(B), intsL(Ds), intsL(Ss), intsL(Bs), Z, intsL(Ls), intsL(Us), intsL(Es)) ... </k>
  rule <k> #tensormapI(N, T, B, Ds, Ss, Bs, Z, Ls, Us, Es) => #alloc(N, ty("u", 64), 16, tmBlobI(tmBlob(T, A, Ds, Ss, Bs, Z), Ls, Us, Es)) ... </k>
       <syms> ... B |-> sym(_, A) ... </syms>
  // corners are stored offset-binary (+32768) in bits 8.. of words 1 (lower) and 2 (upper); bit 40 of word 14 flags an im2col map; word 15 = traversal strides
  syntax List ::= tmBlobI(List, List, List, List) [function, total]
  rule tmBlobI(ListItem(W0) ListItem(W1) ListItem(W2) ListItem(W3) Ws, Ls, Us, Es) =>
       ListItem(W0) ListItem(W1:Int |Int (cornerPack(Ls, 0) <<Int 8)) ListItem(W2:Int |Int (cornerPack(Us, 0) <<Int 8)) ListItem(W3) tmBlobI2(Ws, Es)
  rule tmBlobI(L, _, _, _) => L [owise]
  syntax List ::= tmBlobI2(List, List) [function, total]
  // words 4..13 unchanged, word 14 gets the flag, word 15 the traversal strides
  rule tmBlobI2(ListItem(W4) ListItem(W5) ListItem(W6) ListItem(W7) ListItem(W8) ListItem(W9) ListItem(W10) ListItem(W11) ListItem(W12) ListItem(W13) ListItem(W14:Int) ListItem(_), Es) =>
       ListItem(W4) ListItem(W5) ListItem(W6) ListItem(W7) ListItem(W8) ListItem(W9) ListItem(W10) ListItem(W11) ListItem(W12) ListItem(W13)
       ListItem(W14 |Int (1 <<Int 40)) ListItem(strideWord(Es, 0))
  rule tmBlobI2(L, _) => L [owise]
  syntax Int ::= cornerPack(List, Int) [function, total] | strideWord(List, Int) [function, total]
  rule cornerPack(ListItem(V:Int) Vs, I) => ((V +Int 32768) <<Int (16 *Int I)) |Int cornerPack(Vs, I +Int 1)
  rule cornerPack(_, _) => 0 [owise]
  rule strideWord(ListItem(V:Int) Vs, I) => (V <<Int (8 *Int I)) |Int strideWord(Vs, I +Int 1)
  rule strideWord(_, _) => 0 [owise]
  rule <k> .tensormap N T B .dims { Ds } .strides { Ss } .box { Bs } .swizzle Z ; =>
           #tensormap(Name2String(N), typeOf(T), Name2String(B), intsL(Ds), intsL(Ss), intsL(Bs), Z) ... </k>
  rule <k> #tensormap(N, T, B, Ds, Ss, Bs, Z) => #alloc(N, ty("u", 64), 16, tmBlob(T, A, Ds, Ss, Bs, Z)) ... </k>
       <syms> ... B |-> sym(_, A) ... </syms>
  // .texture TX .f32 BUF .dims { w, h, d } .attrs { channels, filter (0 point / 1 linear), addrmode (0 wrap / 1 clamp / 2 border / 3 mirror),
  //                                                normalized-coords, read-normalized, layers } ;
  // .sampler SM .attrs { filter, addrmode, normalized } ;       .surface SF .b32 BUF .dims { w, h, d } .attrs { layers } ;
  // Each builds a small descriptor object in global memory (our encoding, see ptx-exec-tex.k); the object's
  // address is the 64-bit handle that kernels take as an argument.
  syntax KItem ::= #objAlloc(String, String, Int, Ty, List, List)
  rule <k> .texture N T B .dims { Ds } .attrs { As } ; => #objAlloc(Name2String(N), Name2String(B), 0, typeOf(T), intsL(As), intsL(Ds)) ... </k>
  rule <k> .surface N T B .dims { Ds } .attrs { As } ; => #objAlloc(Name2String(N), Name2String(B), 1, typeOf(T), intsL(As), intsL(Ds)) ... </k>
  rule <k> .sampler N .attrs { As } ; => #alloc(Name2String(N), ty("u", 64), 16, smBlob(intsL(As))) ... </k>
  // sampler: 0 magic, 1 filter, 2 addrmode, 3 normalized, 4 mip filter, 5 minLOD, 6 maxLOD (1/256 units; defaults 0 / 0 / no limit), padded to 16 words
  syntax List ::= smBlob(List) [function, total]
  rule smBlob(As) => ListItem(1396920397) ListItem(nthD(As, 0, 0)) ListItem(nthD(As, 1, 0)) ListItem(nthD(As, 2, 0)) ListItem(nthD(As, 3, 0))
                     ListItem(nthD(As, 4, 0)) ListItem(nthD(As, 5, 1048576)) ListItem(0) ListItem(0) ListItem(0) ListItem(0) ListItem(0) ListItem(0) ListItem(0) ListItem(0) ListItem(0)
  syntax Int ::= nthD(List, Int, Int) [function, total]
  rule nthD(L, I, D) => {L[I]}:>Int requires I >=Int 0 andBool I <Int size(L)
  rule nthD(_, _, D) => D [owise]
  rule <k> #objAlloc(N, B, K, T, As, Ds) => #alloc(N, ty("u", 64), 16, objBlob(K, T, A, As, Ds)) ... </k>
       <syms> ... B |-> sym(_, A) ... </syms>
  // texture: 0 magic, 1 base, 2 type code, 3 channels, 4 filter, 5 addrmode, 6 normalized, 7 readnorm, 8 layers, 9 w, 10 h, 11 d
  // surface: 0 magic, 1 base, 2 texel bytes, 3 layers, 4 w, 5 h, 6 d
  syntax List ::= objBlob(Int, Ty, Int, List, List) [function, total]
  rule objBlob(0, T, A, As, Ds) =>
       ListItem(1415075924) ListItem(A) ListItem(tmCode(T)) ListItem(nth(As, 0)) ListItem(nth(As, 1)) ListItem(nth(As, 2))
       ListItem(nth(As, 3)) ListItem(nth(As, 4)) ListItem(nth(As, 5)) ListItem(nth(Ds, 0)) ListItem(nth(Ds, 1)) ListItem(nth(Ds, 2))
       ListItem(nthD(As, 6, 1)) ListItem(nthD(As, 7, 0)) ListItem(0) ListItem(nthD(As, 8, 0))
  rule objBlob(_, T, A, As, Ds) =>
       ListItem(1398035526) ListItem(A) ListItem(bytes(T)) ListItem(nth(As, 0)) ListItem(nth(Ds, 0)) ListItem(nth(Ds, 1)) ListItem(nth(Ds, 2))
       ListItem(nthD(As, 1, 11)) ListItem(nthD(As, 2, 1)) ListItem(nthD(As, 3, 0)) ListItem(nthD(As, 4, 0)) ListItem(nthD(As, 5, 1))
       ListItem(nthD(As, 6, b2i(nth(As, 0) >Int 1))) ListItem(0) ListItem(0) ListItem(0) ListItem(0) [owise]
  // surface attrs { layers, sample format (0 UNORM8 1 SNORM8 2 UINT8 3 SINT8 4 UNORM16 5 SNORM16 6 UINT16 7 SINT16 8 FLOAT16 9 UINT32 10 SINT32 11 FLOAT32; default 11),
  //                 channels (1), channel data type (0), channel order (0), linear memory layout (1), is-array (layers > 1) }
  // surface words 7 format, 8 channels, 9 channel data type, 10 channel order, 11 memory layout, 12 array flag
  // 16 little-endian 64-bit words:  0 magic | 1 element type code | 2 rank | 3 global address | 4..8 dims |
  // 9..12 byte strides of dims 1..4 | 13 box dims 0..3 (16 bits each) | 14 box dim 4, swizzle<<16, fill<<24, interleave<<32 |
  // 15 element strides (8 bits each)
  syntax List ::= tmBlob(Ty, Int, List, List, List, Int) [function, total]
  rule tmBlob(T, A, Ds, Ss, Bs, Z) =>
       ListItem(1414021456) ListItem(tmEt(T)) ListItem(size(Ds)) ListItem(A)
       ListItem(nth(Ds, 0)) ListItem(nth(Ds, 1)) ListItem(nth(Ds, 2)) ListItem(nth(Ds, 3)) ListItem(nth(Ds, 4))
       ListItem(nth(Ss, 0)) ListItem(nth(Ss, 1)) ListItem(nth(Ss, 2)) ListItem(nth(Ss, 3))
       ListItem(nth(Bs, 0) |Int (nth(Bs, 1) <<Int 16) |Int (nth(Bs, 2) <<Int 32) |Int (nth(Bs, 3) <<Int 48))
       ListItem(nth(Bs, 4) |Int (swzMode(Z &Int 255) <<Int 16) |Int ((Z >>Int 8) <<Int 24) |Int (swzAtom(Z &Int 255) <<Int 48))
       ListItem(72340172838076673)
  // tensor-map element type: the value of tensormap.replace.elemtype (not the driver's CUtensorMapDataType)
  syntax Int ::= tmEt(Ty) [function, total]
  rule tmEt(ty("u", 8)) => 0    rule tmEt(ty("u", 16)) => 1   rule tmEt(ty("u", 32)) => 2   rule tmEt(ty("s", 32)) => 3
  rule tmEt(ty("u", 64)) => 4   rule tmEt(ty("s", 64)) => 5   rule tmEt(ty("f", 16)) => 6   rule tmEt(ty("f", 32)) => 7
  rule tmEt(ty("f", 64)) => 9   rule tmEt(ty("bf", 16)) => 10 rule tmEt(ty("tf", 32)) => 11
  rule tmEt(_) => 0 [owise]
  // `.swizzle Z`: the driver's swizzle enum (0 none, 1 32B, 2 64B, 3 128B, 4 / 5 / 6 = 128B with 32B atomicity / + 8B flip / 64B atomicity), fill mode in bits 8..;
  // stored as tensormap.replace's swizzle_mode (0 none, 1 32B, 2 64B, 3 128B, 4 96B) and swizzle_atomicity (0 16B, 1 32B, 2 32B+8B flip, 3 64B)
  syntax Int ::= swzMode(Int) [function, total] | swzAtom(Int) [function, total]
  rule swzMode(Z) => 3 requires Z >=Int 3 andBool Z <=Int 6
  rule swzMode(Z) => Z [owise]
  rule swzAtom(Z) => Z -Int 3 requires Z >=Int 4 andBool Z <=Int 6
  rule swzAtom(_) => 0 [owise]
  syntax Int ::= tmCode(Ty) [function, total]
  rule tmCode(ty("u", 8)) => 0   rule tmCode(ty("u", 16)) => 1   rule tmCode(ty("u", 32)) => 2   rule tmCode(ty("s", 32)) => 3
  rule tmCode(ty("u", 64)) => 4  rule tmCode(ty("s", 64)) => 5   rule tmCode(ty("f", 16)) => 6   rule tmCode(ty("f", 32)) => 7
  rule tmCode(ty("s", 8)) => 20  rule tmCode(ty("s", 16)) => 21      // texture-only codes (not CUtensorMapDataType values)
  rule tmCode(ty("f", 64)) => 8  rule tmCode(ty("bf", 16)) => 9  rule tmCode(ty("tf", 32)) => 11
  rule tmCode(_) => 0 [owise]
  syntax List ::= intsL(Ints) [function, total]
  rule intsL(X:Int , Xs:Ints) => ListItem(X) intsL(Xs)
  rule intsL(X:Int) => ListItem(X)
  rule intsL(.Ints) => .List
  rule <k> .memmodel N ; => .K ... </k> <mm> _ => Name2String(N) </mm>
  rule <k> .patch N T @ O = { Vs } ; => #patch(Name2String(N), typeOf(T), O, ivList(Vs)) ... </k>
  rule <k> #patch(N, T, O, Vs) => .K ... </k>
       <syms> ... N |-> sym(_, A) ... </syms>
       <gmem> M => initMem(M, A +Int O *Int bytes(T), T, Vs) </gmem>
  rule <k> .append N T = { Vs } ; => #appendTo(Name2String(N), typeOf(T), ivList(Vs)) ... </k>
  rule <k> #appendTo(N, T, Vs) => .K ... </k>
       <syms> ... N |-> sym(_, A) ... </syms>
       <bufs> ... N |-> buf(A, T0, C => C +Int size(Vs)) ... </bufs>
       <gtop> G => maxInt(G, A +Int (C +Int size(Vs)) *Int bytes(T0)) </gtop>
       <gmem> M => initMem(M, A +Int C *Int bytes(T0), T0, Vs) </gmem>
    requires T ==K T0

  // ------------------------------------------------------------ launch
  syntax Dim3 ::= dimOf(Dim) [function, total]
  rule dimOf(X:Int) => dim3(X, 1, 1)
  rule dimOf((X, Y)) => dim3(X, Y, 1)
  rule dimOf((X, Y, Z)) => dim3(X, Y, Z)
  syntax Int ::= vol(Dim3) [function, total]
  rule vol(dim3(X, Y, Z)) => X *Int Y *Int Z

  syntax KItem ::= #launch(String, Dim3, Dim3, List)
                 | #mkCtas(Int, Int) | #mkThreads(Int, Int, Int) | "#waitGrid"
  rule <k> .launch K <<< G , B >>> ( As ) ; => #isaChk ~> #launch(Name2String(K), dimOf(G), dimOf(B), opsList(As)) ... </k>
  rule <k> .launch K <<< G , B , _ >>> ( As ) ; => #isaChk ~> #launch(Name2String(K), dimOf(G), dimOf(B), opsList(As)) ... </k>

  // threads are created while the host holds the turn; the grid starts when
  // it is released
  syntax KItem ::= "#startGrid"
  rule <k> #startGrid => .K ... </k> <turn> _ => -1 </turn>
  rule <k> #launch(K, GD, BD, As) => #mkCtas(0, vol(GD)) ~> #startGrid ~> #waitGrid ... </k>
       <funcs> ... K |-> fdef("entry", _, PS, _, _, _) ... </funcs>
       <syms> S => fixDyn(S, keys_list(S), alignUp(ST, DYNSHARED)) </syms>
       <stop> ST </stop>
       <kernel> _ => K </kernel> <gridDim> _ => GD </gridDim> <blockDim> _ => BD </blockDim>
       <clusterDim> _ => {CDs[K] orDefault dim3(1, 1, 1)}:>Dim3 </clusterDim> <clusterDims> CDs </clusterDims>
       <kparams> _ => kpmem(PS, As, S, 0, .Bytes) </kparams>
       <kenv> _ => kpenv(PS, 0, .Map) </kenv>
       <kptop> _ => kptop(PS, 0) </kptop>
    requires size(PS) ==Int size(As)
  rule <k> #launch(K, _, _, _) => #fault("cannot launch " +String K, 0) ... </k> [owise]

  syntax Map ::= fixDyn(Map, List, Int) [function, total]
  rule fixDyn(S, ListItem(N) Ns, B) => fixDyn(S[N <- sym("shared", B)], Ns, B)
    requires S[N] ==K sym("dynshared", 0)
  rule fixDyn(S, ListItem(_) Ns, B) => fixDyn(S, Ns, B) [owise]
  rule fixDyn(S, .List, _) => S

  // kernel parameter space image
  syntax Bytes ::= kpmem(List, List, Map, Int, Bytes) [function]
  syntax Map   ::= kpenv(List, Int, Map) [function, total]
  syntax Int   ::= kptop(List, Int) [function, total]
                 | argVal(Operand, Ty, Map) [function]
  rule kpmem(ListItem(pdecl(_, T, SZ, AL)) PS, ListItem(A:Operand) As, S, O, M) =>
       kpmem(PS, As, S, alignUp(O, AL) +Int SZ, writeBytes(M, alignUp(O, AL), SZ, argVal(A, T, S)))
  rule kpmem(_, _, _, _, M) => M [owise]
  rule kpenv(ListItem(pdecl(N, _, SZ, AL)) PS, O, E) => kpenv(PS, alignUp(O, AL) +Int SZ, E[N <- sym("param", alignUp(O, AL))])
  rule kpenv(_, _, E) => E [owise]
  rule kptop(ListItem(pdecl(_, _, SZ, AL)) PS, O) => kptop(PS, alignUp(O, AL) +Int SZ)
  rule kptop(_, O) => O [owise]
  rule argVal(N:Name, _, S) => symAddr({S[Name2String(N)] orDefault noSym(Name2String(N))}:>Sym)
  rule argVal(I:Int, T, _) => intLit(I, T)
  rule argVal(F:Float, T, _) => fltLit(F, T)
  rule argVal(H:HexLit, T, _) => hexBits(H, T)

  rule <k> #mkCtas(C, N) => #mkThreads(C, 0, vol(BD)) ~> #mkCtas(C +Int 1, N) ... </k>
       <blockDim> BD </blockDim>
       <ctas> ... .Bag => <cta> <ctaid> C </ctaid> <live> vol(BD) </live> ... </cta> ... </ctas>
    requires C <Int N
  rule <k> #mkCtas(C, N) => .K ... </k> requires C >=Int N

  rule <k> #mkThreads(C, T, N) => #mkThreads(C, T +Int 1, N) ... </k>
       <kernel> K </kernel> <gridDim> GD </gridDim> <blockDim> BD </blockDim> <clusterDim> CD </clusterDim>
       <kparams> KP </kparams> <kenv> KE </kenv> <kptop> KPT </kptop>
       <funcs> ... K |-> fdef(_, _, _, _, _, FRW) ... </funcs>
       <cta>
         <ctaid> C </ctaid>
         <wlive> WL => WL[T /Int 32 <- orBits({WL[T /Int 32] orDefault 0}:>Int, laneBit(T %Int 32))] </wlive>
         <threads> ... .Bag =>
           <thread>
             <tid> T </tid>
             <fn> K </fn>
             <regs> sregs(C, T, GD, BD, CD) </regs>
             <sregs> sregs(C, T, GD, BD, CD) </sregs>
             <rw> FRW </rw>
             <env> KE </env>
             <pmem> KP </pmem>
             <ptop> KPT </ptop>
             ...
           </thread>
         ... </threads>
         ...
       </cta>
    requires T <Int N
  rule <k> #mkThreads(_, T, N) => .K ... </k> requires T >=Int N

  // special registers
  syntax Map ::= sregs(Int, Int, Dim3, Dim3, Dim3) [function, total]
               | dimRegs(String, Int, Dim3) [function, total]
               | sizeRegs(String, Dim3) [function, total]
  rule sregs(C, T, GD, BD, CD) =>
       dimRegs("%tid", T, BD) dimRegs("%ctaid", C, GD) sizeRegs("%ntid", BD) sizeRegs("%nctaid", GD)
       lanemasks(T %Int 32)
       "%laneid" |-> T %Int 32  "%warpid" |-> T /Int 32  "%nwarpid" |-> (vol(BD) +Int 31) /Int 32
       clusterRegs(C, GD, CD)
       "%smid" |-> 0  "%nsmid" |-> 1  "%gridid" |-> 0  "%clock" |-> 0  "%clock64" |-> 0  "%warpsize" |-> 32
       "%total_smem_size" |-> 233472  "%aggr_smem_size" |-> 233472  "%dynamic_smem_size" |-> 0  "%current_graph_exec" |-> 0
       "%reserved_smem_offset_begin" |-> 0  "%reserved_smem_offset_end" |-> 1024  "%reserved_smem_offset_cap" |-> 1024
       "%reserved_smem_offset_0" |-> 0  "%reserved_smem_offset_1" |-> 0
       pmRegs(0) envRegs(0)
  // cluster special registers (a launch without clusters is a 1x1x1 cluster)
  syntax Map ::= clusterRegs(Int, Dim3, Dim3) [function, total]
  rule clusterRegs(C, dim3(GX, GY, GZ), dim3(CX, CY, CZ)) =>
       "%clusterid.x" |-> (C %Int GX) /Int CX  "%clusterid.y" |-> ((C /Int GX) %Int GY) /Int CY  "%clusterid.z" |-> (C /Int (GX *Int GY)) /Int CZ
       "%nclusterid.x" |-> GX /Int CX  "%nclusterid.y" |-> GY /Int CY  "%nclusterid.z" |-> GZ /Int CZ
       "%cluster_ctaid.x" |-> (C %Int GX) %Int CX  "%cluster_ctaid.y" |-> ((C /Int GX) %Int GY) %Int CY  "%cluster_ctaid.z" |-> (C /Int (GX *Int GY)) %Int CZ
       "%cluster_nctaid.x" |-> CX  "%cluster_nctaid.y" |-> CY  "%cluster_nctaid.z" |-> CZ
       "%cluster_ctarank" |-> clRank(C, dim3(GX, GY, GZ), dim3(CX, CY, CZ))
       "%cluster_nctarank" |-> CX *Int CY *Int CZ
       "%is_explicit_cluster" |-> 0
  syntax Int ::= clRank(Int, Dim3, Dim3) [function, total]
  rule clRank(C, dim3(GX, GY, _), dim3(CX, CY, CZ)) =>
       ((C %Int GX) %Int CX) +Int (((C /Int GX) %Int GY) %Int CY) *Int CX +Int ((C /Int (GX *Int GY)) %Int CZ) *Int CX *Int CY
  // performance-monitor and environment registers read as zero
  syntax Map ::= pmRegs(Int) [function, total] | envRegs(Int) [function, total]
  rule pmRegs(I) => ("%pm" +String Int2String(I)) |-> 0 ("%pm" +String Int2String(I) +String "_64") |-> 0 pmRegs(I +Int 1) requires I <Int 8
  rule pmRegs(_) => .Map [owise]
  rule envRegs(I) => ("%envreg" +String Int2String(I)) |-> 0 envRegs(I +Int 1) requires I <Int 32
  rule envRegs(_) => .Map [owise]
  syntax Int ::= orBits(Int, Int) [function, total]
  rule orBits(A, B) => A |Int B
  syntax Map ::= lanemasks(Int) [function, total]
  // note: operands in `(...)` are avoided at value positions here, because the
  // PTX grammar's call-argument production "(" Operands ")" makes them ambiguous
  syntax Int ::= lmEq(Int) [function, total] | lmLt(Int) [function, total] | lmLe(Int) [function, total]
               | lmGt(Int) [function, total] | lmGe(Int) [function, total]
  rule lmEq(L) => 1 <<Int L
  rule lmLt(L) => lmEq(L) -Int 1
  rule lmLe(L) => lmLt(L) |Int lmEq(L)
  rule lmGt(L) => 4294967295 &Int (~Int lmLe(L))
  rule lmGe(L) => 4294967295 &Int (~Int lmLt(L))
  rule lanemasks(L) =>
       "%lanemask_eq" |-> lmEq(L)  "%lanemask_lt" |-> lmLt(L)  "%lanemask_le" |-> lmLe(L)
       "%lanemask_gt" |-> lmGt(L)  "%lanemask_ge" |-> lmGe(L)
  rule dimRegs(P, I, dim3(X, Y, _)) =>
       (P +String ".x") |-> I %Int X  (P +String ".y") |-> (I /Int X) %Int Y  (P +String ".z") |-> I /Int (X *Int Y)
  rule sizeRegs(P, dim3(X, Y, Z)) =>
       (P +String ".x") |-> X  (P +String ".y") |-> Y  (P +String ".z") |-> Z

  rule <k> #waitGrid => .K ... </k> <ctas> .Bag </ctas> <kernel> _ => "" </kernel> <turn> _ => -2 </turn>

  // ------------------------------------------------------------ inspection
  syntax KItem ::= #print(String, Ty, Int) | #expect(String, Ty, List)
  rule <k> .print N T ; => #print(Name2String(N), typeOf(T), -1) ... </k>
  rule <k> .print N T C ; => #print(Name2String(N), typeOf(T), C) ... </k>
  rule <k> #print(N, T, C) => .K ... </k>
       <syms> ... N |-> sym(_, A) ... </syms> <bufs> B </bufs> <gmem> M </gmem>
       <out> ... .List => ListItem(N +String " = [" +String fmtList(M, A, T, count(N, B, T, C)) +String "]\n") </out>

  syntax Int ::= count(String, Map, Ty, Int) [function, total]
  rule count(_, _, _, C) => C requires C >=Int 0
  rule count(N, B, T, _) => countBuf({B[N]}:>Buf, T) requires N in_keys(B)
  rule count(_, _, _, _) => 1 [owise]
  syntax Int ::= countBuf(Buf, Ty) [function, total]
  rule countBuf(buf(_, BT, C), T) => (C *Int bytes(BT)) /Int bytes(T)

  syntax String ::= fmtList(Bytes, Int, Ty, Int) [function]
  rule fmtList(_, _, _, N) => "" requires N <=Int 0
  rule fmtList(M, A, T, 1) => fmtVal(readBytes(M, A, bytes(T)), T)
  rule fmtList(M, A, T, N) => fmtVal(readBytes(M, A, bytes(T)), T) +String ", " +String fmtList(M, A +Int bytes(T), T, N -Int 1)
    requires N >Int 1

  rule <k> .expect N T = { Vs } ; => #expect(Name2String(N), typeOf(T), ivList(Vs), 0) ... </k>
  rule <k> .expect N T @ O = { Vs } ; => #expect(Name2String(N), typeOf(T), ivList(Vs), O) ... </k>
  syntax KItem ::= #expect(String, Ty, List, Int)
  rule <k> #expect(N, T, Vs, O) => .K ... </k>
       <syms> ... N |-> sym(_, A0) ... </syms> <gmem> M </gmem>
       <out> ... .List => ListItem("PASS " +String N +String "\n") </out>
    requires mismatches(M, A0 +Int O *Int bytes(T), T, Vs, O) ==String ""
  rule <k> #expect(N, T, Vs, O) => .K ... </k>
       <syms> ... N |-> sym(_, A0) ... </syms> <gmem> M </gmem>
       <fails> F => F +Int 1 </fails>
       <out> ... .List => ListItem("FAIL " +String N +String "\n" +String mismatches(M, A0 +Int O *Int bytes(T), T, Vs, O)) </out>
    requires mismatches(M, A0 +Int O *Int bytes(T), T, Vs, O) =/=String ""

  syntax String ::= mismatches(Bytes, Int, Ty, List, Int) [function]
  rule mismatches(_, _, _, .List, _) => ""
  rule mismatches(M, A, T, ListItem(V:InitVal) Vs, I) =>
       #if readBytes(M, A, bytes(T)) ==Int initBits(V, T)
       #then ""
       #else "  [" +String Int2String(I) +String "] expected " +String fmtVal(initBits(V, T), T)
             +String ", got " +String fmtVal(readBytes(M, A, bytes(T)), T) +String "\n"
       #fi +String mismatches(M, A +Int bytes(T), T, Vs, I +Int 1)
endmodule

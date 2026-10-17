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
  rule <k> .version _      => .K ... </k>
  rule <k> .target _       => .K ... </k>
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
           #addFunc(fname(H), fkind(H), frets(H), fparams(H), pp(Ss, pps(.Map, 0, .Map, .Map, .List))) ... </k>
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
  rule <k> #addFunc(F, K, RS, PS, pps(C, _, L, RW, SH)) => #static(SH) ... </k>
       <funcs> FS => FS[F <- fdef(K, RS, PS, C, L, RW)] </funcs>

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

  syntax KItem ::= #patch(String, Ty, Int, List) | #appendTo(String, Ty, List)
  // .tensormap TM .f16 BUF .dims { d0, d1, ... } .strides { s1, ... } .box { b0, b1, ... } .swizzle 0 ;
  // builds a 128-byte tensor-map object (our own encoding, see ptx-exec-tma.k) in global memory
  syntax KItem ::= #tensormap(String, Ty, String, List, List, List, Int)
  rule <k> .tensormap N T B .dims { Ds } .strides { Ss } .box { Bs } .swizzle Z ; =>
           #tensormap(Name2String(N), typeOf(T), Name2String(B), intsL(Ds), intsL(Ss), intsL(Bs), Z) ... </k>
  rule <k> #tensormap(N, T, B, Ds, Ss, Bs, Z) => #alloc(N, ty("u", 64), 16, tmBlob(T, A, Ds, Ss, Bs, Z)) ... </k>
       <syms> ... B |-> sym(_, A) ... </syms>
  // 16 little-endian 64-bit words:  0 magic | 1 element type code | 2 rank | 3 global address | 4..8 dims |
  // 9..12 byte strides of dims 1..4 | 13 box dims 0..3 (16 bits each) | 14 box dim 4, swizzle<<16, fill<<24, interleave<<32 |
  // 15 element strides (8 bits each)
  syntax List ::= tmBlob(Ty, Int, List, List, List, Int) [function, total]
  rule tmBlob(T, A, Ds, Ss, Bs, Z) =>
       ListItem(1414021456) ListItem(tmCode(T)) ListItem(size(Ds)) ListItem(A)
       ListItem(nth(Ds, 0)) ListItem(nth(Ds, 1)) ListItem(nth(Ds, 2)) ListItem(nth(Ds, 3)) ListItem(nth(Ds, 4))
       ListItem(nth(Ss, 0)) ListItem(nth(Ss, 1)) ListItem(nth(Ss, 2)) ListItem(nth(Ss, 3))
       ListItem(nth(Bs, 0) |Int (nth(Bs, 1) <<Int 16) |Int (nth(Bs, 2) <<Int 32) |Int (nth(Bs, 3) <<Int 48))
       ListItem(nth(Bs, 4) |Int (Z <<Int 16))
       ListItem(72340172838076673)
  syntax Int ::= tmCode(Ty) [function, total]
  rule tmCode(ty("u", 8)) => 0   rule tmCode(ty("u", 16)) => 1   rule tmCode(ty("u", 32)) => 2   rule tmCode(ty("s", 32)) => 3
  rule tmCode(ty("u", 64)) => 4  rule tmCode(ty("s", 64)) => 5   rule tmCode(ty("f", 16)) => 6   rule tmCode(ty("f", 32)) => 7
  rule tmCode(ty("f", 64)) => 8  rule tmCode(ty("bf", 16)) => 9  rule tmCode(ty("tf", 32)) => 11
  rule tmCode(_) => 0 [owise]
  syntax List ::= intsL(Ints) [function, total]
  rule intsL(X:Int , Xs:Ints) => ListItem(X) intsL(Xs)
  rule intsL(X:Int) => ListItem(X)
  rule intsL(.Ints) => .List
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
  rule <k> .launch K <<< G , B >>> ( As ) ; => #launch(Name2String(K), dimOf(G), dimOf(B), opsList(As)) ... </k>
  rule <k> .launch K <<< G , B , _ >>> ( As ) ; => #launch(Name2String(K), dimOf(G), dimOf(B), opsList(As)) ... </k>

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

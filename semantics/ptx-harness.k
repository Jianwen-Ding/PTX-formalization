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

  rule <k> A:DAttrs H:FuncHead _:PerfDirs { Ss:Stmts } =>
           #addFunc(fname(H), fkind(H), frets(H), fparams(H), pp(Ss, pps(.Map, 0, .Map, .Map, .List))) ... </k>
    requires notBool "extern" in aset(A)
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
       <kernel> K </kernel> <gridDim> GD </gridDim> <blockDim> BD </blockDim>
       <kparams> KP </kparams> <kenv> KE </kenv> <kptop> KPT </kptop>
       <funcs> ... K |-> fdef(_, _, _, _, _, FRW) ... </funcs>
       <cta>
         <ctaid> C </ctaid>
         <threads> ... .Bag =>
           <thread>
             <tid> T </tid>
             <fn> K </fn>
             <regs> sregs(C, T, GD, BD) </regs>
             <sregs> sregs(C, T, GD, BD) </sregs>
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
  syntax Map ::= sregs(Int, Int, Dim3, Dim3) [function, total]
               | dimRegs(String, Int, Dim3) [function, total]
               | sizeRegs(String, Dim3) [function, total]
  rule sregs(C, T, GD, BD) =>
       dimRegs("%tid", T, BD) dimRegs("%ctaid", C, GD) sizeRegs("%ntid", BD) sizeRegs("%nctaid", GD)
       lanemasks(T %Int 32)
       "%laneid" |-> T %Int 32  "%warpid" |-> T /Int 32  "%nwarpid" |-> (vol(BD) +Int 31) /Int 32
       "%smid" |-> 0  "%gridid" |-> 0  "%clock" |-> 0  "%clock64" |-> 0  "%warpsize" |-> 32
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

  rule <k> .expect N T = { Vs } ; => #expect(Name2String(N), typeOf(T), ivList(Vs)) ... </k>
  rule <k> #expect(N, T, Vs) => .K ... </k>
       <syms> ... N |-> sym(_, A) ... </syms> <gmem> M </gmem>
       <out> ... .List => ListItem("PASS " +String N +String "\n") </out>
    requires mismatches(M, A, T, Vs, 0) ==String ""
  rule <k> #expect(N, T, Vs) => .K ... </k>
       <syms> ... N |-> sym(_, A) ... </syms> <gmem> M </gmem>
       <fails> F => F +Int 1 </fails>
       <out> ... .List => ListItem("FAIL " +String N +String "\n" +String mismatches(M, A, T, Vs, 0)) </out>
    requires mismatches(M, A, T, Vs, 0) =/=String ""

  syntax String ::= mismatches(Bytes, Int, Ty, List, Int) [function]
  rule mismatches(_, _, _, .List, _) => ""
  rule mismatches(M, A, T, ListItem(V:InitVal) Vs, I) =>
       #if readBytes(M, A, bytes(T)) ==Int initBits(V, T)
       #then ""
       #else "  [" +String Int2String(I) +String "] expected " +String fmtVal(initBits(V, T), T)
             +String ", got " +String fmtVal(readBytes(M, A, bytes(T)), T) +String "\n"
       #fi +String mismatches(M, A +Int bytes(T), T, Vs, I +Int 1)
endmodule

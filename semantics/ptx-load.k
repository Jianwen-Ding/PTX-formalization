// =============================================================================
// PTX-LOAD: turning parsed PTX into the internal representation.
//
//  * Function bodies are flattened into a code map `pc |-> instruction`;
//    nested `{ }` scopes are inlined (re-declarations simply rebind names, which
//    is how compiler-generated call sequences reuse `param0`, `retval0`, ...).
//  * Labels are resolved to pcs, `.reg` declarations record register widths.
//  * `.shared` / `.global` / `.const` variables are statically allocated;
//    `.local` / `.param` variables declared in a body become `declVar`
//    instructions that allocate in the frame when executed.
// =============================================================================
requires "ptx-configuration.k"

module PTX-LOAD
  imports PTX-CONFIGURATION

  syntax PGuard ::= "noG" | g(String, Bool)                  // register, negated?
  syntax KItem  ::= ins(PGuard, Opc, Operands)
                  | declVar(String, String, Int, Int)        // space, name, size, align

  // ------------------------------------------------------------ declarations
  syntax Set ::= aset(DAttrs) [function, total]
  rule aset(.visible R:DAttrs) => SetItem("visible") aset(R)
  rule aset(.extern  R:DAttrs) => SetItem("extern")  aset(R)
  rule aset(.weak    R:DAttrs) => SetItem("weak")    aset(R)
  rule aset(.common  R:DAttrs) => SetItem("common")  aset(R)
  rule aset(.global  R:DAttrs) => SetItem("global")  aset(R)
  rule aset(.shared  R:DAttrs) => SetItem("shared")  aset(R)
  rule aset(.local   R:DAttrs) => SetItem("local")   aset(R)
  rule aset(.const   R:DAttrs) => SetItem("const")   aset(R)
  rule aset(.param   R:DAttrs) => SetItem("param")   aset(R)
  rule aset(.reg     R:DAttrs) => SetItem("reg")     aset(R)
  rule aset(.ptr     R:DAttrs) => SetItem("ptr")     aset(R)
  rule aset(.v2      R:DAttrs) => SetItem("v2")      aset(R)
  rule aset(.v4      R:DAttrs) => SetItem("v4")      aset(R)
  rule aset(.v8      R:DAttrs) => SetItem("v8")      aset(R)
  rule aset(.attribute ( .managed ) R:DAttrs) => SetItem("managed") aset(R)
  rule aset(.align _:Int R:DAttrs) => aset(R)
  rule aset(.DAttrs) => .Set

  syntax Int ::= alignOf(DAttrs) [function, total]
  rule alignOf(.align N:Int _:DAttrs) => N
  rule alignOf(_:DAttr R:DAttrs) => alignOf(R) [owise]
  rule alignOf(.DAttrs) => 0

  syntax Int ::= alignOr(DAttrs, DAttrs, Ty) [function, total]
  rule alignOr(A1, A2, T) => #if alignOf(A1) >Int 0 #then alignOf(A1)
                             #else #if alignOf(A2) >Int 0 #then alignOf(A2) #else bytes(T) #fi #fi

  syntax String ::= dspace(Set) [function, total]
  rule dspace(S) => pick(S, ListItem("reg") ListItem("shared") ListItem("local") ListItem("param")
                            ListItem("global") ListItem("const"), "global")

  syntax List ::= itemsL(DItems) [function, total]
  rule itemsL(I:DItem, Is:DItems) => ListItem(I) itemsL(Is)
  rule itemsL(.DItems) => .List

  syntax String ::= itemName(DItem) [function, total]
  rule itemName(N:Name)          => Name2String(N)
  rule itemName(R:Reg)           => Reg2String(R)
  rule itemName(R:Reg < _ >)     => Reg2String(R)
  rule itemName(N:Name < _ >)    => Name2String(N)
  rule itemName(N:Name _:ArrDims) => Name2String(N)

  syntax Int ::= itemSize(DItem, Ty) [function, total]
  rule itemSize(_:Name D:ArrDims, T)    => dimsCount(D) *Int bytes(T)
  rule itemSize(_, T)                   => bytes(T) [owise]

  syntax Int ::= dimsCount(ArrDims) [function, total]
  rule dimsCount([ N:Int ] R:ArrDims) => N *Int dimsCount(R)
  rule dimsCount([ N:Int ]) => N
  rule dimsCount([ ]) => 0

  syntax List ::= namesL(Names) [function, total]
  rule namesL(N:Name, Ns:Names) => ListItem(Name2String(N)) namesL(Ns)
  rule namesL(N:Name) => ListItem(Name2String(N))
  rule namesL(_) => .List [owise]

  // static variables (module scope, or .shared/.global/.const in a body)
  syntax List ::= sdecls(Set, Int, Ty, List, List) [function, total]
  rule sdecls(S, AL, T, ListItem(I:DItem) Is, Init) =>
       ListItem(sdecl(dspace(S), itemName(I), T, itemSize(I, T), AL, "extern" in S, Init))
       sdecls(S, AL, T, Is, Init)
  rule sdecls(_, _, _, _, _) => .List [owise]

  syntax List ::= initOf(InitVal) [function, total]
  rule initOf({ Vs }) => ivList(Vs)
  rule initOf(V) => ListItem(V) [owise]

  // ------------------------------------------------------------ parameters
  syntax List ::= pdecls(ParamDecls) [function, total]
  rule pdecls(A1:DAttrs T:Type A2:DAttrs I:DItem, Ps:ParamDecls) =>
       ListItem(pdecl(itemName(I), typeOf(T), itemSize(I, typeOf(T)), alignOr(A1, A2, typeOf(T))))
       pdecls(Ps)
    requires notBool "reg" in (aset(A1) |Set aset(A2))
  // `.reg` formals are passed by value: alignment -1 (scalar) / -2 (first component of a vector of `size` components,
  // flattened to name.x, name.y, ...)
  rule pdecls(A1:DAttrs T:Type A2:DAttrs I:DItem, Ps:ParamDecls) =>
       regPD(vwidth(aset(A1) |Set aset(A2)), itemName(I), typeOf(T)) pdecls(Ps)
    requires "reg" in (aset(A1) |Set aset(A2))
  rule pdecls(.ParamDecls) => .List
  syntax Int ::= vwidth(Set) [function, total]
  rule vwidth(S) => 2 requires "v2" in S
  rule vwidth(S) => 4 requires "v4" in S
  rule vwidth(S) => 8 requires "v8" in S
  rule vwidth(S) => 0 requires notBool ("v2" in S orBool "v4" in S orBool "v8" in S)
  syntax List ::= regPD(Int, String, Ty) [function, total] | regComps(String, Ty, Int, Int) [function, total]
  rule regPD(0, N, T) => ListItem(pdecl(N, T, 1, -1))
  rule regPD(K, N, T) => regComps(N, T, 0, K) requires K >Int 0
  rule regComps(N, T, I, K) => ListItem(pdecl(N +String "." +String vcomp(I), T, #if I ==Int 0 #then K #else 1 #fi, #if I ==Int 0 #then -2 #else -1 #fi))
                               regComps(N, T, I +Int 1, K) requires I <Int K
  rule regComps(_, _, I, K) => .List requires I >=Int K
  // register-width / vector-name seed for a function: its `.reg` formals
  syntax Map ::= fmarks(List) [function, total] | fmark1(PDecl) [function, total]
  rule fmarks(.List) => .Map
  rule fmarks(ListItem(P) Ps) => updateMap(fmark1(P), fmarks(Ps))
  rule fmark1(pdecl(N, T, _, -1)) => N |-> wid(T)
  rule fmark1(pdecl(N, T, K, -2)) => (N |-> wid(T)) ("@vec:" +String substrString(N, 0, lengthString(N) -Int 2) |-> K)
  rule fmark1(_) => .Map [owise]

  // vector registers: .reg .v4 .f32 v;  declares the components v.x v.y v.z v.w and the marker "@vec:v"
  syntax String ::= vcomp(Int) [function, total]
  rule vcomp(0) => "x"
  rule vcomp(1) => "y"
  rule vcomp(2) => "z"
  rule vcomp(3) => "w"
  rule vcomp(I) => "s" +String Int2String(I) requires I >=Int 4 orBool I <Int 0
  syntax Map ::= vdeclare(Map, String, Int, Int, Int) [function, total]
  rule vdeclare(RW, N, W, K, I) => vdeclare(RW[N +String "." +String vcomp(I) <- W], N, W, K, I +Int 1) requires I <Int K
  rule vdeclare(RW, N, _, K, I) => RW["@vec:" +String N <- K] requires I >=Int K
  syntax Name ::= String2Name(String) [function, total, hook(STRING.string2token)]
  syntax Reg  ::= String2Reg(String)  [function, total, hook(STRING.string2token)]
  syntax Operand ::= compTok(String) [function, total]
  rule compTok(S) => String2Reg(S)  requires substrString(S, 0, 1) ==String "%"
  rule compTok(S) => String2Name(S) requires substrString(S, 0, 1) =/=String "%"
  syntax Operands ::= vcomps(String, Int, Int) [function, total]
  rule vcomps(B, I, K) => compTok(B +String "." +String vcomp(I)) , vcomps(B, I +Int 1, K) requires I +Int 1 <Int K
  rule vcomps(B, I, K) => compTok(B +String "." +String vcomp(I)) requires I +Int 1 >=Int K
  syntax Int ::= vecK(Operand, Map) [function, total]
  rule vecK(N:Name, RW) => {RW["@vec:" +String Name2String(N)] orDefault 0}:>Int
  rule vecK(R:Reg, RW) => {RW["@vec:" +String Reg2String(R)] orDefault 0}:>Int
  rule vecK(_, _) => 0 [owise]
  // bare vector registers in an instruction become {v.x, v.y, ...}; inside call argument lists they are flattened
  syntax Operands ::= vexp(Map, Operands) [function, total] | vflat(Map, Operands) [function, total]
  rule vexp(_, .Operands) => .Operands
  rule vexp(RW, ( As:Operands ) , Os:Operands) => ( vflat(RW, As) ) , vexp(RW, Os)
  rule vexp(RW, O:Operand , Os:Operands) => vexp1(RW, O) , vexp(RW, Os) [owise]
  syntax Operand ::= vexp1(Map, Operand) [function, total]
  rule vexp1(RW, O) => { vcomps(opStr(O), 0, vecK(O, RW)) } requires vecK(O, RW) >Int 0
  rule vexp1(RW, O) => O requires vecK(O, RW) <=Int 0
  rule vflat(_, .Operands) => .Operands
  rule vflat(RW, O:Operand , Os:Operands) => vcompsT(opStr(O), 0, vecK(O, RW), vflat(RW, Os)) requires vecK(O, RW) >Int 0
  rule vflat(RW, O:Operand , Os:Operands) => O , vflat(RW, Os) requires vecK(O, RW) <=Int 0
  syntax Operands ::= vcompsT(String, Int, Int, Operands) [function, total]
  rule vcompsT(B, I, K, T) => compTok(B +String "." +String vcomp(I)) , vcompsT(B, I +Int 1, K, T) requires I <Int K
  rule vcompsT(_, I, K, T) => T requires I >=Int K


  // ------------------------------------------------------------ scoped register declarations
  // A `.reg` declared inside a nested block with the name of an already visible register shadows it: it is
  // renamed to name@id for the rest of the block (L["@ren:name"] = id; L["@depth"] = nesting depth).
  syntax Operand ::= rnO(Map, Operand) [function, total]
  syntax Addr ::= rnA(Map, Addr) [function, total]
  syntax Operands ::= rnOs(Map, Operands) [function, total]
  syntax String ::= rnS(Map, String) [function, total]
  syntax PGuard ::= rnG(Map, PGuard) [function, total]
  rule rnS(L, S) => S +String "@" +String Int2String({L["@ren:" +String S]}:>Int) requires ("@ren:" +String S) in_keys(L)
  rule rnS(L, S) => S requires notBool (("@ren:" +String S) in_keys(L))
  rule rnG(_, noG) => noG
  rule rnG(L, g(P, N)) => g(rnS(L, P), N)
  rule rnA(L, R:Reg) => String2Reg(rnS(L, Reg2String(R)))
  rule rnA(L, N:Name) => String2Name(rnS(L, Name2String(N)))
  rule rnA(L, R:Reg I:Int) => String2Reg(rnS(L, Reg2String(R))) I
  rule rnA(L, N:Name I:Int) => String2Name(rnS(L, Name2String(N))) I
  rule rnA(L, R:Reg + I:Int) => String2Reg(rnS(L, Reg2String(R))) + I
  rule rnA(L, N:Name + I:Int) => String2Name(rnS(L, Name2String(N))) + I
  rule rnA(_, A) => A [owise]
  rule rnO(L, R:Reg) => String2Reg(rnS(L, Reg2String(R)))
  rule rnO(L, N:Name) => String2Name(rnS(L, Name2String(N)))
  syntax Addr ::= rnB(Map, Addr) [function, total]
  rule rnB(L, A) => rnA(L, A)
  syntax Operand ::= idO(Operand) [function, total]
  rule idO(O) => O
  rule rnO(L, [ A ]) => idO([ rnA(L, A) ])
  rule rnO(L, [ A , O ]) => idO([ rnA(L, A) , rnO(L, O) ])
  rule rnO(L, [ A , B , O ]) => idO([ rnA(L, A) , rnB(L, B) , rnO(L, O) ])
  rule rnO(L, { Os }) => { rnOs(L, Os) }
  rule rnO(L, ( Os )) => ( rnOs(L, Os) )
  rule rnO(L, R1:Reg | R2:Reg) => String2Reg(rnS(L, Reg2String(R1))) | String2Reg(rnS(L, Reg2String(R2)))
  rule rnO(L, N1:Name | N2:Name) => String2Name(rnS(L, Name2String(N1))) | String2Name(rnS(L, Name2String(N2)))
  rule rnO(L, ! R:Reg) => ! String2Reg(rnS(L, Reg2String(R)))
  rule rnO(L, ! N:Name) => ! String2Name(rnS(L, Name2String(N)))
  rule rnO(_, O) => O [owise]
  rule rnOs(_, .Operands) => .Operands
  rule rnOs(L, O:Operand , Os:Operands) => rnO(L, O) , rnOs(L, Os)
  // leaving a block: forget its renames and restore the enclosing ones
  syntax PPS ::= leaveBlock(PPS, Map) [function, total]
  rule leaveBlock(pps(C, I, L2, RW, SH), L) => pps(C, I, addRen(dropRen(L2, keys_list(L2)), L, keys_list(L)), RW, SH)
  syntax Map ::= dropRen(Map, List) [function, total] | addRen(Map, Map, List) [function, total]
  rule dropRen(L, .List) => L
  rule dropRen(L, ListItem(K:String) Ks) => dropRen(#if pfx(K, "@ren:") orBool K ==String "@depth" #then L[K <- undef] #else L #fi, Ks)
  rule dropRen(L, _) => L [owise]
  rule addRen(L, _, .List) => L
  rule addRen(L, Old, ListItem(K:String) Ks) => addRen(#if pfx(K, "@ren:") orBool K ==String "@depth" #then L[K <- {Old[K]}:>Int] #else L #fi, Old, Ks)
  rule addRen(L, _, _) => L [owise]
  syntax Bool ::= isPlain(DItem) [function, total]
  rule isPlain(_:Name) => true
  rule isPlain(_:Reg) => true
  rule isPlain(_) => false [owise]
  syntax DItem ::= renamedItem(DItem, Int) [function, total]
  rule renamedItem(R:Reg, Id) => String2Reg(Reg2String(R) +String "@" +String Int2String(Id))
  rule renamedItem(N:Name, Id) => String2Name(Name2String(N) +String "@" +String Int2String(Id))
  rule renamedItem(I, _) => I [owise]

  // ------------------------------------------------------------ function heads
  syntax String ::= fname(FuncHead) [function, total]
                  | fkind(FuncHead) [function, total]
  syntax List   ::= frets(FuncHead)   [function, total]
                  | fparams(FuncHead) [function, total]
  rule fname(.entry N ( _ ))       => Name2String(N)
  rule fname(.entry N:Name)        => Name2String(N)
  rule fname(.func N ( _ ))        => Name2String(N)
  rule fname(.func N:Name)         => Name2String(N)
  rule fname(.func ( _ ) N ( _ ))  => Name2String(N)
  rule fname(.func ( _ ) N:Name)   => Name2String(N)
  rule fkind(.entry _ ( _ ))  => "entry"
  rule fkind(.entry _:Name)   => "entry"
  rule fkind(_)               => "func" [owise]
  rule frets(.func ( R ) _ ( _ ))  => pdecls(R)
  rule frets(.func ( R ) _:Name)   => pdecls(R)
  rule frets(_)                    => .List [owise]
  rule fparams(.entry _ ( P ))       => pdecls(P)
  rule fparams(.func _ ( P ))        => pdecls(P)
  rule fparams(.func ( _ ) _ ( P ))  => pdecls(P)
  rule fparams(_)                    => .List [owise]

  // ------------------------------------------------------------ bodies
  // code, next pc, labels, register widths, static declarations
  syntax PPS ::= pps(Map, Int, Map, Map, List)
  syntax PPS ::= pp(Stmts, PPS)            [function, total]
               | pp1(Stmt, PPS)            [function, total]
               | emit(KItem, PPS)          [function, total]
               | ppDecl(String, Set, Int, Ty, List, PPS) [function, total]
               | ppRegs(String, Int, Int, Int, PPS)       [function, total]

  rule pp(S:Stmt Ss:Stmts, P) => pp(Ss, pp1(S, P))
  rule pp(.Stmts, P) => P

  rule emit(ins(G, O, Os), pps(C, I, L, RW, SH)) => pps(C[I <- ins(rnG(L, G), O, vexp(RW, rnOs(L, Os)))], I +Int 1, L, RW, SH)
  rule emit(X, pps(C, I, L, RW, SH)) => pps(C[I <- X], I +Int 1, L, RW, SH) [owise]

  rule pp1(N:Name :, pps(C, I, L, RW, SH)) => pps(C, I, L[Name2String(N) <- I], RW, SH)
  rule pp1(N:Name Os:Operands ;, P) => emit(ins(noG, decodeOpc(Name2String(N)), Os), P)
  rule pp1(N:QName Os:Operands ;, P) => emit(ins(noG, decodeOpc(QName2String(N)), Os), P)
  rule pp1(@ R:Reg N:QName Os:Operands ;, P) => emit(ins(g(Reg2String(R), false), decodeOpc(QName2String(N)), Os), P)
  rule pp1(@ ! R:Reg N:QName Os:Operands ;, P) => emit(ins(g(Reg2String(R), true), decodeOpc(QName2String(N)), Os), P)
  rule pp1(@ G:Name N:QName Os:Operands ;, P) => emit(ins(g(Name2String(G), false), decodeOpc(QName2String(N)), Os), P)
  rule pp1(@ ! G:Name N:QName Os:Operands ;, P) => emit(ins(g(Name2String(G), true), decodeOpc(QName2String(N)), Os), P)
  rule pp1(@ R:Reg N:Name Os:Operands ;, P) =>
       emit(ins(g(Reg2String(R), false), decodeOpc(Name2String(N)), Os), P)
  rule pp1(@ ! R:Reg N:Name Os:Operands ;, P) =>
       emit(ins(g(Reg2String(R), true), decodeOpc(Name2String(N)), Os), P)
  rule pp1(@ G:Name N:Name Os:Operands ;, P) =>
       emit(ins(g(Name2String(G), false), decodeOpc(Name2String(N)), Os), P)
  rule pp1(@ ! G:Name N:Name Os:Operands ;, P) =>
       emit(ins(g(Name2String(G), true), decodeOpc(Name2String(N)), Os), P)
  rule pp1({ Ss:Stmts }, pps(C, I, L, RW, SH)) => leaveBlock(pp(Ss, pps(C, I, L["@depth" <- {L["@depth"] orDefault 0}:>Int +Int 1], RW, SH)), L)
  rule pp1(.pragma _ ;, P) => P
  rule pp1(.loc _:Int _:Int _:Int _:LocOpts, P) => P
  rule pp1(_:Name : .callprototype ( _:ParamDecls ) _:Name ( _:ParamDecls ) _:PerfDirs ;, P) => P
  rule pp1(_:Name : .callprototype _:Name ( _:ParamDecls ) _:PerfDirs ;, P) => P
  rule pp1(_:Name : .callprototype _:Name ;, P) => P
  rule pp1(_:Name : .callprototype ( _:ParamDecls ) _:Name ;, P) => P
  rule pp1(.loc_intermediate _:Int _:Int _:Int, P) => P
  rule pp1(_:DAttrs _:TexKind _:DItems ;, P) => P      // textures: see task 35
  // target lists are kept in the label table under "@calltargets:L" / "@branchtargets:L"
  rule pp1(N:Name : .calltargets Ns:Names ;, pps(C, I, L, RW, SH)) =>
       pps(C, I, L[("@calltargets:" +String Name2String(N)) <- namesL(Ns)], RW, SH)
  rule pp1(N:Name : .branchtargets Ns:Names ;, pps(C, I, L, RW, SH)) =>
       pps(C, I, L[("@branchtargets:" +String Name2String(N)) <- namesL(Ns)], RW, SH)
  rule pp1(A1:DAttrs T:Type A2:DAttrs Is:DItems ;, P) =>
       ppDecl(dspace(aset(A1) |Set aset(A2)), aset(A1) |Set aset(A2),
              alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), P)
  rule pp1(A1:DAttrs T:Type A2:DAttrs Is:DItems = I:InitVal ;, pps(C, PC, L, RW, SH)) =>
       pps(C, PC, L, RW, SH sdecls(aset(A1) |Set aset(A2), alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), initOf(I)))
    requires dspace(aset(A1) |Set aset(A2)) in (SetItem("shared") SetItem("global") SetItem("const"))
  rule pp1(A1:DAttrs T:Type A2:DAttrs Is:DItems = _ ;, P) =>
       ppDecl(dspace(aset(A1) |Set aset(A2)), aset(A1) |Set aset(A2),
              alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), P) [owise]

  // registers: remember declared widths (%r<4> declares %r0 .. %r3)
  rule ppDecl("reg", S, AL, T, ListItem(I:DItem) Is, pps(C, PC, L, RW, SH)) =>
       ppDecl("reg", S, AL, T, ListItem(renamedItem(I, {L["@renid"] orDefault 1}:>Int)) Is,
              pps(C, PC, L[("@ren:" +String itemName(I)) <- {L["@renid"] orDefault 1}:>Int]["@renid" <- {L["@renid"] orDefault 1}:>Int +Int 1], RW, SH))
    requires isPlain(I) andBool {L["@depth"] orDefault 0}:>Int >Int 0 andBool itemName(I) in_keys(RW)
  rule ppDecl("reg", S, AL, T, ListItem(R:Reg < N:Int >) Is, P) =>
       ppDecl("reg", S, AL, T, Is, ppRegs(Reg2String(R), 0, N, wid(T), P)) requires vwidth(S) ==Int 0
  rule ppDecl("reg", S, AL, T, ListItem(R:Name < N:Int >) Is, P) =>
       ppDecl("reg", S, AL, T, Is, ppRegs(Name2String(R), 0, N, wid(T), P)) requires vwidth(S) ==Int 0
  // vector register arrays: .reg .v2 .b16 x<4>;
  rule ppDecl("reg", S, AL, T, ListItem(R:Reg < N:Int >) Is, pps(C, PC, L, RW, SH)) =>
       ppDecl("reg", S, AL, T, Is, pps(C, PC, L, vdeclareN(RW, Reg2String(R), wid(T), vwidth(S), 0, N), SH)) requires vwidth(S) >Int 0
  rule ppDecl("reg", S, AL, T, ListItem(R:Name < N:Int >) Is, pps(C, PC, L, RW, SH)) =>
       ppDecl("reg", S, AL, T, Is, pps(C, PC, L, vdeclareN(RW, Name2String(R), wid(T), vwidth(S), 0, N), SH)) requires vwidth(S) >Int 0
  syntax Map ::= vdeclareN(Map, String, Int, Int, Int, Int) [function, total]
  rule vdeclareN(RW, B, W, K, I, N) => vdeclareN(vdeclare(RW, B +String Int2String(I), W, K, 0), B, W, K, I +Int 1, N) requires I <Int N
  rule vdeclareN(RW, _, _, _, I, N) => RW requires I >=Int N
  rule ppDecl("reg", S, AL, T, ListItem(I:DItem) Is, pps(C, PC, L, RW, SH)) =>
       ppDecl("reg", S, AL, T, Is, pps(C, PC, L, vdeclare(RW, itemName(I), wid(T), vwidth(S), 0), SH)) requires vwidth(S) >Int 0 andBool isPlain(I)
  rule ppDecl("reg", S, AL, T, ListItem(I:DItem) Is, pps(C, PC, L, RW, SH)) =>
       ppDecl("reg", S, AL, T, Is, pps(C, PC, L, RW[itemName(I) <- wid(T)], SH)) [owise]
  rule ppRegs(R, I, N, W, pps(C, PC, L, RW, SH)) =>
       ppRegs(R, I +Int 1, N, W, pps(C, PC, L, RW[R +String Int2String(I) <- W], SH))
    requires I <Int N
  rule ppRegs(_, _, _, _, P) => P [owise]
  // frame-allocated variables
  rule ppDecl(SP, S, AL, T, ListItem(I:DItem) Is, P) =>
       ppDecl(SP, S, AL, T, Is, emit(declVar(SP, itemName(I), itemSize(I, T), AL), P))
    requires SP ==String "local" orBool SP ==String "param"
  // statically allocated variables
  rule ppDecl(SP, S, AL, T, Is, pps(C, PC, L, RW, SH)) =>
       pps(C, PC, L, RW, SH sdecls(S, AL, T, Is, .List))
    requires (SP ==String "shared" orBool SP ==String "global" orBool SP ==String "const")
     andBool Is =/=K .List
  rule ppDecl(_, _, _, _, .List, P) => P
endmodule

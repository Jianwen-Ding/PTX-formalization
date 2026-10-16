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
  rule itemName(N:Name [ _ ])    => Name2String(N)
  rule itemName(N:Name [ ])      => Name2String(N)
  rule itemName(N:Name [ _ ] [ _ ]) => Name2String(N)

  syntax Int ::= itemSize(DItem, Ty) [function, total]
  rule itemSize(_:Name [ N ], T)        => N *Int bytes(T)
  rule itemSize(_:Name [ ], _)          => 0
  rule itemSize(_:Name [ N ] [ M ], T)  => N *Int M *Int bytes(T)
  rule itemSize(_, T)                   => bytes(T) [owise]

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
  rule pdecls(.ParamDecls) => .List

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

  rule emit(X, pps(C, I, L, RW, SH)) => pps(C[I <- X], I +Int 1, L, RW, SH)

  rule pp1(N:Name :, pps(C, I, L, RW, SH)) => pps(C, I, L[Name2String(N) <- I], RW, SH)
  rule pp1(N:Name Os:Operands ;, P) => emit(ins(noG, decodeOpc(Name2String(N)), Os), P)
  rule pp1(@ R:Reg N:Name Os:Operands ;, P) =>
       emit(ins(g(Reg2String(R), false), decodeOpc(Name2String(N)), Os), P)
  rule pp1(@ ! R:Reg N:Name Os:Operands ;, P) =>
       emit(ins(g(Reg2String(R), true), decodeOpc(Name2String(N)), Os), P)
  rule pp1({ Ss:Stmts }, P) => pp(Ss, P)
  rule pp1(.pragma _ ;, P) => P
  rule pp1(.loc _ _ _, P) => P
  rule pp1(A1:DAttrs T:Type A2:DAttrs Is:DItems ;, P) =>
       ppDecl(dspace(aset(A1) |Set aset(A2)), aset(A1) |Set aset(A2),
              alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), P)
  rule pp1(A1:DAttrs T:Type A2:DAttrs Is:DItems = _ ;, P) =>
       ppDecl(dspace(aset(A1) |Set aset(A2)), aset(A1) |Set aset(A2),
              alignOr(A1, A2, typeOf(T)), typeOf(T), itemsL(Is), P)

  // registers: remember declared widths (%r<4> declares %r0 .. %r3)
  rule ppDecl("reg", S, AL, T, ListItem(R:Reg < N:Int >) Is, P) =>
       ppDecl("reg", S, AL, T, Is, ppRegs(Reg2String(R), 0, N, wid(T), P))
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

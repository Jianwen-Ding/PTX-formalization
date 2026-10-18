// =============================================================================
// PTX-EXEC-ISA: optional legality check of instructions against the declared
// `.version` and `.target` (task 37).  Enabled by the harness directive
// `.isacheck ;`.  The minimum version / sm / arch-specific requirement of each
// base instruction comes from the generated ptx-isa-table.k (mined from the PTX
// ISA document); qualifier-level requirements are checked from the sentences of the document that name them (scripts/gen_isa_table.py).
// =============================================================================
requires "ptx-exec-core.k"
requires "ptx-isa-table.k"

module PTX-EXEC-ISA
  imports PTX-EXEC-CORE
  imports PTX-ISA-TABLE

  // "sm_90a" -> 90 ; suffix "a"/"f"/""
  syntax Int ::= targetSm(List) [function, total]
  syntax String ::= targetArch(List) [function, total]
  rule targetSm(ListItem(S:String) Rest) => String2Int(substrString(S, 3, findChar(S, 3))) requires lengthString(S) >=Int 5 andBool substrString(S, 0, 3) ==String "sm_"
  rule targetSm(ListItem(_) Rest) => targetSm(Rest) [owise]
  rule targetSm(_) => 0 [owise]
  rule targetArch(ListItem(S:String) Rest) => substrString(S, findChar(S, 3), lengthString(S)) requires lengthString(S) >=Int 5 andBool substrString(S, 0, 3) ==String "sm_"
  rule targetArch(ListItem(_) Rest) => targetArch(Rest) [owise]
  rule targetArch(_) => "" [owise]
  // index of the first non-digit at or after I
  syntax Int ::= findChar(String, Int) [function, total]
  rule findChar(S, I) => I requires I >=Int lengthString(S) orBool notBool isDigitCh(substrString(S, I, I +Int 1))
  rule findChar(S, I) => findChar(S, I +Int 1) [owise]
  syntax Bool ::= isDigitCh(String) [function, total]
  rule isDigitCh(C) => C >=String "0" andBool C <=String "9"

  // names an instruction carries: its modifiers and the names of its types
  syntax String ::= tyName(Ty) [function, total]
  rule tyName(ty("p", _)) => "pred"
  rule tyName(ty("bf", 16)) => "bf16"
  rule tyName(ty("tf", 32)) => "tf32"
  rule tyName(ty(K, W)) => K requires lengthString(K) >Int 1 andBool K =/=String "bf" andBool K =/=String "tf"
  rule tyName(ty(K, W)) => K +String Int2String(W) [owise]
  syntax Set ::= insNames(Set, List) [function, total]
  rule insNames(M, ListItem(T:Ty) Ts) => SetItem(tyName(T)) insNames(M, Ts)
  rule insNames(M, ListItem(_) Ts) => insNames(M, Ts) [owise]
  rule insNames(M, .List) => M
  // row qualifiers "a+b" all present
  syntax Bool ::= qualsHold(String, Set) [function, total]
  rule qualsHold("", _) => true
  rule qualsHold(Q, N) => Q in N requires Q =/=String "" andBool findString(Q, "+", 0) <Int 0
  rule qualsHold(Q, N) => substrString(Q, 0, findString(Q, "+", 0)) in N andBool qualsHold(substrString(Q, findString(Q, "+", 0) +Int 1, lengthString(Q)), N)
    requires findString(Q, "+", 0) >=Int 0

  syntax String ::= isaBad(List, Set, Int, Int, String) [function, total]
  rule isaBad(ListItem(isaRow(Q, V, S, A)) Rest, N, Ver, Sm, Arch) =>
         "instruction needs PTX ISA " +String Int2String(V /Int 10) +String "." +String Int2String(V %Int 10)
    requires qualsHold(Q, N) andBool Ver >Int 0 andBool V >Int Ver
  rule isaBad(ListItem(isaRow(Q, V, S, A)) Rest, N, Ver, Sm, Arch) =>
         "instruction needs target sm_" +String Int2String(S)
    requires qualsHold(Q, N) andBool notBool (Ver >Int 0 andBool V >Int Ver) andBool Sm >Int 0 andBool S >Int Sm
  rule isaBad(ListItem(isaRow(Q, V, S, A)) Rest, N, Ver, Sm, Arch) =>
         "instruction needs an architecture-specific target (sm_" +String Int2String(S) +String "a)"
    requires qualsHold(Q, N) andBool notBool (Ver >Int 0 andBool V >Int Ver) andBool notBool (Sm >Int 0 andBool S >Int Sm) andBool A andBool Sm >Int 0 andBool Arch ==String ""
  rule isaBad(ListItem(_) Rest, N, Ver, Sm, Arch) => isaBad(Rest, N, Ver, Sm, Arch) [owise]
  rule isaBad(.List, _, _, _, _) => ""

  // features the specification deprecated and later removed (Release Notes: Deprecated / Removed Features)
  syntax String ::= isaRemoved(String, Set, Set, Int, Int) [function, total]     // opcode, modifiers, names, .version, sm
  rule isaRemoved(OP, M, N, Ver, Sm) => OP +String " without .sync was removed for sm_70 and higher (PTX ISA 6.4)"
    requires (OP ==String "shfl" orBool OP ==String "vote") andBool notBool "sync" in M andBool Sm >=Int 70 andBool Ver >=Int 64
  rule isaRemoved(OP, M, N, Ver, Sm) => OP +String ".satfinite on floating-point wmma.mma was removed in PTX ISA 6.5"
    requires OP ==String "wmma" andBool "mma" in M andBool "satfinite" in M andBool ("f16" in N orBool "f32" in N) andBool Ver >=Int 65
  rule isaRemoved(OP, M, N, Ver, Sm) => OP +String ".min/.max on f32 does not exist (removed from the specification)"
    requires (OP ==String "atom" orBool OP ==String "red") andBool ("min" in M orBool "max" in M) andBool "f32" in N
  rule isaRemoved(OP, M, N, Ver, Sm) => "suld.p does not exist (removed from the specification)" requires OP ==String "suld" andBool "p" in M
  rule isaRemoved(OP, M, N, Ver, Sm) => "sust.p with a 32-bit type does not exist (removed from the specification)"
    requires OP ==String "sust" andBool "p" in M andBool ("u32" in N orBool "s32" in N orBool "f32" in N)
  rule isaRemoved(_, _, _, _, _) => "" [owise]

  // static check at launch: every instruction of every function against the declared .version / .target
  syntax KItem ::= "#isaChk"
  syntax String ::= isaScanF(List, Map, Int, Int, String) [function, total]
                  | isaScanC(List, Map, Int, Int, String) [function, total]
  rule isaScanF(.List, _, _, _, _) => ""
  rule isaScanF(ListItem(F) Fs, FM, V, Sm, Ar) => #if isaScanC(keys_list(codeOf(FM, F)), codeOf(FM, F), V, Sm, Ar) ==String ""
                                                  #then isaScanF(Fs, FM, V, Sm, Ar) #else isaScanC(keys_list(codeOf(FM, F)), codeOf(FM, F), V, Sm, Ar) #fi
  rule isaScanC(.List, _, _, _, _) => ""
  rule isaScanC(ListItem(K) Ks, CM, V, Sm, Ar) => #if isaIns({CM[K]}:>KItem, V, Sm, Ar) ==String ""
                                                  #then isaScanC(Ks, CM, V, Sm, Ar) #else isaIns({CM[K]}:>KItem, V, Sm, Ar) #fi
  syntax String ::= isaIns(KItem, Int, Int, String) [function, total]
  rule isaIns(ins(_, opc(OP, M, Ts), _), V, Sm, Ar) => #if isaRemoved(OP, M, insNames(M, Ts), V, Sm) ==String "" #then isaBad(isaRows(OP), insNames(M, Ts), V, Sm, Ar) #else isaRemoved(OP, M, insNames(M, Ts), V, Sm) #fi
  rule isaIns(_, _, _, _) => "" [owise]
  syntax Map ::= codeOf(Map, String) [function, total]
  rule codeOf(FM, F) => code({FM[F]}:>FDef) requires F in_keys(FM) andBool isFDef(FM[F])
  rule codeOf(_, _) => .Map [owise]
  syntax Map ::= code(FDef) [function, total]
  rule code(fdef(_, _, _, C, _, _)) => C
  rule <k> #isaChk => .K ... </k> <isachk> false </isachk>
  rule <k> #isaChk => #fault(isaScanF(keys_list(FM), FM, V, Sm, Ar), 0) ... </k>
       <isachk> true </isachk> <funcs> FM </funcs> <isaver> V </isaver> <isasm> Sm </isasm> <isaarch> Ar </isaarch>
    requires isaScanF(keys_list(FM), FM, V, Sm, Ar) =/=String ""
  rule <k> #isaChk => .K ... </k>
       <isachk> true </isachk> <funcs> FM </funcs> <isaver> V </isaver> <isasm> Sm </isasm> <isaarch> Ar </isaarch>
    requires isaScanF(keys_list(FM), FM, V, Sm, Ar) ==String ""
endmodule

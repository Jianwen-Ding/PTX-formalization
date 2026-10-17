// =============================================================================
// PTX-EXEC-ISA: optional legality check of instructions against the declared
// `.version` and `.target` (task 37).  Enabled by the harness directive
// `.isacheck ;`.  The minimum version / sm / arch-specific requirement of each
// base instruction comes from the generated ptx-isa-table.k (mined from the PTX
// ISA document); qualifier-level requirements are not checked.
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

  syntax String ::= isaBad(List, Int, Int, String) [function, total]
  rule isaBad(ListItem(isaRow(_, V, S, A)) _, Ver, Sm, Arch) =>
         "instruction needs PTX ISA " +String Int2String(V /Int 10) +String "." +String Int2String(V %Int 10)
    requires Ver >Int 0 andBool V >Int Ver
  rule isaBad(ListItem(isaRow(_, V, S, A)) _, Ver, Sm, Arch) =>
         "instruction needs target sm_" +String Int2String(S)
    requires notBool (Ver >Int 0 andBool V >Int Ver) andBool Sm >Int 0 andBool S >Int Sm
  rule isaBad(ListItem(isaRow(_, V, S, A)) _, Ver, Sm, Arch) =>
         "instruction needs an architecture-specific target (sm_" +String Int2String(S) +String "a)"
    requires notBool (Ver >Int 0 andBool V >Int Ver) andBool notBool (Sm >Int 0 andBool S >Int Sm) andBool A andBool Sm >Int 0 andBool Arch ==String ""
  rule isaBad(_, _, _, _) => "" [owise]

  // static check at launch: every instruction of every function against the declared .version / .target
  syntax KItem ::= #isaChk
  syntax String ::= isaScanF(List, Map, Int, Int, String) [function, total]
                  | isaScanC(List, Map, Int, Int, String) [function, total]
  rule isaScanF(.List, _, _, _, _) => ""
  rule isaScanF(ListItem(F) Fs, FM, V, Sm, Ar) => #if isaScanC(keys_list(codeOf(FM, F)), codeOf(FM, F), V, Sm, Ar) ==String ""
                                                  #then isaScanF(Fs, FM, V, Sm, Ar) #else isaScanC(keys_list(codeOf(FM, F)), codeOf(FM, F), V, Sm, Ar) #fi
  rule isaScanC(.List, _, _, _, _) => ""
  rule isaScanC(ListItem(K) Ks, CM, V, Sm, Ar) => #if isaIns({CM[K]}:>KItem, V, Sm, Ar) ==String ""
                                                  #then isaScanC(Ks, CM, V, Sm, Ar) #else isaIns({CM[K]}:>KItem, V, Sm, Ar) #fi
  syntax String ::= isaIns(KItem, Int, Int, String) [function, total]
  rule isaIns(ins(_, opc(OP, _, _), _), V, Sm, Ar) => isaBad(isaRows(OP), V, Sm, Ar)
  rule isaIns(_, _, _, _) => "" [owise]
  syntax Map ::= codeOf(Map, String) [function, total]
  rule codeOf(FM, F) => code({FM[F]}:>FDef) requires F in_keys(FM) andBool isFDef(FM[F])
  rule codeOf(_, _) => .Map [owise]
  syntax Map ::= code(FDef) [function, total]
  rule code(fdef(_, _, _, C, _, _)) => C
  syntax Bool ::= isFDef(KItem) [function, total]
  rule isFDef(_:FDef) => true
  rule isFDef(_) => false [owise]
  rule <k> #isaChk => .K ... </k> <isachk> false </isachk>
  rule <k> #isaChk => #fault(isaScanF(keys_list(FM), FM, V, Sm, Ar), 0) ... </k>
       <isachk> true </isachk> <funcs> FM </funcs> <isaver> V </isaver> <isasm> Sm </isasm> <isaarch> Ar </isaarch>
    requires isaScanF(keys_list(FM), FM, V, Sm, Ar) =/=String ""
  rule <k> #isaChk => .K ... </k>
       <isachk> true </isachk> <funcs> FM </funcs> <isaver> V </isaver> <isasm> Sm </isasm> <isaarch> Ar </isaarch>
    requires isaScanF(keys_list(FM), FM, V, Sm, Ar) ==String ""
endmodule

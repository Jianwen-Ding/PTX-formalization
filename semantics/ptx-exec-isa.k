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

endmodule

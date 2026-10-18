// =============================================================================
// PTX-SYNTAX: concrete syntax of (a large subset of) NVIDIA PTX ISA, plus a
// small host-side "harness" language used to allocate buffers, launch kernels
// and inspect results.
//
// Design notes
//  * Opcodes are lexed as a single dotted token (e.g. `ld.global.nc.f32`).
//    The semantics splits the token into a base name, a set of modifiers and
//    a list of types (see PTX-INSTR-DECODE in ptx-common.k).  This keeps the
//    grammar small while covering the combinatorial space of PTX modifiers.
//  * Registers (`%r1`, `%tid.x`, ...) and identifiers (`$L__BB0_2`,
//    `_Z6vecAddPfS_S_i`, ...) are distinct token sorts.
//  * Decimal integers and floats reuse K's builtin Int / Float tokens.
//    `[%rd1+4]` therefore lexes as Reg followed by the signed Int `+4`.
// =============================================================================

module PTX-SYNTAX
  imports INT-SYNTAX
  imports FLOAT-SYNTAX
  imports STRING-SYNTAX

  // ---------------------------------------------------------------- lexical
  // whitespace, // line comments and /* block comments */
  syntax #Layout ::= r"(\\/\\*([^\\*]|(\\*+([^\\*\\/])))*\\*+\\/)|(\\/\\/[^\\n\\r]*)|([\\ \\n\\r\\t])"

  // Identifiers; dotted forms are used for opcodes (add.s32, ld.global.u32 ...)
  syntax Name   ::= r"[a-zA-Z_$][a-zA-Z0-9_$]*(\\.[a-zA-Z0-9_$]+)*" [token]
  // opcodes with `::` qualifiers (ld.shared::cta.u32, mbarrier.arrive.expect_tx.shared::cta.b64): a
  // separate token that needs a lower-case start and at least one `::`, so that K's own `X::Sort`
  // variable syntax in the builtin modules is not affected
  syntax QName  ::= r"[a-z][a-zA-Z0-9_]*(\\.[a-zA-Z0-9_]+)*::[a-zA-Z0-9_]+(:[0-9]+)*((\\.|::)[a-zA-Z0-9_]+(:[0-9]+)*)*" [token]
  // Registers, including special registers such as %tid.x or %ctaid.y
  syntax Reg    ::= r"%[a-zA-Z_$][a-zA-Z0-9_$]*(\\.[a-zA-Z][a-zA-Z0-9]*)?" [token]
  // 0x1F (hex integer), 0b101 (binary), 0f3F800000 (f32 bit pattern),
  // 0d3FF0000000000000 (f64 bit pattern)
  syntax HexLit ::= r"0[xXfFdDbB][0-9a-fA-F]+[uU]?"                  [token]
  // debug section names
  syntax SecName ::= r"\\.debug_[a-z_0-9]+"                          [token]

  // ---------------------------------------------------------------- types
  syntax Type ::= ".u8"  | ".u16" | ".u32" | ".u64"
                | ".s8"  | ".s16" | ".s32" | ".s64"
                | ".b8"  | ".b16" | ".b32" | ".b64"
                | ".f16" | ".f32" | ".f64" | ".pred"
                | ".b128" | ".bf16" | ".tf32"
                | ".f16x2" | ".bf16x2" | ".s16x2" | ".u16x2"
                | ".e4m3"   | ".e5m2"                          // fp8
                | ".e4m3x2" | ".e5m2x2" | ".e4m3x4" | ".e5m2x4"
                | ".e2m3x2" | ".e3m2x2" | ".e2m3x4" | ".e3m2x4" // fp6
                | ".e2m1x2" | ".e2m1x4"                         // fp4
                | ".ue8m0x2"
                | ".u4" | ".s4" | ".u2" | ".s2" | ".b1"

  // ---------------------------------------------------------------- operands
  syntax Addr ::= Reg | Name | Int
                | Reg Int            // [%rd1+4]  (Int token carries the sign)
                | Name Int           // [param+0]
                | Reg "+" Int        // [%rd1 + 4]
                | Name "+" Int

  syntax Operand ::= Reg
                   | Int | Float | HexLit
                   | Name                      // symbol, label, function, `_`
                   | "[" Addr "]"              // memory address
                   | "[" Addr "," Operand "]"             // [tex, {x,y}] / [surf, {x}]
                   | "[" Addr "," Addr "," Operand "]"    // [tex, smp, {x,y}]
                   | "[" Addr "," Addr "," Operand "," Operand "]"                                   // [tmap, gAddr, {size}, {coords}]  (.override::global_dim)
                   | "[" Addr "," Addr "," Operand "," Operand "," Operand "," Operand "]"           // [tmap, gAddr, {size}, {lower}, upper, {coords}]
                   | "{" Operands "}"          // vector operand
                   | "(" Operands ")"          // call argument / return list
                   | "{" Operands "}" "|" Reg     // tex: vector destination with residency predicate
                   | "{" Operands "}" "|" Name
                   | Reg "|" Reg               // setp dual destination
                   | Name "|" Name             // same, for registers declared without %
                   | "-" Reg                   // negated operand (vmad)
                   | "-" Name
                   | "!" Reg                   // negated predicate source
                   | "!" Name
  syntax Operands ::= List{Operand, ","}

  // ---------------------------------------------------------------- statements
  syntax Guard ::= "@" Reg | "@" "!" Reg | "@" Name | "@" "!" Name
  syntax Instr ::= Name Operands ";"
                 | QName Operands ";"

  syntax DAttr  ::= ".visible" | ".extern" | ".weak" | ".common"
                  | ".global" | ".shared" | ".local" | ".const" | ".param" | ".reg"
                  | ".align" Int | ".ptr"
                  | ".v2" | ".v4" | ".v8"
                  | ".attribute" "(" ".managed" ")"
  syntax DAttrs ::= List{DAttr, ""}
  syntax TexKind ::= ".texref" | ".samplerref" | ".surfref"

  syntax DItem  ::= Name | Reg
                  | Reg "<" Int ">"           // %r<10>  declares %r0 .. %r9
                  | Name "<" Int ">"          // tmp<4>  declares tmp0 .. tmp3
                  | Name ArrDims              // arrays of any rank
  syntax ArrDims ::= "[" Int "]" ArrDims | "[" Int "]" | "[" "]"
  syntax DItems ::= NeList{DItem, ","}

  syntax InitVal  ::= Int | Float | HexLit | Name | "{" InitVals "}"
                    | Name Int                 // sym+4 (sign lexed into the Int)
                    | Name "+" Int            // sym + 4
                    | Name "-" Name            // label difference (debug sections)
                    | "generic" "(" Name ")"   // generic address of a symbol

  syntax InitVals ::= List{InitVal, ","}

  syntax Decl ::= DAttrs Type DAttrs DItems ";"
                | DAttrs Type DAttrs DItems "=" InitVal ";"
                | DAttrs TexKind DItems ";"            // .global .texref t;

  syntax Stmt  ::= Instr
                 | Guard Instr
                 | Name ":"                   // label
                 | Decl
                 | "{" Stmts "}"              // nested scope
                 | ".pragma" String ";"
                 | ".loc" Int Int Int LocOpts
                 | Name ":" ".callprototype" "(" ParamDecls ")" Name "(" ParamDecls ")" PerfDirs ";"
                 | Name ":" ".callprototype" Name "(" ParamDecls ")" PerfDirs ";"
                 | Name ":" ".calltargets" Names ";"
                 | Name ":" ".branchtargets" Names ";"
  syntax LocOpt  ::= "prologue_end" | "epilogue_begin" | "is_stmt" Int
                   | "," "function_name" Name
                   | "," "function_name" Name "-" Name
                   | "," "function_name" Name "+" Int
                   | "," "inlined_at" Int Int Int
  syntax LocOpts ::= List{LocOpt, ""}
  syntax Stmts ::= List{Stmt, ""}

  // ---------------------------------------------------------------- functions
  syntax ParamDecl  ::= DAttrs Type DAttrs DItem
  syntax ParamDecls ::= List{ParamDecl, ","}

  syntax Ints     ::= NeList{Int, ","}
  syntax PerfDir  ::= ".maxntid" Ints | ".reqntid" Ints | ".minnctapersm" Int
                    | ".maxnctapersm" Int | ".maxnreg" Int | ".noreturn"
                    | ".maxclusterrank" Int | ".reqnctapercluster" Ints
                    | ".explicitcluster" | ".blocksareclusters"
                    | ".abi_preserve" Int | ".abi_preserve_control" Int
                    | ".pragma" String
  syntax PerfDirs ::= List{PerfDir, ""}

  syntax FuncHead ::= ".entry" Name "(" ParamDecls ")"
                    | ".entry" Name
                    | ".func" Name "(" ParamDecls ")"
                    | ".func" Name
                    | ".func" "(" ParamDecls ")" Name "(" ParamDecls ")"
                    | ".func" "(" ParamDecls ")" Name

  syntax FuncDef ::= DAttrs FuncHead PerfDirs "{" Stmts "}"
                   | DAttrs FuncHead PerfDirs ";"          // prototype

  // ---------------------------------------------------------------- harness
  // Host-side directives (not PTX): allocate global buffers, launch kernels,
  // print buffers and check expected results.
  syntax Dim ::= Int | "(" Int "," Int ")" | "(" Int "," Int "," Int ")"

  syntax HarnessDir ::= ".buffer" Name Type "[" Int "]" ";"
                      | ".buffer" Name Type "=" "{" InitVals "}" ";"
                      | ".buffer" Name Type "[" Int "]" "=" "{" InitVals "}" ";"
                      | ".launch" Name "<<<" Dim "," Dim ">>>" "(" Operands ")" ";"
                      | ".launch" Name "<<<" Dim "," Dim "," Int ">>>" "(" Operands ")" ";"
                      | ".print" Name Type ";"
                      | ".print" Name Type Int ";"
                      | ".expect" Name Type "=" "{" InitVals "}" ";"
                      | ".expect" Name Type "@" Int "=" "{" InitVals "}" ";"     // from element offset
                      | ".patch" Name Type "@" Int "=" "{" InitVals "}" ";"      // write values at an element offset
                      | ".tensormap" Name Type Name ".dims" "{" Ints "}" ".strides" "{" Ints "}" ".box" "{" Ints "}" ".swizzle" Int ";"
                      | ".tensormap" Name Type Name ".dims" "{" Ints "}" ".strides" "{" Ints "}" ".box" "{" Ints "}" ".swizzle" Int
                        ".im2col" "{" Ints "}" "{" Ints "}" ".estrides" "{" Ints "}" ";"
                      | ".texture" Name Type Name ".dims" "{" Ints "}" ".attrs" "{" Ints "}" ";"
                      | ".sampler" Name ".attrs" "{" Ints "}" ";"
                      | ".surface" Name Type Name ".dims" "{" Ints "}" ".attrs" "{" Ints "}" ";"
                      | ".multimem" Name ".size" Int "{" Names "}" ";"
                      | ".memmodel" Name ";"
                      | ".isacheck" ";"
                      | ".append" Name Type "=" "{" InitVals "}" ";"            // extend the most recent buffer (keeps text small)

  // ---------------------------------------------------------------- modules
  syntax Names   ::= NeList{Name, ","}
  syntax SecItem  ::= Type InitVals | Name ":"
  syntax SecItems ::= List{SecItem, ""}
  syntax TopDecl ::= ".version" Float
                   | ".target" Names
                   | ".address_size" Int
                   | ".file" Int String
                   | ".file" Int String "," Int "," Int
                   | ".alias" Name "," Name ";"
                   | ".section" SecName "{" SecItems "}"
                   | ".pragma" String ";"
                   | Decl
                   | FuncDef
                   | HarnessDir
  syntax Module  ::= List{TopDecl, ""}
endmodule

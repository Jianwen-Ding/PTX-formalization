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
  // Registers, including special registers such as %tid.x or %ctaid.y
  syntax Reg    ::= r"%[a-zA-Z_$][a-zA-Z0-9_$]*(\\.[a-zA-Z]+)?"      [token]
  // 0x1F (hex integer), 0f3F800000 (f32 bit pattern), 0d3FF0000000000000 (f64)
  syntax HexLit ::= r"0[xXfFdD][0-9a-fA-F]+"                        [token]

  // ---------------------------------------------------------------- types
  syntax Type ::= ".u8"  | ".u16" | ".u32" | ".u64"
                | ".s8"  | ".s16" | ".s32" | ".s64"
                | ".b8"  | ".b16" | ".b32" | ".b64"
                | ".f16" | ".f32" | ".f64" | ".pred"

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
                   | "{" Operands "}"          // vector operand
                   | "(" Operands ")"          // call argument / return list
                   | Reg "|" Reg               // setp dual destination
                   | "!" Reg                   // negated predicate source
  syntax Operands ::= List{Operand, ","}

  // ---------------------------------------------------------------- statements
  syntax Guard ::= "@" Reg | "@" "!" Reg
  syntax Instr ::= Name Operands ";"

  syntax DAttr  ::= ".visible" | ".extern" | ".weak" | ".common"
                  | ".global" | ".shared" | ".local" | ".const" | ".param" | ".reg"
                  | ".align" Int | ".ptr"
  syntax DAttrs ::= List{DAttr, ""}

  syntax DItem  ::= Name | Reg
                  | Reg "<" Int ">"           // %r<10>  declares %r0 .. %r9
                  | Name "[" Int "]"
                  | Name "[" "]"
                  | Name "[" Int "]" "[" Int "]"
  syntax DItems ::= NeList{DItem, ","}

  syntax InitVal  ::= Int | Float | HexLit | Name | "{" InitVals "}"
  syntax InitVals ::= List{InitVal, ","}

  syntax Decl ::= DAttrs Type DAttrs DItems ";"
                | DAttrs Type DAttrs DItems "=" InitVal ";"

  syntax Stmt  ::= Instr
                 | Guard Instr
                 | Name ":"                   // label
                 | Decl
                 | "{" Stmts "}"              // nested scope
                 | ".pragma" String ";"
                 | ".loc" Int Int Int
  syntax Stmts ::= List{Stmt, ""}

  // ---------------------------------------------------------------- functions
  syntax ParamDecl  ::= DAttrs Type DAttrs DItem
  syntax ParamDecls ::= List{ParamDecl, ","}

  syntax Ints     ::= NeList{Int, ","}
  syntax PerfDir  ::= ".maxntid" Ints | ".reqntid" Ints | ".minnctapersm" Int
                    | ".maxnctapersm" Int | ".maxnreg" Int | ".noreturn"
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

  // ---------------------------------------------------------------- modules
  syntax Names   ::= NeList{Name, ","}
  syntax TopDecl ::= ".version" Float
                   | ".target" Names
                   | ".address_size" Int
                   | ".file" Int String
                   | ".file" Int String "," Int "," Int
                   | Decl
                   | FuncDef
                   | HarnessDir
  syntax Module  ::= List{TopDecl, ""}
endmodule

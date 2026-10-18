// =============================================================================
// AGX: an executable K semantics of a subset of the Apple G13 (M1) GPU ISA.
//
// Source: the public reverse-engineering documentation by dougallj (applegpu); see docs/agx-mapping.md.  The model is one
// SIMD-group of 32 threads executed in lock step with a program counter, a 32-bit execution mask and the execution-mask stack
// kept in the low half of r0 (r0l), exactly as the documents describe.  Instructions with semantics written out in the documents
// (mov, get_sr, iadd, imadd, bitop, bfi, asr, popcount, bitrev, ffs, fadd, fmul, fmadd, icmpsel, fcmpsel, if/else/while
// compares, pop_exec, jumps, ballots, shuffles, wait) follow them; the memory instructions follow the prose descriptions.
// Anything the documents leave as TODO is not modelled.  Float arithmetic reuses the IEEE core of the PTX semantics (PTX-FLOAT):
// f32 results flush subnormals to zero (RoundToFloat32Wrapper(flush_to_zero=True)); f16 results do not.
//
// Assembly dialect (statements end with ';', labels start with an upper-case letter):
//     r5 = 32-bit register, r5l / r5h = 16-bit halves, u3 = uniform register; `.discard` / `.cache` hints are accepted and ignored.
// Test directives:  .all r3 = 7;  .laneid r4;  .lane 5 r3 = 9;  .uni u2 = 1;  .sr 80 = laneid;  .dev 16 = 0x12345678;  .run;
//                   .expectall r3 = 7;  .expectlane 5 r3 = 9;  .expectdev 16 = 305419896;  .expectmask Int (exec mask)
// =============================================================================
requires "../semantics/ptx-float.k"

module AGX-SYNTAX
  imports DOMAINS-SYNTAX
  imports FLOAT-SYNTAX

  syntax #Layout ::= r"(\\/\\*([^\\*]|(\\*+([^\\*\\/])))*\\*+\\/)|(\\/\\/[^\\n\\r]*)|([\\ \\n\\r\\t])"
  syntax AReg   ::= r"[ru][0-9]+[lh]?(\\.discard|\\.cache)?" [token, prec(2)]
  syntax AUPair ::= r"u[0-9]+_u[0-9]+" [token, prec(3)]
  syntax ALabel ::= r"[A-Z][A-Za-z0-9_]*" [token, prec(2)]
  syntax ACc    ::= "eq" | "ne" | "ult" | "ule" | "ugt" | "uge" | "slt" | "sle" | "sgt" | "sge"
                 | "feq" | "fne" | "flt" | "fle" | "fgt" | "fge"
  syntax AFmt   ::= "u8" | "u16" | "u32"
  syntax AOperand ::= AReg | Int | Float

  syntax AInstr ::= "mov" AReg "," AOperand
                 | "get_sr" AReg "," Int
                 | "iadd" AReg "," AOperand "," AOperand
                 | "isub" AReg "," AOperand "," AOperand
                 | "imadd" AReg "," AOperand "," AOperand "," AOperand
                 | "bitop" Int AReg "," AOperand "," AOperand
                 | "bfi" Int AReg "," AOperand "," AOperand "," AOperand
                 | "asr" AReg "," AOperand "," AOperand
                 | "popcount" AReg "," AOperand
                 | "bitrev" AReg "," AOperand
                 | "ffs" AReg "," AOperand
                 | "fadd" AReg "," AOperand "," AOperand
                 | "fmul" AReg "," AOperand "," AOperand
                 | "fmadd" AReg "," AOperand "," AOperand "," AOperand
                 | "fadd16" AReg "," AOperand "," AOperand
                 | "fmul16" AReg "," AOperand "," AOperand
                 | "fmadd16" AReg "," AOperand "," AOperand "," AOperand
                 | "rcp" AReg "," AOperand
                 | "icmpsel" ACc AReg "," AOperand "," AOperand "," AOperand "," AOperand
                 | "fcmpsel" ACc AReg "," AOperand "," AOperand "," AOperand "," AOperand
                 | "if_icmp" ACc AOperand "," AOperand
                 | "else_icmp" ACc AOperand "," AOperand
                 | "while_icmp" ACc AOperand "," AOperand
                 | "if_fcmp" ACc AOperand "," AOperand
                 | "else_fcmp" ACc AOperand "," AOperand
                 | "while_fcmp" ACc AOperand "," AOperand
                 | "pop_exec" Int
                 | "jmp_exec_any" ALabel
                 | "jmp_exec_none" ALabel
                 | "icmp_ballot" ACc AReg "," AOperand "," AOperand
                 | "fcmp_ballot" ACc AReg "," AOperand "," AOperand
                 | "simd_shuffle" AReg "," AOperand "," AOperand
                 | "simd_shuffle_xor" AReg "," AOperand "," AOperand
                 | "simd_shuffle_up" AReg "," AOperand "," AOperand
                 | "simd_shuffle_down" AReg "," AOperand "," AOperand
                 | "device_load" AFmt Int AReg "," AUPair "," AOperand "," Int
                 | "device_store" AFmt Int AReg "," AUPair "," AOperand "," Int
                 | "threadgroup_load" AFmt Int AReg "," AOperand "," Int
                 | "threadgroup_store" AFmt Int AReg "," AOperand "," Int
                 | "wait" Int
                 | "threadgroup_barrier"
                 | "stop"

  syntax ADirective ::= ".all" AReg "=" Int
                     | ".laneid" AReg
                     | ".lane" Int AReg "=" Int
                     | ".uni" AReg "=" Int
                     | ".sr" Int "=" "laneid"
                     | ".dev" Int "=" Int
                     | ".run"
                     | ".expectall" AReg "=" Int
                     | ".expectlane" Int AReg "=" Int
                     | ".expectdev" Int "=" Int
                     | ".expectmask" Int

  syntax AStmt  ::= ALabel ":" | AInstr | ADirective
  syntax AProgram ::= List{AStmt, ";"}
endmodule

module AGX
  imports AGX-SYNTAX
  imports DOMAINS
  imports PTX-FLOAT

  configuration
    <agx>
      <k> $PGM:AProgram </k>
      <code> .Map </code>              // pc |-> AInstr
      <labels> .Map </labels>          // ALabel |-> pc
      <loadpc> 0 </loadpc>
      <pc> 0 </pc>
      <em> 4294967295 </em>            // execution mask
      <gpr> .Map </gpr>               // lane * 256 + register |-> 32-bit value
      <upr> .Map </upr>               // uniform register |-> 32-bit value
      <dmem> .Map </dmem>             // device memory: byte address |-> byte
      <tmem> .Map </tmem>             // threadgroup memory
      <srs> .Map </srs>               // special register * 32 + lane |-> value
      <pend> .List </pend>            // loads not yet waited for: ld(lane, register, value)
      <status> "loading" </status>
      <exit-code exit=""> 0 </exit-code>
      <msg> "" </msg>
    </agx>

  // ---------------------------------------------------------------- register naming
  syntax String ::= AReg2String(AReg) [function, total, hook(STRING.token2string)]
  syntax Int ::= numEnd(String, Int) [function, total]
  rule numEnd(S, I) => I requires I >=Int lengthString(S) orBool substrString(S, I, I +Int 1) <String "0" orBool substrString(S, I, I +Int 1) >String "9"
  rule numEnd(S, I) => numEnd(S, I +Int 1) [owise]
  syntax Int ::= regNum(AReg) [function, total] | regHalf(AReg) [function, total] | regBits(AReg) [function, total]
  rule regNum(R) => String2Int(substrString(AReg2String(R), 1, numEnd(AReg2String(R), 1)))
  rule regHalf(R) => #if numEnd(AReg2String(R), 1) >=Int lengthString(AReg2String(R)) #then 0
                     #else #if substrString(AReg2String(R), numEnd(AReg2String(R), 1), numEnd(AReg2String(R), 1) +Int 1) ==String "l" #then 1
                     #else #if substrString(AReg2String(R), numEnd(AReg2String(R), 1), numEnd(AReg2String(R), 1) +Int 1) ==String "h" #then 2 #else 0 #fi #fi #fi
  rule regBits(R) => #if regHalf(R) ==Int 0 #then 32 #else 16 #fi
  syntax Bool ::= isU(AReg) [function, total]
  rule isU(R) => substrString(AReg2String(R), 0, 1) ==String "u"

  syntax Int ::= m32() [function, total] | m16() [function, total]
  rule m32() => 4294967295
  rule m16() => 65535

  // raw register value (32-bit container) and the view selected by the name
  syntax Int ::= raw(Map, Map, Int, AReg) [function, total]
  rule raw(G, U, T, R) => #if isU(R) #then {U[regNum(R)] orDefault 0}:>Int #else {G[T *Int 256 +Int regNum(R)] orDefault 0}:>Int #fi
  syntax Int ::= val(Map, Map, Int, AReg) [function, total]
  rule val(G, U, T, R) => #if regHalf(R) ==Int 0 #then raw(G, U, T, R) #else #if regHalf(R) ==Int 1 #then raw(G, U, T, R) &Int m16() #else (raw(G, U, T, R) >>Int 16) &Int m16() #fi #fi
  syntax Map ::= put(Map, Int, AReg, Int) [function, total]            // lane, destination (general register), value (truncated to the view)
  rule put(G, T, R, V) => G[T *Int 256 +Int regNum(R) <- #if regHalf(R) ==Int 0 #then V &Int m32()
                          #else #if regHalf(R) ==Int 1 #then ({G[T *Int 256 +Int regNum(R)] orDefault 0}:>Int &Int 4294901760) |Int (V &Int m16())
                          #else ({G[T *Int 256 +Int regNum(R)] orDefault 0}:>Int &Int m16()) |Int ((V &Int m16()) <<Int 16) #fi #fi]

  // operands: registers, integer immediates (broadcast), float immediates (encoded at the destination's width)
  syntax Int ::= opv(Map, Map, Int, AOperand, Int) [function, total]   // ..., lane, operand, destination bits
  rule opv(G, U, T, R:AReg, _) => val(G, U, T, R)
  rule opv(_, _, _, I:Int, B) => I &Int amask(B)
  rule opv(_, _, _, F:Float, B) => encodeF(F, B)
  syntax Int ::= amask(Int) [function, total]
  rule amask(B) => (1 <<Int B) -Int 1

  // ---------------------------------------------------------------- loading the program
  rule <k> L:ALabel : ; Rest => Rest ... </k> <labels> M => M[L <- PC] </labels> <loadpc> PC </loadpc> <status> "loading" </status>
  rule <k> L:ALabel : => .K ... </k> <labels> M => M[L <- PC] </labels> <loadpc> PC </loadpc> <status> "loading" </status>
  rule <k> I:AInstr ; Rest => Rest ... </k> <code> C => C[PC <- I] </code> <loadpc> PC => PC +Int 1 </loadpc> <status> "loading" </status>
  rule <k> I:AInstr => .K ... </k> <code> C => C[PC <- I] </code> <loadpc> PC => PC +Int 1 </loadpc> <status> "loading" </status>
  rule <k> D:ADirective ; Rest => D ~> Rest ... </k>
  rule <k> .AProgram => .K ... </k>
  rule <k> (.K => .K) ~> .AProgram ... </k> [owise]

  // ---------------------------------------------------------------- test directives
  rule <k> .all R = V => .K ... </k> <gpr> G => setAll(G, 0, R, V) </gpr>
  syntax Map ::= setAll(Map, Int, AReg, Int) [function, total]
  rule setAll(G, T, R, V) => setAll(put(G, T, R, V), T +Int 1, R, V) requires T <Int 32
  rule setAll(G, _, _, _) => G [owise]
  rule <k> .laneid R => .K ... </k> <gpr> G => setLanes(G, 0, R) </gpr>
  syntax Map ::= setLanes(Map, Int, AReg) [function, total]
  rule setLanes(G, T, R) => setLanes(put(G, T, R, T), T +Int 1, R) requires T <Int 32
  rule setLanes(G, _, _) => G [owise]
  rule <k> .lane T R = V => .K ... </k> <gpr> G => put(G, T, R, V) </gpr>
  rule <k> .uni R = V => .K ... </k> <upr> U => U[regNum(R) <- V &Int m32()] </upr>
  rule <k> .sr N = laneid => .K ... </k> <srs> S => setSr(S, N, 0) </srs>
  syntax Map ::= setSr(Map, Int, Int) [function, total]
  rule setSr(S, N, T) => setSr(S[N *Int 32 +Int T <- T], N, T +Int 1) requires T <Int 32
  rule setSr(S, _, _) => S [owise]
  rule <k> .dev A = V => .K ... </k> <dmem> M => wrBytes(M, A, V, 4) </dmem>

  rule <k> .run => #run ... </k> <status> "loading" => "running" </status>
  syntax KItem ::= "#run"
  rule <k> #run => #exec(I) ~> #run ... </k> <code> ... PC |-> I ... </code> <pc> PC => PC +Int 1 </pc> <status> "running" </status>
  rule <k> #run => .K ... </k> <status> "stopped" </status>
  rule <k> #run => #fatal("fell off the end of the program") ... </k> <code> C </code> <pc> PC </pc> <status> "running" </status> requires notBool PC in_keys(C)
  syntax KItem ::= #fatal(String)
  rule <k> #fatal(S) ~> _ => .K </k> <msg> _ => S </msg> <exit-code> _ => 1 </exit-code> <status> _ => "fault" </status>

  rule <k> .expectall R = V => .K ... </k> <gpr> G </gpr> <upr> U </upr>
    requires allEq(G, U, 0, R, V)
  rule <k> .expectall R = V => #fatal("expectall " +String AReg2String(R) +String " = " +String Int2String(V) +String " failed") ... </k> <gpr> G </gpr> <upr> U </upr>
    requires notBool allEq(G, U, 0, R, V)
  syntax Bool ::= allEq(Map, Map, Int, AReg, Int) [function, total]
  rule allEq(G, U, T, R, V) => val(G, U, T, R) ==Int (V &Int amask(regBits(R))) andBool allEq(G, U, T +Int 1, R, V) requires T <Int 32
  rule allEq(_, _, _, _, _) => true [owise]
  rule <k> .expectlane T R = V => .K ... </k> <gpr> G </gpr> <upr> U </upr> requires val(G, U, T, R) ==Int (V &Int amask(regBits(R)))
  rule <k> .expectlane T R = V => #fatal("expectlane " +String Int2String(T) +String " " +String AReg2String(R) +String " got " +String Int2String(val(G, U, T, R))) ... </k> <gpr> G </gpr> <upr> U </upr>
    requires val(G, U, T, R) =/=Int (V &Int amask(regBits(R)))
  rule <k> .expectdev A = V => .K ... </k> <dmem> M </dmem> requires rdBytes(M, A, 4) ==Int (V &Int m32())
  rule <k> .expectdev A = V => #fatal("expectdev " +String Int2String(A) +String " got " +String Int2String(rdBytes(M, A, 4))) ... </k> <dmem> M </dmem> requires rdBytes(M, A, 4) =/=Int (V &Int m32())
  rule <k> .expectmask V => .K ... </k> <em> E </em> requires E ==Int V
  rule <k> .expectmask V => #fatal("exec mask is " +String Int2String(E)) ... </k> <em> E </em> requires E =/=Int V

  // ---------------------------------------------------------------- bytes
  syntax Map ::= wrBytes(Map, Int, Int, Int) [function, total]       // memory, address, value, byte count
  rule wrBytes(M, A, V, N) => wrBytes(M[A <- V &Int 255], A +Int 1, V >>Int 8, N -Int 1) requires N >Int 0
  rule wrBytes(M, _, _, _) => M [owise]
  syntax Int ::= rdBytes(Map, Int, Int) [function, total]
  rule rdBytes(M, A, N) => ({M[A] orDefault 0}:>Int) |Int (rdBytes(M, A +Int 1, N -Int 1) <<Int 8) requires N >Int 0
  rule rdBytes(_, _, _) => 0 [owise]

  // ---------------------------------------------------------------- lane loop
  syntax ALaneOp ::= lane1(String, AReg, AOperand) | lane2(String, AReg, AOperand, AOperand) | lane3(String, AReg, AOperand, AOperand, AOperand)
                  | lane5(String, AReg, AOperand, AOperand, AOperand, AOperand) | laneI(Int, String, AReg, AOperand, AOperand) | laneI3(Int, String, AReg, AOperand, AOperand, AOperand)
  syntax Int ::= opFn(ALaneOp, Map, Map, Int) [function, total]       // result for one lane (reads the original state)
  syntax Map ::= lanes(Map, Map, Map, Int, Int, ALaneOp) [function, total]   // original G, U, new G, exec mask, lane, operation
  rule lanes(G0, U, G, EM, T, OP) => lanes(G0, U, #if ((EM >>Int T) &Int 1) ==Int 1 #then put(G, T, dstOf(OP), opFn(OP, G0, U, T)) #else G #fi, EM, T +Int 1, OP)
    requires T <Int 32
  rule lanes(_, _, G, _, _, _) => G [owise]
  syntax AReg ::= dstOf(ALaneOp) [function, total]
  rule dstOf(lane1(_, D, _)) => D
  rule dstOf(lane2(_, D, _, _)) => D
  rule dstOf(lane3(_, D, _, _, _)) => D
  rule dstOf(lane5(_, D, _, _, _, _)) => D
  rule dstOf(laneI(_, _, D, _, _)) => D
  rule dstOf(laneI3(_, _, D, _, _, _)) => D
  rule <k> #lanes(OP) => .K ... </k> <gpr> G => lanes(G, U, G, EM, 0, OP) </gpr> <upr> U </upr> <em> EM </em>

  syntax KItem ::= #exec(AInstr) | #lanes(ALaneOp)

  // ---------------------------------------------------------------- moves and integer arithmetic
  rule <k> #exec(mov D, A) => #lanes(lane1("mov", D, A)) ... </k>
  rule <k> #exec(get_sr D, N) => #lanes(lane1("sr", D, N)) ... </k>
  rule <k> #exec(iadd D, A, B) => #lanes(lane2("iadd", D, A, B)) ... </k>
  rule <k> #exec(isub D, A, B) => #lanes(lane2("isub", D, A, B)) ... </k>
  rule <k> #exec(imadd D, A, B, C) => #lanes(lane3("imadd", D, A, B, C)) ... </k>
  rule <k> #exec(bitop TT D, A, B) => #lanes(laneI(TT, "bitop", D, A, B)) ... </k>
  rule <k> #exec(bfi M D, A, B, C) => #lanes(laneI3(M, "bfi", D, A, B, C)) ... </k>
  rule <k> #exec(asr D, A, B) => #lanes(lane2("asr", D, A, B)) ... </k>
  rule <k> #exec(popcount D, A) => #lanes(lane1("popcount", D, A)) ... </k>
  rule <k> #exec(bitrev D, A) => #lanes(lane1("bitrev", D, A)) ... </k>
  rule <k> #exec(ffs D, A) => #lanes(lane1("ffs", D, A)) ... </k>
  rule <k> #exec(fadd D, A, B) => #lanes(lane2("fadd", D, A, B)) ... </k>
  rule <k> #exec(fmul D, A, B) => #lanes(lane2("fmul", D, A, B)) ... </k>
  rule <k> #exec(fmadd D, A, B, C) => #lanes(lane3("fmadd", D, A, B, C)) ... </k>
  rule <k> #exec(fadd16 D, A, B) => #lanes(lane2("fadd16", D, A, B)) ... </k>
  rule <k> #exec(fmul16 D, A, B) => #lanes(lane2("fmul16", D, A, B)) ... </k>
  rule <k> #exec(fmadd16 D, A, B, C) => #lanes(lane3("fmadd16", D, A, B, C)) ... </k>
  rule <k> #exec(rcp D, A) => #lanes(lane1("rcp", D, A)) ... </k>
  rule <k> #exec(icmpsel C D, A, B, X, Y) => #lanes(lane5("icmpsel " +String ccName(C), D, A, B, X, Y)) ... </k>
  rule <k> #exec(fcmpsel C D, A, B, X, Y) => #lanes(lane5("fcmpsel " +String ccName(C), D, A, B, X, Y)) ... </k>
  rule <k> #exec(simd_shuffle D, A, B) => #lanes(lane2("shuf", D, A, B)) ... </k>
  rule <k> #exec(simd_shuffle_xor D, A, B) => #lanes(lane2("shufxor", D, A, B)) ... </k>
  rule <k> #exec(simd_shuffle_up D, A, B) => #lanes(lane2("shufup", D, A, B)) ... </k>
  rule <k> #exec(simd_shuffle_down D, A, B) => #lanes(lane2("shufdown", D, A, B)) ... </k>

  syntax String ::= ccName(ACc) [function, total]
  rule ccName(eq) => "eq"   rule ccName(ne) => "ne"   rule ccName(ult) => "ult" rule ccName(ule) => "ule" rule ccName(ugt) => "ugt" rule ccName(uge) => "uge"
  rule ccName(slt) => "slt" rule ccName(sle) => "sle" rule ccName(sgt) => "sgt" rule ccName(sge) => "sge"
  rule ccName(feq) => "feq" rule ccName(fne) => "fne" rule ccName(flt) => "flt" rule ccName(fle) => "fle" rule ccName(fgt) => "fgt" rule ccName(fge) => "fge"

  // destination width
  syntax Int ::= db(AReg) [function, total]
  rule db(D) => regBits(D)
  syntax Set ::= FZ() [function, total]
  rule FZ() => SetItem("ftz")

  rule opFn(lane1("mov", D, A), G, U, T) => opv(G, U, T, A, db(D))
  rule opFn(lane1("sr", D, N:Int), G, U, T) => 0   // overridden below through the special-register rule
  rule opFn(lane2("iadd", D, A, B), G, U, T) => (opv(G, U, T, A, db(D)) +Int opv(G, U, T, B, db(D))) &Int amask(db(D))
  rule opFn(lane2("isub", D, A, B), G, U, T) => (opv(G, U, T, A, db(D)) -Int opv(G, U, T, B, db(D))) &Int amask(db(D))
  rule opFn(lane3("imadd", D, A, B, C), G, U, T) => (opv(G, U, T, A, db(D)) *Int opv(G, U, T, B, db(D)) +Int opv(G, U, T, C, db(D))) &Int amask(db(D))
  rule opFn(laneI(TT, "bitop", D, A, B), G, U, T) => bitopv(TT, opv(G, U, T, A, db(D)), opv(G, U, T, B, db(D)), db(D))
  rule opFn(laneI3(M, "bfi", D, A, B, C), G, U, T) => bfiv(M, opv(G, U, T, A, 32), opv(G, U, T, B, 32), opv(G, U, T, C, 32))
  rule opFn(lane2("asr", D, A, B), G, U, T) => (asext(opv(G, U, T, A, db(D)), db(D)) >>Int (opv(G, U, T, B, db(D)) &Int 127)) &Int amask(db(D))
  rule opFn(lane1("popcount", D, A), G, U, T) => popc(opv(G, U, T, A, 32), 0)
  rule opFn(lane1("bitrev", D, A), G, U, T) => brev(opv(G, U, T, A, 32), 0, 0)
  rule opFn(lane1("ffs", D, A), G, U, T) => ffsv(opv(G, U, T, A, 32), 31)
  rule opFn(lane2("fadd", D, A, B), G, U, T) => fpAdd(FZ(), 32, opv(G, U, T, A, 32), opv(G, U, T, B, 32))
  rule opFn(lane2("fmul", D, A, B), G, U, T) => fpMul(FZ(), 32, opv(G, U, T, A, 32), opv(G, U, T, B, 32))
  rule opFn(lane3("fmadd", D, A, B, C), G, U, T) => fpFma(FZ(), 32, opv(G, U, T, A, 32), opv(G, U, T, B, 32), opv(G, U, T, C, 32))
  rule opFn(lane2("fadd16", D, A, B), G, U, T) => fpAdd(.Set, 16, opv(G, U, T, A, 16), opv(G, U, T, B, 16))
  rule opFn(lane2("fmul16", D, A, B), G, U, T) => fpMul(.Set, 16, opv(G, U, T, A, 16), opv(G, U, T, B, 16))
  rule opFn(lane3("fmadd16", D, A, B, C), G, U, T) => fpFma(.Set, 16, opv(G, U, T, A, 16), opv(G, U, T, B, 16), opv(G, U, T, C, 16))
  rule opFn(lane1("rcp", D, A), G, U, T) => fpDiv(FZ(), 32, 1065353216, opv(G, U, T, A, 32))
  rule opFn(lane5(S, D, A, B, X, Y), G, U, T) => #if cmpv(S, opv(G, U, T, A, 32), opv(G, U, T, B, 32)) #then opv(G, U, T, X, db(D)) #else opv(G, U, T, Y, db(D)) #fi
  rule opFn(lane2("shuf", D, A, B), G, U, T) => #if opv(G, U, T, B, 32) <Int 32 #then val(G, U, opv(G, U, T, B, 32), srcReg(A)) #else 0 #fi
  rule opFn(lane2("shufxor", D, A, B), G, U, T) => val(G, U, T xorInt (opv(G, U, T, B, 32) &Int 31), srcReg(A))
  rule opFn(lane2("shufup", D, A, B), G, U, T) => #if T -Int opv(G, U, T, B, 32) >=Int 0 #then val(G, U, T -Int opv(G, U, T, B, 32), srcReg(A)) #else val(G, U, T, srcReg(A)) #fi
  rule opFn(lane2("shufdown", D, A, B), G, U, T) => #if T +Int opv(G, U, T, B, 32) <Int 32 #then val(G, U, T +Int opv(G, U, T, B, 32), srcReg(A)) #else val(G, U, T, srcReg(A)) #fi
  rule opFn(_, _, _, _) => 0 [owise]
  syntax AReg ::= srcReg(AOperand) [function]
  rule srcReg(R:AReg) => R
  // get_sr needs the special-register map: handled with a dedicated rule
  rule <k> #exec(get_sr D, N) => .K ... </k> <gpr> G => srLoop(G, S, EM, 0, D, N) </gpr> <srs> S </srs> <em> EM </em> [priority(40)]
  syntax Map ::= srLoop(Map, Map, Int, Int, AReg, Int) [function, total]
  rule srLoop(G, S, EM, T, D, N) => srLoop(#if ((EM >>Int T) &Int 1) ==Int 1 #then put(G, T, D, {S[N *Int 32 +Int T] orDefault 0}:>Int) #else G #fi, S, EM, T +Int 1, D, N) requires T <Int 32
  rule srLoop(G, _, _, _, _, _) => G [owise]

  // integer helpers
  syntax Int ::= asext(Int, Int) [function, total]
  rule asext(V, W) => #if V &Int (1 <<Int (W -Int 1)) ==Int 0 #then V &Int amask(W) #else (V &Int amask(W)) -Int (1 <<Int W) #fi
  syntax Int ::= bitopv(Int, Int, Int, Int) [function, total]
  rule bitopv(TT, A, B, W) => (#if (TT &Int 1) =/=Int 0 #then (~Int A) &Int (~Int B) #else 0 #fi
                           |Int #if (TT &Int 2) =/=Int 0 #then A &Int (~Int B) #else 0 #fi
                           |Int #if (TT &Int 4) =/=Int 0 #then (~Int A) &Int B #else 0 #fi
                           |Int #if (TT &Int 8) =/=Int 0 #then A &Int B #else 0 #fi) &Int amask(W)
  syntax Int ::= bfiv(Int, Int, Int, Int) [function, total]
  rule bfiv(M, A, B, C) => ((A &Int ~Int ((#if M ==Int 0 #then 4294967295 #else amask(M) #fi) <<Int (C &Int 127))) |Int ((B &Int (#if M ==Int 0 #then 4294967295 #else amask(M) #fi)) <<Int (C &Int 127))) &Int 4294967295
  syntax Int ::= popc(Int, Int) [function, total] | brev(Int, Int, Int) [function, total] | ffsv(Int, Int) [function, total]
  rule popc(V, I) => (V >>Int I) &Int 1 +Int popc(V, I +Int 1) requires I <Int 32
  rule popc(_, _) => 0 [owise]
  rule brev(V, I, R) => brev(V, I +Int 1, R |Int (((V >>Int I) &Int 1) <<Int (31 -Int I))) requires I <Int 32
  rule brev(_, _, R) => R [owise]
  rule ffsv(V, I) => #if I <Int 0 #then 4294967295 #else #if (V >>Int I) &Int 1 ==Int 1 #then I #else ffsv(V, I -Int 1) #fi #fi

  // comparison conditions (integer: unsigned / signed views of the 32-bit values; float: ordered comparisons on f32 bits)
  syntax Bool ::= cmpv(String, Int, Int) [function, total]
  rule cmpv(S, A, B) => A ==Int B requires endsWith(S, "eq") andBool notBool startsWith(S, "f")
  rule cmpv(S, A, B) => A =/=Int B requires endsWith(S, "ne") andBool notBool startsWith(S, "f")
  rule cmpv(S, A, B) => A <Int B requires endsWith(S, "ult")
  rule cmpv(S, A, B) => A <=Int B requires endsWith(S, "ule")
  rule cmpv(S, A, B) => A >Int B requires endsWith(S, "ugt")
  rule cmpv(S, A, B) => A >=Int B requires endsWith(S, "uge")
  rule cmpv(S, A, B) => asext(A, 32) <Int asext(B, 32) requires endsWith(S, "slt")
  rule cmpv(S, A, B) => asext(A, 32) <=Int asext(B, 32) requires endsWith(S, "sle")
  rule cmpv(S, A, B) => asext(A, 32) >Int asext(B, 32) requires endsWith(S, "sgt")
  rule cmpv(S, A, B) => asext(A, 32) >=Int asext(B, 32) requires endsWith(S, "sge")
  rule cmpv(S, A, B) => fcmpv(S, decodeF(A, 32), decodeF(B, 32)) requires startsWith(S, "fcmpsel") orBool startsWith(S, "f")
  rule cmpv(_, _, _) => false [owise]
  syntax Bool ::= fcmpv(String, Float, Float) [function, total]
  rule fcmpv(S, X, Y) => false requires isNaN(X) orBool isNaN(Y)
  rule fcmpv(S, X, Y) => X ==Float Y requires notBool (isNaN(X) orBool isNaN(Y)) andBool endsWith(S, "feq")
  rule fcmpv(S, X, Y) => X =/=Float Y requires notBool (isNaN(X) orBool isNaN(Y)) andBool endsWith(S, "fne")
  rule fcmpv(S, X, Y) => X <Float Y requires notBool (isNaN(X) orBool isNaN(Y)) andBool endsWith(S, "flt")
  rule fcmpv(S, X, Y) => X <=Float Y requires notBool (isNaN(X) orBool isNaN(Y)) andBool endsWith(S, "fle")
  rule fcmpv(S, X, Y) => X >Float Y requires notBool (isNaN(X) orBool isNaN(Y)) andBool endsWith(S, "fgt")
  rule fcmpv(S, X, Y) => X >=Float Y requires notBool (isNaN(X) orBool isNaN(Y)) andBool endsWith(S, "fge")
  rule fcmpv(_, _, _) => false [owise]
  syntax Bool ::= endsWith(String, String) [function, total] | startsWith(String, String) [function, total]
  rule endsWith(S, E) => lengthString(S) >=Int lengthString(E) andBool substrString(S, lengthString(S) -Int lengthString(E), lengthString(S)) ==String E
  rule startsWith(S, P) => lengthString(S) >=Int lengthString(P) andBool substrString(S, 0, lengthString(P)) ==String P

  // ---------------------------------------------------------------- execution mask stack (r0l) and flow control
  syntax KItem ::= #stackOp(String, String, AOperand, AOperand, Int)
  rule <k> #exec(if_icmp C A, B) => #stackOp("if", "icmpsel " +String ccName(C), A, B, 1) ... </k>
  rule <k> #exec(else_icmp C A, B) => #stackOp("else", "icmpsel " +String ccName(C), A, B, 1) ... </k>
  rule <k> #exec(while_icmp C A, B) => #stackOp("while", "icmpsel " +String ccName(C), A, B, 1) ... </k>
  rule <k> #exec(if_fcmp C A, B) => #stackOp("if", "fcmpsel " +String ccName(C), A, B, 1) ... </k>
  rule <k> #exec(else_fcmp C A, B) => #stackOp("else", "fcmpsel " +String ccName(C), A, B, 1) ... </k>
  rule <k> #exec(while_fcmp C A, B) => #stackOp("while", "fcmpsel " +String ccName(C), A, B, 1) ... </k>
  rule <k> #exec(pop_exec N) => #stackOp("pop", "", 0, 0, N) ... </k>
  rule <k> #stackOp(K, S, A, B, N) => .K ... </k> <gpr> G => stk(G, U, 0, K, S, A, B, N) </gpr> <upr> U </upr> <em> _ => stkEm(G, U, 0, K, S, A, B, N, 0) </em>
  // r0l of lane t
  syntax Int ::= r0l(Map, Int) [function, total]
  rule r0l(G, T) => {G[T *Int 256] orDefault 0}:>Int &Int m16()
  syntax Int ::= stkNew(Int, String, Bool, Int) [function, total]     // v, kind, comparison result, n
  rule stkNew(V, "pop", _, N) => #if V -Int N <Int 0 #then 0 #else V -Int N #fi
  rule stkNew(V, "if", C, N) => #if V =/=Int 0 #then V +Int N #else #if C #then 0 #else 1 #fi #fi
  rule stkNew(V, "while", C, N) => #if V <Int N #then (#if C #then 0 #else N #fi) #else V #fi
  rule stkNew(V, "else", C, N) => #if V ==Int 0 #then N #else #if V ==Int 1 #then (#if C #then 0 #else 1 #fi) #else V #fi #fi
  rule stkNew(V, _, _, _) => V [owise]
  syntax Map ::= stk(Map, Map, Int, String, String, AOperand, AOperand, Int) [function, total]
  rule stk(G, U, T, K, S, A, B, N) => stk(G[T *Int 256 <- ({G[T *Int 256] orDefault 0}:>Int &Int 4294901760) |Int stkNew(r0l(G, T), K, K ==String "pop" orBool cmpv(S, opv(G, U, T, A, 32), opv(G, U, T, B, 32)), N)],
                                          U, T +Int 1, K, S, A, B, N) requires T <Int 32
  rule stk(G, _, _, _, _, _, _, _) => G [owise]
  // the new execution mask is computed from the new r0l values: lane t is active iff r0l == 0
  syntax Int ::= stkEm(Map, Map, Int, String, String, AOperand, AOperand, Int, Int) [function, total]
  rule stkEm(G, U, T, K, S, A, B, N, EM) => stkEm(G, U, T +Int 1, K, S, A, B, N, EM |Int (#if stkNew(r0l(G, T), K, K ==String "pop" orBool cmpv(S, opv(G, U, T, A, 32), opv(G, U, T, B, 32)), N) ==Int 0 #then 1 <<Int T #else 0 #fi)) requires T <Int 32
  rule stkEm(_, _, _, _, _, _, _, _, EM) => EM [owise]

  rule <k> #exec(jmp_exec_any L) => .K ... </k> <em> EM </em> <labels> ... L |-> P ... </labels> <pc> PC => #if EM =/=Int 0 #then P #else PC #fi </pc>
  rule <k> #exec(jmp_exec_none L) => .K ... </k> <em> EM </em> <labels> ... L |-> P ... </labels> <pc> PC => #if EM ==Int 0 #then P #else PC #fi </pc>
  rule <k> #exec(stop) => .K ... </k> <status> _ => "stopped" </status>
  rule <k> #exec(threadgroup_barrier) => .K ... </k>

  // ---------------------------------------------------------------- ballots (result broadcast to the active lanes)
  rule <k> #exec(icmp_ballot C D, A, B) => .K ... </k> <gpr> G => bcast(G, EM, 0, D, ballot(G, U, EM, 0, "icmpsel " +String ccName(C), A, B, 0)) </gpr> <upr> U </upr> <em> EM </em>
  rule <k> #exec(fcmp_ballot C D, A, B) => .K ... </k> <gpr> G => bcast(G, EM, 0, D, ballot(G, U, EM, 0, "fcmpsel " +String ccName(C), A, B, 0)) </gpr> <upr> U </upr> <em> EM </em>
  syntax Int ::= ballot(Map, Map, Int, Int, String, AOperand, AOperand, Int) [function, total]
  rule ballot(G, U, EM, T, S, A, B, R) => ballot(G, U, EM, T +Int 1, S, A, B,
        R |Int #if ((EM >>Int T) &Int 1) ==Int 1 andBool cmpv(S, opv(G, U, T, A, 32), opv(G, U, T, B, 32)) #then 1 <<Int T #else 0 #fi) requires T <Int 32
  rule ballot(_, _, _, _, _, _, _, R) => R [owise]
  syntax Map ::= bcast(Map, Int, Int, AReg, Int) [function, total]
  rule bcast(G, EM, T, D, V) => bcast(#if ((EM >>Int T) &Int 1) ==Int 1 #then put(G, T, D, V) #else G #fi, EM, T +Int 1, D, V) requires T <Int 32
  rule bcast(G, _, _, _, _) => G [owise]

  // ---------------------------------------------------------------- memory
  // device_load: up to 4 consecutive elements per lane from base (64-bit uniform pair) + (index << (log2 size + shift)); results become
  // visible at the next `wait` (reading the destination earlier returns the old value)
  syntax Int ::= fsize(AFmt) [function, total]
  rule fsize(u8) => 1   rule fsize(u16) => 2   rule fsize(u32) => 4
  syntax Int ::= flog(AFmt) [function, total]
  rule flog(u8) => 0   rule flog(u16) => 1   rule flog(u32) => 2
  syntax Int ::= ubase(Map, AUPair) [function, total]
  syntax Int ::= upairLo(AUPair) [function, total] | upairHi(AUPair) [function, total]
  rule upairLo(P) => String2Int(substrString(AUPair2String(P), 1, findString(AUPair2String(P), "_", 0)))
  rule upairHi(P) => String2Int(substrString(AUPair2String(P), findString(AUPair2String(P), "_", 0) +Int 2, lengthString(AUPair2String(P))))
  syntax String ::= AUPair2String(AUPair) [function, total, hook(STRING.token2string)]
  rule ubase(U, P) => ({U[upairLo(P)] orDefault 0}:>Int) |Int (({U[upairHi(P)] orDefault 0}:>Int) <<Int 32)
  syntax Int ::= regAdd(AReg, Int) [function]            // register number of the i-th consecutive register is passed around as an Int
  rule <k> #exec(device_load F N D, P, I, S) => #ld(F, N, D, ubase(U, P), I, S, "dev") ... </k> <upr> U </upr>
  rule <k> #exec(threadgroup_load F N D, I, S) => #ld(F, N, D, 0, I, S, "tg") ... </k>
  rule <k> #exec(device_store F N D, P, I, S) => #st(F, N, D, ubase(U, P), I, S, "dev") ... </k> <upr> U </upr>
  rule <k> #exec(threadgroup_store F N D, I, S) => #st(F, N, D, 0, I, S, "tg") ... </k>
  syntax KItem ::= #ld(AFmt, Int, AReg, Int, AOperand, Int, String) | #st(AFmt, Int, AReg, Int, AOperand, Int, String)
  rule <k> #ld(F, N, D, B, I, S, W) => .K ... </k> <gpr> G </gpr> <upr> U </upr> <em> EM </em> <dmem> DM </dmem> <tmem> TM </tmem> <pend> P => P ldAll(G, U, EM, DM, TM, 0, F, N, D, B, I, S, W) </pend>
  rule <k> #st(F, N, D, B, I, S, W) => .K ... </k> <gpr> G </gpr> <upr> U </upr> <em> EM </em> <dmem> DM => #if W ==String "dev" #then stAll(G, U, EM, DM, 0, F, N, D, B, I, S) #else DM #fi </dmem>
       <tmem> TM => #if W ==String "tg" #then stAll(G, U, EM, TM, 0, F, N, D, B, I, S) #else TM #fi </tmem>
  // effective address of lane t
  syntax Int ::= ea(Map, Map, Int, AFmt, AOperand, Int, Int) [function, total]
  rule ea(G, U, T, F, I, S, B) => ((B +Int (opv(G, U, T, I, 32) <<Int (flog(F) +Int S))) >>Int flog(F)) <<Int flog(F)      // rounded down to the element size
  syntax List ::= ldAll(Map, Map, Int, Map, Map, Int, AFmt, Int, AReg, Int, AOperand, Int, String) [function, total]
  rule ldAll(G, U, EM, DM, TM, T, F, N, D, B, I, S, W) => (#if ((EM >>Int T) &Int 1) ==Int 1 #then ldLane(DM, TM, T, F, N, D, ea(G, U, T, F, I, S, B), 0, W) #else .List #fi)
        ldAll(G, U, EM, DM, TM, T +Int 1, F, N, D, B, I, S, W) requires T <Int 32
  rule ldAll(_, _, _, _, _, _, _, _, _, _, _, _, _) => .List [owise]
  syntax List ::= ldLane(Map, Map, Int, AFmt, Int, AReg, Int, Int, String) [function, total]
  rule ldLane(DM, TM, T, F, N, D, A, K, W) => ListItem(ld(T, regNum(D) +Int K, rdBytes(#if W ==String "dev" #then DM #else TM #fi, A +Int K *Int fsize(F), fsize(F)))) ldLane(DM, TM, T, F, N, D, A, K +Int 1, W) requires K <Int N
  rule ldLane(_, _, _, _, _, _, _, _, _) => .List [owise]
  syntax Map ::= stAll(Map, Map, Int, Map, Int, AFmt, Int, AReg, Int, AOperand, Int) [function, total]
  rule stAll(G, U, EM, M, T, F, N, D, B, I, S) => stAll(G, U, EM, #if ((EM >>Int T) &Int 1) ==Int 1 #then stLane(G, M, T, F, N, D, ea(G, U, T, F, I, S, B), 0) #else M #fi, T +Int 1, F, N, D, B, I, S) requires T <Int 32
  rule stAll(_, _, _, M, _, _, _, _, _, _, _) => M [owise]
  syntax Map ::= stLane(Map, Map, Int, AFmt, Int, AReg, Int, Int) [function, total]
  rule stLane(G, M, T, F, N, D, A, K) => stLane(G, wrBytes(M, A +Int K *Int fsize(F), {G[T *Int 256 +Int regNum(D) +Int K] orDefault 0}:>Int, fsize(F)), T, F, N, D, A, K +Int 1) requires K <Int N
  rule stLane(_, M, _, _, _, _, _, _) => M [owise]
  // wait: apply the pending loads
  rule <k> #exec(wait _) => .K ... </k> <gpr> G => applyLd(G, P) </gpr> <pend> P => .List </pend>
  syntax APendLd ::= ld(Int, Int, Int)
  syntax Map ::= applyLd(Map, List) [function, total]
  rule applyLd(G, ListItem(ld(T, R, V)) Rest) => applyLd(G[T *Int 256 +Int R <- V &Int m32()], Rest)
  rule applyLd(G, _) => G [owise]
endmodule

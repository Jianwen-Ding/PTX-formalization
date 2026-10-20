// =============================================================================
// AGX: an executable K semantics of a subset of the Apple G13 (M1) GPU ISA.
//
// Source: the public reverse-engineering documentation by dougallj (applegpu); see docs/agx-mapping.md.  The model is one
// SIMD-group of 32 threads executed in lock step with a program counter, a 32-bit execution mask and the execution-mask stack
// kept in the low half of r0 (r0l), exactly as the documents describe.  Instructions with semantics written out in the documents
// (mov, get_sr, iadd, imadd, bitop, bfi, asr, popcount, bitrev, ffs, fadd, fmul, fmadd, icmpsel, fcmpsel, if/else/while
// compares, pop_exec, jumps, ballots, shuffles, wait) follow them; the memory instructions follow the prose descriptions.
// Anything the documents leave as TODO is not modelled.  Where the documents and the hardware differ, the model follows the
// hardware as recorded by the AGX-Performance-Simulator test suite (M2 Max), whose simulator the differential test of
// sail/agx/gen/simcheck.py compares the Sail model of this file with.  Float arithmetic reuses the IEEE core of the PTX
// semantics (PTX-FLOAT): sources are binary16 (16-bit registers) or binary32 (32-bit registers, subnormals read as zero in
// the 32-bit forms); results are rounded once: binary32 results below 2^-126 become zeros before rounding, binary16
// results keep subnormals, NaNs are 0x7fc00000 / 0x7e00.
//
// Assembly dialect (statements end with ';', labels start with an upper-case letter):
//     r5 = 32-bit register, r5l / r5h = 16-bit halves, u3 = uniform register; `.discard` / `.cache` hints are accepted and ignored.
// Labels are written L_name (identifiers made only of capitals collide with K variables in the lexer).
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
  syntax ASign  ::= "signed" | "unsigned"        // memory index extension (applegpu field Ou: 0 = sign-, 1 = zero-extend)
  syntax AAop   ::= "add" | "min" | "max" | "and" | "or" | "xor" | "xchg"
  syntax AOperand ::= AReg | Int | Float

  syntax AInstr ::= "mov" AReg "," AOperand
                 | "get_sr" AReg "," Int
                 | "iadd" AReg "," AOperand "," AOperand
                 | "isub" AReg "," AOperand "," AOperand
                 | "imadd" AReg "," AOperand "," AOperand "," AOperand
                 | "bitop" Int AReg "," AOperand "," AOperand
                 | "bfi" Int AReg "," AOperand "," AOperand "," AOperand
                 | "asr" AReg "," AOperand "," AOperand
                 | "bfeil" Int AReg "," AOperand "," AOperand "," AOperand
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
                 | "device_load" AFmt Int AReg "," AUPair "," AOperand "," ASign "," Int
                 | "device_store" AFmt Int AReg "," AUPair "," AOperand "," ASign "," Int
                 | "threadgroup_load" AFmt Int AReg "," AOperand "," Int
                 | "threadgroup_store" AFmt Int AReg "," AOperand "," Int
                 | "device_atomic" AAop AReg "," AUPair "," AOperand "," AOperand
                 | "threadgroup_atomic" AAop AReg "," AOperand "," AOperand
                 | "wait" Int
                 | "threadgroup_barrier"
                 | "stop"

  syntax ADirective ::= ".all" AReg "=" Int
                     | ".laneid" AReg
                     | ".lane" Int AReg "=" Int
                     | ".uni" AReg "=" Int
                     | ".sr" Int "=" "laneid"
                     | ".srall" Int "=" Int
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
  rule <k> .srall N = V => .K ... </k> <srs> S => setSrAll(S, N, 0, V) </srs>
  syntax Map ::= setSrAll(Map, Int, Int, Int) [function, total]
  rule setSrAll(S, N, T, V) => setSrAll(S[N *Int 32 +Int T <- V], N, T +Int 1, V) requires T <Int 32
  rule setSrAll(S, _, _, _) => S [owise]
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
  rule <k> #exec(bfeil M D, A, B, C) => #lanes(laneI3(M, "bfeil", D, A, B, C)) ... </k>
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
  rule opFn(laneI3(M, "bfeil", D, A, B, C), G, U, T) => ((opv(G, U, T, A, 32) &Int ~Int (#if M ==Int 0 #then 4294967295 #else amask(M) #fi)) |Int ((opv(G, U, T, B, 32) >>Int (opv(G, U, T, C, 32) &Int 127)) &Int (#if M ==Int 0 #then 4294967295 #else amask(M) #fi))) &Int 4294967295
  // asr sign-extends A at its own size (applegpu: sign_extend(a, A.thread_bit_size))
  rule opFn(lane2("asr", D, A, B), G, U, T) => (asext(opv(G, U, T, A, osz(A)), osz(A)) >>Int (opv(G, U, T, B, 32) &Int 127)) &Int amask(db(D))
  rule opFn(lane1("popcount", D, A), G, U, T) => popc(opv(G, U, T, A, 32), 0)
  rule opFn(lane1("bitrev", D, A), G, U, T) => brev(opv(G, U, T, A, 32), 0, 0)
  rule opFn(lane1("ffs", D, A), G, U, T) => ffsv(opv(G, U, T, A, 32), 31)
  // float arithmetic: the exact result of the source values (fvalK: 16-bit registers binary16, 32-bit registers
  // binary32 with subnormals read as zero in the 32-bit forms), rounded once into the destination (agxRes)
  rule opFn(lane2("fadd", D, A, B), G, U, T) => agxRes(fvalK(G, U, T, A, false) +Float fvalK(G, U, T, B, false), D)
  rule opFn(lane2("fmul", D, A, B), G, U, T) => agxRes(fvalK(G, U, T, A, false) *Float fvalK(G, U, T, B, false), D)
  rule opFn(lane3("fmadd", D, A, B, C), G, U, T) => agxRes(fvalK(G, U, T, A, false) *Float fvalK(G, U, T, B, false) +Float fvalK(G, U, T, C, false), D)
  rule opFn(lane2("fadd16", D, A, B), G, U, T) => agxRes(fvalK(G, U, T, A, true) +Float fvalK(G, U, T, B, true), D)
  rule opFn(lane2("fmul16", D, A, B), G, U, T) => agxRes(fvalK(G, U, T, A, true) *Float fvalK(G, U, T, B, true), D)
  rule opFn(lane3("fmadd16", D, A, B, C), G, U, T) => agxRes(fvalK(G, U, T, A, true) *Float fvalK(G, U, T, B, true) +Float fvalK(G, U, T, C, true), D)
  // rcp: the correctly rounded binary32 reciprocal (fp32 subnormal sources read as zero, results below 2^-126 flushed);
  // a 16-bit destination receives that binary32 value rounded to binary16
  rule opFn(lane1("rcp", D, A), G, U, T) => agxRes(rcpK(fvalK(G, U, T, A, false)), D)
  rule opFn(lane5(S, D, A, B, X, Y), G, U, T) => #if cmpl(S, G, U, T, A, B) #then opv(G, U, T, X, db(D)) #else opv(G, U, T, Y, db(D)) #fi
  // shuffles (applegpu simd_shuffle*): every quad takes the OR q of the low two index bits of its four lanes (active or
  // not) and permutes its lanes by q; lane T with index I reads element T & 3 of the quad of lane T + I (down), T - I (up),
  // T xor I (xor), or lane q of quad I >> 2 (idx); its own A when that lane is outside the SIMD-group
  rule opFn(lane2("shuf", D, A, B), G, U, T) => #if opv(G, U, T, B, 32) <Int 32
        #then opv(G, U, 4 *Int (opv(G, U, T, B, 32) >>Int 2) +Int quadOr(G, U, opv(G, U, T, B, 32) >>Int 2, B), A, 32)
        #else opv(G, U, T, A, 32) #fi
  rule opFn(lane2("shufxor", D, A, B), G, U, T) => #if (T xorInt opv(G, U, T, B, 32)) <Int 32
        #then opv(G, U, 4 *Int ((T xorInt opv(G, U, T, B, 32)) >>Int 2) +Int ((T &Int 3) xorInt quadOr(G, U, (T xorInt opv(G, U, T, B, 32)) >>Int 2, B)), A, 32)
        #else opv(G, U, T, A, 32) #fi
  rule opFn(lane2("shufup", D, A, B), G, U, T) => #if T -Int opv(G, U, T, B, 32) >=Int 0
        #then opv(G, U, 4 *Int ((T -Int opv(G, U, T, B, 32)) >>Int 2) +Int (((T &Int 3) -Int quadOr(G, U, (T -Int opv(G, U, T, B, 32)) >>Int 2, B) +Int 4) &Int 3), A, 32)
        #else opv(G, U, T, A, 32) #fi
  rule opFn(lane2("shufdown", D, A, B), G, U, T) => #if T +Int opv(G, U, T, B, 32) <Int 32
        #then opv(G, U, 4 *Int ((T +Int opv(G, U, T, B, 32)) >>Int 2) +Int (((T &Int 3) +Int quadOr(G, U, (T +Int opv(G, U, T, B, 32)) >>Int 2, B)) &Int 3), A, 32)
        #else opv(G, U, T, A, 32) #fi
  rule opFn(_, _, _, _) => 0 [owise]
  syntax Int ::= quadOr(Map, Map, Int, AOperand) [function, total]      // OR of the low two index bits of quad Q's lanes
  rule quadOr(G, U, Q, B) => (opv(G, U, 4 *Int Q, B, 32) |Int opv(G, U, 4 *Int Q +Int 1, B, 32) |Int opv(G, U, 4 *Int Q +Int 2, B, 32)
                              |Int opv(G, U, 4 *Int Q +Int 3, B, 32)) &Int 3

  // ---------------------------------------------------------------- float sources and results
  // the value of a float source: 16-bit registers are binary16, 32-bit registers binary32 (subnormals read as zero unless
  // H, the 16-bit forms), literals rounded to the operation's format (binary16 when H), integers the bits at that format;
  // widened to the exact working precision of PTX-FLOAT
  syntax Float ::= fvalK(Map, Map, Int, AOperand, Bool) [function, total]
  rule fvalK(G, U, T, R:AReg, H) => roundFloat(#if regBits(R) ==Int 16 #then decodeF(val(G, U, T, R), 16)
                                    #else decodeF(#if H #then val(G, U, T, R) #else flushB(val(G, U, T, R), 32) #fi, 32) #fi, WP, WEXP)
  rule fvalK(_, _, _, F:Float, H) => roundFloat(decodeF(encodeF(F, #if H #then 16 #else 32 #fi), #if H #then 16 #else 32 #fi), WP, WEXP)
  rule fvalK(_, _, _, I:Int, H) => roundFloat(decodeF(I &Int amask(#if H #then 16 #else 32 #fi), #if H #then 16 #else 32 #fi), WP, WEXP)
  // a result written to D: binary16 for a 16-bit destination (nearest even, subnormals kept, NaN 0x7e00); binary32
  // otherwise, where a non-zero value of magnitude below 2^-126 becomes a zero of its sign before rounding, NaN 0x7fc00000
  syntax Int ::= agxRes(Float, AReg) [function, total]
  rule agxRes(V, D) => #if regBits(D) ==Int 16 #then 32256 #else 2143289344 #fi requires isNaN(V)
  rule agxRes(V, D) => encodeF(roundFloat(V, 11, 5), 16) requires notBool isNaN(V) andBool regBits(D) ==Int 16
  rule agxRes(V, D) => #if signFloat(V) #then 2147483648 #else 0 #fi
    requires notBool isNaN(V) andBool regBits(D) =/=Int 16 andBool V =/=Float fc(0.0, V) andBool absFloat(V) <Float roundFloat(decodeF(8388608, 32), WP, WEXP)
  rule agxRes(V, D) => encodeF(roundFloat(V, 24, 8), 32) [owise]
  syntax Float ::= rcpK(Float) [function, total]
  rule rcpK(X) => X requires isNaN(X)
  rule rcpK(X) => roundFloat(decodeF(fpDiv(.Set, 32, 1065353216, encodeF(roundFloat(X, 24, 8), 32)), 32), WP, WEXP)
    requires notBool isNaN(X) andBool absFloat(X) <=Float roundFloat(decodeF(2122317824, 32), WP, WEXP)   // |X| <= 2^126
  rule rcpK(X) => #if signFloat(X) #then --Float fc(0.0, X) #else fc(0.0, X) #fi [owise]                // 1/x below 2^-126
  // get_sr needs the special-register map: handled with a dedicated rule
  rule <k> #exec(get_sr D, N) => .K ... </k> <gpr> G => srLoop(G, S, EM, 0, D, N) </gpr> <srs> S </srs> <em> EM </em> [priority(40)]
  syntax Map ::= srLoop(Map, Map, Int, Int, AReg, Int) [function, total]
  rule srLoop(G, S, EM, T, D, N) => srLoop(#if ((EM >>Int T) &Int 1) ==Int 1 #then put(G, T, D, {S[N *Int 32 +Int T] orDefault 0}:>Int) #else G #fi, S, EM, T +Int 1, D, N) requires T <Int 32
  rule srLoop(G, _, _, _, _, _) => G [owise]

  // integer helpers
  syntax Int ::= asext(Int, Int) [function, total]
  rule asext(V, W) => #if V &Int (1 <<Int (W -Int 1)) ==Int 0 #then V &Int amask(W) #else (V &Int amask(W)) -Int (1 <<Int W) #fi
  syntax Int ::= bitopv(Int, Int, Int, Int) [function, total]
  // truth tables 0011 and 1100 give A on the hardware (applegpu: UNDEFINED, result = a)
  rule bitopv(TT, A, B, W) => A &Int amask(W) requires TT ==Int 3 orBool TT ==Int 12
  rule bitopv(TT, A, B, W) => (#if (TT &Int 1) =/=Int 0 #then (~Int A) &Int (~Int B) #else 0 #fi
                           |Int #if (TT &Int 2) =/=Int 0 #then A &Int (~Int B) #else 0 #fi
                           |Int #if (TT &Int 4) =/=Int 0 #then (~Int A) &Int B #else 0 #fi
                           |Int #if (TT &Int 8) =/=Int 0 #then A &Int B #else 0 #fi) &Int amask(W) requires TT =/=Int 3 andBool TT =/=Int 12
  syntax Int ::= bfiv(Int, Int, Int, Int) [function, total]
  rule bfiv(M, A, B, C) => ((A &Int ~Int ((#if M ==Int 0 #then 4294967295 #else amask(M) #fi) <<Int (C &Int 127))) |Int ((B &Int (#if M ==Int 0 #then 4294967295 #else amask(M) #fi)) <<Int (C &Int 127))) &Int 4294967295
  syntax Int ::= popc(Int, Int) [function, total] | brev(Int, Int, Int) [function, total] | ffsv(Int, Int) [function, total]
  rule popc(V, I) => (V >>Int I) &Int 1 +Int popc(V, I +Int 1) requires I <Int 32
  rule popc(_, _) => 0 [owise]
  rule brev(V, I, R) => brev(V, I +Int 1, R |Int (((V >>Int I) &Int 1) <<Int (31 -Int I))) requires I <Int 32
  rule brev(_, _, R) => R [owise]
  rule ffsv(V, I) => #if I <Int 0 #then 4294967295 #else #if (V >>Int I) &Int 1 ==Int 1 #then I #else ffsv(V, I -Int 1) #fi #fi

  // comparison conditions of lane T.  Integer conditions compare the sources at their own sizes (unsigned values, or
  // sign-extended at each operand's size for the signed ones), for both instruction kinds; the float conditions of float
  // instructions compare the float values (fp32 subnormals read as zero, as on the hardware); `eq` / `ne` never hold on
  // float instructions; `feq` / `fne` on integer instructions compare the 32-bit views
  syntax Int ::= osz(AOperand) [function, total]
  rule osz(R:AReg) => regBits(R)
  rule osz(_) => 32 [owise]
  syntax Bool ::= isFC(String) [function, total]
  rule isFC(S) => endsWith(S, "feq") orBool endsWith(S, "fne") orBool endsWith(S, "flt") orBool endsWith(S, "fle")
                  orBool endsWith(S, "fgt") orBool endsWith(S, "fge")
  syntax Bool ::= cmpl(String, Map, Map, Int, AOperand, AOperand) [function, total]
  rule cmpl(S, G, U, T, A, B) => fcmpv(S, fvalK(G, U, T, A, false), fvalK(G, U, T, B, false)) requires isFC(S) andBool startsWith(S, "f")
  rule cmpl(S, G, U, T, A, B) => cmpv(S, opv(G, U, T, A, 32), osz(A), opv(G, U, T, B, 32), osz(B)) [owise]
  syntax Bool ::= cmpv(String, Int, Int, Int, Int) [function, total]   // condition, A, size of A, B, size of B
  rule cmpv(S, A, _, B, _) => A ==Int B requires endsWith(S, "eq") andBool notBool startsWith(S, "f")
  rule cmpv(S, A, _, B, _) => A =/=Int B requires endsWith(S, "ne") andBool notBool startsWith(S, "f")
  rule cmpv(S, A, _, B, _) => A <Int B requires endsWith(S, "ult")
  rule cmpv(S, A, _, B, _) => A <=Int B requires endsWith(S, "ule")
  rule cmpv(S, A, _, B, _) => A >Int B requires endsWith(S, "ugt")
  rule cmpv(S, A, _, B, _) => A >=Int B requires endsWith(S, "uge")
  rule cmpv(S, A, AS, B, BS) => asext(A, AS) <Int asext(B, BS) requires endsWith(S, "slt")
  rule cmpv(S, A, AS, B, BS) => asext(A, AS) <=Int asext(B, BS) requires endsWith(S, "sle")
  rule cmpv(S, A, AS, B, BS) => asext(A, AS) >Int asext(B, BS) requires endsWith(S, "sgt")
  rule cmpv(S, A, AS, B, BS) => asext(A, AS) >=Int asext(B, BS) requires endsWith(S, "sge")
  rule cmpv(_, _, _, _, _) => false [owise]
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
  rule stk(G, U, T, K, S, A, B, N) => stk(G[T *Int 256 <- ({G[T *Int 256] orDefault 0}:>Int &Int 4294901760) |Int stkNew(r0l(G, T), K, K ==String "pop" orBool cmpl(S, G, U, T, A, B), N)],
                                          U, T +Int 1, K, S, A, B, N) requires T <Int 32
  rule stk(G, _, _, _, _, _, _, _) => G [owise]
  // the new execution mask is computed from the new r0l values: lane t is active iff r0l == 0
  syntax Int ::= stkEm(Map, Map, Int, String, String, AOperand, AOperand, Int, Int) [function, total]
  rule stkEm(G, U, T, K, S, A, B, N, EM) => stkEm(G, U, T +Int 1, K, S, A, B, N, EM |Int (#if stkNew(r0l(G, T), K, K ==String "pop" orBool cmpl(S, G, U, T, A, B), N) ==Int 0 #then 1 <<Int T #else 0 #fi)) requires T <Int 32
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
        R |Int #if ((EM >>Int T) &Int 1) ==Int 1 andBool cmpl(S, G, U, T, A, B) #then 1 <<Int T #else 0 #fi) requires T <Int 32
  rule ballot(_, _, _, _, _, _, _, R) => R [owise]
  syntax Map ::= bcast(Map, Int, Int, AReg, Int) [function, total]
  rule bcast(G, EM, T, D, V) => bcast(#if ((EM >>Int T) &Int 1) ==Int 1 #then put(G, T, D, V) #else G #fi, EM, T +Int 1, D, V) requires T <Int 32
  rule bcast(G, _, _, _, _) => G [owise]

  // ---------------------------------------------------------------- memory
  // device_load: up to 4 consecutive elements per lane from base (64-bit uniform pair) + (index << (log2 size + shift)); results become
  // visible at the next `wait` (reading the destination earlier returns the old value).  The 32-bit index is sign-extended
  // (`signed`) or zero-extended (`unsigned`) to 64 bits before the shift; the sum wraps modulo 2^64.  Threadgroup accesses have no
  // signedness field in the encoding and are modelled with a zero-extended index (as before).
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
  rule <k> #exec(device_load F N D, P, I, X, S) => #ld(F, N, D, ubase(U, P), I, X, S, "dev") ... </k> <upr> U </upr>
  rule <k> #exec(threadgroup_load F N D, I, _) => #ld(F, N, D, 0, I, unsigned, 0, "tg") ... </k>
  rule <k> #exec(device_store F N D, P, I, X, S) => #st(F, N, D, ubase(U, P), I, X, S, "dev") ... </k> <upr> U </upr>
  rule <k> #exec(threadgroup_store F N D, I, _) => #st(F, N, D, 0, I, unsigned, 0, "tg") ... </k>
  syntax KItem ::= #ld(AFmt, Int, AReg, Int, AOperand, ASign, Int, String) | #st(AFmt, Int, AReg, Int, AOperand, ASign, Int, String)
  rule <k> #ld(F, N, D, B, I, X, S, W) => .K ... </k> <gpr> G </gpr> <upr> U </upr> <em> EM </em> <dmem> DM </dmem> <tmem> TM </tmem> <pend> P => P ldAll(G, U, EM, DM, TM, 0, F, N, D, B, I, X, S, W) </pend>
  rule <k> #st(F, N, D, B, I, X, S, W) => .K ... </k> <gpr> G </gpr> <upr> U </upr> <em> EM </em> <dmem> DM => #if W ==String "dev" #then stEl(G, U, EM, DM, 0, F, N, D, B, I, X, S, W) #else DM #fi </dmem>
       <tmem> TM => #if W ==String "tg" #then stEl(G, U, EM, TM, 0, F, N, D, B, I, X, S, W) #else TM #fi </tmem>
  // index of lane t extended to 64 bits according to the signedness field
  syntax Int ::= idxv(Map, Map, Int, AOperand, ASign) [function, total]
  rule idxv(G, U, T, I, signed) => asext(opv(G, U, T, I, 32), 32)
  rule idxv(G, U, T, I, unsigned) => opv(G, U, T, I, 32)
  // effective address of lane t
  syntax Int ::= ea(Map, Map, Int, AFmt, AOperand, ASign, Int, Int) [function, total]
  // threadgroup accesses: byte 4 * size * index (the index counts four-element vectors; 16-bit index) + size * element
  // (hardware test suite of AGX-Performance-Simulator); the shift operand has no threadgroup encoding and is ignored
  syntax Int ::= eaw(Map, Map, Int, AFmt, AOperand, ASign, Int, Int, String) [function, total]
  rule eaw(G, U, T, F, I, _, _, B, "tg") => B +Int 4 *Int fsize(F) *Int opv(G, U, T, I, 16)
  rule eaw(G, U, T, F, I, X, S, B, _) => ea(G, U, T, F, I, X, S, B) [owise]
  rule ea(G, U, T, F, I, X, S, B) => (((B +Int (idxv(G, U, T, I, X) <<Int (flog(F) +Int S))) &Int 18446744073709551615) >>Int flog(F)) <<Int flog(F)      // rounded down to the element size
  syntax List ::= ldAll(Map, Map, Int, Map, Map, Int, AFmt, Int, AReg, Int, AOperand, ASign, Int, String) [function, total]
  rule ldAll(G, U, EM, DM, TM, T, F, N, D, B, I, X, S, W) => (#if ((EM >>Int T) &Int 1) ==Int 1 #then ldLane(DM, TM, T, F, N, D, eaw(G, U, T, F, I, X, S, B, W), 0, W) #else .List #fi)
        ldAll(G, U, EM, DM, TM, T +Int 1, F, N, D, B, I, X, S, W) requires T <Int 32
  rule ldAll(_, _, _, _, _, _, _, _, _, _, _, _, _, _) => .List [owise]
  syntax List ::= ldLane(Map, Map, Int, AFmt, Int, AReg, Int, Int, String) [function, total]
  rule ldLane(DM, TM, T, F, N, D, A, K, W) => ListItem(ld(T, mcode(D, K), rdBytes(#if W ==String "dev" #then DM #else TM #fi, A +Int K *Int fsize(F), fsize(F)))) ldLane(DM, TM, T, F, N, D, A, K +Int 1, W) requires K <Int N
  rule ldLane(_, _, _, _, _, _, _, _, _) => .List [owise]
  // the j-th element uses register j of the tuple at D: the 32-bit registers D + j, or for a 16-bit D the 16-bit registers
  // that follow it (r5h, r6l, ...); mcode numbers them (32-bit: the register number, 16-bit: 1024 + the half index).  Loads
  // write the element zero-extended to a 32-bit register or truncated to a 16-bit one; stores write the low bytes.
  syntax Int ::= mcode(AReg, Int) [function, total]
  rule mcode(D, K) => #if regHalf(D) ==Int 0 #then regNum(D) +Int K #else 1024 +Int 2 *Int regNum(D) +Int regHalf(D) -Int 1 +Int K #fi
  syntax Int ::= mget(Map, Int, Int) [function, total]
  rule mget(G, T, C) => {G[T *Int 256 +Int C] orDefault 0}:>Int requires C <Int 1024
  rule mget(G, T, C) => ({G[T *Int 256 +Int ((C -Int 1024) >>Int 1)] orDefault 0}:>Int >>Int (16 *Int ((C -Int 1024) &Int 1))) &Int m16() [owise]
  // stores: element by element, each over all active lanes; device stores go from the highest lane down (the lowest
  // lane's value stays when lanes store to the same bytes), threadgroup stores from the lowest lane up
  syntax Map ::= stEl(Map, Map, Int, Map, Int, AFmt, Int, AReg, Int, AOperand, ASign, Int, String) [function, total]
  rule stEl(G, U, EM, M, K, F, N, D, B, I, X, S, W) => stEl(G, U, EM, stLanes(G, U, EM, M, #if W ==String "dev" #then 31 #else 0 #fi, K, F, D, B, I, X, S, W),
                                                          K +Int 1, F, N, D, B, I, X, S, W) requires K <Int N
  rule stEl(_, _, _, M, _, _, _, _, _, _, _, _, _) => M [owise]
  syntax Map ::= stLanes(Map, Map, Int, Map, Int, Int, AFmt, AReg, Int, AOperand, ASign, Int, String) [function, total]
  rule stLanes(G, U, EM, M, T, K, F, D, B, I, X, S, W) =>
       stLanes(G, U, EM, #if ((EM >>Int T) &Int 1) ==Int 1 #then wrBytes(M, eaw(G, U, T, F, I, X, S, B, W) +Int K *Int fsize(F), mget(G, T, mcode(D, K)), fsize(F)) #else M #fi,
               #if W ==String "dev" #then T -Int 1 #else T +Int 1 #fi, K, F, D, B, I, X, S, W)
    requires T >=Int 0 andBool T <Int 32
  rule stLanes(_, _, _, M, _, _, _, _, _, _, _, _, _) => M [owise]
  // atomics (abstract: the reverse-engineered documents do not describe the atomic instructions; Metal exposes relaxed 32-bit
  // atomics).  Active lanes perform their read-modify-write one after the other in lane order (one valid serialisation); the old
  // value is returned like a load, visible at the next `wait`.  The index is zero-extended (the encoding's signedness field is not modelled).
  syntax AtomRes ::= atomRes(Map, List)
  syntax Int ::= aop(AAop, Int, Int) [function, total]
  rule aop(add, O, V) => (O +Int V) &Int m32()
  rule aop(min, O, V) => #if O <Int V #then O #else V #fi
  rule aop(max, O, V) => #if O >Int V #then O #else V #fi
  rule aop(and, O, V) => O &Int V
  rule aop(or, O, V) => O |Int V
  rule aop(xor, O, V) => O xorInt V
  rule aop(xchg, _, V) => V
  syntax AtomRes ::= atomLoop(Map, Map, Int, Int, Map, AAop, AReg, Int, AOperand, AOperand, List) [function, total]
  rule atomLoop(G, U, EM, T, M, OP, D, B, I, V, L) =>
       #if ((EM >>Int T) &Int 1) ==Int 1
       #then atomLoop(G, U, EM, T +Int 1, wrBytes(M, ea(G, U, T, u32, I, unsigned, 0, B), aop(OP, rdBytes(M, ea(G, U, T, u32, I, unsigned, 0, B), 4), opv(G, U, T, V, 32)), 4), OP, D, B, I, V,
                      L ListItem(ld(T, regNum(D), rdBytes(M, ea(G, U, T, u32, I, unsigned, 0, B), 4))))
       #else atomLoop(G, U, EM, T +Int 1, M, OP, D, B, I, V, L) #fi
    requires T <Int 32
  rule atomLoop(_, _, _, _, M, _, _, _, _, _, L) => atomRes(M, L) [owise]
  rule <k> #exec(device_atomic OP D, P, I, V) => .K ... </k> <gpr> G </gpr> <upr> U </upr> <em> EM </em>
       <dmem> M => memOf(atomLoop(G, U, EM, 0, M, OP, D, ubase(U, P), I, V, .List)) </dmem>
       <pend> L => L pendOf(atomLoop(G, U, EM, 0, M, OP, D, ubase(U, P), I, V, .List)) </pend>
  // threadgroup atomics: the word at byte 16 * index (a threadgroup access with 32-bit elements); the old values are
  // written when the instruction completes (applegpu: threadgroup atomics have no wait group)
  rule <k> #exec(threadgroup_atomic OP D, I, V) => .K ... </k> <gpr> G => applyLd(G, pendOf(tgAtomLoop(G, U, EM, 0, M, OP, D, I, V, .List))) </gpr>
       <upr> U </upr> <em> EM </em>
       <tmem> M => memOf(tgAtomLoop(G, U, EM, 0, M, OP, D, I, V, .List)) </tmem>
  syntax AtomRes ::= tgAtomLoop(Map, Map, Int, Int, Map, AAop, AReg, AOperand, AOperand, List) [function, total]
  rule tgAtomLoop(G, U, EM, T, M, OP, D, I, V, L) =>
       #if ((EM >>Int T) &Int 1) ==Int 1
       #then tgAtomLoop(G, U, EM, T +Int 1, wrBytes(M, 16 *Int opv(G, U, T, I, 16), aop(OP, rdBytes(M, 16 *Int opv(G, U, T, I, 16), 4), opv(G, U, T, V, 32)), 4),
                        OP, D, I, V, L ListItem(ld(T, regNum(D), rdBytes(M, 16 *Int opv(G, U, T, I, 16), 4))))
       #else tgAtomLoop(G, U, EM, T +Int 1, M, OP, D, I, V, L) #fi
    requires T <Int 32
  rule tgAtomLoop(_, _, _, _, M, _, _, _, _, L) => atomRes(M, L) [owise]
  syntax Map ::= memOf(AtomRes) [function, total]
  rule memOf(atomRes(M, _)) => M
  syntax List ::= pendOf(AtomRes) [function, total]
  rule pendOf(atomRes(_, L)) => L

  // wait: apply the pending loads
  rule <k> #exec(wait _) => .K ... </k> <gpr> G => applyLd(G, P) </gpr> <pend> P => .List </pend>
  syntax APendLd ::= ld(Int, Int, Int)
  syntax Map ::= applyLd(Map, List) [function, total]
  rule applyLd(G, ListItem(ld(T, R, V)) Rest) => applyLd(G[T *Int 256 +Int R <- V &Int m32()], Rest) requires R <Int 1024
  rule applyLd(G, ListItem(ld(T, C, V)) Rest) => applyLd(G[T *Int 256 +Int ((C -Int 1024) >>Int 1) <-
       #if (C -Int 1024) &Int 1 ==Int 0 #then ({G[T *Int 256 +Int ((C -Int 1024) >>Int 1)] orDefault 0}:>Int &Int 4294901760) |Int (V &Int m16())
       #else ({G[T *Int 256 +Int ((C -Int 1024) >>Int 1)] orDefault 0}:>Int &Int m16()) |Int ((V &Int m16()) <<Int 16) #fi], Rest) requires C >=Int 1024
  rule applyLd(G, _) => G [owise]
endmodule

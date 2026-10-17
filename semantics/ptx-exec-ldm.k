// =============================================================================
// PTX-EXEC-LDM: warp-level 8x8 b16 matrix moves.
//
//   ldmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans][.shared[::cta]].b16  {r...}, [p]
//   stmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans][.shared[::cta]].b16  [p], {r...}
//   movmatrix.sync.aligned.m8n8.trans.b16  d, a
//
// Matrix i (0..num-1) of ldmatrix/stmatrix has its row r addressed (16 bytes, 8 b16
// elements) by lane 8*i + r.  Without .trans, lane (g = lane/4, t = lane%4) holds
// elements (g, 2t) [low half] and (g, 2t+1) [high half] of register i; with .trans it
// holds (2t, g) and (2t+1, g).  The sm_100 m16n16 / m8n16 / m16n8 forms (b8, b6, b4) follow below.
//
// ldmatrix: each lane reads its row, then a warp rendezvous redistributes the rows.
// stmatrix: a rendezvous builds the rows, each addressing lane then writes its row.
// =============================================================================
requires "ptx-exec-mma.k"

module PTX-EXEC-LDM
  imports PTX-EXEC-MMA

  rule isCollective("ldmatrix") => true
  rule isCollective("stmatrix") => true
  rule isCollective("movmatrix") => true

  syntax Int ::= numOf(Set) [function, total]
  rule numOf(M) => #if "x4" in M #then 4 #else #if "x2" in M #then 2 #else 1 #fi #fi
  syntax List ::= dlist(Operand) [function, total]
  rule dlist({ Os }) => opsList(Os)
  rule dlist(D) => ListItem(D) [owise]

  syntax Int ::= half16(Int, Int) [function, total]            // 16-bit half h of v
  rule half16(V, H) => (V >>Int (16 *Int H)) &Int 65535
  syntax Int ::= row16(Int, Int) [function, total]             // element c of a 128-bit row
  rule row16(V, C) => (V >>Int (16 *Int C)) &Int 65535

  // ---------------------------------------------------------------- ldmatrix
  // Phase 1 (rendezvous): lanes exchange the row addresses; each lane learns the 2*num
  // addresses of the 16-bit elements it needs.  Phase 2: every lane then reads them, so
  // that stores performed by the other lanes before the instruction are visible.
  // The rows' addresses are kept as raw addresses of the instruction's state space.
  rule <tk> exec(opc("ldmatrix", M, ListItem(_)), ops(Dv, [ A ], .Operands), C) =>
            #coll("ldmatrix", 4294967295, ListItem(numOf(M)) ListItem("trans" in M),
                  ListItem(locOff(ea(A, spaceOf(M), C))), ldmScr(0, 2 *Int numOf(M)))
            ~> #ldmFetch(locSp(ea(A, spaceOf(M), C), spaceOf(M)), 2 *Int numOf(M), dlist(Dv)) ... </tk>
    requires notBool ("m16n16" in M orBool "m8n16" in M)

  // addresses of T[a][b], b being the b16 element index in the row
  syntax Int ::= ldmAddr(Map, Int, Int, Int) [function, total]
  rule ldmAddr(Arr, I, A, B) => at(listAt(Arr, 8 *Int I +Int A), 0) +Int 2 *Int B
  // address of the half h (0 low, 1 high) of register i for lane L
  syntax Int ::= ldmHalf(Map, Int, Int, Int, Bool) [function, total]
  rule ldmHalf(Arr, L, I, H, false) => ldmAddr(Arr, I, L /Int 4, 2 *Int (L %Int 4) +Int H)
  rule ldmHalf(Arr, L, I, H, true)  => ldmAddr(Arr, I, 2 *Int (L %Int 4) +Int H, L /Int 4)
  syntax List ::= ldmRegs(Map, Int, Int, Int, Bool) [function, total]
  rule ldmRegs(Arr, L, I, N, T) => ListItem(ldmHalf(Arr, L, I, 0, T)) ListItem(ldmHalf(Arr, L, I, 1, T)) ldmRegs(Arr, L, I +Int 1, N, T) requires I <Int N
  rule ldmRegs(_, _, _, _, _) => .List [owise]
  syntax Map ::= ldmAll(Map, List, Int, Bool) [function, total]
  rule ldmAll(Arr, ListItem(L:Int) Ls, N, T) => (L |-> ldmRegs(Arr, L, 0, N, T)) ldmAll(Arr, Ls, N, T)
  rule ldmAll(_, _, _, _) => .Map [owise]
  rule collCompute("ldmatrix", ListItem(N:Int) ListItem(T:Bool), Arr) => ldmAll(Arr, keys_list(Arr), N, T)

  // scratch registers carrying the element addresses between the two phases
  syntax List ::= ldmScr(Int, Int) [function, total]
  syntax Reg ::= ldmR(Int) [function, total]
  rule ldmR(0) => #token("%__ldm0", "Reg")
  rule ldmR(1) => #token("%__ldm1", "Reg")
  rule ldmR(2) => #token("%__ldm2", "Reg")
  rule ldmR(3) => #token("%__ldm3", "Reg")
  rule ldmR(4) => #token("%__ldm4", "Reg")
  rule ldmR(5) => #token("%__ldm5", "Reg")
  rule ldmR(6) => #token("%__ldm6", "Reg")
  rule ldmR(7) => #token("%__ldm7", "Reg")
  rule ldmR(_) => #token("%__ldm0", "Reg") [owise]
  syntax String ::= locSp(Loc, String) [function, total]
  rule locSp(loc(S, _), _) => S
  rule locSp(_, "generic") => "shared"
  rule locSp(_, S) => S [owise]
  syntax Int ::= locOff(Loc) [function, total]
  rule locOff(loc(_, A)) => A
  rule locOff(_) => 0 [owise]
  rule ldmScr(I, N) => ListItem(ldmR(I)) ldmScr(I +Int 1, N) requires I <Int N
  rule ldmScr(_, _) => .List [owise]
  syntax List ::= ldmVals(Map, Int, Int) [function, total]
  rule ldmVals(R, I, N) => ListItem({R["%__ldm" +String Int2String(I)] orDefault 0}:>Int) ldmVals(R, I +Int 1, N) requires I <Int N
  rule ldmVals(_, _, _) => .List [owise]

  syntax KItem ::= #ldmFetch(String, Int, List) | #ldmRd(String, List, List)
                 | #ldmH(String, Int, List, Operand, List) | #ldmH2(Int, String, List, Operand, List)
  rule <tk> #ldmFetch(Sp, N, Ds) => #ldmRd(Sp, ldmVals(R, 0, N), Ds) ... </tk>
       <regs> R </regs>
  rule <tk> #ldmRd(Sp, ListItem(A1:Int) ListItem(A2:Int) As, ListItem(D:Operand) Ds) =>
            #mem(mkLoc(Sp, A1), rdOp(2)) ~> #ldmH(Sp, A2, As, D, Ds) ... </tk>
  rule <tk> #ldmRd(_, .List, _) => .K ... </tk>
  rule <tk> #res(V) ~> #ldmH(Sp, A2, As, D, Ds) => #mem(mkLoc(Sp, A2), rdOp(2)) ~> #ldmH2(V, Sp, As, D, Ds) ... </tk>
  rule <tk> #res(W) ~> #ldmH2(V, Sp, As, D, Ds) => #wmWr(ListItem(D), ListItem(V |Int (W <<Int 16))) ~> #ldmRd(Sp, As, Ds) ... </tk>


  // ---------------------------------------------------------------- ldmatrix.m8n16 / m16n16 (sm_100a): 8-bit data, optionally expanded from 6 / 4-bit
  //   ldmatrix.sync.aligned.m16n16.num.trans{.ss}.b8 | .b8x16.b6x16_p32 | .b8x16.b4x16_p64   r, [p]   (16 row addresses per matrix, 16 bytes each)
  //   ldmatrix.sync.aligned.m8n16.num{.ss}.b8x16.b6x16_p32 | .b8x16.b4x16_p64 | .s8.s4         r, [p]   (8 row addresses per matrix)
  // m8n16: thread (g, t) receives the elements 4t..4t+3 of row g; m16n16 (transposed): register 2i holds M[4t+b][g] and register 2i+1
  // M[4t+b][g+8] (b = 0..3) -- byte b of the register.  Expanded elements sit in 8-bit containers (6-bit: low bits, 4-bit: central bits, as for
  // mma.sync; s4 is sign-extended).  The per-thread layouts are given by figures in the ISA; this is an assumption.
  syntax Int ::= ldmFmt(Set, List) [function, total]       // 0 plain 8-bit, 6 / 4: unsigned 6 / 4-bit elements, 5: signed 4-bit
  rule ldmFmt(M, _) => 6 requires "b6x16_p32" in M
  rule ldmFmt(M, _) => 4 requires "b4x16_p64" in M
  rule ldmFmt(_, Ts) => 5 requires ty("s", 4) in Ts
  rule ldmFmt(_, _) => 0 [owise]
  syntax Int ::= ldmEl(Int, Int, Int) [function, total]    // format, 128-bit row, column -> 8-bit element
  rule ldmEl(0, R, C) => (R >>Int (8 *Int C)) &Int 255
  rule ldmEl(6, R, C) => (R >>Int (6 *Int C)) &Int 63
  rule ldmEl(4, R, C) => ((R >>Int (4 *Int C)) &Int 15) <<Int 2
  rule ldmEl(5, R, C) => (#if ((R >>Int (4 *Int C)) &Int 8) ==Int 8 #then ((R >>Int (4 *Int C)) &Int 15) |Int 240 #else (R >>Int (4 *Int C)) &Int 15 #fi) &Int 255
  rule ldmEl(_, _, _) => 0 [owise]
  // addresses lane L needs
  syntax List ::= ldm2Addrs(Map, Int, Int, Int, Int) [function, total]     // arrivals, kind (0 m8n16, 1 m16n16), lane, matrix, count
  rule ldm2Addrs(Arr, K, L, I, N) => ldm2A1(Arr, K, L, I) ldm2Addrs(Arr, K, L, I +Int 1, N) requires I <Int N
  rule ldm2Addrs(_, _, _, _, _) => .List [owise]
  syntax List ::= ldm2A1(Map, Int, Int, Int) [function, total]
  rule ldm2A1(Arr, 0, L, I) => ListItem(at(listAt(Arr, 8 *Int I +Int L /Int 4), 0))
  rule ldm2A1(Arr, 1, L, I) => ListItem(at(listAt(Arr, 16 *Int I +Int 4 *Int (L %Int 4)), 0)) ListItem(at(listAt(Arr, 16 *Int I +Int 4 *Int (L %Int 4) +Int 1), 0))
                                ListItem(at(listAt(Arr, 16 *Int I +Int 4 *Int (L %Int 4) +Int 2), 0)) ListItem(at(listAt(Arr, 16 *Int I +Int 4 *Int (L %Int 4) +Int 3), 0))
  rule ldm2A1(_, _, _, _) => .List [owise]
  syntax Map ::= ldm2All(Map, List, Int, Int) [function, total]
  rule ldm2All(Arr, ListItem(L:Int) Ls, K, N) => (L |-> ldm2Addrs(Arr, K, L, 0, N)) ldm2All(Arr, Ls, K, N)
  rule ldm2All(_, _, _, _) => .Map [owise]
  rule collCompute("ldmatrix2", ListItem(K:Int) ListItem(N:Int), Arr) => ldm2All(Arr, keys_list(Arr), K, N)
  // register values from the rows read (ListItem per address, in the order of ldm2Addrs)
  syntax List ::= ldm2Out(Int, Int, Int, List) [function, total]           // kind, format, thread-in-warp, rows
  rule ldm2Out(0, F, T, ListItem(R:Int) Rs) =>
       ListItem(ldmEl(F, R, 4 *Int (T %Int 4)) |Int (ldmEl(F, R, 4 *Int (T %Int 4) +Int 1) <<Int 8) |Int (ldmEl(F, R, 4 *Int (T %Int 4) +Int 2) <<Int 16) |Int (ldmEl(F, R, 4 *Int (T %Int 4) +Int 3) <<Int 24))
       ldm2Out(0, F, T, Rs)
  rule ldm2Out(1, F, T, ListItem(R0:Int) ListItem(R1:Int) ListItem(R2:Int) ListItem(R3:Int) Rs) =>
       ListItem(ldmEl(F, R0, T /Int 4) |Int (ldmEl(F, R1, T /Int 4) <<Int 8) |Int (ldmEl(F, R2, T /Int 4) <<Int 16) |Int (ldmEl(F, R3, T /Int 4) <<Int 24))
       ListItem(ldmEl(F, R0, T /Int 4 +Int 8) |Int (ldmEl(F, R1, T /Int 4 +Int 8) <<Int 8) |Int (ldmEl(F, R2, T /Int 4 +Int 8) <<Int 16) |Int (ldmEl(F, R3, T /Int 4 +Int 8) <<Int 24))
       ldm2Out(1, F, T, Rs)
  rule ldm2Out(_, _, _, _) => .List [owise]
  syntax KItem ::= #ldm2Rd(String, List, Int, List, List) | #ldm2Cont(String, List, Int, List, List) | #ldm2Fin(List, List)
  rule <tk> exec(opc("ldmatrix", M, Ts), ops(Dv, [ A ], .Operands), C) =>
            #coll("ldmatrix2", 4294967295, ListItem(#if "m16n16" in M #then 1 #else 0 #fi) ListItem(numOf(M)),
                  ListItem(locOff(ea(A, spaceOf(M), C))), ldmScr(0, #if "m16n16" in M #then 4 *Int numOf(M) #else numOf(M) #fi))
            ~> #ldm2Go(locSp(ea(A, spaceOf(M), C), spaceOf(M)), #if ldmFmt(M, Ts) ==Int 5 #then 8 #else 16 #fi, ListItem(#if "m16n16" in M #then 1 #else 0 #fi) ListItem(ldmFmt(M, Ts)) ListItem(numOf(M)) ListItem(Dv),
                       #if "m16n16" in M #then 4 *Int numOf(M) #else numOf(M) #fi) ... </tk>
    requires "m16n16" in M orBool "m8n16" in M
  syntax KItem ::= #ldm2Go(String, Int, List, Int)
  rule <tk> #ldm2Go(Sp, Sz, Info, N) => #ldm2Rd(Sp, ldmVals(R, 0, N), Sz, .List, Info) ... </tk>
       <regs> R </regs>
  rule <tk> #ldm2Rd(Sp, ListItem(A:Int) As, Sz, Acc, Info) => #mem(mkLoc(Sp, A), rdOp(Sz)) ~> #ldm2Cont(Sp, As, Sz, Acc, Info) ... </tk>
  rule <tk> #res(V) ~> #ldm2Cont(Sp, As, Sz, Acc, Info) => #ldm2Rd(Sp, As, Sz, Acc ListItem(V), Info) ... </tk>
  rule <tk> #ldm2Rd(_, .List, _, Acc, Info) => #ldm2Fin(Info, Acc) ... </tk>
  rule <tk> #ldm2Fin(ListItem(K:Int) ListItem(F:Int) ListItem(N:Int) ListItem(Dv:Operand), Acc) => #wmWr(dlist(Dv), ldm2Out(K, F, TID %Int 32, Acc)) ... </tk>
       <tid> TID </tid>

  // ---------------------------------------------------------------- stmatrix
  // contribution: the lane's N registers; result: the 128-bit row it must store
  rule <tk> exec(opc("stmatrix", M, ListItem(_)), ops([ A ], Vs, .Operands), C) =>
            #coll("stmatrix", 4294967295, ListItem(numOf(M)) ListItem("trans" in M), vecRegs(Vs, ty("b", 32), C), ListItem(stmTmp()))
            ~> #stmGo(ea(A, spaceOf(M), C), numOf(M)) ... </tk>
    requires notBool "m16n8" in M
  // stmatrix.m16n8 (8-bit): a 16 x 8 matrix X per register, stored transposed: memory row n (16 bytes, addressed by thread 8i + n) holds
  // X[0..15][n].  Thread (g, t) holds X[g][2t], X[g][2t+1] in bytes 0, 1 and X[g+8][2t], X[g+8][2t+1] in bytes 2, 3 (accumulator-like layout; assumed,
  // the ISA gives it as a figure).
  rule <tk> exec(opc("stmatrix", M, ListItem(_)), ops([ A ], Vs, .Operands), C) =>
            #coll("stmatrix2", 4294967295, ListItem(numOf(M)), vecRegs(Vs, ty("b", 32), C), ListItem(stmTmp()))
            ~> #stmGo(ea(A, spaceOf(M), C), numOf(M)) ... </tk>
    requires "m16n8" in M
  syntax Int ::= stmX(Map, Int, Int, Int) [function, total]          // matrix i, row m, column n of X
  rule stmX(Arr, I, Rm, Cn) => (at(listAt(Arr, 4 *Int (Rm %Int 8) +Int Cn /Int 2), I) >>Int (8 *Int (2 *Int (Rm /Int 8) +Int Cn %Int 2))) &Int 255
  syntax Int ::= stmRow2(Map, Int, Int, Int) [function, total]       // matrix i, memory row n, byte m
  rule stmRow2(Arr, I, Cn, Rm) => (stmX(Arr, I, Rm, Cn) <<Int (8 *Int Rm)) |Int stmRow2(Arr, I, Cn, Rm +Int 1) requires Rm <Int 16
  rule stmRow2(_, _, _, _) => 0 [owise]
  syntax Map ::= stmAll2(Map, List, Int) [function, total]
  rule stmAll2(Arr, ListItem(L:Int) Ls, N) => (L |-> #if L <Int 8 *Int N #then ListItem(stmRow2(Arr, L /Int 8, L %Int 8, 0)) #else ListItem(0) #fi) stmAll2(Arr, Ls, N)
  rule stmAll2(_, _, _) => .Map [owise]
  rule collCompute("stmatrix2", ListItem(N:Int), Arr) => stmAll2(Arr, keys_list(Arr), N)
  syntax Reg ::= stmTmp() [function, total]
  rule stmTmp() => #token("%__stm", "Reg")
  rule <tk> #stmGo(L, N) => #mem(L, wrOp(16, V)) ~> #drop ... </tk>
       <tid> TID </tid>
       <regs> ... "%__stm" |-> V ... </regs>
    requires (TID %Int 32) <Int 8 *Int N
  rule <tk> #stmGo(_, _) => .K ... </tk> [owise]
  syntax KItem ::= #stmGo(Loc, Int)

  // element (a, b) of stored matrix i, found in the registers of the owning lane
  syntax Int ::= sel(Map, Int, Int, Int, Bool) [function, total]
  rule sel(Arr, I, A, B, false) => half16(at(listAt(Arr, A *Int 4 +Int B /Int 2), I), B %Int 2)
  rule sel(Arr, I, A, B, true)  => half16(at(listAt(Arr, B *Int 4 +Int A /Int 2), I), A %Int 2)
  syntax Int ::= stmRow(Map, Int, Int, Bool, Int) [function, total]   // matrix, row, trans, element index
  rule stmRow(Arr, I, A, T, C) => (sel(Arr, I, A, C, T) <<Int (16 *Int C)) |Int stmRow(Arr, I, A, T, C +Int 1) requires C <Int 8
  rule stmRow(_, _, _, _, _) => 0 [owise]
  syntax Map ::= stmAll(Map, List, Int, Bool) [function, total]
  rule stmAll(Arr, ListItem(L:Int) Ls, N, T) =>
       (L |-> #if L <Int 8 *Int N #then ListItem(stmRow(Arr, L /Int 8, L %Int 8, T, 0)) #else ListItem(0) #fi) stmAll(Arr, Ls, N, T)
  rule stmAll(_, _, _, _) => .Map [owise]
  rule collCompute("stmatrix", ListItem(N:Int) ListItem(T:Bool), Arr) => stmAll(Arr, keys_list(Arr), N, T)

  // ---------------------------------------------------------------- movmatrix
  rule <tk> exec(opc("movmatrix", M, ListItem(_)), ops(D, A, .Operands), C) =>
            #coll("movmatrix", 4294967295, .List, ListItem(rd(A, ty("b", 32), C)), ListItem(D)) ... </tk>
  rule collCompute("movmatrix", _, Arr) => mvAll(Arr, keys_list(Arr))
  syntax Map ::= mvAll(Map, List) [function, total]
  rule mvAll(Arr, ListItem(L:Int) Ls) => (L |-> ListItem(mvElem(Arr, 2 *Int (L %Int 4), L /Int 4) |Int (mvElem(Arr, 2 *Int (L %Int 4) +Int 1, L /Int 4) <<Int 16))) mvAll(Arr, Ls)
  rule mvAll(_, _) => .Map [owise]
  // element (r, c) of the source matrix
  syntax Int ::= mvElem(Map, Int, Int) [function, total]
  rule mvElem(Arr, R, C) => half16(at(listAt(Arr, R *Int 4 +Int C /Int 2), 0), C %Int 2)
endmodule

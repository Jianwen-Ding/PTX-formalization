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
// holds (2t, g) and (2t+1, g).  Not modelled: the sm_100 m16n8/m8n16 b8/b6/b4 forms.
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

  // ---------------------------------------------------------------- stmatrix
  // contribution: the lane's N registers; result: the 128-bit row it must store
  rule <tk> exec(opc("stmatrix", M, ListItem(_)), ops([ A ], Vs, .Operands), C) =>
            #coll("stmatrix", 4294967295, ListItem(numOf(M)) ListItem("trans" in M), vecRegs(Vs, ty("b", 32), C), ListItem(stmTmp()))
            ~> #stmGo(ea(A, spaceOf(M), C), numOf(M)) ... </tk>
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

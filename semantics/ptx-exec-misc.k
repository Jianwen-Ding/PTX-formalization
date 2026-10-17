// =============================================================================
// PTX-EXEC-MISC: remaining asynchronous and miscellaneous instructions (sm_90 / sm_100)
//
//   st.async / red.async            remote shared::cluster store / reduction signalling an mbarrier
//   st.bulk                         zero-fill of shared memory
//   griddepcontrol, setmaxnreg      no observable effect in this machine
//   clusterlaunchcontrol            try_cancel never cancels a CTA (every CTA is already launched)
//
// Not modelled: multimem.* (needs a multicast object spanning devices).
// =============================================================================
requires "ptx-exec-bulk.k"

module PTX-EXEC-MISC
  imports PTX-EXEC-BULK

  rule <tk> exec(opc(OP, _, _), ops(_), _) => .K ... </tk>
    requires OP in (SetItem("griddepcontrol") SetItem("setmaxnreg"))

  // st.async{.weak}{.shared::cluster}.mbarrier::complete_tx::bytes{.vN}.type [addr], b, [remote_bar]
  syntax KItem ::= #asyncWr(Loc, List, Ty, Loc, Int) | #asyncRed(Loc, String, Ty, Int, Loc)
  rule <tk> exec(opc("st", M, ListItem(T)), ops([ A ], V, [ Mb ], .Operands), C) =>
            #asyncWr(ea(A, "shared_cluster", C), #if isVec(V) #then rdList(opsList(vecItems(V)), T, C) #else ListItem(rd(V, T, C)) #fi,
                     T, ea(Mb, "shared_cluster", C), 0) ... </tk>
    requires "async" in M
  syntax Operands ::= vecItems(Operand) [function, total]
  rule vecItems({ Os }) => Os
  rule vecItems(_) => .Operands [owise]
  rule <tk> #asyncWr(loc(S, A), ListItem(V:Int) Vs, T, Mb, N) =>
            #mem(loc(S, A), wrOp(bytes(T), V)) ~> #drop ~> #asyncWr(loc(S, A +Int bytes(T)), Vs, T, Mb, N +Int bytes(T)) ... </tk>
  rule <tk> #asyncWr(_, .List, _, Mb, N) => #bulkTx(Mb, N) ... </tk>

  // red.async{.sem}{.scope}{.shared::cluster}.mbarrier::complete_tx::bytes.op.type [addr], b, [remote_bar]
  rule <tk> exec(opc("red", M, ListItem(T)), ops([ A ], V, [ Mb ], .Operands), C) =>
            #asyncRed(ea(A, "shared_cluster", C), atomOp(M), T, rd(V, T, C), ea(Mb, "shared_cluster", C)) ... </tk>
    requires "async" in M
  rule <tk> #asyncRed(L, OP, T, V, Mb) => #mem(L, atOp(OP, T, V, 0)) ~> #drop ~> #bulkTx(Mb, bytes(T)) ... </tk>

  // st.bulk{.weak}{.shared::cta} [addr], size, initval   (initval is 0)
  rule <tk> exec(opc("st", M, _), ops([ A ], Sz, _, .Operands), C) => #mem(ea(A, "shared", C), wrOp(rd(Sz, ty("u", 32), C), 0)) ~> #drop ... </tk>
    requires "bulk" in M

  // clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128 [addr], [mbar]
  rule <tk> exec(opc("clusterlaunchcontrol", M, _), ops([ A ], [ Mb ], .Operands), C) =>
            #mem(ea(A, "shared", C), wrOp(16, 0)) ~> #drop ~> #bulkTx(ea(Mb, "shared", C), 16) ... </tk>
    requires "try_cancel" in M
  // clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response  /  .get_first_ctaid{::x,::y,::z}.b32.b128
  rule <tk> exec(opc("clusterlaunchcontrol", M, _), ops(D, _, .Operands), _) => .K ... </tk>
       <regs> R => wr(R, D, 0) </regs>
    requires "query_cancel" in M
  rule isCollective("clusterlaunchcontrol") => true

  // ---------------------------------------------------------------- clocks and vector special registers
  // each read of %clock / %clock64 / %globaltimer observes a larger value
  syntax Bool ::= isClockReg(String) [function, total]
  rule isClockReg(S) => S in (SetItem("%clock") SetItem("%clock64") SetItem("%globaltimer") SetItem("%globaltimer_lo") SetItem("%globaltimer_hi"))
  syntax Int ::= clockVal(String, Int) [function, total]
  rule clockVal("%globaltimer_hi", T) => T >>Int 32
  rule clockVal(_, T) => T [owise]
  rule <tk> exec(opc("mov", _, ListItem(T)), ops(D, S:Reg, .Operands), _) => .K ... </tk>
       <regs> R => wr(R, D, trunc(clockVal(Reg2String(S), TICK), wid(T))) </regs>
       <tick> TICK => TICK +Int 25 </tick>
    requires isClockReg(Reg2String(S)) [priority(40)]

  // mov.v4.u32 {x, y, z, w}, %tid;   (w reads as 0)
  syntax Bool ::= isVecSreg(String) [function, total]
  rule isVecSreg(S) => S in (SetItem("%tid") SetItem("%ntid") SetItem("%ctaid") SetItem("%nctaid") SetItem("%clusterid")
                              SetItem("%nclusterid") SetItem("%cluster_ctaid") SetItem("%cluster_nctaid"))
  rule <tk> exec(opc("mov", M, ListItem(T)), ops({ Ds }, S:Reg, .Operands), C) => .K ... </tk>
       <regs> R => vecSregs(R, opsList(Ds), Reg2String(S), 0, C) </regs>
    requires isVecSreg(Reg2String(S)) andBool ("v4" in M orBool "v2" in M) [priority(40)]
  syntax Map ::= vecSregs(Map, List, String, Int, Ctx) [function, total]
  rule vecSregs(R, ListItem(D:Operand) Ds, S, I, C) =>
       vecSregs(wr(R, D, #if I <Int 3 #then {R[S +String sregSuffix(I)] orDefault 0}:>Int #else 0 #fi), Ds, S, I +Int 1, C)
  rule vecSregs(R, _, _, _, _) => R [owise]
  syntax String ::= sregSuffix(Int) [function, total]
  rule sregSuffix(0) => ".x"   rule sregSuffix(1) => ".y"   rule sregSuffix(_) => ".z" [owise]

  // ---------------------------------------------------------------- dynamic stack: alloca / stacksave / stackrestore
  // The per-thread stack is the .local frame area (<lmem>, top <ltop>); ret restores the caller's top.
  // alloca.u64 yields a generic address in the local window, alloca.u32 a .local offset.
  syntax Int ::= allocaAlign(List, Ctx) [function, total]
  rule allocaAlign(ListItem(X:Operand) _, C) => maxInt(8, rd(X, ty("u", 32), C))
  rule allocaAlign(_, _) => 8 [owise]
  rule <tk> exec(opc("alloca", _, ListItem(T)), ops(D, Sz, Rest:Operands), C) => .K ... </tk>
       <regs> R => wr(R, D, trunc(alignUp(LT, allocaAlign(opsList(Rest), C)) +Int #if wid(T) ==Int 64 #then LOCBASE #else 0 #fi, wid(T))) </regs>
       <ltop> LT => alignUp(LT, allocaAlign(opsList(Rest), C)) +Int rd(Sz, ty("u", 32), C) </ltop>
  rule <tk> exec(opc("stacksave", _, ListItem(T)), ops(D, .Operands), _) => .K ... </tk>
       <regs> R => wr(R, D, trunc(LT +Int #if wid(T) ==Int 64 #then LOCBASE #else 0 #fi, wid(T))) </regs>
       <ltop> LT </ltop>
  rule <tk> exec(opc("stackrestore", _, ListItem(T)), ops(A, .Operands), C) => .K ... </tk>
       <ltop> _ => rd(A, T, C) -Int #if wid(T) ==Int 64 #then LOCBASE #else 0 #fi </ltop>
endmodule

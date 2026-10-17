// =============================================================================
// spcompress / spdecompress (PTX ISA 9.4, sm_107a): structured-sparsity (de)compression of register vectors.
//
//   spcompress.elemsize.idxsize.sp::2:4.numN  mdata, cdata, data, spdesc
//     data:  2*num 32-bit registers, i.e. `num` iterations of 64 bits; cdata: num registers; mdata: ceil(num*idxsize/elemsize).
//     In every group of four consecutive elements two are kept (spdesc[1:0]: 0 max, 1 maxabs, 2 min, 3 minabs; spdesc[4:2]: element
//     type); the kept elements are copied to cdata in index order and their indices (idxsize bits each) are appended to mdata.
//     Assumed where the ISA says "implementation-specific": ties keep the lowest index; NaNs are always kept (lowest index first).
//   spdecompress.elemsize.idxsize.sp::S:T.numN  data, mdata, cdata
//     for every iteration n and source element s: data[n][mdata[n][s]] = cdata[n][s]; all other elements are zero.
//
// Vectors are bit-packed little-endian across their 32-bit registers.  The selection helper is shared with tcgen05.ld.spcompress.
// =============================================================================
requires "ptx-exec-sp.k"

module PTX-EXEC-SPC
  imports PTX-EXEC-SP

  // ---------------------------------------------------------------- bit vectors
  syntax Int ::= bitsOf(List, Int) [function, total]                      // registers (LSB first), index -> big integer
  rule bitsOf(ListItem(W:Int) Ws, I) => (W &Int 4294967295) <<Int (32 *Int I) |Int bitsOf(Ws, I +Int 1)
  rule bitsOf(_, _) => 0 [owise]
  syntax List ::= wordsOf(Int, Int) [function, total]                     // big integer, count -> 32-bit registers
  rule wordsOf(V, N) => ListItem(V &Int 4294967295) wordsOf(V >>Int 32, N -Int 1) requires N >Int 0
  rule wordsOf(_, _) => .List [owise]
  syntax List ::= fitWords(List, Int) [function, total]                   // exactly N registers (pad with zero / truncate)
  rule fitWords(ListItem(W:Int) Ws, N) => ListItem(W) fitWords(Ws, N -Int 1) requires N >Int 0
  rule fitWords(_, N) => spcZeros(N) requires N >Int 0 [owise]
  rule fitWords(_, _) => .List [owise]
  syntax List ::= spcZeros(Int) [function, total]
  rule spcZeros(N) => ListItem(0) spcZeros(N -Int 1) requires N >Int 0
  rule spcZeros(_) => .List [owise]
  syntax List ::= elemsOf(Int, Int, Int, Int) [function, total]           // big integer, element bits, index, count -> elements
  rule elemsOf(V, B, I, N) => ListItem((V >>Int (B *Int I)) &Int mask(B)) elemsOf(V, B, I +Int 1, N) requires I <Int N
  rule elemsOf(_, _, _, _) => .List [owise]
  syntax Int ::= packEl(List, Int, Int) [function, total]                 // elements, element bits, index -> big integer
  rule packEl(ListItem(E:Int) Es, B, I) => (E &Int mask(B)) <<Int (B *Int I) |Int packEl(Es, B, I +Int 1)
  rule packEl(_, _, _) => 0 [owise]

  // ---------------------------------------------------------------- selection of two elements out of four
  // Float values with their NaN-ness; op: 0 max, 1 maxabs, 2 min, 3 minabs.
  syntax Float ::= spcKey(Int, Float) [function, total]
  rule spcKey(Op, F) => #if Op %Int 2 ==Int 1 #then absFloat(F) #else F #fi
  syntax Bool ::= spcBefore(Int, List, Int, Int) [function, total]        // does element J precede element I in the selection order?
  rule spcBefore(Op, Fs, J, I) =>
       #if isNaN({Fs[J]}:>Float) #then (notBool isNaN({Fs[I]}:>Float) orBool J <Int I)
       #else (notBool isNaN({Fs[I]}:>Float) andBool
              (#if Op <Int 2 #then spcKey(Op, {Fs[J]}:>Float) >Float spcKey(Op, {Fs[I]}:>Float) #else spcKey(Op, {Fs[J]}:>Float) <Float spcKey(Op, {Fs[I]}:>Float) #fi
               orBool (spcKey(Op, {Fs[J]}:>Float) ==Float spcKey(Op, {Fs[I]}:>Float) andBool J <Int I))) #fi
  syntax Int ::= spcRank(Int, List, Int, Int, Int) [function, total]      // number of elements preceding I
  rule spcRank(Op, Fs, I, J, Acc) => spcRank(Op, Fs, I, J +Int 1, #if J =/=Int I andBool spcBefore(Op, Fs, J, I) #then Acc +Int 1 #else Acc #fi) requires J <Int 4
  rule spcRank(_, _, _, _, Acc) => Acc [owise]
  syntax List ::= spcSel(Int, List) [function, total]                     // the (ascending) indices of the two kept elements of a group of four
  rule spcSel(Op, Fs) => spcSelI(Op, Fs, 0)
  syntax List ::= spcSelI(Int, List, Int) [function, total]
  rule spcSelI(Op, Fs, I) => #if spcRank(Op, Fs, I, 0, 0) <Int 2 #then ListItem(I) #else .List #fi spcSelI(Op, Fs, I +Int 1) requires I <Int 4
  rule spcSelI(_, _, _) => .List [owise]

  // ---------------------------------------------------------------- spcompress
  // element value from the descriptor's type field and the element size
  syntax Float ::= spcVal(Int, Int, Int) [function, total]                // type code, element bits, bits
  rule spcVal(0, 16, B) => elemF(ty("f", 16), B)
  rule spcVal(1, 16, B) => elemF(ty("bf", 16), B)
  rule spcVal(0, 8, B) => Int2Float(B &Int 255, 300, WEXP)
  rule spcVal(1, 8, B) => Int2Float(#if (B &Int 255) >=Int 128 #then (B &Int 255) -Int 256 #else B &Int 255 #fi, 300, WEXP)
  rule spcVal(6, _, B) => elemF(ty("f", 32), B)
  rule spcVal(2, _, B) => elemF(ty("e5m2", 8), B &Int 255)
  rule spcVal(3, _, B) => elemF(ty("e4m3", 8), B &Int 255)
  rule spcVal(4, _, B) => elemF(ty("e3m2", 8), B &Int 63)
  rule spcVal(5, _, B) => elemF(ty("e2m3", 8), B &Int 63)
  rule spcVal(_, _, B) => Int2Float(0, 300, WEXP) [owise]
  syntax List ::= spcVals(Int, Int, List) [function, total]
  rule spcVals(T, ES, ListItem(B:Int) Bs) => ListItem(spcVal(T, ES, B)) spcVals(T, ES, Bs)
  rule spcVals(_, _, _) => .List [owise]
  // for every group of four elements: kept element bits (in index order) and kept indices; result ListItem(elements) ListItem(indices)
  syntax List ::= spcKeep(Int, Int, Int, List, List, List) [function, total]       // op, type, element bits, elements, kept elements, kept indices
  rule spcKeep(Op, T, ES, ListItem(A:Int) ListItem(B:Int) ListItem(C:Int) ListItem(D:Int) Es, Cs, Ms) =>
       spcKeep(Op, T, ES, Es,
               Cs spcPickB(spcSel(Op, spcVals(T, ES, ListItem(A) ListItem(B) ListItem(C) ListItem(D))), ListItem(A) ListItem(B) ListItem(C) ListItem(D)),
               Ms spcSel(Op, spcVals(T, ES, ListItem(A) ListItem(B) ListItem(C) ListItem(D))))
  rule spcKeep(_, _, _, _, Cs, Ms) => ListItem(Cs) ListItem(Ms) [owise]
  syntax List ::= spcPickB(List, List) [function, total]
  rule spcPickB(ListItem(I:Int) Is, Bs) => ListItem({Bs[I]}:>Int) spcPickB(Is, Bs)
  rule spcPickB(_, _) => .List [owise]
  // the result of the whole instruction: ListItem(mdata regs) ListItem(cdata regs)
  syntax List ::= spcompress(Int, Int, Int, List, Int, Int) [function, total]   // spdesc, element bits, index bits, data regs, #mdata regs, #cdata regs
  rule spcompress(SD, ES, IB, Ds, NM, NC) =>
       spcFin(spcKeep(SD &Int 3, (SD >>Int 2) &Int 7, ES, elemsOf(bitsOf(Ds, 0), ES, 0, size(Ds) *Int 32 /Int ES), .List, .List), ES, IB, NM, NC)
  syntax List ::= spcFin(List, Int, Int, Int, Int) [function, total]
  rule spcFin(ListItem(Cs:List) ListItem(Ms:List), ES, IB, NM, NC) =>
       ListItem(fitWords(wordsOf(packEl(Ms, IB, 0), NM), NM)) ListItem(fitWords(wordsOf(packEl(Cs, ES, 0), NC), NC))
  rule spcFin(_, _, _, NM, NC) => ListItem(spcZeros(NM)) ListItem(spcZeros(NC)) [owise]

  syntax Int ::= spcNum(Set) [function, total]
  rule spcNum(M) => 64 requires "x64" in M
  rule spcNum(M) => 32 requires "x32" in M
  rule spcNum(M) => 16 requires "x16" in M
  rule spcNum(M) => 8 requires "x8" in M
  rule spcNum(M) => 4 requires "x4" in M
  rule spcNum(M) => 2 requires "x2" in M
  rule spcNum(_) => 1 [owise]
  syntax Int ::= spcES(List) [function, total] | spcIB(Set) [function, total]
  rule spcES(Ts) => 16 requires ty("b", 16) in Ts
  rule spcES(_) => 8 [owise]
  rule spcIB(M) => 4 requires "b4" in M
  rule spcIB(_) => 2 [owise]
  rule <tk> exec(opc("spcompress", M, Ts), ops(Mv, Cv, Dv, SD, .Operands), C) =>
            #wmWr(vecOps(Mv), {spcompress(rd(SD, ty("b", 32), C), spcES(Ts), spcIB(M), vecRegs(Dv, ty("b", 32), C), size(vecOps(Mv)), size(vecOps(Cv)))[0]}:>List)
            ~> #wmWr(vecOps(Cv), {spcompress(rd(SD, ty("b", 32), C), spcES(Ts), spcIB(M), vecRegs(Dv, ty("b", 32), C), size(vecOps(Mv)), size(vecOps(Cv)))[1]}:>List) ... </tk>

  // ---------------------------------------------------------------- spdecompress
  syntax Int ::= spfSrc(Set) [function, total] | spfDst(Set) [function, total]
  rule spfSrc(M) => 1 requires "sp::1:2" in M orBool "sp::1:4" in M orBool "sp::1:8" in M orBool "sp::1:16" in M
  rule spfSrc(M) => 2 requires "sp::2:4" in M orBool "sp::2:8" in M orBool "sp::2:16" in M
  rule spfSrc(M) => 4 requires "sp::4:8" in M orBool "sp::4:16" in M
  rule spfSrc(_) => 2 [owise]
  rule spfDst(M) => 2 requires "sp::1:2" in M
  rule spfDst(M) => 4 requires "sp::1:4" in M orBool "sp::2:4" in M
  rule spfDst(M) => 8 requires "sp::1:8" in M orBool "sp::2:8" in M orBool "sp::4:8" in M
  rule spfDst(M) => 16 requires "sp::1:16" in M orBool "sp::2:16" in M orBool "sp::4:16" in M
  rule spfDst(_) => 4 [owise]
  syntax Int ::= spdLoop(Int, Int, Int, Int, Int, Int, Int, Int, Int, Int) [function, total]
  // cdata bits, mdata bits, element bits, index bits, source count, target count, iterations, n, s, accumulated data bits
  rule spdLoop(Cb, Mb, ES, IB, NS, NT, NN, N, S, Acc) =>
       spdLoop(Cb, Mb, ES, IB, NS, NT, NN, #if S +Int 1 ==Int NS #then N +Int 1 #else N #fi, #if S +Int 1 ==Int NS #then 0 #else S +Int 1 #fi,
               Acc |Int ((((Cb >>Int ((N *Int NS +Int S) *Int ES)) &Int mask(ES)) <<Int ((N *Int NT +Int ((Mb >>Int ((N *Int NS +Int S) *Int IB)) &Int mask(IB))) *Int ES))))
    requires N <Int NN
  rule spdLoop(_, _, _, _, _, _, _, _, _, Acc) => Acc [owise]
  rule <tk> exec(opc("spdecompress", M, Ts), ops(Dv, Mv, Cv, .Operands), C) =>
            #wmWr(vecOps(Dv), fitWords(wordsOf(spdLoop(bitsOf(vecRegs(Cv, ty("b", 32), C), 0), bitsOf(vecRegs(Mv, ty("b", 32), C), 0), spcES(Ts), spcIB(M), spfSrc(M), spfDst(M),
                                                      spcNum(M), 0, 0, 0), size(vecOps(Dv))), size(vecOps(Dv)))) ... </tk>
endmodule

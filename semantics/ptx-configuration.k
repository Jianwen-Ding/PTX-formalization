// =============================================================================
// PTX-CONFIGURATION: the machine state.
//
//   <ptx>
//     <k>        host-side driver: processes module directives and harness
//                commands (.buffer / .launch / .print / .expect) in order
//     <funcs>    loaded .entry/.func definitions
//     <syms>     module-scope symbols (global / const / shared variables, buffers)
//     <gmem>     global (and const) memory: byte-addressed, little-endian
//     <ctas>     one <cta> per cooperative thread array of the running grid;
//                each owns its shared memory, barrier state and <thread>s
//   </ptx>
//
// Every thread has its own computation cell <tk>, register file, parameter
// space, local memory and call stack.  Threads take steps independently, so
// the semantics is the set of all instruction-granularity interleavings
// (sequentially consistent memory model); use `krun --search` to explore it.
//
// <turn> implements a partial-order reduction: a thread that holds the turn
// runs its thread-local instructions (which commute with every step of every
// other thread) without interruption, and gives the turn up right before an
// instruction that touches shared state (global/shared/generic memory,
// barriers).  Only those visible steps are interleaving points, so the set of
// reachable outcomes is unchanged while `--search` explores far fewer states.
//   -1 : free      -2 : held by the host (no grid running)      n : thread uid
// =============================================================================
requires "ptx-common.k"

module PTX-CONFIGURATION
  imports PTX-COMMON

  syntax Dim3    ::= dim3(Int, Int, Int)
  syntax TStatus ::= "running"
                   | waiting(Int, Int)      // barrier id, barrier generation

  // symbol: state space + address within that space
  syntax Sym ::= sym(String, Int) | noSym(String)

  // function definition: kind ("entry"/"func"), return params, params,
  // code (pc |-> instruction), labels (label |-> pc), register widths
  syntax ClBar ::= cl(Int, Int)
  syntax MbState ::= mb(Int, Int, Int, Int, Int)
  syntax FDef  ::= fdef(String, List, List, Map, Map, Map)
  syntax PDecl ::= pdecl(String, Ty, Int, Int)          // name, type, size, align
  // static variable: space, name, type, size, align, extern?, initializer
  syntax SDecl ::= sdecl(String, String, Ty, Int, Int, Bool, List)

  // saved caller frame: function, return pc, registers, register widths,
  // symbol environment, local-memory top
  syntax Frame ::= frame(String, Int, Map, Map, Map, Int)

  // warp-collective rendezvous (see ptx-exec-coll.k): key = (warp, kind, mask,
  // per-thread sequence number); entry = (kind, data, mask, arrivals: lane |->
  // contribution, results: lane |-> result, number of lanes that took their result)
  syntax CollKey ::= ck(Int, String, Int, Int)
  syntax Coll    ::= coll(String, List, Int, Map, Map, Int)

  // buffer allocated by the harness: address, element type, element count
  syntax Buf ::= buf(Int, Ty, Int)

  configuration
    <ptx>
      <k> $PGM:Module </k>
      <out stream="stdout"> .List </out>
      <exit-code exit=""> 1 </exit-code>
      <fails> 0 </fails>
      <funcs> .Map </funcs>
      <syms> .Map </syms>
      <bufs> .Map </bufs>
      <gmem> .Bytes </gmem>
      <gtop> 4096 </gtop>
      <turn> -2 </turn>
      <clusterDims> .Map </clusterDims>       // kernel |-> .reqnctapercluster
      <mbars> .Map </mbars>                   // CTA * 2^32 + shared offset |-> mbarrier object mb(phase, pending, expected, tx-count, layout)
      <clbar> .Map </clbar>                   // cluster id |-> cl(arrived threads, generation) for barrier.cluster
      <stop> 0 </stop>
      <launch>
        <kernel> "" </kernel>
        <gridDim> dim3(1, 1, 1) </gridDim>
        <blockDim> dim3(1, 1, 1) </blockDim>
        <clusterDim> dim3(1, 1, 1) </clusterDim>
        <kparams> .Bytes </kparams>
        <kenv> .Map </kenv>
        <kptop> 0 </kptop>
      </launch>
      <ctas>
        <cta multiplicity="*" type="Map">
          <ctaid> 0 </ctaid>
          <smem> .Bytes </smem>
          <live> 0 </live>
          <barCnt> .Map </barCnt>
          <barExp> .Map </barExp>
          <barGen> .Map </barGen>
          <barVal> .Map </barVal>               // barrier id |-> reduction accumulated in the current generation
          <barRes> .Map </barRes>               // barrier id |-> reduction result of the last completed generation
          <colls> .Map </colls>
          <wlive> .Map </wlive>                 // warp |-> mask of lanes that have not exited
          <tmem> .Map </tmem>                   // tensor memory: lane * 65536 + column |-> 32-bit word (sm_100a)
          <tmres> .Map </tmres>                 // warp-collective tcgen05.alloc/dealloc instance |-> result (done once per warp)
          <tmalloc> .Map </tmalloc>             // tensor memory allocations: first column |-> number of columns
          <threads>
            <thread multiplicity="*" type="Map">
              <tid> 0 </tid>
              <tk> .K </tk>
              <status> running </status>
              <fn> "" </fn>
              <pc> 0 </pc>
              <regs> .Map </regs>
              <rw> .Map </rw>
              <env> .Map </env>
              <stack> .List </stack>
              <sregs> .Map </sregs>
              <pmem> .Bytes </pmem>
              <ptop> 0 </ptop>
              <cseq> .Map </cseq>
              <lmem> .Bytes </lmem>
              <ltop> 0 </ltop>
            </thread>
          </threads>
        </cta>
      </ctas>
    </ptx>

  // ------------------------------------------------------------ address map
  // Generic addressing: global addresses are used as-is; shared and local
  // state spaces are mapped into disjoint windows of the generic space.
  syntax Int ::= "SHBASE"  [macro] | "LOCBASE" [macro] | "DYNSHARED" [macro]
  rule SHBASE    => 1099511627776     // 2^40
  rule LOCBASE   => 2199023255552     // 2^41
  rule DYNSHARED => 16                // alignment of `.extern .shared` arrays,
                                      // placed after all static shared data

  syntax Int ::= winBase(String) [function, total]
  rule winBase("shared") => SHBASE
  rule winBase("local")  => LOCBASE
  rule winBase(_)        => 0 [owise]

  syntax Int ::= alignUp(Int, Int) [function, total]
  rule alignUp(X, A) => ((X +Int A -Int 1) /Int A) *Int A requires A >Int 0
  rule alignUp(X, _) => X [owise]

  // ------------------------------------------------------------ byte memory
  // Every state space is a little-endian byte array built on K's builtin
  // BYTES module; reads past the end yield zero bytes, writes grow the array.
  syntax Int ::= readBytes(Bytes, Int, Int) [function, total]
  rule readBytes(B, A, N) =>
       Bytes2Int(substrBytes(padRightBytes(B, A +Int N, 0), A, A +Int N), LE, Unsigned)
    requires A >=Int 0 andBool N >Int 0
  rule readBytes(_, _, _) => 0 [owise]
  syntax Bytes ::= writeBytes(Bytes, Int, Int, Int) [function, total]
  rule writeBytes(B, A, N, V) =>
       replaceAtBytes(padRightBytes(B, A +Int N, 0), A, Int2Bytes(N, V &Int mask(8 *Int N), LE))
    requires A >=Int 0 andBool N >Int 0
  rule writeBytes(B, _, _, _) => B [owise]

  // ------------------------------------------------------------ list helpers
  syntax List ::= opsList(Operands) [function, total]
  rule opsList(O:Operand, Os:Operands) => ListItem(O) opsList(Os)
  rule opsList(.Operands) => .List

  syntax List ::= ivList(InitVals) [function, total]       // flattens nested { }
  rule ivList({ Vs:InitVals }, Rest:InitVals) => ivList(Vs) ivList(Rest)
  rule ivList(V:InitVal, Rest:InitVals) => ListItem(V) ivList(Rest) [owise]
  rule ivList(.InitVals) => .List

  // Symbolic initializers (sym, sym+4, generic(sym), a-b) are resolved against
  // the symbol table before being turned into bits.
  syntax InitVal ::= resolveV(Map, InitVal) [function, total]
  syntax Int ::= symRaw(Map, Name) [function, total]
               | symGen(Map, Name) [function, total]
  rule symRaw(S, N) => symA({S[Name2String(N)]}:>Sym) requires Name2String(N) in_keys(S)
  rule symRaw(_, _) => 0 [owise]
  rule symGen(S, N) => winBase(symSp({S[Name2String(N)]}:>Sym)) +Int symRaw(S, N) requires Name2String(N) in_keys(S)
  rule symGen(_, _) => 0 [owise]
  syntax Int ::= symA(Sym) [function, total]
  rule symA(sym(_, A)) => A
  rule symA(noSym(_)) => 0
  syntax String ::= symSp(Sym) [function, total]
  rule symSp(sym(SP, _)) => SP
  rule symSp(noSym(_)) => "global"
  rule resolveV(S, N:Name) => symRaw(S, N)
  rule resolveV(S, N:Name I:Int) => symRaw(S, N) +Int I
  rule resolveV(S, N:Name + I:Int) => symRaw(S, N) +Int I
  rule resolveV(S, A:Name - B:Name) => symRaw(S, A) -Int symRaw(S, B)
  rule resolveV(S, generic ( N:Name )) => symGen(S, N)
  rule resolveV(_, V) => V [owise]
  syntax List ::= resolveL(Map, List) [function, total]
  rule resolveL(S, ListItem(V:InitVal) L) => ListItem(resolveV(S, V)) resolveL(S, L)
  rule resolveL(_, _) => .List [owise]

  syntax Bytes ::= initMem(Bytes, Int, Ty, List) [function]
  rule initMem(M, A, T, ListItem(V:InitVal) L) => initMem(writeBytes(M, A, bytes(T), initBits(V, T)), A +Int bytes(T), T, L)
  rule initMem(M, _, _, _) => M [owise]
endmodule

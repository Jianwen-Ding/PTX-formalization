/* =============================================================================
 * PTX-EXEC-CORE (semantics/ptx-exec-core.k, ptx-exec-mem.k): operand access in the context of the
 * executing thread `cur` (K's ctx(regs, env, syms, labels, rw)), effective addresses and the
 * byte memories.
 * ============================================================================= */

val cur_fn : unit -> int
function cur_fn() = t_fn[tchk(cur)]
val cur_regs : unit -> int
function cur_regs() = t_regs[tchk(cur)]
val cur_rw : unit -> int
function cur_rw() = f_rw[fchk(cur_fn())]
val cur_lab : unit -> int
function cur_lab() = f_lab[fchk(cur_fn())]

/* rreg(R, C) */
val rreg : string -> int
function rreg(r) = smap_get(cur_regs(), r)

val wreg : (string, int) -> unit
function wreg(r, v) = smap_set(cur_regs(), r, v)

val isRegName : string -> bool
function isRegName(n) = smap_has(cur_rw(), n)

/* lookupSym(N, C): the environment first, then the module's symbols; -1 = noSym */
let NOSYM : int = -1
val lookupSym : string -> int
function lookupSym(n) = {
  let e = t_env[tchk(cur)];
  if smap_has(e, n) then smap_get(e, n)
  else if smap_has(SP_SYMS, n) then smap_get(SP_SYMS, n)
  else NOSYM
}
val symAddr : int -> int
function symAddr(s) = if s < 0 then 0 else sym_addr(s)

/* rd(Operand, Ty, Ctx) */
val rd_sop : (sop, ty) -> int
function rd_sop(o, t) = match o {
  SR(r) => trunc(rreg(r), wid(t)),
  SNOT(_, r) => b2i(rreg(r) == 0),
  SI(i) => intLit(i, t),
  SF(f) => fltLit(f, t),
  SH(h) => hexBits(h, t),
  SN(n) => {
    if isRegName(n) then trunc(rreg(n), wid(t))
    else if n == "WARP_SZ" then trunc(32, wid(t))
    else trunc(symAddr(lookupSym(n)), wid(t))
  },
  SNP(n, i) => trunc(symAddr(lookupSym(n)) + i, wid(t)),
  _ => { fault("operand cannot be read", 0); 0 }
}

val rd : (xop, ty) -> int
function rd(o, t) = match o {
  XS(s) => rd_sop(s, t),
  _ => { fault("vector operand where a scalar is expected", 0); 0 }
}

val rdp : xop -> int
function rdp(o) = rd(o, mkty("p", 1))
val rdu32 : xop -> int
function rdu32(o) = rd(o, mkty("u", 32))
val rdb32 : xop -> int
function rdb32(o) = rd(o, mkty("b", 32))
val rdu64 : xop -> int
function rdu64(o) = rd(o, mkty("u", 64))

/* wr(R, D, V): registers, names (`_` is a sink) */
val wr_sop : (sop, int) -> unit
function wr_sop(d, v) = match d {
  /* special registers are predefined, read-only variables (ISA 10) */
  SR(r) => if smap_has(t_sregs[tchk(cur)], r) then fault(concat_str("write to the read-only special register ", r), 0) else wreg(r, v),
  SN(n) => if n != "_" then wreg(n, v),
  _ => ()
}
val wr : (xop, int) -> unit
function wr(d, v) = match d { XS(s) => wr_sop(s, v), _ => () }

val sop_name : sop -> string
function sop_name(s) = match s { SR(r) => r, SN(n) => n, _ => "" }
/* opStr */
val opStr : xop -> string
function opStr(o) = match o { XS(s) => sop_name(s), _ => "" }

/* regW(RW, D): declared width of a destination register, default 32 */
val regW : xop -> int
function regW(d) = {
  let n = opStr(d);
  if n != "" & smap_has(cur_rw(), n) then smap_get(cur_rw(), n) else 32
}

/* extReg: a narrow signed value is sign-extended to the register that holds it */
val extReg : (int, ty, xop) -> int
function extReg(v, t, d) = if isS(t) & wid(t) < regW(d) then trunc(sext(v, wid(t)), regW(d)) else v

val isPacked : ty -> bool
function isPacked(t) = t.k == "s16x2" | t.k == "u16x2" | t.k == "f16x2" | t.k == "bf16x2" | t.k == "f32x2" | t.k == "u8x4" | t.k == "s8x4"

val isVec : xop -> bool
function isVec(o) = match o { XV(_) => true, _ => false }

val vec_items : xop -> list(sop)
function vec_items(o) = match o { XV(l) => l, _ => [||] }

/* ---------------------------------------------------------------- memory locations */
union loc = {
  Loc : (string, int),
  BadLoc : string,
}

val mkLoc : (string, int) -> loc
function mkLoc(s, a) = {
  if s == "generic" then {
    let w = genWin(a);
    let o = a - winBase(w);
    if w == "cluster" then Loc("cluster", lor_int(shl_int(asr_int(o, 24) + 1, 24), land_int(o, 16777215)))
    else if w == "const" then Loc("global", o)
    else Loc(w, o)
  } else if s == "const" then Loc("global", a)
  else if s == "shared_cluster" then Loc("cluster", a)
  else Loc(s, a)
}

/* `.address_size 32`: addresses wrap modulo 2^32, 64-bit registers are rejected */
val mkLocA : (string, int, string) -> loc
function mkLocA(s, v, r) = {
  if not_bool(asz32) then mkLoc(s, v)
  else if (if smap_has(cur_rw(), r) then smap_get(cur_rw(), r) else 32) == 64 then
    BadLoc(concat_str("address_size 32: 64-bit register ", concat_str(r, " used as an address")))
  else mkLoc(s, fmod_int(v, 4294967296))
}

val symLoc : (int, string, int) -> loc
function symLoc(s, n, i) = {
  if s < 0 | sym_space(s) == "nosym" then BadLoc(concat_str("undefined symbol ", n))
  else if sym_space(s) == "const" then Loc("global", sym_addr(s) + i)
  else Loc(sym_space(s), sym_addr(s) + i)
}

val ea : (addr, string) -> loc
function ea(a, s) = match a {
  AR(r) => mkLocA(s, rreg(r), r),
  ARO(r, i) => mkLocA(s, rreg(r) + i, r),
  AN(n) => if isRegName(n) then mkLocA(s, rreg(n), n) else symLoc(lookupSym(n), n, 0),
  ANO(n, i) => if isRegName(n) then mkLocA(s, rreg(n) + i, n) else symLoc(lookupSym(n), n, i),
  AI(i) => mkLoc(s, i),
  ANR(n, r, i) => symLoc(lookupSym(n), n, rreg(r) + i)
}

/* the address operand of [ A ] */
val mem_addr : xop -> option(addr)
function mem_addr(o) = match o { XB(a :: _, _) => Some(a), _ => None() }

/* is the operand exactly [ A ] */
val is_mem1 : xop -> bool
function is_mem1(o) = match o { XB([|_|], [||]) => true, _ => false }

/* ---------------------------------------------------------------- memory operations (#mem) */
union memop = {
  RdOp : int,                          /* bytes */
  InsOp : (int, int, int),             /* bit offset, width, value */
  WrOp : (int, int),                   /* bytes, value */
  AtOp : (string, ty, int, int),       /* op, type, b, c */
  BulkRedOp : (string, ty, int, int),  /* bulkRed(op, element type, bytes, source array) of ptx-exec-bulk.k */
}

val memBytes : memop -> int
function memBytes(op) = match op { RdOp(n) => n, WrOp(n, _) => n, AtOp(_, t, _, _) => bytes(t), _ => 0 }

/* the byte space of a location for the executing thread */
val cta_smem : unit -> int
function cta_smem() = c_smem[cchk(cur_cta())]

/* memNew / memVal on a byte space */
val mem_apply : (int, int, memop) -> int
function mem_apply(sp, a, op) = match op {
  RdOp(n) => mem_rdn(sp, a, n),
  WrOp(n, v) => { mem_wrn(sp, a, n, v); 0 },
  AtOp(o, t, b, c) => {
    let old = mem_rdn(sp, a, bytes(t));
    mem_wrn(sp, a, bytes(t), atomNew(o, t, old, b, c));
    old
  },
  BulkRedOp(o, t, n, v) => {
    let old = mem_rdn(sp, a, n);
    let eb = 8 * bytes(t);
    var r : int = 0;
    foreach (i from 0 to (tdiv_int(n, bytes(t)) - 1))
      r = lor_int(r, shl_int(atomNew(o, t, land_int(asr_int(old, eb * i), mask(eb)), land_int(asr_int(v, eb * i), mask(eb)), 0), eb * i));
    mem_wrn(sp, a, n, r);
    0
  },
  InsOp(o, w, v) => {
    let old = mem_rdn(sp, a, 1);
    mem_wrn(sp, a, 1, lor_int(land_int(old, lnot_int(shl_int(mask(w), o))), shl_int(land_int(v, mask(w)), o)));
    0
  }
}

/* hook for the cluster module: accesses of shared::cluster locations */
val cluster_mem : (int, memop) -> option(int)

/* [a, a + n) overlaps [%reserved_smem_offset_begin, %reserved_smem_offset_end) of the executing thread's CTA */
val resvHit : (int, int) -> bool
function resvHit(a, n) = {
  let sp = t_sregs[tchk(cur)];
  if smap_has(sp, "%reserved_smem_offset_begin") then {
    let b = smap_get(sp, "%reserved_smem_offset_begin");
    let e = smap_get(sp, "%reserved_smem_offset_end");
    a < e & a + n > b
  } else false
}

/* [a, a + n) overlaps a multimem handle's address range */
val mmHit : (int, int) -> bool
function mmHit(a, n) = {
  var hit : bool = false;
  var l = mmtab;
  while is_cons(l) & not_bool(hit) do { match l { (h, sz, _) :: rest => { if a < h + sz & a + n > h then hit = true; l = rest }, [||] => () } };
  hit
}

/* [a, a + n) overlaps an initialised mbarrier object (8 bytes, keyed by CTA and offset as in ptx_async.sail) */
val mbarHit : (int, int) -> bool
function mbarHit(a, n) = {
  var hit : bool = false;
  var x : int = a - tmod_int(a, 8);
  while x < a + n & not_bool(hit) do {
    if smap_has(SP_MBARS, concat_str("p", dec_str(cur_cta() * 4294967296 + x))) then hit = true;
    x = x + 8
  };
  hit
}

/* [a, a + n) reaches past the CTA's shared memory (static, dynamic and the reserved region after them) */
val resvAbove : (int, int) -> bool
function resvAbove(a, n) = {
  let sp = t_sregs[tchk(cur)];
  smap_has(sp, "%reserved_smem_offset_end") & a + n > smap_get(sp, "%reserved_smem_offset_end")
}

/* #mem(L, OP) => #res(V): one atomic step */
val memop_do : (loc, memop) -> int
function memop_do(l, op) = match l {
  BadLoc(s) => { fault(s, 0); 0 },
  Loc(s, a) => {
    let nb = memBytes(op);
    if ub & (nb == 2 | nb == 4 | nb == 8 | nb == 16) & tmod_int(a, nb) != 0 then {
      fault(concat_str("misaligned ", concat_str(s, concat_str(" access of ", concat_str(dec_str(nb), " bytes")))), a); 0
    } else if ub & s == "global" & nb > 0 & (a < 4096 | a + nb > gtop) then {
      fault("global access outside the allocated data", a); 0
    } else if ub & s == "local" & nb > 0 & a + nb > t_ltop[tchk(cur)] & (a < t_lsp[tchk(cur)] | a + nb > LSTACK) then {
      fault("local access above the stack top", a); 0
    } else if ub & s == "shared" & nb > 0 & cur >= 0 & resvHit(a, nb) then {
      /* "This region of shared memory is not available to users, and accessing this region from user code results in
         undefined behavior" (10.29) */
      fault("shared access to the region reserved for system software", a); 0
    } else if ub & s == "global" & nb > 0 & mmHit(a, nb) then {
      /* "Multimem addresses can be accessed only by multimem.* operations. Accessing a multimem address with ld, st or
         any other memory operations results in undefined behavior" (9.7.9.15) */
      fault("access to a multimem address by a non-multimem operation", a); 0
    } else if ub & s == "shared" & nb > 0 & cur >= 0 & mbarHit(a, nb) then {
      /* "Performing any non-mbarrier or mbarrier.init operations on an initialized mbarrier object results in undefined
         behavior" (9.7.14.16) */
      fault("shared access to a valid mbarrier object", a); 0
    } else if ub & s == "shared" & nb > 0 & cur >= 0 & resvAbove(a, nb) then {
      fault("shared access outside the CTA's shared memory", a); 0
    } else if s == "global" & a >= 4096 then mem_apply(SP_GMEM, a, op)
    else if s == "shared" & a >= 0 then mem_apply(cta_smem(), a, op)
    else if s == "local" & a >= 0 then mem_apply(t_lmem[tchk(cur)], a, op)
    else if s == "param" & a >= 0 then mem_apply(t_pmem[tchk(cur)], a, op)
    else if s == "cluster" then match cluster_mem(a, op) { Some(v) => v, None() => { fault(concat_str("invalid ", concat_str(s, " address")), a); 0 } }
    else { fault(concat_str("invalid ", concat_str(s, " address")), a); 0 }
  }
}

/* #ldwb(D, T): write back a loaded value (sub-word signed loads sign-extend to the register width) */
val ldwb : (xop, ty, int) -> unit
function ldwb(d, t, v) = wr(d, extReg(v, t, d))

val loc_add : (loc, int) -> loc
function loc_add(l, n) = match l { Loc(s, a) => Loc(s, a + n), _ => l }

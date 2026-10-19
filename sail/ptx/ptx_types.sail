/* =============================================================================
 * PTX in Sail: the instruction representation.
 *
 * As in semantics/ptx-common.k, an opcode token `ld.global.nc.f32` is kept as its base name, its
 * modifiers and its types (decodeOpc); the operands are the terms of ptx-syntax.k.  Sail forbids
 * recursive types, so operands are layered: simple operands (sop), addresses, operands inside
 * brackets (iop), and instruction operands (xop).
 * ============================================================================= */

/* ty(kind, width): kind "u" "s" "b" "f" "p" "bf" "tf" "f16x2" ... */
struct ty = {
  k : string,
  w : int,
}

val mkty : (string, int) -> ty
function mkty(k, w) = struct { k = k, w = w }

/* simple operands: Reg (%r1), Name (registers declared without %, symbols, labels, functions, `_`),
 * Int, Float (binary64 bits of the literal), HexLit (the token), Name + Int, - Reg / - Name,
 * ! Reg / ! Name, Reg | Reg / Name | Name.  The bool of SNEG / SNOT / SPIPE: true = Reg token. */
union sop = {
  SR : string,
  SN : string,
  SI : int,
  SF : int,
  SH : string,
  SNP : (string, int),
  SNEG : (bool, string),
  SNOT : (bool, string),
  SPIPE : (bool, string, bool, string),
}

/* Addr: Reg, Name, Int, Reg + Int (also Reg Int), Name + Int (also Name Int) */
union addr = {
  AR : string,
  AN : string,
  AI : int,
  ARO : (string, int),
  ANO : (string, int),
}

/* an operand inside [ ... ] after the addresses: simple or a vector { ... } */
union iop = {
  IS : sop,
  IV : list(sop),
}

/* instruction operands: simple, { Os }, ( Os ), { Os } | P, [ addresses, operands ] */
union xop = {
  XS : sop,
  XV : list(sop),
  XP : list(sop),
  XVP : (list(sop), bool, string),
  XB : (list(addr), list(iop)),
}

/* opc(base, modifiers, types); `shared::cta` is recorded as both "shared::cta" and "shared" */
struct opcode = {
  base : string,
  mods : list(string),
  tys : list(ty),
}

/* code items of ptx-load.k: ins(guard, opcode, operands) and declVar(space, name, size, align) */
union item = {
  INS : (option((string, bool)), opcode, list(xop)),
  DECLVAR : (string, string, int, int),
}

/* parameter declaration pdecl(name, type, size, align) */
struct pdecl = {
  pname : string,
  pty : ty,
  psize : int,
  palign : int,
}

/* static declaration sdecl(space, name, type, size, align, extern?, initializer) */
union initval = {
  IVI : int,
  IVF : int,
  IVH : string,
  IVN : string,
  IVNI : (string, int),
  IVND : (string, string),
  IVG : string,
  IVGO : (string, int),
}

struct sdecl = {
  sspace : string,
  sname : string,
  sty : ty,
  ssize : int,
  salign : int,
  sextern : bool,
  sinit : list(initval),
}

/* ---------------------------------------------------------------- modifier sets and lists */
val has : (list(string), string) -> bool
function has(l, s) = match l { [||] => false, h :: t => h == s | has(t, s) }

/* pick(M, candidates, default): first candidate in M */
val pick : (list(string), list(string), string) -> string
function pick(m, c, d) = match c { [||] => d, h :: t => if has(m, h) then h else pick(m, t, d) }

val ty_nth : (list(ty), int) -> ty
function ty_nth(l, i) = match l { [||] => mkty("b", 32), h :: t => if i == 0 then h else ty_nth(t, i - 1) }

val ty_count : list(ty) -> int
function ty_count(l) = match l { [||] => 0, _ :: t => 1 + ty_count(t) }

val xop_count : list(xop) -> int
function xop_count(l) = match l { [||] => 0, _ :: t => 1 + xop_count(t) }

val sop_count : list(sop) -> int
function sop_count(l) = match l { [||] => 0, _ :: t => 1 + sop_count(t) }

val xop_nth : (list(xop), int) -> xop
function xop_nth(l, i) = match l { [||] => XS(SI(0)), h :: t => if i == 0 then h else xop_nth(t, i - 1) }

val sop_nth : (list(sop), int) -> sop
function sop_nth(l, i) = match l { [||] => SI(0), h :: t => if i == 0 then h else sop_nth(t, i - 1) }

val ty_eq : (ty, ty) -> bool
function ty_eq(a, b) = a.k == b.k & a.w == b.w

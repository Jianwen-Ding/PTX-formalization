/* =============================================================================
 * AGX (G13 / G14) binary encodings: `mapping clause encdec`.
 *
 * Bit layouts follow dougallj/applegpu (applegpu.py instruction and operand descriptors).  An
 * instruction word is little-endian; `encdec` maps an AST to its word zero-extended to 80 bits
 * (the longest form, 10 bytes).  Instructions with a short and a long form use the long form
 * (length bit set) exactly when a bit beyond the short form is non-zero, like applegpu's
 * patch_raw_fields.  When decoding a byte stream, `agx_size` gives the length of the instruction
 * at the start of an 80-bit window; the window is then masked to that length.
 *
 * Each instruction class has an encoder e_<cls>(fields) : option(int) and a decoder
 * d_<cls>(word) : option(ast); the operand helpers mirror applegpu's OperandDesc classes
 * (ALUDstDesc, ALUSrcDesc, MulSrcDesc, FloatSrcDesc, CmpselSrcDesc, MemoryRegDesc, ...).
 * Encodings that applegpu marks UNDEFINED or ambiguous (e.g. odd register numbers for 32-bit
 * operands) are rejected by the decoder, so that decode is the inverse of encode.
 * ============================================================================= */

register enc_err : bool
register dec_err : bool

/* ---------------------------------------------------------------- fields */
val fput : (int, int, int, int) -> int
function fput(w, start, size, v) = {
  if v < 0 | v >= pow2(size) then enc_err = true;
  lor_int(w, shl_int(trunc_int(v, size), start))
}

/* a merged field: the low n1 bits at s1, the next n2 bits at s2 */
val fput2 : (int, int, int, int, int, int) -> int
function fput2(w, s1, n1, s2, n2, v) = {
  if v < 0 | v >= pow2(n1 + n2) then enc_err = true;
  fput(fput(w, s1, n1, trunc_int(v, n1)), s2, n2, trunc_int(shr_int(v, n1), n2))
}

val fput3 : (int, int, int, int, int, int, int, int) -> int
function fput3(w, s1, n1, s2, n2, s3, n3, v) = {
  if v < 0 | v >= pow2(n1 + n2 + n3) then enc_err = true;
  fput(fput2(w, s1, n1, s2, n2, trunc_int(v, n1 + n2)), s3, n3, trunc_int(shr_int(v, n1 + n2), n3))
}

val fget : (int, int, int) -> int
function fget(w, start, size) = trunc_int(shr_int(w, start), size)

val fget2 : (int, int, int, int, int) -> int
function fget2(w, s1, n1, s2, n2) = fget(w, s1, n1) + shl_int(fget(w, s2, n2), n1)

val fget3 : (int, int, int, int, int, int, int) -> int
function fget3(w, s1, n1, s2, n2, s3, n3) = fget2(w, s1, n1, s2, n2) + shl_int(fget(w, s3, n3), n1 + n2)

val bool_bit : bool -> int
function bool_bit(b) = if b then 1 else 0

val dchk : bool -> unit
function dchk(c) = if not_bool(c) then dec_err = true

/* ---------------------------------------------------------------- instruction classes */
struct icls = {
  cid : int,           /* class number */
  short : int,         /* bytes */
  long : int,
  lbit : int,          /* length bit position (when short != long) */
  consts : list((int, int, int)),   /* (start, size, value) */
}

val cls : (int, int, int, int, list((int, int, int))) -> icls
function cls(i, s, l, b, c) = struct { cid = i, short = s, long = l, lbit = b, consts = c }

let C_MOVI16 = cls(1, 4, 6, 15, [|(0, 7, 98), (8, 1, 0)|])
let C_MOVI32 = cls(2, 6, 8, 15, [|(0, 7, 98), (8, 1, 1)|])
let C_GETSR = cls(3, 4, 4, 15, [|(0, 7, 114), (15, 1, 0)|])
let C_IADD = cls(4, 8, 8, 15, [|(0, 6, 14), (15, 1, 0)|])
let C_IMADD = cls(5, 8, 8, 15, [|(0, 6, 30), (15, 1, 0)|])
let C_BFI = cls(6, 8, 8, 15, [|(0, 7, 46), (15, 1, 0), (26, 2, 0)|])
let C_BFEIL = cls(7, 8, 8, 15, [|(0, 7, 46), (15, 1, 1), (26, 2, 0)|])
let C_ASR = cls(8, 8, 8, 15, [|(0, 7, 46), (15, 1, 1), (26, 2, 1)|])
let C_BITOP = cls(9, 6, 6, 15, [|(0, 7, 126), (15, 1, 0)|])
let C_BITREV = cls(10, 6, 6, 15, [|(0, 7, 62), (15, 1, 0), (28, 10, 0), (38, 2, 0), (26, 2, 1)|])
let C_POPCOUNT = cls(11, 6, 6, 15, [|(0, 7, 62), (15, 1, 0), (28, 10, 0), (38, 2, 0), (26, 2, 2)|])
let C_FFS = cls(12, 6, 6, 15, [|(0, 7, 62), (15, 1, 0), (28, 10, 0), (38, 2, 0), (26, 2, 3)|])
let C_FMADD32 = cls(13, 6, 8, 15, [|(0, 6, 58)|])
let C_FMADD16 = cls(14, 6, 8, 15, [|(0, 6, 54)|])
let C_FADD32 = cls(15, 4, 6, 15, [|(0, 6, 42)|])
let C_FADD16 = cls(16, 4, 6, 15, [|(0, 6, 38)|])
let C_FMUL32 = cls(17, 4, 6, 15, [|(0, 6, 26)|])
let C_FMUL16 = cls(18, 4, 6, 15, [|(0, 6, 22)|])
let C_RCP = cls(19, 4, 6, 15, [|(0, 6, 10), (34, 8, 0), (28, 6, 8)|])
let C_STOP = cls(20, 2, 2, 15, [|(0, 16, 136)|])
let C_JMPANY = cls(21, 6, 6, 15, [|(0, 16, 49152)|])
let C_JMPNONE = cls(22, 6, 6, 15, [|(0, 16, 49184)|])
let C_POPEXEC = cls(23, 6, 6, 15, [|(0, 7, 82), (9, 2, 3), (13, 35, 0)|])
let C_IFICMP = cls(24, 6, 6, 15, [|(0, 7, 82), (26, 2, 0), (38, 2, 0), (44, 2, 0), (9, 2, 0)|])
let C_IFFCMP = cls(25, 6, 6, 15, [|(0, 7, 66), (44, 2, 0), (9, 2, 0)|])
let C_WHILEICMP = cls(26, 6, 6, 15, [|(0, 7, 82), (26, 2, 0), (38, 2, 0), (44, 2, 0), (9, 2, 2)|])
let C_WHILEFCMP = cls(27, 6, 6, 15, [|(0, 7, 66), (44, 2, 0), (9, 2, 2)|])
let C_ELSEICMP = cls(28, 6, 6, 15, [|(0, 7, 82), (26, 2, 0), (38, 2, 0), (44, 2, 0), (9, 2, 1)|])
let C_ELSEFCMP = cls(29, 6, 6, 15, [|(0, 7, 66), (44, 2, 0), (9, 2, 1)|])
let C_ICMPSEL = cls(30, 8, 10, 15, [|(0, 7, 18)|])
let C_FCMPSEL = cls(31, 8, 10, 15, [|(0, 7, 2)|])
let C_ICMPBALLOT = cls(32, 8, 8, 15, [|(0, 7, 50), (26, 2, 0), (38, 2, 0), (48, 13, 1)|])
let C_FCMPBALLOT = cls(33, 8, 8, 15, [|(0, 7, 34), (48, 13, 1)|])
let C_SHUF = cls(34, 6, 6, 15, [|(0, 7, 111), (15, 1, 0), (47, 1, 0), (38, 2, 0), (26, 2, 1)|])
let C_SHUFDOWN = cls(35, 6, 6, 15, [|(0, 7, 111), (15, 1, 0), (47, 1, 0), (38, 2, 3), (26, 2, 1)|])
let C_SHUFUP = cls(36, 6, 6, 15, [|(0, 7, 111), (15, 1, 0), (47, 1, 0), (38, 2, 2), (26, 2, 1)|])
let C_SHUFXOR = cls(37, 6, 6, 15, [|(0, 7, 111), (15, 1, 0), (47, 1, 0), (38, 2, 1), (26, 2, 1)|])
let C_WAIT = cls(38, 2, 2, 15, [|(0, 8, 56)|])
let C_DEVLOAD = cls(39, 6, 8, 47, [|(0, 7, 5), (26, 1, 1), (28, 2, 0), (46, 1, 1), (50, 2, 0)|])
let C_DEVSTORE = cls(40, 6, 8, 47, [|(0, 7, 69), (26, 1, 1), (28, 2, 0), (46, 1, 1), (50, 2, 0)|])
let C_TGLOAD = cls(41, 6, 8, 15, [|(0, 4, 9), (5, 2, 3)|])
let C_TGSTORE = cls(42, 6, 8, 15, [|(0, 4, 9), (5, 2, 1)|])
let C_TGBARRIER = cls(43, 2, 2, 15, [|(0, 8, 104)|])
let C_ATOMIC = cls(44, 8, 8, 15, [|(0, 6, 21)|])
let C_TGATOMIC = cls(45, 6, 10, 15, [|(0, 6, 25)|])
let C_MEMBAR = cls(46, 2, 2, 15, [|(0, 8, 245)|])
let C_TRAP = cls(47, 2, 2, 15, [|(0, 16, 8)|])

/* the float unary operations: the rcp class with another op field (28, 6); ceil, trunc and rint set bit 32 or 33,
 * so they have only the long form */
val funop_code : funop -> int
function funop_code(o) = match o {
  FU_floor => 0, FU_rsqrt_special => 1, FU_rsqrt => 9, FU_sin_pt_1 => 10, FU_log2 => 12, FU_exp2 => 13,
  FU_sin_pt_2 => 14, FU_ceil => 16, FU_trunc => 32, FU_rint => 48
}
val funop_cls : funop -> icls
function funop_cls(o) = cls(48 + funop_code(o), 4, 6, 15, [|(0, 6, 10), (34, 8, 0), (28, 6, funop_code(o))|])
let C_FLOOR = funop_cls(FU_floor)
let C_RSQRT_SPECIAL = funop_cls(FU_rsqrt_special)
let C_RSQRT = funop_cls(FU_rsqrt)
let C_SIN_PT_1 = funop_cls(FU_sin_pt_1)
let C_LOG2 = funop_cls(FU_log2)
let C_EXP2 = funop_cls(FU_exp2)
let C_SIN_PT_2 = funop_cls(FU_sin_pt_2)
let C_CEIL = funop_cls(FU_ceil)
let C_TRUNC = funop_cls(FU_trunc)
let C_RINT = funop_cls(FU_rint)
let C_EXTR = cls(200, 8, 8, 15, [|(0, 7, 46), (15, 1, 0), (26, 2, 1)|])
let C_INTL = cls(201, 6, 6, 15, [|(0, 7, 62), (15, 1, 0), (38, 2, 0), (26, 2, 0)|])
let C_CONVERT = cls(202, 6, 6, 15, [|(0, 7, 62), (15, 1, 1), (22, 4, 0), (38, 2, 0), (42, 2, 0)|])
let funops : list(funop) = [|FU_floor, FU_rsqrt_special, FU_rsqrt, FU_sin_pt_1, FU_log2, FU_exp2, FU_sin_pt_2, FU_ceil,
  FU_trunc, FU_rint|]

/* the classes in applegpu's matching order (the first matching descriptor wins) */
let all_classes : list(icls) = [|C_MOVI16, C_MOVI32, C_GETSR, C_IADD, C_IMADD, C_BFI, C_BFEIL, C_ASR,
  C_BITOP, C_BITREV, C_POPCOUNT, C_FFS, C_EXTR, C_INTL, C_CONVERT, C_FMADD32, C_FMADD16, C_FADD32, C_FADD16, C_FMUL32, C_FMUL16,
  C_RCP, C_FLOOR, C_RSQRT_SPECIAL, C_RSQRT, C_SIN_PT_1, C_LOG2, C_EXP2, C_SIN_PT_2, C_CEIL, C_TRUNC, C_RINT, C_STOP, C_JMPANY, C_JMPNONE, C_POPEXEC, C_IFICMP, C_IFFCMP, C_WHILEICMP, C_WHILEFCMP,
  C_ELSEICMP, C_ELSEFCMP, C_ICMPSEL, C_FCMPSEL, C_ICMPBALLOT, C_FCMPBALLOT, C_SHUF, C_SHUFDOWN,
  C_SHUFUP, C_SHUFXOR, C_WAIT, C_DEVLOAD, C_DEVSTORE, C_TGLOAD, C_TGSTORE, C_TGBARRIER, C_ATOMIC,
  C_TGATOMIC, C_MEMBAR, C_TRAP|]
/* memory_barrier (byte 0xf5) and trap (0x0008) match no earlier class: adding them changes no other decoding */

val put_consts : (int, list((int, int, int))) -> int
function put_consts(w, l) = match l {
  [||] => w,
  (s, _, v) :: rest => put_consts(lor_int(w, shl_int(v, s)), rest)
}

val consts_match : (int, list((int, int, int))) -> bool
function consts_match(w, l) = match l {
  [||] => true,
  (s, n, v) :: rest => fget(w, s, n) == v & consts_match(w, rest)
}

/* size in bytes of the instruction of class c at the start of window w */
val cls_size : (icls, int) -> int
function cls_size(c, w) = if c.short == c.long then c.short else if bit_int(w, c.lbit) then c.long else c.short

val cls_matches : (icls, int) -> bool
function cls_matches(c, w) = consts_match(trunc_int(w, 8 * cls_size(c, w)), c.consts)

/* complete an encoding: constants, then the short form if every bit beyond it is zero */
val finish : (icls, int) -> option(int)
function finish(c, w0) = {
  let w = put_consts(w0, c.consts);
  if enc_err then None()
  else if c.short == c.long then Some(w)
  else if shr_int(w, 8 * c.short) == 0 then Some(w)
  else Some(lor_int(w, pow2(c.lbit)))
}

/* the instruction word of class c in window w (masked to its length, length bit cleared) */
val cls_word : (icls, int) -> int
function cls_word(c, w) = {
  let m = trunc_int(w, 8 * cls_size(c, w));
  if c.short != c.long & bit_int(m, c.lbit) then m - pow2(c.lbit) else m
}

/* length of the instruction at the start of the window (0 if no class matches) */
val agx_size_int : int -> int
function agx_size_int(w) = {
  var l = all_classes;
  var r : int = 0;
  while is_cons(l) & r == 0 do {
    match l { c :: rest => { if cls_matches(c, w) then r = cls_size(c, w); l = rest }, [||] => () }
  };
  r
}

/* ---------------------------------------------------------------- float immediates */
/* decode_float_immediate(n) of applegpu as binary64 bits */
val fimm_bits : int -> int
function fimm_bits(n) = {
  let s = bit_int(n, 7);
  let e = fget(n, 4, 3);
  let f = fget(n, 0, 4);
  if e == 0 then encExact(round_fv(FV_Fin(s, f, -6), 53, 11, RN), 64)
  else encExact(round_fv(FV_Fin(s, 16 + f, e - 7), 53, 11, RN), 64)
}

val fimm_code : int -> option(int)
function fimm_code(b) = {
  var r : option(int) = None();
  foreach (n from 0 to 255) { if is_none(r) & fimm_bits(n) == b then r = Some(n) };
  r
}

/* ---------------------------------------------------------------- operands */
val nomods : areg -> bool
function nomods(r) = not_bool(r.sx | r.fabs | r.fneg)

/* ALUDstDesc: value (9, 6) + (ex, 2), flags (7, 2) */
val e_dst : (int, areg, int, bool, bool) -> int
function e_dst(w, r, ex, a32, a64) = {
  if r.file != GPR | not_bool(nomods(r)) | r.hint == HDiscard | r.view == 4 then enc_err = true;
  var value : int = 0;
  var flags : int = 0;
  match r.view {
    0 => { if not_bool(a32) then enc_err = true; value = 2 * r.num; flags = 2 },
    3 => { if not_bool(a64) then enc_err = true; value = 2 * r.num + 1; flags = 2 },
    v => value = 2 * r.num + v - 1
  };
  if r.hint == HCache then flags = flags + 1;
  fput2(fput(w, 7, 2, flags), 9, 6, ex, 2, value)
}

val mkreg : (rfile, int, int, hint) -> areg
function mkreg(f, n, v, h) = struct { file = f, num = n, view = v, hint = h, sx = false, fabs = false, fneg = false }

/* a 16-bit register given by its 16-bit index (r5l = 10, r5h = 11) */
val reg16 : (rfile, int, hint) -> areg
function reg16(f, n16, h) = mkreg(f, shr_int(n16, 1), 1 + fget(n16, 0, 1), h)

val d_dst : (int, int, bool, bool) -> areg
function d_dst(w, ex, a32, a64) = {
  let flags = fget(w, 7, 2);
  let value = fget2(w, 9, 6, ex, 2);
  let h = if bit_int(flags, 0) then HCache else HNone;
  if bit_int(flags, 1) then {
    dchk(a32);
    if bit_int(value, 0) then { dchk(a64); mkreg(GPR, shr_int(value, 1), 3, h) }
    else mkreg(GPR, shr_int(value, 1), 0, h)
  } else reg16(GPR, value, h)
}

val hint_code : hint -> int
function hint_code(h) = match h { HNone => 1, HCache => 2, HDiscard => 3 }
val code_hint : int -> hint
function code_hint(c) = match c { 2 => HCache, 3 => HDiscard, _ => HNone }

/* ALUSrcDesc and its subclasses: value (off, 6) + (ex, 2), flags (off + 6, ts); sign extension bit
 * at sxpos (MulSrcDesc / AddSrcDesc, -1 if none), float modifiers at mpos (FloatSrcDesc, -1 if none);
 * fimm: immediates are 8-bit float codes */
val e_src : (int, aop, int, int, int, bool, bool, bool, int, int) -> int
function e_src(w0, a, off, ex, ts, a32, a64, fimm, sxpos, mpos) = {
  var w : int = w0;
  var value : int = 0;
  var flags : int = 0;
  var sx : int = 0;
  var m : int = 0;
  match a {
    OImm(i) => { if fimm then enc_err = true; value = i },
    OFImm(b) => {
      if not_bool(fimm) then enc_err = true;
      match fimm_code(b) { Some(c) => value = c, None() => enc_err = true }
    },
    /* a float immediate with modifiers (not both clear: that is OFImm) */
    OFMod(b, ab, ng) => {
      if not_bool(fimm) | not_bool(ab | ng) then enc_err = true;
      m = bool_bit(ab) + 2 * bool_bit(ng);
      match fimm_code(b) { Some(c) => value = c, None() => enc_err = true }
    },
    OReg(r) => {
      sx = bool_bit(r.sx);
      m = bool_bit(r.fabs) + 2 * bool_bit(r.fneg);
      if r.file == UNI then {
        if r.hint != HNone then enc_err = true;
        match r.view {
          0 => { if not_bool(a32) then enc_err = true; let n = 2 * r.num in { flags = 6 + fget(n, 8, 1); value = fget(n, 0, 8); if n >= 512 then enc_err = true } },
          4 => { if not_bool(a32) then enc_err = true; let n = 2 * r.num + 1 in { flags = 6 + fget(n, 8, 1); value = fget(n, 0, 8); if n >= 512 then enc_err = true } },
          3 => enc_err = true,
          v => { let n = 2 * r.num + v - 1 in { flags = 4 + fget(n, 8, 1); value = fget(n, 0, 8); if n >= 512 then enc_err = true } }
        }
      } else {
        match r.view {
          0 => { if not_bool(a32) then enc_err = true; flags = 8; value = 2 * r.num },
          3 => { if not_bool(a64) then enc_err = true; flags = 12; value = 2 * r.num },
          4 => enc_err = true,
          v => { flags = 0; value = 2 * r.num + v - 1 }
        };
        flags = flags + hint_code(r.hint)
      }
    }
  };
  if sxpos >= 0 then w = fput(w, sxpos, 1, sx) else if sx != 0 then enc_err = true;
  if mpos >= 0 then w = fput(w, mpos, 2, m) else if m != 0 then enc_err = true;
  fput(fput2(w, off, 6, ex, 2, value), off + 6, ts, flags)
}

val d_src : (int, int, int, int, bool, bool, bool, int, int) -> aop
function d_src(w, off, ex, ts, a32, a64, fimm, sxpos, mpos) = {
  let flags = fget(w, off + 6, ts);
  let value = fget2(w, off, 6, ex, 2);
  let sx = if sxpos >= 0 then bit_int(w, sxpos) else false;
  let m = if mpos >= 0 then fget(w, mpos, 2) else 0;
  if flags == 0 then {
    dchk(not_bool(sx));
    if fimm & m != 0 then OFMod(fimm_bits(value), bit_int(m, 0), bit_int(m, 1))
    else if fimm then OFImm(fimm_bits(value)) else OImm(value)
  } else {
    var r : areg = mkreg(GPR, 0, 0, HNone);
    if land_int(flags, 12) == 4 then {
      let u = value + shl_int(fget(flags, 0, 1), 8);
      if bit_int(flags, 1) then { dchk(a32); r = mkreg(UNI, shr_int(u, 1), if bit_int(u, 0) then 4 else 0, HNone) }
      else r = reg16(UNI, u, HNone)
    } else {
      let top = land_int(flags, 12);
      let h = code_hint(fget(flags, 0, 2));
      dchk(fget(flags, 0, 2) != 0);
      if top == 12 then { dchk(a64 & not_bool(bit_int(value, 0))); r = mkreg(GPR, shr_int(value, 1), 3, h) }
      else if top == 8 then { dchk(a32 & not_bool(bit_int(value, 0))); r = mkreg(GPR, shr_int(value, 1), 0, h) }
      else r = reg16(GPR, value, h)
    };
    OReg({ r with sx = sx, fabs = bit_int(m, 0), fneg = bit_int(m, 1) })
  }
}

/* operand descriptor shorthands: ALUSrc, ALUSrc64, ALUSrc16, MulSrc, AddSrc, FloatSrc, FloatSrc16 */
val e_alu : (int, aop, int, int) -> int
function e_alu(w, a, off, ex) = e_src(w, a, off, ex, 4, true, false, false, -1, -1)
val d_alu : (int, int, int) -> aop
function d_alu(w, off, ex) = d_src(w, off, ex, 4, true, false, false, -1, -1)
val e_alu16 : (int, aop, int, int) -> int
function e_alu16(w, a, off, ex) = e_src(w, a, off, ex, 4, false, false, false, -1, -1)
val d_alu16 : (int, int, int) -> aop
function d_alu16(w, off, ex) = d_src(w, off, ex, 4, false, false, false, -1, -1)
val e_mul : (int, aop, int, int) -> int
function e_mul(w, a, off, ex) = e_src(w, a, off, ex, 4, true, true, false, off + 10, -1)
val d_mul : (int, int, int) -> aop
function d_mul(w, off, ex) = d_src(w, off, ex, 4, true, true, false, off + 10, -1)
val e_add : (int, aop, int, int) -> int
function e_add(w, a, off, ex) = e_src(w, a, off, ex, 4, true, true, false, off + 10, -1)
val d_add : (int, int, int) -> aop
function d_add(w, off, ex) = d_src(w, off, ex, 4, true, true, false, off + 10, -1)
val e_flt : (int, aop, int, int) -> int
function e_flt(w, a, off, ex) = e_src(w, a, off, ex, 4, true, false, true, -1, off + 10)
val d_flt : (int, int, int) -> aop
function d_flt(w, off, ex) = d_src(w, off, ex, 4, true, false, true, -1, off + 10)
val e_flt16 : (int, aop, int, int) -> int
function e_flt16(w, a, off, ex) = e_src(w, a, off, ex, 3, true, false, true, -1, off + 9)
val d_flt16 : (int, int, int) -> aop
function d_flt16(w, off, ex) = d_src(w, off, ex, 3, true, false, true, -1, off + 9)

/* CmpselSrcDesc: value (off, 6) + (ex, 2), flags (off + 6, 3); register size from the destination */
val e_csel : (int, aop, int, int, bool) -> int
function e_csel(w, a, off, ex, is32) = {
  var value : int = 0;
  var flags : int = 0;
  match a {
    OImm(i) => { value = i; flags = 4 },
    OFImm(_) => enc_err = true,
    OFMod(_) => enc_err = true,
    OReg(r) => {
      if not_bool(nomods(r)) then enc_err = true;
      if is32 & r.view != 0 & not_bool(r.view == 4 & r.file == UNI) then enc_err = true;
      if not_bool(is32) & (r.view == 0 | r.view == 3 | r.view == 4) then enc_err = true;
      let n = if is32 then 2 * r.num + (if r.view == 4 then 1 else 0) else 2 * r.num + r.view - 1 in {
        if r.file == UNI then {
          if r.hint != HNone | n >= 512 then enc_err = true;
          flags = 6 + fget(n, 8, 1);
          value = fget(n, 0, 8)
        } else { flags = hint_code(r.hint); value = n }
      }
    }
  };
  fput(fput2(w, off, 6, ex, 2, value), off + 6, 3, flags)
}

val d_csel : (int, int, int, bool) -> aop
function d_csel(w, off, ex, is32) = {
  let flags = fget(w, off + 6, 3);
  let value = fget2(w, off, 6, ex, 2);
  if flags == 4 then OImm(value)
  else if flags == 1 | flags == 2 | flags == 3 then {
    if is32 then { dchk(not_bool(bit_int(value, 0))); OReg(mkreg(GPR, shr_int(value, 1), 0, code_hint(flags))) }
    else OReg(reg16(GPR, value, code_hint(flags)))
  } else if flags == 6 | flags == 7 then {
    let u = value + shl_int(fget(flags, 0, 1), 8);
    if is32 then OReg(mkreg(UNI, shr_int(u, 1), if bit_int(u, 0) then 4 else 0, HNone))
    else OReg(reg16(UNI, u, HNone))
  } else { dec_err = true; OImm(0) }
}

/* conditions: IConditionDesc / FConditionDesc codes (ccn << 3 | cc) */
val icond_code : acc -> option(int)
function icond_code(c) = match c {
  CC_eq => Some(0), CC_ult => Some(1), CC_ugt => Some(2), CC_seq => Some(4), CC_slt => Some(5), CC_sgt => Some(6),
  CC_ne => Some(8), CC_uge => Some(9), CC_ule => Some(10), CC_sne => Some(12), CC_sge => Some(13), CC_sle => Some(14),
  _ => None()
}
val code_icond : int -> option(acc)
function code_icond(n) = match n {
  0 => Some(CC_eq), 1 => Some(CC_ult), 2 => Some(CC_ugt), 4 => Some(CC_seq), 5 => Some(CC_slt), 6 => Some(CC_sgt),
  8 => Some(CC_ne), 9 => Some(CC_uge), 10 => Some(CC_ule), 12 => Some(CC_sne), 13 => Some(CC_sge), 14 => Some(CC_sle),
  _ => None()
}
val fcond_code : acc -> option(int)
function fcond_code(c) = match c {
  CC_feq => Some(0), CC_flt => Some(1), CC_fgt => Some(2), CC_fltn => Some(3), CC_fge => Some(5), CC_fle => Some(6),
  CC_fgtn => Some(7), CC_fnequ => Some(8), CC_fnlt => Some(9), CC_fngt => Some(10), CC_fnltn => Some(11),
  CC_fnge => Some(13), CC_fnle => Some(14), CC_fngtn => Some(15),
  _ => None()
}
val code_fcond : int -> option(acc)
function code_fcond(n) = match n {
  0 => Some(CC_feq), 1 => Some(CC_flt), 2 => Some(CC_fgt), 3 => Some(CC_fltn), 5 => Some(CC_fge), 6 => Some(CC_fle),
  7 => Some(CC_fgtn), 8 => Some(CC_fnequ), 9 => Some(CC_fnlt), 10 => Some(CC_fngt), 11 => Some(CC_fnltn),
  13 => Some(CC_fnge), 14 => Some(CC_fnle), 15 => Some(CC_fngtn),
  _ => None()
}

/* cc (3 bits at ccoff) and the inversion bit at ccnoff (-1: no inversion bit) */
val e_cc : (int, bool, acc, int, int) -> int
function e_cc(w, isf, c, ccoff, ccnoff) = {
  let ccode : int = match (if isf then fcond_code(c) else icond_code(c)) { Some(n) => n, None() => { enc_err = true; 0 } };
  if ccnoff >= 0 then fput(fput(w, ccoff, 3, fget(ccode, 0, 3)), ccnoff, 1, fget(ccode, 3, 1))
  else { if ccode >= 8 then enc_err = true; fput(w, ccoff, 3, fget(ccode, 0, 3)) }
}

val d_cc : (int, bool, int, int) -> acc
function d_cc(w, isf, ccoff, ccnoff) = {
  let ccode = fget(w, ccoff, 3) + (if ccnoff >= 0 then 8 * fget(w, ccnoff, 1) else 0);
  match (if isf then code_fcond(ccode) else code_icond(ccode)) { Some(c) => c, None() => { dec_err = true; CC_eq } }
}

/* MemoryRegDesc / ThreadgroupMemoryRegDesc: first register of the tuple; value (off, 6) + (offx, 2), flag at offt */
val e_memreg : (int, areg, int, int, int) -> int
function e_memreg(w, r, off, offx, offt) = {
  if r.file != GPR | not_bool(nomods(r)) | r.hint != HNone | r.view >= 3 then enc_err = true;
  let t = if r.view == 0 then 1 else 0 in
  let value = if r.view == 0 then 2 * r.num else 2 * r.num + r.view - 1 in
  fput(fput2(w, off, 6, offx, 2, value), offt, 1, t)
}

val d_memreg : (int, int, int, int) -> areg
function d_memreg(w, off, offx, offt) = {
  let value = fget2(w, off, 6, offx, 2);
  if bit_int(w, offt) then { dchk(not_bool(bit_int(value, 0))); mkreg(GPR, shr_int(value, 1), 0, HNone) }
  else reg16(GPR, value, HNone)
}

/* MemoryBaseDesc: 64-bit register pair, value (16, 4) + (36, 4), uniform flag at 27 */
val e_membase : (int, apair) -> int
function e_membase(w, p) = {
  if p.hi != p.lo + 1 then enc_err = true;
  fput(fput2(w, 16, 4, 36, 4, 2 * p.lo), 27, 1, if p.pfile == UNI then 1 else 0)
}
val d_membase : int -> apair
function d_membase(w) = {
  let value = fget2(w, 16, 4, 36, 4);
  dchk(not_bool(bit_int(value, 0)));
  let n = shr_int(value, 1);
  struct { pfile = if bit_int(w, 27) then UNI else GPR, lo = n, hi = n + 1 }
}

/* MemoryIndexDesc: value (20, 4) + (32, 4) + (56, 8), immediate flag at 24 (signed 16-bit immediate
 * or a 32-bit register) */
val e_memidx : (int, aop) -> int
function e_memidx(w, a) = match a {
  OImm(i) => { if i < -32768 | i >= 32768 then enc_err = true; fput(fput3(w, 20, 4, 32, 4, 56, 8, trunc_int(i, 16)), 24, 1, 1) },
  OReg(r) => {
    if r.file != GPR | r.view != 0 | not_bool(nomods(r)) | r.hint != HNone | r.num >= 128 then enc_err = true;
    fput3(w, 20, 4, 32, 4, 56, 8, 2 * r.num)
  },
  _ => { enc_err = true; w }
}
val d_memidx : int -> aop
function d_memidx(w) = {
  let value = fget3(w, 20, 4, 32, 4, 56, 8);
  if bit_int(w, 24) then OImm(sext_int(value, 16))
  else { dchk(not_bool(bit_int(value, 0)) & value < 256); OReg(mkreg(GPR, shr_int(value, 1), 0, HNone)) }
}

/* ThreadgroupMemoryBaseDesc: value (16, 6) + (58, 2), flags (22, 2): 00 r16, 10 immediate 0, x1 u16 */
val e_tgbase : (int, aop) -> int
function e_tgbase(w, a) = match a {
  OImm(i) => { if i != 0 then enc_err = true; fput(w, 22, 2, 2) },
  OReg(r) => {
    if not_bool(nomods(r)) | r.hint != HNone | r.view == 0 | r.view >= 3 then enc_err = true;
    let n = 2 * r.num + r.view - 1 in {
      if r.file == UNI then { if n >= 512 then enc_err = true; fput(fput2(w, 16, 6, 58, 2, fget(n, 0, 8)), 22, 2, 1 + 2 * fget(n, 8, 1)) }
      else fput2(w, 16, 6, 58, 2, n)
    }
  },
  _ => { enc_err = true; w }
}
val d_tgbase : int -> aop
function d_tgbase(w) = {
  let flags = fget(w, 22, 2);
  let value = fget2(w, 16, 6, 58, 2);
  if flags == 0 then OReg(reg16(GPR, value, HNone))
  else if flags == 2 then { dchk(value == 0); OImm(0) }
  else OReg(reg16(UNI, value + shl_int(shr_int(flags, 1), 8), HNone))
}

/* ThreadgroupIndexDesc: value (28, 6) + (48, 10), immediate flag at 34 (signed 16-bit immediate or r16) */
val e_tgidx : (int, aop) -> int
function e_tgidx(w, a) = match a {
  OImm(i) => { if i < -32768 | i >= 32768 then enc_err = true; fput(fput2(w, 28, 6, 48, 10, trunc_int(i, 16)), 34, 1, 1) },
  OReg(r) => {
    if r.file != GPR | not_bool(nomods(r)) | r.hint != HNone | r.view == 0 | r.view >= 3 then enc_err = true;
    fput2(w, 28, 6, 48, 10, 2 * r.num + r.view - 1)
  },
  _ => { enc_err = true; w }
}
val d_tgidx : int -> aop
function d_tgidx(w) = {
  let value = fget2(w, 28, 6, 48, 10);
  if bit_int(w, 34) then OImm(sext_int(value, 16))
  else { dchk(value < 256); OReg(reg16(GPR, value, HNone)) }
}

/* element mask and formats */
val fmt_code : afmt -> int
function fmt_code(f) = match f {
  F_U8 => 0, F_U16 => 1, F_U32 => 2, F_F16 => 3, F_U8NORM => 4, F_S8NORM => 5, F_U16NORM => 6, F_S16NORM => 7,
  F_RGB10A2 => 8, F_SRGBA8 => 10, F_RG11B10F => 12, F_RGB9E5 => 13, F_RG11B10F_RTZ => 14, F_RGB9E5_RTZ => 15
}
/* the format codes (applegpu MEMORY_FORMATS; 9 and 11 have no name and are rejected) */
val code_fmt : int -> afmt
function code_fmt(n) = match n {
  0 => F_U8, 1 => F_U16, 2 => F_U32, 3 => F_F16, 4 => F_U8NORM, 5 => F_S8NORM, 6 => F_U16NORM, 7 => F_S16NORM,
  8 => F_RGB10A2, 10 => F_SRGBA8, 12 => F_RG11B10F, 13 => F_RGB9E5, 14 => F_RG11B10F_RTZ, 15 => F_RGB9E5_RTZ,
  _ => { dec_err = true; F_U32 }
}
/* threadgroup and stack accesses have the integer formats only */
val int_fmt : afmt -> bool
function int_fmt(f) = f == F_U8 | f == F_U16 | f == F_U32

/* the op field of the ten 32-bit operations (both atomics); the device-only forms are rejected here */
val atom_code : aaop -> int
function atom_code(o) = match o {
  A_add => 0, A_sub => 1, A_xchg => 2, A_umin => 4, A_imin => 5, A_umax => 6, A_imax => 7, A_and => 8, A_or => 9, A_xor => 10,
  _ => { enc_err = true; 0 }
}
val code_atom : int -> aaop
function code_atom(n) = match n {
  0 => A_add, 1 => A_sub, 2 => A_xchg, 4 => A_umin, 5 => A_imin, 6 => A_umax, 7 => A_imax, 8 => A_and, 9 => A_or, 10 => A_xor,
  _ => { dec_err = true; A_add }
}

val op_reg32 : aop -> option(areg)
function op_reg32(a) = match a { OReg(r) => if r.file == GPR & r.view == 0 & nomods(r) & r.hint == HNone then Some(r) else None(), _ => None() }

val is_imm0 : aop -> bool
function is_imm0(a) = match a { OImm(i) => i == 0, _ => false }

/* ---------------------------------------------------------------- per-class encoders / decoders */
val start_enc : unit -> unit
function start_enc() = enc_err = false
val start_dec : unit -> unit
function start_dec() = dec_err = false
val fin_dec : ast -> option(ast)
function fin_dec(a) = if dec_err then None() else Some(a)

val e_mov : (areg, aop) -> option(int)
function e_mov(d, a) = {
  start_enc();
  match a {
    OImm(i) => {
      if d.view == 1 | d.view == 2 then finish(C_MOVI16, fput(e_dst(0, d, 44, false, false), 16, 16, i))
      else finish(C_MOVI32, fput(e_dst(0, d, 60, true, false), 16, 32, i))
    },
    /* mov from a register: applegpu's bitop alias `mov` (truth table 0101 = a, B = 0) */
    OReg(_) => finish(C_BITOP, fput(fput(fput(fput(e_alu(e_alu(e_dst(0, d, 44, true, false), a, 16, 42), OImm(0), 28, 40), 26, 1, 0), 27, 1, 1), 38, 1, 0), 39, 1, 1)),
    _ => None()
  }
}

val d_movi16 : int -> option(ast)
function d_movi16(w) = { start_dec(); fin_dec(MOV(d_dst(w, 44, false, false), OImm(fget(w, 16, 16)))) }
val d_movi32 : int -> option(ast)
function d_movi32(w) = { start_dec(); dchk(fget(w, 62, 1) == 0); fin_dec(MOV(d_dst(w, 60, true, false), OImm(fget(w, 16, 32)))) }

val e_getsr : (areg, int) -> option(int)
function e_getsr(d, n) = { start_enc(); finish(C_GETSR, fput2(e_dst(0, d, 28, true, false), 16, 6, 26, 2, n)) }
val d_getsr : int -> option(ast)
function d_getsr(w) = { start_dec(); fin_dec(GET_SR(d_dst(w, 28, true, false), fget2(w, 16, 6, 26, 2))) }

val e_iadd : (bool, bool, areg, aop, aop, int) -> option(int)
function e_iadd(sub, sat, d, a, b, sh) = {
  start_enc();
  let w = e_add(e_add(e_dst(0, d, 44, true, true), a, 16, 42), b, 28, 40);
  finish(C_IADD, fput2(fput(fput(w, 6, 1, bool_bit(sat)), 27, 1, bool_bit(sub)), 39, 1, 52, 2, sh))
}
val d_iadd : int -> option(ast)
function d_iadd(w) = {
  start_dec();
  fin_dec(IADD(bit_int(w, 27), bit_int(w, 6), d_dst(w, 44, true, true), d_add(w, 16, 42), d_add(w, 28, 40), fget2(w, 39, 1, 52, 2)))
}

val e_imadd : (bool, bool, areg, aop, aop, aop, int) -> option(int)
function e_imadd(sub, sat, d, a, b, c, sh) = {
  start_enc();
  let w = e_add(e_mul(e_mul(e_dst(0, d, 60, true, true), a, 16, 58), b, 28, 56), c, 40, 54);
  finish(C_IMADD, fput2(fput(fput(w, 6, 1, bool_bit(sat)), 27, 1, bool_bit(sub)), 39, 1, 52, 2, sh))
}
val d_imadd : int -> option(ast)
function d_imadd(w) = {
  start_dec();
  fin_dec(IMADD(bit_int(w, 27), bit_int(w, 6), d_dst(w, 60, true, true), d_mul(w, 16, 58), d_mul(w, 28, 56), d_add(w, 40, 54), fget2(w, 39, 1, 52, 2)))
}

/* bfi / bfeil: D (9,6)+(60,2), A (16,58), B (28,56), C (40,54), mask m (38,2)+(50,2)+(63,1) */
val e_bf : (icls, int, areg, aop, aop, aop) -> option(int)
function e_bf(c, m, d, a, b, x) = {
  start_enc();
  let w = e_alu(e_alu(e_alu(e_dst(0, d, 60, true, false), a, 16, 58), b, 28, 56), x, 40, 54);
  finish(c, fput3(w, 38, 2, 50, 2, 63, 1, m))
}
val d_bfi : int -> option(ast)
function d_bfi(w) = { start_dec(); fin_dec(BFI(fget3(w, 38, 2, 50, 2, 63, 1), d_dst(w, 60, true, false), d_alu(w, 16, 58), d_alu(w, 28, 56), d_alu(w, 40, 54))) }
val d_bfeil : int -> option(ast)
function d_bfeil(w) = { start_dec(); fin_dec(BFEIL(fget3(w, 38, 2, 50, 2, 63, 1), d_dst(w, 60, true, false), d_alu(w, 16, 58), d_alu(w, 28, 56), d_alu(w, 40, 54))) }

val e_asr : (areg, aop, aop) -> option(int)
function e_asr(d, a, b) = { start_enc(); finish(C_ASR, e_alu(e_alu(e_dst(0, d, 60, true, false), a, 16, 58), b, 28, 56)) }
val d_asr : int -> option(ast)
function d_asr(w) = { start_dec(); fin_dec(ASR(d_dst(w, 60, true, false), d_alu(w, 16, 58), d_alu(w, 28, 56))) }

/* bitop: truth table bits tt0 (26), tt1 (27), tt2 (38), tt3 (39) = bits 0..3 of the K model's TT */
val e_bitop : (int, areg, aop, aop) -> option(int)
function e_bitop(tt, d, a, b) = {
  start_enc();
  if tt < 0 | tt > 15 then enc_err = true;
  /* the mov alias (TT 0101 with B = 0) decodes as MOV */
  if tt == 10 & is_imm0(b) then enc_err = true;
  let w = e_alu(e_alu(e_dst(0, d, 44, true, false), a, 16, 42), b, 28, 40);
  finish(C_BITOP, fput(fput(fput(fput(w, 26, 1, fget(tt, 0, 1)), 27, 1, fget(tt, 1, 1)), 38, 1, fget(tt, 2, 1)), 39, 1, fget(tt, 3, 1)))
}
val d_bitop : int -> option(ast)
function d_bitop(w) = {
  start_dec();
  let tt = fget(w, 26, 1) + 2 * fget(w, 27, 1) + 4 * fget(w, 38, 1) + 8 * fget(w, 39, 1);
  let d = d_dst(w, 44, true, false);
  let a = d_alu(w, 16, 42);
  let b = d_alu(w, 28, 40);
  if tt == 10 & is_imm0(b) then fin_dec(MOV(d, a)) else fin_dec(BITOP(tt, d, a, b))
}

val d_extr : int -> option(ast)
function d_extr(w) = { start_dec(); fin_dec(EXTR(fget3(w, 38, 2, 50, 2, 63, 1), d_dst(w, 60, true, false), d_alu(w, 16, 58), d_alu(w, 28, 56), d_alu(w, 40, 54))) }

/* intl: D (44), A (16, 42), B (28, 40) */
val e_intl : (areg, aop, aop) -> option(int)
function e_intl(d, a, b) = { start_enc(); finish(C_INTL, e_alu(e_alu(e_dst(0, d, 44, true, false), a, 16, 42), b, 28, 40)) }
val d_intl : int -> option(ast)
function d_intl(w) = { start_dec(); fin_dec(INTL(d_dst(w, 44, true, false), d_alu(w, 16, 42), d_alu(w, 28, 40))) }

/* convert: mode (16, 6) (Mesa's first source, an immediate: bits 22-25 and 42-43 clear), direction (26, 2): 0 float
 * -> integer, 1 integer -> float (applegpu's `round` field), D (44), the source (28, 40).  Modes 2, 3 and 12-63 are
 * rejected: Mesa does not emit them and they were not run on the hardware (Apple-M2-Playground 2026-10-16-conversions) */
val convert_mode_ok : int -> bool
function convert_mode_ok(m) = m == 0 | m == 1 | (m >= 4 & m <= 11)
val e_convert : (int, bool, areg, aop) -> option(int)
function e_convert(m, i2f, d, a) = {
  start_enc();
  if not_bool(convert_mode_ok(m)) then enc_err = true;
  finish(C_CONVERT, fput(fput(e_alu(e_dst(0, d, 44, true, false), a, 28, 40), 16, 6, m), 26, 2, bool_bit(i2f)))
}
val d_convert : int -> option(ast)
function d_convert(w) = {
  start_dec();
  let m = fget(w, 16, 6);
  dchk(convert_mode_ok(m) & fget(w, 26, 2) <= 1);
  fin_dec(CONVERT(m, fget(w, 26, 2) == 1, d_dst(w, 44, true, false), d_alu(w, 28, 40)))
}

val iunop_cls : iunop -> icls
function iunop_cls(o) = match o { U_popcount => C_POPCOUNT, U_bitrev => C_BITREV, U_ffs => C_FFS }
val e_iunop : (iunop, areg, aop) -> option(int)
function e_iunop(o, d, a) = { start_enc(); finish(iunop_cls(o), e_alu(e_dst(0, d, 44, true, false), a, 16, 42)) }
val d_iunop : (iunop, int) -> option(ast)
function d_iunop(o, w) = { start_dec(); fin_dec(IUNOP(o, d_dst(w, 44, true, false), d_alu(w, 16, 42))) }

/* FloatDstDesc: ALUDst + saturate bit 6 */
val e_fbin : (fbinop, bool, bool, areg, aop, aop) -> option(int)
function e_fbin(o, h, sat, d, a, b) = {
  start_enc();
  let c : icls = match (o, h) { (FB_add, false) => C_FADD32, (FB_add, true) => C_FADD16, (FB_mul, false) => C_FMUL32, (FB_mul, true) => C_FMUL16 };
  let w0 = fput(e_dst(0, d, 44, not_bool(h), false), 6, 1, bool_bit(sat));
  let w = if h then e_flt16(e_flt16(w0, a, 16, 42), b, 28, 40) else e_flt(e_flt(w0, a, 16, 42), b, 28, 40);
  finish(c, w)
}
val d_fbin : (fbinop, bool, int) -> option(ast)
function d_fbin(o, h, w) = {
  start_dec();
  let d = d_dst(w, 44, not_bool(h), false);
  if h then fin_dec(FBIN(o, h, bit_int(w, 6), d, d_flt16(w, 16, 42), d_flt16(w, 28, 40)))
  else fin_dec(FBIN(o, h, bit_int(w, 6), d, d_flt(w, 16, 42), d_flt(w, 28, 40)))
}

val e_fmadd : (bool, bool, areg, aop, aop, aop) -> option(int)
function e_fmadd(h, sat, d, a, b, c) = {
  start_enc();
  let w0 = fput(e_dst(0, d, 60, not_bool(h), false), 6, 1, bool_bit(sat));
  let w = if h then e_flt16(e_flt16(e_flt16(w0, a, 16, 58), b, 28, 56), c, 40, 54)
          else e_flt(e_flt(e_flt(w0, a, 16, 58), b, 28, 56), c, 40, 54);
  finish(if h then C_FMADD16 else C_FMADD32, w)
}
/* bits 52 and 53 of the 8-byte form are rejected: on G14C bit 53 selects a 10-byte form with a lane offset for A,
 * and bit 52 changes the fmadd32 operation (Apple-M2-Playground 2026-10-11-fmadd-lane-offset, 2026-10-18-fma-b52);
 * neither is modelled (sail/README.md) */
val d_fmadd : (bool, int) -> option(ast)
function d_fmadd(h, w) = {
  start_dec();
  dchk(fget(w, 52, 2) == 0);
  let d = d_dst(w, 60, not_bool(h), false);
  if h then fin_dec(FMADD(h, bit_int(w, 6), d, d_flt16(w, 16, 58), d_flt16(w, 28, 56), d_flt16(w, 40, 54)))
  else fin_dec(FMADD(h, bit_int(w, 6), d, d_flt(w, 16, 58), d_flt(w, 28, 56), d_flt(w, 40, 54)))
}

val e_rcp : (bool, areg, aop) -> option(int)
function e_rcp(sat, d, a) = { start_enc(); finish(C_RCP, e_flt(fput(e_dst(0, d, 44, true, false), 6, 1, bool_bit(sat)), a, 16, 42)) }
val d_rcp : int -> option(ast)
function d_rcp(w) = { start_dec(); fin_dec(RCP(bit_int(w, 6), d_dst(w, 44, true, false), d_flt(w, 16, 42))) }

val e_funop : (funop, bool, areg, aop) -> option(int)
function e_funop(o, sat, d, a) = { start_enc(); finish(funop_cls(o), e_flt(fput(e_dst(0, d, 44, true, false), 6, 1, bool_bit(sat)), a, 16, 42)) }
val d_funop : (funop, int) -> option(ast)
function d_funop(o, w) = { start_dec(); fin_dec(FUNOP(o, bit_int(w, 6), d_dst(w, 44, true, false), d_flt(w, 16, 42))) }

/* the funop of class number n, if any */
val cid_funop : int -> option(funop)
function cid_funop(n) = {
  var r : option(funop) = None();
  var l = funops;
  while is_cons(l) do { match l { o :: rest => { if funop_cls(o).cid == n then r = Some(o); l = rest }, [||] => () } };
  r
}

/* icmpsel / fcmpsel: cc (61, 3), D (9,6)+(76,2), A (16,74), B (28,72), X (40,70), Y (52,68) */
val e_cmpsel : (bool, acc, areg, aop, aop, aop, aop) -> option(int)
function e_cmpsel(isf, cc, d, a, b, x, y) = {
  start_enc();
  let is32 = d.view == 0;
  let w0 = e_cc(e_dst(0, d, 76, true, false), isf, cc, 61, -1);
  let w1 = if isf then e_flt(e_flt(w0, a, 16, 74), b, 28, 72) else e_alu(e_alu(w0, a, 16, 74), b, 28, 72);
  finish(if isf then C_FCMPSEL else C_ICMPSEL, e_csel(e_csel(w1, x, 40, 70, is32), y, 52, 68, is32))
}
val d_cmpsel : (bool, int) -> option(ast)
function d_cmpsel(isf, w) = {
  start_dec();
  let d = d_dst(w, 76, true, false);
  let is32 = d.view == 0;
  let a = if isf then d_flt(w, 16, 74) else d_alu(w, 16, 74);
  let b = if isf then d_flt(w, 28, 72) else d_alu(w, 28, 72);
  fin_dec(CMPSEL(isf, d_cc(w, isf, 61, -1), d, a, b, d_csel(w, 40, 70, is32), d_csel(w, 52, 68, is32)))
}

/* execution-mask stack: op (9, 2), n (11, 2), cc (13, 3) + ccn (8), A (16, 42), B (28, 40), r0l cache hint (7) = 0 */
val e_stack : (stkkind, bool, acc, aop, aop, int) -> option(int)
function e_stack(k, isf, cc, a, b, n) = {
  start_enc();
  let c : icls = match (k, isf) {
    (SK_if, false) => C_IFICMP, (SK_if, true) => C_IFFCMP, (SK_while, false) => C_WHILEICMP,
    (SK_while, true) => C_WHILEFCMP, (SK_else, false) => C_ELSEICMP, (SK_else, true) => C_ELSEFCMP
  };
  let w0 = fput(e_cc(0, isf, cc, 13, 8), 11, 2, n);
  finish(c, if isf then e_flt(e_flt(w0, a, 16, 42), b, 28, 40) else e_alu(e_alu(w0, a, 16, 42), b, 28, 40))
}
val d_stack : (stkkind, bool, int) -> option(ast)
function d_stack(k, isf, w) = {
  start_dec();
  dchk(fget(w, 7, 1) == 0);
  let a = if isf then d_flt(w, 16, 42) else d_alu(w, 16, 42);
  let b = if isf then d_flt(w, 28, 40) else d_alu(w, 28, 40);
  fin_dec(STACKOP(k, isf, d_cc(w, isf, 13, 8), a, b, fget(w, 11, 2)))
}

val e_pop : int -> option(int)
function e_pop(n) = { start_enc(); finish(C_POPEXEC, fput(0, 11, 2, n)) }
val d_pop : int -> option(ast)
function d_pop(w) = { start_dec(); dchk(fget(w, 7, 2) == 0); fin_dec(POP_EXEC(fget(w, 11, 2))) }

val e_jmp : (bool, jtarget) -> option(int)
function e_jmp(any, t) = {
  start_enc();
  match t {
    JRel(off) => { if off < -2147483648 | off >= 2147483648 then enc_err = true; finish(if any then C_JMPANY else C_JMPNONE, fput(0, 16, 32, trunc_int(off, 32))) },
    JLabel(_) => None()
  }
}
val d_jmp : (bool, int) -> option(ast)
function d_jmp(any, w) = { start_dec(); fin_dec(JMP_EXEC(any, JRel(sext_int(fget(w, 16, 32), 32)))) }

/* ballots: D (9,6)+(44,2), cc (61, 3) + ccn (47), A (16, 42), B (28, 40) */
val e_ballot : (bool, acc, areg, aop, aop) -> option(int)
function e_ballot(isf, cc, d, a, b) = {
  start_enc();
  let w0 = e_cc(e_dst(0, d, 44, true, false), isf, cc, 61, 47);
  finish(if isf then C_FCMPBALLOT else C_ICMPBALLOT, if isf then e_flt(e_flt(w0, a, 16, 42), b, 28, 40) else e_alu(e_alu(w0, a, 16, 42), b, 28, 40))
}
val d_ballot : (bool, int) -> option(ast)
function d_ballot(isf, w) = {
  start_dec();
  let a = if isf then d_flt(w, 16, 42) else d_alu(w, 16, 42);
  let b = if isf then d_flt(w, 28, 40) else d_alu(w, 28, 40);
  fin_dec(BALLOT(isf, d_cc(w, isf, 61, 47), d_dst(w, 44, true, false), a, b))
}

/* shuffles: D (9,6)+(44,2), A (16, 42), B (28, 40): a 16-bit source, or a 32-bit one (Mesa's packer accepts it,
 * applegpu and agx_run read the 16-bit register of the same index, the LLVM backend rejects it; not measured) */
val e_shuf : (shufkind, areg, aop, aop) -> option(int)
function e_shuf(k, d, a, b) = {
  start_enc();
  let c : icls = match k { SH_idx => C_SHUF, SH_down => C_SHUFDOWN, SH_up => C_SHUFUP, SH_xor => C_SHUFXOR };
  finish(c, e_alu(e_alu(e_dst(0, d, 44, true, false), a, 16, 42), b, 28, 40))
}
val d_shuf : (shufkind, int) -> option(ast)
function d_shuf(k, w) = { start_dec(); fin_dec(SHUFFLE(k, d_dst(w, 44, true, false), d_alu(w, 16, 42), d_alu(w, 28, 40))) }

/* device_load / device_store: g (30), F (7,3)+(48,1), mask (52,4), R (10,6)+(40,2) flag 49, base, index,
 * Ou (25: 0 signed, 1 unsigned), shift (42,2), u6 (44,2) with bit 46 set (the LLVM backend's trailing operand; Mesa's
 * u4 = bits 44-46 is 4, or 7 for a coherent access; applegpu's plain descriptors cover u6 = 0 for loads and bit 45
 * clear for stores, its `.TODO` descriptors the rest) */
val e_devmem : (bool, afmt, int, areg, apair, aop, bool, int, int, int) -> option(int)
function e_devmem(st, f, mask, r, p, i, sgn, s, g, u6) = {
  start_enc();
  /* shift field value 3: applegpu's notes (commented-out code in MemoryShiftDesc.decode, and its
   * emulator's `if shift == 3: shift = 2`) suggest it acts as 2, which is not confirmed; agx.k adds the
   * shift as given.  The value is rejected in both directions, so every encoded / decoded access has
   * the same meaning in agx.k, this model and the hardware descriptions. */
  if s < 0 | s > 2 then enc_err = true;
  let w0 = fput(fput(fput2(fput(0, 30, 1, g), 7, 3, 48, 1, fmt_code(f)), 52, 4, mask), 25, 1, if sgn then 0 else 1);
  let w1 = fput(e_memidx(e_membase(e_memreg(w0, r, 10, 40, 49), p), i), 42, 2, s);
  finish(if st then C_DEVSTORE else C_DEVLOAD, fput(w1, 44, 2, u6))
}
val d_devmem : (bool, int) -> option(ast)
function d_devmem(st, w) = {
  start_dec();
  let f = code_fmt(fget2(w, 7, 3, 48, 1));
  dchk(fget(w, 42, 2) != 3);   /* see e_devmem */
  fin_dec(DEVMEM(st, f, fget(w, 52, 4), d_memreg(w, 10, 40, 49), d_membase(w), d_memidx(w), not_bool(bit_int(w, 25)),
                 fget(w, 42, 2), fget(w, 30, 1), fget(w, 44, 2)))
}

/* threadgroup_load / store: F (24,4), mask (36,4), R (9,6)+(60,2) flag 8, base, index; no shift field */
val e_tgmem : (bool, afmt, int, areg, aop, aop, int) -> option(int)
function e_tgmem(st, f, mask, r, base, i, s) = {
  start_enc();
  if s != 0 | not_bool(int_fmt(f)) then enc_err = true;
  let w = e_tgidx(e_tgbase(e_memreg(fput(fput(0, 24, 4, fmt_code(f)), 36, 4, mask), r, 9, 60, 8), base), i);
  finish(if st then C_TGSTORE else C_TGLOAD, w)
}
val d_tgmem : (bool, int) -> option(ast)
function d_tgmem(st, w) = {
  start_dec();
  let f = code_fmt(fget(w, 24, 4));
  dchk(int_fmt(f));
  fin_dec(TGMEM(st, f, fget(w, 36, 4), d_memreg(w, 9, 60, 8), d_tgbase(w), d_tgidx(w), 0))
}

/* device atomic: op (6,4), g (30), R (10,6)+(40,2) flag 47, base, index, Ou (25), S (48, 8), bits 45-46;
 * the other unidentified fields hold the values applegpu observed: u1 (26) = 1, u2 (28,2) = 0, u3 (31) = 1,
 * u4 (42,3) = 4.
 * Bits 45-46 (applegpu u5; G14C, Apple-M2-Playground 2026-10-17-atomic-variants) as the AST's mode, the LLVM
 * backend's operand: mode = b45 | (1 - b46) << 1, so Mesa's G14 encoding (b45 = 0, b46 = 1) is mode 0.
 *   op 1: b45 = 0 sub, b45 = 1 fadd (b46 free; the backend emits b46 = 1, mode 1)
 *   op 3: b45 = 0 cmpxchg (b46 free); b45 = 1: b46 = 0 umin64 (mode 3), b46 = 1 umax64 (mode 1)
 *   ops 0, 2, 4-10: the same operation for every mode (b45 is Mesa's multi-die coherency bit)
 *   ops 11-15: end11 .. end15, measured only at Mesa's values (mode 0); other modes are rejected
 * cmpxchg / umin64 / umax64 read the register pair S, S + 1 (applegpu prints S as a pair for op 3). */
val devatom_code : (aaop, int) -> int
function devatom_code(op, m) = {
  let ok : bool = match op {
    A_sub => land_int(m, 1) == 0, A_cmpxchg => land_int(m, 1) == 0, A_fadd => land_int(m, 1) == 1,
    A_umin64 => m == 3, A_umax64 => m == 1,
    A_end11 => m == 0, A_end12 => m == 0, A_end13 => m == 0, A_end14 => m == 0, A_end15 => m == 0,
    _ => true
  };
  if not_bool(ok) | m < 0 | m > 3 then enc_err = true;
  match op {
    A_cmpxchg => 3, A_fadd => 1, A_umin64 => 3, A_umax64 => 3,
    A_end11 => 11, A_end12 => 12, A_end13 => 13, A_end14 => 14, A_end15 => 15,
    _ => atom_code(op)
  }
}

val code_devatom : (int, int) -> aaop
function code_devatom(n, m) = {
  let b45 = land_int(m, 1) == 1;
  match n {
    1 => if b45 then A_fadd else A_sub,
    3 => if not_bool(b45) then A_cmpxchg else if m == 3 then A_umin64 else A_umax64,
    11 => { dchk(m == 0); A_end11 }, 12 => { dchk(m == 0); A_end12 }, 13 => { dchk(m == 0); A_end13 },
    14 => { dchk(m == 0); A_end14 }, 15 => { dchk(m == 0); A_end15 },
    _ => code_atom(n)
  }
}

/* the forms whose value is a register pair */
val devatom_pair : aaop -> bool
function devatom_pair(op) = op == A_cmpxchg | op == A_umin64 | op == A_umax64

val e_devatom : (aaop, option(areg), apair, aop, bool, aop, int, int) -> option(int)
function e_devatom(op, d, p, i, sgn, v, g, m) = {
  start_enc();
  let c = devatom_code(op, m);
  var w : int = fput(fput(fput(0, 6, 4, c), 30, 1, g), 25, 1, if sgn then 0 else 1);
  match d {
    Some(r) => { if r.file != GPR | r.view != 0 | not_bool(nomods(r)) | r.hint != HNone then enc_err = true; w = fput(fput2(w, 10, 6, 40, 2, 2 * r.num), 47, 1, 1) },
    None() => ()
  };
  if devatom_pair(op) then {
    match v {
      OReg(r) => { if r.file != GPR | r.view != 3 | not_bool(nomods(r)) | r.hint != HNone then enc_err = true; w = fput(w, 48, 8, 2 * r.num) },
      _ => enc_err = true
    }
  } else {
    match op_reg32(v) { Some(r) => w = fput(w, 48, 8, 2 * r.num), None() => enc_err = true }
  };
  w = e_memidx(e_membase(w, p), i);
  w = fput(fput(fput(fput(w, 26, 1, 1), 28, 2, 0), 31, 1, 1), 42, 3, 4);
  w = fput(fput(w, 45, 1, land_int(m, 1)), 46, 1, 1 - fget(m, 1, 1));
  finish(C_ATOMIC, w)
}
val d_devatom : int -> option(ast)
function d_devatom(w) = {
  start_dec();
  dchk(fget(w, 26, 1) == 1 & fget(w, 28, 2) == 0 & fget(w, 31, 1) == 1 & fget(w, 42, 3) == 4);
  let m = fget(w, 45, 1) + 2 * (1 - fget(w, 46, 1));
  let rv = fget2(w, 10, 6, 40, 2);
  let d : option(areg) = if bit_int(w, 47) then { dchk(not_bool(bit_int(rv, 0))); Some(mkreg(GPR, shr_int(rv, 1), 0, HNone)) } else { dchk(rv == 0); None() };
  let sv = fget(w, 48, 8);
  dchk(not_bool(bit_int(sv, 0)));
  let op = code_devatom(fget(w, 6, 4), m);
  let v = mkreg(GPR, shr_int(sv, 1), if devatom_pair(op) then 3 else 0, HNone);
  fin_dec(DEVATOM(op, d, d_membase(w), d_memidx(w), not_bool(bit_int(w, 25)), OReg(v), fget(w, 30, 1), m))
}

/* threadgroup atomic: op (24,4), R (9,6)+(60,2) flag 8, present 38, base, index, S (64, 8);
 * observed values u1 (6,2) = 0, u3 (35,3) = 2, u4 (40,8) = 128, u5 (72,8) = 0 */
val e_tgatom : (aaop, option(areg), aop, aop, aop) -> option(int)
function e_tgatom(op, d, base, i, v) = {
  start_enc();
  var w : int = fput(0, 24, 4, atom_code(op));
  match d { Some(r) => w = fput(e_memreg(w, r, 9, 60, 8), 38, 1, 1), None() => () };
  match op_reg32(v) { Some(r) => w = fput(w, 64, 8, 2 * r.num), None() => enc_err = true };
  w = e_tgidx(e_tgbase(w, base), i);
  w = fput(fput(w, 35, 3, 2), 40, 8, 128);
  finish(C_TGATOMIC, w)
}
val d_tgatom : int -> option(ast)
function d_tgatom(w) = {
  start_dec();
  dchk(fget(w, 6, 2) == 0 & fget(w, 35, 3) == 2 & fget(w, 40, 8) == 128 & fget(w, 72, 8) == 0);
  let d : option(areg) = if bit_int(w, 38) then Some(d_memreg(w, 9, 60, 8)) else { dchk(fget2(w, 9, 6, 60, 2) == 0 & fget(w, 8, 1) == 0); None() };
  let sv = fget(w, 64, 8);
  dchk(not_bool(bit_int(sv, 0)));
  fin_dec(TGATOM(code_atom(fget(w, 24, 4)), d, d_tgbase(w), d_tgidx(w), OReg(mkreg(GPR, shr_int(sv, 1), 0, HNone))))
}

val e_wait : int -> option(int)
function e_wait(i) = { start_enc(); finish(C_WAIT, fput(0, 8, 1, i)) }
val e_plain : icls -> option(int)
function e_plain(c) = { start_enc(); finish(c, 0) }

/* ---------------------------------------------------------------- encode / decode */
val agx_encode : ast -> option(int)
function agx_encode(i) = match i {
  MOV(d, a) => e_mov(d, a),
  GET_SR(d, n) => e_getsr(d, n),
  IADD(sub, sat, d, a, b, sh) => e_iadd(sub, sat, d, a, b, sh),
  IMADD(sub, sat, d, a, b, c, sh) => e_imadd(sub, sat, d, a, b, c, sh),
  BITOP(tt, d, a, b) => e_bitop(tt, d, a, b),
  BFI(m, d, a, b, c) => e_bf(C_BFI, m, d, a, b, c),
  BFEIL(m, d, a, b, c) => e_bf(C_BFEIL, m, d, a, b, c),
  EXTR(m, d, a, b, c) => e_bf(C_EXTR, m, d, a, b, c),
  INTL(d, a, b) => e_intl(d, a, b),
  CONVERT(m, i2f, d, a) => e_convert(m, i2f, d, a),
  ASR(d, a, b) => e_asr(d, a, b),
  IUNOP(o, d, a) => e_iunop(o, d, a),
  FBIN(o, h, s, d, a, b) => e_fbin(o, h, s, d, a, b),
  FMADD(h, s, d, a, b, c) => e_fmadd(h, s, d, a, b, c),
  RCP(s, d, a) => e_rcp(s, d, a),
  FUNOP(o, s, d, a) => e_funop(o, s, d, a),
  CMPSEL(isf, cc, d, a, b, x, y) => e_cmpsel(isf, cc, d, a, b, x, y),
  STACKOP(k, isf, cc, a, b, n) => e_stack(k, isf, cc, a, b, n),
  POP_EXEC(n) => e_pop(n),
  JMP_EXEC(any, t) => e_jmp(any, t),
  BALLOT(isf, cc, d, a, b) => e_ballot(isf, cc, d, a, b),
  SHUFFLE(k, d, a, b) => e_shuf(k, d, a, b),
  DEVMEM(st, f, m, r, p, i, sg, s, g, u6) => e_devmem(st, f, m, r, p, i, sg, s, g, u6),
  TGMEM(st, f, m, r, b, i, s) => e_tgmem(st, f, m, r, b, i, s),
  DEVATOM(op, d, p, i, sg, v, g, m) => e_devatom(op, d, p, i, sg, v, g, m),
  TGATOM(op, d, b, i, v) => e_tgatom(op, d, b, i, v),
  WAIT(n) => e_wait(n),
  TG_BARRIER() => e_plain(C_TGBARRIER),
  MEMORY_BARRIER(a, b, c) => { start_enc(); finish(C_MEMBAR, fput(fput(fput(0, 10, 2, a), 8, 2, b), 12, 4, c)) },
  STOP() => e_plain(C_STOP),
  TRAP() => e_plain(C_TRAP)
}

/* decode the instruction at the start of an 80-bit window (applegpu's first matching descriptor) */
val agx_decode : int -> option(ast)
function agx_decode(win) = {
  let c : option(icls) = {
    var l = all_classes;
    var r : option(icls) = None();
    while is_cons(l) & is_none(r) do {
      match l { x :: rest => { if cls_matches(x, win) then r = Some(x); l = rest }, [||] => () }
    };
    r
  };
  match c {
    None() => None(),
    Some(k) => {
      let w = cls_word(k, win);
      if k.cid == C_MOVI16.cid then d_movi16(w)
      else if k.cid == C_MOVI32.cid then d_movi32(w)
      else if k.cid == C_GETSR.cid then d_getsr(w)
      else if k.cid == C_IADD.cid then d_iadd(w)
      else if k.cid == C_IMADD.cid then d_imadd(w)
      else if k.cid == C_BFI.cid then d_bfi(w)
      else if k.cid == C_BFEIL.cid then d_bfeil(w)
      else if k.cid == C_ASR.cid then d_asr(w)
      else if k.cid == C_BITOP.cid then d_bitop(w)
      else if k.cid == C_BITREV.cid then d_iunop(U_bitrev, w)
      else if k.cid == C_POPCOUNT.cid then d_iunop(U_popcount, w)
      else if k.cid == C_FFS.cid then d_iunop(U_ffs, w)
      else if k.cid == C_EXTR.cid then d_extr(w)
      else if k.cid == C_INTL.cid then d_intl(w)
      else if k.cid == C_CONVERT.cid then d_convert(w)
      else if k.cid == C_FMADD32.cid then d_fmadd(false, w)
      else if k.cid == C_FMADD16.cid then d_fmadd(true, w)
      else if k.cid == C_FADD32.cid then d_fbin(FB_add, false, w)
      else if k.cid == C_FADD16.cid then d_fbin(FB_add, true, w)
      else if k.cid == C_FMUL32.cid then d_fbin(FB_mul, false, w)
      else if k.cid == C_FMUL16.cid then d_fbin(FB_mul, true, w)
      else if k.cid == C_RCP.cid then d_rcp(w)
      else if is_some(cid_funop(k.cid)) then match cid_funop(k.cid) { Some(o) => d_funop(o, w), None() => None() }
      else if k.cid == C_STOP.cid then Some(STOP())
      else if k.cid == C_JMPANY.cid then d_jmp(true, w)
      else if k.cid == C_JMPNONE.cid then d_jmp(false, w)
      else if k.cid == C_POPEXEC.cid then d_pop(w)
      else if k.cid == C_IFICMP.cid then d_stack(SK_if, false, w)
      else if k.cid == C_IFFCMP.cid then d_stack(SK_if, true, w)
      else if k.cid == C_WHILEICMP.cid then d_stack(SK_while, false, w)
      else if k.cid == C_WHILEFCMP.cid then d_stack(SK_while, true, w)
      else if k.cid == C_ELSEICMP.cid then d_stack(SK_else, false, w)
      else if k.cid == C_ELSEFCMP.cid then d_stack(SK_else, true, w)
      else if k.cid == C_ICMPSEL.cid then d_cmpsel(false, w)
      else if k.cid == C_FCMPSEL.cid then d_cmpsel(true, w)
      else if k.cid == C_ICMPBALLOT.cid then d_ballot(false, w)
      else if k.cid == C_FCMPBALLOT.cid then d_ballot(true, w)
      else if k.cid == C_SHUF.cid then d_shuf(SH_idx, w)
      else if k.cid == C_SHUFDOWN.cid then d_shuf(SH_down, w)
      else if k.cid == C_SHUFUP.cid then d_shuf(SH_up, w)
      else if k.cid == C_SHUFXOR.cid then d_shuf(SH_xor, w)
      else if k.cid == C_WAIT.cid then Some(WAIT(fget(w, 8, 1)))
      else if k.cid == C_DEVLOAD.cid then d_devmem(false, w)
      else if k.cid == C_DEVSTORE.cid then d_devmem(true, w)
      else if k.cid == C_TGLOAD.cid then d_tgmem(false, w)
      else if k.cid == C_TGSTORE.cid then d_tgmem(true, w)
      else if k.cid == C_TGBARRIER.cid then Some(TG_BARRIER())
      else if k.cid == C_ATOMIC.cid then d_devatom(w)
      else if k.cid == C_TGATOMIC.cid then d_tgatom(w)
      else if k.cid == C_MEMBAR.cid then Some(MEMORY_BARRIER(fget(w, 10, 2), fget(w, 8, 2), fget(w, 12, 4)))
      else if k.cid == C_TRAP.cid then Some(TRAP())
      else None()
    }
  }
}

val w80 : int -> bits(80)
function w80(n) = get_slice_int(80, n, 0)

val is_some_int : option(int) -> bool
function is_some_int(o) = is_some(o)
val the_int : option(int) -> int
function the_int(o) = match o { Some(n) => n, None() => 0 }
val is_some_ast : option(ast) -> bool
function is_some_ast(o) = is_some(o)
val the_ast : option(ast) -> ast
function the_ast(o) = match o { Some(a) => a, None() => STOP() }

/* The encoding mapping.  Forwards: the instruction word (zero-extended to 80 bits); backwards: the
 * instruction at the start of an 80-bit window. */
val encdec : ast <-> bits(80)
scattered mapping encdec
mapping clause encdec = forwards (MOV(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (GET_SR(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (IADD(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (IMADD(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (BITOP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (BFI(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (BFEIL(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (ASR(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (IUNOP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (FBIN(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (FMADD(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (EXTR(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (INTL(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (CONVERT(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (FUNOP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (RCP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (CMPSEL(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (STACKOP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (POP_EXEC(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (JMP_EXEC(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (BALLOT(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (SHUFFLE(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (DEVMEM(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (TGMEM(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (DEVATOM(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (TGATOM(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (WAIT(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (TG_BARRIER(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (STOP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (MEMORY_BARRIER(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = forwards (TRAP(_) as i) if is_some_int(agx_encode(i)) => w80(the_int(agx_encode(i)))
mapping clause encdec = backwards w if is_some_ast(agx_decode(unsigned(w))) => the_ast(agx_decode(unsigned(w)))
end encdec

/* the length in bytes of the instruction at the start of an 80-bit window (0: not decodable) */
val agx_size : bits(80) -> int
function agx_size(w) = agx_size_int(unsigned(w))

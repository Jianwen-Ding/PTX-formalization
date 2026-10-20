/* =============================================================================
 * The assembly dialect of agx/tests (agx.k's AGX-SYNTAX): `mapping clause assembly`.
 *
 *   r5 / r5l / r5h / u3 / u3l   registers (rN_rM: 64-bit pair), suffixes .cache .discard (hints)
 *                               and .sx .abs .neg (the encodings' source modifiers)
 *   7, -1, 4294967295           integer immediates;   2.0, 0.5   float literals
 *   u2_u3                       64-bit base pair of a device access
 * Fields that the K dialect has no syntax for are printed only when they differ from the K
 * model's implicit value: `.sat` on the mnemonic, a trailing `lsl N` shift, the push count of
 * if/else/while compares, `mask=N` for non-contiguous element masks, `g1`, `u6`, a threadgroup
 * base `base=OP`, `signed` on atomics, `mode=N` on device atomics, and `_` for an atomic without destination.
 * The driver (sail/agx/agxtest.py) parses this text; Sail prints it (assembly forwards), and the
 * test driver checks that printing the parsed program gives back the parsed program.
 * ============================================================================= */

val dec_int : int -> string
function dec_int(n) = dec_str(n)

/* exact decimal rendering of a binary64 value (a K Float literal) */
val f64_str : int -> string
function f64_str(b) = match f64v(b) {
  FV_NaN() => "NaN",
  FV_Inf(s) => if s then "-Infinity" else "Infinity",
  FV_Fin(s, m, e) => {
    let sg = if s then "-" else "";
    if e >= 0 then concat_str(sg, concat_str(dec_str(m * pow2(e)), ".0"))
    else {
      let k = negate(e);
      let n = m * pow(5, k);
      let ip = tdiv_int(n, pow(10, k));
      var fp : int = n - ip * pow(10, k);
      var digits : string = "";
      var i : int = 0;
      while i < k do {
        fp = fp * 10;
        let d = tdiv_int(fp, pow(10, k));
        digits = concat_str(digits, dec_str(d));
        fp = fp - d * pow(10, k);
        i = i + 1
      };
      /* strip trailing zeros, keep one digit */
      var len : int = string_length(digits);
      while len > 1 & sdrop(digits, len - 1) == "0" do { digits = stake(digits, len - 1); len = len - 1 };
      concat_str(sg, concat_str(dec_str(ip), concat_str(".", digits)))
    }
  }
}

val reg_str : areg -> string
function reg_str(r) = {
  let p = if r.file == UNI then "u" else "r";
  let base : string = match r.view {
    0 => concat_str(p, dec_str(r.num)),
    1 => concat_str(p, concat_str(dec_str(r.num), "l")),
    2 => concat_str(p, concat_str(dec_str(r.num), "h")),
    _ => concat_str(concat_str(p, dec_str(r.num)), concat_str(concat_str("_", p), dec_str(r.num + 1)))
  };
  let h : string = match r.hint { HNone => "", HCache => ".cache", HDiscard => ".discard" };
  concat_str(base, concat_str(h, concat_str(if r.sx then ".sx" else "", concat_str(if r.fabs then ".abs" else "", if r.fneg then ".neg" else ""))))
}

val op_str : aop -> string
function op_str(a) = match a { OReg(r) => reg_str(r), OImm(i) => dec_str(i), OFImm(b) => f64_str(b) }

val pair_str : apair -> string
function pair_str(p) = {
  let c = if p.pfile == UNI then "u" else "r";
  concat_str(concat_str(c, dec_str(p.lo)), concat_str(concat_str("_", c), dec_str(p.hi)))
}

val cc_str : acc -> string
function cc_str(c) = match c {
  CC_eq => "eq", CC_ne => "ne", CC_ult => "ult", CC_ule => "ule", CC_ugt => "ugt", CC_uge => "uge",
  CC_slt => "slt", CC_sle => "sle", CC_sgt => "sgt", CC_sge => "sge",
  CC_feq => "feq", CC_fne => "fne", CC_flt => "flt", CC_fle => "fle", CC_fgt => "fgt", CC_fge => "fge",
  CC_seq => "seq", CC_sne => "sne", CC_fltn => "fltn", CC_fgtn => "fgtn", CC_fnequ => "fnequ",
  CC_fnlt => "fnlt", CC_fngt => "fngt", CC_fnle => "fnle", CC_fnge => "fnge", CC_fnltn => "fnltn", CC_fngtn => "fngtn"
}

val fmt_str : afmt -> string
function fmt_str(f) = match f { F_U8 => "u8", F_U16 => "u16", F_U32 => "u32" }

val aop_str : aaop -> string
function aop_str(o) = match o {
  A_add => "add", A_sub => "sub", A_xchg => "xchg", A_umin => "min", A_imin => "imin", A_umax => "max",
  A_imax => "imax", A_and => "and", A_or => "or", A_xor => "xor", A_cmpxchg => "cmpxchg", A_fadd => "fadd",
  A_umin64 => "umin64", A_umax64 => "umax64", A_end11 => "end11", A_end12 => "end12", A_end13 => "end13",
  A_end14 => "end14", A_end15 => "end15"
}

/* element mask: the K model's count N for masks 2^N - 1, else mask=M */
val mask_str : int -> string
function mask_str(m) =
  if m == 0 | m == 1 | m == 3 | m == 7 | m == 15 then dec_str(popc_int(m)) else concat_str("mask=", dec_str(m))

/* string mappings used by the clauses (printing direction) */
val comma : unit <-> string
function comma_forwards() = ", "
function comma_forwards_matches() = true
function comma_backwards(_) = ()
function comma_backwards_matches(s) = s == ", "
val comma_matches_prefix : string -> option((unit, nat))
function comma_matches_prefix(s) = if string_startswith(s, ", ") then Some((), 2) else None()

val asm_reg : areg <-> string
function asm_reg_forwards(r) = reg_str(r)
function asm_reg_forwards_matches(_) = true
function asm_reg_backwards(_) = gpr32(0)
function asm_reg_backwards_matches(_) = false
val asm_reg_matches_prefix : string -> option((areg, nat))
function asm_reg_matches_prefix(_) = None()

val asm_op : aop <-> string
function asm_op_forwards(a) = op_str(a)
function asm_op_forwards_matches(_) = true
function asm_op_backwards(_) = OImm(0)
function asm_op_backwards_matches(_) = false
val asm_op_matches_prefix : string -> option((aop, nat))
function asm_op_matches_prefix(_) = None()

val asm_int : int <-> string
function asm_int_forwards(n) = dec_str(n)
function asm_int_forwards_matches(_) = true
function asm_int_backwards(_) = 0
function asm_int_backwards_matches(_) = false
val asm_int_matches_prefix : string -> option((int, nat))
function asm_int_matches_prefix(_) = None()

val asm_cc : acc <-> string
function asm_cc_forwards(c) = cc_str(c)
function asm_cc_forwards_matches(_) = true
function asm_cc_backwards(_) = CC_eq
function asm_cc_backwards_matches(_) = false
val asm_cc_matches_prefix : string -> option((acc, nat))
function asm_cc_matches_prefix(_) = None()

val asm_pair : apair <-> string
function asm_pair_forwards(p) = pair_str(p)
function asm_pair_forwards_matches(_) = true
function asm_pair_backwards(_) = struct { pfile = UNI, lo = 0, hi = 1 }
function asm_pair_backwards_matches(_) = false
val asm_pair_matches_prefix : string -> option((apair, nat))
function asm_pair_matches_prefix(_) = None()

/* optional suffixes */
val asm_sat : bool <-> string
function asm_sat_forwards(b) = if b then ".sat" else ""
function asm_sat_forwards_matches(_) = true
function asm_sat_backwards(s) = s == ".sat"
function asm_sat_backwards_matches(s) = s == ".sat" | s == ""
val asm_sat_matches_prefix : string -> option((bool, nat))
function asm_sat_matches_prefix(s) = if string_startswith(s, ".sat") then Some((true, 4)) else Some((false, 0))

val asm_shift : int <-> string
function asm_shift_forwards(n) = if n == 0 then "" else concat_str(", lsl ", dec_str(n))
function asm_shift_forwards_matches(_) = true
function asm_shift_backwards(_) = 0
function asm_shift_backwards_matches(_) = false
val asm_shift_matches_prefix : string -> option((int, nat))
function asm_shift_matches_prefix(_) = None()

val asm_push : int <-> string
function asm_push_forwards(n) = if n == 1 then "" else concat_str(", ", dec_str(n))
function asm_push_forwards_matches(_) = true
function asm_push_backwards(_) = 1
function asm_push_backwards_matches(_) = false
val asm_push_matches_prefix : string -> option((int, nat))
function asm_push_matches_prefix(_) = None()

val asm_target : jtarget <-> string
function asm_target_forwards(t) = match t {
  JLabel(l) => l,
  JRel(n) => if n >= 0 then concat_str("pc+", dec_str(n)) else concat_str("pc-", dec_str(negate(n)))
}
function asm_target_forwards_matches(_) = true
function asm_target_backwards(_) = JRel(0)
function asm_target_backwards_matches(_) = false
val asm_target_matches_prefix : string -> option((jtarget, nat))
function asm_target_matches_prefix(_) = None()

/* extra fields of memory instructions: wait group, store-order bit */
val asm_memx : (int, int) <-> string
function asm_memx_forwards(g, u6) = concat_str(if g != 0 then ", g1" else "", if u6 != 0 then ", u6" else "")
function asm_memx_forwards_matches(_) = true
function asm_memx_backwards(_) = (0, 0)
function asm_memx_backwards_matches(_) = false
val asm_memx_matches_prefix : string -> option(((int, int), nat))
function asm_memx_matches_prefix(_) = None()

val asm_mask : int <-> string
function asm_mask_forwards(m) = mask_str(m)
function asm_mask_forwards_matches(_) = true
function asm_mask_backwards(_) = 1
function asm_mask_backwards_matches(_) = false
val asm_mask_matches_prefix : string -> option((int, nat))
function asm_mask_matches_prefix(_) = None()

val asm_sign : bool <-> string
function asm_sign_forwards(b) = if b then "signed" else "unsigned"
function asm_sign_forwards_matches(_) = true
function asm_sign_backwards(s) = s == "signed"
function asm_sign_backwards_matches(s) = s == "signed" | s == "unsigned"
val asm_sign_matches_prefix : string -> option((bool, nat))
function asm_sign_matches_prefix(s) = if string_startswith(s, "signed") then Some((true, 6)) else if string_startswith(s, "unsigned") then Some((false, 8)) else None()

/* threadgroup base (the K model's base is 0) */
val asm_tgbase : aop <-> string
function asm_tgbase_forwards(a) = match a { OImm(0) => "", _ => concat_str(", base=", op_str(a)) }
function asm_tgbase_forwards_matches(_) = true
function asm_tgbase_backwards(_) = OImm(0)
function asm_tgbase_backwards_matches(_) = false
val asm_tgbase_matches_prefix : string -> option((aop, nat))
function asm_tgbase_matches_prefix(_) = None()

val asm_adst : option(areg) <-> string
function asm_adst_forwards(d) = match d { Some(r) => reg_str(r), None() => "_" }
function asm_adst_forwards_matches(_) = true
function asm_adst_backwards(_) = None()
function asm_adst_backwards_matches(_) = false
val asm_adst_matches_prefix : string -> option((option(areg), nat))
function asm_adst_matches_prefix(_) = None()

/* device atomics: signedness (K: zero-extended index), wait group, and `mode=N` (bits 45-46) when it differs from
 * the op's default: 1 for fadd and umax64, 3 for umin64, else 0 (Mesa's encoding) */
val atom_mode_default : aaop -> int
function atom_mode_default(o) = match o { A_fadd => 1, A_umax64 => 1, A_umin64 => 3, _ => 0 }

val asm_atomx_forwards : (aaop, bool, int, int) -> string
function asm_atomx_forwards(o, sg, g, m) = concat_str(concat_str(if sg then ", signed" else "", if g != 0 then ", g1" else ""),
                                                     if m != atom_mode_default(o) then concat_str(", mode=", dec_str(m)) else "")

val iunop_mnem : iunop <-> string
mapping iunop_mnem = { U_popcount <-> "popcount", U_bitrev <-> "bitrev", U_ffs <-> "ffs" }

val shuf_mnem : shufkind <-> string
mapping shuf_mnem = { SH_idx <-> "simd_shuffle", SH_xor <-> "simd_shuffle_xor", SH_up <-> "simd_shuffle_up", SH_down <-> "simd_shuffle_down" }

val stk_mnem : stkkind <-> string
mapping stk_mnem = { SK_if <-> "if", SK_else <-> "else", SK_while <-> "while" }

val ftype : bool <-> string
mapping ftype = { true <-> "f", false <-> "i" }

val half16 : bool <-> string
mapping half16 = { true <-> "16", false <-> "" }

val fbin_mnem : fbinop <-> string
mapping fbin_mnem = { FB_add <-> "fadd", FB_mul <-> "fmul" }

val asm_fmt : afmt <-> string
mapping asm_fmt = { F_U8 <-> "u8", F_U16 <-> "u16", F_U32 <-> "u32" }

val asm_aop : aaop <-> string
mapping asm_aop = { A_add <-> "add", A_sub <-> "sub", A_xchg <-> "xchg", A_umin <-> "min", A_imin <-> "imin",
  A_umax <-> "max", A_imax <-> "imax", A_and <-> "and", A_or <-> "or", A_xor <-> "xor", A_cmpxchg <-> "cmpxchg",
  A_fadd <-> "fadd", A_umin64 <-> "umin64", A_umax64 <-> "umax64", A_end11 <-> "end11", A_end12 <-> "end12",
  A_end13 <-> "end13", A_end14 <-> "end14", A_end15 <-> "end15" }

val assembly : ast <-> string
scattered mapping assembly

mapping clause assembly = MOV(d, a) <-> "mov" ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a)
mapping clause assembly = GET_SR(d, n) <-> "get_sr" ^ spc() ^ asm_reg(d) ^ comma() ^ asm_int(n)
mapping clause assembly = IADD(false, s, d, a, b, sh) <-> "iadd" ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ asm_shift(sh)
mapping clause assembly = IADD(true, s, d, a, b, sh) <-> "isub" ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ asm_shift(sh)
mapping clause assembly = IMADD(false, s, d, a, b, c, sh) <-> "imadd" ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ comma() ^ asm_op(c) ^ asm_shift(sh)
mapping clause assembly = IMADD(true, s, d, a, b, c, sh) <-> "imsub" ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ comma() ^ asm_op(c) ^ asm_shift(sh)
mapping clause assembly = BITOP(tt, d, a, b) <-> "bitop" ^ spc() ^ asm_int(tt) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b)
mapping clause assembly = BFI(m, d, a, b, c) <-> "bfi" ^ spc() ^ asm_int(m) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ comma() ^ asm_op(c)
mapping clause assembly = BFEIL(m, d, a, b, c) <-> "bfeil" ^ spc() ^ asm_int(m) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ comma() ^ asm_op(c)
mapping clause assembly = ASR(d, a, b) <-> "asr" ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b)
mapping clause assembly = IUNOP(o, d, a) <-> iunop_mnem(o) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a)
mapping clause assembly = FBIN(o, h, s, d, a, b) <-> fbin_mnem(o) ^ half16(h) ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b)
mapping clause assembly = FMADD(h, s, d, a, b, c) <-> "fmadd" ^ half16(h) ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ comma() ^ asm_op(c)
mapping clause assembly = RCP(s, d, a) <-> "rcp" ^ asm_sat(s) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a)
mapping clause assembly = CMPSEL(f, cc, d, a, b, x, y) <-> ftype(f) ^ "cmpsel" ^ spc() ^ asm_cc(cc) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ comma() ^ asm_op(x) ^ comma() ^ asm_op(y)
mapping clause assembly = STACKOP(k, f, cc, a, b, n) <-> stk_mnem(k) ^ "_" ^ ftype(f) ^ "cmp" ^ spc() ^ asm_cc(cc) ^ spc() ^ asm_op(a) ^ comma() ^ asm_op(b) ^ asm_push(n)
mapping clause assembly = POP_EXEC(n) <-> "pop_exec" ^ spc() ^ asm_int(n)
mapping clause assembly = JMP_EXEC(true, t) <-> "jmp_exec_any" ^ spc() ^ asm_target(t)
mapping clause assembly = JMP_EXEC(false, t) <-> "jmp_exec_none" ^ spc() ^ asm_target(t)
mapping clause assembly = BALLOT(f, cc, d, a, b) <-> ftype(f) ^ "cmp_ballot" ^ spc() ^ asm_cc(cc) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b)
mapping clause assembly = SHUFFLE(k, d, a, b) <-> shuf_mnem(k) ^ spc() ^ asm_reg(d) ^ comma() ^ asm_op(a) ^ comma() ^ asm_op(b)
mapping clause assembly = DEVMEM(false, f, m, r, p, i, sg, s, g, u6) <-> "device_load" ^ spc() ^ asm_fmt(f) ^ spc() ^ asm_mask(m) ^ spc() ^ asm_reg(r) ^ comma() ^ asm_pair(p) ^ comma() ^ asm_op(i) ^ comma() ^ asm_sign(sg) ^ comma() ^ asm_int(s) ^ asm_memx(g, u6)
mapping clause assembly = DEVMEM(true, f, m, r, p, i, sg, s, g, u6) <-> "device_store" ^ spc() ^ asm_fmt(f) ^ spc() ^ asm_mask(m) ^ spc() ^ asm_reg(r) ^ comma() ^ asm_pair(p) ^ comma() ^ asm_op(i) ^ comma() ^ asm_sign(sg) ^ comma() ^ asm_int(s) ^ asm_memx(g, u6)
mapping clause assembly = TGMEM(false, f, m, r, b, i, s) <-> "threadgroup_load" ^ spc() ^ asm_fmt(f) ^ spc() ^ asm_mask(m) ^ spc() ^ asm_reg(r) ^ comma() ^ asm_op(i) ^ comma() ^ asm_int(s) ^ asm_tgbase(b)
mapping clause assembly = TGMEM(true, f, m, r, b, i, s) <-> "threadgroup_store" ^ spc() ^ asm_fmt(f) ^ spc() ^ asm_mask(m) ^ spc() ^ asm_reg(r) ^ comma() ^ asm_op(i) ^ comma() ^ asm_int(s) ^ asm_tgbase(b)
mapping clause assembly = forwards DEVATOM(o, d, p, i, sg, v, g, m) =>
  concat_str("device_atomic ", concat_str(aop_str(o), concat_str(" ", concat_str(asm_adst_forwards(d), concat_str(", ",
  concat_str(pair_str(p), concat_str(", ", concat_str(op_str(i), concat_str(", ", concat_str(op_str(v), asm_atomx_forwards(o, sg, g, m)))))))))))
mapping clause assembly = TGATOM(o, d, b, i, v) <-> "threadgroup_atomic" ^ spc() ^ asm_aop(o) ^ spc() ^ asm_adst(d) ^ comma() ^ asm_op(i) ^ comma() ^ asm_op(v) ^ asm_tgbase(b)
mapping clause assembly = WAIT(n) <-> "wait" ^ spc() ^ asm_int(n)
mapping clause assembly = TG_BARRIER() <-> "threadgroup_barrier"
mapping clause assembly = MEMORY_BARRIER(a, b, c) <-> "memory_barrier" ^ spc() ^ asm_int(a) ^ comma() ^ asm_int(b) ^ comma() ^ asm_int(c)
mapping clause assembly = STOP() <-> "stop"
mapping clause assembly = TRAP() <-> "trap"

end assembly

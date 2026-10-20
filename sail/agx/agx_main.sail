/* =============================================================================
 * Driver: reads a test program from the token stream written by sail/agx/agxtest.py and processes
 * it like the <k> cell of agx/agx.k: instructions and labels are loaded in order, test directives
 * are executed when they are reached, `.run` runs the SIMD-group until `stop`, and a failing
 * `.expect*` directive (#fatal) ends the program.  Prints the final status, exit code and message.
 * ============================================================================= */

val tok_str = impure {c: "sail_tok_str"} : unit -> string
val tok_int = impure {c: "sail_tok_int"} : unit -> int

/* ---------------------------------------------------------------- reading the serialized AST */
val rd_bool : unit -> bool
function rd_bool() = tok_int() != 0

val rd_reg : unit -> areg
function rd_reg() = {
  let f = if tok_int() == 1 then UNI else GPR;
  let n = tok_int();
  let v = tok_int();
  let h : hint = match tok_int() { 1 => HCache, 2 => HDiscard, _ => HNone };
  let sx = rd_bool();
  let ab = rd_bool();
  let ng = rd_bool();
  struct { file = f, num = n, view = v, hint = h, sx = sx, fabs = ab, fneg = ng }
}

val rd_op : unit -> aop
function rd_op() = match tok_str() {
  "R" => OReg(rd_reg()),
  "I" => OImm(tok_int()),
  _ => OFImm(tok_int())
}

val rd_pair : unit -> apair
function rd_pair() = {
  let f = if tok_int() == 1 then UNI else GPR;
  let lo = tok_int();
  let hi = tok_int();
  struct { pfile = f, lo = lo, hi = hi }
}

val rd_cc : unit -> acc
function rd_cc() = match tok_str() {
  "eq" => CC_eq, "ne" => CC_ne, "ult" => CC_ult, "ule" => CC_ule, "ugt" => CC_ugt, "uge" => CC_uge,
  "slt" => CC_slt, "sle" => CC_sle, "sgt" => CC_sgt, "sge" => CC_sge,
  "feq" => CC_feq, "fne" => CC_fne, "flt" => CC_flt, "fle" => CC_fle, "fgt" => CC_fgt, "fge" => CC_fge,
  "seq" => CC_seq, "sne" => CC_sne, "fltn" => CC_fltn, "fgtn" => CC_fgtn, "fnequ" => CC_fnequ,
  "fnlt" => CC_fnlt, "fngt" => CC_fngt, "fnle" => CC_fnle, "fnge" => CC_fnge, "fnltn" => CC_fnltn, _ => CC_fngtn
}

val rd_fmt : unit -> afmt
function rd_fmt() = match tok_str() { "u8" => F_U8, "u16" => F_U16, _ => F_U32 }

val rd_aop : unit -> aaop
function rd_aop() = match tok_str() {
  "add" => A_add, "sub" => A_sub, "xchg" => A_xchg, "min" => A_umin, "imin" => A_imin, "max" => A_umax,
  "imax" => A_imax, "and" => A_and, "or" => A_or, _ => A_xor
}

val rd_adst : unit -> option(areg)
function rd_adst() = match tok_str() { "S" => Some(rd_reg()), _ => None() }

val rd_instr : unit -> ast
function rd_instr() = match tok_str() {
  "MOV" => { let d = rd_reg(); let a = rd_op(); MOV(d, a) },
  "GET_SR" => { let d = rd_reg(); let n = tok_int(); GET_SR(d, n) },
  "IADD" => { let sb = rd_bool(); let st = rd_bool(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); let s = tok_int(); IADD(sb, st, d, a, b, s) },
  "IMADD" => { let sb = rd_bool(); let st = rd_bool(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); let c = rd_op(); let s = tok_int(); IMADD(sb, st, d, a, b, c, s) },
  "BITOP" => { let tt = tok_int(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); BITOP(tt, d, a, b) },
  "BFI" => { let m = tok_int(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); let c = rd_op(); BFI(m, d, a, b, c) },
  "BFEIL" => { let m = tok_int(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); let c = rd_op(); BFEIL(m, d, a, b, c) },
  "ASR" => { let d = rd_reg(); let a = rd_op(); let b = rd_op(); ASR(d, a, b) },
  "IUNOP" => {
    let o : iunop = match tok_str() { "popcount" => U_popcount, "bitrev" => U_bitrev, _ => U_ffs };
    let d = rd_reg(); let a = rd_op(); IUNOP(o, d, a)
  },
  "FBIN" => {
    let o : fbinop = match tok_str() { "add" => FB_add, _ => FB_mul };
    let h = rd_bool(); let st = rd_bool(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); FBIN(o, h, st, d, a, b)
  },
  "FMADD" => { let h = rd_bool(); let st = rd_bool(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); let c = rd_op(); FMADD(h, st, d, a, b, c) },
  "RCP" => { let st = rd_bool(); let d = rd_reg(); let a = rd_op(); RCP(st, d, a) },
  "CMPSEL" => { let f = rd_bool(); let cc = rd_cc(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); let x = rd_op(); let y = rd_op(); CMPSEL(f, cc, d, a, b, x, y) },
  "STACKOP" => {
    let k : stkkind = match tok_str() { "if" => SK_if, "else" => SK_else, _ => SK_while };
    let f = rd_bool(); let cc = rd_cc(); let a = rd_op(); let b = rd_op(); let n = tok_int(); STACKOP(k, f, cc, a, b, n)
  },
  "POP_EXEC" => POP_EXEC(tok_int()),
  "JMP" => {
    let any = rd_bool();
    match tok_str() { "JL" => JMP_EXEC(any, JLabel(tok_str())), _ => JMP_EXEC(any, JRel(tok_int())) }
  },
  "BALLOT" => { let f = rd_bool(); let cc = rd_cc(); let d = rd_reg(); let a = rd_op(); let b = rd_op(); BALLOT(f, cc, d, a, b) },
  "SHUFFLE" => {
    let k : shufkind = match tok_str() { "idx" => SH_idx, "xor" => SH_xor, "up" => SH_up, _ => SH_down };
    let d = rd_reg(); let a = rd_op(); let b = rd_op(); SHUFFLE(k, d, a, b)
  },
  "DEVMEM" => {
    let st = rd_bool(); let f = rd_fmt(); let m = tok_int(); let r = rd_reg(); let p = rd_pair(); let i = rd_op();
    let sg = rd_bool(); let s = tok_int(); let g = tok_int(); let u6 = tok_int();
    DEVMEM(st, f, m, r, p, i, sg, s, g, u6)
  },
  "TGMEM" => { let st = rd_bool(); let f = rd_fmt(); let m = tok_int(); let r = rd_reg(); let b = rd_op(); let i = rd_op(); let s = tok_int(); TGMEM(st, f, m, r, b, i, s) },
  "DEVATOM" => { let o = rd_aop(); let d = rd_adst(); let p = rd_pair(); let i = rd_op(); let sg = rd_bool(); let v = rd_op(); let g = tok_int(); DEVATOM(o, d, p, i, sg, v, g) },
  "TGATOM" => { let o = rd_aop(); let d = rd_adst(); let b = rd_op(); let i = rd_op(); let v = rd_op(); TGATOM(o, d, b, i, v) },
  "WAIT" => WAIT(tok_int()),
  "TG_BARRIER" => TG_BARRIER(),
  _ => STOP()
}

/* ---------------------------------------------------------------- layout and labels */
val enc_word : ast -> option(int)
function enc_word(i) = if encdec_forwards_matches(i) then Some(unsigned(encdec_forwards(i))) else None()

/* byte length used for the layout: the encoding's, 6 for jumps (whose label is not resolved yet),
 * 8 for instructions without an encoding (immediates beyond the encoded ranges, ...) */
val instr_size : ast -> int
function instr_size(i) = match i {
  JMP_EXEC(_) => 6,
  _ => match enc_word(i) { Some(w) => agx_size(w80(w)), None() => 8 }
}

val add_instr : ast -> unit
function add_instr(i) = {
  assert(ncode < 4096, "program too long");
  set_code(ncode, i);
  set_addr(ncode, if ncode == 0 then 0 else get_addr(ncode - 1) + instr_size(get_code(ncode - 1)));
  ncode = ncode + 1
}

val label_pc : (list((string, int)), string) -> option(int)
function label_pc(l, n) = match l {
  [||] => None(),
  (m, p) :: rest => if m == n then Some(p) else label_pc(rest, n)
}

val end_addr : unit -> int
function end_addr() = if ncode == 0 then 0 else get_addr(ncode - 1) + instr_size(get_code(ncode - 1))

val addr_of_pc : int -> int
function addr_of_pc(p) = if p < ncode then get_addr(p) else end_addr()

/* resolve label targets to byte offsets relative to the jump */
val resolve : unit -> unit
function resolve() = {
  foreach (i from 0 to (ncode - 1)) {
    match get_code(i) {
      JMP_EXEC(any, JLabel(l)) => match label_pc(labels, l) {
        Some(p) => set_code(i, JMP_EXEC(any, JRel(addr_of_pc(p) - get_addr(i)))),
        None() => ()
      },
      _ => ()
    }
  }
}

/* ---------------------------------------------------------------- running */
val run : unit -> unit
function run() = {
  resolve();
  status = ST_running;
  var steps : int = 0;
  while status == ST_running do {
    if pc < 0 | pc >= ncode then fatal("fell off the end of the program")
    else {
      let i = get_code(pc);
      pc = pc + 1;
      execute(i);
      steps = steps + 1;
      if steps > 10000000 then fatal("step limit")
    }
  }
}

/* ---------------------------------------------------------------- test directives */
val all_eq : (areg, int) -> bool
function all_eq(r, v) = {
  var ok : bool = true;
  foreach (t from 0 to 31) { if rval(t, r) != land_int(v, amask(view_bits(r))) then ok = false };
  ok
}

val reg_name : areg -> string
function reg_name(r) = reg_str(r)

val directive : string -> unit
function directive(d) = match d {
  "all" => { let r = rd_reg(); let v = tok_int(); foreach (t from 0 to 31) put(t, r, v) },
  "laneid" => { let r = rd_reg(); foreach (t from 0 to 31) put(t, r, t) },
  "lane" => { let t = tok_int(); let r = rd_reg(); let v = tok_int(); put(t, r, v) },
  "uni" => { let r = rd_reg(); let v = tok_int(); set_upr(r.num, get_slice_int(32, v, 0)) },
  "sr" => { let n = tok_int(); foreach (t from 0 to 31) set_sr_v(n * 32 + t, get_slice_int(32, t, 0)) },
  "srall" => { let n = tok_int(); let v = tok_int(); foreach (t from 0 to 31) set_sr_v(n * 32 + t, get_slice_int(32, v, 0)) },
  "dev" => { let a = tok_int(); let v = tok_int(); wrBytes(SP_DEV, a, v, 4) },
  "run" => { if status == ST_loading then run() },
  "expectall" => {
    let r = rd_reg(); let v = tok_int();
    if not_bool(all_eq(r, v)) then fatal(concat_str("expectall ", concat_str(reg_name(r), concat_str(" = ", concat_str(dec_str(v), " failed")))))
  },
  "expectlane" => {
    let t = tok_int(); let r = rd_reg(); let v = tok_int();
    if rval(t, r) != land_int(v, amask(view_bits(r))) then
      fatal(concat_str("expectlane ", concat_str(dec_str(t), concat_str(" ", concat_str(reg_name(r), concat_str(" got ", dec_str(rval(t, r))))))))
  },
  "expectdev" => {
    let a = tok_int(); let v = tok_int();
    if rdBytes(SP_DEV, a, 4) != land_int(v, 4294967295) then fatal(concat_str("expectdev ", concat_str(dec_str(a), concat_str(" got ", dec_str(rdBytes(SP_DEV, a, 4))))))
  },
  "dumpdev" => {
    /* print n bytes of device memory from address a as hex (differential tests) */
    let a = tok_int(); let n = tok_int();
    var s : string = "";
    foreach (i from 0 to (n - 1)) { s = concat_str(s, sdrop(hex_str(mem_rd(SP_DEV, a + i) + 256), 3)) };
    print_endline(concat_str("DEVMEM ", concat_str(dec_str(a), concat_str(" ", s))))
  },
  "expectmask" => { let v = tok_int(); if unsigned(em) != v then fatal(concat_str("exec mask is ", dec_str(unsigned(em)))) },
  _ => fatal(concat_str("unknown directive ", d))
}

/* ---------------------------------------------------------------- encoding checks */
val hex_word : (int, int) -> string
function hex_word(w, n) = {
  var s : string = "";
  foreach (i from 0 to (n - 1)) {
    let b = fget(w, 8 * i, 8);
    let h = hex_str(b + 256);
    s = concat_str(s, sdrop(h, 3))
  };
  s
}

/* print each instruction, its encoding, and check that decoding the encoding gives it back (jump labels are
 * resolved first, so the check can come before or after `.run`) */
val check_encodings : unit -> unit
function check_encodings() = {
  resolve();
  foreach (i from 0 to (ncode - 1)) {
    let a = get_code(i);
    let s = assembly_forwards(a);
    match enc_word(a) {
      None() => print_endline(concat_str("NOENC ", s)),
      Some(w) => {
        let n = agx_size(w80(w));
        /* decode from a window whose bytes beyond the instruction are garbage */
        let win = w + shl_int(782246118574171818927525, 8 * n);
        let back = if encdec_backwards_matches(get_slice_int(80, win, 0)) then assembly_forwards(encdec_backwards(get_slice_int(80, win, 0))) else "<no decode>";
        let n2 = agx_size(get_slice_int(80, win, 0));
        print_endline(concat_str(concat_str(concat_str("ENC ", hex_word(w, n)), concat_str(" ", s)),
                                 if back == s & n2 == n then " RT-OK" else concat_str(" RT-FAIL ", back)))
      }
    }
  }
}

/* decode one 80-bit window: prints DEC <size> <assembly> or DEC 0 */
val decode_window : int -> unit
function decode_window(w) = {
  let n = agx_size(get_slice_int(80, w, 0));
  if n > 0 & encdec_backwards_matches(get_slice_int(80, w, 0)) then
    print_endline(concat_str(concat_str("DEC ", dec_str(n)), concat_str(" ", assembly_forwards(encdec_backwards(get_slice_int(80, w, 0))))))
  else print_endline("DEC 0")
}

/* load instructions from code bytes given as one integer (little-endian) of n bytes */
val load_code : (int, int) -> unit
function load_code(w, n) = {
  var off : int = 0;
  while off < n & status != ST_fault do {
    let win = trunc_int(shr_int(w, 8 * off), 80);
    let sz = agx_size(get_slice_int(80, win, 0));
    if sz == 0 | not_bool(encdec_backwards_matches(get_slice_int(80, win, 0))) then fatal(concat_str("cannot decode at byte ", dec_str(off)))
    else { add_instr(encdec_backwards(get_slice_int(80, win, 0))); off = off + sz }
  }
}

val main : unit -> unit
function main() = {
  init_state();
  var finished : bool = false;
  while not_bool(finished) do {
    match tok_str() {
      "I" => { let i = rd_instr(); if status == ST_loading then add_instr(i) else fatal("instruction after .run") },
      "L" => { let l = tok_str(); labels = (l, ncode) :: labels },
      "D" => { let d = tok_str(); directive(d) },
      "C" => { let n = tok_int(); let w = tok_int(); load_code(w, n) },
      "X" => check_encodings(),
      "V" => decode_window(tok_int()),
      _ => finished = true
    };
    if status == ST_fault then finished = true
  };
  let st : string = match status { ST_loading => "loading", ST_running => "running", ST_stopped => "stopped", ST_fault => "fault" };
  print_endline(concat_str("STATUS ", concat_str(st, concat_str(" EXIT ", concat_str(dec_str(exit_code), concat_str(" MSG ", msg))))))
}

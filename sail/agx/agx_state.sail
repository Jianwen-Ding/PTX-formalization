/* =============================================================================
 * AGX machine state (configuration of agx/agx.k):
 *   <pc> <em> <gpr> <upr> <dmem> <tmem> <srs> <pend> <status> <msg> <code> <labels>
 * One SIMD-group of 32 lanes in lock step.  General registers r0..r255 per lane (16-bit halves are
 * views of the 32-bit registers; r0l is the execution-mask stack counter), 256 uniform registers,
 * special registers per lane, and byte-addressed device and threadgroup memories.
 * ============================================================================= */

/* lane * 256 + register |-> 32-bit value */
register gpr : vector(8192, bits(32))
/* uniform register |-> 32-bit value */
register upr : vector(256, bits(32))
/* special register * 32 + lane |-> value */
register srs : vector(8192, bits(32))
/* execution mask */
register em : bits(32)
/* program counter: index into `code` */
register pc : int

/* ST_trapped: execution ended by `trap` (provisional, see agx_exec.sail) */
enum astatus = { ST_loading, ST_running, ST_stopped, ST_trapped, ST_fault }
register status : astatus
register exit_code : int
register msg : string

/* loads (and atomics' old values) not yet waited for: (lane, register, value) */
register pend : list((int, int, int)) = [||]

/* the program: instructions, their byte addresses, labels */
register code : vector(4096, ast) = vector_init(STOP())
register code_addr : vector(4096, int)
register ncode : int
register labels : list((string, int)) = [||]

/* device and threadgroup memories are sparse byte maps (missing bytes read as 0), kept by the
 * C runtime (sail/common/io.c); space 0 = device, space 1 = threadgroup */
val mem_rd = pure {c: "sail_mem_rd"} : (int, int) -> int
val mem_wr = impure {c: "sail_mem_wr"} : (int, int, int) -> unit

let SP_DEV : int = 0
let SP_TG : int = 1
/* the SIMD group's stack (scratch) allocation, byte-addressed from its start (agx_exec.sail stack_addr) */
let SP_STACK : int = 2

/* the size of the stack allocation in 32-bit words per lane, set by stack_adjust (0: none) */
register stack_words : int = 0

/* wrBytes / rdBytes: little-endian multi-byte access */
val wrBytes : (int, int, int, int) -> unit
function wrBytes(sp, a, v, n) = {
  var x : int = v;
  foreach (i from 0 to (n - 1)) {
    mem_wr(sp, a + i, land_int(x, 255));
    x = asr_int(x, 8)
  }
}

val rdBytes : (int, int, int) -> int
function rdBytes(sp, a, n) = {
  var r : int = 0;
  foreach (i from (n - 1) downto 0) {
    r = r * 256 + mem_rd(sp, a + i)
  };
  r
}

val init_state : unit -> unit
function init_state() = {
  foreach (i from 0 to 8191) { gpr[i] = 0x00000000; srs[i] = 0x00000000 };
  foreach (i from 0 to 255) { upr[i] = 0x00000000 };
  em = 0xFFFFFFFF;
  pc = 0;
  status = ST_loading;
  exit_code = 0;
  msg = "";
  pend = [||];
  stack_words = 0;
  ncode = 0;
  labels = [||];
  foreach (i from 0 to 4095) { code[i] = STOP(); code_addr[i] = 0 }
}

val fatal : string -> unit
function fatal(s) = {
  msg = s;
  exit_code = 1;
  status = ST_fault
}

/* bounds-checked access to the program vectors */
val get_code : int -> ast
function get_code(i) = { assert(0 <= i & i < 4096, "pc out of range"); code[i] }
val set_code : (int, ast) -> unit
function set_code(i, a) = { assert(0 <= i & i < 4096, "program too long"); code[i] = a }
val get_addr : int -> int
function get_addr(i) = { assert(0 <= i & i < 4096, "pc out of range"); code_addr[i] }
val set_addr : (int, int) -> unit
function set_addr(i, a) = { assert(0 <= i & i < 4096, "program too long"); code_addr[i] = a }
val get_upr : int -> bits(32)
function get_upr(n) = { assert(0 <= n & n < 256, "uniform register out of range"); upr[n] }
val set_upr : (int, bits(32)) -> unit
function set_upr(n, v) = { assert(0 <= n & n < 256, "uniform register out of range"); upr[n] = v }
val get_sr_v : int -> bits(32)
function get_sr_v(i) = if 0 <= i & i < 8192 then srs[i] else 0x00000000
val set_sr_v : (int, bits(32)) -> unit
function set_sr_v(i, v) = { assert(0 <= i & i < 8192, "special register out of range"); srs[i] = v }

/* =============================================================================
 * PTX machine state (semantics/ptx-configuration.k).
 *
 * The K configuration's maps are kept in string-keyed maps of the C runtime (one map "space" per
 * K map: a frame's register file, a thread's symbol environment, the module's symbol table, ...)
 * and the byte memories in its sparse byte maps; scalar cells are Sail registers, per-thread and
 * per-CTA cells are vectors indexed by thread / CTA slot.
 * ============================================================================= */

val smap_get = impure {c: "sail_smap_get"} : (int, string) -> int
val smap_has = impure {c: "sail_smap_has"} : (int, string) -> bool
val smap_set = impure {c: "sail_smap_set"} : (int, string, int) -> unit
val smap_del = impure {c: "sail_smap_del"} : (int, string) -> unit
val smap_clear = impure {c: "sail_smap_clear"} : int -> unit
val smap_size = impure {c: "sail_smap_size"} : int -> int
val smap_key = impure {c: "sail_smap_key"} : (int, int) -> string
val smap_copy = impure {c: "sail_smap_copy"} : (int, int) -> unit
val mem_rd = impure {c: "sail_mem_rd"} : (int, int) -> int
val mem_wr = impure {c: "sail_mem_wr"} : (int, int, int) -> unit
val mem_rdn = impure {c: "sail_mem_rdn"} : (int, int, int) -> int
val mem_wrn = impure {c: "sail_mem_wrn"} : (int, int, int, int) -> unit
val mem_copy = impure {c: "sail_mem_copy"} : (int, int, int, int, int) -> unit

val tok_str = impure {c: "sail_tok_str"} : unit -> string
val tok_int = impure {c: "sail_tok_int"} : unit -> int

/* ---------------------------------------------------------------- spaces */
register next_space : int = 100
val new_space : unit -> int
function new_space() = { let s = next_space; next_space = next_space + 1; s }

let SP_GMEM : int = 1      /* <gmem> */
let SP_SYMS : int = 2      /* <syms> */
let SP_FUNCS : int = 3     /* name |-> function number + 1 */
let SP_FADDRS : int = 4    /* function address |-> function number + 1 (keys: decimal) */
let SP_LINKD : int = 5     /* <linkd> */
let SP_CLDIMS : int = 6    /* <clusterDims>: kernel |-> x * 2^40 + y * 2^20 + z */
let SP_MBARS : int = 7     /* <mbars> */
let SP_MISC : int = 8      /* small global tables (tensormaps, multimem, fabric, ...) */
let SP_CLBAR : int = 9     /* <clbar> */
let SP_KPARAMS : int = 10  /* <kparams> (bytes) */
let SP_KENV : int = 11     /* <kenv> */
let SP_FADDRN : int = 12   /* function number + 1 |-> address */

/* symbols sym(space, address), encoded address * 16 + space code; noSym = absent */
val sp_code : string -> int
function sp_code(s) = match s {
  "global" => 0, "shared" => 1, "local" => 2, "param" => 3, "const" => 4, "func" => 5, "dynshared" => 6,
  "regin" => 7, "asz" => 8, "shared_cluster" => 9, "cluster" => 10, "nosym" => 15, _ => 11
}
val code_sp : int -> string
function code_sp(c) = match c {
  0 => "global", 1 => "shared", 2 => "local", 3 => "param", 4 => "const", 5 => "func", 6 => "dynshared",
  7 => "regin", 8 => "asz", 9 => "shared_cluster", 10 => "cluster", 15 => "nosym", _ => "other"
}
val mksym : (string, int) -> int
function mksym(s, a) = a * 16 + sp_code(s)
val sym_space : int -> string
function sym_space(e) = code_sp(fmod_int(e, 16))
val sym_addr : int -> int
function sym_addr(e) = fdiv_int(e, 16)

/* ---------------------------------------------------------------- module / host cells */
register exit_code : int = 1
register fails : int = 0
register gtop : int = 4096
register stop : int = 0
register turn : int = -2
register occ : int = 0
register nextcta : int = 0
register resident : int = 0
register its : bool = false
register ub : bool = false
register memmodel : string = "sc"
register isaver : int = 0
register isasm : int = 0
register isaarch : string = ""
register isachk : bool = false
register tick : int = 1000
register asz32 : bool = false
register nfaddr : int = 0
register stuck : bool = false
register stuck_msg : string = ""

/* buffers: name, address, element type, element count */
register bufs : list((string, int, ty, int)) = [||]
/* names of `.extern .shared` (dynamic) arrays, given addresses at launch */
register dyn_names : list(string) = [||]
/* performance directives per kernel */
register perfd : list((string, list((string, list(int))))) = [||]
/* .branchtargets / .calltargets lists: (function, label, targets) */
register tgts : list((int, string, list(string))) = [||]

/* ---------------------------------------------------------------- functions */
let MAXF : int = 4096
register nfuncs : int = 0
register f_name : vector(4096, string) = vector_init("")
register f_kind : vector(4096, string) = vector_init("")
register f_rets : vector(4096, list(pdecl)) = vector_init([||])
register f_params : vector(4096, list(pdecl)) = vector_init([||])
register f_base : vector(4096, int) = vector_init(0)
register f_len : vector(4096, int) = vector_init(0)
register f_rw : vector(4096, int) = vector_init(0)
register f_lab : vector(4096, int) = vector_init(0)

val fchk : int -> range(0, 4095)
function fchk(f) = { assert(0 <= f & f < 4096, "function table full"); f }

/* the code of all functions */
let MAXCODE : int = 262144
register ncode : int = 0
register code : vector(262144, item) = vector_init(DECLVAR("", "", 0, 0))
val get_item : int -> item
function get_item(i) = { assert(0 <= i & i < 262144, "code index"); code[i] }
val add_item : item -> unit
function add_item(it) = { let i = ncode; assert(0 <= i & i < 262144, "program too large"); code[i] = it; ncode = ncode + 1 }

/* ---------------------------------------------------------------- launch */
struct dim3 = { x : int, y : int, z : int }
val mkdim : (int, int, int) -> dim3
function mkdim(x, y, z) = struct { x = x, y = y, z = z }
val vol : dim3 -> int
function vol(d) = d.x * d.y * d.z

register kernel : string = ""
register kfid : int = 0
register gridDim : dim3 = struct { x = 1, y = 1, z = 1 }
register blockDim : dim3 = struct { x = 1, y = 1, z = 1 }
register clusterDim : dim3 = struct { x = 1, y = 1, z = 1 }
register kptop : int = 0

/* ---------------------------------------------------------------- CTAs (slot = ctaid) */
let MAXC : int = 1024
register ncta_slots : int = 0
register c_alive : vector(1024, bool) = vector_init(false)
register c_live : vector(1024, int) = vector_init(0)
register c_smem : vector(1024, int) = vector_init(0)
register c_bars : vector(1024, int) = vector_init(0)    /* barrier maps: "cnt:B" "exp:B" "gen:B" "val:B" "res:B" */
register c_colls : vector(1024, int) = vector_init(0)   /* collective instances */
register c_wlive : vector(1024, int) = vector_init(0)   /* warp |-> live lanes */
register c_misc : vector(1024, int) = vector_init(0)    /* tensor memory and other per-CTA maps */

val cchk : int -> range(0, 1023)
function cchk(c) = { assert(0 <= c & c < 1024, "too many CTAs"); c }

/* ---------------------------------------------------------------- threads (slot) */
let MAXT : int = 16384

/* blocked continuations of <tk> (#barWaitRed, #collWait, ...) */
union tcont = {
  K_none : unit,
  K_barwaitred : (int, int, xop),            /* barrier, generation, destination */
  K_collwait : (string, list(xop)),          /* instance key, destinations */
  K_wait : (string, int),                    /* generic wait on a condition of another module */
}

/* thread status: 0 running, 1 waiting(B, G) on a barrier, 2 exited */
register nthreads : int = 0
register t_alive : vector(16384, bool) = vector_init(false)
register t_cta : vector(16384, int) = vector_init(0)
register t_tid : vector(16384, int) = vector_init(0)
register t_status : vector(16384, int) = vector_init(0)
register t_wbar : vector(16384, int) = vector_init(0)
register t_wgen : vector(16384, int) = vector_init(0)
register t_fn : vector(16384, int) = vector_init(0)
register t_pc : vector(16384, int) = vector_init(0)
register t_regs : vector(16384, int) = vector_init(0)
register t_env : vector(16384, int) = vector_init(0)
register t_sregs : vector(16384, int) = vector_init(0)
register t_pmem : vector(16384, int) = vector_init(0)
register t_ptop : vector(16384, int) = vector_init(0)
register t_lmem : vector(16384, int) = vector_init(0)
register t_ltop : vector(16384, int) = vector_init(0)
register t_lsp : vector(16384, int) = vector_init(0)    /* the stack pointer of alloca / stacksave / stackrestore */
register t_cseq : vector(16384, int) = vector_init(0)
register t_misc : vector(16384, int) = vector_init(0)
register t_cont : vector(16384, tcont) = vector_init(K_none())
/* call stack: frame(function, return pc, registers, environment, local top, stack pointer, (formal, caller register) outs) */
register t_stack : vector(16384, list((int, int, int, int, int, int, list((string, string))))) = vector_init([||])

val tchk : int -> range(0, 16383)
function tchk(t) = { assert(0 <= t & t < 16384, "too many threads"); t }

/* the executing thread */
register cur : int = 0

val cur_cta : unit -> int
function cur_cta() = t_cta[tchk(cur)]
val cur_tid : unit -> int
function cur_tid() = t_tid[tchk(cur)]
val uid : (int, int) -> int
function uid(c, t) = c * 1048576 + t

/* ---------------------------------------------------------------- output */
/* run-time error: the thread is stuck (#fault / #unsupported) and the program cannot complete */
val fault : (string, int) -> unit
function fault(s, i) = {
  if not_bool(stuck) then {
    stuck = true;
    stuck_msg = concat_str(s, concat_str(" ", dec_str(i)))
  }
}

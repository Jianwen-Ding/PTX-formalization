/* =============================================================================
 * Instruction dispatch (semantics/ptx-exec.k): the opcode-specific `exec` rules of every module,
 * then the variable-opcode `#gx` rules, then #unsupported.  Rules with priority 40 in K are tried
 * first.
 * ============================================================================= */

val exec_prio : (opcode, list(xop)) -> bool
function exec_prio(o, ops) = exec_misc_prio(o, ops) | exec_memq_prio(o, ops)

val exec_specific : (opcode, list(xop)) -> bool
function exec_specific(o, ops) =
  exec_altfp(o, ops) | exec_mini(o, ops) | exec_half(o, ops) | exec_alu(o, ops) | exec_mem(o, ops) | exec_ctrl(o, ops) | exec_warp(o, ops) | exec_bits(o, ops)
  | exec_intx(o, ops) | exec_pred(o, ops) | exec_misc(o, ops) | exec_memq(o, ops) | exec_video(o, ops) | exec_ext(o, ops)

val exec_gx : (opcode, list(xop)) -> bool
function exec_gx(o, ops) =
  gx_alu(o, ops) | gx_int(o, ops) | gx_intx(o, ops) | gx_ctrl(o, ops) | gx_sync(o, ops) | gx_misc(o, ops) | gx_half(o, ops) | gx_memq(o, ops) | gx_video(o, ops) | gx_ext(o, ops)

/* "Single-precision instructions ... flush subnormal inputs and results to sign-preserving zero for sm_1x targets"
 * (PTX ISA 9.3 9.7.3): on an sm_1x target every .f32 instruction that takes .ftz behaves as if it had it */
let FTZ_OPS : list(string) = [|"add", "sub", "mul", "mad", "fma", "div", "rcp", "sqrt", "rsqrt", "sin", "cos", "lg2",
  "ex2", "min", "max", "abs", "neg", "setp", "set", "slct", "cvt"|]
val tysHasF32 : list(ty) -> bool
function tysHasF32(l) = match l { [||] => false, t :: r => (t.k == "f" & t.w == 32) | tysHasF32(r) }
val sm1xFtz : opcode -> opcode
function sm1xFtz(o) =
  if isasm > 0 & isasm < 20 & has(FTZ_OPS, o.base) & not_bool(has(o.mods, "ftz")) & tysHasF32(o.tys)
  then { o with mods = pend_append(o.mods, [|"ftz"|]) } else o

function exec_dispatch(o0, ops) = {
  let o = sm1xFtz(o0);
  wgCheck(o, ops);
  paramUb(o, ops);
  hintUb(o, ops);
  if stuck then ()
  else if exec_prio(o, ops) then ()
  else if exec_specific(o, ops) then ()
  else if exec_gx(o, ops) then ()
  else unsupported(o)
}

function isCollective(op) = isCollective_core(op) | isCollective_ext(op)

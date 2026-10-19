/* =============================================================================
 * Instruction dispatch (semantics/ptx-exec.k): the opcode-specific `exec` rules of every module,
 * then the variable-opcode `#gx` rules, then #unsupported.  Rules with priority 40 in K are tried
 * first.
 * ============================================================================= */

val exec_prio : (opcode, list(xop)) -> bool
function exec_prio(o, ops) = exec_misc_prio(o, ops)

val exec_specific : (opcode, list(xop)) -> bool
function exec_specific(o, ops) =
  exec_alu(o, ops) | exec_mem(o, ops) | exec_ctrl(o, ops) | exec_warp(o, ops) | exec_bits(o, ops)
  | exec_intx(o, ops) | exec_pred(o, ops) | exec_misc(o, ops) | exec_ext(o, ops)

val exec_gx : (opcode, list(xop)) -> bool
function exec_gx(o, ops) =
  gx_alu(o, ops) | gx_int(o, ops) | gx_intx(o, ops) | gx_ctrl(o, ops) | gx_sync(o, ops) | gx_misc(o, ops) | gx_ext(o, ops)

function exec_dispatch(o, ops) = {
  if exec_prio(o, ops) then ()
  else if exec_specific(o, ops) then ()
  else if exec_gx(o, ops) then ()
  else unsupported(o)
}

function isCollective(op) = isCollective_core(op) | isCollective_ext(op)

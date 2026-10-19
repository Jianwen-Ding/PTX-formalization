/* =============================================================================
 * The hooks between the core and the extension modules (mma, ...): each module contributes
 * instructions, collective computations and harness directives; the modules not modelled are
 * stubbed here.
 * ============================================================================= */
function exec_ext(o, ops) = exec_mma(o, ops) | exec_sp(o, ops)
function gx_ext(o, ops) = false
function isCollective_ext(op) = has([|"mma", "wmma"|], op)
function coll_compute_ext(k, d, arr) = coll_mma(k, d, arr)
function ext_wait_ready() = false
function harness_ext(d) = false
function isa_check() = ()
function cluster_mem(a, op) = None()
function cluster_exit() = ()

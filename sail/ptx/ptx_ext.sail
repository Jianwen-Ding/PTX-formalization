/* =============================================================================
 * The hooks between the core and the extension modules: each module contributes instructions,
 * collective computations, collective continuations, blocked waits and harness directives; the
 * hooks of the modules not modelled are stubs here.
 * ============================================================================= */
function exec_ext(o, ops) =
  exec_mma(o, ops) | exec_sp(o, ops) | exec_ldm(o, ops) | exec_cluster(o, ops) | exec_mbar(o, ops) | exec_async(o, ops) | exec_amisc(o, ops)
function gx_ext(o, ops) = false
function isCollective_ext(op) = has([|"mma", "wmma", "ldmatrix", "stmatrix", "movmatrix", "mbarrier", "cp", "clusterlaunchcontrol"|], op)
function coll_compute_ext(k, d, arr) = match coll_mma(k, d, arr) {
  Some(r) => Some(r),
  None() => coll_ldm(k, d, arr)
}
function coll_post_ext(p, vs) = post_ldm(p, vs)
function ext_wait_ready2() = false
function harness_ext(d) = false
function isa_check() = ()

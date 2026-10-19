/* =============================================================================
 * The hooks between the core and the extension modules: each module contributes instructions,
 * collective computations, collective continuations, blocked waits and harness directives; the
 * hooks of the modules not modelled are stubs here.
 * ============================================================================= */
function exec_ext(o, ops) =
  exec_mma(o, ops) | exec_sp(o, ops) | exec_ldm(o, ops) | exec_cluster(o, ops) | exec_mbar(o, ops) | exec_async(o, ops) | exec_amisc(o, ops) | exec_wg(o, ops) | exec_tma(o, ops) | exec_spc(o, ops) | exec_tc(o, ops)
function gx_ext(o, ops) = false
function isCollective_ext(op) = has([|"mma", "wmma", "ldmatrix", "stmatrix", "movmatrix", "mbarrier", "cp", "clusterlaunchcontrol", "wgmma", "tensormap", "tcgen05"|], op)
function coll_compute_ext(k, d, arr) = match coll_mma(k, d, arr) {
  Some(r) => Some(r),
  None() => if k == "wgmma" then Some(wg_compute(d, arr)) else coll_ldm(k, d, arr)
}
function coll_post_ext(p, vs) = match p {
  P_tcalloc(_, _) => post_tc(p, vs),
  P_tcdealloc(_, _) => post_tc(p, vs),
  _ => post_ldm(p, vs)
}
function ext_wait_ready2() = false
function harness_ext(d) = harness_tma(d)
function isa_check() = ()

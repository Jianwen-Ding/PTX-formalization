/* hooks of the modules that are not modelled yet */
function exec_ext(o, ops) = false
function gx_ext(o, ops) = false
function isCollective_ext(op) = false
function coll_compute_ext(k, d, arr) = None()
function ext_wait_ready() = false
function harness_ext(d) = false
function isa_check() = ()
function cluster_mem(a, op) = None()
function cluster_exit() = ()
function transcendental(op, w, a, raw) = { fault(concat_str("not modelled: ", op), 0); FV_NaN() }

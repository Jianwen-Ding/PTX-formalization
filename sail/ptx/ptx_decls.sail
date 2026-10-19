/* forward declarations of the hooks between modules */
val transcendental : (string, int, int, int) -> fv
val exec_ext : (opcode, list(xop)) -> bool
val gx_ext : (opcode, list(xop)) -> bool
val isCollective_ext : string -> bool
/* the extension directives of the harness (tensor maps, textures, samplers, surfaces, multimem, fabric) */
val harness_ext : string -> bool
/* #isaChk */
val isa_check : unit -> unit
/* harness helpers of ptx_main.sail used by the extension directives */
val do_alloc : (string, ty, int, list(initval)) -> unit
val tk_ty : unit -> ty

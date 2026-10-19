/* forward declarations of the hooks between modules */
val transcendental : (string, int, int, int) -> fv
val exec_ext : (opcode, list(xop)) -> bool
val gx_ext : (opcode, list(xop)) -> bool
val isCollective_ext : string -> bool
/* the extension directives of the harness (tensor maps, textures, samplers, surfaces, multimem, fabric) */
val harness_ext : string -> bool
/* #isaChk */
val isa_check : unit -> unit

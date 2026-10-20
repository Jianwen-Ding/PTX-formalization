/* =============================================================================
 * AGX (Apple G13 / G14) in Sail: types and the instruction AST.
 *
 * A re-expression of agx/agx.k.  The AST covers the K model's instructions; operands carry the
 * extra fields of the binary encodings (register-cache hints, source sign-extension and float
 * abs/neg modifiers, saturation, shifts, push counts, ...) so that `encdec` is a bijection on the
 * encodable subset.  With every extra field at its default the semantics is exactly the K model's.
 * ============================================================================= */

enum rfile = { GPR, UNI }

/* register-cache hint (applegpu: cache / discard flags) */
enum hint = { HNone, HCache, HDiscard }

/* A register reference.  view: 0 = 32-bit rN, 1 = low half rNl, 2 = high half rNh,
 * 3 = 64-bit pair rN_rN+1.  sx / fabs / fneg are the source modifiers of the encodings. */
struct areg = {
  file : rfile,
  num : int,
  view : int,
  hint : hint,
  sx : bool,
  fabs : bool,
  fneg : bool,
}

/* operands: registers, integer immediates, and K Float literals (kept as their binary64 bits) */
union aop = {
  OReg : areg,
  OImm : int,
  OFImm : int,
}

/* 64-bit base of a memory access: a register pair given by its low and high register (AUPair u2_u3) */
struct apair = {
  pfile : rfile,
  lo : int,
  hi : int,
}

/* comparison conditions: the K model's names, followed by the hardware-only conditions of the
 * encodings (IConditionDesc / FConditionDesc) that have no K name */
enum acc = {
  CC_eq, CC_ne, CC_ult, CC_ule, CC_ugt, CC_uge, CC_slt, CC_sle, CC_sgt, CC_sge,
  CC_feq, CC_fne, CC_flt, CC_fle, CC_fgt, CC_fge,
  CC_seq, CC_sne,                     /* signed equality (same result as eq / ne) */
  CC_fltn, CC_fgtn,                   /* "nan loses" less / greater than */
  CC_fnequ, CC_fnlt, CC_fngt, CC_fnle, CC_fnge, CC_fnltn, CC_fngtn   /* inverted float conditions */
}

enum afmt = { F_U8, F_U16, F_U32 }
/* atomic operations: the K model's ten, then the device-atomic forms measured on G14C (Apple-M2-Playground
 * 2026-10-17-atomic-variants): 32-bit compare-and-swap (op 3), float32 add (op 1 with bit 45), 64-bit unsigned
 * min / max (op 3 with bit 45, bit 46 clear / set), and ops 11-15, which end the SIMD group */
enum aaop = { A_add, A_sub, A_xchg, A_umin, A_imin, A_umax, A_imax, A_and, A_or, A_xor,
              A_cmpxchg, A_fadd, A_umin64, A_umax64, A_end11, A_end12, A_end13, A_end14, A_end15 }
enum iunop = { U_popcount, U_bitrev, U_ffs }
enum fbinop = { FB_add, FB_mul }
enum stkkind = { SK_if, SK_else, SK_while }
enum shufkind = { SH_idx, SH_xor, SH_up, SH_down }

/* branch targets: a K label (assembly) or a byte offset relative to the jump (binary) */
union jtarget = {
  JLabel : string,
  JRel : int,
}

union ast = {
  /* mov D, A */
  MOV : (areg, aop),
  /* get_sr D, SR */
  GET_SR : (areg, int),
  /* iadd / isub: (subtract, saturate, D, A, B, shift) */
  IADD : (bool, bool, areg, aop, aop, int),
  /* imadd / imsub: (subtract, saturate, D, A, B, C, shift) */
  IMADD : (bool, bool, areg, aop, aop, aop, int),
  /* bitop TT D, A, B */
  BITOP : (int, areg, aop, aop),
  /* bfi M D, A, B, C / bfeil M D, A, B, C */
  BFI : (int, areg, aop, aop, aop),
  BFEIL : (int, areg, aop, aop, aop),
  ASR : (areg, aop, aop),
  IUNOP : (iunop, areg, aop),
  /* fadd / fmul (op, 16-bit form, saturate, D, A, B) */
  FBIN : (fbinop, bool, bool, areg, aop, aop),
  /* fmadd (16-bit form, saturate, D, A, B, C) */
  FMADD : (bool, bool, areg, aop, aop, aop),
  /* rcp (saturate, D, A) */
  RCP : (bool, areg, aop),
  /* icmpsel / fcmpsel (float, cc, D, A, B, X, Y) */
  CMPSEL : (bool, acc, areg, aop, aop, aop, aop),
  /* if / else / while compares on the execution-mask stack (kind, float, cc, A, B, push count n) */
  STACKOP : (stkkind, bool, acc, aop, aop, int),
  POP_EXEC : int,
  /* jmp_exec_any (true) / jmp_exec_none (false) */
  JMP_EXEC : (bool, jtarget),
  /* icmp_ballot / fcmp_ballot (float, cc, D, A, B) */
  BALLOT : (bool, acc, areg, aop, aop),
  SHUFFLE : (shufkind, areg, aop, aop),
  /* device_load / device_store (store, format, element mask, R, base, index, signed index, shift,
   * wait group, store-order bit u6) */
  DEVMEM : (bool, afmt, int, areg, apair, aop, bool, int, int, int),
  /* threadgroup_load / threadgroup_store (store, format, element mask, R, base, index, shift) */
  TGMEM : (bool, afmt, int, areg, aop, aop, int),
  /* device_atomic (op, D, base, index, signed index, value, wait group, mode).  The value is a 32-bit register,
   * or a register pair for cmpxchg (new value, compare value) and umin64 / umax64 (lo, hi).  mode is the LLVM
   * backend's operand for bits 45-46: bit 0 = bit 45, bit 1 = bit 46 clear; 0 is Mesa's G14 encoding.  Where the
   * op does not select a meaning with them, the bits have no effect on G14C (bit 45 is Mesa's multi-die
   * coherency bit). */
  DEVATOM : (aaop, option(areg), apair, aop, bool, aop, int, int),
  /* threadgroup_atomic (op, D, base, index, value) */
  TGATOM : (aaop, option(areg), aop, aop, aop),
  WAIT : int,
  TG_BARRIER : unit,
  /* memory_barrier a, b, c (applegpu MemoryBarrierInstructionDesc) */
  MEMORY_BARRIER : (int, int, int),
  STOP : unit,
  TRAP : unit,
}

/* constructors for plain registers */
val gpr32 : int -> areg
function gpr32(n) = struct { file = GPR, num = n, view = 0, hint = HNone, sx = false, fabs = false, fneg = false }

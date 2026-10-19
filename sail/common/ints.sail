/* Arbitrary-precision integer helpers shared by the AGX and PTX models.
 * The K semantics computes on unbounded K Ints with &Int, |Int, xorInt, <<Int, >>Int; these
 * functions give the same (two's-complement, infinite precision) results on Sail ints. */

val bitlen : int -> int
/* number of bits of |n| (0 for 0) */
function bitlen(n) = {
  var m : int = abs_int(n);
  var k : int = 0;
  while m >= 4294967296 do { m = shr_int(m, 32); k = k + 32 };
  while m > 0 do { m = shr_int(m, 1); k = k + 1 };
  k
}

val bw_width : (int, int) -> int
function bw_width(a, b) = max_int(bitlen(a), bitlen(b)) + 2

val land_int : (int, int) -> int
function land_int(a, b) = {
  if a >= 0 & b >= 0 & a < 4294967296 & b < 4294967296 then
    unsigned(get_slice_int(32, a, 0) & get_slice_int(32, b, 0))
  else {
    let 'w = bw_width(a, b);
    assert('w > 0);
    signed(get_slice_int('w, a, 0) & get_slice_int('w, b, 0))
  }
}

val lor_int : (int, int) -> int
function lor_int(a, b) = {
  if a >= 0 & b >= 0 & a < 4294967296 & b < 4294967296 then
    unsigned(get_slice_int(32, a, 0) | get_slice_int(32, b, 0))
  else {
    let 'w = bw_width(a, b);
    assert('w > 0);
    signed(get_slice_int('w, a, 0) | get_slice_int('w, b, 0))
  }
}

val lxor_int : (int, int) -> int
function lxor_int(a, b) = {
  if a >= 0 & b >= 0 & a < 4294967296 & b < 4294967296 then
    unsigned(xor_vec(get_slice_int(32, a, 0), get_slice_int(32, b, 0)))
  else {
    let 'w = bw_width(a, b);
    assert('w > 0);
    signed(xor_vec(get_slice_int('w, a, 0), get_slice_int('w, b, 0)))
  }
}

/* ~a */
val lnot_int : int -> int
function lnot_int(a) = negate(a) - 1

/* (1 << w) - 1, 0 for w < 0 */
val mask_int : int -> int
function mask_int(w) = if w >= 0 then pow2(w) - 1 else 0

/* a & mask(w): the low w bits (unsigned) */
val trunc_int : (int, int) -> int
function trunc_int(a, w) = if w <= 0 then 0 else fmod_int(a, pow2(w))

/* signed view of the low w bits */
val sext_int : (int, int) -> int
function sext_int(a, w) = {
  let t = trunc_int(a, w);
  if w > 0 & t >= pow2(w - 1) then t - pow2(w) else t
}

/* bit i of a (two's complement) */
val bit_int : (int, int) -> bool
function bit_int(a, i) = fmod_int(shr_int(a, i), 2) == 1

/* K's >>Int: arithmetic shift right (floor) */
val asr_int : (int, int) -> int
function asr_int(a, n) = if n <= 0 then shl_int(a, negate(n)) else fdiv_int(a, pow2(n))

val lsl_int : (int, int) -> int
function lsl_int(a, n) = if n >= 0 then a * pow2(n) else fdiv_int(a, pow2(negate(n)))

val popc_int : int -> int
function popc_int(v) = {
  var m : int = v;
  var c : int = 0;
  while m > 0 do { if fmod_int(m, 2) == 1 then c = c + 1; m = shr_int(m, 1) };
  c
}

val b2i : bool -> int
function b2i(b) = if b then 1 else 0

val log2_int : int -> int
/* floor(log2 n) for n > 0 (K's log2Int) */
function log2_int(n) = bitlen(n) - 1

/* b^k for k >= 0 */
val pow : (int, int) -> int
function pow(b, k) = { var r : int = 1; var i : int = 0; while i < k do { r = r * b; i = i + 1 }; r }

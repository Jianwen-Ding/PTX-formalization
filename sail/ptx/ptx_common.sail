/* =============================================================================
 * PTX-COMMON (semantics/ptx-common.k): types, bit vectors, literals, formatting.
 * The IEEE-754 encode / decode and arithmetic are in sail/common/ieee.sail.
 * ============================================================================= */

val wid : ty -> int
function wid(t) = t.w
val bytes : ty -> int
function bytes(t) = max_int(tdiv_int(t.w, 8), 1)
val isF : ty -> bool
function isF(t) = t.k == "f" | t.k == "bf" | t.k == "tf"
val isS : ty -> bool
function isS(t) = t.k == "s"
/* format code of a float type (16 32 64; 116 bfloat16; 132 tf32) */
val fw : ty -> int
function fw(t) = if t.k == "bf" then 116 else if t.k == "tf" then 132 else t.w
val wide : ty -> ty
function wide(t) = mkty(t.k, 2 * t.w)

val mask : int -> int
function mask(w) = mask_int(w)
val trunc : (int, int) -> int
function trunc(v, w) = trunc_int(v, w)
val sext : (int, int) -> int
function sext(v, w) = sext_int(v, w)
val ival : (int, ty) -> int
function ival(v, t) = if isS(t) then sext(v, wid(t)) else trunc(v, wid(t))
val minOf : ty -> int
function minOf(t) = if isS(t) then negate(pow2(wid(t) - 1)) else 0
val maxOf : ty -> int
function maxOf(t) = if isS(t) then pow2(wid(t) - 1) - 1 else mask(wid(t))
val clamp : (int, ty) -> int
function clamp(v, t) = min_int(max_int(v, minOf(t)), maxOf(t))

val brev : (int, int) -> int
function brev(v, w) = {
  var r : int = 0;
  var x : int = v;
  foreach (i from 1 to w) { r = 2 * r + fmod_int(x, 2); x = fdiv_int(x, 2) };
  r
}
val clz : (int, int) -> int
function clz(v, w) = if v > 0 then w - (log2_int(v) + 1) else w

/* ---------------------------------------------------------------- floats as values (K Float) */
val fv : (int, ty) -> fv
function fv(v, t) = decodeF(v, fw(t))

/* the value rounded to the precision of format w (roundFloat(F, prec(w), ebits(w))) */
val roundW : (fv, int) -> fv
function roundW(f, w) = if w == 132 then round_fv(f, 11, 8, RN) else round_fv(f, prec(w), ebits(w), RN)

/* round to an integral value: "rni" "rzi" "rmi" "rpi" */
val rndInt : (fv, string) -> fv
function rndInt(f, m) = match f {
  FV_Fin(s, q, e) => {
    if e >= 0 | q == 0 then f
    else {
      let d = pow2(negate(e));
      let ip = tdiv_int(q, d);
      let fr = q - ip * d;
      let up : bool = match m {
        "rzi" => false,
        "rmi" => s & fr != 0,
        "rpi" => not_bool(s) & fr != 0,
        _ => 2 * fr > d | (2 * fr == d & fmod_int(ip, 2) == 1)
      };
      FV_Fin(s, if up then ip + 1 else ip, 0)
    }
  },
  _ => f
}

/* Float2Int of an integral value (truncation of a non-integral one) */
val f2int : fv -> int
function f2int(f) = match f {
  FV_Fin(s, q, e) => { let n = if e >= 0 then q * pow2(e) else tdiv_int(q, pow2(negate(e))); if s then negate(n) else n },
  _ => 0
}

/* f2i(F, mode, T): float -> integer with saturation; NaN -> 0 if the source is not .f64 and the destination is not
 * .s64 / .u64, otherwise 1 << (BitWidth(dst) - 1) (ISA 9.7.9.22, Integer Notes) */
val f2i : (fv, string, ty, bool) -> int
function f2i(f, m, t, src64) = match f {
  FV_NaN() => if src64 | wid(t) == 64 then pow2(wid(t) - 1) else 0,
  FV_Inf(s) => trunc(if s then minOf(t) else maxOf(t), wid(t)),
  _ => trunc(clamp(f2int(rndInt(f, m)), t), wid(t))
}

val fv_of_int : int -> fv
function fv_of_int(i) = if i < 0 then FV_Fin(true, negate(i), 0) else FV_Fin(false, i, 0)

/* ---------------------------------------------------------------- literals */
val hex_digit : string -> int
function hex_digit(c) = match c {
  "0" => 0, "1" => 1, "2" => 2, "3" => 3, "4" => 4, "5" => 5, "6" => 6, "7" => 7, "8" => 8, "9" => 9,
  "a" => 10, "b" => 11, "c" => 12, "d" => 13, "e" => 14, "f" => 15,
  "A" => 10, "B" => 11, "C" => 12, "D" => 13, "E" => 14, "F" => 15, _ => 0
}

val str2base : (string, int) -> int
function str2base(s, b) = {
  var r : int = 0;
  var t = s;
  while string_length(t) > 0 do { r = r * b + hex_digit(stake(t, 1)); t = sdrop(t, 1) };
  r
}

/* hexBits(H, T): 0x / 0f / 0d / 0b literals (a trailing u / U is ignored) */
val hexBits : (string, ty) -> int
function hexBits(h0, t) = {
  let n = string_length(h0);
  let last = sdrop(h0, n - 1);
  let h = if last == "u" | last == "U" then stake(h0, n - 1) else h0;
  let p = stake(sdrop(h, 1), 1);
  let d = sdrop(h, 2);
  if (p == "f" | p == "F") & isF(t) & wid(t) != 32 then encodeF(decodeF(str2base(d, 16), 32), fw(t))
  else if (p == "d" | p == "D") & isF(t) & wid(t) != 64 then encodeF(decodeF(str2base(d, 16), 64), fw(t))
  else if p == "b" | p == "B" then trunc(str2base(d, 2), wid(t))
  else trunc(str2base(d, 16), wid(t))
}

/* intLit / fltLit: an integer / float literal at a type */
val intLit : (int, ty) -> int
function intLit(i, t) = if isF(t) then encodeF(roundW(fv_of_int(i), fw(t)), fw(t)) else trunc(i, wid(t))
val fltLit : (int, ty) -> int
function fltLit(f, t) = if isF(t) then encodeF(f64v(f), fw(t)) else trunc(f2int(rndInt(f64v(f), "rzi")), wid(t))

/* ---------------------------------------------------------------- formatting (fmtVal / fmtFloat) */
val zeros : int -> string
function zeros(n) = { var s : string = ""; var i : int = 0; while i < n do { s = concat_str(s, "0"); i = i + 1 }; s }

val stripZeros : string -> string
function stripZeros(s) = {
  var r = s;
  while string_length(r) > 1 & sdrop(r, string_length(r) - 1) == "0" do r = stake(r, string_length(r) - 1);
  r
}

/* mpfr_get_str(base 10, n = 0, RNDN) of a positive value: n = 1 + ceil(p log10 2) digits and the
 * exponent E with value = 0.D1D2... * 10^E (the digits of K's Float2String) */
val ndigits : int -> int
function ndigits(p) = {
  /* smallest k with 10^k >= 2^p, i.e. ceil(p log10 2), then + 1 */
  var k : int = 0;
  while pow(10, k) < pow2(p) do k = k + 1;
  /* exact power of two is impossible for p >= 1, so ceil = k */
  k + 1
}

val dec_digits : (int, int, int) -> (string, int)
function dec_digits(m, e, p) = {
  /* value = m * 2^e, m > 0 */
  let n = ndigits(p);
  /* numerator / denominator of the value */
  let num = if e >= 0 then m * pow2(e) else m;
  let den = if e >= 0 then 1 else pow2(negate(e));
  /* E: 10^(E-1) <= v < 10^E */
  var ex : int = 0;
  if num >= den then { while num >= den * pow(10, ex) do ex = ex + 1 }
  else { while num * pow(10, negate(ex) + 1) < den do ex = ex - 1 };
  /* digits = round(v * 10^(n - E)) */
  let sh = n - ex;
  let nn = if sh >= 0 then num * pow(10, sh) else num;
  let dd = if sh >= 0 then den else den * pow(10, negate(sh));
  var q : int = tdiv_int(nn, dd);
  let r = nn - q * dd;
  if 2 * r > dd | (2 * r == dd & fmod_int(q, 2) == 1) then q = q + 1;
  if q >= pow(10, n) then { q = tdiv_int(q, 10); ex = ex + 1 };
  (dec_str(q), ex)
}

val fmtPos : (string, int) -> string
function fmtPos(d, e) = {
  let l = string_length(d);
  if e <= 0 & e > -5 then concat_str("0.", concat_str(zeros(negate(e)), d))
  else if e > 0 & e <= 9 & l <= e then concat_str(d, concat_str(zeros(e - l), ".0"))
  else if e > 0 & e <= 9 & l > e then concat_str(stake(d, e), concat_str(".", sdrop(d, e)))
  else concat_str(stake(d, 1), concat_str(".", concat_str(if l > 1 then sdrop(d, 1) else "0", concat_str("e", dec_str(e - 1)))))
}

/* fmtFloat of a value of precision p */
val fmtFloat : (fv, int) -> string
function fmtFloat(f, p) = match f {
  FV_NaN() => "nan",
  FV_Inf(s) => if s then "-inf" else "inf",
  FV_Fin(s, m, e) => {
    if m == 0 then (if s then "-0.0" else "0.0")
    else {
      let (d, ex) = dec_digits(m, e, p);
      concat_str(if s then "-" else "", fmtPos(stripZeros(d), ex))
    }
  }
}

val fmtVal : (int, ty) -> string
function fmtVal(v, t) =
  if isF(t) then fmtFloat(fv(v, t), if fw(t) == 132 then 11 else prec(fw(t))) else dec_str(ival(v, t))

/* spaceOf(M): the state space of a memory instruction */
val spaceOf : list(string) -> string
function spaceOf(m) =
  if has(m, "shared::cluster") then "shared_cluster"
  else pick(m, [|"global", "shared", "local", "param", "const"|], "generic")

/* nth(L, I): I-th element of an integer list (1 when absent) */
val nth : (list(int), int) -> int
function nth(l, i) = match l { [||] => 1, h :: t => if i == 0 then h else nth(t, i - 1) }
val nthD : (list(int), int, int) -> int
function nthD(l, i, d) = match l { [||] => d, h :: t => if i == 0 then h else nthD(t, i - 1, d) }
val int_list_len : list(int) -> int
function int_list_len(l) = match l { [||] => 0, _ :: t => 1 + int_list_len(t) }

/* ---------------------------------------------------------------- address map (ptx-configuration.k)
 * Generic addressing (PTX ISA 6.4.1.1): .shared, .local and .const are windows of the generic address space,
 * and so is the shared::cluster window (16 MiB per CTA rank of the cluster); a generic address outside them
 * maps to .global.  The window of the kernel parameters (.param) is contained in the .global window.  Each
 * window maps to its state space by subtracting the window base.  Under `.address_size 32` generic addresses
 * have 32 bits and the windows are 256 MiB each from 2^31 on; otherwise they are 2^40 .. 2^45. */
let DYNSHARED : int = 16
/* the stack top of the per-thread local memory: .local variables grow up from 0, alloca grows down from here */
let LSTACK : int = 16777216

val winBase : string -> int
function winBase(s) = {
  let i : int = match s { "shared" => 0, "local" => 1, "cluster" => 2, "param" => 3, "const" => 4, "end" => 5, _ => -1 };
  if i < 0 then 0
  else if asz32 then 2147483648 + 268435456 * i
  else pow2(40 + i)
}

/* the window of a generic address */
val genWin : int -> string
function genWin(a) =
  if a < winBase("shared") | a >= winBase("end") then "global"
  else if a >= winBase("const") then "const"
  else if a >= winBase("param") then "param"
  else if a >= winBase("cluster") then "cluster"
  else if a >= winBase("local") then "local"
  else "shared"

val alignUp : (int, int) -> int
function alignUp(x, a) = if a > 0 then tdiv_int(x + a - 1, a) * a else x

val is_dec_digit : string -> bool
function is_dec_digit(c) = c == "0" | c == "1" | c == "2" | c == "3" | c == "4" | c == "5" | c == "6" | c == "7" | c == "8" | c == "9"

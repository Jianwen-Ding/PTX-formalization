/* =============================================================================
 * PTX-EXEC-ISA (semantics/ptx-exec-isa.k): the optional legality check (`.isacheck ;`) of every
 * instruction against the declared .version / .target, with the rows of ptx_isa_table.sail.
 * ============================================================================= */

val tyName : ty -> string
function tyName(t) =
  if t.k == "p" then "pred"
  else if t.k == "bf" & t.w == 16 then "bf16"
  else if t.k == "tf" & t.w == 32 then "tf32"
  else if string_length(t.k) > 1 & t.k != "bf" & t.k != "tf" then t.k
  else concat_str(t.k, dec_str(t.w))

/* names an instruction carries: its modifiers (`a::b` also as `a`) and its type names */
/* index of the first occurrence of p in s, or -1 */
val str_find : (string, string) -> int
function str_find(s, p) = {
  var i : int = 0;
  var r : int = -1;
  while r < 0 & i + string_length(p) <= string_length(s) do { if stake(sdrop(s, i), string_length(p)) == p then r = i; i = i + 1 };
  r
}

val insNames : (list(string), list(ty)) -> list(string)
function insNames(m, ts) = {
  var out : list(string) = [||];
  var x = m;
  while is_cons(x) do {
    match x { s :: rest => {
      out = s :: out;
      let i = str_find(s, "::");
      if i >= 0 then out = stake(s, i) :: out;
      x = rest
    }, [||] => () }
  };
  var y = ts;
  while is_cons(y) do { match y { t :: rest => { out = tyName(t) :: out; y = rest }, [||] => () } };
  out
}

val qualsHold : (string, list(string)) -> bool
function qualsHold(q, n) = {
  if q == "" then true
  else {
    let i = str_index(q, "+");
    if i < 0 then has(n, q) else has(n, stake(q, i)) & qualsHold(sdrop(q, i + 1), n)
  }
}

val isaBad : (list((string, int, int, bool)), list(string), int, int, string) -> string
function isaBad(rows, n, ver, sm, arch) = match rows {
  [||] => "",
  (q, v, s, a) :: rest => {
    if qualsHold(q, n) & ver > 0 & v > ver then
      concat_str("instruction needs PTX ISA ", concat_str(dec_str(tdiv_int(v, 10)), concat_str(".", dec_str(tmod_int(v, 10)))))
    else if qualsHold(q, n) & sm > 0 & s > sm then concat_str("instruction needs target sm_", dec_str(s))
    else if qualsHold(q, n) & a & sm > 0 & arch == "" then
      concat_str("instruction needs an architecture-specific target (sm_", concat_str(dec_str(s), "a)"))
    else isaBad(rest, n, ver, sm, arch)
  }
}

val isaRemoved : (string, list(string), list(string), int, int) -> string
function isaRemoved(op, m, n, ver, sm) =
  if (op == "shfl" | op == "vote") & not_bool(has(m, "sync")) & sm >= 70 & ver >= 64 then
    concat_str(op, " without .sync was removed for sm_70 and higher (PTX ISA 6.4)")
  else if op == "wmma" & has(m, "mma") & has(m, "satfinite") & (has(n, "f16") | has(n, "f32")) & ver >= 65 then
    concat_str(op, ".satfinite on floating-point wmma.mma was removed in PTX ISA 6.5")
  else if (op == "atom" | op == "red") & (has(m, "min") | has(m, "max")) & has(n, "f32") then
    concat_str(op, ".min/.max on f32 does not exist (removed from the specification)")
  else if op == "suld" & has(m, "p") then "suld.p does not exist (removed from the specification)"
  else if op == "sust" & has(m, "p") & (has(n, "u32") | has(n, "s32") | has(n, "f32")) then
    "sust.p with a 32-bit type does not exist (removed from the specification)"
  else ""

val isaIns : item -> string
function isaIns(it) = match it {
  INS(_, o, _) => {
    let n = insNames(o.mods, o.tys);
    let r = isaRemoved(o.base, o.mods, n, isaver, isasm);
    if r != "" then r else isaBad(isaRows(o.base), n, isaver, isasm, isaarch)
  },
  _ => ""
}

/* #isaChk: the first offending instruction (functions in definition order, then program order) */
function isa_check() = {
  if isachk then {
    var msg : string = "";
    var f : int = 0;
    while f < nfuncs & msg == "" do {
      let fs = fchk(f);
      var pc : int = 0;
      while pc < f_len[fs] & msg == "" do { msg = isaIns(get_item(f_base[fs] + pc)); pc = pc + 1 };
      f = f + 1
    };
    if msg != "" then fault(msg, 0)
  }
}

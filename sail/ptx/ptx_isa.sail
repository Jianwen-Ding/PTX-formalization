/* =============================================================================
 * The optional legality check (`.isacheck ;`) of a program against its .version / .target, following
 * the PTX ISA 9.3 specification: the module directives (11.1.1 .version, 11.1.2 .target) and the
 * "PTX ISA Notes" / "Target ISA Notes" of every 9.7 instruction section, as rows generated from the
 * spec text by gen_isa_table.py into ptx_isa_table.sail (row format: see gen_isa_table.py).
 *
 * Targets (11.1.2, Table 67):
 *   sm_N      baseline features of sm_N and of every lower target (onion model);
 *   sm_Na     adds the architecture-specific features of exactly sm_N;
 *   sm_Nf     adds the family-specific features of sm_M, M <= N, in the same family (sm_10x = 100, 103;
 *             sm_11x = 110, 101; sm_12x = 120, 121); "can be used with f-targets as well as a-targets of
 *             later generation devices in the same family", so sm_Na has them too.
 *   sm_101{,f,a} were renamed sm_110{,f,a} from PTX ISA 9.0: both spellings denote the same target.
 * ============================================================================= */

val tyName : ty -> string
function tyName(t) =
  if t.k == "p" then "pred"
  else if t.k == "bf" & t.w == 16 then "bf16"
  else if t.k == "tf" & t.w == 32 then "tf32"
  else if string_length(t.k) > 1 & t.k != "bf" & t.k != "tf" then t.k
  else concat_str(t.k, dec_str(t.w))

/* index of the first occurrence of p in s, or -1 */
val str_find : (string, string) -> int
function str_find(s, p) = {
  var i : int = 0;
  var r : int = -1;
  while r < 0 & i + string_length(p) <= string_length(s) do { if stake(sdrop(s, i), string_length(p)) == p then r = i; i = i + 1 };
  r
}

/* s split at every occurrence of the separator c (a non-empty string) */
val isaSplit : (string, string) -> list(string)
function isaSplit(s, c) = {
  /* recursive, without list-valued `var`s (list variables updated in loops lost elements in the C output) */
  let i = str_find(s, c);
  if i < 0 then [|s|]
  else {
    let h : string = stake(s, i);
    let t : string = sdrop(s, i + string_length(c));
    h :: isaSplit(t, c)
  }
}

val isaTyNames : list(ty) -> list(string)
function isaTyNames(ts) = match ts { [||] => [||], t :: rest => tyName(t) :: isaTyNames(rest) }

val isaAny : (list(string), list(string)) -> bool
function isaAny(names, alts) = match alts { [||] => false, a :: rest => has(names, a) | isaAny(names, rest) }

val isaAll : (list(string), list(string)) -> bool
function isaAll(names, alts) = match alts { [||] => true, a :: rest => has(names, a) & isaAll(names, rest) }

val tys_app : (list(string), list(string)) -> list(string)
function tys_app(a, b) = match a { [||] => b, h :: t => h :: tys_app(t, b) }

/* "x::y" also counts as "x" (the front end records both for the first "::") */
val isaPre : (string, string) -> bool
function isaPre(m, a) = m == a | (string_length(m) > string_length(a) & stake(m, string_length(a) + 2) == concat_str(a, "::"))

val isaFirst : (list(string), list(string)) -> bool
function isaFirst(mods, alts) = match (mods, alts) {
  ([||], _) => false,
  (m :: _, [||]) => false,
  (m :: _, a :: rest) => isaPre(m, a) | isaFirst(mods, rest)
}

val isaEnds : (string, list(string)) -> bool
function isaEnds(m, alts) = match alts {
  [||] => false,
  a :: t => (string_length(m) > string_length(a) & sdrop(m, string_length(m) - string_length(a)) == a) | isaEnds(m, t)
}

val isaSuffix : (list(string), list(string)) -> bool
function isaSuffix(mods, alts) = match mods { [||] => false, m :: rest => isaEnds(m, alts) | isaSuffix(rest, alts) }

/* the elements of groups (each a "/"-list) occur in names in this order */
val isaSeq : (list(string), list(string)) -> bool
function isaSeq(names, groups) = match groups {
  [||] => true,
  g :: grest => match names {
    [||] => false,
    n :: nrest => if has(isaSplit(g, "/"), n) then isaSeq(nrest, grest) else isaSeq(nrest, groups)
  }
}

val isaOpCount : list(xop) -> int
function isaOpCount(ops) = match ops { [||] => 0, _ :: t => 1 + isaOpCount(t) }

/* operand shapes: d|p destination, an indirect texture/surface (a register in [ ]), a register operand,
 * the sink `_` destination, a [ ] second operand */
val isaDestPred : list(xop) -> bool
function isaDestPred(ops) = match ops { XVP(_, _, _) :: _ => true, XS(SPIPE(_, _, _, _)) :: _ => true, _ => false }

val isaIndirect : list(xop) -> bool
function isaIndirect(ops) = match ops {
  [||] => false,
  XB(AR(_) :: _, _) :: _ => true,
  XB(ARO(_, _) :: _, _) :: _ => true,
  XB(_, _) :: _ => false,
  _ :: t => isaIndirect(t)
}

val isaRegOp : list(xop) -> bool
function isaRegOp(ops) = match ops { [||] => false, XS(SR(_)) :: _ => true, _ :: t => isaRegOp(t) }

/* the second operand is a [ ] address (mbarrier.test_wait/try_wait without the reportValue operand) */
val isaSecondMem : list(xop) -> bool
function isaSecondMem(ops) = match ops { _ :: XB(_, _) :: _ => true, _ => false }

val isaSink : list(xop) -> bool
function isaSink(ops) = match ops { XS(SN(n)) :: _ => n == "_", _ => false }

/* one term of a row pattern */
val isaTerm : (string, list(string), list(string), list(string), list(xop)) -> bool
function isaTerm(t, mods, tys, names, ops) = {
  let neg = stake(t, 1) == "!";
  let u = if neg then sdrop(t, 1) else t;
  let k = stake(u, 2);
  let v = sdrop(u, 2);
  let r : bool =
    if k == "m=" then isaAny(names, isaSplit(v, "/"))
    else if k == "p=" then isaFirst(mods, isaSplit(v, "/"))
    else if k == "x=" then not_bool(isaAll(names, isaSplit(v, "+")))
    else if k == "t=" then isaSeq(tys, isaSplit(v, ","))
    else if k == "q=" then isaSeq(mods, isaSplit(v, ","))
    else if k == "d=" then (match tys { h :: _ => has(isaSplit(v, "/"), h), [||] => false })
    else if k == "n=" then list_length(tys) == str_int(v)
    else if k == "o=" then isaOpCount(ops) == str_int(v)
    else if k == "o>" then isaOpCount(ops) > str_int(v)
    else if k == "s=" then isaSuffix(mods, isaSplit(v, "/"))
    else if u == "dp" then isaDestPred(ops)
    else if u == "ir" then isaIndirect(ops)
    else if u == "rr" then isaRegOp(ops)
    else if u == "sk" then isaSink(ops)
    else if u == "m2" then isaSecondMem(ops)
    else false;
  if neg then not_bool(r) else r
}

val isaPat : (list(string), list(string), list(string), list(string), list(xop)) -> bool
function isaPat(terms, mods, tys, names, ops) = match terms {
  [||] => true,
  t :: rest => (t == "" | isaTerm(t, mods, tys, names, ops)) & isaPat(rest, mods, tys, names, ops)
}

/* ---------------------------------------------------------------- targets */
/* sm_101 was renamed sm_110 (11.1.2) */
val isaNorm : int -> int
function isaNorm(n) = if n == 101 then 110 else n

val isaFam : int -> int
function isaFam(n) = tdiv_int(isaNorm(n), 10)

/* the target term T of an alternative: "", "N", "Na", "Nf", "<N" */
val isaTgtOk : (string, int, string) -> bool
function isaTgtOk(t, sm, arch) = {
  let l = string_length(t);
  if t == "" then true
  else if stake(t, 1) == "<" then isaNorm(sm) < isaNorm(str_int(sdrop(t, 1)))
  else if sdrop(t, l - 1) == "a" then arch == "a" & isaNorm(sm) == isaNorm(str_int(stake(t, l - 1)))
  else if sdrop(t, l - 1) == "f" then {
    let n = str_int(stake(t, l - 1));
    (arch == "a" | arch == "f") & isaFam(sm) == isaFam(n) & isaNorm(sm) >= isaNorm(n)
  }
  else isaNorm(sm) >= isaNorm(str_int(t))
}

/* "V:T" -> (V, T); V < 0: the form is valid only below PTX ISA -V */
val isaAltVer : string -> int
function isaAltVer(a) = {
  let s = stake(a, str_find(a, ":"));
  if stake(s, 1) == "-" then 0 - str_int(sdrop(s, 1)) else str_int(s)
}

val isaAltTgt : string -> string
function isaAltTgt(a) = sdrop(a, str_find(a, ":") + 1)

val isaAltOk : (string, int, int, string) -> bool
function isaAltOk(a, ver, sm, arch) = {
  let v = isaAltVer(a);
  (if v < 0 then ver < 0 - v else ver >= v) & isaTgtOk(isaAltTgt(a), sm, arch)
}

val isaVerStr : int -> string
function isaVerStr(v) = concat_str(dec_str(tdiv_int(v, 10)), concat_str(".", dec_str(tmod_int(v, 10))))

val isaTgtDesc : string -> string
function isaTgtDesc(t) = {
  let l = string_length(t);
  if stake(t, 1) == "<" then concat_str("below sm_", sdrop(t, 1))
  else if sdrop(t, l - 1) == "f" then concat_str("sm_", concat_str(t, " (or higher in the same family)"))
  else concat_str("sm_", t)
}

/* the message of a failed row (recursive helpers over the alternatives) */
val isaNegVer : list(string) -> int
function isaNegVer(al) = match al { [||] => 0, a :: t => if isaAltVer(a) < 0 then 0 - isaAltVer(a) else isaNegVer(t) }

val isaLow : list(string) -> string
function isaLow(al) = match al {
  [||] => "",
  a :: t => if stake(isaAltTgt(a), 1) == "<" then sdrop(isaAltTgt(a), 1) else isaLow(t)
}

/* the least version of an alternative whose target holds; -1 if none */
val isaVNeed : (list(string), int, string) -> int
function isaVNeed(al, sm, arch) = match al {
  [||] => -1,
  a :: t => {
    let r = isaVNeed(t, sm, arch);
    let v = isaAltVer(a);
    let tg = isaAltTgt(a);
    if v >= 0 & stake(tg, 1) != "<" & isaTgtOk(tg, sm, arch) & (r < 0 | v < r) then v else r
  }
}

val isaIsSpec : string -> bool
function isaIsSpec(tg) = {
  let l = string_length(tg);
  l > 0 & (sdrop(tg, l - 1) == "a" | sdrop(tg, l - 1) == "f")
}

/* the targets of the alternatives, described; whether one is a/f; the least plain one */
val isaTgtList : list(string) -> string
function isaTgtList(al) = match al {
  [||] => "",
  a :: t => {
    let tg = isaAltTgt(a);
    let r = isaTgtList(t);
    if tg == "" | stake(tg, 1) == "<" then r
    else if r == "" then isaTgtDesc(tg)
    else concat_str(isaTgtDesc(tg), concat_str(", ", r))
  }
}

val isaAnySpec : list(string) -> bool
function isaAnySpec(al) = match al { [||] => false, a :: t => isaIsSpec(isaAltTgt(a)) | isaAnySpec(t) }

val isaPlainMin : list(string) -> int
function isaPlainMin(al) = match al {
  [||] => 0,
  a :: t => {
    let tg = isaAltTgt(a);
    let r = isaPlainMin(t);
    if tg == "" | stake(tg, 1) == "<" | isaIsSpec(tg) then r
    else if r == 0 | str_int(tg) < r then str_int(tg) else r
  }
}

val isaMsg : (list(string), int, int, string, string) -> string
function isaMsg(alts, ver, sm, arch, sec) = {
  let negv = isaNegVer(alts);
  let low = isaLow(alts);
  let vneed = isaVNeed(alts, sm, arch);
  let m : string =
    if negv > 0 then concat_str("instruction form is not supported from PTX ISA ", concat_str(isaVerStr(negv),
         if low != "" then concat_str(" on sm_", concat_str(low, " or higher")) else ""))
    else if vneed >= 0 then concat_str("instruction needs PTX ISA ", isaVerStr(vneed))
    else if isaAnySpec(alts) then concat_str("instruction needs an architecture-specific or family-specific target: ", isaTgtList(alts))
    else concat_str("instruction needs target sm_", dec_str(isaPlainMin(alts)));
  concat_str(m, concat_str(" [PTX ISA 9.3 ", concat_str(sec, "]")))
}

val isaAltsOk : (list(string), int, int, string) -> bool
function isaAltsOk(al, ver, sm, arch) = match al { [||] => false, a :: t => isaAltOk(a, ver, sm, arch) | isaAltsOk(t, ver, sm, arch) }

val isaBad : (list((string, string, string)), list(string), list(string), list(string), list(xop), int, int, string) -> string
function isaBad(rows, mods, tys, names, ops, ver, sm, arch) = match rows {
  [||] => "",
  (pat, alts, sec) :: rest => {
    let al = isaSplit(alts, "|");
    if isaPat(isaSplit(pat, " "), mods, tys, names, ops) & not_bool(isaAltsOk(al, ver, sm, arch))
    then isaMsg(al, ver, sm, arch, sec)
    else isaBad(rest, mods, tys, names, ops, ver, sm, arch)
  }
}

/* syntax that is not in the specification any more (13: "Unimplemented instructions {atom,red}.{min,max}.f32
 * have been removed", "Unimplemented instructions suld.p and sust.p.{u32,s32,f32} have been removed") */
val isaRemoved : (string, list(string), list(string)) -> string
function isaRemoved(op, m, n) =
  if (op == "atom" | op == "red") & (has(m, "min") | has(m, "max")) & has(n, "f32") then
    concat_str(op, ".min/.max on f32 does not exist (removed from the specification)")
  else if op == "suld" & has(m, "p") then "suld.p does not exist (removed from the specification)"
  else if op == "sust" & has(m, "p") & (has(n, "u32") | has(n, "s32") | has(n, "f32")) then
    "sust.p with a 32-bit type does not exist (removed from the specification)"
  else ""

val isaIns : item -> string
function isaIns(it) = match it {
  INS(_, o, ops) => {
    let tys = isaTyNames(o.tys);
    let names = tys_app(o.mods, tys);
    let r0 = isaRemoved(o.base, o.mods, names);
    let r = if r0 != "" then r0 else isaTables(o.base, o.mods, o.tys);
    if r != "" then r else isaBad(isaRows(o.base), o.mods, tys, names, ops, isaver, isasm, isaarch)
  },
  _ => ""
}

/* 11.1.1 / 11.1.2: the module begins with .version and .target; the target string exists in that version */
val isaModule : unit -> string
function isaModule() = {
  let t = concat_str("sm_", concat_str(dec_str(isasm), isaarch));
  let tv = isaTargetVer(t);
  if isaver <= 0 then "a PTX module must begin with a .version directive [PTX ISA 9.3 11.1.1]"
  else if isasm <= 0 then "a PTX module must have a .target directive with an sm_ target architecture [PTX ISA 9.3 11.1.2]"
  else if tv < 0 then concat_str("unknown target architecture ", concat_str(t, " [PTX ISA 9.3 11.1.2]"))
  else if tv > isaver then concat_str("target ", concat_str(t, concat_str(" needs PTX ISA ", concat_str(isaVerStr(tv), " [PTX ISA 9.3 11.1.2]"))))
  else ""
}

/* #isaChk: the module directives, then the first offending instruction (functions in definition order,
 * then program order) */
function isa_check() = {
  if isachk then {
    var msg : string = isaModule();
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

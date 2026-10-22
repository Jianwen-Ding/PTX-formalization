/* =============================================================================
 * The optional legality check (`.isacheck ;`) of a program against its .version / .target, following
 * the PTX ISA 9.4 specification: the module directives (11.1.1 .version, 11.1.2 .target) and the
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
  concat_str(m, concat_str(" [PTX ISA 9.4 ", concat_str(sec, "]")))
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

/* qualifier and operand rules that the ISA states in prose (strict mode only: the default mode gives these forms their
   evident meaning) */
val isaImm : xop -> option(int)
function isaImm(o) = match o { XS(SI(i)) => Some(i), XS(SH(h)) => Some(hexBits(h, mkty("u", 64))), _ => None() }

val isaCount : (list(string), list(string)) -> int
function isaCount(m, l) = match l { [||] => 0, a :: r => (if has(m, a) then 1 else 0) + isaCount(m, r) }

val isaPow2 : int -> bool
function isaPow2(n) = n > 0 & land_int(n, n - 1) == 0

/* operand i of n is an immediate outside lo .. hi or not a multiple of k */
val isaBadImm : (list(xop), int, int, int, int, int) -> bool
function isaBadImm(ops, n, i, lo, hi, k) =
  if isaOpCount(ops) != n then false
  else match isaImm(xop_nth(ops, i)) { Some(v) => v < lo | v > hi | fmod_int(v, k) != 0, None() => false }

val isaBadAlign : list(xop) -> bool
function isaBadAlign(ops) =
  if isaOpCount(ops) != 3 then false
  else match isaImm(xop_nth(ops, 2)) { Some(v) => not_bool(isaPow2(v)) | v > 8388608, None() => false }

val isaSyncAligned : (string, list(string)) -> bool
function isaSyncAligned(b, m) =
  isaAny([|b|], [|"ldmatrix", "stmatrix", "movmatrix", "mma", "wmma", "wgmma", "setmaxnreg", "elect"|])
  | (b == "tensormap" & has(m, "cp_fenceproxy"))
  | (b == "tcgen05" & isaAny(m, [|"alloc", "dealloc", "relinquish_alloc_permit", "ld", "st", "wait::ld", "wait::st"|]))

/* the .collector::X::op qualifiers of the buffers xs */
val isaColl : list(string) -> list(string)
function isaColl(xs) = match xs {
  [||] => [||],
  x :: r => {
    let p = concat_str("collector::", concat_str(x, "::"));
    concat_str(p, "fill") :: concat_str(p, "use") :: concat_str(p, "lastuse") :: concat_str(p, "discard") :: isaColl(r)
  }
}

/* a .report_mechanism other than .mbarrier::report::disabled */
val isaRepOn : list(string) -> bool
function isaRepOn(m) = { let (w, _) = repPat(m); w != 0 }

val isaPzoOk : list(ty) -> bool
function isaPzoOk(ts) = match ts {
  [|d, s|] =>
    (s.k == "f" & s.w == 32 & (has([|"f16x2", "bf16x2", "e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2", "e4m3x4", "e5m2x4",
                                     "e2m3x4", "e3m2x4", "e2m1x4"|], d.k) | (d.k == "f" & d.w == 16) | d.k == "bf" | d.k == "tf"))
    | ((s.k == "f16x2" | s.k == "bf16x2") & has([|"e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2"|], d.k)),
  _ => false
}

val isaQual : (opcode, list(xop)) -> string
function isaQual(o, ops) = {
  let m = o.mods;
  let b = o.base;
  let sem = if b == "ld" then "acquire" else "release";
  let cache = isaAny(m, [|"ca", "cg", "cs", "lu", "cv", "wb", "wt"|]);
  let scope = isaAny(m, [|"cta", "cluster", "gpu", "sys"|]);
  let fl : bool = match o.tys { t :: _ => t.k == "f", _ => false };
  let w : int = match o.tys { t :: _ => t.w, _ => 0 };
  let hk : string = match o.tys { [|t|] => if (t.k == "f" & t.w == 16) | t.k == "f16x2" then "f16" else if t.k == "bf" | t.k == "bf16x2" then "bf16" else "", _ => "" };
  let hArith = hk != "" & isaAny([|b|], [|"add", "sub", "mul", "fma"|]);
  let dk : string = match o.tys { t :: _ => t.k, _ => "" };
  if b == "cvt" & isaAny([|dk|], [|"e4m3x2", "e5m2x2", "e2m3x2", "e3m2x2", "e2m1x2", "s2f6x2"|])
     & isaAny(m, [|"rm", "rp", "rna", "rs"|]) then
    concat_str("cvt to .", concat_str(dk, " rounds only with .rn or .rz (.frnd2) [PTX ISA 9.4 9.7.10.24]"))
  else if b == "cvt" & isaAny([|dk|], [|"e4m3x4", "e5m2x4", "e2m3x4", "e3m2x4", "e2m1x4"|])
     & isaAny(m, [|"rn", "rz", "rm", "rp", "rna"|]) then
    concat_str("cvt to .", concat_str(dk, " rounds only with .rs [PTX ISA 9.4 9.7.10.24]"))
  /* "The .pzo qualifier is only supported for float-to-float conversions where the destination type is strictly
     smaller than the source type" (9.4 9.7.10.24): the forms of the syntax, .f32 to f16 / bf16 / tf32 / pairs / x2 / x4
     and .f16x2 / .bf16x2 to the f8 / f6 / f4 x2 types */
  else if b == "cvt" & has(m, "pzo") & not_bool(isaPzoOk(o.tys)) then
    "cvt: .pzo is only supported for float-to-float conversions to a strictly smaller type [PTX ISA 9.4 9.7.10.24]"
  else if b == "cvt" & dk == "ue8m0x2" & isaAny(m, [|"rn", "rm", "rna", "rs"|]) then
    "cvt to .ue8m0x2 rounds with .rz or .rp (.frnd3) [PTX ISA 9.4 9.7.10.24]"
  else if b == "cvt" & dk == "ue5m3x2" & (isaAny(m, [|"rm", "rna", "rs"|]) | (has(m, "scaled::n1::ue8m0") & has(m, "rp"))) then
    "cvt to .ue5m3x2 rounds with .rn, .rz or .rp, and with .rn or .rz when scaled (.frnd4 / .frnd2) [PTX ISA 9.4 9.7.10.24]"
  else if hArith & isaAny(m, [|"rz", "rm", "rp"|]) then
    concat_str(b, " on .f16 / .bf16 values rounds only with .rn [PTX ISA 9.4 9.7.4]")
  else if hk == "bf16" & isaAny([|b|], [|"add", "sub", "mul", "fma", "min", "max"|]) & isaAny(m, [|"ftz", "sat"|]) then
    concat_str(b, ": no .ftz / .sat form on .bf16 / .bf16x2 [PTX ISA 9.4 9.7.4]")
  else if hk != "" & b == "fma" & has(m, "relu") & has(m, "sat") then
    "fma: .relu and .sat are exclusive [PTX ISA 9.4 9.7.4.4]"
  else if hk != "" & b == "fma" & has(m, "oob") & isaAny(m, [|"ftz", "sat"|]) then
    "fma.oob takes no .ftz / .sat [PTX ISA 9.4 9.7.4.4]"
  else if hk != "" & b == "ex2" & (if hk == "bf16" then not_bool(has(m, "ftz")) else has(m, "ftz")) then
    "ex2.approx: .ftz is required on .bf16 / .bf16x2 and not allowed on .f16 / .f16x2 [PTX ISA 9.4 9.7.4.10]"
  else if (b == "ld" | b == "st") & has(m, "v8") & not_bool(w == 32 & isaAny([|match o.tys { t :: _ => t.k, _ => "" }|], [|"b", "s", "u", "f"|])) then
    concat_str(b, ".v8 takes only .b32 / .s32 / .u32 / .f32 [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")
  else if (b == "ld" | b == "st") & (has(m, "v8") | (has(m, "v4") & w == 64))
          & isaAny(m, [|"shared", "local", "param", "const", "shared::cta", "shared::cluster"|]) then
    concat_str(b, ": .v8, and .v4 with a 64-bit type, need the .global state space or generic addressing [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")
  /* "The mandatory .sync qualifier ..." / "The mandatory .aligned qualifier ..." (ldmatrix, stmatrix, movmatrix, mma,
     wmma (.aligned from PTX ISA 6.3), wgmma, setmaxnreg, tensormap.cp_fenceproxy, tcgen05 alloc / dealloc /
     relinquish_alloc_permit / ld / st / wait; elect: .sync) */
  else if isaSyncAligned(b, m) & not_bool(has(m, "sync") & (has(m, "aligned") | (b == "wmma" & isaver < 63) | b == "elect")) then
    concat_str(b, ": the mandatory .sync / .aligned qualifiers are missing [PTX ISA 9.4 9.7]")
  else if b == "mbarrier" & isaAny(m, [|"relaxed", "release", "acquire"|]) != isaAny(m, [|"cta", "cluster", "gpu", "sys"|]) then
    "mbarrier: qualifiers .sem and .scope must be specified together [PTX ISA 9.4 9.7.15.16]"
  else if b == "mbarrier" & isaAny(m, [|"gpu", "sys"|]) then
    "mbarrier: .scope is .cta or .cluster [PTX ISA 9.4 9.7.15.16]"
  else if b == "mbarrier" & isaAny(m, [|"arrive", "arrive_drop"|]) & has(m, "acquire") then
    "mbarrier.arrive: .sem is .release or .relaxed [PTX ISA 9.4 9.7.15.16.16]"
  else if b == "mbarrier" & isaAny(m, [|"test_wait", "try_wait"|]) & has(m, "release") then
    "mbarrier.test_wait / try_wait: .sem is .acquire or .relaxed [PTX ISA 9.4 9.7.15.16.19]"
  else if b == "mbarrier" & isaAny(m, [|"expect_tx", "complete_tx"|]) & not_bool(isaAny(m, [|"arrive", "arrive_drop"|]))
          & isaAny(m, [|"release", "acquire"|]) then
    "mbarrier.expect_tx / complete_tx: .sem is .relaxed [PTX ISA 9.4 9.7.15.16.14 / .15]"
  else if b == "mbarrier" & isaAny(m, [|"test_wait", "try_wait"|]) & has(m, "shared::cluster") then
    "mbarrier.test_wait / try_wait: the state space is .shared::cta [PTX ISA 9.4 9.7.15.16.19]"
  else if b == "mbarrier" & has(m, "noComplete") & isaAny(m, [|"relaxed", "acquire", "cluster"|]) then
    "mbarrier.arrive.noComplete: only .release.cta [PTX ISA 9.4 9.7.15.16.16]"
  else if b == "mbarrier" & has(m, "check_layout") & not_bool(isaAny(m, [|"layout::v0", "layout::v1"|])) then
    "mbarrier.check_layout: a .layout qualifier is required [PTX ISA 9.4 9.7.15.16.21]"
  else if b == "mbarrier" & has(m, "init") & isaBadImm(ops, 2, 1, 1, if has(m, "layout::v1") then 511 else 1048575, 1) then
    "mbarrier.init: count outside the range of the layout [PTX ISA 9.4 9.7.15.16.12]"
  /* "So the valid values of phaseParity operand are 0 and 1" (9.3); 9.4 drops the sentence: "phaseParity, whose
     integer parity denotes ..." (9.4 9.7.15.16.19), so only modules below PTX ISA 9.4 are held to 0 / 1 */
  else if b == "mbarrier" & isaver < 94 & isaAny(m, [|"test_wait", "try_wait"|]) & has(m, "parity")
          & (isaBadImm(ops, 3, 2, 0, 1, 1) | isaBadImm(ops, 4, 2, 0, 1, 1) | isaBadImm(ops, 4, 3, 0, 1, 1) | isaBadImm(ops, 5, 3, 0, 1, 1)) then
    "mbarrier.test_wait / try_wait.parity: the phase parity is 0 or 1 [PTX ISA 9.3 9.7.14.16.19]"
  else if b == "tensormap" & has(m, "replace") & isaBadImm(ops, 3, 1, 0, 4, 1) then
    "tensormap.replace: the only valid values for ord are [0..4] [PTX ISA 9.4 9.7.10.29]"
  /* "In presence of .ignore_oob, the qualifier .report_mechanism, if specified, must be set to
     .mbarrier::report::disabled" (9.4 9.7.10.28.4.1) */
  else if b == "cp" & has(m, "ignore_oob") & isaRepOn(m) then
    "cp.async.bulk: with .ignore_oob the report mechanism must be .mbarrier::report::disabled [PTX ISA 9.4 9.7.10.28.4.1]"
  /* fabric.try_pullred: "Operand imm-membermask is an integer constant and must be 0xFFFFFFFF" (9.4 9.7.11.5.4) */
  else if b == "fabric" & has(m, "try_pullred") & isaBadImm(ops, 5, 4, 4294967295, 4294967295, 1) then
    "fabric.try_pullred: imm-membermask must be 0xFFFFFFFF [PTX ISA 9.4 9.7.11.5.4]"
  /* applypriority.async.bulk: "The value of the size operand needs to be a multiple of 16" (9.4 9.7.10.18) */
  else if b == "applypriority" & has(m, "bulk") & isaBadImm(ops, 2, 1, 16, 4294967295, 16) then
    "applypriority.async.bulk: size must be a multiple of 16 [PTX ISA 9.4 9.7.10.18]"
  else if (b == "discard" | (b == "applypriority" & not_bool(has(m, "bulk")))) & isaBadImm(ops, 2, 1, 128, 128, 1) then
    concat_str(b, if b == "discard" then ": the only supported size is 128 [PTX ISA 9.4 9.7.10.20]"
                  else ": the only supported size is 128 [PTX ISA 9.4 9.7.10.17]")
  else if (b == "ld" | b == "st") & isaCount(m, [|"weak", "volatile", "relaxed", sem|]) > 1 then
    concat_str(b, ": the .weak, .volatile, .relaxed and .acquire / .release qualifiers are mutually exclusive [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")
  else if (b == "ld" & has(m, "release")) | (b == "st" & has(m, "acquire")) then
    concat_str(b, concat_str(": no .", concat_str(if b == "ld" then "release" else "acquire", " form [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")))
  else if (b == "ld" | b == "st") & (has(m, "relaxed") | has(m, sem) | has(m, "volatile")) & cache then
    concat_str(b, ": cache operations are not allowed with .relaxed, .acquire / .release or .volatile [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")
  else if (b == "ld" | b == "st") & (has(m, "relaxed") | has(m, sem)) & not_bool(scope) then
    concat_str(b, ": .relaxed and .acquire / .release need a .scope [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")
  else if (b == "ld" | b == "st") & has(m, "mmio") & not_bool(has(m, "sys") & has(m, "relaxed")) then
    concat_str(b, ": only .relaxed.sys is valid for .mmio [PTX ISA 9.4 9.7.10.8 / 9.7.10.11]")
  else if b == "fma" & fl & not_bool(isaAny(m, [|"rn", "rz", "rm", "rp"|])) then
    "fma needs a rounding modifier [PTX ISA 9.4 9.7.3.6]"
  else if b == "mma" & has(m, "m8n8k4") & (match o.tys { [|dt, _, _, ct|] => ty_eq(ct, mkty("f", 32)) & not_bool(ty_eq(dt, mkty("f", 32))), _ => false }) then
    "mma.m8n8k4: .dtype must be .f32 when .ctype is .f32 [PTX ISA 9.4 9.7.16.5.14]"
  else if b == "setmaxnreg" & isaBadImm(ops, 1, 0, 24, 256, 8) then
    "setmaxnreg: imm-reg-count must be 24 .. 256 and a multiple of 8 [PTX ISA 9.4 9.7.21.5]"
  else if b == "pmevent" & not_bool(has(m, "mask")) & isaBadImm(ops, 1, 0, 0, 15, 1) then
    "pmevent: the event number must be 0 .. 15 [PTX ISA 9.4 9.7.21.3]"
  else if b == "prefetch" & has(m, "L1::32B") & (not_bool(has(m, "valid_addr")) | has(m, "local")) then
    "prefetch.L1::32B: .valid_addr must be specified and the state space, if specified, must be .global [PTX ISA 9.4 9.7.10.16]"
  else if b == "alloca" & isaBadAlign(ops) then
    "alloca: immAlign must be a power of 2 not above 2^23 [PTX ISA 9.4 9.7.19.3]"
  /* tcgen05.mma collector qualifiers (9.4 syntax of 9.7.18.10.10.1-3): .collector::a / ::b for mma{.sp}, ::b only in
     the .kind::ti16 integer forms, ::bN (N = 0..3) for .ws; "It is illegal to specify either of .collector::a::use or
     .collector::a::fill along with .ashift" */
  else if b == "tcgen05" & has(m, "mma") & has(m, "ashift") & isaAny(m, [|"collector::a::use", "collector::a::fill"|]) then
    "tcgen05.mma: .collector::a::use and .collector::a::fill are illegal with .ashift [PTX ISA 9.4 9.7.18.10.10.1]"
  else if b == "tcgen05" & has(m, "mma") & has(m, "ws") & isaAny(m, isaColl([|"a", "b"|])) then
    "tcgen05.mma.ws: the collector qualifier is .collector::bN::op, N = 0..3 [PTX ISA 9.4 9.7.18.10.10.3]"
  else if b == "tcgen05" & has(m, "mma") & not_bool(has(m, "ws")) & isaAny(m, isaColl([|"b0", "b1", "b2", "b3"|])) then
    "tcgen05.mma: .collector::bN::op is a .ws qualifier; the other forms take .collector::a / .collector::b [PTX ISA 9.4 9.7.18.10.10.1]"
  else if b == "tcgen05" & has(m, "mma") & has(m, "kind::i8") & isaAny(m, isaColl([|"b"|])) then
    "tcgen05.mma.kind::i8 has no .collector::b::op form (the integer forms with it are .kind::ti16) [PTX ISA 9.4 9.7.18.10.10.1]"
  /* "Decompression of pre-compressed B matrix" (9.7.18.10.10.1 form 7): tcgen05.mma (no .sp / .ws form) with
     .kind::f8f6f4, or .kind::mxf8f6f4.block_scale with .scale_vectorsize = { .block32 } */
  else if b == "tcgen05" & has(m, "mma") & has(m, "decompress::lut::b")
          & (has(m, "sp") | has(m, "ws") | not_bool(has(m, "kind::f8f6f4") | (has(m, "kind::mxf8f6f4") & has(m, "block_scale")))
             | isaAny(m, [|"scale_vec::1X", "scale_vec::2X", "scale_vec::4X", "block16"|])) then
    "tcgen05.mma.decompress::lut::b: only tcgen05.mma.kind::f8f6f4 and tcgen05.mma.kind::mxf8f6f4.block_scale{.block32} [PTX ISA 9.4 9.7.18.10.10.1]"
  else ""
}

val isaIns : item -> string
function isaIns(it) = match it {
  INS(_, o, ops) => {
    let tys = isaTyNames(o.tys);
    let names = tys_app(o.mods, tys);
    let r0 = isaRemoved(o.base, o.mods, names);
    let q = if r0 != "" then r0 else isaQual(o, ops);
    let r = if q != "" then q else isaTables(o.base, o.mods, o.tys);
    if r != "" then r else isaBad(isaRows(o.base), o.mods, tys, names, ops, isaver, isasm, isaarch)
  },
  _ => ""
}

/* 11.1.1 / 11.1.2: the module begins with .version and .target; the target string exists in that version */
val isaModule : unit -> string
function isaModule() = {
  let t = concat_str("sm_", concat_str(dec_str(isasm), isaarch));
  let tv = isaTargetVer(t);
  if isaver <= 0 then "a PTX module must begin with a .version directive [PTX ISA 9.4 11.1.1]"
  else if isasm <= 0 then "a PTX module must have a .target directive with an sm_ target architecture [PTX ISA 9.4 11.1.2]"
  else if tv < 0 then concat_str("unknown target architecture ", concat_str(t, " [PTX ISA 9.4 11.1.2]"))
  else if tv > isaver then concat_str("target ", concat_str(t, concat_str(" needs PTX ISA ", concat_str(isaVerStr(tv), " [PTX ISA 9.4 11.1.2]"))))
  else ""
}

/* an operand of the instruction is %perctamemoryoffset or %perctamemorysize */
val isaPcmOps : list(xop) -> bool
function isaPcmOps(l) = match l {
  [||] => false,
  XS(s) :: rest => has([|"%perctamemoryoffset", "%perctamemorysize"|], sop_name(s)) | isaPcmOps(rest),
  _ :: rest => isaPcmOps(rest)
}

/* per-CTA global memory (PTX ISA 9.4 10.34, 10.35, 11.4.10): the directive "is only allowed on .entry functions";
 * "Only entry functions specifying this directive are allowed to use special registers %perctamemoryoffset or
 * %perctamemorysize"; both introduced in PTX ISA 9.4 and requiring sm_80 */
val isaPcm : (int, item) -> string
function isaPcm(f, it) = {
  let pd = hasPerfN(perf_of(f_name[fchk(f)]), "minperctamemory");
  match it {
    INS(_, _, ops) =>
      if not_bool(isaPcmOps(ops)) then ""
      else if f_kind[fchk(f)] != "entry" | not_bool(pd) then
        "%perctamemoryoffset / %perctamemorysize are valid only in an entry function with .minperctamemory [PTX ISA 9.4 10.34]"
      else if isaver < 94 then "%perctamemoryoffset / %perctamemorysize need PTX ISA 9.4 [PTX ISA 9.4 10.34]"
      else if isaNorm(isasm) < 80 then "%perctamemoryoffset / %perctamemorysize need target sm_80 [PTX ISA 9.4 10.34]"
      else "",
    _ => ""
  }
}

/* #isaChk: the module directives, then the first offending instruction (functions in definition order,
 * then program order) */
function isa_check() = {
  if isachk then {
    var msg : string = isaModule();
    var f : int = 0;
    while f < nfuncs & msg == "" do {
      let fs = fchk(f);
      if hasPerfN(perf_of(f_name[fs]), "minperctamemory") then {
        msg = if f_kind[fs] != "entry" then ".minperctamemory is only allowed on .entry functions [PTX ISA 9.4 11.4.10]"
              else if isaver < 94 then ".minperctamemory needs PTX ISA 9.4 [PTX ISA 9.4 11.4.10]"
              else if isaNorm(isasm) < 80 then ".minperctamemory needs target sm_80 [PTX ISA 9.4 11.4.10]"
              else ""
      };
      var pc : int = 0;
      while pc < f_len[fs] & msg == "" do {
        let it = get_item(f_base[fs] + pc);
        msg = isaIns(it);
        if msg == "" then msg = isaPcm(fs, it);
        pc = pc + 1
      };
      f = f + 1
    };
    if msg != "" then fault(msg, 0)
  }
}

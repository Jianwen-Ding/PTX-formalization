#!/usr/bin/env python3
"""Generate ptx_isa_table.sail (the `.isacheck ;` legality rows) from the text of the PTX ISA
specification, Release 9.3 (`pdftotext -layout ptx_isa_9.3.pdf isa.txt`).

  gen_isa_table.py ISA_TXT [OUT.sail] [--report]

The spec is not in the repository; the generated ptx_isa_table.sail is committed so that builds do
not need it (sail/Makefile: `make -C sail isa-table ISA_TXT=/path/to/isa.txt`).

Sources of the rows
  * every 9.7.* instruction section's "PTX ISA Notes" / "Target ISA Notes", split into statements;
    the statements of a few fixed forms are parsed automatically:
      Introduced in PTX ISA version X.Y.                     (whole section)
      Requires sm_N or higher. / Requires sm_Na.             (whole section)
      Supported on all target architectures.
      <patterns> introduced in PTX ISA version X.Y.
      <patterns> require(s) sm_N or higher.
      <patterns> is/are supported on following ... architectures:  + bullets
        (sm_Na / sm_Nf or higher in the same family / sm_N or higher,
         "And is supported on following family-specific architectures from PTX ISA version X.Y:" + sub-bullets)
    where <patterns> are instruction strings of the spec (`cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2}.f32`)
    or qualifier lists (`.rs rounding mode`);
  * OVERRIDES: statements in prose that are not parsed reliably, translated by hand (keyed by the section
    and a distinctive substring of the statement; the isa.txt line is cited per entry);
  * GUARDS: the instructions a section is about when several sections share an opcode (add in 9.7.1.1,
    9.7.3.3, 9.7.4.1, 9.7.5.1, ...), written by hand with the line of the section's syntax cited;
  * the "Target string ... introduced in PTX ISA version" lines of 11.1.2 (.target).
Every statement that is neither parsed, overridden nor explicitly skipped is reported (--report) and
makes the generator fail, so a new spec release cannot be half-translated silently.

Row encoding (read by ptx_isa.sail):  isaRows(op) = [|(pattern, alternatives, section)|]
  pattern: space-separated terms, all must hold
     m=a/b        some modifier of the instruction is a or b ("x::y" also counts as "x")
     !m=a/b       no modifier is a or b
     x=a+b        not all of a, b are modifiers
     p=a/b        the first modifier is a or b ("x::y" also counts as "x")
     t=a/b,c/d    the types contain a or b, and after it c or d (ordered subsequence)
     !t=a/b       no type is a or b
     d=a/b        the first (destination) type is a or b
     n=K          exactly K types
     q=a,b        the modifiers contain a and after it b
     s=::a/::b    some modifier ends with ::a or ::b (sub-qualifiers)
     o=K / o>K    exactly / more than K operands
     dp           the destination is d|p        ir   the [ ] operand starts with a %register (indirect)
     rr           some operand is a %register   sk   the first operand is the sink `_`
     m2           the second operand is a [ ] operand          (any term can be negated with !)
  alternatives: "|"-separated  VER:TGT ; the row holds when one alternative holds:
     VER  minimum .version x10 (0: any); -V: the form is valid only below PTX ISA V
          ("required from PTX ISA 1.4", "not supported from PTX ISA 6.4 for sm_70 or higher")
     TGT  ""   any target
          N    sm_N or higher (onion model; also its a/f variants)
          Na   exactly sm_Na (architecture-specific)
          Nf   sm_Mf or sm_Ma with M >= N in the same family (Table 67)
          <N   below sm_N
"""
import os, re, sys

TY = {"u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64", "b8", "b16", "b32", "b64", "b128", "f16", "f32", "f64",
      "pred", "bf16", "tf32", "f16x2", "bf16x2", "s16x2", "u16x2", "u8x4", "s8x4", "f32x2", "ue5m3x2", "s2f6x2",
      "e4m3", "e5m2", "e4m3x2", "e5m2x2", "e4m3x4", "e5m2x4", "e2m3x2", "e3m2x2", "e2m3x4", "e3m2x4", "e2m1x2",
      "e2m1x4", "ue8m0x2", "e3m2", "e2m3", "e2m1", "ue8m0", "ue4m3", "u4", "s4", "u2", "s2", "b1"}

FURNITURE = [re.compile(r"^\s*\d+\s+Chapter \d+\. .*$"), re.compile(r"^\s*9\.7\. Instructions\s+\d+\s*$"),
             re.compile(r"^\s*PTX ISA, Release 9\.3\s*$"), re.compile(r"^\s*[0-9.]+ .{3,80}\s{3,}\d+\s*$")]


def load(path):
    return [l.replace("\f", "") for l in open(path, encoding="utf8", errors="replace").read().split("\n")]


def sections(L):
    """[(number, opcode-title, first line index, end line index)] of the 9.7 instruction sections"""
    hdr = re.compile(r"^(9\.7\.[0-9.]*[0-9])\s+(.+?):\s*(.*)$")
    out = []
    for i, l in enumerate(L):
        m = hdr.match(l)
        if not m or ". . ." in l:
            continue
        title = m.group(3).strip()
        j = i + 1
        while re.search(r"[./,]$|instruc-$", title) and j < i + 4:
            if L[j].strip():
                title = (title[:-1] if title.endswith("-") else title + ("" if title.endswith(".") else " ")) + L[j].strip()
            j += 1
        out.append([m.group(1), re.sub(r"\s+", " ", title), i])
    res = []
    for k, (num, title, i) in enumerate(out):
        end = out[k + 1][2] if k + 1 < len(out) else len(L)
        res.append((num, title, i, end))
    return res


OPNAMES = set()     # opcode names of the section titles (filled by main)


def furniture(l):
    return any(r.match(l) for r in FURNITURE)


def statements(L, a, b):
    """statements of the lines a..b: (line number, text, [bullets]) ; bullets are (line, indent, text)"""
    st = []
    for i in range(a, b):
        raw = L[i]
        if not raw.strip() or furniture(raw):
            continue
        s = raw.strip()
        if s in ("PTX ISA Notes", "Target ISA Notes"):
            continue
        ind = len(raw) - len(raw.lstrip())
        if s.startswith("▶"):
            st[-1][2].append([i + 1, ind, s[1:].strip()])
            continue
        cont = False
        glue = " "
        if st:
            last = st[-1][2][-1][2] if st[-1][2] else st[-1][1]
            if st[-1][2]:
                cont = ind >= 2   # wrapped bullet, or a new statement after the bullets
            elif not re.search(r"[.:]$", last):
                cont = True       # wrapped statement
            elif (re.search(r" \.$", last) or re.search(r"(?:^| )([a-z][a-z0-9_]*)\.$", last)
                  and re.search(r"(?:^| )([a-z][a-z0-9_]*)\.$", last).group(1) in OPNAMES) \
                    and re.match(r"[a-z][a-z0-9_:]*( |$)", s):
                cont, glue = True, ""     # "modifiers .\napprox", "rsqrt.\nf64"
        if cont:
            tgt = st[-1][2][-1] if st[-1][2] else None
            text = tgt[2] if tgt else st[-1][1]
            text = text[:-1] + s if text.endswith("-") and not text.endswith(" -") else text + glue + s
            if tgt:
                tgt[2] = text
            else:
                st[-1][1] = text
        else:
            st.append([i + 1, s, []])
    out = []
    for ln, text, bl in st:
        # several sentences on one line: split after "version X.Y." / "or higher." / before a capital
        parts = re.split(r"(?<=version \d\.\d\.)\s+|(?<=version \d\.\d)\s+(?=[A-Z])|(?<=or higher\.)\s+|(?<=\.)\s+(?=[A-Z])", norm(text))
        for k, p in enumerate(parts):
            out.append([ln, p, [[b[0], b[1], norm(b[2])] for b in bl] if k == len(parts) - 1 else []])
    return out


def norm(t):
    t = t.replace("∕", "/").replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r"\s+", " ", t).strip()
    for a, b in (("familyspecific", "family-specific"), ("architecturespecific", "architecture-specific"),
                 ("family- specific", "family-specific"), ("architecture- specific", "architecture-specific")):
        t = t.replace(a, b)
    return t


def notes(L, sec):
    num, title, i, end = sec
    a = next((j for j in range(i, end) if L[j].strip() == "PTX ISA Notes"), None)
    if a is None:
        return None
    b = next((j for j in range(a, end) if re.match(r"(Examples?|Release Notes)\b", L[j].strip()) or re.match(r"^9\.7\.\d", L[j])), end)
    return statements(L, a, b)



# ---------------------------------------------------------------------------------------------------------
# placeholders of the syntax sections, spelled out (the instruction's own qualifiers are what the model sees)
PH = {
    "rnd": ["rn", "rz", "rm", "rp"],
    "level::cache_hint": ["L2::cache_hint"],                                  # L13201
    "level::prefetch_size": ["L2::64B", "L2::128B", "L2::256B"],              # L13202
    "level1::eviction_priority": ["L1::evict_normal", "L1::evict_unchanged", "L1::evict_first",
                                  "L1::evict_last", "L1::no_allocate"],        # L13198
    "level2::eviction_priority": ["L2::evict_normal", "L2::evict_first", "L2::evict_last"],   # L13200
    # ld/ld.global.nc/st notes say ".level::eviction_priority" for the level1 set of their syntax (L13198)
    "level::eviction_priority": ["L1::evict_normal", "L1::evict_unchanged", "L1::evict_first",
                                 "L1::evict_last", "L1::no_allocate"],
    "scope": ["cta", "cluster", "gpu", "sys"],
    "sem": ["relaxed", "acquire", "release", "acq_rel"],
    "scale_vec_size": ["scale_vec"],
    "to_proxykind::from_proxykind": ["tensormap::generic"],                                       # L20589
    "to_proxykind::from_proxykind_fabric": ["generic::fabric", "fabric::generic", "fabric::fabric"],  # L20590
}
PH_OP = {
    "lop3": {"BoolOp": ["or", "and"]},                                                            # L12105
    "ldmatrix": {"dst_fmt": ["b8x16"], "src_fmt": ["b6x16_p32", "b4x16_p64"]},                   # L27435
    "prefetch": {"level::eviction_priority": ["L2::evict_last", "L2::evict_normal"]},     # L14372
    "applypriority": {"level::eviction_priority": ["L2::evict_normal"]},                 # L14436
}


def ph(name, op):
    return PH_OP.get(op, {}).get(name) or PH.get(name) or [name]


NAME = re.compile(r"^[A-Za-z0-9_]+(::[A-Za-z0-9_*]+)*$")


def match_brace(s, i):
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return j
    return -1


def components(s, op):
    """the qualifier/type components of `.a{.b}.{c/d}...`: [(alternatives, is_type)], None if not parsable"""
    out = []
    i = 0
    s = re.sub(r"\{([^{}]*)\}", lambda m: "{" + m.group(1).replace(" ", "") + "}", s.strip())
    s = re.sub(r"\{::[^{}]*\}", "", s)     # optional sub-qualifiers (.param{::entry})
    if " " in s:
        return None
    while i < len(s):
        if s[i] == "{":
            j = match_brace(s, i)
            if j < 0:
                return None
            items = [x.lstrip(".") for x in re.split(r"[/,|]", s[i + 1:j]) if x.strip(".")]
            i = j + 1
            if not items:
                continue
            opt = len(items) == 1 and items[0] not in TY
        elif s[i] == ".":
            j = i + 1
            while j < len(s) and s[j] not in ".{":
                j += 1
            items = [s[i + 1:j]] if j > i + 1 else []
            i = j
            opt = False
            if not items:
                continue
        else:
            return None
        alts = []
        for x in items:
            if not NAME.match(x):
                return None
            alts += ph(re.sub(r"::\*$", "", x), op)       # .phase_type::* -> phase_type
        if not opt:
            out.append((alts, all(a in TY for a in alts)))
    return out


def comp_terms(cs):
    terms = ["m=" + "/".join(a) for a, ty in cs if not ty]
    tys = [a for a, ty in cs if ty]
    if tys:
        terms.append("t=" + ",".join("/".join(a) for a in tys))
    return terms


def split_top(t):
    """split a feature list at top-level ', ', ', and ', ' and '"""
    out, d, cur, i = [], 0, "", 0
    while i < len(t):
        c = t[i]
        d += c == "{"
        d -= c == "}"
        if d == 0:
            m = re.match(r",\s*and\s+|,\s*|\s+and\s+", t[i:])
            if m:
                out.append(cur)
                cur = ""
                i += m.end()
                continue
        cur += c
        i += 1
    out.append(cur)
    return [x.strip() for x in out if x.strip()]


def parse_feat(text, sec):
    """feature text -> [(op, terms)] or None"""
    t = text.strip()
    t = re.sub(r"^(?:Support for |the |The )+", "", t)
    t = re.sub(r"^(?:Qualifiers?|Modifiers?|Sub-qualifiers?|Types?|Shapes?) (?=[.{])", "", t, flags=re.I)
    t = re.sub(r" (?:qualifiers?|modifiers?|sub-qualifiers?|rounding mode|types?|instruction type|instructions?|"
               r"shapes?|scope qualifier|scope)$", "", t)
    m = re.match(r"^(.*?)(?: type)? with ([a-z][a-z0-9_]*)$", t)      # "Type .b8 with ldmatrix"
    if m and m.group(2) in sec["ops"]:
        t = m.group(2) + m.group(1).strip()
    res = []
    for it in split_top(t):
        m = re.match(r"^([a-z][a-z0-9_]*)(.*)$", it)
        if m and m.group(1) in sec["ops"] and (m.group(2) == "" or m.group(2)[0] in ".{"):
            cs = components(m.group(2), m.group(1))
            if cs is None:
                return None
            res.append((m.group(1), comp_terms(cs)))
        elif it.startswith(".") or it.startswith("{."):
            for op in sec["ops"]:
                cs = components(it, op)
                if cs is None or not cs:
                    return None
                res.append((op, comp_terms(cs)))
        else:
            return None
    return res


def bullet_alts(bl):
    """bullets -> (alternatives, alternatives without the "And ... except E" group, E) or None"""
    alts, plain, exc = [], [], None
    gver, gind = 0, None
    for ln, ind, txt in bl:
        txt = re.sub(r"\s*\((?:Renamed|refer)[^)]*\)", "", txt).strip()
        m = re.match(r"^And (?:is|are) supported on following family-specific architectures from PTX ISA version "
                     r"(\d+)\.(\d+)(?: except (.*))?:$", txt)
        if m:
            gver, gind = int(m.group(1)) * 10 + int(m.group(2)), ind
            if m.group(3):
                exc = [x.strip().lstrip(".").strip() for x in m.group(3).replace(". ", ".").split("/")]
            continue
        ingrp = gind is not None and ind > gind
        v = gver if ingrp else 0
        m1 = re.match(r"^sm_(\d+)a$", txt)
        m2 = re.match(r"^sm_(\d+)f or higher in the same family$", txt)
        m3 = re.match(r"^sm_(\d+) or higher$", txt)
        if m1:
            a = (v, m1.group(1) + "a")
        elif m2:
            a = (v, m2.group(1) + "f")
        elif m3:
            a = (v, m3.group(1))
        else:
            return None
        alts.append(a)
        if not ingrp:
            plain.append(a)
    return alts, plain, exc


def alts_str(alts):
    return "|".join("%d:%s" % a for a in alts)


VER = r"(?:version )?(\d+)\.(\d+)"


def parse_stmt(sec, st):
    """statement -> [(op, terms, alts)] (terms without the section guard), [] for no legality content,
    None when not parsed"""
    ln, t, bl = st
    t = re.sub(r"\.+$", "", t).strip()
    every = [(op, []) for op in sec["ops"]]
    if re.match(r"^(?:\S.* )?(?:is )?[Ss]upported on all target architectures$", t) and not bl:
        return []
    m = re.match(r"^Introduced in PTX ISA %s$" % VER, t)
    if m:
        return [(op, tr, "%d:" % (int(m.group(1)) * 10 + int(m.group(2)))) for op, tr in every]
    m = re.match(r"^Requires (?:target |architecture )?sm_(\d+)(a?)( or higher)?$", t)
    if m and (m.group(2) == "a") != bool(m.group(3)):
        return [(op, tr, "0:%s%s" % (m.group(1), m.group(2))) for op, tr in every]
    if re.match(r"^Supported on following architectures:$", t) and bl:
        return with_bullets(every, bl)
    m = re.match(r"^(.+?) (?:is |are |was |were )?(?:introduced|added) in PTX ISA %s$" % VER, t)
    if m:
        f = parse_feat(m.group(1), sec)
        if f is None:
            return None
        return [(op, tr, "%d:" % (int(m.group(2)) * 10 + int(m.group(3)))) for op, tr in f]
    m = re.match(r"^(.+?) (?:support )?requires? (?:architecture |target )?sm_(\d+) or higher$", t)
    if m:
        f = parse_feat(m.group(1), sec)
        if f is None:
            return None
        return [(op, tr, "0:%s" % m.group(2)) for op, tr in f]
    m = re.match(r"^(.+?) (?:is |are )?supported on (?:the )?following (?:family-specific |architecture-specific )?"
                 r"architectures:$", t)
    if m and bl:
        f = parse_feat(m.group(1), sec)
        if f is None:
            return None
        return with_bullets(f, bl)
    return None


def with_bullets(f, bl):
    r = bullet_alts(bl)
    if r is None:
        return None
    alts, plain, exc = r
    out = []
    for op, tr in f:
        if exc:
            # "And is supported on following family-specific architectures ... except .kind::i8:" -- the
            # excepted qualifiers have their own "Qualifier .kind::i8 is supported on ..." statement
            out.append((op, tr + ["!m=" + "/".join(exc)], alts_str(alts)))
        else:
            out.append((op, tr, alts_str(alts)))
    return out


# ---------------------------------------------------------------------------------------------------------
def title_ops(title):
    """section title -> [(op, [first part], [other parts])]"""
    title = re.sub(r"\s*\(deprecated\)", "", title)
    res = []
    for name in re.split(r"\s*/\s*|,\s*", title):
        parts = [p for p in name.strip().split(".") if p]
        if not parts or not re.match(r"^[a-z][a-z0-9_]*$", parts[0]):
            continue
        res.append((parts[0], parts[1:2], parts[2:]))
    return res


def build_sections(L):
    secs = []
    for num, title, i, end in sections(L):
        st = notes(L, (num, title, i, end))
        tops = title_ops(title)
        if st is None or not tops:
            continue
        secs.append({"num": num, "title": title, "line": i + 1, "st": st, "tops": tops,
                     "ops": sorted({o for o, _, _ in tops})})
    # guards: the title's parts, minus the more specific sections of the same opcode
    for s in secs:
        s["guards"] = {}
        for op, first, rest in s["tops"]:
            g = []
            if first:
                g.append("p=" + first[0])
            g += ["m=" + p for p in rest]
            mine = set(first + rest)
            for o in secs:
                for op2, f2, r2 in o["tops"]:
                    other = set(f2 + r2)
                    if op2 == op and other > mine and (not first or f2 == first):
                        g.append("x=" + "+".join(p for p in f2 + r2 if p not in mine))
            g += GUARDS.get(s["num"], [])
            s["guards"].setdefault(op, []).append(g)
    return secs


def generate(path):
    L = load(path)
    for sec in sections(L):
        OPNAMES.update(re.findall(r"[a-z][a-z0-9_]*", sec[1]))
    secs = build_sections(L)
    rows = {}          # op -> [(pattern, alts, section)]
    unhandled = []
    used = set()
    for s in secs:
        for st in s["st"]:
            key = None
            for k, (num, sub, repl) in enumerate(OVERRIDES):
                if num == s["num"] and sub in st[1]:
                    key = k
                    break
            if key is not None:
                used.add(key)
                r = OVERRIDES[key][2]
                r = [] if r is None else [(op, tr.split(), al) for op, tr, al in r]
            else:
                r = None
                for k, (rx, fn) in enumerate(GENERIC):
                    mm = re.match(rx, re.sub(r"\.+$", "", st[1]))
                    if mm:
                        used.add(("g", k))
                        r = fn(s, mm)
                        break
                if r is None:
                    r = parse_stmt(s, st)
            if r is None:
                unhandled.append((s["num"], s["title"], st))
                continue
            for op, tr, al in r:
                for g in s["guards"].get(op, [[]]):
                    rows.setdefault(op, []).append((" ".join(g + tr), al, s["num"]))
    for op, tr, al, num in EXTRA:
        rows.setdefault(op, []).append((tr, al, num))
    unused = [OVERRIDES[k][:2] for k in range(len(OVERRIDES)) if k not in used]
    targets = target_strings(L)
    return secs, rows, unhandled, unused, targets


def target_strings(L):
    """11.1.2: Target string(s) sm_X ... introduced in PTX ISA version V"""
    out = {}
    for l in L:
        m = re.match(r"^Target strings? (.*?) introduced in PTX ISA version (\d+)\.(\d+)", l.strip())
        if m:
            for x in re.findall(r"sm_\d+[af]?", m.group(1)):
                out[x] = int(m.group(2)) * 10 + int(m.group(3))
    return out


def emit(rows, targets, out):
    o = ["/* GENERATED by gen_isa_table.py from the PTX ISA 9.3 text (PTX ISA Notes / Target ISA Notes of each",
         " * 9.7 instruction section, and 11.1.2 .target) -- do not edit; see gen_isa_table.py for the row format.",
         " * isaRows(op): (pattern, alternatives VER:TGT|..., section) */",
         "val isaRows : string -> list((string, string, string))",
         "function isaRows(op) = match op {"]
    for op in sorted(rows):
        seen, items = set(), []
        for r in rows[op]:
            if r not in seen:
                seen.add(r)
                items.append('("%s", "%s", "%s")' % r)
        o.append('  "%s" => [|%s|],' % (op, ",\n    ".join(items)))
    o.append("  _ => [||]")
    o.append("}")
    o.append("")
    o.append("/* 11.1.2: the PTX ISA version (x10) that introduced a target string; -1 for an unknown one */")
    o.append("val isaTargetVer : string -> int")
    o.append("function isaTargetVer(t) = match t {")
    for k in sorted(targets, key=lambda x: (int(re.sub(r"\D", "", x)), x)):
        o.append('  "%s" => %d,' % (k, targets[k]))
    o.append("  _ => -1")
    o.append("}")
    open(out, "w").write("\n".join(o) + "\n")


# ---------------------------------------------------------------------------------------------------------
# GUARDS: which instructions a section is about, when several sections share an opcode
FLT = "f32/f64/f32x2"
HALF = "f16/f16x2/bf16/bf16x2"
GUARDS = {
    # 9.7.1 integer arithmetic: no floating-point type (L5676 ff.)
    **{n: ["!t=" + FLT + "/" + HALF] for n in ("9.7.1.1", "9.7.1.2", "9.7.1.3", "9.7.1.4", "9.7.1.9",
                                                "9.7.1.11", "9.7.1.12", "9.7.1.13", "9.7.1.14")},
    # 9.7.3 single/double precision: one .f32/.f64/.f32x2 type (L8023 ff.)
    **{n: ["n=1", "t=" + FLT] for n in ("9.7.3.3", "9.7.3.4", "9.7.3.5", "9.7.3.6", "9.7.3.7", "9.7.3.8",
                                         "9.7.3.9", "9.7.3.10", "9.7.3.11", "9.7.3.12", "9.7.3.21", "9.7.3.22")},
    # 9.7.4 half precision: one .f16/.f16x2/.bf16/.bf16x2 type (L9857 ff.)
    **{n: ["n=1", "t=" + HALF] for n in ("9.7.4.1", "9.7.4.2", "9.7.4.3", "9.7.4.4", "9.7.4.5", "9.7.4.6",
                                          "9.7.4.7", "9.7.4.8", "9.7.4.9", "9.7.4.10")},
    # 9.7.5 mixed precision: .f32 destination, .f16/.bf16 source (add.f32.atype, L10997 ff.)
    **{n: ["n=2", "t=f32,f16/bf16"] for n in ("9.7.5.1", "9.7.5.2", "9.7.5.3")},
    # 9.7.6 vs 9.7.7 comparison: half precision source types (set.dtype.f16, setp.f16x2, L11546 ff.)
    "9.7.6.1": ["!t=" + HALF], "9.7.6.2": ["!t=" + HALF],
    "9.7.7.1": ["t=" + HALF], "9.7.7.2": ["t=" + HALF],
}

# rows that are not in a 9.7 section's notes: (op, pattern, alternatives, section)
EXTRA = []

# GENERIC: statements of the same wording in many sections: (regex, fn(section, match) -> rows).
# The regex is matched against the whole statement (final period removed); the rows apply to every
# opcode of the section.
def _v(m, i):
    return "%d:" % (int(m.group(i)) * 10 + int(m.group(i + 1)))


def _req(m, i):
    """'introduced in PTX ISA version X.Y' (groups i, i+1) or 'require(s) sm_N or higher' (group i+2)"""
    return _v(m, i) if m.group(i) else "0:" + m.group(i + 2)


def _all(sec, terms, alts):
    return [(op, terms.split(), alts) for op in sec["ops"]]


SPACES = "const/global/local/param/shared"           # .ss of ld/st (L13183), atom/red (.space)
COP = "ca/cg/cs/lu/cv/wb/wt"                           # cache operators, 9.7.9.1 (L12462)
INTRO_OR_REQ = r"(?:introduced in PTX ISA (?:version )?(\d+)\.(\d+)|requires? sm_(\d+) or higher)"
GENERIC = [
    # "Support for ::cta and ::cluster sub-qualifiers introduced in PTX ISA version 7.8." (ld L13333, st L13805, ...)
    (r"^Support for ::cta and ::cluster sub-qualifiers " + INTRO_OR_REQ + "$",
     lambda s, m: _all(s, "s=::cta/::cluster", _req(m, 1))),
    # "Support for sub-qualifier ::cta on .shared introduced in PTX ISA version 7.8." (mbarrier.* L22355 ...),
    # "Support for sub-qualifier ::cluster introduced in ...", "Support for ::cta sub-qualifier introduced in ..."
    (r"^Support for (?:sub-qualifier )?::(cta|cluster|entry|func)(?: sub-qualifier)?(?: on \.(?:shared|param)"
     r"(?: space)?)? " + INTRO_OR_REQ + "$",
     lambda s, m: _all(s, "s=::" + m.group(1), _req(m, 2))),
    # "Support for ::entry and ::func sub-qualifiers on .param space introduced in PTX ISA version 8.3." (ld L13336)
    (r"^Support for ::entry and ::func sub-qualifiers on \.param space " + INTRO_OR_REQ + "$",
     lambda s, m: _all(s, "s=::entry/::func", _req(m, 1))),
    # "Sub-qualifier ::cta requires sm_30 or higher." (ld L13360, ...)
    (r"^Sub-qualifier ::(cta|cluster) requires sm_(\d+) or higher$",
     lambda s, m: _all(s, "s=::" + m.group(1), "0:" + m.group(2))),
    # "Generic addressing and cache operations introduced in PTX ISA version 2.0." (ld L13327, st L13800):
    # generic addressing = no state space qualifier
    (r"^Generic addressing and cache operations introduced in PTX ISA version (\d+)\.(\d+)$",
     lambda s, m: _all(s, "!m=" + SPACES, _v(m, 1)) + _all(s, "m=" + COP, _v(m, 1))),
    # "Generic addressing requires sm_20 or higher." (ld L13354), "Use of generic addressing requires ..." (atom L21004)
    (r"^(?:Use of generic|Generic) addressing requires sm_(\d+) or higher$",
     lambda s, m: _all(s, "!m=" + SPACES, "0:" + m.group(1))),
    # "Cache operations require sm_20 or higher." (ld L13355, st L13827, suld L19456, sust L19568)
    (r"^Cache operations require sm_(\d+) or higher$",
     lambda s, m: _all(s, "m=" + COP, "0:" + m.group(1))),
    # "Support for scope qualifier, .relaxed, .acquire, .weak qualifiers introduced in PTX ISA version 6.0." (ld L13328)
    (r"^Support for scope qualifier, ((?:\.\w+, )*\.\w+) qualifiers " + INTRO_OR_REQ.replace("requires?", "require")
     + "$",
     lambda s, m: _all(s, "m=cta/cluster/gpu/sys/" + "/".join(x.strip(" .") for x in m.group(1).split(",")),
                       _req(m, 2))),
    # "Support for .level2::eviction_priority qualifier and .v8.b32/.v4.b64 introduced in PTX ISA version 8.8." (ld
    # L13339): .v8 (32-bit elements) and .v4 of 64-bit elements
    (r"^Support for \.level2::eviction_priority qualifier and \.v8\.b32/\.v4\.b64 " + INTRO_OR_REQ.replace(
        "requires?", "require") + "$",
     lambda s, m: _all(s, "m=" + "/".join(PH["level2::eviction_priority"]), _req(m, 1))
     + _all(s, "m=v8", _req(m, 1)) + _all(s, "m=v4 t=b64/u64/s64/f64", _req(m, 1))),
    # "Support for vector types introduced in PTX ISA version 8.1." (atom L20988, red L21258)
    (r"^Support for vector types " + INTRO_OR_REQ + "$", lambda s, m: _all(s, "m=v2/v4/v8", _req(m, 1))),
    # "Indirect texture access introduced in PTX ISA version 3.1." / "... requires sm_20 or higher." (tex L18953,
    # tld4, txq; "Indirect surface access" suld/sust/sured/suq): the texture/surface is a register, not a name
    (r"^Indirect (?:texture|surface) access " + INTRO_OR_REQ + "$", lambda s, m: _all(s, "ir", _req(m, 1))),
    # "Offset vector operand introduced in PTX ISA version 4.3." / "Depth compare operand ..." (tex L18963, tld4
    # L19150): an operand after [a, c] and after the lod / dPdx, dPdy operands of tex.level / tex.grad (L18712)
    (r"^(?:Offset vector|Depth compare) operand " + INTRO_OR_REQ + "$",
     lambda s, m: [(op, ["!m=level/grad", "o>2"], _req(m, 1)) for op in s["ops"]]
     + ([(op, ["m=level", "o>3"], _req(m, 1)) for op in s["ops"]] + [(op, ["m=grad", "o>4"], _req(m, 1))
                                                                       for op in s["ops"]]
        if s["ops"] == ["tex"] else [])),
    # "Support for optional destination predicate introduced in PTX ISA version 7.1." (tex L18965, tld4 L19152): d|p
    (r"^Support for optional destination predicate " + INTRO_OR_REQ + "$", lambda s, m: _all(s, "dp", _req(m, 1))),
    # "Support for .sys scope with .b128 type introduced in PTX ISA version 8.4." (ld L13338, st L13809, atom L20990)
    (r"^Support for \.sys scope with \.b128 type " + INTRO_OR_REQ + "$", lambda s, m: _all(s, "m=sys t=b128",
                                                                                          _req(m, 1))),
    # "Support for .volatile qualifier with .local state space introduced in PTX ISA version 9.1." (ld L13341)
    (r"^Support for \.volatile qualifier with \.local state space " + INTRO_OR_REQ + "$",
     lambda s, m: _all(s, "m=volatile m=local", _req(m, 1))),
    # "Support for .acquire qualifier with .mmio qualifier introduced in PTX ISA version 9.3." (ld L13342, L13367)
    (r"^Support for \.(\w+) qualifier with \.(\w+) qualifier " + INTRO_OR_REQ + "$",
     lambda s, m: _all(s, "m=%s m=%s" % (m.group(1), m.group(2)), _req(m, 3))),
    # "The .array_size query was added in PTX ISA version 4.1." (suq L19728), "The .force_unnormalized_coords
    # query ..." (txq L19285), ".array_size, .num_mipmap_levels, .num_samples samples queries were added ..." (L19287)
    (r"^(?:The )?((?:\.\w+(?:, )?)+) (?:samples )?quer(?:y|ies) (?:was|were) added in PTX ISA version (\d+)\.(\d+)$",
     lambda s, m: _all(s, "m=" + "/".join(re.findall(r"\.(\w+)", m.group(1))), _v(m, 2))),
    # "Channel data type and channel order queries were added in PTX ISA version 2.1." (txq L19284, suq L19726)
    (r"^Channel data type and channel order queries (?:were )?added in PTX ISA version (\d+)\.(\d+)$",
     lambda s, m: _all(s, "m=channel_data_type/channel_order", _v(m, 1))),
    # 9.7.15.5.14 mma / 9.7.15.6.3 mma.sp: "<types> <kind> type mma operation with <shapes> shape(s)
    # introduced in PTX ISA version X.Y" / "requires sm_N or higher" (L26973 ...; "sm_75 or higher" without
    # "requires" at L27021, L27022)
    (r"^(?:Support for )?((?:\.\w+(?:/| and |, )?)+) (?:floating point |alternate floating point |integer |"
     r"single-bit integer )?type "
     r"mma operation(?: with ((?:\.\w+(?:,? and |, )?)+) shapes?)? (?:introduced in PTX ISA version (\d+)\.(\d+)|"
     r"(?:requires? )?sm_(\d+) or higher)$",
     lambda s, m: [(op, ["m=" + "/".join(re.findall(r"\.(\w+)", m.group(1)))]
                    + (["m=" + "/".join(re.findall(r"\.(\w+)", m.group(2)))] if m.group(2) else []),
                    _req(m, 3)) for op in s["ops"]]),
]

# OVERRIDES: (section, distinctive substring of the statement, rows | None) ; None = no version/target
# content (or not checkable from the instruction; listed in the report); rows: [(op, terms, alternatives)].
# The section guard is added to every row.
I64 = "t=u64/s64/b64"
I32 = "t=u32/s32/b32"
FAM100 = "0:100a|0:101a|88:100f|88:101f|0:110f"        # the usual "sm_100a, sm_101a, family from 8.8" list
OVERRIDES = [
    # 9.7.2 extended-precision arithmetic: 32-bit / 64-bit forms (L7480 ...)
    *[(n, s32, [(op, I32, v)]) for n, op, s32, v in (
        ("9.7.2.1", "add", "32-bit add.cc introduced", "12:"), ("9.7.2.2", "addc", "32-bit addc introduced", "12:"),
        ("9.7.2.3", "sub", "32-bit sub.cc introduced", "12:"), ("9.7.2.4", "subc", "32-bit subc introduced", "12:"),
        ("9.7.2.5", "mad", "32-bit mad.cc introduced", "30:"), ("9.7.2.6", "madc", "32-bit madc introduced", "30:"))],
    *[(n, s64, [(op, I64, v)]) for n, op, s64, v in (
        ("9.7.2.1", "add", "64-bit add.cc introduced", "43:"), ("9.7.2.1", "add", "64-bit add.cc requires", "0:20"),
        ("9.7.2.2", "addc", "64-bit addc introduced", "43:"), ("9.7.2.2", "addc", "64-bit addc requires", "0:20"),
        ("9.7.2.3", "sub", "64-bit sub.cc introduced", "43:"), ("9.7.2.3", "sub", "64-bit sub.cc requires", "0:20"),
        ("9.7.2.4", "subc", "64-bit subc introduced", "43:"), ("9.7.2.4", "subc", "64-bit subc requires", "0:20"),
        ("9.7.2.5", "mad", "64-bit mad.cc introduced", "43:"), ("9.7.2.6", "madc", "64-bit madc introduced", "43:"))],
    # 9.7.3.3-5 "Rounding modifiers have the following target requirements: .rn, .rz available for all targets;
    # .rm, .rp for add.f64, requires sm_13 or higher; for add.f32, requires sm_20 or higher." (L8105-L8110)
    *[(n, "Rounding modifiers have the following target requirements", None) for n in ("9.7.3.3", "9.7.3.4",
                                                                                         "9.7.3.5")],
    *[(n, ".rn, .rz available for all targets .rm, .rp for %s.f64" % op, [(op, "m=rm/rp t=f64", "0:13")])
      for n, op in (("9.7.3.3", "add"), ("9.7.3.4", "sub"), ("9.7.3.5", "mul"))],
    *[(n, "for %s.f32, requires sm_20 or higher" % op, [(op, "m=rm/rp t=f32", "0:20")])
      for n, op in (("9.7.3.3", "add"), ("9.7.3.4", "sub"), ("9.7.3.5", "mul"))],
    # 9.7.3.7 mad (L8512-L8532)
    ("9.7.3.7", "In PTX ISA versions 1.4 and later, a rounding modifier is required for mad.f64",
     [("mad", "t=f64 !m=rn/rz/rm/rp", "-14:")]),
    ("9.7.3.7", "Legacy mad.f64 instructions having no rounding modifier", None),
    ("9.7.3.7", "In PTX ISA versions 2.0 and later, a rounding modifier is required for mad.f32 for sm_20",
     [("mad", "t=f32 !m=rn/rz/rm/rp", "-20:|0:<20")]),
    ("9.7.3.7", "Errata mad.f32 requires a rounding modifier", None),
    ("9.7.3.7", "However for PTX ISA version 3.0 and earlier, ptxas does not enforce", None),
    ("9.7.3.7", "For PTX ISA version 3.1, ptxas generates a warning", None),
    ("9.7.3.7", "Rounding modifiers have the following target requirements",
     [("mad", "m=rn/rz/rm/rp t=f64", "0:13"), ("mad", "m=rn/rz/rm/rp t=f32", "0:20")]),
    # 9.7.3.8 div, 9.7.3.13 rcp, 9.7.3.15 sqrt (L8613-L8624, L9105-L9124, L9282-L9295)
    ("9.7.3.8", "Explicit modifiers .approx, .full, .ftz, and rounding introduced in PTX ISA version 1.4",
     [("div", "m=approx/full/ftz/rn/rz/rm/rp", "14:")]),
    ("9.7.3.8", "For PTX ISA version 1.4 and later, one of .approx, .full, or .rnd is required",
     [("div", "!m=approx/full/rn/rz/rm/rp", "-14:")]),
    ("9.7.3.8", "For PTX ISA versions 1.0 through 1.3, div.f32 defaults", None),
    ("9.7.3.8", "div.rn.f64 requires sm_13 or higher, or .target map_f64_to_f32",
     [("div", "m=rn t=f64", "0:13")]),                       # map_f64_to_f32 is not modelled
    *[(n, "%s.rn.f64 and explicit modifiers .approx and .ftz were introduced in PTX ISA version 1.4" % op,
       [(op, "m=rn t=f64", "14:"), (op, "m=approx/ftz", "14:")]) for n, op in (("9.7.3.13", "rcp"),
                                                                                ("9.7.3.15", "sqrt"))],
    *[(n, "General rounding modifiers were added in PTX ISA version 2.0",
       [(op, "m=rz/rm/rp", "20:"), (op, "m=rn t=f32", "20:")]) for n, op in (("9.7.3.13", "rcp"),
                                                                              ("9.7.3.15", "sqrt"))],
    *[(n, "For PTX ISA version 1.4 and later, one of .approx or .rnd is required",
       [(op, "!m=approx/rn/rz/rm/rp", "-14:")]) for n, op in (("9.7.3.13", "rcp"), ("9.7.3.15", "sqrt"))],
    *[(n, "For PTX ISA versions 1.0 through 1.3", None) for n in ("9.7.3.13", "9.7.3.15", "9.7.3.16", "9.7.3.18",
                                                                   "9.7.3.19", "9.7.3.20", "9.7.3.21")],
    *[(n, "%s.rn.f64 requires sm_13 or higher, or .target map_f64_to_f32" % op, [(op, "m=rn t=f64", "0:13")])
      for n, op in (("9.7.3.13", "rcp"), ("9.7.3.15", "sqrt"))],
    # 9.7.3.16-21: "Explicit modifiers .approx and .ftz (were) introduced in PTX ISA version 1.4." and
    # "For PTX ISA version 1.4 and later, the .approx modifier is required." (L9362 ...)
    *[(n, "Explicit modifiers .approx and .ftz", [(op, "m=approx/ftz", "14:")])
      for n, op in (("9.7.3.16", "rsqrt"), ("9.7.3.18", "sin"), ("9.7.3.19", "cos"), ("9.7.3.20", "lg2"),
                    ("9.7.3.21", "ex2"))],
    *[(n, "For PTX ISA version 1.4 and later, the .approx modifier is required", [(op, "!m=approx", "-14:")])
      for n, op in (("9.7.3.16", "rsqrt"), ("9.7.3.18", "sin"), ("9.7.3.19", "cos"), ("9.7.3.20", "lg2"),
                    ("9.7.3.21", "ex2"))],
    # 9.7.3.11/12 min, max with three inputs: d, a, b, c (L8872, L8881, L9008, L9023)
    *[(n, "%s with three input arguments introduced" % op, [(op, "o=4", "88:")]) for n, op in (("9.7.3.11", "min"),
                                                                                              ("9.7.3.12", "max"))],
    *[(n, "%s with three input arguments requires" % op, [(op, "o=4", "0:100")]) for n, op in (("9.7.3.11", "min"),
                                                                                               ("9.7.3.12", "max"))],
    # 9.7.4.4 "fma{.oob}.{f16, f16x2, bf16, bf16x2} requires sm_90 or higher." (L10308): about .oob
    ("9.7.4.4", "fma{.oob}.{f16, f16x2, bf16, bf16x2} requires sm_90", [("fma", "m=oob", "0:90")]),
    # 9.7.6.1/2 "set/setp with .f64 source type requires sm_13 or higher." (L11312, L11401)
    ("9.7.6.1", "set with .f64 source type", [("set", "t=f64", "0:13")]),
    ("9.7.6.2", "setp with .f64 source type", [("setp", "t=f64", "0:13")]),
    # 9.7.9.3 mov: the address of a kernel entry (mov r, entry_name) is not visible in the instruction
    ("9.7.9.3", "Taking the address of kernel entry functions requires PTX ISA version 3.1", None),
    ("9.7.9.3", "Kernel function addresses should only be used", None),
    ("9.7.9.3", "See the CUDA Dynamic Parallelism Programming Guide", None),
    ("9.7.9.3", "Taking the address of kernel entry functions requires sm_35", None),
    # 9.7.9.5 shfl / 9.7.14.9 vote (deprecated): "Not supported in PTX ISA version 6.4 for .target sm_70 or higher."
    ("9.7.9.5", "Deprecated in PTX ISA version 6.0", None),
    ("9.7.9.5", "Not supported in PTX ISA version 6.4 for .target sm_70 or higher", [("shfl", "", "-64:|0:<70")]),
    ("9.7.9.5", "shfl is not supported on sm_70 or higher starting PTX ISA version 6.4", None),     # = L12857
    ("9.7.14.9", "Deprecated in PTX ISA version 6.0", None),
    ("9.7.14.9", "Not supported in PTX ISA version 6.4 for .target sm_70 or higher", [("vote", "", "-64:|0:<70")]),
    ("9.7.14.9", "vote is not supported on sm_70 or higher starting PTX ISA version 6.4", None),   # = L21631
    # 9.7.9.8 ld: generic addressing of .const depends on the address value (L13329)
    ("9.7.9.8", "Support for generic addressing of .const space added", None),
    # 9.7.9.14 st.bulk: a 32-bit size operand is a register/immediate width, not in the instruction (L14124)
    ("9.7.9.14", "Support for size operand with 32-bit length", None),
    ("9.7.9.21", "Note: The current implementation does not allow generic pointers", None),
    # 9.7.9.22 cvt (L15278 ...)
    ("9.7.9.22", ".relu modifier and {.f16x2, .bf16, .bf16x2, .tf32} destination formats introduced",
     [("cvt", "m=relu", "70:"), ("cvt", "d=f16x2/bf16/bf16x2/tf32", "70:")]),
    ("9.7.9.22", ".relu modifier and {.f16x2, .bf16, .bf16x2, .tf32} destination formats require sm_80",
     [("cvt", "m=relu", "0:80"), ("cvt", "d=f16x2/bf16/bf16x2/tf32", "0:80")]),
    # "cvt with .e4m3x2/.e5m2x2 for sm_90 or higher introduced in 7.8" + "... for sm_89 introduced in 8.1"
    # (L15285, L15294), and the same for cvt.satfinite.{e4m3x2, e5m2x2}.{f32, f16x2} (L15286, L15295)
    ("9.7.9.22", "cvt with .e4m3x2/.e5m2x2 for sm_90 or higher introduced",
     [("cvt", "m=e4m3x2/e5m2x2", "78:90|81:89")]),
    ("9.7.9.22", "cvt.satfinite.{e4m3x2, e5m2x2}.{f32, f16x2} for sm_90 or higher introduced",
     [("cvt", "m=satfinite t=e4m3x2/e5m2x2,f32/f16x2", "78:90|81:89")]),
    ("9.7.9.22", "cvt with .e4m3x2/.e5m2x2 for sm_89 introduced", None),                     # in the rows above
    ("9.7.9.22", "cvt.satfinite.{e4m3x2, e5m2x2}.{f32, f16x2} for sm_89 introduced", None),
    ("9.7.9.22", "cvt with .e4m3x2/.e5m2x2 requires sm89 or higher", [("cvt", "m=e4m3x2/e5m2x2", "0:89")]),
    ("9.7.9.22", "cvt with .s2f6x2 instruction type introduced", [("cvt", "m=s2f6x2", "91:")]),
    ("9.7.9.22", "cvt with .s2f6x2 instruction type is supported on following architectures",
     [("cvt", "m=s2f6x2", "0:100a|0:103a|0:110a|0:120a|0:121a")]),
    ("9.7.9.22", "cvt to or from .f64 requires sm_13", [("cvt", "t=f64", "0:13")]),
    # 9.7.9.23 cvt.pack "Sub byte types (.u4/.s4 and .u2/.s2) requires sm_75 or higher." (L15575)
    ("9.7.9.23", "Sub byte types (.u4/.s4 and .u2/.s2) requires sm_75", [("cvt", "t=u4/s4/u2/s2", "0:75")]),
    # 9.7.9.26.3.1 cp.async: the optional ignore-src predicate operand cannot be told from src-size (L15900)
    ("9.7.9.26.3.1", "Support for ignore-src operand", None),
    # ".multicast::cluster qualifier advised to be used with .target sm_90a or ..." is advice (L16299, L17219)
    ("9.7.9.26.4.1", ".multicast::cluster qualifier advised", None),
    ("9.7.9.26.5.2", ".multicast::cluster qualifier advised", None),
    # 9.7.9.26.5.2 "Qualifiers .tile::gather4 and .im2col::w require: sm_100a when destination state space is
    # .shared::cluster and is supported on sm_100f from PTX ISA version 8.8; sm_100 or higher when destination state
    # space is .shared::cta." (L17221-L17224)
    ("9.7.9.26.5.2", "Qualifiers .tile::gather4 and .im2col::w require",
     [("cp", "m=tile::gather4/im2col::w m=shared::cluster", "0:100a|88:100f"),
      ("cp", "m=tile::gather4/im2col::w m=shared::cta", "0:100")]),
    # 9.7.9.27 tensormap.replace: values of the new_val operand (L17809, L17811, L17835, L17851)
    ("9.7.9.27", "Qualifier .elemtype with values from 13 to 15", None),
    ("9.7.9.27", "Qualifier .swizzle_mode with value 4", None),
    ("9.7.9.27", ".field3 variant .elemtype corresponding to new_val values", None),
    ("9.7.9.27", ".field3 variant .swizzle_mode corresponding to new_val value 4", None),
    # 9.7.10.5.4 fabric.try_pullred (L18482, L18487)
    ("9.7.10.5.4", "Reduction operations .add.acc::f16 and .add.acc::f32 are supported",
     [("fabric", "m=add m=acc::f16/acc::f32", "0:120a|0:121a|0:100f|0:110f")]),
    ("9.7.10.5.4", "Reduction operations .min and .max on types .e4m3 and .e5m2 are supported",
     [("fabric", "m=min/max m=e4m3/e5m2", "0:120a|0:121a|0:100f|0:110f")]),
    # 9.7.11.3 tex (L18948-L18979); .geom/.mipmap of the syntax at L18712
    ("9.7.11.3", "Unified mode texturing introduced", None),
    ("9.7.11.3", "Extension using opaque .texref and .samplerref types and independent mode", None),
    ("9.7.11.3", "Texture arrays tex.{a1d,a2d} introduced in PTX ISA version 2.3", [("tex", "m=a1d/a2d", "23:")]),
    ("9.7.11.3", "Cubemaps and cubemap arrays introduced in PTX ISA version 3.0", [("tex", "m=cube/acube", "30:")]),
    ("9.7.11.3", "Support for mipmaps introduced in PTX ISA version 3.1", [("tex", "m=base/level/grad", "31:")]),
    ("9.7.11.3", "Multi-sample textures and multi-sample texture arrays introduced",
     [("tex", "m=2dms/a2dms", "32:")]),
    ("9.7.11.3", "Support for textures returning .f16 and .f16x2 data introduced", [("tex", "d=f16/f16x2", "42:")]),
    ("9.7.11.3", "The cubemap array geometry (.acube) requires sm_20", [("tex", "m=acube", "0:20")]),
    ("9.7.11.3", "Mipmaps require sm_20", [("tex", "m=base/level/grad", "0:20")]),
    ("9.7.11.3", "Multi-sample textures and multi-sample texture arrays require sm_30",
     [("tex", "m=2dms/a2dms", "0:30")]),
    ("9.7.11.3", "Texture fetch returning .f16 and .f16x2 data require sm_53", [("tex", "d=f16/f16x2", "0:53")]),
    # 9.7.11.5 txq (L19296, L19297)
    ("9.7.11.5", "Querying the number of mipmap levels requires sm_20", [("txq", "m=num_mipmap_levels", "0:20")]),
    ("9.7.11.5", "Querying the number of samples requires sm_30", [("txq", "m=num_samples", "0:30")]),
    # 9.7.12.1/2 suld/sust (L19445-L19456, L19555-L19568); clamp modes .trap/.clamp/.zero
    ("9.7.12.1", "Additional clamp modifiers and cache operations introduced in PTX ISA version 2.0",
     [("suld", "m=clamp/zero", "20:"), ("suld", "m=" + COP, "20:")]),
    ("9.7.12.1", "sm_1x targets support only the .trap clamping modifier", [("suld", "m=clamp/zero", "0:20")]),
    ("9.7.12.2", "sust.p, additional clamp modifiers, and cache operations introduced in PTX ISA version 2.0",
     [("sust", "m=p", "20:"), ("sust", "m=clamp/zero", "20:"), ("sust", "m=" + COP, "20:")]),
    ("9.7.12.2", "sm_1x targets support only the .trap clamping modifier", [("sust", "m=clamp/zero", "0:20")]),
    # 9.7.12.3 sured (L19646, L19659)
    ("9.7.12.3", ".u64/.s64/.b64 types with .min/.max operations introduced",
     [("sured", "m=min/max t=u64/s64/b64", "81:")]),
    ("9.7.12.3", ".u64/.s64/.b64 types with .min/.max operations requires sm_50",
     [("sured", "m=min/max t=u64/s64/b64", "0:50")]),
    # 9.7.13.3 bra (L19884)
    ("9.7.13.3", "Unimplemented indirect branch", None),
    # 9.7.13.5 call (L20043, L20048): indirect call = the callee is a register
    ("9.7.13.5", "Direct call introduced in PTX ISA version 1.0", [("call", "", "10:")]),
    ("9.7.13.5", "Indirect call introduced in PTX ISA version 2.1", [("call", "rr", "21:")]),
    ("9.7.13.5", "Indirect call requires sm_20", [("call", "rr", "0:20")]),
    # 9.7.14.1 bar, barrier (L20328-L20338); "register operands" = a %register barrier number / thread count,
    # "thread count" = the second operand of bar.sync
    ("9.7.14.1", "bar.sync without a thread count introduced in PTX ISA version 1.0", [("bar", "", "10:")]),
    ("9.7.14.1", "Register operands, thread count, and bar.{arrive,red} introduced in PTX ISA version 2.0",
     [("bar", "m=arrive/red", "20:"), ("bar", "rr", "20:"), ("bar", "!m=arrive/red o>1", "20:")]),
    ("9.7.14.1", "barrier instruction introduced in PTX ISA version 6.0", [("barrier", "", "60:")]),
    ("9.7.14.1", "Register operands, thread count, and bar{.cta}.{arrive,red} require sm_20",
     [("bar", "m=arrive/red", "0:20"), ("bar", "rr", "0:20"), ("bar", "!m=arrive/red o>1", "0:20")]),
    ("9.7.14.1", "Only bar{.cta}.sync with an immediate barrier number is supported for sm_1x", None),  # = L20336
    # 9.7.14.5 atom (L20966-L21013)
    ("9.7.14.5", "32-bit atom.global introduced in PTX ISA version 1.1", [("atom", "m=global", "11:")]),
    ("9.7.14.5", "atom.shared and 64-bit atom.global.{add,cas,exch} introduced in PTX ISA 1.2",
     [("atom", "m=shared", "12:"), ("atom", "m=global m=add/cas/exch " + I64, "12:")]),
    ("9.7.14.5", "atom.add.f32 and 64-bit atom.shared.{add,cas,exch} introduced in PTX ISA 2.0",
     [("atom", "m=add t=f32", "20:"), ("atom", "m=shared m=add/cas/exch " + I64, "20:")]),
    ("9.7.14.5", "64-bit atom.{and,or,xor,min,max} introduced in PTX ISA 3.1",
     [("atom", "m=and/or/xor/min/max " + I64, "31:")]),
    ("9.7.14.5", "Per-element atomicity of atom.f16x2 clarified", None),
    ("9.7.14.5", "64-bit atom.global.{add,cas,exch} require sm_12",
     [("atom", "m=global m=add/cas/exch " + I64, "0:12")]),
    ("9.7.14.5", "64-bit atom.shared.{add,cas,exch} require sm_20",
     [("atom", "m=shared m=add/cas/exch " + I64, "0:20")]),
    ("9.7.14.5", "64-bit atom.{and,or,xor,min,max} require sm_32", [("atom", "m=and/or/xor/min/max " + I64, "0:32")]),
    # 9.7.14.6 red (L21237-L21280)
    ("9.7.14.6", "64-bit red.{and,or,xor,min,max} introduced in PTX ISA 3.1",
     [("red", "m=and/or/xor/min/max " + I64, "31:")]),
    ("9.7.14.6", "Per-element atomicity of red.f16x2 clarified", None),
    # 9.7.9.12 st.async / 9.7.14.7 red.async: the 8.7 forms `st.async{.mmio}.release.{gpu,sys}{.global}` (L13896)
    ("9.7.9.12", "Support for .mmio qualifier, .release semantics, .global state space, and .scope qualifier",
     [("st", "m=mmio/release/global/gpu/sys", "87:")]),
    ("9.7.9.12", ".mmio qualifier, .release semantics, .global state space, and .scope qualifier require sm_100",
     [("st", "m=mmio/release/global/gpu/sys", "0:100")]),
    ("9.7.14.7", "Support for .mmio qualifier, .release semantics, .global state space, and .gpu and .sys scopes",
     [("red", "m=mmio/release/global/gpu/sys", "87:")]),
    ("9.7.14.7", ".mmio qualifier, .release semantics, .global state space, and .gpu and .sys scopes require sm_100",
     [("red", "m=mmio/release/global/gpu/sys", "0:100")]),
    # 9.7.9.22 ".ftz qualifier for cvt.f32.bf16 introduced in PTX ISA version 7.8." / "... requires sm_90" (L15284)
    ("9.7.9.22", ".ftz qualifier for cvt.f32.bf16 introduced", [("cvt", "m=ftz t=f32,bf16", "78:")]),
    ("9.7.9.22", ".ftz qualifier for cvt.f32.bf16 requires sm_90", [("cvt", "m=ftz t=f32,bf16", "0:90")]),
    # 9.7.9.26.4.1 / .5.2 "Support for .shared::cta as destination state space is introduced in PTX ISA version 8.6."
    # (L16283, L17210): .dst = .shared::cta before .src = .global (L16083)
    ("9.7.9.26.4.1", "Support for .shared::cta as destination state space", [("cp", "q=shared::cta,global", "86:")]),
    ("9.7.9.26.5.2", "Support for .shared::cta as destination state space", [("cp", "q=shared::cta,global", "86:")]),
    # "Support for .weak and .relaxed semantics, .scope and .type qualifiers are introduced in PTX ISA version 9.3."
    # and "Qualifiers .weak, .relaxed, .scope and .type supported on following architectures: sm_90a, sm_100f or
    # higher, sm_110f or higher" (L16286, L16302; multimem L16707, L16721); .scope = {.cta, .cluster, .gpu, .sys},
    # .type = {.b128} (L16096)
    *[(n, "Support for .weak and .relaxed semantics, .scope and .type qualifiers",
       [(op, "m=weak/relaxed/cta/cluster/gpu/sys/b128", "93:")]) for n, op in (("9.7.9.26.4.1", "cp"),
                                                                               ("9.7.9.26.4.4", "multimem"))],
    *[(n, "Qualifiers .weak, .relaxed, .scope and .type supported on following architectures",
       [(op, "m=weak/relaxed/cta/cluster/gpu/sys/b128", "0:90a|0:100f|0:110f")])
      for n, op in (("9.7.9.26.4.1", "cp"), ("9.7.9.26.4.4", "multimem"))],
    # 9.7.14.4 fence: ".acquire and .release qualifiers for fence instruction" are the thread fences
    # fence.{acquire,release}.sync_restrict (L20561); the proxy fences and fence.op_restrict.release are older
    ("9.7.14.4", ".acquire and .release qualifiers for fence instruction introduced",
     [("fence", "m=acquire/release !m=proxy/op_restrict", "86:")]),
    ("9.7.14.4", ".acquire and .release qualifiers for fence instruction require sm_90",
     [("fence", "m=acquire/release !m=proxy/op_restrict", "0:90")]),
    # 9.7.15.4.3 ".m8n8k4 and .m16n16k8 on wmma introduced in PTX ISA version 7.0." (L24489)
    ("9.7.15.4.3", ".m8n8k4 and .m16n16k8 on wmma introduced", [("wmma", "m=m8n8k4/m16n16k8", "70:")]),
    # 9.7.16.5.2 / 6.3 "Support for .u8.s8 and .s8.u8 as .atype.btype introduced in PTX ISA version 8.4." (L30319)
    *[(n, "Support for .u8.s8 and .s8.u8 as .atype.btype", [("wgmma", "t=u8,s8", "84:"), ("wgmma", "t=s8,u8", "84:")])
      for n in ("9.7.16.5.2", "9.7.16.6.3")],
    ("9.7.14.6", "red.global requires sm_11 or higher red.shared requires sm_12 or higher",
     [("red", "m=global", "0:11"), ("red", "m=shared", "0:12")]),
    ("9.7.14.6", "64-bit red.{and,or,xor,min,max} require sm_32", [("red", "m=and/or/xor/min/max " + I64, "0:32")]),
    # 9.7.14.13 redux.sync (L21903, L21904)
    ("9.7.14.13", ".f32 type requires sm_100a and is supported on sm_100f from PTX ISA version 8.8",
     [("redux", "t=f32", "0:100a|88:100f")]),
    ("9.7.14.13", "Qualifiers .abs and .NaN require sm_100a and are supported on sm_100f or higher",
     [("redux", "m=abs/NaN", "0:100a|88:100f")]),
    # 9.7.14.16.16/17 mbarrier.arrive / arrive_drop (L22658-L22816): the sink `_` destination; a count operand
    # (state, [addr], count) without .noComplete (.expect_tx takes a txCount instead)
    ("9.7.14.16.16", "Support for sink symbol", [("mbarrier", "sk", "71:")]),
    *[(n, "Support for count argument without the modifier .noComplete introduced",
       [("mbarrier", "o=3 !m=noComplete/expect_tx", "78:")]) for n in ("9.7.14.16.16", "9.7.14.16.17")],
    *[(n, "Support for count argument without the modifier .noComplete requires sm_90",
       [("mbarrier", "o=3 !m=noComplete/expect_tx", "0:90")]) for n in ("9.7.14.16.16", "9.7.14.16.17")],
    # 9.7.14.16.19 mbarrier.test_wait / try_wait: the reportPredicate / reportValue operands (L23121, L23130)
    # (L22975: waitComplete|reportPredicate {, reportValue}, [addr], ...: a d|p destination, or a second operand
    # that is not the [addr] operand)
    ("9.7.14.16.19", "Support for reportPredicate and reportValue operands",
     [("mbarrier", "dp", "93:"), ("mbarrier", "!m2", "93:")]),
    ("9.7.14.16.19", "Operands reportPredicate and reportValue and qualifier .phase_type::* requires sm_90",
     [("mbarrier", "m=phase_type", "0:90"), ("mbarrier", "dp", "0:90"), ("mbarrier", "!m2", "0:90")]),
    # 9.7.15.4.3-5 wmma (L24488-L24845): integer = .u8/.s8 (and their .s32 accumulators), sub-byte = .u4/.s4,
    # single-bit = .b1, double = .f64, alternate floating point = .bf16/.tf32
    *[(n, "Integer, sub-byte integer and single-bit wmma introduced in PTX ISA version 6.3",
       [("wmma", "m=u8/s8/s32/u4/s4/b1/m8n8k32/m8n8k128", "63:")]) for n in ("9.7.15.4.3", "9.7.15.4.4",
                                                                          "9.7.15.4.5")],
    *[(n, "Double precision and alternate floating point precision wmma introduced in PTX ISA version 7.0",
       [("wmma", "m=f64/bf16/tf32", "70:")]) for n in ("9.7.15.4.3", "9.7.15.4.5")],
    ("9.7.15.4.4", "Double precision wmma introduced in PTX ISA version 7.0", [("wmma", "m=f64", "70:")]),
    *[(n, "Modifier .aligned is required from PTX ISA version 6.3 onwards", [("wmma", "!m=aligned", "-63:")])
      for n in ("9.7.15.4.3", "9.7.15.4.4", "9.7.15.4.5")],
    *[(n, "Floating point wmma requires sm_70", [("wmma", "m=f16/f32", "0:70")])
      for n in ("9.7.15.4.3", "9.7.15.4.4", "9.7.15.4.5")],
    *[(n, "Integer wmma requires sm_72", [("wmma", "m=u8/s8/s32", "0:72")])
      for n in ("9.7.15.4.3", "9.7.15.4.4", "9.7.15.4.5")],
    # (sub-byte / single-bit .c/.d fragments are .s32: their shapes .m8n8k32 / .m8n8k128 mark them, L24420)
    *[(n, "Sub-byte and single-bit wmma requires sm_75", [("wmma", "m=u4/s4/b1/m8n8k32/m8n8k128", "0:75")])
      for n in ("9.7.15.4.3", "9.7.15.4.4", "9.7.15.4.5")],
    ("9.7.15.4.3", "Double precision and alternate floating point precision wmma requires sm_80",
     [("wmma", "m=f64/bf16/tf32/m16n16k8", "0:80")]),
    ("9.7.15.4.4", "Double precision wmma and shape .m16n16k8 requires sm_80",
     [("wmma", "m=f64", "0:80"), ("wmma", "m=m16n16k8", "0:80")]),
    ("9.7.15.4.5", "Double precision, alternate floating point precision wmma require sm_80",
     [("wmma", "m=f64/bf16/tf32/m16n16k8", "0:80")]),
    ("9.7.15.4.5", "Support for .and operation in single-bit wmma introduced", [("wmma", "m=and", "71:")]),
    ("9.7.15.4.5", ".and operation in single-bit wmma requires sm_80", [("wmma", "m=and", "0:80")]),
    ("9.7.15.4.5", "Support for .satfinite on floating point wmma.mma is deprecated in PTX ISA version 6.4 and is "
     "removed from PTX ISA version 6.5", [("wmma", "m=satfinite m=f16/f32", "-65:")]),
    # 9.7.15.5.14 mma (L26989-L27039)
    ("9.7.15.5.14", "Support for .and operation in single-bit mma introduced", [("mma", "m=and", "71:")]),
    ("9.7.15.5.14", ".and operation in single-bit mma requires sm_80", [("mma", "m=and", "0:80")]),
    ("9.7.15.5.14", "Support for shape .m16n8k16 and .f16 dtype/ctype with .e4m3/.e5m2",
     [("mma", "m=e4m3/e5m2 m=m16n8k16", "87:"), ("mma", "m=e4m3/e5m2 d=f16", "87:")]),
    ("9.7.15.5.14", "Support for .scale_vec::4X on .ue8m0 as .stype with .kind::mxf4nvf4",
     [("mma", "m=scale_vec::4X m=ue8m0 m=kind::mxf4nvf4", "91:")]),
    ("9.7.15.5.14", "Note: mma.sync.m8n8k4 is optimized", None),
    ("9.7.15.5.14", ".e3m2, .e2m3 and .e2m1 alternate floating point type mma operation requires sm_120a",
     [("mma", "m=e3m2/e2m3/e2m1", "0:120a|88:120f")]),
    ("9.7.15.5.14", "Support for .kind, .block_scale, .scale_vec_size qualifier requires sm_120a",
     [("mma", "m=kind/block_scale/scale_vec", "0:120a|88:120f")]),
    # 9.7.15.6.3 mma.sp (L28975-L28998)
    ("9.7.15.6.3", "Support for shape .m16n8k32 and .f16 dtype/ctype with .e4m3/.e5m2 alternate floating point "
     "type mma operation introduced", [("mma", "m=e4m3/e5m2 m=m16n8k32", "87:"), ("mma", "m=e4m3/e5m2 d=f16", "87:")]),
    ("9.7.15.6.3", "Support for shape .m16n8k32 and .f16 dtype/ctype with .e4m3/.e5m2 alternate floating point "
     "type mma operation requires sm_120",
     [("mma", "m=e4m3/e5m2 m=m16n8k32", "0:120"), ("mma", "m=e4m3/e5m2 d=f16", "0:120")]),
    ("9.7.15.6.3", "Support for .scale_vec::4X on .ue8m0 as .stype with .kind::mxf4nvf4",
     [("mma", "m=scale_vec::4X m=ue8m0 m=kind::mxf4nvf4", "91:")]),
    ("9.7.15.6.3", ".e3m2, .e2m3 and .e2m1 alternate floating point type mma operation requires sm_120a",
     [("mma", "m=e3m2/e2m3/e2m1", "0:120a|88:120f")]),
    # "... requires sm_120a and are supported on sm_120f and later generation targets in the same family from PTX
    # ISA version 8.8 except for .kind::mxf4nvf4/.kind::mxf4." (L28993-L28995); those two kinds: L28996
    ("9.7.15.6.3", "Support for .kind, .block_scale, .scale_vec_size qualifier requires sm_120a",
     [("mma", "m=kind/block_scale/scale_vec !m=kind::mxf4nvf4/kind::mxf4", "0:120a|88:120f")]),
    ("9.7.15.6.3", "kind::mxf4nvf4/.kind::mxf4", None),
    # 9.7.17.10.9.1/2 tcgen05.mma{.sp} (L34903-L34909, L35238-L35244)
    *[(n, "Argument scale-input-d requires sm_100a", None) for n in ("9.7.17.10.9.1", "9.7.17.10.9.2")],
    *[(n, "For .scale_vectorsize",
       [("tcgen05", "m=scale_vec::1X/scale_vec::2X/scale_vec::4X", "0:100a"),
        ("tcgen05", "m=block16/block32", "0:100f|0:110f")]) for n in ("9.7.17.10.9.1", "9.7.17.10.9.2")],
    *[(n, "For Target ISA details on", None) for n in ("9.7.17.10.9.1", "9.7.17.10.9.2")],
    # 9.7.19.2.1/3: "varvg2" / "varvg4" are typos of vavrg2 / vavrg4 (L36614, L36824)
    ("9.7.19.2.1", "vadd2, vsub2, varvg2, vabsdiff2, vmin2, vmax2 require sm_30",
     [(op, "", "0:30") for op in ("vadd2", "vsub2", "vavrg2", "vabsdiff2", "vmin2", "vmax2")]),
    ("9.7.19.2.3", "vadd4, vsub4, varvg4, vabsdiff4, vmin4, vmax4 require sm_30",
     [(op, "", "0:30") for op in ("vadd4", "vsub4", "vavrg4", "vabsdiff4", "vmin4", "vmax4")]),
]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    out = args[1] if len(args) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptx_isa_table.sail")
    secs, rows, unhandled, unused, targets = generate(args[0])
    if "--dump" in sys.argv:
        for s in secs:
            print("=" * 20, s["num"], s["title"], s["guards"])
            for st in s["st"]:
                print("  L%d: %s" % (st[0], st[1]))
                for b in st[2]:
                    print("      L%d %s- %s" % (b[0], " " * (b[1] // 4), b[2]))
        return
    if "--report" in sys.argv or unhandled or unused:
        for num, title, (ln, t, bl) in unhandled:
            print("UNHANDLED %s %s L%d: %s%s" % (num, title, ln, t, " [+%d bullets]" % len(bl) if bl else ""))
        for u in unused:
            print("UNUSED OVERRIDE", u)
    if "--report" in sys.argv:
        for op in sorted(rows):
            for r in rows[op]:
                print("ROW %-12s %-60s %-40s %s" % (op, r[0], r[1], r[2]))
    if unhandled or unused:
        sys.exit("gen_isa_table: %d unhandled statements, %d unused overrides" % (len(unhandled), len(unused)))
    emit(rows, targets, out)


if __name__ == "__main__":
    main()

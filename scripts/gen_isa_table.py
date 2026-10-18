#!/usr/bin/env python3
"""Generate semantics/ptx-isa-table.k from the PTX ISA HTML document (replaces the earlier text-based generator).

usage: gen_isa_table.py ptx-isa.html > semantics/ptx-isa-table.k

Each instruction section has "PTX ISA Notes" and "Target ISA Notes".  Sentences of the forms
    <expr> introduced in PTX ISA version V
    <expr> requires sm_NN or higher
    <expr> supported on following [family-specific] architectures: sm_NN[a|f] ...
give rows isaRow(qualifiers, version x10, sm, arch-specific?) where <expr> is an instruction pattern: braces and slashes
expand to alternatives, `{.x}` (optional) is dropped, and the pattern is the set of qualifiers (including type names) that an
instruction must carry for the row to apply.  Sentences that do not parse into such a pattern are ignored (the check is a
lower bound on the requirements, never stricter than the document).
"""
import html, itertools, re, sys

s = open(sys.argv[1], encoding="utf8", errors="ignore").read()
secs = [(m.start(), m.group(1)) for m in re.finditer(r'<section id="([^"]*instructions-[^"]+)">', s)]
txt = lambda x: html.unescape(re.sub(r"<[^>]+>", "", x))

def expand(item):
    """'cvt.rn{.relu}.{a/b}.f32' -> list of qualifier-tuples (opcode first)"""
    item = item.strip().rstrip(",").strip()
    # optional single qualifiers {.x} are dropped
    item = re.sub(r"\{\.[A-Za-z0-9_:]+\}", "", item)
    # alternatives {a/b} or {a, b} or .{a/b}
    parts = re.split(r"(\{[^}]*\})", item)
    opts = []
    parts_alt = []
    for p in parts:
        parts_alt.append(p.startswith("{"))
        if p.startswith("{"):
            alts = [a.strip().lstrip(".") for a in re.split(r"[/,]", p[1:-1]) if a.strip()]
            opts.append(alts)
        else:
            opts.append([p])
    out = []
    for combo in itertools.product(*opts):
        t = ""
        for i, c in enumerate(combo):
            if parts_alt[i]:
                t += c if t.endswith(".") else "." + c
            else:
                t += c
        t = re.sub(r"\.\.+", ".", t)
        out.append(t)
    return out

def pattern(t, ops):
    """token string like 'add.sat.u32' -> (opcode, frozenset(qualifiers)) or None"""
    t = t.strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[A-Za-z0-9_:]+)*", t.replace(" ", "")):
        return None
    f = t.split(".")
    return f[0], f[1:]

rows = {}      # op -> {(quals tuple, kind): value}
def add(op, quals, ver=None, sm=None, arch=None):
    # the half-precision chapter repeats the opcodes of the f32 chapters: its plain notes apply to the half types only
    if not list(quals) and globals().get("sec_family", "") == "half":
        for t in ("f16", "f16x2", "bf16", "bf16x2"): add(op, [t], ver, sm, arch)
        return
    quals = list(quals) + list(globals().get("sec_extra", []))
    if any(not re.fullmatch(r"[A-Za-z0-9_:]+", q) or q in ("scope", "sem", "space", "support", "for", "cc") for q in quals): return
    key = tuple(sorted(set(quals)))
    r = rows.setdefault(op, {}).setdefault(key, [None, None, False])
    # several sections of one opcode (integer / half / mixed precision) give different requirements for the same pattern: keep the weakest
    if ver: r[0] = ver if r[0] is None else min(r[0], ver)
    if sm:
        if r[1] is None or sm < r[1]: r[1], r[2] = sm, bool(arch)
        elif sm == r[1]: r[2] = r[2] and bool(arch)

def parse_expr(expr, sec_op):
    mw = re.match(r"^(\w+) with (.*?)(?: as instruction types?| types?)?$", expr.strip())
    if mw and mw.group(1) == sec_op:
        return [[q.lstrip(".")] for q in re.findall(r"\.[A-Za-z0-9_:]+", mw.group(2))]
    expr = re.sub(r"\b(?:the )?(?:support for|Support for|support of)\b", "", expr)
    expr = re.sub(r"\b(qualifiers?|modifiers?|types?|instructions?|instruction types?|destination formats?|rounding modes?)\b", "", expr)
    expr = expr.replace(" with ", " ").strip()
    pats = []
    for item in re.split(r"\s+and\s+|,\s+(?![^{]*\})", expr):
        item = item.strip()
        if not item: continue
        if item.startswith("."):                       # bare qualifier(s) of this instruction
            item = sec_op + item
        for t in expand(item):
            p = pattern(t, None)
            if p and p[0] == sec_op:
                pats.append(p[1])
            elif p is None:
                return []
    return pats

for k, (p, sid) in enumerate(secs):
    end = secs[k + 1][0] if k + 1 < len(secs) else len(s)
    t = re.sub(r"\n\s*\n+", "\n", txt(s[p:end]))
    a = t.find("PTX ISA Notes")
    b = t.find("Examples", a)
    if a < 0: continue
    if sid.startswith(("mixed-precision", "alternate-floating")): continue   # type-list forms overlap the plain half-precision rows
    sec_family = "half" if sid.startswith("half-precision") else ""
    parts_ = sid.split("instructions-")[-1].split("-")
    sec_op = parts_[0]
    sec_extra = [x for x in parts_[1:] if x not in ("approx",)]      # sections of variants (st-async, cp-async-bulk, ...) apply to instructions carrying these qualifiers
    note = " ".join(t[a:b].split("\n"))
    # sentences (a period followed by a space and a capital / lower-case word, not inside a version number)
    for sent in re.split(r"(?<![0-9])\.\s+|(?<=[0-9]{1}\.[0-9])\.\s+", note):
        sent = re.sub(r"^\s*(PTX ISA Notes|Target ISA Notes)\s*", "", sent)
        if re.match(r"^\s*(Target ISA Notes\s*)?Supported on all target architectures", sent, re.I):
            add(sec_op, [], sm=1); continue
        m = re.search(r"^(.*?)\s*(?:is |are )?introduced in PTX ISA version (\d+)\.(\d+)", sent, re.I)
        if m:
            ver = int(m.group(2)) * 10 + int(m.group(3))
            expr = m.group(1).strip()
            if re.match(r"^(PTX ISA Notes)?$", expr) or expr.lower() in ("", "introduced", sec_op):
                add(sec_op, [], ver=ver); continue
            if re.search(r"\bfor sm_|\bsm_\d", expr): continue
            for q in parse_expr(expr, sec_op): add(sec_op, q, ver=ver)
            continue
        m = re.search(r"^(.*?)\s*(?:requires?|is supported on|are supported on|requires)\s+(?:target\s+)?sm_(\d+)([af]?)(?:\s+or higher|\b)", sent, re.I)
        if m:
            sm = int(m.group(2)); expr = m.group(1).strip()
            if expr.lower() in ("", "target isa notes", "requires"):
                add(sec_op, [], sm=sm, arch=m.group(3) != "")
            elif not re.search(r"\bsm_\d", expr):
                for q in parse_expr(expr, sec_op): add(sec_op, q, sm=sm, arch=m.group(3) != "")
    # family/architecture-specific lists:  "<expr> (is|are) supported on following ... architectures:" then sm_NNx lines
    for m in re.finditer(r"([^.\n]*?)\s*(?:is |are )?supported on (?:the )?following (?:family-specific )?architectures:\s*((?:sm_\d+[af]?[^\n]*\n?\s*)+)", t[a:b], re.I):
        sms = [int(x) for x in re.findall(r"sm_(\d+)", m.group(2))]
        if not sms: continue
        expr = m.group(1).strip()
        if expr.lower() in ("", "target isa notes"):
            add(sec_op, [], sm=min(sms), arch=True)
        elif not re.search(r"\bsm_\d", expr):
            for q in parse_expr(expr, sec_op): add(sec_op, q, sm=min(sms), arch=True)

print("// GENERATED by scripts/gen_isa_table.py from the PTX ISA document -- do not edit.")
print("// isaRows(op) lists rows isaRow(qualifiers joined by +, min PTX version x10, min sm, arch-specific?): a row applies to an")
print("// instruction that carries all the qualifiers (modifiers and type names).")
print('requires "ptx-common.k"\n\nmodule PTX-ISA-TABLE\n  imports PTX-COMMON\n')
print("  syntax IsaRow ::= isaRow(String, Int, Int, Bool)\n  syntax List ::= isaRows(String) [function, total]")
for op in sorted(rows):
    items = " ".join('ListItem(isaRow("%s", %d, %d, %s))' % ("+".join(q), r[0] or 0, r[1] or 0, "true" if r[2] else "false")
                     for q, r in sorted(rows[op].items()) if (r[0] or r[1]))
    if items: print('  rule isaRows("%s") => %s' % (op, items))
print("  rule isaRows(_) => .List [owise]\nendmodule")

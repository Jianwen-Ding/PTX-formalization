#!/usr/bin/env python3
"""Differential test of the LLVM AGX MC layer against the AGX Sail model's `encdec` mapping.

For every generated instruction (sail/agx/gen/agxgen.py: all forms one factor at a time, plus random
instructions, plus random instruction words the Sail model decodes):

  sail       the Sail model encodes the instruction (encdec) and decodes it back (RT-OK)
  applegpu   applegpu disassembles the Sail bytes to the instruction's text (the authority)
  decode     llvm-mc --disassemble decodes the Sail bytes as one instruction of the same length
  text       LLVM's disassembly is the instruction's text in LLVM syntax (llvm_text)
  reencode   LLVM's encoding of the decoded MCInst is the Sail bytes
  asm        llvm-mc assembles the expected LLVM text to the Sail bytes

usage: mccheck.py [-s SEED] [-n RANDOM] [-w WORDS] [--reps R] [-v] [--json FILE]
"""
import argparse, json, os, random, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import agxgen  # noqa: E402
import agxtest  # noqa: E402

LLVM_BIN = os.environ.get("LLVM_BIN", "/home/user/build-llvm/bin")
MC = os.path.join(LLVM_BIN, "llvm-mc")
CPU = os.environ.get("AGX_CPU", "agx-g13g")

try:
    import xcheck_applegpu as XA   # needs applegpu (APPLEGPU=...)
except Exception:  # pragma: no cover
    XA = None

FCC = {"feq": "eq", "flt": "lt", "fgt": "gt", "fltn": "ltn", "fge": "gte", "fle": "lte", "fgtn": "gtn", "fnequ": "neq",
       "fnlt": "nlt", "fngt": "ngt", "fnltn": "nltn", "fnge": "ngte", "fnle": "nlte", "fngtn": "ngtn"}
ICC = {"eq": "ueq", "ult": "ult", "ugt": "ugt", "seq": "seq", "slt": "slt", "sgt": "sgt", "ne": "nueq", "uge": "ugte",
       "ule": "ulte", "sne": "nseq", "sge": "sgte", "sle": "slte"}
MASKS = {a: ('x' if a & 1 else '') + ('y' if a & 2 else '') + ('z' if a & 4 else '') + ('w' if a & 8 else '') for a in range(16)}


# applegpu's / LLVM's names of the convert modes (the direction field prints as rtz / rte)
CONVERT_NAMES = {0: "u8_to_f", 1: "s8_to_f", 4: "f_to_u16", 5: "f_to_s16", 6: "u16_to_f", 7: "s16_to_f", 8: "f_to_u32",
                 9: "f_to_s32", 10: "u32_to_f", 11: "s32_to_f"}
SR_NAMES = dict(XA.applegpu.SR_NAMES) if XA is not None else {}


def sail_encode(instrs):
    code, out, err = agxtest.run_tokens([["I"] + t for t in instrs] + [["X"]])
    printed = [l for l in out.splitlines() if l.startswith(("ENC ", "NOENC "))]
    assert len(printed) == len(instrs), (len(printed), len(instrs), err[-500:])
    res = []
    for l in printed:
        if l.startswith("NOENC"):
            res.append(None)
        else:
            res.append((l.split()[1], l.endswith("RT-OK"), l.split(" ", 2)[2].rsplit(" ", 1)[0]))
    return res


def sail_decode(words):
    code, out, err = agxtest.run_tokens([["V", w] for w in words])
    decs = [l for l in out.splitlines() if l.startswith("DEC ")]
    assert len(decs) == len(words), err[-500:]
    res = []
    for l in decs:
        p = l.split(" ", 2)
        res.append(None if p[1] == "0" else (int(p[1]), p[2]))
    return res


def llvm_disasm(h):
    src = ",".join("0x" + h[i:i + 2] for i in range(0, len(h), 2))
    r = subprocess.run([MC, "-triple=agx", "-mcpu=" + CPU, "--disassemble", "-show-encoding"], input=src, capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if "encoding:" in l]
    warn = "invalid instruction encoding" in r.stderr
    if not lines:
        return None
    l = lines[0]
    enc = re.search(r"encoding: \[(.*?)\]", l).group(1).replace("0x", "").replace(",", "")
    txt = " ".join(l.split("//")[0].split())
    # the first instruction decoded from the start of the bytes (a leading invalid encoding makes llvm-mc skip)
    first = not (warn and r.stderr.find("<stdin>:1:1:") >= 0)
    return txt, enc.lower(), first, len(lines)


def llvm_asm(txt):
    r = subprocess.run([MC, "-triple=agx", "-mcpu=" + CPU, "-show-encoding"], input=txt + "\n", capture_output=True, text=True)
    m = re.search(r"encoding: \[(.*?)\]", r.stdout)
    if not m:
        return None, " ".join(r.stderr.split())[:200]
    if "fixup" in r.stdout:
        return None, "fixup: " + " ".join(r.stdout.split())[:200]
    return m.group(1).replace("0x", "").replace(",", "").lower(), ""


# ---------------------------------------------------------------- expected LLVM text
def reg_text(f, n, view, hint, sx, ab, ng):
    p = "u" if f else "r"
    base = {0: "%s%d" % (p, n), 1: "%s%dl" % (p, n), 2: "%s%dh" % (p, n), 3: "%s%d_%s%d" % (p, n, p, n + 1), 4: "%s%d" % (p, n)}[view]
    return base + (".cache" if hint == 1 else ".discard" if hint == 2 else "") + (".sx" if sx else "") + (".abs" if ab else "") + \
        (".neg" if ng else "")


class Toks:
    def __init__(self, t):
        self.t, self.i = list(t), 0

    def next(self):
        v = self.t[self.i]
        self.i += 1
        return v

    def int(self):
        return int(self.next())

    def reg(self):
        return [self.int() for _ in range(7)]

    def op(self):
        k = self.next()
        if k == "R":
            return reg_text(*self.reg())
        v = self.int()
        if k == "I":
            return str(v)
        if k == "M":
            # LLVM prints a float immediate without its modifier bits (and its assembler has no syntax for them)
            self.int(), self.int()
        return fmt_float(v)


def fmt_float(bits):
    import struct
    x = struct.unpack("<d", struct.pack("<Q", bits))[0]
    return repr(x)


def tuple_text(r, count):
    f, n, view = r[0], r[1], r[2]
    if count == 0:
        return reg_text(*r[:3], 0, 0, 0, 0)
    if view == 0:
        return "_".join("r%d" % (n + i) for i in range(count))
    n16 = 2 * n + view - 1
    return "_".join("r%d%s" % ((n16 + i) >> 1, "lh"[(n16 + i) & 1]) for i in range(count))


def llvm_text(toks):
    """the instruction in the syntax of the LLVM AGX assembler (AGXInstPrinter)"""
    T = Toks(toks)
    c = T.next()
    if c == "MOV":
        d = T.reg()
        k = T.t[T.i]
        a = T.op()
        if k == "I":
            return "mov_imm %s, %s" % (reg_text(*d), a)
        return "bitop 0101, %s, %s, 0" % (reg_text(*d), a)
    if c == "GET_SR":
        d = T.reg()
        n = T.int()
        name = SR_NAMES.get(n)
        return "get_sr %s, sr%d%s" % (reg_text(*d), n, " (%s)" % name if name else "")
    if c in ("IADD", "IMADD"):
        sub, sat = T.int(), T.int()
        d = T.reg()
        ops = [T.op() for _ in range(2 if c == "IADD" else 3)]
        s = T.int()
        nm = "i" + ("m" if c == "IMADD" else "") + ("sub" if sub else "add") + (".sat" if sat else "")
        return "%s %s, %s%s" % (nm, reg_text(*d), ", ".join(ops), ", lsl %d" % s if s else "")
    if c == "BITOP":
        tt = T.int()
        d = T.reg()
        a, b = T.op(), T.op()
        alias = {8: "and", 14: "or", 6: "xor", 7: "nand", 1: "nor", 9: "xnor"}
        if tt in alias:
            return "%s %s, %s, %s" % (alias[tt], reg_text(*d), a, b)
        return "bitop %s, %s, %s, %s" % ("".join(str((tt >> i) & 1) for i in range(4)), reg_text(*d), a, b)
    if c == "INTL":
        d = T.reg()
        return "intl %s, %s, %s" % (reg_text(*d), T.op(), T.op())
    if c == "CONVERT":
        m, i2f = T.int(), T.int()
        d = T.reg()
        return "convert %s, %s, %s, %s" % (CONVERT_NAMES[m], reg_text(*d), T.op(), "rte" if i2f else "rtz")
    if c in ("BFI", "BFEIL", "EXTR"):
        m = T.int()
        d = T.reg()
        ops = [T.op() for _ in range(3)]
        return "%s %s, %s%s" % (c.lower(), reg_text(*d), ", ".join(ops), ", mask 0x%X" % ((1 << m) - 1) if m else "")
    if c == "ASR":
        d = T.reg()
        return "asr %s, %s, %s" % (reg_text(*d), T.op(), T.op())
    if c == "IUNOP":
        o = T.next()
        d = T.reg()
        return "%s %s, %s" % (o, reg_text(*d), T.op())
    if c == "FBIN":
        o, h, sat = T.next(), T.int(), T.int()
        d = T.reg()
        return "f%s%s%s %s, %s, %s" % (o, "16" if h else "32", ".sat" if sat else "", reg_text(*d), T.op(), T.op())
    if c == "FMADD":
        h, sat = T.int(), T.int()
        d = T.reg()
        return "fmadd%s%s %s, %s, %s, %s" % ("16" if h else "32", ".sat" if sat else "", reg_text(*d), T.op(), T.op(), T.op())
    if c == "RCP":
        sat = T.int()
        d = T.reg()
        return "rcp%s %s, %s" % (".sat" if sat else "", reg_text(*d), T.op())
    if c == "FUNOP":
        o, sat = T.next(), T.int()
        d = T.reg()
        return "%s%s %s, %s" % (o, ".sat" if sat else "", reg_text(*d), T.op())
    if c == "CMPSEL":
        f, cc = T.int(), T.next()
        d = T.reg()
        ops = [T.op() for _ in range(4)]
        return "%s %s, %s, %s" % ("fcmpsel" if f else "icmpsel", (FCC if f else ICC)[cc], reg_text(*d), ", ".join(ops))
    if c == "STACKOP":
        k, f, cc = T.next(), T.int(), T.next()
        a, b, n = T.op(), T.op(), T.int()
        return "%s_%scmp r0l, %s, %s, %s, %d" % (k, "f" if f else "i", (FCC if f else ICC)[cc], a, b, n)
    if c == "POP_EXEC":
        return "pop_exec r0l, %d" % T.int()
    if c == "JMP":
        a = T.int()
        T.next()
        off = T.int()
        return "%s %s" % ("jmp_exec_any" if a else "jmp_exec_none", "pc+%d" % off if off >= 0 else "pc-%d" % -off)
    if c == "BALLOT":
        f, cc = T.int(), T.next()
        d = T.reg()
        return "%s %s, %s, %s, %s" % ("fcmp_ballot" if f else "icmp_ballot", reg_text(*d), (FCC if f else ICC)[cc], T.op(), T.op())
    if c == "SHUFFLE":
        k = T.next()
        d = T.reg()
        return "simd_shuffle%s %s, %s, %s" % ("" if k == "idx" else "_" + k, reg_text(*d), T.op(), T.op())
    if c == "DEVMEM":
        st, fmt, mask = T.int(), T.next(), T.int()
        r = T.reg()
        pf, lo, hi = T.int(), T.int(), T.int()
        i = T.op()
        sg, s, g, u6 = T.int(), T.int(), T.int(), T.int()
        base = "%s%d_%s%d" % ("u" if pf else "r", lo, "u" if pf else "r", hi)
        F = {"u8": "i8", "u16": "i16", "u32": "i32"}.get(fmt, fmt)
        return "%s %d, %s, %s, %s, %s, %s, %s%s%s" % ("device_store" if st else "device_load", g, F, MASKS[mask] or "0",
                                                     tuple_text(r, bin(mask).count("1")), base, i, "signed" if sg else "unsigned",
                                                     ", lsl %d" % s if s else "", ", %d" % u6 if u6 else "")
    if c == "TGMEM":
        st, fmt, mask = T.int(), T.next(), T.int()
        r = T.reg()
        b, i, s = T.op(), T.op(), T.int()
        F = {"u8": "i8", "u16": "i16", "u32": "i32"}[fmt]
        return "%s %s, %s, %s, %s, %s" % ("threadgroup_store" if st else "threadgroup_load", F, MASKS[mask] or "0",
                                          tuple_text(r, bin(mask).count("1")), b, i)
    if c == "DEVATOM":
        o, dk = T.next(), T.next()
        d = reg_text(*T.reg()) if dk == "S" else "_"
        pf, lo, hi = T.int(), T.int(), T.int()
        i, sg, v, g, m = T.op(), T.int(), T.op(), T.int(), T.int()
        base = "%s%d_%s%d" % ("u" if pf else "r", lo, "u" if pf else "r", hi)
        # LLVM names the op field (11-15 as numbers); bits 45-46 are its last operand, the AST's mode
        A = dict({"min": "umin", "max": "umax", "fadd": "sub", "umin64": "cmpxchg", "umax64": "cmpxchg"},
                 **{"end%d" % n: str(n) for n in range(11, 16)}).get(o, o)
        return "atomic %s, %d, %s, %s, %s, %s, %s, %d" % (A, g, d, base, i, "signed" if sg else "unsigned", v, m)
    if c == "TGATOM":
        o, dk = T.next(), T.next()
        d = reg_text(*T.reg()) if dk == "S" else "_"
        b, i, v = T.op(), T.op(), T.op()
        A = {"min": "umin", "max": "umax"}.get(o, o)
        return "threadgroup_atomic %s, %s, %s, %s, %s" % (A, d, b, i, v)
    if c == "WAIT":
        return "wait %d" % T.int()
    if c == "TG_BARRIER":
        return "threadgroup_barrier"
    if c == "MEMORY_BARRIER":
        return "memory_barrier %d, %d, %d" % (T.int(), T.int(), T.int())
    if c == "STOP":
        return "stop"
    if c == "TRAP":
        return "trap"
    raise ValueError(c)


def norm(s):
    # special-register names in parentheses are comments (LLVM and applegpu name different sets)
    s = re.sub(r" \([a-z_.0-9]+\)", "", s)
    return " ".join(s.replace(",", " , ").split())


# ---------------------------------------------------------------- the check
def llvm_known(t):
    """documented differences between the LLVM AGX MC layer and the Sail model (sail/README.md): the failure kinds
    expected for this instruction, and why"""
    regs = agxgen._regs(t)
    if t[0] == "CMPSEL" and any(r[2] == 4 for r in regs[-2:]):
        return {"applegpu", "decode", "reencode", "text", "asm"}, \
            "LLVM rejects 32-bit uniforms at odd 16-bit indices; applegpu's CmpselSrcDesc asserts on them"
    if any(r[2] == 4 for r in regs):
        return {"decode", "reencode", "text", "asm"}, "LLVM rejects 32-bit uniforms at odd 16-bit indices"
    if agxgen.unmeasured(t) == "imadd 64-bit multiplicand":
        return {"decode", "reencode", "text", "asm"}, "LLVM rejects 64-bit imadd multiplicands"
    if agxgen.unmeasured(t) == "32-bit lane index":
        return {"decode", "reencode", "text", "asm"}, "LLVM rejects 32-bit lane indices of simd_shuffle"
    if "M" in t:
        return {"asm"}, "LLVM has no syntax for float immediate modifiers (its disassembler drops them)"
    return set(), None


def check(instrs, origin, jobs, verbose, results):
    enc = sail_encode(instrs)
    todo = [(t, e) for t, e in zip(instrs, enc)]
    for t, e in todo:
        if e is None:
            results["sail-noenc"].append((origin, agxgen.form_key(t), t))
    todo = [(t, e) for t, e in todo if e is not None]

    def one(te):
        t, (h, rt, sasm) = te
        out = {"t": t, "hex": h, "key": agxgen.form_key(t), "fails": []}
        if not rt:
            out["fails"].append(("sail", "encdec round trip failed: " + sasm))
        if XA is not None:
            try:
                exp = XA.applegpu_text(t)
                got = XA.disasm(h)
                if not XA.same(got, exp):
                    out["fails"].append(("applegpu", "expected %s, applegpu %s" % (exp, got)))
            except Exception as ex:
                out["fails"].append(("applegpu", "applegpu error %s" % ex))
        exp_llvm = llvm_text(t)
        d = llvm_disasm(h)
        if d is None or not d[2]:
            out["fails"].append(("decode", "llvm-mc does not decode %s (expected %s)" % (h, exp_llvm)))
        else:
            txt, renc, _, n = d
            if n != 1 or len(renc) != len(h) and renc != h[:len(renc)]:
                out["fails"].append(("decode", "llvm-mc decodes %s as %d instruction(s), first %r [%s]" % (h, n, txt, renc)))
            elif renc != h:
                out["fails"].append(("reencode", "%s decodes to %r, which encodes as %s" % (h, txt, renc)))
            if norm(txt) != norm(exp_llvm):
                out["fails"].append(("text", "%s: llvm %r, expected %r" % (h, txt, exp_llvm)))
        a, err = llvm_asm(exp_llvm)
        if a != h:
            out["fails"].append(("asm", "%r assembles to %s, expected %s %s" % (exp_llvm, a, h, err)))
        kinds, why = llvm_known(t)
        out["known"] = why if any(k in kinds for k, _ in out["fails"]) else None
        out["fails"] = [f for f in out["fails"] if f[0] not in kinds]
        return out

    with ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(one, todo):
            r["origin"] = origin
            results["checked"].append(r)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int, default=1)
    ap.add_argument("-n", type=int, default=2000, help="random instructions")
    ap.add_argument("-w", type=int, default=2000, help="random instruction words")
    ap.add_argument("--reps", type=int, default=2, help="repetitions per form value")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--max-print", type=int, default=6)
    a = ap.parse_args(argv)
    R = random.Random(a.seed)
    results = {"checked": [], "sail-noenc": []}
    forms = agxgen.forms(R, reps=a.reps)
    check(forms, "forms", a.j, a.v, results)
    rnd = [agxgen.random_instr(R) for _ in range(a.n)]
    check(rnd, "random", a.j, a.v, results)
    # random words: those the Sail model decodes, as instructions
    if a.w and XA is not None:
        words = []
        descs = [d for d in XA.applegpu.instruction_descriptors if d.name.split(".")[0] in {x.split(".")[0] for x in XA.MODELLED}]
        while len(words) < a.w:
            w = R.getrandbits(80)
            o = R.choice(descs)
            words.append((w & ~o.mask) | o.bits)
        decs = sail_decode(words)
        dins = []
        for w, d in zip(words, decs):
            if d is not None:
                try:
                    dins.append(agxtest.instr(d[1])[0:])
                except agxtest.ParseError:
                    pass
        check(dins, "words", a.j, a.v, results)
    # summary
    by = {}
    keys = set()
    for r in results["checked"]:
        keys.add(r["key"])
        for kind, msg in r["fails"]:
            by.setdefault(kind, []).append((r["key"], msg))
    n = len(results["checked"])
    print("AGX MC differential test (seed %d): %d instructions encoded by the Sail model (%d forms, %d random, %d decoded words), "
          "%d distinct forms; %d without a Sail encoding" % (a.seed, n, len(forms), len(rnd), n - len(forms) - len(rnd) + len(results["sail-noenc"]),
                                                            len(keys), len(results["sail-noenc"])))
    known = {}
    for r in results["checked"]:
        if r.get("known"):
            known[r["known"]] = known.get(r["known"], 0) + 1
    for why, c in sorted(known.items()):
        print("  known     %6d  %s" % (c, why))
    for kind in ("sail", "applegpu", "decode", "reencode", "text", "asm"):
        bad = by.get(kind, [])
        print("  %-9s %6d / %d agree (%.2f%%)" % (kind, n - len(bad), n, 100.0 * (n - len(bad)) / max(n, 1)))
        groups = {}
        for k, m in bad:
            groups.setdefault(k.split()[0], []).append(m)
        for g, ms in sorted(groups.items(), key=lambda x: -len(x[1])):
            print("      %-22s %5d  e.g. %s" % (g, len(ms), ms[0]))
            if a.v:
                for m in ms[1:a.max_print]:
                    print("      %-22s        %s" % ("", m))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"checked": n, "forms": len(keys), "fails": {k: len(v) for k, v in by.items()},
                       "details": {k: [m for _, m in v[:200]] for k, v in by.items()}}, f, indent=1)
    return 0 if not by else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cross-check the Sail `encdec` mapping against dougallj/applegpu (read-only, APPLEGPU=/home/user/applegpu).

1. encode: random instructions (and the instructions of agx/tests) are encoded by the Sail model;
   each encoding is disassembled by applegpu and compared with the instruction rendered in
   applegpu's syntax; the Sail model also decodes its own encoding back (RT-OK).
2. decode: random instruction words are decoded by both the Sail model and applegpu; whenever
   applegpu selects one of the modelled descriptors, the Sail decoding must print the same text,
   or be one of the rejections documented in agx_encdec.sail (UNDEFINED operand forms, fields with
   no observed values, ...), which are counted separately.

usage: xcheck_applegpu.py [-n COUNT] [-s SEED] [-v]
"""
import os, random, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agxtest  # noqa: E402

sys.dont_write_bytecode = True
APPLEGPU = os.environ.get("APPLEGPU", "/home/user/applegpu")
sys.path.insert(0, APPLEGPU)
import applegpu  # noqa: E402

ICC = {"eq": "ueq", "ult": "ult", "ugt": "ugt", "seq": "seq", "slt": "slt", "sgt": "sgt", "ne": "nueq", "uge": "ugte",
       "ule": "ulte", "sne": "nseq", "sge": "sgte", "sle": "slte"}
FCC = {"feq": "eq", "flt": "lt", "fgt": "gt", "fltn": "ltn", "fge": "gte", "fle": "lte", "fgtn": "gtn", "fnequ": "neq",
       "fnlt": "nlt", "fngt": "ngt", "fnltn": "nltn", "fnge": "ngte", "fnle": "nlte", "fngtn": "ngtn"}
AOP = {"add": "add", "sub": "sub", "xchg": "xchg", "min": "umin", "imin": "imin", "max": "umax", "imax": "imax",
       "and": "and", "or": "or", "xor": "xor"}
MASKS = {a: ('x' if a & 1 else '') + ('y' if a & 2 else '') + ('z' if a & 4 else '') + ('w' if a & 8 else '') for a in range(16)}

def fimm_value(n):
    return applegpu.decode_float_immediate(n)

FIMM_BITS = {struct.unpack("<Q", struct.pack("<d", fimm_value(n)))[0]: n for n in range(256)}

# ------------------------------------------------------------------ rendering in applegpu syntax
KNOWN = [0]   # operands affected by the applegpu decoder issue noted in applegpu_text
class Toks:
    def __init__(self, toks):
        self.t = list(toks)
        self.i = 0
    def next(self):
        v = self.t[self.i]; self.i += 1; return v
    def int(self):
        return int(self.next())

def reg_text(f, n, view, hint, sx, ab, ng):
    p = "u" if f else "r"
    base = {0: "%s%d" % (p, n), 1: "%s%dl" % (p, n), 2: "%s%dh" % (p, n), 3: "%s%d_%s%d" % (p, n, p, n + 1)}[view]
    flags = []
    if hint == 1: flags.append("cache")
    if hint == 2: flags.append("discard")
    if sx: flags.append("sx")
    if ab: flags.append("abs")
    if ng: flags.append("neg")
    return ".".join([base] + flags)

def rd_reg(T):
    return [T.int() for _ in range(7)]

def rd_op(T, fimm=False):
    k = T.next()
    if k == "R":
        return reg_text(*rd_reg(T))
    v = T.int()
    if k == "I":
        return str(v)
    return str(struct.unpack("<d", struct.pack("<Q", v))[0])

def tuple_text(r, count):
    f, n, view = r[0], r[1], r[2]
    if count == 0:
        return ""
    if view == 0:
        return "_".join("r%d" % (n + i) for i in range(count))
    n16 = 2 * n + view - 1
    return "_".join(applegpu.reg16_names[n16 + i] for i in range(count))

def applegpu_text(toks):
    """the expected applegpu disassembly (mnemonic, operand strings) of a serialized instruction"""
    T = Toks(toks)
    c = T.next()
    if c == "MOV":
        d = rd_reg(T); k = T.t[T.i]
        a = rd_op(T)
        if k == "I":
            if d[2] in (1, 2):
                return "mov_imm", [reg_text(*d), a]
            return "mov_imm", [reg_text(*d), a, "0b0"]
        return "mov", [reg_text(*d), a]
    if c == "GET_SR":
        d = rd_reg(T); n = T.int()
        name = "sr%d" % n + (" (%s)" % applegpu.SR_NAMES[n] if n in applegpu.SR_NAMES else "")
        return "get_sr", [reg_text(*d), name]
    if c == "IADD":
        sub, sat = T.int(), T.int(); d = rd_reg(T); a = rd_op(T); b = rd_op(T); s = T.int()
        return ("isub" if sub else "iadd") + (".sat" if sat else ""), [reg_text(*d), a, b] + (["lsl %d" % s] if s else [])
    if c == "IMADD":
        sub, sat = T.int(), T.int(); d = rd_reg(T); a = rd_op(T); b = rd_op(T); cc = rd_op(T); s = T.int()
        return ("imsub" if sub else "imadd") + (".sat" if sat else ""), [reg_text(*d), a, b, cc] + (["lsl %d" % s] if s else [])
    if c == "BITOP":
        tt = T.int(); d = rd_reg(T); a = rd_op(T); b = rd_op(T)
        tts = "".join(str((tt >> i) & 1) for i in range(4))
        binary = {'0001': 'and', '0111': 'or', '0110': 'xor', '1110': 'nand', '1000': 'nor', '1001': 'xnor'}
        unary = {'1010': 'not', '0101': 'mov'}
        if tts in binary:
            return binary[tts], [reg_text(*d), a, b]
        if b == "0" and tts in unary:
            return unary[tts], [reg_text(*d), a]
        if tts in ('0011', '1100'):
            return "bitop_mov_a", [tts, reg_text(*d), a, b]
        return "bitop", [tts, reg_text(*d), a, b]
    if c in ("BFI", "BFEIL"):
        m = T.int(); d = rd_reg(T); a = rd_op(T); b = rd_op(T); x = rd_op(T)
        return c.lower(), [reg_text(*d), a, b, x] + (["mask 0x%X" % ((1 << m) - 1)] if m else [])
    if c == "ASR":
        d = rd_reg(T); a = rd_op(T); b = rd_op(T)
        return "asr", [reg_text(*d), a, b]
    if c == "IUNOP":
        o = T.next(); d = rd_reg(T); a = rd_op(T)
        return o, [reg_text(*d), a]
    if c == "FBIN":
        o = T.next(); h = T.int(); sat = T.int(); d = rd_reg(T); a = rd_op(T); b = rd_op(T)
        return "f%s%s%s" % (o, "16" if h else "32", ".sat" if sat else ""), [reg_text(*d), a, b]
    if c == "FMADD":
        h = T.int(); sat = T.int(); d = rd_reg(T); a = rd_op(T); b = rd_op(T); x = rd_op(T)
        return "fmadd%s%s" % ("16" if h else "32", ".sat" if sat else ""), [reg_text(*d), a, b, x]
    if c == "RCP":
        sat = T.int(); d = rd_reg(T); a = rd_op(T)
        return "rcp" + (".sat" if sat else ""), [reg_text(*d), a]
    if c == "CMPSEL":
        f = T.int(); cc = T.next(); d = rd_reg(T); a = rd_op(T); b = rd_op(T)
        def csel():
            # applegpu's CmpselSrcDesc.decode drops bit 8 of a uniform register index (its encoder and
            # pseudocode keep it, as the Sail model does): render what its disassembler prints
            if T.t[T.i] == "R" and int(T.t[T.i + 1]) == 1:
                T.next(); r = rd_reg(T)
                n16 = 2 * r[1] + (r[2] - 1 if r[2] else 0)
                if n16 >= 256:
                    KNOWN[0] += 1
                    n16 -= 256
                    r[1] = n16 >> 1
                    if r[2]:
                        r[2] = 1 + (n16 & 1)
                return reg_text(*r)
            return rd_op(T)
        x = csel(); y = csel()
        return ("fcmpsel" if f else "icmpsel"), [(FCC if f else ICC)[cc], reg_text(*d), a, b, x, y]
    if c == "STACKOP":
        k = T.next(); f = T.int(); cc = T.next(); a = rd_op(T); b = rd_op(T); n = T.int()
        ccs = (FCC if f else ICC)[cc]
        if k == "if" and f and ccs == "eq" and a == "0.0" and b == "0.0":
            return ("update_exec", ["r0l"]) if n == 0 else ("push_exec", ["r0l", str(n)])
        return "%s_%scmp" % (k, "f" if f else "i"), ["r0l", ccs, a, b, str(n)]
    if c == "POP_EXEC":
        return "pop_exec", ["r0l", str(T.int())]
    if c == "JMP":
        any_ = T.int(); T.next(); off = T.int()
        return ("jmp_exec_any" if any_ else "jmp_exec_none"), ["pc+%d" % off if off >= 0 else "pc-%d" % -off]
    if c == "BALLOT":
        f = T.int(); cc = T.next(); d = rd_reg(T); a = rd_op(T); b = rd_op(T)
        return ("fcmp_ballot" if f else "icmp_ballot"), [reg_text(*d), (FCC if f else ICC)[cc], a, b]
    if c == "SHUFFLE":
        k = T.next(); d = rd_reg(T); a = rd_op(T); b = rd_op(T)
        return "simd_shuffle" + ("" if k == "idx" else "_" + k), [reg_text(*d), a, b]
    if c == "DEVMEM":
        st = T.int(); fmt = T.next(); mask = T.int(); r = rd_reg(T)
        pf, lo, hi = T.int(), T.int(), T.int()
        i = rd_op(T); sg = T.int(); s = T.int(); g = T.int(); u6 = T.int()
        base = "%s%d_%s%d" % ("u" if pf else "r", lo, "u" if pf else "r", hi)
        F = {"u8": "i8", "u16": "i16", "u32": "i32"}[fmt]
        ops = [str(g), F, MASKS[mask], tuple_text(r, bin(mask).count("1")), base, i, "signed" if sg else "unsigned"] + (["lsl %d" % s] if s else [])
        if st:
            ops.append(str(u6))
        return ("device_store" if st else "device_load"), ops
    if c == "TGMEM":
        st = T.int(); fmt = T.next(); mask = T.int(); r = rd_reg(T); b = rd_op(T); i = rd_op(T); s = T.int()
        F = {"u8": "i8", "u16": "i16", "u32": "i32"}[fmt]
        return ("threadgroup_store" if st else "threadgroup_load"), [F, MASKS[mask], tuple_text(r, bin(mask).count("1")), b, i]
    if c == "DEVATOM":
        o = T.next(); dk = T.next(); d = reg_text(*rd_reg(T)) if dk == "S" else "None"
        pf, lo, hi = T.int(), T.int(), T.int(); i = rd_op(T); sg = T.int(); v = rd_op(T); g = T.int()
        base = "%s%d_%s%d" % ("u" if pf else "r", lo, "u" if pf else "r", hi)
        return "atomic", [AOP[o], str(g), d, base, i, "signed" if sg else "unsigned", v, "1", "0", "1", "4", "2"]
    if c == "TGATOM":
        o = T.next(); dk = T.next(); d = reg_text(*rd_reg(T)) if dk == "S" else "None"
        b = rd_op(T); i = rd_op(T); v = rd_op(T)
        return "threadgroup_atomic", [AOP[o], d, b, i, v, "0", "2", "128", "0"]
    if c == "WAIT":
        return "wait", [str(T.int())]
    if c == "TG_BARRIER":
        return "threadgroup_barrier", []
    if c == "STOP":
        return "stop", []
    raise ValueError(c)

def norm(s):
    return " ".join(s.split())

def same(got, exp):
    return got is not None and got[0] == exp[0] and [norm(x) for x in got[1] if x] == [norm(x) for x in exp[1] if x]

def disasm(hexs):
    d = applegpu.disassemble_bytes(bytes.fromhex(hexs) + bytes(12))
    if d is None:
        return None
    return d.mnem, [o for o in (str(x) for x in d.operands) if o]

# ------------------------------------------------------------------ random instructions
R = random.Random()

def rreg(views=(0, 1, 2), files=(0,), mods=False, sx=False, fmods=False, hints=True, n32=128):
    f = R.choice(files)
    v = R.choice(views)
    if f == 1:
        n = R.randrange(256)
        hint = 0
    else:
        n = R.randrange(n32 if v in (0, 3) else 128)
        if v == 3:
            n = R.randrange(127)
        hint = R.choice((0, 1, 2)) if hints else 0
    return [f, n, v, hint, int(sx and R.random() < 0.3), int(fmods and R.random() < 0.3), int(fmods and R.random() < 0.3)]

def rdst(views=(0, 1, 2)):
    r = rreg(views=views)
    r[3] = R.choice((0, 1))
    return r

def rsrc(views=(0, 1, 2), uni=True, sx=False, fmods=False, imm=True, fimm=False):
    k = R.random()
    if imm and k < 0.25:
        if fimm:
            return ["F", R.choice(list(FIMM_BITS))]
        return ["I", R.randrange(256)]
    files = (0, 1) if uni else (0,)
    r = rreg(views=views, files=files, sx=sx, fmods=fmods)
    if r[0] == 1:
        r[2] = R.choice([v for v in views if v != 3] or [1])
    return ["R"] + r

def rcsel(is32):
    k = R.random()
    if k < 0.3:
        return ["I", R.randrange(256)]
    r = rreg(views=(0,) if is32 else (1, 2), files=(0, 1))
    if r[0] == 1:
        r[3] = 0
    return ["R"] + r

ICCS = list(ICC)
FCCS = list(FCC)

def rinstr():
    c = R.choice(["MOVI", "MOVR", "GET_SR", "IADD", "IMADD", "BITOP", "BFI", "BFEIL", "ASR", "IUNOP", "FBIN", "FMADD", "RCP",
                  "ICMPSEL", "FCMPSEL", "STACKOP", "POP_EXEC", "JMP", "BALLOT", "SHUFFLE", "DEVMEM", "TGMEM", "DEVATOM",
                  "TGATOM", "WAIT", "TG_BARRIER", "STOP"])
    if c == "MOVI":
        d = rdst()
        return ["MOV"] + d + ["I", R.randrange(1 << (16 if d[2] in (1, 2) else 32))]
    if c == "MOVR":
        return ["MOV"] + rdst() + rsrc(imm=False)
    if c == "GET_SR":
        return ["GET_SR"] + rdst() + [R.randrange(256)]
    if c == "IADD":
        return ["IADD", R.randrange(2), R.randrange(2)] + rdst((0, 1, 2, 3)) + rsrc((0, 1, 2, 3), sx=True) + rsrc((0, 1, 2, 3), sx=True) + [R.randrange(8)]
    if c == "IMADD":
        return ["IMADD", R.randrange(2), R.randrange(2)] + rdst((0, 1, 2, 3)) + rsrc(sx=True) + rsrc(sx=True) + rsrc((0, 1, 2, 3), sx=True) + [R.randrange(8)]
    if c == "BITOP":
        while True:
            t = ["BITOP", R.randrange(16)] + rdst() + rsrc() + rsrc()
            if not (t[1] == 10 and t[-2:] == ["I", 0]):
                return t
    if c in ("BFI", "BFEIL"):
        return [c, R.randrange(32)] + rdst() + rsrc() + rsrc() + rsrc()
    if c == "ASR":
        return ["ASR"] + rdst() + rsrc() + rsrc()
    if c == "IUNOP":
        return ["IUNOP", R.choice(["popcount", "bitrev", "ffs"])] + rdst() + rsrc()
    if c == "FBIN":
        h = R.randrange(2)
        v = (1, 2) if h else (0, 1, 2)
        return ["FBIN", R.choice(["add", "mul"]), h, R.randrange(2)] + rdst(v) + rsrc(v, fmods=True, fimm=True) + rsrc(v, fmods=True, fimm=True)
    if c == "FMADD":
        h = R.randrange(2)
        v = (1, 2) if h else (0, 1, 2)
        return ["FMADD", h, R.randrange(2)] + rdst(v) + rsrc(v, fmods=True, fimm=True) + rsrc(v, fmods=True, fimm=True) + rsrc(v, fmods=True, fimm=True)
    if c == "RCP":
        return ["RCP", R.randrange(2)] + rdst() + rsrc(fmods=True, fimm=True)
    if c in ("ICMPSEL", "FCMPSEL"):
        f = int(c == "FCMPSEL")
        d = rdst()
        is32 = d[2] == 0
        cc = R.choice([x for x in (FCCS if f else ICCS) if (FCC if f else ICC)[x] in
                       (("eq", "lt", "gt", "ltn", "gte", "lte", "gtn") if f else ("ueq", "ult", "ugt", "seq", "slt", "sgt"))])
        a = rsrc(fmods=bool(f), fimm=bool(f))
        b = rsrc(fmods=bool(f), fimm=bool(f))
        return ["CMPSEL", f, cc] + d + a + b + rcsel(is32) + rcsel(is32)
    if c == "STACKOP":
        f = R.randrange(2)
        return ["STACKOP", R.choice(["if", "else", "while"]), f, R.choice(FCCS if f else ICCS)] + rsrc(fmods=bool(f), fimm=bool(f)) + rsrc(fmods=bool(f), fimm=bool(f)) + [R.randrange(4)]
    if c == "POP_EXEC":
        return ["POP_EXEC", R.randrange(4)]
    if c == "JMP":
        return ["JMP", R.randrange(2), "JR", R.randrange(-(1 << 31), 1 << 31) & ~1]
    if c == "BALLOT":
        f = R.randrange(2)
        return ["BALLOT", f, R.choice(FCCS if f else ICCS)] + rdst() + rsrc(fmods=bool(f), fimm=bool(f)) + rsrc(fmods=bool(f), fimm=bool(f))
    if c == "SHUFFLE":
        return ["SHUFFLE", R.choice(["idx", "xor", "up", "down"])] + rdst() + rsrc() + rsrc((1, 2))
    if c == "DEVMEM":
        st = R.randrange(2)
        r = rreg(hints=False)
        r[2] = R.choice((0, 1, 2))
        if r[2] == 0:
            r[1] = R.randrange(124)
        fmt = R.choice(["u8", "u16", "u32"]) if r[2] == 0 else R.choice(["u8", "u16"])
        lo = R.randrange(127)
        idx = ["I", R.randrange(-32768, 32768)] if R.random() < 0.5 else ["R", 0, R.randrange(128), 0, 0, 0, 0, 0]
        return ["DEVMEM", st, fmt, R.randrange(16)] + r + [R.randrange(2), lo, lo + 1] + idx + [R.randrange(2), R.randrange(3), R.randrange(2), R.randrange(2) if st else 0]
    if c == "TGMEM":
        st = R.randrange(2)
        r = rreg(hints=False)
        if r[2] == 0:
            r[1] = R.randrange(124)
        base = ["I", 0] if R.random() < 0.4 else ["R"] + [R.randrange(2), R.randrange(128), R.choice((1, 2)), 0, 0, 0, 0]
        idx = ["I", R.randrange(-32768, 32768)] if R.random() < 0.5 else ["R", 0, R.randrange(128), R.choice((1, 2)), 0, 0, 0, 0]
        return ["TGMEM", st, R.choice(["u8", "u16", "u32"]), R.randrange(16)] + r + base + idx + [0]
    if c == "DEVATOM":
        d = ["S", 0, R.randrange(128), 0, 0, 0, 0, 0] if R.random() < 0.8 else ["N"]
        lo = R.randrange(127)
        idx = ["I", R.randrange(-32768, 32768)] if R.random() < 0.5 else ["R", 0, R.randrange(128), 0, 0, 0, 0, 0]
        return ["DEVATOM", R.choice(list(AOP))] + d + [R.randrange(2), lo, lo + 1] + idx + [R.randrange(2)] + ["R", 0, R.randrange(128), 0, 0, 0, 0, 0] + [R.randrange(2)]
    if c == "TGATOM":
        d = ["S", 0, R.randrange(124), R.choice((0, 1, 2)), 0, 0, 0, 0] if R.random() < 0.8 else ["N"]
        base = ["I", 0] if R.random() < 0.4 else ["R"] + [R.randrange(2), R.randrange(128), R.choice((1, 2)), 0, 0, 0, 0]
        idx = ["I", R.randrange(-32768, 32768)] if R.random() < 0.5 else ["R", 0, R.randrange(128), R.choice((1, 2)), 0, 0, 0, 0]
        return ["TGATOM", R.choice(list(AOP))] + d + base + idx + ["R", 0, R.randrange(128), 0, 0, 0, 0, 0]
    if c == "WAIT":
        return ["WAIT", R.randrange(2)]
    return [c]

# ------------------------------------------------------------------ checks
MODELLED = {"mov_imm", "get_sr", "iadd", "isub", "iadd.sat", "isub.sat", "imadd", "imsub", "imadd.sat", "imsub.sat", "bfi",
            "bfeil", "asr", "bitop", "popcount", "bitrev", "ffs", "fmadd32", "fmadd16", "fadd32", "fadd16", "fmul32", "fmul16",
            "rcp", "stop", "jmp_exec_any", "jmp_exec_none", "pop_exec", "if_icmp", "if_fcmp", "while_icmp", "while_fcmp",
            "else_icmp", "else_fcmp", "icmpsel", "fcmpsel", "icmp_ballot", "fcmp_ballot", "simd_shuffle", "simd_shuffle_down",
            "simd_shuffle_up", "simd_shuffle_xor", "wait", "device_load", "device_store", "threadgroup_load",
            "threadgroup_store", "threadgroup_barrier", "atomic", "threadgroup_atomic"}

def desc_name(word):
    for o in applegpu.instruction_descriptors:
        if o.matches(word):
            return o.name
    return None

def check_encode(instrs, verbose):
    lines = [["I"] + t for t in instrs] + [["X"]]
    code, out, err = agxtest.run_tokens(lines)
    printed = [l for l in out.splitlines() if l.startswith(("ENC ", "NOENC "))]
    assert len(printed) == len(instrs), (len(printed), len(instrs), err[-500:])
    ok = bad = noenc = 0
    for t, line in zip(instrs, printed):
        if line.startswith("NOENC"):
            noenc += 1
            if verbose:
                print("NOENC", " ".join(map(str, t)), "|", line)
            continue
        hexs = line.split()[1]
        exp_m, exp_o = applegpu_text(t)
        got = disasm(hexs)
        good = same(got, (exp_m, exp_o)) and line.endswith("RT-OK")
        if good:
            ok += 1
        else:
            bad += 1
            if bad <= 25 or verbose:
                print("MISMATCH %s\n   sail:     %s\n   expected: %s %s\n   applegpu: %s" % (" ".join(map(str, t)), line, exp_m, exp_o, got))
    return ok, bad, noenc

def check_decode(n, verbose):
    words = []
    names = sorted(MODELLED)
    while len(words) < n:
        # bias the low bytes towards the opcodes of the modelled descriptors
        w = R.getrandbits(80)
        if R.random() < 0.9:
            o = R.choice([d for d in applegpu.instruction_descriptors if d.name.split(".")[0] in {x.split(".")[0] for x in names}])
            w = (w & ~o.mask) | o.bits
        words.append(w)
    lines = [["V", w] for w in words]
    code, out, err = agxtest.run_tokens(lines)
    decs = [l for l in out.splitlines() if l.startswith("DEC ")]
    assert len(decs) == len(words), err[-500:]
    agree = rejected = mismatch = other = 0
    for w, line in zip(words, decs):
        b = w.to_bytes(10, "little")
        name = desc_name(w)
        try:
            d = applegpu.disassemble_bytes(b + bytes(12))
            got = (d.mnem, [o for o in (str(x) for x in d.operands) if o]) if d is not None else None
        except Exception:
            got = None   # applegpu asserts on UNDEFINED operand forms
        modelled = name is not None and (name in MODELLED or name.split(".")[0] in MODELLED)
        parts = line.split(" ", 2)
        size = int(parts[1])
        if not modelled:
            if size == 0:
                other += 1
            else:
                mismatch += 1
                if mismatch <= 25 or verbose:
                    print("DECODED-UNMODELLED %020x %s applegpu=%s %s" % (w, line, name, got))
            continue
        if size == 0:
            rejected += 1
            if verbose:
                print("REJECTED %020x applegpu: %s" % (w, got))
            continue
        asm = parts[2]
        exp = applegpu_text(agxtest.instr(asm))
        if same(got, exp):
            agree += 1
        else:
            mismatch += 1
            if mismatch <= 25 or verbose:
                print("DECODE-MISMATCH %020x\n   sail: %s -> %s\n   applegpu: %s (%s)" % (w, asm, exp, got, name))
    return agree, rejected, mismatch, other

def main(argv):
    n, seed, verbose = 3000, 1, "-v" in argv
    if "-n" in argv:
        n = int(argv[argv.index("-n") + 1])
    if "-s" in argv:
        seed = int(argv[argv.index("-s") + 1])
    R.seed(seed)
    # the agx/tests and sail/agx/tests instructions with an encoding
    tests = []
    for tdir in (os.path.join(agxtest.ROOT, "agx", "tests"), os.path.join(HERE, "tests")):
        for f in sorted(os.listdir(tdir)):
            for kind, toks, _ in agxtest.statements(open(os.path.join(tdir, f)).read()):
                if kind == "I" and not (toks[1] == "JMP" and toks[3] == "JL"):
                    tests.append(toks[1:])
    ok, bad, noenc = check_encode(tests, verbose)
    print("encode (agx/tests, sail/agx/tests): %d agree with applegpu, %d mismatches, %d without encoding" % (ok, bad, noenc))
    rnd = [rinstr() for _ in range(n)]
    ok2, bad2, noenc2 = check_encode(rnd, verbose)
    print("encode (%d random instructions): %d agree with applegpu, %d mismatches, %d without encoding" % (n, ok2, bad2, noenc2))
    print("  (%d select operands u128..u255 compared modulo applegpu's CmpselSrcDesc.decode, which drops bit 8 of a uniform index)" % KNOWN[0])
    agree, rejected, mismatch, other = check_decode(n, verbose)
    print("decode (%d random words): %d agree with applegpu, %d rejected by the Sail decoder, %d mismatches, %d outside the model"
          % (n, agree, rejected, mismatch, other))
    return 0 if bad == 0 and bad2 == 0 and mismatch == 0 and noenc2 == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

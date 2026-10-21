#!/usr/bin/env python3
"""Driver for the AGX Sail model (sail/agx/build/agx).

Parses agx/tests/*.agx (the assembly dialect of agx/agx.k), serializes the program into the token
stream that sail/agx/agx_main.sail reads, runs the model and reports PASS / FAIL like
scripts/run-agx.sh: a test passes when the run ends in status `stopped` (or the status a
`.expectstatus S` directive names, e.g. `trapped`) with exit code 0 (every .expect* directive held);
`.expectstatus fault` expects the model to stop with an error (exit code 1), for forms it does not define;
`.expectmsg TEXT` requires TEXT in the final message.

  agxtest.py [tests...]          run the tests (default: agx/tests/*.agx)
  agxtest.py --roundtrip [...]   also check, for every instruction, that the assembly printed by the
                                 Sail `assembly` mapping parses back to the same instruction and that
                                 `encdec` decodes its encoding back to it
"""
import os, re, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EXE = os.environ.get("AGX_SAIL", os.path.join(HERE, "build", "agx"))

CCS = ["eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge", "feq", "fne", "flt", "fle", "fgt", "fge",
       "seq", "sne", "fltn", "fgtn", "fnequ", "fnlt", "fngt", "fnle", "fnge", "fnltn", "fngtn"]
AOPS = ["add", "sub", "xchg", "min", "imin", "max", "imax", "and", "or", "xor"]
# device atomics only: the G14C forms of bits 45-46 and ops 11-15 (agx_types.sail aaop); default mode (bits 45-46)
DEV_AOPS = AOPS + ["cmpxchg", "fadd", "umin64", "umax64", "end11", "end12", "end13", "end14", "end15"]
ATOM_MODE_DEFAULT = {"fadd": 1, "umax64": 1, "umin64": 3}
FMTS = ["u8", "u16", "u32"]
FUNOPS = ["floor", "ceil", "trunc", "rint", "rsqrt", "rsqrt_special", "sin_pt_1", "sin_pt_2", "log2", "exp2"]

class ParseError(Exception):
    pass

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n\r]*", " ", text)

REG_RE = re.compile(r"^([ru])(\d+)([lh])?(?:_([ru])(\d+))?((?:\.(?:discard|cache|sx|abs|neg))*)$")

def reg(s):
    """register -> the 7 integers of the serialized areg"""
    m = REG_RE.match(s)
    if not m:
        raise ParseError("bad register %r" % s)
    f = 1 if m.group(1) == "u" else 0
    n = int(m.group(2))
    view = {None: 0, "l": 1, "h": 2}[m.group(3)]
    if m.group(4):
        if m.group(3) or m.group(4) != m.group(1) or int(m.group(5)) != n + 1:
            raise ParseError("bad register pair %r" % s)
        view = 3
    flags = m.group(6).split(".")[1:]
    hint = 1 if "cache" in flags else 2 if "discard" in flags else 0
    return [f, n, view, hint, int("sx" in flags), int("abs" in flags), int("neg" in flags)]

def is_reg(s):
    return REG_RE.match(s) is not None

INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*([eE][+-]?\d+)?|\d+[eE][+-]?\d+|\.\d+([eE][+-]?\d+)?|Infinity|NaN)[fFdD]?$")

def f64bits(s):
    s = s.rstrip("fFdD") if not s.endswith("Infinity") else s
    return struct.unpack("<Q", struct.pack("<d", float(s)))[0]

def op(s):
    s = s.strip()
    if is_reg(s):
        return ["R"] + reg(s)
    if INT_RE.match(s):
        return ["I", int(s)]
    if FLOAT_RE.match(s):
        return ["F", f64bits(s)]
    raise ParseError("bad operand %r" % s)

def pair(s):
    m = re.match(r"^([ru])(\d+)_([ru])(\d+)$", s.strip())
    if not m or m.group(1) != m.group(3):
        raise ParseError("bad register pair %r" % s)
    return [1 if m.group(1) == "u" else 0, int(m.group(2)), int(m.group(4))]

def intv(s):
    s = s.strip()
    if not INT_RE.match(s):
        raise ParseError("expected an integer: %r" % s)
    return int(s)

def mask_field(s):
    if s.startswith("mask="):
        return int(s[5:])
    n = intv(s)
    return (1 << n) - 1

def instr(stmt):
    """one instruction statement -> list of tokens (without the leading I)"""
    parts = [p.strip() for p in stmt.split(",")]
    head = parts[0].split()
    mnem, pre = head[0], head[1:]
    ops = pre[-1:] + parts[1:] if pre else parts[1:]
    pre = pre[:-1] if pre else []
    def want(n):
        if len(ops) < n:
            raise ParseError("too few operands in %r" % stmt)
    base, sat = (mnem[:-4], 1) if mnem.endswith(".sat") else (mnem, 0)
    def shift_of(rest):
        if not rest:
            return 0
        if len(rest) == 1 and rest[0].startswith("lsl "):
            return int(rest[0][4:])
        raise ParseError("bad trailing operands in %r" % stmt)
    if base in ("mov",):
        want(2); return ["MOV"] + reg(ops[0]) + op(ops[1])
    if base == "get_sr":
        want(2); return ["GET_SR"] + reg(ops[0]) + [intv(ops[1])]
    if base in ("iadd", "isub"):
        want(3); return ["IADD", int(base == "isub"), sat] + reg(ops[0]) + op(ops[1]) + op(ops[2]) + [shift_of(ops[3:])]
    if base in ("imadd", "imsub"):
        want(4); return ["IMADD", int(base == "imsub"), sat] + reg(ops[0]) + op(ops[1]) + op(ops[2]) + op(ops[3]) + [shift_of(ops[4:])]
    if base in ("bitop", "bfi", "bfeil"):
        if len(pre) != 1:
            raise ParseError("missing field in %r" % stmt)
        if base == "bitop":
            want(3); return ["BITOP", intv(pre[0])] + reg(ops[0]) + op(ops[1]) + op(ops[2])
        want(4); return [base.upper(), intv(pre[0])] + reg(ops[0]) + op(ops[1]) + op(ops[2]) + op(ops[3])
    if base == "asr":
        want(3); return ["ASR"] + reg(ops[0]) + op(ops[1]) + op(ops[2])
    if base in ("popcount", "bitrev", "ffs"):
        want(2); return ["IUNOP", base] + reg(ops[0]) + op(ops[1])
    m = re.match(r"^(fadd|fmul|fmadd)(16)?$", base)
    if m:
        h = int(m.group(2) is not None)
        if m.group(1) == "fmadd":
            want(4); return ["FMADD", h, sat] + reg(ops[0]) + op(ops[1]) + op(ops[2]) + op(ops[3])
        want(3); return ["FBIN", m.group(1)[1:], h, sat] + reg(ops[0]) + op(ops[1]) + op(ops[2])
    if base == "rcp":
        want(2); return ["RCP", sat] + reg(ops[0]) + op(ops[1])
    if base in FUNOPS:
        want(2); return ["FUNOP", base, sat] + reg(ops[0]) + op(ops[1])
    if mnem in ("icmpsel", "fcmpsel"):
        if len(pre) != 1 or pre[0] not in CCS:
            raise ParseError("bad condition in %r" % stmt)
        want(5); return ["CMPSEL", int(mnem[0] == "f"), pre[0]] + reg(ops[0]) + op(ops[1]) + op(ops[2]) + op(ops[3]) + op(ops[4])
    m = re.match(r"^(if|else|while)_([if])cmp$", mnem)
    if m:
        if len(head) < 2 or head[1] not in CCS:
            raise ParseError("bad condition in %r" % stmt)
        # here the head is "if_icmp CC A": the first operand is in ops[0]
        want(2)
        n = intv(ops[2]) if len(ops) > 2 else 1
        return ["STACKOP", m.group(1), int(m.group(2) == "f"), head[1]] + op(ops[0]) + op(ops[1]) + [n]
    if mnem == "pop_exec":
        return ["POP_EXEC", intv(ops[0])]
    if mnem in ("jmp_exec_any", "jmp_exec_none"):
        t = ops[0]
        any_ = int(mnem.endswith("any"))
        m = re.match(r"^pc([+-])(\d+)$", t)
        if m:
            return ["JMP", any_, "JR", int(m.group(2)) * (1 if m.group(1) == "+" else -1)]
        if not re.match(r"^[A-Z][A-Za-z0-9_]*$", t):
            raise ParseError("bad label %r" % t)
        return ["JMP", any_, "JL", t]
    if mnem in ("icmp_ballot", "fcmp_ballot"):
        if len(pre) != 1 or pre[0] not in CCS:
            raise ParseError("bad condition in %r" % stmt)
        want(3); return ["BALLOT", int(mnem[0] == "f"), pre[0]] + reg(ops[0]) + op(ops[1]) + op(ops[2])
    m = re.match(r"^simd_shuffle(_xor|_up|_down)?$", mnem)
    if m:
        want(3); return ["SHUFFLE", (m.group(1) or "_idx")[1:]] + reg(ops[0]) + op(ops[1]) + op(ops[2])
    if mnem in ("device_load", "device_store"):
        if len(pre) != 2 or pre[0] not in FMTS:
            raise ParseError("bad format in %r" % stmt)
        want(5)
        extra = ops[5:]
        g = int("g1" in extra); u6 = int("u6" in extra)
        if any(x not in ("g1", "u6") for x in extra):
            raise ParseError("bad trailing operands in %r" % stmt)
        if ops[3] not in ("signed", "unsigned"):
            raise ParseError("bad signedness in %r" % stmt)
        return ["DEVMEM", int(mnem.endswith("store")), pre[0], mask_field(pre[1])] + reg(ops[0]) + pair(ops[1]) + op(ops[2]) + \
               [int(ops[3] == "signed"), intv(ops[4]), g, u6]
    if mnem in ("threadgroup_load", "threadgroup_store"):
        if len(pre) != 2 or pre[0] not in FMTS:
            raise ParseError("bad format in %r" % stmt)
        want(3)
        base = ["I", 0]
        for x in ops[3:]:
            if x.startswith("base="):
                base = op(x[5:])
            else:
                raise ParseError("bad trailing operands in %r" % stmt)
        return ["TGMEM", int(mnem.endswith("store")), pre[0], mask_field(pre[1])] + reg(ops[0]) + base + op(ops[1]) + [intv(ops[2])]
    if mnem in ("device_atomic", "threadgroup_atomic"):
        if len(pre) != 1 or pre[0] not in (DEV_AOPS if mnem == "device_atomic" else AOPS):
            raise ParseError("bad atomic op in %r" % stmt)
        dst = ["N"] if ops[0] == "_" else ["S"] + reg(ops[0])
        if mnem == "device_atomic":
            want(4)
            extra = ops[4:]
            mode = ATOM_MODE_DEFAULT.get(pre[0], 0)
            for x in extra:
                if x.startswith("mode="):
                    mode = intv(x[5:])
                elif x not in ("signed", "g1"):
                    raise ParseError("bad trailing operands in %r" % stmt)
            return ["DEVATOM", pre[0]] + dst + pair(ops[1]) + op(ops[2]) + [int("signed" in extra)] + op(ops[3]) + \
                [int("g1" in extra), mode]
        want(3)
        base = ["I", 0]
        for x in ops[3:]:
            if x.startswith("base="):
                base = op(x[5:])
            else:
                raise ParseError("bad trailing operands in %r" % stmt)
        return ["TGATOM", pre[0]] + dst + base + op(ops[1]) + op(ops[2])
    if mnem == "wait":
        return ["WAIT", intv(ops[0])]
    if mnem == "threadgroup_barrier":
        return ["TG_BARRIER"]
    if mnem == "memory_barrier":
        want(3); return ["MEMORY_BARRIER", intv(ops[0]), intv(ops[1]), intv(ops[2])]
    if mnem == "stop":
        return ["STOP"]
    if mnem == "trap":
        return ["TRAP"]
    raise ParseError("unknown instruction %r" % stmt)

def directive(stmt):
    t = stmt.replace("=", " = ").split()
    d = t[0][1:]
    try:
        if d == "all":      return ["D", "all"] + reg(t[1]) + [intv(t[3])]
        if d == "laneid":   return ["D", "laneid"] + reg(t[1])
        if d == "lane":     return ["D", "lane", intv(t[1])] + reg(t[2]) + [intv(t[4])]
        if d == "uni":      return ["D", "uni"] + reg(t[1]) + [intv(t[3])]
        if d == "sr" and t[3] == "laneid": return ["D", "sr", intv(t[1])]
        if d == "srall":    return ["D", "srall", intv(t[1]), intv(t[3])]
        if d == "dev":      return ["D", "dev", intv(t[1]), intv(t[3])]
        if d == "run":      return ["D", "run"]
        if d == "expectall": return ["D", "expectall"] + reg(t[1]) + [intv(t[3])]
        if d == "expectlane": return ["D", "expectlane", intv(t[1])] + reg(t[2]) + [intv(t[4])]
        if d == "expectdev": return ["D", "expectdev", intv(t[1]), intv(t[3])]
        if d == "expectmask": return ["D", "expectmask", intv(t[1])]
        # the final status the run must end in (default: stopped); checked by this driver, not sent to the model
        if d == "expectstatus" and len(t) == 2 and re.match(r"^[a-z]+$", t[1]): return ["S", t[1]]
        # text the final message must contain (with .expectstatus fault: which error)
        if d == "expectmsg" and len(t) >= 2: return ["S", "msg", stmt.split(None, 1)[1]]
    except IndexError:
        pass
    raise ParseError("bad directive %r" % stmt)

def statements(text):
    """the program as a list of (kind, tokens, source text)"""
    out = []
    for raw in strip_comments(text).split(";"):
        s = " ".join(raw.split())
        if not s:
            continue
        m = re.match(r"^([A-Z][A-Za-z0-9_]*)\s*:\s*(.*)$", s)
        if m:
            out.append(("L", ["L", m.group(1)], s))
            s = m.group(2)
            if not s:
                continue
        if s.startswith("."):
            d = directive(s)
            out.append(("S" if d[0] == "S" else "D", d, s))
        else:
            out.append(("I", ["I"] + instr(s), s))
    return out

def run_tokens(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".tok", delete=False) as f:
        f.write("\n".join(" ".join(str(x) for x in l) for l in lines) + "\nE\n")
        name = f.name
    try:
        p = subprocess.run([EXE], env={**os.environ, "SAIL_INPUT": name}, capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(name)
    return p.returncode, p.stdout, p.stderr

def status_of(out):
    for line in out.splitlines():
        if line.startswith("STATUS "):
            m = re.match(r"STATUS (\S+) EXIT (\d+) MSG ?(.*)", line)
            return m.group(1), int(m.group(2)), m.group(3)
    return None, None, ""

def expected_status(prog):
    s = [toks[1] for kind, toks, _ in prog if kind == "S" and toks[1] != "msg"]
    return s[-1] if s else "stopped"

def msg_ok(prog, msg):
    return all(toks[2] in msg for kind, toks, _ in prog if kind == "S" and toks[1] == "msg")

def run_test(path, roundtrip=False):
    prog = statements(open(path).read())
    lines = [toks for kind, toks, _ in prog if kind != "S"]
    want = expected_status(prog)
    if roundtrip:
        # the encoding check (X) before the run, so that it also covers programs whose run ends in a fault
        run_at = [i for i, l in enumerate(lines) if l[:2] == ["D", "run"]]
        lines.insert(run_at[0] if run_at else len(lines), ["X"])
    code, out, err = run_tokens(lines)
    st, ex, msg = status_of(out)
    ok = code == 0 and st == want and ex == (1 if want == "fault" else 0) and msg_ok(prog, msg)
    notes = []
    if roundtrip:
        instrs = [(toks, src) for kind, toks, src in prog if kind == "I"]
        printed = [l for l in out.splitlines() if l.startswith(("ENC ", "NOENC "))]
        nenc = nrt = nasm = 0
        if len(printed) != len(instrs):
            notes.append("instruction count mismatch")
        # byte layout (as agx_main.sail computes it): encoding length, 6 for jumps, 8 without encoding
        addr, a = [], 0
        for (toks, src), line in zip(instrs, printed):
            addr.append(a)
            a += 6 if toks[1] == "JMP" else len(line.split()[1]) // 2 if line.startswith("ENC ") else 8
        addr.append(a)
        label_addr, k = {}, 0
        for kind, toks, src in prog:
            if kind == "L":
                label_addr[toks[1]] = addr[k] if k < len(addr) else a
            elif kind == "I":
                k += 1
        for idx, ((toks, src), line) in enumerate(zip(instrs, printed)):
            if toks[1] == "JMP" and toks[3] == "JL" and toks[4] in label_addr:
                toks = toks[:3] + ["JR", label_addr[toks[4]] - addr[idx]]
            if line.startswith("ENC "):
                nenc += 1
                body = line.split(" ", 2)[2]
                asm, rt = body.rsplit(" ", 1) if body.endswith("RT-OK") else (body.split(" RT-FAIL ")[0], "RT-FAIL")
                if rt == "RT-OK":
                    nrt += 1
                else:
                    notes.append("encoding round trip failed: %s" % line)
            else:
                asm = line[len("NOENC "):]
            try:
                back = ["I"] + instr(asm)
            except ParseError as e:
                back = None
                notes.append("printed assembly does not parse: %r (%s)" % (asm, e))
            if back is not None:
                if [str(x) for x in back] == [str(x) for x in toks]:
                    nasm += 1
                else:
                    notes.append("assembly round trip: %r printed as %r" % (src, asm))
        notes.insert(0, "%d instructions, assembly round trip %d/%d, encodable %d, encoding round trip %d/%d"
                     % (len(instrs), nasm, len(instrs), nenc, nrt, nenc))
        ok = ok and nasm == len(instrs) and nrt == nenc
    return ok, st, ex, msg + (err.strip()[-300:] if code != 0 else ""), notes

def run_binary(path):
    """encode the program with the Sail model, then run it again from its binary encoding (decoded by
    encdec): only for programs whose instructions all have an encoding"""
    prog = statements(open(path).read())
    code, out, err = run_tokens([t for k, t, _ in prog if k in ("I", "L")] + [["X"]])
    printed = [l for l in out.splitlines() if l.startswith(("ENC ", "NOENC "))]
    if any(l.startswith("NOENC") for l in printed):
        return None
    blob = bytes.fromhex("".join(l.split()[1] for l in printed))
    lines, placed = [], False
    for kind, toks, _ in prog:
        if kind == "I" or kind == "L":
            if not placed:
                lines.append(["C", len(blob), int.from_bytes(blob, "little")])
                placed = True
        elif kind != "S":
            lines.append(toks)
    code, out, err = run_tokens(lines)
    st, ex, msg = status_of(out)
    want = expected_status(prog)
    return code == 0 and st == want and ex == (1 if want == "fault" else 0) and msg_ok(prog, msg), len(blob), msg

def main(argv):
    if "--binary" in argv:
        tests = [a for a in argv if not a.startswith("--")] or sorted(
            os.path.join(ROOT, "agx", "tests", f) for f in os.listdir(os.path.join(ROOT, "agx", "tests")) if f.endswith(".agx"))
        bad = 0
        for t in tests:
            r = run_binary(t)
            rel = os.path.relpath(t, ROOT)
            if r is None:
                print("SKIP  %s (not every instruction has an encoding)" % rel)
            elif r[0]:
                print("PASS  %s (run from its %d-byte binary encoding)" % (rel, r[1]))
            else:
                bad += 1
                print("FAIL  %s (binary) %s" % (rel, r[2]))
        return 1 if bad else 0
    roundtrip = "--roundtrip" in argv
    tests = [a for a in argv if not a.startswith("--")] or sorted(
        os.path.join(ROOT, "agx", "tests", f) for f in os.listdir(os.path.join(ROOT, "agx", "tests")) if f.endswith(".agx"))
    npass = nfail = 0
    for t in tests:
        rel = os.path.relpath(t, ROOT)
        try:
            ok, st, ex, msg, notes = run_test(t, roundtrip)
        except ParseError as e:
            ok, st, ex, msg, notes = False, "parse-error", 1, str(e), []
        if ok:
            npass += 1
            print("PASS  %s" % rel)
        else:
            nfail += 1
            print("FAIL  %s (exit %s, status %s) %s" % (rel, ex, st, msg))
        for n in notes:
            print("      " + n)
    print("---- %d passed, %d failed" % (npass, nfail))
    return 0 if nfail == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

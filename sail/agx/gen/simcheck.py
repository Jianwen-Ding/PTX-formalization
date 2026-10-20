#!/usr/bin/env python3
"""Differential test of the AGX functional simulator (agx_run) against the AGX Sail model.

Generates random programs over the instructions of the Sail model (sail/agx/gen/agxgen.py), encodes
them with the Sail model's `encdec`, and runs the same binary on

  * the Sail model (decoded from the binary, `C` token of agx_main.sail), and
  * agx_run (AGX-Performance-Simulator/rust/target/release/agx_run), one SIMD group of 32 threads,

then compares the final device memory: every program stores all its registers to an output buffer at
the end, and memory instructions access a scratch buffer that is compared too.

Program layout (registers):
  r0l     execution-mask stack counter          r1      lane id (get_sr sr80)
  r2, r3  prolog scratch                        r4..r35 data, loaded from the input buffer
  r36     output index                          r37     memory index scratch
  r38_r39 scratch buffer base                   r40..r43 loop counters
  u0_u1 input buffer, u2_u3 output buffer, u4_u5 scratch buffer, u6..u15 random values

usage: simcheck.py [-s SEED] [-n PROGRAMS] [-l LENGTH] [--kinds alu,cf,mem] [-v] [--keep DIR] [--only CLASS]
"""
import argparse, os, random, struct, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import agxgen  # noqa: E402
import agxtest  # noqa: E402

AGX_RUN = os.environ.get("AGX_RUN", "/home/user/AGX-Performance-Simulator/rust/target/release/agx_run")
IN, OUT, MEM = 0x10000, 0x20000, 0x30000
NIN = 32            # words per lane in the input buffer (r4..r35)
NOUT = 44           # words per lane stored (r0..r43)
MEMSZ = 4096
SHARED = 4096
DATA = list(range(4, 36))
UNI = list(range(6, 16))
# special registers with the same value in both (grid 32, workgroup 32): name -> value per lane
SRS = {0: lambda t: 0, 1: lambda t: 0, 2: lambda t: 0, 4: lambda t: 32, 5: lambda t: 1, 6: lambda t: 1,
       48: lambda t: t, 49: lambda t: 0, 50: lambda t: 0, 51: lambda t: t, 52: lambda t: t, 53: lambda t: 0,
       80: lambda t: t, 81: lambda t: 0, 82: lambda t: 0}

F32_EDGE = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00000, 0xffc00001,
            0x00000001, 0x807fffff, 0x00800000, 0x7f7fffff, 0xff7fffff, 0x3f000000, 0x40490fdb, 0x4b000000, 0x3effffff,
            0x33800000, 0x00400000]
F16_EDGE = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x0001, 0x83ff, 0x0400, 0x7bff, 0xfbff, 0x3800,
            0x3555, 0x1400]
I_EDGE = [0, 1, 2, 3, 0x7fffffff, 0x80000000, 0xffffffff, 0xfffffffe, 0x0000ffff, 0x00010000, 0x00008000, 0xffff7fff,
          0x000000ff, 0x80008000, 0x55555555, 0xaaaaaaaa, 31, 32, 33, 64]


def reg(f, n, view=0, hint=0):
    return [f, n, view, hint, 0, 0, 0]


def RG(n, view=0):
    return ["R"] + reg(0, n, view)


def mov_imm(r, v, view=0):
    return ["MOV"] + reg(0, r, view) + ["I", v]


def imadd(d, a, b, c):
    return ["IMADD", 0, 0] + reg(0, d) + RG(a) + ["I", b] + ["I", c] + [0]


def iand(d, a, imm, dview=0, aview=0):
    return ["BITOP", 8] + reg(0, d, dview) + RG(a, aview) + ["I", imm]


def dev(st, fmt, mask, r, view, pair, idx, sg=0, s=0, g=0, u6=0):
    return ["DEVMEM", st, fmt, mask] + reg(0, r, view) + pair + idx + [sg, s, g, u6]


class ProgGen:
    def __init__(self, rng, kinds, only=None):
        self.R = rng
        self.kinds = kinds
        self.g = agxgen.Gen(rng, regs=DATA, uregs=UNI)
        self.only = only
        self.depth = 0
        self.loops = 0
        alu = ["mov_imm", "mov", "get_sr", "iadd", "isub", "iadd.sat", "isub.sat", "imadd", "imsub", "imadd.sat", "imsub.sat", "bitop",
               "bfi", "bfeil", "asr", "popcount", "bitrev", "ffs", "fadd32", "fadd16", "fmul32", "fmul16", "fmadd32", "fmadd16", "rcp",
               "icmpsel", "fcmpsel", "icmp_ballot", "fcmp_ballot", "simd_shuffle", "simd_shuffle_xor", "simd_shuffle_up",
               "simd_shuffle_down"]
        self.alu = [a for a in alu if only is None or a in only]

    # one data instruction, operands restricted to the data registers
    def alu_instr(self):
        name = self.R.choice(self.alu)
        dims, build = agxgen.SCHEMAS[name]
        c = agxgen._choose(dims, self.R)
        if name == "get_sr":
            c["sr"] = self.R.choice(sorted(SRS))
        t = build(c, self.g)
        if t[0] == "IADD" or t[0] == "IMADD":
            pass
        return t

    def fix_pairs(self, t):
        """64-bit register pairs must stay inside r4..r35"""
        return t

    def mem_instr(self):
        R = self.R
        out = []
        k = R.random()
        if k < 0.45:
            st = R.randrange(2)
            fmt = R.choice(agxgen.FMTS)
            view = R.choice((0, 1, 2)) if fmt != "u32" else R.choice((0, 0, 1, 2))
            mask = R.randrange(1, 16)
            n = bin(mask).count("1")
            if view == 0:
                r = R.randrange(4, 36 - n + 1)
            else:
                r = R.randrange(4, 36 - (n + 1) // 2)
            pair = [1, 4, 5] if R.random() < 0.5 else [0, 38, 39]
            if R.random() < 0.5:
                idx = ["I", R.randrange(0, 64)]
            else:
                out.append(iand(37, R.choice(DATA), 63))
                idx = RG(37)
            g = R.randrange(2)
            out.append(dev(st, fmt, mask, r, view, pair, idx, sg=R.randrange(2), s=R.randrange(3), g=g, u6=R.randrange(2) if st else 0))
            if not st:
                out.append(["WAIT", g])
        elif k < 0.65:
            # every device atomic but ops 11-15 (they end the program: atomic-end.agx); the 64-bit min / max at
            # 8-byte aligned words (even index) and without the changed flag (its value when the source equals
            # memory is not measured, and the model stops there), fadd always with a destination (red.add.f32
            # is not measured)
            op = R.choice([o for o in agxgen.DEV_AOPS if not o.startswith("end")])
            wide = op in ("umin64", "umax64")
            pair = [1, 4, 5] if R.random() < 0.5 else [0, 38, 39]
            if R.random() < 0.5:
                idx = ["I", R.randrange(0, 64, 2 if wide else 1)]
            else:
                out.append(iand(37, R.choice(DATA), 62 if wide else 63))
                idx = RG(37)
            g = R.randrange(2)
            d = ["S"] + reg(0, R.choice(DATA)) if (R.random() < 0.8 or op == "fadd") and not wide else ["N"]
            v = RG(R.choice(DATA[:-1]), 3) if op in agxgen.DEV_PAIR else RG(R.choice(DATA))
            out.append(["DEVATOM", op] + d + pair + idx + [R.randrange(2)] + v + [g, R.choice(agxgen.DEV_MODES[op])])
            out.append(["WAIT", g])
        elif k < 0.85:
            st = R.randrange(2)
            fmt = R.choice(agxgen.FMTS)
            view = R.choice((0, 1, 2))
            mask = R.randrange(1, 16)
            n = bin(mask).count("1")
            r = R.randrange(4, 36 - n + 1) if view == 0 else R.randrange(4, 36 - (n + 1) // 2)
            if R.random() < 0.5:
                base = ["I", 0]
            else:
                out.append(mov_imm(37, R.randrange(0, 256, 4), 2))
                base = RG(37, 2)
            if R.random() < 0.5:
                idx = ["I", R.randrange(0, 32)]
            else:
                out.append(iand(37, R.choice(DATA), 31, 1, 1))
                idx = RG(37, 1)
            out.append(["TGMEM", st, fmt, mask] + reg(0, r, view) + base + idx + [0])
            if not st:
                out.append(["WAIT", 0])
        else:
            op = R.choice(agxgen.AOPS)
            if R.random() < 0.5:
                base = ["I", 0]
            else:
                out.append(mov_imm(37, R.randrange(0, 256, 4), 2))
                base = RG(37, 2)
            if R.random() < 0.5:
                idx = ["I", R.randrange(0, 32)]
            else:
                out.append(iand(37, R.choice(DATA), 31, 1, 1))
                idx = RG(37, 1)
            d = ["S"] + reg(0, R.choice(DATA), R.choice((0, 0, 1, 2))) if R.random() < 0.8 else ["N"]
            out.append(["TGATOM", op] + d + base + idx + RG(R.choice(DATA)))
        if R.random() < 0.2:
            out.append(["TG_BARRIER"])
        return out

    def cond(self, kind_f=None):
        R = self.R
        f = R.randrange(2) if kind_f is None else kind_f
        cc = R.choice(agxgen.FCCS if f else agxgen.ICCS)
        kinds = agxgen.FLT_SRC if f else agxgen.INT_SRC
        a = self.g.src(R.choice(kinds), hint=0)
        b = self.g.src(R.choice(kinds), hint=0)
        return f, cc, a, b

    def block(self, n):
        out = []
        while n > 0:
            k = self.R.random()
            if "cf" in self.kinds and self.depth < 3 and k < 0.12:
                out += self.if_block(n // 2 + 1)
                n -= n // 2 + 1
            elif "cf" in self.kinds and self.depth < 3 and self.loops < 4 and k < 0.18:
                out += self.loop_block(n // 2 + 1)
                n -= n // 2 + 1
            elif "mem" in self.kinds and k < 0.35:
                out += self.mem_instr()
                n -= 1
            elif "alu" in self.kinds:
                out.append(self.alu_instr())
                n -= 1
            else:
                n -= 1
        return out

    def if_block(self, n):
        self.depth += 1
        R = self.R
        f, cc, a, b = self.cond()
        out = [["STACKOP", "if", f, cc] + a + b + [1]]
        skip = R.random() < 0.3
        lab = "L%d" % R.randrange(1 << 30)
        if skip:
            out.append(["JMP", 0, "JL", lab])
        out += self.block(max(1, n // 2))
        if R.random() < 0.6:
            f2, cc2, a2, b2 = self.cond()
            if skip:
                out.append(["LABEL", lab])
                skip = False
            out.append(["STACKOP", "else", f2, cc2] + a2 + b2 + [1])
            out += self.block(max(1, n - n // 2))
        if skip:
            out.append(["LABEL", lab])
        out.append(["POP_EXEC", 1])
        self.depth -= 1
        return out

    def loop_block(self, n):
        self.depth += 1
        R = self.R
        c = 40 + self.loops
        self.loops += 1
        lab = "W%d" % R.randrange(1 << 30)
        out = [iand(c, R.choice(DATA), 3), ["IADD", 0, 0] + reg(0, c) + RG(c) + ["I", 1] + [0]]
        # push (always true for the active lanes)
        out.append(["STACKOP", "if", 0, "eq"] + ["I", 0] + ["I", 0] + [1])
        out.append(["LABEL", lab])
        out += self.block(n)
        out.append(["IADD", 1, 0] + reg(0, c) + RG(c) + ["I", 1] + [0])
        out.append(["STACKOP", "while", 0, "ugt"] + RG(c) + ["I", 0] + [1])
        out.append(["JMP", 1, "JL", lab])
        out.append(["POP_EXEC", 1])
        self.depth -= 1
        return out


def prolog():
    out = [["GET_SR"] + reg(0, 1) + [80]]
    for k in range(NIN // 4):
        out.append(imadd(2, 1, NIN, 4 * k))
        out.append(dev(0, "u32", 15, 4 + 4 * k, 0, [1, 0, 1], RG(2)))
    out.append(["WAIT", 0])
    out.append(mov_imm(38, MEM))
    out.append(mov_imm(39, 0))
    return out


def epilog():
    out = []
    for k in range(NOUT // 4):
        out.append(imadd(36, 1, NOUT, 4 * k))
        out.append(dev(1, "u32", 15, 4 * k, 0, [1, 2, 3], RG(36)))
    out.append(["STOP"])
    return out


def assemble(prog):
    """encode with the Sail model; resolve labels; returns (bytes, list of (tokens, hex)) or raises"""
    instrs = [t for t in prog if t[0] != "LABEL"]
    probe = [["JMP", t[1], "JR", 0] if t[0] == "JMP" else t for t in instrs]
    code, out, err = agxtest.run_tokens([["I"] + t for t in probe] + [["X"]])
    printed = [l for l in out.splitlines() if l.startswith(("ENC ", "NOENC "))]
    if len(printed) != len(probe):
        raise RuntimeError("encode failed: " + err[-300:])
    sizes = []
    for t, l in zip(probe, printed):
        if l.startswith("NOENC"):
            raise RuntimeError("no encoding: " + l)
        sizes.append(len(l.split()[1]) // 2)
    addr, a, labels, i = [], 0, {}, 0
    for t in prog:
        if t[0] == "LABEL":
            labels[t[1]] = a
            continue
        addr.append(a)
        a += sizes[i]
        i += 1
    final = []
    for j, t in enumerate(instrs):
        if t[0] == "JMP":
            final.append(["JMP", t[1], "JR", labels[t[3]] - addr[j]])
        else:
            final.append(t)
    code, out, err = agxtest.run_tokens([["I"] + t for t in final] + [["X"]])
    printed = [l for l in out.splitlines() if l.startswith(("ENC ", "NOENC "))]
    hexs = [l.split()[1] for l in printed]
    return bytes.fromhex("".join(hexs)), list(zip(final, hexs))


def inputs(rng):
    words = []
    for t in range(32):
        for k in range(NIN):
            x = rng.random()
            if x < 0.25:
                w = rng.getrandbits(32)
            elif x < 0.45:
                w = rng.choice(F32_EDGE)
            elif x < 0.6:
                w = rng.choice(F16_EDGE) | rng.choice(F16_EDGE) << 16
            elif x < 0.75:
                w = rng.choice(I_EDGE)
            elif x < 0.9:
                w = struct.unpack("<I", struct.pack("<f", rng.choice([0.5, 1.5, 2.0, -3.25, 100.0, 0.1, 1e-3, 65504.0, 1e30, -1e-30])))[0]
            else:
                w = rng.randrange(64)
            words.append(w)
    mem = [rng.getrandbits(32) for _ in range(MEMSZ // 4)]
    uni = {u: rng.choice([rng.getrandbits(32), rng.choice(F32_EDGE), rng.choice(I_EDGE)]) for u in UNI}
    return words, mem, uni


def run_sail(code, words, mem, uni):
    toks = []
    for i, (lo, hi) in enumerate([(IN, 0), (OUT, 0), (MEM, 0)]):
        toks.append(["D", "uni"] + reg(1, 2 * i) + [lo])
        toks.append(["D", "uni"] + reg(1, 2 * i + 1) + [hi])
    for u, v in uni.items():
        toks.append(["D", "uni"] + reg(1, u) + [v])
    for n, f in SRS.items():
        for t in range(32):
            pass
    for i, w in enumerate(words):
        toks.append(["D", "dev", IN + 4 * i, w])
    for i, w in enumerate(mem):
        toks.append(["D", "dev", MEM + 4 * i, w])
    # special registers: .srall for uniform values, .sr N = laneid for lane ids
    for n in SRS:
        v0, v1 = SRS[n](0), SRS[n](1)
        if v0 == v1:
            toks.append(["D", "srall", n, v0])
        else:
            toks.append(["D", "sr", n])
    toks.append(["C", len(code), int.from_bytes(code, "little")])
    toks.append(["D", "run"])
    toks.append(["D", "dumpdev", OUT, 32 * NOUT * 4])
    toks.append(["D", "dumpdev", MEM, MEMSZ])
    rc, out, err = agxtest.run_tokens(toks)
    st, ex, msg = agxtest.status_of(out)
    dumps = {}
    for l in out.splitlines():
        if l.startswith("DEVMEM "):
            _, a, h = l.split()
            dumps[int(a)] = bytes.fromhex(h)
    if st != "stopped":
        return None, "sail: status %s %s %s" % (st, msg, err.strip()[-200:])
    return dumps, None


def run_sim(code, words, mem, uni, tmp):
    fc, fi, fm, fo = (os.path.join(tmp, x) for x in ("code.bin", "in.bin", "mem.bin", "out.bin"))
    with open(fc, "wb") as f:
        f.write(code)
    with open(fi, "wb") as f:
        f.write(b"".join(struct.pack("<I", w) for w in words))
    with open(fm, "wb") as f:
        f.write(b"".join(struct.pack("<I", w) for w in mem))
    cmd = [AGX_RUN, fc, "--grid", "32", "--wg", "32", "--shared", str(SHARED), "--map", "%#x=%s" % (IN, fi), "--map", "%#x=%s" % (MEM, fm),
           "--alloc", "%#x=%d" % (OUT, 32 * NOUT * 4), "--dump", "%#x=%d:%s" % (OUT, 32 * NOUT * 4, fo), "--dump", "%#x=%d:%s" % (MEM, MEMSZ, fm),
           "--max-steps", "200000"]
    for i, v in enumerate([IN, 0, OUT, 0, MEM, 0]):
        cmd += ["--uniform", "%d=%d" % (i, v)]
    for u, v in uni.items():
        cmd += ["--uniform", "%d=%d" % (u, v)]
    # without the recorded transcendental tables agx_run computes the correctly rounded reciprocal the
    # model specifies (the hardware's rcp approximation is a documented divergence)
    env = dict(os.environ, AGX_TRANS_TABLES=os.path.join(tmp, "no-tables"))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if r.returncode != 0:
        return None, "agx_run: " + (r.stdout + r.stderr).strip()[-300:]
    return {OUT: open(fo, "rb").read(), MEM: open(fm, "rb").read()}, None


def compare(a, b):
    """list of (region, lane, word, sail, sim)"""
    diffs = []
    for base, n in ((OUT, NOUT), (MEM, None)):
        x, y = a[base], b[base]
        for i in range(0, min(len(x), len(y)), 4):
            u, v = int.from_bytes(x[i:i + 4], "little"), int.from_bytes(y[i:i + 4], "little")
            if u != v:
                if n:
                    diffs.append(("out", (i // 4) // n, "r%d" % ((i // 4) % n), u, v))
                else:
                    diffs.append(("mem", None, "+%d" % i, u, v))
    return diffs


def one(args):
    seed, length, kinds, only, keep = args
    rng = random.Random(seed)
    pg = ProgGen(rng, kinds, only)
    body = pg.block(length)
    prog = prolog() + body + epilog()
    try:
        code, listing = assemble(prog)
    except Exception as e:
        return seed, "gen-error", str(e), [], body
    words, mem, uni = inputs(rng)
    with tempfile.TemporaryDirectory() as tmp:
        sa, e1 = run_sail(code, words, mem, uni)
        si, e2 = run_sim(code, words, mem, uni, tmp)
        if keep:
            d = os.path.join(keep, "p%d" % seed)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "code.bin"), "wb").write(code)
            with open(os.path.join(d, "listing.txt"), "w") as f:
                for t, h in listing:
                    f.write("%-22s %s\n" % (h, agxgen.form_key(t)))
    if e1 or e2:
        return seed, "run-error", (e1 or "") + " " + (e2 or ""), listing, body
    diffs = compare(sa, si)
    return seed, "ok" if not diffs else "diff", diffs, listing, body


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int, default=1)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("-l", type=int, default=30)
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("--kinds", default="alu,cf,mem")
    ap.add_argument("--only", help="comma-separated instruction schemas (ALU)")
    ap.add_argument("--keep")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--tests", nargs="*", help="also run these Sail-dialect tests (registers r4..r35) on agx_run")
    a = ap.parse_args(argv)
    tbad = 0
    for f in a.tests or []:
        bad = run_test_file(f)
        tbad += bool(bad)
        print("%s  %s (agx_run) %s" % ("FAIL" if bad else "PASS", os.path.relpath(f), "; ".join(bad[:3])))
    kinds = set(a.kinds.split(","))
    only = set(a.only.split(",")) if a.only else None
    jobs = [(a.seed * 100000 + i, a.l, kinds, only, a.keep) for i in range(a.n)]
    with ThreadPoolExecutor(a.j) as ex:
        res = list(ex.map(one, jobs))
    counts = {}
    forms = set()
    for seed, st, d, listing, body in res:
        counts[st] = counts.get(st, 0) + 1
        for t, h in listing:
            forms.add(agxgen.form_key(t))
        if st != "ok" and (a.v or counts[st] <= 5):
            print("%s seed %d: %s" % (st.upper(), seed, d if st != "diff" else ""))
            if st == "diff":
                if a.v:
                    import mccheck
                    for t, h in listing[len(prolog()):len(listing) - len(epilog())]:
                        print("      %-20s %s" % (h, mccheck.llvm_text(t)))
                for x in d[:8]:
                    print("   %s lane %s %s: sail %#010x sim %#010x" % x)
    total = len(res)
    print("AGX simulator differential test: %d programs, %d agree (%.1f%%), %s; %d instruction forms executed"
          % (total, counts.get("ok", 0), 100.0 * counts.get("ok", 0) / max(total, 1),
             ", ".join("%d %s" % (v, k) for k, v in sorted(counts.items()) if k != "ok"), len(forms)))
    return 0 if counts.get("ok", 0) == total and not tbad else 1



def probe(lines, regs=None, uni=None, show=(4, 5, 6, 7)):
    """triage helper: run LLVM-syntax instructions (assembled by llvm-mc) between the prolog and the epilog with the
    given initial data registers {reg: value or [32 lane values]} and uniforms; print both results for `show`"""
    import mccheck
    body = b""
    for l in lines:
        h, err = mccheck.llvm_asm(l)
        if h is None:
            raise RuntimeError(err)
        body += bytes.fromhex(h)
    pro, _ = assemble(prolog())
    epi, _ = assemble(epilog())
    code = pro + body + epi
    rng = random.Random(0)
    words, mem, u = inputs(rng)
    for r, v in (regs or {}).items():
        for t in range(32):
            words[t * NIN + r - 4] = v[t] if isinstance(v, list) else v
    u.update(uni or {})
    with tempfile.TemporaryDirectory() as tmp:
        sa, e1 = run_sail(code, words, mem, u)
        si, e2 = run_sim(code, words, mem, u, tmp)
    if e1 or e2:
        print(e1, e2)
        return
    for r in show:
        a = [int.from_bytes(sa[OUT][(t * NOUT + r) * 4:(t * NOUT + r) * 4 + 4], "little") for t in range(32)]
        b = [int.from_bytes(si[OUT][(t * NOUT + r) * 4:(t * NOUT + r) * 4 + 4], "little") for t in range(32)]
        print("r%d sail %s" % (r, " ".join("%08x" % x for x in a[:8])), "| sim %s" % " ".join("%08x" % x for x in b[:8]),
              "" if a == b else " DIFFER")


def run_test_file(path):
    """run a Sail-dialect test (sail/agx/tests/*.agx whose registers are r4..r35, set by .all / .laneid / .lane) on
    agx_run and check its .expectall / .expectlane directives; returns a list of failures"""
    prog = agxtest.statements(open(path).read())
    body = [t[1:] for k, t, _ in prog if k == "I" and t[1] != "STOP"]
    vals = {r: [0] * 32 for r in DATA}
    checks = []
    for k, t, src in prog:
        if k != "D":
            continue
        d = t[1]
        if d in ("all", "laneid", "lane"):
            if d == "lane":
                r, lanes, v = t[4], [t[2]], t[10]
            else:
                r, lanes, v = t[3], range(32), (t[9] if d == "all" else None)
            for l in lanes:
                vals[r][l] = l if d == "laneid" else v
        elif d in ("expectall", "expectlane"):
            checks.append((src, t))
    code, _ = assemble(prolog() + body + epilog())
    rng = random.Random(0)
    words, mem, uni = inputs(rng)
    for r in DATA:
        for l in range(32):
            words[l * NIN + r - 4] = vals[r][l]
    with tempfile.TemporaryDirectory() as tmp:
        si, err = run_sim(code, words, mem, uni, tmp)
    if err:
        return [err]
    bad = []
    for src, t in checks:
        lanes = range(32) if t[1] == "expectall" else [t[2]]
        reg_t = t[2:9] if t[1] == "expectall" else t[3:10]
        want = t[-1]
        n, view = reg_t[1], reg_t[2]
        for l in lanes:
            w = int.from_bytes(si[OUT][(l * NOUT + n) * 4:(l * NOUT + n) * 4 + 4], "little")
            got = w if view == 0 else (w & 0xffff if view == 1 else w >> 16)
            if got != want & (0xffffffff if view == 0 else 0xffff):
                bad.append("%s: lane %d got %#x" % (src, l, got))
                break
    return bad


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Seeded random PTX program generator for differential testing of the Sail
model against GPGPU-Sim (see README.md).

    gen_random.py gen SEED [--cats int,float,...] [-o prog.ptx]
    gen_random.py fuzz [--cats ...] [--seeds A:B] [--out DIR] [run_compare options]
    gen_random.py min PROG.json [--out DIR]          (re-minimise a saved case)

A program is one kernel k(IN, OUT, SCR, SHO) run by NCTA CTAs of NT threads.
Each thread loads its typed registers from IN, executes a random statement
list, and stores every register into its OUT slots (8 bytes per register);
SCR is a per-thread 64-byte scratch area for ld/st/atom, SHO receives each
thread's 8-byte word of the CTA's shared array at the end.

Only programs whose result is fully specified by the PTX ISA are produced:
  * integer div/rem get a non-zero (and, for signed types, positive) divisor,
    signed rem a non-negative dividend (the ISA leaves the rest
    machine-specific);
  * atomics on shared locations use commutative operations and discard the
    old value; atomics that return a value use thread-private addresses;
  * shared memory is exchanged only between bar.sync barriers;
  * warp-level instructions (shfl.sync, vote.sync, match) only
    occur at nesting depth 0 with a full member mask and NT a multiple of 32
    (activemask is not generated: the ISA fixes no reconvergence point, so
    its value after a divergent branch is implementation-defined);
  * single-precision NaN results are compared as "some NaN" (the ISA says
    single-precision instructions return an unspecified NaN, 9.7.3).
Categories (--cats): int, float, cvt, mem, ctrl, warp (default: all).
"""
import argparse
import json
import math
import os
import random
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ALL_CATS = ("int", "float", "cvt", "mem", "ctrl", "warp")

# register classes: name prefix, count, PTX type, slot kind
RCLS = {
    "r": (8, ".b32", "b32"),
    "rd": (4, ".b64", "b64"),
    "f": (6, ".f32", "f32"),
    "fd": (3, ".f64", "f64"),
    "h": (2, ".b16", "b16"),
    "p": (3, ".pred", "pred"),
}
ORDER = ["r", "rd", "f", "fd", "h", "p"]
SLOTS = sum(RCLS[c][0] for c in ORDER)
COMMON_OPS = ["add.u32", "and.b32", "or.b32", "xor.b32", "min.u32", "max.s32"]
NCOM = len(COMMON_OPS)   # common shared words (8 bytes apart) at the start of sh
INW = 32          # input words (u32) per thread
SCRB = 64         # scratch bytes per thread


def regs(c):
    return ["%%%s%d" % (c, i) for i in range(RCLS[c][0])]


# ------------------------------------------------------------------ values
F32_SPECIAL = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000,
               0x7f800000, 0xff800000, 0x7fc00000, 0x7fffffff, 0xffc00000,
               0x00000001, 0x80000001, 0x007fffff, 0x00800000, 0x80800000,
               0x7f7fffff, 0xff7fffff, 0x3f800001, 0x3fffffff, 0x40000000,
               0x4b000000, 0x4b7fffff, 0x4f000000, 0xcf000000, 0x4effffff,
               0x5f000000, 0xdf000000, 0x3effffff, 0x3f7fffff, 0x33800000,
               0x477fe000, 0x47800000, 0x38800000, 0x33000000]
F64_SPECIAL = [0x0, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000,
               0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000,
               0x7ff4000000000001, 0xfff8000000000000, 0x0000000000000001,
               0x000fffffffffffff, 0x0010000000000000, 0x7fefffffffffffff,
               0x3ff0000000000001, 0x41e0000000000000, 0xc1e0000000000000,
               0x43e0000000000000, 0xc3e0000000000000, 0x41efffffffe00000,
               0x3fe0000000000000, 0x47efffffe0000000, 0x47efffffffffffff,
               0x36a0000000000000, 0x3810000000000000, 0x4330000000000000]
I32_SPECIAL = [0, 1, 2, 3, 0x7fffffff, 0x80000000, 0xffffffff, 0x80000001,
               0xfffffffe, 31, 32, 33, 63, 64, 0xff, 0xff00, 0xffff, 0x10000,
               0x12345678, 0x89abcdef, 7, 8, 16, 24]


def rnd_u32(rng):
    k = rng.random()
    if k < 0.35:
        return rng.getrandbits(32)
    if k < 0.55:
        return rng.randint(-20, 20) & 0xFFFFFFFF
    if k < 0.75:
        return rng.choice(I32_SPECIAL)
    if k < 0.85:
        return (1 << rng.randrange(32)) ^ (rng.choice([0, 0xFFFFFFFF]))
    if k < 0.93:
        return rnd_f32(rng)
    return rng.getrandbits(rng.randint(1, 32))


def rnd_f32(rng):
    k = rng.random()
    if k < 0.3:
        return rng.choice(F32_SPECIAL)
    if k < 0.6:
        v = rng.uniform(-100, 100) * (2.0 ** rng.randint(-10, 10))
        return struct.unpack("<I", struct.pack("<f", v))[0]
    if k < 0.75:
        v = float(rng.randint(-70000, 70000)) + rng.choice([0, 0.5, 0.25, 0.75, 0.4999, 0.5001])
        return struct.unpack("<I", struct.pack("<f", v))[0]
    if k < 0.85:
        e = rng.randint(0, 255)
        return (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)
    if k < 0.92:
        return (rng.getrandbits(1) << 31) | rng.getrandbits(rng.randint(1, 23))   # subnormal
    return rng.getrandbits(32)


def rnd_f64(rng):
    k = rng.random()
    if k < 0.3:
        return rng.choice(F64_SPECIAL)
    if k < 0.6:
        v = rng.uniform(-100, 100) * (2.0 ** rng.randint(-30, 30))
        return struct.unpack("<Q", struct.pack("<d", v))[0]
    if k < 0.75:
        v = float(rng.randint(-5000000000, 5000000000)) + rng.choice([0, 0.5, 0.25, 0.75])
        return struct.unpack("<Q", struct.pack("<d", v))[0]
    if k < 0.85:
        f = rnd_f32(rng)   # exactly representable single values (exercises cvt)
        return struct.unpack("<Q", struct.pack("<d", struct.unpack("<f", struct.pack("<I", f))[0]))[0]
    if k < 0.92:
        return (rng.getrandbits(1) << 63) | rng.getrandbits(rng.randint(1, 52))
    return rng.getrandbits(64)


def rnd_u64(rng):
    k = rng.random()
    if k < 0.4:
        return rng.getrandbits(64)
    if k < 0.6:
        return rng.randint(-20, 20) & ((1 << 64) - 1)
    if k < 0.8:
        return rnd_u32(rng) | (rng.choice([0, 0xFFFFFFFF, rng.getrandbits(32)]) << 32)
    return rng.choice([0, 1, (1 << 63), (1 << 64) - 1, (1 << 63) - 1, 63, 64, 1 << 32])


def rnd_u16(rng):
    k = rng.random()
    if k < 0.4:
        return rng.getrandbits(16)
    if k < 0.7:   # f16 specials
        return rng.choice([0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e00, 0x0001,
                           0x03ff, 0x0400, 0x7bff, 0x3555, 0x7fff, 0x4000])
    return rng.randint(-20, 20) & 0xFFFF


# ------------------------------------------------------------------ program
class Prog:
    def __init__(self, seed, cats, nt=None, ncta=None, nstmts=None, all_forms=False, raw=False):
        # all_forms=False leaves out forms GPGPU-Sim cannot execute at all
        # (setp with a boolean operator, 64-bit atom.and/or/xor/min/max):
        # those would only measure GPGPU-Sim's coverage.
        self.all_forms = all_forms
        # raw=False also steers clear of the known GPGPU-Sim deviations listed
        # in README.md (operands chosen inside GPGPU-Sim's correct domain, or
        # the form left out), so that new differences stand out.
        self.safe = not raw
        self.seed = seed
        self.rng = random.Random(seed)
        self.cats = set(cats)
        r = self.rng
        self.nt = nt or r.choice([32, 32, 64])
        self.ncta = ncta or r.choice([1, 1, 2])
        self.nstmts = nstmts or r.randint(12, 40)
        self.label = 0
        self.body = [self.stmt(depth=0) for _ in range(self.nstmts)]

    # ---- helpers
    def pick(self, c):
        return self.rng.choice(regs(c))

    def lab(self):
        self.label += 1
        return "$L%d" % self.label

    def guard(self):
        r = self.rng
        if "ctrl" in self.cats and r.random() < 0.15:
            return "@%s%s " % (r.choice(["", "!"]), self.pick("p"))
        return ""

    def imm32(self):
        return str(self.rng.choice([0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 255, -1, -2, -128,
                                    0x7fffffff, 0x80000000 - (1 << 32), 1000, 65535]))

    def src32(self):
        return self.imm32() if self.rng.random() < 0.15 else self.pick("r")

    def stmt(self, depth):
        r = self.rng
        kinds = []
        for c in self.cats:
            kinds += {"int": [("int", 6)], "float": [("float", 5)], "cvt": [("cvt", 3)],
                      "mem": [("mem", 2)], "ctrl": [("ctrl", 2 if depth < 2 else 0)],
                      "warp": [("warp", 2 if depth == 0 else 0)]}[c]
        if depth == 0 and "mem" in self.cats:
            kinds.append(("shared", 1))
        tot = sum(w for _, w in kinds)
        x = r.uniform(0, tot)
        for k, w in kinds:
            if x < w:
                break
            x -= w
        if k == "ctrl":
            return self.ctrl(depth)
        if k == "warp":
            return ("warp", self.warp())
        if k == "shared":
            return ("bar", self.shared_exchange())
        for _ in range(50):
            s = getattr(self, "op_" + k)()
            if not EXCLUDE or not re.search(EXCLUDE, str(s)):
                break
        return s

    # ---- control flow
    def ctrl(self, depth):
        r = self.rng
        p = self.pick("p")
        n = r.randint(1, 4)
        k = r.random()
        if k < 0.4:
            return ("if", r.choice(["", "!"]) + p, [self.stmt(depth + 1) for _ in range(n)])
        if k < 0.75:
            return ("ifelse", r.choice(["", "!"]) + p,
                    [self.stmt(depth + 1) for _ in range(n)],
                    [self.stmt(depth + 1) for _ in range(r.randint(1, 3))])
        # bounded, possibly divergent loop on a private counter
        cnt = r.choice(["const", "tid"])
        return ("loop", depth, cnt, r.randint(1, 3), [self.stmt(depth + 1) for _ in range(n)])

    # ---- integer
    def op_int(self):
        r = self.rng
        g = self.guard()
        a, b, c = self.pick("r"), self.src32(), self.pick("r")
        d = self.pick("r")
        A, B, C, D = self.pick("rd"), self.pick("rd"), self.pick("rd"), self.pick("rd")
        t32 = r.choice([".u32", ".s32"])
        t64 = r.choice([".u64", ".s64"])
        k = r.randrange(36)
        if k >= 32:
            return self.op_int_wide()
        if k == 0:
            return g + "%s%s %s, %s, %s;" % (r.choice(["add", "sub"]), r.choice([".u32", ".s32"] + ([] if self.safe else [".sat.s32"])), d, a, b)
        if k == 1:
            return g + "%s%s %s, %s, %s;" % (r.choice(["add", "sub"]), t64, D, A, B)
        if k == 2:
            return g + "mul.%s%s %s, %s, %s;" % (r.choice(["lo", "hi"]), t32, d, a, b)
        if k == 3:
            return g + "mul.wide%s %s, %s, %s;" % (t32, D, a, self.pick("r"))
        if k == 4:
            return g + "mad.%s%s %s, %s, %s, %s;" % (r.choice(["lo", "hi"]), t32, d, a, b, c)
        if k == 5:
            return g + "mad.wide%s %s, %s, %s, %s;" % (t32, D, a, self.pick("r"), C)
        if k == 6:
            return g + "mul.%s%s %s, %s, %s;" % (r.choice(["lo", "hi"]), t64, D, A, B)
        if k == 7:
            return g + "mad.lo%s %s, %s, %s, %s;" % (t64, D, A, B, C)
        if k == 8:
            return g + "sad%s %s, %s, %s, %s;" % (t32, d, a, b, c)
        if k == 9:
            return g + "mul24.%s%s %s, %s, %s;" % (r.choice(["lo", "hi"]), t32, d, a, b)
        if k == 10:
            return g + "mad24.%s%s %s, %s, %s, %s;" % (r.choice(["lo", "hi"]), t32, d, a, b, c)
        if k == 11:   # div / rem with a specified result
            op = r.choice(["div", "rem"])
            if t32 == ".u32":
                return [g + "or.b32 %s, %s, 1;" % (c, c), g + "%s.u32 %s, %s, %s;" % (op, d, a, c)]
            pre = [g + "and.b32 %s, %s, 2147483647;" % (c, c), g + "or.b32 %s, %s, 1;" % (c, c)]
            if op == "rem":
                pre.append(g + "and.b32 %s, %s, 2147483647;" % (a, a))
            return pre + [g + "%s.s32 %s, %s, %s;" % (op, d, a, c)]
        if k == 12:
            return g + "%s%s %s, %s, %s;" % (r.choice(["min", "max"]), t32, d, a, b)
        if k == 13:
            return g + "%s%s %s, %s, %s;" % (r.choice(["min", "max"]), t64, D, A, B)
        if k == 14:
            return g + "%s.s32 %s, %s;" % (r.choice(["abs", "neg"]), d, a)
        if k == 15:
            return g + "%s.b32 %s, %s, %s;" % (r.choice(["and", "or", "xor"]), d, a, b)
        if k == 16:
            return g + "%s.b64 %s, %s, %s;" % (r.choice(["and", "or", "xor"]), D, A, B)
        if k == 17:
            return g + "%s.b32 %s, %s;" % (r.choice(["not", "cnot"]), d, a)
        if k == 18:
            op = r.choice(["shl.b32", "shr.u32", "shr.s32", "shr.b32"])
            return g + "%s %s, %s, %s;" % (op, d, a, self.src32())
        if k == 19:
            op = r.choice(["shl.b64", "shr.u64", "shr.s64"])
            return g + "%s %s, %s, %s;" % (op, D, A, str(r.randrange(64)) if self.safe else self.src32())
        if k == 20:
            if self.safe:
                w = r.choice([32, 64])
                pos = r.randrange(w)
                ln = r.randint(1, w - 1)
                if w == 32:
                    return g + "bfe%s %s, %s, %d, %d;" % (t32, d, a, pos, ln)
                return g + "bfe%s %s, %s, %d, %d;" % (t64, D, A, pos, ln)
            if r.random() < 0.5:
                return g + "bfe%s %s, %s, %s, %s;" % (t32, d, a, self.src32(), self.src32())
            return g + "bfe%s %s, %s, %s, %s;" % (t64, D, A, self.src32(), self.src32())
        if k == 21:
            if self.safe or r.random() < 0.5:
                return g + "bfi.b32 %s, %s, %s, %s, %s;" % (d, a, c, self.src32(), self.src32())
            return g + "bfi.b64 %s, %s, %s, %s, %s;" % (D, A, B, self.src32(), self.src32())
        if k == 22:
            mode = r.choice(["", "", ".f4e", ".b4e", ".rc8", ".ecl", ".ecr", ".rc16"])
            return g + "prmt.b32%s %s, %s, %s, %s;" % (mode, d, a, c, self.src32())
        if k == 23:
            if r.random() < 0.5:
                return g + "%s.b32 %s, %s;" % (r.choice(["brev", "popc", "clz"]), d, a)
            op = r.choice(["popc", "clz"] + ([] if self.safe else ["brev"]))
            return g + ("brev.b64 %s, %s;" % (D, A) if op == "brev" else "%s.b64 %s, %s;" % (op, d, A))
        if k == 24:
            sh = r.choice(["", ".shiftamt"])
            if r.random() < 0.5:
                return g + "bfind%s%s %s, %s;" % (sh, t32, d, a)
            return g + "bfind%s%s %s, %s;" % (sh, t64, d, A)
        if k == 25:
            return g + self.setp_int()
        if k == 26:
            p = self.pick("p")
            if r.random() < 0.5:
                return g + "selp.b32 %s, %s, %s, %s;" % (d, a, b, p)
            return g + "selp.b64 %s, %s, %s, %s;" % (D, A, B, p)
        if k == 27:
            cmp = r.choice(["eq", "ne", "lt", "le", "gt", "ge"] if t32 == ".s32" else
                           ["eq", "ne", "lo", "ls", "hi", "hs"])
            return g + "set.%s.u32%s %s, %s, %s;" % (cmp, t32, d, a, b)
        if k == 28:
            return g + "slct.b32.s32 %s, %s, %s, %s;" % (d, a, c, self.pick("r"))
        if k == 29:
            return g + "shf.%s.%s.b32 %s, %s, %s, %s;" % (r.choice(["l", "r"]), r.choice(["wrap", "clamp"]),
                                                       d, a, c, str(r.randint(1, 31)) if self.safe else self.src32())
        if k == 30:
            p = self.pick("p")
            return g + "%s.pred %s, %s, %s;" % (r.choice(["and", "or", "xor"]), p, self.pick("p"), self.pick("p"))
        return g + "mov.b32 %s, %s;" % (d, self.src32())

    def op_int_wide(self):
        """16-bit integer arithmetic on the .b16 registers, and 64-bit forms
        not covered by op_int (mul.hi/mad.hi/div/rem/popc/clz/bfind on .u64/.s64)."""
        r = self.rng
        g = self.guard()
        h, h1, h2 = self.pick("h"), self.pick("h"), self.pick("h")
        t16 = r.choice([".u16", ".s16"])
        k = r.randrange(12)
        if k == 0:
            return g + "%s%s %s, %s, %s;" % (r.choice(["add", "sub", "min", "max"]), t16, h, h1, h2)
        if k == 1:
            return g + "mul.%s%s %s, %s, %s;" % (r.choice(["lo", "hi"]), t16, h, h1, h2)
        if k == 2:
            return g + "mul.wide%s %s, %s, %s;" % (t16, self.pick("r"), h1, h2)
        if k == 3:
            return g + "mad.%s%s %s, %s, %s, %s;" % (r.choice(["lo", "hi"]), t16, h, h1, h2, self.pick("h"))
        if k == 4:
            return g + "%s.b16 %s, %s, %s;" % (r.choice(["and", "or", "xor"]), h, h1, h2)
        if k == 5:
            op = r.choice(["shl.b16", "shr.u16", "shr.s16"])
            return g + "%s %s, %s, %s;" % (op, h, h1, str(r.randrange(20)) if self.safe else self.src32())
        if k == 6:
            cmp = r.choice(["eq", "ne", "lt", "le", "gt", "ge"] if t16 == ".s16" else ["eq", "ne", "lo", "ls", "hi", "hs"])
            return g + "setp.%s%s %s, %s, %s;" % (cmp, t16, self.pick("p"), h1, h2)
        if k == 7:
            op = r.choice(["div", "rem"] if self.all_forms else ["div"])   # divisor forced odd positive; no rem.16 in GPGPU-Sim
            pre = [g + "or.b16 %s, %s, 1;" % (h2, h2)]
            if t16 == ".s16":
                pre = [g + "and.b16 %s, %s, 32767;" % (h2, h2), g + "or.b16 %s, %s, 1;" % (h2, h2)]
                if op == "rem":
                    pre.append(g + "and.b16 %s, %s, 32767;" % (h1, h1))
            return pre + [g + "%s%s %s, %s, %s;" % (op, t16, h, h1, h2)]
        A, B, C, D = self.pick("rd"), self.pick("rd"), self.pick("rd"), self.pick("rd")
        t64 = r.choice([".u64", ".s64"])
        if k == 8:
            return g + "mad.hi%s %s, %s, %s, %s;" % (t64, D, A, B, C)
        if k == 9:
            op = r.choice(["div", "rem"])
            if t64 == ".u64":
                return [g + "or.b64 %s, %s, 1;" % (B, B), g + "%s.u64 %s, %s, %s;" % (op, D, A, B)]
            pre = [g + "and.b64 %s, %s, 9223372036854775807;" % (B, B), g + "or.b64 %s, %s, 1;" % (B, B)]
            if op == "rem":
                pre.append(g + "and.b64 %s, %s, 9223372036854775807;" % (A, A))
            return pre + [g + "%s.s64 %s, %s, %s;" % (op, D, A, B)]
        if k == 10:
            return g + "%s%s %s, %s;" % (r.choice(["abs", "neg"]), ".s64", D, A)
        return g + "sad%s %s, %s, %s, %s;" % (t64, D, A, B, C)

    def setp_int(self):
        r = self.rng
        t = r.choice([".u32", ".s32", ".b32", ".u64", ".s64"])
        cmps = ["eq", "ne"] + (["lt", "le", "gt", "ge"] if t[1] == "s" else
                               ["lo", "ls", "hi", "hs"] if t[1] == "u" else [])
        cmp = r.choice(cmps)
        if t.endswith("32"):
            a, b = self.pick("r"), self.src32()
        else:
            a, b = self.pick("rd"), self.pick("rd")
        p = self.pick("p")
        if self.all_forms and r.random() < 0.3:
            bop = r.choice(["and", "or", "xor"])
            return "setp.%s.%s%s %s, %s, %s, %s%s;" % (cmp, bop, t, p, a, b, r.choice(["", "!"]), self.pick("p"))
        return "setp.%s%s %s, %s, %s;" % (cmp, t, p, a, b)

    # ---- floating point
    def op_float(self):
        r = self.rng
        g = self.guard()
        rnd = r.choice([".rn", ".rz", ".rm", ".rp"])
        if r.random() < 0.6:   # single
            d, a, b, c = self.pick("f"), self.pick("f"), self.pick("f"), self.pick("f")
            ftz = "" if self.safe else r.choice(["", "", ".ftz"])
            sat = "" if self.safe else r.choice(["", "", "", ".sat"])
            k = r.randrange(12)
            if self.safe and k == 6:
                k = 5
            if k == 0:
                return g + "%s%s%s%s.f32 %s, %s, %s;" % (r.choice(["add", "sub", "mul"]), rnd, ftz, sat, d, a, b)
            if k == 1:
                return g + "%s%s%s%s.f32 %s, %s, %s, %s;" % (r.choice(["fma", "mad"]), rnd, ftz, sat, d, a, b, c)
            if k == 2:
                return g + "div%s%s.f32 %s, %s, %s;" % (rnd, ftz, d, a, b)
            if k == 3:
                return g + "sqrt%s%s.f32 %s, %s;" % (rnd, ftz, d, a)
            if k == 4:
                return g + "rcp%s%s.f32 %s, %s;" % (".rn" if self.safe else rnd, ftz, d, a)
            if k == 5:
                return g + "%s%s.f32 %s, %s, %s;" % (r.choice(["min", "max"]), ftz, d, a, b)
            if k == 6:
                return g + "%s%s.f32 %s, %s;" % (r.choice(["abs", "neg"]), ftz, d, a)
            if k == 7:
                return g + self.setp_float(".f32", a, b, ftz)
            if k == 8:
                return g + "selp.f32 %s, %s, %s, %s;" % (d, a, b, self.pick("p"))
            if k == 9:
                return g + "mov.f32 %s, %s;" % (d, r.choice(["0f3F800000", "0fBF000000", "0f7F800000",
                                                             "0f00000001", "0f4B000001", a]))
            if k == 11:
                cmp = r.choice(["eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan"])
                if r.random() < 0.5:
                    return g + "set.%s%s.f32.f32 %s, %s, %s;" % (cmp, ftz, d, a, b)
                return g + "set.%s%s.u32.f32 %s, %s, %s;" % (cmp, ftz, self.pick("r"), a, b)
            return g + "%s%s.f32 %s, %s, %s;" % (r.choice(["add", "mul"]), ftz, d, a, b)   # default rounding
        d, a, b, c = self.pick("fd"), self.pick("fd"), self.pick("fd"), self.pick("fd")
        k = r.randrange(9)
        if self.safe and k == 6:
            k = 5
        if k == 0:
            return g + "%s%s.f64 %s, %s, %s;" % (r.choice(["add", "sub", "mul"]), rnd, d, a, b)
        if k == 1:
            return g + "%s%s.f64 %s, %s, %s, %s;" % (r.choice(["fma", "mad"]), rnd, d, a, b, c)
        if k == 2:
            return g + "div%s.f64 %s, %s, %s;" % (rnd, d, a, b)
        if k == 3:
            return g + "sqrt%s.f64 %s, %s;" % (rnd, d, a)
        if k == 4:
            return g + "rcp%s.f64 %s, %s;" % (".rn" if self.safe else rnd, d, a)
        if k == 5:
            return g + "%s.f64 %s, %s, %s;" % (r.choice(["min", "max"]), d, a, b)
        if k == 6:
            return g + "%s.f64 %s, %s;" % (r.choice(["abs", "neg"]), d, a)
        if k == 7:
            return g + self.setp_float(".f64", a, b, "")
        return g + "selp.f64 %s, %s, %s, %s;" % (d, a, b, self.pick("p"))

    def setp_float(self, t, a, b, ftz):
        r = self.rng
        cmp = r.choice(["eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu",
                        "num", "nan"])
        p = self.pick("p")
        if self.all_forms and r.random() < 0.3:
            bop = r.choice(["and", "or", "xor"])
            return "setp.%s.%s%s%s %s, %s, %s, %s%s;" % (cmp, bop, ftz, t, p, a, b, r.choice(["", "!"]), self.pick("p"))
        return "setp.%s%s%s %s, %s, %s;" % (cmp, ftz, t, p, a, b)

    # ---- conversions
    def op_cvt(self):
        r = self.rng
        g = self.guard()
        k = r.randrange(9)
        ints32 = [".u8", ".s8", ".u16", ".s16", ".u32", ".s32"]
        sat = "" if self.safe else r.choice(["", "", ".sat"])   # GPGPU-Sim ignores .sat on int->int
        if k == 0:   # int -> int (32-bit registers)
            # narrow destination types in a 32-bit register: sign/zero
            # extension of the result per Table 28 (9.4.1)
            dt, st = r.choice(ints32[4:] if self.safe else ints32), r.choice(ints32)   # GPGPU-Sim does not extend narrow results
            return g + "cvt%s%s%s %s, %s;" % (sat, dt, st, self.pick("r"), self.pick("r"))
        if k == 1:   # 64 <-> 32
            if r.random() < 0.5:
                return g + "cvt%s%s%s %s, %s;" % (sat, r.choice([".u64", ".s64"]), r.choice(ints32), self.pick("rd"), self.pick("r"))
            return g + "cvt%s%s%s %s, %s;" % (sat, r.choice([".u32", ".s32"]), r.choice([".u64", ".s64"]), self.pick("r"), self.pick("rd"))
        irnd = r.choice([".rni", ".rzi", ".rmi", ".rpi"])
        frnd = r.choice([".rn", ".rz", ".rm", ".rp"])
        ftz = r.choice(["", ".ftz"])
        if self.safe:
            ftz = ""
            sat = ""
            if k == 7:   # GPGPU-Sim ignores the rounding mode of cvt.f16.f32
                frnd = ".rn"
        if k in (2, 3):   # float -> int
            dt = r.choice(["u32", "s32"] if k == 2 else ["u64", "s64"])
            single = r.random() < 0.55
            src = self.pick("f") if single else self.pick("fd")
            dst = self.pick("r") if k == 2 else self.pick("rd")
            pre = []
            if self.safe:   # keep the source inside the destination range (GPGPU-Sim does not saturate)
                # GPGPU-Sim converts through a 32-bit signed int: stay below 2^31
                lo, hi = {"u32": ("0f00000000", "0f4EFFFFFF"), "s32": ("0fCEFFFFFF", "0f4EFFFFFF"),
                          "u64": ("0f00000000", "0f4EFFFFFF"), "s64": ("0fCEFFFFFF", "0f4EFFFFFF")}[dt]
                if not single:
                    lo, hi = ("0d%016x" % struct.unpack("<Q", struct.pack("<d", struct.unpack("<f", struct.pack("<I", int(x[2:], 16)))[0]))[0]
                              for x in (lo, hi))
                t = ".f32" if single else ".f64"
                tmp = "%tf" if single else "%tfd"
                pre = [g + "max%s %s, %s, %s;" % (t, tmp, src, lo), g + "min%s %s, %s, %s;" % (t, tmp, tmp, hi)]
                src = tmp
            return pre + [g + "cvt%s%s%s.%s.%s %s, %s;" % (irnd, ftz if single else "", sat, dt,
                                                             "f32" if single else "f64", dst, src)]
        if k == 4:   # int -> float
            if r.random() < 0.5:
                return g + "cvt%s.f32.%s %s, %s;" % (frnd, r.choice(["u32", "s32"]), self.pick("f"), self.pick("r"))
            if r.random() < 0.5:
                return g + "cvt%s.f32.%s %s, %s;" % (frnd, r.choice(["u64", "s64"]), self.pick("f"), self.pick("rd"))
            st = r.choice(["u32", "s32", "u64", "s64"])
            return g + "cvt%s.f64.%s %s, %s;" % (frnd, st, self.pick("fd"),
                                               self.pick("rd") if st.endswith("64") else self.pick("r"))
        if k == 5:   # f32 <-> f64
            if r.random() < 0.5:
                return g + "cvt%s%s%s.f32.f64 %s, %s;" % (".rn" if self.safe else frnd, ftz, sat, self.pick("f"), self.pick("fd"))
            return g + "cvt%s%s.f64.f32 %s, %s;" % (ftz, sat, self.pick("fd"), self.pick("f"))
        if k == 6:   # float -> float, integral rounding
            if r.random() < 0.5:
                return g + "cvt%s%s%s.f32.f32 %s, %s;" % (irnd, ftz, sat, self.pick("f"), self.pick("f"))
            return g + "cvt%s%s.f64.f64 %s, %s;" % (irnd, sat, self.pick("fd"), self.pick("fd"))
        if k == 7:   # half
            if r.random() < 0.5 and not self.safe:
                return g + "cvt%s%s.f16.f32 %s, %s;" % (frnd, sat, self.pick("h"), self.pick("f"))
            return g + "cvt%s%s.f32.f16 %s, %s;" % (ftz, sat, self.pick("f"), self.pick("h"))
        # 16-bit integer registers
        return g + "cvt%s%s%s %s, %s;" % (sat, r.choice([".u16", ".s16"]), r.choice([".u32", ".s32", ".u8", ".s8"]),
                                          self.pick("h"), self.pick("r"))

    # ---- memory (thread-private scratch, private and shared atomics)
    def op_mem(self):
        r = self.rng
        g = self.guard()
        k = r.randrange(6)
        if k == 0:
            w = r.choice([1, 2, 4, 8])
            off = r.randrange(0, SCRB // w) * w
            if w == 8:
                return g + "st.global.b64 [%%scr+%d], %s;" % (off, self.pick("rd"))
            t = {1: ".b8", 2: ".b16", 4: ".b32"}[w]
            src = self.pick("h") if w == 2 and r.random() < 0.5 else self.pick("r")
            if w == 1 and src.startswith("%h"):
                src = self.pick("r")
            return g + "st.global%s [%%scr+%d], %s;" % (t, off, src)
        if k == 1:
            w = r.choice([1, 2, 4, 8])
            off = r.randrange(0, SCRB // w) * w
            if w == 8:
                return g + "ld.global.b64 %s, [%%scr+%d];" % (self.pick("rd"), off)
            t = r.choice({1: [".u8"] + ([] if self.safe else [".s8"]), 2: [".u16"] + ([] if self.safe else [".s16"]),
                          4: [".b32"]}[w])
            return g + "ld.global%s %s, [%%scr+%d];" % (t, self.pick("r"), off)
        if k == 2:   # vector ld/st
            n = r.choice([2, 4])
            off = r.randrange(0, SCRB // (4 * n)) * 4 * n
            rs = ", ".join(self.pick("r") for _ in range(n))
            if r.random() < 0.5:
                return g + "st.global.v%d.b32 [%%scr+%d], {%s};" % (n, off, rs)
            ds = ", ".join(r.sample(regs("r"), n))
            return g + "ld.global.v%d.b32 {%s}, [%%scr+%d];" % (n, ds, off)
        if k == 3:   # private atomic with returned value
            sp = r.choice(["global", "shared"])
            base = "%scr" if sp == "global" else "%shp"
            off = r.randrange(0, 8) * 4 if sp == "global" else r.randrange(0, 2) * 4
            op = r.choice(["add.u32", "add.s32", "min.u32", "min.s32", "max.u32", "max.s32", "inc.u32",
                           "dec.u32", "and.b32", "or.b32", "xor.b32", "exch.b32", "add.f32"] +
                          ([] if self.safe else ["cas.b32"]))
            if op == "cas.b32":
                return g + "atom.%s.cas.b32 %s, [%s+%d], %s, %s;" % (sp, self.pick("r"), base, off, self.pick("r"), self.pick("r"))
            if op == "add.f32":
                return g + "atom.%s.add.f32 %s, [%s+%d], %s;" % (sp, self.pick("f"), base, off, self.pick("f"))
            return g + "atom.%s.%s %s, [%s+%d], %s;" % (sp, op, self.pick("r"), base, off, self.src32())
        if k == 4:   # 64-bit private atomic
            off = r.randrange(0, SCRB // 8) * 8
            op = r.choice(["add.u64", "exch.b64"] + ([] if self.safe else ["cas.b64"]) +
                          (["min.u64", "max.s64", "and.b64", "or.b64", "xor.b64"] if self.all_forms else []))
            if op == "cas.b64":
                return g + "atom.global.cas.b64 %s, [%%scr+%d], %s, %s;" % (self.pick("rd"), off, self.pick("rd"), self.pick("rd"))
            return g + "atom.global.%s %s, [%%scr+%d], %s;" % (op, self.pick("rd"), off, self.pick("rd"))
        # common shared word, commutative op, value discarded
        # one fixed operation per common word, so that the final value does
        # not depend on the order in which threads perform their atomics
        i = r.randrange(len(COMMON_OPS))
        return g + "atom.shared.%s %%junk, [sh+%d], %s;" % (COMMON_OPS[i], 8 * i, self.src32())

    def shared_exchange(self):
        r = self.rng
        k = r.randint(1, self.nt - 1)
        src = self.pick("rd")
        dst = self.pick("rd")
        return ["bar.sync 0;",
                "st.shared.b64 [%%shq], %s;" % src,
                "bar.sync 0;",
                "add.u32 %%t32, %%xtid, %d;" % k,
                "rem.u32 %%t32, %%t32, %d;" % self.nt,
                "mul.wide.u32 %t64, %t32, 8;",
                "mov.u64 %t64b, sh;",
                "add.u64 %t64, %t64, %t64b;",
                "ld.shared.b64 %s, [%%t64+%d];" % (dst, 8 * NCOM),
                "bar.sync 0;"]

    # ---- warp-level (converged code only)
    def warp(self):
        r = self.rng
        k = r.randrange(7)
        d, a = self.pick("r"), self.pick("r")
        if k <= 2:
            mode = r.choice(["up", "down", "bfly", "idx"])
            b = r.choice([str(r.randrange(0, 32)), self.pick("r")])
            c = r.choice(["31", "0", "0x1f", "0x1c1f", "0x181f", "0x101f", "7", "0x1e00", str(r.randrange(0, 1 << 13)), self.pick("r")])
            if r.random() < 0.4:
                return ["shfl.sync.%s.b32 %s|%s, %s, %s, %s, 0xffffffff;" % (mode, d, self.pick("p"), a, b, c)]
            return ["shfl.sync.%s.b32 %s, %s, %s, %s, 0xffffffff;" % (mode, d, a, b, c)]
        if k == 3:
            return ["vote.sync.ballot.b32 %s, %s%s, 0xffffffff;" % (d, r.choice(["", "!"]), self.pick("p"))]
        if k == 4:
            return ["vote.sync.%s.pred %s, %s%s, 0xffffffff;" % (r.choice(["all", "any"] + ([] if self.safe else ["uni"])), self.pick("p"),
                                                                r.choice(["", "!"]), self.pick("p"))]
        return ["match.any.sync.b32 %s, %s, 0xffffffff;" % (d, a)]

    # ---- rendering
    def render_stmts(self, ss, out, ind):
        for s in ss:
            if isinstance(s, str):
                out.append(ind + s)
            elif isinstance(s, list):
                out += [ind + x for x in s]
            elif s[0] in ("warp", "bar"):
                out += [ind + x for x in s[1]]
            elif s[0] == "if":
                L = self.lab()
                neg = s[1].startswith("!")
                p = s[1].lstrip("!")
                out.append(ind + "@%s%s bra %s;" % ("" if neg else "!", p, L))
                self.render_stmts(s[2], out, ind + "  ")
                out.append(L + ":")
            elif s[0] == "ifelse":
                Le, Lj = self.lab(), self.lab()
                neg = s[1].startswith("!")
                p = s[1].lstrip("!")
                out.append(ind + "@%s%s bra %s;" % ("" if neg else "!", p, Le))
                self.render_stmts(s[2], out, ind + "  ")
                out.append(ind + "bra %s;" % Lj)
                out.append(Le + ":")
                self.render_stmts(s[3], out, ind + "  ")
                out.append(Lj + ":")
            elif s[0] == "loop":
                _, depth, cnt, n, body = s
                L = self.lab()
                lc = "%%lc%d" % depth
                if cnt == "const":
                    out.append(ind + "mov.u32 %s, %d;" % (lc, n))
                else:
                    out.append(ind + "and.b32 %s, %%xtid, %d;" % (lc, n))
                    out.append(ind + "add.u32 %s, %s, 1;" % (lc, lc))
                out.append(L + ":")
                self.render_stmts(body, out, ind + "  ")
                out.append(ind + "sub.u32 %s, %s, 1;" % (lc, lc))
                out.append(ind + "setp.ne.u32 %%lp%d, %s, 0;" % (depth, lc))
                out.append(ind + "@%%lp%d bra %s;" % (depth, L))
            else:
                raise ValueError(s)

    def kernel(self, body=None):
        body = self.body if body is None else body
        self.label = 0
        L = []
        L.append(".version 7.0")
        L.append(".target sm_80")
        L.append(".address_size 64")
        L.append("")
        L.append(".visible .entry k(.param .u64 pin, .param .u64 pout, .param .u64 pscr, .param .u64 psho)")
        L.append("{")
        decl = []
        for c in ORDER:
            n, t, _ = RCLS[c]
            decl.append(".reg %s %%%s<%d>;" % (t, c, n))
        L.append("  " + " ".join(decl))
        L.append("  .reg .b64 %in, %out, %scr, %sho, %t64, %t64b, %shq, %shp, %shc; .reg .b32 %xtid, %gid, %t32, %junk, %lc<3>;")
        L.append("  .reg .pred %lp<3>; .reg .f32 %tf; .reg .f64 %tfd;")
        L.append("  .shared .align 8 .b8 sh[%d];" % (8 * (self.nt + NCOM) + 16))
        L.append("  ld.param.u64 %in, [pin]; ld.param.u64 %out, [pout]; ld.param.u64 %scr, [pscr]; ld.param.u64 %sho, [psho];")
        L.append("  mov.u32 %xtid, %tid.x; mov.u32 %t32, %ctaid.x;")
        L.append("  mad.lo.u32 %%gid, %%t32, %d, %%xtid;" % self.nt)
        L.append("  mul.wide.u32 %%t64, %%gid, %d; add.s64 %%in, %%in, %%t64;" % (INW * 4))
        L.append("  mul.wide.u32 %%t64, %%gid, %d; add.s64 %%out, %%out, %%t64;" % (SLOTS * 8))
        L.append("  mul.wide.u32 %%t64, %%gid, %d; add.s64 %%scr, %%scr, %%t64;" % SCRB)
        # SHO: one word per thread, then 4 common words per CTA
        L.append("  mul.wide.u32 %%t64, %%t32, %d; add.s64 %%shc, %%sho, %%t64; add.s64 %%shc, %%shc, %d;"
                 % (8 * NCOM, 8 * self.nt * self.ncta))
        L.append("  mul.wide.u32 %t64, %gid, 8; add.s64 %sho, %sho, %t64;")
        # %shq: this thread's exchange word (after 4 words of common atomics),
        # %shp: this thread's private 8 bytes for shared atomics.
        L.append("  mov.u64 %t64b, sh;")
        L.append("  mul.wide.u32 %%t64, %%xtid, 8; add.u64 %%shq, %%t64b, %%t64; add.u64 %%shq, %%shq, %d;" % (8 * NCOM))
        L.append("  mov.u64 %shp, %shq;")
        L.append("  mov.b64 %t64, 0; st.shared.b64 [%shq], %t64;")
        L.append("  setp.lt.u32 %%lp0, %%xtid, %d;" % NCOM)
        L.append("  mul.wide.u32 %t64, %xtid, 8; add.u64 %t64, %t64b, %t64; mov.b64 %t64b, 0;")
        for i, op in enumerate(COMMON_OPS):   # identity element of each common word's operation
            init = {"and.b32": 0xFFFFFFFF, "min.u32": 0xFFFFFFFF, "max.s32": 0x80000000}.get(op)
            if init is not None:
                L.append("  setp.eq.u32 %%lp1, %%xtid, %d; @%%lp1 mov.b64 %%t64b, %d;" % (i, init))
        L.append("  @%lp0 st.shared.b64 [%t64], %t64b;")
        L.append("  bar.sync 0;")
        w = 0
        for i, rr in enumerate(regs("r")):
            L.append("  ld.global.b32 %s, [%%in+%d];" % (rr, 4 * w)); w += 1
        for rr in regs("rd"):
            L.append("  ld.global.b64 %s, [%%in+%d];" % (rr, 4 * w)); w += 2
        for rr in regs("f"):
            L.append("  ld.global.f32 %s, [%%in+%d];" % (rr, 4 * w)); w += 1
        for rr in regs("fd"):
            L.append("  ld.global.f64 %s, [%%in+%d];" % (rr, 4 * w)); w += 2
        for rr in regs("h"):
            L.append("  ld.global.b16 %s, [%%in+%d];" % (rr, 4 * w)); w += 1
        L.append("  ld.global.b32 %%t32, [%%in+%d];" % (4 * w))
        for i, rr in enumerate(regs("p")):
            L.append("  and.b32 %%junk, %%t32, %d; setp.ne.b32 %s, %%junk, 0;" % (1 << i, rr))
        assert w + 1 <= INW
        out = []
        self.render_stmts(body, out, "  ")
        L += out
        L.append("  bar.sync 0;")
        slot = 0
        for c in ORDER:
            for rr in regs(c):
                if c == "p":
                    L.append("  selp.u32 %%t32, 1, 0, %s; st.global.b32 [%%out+%d], %%t32;" % (rr, 8 * slot))
                else:
                    t = {"r": ".b32", "rd": ".b64", "f": ".f32", "fd": ".f64", "h": ".b16"}[c]
                    L.append("  st.global%s [%%out+%d], %s;" % (t, 8 * slot, rr))
                slot += 1
        L.append("  ld.shared.b64 %t64, [%shq]; st.global.b64 [%sho], %t64;")
        L.append("  setp.lt.u32 %%lp0, %%xtid, %d;" % NCOM)
        L.append("  mov.u64 %t64b, sh; mul.wide.u32 %t64, %xtid, 8; add.u64 %t64b, %t64b, %t64; add.u64 %shc, %shc, %t64;")
        L.append("  @%lp0 ld.shared.b64 %t64, [%t64b]; @%lp0 st.global.b64 [%shc], %t64;")
        L.append("  ret;")
        L.append("}")
        return "\n".join(L) + "\n"

    def inputs(self):
        r = random.Random(self.seed * 7919 + 1)
        f32, f64 = rnd_f32, rnd_f64
        if self.safe:   # GPGPU-Sim flushes subnormal inputs in several places
            def f32(rr):
                while True:
                    v = rnd_f32(rr)
                    if v & 0x7F800000 or not v & 0x7FFFFF:
                        return v
            def f64(rr):
                while True:
                    v = rnd_f64(rr)
                    if v >> 52 & 0x7FF or not v & ((1 << 52) - 1):
                        return v
        nthr = self.nt * self.ncta
        words = []
        for _ in range(nthr):
            w = []
            for _ in regs("r"):
                w.append(rnd_u32(r))
            for _ in regs("rd"):
                v = rnd_u64(r); w += [v & 0xFFFFFFFF, v >> 32]
            for _ in regs("f"):
                w.append(f32(r))
            for _ in regs("fd"):
                v = f64(r); w += [v & 0xFFFFFFFF, v >> 32]
            for _ in regs("h"):
                w.append(rnd_u16(r) | (r.getrandbits(16) << 16))
            w.append(r.getrandbits(32))
            w += [0] * (INW - len(w))
            words += w
        scr = [r.getrandbits(32) for _ in range(nthr * SCRB // 4)]
        return words, scr

    def program(self, body=None):
        words, scr = self.inputs()
        nthr = self.nt * self.ncta
        h = [self.kernel(body)]
        h.append("// seed %d, cats %s, %d CTA(s) x %d threads" % (self.seed, ",".join(sorted(self.cats)), self.ncta, self.nt))
        h.append(".buffer IN .u32 = { %s };" % ", ".join(str(x) for x in words))
        h.append(".buffer OUT .u8 [%d];" % (nthr * SLOTS * 8))
        h.append(".buffer SCR .u32 = { %s };" % ", ".join(str(x) for x in scr))
        h.append(".buffer SHO .u8 [%d];" % (nthr * 8 + 8 * NCOM * self.ncta))
        h.append(".launch k <<< %d, %d >>> (IN, OUT, SCR, SHO);" % (self.ncta, self.nt))
        h.append(".isacheck;   // the Sail front end checks operand types (Tables 26-28)")
        return "\n".join(h) + "\n"


def slot_kinds():
    ks = []
    for c in ORDER:
        ks += [RCLS[c][2]] * RCLS[c][0]
    return ks


# ------------------------------------------------------------------ comparison
def is_nan(kind, b):
    if kind == "f32":
        v = int.from_bytes(b[:4], "little")
        return (v & 0x7F800000) == 0x7F800000 and (v & 0x7FFFFF) != 0
    if kind == "f64":
        v = int.from_bytes(b[:8], "little")
        return (v >> 52 & 0x7FF) == 0x7FF and (v & ((1 << 52) - 1)) != 0
    return False


def classify(res, prog):
    """Turn a run_compare result into a list of slot-level differences, after
    treating any two single-precision NaNs as equal."""
    out = []
    if res["status"] != "DIFF":
        return out
    kinds = slot_kinds()
    nthr = prog.nt * prog.ncta
    for d in res["diffs"]:
        s, g = bytes.fromhex(d["sail"]), bytes.fromhex(d["gpgpusim"])
        if d["buffer"] == "OUT":
            for t in range(nthr):
                for i, k in enumerate(kinds):
                    o = (t * SLOTS + i) * 8
                    w = {"b32": 4, "f32": 4, "pred": 4, "b64": 8, "f64": 8, "b16": 2}[k]
                    a, b = s[o:o + w], g[o:o + w]
                    if a == b:
                        continue
                    if k == "f32" and is_nan(k, a) and is_nan(k, b):
                        continue
                    zs = k in ("f32", "f64") and int.from_bytes(a, "little") << 1 & ((1 << 8 * w) - 1) == 0 \
                        and int.from_bytes(b, "little") << 1 & ((1 << 8 * w) - 1) == 0
                    out.append({"buf": "OUT", "thread": t, "slot": i, "kind": k,
                                "reg": slot_reg(i), "sail": a[::-1].hex(), "gpgpusim": b[::-1].hex(),
                                "f64nan": k == "f64" and is_nan(k, a) and is_nan(k, b),
                                "zero_sign": zs})
        else:
            for o in range(0, max(len(s), len(g)), 4):
                if s[o:o + 4] != g[o:o + 4] and not (is_nan("f32", s[o:o + 4]) and is_nan("f32", g[o:o + 4])):
                    out.append({"buf": d["buffer"], "offset": o, "sail": s[o:o + 4][::-1].hex(),
                                "gpgpusim": g[o:o + 4][::-1].hex()})
    return out


def slot_reg(i):
    for c in ORDER:
        n = RCLS[c][0]
        if i < n:
            return "%%%s%d" % (c, i)
        i -= n
    return "?"


# ------------------------------------------------------------------ fuzzing
class Opts:
    pass


def make_opts(a):
    import run_compare
    o = Opts()
    o.sail_dir = a.sail_dir
    o.sail_bin = a.sail_bin or os.path.join(a.sail_dir, "build", "ptx")
    o.gs = a.gs
    o.timeout = a.timeout
    o.tmp = a.tmp
    o.keep = False
    run_compare.front(o.sail_dir)
    return o


def run_prog(prog, body, opts, path):
    import run_compare
    with open(path, "w") as f:
        f.write(prog.program(body))
    res = run_compare.compare_one(path, opts)
    return res, classify(res, prog)


def flatten_count(body):
    n = 0
    for s in body:
        n += 1
        if isinstance(s, tuple) and s[0] in ("if", "ifelse", "loop"):
            for sub in s[2:]:
                if isinstance(sub, list):
                    n += flatten_count(sub)
    return n


IGNORE_ZERO_SIGN = True
EXCLUDE = None   # regex: regenerate simple statements whose text matches


def interesting(res, diffs):
    """A real difference: not only NaN-vs-NaN, and (unless disabled) not only
    +0 vs -0 (GPGPU-Sim gets the sign of zero wrong in neg/abs/min/max)."""
    return res["status"] == "DIFF" and any(
        not d.get("f64nan") and not (IGNORE_ZERO_SIGN and d.get("zero_sign")) for d in diffs)


def minimise(prog, opts, path, log=print):
    """Greedy statement deletion (top level, then inside compound statements)
    keeping some real difference.  Returns the reduced body."""
    body = list(prog.body)

    def test(b):
        res, diffs = run_prog(prog, b, opts, path)
        return interesting(res, diffs)

    def reduce_list(lst, rebuild):
        i = 0
        chunk = max(1, len(lst) // 2)
        while chunk >= 1:
            i = 0
            changed = False
            while i < len(lst):
                cand = lst[:i] + lst[i + chunk:]
                if test(rebuild(cand)):
                    lst = cand
                    changed = True
                else:
                    i += chunk
            if chunk == 1 and not changed:
                break
            chunk = max(1, chunk // 2) if chunk > 1 else (1 if changed else 0)
            if chunk == 0:
                break
        return lst

    body = reduce_list(body, lambda b: b)
    # descend into compound statements; flatten them where possible
    for idx in range(len(body)):
        s = body[idx]
        if isinstance(s, tuple) and s[0] in ("if", "ifelse", "loop"):
            for inner in ([s[2]] if s[0] == "if" else [s[2], s[3]] if s[0] == "ifelse" else [s[4]]):
                flat = body[:idx] + list(inner) + body[idx + 1:]
                if test(flat):
                    body = flat
                    return reduce_more(prog, body, opts, path, test)
    return reduce_more(prog, body, opts, path, test)


def reduce_more(prog, body, opts, path, test):
    changed = True
    while changed:
        changed = False
        for i in range(len(body)):
            cand = body[:i] + body[i + 1:]
            if test(cand):
                body = cand
                changed = True
                break
        # try to flatten compounds again
        for i, s in enumerate(body):
            if isinstance(s, tuple) and s[0] in ("if", "ifelse", "loop"):
                parts = [s[2]] if s[0] == "if" else [s[2], s[3]] if s[0] == "ifelse" else [s[4]]
                for inner in parts:
                    cand = body[:i] + list(inner) + body[i + 1:]
                    if test(cand):
                        body = cand
                        changed = True
                        break
                if changed:
                    break
    return body


def body_text(prog, body):
    out = []
    prog.label = 0
    prog.render_stmts(body, out, "  ")
    return "\n".join(out)


def cmd_fuzz(a):
    global IGNORE_ZERO_SIGN, EXCLUDE
    IGNORE_ZERO_SIGN = not a.zero_sign
    EXCLUDE = a.exclude
    opts = make_opts(a)
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (int(x) for x in a.seeds.split(":"))
    cats = a.cats.split(",")
    summary = {"MATCH": 0, "DIFF": 0, "NAN-ONLY": 0, "OTHER": 0}
    import concurrent.futures as cf

    def one(seed):
        prog = Prog(seed, cats, all_forms=a.all_forms, raw=a.raw)
        path = os.path.join(a.tmp or a.out, "fz_%d.ptx" % seed)
        res, diffs = run_prog(prog, None, opts, path)
        rec = {"seed": seed, "cats": cats, "raw": a.raw, "all_forms": a.all_forms, "exclude": a.exclude,
               "status": res["status"], "why": res.get("why", "")}
        if res["status"] == "DIFF" and not interesting(res, diffs):
            rec["status"] = "NAN-ONLY"
        if interesting(res, diffs):
            rec["diffs"] = diffs[:20]
            if a.minimise:
                mpath = os.path.join(a.tmp or a.out, "fzmin_%d.ptx" % seed)
                mb = minimise(prog, opts, mpath)
                res2, d2 = run_prog(prog, mb, opts, mpath)
                rec["min_body"] = body_text(prog, mb)
                rec["min_diffs"] = d2[:12]
                with open(os.path.join(a.out, "min_%d.ptx" % seed), "w") as f:
                    f.write(prog.program(mb))
        try:
            os.unlink(path)
        except OSError:
            pass
        return rec

    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for rec in ex.map(one, range(lo, hi)):
            st = rec["status"]
            summary[st if st in summary else "OTHER"] += 1
            line = "%-9s seed %d %s" % (st, rec["seed"], rec.get("why", "")[:150])
            print(line, flush=True)
            if st == "DIFF" and "min_body" in rec:
                print("   minimised:\n" + "\n".join("     " + x for x in rec["min_body"].splitlines()))
                for d in rec["min_diffs"][:4]:
                    print("     ", d)
            with open(os.path.join(a.out, "fuzz.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
    print("---- " + ", ".join("%s %d" % kv for kv in summary.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("seed", type=int)
    g.add_argument("--cats", default=",".join(ALL_CATS))
    g.add_argument("-o", "--output")
    g.add_argument("--all-forms", action="store_true")
    g.add_argument("--raw", action="store_true")
    f = sub.add_parser("fuzz")
    f.add_argument("--cats", default=",".join(ALL_CATS))
    f.add_argument("--seeds", default="0:100")
    f.add_argument("--out", default="fuzz_out")
    f.add_argument("--minimise", action="store_true")
    f.add_argument("--exclude", default=None,
                   help="regex: do not generate simple statements matching it")
    f.add_argument("--zero-sign", action="store_true",
                   help="count +0/-0 differences as real differences")
    f.add_argument("--raw", action="store_true",
                   help="do not steer clear of known GPGPU-Sim deviations")
    f.add_argument("--all-forms", action="store_true",
                   help="also emit forms GPGPU-Sim is known not to execute")
    for p in (f,):
        p.add_argument("--sail-dir", default=os.environ.get("PTX_SAIL_DIR") or
                       os.path.normpath(os.path.join(HERE, "..", "..", "sail", "ptx")))
        p.add_argument("--sail-bin", default=None)
        p.add_argument("--gs", default=os.environ.get("GPGPUSIM_PTX") or
                       os.path.join(os.getcwd(), "_gpgpusim_build", "gpgpusim-ptx"))
        p.add_argument("--timeout", type=int, default=600)
        p.add_argument("--tmp", default=None)
        p.add_argument("-j", "--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "gen":
        text = Prog(a.seed, a.cats.split(","), all_forms=a.all_forms, raw=a.raw).program()
        if a.output:
            open(a.output, "w").write(text)
        else:
            sys.stdout.write(text)
    elif a.cmd == "fuzz":
        cmd_fuzz(a)


if __name__ == "__main__":
    main()

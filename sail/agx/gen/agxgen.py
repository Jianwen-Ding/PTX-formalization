#!/usr/bin/env python3
"""Test generation from the AGX Sail model (sail/agx).

Instructions are produced in the serialized AST form that sail/agx/agxtest.py writes for the model
(agx_main.sail rd_instr), so every generated instruction is an AST value of the Sail model.  Two
generators:

* `forms(rng)`: for every instruction of the model, every value of each of its dimensions
  (operand kinds r32 / r16l / r16h / r64 / u32 / u16l / u16h / integer and float immediates,
  register-cache hints, source modifiers, conditions, opcodes, saturation, shifts, formats, masks,
  signedness, wait groups, push counts, ...) one factor at a time, the other dimensions random;
  boundary register numbers and immediates are part of the value sets.
* `random_instr(rng)`: uniformly random instructions over the same dimensions.

`form_key(toks)` names the form of an instruction (mnemonic and modifiers, operand kinds); coverage is
counted over these keys.
"""
import random
import struct

R32, R16L, R16H, R64, U32, U16L, U16H, IMM, FIMM = "r32", "r16l", "r16h", "r64", "u32", "u16l", "u16h", "imm", "fimm"
# a 32-bit uniform at an odd 16-bit index (view 4: Mesa's packer can produce it, its value is not measured)
U32O = "u32o"

ICCS = ["eq", "ult", "ugt", "seq", "slt", "sgt", "ne", "uge", "ule", "sne", "sge", "sle"]
FCCS = ["feq", "flt", "fgt", "fltn", "fge", "fle", "fgtn", "fnequ", "fnlt", "fngt", "fnltn", "fnge", "fnle", "fngtn"]
# conditions without an inversion bit (icmpsel / fcmpsel)
ICCS_SEL = ["eq", "ult", "ugt", "seq", "slt", "sgt"]
FCCS_SEL = ["feq", "flt", "fgt", "fltn", "fge", "fle", "fgtn"]
AOPS = ["add", "sub", "xchg", "min", "imin", "max", "imax", "and", "or", "xor"]
# device atomics: the G14C forms of bits 45-46 (mode) and ops 11-15; the modes each op encodes with
DEV_AOPS = AOPS + ["cmpxchg", "fadd", "umin64", "umax64", "end11", "end12", "end13", "end14", "end15"]
DEV_PAIR = ("cmpxchg", "umin64", "umax64")
DEV_MODES = dict({o: (0, 1, 2, 3) for o in AOPS}, sub=(0, 2), cmpxchg=(0, 2), fadd=(1, 3), umin64=(3,), umax64=(1,),
                 **{"end%d" % n: (0,) for n in range(11, 16)})
FMTS = ["u8", "u16", "u32"]
DEV_FMTS = FMTS + ["f16", "u8norm", "s8norm", "u16norm", "s16norm", "rgb10a2", "srgba8", "rg11b10f", "rgb9e5", "rg11b10f.rtz",
                   "rgb9e5.rtz"]
# formats whose device accesses the model does not define (they stop it with an error)
PACKED_FMTS = ["rgb10a2", "srgba8", "rg11b10f", "rgb9e5", "rg11b10f.rtz", "rgb9e5.rtz"]
FUNOPS = ["floor", "ceil", "trunc", "rint", "rsqrt", "rsqrt_special", "sin_pt_1", "sin_pt_2", "log2", "exp2"]


def fimm_value(n):
    """applegpu decode_float_immediate"""
    s = (n >> 7) & 1
    e = (n >> 4) & 7
    f = n & 15
    v = (f / 64.0) if e == 0 else ((16 + f) / 16.0) * 2.0 ** (e - 3)
    return -v if s else v


def f64bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


FIMM_CODES = [f64bits(fimm_value(n)) for n in range(256)]
IMM_EDGE = [0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 254, 255]


def reg(f, n, view, hint=0, sx=0, ab=0, ng=0):
    return [f, n, view, hint, sx, ab, ng]


class Gen:
    def __init__(self, rng, regs=None, uregs=None, odd=True):
        self.R = rng
        # generate the unmeasured odd 32-bit uniforms (else they become u32)
        self.odd = odd
        # register numbers to draw from (None: the whole encodable range)
        self.regs = regs
        self.uregs = uregs

    def rn(self, hi=128):
        if self.regs is not None:
            c = [x for x in self.regs if x < hi]
            return self.R.choice(c)
        k = self.R.random()
        if k < 0.1:
            return self.R.choice([0, 1, hi - 1, hi - 2])
        return self.R.randrange(hi)

    def un(self):
        if self.uregs is not None:
            return self.R.choice(self.uregs)
        k = self.R.random()
        if k < 0.1:
            return self.R.choice([0, 1, 127, 128, 254, 255])
        return self.R.randrange(256)

    def even(self, n):
        return n

    # ------------------------------------------------ operands
    def src(self, kind, hint=None, sx=0, ab=0, ng=0, imm=None):
        R = self.R
        if kind == IMM:
            return ["I", imm if imm is not None else (R.choice(IMM_EDGE) if R.random() < 0.3 else R.randrange(256))]
        if kind == FIMM:
            code = FIMM_CODES[imm if imm is not None else R.randrange(256)]
            # a float immediate with modifiers (OFMod)
            return ["M", code, ab, ng] if ab or ng else ["F", code]
        if kind == U32O and (not self.odd or R.random() < 0.75):
            kind = U32
        if kind == U32O:
            return ["R"] + reg(1, self.un(), 4, 0, sx, ab, ng)
        h = R.choice((0, 1, 2)) if hint is None else hint
        if kind == R32:
            return ["R"] + reg(0, self.rn(), 0, h, sx, ab, ng)
        if kind == R16L:
            return ["R"] + reg(0, self.rn(), 1, h, sx, ab, ng)
        if kind == R16H:
            return ["R"] + reg(0, self.rn(), 2, h, sx, ab, ng)
        if kind == R64:
            return ["R"] + reg(0, self.rn(127), 3, h, sx, ab, ng)
        if kind == U32:
            return ["R"] + reg(1, self.un(), 0, 0, sx, ab, ng)
        if kind == U16L:
            return ["R"] + reg(1, self.un(), 1, 0, sx, ab, ng)
        if kind == U16H:
            return ["R"] + reg(1, self.un(), 2, 0, sx, ab, ng)
        raise ValueError(kind)

    def dst(self, kind, hint=None):
        h = self.R.choice((0, 1)) if hint is None else hint
        view = {R32: 0, R16L: 1, R16H: 2, R64: 3}[kind]
        return reg(0, self.rn(127 if view == 3 else 128), view, h)

    def pick(self, xs):
        return self.R.choice(xs)


# ---------------------------------------------------------------------------------------------
# instruction schemas: name -> (dimensions, builder).  A dimension is (name, list of values); the
# builder maps a dict of chosen values to the token list.
INT_SRC = [IMM, R32, R16L, R16H, U32, U16L, U16H, U32O]
INT_SRC64 = INT_SRC + [R64]
FLT_SRC = [FIMM, R32, R16L, R16H, U32, U16L, U16H, U32O]
FLT16_SRC = [FIMM, R16L, R16H, U16L, U16H, U32]
DST = [R32, R16L, R16H]
DST64 = DST + [R64]
HINTS = [0, 1, 2]
BOOL = [0, 1]


def _srcmods(c, i, g, kind, mods):
    """source operand i with its modifiers from the choice dict"""
    if kind == IMM:
        return g.src(kind)
    if kind == FIMM:
        return g.src(kind, ab=c.get("abs%d" % i, 0) if "f" in mods else 0, ng=c.get("neg%d" % i, 0) if "f" in mods else 0)
    sx = c.get("sx%d" % i, 0) if "sx" in mods else 0
    ab = c.get("abs%d" % i, 0) if "f" in mods else 0
    ng = c.get("neg%d" % i, 0) if "f" in mods else 0
    h = c.get("hint%d" % i)
    if kind.startswith("u"):
        h = 0
    return g.src(kind, hint=h, sx=sx, ab=ab, ng=ng)


def schemas():
    S = {}

    def add(name, dims, build):
        S[name] = (dims, build)

    def srcdims(i, kinds, mods=""):
        d = [("s%d" % i, kinds), ("hint%d" % i, HINTS)]
        if "sx" in mods:
            d.append(("sx%d" % i, BOOL))
        if "f" in mods:
            d += [("abs%d" % i, BOOL), ("neg%d" % i, BOOL)]
        return d

    add("mov_imm", [("d", DST), ("imm", ["edge", "rand"]), ("dhint", [0, 1])],
        lambda c, g: ["MOV"] + g.dst(c["d"], c["dhint"]) +
        ["I", (g.pick([0, 1, 0x7fff, 0x8000, 0xffff]) if c["d"] != R32 else g.pick([0, 1, 0x7fffffff, 0x80000000, 0xffffffff]))
         if c["imm"] == "edge" else g.R.randrange(1 << (32 if c["d"] == R32 else 16))])
    add("mov", [("d", DST), ("dhint", [0, 1])] + srcdims(0, [R32, R16L, R16H, U32, U16L, U16H]),
        lambda c, g: ["MOV"] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], ""))
    add("get_sr", [("d", DST), ("dhint", [0, 1]), ("sr", "range256")],
        lambda c, g: ["GET_SR"] + g.dst(c["d"], c["dhint"]) + [c["sr"]])
    for sub in (0, 1):
        for sat in (0, 1):
            nm = ("isub" if sub else "iadd") + (".sat" if sat else "")
            add(nm, [("d", DST64), ("dhint", [0, 1]), ("shift", list(range(8)))] + srcdims(0, INT_SRC64, "sx") + srcdims(1, INT_SRC64, "sx"),
                lambda c, g, sub=sub, sat=sat: ["IADD", sub, sat] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "sx") +
                _srcmods(c, 1, g, c["s1"], "sx") + [c["shift"]])
            nm = ("imsub" if sub else "imadd") + (".sat" if sat else "")
            add(nm, [("d", DST64), ("dhint", [0, 1]), ("shift", list(range(8)))] + srcdims(0, INT_SRC64, "sx") + srcdims(1, INT_SRC64, "sx") +
                srcdims(2, INT_SRC64, "sx"),
                lambda c, g, sub=sub, sat=sat: ["IMADD", sub, sat] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "sx") +
                _srcmods(c, 1, g, c["s1"], "sx") + _srcmods(c, 2, g, c["s2"], "sx") + [c["shift"]])
    add("bitop", [("tt", list(range(16))), ("d", DST), ("dhint", [0, 1])] + srcdims(0, INT_SRC) + srcdims(1, INT_SRC),
        lambda c, g: _bitop(c, g))
    for nm in ("bfi", "bfeil"):
        add(nm, [("m", list(range(32))), ("d", DST), ("dhint", [0, 1])] + srcdims(0, INT_SRC) + srcdims(1, INT_SRC) + srcdims(2, INT_SRC),
            lambda c, g, nm=nm: [nm.upper(), c["m"]] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "") +
            _srcmods(c, 1, g, c["s1"], "") + _srcmods(c, 2, g, c["s2"], ""))
    add("asr", [("d", DST), ("dhint", [0, 1])] + srcdims(0, INT_SRC) + srcdims(1, INT_SRC),
        lambda c, g: ["ASR"] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "") + _srcmods(c, 1, g, c["s1"], ""))
    for o in ("popcount", "bitrev", "ffs"):
        add(o, [("d", DST), ("dhint", [0, 1])] + srcdims(0, INT_SRC),
            lambda c, g, o=o: ["IUNOP", o] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], ""))
    for o in ("add", "mul"):
        for h in (0, 1):
            nm = "f%s%d" % (o, 16 if h else 32)
            srcs = FLT16_SRC if h else FLT_SRC
            add(nm, [("sat", BOOL), ("d", [R16L, R16H] if h else DST), ("dhint", [0, 1])] + srcdims(0, srcs, "f") + srcdims(1, srcs, "f"),
                lambda c, g, o=o, h=h: ["FBIN", o, h, c["sat"]] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "f") +
                _srcmods(c, 1, g, c["s1"], "f"))
    for h in (0, 1):
        srcs = FLT16_SRC if h else FLT_SRC
        add("fmadd%d" % (16 if h else 32), [("sat", BOOL), ("d", [R16L, R16H] if h else DST), ("dhint", [0, 1])] + srcdims(0, srcs, "f") +
            srcdims(1, srcs, "f") + srcdims(2, srcs, "f"),
            lambda c, g, h=h: ["FMADD", h, c["sat"]] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "f") +
            _srcmods(c, 1, g, c["s1"], "f") + _srcmods(c, 2, g, c["s2"], "f"))
    add("rcp", [("sat", BOOL), ("d", DST), ("dhint", [0, 1])] + srcdims(0, FLT_SRC, "f"),
        lambda c, g: ["RCP", c["sat"]] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "f"))
    for o in FUNOPS:
        add(o, [("sat", BOOL), ("d", DST), ("dhint", [0, 1])] + srcdims(0, FLT_SRC, "f"),
            lambda c, g, o=o: ["FUNOP", o, c["sat"]] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "f"))
    for f in (0, 1):
        nm = "fcmpsel" if f else "icmpsel"
        add(nm, [("cc", FCCS_SEL if f else ICCS_SEL), ("d", DST), ("dhint", [0, 1])] + srcdims(0, FLT_SRC if f else INT_SRC, "f" if f else "") +
            srcdims(1, FLT_SRC if f else INT_SRC, "f" if f else "") + [("x", ["imm", "gpr", "uni"]), ("y", ["imm", "gpr", "uni"])],
            lambda c, g, f=f: _cmpsel(c, g, f))
    for k in ("if", "else", "while"):
        for f in (0, 1):
            nm = "%s_%scmp" % (k, "f" if f else "i")
            add(nm, [("cc", FCCS if f else ICCS), ("n", [0, 1, 2, 3])] + srcdims(0, FLT_SRC if f else INT_SRC, "f" if f else "") +
                srcdims(1, FLT_SRC if f else INT_SRC, "f" if f else ""),
                lambda c, g, k=k, f=f: ["STACKOP", k, f, c["cc"]] + _srcmods(c, 0, g, c["s0"], "f" if f else "") +
                _srcmods(c, 1, g, c["s1"], "f" if f else "") + [c["n"]])
    add("pop_exec", [("n", [0, 1, 2, 3])], lambda c, g: ["POP_EXEC", c["n"]])
    for a in (0, 1):
        add("jmp_exec_" + ("any" if a else "none"), [("off", ["small", "neg", "big"])],
            lambda c, g, a=a: ["JMP", a, "JR", {"small": g.R.randrange(0, 256, 2), "neg": -g.R.randrange(2, 256, 2),
                                                 "big": g.R.randrange(-(1 << 31), 1 << 31) & ~1}[c["off"]]])
    for f in (0, 1):
        nm = "fcmp_ballot" if f else "icmp_ballot"
        add(nm, [("cc", FCCS if f else ICCS), ("d", DST), ("dhint", [0, 1])] + srcdims(0, FLT_SRC if f else INT_SRC, "f" if f else "") +
            srcdims(1, FLT_SRC if f else INT_SRC, "f" if f else ""),
            lambda c, g, f=f: ["BALLOT", f, c["cc"]] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "f" if f else "") +
            _srcmods(c, 1, g, c["s1"], "f" if f else ""))
    for k in ("idx", "xor", "up", "down"):
        add("simd_shuffle" + ("" if k == "idx" else "_" + k), [("d", DST), ("dhint", [0, 1])] + srcdims(0, INT_SRC) +
            srcdims(1, [IMM, R16L, R16H, U16L, U16H, R32, U32]),
            lambda c, g, k=k: ["SHUFFLE", k] + g.dst(c["d"], c["dhint"]) + _srcmods(c, 0, g, c["s0"], "") + _srcmods(c, 1, g, c["s1"], ""))
    for st in (0, 1):
        add("device_" + ("store" if st else "load"),
            [("fmt", DEV_FMTS), ("mask", list(range(16))), ("r", [R32, R16L, R16H]), ("base", ["gpr", "uni"]), ("idx", ["imm", "gpr"]),
             ("sg", BOOL), ("shift", [0, 1, 2]), ("g", BOOL), ("u6", [0, 1, 2, 3])],
            lambda c, g, st=st: _devmem(c, g, st))
        add("threadgroup_" + ("store" if st else "load"),
            [("fmt", FMTS), ("mask", list(range(16))), ("r", [R32, R16L, R16H]), ("base", ["zero", "r16", "u16"]), ("idx", ["imm", R16L, R16H])],
            lambda c, g, st=st: _tgmem(c, g, st))
    add("atomic", [("op", DEV_AOPS), ("dst", ["reg", "none"]), ("base", ["gpr", "uni"]), ("idx", ["imm", "gpr"]), ("sg", BOOL), ("g", BOOL),
                   ("mode", [0, 1, 2, 3])],
        lambda c, g: _devatom(c, g))
    add("threadgroup_atomic", [("op", AOPS), ("dst", [R32, R16L, R16H, "none"]), ("base", ["zero", "r16", "u16"]), ("idx", ["imm", R16L, R16H])],
        lambda c, g: _tgatom(c, g))
    add("wait", [("n", [0, 1])], lambda c, g: ["WAIT", c["n"]])
    add("threadgroup_barrier", [], lambda c, g: ["TG_BARRIER"])
    add("memory_barrier", [("a", [0, 1, 2, 3]), ("b", [0, 1, 2, 3]), ("c", list(range(16)))],
        lambda c, g: ["MEMORY_BARRIER", c["a"], c["b"], c["c"]])
    add("stop", [], lambda c, g: ["STOP"])
    add("trap", [], lambda c, g: ["TRAP"])
    return S


def _bitop(c, g):
    a = _srcmods(c, 0, g, c["s0"], "")
    b = _srcmods(c, 1, g, c["s1"], "")
    if c["tt"] == 10 and b == ["I", 0]:
        b = ["I", 1]   # not with B = 0 is the `not` alias (encoded the same as bitop 10 with B = 0)
    return ["BITOP", c["tt"]] + g.dst(c["d"], c["dhint"]) + a + b


def _csel(g, kind, is32):
    if kind == "imm":
        return ["I", g.R.randrange(256)]
    if kind == "gpr":
        return g.src(R32 if is32 else g.pick([R16L, R16H]))
    return g.src(g.pick([U32, U32, U32O]) if is32 else g.pick([U16L, U16H]))


def _cmpsel(c, g, f):
    d = g.dst(c["d"], c["dhint"])
    is32 = d[2] == 0
    return ["CMPSEL", f, c["cc"]] + d + _srcmods(c, 0, g, c["s0"], "f" if f else "") + _srcmods(c, 1, g, c["s1"], "f" if f else "") + \
        _csel(g, c["x"], is32) + _csel(g, c["y"], is32)


def _pair(g, kind):
    if kind == "uni":
        lo = g.un() if g.uregs is None else g.R.choice(g.uregs)
        lo = min(lo, 254)
        return [1, lo, lo + 1]
    lo = g.rn(127)
    return [0, lo, lo + 1]


def _devmem(c, g, st):
    view = {R32: 0, R16L: 1, R16H: 2}[c["r"]]
    fmt = c["fmt"]
    n = bin(c["mask"]).count("1")
    hi = 128 - max(n, 1) if view == 0 else 128 - (max(n, 1) + 1) // 2
    r = reg(0, g.rn(max(hi, 1)), view)
    idx = ["I", g.R.choice([0, 1, -1, 32767, -32768]) if g.R.random() < 0.3 else g.R.randrange(-32768, 32768)] if c["idx"] == "imm" \
        else ["R"] + reg(0, g.rn(128), 0)
    return ["DEVMEM", st, fmt, c["mask"]] + r + _pair(g, c["base"]) + idx + [c["sg"], c["shift"], c["g"], c["u6"]]


def _tgbase(g, kind):
    if kind == "zero":
        return ["I", 0]
    return g.src(g.pick([R16L, R16H]) if kind == "r16" else g.pick([U16L, U16H]), hint=0)


def _tgidx(g, kind):
    if kind == "imm":
        return ["I", g.R.choice([0, 1, -1, 32767, -32768]) if g.R.random() < 0.3 else g.R.randrange(-32768, 32768)]
    return g.src(kind, hint=0)


def _tgmem(c, g, st):
    view = {R32: 0, R16L: 1, R16H: 2}[c["r"]]
    n = bin(c["mask"]).count("1")
    hi = 128 - max(n, 1) if view == 0 else 128 - (max(n, 1) + 1) // 2
    r = reg(0, g.rn(max(hi, 1)), view)
    return ["TGMEM", st, c["fmt"], c["mask"]] + r + _tgbase(g, c["base"]) + _tgidx(g, c["idx"]) + [0]


def _devatom(c, g):
    d = ["S"] + reg(0, g.rn(128), 0) if c["dst"] == "reg" else ["N"]
    idx = ["I", g.R.randrange(-32768, 32768)] if c["idx"] == "imm" else ["R"] + reg(0, g.rn(128), 0)
    o = c["op"]
    v = reg(0, g.rn(127), 3) if o in DEV_PAIR else reg(0, g.rn(128), 0)
    # a mode the op does not encode with is replaced by one it does (the op's dimension covers every op)
    m = c["mode"] if c["mode"] in DEV_MODES[o] else g.pick(list(DEV_MODES[o]))
    return ["DEVATOM", o] + d + _pair(g, c["base"]) + idx + [c["sg"]] + ["R"] + v + [c["g"], m]


def _tgatom(c, g):
    if c["dst"] == "none":
        d = ["N"]
    else:
        d = ["S"] + reg(0, g.rn(128), {R32: 0, R16L: 1, R16H: 2}[c["dst"]])
    return ["TGATOM", c["op"]] + d + _tgbase(g, c["base"]) + _tgidx(g, c["idx"]) + ["R"] + reg(0, g.rn(128), 0)


SCHEMAS = schemas()


def _values(v, R):
    if v == "range256":
        return [0, 1, 2, 63, 64, 127, 128, 255] + [R.randrange(256) for _ in range(8)]
    return v


def _choose(dims, R, fixed=None):
    c = {}
    for name, vals in dims:
        c[name] = R.choice(_values(vals, R))
    if fixed:
        c.update(fixed)
    return c


def forms(R, g=None, names=None, reps=1):
    """one-factor-at-a-time enumeration over every dimension of every instruction schema"""
    g = g or Gen(R)
    out = []
    for name in sorted(names or SCHEMAS):
        dims, build = SCHEMAS[name]
        out.append(build(_choose(dims, R), g))
        for dn, vals in dims:
            for v in _values(vals, R):
                for _ in range(reps):
                    out.append(build(_choose(dims, R, {dn: v}), g))
    return out


def random_instr(R, g=None, names=None):
    g = g or Gen(R)
    name = R.choice(sorted(names or SCHEMAS))
    dims, build = SCHEMAS[name]
    return build(_choose(dims, R), g)


# ---------------------------------------------------------------------------------------------
def _okind(t, i):
    """operand at token index i -> (kind string, next index)"""
    k = t[i]
    if k == "I":
        return "imm", i + 2
    if k == "F":
        return "fimm", i + 2
    if k == "M":
        return "fimm" + (".abs" if t[i + 2] else "") + (".neg" if t[i + 3] else ""), i + 4
    f, n, v, h, sx, ab, ng = t[i + 1:i + 8]
    s = ("u" if f else "r") + {0: "32", 1: "16l", 2: "16h", 3: "64", 4: "32.odd"}[v]
    s += {0: "", 1: ".cache", 2: ".discard"}[h] + (".sx" if sx else "") + (".abs" if ab else "") + (".neg" if ng else "")
    return s, i + 8


def _dkind(t, i):
    f, n, v, h = t[i:i + 4]
    return ("r" + {0: "32", 1: "16l", 2: "16h", 3: "64"}[v]) + (".cache" if h == 1 else ""), i + 7


def form_key(t):
    c = t[0]
    try:
        if c == "MOV":
            d, i = _dkind(t, 1)
            o, _ = _okind(t, i)
            return "mov%s %s, %s" % ("_imm" if o == "imm" else "", d, o)
        if c == "GET_SR":
            d, i = _dkind(t, 1)
            return "get_sr %s, sr%d" % (d, t[i])
        if c in ("IADD", "IMADD"):
            nm = ("i" + ("m" if c == "IMADD" else "") + ("sub" if t[1] else "add")) + (".sat" if t[2] else "")
            d, i = _dkind(t, 3)
            ops = []
            for _ in range(2 if c == "IADD" else 3):
                o, i = _okind(t, i)
                ops.append(o)
            return "%s %s, %s, lsl %d" % (nm, d, ", ".join(ops), t[i])
        if c == "BITOP":
            d, i = _dkind(t, 2)
            a, i = _okind(t, i)
            b, i = _okind(t, i)
            return "bitop %d %s, %s, %s" % (t[1], d, a, b)
        if c in ("BFI", "BFEIL"):
            d, i = _dkind(t, 2)
            ops = []
            for _ in range(3):
                o, i = _okind(t, i)
                ops.append(o)
            return "%s m%d %s, %s" % (c.lower(), t[1], d, ", ".join(ops))
        if c == "ASR":
            d, i = _dkind(t, 1)
            a, i = _okind(t, i)
            b, i = _okind(t, i)
            return "asr %s, %s, %s" % (d, a, b)
        if c == "IUNOP":
            d, i = _dkind(t, 2)
            a, i = _okind(t, i)
            return "%s %s, %s" % (t[1], d, a)
        if c == "FBIN":
            d, i = _dkind(t, 4)
            a, i = _okind(t, i)
            b, i = _okind(t, i)
            return "f%s%s%s %s, %s, %s" % (t[1], "16" if t[2] else "32", ".sat" if t[3] else "", d, a, b)
        if c == "FMADD":
            d, i = _dkind(t, 3)
            ops = []
            for _ in range(3):
                o, i = _okind(t, i)
                ops.append(o)
            return "fmadd%s%s %s, %s" % ("16" if t[1] else "32", ".sat" if t[2] else "", d, ", ".join(ops))
        if c == "RCP":
            d, i = _dkind(t, 2)
            a, i = _okind(t, i)
            return "rcp%s %s, %s" % (".sat" if t[1] else "", d, a)
        if c == "FUNOP":
            d, i = _dkind(t, 3)
            a, i = _okind(t, i)
            return "%s%s %s, %s" % (t[1], ".sat" if t[2] else "", d, a)
        if c == "CMPSEL":
            d, i = _dkind(t, 3)
            ops = []
            for _ in range(4):
                o, i = _okind(t, i)
                ops.append(o)
            return "%scmpsel %s %s, %s" % ("f" if t[1] else "i", t[2], d, ", ".join(ops))
        if c == "STACKOP":
            a, i = _okind(t, 4)
            b, i = _okind(t, i)
            return "%s_%scmp %s %s, %s, n%d" % (t[1], "f" if t[2] else "i", t[3], a, b, t[i])
        if c == "POP_EXEC":
            return "pop_exec %d" % t[1]
        if c == "JMP":
            off = t[3]
            return "jmp_exec_%s %s" % ("any" if t[1] else "none", "fwd" if off >= 0 else "back")
        if c == "BALLOT":
            d, i = _dkind(t, 3)
            a, i = _okind(t, i)
            b, i = _okind(t, i)
            return "%scmp_ballot %s %s, %s, %s" % ("f" if t[1] else "i", t[2], d, a, b)
        if c == "SHUFFLE":
            d, i = _dkind(t, 2)
            a, i = _okind(t, i)
            b, i = _okind(t, i)
            return "simd_shuffle_%s %s, %s, %s" % (t[1], d, a, b)
        if c == "DEVMEM":
            r, i = _dkind(t, 4)
            pf = t[i]
            i += 3
            ik = "imm" if t[i] == "I" else "r32"
            i += 2 if t[i] == "I" else 8
            sg, s, gg, u6 = t[i:i + 4]
            return "device_%s %s m%d %s, %s, %s, %s, lsl %d, g%d, u6=%d" % ("store" if t[1] else "load", t[2], t[3], r.split(".")[0],
                                                                          "u" if pf else "r", ik, "signed" if sg else "unsigned", s, gg, u6)
        if c == "TGMEM":
            r, i = _dkind(t, 4)
            b, i = _okind(t, i)
            x, i = _okind(t, i)
            return "threadgroup_%s %s m%d %s, %s, %s" % ("store" if t[1] else "load", t[2], t[3], r.split(".")[0], b, x)
        if c == "DEVATOM":
            i = 2
            if t[i] == "S":
                d = "r32"
                i += 8
            else:
                d = "none"
                i += 1
            pf = t[i]
            i += 3
            ik = "imm" if t[i] == "I" else "r32"
            i += 2 if t[i] == "I" else 8
            sg = t[i]
            i += 1 + 8
            return "atomic %s %s, %s, %s, %s, g%d, mode %d" % (t[1], d, "u" if pf else "r", ik, "signed" if sg else "unsigned", t[i], t[i + 1])
        if c == "TGATOM":
            i = 2
            if t[i] == "S":
                d, i = _dkind(t, i + 1)
            else:
                d = "none"
                i += 1
            b, i = _okind(t, i)
            x, i = _okind(t, i)
            return "threadgroup_atomic %s %s, %s, %s" % (t[1], d, b, x)
        if c == "WAIT":
            return "wait %d" % t[1]
        if c == "MEMORY_BARRIER":
            return "memory_barrier %d, %d, %d" % (t[1], t[2], t[3])
        return c.lower()
    except (IndexError, KeyError):
        return c.lower() + " ?"


def _regs(t):
    """the serialized registers (file, number, view, ...) in a token list"""
    out, i = [], 0
    while i < len(t):
        if t[i] in ("R", "S") and i + 7 < len(t) + 1 and all(isinstance(x, int) for x in t[i + 1:i + 8]):
            out.append(t[i + 1:i + 8])
            i += 8
        else:
            i += 1
    return out


def unmeasured(t):
    """the instruction has a form whose execution the model leaves undefined (it stops with an error)"""
    if any(r[2] == 4 for r in _regs(t)):
        return "odd 32-bit uniform"
    if t[0] == "IMADD":
        # the sources A, B, C after the destination: A or B a register pair
        ops, i = [], 10
        for _ in range(3):
            if t[i] == "R":
                ops.append(t[i + 3]); i += 8
            else:
                ops.append(None); i += 2
        if 3 in ops[:2]:
            return "imadd 64-bit multiplicand"
    if t[0] == "SHUFFLE" and _regs(t[1:])[-1:] and _regs(t)[-1][2] in (0, 4) and t[-8] == "R":
        return "32-bit lane index"
    if t[0] == "DEVMEM" and t[2] in PACKED_FMTS:
        return "packed format"
    return None


def class_of(t):
    """the Sail-model instruction class name (as in the schema table) of a token list"""
    k = form_key(t)
    return k.split()[0].replace("_imm", "_imm")

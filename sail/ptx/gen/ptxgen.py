#!/usr/bin/env python3
"""PTX test generation from the PTX Sail model (sail/ptx).

Each generated file is a PTX module with one kernel per instruction form and a harness (.buffer / .launch /
.expect, as tests/*.ptx).  A kernel reads its operands for thread t from 8-byte slots of an input buffer,
executes the instruction, and writes its results (predicates as 0 / 1) to 8-byte slots of an output buffer.
The .expect values are computed by running the Sail PTX model on the program (with .print in place of the
.expect directives), so the files are self-checking tests of the model's semantics for other
implementations (ZLUDA + agx_run: sail/ptx/gen/ptxdiff.py).

Families (`FAMILIES`): integer arithmetic and bit operations, carry chains, floating point (f32 / f64 with
every rounding mode, .ftz, .sat), half precision (f16 / f16x2 / bf16 / bf16x2), transcendentals, comparisons
and selection (setp / set / selp / slct with every comparison operator), conversions (all type pairs and
rounding / saturation modifiers), memory (ld / st in .global / .shared / .local, vector widths, cache
operators), atomics and reductions (.global / .shared, every operation and type), warp operations (shfl,
vote, match, redux, activemask), video instructions, and a few barriers.  Operand values are boundary values
of each type (zeros, ones, extremes, subnormals, infinities, NaNs, ties) mixed with random values.

usage: ptxgen.py OUTDIR [-s SEED] [--families a,b] [--threads N] [--no-expect]
"""
import argparse, os, random, re, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PTXDIR = os.path.dirname(HERE)
sys.path.insert(0, PTXDIR)
sys.dont_write_bytecode = True

# ---------------------------------------------------------------- values
WIDTH = {"b8": 8, "u8": 8, "s8": 8, "b16": 16, "u16": 16, "s16": 16, "f16": 16, "bf16": 16, "b32": 32, "u32": 32, "s32": 32,
         "f32": 32, "f16x2": 32, "bf16x2": 32, "b64": 64, "u64": 64, "s64": 64, "f64": 64, "pred": 1, "e4m3x2": 16, "e5m2x2": 16}


def f32b(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f64b(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


F32 = [0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0x3f000000, 0x7f800000, 0xff800000, 0x7fc00000, 0x7fa00001,
       0xffc00000, 0x00000001, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x3fc00000, 0x40490fdb, 0x4b000000,
       0x4b000001, 0x3effffff, 0x3f7fffff, 0x33800000, 0x34000000, 0x00400000, 0x5f000000, 0xdf000000, 0x4f000000, 0xcf000000,
       0x3fa00000, 0x40200000, 0xc0200000, 0x3e800000, 0x41200000, 0x447a0000]
F64 = [0x0, 0x8000000000000000, 0x3ff0000000000000, 0xbff0000000000000, 0x4000000000000000, 0x3fe0000000000000,
       0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000000, 0x7ff4000000000001, 0x1, 0x800fffffffffffff,
       0x0010000000000000, 0x7fefffffffffffff, 0xffefffffffffffff, 0x3ff8000000000000, 0x400921fb54442d18,
       0x4330000000000000, 0x3fdfffffffffffff, 0x3ca0000000000000, 0x43e0000000000000, 0xc3e0000000000000,
       0x41e0000000000000, 0x3ff0000000000001, 0x36a0000000000000, 0x3810000000000000, 0x47efffffe0000000,
       0x3ff4000000000000, 0x4004000000000000, 0x3e70000000000000]
F16 = [0x0000, 0x8000, 0x3c00, 0xbc00, 0x4000, 0x3800, 0x7c00, 0xfc00, 0x7e00, 0x7d01, 0xfe00, 0x0001, 0x83ff, 0x0400, 0x8400, 0x7bff,
       0xfbff, 0x3e00, 0x4248, 0x6400, 0x6401, 0x37ff, 0x3bff, 0x1400, 0x1000, 0x0200, 0x3d00, 0x4100, 0xc100, 0x3400]
BF16 = [0x0000, 0x8000, 0x3f80, 0xbf80, 0x4000, 0x3f00, 0x7f80, 0xff80, 0x7fc0, 0x7fa1, 0x0001, 0x807f, 0x0080, 0x7f7f, 0xff7f,
        0x3fc0, 0x4049, 0x4b00, 0x3eff, 0x3f7f, 0x3400, 0x3fa0, 0x4020]


def ints(w):
    m = (1 << w) - 1
    h = 1 << (w - 1)
    v = [0, 1, 2, 3, m, m - 1, h, h - 1, h + 1, 0x55555555 & m, 0xaaaaaaaa & m, 7, 8, 15, 16, 31, 32, 33, 63, 64, 255, 256,
         0xffff & m, 0x10000 & m, 0x8000 & m, 0x7fff & m, 0x80000000 & m, 0x7fffffff & m, (m ^ 0xff) & m, 12345 & m, 1000000 & m]
    return sorted(set(v))


def values(cls, R, n):
    """n operand values (bit patterns) of value class cls"""
    if cls in ("f32", "ftz32"):
        base = F32
        rnd = lambda: R.getrandbits(32) if R.random() < 0.5 else f32b(R.uniform(-100, 100))
    elif cls == "f64":
        base = F64
        rnd = lambda: R.getrandbits(64) if R.random() < 0.5 else f64b(R.uniform(-100, 100))
    elif cls == "f16":
        base = F16
        rnd = lambda: R.getrandbits(16)
    elif cls == "bf16":
        base = BF16
        rnd = lambda: R.getrandbits(16)
    elif cls == "f16x2":
        base = [a | b << 16 for a, b in zip(F16, F16[::-1])]
        rnd = lambda: R.choice(F16) | R.choice(F16) << 16 if R.random() < 0.5 else R.getrandbits(32)
    elif cls == "bf16x2":
        base = [a | b << 16 for a, b in zip(BF16, BF16[::-1])]
        rnd = lambda: R.choice(BF16) | R.choice(BF16) << 16 if R.random() < 0.5 else R.getrandbits(32)
    elif cls == "pred":
        return [R.randrange(2) for _ in range(n)]
    elif cls.startswith("sh"):          # shift / bit position amounts
        w = int(cls[2:])
        base = [0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 0xffffffff, w - 1, w]
        rnd = lambda: R.randrange(2 * w + 2)
    elif cls.startswith("small"):
        base = list(range(0, 8))
        rnd = lambda: R.randrange(8)
    else:
        w = WIDTH[cls] if cls in WIDTH else int(cls[1:])
        base = ints(w)
        rnd = lambda: R.getrandbits(w)
    out = list(base)
    R.shuffle(out)
    while len(out) < n:
        out.append(rnd() if R.random() < 0.6 else R.choice(base))
    return out[:n]


# ---------------------------------------------------------------- kernels
def regw(t):
    """register width for a PTX type"""
    w = WIDTH[t]
    return 16 if w == 8 else (1 if w == 1 else w)


class Form:
    """one instruction form: `text` with {d0}.. {a0}.. placeholders; dsts / srcs are lists of
    (PTX type, value class); `pre` / `post` extra statements"""
    def __init__(self, text, dsts, srcs, name=None, threads=None, body=None, setup=None, launch_extra=None):
        self.text = text
        self.dsts = dsts
        self.srcs = srcs
        self.name = name or text
        self.threads = threads
        self.body = body          # custom kernel body (memory, warp forms): function(form) -> statements
        self.setup = setup


def kernel(i, f, nthreads):
    nin, nout = max(len(f.srcs), 1), max(len(f.dsts), 1)
    decl = [".reg .pred %p<8>;", ".reg .b16 %h<8>;", ".reg .b32 %r<16>;", ".reg .b64 %rd<16>;"]
    body = ["mov.u32 %r0, %tid.x;", "mul.wide.u32 %rd0, %r0, {};".format(8 * nin), "ld.param.u64 %rd1, [pi];",
            "add.s64 %rd1, %rd1, %rd0;"]
    subst = {}
    for k, (t, _) in enumerate(f.srcs):
        w = regw(t)
        name = "%a{}".format(k)
        if w == 1:
            decl.append(".reg .pred %a{}; .reg .b32 %ai{};".format(k, k))
            body.append("ld.global.b32 %ai{0}, [%rd1+{1}];".format(k, 8 * k))
            body.append("setp.ne.b32 %a{0}, %ai{0}, 0;".format(k))
        else:
            decl.append(".reg .b{} {};".format(w, name))
            body.append("ld.global.{} {}, [%rd1+{}];".format("u8" if WIDTH[t] == 8 else "b%d" % w, name, 8 * k))
        subst["a%d" % k] = name
    for k, (t, _) in enumerate(f.dsts):
        w = regw(t)
        if w == 1:
            decl.append(".reg .pred %d{0}; .reg .b32 %di{0};".format(k))
        else:
            decl.append(".reg .b{} %d{};".format(w, k))
            body.append("mov.b{} %d{}, 0;".format(w, k))
        subst["d%d" % k] = "%d{}".format(k)
    if f.body:
        body += f.body(f, subst)
    else:
        body.append(f.text.format(**subst) + ";")
    body += ["mul.wide.u32 %rd2, %r0, {};".format(8 * nout), "ld.param.u64 %rd3, [po];", "add.s64 %rd3, %rd3, %rd2;"]
    for k, (t, _) in enumerate(f.dsts):
        w = regw(t)
        if w == 1:
            body.append("selp.u32 %di{0}, 1, 0, %d{0};".format(k))
            body.append("st.global.b32 [%rd3+{}], %di{};".format(8 * k, k))
        else:
            body.append("st.global.b{} [%rd3+{}], %d{};".format(w, 8 * k, k))
    body.append("ret;")
    head = ".visible .entry k{}(.param .u64 pi, .param .u64 po)\n{{".format(i)
    return "// {}\n{}\n    {}\n    {}\n}}\n".format(f.name, head, "\n    ".join(decl), "\n    ".join(body))


# ---------------------------------------------------------------- families
def T(t):
    return (t, t)


def fam_int(R):
    F = []
    for t in ("s16", "u16", "s32", "u32", "s64", "u64"):
        w = WIDTH[t]
        for op in ("add", "sub"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
        for op in ("mul.lo", "mul.hi"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
        if w < 64:
            F.append(Form("mul.wide.%s {d0}, {a0}, {a1}" % t, [("b%d" % (2 * w), t)], [T(t), T(t)]))
            F.append(Form("mad.wide.%s {d0}, {a0}, {a1}, {a2}" % t, [("b%d" % (2 * w), t)], [T(t), T(t), ("b%d" % (2 * w), t)]))
        for op in ("mad.lo", "mad.hi"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}, {a2}" % (op, t), [T(t)], [T(t), T(t), T(t)]))
        for op in ("div", "rem", "min", "max"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
        if t[0] == "s":
            F.append(Form("abs.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
            F.append(Form("neg.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
        F.append(Form("sad.%s {d0}, {a0}, {a1}, {a2}" % t, [T(t)], [T(t), T(t), T(t)]))
        b = "b%d" % w
        F.append(Form("shl.%s {d0}, {a0}, {a1}" % b, [T(b)], [T(b), ("u32", "sh%d" % w)]))
        F.append(Form("shr.%s {d0}, {a0}, {a1}" % t, [T(t)], [T(t), ("u32", "sh%d" % w)]))
        for op in ("and", "or", "xor"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}" % (op, b), [T(b)], [T(b), T(b)]))
        F.append(Form("not.%s {d0}, {a0}" % b, [T(b)], [T(b)]))
        F.append(Form("cnot.%s {d0}, {a0}" % b, [T(b)], [T(b)]))
    for t in ("s32",):
        F.append(Form("add.sat.s32 {d0}, {a0}, {a1}", [T(t)], [T(t), T(t)]))
        F.append(Form("sub.sat.s32 {d0}, {a0}, {a1}", [T(t)], [T(t), T(t)]))
        F.append(Form("mad.hi.sat.s32 {d0}, {a0}, {a1}, {a2}", [T(t)], [T(t), T(t), T(t)]))
    for t in ("s16", "u16", "s32", "u32"):
        F.append(Form("min.relu.s32 {d0}, {a0}, {a1}" if t == "s32" else "min.%s {d0}, {a0}, {a1}" % t, [T(t)], [T(t), T(t)],
                      name="min.relu.s32" if t == "s32" else None))
    for t in ("s32", "u32"):
        for op in ("mul24.lo", "mul24.hi"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
        for op in ("mad24.lo", "mad24.hi"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}, {a2}" % (op, t), [T(t)], [T(t), T(t), T(t)]))
    for t in ("b32", "b64"):
        F.append(Form("popc.%s {d0}, {a0}" % t, [T("u32")], [T(t)]))
        F.append(Form("clz.%s {d0}, {a0}" % t, [T("u32")], [T(t)]))
        F.append(Form("brev.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
    for t in ("s32", "u32", "s64", "u64"):
        F.append(Form("bfind.%s {d0}, {a0}" % t, [T("u32")], [T(t)]))
        F.append(Form("bfind.shiftamt.%s {d0}, {a0}" % t, [T("u32")], [T(t)]))
        F.append(Form("bfe.%s {d0}, {a0}, {a1}, {a2}" % t, [T(t)], [T(t), ("u32", "sh%d" % WIDTH[t]), ("u32", "sh%d" % WIDTH[t])]))
    for t in ("b32", "b64"):
        F.append(Form("bfi.%s {d0}, {a0}, {a1}, {a2}, {a3}" % t, [T(t)], [T(t), T(t), ("u32", "sh%d" % WIDTH[t]),
                                                                          ("u32", "sh%d" % WIDTH[t])]))
    for m in ("", ".wrap", ".clamp"):
        F.append(Form("bmsk%s.b32 {d0}, {a0}, {a1}" % (m or ".wrap") if m else "bmsk.clamp.b32 {d0}, {a0}, {a1}",
                      [T("b32")], [("u32", "sh32"), ("u32", "sh32")]))
    for d in ("l", "r"):
        for m in ("wrap", "clamp"):
            F.append(Form("shf.%s.%s.b32 {d0}, {a0}, {a1}, {a2}" % (d, m), [T("b32")], [T("b32"), T("b32"), ("u32", "sh32")]))
    for m in ("", ".f4e", ".b4e", ".rc8", ".ecl", ".ecr", ".rc16"):
        F.append(Form("prmt.b32%s {d0}, {a0}, {a1}, {a2}" % m, [T("b32")], [T("b32"), T("b32"), T("b32")]))
    for tt in (0x00, 0xff, 0x80, 0xfe, 0x96, 0xe8, 0xca, 0x1e, 0x3c, 0x5a, 0xf0, 0x0f, 0x6a, 0xb4):
        F.append(Form("lop3.b32 {d0}, {a0}, {a1}, {a2}, %d" % tt, [T("b32")], [T("b32"), T("b32"), T("b32")]))
    for a in ("u32", "s32"):
        for b in ("u32", "s32"):
            F.append(Form("dp4a.%s.%s {d0}, {a0}, {a1}, {a2}" % (a, b), [T("s32" if "s" in a + b else "u32")], [T("b32"), T("b32"), T("s32")]))
            for hl in ("lo", "hi"):
                F.append(Form("dp2a.%s.%s.%s {d0}, {a0}, {a1}, {a2}" % (hl, a, b), [T("s32" if "s" in a + b else "u32")],
                              [T("b32"), T("b32"), T("s32")]))
    for t in ("s32", "u32", "s64", "u64"):
        for m in ("", ".clamp"):
            F.append(Form("szext%s.%s {d0}, {a0}, {a1}" % (m or ".wrap", t), [T(t)], [T(t), ("u32", "sh%d" % WIDTH[t])])
                     if t in ("s32", "u32") else None)
    return [f for f in F if f]


def fam_carry(R):
    F = []
    for t in ("u32", "s32", "u64", "s64"):
        F.append(Form("", [T(t), T(t)], [T(t), T(t), T(t), T(t)], name="add.cc/addc.cc/addc.%s" % t,
                      body=lambda f, s, t=t: ["add.cc.%s %s, %s, %s;" % (t, s["d0"], s["a0"], s["a1"]),
                                              "addc.cc.%s %s, %s, %s;" % (t, s["d1"], s["a2"], s["a3"]),
                                              "addc.%s %s, %s, %s;" % (t, s["d0"], s["d0"], s["d1"])]))
        F.append(Form("", [T(t), T(t)], [T(t), T(t), T(t), T(t)], name="sub.cc/subc.cc/subc.%s" % t,
                      body=lambda f, s, t=t: ["sub.cc.%s %s, %s, %s;" % (t, s["d0"], s["a0"], s["a1"]),
                                              "subc.cc.%s %s, %s, %s;" % (t, s["d1"], s["a2"], s["a3"]),
                                              "subc.%s %s, %s, %s;" % (t, s["d0"], s["d0"], s["d1"])]))
        F.append(Form("", [T(t), T(t)], [T(t), T(t), T(t), T(t)], name="mad.lo.cc/madc.hi.%s" % t,
                      body=lambda f, s, t=t: ["mad.lo.cc.%s %s, %s, %s, %s;" % (t, s["d0"], s["a0"], s["a1"], s["a2"]),
                                              "madc.hi.%s %s, %s, %s, %s;" % (t, s["d1"], s["a0"], s["a1"], s["a3"])]))
    return F


RNDS = ("rn", "rz", "rm", "rp")


def fam_float(R):
    F = []
    for t in ("f32", "f64"):
        for r in RNDS:
            for ftz in (("", ".ftz") if t == "f32" else ("",)):
                for op in ("add", "sub", "mul"):
                    F.append(Form("%s.%s%s.%s {d0}, {a0}, {a1}" % (op, r, ftz, t), [T(t)], [T(t), T(t)]))
                F.append(Form("fma.%s%s.%s {d0}, {a0}, {a1}, {a2}" % (r, ftz, t), [T(t)], [T(t), T(t), T(t)]))
                F.append(Form("div.%s%s.%s {d0}, {a0}, {a1}" % (r, ftz, t), [T(t)], [T(t), T(t)]))
                F.append(Form("sqrt.%s%s.%s {d0}, {a0}" % (r, ftz, t), [T(t)], [T(t)]))
                F.append(Form("rcp.%s%s.%s {d0}, {a0}" % (r, ftz, t), [T(t)], [T(t)]))
        if t == "f32":
            for op in ("add", "sub", "mul"):
                F.append(Form("%s.sat.f32 {d0}, {a0}, {a1}" % op, [T(t)], [T(t), T(t)]))
                F.append(Form("%s.ftz.sat.f32 {d0}, {a0}, {a1}" % op, [T(t)], [T(t), T(t)]))
                F.append(Form("%s.f32 {d0}, {a0}, {a1}" % op, [T(t)], [T(t), T(t)]))
            F.append(Form("fma.rn.sat.f32 {d0}, {a0}, {a1}, {a2}", [T(t)], [T(t), T(t), T(t)]))
            F.append(Form("mad.rn.ftz.f32 {d0}, {a0}, {a1}, {a2}", [T(t)], [T(t), T(t), T(t)]))
            for m in ("approx", "full", "approx.ftz", "full.ftz"):
                F.append(Form("div.%s.f32 {d0}, {a0}, {a1}" % m, [T(t)], [T(t), T(t)]))
            for m in ("approx", "approx.ftz"):
                F.append(Form("sqrt.%s.f32 {d0}, {a0}" % m, [T(t)], [T(t)]))
                F.append(Form("rcp.%s.f32 {d0}, {a0}" % m, [T(t)], [T(t)]))
                F.append(Form("rsqrt.%s.f32 {d0}, {a0}" % m, [T(t)], [T(t)]))
                for fn in ("sin", "cos", "lg2", "ex2"):
                    F.append(Form("%s.%s.f32 {d0}, {a0}" % (fn, m), [T(t)], [T(t)]))
            F.append(Form("tanh.approx.f32 {d0}, {a0}", [T(t)], [T(t)]))
        else:
            F.append(Form("rcp.approx.ftz.f64 {d0}, {a0}", [T(t)], [T(t)]))
            F.append(Form("rsqrt.approx.f64 {d0}, {a0}", [T(t)], [T(t)]))
        for ftz in (("", ".ftz") if t == "f32" else ("",)):
            for op in ("min", "max"):
                F.append(Form("%s%s.%s {d0}, {a0}, {a1}" % (op, ftz, t), [T(t)], [T(t), T(t)]))
                if t == "f32":
                    F.append(Form("%s%s.NaN.%s {d0}, {a0}, {a1}" % (op, ftz, t), [T(t)], [T(t), T(t)]))
                    F.append(Form("%s%s.xorsign.abs.%s {d0}, {a0}, {a1}" % (op, ftz, t), [T(t)], [T(t), T(t)]))
            F.append(Form("abs%s.%s {d0}, {a0}" % (ftz, t), [T(t)], [T(t)]))
            F.append(Form("neg%s.%s {d0}, {a0}" % (ftz, t), [T(t)], [T(t)]))
        F.append(Form("copysign.%s {d0}, {a0}, {a1}" % t, [T(t)], [T(t), T(t)]))
        for c in ("finite", "infinite", "number", "notanumber", "normal", "subnormal"):
            F.append(Form("testp.%s.%s {d0}, {a0}" % (c, t), [("pred", "pred")], [T(t)]))
    return F


def fam_half(R):
    F = []
    for t in ("f16", "f16x2", "bf16", "bf16x2"):
        mods = ["", ".rn", ".ftz", ".sat", ".rn.ftz"] if t.startswith("f16") else ["", ".rn"]
        for op in ("add", "sub", "mul"):
            for m in mods:
                F.append(Form("%s%s.%s {d0}, {a0}, {a1}" % (op, m, t), [T(t)], [T(t), T(t)]))
        for m in mods + ([".relu", ".rn.relu"] if True else []):
            F.append(Form("fma%s.%s {d0}, {a0}, {a1}, {a2}" % (m if m.startswith(".rn") else ".rn" + m, t), [T(t)], [T(t), T(t), T(t)]))
        for op in ("min", "max"):
            F.append(Form("%s.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
            F.append(Form("%s.NaN.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
            if t.startswith("f16"):
                F.append(Form("%s.ftz.%s {d0}, {a0}, {a1}" % (op, t), [T(t)], [T(t), T(t)]))
        F.append(Form("neg.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
        F.append(Form("abs.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
        if t.startswith("f16"):
            F.append(Form("ex2.approx.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
            F.append(Form("tanh.approx.%s {d0}, {a0}" % t, [T(t)], [T(t)]))
    for c in ("eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan"):
        F.append(Form("setp.%s.f16 {d0}, {a0}, {a1}" % c, [("pred", "pred")], [T("f16"), T("f16")]))
        F.append(Form("setp.%s.f16x2 {d0}|{d1}, {a0}, {a1}" % c, [("pred", "pred"), ("pred", "pred")], [T("f16x2"), T("f16x2")]))
        F.append(Form("set.%s.u32.f16x2 {d0}, {a0}, {a1}" % c, [T("u32")], [T("f16x2"), T("f16x2")]))
    return F


ICMP = ("eq", "ne", "lt", "le", "gt", "ge")
UCMP = ("lo", "ls", "hi", "hs")
FCMP = ("eq", "ne", "lt", "le", "gt", "ge", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan")


def fam_cmp(R):
    F = []
    for t in ("s16", "u16", "s32", "u32", "s64", "u64", "b16", "b32", "b64"):
        cs = ICMP + (UCMP if t[0] in "ub" else ()) if t[0] != "b" else ("eq", "ne")
        for c in cs:
            F.append(Form("setp.%s.%s {d0}, {a0}, {a1}" % (c, t), [("pred", "pred")], [T(t), T(t)]))
        F.append(Form("selp.%s {d0}, {a0}, {a1}, {a2}" % t, [T(t)], [T(t), T(t), ("pred", "pred")]))
    for t in ("f32", "f64"):
        for c in FCMP:
            for ftz in (("", ".ftz") if t == "f32" else ("",)):
                F.append(Form("setp.%s%s.%s {d0}, {a0}, {a1}" % (c, ftz, t), [("pred", "pred")], [T(t), T(t)]))
        F.append(Form("selp.%s {d0}, {a0}, {a1}, {a2}" % t, [T(t)], [T(t), T(t), ("pred", "pred")]))
    for c in ("eq", "lt", "ge", "lo", "hs"):
        F.append(Form("set.%s.u32.%s {d0}, {a0}, {a1}" % (c, "u32" if c in ("lo", "hs") else "s32"), [T("u32")], [T("s32"), T("s32")]))
    for c in ("eq", "lt", "gtu", "nan"):
        F.append(Form("set.%s.f32.f32 {d0}, {a0}, {a1}" % c, [T("f32")], [T("f32"), T("f32")]))
        F.append(Form("set.%s.ftz.s32.f32 {d0}, {a0}, {a1}" % c, [T("s32")], [T("f32"), T("f32")]))
    for op in ("and", "or", "xor"):
        F.append(Form("setp.lt.and.s32 {d0}, {a0}, {a1}, {a2}".replace("and", op, 1).replace("setp.lt.%s" % op, "setp.lt.%s" % op),
                      [("pred", "pred")], [T("s32"), T("s32"), ("pred", "pred")]))
        F.append(Form("%s.pred {d0}, {a0}, {a1}" % op, [("pred", "pred")], [("pred", "pred"), ("pred", "pred")]))
    F.append(Form("setp.ge.s32 {d0}|{d1}, {a0}, {a1}", [("pred", "pred"), ("pred", "pred")], [T("s32"), T("s32")]))
    F.append(Form("not.pred {d0}, {a0}", [("pred", "pred")], [("pred", "pred")]))
    for t in ("s32", "u32", "f32", "u64", "s64"):
        F.append(Form("slct.%s.s32 {d0}, {a0}, {a1}, {a2}" % t, [T(t)], [T(t), T(t), T("s32")]))
        F.append(Form("slct.%s.f32 {d0}, {a0}, {a1}, {a2}" % t, [T(t)], [T(t), T(t), T("f32")]))
    return F


CVT_T = ("u8", "s8", "u16", "s16", "u32", "s32", "u64", "s64", "f16", "bf16", "f32", "f64")


def fam_cvt(R):
    F = []
    for d in CVT_T:
        for s in CVT_T:
            df, sf = d[0] in "fb", s[0] in "fb"
            mods = [""]
            if df and sf:
                if WIDTH[d] < WIDTH[s] or (d == "bf16" and s == "f16") or (d == "f16" and s == "bf16"):
                    mods = [".rn", ".rz", ".rm", ".rp"]
                elif d == s:
                    mods = [".rni", ".rzi", ".rmi", ".rpi"]
                else:
                    mods = [""]
            elif df and not sf:
                mods = [".rn", ".rz", ".rm", ".rp"]
            elif sf and not df:
                mods = [".rni", ".rzi", ".rmi", ".rpi"]
            for m in mods:
                ftzs = [""]
                if "f32" in (d, s):
                    ftzs = ["", ".ftz"]
                sats = [""]
                if df or (not df and not sf and d != s):
                    sats = ["", ".sat"] if (not df or d in ("f16", "bf16", "f32", "f64")) else [""]
                for ftz in ftzs:
                    for sat in sats:
                        if sat and not df and sf:
                            pass
                        dt = d if WIDTH[d] > 8 else d
                        F.append(Form("cvt%s%s%s.%s.%s {d0}, {a0}" % (m, ftz, sat, d, s), [(d, d)], [(s, s)]))
    for m in ("relu", "satfinite", "relu.satfinite"):
        F.append(Form("cvt.rn.%s.f16x2.f32 {d0}, {a0}, {a1}" % m, [T("f16x2")], [T("f32"), T("f32")]))
        F.append(Form("cvt.rn.%s.bf16x2.f32 {d0}, {a0}, {a1}" % m, [T("bf16x2")], [T("f32"), T("f32")]))
    F.append(Form("cvt.rn.f16x2.f32 {d0}, {a0}, {a1}", [T("f16x2")], [T("f32"), T("f32")]))
    F.append(Form("cvt.rz.bf16x2.f32 {d0}, {a0}, {a1}", [T("bf16x2")], [T("f32"), T("f32")]))
    F.append(Form("cvt.rna.tf32.f32 {d0}, {a0}", [T("b32")], [T("f32")]))
    for d in ("e4m3x2", "e5m2x2"):
        F.append(Form("cvt.rn.satfinite.%s.f32 {d0}, {a0}, {a1}" % d, [T("b16")], [T("f32"), T("f32")]))
        F.append(Form("cvt.rn.f16x2.%s {d0}, {a0}" % d, [T("f16x2")], [T("b16")]))
    for t in ("u32", "s32", "u64", "s64"):
        F.append(Form("cvta.to.global.u64 {d0}, {a0}", [T("u64")], [T("u64")]) if t == "u64" else None)
    return [f for f in F if f]


def fam_mem(R):
    """ld / st through .global, .shared, .local: vector widths and types; stores go to a scratch slot area and are
    read back"""
    F = []

    def body_space(space, t, vec):
        n = {"": 1, ".v2": 2, ".v4": 4}[vec]
        w = regw(t)

        def b(f, s):
            out = []
            regs = ["%%m%d" % i for i in range(n)]
            out.append(".reg .b%d %s;" % (w, ", ".join(regs)))
            for i, r in enumerate(regs):
                out.append("add.u%d %s, %s, %d;" % (w, r, s["a0"], 3 * i + 1) if w > 1 else "")
            vec_regs = "{%s}" % ", ".join(regs) if n > 1 else regs[0]
            if space == "global":
                out.append("ld.param.u64 %rd4, [po];")
                out.append("mul.wide.u32 %rd5, %r0, 32;")
                out.append("add.s64 %rd4, %rd4, %rd5;")
                out.append("add.s64 %rd4, %rd4, 1024;")
                addr = "[%rd4]"
            elif space == "shared":
                out.append(".shared .align 16 .b8 sbuf_%s[4096];" % id(f))
                out.append("mov.u64 %rd4, sbuf_{};".format(id(f)))
                out.append("mul.wide.u32 %rd5, %r0, 32;")
                out.append("add.s64 %rd4, %rd4, %rd5;")
                addr = "[%rd4]"
            else:
                out.append(".local .align 16 .b8 lbuf_%s[32];" % id(f))
                out.append("mov.u64 %rd4, lbuf_{};".format(id(f)))
                addr = "[%rd4]"
            out.append("st.%s%s.%s %s, %s;" % (space, vec, "b%d" % w if WIDTH[t] > 8 else t, addr, vec_regs))
            if space == "shared":
                out.append("bar.sync 0;")
            out.append("ld.%s%s.%s %s, %s;" % (space, vec, "b%d" % w if WIDTH[t] > 8 else t, vec_regs, addr))
            out.append("mov.b%d %s, 0;" % (w, s["d0"]))
            for r in regs:
                out.append("xor.b%d %s, %s, %s;" % (w, s["d0"], s["d0"], r))
                out.append("shl.b%d %s, %s, 1;" % (w, s["d0"], s["d0"]))
            return [x for x in out if x]
        return b
    for space in ("global", "shared", "local"):
        for t in ("u8", "b16", "b32", "b64"):
            for vec in ("", ".v2", ".v4"):
                if t == "b64" and vec == ".v4":
                    continue
                F.append(Form("", [("b%d" % regw(t), "x")], [("b%d" % regw(t), t)], name="ld/st.%s%s.%s" % (space, vec, t),
                              body=body_space(space, t, vec)))
    # cache operators and ld.global.nc
    for op in ("ld.global.ca", "ld.global.cg", "ld.global.cs", "ld.global.lu", "ld.global.cv", "ld.global.nc", "ldu.global"):
        F.append(Form("", [T("b32")], [T("u32")], name=op + ".b32",
                      body=lambda f, s, op=op: ["%s.b32 %s, [%%rd1];" % (op, s["d0"])]))
    for op in ("st.global.wb", "st.global.cg", "st.global.cs", "st.global.wt"):
        F.append(Form("", [T("b32")], [T("u32")], name=op + ".b32",
                      body=lambda f, s, op=op: ["ld.param.u64 %rd4, [po];", "mul.wide.u32 %rd5, %r0, 8;", "add.s64 %rd4, %rd4, %rd5;",
                                                "add.s64 %rd4, %rd4, 2048;", "%s.b32 [%%rd4], %s;" % (op, s["a0"]),
                                                "ld.global.b32 %s, [%%rd4];" % s["d0"]]))
    return F


ATOM_OPS = {"b32": ("and", "or", "xor", "exch", "cas"), "u32": ("add", "min", "max", "inc", "dec"), "s32": ("add", "min", "max"),
            "u64": ("add", "min", "max"), "s64": ("min", "max"), "b64": ("and", "or", "xor", "exch", "cas"), "f32": ("add",),
            "f64": ("add",)}


def fam_atom(R):
    F = []
    for space in ("global", "shared"):
        for t, ops in ATOM_OPS.items():
            w = WIDTH[t]
            for op in ops:
                # integer add / and / or / xor / min / max: threads t update word t % 4 together (atom, then red), and
                # every thread reads the final word, which does not depend on the order; the other operations (exch,
                # cas, inc, dec, float add) work on a word of their own and return the old value and the new word
                shared_word = op in ("add", "and", "or", "xor", "min", "max") and t[0] != "f"

                def b(f, s, space=space, t=t, op=op, w=w, sw=shared_word):
                    out = []
                    if space == "global":
                        out += ["ld.param.u64 %rd4, [po];", "add.s64 %rd4, %rd4, 4096;"]
                    else:
                        out += [".shared .align 8 .b8 abuf_%d[512];" % id(f), "mov.u64 %%rd4, abuf_%d;" % id(f)]
                    out += ["and.b32 %r5, %r0, 3;" if sw else "mov.u32 %r5, %r0;", "mul.wide.u32 %rd6, %r5, 8;",
                            "add.s64 %rd4, %rd4, %rd6;"]
                    init = "7" if op in ("inc", "dec") else ("0" if t[0] != "f" else ("0f3f800000" if w == 32 else "0d3ff0000000000000"))
                    if sw:
                        out += ["setp.lt.u32 %p6, %r0, 4;", "@%p6 st.{}.b{} [%rd4], {};".format(space, w, init), "bar.sync 0;"]
                    else:
                        out += ["st.{}.b{} [%rd4], {};".format(space, w, init)]
                    if op == "cas":
                        out.append("atom.%s.cas.b%d %s, [%%rd4], 0, %s;" % (space, w, s["d1"], s["a0"]))
                    else:
                        out.append("atom.%s.%s.%s %s, [%%rd4], %s;" % (space, op, t, s["d1"], s["a0"]))
                    if sw:
                        out.append("red.%s.%s.%s [%%rd4], %s;" % (space, op, t, s["a0"]))
                        out.append("mov.b%d %s, 0;" % (w, s["d1"]))
                    out.append("bar.sync 0;")
                    out.append("ld.%s.b%d %s, [%%rd4];" % (space, w, s["d0"]))
                    return out
                F.append(Form("", [("b%d" % w, "x"), ("b%d" % w, "x")], [(t, "small" if op in ("inc", "dec") else t)],
                              name="atom/red.%s.%s.%s" % (space, op, t), body=b))
    return F


def fam_warp(R):
    F = []
    for m in ("idx", "up", "down", "bfly"):
        F.append(Form("shfl.sync.%s.b32 {d0}|{d1}, {a0}, {a1}, 31, 0xffffffff" % m if m != "up" else
                      "shfl.sync.up.b32 {d0}|{d1}, {a0}, {a1}, 0, 0xffffffff", [T("b32"), ("pred", "pred")], [T("b32"), ("u32", "sh32")]))
        F.append(Form("shfl.sync.%s.b32 {d0}, {a0}, {a1}, 0x1f, 0xffffffff" % m if m != "up" else
                      "shfl.sync.up.b32 {d0}, {a0}, {a1}, 0, 0xffffffff", [T("b32")], [T("b32"), ("u32", "small")]))
        F.append(Form("shfl.sync.%s.b32 {d0}, {a0}, 1, 0x0c1f, 0xffffffff" % m if m != "up" else
                      "shfl.sync.up.b32 {d0}, {a0}, 1, 0x0c00, 0xffffffff", [T("b32")], [T("b32")]))
    for m in ("all", "any", "uni"):
        F.append(Form("vote.sync.%s.pred {d0}, {a0}, 0xffffffff" % m, [("pred", "pred")], [("pred", "pred")]))
    F.append(Form("vote.sync.ballot.b32 {d0}, {a0}, 0xffffffff", [T("b32")], [("pred", "pred")]))
    for t in ("b32", "b64"):
        F.append(Form("match.any.sync.%s {d0}, {a0}, 0xffffffff" % t, [T("b32")], [(t, "small")]))
        F.append(Form("match.all.sync.%s {d0}|{d1}, {a0}, 0xffffffff" % t, [T("b32"), ("pred", "pred")], [(t, "small")]))
    for op, t in (("add", "u32"), ("add", "s32"), ("min", "u32"), ("min", "s32"), ("max", "u32"), ("max", "s32"),
                  ("and", "b32"), ("or", "b32"), ("xor", "b32")):
        F.append(Form("redux.sync.%s.%s {d0}, {a0}, 0xffffffff" % (op, t), [T(t)], [T(t)]))
    F.append(Form("activemask.b32 {d0}", [T("b32")], []))
    F.append(Form("", [T("b32")], [T("b32")], name="bar.warp.sync",
                  body=lambda f, s: ["bar.warp.sync 0xffffffff;", "mov.b32 %s, %s;" % (s["d0"], s["a0"])]))
    F.append(Form("", [T("u32")], [T("u32")], name="mov.u32 %laneid/%warpid/%lanemask_lt",
                  body=lambda f, s: ["mov.u32 %r6, %laneid;", "mov.u32 %r7, %lanemask_lt;", "add.u32 %s, %%r6, %%r7;" % s["d0"]]))
    return F


def fam_video(R):
    F = []
    for op in ("vadd", "vsub", "vabsdiff", "vmin", "vmax"):
        for sat in ("", ".sat"):
            F.append(Form("%s.s32.s32.s32%s {d0}, {a0}, {a1}" % (op, sat), [T("s32")], [T("s32"), T("s32")]))
            F.append(Form("%s.u32.u32.s32%s {d0}, {a0}, {a1}" % (op, sat), [T("u32")], [T("u32"), T("s32")]))
        F.append(Form("%s.s32.s32.s32.add {d0}, {a0}, {a1}, {a2}" % op, [T("s32")], [T("s32"), T("s32"), T("s32")]))
        F.append(Form("%s.u32.u32.u32 {d0}, {a0}.b1, {a1}.h0" % op, [T("u32")], [T("u32"), T("u32")]))
    for op in ("vshl", "vshr"):
        F.append(Form("%s.u32.u32.u32.clamp {d0}, {a0}, {a1}" % op, [T("u32")], [T("u32"), ("u32", "sh32")]))
        F.append(Form("%s.s32.s32.u32.wrap {d0}, {a0}, {a1}" % op, [T("s32")], [T("s32"), ("u32", "sh32")]))
    F.append(Form("vmad.s32.s32.s32 {d0}, {a0}, {a1}, {a2}", [T("s32")], [T("s32"), T("s32"), T("s32")]))
    F.append(Form("vmad.u32.u32.u32.shr7 {d0}, {a0}, {a1}, {a2}", [T("u32")], [T("u32"), T("u32"), T("u32")]))
    for c in ("eq", "lt", "ge"):
        F.append(Form("vset.s32.s32.%s {d0}, {a0}, {a1}" % c, [T("u32")], [T("s32"), T("s32")]))
    for op in ("vadd2", "vsub2", "vavrg2", "vabsdiff2", "vmin2", "vmax2", "vadd4", "vsub4", "vavrg4", "vabsdiff4", "vmin4", "vmax4"):
        F.append(Form("%s.u32.u32.u32 {d0}, {a0}, {a1}, {a2}" % op, [T("u32")], [T("u32"), T("u32"), T("u32")]))
        F.append(Form("%s.s32.s32.s32.sat {d0}, {a0}, {a1}, {a2}" % op, [T("s32")], [T("s32"), T("s32"), T("s32")]))
    return F


FAMILIES = {"int": fam_int, "carry": fam_carry, "float": fam_float, "half": fam_half, "cmp": fam_cmp, "cvt": fam_cvt,
            "mem": fam_mem, "atom": fam_atom, "warp": fam_warp, "video": fam_video}

PROLOG = ".version 8.0\n.target sm_90\n.address_size 64\n\n"


def module(forms, R, nthreads, chunk_name):
    """PTX text with .print directives, and the per-kernel launch data"""
    parts, harness = [PROLOG], []
    for i, f in enumerate(forms):
        parts.append(kernel(i, f, nthreads))
        nin = max(len(f.srcs), 1)
        vals = [values(c, R, nthreads) for _, c in f.srcs] or [[0] * nthreads]
        flat = []
        for t in range(nthreads):
            for k in range(nin):
                flat.append(vals[k][t] if k < len(vals) else 0)
        nout = max(len(f.dsts), 1)
        harness.append(".buffer I%d .b64 = { %s };" % (i, ", ".join(str(v) for v in flat)))
        harness.append(".buffer O%d .b64 [%d];" % (i, nthreads * nout + 512))
        harness.append(".launch k%d <<<1, %d>>>(I%d, O%d);" % (i, nthreads, i, i))
        harness.append(".print O%d .b64 %d;" % (i, nthreads * nout))
    return "".join(parts) + "\n" + "\n".join(harness) + "\n"


def run_model(text):
    import ptxtest
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(text)
        name = f.name
    try:
        return ptxtest.run_one(name)
    finally:
        os.unlink(name)


def with_expects(text, out):
    """replace each .print O<i> by an .expect with the printed values; kernels the model rejects lose their checks"""
    vals = {}
    for line in out.splitlines():
        m = re.match(r"(O\d+) = \[(.*)\]$", line)
        if m:
            vals[m.group(1)] = m.group(2)
    lines = []
    for l in text.splitlines():
        m = re.match(r"\.print (O\d+) \.b64 (\d+);", l)
        if m:
            if m.group(1) in vals:
                lines.append(".expect %s .b64 = { %s };" % (m.group(1), vals[m.group(1)]))
            continue
        lines.append(l)
    return "\n".join(lines) + "\n", len(vals)


def split_unsupported(forms, R, nthreads):
    """forms the Sail model accepts (runs to the end): a module per form would be slow, so run the whole chunk and
    bisect on failure"""
    text = module(forms, random.Random(R.random()), nthreads, "")
    st, note, out, _ = run_model(text)
    if st == "PASS":
        return forms, []
    if len(forms) == 1:
        return [], [(forms[0], note or out.strip()[-200:])]
    h = len(forms) // 2
    a, ra = split_unsupported(forms[:h], R, nthreads)
    b, rb = split_unsupported(forms[h:], R, nthreads)
    return a + b, ra + rb


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("-s", "--seed", type=int, default=1)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=24)
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    total = rejected = 0
    rej_lines = []
    for fam in a.families.split(","):
        R = random.Random("%s-%d" % (fam, a.seed))
        forms = FAMILIES[fam](R)
        ok, rej = split_unsupported(forms, R, a.threads) if forms else ([], [])
        for f, why in rej:
            rej_lines.append("%s\t%s\t%s" % (fam, f.name, " ".join(why.split())[:200]))
        rejected += len(rej)
        for c in range(0, len(ok), a.chunk):
            part = ok[c:c + a.chunk]
            text = module(part, R, a.threads, "%s-%d" % (fam, c // a.chunk))
            st, note, out, _ = run_model(text)
            text, n = with_expects(text, out)
            path = os.path.join(a.outdir, "gen-%s-%d.ptx" % (fam, c // a.chunk))
            with open(path, "w") as f:
                f.write("// GENERATED by sail/ptx/gen/ptxgen.py (seed %d): .expect values from the PTX Sail model\n" % a.seed)
                f.write(text)
            total += n
        print("%-6s %4d forms, %4d run on the model, %3d rejected by the model" % (fam, len(forms), len(ok), len(rej)))
    with open(os.path.join(a.outdir, "model-rejected.txt"), "w") as f:
        f.write("\n".join(rej_lines) + "\n")
    print("---- %d kernels with .expect values, %d forms rejected by the Sail model (%s)" % (
        total, rejected, os.path.join(a.outdir, "model-rejected.txt")))


if __name__ == "__main__":
    main()

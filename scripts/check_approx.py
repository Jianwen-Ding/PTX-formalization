#!/usr/bin/env python3
"""Check the approximate floating-point instructions against the error bounds the PTX ISA states for them.

Each op is evaluated by the K semantics (krun) on sample inputs and compared with the exact value computed in
double precision (math module); the semantics must stay within the documented error bound, otherwise it would
not be a refinement of the hardware behaviour.  Bounds (ISA ch. 9): rcp/sqrt.approx 2^-23 relative, rsqrt.approx
2^-22.9 relative, sin/cos.approx 2^-20.9 absolute on [-100pi, 100pi], ex2.approx 2^-22.5 relative,
lg2.approx 2^-22.6 absolute on [0.5, 2] and 2^-22.3 relative elsewhere (we use the looser bound), tanh.approx 2^-10.987 relative.
"""
import math, os, struct, subprocess, sys, tempfile

DEF = os.environ.get("PTX_DEF", "build/ptx-kompiled")
f2b = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
b2f = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]

OPS = {   # name: (exact, kind, bound, inputs)
    "rcp.approx.f32":   (lambda x: 1.0 / x,        "rel", 2**-23,    [0.1, 0.75, 1.0, 3.0, 7.5, 123.456, 1e-5, 4e5]),
    "sqrt.approx.f32":  (math.sqrt,                "rel", 2**-23,    [0.1, 0.75, 1.0, 2.0, 7.5, 123.456, 1e-5, 4e5]),
    "rsqrt.approx.f32": (lambda x: x ** -0.5,      "rel", 2**-22.9,  [0.1, 0.75, 1.0, 2.0, 7.5, 123.456, 1e-5, 4e5]),
    "sin.approx.f32":   (math.sin,                 "abs", 2**-20.9,  [0.1, 0.75, 1.0, 3.0, -7.5, 123.456, -300.0, 0.0]),
    "cos.approx.f32":   (math.cos,                 "abs", 2**-20.9,  [0.1, 0.75, 1.0, 3.0, -7.5, 123.456, -300.0, 0.0]),
    "ex2.approx.f32":   (lambda x: 2.0 ** x,       "rel", 2**-22.5,  [0.1, 0.75, 1.0, -3.0, 7.5, -20.25, 100.0, 0.0]),
    "lg2.approx.f32":   (math.log2,                "relabs", 2**-22.3, [0.1, 0.75, 1.0, 3.0, 7.5, 123.456, 1e-5, 4e5]),
    "tanh.approx.f32":  (math.tanh,                "rel", 2**-10.987, [0.1, 0.75, 1.0, 3.0, -7.5, 0.01, -0.3, 9.0]),
}

def run(op, xs):
    body = "".join(f"    mov.b32 %x, 0x{f2b(x):08x};\n    {op} %y, %x;\n    st.global.b32 [%o+{4*i}], %y;\n" for i, x in enumerate(xs))
    src = (".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry k(.param .u64 po)\n{\n"
           "    .reg .b64 %o; .reg .f32 %x, %y;\n    ld.param.u64 %o, [po];\n" + body + "    ret;\n}\n"
           f".buffer OUT .u32 [{len(xs)}];\n.launch k <<< 1, 1 >>> (OUT);\n.print OUT .u32 {len(xs)} ;\n")
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(src)
    try:
        out = subprocess.run(["krun", "-d", DEF, f.name, "--output", "pretty"], capture_output=True, text=True,
                             env={**os.environ, "K_OPTS": "-Xss512m"}).stdout
    finally:
        os.unlink(f.name)
    line = [l for l in out.splitlines() if l.startswith("OUT = [")][0]
    return [int(v) for v in line[len("OUT = ["):-1].split(",")]

bad = 0
for op, (ex, kind, bound, xs) in OPS.items():
    got = run(op, xs)
    worst = 0.0
    for x, g in zip(xs, got):
        x = b2f(f2b(x))          # the f32 value actually fed to the instruction
        want, have = ex(x), b2f(g)
        abs_err = abs(have - want)
        rel_err = abs_err / abs(want) if want != 0 else abs_err
        err = abs_err if kind == "abs" else rel_err
        if kind == "relabs":
            err = abs_err if 0.5 <= x <= 2.0 else rel_err
        worst = max(worst, err / bound)
    ok = worst <= 1.0
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {op:20s} worst error = {worst:.3f} x bound")
sys.exit(1 if bad else 0)

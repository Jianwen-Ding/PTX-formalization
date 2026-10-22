"""Build a PTX test program for a group of instruction variants and run it on the Sail model.

One thread per case.  Every input operand lives in its own .u64 buffer (one 8-byte slot per case,
read with ld.global.b8/b16/b32/b64 of the operand width); every variant stores its result into its
own 8-byte slot of OUT (variant-major), which is dumped with `.print OUT .u64`.
"""
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = "/home/user/PTX-formalization/sail/ptx"


def model_paths():
    root = os.environ.get("PTX_MODEL_ROOT", DEFAULT_ROOT)
    return (os.environ.get("PTX_FRONT", os.path.join(root, "ptxfront.py")),
            os.environ.get("PTX_MODEL", os.path.join(root, "build", "ptx")))


def program(target, in_widths, variants, n, inputs, tpb=128):
    """variants: list of (template, out_width).  template uses {d} and {0},{1},... for operands."""
    nin = len(in_widths)
    params = ", ".join([".param .u64 pi%d" % j for j in range(nin)] + [".param .u64 po"])
    L = []
    L.append(".version 9.4\n.target %s\n.address_size 64" % target)
    L.append(".visible .entry k(%s)\n{" % params)
    L.append(" .reg .b64 %rp<16>, %ra, %rt; .reg .b32 %tid, %cta, %nt;")
    for j, w in enumerate(in_widths):
        L.append(" .reg .b%d %%i%d;" % (w, j))
    L.append(" .reg .b8 %d8; .reg .b16 %d16; .reg .b32 %d32; .reg .b64 %d64;")
    L.append(" mov.u32 %tid, %tid.x; mov.u32 %cta, %ctaid.x; mov.u32 %nt, %ntid.x; mad.lo.u32 %tid, %cta, %nt, %tid;")
    L.append(" mul.wide.u32 %rt, %tid, 8;")
    for j, w in enumerate(in_widths):
        L.append(" ld.param.u64 %%rp%d, [pi%d]; add.u64 %%ra, %%rp%d, %%rt; ld.global.b%d %%i%d, [%%ra];" % (j, j, j, w, j))
    L.append(" ld.param.u64 %rp15, [po]; add.u64 %rp15, %rp15, %rt;")
    ops = ["%%i%d" % j for j in range(nin)]
    for v, (tmpl, ow) in enumerate(variants):
        d = "%%d%d" % ow
        L.append(" " + tmpl.format(*ops, d=d))
        L.append(" st.global.b%d [%%rp15+%d], %s;" % (ow, v * n * 8, d))
    L.append(" ret;\n}")
    for j in range(nin):
        L.append(".buffer IN%d .u64 = { %s };" % (j, ", ".join(str(x[j]) for x in inputs)))
    L.append(".buffer OUT .u64 [%d];" % (len(variants) * n))
    assert n % tpb == 0 or n < tpb
    blocks, t = (n // tpb, tpb) if n >= tpb else (1, n)
    L.append(".launch k <<< %d, %d >>> (%s);" % (blocks, t, ", ".join(["IN%d" % j for j in range(nin)] + ["OUT"])))
    L.append(".print OUT .u64;")
    return "\n".join(L) + "\n"


class ModelError(Exception):
    pass


def run_program(text, keep=None):
    front, model = model_paths()
    with tempfile.TemporaryDirectory(prefix="fporacle-") as td:
        p = os.path.join(td, "t.ptx")
        with open(p, "w") as f:
            f.write(text)
        if keep:
            with open(keep, "w") as f:
                f.write(text)
        r = subprocess.run(["python3", front, p], capture_output=True, text=True)
        if r.returncode:
            raise ModelError("front end: " + r.stderr[-2000:])
        tok = os.path.join(td, "t.tok")
        with open(tok, "w") as f:
            f.write(r.stdout)
        env = dict(os.environ, SAIL_INPUT=tok)
        r = subprocess.run([model], env=env, capture_output=True, text=True)
    out = r.stdout
    m = re.search(r"^OUT = \[(.*?)\]", out, re.M | re.S)
    if not m or "EXIT 0" not in out + r.stderr:
        raise ModelError("model: " + (out + r.stderr)[-3000:])
    return [int(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def run(target, in_widths, variants, inputs, keep=None, tpb=128):
    n0 = len(inputs)
    if n0 > tpb and n0 % tpb:                       # pad to whole blocks with copies of the last case
        inputs = list(inputs) + [inputs[-1]] * (tpb - n0 % tpb)
    n = len(inputs)
    vals = run_program(program(target, in_widths, variants, n, inputs, tpb), keep=keep)
    assert len(vals) == n * len(variants), (len(vals), n, len(variants))
    return [[vals[v * n + i] for i in range(n0)] for v in range(len(variants))]

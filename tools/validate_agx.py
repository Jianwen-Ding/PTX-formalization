#!/usr/bin/env python3
"""Translation validation of the PTX -> AGX lowering.

For each test kernel: run the PTX on the K semantics (one warp), lower it with ptx2agx.py, run the AGX program on the AGX K semantics
with the same input memory, and require the AGX memory to equal the PTX result.  usage: validate_agx.py [-v] [name ...]
"""
import os, random, re, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PTX_DEF = os.environ.get("PTX_DEF", os.path.join(ROOT, "build/ptx-kompiled"))
AGX_DEF = os.environ.get("AGX_DEF", os.path.join(ROOT, "build/agx-kompiled"))
ENV = {**os.environ, "K_OPTS": "-Xss512m"}
f2b = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
b2f = lambda b: struct.unpack("<f", struct.pack("<I", b))[0]

HDR = ".version 8.0\n.target sm_90\n.address_size 64\n"

def rnd_f(n, lo=-100.0, hi=100.0, rng=random.Random(7)):
    return [b2f(f2b(rng.uniform(lo, hi))) for _ in range(n)]
def rnd_i(n, lo=-1000, hi=1000, rng=random.Random(11)):
    return [rng.randint(lo, hi) & 0xFFFFFFFF for _ in range(n)]

# name: (ptx body, [(param name, kind, init values or None, is_output)], scalar params)
TESTS = {}
def test(name, ptx, bufs, scalars=()):
    TESTS[name] = (ptx, bufs, scalars)

test("vadd_u32", """
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc)
{ .reg .b32 %r<4>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc];
  mov.u32 %r1, %tid.x; mul.wide.u32 %rd4, %r1, 4;
  add.u64 %rd5, %rd1, %rd4; add.u64 %rd6, %rd2, %rd4; add.u64 %rd7, %rd3, %rd4;
  ld.global.u32 %r2, [%rd5]; ld.global.u32 %r3, [%rd6];
  add.u32 %r2, %r2, %r3; st.global.u32 [%rd7], %r2; ret; }""",
     [("a", "u32", rnd_i(32), False), ("b", "u32", rnd_i(32), False), ("c", "u32", [0] * 32, True)])

test("saxpy_masked", """
.visible .entry k(.param .u64 px, .param .u64 py)
{ .reg .b32 %r<4>; .reg .f32 %f<6>; .reg .b64 %rd<8>; .reg .pred %p<2>;
  ld.param.u64 %rd1, [px]; ld.param.u64 %rd2, [py];
  mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4;
  add.u64 %rd4, %rd1, %rd3; add.u64 %rd5, %rd2, %rd3;
  ld.global.f32 %f1, [%rd4]; ld.global.f32 %f2, [%rd5];
  fma.rn.f32 %f3, %f1, 0f40000000, %f2;
  setp.lt.u32 %p1, %r1, 20;
  @%p1 st.global.f32 [%rd5], %f3; ret; }""",
     [("x", "f32", rnd_f(32), False), ("y", "f32", rnd_f(32), True)])

test("clamp_branch", """
.visible .entry k(.param .u64 pin, .param .u64 pout)
{ .reg .b32 %r<6>; .reg .b64 %rd<8>; .reg .pred %p<3>;
  ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
  mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4;
  add.u64 %rd4, %rd1, %rd3; add.u64 %rd5, %rd2, %rd3;
  ld.global.s32 %r2, [%rd4];
  mov.u32 %r3, 0;
  setp.lt.s32 %p1, %r2, 0;
  @%p1 bra $L1;
  mov.u32 %r3, %r2;
$L1:
  setp.gt.s32 %p2, %r3, 500;
  @%p2 mov.u32 %r3, 500;
  st.global.u32 [%rd5], %r3; ret; }""",
     [("in", "s32", rnd_i(32), False), ("out", "u32", [0] * 32, True)])

test("bitmix", """
.visible .entry k(.param .u64 pin, .param .u64 pout)
{ .reg .b32 %r<8>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
  mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4;
  add.u64 %rd4, %rd1, %rd3; add.u64 %rd5, %rd2, %rd3;
  ld.global.u32 %r2, [%rd4];
  shl.b32 %r3, %r2, 3; shr.u32 %r4, %r2, 5; xor.b32 %r5, %r3, %r4;
  and.b32 %r6, %r5, 0x00ff00ff; or.b32 %r7, %r6, %r1; not.b32 %r7, %r7;
  st.global.u32 [%rd5], %r7; ret; }""",
     [("in", "u32", rnd_i(32, 0, 2**32 - 1), False), ("out", "u32", [0] * 32, True)])

test("minmax_sel", """
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc)
{ .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .pred %p<2>;
  ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc];
  mov.u32 %r1, %tid.x; mul.wide.u32 %rd4, %r1, 4;
  add.u64 %rd5, %rd1, %rd4; add.u64 %rd6, %rd2, %rd4; add.u64 %rd7, %rd3, %rd4;
  ld.global.s32 %r2, [%rd5]; ld.global.s32 %r3, [%rd6];
  min.s32 %r4, %r2, %r3; max.u32 %r5, %r2, %r3;
  setp.ge.s32 %p1, %r2, %r3; selp.b32 %r6, %r4, %r5, %p1;
  mad.lo.s32 %r6, %r6, 3, %r1;
  st.global.u32 [%rd7], %r6; ret; }""",
     [("a", "s32", rnd_i(32), False), ("b", "s32", rnd_i(32), False), ("c", "u32", [0] * 32, True)])

test("float_denormal", """
.visible .entry k(.param .u64 pin, .param .u64 pout)
{ .reg .b32 %r<4>; .reg .f32 %f<4>; .reg .b64 %rd<8>;
  ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
  mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4;
  add.u64 %rd4, %rd1, %rd3; add.u64 %rd5, %rd2, %rd3;
  ld.global.f32 %f1, [%rd4];
  add.f32 %f2, %f1, 0f00000000;
  st.global.f32 [%rd5], %f2; ret; }""",
     [("in", "f32", [0.0] * 30 + [b2f(1), b2f(0x00400000)], False), ("out", "f32", [0.0] * 32, True)])

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV)

def validate(name, verbose=False):
    ptx, bufs, scalars = TESTS[name]
    tmp = tempfile.mkdtemp()
    # --- PTX on the K semantics
    decls = []
    for pname, ty, vals, is_out in bufs:
        vs = ", ".join(repr(v) if ty == "f32" else str(v) for v in vals)
        decls.append(f".buffer {pname} .{ty if ty != 's32' else 'u32'} [{len(vals)}] = {{ {vs} }};")
    kname = re.search(r"\.entry\s+(\w+)", ptx).group(1)
    prog = HDR + ptx + "\n" + "\n".join(decls) + f"\n.launch {kname} <<< 1, 32 >>> ({', '.join(b[0] for b in bufs)});\n"
    prog += "".join(f".print {b[0]} .u32 {len(b[2])} ;\n" for b in bufs)
    pf = os.path.join(tmp, "t.ptx"); open(pf, "w").write(prog)
    out = run(["krun", "-d", PTX_DEF, pf, "--output", "pretty"]).stdout
    res = {}
    for l in out.splitlines():
        m = re.match(r"^(\w+) = \[(.*)\]$", l)
        if m: res[m.group(1)] = [int(x) for x in m.group(2).split(",")]
    if not res: return False, "PTX run produced no output"
    # --- lower
    low = run(["python3", os.path.join(HERE, "ptx2agx.py"), pf])
    if low.returncode: return False, low.stderr.strip()
    body = low.stdout.strip()
    # --- AGX program with the same memory
    lines = [".srall 81 = 32;", ".sr 80 = laneid;", ".srall 82 = 0;"]
    base = {}
    for i, (pname, ty, vals, is_out) in enumerate(bufs):
        base[pname] = 4096 * (i + 1)
        lines += [f".uni u{2 * i} = {base[pname]};", f".uni u{2 * i + 1} = 0;"]
        for j, v in enumerate(vals):
            w = f2b(v) if ty == "f32" else v & 0xFFFFFFFF
            lines.append(f".dev {base[pname] + 4 * j} = {w};")
    lines.append(body + ";")
    lines.append(".run;")
    for pname, ty, vals, is_out in bufs:
        for j, w in enumerate(res[pname]):
            lines.append(f".expectdev {base[pname] + 4 * j} = {w};")
    af = os.path.join(tmp, "t.agx"); open(af, "w").write("\n".join(lines).rstrip(";") + "\n")
    r = run(["krun", "-d", AGX_DEF, af, "--output", "pretty"])
    msg = re.search(r"<msg>\s*\"([^\"]*)\"", r.stdout)
    ok = r.returncode == 0 and (not msg or not msg.group(1))
    if verbose: print(body)
    return ok, "" if ok else (msg.group(1) if msg else r.stderr[-300:])

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "-v"]
    bad = 0
    for name in args or TESTS:
        ok, why = validate(name, "-v" in sys.argv)
        print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + why))
        bad += not ok
    sys.exit(1 if bad else 0)

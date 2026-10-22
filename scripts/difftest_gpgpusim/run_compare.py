#!/usr/bin/env python3
"""Differential testing of the PTX Sail model against GPGPU-Sim's functional
simulator.

    run_compare.py [options] test.ptx ...

Each argument is a PTX program with the harness directives of tests/*.ptx
(.buffer / .launch / .expect / .print).  For every program:

  1. the harness lines are stripped, giving a plain PTX module for GPGPU-Sim;
  2. the Sail model is run on the buffers alone (no launch) to obtain the
     initial byte image of every .buffer, so both simulators start from the
     same memory;
  3. GPGPU-Sim (gpgpusim-ptx, built by build.sh) loads the module, allocates
     the buffers, performs the .launch lines in order and dumps the buffers;
  4. the Sail model runs the program with its .expect/.print lines replaced by
     a byte dump of every buffer;
  5. the final buffer images are compared byte by byte.

Statuses: MATCH, DIFF, SKIP (harness feature the driver does not model),
GS-PARSE (GPGPU-Sim rejects the module), GS-FAIL (GPGPU-Sim aborted, crashed
or timed out while executing), SAIL-FAIL (the Sail model rejected or trapped).
Only MATCH and DIFF are comparisons; see README.md.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAIL_DIR = os.environ.get("PTX_SAIL_DIR") or os.path.normpath(
    os.path.join(HERE, "..", "..", "sail", "ptx"))
DEFAULT_GS = os.environ.get("GPGPUSIM_PTX") or os.path.join(
    os.getcwd(), "_gpgpusim_build", "gpgpusim-ptx")

# Harness directives (ptxfront.HARNESS); the ones in SUPPORTED are modelled.
HARNESS_KW = ("buffer", "launch", "cooplaunch", "print", "expect", "patch",
              "tensormap", "texture", "sampler", "surface", "multimem",
              "fabric", "memmodel", "ub", "occupancy", "convergence",
              "isacheck", "append")
SUPPORTED = {"buffer", "launch", "print", "expect", "isacheck"}
HARNESS_RE = re.compile(
    r"(?m)^[ \t]*\.(%s)\b[^;]*;[ \t]*" % "|".join(HARNESS_KW))

FRONT = None

# Names GPGPU-Sim's lexer always reads as special registers (src/cuda-sim/ptx.l);
# a program that declares a register with one of these names is renamed.
GS_SPECIAL_REGS = ("tid", "ntid", "ctaid", "nctaid", "laneid", "warpid",
                   "nwarpid", "smid", "nsmid", "gridid", "clock", "clock64",
                   "halfclock", "lanemask_eq", "lanemask_le", "lanemask_lt",
                   "lanemask_ge", "lanemask_gt")


def gs_rewrite(module, mod):
    """Semantics-preserving rewrites that only work around GPGPU-Sim's older
    PTX grammar: every .entry gets a prefixed name (kernels called e.g. `add`
    clash with opcode tokens), registers named like GPGPU-Sim special
    registers are renamed, and .reqntid (not in its grammar; it only
    constrains the launch) is dropped.  Returns (text, {kernel: new name})."""
    names = {}
    for td in mod:
        if td[0] == "func" and td[2] == ".entry":
            names[td[4]] = "dt_k_" + td[4]
    for old, new in names.items():
        module = re.sub(r"(\.entry\s+)%s\b" % re.escape(old), r"\g<1>" + new, module)
    declared = set(re.findall(r"%([A-Za-z_][A-Za-z0-9_]*)", " ".join(
        re.findall(r"\.reg\b[^;]*;", module))))
    for r in GS_SPECIAL_REGS:
        if r in declared:
            module = re.sub(r"%%%s\b(?!\.)" % r, "%%dt_%s" % r, module)
    module = re.sub(r"\.reqntid\s+\d+(\s*,\s*\d+){0,2}", "", module)
    return module, names


def front(sail_dir):
    global FRONT
    if FRONT is None:
        sys.path.insert(0, sail_dir)
        import ptxfront  # noqa: E402  (the Sail model's own front end)
        FRONT = ptxfront
    return FRONT


class Skip(Exception):
    pass


def run(cmd, env=None, timeout=120, stdin=None):
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=timeout,
                           stdin=stdin)
        return (p.returncode, p.stdout.decode("utf-8", "replace"),
                p.stderr.decode("utf-8", "replace"))
    except subprocess.TimeoutExpired:
        return (-999, "", "timeout")


# --------------------------------------------------------------- Sail side
def sail_run(text, opts, tmp, tag):
    path = os.path.join(tmp, tag + ".ptx")
    with open(path, "w") as f:
        f.write(text)
    fr = os.path.join(opts.sail_dir, "ptxfront.py")
    rc, out, err = run([sys.executable, fr, path], timeout=opts.timeout)
    if rc != 0:
        return None, "ptxfront: " + (err.strip().splitlines() or ["?"])[-1]
    tok = path + ".tok"
    with open(tok, "w") as f:
        f.write(out)
    env = dict(os.environ, SAIL_INPUT=tok)
    rc, out, err = run([opts.sail_bin], env=env, timeout=opts.timeout)
    bufs = {}
    exitcode = None
    for line in out.splitlines():
        m = re.match(r"^(\S+) = \[(.*)\]$", line)
        if m:
            vals = [int(x) for x in m.group(2).split(",") if x.strip()]
            bufs[m.group(1)] = bytes(v & 0xFF for v in vals)
        m = re.match(r"^EXIT (\d+)", line)
        if m:
            exitcode = int(m.group(1))
    if rc != 0 or exitcode not in (0, None):
        tail = (out + err).strip().splitlines()[-3:]
        return None, "sail rc=%s %s" % (rc, " | ".join(tail))
    return bufs, None


# --------------------------------------------------------------- parameters
def param_layout(mod, kname, fr):
    for td in mod:
        if td[0] == "func" and td[4] == kname:
            params = td[5] or []
            out, off = [], 0
            for (a1, t, a2, it) in params:
                ty = fr.type_of(t)
                size = fr.item_size2(it, ty) if it[0] == "array" else fr.tbytes(ty)
                align = fr.align_of(a1) or fr.align_of(a2) or size
                off = (off + align - 1) // align * align
                out.append((off, size, ty))
                off += size
            return out
    raise Skip("kernel %s not found" % kname)


def arg_bytes(arg, size, ty, fr):
    k = arg[0]
    if k == "int":
        v = arg[1]
        if ty[0] == "f":
            v = float(v)
            return struct.pack("<f", v) if size == 4 else struct.pack("<d", v)
        return (v & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
    if k == "float":
        d = struct.unpack("<d", struct.pack("<Q", arg[1]))[0]
        if size == 4:
            return struct.pack("<f", d)
        if size == 8:
            return struct.pack("<d", d)
        raise Skip("float literal for %d-byte param" % size)
    if k == "hex":
        s = arg[1].lower().rstrip("u")
        neg = s.startswith("-")
        s = s.lstrip("+-")
        if s[:2] in ("0f", "0d"):
            v = int(s[2:], 16)
            if s[:2] == "0d" and size == 4:
                v = struct.unpack("<I", struct.pack(
                    "<f", struct.unpack("<d", struct.pack("<Q", v))[0]))[0]
        elif s[:2] == "0x":
            v = int(s[2:], 16)
        elif s[:2] == "0b":
            v = int(s[2:], 2)
        else:
            raise Skip("literal %s" % arg[1])
        if neg:
            v = -v
        return (v & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
    raise Skip("launch argument %r" % (arg,))


# --------------------------------------------------------------- one test
def prepare(path, opts):
    fr = front(opts.sail_dir)
    text = open(path).read()
    try:
        p = fr.Parser(fr.lex(re.sub(r"(\.source_begin)\b.*?(\.source_end)\b",
                                    r"\1 \2", text, flags=re.S)))
        mod = p.module()
    except Exception as e:  # noqa: BLE001
        raise Skip("ptxfront: %s" % e)
    kinds = {td[0] for td in mod}
    bad = sorted(k for k in kinds if k in HARNESS_KW and k not in SUPPORTED)
    if bad:
        raise Skip("harness " + ",".join(bad))
    launches = [td for td in mod if td[0] == "launch"]
    if not launches:
        raise Skip("no launch")
    for td in launches:
        if td[4]:
            raise Skip("dynamic shared memory")
    buffers = [td for td in mod if td[0] == "buffer"]
    names = [td[1] for td in buffers]
    harness = [m for m in HARNESS_RE.finditer(text)]
    module = HARNESS_RE.sub("", text)
    if re.search(r"(?m)^[ \t]*\.(%s)\b" % "|".join(HARNESS_KW), module):
        raise Skip("harness line not at statement start")
    keep = [m.group(0).strip() for m in harness
            if m.group(1) in ("buffer", "launch", "isacheck")]
    bufdecl = [m.group(0).strip() for m in harness if m.group(1) in ("buffer", "isacheck")]
    prints = ["." + "print %s .u8;" % n for n in names]
    init_prog = module + "\n" + "\n".join(bufdecl + prints) + "\n"
    final_prog = module + "\n" + "\n".join(keep + prints) + "\n"
    gs_module, kmap = gs_rewrite(module, mod)
    return fr, mod, (module, gs_module, kmap), launches, names, init_prog, final_prog


def compare_one(path, opts):
    res = {"test": path}
    try:
        fr, mod, module, launches, names, init_prog, final_prog = prepare(path, opts)
    except Skip as e:
        res.update(status="SKIP", why=str(e))
        return res
    if opts.keep:
        tmp = tempfile.mkdtemp(prefix="dt_%s_" % os.path.basename(path), dir=opts.tmp)
        res["dir"] = tmp
        return compare_in(path, opts, res, tmp, fr, mod, module, launches, names, init_prog, final_prog)
    with tempfile.TemporaryDirectory(prefix="dt_", dir=opts.tmp) as tmp:
        return compare_in(path, opts, res, tmp, fr, mod, module, launches, names, init_prog, final_prog)


def compare_in(path, opts, res, tmp, fr, mod, module, launches, names, init_prog, final_prog):
    if True:
        module, gs_module, kmap = module
        gs_ptx = os.path.join(tmp, "gs.ptx")
        with open(gs_ptx, "w") as f:
            f.write(gs_module)
        init, err = sail_run(init_prog, opts, tmp, "init")
        if init is None:
            res.update(status="SAIL-FAIL", why="initial image: " + err)
            return res
        job = ["ptx " + gs_ptx]
        for n in names:
            b = init.get(n, b"")
            job.append("alloc %s %d %s" % (n, len(b), b.hex() or "-"))
        try:
            for (_, k, g, b, _dyn, args) in launches:
                lay = param_layout(mod, k, fr)
                if len(args) != len(lay):
                    raise Skip("argument count")
                job += ["kernel " + kmap.get(k, k), "grid %d %d %d" % tuple(g),
                        "block %d %d %d" % tuple(b)]
                for a, (off, size, ty) in zip(args, lay):
                    if a[0] == "name" and a[1] in names:
                        job.append("paramptr %d %d %s 0" % (off, size, a[1]))
                    elif a[0] in ("nameplus", "nameint") and a[1] in names:
                        job.append("paramptr %d %d %s %d" % (off, size, a[1], a[2]))
                    else:
                        job.append("param %d %d %s" % (off, size, arg_bytes(a, size, ty, fr).hex()))
                job.append("launch")
        except Skip as e:
            res.update(status="SKIP", why=str(e))
            return res
        job.append("dump")
        jf = os.path.join(tmp, "gs.job")
        with open(jf, "w") as f:
            f.write("\n".join(job) + "\n")
        of = os.path.join(tmp, "gs.out")
        rc, out, err = run([opts.gs, jf, of], timeout=opts.timeout,
                           stdin=subprocess.DEVNULL)
        gs = {}
        done = parsed = False
        if os.path.exists(of):
            for line in open(of):
                if line.startswith("BUF "):
                    _, n, h = line.split()
                    gs[n] = bytes.fromhex(h) if h != "-" else b""
                elif line.strip() == "DONE":
                    done = True
                elif line.strip() == "PARSED":
                    parsed = True
        if rc != 0 or not done:
            text = [l for l in (out + "\n" + err).splitlines() if l.strip()]
            key = [l for l in text if re.search(
                r"not implemented|ERROR|[Ee]rror|Assertion|abort|unsupported|syntax", l)]
            msg = " | ".join((key or text)[-2:])
            msg = re.sub(r"/\S*/", "", msg)   # file names without directories
            if rc == -999:
                msg = "timeout"
            res.update(status="GS-FAIL" if parsed else "GS-PARSE",
                       why=("rc=%s %s" % (rc, msg))[:300])
            return res
        final, err = sail_run(final_prog, opts, tmp, "final")
        if final is None:
            res.update(status="SAIL-FAIL", why=err)
            return res
        diffs = []
        for n in names:
            a, b = final.get(n), gs.get(n)
            if a != b:
                pos = [i for i in range(max(len(a or b""), len(b or b"")))
                       if (a[i:i + 1] if a else b"") != (b[i:i + 1] if b else b"")]
                diffs.append({"buffer": n, "bytes": len(pos), "first": pos[:8],
                              "sail": (a or b"").hex(), "gpgpusim": (b or b"").hex()})
        res.update(status="DIFF" if diffs else "MATCH", diffs=diffs)
        return res


def show_diff(d, width=4):
    s, g = bytes.fromhex(d["sail"]), bytes.fromhex(d["gpgpusim"])
    lines = []
    seen = set()
    for i in d["first"]:
        w = i // width * width
        if w in seen:
            continue
        seen.add(w)
        lines.append("      %s+%d: sail %s  gpgpusim %s" % (
            d["buffer"], w, s[w:w + width].hex(), g[w:w + width].hex()))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="+")
    ap.add_argument("--sail-dir", default=DEFAULT_SAIL_DIR,
                    help="directory holding ptxfront.py and build/ptx")
    ap.add_argument("--sail-bin", default=None)
    ap.add_argument("--gs", default=DEFAULT_GS, help="gpgpusim-ptx binary")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--json", help="write per-test results here")
    ap.add_argument("--known", default=os.path.join(HERE, "known_deviations.json"),
                    help="JSON {test basename: reason} of analysed differences")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the work directories")
    opts = ap.parse_args()
    opts.sail_bin = opts.sail_bin or os.path.join(opts.sail_dir, "build", "ptx")
    known = {}
    if opts.known and os.path.exists(opts.known):
        known = json.load(open(opts.known))
    results = []
    with cf.ThreadPoolExecutor(max_workers=opts.jobs) as ex:
        futs = {ex.submit(compare_one, t, opts): t for t in opts.tests}
        for fu in cf.as_completed(futs):
            r = fu.result()
            base = os.path.basename(r["test"])
            if r["status"] == "DIFF" and base in known:
                r["status"] = "DIFF-KNOWN"
                r["why"] = known[base]
            results.append(r)
            if not opts.quiet or r["status"] in ("DIFF",):
                print("%-10s %s %s%s" % (r["status"], r["test"], r.get("why", ""),
                                         (" [%s]" % r["dir"]) if "dir" in r else ""), flush=True)
                if r["status"] == "DIFF":
                    for d in r["diffs"]:
                        print("    %s: %d byte(s) differ" % (d["buffer"], d["bytes"]))
                        print(show_diff(d))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("---- " + ", ".join("%s %d" % kv for kv in sorted(counts.items())))
    if opts.json:
        with open(opts.json, "w") as f:
            json.dump(sorted(results, key=lambda r: r["test"]), f, indent=1)
    sys.exit(1 if counts.get("DIFF") else 0)


if __name__ == "__main__":
    main()

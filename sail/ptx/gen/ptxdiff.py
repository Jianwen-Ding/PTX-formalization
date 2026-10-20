#!/usr/bin/env python3
"""Run the generated PTX tests (sail/ptx/gen/ptxgen.py) through ZLUDA (zoc --target agx) and the AGX functional
simulator (agx_run) with ZLUDA's harness (ptx/tools/agx_ptx_formalization_check.py), and compare with the
.expect values of the PTX Sail model, per instruction form (kernel).

A file whose compilation fails as a whole (a crash, a compile error) is split into one file per kernel, so that
every form gets its own verdict:
  pass        every element agrees bit-exactly
  fail        some element differs (listed; in default mode classified as a documented FP difference or not)
  rejected    ZLUDA reports the instruction as unsupported
  crash       zoc panics / aborts on the form (a bug of ZLUDA or the AGX backend)
  error       other compile or simulation errors

Default mode (no --strict-fp) has documented FP differences (ZLUDA docs/src/agx.md): f32 always flushes subnormals,
NaN bit patterns, div / sqrt / rcp / transcendental precision.  A default-mode difference is "documented" when each
differing element is explained by one of them.

usage: ptxdiff.py GENDIR [--mode strict|default|both] [-j N] [--json FILE]
"""
import argparse, json, os, re, struct, sys, tempfile, types
from concurrent.futures import ThreadPoolExecutor

ZLUDA = os.environ.get("ZLUDA", "/home/user/ZLUDA")
sys.path.insert(0, os.path.join(ZLUDA, "ptx", "tools"))
sys.dont_write_bytecode = True
import agx_ptx_formalization_check as H  # noqa: E402

ZOC = os.environ.get("ZOC", os.path.join(ZLUDA, "target", "debug", "zoc"))
AGX_RUN = os.environ.get("AGX_RUN", "/home/user/AGX-Performance-Simulator/rust/target/release/agx_run")
LLVM_BIN = os.environ.get("LLVM_BIN", "/home/user/build-llvm/bin")


def split_kernels(text):
    """header, list of (index, kernel text, harness lines) of a generated module"""
    head_end = text.index("// ", text.index(".address_size"))
    head = text[:head_end]
    kern = {}
    for m in re.finditer(r"(// [^\n]*\n\.visible \.entry k(\d+)\(.*?\n\}\n)", text, re.S):
        kern[int(m.group(2))] = m.group(1)
    harn = {}
    for l in text.splitlines():
        m = re.match(r"\.(buffer|launch|expect) [IOk]?(\d+)\b", l)
        if m:
            harn.setdefault(int(m.group(2)), []).append(l)
    return head, [(i, kern[i], harn.get(i, [])) for i in sorted(kern)]


def form_names(text):
    return {int(m.group(2)): m.group(1) for m in re.finditer(r"// ([^\n]*)\n\.visible \.entry k(\d+)\(", text)}


def run(path, strict, timeout=300):
    a = types.SimpleNamespace(zoc=ZOC, agx_run=AGX_RUN, readelf=os.path.join(LLVM_BIN, "llvm-readelf"),
                              objcopy=os.path.join(LLVM_BIN, "llvm-objcopy"), arch="agx-g14g", strict_fp=strict, timeout=timeout,
                              max_failures=100000, counts={"pass": 0, "fail": 0, "skip": 0})
    try:
        st, detail = H.run_test(path, a)
    except H.Unsupported as e:
        st, detail = "UNSUPPORTED", str(e)
    except Exception as e:
        st, detail = "HARNESS-ERROR", "%s: %s" % (type(e).__name__, e)
    return st, detail


def failures(detail):
    """O<i>[k] expected X, got Y -> {i: [(k, X, Y)]}"""
    out = {}
    for m in re.finditer(r"O(\d+)\[(\d+)\] expected (0x[0-9a-f]+), got (0x[0-9a-f]+)", detail):
        out.setdefault(int(m.group(1)), []).append((int(m.group(2)), int(m.group(3), 16), int(m.group(4), 16)))
    return out


def rejected_names(detail):
    return set(x.strip().rstrip(";") for x in detail.split("rejected:")[-1].split("|")) if "rejected" in detail else set()


def per_kernel(path, strict, tmpdir):
    """{kernel index: (verdict, info)} for one generated file"""
    text = open(path).read()
    names = form_names(text)
    st, detail = run(path, strict)
    res = {}
    if st in ("PASS", "PARTIAL"):
        fl = failures(detail)
        skipped = set()
        m = re.search(r"skipped after rejected kernels ([\w,]+)", detail)
        if m:
            skipped = {int(k[1:]) for k in m.group(1).split(",")}
        for i in names:
            if i in skipped:
                res[i] = ("rejected", "")
            elif i in fl:
                res[i] = ("fail", fl[i])
            else:
                res[i] = ("pass", "")
        return res
    head, ks = split_kernels(text)
    for i, k, h in ks:
        p = os.path.join(tmpdir, "%s-k%d.ptx" % (os.path.basename(path)[:-4], i))
        with open(p, "w") as f:
            f.write(head + k + "\n" + "\n".join(h) + "\n")
        st, d = run(p, strict)
        os.unlink(p)
        if st == "PASS":
            res[i] = ("pass", "")
        elif st == "FAIL":
            res[i] = ("fail", failures(d).get(i, []))
        elif st in ("PARTIAL", "REJECTED", "UNSUPPORTED") or re.search(r"Unsupported|not supported|Not yet implemented", d):
            res[i] = ("rejected", d[:200])
        elif st == "CRASH" or "Assertion" in d or "panicked" in d:
            res[i] = ("crash", d[:300])
        else:
            res[i] = ("error", "%s %s" % (st, d[:300]))
    return res


# ---------------------------------------------------------------- default-mode classification
def ftype(name):
    for t in ("bf16x2", "f16x2", "bf16", "f16", "f64", "f32"):
        if re.search(r"\.%s\b" % t, name):
            return t
    return None


def is_nan(v, t):
    if t == "f32":
        return (v & 0x7f800000) == 0x7f800000 and v & 0x7fffff
    if t == "f64":
        return (v & 0x7ff0000000000000) == 0x7ff0000000000000 and v & 0xfffffffffffff
    if t in ("f16", "f16x2"):
        return (v & 0x7c00) == 0x7c00 and v & 0x3ff
    return False


def is_sub32(v):
    return (v & 0x7f800000) == 0 and (v & 0x7fffff) != 0


def documented(name, inputs, k, want, got):
    """reason string if the default-mode difference is one of the documented ones, else None"""
    t = ftype(name)
    w32 = want & 0xffffffff
    g32 = got & 0xffffffff
    if t in ("f32", "f64", "f16", "f16x2") and is_nan(want, t if t != "f16x2" else "f16") and is_nan(got, t if t != "f16x2" else "f16"):
        return "NaN bits"
    if t == "f32" or ".f32" in name:
        if is_sub32(w32) or is_sub32(g32) or any(is_sub32(x & 0xffffffff) for x in inputs):
            return "f32 subnormal flush"
        if is_nan(w32, "f32") and is_nan(g32, "f32"):
            return "NaN bits"
    if re.search(r"\b(div|sqrt|rcp|rsqrt|sin|cos|lg2|ex2|tanh)\b", name.split()[0].replace(".", " ")):
        return "precision"
    return None


# approximate instructions: PTX ISA error bounds (relative, absolute), loosely rounded up
APPROX = [(r"\b(rcp|div)\.approx", 2.0 ** -21, 0), (r"\bdiv\.full", 2.0 ** -21, 0), (r"\b(rsqrt|sqrt)\.approx", 2.0 ** -20, 0),
          (r"\bex2\.approx\.(ftz\.)?f32", 2.0 ** -21, 0), (r"\blg2\.approx", 2.0 ** -21, 2.0 ** -21),
          (r"\b(sin|cos)\.approx", 0, 2.0 ** -20), (r"\btanh\.approx", 2.0 ** -10, 2.0 ** -10),
          (r"\bex2\.approx\.f16", 2.0 ** -9, 0)]


def fvalue(v, t):
    if t == "f64":
        return struct.unpack("<d", struct.pack("<Q", v & (2 ** 64 - 1)))[0]
    if t in ("f16", "f16x2"):
        return struct.unpack("<e", struct.pack("<H", v & 0xffff))[0]
    return struct.unpack("<f", struct.pack("<I", v & 0xffffffff))[0]


def approx_ok(name, want, got):
    """'approx' when an approximate instruction's result is within the PTX ISA bound, 'approx-subnormal' when the
    two differ only below the smallest normal (a flushed subnormal result), else None"""
    for pat, rel, ab in APPROX:
        if re.search(pat, name):
            t = ftype(name) or "f32"
            lanes = [(want, got)] if t != "f16x2" else [(want & 0xffff, got & 0xffff), (want >> 16 & 0xffff, got >> 16 & 0xffff)]
            verdict = "approx"
            for w, g in lanes:
                x, y = fvalue(w, t), fvalue(g, t)
                if w == g or (x != x and y != y) or x == y:
                    continue
                if abs(x - y) <= max(rel * abs(x), ab):
                    continue
                tiny = {"f32": 2.0 ** -126, "f64": 2.0 ** -1022, "f16": 2.0 ** -14, "f16x2": 2.0 ** -14}.get(t, 2.0 ** -126)
                if abs(x) < tiny and abs(y) < tiny:
                    verdict = "approx-subnormal"
                    continue
                return None
            return verdict
    return None


def nan_unspecified(name, want, got):
    """abs / neg: the PTX ISA leaves the NaN result unspecified"""
    t = ftype(name)
    if t and re.match(r"(abs|neg)\b", name):
        return t in ("f32", "f64", "f16", "bf16") and is_nan(want, t if t != "bf16" else "f32") is not None and \
            bool(is_nan(want, t)) and bool(is_nan(got, t))
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gendir")
    ap.add_argument("--mode", default="both", choices=["strict", "default", "both"])
    ap.add_argument("-j", type=int, default=3)
    ap.add_argument("--json")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    files = sorted(os.path.join(a.gendir, f) for f in os.listdir(a.gendir) if f.startswith("gen-") and f.endswith(".ptx"))
    summary = {}
    modes = ["strict", "default"] if a.mode == "both" else [a.mode]
    with tempfile.TemporaryDirectory() as tmp:
        for mode in modes:
            with ThreadPoolExecutor(a.j) as ex:
                results = list(ex.map(lambda p: (p, per_kernel(p, mode == "strict", tmp)), files))
            counts, rows = {}, []
            for p, res in results:
                text = open(p).read()
                names = form_names(text)
                inputs = {}
                for m in re.finditer(r"\.buffer I(\d+) \.b64 = \{ (.*?) \};", text):
                    inputs[int(m.group(1))] = [int(x) for x in m.group(2).split(",")]
                nin = {i: max(1, len(re.findall(r"ld\.global\.\w+ %a\d+", k))) for i, k in
                       ((i, k) for i, k, _ in split_kernels(text)[1])}
                for i, (v, info) in sorted(res.items()):
                    if v == "fail":
                        vs = set()
                        for (k, want, got) in info:
                            vs.add(approx_ok(names[i], want, got) or ("nan-unspecified" if nan_unspecified(names[i], want, got) else None))
                        if None not in vs:
                            v = "approx-subnormal" if "approx-subnormal" in vs else (vs.pop() if len(vs) == 1 else "approx")
                    if v == "fail" and mode == "default":
                        reasons = set()
                        for (k, want, got) in info:
                            thread = k // max(1, len(re.findall(r"st\.global\.\w+ \[%rd3", split_kernels(text)[1][i][1])))
                            ins = inputs.get(i, [])[thread * nin[i]:(thread + 1) * nin[i]]
                            reasons.add(documented(names[i], ins, k, want, got))
                        if None not in reasons:
                            v = "documented"
                            info = sorted(reasons)
                    counts[v] = counts.get(v, 0) + 1
                    rows.append((os.path.basename(p), i, names[i], v, info))
            summary[mode] = {"counts": counts, "rows": rows}
            total = sum(counts.values())
            print("PTX %s mode: %d forms: %s" % (mode, total, ", ".join("%d %s" % (n, k) for k, n in sorted(counts.items()))))
            for f, i, n, v, info in rows:
                if v in ("fail", "crash", "error") or (a.v and v != "pass"):
                    shown = info if isinstance(info, str) else ", ".join(
                        "[%d] %#x vs %#x" % x if isinstance(x, tuple) else str(x) for x in info[:4])
                    print("  %-9s %-18s k%-3d %-48s %s" % (v.upper(), f, i, n[:48], shown[:160]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({m: {"counts": s["counts"], "rows": [r[:4] + (str(r[4])[:300],) for r in s["rows"]]} for m, s in summary.items()}, f,
                      indent=1)
    bad = sum(summary[m]["counts"].get(k, 0) for m in summary for k in ("fail", "crash", "error"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

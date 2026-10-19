#!/usr/bin/env python3
"""Run tests/*.ptx on the PTX Sail model and compare with the golden tests/*.out, like
scripts/run-tests.sh does for the K interpreter: a test passes when the model's output equals the
.out file and it exits with 0 (every .expect held).  Without a .out file, the .expect directives are
the reference (exit 0 and no FAIL line).

  ptxtest.py [-j N] [-v] [--report FILE] [tests...]
"""
import os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import ptxfront  # noqa: E402

EXE = os.environ.get("PTX_SAIL", os.path.join(HERE, "build", "ptx"))

def run_one(path, timeout=900):
    t0 = time.time()
    try:
        toks = ptxfront.front(open(path).read())
    except Exception as e:
        return ("FAIL", "front end: %s" % e, "", 0.0)
    with tempfile.NamedTemporaryFile("w", suffix=".tok", delete=False) as f:
        f.write(toks)
        name = f.name
    try:
        p = subprocess.run([EXE], env={**os.environ, "SAIL_INPUT": name}, capture_output=True, text=True, timeout=timeout)
        out, err, code = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return ("FAIL", "timeout", "", time.time() - t0)
    finally:
        os.unlink(name)
    m = re.search(r"EXIT (\d+)", err)
    ex = int(m.group(1)) if m else None
    if os.sep + "neg" + os.sep in path:
        # tests/neg: the program must fail (non-zero exit), with the `// expect: text` of the first line in the output
        first = open(path).readline()
        want = first[len("// expect: "):].strip() if first.startswith("// expect: ") else ""
        ok = ex != 0 and (want == "" or want in out + err)
        return ("PASS" if ok else "FAIL", "" if ok else "exit %s, expected failure %r" % (ex, want), out, time.time() - t0)
    golden = path[:-4] + ".out"
    stuck = re.search(r"stuck: (.*)", err)
    note = stuck.group(1) if stuck else ("crash: " + err.strip()[-300:] if code != 0 or ex is None else "")
    if os.sep + "zluda" + os.sep in path:
        # tests/zluda (scripts/run-zluda.py): the program passes when it exits 0 and its output starts with PASS
        ok = ex == 0 and out.strip().startswith("PASS")
        return ("PASS" if ok else "FAIL", "" if ok else (note or out.strip().replace("\n", " | ")[:160]), out, time.time() - t0)
    if os.path.exists(golden):
        g = open(golden).read()
        ok = ex == 0 and out.rstrip("\n") == g.rstrip("\n")
        if not ok and not note:
            note = "output differs" if ex == 0 else "exit %s" % ex
    else:
        ok = ex == 0 and "FAIL" not in out
    return ("PASS" if ok else "FAIL", note, out, time.time() - t0)

def main(argv):
    jobs, verbose, report = 2, False, None
    args = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-j": jobs = int(argv[i + 1]); i += 2; continue
        if a == "-v": verbose = True; i += 1; continue
        if a == "--report": report = argv[i + 1]; i += 2; continue
        args.append(a); i += 1
    tests = args or sorted(os.path.join(ROOT, "tests", f) for f in os.listdir(os.path.join(ROOT, "tests")) if f.endswith(".ptx"))
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        results = list(ex.map(run_one, tests))
    # tests/zluda: only the programs of tests/zluda/EXPECTED_PASS must pass (scripts/run-zluda.py); the others
    # are reported as XFAIL / XPASS
    zexp_path = os.path.join(ROOT, "tests", "zluda", "EXPECTED_PASS")
    zexp = set(open(zexp_path).read().split()) if os.path.exists(zexp_path) else set()
    npass = nfail = nx = 0
    lines = []
    for t, (st, note, out, dt) in zip(tests, results):
        rel = os.path.relpath(t, ROOT)
        if os.sep + "zluda" + os.sep in t and os.path.basename(t)[:-4] not in zexp:
            st = "XPASS" if st == "PASS" else "XFAIL"
        if st == "PASS":
            npass += 1
        elif st == "FAIL":
            nfail += 1
        else:
            nx += 1
        line = "%s  %s%s" % (st, rel, "" if st in ("PASS", "XPASS") else "  (%s)" % note)
        lines.append(line)
        print(line, flush=True)
        if verbose and st == "FAIL":
            golden = t[:-4] + ".out"
            if os.path.exists(golden):
                import difflib
                for d in list(difflib.unified_diff(open(golden).read().splitlines(), out.splitlines(), "golden", "sail", lineterm=""))[:20]:
                    print("      " + d)
    summary = "---- %d passed, %d failed%s" % (npass, nfail, ", %d not expected to pass (XFAIL / XPASS)" % nx if nx else "")
    print(summary)
    if report:
        with open(report, "w") as f:
            f.write("\n".join(lines) + "\n" + summary + "\n")
    return 0 if nfail == 0 else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

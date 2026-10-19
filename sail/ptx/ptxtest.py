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
    golden = path[:-4] + ".out"
    stuck = re.search(r"stuck: (.*)", err)
    note = stuck.group(1) if stuck else ("crash: " + err.strip()[-300:] if code != 0 or ex is None else "")
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
    npass = 0
    lines = []
    for t, (st, note, out, dt) in zip(tests, results):
        rel = os.path.relpath(t, ROOT)
        if st == "PASS":
            npass += 1
        line = "%s  %s%s" % (st, rel, "" if st == "PASS" else "  (%s)" % note)
        lines.append((rel, st, note, dt))
        print(line, flush=True)
        if verbose and st != "PASS":
            golden = t[:-4] + ".out"
            if os.path.exists(golden):
                import difflib
                for d in list(difflib.unified_diff(open(golden).read().splitlines(), out.splitlines(), "golden", "sail", lineterm=""))[:20]:
                    print("      " + d)
    print("---- %d passed, %d failed" % (npass, len(tests) - npass))
    if report:
        with open(report, "w") as f:
            for rel, st, note, dt in lines:
                f.write("%s\t%s\t%.1f\t%s\n" % (rel, st, dt, note))
    return 0 if npass == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

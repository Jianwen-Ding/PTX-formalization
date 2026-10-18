#!/usr/bin/env python3
"""Parse-check every example snippet of the PTX ISA document.

usage: isa_examples.py examples.json   (list of {"sec", "code"}; see docs/prior-work.md for how it is extracted)
Each snippet is wrapped in a kernel body (or placed at module level when it defines functions or directives) and given to
`kast` with the PTX grammar.  The ISA examples are fragments without declarations of their registers, so only syntactic
acceptance is checked here; execution is covered by the instruction tests.
"""
import json, os, re, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

DEF = os.environ.get("PTX_DEF", "build/ptx-kompiled")
HDR = ".version 9.4\n.target sm_100a\n.address_size 64\n"

def wrap(code):
    code = re.sub(r'\.{3,}|…', '', code)                                    # elision marks of the document
    lines = [l for l in code.splitlines() if not re.match(r'^\s*Examples?\s*\d*\s*:\s*$', l)]
    code = "\n".join(lines)
    top = re.search(r'^\s*(\.visible|\.entry|\.func|\.weak|\.extern|\.version|\.target|\.alias|\.callprototype|\.pragma|\.section|\.file|\.loc|\.branchtargets|\.common|\.global|\.const|\.tex|\.surf|\.sampler)', code, re.M)
    if re.search(r'^\s*(\.visible|\.entry|\.func|\.weak|\.extern|\.version|\.target|\.alias|\.section|\.file|\.common)', code, re.M):
        return HDR + code + "\n"
    return HDR + ".visible .entry k()\n{\n" + code + "\n}\n"

def check(item):
    i, ex = item
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(wrap(ex["code"]))
    try:
        r = subprocess.run(["kast", "-d", DEF, f.name, "-o", "none"], capture_output=True, text=True,
                           env={**os.environ, "K_OPTS": "-Xss512m"})
        ok = r.returncode == 0
        msg = "" if ok else next((l for l in r.stderr.splitlines() if "Error" in l or "error" in l or "Parse" in l), r.stderr[-200:])[:240]
    finally:
        os.unlink(f.name)
    return i, ex["sec"], ok, msg

if __name__ == "__main__":
    exs = json.load(open(sys.argv[1]))
    jobs = int(os.environ.get("JOBS", "3"))
    with ThreadPoolExecutor(jobs) as pool:
        res = list(pool.map(check, enumerate(exs)))
    bad = [r for r in res if not r[2]]
    for i, sec, ok, msg in bad:
        print(f"FAIL  #{i} {sec}: {msg}")
    print(f"---- {len(res) - len(bad)} parsed, {len(bad)} rejected of {len(res)}")
    sys.exit(1 if bad else 0)

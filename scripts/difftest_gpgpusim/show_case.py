#!/usr/bin/env python3
"""Print the minimised cases of a gen_random.py fuzz run with, for each
differing register, the thread's initial register values and both results.

    show_case.py DIR/fuzz.jsonl [--seed N] [--max 3]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_random as G  # noqa: E402


def init_regs(prog, t):
    words, _ = prog.inputs()
    w = words[t * G.INW:(t + 1) * G.INW]
    out, i = {}, 0
    for r in G.regs("r"):
        out[r] = "%08x" % w[i]; i += 1
    for r in G.regs("rd"):
        out[r] = "%016x" % (w[i] | w[i + 1] << 32); i += 2
    for r in G.regs("f"):
        out[r] = "%08x" % w[i]; i += 1
    for r in G.regs("fd"):
        out[r] = "%016x" % (w[i] | w[i + 1] << 32); i += 2
    for r in G.regs("h"):
        out[r] = "%04x" % (w[i] & 0xFFFF); i += 1
    for k, r in enumerate(G.regs("p")):
        out[r] = str(w[i] >> k & 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max", type=int, default=3)
    a = ap.parse_args()
    for line in open(a.jsonl):
        rec = json.loads(line)
        if "min_body" not in rec or (a.seed is not None and rec["seed"] != a.seed):
            continue
        G.EXCLUDE = rec.get("exclude")
        prog = G.Prog(rec["seed"], rec["cats"], all_forms=rec.get("all_forms", False),
                      raw=rec.get("raw", False))
        print("== seed %d" % rec["seed"])
        print("\n".join("   " + x for x in rec["min_body"].splitlines()))
        used = set(tok.strip(",;[]{}!@") for tok in rec["min_body"].replace(",", " ").split())
        for d in rec["min_diffs"][:a.max]:
            if "thread" not in d:
                print("   %s" % d)
                continue
            regs = init_regs(prog, d["thread"])
            ins = " ".join("%s=%s" % (k, v) for k, v in regs.items() if k in used)
            print("   t%d %s: sail %s gpgpusim %s | in: %s" % (d["thread"], d["reg"], d["sail"], d["gpgpusim"], ins))


if __name__ == "__main__":
    main()

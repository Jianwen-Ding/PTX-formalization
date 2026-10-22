#!/usr/bin/env python3
"""Seeded differential fuzzer: PTX Sail model vs an independent MPFR / OCP-MX oracle.

  fuzz.py [--seed S] [--cases N] [--batch B] [--jobs J] [--only REGEX] [--list] [--json FILE]

For every group of instruction variants (ops.py) it draws N edge-biased input tuples (or the
exhaustive input set for the small up-conversions), runs them on the model in programs of B
threads (every thread evaluates every variant of the group), and compares each result lane with
the oracle.  Every disagreement is re-run as a single-thread, single-instruction program (the
minimal case) before it is reported.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ops                     # noqa: E402
import runner                  # noqa: E402
from oracle import check       # noqa: E402


def mask(w):
    return (1 << w) - 1


def compare(var, inputs, got):
    """-> list of (status, tag, lane, expected, got-lane)"""
    res = []
    for (sh, w, fmt, e) in var.expect(inputs):
        g = (got >> sh) & mask(w)
        st, tag = check(fmt, e, g)
        res.append((st, tag, sh, e, g))
    # bits outside every lane must be zero (zero extension, padding)
    lanes = 0
    for (sh, w, fmt, e) in var.expect(inputs):
        lanes |= mask(w) << sh
    extra = got & ~lanes & mask(var.ow)
    if extra:
        res.append(("fail", "nonzero bits outside the result lanes", -1, 0, extra))
    return res


def minimise(group, var, inputs):
    """Single-thread program with only this variant; then try zeroing other packed lanes."""
    def run1(x):
        try:
            got = runner.run(group.target, group.in_widths, [(var.tmpl, var.ow)], [x])[0][0]
        except runner.ModelError as ex:
            return None, str(ex)[:300]
        bad = [r for r in compare(var, x, got) if r[0] == "fail"]
        return got, bad

    got, bad = run1(inputs)
    if got is None or not bad:
        return {"reproduced": False, "inputs": inputs, "note": "did not reproduce alone" if got is not None else bad}
    x = list(inputs)
    # try simplifying: clear bytes of the inputs one at a time while the failure persists
    for j, w in enumerate(group.in_widths):
        for byte in range(w // 8):
            y = list(x)
            y[j] = y[j] & ~(0xFF << (8 * byte))
            if y[j] == x[j]:
                continue
            g2, b2 = run1(y)
            if g2 is not None and b2:
                x, got, bad = y, g2, b2
    return {"reproduced": True, "inputs": x, "got": got,
            "lanes": [(sh, repr(e), g) for (_, _, sh, e, g) in bad]}


def run_group(args):
    gname, seed, ncases, batch, keepdir = args
    G = {g.name: g for g in ops.groups()}
    group = G[gname]
    rng = random.Random("%s:%s" % (seed, gname))
    if group.exhaustive:
        cases = group.exhaustive()
    else:
        cases = [group.gen(rng) for _ in range(ncases)]
    t0 = time.time()
    stats = {v.name: Counter() for v in group.variants}
    tags = {v.name: Counter() for v in group.variants}
    anyobs = defaultdict(Counter)
    fails = defaultdict(list)
    model_time = 0.0
    errors = []
    for k in range(0, len(cases), batch):
        for part in [cases[k:k + batch]]:
            t1 = time.time()
            try:
                outs = runner.run(group.target, group.in_widths, [(v.tmpl, v.ow) for v in group.variants], part,
                                  keep=os.path.join(keepdir, gname + ".ptx") if keepdir else None)
            except runner.ModelError as ex:
                errors.append(str(ex)[:500])
                continue
            model_time += time.time() - t1
            for vi, v in enumerate(group.variants):
                for ci, x in enumerate(part):
                    res = compare(v, x, outs[vi][ci])
                    stats[v.name]["cases"] += 1
                    worst = "ok"
                    for (st, tag, sh, e, g) in res:
                        stats[v.name]["lanes"] += 1
                        stats[v.name][st] += 1
                        if st in ("alt", "soft", "any"):
                            tags[v.name][(st, tag, g if st == "any" else None)] += 1
                        if st == "fail":
                            worst = "fail"
                    if worst == "fail":
                        stats[v.name]["fail_cases"] += 1
                        if len(fails[v.name]) < 6:
                            fails[v.name].append((x, outs[vi][ci], [(sh, repr(e), g) for (st, tag, sh, e, g) in res if st == "fail"]))
    minimal = {}
    for vn, fl in fails.items():
        v = next(v for v in group.variants if v.name == vn)
        minimal[vn] = minimise(group, v, fl[0][0])
    return {"group": gname, "area": group.area, "cases": len(cases), "variants": len(group.variants),
            "stats": {k: dict(c) for k, c in stats.items()},
            "tags": {k: [[list(t), n] for t, n in c.items()] for k, c in tags.items() if c},
            "fails": {k: [[list(map(hex, x)), hex(g), l] for (x, g, l) in fl] for k, fl in fails.items()},
            "minimal": minimal, "errors": errors, "model_time": model_time, "time": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="1")
    ap.add_argument("--cases", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=None, help="regex on group names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep", default=None, help="directory to keep the last program of each group")
    a = ap.parse_args()
    groups = ops.groups()
    if a.only:
        groups = [g for g in groups if re.search(a.only, g.name)]
    if a.list:
        for g in groups:
            print("%-28s %3d variants  %s" % (g.name, len(g.variants), g.area))
        print(len(groups), "groups,", sum(len(g.variants) for g in groups), "variants")
        return
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    jobs = [(g.name, a.seed, a.cases, a.batch, a.keep) for g in groups]
    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(run_group, jobs):
            results.append(r)
            tot = Counter()
            for c in r["stats"].values():
                tot.update(c)
            print("%-28s cases %5d x %3d variants  lanes %7d  ok %7d alt %5d soft %4d any %4d FAIL %5d  (%.0fs, model %.0fs)%s" % (
                r["group"], r["cases"], r["variants"], tot["lanes"], tot["ok"], tot["alt"], tot["soft"], tot["any"],
                tot["fail"], r["time"], r["model_time"], "  ERRORS %d" % len(r["errors"]) if r["errors"] else ""), flush=True)
            agg = Counter()
            for vn, tl in r["tags"].items():
                for (st, tag, g), n in tl:
                    agg[(st, tag)] += n
            for (st, tag), n in sorted(agg.items(), key=lambda t: -t[1]):
                print("    %-5s %6d  %s" % (st, n, tag), flush=True)
            for vn, mn in r["minimal"].items():
                print("    FAIL %-40s %s" % (vn, json.dumps(mn, default=str)[:400]), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, default=str)
    grand = Counter()
    for r in results:
        for c in r["stats"].values():
            grand.update(c)
    print("TOTAL: %d variant-cases, %d lanes, ok %d, alt %d, soft %d, any %d, FAIL %d" % (
        grand["cases"], grand["lanes"], grand["ok"], grand["alt"], grand["soft"], grand["any"], grand["fail"]))


if __name__ == "__main__":
    main()

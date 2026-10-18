#!/usr/bin/env python3
"""Check schedule-independence properties of a PTX program by exhaustive search (krun --search).

  properties.py prog.ptx [prog2.ptx ...]

For every program the search explores all thread interleavings and reports
  * solutions       number of distinct final configurations,
  * final memories  number of distinct final global memories      (1  <=> the result does not depend on the schedule: the program is
                    deterministic; every data-race-free program must have 1),
  * stuck           final configurations in which some CTA still exists (deadlock / mismatched collective),
and the verdicts  DETERMINISTIC | NONDETERMINISTIC (racy or order-dependent)  and  TERMINATES | DEADLOCKS.
PTX_DEF selects the definition (default build/ptx-mm-kompiled, the reduced definition built with `make build-mm`).
"""
import os, re, subprocess, sys

DEF = os.environ.get("PTX_DEF", "build/ptx-mm-kompiled")

def search(prog, pattern):
    out = subprocess.run(["krun", "-d", DEF, prog, "--search", "--pattern", pattern], capture_output=True, text=True,
                         env={**os.environ, "K_OPTS": os.environ.get("K_OPTS", "-Xss512m")}, timeout=int(os.environ.get("PROP_TIMEOUT", "600"))).stdout
    # the output is a matching-logic disjunction:  { G:Bytes #Equals b"..." } #Or { ... }
    sols = [re.sub(r"\s+", " ", b.strip()) for b in re.split(r"^#Or\s*$", out, flags=re.M) if "#Equals" in b]
    return [b.split("#Equals", 1)[1].strip(" }") for b in sols]

def check(prog):
    mems = search(prog, "<gmem> G:Bytes </gmem>")
    ctas = search(prog, "<ctas> C:Bag </ctas>")
    stuck = sum(1 for c in ctas if c.strip() != ".CtaCellMap")
    print(f"{prog}: {len(mems)} solutions, {len(set(mems))} final memories, {stuck} stuck -> "
          f"{'DETERMINISTIC' if len(set(mems)) <= 1 else 'NONDETERMINISTIC'}, {'DEADLOCKS' if stuck else 'TERMINATES'}")
    return len(set(mems)) <= 1 and not stuck

if __name__ == "__main__":
    ok = [check(p) for p in sys.argv[1:]]
    sys.exit(0 if all(ok) else 1)

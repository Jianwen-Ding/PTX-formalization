#!/usr/bin/env python3
"""Cross-check the K semantics (relaxed memory mode, exhaustive search) against the axiomatic PTX model.

For every litmus test of tests/memmodel/*.litmus that can be translated (scripts/litmus2ptx.py) the axiomatic checker
decides whether the outcome of interest is allowed, and `krun --search` explores all schedules of the K semantics.
  sound      K never reaches an outcome the model forbids              (must hold: reported as PASS / FAIL)
  complete   K reaches every outcome the model allows                  (informational: reported as `gap`)
"""
import os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import ptxmm, litmus2ptx

DEF = os.environ.get('PTX_DEF', 'build/ptx-search-kompiled')


def k_reachable(src):
    with tempfile.NamedTemporaryFile('w', suffix='.ptx', delete=False) as f:
        f.write(src)
    try:
        env = dict(os.environ, K_OPTS=os.environ.get('K_OPTS', '-Xss512m'))
        r = subprocess.run(['krun', '-d', DEF, f.name, '--search', '--pattern', '<exit-code> X:Int </exit-code>'],
                           capture_output=True, text=True, env=env, timeout=1800)
        codes = set(re.findall(r'^\s+(\d+)\s*$', r.stdout.split('X:Int', 1)[-1], re.M)) if 'X:Int' in r.stdout else set()
        return '0' in codes, codes
    finally:
        os.unlink(f.name)


def main(argv):
    names = set(argv[1:])
    bad = gaps = n = 0
    for fn in sorted(os.listdir('tests/memmodel')):
        if not fn.endswith('.litmus'): continue
        for t in ptxmm.parse(open('tests/memmodel/' + fn).read()):
            if names and t['name'] not in names: continue
            if not litmus2ptx.translatable(t): continue
            allowed = any(ptxmm.holds(t['exists'], o) for o in ptxmm.run(t))
            reach, codes = k_reachable(litmus2ptx.translate(t))
            n += 1
            if reach and not allowed:
                print(f"FAIL  {t['name']:28s} K reaches an outcome the PTX model forbids"); bad += 1
            elif allowed and not reach:
                print(f"gap   {t['name']:28s} allowed by the model, not reached by K"); gaps += 1
            else:
                print(f"PASS  {t['name']:28s} {'allowed' if allowed else 'forbidden'}")
    print(f"---- {n} tests, {bad} unsound, {gaps} gaps")
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

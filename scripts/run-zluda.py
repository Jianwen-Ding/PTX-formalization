#!/usr/bin/env python3
"""Run the imported ZLUDA conformance programs (tests/zluda/*.ptx).

Each program is classified PASS / FAIL (ran, output differs) / STUCK (parse
error, unsupported instruction, deadlock, crash).  tests/zluda/EXPECTED_PASS
lists the programs that must pass; any listed program that does not is an
error, and newly passing programs are reported so the list can be updated.

usage: run-zluda.py [--update] [name ...]
"""
import concurrent.futures, os, subprocess, sys

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
tests = os.path.join(root, 'tests', 'zluda')
exp_path = os.path.join(tests, 'EXPECTED_PASS')
update = '--update' in sys.argv
names = [a for a in sys.argv[1:] if not a.startswith('--')] or \
        sorted(f[:-4] for f in os.listdir(tests) if f.endswith('.ptx'))
env = dict(os.environ)
env['JAVA_TOOL_OPTIONS'] = env.get('JAVA_TOOL_OPTIONS', '') + ' -Xss512m'

def run(name):
    try:
        r = subprocess.run(['krun', '-d', os.path.join(root, 'build/ptx-kompiled'),
                            os.path.join(tests, name + '.ptx'), '--output', 'none'],
                           capture_output=True, text=True, timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return name, 'STUCK', 'timeout'
    out = r.stdout.strip()
    if r.returncode == 0 and out.startswith('PASS'):
        return name, 'PASS', ''
    if 'FAIL' in out:
        return name, 'FAIL', out.replace('\n', ' | ')[:200]
    err = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and 'Picked up' not in l and not l.startswith('/tmp')]
    return name, 'STUCK', (err[0] if err else '')[:160]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results = sorted(ex.map(run, names))
res = {n: (s, m) for n, s, m in results}
passed = [n for n, (s, _) in res.items() if s == 'PASS']
for n, (s, m) in res.items():
    if s != 'PASS':
        print(f'{s:5} {n}: {m}')
print(f'---- {len(passed)} pass, {sum(1 for s, _ in res.values() if s == "FAIL")} fail, '
      f'{sum(1 for s, _ in res.values() if s == "STUCK")} stuck, of {len(res)}')
expected = set(open(exp_path).read().split()) if os.path.exists(exp_path) else set()
if update:
    open(exp_path, 'w').write('\n'.join(sorted(set(passed) | (expected - set(names)))) + '\n')
    print('EXPECTED_PASS updated')
    sys.exit(0)
regress = sorted(n for n in expected if n in res and res[n][0] != 'PASS')
for n in regress: print('REGRESSION', n)
new = sorted(set(passed) - expected)
if new: print('newly passing (run --update):', ' '.join(new))
sys.exit(1 if regress else 0)

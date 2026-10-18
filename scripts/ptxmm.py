#!/usr/bin/env python3
"""An executable axiomatic model of the PTX memory consistency model (PTX ISA, chapter 8).

Litmus tests are small text files (tests/memmodel/*.litmus):

    test MP+fences
    init x=0 flag=0
    thread T1 cta=0 cluster=0 gpu=0
      st.weak x 1
      fence.acq_rel.sys
      st.relaxed.sys flag 1
    thread T2 cta=1 cluster=0 gpu=0
      r0 = ld.relaxed.sys flag
      fence.acq_rel.sys
      r1 = ld.weak x
    exists r0==1 && r1==0
    expect forbidden

Instructions:  st[.weak|.relaxed|.release|.volatile][.cta|.cluster|.gpu|.sys] ADDR VAL
               rN = ld[.weak|.relaxed|.acquire|.volatile][.scope] ADDR
               rN = atom[.sem][.scope].OP ADDR VAL [VAL2]      red[.sem][.scope].OP ADDR VAL
               fence.sc.SCOPE | fence.acq_rel.SCOPE | fence.proxy.KIND        bar.sync N | bar.arrive N
VAL is an integer or a register (a data dependency).  A leading `? rN==V :` makes the instruction conditional
(a control dependency).  `@proxy=NAME` after an instruction sets the memory proxy (default generic), and
`alias a b` declares two virtual addresses that map to the same location (each is its own proxy).

The checker enumerates read-from choices, per-location write orders and Fence-SC orders and keeps the
executions that satisfy the axioms Coherence, Fence-SC, Atomicity, No-Thin-Air, SC-per-Location and Causality.
"""
import itertools, re, sys, os
from collections import defaultdict

SCOPES = {'cta': 1, 'cluster': 2, 'gpu': 3, 'sys': 4}
STRONG = {'relaxed', 'acquire', 'release', 'acq_rel', 'volatile', 'mmio'}


class Ev:
    def __init__(s, **k):
        s.__dict__.update(k)
    def __repr__(s):
        return f"{s.tid}.{s.idx}:{s.kind}"


def parse(text):
    tests = []
    cur = None
    thread = None
    for raw in text.split('\n'):
        line = raw.split('#')[0].rstrip()
        if not line.strip():
            continue
        w = line.split()
        if w[0] == 'test':
            cur = dict(name=w[1], init={}, threads=[], alias=[], exists=None, expect=None, desc='')
            tests.append(cur)
        elif w[0] == 'init':
            for kv in w[1:]:
                k, v = kv.split('=')
                cur['init'][k] = int(v)
        elif w[0] == 'alias':
            cur['alias'].append((w[1], w[2]))
        elif w[0] == 'thread':
            at = dict(kv.split('=') for kv in w[2:])
            thread = dict(name=w[1], cta=int(at.get('cta', 0)), cluster=int(at.get('cluster', 0)), gpu=int(at.get('gpu', 0)), ins=[])
            cur['threads'].append(thread)
        elif w[0] == 'exists':
            cur['exists'] = line.strip()[len('exists'):].strip()
        elif w[0] == 'expect':
            cur['expect'] = w[1]
        elif w[0] == 'desc':
            cur['desc'] = line.strip()[4:].strip()
        else:
            thread['ins'].append(line.strip())
    return tests


def parse_ins(s):
    cond = None
    m = re.match(r'\?\s*(\w+)\s*==\s*(-?\d+)\s*:\s*(.*)', s)
    if m:
        cond = (m.group(1), int(m.group(2)))
        s = m.group(3)
    proxy = 'generic'
    m = re.search(r'@proxy=(\S+)', s)
    if m:
        proxy = m.group(1)
        s = s.replace(m.group(0), '').strip()
    dst = None
    m = re.match(r'(\w+)\s*=\s*(.*)', s)
    if m:
        dst, s = m.group(1), m.group(2)
    w = s.split()
    parts = w[0].split('.')
    op = parts[0]
    d = dict(op=op, dst=dst, cond=cond, proxy=proxy, sem='weak', scope=None, aop=None, args=w[1:], parts=parts[1:])
    if op in ('ld', 'st', 'atom', 'red'):
        for p in parts[1:]:
            if p in ('weak', 'relaxed', 'acquire', 'release', 'acq_rel', 'volatile', 'mmio'):
                d['sem'] = p
            elif p in SCOPES:
                d['scope'] = p
            else:
                d['aop'] = p
        if d['sem'] in STRONG and d['scope'] is None:
            d['scope'] = 'sys'
        if d['sem'] == 'volatile':
            d['sem'], d['scope'] = 'relaxed', 'sys'
        if op in ('atom', 'red') and d['sem'] == 'weak':
            d['sem'] = 'relaxed'
            d['scope'] = d['scope'] or 'gpu'
    elif op == 'fence':
        d['fkind'] = parts[1]
        d['scope'] = parts[2] if len(parts) > 2 and parts[2] in SCOPES else 'sys'
        if parts[1] == 'proxy':
            d['pkind'] = parts[2] if len(parts) > 2 else 'alias'
            d['scope'] = 'sys'
    elif op == 'bar':
        d['bkind'] = parts[1]
    return d


def val(x, regs):
    if re.fullmatch(r'-?\d+', x):
        return int(x)
    return regs.get(x)


def rmw_new(aop, old, a, b):
    if aop == 'add': return old + a
    if aop == 'exch': return a
    if aop == 'cas': return b if old == a else old
    if aop == 'and': return old & a
    if aop == 'or': return old | a
    if aop == 'inc': return 0 if old >= a else old + 1
    if aop == 'min': return min(old, a)
    if aop == 'max': return max(old, a)
    raise ValueError(aop)


def build(test):
    evs = []
    tinfo = []
    alias = {}
    for a, b in test['alias']:
        alias[b] = a
    for ti, th in enumerate(test['threads']):
        tinfo.append(th)
        for ii, s in enumerate(th['ins']):
            d = parse_ins(s)
            op = d['op']
            e = Ev(tid=ti, idx=len(evs), pidx=ii, **d)
            e.loc = None
            e.addr = None
            if op in ('ld', 'st', 'atom', 'red'):
                e.addr = d['args'][0]
                e.loc = alias.get(e.addr, e.addr)
                if e.addr in alias or e.addr in alias.values():
                    if e.proxy == 'generic' and any(e.addr in p for p in test['alias']):
                        e.proxy = 'alias:' + e.addr
                e.is_read = op in ('ld', 'atom', 'red')
                e.is_write = op in ('st', 'atom', 'red')
                e.is_mem = True
                e.is_atomic = op in ('atom', 'red')
                e.is_red = op == 'red'
                e.read_role = op in ('ld', 'atom')
                e.strong = d['sem'] != 'weak'
            else:
                e.is_read = e.is_write = e.is_mem = e.is_atomic = e.is_red = e.read_role = False
                e.strong = op == 'fence'
            e.fsc = op == 'fence' and d.get('fkind') == 'sc'
            e.fence = op == 'fence' and d.get('fkind') in ('sc', 'acq_rel', 'release', 'acquire')
            e.rel_fence = op == 'fence' and d.get('fkind') in ('sc', 'acq_rel', 'release')
            e.acq_fence = op == 'fence' and d.get('fkind') in ('sc', 'acq_rel', 'acquire')
            e.pfence = op == 'fence' and d.get('fkind') == 'proxy'
            e.is_release = e.is_mem and d['sem'] in ('release', 'acq_rel')
            e.is_acquire = e.is_mem and d['sem'] in ('acquire', 'acq_rel')
            evs.append(e)
    return evs, tinfo, alias


def includes(scope, t1, t2, tinfo):
    a, b = tinfo[t1], tinfo[t2]
    if scope == 'sys': return True
    if scope == 'gpu': return a['gpu'] == b['gpu']
    if scope == 'cluster': return a['gpu'] == b['gpu'] and a['cluster'] == b['cluster']
    return a['gpu'] == b['gpu'] and a['cluster'] == b['cluster'] and a['cta'] == b['cta']


def closure(n, edges):
    reach = [set() for _ in range(n)]
    for a, b in edges:
        reach[a].add(b)
    changed = True
    while changed:
        changed = False
        for a in range(n):
            new = set(reach[a])
            for b in list(reach[a]):
                new |= reach[b]
            if new != reach[a]:
                reach[a] = new
                changed = True
    return reach


def acyclic(n, edges):
    reach = closure(n, edges)
    return all(i not in reach[i] for i in range(n))


PROXY_FENCE_COVERS = {
    'alias': lambda p, q: p != q,
    'async': lambda p, q: {p, q} <= {'generic', 'async'},
    'tensormap': lambda p, q: {p, q} <= {'generic', 'tensormap'},
    'texture': lambda p, q: {p, q} <= {'generic', 'texture'},
    'surface': lambda p, q: {p, q} <= {'generic', 'surface'},
}


def evaluate(evs, rf):
    src = {e.dst: e for e in evs if e.dst}
    value, exe, env = {}, {e.idx: None for e in evs}, {}
    progress = True
    while progress:
        progress = False
        for e in evs:
            if exe[e.idx] is not None:
                continue
            if e.cond is not None:
                se = src.get(e.cond[0])
                if se is None or exe[se.idx] is False:
                    exe[e.idx] = False; progress = True; continue
                if exe[se.idx] is None:
                    continue
                if env[e.cond[0]] != e.cond[1]:
                    exe[e.idx] = False; progress = True; continue
            if e.op in ('ld', 'atom', 'red'):
                w = rf[e.idx]
                if w.tid == -1:
                    wv = w.value
                else:
                    if exe[w.idx] is False:
                        return None
                    if exe[w.idx] is None:
                        continue
                    wv = value[w.idx]
                if e.op == 'ld':
                    value[e.idx] = wv; env[e.dst] = wv
                else:
                    a = val(e.args[1], env)
                    b = val(e.args[2], env) if len(e.args) > 2 else 0
                    if a is None or b is None:
                        continue
                    value[e.idx] = rmw_new(e.aop, wv, a, b)
                    if e.dst: env[e.dst] = wv
            elif e.op == 'st':
                v = val(e.args[1], env)
                if v is None:
                    continue
                value[e.idx] = v
            exe[e.idx] = True; progress = True
    if any(exe[e.idx] is None for e in evs):
        return None
    return value, exe, env


def enumerate_executions(test):
    evs, tinfo, alias = build(test)
    n = len(evs)
    reads = [e for e in evs if e.read_role or e.is_red]
    writes = [e for e in evs if e.is_write]
    locs = sorted({e.loc for e in evs if e.loc})
    # init writes: pseudo events with idx n + k
    init = {l: Ev(idx=n + k, tid=-1, kind='init', loc=l, addr=l, value=test['init'].get(l, 0)) for k, l in enumerate(locs)}
    for l, e in init.items():
        e.is_write = True; e.is_mem = True; e.is_read = False; e.read_role = False; e.is_atomic = False; e.is_red = False
        e.strong = True; e.proxy = 'generic'; e.sem = 'init'; e.scope = 'sys'; e.fence = False; e.fsc = False
        e.is_release = e.is_acquire = False
    N = n + len(locs)
    allev = evs + [init[l] for l in locs]
    po = [(a.idx, b.idx) for a in evs for b in evs if a.tid == b.tid and a.pidx < b.pidx]
    po_set = set(po)
    fsc_events = [e.idx for e in evs if e.fsc]
    bars = [e for e in evs if e.op == 'bar']

    def ms(a, b):
        """morally strong"""
        if a.tid == -1 or b.tid == -1:
            return a is b or (a.tid == -1 and b.tid == -1) or False
        same_thread = a.tid == b.tid
        if not same_thread:
            if not (a.strong and b.strong): return False
            if not (a.scope and b.scope): return False
            if not (includes(a.scope, a.tid, b.tid, tinfo) and includes(b.scope, b.tid, a.tid, tinfo)): return False
        if a.proxy != b.proxy: return False
        if a.is_mem and b.is_mem and a.addr != b.addr: return False
        return True

    for rf_choice in itertools.product(*[[init[r.loc]] + [w for w in writes if w.loc == r.loc and w is not r] for r in reads]):
        rf = {r.idx: w for r, w in zip(reads, rf_choice)}
        # values by fixpoint; a cycle through data or control dependencies stays unresolved (thin air) and is rejected
        res = evaluate(evs, rf)
        if res is None:
            continue
        value, exe, env = res
        live = [e for e in evs if exe[e.idx] is not False]
        livei = {e.idx for e in live}
        if any((r.idx in livei) and (rf[r.idx].tid != -1 and rf[r.idx].idx not in livei) for r in reads):
            continue
        lw = [w for w in writes if w.idx in livei]
        lr = [r for r in reads if r.idx in livei]
        # write orders per location (init first)
        per_loc = {l: [w for w in lw if w.loc == l] for l in locs}
        for co_choice in itertools.product(*[itertools.permutations(per_loc[l]) for l in locs]):
            order = {}
            for l, perm in zip(locs, co_choice):
                seq = [init[l]] + list(perm)
                for i, w in enumerate(seq):
                    order[w.idx] = (l, i)
            for fsc in itertools.permutations([i for i in fsc_events if i in livei]):
                res = check(evs, init, tinfo, alias, live, livei, lw, lr, rf, order, fsc, ms, po, value, env, locs, bars, N, allev)
                if res is not None:
                    yield res


def check(evs, init, tinfo, alias, live, livei, lw, lr, rf, order, fsc, ms, po, value, env, locs, bars, N, allev):
    byidx = {e.idx: e for e in allev}
    liveo = [e for e in live]
    # --- synchronizes-with
    sw = set()
    fpos = {i: k for k, i in enumerate(fsc)}
    for a in fsc:
        for b in fsc:
            if fpos[a] < fpos[b] and ms(byidx[a], byidx[b]):
                sw.add((a, b))
    for x in bars:
        for y in bars:
            if x.idx in livei and y.idx in livei and x.tid != y.tid and x.args == y.args and tinfo[x.tid]['cta'] == tinfo[y.tid]['cta'] \
               and x.bkind in ('sync', 'red', 'arrive') and y.bkind in ('sync', 'red'):
                sw.add((x.idx, y.idx))
    # observation order
    def obs_base(w, r):   # r reads from w, morally strong
        return rf.get(r.idx) is w and r.read_role and ms(r, w)
    obs = set()
    for r in lr:
        if r.read_role and ms(r, rf[r.idx]) and rf[r.idx].idx < len(evs) + len(locs):
            obs.add((rf[r.idx].idx, r.idx))
    # close through atomic reads
    changed = True
    while changed:
        changed = False
        for (w, z) in list(obs):
            ze = byidx[z]
            if ze.is_atomic and not ze.is_red:
                for (z2, r) in list(obs):
                    if z2 == z and (w, r) not in obs:
                        obs.add((w, r)); changed = True
    # patterns
    thr = defaultdict(list)
    for e in liveo:
        thr[e.tid].append(e)
    relp, acqp = [], []     # (first, write) / (read, last)
    for t, es in thr.items():
        es.sort(key=lambda e: e.pidx)
        for i, x in enumerate(es):
            if x.is_release and x.is_write:
                relp.append((x.idx, x.idx))
            if x.is_release and x.is_write:
                for y in es[i + 1:]:
                    if y.is_write and y.strong and y.loc == x.loc:
                        relp.append((x.idx, y.idx))
            if x.rel_fence:
                for y in es[i + 1:]:
                    if y.is_write and y.strong:
                        relp.append((x.idx, y.idx))
            if x.is_acquire and x.read_role:
                acqp.append((x.idx, x.idx))
            if x.read_role and x.strong and not x.is_red:
                for y in es[i + 1:]:
                    if y.is_acquire and y.loc == x.loc and y.read_role:
                        acqp.append((x.idx, y.idx))
                    if y.acq_fence:
                        acqp.append((x.idx, y.idx))
    for (f, w) in relp:
        for (r, l) in acqp:
            if byidx[f].tid != byidx[l].tid and (w, r) in obs and ms(byidx[f], byidx[l]):
                sw.add((f, l))
    # base causality order
    n = len(allev)
    bco = closure(n, list(po) + list(sw))
    def mem_pair_ok(x, y):
        if not (x.is_mem and y.is_mem): return True
        if x.loc != y.loc: return False
        if x.proxy == 'generic' and y.proxy == 'generic': return True
        if x.proxy == y.proxy and tinfo[x.tid]['cta'] == tinfo[y.tid]['cta'] and tinfo[x.tid]['gpu'] == tinfo[y.tid]['gpu']: return True
        # different proxies (aliases, async, ...): a proxy fence along the base causality path
        for f in liveo:
            if f.pfence and (x.idx == f.idx or f.idx in bco[x.idx]) and (y.idx in bco[f.idx]):
                cov = PROXY_FENCE_COVERS.get(f.pkind, PROXY_FENCE_COVERS['alias'])
                px, py = x.proxy, y.proxy
                if px.startswith('alias:') or py.startswith('alias:'):
                    if f.pkind == 'alias': return True
                elif cov(px, py):
                    return True
        return False
    ppbco = set()
    for x in liveo:
        for y in liveo:
            if y.idx in bco[x.idx] and x is not y and mem_pair_ok(x, y):
                ppbco.add((x.idx, y.idx))
    cause = set(ppbco)
    for (x, z) in obs:
        for (z2, y) in ppbco:
            if z2 == z:
                cause.add((x, y))
    # init writes causally precede everything on their location
    # --- coherence order: chosen total order restricted to morally strong or causality-related pairs, closed transitively
    base = set()
    for l in locs:
        ws = sorted([w for w in lw if w.loc == l] + [init[l]], key=lambda w: order[w.idx][1])
        for i, a in enumerate(ws):
            for b in ws[i + 1:]:
                if a.tid == -1 or ms(a, b) or (a.idx, b.idx) in cause:
                    base.add((a.idx, b.idx))
    co = closure(n, list(base))
    def co_lt(a, b): return b.idx in co[a.idx]
    # Coherence
    for x in lw:
        for y in lw:
            if x is not y and x.loc == y.loc and (x.idx, y.idx) in cause and order[x.idx][1] > order[y.idx][1]:
                return None
    # Fence-SC
    for a in fsc:
        for b in fsc:
            if a != b and ms(byidx[a], byidx[b]) and (a, b) in cause and fpos_lt(fsc, b, a):
                return None
    # Atomicity of RMW
    for a in lr + [e for e in lw if e.is_red]:
        if not a.is_atomic: continue
        w0 = rf[a.idx]
        for w in lw:
            if w is not a and w.loc == a.loc and ms(a, w) and co_lt(w0, w) and co_lt(w, a):
                return None
    # Causality
    for r in lr:
        if not r.read_role: continue
        w0 = rf[r.idx]
        for w in lw:
            if w.loc != r.loc or w is r: continue
            if (r.idx, w.idx) in cause and w0 is w:
                return None
            if (w.idx, r.idx) in cause and w0 is not w and co_lt(w0, w):
                return None
    # SC per location
    edges = []
    for l in locs:
        es = [e for e in liveo if e.is_mem and e.loc == l]
        for a in es:
            for b in es:
                if a.tid == b.tid and a.pidx < b.pidx:
                    edges.append((a.idx, b.idx))
        for r in lr:
            if r.loc == l and r.read_role:
                w0 = rf[r.idx]
                if ms(w0, r) or w0.tid == -1:
                    edges.append((w0.idx, r.idx))
                for w in lw:
                    if w.loc == l and w is not r and co_lt(w0, w) and ms(r, w):
                        edges.append((r.idx, w.idx))
        for a in lw:
            for b in lw:
                if a is not b and a.loc == l and b.loc == l and co_lt(a, b) and ms(a, b):
                    edges.append((a.idx, b.idx))
    if not acyclic(n, edges):
        return None
    final = {}
    for l in locs:
        ws = [w for w in lw if w.loc == l]
        last = max(ws, key=lambda w: order[w.idx][1]) if ws else init[l]
        final[l] = last.value if last.tid == -1 else value[last.idx]
    outcome = dict(env)
    outcome.update({'[' + k + ']': v for k, v in final.items()})
    # data races: conflicting operations (same location, one a write) that are neither morally strong nor causality-ordered
    race = False
    mems = [e for e in liveo if e.is_mem]
    for i, a in enumerate(mems):
        for b in mems[i + 1:]:
            if a.tid != b.tid and a.loc == b.loc and (a.is_write or b.is_write) and not ms(a, b) \
               and (a.idx, b.idx) not in cause and (b.idx, a.idx) not in cause:
                race = True
    outcome['#race'] = int(race)
    return tuple(sorted(outcome.items()))


def fpos_lt(fsc, a, b):
    return fsc.index(a) < fsc.index(b)


def holds(cond, outcome):
    d = dict(outcome)
    expr = cond
    for name in sorted(d, key=len, reverse=True):
        expr = re.sub(r'(?<![\w\[])' + re.escape(name) + r'(?![\w\]])', str(d[name]), expr)
    expr = re.sub(r'\[(\w+)\]', lambda m: str(d.get('[' + m.group(1) + ']', 0)), expr)
    expr = expr.replace('&&', ' and ').replace('||', ' or ')
    return bool(eval(expr))


def run(test):
    outs = set(enumerate_executions(test))
    return outs


def main(argv):
    files = argv[1:] or sorted(os.path.join('tests/memmodel', f) for f in os.listdir('tests/memmodel') if f.endswith('.litmus'))
    bad = 0
    for f in files:
        for t in parse(open(f).read()):
            outs = run(t)
            hit = any(holds(t['exists'], o) for o in outs) if t['exists'] else None
            verdict = 'allowed' if hit else 'forbidden'
            ok = (t['expect'] is None) or (t['expect'] == verdict)
            racy = any(dict(o).get('#race') for o in outs)
            print(f"{'PASS' if ok else 'FAIL'}  {t['name']:28s} {verdict:9s} {'racy' if racy else 'race-free':9s} ({len(outs)} outcomes)")
            if not ok:
                bad += 1
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

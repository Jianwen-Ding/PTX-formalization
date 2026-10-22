#!/usr/bin/env python3
"""Cross-check scripts/ptxmm.py against NVIDIA's published formal PTX memory models (both BSD-3, Alloy).

  ASPLOS'19   github.com/NVlabs/ptxmemorymodel  Lustig, Sahasrabuddhe, Giroux, "A Formal Analysis of the NVIDIA PTX
              Memory Consistency Model": ptx.als (scopes as a free scope tree, weak/relaxed/acquire/release accesses,
              RMWs, fence.sc/acq_rel/acquire/release, bar.sync / bar.arrive).  No values, no proxies.
  ISCA'22     github.com/NVlabs/mixedproxy  Lustig, Cooksey, Giroux, "Mixed-Proxy Extensions for the NVIDIA PTX Memory
              Consistency Model": alloy/ptx.als plus the "nvlitmus" front end (src/test_to_alloy.py, .test format,
              tests/*.test).  Values, virtual aliases, surface / texture / constant proxies, proxy and alias fences;
              scopes cta / gpu / sys only, atom.add only, no barriers, no control dependencies.

Setup (Java and a Python with `lark-parser`, the front end's parser, are needed; nothing else is built):
    mkdir -p ext && cd ext
    git clone https://github.com/NVlabs/ptxmemorymodel && git clone https://github.com/NVlabs/mixedproxy
    (cd ptxmemorymodel && javac -cp alloy4.2.jar RunAlloy.java)
    (cd mixedproxy/alloy && wget https://github.com/AlloyTools/org.alloytools.alloy/releases/download/v5.1.0/org.alloytools.alloy.dist.jar)
    (cd mixedproxy && make)                       # builds alloy/RunAlloy.class
    python3 -m venv venv && venv/bin/pip install lark-parser
Run (one Alloy instance at a time, java -Xmx$NVMM_XMX, default 4g):
    NVMM_EXT=ext ext/venv/bin/python scripts/nvmm.py            # everything, summary table
    NVMM_EXT=ext ext/venv/bin/python scripts/nvmm.py repo        # tests/memmodel/*.litmus -> both models
    NVMM_EXT=ext ext/venv/bin/python scripts/nvmm.py mixedproxy  # mixedproxy tests/*.test -> ptxmm
    NVMM_EXT=ext ext/venv/bin/python scripts/nvmm.py extra       # tests/memmodel/nvidia/{asplos2019,differential}.litmus
    NVMM_EXT=ext ext/venv/bin/python scripts/nvmm.py FILE.litmus # any litmus file -> both models
    NVMM_EXT=ext ext/venv/bin/python scripts/nvmm.py --write     # regenerate tests/memmodel/nvidia/mixedproxy.litmus
    ... --mp-fence-fix   run mixedproxy with fences counted as morally strong (see mp_model; one-line model change)
    scripts/nvmm.py --emit-mp FILE.litmus | --emit-asplos FILE.litmus   # print the translations only

Translation, repo -> mixedproxy: the litmus test becomes a .test file (dD.bB.tT threads, `.global`, `.texref` /
`.surfref` for @proxy=texture / surface, `physically aliases` for `alias`), the nvlitmus front end turns it into
Alloy, and the exists clause is appended as `run outcome { ptx_mm and ... }` (registers are `rN.value`, a final value
[x]==v is "some coherence-maximal write of v to x", as ptxmm takes the last write of a linear extension).  Data
dependencies are added as `(R -> W) in dep` (the front end leaves `dep` free, so No-Thin-Air would never fire).
`red` is sent as an atom (the front end's `red` rule is broken; the model has no reductions).
Repo -> ASPLOS: the model has no values, so the outcome is turned into read-from / coherence constraints: every
assignment of values to the reads (from the constants, initial values, RMW results and an out-of-thin-air value 42)
that is consistent with some read-from choice and satisfies `exists` gives one disjunct; conditional instructions
split the test into one Alloy module per set of executed instructions.  dep = data + control dependencies + rmw.
mixedproxy -> repo: each template instance and each permit/assert command becomes one litmus test whose `exists` is
the conjunction of the `== v` load filters and the condition (negated for assert); the NVIDIA verdict is taken from
the unmodified front end on the original instance.
"""
import itertools, os, re, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptxmm

EXT = os.environ.get('NVMM_EXT', 'ext')
ASP = os.path.join(EXT, 'ptxmemorymodel')
MPX = os.path.join(EXT, 'mixedproxy')
XMX = '-Xmx' + os.environ.get('NVMM_XMX', '4g')
OOTA = 42
MP_MODEL = os.environ.get('NVMM_MP_MODEL')   # an alternative mixedproxy ptx.als, for experiments
MP_FENCE_FIX = False


class Untranslatable(Exception):
    pass


# ---------------------------------------------------------------- conditions (exists clauses, permit/assert)

def cond_parse(s):
    toks = re.findall(r'\[\w+\]|-?\d+|\w+|==|!=|&&|\|\||\(|\)', s)
    pos = [0]
    def peek(): return toks[pos[0]] if pos[0] < len(toks) else None
    def take(): pos[0] += 1; return toks[pos[0] - 1]
    def term():
        x = take()
        if re.fullmatch(r'-?\d+', x): return ('int', int(x))
        if x.startswith('['): return ('mem', x[1:-1])
        return ('reg', x)
    def unary():
        if peek() == 'not': take(); return ('not', unary())
        if peek() == '(':
            take(); e = disj(); assert take() == ')'; return e
        a = term(); op = take(); b = term()
        assert op in ('==', '!='), s
        return ('cmp', op, a, b)
    def conj():
        e = unary()
        while peek() == '&&': take(); e = ('and', e, unary())
        return e
    def disj():
        e = conj()
        while peek() == '||': take(); e = ('or', e, conj())
        return e
    e = disj()
    assert pos[0] == len(toks), s
    return e


def cond_neg(e):
    k = e[0]
    if k == 'and': return ('or', cond_neg(e[1]), cond_neg(e[2]))
    if k == 'or': return ('and', cond_neg(e[1]), cond_neg(e[2]))
    if k == 'not': return e[1]
    return ('cmp', '!=' if e[1] == '==' else '==', e[2], e[3])


def cond_str(e, regmap=None, top=True):
    k = e[0]
    if k in ('and', 'or'):
        s = f"{cond_str(e[1], regmap, False)} {'&&' if k == 'and' else '||'} {cond_str(e[2], regmap, False)}"
        return s if top or k == 'and' else f'({s})'
    if k == 'not':
        return cond_str(cond_neg(e[1]), regmap, top)
    def t(x):
        if x[0] == 'int': return str(x[1])
        if x[0] == 'mem': return f'[{x[1]}]'
        return (regmap or {}).get(x[1], x[1])
    return f'{t(e[2])}{e[1]}{t(e[3])}'


def cond_terms(e, kind):
    if e[0] in ('and', 'or'): return cond_terms(e[1], kind) | cond_terms(e[2], kind)
    if e[0] == 'not': return cond_terms(e[1], kind)
    return {x[1] for x in e[2:] if x[0] == kind}


# ---------------------------------------------------------------- running Alloy

def java(cp, args, stdin=None, timeout=1800):
    r = subprocess.run(['java', XMX, '-cp', cp] + args, input=stdin, capture_output=True, text=True, timeout=timeout)
    return r.stdout, r.stderr


def mp_run_alloy(text):
    """Run an Alloy text with mixedproxy's RunAlloy; return {command label: SAT?}."""
    cp = os.path.join(MPX, 'alloy') + ':' + os.path.join(MPX, 'alloy', 'org.alloytools.alloy.dist.jar')
    out, err = java(cp, ['RunAlloy'], stdin=text)
    if 'Exception' in err or 'Error' in err.replace('Relevance', ''):
        raise RuntimeError(err[-2000:])
    res = {}
    for ln in out.split('\n'):
        m = re.match(r'(\w+): (SAT|UNSAT)', ln)
        if m: res[m.group(1)] = m.group(2) == 'SAT'
    return res


def asp_run(files, cmd):
    """Run command `cmd` of an Alloy module with the ASPLOS RunAlloy (Alloy 4.2); True if an instance exists."""
    cp = ASP + ':' + os.path.join(ASP, 'alloy4.2.jar')
    out, err = java(cp, ['RunAlloy', '-f', files, '-n', '1', cmd])
    if '<command' not in out:
        raise RuntimeError(err[-2000:])
    return '<instance' in out


# ---------------------------------------------------------------- repo litmus -> mixedproxy

def _inst(t):
    alias = {b: a for a, b in t['alias']}
    out = []
    for ti, th in enumerate(t['threads']):
        for k, s in enumerate(th['ins']):
            d = ptxmm.parse_ins(s)
            d['tid'], d['k'], d['src'] = ti, k, s
            if d['op'] in ('ld', 'st', 'atom', 'red'):
                d['addr'] = d['args'][0]
                d['loc'] = alias.get(d['addr'], d['addr'])
            out.append(d)
    return out


def _isint(x): return re.fullmatch(r'-?\d+', x) is not None


def litmus_to_mp(t):
    """Return (.test text, list of (dep-source instruction index, dep-target index), notes, instruction list)."""
    if any(v != 0 for v in t['init'].values()):
        raise Untranslatable('non-zero initial value (mixedproxy reads 0 from the initial state)')
    ins = _inst(t)
    notes = []
    blocks, tids = {}, []
    for th in t['threads']:
        key = (th['gpu'], th['cluster'], th['cta'])
        bl = blocks.setdefault(th['gpu'], {})
        b = bl.setdefault(key, [len(bl), 0])
        tids.append(f"d{th['gpu']}.b{b[0]}.t{b[1]}"); b[1] += 1
    regs = {}
    def reg(r):
        if r not in regs:
            regs[r] = next(f'r{k}' for k in itertools.count(100) if f'r{k}' not in regs.values())
        return regs[r]
    for r in sorted({d['dst'] for d in ins if d['dst']}):
        if re.fullmatch(r'r\d+', r): regs[r] = r
    decl, refs = [], {}
    locs = []
    for d in ins:
        if 'loc' in d and d['loc'] not in locs: locs.append(d['loc'])
    aliased = {b: a for a, b in t['alias']}
    for l in locs:
        decl.append(f'.global {l};')
    for a, b in t['alias']:
        decl.append(f'.global {b} physically aliases {a};')
    def ref(addr, proxy):
        if proxy == 'generic': return addr
        if proxy not in ('texture', 'surface'):
            raise Untranslatable(f'proxy {proxy} (mixedproxy has generic, surface, texture, constant)')
        name = ('t' if proxy == 'texture' else 's') + '_' + addr
        if name not in refs:
            refs[name] = f".{'texref' if proxy == 'texture' else 'surfref'} {name} virtually aliases {addr};"
        return name
    def ss(d):
        if d['scope'] == 'cluster': raise Untranslatable('.cluster scope')
        if d['sem'] == 'mmio': raise Untranslatable('.mmio')
        return '.weak' if d['sem'] == 'weak' else f".{d['sem']}.{d['scope']}"
    body = {i: [] for i in range(len(t['threads']))}
    deps = []
    defs = {}
    fresh = [0]
    for n, d in enumerate(ins):
        if d['cond'] is not None: raise Untranslatable('control dependency (no conditional instructions in nvlitmus)')
        op = d['op']
        def val(x):
            if _isint(x): return x
            deps.append((defs[x], n)); return reg(x)
        if op == 'ld':
            p = d['proxy']
            o = {'generic': 'ld', 'texture': 'tld', 'surface': 'suld'}.get(p)
            if o is None: raise Untranslatable(f'proxy {p}')
            if p == 'texture' and d['sem'] != 'weak': raise Untranslatable('strong texture load')
            s = '' if p == 'texture' else ss(d)
            body[d['tid']].append(f"{o}{s} {reg(d['dst'])}, [{ref(d['addr'], p)}];")
            defs[d['dst']] = n
        elif op == 'st':
            p = d['proxy']
            o = {'generic': 'st', 'surface': 'sust'}.get(p)
            if o is None: raise Untranslatable(f'store via proxy {p}')
            body[d['tid']].append(f"{o}{ss(d)} [{ref(d['addr'], p)}], {val(d['args'][1])};")
        elif op in ('atom', 'red'):
            if d['aop'] != 'add' or len(d['args']) != 2: raise Untranslatable(f"atom.{d['aop']} (nvlitmus: add only)")
            p = d['proxy']
            o = {'generic': 'atom', 'surface': 'suatom'}.get(p)
            if o is None: raise Untranslatable(f'atom via proxy {p}')
            dst = d['dst']
            if dst is None:
                dst = f'_red{fresh[0]}'; fresh[0] += 1
                if op == 'red': notes.append('red sent as atom')
            body[d['tid']].append(f"{o}.add{ss(d)} {reg(dst)}, [{ref(d['addr'], p)}], {val(d['args'][1])};")
            defs[dst] = n
        elif op == 'fence':
            if d['fkind'] == 'proxy':
                if d['pkind'] not in ('alias', 'surface', 'texture', 'constant'):
                    raise Untranslatable(f"fence.proxy.{d['pkind']}")
                body[d['tid']].append(f"fence.proxy.{d['pkind']};")
            elif d['fkind'] in ('sc', 'acq_rel'):
                if d['scope'] == 'cluster': raise Untranslatable('.cluster scope')
                body[d['tid']].append(f"fence.{d['fkind']}.{d['scope']};")
            else:
                raise Untranslatable(f"fence.{d['fkind']} (nvlitmus: fence.sc / fence.acq_rel)")
        else:
            raise Untranslatable(f'{op} (no barriers in mixedproxy)')
    text = '\n'.join(decl + list(refs.values())) + '\n\n'
    for i, tid in enumerate(tids):
        text += tid + ' {\n' + ''.join(f'  {x}\n' for x in body[i]) + '}\n\n'
    return text, deps, notes, ins, regs


MP_STRONG = '  symmetric[scope.~scopes] & proxy.~proxy\n'
MP_STRONG_FIXED = '  symmetric[scope.~scopes] & (proxy.~proxy + (Op - MemoryOp)->Op + Op->(Op - MemoryOp))\n'


def mp_model():
    """The mixedproxy Alloy model.  With --mp-fence-fix, fences (which have no `proxy` field, so `strong` never
    relates them to anything and fence.acq_rel never forms a release / acquire pattern) count as morally strong."""
    model = open(MP_MODEL or os.path.join(MPX, 'alloy', 'ptx.als')).read()
    if MP_FENCE_FIX:
        assert MP_STRONG in model
        model = model.replace(MP_STRONG, MP_STRONG_FIXED)
    return model


def mp_parser():
    sys.path.insert(0, os.path.join(MPX, 'src'))
    import litmus_parser
    return litmus_parser


def mp_alloy_expr(e, regs):
    k = e[0]
    if k in ('and', 'or'):
        return f"({mp_alloy_expr(e[1], regs)} {k} {mp_alloy_expr(e[2], regs)})"
    if k == 'not':
        return f"(not {mp_alloy_expr(e[1], regs)})"
    op, a, b = e[1:]
    if a[0] == 'int' and b[0] != 'int': a, b = b, a
    if a[0] == 'mem':
        assert b[0] == 'int'
        ws = f"(Write & address.(aliases.{a[1]}))"
        cmp = '=' if op == '==' else '!='
        init = f" or (no {ws} and 0 {cmp} {b[1]})"
        return f"((some w: {ws} | no w.co and w.value {cmp} {b[1]}){init})"
    def t(x): return str(x[1]) if x[0] == 'int' else f"{regs[x[1]]}.value"
    return f"({t(a)} {'=' if op == '==' else '!='} {t(b)})"


def run_mp(t):
    """Verdict of the mixedproxy model for repo litmus test t: (allowed?, notes)."""
    text, deps, notes, ins, regs = litmus_to_mp(t)
    lp = mp_parser()
    model = mp_model()
    test = lp.parse(model, text)
    names = [i.name for th in test.threads.values() for i in th.insts]
    assert len(names) == len(ins)
    def rd(n): return names[n] + ('_r' if ins[n]['op'] in ('atom', 'red') else '')
    def wr(n): return names[n] + ('_w' if ins[n]['op'] in ('atom', 'red') else '')
    alloy = test.to_alloy()
    if deps:
        alloy += 'fact { ' + ' + '.join(f'({rd(a)} -> {wr(b)})' for a, b in deps) + ' in dep }\n'
    e = cond_parse(t['exists'])
    alloy += f"run outcome {{ ptx_mm and {mp_alloy_expr(e, regs)} }} for 1 but 6 Int\n"
    res = mp_run_alloy(alloy)
    if res.get('sanity') is False:
        raise Untranslatable('sanity check failed (ill-formed translation)')
    return res['outcome'], notes


# ---------------------------------------------------------------- repo litmus -> ASPLOS'19 (no values)

def rf_candidates(t, ins):
    """Group the value assignments satisfying `exists` by the set of executed instructions.
    Returns {frozenset(executed instruction indices): set of (rf constraints, final constraints)}."""
    reads = [n for n, d in enumerate(ins) if d['op'] in ('ld', 'atom', 'red')]
    writes = [n for n, d in enumerate(ins) if d['op'] in ('st', 'atom', 'red')]
    locs = sorted({d['loc'] for d in ins if 'loc' in d})
    init = {l: t['init'].get(l, 0) for l in locs}
    ex = cond_parse(t['exists'])
    D = set(init.values()) | {0, OOTA} | {x[1] for x in [('int', v) for v in cond_terms(ex, 'int')]}
    for n in writes:
        D |= {int(a) for a in ins[n]['args'][1:] if _isint(a)}
    for _ in range(3):
        for n in writes:
            d = ins[n]
            if d['op'] == 'st' or len(D) > 10: continue
            for old in list(D):
                args = [int(a) if _isint(a) else old for a in d['args'][1:]] + [0]
                D.add(ptxmm.rmw_new(d['aop'], old, args[0], args[1]))
    D = sorted(D)
    out = {}
    for assign in itertools.product(D, repeat=len(reads)):
        rv = dict(zip(reads, assign))
        env, exe, wv, ok = {}, set(), {}, True
        for n, d in enumerate(ins):
            if d['cond'] is not None:
                r, v = d['cond']
                if env.get(r) != v: continue
            exe.add(n)
            if n in rv and d['dst']: env[d['dst']] = rv[n]
            if n in writes:
                a = [int(x) if _isint(x) else env.get(x) for x in d['args'][1:]]
                if None in a: ok = False; break
                wv[n] = a[0] if d['op'] == 'st' else ptxmm.rmw_new(d['aop'], rv[n], a[0], a[1] if len(a) > 1 else 0)
        if not ok or any(n not in exe and rv[n] != D[0] for n in reads):
            continue
        rfc = []
        for n in reads:
            if n not in exe: continue
            src = tuple(w for w in writes if w in exe and w != n and ins[w]['loc'] == ins[n]['loc'] and wv[w] == rv[n])
            ini = init[ins[n]['loc']] == rv[n]
            if not src and not ini: ok = False; break
            rfc.append((n, src, ini))
        if not ok: continue
        floc = sorted(cond_terms(ex, 'mem'))
        fch = []
        for l in floc:
            ws = [w for w in writes if w in exe and ins[w]['loc'] == l]
            fch.append(sorted({wv[w] for w in ws}) if ws else [init.get(l, 0)])
        for fv in itertools.product(*fch):
            o = dict(env)
            o.update({f'[{l}]': v for l, v in zip(floc, fv)})
            try:
                hit = ptxmm.holds(t['exists'], tuple(o.items()))
            except Exception:
                hit = False
            if not hit: continue
            fin = []
            for l, v in zip(floc, fv):
                ws = tuple(w for w in writes if w in exe and ins[w]['loc'] == l and wv[w] == v)
                if ws: fin.append(ws)
            out.setdefault(frozenset(exe), set()).add((tuple(rfc), tuple(fin)))
    return out


def litmus_to_asplos(t):
    """Return a list of (module name, Alloy text): the outcome is allowed if `outcome` is SAT in one of them."""
    if t['alias']: raise Untranslatable('virtual aliases (no proxies in the ASPLOS model)')
    ins = _inst(t)
    for d in ins:
        if d.get('proxy', 'generic') != 'generic' or (d['op'] == 'fence' and d['fkind'] == 'proxy'):
            raise Untranslatable('proxies (no proxies in the ASPLOS model)')
    th = t['threads']
    # scope tree
    sc = {'SYS': None}
    tsc = []
    for i, x in enumerate(th):
        g, cl, c = f"G{x['gpu']}", f"CL{x['gpu']}_{x['cluster']}", f"C{x['gpu']}_{x['cluster']}_{x['cta']}"
        sc.setdefault(g, 'SYS'); sc.setdefault(cl, g); sc.setdefault(c, cl)
        tsc.append({'sys': 'SYS', 'gpu': g, 'cluster': cl, 'cta': c})
    cands = rf_candidates(t, ins)
    mods = []
    for exe, cs in sorted(cands.items(), key=lambda kv: sorted(kv[0])):
        ev, kind, scope, addr, po, rmw, dep = {}, {}, {}, {}, [], [], []
        rd, wr = {}, {}
        bars = {}
        for n, d in enumerate(ins):
            if n not in exe: continue
            b = f"e{d['tid']}_{d['k']}"
            T = f"T{d['tid']}"
            s = T if d['sem'] == 'weak' and d['op'] in ('ld', 'st') else tsc[d['tid']].get(d['scope'] or 'sys')
            op = d['op']
            if op == 'ld':
                es = [b]; kind[b] = 'Acquire' if d['sem'] in ('acquire', 'acq_rel') else 'Read'; rd[n] = b
            elif op == 'st':
                es = [b]; kind[b] = 'Release' if d['sem'] in ('release', 'acq_rel') else 'Write'; wr[n] = b
            elif op in ('atom', 'red'):
                es = [b + 'r', b + 'w']
                kind[b + 'r'] = 'Acquire' if d['sem'] in ('acquire', 'acq_rel') else 'Read'
                kind[b + 'w'] = 'Release' if d['sem'] in ('release', 'acq_rel') else 'Write'
                rd[n], wr[n] = b + 'r', b + 'w'
                rmw.append((b + 'r', b + 'w'))
            elif op == 'fence':
                es = [b]; kind[b] = {'sc': 'FenceSC', 'acq_rel': 'FenceAcqRel', 'acquire': 'FenceAcq', 'release': 'FenceRel'}[d['fkind']]
            elif op == 'bar':
                s = tsc[d['tid']]['cta']
                key = (tsc[d['tid']]['cta'], d['args'][0])
                es = [b + 'a'] + ([b + 'w'] if d['bkind'] in ('sync', 'red') else [])
                kind[b + 'a'] = 'BarrierArrive'
                bars.setdefault(key, [[], []])[0].append(b + 'a')
                if len(es) > 1:
                    kind[b + 'w'] = 'BarrierWait'; bars[key][1].append(b + 'w')
            else:
                raise Untranslatable(op)
            for e in es:
                ev.setdefault(T, []).append(e); scope[e] = s
                if 'loc' in d: addr[e] = 'A_' + d['loc']
            d['_ev'] = es
        for n, d in enumerate(ins):
            if n not in exe: continue
            srcs = [x for x in d['args'][1:] if not _isint(x)] if d['op'] in ('st', 'atom', 'red') else []
            for r in srcs:
                m = next(m for m, e in enumerate(ins) if e['dst'] == r)
                dep.append((rd[m], wr[n]))
            if d['cond'] is not None:
                m = next(m for m, e in enumerate(ins) if e['dst'] == d['cond'][0])
                dep += [(rd[m], e) for e in d['_ev']]
        dep += rmw
        sync = [(a, w) for A, W in bars.values() for a in A for w in W]
        events = [e for T in ev for e in ev[T]]
        threads = [f'T{i}' for i in range(len(th))]
        addrs = sorted(set(addr.values()))
        L = ['open ptx', '']
        L.append('one sig ' + ', '.join(sc) + ' extends Scope {}')
        L.append('one sig ' + ', '.join(threads) + ' extends Thread {}')
        L.append('one sig ' + ', '.join(addrs) + ' extends Address {}')
        for e in events: L.append(f'one sig {e} extends {kind[e]} {{}}')
        def rel(pairs): return ' + '.join(f'{a}->{b}' for a, b in pairs) if pairs else 'none->none'
        tree = [(p, c) for c, p in sc.items() if p] + [(tsc[i]['cta'], f'T{i}') for i in range(len(th))]
        L.append(f'fact {{ subscope = {rel(tree)} }}')
        L.append(f"fact {{ Scope = {' + '.join(list(sc) + threads)} }}")
        L.append(f"fact {{ Event = {' + '.join(events)} }}")
        L.append(f"fact {{ Address = {' + '.join(addrs)} }}")
        pos = [(es[i], es[i + 1]) for es in ev.values() for i in range(len(es) - 1)]
        starts = ' and '.join(f'{T}.start = {es[0]}' for T, es in ev.items())
        idle = [T for T in threads if T not in ev]
        if idle: raise Untranslatable('a thread executes no instruction')
        L.append(f'fact {{ po = {rel(pos)} and {starts} }}')
        L.append('fact { ' + ' and '.join(f'{e}.scope = {scope[e]}' for e in events) + ' }')
        if addr: L.append('fact { ' + ' and '.join(f'{e}.address = {a}' for e, a in addr.items()) + ' }')
        L.append(f'fact {{ rmw = {rel(rmw)} and dep = {rel(sorted(set(dep)))} and synchronizes = {rel(sync)} }}')
        disj = []
        for rfc, fin in sorted(cs):
            c = []
            for n, src, ini in rfc:
                r = rd[n]
                ws = ' + '.join(wr[w] for w in src)
                if src and ini: c.append(f'(no {r}.~rf or {r}.~rf in {ws})')
                elif src: c.append(f'(some {r}.~rf and {r}.~rf in {ws})')
                else: c.append(f'no {r}.~rf')
            for ws in fin:
                c.append(f"(some w: {' + '.join(wr[x] for x in ws)} | no w.co)")
            disj.append('(' + (' and '.join(c) or 'some Event') + ')')
        N = max(len(events), len(sc) + len(threads), len(addrs)) + 1
        L.append(f'run sanity {{}} for {N} but 5 Int')
        L.append(f"run outcome {{ ptx_mm and ({' or '.join(disj)}) }} for {N} but 5 Int")
        mods.append(('exe' + ''.join(str(n) for n in sorted(exe)), '\n'.join(L) + '\n'))
    return mods


def run_asplos(t):
    mods = litmus_to_asplos(t)
    if not mods: return False, ['no value assignment satisfies exists']
    notes = []
    d = tempfile.mkdtemp(prefix='nvmm-asplos-')
    for f in ('ptx.als', 'util.als'):
        open(os.path.join(d, f), 'w').write(open(os.path.join(ASP, f)).read())
    hit = False
    for name, text in mods:
        p = os.path.join(d, name + '.als')
        open(p, 'w').write(text)
        if not asp_run(p, 'sanity'):
            notes.append(f'{name}: ill-formed (ASPLOS Address facts need >1 access and a write per address)')
            continue
        if asp_run(p, 'outcome'):
            hit = True; break
    return hit, notes


# ---------------------------------------------------------------- mixedproxy .test -> repo litmus

def mp_instances(path):
    text = open(path).read()
    if '$$' not in text:
        return [text]
    test, params = text.split('$$\n')
    out = []
    for p in [i for i in params.split('\n') if i.strip() and i[0] != '#']:
        inst = test
        for i, x in enumerate(p.split('|')):
            inst = inst.replace(f'${i}', x.strip())
        out.append(inst)
    return out


def mp_to_litmus(text, name):
    """Translate one (expanded) nvlitmus test into repo litmus tests, one per command.  Returns [(test text, cmd)]."""
    text = re.sub(r'//[^\n]*', '', text)
    addr = {}   # name -> (location, address, proxy)
    alias = []
    for m in re.finditer(r'\.(global|shared|texref|surfref)\s+(\w+)(?:\s+(physically|virtually)\s+aliases\s+(\w+))?\s*;', text):
        sp, n, kind, base = m.groups()
        if sp in ('global', 'shared'):
            if kind == 'physically':
                alias.append((base, n)); addr[n] = (n, 'generic')
            else:
                addr[n] = (n, 'generic')
        else:
            addr[n] = (addr[base][0], 'texture' if sp == 'texref' else 'surface')
    threads = []
    filt = []
    for m in re.finditer(r'd(\d+)\.b(\d+)\.t(\d+)\s*\{(.*?)\}', text, re.S):
        d, b, tt, body = m.groups()
        lines = []
        for s in [x.strip() for x in body.split(';') if x.strip()]:
            w = re.match(r'(\w+)((?:\.\w+)*)\s*(.*)', s)
            op, mods, rest = w.group(1), w.group(2).split('.')[1:], w.group(3)
            sem = [x for x in mods if x in ('weak', 'relaxed', 'acquire', 'release', 'acq_rel', 'sc', 'volatile')]
            scope = [x for x in mods if x in ('cta', 'gpu', 'sys')]
            q = '.' + '.'.join(sem + scope) if sem or scope else '.weak'
            ops = [x.strip() for x in rest.split(',')]
            def mem(x):
                a = addr[x.strip()[1:-1]]
                return a
            if op in ('ld', 'tld', 'suld', 'ldc'):
                rv = re.match(r'(\[\w+\])\s*(?:==\s*(-?\d+))?', ops[1])
                loc, px = mem(rv.group(1))
                px = {'tld': 'texture', 'suld': 'surface', 'ldc': 'constant'}.get(op, px)
                lines.append(f"{ops[0]} = ld{q} {loc}" + (f' @proxy={px}' if px != 'generic' else ''))
                if rv.group(2) is not None: filt.append(('cmp', '==', ('reg', ops[0]), ('int', int(rv.group(2)))))
            elif op in ('st', 'sust'):
                loc, px = mem(ops[0])
                px = 'surface' if op == 'sust' else px
                lines.append(f"st{q} {loc} {ops[1]}" + (f' @proxy={px}' if px != 'generic' else ''))
            elif op in ('atom', 'suatom', 'red', 'sured'):
                aop = mods[0]
                q = '.' + '.'.join(sem + scope) if sem or scope else ''
                if op in ('atom', 'suatom'):
                    loc, px = mem(ops[1]); px = 'surface' if op == 'suatom' else px
                    rv = re.match(r'(\w+)\s*(?:==\s*(-?\d+))?', ops[2])
                    lines.append(f"{ops[0]} = atom{q}.{aop} {loc} {rv.group(1)}" + (f' @proxy={px}' if px != 'generic' else ''))
                    if rv.group(2) is not None: filt.append(('cmp', '==', ('reg', ops[0]), ('int', int(rv.group(2)))))
                else:
                    loc, px = mem(ops[0]); px = 'surface' if op == 'sured' else px
                    lines.append(f"red{q}.{aop} {loc} {ops[1]}" + (f' @proxy={px}' if px != 'generic' else ''))
            elif op == 'fence':
                if mods[0] == 'proxy': lines.append(f'fence.proxy.{mods[1]}')
                elif mods[0] == 'alias': lines.append('fence.proxy.alias')
                else: lines.append(f'fence.{mods[0]}.{mods[1]}')
            else:
                raise Untranslatable(op)
        threads.append((d, b, tt, lines))
    cmds = re.findall(r'(permit|assert|check)\s*\((.*?)\)\s*as\s+(\w+)\s*;', text, re.S)
    out = []
    for kind, c, cname in cmds:
        e = cond_parse(c)
        if kind == 'assert': e = cond_neg(e)
        for f in reversed(filt): e = ('and', f, e)
        locs = []
        for _, _, _, ls in threads:
            pass
        L = [f'test {name}:{cname}', f"desc mixedproxy {kind}({' '.join(c.split())})"]
        for a, b in alias: L.append(f'alias {a} {b}')
        names = sorted({v[0] for v in addr.values()} - {b for a, b in alias})
        L.append('init ' + ' '.join(f'{n}=0' for n in names))
        for d, b, tt, ls in threads:
            L.append(f'thread T{d}_{b}_{tt} cta={b} cluster=0 gpu={d}')
            L += ['  ' + x for x in ls]
        L.append('exists ' + cond_str(e))
        L.append('expect ' + ('allowed' if kind != 'assert' else 'forbidden'))
        out.append(('\n'.join(L) + '\n', kind, cname))
    return out


def mp_native_verdicts(inst_text):
    """Run an nvlitmus instance through the unmodified front end; {command: outcome-of-exists allowed?}."""
    lp = mp_parser()
    model = mp_model()
    test = lp.parse(model, inst_text)
    res = mp_run_alloy(test.to_alloy())
    if any(k.startswith('sanity') and v is False for k, v in res.items()):
        raise RuntimeError('sanity failed')
    return {k: v for k, v in res.items() if not k.startswith('sanity')}


# ---------------------------------------------------------------- drivers

def ptxmm_verdict(t):
    outs = ptxmm.run(t)
    return any(ptxmm.holds(t['exists'], o) for o in outs)


def V(b): return 'allowed' if b else 'forbidden'


def check_litmus_files(files, label):
    rows = []
    for f in files:
        for t in ptxmm.parse(open(f).read()):
            p = V(ptxmm_verdict(t))
            row = dict(name=t['name'], file=label, expect=t['expect'], ptxmm=p)
            for m, fn in (('mixedproxy', run_mp), ('asplos', run_asplos)):
                try:
                    v, notes = fn(t)
                    row[m] = V(v); row[m + '_notes'] = notes
                except Untranslatable as u:
                    row[m] = '-'; row[m + '_notes'] = [str(u)]
            flag = [m for m in ('mixedproxy', 'asplos') if row[m] != '-' and row[m] != p]
            if t['expect'] and t['expect'] != p: flag.append('expect')
            row['flag'] = flag
            print(f"{'DIFF' if flag else 'same'}  {t['name']:34s} expect={t['expect'] or '-':9s} ptxmm={p:9s} "
                  f"mixedproxy={row['mixedproxy']:9s} asplos={row['asplos']:9s} "
                  + '; '.join(row['mixedproxy_notes'] + row['asplos_notes']), flush=True)
            rows.append(row)
    return rows


def check_mixedproxy(write=None):
    rows, lit = [], []
    for f in sorted(os.listdir(os.path.join(MPX, 'tests'))):
        if not f.endswith('.test'): continue
        for i, inst in enumerate(mp_instances(os.path.join(MPX, 'tests', f))):
            base = f[:-5] + (f'-{i + 1}' if '$$' in open(os.path.join(MPX, 'tests', f)).read() else '')
            nat = mp_native_verdicts(inst)
            for text, kind, cname in mp_to_litmus(inst, base):
                lit.append(text)
                t = ptxmm.parse(text)[0]
                p = V(ptxmm_verdict(t))
                nv = V(nat[cname] if cname in nat else nat['check_' + cname])
                exp = t['expect']
                try:
                    av = V(run_asplos(t)[0])
                except Untranslatable:
                    av = '-'
                flag = (p != nv) or (nv != exp) or (av != '-' and av != p)
                body = ' / '.join(';'.join(x.strip() for x in th['ins']) for th in t['threads'])
                print(f"{'DIFF' if flag else 'same'}  {t['name']:34s} nvidia-expect={exp:9s} mixedproxy={nv:9s} "
                      f"ptxmm={p:9s} asplos={av:9s} {body} | {t['exists']}", flush=True)
                rows.append(dict(name=t['name'], expect=exp, mixedproxy=nv, ptxmm=p, asplos=av, body=body, flag=flag))
    if write:
        hdr = ('# Generated by scripts/nvmm.py --write from NVlabs/mixedproxy tests/*.test (BSD-3-Clause, NVIDIA).\n'
               '# One test per template instance and permit/assert command; `expect` is the nvlitmus expectation.\n'
               '# Not run by `make memmodel` (subdirectory); see docs/review-checklist.md for the disagreements.\n\n')
        open(write, 'w').write(hdr + '\n'.join(lit))
    return rows


def main(argv):
    root = os.path.dirname(HERE)
    args = argv[1:]
    if args and args[0] in ('--emit-mp', '--emit-asplos'):
        for t in ptxmm.parse(open(args[1]).read()):
            print(f'// ==== {t["name"]}')
            try:
                if args[0] == '--emit-mp':
                    text, deps, notes, _, _ = litmus_to_mp(t); print(text, '// deps', deps, notes)
                else:
                    for n, m in litmus_to_asplos(t): print(f'// module {n}\n{m}')
            except Untranslatable as u:
                print('// untranslatable:', u)
        return 0
    global MP_FENCE_FIX
    MP_FENCE_FIX = '--mp-fence-fix' in args
    what = [a for a in args if not a.startswith('-')] or ['repo', 'extra', 'mixedproxy']
    out = os.path.join(root, 'tests', 'memmodel', 'nvidia', 'mixedproxy.litmus') if '--write' in args else None
    allrows = {}
    files = [a for a in what if a.endswith('.litmus')]
    if files:
        allrows['files'] = check_litmus_files(files, 'files')
    if 'repo' in what:
        d = os.path.join(root, 'tests', 'memmodel')
        allrows['repo'] = check_litmus_files(sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.litmus')), 'repo')
    if 'extra' in what:
        allrows['extra'] = check_litmus_files([os.path.join(root, 'tests', 'memmodel', 'nvidia', f + '.litmus')
                                               for f in ('asplos2019', 'differential')], 'extra')
    if 'mixedproxy' in what:
        allrows['mixedproxy'] = check_mixedproxy(out)
    print('---- summary')
    for k, rows in allrows.items():
        if k == 'mixedproxy':
            agree = sum(r['ptxmm'] == r['mixedproxy'] for r in rows)
            print(f"mixedproxy suite -> ptxmm: {len(rows)} tests, ptxmm agrees with the model on {agree}, "
                  f"model matches its own expectation on {sum(r['mixedproxy'] == r['expect'] for r in rows)}")
            tr = [r for r in rows if r['asplos'] != '-']
            print(f"mixedproxy suite -> asplos: {len(tr)} translated (no proxies / aliases), ptxmm agrees on "
                  f"{sum(r['asplos'] == r['ptxmm'] for r in tr)}, mixedproxy agrees on "
                  f"{sum(r['asplos'] == r['mixedproxy'] for r in tr)}")
        else:
            for m in ('mixedproxy', 'asplos'):
                tr = [r for r in rows if r[m] != '-']
                print(f"{k} -> {m}: {len(rows)} tests, {len(tr)} translated, ptxmm agrees on {sum(r[m] == r['ptxmm'] for r in tr)}, "
                      f"model matches expect on {sum(r[m] == r['expect'] for r in tr)}")
            print(f"{k}: ptxmm matches expect on {sum(r['ptxmm'] == r['expect'] for r in rows)}/{len(rows)}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

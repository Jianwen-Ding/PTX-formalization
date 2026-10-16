#!/usr/bin/env python3
"""Generate tests/warp-*.ptx: shfl/vote/match/elect/redux/activemask against
Python transcriptions of the PTX ISA semantics.   usage: gen_warp.py [SEED] [PART PARTS]"""
import random, sys
random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 51)
PART, PARTS = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
M32 = 0xFFFFFFFF
def r32(): return random.choice([0, M32, 1, 0x80000000]) if random.random() < .2 else random.getrandbits(32)
def s32(x): return x - (1 << 32) if x >> 31 else x

def shfl(mode, lane, a_of, b, c, active):
    bval, cval, seg = b & 31, c & 31, (c >> 8) & 31
    maxl = (lane & seg) | (cval & ~seg & 31); minl = lane & seg
    if mode == 'up': j = lane - bval; p = j >= maxl
    elif mode == 'down': j = lane + bval; p = j <= maxl
    elif mode == 'bfly': j = lane ^ bval; p = j <= maxl
    else: j = minl | (bval & ~seg & 31); p = j <= maxl
    if not p: j = lane
    own = a_of[lane]
    return (a_of[j] if j in a_of else own), int(p)

cases = []   # (name, threads, per-thread input lists, mask, per-thread outputs (list per thread of out words), ptx body fn)
def lanes_mask(active): return sum(1 << l for l in active)
for mode in ('up', 'down', 'bfly', 'idx'):
    for mask_kind in ('full', 'low16'):
        T = 32
        active = list(range(32)) if mask_kind == 'full' else list(range(16))
        mask = lanes_mask(active)
        a = [r32() for _ in range(T)]
        b = [random.choice([0, 1, 2, 5, 16, 31, random.randint(0, 63)]) for _ in range(T)]
        c = [random.choice([0x1f, 0x1f | (16 << 8), 0x1f | (24 << 8), 0, 0 | (16 << 8), random.getrandbits(13)]) for _ in range(T)]
        a_of = {l: a[l] for l in active}
        outs = []
        for t in range(T):
            if t in active:
                d, p = shfl(mode, t, a_of, b[t], c[t], active)
                outs.append([d, p])
            else: outs.append([0xDEAD, 0xDEAD])
        cases.append((f'shfl.sync.{mode}.b32 %r10|%p1, %r4, %r5, %r6, {mask}', T, [a, b, c], mask, outs, 'shfl'))
for kind in ('all', 'any', 'uni', 'ballot'):
    for negate in (False, True):
        for mask_kind in ('full', 'odd'):
            T = 32
            active = list(range(32)) if mask_kind == 'full' else [l for l in range(32) if l % 2]
            mask = lanes_mask(active)
            bias = random.choice([.1, .5, .9, 1.0, 0.0])
            q = [int(random.random() < bias) for _ in range(T)]
            vals = [(not x if negate else x) for x in (bool(q[l]) for l in active)]
            if kind == 'all': res = int(all(vals))
            elif kind == 'any': res = int(any(vals))
            elif kind == 'uni': res = int(all(vals) or not any(vals))
            else: res = sum(1 << l for l, v in zip(active, vals) if v)
            outs = [[res] if t in active else [0xDEAD] for t in range(T)]
            neg = '!' if negate else ''
            text = (f'vote.sync.{kind}.pred %p2, {neg}%p3, {mask}' if kind != 'ballot' else f'vote.sync.ballot.b32 %r10, {neg}%p3, {mask}')
            cases.append((text, T, [q], mask, outs, 'vote' + ('b' if kind == 'ballot' else 'p')))
for width in (32, 64):
    for kind in ('any', 'all'):
        T = 32
        active = list(range(32)); mask = M32
        pool = [random.getrandbits(width) for _ in range(3)]
        a = [random.choice(pool) if (kind == 'any' or random.random() < .3) else pool[0] for _ in range(T)]
        if kind == 'all' and random.random() < .5: a = [pool[0]] * T
        outs = []
        for t in range(T):
            if kind == 'any': outs.append([sum(1 << l for l in active if a[l] == a[t])])
            else:
                eq = all(a[l] == a[active[0]] for l in active)
                outs.append([mask if eq else 0, int(eq)])
        txt = f'match.sync.{kind}.b{width} %r10' + ('|%p1' if kind == 'all' else '') + (f', %rd4, {mask}' if width == 64 else f', %r4, {mask}')
        cases.append((txt, T, [a], mask, outs, f'match{width}'))
for mask_kind in ('full', 'sparse'):
    T = 32; active = list(range(32)) if mask_kind == 'full' else [3, 4, 9, 20, 31]
    mask = lanes_mask(active)
    leader = min(active)
    outs = [[leader, int(t == leader)] if t in active else [0xDEAD, 0xDEAD] for t in range(T)]
    cases.append((f'elect.sync %r10|%p1, {mask}', T, [], mask, outs, 'elect'))
for op in ('add', 'min', 'max', 'and', 'or', 'xor'):
    for ty in (['u32', 's32'] if op in ('add', 'min', 'max') else ['b32']):
        T = 32; active = list(range(32)); mask = M32
        a = [r32() for _ in range(T)]
        v = [a[l] for l in active]
        sv = [s32(x) for x in v]
        if op == 'add': res = sum(v) & M32
        elif op == 'min': res = (min(sv) & M32) if ty == 's32' else min(v)
        elif op == 'max': res = (max(sv) & M32) if ty == 's32' else max(v)
        elif op == 'and':
            res = M32
            for x in v: res &= x
        elif op == 'or':
            res = 0
            for x in v: res |= x
        else:
            res = 0
            for x in v: res ^= x
        cases.append((f'redux.sync.{op}.{ty} %r10, %r4, {mask}', T, [a], mask, [[res]] * T, 'redux'))
cases = cases[PART::PARTS]

print("// GENERATED by scripts/gen_warp.py -- warp-synchronous instructions vs. ISA transcriptions.")
print(".version 8.0\n.target sm_90\n.address_size 64\n")
for k, (text, T, ins, mask, outs, kind) in enumerate(cases):
    nin = len(ins); nout = len(outs[0])
    ps = ", ".join(f".param .u64 p{j}" for j in range(nin + 1))
    print(f".visible .entry k{k}({ps})\n{{\n    .reg .b32 %r<16>; .reg .b64 %rd<8>; .reg .pred %p<6>;\n    mov.u32 %r0, %tid.x; mul.wide.u32 %rd1, %r0, 4; mul.wide.u32 %rd7, %r0, 8;")
    for j in range(nin):
        if kind.startswith('vote'):
            print(f"    ld.param.u64 %rd2, [p{j}]; add.s64 %rd2, %rd2, %rd1; ld.global.b32 %r{4+j}, [%rd2]; setp.ne.u32 %p3, %r{4+j}, 0;")
        elif kind == 'match64':
            print(f"    ld.param.u64 %rd2, [p{j}]; add.s64 %rd2, %rd2, %rd7; ld.global.b64 %rd4, [%rd2];")
        else:
            print(f"    ld.param.u64 %rd2, [p{j}]; add.s64 %rd2, %rd2, %rd1; ld.global.b32 %r{4+j}, [%rd2];")
    print(f"    ld.param.u64 %rd3, [p{nin}]; mul.wide.u32 %rd5, %r0, {4 * nout}; add.s64 %rd3, %rd3, %rd5;")
    print(f"    mov.u32 %r10, 57005; mov.u32 %r11, 57005;")
    if mask != M32:
        print(f"    mov.u32 %r12, 1; shl.b32 %r12, %r12, %r0; and.b32 %r13, %r12, {mask}; setp.eq.u32 %p5, %r13, 0; @%p5 bra $L_done;")
    print(f"    {text};")
    if kind in ('shfl', 'elect', 'match32', 'match64') and '|%p1' in text:
        print("    selp.u32 %r11, 1, 0, %p1;")
    if kind == 'votep':
        print("    selp.u32 %r10, 1, 0, %p2;")
    print("$L_done:")
    print("    st.global.u32 [%rd3], %r10;")
    if nout == 2: print("    st.global.u32 [%rd3+4], %r11;")
    print("    ret;\n}\n")
for k, (text, T, ins, mask, outs, kind) in enumerate(cases):
    nin = len(ins); nout = len(outs[0])
    for j, col in enumerate(ins):
        ty = 'u64' if kind == 'match64' else 'u32'
        print(f".buffer I{k}_{j} .{ty} = {{ {', '.join(str(x) for x in col)} }};")
    print(f".buffer O{k} .u32 [{T * nout}];\n.launch k{k} <<< 1, {T} >>> ({', '.join([f'I{k}_{j}' for j in range(nin)] + [f'O{k}'])});")
    print(f".expect O{k} .u32 = {{ {', '.join(str(x) for o in outs for x in o)} }};")

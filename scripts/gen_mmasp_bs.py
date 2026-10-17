#!/usr/bin/env python3
"""Generate tests/sp-bs-*.ptx: mma.sp::ordered_metadata with kind::f8f6f4 / kind::mxf8f6f4 / kind::mxf4 / kind::mxf4nvf4 (block scaled)
against a dense model.  The sparse structure, compression and metadata layout come from gen_mmasp.py (integer element patterns are
reinterpreted as the floating-point element types).  usage: gen_mmasp_bs.py"""
import random, sys
from fractions import Fraction
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_mmasp.py').read()
pre = src[:src.index("h = lambda xs")].replace("    return shape, dt, at, ct, mods, sel, fa, fb, fc, fd, meta", "    return shape, dt, at, ct, mods, sel, fa, fb, fc, fd, meta, A, B, C")
# teach the generator's tables about the fp6 / fp4 element types ('e2m1p' = two packed 4-bit elements per byte, behaves like u4)
pre = pre.replace("'e4m3': 8, 'e5m2': 8,", "'e4m3': 8, 'e5m2': 8, 'e2m3': 8, 'e3m2': 8, 'e2m1': 8, 'e2m1p': 16,")
pre = pre.replace("'e4m3': 32, 'e5m2': 32,", "'e4m3': 32, 'e5m2': 32, 'e2m3': 32, 'e3m2': 32, 'e2m1': 32, 'e2m1p': 64,")
pre = pre.replace("isf = at in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2')", "isf = at in ('f16', 'bf16', 'tf32', 'e4m3', 'e5m2', 'e2m3', 'e3m2', 'e2m1', 'e2m1p')")
pre = pre.replace("if at in ('s4', 'u4'): return 8, 4", "if at in ('s4', 'u4', 'e2m1p'): return 8, 4")
pre = pre.replace("elems += [A[(r, cw * c + x)] for x in cols]", "elems += [CONT(at, A[(r, cw * c + x)]) for x in cols]")
sys.argv = [sys.argv[0]]
exec(compile(pre, 'gen_mmasp_prefix', 'exec'))
random.seed(53)
def ew_of(ty): return 4 if ty == 'e2m1p' else (8 if ty in ('e4m3', 'e5m2', 'e2m3', 'e3m2', 'e2m1') else (32 if ty == 'tf32' else (int(ty[1:]) if ty[0] in 'sufb' and ty[1:].isdigit() else 16)))
def mini_tab2(e, m):
    bias = (1 << (e - 1)) - 1; t = {}
    for c in range(1 << (e + m + 1)):
        s_, E, M = c >> (e + m), (c >> m) & ((1 << e) - 1), c & ((1 << m) - 1)
        v = Fraction(M, 1 << m) * Fraction(2) ** (1 - bias) if E == 0 else Fraction((1 << m) + M, 1 << m) * Fraction(2) ** (E - bias)
        t[c] = -v if s_ else v
    return t
MINI['e2m1p'] = None
MINI.update({'e2m3': mini_tab2(2, 3), 'e3m2': mini_tab2(3, 2), 'e2m1': mini_tab2(2, 1)})
MINI['e2m1p'] = MINI['e2m1']
def CONT(ty, c):            # element code -> 8-bit container with random padding (e2m1: central 4 bits, fp6: low 6 bits); packed 4-bit: unchanged
    if ty == 'e2m1': return (random.getrandbits(2) << 6) | (c << 2) | random.getrandbits(2)
    if ty in ('e2m3', 'e3m2'): return (random.getrandbits(2) << 6) | c
    return c
def ue8m0(x): return Fraction(2) ** (x - 127)
def ue4m3(x):
    e, m = x >> 3, x & 7
    return Fraction(m, 8) * Fraction(2) ** (1 - 7) if e == 0 else Fraction(8 + m, 8) * Fraction(2) ** (e - 7)
def val(ty, bits, packed4): return fval('e2m1' if ty == 'e2m1p' else ty, bits)
# (shape, kind, scale_vec qualifier, atype, btype, stype, ctype/dtype, byte-id-a, thread-id-a, byte-id-b, thread-id-b)
CFG = [('m16n8k64', 'kind::f8f6f4', None, 'e3m2', 'e2m3', None, 'f32', 0, 0, 0, 0),
       ('m16n8k64', 'kind::f8f6f4', None, 'e2m1', 'e5m2', None, 'f16', 0, 0, 0, 0),
       ('m16n8k64', 'kind::mxf8f6f4', '.scale_vec::1X', 'e4m3', 'e2m3', 'ue8m0', 'f32', 2, 1, 3, 2),
       ('m16n8k128', 'kind::mxf4', '', 'e2m1', 'e2m1', 'ue8m0', 'f32', 2, 1, 0, 3),
       ('m16n8k128', 'kind::mxf4nvf4', '.scale_vec::4X', 'e2m1', 'e2m1', 'ue4m3', 'f32', 0, 1, 0, 2)]
h = lambda xs: ", ".join(str(x) for x in xs)
for kk, (shape, kind, sv, at, bt, st, dt, bida, tida, bidb, tidb) in enumerate(CFG):
    m, n, k = [int(x) for x in shape[1:].replace('n', ' ').replace('k', ' ').split()]
    packed4 = kind in ('kind::mxf4', 'kind::mxf4nvf4')
    atx = 'e2m1p' if packed4 else at
    shape_, _, _, _, _, sel, fa, fb, fc, fd_int, meta, A, B, C = build_sp((shape, 'f32', atx, 'f32', ''))
    # B with its own element type
    ewb = 4 if packed4 else 8; ek = 32 // ewb
    LB = {}
    for lane in range(32):
        g_, t_ = lane >> 2, lane & 3
        for i in range(k * n // 32): LB[(lane, i)] = (ek * t_ + i % ek + 4 * ek * (i // ek), g_)
    Bc = {pos: (rand_float_bits('e2m1' if packed4 else bt)) for pos in LB.values()}
    fb = frag_regs(LB, {p: (v if packed4 else CONT(bt, v)) for p, v in Bc.items()}, ewb, k * n // 32, 32)
    B = Bc
    # C must be an accumulator of the right type
    LC = layout('C', shape, 32)
    C = {pos: rand_float_bits(dt) for pos in LC.values()}
    nsc = 1 if st is None else (int(sv[-2]) if sv else (2 if kind == 'kind::mxf4' else 1))
    dec = ue4m3 if st == 'ue4m3' else ue8m0
    rnd_s = (lambda: random.randint(0x30, 0x48)) if st == 'ue4m3' else (lambda: random.randint(122, 131))
    SA = [[rnd_s() for _ in range(nsc)] for _ in range(m)]; SB = [[rnd_s() for _ in range(nsc)] for _ in range(n)]
    blk = k // nsc
    D = {}
    for r in range(m):
        for c in range(n):
            s = fval(dt, C[(r, c)])
            for k2 in range(k):
                p = val(at, A[(r, k2)], packed4) * val(bt, B[(k2, c)], packed4)
                if st is not None: p *= dec(SA[r][k2 // blk]) * dec(SB[c][k2 // blk])
                s += p
            D[(r, c)] = to_bits(dt, s)
    fc = frag_regs(LC, C, 16 if dt == 'f16' else 32, m * n // 32, 32)
    fd = frag_regs(LC, D, 16 if dt == 'f16' else 32, m * n // 32, 32)
    wsa = [random.getrandbits(32) for _ in range(32)]; wsb = [random.getrandbits(32) for _ in range(32)]
    if st is not None:
        for r in range(m):
            lane = (r % 8) * 4 + 2 * tida + r // 8
            for j in range(nsc):
                sh = 8 * (bida + j); wsa[lane] = (wsa[lane] & ~(255 << sh)) | (SA[r][j] << sh)
        for c in range(n):
            lane = c * 4 + tidb
            for j in range(nsc):
                sh = 8 * (bidb + j); wsb[lane] = (wsb[lane] & ~(255 << sh)) | (SB[c][j] << sh)
    na, nb_, nc_, nd = len(fa[0]), len(fb[0]), len(fc[0]), len(fd[0])
    rg = lambda p, i: f"%{p}{i}"
    vec = lambda p, c: "{" + ", ".join(rg(p, i) for i in range(c)) + "}"
    L = ["// GENERATED by scripts/gen_mmasp_bs.py -- sparse mma with fp6/fp4 kinds and block scaling vs. a dense model.", ".version 8.7\n.target sm_120a\n.address_size 64\n",
         f".visible .entry k{kk}(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pe, .param .u64 ps, .param .u64 po)\n{{\n    .reg .b32 %r<4>, %SA, %SB, %E; .reg .b64 %rd<4>; .reg .b32 %A<{na}>, %B<{nb_}>, %C<{nc_}>, %D<{nd}>;",
         "    mov.u32 %r0, %tid.x;"]
    for nm, cnt, par in (('A', na, 'pa'), ('B', nb_, 'pb'), ('C', nc_, 'pc'), ('E', 1, 'pe')):
        L.append(f"    ld.param.u64 %rd0, [{par}]; mul.wide.u32 %rd1, %r0, {cnt * 4}; add.s64 %rd0, %rd0, %rd1;")
        for i in range(cnt): L.append(f"    ld.global.b32 {rg(nm, i) if nm != 'E' else '%E'}, [%rd0+{i * 4}];")
    L.append("    ld.param.u64 %rd0, [ps]; mul.wide.u32 %rd1, %r0, 4; add.s64 %rd0, %rd0, %rd1;\n    ld.global.b32 %SA, [%rd0];\n    ld.global.b32 %SB, [%rd0+128];")
    if st is None:
        L.append(f"    mma.sp::ordered_metadata.sync.aligned.{shape}.row.col.{kind}.{dt}.{at}.{bt}.{dt} {vec('D', nd)}, {vec('A', na)}, {vec('B', nb_)}, {vec('C', nc_)}, %E, {sel};")
    else:
        L.append(f"    mma.sp::ordered_metadata.sync.aligned.{shape}.row.col.{kind}.block_scale{sv}.f32.{at}.{bt}.f32.{st} {vec('D', nd)}, {vec('A', na)}, {vec('B', nb_)}, {vec('C', nc_)}, %E, {sel}, %SA, {{{bida}, {tida}}}, %SB, {{{bidb}, {tidb}}};")
    L.append(f"    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %r0, {nd * 4}; add.s64 %rd0, %rd0, %rd1;")
    for i in range(nd): L.append(f"    st.global.b32 [%rd0+{i * 4}], %D{i};")
    L.append("    ret;\n}\n")
    flat = lambda f: h(x for lane in f for x in lane)
    L.append(f".buffer A .u32 = {{ {flat(fa)} }};\n.buffer B .u32 = {{ {flat(fb)} }};\n.buffer C .u32 = {{ {flat(fc)} }};\n.buffer E .u32 = {{ {h(meta)} }};\n.buffer S .u32 = {{ {h(wsa + wsb)} }};")
    L.append(f".buffer O .u32 [{32 * nd}];\n.launch k{kk} <<< 1, 32 >>> (A, B, C, E, S, O);\n.expect O .u32 = {{ {flat(fd)} }};")
    open(f'tests/sp-bs-{kk}.ptx', 'w').write("\n".join(L) + "\n")
print(len(CFG), "tests")

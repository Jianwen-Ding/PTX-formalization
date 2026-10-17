#!/usr/bin/env python3
"""Generate tests/tcg2-cp.ptx: tcgen05.cp.cta_group::2 and tcgen05.shift.cta_group::2 on a CTA pair (both CTAs copy from their own
shared memory / shift their own tensor memory).  usage: gen_tcgen05_cp2.py"""
import random, sys
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_wgmma.py').read().split('# name: (N, K')[0]
sys.argv = [sys.argv[0]]
exec(compile(src, 'gen_wgmma_prefix', 'exec'))
t5 = open(here + '/gen_tcgen05.py').read()
exec(compile(t5[t5.index('h = lambda xs'):t5.index('# ---------------------------------------------------------------- alloc')], 'gen_tcgen05_prefix', 'exec'))
random.seed(31)
R = 128; W = 4
SRC = [[[random.getrandbits(32) for _ in range(W)] for _ in range(R)] for _ in range(2)]
imgs = []
for c in range(2):
    img, lbo, sbo = build_tile(SRC[c], R, W, 32, True, 0, 0)
    imgs.append(img)
size = (max(max(i) for i in imgs) + 1 + 2047) // 2048 * 2048
data = b"".join(bytes(i.get(j, 0) for j in range(size)) for i in imgs)
D = desc100(0, lbo, sbo, 0)
NC = 4
words = []
for c in range(2):
    exp = {(r, w): SRC[c][r][w] for r in range(R) for w in range(W)}
    # shift down the 32-lane block at lane 64, columns 0..7 (only 0..3 are populated)
    old = dict(exp)
    for r in range(31):
        for w in range(W): exp[(64 + r + 1, w)] = old[(64 + r, w)]
    words += [[exp[(u, w)] for w in range(NC)] for u in range(128)]
L = HDR[:]
L.append(f""".visible .entry k(.param .u64 pimg, .param .u64 po) .reqnctapercluster 2, 1, 1
{{
    .reg .b32 %r<{NC}>, %x<4>, %tl, %tw, %t0, %cta, %sh; .reg .b64 %rd<8>, %sd; .reg .pred %p, %q, %q0;
    .shared .align 1024 .b8 buf[{size}];
    .shared .align 16 .b32 slot[2];
    mov.u32 %tl, %tid.x;
    mov.u32 %cta, %ctaid.x;
    mov.u64 %rd5, slot;
    setp.lt.u32 %p, %tl, 32;
    setp.eq.u32 %q, %tl, 0;
    @%p tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%rd5], 32;
    ld.param.u64 %rd0, [pimg]; mul.wide.u32 %rd6, %cta, {size}; add.u64 %rd0, %rd0, %rd6;
    mov.u64 %rd2, buf; mul.wide.u32 %rd1, %tl, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;""")
for it in range(size // 2048):
    o = it * 2048
    L.append(f"    ld.global.b32 %x0, [%rd0+{o}]; ld.global.b32 %x1, [%rd0+{o + 4}]; ld.global.b32 %x2, [%rd0+{o + 8}]; ld.global.b32 %x3, [%rd0+{o + 12}];")
    L.append(f"    st.shared.b32 [%rd2+{o}], %x0; st.shared.b32 [%rd2+{o + 4}], %x1; st.shared.b32 [%rd2+{o + 8}], %x2; st.shared.b32 [%rd2+{o + 12}], %x3;")
L.append(f"""    bar.sync 0;
    barrier.cluster.arrive.release.aligned;
    barrier.cluster.wait.acquire.aligned;
    mov.b64 %sd, {D};
    shr.u32 %tw, %tl, 5; shl.b32 %tw, %tw, 21;
    mov.u32 %t0, 0;
    mov.u32 %sh, 4194304;
    setp.eq.u32 %q0, %cta, 0;
    and.pred %q, %q, %q0;
    @%q tcgen05.cp.cta_group::2.128x128b [%t0], %sd;
    @%q tcgen05.shift.cta_group::2.down [%sh];
    tcgen05.fence::after_thread_sync;
    bar.sync 0;
    barrier.cluster.arrive.release.aligned;
    barrier.cluster.wait.acquire.aligned;
    tcgen05.ld.sync.aligned.32x32b.x{NC}.b32 {vec('r', NC)}, [%tw];
    tcgen05.wait::ld.sync.aligned;
    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd6, %cta, {128 * NC * 4}; add.u64 %rd0, %rd0, %rd6; mul.wide.u32 %rd1, %tl, {NC * 4}; add.u64 %rd0, %rd0, %rd1;""")
for i in range(NC): L.append(f"    st.global.b32 [%rd0+{i * 4}], %r{i};")
L.append("    bar.sync 0;\n    barrier.cluster.arrive.release.aligned;\n    barrier.cluster.wait.acquire.aligned;\n    @%p tcgen05.dealloc.cta_group::2.sync.aligned.b32 %t0, 32;\n    ret;\n}\n")
code = "\n".join(L) + f"\n.buffer IMG .u8 [{len(data)}];\n"
i = 0
while i < len(data):
    if data[i] == 0: i += 1; continue
    j = i
    while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
    code += f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};\n"
    i = j
code += f".buffer O .u32 [{256 * NC}];\n.launch k <<< 2, 128 >>> (IMG, O);\n" + expect('O', 'u32', [x for r in words for x in r]) + "\n"
open('tests/tcg2-cp.ptx', 'w').write(code)

#!/usr/bin/env python3
"""Generate tests/tc-cp.ptx: tcgen05.cp (plain and warp-multicast shapes), tcgen05.shift and tcgen05.commit
(mbarrier arrive) against direct models.  usage: gen_tcgen05_cp.py"""
import random, sys
here = __file__.rsplit('/', 1)[0]
src = open(here + '/gen_wgmma.py').read().split('# name: (N, K')[0]
sys.argv = [sys.argv[0]]
exec(compile(src, 'gen_wgmma_prefix', 'exec'))
t5 = open(here + '/gen_tcgen05.py').read()
exec(compile(t5[t5.index('h = lambda xs'):t5.index('# ---------------------------------------------------------------- alloc')], 'gen_tcgen05_prefix', 'exec'))
random.seed(5)
R = 128; W = 8
SRC = [[random.getrandbits(32) for _ in range(W)] for _ in range(R)]
img, lbo, sbo = build_tile(SRC, R, W, 32, True, 0, 0)
size = (max(img) + 1 + 2047) // 2048 * 2048
data = bytes(img.get(i, 0) for i in range(size))
D = desc100(0, lbo, sbo, 0)
# expected tensor memory (lane, col) after each step, as read back by thread t = lane
exp = {}
def put(lane, col, row, w0=0, nw=W):
    for w in range(nw): exp[(lane, col + w)] = SRC[row][w0 + w]
# step A: cp.128x256b to columns 0..7
for r in range(128): put(r, 0, r)
# step B: cp.4x256b to columns 8..15, lanes 0..3 (source rows 0..3)
for r in range(4): put(r, 8, r)
# step C: cp.64x128b.warpx2::02_13 to columns 16..19: rows 0..31 -> warps 0,1 ; rows 32..63 -> warps 2,3 (lane = 32*w + r)
for i in range(64):
    ws = (0, 1) if i < 32 else (2, 3)
    for w in ws: put(32 * w + i % 32, 16, i, 0, 4)
# step D: cp.64x128b.warpx2::01_23 to columns 20..23: rows 0..31 -> warps 0,2 ; rows 32..63 -> warps 1,3
for i in range(64):
    ws = (0, 2) if i < 32 else (1, 3)
    for w in ws: put(32 * w + i % 32, 20, i, 0, 4)
# step E: cp.32x128b.warpx4 to columns 24..27: rows 0..31 -> every warp
for i in range(32):
    for w in range(4): put(32 * w + i, 24, i, 0, 4)
# step F: shift down the 32-lane block at lane 32 (warp 1), columns 0..7: lane 33+r gets old lane 32+r
old = {(l, c): exp[(l, c)] for l in range(32, 64) for c in range(8)}
for r in range(31):
    for c in range(8): exp[(32 + r + 1, c)] = old[(32 + r, c)]
NC = 32
words = [[exp.get((u, c), 0) for c in range(NC)] for u in range(128)]
body = f""".visible .entry k(.param .u64 pimg, .param .u64 po)
{{
    .reg .b32 %r<64>, %x<4>, %tl, %tw, %t0; .reg .b64 %rd<8>, %sd; .reg .pred %p, %q, %w;
    .shared .align 1024 .b8 buf[{size}];
    .shared .align 16 .b32 slot[2];
    .shared .align 8 .b64 mbar;
    mov.u32 %tl, %tid.x;
    mov.u64 %rd5, slot;
    setp.lt.u32 %p, %tl, 32;
    setp.eq.u32 %q, %tl, 0;
    @%p tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%rd5], 32;
    ld.param.u64 %rd0, [pimg]; mov.u64 %rd2, buf; mul.wide.u32 %rd1, %tl, 16; add.u64 %rd0, %rd0, %rd1; add.u64 %rd2, %rd2, %rd1;"""
L = HDR[:] + [body]
for it in range(size // 2048):
    o = it * 2048
    L.append(f"    ld.global.b32 %x0, [%rd0+{o}]; ld.global.b32 %x1, [%rd0+{o + 4}]; ld.global.b32 %x2, [%rd0+{o + 8}]; ld.global.b32 %x3, [%rd0+{o + 12}];")
    L.append(f"    st.shared.b32 [%rd2+{o}], %x0; st.shared.b32 [%rd2+{o + 4}], %x1; st.shared.b32 [%rd2+{o + 8}], %x2; st.shared.b32 [%rd2+{o + 12}], %x3;")
L.append(f"""    mov.u64 %rd6, mbar;
    @%q mbarrier.init.shared.b64 [%rd6], 1;
    bar.sync 0;
    mov.b64 %sd, {D};
    shr.u32 %tw, %tl, 5; shl.b32 %tw, %tw, 21;
    mov.u32 %t0, 0;
    // thread 0 issues the copies, each followed by commit and a wait on the mbarrier phase
    @%q tcgen05.cp.cta_group::1.128x256b [%t0], %sd;
    @%q tcgen05.cp.cta_group::1.4x256b [%t0+8], %sd;
    @%q tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%t0+16], %sd;
    @%q tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%t0+20], %sd;
    @%q tcgen05.cp.cta_group::1.32x128b.warpx4 [%t0+24], %sd;
    mov.u32 %r60, 2097152;
    @%q tcgen05.shift.cta_group::1.down [%r60];
    @%q tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%rd6];
W1: mbarrier.try_wait.parity.shared.b64 %w, [%rd6], 0;
    @!%w bra W1;
    tcgen05.fence::after_thread_sync;
    bar.sync 0;
    tcgen05.ld.sync.aligned.32x32b.x{NC}.b32 {vec('r', NC)}, [%tw];
    tcgen05.wait::ld.sync.aligned;
    ld.param.u64 %rd0, [po]; mul.wide.u32 %rd1, %tl, {NC * 4}; add.u64 %rd0, %rd0, %rd1;""")
for i in range(NC): L.append(f"    st.global.b32 [%rd0+{i * 4}], %r{i};")
L.append("    bar.sync 0;\n    @%p tcgen05.dealloc.cta_group::1.sync.aligned.b32 %t0, 32;\n    ret;\n}\n")
code = "\n".join(L) + f"\n.buffer IMG .u8 [{len(data)}];\n"
i = 0
while i < len(data):
    if data[i] == 0: i += 1; continue
    j = i
    while j < len(data) and (j - i < 400) and not (data[j] == 0 and all(x == 0 for x in data[j:j + 16])): j += 1
    code += f".patch IMG .u8 @ {i} = {{ {h(data[i:j])} }};\n"
    i = j
code += f".buffer O .u32 [{128 * NC}];\n.launch k <<< 1, 128 >>> (IMG, O);\n" + expect('O', 'u32', [x for r in words for x in r]) + "\n"
open('tests/tc-cp.ptx', 'w').write(code)
print("tc-cp", len(code), "bytes")

#!/usr/bin/env python3
"""Generate tests/corpus-*.ptx: small realistic kernels (the idioms of CUDA-compiled code) with expected results
computed by independent Python models.  usage: gen_corpus.py"""
import struct, random
random.seed(7)
def f2b(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def lst(v): return ', '.join(str(x) for x in v)
HDR = '.version 8.0\n.target sm_90\n.address_size 64\n\n'
def emit(name, kernel, buffers, launch, expects):
    s = HDR + kernel + '\n'
    for n, ty, data in buffers:
        s += f'.buffer {n} .{ty} = {{ {lst(data)} }};\n' if isinstance(data, list) else f'.buffer {n} .{ty} [{data}];\n'
    s += launch + '\n'
    for n, ty, data in expects: s += f'.expect {n} .{ty} = {{ {lst(data)} }};\n'
    open(f'tests/corpus-{name}.ptx', 'w').write(s)
M = 0xffffffff
# 1 vecadd with bounds check
n = 100; a = [float(random.randint(-50, 50)) for _ in range(n)]; b = [float(random.randint(-50, 50)) for _ in range(n)]
emit('vecadd', '''.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u32 pn)
{
    .reg .pred %p; .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .f32 %f<4>;
    ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc]; ld.param.u32 %r1, [pn];
    mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p, %r5, %r1;
    @%p bra DONE;
    mul.wide.s32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4; add.s64 %rd6, %rd2, %rd4; add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5]; ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
''', [('A', 'u32', [f2b(x) for x in a]), ('B', 'u32', [f2b(x) for x in b]), ('C', 'u32', n)],
 '.launch k <<< 4, 32 >>> (A, B, C, 100);', [('C', 'u32', [f2b(x + y) for x, y in zip(a, b)])])
# 2 saxpy with fma
x = [float(random.randint(-9, 9)) for _ in range(64)]; y = [float(random.randint(-9, 9)) for _ in range(64)]
emit('saxpy', '''.visible .entry k(.param .f32 pa, .param .u64 px, .param .u64 py)
{
    .reg .b32 %r<4>; .reg .b64 %rd<6>; .reg .f32 %f<5>;
    ld.param.f32 %f1, [pa]; ld.param.u64 %rd1, [px]; ld.param.u64 %rd2, [py];
    mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3; add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4]; ld.global.f32 %f3, [%rd5];
    fma.rn.f32 %f4, %f1, %f2, %f3;
    st.global.f32 [%rd5], %f4;
    ret;
}
''', [('X', 'u32', [f2b(v) for v in x]), ('Y', 'u32', [f2b(v) for v in y])], '.launch k <<< 1, 64 >>> (0f40000000, X, Y);',
 [('Y', 'u32', [f2b(2.0 * p + q) for p, q in zip(x, y)])])
# 3 block reduction (shared tree)
v = [random.randint(0, 1000) for _ in range(256)]
emit('reduce', '''.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .shared .align 4 .b32 sm[256];
    .reg .pred %p; .reg .b32 %r<10>; .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd1, %rd3;
    ld.global.u32 %r2, [%rd4];
    mov.u64 %rd5, sm; add.s64 %rd5, %rd5, %rd3;
    st.shared.u32 [%rd5], %r2;
    bar.sync 0;
    mov.u32 %r3, 128;
LOOP:
    setp.ge.u32 %p, %r1, %r3;
    @%p bra SKIP;
    add.u32 %r4, %r1, %r3; mul.wide.u32 %rd3, %r4, 4;
    mov.u64 %rd4, sm; add.s64 %rd4, %rd4, %rd3;
    ld.shared.u32 %r5, [%rd4]; ld.shared.u32 %r6, [%rd5];
    add.u32 %r6, %r6, %r5; st.shared.u32 [%rd5], %r6;
SKIP:
    bar.sync 0;
    shr.u32 %r3, %r3, 1;
    setp.ne.u32 %p, %r3, 0;
    @%p bra LOOP;
    setp.ne.u32 %p, %r1, 0;
    @%p bra END;
    ld.shared.u32 %r7, [sm];
    st.global.u32 [%rd2], %r7;
END:
    ret;
}
''', [('IN', 'u32', v), ('OUT', 'u32', 1)], '.launch k <<< 1, 256 >>> (IN, OUT);', [('OUT', 'u32', [sum(v)])])
# 4 warp inclusive scan with shfl.up
v = [random.randint(0, 100) for _ in range(64)]
sc = []
for w in range(2):
    acc = 0
    for i in range(32): acc += v[w * 32 + i]; sc.append(acc)
emit('warpscan', '''.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .reg .pred %p; .reg .b32 %r<10>; .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd1, %rd3; add.s64 %rd5, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    and.b32 %r3, %r1, 31;
    mov.u32 %r4, 1;
LOOP:
    shfl.sync.up.b32 %r5|%p, %r2, %r4, 0, 0xffffffff;
    setp.ge.u32 %p, %r3, %r4;
    @%p add.u32 %r2, %r2, %r5;
    shl.b32 %r4, %r4, 1;
    setp.lt.u32 %p, %r4, 32;
    @%p bra LOOP;
    st.global.u32 [%rd5], %r2;
    ret;
}
''', [('IN', 'u32', v), ('OUT', 'u32', 64)], '.launch k <<< 1, 64 >>> (IN, OUT);', [('OUT', 'u32', sc)])
# 5 histogram: shared bins, global merge
d = [random.randint(0, 15) for _ in range(128)]
h = [d.count(i) for i in range(16)]
emit('histogram', '''.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .shared .align 4 .b32 bins[16];
    .reg .pred %p; .reg .b32 %r<8>; .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p, %r1, 16;
    @%p bra GO;
    mul.wide.u32 %rd3, %r1, 4; mov.u64 %rd4, bins; add.s64 %rd4, %rd4, %rd3;
    st.shared.u32 [%rd4], 0;
GO:
    bar.sync 0;
    mul.wide.u32 %rd3, %r1, 4; add.s64 %rd5, %rd1, %rd3;
    ld.global.u32 %r2, [%rd5];
    mul.wide.u32 %rd6, %r2, 4; mov.u64 %rd7, bins; add.s64 %rd7, %rd7, %rd6;
    atom.shared.add.u32 %r3, [%rd7], 1;
    bar.sync 0;
    @%p bra END;
    mul.wide.u32 %rd3, %r1, 4; mov.u64 %rd4, bins; add.s64 %rd4, %rd4, %rd3;
    ld.shared.u32 %r4, [%rd4];
    add.s64 %rd5, %rd2, %rd3;
    atom.global.add.u32 %r5, [%rd5], %r4;
END:
    ret;
}
''', [('IN', 'u32', d), ('OUT', 'u32', 16)], '.launch k <<< 1, 128 >>> (IN, OUT);', [('OUT', 'u32', h)])
# 6 tiled matmul 8x8 (one block, shared tiles), u32
A = [[random.randint(0, 9) for _ in range(8)] for _ in range(8)]; B = [[random.randint(0, 9) for _ in range(8)] for _ in range(8)]
Cm = [[sum(A[i][k] * B[k][j] for k in range(8)) for j in range(8)] for i in range(8)]
fl = lambda m: [x for r in m for x in r]
emit('matmul', '''.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc)
{
    .shared .align 4 .b32 sa[64];
    .shared .align 4 .b32 sb[64];
    .reg .pred %p; .reg .b32 %r<16>; .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc];
    mov.u32 %r1, %tid.x;                 // row-major id: row = id/8, col = id%8
    shr.u32 %r2, %r1, 3; and.b32 %r3, %r1, 7;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4; add.s64 %rd6, %rd2, %rd4;
    ld.global.u32 %r4, [%rd5]; ld.global.u32 %r5, [%rd6];
    mov.u64 %rd7, sa; add.s64 %rd7, %rd7, %rd4; st.shared.u32 [%rd7], %r4;
    mov.u64 %rd8, sb; add.s64 %rd8, %rd8, %rd4; st.shared.u32 [%rd8], %r5;
    bar.sync 0;
    mov.u32 %r6, 0; mov.u32 %r7, 0;
LOOP:
    shl.b32 %r8, %r2, 3; add.u32 %r8, %r8, %r6;      // A[row][k]
    shl.b32 %r9, %r6, 3; add.u32 %r9, %r9, %r3;      // B[k][col]
    mul.wide.u32 %rd9, %r8, 4; mov.u64 %rd10, sa; add.s64 %rd10, %rd10, %rd9; ld.shared.u32 %r10, [%rd10];
    mul.wide.u32 %rd9, %r9, 4; mov.u64 %rd10, sb; add.s64 %rd10, %rd10, %rd9; ld.shared.u32 %r11, [%rd10];
    mad.lo.u32 %r7, %r10, %r11, %r7;
    add.u32 %r6, %r6, 1;
    setp.lt.u32 %p, %r6, 8;
    @%p bra LOOP;
    add.s64 %rd11, %rd3, %rd4;
    st.global.u32 [%rd11], %r7;
    ret;
}
''', [('A', 'u32', fl(A)), ('B', 'u32', fl(B)), ('C', 'u32', 64)], '.launch k <<< 1, 64 >>> (A, B, C);', [('C', 'u32', fl(Cm))])
# 7 transpose via padded shared tile
T = [[random.randint(0, 255) for _ in range(8)] for _ in range(8)]
emit('transpose', '''.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .shared .align 4 .b32 tile[72];
    .reg .b32 %r<10>; .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x; shr.u32 %r2, %r1, 3; and.b32 %r3, %r1, 7;
    mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd1, %rd3; ld.global.u32 %r4, [%rd4];
    mad.lo.u32 %r5, %r2, 9, %r3;                    // tile[row][col], pitch 9
    mul.wide.u32 %rd5, %r5, 4; mov.u64 %rd6, tile; add.s64 %rd6, %rd6, %rd5;
    st.shared.u32 [%rd6], %r4;
    bar.sync 0;
    mad.lo.u32 %r6, %r3, 9, %r2;                    // tile[col][row]
    mul.wide.u32 %rd5, %r6, 4; mov.u64 %rd6, tile; add.s64 %rd6, %rd6, %rd5;
    ld.shared.u32 %r7, [%rd6];
    add.s64 %rd7, %rd2, %rd3; st.global.u32 [%rd7], %r7;
    ret;
}
''', [('IN', 'u32', fl(T)), ('OUT', 'u32', 64)], '.launch k <<< 1, 64 >>> (IN, OUT);', [('OUT', 'u32', fl([[T[j][i] for j in range(8)] for i in range(8)]))])
# 8 dot product: fma per thread, shuffle-down warp reduce, atomic add.f32
p = [float(random.randint(-4, 4)) for _ in range(64)]; q = [float(random.randint(-4, 4)) for _ in range(64)]
emit('dot', '''.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pout)
{
    .reg .pred %p; .reg .b32 %r<8>; .reg .b64 %rd<8>; .reg .f32 %f<6>;
    ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pout];
    mov.u32 %r1, %tid.x; mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4; add.s64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5]; ld.global.f32 %f2, [%rd6];
    mul.f32 %f3, %f1, %f2;
    mov.u32 %r2, 16;
LOOP:
    shfl.sync.down.b32 %f4, %f3, %r2, 31, 0xffffffff;
    add.f32 %f3, %f3, %f4;
    shr.u32 %r2, %r2, 1;
    setp.ne.u32 %p, %r2, 0;
    @%p bra LOOP;
    and.b32 %r3, %r1, 31;
    setp.ne.u32 %p, %r3, 0;
    @%p bra END;
    atom.global.add.f32 %f5, [%rd3], %f3;
END:
    ret;
}
''', [('A', 'u32', [f2b(x) for x in p]), ('B', 'u32', [f2b(x) for x in q]), ('OUT', 'u32', [0])], '.launch k <<< 1, 64 >>> (A, B, OUT);',
 [('OUT', 'u32', [f2b(sum(x * y for x, y in zip(p, q)))])])
# 9 Hillis-Steele inclusive scan in shared memory
v = [random.randint(0, 20) for _ in range(64)]
acc = 0; sc = []
for x in v: acc += x; sc.append(acc)
emit('scan', '''.visible .entry k(.param .u64 pin, .param .u64 pout)
{
    .shared .align 4 .b32 buf[64];
    .reg .pred %p; .reg .b32 %r<10>; .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pin]; ld.param.u64 %rd2, [pout];
    mov.u32 %r1, %tid.x; mul.wide.u32 %rd3, %r1, 4; add.s64 %rd4, %rd1, %rd3;
    ld.global.u32 %r2, [%rd4];
    mov.u64 %rd5, buf; add.s64 %rd5, %rd5, %rd3; st.shared.u32 [%rd5], %r2;
    bar.sync 0;
    mov.u32 %r3, 1;
LOOP:
    mov.u32 %r4, 0;
    setp.lt.u32 %p, %r1, %r3;
    @%p bra NOADD;
    sub.u32 %r5, %r1, %r3; mul.wide.u32 %rd6, %r5, 4; mov.u64 %rd7, buf; add.s64 %rd7, %rd7, %rd6;
    ld.shared.u32 %r4, [%rd7];
NOADD:
    ld.shared.u32 %r6, [%rd5];
    bar.sync 0;
    add.u32 %r6, %r6, %r4;
    st.shared.u32 [%rd5], %r6;
    bar.sync 0;
    shl.b32 %r3, %r3, 1;
    setp.lt.u32 %p, %r3, 64;
    @%p bra LOOP;
    ld.shared.u32 %r8, [%rd5];
    add.s64 %rd6, %rd2, %rd3; st.global.u32 [%rd6], %r8;
    ret;
}
''', [('IN', 'u32', v), ('OUT', 'u32', 64)], '.launch k <<< 1, 64 >>> (IN, OUT);', [('OUT', 'u32', sc)])
# 10 function call: device function computing gcd by Euclid, applied per thread
import math
ga = [random.randint(1, 200) for _ in range(32)]; gb = [random.randint(1, 200) for _ in range(32)]
emit('gcd', '''.func (.param .b32 ret) gcd(.param .b32 pa, .param .b32 pb)
{
    .reg .pred %p; .reg .b32 %r<5>;
    ld.param.b32 %r1, [pa]; ld.param.b32 %r2, [pb];
L:
    setp.eq.u32 %p, %r2, 0;
    @%p bra D;
    rem.u32 %r3, %r1, %r2; mov.u32 %r1, %r2; mov.u32 %r2, %r3;
    bra L;
D:
    st.param.b32 [ret], %r1;
    ret;
}
.visible .entry k(.param .u64 pa, .param .u64 pb, .param .u64 pc)
{
    .reg .b32 %r<6>; .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pa]; ld.param.u64 %rd2, [pb]; ld.param.u64 %rd3, [pc];
    mov.u32 %r1, %tid.x; mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd1, %rd4; add.s64 %rd6, %rd2, %rd4; add.s64 %rd7, %rd3, %rd4;
    ld.global.u32 %r2, [%rd5]; ld.global.u32 %r3, [%rd6];
    {
        .param .b32 pr; .param .b32 qa; .param .b32 qb;
        st.param.b32 [qa], %r2; st.param.b32 [qb], %r3;
        call (pr), gcd, (qa, qb);
        ld.param.b32 %r4, [pr];
    }
    st.global.u32 [%rd7], %r4;
    ret;
}
''', [('A', 'u32', ga), ('B', 'u32', gb), ('C', 'u32', 32)], '.launch k <<< 1, 32 >>> (A, B, C);', [('C', 'u32', [math.gcd(a, b) for a, b in zip(ga, gb)])])

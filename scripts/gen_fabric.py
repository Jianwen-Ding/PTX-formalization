#!/usr/bin/env python3
"""Generate tests/fabric.ptx: fabric.try_get/put/red/pullred/atom, submit/wait and the tensor forms on logical endpoints."""
import struct
M32 = (1 << 32) - 1
def f32b(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def f32v(b): return struct.unpack('<f', struct.pack('<I', b & M32))[0]
SZ = 512
F1 = bytearray(SZ); G = [bytearray(SZ) for _ in range(3)]
def w(buf, off, n, v): buf[off:off + n] = (v & ((1 << 8 * n) - 1)).to_bytes(n, 'little')
def r(buf, off, n): return int.from_bytes(buf[off:off + n], 'little')
for i in range(SZ // 4):
    w(F1, 4 * i, 4, i * 7 + 5)
    for d in range(3): w(G[d], 4 * i, 4, f32b(0.5 * (d + 1) + i * 0.25) if i >= 32 else (i * 11 + d * 1000) & M32)
init = [bytes(F1)] + [bytes(g) for g in G]
smem = bytearray(512)
body, exp = [], []
def emit(reg, val): body.append(f"    st.global.b32 [%o+{4 * len(exp)}], {reg};"); exp.append(val & M32)
def dump(off, n):
    for i in range(n):
        body.append(f"    ld.shared.u32 %r0, [data+{off + 4 * i}];"); emit("%r0", r(smem, off + 4 * i, 4))
ph = [0]
def wait():
    body.append("    fabric.submit;")
    body.append(f"    mbarrier.try_wait.parity.shared.b64 %p, [bar], {ph[0]};"); ph[0] ^= 1
    body.append("    selp.u32 %r0, 1, 0, %p;"); emit("%r0", 1)
def expect(n): body.append(f"    mbarrier.arrive.expect_tx.shared.b64 _, [bar], {n};")
body.append("    mbarrier.init.shared.layout::v1.b64 [bar], 1;")
# A: try_get 64 bytes from LE1 offset 32
body.append("    mov.u32 %le1, 1; mov.u64 %off, 32; mov.u32 %le2, 2;")
expect(64)
body.append("    fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128 [data], [%le1, %off], 64, [bar];")
smem[0:64] = F1[32:96]
wait(); dump(0, 16)
# B: try_put counted: smem[0..64) -> LE1 off 128, counter at 256
body.append("    mov.u64 %off, 128; mov.u64 %coff, 256;")
expect(4)
body.append("    fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128 [%le1, %off, %coff], [data], 64, [bar];")
F1[128:192] = smem[0:64]; w(F1, 256, 8, r(F1, 256, 8) + 64)
wait()
# C: multimem masked put (low 8 bytes of every 16): smem[64..96) -> LE2 off 64
for i in range(8): body.append(f"    st.shared.u32 [data+{64 + 4 * i}], {0x11111111 * (i + 1) & M32};"); w(smem, 64 + 4 * i, 4, 0x11111111 * (i + 1))
body.append("    mov.u64 %off, 64; mov.u32 %mask, 0x00FF;")
expect(2)
body.append("    fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask.relaxed.sys.b128 [%le2, %off], [data+64], 32, [bar], %mask;")
for d in range(3):
    for c in range(2):
        for b in range(8): G[d][64 + 16 * c + b] = smem[64 + 16 * c + b]
wait()
# D: try_red add.u32 into LE1 off 128 (8 words of smem[64..96)), multimem red max.s32 on LE2 off 0
body.append("    mov.u64 %off, 128;")
expect(2)
body.append("    fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32 [%le1, %off], [data+64], 32, [bar];")
for i in range(8): w(F1, 128 + 4 * i, 4, r(F1, 128 + 4 * i, 4) + r(smem, 64 + 4 * i, 4))
wait()
for i in range(8): body.append(f"    st.shared.s32 [data+{128 + 4 * i}], {(i * 37 - 100)};"); w(smem, 128 + 4 * i, 4, i * 37 - 100)
body.append("    mov.u64 %off, 0;")
expect(2)
body.append("    fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.max.s32 [%le2, %off], [data+128], 32, [bar];")
sg = lambda x: x - (1 << 32) if x >> 31 else x
for d in range(3):
    for i in range(8): w(G[d], 4 * i, 4, max(sg(r(G[d], 4 * i, 4)), sg(r(smem, 128 + 4 * i, 4))))
wait()
# E: try_pullred add.f32 from LE2 off 128 (floats), 32 bytes -> smem[192..224)
body.append("    mov.u64 %off, 128;")
expect(32)
body.append("    fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.f32.sync [data+192], [%le2, %off], 32, [bar], 0xFFFFFFFF;")
for i in range(8):
    s = f32v(r(G[0], 128 + 4 * i, 4)) + f32v(r(G[1], 128 + 4 * i, 4)) + f32v(r(G[2], 128 + 4 * i, 4))
    w(smem, 192 + 4 * i, 4, f32b(s))
wait(); dump(192, 8)
# F: try_atom add.u32 at LE1 off 20; old value to smem_dst+4
body.append("    st.shared.u32 [data+256], 0; st.shared.u32 [data+260], 100; st.shared.u32 [data+264], 0; st.shared.u32 [data+268], 0;")
w(smem, 260, 4, 100)
body.append("    mov.u64 %off, 20;")
expect(1)
body.append("    fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.add.u32 [%le1, %off], [data+288], [data+256], [bar];")
# smem_dst = data+288, smem_src = data+256 : src element at +4
w(smem, 288 + 4, 4, r(F1, 20, 4)); w(F1, 20, 4, r(F1, 20, 4) + 100)
wait(); body.append("    ld.shared.u32 %r0, [data+292];"); emit("%r0", r(smem, 292, 4))
# cas.b32: compare at src+4 (value F1[36]), swap at src+16+4
old = r(F1, 36, 4)
body.append(f"    st.shared.u32 [data+260], {old}; st.shared.u32 [data+276], 777;")
w(smem, 260, 4, old); w(smem, 276, 4, 777)
body.append("    mov.u64 %off, 36;")
expect(1)
body.append("    fabric.try_atom.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.cas.b32 [%le1, %off], [data+320], [data+256], [bar];")
w(smem, 324, 4, old); w(F1, 36, 4, 777)
wait(); body.append("    ld.shared.u32 %r0, [data+324];"); emit("%r0", r(smem, 324, 4))
# G: tensor get: 4x4 u32 tensor at LE1 off 0, box 2x2 at (1,1), override dims {4,4}
body.append("    mov.u64 %off, 0; mov.s32 %c0, 1; mov.s32 %c1, 1; mov.u32 %d0, 4; mov.u32 %d1, 4;")
expect(16)
body.append("    fabric.try_get.async.tensor.2d.shared::cta.tile.mbarrier::complete_tx::bytes.mbarrier::report::fabric.override::global_address.override::global_dim.relaxed.sys.b128 [data+352], [TM, %le1, %off, {%d0, %d1}, {%c0, %c1}], [bar];")
for j in range(2):
    for i in range(2): w(smem, 352 + 4 * (2 * j + i), 4, r(F1, (c1 := 1 + j) * 16 + (1 + i) * 4, 4))
wait(); dump(352, 4)
# tensor put multimem: the 2x2 box back at (2,2) of every device (LE2 off 0), size 16
body.append("    mov.s32 %c0, 2; mov.s32 %c1, 2;")
expect(1)
body.append("    fabric.try_put.async.tensor.multimem.2d.shared::cta.tile.mbarrier::complete_tx::16B.mbarrier::report::fabric.override::global_address.override::global_dim.relaxed.sys.b128 [TM, %le2, %off, {%d0, %d1}, {%c0, %c1}], [data+352], 16, [bar];")
for d in range(3):
    for j in range(2):
        for i in range(2): w(G[d], (2 + j) * 16 + (2 + i) * 4, 4, r(smem, 352 + 4 * (2 * j + i), 4))
wait()
body.append("    fabric.wait.sync_restrict::reads;")
ptx = f'''// fabric.* on logical endpoints: try_get / try_put (counted, masked, multimem) / try_red / try_pullred / try_atom and tensor forms.
// GENERATED by scripts/gen_fabric.py
.version 9.4
.target sm_100a
.address_size 64

.visible .entry k(.param .u64 po, .param .u64 ptm)
{{
    .reg .b64 %o, %off, %coff, %tm; .reg .b32 %r0, %mask, %c0, %c1, %d0, %d1; .reg .u32 %le1, %le2; .reg .pred %p;
    .shared .align 16 .b8 data[512];
    .shared .align 8 .b64 bar;
    ld.param.u64 %o, [po]; ld.param.u64 %tm, [ptm];
''' + "\n".join(body).replace("[TM,", "[%tm,") + '''
    ret;
}
'''
ptx += ".buffer BF1 .u8 = { " + ", ".join(map(str, init[0])) + " };\n"
for d in range(3): ptx += f".buffer BG{d} .u8 = {{ " + ", ".join(map(str, init[1 + d])) + " };\n"
ptx += ".buffer BTM .u32 [16];\n.tensormap TM .u32 BTM .dims { 4, 4 } .strides { 16 } .box { 2, 2 } .swizzle 0 ;\n"
ptx += f".fabric 1 .size {SZ} {{ BF1 }} ;\n.fabric 2 .size {SZ} {{ BG0, BG1, BG2 }} ;\n"
ptx += f".buffer O .u32 [{len(exp)}];\n.launch k <<< 1, 1 >>> (O, TM);\n"
ptx += f".expect O .u32 = {{ {', '.join(map(str, exp))} }};\n"
ptx += ".expect BF1 .u8 = { " + ", ".join(map(str, F1)) + " };\n"
for d in range(3): ptx += f".expect BG{d} .u8 = {{ " + ", ".join(map(str, G[d])) + " };\n"
open('tests/fabric.ptx', 'w').write(ptx)
print(len(exp), "values", len(ptx))

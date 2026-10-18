#!/usr/bin/env python3
"""Hardware differential testing: run the same PTX test on an NVIDIA GPU and on the K semantics and compare.

  hwdiff.py emit  test.ptx outdir     split a harness test into outdir/kernel.ptx and a CUDA-driver host program outdir/host.c
                                      (build with: cc host.c -lcuda -o host; run: ./host > hw.txt on a machine with a GPU)
  hwdiff.py k     test.ptx            print the buffers after running the K semantics (same format as the host program)
  hwdiff.py cmp   test.ptx hw.txt     compare the K result with a hardware result; floats may differ by --ulps units in the last place

The harness directives understood: .buffer (sized or initialised), .launch k <<<grid, block>>> (buffers...), .print is generated.
Tests that use other directives (.memmodel, textures, clusters, ...) must be run by hand.  No GPU is available in the
development environment, so the emit step is checked only by compiling host.c against a stub header (hwdiff/cuda_stub.h).
"""
import os, re, struct, subprocess, sys

DIRECTIVES = ("// ", ".buffer", ".launch", ".expect", ".print", ".patch", ".append", ".memmodel", ".isacheck", ".ub", ".occupancy", ".convergence")
TY = {"u8": ("uint8_t", 1, "%u"), "s8": ("int8_t", 1, "%d"), "u16": ("uint16_t", 2, "%u"), "s16": ("int16_t", 2, "%d"),
      "u32": ("uint32_t", 4, "%u"), "s32": ("int32_t", 4, "%d"), "u64": ("uint64_t", 8, "%llu"), "s64": ("int64_t", 8, "%lld"),
      "b32": ("uint32_t", 4, "%u"), "b64": ("uint64_t", 8, "%llu"), "f32": ("uint32_t", 4, "%u"), "f64": ("uint64_t", 8, "%llu"),
      "b16": ("uint16_t", 2, "%u"), "b8": ("uint8_t", 1, "%u")}

def parse(path):
    text = open(path).read()
    bufs, launches, dev = [], [], []
    for line in text.split("\n"):
        s = line.strip()
        m = re.match(r"\.buffer\s+(\w+)\s+\.(\w+)\s*(?:\[\s*(\d+)\s*\])?\s*(?:=\s*\{([^}]*)\})?\s*;", s)
        if m:
            name, ty, n, init = m.groups()
            vals = [parse_val(v.strip(), ty) for v in init.split(",") if v.strip()] if init else []
            bufs.append((name, ty, int(n) if n else len(vals), vals)); continue
        m = re.match(r"\.launch\s+(\w+)\s*<<<\s*(.+?)\s*,\s*(.+?)\s*>>>\s*\((.*)\)\s*;", s)
        if m:
            launches.append((m.group(1), dim(m.group(2)), dim(m.group(3)), [a.strip() for a in m.group(4).split(",") if a.strip()])); continue
        if s.startswith(DIRECTIVES) and not s.startswith("//"): continue
        dev.append(line)
    return bufs, launches, "\n".join(dev)

def dim(s):
    nums = [int(x) for x in re.findall(r"\d+", s)]
    return (nums + [1, 1, 1])[:3]

def parse_val(v, ty):
    if v.startswith(("0x", "0X")): return int(v, 16)
    if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", v) or re.fullmatch(r"0[dD][0-9a-fA-F]{16}", v): return int(v[2:], 16)
    if ty == "f32" and re.match(r"^-?[0-9.]+(e[-+]?\d+)?$", v): return struct.unpack("<I", struct.pack("<f", float(v)))[0]
    if ty == "f64" and re.match(r"^-?[0-9.]+(e[-+]?\d+)?$", v): return struct.unpack("<Q", struct.pack("<d", float(v)))[0]
    return int(v) & ((1 << (8 * TY[ty][1])) - 1)

def emit(path, out):
    bufs, launches, dev = parse(path)
    if not launches: sys.exit("no .launch in " + path)
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "kernel.ptx"), "w").write(dev + "\n")
    c = ['#include <cuda.h>', '#include <stdio.h>', '#include <stdint.h>', '#include <stdlib.h>',
         '#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { fprintf(stderr, "CUDA error %d at line %d\\n", r, __LINE__); return 1; } } while (0)',
         'int main(void) {', '  CUdevice d; CUcontext cx; CUmodule m; CUfunction f;',
         '  CK(cuInit(0)); CK(cuDeviceGet(&d, 0)); CK(cuCtxCreate(&cx, 0, d));',
         '  FILE *fp = fopen("kernel.ptx", "rb"); fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);',
         '  char *src = malloc(n + 1); fread(src, 1, n, fp); src[n] = 0; fclose(fp);',
         '  CK(cuModuleLoadData(&m, src));']
    for name, ty, n, vals in bufs:
        ct, sz, _ = TY[ty]
        c.append(f'  {ct} h_{name}[{max(n,1)}] = {{{", ".join(str(v) + "ull" for v in vals) or "0"}}}; CUdeviceptr d_{name};')
        c.append(f'  CK(cuMemAlloc(&d_{name}, {max(n,1) * sz})); CK(cuMemcpyHtoD(d_{name}, h_{name}, {max(n,1) * sz}));')
    for k, g, b, args in launches:
        c.append('  {')
        c.append(f'  CK(cuModuleGetFunction(&f, m, "{k}"));')
        names = [a for a in args if a in [x[0] for x in bufs]]
        ptrs = ", ".join(f"&d_{a}" if a in names else f"&arg{i}" for i, a in enumerate(args))
        for i, a in enumerate(args):
            if a not in names: c.append(f'  unsigned long long arg{i} = {int(a, 0)}ull;')
        c.append(f'  void *params[] = {{{ptrs or "0"}}};')
        c.append(f'  CK(cuLaunchKernel(f, {g[0]}, {g[1]}, {g[2]}, {b[0]}, {b[1]}, {b[2]}, 0, 0, params, 0)); CK(cuCtxSynchronize());')
        c.append('  }')
    for name, ty, n, vals in bufs:
        ct, sz, fmt = TY[ty]
        c.append(f'  CK(cuMemcpyDtoH(h_{name}, d_{name}, {max(n,1) * sz}));')
        c.append(f'  printf("{name} = ["); for (int i = 0; i < {n}; i++) printf(i ? ", {fmt}" : "{fmt}", ({"unsigned long long" if sz == 8 else "unsigned"})h_{name}[i]); printf("]\\n");')
    c += ["  return 0;", "}"]
    open(os.path.join(out, "host.c"), "w").write("\n".join(c) + "\n")
    print("wrote", out)

def kvals(path):
    bufs, launches, dev = parse(path)
    src = open(path).read() + "".join(f"\n.print {n} .{ty} {max(cnt,1)} ;" for n, ty, cnt, _ in bufs)
    tmp = "/tmp/hwdiff_k.ptx"
    # .print is only legal after the .launch: strip existing .expect lines and append
    lines = [l for l in open(path).read().split("\n") if not l.strip().startswith(".expect")]
    open(tmp, "w").write("\n".join(lines) + "".join(f"\n.print {n} .{ty} {max(cnt,1)} ;" for n, ty, cnt, _ in bufs) + "\n")
    out = subprocess.run(["krun", "-d", os.environ.get("PTX_DEF", "build/ptx-kompiled"), tmp, "--output", "pretty"], capture_output=True, text=True,
                         env={**os.environ, "K_OPTS": "-Xss512m"}).stdout
    res = {}
    for l in out.splitlines():
        m = re.match(r"^(\w+) = \[(.*)\]$", l)
        if m: res[m.group(1)] = [int(x) for x in m.group(2).split(",") if x.strip()]
    return res

def cmp(path, hwfile, ulps=0):
    k = kvals(path)
    hw = {}
    for l in open(hwfile):
        m = re.match(r"^(\w+) = \[(.*)\]$", l.strip())
        if m: hw[m.group(1)] = [int(x) for x in m.group(2).split(",") if x.strip()]
    bufs = {b[0]: b for b in parse(path)[0]}
    bad = 0
    for name in k:
        a, b = k[name], hw.get(name)
        if b is None: print("MISSING", name); bad += 1; continue
        ty = bufs[name][1]
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y and not (ty in ("f32", "f64") and abs(x - y) <= ulps):
                print(f"DIFF {name}[{i}]: K {x:#x} hardware {y:#x}"); bad += 1
    print("---- %s" % ("identical" if not bad else f"{bad} differences")); return bad == 0

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "emit": emit(a[1], a[2])
    elif a[0] == "k":
        for n, v in kvals(a[1]).items(): print(f"{n} = [{', '.join(map(str, v))}]")
    elif a[0] == "cmp": sys.exit(0 if cmp(a[1], a[2], int(a[3]) if len(a) > 3 else 0) else 1)
    else: sys.exit(__doc__)

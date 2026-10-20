#!/usr/bin/env python3
"""The Mesa subset of the AGX Sail model.

Mesa's asahi compiler emits a subset of the instruction set: the forms that llvm/utils/agx-mesa-isa/mesa_isa.py in
the LLVM AGX fork reads off its packer (src/asahi/compiler/agx_pack.c). That subset is what the LLVM backend's
mesa-isa feature and the AGX Performance Simulator's --mesa-isa option allow. This checks the Sail model against it
in both directions:

  forms     every instruction agxgen generates (each form one factor at a time, plus random ones) is encoded by the
            Sail model (encdec) and classified by mesa_isa.py: the Sail forms Mesa emits, and those it never does
  corpus    instructions packed by Mesa's own agx_pack.c (the simulator's rust/src/isa/mesa_corpus.txt, or
            mesa_isa.py's tools/fuzz.sh output) are decoded by the Sail model: per Mesa opcode, how many of Mesa's
            instructions the model describes (decodes, at the right length). The corpus has random operands over
            everything Mesa packs (all memory formats, every register and uniform), so this is the coverage of Mesa's
            subset by the model; an instruction decoded at a different length is an error (the exit status)

usage: mesacheck.py [-s SEED] [--reps R] [-n RANDOM] [--gen g13|g13x|g14|g14d] [--mesa-isa DIR] [--corpus FILE]
                    [--json FILE]
"""
import argparse, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.dont_write_bytecode = True
import agxgen  # noqa: E402
import agxtest  # noqa: E402
from mccheck import sail_encode, sail_decode  # noqa: E402

DEFAULT_MESA_ISA = os.environ.get("MESA_ISA_DIR", "/home/user/llvm-project/llvm/utils/agx-mesa-isa")
DEFAULT_CORPUS = os.environ.get("MESA_CORPUS",
                                "/home/user/AGX-Performance-Simulator/rust/src/isa/mesa_corpus.txt")

# Mesa opcodes the Sail model does not describe: graphics (tile, varyings, textures, images, depth/stencil, sample
# masks), the preamble's uniform store, the stack mapping instructions and the doorbell. Compute kernels need none.
NOT_MODELLED = {
    "ld_tile", "st_tile", "sample_mask", "zs_emit", "iter", "iterproj", "ldcf", "st_vary", "no_varyings",
    "texture_sample", "texture_load", "image_load", "image_write", "block_image_store", "tex_state_store",
    "sampler_state_store", "wait_pix", "signal_pix", "uniform_store", "stack_map", "stack_unmap", "doorbell",
    "dfdx", "dfdy", "image_barrier_1", "image_barrier_2", "image_barrier_3", "image_barrier_4",
    "flush_memory_to_texture",
}


def load_mesa_isa(path):
    sys.path.insert(0, path)
    import mesa_isa  # noqa: E402
    return mesa_isa


def classify(mesa_isa, gen, hexes):
    out = []
    for h in hexes:
        data = bytes.fromhex(h)
        try:
            name, _ = mesa_isa.check(data, gen, len(data))
            out.append((True, name))
        except ValueError as e:
            out.append((False, str(e)))
    return out


def forms_part(a, mesa_isa, results):
    R = random.Random(a.seed)
    instrs = agxgen.forms(R, reps=a.reps) + [agxgen.random_instr(R) for _ in range(a.n)]
    # the model reads at most a few thousand instructions per run
    enc = []
    for i in range(0, len(instrs), 1000):
        enc += sail_encode(instrs[i:i + 1000])
    keys = {}
    mesa_forms = {}
    hexes, kept = [], []
    for t, e in zip(instrs, enc):
        if e is None:
            continue
        hexes.append(e[0])
        kept.append(t)
    for t, h, (ok, what) in zip(kept, hexes, classify(mesa_isa, a.gen, hexes)):
        key = agxgen.form_key(t)
        k = keys.setdefault(key, [0, 0, None])
        k[0 if ok else 1] += 1
        if ok:
            mesa_forms.setdefault(what.split(".")[0], set()).add(key.split()[0])
        elif k[2] is None:
            k[2] = (h, what)
    in_mesa = sorted(k for k, v in keys.items() if v[0])
    never = sorted(k for k, v in keys.items() if not v[0])
    results["forms"] = {
        "instructions": len(hexes), "forms": len(keys), "forms_mesa_emits": len(in_mesa),
        "forms_mesa_never_emits": len(never), "mesa_opcodes_reached": sorted(mesa_forms),
        "never": {k: keys[k][2] for k in never},
    }
    groups = {}
    for k in never:
        groups.setdefault(k.split()[0], []).append(k)
    print("Sail forms (seed %d): %d instructions, %d distinct forms; %d have instructions Mesa emits for %s, "
          "%d never do" % (a.seed, len(hexes), len(keys), len(in_mesa), a.gen, len(never)))
    print("  Mesa opcodes reached by the Sail model: %d (%s)" % (len(mesa_forms), ", ".join(sorted(mesa_forms))))
    for g, ks in sorted(groups.items(), key=lambda x: -len(x[1]))[:a.max_print]:
        h, why = keys[ks[0]][2]
        print("  never Mesa: %-20s %4d forms, e.g. %s (%s: %s)" % (g, len(ks), ks[0], h, why))


def corpus_part(a, mesa_isa, results):
    lines = [l.split() for l in open(a.corpus) if l.strip() and not l.startswith("#")]
    cases = [(g, op, h) for g, op, h in lines if g == a.gen]
    modelled = [(op, h) for _, op, h in cases if op.replace("_coverage", "").replace("_barrier", "")
                not in NOT_MODELLED and len(h) <= 20]
    words = [int.from_bytes(bytes.fromhex(h), "little") for _, h in modelled]
    decs = []
    for i in range(0, len(words), 1000):
        decs += sail_decode(words[i:i + 1000])
    per = {}
    for (op, h), d in zip(modelled, decs):
        p = per.setdefault(op, [0, 0, 0, None])
        if d is None:
            p[1] += 1
            p[3] = p[3] or (h, "not decoded")
        elif d[0] != len(h) // 2:
            p[2] += 1
            p[3] = p[3] or (h, "decoded as %d bytes: %s" % (d[0], d[1]))
        else:
            p[0] += 1
    ok = sum(p[0] for p in per.values())
    bad = {op: p for op, p in per.items() if p[1] or p[2]}
    wrong = {op: p for op, p in per.items() if p[2]}
    results["corpus"] = {
        "instructions": len(cases), "modelled": len(modelled), "decoded": ok,
        "opcodes": len(per), "opcodes_not_modelled": sorted({op for _, op, _ in cases} - set(per)),
        "complete": sorted(op for op in per if op not in bad),
        "partial": {op: {"decoded": p[0], "undecoded": p[1], "wrong_length": p[2], "example": p[3]}
                    for op, p in bad.items()},
        "wrong_length": sorted(wrong),
    }
    print("Mesa corpus (%s, %s): %d instructions of %d opcodes; the Sail model decodes %d of the %d that are not "
          "graphics (%d opcodes complete, %d partial)"
          % (a.gen, os.path.basename(a.corpus), len(cases), len({op for _, op, _ in cases}), ok, len(modelled),
             len(per) - len(bad), len(bad)))
    print("  complete: " + ", ".join(sorted(op for op in per if op not in bad)))
    for op, p in sorted(bad.items(), key=lambda x: x[1][0] / max(1, sum(x[1][:3]))):
        print("  partial  %-20s %3d/%-3d decoded, e.g. not %s (%s)" % (op, p[0], sum(p[:3]), p[3][0], p[3][1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("-n", type=int, default=2000, help="random instructions")
    ap.add_argument("--gen", default="g14", choices=["g13", "g13x", "g14", "g14d"])
    ap.add_argument("--mesa-isa", default=DEFAULT_MESA_ISA, help="llvm/utils/agx-mesa-isa of the LLVM AGX fork")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--max-print", type=int, default=12)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    mesa_isa = load_mesa_isa(a.mesa_isa)
    results = {"gen": a.gen, "seed": a.seed}
    forms_part(a, mesa_isa, results)
    corpus_part(a, mesa_isa, results)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1, default=list)
    return 1 if results["corpus"]["wrong_length"] else 0


if __name__ == "__main__":
    sys.exit(main())

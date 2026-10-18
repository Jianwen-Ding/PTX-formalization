#!/usr/bin/env python3
"""Lower a restricted PTX kernel to Apple AGX (G13) assembly in the dialect of agx/agx.k.

usage: ptx2agx.py kernel.ptx            prints the AGX program body and a JSON description of the parameter / buffer mapping

Supported PTX (one kernel, 32-bit scalar data, forward-structured control flow):
  .param .u64 pointers (uniform registers u(2k),u(2k+1)) and .u32 scalars; %tid.x / %ntid.x / %ctaid.x (special registers)
  mov, add, sub, mul.lo, mad.lo, and, or, xor, not, shl/shr by an immediate, min, max, neg (u32 s32 b32)
  add, sub, mul, fma, neg (f32, round-to-nearest)
  setp (+ selp, + predicated instructions, + forward `@p bra L` / `@p ret` regions) -> compare/select and execution-mask stack
  ld.global / st.global (32-bit) with addresses of the form  param pointer + index * scale (+ displacement)
Everything else is rejected with an error (the lowering is a translation-validation target, not a compiler).
Differences from PTX that the lowering cannot remove are documented in docs/agx-mapping.md (f32 subnormals flush to zero on AGX).
"""
import json, re, struct, sys

class Reject(Exception): pass

INT_T = ("u32", "s32", "b32")
CC = {"eq": "eq", "ne": "ne", "lt": "slt", "le": "sle", "gt": "sgt", "ge": "sge", "lo": "ult", "ls": "ule", "hi": "ugt", "hs": "uge"}
FCC = {"eq": "feq", "ne": "fne", "lt": "flt", "le": "fle", "gt": "fgt", "ge": "fge"}
INV = {"eq": "ne", "ne": "eq", "ult": "uge", "uge": "ult", "ule": "ugt", "ugt": "ule", "slt": "sge", "sge": "slt", "sle": "sgt", "sgt": "sle"}
SR = {"%tid.x": 80, "%ntid.x": 81, "%ctaid.x": 82}

def fimm(tok):
    m = re.fullmatch(r"0[fF]([0-9a-fA-F]{8})", tok)
    if m:
        v = struct.unpack("<f", struct.pack("<I", int(m.group(1), 16)))[0]
        if v != v or v in (float("inf"), float("-inf")): raise Reject("float immediate " + tok)
        return repr(v) if "." in repr(v) or "e" in repr(v) else repr(v) + ".0"
    if re.fullmatch(r"-?\d+\.\d+(e[-+]?\d+)?", tok): return tok
    return None

class Lower:
    def __init__(self, text):
        self.lines = self.clean(text)
        self.params = []        # (name, kind)  kind u64 / u32
        self.regs = {}          # ptx register -> agx register number
        self.next = 8
        self.sym = {}           # u64 register -> ('ptr', param) | ('addr', param, idxreg(str|None), scale, disp)
        self.cond = {}          # predicate -> (kind, cc, A, B, versions)
        self.ver = {}
        self.out = []
        self.regions = []       # label stack of open if-regions
        self.uni = {}
        self.pending_wait = False

    @staticmethod
    def clean(text):
        text = re.sub(r"//[^\n]*", "", text)
        body = text[text.index("{", text.index(".entry")) + 1: text.rindex("}")] if ".entry" in text else text
        head = text[text.index(".entry"): text.index("{", text.index(".entry"))]
        lines = []
        for st in re.split(r";|\n", body):
            st = st.strip()
            if st: lines.append(st)
        return head, lines

    def reg(self, name):
        if name not in self.regs:
            self.regs[name] = self.next; self.next += 1
            if self.next > 127: raise Reject("out of registers")
        return self.regs[name]

    def operand(self, tok, ty):
        tok = tok.strip()
        if tok.startswith("%") and tok not in SR:
            return f"r{self.reg(tok)}"
        if ty == "f32":
            f = fimm(tok)
            if f is not None: return f
        if re.fullmatch(r"-?\d+", tok): return tok if int(tok) >= 0 else str(int(tok) & 0xFFFFFFFF)
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", tok): return str(int(tok, 16))
        raise Reject("operand " + tok)

    def emit(self, s): self.out.append(s)

    def define(self, name):
        self.ver[name] = self.ver.get(name, 0) + 1

    def lower(self):
        head, lines = self.lines
        for m in re.finditer(r"\.param\s+\.(u64|u32|s32|f32|b32)\s+(\w+)", head):
            k = len(self.params); self.params.append((m.group(2), m.group(1)))
        for st in lines:
            self.stmt(st)
        while self.regions:
            raise Reject("unterminated region")
        self.emit("stop")
        return self.out

    def stmt(self, st):
        if st.startswith("."): return                         # .reg / .shared declarations
        if st == "ret": return
        m = re.fullmatch(r"(\$?\w+):", st)
        if m:
            lab = m.group(1)
            if self.regions and self.regions[-1] == lab:
                self.regions.pop(); self.emit("pop_exec 1")
            elif lab in self.regions: raise Reject("improperly nested control flow at " + lab)
            return
        guard = None
        m = re.match(r"@(!?)(%\w+)\s+(.*)", st)
        if m: guard = (m.group(1) == "!", m.group(2)); st = m.group(3)
        # branches and returns open regions that skip to the label / the end
        mb = re.fullmatch(r"bra(?:\.uni)?\s+(\$?\w+)", st)
        if mb:
            if guard is None: raise Reject("unconditional branch")
            self.open_region(guard, mb.group(1)); return
        if st == "ret":
            return
        if guard is not None and st == "ret": raise Reject("guarded ret")
        if guard is not None:
            self.open_region((not guard[0], guard[1]), None)   # run the instruction only when the guard holds
            self.instr(st)
            self.emit("pop_exec 1")
            return
        self.instr(st)

    def open_region(self, guard, label):
        """@p bra L : skip to L when p holds, i.e. run the code between only when p does not hold."""
        neg, p = guard
        if p not in self.cond: raise Reject("predicate not produced by setp: " + p)
        kind, cc, a, b, vers = self.cond[p]
        for r, v in vers.items():
            if self.ver.get(r, 0) != v: raise Reject("predicate operand redefined: " + p)
        run_when_true = neg                                    # `@!p bra` skips when p is false -> body runs when p is true
        if kind == "f" and not run_when_true:
            raise Reject("negated float predicate (NaN semantics)")
        c = cc if run_when_true else INV[cc] if kind == "i" else INV_F[cc]
        self.emit(("if_icmp " if kind == "i" else "if_fcmp ") + f"{c} {a}, {b}")
        if label is not None: self.regions.append(label)

    def instr(self, st):
        m = re.fullmatch(r"([\w.]+)\s*(.*)", st)
        op, rest = m.group(1), [x.strip() for x in m.group(2).split(",")] if m.group(2) else []
        parts = op.split("."); base = parts[0]; ty = parts[-1]
        if base == "ld" and parts[1] == "param":
            name = rest[1].strip("[]")
            idx = [p[0] for p in self.params].index(name)
            kind = self.params[idx][1]
            if kind == "u64": self.sym[rest[0]] = ("ptr", idx)
            else:
                self.emit(f"mov r{self.reg(rest[0])}, u{2 * idx}"); self.define(rest[0])
                self.scalar_params = getattr(self, "scalar_params", {}); self.scalar_params[idx] = 2 * idx
            return
        if base == "mov" and rest[1] in SR:
            self.emit(f"get_sr r{self.reg(rest[0])}, {SR[rest[1]]}"); self.define(rest[0]); return
        if base == "mov" and ty == "u64":
            self.sym[rest[0]] = self.sym[rest[1]]; return
        if base == "cvt" and parts[-2:] == ["u64", "u32"] or (base == "cvt" and parts[1:3] == ["u64", "u32"]):
            self.sym[rest[0]] = ("idx", rest[1], 1, 0); return
        if base == "mul" and parts[1] == "wide":
            self.sym[rest[0]] = ("idx", rest[1], int(rest[2]), 0); return
        if base == "add" and ty == "u64":
            a, b = self.sym.get(rest[1]), self.sym.get(rest[2])
            if a and b:
                if a[0] == "ptr" and b[0] == "idx": self.sym[rest[0]] = ("addr", a[1], b[1], b[2], b[3]); return
                if b[0] == "ptr" and a[0] == "idx": self.sym[rest[0]] = ("addr", b[1], a[1], a[2], a[3]); return
            if a and a[0] == "addr" and re.fullmatch(r"\d+", rest[2]):
                self.sym[rest[0]] = ("addr", a[1], a[2], a[3], a[4] + int(rest[2])); return
            raise Reject("64-bit arithmetic: " + st)
        if base in ("ld", "st") and parts[1] == "global":
            return self.memop(base, ty, rest)
        if base == "setp":
            cmp_, t = parts[1], parts[2]
            a, b = self.operand(rest[1], t), self.operand(rest[2], t)
            if t == "f32":
                if cmp_ not in FCC: raise Reject("float compare " + cmp_)
                self.cond[rest[0]] = ("f", FCC[cmp_], a, b, self.vers(rest[1:3]))
            else:
                cc = CC.get(cmp_)
                if cc is None: raise Reject("compare " + cmp_)
                if t == "u32" and cmp_ in ("lt", "le", "gt", "ge"): cc = {"lt": "ult", "le": "ule", "gt": "ugt", "ge": "uge"}[cmp_]
                self.cond[rest[0]] = ("i", cc, a, b, self.vers(rest[1:3]))
            return
        if base == "selp":
            p = rest[3]
            kind, cc, a, b, vers = self.cond[p]
            t = ty
            x, y = self.operand(rest[1], t), self.operand(rest[2], t)
            self.emit(("icmpsel " if kind == "i" else "fcmpsel ") + f"{cc} r{self.reg(rest[0])}, {a}, {b}, {x}, {y}"); self.define(rest[0]); return
        self.alu(base, parts, rest, ty)

    def vers(self, toks): return {t: self.ver.get(t, 0) for t in toks if t.startswith("%")}

    def alu(self, base, parts, rest, ty):
        d = rest[0]
        if ty == "f32":
            a = self.operand(rest[1], ty)
            if base == "add": self.emit(f"fadd r{self.reg(d)}, {a}, {self.operand(rest[2], ty)}")
            elif base == "mul": self.emit(f"fmul r{self.reg(d)}, {a}, {self.operand(rest[2], ty)}")
            elif base == "fma": self.emit(f"fmadd r{self.reg(d)}, {a}, {self.operand(rest[2], ty)}, {self.operand(rest[3], ty)}")
            elif base == "sub": self.emit(f"fmadd r{self.reg(d)}, {self.operand(rest[2], ty)}, -1.0, {a}")
            elif base == "neg": self.emit(f"fmul r{self.reg(d)}, {a}, -1.0")
            elif base == "mov": self.emit(f"mov r{self.reg(d)}, {a}")
            else: raise Reject("f32 " + base)
            self.define(d); return
        if ty not in INT_T and not (base == "mul" and ty in INT_T): raise Reject(f"type .{ty} in {'.'.join(parts)}")
        a = self.operand(rest[1], ty) if len(rest) > 1 else None
        R = f"r{self.reg(d)}"
        if base == "mov": self.emit(f"mov {R}, {a}")
        elif base == "add": self.emit(f"iadd {R}, {a}, {self.operand(rest[2], ty)}")
        elif base == "sub": self.emit(f"isub {R}, {a}, {self.operand(rest[2], ty)}")
        elif base == "mul" and parts[1] == "lo": self.emit(f"imadd {R}, {a}, {self.operand(rest[2], ty)}, 0")
        elif base == "mad" and parts[1] == "lo": self.emit(f"imadd {R}, {a}, {self.operand(rest[2], ty)}, {self.operand(rest[3], ty)}")
        elif base == "and": self.emit(f"bitop 8 {R}, {a}, {self.operand(rest[2], ty)}")
        elif base == "or": self.emit(f"bitop 14 {R}, {a}, {self.operand(rest[2], ty)}")
        elif base == "xor": self.emit(f"bitop 6 {R}, {a}, {self.operand(rest[2], ty)}")
        elif base == "not": self.emit(f"bitop 1 {R}, {a}, {a}")
        elif base == "neg": self.emit(f"isub {R}, 0, {a}")
        elif base == "shl" and re.fullmatch(r"\d+", rest[2]): self.emit(f"imadd {R}, {a}, {1 << int(rest[2])}, 0")
        elif base == "shr" and re.fullmatch(r"\d+", rest[2]):
            if ty == "s32": self.emit(f"asr {R}, {a}, {rest[2]}")
            else: self.emit(f"bfeil 0 {R}, 0, {a}, {rest[2]}")
        elif base in ("min", "max"):
            cc = {"u32": ("ult", "ugt"), "s32": ("slt", "sgt"), "b32": ("ult", "ugt")}[ty][0 if base == "min" else 1]
            b = self.operand(rest[2], ty)
            self.emit(f"icmpsel {cc} {R}, {a}, {b}, {a}, {b}")
        else: raise Reject("instruction " + ".".join(parts))
        self.define(d)

    def memop(self, base, ty, rest):
        if ty not in ("u32", "s32", "b32", "f32"): raise Reject("memory type " + ty)
        addr = rest[1] if base == "ld" else rest[0]
        m = re.fullmatch(r"\[\s*(%\w+)\s*(?:\+\s*(\d+))?\s*\]", addr)
        if not m: raise Reject("address " + addr)
        s = self.sym.get(m.group(1))
        disp = int(m.group(2) or 0)
        if s is None: raise Reject("address register not derived from a pointer parameter")
        if s[0] == "ptr": s = ("addr", s[1], None, 4, 0)
        if s[0] != "addr": raise Reject("address")
        _, pidx, idx, scale, d0 = s
        disp += d0
        if scale not in (1, 2, 4, 8, 16) or scale == 0: raise Reject("scale")
        # AGX address = base + (index << (log2(4) + shift)); the PTX byte offset is idx*scale + disp
        if scale == 4 and disp % 4 == 0: shift, ireg = 0, idx
        else: raise Reject(f"index scale {scale} / displacement {disp} not expressible as an element index")
        if idx is None: ireg = None
        elems = disp // 4
        if ireg is None:
            tmp = f"r{self.next_tmp()}"; self.emit(f"mov {tmp}, {elems}")
        elif elems:
            tmp = f"r{self.next_tmp()}"; self.emit(f"iadd {tmp}, r{self.reg(ireg)}, {elems}")
        else: tmp = f"r{self.reg(ireg)}"
        pair = f"u{2 * pidx}_u{2 * pidx + 1}"
        if base == "ld":
            self.emit(f"device_load u32 1 r{self.reg(rest[0])}, {pair}, {tmp}, {shift}"); self.emit("wait 0"); self.define(rest[0])
        else:
            self.emit(f"device_store u32 1 {self.operand(rest[1], ty)}, {pair}, {tmp}, {shift}") if rest[1].startswith("%") else self.imm_store(rest[1], pair, tmp, shift)

    def imm_store(self, imm, pair, tmp, shift):
        t = f"r{self.next_tmp()}"; self.emit(f"mov {t}, {imm}"); self.emit(f"device_store u32 1 {t}, {pair}, {tmp}, {shift}")

    def next_tmp(self):
        self.next += 1
        if self.next > 127: raise Reject("out of registers")
        return self.next - 1

INV_F = {"feq": "fne", "fne": "feq", "flt": "fge", "fge": "flt", "fle": "fgt", "fgt": "fle"}

def main():
    L = Lower(open(sys.argv[1]).read())
    try:
        code = L.lower()
    except (Reject, KeyError, ValueError, IndexError) as e:
        sys.exit(f"cannot lower: {e!r}")
    print("\n".join(c + ";" for c in code[:-1]) + "\n" + code[-1])
    print(json.dumps({"params": L.params}), file=sys.stderr)

if __name__ == "__main__":
    main()

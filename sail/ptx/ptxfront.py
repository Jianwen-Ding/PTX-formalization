#!/usr/bin/env python3
"""Front end of the PTX Sail model: parses a test program (PTX module + host harness, the grammar of
semantics/ptx-syntax.k), performs the loading of semantics/ptx-load.k (flattening function bodies to
code lists, nested-scope register renaming, vector-register expansion, labels, register widths,
static declarations), and serializes the module as the token stream read by sail/ptx/ptx_main.sail.

Everything that has run-time meaning (linking, static allocation, buffers, launches, the harness
checks, every instruction) is left to the Sail model; this file only does what ptx-syntax.k (parsing)
and ptx-load.k (the purely syntactic `pp` pass) do.
"""
import re, struct, sys

class PtxError(Exception):
    pass

# ------------------------------------------------------------------ lexer
TOKEN_RES = [
    ("ws", r"\s+|//[^\n\r]*|/\*([^*]|\*+[^*/])*\*+/"),
    ("string", r'"([^"\\]|\\.)*"'),
    ("secname", r"\.debug_[a-z_0-9]+"),
    ("dotword", r"\.[a-zA-Z_][a-zA-Z0-9_]*"),
    ("reg", r"%[a-zA-Z_$][a-zA-Z0-9_$]*(\.[a-zA-Z][a-zA-Z0-9]*)?"),
    ("qname", r"[a-z][a-zA-Z0-9_]*(\.[a-zA-Z0-9_]+)*::[a-zA-Z0-9_]+(:[0-9]+)*((\.|::)[a-zA-Z0-9_]+(:[0-9]+)*)*"),
    ("hex", r"[+-]?0[xXfFdDbB][0-9a-fA-F]+[uU]?"),
    ("float", r"([+-]?[0-9]+(\.[0-9]*)?|[+-]?\.[0-9]+)([eE][+-]?[0-9]+)?([fFdD]|[pP][0-9]+[xX][0-9]+)?"),
    ("int", r"[+-]?[0-9]+[uU]?"),
    ("name", r"[a-zA-Z_$][a-zA-Z0-9_$]*(\.[a-zA-Z0-9_$]+)*"),
    ("punct", r"<<<|>>>|<<|>>|<=|>=|==|!=|&&|\|\||[,;:{}\[\]()<>=+\-!|@*/%&^~?]"),
]
TOKEN_CRE = [(k, re.compile(r)) for k, r in TOKEN_RES]
PRIO = {"hex": 3, "int": 2, "float": 1}

def lex(text):
    toks, i, n = [], 0, len(text)
    while i < n:
        best = None
        for kind, cre in TOKEN_CRE:
            m = cre.match(text, i)
            if m and m.end() > i:
                L = m.end() - i
                if best is None or L > best[1] or (L == best[1] and PRIO.get(kind, 0) > PRIO.get(best[0], 0)):
                    best = (kind, L)
        if best is None:
            raise PtxError("lexical error at %r" % text[i:i + 30])
        kind, L = best
        s = text[i:i + L]
        i += L
        if kind == "ws":
            continue
        if kind == "float":
            # a float literal without '.', exponent or suffix is an Int (K: Int has the higher priority)
            if re.fullmatch(r"[+-]?[0-9]+", s):
                kind = "int"
        toks.append((kind, s))
    toks.append(("eof", ""))
    return toks

def ptx_int(s):
    """an integer literal (ISA 4.5.1): decimal, octal 0{octal digit}+, optional U suffix (0x / 0b go through hex)"""
    s = s.rstrip("uU")
    neg = s.startswith("-")
    d = s.lstrip("+-")
    if len(d) > 1 and d[0] == "0" and all(c in "01234567" for c in d):
        v = int(d, 8)
    else:
        v = int(d)   # loose: a leading 0 followed by 8 / 9 is read as decimal
    return -v if neg else v

def float_bits(s):
    """the value of a K Float token as binary64 bits (f/F suffix: binary32 precision)"""
    single = False
    t = s
    m = re.fullmatch(r"(.*?)([pP]([0-9]+)[xX]([0-9]+))", t)
    if m:
        t = m.group(1)
    elif t[-1] in "fF":
        single, t = True, t[:-1]
    elif t[-1] in "dD":
        t = t[:-1]
    v = float(t)
    if single:
        v = struct.unpack("<f", struct.pack("<f", v))[0]
    return struct.unpack("<Q", struct.pack("<d", v))[0]

TYPES = {".u8", ".u16", ".u32", ".u64", ".s8", ".s16", ".s32", ".s64", ".b8", ".b16", ".b32", ".b64", ".f16", ".f32",
         ".f64", ".pred", ".b128", ".bf16", ".tf32", ".f16x2", ".bf16x2", ".s16x2", ".u16x2", ".u8x4", ".s8x4", ".f32x2",
         ".ue5m3x2", ".s2f6x2", ".e4m3", ".e5m2", ".e4m3x2", ".e5m2x2", ".e4m3x4", ".e5m2x4", ".e2m3x2", ".e3m2x2",
         ".e2m3x4", ".e3m2x4", ".e2m1x2", ".e2m1x4", ".ue8m0x2", ".u4", ".s4", ".u2", ".s2", ".b1"}
DATTRS = {".visible", ".extern", ".weak", ".common", ".global", ".shared", ".local", ".const", ".param", ".reg",
          ".align", ".ptr", ".v2", ".v4", ".v8", ".attribute"}
TEXKINDS = {".texref", ".samplerref", ".surfref"}
PERFDIRS = {".maxntid", ".reqntid", ".minnctapersm", ".maxnctapersm", ".maxnreg", ".noreturn", ".maxclusterrank",
            ".reqnctapercluster", ".explicitcluster", ".blocksareclusters", ".abi_preserve", ".abi_preserve_control",
            ".pragma", ".minperctamemory", ".language"}

# typeOf (ptx-common.k): Type -> (kind, width)
def type_of(t):
    s = t[1:]
    special = {"pred": ("p", 1), "b128": ("b", 128), "bf16": ("bf", 16), "tf32": ("tf", 32), "f16x2": ("f16x2", 32),
               "bf16x2": ("bf16x2", 32), "s16x2": ("s16x2", 32), "u16x2": ("u16x2", 32), "u8x4": ("u8x4", 32),
               "s8x4": ("s8x4", 32), "f32x2": ("f32x2", 64), "ue5m3x2": ("ue5m3x2", 16), "s2f6x2": ("s2f6x2", 16),
               "e4m3": ("e4m3", 8), "e5m2": ("e5m2", 8), "e4m3x2": ("e4m3x2", 16), "e5m2x2": ("e5m2x2", 16),
               "e4m3x4": ("e4m3x4", 32), "e5m2x4": ("e5m2x4", 32), "e2m3x2": ("e2m3x2", 16), "e3m2x2": ("e3m2x2", 16),
               "e2m3x4": ("e2m3x4", 32), "e3m2x4": ("e3m2x4", 32), "e2m1x2": ("e2m1x2", 8), "e2m1x4": ("e2m1x4", 16),
               "ue8m0x2": ("ue8m0x2", 16), "b1": ("b", 1)}
    if s in special:
        return special[s]
    return (s[0], int(s[1:]))

def tbytes(ty):
    return max(ty[1] // 8, 1)

# ------------------------------------------------------------------ constant expressions (ISA 4.5.4 - 4.5.6)
# A value is (v, unsigned, isfloat); integers are 64-bit .s64 / .u64, floats binary64.
M64 = (1 << 64) - 1
CPREC = {"||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6, "<": 7, ">": 7, "<=": 7, ">=": 7,
         "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10}

def cnorm(v, u):
    v &= M64
    return v - (1 << 64) if not u and v >> 63 else v

def cval(v, u):
    return (cnorm(v, u), u, False)

def cfloat(x):
    return x[0] if x[2] else float(x[0])

def cusual(a, b):
    u = a[1] or b[1]
    return cnorm(a[0], u), cnorm(b[0], u), u

def cbin(op, a, b):
    if a[2] or b[2]:
        if op in ("<<", ">>", "%", "&", "|", "^"):
            raise PtxError("operator %s needs integer operands in a constant expression" % op)
        x, y = cfloat(a), cfloat(b)
        if op in ("+", "-", "*", "/"):
            if op == "/" and y == 0.0:
                r = float("nan") if x == 0.0 or x != x else (float("inf") if (x > 0) == (str(y)[0] != "-") else float("-inf"))
            else:
                r = {"+": x + y, "-": x - y, "*": x * y, "/": (x / y) if y != 0.0 else 0.0}[op]
            return (r, False, True)
        if op in ("&&", "||"):
            return (int((x != 0.0) and (y != 0.0)) if op == "&&" else int((x != 0.0) or (y != 0.0)), False, False)
        return (int({"<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y, "==": x == y, "!=": x != y}[op]), False, False)
    if op in ("&&", "||"):
        return (int((a[0] != 0) and (b[0] != 0)) if op == "&&" else int((a[0] != 0) or (b[0] != 0)), False, False)
    if op in ("<<", ">>"):                       # 2nd operand .u64, result type of the 1st (4.5.5)
        n = cnorm(b[0], True)
        if op == "<<":
            return cval(0 if n >= 64 else a[0] << n, a[1])
        if a[1]:
            return cval(0 if n >= 64 else cnorm(a[0], True) >> n, True)
        return cval(a[0] >> min(n, 63), False)  # arithmetic shift of a signed value
    if op == "%":                                # operands .u64, result .s64 (Table 5)
        x, y = cnorm(a[0], True), cnorm(b[0], True)
        if y == 0:
            raise PtxError("remainder by zero in a constant expression")
        return cval(x % y, False)
    x, y, u = cusual(a, b)
    if op in ("<", ">", "<=", ">=", "==", "!="):
        return (int({"<": x < y, ">": x > y, "<=": x <= y, ">=": x >= y, "==": x == y, "!=": x != y}[op]), False, False)
    if op == "/":
        if y == 0:
            raise PtxError("division by zero in a constant expression")
        q = abs(x) // abs(y)
        return cval(q if (x < 0) == (y < 0) else -q, u)   # C: truncation toward zero
    # & | ^ keep the converted type (4.5.5; Table 5 says .u64 -- the same bits)
    return cval({"+": x + y, "-": x - y, "*": x * y, "&": x & y, "|": x | y, "^": x ^ y}[op], u)

# ------------------------------------------------------------------ parser
class Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def peek(self, k=0):
        return self.t[self.i + k]

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def at(self, s, k=0):
        return self.t[self.i + k][1] == s and self.t[self.i + k][0] in ("punct", "dotword", "name", "secname")

    def expect(self, s):
        tok = self.next()
        if tok[1] != s:
            raise PtxError("expected %r, got %r near token %d" % (s, tok[1], self.i))
        return tok

    def kind(self, k=0):
        return self.t[self.i + k][0]

    # -------------------------------------------------------------- constant expressions
    def lit_at(self, k=0):
        kk, ss = self.peek(k)
        return kk in ("int", "float", "hex") or (kk == "name" and ss == "WARP_SZ")

    def binop_at(self, k=0):
        kk, ss = self.peek(k)
        if kk == "punct" and (ss in CPREC or ss == "?"):
            return True
        return kk in ("int", "float", "hex") and ss[:1] in "+-"     # 1-2 lexes as 1, -2

    def cstart_at(self, k=0):
        """a constant expression (not a single literal) starts at token k"""
        kk, ss = self.peek(k)
        if self.lit_at(k):
            return self.binop_at(k + 1) or (kk == "hex" and ss[:1] in "+-")   # -0x10
        if kk == "punct" and ss == "(":
            # not a list such as (a, b) or the launch arguments (0f40000000, X)
            d, j = 0, k
            while self.peek(j)[0] != "eof":
                tj = self.peek(j)
                if tj[0] == "punct" and tj[1] == "(":
                    d += 1
                elif tj[0] == "punct" and tj[1] == ")":
                    d -= 1
                    if d == 0:
                        break
                elif tj[0] == "punct" and tj[1] in (",", ";") and d == 1:
                    return False
                j += 1
            k2, s2 = self.peek(k + 1)
            return self.lit_at(k + 1) or (k2 == "dotword" and s2 in (".s64", ".u64")) or \
                (k2 == "punct" and s2 in ("(", "-", "+", "!", "~")) or self.cstart_at(k + 1)
        if kk == "punct" and ss in ("-", "+", "!", "~"):
            k2, s2 = self.peek(k + 1)
            return self.lit_at(k + 1) or (k2 == "punct" and s2 in ("(", "-", "+", "!", "~"))
        return False

    def split_sign(self):
        """turn a sign-folded literal token (-2) into the operator and the literal"""
        kk, ss = self.peek()
        if kk in ("int", "float", "hex") and ss[:1] in "+-":
            self.t[self.i:self.i + 1] = [("punct", ss[0]), (kk, ss[1:])]

    def cprimary(self):
        self.split_sign()
        kk, ss = self.peek()
        if kk == "punct" and ss in ("-", "+", "!", "~"):
            self.next()
            x = self.cprimary()
            if ss == "+":
                return x
            if ss == "-":
                return (-x[0], False, True) if x[2] else cval(-x[0], x[1])
            if ss == "!":
                return (int(x[0] == 0), False, False)
            if x[2]:
                raise PtxError("~ needs an integer operand in a constant expression")
            return cval(~cnorm(x[0], True), True)
        if kk == "punct" and ss == "(":
            if self.peek(1)[0] == "dotword" and self.peek(1)[1] in (".s64", ".u64") and self.at(")", 2):
                self.next(); c = self.next()[1]; self.next()
                x = self.cprimary()
                if x[2]:
                    raise PtxError("cast of a floating-point constant expression")
                return cval(x[0], c == ".u64")
            self.next()
            x = self.cexpr()
            self.expect(")")
            return x
        self.next()
        if kk == "name" and ss == "WARP_SZ":
            return (32, False, False)
        if kk == "int":
            v = ptx_int(ss)
            return cval(v, ss[-1] in "uU" or v > (1 << 63) - 1)
        if kk == "hex" and ss[1] in "fF":     # loose: 0f / 0d literals by their value (the ISA keeps them out)
            return (struct.unpack("<f", struct.pack("<I", int(ss.rstrip("uU")[2:], 16)))[0], False, True)
        if kk == "hex" and ss[1] in "dD":
            return (struct.unpack("<d", struct.pack("<Q", int(ss.rstrip("uU")[2:], 16)))[0], False, True)
        if kk == "hex":
            d = ss.rstrip("uU")
            v = int(d[2:], 16 if d[1] in "xX" else 2)
            return cval(v, ss[-1] in "uU" or v > (1 << 63) - 1)
        if kk == "float":
            return (struct.unpack("<d", struct.pack("<Q", float_bits(ss)))[0], False, True)
        raise PtxError("bad constant expression at %r" % ss)

    def cbinrest(self, lhs, minp):
        while True:
            self.split_sign()
            kk, op = self.peek()
            if kk != "punct" or op not in CPREC or CPREC[op] < minp:
                return lhs
            self.next()
            rhs = self.cprimary()
            while True:
                self.split_sign()
                k2, op2 = self.peek()
                if k2 == "punct" and op2 in CPREC and CPREC[op2] > CPREC[op]:
                    rhs = self.cbinrest(rhs, CPREC[op2])
                else:
                    break
            lhs = cbin(op, lhs, rhs)

    def cexpr(self, lhs=None):
        c = self.cbinrest(self.cprimary() if lhs is None else lhs, 1)
        if self.at("?"):
            self.next()
            a = self.cexpr(); self.expect(":"); b = self.cexpr()
            if c[2]:
                raise PtxError("the condition of ?: must be an integer")
            if a[2] or b[2]:
                return (cfloat(a) if c[0] != 0 else cfloat(b), False, True)
            x, y, u = cusual(a, b)
            return cval(x if c[0] != 0 else y, u)
        return c

    def cint(self):
        x = self.cexpr()
        if x[2]:
            raise PtxError("an integer constant expression is required")
        return x[0]

    def cop(self, x):
        """the operand / initializer of a constant expression's value"""
        return ("float", struct.unpack("<Q", struct.pack("<d", x[0]))[0]) if x[2] else ("int", x[0])

    def take_int(self):
        tok = self.next()
        if tok[0] != "int":
            raise PtxError("expected an integer, got %r" % (tok[1],))
        return ptx_int(tok[1])

    def take_name(self):
        tok = self.next()
        if tok[0] != "name":
            raise PtxError("expected a name, got %r" % (tok[1],))
        return tok[1]

    # -------------------------------------------------------------- module
    def module(self):
        out = []
        while self.kind() != "eof":
            out.append(self.topdecl())
        return out

    def topdecl(self):
        k, s = self.peek()
        if s == ".version":
            self.next(); tok = self.next()
            return ("version", tok[1])
        if s == ".target":
            self.next(); return ("target", self.names())
        if s == ".address_size":
            self.next(); return ("address_size", self.take_int())
        if s == ".file":
            self.next(); self.take_int(); self.next()
            if self.at(","):
                self.next(); self.take_int(); self.expect(","); self.take_int()
            return ("skip",)
        if s == ".alias":
            self.next(); a = self.take_name(); self.expect(","); b = self.take_name(); self.expect(";")
            return ("alias", a, b)
        if s == ".section":
            self.next(); self.next(); self.skip_braces(); return ("skip",)
        if s == ".nv_intermediate_source_section":
            self.next(); self.skip_braces(); return ("skip",)
        if s == ".pragma":
            self.next(); self.pragma_strings(); self.expect(";"); return ("skip",)
        if s in HARNESS:
            return self.harness()
        return self.decl_or_func(top=True)

    def skip_braces(self):
        self.expect("{")
        depth = 1
        while depth:
            tok = self.next()
            if tok[1] == "{": depth += 1
            elif tok[1] == "}": depth -= 1
            elif tok[0] == "eof": raise PtxError("unterminated section")

    def names(self):
        ns = [self.take_name()]
        while self.at(","):
            self.next(); ns.append(self.take_name())
        return ns

    def ints(self):
        xs = [self.take_int()]
        while self.at(",") and self.kind(1) == "int":
            self.next(); xs.append(self.take_int())
        return xs

    def dattrs(self):
        attrs = []
        while self.peek()[1] in DATTRS:
            a = self.next()[1]
            if a == ".align":
                attrs.append((".align", self.take_int()))
            elif a == ".attribute":
                self.expect("(")
                b = self.next()[1]
                if b == ".unified" and self.at("("):
                    self.next(); self.next(); self.expect(","); self.next(); self.expect(")")
                self.expect(")")
                attrs.append((".attribute", b))
            else:
                attrs.append((a,))
        return attrs

    def ditem(self):
        k, s = self.next()
        if k not in ("name", "reg"):
            raise PtxError("bad declarator %r" % s)
        if self.at("<"):
            self.next(); n = self.take_int(); self.expect(">")
            return ("range", k, s, n)
        if k == "name" and self.at("["):
            dims = []
            while self.at("["):
                self.next()
                if self.at("]"):
                    dims.append(None)
                else:
                    dims.append(self.cint())
                self.expect("]")
            return ("array", s, dims)
        return ("plain", k, s)

    def ditems(self):
        items = [self.ditem()]
        while self.at(","):
            self.next(); items.append(self.ditem())
        return items

    def initval(self):
        if self.cstart_at():
            return self.cop(self.cexpr())
        k, s = self.peek()
        if s == "{":
            self.next()
            vs = []
            if not self.at("}"):
                vs.append(self.initval())
                while self.at(","):
                    self.next(); vs.append(self.initval())
            self.expect("}")
            return ("brace", vs)
        if k == "hex" and s[1] in "xX" and self.at("(", 1):
            # mask(expr) (PTX ISA 5.4.4): an integer constant expression or a symbol expression
            self.next(); self.next()
            inner = self.initval()
            if inner[0] == "hex" and inner[1][1] in "xX":
                inner = ("int", int(inner[1].rstrip("uU")[2:], 16))
            while inner[0] == "int" and ((self.at("+") and self.kind(1) == "int") or self.kind() == "int"):
                if self.at("+"):
                    self.next()
                inner = ("int", inner[1] + self.take_int())
            self.expect(")")
            if inner[0] not in ("int", "name", "nameoff", "generic", "genericoff"):
                raise PtxError("bad expression in the mask() operator %s" % s)
            return ("mask", int(s.rstrip("uU")[2:], 16), inner)
        if k == "name" and s == "generic" and self.at("(", 1):
            self.next(); self.next(); n = self.take_name(); self.expect(")")
            if self.at("+"):
                self.next(); return ("genericoff", n, self.take_int())
            if self.kind() == "int":
                return ("genericoff", n, self.take_int())
            return ("generic", n)
        self.next()
        if k == "int": return ("int", ptx_int(s))
        if k == "float": return ("float", float_bits(s))
        if k == "hex": return ("hex", s)
        if k == "name":
            if self.at("+") and self.kind(1) == "int":
                self.next(); return ("nameoff", s, self.take_int())
            if self.kind() == "int":
                return ("nameoff", s, self.take_int())
            if self.at("-") and self.kind(1) == "name":
                self.next(); return ("namediff", s, self.take_name())
            return ("name", s)
        raise PtxError("bad initializer %r" % s)

    def paramdecl(self):
        a1 = self.dattrs()
        t = self.next()[1]
        if t not in TYPES:
            raise PtxError("bad parameter type %r" % t)
        a2 = self.dattrs()
        return (a1, t, a2, self.ditem())

    def paramdecls(self):
        self.expect("(")
        ps = []
        if not self.at(")"):
            ps.append(self.paramdecl())
            while self.at(","):
                self.next(); ps.append(self.paramdecl())
        self.expect(")")
        return ps

    def perfdirs(self):
        ds = []
        while self.peek()[1] in PERFDIRS:
            d = self.next()[1]
            if d in (".maxntid", ".reqntid", ".reqnctapercluster"):
                ds.append((d, self.ints()))
            elif d in (".minnctapersm", ".maxnctapersm", ".maxnreg", ".maxclusterrank", ".abi_preserve",
                       ".abi_preserve_control", ".minperctamemory"):
                ds.append((d, [self.take_int()]))
            elif d in (".pragma",):
                # .pragma list-of-strings ; (11.4.1), also between a kernel's header and its body
                self.pragma_strings()
                if self.at(";"):
                    self.next()
                ds.append((d, []))
            elif d == ".language":
                self.next()
                if self.at(","):
                    self.next(); self.next()
                ds.append((d, []))
            else:
                ds.append((d, []))
        return ds

    def decl_or_func(self, top):
        a1 = self.dattrs()
        k, s = self.peek()
        if s in (".entry", ".func"):
            return self.funcdef(a1)
        if s in TEXKINDS:
            self.next(); self.ditems(); self.expect(";")
            return ("texdecl",)
        if s not in TYPES:
            raise PtxError("unexpected %r" % s)
        t = self.next()[1]
        a2 = self.dattrs()
        items = self.ditems()
        init = None
        if self.at("="):
            self.next(); init = self.initval()
        self.expect(";")
        return ("decl", a1, t, a2, items, init)

    def pragma_strings(self):
        self.next()
        while self.at(","):
            self.next(); self.next()

    def funcdef(self, a1):
        kw = self.next()[1]
        if self.at(".attribute"):
            a1 = list(a1) + self.dattrs()   # .func .attribute(attr-list) fname (11.3.2)
        rets = None
        if kw == ".func" and self.at("("):
            rets = self.paramdecls()
        name = self.take_name()
        params = self.paramdecls() if self.at("(") else None
        pd = self.perfdirs()
        if self.at(";"):
            self.next()
            return ("proto", a1, kw, name)
        body = self.block()
        return ("func", a1, kw, rets, name, params, pd, body)

    def block(self):
        self.expect("{")
        stmts = []
        while not self.at("}"):
            stmts.append(self.stmt())
        self.expect("}")
        return stmts

    def stmt(self):
        k, s = self.peek()
        if s == "{" and k == "punct":
            return ("block", self.block())
        if s == ".pragma":
            self.next(); self.pragma_strings(); self.expect(";"); return ("skip",)
        if s == ".loc":
            self.next(); self.take_int(); self.take_int(); self.take_int()
            while True:
                if self.at("prologue_end") or self.at("epilogue_begin"):
                    self.next()
                elif self.at("is_stmt"):
                    self.next(); self.take_int()
                elif self.at(",") and (self.at("function_name", 1) or self.at("inlined_at", 1)):
                    self.next()
                    if self.next()[1] == "function_name":
                        self.take_name()
                        if self.at("-"): self.next(); self.take_name()
                        elif self.at("+"): self.next(); self.take_int()
                    else:
                        self.take_int(); self.take_int(); self.take_int()
                else:
                    break
            return ("skip",)
        if s == ".loc_intermediate":
            self.next(); self.take_int(); self.take_int(); self.take_int(); return ("skip",)
        if s == "@" and k == "punct":
            self.next()
            neg = False
            if self.at("!"):
                self.next(); neg = True
            gk, gs = self.next()
            ins = self.instr()
            return ("ins", (gs, neg), ins[0], ins[1])
        if k == "name" and self.at(":", 1):
            name = self.next()[1]; self.next()
            if self.at(".callprototype"):
                while not self.at(";"):
                    self.next()
                self.next()
                return ("skip",)
            if self.at(".calltargets") or self.at(".branchtargets"):
                kw = self.next()[1]
                ns = self.names(); self.expect(";")
                return ("targets", kw, name, ns)
            return ("label", name)
        if k in ("name", "qname"):
            ins = self.instr()
            return ("ins", None, ins[0], ins[1])
        d = self.decl_or_func(top=False)
        return d

    def instr(self):
        k, opc = self.next()
        ops = []
        if not self.at(";"):
            ops.append(self.operand())
            while self.at(","):
                self.next(); ops.append(self.operand())
        self.expect(";")
        # an array element as a memory operand, `ld.global.u32 s, a[0];` (6.4.1, 6.4.2): the location a + 0
        base = opc.split(".")[0]
        ai = {"ld": 1, "ldu": 1, "atom": 1, "st": 0, "red": 0}.get(base)
        if ai is not None and len(ops) > ai and ops[ai][0] == "nameplus":
            ops[ai] = ("mem", [("anameoff", ops[ai][1], ops[ai][2])])
        if ai is not None and len(ops) > ai and ops[ai][0] == "nameidx":
            ops[ai] = ("mem", [("anamereg", ops[ai][1], ops[ai][2], ops[ai][3])])
        if any(o[0] == "nameidx" for o in ops):
            raise PtxError("a register-indexed array element is supported as a memory operand only (%s)" % opc)
        return opc, ops

    # -------------------------------------------------------------- operands
    def addr_or_op(self):
        k, s = self.peek()
        if k in ("reg", "name"):
            self.next()
            # [r + off], [r - off], [r+8-4], [r + (2*4)]: the offset is an integer constant expression (6.4.1)
            self.split_sign()
            if (self.at("+") or self.at("-")) and not self.at("]", 1):
                off = self.cexpr((0, False, False))
                if off[2]:
                    raise PtxError("the address offset must be an integer")
                return ("aregoff" if k == "reg" else "anameoff", s, off[0])
            return ("areg" if k == "reg" else "aname", s)
        if k == "int" or self.cstart_at():
            return ("aint", self.cint())
        return ("op", self.operand())

    def operand(self):
        if self.cstart_at():
            return self.cop(self.cexpr())
        k, s = self.peek()
        if k == "punct" and s == "[":
            self.next()
            items = [self.addr_or_op()]
            while self.at(","):
                self.next(); items.append(self.addr_or_op())
            self.expect("]")
            return ("mem", items)
        if k == "punct" and s == "{":
            self.next()
            ops = []
            if not self.at("}"):
                ops.append(self.operand())
                while self.at(","):
                    self.next(); ops.append(self.operand())
            self.expect("}")
            if self.at("|"):
                self.next()
                pk, ps = self.next()
                return ("vecpipe", ops, ("reg" if pk == "reg" else "name", ps))
            return ("vec", ops)
        if k == "punct" and s == "(":
            self.next()
            ops = []
            if not self.at(")"):
                ops.append(self.operand())
                while self.at(","):
                    self.next(); ops.append(self.operand())
            self.expect(")")
            return ("paren", ops)
        if k == "punct" and s in ("-", "!"):
            self.next()
            pk, ps = self.next()
            if pk not in ("reg", "name"):
                raise PtxError("bad operand after %s: %r" % (s, ps))
            return ("neg" if s == "-" else "not", ("reg" if pk == "reg" else "name", ps))
        self.next()
        # d|p: a register or name (the sink _ included) after the bar, never a literal
        if k == "reg":
            if self.at("|"):
                self.next(); b = self.next()
                if b[0] not in ("reg", "name"):
                    raise PtxError("bad operand after %s|: %r" % (s, b[1]))
                return ("pipe", ("reg", s), ("reg", b[1]))
            return ("reg", s)
        if k == "name":
            if self.at("|"):
                self.next(); b = self.next()
                if b[0] not in ("reg", "name"):
                    raise PtxError("bad operand after %s|: %r" % (s, b[1]))
                return ("pipe", ("name", s), ("name", b[1]))
            if self.at("+") and self.kind(1) == "int":
                self.next(); return ("nameplus", s, ptx_int(self.next()[1]))
            if self.kind() == "int":
                return ("nameint", s, ptx_int(self.next()[1]))
            if self.at("[") and self.kind(1) == "reg":
                # a[r], a[r+4], a[r-4]: register index (6.4.2) -- an address of ld / st / atom / red
                self.next(); r = self.next()[1]
                self.split_sign()
                off = self.cexpr((0, False, False))[0] if (self.at("+") or self.at("-")) else 0
                self.expect("]")
                return ("nameidx", s, r, off)
            if self.at("[") and (self.lit_at(1) or self.cstart_at(1)):
                # a[1], a[N-1]: the address of the array element (PTX ISA 6.4.2); the index is a byte offset as in
                # the example of 4.3 (array[r1] with r1 = tid << 2)
                self.next(); i = self.cint(); self.expect("]")
                return ("nameplus", s, i)
            return ("name", s)
        if k == "int": return ("int", ptx_int(s))
        if k == "float": return ("float", float_bits(s))
        if k == "hex": return ("hex", s)
        raise PtxError("bad operand %r" % s)

    # -------------------------------------------------------------- harness
    def dim(self):
        if self.at("("):
            self.next(); xs = [self.take_int()]
            while self.at(","):
                self.next(); xs.append(self.take_int())
            self.expect(")")
            return xs + [1] * (3 - len(xs))
        return [self.take_int(), 1, 1]

    def initvals_braced(self):
        v = self.initval()
        if v[0] != "brace":
            raise PtxError("expected { ... }")
        return v[1]

    def harness(self):
        d = self.next()[1]
        if d == ".buffer":
            n = self.take_name(); t = self.next()[1]
            if self.at("["):
                self.next(); c = self.take_int(); self.expect("]")
                vals = []
                if self.at("="):
                    self.next(); vals = self.initvals_braced()
                self.expect(";")
                return ("buffer", n, t, c, vals)
            self.expect("="); vals = self.initvals_braced(); self.expect(";")
            return ("buffer", n, t, None, vals)
        if d in (".launch", ".cooplaunch"):
            k = self.take_name(); self.expect("<<<"); g = self.dim(); self.expect(","); b = self.dim()
            dyn = 0
            if self.at(","):
                self.next(); dyn = self.take_int()
            self.expect(">>>")
            args = self.operand()
            self.expect(";")
            return ("launch" if d == ".launch" else "cooplaunch", k, g, b, dyn, args[1])
        if d == ".print":
            n = self.take_name(); t = self.next()[1]
            c = -1
            if self.kind() == "int":
                c = self.take_int()
            self.expect(";")
            return ("print", n, t, c)
        if d in (".expect", ".patch", ".append"):
            n = self.take_name(); t = self.next()[1]
            off = 0
            if self.at("@"):
                self.next(); off = self.take_int()
            self.expect("="); vals = self.initvals_braced(); self.expect(";")
            return (d[1:], n, t, off, vals)
        if d == ".tensormap":
            n = self.take_name(); t = self.next()[1]; b = self.take_name()
            self.expect(".dims"); self.expect("{"); ds = self.ints(); self.expect("}")
            self.expect(".strides"); self.expect("{"); ss = self.ints() if not self.at("}") else []; self.expect("}")
            self.expect(".box"); self.expect("{"); bs = self.ints(); self.expect("}")
            self.expect(".swizzle"); z = self.take_int()
            if self.at(".im2col"):
                self.next()
                self.expect("{"); ls = self.ints(); self.expect("}")
                self.expect("{"); us = self.ints(); self.expect("}")
                self.expect(".estrides"); self.expect("{"); es = self.ints(); self.expect("}")
                self.expect(";")
                return ("tensormapi", n, t, b, ds, ss, bs, z, ls, us, es)
            self.expect(";")
            return ("tensormap", n, t, b, ds, ss, bs, z)
        if d in (".texture", ".surface"):
            n = self.take_name(); t = self.next()[1]; b = self.take_name()
            self.expect(".dims"); self.expect("{"); ds = self.ints(); self.expect("}")
            self.expect(".attrs"); self.expect("{"); at = self.ints(); self.expect("}"); self.expect(";")
            return (d[1:], n, t, b, ds, at)
        if d == ".sampler":
            n = self.take_name(); self.expect(".attrs"); self.expect("{"); at = self.ints(); self.expect("}"); self.expect(";")
            return ("sampler", n, at)
        if d == ".multimem":
            n = self.take_name(); self.expect(".size"); sz = self.take_int(); self.expect("{"); ns = self.names(); self.expect("}"); self.expect(";")
            return ("multimem", n, sz, ns)
        if d == ".fabric":
            i = self.take_int(); self.expect(".size"); sz = self.take_int(); self.expect("{"); ns = self.names(); self.expect("}"); self.expect(";")
            return ("fabric", i, sz, ns)
        if d in (".memmodel", ".ub", ".convergence"):
            n = self.take_name(); self.expect(";")
            return (d[1:], n)
        if d == ".occupancy":
            n = self.take_int(); self.expect(";")
            return ("occupancy", n)
        if d == ".isacheck":
            self.expect(";")
            return ("isacheck",)
        raise PtxError("unknown directive %s" % d)

HARNESS = {".buffer", ".launch", ".cooplaunch", ".print", ".expect", ".patch", ".tensormap", ".texture", ".sampler",
           ".surface", ".multimem", ".fabric", ".memmodel", ".ub", ".occupancy", ".convergence", ".isacheck", ".append"}

# ------------------------------------------------------------------ loading (ptx-load.k)
def aset(attrs):
    s = set()
    for a in attrs:
        if a[0] == ".align":
            continue
        if a[0] == ".attribute":
            if a[1] == ".managed":
                s.add("managed")
            continue
        s.add(a[0][1:])
    return s

def align_of(attrs):
    for a in attrs:
        if a[0] == ".align":
            return a[1]
    return 0

def align_or(a1, a2, ty, st=()):
    # vector variables are aligned to their overall size by default (PTX ISA 5.4.2)
    return align_of(a1) or align_of(a2) or tbytes(ty) * (vwidth(st) or 1)

def dspace(s):
    for k in ("reg", "shared", "local", "param", "global", "const"):
        if k in s:
            return k
    return "global"

def item_name(it):
    return it[2] if it[0] in ("plain", "range") else it[1]

def dims_count(dims):
    # dimsCount: [N] R => N * dimsCount(R); [N] => N; [] => 0 (an unsized array without initializer)
    n = 1
    for d in dims:
        n *= d or 0
    return n

def item_dims(it, st):
    """the dimensions of a declarator; a .v2 / .v4 / .v8 vector is an innermost dimension (PTX ISA 5.4.2)"""
    dims = list(it[2]) if it[0] == "array" else []
    k = vwidth(st)
    return dims + [k] if k else dims

def item_size2(it, ty, st=()):
    return dims_count(item_dims(it, st)) * tbytes(ty)

def shape_init(init, dims):
    """the flat element list of an initializer for an object of the given dimensions, and the dimensions with
    an omitted first one taken from the initializer (PTX ISA 5.4.3, 5.4.4).  As in C, a nested list may be
    incomplete: the rest of its sub-array is filled with zeros (only up to the last initialized element)."""
    if not dims:
        if init[0] == "brace":
            if len(init[1]) != 1:
                raise PtxError("a scalar takes a single initializer value")
            return shape_init(init[1][0], [])
        return [init], []
    if init[0] != "brace":
        return [init], [dims[0] or 1] + dims[1:]
    inner = dims_count(dims[1:])
    out, pad = [], False
    for v in init[1]:
        if pad or (v[0] == "brace" and len(dims) > 1):
            out.extend([("int", 0)] * (-len(out) % inner))
        pad = False
        if v[0] == "brace" and len(dims) > 1:
            out.extend(shape_init(v, dims[1:])[0])
            pad = True
        else:
            out.extend(shape_init(v, [])[0])
    n = -(-len(out) // inner) if inner else 0
    if dims[0] is not None and n > dims[0]:
        raise PtxError("too many initializers")
    return out, [dims[0] if dims[0] is not None else n] + dims[1:]

def vwidth(s):
    for k, n in (("v2", 2), ("v4", 4), ("v8", 8)):
        if k in s:
            return n
    return 0

VCOMP = ["x", "y", "z", "w"]
def vcomp(i):
    return VCOMP[i] if 0 <= i < 4 else "s%d" % i

def pdecls(ps):
    """ParamDecls -> list of (name, ty, size, align); .reg formals by value (align -1 / -2)"""
    out = []
    for a1, t, a2, it in ps or []:
        ty = type_of(t)
        s = aset(a1) | aset(a2)
        name = item_name(it)
        if "reg" not in s:
            out.append((name, ty, item_size2(it, ty, s), align_or(a1, a2, ty, s)))
        else:
            k = vwidth(s)
            if k == 0:
                out.append((name, ty, 1, -1))
            else:
                for i in range(k):
                    out.append((name + "." + vcomp(i), ty, k if i == 0 else 1, -2 if i == 0 else -1))
    return out

def fmarks(pl):
    rw = {}
    for (n, ty, k, al) in pl:
        if al == -1:
            rw[n] = ty[1]
        elif al == -2:
            rw[n] = ty[1]
            rw["@vec:" + n[:-2]] = k
    return rw

def comp_tok(s):
    return ("reg", s) if s.startswith("%") else ("name", s)

def op_str(o):
    return o[1] if o[0] in ("reg", "name") else ""

def vec_k(o, rw):
    if o[0] in ("reg", "name"):
        return rw.get("@vec:" + o[1], 0)
    return 0

class Loader:
    def __init__(self, rw):
        self.code = []
        self.L = {}
        self.rw = rw
        self.rk = {}          # register name -> declared type (kind, width), for the .isacheck type rules
        self.sh = []

    # renaming (rnS / rnO / rnA)
    def rn_s(self, s):
        k = "@ren:" + s
        return s + "@" + str(self.L[k]) if k in self.L else s

    def rn_a(self, a):
        if a[0] == "areg": return ("areg", self.rn_s(a[1]))
        if a[0] == "aname": return ("aname", self.rn_s(a[1]))
        if a[0] == "aregoff": return ("aregoff", self.rn_s(a[1]), a[2])
        if a[0] == "anameoff": return ("anameoff", self.rn_s(a[1]), a[2])
        if a[0] == "anamereg": return ("anamereg", self.rn_s(a[1]), self.rn_s(a[2]), a[3])
        return a

    def rn_item(self, it):
        # second / third bracket items: an address is renamed by rnA (rnB), an operand by rnO
        if it[0] == "op":
            return ("op", self.rn_o(it[1]))
        return self.rn_a(it)

    def rn_o(self, o):
        t = o[0]
        if t in ("reg", "name"):
            return (t, self.rn_s(o[1]))
        if t == "mem":
            items = o[1]
            if len(items) == 1:
                return ("mem", [self.rn_a(items[0])])
            if len(items) == 2:
                return ("mem", [self.rn_a(items[0]), self.rn_item(items[1])])
            if len(items) == 3:
                return ("mem", [self.rn_a(items[0]), self.rn_item(items[1]), self.rn_item(items[2])])
            return o
        if t == "vec":
            return ("vec", [self.rn_o(x) for x in o[1]])
        if t == "paren":
            return ("paren", [self.rn_o(x) for x in o[1]])
        if t == "pipe" and o[1][0] == o[2][0]:
            return ("pipe", (o[1][0], self.rn_s(o[1][1])), (o[2][0], self.rn_s(o[2][1])))
        if t == "not":
            return ("not", (o[1][0], self.rn_s(o[1][1])))
        return o

    # vector registers (vexp / vflat)
    def vexp(self, ops):
        out = []
        for o in ops:
            if o[0] == "paren":
                out.append(("paren", self.vflat(o[1])))
            else:
                k = vec_k(o, self.rw)
                if k > 0:
                    out.append(("vec", [comp_tok(op_str(o) + "." + vcomp(i)) for i in range(k)]))
                else:
                    out.append(o)
        return out

    def vflat(self, ops):
        out = []
        for o in ops:
            k = vec_k(o, self.rw)
            if k > 0:
                out.extend(comp_tok(op_str(o) + "." + vcomp(i)) for i in range(k))
            else:
                out.append(o)
        return out

    def emit_ins(self, guard, opc, ops):
        g = None if guard is None else (self.rn_s(guard[0]), guard[1])
        self.code.append(("ins", g, opc, self.vexp([self.rn_o(o) for o in ops])))

    def pp(self, stmts):
        for s in stmts:
            self.pp1(s)

    def pp1(self, s):
        k = s[0]
        if k == "label":
            self.L[s[1]] = len(self.code)
        elif k == "ins":
            self.emit_ins(s[1], s[2], s[3])
        elif k == "block":
            old = {kk: v for kk, v in self.L.items() if kk.startswith("@ren:") or kk == "@depth"}
            self.L["@depth"] = self.L.get("@depth", 0) + 1
            self.pp(s[1])
            for kk in [kk for kk in self.L if kk.startswith("@ren:") or kk == "@depth"]:
                del self.L[kk]
            self.L.update(old)
        elif k == "targets":
            key = ("@calltargets:" if s[1] == ".calltargets" else "@branchtargets:") + s[2]
            self.L[key] = list(s[3])
        elif k == "decl":
            _, a1, t, a2, items, init = s
            st = aset(a1) | aset(a2)
            ty = type_of(t)
            al = align_or(a1, a2, ty, st)
            sp = dspace(st)
            # loose: a declaration of %-registers without a state space (`.pred %p;`) declares registers
            if not (st & {"reg", "shared", "local", "param", "global", "const"}) and items and \
                    all(it[0] in ("plain", "range") and it[1] == "reg" for it in items):
                sp = "reg"
            if init is not None and sp in ("shared", "global", "const"):
                self.sh.extend(sdecls(st, al, ty, items, init))
            else:
                self.pp_decl(sp, st, al, ty, list(items))
        elif k in ("skip", "texdecl"):
            pass
        elif k in ("func", "proto"):
            raise PtxError("function definition inside a body")
        else:
            raise PtxError("unexpected statement %r" % (k,))

    def pp_decl(self, sp, st, al, ty, items):
        while items:
            it = items.pop(0)
            if sp == "reg":
                plain = it[0] == "plain"
                if plain and self.L.get("@depth", 0) > 0 and item_name(it) in self.rw:
                    rid = self.L.get("@renid", 1)
                    self.L["@ren:" + item_name(it)] = rid
                    self.L["@renid"] = rid + 1
                    items.insert(0, ("plain", it[1], item_name(it) + "@" + str(rid)))
                    continue
                k = vwidth(st)
                if it[0] == "range":
                    if k == 0:
                        for i in range(it[3]):
                            self.rw[it[2] + str(i)] = ty[1]
                            self.rk[it[2] + str(i)] = ty
                    else:
                        for i in range(it[3]):
                            self.vdeclare(it[2] + str(i), ty[1], k, ty)
                elif k > 0 and plain:
                    self.vdeclare(item_name(it), ty[1], k, ty)
                else:
                    self.rw[item_name(it)] = ty[1]
                    self.rk[item_name(it)] = ty
            elif sp in ("local", "param"):
                self.code.append(("declvar", sp, item_name(it), item_size2(it, ty, st), al))
            else:
                self.sh.extend(sdecls(st, al, ty, [it] + items, None))
                return

    def vdeclare(self, n, w, k, ty=None):
        for i in range(k):
            self.rw[n + "." + vcomp(i)] = w
            if ty is not None: self.rk[n + "." + vcomp(i)] = ty
        self.rw["@vec:" + n] = k

def iv_list(vs):
    out = []
    for v in vs:
        if v[0] == "brace":
            out.extend(iv_list(v[1]))
        else:
            out.append(v)
    return out

def sdecls(st, al, ty, items, init):
    out = []
    for it in items:
        dims = item_dims(it, st)
        vals = []
        if init is not None:
            vals, dims = shape_init(init, dims)
        out.append((dspace(st), item_name(it), ty, dims_count(dims) * tbytes(ty), al, "extern" in st, vals))
    return out

def func_head(kw, rets, name, params):
    return name, ("entry" if kw == ".entry" else "func"), pdecls(rets), pdecls(params)

# ------------------------------------------------------------------ serialization
def ser_ty(ty):
    return "%s %d" % ty

def ser_sop(o):
    t = o[0]
    if t == "reg": return "R " + o[1]
    if t == "name": return "N " + o[1]
    if t == "int": return "I %d" % o[1]
    if t == "float": return "F %d" % o[1]
    if t == "hex": return "H " + o[1]
    if t in ("nameplus", "nameint"): return "NP %s %d" % (o[1], o[2])
    if t == "neg": return "NEG %s %s" % (o[1][0][0], o[1][1])
    if t == "not": return "NOT %s %s" % (o[1][0][0], o[1][1])
    if t == "pipe": return "PIPE %s %s %s %s" % (o[1][0][0], o[1][1], o[2][0][0], o[2][1])
    raise PtxError("operand %r cannot be nested here" % (o,))

def ser_addr(a):
    t = a[0]
    if t == "areg": return "AR " + a[1]
    if t == "aname": return "AN " + a[1]
    if t == "aint": return "AI %d" % a[1]
    if t == "aregoff": return "ARO %s %d" % (a[1], a[2])
    if t == "anameoff": return "ANO %s %d" % (a[1], a[2])
    if t == "anamereg": return "ANR %s %s %d" % (a[1], a[2], a[3])
    raise PtxError("bad address %r" % (a,))

def addr_as_op(a):
    t = a[0]
    if t == "areg": return ("reg", a[1])
    if t == "aname": return ("name", a[1])
    if t == "aint": return ("int", a[1])
    if t == "anameoff": return ("nameplus", a[1], a[2])
    if t == "aregoff": raise PtxError("register + offset is not an operand")
    return a[1]

def as_addr(it):
    if it[0] == "op":
        raise PtxError("expected an address in [ ... ]")
    return it

def ser_iop(o):
    if o[0] == "vec":
        return "V %d %s" % (len(o[1]), " ".join(ser_sop(x) for x in o[1]))
    return "S " + ser_sop(o)

def ser_xop(o):
    t = o[0]
    if t == "vec":
        return "V %d %s" % (len(o[1]), " ".join(ser_sop(x) for x in o[1]))
    if t == "paren":
        return "P %d %s" % (len(o[1]), " ".join(ser_sop(x) for x in o[1]))
    if t == "vecpipe":
        return "VP %d %s %s %s" % (len(o[1]), " ".join(ser_sop(x) for x in o[1]), o[2][0][0], o[2][1])
    if t == "mem":
        items = o[1]
        na = 1 if len(items) <= 2 else 2
        addrs = [as_addr(items[i]) for i in range(min(na, len(items)))]
        rest = [addr_as_op(it) if it[0] != "op" else it[1] for it in items[na:]]
        return "B %d %s %d %s" % (len(addrs), " ".join(ser_addr(a) for a in addrs), len(rest), " ".join(ser_iop(x) for x in rest))
    return "S " + ser_sop(o)

def split_opc(s):
    """decodeOpc (ptx-common.k): base name, modifier list (in order), type list"""
    parts = s.split(".")
    base, mods, tys = parts[0], [], []
    for x in parts[1:]:
        if is_ty_str(x):
            tys.append(str_ty(x))
        else:
            mods.append(x)
            if "::" in x and x.index("::") > 0:
                mods.append(x[:x.index("::")])
    return base, mods, tys

TY_STRS = {"u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64", "b8", "b16", "b32", "b64", "f16", "f32", "f64", "pred",
           "b128", "bf16", "tf32", "f16x2", "bf16x2", "s16x2", "u16x2", "u8x4", "s8x4", "f32x2", "ue5m3x2", "s2f6x2",
           "e4m3", "e5m2", "e4m3x2", "e5m2x2", "e4m3x4", "e5m2x4", "e2m3x2", "e3m2x2", "e2m3x4", "e3m2x4", "e2m1x2",
           "e2m1x4", "ue8m0x2", "e3m2", "e2m3", "e2m1", "ue8m0", "ue4m3", "u4", "s4", "u2", "s2", "b1"}
def is_ty_str(x):
    return x in TY_STRS

def str_ty(x):
    special = {"pred": ("p", 1), "b128": ("b", 128), "bf16": ("bf", 16), "tf32": ("tf", 32), "e3m2": ("e3m2", 8),
               "e2m3": ("e2m3", 8), "e2m1": ("e2m1", 8), "ue8m0": ("ue8m0", 8), "ue4m3": ("ue4m3", 8)}
    if x in special:
        return special[x]
    t = type_of("." + x) if ("." + x) in TYPES else None
    if t:
        return t
    return (x[0], int(x[1:]))

def ser_code_item(c):
    if c[0] == "declvar":
        return "DV %s %s %d %d" % (c[1], c[2], c[3], c[4])
    _, g, opc, ops = c
    base, mods, tys = split_opc(opc)
    gs = "G0" if g is None else "G1 %s %d" % (g[0], int(g[1]))
    return "IN %s %s %d %s %d %s %d %s" % (gs, base, len(mods), " ".join(mods), len(tys), " ".join(ser_ty(t) for t in tys),
                                         len(ops), " ".join(ser_xop(o) for o in ops))

def ser_init(v):
    t = v[0]
    if t == "int": return "I %d" % v[1]
    if t == "float": return "F %d" % v[1]
    if t == "hex": return "H " + v[1]
    if t == "name": return "N " + v[1]
    if t == "nameoff": return "NI %s %d" % (v[1], v[2])
    if t == "namediff": return "ND %s %s" % (v[1], v[2])
    if t == "generic": return "G " + v[1]
    if t == "genericoff": return "GO %s %d" % (v[1], v[2])
    if t == "mask":
        x = v[2]
        kind, name, off = {"int": (0, "-", x[1]), "name": (1, x[1], 0), "generic": (2, x[1], 0),
                           "nameoff": (1, x[1], x[2]) if len(x) > 2 else None,
                           "genericoff": (2, x[1], x[2]) if len(x) > 2 else None}[x[0]]
        return "M %d %d %s %d" % (v[1], kind, name, off)
    raise PtxError("bad initializer %r" % (v,))

def ser_sdecl(d):
    sp, name, ty, size, al, ext, init = d
    return "SD %s %s %s %d %d %d %d %s" % (sp, name, ser_ty(ty), size, al, int(ext), len(init), " ".join(ser_init(v) for v in init))

def ser_pdecl(p):
    return "%s %s %d %d" % (p[0], ser_ty(p[1]), p[2], p[3])

def ser_perf(pd):
    return " ".join("%s %d %s" % (d[0][1:], len(xs), " ".join(map(str, xs))) for d, xs in [((d,), xs) for d, xs in pd])

# the predefined, read-only special registers of PTX ISA 9.3 chapter 10 (%warpsize: older code, value 32)
SREGS = ({"%tid", "%ntid", "%laneid", "%warpid", "%nwarpid", "%ctaid", "%nctaid", "%smid", "%nsmid", "%gridid",
          "%is_explicit_cluster", "%clusterid", "%nclusterid", "%cluster_ctaid", "%cluster_nctaid", "%cluster_ctarank",
          "%cluster_nctarank", "%lanemask_eq", "%lanemask_le", "%lanemask_lt", "%lanemask_ge", "%lanemask_gt",
          "%clock", "%clock_hi", "%clock64", "%globaltimer", "%globaltimer_lo", "%globaltimer_hi",
          "%reserved_smem_offset_begin", "%reserved_smem_offset_end", "%reserved_smem_offset_cap",
          "%reserved_smem_offset_0", "%reserved_smem_offset_1", "%total_smem_size", "%aggr_smem_size",
          "%dynamic_smem_size", "%current_graph_exec", "%warpsize"}
         | {"%%pm%d" % i for i in range(8)} | {"%%pm%d_64" % i for i in range(8)} | {"%%envreg%d" % i for i in range(32)})

COLOUR = {"r": "x", "g": "y", "b": "z", "a": "w"}

def colour_fields(ld):
    """vector elements may be named by the colour fields .r .g .b .a as well as .x .y .z .w (ISA 6.4.2)"""
    def fix(x):
        if isinstance(x, tuple):
            if len(x) == 2 and x[0] == "reg" and isinstance(x[1], str) and "." in x[1]:
                base, c = x[1].rsplit(".", 1)
                if c in COLOUR and "@vec:" + base in ld.rw:
                    return ("reg", base + "." + COLOUR[c])
            return tuple(fix(y) for y in x)
        if isinstance(x, list):
            return [fix(y) for y in x]
        return x
    ld.code = [c if c[0] == "declvar" else (c[0], c[1], c[2], fix(c[3])) for c in ld.code]

def check_regs(fname, ld):
    """every %-register operand is declared in the function or is a special register (ISA 6.1, 10)"""
    def walk(x):
        if isinstance(x, tuple):
            if len(x) >= 2 and x[0] == "reg" and isinstance(x[1], str):
                yield x[1]
            if len(x) == 4 and x[0] == "anamereg":
                yield x[2]
            for y in x:
                yield from walk(y)
        elif isinstance(x, list):
            for y in x:
                yield from walk(y)
    for c in ld.code:
        if c[0] == "declvar":
            continue
        _, g, opc, ops = c
        names = list(walk(ops)) + ([g[0]] if g is not None else [])
        for n in names:
            base = n.split(".")[0]          # %r.b1 / %r.h10: video selectors; %tid.x: components
            if not n.startswith("%") or n in ld.rw or base in ld.rw or base in SREGS or n.split("@")[0] in ld.rw:
                continue
            raise PtxError("%s: register %s is not declared" % (fname, n))

# ---------------------------------------------------------------- operand type rules (ISA 9.4, Tables 26-28)
SIMPLE_TY = {"b8": ("b", 8), "b16": ("b", 16), "b32": ("b", 32), "b64": ("b", 64), "b128": ("b", 128),
             "s8": ("s", 8), "s16": ("s", 16), "s32": ("s", 32), "s64": ("s", 64),
             "u8": ("u", 8), "u16": ("u", 16), "u32": ("u", 32), "u64": ("u", 64),
             "f16": ("f", 16), "f32": ("f", 32), "f64": ("f", 64)}
# instructions whose register operands all carry the instruction type (Table 26); setp from its second operand
SAME_TY = {"add", "sub", "mul", "mad", "min", "max", "abs", "neg", "div", "rem", "and", "or", "xor", "not", "cnot",
           "fma", "rcp", "sqrt", "rsqrt", "sin", "cos", "lg2", "ex2", "copysign", "mov", "setp", "selp"}

def ty_agrees(it, ot, relaxed):
    """Table 26 (exact) or Tables 27 / 28 (ld, st, cvt: wider operands allowed)"""
    (ik, iw), (ok, ow) = it, ot
    if ok not in ("b", "s", "u", "f"):
        return True
    if relaxed:
        if ow < iw: return False
    elif ow != iw:
        return False
    if ik == "f":
        return ok == "b" or (ok == "f" and ow == iw)
    if ik in ("s", "u"):
        return ok != "f"
    return True

def check_types(fname, ld):
    """.isacheck: register operands must agree with the instruction type (ISA 9.4)"""
    def regs(o):
        if o[0] == "reg": return [o[1]]
        if o[0] == "vec": return [x[1] for x in o[1] if x[0] == "reg"]
        if o[0] == "pipe": return [x[1] for x in o[1:] if isinstance(x, tuple) and x[0] == "reg"]
        return []
    for c in ld.code:
        if c[0] != "ins": continue
        opc, ops = c[2], c[3]
        parts = opc.split(".")
        base = parts[0]
        tys = [SIMPLE_TY[p] for p in parts[1:] if p in SIMPLE_TY]
        alltys = [p for p in parts[1:] if p in SIMPLE_TY or p in ("pred", "bf16", "tf32") or "x" in p and p[-1].isdigit()]
        checks = []
        if base in SAME_TY and len(tys) == 1 and len(alltys) == 1 and "wide" not in parts:
            sel = ops[1:] if base == "setp" else (ops[:3] if base == "selp" else ops)
            # mov d, {a, b} packs / unpacks elements of their own size (9.7.9.10)
            checks = [(o, tys[0], False) for o in sel if not (base == "mov" and o[0] == "vec")]
        elif base in ("shl", "shr") and len(tys) == 1 and len(alltys) == 1 and len(ops) == 3:
            # "the shift amount operand for left and right shift instructions always has type .u32" (9.4)
            checks = [(ops[0], tys[0], False), (ops[1], tys[0], False), (ops[2], ("u", 32), False)]
        elif base in ("mul", "mad") and "wide" in parts and len(tys) == 1 and len(alltys) == 1 and len(ops) >= 3:
            # .wide: d (and c of mad) are twice the instruction width (9.7.1.3, 9.7.1.4)
            wt = (tys[0][0], 2 * tys[0][1])
            checks = [(ops[0], wt, False), (ops[1], tys[0], False), (ops[2], tys[0], False)] + \
                     ([(ops[3], wt, False)] if base == "mad" and len(ops) == 4 else [])
        elif base in ("popc", "clz") and len(tys) == 1 and len(alltys) == 1 and len(ops) == 2:
            # d has type .u32, a the instruction type (9.7.1.15, 9.7.1.16)
            checks = [(ops[0], ("u", 32), False), (ops[1], tys[0], False)]
        elif base == "ld" and len(tys) == 1 and len(alltys) == 1 and ops:
            checks = [(ops[0], tys[0], True)]
        elif base == "st" and len(tys) == 1 and len(alltys) == 1 and ops:
            checks = [(ops[-1], tys[0], True)]
        elif base == "cvt" and len(tys) == 2 and len(alltys) == 2 and len(ops) == 2:
            checks = [(ops[0], tys[0], True), (ops[1], tys[1], True)]
        for o, it, rel in checks:
            for r in regs(o):
                ot = ld.rk.get(r)
                if ot is not None and not ty_agrees(it, ot, rel):
                    raise PtxError("%s: %s: operand %s of type .%s%d does not agree with the instruction type .%s%d "
                                   "[PTX ISA 9.3 9.4, Table %s]" % (fname, opc, r.split("@")[0], ot[0], ot[1], it[0], it[1],
                                                                    "27 / 28" if rel else "26"))

def serialize(mod):
    lines = []
    for td in mod:
        k = td[0]
        if k == "version":
            lines.append("VERSION " + str(float_bits(td[1]) if re.search(r"[.eE]", td[1]) else float_bits(td[1] + ".0")))
        elif k == "target":
            lines.append("TARGET %d %s" % (len(td[1]), " ".join(td[1])))
        elif k == "address_size":
            lines.append("ASZ %d" % td[1])
        elif k == "alias":
            lines.append("ALIAS %s %s" % (td[1], td[2]))
        elif k in ("skip", "texdecl", "proto"):
            pass
        elif k == "decl":
            _, a1, t, a2, items, init = td
            st = aset(a1) | aset(a2)
            ty = type_of(t)
            sds = sdecls(st, align_or(a1, a2, ty, st), ty, items, init)
            lines.append("DECL %d %s %d %s" % (len(st), " ".join(sorted(st)), len(sds), " ".join(ser_sdecl(d) for d in sds)))
        elif k == "func":
            _, a1, kw, rets, name, params, pd, body = td
            st = aset(a1)
            fname, fkind, rs, ps = func_head(kw, rets, name, params)
            ld = Loader(fmarks(rs + ps))
            ld.pp(body)
            colour_fields(ld)
            check_regs(fname, ld)
            if any(t[0] == "isacheck" for t in mod):
                check_types(fname, ld)
            labels = {k2: v for k2, v in ld.L.items() if not (k2.startswith("@ren:") or k2 in ("@depth", "@renid"))}
            lab_lines = []
            tgt_lines = []
            for k2, v in labels.items():
                if isinstance(v, list):
                    tgt_lines.append("%s %d %s" % (k2, len(v), " ".join(v)))
                else:
                    lab_lines.append("%s %d" % (k2, v))
            rw_items = ["%s %d" % kv for kv in ld.rw.items()]
            lines.append("FUNC %s %s %d %s PERF %d %s RETS %d %s PARAMS %d %s RW %d %s LABELS %d %s TARGETS %d %s CODE %d"
                         % (fname, fkind, len(st), " ".join(sorted(st)), len(pd), ser_perf(pd),
                            len(rs), " ".join(ser_pdecl(p) for p in rs), len(ps), " ".join(ser_pdecl(p) for p in ps),
                            len(rw_items), " ".join(rw_items), len(lab_lines), " ".join(lab_lines),
                            len(tgt_lines), " ".join(tgt_lines), len(ld.code)))
            for c in ld.code:
                lines.append(ser_code_item(c))
            lines.append("STATICS %d %s" % (len(ld.sh), " ".join(ser_sdecl(d) for d in ld.sh)))
        elif k == "buffer":
            _, n, t, c, vals = td
            ty = type_of(t)
            vals = iv_list(vals)
            lines.append("BUFFER %s %s %d %d %s" % (n, ser_ty(ty), len(vals) if c is None else c, len(vals), " ".join(ser_init(v) for v in vals)))
        elif k in ("launch", "cooplaunch"):
            _, kn, g, b, dyn, args = td
            lines.append("%s %s %d %d %d %d %d %d %d %d %s" % ("LAUNCH" if k == "launch" else "COOPLAUNCH", kn, *g, *b, dyn, len(args),
                                                           " ".join(ser_sop(a) for a in args)))
        elif k == "print":
            lines.append("PRINT %s %s %d" % (td[1], ser_ty(type_of(td[2])), td[3]))
        elif k in ("expect", "patch", "append"):
            _, n, t, off, vals = td
            vals = iv_list(vals)
            lines.append("%s %s %s %d %d %s" % (k.upper(), n, ser_ty(type_of(t)), off, len(vals), " ".join(ser_init(v) for v in vals)))
        elif k == "tensormap":
            _, n, t, b, ds, ss, bs, z = td
            lines.append("TENSORMAP %s %s %s %d %s %d %s %d %s %d" % (n, ser_ty(type_of(t)), b, len(ds), " ".join(map(str, ds)), len(ss),
                                                                     " ".join(map(str, ss)), len(bs), " ".join(map(str, bs)), z))
        elif k == "tensormapi":
            _, n, t, b, ds, ss, bs, z, ls, us, es = td
            lines.append("TENSORMAPI %s %s %s %s %d" % (n, ser_ty(type_of(t)), b, " ".join("%d %s" % (len(x), " ".join(map(str, x))) for x in (ds, ss, bs)), z) +
                         " " + " ".join("%d %s" % (len(x), " ".join(map(str, x))) for x in (ls, us, es)))
        elif k in ("texture", "surface"):
            _, n, t, b, ds, at = td
            lines.append("%s %s %s %s %d %s %d %s" % (k.upper(), n, ser_ty(type_of(t)), b, len(ds), " ".join(map(str, ds)), len(at), " ".join(map(str, at))))
        elif k == "sampler":
            lines.append("SAMPLER %s %d %s" % (td[1], len(td[2]), " ".join(map(str, td[2]))))
        elif k == "multimem":
            lines.append("MULTIMEM %s %d %d %s" % (td[1], td[2], len(td[3]), " ".join(td[3])))
        elif k == "fabric":
            lines.append("FABRIC %d %d %d %s" % (td[1], td[2], len(td[3]), " ".join(td[3])))
        elif k in ("memmodel", "ub", "convergence"):
            lines.append("%s %s" % (k.upper(), td[1]))
        elif k == "occupancy":
            lines.append("OCCUPANCY %d" % td[1])
        elif k == "isacheck":
            lines.append("ISACHECK")
        else:
            raise PtxError("unhandled top-level item %r" % (k,))
    lines.append("END")
    return "\n".join(lines) + "\n"

def front(text):
    return serialize(Parser(lex(text)).module())

if __name__ == "__main__":
    sys.stdout.write(front(open(sys.argv[1]).read()))

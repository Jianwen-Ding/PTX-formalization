/* =============================================================================
 * PTX-HARNESS (semantics/ptx-harness.k): the host side.  The module is processed top to bottom:
 * PTX directives load functions and variables, harness directives allocate buffers, launch grids
 * (synchronously) and print / check results.  The token stream is written by sail/ptx/ptxfront.py.
 * ============================================================================= */

/* ---------------------------------------------------------------- reading the token stream */
function tk_ty() = { let k = tok_str(); let w = tok_int(); mkty(k, w) }

val tk_sop : unit -> sop
function tk_sop() = match tok_str() {
  "R" => SR(tok_str()),
  "N" => SN(tok_str()),
  "I" => SI(tok_int()),
  "F" => SF(tok_int()),
  "H" => SH(tok_str()),
  "NP" => { let n = tok_str(); let i = tok_int(); SNP(n, i) },
  "NEG" => { let k = tok_str(); let n = tok_str(); SNEG(k == "r", n) },
  "NOT" => { let k = tok_str(); let n = tok_str(); SNOT(k == "r", n) },
  "PIPE" => { let k1 = tok_str(); let a = tok_str(); let k2 = tok_str(); let b = tok_str(); SPIPE(k1 == "r", a, k2 == "r", b) },
  s => { print_endline(concat_str("bad operand token ", s)); SI(0) }
}

val tk_sops : int -> list(sop)
function tk_sops(n) = { var l : list(sop) = [||]; foreach (i from 1 to n) l = tk_sop() :: l; reverse_list(l) }

val tk_addr : unit -> addr
function tk_addr() = match tok_str() {
  "AR" => AR(tok_str()),
  "AN" => AN(tok_str()),
  "AI" => AI(tok_int()),
  "ARO" => { let n = tok_str(); let i = tok_int(); ARO(n, i) },
  _ => { let n = tok_str(); let i = tok_int(); ANO(n, i) }
}

val tk_iop : unit -> iop
function tk_iop() = match tok_str() { "V" => IV(tk_sops(tok_int())), _ => IS(tk_sop()) }

val tk_xop : unit -> xop
function tk_xop() = match tok_str() {
  "S" => XS(tk_sop()),
  "V" => XV(tk_sops(tok_int())),
  "P" => XP(tk_sops(tok_int())),
  "VP" => { let l = tk_sops(tok_int()); let k = tok_str(); let n = tok_str(); XVP(l, k == "r", n) },
  _ => {
    let na = tok_int();
    var al : list(addr) = [||];
    foreach (i from 1 to na) al = tk_addr() :: al;
    let ni = tok_int();
    var il : list(iop) = [||];
    foreach (i from 1 to ni) il = tk_iop() :: il;
    XB(reverse_list(al), reverse_list(il))
  }
}

val tk_strs : int -> list(string)
function tk_strs(n) = { var l : list(string) = [||]; foreach (i from 1 to n) l = tok_str() :: l; reverse_list(l) }

val tk_item : unit -> item
function tk_item() = match tok_str() {
  "DV" => { let sp = tok_str(); let n = tok_str(); let sz = tok_int(); let al = tok_int(); DECLVAR(sp, n, sz, al) },
  _ => {
    let g : option((string, bool)) = match tok_str() { "G1" => { let p = tok_str(); let neg = tok_int(); Some((p, neg != 0)) }, _ => None() };
    let base = tok_str();
    let mods = tk_strs(tok_int());
    let nt = tok_int();
    var tys : list(ty) = [||];
    foreach (i from 1 to nt) tys = tk_ty() :: tys;
    let no = tok_int();
    var ops : list(xop) = [||];
    foreach (i from 1 to no) ops = tk_xop() :: ops;
    INS(g, struct { base = base, mods = mods, tys = reverse_list(tys) }, reverse_list(ops))
  }
}

val tk_pdecls : int -> list(pdecl)
function tk_pdecls(n) = {
  var l : list(pdecl) = [||];
  foreach (i from 1 to n) {
    let nm = tok_str(); let t = tk_ty(); let sz = tok_int(); let al = tok_int();
    l = struct { pname = nm, pty = t, psize = sz, palign = al } :: l
  };
  reverse_list(l)
}

val tk_initval : unit -> initval
function tk_initval() = match tok_str() {
  "I" => IVI(tok_int()),
  "F" => IVF(tok_int()),
  "H" => IVH(tok_str()),
  "N" => IVN(tok_str()),
  "NI" => { let n = tok_str(); let i = tok_int(); IVNI(n, i) },
  "ND" => { let a = tok_str(); let b = tok_str(); IVND(a, b) },
  "G" => IVG(tok_str()),
  "M" => { let m = tok_int(); let k = tok_int(); let n = tok_str(); let o = tok_int(); IVM(m, k, n, o) },
  _ => { let n = tok_str(); let i = tok_int(); IVGO(n, i) }
}

val tk_initvals : int -> list(initval)
function tk_initvals(n) = { var l : list(initval) = [||]; foreach (i from 1 to n) l = tk_initval() :: l; reverse_list(l) }

val tk_sdecls : int -> list(sdecl)
function tk_sdecls(n) = {
  var l : list(sdecl) = [||];
  foreach (i from 1 to n) {
    let _ = tok_str();   /* SD */
    let sp = tok_str(); let nm = tok_str(); let t = tk_ty(); let sz = tok_int(); let al = tok_int(); let ex = tok_int();
    let init = tk_initvals(tok_int());
    l = struct { sspace = sp, sname = nm, sty = t, ssize = sz, salign = al, sextern = ex != 0, sinit = init } :: l
  };
  reverse_list(l)
}

val tk_ints : int -> list(int)
function tk_ints(n) = { var l : list(int) = [||]; foreach (i from 1 to n) l = tok_int() :: l; reverse_list(l) }

/* ---------------------------------------------------------------- initializers and memory images */
val symRaw : string -> int
function symRaw(n) = if smap_has(SP_SYMS, n) then sym_addr(smap_get(SP_SYMS, n)) else 0
val symGen : string -> int
function symGen(n) = if smap_has(SP_SYMS, n) then winBase(sym_space(smap_get(SP_SYMS, n))) + sym_addr(smap_get(SP_SYMS, n)) else 0

/* the number of trailing zero bits of a positive integer */
val ctz : int -> int
function ctz(m) = { var n : int = 0; while m > 0 & land_int(asr_int(m, n), 1) == 0 do n = n + 1; n }

/* resolveV(S, V) then initBits(V, T); unresolvable initializers are stuck */
val initBits : (initval, ty) -> int
function initBits(v, t) = match v {
  IVI(i) => intLit(i, t),
  IVF(f) => fltLit(f, t),
  IVH(h) => hexBits(h, t),
  IVN(n) => intLit(symRaw(n), t),
  IVNI(n, i) => intLit(symRaw(n) + i, t),
  IVND(a, b) => intLit(symRaw(a) - symRaw(b), t),
  IVG(n) => intLit(symGen(n), t),
  IVGO(n, i) => intLit(symGen(n) + i, t),
  /* mask(expr) (PTX ISA 5.4.4): the bits of the mask, moved to the lowest position */
  IVM(m, k, n, o) => intLit(asr_int(land_int(if k == 0 then o else if k == 1 then symRaw(n) + o else symGen(n) + o, m), ctz(m)), t)
}

/* initBits without symbol resolution (buffer and .expect values) */
val initBitsRaw : (initval, ty) -> int
function initBitsRaw(v, t) = match v {
  IVI(i) => intLit(i, t),
  IVF(f) => fltLit(f, t),
  IVH(h) => hexBits(h, t),
  _ => { fault("symbolic value in a harness directive", 0); 0 }
}

val initMem : (int, ty, list(initval), bool) -> unit
function initMem(a, t, vs, resolve) = match vs {
  [||] => (),
  v :: rest => { mem_wrn(SP_GMEM, a, bytes(t), if resolve then initBits(v, t) else initBitsRaw(v, t)); initMem(a + bytes(t), t, rest, resolve) }
}

/* ---------------------------------------------------------------- linking and static allocation */
val lkKind : list(string) -> int
function lkKind(s) = if has(s, "weak") | has(s, "common") then 1 else 0

/* lkAct: 0 take the definition, 1 it yields, 2 two strong definitions */
val lkAct : (string, int) -> int
function lkAct(n, k) = {
  if not_bool(smap_has(SP_LINKD, n)) then 0
  else if k == 0 & smap_get(SP_LINKD, n) == 0 then 2
  else if k == 0 then 0
  else 1
}

val do_static : list(sdecl) -> unit
function do_static(l) = match l {
  [||] => (),
  d :: rest => {
    if d.sspace == "shared" & d.sextern then { smap_set(SP_SYMS, d.sname, mksym("dynshared", 0)); dyn_names = d.sname :: dyn_names }
    else if d.sspace == "shared" then {
      let a = alignUp(stop, d.salign);
      smap_set(SP_SYMS, d.sname, mksym("shared", a));
      stop = a + d.ssize
    } else if (d.sspace == "global" | d.sspace == "const") & not_bool(d.sextern) then {
      let a = alignUp(gtop, d.salign);
      gtop = a + max_int(d.ssize, 1);
      initMem(a, d.sty, d.sinit, true);
      smap_set(SP_SYMS, d.sname, mksym(d.sspace, a))
    };
    do_static(rest)
  }
}

/* #link(A, sdecls, Acc) */
val do_link : (list(string), list(sdecl)) -> unit
function do_link(attrs, l) = {
  var acc : list(sdecl) = [||];
  var x = l;
  var ok : bool = true;
  while ok & is_cons(x) do {
    match x {
      d :: rest => {
        if d.sextern | not_bool(d.sspace == "global" | d.sspace == "const") then acc = d :: acc
        else {
          let a = lkAct(d.sname, lkKind(attrs));
          if a == 0 then { acc = d :: acc; smap_set(SP_LINKD, d.sname, lkKind(attrs)) }
          else if a == 2 then { fault(concat_str("multiple definition of ", d.sname), 0); ok = false }
        };
        x = rest
      },
      [||] => ()
    }
  };
  if ok then do_static(reverse_list(acc))
}

/* ---------------------------------------------------------------- functions */
val perf_of : string -> list((string, list(int)))
function perf_of(k) = {
  var x = perfd;
  var r : list((string, list(int))) = [||];
  while is_cons(x) do { match x { (n, p) :: rest => { if n == k then r = p; x = rest }, [||] => () } };
  r
}

val clusterOf : list((string, list(int))) -> dim3
function clusterOf(p) = match p {
  [||] => mkdim(1, 1, 1),
  (d, xs) :: rest => if d == "reqnctapercluster" then mkdim(nthD(xs, 0, 1), nthD(xs, 1, 1), nthD(xs, 2, 1)) else clusterOf(rest)
}

val read_func : unit -> unit
function read_func() = {
  let name = tok_str();
  let kind = tok_str();
  let attrs = tk_strs(tok_int());
  let _ = tok_str();   /* PERF */
  let np = tok_int();
  var pd : list((string, list(int))) = [||];
  foreach (i from 1 to np) { let d = tok_str(); let xs = tk_ints(tok_int()); pd = (d, xs) :: pd };
  pd = reverse_list(pd);
  let _ = tok_str();   /* RETS */
  let rets = tk_pdecls(tok_int());
  let _ = tok_str();   /* PARAMS */
  let params = tk_pdecls(tok_int());
  let _ = tok_str();   /* RW */
  let nrw = tok_int();
  let rw = new_space();
  foreach (i from 1 to nrw) { let n = tok_str(); let w = tok_int(); smap_set(rw, n, w) };
  let _ = tok_str();   /* LABELS */
  let nl = tok_int();
  let lab = new_space();
  foreach (i from 1 to nl) { let n = tok_str(); let p = tok_int(); smap_set(lab, n, p) };
  let _ = tok_str();   /* TARGETS */
  let nt = tok_int();
  var tl : list((string, list(string))) = [||];
  foreach (i from 1 to nt) { let n = tok_str(); let ts = tk_strs(tok_int()); tl = (n, ts) :: tl };
  let _ = tok_str();   /* CODE */
  let nc = tok_int();
  let base = ncode;
  foreach (i from 1 to nc) add_item(tk_item());
  let _ = tok_str();   /* STATICS */
  let sh = tk_sdecls(tok_int());
  if not_bool(stuck) then {
    let key = concat_str("func:", name);
    let a = lkAct(key, lkKind(attrs));
    if a == 1 then ()
    else if a == 2 then fault(concat_str("multiple definition of function ", name), 0)
    else if not_bool(has(attrs, "extern")) then {
      smap_set(SP_LINKD, key, lkKind(attrs));
      perfd = (name, pd) :: perfd;
      let f = fchk(nfuncs);
      nfuncs = nfuncs + 1;
      f_name[f] = name; f_kind[f] = kind; f_rets[f] = rets; f_params[f] = params;
      f_base[f] = base; f_len[f] = nc; f_rw[f] = rw; f_lab[f] = lab;
      var x = tl;
      while is_cons(x) do { match x { (n, ts) :: rest => { tgts = (f, n, ts) :: tgts; x = rest }, [||] => () } };
      smap_set(SP_FUNCS, name, f + 1);
      /* every function also gets an address (symbol F) */
      if not_bool(smap_has(SP_SYMS, name)) then {
        let fa = 536870912 + 64 * nfaddr;
        smap_set(SP_SYMS, name, mksym("func", fa));
        smap_set(SP_FADDRS, dec_str(fa), f + 1);
        nfaddr = nfaddr + 1
      } else {
        /* a later definition of the same name keeps the address and maps it to the new code */
        let fa = sym_addr(smap_get(SP_SYMS, name));
        if smap_has(SP_FADDRS, dec_str(fa)) then smap_set(SP_FADDRS, dec_str(fa), f + 1)
      };
      do_static(sh)
    }
  }
}

/* ---------------------------------------------------------------- buffers */
function do_alloc(n, t, c, vs) = {
  let a = alignUp(gtop, 256);
  gtop = a + max_int(c * bytes(t), 1);
  smap_set(SP_SYMS, n, mksym("global", a));
  bufs = (n, a, t, c) :: bufs;
  initMem(a, t, vs, false)
}

val buf_find : string -> option((int, ty, int))
function buf_find(n) = {
  var x = bufs;
  var r : option((int, ty, int)) = None();
  while is_cons(x) & is_none(r) do { match x { (m, a, t, c) :: rest => { if m == n then r = Some((a, t, c)); x = rest }, [||] => () } };
  r
}

val buf_update : (string, int) -> unit
function buf_update(n, c2) = {
  var x = bufs;
  var out : list((string, int, ty, int)) = [||];
  var done : bool = false;
  while is_cons(x) do {
    match x { (m, a, t, c) :: rest => { if m == n & not_bool(done) then { out = (m, a, t, c2) :: out; done = true } else out = (m, a, t, c) :: out; x = rest }, [||] => () }
  };
  bufs = reverse_list(out)
}

/* ---------------------------------------------------------------- launch */
val lcHw : (dim3, dim3) -> string
function lcHw(b, g) = {
  if b.x > 1024 | b.y > 1024 | b.z > 64 | b.x * b.y * b.z > 1024 then "block dimensions exceed 1024 x 1024 x 64 or 1024 threads"
  else if g.x > 2147483647 | g.y > 65535 | g.z > 65535 then "grid dimensions exceed 2^31-1 x 65535 x 65535"
  else if b.x <= 0 | b.y <= 0 | b.z <= 0 | g.x <= 0 | g.y <= 0 | g.z <= 0 then "empty block or grid"
  else ""
}

/* .maxntid bounds the total number of threads, the product of the extents (PTX ISA 11.4.2), not each dimension */
val dimLe : (dim3, list(int)) -> bool
function dimLe(d, l) = match l {
  [|a, b, c|] => d.x * d.y * d.z <= a * b * c,
  [|a, b|] => d.x * d.y * d.z <= a * b,
  [|a|] => d.x * d.y * d.z <= a,
  _ => true
}
val dimEq : (dim3, list(int)) -> bool
function dimEq(d, l) = match l {
  [|a, b, c|] => d.x == a & d.y == b & d.z == c,
  [|a, b|] => d.x == a & d.y == b & d.z == 1,
  [|a|] => d.x == a & d.y == 1 & d.z == 1,
  _ => true
}

val lcErr : (list((string, list(int))), dim3, dim3) -> string
function lcErr(p, b, cd) = match p {
  [||] => "",
  (d, xs) :: rest => {
    if d == "maxntid" & not_bool(dimLe(b, xs)) then "launch exceeds the kernel's .maxntid"
    else if d == "reqntid" & not_bool(dimEq(b, xs)) then "launch does not match the kernel's .reqntid"
    else if d == "maxclusterrank" & cd.x * cd.y * cd.z > nthD(xs, 0, 0) then "cluster larger than the kernel's .maxclusterrank"
    else lcErr(rest, b, cd)
  }
}

/* argVal(A, T, S) */
val argVal : (sop, ty) -> int
function argVal(a, t) = match a {
  SN(n) => if smap_has(SP_SYMS, n) then sym_addr(smap_get(SP_SYMS, n)) else 0,
  SI(i) => intLit(i, t),
  SF(f) => fltLit(f, t),
  SH(h) => hexBits(h, t),
  _ => 0
}


val hasPerf : (list((string, list(int))), string) -> bool
function hasPerf(p, d) = match p { [||] => false, (n, _) :: rest => n == d | hasPerf(rest, d) }

val do_launch : (string, dim3, dim3, int, list(sop)) -> unit
function do_launch(k, g, b, dyn, args) = {
  isa_check();
  if stuck then ()
  else match lookup_func(k) {
    None() => fault(concat_str("cannot launch ", k), 0),
    Some(f) => {
      let ps = f_params[fchk(f)];
      if f_kind[fchk(f)] != "entry" | list_length(ps) != sop_count(args) then fault(concat_str("cannot launch ", k), 0)
      else {
        let cd = clusterOf(perf_of(k));
        let hw = lcHw(b, g);
        if hw != "" then fault(concat_str("cannot launch ", concat_str(k, concat_str(": ", hw))), 0)
        else {
          /* the harness has no launch-time cluster dimensions: only .reqnctapercluster specifies them (PTX ISA 11.7.2) */
          let le = if hasPerf(perf_of(k), "explicitcluster") & not_bool(hasPerf(perf_of(k), "reqnctapercluster"))
                   then "a kernel with .explicitcluster launched without cluster dimensions"
                   else lcErr(perf_of(k), b, cd);
          if le != "" then fault(concat_str("cannot launch: ", le), 0)
          else {
            /* kernel parameter image and environment */
            smap_clear(SP_KENV);
            var o : int = 0;
            var x = ps;
            var y = args;
            while is_cons(x) do {
              match (x, y) {
                (p :: pr, a :: ar) => {
                  let off = alignUp(o, p.palign);
                  mem_wrn(SP_KPARAMS, off, p.psize, argVal(a, p.pty));
                  smap_set(SP_KENV, p.pname, mksym("param", off));
                  o = off + p.psize;
                  x = pr; y = ar
                },
                _ => x = [||]
              }
            };
            if o > 32764 then fault(concat_str("kernel parameters of ", concat_str(dec_str(o), " bytes exceed the 32764-byte limit")), o)
            else {
              kptop = o;
              /* fixDyn: `.extern .shared` arrays start after the static shared data */
              var dn = dyn_names;
              while is_cons(dn) do {
                match dn { n :: rest => { if smap_get(SP_SYMS, n) == mksym("dynshared", 0) then smap_set(SP_SYMS, n, mksym("shared", alignUp(stop, DYNSHARED))); dn = rest }, [||] => () }
              };
              kernel = k; kfid = f; gridDim = g; blockDim = b; clusterDim = cd;
              dynsmem = dyn; explcl = hasPerf(perf_of(k), "reqnctapercluster");
              nextcta = 0; resident = 0; nthreads = 0; ncta_slots = 0;
              colls = [||];
              let lim = if occ > 0 & occ < vol(g) then occ else vol(g);
              mkCtas(0, lim);
              turn = -1;
              run_grid()
            }
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------- inspection */
val fmtList : (int, ty, int) -> string
function fmtList(a, t, n) = {
  var s : string = "";
  foreach (i from 0 to (n - 1)) {
    let v = fmtVal(mem_rdn(SP_GMEM, a + i * bytes(t), bytes(t)), t);
    s = if i == 0 then v else concat_str(s, concat_str(", ", v))
  };
  s
}

val do_print : (string, ty, int) -> unit
function do_print(n, t, c) = {
  if smap_has(SP_SYMS, n) then {
    let a = sym_addr(smap_get(SP_SYMS, n));
    let cnt : int = if c >= 0 then c else match buf_find(n) { Some((_, bt, bc)) => tdiv_int(bc * bytes(bt), bytes(t)), None() => 1 };
    print(concat_str(n, concat_str(" = [", concat_str(fmtList(a, t, cnt), "]\n"))))
  } else fault(concat_str("print of unknown symbol ", n), 0)
}

val do_expect : (string, ty, list(initval), int) -> unit
function do_expect(n, t, vs, off) = {
  if smap_has(SP_SYMS, n) then {
    let a0 = sym_addr(smap_get(SP_SYMS, n)) + off * bytes(t);
    var mis : string = "";
    var x = vs;
    var i : int = off;
    var a : int = a0;
    while is_cons(x) do {
      match x {
        v :: rest => {
          let got = mem_rdn(SP_GMEM, a, bytes(t));
          let exp = initBitsRaw(v, t);
          if got != exp then
            mis = concat_str(mis, concat_str("  [", concat_str(dec_str(i), concat_str("] expected ", concat_str(fmtVal(exp, t),
                  concat_str(", got ", concat_str(fmtVal(got, t), "\n")))))));
          i = i + 1; a = a + bytes(t); x = rest
        },
        [||] => ()
      }
    };
    if mis == "" then print(concat_str("PASS ", concat_str(n, "\n")))
    else { fails = fails + 1; print(concat_str("FAIL ", concat_str(n, concat_str("\n", mis)))) }
  } else fault(concat_str("expect of unknown symbol ", n), 0)
}


/* ---------------------------------------------------------------- the module */
val targetSm : list(string) -> int
function targetSm(l) = match l {
  [||] => 0,
  s :: rest => {
    if string_length(s) >= 5 & stake(s, 3) == "sm_" then {
      var i : int = 3;
      var v : int = 0;
      while i < string_length(s) & hex_digit(stake(sdrop(s, i), 1)) < 10 & is_dec_digit(stake(sdrop(s, i), 1)) do {
        v = v * 10 + hex_digit(stake(sdrop(s, i), 1)); i = i + 1
      };
      v
    } else targetSm(rest)
  }
}
val targetArch : list(string) -> string
function targetArch(l) = match l {
  [||] => "",
  s :: rest => {
    if string_length(s) >= 5 & stake(s, 3) == "sm_" then {
      var i : int = 3;
      while i < string_length(s) & is_dec_digit(stake(sdrop(s, i), 1)) do i = i + 1;
      sdrop(s, i)
    } else targetArch(rest)
  }
}

val main : unit -> unit
function main() = {
  var go : bool = true;
  while go do {
    let d = tok_str();
    if stuck then go = false
    else match d {
      "VERSION" => {
        let f = tok_int();
        isaver = f2int(rndInt(fv_add_exact(fv_mul_exact(f64v(f), FV_Fin(false, 10, 0)), FV_Fin(false, 1, -1), RN), "rmi"))
      },
      "TARGET" => { let ns = tk_strs(tok_int()); isasm = targetSm(ns); isaarch = targetArch(ns) },
      "ASZ" => { let n = tok_int(); asz32 = n == 32; if asz32 then smap_set(SP_SYMS, "%%asz", mksym("asz", 32)) },
      "ISACHECK" => isachk = true,
      "ALIAS" => {
        let a = tok_str(); let b = tok_str();
        if smap_has(SP_FUNCS, b) then smap_set(SP_FUNCS, a, smap_get(SP_FUNCS, b))
      },
      "DECL" => { let attrs = tk_strs(tok_int()); let sds = tk_sdecls(tok_int()); do_link(attrs, sds) },
      "FUNC" => read_func(),
      "BUFFER" => { let n = tok_str(); let t = tk_ty(); let c = tok_int(); let vs = tk_initvals(tok_int()); do_alloc(n, t, c, vs) },
      "LAUNCH" => {
        let k = tok_str();
        let g = mkdim(tok_int(), tok_int(), tok_int());
        let b = mkdim(tok_int(), tok_int(), tok_int());
        let dyn = tok_int();
        let args = tk_sops(tok_int());
        do_launch(k, g, b, dyn, args)
      },
      "COOPLAUNCH" => {
        let k = tok_str();
        let g = mkdim(tok_int(), tok_int(), tok_int());
        let b = mkdim(tok_int(), tok_int(), tok_int());
        let dyn = tok_int();
        let args = tk_sops(tok_int());
        if occ > 0 & vol(g) > occ then
          fault(concat_str("cooperative launch of ", concat_str(dec_str(vol(g)), " CTAs exceeds the resident-CTA limit")), 0)
        else do_launch(k, g, b, dyn, args)
      },
      "PRINT" => { let n = tok_str(); let t = tk_ty(); let c = tok_int(); do_print(n, t, c) },
      "EXPECT" => { let n = tok_str(); let t = tk_ty(); let off = tok_int(); let vs = tk_initvals(tok_int()); do_expect(n, t, vs, off) },
      "PATCH" => {
        let n = tok_str(); let t = tk_ty(); let off = tok_int(); let vs = tk_initvals(tok_int());
        if smap_has(SP_SYMS, n) then initMem(sym_addr(smap_get(SP_SYMS, n)) + off * bytes(t), t, vs, false)
        else fault(concat_str("patch of unknown symbol ", n), 0)
      },
      "APPEND" => {
        let n = tok_str(); let t = tk_ty(); let _ = tok_int(); let vs = tk_initvals(tok_int());
        match buf_find(n) {
          Some((a, t0, c)) => {
            if ty_eq(t, t0) then {
              let c2 = c + list_length(vs);
              initMem(a + c * bytes(t0), t0, vs, false);
              buf_update(n, c2);
              gtop = max_int(gtop, a + c2 * bytes(t0))
            } else fault("append with a different element type", 0)
          },
          None() => fault(concat_str("append to unknown buffer ", n), 0)
        }
      },
      "MEMMODEL" => { memmodel = tok_str(); if memmodel != "sc" then fault("the relaxed memory model is not modelled", 0) },
      "UB" => ub = tok_str() == "strict",
      "OCCUPANCY" => occ = tok_int(),
      "CONVERGENCE" => its = tok_str() == "its",
      "END" => go = false,
      "" => go = false,
      _ => { if not_bool(harness_ext(d)) then { fault(concat_str("unknown directive ", d), 0); go = false } }
    }
  };
  if not_bool(stuck) & fails == 0 then exit_code = 0;
  if stuck then prerr_endline(concat_str("stuck: ", stuck_msg));
  prerr_endline(concat_str("EXIT ", dec_str(exit_code)))
}

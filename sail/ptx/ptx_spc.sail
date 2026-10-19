/* =============================================================================
 * PTX-EXEC-SPC (semantics/ptx-exec-spc.k): spcompress / spdecompress, structured-sparsity
 * (de)compression of register vectors; the selection helper is shared with tcgen05.ld.spcompress.
 * ============================================================================= */

/* ---------------------------------------------------------------- bit vectors */
val bitsOf : list(int) -> int
function bitsOf(ws) = {
  var r : int = 0;
  var i : int = 0;
  var x = ws;
  while is_cons(x) do { match x { w :: rest => { r = lor_int(r, shl_int(land_int(w, 4294967295), 32 * i)); i = i + 1; x = rest }, [||] => () } };
  r
}
val wordsOf : (int, int) -> list(int)
function wordsOf(v, n) = { var out : list(int) = [||]; foreach (i from 0 to (n - 1)) out = pend_append(out, [|land_int(asr_int(v, 32 * i), 4294967295)|]); out }
val elemsOf : (int, int, int) -> list(int)
function elemsOf(v, b, n) = { var out : list(int) = [||]; foreach (i from 0 to (n - 1)) out = pend_append(out, [|land_int(asr_int(v, b * i), mask(b))|]); out }
val packElems : (list(int), int) -> int
function packElems(es, b) = {
  var r : int = 0;
  var i : int = 0;
  var x = es;
  while is_cons(x) do { match x { e :: rest => { r = lor_int(r, shl_int(land_int(e, mask(b)), b * i)); i = i + 1; x = rest }, [||] => () } };
  r
}

/* ---------------------------------------------------------------- selection of two elements out of four */
val spcKey : (int, fv) -> fv
function spcKey(op, f) = if fmod_int(op, 2) == 1 then fv_abs(f) else f

val fv_nth : (list(fv), int) -> fv
function fv_nth(l, i) = match l { [||] => FV_Fin(false, 0, 0), h :: t => if i == 0 then h else fv_nth(t, i - 1) }

val spcBefore : (int, list(fv), int, int) -> bool
function spcBefore(op, fs, j, i) = {
  let fj = fv_nth(fs, j);
  let fi = fv_nth(fs, i);
  if fv_is_nan(fj) then not_bool(fv_is_nan(fi)) | j < i
  else not_bool(fv_is_nan(fi)) &
       ((if op < 2 then fv_lt(spcKey(op, fi), spcKey(op, fj)) else fv_lt(spcKey(op, fj), spcKey(op, fi)))
        | (fv_eq(spcKey(op, fj), spcKey(op, fi)) & j < i))
}

val spcSel : (int, list(fv)) -> list(int)
function spcSel(op, fs) = {
  var out : list(int) = [||];
  foreach (i from 0 to 3) {
    var r : int = 0;
    foreach (j from 0 to 3) if j != i & spcBefore(op, fs, j, i) then r = r + 1;
    if r < 2 then out = pend_append(out, [|i|])
  };
  out
}

val spcVal : (int, int, int) -> fv
function spcVal(t, es, b) =
  if t == 0 & es == 16 then elemF(mkty("f", 16), b)
  else if t == 1 & es == 16 then elemF(mkty("bf", 16), b)
  else if t == 0 & es == 8 then fv_of_int(land_int(b, 255))
  else if t == 1 & es == 8 then fv_of_int(if land_int(b, 255) >= 128 then land_int(b, 255) - 256 else land_int(b, 255))
  else if t == 6 then elemF(mkty("f", 32), b)
  else if t == 2 then elemF(mkty("e5m2", 8), land_int(b, 255))
  else if t == 3 then elemF(mkty("e4m3", 8), land_int(b, 255))
  else if t == 4 then elemF(mkty("e3m2", 8), land_int(b, 63))
  else if t == 5 then elemF(mkty("e2m3", 8), land_int(b, 63))
  else FV_Fin(false, 0, 0)

/* spcKeep: the kept elements (in index order) and their indices of every group of four */
val spcKeep : (int, int, int, list(int)) -> (list(int), list(int))
function spcKeep(op, t, es, elems) = {
  var cs : list(int) = [||];
  var ms : list(int) = [||];
  var x = elems;
  while list_length(x) >= 4 do {
    let g = [|at(x, 0), at(x, 1), at(x, 2), at(x, 3)|];
    let sel = spcSel(op, [|spcVal(t, es, at(g, 0)), spcVal(t, es, at(g, 1)), spcVal(t, es, at(g, 2)), spcVal(t, es, at(g, 3))|]);
    var s = sel;
    while is_cons(s) do { match s { i :: rest => { cs = pend_append(cs, [|at(g, i)|]); s = rest }, [||] => () } };
    ms = pend_append(ms, sel);
    x = ldrop(x, 4)
  };
  (cs, ms)
}

val spcFin : ((list(int), list(int)), int, int, int, int) -> (list(int), list(int))
function spcFin(k, es, ib, nm, nc) = { let (cs, ms) = k; (wordsOf(packElems(ms, ib), nm), wordsOf(packElems(cs, es), nc)) }

val spcNum : list(string) -> int
function spcNum(m) = if has(m, "x64") then 64 else if has(m, "x32") then 32 else if has(m, "x16") then 16 else if has(m, "x8") then 8
                     else if has(m, "x4") then 4 else if has(m, "x2") then 2 else 1
val spcES : list(ty) -> int
function spcES(ts) = { var r : int = 8; var l = ts; while is_cons(l) do { match l { t :: rest => { if ty_eq(t, mkty("b", 16)) then r = 16; l = rest }, [||] => () } }; r }
val spcIB : list(string) -> int
function spcIB(m) = if has(m, "b4") then 4 else 2

val spfSrc : list(string) -> int
function spfSrc(m) =
  if has(m, "sp::1:2") | has(m, "sp::1:4") | has(m, "sp::1:8") | has(m, "sp::1:16") then 1
  else if has(m, "sp::2:4") | has(m, "sp::2:8") | has(m, "sp::2:16") then 2
  else if has(m, "sp::4:8") | has(m, "sp::4:16") then 4 else 2
val spfDst : list(string) -> int
function spfDst(m) =
  if has(m, "sp::1:2") then 2
  else if has(m, "sp::1:4") | has(m, "sp::2:4") then 4
  else if has(m, "sp::1:8") | has(m, "sp::2:8") | has(m, "sp::4:8") then 8
  else if has(m, "sp::1:16") | has(m, "sp::2:16") | has(m, "sp::4:16") then 16 else 4

val exec_spc : (opcode, list(xop)) -> bool
function exec_spc(o, ops) = {
  let m = o.mods;
  let b32 = mkty("b", 32);
  match (o.base, ops) {
    ("spcompress", [|mv, cv, dv, sd|]) => {
      let s = rd(sd, b32);
      let es = spcES(o.tys);
      let ds = vecRegs(dv, b32);
      let (mw, cw_) = spcFin(spcKeep(land_int(s, 3), land_int(asr_int(s, 2), 7), es, elemsOf(bitsOf(ds), es, tdiv_int(list_length(ds) * 32, es))),
                            es, spcIB(m), xop_count(vecOps(mv)), xop_count(vecOps(cv)));
      collWrite(vecOps(mv), mw);
      collWrite(vecOps(cv), cw_);
      true
    },
    ("spdecompress", [|dv, mv, cv|]) => {
      let cb = bitsOf(vecRegs(cv, b32));
      let mb = bitsOf(vecRegs(mv, b32));
      let es = spcES(o.tys);
      let ib = spcIB(m);
      let ns = spfSrc(m);
      let nt = spfDst(m);
      var acc : int = 0;
      foreach (n from 0 to (spcNum(m) - 1))
        foreach (s from 0 to (ns - 1)) {
          let k = n * ns + s;
          acc = lor_int(acc, shl_int(land_int(asr_int(cb, k * es), mask(es)), (n * nt + land_int(asr_int(mb, k * ib), mask(ib))) * es))
        };
      collWrite(vecOps(dv), wordsOf(acc, xop_count(vecOps(dv))));
      true
    },
    _ => false
  }
}

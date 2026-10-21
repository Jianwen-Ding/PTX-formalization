/* =============================================================================
 * PTX-EXEC-MM (semantics/ptx-exec-mm.k): multimem.ld_reduce / st / red over the multimem windows
 * of the harness directive .multimem, and PTX-EXEC-FABRIC (ptx-exec-fabric.k): fabric.* operations
 * on the logical endpoints of the directive .fabric.
 * ============================================================================= */

/* <mmtab>: window start, size, device buffer addresses; <fabtab>: endpoint id, size, member buffers */
register mmtab : list((int, int, list(int))) = [||]
register fabtab : list((int, int, list(int))) = [||]

val mmFind : int -> option((int, int, list(int)))
function mmFind(a) = {
  var r : option((int, int, list(int))) = None();
  var l = mmtab;
  while is_cons(l) & is_none(r) do { match l { (s, n, bs) :: rest => { if a >= s & a < s + n then r = Some((s, n, bs)); l = rest }, [||] => () } };
  r
}
val addAll : (list(int), int) -> list(int)
function addAll(bs, o) = match bs { [||] => [||], b :: rest => (b + o) :: addAll(rest, o) }

val hAddrA : addr -> int
function hAddrA(a) = locOff(ea(a, "global"))

/* ---------------------------------------------------------------- reductions */
val mmOp : list(string) -> string
function mmOp(m) = pick(m, [|"add", "min", "max", "and", "or", "xor"|], "add")
val mmLaneTy : ty -> ty
function mmLaneTy(t) = match t.k {
  "f16x2" => mkty("f", 16), "bf16x2" => mkty("bf", 16), "e4m3x2" => mkty("e4m3", 8), "e4m3x4" => mkty("e4m3", 8),
  "e5m2x2" => mkty("e5m2", 8), "e5m2x4" => mkty("e5m2", 8), _ => t
}
val mmLaneN : ty -> int
function mmLaneN(t) = match t.k { "f16x2" => 2, "bf16x2" => 2, "e4m3x2" => 2, "e5m2x2" => 2, "e4m3x4" => 4, "e5m2x4" => 4, _ => 1 }
val mmAcc : (list(string), ty) -> ty
function mmAcc(m, t) = if has(m, "acc::f32") then mkty("f", 32) else if has(m, "acc::f16") then mkty("f", 16) else mmLaneTy(t)
val mmIsFp8 : ty -> bool
function mmIsFp8(t) = t.k == "e4m3" | t.k == "e5m2"

val mmTo : (ty, ty, int) -> int
function mmTo(l, a, v) =
  if ty_eq(l, a) then v
  else if mmIsFp8(l) & ty_eq(a, mkty("f", 16)) then miniToF16(l.k, v)
  else if mmIsFp8(l) then cvt([||], a, mkty("f", 16), miniToF16(l.k, v))
  else cvt([||], a, l, v)
val mmBack : (ty, ty, int) -> int
function mmBack(l, a, v) =
  if ty_eq(l, a) then v
  else if mmIsFp8(l) then miniFromR(l.k, decodeF(v, a.w), false, 0)
  else cvt([|"rn"|], l, a, v)
val mmComb : (string, ty, int, int) -> int
function mmComb(op, a, x, y) =
  if mmIsFp8(a) then miniFromR(a.k, decodeF(bin(op, [||], mkty("f", 16), miniToF16(a.k, x), miniToF16(a.k, y)), 16), false, 0)
  else bin(op, [||], a, x, y)

val mmRed : (string, ty, ty, list(int)) -> int
function mmRed(op, t, a, vs) = {
  if ty_eq(a, t) & not_bool(mmIsFp8(t)) then match vs {
    [||] => 0,
    v0 :: rest => { var acc : int = v0; var x = rest; while is_cons(x) do { match x { v :: r => { acc = atomNew(op, t, acc, v, 0); x = r }, [||] => () } }; acc }
  } else {
    let l = mmLaneTy(t);
    let lw = 8 * bytes(l);
    var out : int = 0;
    foreach (j from 0 to (mmLaneN(t) - 1)) {
      let lane : int = match vs {
        [||] => 0,
        v0 :: rest => {
          var acc : int = mmTo(l, a, land_int(asr_int(v0, lw * j), mask(lw)));
          var x = rest;
          while is_cons(x) do { match x { v :: r => { acc = mmComb(op, a, acc, mmTo(l, a, land_int(asr_int(v, lw * j), mask(lw)))); x = r }, [||] => () } };
          mmBack(l, a, acc)
        }
      };
      out = lor_int(out, shl_int(lane, lw * j))
    };
    out
  }
}

val mmVec : list(string) -> int
function mmVec(m) = if has(m, "v8") then 8 else if has(m, "v4") then 4 else if has(m, "v2") then 2 else 1

/* multimem.ld_reduce / st / red (PTX ISA 9.3 section 9.7.9.15, ~14157-14260): the integer forms take
 * .b32 .b64 .u32 .u64 .s32 .s64 without .vec / .acc_prec; the floating-point forms: ".vec and base floating-point type"
 * table (~14226-14234; multimem.red: .redtype without the fp8 types, .redop = { .add }), ".op and base type" table
 * (~14239-14247), ".acc::f32" only with .f16 .f16x2 .bf16 .bf16x2 and ".acc::f16" only with the fp8 types (~14252-14256,
 * multimem.ld_reduce only).  The example "multimem.red.relaxed.gpu.max.f64" (~14287) contradicts both the syntax and the
 * table, which the model follows.  "" when legal, else the error.  Reported by `.isacheck ;` (isaTables); without it every
 * combination runs with the natural meaning: the operation on each element of the type (integers as in atom, .and / .or /
 * .xor on the bits, floats in their format or in the .acc_prec format), .vec elements of any count. */
val mmErr : (list(string), ty) -> string
function mmErr(m, t) = {
  let n = tnm(t);
  let ld = has(m, "ld_reduce");
  let st = has(m, "st") & not_bool(ld);
  let red = has(m, "red") & not_bool(ld);
  let op = mmOp(m);
  let ints = [|"b32", "b64", "u32", "u64", "s32", "s64"|];
  let fp8 = [|"e5m2", "e5m2x2", "e5m2x4", "e4m3", "e4m3x2", "e4m3x4"|];
  let floats = [|"f16", "f16x2", "bf16", "bf16x2", "f32", "f64", "e5m2", "e5m2x2", "e5m2x4", "e4m3", "e4m3x2", "e4m3x4"|];
  let v = mmVec(m);
  let vecOk : bool = match v {
    1 => has([|"f16x2", "bf16x2", "f32", "f64", "e5m2x4", "e4m3x4"|], n),
    2 => has([|"f16", "f16x2", "bf16", "bf16x2", "f32", "e5m2x2", "e5m2x4", "e4m3x2", "e4m3x4"|], n),
    4 => has([|"f16", "f16x2", "bf16", "bf16x2", "f32", "e5m2", "e5m2x2", "e5m2x4", "e4m3", "e4m3x2", "e4m3x4"|], n),
    _ => has([|"f16", "bf16", "e5m2", "e4m3", "e5m2x2", "e4m3x2"|], n)
  };
  let opOk : bool = match op {
    "add" => has([|"u32", "u64", "s32", "f16", "f16x2", "bf16", "bf16x2", "f32", "f64"|], n) | has(fp8, n),
    "min" => has([|"u32", "s32", "u64", "s64", "f16", "f16x2", "bf16", "bf16x2"|], n) | has(fp8, n),
    "max" => has([|"u32", "s32", "u64", "s64", "f16", "f16x2", "bf16", "bf16x2"|], n) | has(fp8, n),
    _ => has([|"b32", "b64"|], n)
  };
  let acc = has(m, "acc::f32") | has(m, "acc::f16");
  let accOk = if has(m, "acc::f32") then has([|"f16", "f16x2", "bf16", "bf16x2"|], n) else if has(m, "acc::f16") then has(fp8, n) else true;
  let pre = concat_str("multimem.", if ld then "ld_reduce" else if st then "st" else "red");
  if has(ints, n) then {
    if v != 1 then concat_str(pre, ": .vec is not allowed with an integer type")
    else if acc then concat_str(pre, ": .acc_prec is not allowed with an integer type")
    else if not_bool(st) & not_bool(opOk) then concat_str(pre, concat_str(": .", concat_str(op, concat_str(" is not valid with .", n))))
    else ""
  } else if not_bool(has(floats, n)) then concat_str(pre, concat_str(": type .", concat_str(n, " is not allowed")))
  else if red & has(fp8, n) then concat_str(pre, concat_str(": type .", concat_str(n, " is not allowed")))
  else if not_bool(vecOk) then concat_str(pre, concat_str(": .vec and .", concat_str(n, " is not a valid combination")))
  else if red & op != "add" then concat_str(pre, ": floating-point types only allow .add")
  else if not_bool(st) & not_bool(opOk) then concat_str(pre, concat_str(": .", concat_str(op, concat_str(" is not valid with .", n))))
  else if acc & not_bool(ld) then concat_str(pre, ": .acc_prec is only allowed on multimem.ld_reduce")
  else if not_bool(accOk) then concat_str(pre, concat_str(": .acc_prec is not valid with .", n))
  else ""
}

/* the operation / type tables of PTX ISA 9.3 that `.isacheck ;` reports (ptx_isa.sail isaIns): sured, cp.reduce.async.bulk,
 * multimem.ld_reduce / st / red.  cp.reduce.async.bulk.tensor takes its element type from the tensor map, so its table is
 * checked when it runs (tm_store, ptx_tma.sail). */
val isaTables : (string, list(string), list(ty)) -> string
function isaTables(base, m, ts) = {
  let t = ty_nth(ts, 0);
  let (e, sec) : (string, string) =
    if base == "sured" then (suredErr(has(m, "p"), atomOp(m), tnm(t)), "9.7.12.3")
    else if base == "cp" & has(m, "reduce") & has(m, "bulk") & not_bool(has(m, "tensor")) then
      (bulkRedErr(has(m, "global"), atomOp(m), t, has(m, "noftz")), "9.7.9.26.4.2")
    else if base == "multimem" & (has(m, "ld_reduce") | has(m, "st") | has(m, "red")) & not_bool(has(m, "cp")) then (mmErr(m, t), "9.7.9.15")
    else ("", "");
  if e == "" then "" else concat_str(e, concat_str(" [PTX ISA 9.3 ", concat_str(sec, "]")))
}

val exec_mm : (opcode, list(xop)) -> bool
function exec_mm(o, ops) = {
  let m = o.mods;
  if o.base != "multimem" then false
  else match ops {
    [|XB([|a|], [||]), b|] if has(m, "st") | has(m, "red") => {
      let h = hAddrA(a);
      match mmFind(h) {
        Some((s, _, bs)) => {
          var x = addAll(bs, h - s);
          while is_cons(x) & not_bool(stuck) do {
            match x { d :: rest => {
              exec_dispatch(struct { base = if has(m, "st") then "st" else "red", mods = pend_append(m, [|"global"|]), tys = o.tys }, [|XB([|AI(d)|], [||]), b|]);
              x = rest
            }, [||] => () }
          }
        },
        None() => fault("multimem address expected", h)
      };
      true
    },
    [|d, XB([|a|], [||])|] if has(m, "ld_reduce") => {
      let h = hAddrA(a);
      match mmFind(h) {
        Some((s, _, bs)) => {
          let t = ty1(o);
          let ne = mmVec(m);
          var res : list(int) = [||];
          foreach (e from 0 to (ne - 1)) {
            var vs : list(int) = [||];
            var x = addAll(bs, h - s + e * bytes(t));
            while is_cons(x) & not_bool(stuck) do { match x { ad :: rest => { vs = pend_append(vs, [|memop_do(Loc("global", ad), RdOp(bytes(t)))|]); x = rest }, [||] => () } };
            res = pend_append(res, [|mmRed(mmOp(m), t, mmAcc(m, t), vs)|])
          };
          if not_bool(stuck) then collWrite(match d { XV(_) => vecOps(d), _ => [|d|] }, res)
        },
        None() => fault("multimem address expected", h)
      };
      true
    },
    _ => false
  }
}

/* ---------------------------------------------------------------- fabric */
/* device addresses of [leId, off .. off+n) */
val fbBases : (int, int, int, bool) -> list(int)
function fbBases(le, off, n, all) = {
  var r : list(int) = [||];
  var l = fabtab;
  var found : bool = false;
  while is_cons(l) & not_bool(found) do {
    match l { (i, sz, bs) :: rest => {
      if i == le then {
        found = true;
        if off >= 0 & off + n <= sz then r = if all then addAll(bs, off) else (match bs { b :: _ => [|b + off|], [||] => [||] })
      };
      l = rest
    }, [||] => () }
  };
  r
}
val fbHead : list(int) -> int
function fbHead(l) = match l { a :: _ => a, [||] => -1 }

val fbCnt : (int, int) -> unit
function fbCnt(cn, n) = if cn >= 0 & not_bool(stuck) then { let _ = memop_do(Loc("global", cn), AtOp("add", mkty("u", 64), n, 0)); () }

val fbOff : list(iop) -> int
function fbOff(l) = match l { IS(o) :: _ => rd_sop(o, mkty("u", 64)), _ => 0 }

val xtake1 : list(xop) -> list(xop)
function xtake1(l) = match l { x :: _ => [|x|], [||] => [||] }

val fb_put : (list(string), list(int), loc, loc, int, int, int) -> unit
function fb_put(m, ds, s, b, n, mk, cn) = {
  if not_bool(is_cons(ds)) then fault("fabric access outside the logical endpoint", 0)
  else {
    let v = memop_do(s, RdOp(n));
    var x = ds;
    while is_cons(x) & not_bool(stuck) do {
      match x { a :: rest => {
        if mk < 0 then { let _ = memop_do(Loc("global", a), WrOp(n, v)); () }
        else {
          let old = memop_do(Loc("global", a), RdOp(n));
          let em = expandMask(mk, tdiv_int(n, 16));
          if not_bool(stuck) then { let _ = memop_do(Loc("global", a), WrOp(n, lor_int(land_int(v, em), land_int(old, lnot_int(em))))); () }
        };
        x = rest
      }, [||] => () }
    };
    fbCnt(cn, n);
    if not_bool(stuck) then bulkTx(b, tdiv_int(n, 16))
  }
}

val fb_red : (list(string), list(int), loc, loc, int, ty, int) -> unit
function fb_red(m, ds, s, b, n, t, cn) = {
  if not_bool(is_cons(ds)) then fault("fabric access outside the logical endpoint", 0)
  else {
    let v = memop_do(s, RdOp(n));
    let eb = bytes(t);
    var x = ds;
    while is_cons(x) & not_bool(stuck) do {
      match x { a :: rest => {
        foreach (i from 0 to (tdiv_int(n, eb) - 1))
          if not_bool(stuck) then { let _ = memop_do(Loc("global", a + i * eb), AtOp(atomOp(m), t, land_int(asr_int(v, 8 * eb * i), mask(8 * eb)), 0)); () };
        x = rest
      }, [||] => () }
    };
    fbCnt(cn, n);
    if not_bool(stuck) then bulkTx(b, tdiv_int(n, 16))
  }
}

val fbTM : list(string) -> list(string)
function fbTM(m) = if has(m, "try_red") then pend_append(m, [|"bulk", "bulk_group", "reduce"|]) else pend_append(m, [|"bulk", "bulk_group"|])

val exec_fabric : (opcode, list(xop)) -> bool
function exec_fabric(o, ops) = {
  let m = o.mods;
  let u32 = mkty("u", 32);
  if o.base != "fabric" then false
  else if has(m, "submit") | has(m, "wait") then xop_count(ops) == 0
  else if has(m, "tensor") then match ops {
    /* try_get.async.tensor [dst], [tmap, leId, off, override, coords], [mbar] */
    [|XB([|d|], [||]), XB([|t, l|], [|IS(off), ov, cs|]), XB([|b|], [||])|] if has(m, "try_get") => {
      let base = fbHead(fbBases(hAddrA(l), rd_sop(off, mkty("u", 64)), 0, false));
      let bl = tmOv(tmObj(ea(t, "generic")), base, tmCoords(ov), [||], 0);
      tm_load(pend_append(m, [|"bulk"|]), d, bl, tmCoords(cs), b, [||]);
      true
    },
    XB([|t, l|], is) :: XB([|s|], [||]) :: sz :: XB([|b|], [||]) :: _ if has(m, "try_put") | has(m, "try_red") => {
      let (off, cn, ov, cs) : (int, int, iop, iop) = match is {
        [|IS(off), ov, cs|] => (rd_sop(off, mkty("u", 64)), -1, ov, cs),
        [|IS(off), IS(cn), ov, cs|] => (rd_sop(off, mkty("u", 64)), rd_sop(cn, mkty("u", 64)), ov, cs),
        _ => (0, -1, IV([||]), IV([||]))
      };
      let blob = tmObj(ea(t, "generic"));
      var x = fbBases(hAddrA(l), off, 0, has(m, "multimem"));
      while is_cons(x) & not_bool(stuck) do {
        match x { a :: rest => { tm_store(fbTM(m), tmOv(blob, a, tmCoords(ov), [||], 0), tmCoords(cs), s); x = rest }, [||] => () }
      };
      if cn >= 0 then fbCnt(fbHead(fbBases(hAddrA(l), cn, 8, false)), rdu32(sz));
      if not_bool(stuck) then bulkTx(ea(b, "shared"), tdiv_int(rdu32(sz), 16));
      true
    },
    _ => false
  } else match ops {
    /* try_get [dst], [leId, off], size, [mbar] */
    XB([|d|], [||]) :: XB([|l|], [|IS(off)|]) :: sz :: XB([|b|], [||]) :: _ if has(m, "try_get") => {
      let n = rdu32(sz);
      match fbBases(hAddrA(l), rd_sop(off, mkty("u", 64)), n, false) {
        a :: _ => {
          let v = memop_do(Loc("global", a), RdOp(n));
          if not_bool(stuck) then { let _ = memop_do(ea(d, "shared"), WrOp(n, v)); () };
          if not_bool(stuck) then bulkTx(ea(b, "shared"), n)
        },
        [||] => fault("fabric access outside the logical endpoint", 0)
      };
      true
    },
    /* try_pullred [dst], [leId, off], size, [mbar] (lane 0) */
    XB([|d|], [||]) :: XB([|l|], [|IS(off)|]) :: sz :: XB([|b|], [||]) :: _ if has(m, "try_pullred") => {
      if rreg("%laneid") == 0 then {
        let n = rdu32(sz);
        let t = ty1(o);
        let ds = fbBases(hAddrA(l), rd_sop(off, mkty("u", 64)), n, true);
        if not_bool(is_cons(ds)) then fault("fabric access outside the logical endpoint", 0)
        else {
          var vs : list(int) = [||];
          var x = ds;
          while is_cons(x) & not_bool(stuck) do { match x { a :: rest => { vs = pend_append(vs, [|memop_do(Loc("global", a), RdOp(n))|]); x = rest }, [||] => () } };
          let w = bytes(t);
          var r : int = 0;
          foreach (i from 0 to (tdiv_int(n, w) - 1)) {
            var es : list(int) = [||];
            var y = vs;
            while is_cons(y) do { match y { v :: rest => { es = pend_append(es, [|land_int(asr_int(v, 8 * w * i), mask(8 * w))|]); y = rest }, [||] => () } };
            r = lor_int(r, shl_int(mmRed(mmOp(m), t, mmAcc(m, t), es), 8 * w * i))
          };
          if not_bool(stuck) then { let _ = memop_do(ea(d, "shared"), WrOp(n, r)); () };
          if not_bool(stuck) then bulkTx(ea(b, "shared"), n)
        }
      };
      true
    },
    /* try_put [leId, off], [src], size, [mbar] {, byteMask}  /  [leId, off, counter], ... */
    XB([|l|], [|IS(off)|]) :: XB([|s|], [||]) :: sz :: XB([|b|], [||]) :: rest if has(m, "try_put") => {
      let n = rdu32(sz);
      fb_put(m, fbBases(hAddrA(l), rd_sop(off, mkty("u", 64)), n, has(m, "multimem")), ea(s, "shared"), ea(b, "shared"), n,
             if has(m, "cp_mask") & xop_count(rest) > 0 then lastArg(xtake1(rest)) else -1, -1);
      true
    },
    XB([|l, off|], [|IS(cn)|]) :: XB([|s|], [||]) :: sz :: XB([|b|], [||]) :: _ if has(m, "try_put") => {
      let n = rdu32(sz);
      fb_put(m, fbBases(hAddrA(l), hAddrA(off), n, has(m, "multimem")), ea(s, "shared"), ea(b, "shared"), n, -1,
             fbHead(fbBases(hAddrA(l), rd_sop(cn, mkty("u", 64)), 8, false)));
      true
    },
    XB([|l|], [|IS(off)|]) :: XB([|s|], [||]) :: sz :: XB([|b|], [||]) :: _ if has(m, "try_red") => {
      let n = rdu32(sz);
      fb_red(m, fbBases(hAddrA(l), rd_sop(off, mkty("u", 64)), n, has(m, "multimem")), ea(s, "shared"), ea(b, "shared"), n, ty1(o), -1);
      true
    },
    XB([|l, off|], [|IS(cn)|]) :: XB([|s|], [||]) :: sz :: XB([|b|], [||]) :: _ if has(m, "try_red") => {
      let n = rdu32(sz);
      fb_red(m, fbBases(hAddrA(l), hAddrA(off), n, has(m, "multimem")), ea(s, "shared"), ea(b, "shared"), n, ty1(o),
             fbHead(fbBases(hAddrA(l), rd_sop(cn, mkty("u", 64)), 8, false)));
      true
    },
    /* try_atom [leId, off], [smem_dst], [smem_src], [mbar] */
    XB([|l|], [|IS(offo)|]) :: XB([|dd|], [||]) :: XB([|ss|], [||]) :: XB([|b|], [||]) :: _ if has(m, "try_atom") => {
      let t = ty1(o);
      let off = rd_sop(offo, mkty("u", 64));
      let a = fbHead(fbBases(hAddrA(l), off, bytes(t), false));
      let so = tmod_int(off, 16);
      if a < 0 then fault("fabric access outside the logical endpoint", 0)
      else {
        let v = memop_do(loc_add(ea(ss, "shared"), so), RdOp(bytes(t)));
        let op = atomOp(m);
        let x = if op == "cas" & not_bool(stuck) then memop_do(loc_add(ea(ss, "shared"), 16 + so), RdOp(bytes(t))) else 0;
        let old = if not_bool(stuck) then memop_do(Loc("global", a), if op == "cas" then AtOp(op, t, v, x) else AtOp(op, t, v, 0)) else 0;
        if not_bool(stuck) then { let _ = memop_do(loc_add(ea(dd, "shared"), so), WrOp(bytes(t), old)); () };
        if not_bool(stuck) then bulkTx(ea(b, "shared"), 1)
      };
      true
    },
    _ => false
  }
}

/* ---------------------------------------------------------------- the .multimem / .fabric directives */
val symA : string -> int
function symA(n) = match buf_addr(n) { Some(a) => a, None() => 0 }
val tk_names : int -> list(string)
function tk_names(k) = { var l : list(string) = [||]; foreach (i from 1 to k) l = pend_append(l, [|tok_str()|]); l }
val namesA : list(string) -> list(int)
function namesA(l) = match l { [||] => [||], n :: rest => symA(n) :: namesA(rest) }

val harness_mm : string -> bool
function harness_mm(d) = {
  if d == "MULTIMEM" then {
    let n = tok_str();
    let s = tok_int();
    let bs = tk_names(tok_int());
    do_alloc(n, mkty("b", 8), s, [||]);
    mmtab = pend_append(mmtab, [|(symA(n), s, namesA(bs))|]);
    true
  } else if d == "FABRIC" then {
    let i = tok_int();
    let s = tok_int();
    let bs = tk_names(tok_int());
    var l : list((int, int, list(int))) = [||];
    var x = fabtab;
    while is_cons(x) do { match x { (j, sz, b) :: rest => { if j != i then l = pend_append(l, [|(j, sz, b)|]); x = rest }, [||] => () } };
    fabtab = pend_append(l, [|(i, s, namesA(bs))|]);
    true
  } else false
}

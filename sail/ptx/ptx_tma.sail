/* =============================================================================
 * PTX-EXEC-TMA (semantics/ptx-exec-tma.k): tensor maps and the Tensor Memory Accelerator:
 * cp.async.bulk.tensor loads (tile, gather4, packed sub-byte types, im2col modes), stores and
 * reductions (tile, scatter4, im2col_no_offs), tensormap.replace / cp_fenceproxy, and the
 * harness directives .tensormap (ptx-harness.k) that build the 128-byte objects in our encoding.
 *
 * As in K, the tensor-map object and the global source data are read directly from global memory
 * and the shared source of a store from the CTA's shared memory; the writes go through #mem.
 * Divisions and remainders that may see negative values use K's truncating /Int and %Int
 * (tdiv_int / tmod_int) and its flooring divInt (fdiv_int).
 * ============================================================================= */

/* ---------------------------------------------------------------- tensor-map fields */
val tmW : (int, int) -> int
function tmW(b, i) = land_int(asr_int(b, 64 * i), 18446744073709551615)
val tmRank : int -> int
function tmRank(b) = land_int(tmW(b, 2), 255)
val tmAddrOf : int -> int
function tmAddrOf(b) = tmW(b, 3)
val tmDim : (int, int) -> int
function tmDim(b, i) = tmW(b, 4 + i)
val tmStride : (int, int) -> int
function tmStride(b, i) = tmW(b, 8 + i)
val tmBox : (int, int) -> int
function tmBox(b, i) = if i < 4 then land_int(asr_int(tmW(b, 13), 16 * i), 65535) else land_int(tmW(b, 14), 65535)
val tmEst : (int, int) -> int
function tmEst(b, i) = land_int(asr_int(tmW(b, 15), 8 * i), 255)
val tmSwz : int -> int
function tmSwz(b) = land_int(asr_int(tmW(b, 14), 16), 255)
val tmAtom : int -> int
function tmAtom(b) = land_int(asr_int(tmW(b, 14), 48), 255)
val tmFillMode : int -> int
function tmFillMode(b) = land_int(asr_int(tmW(b, 14), 24), 255)
val tmEtc : int -> int
function tmEtc(b) = land_int(tmW(b, 1), 255)

val tmBytesOf : int -> int
function tmBytesOf(c) = match c { 0 => 1, 1 => 2, 2 => 4, 3 => 4, 4 => 8, 5 => 8, 6 => 2, 7 => 4, 8 => 4, 9 => 8, 10 => 2, 11 => 4, 12 => 4, _ => 1 }
val tmEb : int -> int
function tmEb(b) = tmBytesOf(tmEtc(b))

val tmTyOf : int -> ty
function tmTyOf(c) = match c {
  0 => mkty("u", 8), 1 => mkty("u", 16), 2 => mkty("u", 32), 3 => mkty("s", 32), 4 => mkty("u", 64), 5 => mkty("s", 64),
  6 => mkty("f", 16), 7 => mkty("f", 32), 8 => mkty("f", 32), 9 => mkty("f", 64), 10 => mkty("bf", 16), 11 => mkty("f", 32),
  12 => mkty("f", 32), _ => mkty("u", 8)
}
val tmTy : int -> ty
function tmTy(b) = tmTyOf(tmEtc(b))

/* the swizzle of the shared-memory addresses (swizzle of ptx_wg.sail) for the tensor map's swizzle mode (Table 33:
 * 0 none, 1 32B, 2 64B, 3 128B, 4 96B).  The 96B mode has the pattern of the 32B mode ("0 1 2 3 4 5 6 7 / 1 0 3 2 5 4 7 6 /
 * ... Pattern repeats" from a 256-byte boundary, PTX ISA 9.3 section 5.5.7, and Figure 31). */
val tmEffMode : int -> int
function tmEffMode(b) =
  if tmSwz(b) == 1 | tmSwz(b) == 4 then 3 else if tmSwz(b) == 2 then 2
  else if tmSwz(b) == 3 then (match tmAtom(b) { 0 => 1, 1 => 4, 2 => 5, 3 => 6, _ => 0 })
  else 0

val tmFill : int -> int
function tmFill(b) =
  if tmFillMode(b) == 0 then 0
  else if tmEb(b) == 4 then 2143289344
  else if tmEtc(b) == 6 then oobNaN(16)
  else if tmEtc(b) == 10 then oobNaN(116)
  else if tmEtc(b) == 9 then 9221120237041090560
  else 0

/* ---------------------------------------------------------------- element addresses */
/* tiled mode: the traversal stride "specifies the exact number of elements to be skipped" while the bounding box iterates
 * a dimension (PTX ISA 9.3 section 5.5.3.2, ~3036), and it reduces the number of elements accessed, "unlike the tiled
 * mode" in im2col mode (~3258-3259): dimension i holds ceil(boxDim[i] / elementStride[i]) elements, at coordinates
 * c + k * elementStride[i] */
val tmN : (int, int) -> int
function tmN(b, i) = { let e = max_int(1, tmEst(b, i)); tdiv_int(tmBox(b, i) + e - 1, e) }
val tmVol : (int, int) -> int
function tmVol(b, i0) = { var a : int = 1; var i : int = i0; while i < tmRank(b) do { a = a * tmN(b, i); i = i + 1 }; a }
val tmCount : (int, bool) -> int
function tmCount(b, g) = if g then 4 * tmBox(b, 0) else tmVol(b, 0)

val gcoord : (int, list(int), int, int, int) -> int
function gcoord(b, cs, e, i, dv) = nth(cs, i) + tmod_int(tdiv_int(e, dv), tmN(b, i)) * tmEst(b, i)

val tmAddrT : (int, list(int), int) -> int
function tmAddrT(b, cs, e) = {
  var i : int = 0;
  var dv : int = 1;
  var acc : int = 0;
  var bad : bool = false;
  while i < tmRank(b) & not_bool(bad) do {
    let g = gcoord(b, cs, e, i, dv);
    if g < 0 | g >= tmDim(b, i) then bad = true
    else { acc = acc + (if i == 0 then g * tmEb(b) else g * tmStride(b, i)); dv = dv * tmN(b, i); i = i + 1 }
  };
  if bad then -1 else tmAddrOf(b) + acc
}

val tmAddrG : (int, list(int), int) -> int
function tmAddrG(b, cs, e) = {
  let c = nth(cs, 0) + tmod_int(e, tmBox(b, 0));
  let r = nth(cs, 1 + tdiv_int(e, tmBox(b, 0)));
  if c < 0 | c >= tmDim(b, 0) | r < 0 | r >= tmDim(b, 1) then -1
  else tmAddrOf(b) + c * tmEb(b) + r * tmStride(b, 1)
}

val tmAddr : (int, list(int), int, bool) -> int
function tmAddr(b, cs, e, g) = if g then tmAddrG(b, cs, e) else tmAddrT(b, cs, e)

/* ---------------------------------------------------------------- multicast / cta_group::2 locations */
val rankBases : (int, int) -> list(int)
function rankBases(off, mk) = {
  var out : list(int) = [||];
  foreach (r from 0 to 31) if bit_int(mk, r) & asr_int(mk, r) > 0 then out = pend_append(out, [|lor_int(shl_int(r + 1, 24), off)|]);
  out
}
val tmBases : (loc, int, bool) -> list(int)
function tmBases(l, mk, mc) = match l { Loc(_, a) => if mc then rankBases(land_int(a, 16777215), mk) else [|a|], _ => [||] }

val tmBarLocs : (loc, int, bool, bool, int) -> list(loc)
function tmBarLocs(l, mk, mc, cg2, own) = {
  if not_bool(cg2) then bulkLocs(l, mk, mc)
  else if not_bool(mc) then [|l|]
  else match l {
    Loc(_, a) => {
      let off = land_int(a, 16777215);
      let par = if asr_int(a, 24) > 0 then tmod_int(asr_int(a, 24) - 1, 2) else tmod_int(own, 2);
      var out : list(loc) = [||];
      foreach (r from 0 to 31)
        if bit_int(mk, r) & asr_int(mk, r) > 0 then out = pend_append(out, [|Loc("cluster", lor_int(shl_int(r - tmod_int(r, 2) + par + 1, 24), off))|]);
      out
    },
    _ => [|l|]
  }
}

/* ---------------------------------------------------------------- objects and overrides */
val setWord : (int, int, int) -> int
function setWord(b, i, v) = lor_int(land_int(b, lnot_int(shl_int(18446744073709551615, 64 * i))), shl_int(land_int(v, 18446744073709551615), 64 * i))
val setBox : (int, int, int) -> int
function setBox(b, o, v) =
  if o < 4 then setWord(b, 13, lor_int(land_int(tmW(b, 13), lnot_int(shl_int(65535, 16 * o))), shl_int(land_int(v, 65535), 16 * o)))
  else setWord(b, 14, lor_int(land_int(tmW(b, 14), lnot_int(65535)), land_int(v, 65535)))

val tmObj : loc -> int
function tmObj(l) = mem_rdn(SP_GMEM, locOff(l), 128)

val tmCoords : iop -> list(int)
function tmCoords(o) = match o { IV(l) => { var out : list(int) = [||]; var x = l;
    while is_cons(x) do { match x { s :: rest => { out = pend_append(out, [|sext(rd_sop(s, mkty("s", 32)), 32)|]); x = rest }, [||] => () } };
    out }, _ => [||] }

/* tmOv(B, G, dims, lower strides, upper strides) */
val tmOv : (int, int, list(int), list(int), int) -> int
function tmOv(b0, g, ds, ls, up) = {
  var b = setWord(b0, 3, g);
  var i : int = 0;
  var x = ds;
  while is_cons(x) do { match x { v :: rest => { b = setWord(b, 4 + i, v); i = i + 1; x = rest }, [||] => () } };
  i = 0;
  x = ls;
  while is_cons(x) do { match x { v :: rest => { b = setWord(b, 9 + i, shl_int(v + shl_int(land_int(asr_int(up, 4 * i), 15), 32), 4)); i = i + 1; x = rest }, [||] => () } };
  b
}

/* the tensor-map object of a [tmap, ...] operand and its coordinates */
val tmOperand : (list(addr), list(iop)) -> option((int, list(int)))
function tmOperand(as_, is) = match (as_, is) {
  ([|t|], [|cs|]) => Some((tmObj(ea(t, "generic")), tmCoords(cs))),
  ([|t, g|], [|cs|]) => Some((setWord(tmObj(ea(t, "generic")), 3, locOff(ea(g, "generic"))), tmCoords(cs))),
  ([|t, g|], [|sz, cs|]) => Some((tmOv(tmObj(ea(t, "generic")), locOff(ea(g, "generic")), tmCoords(sz), [||], 0), tmCoords(cs))),
  ([|t, g|], [|sz, lo, IS(up), cs|]) =>
    Some((tmOv(tmObj(ea(t, "generic")), locOff(ea(g, "generic")), tmCoords(sz), tmCoords(lo), rd_sop(up, mkty("b", 16))), tmCoords(cs))),
  _ => None()
}

/* ---------------------------------------------------------------- packed sub-byte types */
val tmSub : int -> bool
function tmSub(b) = tmEtc(b) >= 13
val tmGB : int -> int
function tmGB(b) = if tmEtc(b) == 15 then 12 else 8
val tmSB : int -> int
function tmSB(b) = if tmEtc(b) == 13 then 8 else 16
val tmNb : int -> int
function tmNb(b) = if tmEtc(b) == 15 then 6 else 4
val tmGroups : int -> int
function tmGroups(b) = tdiv_int(tmBox(b, 0), 16) * tmVol(b, 1)

val tmGrpAddr : (int, list(int), int) -> int
function tmGrpAddr(b, cs, q0) = {
  let gpr = tdiv_int(tmBox(b, 0), 16);
  let c0 = nth(cs, 0) + tmod_int(q0, gpr) * 16;
  if c0 < 0 | c0 >= tmDim(b, 0) then -1
  else {
    var q : int = tdiv_int(q0, gpr);
    var i : int = 1;
    var acc : int = tmAddrOf(b) + tdiv_int(c0 * tmNb(b), 8);
    var bad : bool = false;
    while i < tmRank(b) & not_bool(bad) do {
      let v = nth(cs, i) + tmod_int(q, tmN(b, i)) * tmEst(b, i);
      if v < 0 | v >= tmDim(b, i) then bad = true
      else { acc = acc + v * tmStride(b, i); q = tdiv_int(q, tmN(b, i)); i = i + 1 }
    };
    if bad then -1 else acc
  }
}
val tmGrpVal : (int, list(int), int) -> int
function tmGrpVal(b, cs, q) =
  if q >= tmGroups(b) | tmGrpAddr(b, cs, q) < 0 then 0 else mem_rdn(SP_GMEM, tmGrpAddr(b, cs, q), tmGB(b))
val tmChunkS : (int, list(int), int) -> int
function tmChunkS(b, cs, k) =
  if tmSB(b) == 16 then tmGrpVal(b, cs, k) else lor_int(tmGrpVal(b, cs, 2 * k), shl_int(tmGrpVal(b, cs, 2 * k + 1), 64))

/* ---------------------------------------------------------------- im2col */
val isI2c : list(string) -> bool
function isI2c(m) = has(m, "im2col") | has(m, "im2col::w") | has(m, "im2col::w::128") | has(m, "im2col_no_offs::w")
val imMode : list(string) -> int
function imMode(m) = if has(m, "im2col::w") then 1 else if has(m, "im2col::w::128") then 2 else if has(m, "im2col_no_offs::w") then 3 else 0
val tmLo : (int, int) -> int
function tmLo(b, k) = land_int(asr_int(tmW(b, 1), 8 + 16 * k), 65535) - 32768
val tmUp : (int, int) -> int
function tmUp(b, k) = land_int(asr_int(tmW(b, 2), 8 + 16 * k), 65535) - 32768
val tmNk : (int, int) -> int
function tmNk(b, k) = max_int(1, tdiv_int(tmDim(b, k) - 1 + tmUp(b, k - 1) - tmLo(b, k - 1), tmEst(b, k)) + 1)

val tmStart : (int, list(int)) -> int
function tmStart(b, cs) = {
  var k : int = 1;
  var mu : int = 1;
  var acc : int = 0;
  while k <= tmRank(b) - 2 do {
    acc = acc + fdiv_int(nth(cs, k) - tmLo(b, k - 1), tmEst(b, k)) * mu;
    mu = mu * tmNk(b, k);
    k = k + 1
  };
  acc + nth(cs, tmRank(b) - 1) * mu
}

val tmPix : (int, list(int), int) -> int
function tmPix(b, inf, md) = match md { 1 => tmBox(b, 1) + nth(inf, 0), 2 => 128 + 4 * nth(inf, 0), _ => tmBox(b, 1) }

val tmQ : (int, int, int) -> int
function tmQ(md, halo, p) = if md == 2 then 32 * tdiv_int(p, 32 + halo) + tmod_int(p, 32 + halo) else p
val tmWOff : (int, list(int)) -> int
function tmWOff(md, inf) = if md == 3 then 0 else nth(inf, 1)

val tmAddrW : (int, list(int), int, int) -> int
function tmAddrW(b, cs, sh, acc0) = {
  var k : int = 1;
  var acc : int = acc0;
  var bad : bool = false;
  while k <= tmRank(b) - 2 & not_bool(bad) do {
    let v = nth(cs, k) + (if k == 1 then sh else 0);
    if v < 0 | v >= tmDim(b, k) then bad = true else { acc = acc + v * tmStride(b, k); k = k + 1 }
  };
  let n = nth(cs, tmRank(b) - 1);
  if bad | n < 0 | n >= tmDim(b, tmRank(b) - 1) then -1 else acc + n * tmStride(b, tmRank(b) - 1)
}

val tmAddrI : (int, list(int), list(int), int, int) -> int
function tmAddrI(b, cs, inf, md, e) = {
  let c0 = nth(cs, 0) + tmod_int(e, tmBox(b, 0)) * tmEst(b, 0);
  if c0 < 0 | c0 >= tmDim(b, 0) then -1
  else {
    let p = tdiv_int(e, tmBox(b, 0));
    let acc0 = tmAddrOf(b) + c0 * tmEb(b);
    if md != 0 then tmAddrW(b, cs, tmQ(md, nth(inf, 0), p) * tmEst(b, 1) + tmWOff(md, inf), acc0)
    else {
      var idx : int = tmStart(b, cs) + p;
      var k : int = 1;
      var acc : int = acc0;
      var bad : bool = false;
      while k <= tmRank(b) - 2 & not_bool(bad) do {
        let pos = tmLo(b, k - 1) + tmod_int(idx, tmNk(b, k)) * tmEst(b, k) + nth(inf, k - 1);
        if pos < 0 | pos >= tmDim(b, k) then bad = true
        else { acc = acc + pos * tmStride(b, k); idx = tdiv_int(idx, tmNk(b, k)); k = k + 1 }
      };
      if bad | idx < 0 | idx >= tmDim(b, tmRank(b) - 1) then -1 else acc + idx * tmStride(b, tmRank(b) - 1)
    }
  }
}

/* ---------------------------------------------------------------- loads */
/* tmWrap: the swizzled destination of a 16-byte chunk (mode 5 also exchanges the 8-byte halves on odd lines) */
val tmWrite : (string, int, int, int) -> unit
function tmWrite(sp, md, a, v) = {
  let (a2, v2) : (int, int) = if md == 5 then (swizzle(4, a), if land_int(asr_int(a, 7), 1) == 1 then lor_int(shl_int(land_int(v, 18446744073709551615), 64), asr_int(v, 64)) else v)
                 else (swizzle(md, a), v);
  if not_bool(stuck) then { let _ = memop_do(Loc(sp, a2), WrOp(16, v2)); () }
}

val tmElemV : (int, list(int), int, bool) -> int
function tmElemV(b, cs, e, g) = { let a = tmAddr(b, cs, e, g); if a < 0 then tmFill(b) else mem_rdn(SP_GMEM, a, tmEb(b)) }
val tmElemI : (int, list(int), list(int), int, int) -> int
function tmElemI(b, cs, inf, md, e) = { let a = tmAddrI(b, cs, inf, md, e); if a < 0 then tmFill(b) else mem_rdn(SP_GMEM, a, tmEb(b)) }

val xnovec : list(xop) -> list(xop)
function xnovec(l) = match l { [||] => [||], x :: rest => if isVec(x) then xnovec(rest) else x :: xnovec(rest) }
val tmInfo : list(xop) -> list(int)
function tmInfo(l) = match l { [||] => [||], XV(v) :: _ => tmCoords(IV(v)), _ :: rest => tmInfo(rest) }

/* the restrictions on tensor copies (9.7.9.26.5.1) that the tensor map and the coordinates decide, under .ub strict
 * (codes of Table 33: element type 13 .b4x16, 14 .b4x16_p64, 15 .b6x16_p32 / .b6p2x16; swizzle 4 96B; atomicity 0 16B,
 * 2 32B + 8B flip; fill 1 OOB-NaN) */
val tmUb : (list(string), int, list(int), bool) -> bool
function tmUb(m, b, cs, store) =
  if not_bool(ub) then true
  else {
    let c = tmEtc(b);
    let padded = c == 14 | c == 15;
    let c0 : int = match cs { x :: _ => x, [||] => 0 };
    var e : string =
      if tmSub(b) & tmFillMode(b) == 1 then "OOB-NaN fill mode does not support the sub-byte types"
      else if padded & tmod_int(c0, 128) != 0 then "the first coordinate must be a multiple of 128 for the padded sub-byte types"
      else if padded & tmod_int(tmAddrOf(b), 32) != 0 then "the global address must be 32B aligned for the padded sub-byte types"
      else if tmSwz(b) == 4 & tmAtom(b) != 0 then "the 96B swizzle mode needs .swizzle_atomicity 16B"
      else if tmSwz(b) == 4 & padded then "the 96B swizzle mode does not support the padded sub-byte types"
      else if tmSwz(b) == 4 & not_bool(tmSub(b)) & tmBox(b, 0) * tmEb(b) > 96 then "the 96B swizzle mode needs Box-Size[0] <= 96B"
      else if tmSwz(b) == 4 & has(m, "im2col::w::128") then "the 96B swizzle mode does not support .im2col::w::128"
      else if tmSwz(b) == 3 & tmAtom(b) == 2 & store then "128B swizzle with 32B atomicity and 8B flip only for copies into shared memory"
      else if tmSwz(b) == 3 & tmAtom(b) == 2 & (has(m, "im2col::w") | has(m, "im2col::w::128")) then "128B swizzle with 32B atomicity and 8B flip does not support .im2col::w modes"
      else "";
    if padded then foreach (i from 1 to (tmRank(b) - 1)) if e == "" & tmod_int(tmStride(b, i), 32) != 0 then e = "tensor strides must be 32B aligned for the padded sub-byte types";
    /* .global.shared::cta: "Starting co-ordinates for Bounding Box (tensorCoords) must be non-negative".  The second
       rule, "The bounding box along the D, W and H dimensions must stay within the tensor boundaries", names the im2col
       dimensions; tile-mode stores past the boundary keep the clipping of 5.5 */
    if store & e == "" then {
      var x = cs;
      while is_cons(x) do { match x { v :: rest => { if e == "" & v < 0 then e = "the starting coordinates of a .global.shared::cta copy must be non-negative"; x = rest }, [||] => () } }
    };
    if e != "" then { fault(concat_str("cp.async.bulk.tensor: ", e), 0); false } else true
  }

val tm_load : (list(string), addr, int, list(int), addr, list(xop)) -> unit
function tm_load(m, d, b, cs, mb, rest) = {
  if not_bool(tmUb(m, b, cs, false)) then return ();
  let cl = has(m, "shared::cluster");
  let sp = if cl then "cluster" else "shared";
  let dsp = if cl then "shared_cluster" else "shared";
  let md = tmEffMode(b);
  let eb = tmEb(b);
  let i2c = isI2c(m);
  let sub = tmSub(b);
  let mk = bulkMask(XB([|mb|], [||]) :: (if i2c | sub then xnovec(rest) else rest));
  let bases = tmBases(ea(d, dsp), mk, isMc(m));
  let g = has(m, "gather4") | has(m, "tile::gather4");
  let inf = tmInfo(rest);
  let imd = imMode(m);
  var bs = bases;
  while is_cons(bs) & not_bool(stuck) do {
    match bs { base :: brest => {
      if i2c then {
        var k : int = 0;
        while k * 16 < tmPix(b, inf, imd) * tmBox(b, 0) * eb & not_bool(stuck) do {
          var v : int = 0;
          foreach (j from 0 to (tdiv_int(16, eb) - 1)) v = lor_int(v, shl_int(tmElemI(b, cs, inf, imd, k * tdiv_int(16, eb) + j), 8 * eb * j));
          tmWrite(sp, md, base + 16 * k, v);
          k = k + 1
        }
      } else if sub then {
        var k : int = 0;
        while k * 16 < tmGroups(b) * tmSB(b) & not_bool(stuck) do { tmWrite(sp, md, base + 16 * k, tmChunkS(b, cs, k)); k = k + 1 }
      } else {
        var k : int = 0;
        while k * 16 < tmCount(b, g) * eb & not_bool(stuck) do {
          var v : int = 0;
          foreach (j from 0 to (tdiv_int(16, eb) - 1)) v = lor_int(v, shl_int(tmElemV(b, cs, k * tdiv_int(16, eb) + j, g), 8 * eb * j));
          tmWrite(sp, md, base + 16 * k, v);
          k = k + 1
        }
      };
      bs = brest
    }, [||] => () }
  };
  let n = if i2c then tmPix(b, inf, imd) * tmBox(b, 0) * eb else if sub then tmGroups(b) * tmSB(b)
          else tmCount(b, has(m, "tile::gather4")) * eb;
  var ms = tmBarLocs(ea(mb, dsp), mk, isMc(m), has(m, "cta_group::2"), rreg("%cluster_ctarank"));
  while is_cons(ms) & not_bool(stuck) do { match ms { l :: mr => { bulkTx(l, n); ms = mr }, [||] => () } }
}

/* ---------------------------------------------------------------- stores / reductions */
val tmSrcV : (int, int, int) -> int
function tmSrcV(b, base, e) = mem_rdn(cta_smem(), swizzle(tmEffMode(b), base + e * tmEb(b)), tmEb(b))

val tmBad : (int, list(int), int) -> int
function tmBad(b, cs, cnt) = {
  var q : int = 0;
  while q < cnt & nth(cs, 1) + q * tmEst(b, 1) >= 0 & nth(cs, 1) + q * tmEst(b, 1) < tmDim(b, 1) do q = q + 1;
  q
}

/* cp.reduce.async.bulk.tensor: "the valid combinations of .redOp and element type" (PTX ISA 9.3 ~17375-17380):
 * .add .u32 .s32 .u64 .f32 .f16 .bf16; .min/.max .u32 .s32 .u64 .s64 .f16 .bf16; .inc/.dec .u32; .and/.or/.xor .b32 .b64.
 * The tensor map gives the element type (tensormap.replace.elemtype values); f32 includes the .ftz variant (8), and the
 * bit-size types .b32 / .b64 are taken as the 32- and 64-bit integer element types. */
val tmRedOk : (string, int) -> bool
function tmRedOk(op, c) = match op {
  "add" => has([|"2", "3", "4", "7", "8", "6", "10"|], dec_str(c)),
  "min" => has([|"2", "3", "4", "5", "6", "10"|], dec_str(c)),
  "max" => has([|"2", "3", "4", "5", "6", "10"|], dec_str(c)),
  "inc" => c == 2,
  "dec" => c == 2,
  "and" => c >= 2 & c <= 5,
  "or" => c >= 2 & c <= 5,
  "xor" => c >= 2 & c <= 5,
  _ => false
}

/* the element type is known only when the instruction runs, so the table is checked then, under `.isacheck ;`; without it
 * the other combinations run with the natural meaning (as for cp.reduce.async.bulk, bulkRedErr in ptx_async.sail):
 * integer operations on the bits of integer types, .and / .or / .xor on the bits of any type, .add / .min / .max on
 * floats; .inc / .dec on a floating-point element type stop with an error */
val tmRedFloat : int -> bool
function tmRedFloat(c) = c >= 6 & c <= 12

val tm_store : (list(string), int, list(int), addr) -> unit
function tm_store(m, b, cs, s) = if has(m, "reduce") & isachk & not_bool(tmRedOk(atomOp(m), tmEtc(b))) then
  fault(concat_str("cp.reduce.async.bulk.tensor: .", concat_str(atomOp(m),
    " is not a valid combination with the tensor-map element type (PTX ISA 9.3 9.7.9.26.5.3), element type")), tmEtc(b))
else if has(m, "reduce") & (atomOp(m) == "inc" | atomOp(m) == "dec") & tmRedFloat(tmEtc(b)) then
  fault(concat_str("cp.reduce.async.bulk.tensor: .", concat_str(atomOp(m), " on a floating-point tensor-map element type")), tmEtc(b))
else if not_bool(tmUb(m, b, cs, true)) then ()
else {
  let base = locOff(ea(s, "shared"));
  let op = if has(m, "reduce") then atomOp(m) else "";
  let t = tmTy(b);
  let eb = tmEb(b);
  /* the (address, value) list is built from the shared image first, then written */
  var ws : list((int, int)) = [||];
  if has(m, "im2col_no_offs::w") | has(m, "im2col_no_offs") then {
    let w = has(m, "im2col_no_offs::w");
    let bad = if w then tmBad(b, cs, tmBox(b, 1)) else tmBox(b, 1);
    let md = if w then 3 else 0;
    foreach (e from 0 to (tmBox(b, 1) * tmBox(b, 0) - 1)) {
      let a = tmAddrI(b, cs, [|0, 0, 0|], md, e);
      if not_bool(tdiv_int(e, tmBox(b, 0)) >= bad | a < 0) then ws = pend_append(ws, [|(a, tmSrcV(b, base, e))|])
    }
  } else {
    let g = has(m, "scatter4") | has(m, "tile::scatter4");
    foreach (e from 0 to (tmCount(b, g) - 1)) {
      let a = tmAddr(b, cs, e, g);
      if a >= 0 then ws = pend_append(ws, [|(a, tmSrcV(b, base, e))|])
    }
  };
  while is_cons(ws) & not_bool(stuck) do {
    match ws { (a, v) :: rest => {
      let _ = memop_do(Loc("global", a), if op == "" then WrOp(eb, v) else BulkRedOp(op, t, eb, v));
      ws = rest
    }, [||] => () }
  }
}

/* ---------------------------------------------------------------- tensormap.replace */
val tmField : list(string) -> string
function tmField(m) = pick(m, [|"global_address", "rank", "box_dim", "global_dim", "global_stride", "element_stride", "elemtype",
                                "interleave_layout", "swizzle_mode", "swizzle_atomicity", "fill_mode"|], "")

val tmReplace : (int, string, int, int) -> int
function tmReplace(b, f, o, v) = match f {
  "global_address" => setWord(b, 3, v),
  "rank" => setWord(b, 2, lor_int(land_int(tmW(b, 2), lnot_int(255)), v + 1)),
  "elemtype" => setWord(b, 1, lor_int(land_int(tmW(b, 1), lnot_int(255)), land_int(v, 255))),
  "global_dim" => setWord(b, 4 + o, v),
  "global_stride" => setWord(b, 9 + o, v),
  "box_dim" => setBox(b, o, v),
  "element_stride" => setWord(b, 15, lor_int(land_int(tmW(b, 15), lnot_int(shl_int(255, 8 * o))), shl_int(v, 8 * o))),
  "swizzle_mode" => setWord(b, 14, lor_int(land_int(tmW(b, 14), lnot_int(shl_int(255, 16))), shl_int(v, 16))),
  "swizzle_atomicity" => setWord(b, 14, lor_int(land_int(tmW(b, 14), lnot_int(shl_int(255, 48))), shl_int(v, 48))),
  "interleave_layout" => setWord(b, 14, lor_int(land_int(tmW(b, 14), lnot_int(shl_int(255, 32))), shl_int(v, 32))),
  "fill_mode" => setWord(b, 14, lor_int(land_int(tmW(b, 14), lnot_int(shl_int(255, 24))), shl_int(v, 24))),
  _ => b
}

val exec_tma : (opcode, list(xop)) -> bool
function exec_tma(o, ops) = {
  let m = o.mods;
  if o.base == "cp" & has(m, "async") & has(m, "bulk") & has(m, "tensor") then {
    if has(m, "prefetch") then true
    else if has(m, "mbarrier::complete_tx::bytes") then match ops {
      XB([|d|], [||]) :: XB(ta, ti) :: XB([|mb|], [||]) :: rest => match tmOperand(ta, ti) {
        Some((b, cs)) => { tm_load(m, d, b, cs, mb, rest); true },
        None() => false
      },
      _ => false
    } else if has(m, "bulk_group") then match ops {
      XB(ta, ti) :: XB([|s|], [||]) :: _ => match tmOperand(ta, ti) {
        Some((b, cs)) => { tm_store(m, b, cs, s); true },
        None() => false
      },
      _ => false
    } else false
  } else if o.base == "tensormap" then {
    let sp = if has(m, "shared::cta") then "shared" else "generic";
    let f = tmField(m);
    let ordf = f == "box_dim" | f == "global_dim" | f == "global_stride" | f == "element_stride";
    match ops {
      /* "If the address specified by addr does not fall within the address window of .global or .shared::cta state space
         then the behavior is undefined" (9.7.9.27) */
      XB([|a|], _) :: _ if has(m, "replace") & not_bool(winUb("tensormap.replace", "addr", ea(a, sp), [|"global", "shared"|])) => true,
      [|XB([|a|], [||]), n|] if has(m, "replace") & not_bool(ordf) => {
        let l = ea(a, sp);
        let b = memop_do(l, RdOp(128));
        if not_bool(stuck) then { let _ = memop_do(l, WrOp(128, tmReplace(b, f, 0, rdu64(n)))); () };
        true
      },
      [|XB([|a|], [||]), od, n|] if has(m, "replace") & ordf => {
        let l = ea(a, sp);
        let b = memop_do(l, RdOp(128));
        if not_bool(stuck) then { let _ = memop_do(l, WrOp(128, tmReplace(b, f, rdu32(od), rdu64(n)))); () };
        true
      },
      [|XB([|d|], [||]), XB([|s|], [||]), _|] if has(m, "cp_fenceproxy") => {
        let v = memop_do(ea(s, "shared"), RdOp(128));
        if not_bool(stuck) then { let _ = memop_do(ea(d, "generic"), WrOp(128, v)); () };
        true
      },
      _ => false
    }
  } else false
}

/* ---------------------------------------------------------------- the .tensormap harness directives */
val tk_intl : unit -> list(int)
function tk_intl() = { let n = tok_int(); var l : list(int) = [||]; foreach (i from 1 to n) l = pend_append(l, [|tok_int()|]); l }

val tmEt : ty -> int
function tmEt(t) = match (t.k, t.w) {
  ("u", 8) => 0, ("u", 16) => 1, ("u", 32) => 2, ("s", 32) => 3, ("u", 64) => 4, ("s", 64) => 5, ("f", 16) => 6, ("f", 32) => 7,
  ("f", 64) => 9, ("bf", 16) => 10, ("tf", 32) => 11, _ => 0
}
val swzMode : int -> int
function swzMode(z) = if z >= 3 & z <= 6 then 3 else z
val swzAtom : int -> int
function swzAtom(z) = if z >= 4 & z <= 6 then z - 3 else 0

val tmBlob : (ty, int, list(int), list(int), list(int), int) -> list(int)
function tmBlob(t, a, ds, ss, bs, z) =
  [|1414021456, tmEt(t), int_list_len(ds), a, nth(ds, 0), nth(ds, 1), nth(ds, 2), nth(ds, 3), nth(ds, 4),
    nth(ss, 0), nth(ss, 1), nth(ss, 2), nth(ss, 3),
    lor_int(lor_int(nth(bs, 0), shl_int(nth(bs, 1), 16)), lor_int(shl_int(nth(bs, 2), 32), shl_int(nth(bs, 3), 48))),
    lor_int(lor_int(nth(bs, 4), shl_int(swzMode(land_int(z, 255)), 16)), lor_int(shl_int(asr_int(z, 8), 24), shl_int(swzAtom(land_int(z, 255)), 48))),
    72340172838076673|]

val packList : (list(int), int, int) -> int
function packList(l, w, off) = {
  var r : int = 0;
  var i : int = 0;
  var x = l;
  while is_cons(x) do { match x { v :: rest => { r = lor_int(r, shl_int(v + off, w * i)); i = i + 1; x = rest }, [||] => () } };
  r
}

val words_iv : list(int) -> list(initval)
function words_iv(l) = match l { [||] => [||], v :: rest => IVI(v) :: words_iv(rest) }

val buf_addr : string -> option(int)
function buf_addr(b) = if smap_has(SP_SYMS, b) then Some(sym_addr(smap_get(SP_SYMS, b))) else None()

val harness_tma : string -> bool
function harness_tma(d) = {
  if d == "TENSORMAP" | d == "TENSORMAPI" then {
    let n = tok_str();
    let t = tk_ty();
    let b = tok_str();
    let ds = tk_intl();
    let ss = tk_intl();
    let bs = tk_intl();
    let z = tok_int();
    let ls = if d == "TENSORMAPI" then tk_intl() else [||];
    let us = if d == "TENSORMAPI" then tk_intl() else [||];
    let es = if d == "TENSORMAPI" then tk_intl() else [||];
    match buf_addr(b) {
      Some(a) => {
        let w0 = tmBlob(t, a, ds, ss, bs, z);
        let w = if d == "TENSORMAP" then w0 else
          [|at(w0, 0), lor_int(at(w0, 1), shl_int(packList(ls, 16, 32768), 8)), lor_int(at(w0, 2), shl_int(packList(us, 16, 32768), 8)), at(w0, 3),
            at(w0, 4), at(w0, 5), at(w0, 6), at(w0, 7), at(w0, 8), at(w0, 9), at(w0, 10), at(w0, 11), at(w0, 12), at(w0, 13),
            lor_int(at(w0, 14), shl_int(1, 40)), packList(es, 8, 0)|];
        do_alloc(n, mkty("u", 64), 16, words_iv(w))
      },
      None() => fault(concat_str("tensor map over an unknown buffer ", b), 0)
    };
    true
  } else false
}

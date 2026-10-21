/* =============================================================================
 * PTX-EXEC-TEX (semantics/ptx-exec-tex.k): textures (tex, tld4, txq), surfaces (suld, sust, sured,
 * suq), istypep, and the harness directives .texture / .surface / .sampler (ptx-harness.k) that
 * build the descriptor objects in global memory.
 *
 * K computes in 4400-bit floats (WP); here values are exact dyadic numbers rounded to 4400 bits
 * after every operation (wr4), divisions at double precision where K uses divD, and logarithms
 * correctly rounded at double precision (ptx_math.sail), as mpfr does in K.
 * ============================================================================= */

/* ---------------------------------------------------------------- 4400-bit working precision */
val wr4 : fv -> fv
function wr4(f) = match f {
  FV_Fin(s, m, e) => {
    let nb = bitlen(m);
    if nb <= 4400 then f
    else {
      let sh = nb - 4400;
      var q = shr_int(m, sh);
      let r = m - q * pow2(sh);
      let h = pow2(sh - 1);
      if r > h | (r == h & fmod_int(q, 2) == 1) then q = q + 1;
      FV_Fin(s, q, e + sh)
    }
  },
  _ => f
}
val wadd : (fv, fv) -> fv
function wadd(a, b) = wr4(fv_add_exact(a, b, RN))
val wsub : (fv, fv) -> fv
function wsub(a, b) = wr4(fv_add_exact(a, fv_neg(b), RN))
val wmul : (fv, fv) -> fv
function wmul(a, b) = wr4(fv_mul_exact(a, b))
/* a quotient correctly rounded to 4400 bits */
val wdiv : (fv, fv) -> fv
function wdiv(a, b) = match (a, b) {
  (FV_Fin(sa, ma, xa), FV_Fin(sb, mb, xb)) => {
    if mb == 0 then (if ma == 0 then FV_NaN() else FV_Inf(sa != sb))
    else if ma == 0 then FV_Fin(sa != sb, 0, 0)
    else {
      let sh = max_int(0, 4402 + bitlen(mb) - bitlen(ma));
      let n = ma * pow2(sh);
      let q = tdiv_int(n, mb);
      let r = n - q * mb;
      fround(sa != sb, q, xa - xb - sh, r != 0, 4400, 40, RN)
    }
  },
  _ => FV_NaN()
}
val zF : unit -> fv
function zF() = FV_Fin(false, 0, 0)
val oneF : unit -> fv
function oneF() = FV_Fin(false, 1, 0)
val halfF : unit -> fv
function halfF() = FV_Fin(false, 1, -1)
val intF : int -> fv
function intF(i) = fv_of_int(i)
val fge : (fv, fv) -> bool
function fge(a, b) = not_bool(fv_is_nan(a) | fv_is_nan(b)) & fv_cmp(a, b) >= 0
val fle : (fv, fv) -> bool
function fle(a, b) = not_bool(fv_is_nan(a) | fv_is_nan(b)) & fv_cmp(a, b) <= 0
val maxF : (fv, fv) -> fv
function maxF(a, b) = if fge(a, b) then a else b
val minF : (fv, fv) -> fv
function minF(a, b) = if fle(a, b) then a else b
val floorF : fv -> fv
function floorF(x) = rndInt(x, "rmi")
val flo : fv -> int
function flo(x) = f2int(floorF(x))
/* divD: the quotient at double precision */
val divD : (fv, fv) -> fv
function divD(a, b) = div_r(round_fv(a, 53, 11, RN), round_fv(b, 53, 11, RN), 64)
val inv256 : unit -> fv
function inv256() = FV_Fin(false, 1, -8)
val qw : fv -> fv
function qw(xb) = wmul(intF(flo(wadd(wmul(wsub(xb, floorF(xb)), intF(256)), halfF()))), inv256())
val absF : fv -> fv
function absF(x) = if fv_lt(x, zF()) then wsub(zF(), x) else x

/* ---------------------------------------------------------------- descriptor fields */
val tdW : (int, int) -> int
function tdW(b, i) = land_int(asr_int(b, 64 * i), 18446744073709551615)
val hBlob : addr -> int
function hBlob(a) = mem_rdn(SP_GMEM, locOff(ea(a, "global")), 128)

val txBase : int -> int
function txBase(b) = tdW(b, 1)
val txCode : int -> int
function txCode(b) = tdW(b, 2)
val txCh : int -> int
function txCh(b) = tdW(b, 3)
val txLayers : int -> int
function txLayers(b) = max_int(1, tdW(b, 8))
val txDim : (int, int) -> int
function txDim(b, i) = max_int(1, tdW(b, 9 + i))
val txLevels : int -> int
function txLevels(b) = max_int(1, tdW(b, 12))
val lvD : (int, int, int) -> int
function lvD(b, lv, i) = max_int(1, asr_int(txDim(b, i), lv))
val lvTexels : (int, int) -> int
function lvTexels(b, lv) = txLayers(b) * lvD(b, lv, 0) * lvD(b, lv, 1) * lvD(b, lv, 2)
val lvOff : (int, int) -> int
function lvOff(b, lv) = { var o : int = 0; var l : int = 0; while l < lv do { o = o + lvTexels(b, l); l = l + 1 }; o }

/* s < 0: no sampler */
val effMip : (int, int) -> int
function effMip(b, s) = if s >= 0 then tdW(s, 4) else tdW(b, 15)
val minLodW : int -> int
function minLodW(s) = if s >= 0 then sext(tdW(s, 5), 64) else 0
val maxLodW : int -> int
function maxLodW(s) = if s >= 0 then sext(tdW(s, 6), 64) else 1048576
val txEbOf : int -> int
function txEbOf(c) = match c { 0 => 1, 20 => 1, 1 => 2, 21 => 2, 6 => 2, _ => 4 }
val txEb : int -> int
function txEb(b) = txEbOf(txCode(b))
val effFilter : (int, int) -> int
function effFilter(b, s) = if s >= 0 then tdW(s, 1) else tdW(b, 4)
val effAddr : (int, int) -> int
function effAddr(b, s) = if s >= 0 then tdW(s, 2) else tdW(b, 5)
val effNorm : (int, int) -> int
function effNorm(b, s) = if s >= 0 then tdW(s, 3) else tdW(b, 6)

/* ---------------------------------------------------------------- texels */
val adIdx : (int, int, int) -> int
function adIdx(md, i, n) = match md {
  0 => tmod_int(tmod_int(i, n) + n, n),
  3 => { let m = tmod_int(tmod_int(i, 2 * n) + 2 * n, 2 * n); if m < n then m else 2 * n - m - 1 },
  2 => if i >= 0 & i < n then i else -1,
  _ => min_int(max_int(i, 0), n - 1)
}

/* wrap and mirror need normalized coordinates: unnormalized coordinates clamp */
val amEff : (int, int) -> int
function amEff(am, nm) = if nm != 1 & (am == 0 | am == 3) then 1 else am

val codeBits : int -> int
function codeBits(c) = if c == 0 | c == 20 then 8 else if c == 1 | c == 21 then 16 else 32
val codeVal : (int, int) -> int
function codeVal(c, v) = if c == 20 | c == 21 | c == 3 | c == 5 then sext(v, codeBits(c)) else v
val codeMax : int -> int
function codeMax(c) = if c == 0 | c == 1 | c == 2 | c == 4 then pow2(codeBits(c)) - 1 else pow2(codeBits(c) - 1) - 1

val convF : (int, int, int) -> fv
function convF(c, v, n) =
  if c == 6 then decodeF(v, 16) else if c == 7 then decodeF(v, 32)
  else if n == 0 then intF(codeVal(c, v))
  else maxF(wdiv(intF(codeVal(c, v)), intF(codeMax(c))), intF(-1))

val texC : (int, int, int) -> fv
function texC(b, a, c) =
  if c < txCh(b) then convF(txCode(b), mem_rdn(SP_GMEM, a + c * txEb(b), txEb(b)), tdW(b, 7))
  else if c == 3 then oneF() else zF()

/* Q = [level, offset x, y, z, has depth reference, reference bits] */
val texel : (int, list(int), int, int, int, int) -> list(fv)
function texel(b, q, i, j, k, l) = {
  let lv = at(q, 0);
  let t : list(fv) = if i < 0 | j < 0 | k < 0 then [|zF(), zF(), zF(), zF()|]
    else {
      let a = txBase(b) + (lvOff(b, lv) + ((l * lvD(b, lv, 2) + k) * lvD(b, lv, 1) + j) * lvD(b, lv, 0) + i) * txCh(b) * txEb(b);
      [|texC(b, a, 0), texC(b, a, 1), texC(b, a, 2), texC(b, a, 3)|]
    };
  if at(q, 4) == 0 then t
  else [|if fle(decodeF(at(q, 5), 32), fv_nth(t, 0)) then oneF() else zF(), zF(), zF(), oneF()|]
}

val lerpF : (fv, fv, fv) -> fv
function lerpF(a, b, w) = wadd(wmul(wsub(oneF(), w), a), wmul(w, b))
val lerp4 : (list(fv), list(fv), fv) -> list(fv)
function lerp4(a, b, w) = [|lerpF(fv_nth(a, 0), fv_nth(b, 0), w), lerpF(fv_nth(a, 1), fv_nth(b, 1), w),
                             lerpF(fv_nth(a, 2), fv_nth(b, 2), w), lerpF(fv_nth(a, 3), fv_nth(b, 3), w)|]

/* ---------------------------------------------------------------- coordinates and filtering */
val posF : (fv, int, int) -> fv
function posF(x, n, nm) = if nm == 1 then wmul(x, intF(n)) else x

val tIdx : (int, list(int), int, int, fv, int, int, int) -> int
function tIdx(b, q, am, nm, x, d, dim, lin) = {
  let p = posF(x, lvD(b, at(q, 0), dim), nm);
  adIdx(amEff(am, nm), flo(if lin == 1 then wsub(p, halfF()) else p) + d + at(q, 1 + dim), lvD(b, at(q, 0), dim))
}
val tW : (int, list(int), int, fv, int) -> fv
function tW(b, q, nm, x, dim) = qw(wsub(posF(x, lvD(b, at(q, 0), dim), nm), halfF()))

val sample : (int, list(int), int, int, int, fv, fv, fv, int, int) -> list(fv)
function sample(b, q, f, am, nm, x, y, z, l, r) = {
  if f == 1 then {
    let ix0 = tIdx(b, q, am, nm, x, 0, 0, 1);
    let ix1 = tIdx(b, q, am, nm, x, 1, 0, 1);
    let jy0 = if r >= 2 then tIdx(b, q, am, nm, y, 0, 1, 1) else 0;
    let jy1 = if r >= 2 then tIdx(b, q, am, nm, y, 1, 1, 1) else 0;
    let kz0 = if r >= 3 then tIdx(b, q, am, nm, z, 0, 2, 1) else 0;
    let kz1 = if r >= 3 then tIdx(b, q, am, nm, z, 1, 2, 1) else 0;
    let wx = tW(b, q, nm, x, 0);
    let wy = if r >= 2 then tW(b, q, nm, y, 1) else zF();
    let wz = if r >= 3 then tW(b, q, nm, z, 2) else zF();
    lerp4(lerp4(lerp4(texel(b, q, ix0, jy0, kz0, l), texel(b, q, ix1, jy0, kz0, l), wx),
                lerp4(texel(b, q, ix0, jy1, kz0, l), texel(b, q, ix1, jy1, kz0, l), wx), wy),
          lerp4(lerp4(texel(b, q, ix0, jy0, kz1, l), texel(b, q, ix1, jy0, kz1, l), wx),
                lerp4(texel(b, q, ix0, jy1, kz1, l), texel(b, q, ix1, jy1, kz1, l), wx), wy), wz)
  } else
    texel(b, q, tIdx(b, q, am, nm, x, 0, 0, 0), if r >= 2 then tIdx(b, q, am, nm, y, 0, 1, 0) else 0,
          if r >= 3 then tIdx(b, q, am, nm, z, 0, 2, 0) else 0, l)
}

/* ---------------------------------------------------------------- mipmaps */
val lodF : int -> fv
function lodF(v) = wmul(intF(v), inv256())
val lamClamp : (int, int, fv) -> fv
function lamClamp(b, s, lam) = minF(maxF(maxF(lam, lodF(minLodW(s))), zF()), minF(lodF(maxLodW(s)), intF(txLevels(b) - 1)))
val setLv : (list(int), int) -> list(int)
function setLv(q, lv) = [|lv, at(q, 1), at(q, 2), at(q, 3), at(q, 4), at(q, 5)|]

val mipSample : (int, int, bool, fv, list(int), int, int, int, fv, fv, fv, int, int) -> list(fv)
function mipSample(b, s, uselod, lam, q, f, am, nm, x, y, z, l, r) = {
  if not_bool(uselod) then sample(b, q, f, am, nm, x, y, z, l, r)
  else {
    let lc = lamClamp(b, s, lam);
    if effMip(b, s) == 1 then
      lerp4(sample(b, setLv(q, flo(lc)), f, am, nm, x, y, z, l, r),
            sample(b, setLv(q, min_int(flo(lc) + 1, txLevels(b) - 1)), f, am, nm, x, y, z, l, r), qw(lc))
    else sample(b, setLv(q, min_int(flo(wadd(lc, halfF())), txLevels(b) - 1)), f, am, nm, x, y, z, l, r)
  }
}

val lamLog : fv -> fv
function lamLog(s) =
  if fv_lt(zF(), s) then divD(log_r(round_fv(s, 53, 11, RN), 64), wmul(intF(2), log_r(intF(2), 64))) else intF(-1000)

val gradLam : (int, int, int, list(fv), list(fv), fv) -> fv
function gradLam(b, nm, r, dx, dy, sd) = {
  var lx : fv = zF();
  var ly : fv = zF();
  /* gLen2 sums from the last term: t0 + (t1 + (t2 + 0)) */
  foreach (i from (r - 1) downto 0) {
    let sc = if nm == 1 then divD(intF(txDim(b, i)), sd) else divD(oneF(), sd);
    let tx = wmul(fv_nth(dx, i), sc);
    let ty = wmul(fv_nth(dy, i), sc);
    lx = wadd(wmul(tx, tx), lx);
    ly = wadd(wmul(ty, ty), ly)
  };
  lamLog(maxF(lx, ly))
}

/* ---------------------------------------------------------------- cubemaps */
val cubeFace : (fv, fv, fv) -> int
function cubeFace(x, y, z) =
  if fge(absF(x), absF(y)) & fge(absF(x), absF(z)) then (if fge(x, zF()) then 0 else 1)
  else if fv_lt(absF(x), absF(y)) & fge(absF(y), absF(z)) then (if fge(y, zF()) then 2 else 3)
  else (if fge(z, zF()) then 4 else 5)
val cubeMa : (fv, fv, fv) -> fv
function cubeMa(x, y, z) = { let f = cubeFace(x, y, z); if f < 2 then absF(x) else if f < 4 then absF(y) else absF(z) }
val negF : fv -> fv
function negF(x) = wsub(zF(), x)
val cubeSC : (fv, fv, fv) -> fv
function cubeSC(x, y, z) = match cubeFace(x, y, z) { 0 => negF(z), 1 => z, 5 => negF(x), _ => x }
val cubeTC : (fv, fv, fv) -> fv
function cubeTC(x, y, z) = match cubeFace(x, y, z) { 2 => z, 3 => negF(z), _ => negF(y) }
val cubeNorm : (fv, fv) -> fv
function cubeNorm(s, ma) = if fv_lt(zF(), ma) then wmul(wadd(divD(s, ma), oneF()), halfF()) else halfF()

/* ---------------------------------------------------------------- operands */
val outBits : (ty, fv) -> int
function outBits(t, f) =
  if (t.k == "s" | t.k == "u") & t.w == 32 then trunc(f2int(wr4(f)), 32) else fin(noMods(), 32, f, false)

val rankOf : list(string) -> int
function rankOf(m) = if has(m, "1d") | has(m, "a1d") then 1 else if has(m, "3d") then 3 else 2
val isArr : list(string) -> bool
function isArr(m) = has(m, "a1d") | has(m, "a2d") | has(m, "acube")
val isCube : list(string) -> bool
function isCube(m) = has(m, "cube") | has(m, "acube")

val coordF : (sop, ty) -> fv
function coordF(o, t) = if ty_eq(t, mkty("f", 32)) then decodeF(rd_sop(o, mkty("b", 32)), 32) else intF(sext(rd_sop(o, mkty("s", 32)), 32))
val vecAt : (list(sop), int) -> sop
function vecAt(l, i) = if i >= 0 & i < sop_count(l) then sop_nth(l, i) else SI(0)

val nLod : list(string) -> int
function nLod(m) = if has(m, "level") then 1 else if has(m, "grad") then 2 else 0

val txOffs : list(xop) -> list(int)
function txOffs(l) = match l {
  XV(v) :: _ => if is_cons(v) then [|sext(rd_sop(vecAt(v, 0), mkty("s", 32)), 32), if sop_count(v) > 1 then sext(rd_sop(vecAt(v, 1), mkty("s", 32)), 32) else 0,
                                     if sop_count(v) > 2 then sext(rd_sop(vecAt(v, 2), mkty("s", 32)), 32) else 0|] else [||],
  _ => [||]
}
val txRest : list(xop) -> list(xop)
function txRest(l) = match l { XV(v) :: rest => if is_cons(v) then rest else l, _ => l }
val mkQ : (list(int), list(xop)) -> list(int)
function mkQ(offs, rest) = {
  let r = txRest(rest);
  [|0, nthD(offs, 0, 0), nthD(offs, 1, 0), nthD(offs, 2, 0), b2i(is_cons(r)), match r { x :: _ => rd(x, mkty("b", 32)), [||] => 0 }|]
}
val lodArg : list(xop) -> fv
function lodArg(l) = match l { XS(s) :: _ => coordF(s, mkty("f", 32)), _ => zF() }
val gradVec : (list(xop), int) -> list(fv)
function gradVec(l, i) = match xop_nth(l, i) {
  XV(v) => { var out : list(fv) = [||]; var x = v;
             while is_cons(x) do { match x { s :: rest => { out = pend_append(out, [|coordF(s, mkty("f", 32))|]); x = rest }, [||] => () } }; out },
  _ => [||]
}

val tex_run : (int, int, list(string), ty, list(sop), list(xop)) -> list(fv)
function tex_run(b, s, m, ct, cs, ex) = {
  let rest = ldropx(ex, nLod(m));
  let q = mkQ(txOffs(rest), rest);
  let f32 = mkty("f", 32);
  if isCube(m) then {
    let o = if has(m, "acube") then 1 else 0;
    let x = coordF(vecAt(cs, o), f32);
    let y = coordF(vecAt(cs, o + 1), f32);
    let z = coordF(vecAt(cs, o + 2), f32);
    let ma = cubeMa(x, y, z);
    let lam = if nLod(m) == 0 then zF() else if has(m, "level") then lodArg(ex) else gradLam(b, 1, 3, gradVec(ex, 0), gradVec(ex, 1), wmul(ma, intF(2)));
    mipSample(b, s, nLod(m) > 0, lam, q, effFilter(b, s), 1, 1, cubeNorm(cubeSC(x, y, z), ma), cubeNorm(cubeTC(x, y, z), ma), zF(),
              6 * min_int(if has(m, "acube") then rd_sop(vecAt(cs, 0), mkty("u", 32)) else 0, max_int(tdiv_int(txLayers(b), 6) - 1, 0)) + cubeFace(x, y, z), 2)
  } else {
    let o = if isArr(m) then 1 else 0;
    let lam = if nLod(m) == 0 then zF() else if has(m, "level") then lodArg(ex) else gradLam(b, effNorm(b, s), rankOf(m), gradVec(ex, 0), gradVec(ex, 1), oneF());
    mipSample(b, s, nLod(m) > 0, lam, q, if ty_eq(ct, f32) then effFilter(b, s) else 0, effAddr(b, s), effNorm(b, s),
              coordF(vecAt(cs, o), ct), coordF(vecAt(cs, o + 1), ct), coordF(vecAt(cs, o + 2), ct),
              if isArr(m) then min_int(rd_sop(vecAt(cs, 0), mkty("u", 32)), txLayers(b) - 1) else 0, rankOf(m))
  }
}

val gcomp : list(string) -> int
function gcomp(m) = if has(m, "g") then 1 else if has(m, "b") then 2 else if has(m, "a") then 3 else 0

val gather_run : (int, int, list(string), list(sop), list(xop)) -> list(fv)
function gather_run(b, s, m, cs, ex) = {
  let q = mkQ(txOffs(ex), ex);
  let f32 = mkty("f", 32);
  let (x, y, l, nm) : (fv, fv, int, int) = if isCube(m) then {
    let o = if has(m, "acube") then 1 else 0;
    let cx = coordF(vecAt(cs, o), f32);
    let cy = coordF(vecAt(cs, o + 1), f32);
    let cz = coordF(vecAt(cs, o + 2), f32);
    let ma = cubeMa(cx, cy, cz);
    (cubeNorm(cubeSC(cx, cy, cz), ma), cubeNorm(cubeTC(cx, cy, cz), ma),
     6 * min_int(if has(m, "acube") then rd_sop(vecAt(cs, 0), mkty("u", 32)) else 0, max_int(tdiv_int(txLayers(b), 6) - 1, 0)) + cubeFace(cx, cy, cz), 1)
  } else {
    let o = if isArr(m) then 1 else 0;
    (coordF(vecAt(cs, o), f32), coordF(vecAt(cs, o + 1), f32), if isArr(m) then min_int(rd_sop(vecAt(cs, 0), mkty("u", 32)), txLayers(b) - 1) else 0,
     effNorm(b, s))
  };
  let am = effAddr(b, s);
  let cc = gcomp(m);
  let i0 = tIdx(b, q, am, nm, x, 0, 0, 1);
  let i1 = tIdx(b, q, am, nm, x, 1, 0, 1);
  let j0 = tIdx(b, q, am, nm, y, 0, 1, 1);
  let j1 = tIdx(b, q, am, nm, y, 1, 1, 1);
  [|fv_nth(texel(b, q, i0, j1, 0, l), cc), fv_nth(texel(b, q, i1, j1, 0, l), cc), fv_nth(texel(b, q, i1, j0, 0, l), cc), fv_nth(texel(b, q, i0, j0, 0, l), cc)|]
}

val texOut : (xop, ty, list(fv)) -> unit
function texOut(dv, t, vs) = {
  var bits : list(int) = [||];
  var x = vs;
  while is_cons(x) do { match x { f :: rest => { bits = pend_append(bits, [|outBits(t, f)|]); x = rest }, [||] => () } };
  match dv {
    XVP(l, isreg, p) => { collWrite(vecOps(XV(l)), bits); wr_sop(if isreg then SR(p) else SN(p), 1) },
    _ => collWrite(vecOps(dv), bits)
  }
}

val txqVal : (list(string), int) -> int
function txqVal(m, b) =
  if has(m, "width") then txDim(b, 0) else if has(m, "height") then txDim(b, 1) else if has(m, "depth") then txDim(b, 2)
  else if has(m, "array_size") then (if land_int(tdW(b, 13), 2) != 0 then tdiv_int(tdW(b, 8), 6) else tdW(b, 8))
  else if has(m, "normalized_coords") then tdW(b, 6) else if has(m, "filter_mode") then tdW(b, 4)
  else if has(m, "addr_mode_0") | has(m, "addr_mode_1") | has(m, "addr_mode_2") then tdW(b, 5)
  else if has(m, "num_mipmap_levels") then txLevels(b) else 0

/* ---------------------------------------------------------------- surfaces */
val sfBase : int -> int
function sfBase(b) = tdW(b, 1)
val sfTb : int -> int
function sfTb(b) = max_int(1, tdW(b, 2))
val sfLayers : int -> int
function sfLayers(b) = max_int(1, tdW(b, 3))
val sfDim : (int, int) -> int
function sfDim(b, i) = max_int(1, tdW(b, 4 + i))

val sfAddr : (int, int, int, int, int, int) -> int
function sfAddr(b, x, y, z, l, n) =
  if x < 0 | x + n > sfDim(b, 0) * sfTb(b) | y < 0 | y >= sfDim(b, 1) | z < 0 | z >= sfDim(b, 2) | l < 0 | l >= sfLayers(b) then -1
  else sfBase(b) + (((l * sfDim(b, 2) + z) * sfDim(b, 1) + y) * sfDim(b, 0)) * sfTb(b) + x

val sfCoords : (list(string), list(sop)) -> list(int)
function sfCoords(m, cs) = {
  if isArr(m) then [|sext(rd_sop(vecAt(cs, 1), mkty("s", 32)), 32), if rankOf(m) >= 2 then sext(rd_sop(vecAt(cs, 2), mkty("s", 32)), 32) else 0, 0,
                     rd_sop(vecAt(cs, 0), mkty("u", 32))|]
  else [|sext(rd_sop(vecAt(cs, 0), mkty("s", 32)), 32), if rankOf(m) >= 2 then sext(rd_sop(vecAt(cs, 1), mkty("s", 32)), 32) else 0,
         if rankOf(m) >= 3 then sext(rd_sop(vecAt(cs, 2), mkty("s", 32)), 32) else 0, 0|]
}

val sfVec : list(string) -> int
function sfVec(m) = if has(m, "v4") then 4 else if has(m, "v2") then 2 else 1

val sfAN : (int, list(string), int, list(int)) -> int
function sfAN(b, m, n, xs) =
  if has(m, "clamp") then
    sfAddr(b, min_int(max_int(at(xs, 0), 0), max_int(sfDim(b, 0) * sfTb(b) - n, 0)), min_int(max_int(at(xs, 1), 0), sfDim(b, 1) - 1),
           min_int(max_int(at(xs, 2), 0), sfDim(b, 2) - 1), min_int(max_int(at(xs, 3), 0), sfLayers(b) - 1), n)
  else sfAddr(b, at(xs, 0), at(xs, 1), at(xs, 2), at(xs, 3), n)
val sfA : (int, list(string), ty, list(int)) -> int
function sfA(b, m, t, xs) = sfAN(b, m, sfVec(m) * bytes(t), xs)

val dstList : xop -> list(xop)
function dstList(o) = match o { XV(_) => vecOps(o), _ => [|o|] }
val srcList : (xop, ty) -> list(int)
function srcList(o, t) = match o { XV(l) => rdList(l, t), _ => [|rd(o, t)|] }

val sfScaleX : (list(int), int) -> list(int)
function sfScaleX(xs, n) = match xs { x :: rest => (x * n) :: rest, [||] => [||] }

val chBytesOf : int -> int
function chBytesOf(f) = if f <= 3 then 1 else if f <= 8 then 2 else 4
val maskPow : int -> int
function maskPow(f) = pow2(8 * chBytesOf(f) - (if f == 1 | f == 5 then 1 else 0)) - 1
val sfClamp : (fv, fv, fv) -> fv
function sfClamp(x, lo, hi) = if fv_is_nan(x) then zF() else if fv_lt(x, lo) then lo else if fv_lt(hi, x) then hi else x
val rneI : fv -> int
function rneI(x) = {
  let fl = floorF(x);
  let fr = wsub(x, fl);
  if fv_lt(halfF(), fr) | (fv_eq(fr, halfF()) & tmod_int(f2int(fl), 2) == 1) then f2int(fl) + 1 else f2int(fl)
}
val fmtConv : (int, int) -> int
function fmtConv(f, v) = {
  let cb = 8 * chBytesOf(f);
  if f == 0 | f == 4 then rneI(wmul(sfClamp(decodeF(v, 32), zF(), oneF()), intF(maskPow(f))))
  else if f == 1 | f == 5 then trunc(rneI(wmul(sfClamp(decodeF(v, 32), intF(-1), oneF()), intF(maskPow(f)))), cb)
  else if f == 2 | f == 6 | f == 9 then min_int(v, pow2(cb) - 1)
  else if f == 3 | f == 7 | f == 10 then trunc(min_int(max_int(sext(v, 32), negate(pow2(cb - 1))), pow2(cb - 1) - 1), cb)
  else if f == 8 then fin(noMods(), 16, decodeF(v, 32), false)
  else v
}

val sfStore : (int, int, list(int), list(string)) -> unit
function sfStore(a, eb, vs, m) = {
  if a >= 0 then { let _ = memop_do(Loc("global", a), WrOp(list_length(vs) * eb, packElems(vs, 8 * eb))); () }
  else if not_bool(has(m, "zero")) then fault("surface access out of bounds", a)
}
val sfRed : (int, string, ty, int, list(string)) -> unit
function sfRed(a, op, t, v, m) = {
  if a >= 0 then { let _ = memop_do(Loc("global", a), AtOp(op, t, v, 0)); () }
  else if not_bool(has(m, "zero")) then fault("surface access out of bounds", a)
}

val suqVal : (list(string), int) -> int
function suqVal(m, b) =
  if has(m, "width") then sfDim(b, 0) else if has(m, "height") then sfDim(b, 1) else if has(m, "depth") then sfDim(b, 2)
  else if has(m, "array_size") then (if tdW(b, 12) != 0 then tdW(b, 3) else 0)
  else if has(m, "channel_data_type") then tdW(b, 9) else if has(m, "channel_order") then tdW(b, 10)
  else if has(m, "memory_layout") then tdW(b, 11) else 0

val istpMagic : list(string) -> int
function istpMagic(m) = if has(m, "texref") then 1415075924 else if has(m, "samplerref") then 1396920397 else if has(m, "surfref") then 1398035526 else -1

/* ---------------------------------------------------------------- instructions */
val exec_tex : (opcode, list(xop)) -> bool
function exec_tex(o, ops) = {
  let m = o.mods;
  match (o.base, o.tys, ops) {
    ("tex", [|dt, ct|], dv :: XB([|t|], [|IV(cs)|]) :: ex) => { texOut(dv, dt, tex_run(hBlob(t), -1, m, ct, cs, ex)); true },
    ("tex", [|dt, ct|], dv :: XB([|t, s|], [|IV(cs)|]) :: ex) => { texOut(dv, dt, tex_run(hBlob(t), hBlob(s), m, ct, cs, ex)); true },
    ("tld4", [|dt, _|], dv :: XB([|t|], [|IV(cs)|]) :: ex) => { texOut(dv, dt, gather_run(hBlob(t), -1, m, cs, ex)); true },
    ("tld4", [|dt, _|], dv :: XB([|t, s|], [|IV(cs)|]) :: ex) => { texOut(dv, dt, gather_run(hBlob(t), hBlob(s), m, cs, ex)); true },
    ("txq", _, [|d, XB([|t|], [||])|]) => { wr(d, txqVal(m, hBlob(t))); true },
    ("suld", [|t|], [|dv, XB([|s|], [|IV(cs)|])|]) => {
      if has(m, "b") then {
        let n = sfVec(m);
        let a = sfA(hBlob(s), m, t, sfCoords(m, cs));
        if a >= 0 then {
          let v = memop_do(Loc("global", a), RdOp(n * bytes(t)));
          if not_bool(stuck) then collWrite(dstList(dv), elemsOf(v, 8 * bytes(t), n))
        } else if has(m, "zero") then collWrite(dstList(dv), wordsOf(0, n))
        else fault("surface access out of bounds", a);
        true
      } else false
    },
    ("sust", [|t|], [|XB([|s|], [|IV(cs)|]), vs|]) => {
      let b = hBlob(s);
      if has(m, "b") then { sfStore(sfA(b, m, t, sfCoords(m, cs)), bytes(t), srcList(vs, t), m); true }
      else if has(m, "p") then {
        let src = srcList(vs, mkty("b", 32));
        let ch = max_int(1, tdW(b, 8));
        let f = tdW(b, 7);
        var cv : list(int) = [||];
        var x = src;
        var k : int = 0;
        while is_cons(x) & k < ch do { match x { v :: rest => { cv = pend_append(cv, [|fmtConv(f, v)|]); x = rest; k = k + 1 }, [||] => () } };
        sfStore(sfAN(b, m, min_int(list_length(src), ch) * chBytesOf(f), sfScaleX(sfCoords(m, cs), sfTb(b))), chBytesOf(f), cv, m);
        true
      } else false
    },
    ("sured", [|t|], [|XB([|s|], [|IV(cs)|]), v|]) => {
      let b = hBlob(s);
      if has(m, "p") then {
        let f = tdW(b, 7);
        sfRed(sfA(b, m, t, sfScaleX(sfCoords(m, cs), bytes(t))), atomOp(m), mkty(if f == 3 | f == 7 | f == 10 then "s" else "u", 8 * bytes(t)), rd(v, t), m)
      } else sfRed(sfA(b, m, t, sfCoords(m, cs)), atomOp(m), t, rd(v, t), m);
      true
    },
    ("suq", _, [|d, XB([|s|], [||])|]) => { wr(d, suqVal(m, hBlob(s))); true },
    ("istypep", _, [|d, a|]) => {
      let h = rdu64(a);
      if h >= 4096 & h + 8 <= gtop then {
        let v = memop_do(Loc("global", h), RdOp(8));
        if not_bool(stuck) then wr(d, b2i(v == istpMagic(m)))
      } else wr(d, 0);
      true
    },
    _ => false
  }
}

/* ---------------------------------------------------------------- the .texture / .surface / .sampler directives */
val tmCode : ty -> int
function tmCode(t) = match (t.k, t.w) {
  ("u", 8) => 0, ("u", 16) => 1, ("u", 32) => 2, ("s", 32) => 3, ("u", 64) => 4, ("s", 64) => 5, ("f", 16) => 6, ("f", 32) => 7,
  ("s", 8) => 20, ("s", 16) => 21, ("f", 64) => 8, ("bf", 16) => 9, ("tf", 32) => 11, _ => 0
}

val harness_tex : string -> bool
function harness_tex(d) = {
  if d == "TEXTURE" | d == "SURFACE" then {
    let n = tok_str();
    let t = tk_ty();
    let b = tok_str();
    let ds = tk_intl();
    let as_ = tk_intl();
    match buf_addr(b) {
      Some(a) => {
        let w : list(int) = if d == "TEXTURE" then
            [|1415075924, a, tmCode(t), nth(as_, 0), nth(as_, 1), nth(as_, 2), nth(as_, 3), nth(as_, 4), nth(as_, 5), nth(ds, 0), nth(ds, 1), nth(ds, 2),
              nthD(as_, 6, 1), nthD(as_, 7, 0), 0, nthD(as_, 8, 0)|]
          else
            [|1398035526, a, bytes(t), nth(as_, 0), nth(ds, 0), nth(ds, 1), nth(ds, 2), nthD(as_, 1, 11), nthD(as_, 2, 1), nthD(as_, 3, 0), nthD(as_, 4, 0),
              nthD(as_, 5, 1), nthD(as_, 6, b2i(nth(as_, 0) > 1)), 0, 0, 0, 0|];
        do_alloc(n, mkty("u", 64), 16, words_iv(w))
      },
      None() => fault(concat_str("texture over an unknown buffer ", b), 0)
    };
    true
  } else if d == "SAMPLER" then {
    let n = tok_str();
    let as_ = tk_intl();
    do_alloc(n, mkty("u", 64), 16, words_iv([|1396920397, nthD(as_, 0, 0), nthD(as_, 1, 0), nthD(as_, 2, 0), nthD(as_, 3, 0), nthD(as_, 4, 0),
                                               nthD(as_, 5, 1048576), 0, 0, 0, 0, 0, 0, 0, 0, 0|]));
    true
  } else false
}

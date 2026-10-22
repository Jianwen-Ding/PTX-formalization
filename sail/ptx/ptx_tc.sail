/* =============================================================================
 * PTX-EXEC-TC (semantics/ptx-exec-tc.k): fifth-generation tensor core (tcgen05): tensor memory
 * allocation, ld / st (with .red and .spcompress), mma (dense, sparse, block-scaled, .ws,
 * .ashift, cta_group::2, .kind::ti16, .decompress::lut::b), commit, cp (with decompression) and shift.
 *
 * Tensor memory, its allocations and the warp-collective alloc / dealloc results are per-CTA maps
 * (spaces created on first use).  Asynchronous operations complete at issue, as in K.
 * ============================================================================= */

/* the per-CTA spaces: "tm" tensor memory, "al" allocations, "rs" alloc / dealloc results */
val tcSpace : (int, string) -> int
function tcSpace(c, k) = {
  let mp = c_misc[cchk(c)];
  let key = concat_str("sp:", k);
  if smap_has(mp, key) then smap_get(mp, key) else { let s = new_space(); smap_set(mp, key, s); s }
}

val tcLane : int -> int
function tcLane(a) = land_int(asr_int(a, 16), 65535)
val tcCol : int -> int
function tcCol(a) = land_int(a, 65535)
val tcKey : (int, int) -> int
function tcKey(l, c) = l * 65536 + c
val tcAddr : addr -> int
function tcAddr(a) = locOff(ea(a, "global"))

val tcWord : (int, int) -> int
function tcWord(tm, k) = smap_get(tm, dec_str(k))
val tcPut : (int, int, int) -> unit
function tcPut(tm, k, v) = smap_set(tm, dec_str(k), v)

/* ---------------------------------------------------------------- allocation */
val tcOverlap : (int, int, int) -> bool
function tcOverlap(al, b, n) = {
  var r : bool = false;
  foreach (i from 0 to (smap_size(al) - 1)) {
    let k = str_int(smap_key(al, i));
    if b < k + smap_get(al, smap_key(al, i)) & k < b + n then r = true
  };
  r
}
val tcFind : (int, int) -> int
function tcFind(al, n) = {
  var b : int = 0;
  var r : int = -1;
  while r < 0 & b + n <= 512 & n > 0 do { if not_bool(tcOverlap(al, b, n)) then r = b else b = b + n };
  r
}
val tcPow2 : int -> bool
function tcPow2(n) = n >= 32 & n <= 512 & popc_int(n) == 1
/* .exclusive (PTX ISA 9.4 9.7.18.7.1, Table 60): "nCols must be a multiple of 32, in the range [32, 512] on target
   sm_100f, or the range [32, 576] on target sm_107f" (sm_100, sm_103, sm_110: 512; sm_107: 576) */
val tcExMax : unit -> int
function tcExMax() = if isasm == 107 then 576 else 512
val tcExOk : int -> bool
function tcExOk(n) = n >= 32 & n <= tcExMax() & tmod_int(n, 32) == 0

val tcInst : string -> string
function tcInst(k) = {
  let t = tchk(cur);
  concat_str(dec_str(tdiv_int(t_tid[t], 32)), concat_str(":", concat_str(k, concat_str(":",
    dec_str((if smap_has(t_cseq[t], seqKey(k, 4294967295)) then smap_get(t_cseq[t], seqKey(k, 4294967295)) else 1) - 1)))))
}

/* a thread whose tcgen05.alloc waits for columns: its destination and size */
register t_tcal : vector(16384, loc) = vector_init(BadLoc(""))

/* tcgen05.alloc (ISA 9.7.17.5): the warp's result is computed once per instance; when the columns are not available
   the instruction blocks until they become available (a thread waits in K_wait("tcalloc") and retries) */
/* .exclusive: "An exclusive allocation operation blocks until there is no other live allocation. Until the sole live
   allocation is deallocated, no other CTA may allocate, whether exclusively or nonexclusively" (9.4 9.7.18.7.1); the
   model's tensor memory is per CTA, so the rule applies to the CTA's own allocations.  "tcex" in the CTA's misc map:
   an exclusive allocation is live (it starts at column 0) */
val tcAllocTry : (loc, int, bool) -> bool
function tcAllocTry(l, n, ex) = {
  let c = cur_cta();
  let rs = tcSpace(c, "rs");
  let al = tcSpace(c, "al");
  let inst = tcInst("tcgen05.alloc");
  let exLive = smap_has(c_misc[cchk(c)], "tcex");
  if smap_has(rs, inst) then { let _ = memop_do(l, WrOp(4, smap_get(rs, inst))); true }
  else if not_bool(if ex then tcExOk(n) else tcPow2(n)) then { fault("tcgen05.alloc: invalid size", n); true }
  /* "it is illegal for a CTA to perform tcgen05.alloc after any of its constituent threads execute
     tcgen05.relinquish_alloc_permit"; "The number of columns allocated should not increase between any two allocations
     in the execution order within the CTA" (9.7.17.7.1): reported under .ub strict */
  else if ub & smap_has(c_misc[cchk(c)], "tcrelinq") then { fault("tcgen05.alloc after tcgen05.relinquish_alloc_permit", n); true }
  else if ub & smap_has(c_misc[cchk(c)], "tclast") & n > smap_get(c_misc[cchk(c)], "tclast") then {
    fault("tcgen05.alloc: the number of columns increases between two allocations", n); true
  } else if ex then {
    if smap_size(al) > 0 then false
    else {
      let _ = memop_do(l, WrOp(4, 0));
      smap_set(rs, inst, 0);
      smap_set(al, "0", n);
      smap_set(c_misc[cchk(c)], "tcex", 1);
      smap_set(c_misc[cchk(c)], "tclast", n);
      true
    }
  } else if not_bool(exLive) & tcFind(al, n) >= 0 then {
    let b = tcFind(al, n);
    let _ = memop_do(l, WrOp(4, b));
    smap_set(rs, inst, b);
    smap_set(al, dec_str(b), n);
    smap_set(c_misc[cchk(c)], "tclast", n);
    true
  } else false
}

/* retry of a blocked tcgen05.alloc (from the scheduler's wait hook) */
val tcAllocRetry : unit -> bool
function tcAllocRetry() = {
  let t = tchk(cur);
  if tcAllocTry(t_tcal[t], at(t_wd[t], 0), at(t_wd[t], 1) == 1) then { t_cont[t] = K_none(); true } else false
}

val post_tc : (cpost, list(int)) -> unit
function post_tc(p, vs) = {
  let c = cur_cta();
  let rs = tcSpace(c, "rs");
  let al = tcSpace(c, "al");
  match p {
    P_tcalloc(l, n, ex) => {
      if not_bool(tcAllocTry(l, n, ex)) then {
        let t = tchk(cur);
        t_tcal[t] = l;
        t_wd[t] = [|n, b2i(ex)|];
        t_cont[t] = K_wait("tcalloc", 0)
      }
    },
    P_tcdealloc(b, n, ex) => {
      let inst = tcInst("tcgen05.dealloc");
      let exLive = smap_has(c_misc[cchk(c)], "tcex");
      if smap_has(rs, inst) then ()
      /* "Memory must be deallocated with .exclusive if and only if it is allocated with .exclusive" (9.4 9.7.18.7.1):
         under .ub strict; otherwise the allocation is freed */
      else if ub & smap_has(al, dec_str(b)) & ex != exLive then
        fault(if ex then "tcgen05.dealloc.exclusive of a non-exclusive allocation" else "tcgen05.dealloc of an exclusive allocation without .exclusive", b)
      else if smap_has(al, dec_str(b)) & smap_get(al, dec_str(b)) == n then {
        smap_set(rs, inst, 0); smap_del(al, dec_str(b));
        if exLive then smap_del(c_misc[cchk(c)], "tcex")
      }
      else fault("tcgen05.dealloc: not an allocation of that size", b)
    },
    _ => ()
  }
}

/* ---------------------------------------------------------------- ld / st */
val tcNum : list(string) -> int
function tcNum(m) = if has(m, "x128") then 128 else if has(m, "x64") then 64 else if has(m, "x32") then 32 else if has(m, "x16") then 16
                    else if has(m, "x8") then 8 else if has(m, "x4") then 4 else if has(m, "x2") then 2 else 1
val tcPer : list(string) -> int
function tcPer(m) = if has(m, "16x256b") then 4 else if has(m, "16x128b") then 2 else 1

/* (lane, column) of register i of thread t (ISA 9.7.17.2.3, Figures 183-187): in the 16-lane shapes thread t covers
   lane t / 4 (and lane t / 4 + 8): .16x64b lane t / 4 + 8 (t % 2), column (t / 2) % 2 of each 64-bit repetition;
   .16x128b column t % 4 of each 128-bit repetition, r0 / r1 in lanes t / 4 / t / 4 + 8; .16x256b columns
   2 (t % 4), 2 (t % 4) + 1 of each 256-bit repetition, r0 r1 / r2 r3 in lanes t / 4 / t / 4 + 8 */
val tcCell : (list(string), int, int, int, int) -> int
function tcCell(m, a, t, i, h) = {
  let l = tcLane(a);
  let c = tcCol(a);
  let g = tdiv_int(t, 4);
  if has(m, "32x32b") then tcKey(l + t, c + i)
  else if has(m, "16x64b") then tcKey(l + g + 8 * fmod_int(t, 2), c + 2 * i + fmod_int(tdiv_int(t, 2), 2))
  else if has(m, "16x128b") then tcKey(l + g + 8 * fmod_int(i, 2), c + 4 * tdiv_int(i, 2) + fmod_int(t, 4))
  else if has(m, "16x256b") then tcKey(l + g + 8 * tdiv_int(fmod_int(i, 4), 2), c + 8 * tdiv_int(i, 4) + 2 * fmod_int(t, 4) + fmod_int(i, 2))
  else if has(m, "16x32bx2") then tcKey(l + fmod_int(t, 16), c + i + tdiv_int(t, 16) * h)
  else -1
}

/* .pack::16b / .unpack::16b (ISA 9.7.17.8.2, Figure 193): the 32-bit cell of register i becomes two adjacent 16-bit
   columns, so its column (relative to taddr) doubles; for .32x32b and .16x32bx2 (whose split offset is a column offset)
   those are the cells of elements 2i and 2i + 1 */
val tcPkCell : (list(string), int, int, int, int, int) -> int
function tcPkCell(m, a, t, i, h, half) =
  if has(m, "16x64b") | has(m, "16x128b") | has(m, "16x256b") then {
    let k = tcCell(m, a, t, i, h);
    tcKey(tdiv_int(k, 65536), tcCol(a) + 2 * (fmod_int(k, 65536) - tcCol(a)) + half)
  } else tcCell(m, a, t, 2 * i + half, h)

val tcLoad : (int, list(string), int, int, int, int, bool) -> list(int)
function tcLoad(tm, m, a, t, h, n, pk) = {
  var out : list(int) = [||];
  foreach (i from 0 to (n - 1))
    out = pend_append(out, [|if pk then lor_int(land_int(tcWord(tm, tcPkCell(m, a, t, i, h, 0)), 65535), shl_int(land_int(tcWord(tm, tcPkCell(m, a, t, i, h, 1)), 65535), 16))
                            else tcWord(tm, tcCell(m, a, t, i, h))|]);
  out
}

val tcStore : (int, list(string), int, int, int, list(int), bool) -> unit
function tcStore(tm, m, a, t, h, vs, up) = {
  var i : int = 0;
  var x = vs;
  while is_cons(x) do {
    match x { v :: rest => {
      if up then { tcPut(tm, tcPkCell(m, a, t, i, h, 0), land_int(v, 65535)); tcPut(tm, tcPkCell(m, a, t, i, h, 1), land_int(asr_int(v, 16), 65535)) }
      else tcPut(tm, tcCell(m, a, t, i, h), v);
      i = i + 1;
      x = rest
    }, [||] => () }
  }
}

val tcSg : int -> int
function tcSg(a) = if a >= 2147483648 then a - 4294967296 else a

val tyIn : (list(ty), ty) -> bool
function tyIn(ts, t) = match ts { [||] => false, h :: r => ty_eq(h, t) | tyIn(r, t) }

val tcRed : (list(string), list(ty), list(int)) -> int
function tcRed(m, ts, vs) = match vs {
  [||] => 0,
  a0 :: rest => {
    let ab = has(m, "abs");
    let mn = has(m, "min");
    var acc : int = if ab then land_int(a0, 2147483647) else a0;
    var x = rest;
    while is_cons(x) do {
      match x { a1 :: r => {
        let a = if ab then land_int(a1, 2147483647) else a1;
        if tyIn(ts, mkty("f", 32)) then acc = fpMinMax(false, has(m, "NaN"), false, 32, acc, a, mn)
        else if tyIn(ts, mkty("u", 32)) then acc = (if mn then (if a1 < acc then a1 else acc) else (if a1 > acc then a1 else acc))
        else if tyIn(ts, mkty("s", 32)) then acc = (if mn then (if tcSg(a1) < tcSg(acc) then a1 else acc) else (if tcSg(a1) > tcSg(acc) then a1 else acc));
        x = r
      }, [||] => () }
    };
    acc
  }
}

/* ---------------------------------------------------------------- tcgen05.mma */
val tcSwz : int -> int
function tcSwz(c) = match c { 2 => 1, 4 => 2, 6 => 3, 1 => 4, _ => 0 }
val tcDesc : int -> int
function tcDesc(d) =
  lor_int(lor_int(lor_int(land_int(d, 16383), shl_int(land_int(asr_int(d, 16), 16383), 16)), shl_int(dBase(d), 49)),
          lor_int(shl_int(land_int(asr_int(d, 32), 16383), 32),
                  lor_int(shl_int(land_int(tcSwz(land_int(asr_int(d, 61), 7)), 3), 62), shl_int(tdiv_int(tcSwz(land_int(asr_int(d, 61), 7)), 4), 56))))

/* bit 52 of the shared memory descriptor (ISA 9.7.17.3.1, Table 45): 1 = the leading dimension field is an absolute
   byte address of a second data chunk (9.7.17.3.1.2, K dimension of 48 bytes).  The ISA does not say which part of a
   row the second chunk holds, so the model stops with an error instead of reading the field as a relative offset */
val tcAbsBad : (bool, int, int) -> bool
function tcAbsBad(ad, a, b) = {
  let bad = (ad & land_int(asr_int(a, 52), 1) == 1) | land_int(asr_int(b, 52), 1) == 1;
  if bad then fault("tcgen05: the absolute leading-dimension address mode (descriptor bit 52) is not modelled", 0);
  bad
}

val isMxK : list(string) -> bool
function isMxK(m) = has(m, "kind::mxf8f6f4") | has(m, "kind::mxf4") | has(m, "kind::mxf4nvf4")

val tcDT : (list(string), int) -> ty
function tcDT(m, i) =
  if isMxK(m) then mkty("f", 32) else if has(m, "kind::i8") | has(m, "kind::ti16") then mkty("s", 32)
  else if land_int(asr_int(i, 4), 3) == 0 then mkty("f", 16) else mkty("f", 32)

val tcF8 : int -> ty
function tcF8(c) = match c { 0 => mkty("e4m3", 8), 1 => mkty("e5m2", 8), 3 => mkty("e2m3", 8), 4 => mkty("e3m2", 8), 5 => mkty("e2m1", 8), _ => mkty("e4m3", 8) }

val tcAT : (list(string), int) -> ty
function tcAT(m, i) =
  if has(m, "kind::mxf4") | has(m, "kind::mxf4nvf4") then mkty("e2m1p", 4)
  else if has(m, "kind::tf32") then mkty("tf", 32)
  else if has(m, "kind::i8") then (if land_int(asr_int(i, 7), 7) == 0 then mkty("u", 8) else mkty("s", 8))
  /* "In case of .kind::ti16 qualifier, the atype and btype are of type .s1z4m11 which is a 16-bit signed integer with the
     range [-2^11 + 1, 2^11 - 1]" (PTX ISA 9.4 9.7.18.4.2, Table 53): held as .s16 (two's complement, see tcTi16) */
  else if has(m, "kind::ti16") then mkty("s", 16)
  else if has(m, "kind::f8f6f4") | has(m, "kind::mxf8f6f4") then tcF8(land_int(asr_int(i, 7), 7))
  else if land_int(asr_int(i, 7), 7) == 0 then mkty("f", 16) else mkty("bf", 16)
val tcBT : (list(string), int) -> ty
function tcBT(m, i) = tcAT(m, asr_int(i, 3))

val tcMD : (list(string), int) -> int
function tcMD(m, i) = if isMxK(m) then land_int(asr_int(i, 27), 3) * 128 else land_int(asr_int(i, 24), 31) * 16
val tcN : int -> int
function tcN(i) = land_int(asr_int(i, 17), 63) * 8

val tcK : (list(string), int) -> int
function tcK(m, i) =
  if has(m, "kind::mxf8f6f4") then (if land_int(asr_int(i, 31), 1) == 1 then 64 else 32)
  else if has(m, "kind::mxf4") | has(m, "kind::mxf4nvf4") then (if land_int(asr_int(i, 3), 1) == 1 then 128 else if land_int(asr_int(i, 31), 1) == 1 then 96 else 64)
  else if has(m, "kind::tf32") then 8
  else if has(m, "kind::i8") then 32
  else if has(m, "kind::ti16") then 16
  else if has(m, "kind::f8f6f4") then (if land_int(asr_int(i, 29), 1) == 1 then 64 else 32)
  else 16

val tcKS : (list(string), int, int) -> int
function tcKS(m, i, meta) = if meta >= 0 then 2 * tcK(m, i) else tcK(m, i)

/* the shared-memory element (mn, k) of an A / B tile (space sp); FP6 / FP4 K-major tiles are bit-packed */
val wgElemS : (int, int, bool, int, int, int) -> int
function wgElemS(sp, d, km, w, mn, k) = {
  let bo = wgBitOff(d, km, w, mn, k);
  let a = swizzleB(dSwz(d), dStart(d) + tdiv_int(bo, 8), dBase(d));
  land_int(asr_int(mem_rdn(sp, a, max_int(1, tdiv_int(w, 8))), fmod_int(bo, 8)), mask(w))
}

val tcSub : ty -> bool
function tcSub(t) = t.k == "e2m1" | t.k == "e2m3" | t.k == "e3m2"
val tcNb : ty -> int
function tcNb(t) = if t.k == "e2m1" then 4 else 6

val tcSE : (int, int, bool, ty, int, int, int) -> int
function tcSE(sp, d, km, t, r, kk, k) = {
  if km & tcSub(t) then {
    let bp = if k == 32 then 128 * tdiv_int(kk, 16) + fmod_int(kk, 16) * tcNb(t) else kk * tcNb(t);
    let lo = wgElemS(sp, tcDesc(d), true, 8, r, tdiv_int(bp, 8));
    let hi = if fmod_int(bp, 8) + tcNb(t) > 8 then shl_int(wgElemS(sp, tcDesc(d), true, 8, r, tdiv_int(bp, 8) + 1), 8) else 0;
    shl_int(land_int(asr_int(lor_int(lo, hi), fmod_int(bp, 8)), mask(tcNb(t))), if tcNb(t) == 4 then 2 else 0)
  } else wgElemS(sp, tcDesc(d), km, ew(t), r, kk)
}

/* .kind::ti16 elements (.s16 here): the ISA gives the range of .s1z4m11, [-2^11 + 1, 2^11 - 1] (PTX ISA 9.4
   9.7.18.4.2), but of its encoding only "a 16-bit signed integer"; the model reads two's complement .s16 (the name
   suggests sign / 4 zero bits / 11-bit magnitude, which agrees on the non-negative values only).  An element outside
   the range is not an .s1z4m11 value: reported under .ub strict */
val tcTi16 : (ty, int) -> int
function tcTi16(t, v) = {
  if ub & t.k == "s" & t.w == 16 & not_bool(stuck) & (sext(v, 16) < -2047 | sext(v, 16) > 2047) then
    fault("tcgen05.mma.kind::ti16: an A / B element outside the .s1z4m11 range [-2^11 + 1, 2^11 - 1] [PTX ISA 9.4 9.7.18.4.2]", sext(v, 16));
  v
}

/* negation (idesc bits 13 / 14): the sign of a float element; .kind::ti16 negates the integer (Table 53 "Negate = 1";
   Table 64 "Is Negate A/B supported": ti16 Yes, i8 No, so the bits stay ignored for .kind::i8) */
val tcNeg : (ty, int, int) -> int
function tcNeg(t, sg, v) = if t.k == "s" & t.w == 16 then (if sg < 0 then trunc(negate(sext(v, 16)), 16) else v) else wgNeg(t, sg, v)

/* A (M x K) into SP_MA: from shared memory (descriptor) or tensor memory */
val tcTileA : (bool, int, int, int, ty, int, bool, int, int) -> unit
function tcTileA(desc, v, tm, sp, t, sg, km, m, k) = {
  smap_clear(SP_MA);
  foreach (r from 0 to (m - 1))
    foreach (kk from 0 to (k - 1))
      mset(SP_MA, rc(r, kk), tcNeg(t, sg, tcTi16(t,
        if desc then tcSE(sp, v, km, t, r, kk, k)
        else land_int(asr_int(tcWord(tm, tcKey(r, tcCol(v) + tdiv_int(kk * ew(t), 32))), fmod_int(kk * ew(t), 32)), mask(ew(t))))))
}

/* zero-column mask descriptor (.ws) */
val tcRun : (int, int, int, int, int) -> bool
function tcRun(v0, sc, on, zr, p) =
  (v0 == 1 & fmod_int(p + sc, on + zr) < on) | (v0 == 0 & fmod_int(p + sc, on + zr) >= zr)
val tcZBit : (int, int, int) -> bool
function tcZBit(z, i, p) =
  land_int(asr_int(z, 39), 1) == 1 &
  tcRun(land_int(asr_int(z, 32 + i), 1), land_int(asr_int(z, 8 * i), 255), land_int(asr_int(z, 40), 255) + 1, land_int(asr_int(z, 48), 255) + 1, p)
val tcZero : (int, int, int, int) -> bool
function tcZero(z, md, n, c) =
  if z >= 0 & md > 0 & md <= 128 & tdiv_int(n, tdiv_int(128, md)) > 0 then {
    let w = tdiv_int(n, tdiv_int(128, md));
    tcZBit(z, tdiv_int(c, w), fmod_int(c, w))
  } else false

/* .decompress::lut::b (PTX ISA 9.4 9.7.18.10.8, 9.7.18.4.1, 9.7.18.10.10.1): B element (n, k) is the 8-bit entry of a
   lookup table selected by a 3-bit index.  "Data loaded from shared memory contains a 3-bit index for each element";
   the instruction "reads 48 bytes of B matrix from shared memory" (one K = 128 block of indices of row n, read as bytes
   0..47 of a K-major row of 8-bit elements of b-compressed-desc) and needs 24 (3 * 64 / 8): the first or the second
   24-byte chunk as descriptor bit 53 says ("K_Segment start offset in K=128 (48B) block - 0: 0 - 1: 24").  The indices
   are packed least significant bit first, as the model's other bit-packed elements (the ISA does not give the order).
   "Every element in the same 8-column-aligned 8-column slice of the B matrix shares the same set of 8 possible values":
   the table of N slice n / 8 is TMEM row n / 8 (Figures 279, 282), 8 bytes from the two-column-aligned column lc of
   b-decompress-metadata, entry j in column lc + j / 4, byte j % 4 */
val tcLutE : (int, int, int, int, int, int) -> int
function tcLutE(tm, sp, d, lc, n, kk) = {
  let bp = 192 * land_int(asr_int(d, 53), 1) + 3 * kk;
  let lo = wgElemS(sp, tcDesc(d), true, 8, n, tdiv_int(bp, 8));
  let hi = if fmod_int(bp, 8) + 3 > 8 then shl_int(wgElemS(sp, tcDesc(d), true, 8, n, tdiv_int(bp, 8) + 1), 8) else 0;
  let j = land_int(asr_int(lor_int(lo, hi), fmod_int(bp, 8)), 7);
  land_int(asr_int(tcWord(tm, tcKey(tdiv_int(n, 8), lc + tdiv_int(j, 4))), 8 * fmod_int(j, 4)), 255)
}

/* B (K x N) into SP_MB at column offset co; lut >= 0: the lookup-table column of .decompress::lut::b */
val tcTileB : (int, int, int, ty, int, bool, int, int, int, int, int, int, int) -> unit
function tcTileB(tm, v, sp, t, sg, km, n, k, sh, z, md, co, lut) =
  foreach (nn from 0 to (n - 1))
    foreach (kk from 0 to (k - 1))
      mset(SP_MB, rc(kk, nn + co),
           if tcZero(z, md, n, nn) then 0
           else tcNeg(t, sg, if lut >= 0 then tcLutE(tm, sp, v, lut, nn + sh, kk) else tcTi16(t, tcSE(sp, v, km, t, nn + sh, kk, k))))

/* the D accumulator input, scaled by 2^-s, into SP_MC */
val tcTileD : (int, ty, int, int, int, int) -> unit
function tcTileD(tm, t, base, m, n, s) = {
  smap_clear(SP_MC);
  foreach (r from 0 to (m - 1))
    foreach (c from 0 to (n - 1)) {
      let b = land_int(tcWord(tm, tcKey(r, base + c)), mask(ew(t)));
      mset(SP_MC, rc(r, c),
           if s > 0 & isFloatElem(t) then fin(noMods(), fw(t), match elemF(t, b) { FV_Fin(sg, q, e) => FV_Fin(sg, q, e - s), f => f }, false) else b)
    }
}

/* write the result back, honouring the disable-output-lane mask */
val tcPutD : (int, int, int, int, int) -> unit
function tcPutD(tm, base, mk, m, n) =
  foreach (r from 0 to (m - 1))
    if not_bool(bit_int(mk, r)) then
      foreach (c from 0 to (n - 1)) tcPut(tm, tcKey(r, base + c), mget(SP_MD, rc(r, c)))

/* sparse A: expand the packed matrix in SP_MA with the metadata in tensor memory.  Eight chunks per row
   (.kind::f16, .kind::tf32; Figures 262-265): column mc + sel; in each block of 16 rows, lane 8 (c / 4) + r % 8
   holds chunks 4 (c / 4) .. + 3 of rows r % 8 (nibbles 0-3) and r % 8 + 8 (nibbles 4-7).  Sixteen chunks
   (Figures 266, 267): lane r, columns mc, mc + 1.  The 16-row blocks are at lanes 16 (r / 16) as the model's A and
   D rows (it does not model the half-lane data-path layouts of M = 64) */
val tcNib : (int, ty, int, int, int, int, int) -> int
function tcNib(tm, t, r, mc, sel, c, kp) =
  if tdiv_int(2 * kp, spCw(t)) <= 8 then
    land_int(asr_int(tcWord(tm, tcKey(16 * tdiv_int(r, 16) + 8 * tdiv_int(c, 4) + fmod_int(r, 8), mc + sel)),
                     16 * tdiv_int(fmod_int(r, 16), 8) + 4 * fmod_int(c, 4)), 15)
  else land_int(asr_int(tcWord(tm, tcKey(r, mc + tdiv_int(c, 8))), 4 * fmod_int(c, 8)), 15)

let SP_MT : int = 26

/* "The only meaningful values of the index are" (9.7.17.10.8): .kind::tf32 0b1110 0b0100; the 2:4 and pair-wise 4:8
   kinds 0b0100 0b1000 0b1100 0b1001 0b1101 0b0110 0b1110 ("Rest of the values result in undefined behavior"); checked
   under .ub strict */
val tcNibOk : (ty, int) -> bool
function tcNibOk(t, nib) =
  if t.k == "tf" then nib == 14 | nib == 4 else has([|"4", "8", "12", "9", "13", "6", "14"|], dec_str(nib))

val tcSpExp : (int, ty, int, int, int, int) -> unit
function tcSpExp(tm, t, m, kp, mc, sel) = {
  if ub then foreach (r from 0 to (m - 1)) foreach (c from 0 to (tdiv_int(kp, spNz(t)) - 1)) {
    let nib = tcNib(tm, t, r, mc, sel, c, kp);
    if not_bool(stuck) & not_bool(tcNibOk(t, nib)) then fault("tcgen05.mma.sp: metadata index value that is not meaningful", nib)
  };
  smap_clear(SP_MT);
  smap_copy(SP_MT, SP_MA);
  smap_clear(SP_MA);
  foreach (r from 0 to (m - 1))
    foreach (k from 0 to (kp - 1))
      mset(SP_MA, rc(r, spCw(t) * tdiv_int(k, spNz(t)) + spPos(t, tcNib(tm, t, r, mc, sel, tdiv_int(k, spNz(t)), kp), fmod_int(k, spNz(t)))),
           mget(SP_MT, rc(r, k)))
}

/* .ashift: the rows of A move down one row (row 0 is kept); every key of row r moves to row r + 1, so a sparse A whose
   rows have different non-zero positions shifts correctly (the extra row M is never read) */
val tcAshift : unit -> unit
function tcAshift() = {
  smap_clear(SP_MT);
  smap_copy(SP_MT, SP_MA);
  smap_clear(SP_MA);
  foreach (i from 0 to (smap_size(SP_MT) - 1)) {
    let ks = smap_key(SP_MT, i);
    let k = str_int(ks);
    if k < 1024 then mset(SP_MA, k, mget(SP_MT, k));
    mset(SP_MA, k + 1024, mget(SP_MT, k))
  }
}

val tcAMat : (list(string), int, int, bool, int, int, int) -> unit
function tcAMat(m, tm, sp, ad, av, id, meta) = {
  let at_ = tcAT(m, id);
  tcTileA(ad, av, tm, sp, at_, if land_int(asr_int(id, 13), 1) == 1 then -1 else 1, land_int(asr_int(id, 15), 1) == 0, tcMD(m, id), tcK(m, id));
  if meta >= 0 then tcSpExp(tm, at_, tcMD(m, id), tcK(m, id), meta, land_int(id, 3));
  if has(m, "ashift") then tcAshift()
}

val tcShiftZ : int -> int
function tcShiftZ(z) = if z >= 0 then land_int(asr_int(z, 56), 63) else 0

val tcSgB : int -> int
function tcSgB(id) = if land_int(asr_int(id, 14), 1) == 1 then -1 else 1
val tcKmB : int -> bool
function tcKmB(id) = land_int(asr_int(id, 16), 1) == 0

/* tcMmaB: A, C, the product and the write-back, B being in SP_MB */
val tcMmaB : (list(string), int, int, bool, int, int, int, bool, int, int, int) -> unit
function tcMmaB(m, tm, sp, ad, av, id, da, en, s, mk, meta) = {
  let md = tcMD(m, id);
  let n = tcN(id);
  let ks = tcKS(m, id, meta);
  let shape = concat_str("m", concat_str(dec_str(md), concat_str("n", concat_str(dec_str(n), concat_str("k", dec_str(ks))))));
  let dt = tcDT(m, id);
  tcAMat(m, tm, sp, ad, av, id, meta);
  if en then tcTileD(tm, dt, tcCol(da), md, n, s) else smap_clear(SP_MC);
  resRows(shape, dt, tcAT(m, id), tcBT(m, id), dt, if land_int(asr_int(id, 3), 1) == 1 then [|"satfinite"|] else [||]);
  tcPutD(tm, tcCol(da), mk, md, n)
}

val tcHalfM : int -> int
function tcHalfM(id) = lor_int(land_int(id, lnot_int(shl_int(31, 24))), shl_int(tdiv_int(land_int(asr_int(id, 24), 31), 2), 24))

/* operands after idesc: optional vector mask, enable predicate, optional scale / zero-column mask */
val tcOps : list(xop) -> list(xop)
function tcOps(l) = match l { x :: rest => if isVec(x) then rest else l, [||] => [||] }
val tcMask : list(int) -> int
function tcMask(vs) = { var r : int = 0; var i : int = 0; var x = vs;
  while is_cons(x) do { match x { v :: rest => { r = lor_int(r, shl_int(v, 32 * i)); i = i + 1; x = rest }, [||] => () } }; r }
val tcMk : list(xop) -> int
function tcMk(l) = match l { x :: _ => if isVec(x) then tcMask(vecRegs(x, mkty("b", 32))) else 0, [||] => 0 }
val tcEn : list(xop) -> bool
function tcEn(l) = match tcOps(l) { x :: _ => rdu32(x) != 0, [||] => true }
val tcZm : (list(string), list(xop)) -> int
function tcZm(m, l) = match tcOps(l) { _ :: x :: _ => if has(m, "ws") then rd(x, mkty("b", 64)) else -1, _ => -1 }
val tcSc : list(xop) -> int
function tcSc(l) = match tcOps(l) { _ :: x :: _ => rdu32(x), _ => 0 }

/* the peer CTA of a cta_group::2 pair */
val tcPeer : unit -> option(int)
function tcPeer() = {
  let c = cur_cta();
  let pt = clTarget(c, lxor_int(rreg("%cluster_ctarank"), 1), gridDim, clusterDim);
  if pt != c & 0 <= pt & pt < ncta_slots & c_alive[cchk(pt)] then Some(pt) else None()
}

/* .kind::ti16 instruction descriptor (PTX ISA 9.4 9.7.18.4.2, Table 53): dtype S32 = 2, atype / btype S1Z4M11 = 3,
   bit 3 (Saturate) "0 (NA)", bit 29 (K Dimension) "NA" (K = 16 dense / 32 sparse is unique, so "the instruction
   descriptor bit corresponding to K must be set to 0", 9.7.18.2.2): other values reported under .isacheck.  The default
   mode reads the elements as .s1z4m11 and the result as .s32 whatever the type fields say, and gives bit 3 its integer
   meaning (saturation, as .kind::i8).  Transpose (bits 15 / 16) is "No Transpose = 0" in Table 53 but "Yes" in Table 64:
   the bits are honoured and not reported */
val tcTi16Idesc : (list(string), int) -> bool
function tcTi16Idesc(m, id) =
  has(m, "kind::ti16") & (land_int(asr_int(id, 4), 3) != 2 | land_int(asr_int(id, 7), 7) != 3 | land_int(asr_int(id, 10), 7) != 3
                          | land_int(asr_int(id, 3), 1) != 0 | land_int(asr_int(id, 29), 1) != 0)

/* .decompress::lut::b (PTX ISA 9.4 9.7.18.10.10.1): "matrix B doesn't support transpose and the only type allowed is
   E4M3.  Layout B (M = 128 + cta_group::2 + Dense A matrix), Layout C (M = 128 + cta_group::2 + Sparse A matrix) and
   Layout F (M = 64 + non .ws mode) are not allowed for storing the lookup tables in Tensor Memory" (btype and the
   layouts under .isacheck; the default mode reads the table entries as btype); b-decompress-metadata "is the two column
   aligned address" (under .ub strict).  A transposed index matrix, a K other than 64 ("needs only 24 bytes(3 * 64 /
   8)"), a kind other than f8f6f4 / mxf8f6f4 and the CTA pair (which CTA's table holds which N slices) have no meaning
   in the ISA: errors.  Returns false when the instruction cannot run */
val tcLutOk : (list(string), int, int) -> bool
function tcLutOk(m, id, lut) = {
  if lut < 0 then return true;
  let md = tcMD(m, id);
  if isachk & land_int(asr_int(id, 10), 7) != 0 then
    fault("tcgen05.mma.decompress::lut::b: the only B type allowed is E4M3 [PTX ISA 9.4 9.7.18.10.10.1]", land_int(asr_int(id, 10), 7))
  else if isachk & ((not_bool(has(m, "cta_group::2")) & md == 64) | (has(m, "cta_group::2") & md == 128)) then
    fault("tcgen05.mma.decompress::lut::b: data path Layout B, C or F (M = 64, or M = 128 with .cta_group::2) is not allowed for the lookup tables [PTX ISA 9.4 9.7.18.10.10.1]", md)
  else if ub & fmod_int(lut, 2) != 0 then
    fault("tcgen05.mma.decompress::lut::b: b-decompress-metadata is not two-column aligned [PTX ISA 9.4 9.7.18.10.10.1]", lut)
  else if not_bool(has(m, "kind::f8f6f4") | has(m, "kind::mxf8f6f4")) then
    fault("tcgen05.mma.decompress::lut::b applies to .kind::f8f6f4 and .kind::mxf8f6f4 only", 0)
  else if not_bool(tcKmB(id)) then
    fault("tcgen05.mma.decompress::lut::b: matrix B doesn't support transpose (no layout of a transposed index matrix)", 0)
  else if tcK(m, id) != 64 then
    fault("tcgen05.mma.decompress::lut::b: the index layout is given for K = 64 only", tcK(m, id))
  else if has(m, "cta_group::2") then
    fault("tcgen05.mma.cta_group::2 with .decompress::lut::b is not modelled", 0);
  not_bool(stuck)
}

val tc_mma : (list(string), bool, int, int, int, int, int, int, list(xop)) -> unit
function tc_mma(m, ad, av, bv, id, da, meta, lut, rest) = {
  let c = cur_cta();
  /* "The valid range of values for argument scale-input-d is [0, 15]. The argument scale-input-d is only valid for
     .kind::tf32 and .kind::f16" (9.7.17.10.9.1): reported under .isacheck */
  let nsc = list_length(tcOps(rest));
  if isachk & nsc >= 2 & not_bool(has(m, "ws")) & (tcSc(rest) < 0 | tcSc(rest) > 15) then
    fault("tcgen05.mma: scale-input-d outside [0, 15] [PTX ISA 9.4 9.7.18.10.10.1]", tcSc(rest))
  else if isachk & nsc >= 2 & not_bool(has(m, "ws")) & not_bool(has(m, "kind::tf32") | has(m, "kind::f16")) then
    fault("tcgen05.mma: scale-input-d is only valid for .kind::tf32 and .kind::f16 [PTX ISA 9.4 9.7.18.10.10.1]", 0)
  else if isachk & tcTi16Idesc(m, id) then
    fault("tcgen05.mma.kind::ti16: the instruction descriptor needs dtype S32 = 2, atype / btype S1Z4M11 = 3, Saturate 0 (NA) and K Dimension 0 (NA) [PTX ISA 9.4 9.7.18.4.2]", id);
  if stuck | not_bool(tcLutOk(m, id, lut)) then return ();
  if not_bool(has(m, "cta_group::2")) then {
    let tm = tcSpace(c, "tm");
    let sp = c_smem[cchk(c)];
    let z = tcZm(m, rest);
    smap_clear(SP_MB);
    tcTileB(tm, bv, sp, tcBT(m, id), tcSgB(id), tcKmB(id), tcN(id), tcKS(m, id, meta), tcShiftZ(z), z, tcMD(m, id), 0, lut);
    tcMmaB(m, tm, sp, ad, av, id, da, tcEn(rest), if has(m, "ws") then 0 else tcSc(rest), tcMk(rest), meta)
  } else match tcPeer() {
    Some(pt) => {
      let even = fmod_int(rreg("%cluster_ctarank"), 2) == 0;
      let se = if even then c_smem[cchk(c)] else c_smem[cchk(pt)];
      let so = if even then c_smem[cchk(pt)] else c_smem[cchk(c)];
      let hid = tcHalfM(id);
      let n2 = land_int(asr_int(id, 17), 63) * 4;
      let en = tcEn(rest);
      let s = tcSc(rest);
      /* the 8-register disable-output-lane vector of a CTA pair: which lanes of which CTA it names is not modelled */
      if tcMk(rest) != 0 then fault("tcgen05.mma.cta_group::2 with disable-output-lane is not modelled", 0) else
      foreach (j from 0 to 1) {
        let cc = if j == 0 then c else pt;
        smap_clear(SP_MB);
        tcTileB(0, bv, se, tcBT(m, id), tcSgB(id), tcKmB(id), n2, tcKS(m, hid, meta), 0, -1, tcMD(m, hid), 0, -1);
        tcTileB(0, bv, so, tcBT(m, id), tcSgB(id), tcKmB(id), n2, tcKS(m, hid, meta), 0, -1, tcMD(m, hid), n2, -1);
        tcMmaB(m, tcSpace(cc, "tm"), c_smem[cchk(cc)], ad, av, hid, da, en, s, 0, meta)
      }
    },
    None() => fault("tcgen05.mma.cta_group::2 without a peer CTA", 0)
  }
}

/* ---------------------------------------------------------------- block-scaled mma */
val tcBlk : (list(string), int) -> int
function tcBlk(m, k) =
  if has(m, "kind::mxf8f6f4") then k else if has(m, "block16") then 16 else if has(m, "scale_vec::4X") then tdiv_int(k, 4)
  else if has(m, "scale_vec::2X") then tdiv_int(k, 2) else 32

val tcUE : (int, int, int) -> fv
function tcUE(x, mn, bi) =
  if asr_int(x, mn) == 0 then FV_Fin(false, land_int(x, mask(mn)), 1 - bi - mn)
  else FV_Fin(false, land_int(x, mask(mn)) + pow2(mn), asr_int(x, mn) - bi - mn)
val tcScaleF : (int, int) -> fv
/* the scale matrix type (PTX ISA 9.4 9.7.18.4.2, Tables 54, 55): UE8M0 = 1 (NaN 0xff), UE4M3 = 0 (MSB padded, NaN 0x7f),
   UE5M3 = 2 ("unsigned floating-point format with 5 bits for exponent and 3 bits for mantissa ... does not support
   infinity.  NaN value is limited to 0xff", 5.2.3; bias 15 as the model's cvt .ue5m3x2) */
function tcScaleF(st, x) = match st {
  1 => if x == 255 then FV_NaN() else FV_Fin(false, 1, x - 127),
  2 => miniDecF("ue5m3", x),
  _ => if land_int(x, 127) == 127 then FV_NaN() else tcUE(land_int(x, 127), 3, 7)
}
/* bits 23-24 for .kind::mxf4 / .kind::mxf4nvf4 (Table 55), bit 23 for .kind::mxf8f6f4 (Table 54, whose only value is
   UE8M0 = 1; the model reads UE8M0 whatever the bit) */
val tcScaleTy : (list(string), int) -> int
function tcScaleTy(m, id) = if has(m, "kind::mxf4nvf4") | has(m, "kind::mxf4") then land_int(asr_int(id, 23), 3) else 1

/* the "sm_107f or higher in the same family" targets (sm_107f, sm_107a); no .target: not known */
val tc107f : unit -> bool
function tc107f() = isasm <= 0 | ((isaarch == "a" | isaarch == "f") & isasm >= 107 & isasm < 110)

/* scale type rules (PTX ISA 9.4 9.7.18.10.7.1, Table 71): .kind::mxf8f6f4 and .kind::mxf4 take UE8M0 only (under
   .isacheck; Table 55 also gives .kind::mxf4 the 2-bit field, whose UE4M3 / UE5M3 values the default mode honours);
   .kind::mxf4nvf4 takes UE8M0, UE4M3 or UE5M3 with every .scale_vectorsize; 3 is no scale type (error).  "Support for
   scale type UE5M3 requires target architecture sm_107f or higher in the same family, otherwise behavior is undefined";
   likewise "scale type UE4M3 with scale_vectorsize .block32" (both under .ub strict, against the .target) */
val tcScaleOk : (list(string), int) -> bool
function tcScaleOk(m, id) = {
  let st = tcScaleTy(m, id);
  if isachk & ((has(m, "kind::mxf8f6f4") & land_int(asr_int(id, 23), 1) != 1) | (has(m, "kind::mxf4") & st != 1)) then
    fault("tcgen05.mma: the scale type of .kind::mxf8f6f4 / .kind::mxf4 is UE8M0 [PTX ISA 9.4 9.7.18.10.7.1]", st)
  else if st == 3 then fault("tcgen05.mma.block_scale: scale type 3 in the instruction descriptor is not defined", id)
  else if ub & st == 2 & not_bool(tc107f()) then
    fault("tcgen05.mma.block_scale: scale type UE5M3 requires sm_107f or higher in the same family [PTX ISA 9.4 9.7.18.10.7.1]", isasm)
  else if ub & st == 0 & has(m, "block32") & not_bool(tc107f()) then
    fault("tcgen05.mma.block_scale: scale type UE4M3 with .block32 requires sm_107f or higher in the same family [PTX ISA 9.4 9.7.18.10.7.1]", isasm);
  not_bool(stuck)
}

/* scale factor byte b (= SF ID + j) of row / column x of an M x SF or SF x N scale matrix at column x0
   (ISA 9.7.17.10.7, Figures 231-258): lane x % 32, column x0 + x / 32, byte b % 4; bytes 4.. of the K = 96 forms
   (3 or 6 factors) are in the next group of columns, 4 columns on (8 when N > 128) */
val tcSF : (int, int, int, int, int) -> int
function tcSF(tm, x0, b, x, dim) =
  land_int(asr_int(tcWord(tm, tcKey(fmod_int(x, 32), x0 + tdiv_int(b, 4) * (if dim > 128 then 8 else 4) + tdiv_int(x, 32))),
                   8 * fmod_int(b, 4)), 255)

val tc_mmabs : (list(string), bool, int, int, int, int, bool, int, int, int, int) -> unit
function tc_mmabs(m, ad, av, bv, id, da, en, sac, sbc, meta, lut) = {
  if not_bool(tcScaleOk(m, id)) | not_bool(tcLutOk(m, id, lut)) then return ();
  let c = cur_cta();
  let tm = tcSpace(c, "tm");
  let sp = c_smem[cchk(c)];
  let md = tcMD(m, id);
  let n = tcN(id);
  let ks = tcKS(m, id, meta);
  let blk = tcBlk(m, ks);
  let nb = tdiv_int(ks, blk);
  let at_ = tcAT(m, id);
  let bt = tcBT(m, id);
  let st = tcScaleTy(m, id);
  tcAMat(m, tm, sp, ad, av, id, meta);
  smap_clear(SP_MB);
  tcTileB(tm, bv, sp, bt, tcSgB(id), tcKmB(id), n, ks, 0, -1, md, 0, lut);
  if en then tcTileD(tm, mkty("f", 32), tcCol(da), md, n, 0) else smap_clear(SP_MC);
  smap_clear(SP_SA);
  smap_clear(SP_SB);
  let ida = land_int(asr_int(id, 29), 3);
  let idb = land_int(asr_int(id, 4), 3);
  foreach (r from 0 to (md - 1))
    foreach (j from 0 to (nb - 1)) mset(SP_SA, rc(r, j), tcSF(tm, sac, ida + j, r, md));
  foreach (nn from 0 to (n - 1))
    foreach (j from 0 to (nb - 1)) mset(SP_SB, rc(j, nn), tcSF(tm, sbc, idb + j, nn, n));
  smap_clear(SP_MD);
  foreach (r from 0 to (md - 1))
    foreach (cc from 0 to (n - 1)) {
      var acc : fv = elemF(mkty("f", 32), mget(SP_MC, rc(r, cc)));
      foreach (i from 0 to (ks - 1)) {
        let p = fv_mul_exact(fv_mul_exact(fv_mul_exact(elemF(at_, mget(SP_MA, rc(r, i))), elemF(bt, mget(SP_MB, rc(i, cc)))),
                                          tcScaleF(st, mget(SP_SA, rc(r, tdiv_int(i, blk))))),
                             tcScaleF(st, mget(SP_SB, rc(tdiv_int(i, blk), cc))));
        acc = fv_add_exact(acc, p, RN)
      };
      mset(SP_MD, rc(r, cc), fin(noMods(), 32, acc, false))
    };
  tcPutD(tm, tcCol(da), 0, md, n)
}

/* ---------------------------------------------------------------- commit / cp / shift */
val tcCpRows : list(string) -> int
function tcCpRows(m) =
  if has(m, "128x256b") | has(m, "128x128b") then 128 else if has(m, "4x256b") then 4 else if has(m, "64x128b") then 64
  else if has(m, "32x128b") then 32 else 0
val tcCpWords : list(string) -> int
function tcCpWords(m) = if has(m, "128x256b") | has(m, "4x256b") then 8 else 4

val tcCpDst : (list(string), int) -> list(int)
function tcCpDst(m, i) =
  if not_bool(has(m, "warpx2") | has(m, "warpx4")) then [|i|]
  else if has(m, "warpx4") then [|i, i + 32, i + 64, i + 96|]
  else if has(m, "warpx2::02_13") then [|fmod_int(i, 32) + 32 * (2 * tdiv_int(i, 32)), fmod_int(i, 32) + 32 * (2 * tdiv_int(i, 32) + 1)|]
  else if has(m, "warpx2::01_23") then [|fmod_int(i, 32) + 32 * tdiv_int(i, 32), fmod_int(i, 32) + 32 * (tdiv_int(i, 32) + 2)|]
  else [||]

val tcDecomp : (int, list(int)) -> list(int)
function tcDecomp(nb, ws) = {
  var out : list(int) = [||];
  var x = ws;
  while list_length(x) >= 4 do {
    let es = elemsOf(bitsOf([|at(x, 0), at(x, 1), at(x, 2)|]), nb, 16);
    var sh : list(int) = [||];
    var e = es;
    while is_cons(e) do { match e { v :: rest => { sh = pend_append(sh, [|shl_int(v, if nb == 4 then 2 else 0)|]); e = rest }, [||] => () } };
    out = pend_append(out, wordsOf(packElems(sh, 8), 4));
    x = ldrop(x, 4)
  };
  out
}

val tcCpWr : (int, int, int, list(int)) -> unit
function tcCpWr(tm, ln, cl, ws) = {
  var c : int = cl;
  var x = ws;
  while is_cons(x) do { match x { v :: rest => { tcPut(tm, tcKey(ln, c), v); c = c + 1; x = rest }, [||] => () } }
}

val tcCp : (int, int, list(string), int, int, int) -> unit
function tcCp(tm, sp, m, l, cl, d) =
  foreach (i from 0 to (tcCpRows(m) - 1)) {
    var row : list(int) = [||];
    foreach (w from 0 to (tcCpWords(m) - 1)) row = pend_append(row, [|wgElemS(sp, d, true, 32, i, w)|]);
    let ws = if has(m, "b6x16_p32") then tcDecomp(6, row) else if has(m, "b4x16_p64") then tcDecomp(4, row) else row;
    var ds = tcCpDst(m, i);
    while is_cons(ds) do { match ds { r :: rest => { tcCpWr(tm, l + r, cl, ws); ds = rest }, [||] => () } }
  }

val tcShift : (int, int, int) -> unit
function tcShift(tm, l, cl) =
  foreach (r from 30 downto 0) {
    var ws : list(int) = [||];
    foreach (w from 0 to 7) ws = pend_append(ws, [|tcWord(tm, tcKey(l + r, cl + w))|]);
    tcCpWr(tm, l + r + 1, cl, ws)
  }

val mbArriveKey : int -> unit
function mbArriveKey(k) = if mbHas(k) then mbPut(k, mbArrive(mbGet(k), 1))

/* ---------------------------------------------------------------- instructions */
/* "All the threads in the warp must specify the same value of taddr ... Otherwise, the behavior is undefined"
 * (tcgen05.ld 9.7.17.8.3, tcgen05.st 9.7.17.8.4), under .ub strict (warpUniform, ptx_coll.sail) */
val tcFirstAddr : list(xop) -> option(addr)
function tcFirstAddr(l) = match l { [||] => None(), XB(a :: _, _) :: _ => Some(a), _ :: rest => tcFirstAddr(rest) }
val tcUniform : (string, list(xop)) -> unit
function tcUniform(op, ops) = match tcFirstAddr(ops) { Some(a) => { let _ = warpUniform(op, "taddr", tcAddr(a)); () }, None() => () }

val tcCgOk : (int, int) -> bool
function tcCgOk(c, g) = {
  let mp = c_misc[cchk(c)];
  if smap_has(mp, "tccg") & smap_get(mp, "tccg") != g then {
    fault("tcgen05: instructions of the kernel specify different .cta_group values [PTX ISA 9.4 9.7.18.7.1]", g); false
  } else { smap_set(mp, "tccg", g); true }
}

/* collector buffer usage (PTX ISA 9.4 9.7.18.10, 9.7.18.10.10.1-3): .collector::{a,b,bN}::{fill,use,lastuse,discard}
   only give the TensorCore "permission to reuse a previously loaded A or B matrix; however reuse is opportunistic ...
   the source memory of an A or B matrix must not be modified while the MMA instruction using those matrices has not
   completed - regardless of collector qualifier permissions", so they have no effect on the result.  ::use and ::lastuse
   "require a previous fill to the collector buffer to be still valid": under .ub strict a fill makes buffer x valid,
   ::lastuse and ::discard (also the default ::discard of a missing qualifier: .collector::a / ::b for tcgen05.mma{.sp},
   .collector::b0 for .ws when no .collector qualifier is given) end it, and a ::use / ::lastuse without a valid fill is
   reported.  The ISA does not say whose buffer it is; the model keeps one per CTA (the issuing thread's) */
val tcCollOp : (list(string), string) -> string
function tcCollOp(m, x) = {
  let p = concat_str("collector::", concat_str(x, "::"));
  if has(m, concat_str(p, "fill")) then "fill" else if has(m, concat_str(p, "use")) then "use"
  else if has(m, concat_str(p, "lastuse")) then "lastuse" else if has(m, concat_str(p, "discard")) then "discard" else ""
}
val tcCollBuf : (int, string, string) -> unit
function tcCollBuf(mp, x, op) = {
  let key = concat_str("tccol:", x);
  if (op == "use" | op == "lastuse") & not_bool(smap_has(mp, key)) then
    fault(concat_str("tcgen05.mma: .collector::", concat_str(x, concat_str("::", concat_str(op,
          " without a valid previous fill of the collector buffer [PTX ISA 9.4 9.7.18.10.10.1]")))), 0);
  if op == "fill" then smap_set(mp, key, 1) else if (op == "lastuse" | op == "discard") & smap_has(mp, key) then smap_del(mp, key)
}
val tcColl : list(string) -> bool
function tcColl(m) = {
  if ub then {
    let mp = c_misc[cchk(cur_cta())];
    if has(m, "ws") then {
      let o0 = tcCollOp(m, "b0"); let o1 = tcCollOp(m, "b1"); let o2 = tcCollOp(m, "b2"); let o3 = tcCollOp(m, "b3");
      tcCollBuf(mp, "b0", if o0 == "" & o1 == "" & o2 == "" & o3 == "" then "discard" else o0);
      tcCollBuf(mp, "b1", o1); tcCollBuf(mp, "b2", o2); tcCollBuf(mp, "b3", o3)
    } else {
      let oa = tcCollOp(m, "a"); let ob = tcCollOp(m, "b");
      tcCollBuf(mp, "a", if oa == "" then "discard" else oa);
      tcCollBuf(mp, "b", if ob == "" then "discard" else ob)
    }
  };
  stuck
}

val exec_tc : (opcode, list(xop)) -> bool
function exec_tc(o, ops) = {
  let m = o.mods;
  let t = tchk(cur);
  let c = cur_cta();
  let tl = fmod_int(t_tid[t], 32);
  let full : int = 4294967295;
  if o.base != "tcgen05" then false
  /* "All tcgen05 instructions within a kernel must specify the same value for the .cta_group qualifier" (9.7.17.7.1):
     reported under .isacheck, per CTA ("tccg" in the CTA's misc map holds the first value seen) */
  else if isachk & (has(m, "cta_group::1") | has(m, "cta_group::2")) & not_bool(tcCgOk(c, if has(m, "cta_group::2") then 2 else 1)) then true
  else if has(m, "relinquish_alloc_permit") then { smap_set(c_misc[cchk(c)], "tcrelinq", 1); true }
  else if has(m, "wait") | has(m, "fence") then true
  else if has(m, "alloc") then match ops {
    [|XB([|d|], [||]), n|] if not_bool(warpUniform("tcgen05.alloc", "nCols", rdu32(n))) => true,
    /* "If the address specified by dst does not fall within the address window of .shared::cta state space then the
       behavior is undefined" (9.7.17.7.1) */
    [|XB([|d|], [||]), n|] if not_bool(winUb("tcgen05.alloc", "dst", ea(d, if has(m, "shared::cta") | has(m, "shared") then "shared" else "generic"), [|"shared"|])) => true,
    [|XB([|d|], [||]), n|] => {
      t_post[t] = P_tcalloc(ea(d, if has(m, "shared::cta") | has(m, "shared") then "shared" else "generic"), rdu32(n), has(m, "exclusive"));
      do_coll("tcgen05.alloc", full, [||], [|0|], [||]);
      true
    },
    _ => false
  } else if has(m, "dealloc") then match ops {
    [|ta, n|] => {
      t_post[t] = P_tcdealloc(tcCol(rd(ta, mkty("b", 32))), rdu32(n), has(m, "exclusive"));
      do_coll("tcgen05.dealloc", full, [||], [|0|], [||]);
      true
    },
    _ => false
  } else if has(m, "ld") then {
    tcUniform("tcgen05.ld", ops);
    let tm = tcSpace(c, "tm");
    let pk = has(m, "pack::16b");
    match ops {
      [|dv, XB([|a|], [||])|] => { collWrite(vecOps(dv), tcLoad(tm, m, tcAddr(a), tl, 0, xop_count(vecOps(dv)), pk)); true },
      [|dv, XB([|a|], [||]), off|] => { collWrite(vecOps(dv), tcLoad(tm, m, tcAddr(a), tl, rdu32(off), xop_count(vecOps(dv)), pk)); true },
      [|dv, rv, XB([|a|], [||])|] if has(m, "red") & not_bool(has(m, "spcompress")) => {
        let vs = tcLoad(tm, m, tcAddr(a), tl, 0, tcNum(m) * tcPer(m), pk);
        collWrite(vecOps(dv), vs); wr(rv, tcRed(m, o.tys, vs)); true
      },
      [|dv, rv, XB([|a|], [||]), off|] if has(m, "red") & not_bool(has(m, "spcompress")) => {
        let vs = tcLoad(tm, m, tcAddr(a), tl, rdu32(off), tcNum(m) * tcPer(m), pk);
        collWrite(vecOps(dv), vs); wr(rv, tcRed(m, o.tys, vs)); true
      },
      [|mdv, cdv, XB([|a|], [||])|] if has(m, "spcompress") & not_bool(has(m, "red")) => {
        let vs = tcLoad(tm, m, tcAddr(a), tl, 0, tcNum(m) * tcPer(m), pk);
        let (mw, cw_) = spcFin(spcKeep((if has(m, "min") then 2 else 0) + (if has(m, "abs") then 1 else 0), 6, 32, vs), 32, 2,
                              xop_count(vecOps(mdv)), xop_count(vecOps(cdv)));
        collWrite(vecOps(mdv), mw); collWrite(vecOps(cdv), cw_); true
      },
      [|mdv, cdv, rv, XB([|a|], [||])|] if has(m, "spcompress") & has(m, "red") => {
        let vs = tcLoad(tm, m, tcAddr(a), tl, 0, tcNum(m) * tcPer(m), pk);
        let (mw, cw_) = spcFin(spcKeep((if has(m, "min") then 2 else 0) + (if has(m, "abs") then 1 else 0), 6, 32, vs), 32, 2,
                              xop_count(vecOps(mdv)), xop_count(vecOps(cdv)));
        collWrite(vecOps(mdv), mw); collWrite(vecOps(cdv), cw_); wr(rv, tcRed(m, o.tys, vs)); true
      },
      _ => false
    }
  } else if has(m, "st") then {
    tcUniform("tcgen05.st", ops);
    let tm = tcSpace(c, "tm");
    match ops {
      [|XB([|a|], [||]), vs|] => { tcStore(tm, m, tcAddr(a), tl, 0, vecRegs(vs, mkty("b", 32)), has(m, "unpack::16b")); true },
      [|XB([|a|], [||]), off, vs|] => { tcStore(tm, m, tcAddr(a), tl, rdu32(off), vecRegs(vs, mkty("b", 32)), has(m, "unpack::16b")); true },
      _ => false
    }
  }
  /* .collector qualifiers: no effect on the result; the fill rules under .ub strict (tcColl) */
  else if has(m, "mma") & tcColl(m) then true
  else if has(m, "mma") & has(m, "block_scale") then match ops {
    [|XB([|d|], [||]), av, bd, id, XB([|sa|], [||]), XB([|sb|], [||]), en|] if not_bool(has(m, "sp")) => {
      let (ad, a) : (bool, int) = match av { XB([|x|], [||]) => (false, tcAddr(x)), _ => (true, rd(av, mkty("b", 64))) };
      if not_bool(tcAbsBad(ad, a, rd(bd, mkty("b", 64)))) then
        tc_mmabs(m, ad, a, rd(bd, mkty("b", 64)), rdb32(id), tcAddr(d), rdu32(en) != 0, tcCol(tcAddr(sa)), tcCol(tcAddr(sb)), -1, -1);
      true
    },
    /* .decompress::lut::b: [d-tmem], a, b-compressed-desc, [b-decompress-metadata], idesc, ... */
    [|XB([|d|], [||]), av, bd, XB([|lt|], [||]), id, XB([|sa|], [||]), XB([|sb|], [||]), en|] if has(m, "decompress::lut::b") & not_bool(has(m, "sp")) => {
      let (ad, a) : (bool, int) = match av { XB([|x|], [||]) => (false, tcAddr(x)), _ => (true, rd(av, mkty("b", 64))) };
      if not_bool(tcAbsBad(ad, a, rd(bd, mkty("b", 64)))) then
        tc_mmabs(m, ad, a, rd(bd, mkty("b", 64)), rdb32(id), tcAddr(d), rdu32(en) != 0, tcCol(tcAddr(sa)), tcCol(tcAddr(sb)), -1, tcCol(tcAddr(lt)));
      true
    },
    [|XB([|d|], [||]), av, bd, XB([|mt|], [||]), id, XB([|sa|], [||]), XB([|sb|], [||]), en|] if has(m, "sp") & not_bool(has(m, "decompress::lut::b")) => {
      let (ad, a) : (bool, int) = match av { XB([|x|], [||]) => (false, tcAddr(x)), _ => (true, rd(av, mkty("b", 64))) };
      if not_bool(tcAbsBad(ad, a, rd(bd, mkty("b", 64)))) then
        tc_mmabs(m, ad, a, rd(bd, mkty("b", 64)), rdb32(id), tcAddr(d), rdu32(en) != 0, tcCol(tcAddr(sa)), tcCol(tcAddr(sb)), tcCol(tcAddr(mt)), -1);
      true
    },
    _ => false
  } else if has(m, "mma") then match ops {
    XB([|d|], [||]) :: av :: bd :: XB([|mt|], [||]) :: id :: rest if has(m, "sp") & not_bool(has(m, "decompress::lut::b")) => {
      let (ad, a) : (bool, int) = match av { XB([|x|], [||]) => (false, tcAddr(x)), _ => (true, rd(av, mkty("b", 64))) };
      if not_bool(tcAbsBad(ad, a, rd(bd, mkty("b", 64)))) then tc_mma(m, ad, a, rd(bd, mkty("b", 64)), rdb32(id), tcAddr(d), tcCol(tcAddr(mt)), -1, rest);
      true
    },
    XB([|d|], [||]) :: av :: bd :: XB([|lt|], [||]) :: id :: rest if has(m, "decompress::lut::b") & not_bool(has(m, "sp")) => {
      let (ad, a) : (bool, int) = match av { XB([|x|], [||]) => (false, tcAddr(x)), _ => (true, rd(av, mkty("b", 64))) };
      if not_bool(tcAbsBad(ad, a, rd(bd, mkty("b", 64)))) then tc_mma(m, ad, a, rd(bd, mkty("b", 64)), rdb32(id), tcAddr(d), -1, tcCol(tcAddr(lt)), rest);
      true
    },
    XB([|d|], [||]) :: av :: bd :: id :: rest if not_bool(has(m, "sp")) & not_bool(has(m, "decompress::lut::b")) => {
      let (ad, a) : (bool, int) = match av { XB([|x|], [||]) => (false, tcAddr(x)), _ => (true, rd(av, mkty("b", 64))) };
      if not_bool(tcAbsBad(ad, a, rd(bd, mkty("b", 64)))) then tc_mma(m, ad, a, rd(bd, mkty("b", 64)), rdb32(id), tcAddr(d), -1, -1, rest);
      true
    },
    _ => false
  } else if has(m, "commit") then match ops {
    /* "If the address specified by mbar does not fall within the address window of .shared::cluster state space then
       the behavior is undefined" (9.7.17.12.1) */
    XB([|a|], [||]) :: _ if not_bool(winUb("tcgen05.commit", "mbar", ea(a, spaceOf(m)), [|"shared", "cluster"|])) => true,
    [|XB([|a|], [||])|] => { mbArriveKey(mbKey(ea(a, spaceOf(m)))); true },
    [|XB([|a|], [||]), mk|] if has(m, "multicast") => {
      var ks = mbMcKeys(land_int(locOffA(ea(a, spaceOf(m))), 16777215), mcMask(m, rdu32(mk)));
      while is_cons(ks) do { match ks { k :: rest => { mbArriveKey(k); ks = rest }, [||] => () } };
      true
    },
    _ => false
  } else if has(m, "cp") then match ops {
    [|XB([|ta|], [||]), dd|] => {
      let a = tcAddr(ta);
      let d = tcDesc(rd(dd, mkty("b", 64)));
      if tcAbsBad(false, 0, rd(dd, mkty("b", 64))) then true
      else if not_bool(has(m, "cta_group::2")) then { tcCp(tcSpace(c, "tm"), c_smem[cchk(c)], m, tcLane(a), tcCol(a), d); true }
      else match tcPeer() {
        Some(pt) => {
          tcCp(tcSpace(c, "tm"), c_smem[cchk(c)], m, tcLane(a), tcCol(a), d);
          tcCp(tcSpace(pt, "tm"), c_smem[cchk(pt)], m, tcLane(a), tcCol(a), d);
          true
        },
        None() => { fault("tcgen05.cp.cta_group::2 without a peer CTA", 0); true }
      }
    },
    _ => false
  } else if has(m, "shift") then match ops {
    [|XB([|ta|], [||])|] => {
      let a = tcAddr(ta);
      if not_bool(has(m, "cta_group::2")) then { tcShift(tcSpace(c, "tm"), tcLane(a), tcCol(a)); true }
      else match tcPeer() {
        Some(pt) => { tcShift(tcSpace(c, "tm"), tcLane(a), tcCol(a)); tcShift(tcSpace(pt, "tm"), tcLane(a), tcCol(a)); true },
        None() => { fault("tcgen05.shift.cta_group::2 without a peer CTA", 0); true }
      }
    },
    _ => false
  } else false
}

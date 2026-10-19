/* C runtime of the Sail models: the token reader of the drivers and the sparse byte memories. */
#include "sail.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static FILE *tok_in = NULL;
static char tokbuf[1 << 20];

static const char *next_tok(void) {
  if (!tok_in) {
    const char *p = getenv("SAIL_INPUT");
    tok_in = p ? fopen(p, "r") : stdin;
    if (!tok_in) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
  }
  if (fscanf(tok_in, "%1048575s", tokbuf) != 1) tokbuf[0] = 0;
  return tokbuf;
}

void sail_tok_str(sail_string *rop, const unit u) {
  const char *t = next_tok();
  size_t n = strlen(t);
  *rop = realloc(*rop, n + 1);
  memcpy(*rop, t, n + 1);
}

void sail_tok_int(sail_int *rop, const unit u) {
  const char *t = next_tok();
  if (!*t || mpz_set_str(*rop, t, 0) != 0) mpz_set_ui(*rop, 0);
}

/* sparse byte memories: (space, address) |-> byte, open addressing */
typedef struct { uint64_t space, addr; uint8_t val; uint8_t used; } cell;
static cell *tab = NULL;
static uint64_t cap = 0, cnt = 0;

static uint64_t hsh(uint64_t s, uint64_t a) {
  uint64_t h = a * 0x9E3779B97F4A7C15ULL ^ (s + 0x632BE59BD9B4E019ULL) * 0xC2B2AE3D27D4EB4FULL;
  h ^= h >> 29;
  return h;
}

static cell *find(uint64_t s, uint64_t a, int create) {
  if (cap == 0) {
    if (!create) return NULL;
    cap = 1 << 16; tab = calloc(cap, sizeof(cell));
  }
  if (create && (cnt + 1) * 2 > cap) {
    cell *old = tab; uint64_t oc = cap;
    cap *= 2; tab = calloc(cap, sizeof(cell)); cnt = 0;
    for (uint64_t i = 0; i < oc; i++) if (old[i].used) { cell *c = find(old[i].space, old[i].addr, 1); c->val = old[i].val; }
    free(old);
  }
  uint64_t i = hsh(s, a) & (cap - 1);
  while (tab[i].used) {
    if (tab[i].space == s && tab[i].addr == a) return &tab[i];
    i = (i + 1) & (cap - 1);
  }
  if (!create) return NULL;
  tab[i].used = 1; tab[i].space = s; tab[i].addr = a; tab[i].val = 0; cnt++;
  return &tab[i];
}

static int key(const sail_int sp, const sail_int a, uint64_t *s, uint64_t *ad) {
  if (mpz_sgn(a) < 0 || mpz_sizeinbase(a, 2) > 64 || mpz_sgn(sp) < 0 || mpz_sizeinbase(sp, 2) > 64) return 0;
  *s = mpz_get_ui(sp); *ad = mpz_get_ui(a);
  if (sizeof(unsigned long) < 8) return 0;
  return 1;
}

void sail_mem_rd(sail_int *rop, const sail_int sp, const sail_int a) {
  uint64_t s, ad;
  if (!key(sp, a, &s, &ad)) { mpz_set_ui(*rop, 0); return; }
  cell *c = find(s, ad, 0);
  mpz_set_ui(*rop, c ? c->val : 0);
}

unit sail_mem_wr(const sail_int sp, const sail_int a, const sail_int v) {
  uint64_t s, ad;
  if (!key(sp, a, &s, &ad)) return UNIT;
  cell *c = find(s, ad, 1);
  c->val = (uint8_t)mpz_get_ui(v);
  return UNIT;
}

/* little-endian n-byte access (n <= 64) */
void sail_mem_rdn(sail_int *rop, const sail_int sp, const sail_int a, const sail_int n) {
  uint64_t s, ad;
  long k = mpz_get_si(n);
  mpz_set_ui(*rop, 0);
  if (!key(sp, a, &s, &ad)) return;
  for (long i = k - 1; i >= 0; i--) {
    cell *c = find(s, ad + (uint64_t)i, 0);
    mpz_mul_2exp(*rop, *rop, 8);
    if (c) mpz_add_ui(*rop, *rop, c->val);
  }
}

unit sail_mem_wrn(const sail_int sp, const sail_int a, const sail_int n, const sail_int v) {
  uint64_t s, ad;
  long k = mpz_get_si(n);
  if (!key(sp, a, &s, &ad)) return UNIT;
  mpz_t x; mpz_init(x);
  mpz_fdiv_r_2exp(x, v, 8 * k);
  for (long i = 0; i < k; i++) {
    cell *c = find(s, ad + (uint64_t)i, 1);
    c->val = (uint8_t)mpz_fdiv_ui(x, 256);
    mpz_fdiv_q_2exp(x, x, 8);
  }
  mpz_clear(x);
  return UNIT;
}

unit sail_mem_copy(const sail_int dsp, const sail_int da, const sail_int ssp, const sail_int sa, const sail_int n) {
  uint64_t s1, a1, s2, a2;
  if (!key(dsp, da, &s1, &a1) || !key(ssp, sa, &s2, &a2)) return UNIT;
  long k = mpz_get_si(n);
  for (long i = 0; i < k; i++) {
    cell *c = find(s2, a2 + (uint64_t)i, 0);
    uint8_t v = c ? c->val : 0;
    if (c || find(s1, a1 + (uint64_t)i, 0)) { cell *d = find(s1, a1 + (uint64_t)i, 1); d->val = v; }
  }
  return UNIT;
}

/* ---------------------------------------------------------------- string-keyed maps of integers
 * (space, key) |-> value; the model uses one space per K map (register file of a frame, symbol
 * environment, ...).  Keys of a space can be listed in insertion order. */
typedef struct ent { uint64_t space; char *key; mpz_t val; struct ent *hnext, *snext, *sprev; } ent;
typedef struct spc { uint64_t id; ent *head, *tail; long n; struct spc *next; } spc;
static ent **hb = NULL; static uint64_t hnb = 0, hcnt = 0;
static spc **sb = NULL; static uint64_t snb = 0;

static uint64_t shash(uint64_t s, const char *k) {
  uint64_t h = 1469598103934665603ULL ^ (s * 0x9E3779B97F4A7C15ULL);
  for (; *k; k++) { h ^= (uint8_t)*k; h *= 1099511628211ULL; }
  return h ^ (h >> 31);
}

static spc *getspc(uint64_t id, int create) {
  if (!sb) { snb = 1024; sb = calloc(snb, sizeof(spc *)); }
  uint64_t i = (id * 0x9E3779B97F4A7C15ULL >> 20) & (snb - 1);
  for (spc *p = sb[i]; p; p = p->next) if (p->id == id) return p;
  if (!create) return NULL;
  spc *p = calloc(1, sizeof(spc)); p->id = id; p->next = sb[i]; sb[i] = p;
  return p;
}

static void hrehash(void) {
  uint64_t nnb = hnb ? hnb * 2 : 4096;
  ent **nb = calloc(nnb, sizeof(ent *));
  for (uint64_t i = 0; i < hnb; i++) {
    ent *e = hb[i];
    while (e) { ent *nx = e->hnext; uint64_t j = shash(e->space, e->key) & (nnb - 1); e->hnext = nb[j]; nb[j] = e; e = nx; }
  }
  free(hb); hb = nb; hnb = nnb;
}

static ent *sfind(uint64_t s, const char *k, int create) {
  if (!hb) hrehash();
  uint64_t i = shash(s, k) & (hnb - 1);
  for (ent *e = hb[i]; e; e = e->hnext) if (e->space == s && strcmp(e->key, k) == 0) return e;
  if (!create) return NULL;
  if (hcnt + 1 > 2 * hnb) { hrehash(); i = shash(s, k) & (hnb - 1); }
  ent *e = calloc(1, sizeof(ent)); e->space = s; e->key = strdup(k); mpz_init(e->val);
  e->hnext = hb[i]; hb[i] = e; hcnt++;
  spc *p = getspc(s, 1);
  e->sprev = p->tail; if (p->tail) p->tail->snext = e; else p->head = e; p->tail = e; p->n++;
  return e;
}

static void sremove(ent *e) {
  uint64_t i = shash(e->space, e->key) & (hnb - 1);
  ent **pp = &hb[i];
  while (*pp != e) pp = &(*pp)->hnext;
  *pp = e->hnext; hcnt--;
  spc *p = getspc(e->space, 1);
  if (e->sprev) e->sprev->snext = e->snext; else p->head = e->snext;
  if (e->snext) e->snext->sprev = e->sprev; else p->tail = e->sprev;
  p->n--;
  free(e->key); mpz_clear(e->val); free(e);
}

void sail_smap_get(sail_int *rop, const sail_int sp, const_sail_string k) {
  ent *e = sfind(mpz_get_ui(sp), k, 0);
  if (e) mpz_set(*rop, e->val); else mpz_set_ui(*rop, 0);
}
bool sail_smap_has(const sail_int sp, const_sail_string k) { return sfind(mpz_get_ui(sp), k, 0) != NULL; }
unit sail_smap_set(const sail_int sp, const_sail_string k, const sail_int v) { ent *e = sfind(mpz_get_ui(sp), k, 1); mpz_set(e->val, v); return UNIT; }
unit sail_smap_del(const sail_int sp, const_sail_string k) { ent *e = sfind(mpz_get_ui(sp), k, 0); if (e) sremove(e); return UNIT; }
unit sail_smap_clear(const sail_int sp) {
  spc *p = getspc(mpz_get_ui(sp), 0);
  while (p && p->head) sremove(p->head);
  return UNIT;
}
void sail_smap_size(sail_int *rop, const sail_int sp) { spc *p = getspc(mpz_get_ui(sp), 0); mpz_set_si(*rop, p ? p->n : 0); }
void sail_smap_key(sail_string *rop, const sail_int sp, const sail_int idx) {
  spc *p = getspc(mpz_get_ui(sp), 0);
  long i = mpz_get_si(idx);
  ent *e = p ? p->head : NULL;
  while (e && i > 0) { e = e->snext; i--; }
  const char *k = e ? e->key : "";
  size_t n = strlen(k);
  *rop = realloc(*rop, n + 1); memcpy(*rop, k, n + 1);
}
unit sail_smap_copy(const sail_int dst, const sail_int src) {
  uint64_t d = mpz_get_ui(dst);
  spc *p = getspc(mpz_get_ui(src), 0);
  for (ent *e = p ? p->head : NULL; e; e = e->snext) { ent *f = sfind(d, e->key, 1); mpz_set(f->val, e->val); }
  return UNIT;
}

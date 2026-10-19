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

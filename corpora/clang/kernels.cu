// Kernels compiled with clang's NVPTX back end (no CUDA SDK needed: only compiler builtins) to obtain PTX from an independent
// producer.  scripts/gen_clang_corpus.sh compiles every kernel at -O0/-O1/-O2/-O3 for several targets.
#define G extern "C" __attribute__((global))
#define SH __attribute__((shared))
static __inline__ __attribute__((device)) int tid() { return __nvvm_read_ptx_sreg_tid_x(); }
static __inline__ __attribute__((device)) int bid() { return __nvvm_read_ptx_sreg_ctaid_x(); }
static __inline__ __attribute__((device)) int bdim() { return __nvvm_read_ptx_sreg_ntid_x(); }
static __inline__ __attribute__((device)) void sync() { __nvvm_bar_sync(0); }

G void saxpy(float a, const float *x, float *y, int n) {
  int i = bid() * bdim() + tid();
  if (i < n) y[i] = a * x[i] + y[i];
}
G void iota(int *out, int n) {
  int i = bid() * bdim() + tid();
  if (i < n) out[i] = i * 3 + 1;
}
G void block_sum(const int *in, int *out) {          // tree reduction in shared memory
  SH int s[64];
  int t = tid();
  s[t] = in[bid() * bdim() + t];
  sync();
  for (int k = bdim() / 2; k > 0; k >>= 1) {
    if (t < k) s[t] += s[t + k];
    sync();
  }
  if (t == 0) out[bid()] = s[0];
}
G void histogram(const int *in, int *bins, int n) {   // global atomics
  int i = bid() * bdim() + tid();
  if (i < n) __nvvm_atom_add_gen_i(&bins[in[i] & 7], 1);
}
G void warp_sum(const int *in, int *out) {            // shuffle reduction
  int v = in[tid()];
  for (int o = 16; o > 0; o >>= 1) v += __nvvm_shfl_sync_down_i32(0xffffffff, v, o, 31);
  if (tid() == 0) out[0] = v;
}
G void transpose_tile(const float *in, float *out) {  // 8x8 tile through shared memory
  SH float tile[64];
  int x = tid() & 7, y = tid() >> 3;
  tile[y * 8 + x] = in[y * 8 + x];
  sync();
  out[x * 8 + y] = tile[y * 8 + x];
}
G void matmul8(const float *a, const float *b, float *c) {   // 8x8 matrix product, one thread per element
  int r = tid() >> 3, col = tid() & 7;
  float acc = 0.f;
  for (int k = 0; k < 8; k++) acc += a[r * 8 + k] * b[k * 8 + col];
  c[r * 8 + col] = acc;
}
G void bitops(const unsigned *in, unsigned *out) {
  unsigned v = in[tid()];
  out[tid()] = __builtin_popcount(v) + (__builtin_clz(v | 1) << 8) + (__builtin_bswap32(v) >> 24);
}
G void divmod(const int *a, const int *b, int *q, int *r) {
  int i = tid();
  q[i] = a[i] / b[i];
  r[i] = a[i] % b[i];
}
G void clamp_float(const float *in, float *out) {
  float v = in[tid()];
  out[tid()] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
}
G void prefix_sum(const int *in, int *out) {          // Hillis-Steele scan in shared memory
  SH int s[32];
  int t = tid();
  s[t] = in[t];
  sync();
  for (int d = 1; d < 32; d <<= 1) {
    int v = t >= d ? s[t - d] : 0;
    sync();
    s[t] += v;
    sync();
  }
  out[t] = s[t];
}
G void cas_max(const int *in, int *out, int n) {      // atomic max via compare-and-swap loop
  int i = bid() * bdim() + tid();
  if (i >= n) return;
  int old = *out;
  while (in[i] > old) {
    int seen = __sync_val_compare_and_swap(out, old, in[i]);
    if (seen == old) break;
    old = seen;
  }
}
G void gather_scatter(const int *idx, const int *src, int *dst) {
  int i = tid();
  dst[idx[i]] = src[i] * 2;
}
G void sqrt_poly(const float *in, float *out) {
  float x = in[tid()];
  out[tid()] = __builtin_sqrtf(x) * 0.5f + x * x * 0.125f;
}

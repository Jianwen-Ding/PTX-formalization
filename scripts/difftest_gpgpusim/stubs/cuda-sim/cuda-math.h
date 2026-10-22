/* Clean-room replacement for GPGPU-Sim's src/cuda-sim/cuda-math.h.
 *
 * The upstream file is derived from the CUDA toolkit (see GPGPU-Sim's
 * COPYRIGHT, NOTE 1) and pulls in NVIDIA's <math_functions.h>.  The
 * PTX-formalization differential tester deletes the upstream file and drops
 * this one in its place.  It provides exactly the cuda_math:: helpers that
 * src/cuda-sim/instructions.cc calls, implemented with the host C library
 * and <fenv.h>; the signatures (all single precision) are chosen so that
 * GPGPU-Sim's call sites behave as they do upstream, including the implicit
 * double->float narrowing at the cvt.rni.f64 call site. */
#ifndef CUDA_MATH
#define CUDA_MATH

#include <cmath>
#include <cstdlib>
#include <fenv.h>
#include "device_types.h"

namespace cuda_math {

#define DIFFTEST_CVT_WITH_MODE(name, T, mode)       \
  static inline float name(T a) {                   \
    int saved = fegetround();                       \
    fesetround(mode);                               \
    volatile float r = (float)a;                    \
    fesetround(saved);                              \
    return r;                                       \
  }
DIFFTEST_CVT_WITH_MODE(__int2float_rn, int, FE_TONEAREST)
DIFFTEST_CVT_WITH_MODE(__int2float_rz, int, FE_TOWARDZERO)
DIFFTEST_CVT_WITH_MODE(__int2float_ru, int, FE_UPWARD)
DIFFTEST_CVT_WITH_MODE(__int2float_rd, int, FE_DOWNWARD)
DIFFTEST_CVT_WITH_MODE(__uint2float_rn, unsigned int, FE_TONEAREST)
DIFFTEST_CVT_WITH_MODE(__uint2float_rz, unsigned int, FE_TOWARDZERO)
DIFFTEST_CVT_WITH_MODE(__uint2float_ru, unsigned int, FE_UPWARD)
DIFFTEST_CVT_WITH_MODE(__uint2float_rd, unsigned int, FE_DOWNWARD)
DIFFTEST_CVT_WITH_MODE(__ll2float_rn, long long, FE_TONEAREST)
DIFFTEST_CVT_WITH_MODE(__ll2float_rz, long long, FE_TOWARDZERO)
DIFFTEST_CVT_WITH_MODE(__ll2float_ru, long long, FE_UPWARD)
DIFFTEST_CVT_WITH_MODE(__ll2float_rd, long long, FE_DOWNWARD)
DIFFTEST_CVT_WITH_MODE(__ull2float_rn, unsigned long long, FE_TONEAREST)
DIFFTEST_CVT_WITH_MODE(__ull2float_rz, unsigned long long, FE_TOWARDZERO)
DIFFTEST_CVT_WITH_MODE(__ull2float_ru, unsigned long long, FE_UPWARD)
DIFFTEST_CVT_WITH_MODE(__ull2float_rd, unsigned long long, FE_DOWNWARD)
#undef DIFFTEST_CVT_WITH_MODE

static inline float difftest_round(float a, enum cudaRoundMode mode) {
  switch (mode) {
    case cudaRoundZero: return truncf(a);
    case cudaRoundNearest: return nearbyintf(a);
    case cudaRoundMinInf: return floorf(a);
    case cudaRoundPosInf: return ceilf(a);
  }
  abort();
}
/* float -> integer: like upstream, a plain C conversion of the rounded value
 * (out-of-range inputs are therefore host-defined, not PTX-saturating). */
static inline int float2int(float a, enum cudaRoundMode mode) {
  return (int)difftest_round(a, mode);
}
static inline unsigned int float2uint(float a, enum cudaRoundMode mode) {
  return (unsigned int)difftest_round(a, mode);
}

static inline float __internal_nearbyintf(float a) { return nearbyintf(a); }
static inline float __cuda_nearbyintf(float a) { return nearbyintf(a); }
static inline int __cuda___isnanf(float a) { return std::isnan(a); }

static inline float __internal_accurate_fdividef(float a, float b) {
  return a / b;
}
static inline float __powf(float a, float b) { return powf(a, b); }
static inline float __saturatef(float a) {
  if (std::isnan(a)) return 0.0f;
  if (a >= 1.0f) return 1.0f;
  if (a <= 0.0f) return 0.0f;
  return a;
}

}  // namespace cuda_math

#endif

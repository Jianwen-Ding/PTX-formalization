/* Minimal clean-room stand-in for the CUDA vector types that GPGPU-Sim's PTX
 * core refers to.  Written for the PTX-formalization differential tester;
 * contains no NVIDIA code.  Only the members GPGPU-Sim touches are defined. */
#ifndef DIFFTEST_STUB_VECTOR_TYPES_H
#define DIFFTEST_STUB_VECTOR_TYPES_H
struct uint3 { unsigned int x, y, z; };
struct int2 { int x, y; };
struct int4 { int x, y, z, w; };
struct uint2 { unsigned int x, y; };
struct uint4 { unsigned int x, y, z, w; };
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct double2 { double x, y; };
struct dim3 {
  unsigned int x, y, z;
  dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
  dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
  operator uint3() const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
};
typedef struct dim3 dim3;
#endif

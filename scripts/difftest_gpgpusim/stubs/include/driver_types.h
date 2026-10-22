/* Minimal clean-room stand-in for the CUDA runtime driver types used by
 * GPGPU-Sim's PTX core (written for the PTX-formalization differential
 * tester; contains no NVIDIA code). */
#ifndef DIFFTEST_STUB_DRIVER_TYPES_H
#define DIFFTEST_STUB_DRIVER_TYPES_H
#include <stddef.h>
enum cudaError { cudaSuccess = 0, cudaErrorUnknown = 999 };
typedef enum cudaError cudaError_t;
struct CUstream_st;
struct CUevent_st;
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
#define cudaStreamNonBlocking 0x01
enum cudaChannelFormatKind {
  cudaChannelFormatKindSigned = 0,
  cudaChannelFormatKindUnsigned = 1,
  cudaChannelFormatKindFloat = 2,
  cudaChannelFormatKindNone = 3
};
struct cudaChannelFormatDesc { int x, y, z, w; enum cudaChannelFormatKind f; };
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3
};
struct cudaDeviceProp;  /* only used through pointers */
#endif

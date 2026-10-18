/* declarations just enough to type-check the generated host.c without the CUDA SDK (real runs use <cuda.h>) */
#include <stddef.h>
typedef int CUresult; typedef int CUdevice; typedef struct CUctx_st *CUcontext; typedef struct CUmod_st *CUmodule; typedef struct CUfunc_st *CUfunction;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS 0
CUresult cuInit(unsigned); CUresult cuDeviceGet(CUdevice *, int); CUresult cuCtxCreate(CUcontext *, unsigned, CUdevice);
CUresult cuModuleLoadData(CUmodule *, const void *); CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuMemAlloc(CUdeviceptr *, size_t); CUresult cuMemcpyHtoD(CUdeviceptr, const void *, size_t); CUresult cuMemcpyDtoH(void *, CUdeviceptr, size_t);
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **);
CUresult cuCtxSynchronize(void);

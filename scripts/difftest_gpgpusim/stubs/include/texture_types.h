/* Minimal clean-room stand-in (PTX-formalization differential tester). */
#ifndef DIFFTEST_STUB_TEXTURE_TYPES_H
#define DIFFTEST_STUB_TEXTURE_TYPES_H
#include "driver_types.h"
enum cudaTextureAddressMode {
  cudaAddressModeWrap = 0, cudaAddressModeClamp = 1,
  cudaAddressModeMirror = 2, cudaAddressModeBorder = 3
};
enum cudaTextureFilterMode { cudaFilterModePoint = 0, cudaFilterModeLinear = 1 };
enum cudaTextureReadMode {
  cudaReadModeElementType = 0, cudaReadModeNormalizedFloat = 1
};
struct textureReference {
  int normalized;
  enum cudaTextureFilterMode filterMode;
  enum cudaTextureAddressMode addressMode[3];
  struct cudaChannelFormatDesc channelDesc;
};
#endif

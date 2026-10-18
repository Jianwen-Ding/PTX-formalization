#!/usr/bin/env bash
# Compile corpora/clang/kernels.cu to PTX with clang's NVPTX back end (needs only clang, no CUDA SDK) into corpora/clang/ptx/.
set -eu
cd "$(dirname "$0")/.."
mkdir -p corpora/clang/ptx
for sm in sm_70 sm_80 sm_90; do
  for o in O0 O1 O2 O3; do
    clang -x cuda --cuda-device-only --offload-arch=$sm -nocudainc -nocudalib --cuda-path=/nonexistent -Xclang -target-feature -Xclang +ptx78 -$o -S corpora/clang/kernels.cu \
          -o corpora/clang/ptx/kernels-$sm-$o.ptx
  done
done
ls corpora/clang/ptx | wc -l

#ifndef _cuda_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

cudaError_t tryCudaMalloc(int* tmp);
cudaError_t tryCudaFree(int* tmp);

#define _cuda_H
#endif

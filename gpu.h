#ifndef _gpu_runner_gpu_H

#include "CL/cl.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvrtc.h"

cudaError_t tryCudaMalloc(int* tmp);
cudaError_t tryCudaFree(int* tmp);

#define _gpu_runner_gpu_H
#endif

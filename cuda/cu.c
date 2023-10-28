#include "cu.h"

cudaError_t tryCudaMalloc(int* tmp) {
	return cudaMalloc((void **)&tmp, 256);
}

cudaError_t tryCudaFree(int* tmp) {
	return cudaFree((void *)tmp);
}


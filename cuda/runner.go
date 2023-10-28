package cuda

// #include "cu.h"
import "C"

import (
	"fmt"
	"unsafe"
)

func RunCuda(device *CudaDevice) error {
	var cu_code = C.CString(`
		__global__
		void saxpy(float a, float *x, float *y, float *out, size_t n)
		{
		   size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		   if (tid < n) {
			  out[tid] = a * x[tid] + y[tid];
		   }
		}
	`)
	defer C.free(unsafe.Pointer(cu_code))

	// nvrtcCreateProgram
	var program C.nvrtcProgram
	var err = C.nvrtcCreateProgram(&program, cu_code, nil, 0, nil, nil)
	if err != C.NVRTC_SUCCESS {
		return fmt.Errorf("nvrtcCreateProgram Err: %s", C.GoString(C.nvrtcGetErrorString(err)))
	}
	defer C.nvrtcDestroyProgram(&program)

	// nvrtcAddNameExpression
	var kernel_name = C.CString("saxpy")
	defer C.free(unsafe.Pointer(kernel_name))
	err = C.nvrtcAddNameExpression(program, kernel_name)
	if err != C.NVRTC_SUCCESS {
		return fmt.Errorf("nvrtcAddNameExpression Err: %s", C.GoString(C.nvrtcGetErrorString(err)))
	}

	// nvrtcCompileProgram
	var arch_opt = C.CString(fmt.Sprintf("-arch=compute_%d", device.Props.major*10+device.Props.minor))
	defer C.free(unsafe.Pointer(arch_opt))
	var opts = [1](*C.char){arch_opt}
	err = C.nvrtcCompileProgram(program, C.int(len(opts)), &opts[0])
	if err != C.NVRTC_SUCCESS {
		var logSize C.size_t
		var err2 = C.nvrtcGetProgramLogSize(program, &logSize)
		if err2 != C.NVRTC_SUCCESS {
			return fmt.Errorf("nvrtcGetProgramLogSize Err: %v", C.GoString(C.nvrtcGetErrorString(err2)))
		}
		var log_buf = make([]byte, logSize, logSize)
		err2 = C.nvrtcGetProgramLog(program, (*C.char)(unsafe.Pointer(&log_buf[0])))

		if err2 != C.NVRTC_SUCCESS {
			return fmt.Errorf("nvrtcGetProgramLog Err: %v", C.GoString(C.nvrtcGetErrorString(err2)))
		}

		fmt.Printf("nvrtcCompileProgram Err log: %s\n", string(log_buf))

		return fmt.Errorf("nvrtcCompileProgram Err: %s", C.GoString(C.nvrtcGetErrorString(err)))
	}

	// nvrtcGetLoweredName
	var lowered_kernel_name *C.char
	err = C.nvrtcGetLoweredName(program, kernel_name, &lowered_kernel_name)
	if err != C.NVRTC_SUCCESS {
		return fmt.Errorf("nvrtcGetLoweredName Err: %s", C.GoString(C.nvrtcGetErrorString(err)))
	}

	// Obtain PTX from the program.
	var ptxSize C.size_t
	err = C.nvrtcGetPTXSize(program, &ptxSize)
	if err != C.NVRTC_SUCCESS {
		return fmt.Errorf("nvrtcGetPTXSize Err: %v", C.GoString(C.nvrtcGetErrorString(err)))
	}
	var ptx_buf = make([]C.char, ptxSize, ptxSize)
	err = C.nvrtcGetPTX(program, &ptx_buf[0])
	if err != C.NVRTC_SUCCESS {
		return fmt.Errorf("nvrtcGetPTX Err: %v", C.GoString(C.nvrtcGetErrorString(err)))
	}

	// Get kernel
	var cu_err = C.cuInit(0)
	if cu_err != C.CUDA_SUCCESS {
		return fmt.Errorf("cuInit Err: %v", cu_err)
	}

	var module C.CUmodule
	cu_err = C.cuModuleLoadDataEx(&module, unsafe.Pointer(&ptx_buf[0]), 0, nil, nil)
	defer C.cuModuleUnload(module)

	if cu_err != C.CUDA_SUCCESS {
		return fmt.Errorf("cuModuleLoadDataEx Err: %v", cu_err)
	}

	var kernel C.CUfunction
	cu_err = C.cuModuleGetFunction(&kernel, module, lowered_kernel_name)
	if cu_err != C.CUDA_SUCCESS {
		return fmt.Errorf("cuModuleGetFunction Err: %v", cu_err)
	}

	// set kernel args
	const NUM_BLOCKS = 32
	const NUM_THREADS = 128

	var a C.float = 5.1
	var n C.size_t = NUM_BLOCKS * NUM_THREADS

	var buffer_size = n * C.sizeof_float

	var hX = make([]C.float, n, n)
	var hY = make([]C.float, n, n)
	var hOut = make([]C.float, n, n)

	for i := C.size_t(0); i < n; i++ {
		hX[i] = C.float(i)
		hY[i] = C.float(i * i)
	}

	var dX, dY, dOut unsafe.Pointer

	var cuda_err = C.cudaSetDevice(device.Device_id)
	if cuda_err != C.cudaSuccess {
		return fmt.Errorf("cudaSetDevice Err: %s", C.GoString(C.cudaGetErrorName(cuda_err)))
	}
	defer C.cudaDeviceReset()

	cuda_err = C.cudaMalloc(&dX, buffer_size)
	if cuda_err != C.cudaSuccess {
		return fmt.Errorf("cudaMalloc Err: %s", C.GoString(C.cudaGetErrorName(cuda_err)))
	}
	defer C.cudaFree(dX)

	cuda_err = C.cudaMalloc(&dY, buffer_size)
	if cuda_err != C.cudaSuccess {
		return fmt.Errorf("cudaMalloc Err: %s", C.GoString(C.cudaGetErrorName(cuda_err)))
	}
	defer C.cudaFree(dY)

	cuda_err = C.cudaMalloc(&dOut, buffer_size)
	if cuda_err != C.cudaSuccess {
		return fmt.Errorf("cudaMalloc Err: %s", C.GoString(C.cudaGetErrorName(cuda_err)))
	}
	defer C.cudaFree(dOut)

	cuda_err = C.cudaMemcpy(dX, unsafe.Pointer(&hX[0]), buffer_size, C.cudaMemcpyHostToDevice)
	cuda_err = C.cudaMemcpy(dY, unsafe.Pointer(&hY[0]), buffer_size, C.cudaMemcpyHostToDevice)

	var args = []unsafe.Pointer{unsafe.Pointer(&a), unsafe.Pointer(&dX), unsafe.Pointer(&dY), unsafe.Pointer(&dOut), unsafe.Pointer(&n)}

	// Run kernel
	cu_err = C.cuLaunchKernel(kernel,
		32, 1, 1, //grid dim
		128, 1, 1, // block dim
		0, // sharedMemBytes
		nil,
		&args[0],
		nil,
	)
	if cu_err != C.CUDA_SUCCESS {
		return fmt.Errorf("cuLaunchKernel Err: %v", cu_err)
	}

	cu_err = C.cuCtxSynchronize()
	if cu_err != C.CUDA_SUCCESS {
		return fmt.Errorf("cuCtxSynchronize Err: %v", cu_err)
	}

	// Copy data to host
	cuda_err = C.cudaMemcpy(unsafe.Pointer(&hOut[0]), dOut, buffer_size, C.cudaMemcpyDeviceToHost)
	if cuda_err != C.cudaSuccess {
		return fmt.Errorf("cudaMemcpy Err: %s", C.GoString(C.cudaGetErrorName(cuda_err)))
	}

	fmt.Println(hOut[:20])

	return nil
}

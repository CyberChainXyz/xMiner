package gpurunner

// #include "gpu.h"
import "C"

import "fmt"

func getCudaInfo() (*CudaInfo, error) {
	var info CudaInfo

	info.Cudart_version = C.CUDART_VERSION

	var err = C.cudaDriverGetVersion(&info.Driver_version)
	if err != C.cudaSuccess {
		return &info, fmt.Errorf("cudaDriverGetVersion Err: %s", C.GoString(C.cudaGetErrorName(err)))
	}

	var nvrtc_err = C.nvrtcVersion(&info.Nvrtc_version[0], &info.Nvrtc_version[1])
	if nvrtc_err != C.NVRTC_SUCCESS {
		return &info, fmt.Errorf("nvrtcVersion Err: %s", C.GoString(C.nvrtcGetErrorString(nvrtc_err)))
	}

	err = C.cudaGetDeviceCount(&info.Device_count)
	if err == C.cudaErrorNoDevice {
		return &info, nil
	}
	if err != C.cudaSuccess {
		return &info, fmt.Errorf("cudaGetDeviceCount Err: %s", C.GoString(C.cudaGetErrorName(err)))
	}

	for i := C.int(0); i <= info.Device_count; i++ {
		device := CudaDevice{Device_id: i}
		info.Devices = append(info.Devices, device)

		err = C.cudaGetDeviceProperties(&device.Props, device.Device_id)
		if err != C.cudaSuccess {
			return &info, fmt.Errorf("cudaGetDeviceProperties Err: %s", C.GoString(C.cudaGetErrorName(err)))
		}

		device.name = C.GoString(&device.Props.name[0])

		err = C.cudaSetDevice(device.Device_id)
		if err != C.cudaSuccess {
			return &info, fmt.Errorf("cudaSetDevice Err: %s", C.GoString(C.cudaGetErrorName(err)))
		}

		var tmp *C.int
		err = C.tryCudaMalloc(tmp)
		if err != C.cudaSuccess {
			return &info, fmt.Errorf("cudaMalloc Err: %s", C.GoString(C.cudaGetErrorName(err)))
		}

		err = C.cudaMemGetInfo(&device.Free_memory, &device.Total_memory)
		if err != C.cudaSuccess {
			return &info, fmt.Errorf("cudaMemGetInfo Err: %s", C.GoString(C.cudaGetErrorName(err)))
		}

		err = C.tryCudaFree(tmp)
		if err != C.cudaSuccess {
			return &info, fmt.Errorf("cudaFree Err: %s", C.GoString(C.cudaGetErrorName(err)))
		}

		err = C.cudaDeviceReset()
		if err != C.cudaSuccess {
			return &info, fmt.Errorf("cudaDeviceReset Err: %s", C.GoString(C.cudaGetErrorName(err)))
		}
	}

	return &info, nil
}

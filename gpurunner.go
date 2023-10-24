package gpurunner

// #cgo CFLAGS: -DCL_TARGET_OPENCL_VERSION=300
// #cgo LDFLAGS: -lOpenCL -lnvrtc -lcudart
import "C"

import "fmt"

func GetAllGpusInfo() GpusInfo {
	cuda, err_cuda := getCudaInfo()
	opencl, err_opencl := getOpenCLInfo()

	if err_cuda != nil {
		fmt.Printf("getCudaInfo Err: %v\n", err_cuda)
	}

	if err_opencl != nil {
		fmt.Printf("getOpenCLInfo Err: %v\n", err_opencl)
	}

	return GpusInfo{
		Cuda:   *cuda,
		OpenCL: *opencl,
	}
}

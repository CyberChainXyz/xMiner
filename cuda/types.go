package cuda

// #include "cu.h"
import "C"

// cuda
type CudaDevice struct {
	Device_id    C.int
	Free_memory  C.size_t
	Total_memory C.size_t
	name         string
	Props        C.struct_cudaDeviceProp
}

type CudaInfo struct {
	Device_count   C.int
	Driver_version C.int
	Cudart_version C.int
	Nvrtc_version  [2]C.int
	Devices        []CudaDevice
}

package gpurunner

// #include "gpu.h"
import "C"

import "fmt"
import "unsafe"

func Run(device *OpenCLDevice) error {
	// clCreateContext
	var context_properties = [3]C.cl_context_properties{
		C.CL_CONTEXT_PLATFORM,
		C.cl_context_properties(uintptr(unsafe.Pointer(device.Platform_id))),
		0,
	}
	var err C.cl_int
	var context = C.clCreateContext(&context_properties[0], 1, &device.Device_id, nil, nil, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("clCreateContext Err: %v", err)
	}

	// clCreateCommandQueueWithProperties
	var commandQueue = C.clCreateCommandQueueWithProperties(context, device.Device_id, nil, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("clCreateCommandQueueWithProperties Err: %v", err)
	}

	var code_src = C.CString(`
		__kernel void helloworld(__global float* in, __global float* out)
		 {
			 int num = get_global_id(0);
			 out[num] = in[num] / 24 *(in[num]/60) * in[num];
		 }`)
	defer C.free(unsafe.Pointer(code_src))
	var codes = [1](*C.char){code_src}

	// clCreateProgramWithSource
	var program = C.clCreateProgramWithSource(context, C.cl_uint(len(codes)), &codes[0], nil, &err)

	if err != C.CL_SUCCESS {
		return fmt.Errorf("clCreateProgramWithSource Err: %v", err)
	}

	// clBuildProgram
	options := C.CString("")
	defer C.free(unsafe.Pointer(options))
	err = C.clBuildProgram(program, 1, &device.Device_id, options, nil, nil)
	if err != C.CL_SUCCESS {
		var logSize C.size_t
		var err2 = C.clGetProgramBuildInfo(program, device.Device_id, C.CL_PROGRAM_BUILD_LOG, 0, nil, &logSize)
		if err2 != C.CL_SUCCESS {
			return fmt.Errorf("clGetProgramBuildInfo Err: %v", err2)
		}

		var log_buf = make([]byte, logSize, logSize)
		err2 = C.clGetProgramBuildInfo(program, device.Device_id, C.CL_PROGRAM_BUILD_LOG, logSize, unsafe.Pointer(&log_buf[0]), nil)

		if err2 != C.CL_SUCCESS {
			return fmt.Errorf("clGetProgramBuildInfo Err: %v", err2)
		}

		fmt.Printf("clBuildProgram Err log: %s\n", string(log_buf))

		return fmt.Errorf("clBuildProgram Err: %v", err)
	}

	// clCreateKernel
	var kernel_name = C.CString("helloworld")
	defer C.free(unsafe.Pointer(kernel_name))
	var kernel = C.clCreateKernel(program, kernel_name, &err)
	if err != C.CL_SUCCESS {
		return fmt.Errorf("clCreateKernel Err: %v", err)
	}

	const NUM = 1000
	var input = [NUM]C.float{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	var output = [NUM]C.float{}
	for i, _ := range input {
		input[i] = C.float(float32(i + 1))
	}

	// clCreateBuffer
	var inputBuffer = C.clCreateBuffer(context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR, C.size_t(unsafe.Sizeof(input[0])*NUM), unsafe.Pointer(&input[0]), &err)
	var outputBuffer = C.clCreateBuffer(context, C.CL_MEM_WRITE_ONLY, C.size_t(unsafe.Sizeof(input[0])*NUM), nil, &err)
	// clSetKernelArg
	err = C.clSetKernelArg(kernel, 0, C.sizeof_cl_mem, unsafe.Pointer(&inputBuffer))
	err = C.clSetKernelArg(kernel, 1, C.sizeof_cl_mem, unsafe.Pointer(&outputBuffer))

	// clEnqueueNDRangeKernel
	var evt C.cl_event
	var items C.size_t = 15
	err = C.clEnqueueNDRangeKernel(commandQueue, kernel, 1, nil, &items, nil, 0, nil, &evt)
	err = C.clWaitForEvents(1, &evt)
	err = C.clReleaseEvent(evt)

	// clEnqueueReadBuffer
	err = C.clEnqueueReadBuffer(commandQueue, outputBuffer, C.CL_TRUE, 0, C.size_t(unsafe.Sizeof(input[0])*NUM), unsafe.Pointer(&output[0]), 0, nil, nil)

	// clean

	err = C.clReleaseMemObject(inputBuffer)
	err = C.clReleaseMemObject(outputBuffer)
	err = C.clReleaseKernel(kernel)
	err = C.clReleaseProgram(program)
	err = C.clReleaseCommandQueue(commandQueue)
	err = C.clReleaseContext(context)

	fmt.Println(output[:20])

	return nil
}

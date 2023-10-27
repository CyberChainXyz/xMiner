package main

import gpu "github.com/dextech/gpu-runner"
import "github.com/kr/pretty"

import "log"
import "unsafe"
import "fmt"
import "time"

func main() {
	gpus := gpu.GetAllGpusInfo()
	pretty.Println(gpus)

	if len(gpus.OpenCL.Platforms) < 1 {
		log.Fatalln("No OpenCL Devices")
	}
	if len(gpus.OpenCL.Platforms[0].Devices) < 1 {
		log.Fatalln("No OpenCL Devices")
	}

	code := `__kernel void helloworld(__global float* in, __global float* out, int factor)
		 {
			 int num = get_global_id(0);
			 for (int i=0; i < 1000; i++)
				 out[num] = out[num] / (factor-7) + in[num] / factor;
		 }`

	// InitRunner
	device := &gpus.OpenCL.Platforms[0].Devices[0]
	t := time.Now()
	runner, err := device.InitRunner()
	fmt.Println("InitRunner: ", time.Since(t))
	if err != nil {
		log.Fatalln("InitRunner err:", err)
	}
	defer runner.Free()

	// CompileKernels
	codes := []string{code}
	kernelNameList := []string{"helloworld"}
	t = time.Now()
	err = runner.CompileKernels(codes, kernelNameList, "")
	fmt.Println("CompileKernels: ", time.Since(t))
	if err != nil {
		log.Fatalln("CompileKernels err:", err)
	}

	// create kernel params
	itemSize := int(unsafe.Sizeof(gpu.Float(0)))
	itemCount := 15
	/* buffer 1 param */
	input := []gpu.Float{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
	t = time.Now()
	buf_1, err := gpu.CreateBuffer(runner, gpu.READ_ONLY|gpu.COPY_HOST_PTR, input)
	fmt.Println("CreateBuffer: ", time.Since(t))
	if err != nil {
		log.Fatalln("CreateBuffer err:", err)
	}
	/* buffer 2 param */
	t = time.Now()
	buf_2, err := runner.CreateEmptyBuffer(gpu.WRITE_ONLY, itemCount*itemSize)
	fmt.Println("CreateEmptyBuffer: ", time.Since(t))
	if err != nil {
		log.Fatalln("CreateEmptyBuffer err:", err)
	}
	/* factor param */
	var factor gpu.Int = 13

	// RunKernel
	t = time.Now()
	err = runner.RunKernel("helloworld", 1, nil, []int{itemCount}, nil, []gpu.KernelParam{
		gpu.Param(&buf_1),
		gpu.Param(&buf_2),
		gpu.Param(&factor),
	}, true)
	fmt.Println("RunKernel: ", time.Since(t))
	if err != nil {
		log.Fatalln("RunKernel err:", err)
	}

	// ReadBuffer
	output := make([]gpu.Float, len(input))
	t = time.Now()
	err = gpu.ReadBuffer(runner, 0, buf_2, output)
	fmt.Println("ReadBuffer: ", time.Since(t))
	if err != nil {
		log.Fatalln("ReadBuffer err:", err)
	}

	// Show Result
	fmt.Println(output)
}

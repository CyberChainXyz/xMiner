package main

import gpu "github.com/dextech/gpu-runner"
import "github.com/kr/pretty"

import "log"
import "fmt"
import "time"
import _ "embed"
import "math"
import "sort"
// import "unsafe"
// import "math/bits"

import "example.com/cn-gpu-go"

//go:embed cn.cl
var code string

const ITERATIONS = 768
const MEMORY = 32 * 1024
const MASK = ((MEMORY - 1) >> 6) << 6
const NonceLen = 4

func main() {
	// fmt.Println(unsafe.Sizeof(uint(0)))
	//fmt.Println(bits.UintSize)
	gpus := gpu.GetAllGpusInfo()
	pretty.Println(gpus)

	if len(gpus.OpenCL.Platforms) < 1 {
		log.Fatalln("No OpenCL Devices")
	}
	if len(gpus.OpenCL.Platforms[0].Devices) < 1 {
		log.Fatalln("No OpenCL Devices")
	}

	// InitRunner
	device := &gpus.OpenCL.Platforms[0].Devices[0]
	t := time.Now()
	runner, err := device.InitRunner()
	fmt.Println("InitRunner: ", time.Since(t))
	if err != nil {
		log.Fatalln("InitRunner err:", err)
	}
	defer runner.Free()

	
	// kernel constants
	compMode := 0
	unroll := 1
	workSize := min(int(device.Max_work_group_size)/16, 8)
	maxThreads := int(device.Max_compute_units * 6 * 8)

	// CompileKernels
	codes := []string{code}
	kernelNameList := []string{"cn0_cn_gpu", "cn00_cn_gpu", "cn1_cn_gpu", "cn2"}
	options := fmt.Sprintf("-DITERATIONS=%d"+" -DMASK=%dU"+" -DWORKSIZE=%dU"+" -DCOMP_MODE=%d"+" -DMEMORY=%dLU"+" -DCN_UNROLL=%d"+" -cl-fp32-correctly-rounded-divide-sqrt", ITERATIONS, MASK, workSize, compMode, MEMORY, unroll)

	t = time.Now()
	err = runner.CompileKernels(codes, kernelNameList, options)
	fmt.Println("CompileKernels: ", time.Since(t))
	if err != nil {
		log.Fatalln("CompileKernels err:", err)
	}

	maxThreads *= 32

	// create buffers
	g_thd := maxThreads
	scratchPadSize := MEMORY
	t = time.Now()
	input_buf, err := runner.CreateEmptyBuffer(gpu.READ_ONLY, 128)
	scratchpads_buf, err := runner.CreateEmptyBuffer(gpu.READ_WRITE, scratchPadSize*g_thd)
	states_buf, err := runner.CreateEmptyBuffer(gpu.READ_WRITE, 200*g_thd)
	output_buf, err := runner.CreateEmptyBuffer(gpu.READ_WRITE, NonceLen*0x100)
	fmt.Println("CreateBuffer:", time.Since(t))
	if err != nil {
		log.Fatalln("CreateBuffer err:", err)
	}

	// input
	var source = []byte{1, 2, 3, 4, 5, 6, 7, 8}
	// Target
	var target uint64 = math.MaxUint64 / 100000
	// nonce
	var startNonce uint32 = 0

	nonce_list := hash_iter(runner, source, startNonce, target, compMode, workSize, maxThreads, 
		input_buf, output_buf, scratchpads_buf, states_buf)
	fmt.Println("nonce_list:", nonce_list)

	// cpu hash
	// source_1 := append(source, 0, 0, 73, 233)
	source_1 := append(source, 0, 0, 5, 12)
	hash := cngpugo.Hash(source_1)
	fmt.Println("hash:", hash)


}

func hash_iter(runner *gpu.OpenCLRunner, source []byte, startNonce uint32, target uint64, 
	compMode int, workSize int, maxThreads int,
	input_buf *gpu.Buffer, output_buf *gpu.Buffer, scratchpads_buf *gpu.Buffer, states_buf *gpu.Buffer) []uint32{

	// input 
	input := make([]byte, 128, 128)
	input_len := len(source)
	copy(input[:input_len], source)
	input[input_len] = 0x01
	// numThreads
	numThreads := maxThreads
	// output
	var output = make([]uint32, 0x100, 0x100)
	
	// set input buffer
	t := time.Now()
	err := gpu.WriteBuffer(runner, 0, input_buf, input, true)
	fmt.Println("WriteBuffer input_buf: ", time.Since(t))
	if err != nil {
		log.Fatalln("WriteBuffer err:", err)
	}
	
	// kernel params
	k0_args := []gpu.KernelParam{
		gpu.BufferParam(input_buf),
		gpu.BufferParam(scratchpads_buf),
		gpu.BufferParam(states_buf),
		gpu.Param(&numThreads),
	}

	// kernel params
	k00_args := []gpu.KernelParam{
		gpu.BufferParam(scratchpads_buf),
		gpu.BufferParam(states_buf),
	}

	// kernel params
	k1_args := []gpu.KernelParam{
		gpu.BufferParam(scratchpads_buf),
		gpu.BufferParam(states_buf),
		gpu.Param(&numThreads),
	}

	// kernel params
	k2_args := []gpu.KernelParam{
		gpu.BufferParam(scratchpads_buf),
		gpu.BufferParam(states_buf),
		gpu.BufferParam(output_buf),
		gpu.Param(&target),
		gpu.Param(&numThreads),
	}

	// ===============================================
	// Run kernels loop
	g_intensity := maxThreads
	w_size := workSize
	g_thd := g_intensity

	if g_thd % w_size == 0 {
		compMode = 0 
	}
	if compMode != 0 {
		// round up to next multiple of w_size
		g_thd = ((g_intensity + w_size - 1) / w_size) * w_size
	}

	t = time.Now()
	err = gpu.WriteBuffer(runner, NonceLen*0xFF, output_buf, []uint32{0}, false)
	fmt.Println("WriteBuffer output_buf reset: ", time.Since(t))
	if err != nil {
		log.Fatalln("WriteBuffer err:", err)
	}

	t = time.Now()
	err = runner.RunKernel("cn0_cn_gpu", 2, []int{int(startNonce), 1}, []int{g_thd, 8}, []int{8, 8}, k0_args, false)
	fmt.Println("RunKernel cn0_cn_gpu: ", time.Since(t))

	t = time.Now()
	err = runner.RunKernel("cn00_cn_gpu", 1, nil, []int{g_intensity * 64}, []int{64}, k00_args, false)
	fmt.Println("RunKernel cn00_cn_gpu: ", time.Since(t))

	t = time.Now()
	err = runner.RunKernel("cn1_cn_gpu", 1, nil, []int{g_thd * 16}, []int{w_size * 16}, k1_args, false)
	fmt.Println("RunKernel cn1_cn_gpu: ", time.Since(t))

	t = time.Now()
	err = runner.RunKernel("cn2", 2, []int{0, int(startNonce)}, []int{8, g_thd}, []int{8, w_size}, k2_args, false)
	fmt.Println("RunKernel cn2: ", time.Since(t))

	if err != nil {
		log.Fatalln("RunKernel err:", err)
	}

	// ReadBuffer
	t = time.Now()
	err = gpu.ReadBuffer(runner, 0, output_buf, output)
	fmt.Println("ReadBuffer: ", time.Since(t))
	if err != nil {
		log.Fatalln("ReadBuffer err:", err)
	}
	resultCount := min(output[0xFF], 0xFF)

	var states []([]uint64)
	for i := 0; i < maxThreads; i++ {
		var state = make([]uint64, 4, 4)
		err = gpu.ReadBuffer(runner, 200 *i, states_buf, state)
		states = append(states, state)
	}

	sort.Slice(states, func(i, j int) bool {
		return states[i][3] < states[j][3]
	})
	for _, state := range states[:resultCount] {
		fmt.Println(state)
	}

	// Show Result
	return output[:resultCount]
}

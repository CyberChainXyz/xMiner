package main

import (
	"fmt"
	cl "github.com/nexis-dev/ccxminer/opencl"
	"github.com/nexis-dev/ccxminer/stratum"
	"log"
	"sync/atomic"
	"time"
)

type Miner struct {
	index           int
	device          *cl.OpenCLDevice
	runner          *cl.OpenCLRunner
	input_buf       *cl.Buffer
	output_buf      *cl.Buffer
	scratchpads_buf *cl.Buffer
	states_buf      *cl.Buffer
	compMode        int
	unroll          int
	workSize        int
	maxThreads      uint64
	hashRate        atomic.Uint64
}

func newMiner(index int, device *cl.OpenCLDevice) (*Miner, error) {
	miner := &Miner{index: index, device: device}
	err := miner.init()
	if err != nil {
		return nil, err
	}
	return miner, nil
}

func (miner *Miner) free() {
	miner.runner.Free()
}

func (miner *Miner) init() error {
	device := miner.device
	runner, err := device.InitRunner()
	if err != nil {
		return fmt.Errorf("InitRunner err: %s, %v", device.Name, err)
	}

	miner.runner = runner

	// kernel constants
	miner.compMode = 0
	miner.unroll = 1
	miner.workSize = min(int(device.Max_work_group_size)/16, 8)
	miner.maxThreads = uint64(device.Max_compute_units * 6 * 8)

	// CompileKernels
	codes := []string{cngpucode}
	kernelNameList := []string{"cn0_cn_gpu", "cn00_cn_gpu", "cn1_cn_gpu", "cn2"}
	options := fmt.Sprintf("-DITERATIONS=%d"+" -DMASK=%dU"+" -DWORKSIZE=%dU"+" -DCOMP_MODE=%d"+" -DMEMORY=%dLU"+" -DCN_UNROLL=%d"+" -cl-fp32-correctly-rounded-divide-sqrt", ITERATIONS, MASK, miner.workSize, miner.compMode, MEMORY, miner.unroll)

	err = runner.CompileKernels(codes, kernelNameList, options)
	if err != nil {
		return fmt.Errorf("CompileKernels err: %s, %v", device.Name, err)
	}

	// create buffers
	g_thd := miner.maxThreads
	var scratchPadSize uint64 = MEMORY
	miner.input_buf, err = runner.CreateEmptyBuffer(cl.READ_ONLY, 128)
	if err != nil {
		return fmt.Errorf("CreateBuffer input_buf err: %s, %v", device.Name, err)
	}
	miner.scratchpads_buf, err = runner.CreateEmptyBuffer(cl.READ_WRITE, int(scratchPadSize*g_thd))
	if err != nil {
		return fmt.Errorf("CreateBuffer scratchpads_buf err: %s, %v", device.Name, err)
	}
	miner.states_buf, err = runner.CreateEmptyBuffer(cl.READ_WRITE, int(200*g_thd))
	if err != nil {
		return fmt.Errorf("CreateBuffer states_buf err: %s, %v", device.Name, err)
	}
	miner.output_buf, err = runner.CreateEmptyBuffer(cl.READ_WRITE, NonceLen*0x100)
	if err != nil {
		return fmt.Errorf("CreateBuffer output_buf err: %s, %v", device.Name, err)
	}
	return nil
}

func (miner *Miner) run(pool *stratum.Pool) {
	for {
		job := pool.LastJob()
		// wait for first job
		if job == nil {
			time.Sleep(time.Second * 3)
			continue
		}
		miner.setJob(job.Input(), job.Target, job.ExtraNonce)

		for pool.LastJob().JobId == job.JobId {
			// If no new job are received within 5 minutes, pause.
			if time.Since(job.ReceiveTime) > time.Minute*5 {
				log.Printf("No new job are received within 5 minutes, miner %d-%s pause!", miner.index, miner.device.Name)
				time.Sleep(5 * time.Second)
				continue
			}
			startNonce := job.GetNonce(miner.maxThreads)

			t := time.Now()
			output, err := miner.runJob(startNonce)
			miner.hashRate.Store(miner.maxThreads * 1000 / uint64(time.Since(t).Milliseconds()))
			if err != nil {
				log.Printf("RunKernel err: %s, %v\n", miner.device.Name, err)
			} else {
				for _, nonce := range output {
					go func(nonce uint32) {
						realNonce := job.ExtraNonce + startNonce + uint64(nonce)

						result, err := pool.SubmitJobWork(job, realNonce)
						if err != nil {
							log.Printf("SubmitJobWork err: %d-%s, 0x%x, %v\n", miner.index, miner.device.Name, realNonce, err)
						} else {
							if result {
								log.Printf("Solutions accepted: %d-%s, 0x%x\n", miner.index, miner.device.Name, realNonce)
							} else {
								log.Printf("Solutions rejected: %d-%s, 0x%x\n", miner.index, miner.device.Name, realNonce)
							}
						}
					}(nonce)
				}
			}
		}
	}
}

func (miner *Miner) setJob(source []byte, target uint64, extraNonce uint64) error {

	runner := miner.runner
	input_buf := miner.input_buf
	output_buf := miner.output_buf
	scratchpads_buf := miner.scratchpads_buf
	states_buf := miner.states_buf

	// input
	input := make([]byte, 128, 128)
	input_len := len(source)
	copy(input[:input_len], source)
	input[input_len] = 0x01
	// numThreads
	numThreads := uint32(miner.maxThreads)

	// set input buffer
	err := cl.WriteBuffer(runner, 0, input_buf, input, true)
	if err != nil {
		return fmt.Errorf("WriteBuffer input_buf err: %v", err)
	}

	// kernel params
	k0_args := []cl.KernelParam{
		cl.BufferParam(input_buf),
		cl.BufferParam(scratchpads_buf),
		cl.BufferParam(states_buf),
		cl.Param(&numThreads),
		cl.Param(&extraNonce),
	}

	// kernel params
	k00_args := []cl.KernelParam{
		cl.BufferParam(scratchpads_buf),
		cl.BufferParam(states_buf),
	}

	// kernel params
	k1_args := []cl.KernelParam{
		cl.BufferParam(scratchpads_buf),
		cl.BufferParam(states_buf),
		cl.Param(&numThreads),
	}

	// kernel params
	k2_args := []cl.KernelParam{
		cl.BufferParam(scratchpads_buf),
		cl.BufferParam(states_buf),
		cl.BufferParam(output_buf),
		cl.Param(&target),
		cl.Param(&numThreads),
	}
	err = runner.SetKernelArgs("cn0_cn_gpu", k0_args)
	if err != nil {
		return fmt.Errorf("SetKernelArgs cn0_cn_gpu err: %v", err)
	}
	err = runner.SetKernelArgs("cn00_cn_gpu", k00_args)
	if err != nil {
		return fmt.Errorf("SetKernelArgs cn00_cn_gpu err: %v", err)
	}
	err = runner.SetKernelArgs("cn1_cn_gpu", k1_args)
	if err != nil {
		return fmt.Errorf("SetKernelArgs cn1_cn_gpu err: %v", err)
	}
	err = runner.SetKernelArgs("cn2", k2_args)
	if err != nil {
		return fmt.Errorf("SetKernelArgs cn2 err: %v", err)
	}
	return nil
}

func (miner *Miner) runJob(startNonce uint64) ([]uint32, error) {

	runner := miner.runner
	output_buf := miner.output_buf

	// output
	var output = make([]uint32, 0x100, 0x100)

	// ===============================================
	// Run kernels loop
	g_intensity := miner.maxThreads
	w_size := uint64(miner.workSize)
	g_thd := g_intensity

	compMode := miner.compMode
	if g_thd%w_size == 0 {
		compMode = 0
	}
	if compMode != 0 {
		// round up to next multiple of w_size
		g_thd = ((g_intensity + w_size - 1) / w_size) * w_size
	}

	err := cl.WriteBuffer(runner, NonceLen*0xFF, output_buf, []uint32{0}, false)
	if err != nil {
		return nil, fmt.Errorf("WriteBuffer output_buf err: %v", err)
	}

	err = runner.RunKernel("cn0_cn_gpu", 2, []uint64{startNonce, 1}, []uint64{g_thd, 8}, []uint64{8, 8}, nil, false)
	if err != nil {
		return nil, fmt.Errorf("RunKernel cn0_cn_gpu err: %v", err)
	}

	err = runner.RunKernel("cn00_cn_gpu", 1, nil, []uint64{g_intensity * 64}, []uint64{64}, nil, false)
	if err != nil {
		return nil, fmt.Errorf("RunKernel cn00_cn_gpu err: %v", err)
	}

	err = runner.RunKernel("cn1_cn_gpu", 1, nil, []uint64{g_thd * 16}, []uint64{w_size * 16}, nil, false)
	if err != nil {
		return nil, fmt.Errorf("RunKernel cn1_cn_gpu err: %v", err)
	}

	err = runner.RunKernel("cn2", 2, []uint64{0, startNonce}, []uint64{8, g_thd}, []uint64{8, w_size}, nil, false)
	if err != nil {
		return nil, fmt.Errorf("RunKernel cn2 err: %v", err)
	}

	// ReadBuffer
	err = cl.ReadBuffer(runner, 0, output_buf, output)
	if err != nil {
		return nil, fmt.Errorf("ReadBuffer output_buf err: %v", err)
	}
	resultCount := min(output[0xFF], 0xFF)
	// return Result
	return output[:resultCount], nil
}

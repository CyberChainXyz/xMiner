package main

import "github.com/dextech/gpu-runner"
import "github.com/kr/pretty"

import "fmt"

func main() {
	gpus := gpurunner.GetAllGpusInfo()
	pretty.Println(gpus)

	err := gpurunner.Run(&gpus.OpenCL.Platforms[0].Devices[0])
	fmt.Println(err)
}

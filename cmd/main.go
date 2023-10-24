package main

import "github.com/dextech/gpu-runner"
import "github.com/kr/pretty"

func main() {
	gpus := gpurunner.GetAllGpusInfo()
	pretty.Println(gpus)
}

package main

import (
	cl "github.com/dextech/gpu-runner/opencl"
	"github.com/kr/pretty"
)

func main() {
	info, _ := cl.Info()
	pretty.Println(info)
}

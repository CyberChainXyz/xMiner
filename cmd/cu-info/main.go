package main

import (
	cu "github.com/dextech/gpu-runner/cuda"
	"github.com/kr/pretty"
)

func main() {
	info, _ := cu.Info()
	pretty.Println(info)
}

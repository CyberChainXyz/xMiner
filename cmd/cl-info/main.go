package main

import (
	"github.com/kr/pretty"
	cl "github.com/nexis-dev/ccxminer/opencl"
)

func main() {
	info, _ := cl.Info()
	pretty.Println(info)
}

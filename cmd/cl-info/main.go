package main

import (
	cl "github.com/nexis-dev/ccxminer/opencl"
	"github.com/kr/pretty"
)

func main() {
	info, _ := cl.Info()
	pretty.Println(info)
}

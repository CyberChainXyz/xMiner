package main

import (
	cu "github.com/nexis-dev/ccxminer/cuda"
	"github.com/kr/pretty"
)

func main() {
	info, _ := cu.Info()
	pretty.Println(info)
}

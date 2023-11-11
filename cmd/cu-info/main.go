package main

import (
	"github.com/kr/pretty"
	cu "github.com/nexis-dev/ccxminer/cuda"
)

func main() {
	info, _ := cu.Info()
	pretty.Println(info)
}

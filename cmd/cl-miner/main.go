package main

import (
	_ "embed"
	"flag"
	cl "github.com/CyberChainXyz/go-opencl"
	stratum "github.com/CyberChainXyz/stratum-jsonrpc2-ws"
	"github.com/kr/pretty"

	"log"
	"os"
	"strings"
	"time"
)

//go:embed fphash.cl
var fphashClCode string

const ITERATIONS = 768
const MEMORY = 32 * 1024
const MASK = ((MEMORY - 1) >> 6) << 6
const NonceLen = 4

var showInfo bool
var mock bool
var all bool
var poolUrl string
var user string
var pass string
var intensity float64

func init() {
	flag.BoolVar(&showInfo, "info", false, "Show all OpenCL device informations and exit.")
	flag.BoolVar(&mock, "mock", false, "Run performance testing.")
	flag.BoolVar(&all, "all", false, "Use all OpenCL devices, otherwise only AMD and NVIDIA GPU cards.")
	flag.StringVar(&poolUrl, "pool", "ws://127.0.0.1:8546", "pool url")
	flag.StringVar(&user, "user", "", "username for pool")
	flag.StringVar(&pass, "pass", "", "password for pool")
	flag.Float64Var(&intensity, "intensity", 1, "Miner intensity factor")
}

func main() {
	flag.Parse()

	// Show all OpenCL device informations and exit
	if showInfo {
		info, _ := cl.Info()
		pretty.Println(info)
		os.Exit(0)
	}

	var pool stratum.PoolIntf
	var err error
	if mock {
		pool = stratum.NewFakeFool()
	} else {
		pool, err = stratum.NewPool(poolUrl, user, pass, "ccxminer")
		if err != nil {
			log.Println("newPool Err:", err)
			return
		}
	}

	log.Printf("Pool connected: %s\n", pool.Url())

	// get all OpenCL devices
	info, _ := cl.Info()
	var devices []*cl.OpenCLDevice
	for _, p := range info.Platforms {
		isAmd := strings.Index(p.Vendor, "Advanced Micro Devices") != -1
		isNvidia := strings.Index(p.Vendor, "NVIDIA Corporation") != -1 || strings.Index(p.Vendor, "NVIDIA") != -1
		if all || isAmd || isNvidia {
			devices = append(devices, p.Devices...)
		}
	}

	// Init miners
	miners := make([]*Miner, len(devices))
	for i, device := range devices {
		miner, err := newMiner(i+1, device, intensity)
		if err != nil {
			log.Printf("init device fail: %v\n", err)
			return
		}
		go miner.run(pool)
		miners[i] = miner
		log.Printf("New miner %d: %s, maxThreads: %d, workSize: %d\n", miner.index, miner.device.Name, miner.maxThreads, miner.workSize)
	}

	if len(miners) == 0 {
		log.Println("no OpenCL devices")
		return
	}

	// show miners hahsRate
	hashRateTick := time.Tick(time.Second * 10)
	for {
		<-hashRateTick
		for _, miner := range miners {
			log.Printf("Miner %d hashRate: %.3f kH", miner.index, float64(miner.hashRate.Load())/1000)
		}
	}
}

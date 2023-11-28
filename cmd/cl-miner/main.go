package main

import (
	_ "embed"
	"flag"
	cl "github.com/nexis-dev/go-opencl"
	stratum "github.com/nexis-dev/stratum-jsonrpc2-ws"
	"log"
	"time"
)

//go:embed cn.cl
var cngpucode string

const ITERATIONS = 768
const MEMORY = 32 * 1024
const MASK = ((MEMORY - 1) >> 6) << 6
const NonceLen = 4

var mock bool
var poolUrl string
var user string
var pass string
var intensity float64

func init() {
	flag.BoolVar(&mock, "mock", false, "run performance testing")
	flag.StringVar(&poolUrl, "pool", "ws://127.0.0.1:8546", "pool url")
	flag.StringVar(&user, "user", "", "username for pool")
	flag.StringVar(&pass, "pass", "", "password for pool")
	flag.Float64Var(&intensity, "intensity", 1, "miner intensity factor")
}

func main() {
	flag.Parse()

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
		devices = append(devices, p.Devices...)
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

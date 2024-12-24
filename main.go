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
var cpuThreads int
var cpuOnly bool

func init() {
	flag.BoolVar(&showInfo, "info", false, "Show all OpenCL device informations and exit.")
	flag.BoolVar(&mock, "mock", false, "Run performance testing.")
	flag.BoolVar(&all, "all", false, "Use all OpenCL devices, otherwise only AMD and NVIDIA GPU cards.")
	flag.StringVar(&poolUrl, "pool", "ws://127.0.0.1:8546", "pool url")
	flag.StringVar(&user, "user", "", "username for pool")
	flag.StringVar(&pass, "pass", "", "password for pool")
	flag.Float64Var(&intensity, "intensity", 1, "Miner intensity factor")
	flag.IntVar(&cpuThreads, "cpu", -1, "Number of CPU threads (0 = number of CPU cores)")
	flag.BoolVar(&cpuOnly, "cpu-only", false, "Use CPU mining only (no GPU)")
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

	var miners []interface{} // Interface to hold both CPU and GPU miners

	// Initialize CPU miner if requested
	if cpuThreads >= 0 && !mock {
		cpuMiner := newCpuMiner(1, cpuThreads)
		miners = append(miners, cpuMiner)
		go cpuMiner.run(pool)
		log.Printf("New CPU miner: threads: %d\n", cpuMiner.threads)
	}

	// Initialize GPU miners if not in CPU-only mode
	if !cpuOnly {
		// Init miners
		for i, device := range devices {
			miner, err := newMiner(i+1, device, intensity)
			if err != nil {
				log.Printf("init device fail: %v\n", err)
				continue
			}
			go miner.run(pool)
			miners = append(miners, miner)
			log.Printf("New GPU miner %d: %s, maxThreads: %d, workSize: %d\n",
				miner.index, miner.device.Name, miner.maxThreads, miner.workSize)
		}
	}

	if len(miners) == 0 {
		log.Println("no mining devices available")
		return
	}

	// Show miners hashRate
	hashRateTick := time.Tick(time.Second * 10)
	for {
		<-hashRateTick
		for _, miner := range miners {
			switch m := miner.(type) {
			case *Miner:
				log.Printf("GPU Miner %d hashRate: %.3f kH",
					m.index, float64(m.hashRate.Load())/1000)
			case *CpuMiner:
				log.Printf("CPU Miner %d hashRate: %.3f kH",
					m.index, float64(m.hashRate.Load())/1000)
			}
		}
	}
}

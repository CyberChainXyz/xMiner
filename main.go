package main

import (
	_ "embed"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	cl "github.com/CyberChainXyz/go-opencl"
	stratum "github.com/CyberChainXyz/stratum-jsonrpc2-ws"
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
var deviceIndices string

func init() {
	flag.BoolVar(&showInfo, "info", false, "Show all OpenCL device informations and exit.")
	flag.BoolVar(&mock, "mock", false, "Run performance testing.")
	flag.BoolVar(&all, "all", false, "Use all OpenCL devices, otherwise only AMD and NVIDIA GPU cards.")
	flag.StringVar(&poolUrl, "pool", "ws://127.0.0.1:8546", "pool url")
	flag.StringVar(&user, "user", "", "username for pool")
	flag.StringVar(&pass, "pass", "", "password for pool")
	flag.Float64Var(&intensity, "intensity", 1, "Miner intensity factor")
	flag.StringVar(&deviceIndices, "devices", "", "Comma-separated list of device indices to use (e.g., '1,3,4'). Empty means use all available devices.")
	flag.StringVar(&deviceIndices, "d", "", "Short for devices")
}

// sortDevicesByPCIInfo sorts devices by Domain, Bus, Device, and Function
func sortDevicesByPCIInfo(devices []*cl.OpenCLDevice) {
	sort.Slice(devices, func(i, j int) bool {
		a := devices[i].PCIInfo
		b := devices[j].PCIInfo

		// Compare Domain
		if a.Domain != b.Domain {
			return a.Domain < b.Domain
		}
		// Compare Bus
		if a.Bus != b.Bus {
			return a.Bus < b.Bus
		}
		// Compare Device
		if a.Device != b.Device {
			return a.Device < b.Device
		}
		// Compare Function
		return a.Function < b.Function
	})
}

// getAllDevices returns all available devices sorted by PCI info
func getAllDevices(info *cl.OpenCLInfo, useAll bool) []*cl.OpenCLDevice {
	var devices []*cl.OpenCLDevice
	for _, p := range info.Platforms {
		isAmd := strings.Contains(p.Vendor, "Advanced Micro Devices")
		isNvidia := strings.Contains(p.Vendor, "NVIDIA Corporation") || strings.Contains(p.Vendor, "NVIDIA")
		if useAll || isAmd || isNvidia {
			devices = append(devices, p.Devices...)
		}
	}
	sortDevicesByPCIInfo(devices)
	return devices
}

// parseDeviceIndices parses the device indices string into a map (using 1-based indexing)
func parseDeviceIndices(indices string) map[int]bool {
	selected := make(map[int]bool)
	if indices == "" {
		return selected
	}

	parts := strings.Split(indices, ",")
	for _, p := range parts {
		if idx, err := strconv.Atoi(strings.TrimSpace(p)); err == nil && idx > 0 {
			selected[idx] = true
		}
	}
	return selected
}

func main() {
	flag.Parse()

	// get all OpenCL devices
	info, _ := cl.Info()
	devices := getAllDevices(info, all)

	// Show all OpenCL device informations and exit
	if showInfo {
		fmt.Println("Available OpenCL devices:")
		for i, d := range devices {
			fmt.Printf("[%d] %s (Vendor: %s)\n", i+1, d.Name, d.Vendor)
		}
		os.Exit(0)
	}

	var pool stratum.PoolIntf
	var err error
	if mock {
		pool = stratum.NewFakeFool()
	} else {
		pool, err = stratum.NewPool(poolUrl, user, pass, "xMiner")
		if err != nil {
			log.Println("newPool Err:", err)
			return
		}
	}

	log.Printf("Pool connected: %s\n", pool.Url())

	// Get selected devices
	selectedDevices := parseDeviceIndices(deviceIndices)

	// Init miners
	var miners []*Miner
	for i, device := range devices {
		deviceIndex := i + 1
		if len(selectedDevices) == 0 || selectedDevices[deviceIndex] {
			miner, err := newMiner(deviceIndex, device, intensity)
			if err != nil {
				log.Printf("Device [%d] %s: Initialized failed - %v\n", deviceIndex, device.Name, err)
				return
			}
			go miner.run(pool)
			miners = append(miners, miner)
			log.Printf("Device [%d] %s: Initialized (maxThreads: %d, workSize: %d)\n",
				miner.index, miner.device.Name, miner.maxThreads, miner.workSize)
		} else {
			log.Printf("Device [%d] %s: Not selected", deviceIndex, device.Name)
		}
	}

	if len(miners) == 0 {
		log.Println("No OpenCL devices for mining")
		return
	}

	// show miners hashRate
	hashRateTick := time.Tick(time.Second * 10)
	for {
		<-hashRateTick
		for _, miner := range miners {
			log.Printf("Device [%d] hashRate: %.3fkH", miner.index, float64(miner.hashRate.Load())/1000)
		}
	}
}

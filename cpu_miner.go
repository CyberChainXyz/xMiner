package main

import (
	"encoding/binary"
	"log"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/CyberChainXyz/fphash-go"
	stratum "github.com/CyberChainXyz/stratum-jsonrpc2-ws"
)

const NONCES_PER_THREAD = 1000

type CpuMiner struct {
	index         int
	threads       int
	hashLoopNum   uint64
	hashLastStamp uint64
	hashRate      atomic.Uint64
}

func newCpuMiner(index int, threads int) *CpuMiner {
	if threads <= 0 {
		threads = runtime.NumCPU()
	}
	return &CpuMiner{
		index:   index,
		threads: threads,
	}
}

func (m *CpuMiner) updateStats() {
	m.hashLoopNum += uint64(m.threads * NONCES_PER_THREAD)

	now := uint64(time.Now().UnixMilli())
	timeDiff := now - m.hashLastStamp
	if timeDiff < 1000 {
		return
	}

	hashRate := m.hashLoopNum * 1000 / timeDiff
	lastHashRate := m.hashRate.Load()
	if lastHashRate == 0 {
		m.hashRate.Store(hashRate)
		m.hashLastStamp = now
		m.hashLoopNum = 0
		return
	}

	averagingBias := uint64(1)
	m.hashRate.Store((lastHashRate*(10-averagingBias) + hashRate*averagingBias) / 10)
	m.hashLastStamp = now
	m.hashLoopNum = 0
}

func (m *CpuMiner) run(pool stratum.PoolIntf) {
	m.hashLastStamp = uint64(time.Now().UnixMilli())
	m.hashLoopNum = 0

	for {
		job := pool.LastJob()
		// wait for first job
		if job == nil {
			time.Sleep(time.Second * 3)
			continue
		}
		job_input := job.Input()

		for pool.LastJob().JobId == job.JobId {
			// If no new job are received within 5 minutes, pause.
			if time.Since(job.ReceiveTime) > time.Minute*5 {
				log.Printf("No new job are received within 5 minutes, CPU miner %d pause!", m.index)
				time.Sleep(5 * time.Second)
				continue
			}

			startNonce := job.GetNonce(uint64(m.threads * NONCES_PER_THREAD))

			// Create channels for result collection
			results := make(chan uint64, m.threads*10)
			done := make(chan bool, m.threads)

			// Start worker goroutines
			for i := 0; i < m.threads; i++ {
				go func(threadID int) {
					threadStartNonce := job.ExtraNonce + startNonce + uint64(threadID*NONCES_PER_THREAD)
					// Copy input and set nonce in last 8 bytes
					hashSource := make([]byte, 72)
					copy(hashSource, job_input[:64])

					// Process NONCES_PER_THREAD consecutive nonces
					for nonce := threadStartNonce; nonce < threadStartNonce+NONCES_PER_THREAD; nonce++ {
						// Append nonce bytes
						nonceBytes := make([]byte, 8)
						binary.BigEndian.PutUint64(nonceBytes, nonce)
						copy(hashSource[len(hashSource)-8:], nonceBytes)

						// Calculate hash using fphash
						hash := fphash.Hash(hashSource)
						// first 8 bytes of hash
						hashBytes := hash[:8]
						hashUint64 := binary.BigEndian.Uint64(hashBytes)
						if hashUint64 <= job.Target {
							results <- nonce
						}
					}

					done <- true
				}(i)
			}

			// Wait for all workers to complete
			for i := 0; i < m.threads; i++ {
				<-done
			}
			close(results)
			close(done)

			m.updateStats()

			if pool.IsFake() {
				continue
			}

			// Submit any found solutions
			for nonce := range results {
				go func(nonce uint64) {
					result, err := pool.SubmitJobWork(job, nonce)
					if err != nil {
						log.Printf("SubmitJobWork err: CPU-%d, 0x%x, %v\n", m.index, nonce, err)
					} else {
						if result {
							log.Printf("Solutions accepted: CPU-%d, 0x%x\n", m.index, nonce)
						} else {
							log.Printf("Solutions rejected: CPU-%d, 0x%x\n", m.index, nonce)
						}
					}
				}(nonce)
			}

		}
	}
}

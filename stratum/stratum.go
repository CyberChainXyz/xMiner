package stratum

import (
	"context"
	"encoding/binary"
	"fmt"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/rpc"
	"log"
	"math"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type Pool struct {
	url     string
	worker  *worker
	client  *rpc.Client
	lastJob atomic.Value
}

type worker struct {
	user  string
	pass  string
	agent string
}

type Job struct {
	ReceiveTime        time.Time
	JobId              string
	PowHash            []byte
	SeedHash           []byte
	Target             uint64
	BlockNumber        uint64
	ExtraNonce         uint64
	ExtraNonceSizeBits int
	workNonce          *atomic.Uint64
}

type PoolIntf interface {
	LastJob() *Job
	Url() string
	SubmitJobWork(*Job, uint64) (bool, error)
}

func (job *Job) Input() []byte {
	input := make([]byte, 72)
	copy(input[:32], job.PowHash)
	copy(input[32:64], job.SeedHash)
	return input
}

func (job *Job) GetNonce(reserveCount uint64) uint64 {
	return job.workNonce.Add(reserveCount) - reserveCount
}

func NewPool(url string, user string, pass string, agent string) (*Pool, error) {
	initCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	client, err := rpc.DialContext(initCtx, url)
	if err != nil {
		return nil, fmt.Errorf("DialContext Err: %v", err)
	}

	subch := make(chan [6]string)
	go func() {
		for {
			subCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			sub, err := client.EthSubscribe(subCtx, subch, "newWork", user, pass, agent)
			cancel()
			if err != nil {
				log.Println("EthSubscribe Err:", err)
				time.Sleep(2 * time.Second)
				continue
			}
			log.Println("EthSubscribe lost, resubscribe: ", <-sub.Err())
		}
	}()

	pool := Pool{
		url:    url,
		client: client,
		worker: &worker{user, pass, agent},
	}

	go func() {
		for job := range subch {
			jobId := job[0]

			if jobId == "" {
				log.Printf(">> Receive invalid job: jobId is empty")
				continue
			}

			powHash, err := hexutil.Decode(job[1])
			if err != nil || len(powHash) != 32 {
				log.Println(">> Receive invalid job: powHash")
				continue
			}
			seedHash, err := hexutil.Decode(job[2])
			if err != nil || len(seedHash) != 32 {
				log.Println(">> Receive invalid job: seedHash")
				continue
			}
			target, err := strconv.ParseUint(job[3][:18], 0, 64)
			if err != nil {
				log.Println(">> Receive invalid job: target")
				continue
			}
			blockNumber, err := hexutil.DecodeUint64(job[4])
			if err != nil {
				log.Println(">> Receive invalid job: blockNumber")
				continue
			}

			var extraNonce uint64
			var extraNonceSizeBits int

			if len(job[5]) > 10 {
				log.Println(">> Receive invalid job: extraNonce")
				continue
			}

			if len(job[5]) == 0 {
				extraNonce = 0
				extraNonceSizeBits = 0
			} else {
				extraNonce, err = strconv.ParseUint(job[5]+strings.Repeat("0", 16-len(job[5])), 16, 64)
				if err != nil || len(job[5]) > 10 {
					log.Println(">> Receive invalid job: extraNonce")
					continue
				}
				extraNonceSizeBits = len(job[5]) * 4
			}

			log.Printf("Receive new job: jobId: %s, blockNumber: %d, difficulty: %d, extraNonce: %s",
				job[0], blockNumber, math.MaxUint64/target+1, job[5])

			var workNonce atomic.Uint64
			workNonce.Store(0)

			poolJob := Job{
				ReceiveTime:        time.Now(),
				JobId:              jobId,
				PowHash:            powHash,
				SeedHash:           seedHash,
				Target:             target,
				BlockNumber:        blockNumber,
				ExtraNonce:         extraNonce,
				ExtraNonceSizeBits: extraNonceSizeBits,
				workNonce:          &workNonce,
			}

			pool.lastJob.Store(&poolJob)

		}
	}()

	return &pool, nil
}

func (pool *Pool) LastJob() *Job {
	v := pool.lastJob.Load()
	if v == nil {
		return nil
	} else {
		return v.(*Job)
	}
}

func (pool *Pool) Url() string {
	return pool.url
}

func (pool *Pool) SubmitJobWork(job *Job, nonce uint64) (bool, error) {
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, nonce)
	nonceHex := hexutil.Encode(nonceBytes)

	powHashHex := hexutil.Encode(job.PowHash)
	return pool.submitWork(job.JobId, nonceHex, powHashHex)
}

func (pool *Pool) submitWork(jobId string, nonce string, powHash string) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var result bool
	err := pool.client.CallContext(ctx, &result, "eth_submitWork", jobId, nonce, powHash)
	return result, err
}

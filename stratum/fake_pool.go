package stratum

import (
	"github.com/ethereum/go-ethereum/common/hexutil"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

type FakePool struct {
	url string
}

func NewFakeFool() *FakePool {
	return &FakePool{url: "FakePool"}
}

func (pool *FakePool) LastJob() *Job {
	job := [6]string{"jobid", "0x57a811dba04d6870dd844d59e09f8b9c9058fcbfa7765ffe5492af778dbdc9fd", "0x0000000000000000000000000000000000000000000000000000000000000000", "0x0000800000000000000000000000000000000000000000000000000000000000", "0xe5", "e49f"}

	jobId := job[0]
	powHash, _ := hexutil.Decode(job[0])
	seedHash, _ := hexutil.Decode(job[1])
	target, _ := strconv.ParseUint(job[3][:18], 0, 64)
	blockNumber, _ := hexutil.DecodeUint64(job[4])
	extraNonce, _ := strconv.ParseUint(job[5]+strings.Repeat("0", 16-len(job[5])), 16, 64)
	extraNonceSizeBits := len(job[5]) * 4
	var workNonce atomic.Uint64
	workNonce.Store(0)

	return &Job{
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

}

func (pool *FakePool) Url() string {
	return pool.url
}

func (pool *FakePool) IsFake() bool {
	return true
}

func (pool *FakePool) SubmitJobWork(job *Job, nonce uint64) (bool, error) {
	return true, nil
}

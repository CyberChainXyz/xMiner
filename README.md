# CCXminer

Cryptonight-GPU-light miner for [CyberChain](https://github.com/cyberchain/ccx).


The OpenCL and CUDA kernel code originates from [xmr-stak](https://github.com/fireice-uk/xmr-stak/tree/master/xmrstak/backend)


## Development Status

**WARNING**: This project is currently under development and has not been fully tested. Use it at your own risk. We welcome any feedback and contributions.

## Features

- [x] cl-miner: OpenCL miner
- [ ] cu-miner: CUDA miner


## cl-miner

### Download
```
TODO
```

### Performance Testing

```bash
./cl-miner -mock
```

### Solo mining

**Run a [CyberChain](https://github.com/cyberchain/ccx) client with mining feature enabled**
```
ccx -ws -mine -miner.etherbase=0x123...fff
```
Replace 0x123...fff with your own address.

**Run the miner**
```bash
./cl-miner
```
The default pool address is ws://127.0.0.1:8546.

### Pool Mining
```bash
./cl-miner -user=username -pass=password -pool=wss://pool-address.com:3333
```
Replace `user, pass, pool` with your actual values provided by the mining pool.

### Compile from source

**Requirements**

linux
```bash
sudo apt install ocl-icd-opencl-dev opencl-headers
```

**Clone the repository and build the miner**
```bash
git clone https://github.com/cyberchain/ccxminer)
cd ccxminer
go build ./cmd/cl-miner/
```

## cu-miner

TODO

## Other resources

[go-opencl](https://github.com/nexis-dev/go-opencl): high-level Go interface to the OpenCL API.

[go-cuda](https://github.com/nexis-dev/go-cuda): high-level Go interface to the CUDA API.

# xMiner

Open source [fphash](https://github.com/CyberChainXyz/fphash-go) miner in Go with [stratum-jsonrpc2.0-ws](https://github.com/CyberChainXyz/stratum-jsonrpc2-ws) protocol.


The OpenCL kernel code originates from [xmr-stak](https://github.com/fireice-uk/xmr-stak/tree/master/xmrstak/backend).


## Performance Testing

```bash
./xMiner -mock -all
```

## Show OpenCL devices informations

```bash
./xMiner -info -all
```

## Solo mining

**Run a [CyberChain](https://github.com/CyberChainXyz/go-cyberchain) client with mining feature enabled**
```
./ccx -ws -mine -miner.etherbase=0x123...fff
```
Replace 0x123...fff with your own address.

**Run the miner**
```bash
./xMiner
```
The default pool address is ws://127.0.0.1:8546.

## Pool Mining
```bash
./xMiner -user=username -pass=password -pool=wss://pool-address.com:port
```
Replace `user, pass, pool` with your actual values provided by the mining pool.

## Compile from source

**Requirements**

linux
```bash
sudo apt install ocl-icd-opencl-dev opencl-headers
```

**Clone the repository and build the miner**
```bash
git clone https://github.com/CyberChainXyz/xMiner
cd xMiner
go build .
```

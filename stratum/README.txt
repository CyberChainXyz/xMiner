stratum-jsonrpc2.0-ws: Stratum JSON-RPC-2.0 over Websocket

JSON-RPC 2.0: https://www.jsonrpc.org/specification

Subscribe:

client: {"id":1,"jsonrpc":"2.0","method":"eth_subscribe","params":["newWork","0xadadfa==============","skdfadf","clksdfadadfa"]}
server: {"jsonrpc":"2.0","id":1,"result":"0x6245a28462ff2753df66800e89c7fdb5"}

Job notification:

{"jsonrpc":"2.0","method":"eth_subscription","params":{"subscription":"0xee571162a864f5cc659e3f392aef4732","result":["4orr4","0xd9dcd16aec94afca6ccb4e7da9b733946afb9b4d47004215326c5c5ceeab7279","0x0000000000000000000000000000000000000000000000000000000000000000","0x0000800000000000000000000000000000000000000000000000000000000000","0x4d","86a3"]}}

Submit solution:

client: {"id":2,"jsonrpc":"2.0","method":"eth_submitWork","params":["jobId","0x90989098111111aa","0x9867fe999999999999999867fe999999999999999867fe999999999999991234"]}
server: {"jsonrpc":"2.0","id":2,"result":true}


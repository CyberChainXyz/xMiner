package main

import (
	"bufio"
	"crypto/tls"
	"flag"
	"fmt"
	"golang.org/x/net/proxy"
	"net"
	"net/url"
	"strings"
	"time"
	"encoding/json"
)

var poolURI = flag.String("pool", "", "Pool URI: scheme://user[.workername][:password]@hostname:port")
var proxyURI = flag.String("proxy", "", "Proxy: socks5/socks5h://[user][:password]@hostname:port")

type Pool struct {
	host    string
	user    *url.Userinfo
	getwork bool
	tls     bool
	proxy   *proxy.Dialer
}

type Message struct {
	method string `json:"method"`
	params []string    `json:"params"`
	id int `json:"id"`
	jsonrpc string `json:"jsonrpc"`
}

func main() {
	flag.Parse()

	pool := new(Pool)

	poolUrl, err := url.Parse(*poolURI)
	if err != nil {
		fmt.Println("Invalid pool URI:", err)
		return
	}

	if poolUrl.Scheme == "" || poolUrl.Host == "" {
		fmt.Println("Invalid pool URI")
		return
	}

	if poolUrl.Scheme == "getwork" || poolUrl.Scheme == "http" {
		pool.getwork = true
	} else if poolUrl.Scheme == "stratums" || poolUrl.Scheme == "stratumss" {
		pool.tls = true
	} else if !strings.HasPrefix(poolUrl.Scheme, "stratum") {
		fmt.Println("Invalid pool URI:", "invalid scheme")
		return
	} else {
		if strings.Contains(poolUrl.Scheme, "+tls") || strings.Contains(poolUrl.Scheme, "+ssl") {
			pool.tls = true
		}
	}
	pool.host = poolUrl.Host
	pool.user = poolUrl.User
	
	if *proxyURI != "" {
		proxyUrl, err := url.Parse(*proxyURI)
		if err != nil {
			fmt.Println("Invalid proxy URI:", err)
			return
		}

		if (proxyUrl.Scheme != "socks5" && proxyUrl.Scheme != "socks5h") || poolUrl.Host == "" {
			fmt.Println("Invalid proxy URI")
			return
		}

		proxy, err := proxy.FromURL(proxyUrl, &net.Dialer{Timeout: time.Second * 50})
		if err != nil {
			fmt.Println("Invalid proxy URI:", err)
			return
		}
		pool.proxy = &proxy
	}


	serverName := poolUrl.Hostname()

	fmt.Printf("%#v, %s\n", pool, serverName)

	// connect
	var conn net.Conn
	if pool.proxy != nil {
		_conn, err := (*pool.proxy).Dial("tcp", pool.host)
		if err != nil {
			fmt.Println("Connect err:", err)
			return
		}
		if pool.tls {
			_conn = tls.Client(_conn, &tls.Config{ServerName:serverName})
		}
		conn = _conn
	} else {
		_conn, err := net.DialTimeout("tcp", pool.host, time.Second * 50)
		if err != nil {
			fmt.Println("Connect err:", err)
			return
		}
		if pool.tls {
			_conn = tls.Client(_conn, &tls.Config{ServerName:serverName})
		}
		conn = _conn
	}
	defer conn.Close()

	// x := bufio.NewReadWriter( bufio.NewReader(conn), bufio.NewWriter(conn))
	x := bufio.NewReader(conn)


	conn.Write([]byte("GET / HTTP/1.2\r\nHost: www.baidu.com\r\nAccept-Language: en-US,en;q=0.5\r\n\r\n"))
	line, err := x.ReadString('\n')
	fmt.Println("read http:", line, err)

	sub_msg := Message{
		id: 1,
		method: "mining.subscribe",
		params: []string{"gominer", "EthereumStratum/1.0.0"},
	}
	msg, err := json.Marshal(sub_msg)
	nn, err := conn.Write(msg)
	fmt.Println("write 1", nn, err)
	nn, err = conn.Write([]byte("\n"))
	fmt.Println("write 11", nn, err)

	line, err = x.ReadString('\n')
	fmt.Println("readline", line, err)
	
	auth_msg := Message {
		id: 2,
		method: "mining.authorize",
		params: []string{"user","pass"},
	}
	msg, err = json.Marshal(auth_msg)
	nn, err = conn.Write(msg)
	fmt.Println("write 2", nn, err)
	nn, err = conn.Write([]byte("\n"))
	fmt.Println("write 22", nn, err)

	extra_msg := Message {
		id: 3,
		method: "mining.extranonce.subscribe",
		params: []string{},
	}
	msg, err = json.Marshal(extra_msg)
	nn, err = conn.Write(msg)
	fmt.Println("write 3", nn, err)
	nn, err = conn.Write([]byte("\n"))
	fmt.Println("write 33", nn, err)

	for {
		line, err = x.ReadString('\n')
		fmt.Println("readline", line, err)
		if err != nil {
			break
		}
	}

}

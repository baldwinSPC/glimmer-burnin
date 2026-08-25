// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"time"
)

// The control channel: a minimal TCP handshake between the two halves of a
// Pair, used ONLY to let the SERVER learn its peer's real address.
//
// # Why this exists
//
// selectPort (rdma.go) picks the local RDMA device by the route to the peer,
// and on a node with more than one usable RDMA-capable address that route is
// the only thing that tells the two ends of a link apart from two ends of
// DIFFERENT links. The CLIENT can resolve its peer from BURNIN_PEER_HOST
// directly, because the server (its peer) already exists and answers DNS by
// the time the client starts. The SERVER cannot: it starts first, and
// in-cluster its own peer's DNS name — the client's — does not resolve until
// the operator creates the client pod, which the operator will not do until
// the server reports Ready. Resolving BURNIN_PEER_HOST from the server was
// therefore not a transient race, it was a lookup against a name that could
// not exist yet, and it silently fell back to a DIFFERENT selection rule
// ("first ACTIVE port") than the one the client used successfully — on a node
// with more than one usable rail, the two ends could and did pick different
// ports (#484, #489).
//
// This is the same problem runners/ib-write-bw/rendezvous.go solves, with the
// same answer: the server learns its peer from the accepted connection
// (conn.RemoteAddr), which is the address the peer's packets actually arrive
// from rather than one resolved in advance. This is a deliberately TRIMMED
// fork of that file, not a shared copy or a port of the whole thing — see
// fabric_contract_test.go's own reasoning for the fabric trio, and #484's
// tracking issue for why fabric-soak did not already share it. fabric-soak
// needs none of ib-write-bw's phase negotiation (bandwidth then latency, over
// the SAME connection): every window here is its own independent ib_write_bw
// invocation that the wrapper already orchestrates, so this channel only ever
// carries one message, one way — a hello — and nothing else.
//
// # Why it is also the readiness port
//
// A Pair server should expose a readinessProbe, and perftest's own port is the
// wrong target: ib_write_bw accepts a BOUNDED number of connections, so a
// kubelet tcpSocket probe against it can consume the slot the real client
// needed — the same mistake this project's own e2e suite made once against a
// plain nc -l server. This listener is the runner's own, bound before the
// first ib_write_bw invocation, and it tolerates a probe connection that never
// sends a hello by design.

const protocolVersion = 1

// message is the one thing this channel ever says.
type message struct {
	Version int    `json:"version"`
	Kind    string `json:"kind"`
}

// kindHello is the only message kind this channel needs: the client
// identifying itself so the server can read conn.RemoteAddr and know who it
// is talking to. There is no phase negotiation to acknowledge and nothing to
// abort — a handshake that fails is reported by its caller, not over the wire.
const kindHello = "hello"

// channel is one end of the control connection.
type channel struct {
	conn net.Conn
	r    *bufio.Reader
}

func newChannel(c net.Conn) *channel { return &channel{conn: c, r: bufio.NewReader(c)} }

func (c *channel) Close() error { return c.conn.Close() }

// send writes one message. The deadline is short because the channel carries
// nothing large and a stalled write means the peer is gone.
func (c *channel) send(m message) error {
	m.Version = protocolVersion
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if err := c.conn.SetWriteDeadline(time.Now().Add(30 * time.Second)); err != nil {
		return err
	}
	_, err = c.conn.Write(append(b, '\n'))
	return err
}

// recv reads one message, waiting up to timeout.
func (c *channel) recv(timeout time.Duration) (message, error) {
	if err := c.conn.SetReadDeadline(time.Now().Add(timeout)); err != nil {
		return message{}, err
	}
	line, err := c.r.ReadBytes('\n')
	if err != nil {
		return message{}, err
	}
	var m message
	if err := json.Unmarshal(line, &m); err != nil {
		return message{}, fmt.Errorf("undecodable control message: %w", err)
	}
	return m, nil
}

// peerIP is the address the other end of this connection is at — authoritative,
// because it is the address the peer's packets actually arrive from, not one
// it claimed.
func (c *channel) peerIP() net.IP {
	if a, ok := c.conn.RemoteAddr().(*net.TCPAddr); ok {
		return a.IP
	}
	return nil
}

// awaitPeer waits on the readiness listener for a connection that identifies
// itself as this runner's client, and returns the address it connected from.
// The listener itself is left OPEN on return: the soak has not started yet,
// and the readinessProbe must keep answering for the rest of this pod's life —
// see tolerateProbes, which the caller starts once this returns.
//
// Connections that are NOT the client are expected and are not errors. The
// operator does not create the client pod at all until this probe first
// succeeds, so every connection before the real client shows up is a
// kubelet readinessProbe (or, off-cluster, a port scan) — each is read with a
// short deadline and dropped, and the loop keeps waiting for the real peer.
func awaitPeer(ln *net.TCPListener, within time.Duration, log func(string, ...any)) (net.IP, error) {
	deadline := time.Now().Add(within)
	probes := 0
	for {
		if err := ln.SetDeadline(deadline); err != nil {
			return nil, err
		}
		conn, err := ln.Accept()
		if err != nil {
			return nil, fmt.Errorf("no client connected within %s: %w", within, err)
		}
		ch := newChannel(conn)
		m, err := ch.recv(5 * time.Second)
		if err != nil || m.Kind != kindHello {
			_ = ch.Close()
			probes++
			if probes%10 == 0 {
				log("fabric-soak: still waiting for the client; %d non-peer connections so far "+
					"(readiness probes look like this)", probes)
			}
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("no client identified itself within %s", within)
			}
			continue
		}
		peerIP := ch.peerIP()
		_ = ch.Close()
		return peerIP, nil
	}
}

// tolerateProbes keeps accepting and immediately closing every connection to
// ln for the rest of the process's life, so the kubelet's readinessProbe keeps
// succeeding after the one real handshake awaitPeer already consumed. Meant to
// run in its own goroutine; it returns (and so exits) only once ln itself is
// closed, which happens when the process does.
func tolerateProbes(ln *net.TCPListener) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		_ = conn.Close()
	}
}

// dialPeer connects to the server's readiness listener and says hello,
// retrying until it answers.
//
// The retry is not belt-and-braces. DNS for the peer's name is populated by
// the endpoint controller and can lag the pod by a moment, and a client that
// resolved once and gave up would report a fabric fault about a name that was
// simply not published yet.
func dialPeer(host string, port int, within time.Duration, log func(string, ...any)) error {
	deadline := time.Now().Add(within)
	addr := net.JoinHostPort(host, fmt.Sprint(port))
	var lastErr error
	for attempt := 1; ; attempt++ {
		conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
		if err == nil {
			ch := newChannel(conn)
			sendErr := ch.send(message{Kind: kindHello})
			_ = ch.Close()
			if sendErr == nil {
				return nil
			}
			lastErr = sendErr
		} else {
			lastErr = err
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("could not reach the %s endpoint at %s within %s: %w", pairRoleServer, addr, within, lastErr)
		}
		if attempt == 1 || attempt%10 == 0 {
			log("fabric-soak: waiting for the %s endpoint at %s (attempt %d): %v", pairRoleServer, addr, attempt, lastErr)
		}
		time.Sleep(time.Second)
	}
}

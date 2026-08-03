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

// The control channel: one TCP connection between the two halves of a Pair.
//
// # Why a runner-owned channel exists at all
//
// The operator's rendezvous contract gives each pod its role and its peer's DNS
// name, and deliberately nothing else. Two things this runner needs cannot come
// from there, and both are why this file exists rather than a sleep and a retry
// loop:
//
//  1. WHICH LOCAL RDMA DEVICE TO USE. Port numbering is not symmetric across a
//     cable — on the pair of DGX Sparks this was developed on, one end of the
//     link is roceP2p1s0f1 and the other end of the SAME link is
//     roceP2p1s0f0 — so each side must pick the device that routes to its peer.
//     The client can do that from BURNIN_PEER_HOST. The SERVER cannot: it starts
//     first, and its peer's DNS name does not resolve until the operator creates
//     the client, which the operator will not do until the server is Ready. That
//     is a deadlock, and the way out is that the server learns its peer's
//     address from the accepted connection itself (conn.RemoteAddr).
//
//  2. AGREEMENT ON PARAMETERS. perftest requires the two ends to be started with
//     matching message size, queue-pair count and mode; mismatch it and the run
//     dies with "Failed to complete run_iter_bw function successfully", which
//     reads exactly like a broken fabric. Sending the parameters over this
//     channel makes a mismatch impossible rather than merely unlikely.
//
// # Why it is also the readiness port
//
// A Pair server should expose a readinessProbe, and the obvious target —
// perftest's own out-of-band port — is the wrong one: perftest listens with a
// BACKLOG OF ONE and treats the next connection as its client, so a kubelet
// tcpSocket probe would be accepted as the peer, feed it garbage, and break the
// test it was meant to guard. This listener is the runner's own, it is bound
// before perftest is started, and it tolerates probe connections that say
// nothing (they are read with a short deadline and dropped). A tcpSocket probe
// against it is therefore a true statement that this end is ready to be
// rendezvous'd with.

const protocolVersion = 1

// message is one line of the control channel. It is a single flat struct rather
// than a union so that an older peer's message still decodes: unknown fields are
// ignored and absent ones are zero, which keeps a version skew from becoming a
// parse error that looks like a fabric fault.
type message struct {
	Version int    `json:"version"`
	Kind    string `json:"kind"`
	Phase   string `json:"phase,omitempty"`
	Port    int    `json:"port,omitempty"`
	Bytes   int    `json:"bytes,omitempty"`
	QPs     int    `json:"qps,omitempty"`
	Seconds int    `json:"seconds,omitempty"`
	Iters   int    `json:"iters,omitempty"`
	Reason  string `json:"reason,omitempty"`
	// Memlock is the sender's RLIMIT_MEMLOCK soft limit in bytes. The client
	// reports it in its hello so the SERVER can size a plan that fits BOTH ends:
	// the two pods can be given different limits, and a plan that fits only the
	// side that chose it fails on the other with an error that names neither the
	// limit nor the peer.
	Memlock uint64 `json:"memlock,omitempty"`
}

const (
	kindHello = "hello" // client -> server, identifies a real peer
	kindPhase = "phase" // server -> client, "start this test now"
	kindDone  = "done"  // client -> server, "that phase finished"
	kindEnd   = "end"   // server -> client, "no more phases"
	kindAbort = "abort" // either way, with a reason
)

const (
	phaseBandwidth = "bw"
	phaseLatency   = "lat"
)

// channel is one end of the control connection.
type channel struct {
	conn net.Conn
	r    *bufio.Reader
	// peerHello is the client's opening message, kept because it carries the
	// peer's memlock budget and the server must plan against it.
	peerHello message
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

// recv reads one message, waiting up to timeout. Callers pass a timeout long
// enough to cover the peer RUNNING a phase, because the next message after a
// phase start does not arrive until the phase is over.
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

// peerIP is the address the other end of this connection is at. This is the
// value the server cannot get any other way, and it is authoritative: it is the
// address the peer's packets actually arrive from, not one it claimed.
func (c *channel) peerIP() net.IP {
	if a, ok := c.conn.RemoteAddr().(*net.TCPAddr); ok {
		return a.IP
	}
	return nil
}

// acceptPeer waits on the readiness listener for a connection that identifies
// itself as this runner's client.
//
// Connections that are NOT the client are expected and are not errors: the
// kubelet's readinessProbe opens one and closes it immediately, and it does that
// on a period for as long as the pod lives. Each is read with a short deadline
// and dropped, and the loop keeps waiting for the real peer.
func acceptPeer(ln net.Listener, within time.Duration, log func(string, ...any)) (*channel, error) {
	deadline := time.Now().Add(within)
	probes := 0
	for {
		if err := ln.(*net.TCPListener).SetDeadline(deadline); err != nil {
			return nil, err
		}
		conn, err := ln.Accept()
		if err != nil {
			return nil, fmt.Errorf("no client connected within %s: %w", within, err)
		}
		ch := newChannel(conn)
		m, err := ch.recv(5 * time.Second)
		if err != nil || m.Kind != kindHello {
			// A readiness probe, a port scan, or a peer that spoke too late.
			_ = ch.Close()
			probes++
			if probes%10 == 0 {
				log("still waiting for the client; %d non-peer connections so far (readiness probes look like this)", probes)
			}
			if time.Now().After(deadline) {
				return nil, fmt.Errorf("no client identified itself within %s", within)
			}
			continue
		}
		ch.peerHello = m
		if m.Version != protocolVersion {
			_ = ch.send(message{Kind: kindAbort, Reason: fmt.Sprintf(
				"control protocol version %d, peer speaks %d", protocolVersion, m.Version)})
			_ = ch.Close()
			return nil, fmt.Errorf("peer speaks control protocol version %d, this runner speaks %d", m.Version, protocolVersion)
		}
		return ch, nil
	}
}

// dialPeer connects to the server's readiness listener, retrying until it
// answers.
//
// The retry is not belt-and-braces. DNS for the peer's name is populated by the
// endpoint controller and can lag the pod by a moment, and a client that
// resolved once and gave up would report a fabric fault about a name that was
// simply not published yet.
func dialPeer(host string, port int, hello message, within time.Duration, log func(string, ...any)) (*channel, error) {
	deadline := time.Now().Add(within)
	addr := net.JoinHostPort(host, fmt.Sprint(port))
	var lastErr error
	for attempt := 1; ; attempt++ {
		conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
		if err == nil {
			ch := newChannel(conn)
			hello.Kind = kindHello
			if err := ch.send(hello); err == nil {
				return ch, nil
			}
			lastErr = err
			_ = ch.Close()
		} else {
			lastErr = err
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("could not reach the %s endpoint at %s within %s: %w", pairRoleServer, addr, within, lastErr)
		}
		if attempt == 1 || attempt%10 == 0 {
			log("waiting for the %s endpoint at %s (attempt %d): %v", pairRoleServer, addr, attempt, lastErr)
		}
		time.Sleep(time.Second)
	}
}

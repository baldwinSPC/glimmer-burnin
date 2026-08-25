// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"net"
	"strconv"
	"strings"
	"testing"
	"time"
)

func discardLog(string, ...any) {}

// TestAwaitPeer_LearnsThePeerFromTheAcceptedConnection is the regression test
// for #484/#489: the server must learn its peer's address from the accepted
// connection, never from resolving BURNIN_PEER_HOST, because that name cannot
// resolve before the client pod exists.
func TestAwaitPeer_LearnsThePeerFromTheAcceptedConnection(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	tln := ln.(*net.TCPListener)
	port := tln.Addr().(*net.TCPAddr).Port

	type result struct {
		ip  net.IP
		err error
	}
	got := make(chan result, 1)
	go func() {
		ip, err := awaitPeer(tln, 5*time.Second, discardLog)
		got <- result{ip, err}
	}()

	if err := dialPeer("127.0.0.1", port, 5*time.Second, discardLog); err != nil {
		t.Fatalf("dialPeer: %v", err)
	}

	select {
	case r := <-got:
		if r.err != nil {
			t.Fatalf("awaitPeer: %v", r.err)
		}
		if r.ip == nil || !r.ip.IsLoopback() {
			t.Errorf("peerIP = %v, want a loopback address (the dialer's own)", r.ip)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("awaitPeer never returned")
	}
}

// TestAwaitPeer_DropsConnectionsWithNoHelloAndKeepsWaiting asserts the
// probe-tolerance the readinessProbe depends on: a connection that never
// sends a hello (a bare TCP connect-then-close, exactly what a kubelet
// tcpSocket probe does) must not be mistaken for the real client, and must
// not stop awaitPeer from waiting for one that actually is.
func TestAwaitPeer_DropsConnectionsWithNoHelloAndKeepsWaiting(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	tln := ln.(*net.TCPListener)
	port := tln.Addr().(*net.TCPAddr).Port

	type result struct {
		ip  net.IP
		err error
	}
	got := make(chan result, 1)
	go func() {
		ip, err := awaitPeer(tln, 5*time.Second, discardLog)
		got <- result{ip, err}
	}()

	// A bare probe: connect, send nothing, close — three times, the way a
	// kubelet's tcpSocket probe would across several periods.
	for i := 0; i < 3; i++ {
		c, err := net.Dial("tcp", "127.0.0.1:"+strconv.Itoa(port))
		if err != nil {
			t.Fatalf("probe dial %d: %v", i, err)
		}
		_ = c.Close()
	}

	select {
	case r := <-got:
		t.Fatalf("awaitPeer returned after bare probes alone (ip=%v err=%v) — it must keep waiting", r.ip, r.err)
	case <-time.After(200 * time.Millisecond):
		// Still waiting, as it should be.
	}

	if err := dialPeer("127.0.0.1", port, 5*time.Second, discardLog); err != nil {
		t.Fatalf("dialPeer: %v", err)
	}

	select {
	case r := <-got:
		if r.err != nil {
			t.Fatalf("awaitPeer: %v", r.err)
		}
		if r.ip == nil || !r.ip.IsLoopback() {
			t.Errorf("peerIP = %v, want a loopback address", r.ip)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("awaitPeer never returned after the real hello")
	}
}

// TestAwaitPeer_TimesOutIfNoRealPeerEverConnects — if the client never shows
// up at all, this must fail closed as an Error, not hang the pod forever or
// silently proceed with no peer.
func TestAwaitPeer_TimesOutIfNoRealPeerEverConnects(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer ln.Close()
	tln := ln.(*net.TCPListener)

	start := time.Now()
	_, err = awaitPeer(tln, 200*time.Millisecond, discardLog)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("awaitPeer returned no error when no client ever connected")
	}
	if !strings.Contains(err.Error(), "no client") {
		t.Errorf("error = %v, want it to name the missing client", err)
	}
	if elapsed > 2*time.Second {
		t.Errorf("awaitPeer took %s to time out against a 200ms budget", elapsed)
	}
}

// TestDialPeer_RetriesUntilTheListenerAnswers mirrors the real skew this
// exists for: the endpoint controller populating DNS can lag the pod by a
// moment. Here the analogous lag is the listener not existing yet when the
// first dial is attempted — dialPeer must retry rather than give up on the
// first refused connection.
func TestDialPeer_RetriesUntilTheListenerAnswers(t *testing.T) {
	// Reserve a port, then close it immediately so nothing is listening —
	// dialPeer's first attempt(s) must see a refused connection.
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserving a port: %v", err)
	}
	port := probe.Addr().(*net.TCPAddr).Port
	_ = probe.Close()

	go func() {
		time.Sleep(300 * time.Millisecond)
		ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(port))
		if err != nil {
			return // the test below will time out and report the real failure
		}
		defer ln.Close()
		tln := ln.(*net.TCPListener)
		_, _ = awaitPeer(tln, 5*time.Second, discardLog)
	}()

	if err := dialPeer("127.0.0.1", port, 5*time.Second, discardLog); err != nil {
		t.Fatalf("dialPeer gave up instead of retrying: %v", err)
	}
}

// TestTolerateProbes_KeepsAcceptingAfterTheHandshake asserts the second half
// of the readinessProbe story: once the one real handshake is done,
// subsequent connections (the kubelet's probe, on its own period, for the
// rest of the pod's life) must keep succeeding rather than finding a listener
// that stopped accepting.
func TestTolerateProbes_KeepsAcceptingAfterTheHandshake(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	tln := ln.(*net.TCPListener)
	port := tln.Addr().(*net.TCPAddr).Port
	defer tln.Close()

	done := make(chan struct{})
	go func() {
		_, _ = awaitPeer(tln, 5*time.Second, discardLog)
		close(done)
	}()
	if err := dialPeer("127.0.0.1", port, 5*time.Second, discardLog); err != nil {
		t.Fatalf("dialPeer: %v", err)
	}
	<-done

	go tolerateProbes(tln)

	for i := 0; i < 3; i++ {
		c, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), time.Second)
		if err != nil {
			t.Fatalf("probe connection %d after handshake: %v", i, err)
		}
		_ = c.Close()
	}
}

// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"bufio"
	"net"
	"strconv"
	"testing"
	"time"
)

// ── classifyHandshake: the decision, isolated from any real socket ─────────

func TestClassifyHandshakeUsesTheConnectionsOwnAddress(t *testing.T) {
	// The whole reason #482 exists: the server has no peer address to
	// pre-classify, so it has to learn which of ITS OWN interfaces a real
	// connection landed on. Here that lands on the fabric interface.
	ifaces := []hostIface{{Name: "fab0", Addrs: []net.IP{net.ParseIP("10.0.0.5")}}}
	local := &net.TCPAddr{IP: net.ParseIP("10.0.0.5")}

	got := classifyHandshake(local, "", ifaces, "eth0", netnsHost)
	if got.decision != routeOK {
		t.Fatalf("decision = %v, want routeOK: %+v", got.decision, got)
	}
	if got.iface != "fab0" {
		t.Errorf("iface = %q, want fab0", got.iface)
	}
}

func TestClassifyHandshakeCatchesTheManagementPathFromTheConnectionItself(t *testing.T) {
	// A connection that happened to land on the interface carrying the
	// default route — a single-NIC node reaching the server the only way it
	// can. This is the case #482's server could not previously detect at
	// all: it had no peer address to compare against anything.
	ifaces := []hostIface{{Name: "wlan0", Addrs: []net.IP{net.ParseIP("192.168.1.9")}}}
	local := &net.TCPAddr{IP: net.ParseIP("192.168.1.9")}

	got := classifyHandshake(local, "", ifaces, "wlan0", netnsHost)
	if got.decision != routeIsManagement {
		t.Fatalf("decision = %v, want routeIsManagement: %+v", got.decision, got)
	}
}

func TestClassifyHandshakeExplicitOverrideIgnoresTheConnectionsAddress(t *testing.T) {
	// The operator's TCP_BASELINE_INTERFACE still wins, unchanged from before
	// #482 — even though the connection that arrived landed on a DIFFERENT,
	// perfectly good interface. Naming the management interface explicitly
	// is still the harsher, Error-not-Skip outcome (see classifyRoute).
	ifaces := []hostIface{{Name: "fab0", Addrs: []net.IP{net.ParseIP("10.0.0.5")}}}
	local := &net.TCPAddr{IP: net.ParseIP("10.0.0.5")}

	got := classifyHandshake(local, "wlan0", ifaces, "wlan0", netnsHost)
	if got.decision != routeUnknown {
		t.Fatalf("decision = %v, want routeUnknown (the author named the management interface): %+v",
			got.decision, got)
	}
}

func TestClassifyHandshakeWithNoLocalAddressFailsClosed(t *testing.T) {
	// Should never happen in practice (readHello only returns isHello=true
	// alongside a real *net.TCPAddr), but a nil local address must still
	// resolve to "cannot classify" rather than a zero-value interface name
	// that happens to compare unequal to mgmt and silently passes.
	ifaces := []hostIface{{Name: "fab0", Addrs: []net.IP{net.ParseIP("10.0.0.5")}}}

	got := classifyHandshake(nil, "", ifaces, "eth0", netnsHost)
	if got.decision != routeUnknown {
		t.Fatalf("decision = %v, want routeUnknown with no address to classify", got.decision)
	}
}

// ── the wire protocol, against real loopback sockets ────────────────────────

// freePort grabs an OS-assigned port and releases it immediately, the same
// pattern main_test.go already uses for waitForListener's own tests.
func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()
	return port
}

// Every test below drives runGuardListener directly with synthetic
// mgmt/ns/explicit inputs rather than through serveGuard, for the same reason
// classifyRoute's own tests do not go through guardPath: /proc/net/route does
// not exist on the machine most of these tests run on. serveGuard's own
// OS-reading half is four lines of glue over already-tested primitives
// (os.ReadFile, defaultRouteIface, os.Readlink, netnsFromLinks) and is not
// itself given a direct test, matching guardPath's existing precedent.

func TestHandshakeEndToEndOnAFabricLikeAddress(t *testing.T) {
	port := freePort(t)
	done := make(chan guardResult, 1)
	errCh := make(chan error, 1)
	go func() {
		v, err := runGuardListener(port, 5*time.Second, "", "wlan0", netnsHost)
		done <- v
		errCh <- err
	}()

	v, err := dialGuard("127.0.0.1", port, 5*time.Second)
	if err != nil {
		t.Fatalf("dialGuard: %v", err)
	}
	if !v.ok {
		t.Fatalf("verdict = %+v, want ok (a loopback connection's own address does not carry the "+
			"synthetic management interface wlan0)", v)
	}

	if serverResult, err := <-done, <-errCh; err != nil {
		t.Fatalf("runGuardListener returned an error: %v", err)
	} else if serverResult.decision != routeOK {
		t.Errorf("server-side decision = %v, want routeOK", serverResult.decision)
	}
}

func TestHandshakeRelaysARefusalToTheClient(t *testing.T) {
	// The explicit override IS the management interface — classifyRoute's
	// "the author named it" branch, deterministic regardless of what
	// interfaces this test machine actually has.
	port := freePort(t)
	go func() { _, _ = runGuardListener(port, 5*time.Second, "wlan0", "wlan0", netnsHost) }()

	v, err := dialGuard("127.0.0.1", port, 5*time.Second)
	if err != nil {
		t.Fatalf("dialGuard: %v", err)
	}
	if v.ok {
		t.Fatal("verdict = ok, want a refusal: the explicit interface named this node's own default route")
	}
	if v.reason == "" {
		t.Error("a refusal with no reason leaves nothing for the client to report")
	}
}

func TestServeGuardIgnoresABareProbeConnection(t *testing.T) {
	// A kubelet tcpSocket probe connects and disconnects without sending
	// anything. The guard must not spend its one real classification on
	// that — it has to keep listening for the actual peer.
	port := freePort(t)
	resultCh := make(chan guardResult, 1)
	go func() {
		v, _ := runGuardListener(port, 5*time.Second, "", "wlan0", netnsHost)
		resultCh <- v
	}()

	// The listener's own goroutine start-up is otherwise a race; wait for it
	// exactly as a real client would (waitForListener, from main.go).
	if err := waitForListener("127.0.0.1", port, 2*time.Second); err != nil {
		t.Fatalf("guard listener never came up: %v", err)
	}

	// The bare probe: connect, then close without writing anything.
	probe, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)), 2*time.Second)
	if err != nil {
		t.Fatalf("probe dial: %v", err)
	}
	probe.Close()

	// The real client, after the probe.
	v, err := dialGuard("127.0.0.1", port, 5*time.Second)
	if err != nil {
		t.Fatalf("dialGuard after a bare probe: %v", err)
	}
	if !v.ok {
		t.Fatalf("verdict = %+v, want ok", v)
	}
	<-resultCh
}

func TestServeGuardTimesOutRatherThanHangingForever(t *testing.T) {
	port := freePort(t)
	start := time.Now()
	_, err := runGuardListener(port, 500*time.Millisecond, "", "wlan0", netnsHost)
	if err == nil {
		t.Fatal("runGuardListener returned no error after its deadline with no peer ever connecting")
	}
	if elapsed := time.Since(start); elapsed > 3*time.Second {
		t.Errorf("took %s to give up after a 500ms budget", elapsed)
	}
}

func TestDialGuardRetriesAcrossTheOrderingGap(t *testing.T) {
	// Mirrors TestAClientThatStartsFirstWaitsInsteadOfDying in main_test.go:
	// the client must not give up just because nothing is listening on its
	// very first attempt.
	port := freePort(t) // nothing listening yet

	ready := make(chan struct{})
	go func() {
		time.Sleep(300 * time.Millisecond)
		close(ready)
		_, _ = runGuardListener(port, 5*time.Second, "", "wlan0", netnsHost)
	}()

	start := time.Now()
	v, err := dialGuard("127.0.0.1", port, 10*time.Second)
	if err != nil {
		t.Fatalf("dialGuard: %v", err)
	}
	if time.Since(start) < 200*time.Millisecond {
		t.Error("returned before the server could possibly have been listening")
	}
	if !v.ok {
		t.Errorf("verdict = %+v, want ok", v)
	}
	<-ready
}

func TestReadHelloRejectsAnythingThatIsNotExactlyHello(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()

	go func() {
		w := client
		_, _ = w.Write([]byte("not-hello\n"))
		client.Close()
	}()

	if _, ok := readHello(server); ok {
		t.Error("an arbitrary line was accepted as the guard hello")
	}
}

func TestReadHelloAcceptsExactlyHello(t *testing.T) {
	server, client := net.Pipe()
	defer server.Close()

	go func() {
		_, _ = client.Write([]byte(guardHello + "\n"))
		client.Close()
	}()

	if _, ok := readHello(server); !ok {
		t.Error("the real guard hello was not recognised")
	}
}

func TestRespondGuardWireFormatRoundTrips(t *testing.T) {
	cases := []struct {
		name   string
		result guardResult
		want   guardVerdict
	}{
		{"ok", guardResult{decision: routeOK}, guardVerdict{ok: true}},
		{"skip", guardResult{decision: routeIsManagement, reason: "single-NIC node"},
			guardVerdict{skip: true, reason: "single-NIC node"}},
		{"unknown", guardResult{decision: routeUnknown, reason: "could not tell"},
			guardVerdict{reason: "could not tell"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			server, client := net.Pipe()
			go func() {
				respondGuard(server, c.result)
				server.Close()
			}()

			line, err := bufio.NewReader(client).ReadString('\n')
			client.Close()
			if err != nil {
				t.Fatalf("reading the response: %v", err)
			}
			got, perr := parseGuardLine(line)
			if perr != nil {
				t.Fatalf("%v (line %q)", perr, line)
			}
			if got != c.want {
				t.Errorf("got %+v, want %+v (line %q)", got, c.want, line)
			}
		})
	}
}

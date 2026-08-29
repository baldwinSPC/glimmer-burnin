// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"bufio"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

// The accept-then-classify handshake — see #482.
//
// guardPath (in main.go) classifies a POD's own route to its peer, and that
// works for the client: it has a resolvable BURNIN_PEER_HOST from the moment
// it starts. It does not work for the server, structurally: at Pair scope the
// operator does not create the client pod — and so its DNS record does not
// exist — until the server is Ready, so the server's peer is a name with
// nothing behind it at the point it would need to classify anything.
//
// What the server CAN do is wait for its real peer to connect and read which
// of its OWN interfaces that connection actually landed on — a fact the
// kernel can only hand over once a connection exists. nccl's server already
// relies on the identical move for the identical structural reason (see
// rendezvous.go there: "learns its peer from the accepted connection").
//
// guardHello is what tells a real client apart from Kubernetes' own
// readinessProbe, which the guard port must also tolerate: a tcpSocket probe
// only opens and closes a connection, it never speaks. Anything that is not
// exactly this line is treated as a probe (or noise) and ignored, and the
// listener goes back to accepting rather than spending its one real
// classification on it.
const guardHello = "HELLO"

// guardVerdict is what a client learns from the handshake: whether to
// proceed, and — when not — the server's own reason, so a refusal here reads
// as the guard decision it is rather than as a listener that never appeared.
type guardVerdict struct {
	ok     bool
	skip   bool // meaningful only when !ok: Skip vs Error
	reason string
}

// serveGuard binds guardPort immediately, before anything else — this is what
// the server's readinessProbe now names, precisely so the operator can create
// the client pod before iperf3 has anything to say. iperf3's own port does
// not open until this returns an OK verdict.
//
// It accepts connections until a real client speaks guardHello, classifies
// the path using THAT connection's own local address (or the operator's
// TCP_BASELINE_INTERFACE override, unchanged from before #482), answers the
// client with the verdict, and returns it so the caller can start iperf3 or
// refuse. Every non-hello connection — a health probe, or a stray connect —
// is closed and ignored rather than treated as the peer; only a real hello
// can end the wait, which is what keeps a probe from being able to consume
// the one classification this pod will ever make.
func serveGuard(guardPort int, within time.Duration) (guardResult, error) {
	route, err := os.ReadFile("/proc/net/route")
	if err != nil {
		return guardResult{}, fmt.Errorf("could not read /proc/net/route (%v), so the management-path guard "+
			"cannot run. This image is Linux-only and needs the host network namespace; a burn-in must never "+
			"saturate the interface the cluster is managed through, so it refuses rather than guesses", err)
	}
	mgmt := defaultRouteIface(string(route))

	// Established once, up front — see netnsKind (#285). The routing table
	// above describes whatever namespace we are in, and in a pod that is the
	// pod's.
	selfNS, _ := os.Readlink("/proc/self/ns/net")
	pid1NS, _ := os.Readlink("/proc/1/ns/net")
	ns := netnsFromLinks(selfNS, pid1NS)
	if selfNS != "" {
		metric("tcpNetNamespace", selfNS)
	}

	// The author's own override, unchanged in spirit from before #482: still
	// judged more harshly on a management-interface match (see
	// classifyRoute), and now also skips waiting on the connection's own
	// address to know which interface to classify — but the handshake still
	// happens, so the client always has exactly one thing to wait for
	// regardless of which way the server reached its answer.
	explicit := strings.TrimSpace(os.Getenv("TCP_BASELINE_INTERFACE"))

	return runGuardListener(guardPort, within, explicit, mgmt, ns)
}

// runGuardListener is serveGuard's socket loop, taking what serveGuard would
// otherwise read from /proc/net/route and /proc/self,1/ns/net as plain
// parameters instead. That is what keeps it testable on a laptop: this image
// is Linux-only and /proc/net/route does not exist on the machine most of
// these tests run on, the same reason classifyRoute (mgmtguard.go) takes its
// inputs rather than reading them.
func runGuardListener(guardPort int, within time.Duration, explicit, mgmt string, ns netnsKind) (guardResult, error) {
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", guardPort))
	if err != nil {
		return guardResult{}, fmt.Errorf("could not bind the guard rendezvous port :%d (%v): the "+
			"accept-then-classify handshake needs it to learn which interface the peer actually reaches "+
			"this node through", guardPort, err)
	}
	defer ln.Close()

	deadline := time.Now().Add(within)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return guardResult{}, fmt.Errorf("no client completed the guard handshake on :%d within %s — "+
				"the guard cannot classify a path with no peer, and refuses rather than binding blind",
				guardPort, within)
		}
		if tl, ok := ln.(*net.TCPListener); ok {
			_ = tl.SetDeadline(time.Now().Add(remaining))
		}
		conn, aerr := ln.Accept()
		if aerr != nil {
			if ne, ok := aerr.(net.Error); ok && ne.Timeout() {
				continue // re-check the overall deadline above
			}
			return guardResult{}, fmt.Errorf("accepting on the guard port :%d: %w", guardPort, aerr)
		}

		local, isHello := readHello(conn)
		if !isHello {
			conn.Close()
			continue // a health probe or stray connection; keep waiting for the real peer
		}

		result := classifyHandshake(local, explicit, hostIfaces(), mgmt, ns)

		// Recorded on every path, including the refusals: the guard's
		// decision is only auditable if the interfaces are in the result.
		if result.iface != "" {
			metric("tcpTestInterface", result.iface)
		}
		if result.mgmtIface != "" {
			metric("tcpMgmtInterface", result.mgmtIface)
		}

		respondGuard(conn, result)
		conn.Close()
		return result, nil
	}
}

// classifyHandshake turns an accepted connection's own local address (or the
// operator's explicit override) into a guard decision, by the same rule
// classifyRoute already applies to a client's route lookup — see
// mgmtguard.go. Factored out so the decision is testable the way every other
// rule in this guard is: synchronously, against inputs a test controls
// entirely, rather than only through a real accepted socket.
func classifyHandshake(local *net.TCPAddr, explicit string, ifaces []hostIface, mgmt string, ns netnsKind) guardResult {
	var peerIface string
	switch {
	case explicit != "":
		peerIface = explicit
	case local != nil:
		peerIface = ifaceOwning(local.IP, ifaces)
	}
	return classifyRoute(peerIface, mgmt, explicit != "", ns)
}

// readHello reads exactly one line with a short deadline and reports whether
// it was the client's hello, along with the local address the connection
// arrived on — the fact the whole handshake exists to obtain. A probe that
// connects and disconnects without sending anything, or any unrecognised
// line, comes back false: it was seen, but it was not the peer.
func readHello(conn net.Conn) (*net.TCPAddr, bool) {
	_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil || strings.TrimSpace(line) != guardHello {
		return nil, false
	}
	local, _ := conn.LocalAddr().(*net.TCPAddr)
	return local, true
}

// respondGuard writes the verdict a dialGuard call on the other end reads.
func respondGuard(conn net.Conn, result guardResult) {
	_ = conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	switch result.decision {
	case routeIsManagement:
		fmt.Fprintf(conn, "SKIP:%s\n", result.reason)
	case routeUnknown:
		fmt.Fprintf(conn, "ERROR:%s\n", result.reason)
	default:
		fmt.Fprintln(conn, "OK")
	}
}

// dialGuard is the client side: say hello, retry across the ordering gap the
// server's own pod-creation dependency should already close (the operator
// will not create this pod until the server's guard port is bound and Ready
// — but a retry costs nothing and removes the assumption, the same way
// waitForListener does not trust iperf3's own start-up ordering below), and
// read what the server decided about ITS OWN path — the half of the guard
// this client cannot establish any other way.
func dialGuard(host string, guardPort int, within time.Duration) (guardVerdict, error) {
	deadline := time.Now().Add(within)
	addr := net.JoinHostPort(host, strconv.Itoa(guardPort))

	var lastErr error
	for attempt := 1; ; attempt++ {
		v, err := tryGuardHandshake(addr)
		if err == nil {
			if attempt > 1 {
				logf("tcp-baseline: guard handshake with %s succeeded after %d attempt(s)", addr, attempt)
			}
			return v, nil
		}
		lastErr = err
		if time.Now().After(deadline) {
			return guardVerdict{}, lastErr
		}
		time.Sleep(time.Second)
	}
}

func tryGuardHandshake(addr string) (guardVerdict, error) {
	conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
	if err != nil {
		return guardVerdict{}, err
	}
	defer conn.Close()

	_ = conn.SetDeadline(time.Now().Add(15 * time.Second))
	if _, err := fmt.Fprintf(conn, "%s\n", guardHello); err != nil {
		return guardVerdict{}, fmt.Errorf("sending the guard hello: %w", err)
	}
	line, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		return guardVerdict{}, fmt.Errorf("reading the guard verdict: %w", err)
	}
	return parseGuardLine(line)
}

// parseGuardLine reads the one line respondGuard writes. Factored out from
// tryGuardHandshake so a test can verify the wire format round-trips through
// the SAME parser the real client uses, rather than a reimplementation of it.
func parseGuardLine(line string) (guardVerdict, error) {
	line = strings.TrimSpace(line)
	switch {
	case line == "OK":
		return guardVerdict{ok: true}, nil
	case strings.HasPrefix(line, "SKIP:"):
		return guardVerdict{skip: true, reason: strings.TrimPrefix(line, "SKIP:")}, nil
	case strings.HasPrefix(line, "ERROR:"):
		return guardVerdict{reason: strings.TrimPrefix(line, "ERROR:")}, nil
	default:
		return guardVerdict{}, fmt.Errorf("unrecognised guard verdict %q", line)
	}
}

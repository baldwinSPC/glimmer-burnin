// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"net"
	"strconv"
	"strings"
)

// The management-path guard.
//
// GEP-0178 establishes it for exactly this test, and it is the one rule here
// that is not about measurement quality: a burn-in must never saturate the
// interface the cluster is managed through. Every other runner in this suite
// loads a device. This one loads a NETWORK, and the failure mode of getting it
// wrong is losing control of the fleet you are testing — kubelet heartbeats
// missed, nodes marked NotReady, pods evicted fleet-wide, in the middle of an
// acceptance run whose whole purpose was to avoid surprises.
//
// A saturated management path also invalidates the measurement it was taken
// for: contention with control-plane traffic is not a property of the link.
//
// The guard is deliberately hard to opt out of. There is no --force, because
// the only honest reason to want one is "I am confident this is fine", and the
// cases where an operator is confident and wrong are precisely the ones that
// cost a fleet.

// routeDecision is what the guard concluded, and it maps directly onto the
// runner contract's three non-pass outcomes.
type routeDecision int

const (
	// routeOK: the path to the peer is not the management path.
	routeOK routeDecision = iota
	// routeIsManagement: the ONLY path to the peer is the management
	// interface. A SKIP — this node genuinely cannot run this test safely, and
	// that is a fact about the hardware, not a fault in it. A single-NIC node
	// is the ordinary case and must not be reported as broken.
	routeIsManagement
	// routeUnknown: the guard could not tell. An ERROR, and it FAILS CLOSED:
	// not knowing which interface is the management one is not permission to
	// find out by saturating it.
	routeUnknown
)

// guardResult carries the decision and the sentence a human needs.
type guardResult struct {
	decision  routeDecision
	iface     string
	mgmtIface string
	reason    string
}

// netnsKind is what could be POSITIVELY established about this process's
// network namespace. It exists because a one-NIC node and a pod namespace are
// indistinguishable from inside — each has exactly one non-loopback interface,
// which is also that namespace's default route — and the guard owes them
// opposite verdicts (#285).
//
// Getting that wrong is a fleet-wide fail-open rather than a cosmetic error. A
// profile that merely forgot `hostNetwork: true` had every node report
// tcp-baseline as a declared Skip: never retried, not counted against the run,
// the fabric never measured, and the acceptance reported clean. That is the
// outcome the `*_SKIP`-marker rule exists to prevent, reached through the guard
// instead of through a crash.
type netnsKind int

const (
	// netnsUnknown: nothing could be established. Fails closed.
	netnsUnknown netnsKind = iota
	// netnsHost: this IS the host's network namespace, so what the guard can
	// see is what the node has.
	netnsHost
	// netnsNotHost: definitively a different namespace from PID 1's.
	netnsNotHost
)

// initialNetnsLink is what /proc/self/ns/net reads as in the host's network
// namespace. The initial namespace's inode is fixed at boot on every mainstream
// kernel.
//
// This is CONVENTION and not kernel ABI, which is precisely why it is used only
// as a POSITIVE signal. If a kernel ever numbered the initial namespace
// differently, netnsFromLinks reports Unknown and the guard errors where it
// would have skipped: a node reported unjudged and retried, which is
// survivable. Trusting the constant in the other direction would let a pod
// namespace pass itself off as a single-NIC node, which is the bug being fixed.
const initialNetnsLink = "net:[4026531840]"

// netnsFromLinks classifies the namespace from two /proc readlink results.
//
// It takes the resolved links rather than reading them, so the reasoning is
// testable on a laptop, in CI and inside a container — none of which can
// arrange all the cases it has to get right.
//
// selfLink is /proc/self/ns/net; pid1Link is /proc/1/ns/net, which is evidence
// ONLY when the pod shares the host's PID namespace. When it does not, PID 1 is
// the container's own init: the two links agree while proving nothing, so a
// match is deliberately not treated as establishing the host namespace.
func netnsFromLinks(selfLink, pid1Link string) netnsKind {
	if selfLink == "" {
		return netnsUnknown
	}
	if pid1Link != "" && pid1Link != selfLink {
		// A different namespace from init's. Conclusive, and the only
		// conclusive negative available.
		return netnsNotHost
	}
	if selfLink == initialNetnsLink {
		return netnsHost
	}
	return netnsUnknown
}

// classifyRoute decides whether traffic to peer would cross the management path.
//
// The default route's interface IS the management path, for this purpose. It is
// how the node reaches the apiserver, how the kubelet is talked to, and how an
// operator SSHes in to find out why everything is on fire. A cluster whose
// control plane is reachable only through some non-default route is possible in
// principle, but naming the default route is the conservative reading — it can
// only ever refuse a test that would have been safe, never permit one that
// would not.
//
// Both inputs are supplied by the caller rather than read here, which is what
// keeps this function pure and testable on a laptop with one interface.
func classifyRoute(peerIface, mgmtIface string, explicit bool, ns netnsKind) guardResult {
	switch {
	case mgmtIface == "":
		return guardResult{
			decision: routeUnknown,
			reason: "could not determine this node's default route, so there is no way to tell the " +
				"management interface from a fabric one. Refusing to generate load rather than guess: " +
				"saturating the management path costs the cluster, not just this test",
		}

	case peerIface == "":
		return guardResult{
			decision:  routeUnknown,
			mgmtIface: mgmtIface,
			reason: "could not determine which interface reaches the peer, so it cannot be compared " +
				"against the management interface (" + mgmtIface + ")",
		}

	case peerIface == mgmtIface && explicit:
		// The author NAMED this interface. That is a mistake worth reporting as
		// one rather than skipping past: a profile that asks for the management
		// path wants a different profile, and a silent skip would leave the
		// author believing the test ran somewhere.
		return guardResult{
			decision:  routeUnknown,
			iface:     peerIface,
			mgmtIface: mgmtIface,
			reason: fmt.Sprintf(
				"the interface named for this test (%s) is this node's management interface — the one carrying "+
					"its default route. Point the test at a fabric address instead; there is no override, because "+
					"the cost of being wrong is the cluster and not the test",
				peerIface),
		}

	case peerIface == mgmtIface && ns != netnsHost:
		// The same observation, from somewhere it cannot mean what it says.
		//
		// Inside a pod namespace there is exactly one non-loopback interface
		// and it carries that namespace's default route — identical to a
		// one-NIC node, and nothing about the NODE follows from it. The
		// default route seen here is the POD's, not the machine's.
		//
		// So this is not a declared skip. "Only one interface is visible" is
		// not a declaration that this is a one-NIC node; it is an absence of
		// evidence, and the project's rule is that absence is not a
		// declaration. Error is the honest answer: the guard could not
		// classify the path, the hardware is unjudged, and it is retryable —
		// where a Skip would be permanent, silent, and fleet-wide (#285).
		return guardResult{
			decision:  routeUnknown,
			iface:     peerIface,
			mgmtIface: mgmtIface,
			reason: fmt.Sprintf(
				"the only visible route to the peer is %s, which also carries the default route — but this "+
					"process is not in the host's network namespace, so that describes the POD's routing and "+
					"says nothing about the node's. A one-NIC node and a pod namespace look identical from "+
					"inside. Set hostNetwork: true so the test can see the node's real interfaces; refusing "+
					"rather than reporting this node as not-applicable, which would leave the fabric unmeasured",
				peerIface),
		}

	case peerIface == mgmtIface:
		// No second path exists, and we are looking at the node's own
		// interfaces. Not a fault: a single-NIC node cannot run this test
		// safely, which is exactly what a declared skip is for.
		return guardResult{
			decision:  routeIsManagement,
			iface:     peerIface,
			mgmtIface: mgmtIface,
			reason: fmt.Sprintf(
				"the only route to the peer is %s, which carries this node's default route and is therefore its "+
					"management path. A node with one interface cannot be load-tested without loading the path the "+
					"cluster is managed through",
				peerIface),
		}
	}

	return guardResult{decision: routeOK, iface: peerIface, mgmtIface: mgmtIface}
}

// defaultRouteIface parses /proc/net/route for the interface carrying the
// default route.
//
// Takes the file's CONTENT rather than reading it, so the parsing is testable
// against real captured tables — including the shapes that matter: multiple
// defaults with different metrics, a down interface still listed, and the
// header line.
//
// Linux-only by construction, which is what this runner ships on. It returns
// "" rather than an error when there is simply no default route; the caller
// turns that into the fail-closed Error, because "no default route" and "I
// could not read the table" both mean the same thing here — the guard cannot
// do its job.
func defaultRouteIface(procNetRoute string) string {
	const (
		fieldIface  = 0
		fieldDest   = 1
		fieldFlags  = 3
		fieldMetric = 6

		// From linux/route.h. A default route is UP and is a GATEWAY.
		rtfUp      = 0x0001
		rtfGateway = 0x0002
	)

	best := ""
	bestMetric := int64(-1)

	for _, line := range strings.Split(procNetRoute, "\n") {
		fields := strings.Fields(line)
		if len(fields) <= fieldMetric {
			continue
		}
		// The header line, and anything else non-numeric, falls out here.
		if fields[fieldDest] != "00000000" {
			continue
		}
		flags, err := strconv.ParseInt(fields[fieldFlags], 16, 64)
		if err != nil || flags&rtfUp == 0 || flags&rtfGateway == 0 {
			continue
		}
		metric, err := strconv.ParseInt(fields[fieldMetric], 10, 64)
		if err != nil {
			continue
		}
		// Lowest metric wins, which is the kernel's own rule. Picking the first
		// listed instead would name the wrong interface on any node with a
		// backup default route — and naming the wrong one is worse than naming
		// none, because it would let the guard pass traffic onto the real
		// management path while reporting that it had checked.
		if bestMetric < 0 || metric < bestMetric {
			best, bestMetric = fields[fieldIface], metric
		}
	}
	return best
}

// ifaceForAddr reports which interface owns the local address the kernel would
// use to reach a peer.
//
// The UDP "connection" sends nothing — it only asks the kernel to run its route
// lookup and bind a source address — so this is a routing question answered by
// the routing table, not a probe of the peer. That matters: the peer may not be
// up yet, and a guard that needed it to be would refuse to protect the cluster
// exactly when the run is starting.
func ifaceForAddr(peer string, ifaces []hostIface) string {
	conn, err := net.Dial("udp", net.JoinHostPort(peer, "9"))
	if err != nil {
		return ""
	}
	defer conn.Close()

	local, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok || local.IP == nil {
		return ""
	}
	return ifaceOwning(local.IP, ifaces)
}

// hostIface is an interface reduced to what the guard needs.
//
// A plain struct rather than net.Interface because net.Interface.Addrs() reads
// the live host: the matching below is where the edge cases are, and it has to
// be testable on a machine whose real interfaces are nothing like a node's.
type hostIface struct {
	Name  string
	Addrs []net.IP
}

// ifaceOwning finds the interface holding an address.
func ifaceOwning(ip net.IP, ifaces []hostIface) string {
	for _, iface := range ifaces {
		for _, candidate := range iface.Addrs {
			if candidate != nil && candidate.Equal(ip) {
				return iface.Name
			}
		}
	}
	return ""
}

// hostIfaces reads the live interfaces into the shape above.
func hostIfaces() []hostIface {
	raw, err := net.Interfaces()
	if err != nil {
		return nil
	}
	out := make([]hostIface, 0, len(raw))
	for _, iface := range raw {
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		h := hostIface{Name: iface.Name}
		for _, addr := range addrs {
			switch a := addr.(type) {
			case *net.IPNet:
				h.Addrs = append(h.Addrs, a.IP)
			case *net.IPAddr:
				h.Addrs = append(h.Addrs, a.IP)
			}
		}
		out = append(out, h)
	}
	return out
}

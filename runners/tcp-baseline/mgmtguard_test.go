// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"net"
	"strings"
	"testing"
)

// A real table from a two-NIC node: eth0 carries the default route, enp1s0f0 is
// the fabric and has no gateway.
const twoNIC = `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
eth0	00000000	0101A8C0	0003	0	0	100	00000000	0	0	0
eth0	0001A8C0	00000000	0001	0	0	100	00FFFFFF	0	0	0
enp1s0f0	0000000A	00000000	0001	0	0	0	0000FFFF	0	0	0
`

func TestTheDefaultRouteIsFoundInARealTable(t *testing.T) {
	if got := defaultRouteIface(twoNIC); got != "eth0" {
		t.Errorf("default route = %q, want eth0", got)
	}
}

func TestTheLOWESTMetricDefaultWins(t *testing.T) {
	// A node with a backup default route. Taking the first listed would name
	// the wrong interface — worse than naming none, because the guard would
	// then pass traffic onto the real management path while reporting that it
	// had checked.
	table := `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
wwan0	00000000	0101A8C0	0003	0	0	700	00000000	0	0	0
eth0	00000000	0101A8C0	0003	0	0	100	00000000	0	0	0
`
	if got := defaultRouteIface(table); got != "eth0" {
		t.Errorf("default route = %q, want eth0 (metric 100 beats 700)", got)
	}
}

func TestARouteThatIsNotADefaultIsNotOne(t *testing.T) {
	rows := []struct {
		name  string
		table string
	}{
		{"only the header", "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"},
		{"empty", ""},
		{"no default, only a link route", `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
enp1s0f0	0000000A	00000000	0001	0	0	0	0000FFFF	0	0	0
`},
		{"a default that is DOWN", `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
eth0	00000000	0101A8C0	0002	0	0	100	00000000	0	0	0
`},
		{"a default with no gateway flag", `Iface	Destination	Gateway 	Flags	RefCnt	Use	Metric	Mask		MTU	Window	IRTT
eth0	00000000	00000000	0001	0	0	100	00000000	0	0	0
`},
		{"truncated garbage", "eth0 00000000\n"},
	}
	for _, r := range rows {
		if got := defaultRouteIface(r.table); got != "" {
			t.Errorf("%s: got %q, want no default route", r.name, got)
		}
	}
}

func TestTheGuardFailsClosedWhenItCannotTell(t *testing.T) {
	// Not knowing which interface is the management one is not permission to
	// find out by saturating it.
	rows := []struct {
		name      string
		peerIface string
		mgmtIface string
		want      string
	}{
		{"no default route at all", "enp1s0f0", "", "default route"},
		{"peer interface unknown", "", "eth0", "reaches the peer"},
	}
	for _, r := range rows {
		got := classifyRoute(r.peerIface, r.mgmtIface, false, netnsHost)
		if got.decision != routeUnknown {
			t.Errorf("%s: decision = %v, want routeUnknown", r.name, got.decision)
		}
		if !strings.Contains(got.reason, r.want) {
			t.Errorf("%s: reason %q does not explain itself", r.name, got.reason)
		}
	}
}

func TestASingleNICNodeSkipsAndIsNotCondemned(t *testing.T) {
	// The ordinary case, and it must not read as a fault. A node with one
	// interface cannot be load-tested safely, which is a fact about the
	// hardware — exactly what a declared skip is for.
	//
	// netnsHost is load-bearing rather than boilerplate: this conclusion is
	// only available once we know the interfaces being described are the
	// NODE's. The same inputs from a pod namespace are #285's fail-open, and
	// are asserted below.
	got := classifyRoute("eth0", "eth0", false, netnsHost)
	if got.decision != routeIsManagement {
		t.Fatalf("decision = %v, want routeIsManagement", got.decision)
	}
	if !strings.Contains(got.reason, "one interface") {
		t.Errorf("the reason should explain why this node cannot run it: %q", got.reason)
	}
}

func TestNamingTheManagementInterfaceIsAnErrorNotASkip(t *testing.T) {
	// The author NAMED it. A silent skip would leave them believing the test
	// ran somewhere; the distinction between "this node can't" and "you asked
	// for the wrong thing" is the whole Error/Skip discipline.
	got := classifyRoute("eth0", "eth0", true, netnsHost)
	if got.decision != routeUnknown {
		t.Fatalf("decision = %v, want routeUnknown (an Error)", got.decision)
	}
	if !strings.Contains(got.reason, "no override") {
		t.Errorf("the reason should say there is no override: %q", got.reason)
	}
}

func TestAFabricPathIsAllowed(t *testing.T) {
	got := classifyRoute("enp1s0f0", "eth0", false, netnsHost)
	if got.decision != routeOK {
		t.Fatalf("decision = %v, want routeOK — a separate fabric interface is the case this test exists for", got.decision)
	}
	if got.iface != "enp1s0f0" || got.mgmtIface != "eth0" {
		t.Errorf("the result should record both interfaces: %+v", got)
	}
}

func TestTheOwningInterfaceIsFoundByAddress(t *testing.T) {
	ifaces := []hostIface{
		{Name: "lo", Addrs: []net.IP{net.ParseIP("127.0.0.1")}},
		{Name: "eth0", Addrs: []net.IP{net.ParseIP("192.168.1.20")}},
		{Name: "enp1s0f0", Addrs: []net.IP{net.ParseIP("10.0.0.11"), net.ParseIP("fe80::1")}},
	}
	rows := []struct {
		ip   string
		want string
	}{
		{"10.0.0.11", "enp1s0f0"},
		{"fe80::1", "enp1s0f0"},
		{"192.168.1.20", "eth0"},
		{"10.9.9.9", ""}, // owned by nothing here
	}
	for _, r := range rows {
		if got := ifaceOwning(net.ParseIP(r.ip), ifaces); got != r.want {
			t.Errorf("%s → %q, want %q", r.ip, got, r.want)
		}
	}
}

func TestAnIPv4MappedAddressStillMatches(t *testing.T) {
	// net.IP.Equal treats 4-byte and 16-byte forms of the same v4 address as
	// equal, and the kernel hands back either depending on the socket. Asserted
	// rather than assumed: if this stopped holding, the guard would fail to
	// identify the interface and — correctly but uselessly — refuse every run.
	ifaces := []hostIface{{Name: "eth0", Addrs: []net.IP{net.ParseIP("192.168.1.20").To4()}}}
	if got := ifaceOwning(net.ParseIP("192.168.1.20").To16(), ifaces); got != "eth0" {
		t.Errorf("a 16-byte form of a v4 address did not match its 4-byte form: got %q", got)
	}
}

// THE OBSERVED FAIL-OPEN (#285).
//
// Found on real hardware: run without --network host, the guard saw exactly one
// non-loopback interface which was also the default route, concluded
// "single-NIC node", and exited 2 with TCP_BASELINE_SKIP.
//
//	exit=2  tcpTestInterface=eth0  tcpMgmtInterface=eth0
//	TCP_BASELINE_SKIP: the only route to the peer is eth0, which carries this
//	node's default route and is therefore its management path...
//
// The inputs are identical to a genuine one-NIC node — that is the whole
// problem — so the namespace is the only thing that can separate them. A Skip
// here is never retried and does not count against the run, so a profile that
// merely forgot `hostNetwork: true` reported an entire fleet as not-applicable
// with the fabric unmeasured and the acceptance clean.
func TestAPodNamespaceIsUnjudgedRatherThanASingleNICNode(t *testing.T) {
	for _, ns := range []struct {
		name string
		kind netnsKind
	}{
		{"not the host namespace", netnsNotHost},
		{"namespace could not be established", netnsUnknown},
	} {
		got := classifyRoute("eth0", "eth0", false, ns.kind)
		if got.decision != routeUnknown {
			t.Errorf("%s: decision = %v, want routeUnknown (an Error): the fabric was never measured, "+
				"and a Skip would say it does not apply", ns.name, got.decision)
		}
		if !strings.Contains(got.reason, "hostNetwork") {
			t.Errorf("%s: reason does not name the fix an operator has to apply: %q", ns.name, got.reason)
		}
	}
}

// The namespace signal itself. Only a POSITIVE reading may unlock the skip.
func TestNetnsIsOnlyEstablishedPositively(t *testing.T) {
	const initial = "net:[4026531840]"
	const pod = "net:[4026532567]"

	rows := []struct {
		name, self, pid1 string
		want             netnsKind
	}{
		{"host: initial namespace, PID 1 agrees", initial, initial, netnsHost},
		{"host: initial namespace, no PID 1 visibility", initial, "", netnsHost},

		// The reported case: no hostPID, so PID 1 is the container's own init
		// and the links AGREE while proving nothing. A match must not be read
		// as evidence of the host namespace.
		{"pod without hostPID: links agree but prove nothing", pod, pod, netnsUnknown},

		// With hostPID, the mismatch is conclusive.
		{"pod with hostPID: differs from init", pod, initial, netnsNotHost},

		// Nothing readable at all fails closed rather than assuming the host.
		{"no readable namespace", "", "", netnsUnknown},
		{"no self link but a pid1 link", "", initial, netnsUnknown},
	}
	for _, r := range rows {
		if got := netnsFromLinks(r.self, r.pid1); got != r.want {
			t.Errorf("%s: netnsFromLinks(%q, %q) = %v, want %v", r.name, r.self, r.pid1, got, r.want)
		}
	}
}

// The namespace check must not weaken the two verdicts that were already right.
func TestTheNamespaceCheckDoesNotChangeTheOtherOutcomes(t *testing.T) {
	for _, ns := range []netnsKind{netnsHost, netnsNotHost, netnsUnknown} {
		// A real fabric path is still allowed: the guard's job is to refuse the
		// MANAGEMENT path, not to refuse pods.
		if got := classifyRoute("enp1s0f0", "eth0", false, ns); got.decision != routeOK {
			t.Errorf("ns=%v: a fabric path was refused (%v): %s", ns, got.decision, got.reason)
		}
		// Naming the management interface stays an Error under every namespace.
		if got := classifyRoute("eth0", "eth0", true, ns); got.decision != routeUnknown {
			t.Errorf("ns=%v: naming the management interface = %v, want routeUnknown", ns, got.decision)
		}
	}
}

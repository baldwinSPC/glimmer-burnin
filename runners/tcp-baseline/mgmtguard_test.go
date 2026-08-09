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
		got := classifyRoute(r.peerIface, r.mgmtIface, false)
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
	got := classifyRoute("eth0", "eth0", false)
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
	got := classifyRoute("eth0", "eth0", true)
	if got.decision != routeUnknown {
		t.Fatalf("decision = %v, want routeUnknown (an Error)", got.decision)
	}
	if !strings.Contains(got.reason, "no override") {
		t.Errorf("the reason should say there is no override: %q", got.reason)
	}
}

func TestAFabricPathIsAllowed(t *testing.T) {
	got := classifyRoute("enp1s0f0", "eth0", false)
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

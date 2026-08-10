// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// RDMA device and GID discovery, from sysfs and the routing table only.
//
// # Why this is auto-detected rather than configured
//
// A fabric runner that names its device in a flag is wrong on the first node
// that cables its ports differently. On the two DGX Sparks this runner was
// developed against, the SAME cable lands on roceP2p1s0f1 at one end and
// roceP2p1s0f0 at the other — the port numbering is not symmetric across a
// link. A hard-coded "-d roceP2p1s0f1" therefore measures the correct link from
// one node and a dead port from the other, and reports the difference as a
// fabric fault.
//
// So the device is derived from the only thing that is true on both ends: the
// route to the peer. The kernel is asked which local address it would use to
// reach the peer, that address is matched to a netdev, and the netdev is matched
// to the RDMA device that fronts it via
// /sys/class/infiniband/<dev>/device/net/<netdev>. The answer is whatever the
// traffic would actually take.
//
// # Why the GID index is derived too
//
// On RoCE, perftest needs -x <gid index> and there is no safe default: index 0
// and 1 are the link-local IPv6 GIDs, and the RoCE v2 IPv4 GID sits at whatever
// index the driver assigned it (index 3 on these nodes, but that is an
// observation, not a contract). Picking the wrong one produces a connection that
// fails or, worse, one that succeeds over an unintended address family. The
// index is therefore matched by content: the RoCE v2 GID whose embedded IPv4
// address is the local address the route chose.

const sysInfiniband = "/sys/class/infiniband"

// linkLayerEthernet is the sysfs spelling for RoCE. InfiniBand ports report
// "InfiniBand" and need no GID index at all, which is why the two are
// distinguished rather than assumed.
const linkLayerEthernet = "Ethernet"

// rdmaPort is one port of one RDMA device, with everything perftest needs to be
// pointed at it.
type rdmaPort struct {
	Device    string // e.g. "roceP2p1s0f1"
	Port      int    // IB port number, 1-based
	NetDev    string // the Ethernet netdev this port fronts, "" on true IB
	LinkLayer string // "Ethernet" (RoCE) or "InfiniBand"
	State     string // e.g. "ACTIVE", "DOWN"
	// GIDIndex is the RoCE v2 GID to pass to perftest as -x, or -1 when the
	// port needs none (InfiniBand) or none could be found.
	GIDIndex int
	// GIDAddr is the IPv4 address embedded in that GID, for the log line that
	// tells a human which address the test actually ran over.
	GIDAddr string
}

func (p rdmaPort) active() bool { return strings.Contains(strings.ToUpper(p.State), "ACTIVE") }

func (p rdmaPort) String() string {
	s := fmt.Sprintf("%s port %d (%s, %s", p.Device, p.Port, p.LinkLayer, p.State)
	if p.NetDev != "" {
		s += ", netdev " + p.NetDev
	}
	if p.GIDIndex >= 0 {
		s += fmt.Sprintf(", gid index %d", p.GIDIndex)
		if p.GIDAddr != "" {
			s += " = " + p.GIDAddr
		}
	}
	return s + ")"
}

// errNoRDMA is the SKIP condition: this node has no RDMA hardware at all, so
// there is nothing here to judge. It is deliberately distinct from "the hardware
// is present and the link is down", which is a verdict.
var errNoRDMA = errors.New("no RDMA device present")

// discoverPorts lists every port of every RDMA device sysfs knows about.
func discoverPorts() ([]rdmaPort, error) {
	devs, err := os.ReadDir(sysInfiniband)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, errNoRDMA
		}
		return nil, fmt.Errorf("reading %s: %w", sysInfiniband, err)
	}

	var out []rdmaPort
	for _, d := range devs {
		dev := d.Name()
		netdev := netdevOf(dev)
		portDirs, err := os.ReadDir(filepath.Join(sysInfiniband, dev, "ports"))
		if err != nil {
			continue
		}
		for _, pd := range portDirs {
			n, err := strconv.Atoi(pd.Name())
			if err != nil {
				continue
			}
			base := filepath.Join(sysInfiniband, dev, "ports", pd.Name())
			p := rdmaPort{
				Device:    dev,
				Port:      n,
				NetDev:    netdev,
				LinkLayer: readTrimmed(filepath.Join(base, "link_layer")),
				State:     readTrimmed(filepath.Join(base, "state")),
				GIDIndex:  -1,
			}
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return nil, errNoRDMA
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Device != out[j].Device {
			return out[i].Device < out[j].Device
		}
		return out[i].Port < out[j].Port
	})
	return out, nil
}

// netdevOf returns the Ethernet interface an RDMA device fronts, or "" for a
// device that fronts none (true InfiniBand).
func netdevOf(dev string) string {
	entries, err := os.ReadDir(filepath.Join(sysInfiniband, dev, "device", "net"))
	if err != nil || len(entries) == 0 {
		return ""
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	return names[0]
}

// selectPort picks the port the test should run over.
//
// The route to the peer decides, because that is the link the operator asked
// about. Everything else is a fallback for a node whose fabric addresses are not
// the ones the peer is named by, and each fallback is reported in the log so a
// surprising choice is visible rather than silent.
func selectPort(ports []rdmaPort, peerIP net.IP) (rdmaPort, string, error) {
	if len(ports) == 0 {
		return rdmaPort{}, "", errNoRDMA
	}

	// Preferred: the device carrying the route to the peer.
	if peerIP != nil {
		if localIP, err := localSourceIP(peerIP); err == nil {
			if nd, err := netdevForIP(localIP); err == nil {
				for _, p := range ports {
					if p.NetDev != nd || !p.active() {
						continue
					}
					resolveGID(&p, localIP)
					return p, fmt.Sprintf("selected by route to peer %s via %s (local address %s)", peerIP, nd, localIP), nil
				}
			}
		}
	}

	// Fallback: the first ACTIVE port that has a usable GID. A node whose peer
	// is named by an address that does not live on the fabric NIC still has
	// exactly one sensible answer here, and refusing to run would report an
	// Error about hardware that is fine.
	for _, p := range ports {
		if !p.active() {
			continue
		}
		resolveGID(&p, nil)
		if p.LinkLayer != linkLayerEthernet || p.GIDIndex >= 0 {
			return p, "selected as the first ACTIVE port; the route to the peer did not land on an RDMA netdev", nil
		}
	}

	// Hardware is present but nothing is up. This is a statement about the
	// fabric, not about the runner, and main.go turns it into a verdict rather
	// than into an Error.
	return rdmaPort{}, "", fmt.Errorf("no ACTIVE RDMA port: %s", summarise(ports))
}

func summarise(ports []rdmaPort) string {
	parts := make([]string, 0, len(ports))
	for _, p := range ports {
		parts = append(parts, fmt.Sprintf("%s/%d %s", p.Device, p.Port, p.State))
	}
	return strings.Join(parts, ", ")
}

// resolveGID fills in the RoCE v2 GID index for an Ethernet port.
//
// want is the local address the route chose; when it is non-nil the GID that
// embeds exactly that address is used, which is the only choice guaranteed to
// match the interface the traffic leaves by. When it is nil, or no GID carries
// that address (a secondary address with no GID of its own is normal), the
// lowest-indexed RoCE v2 IPv4 GID is used — still a v2 IPv4 GID, just not a
// specific one.
func resolveGID(p *rdmaPort, want net.IP) {
	if p.LinkLayer != linkLayerEthernet {
		return // InfiniBand: perftest needs no -x.
	}
	base := filepath.Join(sysInfiniband, p.Device, "ports", strconv.Itoa(p.Port))
	types, err := os.ReadDir(filepath.Join(base, "gid_attrs", "types"))
	if err != nil {
		return
	}
	idxs := make([]int, 0, len(types))
	for _, t := range types {
		if n, err := strconv.Atoi(t.Name()); err == nil {
			idxs = append(idxs, n)
		}
	}
	sort.Ints(idxs)

	firstV4 := -1
	firstV4Addr := ""
	for _, i := range idxs {
		if !strings.Contains(readTrimmed(filepath.Join(base, "gid_attrs", "types", strconv.Itoa(i))), "RoCE v2") {
			continue
		}
		gid := readTrimmed(filepath.Join(base, "gids", strconv.Itoa(i)))
		v4 := ipv4OfGID(gid)
		if v4 == nil {
			continue
		}
		if want != nil && v4.Equal(want) {
			p.GIDIndex, p.GIDAddr = i, v4.String()
			return
		}
		if firstV4 < 0 {
			firstV4, firstV4Addr = i, v4.String()
		}
	}
	p.GIDIndex, p.GIDAddr = firstV4, firstV4Addr
}

// ipv4OfGID extracts the IPv4 address an IPv4-mapped RoCE v2 GID carries, e.g.
// "0000:0000:0000:0000:0000:ffff:0afc:a1d1" -> 10.252.161.209. A link-local
// IPv6 GID returns nil, which is how the v6 GIDs are excluded by content rather
// than by index.
func ipv4OfGID(gid string) net.IP {
	if gid == "" {
		return nil
	}
	ip := net.ParseIP(strings.ReplaceAll(gid, " ", ""))
	if ip == nil {
		return nil
	}
	v4 := ip.To4()
	if v4 == nil {
		return nil
	}
	return v4
}

// localSourceIP asks the kernel which local address it would use to reach the
// peer. The UDP socket is never sent on: connect() on a datagram socket only
// performs the route lookup and binds the source address, which is exactly the
// question, and it needs no privilege and no external command.
func localSourceIP(peer net.IP) (net.IP, error) {
	c, err := net.Dial("udp", net.JoinHostPort(peer.String(), "9"))
	if err != nil {
		return nil, err
	}
	defer c.Close()
	ua, ok := c.LocalAddr().(*net.UDPAddr)
	if !ok {
		return nil, fmt.Errorf("unexpected local address type %T", c.LocalAddr())
	}
	return ua.IP, nil
}

// netdevForIP finds the interface that carries an address.
func netdevForIP(ip net.IP) (string, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return "", err
	}
	for _, ifc := range ifaces {
		addrs, err := ifc.Addrs()
		if err != nil {
			continue
		}
		for _, a := range addrs {
			if n, ok := a.(*net.IPNet); ok && n.IP.Equal(ip) {
				return ifc.Name, nil
			}
		}
	}
	return "", fmt.Errorf("no interface carries %s", ip)
}

func readTrimmed(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	// sysfs "state" reads as "4: ACTIVE"; the number is an enum nobody should
	// depend on, so only the name is kept.
	s := strings.TrimSpace(string(b))
	if i := strings.Index(s, ": "); i >= 0 && i <= 3 {
		s = strings.TrimSpace(s[i+2:])
	}
	return s
}

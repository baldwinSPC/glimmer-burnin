// Package hostinfo describes a machine that is not a cluster member.
//
// In a cluster, NodeFingerprint answers this: it reads labels a device plugin
// wrote and turns them into an inventory. At provisioning time none of that
// exists — no kubelet, no device plugin, no labels — and the machine still has
// to be described. A report needs the GPU model and driver version, test
// selection needs to know whether there is an accelerator at all, and a fabric
// test needs to know which interfaces are candidates.
//
// # Standard library only
//
// The same discipline every runner follows, for the same reason: this code runs
// on machines whose state is unknown, at the moment they are least trustworthy.
// A dependency that needs a package manager is a dependency that fails on the
// box you most need to inspect. TestNoThirdPartyImports keeps it that way.
//
// # PCI IDs are what the hardware says about itself
//
// The cluster path derives a vendor from the DNS domain of a label, which is
// elegant and depends entirely on something else having labelled the node. This
// reads /sys/bus/pci instead. A label can be absent, stale, or wrong; the vendor
// ID in a device's config space is what the card reports about itself.
//
// # Absent, never guessed
//
// A field this cannot establish is left at its zero value. An unrecognised PCI
// vendor is reported as its hex ID rather than as "unknown", because the ID is
// true and the word is not. That is the same rule pkg/report renders under, and
// the same rule the runners emit under.
package hostinfo

import (
	"bufio"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
)

// Host is what could be established about a machine.
type Host struct {
	Hostname string
	// OSImage is the pretty name from /etc/os-release, e.g. "Ubuntu 24.04.1 LTS".
	OSImage string
	Kernel  string
	Arch    string
	CPUs    int
	// MemoryBytes is total physical memory, or zero when it could not be read.
	MemoryBytes int64

	Accelerators []Accelerator
	NICs         []NIC
}

// Accelerator is one PCI device that presents as a GPU or other accelerator.
type Accelerator struct {
	// PCIAddress is the slot, e.g. "0000:01:00.0" — what an engineer walking to
	// the rack needs.
	PCIAddress string
	// Vendor is the resolved name for a known ID: nvidia, amd, intel,
	// tenstorrent. For anything else it is the raw hex ID, because that is true
	// and "unknown" is not.
	Vendor string
	// VendorID and DeviceID are as reported by the device.
	VendorID string
	DeviceID string
	// Class is the PCI class code, kept so a caller can tell a 3D controller
	// from a processing accelerator without re-reading sysfs.
	Class string
	// Driver is the kernel module bound to the device, when one is.
	Driver string
}

// NIC is one network interface.
type NIC struct {
	Name string
	// SpeedMbps is zero when the link is down or the driver does not report it.
	SpeedMbps int32
	MTU       int32
	// LinkLayer is "ethernet" or "infiniband" when it could be determined.
	LinkLayer string
	// RDMADevice is the verbs device bound to this interface, when there is one.
	// Its presence is what makes an interface a fabric candidate.
	RDMADevice string
	// OperState is the kernel's own word: "up", "down", "unknown".
	OperState  string
	MACAddress string
}

// PCI vendor IDs for accelerators this project knows by name.
//
// A short list on purpose. It resolves the four vendors the suite has runners
// or plans for; everything else keeps its hex ID, which is more useful than a
// guess and cannot become wrong.
var pciVendors = map[string]string{
	"0x10de": "nvidia",
	"0x1002": "amd",
	"0x8086": "intel",
	"0x1e52": "tenstorrent",
}

// acceleratorClasses are the PCI class prefixes that identify a device as one.
//
// 0x0302 is a 3D controller, which is what a datacentre GPU presents as.
// 0x1200 is a processing accelerator. 0x0300 (VGA) is deliberately EXCLUDED:
// including it would report the motherboard's display adapter as an
// accelerator, and a report claiming a burn-in target that is really a BMC's
// video output is worse than one that missed a card.
var acceleratorClasses = []string{"0x0302", "0x1200"}

// Options controls where the probe reads from. The zero value reads the real
// host; tests supply a fixture root.
type Options struct {
	// SysfsRoot defaults to /sys.
	SysfsRoot string
	// EtcRoot defaults to /etc.
	EtcRoot string
	// ProcRoot defaults to /proc.
	ProcRoot string
}

func (o Options) sysfs() string {
	if o.SysfsRoot != "" {
		return o.SysfsRoot
	}
	return "/sys"
}

func (o Options) etc() string {
	if o.EtcRoot != "" {
		return o.EtcRoot
	}
	return "/etc"
}

func (o Options) proc() string {
	if o.ProcRoot != "" {
		return o.ProcRoot
	}
	return "/proc"
}

// Probe describes the host.
//
// It never returns an error. Every part is independent and optional: a machine
// with no accelerators, no RDMA, and no /etc/os-release is a legitimate thing to
// describe, and the answer is a Host with fewer fields set rather than a
// failure. A caller that needs a particular field checks for it.
func Probe(opts Options) Host {
	h := Host{
		Arch: runtime.GOARCH,
		CPUs: runtime.NumCPU(),
	}
	if name, err := os.Hostname(); err == nil {
		h.Hostname = name
	}
	h.OSImage = osPrettyName(opts.etc())
	h.Kernel = kernelRelease(opts.proc())
	h.MemoryBytes = totalMemory(opts.proc())
	h.Accelerators = accelerators(opts.sysfs())
	h.NICs = nics(opts.sysfs())
	return h
}

// osPrettyName reads PRETTY_NAME from os-release.
//
// The value is quoted in the file and the quotes are not part of it.
func osPrettyName(etcRoot string) string {
	f, err := os.Open(filepath.Join(etcRoot, "os-release"))
	if err != nil {
		return ""
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		key, value, ok := strings.Cut(s.Text(), "=")
		if ok && key == "PRETTY_NAME" {
			return strings.Trim(strings.TrimSpace(value), `"'`)
		}
	}
	return ""
}

// kernelRelease reads the version from /proc/sys/kernel/osrelease.
//
// Read rather than shelled out to `uname`: the file is the same source uname
// uses, and reading it needs no subprocess on a box that may be short of them.
func kernelRelease(procRoot string) string {
	b, err := os.ReadFile(filepath.Join(procRoot, "sys", "kernel", "osrelease"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// totalMemory reads MemTotal from /proc/meminfo, in bytes.
func totalMemory(procRoot string) int64 {
	f, err := os.Open(filepath.Join(procRoot, "meminfo"))
	if err != nil {
		return 0
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		// "MemTotal:  131923412 kB"
		if len(fields) >= 2 && fields[0] == "MemTotal:" {
			kb, err := strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return 0
			}
			return kb * 1024
		}
	}
	return 0
}

// accelerators walks /sys/bus/pci/devices.
func accelerators(sysfsRoot string) []Accelerator {
	entries, err := os.ReadDir(filepath.Join(sysfsRoot, "bus", "pci", "devices"))
	if err != nil {
		return nil
	}

	var out []Accelerator
	for _, e := range entries {
		dir := filepath.Join(sysfsRoot, "bus", "pci", "devices", e.Name())
		class := readTrimmed(filepath.Join(dir, "class"))
		if !isAccelerator(class) {
			continue
		}

		vendorID := readTrimmed(filepath.Join(dir, "vendor"))
		a := Accelerator{
			PCIAddress: e.Name(),
			VendorID:   vendorID,
			DeviceID:   readTrimmed(filepath.Join(dir, "device")),
			Class:      class,
			Driver:     driverOf(dir),
		}
		if name, ok := pciVendors[strings.ToLower(vendorID)]; ok {
			a.Vendor = name
		} else {
			// The hex ID is true; "unknown" is not.
			a.Vendor = vendorID
		}
		out = append(out, a)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].PCIAddress < out[j].PCIAddress })
	return out
}

// isAccelerator matches a PCI class against the accelerator prefixes.
//
// The class file is "0x030200" — class, subclass, then programming interface —
// so the comparison is on the first six characters' worth of prefix.
func isAccelerator(class string) bool {
	c := strings.ToLower(class)
	for _, want := range acceleratorClasses {
		if strings.HasPrefix(c, want) {
			return true
		}
	}
	return false
}

// driverOf resolves the bound kernel module from the driver symlink.
func driverOf(deviceDir string) string {
	target, err := os.Readlink(filepath.Join(deviceDir, "driver"))
	if err != nil {
		return ""
	}
	return filepath.Base(target)
}

// nics walks /sys/class/net.
//
// Loopback is skipped: it is never a burn-in candidate and listing it invites a
// caller to treat interface count as meaningful.
func nics(sysfsRoot string) []NIC {
	entries, err := os.ReadDir(filepath.Join(sysfsRoot, "class", "net"))
	if err != nil {
		return nil
	}

	rdma := rdmaDevices(sysfsRoot)

	var out []NIC
	for _, e := range entries {
		name := e.Name()
		if name == "lo" {
			continue
		}
		dir := filepath.Join(sysfsRoot, "class", "net", name)

		n := NIC{
			Name:       name,
			OperState:  readTrimmed(filepath.Join(dir, "operstate")),
			MACAddress: readTrimmed(filepath.Join(dir, "address")),
			LinkLayer:  linkLayer(readTrimmed(filepath.Join(dir, "type"))),
			RDMADevice: rdma[name],
		}
		// speed and mtu are absent or unreadable on a down link, which is a
		// fact about the link rather than an error.
		n.SpeedMbps = int32(readInt(filepath.Join(dir, "speed")))
		n.MTU = int32(readInt(filepath.Join(dir, "mtu")))
		out = append(out, n)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// linkLayer maps the ARPHRD type in /sys/class/net/*/type.
//
// 1 is Ethernet and 32 is InfiniBand. Anything else is left empty rather than
// guessed — a wrong link layer would send a fabric test at the wrong interface.
func linkLayer(t string) string {
	switch strings.TrimSpace(t) {
	case "1":
		return "ethernet"
	case "32":
		return "infiniband"
	default:
		return ""
	}
}

// rdmaDevices maps a network interface name to the verbs device bound to it.
//
// The presence of one is what makes an interface a fabric candidate — it is the
// difference between an interface a burn-in can measure with RDMA and one it
// cannot. The mapping lives under each verbs device's own directory, so this
// walks those rather than guessing a name from the interface.
func rdmaDevices(sysfsRoot string) map[string]string {
	out := map[string]string{}
	devices, err := os.ReadDir(filepath.Join(sysfsRoot, "class", "infiniband"))
	if err != nil {
		return out
	}
	for _, d := range devices {
		// .../infiniband/<verbs>/device/net/<ifname>
		netDir := filepath.Join(sysfsRoot, "class", "infiniband", d.Name(), "device", "net")
		ifaces, err := os.ReadDir(netDir)
		if err != nil {
			continue
		}
		for _, iface := range ifaces {
			out[iface.Name()] = d.Name()
		}
	}
	return out
}

func readTrimmed(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

// readInt returns 0 for an unreadable or non-numeric file.
//
// A down link's speed file returns -1 or EINVAL depending on the driver, and
// neither is a speed. Zero means "not established", consistently.
func readInt(path string) int64 {
	v, err := strconv.ParseInt(readTrimmed(path), 10, 64)
	if err != nil || v < 0 {
		return 0
	}
	return v
}

package hostinfo

import (
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// fakeHost builds a sysfs/etc/proc tree resembling a two-NIC GB10 box.
func fakeHost(t *testing.T) Options {
	t.Helper()
	root := t.TempDir()

	write := func(path, content string) {
		full := filepath.Join(root, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	write("etc/os-release", "NAME=\"Ubuntu\"\nPRETTY_NAME=\"Ubuntu 24.04.1 LTS\"\nID=ubuntu\n")
	write("proc/sys/kernel/osrelease", "6.11.0-1010-nvidia\n")
	write("proc/meminfo", "MemTotal:       131923412 kB\nMemFree:         2000000 kB\n")

	// An accelerator (3D controller, NVIDIA).
	write("sys/bus/pci/devices/0000:01:00.0/class", "0x030200\n")
	write("sys/bus/pci/devices/0000:01:00.0/vendor", "0x10de\n")
	write("sys/bus/pci/devices/0000:01:00.0/device", "0x2e12\n")
	if err := os.Symlink("../../../bus/pci/drivers/nvidia", filepath.Join(root, "sys/bus/pci/devices/0000:01:00.0/driver")); err != nil {
		t.Fatal(err)
	}

	// A VGA controller — the BMC's display output. Must NOT be reported.
	write("sys/bus/pci/devices/0000:00:02.0/class", "0x030000\n")
	write("sys/bus/pci/devices/0000:00:02.0/vendor", "0x1a03\n")
	write("sys/bus/pci/devices/0000:00:02.0/device", "0x2000\n")

	// An accelerator from a vendor this package has no name for.
	write("sys/bus/pci/devices/0000:02:00.0/class", "0x120000\n")
	write("sys/bus/pci/devices/0000:02:00.0/vendor", "0xabcd\n")
	write("sys/bus/pci/devices/0000:02:00.0/device", "0x0001\n")

	// A fabric NIC with an RDMA device, and a management NIC without one.
	write("sys/class/net/enp1s0f0/operstate", "up\n")
	write("sys/class/net/enp1s0f0/address", "aa:bb:cc:dd:ee:f0\n")
	write("sys/class/net/enp1s0f0/type", "1\n")
	write("sys/class/net/enp1s0f0/speed", "200000\n")
	write("sys/class/net/enp1s0f0/mtu", "4096\n")

	write("sys/class/net/wlP9s9/operstate", "down\n")
	write("sys/class/net/wlP9s9/address", "aa:bb:cc:dd:ee:f1\n")
	write("sys/class/net/wlP9s9/type", "1\n")
	write("sys/class/net/wlP9s9/speed", "-1\n") // a down link reports -1
	write("sys/class/net/wlP9s9/mtu", "1500\n")

	write("sys/class/net/lo/operstate", "unknown\n")
	write("sys/class/net/lo/type", "772\n")

	// mlx5_0 is bound to the fabric interface.
	write("sys/class/infiniband/mlx5_0/device/net/enp1s0f0/.keep", "")

	return Options{
		SysfsRoot: filepath.Join(root, "sys"),
		EtcRoot:   filepath.Join(root, "etc"),
		ProcRoot:  filepath.Join(root, "proc"),
	}
}

func TestProbeReadsTheHostFacts(t *testing.T) {
	h := Probe(fakeHost(t))

	if h.OSImage != "Ubuntu 24.04.1 LTS" {
		t.Errorf("OSImage = %q, want the unquoted PRETTY_NAME with its spaces intact", h.OSImage)
	}
	if h.Kernel != "6.11.0-1010-nvidia" {
		t.Errorf("Kernel = %q", h.Kernel)
	}
	if want := int64(131923412) * 1024; h.MemoryBytes != want {
		t.Errorf("MemoryBytes = %d, want %d (kB converted)", h.MemoryBytes, want)
	}
	if h.CPUs == 0 || h.Arch == "" {
		t.Errorf("CPUs/Arch not populated: %d %q", h.CPUs, h.Arch)
	}
}

func TestAVGAControllerIsNotAnAccelerator(t *testing.T) {
	// A BMC's display output is not a burn-in target. Reporting it would claim
	// a device the suite cannot test and would inflate the accelerator count a
	// caller uses to decide what to run.
	h := Probe(fakeHost(t))

	for _, a := range h.Accelerators {
		if a.PCIAddress == "0000:00:02.0" {
			t.Error("a VGA controller was reported as an accelerator")
		}
	}
	if len(h.Accelerators) != 2 {
		t.Fatalf("got %d accelerators, want the 3D controller and the processing accelerator", len(h.Accelerators))
	}
}

func TestAKnownVendorIsNamedAndAnUnknownOneKeepsItsID(t *testing.T) {
	// "unknown" would be a claim; the hex ID is a fact, and it is the thing
	// someone would search for.
	h := Probe(fakeHost(t))

	byAddr := map[string]Accelerator{}
	for _, a := range h.Accelerators {
		byAddr[a.PCIAddress] = a
	}

	if got := byAddr["0000:01:00.0"].Vendor; got != "nvidia" {
		t.Errorf("known vendor = %q, want nvidia", got)
	}
	if got := byAddr["0000:01:00.0"].Driver; got != "nvidia" {
		t.Errorf("driver = %q, want the bound module resolved from the symlink", got)
	}
	if got := byAddr["0000:02:00.0"].Vendor; got != "0xabcd" {
		t.Errorf("unknown vendor = %q, want the raw ID rather than a guess", got)
	}
}

func TestAnRDMADeviceMarksAFabricCandidate(t *testing.T) {
	// The presence of a verbs device is what separates an interface a burn-in
	// can measure with RDMA from one it cannot.
	h := Probe(fakeHost(t))

	byName := map[string]NIC{}
	for _, n := range h.NICs {
		byName[n.Name] = n
	}

	if byName["enp1s0f0"].RDMADevice != "mlx5_0" {
		t.Errorf("fabric NIC RDMADevice = %q, want mlx5_0", byName["enp1s0f0"].RDMADevice)
	}
	if byName["wlP9s9"].RDMADevice != "" {
		t.Errorf("a NIC with no verbs device reported %q", byName["wlP9s9"].RDMADevice)
	}
}

func TestADownLinkReportsNoSpeedRatherThanANegativeOne(t *testing.T) {
	// The kernel reports -1 for a down link. That is not a speed, and a caller
	// comparing against a threshold must not see it as one.
	h := Probe(fakeHost(t))

	for _, n := range h.NICs {
		if n.SpeedMbps < 0 {
			t.Errorf("%s reported a negative speed %d", n.Name, n.SpeedMbps)
		}
		if n.Name == "wlP9s9" && n.SpeedMbps != 0 {
			t.Errorf("a down link reported speed %d, want 0 meaning not established", n.SpeedMbps)
		}
		if n.Name == "enp1s0f0" && n.SpeedMbps != 200000 {
			t.Errorf("fabric NIC speed = %d, want 200000", n.SpeedMbps)
		}
	}
}

func TestLoopbackIsNotListed(t *testing.T) {
	// It is never a burn-in candidate, and listing it invites a caller to treat
	// interface count as meaningful.
	for _, n := range Probe(fakeHost(t)).NICs {
		if n.Name == "lo" {
			t.Error("loopback was listed as an interface")
		}
	}
}

func TestLinkLayerIsEmptyRatherThanGuessed(t *testing.T) {
	h := Probe(fakeHost(t))
	for _, n := range h.NICs {
		if n.Name == "enp1s0f0" && n.LinkLayer != "ethernet" {
			t.Errorf("ARPHRD 1 = %q, want ethernet", n.LinkLayer)
		}
	}

	// An unrecognised ARPHRD stays empty: a wrong link layer would send a
	// fabric test at the wrong interface.
	if got := linkLayer("999"); got != "" {
		t.Errorf("an unrecognised link type resolved to %q", got)
	}
	if got := linkLayer("32"); got != "infiniband" {
		t.Errorf("ARPHRD 32 = %q, want infiniband", got)
	}
}

func TestAnEmptyMachineIsDescribedRatherThanFailing(t *testing.T) {
	// No accelerators, no RDMA, no os-release. A laptop, a CI runner, a
	// half-provisioned box — all legitimate things to describe, and the answer
	// is fewer fields rather than an error.
	empty := t.TempDir()
	h := Probe(Options{
		SysfsRoot: filepath.Join(empty, "sys"),
		EtcRoot:   filepath.Join(empty, "etc"),
		ProcRoot:  filepath.Join(empty, "proc"),
	})

	if len(h.Accelerators) != 0 || len(h.NICs) != 0 {
		t.Errorf("invented hardware on an empty tree: %+v", h)
	}
	if h.OSImage != "" || h.Kernel != "" || h.MemoryBytes != 0 {
		t.Errorf("invented host facts on an empty tree: %+v", h)
	}
	// These come from the Go runtime and are always knowable.
	if h.CPUs == 0 || h.Arch == "" {
		t.Error("runtime-derived fields should still be populated")
	}
}

func TestProbeIsDeterministic(t *testing.T) {
	// Directory iteration order is not guaranteed, and a fingerprint that
	// reorders between reads looks like hardware that changed.
	opts := fakeHost(t)
	first := Probe(opts)
	for i := 0; i < 5; i++ {
		again := Probe(opts)
		if len(again.Accelerators) != len(first.Accelerators) {
			t.Fatal("accelerator count varies between probes")
		}
		for j := range first.Accelerators {
			if again.Accelerators[j].PCIAddress != first.Accelerators[j].PCIAddress {
				t.Fatal("accelerator order varies between probes")
			}
		}
		for j := range first.NICs {
			if again.NICs[j].Name != first.NICs[j].Name {
				t.Fatal("NIC order varies between probes")
			}
		}
	}
}

// TestNoThirdPartyImports holds the standard-library-only rule.
//
// This runs on machines at their least trustworthy moment. A dependency that
// needs a package manager is one that fails on the box you most need to inspect.
func TestNoThirdPartyImports(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/baldwinSPC/glimmer-burnin/pkg/hostinfo").Output()
	if err != nil {
		t.Skipf("go list unavailable: %v", err)
	}
	const self = "github.com/baldwinSPC/glimmer-burnin/pkg/hostinfo"
	for _, dep := range strings.Fields(string(out)) {
		if dep == self {
			continue
		}
		// Standard library packages have no dot in their first path element.
		first, _, _ := strings.Cut(dep, "/")
		if strings.Contains(first, ".") {
			t.Errorf("pkg/hostinfo depends on %s — it must stay standard-library only", dep)
		}
	}
}

// PrimaryIP answers a routing question, so a unit test cannot assert WHICH
// address it returns without inheriting the CI machine's network. What it can
// assert — and what actually matters — is that whatever comes back is a real,
// usable address rather than a plausible-looking placeholder, because the
// caller injects it into a runner's environment as status.hostIP.
func TestPrimaryIPIsEitherAUsableAddressOrEmpty(t *testing.T) {
	got := PrimaryIP()
	if got == "" {
		// Legitimate: a host with no default route and no global unicast
		// address. Absence is the honest answer and the caller refuses rather
		// than substituting one.
		return
	}
	ip := net.ParseIP(got)
	if ip == nil {
		t.Fatalf("PrimaryIP() = %q, which does not parse as an IP", got)
	}
	if ip.IsLoopback() {
		t.Errorf("PrimaryIP() = %q — a loopback address is never reachable from another machine, "+
			"and handing it to a runner as status.hostIP would produce a peer address nothing can dial", got)
	}
	if ip.IsUnspecified() {
		t.Errorf("PrimaryIP() = %q — the unspecified address is a bind wildcard, not a host address", got)
	}
	if ip.IsMulticast() {
		t.Errorf("PrimaryIP() = %q is multicast", got)
	}
}

// Probe carries it, so a caller needs one probe rather than two.
func TestProbeCarriesThePrimaryIP(t *testing.T) {
	if Probe(Options{}).PrimaryIP != PrimaryIP() {
		t.Error("Probe().PrimaryIP disagrees with PrimaryIP()")
	}
}

// vgaFixture builds a single VGA-class (0x0300) PCI device, with a drm/
// render node only when renderNode is true.
func vgaFixture(t *testing.T, vendor string, renderNode bool) Options {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "sys/bus/pci/devices/000f:01:00.0")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("class", "0x030000\n")
	write("vendor", vendor+"\n")
	write("device", "0x2e12\n")
	if renderNode {
		if err := os.MkdirAll(filepath.Join(dir, "drm", "renderD128"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(filepath.Join(dir, "drm", "card0"), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return Options{SysfsRoot: filepath.Join(root, "sys")}
}

// TestAGB10CountsDespiteBeingVGAClass is the defect from #380, ported here
// after runners/fingerprint-probe/rendernode_test.go — this package carried
// the fix later than its sibling, and this is the fixture that would have
// caught the gap: a GB10-shaped device (class 0x030000, vendor 0x10de, a
// drm/renderD128 node) reporting acceleratorCount effectively 0 before this
// change, on a node compute-smoke measured at 104 TFLOPS in the same run.
func TestAGB10CountsDespiteBeingVGAClass(t *testing.T) {
	h := Probe(vgaFixture(t, "0x10de", true))
	if len(h.Accelerators) != 1 {
		t.Fatalf("got %d accelerators, want 1 — a GB10 is class 0x030000 and is still an accelerator "+
			"when it exposes a DRM render node", len(h.Accelerators))
	}
	if h.Accelerators[0].Vendor != "nvidia" {
		t.Errorf("vendor = %q, want nvidia", h.Accelerators[0].Vendor)
	}
}

// TestAVGADeviceWithNoRenderNodeIsNotCountedEvenFromAKnownVendor is the other
// half: a display-class device from a vendor this package names, but with NO
// render node, is unclassifiable from sysfs alone and must not be guessed
// into either state.
func TestAVGADeviceWithNoRenderNodeIsNotCountedEvenFromAKnownVendor(t *testing.T) {
	h := Probe(vgaFixture(t, "0x10de", false))
	if len(h.Accelerators) != 0 {
		t.Errorf("got %d accelerators, want 0 — no render node means this could be a real display "+
			"adapter, and guessing it in is exactly the false positive the render-node check exists "+
			"to prevent", len(h.Accelerators))
	}
}

// TestPCIVendorsAgreeWithTheContractList holds this package's PCI-vendor-ID
// table to pkg/contract.AcceleratorVendors — see that list's own doc comment
// for why four sources are required to agree and how each is checked.
func TestPCIVendorsAgreeWithTheContractList(t *testing.T) {
	got := map[string]bool{}
	for _, name := range pciVendors {
		got[name] = true
	}
	want := map[string]bool{}
	for _, name := range contract.AcceleratorVendors {
		want[name] = true
	}
	for name := range got {
		if !want[name] {
			t.Errorf("pciVendors names %q, which is not in contract.AcceleratorVendors", name)
		}
	}
	for name := range want {
		if !got[name] {
			t.Errorf("contract.AcceleratorVendors names %q, which pciVendors has no PCI ID for — "+
				"add one, or the vendor can never actually be detected from hardware", name)
		}
	}
}

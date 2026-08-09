// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/hostinfo"
)

// This runner's PCI reading is a SECOND implementation of
// pkg/hostinfo.accelerators, forced by the image build: a runner's non-test
// sources cannot import a repo package, because the image builds a throwaway
// module from `COPY *.go` with GOPROXY=off (issue #250).
//
// This test is the price of that, and it is the whole reason the duplication is
// acceptable rather than merely unavoidable: both implementations run over the
// same fixture tree and must agree on every field. A test file MAY import the
// repo — the image build deletes *_test.go before compiling — so the guard
// costs the shipped image nothing.
//
// If this ever fails, the two have drifted, and the fix is to bring this
// runner's pci.go back in line rather than to relax the comparison.

// fixture builds a sysfs tree with the devices described.
func fixture(t *testing.T, devices []fixtureDevice) string {
	t.Helper()
	root := t.TempDir()
	for _, d := range devices {
		dir := filepath.Join(root, "bus", "pci", "devices", d.addr)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		write := func(name, body string) {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(body+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		write("class", d.class)
		write("vendor", d.vendor)
		write("device", d.device)
		if d.driver != "" {
			target := filepath.Join(root, "bus", "pci", "drivers", d.driver)
			if err := os.MkdirAll(target, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, filepath.Join(dir, "driver")); err != nil {
				t.Fatal(err)
			}
		}
	}
	return root
}

type fixtureDevice struct{ addr, class, vendor, device, driver string }

// A node worth testing against: two NVIDIA accelerators, an AMD one, a
// processing accelerator from a vendor this project has no name for, a VGA
// adapter that must NOT count, and a card with no driver bound.
var realisticNode = []fixtureDevice{
	{"0000:01:00.0", "0x030200", "0x10de", "0x2901", "nvidia"},
	{"0000:02:00.0", "0x030200", "0x10de", "0x2901", "nvidia"},
	{"0000:03:00.0", "0x030200", "0x1002", "0x74a1", "amdgpu"},
	{"0000:04:00.0", "0x120000", "0xbeef", "0x0001", ""},
	{"0000:00:02.0", "0x030000", "0x8086", "0x9a49", "i915"}, // VGA — not an accelerator
}

func TestTheProbeAgreesWithHostinfoDeviceForDevice(t *testing.T) {
	root := fixture(t, realisticNode)

	mine := scanPCI(root)
	theirs := hostinfo.Probe(hostinfo.Options{SysfsRoot: root}).Accelerators

	if len(mine) != len(theirs) {
		t.Fatalf("found %d accelerators, hostinfo found %d — the two implementations disagree about "+
			"what counts as one", len(mine), len(theirs))
	}
	for i := range mine {
		m, h := mine[i], theirs[i]
		if m.PCIAddress != h.PCIAddress || m.Vendor != h.Vendor || m.VendorID != h.VendorID ||
			m.DeviceID != h.DeviceID || m.Class != h.Class || m.Driver != h.Driver {
			t.Errorf("device %d differs:\n  probe:    %+v\n  hostinfo: %+v", i, m, h)
		}
	}
}

func TestAVGAAdapterIsNotAnAccelerator(t *testing.T) {
	// The trap this fixture exists for: on a node with onboard graphics, a
	// display adapter counted as an accelerator would report hardware the fleet
	// does not have — and `acceleratorCount` is a gateable metric.
	for _, d := range scanPCI(fixture(t, realisticNode)) {
		if d.Class == "0x030000" {
			t.Errorf("a VGA adapter was counted as an accelerator: %+v", d)
		}
	}
}

func TestAnUnknownVendorKeepsItsHexIDRatherThanBecomingUnknown(t *testing.T) {
	// The hex ID is true and "unknown" is not. It is also what lets someone
	// look the part up.
	var found bool
	for _, d := range scanPCI(fixture(t, realisticNode)) {
		if d.VendorID == "0xbeef" {
			found = true
			if d.Vendor != "0xbeef" {
				t.Errorf("an unnamed vendor became %q; the raw ID is the honest answer", d.Vendor)
			}
		}
	}
	if !found {
		t.Error("the processing accelerator (class 0x1200) was not detected at all")
	}
}

func TestADeviceWithNoDriverIsReportedWithoutOne(t *testing.T) {
	// A card the kernel sees and no driver claims is the shape of a node whose
	// device plugin never came up — one of the cases this runner exists to make
	// visible, so it must survive the scan rather than be filtered out.
	var seen bool
	for _, d := range scanPCI(fixture(t, realisticNode)) {
		if d.PCIAddress == "0000:04:00.0" {
			seen = true
			if d.Driver != "" {
				t.Errorf("expected no bound driver, got %q", d.Driver)
			}
		}
	}
	if !seen {
		t.Error("the driverless device was dropped")
	}
}

func TestAnAbsentSysfsIsAnOmissionNotZeroDevices(t *testing.T) {
	// "This node has no accelerators" and "I could not look" are different
	// claims, and only one is safe to certify against. scanPCI returns nothing;
	// run() turns that into a SKIP rather than acceleratorCount=0.
	if got := scanPCI(filepath.Join(t.TempDir(), "nope")); got != nil {
		t.Errorf("scanning an absent sysfs returned %v", got)
	}
}

func TestTheScanIsOrderedBySlotSoTwoReadsMatch(t *testing.T) {
	devices := scanPCI(fixture(t, realisticNode))
	for i := 1; i < len(devices); i++ {
		if devices[i-1].PCIAddress > devices[i].PCIAddress {
			t.Errorf("not sorted by slot: %q before %q", devices[i-1].PCIAddress, devices[i].PCIAddress)
		}
	}
}

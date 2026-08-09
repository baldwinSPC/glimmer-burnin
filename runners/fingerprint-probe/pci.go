// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The PCI reading, deliberately a SECOND implementation of what
// pkg/hostinfo.accelerators does — and the reason is the image build, not a
// preference.
//
// Every runner image builds a throwaway module: `COPY *.go` from a context of
// this directory, `go mod init`, `GOPROXY=off`. A runner's non-test sources are
// therefore standard-library-only by construction and cannot import a repo
// package. See issue #250 for the three ways out and why this one was taken:
// the alternative was making one runner build from the repository root, which
// would make it different from the other thirteen in how it is built.
//
// What keeps the two honest is probe_conformance_test.go, which runs BOTH over
// the same fixture tree and fails if they disagree about a single field. A test
// file may import the repo — the image build deletes *_test.go before compiling
// — so the guard costs the shipped image nothing.
//
// KEEP THIS SUBSET SMALL. It mirrors accelerators() and nothing else: no OS, no
// kernel, no memory, no NICs. hostinfo is 378 lines and most of it is not what
// a runner needs; copying all of it would be a fork rather than a subset.

// acceleratorClasses are the PCI class prefixes that mean "accelerator".
//
// 0x0302 is a 3D controller and 0x1200 is a processing accelerator. VGA
// (0x0300) is deliberately absent: a display adapter is not what this operator
// measures, and on a node with an onboard VGA it would otherwise be reported as
// an accelerator the fleet does not have.
var acceleratorClasses = []string{"0x0302", "0x1200"}

// pciVendors resolves the IDs this project has names for.
var pciVendors = map[string]string{
	"0x10de": "nvidia",
	"0x1002": "amd",
	"0x8086": "intel",
	"0x1e52": "tenstorrent",
}

// device is one accelerator as the hardware describes itself.
type device struct {
	PCIAddress string
	Vendor     string
	VendorID   string
	DeviceID   string
	Class      string
	Driver     string
}

// scanPCI reads the accelerators under a sysfs root.
//
// This is the whole point of the runner: PCI IDs are what the hardware says
// about itself, readable on a node with no device plugin, no NFD and no labels
// at all — the case where both the label derivation and the allocatable
// fallback have nothing to work from.
func scanPCI(sysfsRoot string) []device {
	base := filepath.Join(sysfsRoot, "bus", "pci", "devices")
	entries, err := os.ReadDir(base)
	if err != nil {
		// Absent sysfs is an OMISSION, not zero devices. The caller reports it
		// as such: "this node has no accelerators" and "I could not look" are
		// different claims, and only one of them is safe to certify against.
		return nil
	}

	var out []device
	for _, e := range entries {
		dir := filepath.Join(base, e.Name())
		class := readTrimmed(filepath.Join(dir, "class"))
		if !isAccelerator(class) {
			continue
		}

		vendorID := readTrimmed(filepath.Join(dir, "vendor"))
		d := device{
			PCIAddress: e.Name(),
			VendorID:   vendorID,
			DeviceID:   readTrimmed(filepath.Join(dir, "device")),
			Class:      class,
			Driver:     driverOf(dir),
		}
		if name, ok := pciVendors[strings.ToLower(vendorID)]; ok {
			d.Vendor = name
		} else {
			// The hex ID is true; "unknown" is not. A vendor this project has
			// never heard of is still a fact about the node, and recording the
			// raw ID is what lets someone look it up.
			d.Vendor = vendorID
		}
		out = append(out, d)
	}

	// Sorted by slot, so two reads of an unchanged node produce an identical
	// result and a diff between runs means the hardware moved.
	sort.Slice(out, func(i, j int) bool { return out[i].PCIAddress < out[j].PCIAddress })
	return out
}

// isAccelerator matches a PCI class against the accelerator prefixes.
//
// The class file is "0x030200" — class, subclass, then programming interface —
// so the comparison is on the prefix.
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
//
// Empty when nothing is bound, which is a real and interesting state: a device
// the kernel can see but no driver claims is exactly the shape of a node whose
// device plugin never came up, and it is the reason this runner exists.
func driverOf(deviceDir string) string {
	target, err := os.Readlink(filepath.Join(deviceDir, "driver"))
	if err != nil {
		return ""
	}
	return filepath.Base(target)
}

func readTrimmed(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

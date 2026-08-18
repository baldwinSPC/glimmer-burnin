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

// acceleratorClasses are the PCI class prefixes that mean "accelerator"
// WITHOUT further questions being asked.
//
// 0x0302 is a 3D controller and 0x1200 is a processing accelerator. VGA
// (0x0300) is deliberately absent HERE, and is handled separately below: a
// display adapter is not what this operator measures, and on a node with an
// onboard VGA it would otherwise be reported as an accelerator the fleet does
// not have.
var acceleratorClasses = []string{"0x0302", "0x1200"}

// displayClass is the VGA-compatible-controller prefix.
//
// THIS IS NOT A SYNONYM FOR "NOT AN ACCELERATOR", which is what this runner
// used to assume, and it cost a fleet a false verdict. A GB10 presents as
// 0x030000 — measured, device 0x2e12, driver nvidia — so excluding VGA
// outright reported acceleratorCount=0 on a node compute-smoke measured at
// 104 TFLOPS in the same run, and the sample's own threshold turned that into
// a FAIL, which is never retried (#380).
//
// The discriminator is not the class and not the vendor: it is whether the
// device exposes a DRM RENDER NODE. See hasRenderNode.
const displayClass = "0x0300"

// pciVendors resolves the IDs this project has names for.
var pciVendors = map[string]string{
	"0x10de": "nvidia",
	"0x1002": "amd",
	"0x8086": "intel",
	"0x1e52": "tenstorrent",
}

// hasRenderNode reports whether a PCI device exposes a DRM render node.
//
// This is the question "can this thing COMPUTE?", asked of the kernel rather
// than inferred from a class code or a vendor table.
//
// A DRM driver only gets a renderD* node when it declares DRIVER_RENDER. A
// display-only controller — the ASPEED or Matrox VGA on a server baseboard —
// exposes drm/card* and no render node, which is exactly the false positive
// the old VGA exclusion was protecting against. An accelerator exposes both.
//
// Measured on a GB10 (2026-08-17): 000f:01:00.0, class 0x030000, driver nvidia,
// drm/ containing card0 AND renderD128 — and it was the only PCI device on the
// node with a drm/ directory at all.
//
// Deliberately NOT a vendor allowlist and NOT a driver allowlist. Both work on
// today's evidence and both are lists somebody has to maintain, which is the
// shape this project keeps out of the reconciler: adding support for a vendor
// should mean adding a runner, not adding a name to a table here.
func hasRenderNode(deviceDir string) bool {
	entries, err := os.ReadDir(filepath.Join(deviceDir, "drm"))
	if err != nil {
		return false
	}
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), "renderD") {
			return true
		}
	}
	return false
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
// scanResult is what one pass over sysfs established.
//
// Ambiguous is the honest half. A device that is DISPLAY class, from a vendor
// this project knows makes accelerators, with NO render node, cannot be
// classified from sysfs alone: it is either a display adapter (not an
// accelerator, count it out) or an accelerator whose driver never bound (an
// accelerator, count it in). Reporting either answer would be a guess, so the
// caller reports the COUNT as unmeasurable instead. Absence is not a
// declaration, and neither is presence.
type scanResult struct {
	Devices   []device
	Ambiguous []device
}

func scanPCI(sysfsRoot string) []device { return scan(sysfsRoot).Devices }

func scan(sysfsRoot string) scanResult {
	base := filepath.Join(sysfsRoot, "bus", "pci", "devices")
	entries, err := os.ReadDir(base)
	if err != nil {
		// Absent sysfs is an OMISSION, not zero devices. The caller reports it
		// as such: "this node has no accelerators" and "I could not look" are
		// different claims, and only one of them is safe to certify against.
		return scanResult{}
	}

	var out, ambiguous []device
	for _, e := range entries {
		dir := filepath.Join(base, e.Name())
		class := readTrimmed(filepath.Join(dir, "class"))
		vendorID := readTrimmed(filepath.Join(dir, "vendor"))

		accel := isAccelerator(class)
		display := strings.HasPrefix(strings.ToLower(class), displayClass)
		render := display && hasRenderNode(dir)

		switch {
		case accel:
			// An unambiguous accelerator class. No further question.
		case render:
			// Display class WITH a render node: it computes, so it counts.
		case display && knownAcceleratorVendor(vendorID):
			// Display class, no render node, from a vendor that makes
			// accelerators. Unclassifiable from sysfs — recorded, not guessed.
			ambiguous = append(ambiguous, device{
				PCIAddress: e.Name(), VendorID: vendorID, Class: class,
				DeviceID: readTrimmed(filepath.Join(dir, "device")), Driver: driverOf(dir),
			})
			continue
		default:
			continue
		}
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
	sort.Slice(ambiguous, func(i, j int) bool { return ambiguous[i].PCIAddress < ambiguous[j].PCIAddress })
	return scanResult{Devices: out, Ambiguous: ambiguous}
}

// knownAcceleratorVendor reports whether this project has a NAME for the vendor.
//
// Used ONLY to decide whether an unclassifiable display device is worth
// reporting as ambiguous — never to decide that something IS an accelerator.
// That distinction is the whole reason a vendor table is tolerable here: a
// vendor nobody has heard of making a display adapter is not a fleet's problem,
// while an NVIDIA display-class device with no render node is exactly the case
// that should stop the runner asserting a number.
func knownAcceleratorVendor(vendorID string) bool {
	_, ok := pciVendors[strings.ToLower(vendorID)]
	return ok
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

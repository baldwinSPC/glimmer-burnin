// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Command fingerprint-probe reports what the hardware says about itself.
//
// Every other way this operator learns a node's identity depends on something
// else having described it: the NodeFingerprint derives a vendor from the DNS
// domain of labels a device plugin wrote, and falls back to the domain of the
// node's extended resources. Both are excellent, and both are silent on a node
// with no plugin, no NFD and no labels — where the node simply looks like it
// has no accelerator at all.
//
// PCI IDs are ground truth in a way labels are not: they are what the device
// reports, over a bus, with no software having chosen to describe it. This
// runner reads them and puts them in the result, so the evidence travels the
// ordinary channel into envelopes and reports.
//
// THE RUNNER ITSELF NEVER FAILS A NODE: it passes whenever it could look and
// skips when it could not. A verdict, if a profile wants one, comes from a
// threshold — see the note at the bottom of this file for why a count may be
// gated and an identity string may not.
//
// Contract:
//
//	exit 0                                     the hardware was read
//	exit 2  FINGERPRINT_PROBE_SKIP: <reason>   nothing to read from
//	exit 3  FINGERPRINT_PROBE_ERROR: <reason>  machinery
//
// There is deliberately no exit 1: this runner reads, it does not judge. See the
// note at the bottom of this file for what may and may not be gated on.
package main

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	exitPass  = 0
	exitSkip  = 2
	exitError = 3
)

// defaultSysfs is where the host's sysfs is expected to be mounted.
//
// The BurnInTest declares the mount through spec.runner.hostPaths and nothing
// else — a test that declares nothing gets a pod with no volumes at all — so
// this is the path the sample mounts it at, not a path this runner can conjure.
const defaultSysfs = "/host/sys"

func main() { os.Exit(run()) }

func logf(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }

func metric(key, value string) { fmt.Printf("%s=%s\n", key, value) }

func fin(code int, format string, args ...any) int {
	msg := fmt.Sprintf(format, args...)
	switch code {
	case exitSkip:
		fmt.Printf("FINGERPRINT_PROBE_SKIP: %s\n", msg)
	case exitError:
		fmt.Printf("FINGERPRINT_PROBE_ERROR: %s\n", msg)
	}
	logf("%s", msg)
	return code
}

func run() int {
	sysfs := strings.TrimSpace(os.Getenv("FINGERPRINT_PROBE_SYSFS"))
	if sysfs == "" {
		sysfs = defaultSysfs
	}

	if _, err := os.Stat(sysfs); err != nil {
		// A SKIP, and this is the honest phase: the runner was not given the
		// mount it needs, so it did not look. Reporting zero accelerators here
		// would be a fabricated measurement of exactly the kind that certifies
		// a fleet nobody examined.
		return fin(exitSkip, "%s is not readable (%v) — this runner needs the host's sysfs mounted "+
			"read-only through spec.runner.hostPaths, and reports nothing rather than guessing without it",
			sysfs, err)
	}

	devices := scanPCI(sysfs)
	report(devices)

	logf("fingerprint-probe: %d accelerator(s) under %s", len(devices), sysfs)
	return exitPass
}

// report turns the devices into metrics.
//
// Aggregated into a handful of label-valued metrics rather than one per device,
// because the metric registry is a flat namespace of CANONICAL names: a
// per-device key would have to be synthesised (`pciVendorId0`, `pciVendorId1`)
// and no threshold could ever be written against a name that depends on how
// many cards a node happens to hold.
//
// The joined forms are stable — scanPCI sorts by slot — so two reads of an
// unchanged node produce identical strings and a diff means the hardware moved.
func report(devices []device) {
	// Always emitted, including zero. Unlike the identity strings, a count of
	// zero IS a measurement here: the runner reached sysfs and found no
	// accelerator, which is a true and useful thing to say about a CPU-only
	// node. The skip path above is what covers "could not look".
	metric("acceleratorCount", strconv.Itoa(len(devices)))
	if len(devices) == 0 {
		return
	}

	var addresses, vendorIDs, deviceIDs, drivers []string
	vendorSet := map[string]bool{}
	for _, d := range devices {
		addresses = append(addresses, d.PCIAddress)
		vendorIDs = append(vendorIDs, d.VendorID)
		deviceIDs = append(deviceIDs, d.DeviceID)
		vendorSet[d.Vendor] = true
		if d.Driver != "" {
			drivers = append(drivers, d.Driver)
		}
	}

	metric("pciAddresses", strings.Join(addresses, ","))
	metric("pciVendorIds", strings.Join(vendorIDs, ","))
	metric("pciDeviceIds", strings.Join(deviceIDs, ","))
	metric("acceleratorVendors", strings.Join(sortedKeys(vendorSet), ","))

	// OMITTED when nothing is bound, rather than reported as an empty string.
	// A device with no driver is the shape of a node whose plugin never came up
	// — one of the cases this runner exists to make visible — and an empty
	// value would read as "there is no driver information" rather than "the
	// kernel has bound nothing to this card".
	if len(drivers) > 0 {
		metric("acceleratorDrivers", strings.Join(drivers, ","))
	} else {
		logf("fingerprint-probe: no kernel driver is bound to any accelerator on this node")
	}
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Why there is no exit 1.
//
// This runner does not judge. It reads an identity and reports it, and the
// verdict — if a profile wants one — comes from a THRESHOLD, which is the only
// place in this system that turns a number into an acceptance.
//
// That distinction matters for what may be gated. `acceleratorCount` is a
// legitimate acceptance gate: a node that should hold eight cards and reports
// seven has lost one, and that is a hardware fault worth failing. The identity
// STRINGS are not: `acceleratorVendors Equal nvidia` would fail an AMD node for
// being AMD, and it would read as a hardware verdict. pkg/contract marks them
// Evidence so the threshold linter says so at authoring time, and
// fleet-composition rules stay in a node selector — where they are a targeting
// decision rather than an accusation.

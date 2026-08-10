// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// AMD RAS counters, read from sysfs.
//
// host-health is the most portable runner in the suite — standard-library Go,
// no vendor SDK — and on an AMD node it was also the emptiest. The kernel-log
// scanner already carried an `amdgpu.*(gpu reset|ras)` pattern, but that feeds
// kernel_hw_errors, which is EVIDENCE and never a gate. The runner ran, and it
// barely measured: no ECC counters, no RAS counters, nothing gated.
//
// amdgpu exposes RAS through sysfs at
// /sys/class/drm/card*/device/ras/<block>_err_count, each file holding
//
//	ue: 0
//	ce: 0
//
// which needs no privilege, no ioctl and no amd-smi. That matters: the whole
// reason this runner works everywhere is that it links nothing, and reaching
// for a vendor tool to get these numbers would trade that away for counters
// already sitting in a file.
//
// THE THREE STATES ARE THE SAME THREE, and they are why this is trustworthy:
//
//   - A counter that could not be READ is omitted, so a gate on it fails the
//     node. Absence is not a declaration.
//   - `n/a` is a claim about the HARDWARE, emitted only where the runner
//     positively established the part has nothing to report — here, an amdgpu
//     device with no ras/ directory at all, which is the driver saying this
//     ASIC has no RAS block rather than us failing to look.
//   - A partial answer is not a smaller answer. If one card of eight cannot be
//     read, the node total is omitted: the card that did not answer is exactly
//     the one most likely to be broken.

const (
	// amdDriver is the kernel module name that marks a card as AMD's.
	//
	// Matched on the DRIVER rather than the PCI vendor ID because it is the
	// driver that owns the ras/ directory: a card bound to nothing exposes no
	// RAS regardless of who made it, and calling that "unmeasurable hardware"
	// would be a claim about silicon when the truth is a missing module.
	amdDriver = "amdgpu"

	// rasSuffix marks the per-block counter files.
	rasSuffix = "_err_count"
)

// amdSample is one reading of a node's AMD RAS counters.
type amdSample struct {
	// cards is how many amdgpu devices were found at all.
	cards int
	// withRAS is how many of them exposed a ras/ directory.
	withRAS int
	// uncorrected and corrected are node totals across every block of every
	// card, valid only when read is true.
	uncorrected int64
	corrected   int64
	// blocks is how many per-block files contributed, for evidence.
	blocks int
	// read is true only when EVERY card with a ras/ directory answered
	// completely. A subset is a wrong total, not a partial one.
	read bool
	// failures records what could not be read, so an omission is explainable.
	failures []string
}

// isAMDNode reports whether this node has any amdgpu device at all.
//
// The gate on emitting anything: on an NVIDIA or CPU-only node these metrics
// must not appear, because "no AMD RAS counters here" is not a measurement and
// an `n/a` would be a claim about hardware that is not present.
func (s amdSample) isAMDNode() bool { return s.cards > 0 }

// probeAMD reads the RAS counters under a sysfs root.
func probeAMD(sysfsRoot string) amdSample {
	var s amdSample

	cards, err := filepath.Glob(filepath.Join(sysfsRoot, "class", "drm", "card*"))
	if err != nil {
		return s
	}
	sort.Strings(cards)

	complete := true
	for _, card := range cards {
		// card1-DP-1 and friends are connectors, not devices. Filtered by
		// requiring the device/ directory rather than by parsing the name,
		// which would be a second place to get the naming convention wrong.
		device := filepath.Join(card, "device")
		if driverOfDevice(device) != amdDriver {
			continue
		}
		s.cards++

		rasDir := filepath.Join(device, "ras")
		entries, err := os.ReadDir(rasDir)
		if err != nil {
			// No ras/ on an amdgpu device. The driver is loaded and this ASIC
			// has no RAS block — a positive fact about the part, and the one
			// case that earns the `n/a` sentinel. Not counted as a failure.
			continue
		}
		s.withRAS++

		for _, e := range entries {
			name := e.Name()
			if !strings.HasSuffix(name, rasSuffix) {
				continue
			}
			ue, ce, ok := readRASCounts(filepath.Join(rasDir, name))
			if !ok {
				complete = false
				s.failures = append(s.failures, filepath.Base(card)+"/"+name)
				continue
			}
			s.uncorrected += ue
			s.corrected += ce
			s.blocks++
		}
	}

	// read is true only when at least one block was counted AND nothing failed.
	// Both halves matter: a ras/ directory holding no *_err_count files at all
	// is not a zero, it is a layout this runner does not understand.
	s.read = complete && s.blocks > 0
	sort.Strings(s.failures)
	return s
}

// readRASCounts parses one block's counter file.
//
// The format is two labelled lines:
//
//	ue: 0
//	ce: 0
//
// BOTH are required. A file with only one is not half a reading — it is a
// format this runner does not recognise, and guessing the other half as zero
// would understate the damage while looking like a complete answer.
func readRASCounts(path string) (ue, ce int64, ok bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, false
	}
	var haveUE, haveCE bool
	for _, line := range strings.Split(string(b), "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		n, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
		if err != nil {
			continue
		}
		switch strings.TrimSpace(key) {
		case "ue":
			ue, haveUE = n, true
		case "ce":
			ce, haveCE = n, true
		}
	}
	return ue, ce, haveUE && haveCE
}

// driverOfDevice resolves the kernel module bound to a device directory.
func driverOfDevice(deviceDir string) string {
	target, err := os.Readlink(filepath.Join(deviceDir, "driver"))
	if err != nil {
		return ""
	}
	return filepath.Base(target)
}

// amdState maps a sample onto the same counterState vocabulary the NVIDIA path
// uses, so both feed setCounter and neither invents a fourth answer.
//
// The `n/a` case is narrow ON PURPOSE: every amdgpu device present, and not one
// of them exposing a ras/ directory. That is the driver saying these ASICs have
// no RAS block — the AMD analogue of GB10 answering [N/A] for ecc.errors, and
// the same shape of positive determination. A node where SOME cards have RAS
// and others do not is not unmeasurable; it is a node this runner could not
// read completely, and it is omitted so a gate fails it.
func amdState(s amdSample) counterState {
	switch {
	case !s.isAMDNode():
		return counterUnknown
	case s.withRAS == 0:
		return counterUnmeasurable
	case s.read && s.withRAS == s.cards:
		return counterOK
	default:
		return counterUnknown
	}
}

// emitAMD writes the AMD counters into the same canonical names the NVIDIA path
// uses, plus the evidence that explains them.
//
// eccErrors carries the UNCORRECTED total, which is the same measurand the
// NVIDIA path puts there (ecc.errors.uncorrected.aggregate.total) rather than
// merely a similarly-named one. Corrected errors go to their own evidence key:
// they are a wear signal, not a fault, and folding them into a gated counter
// would fail nodes for doing exactly what ECC is for.
func emitAMD(out *emitter, before, after amdSample) {
	if !after.isAMDNode() {
		return
	}

	// A node holding BOTH vendors' accelerators is not a case this project
	// supports — one pod gets one image, which is why the imagesByVendor model
	// keys on a single vendor per node — and it is not one to paper over here.
	// If the NVIDIA path already claimed eccErrors, its value stands and the
	// situation is RECORDED, because silently overwriting would report one
	// vendor's errors as the node's total and silently summing would combine
	// two counters nobody has established mean the same thing on one machine.
	if _, taken := out.get(keyECCErrors); taken {
		out.set("ecc_vendor_conflict", "nvidia+amd")
		out.setInt("amd_gpus", int64(after.cards))
		return
	}

	out.setInt("amd_gpus", int64(after.cards))
	out.setInt("amd_gpus_with_ras", int64(after.withRAS))
	out.set("ecc_source", "amdgpu-sysfs")

	state := amdState(after)
	out.setCounter(keyECCErrors, after.uncorrected, state)

	if after.read {
		out.setInt("amd_ras_blocks", int64(after.blocks))
		out.setInt("ecc_corrected_aggregate", after.corrected)

		// An uncorrected error that happened WHILE we watched is a different
		// and worse thing than an old one, exactly as on the NVIDIA path.
		if before.read {
			if d, ok := delta(before.uncorrected, after.uncorrected, true, true); ok {
				out.setInt("ecc_uncorrected_window", d)
			}
		}
	}

	// The omission is explained rather than silent. Without this, a node whose
	// eccErrors is missing looks identical to one where the metric was never
	// implemented, and the operator's fail-closed verdict has no reason
	// attached to it.
	if len(after.failures) > 0 {
		out.setInt("amd_ras_unreadable", int64(len(after.failures)))
		out.set("amd_ras_unreadable_files", strings.Join(after.failures, ","))
	}
}

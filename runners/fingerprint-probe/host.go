// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// hostIdentity is what the host's own sysfs says about its CPUs, memory and
// NVMe drives (#541). It answers "is this the machine we bought": a missing
// performance cluster, a DIMM that did not train, or a drive in the wrong slot
// all pass every load test and are visible here.
//
// Every field is read from the same read-only sysfs mount the PCI scan uses,
// so this adds no privilege and no mount. Each field carries its own "known"
// flag, because a value the runner could not read is OMITTED rather than
// reported as zero: a gate on it must fail closed, not certify a guess.
type hostIdentity struct {
	cpuCount      int
	cpuCountKnown bool

	// perfCores counts the CPUs in the highest cpu_capacity class. On an
	// asymmetric part (GB10: 10x Cortex-X925 at 997-1024 plus 10x Cortex-A725
	// at 718-731) a lost performance cluster changes this and nothing else.
	perfCores int
	// perfCoresState is "known", "n/a" (the kernel publishes no capacity
	// classes: every core is the same, positively), or "" (could not read).
	perfCoresState string

	memTotalKiB   int64
	memTotalKnown bool

	nvme []nvmeController
}

type nvmeController struct {
	name       string // nvme0
	model      string
	pciAddress string // the slot, e.g. 0004:01:00.0
	bytes      int64  // sum over the controller's namespaces
	// serialDigest identifies the drive without naming it (#540): the first
	// 16 hex of SHA-256 over the serial. Two nodes reporting the same digest
	// are one drive seen twice — a cloned image or the same machine reached
	// under two names. The serial itself is never emitted.
	serialDigest string
}

// capacityClassFraction is how close to the maximum cpu_capacity a core must be
// to count as a performance core. GB10's two classes sit at 718-731 and
// 997-1024, so any value between 0.72 and 0.97 separates them; 0.9 leaves
// margin on both sides for a firmware that reports the classes a little apart.
const capacityClassFraction = 0.9

func scanHost(sysfs string) hostIdentity {
	var h hostIdentity

	cpuDir := filepath.Join(sysfs, "devices", "system", "cpu")
	if b, err := os.ReadFile(filepath.Join(cpuDir, "present")); err == nil {
		if cpus, err := parseCPUList(strings.TrimSpace(string(b))); err == nil {
			h.cpuCount, h.cpuCountKnown = len(cpus), true
			h.perfCores, h.perfCoresState = performanceCores(cpuDir, cpus)
		}
	}

	nodes, _ := filepath.Glob(filepath.Join(sysfs, "devices", "system", "node", "node*", "meminfo"))
	var total int64
	for _, n := range nodes {
		kib, ok := readMemTotalKiB(n)
		if !ok {
			// One unreadable NUMA node makes the sum a floor on an unknown total,
			// which is not a measurement anything should be gated against.
			total = -1
			break
		}
		total += kib
	}
	if len(nodes) > 0 && total >= 0 {
		h.memTotalKiB, h.memTotalKnown = total, true
	}

	h.nvme = scanNVMe(sysfs)
	return h
}

// parseCPUList parses the kernel's cpulist format ("0-9,12,14-15").
func parseCPUList(s string) ([]int, error) {
	if s == "" {
		return nil, fmt.Errorf("empty cpu list")
	}
	var out []int
	for _, part := range strings.Split(s, ",") {
		lo, hi, isRange := strings.Cut(part, "-")
		a, err := strconv.Atoi(lo)
		if err != nil {
			return nil, err
		}
		b := a
		if isRange {
			if b, err = strconv.Atoi(hi); err != nil {
				return nil, err
			}
		}
		if b < a {
			return nil, fmt.Errorf("descending range %q", part)
		}
		for i := a; i <= b; i++ {
			out = append(out, i)
		}
	}
	return out, nil
}

func performanceCores(cpuDir string, cpus []int) (int, string) {
	caps := make([]int, 0, len(cpus))
	missing := 0
	for _, c := range cpus {
		b, err := os.ReadFile(filepath.Join(cpuDir, fmt.Sprintf("cpu%d", c), "cpu_capacity"))
		if err != nil {
			missing++
			continue
		}
		v, err := strconv.Atoi(strings.TrimSpace(string(b)))
		if err != nil {
			return 0, ""
		}
		caps = append(caps, v)
	}
	switch {
	case missing == len(cpus):
		// No core publishes a capacity: the kernel is describing a symmetric
		// part. That is established, so it is declared rather than omitted.
		return 0, "n/a"
	case missing > 0:
		// Some cores publish one and some do not. Offline cores do this; the
		// count would be wrong in a direction nobody could see.
		return 0, ""
	}
	max := 0
	for _, v := range caps {
		if v > max {
			max = v
		}
	}
	n := 0
	for _, v := range caps {
		if float64(v) >= capacityClassFraction*float64(max) {
			n++
		}
	}
	return n, "known"
}

func readMemTotalKiB(path string) (int64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(b), "\n") {
		// "Node 0 MemTotal:       125489732 kB"
		f := strings.Fields(line)
		if len(f) == 5 && f[2] == "MemTotal:" && f[4] == "kB" {
			v, err := strconv.ParseInt(f[3], 10, 64)
			return v, err == nil
		}
	}
	return 0, false
}

func scanNVMe(sysfs string) []nvmeController {
	ctrls, _ := filepath.Glob(filepath.Join(sysfs, "class", "nvme", "nvme*"))
	sort.Strings(ctrls)
	var out []nvmeController
	for _, c := range ctrls {
		n := nvmeController{name: filepath.Base(c)}
		if b, err := os.ReadFile(filepath.Join(c, "model")); err == nil {
			n.model = strings.Join(strings.Fields(string(b)), " ")
		}
		if b, err := os.ReadFile(filepath.Join(c, "serial")); err == nil {
			if serial := strings.TrimSpace(string(b)); serial != "" {
				sum := sha256.Sum256([]byte(serial))
				n.serialDigest = hex.EncodeToString(sum[:])[:16]
			}
		}
		if dev, err := filepath.EvalSymlinks(filepath.Join(c, "device")); err == nil {
			n.pciAddress = filepath.Base(dev)
		}
		// /sys/block/<ns>/size is ALWAYS in 512-byte sectors, whatever the
		// namespace's logical block size.
		nss, _ := filepath.Glob(filepath.Join(c, n.name+"n*", "size"))
		for _, s := range nss {
			if b, err := os.ReadFile(s); err == nil {
				if v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64); err == nil {
					n.bytes += v * 512
				}
			}
		}
		out = append(out, n)
	}
	return out
}

// reportHost emits the identity. Counts and sizes are Acceptance: a node that
// should have 20 CPUs, 128 GB and one 4 TB drive and reports less has lost
// something. The model and slot strings are Evidence, for the reason the PCI
// identity strings are.
func reportHost(h hostIdentity) {
	if h.cpuCountKnown {
		metric("cpuCount", strconv.Itoa(h.cpuCount))
	}
	switch h.perfCoresState {
	case "known":
		metric("performanceCoreCount", strconv.Itoa(h.perfCores))
	case "n/a":
		metric("performanceCoreCount", "n/a")
	}
	if h.memTotalKnown {
		// Decimal gigabytes, one decimal place: the kernel reports KiB.
		metric("memoryTotalGB", strconv.FormatFloat(float64(h.memTotalKiB)*1024/1e9, 'f', 1, 64))
	}

	// Zero drives is a measurement when the class directory was readable.
	metric("nvmeCount", strconv.Itoa(len(h.nvme)))
	if len(h.nvme) == 0 {
		return
	}
	var models, slots, digests []string
	var total int64
	for _, n := range h.nvme {
		models = append(models, n.model)
		slots = append(slots, n.pciAddress)
		total += n.bytes
		if n.serialDigest != "" {
			digests = append(digests, n.serialDigest)
		}
	}
	// Emitted only when EVERY drive's serial was read: a partial list could
	// hide the one drive two nodes share.
	if len(digests) == len(h.nvme) {
		metric("nvmeSerialDigests", strings.Join(digests, ","))
	}
	metric("nvmeModels", strings.Join(models, ","))
	metric("nvmePciAddresses", strings.Join(slots, ","))
	metric("nvmeTotalCapacityGB", strconv.FormatFloat(float64(total)/1e9, 'f', 1, 64))
}

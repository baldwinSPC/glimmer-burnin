package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// sysfs probes: PCIe AER, and fabric NIC link state.
//
// Both read the host's sysfs, which an ordinary unprivileged container already
// has mounted read-only at /sys — no device mounts, no capabilities. That is
// why these two probes are the ones that keep working when the kernel-log probe
// cannot run, and why a host-health run on a locked-down node still produces a
// verdict instead of an Error.
//
// The one thing they DO need is the host's network namespace: /sys/class/net is
// namespaced, so inside a pod's own netns it shows the pod's veth and nothing
// else. Set hostNetwork: true on the BurnInTest (the spec supports it) or the
// NIC probe reports nic_count=0 and emits no nicLinkDownEvents.
//
// BURNIN_SYSFS_ROOT relocates the tree, which is what makes both probes
// testable against a fixture directory.

// ---------------------------------------------------------------------------
// PCIe Advanced Error Reporting
// ---------------------------------------------------------------------------

type aerSample struct {
	present bool
	devices int
	// Each severity has its own presence flag: a kernel that exposes one file
	// and not another must not have the missing one reported as a zero.
	correctable, nonfatal, fatal             int64
	haveCorrectable, haveNonfatal, haveFatal bool
	// replay is the subset of correctable errors that are link replays:
	// "Timeout" is the replay timer expiring, "Rollover" is REPLAY_NUM
	// rolling over. They are the same measurand as NVML's replay counter, for
	// hosts (or accelerators) where NVML cannot supply it.
	replay int64
}

// probeAER sums the per-device AER counters the kernel exposes.
//
// These are link-integrity counters, and a link that is quietly retrying is the
// usual precursor to a bandwidth shortfall that a later NCCL or ib_write_bw
// test would blame on the fabric.
func probeAER(sysfsRoot string) aerSample {
	var s aerSample
	devs, err := filepath.Glob(filepath.Join(sysfsRoot, "bus", "pci", "devices", "*"))
	if err != nil {
		return s
	}
	for _, dev := range devs {
		counted := false
		if counts, ok := readAERFile(filepath.Join(dev, "aer_dev_correctable")); ok {
			s.correctable += sumCounts(counts)
			s.replay += counts["Timeout"] + counts["Rollover"]
			s.haveCorrectable = true
			counted = true
		}
		if counts, ok := readAERFile(filepath.Join(dev, "aer_dev_nonfatal")); ok {
			s.nonfatal += sumCounts(counts)
			s.haveNonfatal = true
			counted = true
		}
		if counts, ok := readAERFile(filepath.Join(dev, "aer_dev_fatal")); ok {
			s.fatal += sumCounts(counts)
			s.haveFatal = true
			counted = true
		}
		if counted {
			s.devices++
			s.present = true
		}
	}
	return s
}

// readAERFile parses one aer_dev_* file: lines of "<ErrorName> <count>".
func readAERFile(path string) (map[string]int64, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}
	counts := map[string]int64{}
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		n, err := strconv.ParseInt(f[1], 10, 64)
		if err != nil {
			continue
		}
		counts[f[0]] = n
	}
	return counts, true
}

// sumCounts totals one AER file, skipping any TOTAL_* line so a kernel that
// prints a summary row cannot make every count double.
func sumCounts(counts map[string]int64) int64 {
	var total int64
	for name, n := range counts {
		if strings.HasPrefix(name, "TOTAL") {
			continue
		}
		total += n
	}
	return total
}

func emitAER(out *emitter, before, after aerSample) {
	if !after.present {
		out.set("aer_status", "absent")
		return
	}
	out.set("aer_status", "ok")
	out.setInt("pcie_aer_devices", int64(after.devices))
	out.setIntIf("pcie_aer_correctable_total", after.correctable, after.haveCorrectable)
	out.setIntIf("pcie_aer_nonfatal_total", after.nonfatal, after.haveNonfatal)
	out.setIntIf("pcie_aer_fatal_total", after.fatal, after.haveFatal)

	if d, ok := delta(before.correctable, after.correctable, before.haveCorrectable, after.haveCorrectable); ok {
		out.setInt("pcie_aer_correctable", d)
	}
	if d, ok := delta(before.nonfatal, after.nonfatal, before.haveNonfatal, after.haveNonfatal); ok {
		out.setInt("pcie_aer_nonfatal", d)
	}
	if d, ok := delta(before.fatal, after.fatal, before.haveFatal, after.haveFatal); ok {
		out.setInt("pcie_aer_fatal", d)
	}
}

// reconcilePCIeReplay produces the gated pcie_replay_count from whichever
// source this host has.
//
// NVML is preferred because its counter is scoped to the accelerator's own
// link, which is the link acceptance cares about; the sysfs AER fallback is a
// node-wide sum and therefore coarser. The source is emitted alongside so a
// stored measurement can never be silently compared against one taken a
// different way.
func reconcilePCIeReplay(out *emitter, gpu0, gpu1 gpuSample, aer0, aer1 aerSample) {
	if gpu0.ok() && gpu1.ok() {
		b, bOK := gpu0.tbl.sum(fPCIeReplay)
		a, aOK := gpu1.tbl.sum(fPCIeReplay)
		if d, ok := delta(b, a, bOK, aOK); ok {
			out.setInt(keyPCIeReplayCount, d)
			out.set("pcie_replay_source", "nvml")
			return
		}
	}
	if aer0.present && aer1.present {
		if d, ok := delta(aer0.replay, aer1.replay, true, true); ok {
			out.setInt(keyPCIeReplayCount, d)
			out.set("pcie_replay_source", "sysfsAer")
			return
		}
	}
	out.set("pcie_replay_source", "none")
}

// ---------------------------------------------------------------------------
// fabric NIC link state
// ---------------------------------------------------------------------------

type nicSample struct {
	present bool
	count   int // physical ethernet interfaces considered
	up      int

	ethDown   int64
	ethDownOK bool
	ibDown    int64
	ibDownOK  bool
	ibPorts   int
	ibActive  int
}

func (n nicSample) linkDownTotal() (int64, bool) {
	if !n.present || !n.ethDownOK || !n.ibDownOK {
		return 0, false
	}
	return n.ethDown + n.ibDown, true
}

// probeNIC reads link state and link-down counters for the node's physical
// NICs and InfiniBand ports.
//
// "Physical" means the interface has a backing device in sysfs. That filter is
// what keeps the probe honest inside a container: veth, bridge, and tunnel
// interfaces have no device link, so a pod that was NOT given hostNetwork sees
// zero qualifying interfaces and the runner emits no link-down metric at all —
// rather than a confident zero measured from a virtual interface that could
// never have failed.
func probeNIC(sysfsRoot string) nicSample {
	s := nicSample{ethDownOK: true, ibDownOK: true}

	entries, err := os.ReadDir(filepath.Join(sysfsRoot, "class", "net"))
	if err == nil {
		for _, e := range entries {
			name := e.Name()
			if name == "lo" {
				continue
			}
			ifdir := filepath.Join(sysfsRoot, "class", "net", name)
			if _, err := os.Stat(filepath.Join(ifdir, "device")); err != nil {
				continue // virtual interface
			}
			s.present = true
			s.count++
			if state, ok := readSysfsString(filepath.Join(ifdir, "operstate")); ok && state == "up" {
				s.up++
			}
			if n, ok := readSysfsInt(filepath.Join(ifdir, "carrier_down_count")); ok {
				s.ethDown += n
			} else {
				// Kernel too old for the counter: say so by withholding the
				// aggregate rather than under-reporting it.
				s.ethDownOK = false
			}
		}
	}

	ibDevs, err := os.ReadDir(filepath.Join(sysfsRoot, "class", "infiniband"))
	if err == nil {
		for _, dev := range ibDevs {
			portsDir := filepath.Join(sysfsRoot, "class", "infiniband", dev.Name(), "ports")
			ports, err := os.ReadDir(portsDir)
			if err != nil {
				continue
			}
			for _, p := range ports {
				s.present = true
				s.ibPorts++
				// state reads like "4: ACTIVE".
				if state, ok := readSysfsString(filepath.Join(portsDir, p.Name(), "state")); ok &&
					strings.Contains(strings.ToUpper(state), "ACTIVE") {
					s.ibActive++
				}
				if n, ok := readSysfsInt(filepath.Join(portsDir, p.Name(), "counters", "link_downed")); ok {
					s.ibDown += n
				} else {
					s.ibDownOK = false
				}
			}
		}
	}
	return s
}

func emitNIC(out *emitter, before, after nicSample) {
	if !after.present {
		out.set("nic_status", "absent")
		return
	}
	total, totalOK := after.linkDownTotal()
	if totalOK {
		out.set("nic_status", "ok")
	} else {
		out.set("nic_status", "partial")
	}
	out.setInt("nic_count", int64(after.count))
	out.setInt("nic_up", int64(after.up))
	if after.ibPorts > 0 {
		out.setInt("ib_ports", int64(after.ibPorts))
		out.setInt("ib_ports_active", int64(after.ibActive))
	}
	out.setIntIf("eth_link_down_total", after.ethDown, after.ethDownOK)
	out.setIntIf("ib_link_down_total", after.ibDown, after.ibDownOK)
	out.setIntIf("nic_link_down_total", total, totalOK)

	beforeTotal, beforeOK := before.linkDownTotal()
	if d, ok := delta(beforeTotal, total, beforeOK, totalOK); ok {
		out.setInt(keyNICLinkDown, d)
	}
}

func readSysfsString(path string) (string, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(string(data)), true
}

func readSysfsInt(path string) (int64, bool) {
	s, ok := readSysfsString(path)
	if !ok {
		return 0, false
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, false
	}
	return n, true
}

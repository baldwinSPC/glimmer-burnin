package main

import (
	"os"
	"path/filepath"
	"sort"
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

	// fabric holds the per-counter totals summed across every IB/RoCE port, and
	// whether each was readable everywhere it was looked for. Absence is per
	// COUNTER, not per probe: kernels and drivers expose different subsets, and
	// withholding every counter because one was missing would throw away the
	// evidence that is there.
	fabric map[string]int64
	// fabricOK is false for a counter that was unreadable on at least one port,
	// or that was SATURATED — see fabricCounters.
	fabricOK map[string]bool
	// fabricSaturated names the counters observed AT their ceiling. It is kept
	// apart from fabricOK because the two mean opposite things about the node:
	// a counter this driver does not expose is normal and says nothing, while a
	// PEGGED counter is a link whose error history is unreadable and is very
	// likely bad. Collapsing them into one "partial" state made the common,
	// meaningless case indistinguishable from the alarming one.
	fabricSaturated map[string]bool
}

// fabricCounters are the IB/RoCE error counters worth reading, with the width
// each saturates at.
//
// THE SATURATION IS THE TRAP, and it fails towards certifying broken hardware.
// The IB-spec PMA counters do not wrap — they PIN at their maximum and stay
// there until something resets them. A port that has been throwing symbol errors
// for a month sits at 65535 forever, so the delta across a burn-in window is
// zero and the link reads as clean. That is a broken link reported as healthy,
// which is the one direction this project never accepts.
//
// So a counter observed at its ceiling is not a measurement. It is reported
// unmeasurable — the same rule the ECC probe already applies — and a threshold
// on it then fails closed rather than passing on a zero that means "pegged".
//
// The hw_counters are driver-provided rather than IB-spec: they are 64-bit and
// do not saturate in any practical time, so their ceiling is recorded as 0
// meaning "no ceiling".
var fabricCounters = []struct {
	// dir is "counters" (IB-spec PMA) or "hw_counters" (driver-provided).
	dir string
	// file is the sysfs name; key is the runner's metric key.
	file, key string
	// ceiling is the saturating maximum, or 0 when the counter cannot saturate.
	ceiling int64
	// why says what a non-zero value means, for the log.
	why string
}{
	// IB-spec PMA counters. Widths are from the InfiniBand Architecture
	// Specification's PortCounters attribute.
	{"counters", "symbol_error", "ib_symbol_errors", 65535,
		"physical-layer errors on the wire: a cable, a transceiver, or a port"},
	{"counters", "link_error_recovery", "ib_link_error_recovery", 255,
		"the link retrained itself; a marginal physical link does this repeatedly"},
	{"counters", "port_rcv_errors", "ib_port_rcv_errors", 65535,
		"packets received with an error"},
	{"counters", "port_rcv_remote_physical_errors", "ib_rcv_remote_physical_errors", 65535,
		"the PEER reported a physical error — the fault is on the other end or in between"},
	{"counters", "local_link_integrity_errors", "ib_local_link_integrity_errors", 15,
		"local link integrity threshold exceeded; a very small ceiling, so it pegs quickly"},
	{"counters", "excessive_buffer_overrun_errors", "ib_excessive_buffer_overruns", 15,
		"flow control broke down; also a very small ceiling"},
	{"counters", "port_xmit_discards", "ib_port_xmit_discards", 65535,
		"packets dropped on transmit, usually congestion or a downed peer"},
	{"counters", "VL15_dropped", "ib_vl15_dropped", 65535,
		"subnet-management packets dropped"},

	// Driver-provided. 64-bit, no practical ceiling.
	{"hw_counters", "packet_seq_err", "roce_packet_seq_errors", 0,
		"out-of-sequence packets: retransmission, and a direct drag on collective throughput"},
	{"hw_counters", "out_of_sequence", "roce_out_of_sequence", 0,
		"receiver saw a gap in the sequence"},
	{"hw_counters", "out_of_buffer", "roce_out_of_buffer", 0,
		"no receive buffer was available; the classic RoCE congestion signature"},
	{"hw_counters", "rnr_nak_retry_err", "roce_rnr_nak_retries", 0,
		"receiver-not-ready retries exhausted"},
	{"hw_counters", "local_ack_timeout_err", "roce_local_ack_timeouts", 0,
		"an acknowledgement never arrived"},
	{"hw_counters", "rx_icrc_encapsulated", "roce_icrc_errors", 0,
		"CRC failures on encapsulated RoCE frames — corruption on the wire"},
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
				s.readFabricCounters(filepath.Join(portsDir, p.Name()))
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

// readFabricCounters accumulates one port's error counters.
//
// A counter sitting AT its ceiling is refused rather than added: see
// fabricCounters for why a saturated counter is a broken link that reads as a
// clean one. Refusal is per counter and per port — one pegged counter on one
// port withholds that counter for the node, and says nothing about the others.
func (s *nicSample) readFabricCounters(portDir string) {
	if s.fabric == nil {
		s.fabric, s.fabricOK = map[string]int64{}, map[string]bool{}
		s.fabricSaturated = map[string]bool{}
	}
	for _, c := range fabricCounters {
		if _, seen := s.fabricOK[c.key]; !seen {
			s.fabricOK[c.key] = true
		}
		n, ok := readSysfsInt(filepath.Join(portDir, c.dir, c.file))
		switch {
		case !ok:
			// Not exposed by this kernel or driver. Withhold the aggregate
			// rather than under-report it.
			s.fabricOK[c.key] = false
		case c.ceiling > 0 && n >= c.ceiling:
			// Pegged. The true value is unknown and at least this, so a delta
			// across the window is meaningless — and would read as zero.
			s.fabricOK[c.key] = false
			s.fabricSaturated[c.key] = true
		default:
			s.fabric[c.key] += n
		}
	}
}

// emitFabricCounters reports the DELTA over the observation window, plus the
// since-boot total as evidence.
//
// The delta is the thresholdable form. A since-boot total counts every error
// since the machine last rebooted, including ones from a cable somebody has
// already replaced, so gating on it condemns a node for its history rather than
// for its present.
func emitFabricCounters(out *emitter, before, after nicSample) {
	if after.fabric == nil {
		// No IB or RoCE port was visible at all — which happens legitimately,
		// because /sys/class/infiniband may be invisible inside the pod depending
		// on the RDMA namespace mode. Distinct from "visible and clean", and a
		// threshold on any of these then fails closed.
		out.set("ib_counter_status", "absent")
		return
	}
	out.set("ib_counter_status", "ok")

	for _, c := range fabricCounters {
		if !after.fabricOK[c.key] {
			// Either this driver does not expose the counter — normal, and
			// says nothing about the node — or it is pegged, which is reported
			// as its own signal below. Either way it is OMITTED rather than
			// zeroed, so a gate on it fails closed.
			continue
		}
		out.setInt(c.key+"_total", after.fabric[c.key])
		if d, ok := delta(before.fabric[c.key], after.fabric[c.key],
			before.fabricOK[c.key], after.fabricOK[c.key]); ok {
			out.setInt(c.key, d)
		}
	}

	// SATURATION IS ITS OWN FINDING, and a gateable one.
	//
	// It is not "a counter was missing" — it is a counter pinned at its maximum,
	// which means this port has been erroring long enough to peg a 16-bit (or,
	// for link integrity, a 4-bit) counter and has not been reset since. The
	// delta across any window is then zero, so without this the link reads as
	// clean. A profile can gate `ibCountersSaturated Equal 0` and catch exactly
	// the case the delta cannot see.
	saturated := make([]string, 0, len(after.fabricSaturated))
	for key := range after.fabricSaturated {
		saturated = append(saturated, key)
	}
	sort.Strings(saturated)
	out.setInt("ib_counters_saturated", int64(len(saturated)))
	if len(saturated) > 0 {
		out.set("ib_saturated_counters", strings.Join(saturated, ","))
	}
}

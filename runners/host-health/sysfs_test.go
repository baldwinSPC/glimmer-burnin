package main

import (
	"os"
	"path/filepath"
	"testing"
)

// aerCorrectable is the shape the kernel prints for aer_dev_correctable.
const aerCorrectable = `RxErr 0
BadTLP 2
BadDLLP 0
Rollover 3
Timeout 4
NonFatalErr 0
CorrIntErr 0
HeaderOF 0
`

func TestReadAERFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "aer_dev_correctable")
	mustWrite(t, path, aerCorrectable)

	counts, ok := readAERFile(path)
	if !ok {
		t.Fatal("readAERFile reported failure on a well-formed file")
	}
	if counts["Timeout"] != 4 || counts["Rollover"] != 3 || counts["BadTLP"] != 2 {
		t.Fatalf("counts = %v", counts)
	}
	if got := sumCounts(counts); got != 9 {
		t.Errorf("sumCounts = %d, want 9", got)
	}

	if _, ok := readAERFile(filepath.Join(dir, "missing")); ok {
		t.Error("a missing AER file must report absence, not zero")
	}
}

// A kernel that prints a TOTAL_ERR_COR summary row must not make every count
// double.
func TestSumCountsIgnoresTotals(t *testing.T) {
	counts := map[string]int64{"BadTLP": 2, "Timeout": 3, "TOTAL_ERR_COR": 5}
	if got := sumCounts(counts); got != 5 {
		t.Errorf("sumCounts = %d, want 5 (the TOTAL row must be skipped)", got)
	}
}

func TestProbeAER(t *testing.T) {
	root := t.TempDir()
	dev := filepath.Join(root, "bus", "pci", "devices", "0000:01:00.0")
	mustWrite(t, filepath.Join(dev, "aer_dev_correctable"), aerCorrectable)
	mustWrite(t, filepath.Join(dev, "aer_dev_fatal"), "Undefined 0\nDataLinkProtocol 1\n")
	// A second device with no AER capability at all must not be counted.
	mustWrite(t, filepath.Join(root, "bus", "pci", "devices", "0000:02:00.0", "vendor"), "0x10de\n")

	s := probeAER(root)
	if !s.present {
		t.Fatal("probeAER reported absent on a tree with AER files")
	}
	if s.devices != 1 {
		t.Errorf("devices = %d, want 1", s.devices)
	}
	if s.correctable != 9 {
		t.Errorf("correctable = %d, want 9", s.correctable)
	}
	if s.fatal != 1 {
		t.Errorf("fatal = %d, want 1", s.fatal)
	}
	// Replay = Timeout (replay timer expiry) + Rollover (REPLAY_NUM rollover).
	if s.replay != 7 {
		t.Errorf("replay = %d, want 7", s.replay)
	}
}

func TestProbeAERAbsentTree(t *testing.T) {
	if s := probeAER(t.TempDir()); s.present {
		t.Error("probeAER must report absent when sysfs has no PCI devices")
	}
}

func nicFixture(t *testing.T, ethDown, ibDown string) string {
	t.Helper()
	root := t.TempDir()
	net := filepath.Join(root, "class", "net")

	// A real NIC: has a backing device, is up, and reports the counter.
	mustWrite(t, filepath.Join(net, "eth0", "operstate"), "up\n")
	mustWrite(t, filepath.Join(net, "eth0", "carrier_down_count"), ethDown+"\n")
	if err := os.MkdirAll(filepath.Join(net, "eth0", "device"), 0o755); err != nil {
		t.Fatal(err)
	}

	// A veth: no backing device, so it is not a fabric NIC and must be ignored.
	mustWrite(t, filepath.Join(net, "veth9a1", "operstate"), "up\n")
	mustWrite(t, filepath.Join(net, "veth9a1", "carrier_down_count"), "42\n")

	// Loopback is skipped by name even though it has a device directory.
	mustWrite(t, filepath.Join(net, "lo", "operstate"), "unknown\n")
	if err := os.MkdirAll(filepath.Join(net, "lo", "device"), 0o755); err != nil {
		t.Fatal(err)
	}

	// An InfiniBand port.
	port := filepath.Join(root, "class", "infiniband", "mlx5_0", "ports", "1")
	mustWrite(t, filepath.Join(port, "state"), "4: ACTIVE\n")
	mustWrite(t, filepath.Join(port, "counters", "link_downed"), ibDown+"\n")
	return root
}

func TestProbeNIC(t *testing.T) {
	root := nicFixture(t, "3", "2")
	s := probeNIC(root)

	if !s.present {
		t.Fatal("probeNIC reported absent on a tree with a physical NIC")
	}
	if s.count != 1 {
		t.Errorf("count = %d, want 1 (veth and lo must be excluded)", s.count)
	}
	if s.up != 1 {
		t.Errorf("up = %d, want 1", s.up)
	}
	if s.ibPorts != 1 || s.ibActive != 1 {
		t.Errorf("ib ports = %d active = %d, want 1/1", s.ibPorts, s.ibActive)
	}
	total, ok := s.linkDownTotal()
	if !ok {
		t.Fatal("linkDownTotal not ok, want ethernet + infiniband summed")
	}
	if total != 5 {
		t.Errorf("linkDownTotal = %d, want 5", total)
	}
}

// A pod without hostNetwork sees only its own veth. The probe must then report
// nothing rather than a confident zero from an interface that could never fail.
func TestProbeNICInsidePodNetnsReportsAbsent(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "class", "net", "eth0", "operstate"), "up\n")
	mustWrite(t, filepath.Join(root, "class", "net", "lo", "operstate"), "unknown\n")

	s := probeNIC(root)
	if s.present {
		t.Fatal("a veth-only tree must not count as fabric NICs")
	}
	out := newEmitter()
	emitNIC(out, s, s)
	if _, ok := out.get(keyNICLinkDown); ok {
		t.Error("nic_link_down must be omitted when no physical NIC is visible")
	}
	if got, _ := out.get("nic_status"); got != "absent" {
		t.Errorf("nic_status = %q, want absent", got)
	}
}

// A kernel too old for carrier_down_count must withhold the aggregate rather
// than under-report it.
func TestProbeNICWithoutCounterIsPartial(t *testing.T) {
	root := t.TempDir()
	eth := filepath.Join(root, "class", "net", "eth0")
	mustWrite(t, filepath.Join(eth, "operstate"), "up\n")
	if err := os.MkdirAll(filepath.Join(eth, "device"), 0o755); err != nil {
		t.Fatal(err)
	}

	s := probeNIC(root)
	if !s.present {
		t.Fatal("the NIC itself is visible and must be counted")
	}
	if _, ok := s.linkDownTotal(); ok {
		t.Error("linkDownTotal must not be ok when an interface lacks the counter")
	}
	out := newEmitter()
	emitNIC(out, s, s)
	if got, _ := out.get("nic_status"); got != "partial" {
		t.Errorf("nic_status = %q, want partial", got)
	}
	if _, ok := out.get(keyNICLinkDown); ok {
		t.Error("nic_link_down must be omitted when the aggregate is incomplete")
	}
	if got, _ := out.get("nic_count"); got != "1" {
		t.Errorf("nic_count = %q, want 1 — the NIC was still seen", got)
	}
}

func TestEmitNICWindowDelta(t *testing.T) {
	before := probeNIC(nicFixture(t, "3", "2"))
	after := probeNIC(nicFixture(t, "5", "2"))

	out := newEmitter()
	emitNIC(out, before, after)
	if got, _ := out.get(keyNICLinkDown); got != "2" {
		t.Errorf("%s = %q, want 2", keyNICLinkDown, got)
	}
	if got, _ := out.get("nic_link_down_total"); got != "7" {
		t.Errorf("nic_link_down_total = %q, want 7", got)
	}
}

func TestReconcilePCIeReplayPrefersNVML(t *testing.T) {
	gpu0 := gpuSample{status: "ok", tbl: &gpuTable{rows: 1, col: map[string][]string{fPCIeReplay: {"10"}}}}
	gpu1 := gpuSample{status: "ok", tbl: &gpuTable{rows: 1, col: map[string][]string{fPCIeReplay: {"14"}}}}
	aer0 := aerSample{present: true, replay: 100}
	aer1 := aerSample{present: true, replay: 200}

	out := newEmitter()
	reconcilePCIeReplay(out, gpu0, gpu1, aer0, aer1)
	if got, _ := out.get(keyPCIeReplayCount); got != "4" {
		t.Errorf("%s = %q, want 4 from NVML", keyPCIeReplayCount, got)
	}
	if got, _ := out.get("pcie_replay_source"); got != "nvml" {
		t.Errorf("pcie_replay_source = %q, want nvml", got)
	}
}

func TestReconcilePCIeReplayFallsBackToSysfs(t *testing.T) {
	out := newEmitter()
	reconcilePCIeReplay(out,
		gpuSample{status: "absent"}, gpuSample{status: "absent"},
		aerSample{present: true, replay: 5}, aerSample{present: true, replay: 9})
	if got, _ := out.get(keyPCIeReplayCount); got != "4" {
		t.Errorf("%s = %q, want 4 from sysfs AER", keyPCIeReplayCount, got)
	}
	if got, _ := out.get("pcie_replay_source"); got != "sysfsAer" {
		t.Errorf("pcie_replay_source = %q, want sysfsAer", got)
	}
}

func TestReconcilePCIeReplayEmitsNothingWithoutASource(t *testing.T) {
	out := newEmitter()
	reconcilePCIeReplay(out, gpuSample{status: "absent"}, gpuSample{status: "absent"}, aerSample{}, aerSample{})
	if _, ok := out.get(keyPCIeReplayCount); ok {
		t.Error("pcie_replay_count must be omitted when neither source exists")
	}
	if got, _ := out.get("pcie_replay_source"); got != "none" {
		t.Errorf("pcie_replay_source = %q, want none", got)
	}
}

// A SATURATED fabric counter is not a measurement — the trap that fails towards
// certifying broken hardware.
//
// The IB-spec PMA counters do not wrap. They PIN at their maximum and stay there
// until something resets them, so a port that has been throwing symbol errors
// for a month sits at 65535 forever — and the delta across a burn-in window is
// zero. A broken link then reads as a clean one, which is the one direction this
// project never accepts.
func TestSaturatedFabricCounterIsNotReportedAsClean(t *testing.T) {
	build := func(symbolErrors string) nicSample {
		root := t.TempDir()
		port := filepath.Join(root, "class", "infiniband", "mlx5_0", "ports", "1")
		mustWrite(t, filepath.Join(port, "state"), "4: ACTIVE\n")
		mustWrite(t, filepath.Join(port, "counters", "link_downed"), "0\n")
		mustWrite(t, filepath.Join(port, "counters", "symbol_error"), symbolErrors+"\n")
		// A second, healthy counter proves the refusal is PER COUNTER.
		mustWrite(t, filepath.Join(port, "counters", "port_rcv_errors"), "3\n")
		return probeNIC(root)
	}

	t.Run("pegged at the ceiling", func(t *testing.T) {
		before, after := build("65535"), build("65535")
		out := newEmitter()
		emitFabricCounters(out, before, after)

		if v, ok := out.get("ib_symbol_errors"); ok {
			t.Errorf("ib_symbol_errors=%q was emitted for a counter pinned at its ceiling — "+
				"the delta is zero and the link reads as clean", v)
		}
		// The counters that were NOT pegged still report.
		if v, _ := out.get("ib_port_rcv_errors"); v != "0" {
			t.Errorf("a healthy counter was withheld because a different one was pegged: %q", v)
		}
		// Saturation is its OWN signal, gateable, and distinct from a counter
		// this driver simply does not expose. Without it the pegged link is
		// invisible: the delta is zero and everything reads clean.
		if v, _ := out.get("ib_counters_saturated"); v != "1" {
			t.Errorf("ib_counters_saturated = %q, want 1 — a pegged counter is a link whose "+
				"error history is unreadable, and the delta cannot see it", v)
		}
		if names, _ := out.get("ib_saturated_counters"); names != "ib_symbol_errors" {
			t.Errorf("ib_saturated_counters = %q, want the pegged counter named", names)
		}
	})

	t.Run("below the ceiling is measured normally", func(t *testing.T) {
		before, after := build("10"), build("14")
		out := newEmitter()
		emitFabricCounters(out, before, after)
		if v, _ := out.get("ib_symbol_errors"); v != "4" {
			t.Errorf("delta = %q, want 4", v)
		}
		if v, _ := out.get("ib_counters_saturated"); v != "0" {
			t.Errorf("ib_counters_saturated = %q, want 0 — nothing was pegged", v)
		}
	})
}

// No IB or RoCE port visible at all is ABSENT, never a clean zero. /sys/class/
// infiniband may be invisible inside the pod depending on RDMA namespace mode,
// and a threshold on any fabric counter must then fail closed.
func TestNoFabricPortIsAbsentNotClean(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "class", "net", "eth0", "carrier_down_count"), "0\n")

	out := newEmitter()
	emitFabricCounters(out, probeNIC(root), probeNIC(root))

	if st, _ := out.get("ib_counter_status"); st != "absent" {
		t.Errorf("ib_counter_status = %q, want absent", st)
	}
	for _, c := range fabricCounters {
		if v, ok := out.get(c.key); ok {
			t.Errorf("%s=%q was emitted with no fabric port visible", c.key, v)
		}
	}
}

// A counter this kernel does not expose is withheld, not zeroed — and only that
// counter.
func TestUnexposedFabricCounterIsWithheldAlone(t *testing.T) {
	root := t.TempDir()
	port := filepath.Join(root, "class", "infiniband", "mlx5_0", "ports", "1")
	mustWrite(t, filepath.Join(port, "state"), "4: ACTIVE\n")
	mustWrite(t, filepath.Join(port, "counters", "link_downed"), "0\n")
	mustWrite(t, filepath.Join(port, "counters", "symbol_error"), "2\n")
	// port_rcv_errors deliberately absent.

	out := newEmitter()
	emitFabricCounters(out, probeNIC(root), probeNIC(root))

	if _, ok := out.get("ib_symbol_errors"); !ok {
		t.Error("a readable counter was withheld because a different one was missing")
	}
	if v, ok := out.get("ib_port_rcv_errors"); ok {
		t.Errorf("ib_port_rcv_errors=%q was emitted for a counter this kernel does not expose", v)
	}
}

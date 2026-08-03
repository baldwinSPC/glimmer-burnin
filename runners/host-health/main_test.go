package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestClampWindowSeconds(t *testing.T) {
	cases := []struct{ in, want int }{
		{0, defaultWindowSeconds},  // unset
		{-5, defaultWindowSeconds}, // nonsense
		{5, 5},                     // honoured
		{600, maxWindowSeconds},    // the reconciler's default duration, capped
		{maxWindowSeconds, maxWindowSeconds},
	}
	for _, tc := range cases {
		if got := clampWindowSeconds(tc.in); got != tc.want {
			t.Errorf("clampWindowSeconds(%d) = %d, want %d", tc.in, got, tc.want)
		}
	}
}

func TestEmitterKeepsInsertionOrderOnOverwrite(t *testing.T) {
	e := newEmitter()
	e.set("a", "1")
	e.set("b", "2")
	e.set("a", "3")

	var buf bytes.Buffer
	e.writeTo(&buf)
	if got := buf.String(); got != "a=3\nb=2\n" {
		t.Errorf("writeTo = %q, want a=3 then b=2", got)
	}
}

func TestEmitterSetIntIfOmitsUnmeasured(t *testing.T) {
	e := newEmitter()
	e.setIntIf("measured", 0, true)
	e.setIntIf("unmeasured", 0, false)
	if _, ok := e.get("measured"); !ok {
		t.Error("a measured zero must be emitted")
	}
	if _, ok := e.get("unmeasured"); ok {
		t.Error("an unmeasured counter must be omitted, never zeroed")
	}
}

func TestDecide(t *testing.T) {
	t.Run("clean counters pass", func(t *testing.T) {
		e := newEmitter()
		e.set(keyXidCount, "0")
		e.set(keyNICLinkDown, "0")
		code, msg := decide(e)
		if code != exitPass || msg != "HOST_HEALTH_OK" {
			t.Errorf("decide = %d, %q; want 0, HOST_HEALTH_OK", code, msg)
		}
	})

	t.Run("a non-zero gated counter fails", func(t *testing.T) {
		e := newEmitter()
		e.set(keyXidCount, "0")
		e.set(keyECCErrors, "2")
		code, msg := decide(e)
		if code != exitFail {
			t.Errorf("code = %d, want 1", code)
		}
		if !strings.Contains(msg, "ecc_errors=2") {
			t.Errorf("message %q must name the counter that failed", msg)
		}
	})

	t.Run("evidence counters do not gate", func(t *testing.T) {
		e := newEmitter()
		e.set(keyXidCount, "0")
		// Pre-existing Xids, correctable ECC and AER noise are all real
		// evidence, but whether they condemn a node is the profile's decision.
		e.set("xid_preexisting", "17")
		e.set("ecc_corrected_aggregate", "5000")
		e.set("pcie_aer_correctable", "9")
		if code, _ := decide(e); code != exitPass {
			t.Errorf("code = %d, want 0 — evidence must not gate", code)
		}
	})

	t.Run("nothing measured is Error, not Fail and never Skip", func(t *testing.T) {
		e := newEmitter()
		e.set("xid_source", "none")
		code, msg := decide(e)
		if code != exitError {
			t.Errorf("code = %d, want %d (Error: the node is unjudged)", code, exitError)
		}
		if code == exitNotApplicable {
			t.Error("host-health applies to every node and must never report Skip")
		}
		if !strings.HasPrefix(msg, "HOST_HEALTH_ERROR") {
			t.Errorf("message = %q, want a HOST_HEALTH_ERROR marker", msg)
		}
	})
}

// e2eFixture builds a fake node: a sysfs tree, a kernel log, and an nvidia-smi.
// gpu may be nil for a node with no NVIDIA driver.
type e2eFixture struct {
	cfg      config
	kmsgPath string
}

func newE2EFixture(t *testing.T, gpu map[string]string, ethDown string) *e2eFixture {
	t.Helper()
	root := nicFixture(t, ethDown, "0")
	dev := filepath.Join(root, "bus", "pci", "devices", "0000:01:00.0")
	mustWrite(t, filepath.Join(dev, "aer_dev_correctable"), aerCorrectable)
	mustWrite(t, filepath.Join(dev, "aer_dev_fatal"), "Undefined 0\n")

	kmsg := filepath.Join(root, "kmsg")
	mustWrite(t, kmsg, "6,1,1,-;systemd[1]: Started something\n")

	f := &e2eFixture{kmsgPath: kmsg}
	smi := noNvidiaSMI
	if gpu != nil {
		smi = fakeSMI(gpu)
	}
	f.cfg = config{
		window:       time.Second,
		kmsgPath:     kmsg,
		sysfsRoot:    root,
		nvidiaSMI:    "nvidia-smi",
		probeTimeout: probeTimeoutSeconds * time.Second,
		run:          smi,
		now:          time.Now,
		sleep:        func(time.Duration) {},
	}
	return f
}

// logDuringWindow makes the injected sleep append to the kernel log, which is
// how a test simulates a fault occurring while the runner is watching.
func (f *e2eFixture) logDuringWindow(t *testing.T, record string) {
	t.Helper()
	f.cfg.sleep = func(time.Duration) { appendTo(t, f.kmsgPath, record) }
}

func TestRunHealthyNode(t *testing.T) {
	f := newE2EFixture(t, healthyGPU(), "0")
	var buf bytes.Buffer
	code := run(f.cfg, &buf)

	if code != exitPass {
		t.Fatalf("exit = %d, want 0\n%s", code, buf.String())
	}
	out := buf.String()
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if last := lines[len(lines)-1]; last != "HOST_HEALTH_OK" {
		t.Errorf("last line = %q, want HOST_HEALTH_OK", last)
	}
	for _, want := range []string{
		keyXidCount + "=0",
		keyECCErrors + "=0",
		keyRowsRemapped + "=0",
		keyPCIeReplayCount + "=0",
		keyNICLinkDown + "=0",
		"node_ready=true",
		"observation_window_s=1",
		"xid_source=kmsg",
		"nvml_status=ok",
	} {
		if !strings.Contains(out, want+"\n") {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
}

// The whole reason the sentinel exists, end to end: a healthy GB10 has no ECC
// for NVML to read, and must come out of this runner PASSING, with the two
// counters it cannot produce declared rather than omitted or zeroed.
func TestRunOnGB10DeclaresECCUnmeasurableAndStillPasses(t *testing.T) {
	f := newE2EFixture(t, sparkGPU(), "0")
	var buf bytes.Buffer
	code := run(f.cfg, &buf)

	out := buf.String()
	if code != exitPass {
		t.Fatalf("exit = %d, want 0 — a healthy Spark must not fail its own runner\n%s", code, out)
	}
	for _, want := range []string{
		keyECCErrors + "=" + unmeasurableValue,
		keyRowsRemapped + "=" + unmeasurableValue,
		"ecc_mode=unsupported",
		// Everything the part CAN report is still measured and gated.
		keyXidCount + "=0",
		keyNICLinkDown + "=0",
		"node_ready=true",
	} {
		if !strings.Contains(out, want+"\n") {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	// The sentinel is a declaration, not a reading: it must never be dressed up
	// as a clean counter.
	if strings.Contains(out, keyECCErrors+"=0") {
		t.Error("an unmeasurable ECC counter was reported as zero")
	}
}

// A failing run must still yield its evidence: every metric is printed before
// the verdict line.
func TestRunFaultyNodeStillPrintsEvidence(t *testing.T) {
	gpu := healthyGPU()
	gpu[fECCUncAgg] = "2"
	gpu[fRemapUnc] = "1"
	f := newE2EFixture(t, gpu, "4")
	f.logDuringWindow(t, "3,9,9,-;NVRM: Xid (PCI:0000:01:00): 48, DBE\n")

	var buf bytes.Buffer
	code := run(f.cfg, &buf)
	if code != exitFail {
		t.Fatalf("exit = %d, want 1\n%s", code, buf.String())
	}

	out := buf.String()
	idxMetric := strings.Index(out, keyECCErrors+"=2")
	idxVerdict := strings.Index(out, "HOST_HEALTH_FAULT")
	if idxMetric < 0 || idxVerdict < 0 {
		t.Fatalf("missing metric or verdict:\n%s", out)
	}
	if idxMetric > idxVerdict {
		t.Error("metrics must be printed before the pass/fail decision")
	}
	for _, want := range []string{keyECCErrors + "=2", keyRowsRemapped + "=1", keyXidCount + "=1", "node_ready=false"} {
		if !strings.Contains(out, want+"\n") {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(out, "xid_preexisting=0") {
		t.Error("a Xid logged during the window must not be counted as pre-existing")
	}
}

// A node nothing can be read from is UNJUDGED. That is Error (3), never Fail
// and never Skip.
func TestRunUnreadableNodeIsError(t *testing.T) {
	empty := t.TempDir()
	cfg := config{
		window:       0,
		kmsgPath:     filepath.Join(empty, "no-kmsg"),
		kernLogPaths: []string{filepath.Join(empty, "no-log")},
		sysfsRoot:    filepath.Join(empty, "no-sys"),
		nvidiaSMI:    "nvidia-smi",
		probeTimeout: probeTimeoutSeconds * time.Second,
		run:          noNvidiaSMI,
		now:          time.Now,
		sleep:        func(time.Duration) {},
	}
	var buf bytes.Buffer
	code := run(cfg, &buf)
	if code != exitError {
		t.Fatalf("exit = %d, want %d\n%s", code, exitError, buf.String())
	}
	if code == exitNotApplicable {
		t.Fatal("host-health must never skip")
	}
	out := buf.String()
	if !strings.Contains(out, "HOST_HEALTH_ERROR") {
		t.Errorf("output missing the error marker:\n%s", out)
	}
	// Even here the run reports what it looked for.
	for _, want := range []string{"xid_source=none", "nvml_status=absent", "aer_status=absent", "nic_status=absent"} {
		if !strings.Contains(out, want+"\n") {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
}

// A node with no NVIDIA driver is a normal, vendor-neutral outcome: the
// remaining probes still produce a verdict.
func TestRunNonNVIDIANode(t *testing.T) {
	f := newE2EFixture(t, nil, "0")
	var buf bytes.Buffer
	code := run(f.cfg, &buf)
	if code != exitPass {
		t.Fatalf("exit = %d, want 0\n%s", code, buf.String())
	}
	out := buf.String()
	if !strings.Contains(out, "nvml_status=absent\n") {
		t.Errorf("nvml_status must say absent:\n%s", out)
	}
	if strings.Contains(out, keyECCErrors+"=") {
		t.Error("ecc_errors must be omitted with no NVML")
	}
	if !strings.Contains(out, keyNICLinkDown+"=0\n") {
		t.Error("the vendor-neutral NIC probe must still report")
	}
	if !strings.Contains(out, "pcie_replay_source=sysfsAer\n") {
		t.Error("PCIe replay must fall back to sysfs AER when NVML is absent")
	}
}

func TestRunHonoursDurationEnv(t *testing.T) {
	t.Setenv("BURNIN_DURATION_SECONDS", "7")
	cfg := configFromEnv(time.Now())
	if cfg.window != 7*time.Second {
		t.Errorf("window = %v, want 7s", cfg.window)
	}

	t.Setenv("BURNIN_DURATION_SECONDS", "600")
	if cfg := configFromEnv(time.Now()); cfg.window != maxWindowSeconds*time.Second {
		t.Errorf("window = %v, want the %ds cap", cfg.window, maxWindowSeconds)
	}
}

func TestSleepWindowRespectsTheRunnerDeadline(t *testing.T) {
	start := time.Now()
	var slept time.Duration
	cfg := config{
		window:   maxWindowSeconds * time.Second,
		deadline: start.Add(20 * time.Second),
		now:      func() time.Time { return start },
		sleep:    func(d time.Duration) { slept = d },
	}
	cfg.sleepWindow()
	if want := (20 - closingReserveSeconds) * time.Second; slept != want {
		t.Errorf("slept %v, want %v — the closing sample must still fit", slept, want)
	}
}

func TestConfigFromEnvOverrides(t *testing.T) {
	t.Setenv("BURNIN_SYSFS_ROOT", "/fake/sys")
	t.Setenv("BURNIN_KMSG_PATH", "/fake/kmsg")
	t.Setenv("BURNIN_KERN_LOG_PATHS", "/a/one.log:/b/two.log")
	t.Setenv("BURNIN_NVIDIA_SMI", "/opt/bin/nvidia-smi")

	cfg := configFromEnv(time.Now())
	if cfg.sysfsRoot != "/fake/sys" || cfg.kmsgPath != "/fake/kmsg" || cfg.nvidiaSMI != "/opt/bin/nvidia-smi" {
		t.Errorf("overrides not honoured: %+v", cfg)
	}
	if len(cfg.kernLogPaths) != 2 || cfg.kernLogPaths[1] != "/b/two.log" {
		t.Errorf("kernLogPaths = %v", cfg.kernLogPaths)
	}
}

// noNvidiaSMI is a node with no NVIDIA driver installed.
func noNvidiaSMI(context.Context, string, []string) (string, error) {
	return "", errCommandNotFound
}

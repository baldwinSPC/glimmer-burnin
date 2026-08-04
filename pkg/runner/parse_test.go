package runner

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Verbatim stdout from ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.1.0
// running on a real NVIDIA GB10 (driver 580.82.09), captured 2026-07-28. This
// is the golden case: if parsing ever stops handling the output our own shipped
// runner actually produces, that is the regression that matters most.
const computeSmokeStdout = `gpu_name=NVIDIA GB10
compute_cap=12.1
m=1024
n=1024
k=1024
max_abs_ref=511
max_abs_error=1
max_rel_error=0.00195695
nonfinite_count=0
elapsed_ms=0.0211
tflops=101.99
FP4_GEMM_PASS
`

// Verbatim stdout from ghcr.io/baldwinspc/glimmer-burnin-clockprobe:v0.2.0 on a
// real NVIDIA GB10 (driver 580.82.09), captured 2026-08-03 with
// BURNIN_DURATION_SECONDS=20.
//
// This is the whole emitted set, which is the point: issue #65 found that 38 of
// clockprobe's metrics were unregistered in pkg/contract, and an AUDIT of the
// source missed nine of them (the per-reason sample counters below). Captured
// output cannot miss any.
const clockprobeGB10Stdout = `gpu_name=NVIDIA GB10
compute_cap=12.1
pci_bus_id=000F:01:00.0
duration_requested_s=20
warmup_s=4
sample_window_s=16
clock_floor_pct=70.00
thermal_clock_floor_pct=50.00
thermal_temp_threshold_c=80.00
driver_version=580.82.09
rated_boost_clock_mhz=3003
load_threads=98304
load_iters_per_launch=880945
elapsed_s=21.01
samples_taken=165
load_launches=300
sm_clock_mhz=2512
sustained_clock_pct=83.65
min_sm_clock_pct=82.45
max_sm_clock_pct=83.98
gpu_temp_c=61.0
mean_temp_under_load_c=57.4
temp_at_min_clock_c=56.0
power_draw_w=66.83
mean_power_w=62.63
gpu_utilization_pct=85.39
mem_utilization_pct=0.00
sustained_fma_throughput_tflops=27.474
peak_fma_throughput_tflops=27.477
throughput_consistency_pct=99.99
throttle_events=0
throttled_samples=0
throttle_reasons_mask=0
gpu_idle_samples=0
applications_clocks_setting_samples=0
sw_power_cap_samples=0
hw_slowdown_samples=0
sync_boost_samples=0
sw_thermal_slowdown_samples=0
hw_thermal_slowdown_samples=0
hw_power_brake_samples=0
display_clock_setting_samples=0
throttle_reasons=none
throttle_classification=none
pd_wedge_suspected=false
clock_floor_applied_pct=70.00
clock_floor_basis=general
nvml_unsupported=enforcedPowerLimit,defaultPowerLimit,memClock
unsupported_reads=3
CLOCKPROBE_PASS
`

// clockprobeDiagnosticNames are the metrics clockprobe emits that are
// deliberately NOT registered in pkg/contract: the per-bit decomposition of
// throttleReasonsMask.
//
// They are listed here rather than pattern-matched so that adding a tenth
// reason counter is a decision somebody makes, not something a suffix rule
// swallows. See the note beside throttleReasonsMask in pkg/contract/metrics.go
// for why they stay out: the vendor's own enum, scaled by the sample interval,
// and already summarised for acceptance by throttledSamples/throttleEvents.
//
// Staying out is not the same as being invisible. They are integers, so no gate
// on one fails closed — but verdict.ValidateThresholdsForKind does report an
// unregistered metric on a kind this project ships, so an author who gates on
// one is told to register it, which is the moment the name stops being
// incidental evidence. That interaction is asserted in pkg/verdict's
// TestValidateThresholdsForKind_FlagsAnUnregisteredMetricOnABuiltInKind, which
// uses these very names.
var clockprobeDiagnosticNames = map[string]bool{
	"gpuIdleSamples":                   true,
	"applicationsClocksSettingSamples": true,
	"swPowerCapSamples":                true,
	"hwSlowdownSamples":                true,
	"syncBoostSamples":                 true,
	"swThermalSlowdownSamples":         true,
	"hwThermalSlowdownSamples":         true,
	"hwPowerBrakeSamples":              true,
	"displayClockSettingSamples":       true,
}

// TestParse_RealClockProbeOutputIsRegistered is issue #65's fix, asserted
// against what the runner actually prints rather than against a reading of its
// source.
//
// The registry is an open world on purpose, so being unregistered is not a bug
// in general. It is a bug for a FIRST-PARTY runner, and specifically for a
// metric whose value is a label: the grammar passes, UnitOf answers UnitNone,
// and SafeToThresholdOn answers true for an unregistered name — so a gate on
// "pdWedgeSuspected" passed every authoring-time check and then failed closed on
// every node forever, reading as a hardware verdict on healthy hardware.
func TestParse_RealClockProbeOutputIsRegistered(t *testing.T) {
	got := Parse("clockprobe", clockprobeGB10Stdout, 0)

	if got.Verdict != VerdictPass || got.Message != "CLOCKPROBE_PASS" {
		t.Fatalf("Verdict = %q, Message = %q", got.Verdict, got.Message)
	}
	if len(got.InvalidNames) != 0 {
		t.Errorf("our own runner emitted names the contract rejects: %v", got.InvalidNames)
	}

	for name, value := range got.Metrics {
		if clockprobeDiagnosticNames[name] {
			// Still has to obey the grammar and stay dimensionless: these are
			// counts, and a name that picked up a unit suffix would be charted
			// as a physical quantity.
			if u := contract.UnitOf(name); u != contract.UnitNone {
				t.Errorf("diagnostic counter %q declares unit %q; it is a count", name, u)
			}
			continue
		}
		m, ok := contract.Lookup(name)
		if !ok {
			t.Errorf("clockprobe emits %q=%q, which pkg/contract does not register and "+
				"clockprobeDiagnosticNames does not except. Register it, or add it there with a "+
				"reason — an unregistered first-party name is assumed safe to threshold on",
				name, value)
			continue
		}
		// The unit the name declares and the unit the registry claims must
		// agree, or the registry is lying about data consumers will chart.
		if u := contract.UnitOf(name); u != m.Unit {
			t.Errorf("metric %q declares unit %q by name but is registered as %q", name, u, m.Unit)
		}

		// THE check this issue is about. If the value is not a number, the
		// registry must refuse a threshold on it — a gate would be evaluated as
		// a float64 and fail closed on every node forever.
		if _, numeric := got.Numeric(name); !numeric {
			if contract.SafeToThresholdOn(name) {
				t.Errorf("clockprobe emits %q=%q, which is not numeric, yet the registry says it is "+
					"safe to threshold on. A gate on it fails closed on every node forever and reads "+
					"as a hardware verdict", name, value)
			}
		}
	}

	// The label-valued set, asserted by name as well as by shape: a future
	// build that happened to emit a numeric-looking value must not quietly make
	// one of these gateable again.
	for _, name := range []string{
		"gpuName", "computeCap", "pciBusId", "driverVersion",
		"throttleReasons", "throttleClassification", "pdWedgeSuspected",
		"clockFloorBasis", "nvmlUnsupported",
	} {
		if _, ok := got.Metrics[name]; !ok {
			t.Errorf("the captured GB10 run no longer contains %q; this fixture has drifted", name)
		}
		if contract.SafeToThresholdOn(name) {
			t.Errorf("%q is label-valued and must not be safe to threshold on", name)
		}
	}

	// A GB10 finding worth pinning, because it decides how a threshold on the
	// power metrics behaves in the field: this part's driver refuses BOTH power
	// limit reads, so enforcedPowerLimitW, defaultPowerLimitW and the ratio
	// derived from them are OMITTED — not zero. Omission fails a threshold
	// closed, which is correct and is exactly why pkg/contract's description of
	// powerLimitRatioPct says a gate on it must be scoped to parts that answer.
	if got.Metrics["nvmlUnsupported"] != "enforcedPowerLimit,defaultPowerLimit,memClock" {
		t.Errorf("nvmlUnsupported = %q; the fixture's GB10 finding has changed", got.Metrics["nvmlUnsupported"])
	}
	for _, absent := range []string{"powerLimitRatioPct", "enforcedPowerLimitW", "defaultPowerLimitW", "memClockMHz"} {
		if v, ok := got.Metrics[absent]; ok {
			t.Errorf("%q = %q was reported, but this part's driver refused the read; a runner must "+
				"omit what it could not measure rather than print a number nobody took", absent, v)
		}
		// And it is NOT the unmeasurable sentinel either: "n/a" is a positive
		// declaration that the hardware has nothing to report, and a driver
		// refusing a read is not that claim.
		if got.IsUnmeasurable(absent) {
			t.Errorf("%q was declared unmeasurable; a refused read is an omission, not a declaration", absent)
		}
	}
}

// Verbatim stdout from compute-smoke taking its SKIP path, captured 2026-08-03
// on the same real GB10 (driver 580.82.09, CUDA 13.0.1, CUTLASS v4.6.1).
//
// Exit 2 is the contract's load-bearing distinction — a node that cannot run a
// test has NOT failed it — and until this capture it had never once executed
// (issue #9). It fires only on a part that is not compute capability 12.0/12.1,
// and this project has no such accelerator; dcgm-diag was expected to be the
// first real-silicon exerciser on the theory that DCGM does not support GB10,
// and it does, so that route is gone.
//
// So this is the real binary, the real toolchain and the real GPU, built with
// the compile-time capability injection described in fp4_smoke.cu. The
// `forced_compute_cap=9.0` line is left in ON PURPOSE rather than tidied away:
// it is exactly what such a build prints, and it is what makes any verdict from
// one self-identifying. A fixture that hid it would be claiming an H100
// produced this.
const computeSmokeSkipStdout = `built_cuda_arch=sm_121a
forced_compute_cap=9.0
gpu_name=NVIDIA GB10
compute_cap=9.0
FP4_GEMM_SKIP: NVFP4 block-scaled GEMM requires compute capability 12.0/12.1, and this part reports 9.0 — the test does not apply to this hardware and the part is NOT failed; run a kind whose runner covers this architecture
`

// The skip half of the golden case. Skip must never collapse into Fail: exit 1
// is a hardware verdict the operator never retries, so misreading this output
// would permanently indict every node that merely cannot take the test.
func TestParse_RealComputeSmokeSkipOutput(t *testing.T) {
	got := Parse("compute-smoke", computeSmokeSkipStdout, 2)

	if got.Verdict != VerdictSkip {
		t.Fatalf("Verdict = %q, want Skip — a part that cannot take this test has not failed it", got.Verdict)
	}
	if got.Verdict == VerdictFail {
		t.Fatal("a skip was read as a hardware failure")
	}
	if !strings.HasPrefix(got.Message, "FP4_GEMM_SKIP:") {
		t.Errorf("Message = %q, want the FP4_GEMM_SKIP sentinel", got.Message)
	}
	// The reason has to survive into the message, or an operator sees a bare
	// "Skipped" with nothing saying which hardware property caused it.
	if !strings.Contains(got.Message, "compute capability 12.0/12.1") {
		t.Errorf("Message = %q, want it to carry the runner's reason", got.Message)
	}
	if len(got.InvalidNames) != 0 {
		t.Errorf("our own runner emitted names the contract rejects: %v", got.InvalidNames)
	}

	// A skip still reports what it learned before deciding. That matters for the
	// Error/Skip distinction downstream: a runner that reported nothing at all
	// measured nothing, which is a different finding (see ReportedNothing).
	if got.ReportedNothing() {
		t.Error("the skip reported no key=value line at all; it printed identity before deciding")
	}
	for name, want := range map[string]string{
		"builtCudaArch":    "sm_121a",
		"gpuName":          "NVIDIA GB10",
		"computeCap":       "9.0",
		"forcedComputeCap": "9.0",
	} {
		if got.Metrics[name] != want {
			t.Errorf("Metrics[%q] = %q, want %q", name, got.Metrics[name], want)
		}
	}
}

func TestParse_RealComputeSmokeOutput(t *testing.T) {
	got := Parse("compute-smoke", computeSmokeStdout, 0)

	if got.Verdict != VerdictPass {
		t.Errorf("Verdict = %q, want Pass", got.Verdict)
	}
	// The pass marker is not a metric, but it is the most useful trailing line.
	if got.Message != "FP4_GEMM_PASS" {
		t.Errorf("Message = %q, want FP4_GEMM_PASS", got.Message)
	}
	if len(got.InvalidNames) != 0 {
		t.Errorf("our own runner emitted names the contract rejects: %v", got.InvalidNames)
	}

	want := map[string]string{
		"gpuName":     "NVIDIA GB10",
		"computeCap":  "12.1",
		"m":           "1024",
		"n":           "1024",
		"k":           "1024",
		"maxAbsRef":   "511",
		"maxAbsError": "1",
		"maxRelError": "0.00195695",
		// Registered, dimensionless.
		"nonfiniteCount": "0",
		// Generic normalisation lands on a valid unit suffix.
		"elapsedMs": "0.0211",
		// The alias case: "tflops" is a bare unit naming no measurand.
		"throughputTflops": "101.99",
	}
	if len(got.Metrics) != len(want) {
		t.Errorf("parsed %d metrics, want %d: %v", len(got.Metrics), len(want), got.Metrics)
	}
	for name, wantVal := range want {
		if gotVal, ok := got.Metrics[name]; !ok {
			t.Errorf("missing canonical metric %q", name)
		} else if gotVal != wantVal {
			t.Errorf("metric %q = %q, want %q", name, gotVal, wantVal)
		}
	}
}

// Every canonical name we produce must survive the contract, or the envelope
// would be rejected at delivery time with the run already finished.
func TestParse_ProducesContractValidNames(t *testing.T) {
	got := Parse("compute-smoke", computeSmokeStdout, 0)
	for name := range got.Metrics {
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("parser produced a name the contract rejects: %v", err)
		}
	}
}

// Exit codes are the runner contract. Skip must never collapse into Fail.
func TestVerdictFor(t *testing.T) {
	cases := map[int]Verdict{
		0: VerdictPass,
		1: VerdictFail,
		2: VerdictSkip,
		// 3 is the code every runner in this repo uses to say "I could not
		// measure, so this part is unjudged". It is not special-cased here —
		// it falls through the same default as any other code — but it is
		// pinned because it is the one a runner author actually writes.
		3: VerdictError,
		// Anything else means the runner malfunctioned; the hardware is
		// unjudged, which is not the same as failed.
		125: VerdictError,
		137: VerdictError,
		-1:  VerdictError,
	}
	for code, want := range cases {
		if got := VerdictFor(code); got != want {
			t.Errorf("VerdictFor(%d) = %q, want %q", code, got, want)
		}
	}
}

// TestParse_ComputeSmokeExitContract pins the exit semantics of the oldest and
// most-trusted runner in the suite.
//
// compute-smoke up to and including the published v0.1.0 image reported "no
// usable CUDA device", "this image has no cubin for this part" and every CUDA
// runtime error as exit 1 — Fail. That is a permanent hardware verdict against a
// node whose tensor cores were never exercised, and because a Fail is never
// retried it was recorded with the run's retry budget entirely unspent. Those
// paths are exit 3 now; exit 1 is reserved for the three things the runner
// genuinely measures (NaN/Inf, an all-zero device output, tolerance exceeded).
//
// The stdout below is synthetic — it is what the fixed runner is written to
// print, not a capture from hardware. computeSmokeStdout above is the real
// capture and is deliberately left alone.
func TestParse_ComputeSmokeExitContract(t *testing.T) {
	cases := []struct {
		name     string
		stdout   string
		exitCode int
		want     Verdict
	}{{
		name:     "no GPU visible is Error, not a verdict about the node",
		stdout:   "built_cuda_arch=sm_121a\nFP4_GEMM_ERROR: no usable CUDA device (cudaGetDevice): CUDA driver version is insufficient\n",
		exitCode: 3,
		want:     VerdictError,
	}, {
		// The reconciliation case: a CC 12.0 part clears the 12.0/12.1 scope
		// gate, then finds the sm_121a binary has no cubin for it. That is a
		// statement about the image that was pinned, never about the silicon.
		name:     "wrong-arch image is Error, not Fail",
		stdout:   "built_cuda_arch=sm_121a\ngpu_name=NVIDIA Blackwell\ncompute_cap=12.0\nFP4_GEMM_ERROR: warmup gemm.run failed: no kernel image is available for execution on the device — this image carries no cubin for the part it landed on (built for sm_121a)\n",
		exitCode: 3,
		want:     VerdictError,
	}, {
		name:     "out-of-scope hardware still Skips",
		stdout:   "built_cuda_arch=sm_121a\ngpu_name=NVIDIA A100\ncompute_cap=8.0\nFP4_GEMM_SKIP: NVFP4 block-scaled GEMM requires compute capability 12.0/12.1\n",
		exitCode: 2,
		want:     VerdictSkip,
	}, {
		name:     "a measured numerical mismatch is still Fail",
		stdout:   "built_cuda_arch=sm_121a\nmax_rel_error=0.446\nnonfinite_count=0\nFP4_GEMM_FAIL: numerical mismatch exceeds tolerance\n",
		exitCode: 1,
		want:     VerdictFail,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Parse("compute-smoke", tc.stdout, tc.exitCode)
			if got.Verdict != tc.want {
				t.Errorf("Verdict = %q, want %q", got.Verdict, tc.want)
			}
			// Identity is printed before every gate, so even a run that never
			// launched a kernel records which image met which part. An Error
			// with no evidence is untriageable.
			if got.Metrics["builtCudaArch"] != "sm_121a" {
				t.Errorf("builtCudaArch = %q, want the arch to survive on every path: %v",
					got.Metrics["builtCudaArch"], got.Metrics)
			}
			if len(got.InvalidNames) != 0 {
				t.Errorf("our own runner emitted names the contract rejects: %v", got.InvalidNames)
			}
			// The marker line is not a metric, and it is the most useful thing
			// a human reads off a non-passing run.
			if !strings.HasPrefix(got.Message, "FP4_GEMM_") {
				t.Errorf("Message = %q, want the trailing FP4_GEMM_* marker", got.Message)
			}
		})
	}
}

func TestSnakeToLowerCamel(t *testing.T) {
	cases := map[string]string{
		"max_rel_error":   "maxRelError",
		"nonfinite_count": "nonfiniteCount",
		"elapsed_ms":      "elapsedMs",
		"gpu_name":        "gpuName",
		"bus-bandwidth":   "busBandwidth",
		// Already canonical: pass through so a runner may emit either spelling.
		"busBandwidthGBs": "busBandwidthGBs",
		"m":               "m",
	}
	for in, want := range cases {
		if got := snakeToLowerCamel(in); got != want {
			t.Errorf("snakeToLowerCamel(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestParse_IgnoresNonMetricLines(t *testing.T) {
	got := Parse("compute-smoke", `
== starting compute smoke ==
warming up, please wait
bandwidthGbps=180
this line = has spaces in the key
done
`, 0)

	if len(got.Metrics) != 1 || got.Metrics["bandwidthGbps"] != "180" {
		t.Errorf("metrics = %v, want only bandwidthGbps=180", got.Metrics)
	}
	if got.Message != "done" {
		t.Errorf("Message = %q, want the last non-metric line", got.Message)
	}
}

// A value may legitimately contain "=".
func TestParse_SplitsOnFirstEquals(t *testing.T) {
	got := Parse("custom", "note=a=b=c", 0)
	if got.Metrics["note"] != "a=b=c" {
		t.Errorf("note = %q, want a=b=c", got.Metrics["note"])
	}
}

// Runners may report progressively; the settled value is the last one.
func TestParse_LastValueWins(t *testing.T) {
	got := Parse("nccl", "busBandwidthGBs=10\nbusBandwidthGBs=23.4", 0)
	if got.Metrics["busBandwidthGBs"] != "23.4" {
		t.Errorf("busBandwidthGBs = %q, want the final 23.4", got.Metrics["busBandwidthGBs"])
	}
}

// An unusable name must be surfaced, not silently dropped.
func TestParse_ReportsInvalidNames(t *testing.T) {
	got := Parse("custom", "9bad=1\ngoodOne=2", 0)
	if len(got.InvalidNames) != 1 || got.InvalidNames[0] != "9bad" {
		t.Errorf("InvalidNames = %v, want [9bad]", got.InvalidNames)
	}
	if _, ok := got.Metrics["9bad"]; ok {
		t.Error("an invalid name was still stored as a metric")
	}
	if got.Metrics["goodOne"] != "2" {
		t.Error("a valid metric was lost alongside an invalid one")
	}
}

// A custom kind gets the generic scan but no kind-specific mapping: only the
// kind's author knows what its keys mean.
func TestParse_CustomKindGetsNoAliasMapping(t *testing.T) {
	got := Parse("custom", "tflops=99", 0)
	if _, mapped := got.Metrics["throughputTflops"]; mapped {
		t.Error("applied compute-smoke's alias table to a custom kind")
	}
	if got.Metrics["tflops"] != "99" {
		t.Errorf("custom kind lost its key=value metric: %v", got.Metrics)
	}
}

// Per-kind stdout fixtures, in the shape each kind's runner wrapper prints:
// the underlying tool's own vocabulary, normalised to key=value lines. They
// exist to pin the mapping from a tool's words to ours, which is the part a
// refactor can silently get wrong — a dropped alias does not fail to parse, it
// parses to a name no threshold references, and the test then fails closed on
// a healthy node.
func TestParse_PerKindFixtures(t *testing.T) {
	cases := []struct {
		kind        string
		stdout      string
		exitCode    int
		wantVerdict Verdict
		wantMessage string
		wantMetrics map[string]string
	}{
		{
			kind: "dcgm-diag",
			stdout: `dcgm_version=3.3.9
driver_version=580.82.09
diag_level=3
tests_run=14
tests_failed=0
xid_errors=0
ecc_sbe_total=0
ecc_dbe_total=0
rows_remapped=0
pcie_replay_count=0
elapsed_s=412.7
DCGM_DIAG_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "DCGM_DIAG_PASS",
			wantMetrics: map[string]string{
				"dcgmVersion":   "3.3.9",
				"driverVersion": "580.82.09",
				"diagLevel":     "3",
				"testsRun":      "14",
				// DCGM's "tests_failed" is a subtest count, not a verdict.
				"diagTestsFailed": "0",
				"xidEvents":       "0",
				// Correctable and uncorrectable ECC stay two numbers: a few
				// SBEs is a working part, one DBE is a failing one.
				"eccSbeTotal":      "0",
				"eccDbeTotal":      "0",
				"remappedRows":     "0",
				"pcieReplayErrors": "0",
				"elapsedS":         "412.7",
			},
		},
		{
			kind: "memory-bw",
			stdout: `gpu_name=NVIDIA GB10
transfer_size_bytes=268435456
h2d_bandwidth_gbs=24.83
d2h_bandwidth_gbs=26.11
d2d_bandwidth_gbs=612.40
memory_bandwidth_gbs=598.72
elapsed_s=18.4
MEMORY_BW_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "MEMORY_BW_PASS",
			wantMetrics: map[string]string{
				"gpuName":                    "NVIDIA GB10",
				"transferSizeBytes":          "268435456",
				"hostToDeviceBandwidthGBs":   "24.83",
				"deviceToHostBandwidthGBs":   "26.11",
				"deviceToDeviceBandwidthGBs": "612.40",
				"memoryBandwidthGBs":         "598.72",
				"elapsedS":                   "18.4",
			},
		},
		{
			kind: "host-health",
			stdout: `node_ready=true
gpu_count=1
driver_version=580.82.09
xid_count=0
rows_remapped=0
pcie_replay_count=0
nic_link_down=0
ecc_errors=0
gpu_temp_c=41
power_draw_w=38.2
HOST_HEALTH_OK
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "HOST_HEALTH_OK",
			wantMetrics: map[string]string{
				"nodeReady":         "true",
				"gpuCount":          "1",
				"driverVersion":     "580.82.09",
				"xidEvents":         "0",
				"remappedRows":      "0",
				"pcieReplayErrors":  "0",
				"nicLinkDownEvents": "0",
				"eccErrors":         "0",
				"gpuTempC":          "41",
				"powerDrawW":        "38.2",
			},
		},
		{
			// A real failure: the part held its temperature limit by dropping
			// clocks. Metrics must survive a non-zero exit — the evidence for
			// a Fail is exactly what an operator needs afterwards.
			kind: "thermal-soak",
			stdout: `soak_seconds=3600
peak_temp_c=91
peak_power_w=142.5
throttle_count=3
sustained_clock_pct=78.4
iterations_completed=8640
THERMAL_THROTTLE_DETECTED
`,
			exitCode:    1,
			wantVerdict: VerdictFail,
			wantMessage: "THERMAL_THROTTLE_DETECTED",
			wantMetrics: map[string]string{
				"elapsedS":            "3600",
				"gpuTempC":            "91",
				"powerDrawW":          "142.5",
				"throttleEvents":      "3",
				"sustainedClockPct":   "78.4",
				"iterationsCompleted": "8640",
			},
		},
		{
			kind: "gpu-burn",
			stdout: `gpu_name=NVIDIA GB10
iterations=420
errors=0
tflops=48.6
gpu_temp_c=86
power_draw_w=139.8
ecc_errors=0
elapsed_s=900.0
GPU_BURN_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "GPU_BURN_PASS",
			wantMetrics: map[string]string{
				"gpuName":             "NVIDIA GB10",
				"iterationsCompleted": "420",
				// gpu_burn's "errors" are wrong answers from hardware that
				// reported success — not ECC events, which are counted apart.
				"miscompares": "0",
				// Sustained across a 15-minute burn, so NOT throughputTflops.
				"sustainedThroughputTflops": "48.6",
				"gpuTempC":                  "86",
				"powerDrawW":                "139.8",
				"eccErrors":                 "0",
				"elapsedS":                  "900.0",
			},
		},
		{
			kind: "nccl",
			stdout: `ranks=2
size_bytes=1073741824
algbw=23.41
busbw=46.82
avg_time_us=45872.1
wrong_count=0
elapsed_s=62.3
NCCL_ALLREDUCE_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "NCCL_ALLREDUCE_PASS",
			wantMetrics: map[string]string{
				"ranks":           "2",
				"sizeBytes":       "1073741824",
				"algBandwidthGBs": "23.41",
				"busBandwidthGBs": "46.82",
				// Deliberately NOT latencyUs: a collective's completion time
				// is not a round trip, and registered names must not be
				// stretched to cover a different quantity. Well-formed and
				// unregistered is the honest outcome.
				"avgTimeUs":   "45872.1",
				"miscompares": "0",
				"elapsedS":    "62.3",
			},
		},
		{
			kind: "ib-write-bw",
			stdout: `device=mlx5_0
message_size_bytes=65536
bw_peak=192.31
bw_average=186.44
elapsed_s=9.8
IB_WRITE_BW_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "IB_WRITE_BW_PASS",
			wantMetrics: map[string]string{
				"device":            "mlx5_0",
				"messageSizeBytes":  "65536",
				"peakBandwidthGbps": "192.31",
				"bandwidthGbps":     "186.44",
				"elapsedS":          "9.8",
			},
		},
		{
			kind: "gpudirect-rdma",
			stdout: `gpudirect_supported=true
device=mlx5_0
bw_average=178.92
t_avg_usec=3.71
elapsed_s=11.2
GPUDIRECT_RDMA_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "GPUDIRECT_RDMA_PASS",
			wantMetrics: map[string]string{
				"gpudirectSupported": "true",
				"device":             "mlx5_0",
				// Same link measurand as ib-write-bw, so the same canonical
				// name: a separate one would split a fleet's history in two.
				"bandwidthGbps": "178.92",
				"latencyUs":     "3.71",
				"elapsedS":      "11.2",
			},
		},
		{
			kind: "clockprobe",
			stdout: `gpu_name=NVIDIA GB10
rated_boost_clock_mhz=1980
sm_clock_mhz=1837
mem_clock_mhz=4266
sustained_clock_pct=92.8
throttle_events=0
elapsed_s=30.0
CLOCKPROBE_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "CLOCKPROBE_PASS",
			wantMetrics: map[string]string{
				"gpuName":            "NVIDIA GB10",
				"ratedBoostClockMHz": "1980",
				"smClockMHz":         "1837",
				"memClockMHz":        "4266",
				"sustainedClockPct":  "92.8",
				"throttleEvents":     "0",
				"elapsedS":           "30.0",
			},
		},
		{
			kind: "memory-stress",
			stdout: `tool=stressapptest
duration_requested_s=600
hardware_incidents=0
sdc_count=0
miscompares=0
read_bandwidth_mbs=18342.6
write_bandwidth_mbs=17110.2
elapsed_s=600.4
STRESSAPPTEST_PASS
`,
			exitCode:    0,
			wantVerdict: VerdictPass,
			wantMessage: "STRESSAPPTEST_PASS",
			wantMetrics: map[string]string{
				"tool":               "stressapptest",
				"durationRequestedS": "600",
				"memoryErrors":       "0",
				"sdcDetections":      "0",
				"miscompares":        "0",
				"readBandwidthMBs":   "18342.6",
				"writeBandwidthMBs":  "17110.2",
				"elapsedS":           "600.4",
			},
		},
	}

	for _, c := range cases {
		t.Run(c.kind, func(t *testing.T) {
			got := Parse(c.kind, c.stdout, c.exitCode)

			if got.Verdict != c.wantVerdict {
				t.Errorf("Verdict = %q, want %q", got.Verdict, c.wantVerdict)
			}
			if got.Message != c.wantMessage {
				t.Errorf("Message = %q, want %q", got.Message, c.wantMessage)
			}
			if len(got.InvalidNames) != 0 {
				t.Errorf("a shipped runner's output produced names the contract rejects: %v", got.InvalidNames)
			}
			for name, want := range c.wantMetrics {
				gotVal, ok := got.Metrics[name]
				if !ok {
					t.Errorf("missing canonical metric %q; got %v", name, got.Metrics)
					continue
				}
				if gotVal != want {
					t.Errorf("metric %q = %q, want %q", name, gotVal, want)
				}
			}
			for name := range got.Metrics {
				if _, expected := c.wantMetrics[name]; !expected {
					t.Errorf("unexpected canonical metric %q=%q; every name a runner produces is stored and charted, so none may appear by accident", name, got.Metrics[name])
				}
			}
		})
	}
}

// gpudirect-rdma on a part with unified host-device memory has not failed; the
// test does not apply. Skip must reach the caller as Skip, with the runner's
// explanation intact — this is the false negative that made healthy hardware
// look broken.
func TestParse_GPUDirectSkipIsNotAFailure(t *testing.T) {
	got := Parse("gpudirect-rdma", `gpudirect_supported=false
reason=unified host-device memory; the discrete-GPU RDMA path does not apply
GPUDIRECT_RDMA_SKIP
`, 2)

	if got.Verdict != VerdictSkip {
		t.Fatalf("Verdict = %q, want Skip", got.Verdict)
	}
	if got.Metrics["gpudirectSupported"] != "false" {
		t.Errorf("gpudirectSupported = %q, want false", got.Metrics["gpudirectSupported"])
	}
	if got.Metrics["reason"] == "" {
		t.Error("the runner's explanation for skipping was dropped")
	}
	if got.Message != "GPUDIRECT_RDMA_SKIP" {
		t.Errorf("Message = %q, want GPUDIRECT_RDMA_SKIP", got.Message)
	}
}

// A runner that malfunctions leaves the hardware unjudged. Whatever it managed
// to print is still evidence, but the verdict must not read as a failure.
func TestParse_RunnerMalfunctionIsErrorNotFail(t *testing.T) {
	got := Parse("dcgm-diag", "dcgm_version=3.3.9\ncould not open /dev/nvidiactl\n", 125)
	if got.Verdict != VerdictError {
		t.Errorf("Verdict = %q, want Error; a broken runner is not a broken GPU", got.Verdict)
	}
	if got.Metrics["dcgmVersion"] != "3.3.9" {
		t.Error("evidence printed before the malfunction was discarded")
	}
}

// Every value in the alias tables is a name this project mints, so it must be a
// name pkg/contract knows. An alias to an unregistered name would deliver a
// metric no threshold, chart or consumer is expecting.
func TestAliasTargetsAreRegistered(t *testing.T) {
	for kind, table := range aliases {
		for rawKey, canonical := range table {
			if err := contract.ValidateMetricName(canonical); err != nil {
				t.Errorf("%s: alias %q maps to a name the contract rejects: %v", kind, rawKey, err)
			}
			if _, ok := contract.Lookup(canonical); !ok {
				t.Errorf("%s: alias %q maps to unregistered name %q; register it in pkg/contract or leave the key to generic normalisation", kind, rawKey, canonical)
			}
		}
	}
}

// Parsing is last-occurrence-wins, so two keys of one kind mapping to the same
// canonical name would silently discard one of two real measurements.
func TestAliasTargetsAreUniqueWithinAKind(t *testing.T) {
	for kind, table := range aliases {
		seen := map[string]string{}
		for rawKey, canonical := range table {
			if prev, dup := seen[canonical]; dup {
				t.Errorf("%s: %q and %q both map to %q; last-occurrence-wins would drop one", kind, prev, rawKey, canonical)
			}
			seen[canonical] = rawKey
		}
	}
}

// An alias that generic normalisation would have produced anyway is dead
// weight, and dead weight is what stops a reader trusting that the entries
// which remain are load-bearing.
func TestAliasEntriesAreNecessary(t *testing.T) {
	for kind, table := range aliases {
		for rawKey, canonical := range table {
			if snakeToLowerCamel(rawKey) == canonical {
				t.Errorf("%s: alias %q → %q is redundant; generic normalisation already produces it", kind, rawKey, canonical)
			}
		}
	}
}

// The quiet failure the alias tables exist for. These raw keys carry a unit in
// the tool's own casing; normalised generically they produce names UnitOf()
// reads as dimensionless, so a bandwidth or a clock would be stored as a bare
// number that charts happily and compares against nothing.
func TestParse_UnitCasingTrapsAreAliased(t *testing.T) {
	cases := []struct{ kind, rawKey string }{
		{"memory-bw", "h2d_bandwidth_gbs"},
		{"memory-bw", "d2h_bandwidth_gbs"},
		{"memory-bw", "d2d_bandwidth_gbs"},
		{"memory-bw", "memory_bandwidth_gbs"},
		{"memory-stress", "read_bandwidth_mbs"},
		{"memory-stress", "write_bandwidth_mbs"},
		{"clockprobe", "sm_clock_mhz"},
		{"clockprobe", "mem_clock_mhz"},
		{"clockprobe", "rated_boost_clock_mhz"},
		{"thermal-soak", "soak_seconds"},
		{"gpudirect-rdma", "t_avg_usec"},
	}
	for _, c := range cases {
		// The premise: generic normalisation really does lose the unit here.
		// If this ever stops being true the alias may be revisited, but the
		// test must not quietly pass on a false premise.
		if u := contract.UnitOf(snakeToLowerCamel(c.rawKey)); u != contract.UnitNone {
			t.Errorf("%s: %q normalises generically to unit %q; this case no longer demonstrates the trap", c.kind, c.rawKey, u)
			continue
		}
		got := Parse(c.kind, c.rawKey+"=1", 0)
		for name := range got.Metrics {
			if contract.UnitOf(name) == contract.UnitNone {
				t.Errorf("%s: %q parsed to %q, which declares no unit; a dimensional measurement was recorded as a bare number", c.kind, c.rawKey, name)
			}
		}
	}
}

// Alias tables are per-kind because the same word means different things to
// different tools. compute-smoke's "tflops" is one unwarmed launch; gpu-burn's
// is a sustained average. Sharing a canonical name would make one of them
// wrong, and would hand a liveness signal to a profile author as if it were
// gateable.
func TestParse_SameKeyDiffersByKind(t *testing.T) {
	smoke := Parse("compute-smoke", "tflops=101.99", 0)
	if smoke.Metrics["throughputTflops"] != "101.99" {
		t.Errorf("compute-smoke tflops = %v, want throughputTflops=101.99", smoke.Metrics)
	}
	if contract.SafeToThresholdOn("throughputTflops") {
		t.Error("compute-smoke's throughput is a liveness signal and must not be thresholdable")
	}

	burn := Parse("gpu-burn", "tflops=48.6", 0)
	if burn.Metrics["sustainedThroughputTflops"] != "48.6" {
		t.Errorf("gpu-burn tflops = %v, want sustainedThroughputTflops=48.6", burn.Metrics)
	}
	if _, collided := burn.Metrics["throughputTflops"]; collided {
		t.Error("gpu-burn's sustained figure took compute-smoke's name, which would mark a real benchmark unthresholdable")
	}
	if !contract.SafeToThresholdOn("sustainedThroughputTflops") {
		t.Error("gpu-burn's sustained throughput is a benchmark and must stay thresholdable")
	}
}

// Every canonical name any shipped kind can produce must survive the contract:
// a name rejected at delivery time fails after the run is already finished and
// the hardware is no longer under test.
func TestParse_AllKindsProduceContractValidNames(t *testing.T) {
	for kind, table := range aliases {
		var b strings.Builder
		for rawKey := range table {
			b.WriteString(rawKey)
			b.WriteString("=1\n")
		}
		got := Parse(kind, b.String(), 0)
		if len(got.InvalidNames) != 0 {
			t.Errorf("%s: alias table produces contract-invalid names: %v", kind, got.InvalidNames)
		}
		if len(got.Metrics) != len(table) {
			t.Errorf("%s: parsed %d metrics from %d alias keys: %v", kind, len(got.Metrics), len(table), got.Metrics)
		}
	}
}

// ─── The unmeasurable sentinel ────────────────────────────────────────────────

// The GB10 case, in one line of stdout: the runner asked, the hardware has no
// ECC to report, and it says so. The declaration must not become a metric — a
// stored "n/a" would be a value a threshold could try to compare against, and a
// stored 0 would be a lie.
func TestParse_UnmeasurableSentinelIsDeclaredNotMeasured(t *testing.T) {
	got := Parse("host-health", "xid_count=0\necc_errors=n/a\nrows_remapped=n/a\nHOST_HEALTH_OK\n", 0)

	for _, name := range []string{"eccErrors", "remappedRows"} {
		if !got.IsUnmeasurable(name) {
			t.Errorf("%s was not recorded as unmeasurable: %v", name, got.Unmeasurable)
		}
		if v, stored := got.Metrics[name]; stored {
			t.Errorf("%s=%q leaked into Metrics; the sentinel denies a measurement, it is not one", name, v)
		}
		if _, ok := got.Numeric(name); ok {
			t.Errorf("Numeric(%s) invented a number for an unmeasurable metric", name)
		}
	}
	if got.Metrics["xidEvents"] != "0" {
		t.Errorf("a real measurement alongside the sentinel was lost: %v", got.Metrics)
	}
	if len(got.InvalidNames) != 0 {
		t.Errorf("the sentinel must not affect name validation: %v", got.InvalidNames)
	}
	if got.Message != "HOST_HEALTH_OK" {
		t.Errorf("Message = %q, want HOST_HEALTH_OK", got.Message)
	}
}

// Runners are written in whatever language suits the probe, and nvidia-smi
// itself prints "N/A". The spelling of the sentinel must not decide a verdict.
func TestParse_UnmeasurableSentinelIsCaseInsensitive(t *testing.T) {
	for _, spelling := range []string{"n/a", "N/A", "N/a", " n/a "} {
		got := Parse("host-health", "ecc_errors="+spelling, 0)
		if !got.IsUnmeasurable("eccErrors") {
			t.Errorf("%q was not recognised as the sentinel", spelling)
		}
	}
}

// Everything else is an ordinary value. A gate must never relax on a guess at
// what a runner meant.
func TestParse_OnlyTheExactSentinelDeclaresUnmeasurable(t *testing.T) {
	for _, value := range []string{"unknown", "none", "null", "NA", "-", "0"} {
		got := Parse("host-health", "ecc_errors="+value, 0)
		if got.IsUnmeasurable("eccErrors") {
			t.Errorf("%q was treated as a declaration of unmeasurability", value)
		}
		if got.Metrics["eccErrors"] != value {
			t.Errorf("eccErrors = %q, want the value kept as reported", got.Metrics["eccErrors"])
		}
	}
}

// Parsing is last-occurrence-wins, and that has to hold across the two maps or
// they could both claim the same name: a runner that measures after declaring
// has measured it, and one that declares after reporting has retracted it.
func TestParse_UnmeasurableIsLastOccurrenceWins(t *testing.T) {
	declaredThenMeasured := Parse("host-health", "ecc_errors=n/a\necc_errors=2", 0)
	if declaredThenMeasured.IsUnmeasurable("eccErrors") {
		t.Error("a later real reading did not retract the sentinel")
	}
	if declaredThenMeasured.Metrics["eccErrors"] != "2" {
		t.Errorf("eccErrors = %q, want 2", declaredThenMeasured.Metrics["eccErrors"])
	}

	measuredThenDeclared := Parse("host-health", "ecc_errors=2\necc_errors=n/a", 0)
	if !measuredThenDeclared.IsUnmeasurable("eccErrors") {
		t.Error("a later sentinel did not retract the reading")
	}
	if _, stored := measuredThenDeclared.Metrics["eccErrors"]; stored {
		t.Error("the retracted reading was left in Metrics; the two sets must stay disjoint")
	}
}

// ReportedNothing is the difference between "this runner did not measure" and
// "this runner did not measure THAT". Only the first is a machinery fault; the
// second is fail-closed's job. Widening this predicate weakens acceptance for
// every consumer, so each boundary is pinned.
func TestResult_ReportedNothing(t *testing.T) {
	cases := []struct {
		name   string
		stdout string
		exit   int
		want   bool
	}{
		{"no output whatsoever", "", 0, true},
		{"a marker line and nothing else", "FP4_GEMM_PASS\n", 0, true},
		{"prose only", "CUDA driver version is insufficient\n", 1, true},
		{"blank lines only", "\n\n   \n", 0, true},

		// Everything below is a runner that DID emit key=value output, however
		// unhelpfully. None of it may qualify.
		{"one real metric", "nonfinite_count=0\n", 0, false},
		{"a full report", computeSmokeStdout, 0, false},
		// An "n/a" is a positive declaration about the hardware, not silence.
		{"only unmeasurable declarations", "ecc_errors=n/a\nrows_remapped=n/a\n", 0, false},
		// A rejected name is still a line the runner printed: it looked, and
		// named its finding badly. That is a runner bug, not an absent harvest.
		{"only contract-invalid names", "Ecc_Errors=4\n", 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Parse("compute-smoke", tc.stdout, tc.exit).ReportedNothing(); got != tc.want {
				t.Errorf("ReportedNothing() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestResult_Numeric(t *testing.T) {
	got := Parse("compute-smoke", computeSmokeStdout, 0)

	if v, ok := got.Numeric("throughputTflops"); !ok || v != 101.99 {
		t.Errorf("Numeric(throughputTflops) = %v, %v; want 101.99, true", v, ok)
	}
	// A GPU's name is legitimate evidence but is not comparable, so a threshold
	// must not be able to accidentally treat it as a number.
	if _, ok := got.Numeric("gpuName"); ok {
		t.Error("Numeric() accepted a non-numeric metric")
	}
	if _, ok := got.Numeric("noSuchMetric"); ok {
		t.Error("Numeric() invented a value for a metric that was never reported")
	}
}

// panicLog is what a container holds when a Go runner panics: stdout and stderr
// are MERGED by the container runtime, and a runner that prints its metrics at
// the end (host-health does) has printed none. So there is no key=value line at
// all — which is byte-for-byte the shape of a legitimate skip, whose normal form
// is exactly "nothing to measure, nothing reported".
const panicLog = `probing kernel ring buffer
panic: runtime error: index out of range [3] with length 2

goroutine 1 [running]:
main.run(...)
	/src/main.go:190
main.main()
	/src/main.go:132 +0x1d
`

// TestExitTwoWithoutADeclarationIsAnErrorNotASkip is the regression guard for the
// worst-shaped bug this parser can have.
//
// An unrecovered Go panic exits 2. So does every Go RUNTIME fatal error —
// out-of-memory, concurrent map writes, stack exhaustion — none of which can be
// recovered even by a runner that wants to. Exit 2 is the skip code, and Skip is
// the single worst landing place for a crash: it is never retried, it does not
// affect the run's verdict, and it reports the node as one the test did not
// apply to. A crashed runner therefore used to certify a fleet nobody measured,
// inside a run that settled Passed.
func TestExitTwoWithoutADeclarationIsAnErrorNotASkip(t *testing.T) {
	got := Parse("host-health", panicLog, 2)
	if got.Verdict != VerdictError {
		t.Errorf("a panic exiting 2 is %q, want %q — the hardware was never judged",
			got.Verdict, VerdictError)
	}
	if !got.UndeclaredSkip {
		t.Error("UndeclaredSkip is false, so a consumer cannot tell this from any other Error")
	}
	// The evidence must survive: the tail of the trace is the only clue to what
	// actually died, and overwriting it with a tidier message would cost the
	// person debugging it the one thing they need.
	if !strings.Contains(got.Message, "main.go") {
		t.Errorf("Message = %q, want the tail of the panic retained", got.Message)
	}
}

// TestADeclaredSkipIsStillASkip is the other half, and it is what stops the
// guard above from being a blanket refusal of exit 2. A runner that says it
// skipped is believed.
func TestADeclaredSkipIsStillASkip(t *testing.T) {
	// Every marker a runner in this repository actually prints. If one is ever
	// renamed, this fails rather than silently reclassifying that runner's
	// skips as crashes.
	for _, out := range []string{
		"FP4_GEMM_SKIP: this part is not compute capability 12.0/12.1",
		"NCCL_SKIP: fewer than two ranks",
		"IB_WRITE_BW_SKIP: this node has no RDMA device",
		"GPUDIRECT_RDMA_SKIP: no peer-memory provider",
		"STRESSAPPTEST_SKIP: not enough free memory to size a run",
		"DCGM_DIAG_SKIP: no DCGM at /usr/local/dcgm",
		"MEMORY_BW_SKIP: fewer than two devices",
		"CLOCKPROBE_SKIP: no NVML on this host",
	} {
		got := Parse("custom", out+"\n", 2)
		if got.Verdict != VerdictSkip {
			t.Errorf("Parse(%q, exit 2) = %q, want %q", out, got.Verdict, VerdictSkip)
		}
		if got.UndeclaredSkip {
			t.Errorf("%q set UndeclaredSkip on a declared skip", out)
		}
	}
}

// TestOnlyExitTwoNeedsADeclaration keeps the new rule where it belongs. A pass,
// a fail and an error say what they mean by their exit code alone; only the skip
// code collides with a crash, so only the skip code asks for corroboration.
func TestOnlyExitTwoNeedsADeclaration(t *testing.T) {
	for code, want := range map[int]Verdict{0: VerdictPass, 1: VerdictFail, 3: VerdictError, 137: VerdictError} {
		got := Parse("host-health", panicLog, code)
		if got.Verdict != want {
			t.Errorf("exit %d = %q, want %q", code, got.Verdict, want)
		}
		if got.UndeclaredSkip {
			t.Errorf("exit %d set UndeclaredSkip, which is only about exit 2", code)
		}
	}
}

// TestDeclaresSkipIsNotFooledByProse guards the marker itself. It has to be
// recognisable in a merged stdout/stderr stream full of arbitrary text, without
// matching text that merely talks about skipping.
func TestDeclaresSkipIsNotFooledByProse(t *testing.T) {
	for _, out := range []string{
		"we will skip this if the device is absent",
		"SKIPPING: not a marker",
		"the runner printed skip_reason=none",
		"nested_SKIP is not at the start of a line",
	} {
		if DeclaresSkip(out) {
			t.Errorf("DeclaresSkip(%q) = true, want false", out)
		}
	}
	if !DeclaresSkip("first line\nHOST_HEALTH_SKIP: reason\nlast line") {
		t.Error("a marker on a middle line was not found; stdout and stderr are merged, so it can be anywhere")
	}
}

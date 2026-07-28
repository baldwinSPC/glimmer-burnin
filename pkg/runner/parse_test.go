package runner

import (
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

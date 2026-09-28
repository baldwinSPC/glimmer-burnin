package runner

import (
	_ "embed"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// cpuBenchGB10 is real stdout: cpu-bench, 60 s on spark-043a (GB10, 10x
// Cortex-X925 + 10x Cortex-A725), 2026-09-28.
//
//go:embed testdata/cpu-bench-gb10.txt
var cpuBenchGB10 string

func TestParse_RealCPUBenchOutputIsRegistered(t *testing.T) {
	got := Parse("cpu-bench", cpuBenchGB10, 0)
	if got.Verdict != VerdictPass || got.Message != "CPU_BENCH_PASS: measured" {
		t.Fatalf("verdict %q message %q", got.Verdict, got.Message)
	}
	if len(got.InvalidNames) != 0 {
		t.Errorf("names the contract rejects: %v", got.InvalidNames)
	}
	for name := range got.Metrics {
		if _, ok := contract.Lookup(name); !ok {
			t.Errorf("%q is emitted by our own runner and not registered", name)
		}
	}
	for name, want := range map[string]string{
		"fmaSingleCoreGflops": "15.60",
		"cpuBenchPerfCore":    "19", // an X925, not cpu0
		"streamArrayBytes":    "67108864",
	} {
		if got.Metrics[name] != want {
			t.Errorf("%s = %q, want %q", name, got.Metrics[name], want)
		}
	}
	for _, name := range []string{"fmaAllCoreGflops", "hostStreamCopyGBs", "hostStreamScaleGBs", "hostStreamAddGBs", "hostStreamTriadGBs"} {
		if _, ok := got.Metrics[name]; !ok {
			t.Errorf("%s missing", name)
		}
		if contract.UnitOf(name) == contract.UnitNone {
			t.Errorf("%s reads as dimensionless", name)
		}
	}
	if contract.SafeToThresholdOn("cpuBenchPerfCore") {
		t.Error("a core id must not be thresholdable")
	}
}

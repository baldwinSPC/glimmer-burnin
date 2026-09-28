package runner

import (
	_ "embed"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// gemmSweepGB10BF16 is real stdout: gemm-sweep's bf16 cell with the #544 load
// envelope, 45 s on spark-043a (GB10), 2026-09-28.
//
//go:embed testdata/gemm-sweep-gb10-bf16-envelope.txt
var gemmSweepGB10BF16 string

func TestParse_GemmSweepReportsTheStateItMeasuredUnder(t *testing.T) {
	got := Parse("gemm-sweep", gemmSweepGB10BF16, 0)
	if got.Verdict != VerdictPass {
		t.Fatalf("verdict %q (%q)", got.Verdict, got.Message)
	}
	for _, name := range []string{"smClockMHz", "gpuTempC", "powerDrawW", "throttleReasons", "throttleReasonsMask"} {
		if _, ok := got.Metrics[name]; !ok {
			t.Errorf("%s missing from %v", name, got.Metrics)
		}
		if _, ok := contract.Lookup(name); !ok {
			t.Errorf("%s is not registered", name)
		}
	}
	// The one that needs the alias: "mhz" would otherwise fold to "Mhz".
	if _, leaked := got.Metrics["smClockMhz"]; leaked {
		t.Error("sm_clock_mhz normalised to smClockMhz, which reads as dimensionless")
	}
	if contract.UnitOf("smClockMHz") != contract.UnitMegahertz {
		t.Error("smClockMHz lost its unit")
	}
}

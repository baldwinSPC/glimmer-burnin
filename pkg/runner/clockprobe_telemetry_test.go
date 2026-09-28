package runner

import (
	_ "embed"
	"strings"
	"testing"
)

// clockprobeGB10Telemetry is real stdout: clockprobe with the #547 series,
// 60 s on spark-85a9 (GB10), 2026-09-28. Steady at 83% of rated boost.
//
//go:embed testdata/clockprobe-gb10-telemetry.txt
var clockprobeGB10Telemetry string

func TestParse_ClockprobeEmitsItsSeriesAndDrift(t *testing.T) {
	got := Parse("clockprobe", clockprobeGB10Telemetry, 0)
	if got.Verdict != VerdictPass {
		t.Fatalf("verdict %q (%q)", got.Verdict, got.Message)
	}
	if d := got.Metrics["smClockSteadyStateDeltaPct"]; d != "-0.02" {
		t.Errorf("smClockSteadyStateDeltaPct = %q, want -0.02 (a steady clock)", d)
	}
	var found bool
	for _, a := range got.Artifacts {
		if a.Name == "telemetry.jsonl" && a.Dropped == "" {
			found = true
			// header + samples_taken
			if n := strings.Count(a.Payload, "\n"); n != 496 {
				t.Errorf("%d lines, want 496", n)
			}
			if strings.Contains(a.Payload, `"warmup": true`) {
				t.Error("clockprobe samples only after warm-up, yet the series marks warm-up points")
			}
		}
	}
	if !found {
		t.Fatalf("telemetry.jsonl not extracted: %+v", got.Artifacts)
	}
}

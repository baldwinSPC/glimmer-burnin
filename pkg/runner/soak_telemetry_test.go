package runner

import (
	_ "embed"
	"strconv"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// thermalSoakGB10Telemetry is real stdout: the #546/#547 thermal-soak build,
// 90 s on spark-043a (GB10, driver 580.82.09), 2026-09-28. The clock slid from
// 79.35% to 76.87% of rated across the run and the soak still passed.
//
//go:embed testdata/thermal-soak-gb10-telemetry.txt
var thermalSoakGB10Telemetry string

func TestParse_SoakTelemetryIsLiftedOutAndTheDriftIsRegistered(t *testing.T) {
	got := Parse("thermal-soak", thermalSoakGB10Telemetry, 0)
	if got.Verdict != VerdictPass {
		t.Fatalf("verdict %q (%q)", got.Verdict, got.Message)
	}
	if len(got.InvalidNames) != 0 {
		t.Errorf("names the contract rejects: %v", got.InvalidNames)
	}

	var tel *Artifact
	for i := range got.Artifacts {
		if got.Artifacts[i].Name == "telemetry.jsonl" {
			tel = &got.Artifacts[i]
		}
	}
	if tel == nil || tel.Dropped != "" {
		t.Fatalf("telemetry.jsonl not extracted: %+v", got.Artifacts)
	}
	if tel.MediaType != "application/x-ndjson" {
		t.Errorf("media type %q", tel.MediaType)
	}
	lines := strings.Count(tel.Payload, "\n")
	if lines != 360 { // header + 359 samples
		t.Errorf("%d lines, want 360 (header + samples_taken)", lines)
	}
	// Every "key": value line of the series must have stayed out of Metrics.
	for name := range got.Metrics {
		if strings.Contains(name, "\"") || name == "smMHz" || name == "tempC" {
			t.Errorf("series content leaked into metrics as %q", name)
		}
	}

	d, err := strconv.ParseFloat(got.Metrics["smClockSteadyStateDeltaPct"], 64)
	if err != nil || d > -4 || d < -5 {
		t.Errorf("smClockSteadyStateDeltaPct = %q, want about -4.43", got.Metrics["smClockSteadyStateDeltaPct"])
	}
	if contract.SafeToThresholdOn("smClockSteadyStateDeltaPct") {
		t.Error("the drift is evidence and must not be thresholdable")
	}
}

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// This file is the seam between the runner and the operator, and it is the
// reason the runner's source lives in the repository's root module instead of
// beside its Dockerfile as an island.
//
// The runner prints its OWN key names; pkg/runner maps them to the canonical
// names a threshold is written against, and pkg/contract owns the grammar those
// names must obey. Nothing at runtime reconciles a disagreement — a key that
// normalises to an illegal or unexpected name is silently unthresholdable, and
// the failure shows up as a node that cannot be accepted for reasons nobody can
// see. Running the real runner's real output through the real parser is what
// keeps the three in step.
//
// These imports are test-only. The image build removes *_test.go, which is what
// lets the non-test sources stay standard-library-only.

func TestEmittedMetricNamesObeyTheContract(t *testing.T) {
	gpu := healthyGPU()
	gpu[fECCUncAgg] = "2" // exercise a failing run: the evidence must survive it
	f := newE2EFixture(t, gpu, "1")
	f.logDuringWindow(t, "3,9,9,-;NVRM: Xid (PCI:0000:01:00): 48, DBE\n")

	var buf bytes.Buffer
	code := run(f.cfg, &buf)
	out := buf.String()

	res := runner.Parse("host-health", out, code)

	if len(res.InvalidNames) != 0 {
		t.Errorf("runner emitted keys that fail the metric grammar: %v", res.InvalidNames)
	}
	for name := range res.Metrics {
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("canonical name from this runner is invalid: %v", err)
		}
	}

	// A key that normalises onto another key's canonical name would silently
	// discard one of two real measurements (parsing is last-occurrence-wins).
	// The final line is the verdict, not a metric — and on a failing run it
	// carries the offending "key=value" inside prose, which the parser
	// correctly treats as a message rather than a measurement.
	//
	// DISTINCT keys, not lines: the stage breadcrumb is deliberately printed
	// once per stage, and last-occurrence-wins is what makes the recorded value
	// the furthest stage reached. Counting lines would read a progress marker as
	// a name collision.
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	metricLines, verdictLine := lines[:len(lines)-1], lines[len(lines)-1]
	if !strings.HasPrefix(verdictLine, "HOST_HEALTH_") {
		t.Fatalf("last line = %q, want the verdict marker", verdictLine)
	}
	emitted := map[string]bool{}
	for _, line := range metricLines {
		if key, _, ok := strings.Cut(line, "="); ok {
			emitted[strings.TrimSpace(key)] = true
		}
	}
	if parsed := len(res.Metrics) + len(res.Unmeasurable); len(emitted) != parsed {
		t.Errorf("%d distinct printed keys parsed into %d canonical names — two keys collide", len(emitted), parsed)
	}

	// The five counters the host-health TestKind is defined by must reach the
	// operator under their registered names.
	wantCanonical := map[string]string{
		"xidEvents":         "1",
		"eccErrors":         "2",
		"remappedRows":      "0",
		"pcieReplayErrors":  "0",
		"nicLinkDownEvents": "0",
	}
	for name, want := range wantCanonical {
		got, ok := res.Metrics[name]
		if !ok {
			t.Errorf("canonical metric %q was not produced by parsing this runner's output:\n%s", name, out)
			continue
		}
		if got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
		if _, registered := contract.Lookup(name); !registered {
			t.Errorf("%s must be a registered metric", name)
		}
		// A gate the profile cannot express is a gate that does not exist.
		if !contract.SafeToThresholdOn(name) {
			t.Errorf("%s is not safe to threshold on, so this runner cannot be gated on it", name)
		}
	}

	if res.Verdict != runner.VerdictFail {
		t.Errorf("verdict = %q, want Fail for a node with an uncorrected ECC error", res.Verdict)
	}
	if !strings.HasPrefix(res.Message, "HOST_HEALTH_FAULT") {
		t.Errorf("parsed message = %q, want the fault marker", res.Message)
	}
}

// The sentinel is a shared constant that cannot be shared: this runner's
// non-test sources are standard-library-only, so it re-declares the literal
// instead of importing pkg/runner. A drift between the two would not fail to
// compile — it would parse "n/a" as an ordinary metric value, and every
// RequiredIfMeasurable gate on this runner would silently start failing healthy
// nodes.
func TestUnmeasurableSentinelMatchesTheParser(t *testing.T) {
	if unmeasurableValue != runner.Unmeasurable {
		t.Errorf("runner emits %q but pkg/runner reserves %q", unmeasurableValue, runner.Unmeasurable)
	}
}

// The seam for GB10: what this runner prints for a part with no ECC must reach
// pkg/verdict as an explicit declaration under the CANONICAL names a profile's
// thresholds are written against — not as a value, and not as silence.
func TestUnmeasurableCountersReachTheParserAsDeclarations(t *testing.T) {
	f := newE2EFixture(t, sparkGPU(), "0")
	var buf bytes.Buffer
	code := run(f.cfg, &buf)
	res := runner.Parse("host-health", buf.String(), code)

	if res.Verdict != runner.VerdictPass {
		t.Fatalf("verdict = %q, want Pass for a healthy Spark:\n%s", res.Verdict, buf.String())
	}
	for _, name := range []string{"eccErrors", "remappedRows"} {
		if !res.IsUnmeasurable(name) {
			t.Errorf("%q did not reach the parser as unmeasurable: %v", name, res.Unmeasurable)
		}
		if v, stored := res.Metrics[name]; stored {
			t.Errorf("%q reached the parser as the value %q; a threshold would compare against it", name, v)
		}
	}
	// The counters this hardware CAN produce are unaffected.
	if res.Metrics["xidEvents"] != "0" {
		t.Errorf("xidEvents = %q, want 0", res.Metrics["xidEvents"])
	}
	if len(res.InvalidNames) != 0 {
		t.Errorf("declaring a counter unmeasurable produced invalid names: %v", res.InvalidNames)
	}
}

// Evidence metrics are unregistered by design (the registry is an open world),
// but they still have to obey the grammar and remain thresholdable.
func TestEvidenceMetricNamesAreWellFormed(t *testing.T) {
	f := newE2EFixture(t, healthyGPU(), "0")
	var buf bytes.Buffer
	code := run(f.cfg, &buf)
	res := runner.Parse("host-health", buf.String(), code)

	for _, name := range []string{
		"xidPreexisting",
		"eccMode",
		"eccCorrectedAggregate",
		"eccUncorrectedVolatile",
		"remappedRowsCorrectable",
		"retiredPagesTotal",
		"pcieReplayTotal",
		"pcieAerCorrectable",
		"pcieAerFatalTotal",
		"nicLinkDownTotal",
		"ibLinkDownTotal",
		"kernelHwErrors",
		"observationWindowS",
		"elapsedS",
	} {
		if _, ok := res.Metrics[name]; !ok {
			t.Errorf("expected evidence metric %q in:\n%s", name, buf.String())
			continue
		}
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("evidence metric name is invalid: %v", err)
		}
		if !contract.SafeToThresholdOn(name) {
			t.Errorf("%s cannot be thresholded, so a stricter profile could not use it", name)
		}
	}

	// Rule 1 of the grammar: a metric carrying a unit must declare it in the
	// name. These two are the runner's only dimensional numbers.
	if contract.UnitOf("observationWindowS") != contract.UnitSeconds {
		t.Error("observationWindowS must declare seconds")
	}
	if contract.UnitOf("gpuTempC") != contract.UnitCelsius {
		t.Error("gpuTempC must declare celsius")
	}
}

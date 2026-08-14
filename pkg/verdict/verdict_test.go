package verdict

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func th(metric string, op contract.Comparison, val string) contract.Threshold {
	return contract.Threshold{Metric: metric, Comparison: op, Value: val}
}

// ifMeasurable is th() with the only applicability that relaxes anything.
func ifMeasurable(metric string, op contract.Comparison, val string) contract.Threshold {
	t := th(metric, op, val)
	t.Applicability = contract.RequiredIfMeasurable
	return t
}

// unmeasured is the runner's explicit declaration, as pkg/runner produces it
// from an "n/a" value.
func unmeasured(names ...string) map[string]bool {
	set := map[string]bool{}
	for _, n := range names {
		set[n] = true
	}
	return set
}

func TestEvaluate(t *testing.T) {
	cases := []struct {
		name       string
		metrics    map[string]string
		thresholds []contract.Threshold
		wantPass   bool
	}{
		{
			name:     "no thresholds passes",
			metrics:  map[string]string{"busBandwidthGBs": "23.4"},
			wantPass: true,
		},
		{
			name:       "nccl bus bandwidth meets floor",
			metrics:    map[string]string{"busBandwidthGBs": "23.4"},
			thresholds: []contract.Threshold{th("busBandwidthGBs", contract.GTE, "20")},
			wantPass:   true,
		},
		{
			name:       "nccl bus bandwidth below floor fails",
			metrics:    map[string]string{"busBandwidthGBs": "12.0"},
			thresholds: []contract.Threshold{th("busBandwidthGBs", contract.GTE, "20")},
			wantPass:   false,
		},
		{
			name:       "ecc errors must be zero",
			metrics:    map[string]string{"eccErrors": "0"},
			thresholds: []contract.Threshold{th("eccErrors", contract.EQ, "0")},
			wantPass:   true,
		},
		{
			name:       "ecc errors nonzero fails",
			metrics:    map[string]string{"eccErrors": "3"},
			thresholds: []contract.Threshold{th("eccErrors", contract.EQ, "0")},
			wantPass:   false,
		},
		{
			name:       "missing metric fails closed (must not silently pass acceptance)",
			metrics:    map[string]string{},
			thresholds: []contract.Threshold{th("busBandwidthGBs", contract.GTE, "20")},
			wantPass:   false,
		},
		{
			name:       "non-numeric metric fails",
			metrics:    map[string]string{"busBandwidthGBs": "fast"},
			thresholds: []contract.Threshold{th("busBandwidthGBs", contract.GTE, "20")},
			wantPass:   false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Evaluate(c.metrics, nil, c.thresholds)
			if got.Passed != c.wantPass {
				t.Fatalf("Evaluate = %v (%q), want %v", got.Passed, got.Message, c.wantPass)
			}
			if !got.Passed && got.Message == "" {
				t.Error("a failing verdict must carry an explanatory message")
			}
			if len(got.NotEvaluated) != 0 {
				t.Errorf("nothing was declared unmeasurable, so nothing may go un-evaluated: %+v", got.NotEvaluated)
			}
		})
	}
}

// ─── The unmeasurable matrix ──────────────────────────────────────────────────
//
// Four cases, and the difference between the first two is the whole design: a
// runner that DID NOT SAY is not a runner that said "cannot".

// Case 1: absent entirely. A runner that forgot to emit, crashed before
// emitting, or had its key renamed is still a failure — under EVERY
// applicability. If RequiredIfMeasurable relaxed on absence it would convert
// every broken probe into an acceptance, which is the exact false negative this
// project exists to prevent.
func TestEvaluate_AbsentMetricFailsClosedUnderEveryApplicability(t *testing.T) {
	for _, th := range []contract.Threshold{
		th("eccErrors", contract.EQ, "0"),
		ifMeasurable("eccErrors", contract.EQ, "0"),
	} {
		applicability := string(th.Applicability)
		if applicability == "" {
			applicability = "(unset)"
		}
		t.Run(applicability, func(t *testing.T) {
			got := Evaluate(map[string]string{}, unmeasured(), []contract.Threshold{th})
			if got.Passed {
				t.Fatal("an absent metric passed; absence is not a declaration of unmeasurability")
			}
			if len(got.NotEvaluated) != 0 {
				t.Errorf("an absent metric was reported as not-evaluated rather than failed: %+v", got.NotEvaluated)
			}
			if !strings.Contains(got.Message, "not reported") {
				t.Errorf("message = %q, want it to say the runner did not report the metric", got.Message)
			}
		})
	}
}

// Case 2: declared unmeasurable, Required. An unmeasured quantity has not been
// shown to be within limits, so the default stays fail-closed — this is what
// keeps an ECC-capable GPU with ECC switched off from sliding through.
func TestEvaluate_UnmeasurableUnderRequiredFails(t *testing.T) {
	got := Evaluate(
		map[string]string{"xidEvents": "0"},
		unmeasured("eccErrors"),
		[]contract.Threshold{th("eccErrors", contract.EQ, "0")},
	)
	if got.Passed {
		t.Fatal("a Required threshold on an unmeasurable metric passed")
	}
	if !strings.Contains(got.Message, "eccErrors") {
		t.Errorf("message = %q, want it to name the metric", got.Message)
	}
	if !strings.Contains(got.Message, ReasonUnmeasurable) {
		t.Errorf("message = %q, want it to say the hardware cannot measure it", got.Message)
	}
	if len(got.NotEvaluated) != 0 {
		t.Errorf("a Required gate must be evaluated and failed, not skipped: %+v", got.NotEvaluated)
	}
}

// Case 3: declared unmeasurable, RequiredIfMeasurable. The threshold is not
// applied — and that must NEVER be silent. The node passes, and the outcome
// carries a reportable record of the gate that did not run.
func TestEvaluate_UnmeasurableUnderRequiredIfMeasurableIsNotEvaluated(t *testing.T) {
	got := Evaluate(
		map[string]string{"xidEvents": "0"},
		unmeasured("eccErrors", "remappedRows"),
		[]contract.Threshold{
			th("xidEvents", contract.EQ, "0"),
			ifMeasurable("eccErrors", contract.EQ, "0"),
			ifMeasurable("remappedRows", contract.EQ, "0"),
		},
	)
	if !got.Passed {
		t.Fatalf("a healthy GB10 failed acceptance: %q", got.Message)
	}
	if len(got.NotEvaluated) != 2 {
		t.Fatalf("NotEvaluated = %+v, want both un-applied gates reported", got.NotEvaluated)
	}
	// Spec order, so a report reads like the profile it came from.
	if got.NotEvaluated[0].Metric != "eccErrors" || got.NotEvaluated[1].Metric != "remappedRows" {
		t.Errorf("NotEvaluated = %+v, want eccErrors then remappedRows", got.NotEvaluated)
	}
	for _, ne := range got.NotEvaluated {
		if ne.Reason != ReasonUnmeasurable {
			t.Errorf("%s: reason = %q, want %q", ne.Metric, ne.Reason, ReasonUnmeasurable)
		}
	}
	msg := got.NotEvaluatedMessage()
	for _, want := range []string{"not evaluated", "eccErrors", "remappedRows", ReasonUnmeasurable} {
		if !strings.Contains(msg, want) {
			t.Errorf("NotEvaluatedMessage = %q, want it to contain %q", msg, want)
		}
	}
}

// Case 4: everything else unchanged. RequiredIfMeasurable is not "optional": on
// hardware that CAN measure, the gate is applied exactly as Required.
func TestEvaluate_RequiredIfMeasurableStillGatesAMeasuredMetric(t *testing.T) {
	thresholds := []contract.Threshold{ifMeasurable("eccErrors", contract.EQ, "0")}

	clean := Evaluate(map[string]string{"eccErrors": "0"}, unmeasured(), thresholds)
	if !clean.Passed || len(clean.NotEvaluated) != 0 {
		t.Errorf("a clean measured counter must pass and be evaluated: %+v", clean)
	}

	dirty := Evaluate(map[string]string{"eccErrors": "4"}, unmeasured(), thresholds)
	if dirty.Passed {
		t.Fatal("RequiredIfMeasurable let a measured ECC error through; it is not an optional gate")
	}
	if !strings.Contains(dirty.Message, "eccErrors") {
		t.Errorf("message = %q, want it to name the failing threshold", dirty.Message)
	}
	if len(dirty.NotEvaluated) != 0 {
		t.Errorf("a measured metric was reported as not-evaluated: %+v", dirty.NotEvaluated)
	}
}

// A value AND a declaration of unmeasurability for the same metric is a
// self-contradictory runner. pkg/runner cannot produce it, but a hand-built
// caller can, and ambiguity resolves toward failing.
func TestEvaluate_ContradictoryRunnerOutputFailsClosed(t *testing.T) {
	for _, th := range []contract.Threshold{
		th("eccErrors", contract.EQ, "0"),
		ifMeasurable("eccErrors", contract.EQ, "0"),
	} {
		got := Evaluate(
			map[string]string{"eccErrors": "0"},
			unmeasured("eccErrors"),
			[]contract.Threshold{th},
		)
		if got.Passed {
			t.Errorf("applicability %q: contradictory output passed", th.Applicability)
		}
		if len(got.NotEvaluated) != 0 {
			t.Errorf("applicability %q: contradiction was reported as not-evaluated: %+v", th.Applicability, got.NotEvaluated)
		}
	}
}

// The enum is validated by the apiserver, but the verdict must not depend on
// that: a value nobody recognises is Required, never the relaxed one.
func TestEvaluate_UnknownApplicabilityFailsClosed(t *testing.T) {
	th := th("eccErrors", contract.EQ, "0")
	th.Applicability = contract.Applicability("Optional")

	got := Evaluate(map[string]string{}, unmeasured("eccErrors"), []contract.Threshold{th})
	if got.Passed {
		t.Fatal("an unrecognised applicability relaxed the gate")
	}
}

// A failure stops evaluation, but the gates already found to be un-evaluated are
// still part of the report: they are evidence about the run either way.
func TestEvaluate_NotEvaluatedSurvivesALaterFailure(t *testing.T) {
	got := Evaluate(
		map[string]string{"xidEvents": "7"},
		unmeasured("eccErrors"),
		[]contract.Threshold{
			ifMeasurable("eccErrors", contract.EQ, "0"),
			th("xidEvents", contract.EQ, "0"),
		},
	)
	if got.Passed {
		t.Fatal("a violated threshold passed")
	}
	if len(got.NotEvaluated) != 1 || got.NotEvaluated[0].Metric != "eccErrors" {
		t.Errorf("NotEvaluated = %+v, want eccErrors kept alongside the failure", got.NotEvaluated)
	}
}

func TestOutcome_NotEvaluatedMessageIsEmptyWhenEverythingRan(t *testing.T) {
	got := Evaluate(map[string]string{"eccErrors": "0"}, nil,
		[]contract.Threshold{ifMeasurable("eccErrors", contract.EQ, "0")})
	if msg := got.NotEvaluatedMessage(); msg != "" {
		t.Errorf("NotEvaluatedMessage = %q, want empty", msg)
	}
}

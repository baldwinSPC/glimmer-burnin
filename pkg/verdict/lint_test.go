package verdict

import (
	"strings"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// ─── Equal / NotEqual are EXACT, and that is the pinned semantics ─────────────

// The case that motivates the whole rule. 83.22 does not survive the trip
// through a float64 average of sampled clocks, so an exact gate on a continuous
// metric is a permanent failure wearing the costume of a hardware verdict.
func TestEvaluate_EqualIsExactAndTrapsAContinuousMetric(t *testing.T) {
	thresholds := []burninv1alpha1.Threshold{th("sustainedClockPct", burninv1alpha1.EQ, "83.22")}

	// The measurement a healthy part actually produces: 83.22 as computed, not
	// as typed. 0.1+0.2 is the canonical demonstration that the two differ.
	measured := 83.22000000000001

	got := Evaluate(map[string]string{"sustainedClockPct": "83.22000000000001"}, nil, thresholds)
	if got.Passed {
		t.Fatalf("%v exactly equalled 83.22; Equal must be exact", measured)
	}
	if !strings.Contains(got.Message, "sustainedClockPct") {
		t.Errorf("message = %q, want it to name the metric", got.Message)
	}

	// And the exactness is real in the other direction: the identical decimal
	// does satisfy it. Equal is not broken, it is exact — which is why it is the
	// wrong tool here and the right tool for a counter.
	if same := Evaluate(map[string]string{"sustainedClockPct": "83.22"}, nil, thresholds); !same.Passed {
		t.Errorf("the same decimal failed its own Equal gate: %q", same.Message)
	}
}

// The other half of the decision: no epsilon. A counter gate means EXACTLY the
// number it names, because a tolerance around zero ECC errors is a tolerance for
// ECC errors. This is the test that fails if anyone adds a fuzz factor.
func TestEvaluate_EqualHasNoEpsilonOnCounters(t *testing.T) {
	thresholds := []burninv1alpha1.Threshold{th("eccErrors", burninv1alpha1.EQ, "0")}

	for _, reported := range []string{"1", "0.5", "0.000001", "-0.000001"} {
		t.Run(reported, func(t *testing.T) {
			if got := Evaluate(map[string]string{"eccErrors": reported}, nil, thresholds); got.Passed {
				t.Fatalf("eccErrors=%s satisfied `Equal 0`; an epsilon has been introduced and a real ECC violation can now pass", reported)
			}
		})
	}

	// Every integer a counter can carry compares exactly, which is what makes
	// exact comparison the correct tool for counters in the first place.
	if got := Evaluate(map[string]string{"eccErrors": "0"}, nil, thresholds); !got.Passed {
		t.Errorf("a clean counter failed its own gate: %q", got.Message)
	}
	if got := Evaluate(map[string]string{"xidEvents": "9007199254740992"}, nil,
		[]burninv1alpha1.Threshold{th("xidEvents", burninv1alpha1.EQ, "9007199254740992")}); !got.Passed {
		t.Errorf("a large integer counter failed an exact gate it matches: %q", got.Message)
	}
}

// NotEqual is exact too, and its failure mode is the interesting one: a
// non-finite measurement compares false against everything, which would make
// NotEqual pass. A value that is not a measurement must never satisfy a gate.
func TestEvaluate_NonFiniteValuesNeverSatisfyAGate(t *testing.T) {
	for _, tc := range []struct {
		name       string
		metrics    map[string]string
		thresholds []burninv1alpha1.Threshold
		wantIn     string
	}{
		{
			name:       "NaN metric under NotEqual",
			metrics:    map[string]string{"eccErrors": "NaN"},
			thresholds: []burninv1alpha1.Threshold{th("eccErrors", burninv1alpha1.NEQ, "1")},
			wantIn:     "not a finite measurement",
		},
		{
			name:       "Inf metric under a ceiling",
			metrics:    map[string]string{"gpuTempC": "+Inf"},
			thresholds: []burninv1alpha1.Threshold{th("gpuTempC", burninv1alpha1.LTE, "85")},
			wantIn:     "not a finite measurement",
		},
		{
			name:       "NaN threshold value under NotEqual",
			metrics:    map[string]string{"eccErrors": "0"},
			thresholds: []burninv1alpha1.Threshold{th("eccErrors", burninv1alpha1.NEQ, "NaN")},
			wantIn:     "not a finite number",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := Evaluate(tc.metrics, nil, tc.thresholds)
			if got.Passed {
				t.Fatal("a non-finite value satisfied a threshold; nothing was measured and nothing may be accepted")
			}
			if !strings.Contains(got.Message, tc.wantIn) {
				t.Errorf("message = %q, want it to contain %q", got.Message, tc.wantIn)
			}
		})
	}
}

// ─── ValidateThresholds: the same rule, said before the run ───────────────────

func problemFor(t *testing.T, threshold burninv1alpha1.Threshold) []Problem {
	t.Helper()
	return ValidateThresholds([]burninv1alpha1.Threshold{threshold})
}

func onlyProblem(t *testing.T, threshold burninv1alpha1.Threshold) Problem {
	t.Helper()
	ps := problemFor(t, threshold)
	if len(ps) != 1 {
		t.Fatalf("got %d problems, want exactly 1: %v", len(ps), ps)
	}
	return ps[0]
}

// The gates an author is supposed to write must not be nagged about, or the
// check gets switched off and stops protecting anything.
func TestValidateThresholds_AcceptsTheGatesItIsFor(t *testing.T) {
	sound := []burninv1alpha1.Threshold{
		th("eccErrors", burninv1alpha1.EQ, "0"),
		th("throttleEvents", burninv1alpha1.EQ, "0"),
		th("miscompares", burninv1alpha1.EQ, "0"),
		th("iterationsCompleted", burninv1alpha1.NEQ, "0"),
		th("busBandwidthGBs", burninv1alpha1.GTE, "20"),
		th("gpuTempC", burninv1alpha1.LTE, "85"),
		th("sustainedClockPct", burninv1alpha1.GTE, "90"),  // the correct form of the trap below
		th("sustainedClockPct", burninv1alpha1.LTE, "105"), // …and its other side
		// An unregistered metric from a third-party runner: the registry is an
		// open world, and this package does not know better than its author.
		th("vendorFabricRetries", burninv1alpha1.EQ, "0"),
		th("vendorLinkUtilPct", burninv1alpha1.GTE, "80"),
	}
	if got := ValidateThresholds(sound); len(got) != 0 {
		t.Fatalf("sound thresholds were rejected: %v", got)
	}
	if got := ValidateThresholds(nil); len(got) != 0 {
		t.Errorf("no thresholds produced problems: %v", got)
	}
}

// The issue's own example, caught while the author can still fix it.
func TestValidateThresholds_FlagsExactComparisonOnAContinuousMetric(t *testing.T) {
	for _, op := range []burninv1alpha1.Comparison{burninv1alpha1.EQ, burninv1alpha1.NEQ} {
		t.Run(string(op), func(t *testing.T) {
			p := onlyProblem(t, th("sustainedClockPct", op, "83.22"))
			if p.Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q", p.Severity, SeverityUnsound)
			}
			if p.Metric != "sustainedClockPct" || p.Index != 0 {
				t.Errorf("problem = %+v, want it to point at threshold 0 / sustainedClockPct", p)
			}
			// The message has to name the fix, or it is just a complaint.
			for _, want := range []string{"Pct", string(burninv1alpha1.GTE), string(burninv1alpha1.LTE)} {
				if !strings.Contains(p.Reason, want) {
					t.Errorf("reason = %q, want it to mention %q", p.Reason, want)
				}
			}
			if !strings.Contains(p.Error(), "sustainedClockPct") {
				t.Errorf("Error() = %q, want it to name the metric", p.Error())
			}
		})
	}
}

// Every dimensional metric is continuous by construction — that is what the unit
// suffix in the name means — so the rule is derivable, not a hand-kept list.
func TestValidateThresholds_FlagsExactComparisonOnEveryUnitBearingMetric(t *testing.T) {
	for _, metric := range []string{
		"busBandwidthGBs", "bandwidthGbps", "readBandwidthMBs", "latencyUs",
		"elapsedS", "gpuTempC", "powerDrawW", "smClockMHz", "sustainedThroughputTflops",
	} {
		t.Run(metric, func(t *testing.T) {
			p := onlyProblem(t, th(metric, burninv1alpha1.EQ, "100"))
			if p.Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q", p.Severity, SeverityUnsound)
			}
		})
	}
}

// A dimensionless name does not make a fractional exact gate sensible: a counter
// counts, and `Equal 0.5` can never hold either.
func TestValidateThresholds_FlagsExactComparisonAgainstAFractionalValue(t *testing.T) {
	p := onlyProblem(t, th("miscompares", burninv1alpha1.EQ, "0.5"))
	if p.Severity != SeverityUnsound {
		t.Errorf("severity = %q, want %q", p.Severity, SeverityUnsound)
	}
	if !strings.Contains(p.Reason, "whole number") {
		t.Errorf("reason = %q, want it to say the value is not a whole number", p.Reason)
	}
}

// The flag pkg/contract already carries, applied where a threshold is written.
// throughputTflops is a single unwarmed launch: a gate on it decides on
// scheduling noise while looking exactly like a hardware judgement.
func TestValidateThresholds_FlagsAGateOnAnEvidenceOnlyMetric(t *testing.T) {
	for _, metric := range []string{"throughputTflops", "ratedBoostClockMHz"} {
		t.Run(metric, func(t *testing.T) {
			ps := problemFor(t, th(metric, burninv1alpha1.GTE, "90"))
			if len(ps) != 1 {
				t.Fatalf("got %d problems, want 1: %v", len(ps), ps)
			}
			if ps[0].Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q", ps[0].Severity, SeverityUnsound)
			}
			if !strings.Contains(ps[0].Reason, "evidence") {
				t.Errorf("reason = %q, want it to say the metric is evidence, not acceptance", ps[0].Reason)
			}
		})
	}
}

// A threshold that cannot be evaluated at all fails every node forever, and the
// report looks like a fleet of bad parts. Catching it at admission costs one
// error message.
func TestValidateThresholds_FlagsThresholdsThatCanNeverBeEvaluated(t *testing.T) {
	cases := []struct {
		name      string
		threshold burninv1alpha1.Threshold
		wantIn    string
	}{
		{
			name:      "ungrammatical metric name is dropped by the parser",
			threshold: th("bus_bandwidth_gbs", burninv1alpha1.GTE, "20"),
			wantIn:    "can never be satisfied",
		},
		{
			name:      "empty metric name",
			threshold: th("", burninv1alpha1.GTE, "20"),
			wantIn:    "must not be empty",
		},
		{
			name:      "non-numeric value",
			threshold: th("busBandwidthGBs", burninv1alpha1.GTE, "fast"),
			wantIn:    "not numeric",
		},
		{
			name:      "non-finite value",
			threshold: th("eccErrors", burninv1alpha1.NEQ, "NaN"),
			wantIn:    "not a finite number",
		},
		{
			name:      "unrecognised comparison",
			threshold: th("eccErrors", burninv1alpha1.Comparison("EqualIsh"), "0"),
			wantIn:    "fails closed",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ps := problemFor(t, c.threshold)
			if len(ps) == 0 {
				t.Fatal("no problem reported")
			}
			var found bool
			for _, p := range ps {
				if p.Severity != SeverityMalformed {
					continue
				}
				if strings.Contains(p.Reason, c.wantIn) {
					found = true
				}
			}
			if !found {
				t.Errorf("problems = %v, want a %s one containing %q", ps, SeverityMalformed, c.wantIn)
			}
		})
	}
}

// An author fixing a profile should see the whole list, and each problem must
// point at the threshold it came from.
func TestValidateThresholds_ReportsEveryProblemWithItsIndex(t *testing.T) {
	got := ValidateThresholds([]burninv1alpha1.Threshold{
		th("busBandwidthGBs", burninv1alpha1.GTE, "20"),       // fine
		th("gpuTempC", burninv1alpha1.EQ, "85"),               // unsound: exact on continuous
		th("throughputTflops", burninv1alpha1.GTE, "90"),      // unsound: evidence only
		th("eccErrors", burninv1alpha1.Comparison("~="), "0"), // malformed: unknown comparison
		th("latencyUs", burninv1alpha1.EQ, "not-a-number"),    // malformed value AND exact on continuous
	})
	if len(got) != 5 {
		t.Fatalf("got %d problems, want 5: %v", len(got), got)
	}
	wantIndices := []int{1, 2, 3, 4, 4}
	for i, p := range got {
		if p.Index != wantIndices[i] {
			t.Errorf("problem %d has index %d, want %d (%v)", i, p.Index, wantIndices[i], p)
		}
	}
}

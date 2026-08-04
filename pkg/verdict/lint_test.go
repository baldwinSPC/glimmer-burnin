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

// ─── Unregistered metrics: open world, except on a runner we ship ─────────────

// The #65 instance, caught. clockprobe emits pdWedgeSuspected as
// "true"/"false"/"unknown", so a gate on it is never numeric and fails closed on
// every node forever — and every check this package had passed it clean, because
// the name is grammatical and an unregistered name is legal.
func TestValidateThresholdsForKind_FlagsAnUnregisteredMetricOnABuiltInKind(t *testing.T) {
	for _, metric := range []string{"pdWedgeSuspected", "throttleClassification", "clockFloorBasis"} {
		t.Run(metric, func(t *testing.T) {
			ps := ValidateThresholdsForKind(burninv1alpha1.KindClockProbe,
				[]burninv1alpha1.Threshold{th(metric, burninv1alpha1.EQ, "0")})
			if len(ps) != 1 {
				t.Fatalf("got %d problems, want 1: %v", len(ps), ps)
			}
			// Advice, never a refusal: this cannot prove the metric is absent,
			// and a first-party runner may legitimately emit an unregistered
			// number that a stricter profile then gates on.
			if ps[0].Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q — an unprovable suspicion must not block a run", ps[0].Severity, SeverityUnsound)
			}
			if ps[0].Metric != metric || ps[0].Index != 0 {
				t.Errorf("problem = %+v, want it to point at threshold 0 / %s", ps[0], metric)
			}
			// Both readings have to reach the author, since only they can tell
			// which one it is.
			for _, want := range []string{"registry", "register it", "never satisfied"} {
				if !strings.Contains(ps[0].Reason, want) {
					t.Errorf("reason = %q, want it to mention %q", ps[0].Reason, want)
				}
			}
		})
	}
}

// The other half of the rule, and the one that keeps the registry OPEN. A runner
// this project does not ship may report whatever it measures; forbidding that —
// or merely nagging about it — is what pushes an author toward stuffing data
// into a message string where nothing can gate on it.
func TestValidateThresholdsForKind_SaysNothingAboutRunnersThisProjectDoesNotShip(t *testing.T) {
	thresholds := []burninv1alpha1.Threshold{
		th("vendorFabricRetries", burninv1alpha1.EQ, "0"),
		th("vendorLinkUtilPct", burninv1alpha1.GTE, "80"),
		th("pdWedgeSuspected", burninv1alpha1.EQ, "0"),
	}
	for _, kind := range []burninv1alpha1.TestKind{
		burninv1alpha1.KindCustom,
		burninv1alpha1.TestKind("some-vendors-own-runner"),
		burninv1alpha1.TestKind(""),
	} {
		t.Run(string(kind), func(t *testing.T) {
			if got := ValidateThresholdsForKind(kind, thresholds); len(got) != 0 {
				t.Errorf("kind %q: unregistered names were flagged on a runner this project does not own: %v", kind, got)
			}
		})
	}
}

// ValidateThresholds is the kind-agnostic form and must stay that way: it is
// public API with two dispatchers, and one of them has no TestKind to give.
func TestValidateThresholds_IsTheKindAgnosticForm(t *testing.T) {
	thresholds := []burninv1alpha1.Threshold{
		th("pdWedgeSuspected", burninv1alpha1.EQ, "0"),
		th("gpuTempC", burninv1alpha1.LTE, "85"),
	}
	if got := ValidateThresholds(thresholds); len(got) != 0 {
		t.Errorf("the kind-agnostic form gained an opinion about an unregistered name: %v", got)
	}
	// The registered metrics this project owns are judged identically either
	// way — the kind changes exactly one check and nothing else.
	registered := []burninv1alpha1.Threshold{
		th("sustainedClockPct", burninv1alpha1.EQ, "83.22"),
		th("throughputTflops", burninv1alpha1.GTE, "90"),
	}
	agnostic := ValidateThresholds(registered)
	aware := ValidateThresholdsForKind(burninv1alpha1.KindClockProbe, registered)
	if len(agnostic) != len(aware) {
		t.Fatalf("kind-aware form reported %d problems and the agnostic form %d: %v vs %v", len(aware), len(agnostic), aware, agnostic)
	}
	for i := range agnostic {
		if agnostic[i] != aware[i] {
			t.Errorf("problem %d differs by kind: %+v vs %+v", i, agnostic[i], aware[i])
		}
	}
}

// A single typo must produce a single finding. "…and it is not registered" adds
// nothing to "the parser will drop it", and two complaints about one mistake is
// how a check earns the reputation that gets it switched off.
func TestValidateThresholdsForKind_DoesNotPileOnAnUngrammaticalName(t *testing.T) {
	ps := ValidateThresholdsForKind(burninv1alpha1.KindHostHealth,
		[]burninv1alpha1.Threshold{th("xid_events", burninv1alpha1.EQ, "0")})
	if len(ps) != 1 {
		t.Fatalf("got %d problems for one malformed name, want 1: %v", len(ps), ps)
	}
	if ps[0].Severity != SeverityMalformed {
		t.Errorf("severity = %q, want %q", ps[0].Severity, SeverityMalformed)
	}
}

// The gates the shipped samples are built from must stay silent under the
// stricter, kind-aware form too, or the rule is one the project does not itself
// keep.
func TestValidateThresholdsForKind_AcceptsFirstPartyGatesOnRegisteredMetrics(t *testing.T) {
	for _, tc := range []struct {
		kind      burninv1alpha1.TestKind
		threshold burninv1alpha1.Threshold
	}{
		{burninv1alpha1.KindHostHealth, th("eccErrors", burninv1alpha1.EQ, "0")},
		{burninv1alpha1.KindHostHealth, th("xidEvents", burninv1alpha1.EQ, "0")},
		{burninv1alpha1.KindClockProbe, th("sustainedClockPct", burninv1alpha1.GTE, "70")},
		{burninv1alpha1.KindNCCL, th("busBandwidthGBs", burninv1alpha1.GTE, "10.8")},
		{burninv1alpha1.KindIBWriteBW, th("bandwidthGbps", burninv1alpha1.GTE, "89")},
		{burninv1alpha1.KindMemoryStress, th("memoryErrors", burninv1alpha1.EQ, "0")},
	} {
		t.Run(string(tc.kind)+"/"+tc.threshold.Metric, func(t *testing.T) {
			if got := ValidateThresholdsForKind(tc.kind, []burninv1alpha1.Threshold{tc.threshold}); len(got) != 0 {
				t.Errorf("a gate this project ships was flagged: %v", got)
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

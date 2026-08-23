package verdict

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// ─── Equal / NotEqual are EXACT, and that is the pinned semantics ─────────────

// The case that motivates the whole rule. 83.22 does not survive the trip
// through a float64 average of sampled clocks, so an exact gate on a continuous
// metric is a permanent failure wearing the costume of a hardware verdict.
func TestEvaluate_EqualIsExactAndTrapsAContinuousMetric(t *testing.T) {
	thresholds := []contract.Threshold{th("sustainedClockPct", contract.EQ, "83.22")}

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
	thresholds := []contract.Threshold{th("eccErrors", contract.EQ, "0")}

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
		[]contract.Threshold{th("xidEvents", contract.EQ, "9007199254740992")}); !got.Passed {
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
		thresholds []contract.Threshold
		wantIn     string
	}{
		{
			name:       "NaN metric under NotEqual",
			metrics:    map[string]string{"eccErrors": "NaN"},
			thresholds: []contract.Threshold{th("eccErrors", contract.NEQ, "1")},
			wantIn:     "not a finite measurement",
		},
		{
			name:       "Inf metric under a ceiling",
			metrics:    map[string]string{"gpuTempC": "+Inf"},
			thresholds: []contract.Threshold{th("gpuTempC", contract.LTE, "85")},
			wantIn:     "not a finite measurement",
		},
		{
			name:       "NaN threshold value under NotEqual",
			metrics:    map[string]string{"eccErrors": "0"},
			thresholds: []contract.Threshold{th("eccErrors", contract.NEQ, "NaN")},
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

func problemFor(t *testing.T, threshold contract.Threshold) []Problem {
	t.Helper()
	return ValidateThresholds([]contract.Threshold{threshold})
}

func onlyProblem(t *testing.T, threshold contract.Threshold) Problem {
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
	sound := []contract.Threshold{
		th("eccErrors", contract.EQ, "0"),
		th("throttleEvents", contract.EQ, "0"),
		th("miscompares", contract.EQ, "0"),
		th("iterationsCompleted", contract.NEQ, "0"),
		th("busBandwidthGBs", contract.GTE, "20"),
		th("gpuTempC", contract.LTE, "85"),
		th("sustainedClockPct", contract.GTE, "90"),  // the correct form of the trap below
		th("sustainedClockPct", contract.LTE, "105"), // …and its other side
		// An unregistered metric from a third-party runner: the registry is an
		// open world, and this package does not know better than its author.
		th("vendorFabricRetries", contract.EQ, "0"),
		th("vendorLinkUtilPct", contract.GTE, "80"),
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
	for _, op := range []contract.Comparison{contract.EQ, contract.NEQ} {
		t.Run(string(op), func(t *testing.T) {
			p := onlyProblem(t, th("sustainedClockPct", op, "83.22"))
			if p.Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q", p.Severity, SeverityUnsound)
			}
			if p.Metric != "sustainedClockPct" || p.Index != 0 {
				t.Errorf("problem = %+v, want it to point at threshold 0 / sustainedClockPct", p)
			}
			// The message has to name the fix, or it is just a complaint.
			for _, want := range []string{"Pct", string(contract.GTE), string(contract.LTE)} {
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
			p := onlyProblem(t, th(metric, contract.EQ, "100"))
			if p.Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q", p.Severity, SeverityUnsound)
			}
		})
	}
}

// A dimensionless name does not make a fractional exact gate sensible: a counter
// counts, and `Equal 0.5` can never hold either.
func TestValidateThresholds_FlagsExactComparisonAgainstAFractionalValue(t *testing.T) {
	p := onlyProblem(t, th("miscompares", contract.EQ, "0.5"))
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
			ps := problemFor(t, th(metric, contract.GTE, "90"))
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

// The regression test for the gap that made issue #65 a bug rather than an
// untidiness: a gate on a metric whose value is a LABEL passed every check this
// linter makes, and then failed closed on every node forever.
//
// All three checks genuinely pass on these names — the grammar is valid
// lowerCamelCase, UnitOf answers UnitNone (which is the legitimate
// dimensionless-counter case, so exact comparison is not flagged), and
// SafeToThresholdOn answers true for anything unregistered because the registry
// is an open world. Nothing was left to notice that "true" is not a number.
//
// The fix is entirely in pkg/contract: a first-party runner's label-valued
// metrics are now registered as Evidence, which is what turns SafeToThresholdOn
// to false and makes the existing evidence check below fire. This test asserts
// the OUTCOME rather than the mechanism, so it keeps holding if the mechanism
// is later sharpened (see #75).
func TestValidateThresholds_FlagsAGateOnALabelValuedMetric(t *testing.T) {
	// The two a profile author is most likely to reach for after reading
	// clockprobe's README — they are what the runner presents as its wedge
	// verdict — plus the rest of the label-valued set across both runners.
	labelled := map[string]string{
		"pdWedgeSuspected":       "true",
		"throttleClassification": "powerCap",
		"clockFloorBasis":        "thermal",
		"throttleReasons":        "swPowerCap,hwPowerBrake",
		"nvmlUnsupported":        "enforcedPowerLimit,defaultPowerLimit",
		"configWarnings":         "NVML handle resolved by index, not PCI bus id",
		"gpuName":                "NVIDIA GB10",
		"computeCap":             "12.1",
		"pciBusId":               "0000:01:00.0",
		"driverVersion":          "580.82.09",
		"builtCudaArch":          "sm_121a",
		"migMode":                "disabled",
	}

	for metric, reported := range labelled {
		t.Run(metric, func(t *testing.T) {
			// "== 1" is the shape an author reaches for on something that reads
			// like a boolean, and it is exactly what #65 reproduced.
			gate := th(metric, contract.EQ, "1")

			ps := problemFor(t, gate)
			if len(ps) == 0 {
				t.Fatalf("ValidateThresholds reported nothing for a gate on %q, whose value is %q; "+
					"this gate fails closed on every node forever and the failure reads as a hardware verdict",
					metric, reported)
			}
			if !strings.Contains(ps[0].Reason, metric) {
				t.Errorf("reason = %q, want it to name the metric", ps[0].Reason)
			}

			// The other half of the finding, and the reason it must be caught at
			// authoring time: evaluation is unchanged and still fails closed.
			// The linter is advisory — it does not rescue this gate, it warns
			// while the author can still fix it.
			out := Evaluate(map[string]string{metric: reported}, nil, []contract.Threshold{gate})
			if out.Passed {
				t.Fatalf("Evaluate passed a gate on the non-numeric value %q; it must fail closed", reported)
			}

			// Registering these UPGRADED the finding rather than replacing one
			// with another of equal worth, and that is the point of registering
			// rather than leaving them to the unregistered rule. Before, the
			// most that could be said on a built-in kind was "this project has
			// never heard of that name"; now the reason is the registry's own
			// description, which tells the author the values the metric
			// actually takes and what to gate on instead. Asserted so a future
			// change cannot quietly demote it back.
			aware := ValidateThresholdsForKind(contract.KindClockProbe,
				[]contract.Threshold{gate})
			if len(aware) == 0 {
				t.Fatalf("the kind-aware form said nothing about a gate on %q", metric)
			}
			if strings.Contains(aware[0].Reason, "not in the metric registry") {
				t.Errorf("reason = %q; %q is registered now, so the author should be told what its "+
					"values ARE, not that we have never heard of it", aware[0].Reason, metric)
			}
			if !strings.Contains(aware[0].Reason, "evidence") {
				t.Errorf("reason = %q, want the evidence finding", aware[0].Reason)
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
		threshold contract.Threshold
		wantIn    string
	}{
		{
			name:      "ungrammatical metric name is dropped by the parser",
			threshold: th("bus_bandwidth_gbs", contract.GTE, "20"),
			wantIn:    "can never be satisfied",
		},
		{
			name:      "empty metric name",
			threshold: th("", contract.GTE, "20"),
			wantIn:    "must not be empty",
		},
		{
			name:      "non-numeric value",
			threshold: th("busBandwidthGBs", contract.GTE, "fast"),
			wantIn:    "not numeric",
		},
		{
			name:      "non-finite value",
			threshold: th("eccErrors", contract.NEQ, "NaN"),
			wantIn:    "not a finite number",
		},
		{
			name:      "unrecognised comparison",
			threshold: th("eccErrors", contract.Comparison("EqualIsh"), "0"),
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

// The #65 instance, caught — now with the examples it applies to NEXT.
//
// This rule was written against pdWedgeSuspected, throttleClassification and
// clockFloorBasis, which clockprobe emits as words and which nothing had
// registered, so a gate on one was never numeric and failed closed on every node
// forever while passing every check this package made. Those three are now
// REGISTERED (as Evidence, which is #65's fix), so they no longer exercise this
// rule — they exercise the evidence rule instead, which is asserted separately
// below. Substituting them silently would have left this test passing while
// checking nothing it was written for.
//
// The names here are their successors, and they are not hypothetical: clockprobe
// emits nine per-reason sample counters that pkg/contract deliberately does NOT
// register, because they are the vendor's own enum, they scale with the sample
// interval, and throttledSamples already carries what acceptance needs. Being
// integers, a gate on one evaluates and means what it says — so this is not the
// fails-closed trap, it is the other reading in the message: a first-party name
// somebody is now gating on, and gating is what promotes a name from incidental
// evidence to acceptance-deciding. Register it at that point, or accept that
// nothing guarantees a runner still emits it.
func TestValidateThresholdsForKind_FlagsAnUnregisteredMetricOnABuiltInKind(t *testing.T) {
	for _, metric := range []string{"hwSlowdownSamples", "swPowerCapSamples", "gpuIdleSamples"} {
		t.Run(metric, func(t *testing.T) {
			ps := ValidateThresholdsForKind(contract.KindClockProbe,
				[]contract.Threshold{th(metric, contract.EQ, "0")})
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
	thresholds := []contract.Threshold{
		th("vendorFabricRetries", contract.EQ, "0"),
		th("vendorLinkUtilPct", contract.GTE, "80"),
		// A first-party name that is unregistered on purpose. It belongs here
		// rather than only among the vendor-shaped ones: the rule keys off the
		// KIND, not off whether a name looks like ours, and a third-party runner
		// reusing a spelling we happen to also emit must still go unremarked.
		th("hwSlowdownSamples", contract.EQ, "0"),
	}
	for _, kind := range []contract.TestKind{
		contract.KindCustom,
		contract.TestKind("some-vendors-own-runner"),
		contract.TestKind(""),
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
	thresholds := []contract.Threshold{
		th("hwSlowdownSamples", contract.EQ, "0"),
		th("gpuTempC", contract.LTE, "85"),
	}
	if got := ValidateThresholds(thresholds); len(got) != 0 {
		t.Errorf("the kind-agnostic form gained an opinion about an unregistered name: %v", got)
	}
	// The registered metrics this project owns are judged identically either
	// way — the kind changes exactly one check and nothing else.
	registered := []contract.Threshold{
		th("sustainedClockPct", contract.EQ, "83.22"),
		th("throughputTflops", contract.GTE, "90"),
	}
	agnostic := ValidateThresholds(registered)
	aware := ValidateThresholdsForKind(contract.KindClockProbe, registered)
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
	ps := ValidateThresholdsForKind(contract.KindHostHealth,
		[]contract.Threshold{th("xid_events", contract.EQ, "0")})
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
		kind      contract.TestKind
		threshold contract.Threshold
	}{
		{contract.KindHostHealth, th("eccErrors", contract.EQ, "0")},
		{contract.KindHostHealth, th("xidEvents", contract.EQ, "0")},
		{contract.KindClockProbe, th("sustainedClockPct", contract.GTE, "70")},
		{contract.KindNCCL, th("busBandwidthGBs", contract.GTE, "10.8")},
		{contract.KindIBWriteBW, th("bandwidthGbps", contract.GTE, "89")},
		{contract.KindMemoryStress, th("memoryErrors", contract.EQ, "0")},
	} {
		t.Run(string(tc.kind)+"/"+tc.threshold.Metric, func(t *testing.T) {
			if got := ValidateThresholdsForKind(tc.kind, []contract.Threshold{tc.threshold}); len(got) != 0 {
				t.Errorf("a gate this project ships was flagged: %v", got)
			}
		})
	}
}

// An author fixing a profile should see the whole list, and each problem must
// point at the threshold it came from.
func TestValidateThresholds_ReportsEveryProblemWithItsIndex(t *testing.T) {
	got := ValidateThresholds([]contract.Threshold{
		th("busBandwidthGBs", contract.GTE, "20"),       // fine
		th("gpuTempC", contract.EQ, "85"),               // unsound: exact on continuous
		th("throughputTflops", contract.GTE, "90"),      // unsound: evidence only
		th("eccErrors", contract.Comparison("~="), "0"), // malformed: unknown comparison
		th("latencyUs", contract.EQ, "not-a-number"),    // malformed value AND exact on continuous
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

// A spread is n/a on every single-device node — a positive claim, "nothing to
// spread across" — and the DEFAULT applicability is Required, which fails
// closed on n/a. So a fleet profile that gates a spread and forgets
// RequiredIfMeasurable fails every healthy single-GPU node forever, on the one
// phase that is never retried, and the failure reads as a hardware verdict.
// This is the authoring-time surface for that: advice, not a refusal, because
// the gate does evaluate — on an eight-GPU node it is exactly the right gate.
func TestValidateThresholds_FlagsARequiredGateOnASpread(t *testing.T) {
	for _, metric := range contract.SpreadMetrics {
		t.Run(metric, func(t *testing.T) {
			// Default applicability (empty = Required).
			ps := problemFor(t, th(metric, contract.LTE, "10"))
			if len(ps) != 1 {
				t.Fatalf("got %d problems, want 1: %v", len(ps), ps)
			}
			if ps[0].Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q — the gate evaluates, so this is advice", ps[0].Severity, SeverityUnsound)
			}
			if !strings.Contains(ps[0].Reason, "RequiredIfMeasurable") || !strings.Contains(ps[0].Reason, "single-device") {
				t.Errorf("reason = %q, want it to name RequiredIfMeasurable and the single-device node it would fail", ps[0].Reason)
			}
			// Explicit Required says the same thing.
			explicit := th(metric, contract.LTE, "10")
			explicit.Applicability = contract.Required
			if ps := problemFor(t, explicit); len(ps) != 1 {
				t.Errorf("explicit Required: got %d problems, want 1: %v", len(ps), ps)
			}
			// RequiredIfMeasurable is the gate the spreads are for: clean.
			ok := th(metric, contract.LTE, "10")
			ok.Applicability = contract.RequiredIfMeasurable
			if ps := problemFor(t, ok); len(ps) != 0 {
				t.Errorf("RequiredIfMeasurable gate on %s reported %v, want nothing", metric, ps)
			}
		})
	}
	// And the check is about spreads, not about every RequiredIfMeasurable-able
	// metric: a Required floor on a plain acceptance metric is fine.
	if ps := problemFor(t, th("sustainedClockPct", contract.GTE, "80")); len(ps) != 0 {
		t.Errorf("a Required floor on sustainedClockPct reported %v, want nothing", ps)
	}
}

// #405: the same trap exists for the peer-bandwidth matrices, which are n/a
// on every single-device node for the same reason a spread is — no peer path
// with one accelerator — even though neither is named "spread". The linter
// reads Metric.MayBeUnmeasurable rather than IsSpreadMetric specifically so
// this is not a special case.
func TestValidateThresholds_FlagsARequiredGateOnAPeerBandwidthMatrix(t *testing.T) {
	for _, metric := range []string{"peerReadBandwidthGBs", "peerWriteBandwidthGBs"} {
		t.Run(metric, func(t *testing.T) {
			ps := problemFor(t, th(metric, contract.GTE, "100"))
			if len(ps) != 1 {
				t.Fatalf("got %d problems, want 1: %v", len(ps), ps)
			}
			if ps[0].Severity != SeverityUnsound {
				t.Errorf("severity = %q, want %q", ps[0].Severity, SeverityUnsound)
			}
			if !strings.Contains(ps[0].Reason, "RequiredIfMeasurable") || !strings.Contains(ps[0].Reason, "single-device") {
				t.Errorf("reason = %q, want it to name RequiredIfMeasurable and the single-device node it would fail", ps[0].Reason)
			}
			ok := th(metric, contract.GTE, "100")
			ok.Applicability = contract.RequiredIfMeasurable
			if ps := problemFor(t, ok); len(ps) != 0 {
				t.Errorf("RequiredIfMeasurable gate on %s reported %v, want nothing", metric, ps)
			}
		})
	}
}

// #405's own caution, made concrete: eccErrors and remappedRows are ALSO n/a
// on some hardware (a part with no ECC subsystem, e.g. GB10's unified
// LPDDR5X), but not on every single-device node the way a spread or a peer
// bandwidth is — a fleet whose parts all have ECC legitimately writes
// `eccErrors Equal 0` with the default Required applicability, and that gate
// is exactly right for them. Flagging it here would be noise on the common
// case rather than advice about a real trap, so these two must NOT gain
// MayBeUnmeasurable even though they share the "sometimes n/a" shape.
func TestValidateThresholds_DoesNotFlagECCOrRemappedRowsAsUnmeasurable(t *testing.T) {
	for _, metric := range []string{"eccErrors", "remappedRows"} {
		t.Run(metric, func(t *testing.T) {
			if contract.MayBeUnmeasurable(metric) {
				t.Errorf("%s.MayBeUnmeasurable = true, want false — it is n/a only on some SKUs, "+
					"not on every single-device node", metric)
			}
			if ps := problemFor(t, th(metric, contract.EQ, "0")); len(ps) != 0 {
				t.Errorf("a Required gate on %s reported %v, want nothing — a fleet with real ECC "+
					"legitimately writes this gate", metric, ps)
			}
		})
	}
}

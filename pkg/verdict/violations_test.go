package verdict

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// ─── The compatibility proof ─────────────────────────────────────────────────
//
// Evaluate now evaluates every threshold, but Message and NotEvaluated are
// FROZEN at what stopping-on-the-first-violation produced. verdict_test.go
// asserts with strings.Contains, which is too weak to catch a reworded message;
// this file pins the strings exactly, by differential comparison against a
// reference implementation of the old control flow.
//
// evaluateFrozenReference is deliberately a copy of the pre-GEP-0561 Evaluate.
// It is not DRY and must not be refactored to share code with the real one —
// its whole value is being an INDEPENDENT statement of the frozen semantics. If
// a change to Evaluate makes this disagree, the change broke the contract for
// both dispatchers; that is the alarm, not a stale test.
func evaluateFrozenReference(
	metrics map[string]string,
	unmeasurable map[string]bool,
	thresholds []burninv1alpha1.Threshold,
) (passed bool, message string, notEvaluated []NotEvaluated) {
	fail := func(format string, args ...any) (bool, string, []NotEvaluated) {
		return false, fmt.Sprintf(format, args...), notEvaluated
	}

	for _, th := range thresholds {
		raw, reported := metrics[th.Metric]

		if unmeasurable[th.Metric] {
			if reported {
				return fail("metric %q was reported as %q AND declared unmeasurable; the runner's output is self-contradictory", th.Metric, raw)
			}
			if th.Applicability == burninv1alpha1.RequiredIfMeasurable {
				notEvaluated = append(notEvaluated, NotEvaluated{
					Metric: th.Metric,
					Reason: ReasonUnmeasurable,
				})
				continue
			}
			return fail("metric %q is %s and this threshold is %s; "+
				"set applicability=RequiredIfMeasurable if the gate should be skipped where the hardware cannot measure it",
				th.Metric, ReasonUnmeasurable, applicabilityOf(th))
		}

		if !reported {
			return fail("metric %q was not reported by the runner", th.Metric)
		}
		got, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return fail("metric %q=%q is not numeric", th.Metric, raw)
		}
		if math.IsNaN(got) || math.IsInf(got, 0) {
			return fail("metric %q=%q is not a finite measurement", th.Metric, raw)
		}
		want, err := strconv.ParseFloat(th.Value, 64)
		if err != nil {
			return fail("threshold value %q for %q is not numeric", th.Value, th.Metric)
		}
		if math.IsNaN(want) || math.IsInf(want, 0) {
			return fail("threshold value %q for %q is not a finite number", th.Value, th.Metric)
		}
		if !compare(got, th.Comparison, want) {
			return fail("%s: got %g, need %s %g", th.Metric, got, th.Comparison, want)
		}
	}
	return true, "", notEvaluated
}

// corpusMetrics and corpusUnmeasurable make each specimen below deterministic.
// contradictoryCount appears in BOTH, which is the self-contradictory case.
var (
	corpusMetrics = map[string]string{
		"goodMetricGBs":      "50",
		"busBandwidthGBs":    "12.0",
		"textMetricGBs":      "fast",
		"nanMetricGBs":       "NaN",
		"contradictoryCount": "0",
	}
	corpusUnmeasurable = map[string]bool{
		"eccErrors":          true,
		"remappedRows":       true,
		"contradictoryCount": true,
	}
)

// corpusThresholds covers every route through Evaluate: one satisfied, all
// eight violation kinds, and both not-evaluated gates.
var corpusThresholds = []burninv1alpha1.Threshold{
	th("goodMetricGBs", burninv1alpha1.GTE, "20"),        // satisfied
	th("busBandwidthGBs", burninv1alpha1.GTE, "20"),      // Unsatisfied
	th("missingMetricGBs", burninv1alpha1.GTE, "20"),     // NotReported
	th("textMetricGBs", burninv1alpha1.GTE, "20"),        // NonNumeric
	th("nanMetricGBs", burninv1alpha1.GTE, "20"),         // NonFinite
	th("contradictoryCount", burninv1alpha1.EQ, "0"),     // Contradictory
	th("eccErrors", burninv1alpha1.EQ, "0"),              // UnmeasurableRequired
	th("goodMetricGBs", burninv1alpha1.GTE, "twenty"),    // ThresholdValueNonNumeric
	th("goodMetricGBs", burninv1alpha1.GTE, "NaN"),       // ThresholdValueNonFinite
	ifMeasurable("eccErrors", burninv1alpha1.EQ, "0"),    // NotEvaluated
	ifMeasurable("remappedRows", burninv1alpha1.EQ, "0"), // NotEvaluated
}

// Every ordered triple drawn from the corpus — 11^3 = 1331 profiles, which is
// enough to put a violation before, after, and between the not-evaluated gates
// in every combination that matters for the truncation.
func TestEvaluate_FrozenFieldsMatchFirstViolationSemantics(t *testing.T) {
	n := len(corpusThresholds)
	checked := 0

	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			for c := 0; c < n; c++ {
				thresholds := []burninv1alpha1.Threshold{
					corpusThresholds[a], corpusThresholds[b], corpusThresholds[c],
				}
				got := Evaluate(corpusMetrics, corpusUnmeasurable, thresholds)
				wantPassed, wantMessage, wantNotEvaluated := evaluateFrozenReference(
					corpusMetrics, corpusUnmeasurable, thresholds)

				if got.Passed != wantPassed {
					t.Fatalf("[%d,%d,%d] Passed = %v, want %v", a, b, c, got.Passed, wantPassed)
				}
				if got.Message != wantMessage {
					t.Fatalf("[%d,%d,%d] Message =\n  %q\nwant\n  %q", a, b, c, got.Message, wantMessage)
				}
				if len(got.NotEvaluated) != len(wantNotEvaluated) ||
					(len(wantNotEvaluated) > 0 && !reflect.DeepEqual(got.NotEvaluated, wantNotEvaluated)) {
					t.Fatalf("[%d,%d,%d] NotEvaluated = %+v, want %+v", a, b, c, got.NotEvaluated, wantNotEvaluated)
				}
				checked++
			}
		}
	}
	if checked != n*n*n {
		t.Fatalf("checked %d profiles, want %d", checked, n*n*n)
	}
}

// Passed and Violations are two views of one fact; they must never disagree.
func TestEvaluate_ViolationsEmptyExactlyWhenPassed(t *testing.T) {
	n := len(corpusThresholds)
	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			thresholds := []burninv1alpha1.Threshold{corpusThresholds[a], corpusThresholds[b]}
			got := Evaluate(corpusMetrics, corpusUnmeasurable, thresholds)

			if got.Passed != (len(got.Violations) == 0) {
				t.Fatalf("[%d,%d] Passed = %v with %d violations", a, b, got.Passed, len(got.Violations))
			}
			// Message is the first violation's reason, always.
			if !got.Passed && got.Message != got.Violations[0].Reason {
				t.Fatalf("[%d,%d] Message = %q, want Violations[0].Reason = %q", a, b, got.Message, got.Violations[0].Reason)
			}
			if got.Passed && len(got.Violations) != 0 {
				t.Fatalf("[%d,%d] a passing outcome carried violations", a, b)
			}
		}
	}
}

// ─── What the new field buys ─────────────────────────────────────────────────

// The motivating case: a node that misses three gates reports three, not one.
func TestEvaluate_ReportsEveryViolationNotJustTheFirst(t *testing.T) {
	got := Evaluate(
		map[string]string{
			"busBandwidthGBs": "12.0",
			"gpuTempC":        "94",
			"eccErrors":       "3",
		},
		nil,
		[]burninv1alpha1.Threshold{
			th("busBandwidthGBs", burninv1alpha1.GTE, "20"),
			th("gpuTempC", burninv1alpha1.LTE, "85"),
			th("eccErrors", burninv1alpha1.EQ, "0"),
		},
	)

	if got.Passed {
		t.Fatal("three violated gates passed")
	}
	if len(got.Violations) != 3 {
		t.Fatalf("Violations = %+v, want all three", got.Violations)
	}
	// Spec order, and each one indexed so a report can point at the entry.
	for i, want := range []string{"busBandwidthGBs", "gpuTempC", "eccErrors"} {
		if got.Violations[i].Metric != want {
			t.Errorf("Violations[%d].Metric = %q, want %q", i, got.Violations[i].Metric, want)
		}
		if got.Violations[i].Index != i {
			t.Errorf("Violations[%d].Index = %d, want %d", i, got.Violations[i].Index, i)
		}
		if got.Violations[i].Cause != CauseMeasurement {
			t.Errorf("%s: Cause = %q, want %q — a real measurement missing its bar is about the hardware",
				want, got.Violations[i].Cause, CauseMeasurement)
		}
	}
	// And the frozen field still names only the first.
	if got.Message != got.Violations[0].Reason {
		t.Errorf("Message = %q, want the first violation only", got.Message)
	}
}

// The separation that makes enumeration safe: a broken profile must not read as
// a hardware verdict just because it arrived in the same list.
func TestEvaluate_CauseSeparatesHardwareFromProfileFromRunner(t *testing.T) {
	got := Evaluate(
		map[string]string{"busBandwidthGBs": "12.0", "goodMetricGBs": "50"},
		nil,
		[]burninv1alpha1.Threshold{
			th("busBandwidthGBs", burninv1alpha1.GTE, "20"),   // hardware fell short
			th("goodMetricGBs", burninv1alpha1.GTE, "twenty"), // the profile is broken
			th("absentMetricGBs", burninv1alpha1.GTE, "20"),   // the runner told us nothing
		},
	)

	if len(got.Violations) != 3 {
		t.Fatalf("Violations = %+v, want three", got.Violations)
	}
	for i, want := range []Cause{CauseMeasurement, CauseAuthoring, CauseEvidence} {
		if got.Violations[i].Cause != want {
			t.Errorf("Violations[%d] (%s): Cause = %q, want %q",
				i, got.Violations[i].Metric, got.Violations[i].Cause, want)
		}
	}

	// Exactly one of these implicates the hardware. That count is the whole
	// point of the taxonomy: two of the three must not send anyone to a node.
	hardware := 0
	for _, v := range got.Violations {
		if v.Cause == CauseMeasurement {
			hardware++
		}
	}
	if hardware != 1 {
		t.Errorf("%d violations implicate the hardware, want exactly 1", hardware)
	}
}

// Every route lands on its documented kind, and every kind on its documented
// cause. This is the table in GEP-0561, executable.
func TestEvaluate_ViolationKindAndCausePerRoute(t *testing.T) {
	cases := []struct {
		name      string
		threshold burninv1alpha1.Threshold
		wantKind  ViolationKind
		wantCause Cause
	}{
		{"comparison failed", th("busBandwidthGBs", burninv1alpha1.GTE, "20"), KindUnsatisfied, CauseMeasurement},
		{"absent", th("missingMetricGBs", burninv1alpha1.GTE, "20"), KindNotReported, CauseEvidence},
		{"non-numeric metric", th("textMetricGBs", burninv1alpha1.GTE, "20"), KindNonNumeric, CauseEvidence},
		{"non-finite metric", th("nanMetricGBs", burninv1alpha1.GTE, "20"), KindNonFinite, CauseEvidence},
		{"contradictory", th("contradictoryCount", burninv1alpha1.EQ, "0"), KindContradictory, CauseEvidence},
		{"unmeasurable under Required", th("eccErrors", burninv1alpha1.EQ, "0"), KindUnmeasurableRequired, CauseEvidence},
		{"threshold value non-numeric", th("goodMetricGBs", burninv1alpha1.GTE, "twenty"), KindThresholdValueNonNumeric, CauseAuthoring},
		{"threshold value non-finite", th("goodMetricGBs", burninv1alpha1.GTE, "NaN"), KindThresholdValueNonFinite, CauseAuthoring},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Evaluate(corpusMetrics, corpusUnmeasurable, []burninv1alpha1.Threshold{c.threshold})
			if len(got.Violations) != 1 {
				t.Fatalf("Violations = %+v, want exactly one", got.Violations)
			}
			v := got.Violations[0]
			if v.Kind != c.wantKind {
				t.Errorf("Kind = %q, want %q", v.Kind, c.wantKind)
			}
			if v.Cause != c.wantCause {
				t.Errorf("Cause = %q, want %q", v.Cause, c.wantCause)
			}
			if v.Cause != v.Kind.Cause() {
				t.Errorf("Cause %q disagrees with Kind.Cause() %q", v.Cause, v.Kind.Cause())
			}
			if v.Reason == "" {
				t.Error("a violation must carry an explanatory reason")
			}
		})
	}
}

// One threshold yields at most one violation — it is not re-judged by later
// rules once it has failed one.
func TestEvaluate_OneThresholdYieldsAtMostOneViolation(t *testing.T) {
	for i, th := range corpusThresholds {
		got := Evaluate(corpusMetrics, corpusUnmeasurable, []burninv1alpha1.Threshold{th})
		if len(got.Violations) > 1 {
			t.Errorf("threshold %d (%s) produced %d violations: %+v", i, th.Metric, len(got.Violations), got.Violations)
		}
	}
}

// The documented, deliberate incompleteness: a RequiredIfMeasurable gate that
// sits AFTER the first violation is absent from the frozen NotEvaluated, and
// present in neither Violations (it was not violated) — so the two fields
// together are what a complete report reads. This test exists so the freeze is
// a recorded decision rather than a surprise. See GEP-0561.
func TestEvaluate_NotEvaluatedStaysTruncatedAtTheFirstViolation(t *testing.T) {
	got := Evaluate(
		map[string]string{"xidEvents": "7"},
		unmeasured("eccErrors", "remappedRows"),
		[]burninv1alpha1.Threshold{
			ifMeasurable("eccErrors", burninv1alpha1.EQ, "0"),    // before the violation
			th("xidEvents", burninv1alpha1.EQ, "0"),              // the violation
			ifMeasurable("remappedRows", burninv1alpha1.EQ, "0"), // after it
		},
	)

	if got.Passed {
		t.Fatal("a violated threshold passed")
	}
	if len(got.NotEvaluated) != 1 || got.NotEvaluated[0].Metric != "eccErrors" {
		t.Fatalf("NotEvaluated = %+v, want only the gate found before the violation", got.NotEvaluated)
	}
	// remappedRows went un-applied too, and is deliberately reported nowhere.
	for _, ne := range got.NotEvaluated {
		if ne.Metric == "remappedRows" {
			t.Error("NotEvaluated widened past the first violation; it is frozen truncated")
		}
	}
	for _, v := range got.Violations {
		if v.Metric == "remappedRows" {
			t.Error("an un-applied RequiredIfMeasurable gate was recorded as a violation")
		}
	}
	// Violations, meanwhile, is complete: the one real violation.
	if len(got.Violations) != 1 || got.Violations[0].Metric != "xidEvents" {
		t.Fatalf("Violations = %+v, want just xidEvents", got.Violations)
	}
}

// An unrecognised kind must not claim the hardware is at fault.
func TestViolationKind_UnknownIsEvidenceNotMeasurement(t *testing.T) {
	if got := ViolationKind("SomethingNew").Cause(); got != CauseEvidence {
		t.Errorf("unknown kind classified as %q, want %q — an unknown route has not established anything about the part", got, CauseEvidence)
	}
}

// ─── ViolationSummary ────────────────────────────────────────────────────────

func TestOutcome_ViolationSummaryIsEmptyBelowTwoViolations(t *testing.T) {
	// Nothing to summarise: Message already says all of it.
	none := Evaluate(map[string]string{"goodMetricGBs": "50"}, nil,
		[]burninv1alpha1.Threshold{th("goodMetricGBs", burninv1alpha1.GTE, "20")})
	if s := none.ViolationSummary(); s != "" {
		t.Errorf("a passing outcome summarised %q, want empty", s)
	}

	one := Evaluate(map[string]string{"busBandwidthGBs": "12"}, nil,
		[]burninv1alpha1.Threshold{th("busBandwidthGBs", burninv1alpha1.GTE, "20")})
	if s := one.ViolationSummary(); s != "" {
		t.Errorf("a single violation summarised %q, want empty — Message already names it", s)
	}
}

// The summary names the CAUSE of each further violation, because that is what
// decides who acts on it.
func TestOutcome_ViolationSummaryNamesCauses(t *testing.T) {
	got := Evaluate(
		map[string]string{"busBandwidthGBs": "12", "goodMetricGBs": "50"},
		nil,
		[]burninv1alpha1.Threshold{
			th("busBandwidthGBs", burninv1alpha1.GTE, "20"),   // first — in Message
			th("gpuTempC", burninv1alpha1.LTE, "85"),          // Evidence: not reported
			th("goodMetricGBs", burninv1alpha1.GTE, "twenty"), // Authoring
		},
	)
	s := got.ViolationSummary()

	for _, want := range []string{"2 more gates failed", "gpuTempC", "not measured", "goodMetricGBs", "profile error"} {
		if !strings.Contains(s, want) {
			t.Errorf("summary = %q, want it to contain %q", s, want)
		}
	}
	// The first violation is Message's job; repeating it here would double-count.
	if strings.Contains(s, "busBandwidthGBs") {
		t.Errorf("summary = %q, want it to cover only the violations beyond the first", s)
	}
}

// Singular/plural, because a report that says "1 more gates" reads as a bug in
// the operator and invites doubt about the number beside it.
func TestOutcome_ViolationSummaryAgreesInNumber(t *testing.T) {
	got := Evaluate(
		map[string]string{"busBandwidthGBs": "12"},
		nil,
		[]burninv1alpha1.Threshold{
			th("busBandwidthGBs", burninv1alpha1.GTE, "20"),
			th("gpuTempC", burninv1alpha1.LTE, "85"),
		},
	)
	if s := got.ViolationSummary(); !strings.Contains(s, "1 more gate failed") {
		t.Errorf("summary = %q, want the singular form", s)
	}
}

// A profile that gates forty metrics must not produce a forty-item sentence in
// a status field — but the COUNT stays exact even when the names are elided.
func TestOutcome_ViolationSummaryBoundsTheNamesNotTheCount(t *testing.T) {
	metrics := map[string]string{}
	var thresholds []burninv1alpha1.Threshold
	for i := 0; i < 12; i++ {
		name := fmt.Sprintf("metric%dGBs", i)
		metrics[name] = "1"
		thresholds = append(thresholds, th(name, burninv1alpha1.GTE, "20"))
	}
	got := Evaluate(metrics, nil, thresholds)
	if len(got.Violations) != 12 {
		t.Fatalf("Violations = %d, want all 12 recorded", len(got.Violations))
	}

	s := got.ViolationSummary()
	if !strings.Contains(s, "11 more gates failed") {
		t.Errorf("summary = %q, want the exact count of the remainder", s)
	}
	if !strings.Contains(s, "and 7 more") {
		t.Errorf("summary = %q, want the elided remainder counted (11 - %d named)", s, maxNamedViolations)
	}
	if named := strings.Count(s, "(hardware)"); named != maxNamedViolations {
		t.Errorf("summary named %d metrics, want %d", named, maxNamedViolations)
	}
}

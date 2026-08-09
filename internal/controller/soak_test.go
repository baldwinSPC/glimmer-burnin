package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/sink"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// ─── Fixtures ─────────────────────────────────────────────────────────────────

// soakTest is a thermal soak divided into segments. thermal-soak is used
// throughout this file rather than compute-smoke because it is a kind that
// actually honours a duration — segmenting a burst kind is refused at plan time,
// and TestSoak_PlanRefusals is where that is asserted.
func soakTest(name string, duration, segment int32, thresholds ...burninv1alpha1.Threshold) *burninv1alpha1.BurnInTest {
	return &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:            burninv1alpha1.KindThermalSoak,
			Scope:           burninv1alpha1.ScopeNode,
			DurationSeconds: duration,
			Soak:            &burninv1alpha1.SoakSpec{SegmentSeconds: segment},
			Thresholds:      thresholds,
		},
	}
}

func withAbortEarly(bt *burninv1alpha1.BurnInTest) *burninv1alpha1.BurnInTest {
	bt.Spec.Soak.AbortEarly = boolp(true)
	return bt
}

// thermalSegmentStdout is one window of a soak in the shape the shipped runner
// prints it, with the keys the thermal-soak alias table maps
// (soak_seconds → elapsedS, peak_temp_c → gpuTempC).
func thermalSegmentStdout(elapsedS, peakTempC, sustainedPct, xidEvents string) string {
	return strings.Join([]string{
		"gpu_name=NVIDIA GB10",
		"soak_seconds=" + elapsedS,
		"peak_temp_c=" + peakTempC,
		"sustained_clock_pct=" + sustainedPct,
		"xid_events=" + xidEvents,
		"throttle_count=0",
		"THERMAL_SOAK_PASS",
	}, "\n") + "\n"
}

// cleanSegment is a healthy fifteen-minute window.
func cleanSegment() string { return thermalSegmentStdout("900", "71.5", "83.2", "0") }

// attemptPod finds the pod carrying one attempt number.
//
// The harness's own pods() picks the highest attempt label as a STRING, which
// orders "10" before "9". A soak runs hundreds of attempts, so it needs a
// numeric lookup or every assertion past attempt 9 would be about the wrong pod.
func (h *harness) attemptPod(runName string, attempt int) *corev1.Pod {
	h.t.Helper()
	want := strconv.Itoa(attempt)
	for _, pod := range h.allPods(runName) {
		if pod.Labels[labelAttempt] == want {
			return pod
		}
	}
	return nil
}

// awaitSegmentPod reconciles until the pod for one segment exists.
func (h *harness) awaitSegmentPod(runName string, attempt int) *corev1.Pod {
	h.t.Helper()
	for i := 0; i < 6; i++ {
		if pod := h.attemptPod(runName, attempt); pod != nil {
			return pod
		}
		h.reconcile(runName)
	}
	h.t.Fatalf("segment %d never got a pod", attempt)
	return nil
}

// burnSegment drives one whole segment: launch the pod, start it, finish it with
// the given exit code and stdout, and let the operator harvest it.
func (h *harness) burnSegment(runName string, attempt, exit int, stdout string) {
	h.t.Helper()
	pod := h.awaitSegmentPod(runName, attempt)
	h.startPod(pod)
	h.reconcile(runName)
	reason := "Completed"
	if exit != 0 {
		reason = "Error"
	}
	h.finishPod(pod, exit, stdout, reason)
	h.reconcile(runName)
}

func soakResult(t *testing.T, run *burninv1alpha1.BurnInRun) *burninv1alpha1.TestResult {
	t.Helper()
	if len(run.Status.Results) != 1 {
		t.Fatalf("expected exactly one result, got %d: %+v", len(run.Status.Results), run.Status.Results)
	}
	return &run.Status.Results[0]
}

// ─── The folding rules ────────────────────────────────────────────────────────

// The aggregation table. Each row states the rule it expects INDEPENDENTLY of
// the registry and then asserts the registry agrees, so a row cannot silently
// start testing a different rule because somebody re-declared a metric — and the
// expected value is written out rather than derived, so the test cannot agree
// with a broken fold by computing it the same broken way.
//
// The Last rows are the ones with teeth. eccErrors and remappedRows are NVML
// aggregates SINCE RESET, so summing them multiplies them by the number of
// windows: a node with four remapped rows would read as 1,152 after a
// 288-segment soak and be condemned for damage it does not have.
func TestFoldMetrics_CombinesByTheDeclaredAggregation(t *testing.T) {
	cases := []struct {
		name     string
		metric   string
		rule     contract.Aggregation
		segments []string
		want     string
	}{
		{"a per-window counter sums", "xidEvents", contract.AggSum, []string{"1", "0", "2"}, "3"},
		{"elapsedS sums, which is what makes a duration gate honest",
			"elapsedS", contract.AggSum, []string{"900", "900", "900"}, "2700"},
		{"a floor keeps the WORST window", "sustainedClockPct", contract.AggMin,
			[]string{"83.2", "40.5", "81"}, "40.5"},
		{"a floor is not improved by a later good window", "sustainedClockPct", contract.AggMin,
			[]string{"40.5", "83.2", "81"}, "40.5"},
		{"a ceiling keeps the HIGHEST window", "gpuTempC", contract.AggMax,
			[]string{"71", "88.5", "70"}, "88.5"},
		{"a lifetime total takes the last reading, never the sum", "eccErrors", contract.AggLast,
			[]string{"4", "4", "4"}, "4"},
		{"a remapped-row count is a lifetime total too", "remappedRows", contract.AggLast,
			[]string{"4", "4", "4"}, "4"},
		{"an unregistered name takes the last reading", "vendorWidgetCount", contract.AggLast,
			[]string{"7", "9", "2"}, "2"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := contract.AggregationFor(c.metric); got != c.rule {
				t.Fatalf("this row is written for %s but the registry declares %q as %s — "+
					"the row is now testing a rule it does not describe", c.rule, c.metric, got)
			}
			var agg map[string]string
			for _, v := range c.segments {
				agg = foldMetrics(agg, map[string]string{c.metric: v})
			}
			if got := agg[c.metric]; got != c.want {
				t.Errorf("%s over %v = %q, want %q", c.rule, c.segments, got, c.want)
			}
		})
	}
}

// A metric no segment reported stays absent. Fail-closed evaluation of that
// absence at the end of the soak is the correct outcome; inventing a zero for it
// would certify hardware nobody measured.
func TestFoldMetrics_NeverInventsAMeasurement(t *testing.T) {
	agg := foldMetrics(nil, map[string]string{"elapsedS": "900"})
	agg = foldMetrics(agg, map[string]string{"elapsedS": "900"})
	if _, present := agg["xidEvents"]; present {
		t.Errorf("aggregate invented a metric no segment reported: %v", agg)
	}
	if len(agg) != 1 {
		t.Errorf("aggregate = %v, want only the one metric that was actually reported", agg)
	}
}

// A reading nothing can add or order poisons its metric for the rest of the
// soak, and a later clean window must not launder it away. verdict.Evaluate
// fails a non-finite value closed under every comparison, which is where the
// poison is meant to land.
func TestFoldMetrics_NonFiniteReadingIsStickyAndFailsClosed(t *testing.T) {
	agg := foldMetrics(nil, map[string]string{"sustainedClockPct": "83.2"})
	agg = foldMetrics(agg, map[string]string{"sustainedClockPct": "NaN"})
	agg = foldMetrics(agg, map[string]string{"sustainedClockPct": "84.0"})
	if got := agg["sustainedClockPct"]; got != "NaN" {
		t.Fatalf("aggregate = %q, want the poisoned reading to survive a later clean window", got)
	}

	out := verdict.Evaluate(agg, nil, []burninv1alpha1.Threshold{
		{Metric: "sustainedClockPct", Comparison: burninv1alpha1.GTE, Value: "60"},
	})
	if out.Passed {
		t.Error("a soak whose gauge went non-finite passed its floor — that is an acceptance nobody measured")
	}
}

// A metric one window could not measure and another one did IS measurable, and
// the gate applies to the aggregate. Keeping both claims would make
// verdict.Evaluate report the runner as self-contradictory, which would be this
// operator inventing a contradiction out of two honest readings.
func TestFoldUnmeasurable_AMeasuredMetricLeavesTheDeclaration(t *testing.T) {
	agg := map[string]string{}
	names := foldUnmeasurable(nil, map[string]bool{"eccErrors": true, "remappedRows": true}, agg)
	if len(names) != 2 {
		t.Fatalf("declaration lost: %v", names)
	}

	agg = foldMetrics(agg, map[string]string{"eccErrors": "0"})
	names = foldUnmeasurable(names, nil, agg)
	if len(names) != 1 || names[0] != "remappedRows" {
		t.Errorf("unmeasurable = %v, want only remappedRows — eccErrors was measured in a later window", names)
	}
}

// ─── The crux: one verdict, over the aggregate ────────────────────────────────

// THE PROPERTY THIS WHOLE CHANGE EXISTS FOR. The gate is on the SUM of the
// soak's windows, and no single window can satisfy it: evaluated per segment it
// fails at segment one and condemns the node fifteen minutes into a
// forty-five-minute soak. Evaluated once, over the aggregate, it passes.
func TestSoak_ThresholdsAreEvaluatedOnceOverTheAggregate(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 2700, 900,
			burninv1alpha1.Threshold{Metric: "elapsedS", Comparison: burninv1alpha1.GTE, Value: "2565"}),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	h.burnSegment("run1", 1, 0, cleanSegment())
	res := soakResult(t, h.run("run1"))
	if res.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("result phase after segment 1 = %q, want Running — a gate on the whole soak "+
			"must not be applied to one window (message: %q)", res.Phase, res.Message)
	}
	if res.SegmentsCompleted != 1 || res.SegmentsRequired != 3 {
		t.Fatalf("segments = %d/%d, want 1/3", res.SegmentsCompleted, res.SegmentsRequired)
	}
	if got := res.AggregatedMetrics["elapsedS"]; got != "900" {
		t.Errorf("aggregate elapsedS after one segment = %q, want 900", got)
	}

	h.burnSegment("run1", 2, 0, cleanSegment())
	h.burnSegment("run1", 3, 0, cleanSegment())
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed (results: %+v)", run.Status.Phase, run.Status.Results)
	}
	res = soakResult(t, run)
	if got := res.AggregatedMetrics["elapsedS"]; got != "2700" {
		t.Errorf("aggregate elapsedS = %q, want 2700 — the gate is on the whole soak", got)
	}
	if got := res.Metrics["elapsedS"]; got != "2700" {
		t.Errorf("result metrics elapsedS = %q, want the aggregate the verdict was read from", got)
	}
	if res.SegmentsCompleted != 3 {
		t.Errorf("segmentsCompleted = %d, want 3", res.SegmentsCompleted)
	}
}

// The same soak against a gate the aggregate does NOT clear still fails, so the
// test above is not passing merely because thresholds stopped being applied at
// all. The aggregate falls two seconds short and the run says so.
func TestSoak_TheAggregateStillFailsClosed(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 2700, 900,
			burninv1alpha1.Threshold{Metric: "elapsedS", Comparison: burninv1alpha1.GTE, Value: "2702"},
			// A gate on a metric no segment reports must still fail closed at the
			// end of a soak, exactly as it does for an unsegmented test.
			burninv1alpha1.Threshold{Metric: "miscompares", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	for i := 1; i <= 3; i++ {
		h.burnSegment("run1", i, 0, cleanSegment())
	}
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("run phase = %q, want Failed", run.Status.Phase)
	}
	res := soakResult(t, run)
	if !strings.Contains(res.Message, "elapsedS") {
		t.Errorf("message does not name the gate the aggregate missed: %q", res.Message)
	}
	var sawMissing bool
	for _, v := range res.Violations {
		if v.Metric == "miscompares" && v.Cause == string(verdict.CauseEvidence) {
			sawMissing = true
		}
	}
	if !sawMissing {
		t.Errorf("a gated metric no segment reported did not fail closed: %+v", res.Violations)
	}
}

// ─── Acceptance: a lost segment costs one segment ─────────────────────────────

// A soak whose segment pod is deleted mid-run loses that segment and no more.
// The operator relaunches the same segment, the aggregate is untouched by the
// interruption, and the run reaches a verdict.
func TestSoak_ADeletedSegmentPodCostsOneSegmentAndNoMore(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 2700, 900),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	h.burnSegment("run1", 1, 0, cleanSegment())

	// Segment 2 is evicted: the pod is running and then simply gone.
	pod := h.awaitSegmentPod("run1", 2)
	h.startPod(pod)
	h.reconcile("run1")
	if err := h.c.Delete(context.Background(), pod); err != nil {
		t.Fatalf("delete segment pod: %v", err)
	}

	res := soakResult(t, h.run("run1"))
	if res.SegmentsCompleted != 1 {
		t.Fatalf("segmentsCompleted = %d after losing segment 2, want 1", res.SegmentsCompleted)
	}
	if got := res.AggregatedMetrics["elapsedS"]; got != "900" {
		t.Fatalf("aggregate elapsedS = %q, want the one segment that finished", got)
	}

	// The operator relaunches segment 2 and the soak carries on from there.
	h.burnSegment("run1", 2, 0, cleanSegment())
	h.burnSegment("run1", 3, 0, cleanSegment())
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed — an evicted segment must not end the soak (results: %+v)",
			run.Status.Phase, run.Status.Results)
	}
	res = soakResult(t, run)
	if got := res.AggregatedMetrics["elapsedS"]; got != "2700" {
		t.Errorf("aggregate elapsedS = %q, want 2700 — three windows were measured, no more and no less", got)
	}
}

// ─── Acceptance: an errored segment retries the same index ────────────────────

// The errored segment PRINTS metrics before it dies, which is what makes this
// discriminating: if an errored window contributed, the aggregate would read
// 2250 rather than 1800, and the soak would settle having burned less than it
// claims.
func TestSoak_AnErroredSegmentRetriesTheSameIndexAndContributesNothing(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(1)
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 1800, 900),
		profile("acceptance", nil, false, testRef("soak")),
		run,
	)

	h.burnSegment("run1", 1, 0, cleanSegment())
	// Segment 2 is killed 450 seconds in, having already printed that far.
	h.burnSegment("run1", 2, 137, thermalSegmentStdout("450", "70.0", "82.0", "0"))

	res := soakResult(t, h.run("run1"))
	if res.SegmentsCompleted != 1 {
		t.Errorf("segmentsCompleted = %d after an errored segment, want 1 — an error measured nothing",
			res.SegmentsCompleted)
	}
	if got := res.AggregatedMetrics["elapsedS"]; got != "900" {
		t.Errorf("aggregate elapsedS = %q, want 900 — an errored window must not contribute", got)
	}
	if res.ErrorRetries != 1 {
		t.Errorf("errorRetries = %d, want 1 — an errored segment spends the retry budget", res.ErrorRetries)
	}
	if res.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("result phase = %q, want Running", res.Phase)
	}

	// The retry re-runs segment 2, not segment 3.
	h.burnSegment("run1", 3, 0, cleanSegment())
	res = soakResult(t, h.run("run1"))
	if res.SegmentsCompleted != 2 {
		t.Fatalf("segmentsCompleted = %d after the retry, want 2", res.SegmentsCompleted)
	}
	h.reconcileUntilSettled("run1")

	run = h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed (results: %+v)", run.Status.Phase, run.Status.Results)
	}
	res = soakResult(t, run)
	if got := res.AggregatedMetrics["elapsedS"]; got != "1800" {
		t.Errorf("aggregate elapsedS = %q, want 1800 — two windows were measured", got)
	}
	if got := res.Attempts[1].Trigger; got != burninv1alpha1.AttemptSegment {
		t.Errorf("attempt 2 trigger = %q, want Segment", got)
	}
	if got := res.Attempts[2].Trigger; got != burninv1alpha1.AttemptErrorRetry {
		t.Errorf("attempt 3 trigger = %q, want ErrorRetry — it followed an Error", got)
	}
}

// ─── Acceptance: exit 1 settles the soak now ──────────────────────────────────

// A fault observed at hour six is a fault. The soak settles Failed on the spot,
// with the retry budget unspent — Failed is the one phase that is never retried
// — and nothing launches segment three.
func TestSoak_ASegmentExitingOneSettlesTheTestImmediately(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(2)
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 2700, 900),
		profile("acceptance", nil, false, testRef("soak")),
		run,
	)

	h.burnSegment("run1", 1, 0, cleanSegment())
	h.burnSegment("run1", 2, 1, "gpu_name=NVIDIA GB10\nxid_events=3\nTHERMAL_SOAK_FAIL: Xid 79 during the soak\n")
	h.reconcileUntilSettled("run1")

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("run phase = %q, want Failed (results: %+v)", got.Status.Phase, got.Status.Results)
	}
	res := soakResult(t, got)
	if res.SegmentsCompleted != 1 {
		t.Errorf("segmentsCompleted = %d, want 1 — a failing window does not complete", res.SegmentsCompleted)
	}
	if res.ErrorRetries != 0 {
		t.Errorf("errorRetries = %d, want 0 — a Fail is never retried, whatever budget is left", res.ErrorRetries)
	}
	if pod := h.attemptPod("run1", 3); pod != nil {
		t.Errorf("segment 3 was launched after a hardware failure: %s", pod.Name)
	}
	if !strings.Contains(res.Message, "Xid 79") {
		t.Errorf("result message lost the runner's own assertion: %q", res.Message)
	}
}

// ─── Acceptance: AbortEarly ───────────────────────────────────────────────────

// A Sum counter under `Equal 0` is monotone in the gated direction: once the sum
// has passed zero it can never return to it, so the remaining windows cannot
// change the answer and burning them would be hoping rather than measuring.
func TestSoak_AbortEarlyFiresOnAMonotoneCounterBreach(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		withAbortEarly(soakTest("soak", 3600, 900,
			burninv1alpha1.Threshold{Metric: "xidEvents", Comparison: burninv1alpha1.EQ, Value: "0"})),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	h.burnSegment("run1", 1, 0, cleanSegment())
	if res := soakResult(t, h.run("run1")); res.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("phase after a clean segment = %q, want Running", res.Phase)
	}

	// Two Xids in window two. The sum is 2 and no later window can bring it back.
	h.burnSegment("run1", 2, 0, thermalSegmentStdout("900", "72.0", "83.0", "2"))
	h.reconcileUntilSettled("run1")

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("run phase = %q, want Failed (results: %+v)", got.Status.Phase, got.Status.Results)
	}
	res := soakResult(t, got)
	if pod := h.attemptPod("run1", 3); pod != nil {
		t.Errorf("segment 3 was burned after the soak was already decided: %s", pod.Name)
	}
	if !strings.Contains(res.Message, "aborted after segment 2 of 4") {
		t.Errorf("message does not say the soak was cut short: %q", res.Message)
	}
	if len(res.Violations) != 1 || res.Violations[0].Metric != "xidEvents" {
		t.Errorf("violations = %+v, want the gate that ended it", res.Violations)
	}
	if got := res.AggregatedMetrics["xidEvents"]; got != "2" {
		t.Errorf("aggregate xidEvents = %q, want 2", got)
	}
}

// The counterpart, and the reason AbortEarly is narrow. sustainedClockPct
// aggregates Min, so a gauge that reads high in the first window and low later
// RETRACTS a ceiling violation: the aggregate at segment one violates the gate
// and the aggregate at the end does not. Aborting on that would end a soak on a
// reading the next fifteen minutes overturned.
func TestSoak_AbortEarlyDoesNotFireOnARetractableViolation(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		withAbortEarly(soakTest("soak", 2700, 900,
			burninv1alpha1.Threshold{Metric: "sustainedClockPct", Comparison: burninv1alpha1.LTE, Value: "50"})),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	// Segment 1: the aggregate is 83.2, which is over the ceiling.
	h.burnSegment("run1", 1, 0, cleanSegment())
	res := soakResult(t, h.run("run1"))
	if res.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("phase = %q, want Running — a Min under a ceiling can still be pulled under it "+
			"by a later window, so this violation was not provable (message: %q)", res.Phase, res.Message)
	}
	if got := verdict.Evaluate(res.AggregatedMetrics, nil, []burninv1alpha1.Threshold{
		{Metric: "sustainedClockPct", Comparison: burninv1alpha1.LTE, Value: "50"},
	}); got.Passed {
		t.Fatal("the aggregate at segment 1 does not violate the gate, so this test is not " +
			"exercising the case it describes")
	}

	// Segment 2 retracts it, and the soak reaches its real verdict.
	h.burnSegment("run1", 2, 0, thermalSegmentStdout("900", "70.0", "40.0", "0"))
	h.burnSegment("run1", 3, 0, cleanSegment())
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed — the violation was retracted (results: %+v)",
			run.Status.Phase, run.Status.Results)
	}
	if got := soakResult(t, run).AggregatedMetrics["sustainedClockPct"]; got != "40.0" {
		t.Errorf("aggregate sustainedClockPct = %q, want the worst window", got)
	}
}

// A soak that does not ask to abort never does, however provable the breach.
// The gate is still applied once, at the end.
func TestSoak_AbortEarlyIsOptIn(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 1800, 900,
			burninv1alpha1.Threshold{Metric: "xidEvents", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	h.burnSegment("run1", 1, 0, thermalSegmentStdout("900", "72.0", "83.0", "2"))
	if res := soakResult(t, h.run("run1")); res.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("phase = %q, want Running — this test never opted into aborting early", res.Phase)
	}
	h.burnSegment("run1", 2, 0, cleanSegment())
	h.reconcileUntilSettled("run1")

	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunFailed {
		t.Fatalf("run phase = %q, want Failed — the gate still decides at the end", got)
	}
}

// The monotonicity rule itself, stated as a table so every combination is named
// rather than inferred from whichever two the end-to-end tests happened to use.
func TestProvableViolations_OnlyWhereNoLaterSegmentCouldRetract(t *testing.T) {
	cases := []struct {
		name       string
		metric     string
		comparison burninv1alpha1.Comparison
		value      string
		aggregate  string
		want       bool
	}{
		{"a sum over its ceiling only grows", "xidEvents", burninv1alpha1.LTE, "1", "4", true},
		{"a sum past an exact value can never return to it", "xidEvents", burninv1alpha1.EQ, "0", "2", true},
		{"a sum short of an exact value can still reach it", "elapsedS", burninv1alpha1.EQ, "2700", "900", false},
		{"a sum short of a floor can still reach it", "elapsedS", burninv1alpha1.GTE, "2700", "900", false},
		{"a minimum below its floor only falls further", "sustainedClockPct", burninv1alpha1.GTE, "60", "40", true},
		{"a minimum over a ceiling can be pulled under it", "sustainedClockPct", burninv1alpha1.LTE, "50", "83", false},
		{"a maximum over its ceiling only rises", "gpuTempC", burninv1alpha1.LTE, "85", "91", true},
		{"a maximum below a floor can still rise to it", "gpuTempC", burninv1alpha1.GTE, "40", "30", false},
		{"a lifetime total can move either way", "eccErrors", burninv1alpha1.EQ, "0", "3", false},
		{"an unregistered name aggregates Last and is never provable",
			"vendorWidgetCount", burninv1alpha1.LTE, "1", "9", false},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			thresholds := []burninv1alpha1.Threshold{
				{Metric: c.metric, Comparison: c.comparison, Value: c.value},
			}
			agg := map[string]string{c.metric: c.aggregate}
			out := verdict.Evaluate(agg, nil, thresholds)
			if out.Passed {
				t.Fatalf("this row's aggregate does not violate its gate, so it proves nothing")
			}
			got := len(provableViolations(agg, out, thresholds)) == 1
			if got != c.want {
				t.Errorf("provable = %v, want %v", got, c.want)
			}
		})
	}
}

// A gate on a metric no segment has reported YET must never abort a soak: the
// next window may well report it. Fail-closed still applies at the end, where
// the absence really is final — TestSoak_TheAggregateStillFailsClosed is that
// half.
// The comparison is deliberately LessThanOrEqual on a Sum, which is the one
// combination the monotonicity rule would otherwise call provable: it is the
// CAUSE that has to stop this, not the arithmetic, and an `Equal 0` gate would
// let the test pass by the wrong route.
func TestProvableViolations_AnUnreportedMetricNeverAborts(t *testing.T) {
	thresholds := []burninv1alpha1.Threshold{
		{Metric: "xidEvents", Comparison: burninv1alpha1.LTE, Value: "1"},
	}
	agg := map[string]string{"elapsedS": "900"}
	out := verdict.Evaluate(agg, nil, thresholds)
	if out.Passed {
		t.Fatal("a gated metric that was not reported must not pass")
	}
	if got := provableViolations(agg, out, thresholds); len(got) != 0 {
		t.Errorf("provable = %+v, want none — a metric the next window may report is not a verdict", got)
	}
}

// A checkpoint is a PARTIAL window, and folding one would count the same seconds
// twice when the segment finishes — a soak certifying a duration it never ran.
// The checkpoint still publishes its evidence onto the result, which is what it
// is for; it simply never touches the aggregate.
func TestSoak_ACheckpointNeverFoldsIntoTheAggregate(t *testing.T) {
	bt := soakTest("soak", 1800, 900)
	bt.Spec.CheckpointIntervalSeconds = int32p(60)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	pod := h.awaitSegmentPod("run1", 1)
	h.startPod(pod)
	h.reconcile("run1")
	// Half a window in, the runner has printed 450 seconds so far.
	h.logs[pod.Name] = thermalSegmentStdout("450", "70.0", "82.0", "0")
	h.nowVal = h.nowVal.Add(90 * time.Second)
	h.reconcile("run1")

	res := soakResult(t, h.run("run1"))
	if res.LastCheckpointAt == nil {
		t.Fatal("no checkpoint was taken, so this test is not exercising the case it describes")
	}
	if got := res.Metrics["elapsedS"]; got != "450" {
		t.Errorf("checkpoint evidence = %q, want the partial window's own reading", got)
	}
	if got, present := res.AggregatedMetrics["elapsedS"]; present {
		t.Fatalf("a checkpoint folded %q into the aggregate — the same seconds will be counted "+
			"again when the segment finishes", got)
	}

	h.finishPod(pod, 0, cleanSegment(), "Completed")
	h.reconcile("run1")
	if got := soakResult(t, h.run("run1")).AggregatedMetrics["elapsedS"]; got != "900" {
		t.Fatalf("aggregate elapsedS = %q after one window, want 900", got)
	}

	h.burnSegment("run1", 2, 0, cleanSegment())
	h.reconcileUntilSettled("run1")
	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed (results: %+v)", run.Status.Phase, run.Status.Results)
	}
	if got := soakResult(t, run).AggregatedMetrics["elapsedS"]; got != "1800" {
		t.Errorf("aggregate elapsedS = %q, want 1800 — two windows were measured", got)
	}
}

// ─── The pod's window is the segment ──────────────────────────────────────────

// Both halves of it, because they are two different consumers of the same
// answer: the runner's own budget and the pod's deadline come from
// executionDurationSeconds, and the operator's reap window does too.
func TestSoak_ThePodWindowIsTheSegmentAndNotTheWholeSoak(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 604800, 900),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	pod := h.awaitSegmentPod("run1", 1)

	if got := envOf(pod, "BURNIN_DURATION_SECONDS"); got != "900" {
		t.Errorf("BURNIN_DURATION_SECONDS = %q, want 900 — the runner is asked for one window", got)
	}
	want := int64(900 + deadlineGraceSeconds)
	if pod.Spec.ActiveDeadlineSeconds == nil || *pod.Spec.ActiveDeadlineSeconds != want {
		t.Errorf("activeDeadlineSeconds = %v, want %d — a pod that outlives its segment is a pod "+
			"an eviction still costs the whole week", pod.Spec.ActiveDeadlineSeconds, want)
	}
}

// The operator-side window too. An unschedulable segment pod is reaped after ONE
// segment plus the graces; sized from the whole soak it would sit there for a
// week while the run held a cordoned node.
func TestSoak_AnUnschedulableSegmentIsReapedOnTheSegmentWindow(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 604800, 900),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	pod := h.awaitSegmentPod("run1", 1)
	pod.CreationTimestamp = metav1.NewTime(h.nowVal)
	if err := h.c.Update(context.Background(), pod); err != nil {
		t.Fatalf("stamp pod creation: %v", err)
	}

	// Just past ONE segment's window. Sized from the whole soak this is not even
	// close — a week-long deadline would leave the pod sitting there, holding a
	// cordoned node, for six more days.
	h.nowVal = h.nowVal.Add(time.Duration(900+deadlineGraceSeconds)*time.Second + schedulingGracePeriod + time.Second)
	h.reconcile("run1")

	res := soakResult(t, h.run("run1"))
	if len(res.Attempts) != 1 || res.Attempts[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("attempts = %+v, want one Error — a segment that never ran must be reaped on the "+
			"segment's window, not the soak's", res.Attempts)
	}
	if res.SegmentsCompleted != 0 {
		t.Errorf("segmentsCompleted = %d, want 0 — a pod that never started measured nothing", res.SegmentsCompleted)
	}
}

// ─── Acceptance: 288 segments stay writable and still render a verdict ────────

// A 72-hour soak at fifteen-minute segments. It is driven through completeAttempt
// rather than through 288 pod lifecycles because what is under test is the
// bookkeeping, and it is the same bookkeeping either way — the pod lifecycle has
// its own tests above.
//
// The load-bearing assertions are the attempt count and TruncatedAttempts. The
// delivery is executed rather than described: the real ConfigMap sink applies the
// real cap to the real envelope, so a status that had grown unwritable would fail
// here rather than in a fleet.
func TestSoak_ManySegmentsStayWritableAndStillRenderTheVerdict(t *testing.T) {
	const segments = 288

	h := newHarness(t, gb10Node("spark-a"))
	run := newRun("run1", "acceptance", "spark-a")
	run.Status.Phase = burninv1alpha1.RunRunning

	bt := soakTest("soak", segments*900, 900,
		burninv1alpha1.Threshold{Metric: "elapsedS", Comparison: burninv1alpha1.GTE, Value: "246240"},
		burninv1alpha1.Threshold{Metric: "xidEvents", Comparison: burninv1alpha1.EQ, Value: "0"})
	pt := plannedTest{Name: "soak", Required: true, Spec: bt.Spec}
	p := &plan{Version: 1, Targets: []string{"spark-a"}, Tests: []plannedTest{pt}}

	for i := 1; i <= segments; i++ {
		h.r.completeAttempt(context.Background(), run, p, pt, []string{"spark-a"},
			int32(i), fmt.Sprintf("run1-soak-%d", i), nil,
			runner.Parse(string(bt.Spec.Kind), cleanSegment(), 0))
	}

	res := soakResult(t, run)
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed after %d clean segments (message: %q)", res.Phase, segments, res.Message)
	}
	if res.SegmentsCompleted != segments {
		t.Errorf("segmentsCompleted = %d, want %d", res.SegmentsCompleted, segments)
	}
	if got := res.AggregatedMetrics["elapsedS"]; got != strconv.Itoa(segments*900) {
		t.Errorf("aggregate elapsedS = %q, want %d — truncating the history must not lose the verdict's input",
			got, segments*900)
	}

	wantKept := keptPassingSegments + 1
	if len(res.Attempts) != wantKept {
		t.Errorf("kept %d attempts, want %d (the first plus the last %d passing)",
			len(res.Attempts), wantKept, keptPassingSegments)
	}
	if got, want := res.TruncatedAttempts, int32(segments-wantKept); got != want {
		t.Errorf("truncatedAttempts = %d, want %d — an elided history must say how much it elided", got, want)
	}
	if res.Attempts[0].Attempt != 1 {
		t.Errorf("the first attempt was dropped: %+v", res.Attempts[0])
	}
	if last := res.Attempts[len(res.Attempts)-1]; last.Attempt != segments {
		t.Errorf("the last attempt is %d, want %d — nextAttempt reads it to decide which segment "+
			"comes next, so dropping it restarts the soak", last.Attempt, segments)
	}

	// The status has to fit in etcd, and the envelope has to fit in the sink.
	body, err := json.Marshal(run.Status)
	if err != nil {
		t.Fatalf("marshal status: %v", err)
	}
	const statusBudget = 32 * 1024
	if len(body) > statusBudget {
		t.Errorf("status is %d bytes, over the %d-byte budget a segmented soak is held to", len(body), statusBudget)
	}

	cm := &sink.ConfigMap{Client: h.c, Namespace: "burnin", Name: "burnin-results"}
	env := sink.EnvelopeFor(run, "acceptance", contract.ReasonPhaseChanged, sink.PhaseKey(burninv1alpha1.RunPassed),
		h.nowVal, nil)
	if err := cm.Deliver(context.Background(), env); err != nil {
		t.Fatalf("the ConfigMap sink refused a %d-segment soak's verdict: %v", segments, err)
	}
	if len(env.Results) != 1 || env.Results[0].Metrics["elapsedS"] != strconv.Itoa(segments*900) {
		t.Errorf("delivered envelope does not carry the aggregate the verdict was read from: %+v", env.Results)
	}
}

// ─── Acceptance: a controller restart resumes at the right segment ────────────

// Driven far enough that the attempt history has already been truncated, because
// that is the interaction with teeth: nextAttempt reads the LAST attempt to
// decide which segment comes next, and a cap that dropped it would restart a
// seven-day soak at segment one after every manager rollout.
func TestSoak_AControllerRestartResumesAtTheRightSegment(t *testing.T) {
	const before = 20

	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 25*900, 900),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	for i := 1; i <= before; i++ {
		h.burnSegment("run1", i, 0, cleanSegment())
	}
	res := soakResult(t, h.run("run1"))
	if res.TruncatedAttempts == 0 {
		t.Fatalf("no attempt was truncated after %d segments, so this test is not exercising the "+
			"interaction it describes", before)
	}

	// A manager restart: a brand-new reconciler, holding nothing, re-deriving
	// everything it knows from the persisted run.
	h.r = h.newReconciler()
	h.reconcile("run1")

	if pod := h.attemptPod("run1", before+1); pod == nil {
		t.Fatalf("no pod for segment %d after a restart; pods so far: %d", before+1, len(h.allPods("run1")))
	}
	if pod := h.attemptPod("run1", 1); pod != nil && pod.Status.Phase != corev1.PodSucceeded {
		t.Errorf("segment 1 was relaunched after a restart: %+v", pod.Status.Phase)
	}
	res = soakResult(t, h.run("run1"))
	if res.SegmentsCompleted != before {
		t.Errorf("segmentsCompleted = %d after a restart, want %d", res.SegmentsCompleted, before)
	}
	if got := res.AggregatedMetrics["elapsedS"]; got != strconv.Itoa(before*900) {
		t.Errorf("aggregate elapsedS = %q after a restart, want %d", got, before*900)
	}

	for i := before + 1; i <= 25; i++ {
		h.burnSegment("run1", i, 0, cleanSegment())
	}
	h.reconcileUntilSettled("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed", got)
	}
}

// ─── Plan-time refusals ───────────────────────────────────────────────────────

func TestSoak_PlanRefusals(t *testing.T) {
	cases := []struct {
		name string
		spec func(*burninv1alpha1.BurnInTestSpec)
		want string
	}{
		{
			name: "a burst kind has no duration to divide",
			spec: func(s *burninv1alpha1.BurnInTestSpec) { s.Kind = burninv1alpha1.KindComputeSmoke },
			want: "burst-only",
		},
		{
			name: "a segment below the floor spends its time changing pods",
			spec: func(s *burninv1alpha1.BurnInTestSpec) { s.Soak.SegmentSeconds = 60 },
			want: "below the 300-second floor",
		},
		{
			name: "a soak has to say how long it soaks",
			spec: func(s *burninv1alpha1.BurnInTestSpec) { s.DurationSeconds = 0 },
			want: "no spec.durationSeconds",
		},
		{
			name: "a segment longer than the soak burns longer than asked",
			spec: func(s *burninv1alpha1.BurnInTestSpec) { s.Soak.SegmentSeconds = 3600 },
			want: "longer than the soak",
		},
		{
			name: "a repeat and a segment are different claims about the verdict",
			spec: func(s *burninv1alpha1.BurnInTestSpec) { s.RepeatCount = int32p(3) },
			want: "repeatCount",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			spec := soakTest("soak", 2700, 900).Spec
			c.spec(&spec)
			_, err := buildPlan(&burninv1alpha1.BurnInProfile{},
				[]resolvedTest{{name: "soak", spec: spec, required: true}},
				[]string{"spark-a"}, 1, false)
			if err == nil {
				t.Fatalf("buildPlan accepted a soak it should have refused")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("refusal %q does not explain %q", err.Error(), c.want)
			}
		})
	}
}

// A test that does not set spec.soak is untouched by any of this: one pod, the
// whole duration, thresholds on that one execution. Segmentation is opt-in, and
// this is what "opt-in" has to mean.
func TestSoak_AnUnsegmentedTestIsUnchanged(t *testing.T) {
	bt := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "soak"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:            burninv1alpha1.KindThermalSoak,
			Scope:           burninv1alpha1.ScopeNode,
			DurationSeconds: 2700,
			Thresholds: []burninv1alpha1.Threshold{
				{Metric: "elapsedS", Comparison: burninv1alpha1.GTE, Value: "2565"},
			},
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	pod := h.awaitSegmentPod("run1", 1)
	if got := envOf(pod, "BURNIN_DURATION_SECONDS"); got != "2700" {
		t.Errorf("BURNIN_DURATION_SECONDS = %q, want the whole duration", got)
	}
	h.startPod(pod)
	h.reconcile("run1")
	h.finishPod(pod, 0, thermalSegmentStdout("2700", "71.5", "83.2", "0"), "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed (results: %+v)", run.Status.Phase, run.Status.Results)
	}
	res := soakResult(t, run)
	if res.SegmentsRequired != 0 || res.SegmentsCompleted != 0 || len(res.AggregatedMetrics) != 0 {
		t.Errorf("an unsegmented result grew segment bookkeeping: %+v", res)
	}
	if len(res.Attempts) != 1 || res.Attempts[0].Trigger != burninv1alpha1.AttemptInitial {
		t.Errorf("attempts = %+v, want one Initial attempt", res.Attempts)
	}
}

// ─── Acceptance: an Error at the end does not discard the soak ────────────────

// A soak that errors in its last window must still report what the windows
// before it measured.
//
// `settle` overwrites res.Metrics with the metrics of the attempt that decided
// the verdict, which is right for a Pass and for a Fail — both are assertions
// about the hardware, explained by the window that made them. An ERROR asserts
// nothing about hardware. It says this attempt produced no measurement, and an
// attempt that produced no measurement is not a reason to discard the 287 that
// did.
//
// The failure is worst exactly where the feature is most valuable: an evicted
// or unschedulable pod prints nothing at all, so parsed.Metrics is empty and a
// six-day soak reports Error with no metrics whatsoever. AggregatedMetrics still
// holds the evidence on the object, but res.Metrics is the field every consumer
// reads and the envelope carries.
func TestSoak_AnErrorInTheLastSegmentKeepsWhatTheEarlierOnesMeasured(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	// No retry budget: the first error settles, which is the state a long soak
	// reaches anyway once its single cumulative budget is spent.
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 1800, 900),
		profile("acceptance", nil, false, testRef("soak")),
		run,
	)

	// The clean window also DECLARES a counter unmeasurable on this part, which
	// foldSegment records on the result. That declaration is evidence about the
	// hardware exactly as the metrics are, and settle overwrites it from the
	// same errored attempt.
	h.burnSegment("run1", 1, 0, cleanSegment()+"ecc_errors=n/a\n")
	res := soakResult(t, h.run("run1"))
	if got := res.AggregatedMetrics["elapsedS"]; got != "900" {
		t.Fatalf("aggregate elapsedS = %q after one clean segment, want 900", got)
	}
	if len(res.Unmeasurable) == 0 {
		t.Fatalf("the clean segment's n/a declaration was not folded onto the result")
	}

	// Segment 2 is evicted before the container printed anything — no stdout at
	// all, which is the ordinary shape of an eviction and the case that loses
	// everything.
	h.burnSegment("run1", 2, 137, "")
	h.reconcileUntilSettled("run1")

	run = h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("run phase = %q, want Error", run.Status.Phase)
	}
	res = soakResult(t, run)
	if res.Phase != burninv1alpha1.RunError {
		t.Fatalf("result phase = %q, want Error", res.Phase)
	}

	if len(res.Metrics) == 0 {
		t.Fatalf("res.Metrics is empty: the errored window measured nothing and replaced 900 seconds "+
			"that did. AggregatedMetrics still holds %v, but nothing downstream reads that field",
			res.AggregatedMetrics)
	}
	if got := res.Metrics["elapsedS"]; got != "900" {
		t.Errorf("res.Metrics[elapsedS] = %q, want 900 — the aggregate, not the errored window", got)
	}
	// The aggregate itself is untouched by the settle.
	if got := res.AggregatedMetrics["elapsedS"]; got != "900" {
		t.Errorf("aggregate elapsedS = %q after the settle, want 900", got)
	}
	// One clean segment really did complete, and the count says so.
	if res.SegmentsCompleted != 1 {
		t.Errorf("segmentsCompleted = %d, want 1 — the errored segment contributes nothing", res.SegmentsCompleted)
	}
	// The n/a declaration survives too. A runner positively established that
	// this part has no ECC to read; an eviction two windows later is not new
	// information about that.
	if len(res.Unmeasurable) == 0 {
		t.Errorf("the folded Unmeasurable set was cleared by the errored attempt's empty evidence")
	}
}

// The counterpart, and the one the segmented carry-over must not swallow.
//
// An ORDINARY test that errors still reports what its runner printed, from the
// attempt that errored. There is no aggregate for it — foldSegment never runs on
// an unsegmented test — so a carry-over written without the SegmentsRequired
// guard would settle it with an empty map and lose the very output that explains
// the error. That mistake is invisible from the segmented test above, because
// widening a condition it already satisfies changes nothing about it.
func TestAnUnsegmentedErrorStillReportsTheAttemptsOwnMetrics(t *testing.T) {
	bt := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "soak"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:            burninv1alpha1.KindThermalSoak,
			Scope:           burninv1alpha1.ScopeNode,
			DurationSeconds: 2700,
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)

	pod := h.awaitSegmentPod("run1", 1)
	h.startPod(pod)
	h.reconcile("run1")
	// Exit 3: the runner reached the hardware, could not judge it, and said so
	// with the readings it did take. Those readings are the whole diagnosis.
	h.finishPod(pod, 3, thermalSegmentStdout("412", "94.0", "31.5", "2"), "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("run phase = %q, want Error", run.Status.Phase)
	}
	res := soakResult(t, run)
	if got := res.Metrics["elapsedS"]; got != "412" {
		t.Errorf("res.Metrics[elapsedS] = %q, want 412 — the errored attempt's own reading", got)
	}
	if got := res.Metrics["xidEvents"]; got != "2" {
		t.Errorf("res.Metrics[xidEvents] = %q, want 2 — the reading that explains the error", got)
	}
	if len(res.AggregatedMetrics) != 0 {
		t.Errorf("an unsegmented result grew an aggregate: %v", res.AggregatedMetrics)
	}
}

// The window length is pinned on the result, like the segment count beside it.
//
// SegmentsCompleted is a COUNT, and a count is not a duration: "40 of 288" says
// how far a soak got only if the reader also knows how long a window is. Neither
// the status nor the delivery envelope carries the run's spec, so without this
// the number cannot be turned into hours by anything downstream — and the
// envelope's segment block (#229) is assembled from these fields.
func TestSoak_TheResultRecordsTheWindowLength(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 1800, 900),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.burnSegment("run1", 1, 0, cleanSegment())

	res := soakResult(t, h.run("run1"))
	if res.SegmentSeconds != 900 {
		t.Errorf("segmentSeconds = %d, want 900 — pinned from the plan alongside segmentsRequired=%d",
			res.SegmentSeconds, res.SegmentsRequired)
	}
}

// And an unsegmented test records none, so zero keeps meaning "not segmented"
// on every field of the block rather than only on the count.
func TestAnUnsegmentedResultRecordsNoWindowLength(t *testing.T) {
	bt := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "soak"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind: burninv1alpha1.KindThermalSoak, Scope: burninv1alpha1.ScopeNode,
			DurationSeconds: 2700,
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"), bt,
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	pod := h.awaitSegmentPod("run1", 1)
	h.startPod(pod)
	h.reconcile("run1")

	res := soakResult(t, h.run("run1"))
	if res.SegmentSeconds != 0 || res.SegmentsRequired != 0 {
		t.Errorf("an unsegmented result grew a window length: segmentSeconds=%d segmentsRequired=%d",
			res.SegmentSeconds, res.SegmentsRequired)
	}
}

// A segmented soak must hold its node for the WHOLE soak, not one window at a
// time.
//
// busyNodes counts LIVE PODS, and between segment N and segment N+1 there is no
// pod: the previous one has terminated and the next has not been created. In
// that gap the node was neither busy nor held, so a second target took the
// concurrency slot and releaseHeldNodes gave the first node's cordon back.
//
// Two consequences, both severe:
//
//   - THE SOAK STOPS BEING A SOAK. At cap 1 over two targets, each node runs one
//     window, then idles for a window while the other node runs, forever. A
//     thermal soak exists to drive a part to thermal steady state and HOLD it
//     there; a part allowed to cool for as long as it was loaded never reaches
//     the condition being tested, so throttling and heat-dependent faults are
//     exactly what this schedule cannot find.
//   - THE CORDON COMES OFF MID-SOAK. Runner pods tolerate
//     node.kubernetes.io/unschedulable, so the cordon is the only thing keeping
//     FOREIGN workload off a node under burn-in. Dropping it halfway through a
//     multi-day acceptance lets ordinary fleet work land beside the measurement.
func TestSoak_ANodeIsHeldForTheWholeSoakAndNotOneSegmentAtATime(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a", "spark-b")
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		soakTest("soak", 2700, 900), // three segments
		profile("acceptance", nil, false, testRef("soak")),
		run,
	)

	// Segment 1 on whichever node the wave admitted first.
	pod := h.awaitSegmentPod("run1", 1)
	first := pod.Spec.NodeSelector[corev1.LabelHostname]
	h.startPod(pod)
	h.reconcile("run1")
	h.finishPod(pod, 0, cleanSegment(), "Completed")
	h.reconcile("run1")

	// The soak owes two more windows on `first`. Nothing else may have taken its
	// slot, and its cordon must still be on.
	got := h.run("run1")
	held := false
	for _, n := range got.Status.CordonedNodes {
		if n == first {
			held = true
		}
	}
	if !held {
		t.Errorf("status.cordonedNodes = %v: %s was released between segments, with two of its three "+
			"windows still owed — the cordon is the only thing keeping foreign workload off a node "+
			"under burn-in", got.Status.CordonedNodes, first)
	}
	if !h.node(first).Spec.Unschedulable {
		t.Errorf("%s is schedulable again mid-soak", first)
	}

	// And the next pod must be the SAME node's segment 2, not the other node's
	// segment 1 — at cap 1 the second target cannot start until this soak is done.
	next := h.awaitSegmentPod("run1", 2)
	if got := next.Spec.NodeSelector[corev1.LabelHostname]; got != first {
		t.Errorf("segment 2 landed on %s, but %s's soak is only 1/3 done: the slot was handed away at "+
			"the segment boundary, so each node runs one window then idles for one", got, first)
	}
}

// A PAIR holds BOTH of its nodes across the gap between executions, not just the
// first one.
//
// A pair is one indivisible unit of load costing two slots, and admission
// already applies that rule when the unit STARTS. It has to hold across the
// pause between one execution and the next too: releasing the second endpoint
// would uncordon a node that is half of a link measurement still in progress,
// and the same reasoning covers a Group, where every rank is held for the life
// of the collective.
//
// Observable through the cordon rather than through a competing target, because
// a Pair consumes every node this run has: if only Nodes[0] were held, the
// client's node would be handed back to the scheduler between repeats while the
// link test is still running.
func TestPair_HoldsBothEndpointsBetweenRepeats(t *testing.T) {
	bt := pairTest("ib")
	bt.Spec.RepeatCount = int32p(2)
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		bt,
		profile("fabric", nil, false, testRef("ib")),
		pairRun("run1", "fabric", "spark-a", "spark-b"),
	)

	server, client := runPairToStart(h)
	h.finishPod(server, 0, "", "Completed")
	h.finishPod(client, 0, ibClientStdout, "Completed")
	h.reconcile("run1")

	res := &h.run("run1").Status.Results[0]
	if res.Phase.IsTerminal() {
		t.Fatalf("the pair settled after one repeat of two (phase %q); this test needs the gap "+
			"between executions to exist at all", res.Phase)
	}

	got := h.run("run1")
	for _, want := range []string{"spark-a", "spark-b"} {
		held := false
		for _, n := range got.Status.CordonedNodes {
			if n == want {
				held = true
			}
		}
		if !held {
			t.Errorf("status.cordonedNodes = %v: %s was released between repeats, but the pair owes "+
				"another execution and a link measured from one end is not measured",
				got.Status.CordonedNodes, want)
		}
		if !h.node(want).Spec.Unschedulable {
			t.Errorf("%s is schedulable again while its pair test is still running", want)
		}
	}
}

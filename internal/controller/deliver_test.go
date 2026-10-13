package controller

import (
	"context"
	"reflect"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// A MID-RUN EDIT TO spec.baseline MUST NOT CHANGE WHAT THE ENVELOPE REPORTS.
//
// `baseline` says this run MEASURED rather than certified, and docs/sinks.md
// promises flatly that "true here is a reliable claim that nothing was gated".
// Reading spec.baseline at delivery time made that a claim about the object's
// state when the envelope was built rather than about the run that executed:
// clearing the flag on a live run turned every later envelope — including the
// terminal one a consumer archives — into a certification of a sweep that
// applied no thresholds at all. The run still delivers phase Passed, so a
// control plane gating admission on "the last run passed" would admit a fleet
// nobody measured.
//
// The plan is pinned at start precisely so a mid-run edit cannot change what an
// execution meant. This is the same rule applied to what it REPORTS.
func TestDeliver_BaselineComesFromThePinnedPlanNotTheLiveSpec(t *testing.T) {
	bt := smokeTest("fp4")
	// A baseline run: no thresholds anywhere, so nothing is gated.
	bt.Spec.Thresholds = nil
	run := newRun("run1", "acceptance", "spark-a")
	tru := true
	run.Spec.Baseline = &tru

	h := newHarness(t,
		gb10Node("spark-a"), bt,
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)
	h.reconcile("run1") // pins the plan, with Baseline: true

	got := h.run("run1")
	p, ok, err := loadPlan(got)
	if err != nil || !ok {
		t.Fatalf("no pinned plan after the first pass (ok=%v err=%v)", ok, err)
	}
	if !p.Baseline {
		t.Fatalf("the plan did not pin baseline=true, so this test cannot observe the case")
	}
	if !pinnedBaseline(got) {
		t.Fatalf("precondition: pinnedBaseline disagrees with the plan it should be reading")
	}

	// The operator now edits the LIVE object, clearing the flag.
	fls := false
	got.Spec.Baseline = &fls
	if err := h.c.Update(context.Background(), got); err != nil {
		t.Fatal(err)
	}

	// What is DELIVERED must still say baseline=true: the run that executed
	// applied no gates, and no edit can retroactively make it a certification.
	if !pinnedBaseline(h.run("run1")) {
		t.Error("delivery reports baseline=false after a mid-run edit to spec.baseline: a " +
			"thresholdless sweep is now indistinguishable from a certification, which is the " +
			"fail-open the flag exists to close")
	}
}

// The fallback, and why it is the spec rather than false.
//
// A run with no pinned plan has not started executing — buildPlan is what pins
// one — so its spec is the only statement of intent in existence. Defaulting to
// false there would report a baseline sweep as a certification before it ran,
// which is the same fail-open in a narrower window.
func TestPinnedBaseline_FallsBackToTheSpecBeforeThePlanExists(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	tru := true
	run.Spec.Baseline = &tru
	if _, ok, _ := loadPlan(run); ok {
		t.Fatal("fixture is wrong: this test needs a run with no pinned plan")
	}
	if !pinnedBaseline(run) {
		t.Error("a not-yet-started baseline run reports as a certification")
	}
	run.Spec.Baseline = nil
	if pinnedBaseline(run) {
		t.Error("a run that never asked for baseline reports as one")
	}
}

// The denominator is the pinned plan's executions, counted the way results are:
// one per target node for a Node-scope test, one for a Pair or Group test.
func TestPinnedProgress_CountsTheExecutionsThePlanOwes(t *testing.T) {
	pair := smokeTest("link")
	pair.Spec.Scope = burninv1alpha1.ScopePair
	group := smokeTest("collective")
	group.Spec.Scope = burninv1alpha1.ScopeGroup
	run := newRun("run1", "acceptance", "spark-a", "spark-b")
	p := &plan{
		Version: 1,
		Targets: []string{"spark-a", "spark-b"},
		Tests: []plannedTest{
			{Name: "smoke", Spec: smokeTest("smoke").Spec},
			{Name: "thermal", Spec: smokeTest("thermal").Spec},
			{Name: "link", Spec: pair.Spec},
			{Name: "collective", Spec: group.Spec},
		},
	}
	if err := pinPlan(run, p); err != nil {
		t.Fatal(err)
	}
	run.Status.Results = []burninv1alpha1.TestResult{
		{Name: "smoke", Phase: burninv1alpha1.RunPassed, Nodes: []string{"spark-a"}},
		{Name: "smoke", Phase: burninv1alpha1.RunRunning, Nodes: []string{"spark-b"}},
	}
	got := pinnedProgress(run)
	if got == nil {
		t.Fatal("a run with a pinned plan delivered no progress")
	}
	// Two Node-scope tests on two nodes, plus one pair and one group.
	if got.PlannedExecutions != 6 {
		t.Errorf("PlannedExecutions = %d, want 6", got.PlannedExecutions)
	}
	if want := []string{"spark-a", "spark-b"}; !reflect.DeepEqual(got.Nodes, want) {
		t.Errorf("Nodes = %v, want %v", got.Nodes, want)
	}
	if want := []string{"spark-b"}; !reflect.DeepEqual(got.CurrentNodes, want) {
		t.Errorf("CurrentNodes = %v, want %v: only the node with an execution in flight", got.CurrentNodes, want)
	}

	// A settled run has no node under test.
	run.Status.Results[1].Phase = burninv1alpha1.RunPassed
	if got := pinnedProgress(run); got == nil || len(got.CurrentNodes) != 0 || got.PlannedExecutions != 6 {
		t.Errorf("after the last execution settled, progress = %+v, want 6 planned and no current node", got)
	}
}

// A pair or group execution holds every target, and so names every target.
func TestPinnedProgress_APairInFlightNamesBothNodes(t *testing.T) {
	pair := smokeTest("link")
	pair.Spec.Scope = burninv1alpha1.ScopePair
	run := newRun("run1", "acceptance", "spark-a", "spark-b")
	if err := pinPlan(run, &plan{Version: 1, Targets: []string{"spark-a", "spark-b"},
		Tests: []plannedTest{{Name: "link", Spec: pair.Spec}}}); err != nil {
		t.Fatal(err)
	}
	run.Status.Results = []burninv1alpha1.TestResult{
		{Name: "link", Phase: burninv1alpha1.RunRunning, Nodes: []string{"spark-a", "spark-b"}},
	}
	got := pinnedProgress(run)
	if got == nil || got.PlannedExecutions != 1 || !reflect.DeepEqual(got.CurrentNodes, []string{"spark-a", "spark-b"}) {
		t.Errorf("progress = %+v, want 1 planned and both nodes current", got)
	}
}

// No pinned plan means nothing was planned, and the envelope says nothing: an
// absent progress reads as not observed where a zero would read as an empty plan.
func TestPinnedProgress_IsAbsentBeforeThePlanExists(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	if got := pinnedProgress(run); got != nil {
		t.Errorf("progress = %+v for a run with no plan, want none", got)
	}
}

// The progress reaches the consumer: every delivery of a started run carries the
// pinned denominator, and the terminal one has no node under test.
func TestDeliver_EveryDeliveryOfAStartedRunCarriesTheDenominator(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	h.finishPod(pods["spark-a"], 0, fp4Stdout, "Completed")
	h.finishPod(pods["spark-b"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	envs := h.envelopesWithReason("burnin-results", contract.ReasonPhaseChanged)
	var terminal *contract.Envelope
	for i := range envs {
		if envs[i].Phase == string(burninv1alpha1.RunPassed) {
			terminal = &envs[i]
		}
		if envs[i].Phase == string(burninv1alpha1.RunRunning) || envs[i].Phase == string(burninv1alpha1.RunPassed) {
			if envs[i].Progress == nil || envs[i].Progress.PlannedExecutions != 2 {
				t.Errorf("%s delivery progress = %+v, want 2 planned (one test on two nodes)", envs[i].Phase, envs[i].Progress)
			}
		}
	}
	if terminal == nil {
		t.Fatalf("no terminal delivery among %d phase changes", len(envs))
	}
	if got := terminal.Progress; got == nil || len(got.CurrentNodes) != 0 || !reflect.DeepEqual(got.Nodes, []string{"spark-a", "spark-b"}) {
		t.Errorf("terminal progress = %+v, want both nodes planned and none under test", got)
	}
}

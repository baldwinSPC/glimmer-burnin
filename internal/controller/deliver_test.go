package controller

import (
	"context"
	"testing"
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

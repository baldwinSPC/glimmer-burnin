package envtest

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The #84 rule, applied to the POD-TIMEOUT path — issue #247.
//
// TestNoPodIsDestroyedBeforeTheStatusJustifyingItIsDurable covers the terminate
// route: a run that reaches a terminal phase. It is the only route that route
// was ever asserted on, and there were five more sites with the ordering
// INVERTED — `advance`'s overdue branch, two in pair.go and three in group.go —
// which deleted the pod, recorded the Error in memory, and left the write to
// `step`.
//
// A lost resourceVersion race there — or an error from any LATER target in the
// same pass, since `execute` runs the whole target list before returning —
// discarded the record while the pod stayed destroyed. The consequences compound:
//
//   - The attempt never happened as far as the apiserver is concerned.
//   - `ErrorRetries` was never incremented, so the retry budget is NOT spent.
//   - The run recreates the same pod under the same attempt number, and does it
//     again on the next conflict — a loop that burns the whole deadline while
//     recording nothing, which is precisely what a rolling update of the manager
//     produces.
//   - If a runner traps SIGTERM and exits 0, the recreated pod is harvested as a
//     clean exit with partial metrics and fails closed on its thresholds: a
//     Failed hardware verdict manufactured by the controller's own kill, on the
//     one phase that is never retried.
//
// This asserts the rule on the timeout path, with a REAL conflict, which is the
// tier the fake client cannot reach.
func TestNoOverduePodIsDestroyedBeforeItsErrorIsDurable(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "overdue")
	newNode(t, node)

	// A short test whose pod will linger well past its window. The RUN's
	// deadline stays generous, so the only thing that fires is the per-pod
	// overdue check in advance — not the terminate path the sibling test covers.
	create(t,
		customTest(ns, "brief", func(bt *burninv1alpha1.BurnInTest) { bt.Spec.DurationSeconds = 30 }),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "brief"}),
		runFor(ns, "run", "acceptance", []string{node}, func(r *burninv1alpha1.BurnInRun) {
			r.Spec.DeadlineSeconds = int32p(86400)
			// One retry available, so the test can also observe whether the
			// budget is actually spent.
			r.Spec.RetryOnErrorLimit = int32p(1)
		}),
	)

	d := newDriver(t, ns)
	d.script("", script{linger: true}) // never finishes on its own

	var violations []string
	d.onPodDelete = func(pod *corev1.Pod) {
		if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			return
		}
		run := getRun(t, ns, "run")
		if justifiesDestroying(run, pod) {
			return
		}
		violations = append(violations, fmt.Sprintf(
			"overdue pod %s (phase %s) was destroyed while the apiserver still had run.phase=%q and no "+
				"terminal result for %s on %s — the attempt is unrecorded, no retry budget was spent, and "+
				"the run will recreate the same pod under the same attempt number",
			pod.Name, pod.Status.Phase, run.Status.Phase,
			pod.Labels["burnin.glimmer.ai/test"], pod.Labels["burnin.glimmer.ai/node"]))
	}

	// The competing writer, same stand-in the sibling test uses for a second
	// manager mid-rolling-update: an annotation bump that moves the run's
	// resourceVersion without changing anything the controller reads.
	compete := func() {
		for attempt := 0; attempt < 5; attempt++ {
			run := getRun(t, ns, "run")
			if run.Annotations == nil {
				run.Annotations = map[string]string{}
			}
			run.Annotations["e2e.test/competing-writer"] = time.Now().Format(time.RFC3339Nano)
			if err := admin.Update(context.Background(), run); err == nil || !apierrors.IsConflict(err) {
				return
			}
		}
	}
	// Conflict on the write that RECORDS THE OVERDUE ATTEMPT, and on nothing
	// before it. Armed after the clock advance below, because the run makes
	// several ordinary Running writes while starting up and burning the
	// injection on one of those would test the setup rather than the defect.
	armed := false
	recordingWrite := func(obj client.Object) bool {
		if !armed {
			return false
		}
		run, ok := obj.(*burninv1alpha1.BurnInRun)
		return ok && run.Status.Phase == burninv1alpha1.RunRunning
	}
	d.reconcilerOver(conflictOnce(t, recordingWrite, compete))

	// Get the pod running.
	for i := 0; i < 6; i++ {
		if _, err := d.reconcile("run"); err != nil && !apierrors.IsConflict(err) {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		d.noteDeletions("run")
		d.playKubelet("run")
	}
	if len(d.pods("run")) == 0 {
		t.Fatal("no pod was ever created, so there is nothing to time out")
	}

	// Past the pod's window, but nowhere near the run's deadline.
	d.advanceClock(10 * time.Minute)
	armed = true

	for i := 0; i < 30; i++ {
		_, err := d.reconcile("run")
		if err != nil && !apierrors.IsConflict(err) {
			t.Fatalf("post-timeout reconcile %d: %v", i, err)
		}
		d.noteDeletions("run")
		if isTerminalPhase(getRun(t, ns, "run").Status.Phase) {
			break
		}
		d.playKubelet("run")
	}

	for _, v := range violations {
		t.Error(v)
	}

	run := getRun(t, ns, "run")
	assertNoManufacturedVerdict(t, run)

	// The retry budget must actually have been spent. That is the half of this
	// defect a "no pod destroyed early" check alone would not see: an attempt
	// that was never recorded costs nothing, so the run loops on the same
	// attempt number forever without ever exhausting its retries.
	var recorded bool
	for i := range run.Status.Results {
		if run.Status.Results[i].ErrorRetries > 0 || len(run.Status.Results[i].Attempts) > 1 {
			recorded = true
		}
	}
	if !recorded && !isTerminalPhase(run.Status.Phase) {
		t.Errorf("the run is still %s with no retry recorded — a timeout that spends no budget "+
			"loops forever: %s", run.Status.Phase, summarise(run))
	}
}

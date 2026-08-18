package controller

import (
	"context"
	"testing"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The release-order invariant on the three routes that reach a release and had
// nothing driving the watcher — issue #372, Gap A.
//
// #246's rule is that a node is not made schedulable while a live pod of the
// run is still on it. releaseorder_test.go asserts it on ordinary completion
// and on CancelImmediate. Three other routes get there, all correctly ordered
// in main today, and each is one line's worth of edit away from inverting:
//
//	deadline expiry  terminate, via the run deadline rather than a cancel
//	run deletion     reconcileDeleted, which has the LEAST margin of any route
//	terminal sweep   reconcileTerminal, repeated on every pass while the
//	                 finalizer is on
//
// These use the same releaseWatcher, which checks AT THE WRITE rather than
// afterwards — afterwards is the state that looks correct however wrong the
// ordering was.

// withDeadline bounds the whole run, measured from status.startedAt.
func withDeadline(run *burninv1alpha1.BurnInRun, seconds int32) *burninv1alpha1.BurnInRun {
	run.Spec.DeadlineSeconds = int32p(seconds)
	return run
}

// twoNodeRunUnderWatch builds the common setup: two cordoned nodes, two pods
// actually running, and a watcher wired into the same fake store the
// reconciler writes through.
//
// Both pods are STARTED deliberately. A pod the fake client never started has
// no phase at all, and an earlier version of the original test checked for
// Running and therefore passed against the very ordering it was written to
// catch.
func twoNodeRunUnderWatch(t *testing.T, failFast bool, prep func(*burninv1alpha1.BurnInRun) *burninv1alpha1.BurnInRun) (*harness, *releaseWatcher) {
	t.Helper()
	w := &releaseWatcher{t: t}
	run := withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2)
	if prep != nil {
		run = prep(run)
	}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, failFast, testRef("fp4")),
		run,
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("setup: expected 2 pods, got %d", len(pods))
	}
	h.startPod(pods["spark-a"])
	h.startPod(pods["spark-b"])
	return h, w
}

// TestRun_TheDeadlineDoesNotHandBackANodeThatIsStillBurning.
//
// The deadline route reaches terminate the same way CancelImmediate does, and
// for a similar reason: the run is being stopped while its load is still on the
// hardware. It differs in who decided — the clock rather than an operator — and
// nothing asserted that the ordering survived that difference.
func TestRun_TheDeadlineDoesNotHandBackANodeThatIsStillBurning(t *testing.T) {
	h, w := twoNodeRunUnderWatch(t, false, func(r *burninv1alpha1.BurnInRun) *burninv1alpha1.BurnInRun {
		// One second, so the deadline is already blown by the time the next
		// reconcile reads it against status.startedAt.
		return withDeadline(r, 1)
	})

	h.nowVal = h.nowVal.Add(time.Hour) // well past the deadline
	h.reconcileUntilSettled("run1")

	if run := h.run("run1"); !isTerminal(run.Status.Phase) {
		t.Fatalf("phase = %q, want a terminal phase — the deadline did not fire, so this "+
			"test proved nothing about the release ordering", run.Status.Phase)
	}
	for _, v := range w.violations {
		t.Errorf("issue #246 on the DEADLINE path: %s", v)
	}
	h.assertNoStrandedCordons()
}

// TestRun_DeletingARunStopsTheLoadBeforeReturningTheNodes.
//
// This route has the least margin of any of them. The run object is the only
// thing that knows which nodes it holds, and it is being destroyed — so
// anything the finalizer does not do before it lets go is done by NOBODY. There
// is no later reconcile to correct an ordering mistake here, which is what
// makes it different from the terminal sweep below.
func TestRun_DeletingARunStopsTheLoadBeforeReturningTheNodes(t *testing.T) {
	h, w := twoNodeRunUnderWatch(t, false, nil)

	if !h.node("spark-a").Spec.Unschedulable || !h.node("spark-b").Spec.Unschedulable {
		t.Fatal("setup: nodes were not cordoned, so a release cannot be observed")
	}

	run := h.run("run1")
	if err := h.c.Delete(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.reconcileUntilSettled("run1")

	for _, v := range w.violations {
		t.Errorf("issue #246 on the DELETION path: %s", v)
	}
	if n := len(h.livePods("run1")); n != 0 {
		t.Errorf("%d pod(s) left burning hardware for a deleted run", n)
	}
	h.assertNoStrandedCordons()
}

// The terminal sweep is NOT covered here, deliberately, and this note is the
// deliverable rather than a test.
//
// I wrote one, it passed, and then it passed against reconcileTerminal with
// deleteLivePods and releaseCordons SWAPPED — which makes it worthless. The
// reason is structural: by the time the sweep runs, terminate has already
// deleted the pods and released the cordons, so there is nothing live to race
// and no cordon left to hand back. The watcher can never fire.
//
// Reaching it needs a state this harness cannot currently build: a terminal run
// whose cordon is still held AND whose pod is still live, which is what a
// FAILED deleteLivePods leaves behind. That wants a second interceptor to fail
// the first delete, and the harness takes one.
//
// Left uncovered and named, because a test that passes against broken code is
// the recurring defect class in this repository, and a vacuous test here would
// also claim the route was checked. See #372.

package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The record-before-kill rule at UNIT tier — issue #372, Gap B.
//
// #247's rule: a pod is destroyed only once the status recording WHY is
// durable. The controller implements it by collecting a pass.kill list and
// calling killPods after the status write.
//
// Its only assertion was test/envtest/podtimeout_test.go — one test, needing
// the control-plane binaries. CLAUDE.md says to write a test at the LOWEST tier
// that can observe the property, and the ordering is observable here, in
// seconds, on every PR.
//
// What breaks when the order inverts is not obvious from the code, which is why
// it is worth stating: the pod is gone and no record says it ever ran, so the
// run re-mints the SAME pod name under the SAME attempt number and waits out
// the same window again, forever, with the retry budget never charged.

// killOrderWatcher fails the test if a pod is deleted before the run's DURABLE
// status justifies it.
//
// Checked at the delete, against what a fresh Get returns — not against the
// in-memory run object, which is exactly the thing that can be ahead of the
// apiserver and is what makes the inverted order look correct from inside the
// reconciler.
type killOrderWatcher struct {
	violations []string
	deletes    int
}

func (w *killOrderWatcher) funcs(inner client.Client) interceptor.Funcs {
	return interceptor.Funcs{
		Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
			pod, isPod := obj.(*corev1.Pod)
			if !isPod {
				return c.Delete(ctx, obj, opts...)
			}
			w.deletes++

			runName := pod.Labels[labelRun]
			if runName != "" {
				var run burninv1alpha1.BurnInRun
				if err := inner.Get(ctx, client.ObjectKey{Namespace: pod.Namespace, Name: runName}, &run); err == nil {
					if !durablyAccountedFor(&run, pod) {
						w.violations = append(w.violations, fmt.Sprintf(
							"%s was destroyed before any durable attempt recorded it; "+
								"nothing in status says this pod ever ran", pod.Name))
					}
				}
			}
			return c.Delete(ctx, obj, opts...)
		},
	}
}

// durablyAccountedFor reports whether the persisted status already carries a
// FINISHED attempt for the node this pod was running on.
//
// A started-but-unfinished attempt is not enough: that is the state the pod is
// already in, and it is precisely what an inverted order leaves behind.
func durablyAccountedFor(run *burninv1alpha1.BurnInRun, pod *corev1.Pod) bool {
	node := pod.Labels[labelNode]
	for i := range run.Status.Results {
		res := &run.Status.Results[i]
		onThisNode := false
		for _, n := range res.Nodes {
			if n == node {
				onThisNode = true
			}
		}
		if !onThisNode {
			continue
		}
		for j := range res.Attempts {
			if res.Attempts[j].FinishedAt != nil {
				return true
			}
		}
	}
	return false
}

// backdate stamps a pod's CreationTimestamp, which the fake client does not set.
//
// podOverdue returns false for a zero CreationTimestamp on purpose — "overdue
// since year 1" is not a judgment — so without this the overdue path is
// unreachable at unit tier and the test would pass while exercising nothing.
// The apiserver owns this field in production; setting it here is standing in
// for the apiserver, not for the controller.
func backdate(t *testing.T, h *harness, pod *corev1.Pod, at time.Time) {
	t.Helper()
	pod.CreationTimestamp = metav1.NewTime(at)
	if err := h.c.Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
}

// withRetryLimit gives the run a retry budget, so "was it charged?" is a
// question with an answer.
func withRetryLimit(run *burninv1alpha1.BurnInRun, n int32) *burninv1alpha1.BurnInRun {
	run.Spec.RetryOnErrorLimit = int32p(n)
	return run
}

// TestKillOrder_AnOverduePodIsRecordedBeforeItIsDestroyed is the Node-scope path.
//
// The overdue check is the one #247 was written from: a pod that outlived
// duration + deadline grace + scheduling grace is condemned, and the Error
// saying so has to be durable first.
func TestKillOrder_AnOverduePodIsRecordedBeforeItIsDestroyed(t *testing.T) {
	w := &killOrderWatcher{}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a"), 1),
	)

	// Two passes: the first cordons, the second creates the pod.
	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 1 {
		t.Fatalf("setup: expected 1 pod, got %d", len(pods))
	}
	h.startPod(pods["spark-a"])
	backdate(t, h, pods["spark-a"], h.nowVal)

	// Well past duration + deadlineGrace + schedulingGrace, so podOverdue fires.
	h.nowVal = h.nowVal.Add(24 * time.Hour)
	// BOUNDED, not reconcileUntilSettled. With the clock this far ahead every
	// replacement pod is overdue the moment it is created, so the run never
	// settles — and settlement is not the property under test. What matters is
	// the ORDER of the kill, which the first pass already exercises.
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	if w.deletes == 0 {
		t.Fatal("no pod was deleted, so the overdue path never ran and this test proved nothing")
	}
	for _, v := range w.violations {
		t.Errorf("issue #247: %s", v)
	}
}

// TestKillOrder_TheRetryBudgetIsChargedForAnOverduePod.
//
// One of the two properties in #372 that is not per-path, and the one the
// original defect was actually about: when the kill happens first, ErrorRetries
// is never incremented, so the run re-mints the same pod name under the same
// attempt number and waits out the same window again. The budget must be spent.
func TestKillOrder_TheRetryBudgetIsChargedForAnOverduePod(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withRetryLimit(withNodeCap(newRun("run1", "acceptance", "spark-a"), 1), 1),
	)

	// Two passes: the first cordons, the second creates the pod.
	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 1 {
		t.Fatalf("setup: expected 1 pod, got %d", len(pods))
	}
	h.startPod(pods["spark-a"])
	backdate(t, h, pods["spark-a"], h.nowVal)
	h.nowVal = h.nowVal.Add(24 * time.Hour)
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	run := h.run("run1")
	if len(run.Status.Results) == 0 {
		t.Fatal("no result recorded for an overdue pod — the attempt vanished with the pod")
	}
	res := run.Status.Results[0]
	if res.ErrorRetries == 0 && len(res.Attempts) < 2 {
		t.Errorf("ErrorRetries=%d attempts=%d: the overdue pod cost the run nothing, so the "+
			"same pod name is re-minted under the same attempt number and the same window is "+
			"waited out again — issue #247", res.ErrorRetries, len(res.Attempts))
	}
}

// The THREE remaining paths are not covered here — all Group — and this note is
// the honest half of #372 Gap B.
//
// An earlier version of this note said the harness could not drive a rendezvous
// to its timeout, and used a SKIPPING Pair test as the evidence. That was
// wrong, and the two Pair tests below are the correction: both condemn sites go
// through podOverdue exactly like the Node one, so what was missing was
// backdate() rather than a harness capability. I read a skip as a limitation
// instead of checking what the code gates on. Recorded because the failure mode
// — a test that skips itself and is then believed — is the one this file is
// otherwise about.
//
// The fourth Pair path, the lingering server, is covered in
// killorder_linger_test.go. It is kept separate because it is the one that was
// BROKEN rather than merely unasserted: harvestPair deleted that pod inline,
// and the test failed against main until #247's ordering was applied there too.
//
// Still owed, each able to regress independently:
//
//	root timeout                     Group
//	hung collective                  Group
//	root exited before its workers   Group
//
// killOrderWatcher works unchanged for all three. What Group needs is a harness
// that can create N ranks and make rank 0 Ready, which pairPod's two-role shape
// does not express.
// ─── Pair scope ───────────────────────────────────────────────────────────────
//
// CORRECTION to the note above, which claimed the harness could not drive a
// rendezvous to its timeout. It can. Both Pair condemn sites go through
// podOverdue, exactly like the Node one, so what was missing was backdate() and
// not a harness capability. The earlier Pair attempt skipped because it never
// backdated the pods, and I read the skip as a limitation rather than checking
// what the code actually gates on.

// TestKillOrder_APairServerThatNeverBecameReadyIsRecordedBeforeItIsDestroyed.
//
// The ready-gate timeout: the server started but never reported Ready, so the
// client was never created and the link was never measured. The Error saying so
// is about the LINK, and it has to be durable before the server pod goes.
func TestKillOrder_APairServerThatNeverBecameReadyIsRecordedBeforeItIsDestroyed(t *testing.T) {
	w := &killOrderWatcher{}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"), gb10Node("spark-b"),
		pairTest("link"),
		profile("acceptance", nil, false, testRef("link")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	server := h.pairPod("run1", pairRoleServer)
	if server == nil {
		t.Fatal("setup: no server pod was created")
	}
	// STARTED but never READY — that is the gate under test. A server whose
	// container is up has not necessarily bound its socket.
	h.startPod(server)
	backdate(t, h, server, h.nowVal)

	h.nowVal = h.nowVal.Add(24 * time.Hour)
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	if w.deletes == 0 {
		t.Fatal("no pod was deleted, so the ready-gate timeout never fired and this proved nothing")
	}
	for _, v := range w.violations {
		t.Errorf("issue #247 on the Pair ready-gate path: %s", v)
	}
}

// TestKillOrder_APairClientTimeoutRecordsBeforeTakingDownBothEnds.
//
// The client timeout takes its PEER down with it, because a pair is one unit —
// which is the case the durability rule matters most for. Losing the record
// restarts the whole unit and destroys a server that was measuring fine.
func TestKillOrder_APairClientTimeoutRecordsBeforeTakingDownBothEnds(t *testing.T) {
	w := &killOrderWatcher{}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"), gb10Node("spark-b"),
		pairTest("link"),
		profile("acceptance", nil, false, testRef("link")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	server := h.pairPod("run1", pairRoleServer)
	if server == nil {
		t.Fatal("setup: no server pod was created")
	}
	// Ready this time, so the client is actually created.
	h.readyPod(server)
	h.reconcile("run1")

	client := h.pairPod("run1", pairRoleClient)
	if client == nil {
		t.Fatal("setup: the client was never created, so the client-timeout path is unreachable")
	}
	h.startPod(client)
	backdate(t, h, server, h.nowVal)
	backdate(t, h, client, h.nowVal)

	h.nowVal = h.nowVal.Add(24 * time.Hour)
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	if w.deletes < 2 {
		t.Fatalf("deletes = %d, want both ends destroyed: a client timeout takes its peer "+
			"down with it, because a pair is one unit", w.deletes)
	}
	for _, v := range w.violations {
		t.Errorf("issue #247 on the Pair client-timeout path: %s", v)
	}
}

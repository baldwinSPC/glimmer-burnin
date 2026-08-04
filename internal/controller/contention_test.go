package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Tests for issues #81, #80 and #60 — the cordon/concurrency family.
//
// THE LESSON THESE ENCODE, and it is the same one staleread_test.go opens with:
// every cordon test written before this family was diagnosed handed the
// reconciler ONE run and a truthful client, and every one of them passed against
// the build that took half a live fleet out of service. A single-run test cannot
// express contention, and a truthful client cannot express an informer that is
// behind. Both failures need a SECOND actor, so these tests always have one:
// another run, or a cache that has not caught up, or a stamp whose owner is
// already gone.

// ─── Test scaffolding for a second actor ──────────────────────────────────────

// runInformerLag is an informer that has not observed some BurnInRuns yet — the
// state a cache is in between a run being created and the watch event that
// carries it. It is the exact window the contending runs in #81 were created in.
//
// Writes and the uncached reader go straight through, as in a real manager.
type runInformerLag struct {
	client.Client
	hidden map[string]bool
}

func (l *runInformerLag) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if _, isRun := obj.(*burninv1alpha1.BurnInRun); isRun && l.hidden[key.Name] {
		return apierrors.NewNotFound(
			schema.GroupResource{Group: burninv1alpha1.GroupVersion.Group, Resource: "burninruns"}, key.Name)
	}
	return l.Client.Get(ctx, key, obj, opts...)
}

func (l *runInformerLag) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	if err := l.Client.List(ctx, list, opts...); err != nil {
		return err
	}
	runs, ok := list.(*burninv1alpha1.BurnInRunList)
	if !ok {
		return nil
	}
	kept := make([]burninv1alpha1.BurnInRun, 0, len(runs.Items))
	for i := range runs.Items {
		if !l.hidden[runs.Items[i].Name] {
			kept = append(kept, runs.Items[i])
		}
	}
	runs.Items = kept
	return nil
}

// hideRuns puts an informer that has not seen `names` between the reconciler and
// the apiserver, and wires the uncached reader to the apiserver — a manager's
// shape.
func (h *harness) hideRuns(names ...string) *runInformerLag {
	h.t.Helper()
	hidden := map[string]bool{}
	for _, n := range names {
		hidden[n] = true
	}
	lag := &runInformerLag{Client: h.c, hidden: hidden}
	h.r.Client = lag
	h.r.APIReader = h.c
	return lag
}

// admissionCondition reads the run's Admitted condition, which is where the
// CAUSE of a refusal lives. The phase says Error; only this says why.
func (h *harness) admissionCondition(name string) *metav1.Condition {
	h.t.Helper()
	return apimeta.FindStatusCondition(h.run(name).Status.Conditions, burninv1alpha1.ConditionRunAdmitted)
}

func withForce(run *burninv1alpha1.BurnInRun) *burninv1alpha1.BurnInRun {
	run.Spec.Force = boolp(true)
	return run
}

// withCreation backdates a run, so a test can state the admission order it means
// instead of relying on the fixture's.
func withCreation(run *burninv1alpha1.BurnInRun, at time.Time) *burninv1alpha1.BurnInRun {
	run.CreationTimestamp = metav1.NewTime(at)
	return run
}

// assertRefusedForBusyTargets checks the whole shape of a refusal: terminal, not
// a hardware verdict, no pod ever created, and a message a human can act on.
func (h *harness) assertRefusedForBusyTargets(name, conflictingRun string) {
	h.t.Helper()
	run := h.run(name)

	if run.Status.Phase != burninv1alpha1.RunError {
		h.t.Errorf("%s phase = %q, want Error — a refused run judged no hardware, so it is neither "+
			"Passed nor Failed", name, run.Status.Phase)
	}
	if run.Status.Failed != 0 {
		h.t.Errorf("%s counted %d FAILED test(s); a refusal must never look like a hardware verdict",
			name, run.Status.Failed)
	}
	if pods := h.allPods(name); len(pods) != 0 {
		h.t.Errorf("%s created %d pod(s) despite being refused — the load the refusal exists to prevent "+
			"landed anyway", name, len(pods))
	}
	if _, pinned, _ := loadPlan(run); pinned {
		h.t.Errorf("%s pinned a plan although it was refused", name)
	}
	for _, f := range run.Finalizers {
		if f == burninv1alpha1.FinalizerCordonCleanup {
			h.t.Errorf("%s holds the cordon finalizer although it never cordoned anything", name)
		}
	}

	cond := h.admissionCondition(name)
	if cond == nil {
		h.t.Fatalf("%s carries no %s condition; the phase says Error and nothing says why",
			name, burninv1alpha1.ConditionRunAdmitted)
	}
	if cond.Status != metav1.ConditionFalse || cond.Reason != burninv1alpha1.ReasonTargetsBusy {
		h.t.Errorf("%s Admitted condition = %s/%s, want False/%s",
			name, cond.Status, cond.Reason, burninv1alpha1.ReasonTargetsBusy)
	}
	if !strings.Contains(cond.Message, conflictingRun) {
		h.t.Errorf("%s refusal message does not name the conflicting run %q: %q — an operator reading "+
			"this has no indication the cause was another run rather than the node",
			name, conflictingRun, cond.Message)
	}

	var found bool
	for _, res := range run.Status.Results {
		if strings.Contains(res.Message, conflictingRun) {
			found = true
		}
	}
	if !found {
		h.t.Errorf("%s status.results carries no result naming %s; the reason must survive on the object, "+
			"not only in a log line", name, conflictingRun)
	}
}

// ─── #81: nothing bounded concurrency ACROSS runs ─────────────────────────────

// The reproduction from issue #81, minus the timing: a second run created while
// the first is driving the same node must not start.
//
// maxConcurrentNodes bounds ONE run's fan-out and cannot see another run at all,
// and the cordon cannot mediate it either — runner pods carry a toleration for
// node.kubernetes.io/unschedulable BY NECESSITY, since the operator cordons the
// node it is about to test, so a second run's pods schedule onto the first run's
// cordoned node without complaint. On real hardware this produced three runs all
// erroring on thermal-soak with exit 137: the kubelet killing sustained
// full-load soaks that were competing for one machine.
func TestRun_ASecondRunIsRefusedWhileTheFirstHoldsTheNodes(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		newRun("run2", "acceptance", "spark-85a9"),
	)

	// run1 starts and takes the node.
	h.reconcile("run1")
	h.reconcile("run1")
	if !h.node("spark-85a9").Spec.Unschedulable {
		t.Fatal("setup: run1 is not holding spark-85a9")
	}

	h.reconcile("run2")
	h.assertRefusedForBusyTargets("run2", "run1")

	// And the refusal cost run1 nothing: it still owns its node and its pod.
	if got := h.node("spark-85a9").Annotations[burninv1alpha1.AnnotationCordonOwner]; got != "burnin/run1/uid-run1" {
		t.Errorf("cordon owner = %q after run2 was refused, want burnin/run1/uid-run1", got)
	}
	if n := len(h.livePods("run1")); n != 1 {
		t.Errorf("run1 has %d live pod(s) after run2 was refused, want 1", n)
	}

	// run1 finishes normally and gives the node back.
	h.finishPod(h.pods("run1")["spark-85a9"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run1 phase = %q, want Passed", h.run("run1").Status.Phase)
	}
	h.assertNoStrandedCordons()
}

// The dead heat, which is the case that actually happened: three runs created in
// the same second, all Pending, none holding anything when the others look.
//
// A guard that only refuses runs which have already STARTED admits all three
// here, because at the moment each looks, none of the others has a pod. The
// order has to be total and derivable from the objects alone — no lock, no
// leader, no timing — so that whichever run the manager happens to reconcile
// first, the same one wins.
func TestRun_ConcurrentRunsAdmitExactlyOneAndAlwaysTheSameOne(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"), gb10Node("spark-043a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("burnin-a", "acceptance", "spark-85a9", "spark-043a"),
		newRun("burnin-b", "acceptance", "spark-85a9", "spark-043a"),
		newRun("burnin-c", "acceptance", "spark-85a9", "spark-043a"),
	)

	// Reconciled in the REVERSE of admission order, so a guard that merely
	// favours whoever went first would let the wrong one through — or all three.
	for _, name := range []string{"burnin-c", "burnin-b", "burnin-a"} {
		h.reconcile(name)
	}

	var admitted []string
	for _, name := range []string{"burnin-a", "burnin-b", "burnin-c"} {
		if !isTerminal(h.run(name).Status.Phase) {
			admitted = append(admitted, name)
		}
	}
	if len(admitted) != 1 || admitted[0] != "burnin-a" {
		t.Fatalf("admitted %v, want exactly [burnin-a] — three runs created in the same second all "+
			"believed the fleet was free, which is the whole failure", admitted)
	}
	h.assertRefusedForBusyTargets("burnin-b", "burnin-a")
	h.assertRefusedForBusyTargets("burnin-c", "burnin-a")
}

// The admission scan must read the APISERVER, not the informer.
//
// The runs that contend are created seconds apart — that is the entire
// reproduction — so the cache is exactly as likely to be missing the first run
// as it is to have it. A cached List answers "nothing is active", the second run
// admits itself onto hardware the first is already saturating, and the guard has
// failed in precisely the scenario it was written for. This test drives the
// reconciler through an informer that has not seen run1 at all.
func TestRun_AdmissionSeesARunTheInformerHasNotObserved(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		newRun("run2", "acceptance", "spark-85a9"),
	)

	// run1 is admitted and holding, through a truthful client.
	h.reconcile("run1")
	h.reconcile("run1")
	if !h.node("spark-85a9").Spec.Unschedulable {
		t.Fatal("setup: run1 is not holding spark-85a9")
	}

	// From here the informer has never heard of run1: no List entry, no Get.
	h.hideRuns("run1")
	h.reconcile("run2")

	h.assertRefusedForBusyTargets("run2", "run1")
}

// The negative, and it matters as much as the positive: a guard that refuses too
// much is a guard that stops the fleet being tested at all. Two runs on disjoint
// nodes contend for nothing and must both proceed.
func TestRun_RunsOnDisjointNodesBothAdmit(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"), gb10Node("spark-043a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		newRun("run2", "acceptance", "spark-043a"),
	)

	for i := 0; i < 2; i++ {
		h.reconcile("run1")
		h.reconcile("run2")
	}
	for _, name := range []string{"run1", "run2"} {
		if got := h.run(name).Status.Phase; got != burninv1alpha1.RunRunning {
			t.Fatalf("%s phase = %q, want Running — the guard refused a run that contends with nobody", name, got)
		}
		if len(h.livePods(name)) != 1 {
			t.Errorf("%s has no live pod", name)
		}
	}

	h.finishPod(h.pods("run1")["spark-85a9"], 0, fp4Stdout, "Completed")
	h.finishPod(h.pods("run2")["spark-043a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	h.reconcileUntilSettled("run2")
	for _, name := range []string{"run1", "run2"} {
		if got := h.run(name).Status.Phase; got != burninv1alpha1.RunPassed {
			t.Errorf("%s phase = %q, want Passed", name, got)
		}
	}
	h.assertNoStrandedCordons()
}

// A run only holds its targets while it is alive. Once it is terminal the nodes
// are free, and refusing on the strength of a finished run would take the fleet
// out of service permanently — the same shape of bug as the orphaned stamp.
func TestRun_ATerminalRunNeverBlocksTheNextOne(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		// Created after run1, so it would lose the tie-break if run1 still counted.
		withCreation(newRun("run2", "acceptance", "spark-85a9"), time.Unix(1750000000, 0).UTC()),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-85a9"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("setup: run1 phase = %q, want Passed", h.run("run1").Status.Phase)
	}

	h.reconcile("run2")
	h.reconcile("run2")
	if got := h.run("run2").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("run2 phase = %q, want Running — a finished run is still taking the node out of "+
			"every future run", got)
	}
	h.finishPod(h.pods("run2")["spark-85a9"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run2")
	h.assertNoStrandedCordons()
}

// A deletion is not instantaneous, and `kubectl delete burninrun` on a
// misbehaving run is exactly the moment somebody launches a replacement.
//
// While the cordon finalizer is still on, the run has not given its nodes back
// and may still have a runner at full power on one — so it still holds them, and
// the replacement waits. The finalizer, not the phase, is what says so: the run
// is already Deleting, and a phase-only guard would wave the replacement
// straight onto the hardware.
func TestRun_ARunStillReleasingItsNodesStillHoldsThem(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		newRun("run2", "acceptance", "spark-85a9"),
		newRun("run3", "acceptance", "spark-85a9"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.startPod(h.pods("run1")["spark-85a9"])

	if err := h.c.Delete(context.Background(), h.run("run1")); err != nil {
		t.Fatal(err)
	}
	// The finalizer holds deletion open: run1 still exists, still owns the
	// cordon, and its runner is still on the node.
	if h.run("run1").DeletionTimestamp.IsZero() {
		t.Fatal("setup: run1 is not being deleted")
	}
	if !h.node("spark-85a9").Spec.Unschedulable {
		t.Fatal("setup: run1 has already released the node")
	}

	h.reconcile("run2")
	h.assertRefusedForBusyTargets("run2", "run1")

	// Once the release completes and run1 is really gone, the node is free.
	h.reconcile("run1")
	if err := h.c.Get(context.Background(),
		types.NamespacedName{Namespace: "burnin", Name: "run1"}, &burninv1alpha1.BurnInRun{}); err == nil {
		t.Fatal("setup: run1 did not finish deleting")
	}

	h.reconcile("run3")
	h.reconcile("run3")
	if got := h.run("run3").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("run3 phase = %q, want Running — a run that is gone is still blocking the fleet", got)
	}
	h.finishPod(h.pods("run3")["spark-85a9"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run3")
	h.assertNoStrandedCordons()
}

// The escape hatch GEP-0178 calls for — "admission refuses busy nodes UNLESS
// FORCED" — with the record that says it was used.
//
// A verdict measured while another run was driving the same node is not
// comparable to one measured alone, and months later the condition is the only
// thing that will still say so.
func TestRun_ForceAdmitsOverAConflictAndRecordsThatItDid(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		withForce(newRun("run2", "acceptance", "spark-85a9")),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	h.reconcile("run2")
	h.reconcile("run2")

	if got := h.run("run2").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("forced run2 phase = %q, want Running — spec.force did not admit it", got)
	}
	cond := h.admissionCondition("run2")
	if cond == nil || cond.Status != metav1.ConditionTrue || cond.Reason != burninv1alpha1.ReasonRunForced {
		t.Fatalf("forced run2 Admitted condition = %+v, want True/%s — a verdict produced under "+
			"deliberate contention must be identifiable as such", cond, burninv1alpha1.ReasonRunForced)
	}
	if !strings.Contains(cond.Message, "run1") {
		t.Errorf("forced admission message does not name the run it overlapped: %q", cond.Message)
	}
}

// An admitted run says so too, so "no condition" can never be mistaken for
// "admitted" by something reading the object.
func TestRun_AnUncontestedRunRecordsThatItWasAdmitted(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
	)
	h.reconcile("run1")

	cond := h.admissionCondition("run1")
	if cond == nil || cond.Status != metav1.ConditionTrue || cond.Reason != burninv1alpha1.ReasonRunAdmitted {
		t.Fatalf("Admitted condition = %+v, want True/%s", cond, burninv1alpha1.ReasonRunAdmitted)
	}
}

// The admission order is a strict total order, checked directly because the
// consequence of it failing is silent: if it ever reports that neither of two
// runs precedes the other, BOTH admit and the interlock is gone.
func TestRun_AdmissionOrderIsATotalOrder(t *testing.T) {
	base := time.Unix(1750000000, 0).UTC()
	older := withCreation(newRun("b-older", "acceptance", "n"), base)
	newer := withCreation(newRun("a-newer", "acceptance", "n"), base.Add(time.Minute))
	// Same instant, different UIDs — the case creation timestamps cannot break,
	// since they carry only second granularity.
	tieA := withCreation(newRun("tie-a", "acceptance", "n"), base)
	tieB := withCreation(newRun("tie-b", "acceptance", "n"), base)
	started := withCreation(newRun("started", "acceptance", "n"), base.Add(time.Hour))
	startedAt := metav1.NewTime(base)
	started.Status.StartedAt = &startedAt

	cases := []struct {
		name string
		a, b *burninv1alpha1.BurnInRun
	}{
		{"older creation wins over newer", older, newer},
		{"UID breaks a same-instant tie", tieA, tieB},
		{"a started run wins however young it is", started, older},
	}
	for _, tc := range cases {
		if !admissionOrder(tc.a, tc.b) {
			t.Errorf("%s: admissionOrder(%s, %s) = false, want true", tc.name, tc.a.Name, tc.b.Name)
		}
		if admissionOrder(tc.b, tc.a) {
			t.Errorf("%s: admissionOrder is not antisymmetric for (%s, %s) — both runs would admit",
				tc.name, tc.a.Name, tc.b.Name)
		}
	}
	if admissionOrder(tieA, tieA) {
		t.Error("a run precedes itself; it would refuse itself on a re-read")
	}
}

// ─── #80: laundering another run's cordon into permanent prior state ──────────

// The mechanism from issue #80, isolated at the unit that decides it.
//
// The bug was never in the release path. Run A cordons a node it found
// schedulable; run B captures its prior state WHILE that cordon is held, reads
// spec.unschedulable=true and records it as pre-existing; run B then faithfully
// restores the node to a "prior" state that was a different run's transient
// cordon. Every overlapping run re-observes it and re-records it, and the last
// one to release also clears its own stamp — so the node ends unschedulable with
// NO annotation at all, which is strictly worse than an orphan because there is
// nothing left for a reaper to key on.
//
// A stamped node is BY DEFINITION not in its prior state, and the stamp carries
// the account of what that state was.
func TestRun_PriorStateIsReadFromTheStampNotFromAnotherRunsCordon(t *testing.T) {
	cases := []struct {
		name      string
		node      func() *corev1.Node
		wantPrior bool
	}{
		{
			name: "another run's cordon over a node it found schedulable",
			node: func() *corev1.Node {
				n := gb10Node("spark-85a9")
				n.Spec.Unschedulable = true
				n.Annotations = map[string]string{
					burninv1alpha1.AnnotationCordonOwner:        "burnin/run-a/uid-run-a",
					burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableFalse,
				}
				return n
			},
			wantPrior: false,
		},
		{
			name: "another run's cordon over a node an administrator had drained",
			node: func() *corev1.Node {
				n := gb10Node("spark-85a9")
				n.Spec.Unschedulable = true
				n.Annotations = map[string]string{
					burninv1alpha1.AnnotationCordonOwner:        "burnin/run-a/uid-run-a",
					burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableTrue,
				}
				return n
			},
			wantPrior: true,
		},
		{
			// THE OVER-APPLICATION GUARD. No stamp means no burn-in put this node
			// where it is, so the live spec IS the prior state. Reading `false`
			// here would return a node under maintenance to service.
			name: "an administrator's cordon, with no burn-in stamp",
			node: func() *corev1.Node {
				n := gb10Node("spark-85a9")
				n.Spec.Unschedulable = true
				return n
			},
			wantPrior: true,
		},
		{
			name:      "an untouched, schedulable node",
			node:      func() *corev1.Node { return gb10Node("spark-85a9") },
			wantPrior: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := preBurnInSchedulability(tc.node()); got != tc.wantPrior {
				t.Errorf("preBurnInSchedulability = %v, want %v", got, tc.wantPrior)
			}
		})
	}
}

// The same rule, end to end, over the only route by which two runs can now share
// a node at all: spec.force.
//
// Admission is the first line and force is the deliberate exception to it — so
// the laundering fix has to hold independently, or the escape hatch reintroduces
// the defect it was carved out of.
func TestRun_AForcedOverlapNeverLaundersTheOtherRunsCordon(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
		withForce(newRun("run2", "acceptance", "spark-85a9")),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	if !h.node("spark-85a9").Spec.Unschedulable {
		t.Fatal("setup: run1 is not holding spark-85a9")
	}

	// run2 captures its prior state WHILE run1's cordon is on the node. This is
	// the exact moment the three field runs recorded prior=true.
	h.reconcile("run2")

	if got, ok := h.run("run2").Status.PriorUnschedulable["spark-85a9"]; !ok || got {
		t.Fatalf("run2 status.priorUnschedulable[spark-85a9] = %v (recorded %v) — it adopted another "+
			"run's transient cordon as the node's pre-existing state, and will re-assert it at teardown "+
			"after the real owner has let go", got, ok)
	}

	// Both runs finish. The node has to come back.
	h.reconcile("run2")
	for i := 0; i < 40; i++ {
		for _, name := range []string{"run1", "run2"} {
			for _, pod := range h.livePods(name) {
				h.finishPod(pod, 0, fp4Stdout, "Completed")
			}
		}
		done := true
		for _, name := range []string{"run1", "run2"} {
			if res := h.reconcile(name); res.Requeue || res.RequeueAfter != 0 {
				done = false
			}
		}
		if done {
			break
		}
	}

	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("spark-85a9 was schedulable before either run and is cordoned now that both are over — " +
			"a transient cordon was laundered into permanent prior state, and no annotation is left " +
			"for a reaper to key on")
	}
	h.assertNoStrandedCordons()
}

// The end-state regression the issue asks for, over the fleet shape that was
// actually observed: two nodes, one of them deliberately drained beforehand, and
// overlapping runs against both.
//
// After every run reaches a terminal phase, each node's schedulability must
// equal what it was before the FIRST of them started. Both directions are
// asserted, because asserting only one would pass an operator that never
// uncordons anything, and asserting only the other would pass one that uncordons
// everything.
func TestRun_OverlappingRunsLeaveTheFleetExactlyAsTheyFoundIt(t *testing.T) {
	drained := gb10Node("spark-043a")
	drained.Spec.Unschedulable = true

	// Created in the order the field report records them, seconds apart. The two
	// later ones are forced, because that is now the only way an overlap can
	// happen at all — and the point of the test is that the escape hatch must
	// not reintroduce the corruption admission was added to prevent.
	created := time.Unix(1750000000, 0).UTC().Add(-10 * time.Minute)
	h := newHarness(t,
		gb10Node("spark-85a9"), drained,
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withCreation(withNodeCap(newRun("burnin-8fh5v", "acceptance", "spark-85a9", "spark-043a"), 2), created),
		withCreation(withForce(withNodeCap(newRun("burnin-6ffch", "acceptance", "spark-85a9", "spark-043a"), 2)),
			created.Add(11*time.Second)),
		withCreation(withForce(withNodeCap(newRun("burnin-8nzfw", "acceptance", "spark-85a9", "spark-043a"), 2)),
			created.Add(22*time.Second)),
	)
	runs := []string{"burnin-8fh5v", "burnin-6ffch", "burnin-8nzfw"}

	// THE ORDERING IS THE TEST, and it is the field timeline verbatim: the first
	// run starts and cordons both nodes, and only THEN do the other two capture
	// their prior state. Start all three together instead and each one captures
	// before any cordon exists, so the laundering never happens and the test
	// proves nothing. That is what makes this bug so easy to write a passing
	// test for.
	h.reconcile("burnin-8fh5v")
	h.reconcile("burnin-8fh5v")
	for _, name := range []string{"spark-85a9", "spark-043a"} {
		if !h.node(name).Spec.Unschedulable {
			t.Fatalf("setup: burnin-8fh5v is not holding %s", name)
		}
	}

	h.reconcile("burnin-6ffch")
	h.reconcile("burnin-8nzfw")

	// The observed corruption, named at the moment it happens rather than only
	// through its end state: `prior={"spark-043a":false,"spark-85a9":true}` on
	// the two later runs, against a first run that recorded the truth,
	// `prior={"spark-043a":true,"spark-85a9":false}`.
	for _, name := range []string{"burnin-6ffch", "burnin-8nzfw"} {
		prior := h.run(name).Status.PriorUnschedulable
		if prior["spark-85a9"] {
			t.Errorf("%s recorded spark-85a9 as already unschedulable; it observed burnin-8fh5v's "+
				"transient cordon and latched it as the node's pre-existing state", name)
		}
		if !prior["spark-043a"] {
			t.Errorf("%s recorded spark-043a as schedulable; it was drained by an administrator before "+
				"any run existed and will now be returned to service", name)
		}
	}

	for i := 0; i < 60; i++ {
		settled := true
		for _, name := range runs {
			for _, pod := range h.livePods(name) {
				h.finishPod(pod, 0, fp4Stdout, "Completed")
			}
			if res := h.reconcile(name); res.Requeue || res.RequeueAfter != 0 {
				settled = false
			}
		}
		if settled {
			break
		}
	}

	for _, name := range runs {
		if !isTerminal(h.run(name).Status.Phase) {
			t.Fatalf("%s never reached a terminal phase (%q)", name, h.run(name).Status.Phase)
		}
	}
	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("spark-85a9 went in schedulable and came out cordoned — the fleet has silently lost it, " +
			"and with no stamp left there is nothing in the cluster that knows")
	}
	if !h.node("spark-043a").Spec.Unschedulable {
		t.Error("spark-043a was drained before any run and was returned to service by the runs ending — " +
			"a node under maintenance is now taking production traffic")
	}
	h.assertNoStrandedCordons()
}

// ─── #60: a stamp naming a deleted run is never reaped ────────────────────────

// stampedNode is a node a burn-in cordoned: unschedulable, stamped with its
// owner, and carrying that owner's record of how it found the node.
func stampedNode(name, owner string, priorUnschedulable bool) *corev1.Node {
	n := gb10Node(name)
	n.Spec.Unschedulable = true
	prior := burninv1alpha1.PriorUnschedulableFalse
	if priorUnschedulable {
		prior = burninv1alpha1.PriorUnschedulableTrue
		n.Spec.Unschedulable = true
	}
	n.Annotations = map[string]string{
		burninv1alpha1.AnnotationCordonOwner:        owner,
		burninv1alpha1.AnnotationPriorUnschedulable: prior,
	}
	return n
}

func liveRun(namespace, name, uid string) *burninv1alpha1.BurnInRun {
	return &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name, UID: types.UID(uid)},
		Spec: burninv1alpha1.BurnInRunSpec{
			ProfileRef: "acceptance",
			Target:     burninv1alpha1.TargetSelector{NodeNames: []string{"spark-85a9"}},
		},
	}
}

func (h *nfpHarness) reconcileNode(node string) ctrl.Result {
	h.t.Helper()
	res, err := h.r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Name: node}})
	if err != nil {
		h.t.Fatalf("Reconcile(%s): %v", node, err)
	}
	return res
}

func nodeUpdate(before, after *corev1.Node) event.UpdateEvent {
	return event.UpdateEvent{ObjectOld: before, ObjectNew: after}
}

func (h *nfpHarness) node(name string) *corev1.Node {
	h.t.Helper()
	var n corev1.Node
	if err := h.c.Get(context.Background(), types.NamespacedName{Name: name}, &n); err != nil {
		h.t.Fatalf("get node %s: %v", name, err)
	}
	return &n
}

// The whole of issue #60: a stamp naming a run that no longer exists takes its
// node out of every FUTURE run, permanently and invisibly.
//
// The ownership rule is strict by design — never uncordon a node this run did
// not cordon — so every subsequent run correctly refuses to touch it, and the
// node is beyond the operator's reach for good. The strictness is right; the gap
// is that nothing reaps a stamp whose owner is provably gone.
//
// It is reaped from the NODE side deliberately. The BurnInRun reconciler only
// runs when some run reconciles, and a cluster with no runs at all is exactly
// the state a stranded node is left in — which is exactly when nobody is looking.
func TestReaper_AStampNamingADeletedRunIsReapedAndTheNodeComesBack(t *testing.T) {
	h := newNFPHarness(t, stampedNode("spark-85a9", "burnin/burnin-8fh5v/uid-8fh5v", false))
	h.r.APIReader = h.c

	h.reconcileNode("spark-85a9")

	node := h.node("spark-85a9")
	if node.Spec.Unschedulable {
		t.Error("spark-85a9 is still cordoned by a run that does not exist — no future run will ever " +
			"release it, because no future run owns it")
	}
	if _, ok := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; ok {
		t.Error("the orphaned ownership stamp survived the reap")
	}
	if _, ok := node.Annotations[burninv1alpha1.AnnotationPriorUnschedulable]; ok {
		t.Error("the orphaned prior-state record survived the reap")
	}
}

// The reap RESTORES, it does not uncordon. A node an administrator had drained
// before the burn-in ever saw it stays drained; only the stamp goes.
func TestReaper_ReapRestoresPriorStateRatherThanUncordoning(t *testing.T) {
	h := newNFPHarness(t, stampedNode("spark-043a", "burnin/gone/uid-gone", true))
	h.r.APIReader = h.c

	h.reconcileNode("spark-043a")

	node := h.node("spark-043a")
	if !node.Spec.Unschedulable {
		t.Error("a node that was drained before the burn-in was returned to service by the reaper — " +
			"a node under maintenance is now taking production traffic")
	}
	if _, ok := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; ok {
		t.Error("the orphaned stamp survived; the node is still unreachable by any future run")
	}
}

// The negative that keeps the reaper from becoming the bug it fixes: a stamp
// naming a run that STILL EXISTS is never touched. Reaping it would release a
// node a run is driving at full power, which is the exact violation the
// ownership rule exists to prevent.
func TestReaper_AStampNamingALiveRunIsNeverReaped(t *testing.T) {
	h := newNFPHarness(t,
		stampedNode("spark-85a9", "burnin/run1/uid-run1", false),
		liveRun("burnin", "run1", "uid-run1"),
	)
	h.r.APIReader = h.c

	res := h.reconcileNode("spark-85a9")

	node := h.node("spark-85a9")
	if !node.Spec.Unschedulable {
		t.Fatal("the reaper uncordoned a node whose owning run is alive and testing it")
	}
	if got := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; got != "burnin/run1/uid-run1" {
		t.Fatalf("cordon owner = %q; the live run can no longer release its own node", got)
	}
	if res.RequeueAfter == 0 {
		t.Error("a node with a live stamp was not scheduled for re-examination — a run disappearing " +
			"raises no Node event, so nothing but this timer will ever notice it became an orphan")
	}
}

// Names are reusable and UIDs are not. A run deleted and recreated under the
// same name is a DIFFERENT run: it never cordoned this node and must not inherit
// authority over it, so the stamp its predecessor left is an orphan.
func TestReaper_ARecreatedRunDoesNotInheritTheCordon(t *testing.T) {
	h := newNFPHarness(t,
		stampedNode("spark-85a9", "burnin/run1/uid-first", false),
		liveRun("burnin", "run1", "uid-second"),
	)
	h.r.APIReader = h.c

	h.reconcileNode("spark-85a9")

	node := h.node("spark-85a9")
	if node.Spec.Unschedulable {
		t.Error("the stamp of a deleted run survived because a DIFFERENT run took its name; the node " +
			"is out of the fleet for good")
	}
	if _, ok := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; ok {
		t.Error("the stamp was left in place, so the recreated run now appears to own a node it " +
			"never cordoned")
	}
}

// A CACHE MISS IS NOT AN ABSENCE, and this is the assertion that says so.
//
// The confirmation has to be an uncached read of the specific run. An informer
// that has not yet observed a freshly created run reports exactly the same thing
// a deleted one does, and reaping on it would hand the node's cordon away while
// a live run is driving the hardware — the failure mode the ownership rule
// exists to prevent, reintroduced by the code meant to repair it.
func TestReaper_ACacheMissIsNotAnAbsence(t *testing.T) {
	h := newNFPHarness(t,
		stampedNode("spark-85a9", "burnin/run1/uid-run1", false),
		liveRun("burnin", "run1", "uid-run1"),
	)
	// The informer has never heard of run1; the apiserver has.
	h.r.Client = &runInformerLag{Client: h.c, hidden: map[string]bool{"run1": true}}
	h.r.APIReader = h.c

	h.reconcileNode("spark-85a9")

	node := h.node("spark-85a9")
	if !node.Spec.Unschedulable {
		t.Fatal("the reaper released a node on the strength of a CACHE MISS — run1 exists and is " +
			"testing this node; the informer merely had not caught up")
	}
	if got := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; got != "burnin/run1/uid-run1" {
		t.Fatalf("cordon owner = %q; a live run's ownership was erased by a stale read", got)
	}
}

// A stamp this operator did not write is not evidence that a node is free. It is
// left exactly as found — including its cordon — because something else in the
// cluster may be relying on it.
func TestReaper_AnUnreadableStampIsLeftAlone(t *testing.T) {
	for _, stamp := range []string{"", "burnin/run1", "burnin//uid", "some-other-tool", "a/b/c/d"} {
		t.Run("stamp="+stamp, func(t *testing.T) {
			node := gb10Node("spark-85a9")
			node.Spec.Unschedulable = true
			node.Annotations = map[string]string{burninv1alpha1.AnnotationCordonOwner: stamp}

			h := newNFPHarness(t, node)
			h.r.APIReader = h.c
			h.reconcileNode("spark-85a9")

			got := h.node("spark-85a9")
			if !got.Spec.Unschedulable {
				t.Errorf("a node stamped %q was uncordoned; an unreadable stamp is not proof the node is free", stamp)
			}
			if got.Annotations[burninv1alpha1.AnnotationCordonOwner] != stamp {
				t.Errorf("stamp %q was removed by the reaper", stamp)
			}
		})
	}
}

// An unstamped node is not the reaper's business at all, and must not be woken
// for one either — every node in the cluster passes through here.
func TestReaper_AnUnstampedNodeIsNeverTouchedOrPolled(t *testing.T) {
	drained := gb10Node("spark-043a")
	drained.Spec.Unschedulable = true
	h := newNFPHarness(t, gb10Node("spark-85a9"), drained)
	h.r.APIReader = h.c

	for _, name := range []string{"spark-85a9", "spark-043a"} {
		if res := h.reconcileNode(name); res.RequeueAfter != 0 {
			t.Errorf("%s has no burn-in stamp but was scheduled for re-examination in %v", name, res.RequeueAfter)
		}
	}
	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("a schedulable node was cordoned")
	}
	if !h.node("spark-043a").Spec.Unschedulable {
		t.Error("an administratively drained node with no burn-in stamp was uncordoned")
	}
}

// The reap must not depend on the fingerprint succeeding. A node whose
// fingerprint cannot be written is still a node that may be stranded, and
// stranded capacity is the more urgent of the two jobs.
func TestReaper_ReapsEvenWhenTheFingerprintCannotBeWritten(t *testing.T) {
	// A NodeFingerprint already occupying this node's name, but describing a
	// different node — ensureFingerprint refuses to touch it and gives up.
	squatter := &burninv1alpha1.NodeFingerprint{
		ObjectMeta: metav1.ObjectMeta{Namespace: nfpNamespace, Name: "spark-85a9"},
		Spec:       burninv1alpha1.NodeFingerprintSpec{NodeName: "somebody-else"},
	}
	h := newNFPHarness(t, stampedNode("spark-85a9", "burnin/gone/uid-gone", false), squatter)
	h.r.APIReader = h.c

	h.reconcileNode("spark-85a9")

	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("the node stayed stranded because an unrelated fingerprint collision aborted the pass")
	}
}

// The stamp is not a hardware fact and must never look like drift: a burn-in
// cordoning a node cannot be allowed to make the node's identity appear to have
// changed. It IS watched, though, because that is what arms the reap timer.
func TestReaper_CordonStampArmsTheSweepWithoutLookingLikeDrift(t *testing.T) {
	h := newNFPHarness(t, gb10Node("spark-85a9"))
	h.r.APIReader = h.c
	h.reconcileNode("spark-85a9")
	baseline := h.fingerprint("spark-85a9").Status.Digest

	node := h.node("spark-85a9")
	node.Annotations = map[string]string{
		burninv1alpha1.AnnotationCordonOwner:        "burnin/run1/uid-run1",
		burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableFalse,
	}
	if err := h.c.Update(context.Background(), node); err != nil {
		t.Fatal(err)
	}
	h.reconcileNode("spark-85a9")

	if got := h.fingerprint("spark-85a9").Status.Digest; got != baseline {
		t.Errorf("digest moved from %q to %q when the operator cordoned the node — a burn-in must not "+
			"make the hardware look like it drifted", baseline, got)
	}
	if cond := drifted(h.fingerprint("spark-85a9")); cond != nil && cond.Status == metav1.ConditionTrue {
		t.Error("cordoning a node raised a hardware-drift alarm")
	}

	// And the watch predicate lets the stamp change through, or nothing would
	// ever come back to check whether it has become an orphan.
	before := gb10Node("spark-85a9")
	after := gb10Node("spark-85a9")
	after.Annotations = map[string]string{burninv1alpha1.AnnotationCordonOwner: "burnin/run1/uid-run1"}
	if !nodeIdentityChanged().Update(nodeUpdate(before, after)) {
		t.Error("a node acquiring a cordon stamp is filtered out of the watch; the reap timer is never armed")
	}
	if nodeIdentityChanged().Update(nodeUpdate(gb10Node("spark-85a9"), gb10Node("spark-85a9"))) {
		t.Error("an unchanged node woke the controller; kubelet heartbeats would flood it")
	}
}

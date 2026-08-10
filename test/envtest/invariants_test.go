package envtest

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// THE INVARIANT THIS PACKAGE WAS BUILT AROUND:
//
//	A POD MAY ONLY BE DESTROYED ONCE THE STATUS JUSTIFYING ITS DESTRUCTION IS
//	READABLE FROM THE APISERVER.
//
// The bug it comes from: on a resourceVersion conflict the terminal status was
// discarded, the run believed it was still Running, and re-harvested the pod it
// had already killed — recording exit 137 against healthy hardware. Every part
// of that needs a real apiserver to exist at all. The fake client's writes
// never conflict, its status is not a subresource, and a pod it "deletes"
// leaves no dying container behind to be misread.
//
// Reproducing the conflict deterministically is the only interesting part.
// Racing two goroutines would be a flaky approximation of it; instead the
// reconciler's client is wrapped so that ONE status write is preceded by a
// competing write from another actor. The conflict that comes back is a genuine
// 409 from the apiserver, which is exactly what the second manager of an
// overlapping rolling update produces.

// conflictOnce wraps a client so that the first status update matching `when`
// races a competing writer and therefore fails with a real conflict.
//
// It does not FABRICATE the error: it performs the competing write and lets the
// apiserver reject the stale one. A synthesised 409 would prove only that the
// controller handles a particular error value.
func conflictOnce(t *testing.T, when func(obj client.Object) bool, compete func()) client.Client {
	inner, err := client.NewWithWatch(cfg, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatalf("build watch client: %v", err)
	}
	fired := false
	return interceptor.NewClient(inner, interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, c client.Client, sub string, obj client.Object,
			opts ...client.SubResourceUpdateOption) error {
			if sub == "status" && !fired && when(obj) {
				fired = true
				compete()
			}
			return c.Status().Update(ctx, obj, opts...)
		},
	})
}

// TestNoPodIsDestroyedBeforeTheStatusJustifyingItIsDurable is the headline
// assertion.
//
// The scenario is the one that produced the bug: a run with an in-flight pod
// hits its deadline, so the controller has to stop the hardware AND record why.
// A competing writer takes the resourceVersion out from under the terminal
// status write, so that write fails. What must not happen is the pod being gone
// while the apiserver still believes the run is Running — because the next pass
// then finds a container the controller itself killed and reads its exit code
// as a verdict about the node.
func TestNoPodIsDestroyedBeforeTheStatusJustifyingItIsDurable(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "durable")
	newNode(t, node)

	create(t,
		customTest(ns, "soak", func(bt *burninv1alpha1.BurnInTest) { bt.Spec.DurationSeconds = 600 }),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "soak"}),
		runFor(ns, "run", "acceptance", []string{node}, func(r *burninv1alpha1.BurnInRun) {
			r.Spec.DeadlineSeconds = int32p(30)
		}),
	)

	d := newDriver(t, ns)
	d.script("", script{linger: true}) // a soak that is still running when the deadline lands

	var violations []string
	d.onPodDelete = func(pod *corev1.Pod) {
		// The pod finished on its own; nothing destroyed it.
		if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			return
		}
		run := getRun(t, ns, "run")
		if justifiesDestroying(run, pod) {
			return
		}
		violations = append(violations, fmt.Sprintf(
			"pod %s (phase %s) was destroyed while the apiserver still had run.phase=%q and "+
				"no terminal result for %s on %s — the next pass will re-harvest a container this "+
				"controller killed and read its exit code as a hardware verdict",
			pod.Name, pod.Status.Phase, run.Status.Phase,
			pod.Labels["burnin.glimmer.ai/test"], pod.Labels["burnin.glimmer.ai/node"]))
	}

	// The competing writer: a bare annotation bump on the run, which moves its
	// resourceVersion without changing anything the controller cares about.
	// That is the smallest faithful stand-in for a second manager mid-reconcile.
	compete := func() {
		for attempt := 0; attempt < 5; attempt++ {
			run := getRun(t, ns, "run")
			if run.Annotations == nil {
				run.Annotations = map[string]string{}
			}
			run.Annotations["e2e.test/competing-writer"] = time.Now().Format(time.RFC3339Nano)
			err := admin.Update(context.Background(), run)
			if err == nil || !apierrors.IsConflict(err) {
				return
			}
		}
	}
	terminalWrite := func(obj client.Object) bool {
		run, ok := obj.(*burninv1alpha1.BurnInRun)
		return ok && isTerminalPhase(run.Status.Phase)
	}
	d.reconcilerOver(conflictOnce(t, terminalWrite, compete))

	// Get the pod running, then let the deadline pass under it.
	for i := 0; i < 6; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		d.noteDeletions("run")
		d.playKubelet("run")
	}
	if len(d.pods("run")) == 0 {
		t.Fatal("no pod was ever created, so there is nothing for the deadline to interrupt")
	}
	d.advanceClock(60 * time.Second)

	// From here the run is over its deadline. The first terminal status write
	// hits the injected conflict; the loop must still converge, and must not
	// destroy the pod before the verdict is durable.
	for i := 0; i < 30; i++ {
		_, err := d.reconcile("run")
		if err != nil && !apierrors.IsConflict(err) {
			t.Fatalf("post-deadline reconcile %d: %v", i, err)
		}
		d.noteDeletions("run")
		if isTerminalPhase(getRun(t, ns, "run").Status.Phase) {
			break
		}
		d.playKubelet("run")
	}

	run := getRun(t, ns, "run")
	for _, v := range violations {
		t.Error(v)
	}

	// The run must land on the phase the deadline calls for. Error, not Failed:
	// a run that ran out of time did not judge the hardware it never reached.
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("phase = %q, want Error (a deadline is not a hardware verdict): %s",
			run.Status.Phase, summarise(run))
	}
	assertNoManufacturedVerdict(t, run)
	// And the hardware must actually be given back, conflict or no conflict.
	for i := 0; i < 5 && len(d.pods("run")) > 0; i++ {
		if _, err := d.reconcile("run"); err != nil && !apierrors.IsConflict(err) {
			t.Fatalf("convergence reconcile: %v", err)
		}
	}
	if n := len(livePods(d.pods("run"))); n != 0 {
		t.Errorf("%d pod(s) are still running after the run reached %s — the run is no longer accounting for that load",
			n, run.Status.Phase)
	}
	assertFullyReleased(t, getRun(t, ns, "run"), node)
}

// TestAnImmediateCancelUnderConflictNeverBecomesAHardwareVerdict is the same
// bug from the direction an operator is most likely to meet it.
//
// CancelImmediate exists for a facility power or cooling event: the load itself
// is the reason for stopping, so the pods go now. Executions killed that way are
// recorded as Cancelled — the operator stopped them, the hardware did not fail
// them — and a lost status write must not be able to turn one into a Failed or
// an Error about the part.
func TestAnImmediateCancelUnderConflictNeverBecomesAHardwareVerdict(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "cancelled")
	newNode(t, node)

	create(t,
		customTest(ns, "soak", func(bt *burninv1alpha1.BurnInTest) { bt.Spec.DurationSeconds = 600 }),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "soak"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.script("", script{linger: true})

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
	d.reconcilerOver(conflictOnce(t, func(obj client.Object) bool {
		run, ok := obj.(*burninv1alpha1.BurnInRun)
		return ok && run.Status.Phase == burninv1alpha1.RunCancelled
	}, compete))

	for i := 0; i < 6; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		d.playKubelet("run")
	}
	if n := len(livePods(d.pods("run"))); n == 0 {
		t.Fatal("no live pod to cancel")
	}

	run := getRun(t, ns, "run")
	run.Spec.Cancel = boolp(true)
	run.Spec.CancelPolicy = burninv1alpha1.CancelImmediate
	run.Spec.CancelReason = "facility power event"
	if err := admin.Update(context.Background(), run); err != nil {
		t.Fatalf("request cancel: %v", err)
	}

	for i := 0; i < 30; i++ {
		_, err := d.reconcile("run")
		if err != nil && !apierrors.IsConflict(err) {
			t.Fatalf("cancel reconcile %d: %v", i, err)
		}
		if isTerminalPhase(getRun(t, ns, "run").Status.Phase) {
			break
		}
		// A pod the controller deleted dies; the kubelet would report the
		// signal. Reproducing that is the whole point — a re-harvest of this
		// pod is what manufactured the false verdict.
		killDeletedPods(t, d.pods("run"))
		d.playKubelet("run")
	}

	final := getRun(t, ns, "run")
	if final.Status.Phase != burninv1alpha1.RunCancelled {
		t.Errorf("phase = %q, want Cancelled — an operator's decision must never read as a hardware verdict: %s",
			final.Status.Phase, summarise(final))
	}
	assertNoManufacturedVerdict(t, final)
	assertFullyReleased(t, final, node)
}

// TestATerminalRunNeverGoesBackToRunning states the property that makes a
// verdict trustworthy at all.
//
// A terminal phase is FINAL. Once a consumer has read Passed off this object,
// nothing — a late pod event, a spec edit, a second manager still draining its
// in-flight reconciles — may move it. The competing writer here plays that
// second manager.
func TestATerminalRunNeverGoesBackToRunning(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "final")
	newNode(t, node)

	create(t,
		customTest(ns, "smoke", nil),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "smoke"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.script("", script{stdout: "miscompares=0\n"})
	d.reconcilerOver(admin)
	run := d.run("run", 40)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}

	// Everything a late event could plausibly look like, applied after the
	// verdict: a spec edit, a cancel request, and more reconciles.
	late := getRun(t, ns, "run")
	late.Spec.Cancel = boolp(true)
	late.Spec.CancelReason = "arrived after the verdict"
	if err := admin.Update(context.Background(), late); err != nil {
		t.Fatalf("late spec edit: %v", err)
	}
	for i := 0; i < 5; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("post-terminal reconcile %d: %v", i, err)
		}
		if got := getRun(t, ns, "run").Status.Phase; got != burninv1alpha1.RunPassed {
			t.Fatalf("phase moved to %q after the run was already Passed", got)
		}
	}
}

// ─── helpers ──────────────────────────────────────────────────────────────────

// justifiesDestroying reports whether the run's DURABLE status explains why a
// still-executing pod was destroyed.
//
// Three things justify it, and nothing else does: the run has reached a
// terminal phase (its verdict is recorded, and a terminal run must not keep
// burning hardware); the execution this pod belongs to already has a terminal
// result (the attempt has been judged and the pod is now just load); or THIS
// POD's own attempt is recorded as finished.
//
// The third was added with issue #247 and is not a loosening. An attempt that
// errors with retry budget left does NOT settle its result — completeAttempt
// increments ErrorRetries and leaves the result open for the next attempt — so
// on the timeout path the run stays Running and the result stays non-terminal
// even after the record is durable. What makes the pod safe to destroy there is
// that the apiserver already knows this attempt happened: the retry is counted,
// the next pass will not repeat it under the same number, and nothing can
// re-harvest the container as a fresh execution. Without this case the rule
// would be unsatisfiable on any retryable timeout, which is most of them.
func justifiesDestroying(run *burninv1alpha1.BurnInRun, pod *corev1.Pod) bool {
	if isTerminalPhase(run.Status.Phase) {
		return true
	}
	res := resultFor(run, pod.Labels["burnin.glimmer.ai/test"], pod.Labels["burnin.glimmer.ai/node"])
	if res == nil {
		return false
	}
	if res.Phase.IsTerminal() {
		return true
	}
	for i := range res.Attempts {
		a := &res.Attempts[i]
		if a.PodName == pod.Name && a.Phase.IsTerminal() {
			return true
		}
	}
	return false
}

// assertNoManufacturedVerdict checks that no result blames the hardware for
// something the controller did to it. A signal death (128+n) recorded as a
// verdict is the exact fingerprint of the bug: the controller killed the pod
// and then read its exit code.
func assertNoManufacturedVerdict(t *testing.T, run *burninv1alpha1.BurnInRun) {
	t.Helper()
	for _, res := range run.Status.Results {
		if res.Phase == burninv1alpha1.RunFailed {
			t.Errorf("result %q on %v is Failed, which is a claim about the hardware — "+
				"nothing in this run measured a fault: %s", res.Name, res.Nodes, res.Message)
		}
		for _, a := range res.Attempts {
			if a.ExitCode == nil {
				continue
			}
			switch *a.ExitCode {
			case 137, 143: // SIGKILL, SIGTERM
				t.Errorf("result %q on %v recorded exit %d — that is the controller's own kill "+
					"being read back as evidence about the node: %s",
					res.Name, res.Nodes, *a.ExitCode, res.Message)
			}
		}
	}
}

func livePods(pods []corev1.Pod) []corev1.Pod {
	var out []corev1.Pod
	for _, pod := range pods {
		switch pod.Status.Phase {
		case corev1.PodSucceeded, corev1.PodFailed:
		default:
			out = append(out, pod)
		}
	}
	return out
}

// killDeletedPods gives a pod that is being deleted the ending a kubelet would:
// the container is SIGKILLed and reports 137. envtest has no kubelet, and a pod
// under graceful deletion is exactly the object the buggy re-harvest read.
func killDeletedPods(t *testing.T, pods []corev1.Pod) {
	t.Helper()
	ctx := context.Background()
	for i := range pods {
		pod := pods[i]
		if pod.DeletionTimestamp == nil || pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			continue
		}
		pod.Status.Phase = corev1.PodFailed
		pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
			Name: "runner",
			State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
				ExitCode: 137,
				Reason:   "Error",
			}},
		}}
		if err := admin.Status().Update(ctx, &pod); err != nil && !apierrors.IsNotFound(err) {
			t.Fatalf("kill pod %s: %v", pod.Name, err)
		}
	}
}

//go:build e2e

package e2e

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Chaos: what happens to a run when the thing reconciling it goes away.
//
// The reconciler is level-based and keeps nothing in process memory, so in
// principle a restart is free. In practice three of the five bugs that reached
// production were about exactly this seam — a cordon inherited from a previous
// incarnation and made permanent, an empty harvest from a pod that was SIGTERMed
// out from under the controller, and a terminal status discarded on a conflict
// so the run re-harvested a pod it had already killed.
//
// Two flavours, and the difference matters:
//
//   - killing the manager pod is a RESTART. One reconciler, gone and replaced,
//     with a gap in between;
//   - rolling the Deployment is an OVERLAP. replicas:1 with the default
//     RollingUpdate strategy has maxSurge 25%, which rounds up to 1, so an
//     apply starts the new manager BEFORE the old one is gone. Two managers,
//     both with live caches, both draining work. That is the condition the
//     status-conflict bug needed, and a plain restart does not produce it.
//
// The deterministic form of the same property lives in test/envtest, where a
// competing writer is injected at an exact point instead of being raced for.
// This is the cluster-level confirmation: it proves the scenario is REAL, and
// the envtest suite proves the invariant holds at every point.

// TestKillingTheManagerMidRunStrandsNothing restarts the operator while a node
// is under load.
//
// `kubectl delete pod` on a controller is the most natural reaction to it
// behaving oddly, and it is therefore the likeliest moment to strand a node:
// the only object that knows which nodes are held is the run, and the only
// thing that reads it is the reconciler that just died.
func TestKillingTheManagerMidRunStrandsNothing(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]
	watcher := watchCordons(t, node)

	create(t,
		shellTest(ns, "soak", slowRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 90
			bt.Spec.CheckpointIntervalSeconds = int32p(5)
			bt.Spec.Thresholds = []burninv1alpha1.Threshold{
				{Metric: "miscompares", Comparison: burninv1alpha1.EQ, Value: "0"},
			}
		}),
		profileFor(ns, "acceptance", nil, "soak"),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	awaitPodRunning(t, ns, "run")
	before := managerPodNames(t)

	// The kill. The Deployment brings a replacement up; the new manager has to
	// re-derive everything from cluster state, because the old one persisted
	// nothing else.
	if err := k8s.DeleteAllOf(context.Background(), &corev1.Pod{},
		client.InNamespace(managerNamespace),
		client.MatchingLabels{"control-plane": managerSelector}); err != nil {
		t.Fatalf("kill the manager: %v", err)
	}
	awaitManagerReplaced(t, before, 3*time.Minute)
	awaitManagerAvailable(t, 3*time.Minute)

	run := awaitTerminal(t, ns, "run", 8*time.Minute)

	// A healthy runner interrupted by nothing but a controller restart must
	// still pass. This is where the empty-harvest bug landed: a pod SIGTERMed
	// out from under the controller yielded zero metrics, and fail-closed
	// evaluation turned that absence into a hardware verdict.
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — the hardware was fine and only the operator restarted: %s",
			run.Status.Phase, summarise(run))
	}
	assertNoManufacturedVerdict(t, run)
	assertExactlyOnePodPerAttempt(t, ns, "run")
	if !watcher.sawCordon(node) {
		t.Errorf("node %s was never cordoned, so the restart did not happen while the run held anything", node)
	}
	assertNoStrandedCordon(t, ns, "run", node)
}

// TestARollingUpdateOverlapsManagersAndStillProducesOneHonestVerdict is the
// overlap case, and the one the status-conflict bug actually needed.
//
// The shipped Deployment is replicas:1 under the default RollingUpdate
// strategy, so applying it starts a second manager before the first is gone.
// Both hold live caches of the same run for the length of the changeover, and
// the loser of a resourceVersion race gets a 409 on a status write it had
// already acted on. What must never come out the other side is a verdict about
// hardware that nothing measured.
func TestARollingUpdateOverlapsManagersAndStillProducesOneHonestVerdict(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]
	watcher := watchCordons(t, node)
	overlap := watchManagerOverlap(t)

	create(t,
		shellTest(ns, "soak", slowRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 90
			bt.Spec.CheckpointIntervalSeconds = int32p(5)
			bt.Spec.Thresholds = []burninv1alpha1.Threshold{
				{Metric: "miscompares", Comparison: burninv1alpha1.EQ, Value: "0"},
			}
		}),
		profileFor(ns, "acceptance", nil, "soak"),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	awaitPodRunning(t, ns, "run")

	// A rollout, triggered the way `kubectl rollout restart` does it: an
	// annotation on the pod template.
	dep := managerDeployment(t)
	if dep.Spec.Template.Annotations == nil {
		dep.Spec.Template.Annotations = map[string]string{}
	}
	dep.Spec.Template.Annotations["e2e.burnin.glimmer.ai/restartedAt"] = time.Now().Format(time.RFC3339Nano)
	if err := k8s.Update(context.Background(), dep); err != nil {
		t.Fatalf("roll the manager: %v", err)
	}
	awaitManagerAvailable(t, 4*time.Minute)

	run := awaitTerminal(t, ns, "run", 8*time.Minute)

	// If the two managers never coexisted, this test proved nothing about
	// overlap — say so rather than claim a pass. It is reported and not failed
	// because a fast enough rollout is a scheduling accident, not a regression.
	if !overlap.observed() {
		t.Log("NOTE: two managers were never observed at once, so this run exercised a restart " +
			"rather than an overlap. The deterministic overlap case is covered by " +
			"test/envtest/invariants_test.go, which injects the conflict instead of racing for it.")
	}

	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — only the operator changed: %s", run.Status.Phase, summarise(run))
	}
	assertNoManufacturedVerdict(t, run)
	assertExactlyOnePodPerAttempt(t, ns, "run")
	if !watcher.sawCordon(node) {
		t.Errorf("node %s was never cordoned, so the rollout did not overlap anything the run was holding", node)
	}
	assertNoStrandedCordon(t, ns, "run", node)
}

// TestDeletingARunMidFlightReleasesTheNodeAndStopsTheLoad is the operator's own
// escape hatch, tested against a real kubelet.
//
// Without the finalizer, deleting the run removes the only object that knows
// which nodes are held. With it, deletion blocks until the last cordon is
// released — and the pod has to actually stop, because load left behind belongs
// to no run at all.
func TestDeletingARunMidFlightReleasesTheNodeAndStopsTheLoad(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]

	create(t,
		// A runner that never finishes: the assertion is about what happens to
		// load that is still on the floor when the run is taken away.
		shellTest(ns, "soak", lingeringRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 300
		}),
		profileFor(ns, "acceptance", nil, "soak"),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	awaitPodRunning(t, ns, "run")
	eventually(t, 2*time.Minute, "the node to be cordoned", func() error {
		if getNode(t, node).Spec.Unschedulable {
			return nil
		}
		return fmt.Errorf("node %s is still schedulable", node)
	})

	if err := k8s.Delete(context.Background(), getRun(t, ns, "run")); err != nil {
		t.Fatalf("delete run: %v", err)
	}

	eventually(t, 3*time.Minute, "the run to finish deleting", func() error {
		var run burninv1alpha1.BurnInRun
		err := k8s.Get(context.Background(),
			client.ObjectKey{Namespace: ns, Name: "run"}, &run)
		if err == nil {
			return fmt.Errorf("still present with finalizers %v and cordons %v",
				run.Finalizers, run.Status.CordonedNodes)
		}
		return nil
	})

	if getNode(t, node).Spec.Unschedulable {
		t.Errorf("node %s is still cordoned after its run was deleted — nothing in the cluster now "+
			"knows it was taken", node)
	}
	eventually(t, 2*time.Minute, "the runner pod to stop", func() error {
		var pods corev1.PodList
		if err := k8s.List(context.Background(), &pods, client.InNamespace(ns)); err != nil {
			return err
		}
		for i := range pods.Items {
			switch pods.Items[i].Status.Phase {
			case corev1.PodSucceeded, corev1.PodFailed:
			default:
				return fmt.Errorf("pod %s is still %s", pods.Items[i].Name, pods.Items[i].Status.Phase)
			}
		}
		return nil
	})
}

// ─── chaos helpers ────────────────────────────────────────────────────────────

// awaitPodRunning waits until the run has a runner pod actually executing, so
// the chaos lands mid-test rather than before anything started.
func awaitPodRunning(t *testing.T, ns, runName string) {
	t.Helper()
	eventually(t, 4*time.Minute, "a runner pod to be Running", func() error {
		var pods corev1.PodList
		if err := k8s.List(context.Background(), &pods, client.InNamespace(ns),
			client.MatchingLabels{"burnin.glimmer.ai/run": runName}); err != nil {
			return err
		}
		for i := range pods.Items {
			if pods.Items[i].Status.Phase == corev1.PodRunning {
				return nil
			}
		}
		return fmt.Errorf("%s", podDiagnosis(t, ns, runName))
	})
}

func managerPodNames(t *testing.T) map[string]bool {
	t.Helper()
	out := map[string]bool{}
	for _, pod := range managerPods(t) {
		out[pod.Name] = true
	}
	return out
}

// awaitManagerReplaced waits until at least one manager pod exists that is not
// one of the originals.
func awaitManagerReplaced(t *testing.T, before map[string]bool, timeout time.Duration) {
	t.Helper()
	eventually(t, timeout, "the manager to be replaced", func() error {
		for _, pod := range managerPods(t) {
			if !before[pod.Name] && pod.Status.Phase == corev1.PodRunning {
				return nil
			}
		}
		return fmt.Errorf("no new manager pod is Running yet")
	})
}

// managerOverlapWatcher records whether two manager pods were ever alive at the
// same moment. Without that, a rollout test is only a restart test wearing a
// different name.
type managerOverlapWatcher struct {
	seen chan struct{}
	stop chan struct{}
	done chan struct{}
	hit  bool
}

func watchManagerOverlap(t *testing.T) *managerOverlapWatcher {
	t.Helper()
	w := &managerOverlapWatcher{seen: make(chan struct{}, 1), stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(w.done)
		for {
			select {
			case <-w.stop:
				return
			case <-time.After(200 * time.Millisecond):
			}
			var pods corev1.PodList
			if err := k8s.List(context.Background(), &pods, client.InNamespace(managerNamespace),
				client.MatchingLabels{"control-plane": managerSelector}); err != nil {
				continue
			}
			live := 0
			for i := range pods.Items {
				if pods.Items[i].DeletionTimestamp == nil &&
					pods.Items[i].Status.Phase == corev1.PodRunning {
					live++
				}
			}
			if live > 1 {
				select {
				case w.seen <- struct{}{}:
				default:
				}
			}
		}
	}()
	t.Cleanup(func() { w.finish() })
	return w
}

func (w *managerOverlapWatcher) finish() {
	select {
	case <-w.stop:
	default:
		close(w.stop)
		<-w.done
		select {
		case <-w.seen:
			w.hit = true
		default:
		}
	}
}

func (w *managerOverlapWatcher) observed() bool {
	w.finish()
	return w.hit
}

// assertExactlyOnePodPerAttempt is the restart-recovery invariant.
//
// Pod names are derived from (run UID, test index, node, attempt) precisely so
// that a controller which restarted mid-test finds the pod it already created
// instead of starting the test a second time. Two pods for one attempt means
// the hardware was burned twice and one of the two results is about an
// execution nobody is tracking.
func assertExactlyOnePodPerAttempt(t *testing.T, ns, runName string) {
	t.Helper()
	var pods corev1.PodList
	if err := k8s.List(context.Background(), &pods, client.InNamespace(ns),
		client.MatchingLabels{"burnin.glimmer.ai/run": runName}); err != nil {
		t.Fatalf("list pods: %v", err)
	}
	seen := map[string][]string{}
	for i := range pods.Items {
		pod := &pods.Items[i]
		key := fmt.Sprintf("%s/%s/attempt=%s/%s",
			pod.Labels["burnin.glimmer.ai/test"],
			pod.Labels["burnin.glimmer.ai/node"],
			pod.Labels["burnin.glimmer.ai/attempt"],
			pod.Labels["burnin.glimmer.ai/pair-role"])
		seen[key] = append(seen[key], pod.Name)
	}
	for key, names := range seen {
		if len(names) > 1 {
			t.Errorf("%d pods exist for a single execution (%s): %v — the restarted manager started a "+
				"test that was already running", len(names), key, names)
		}
	}
}

package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Tests for issue #84 and #59, and for the assumption underneath both of them.
//
// THE LESSON THESE ENCODE. Every cordon test in this package before now handed
// the reconciler a client whose Get is truthful, and every one of them passed
// against the build that stranded a node on real hardware. The fake client is
// an apiserver; a manager does not talk to an apiserver, it talks to an
// informer, and an informer is a cache that is allowed to be behind. A test
// that models `Get` as truth cannot express "the controller was told something
// that used to be true", which is the entire failure. So these tests model the
// cache explicitly and drive the reconciler through it.
//
// What they deliberately do NOT model is a cache that lags forever. It lags for
// a window and then catches up, which is why each of these thaws: the bug is
// what the controller PERSISTS from the reading it took inside that window, and
// a test that left the cache stale for the whole run would be testing something
// no cluster does.

// staleCache is an informer that is behind the apiserver for the objects it has
// been given, and current for everything else. Writes and the uncached reader
// go straight through, so a write made from a stale read still meets the
// apiserver's optimistic concurrency, exactly as in a real manager.
type staleCache struct {
	client.Client
	// nodes is what the informer still believes, by node name. A nil map means
	// the informer has caught up.
	nodes map[string]*corev1.Node
	// hidden are nodes the informer has not observed AT ALL — the state a List
	// is in between a write and the watch event that carries it.
	hidden map[string]bool
}

func (s *staleCache) thaw() { s.nodes, s.hidden = nil, nil }

func (s *staleCache) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if node, ok := obj.(*corev1.Node); ok {
		if s.hidden[key.Name] {
			return apierrors.NewNotFound(schema.GroupResource{Resource: "nodes"}, key.Name)
		}
		if cached, ok := s.nodes[key.Name]; ok {
			cached.DeepCopyInto(node)
			return nil
		}
	}
	return s.Client.Get(ctx, key, obj, opts...)
}

func (s *staleCache) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	if err := s.Client.List(ctx, list, opts...); err != nil {
		return err
	}
	nodes, ok := list.(*corev1.NodeList)
	if !ok {
		return nil
	}
	kept := make([]corev1.Node, 0, len(nodes.Items))
	for i := range nodes.Items {
		name := nodes.Items[i].Name
		if s.hidden[name] {
			continue
		}
		if cached, ok := s.nodes[name]; ok {
			kept = append(kept, *cached.DeepCopy())
			continue
		}
		kept = append(kept, nodes.Items[i])
	}
	nodes.Items = kept
	return nil
}

// throughCache puts an informer between the reconciler and the apiserver, and
// wires the uncached reader to the apiserver itself — the shape a manager has.
func (h *harness) throughCache() *staleCache {
	h.t.Helper()
	sc := &staleCache{Client: h.c}
	h.r.Client = sc
	h.r.APIReader = h.c
	return sc
}

// A run must not inherit the PREVIOUS run's cordon as this node's pre-existing
// state.
//
// The reproduction is the one from issue #84: a schedule fires runs
// back-to-back, so by the time run N starts, run N-1 cordoned and released this
// same node seconds ago. The apiserver has the released version. The informer
// has not caught up, and still holds the cordoned one.
//
// Capturing at the right MOMENT — markRunning, before this run's first cordon —
// is not enough on its own, and that is the whole point: the contamination
// arrives through the cache from a run that has already finished, so no amount
// of care about ordering within THIS run can see it. The capture has to read
// the apiserver.
func TestRun_PriorSchedulabilityIsNotInheritedFromAStaleCache(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
	)
	sc := h.throughCache()

	// The apiserver's truth: schedulable. The informer's memory: the cordon the
	// previous run placed and has already taken off again.
	stale := h.node("spark-85a9").DeepCopy()
	stale.Spec.Unschedulable = true
	stale.Annotations = map[string]string{
		burninv1alpha1.AnnotationCordonOwner:        "burnin/previous-run/uid-previous",
		burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableFalse,
	}
	sc.nodes = map[string]*corev1.Node{"spark-85a9": stale}

	// Pending → Running is where the capture happens.
	h.reconcile("run1")
	sc.thaw()

	if got, ok := h.run("run1").Status.PriorUnschedulable["spark-85a9"]; !ok || got {
		t.Fatalf("status.priorUnschedulable[spark-85a9] = %v (recorded %v) — the run believes a node "+
			"it found SCHEDULABLE was already cordoned, and will make that permanent at teardown", got, ok)
	}

	h.reconcile("run1")
	pods := h.pods("run1")
	h.startPod(pods["spark-85a9"])
	h.finishPod(pods["spark-85a9"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("spark-85a9 was left unschedulable — the fleet has silently lost a node that was " +
			"schedulable when the run found it")
	}
	h.assertNoStrandedCordons()
}

// The same capture, through the reading the test above cannot reach: a node the
// informer remembers as unschedulable with NO stamp on it.
//
// The test above passes whether capturePriorSchedulability reads the cache or
// the apiserver, and that is not a spare assertion — it is a hole. Its stale
// node carries a well-formed stamp, so preBurnInSchedulability answers from the
// STAMP and never looks at spec.unschedulable; the cache could be arbitrarily
// far behind and the answer would still be right. The reading that has nothing
// to correct it is the unstamped one, and an administrator's drain lifted
// moments before the run started is exactly that: the informer holds
// unschedulable=true with no annotations, so prior=true is recorded about a node
// that is schedulable, and teardown makes it permanent. A cordon nobody placed,
// held forever, with no stamp left for the reaper to key on.
func TestRun_PriorSchedulabilityIsNotTakenFromAStaleUnstampedCordon(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
	)
	sc := h.throughCache()

	stale := h.node("spark-85a9").DeepCopy()
	stale.Spec.Unschedulable = true
	sc.nodes = map[string]*corev1.Node{"spark-85a9": stale}

	h.reconcile("run1")
	sc.thaw()

	if got, ok := h.run("run1").Status.PriorUnschedulable["spark-85a9"]; !ok || got {
		t.Fatalf("status.priorUnschedulable[spark-85a9] = %v (recorded %v) — the run took a drain "+
			"that had already been lifted for the node's pre-existing state", got, ok)
	}

	h.reconcile("run1")
	pods := h.pods("run1")
	h.startPod(pods["spark-85a9"])
	h.finishPod(pods["spark-85a9"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("spark-85a9 was left unschedulable, restoring a cordon that was already gone when " +
			"the run arrived — and with the stamp removed, nothing is left to key a reaper on")
	}
	h.assertNoStrandedCordons()
}

// The release must find the nodes this run is holding even when the informer
// has not yet observed that it stamped them. Issue #59.
//
// releaseCordons scans nodes by annotation rather than walking
// status.CordonedNodes precisely so that a manager which died mid-sequence
// still finds what it left behind. Scanning a CACHE reintroduces the hole it
// was closing: the stamp is written and the watch event has not arrived, the
// scan comes up empty, the finalizer comes off, and the node is left cordoned
// with an owner annotation naming a run that will never step again.
func TestRun_ReleaseFindsACordonTheInformerHasNotSeen(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-85a9"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-85a9"),
	)
	sc := h.throughCache()

	h.reconcile("run1")
	h.reconcile("run1")
	if !h.node("spark-85a9").Spec.Unschedulable {
		t.Fatal("precondition: the run should be holding spark-85a9 by now")
	}

	// From here the informer is stuck on the version it had BEFORE the cordon:
	// no owner stamp, still schedulable. That is what a scan of the cache would
	// find, and it matches nothing.
	preCordon := gb10Node("spark-85a9")
	preCordon.ResourceVersion = h.node("spark-85a9").ResourceVersion
	sc.nodes = map[string]*corev1.Node{"spark-85a9": preCordon}

	// `kubectl delete burninrun` on a run that is misbehaving — the most likely
	// moment for a node to be stranded, and the reason the finalizer exists.
	// The teardown scan is the LAST thing that can give this node back: once the
	// finalizer comes off, nothing in the cluster knows the node is held.
	if err := h.c.Delete(context.Background(), h.run("run1")); err != nil {
		h.t.Fatalf("delete run: %v", err)
	}
	h.reconcile("run1")
	sc.thaw()

	if h.node("spark-85a9").Spec.Unschedulable {
		t.Error("spark-85a9 is still cordoned after its run was deleted — the release scanned a cache " +
			"that had not seen the stamp, found nothing to give back, and dropped the finalizer")
	}
	h.assertNoStrandedCordons()
}

// betweenWaves drives a two-node, cap-1 run to the moment it is about to hand
// back its first node: spark-a is cordoned and its pod has just finished.
//
// The mid-run release is where a per-node release is exercised at all:
// releaseCordons only ever runs at teardown, so this is the only path that
// decides, node by node, whether the run still owes one back.
func (h *harness) betweenWaves() {
	h.t.Helper()
	h.reconcile("run1") // Pending → Running: capture prior schedulability
	h.reconcile("run1") // wave 1: cordon spark-a, create its pod
	pod := h.pods("run1")["spark-a"]
	if pod == nil {
		h.t.Fatal("precondition: wave 1 should have created a pod on spark-a")
	}
	if !h.node("spark-a").Spec.Unschedulable {
		h.t.Fatal("precondition: the run should be holding spark-a by now")
	}
	h.startPod(pod)
	h.finishPod(pod, 0, fp4Stdout, "Completed")
}

// assertReleasedFirstWave is the fleet-visible half of the invariant: once the
// run has dropped its claim on spark-a, spark-a is back in the scheduler.
func (h *harness) assertReleasedFirstWave(how string) {
	h.t.Helper()
	for _, n := range h.run("run1").Status.CordonedNodes {
		if n == "spark-a" {
			h.t.Fatalf("precondition: the run has not released spark-a yet (%s)", how)
		}
	}
	if h.node("spark-a").Spec.Unschedulable {
		h.t.Errorf("spark-a is still cordoned with this run's stamp on it, but the run no longer "+
			"claims it: %s. status.cordonedNodes now disagrees with the fleet, and the lost "+
			"scheduling capacity is invisible until somebody diffs it against kubectl get nodes", how)
	}
}

// A run must not drop its claim on a node it is still holding because the
// informer cannot see the stamp. Issue #245.
//
// The reproduction: a cap-1 run steps from spark-a to spark-b, and at the moment
// it goes to give spark-a back the informer is still holding the version from
// before the stamp went on. releaseNode read that through the cache, concluded
// the node was not its own any more, wrote NOTHING — and dropped spark-a from
// status.CordonedNodes anyway.
//
// That is a conclusion of ABSENCE drawn from a cache, which is the class of read
// that has no optimistic-concurrency check behind it: an Update would have been
// refused on resourceVersion, but this arm makes no Update at all. The node
// stays cordoned, with this run's own stamp on it, and nothing owes it back
// until the run next cordons it or reaches teardown — for a single-test profile
// that is the rest of the run.
//
// It is scheduling capacity that is lost, not power: busyNodes counts live pods,
// so no extra load lands in the room. What is lost is a node the cluster can no
// longer place anything on, and a status.cordonedNodes that lies about it.
func TestRun_ReleaseDoesNotDropAClaimOnACordonTheInformerHasNotSeen(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a", "spark-b"),
	)
	sc := h.throughCache()
	h.betweenWaves()

	// The informer is stuck on the version it had BEFORE this run stamped
	// spark-a: no owner annotation, still schedulable.
	preCordon := gb10Node("spark-a")
	preCordon.ResourceVersion = h.node("spark-a").ResourceVersion
	sc.nodes = map[string]*corev1.Node{"spark-a": preCordon}

	h.reconcile("run1") // harvest spark-a, release it, move to spark-b
	sc.thaw()

	h.assertReleasedFirstWave("the informer was still holding the pre-cordon version")
}

// The same defect through the other arm: an informer that has not observed the
// node AT ALL answers NotFound, and "the node is gone" is the strongest
// conclusion of absence there is. A cache may not be the source of it.
func TestRun_ReleaseDoesNotDropAClaimOnANodeTheInformerCannotSee(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a", "spark-b"),
	)
	sc := h.throughCache()
	h.betweenWaves()

	sc.hidden = map[string]bool{"spark-a": true}

	h.reconcile("run1")
	sc.thaw()

	h.assertReleasedFirstWave("the informer had not observed spark-a at all")
}

// conflictOnTerminalStatus fails the run's status write once, the way a
// resourceVersion race does, and records that it did.
type conflictOnTerminalStatus struct {
	client.Client
	armed bool
	fired bool
}

func (c *conflictOnTerminalStatus) Status() client.SubResourceWriter {
	return &conflictWriter{SubResourceWriter: c.Client.Status(), owner: c}
}

type conflictWriter struct {
	client.SubResourceWriter
	owner *conflictOnTerminalStatus
}

func (w *conflictWriter) Update(ctx context.Context, obj client.Object, opts ...client.SubResourceUpdateOption) error {
	run, isRun := obj.(*burninv1alpha1.BurnInRun)
	if w.owner.armed && isRun && isTerminal(run.Status.Phase) {
		w.owner.armed, w.owner.fired = false, true
		return apierrors.NewConflict(
			schema.GroupResource{Group: burninv1alpha1.GroupVersion.Group, Resource: "burninruns"},
			run.Name, errObjectModified)
	}
	return w.SubResourceWriter.Update(ctx, obj, opts...)
}

var errObjectModified = &staticError{"the object has been modified; please apply your changes to the latest version and try again"}

type staticError struct{ msg string }

func (e *staticError) Error() string { return e.msg }

// A run must not kill a pod that is still working on the strength of a verdict
// it has not managed to write down. Issue #84, symptom 2.
//
// The reported sequence: the run reached a terminal phase with an execution
// still in flight, terminate() killed that pod, and then the status write lost
// the resourceVersion race — the conflict that appears in the report next to
// the kills. The reconcile returned an error, the terminal status went with it,
// and the run went back to believing it was Running with an open execution. Its
// next pass found the pod IT had just killed, harvested the SIGKILL, and
// recorded `Error: runner terminated abnormally (exit 137)` against the node.
// A healthy Spark condemned for the operator's own act, 30 s into a 100 s soak.
//
// The invariant, stated so it is not weakened later: NOTHING IRREVERSIBLE
// HAPPENS TO THE HARDWARE BEFORE THE VERDICT THAT JUSTIFIES IT IS DURABLE. A
// lost write must cost a wasted pass, never an execution.
func TestRun_ATerminalRunDoesNotKillPodsBeforeItsVerdictIsDurable(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		// failFast, so one node's failure terminates the run while the other
		// node's pod is still legitimately running.
		profile("acceptance", nil, true, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	blocker := &conflictOnTerminalStatus{Client: h.c, armed: true}
	h.r.Client = blocker

	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("expected 2 pods, got %d", len(pods))
	}
	h.startPod(pods["spark-a"])
	h.startPod(pods["spark-b"])
	h.finishPod(pods["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")

	// Drive up to and through the pass that trips failFast. That pass's terminal
	// status write is refused, which surfaces as a reconcile error.
	var err error
	for i := 0; i < 10 && err == nil; i++ {
		_, err = h.r.Reconcile(context.Background(), reqFor("run1"))
	}
	if err == nil {
		t.Fatal("expected the refused status write to surface as a reconcile error")
	}
	if !blocker.fired {
		t.Fatalf("the test never exercised the conflict it exists to exercise (error was %v)", err)
	}

	live := map[string]bool{}
	for _, pod := range h.livePods("run1") {
		live[pod.Labels[labelNode]] = true
	}
	if !live["spark-b"] {
		t.Fatal("spark-b's pod was killed although the verdict that justified killing it was never " +
			"written — the run will come back believing it is still Running and harvest its own SIGKILL " +
			"as a hardware Error against a healthy node")
	}

	// And the run converges once the write lands: the verdict is recorded, and
	// only then is the in-flight pod stopped.
	h.reconcileUntilSettled("run1")
	if phase := h.run("run1").Status.Phase; phase != burninv1alpha1.RunFailed {
		t.Errorf("phase = %q, want Failed", phase)
	}
	for _, pod := range h.livePods("run1") {
		t.Errorf("pod %s is still burning a node behind a terminal run", pod.Name)
	}
	h.assertNoStrandedCordons()
}

// refusePodDelete fails the first pod deletion, the way a throttled or briefly
// unavailable apiserver does.
type refusePodDelete struct {
	client.Client
	armed bool
}

func (c *refusePodDelete) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	if _, isPod := obj.(*corev1.Pod); isPod && c.armed {
		c.armed = false
		return apierrors.NewServiceUnavailable("apiserver is having a moment")
	}
	return c.Client.Delete(ctx, obj, opts...)
}

// A pod that outlives the call meant to stop it must still be reaped.
//
// Stopping the hardware now happens AFTER the verdict is durable, which is the
// fix for #84 — but it means the delete can fail with the run already terminal.
// The finalizer is therefore held until the pods are actually gone, and
// reconcileTerminal sweeps: without both, a transient API error would leave a
// runner at full power on a node the run has already handed back to the
// scheduler, with nothing in the cluster left to notice.
func TestRun_ATerminalRunSweepsAPodItsCleanupFailedToStop(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, true, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	flaky := &refusePodDelete{Client: h.c, armed: true}
	h.r.Client = flaky

	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	h.startPod(pods["spark-a"])
	h.startPod(pods["spark-b"])
	h.finishPod(pods["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")

	var err error
	for i := 0; i < 10 && err == nil; i++ {
		_, err = h.r.Reconcile(context.Background(), reqFor("run1"))
	}
	if err == nil || flaky.armed {
		t.Fatalf("the test never exercised the refused pod deletion (error was %v)", err)
	}
	if h.run("run1").Status.Phase != burninv1alpha1.RunFailed {
		t.Fatal("precondition: the verdict should already be durable when the cleanup fails")
	}

	h.reconcileUntilSettled("run1")
	for _, pod := range h.livePods("run1") {
		t.Errorf("pod %s is still burning %s behind a terminal run — the cleanup failed once and "+
			"nothing came back for it", pod.Name, pod.Labels[labelNode])
	}
	h.assertNoStrandedCordons()
}

func reqFor(name string) ctrl.Request {
	return ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "burnin", Name: name}}
}

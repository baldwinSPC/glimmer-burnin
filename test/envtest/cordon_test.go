package envtest

import (
	"context"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Cordoning, against real Node objects.
//
// A cordon is a destructive edit to an object the operator does not own, and
// there are exactly two ways to get it wrong: uncordoning somebody else's node,
// and never uncordoning your own. Both are cluster state, and both are what the
// fake client is least able to say anything about — every existing cordon test
// hands the reconciler a Get that tells the truth, on a store that has no other
// writers.

// TestCordonTracksTheWaveAndNotTheTargetList holds the interlock's premise to
// account: a run's footprint on the fleet is MaxConcurrentNodes, not the size
// of its target list.
//
// Cordoning every target up front took a two-node cluster entirely out of
// scheduling for the length of the run and hollowed out the interlock. The
// assertion is made between reconciles, from the apiserver, so it is about what
// the CLUSTER looked like rather than about what the controller intended.
func TestCordonTracksTheWaveAndNotTheTargetList(t *testing.T) {
	ns := newNamespace(t)
	a, b := nodeName(t, "wave-a"), nodeName(t, "wave-b")
	newNode(t, a)
	newNode(t, b)

	create(t,
		customTest(ns, "smoke", nil),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "smoke"}),
		// Cap 1 is the default and the safe floor: strictly sequential.
		runFor(ns, "run", "acceptance", []string{a, b}, nil),
	)

	d := newDriver(t, ns)
	d.script("", script{stdout: "miscompares=0\n"})
	d.reconcilerOver(admin)

	maxCordonedAtOnce := 0
	for i := 0; i < 40; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		if n := countCordoned(t, a, b); n > maxCordonedAtOnce {
			maxCordonedAtOnce = n
		}
		if isTerminalPhase(getRun(t, ns, "run").Status.Phase) {
			break
		}
		d.playKubelet("run")
	}

	if maxCordonedAtOnce > 1 {
		t.Errorf("%d of the run's targets were out of the scheduler at once, but maxConcurrentNodes is 1 — "+
			"the cordon is following the target list rather than the wave", maxCordonedAtOnce)
	}

	run := d.run("run", 40)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}
	assertFullyReleased(t, run, a, b)
}

// TestACordonPlacedAfterTheRunStartedIsNotAdopted is the regression test for
// the stranded-cordon bug.
//
// "Was this node already out of the scheduler?" is a question about the moment
// the run ARRIVED, and the only place it can be answered is before the run has
// a footprint of its own. Asked again on a later wave — which is what a
// per-wave cordon forces — the reading includes the run's own hold, or a hold
// laid over the top of it, and the two are indistinguishable from the node
// alone. A run that answered "yes" then honoured that answer at teardown and
// left the node unschedulable forever, with an annotation saying it was
// deliberate.
//
// The scenario below applies a cordon to a target the run has already released
// between waves. That is exactly what a stale read of the run's own
// just-released cordon looks like from inside cordonNode, and it is also a real
// thing an administrator can do. status.priorUnschedulable, captured once at
// start, is what has to win.
func TestACordonPlacedAfterTheRunStartedIsNotAdopted(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "contaminated")
	newNode(t, node)

	create(t,
		customTest(ns, "first", nil),
		customTest(ns, "second", nil),
		profile(ns, "two-tests", nil,
			burninv1alpha1.ProfileTest{TestRef: "first"},
			burninv1alpha1.ProfileTest{TestRef: "second"}),
		runFor(ns, "run", "two-tests", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.script("", script{stdout: "miscompares=0\n"})
	d.reconcilerOver(admin)

	// Drive until the first test has settled. That is the point at which the
	// run has cordoned and released this node once already.
	contaminated := false
	for i := 0; i < 40; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		run := getRun(t, ns, "run")
		if isTerminalPhase(run.Status.Phase) {
			break
		}
		if !contaminated {
			if res := resultFor(run, "first", node); res != nil && res.Phase.IsTerminal() {
				cordonExternally(t, node)
				contaminated = true
			}
		}
		d.playKubelet("run")
	}
	if !contaminated {
		t.Fatal("the first test never settled, so the contamination window was never reached")
	}

	run := d.run("run", 40)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}

	// The record taken at start is the only uncontaminated reading, and it is
	// what the release restores from.
	if got := run.Status.PriorUnschedulable[node]; got {
		t.Errorf("status.priorUnschedulable[%s] = true — the run adopted a cordon that was not there when it started, "+
			"and would make it permanent at teardown", node)
	}
	assertFullyReleased(t, run, node)
}

// TestAPriorCordonIsRestoredAndNotUncordoned is the other half of the same
// rule, in the direction that loses a maintenance window rather than a node.
//
// A node that was already unschedulable when the run found it was somebody's
// deliberate act — a drain for a firmware update, most likely. Returning it to
// service because a burn-in finished is how a node under maintenance starts
// taking production traffic mid-update.
func TestAPriorCordonIsRestoredAndNotUncordoned(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "drained")
	newNode(t, node)
	cordonExternally(t, node)

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
	if !run.Status.PriorUnschedulable[node] {
		t.Errorf("status.priorUnschedulable[%s] = false, but the node was drained before the run started", node)
	}
	after := getNode(t, node)
	if !after.Spec.Unschedulable {
		t.Errorf("node %s was returned to the scheduler, undoing a drain the run did not place", node)
	}
	// The node is restored, but the operator's bookkeeping must not be left
	// behind: an owner stamp with no owner is what makes the next run's
	// reasoning wrong.
	assertNoOperatorAnnotations(t, node)
	if len(run.Status.CordonedNodes) != 0 {
		t.Errorf("status.cordonedNodes = %v after the run finished, want empty", run.Status.CordonedNodes)
	}
}

// TestACordonOwnedBySomeoneElseIsNeitherStolenNorReleased covers the second of
// the two ways to get this wrong.
//
// Another run — or an administrator's tooling using the same stamp — owns this
// node. Recording it as ours would make this run release a hold it never
// placed, which is a node handed back to the scheduler while something else is
// still burning it.
func TestACordonOwnedBySomeoneElseIsNeitherStolenNorReleased(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "someone-elses")
	newNode(t, node)

	const otherOwner = "other-ns/other-run/11111111-2222-3333-4444-555555555555"
	stampNode(t, node, map[string]string{
		burninv1alpha1.AnnotationCordonOwner:        otherOwner,
		burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableTrue,
	}, true)

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
	after := getNode(t, node)
	if got := after.Annotations[burninv1alpha1.AnnotationCordonOwner]; got != otherOwner {
		t.Errorf("cordon owner annotation is now %q, want it left as %q — this run took over somebody else's hold",
			got, otherOwner)
	}
	if !after.Spec.Unschedulable {
		t.Errorf("node %s was uncordoned, but the hold belonged to %s", node, otherOwner)
	}
	for _, held := range run.Status.CordonedNodes {
		if held == node {
			t.Errorf("the run recorded %s as one of ITS cordons; it would release a hold it never placed", node)
		}
	}
}

// TestDeletingARunReleasesEveryCordonItHeld exercises the finalizer.
//
// `kubectl delete burninrun` is the most natural reaction to a run behaving
// badly, and therefore the likeliest moment to strand a node: deleting the run
// removes the only object that knows which nodes are held. envtest is where
// this can be tested at all, because the fake client's deletion does not honour
// finalizers the way the apiserver does.
func TestDeletingARunReleasesEveryCordonItHeld(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "deleted-run")
	newNode(t, node)

	create(t,
		customTest(ns, "soak", func(bt *burninv1alpha1.BurnInTest) { bt.Spec.DurationSeconds = 600 }),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "soak"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	// A pod that never finishes, so the run is still holding the node when the
	// delete arrives.
	d.script("", script{linger: true})
	d.reconcilerOver(admin)

	for i := 0; i < 10 && !getNode(t, node).Spec.Unschedulable; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		d.playKubelet("run")
	}
	if !getNode(t, node).Spec.Unschedulable {
		t.Fatal("the run never cordoned its target, so there is nothing to strand")
	}

	ctx := context.Background()
	if err := admin.Delete(ctx, getRun(t, ns, "run")); err != nil {
		t.Fatalf("delete run: %v", err)
	}
	// The finalizer is holding deletion open. Nothing else in the cluster knows
	// this node is held, so if the reconciler does not release it here, it never
	// gets released at all.
	var deleted bool
	for i := 0; i < 10; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("post-delete reconcile %d: %v", i, err)
		}
		var run burninv1alpha1.BurnInRun
		err := admin.Get(ctx, types.NamespacedName{Namespace: ns, Name: "run"}, &run)
		if apierrors.IsNotFound(err) {
			deleted = true
			break
		}
	}
	if !deleted {
		t.Error("the run object is still present after its finalizer should have been removed")
	}
	if getNode(t, node).Spec.Unschedulable {
		t.Errorf("node %s is still cordoned after its run was deleted — the fleet has silently lost it", node)
	}
	assertNoOperatorAnnotations(t, node)
}

// TestAHeldNodeDeclaresTheHeatAndStopsWhenReleased is issue #280 against a real
// apiserver.
//
// A separate fleet agent's thermal watchdog drains a node whose part passes its
// trip point, and a soak drives the part past that point on purpose — so every soak
// on this fleet was drained by the safety system and its runner pods killed with
// SIGKILL, which read back as `Error — hardware unjudged`. The operator's half
// of the fix is to say on the node that the heat is deliberate.
//
// It is asserted HERE as well as in the unit tests because the declaration is
// read by a DIFFERENT PROGRAM off a real Node object. What the fake client
// cannot say anything about is whether the value the operator writes survives a
// round trip through the apiserver intact — and a value the agent cannot parse
// is a value the agent refuses, which restores the exact bug being fixed.
func TestAHeldNodeDeclaresTheHeatAndStopsWhenReleased(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "declared")
	newNode(t, node)

	// A kind that HOLDS a load, because that is the only kind the declaration is
	// written for — a burst or a passive read heats nothing and claims nothing.
	create(t,
		customTest(ns, "soak", func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.Kind = burninv1alpha1.KindThermalSoak
			bt.Spec.Runner = nil
			bt.Spec.DurationSeconds = 600
		}),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "soak"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	// A pod that never finishes, so the node is still under load — and still
	// owed a declaration — while it is inspected.
	d.script("", script{linger: true})
	d.reconcilerOver(admin)

	for i := 0; i < 10 && !getNode(t, node).Spec.Unschedulable; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("reconcile %d: %v", i, err)
		}
		d.playKubelet("run")
	}
	if !getNode(t, node).Spec.Unschedulable {
		t.Fatal("the run never cordoned its target, so no declaration is owed")
	}

	run := getRun(t, ns, "run")
	held := getNode(t, node)
	raw, present := held.Annotations[burninv1alpha1.AnnotationHeatExpected]
	if !present {
		t.Fatalf("node %s is under a burn-in and declares no heat — the watchdog reads the soak as a "+
			"cooling failure, drains the node and kills the runner", node)
	}

	// Parsed the way the agent must: four fields, the owning run, and an expiry
	// that has not passed.
	parts := strings.Split(raw, "/")
	if len(parts) != 4 {
		t.Fatalf("declaration %q is not <namespace>/<name>/<uid>/<expiry>", raw)
	}
	if got, want := strings.Join(parts[:3], "/"), ns+"/run/"+string(run.UID); got != want {
		t.Errorf("declaration names %q, want %q — a reader cannot tell which run is producing the heat", got, want)
	}
	expiry, err := time.Parse(time.RFC3339, parts[3])
	if err != nil {
		t.Fatalf("declaration expiry %q is not RFC3339: %v — the agent refuses what it cannot parse, "+
			"which is the drain this change exists to stop", parts[3], err)
	}
	if !expiry.After(time.Now()) {
		t.Errorf("declaration expired at %s while the run is still loading the node", expiry)
	}

	if err := admin.Delete(context.Background(), getRun(t, ns, "run")); err != nil {
		t.Fatalf("delete run: %v", err)
	}
	for i := 0; i < 10; i++ {
		if _, err := d.reconcile("run"); err != nil {
			t.Fatalf("post-delete reconcile %d: %v", i, err)
		}
		var gone burninv1alpha1.BurnInRun
		if apierrors.IsNotFound(admin.Get(context.Background(),
			types.NamespacedName{Namespace: ns, Name: "run"}, &gone)) {
			break
		}
	}

	// assertNoOperatorAnnotations covers the declaration; the node also has to
	// be back in the scheduler, because a node returned to service still
	// declaring heat is the worse of the two half-released states.
	if getNode(t, node).Spec.Unschedulable {
		t.Errorf("node %s is still cordoned after its run was deleted", node)
	}
	assertNoOperatorAnnotations(t, node)
}

// ─── helpers ──────────────────────────────────────────────────────────────────

func countCordoned(t *testing.T, names ...string) int {
	t.Helper()
	n := 0
	for _, name := range names {
		if getNode(t, name).Spec.Unschedulable {
			n++
		}
	}
	return n
}

// cordonExternally cordons a node the way `kubectl cordon` does: spec only, no
// operator annotations.
func cordonExternally(t *testing.T, name string) {
	t.Helper()
	stampNode(t, name, nil, true)
}

func stampNode(t *testing.T, name string, annotations map[string]string, unschedulable bool) {
	t.Helper()
	ctx := context.Background()
	for attempt := 0; attempt < 5; attempt++ {
		node := getNode(t, name)
		if node.Annotations == nil {
			node.Annotations = map[string]string{}
		}
		for k, v := range annotations {
			node.Annotations[k] = v
		}
		node.Spec.Unschedulable = unschedulable
		err := admin.Update(ctx, node)
		if err == nil {
			return
		}
		if !apierrors.IsConflict(err) {
			t.Fatalf("stamp node %s: %v", name, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("stamp node %s: gave up after repeated conflicts", name)
}

// assertFullyReleased is the "no stranded cordon" assertion: every named node
// is back in the scheduler, carries none of the operator's bookkeeping, and the
// run claims to hold nothing.
func assertFullyReleased(t *testing.T, run *burninv1alpha1.BurnInRun, nodes ...string) {
	t.Helper()
	for _, name := range nodes {
		node := getNode(t, name)
		if node.Spec.Unschedulable {
			t.Errorf("node %s is still cordoned after the run reached %s — a node the fleet has silently lost",
				name, run.Status.Phase)
		}
		assertNoOperatorAnnotations(t, name)
	}
	if len(run.Status.CordonedNodes) != 0 {
		t.Errorf("status.cordonedNodes = %v after the run finished, want empty", run.Status.CordonedNodes)
	}
}

func assertNoOperatorAnnotations(t *testing.T, name string) {
	t.Helper()
	node := getNode(t, name)
	for _, key := range []string{
		burninv1alpha1.AnnotationCordonOwner,
		burninv1alpha1.AnnotationPriorUnschedulable,
		// The heat declaration is released with the cordon, in the same update.
		// A node back in the scheduler still declaring heat is one the agent's
		// thermal watchdog has been told to stand down for while ordinary
		// workload lands on it.
		burninv1alpha1.AnnotationHeatExpected,
	} {
		if v, ok := node.Annotations[key]; ok {
			t.Errorf("node %s still carries %s=%q — an ownership stamp with no owner misleads the next run",
				name, key, v)
		}
	}
}

package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The heat declaration. Issue #280.
//
// The Glimmer agent's thermal watchdog drains a node that passes its trip point,
// and a thermal soak drives the part past that point ON PURPOSE. Every soak on
// this fleet was therefore drained by the safety system, its runner pods killed
// with SIGKILL, and the hardware recorded as "Error — hardware unjudged". The
// watchdog was not wrong; it had no way to tell a soak from a cooling failure.
//
// AnnotationHeatExpected is that way. These tests hold the operator's half to
// account, and every one of them asserts the state of a NODE — what the agent
// would actually read — rather than that some function was called.

// heatOn is the declaration a node currently carries, parsed. The second return
// is false when the node carries none, or one that does not parse — which is
// the same thing as far as the agent is concerned, since its rule is to refuse
// what it cannot read.
func heatOn(t *testing.T, node *corev1.Node) (heatMarker, bool) {
	t.Helper()
	raw, present := node.Annotations[burninv1alpha1.AnnotationHeatExpected]
	if !present {
		return heatMarker{}, false
	}
	return parseHeatExpected(raw)
}

// assertDeclaresHeat is the assertion the agent's reader implies: the node says
// heat is expected, the run named is the one testing it, and the declaration has
// not lapsed as of `now`.
func assertDeclaresHeat(t *testing.T, node *corev1.Node, wantUID string, now time.Time) heatMarker {
	t.Helper()
	raw := node.Annotations[burninv1alpha1.AnnotationHeatExpected]
	marker, ok := heatOn(t, node)
	if !ok {
		t.Fatalf("node %s declares no readable heat (annotation %q) — the watchdog will read a soak as a "+
			"cooling failure, drain the node and kill the runner", node.Name, raw)
	}
	if marker.uid != wantUID {
		t.Fatalf("node %s declares heat for run uid %q, want %q", node.Name, marker.uid, wantUID)
	}
	if !marker.expiry.After(now) {
		t.Fatalf("node %s declares heat that expired at %s, and it is %s — the declaration is inert while "+
			"the run is still loading the part", node.Name, marker.expiry, now)
	}
	return marker
}

// loadTest is a test of one kind, with one duration. It exists because the
// declaration is now the KIND's decision, so these cases have to name kinds
// rather than reach for the nearest fixture.
func loadTest(name string, kind burninv1alpha1.TestKind, duration int32) *burninv1alpha1.BurnInTest {
	return &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:            kind,
			Scope:           burninv1alpha1.ScopeNode,
			DurationSeconds: duration,
		},
	}
}

// ─── 1. Written on the cordon, removed on the release ─────────────────────────

// The whole fix in one pass: a node cordoned for a run declares the heat, and a
// node the run has finished with does not.
//
// Both halves matter and they fail in opposite directions. Without the first,
// the soak is drained and the hardware comes back unjudged — the bug. Without
// the second, a node returned to the scheduler keeps its thermal watchdog
// switched off while ordinary workload lands on it, which is lost hardware
// rather than a lost measurement.
func TestHeat_DeclaredWhileTheNodeIsHeldAndWithdrawnWhenTheRunSettles(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("thermal-soak", 900, 900),
		profile("acceptance", nil, false, testRef("thermal-soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	node := h.node("spark-a")
	if !node.Spec.Unschedulable {
		t.Fatal("setup: the target was never cordoned, so there is no declaration to make")
	}
	marker := assertDeclaresHeat(t, node, "uid-run1", h.nowVal)
	if marker.namespace != "burnin" || marker.name != "run1" {
		t.Errorf("declaration names %s/%s, want burnin/run1 — a human reading the node cannot find the run",
			marker.namespace, marker.name)
	}

	h.burnSegment("run1", 1, 0, cleanSegment())
	h.reconcileUntilSettled("run1")

	if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("setup: run did not pass (%s)", h.run("run1").Status.Phase)
	}
	// assertNoStrandedCordons covers the declaration too; this says it plainly.
	if raw, present := h.node("spark-a").Annotations[burninv1alpha1.AnnotationHeatExpected]; present {
		t.Errorf("node still declares heat as %q after the run settled — the thermal watchdog is "+
			"disabled on a node nothing is burning", raw)
	}
	h.assertNoStrandedCordons()
}

// ─── 2. The expiry is derived from the run's own window ───────────────────────

// A seven-day segmented soak and a sixty-second smoke test are the same code
// path and need very different windows, so the expiry is sized from the run's
// own per-pod duration and not from a constant.
//
// The assertion is on the expiry the NODE carries, against the window this
// operator itself would give the pod (podWindow — the point at which podOverdue
// gives up on it). Two things have to hold at once and they pull opposite ways:
// the declaration must outlive the pod, or a soak is drained mid-measurement;
// and it must not outlive it by much, or a crashed operator leaves thermal
// protection off for longer than one pod could ever have justified.
func TestHeat_ExpiryIsSizedFromTheTestsOwnDuration(t *testing.T) {
	for _, tc := range []struct {
		name       string
		test       *burninv1alpha1.BurnInTest
		perPodSecs int32
	}{
		// A load-holding kind that states no duration, so the operator's own
		// default is what sizes the window.
		{"unbounded burn", loadTest("gpu-burn", burninv1alpha1.KindGPUBurn, 0), defaultDurationSeconds},
		{"short soak", soakTest("thermal-soak", 600, 600), 600},
		// A seven-day soak in fifteen-minute segments: the declaration covers
		// ONE SEGMENT, because one segment is all one pod can justify.
		{"segmented week", soakTest("thermal-soak", 604800, 900), 900},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t,
				gb10Node("spark-a"),
				tc.test,
				profile("acceptance", nil, false, testRef(tc.test.Name)),
				newRun("run1", "acceptance", "spark-a"),
			)
			h.reconcile("run1")
			h.reconcile("run1")

			marker := assertDeclaresHeat(t, h.node("spark-a"), "uid-run1", h.nowVal)
			granted := marker.expiry.Sub(h.nowVal)

			podLife := time.Duration(tc.perPodSecs+deadlineGraceSeconds)*time.Second + schedulingGracePeriod
			if granted <= podLife {
				t.Errorf("the declaration expires %v from now but the operator lets this test's pod live "+
					"%v — the watchdog would drain the node while the pod is still inside its own budget",
					granted, podLife)
			}
			if granted > podLife+heatReleaseMargin {
				t.Errorf("the declaration expires %v from now, more than one pod window (%v) plus the "+
					"release margin (%v) — an abandoned marker disables thermal protection for longer "+
					"than any pod could justify", granted, podLife, heatReleaseMargin)
			}
		})
	}
}

// The two windows must come from ONE source, or the operator can reap a pod it
// had already stopped declaring heat for. This is the arithmetic identity that
// keeps them from drifting apart.
func TestHeat_ExpiryOutlivesThePodTheOperatorWouldReap(t *testing.T) {
	spec := soakTest("thermal-soak", 3600, 900).Spec
	now := time.Unix(1750000000, 0).UTC()

	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{CreationTimestamp: metav1.NewTime(now)}}
	r := &BurnInRunReconciler{}

	// One instant before the operator gives up on the pod, the declaration must
	// still stand — otherwise the watchdog gets there first.
	r.Now = func() time.Time { return now.Add(podWindow(&spec)) }
	if r.podOverdue(pod, &spec) {
		t.Fatal("setup: the pod is already overdue at exactly its window")
	}
	if !heatExpiry(now, &spec).After(r.now()) {
		t.Errorf("the declaration lapses at %s while the operator still considers the pod live at %s",
			heatExpiry(now, &spec), r.now())
	}
}

// ─── 3. A run that ends abnormally ────────────────────────────────────────────

// A run deleted mid-flight cannot take its own declaration off, so the reaper
// does it — in the same update that gives the cordon back, because a node
// returned to the scheduler still declaring heat is the worse of the two states.
func TestHeat_AReapedOrphanLosesItsDeclarationWithItsCordon(t *testing.T) {
	node := stampedNode("spark-85a9", "burnin/gone/uid-gone", false)
	node.Annotations[burninv1alpha1.AnnotationHeatExpected] = "burnin/gone/uid-gone/2099-01-01T00:00:00Z"

	h := newNFPHarness(t, node)
	h.r.APIReader = h.c

	h.reconcileNode("spark-85a9")

	after := h.node("spark-85a9")
	if after.Spec.Unschedulable {
		t.Fatal("setup: the orphaned cordon was not reaped, so nothing was owed the declaration either")
	}
	if raw, present := after.Annotations[burninv1alpha1.AnnotationHeatExpected]; present {
		t.Errorf("the node is back in the scheduler and still declares heat as %q for a run that no "+
			"longer exists — the watchdog is off on a node taking production workload", raw)
	}
}

// And independently of the reaper: a declaration whose expiry has passed is
// inert on its own. This is what makes the marker safe to ship at all — the
// reader is on the node, knows nothing about whether the run is alive, and the
// operator may have crashed, lost its lease or been deleted. Nothing in this
// cluster has to run for an abandoned declaration to stop counting.
func TestHeat_AnExpiredDeclarationIsInertWithNobodyToRemoveIt(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	spec := soakTest("thermal-soak", 900, 900).Spec
	stamped := time.Unix(1750000000, 0).UTC()

	value := heatExpectedID(run, heatExpiry(stamped, &spec))
	marker, ok := parseHeatExpected(value)
	if !ok {
		t.Fatalf("the operator cannot read back the value it wrote: %q", value)
	}

	// Far enough past that no pod of this test could still be running.
	abandoned := stamped.Add(podWindow(&spec) + heatReleaseMargin + time.Second)
	if !abandoned.After(marker.expiry) {
		t.Fatalf("a declaration written at %s still stands at %s, %v later, with the run long gone — "+
			"thermal protection is disabled with nothing left to re-enable it",
			stamped, abandoned, abandoned.Sub(stamped))
	}
	if !marker.expiry.After(stamped) {
		t.Fatalf("the declaration expired at %s, before it was written at %s", marker.expiry, stamped)
	}
}

// The value the node carries is what a reader in another program parses, so its
// shape is a contract and not an implementation detail. UTC, RFC3339, four
// slash-separated fields.
func TestHeat_TheValueIsTheShapeTheAgentParses(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	expiry := time.Unix(1750003600, 0).In(time.FixedZone("CEST", 2*60*60))

	got := heatExpectedID(run, expiry)
	const want = "burnin/run1/uid-run1/2025-06-15T16:06:40Z"
	if got != want {
		t.Errorf("declaration value = %q, want %q — the agent parses this byte for byte", got, want)
	}
	if strings.Count(got, "/") != 3 {
		t.Errorf("declaration value %q is not four slash-separated fields", got)
	}
}

// ─── 4. Authority is by UID ───────────────────────────────────────────────────

// A declaration another run made is that run's thermal protection. Removing it
// hands that run's soak to the watchdog — the exact failure this marker exists
// to prevent, committed by the code meant to prevent it.
//
// Same rule as the cordon, and for the same reason: names are reusable, so a
// deleted-and-recreated run of the same name is a different run.
func TestHeat_ADeclarationOwnedByADifferentRunIsNeverRemoved(t *testing.T) {
	const other = "burnin/other-run/uid-other/2099-01-01T00:00:00Z"

	t.Run("on release", func(t *testing.T) {
		node := gb10Node("spark-a")
		node.Annotations = map[string]string{burninv1alpha1.AnnotationHeatExpected: other}
		h := newHarness(t,
			node,
			soakTest("thermal-soak", 900, 900),
			profile("acceptance", nil, false, testRef("thermal-soak")),
			newRun("run1", "acceptance", "spark-a"),
		)
		h.reconcile("run1")
		h.reconcile("run1")
		// run1 holds the cordon, but the declaration on the node is not its own
		// and must be left exactly as found — including through the release.
		if got := h.node("spark-a").Annotations[burninv1alpha1.AnnotationHeatExpected]; got != other {
			t.Fatalf("declaration = %q while run1 held the node, want it untouched as %q", got, other)
		}

		h.burnSegment("run1", 1, 0, cleanSegment())
		h.reconcileUntilSettled("run1")

		if got := h.node("spark-a").Annotations[burninv1alpha1.AnnotationHeatExpected]; got != other {
			t.Errorf("declaration = %q after run1 released the node, want %q — run1 stripped another "+
				"run's thermal protection and that run's soak is now drainable", got, other)
		}
	})

	t.Run("on reap", func(t *testing.T) {
		// The cordon's owner is provably gone, so the cordon is reaped — but the
		// declaration names somebody else and is not the reaper's to take.
		node := stampedNode("spark-85a9", "burnin/gone/uid-gone", false)
		node.Annotations[burninv1alpha1.AnnotationHeatExpected] = other

		h := newNFPHarness(t, node)
		h.r.APIReader = h.c
		h.reconcileNode("spark-85a9")

		after := h.node("spark-85a9")
		if after.Spec.Unschedulable {
			t.Fatal("setup: the orphaned cordon was not reaped, so the reaper never got to the declaration")
		}
		if got := after.Annotations[burninv1alpha1.AnnotationHeatExpected]; got != other {
			t.Errorf("declaration = %q after the reap, want %q — the reaper removed a live run's "+
				"thermal protection while reaping somebody else's cordon", got, other)
		}
	})
}

// ─── 5. An unreadable value is not evidence ───────────────────────────────────

// "I cannot read this" is not evidence that the heat is over, and it is not
// evidence that the value is this operator's to overwrite either. Both paths
// leave it exactly as found — the same rule parseCordonOwner has held to since
// the reaper was written.
func TestHeat_AnUnparseableDeclarationIsLeftAlone(t *testing.T) {
	for _, raw := range []string{
		"burnin/run1/uid-run1",                       // no expiry at all
		"burnin/run1/uid-run1/not-a-timestamp",       // expiry the agent cannot read
		"burnin/run1/uid-run1/2026-08-17 12:00:00",   // not RFC3339
		"burnin//uid-run1/2099-01-01T00:00:00Z",      // empty segment
		"burnin/run1/uid-run1/2099-01-01T00:00:00Z/", // a fifth field
		"some-other-tool",
	} {
		t.Run(raw, func(t *testing.T) {
			if _, ok := parseHeatExpected(raw); ok {
				t.Fatalf("setup: %q parses, so this case is not about an unreadable value", raw)
			}

			t.Run("release leaves it", func(t *testing.T) {
				node := gb10Node("spark-a")
				node.Annotations = map[string]string{burninv1alpha1.AnnotationHeatExpected: raw}
				h := newHarness(t,
					node,
					soakTest("thermal-soak", 900, 900),
					profile("acceptance", nil, false, testRef("thermal-soak")),
					newRun("run1", "acceptance", "spark-a"),
				)
				h.reconcile("run1")
				h.reconcile("run1")
				h.burnSegment("run1", 1, 0, cleanSegment())
				h.reconcileUntilSettled("run1")

				if got := h.node("spark-a").Annotations[burninv1alpha1.AnnotationHeatExpected]; got != raw {
					t.Errorf("declaration = %q, want it left as %q — something else in the cluster may "+
						"be relying on a value this operator merely cannot read", got, raw)
				}
			})

			t.Run("reap leaves it", func(t *testing.T) {
				node := stampedNode("spark-85a9", "burnin/gone/uid-gone", false)
				node.Annotations[burninv1alpha1.AnnotationHeatExpected] = raw

				h := newNFPHarness(t, node)
				h.r.APIReader = h.c
				h.reconcileNode("spark-85a9")

				if got := h.node("spark-85a9").Annotations[burninv1alpha1.AnnotationHeatExpected]; got != raw {
					t.Errorf("declaration = %q after the reap, want it left as %q", got, raw)
				}
			})
		})
	}
}

// ─── 6. The two annotations move together ─────────────────────────────────────

// cordonreaper.go states it as an invariant: the annotations are removed
// together, and splitting them leaves a node half-released. This holds the whole
// release path to that, at every route out of a run.
//
// The failing shape is not hypothetical in either direction. Cordon without
// declaration is the bug (#280): a soak drained by its own safety system.
// Declaration without cordon is worse: a node back in the scheduler with its
// thermal watchdog switched off.
func TestHeat_TheCordonAndTheDeclarationAreNeverSeparated(t *testing.T) {
	for _, tc := range []struct {
		name string
		end  func(h *harness)
	}{
		{"run passes", func(h *harness) {
			h.burnSegment("run1", 1, 0, cleanSegment())
			h.reconcileUntilSettled("run1")
		}},
		{"run fails", func(h *harness) {
			h.burnSegment("run1", 1, 1, "THERMAL_SOAK_FAIL\n")
			h.reconcileUntilSettled("run1")
		}},
		{"run is deleted mid-flight", func(h *harness) {
			run := h.run("run1")
			if err := h.c.Delete(context.Background(), run); err != nil {
				h.t.Fatalf("delete run: %v", err)
			}
			h.reconcileUntilSettled("run1")
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t,
				gb10Node("spark-a"),
				soakTest("thermal-soak", 900, 900),
				profile("acceptance", nil, false, testRef("thermal-soak")),
				newRun("run1", "acceptance", "spark-a"),
			)
			h.reconcile("run1")
			h.reconcile("run1")

			held := h.node("spark-a")
			if !held.Spec.Unschedulable {
				t.Fatal("setup: the node was never cordoned")
			}
			assertDeclaresHeat(t, held, "uid-run1", h.nowVal)

			tc.end(h)

			after := h.node("spark-a")
			_, cordonStamp := after.Annotations[burninv1alpha1.AnnotationCordonOwner]
			_, declaration := after.Annotations[burninv1alpha1.AnnotationHeatExpected]
			if cordonStamp != declaration {
				t.Errorf("cordon stamp present=%v, heat declaration present=%v — the two came apart, "+
					"leaving the node half-released", cordonStamp, declaration)
			}
			if declaration {
				t.Error("the node still declares heat after the run ended")
			}
		})
	}
}

// ─── Extending across a long run ──────────────────────────────────────────────

// A declaration sized for one pod expires under a run that outlives one pod, and
// a seven-day soak is hundreds of pods. So each segment pushes the expiry out.
//
// Without this the marker would work for the length of the first segment and
// then the watchdog would drain the node — the same bug, arriving fifteen
// minutes late instead of immediately, which is strictly harder to diagnose.
func TestHeat_ASegmentedSoakKeepsItsDeclarationAheadOfTheClock(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("thermal-soak", 2700, 900), // three fifteen-minute segments
		profile("acceptance", nil, false, testRef("thermal-soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	first := assertDeclaresHeat(t, h.node("spark-a"), "uid-run1", h.nowVal)

	var last time.Time
	for segment := 1; segment <= 3; segment++ {
		assertDeclaresHeat(t, h.node("spark-a"), "uid-run1", h.nowVal)
		h.burnSegment("run1", segment, 0, cleanSegment())
		// Real time passes while a segment burns; the declaration has to keep
		// ahead of it or it lapses under the pod after it.
		h.nowVal = h.nowVal.Add(15 * time.Minute)
		h.reconcile("run1")
		if segment < 3 {
			last = assertDeclaresHeat(t, h.node("spark-a"), "uid-run1", h.nowVal).expiry
		}
	}

	if !last.After(first.expiry) {
		t.Errorf("the declaration still expires at %s after three segments, exactly as it did at the "+
			"first (%s) — a soak longer than one pod window outlives its own declaration",
			last, first.expiry)
	}

	h.reconcileUntilSettled("run1")
	if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("setup: run did not pass (%s): %+v", h.run("run1").Status.Phase, h.run("run1").Status.Results)
	}
	h.assertNoStrandedCordons()
}

// The expiry only ever moves FORWARD. A re-stamp that shortened one already in
// force would let a slow reconcile pull the window in under a pod that is still
// running.
func TestHeat_ARestampNeverShortensADeclarationAlreadyInForce(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	far := time.Unix(1750000000, 0).UTC().Add(6 * time.Hour)
	node := gb10Node("spark-a")
	node.Annotations = map[string]string{
		burninv1alpha1.AnnotationHeatExpected: heatExpectedID(run, far),
	}

	if stampHeatExpected(context.Background(), node, run, far.Add(-time.Hour)) {
		t.Error("a nearer expiry overwrote one already in force")
	}
	marker, ok := parseHeatExpected(node.Annotations[burninv1alpha1.AnnotationHeatExpected])
	if !ok || !marker.expiry.Equal(far) {
		t.Errorf("declaration expiry = %v, want it left at %v", marker.expiry, far)
	}

	if !stampHeatExpected(context.Background(), node, run, far.Add(time.Hour)) {
		t.Fatal("a later expiry did not extend the declaration; a long run would outlive it")
	}
	marker, _ = parseHeatExpected(node.Annotations[burninv1alpha1.AnnotationHeatExpected])
	if !marker.expiry.Equal(far.Add(time.Hour)) {
		t.Errorf("declaration expiry = %v, want %v", marker.expiry, far.Add(time.Hour))
	}
}

// ─── The key itself ───────────────────────────────────────────────────────────

// THE ONLY COORDINATION POINT WITH THE AGENT. A mismatch fails OPEN — the drain
// still happens, the pod still dies with SIGKILL, and the result looks exactly
// like the bug this change fixes. Pinned as a literal here so that changing it
// is a deliberate act with a matching change on the reader's side, rather than
// a rename that compiles.
func TestHeat_TheAnnotationKeyIsPinned(t *testing.T) {
	const want = "burnin.glimmer.ai/heat-expected"
	if burninv1alpha1.AnnotationHeatExpected != want {
		t.Errorf("AnnotationHeatExpected = %q, want %q — the Glimmer agent reads this key byte for "+
			"byte, and a mismatch silently restores the drain this change exists to stop",
			burninv1alpha1.AnnotationHeatExpected, want)
	}
}

// A node the run never cordoned is never declared for. The declaration says
// "this operator is loading this node right now", and saying it about a node the
// run does not hold would disable the watchdog on hardware nobody is testing.
func TestHeat_ANodeThisRunDoesNotHoldIsNeverDeclaredFor(t *testing.T) {
	other := gb10Node("spark-a")
	other.Spec.Unschedulable = true
	other.Annotations = map[string]string{
		burninv1alpha1.AnnotationCordonOwner:        "burnin/other-run/uid-other",
		burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableFalse,
	}
	h := newHarness(t,
		other,
		soakTest("thermal-soak", 900, 900),
		profile("acceptance", nil, false, testRef("thermal-soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	if raw, present := h.node("spark-a").Annotations[burninv1alpha1.AnnotationHeatExpected]; present {
		t.Errorf("run1 declared heat as %q on a node held by another run — it never cordoned it and "+
			"never put load on it", raw)
	}
}

// A run holding several nodes declares heat on every one of them. A pair or a
// group is admitted as a unit and loaded as a unit, so a declaration on only one
// end leaves the other's soak drainable.
func TestHeat_EveryNodeOfAWaveIsDeclaredFor(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		gb10Node("spark-b"),
		soakTest("thermal-soak", 900, 900),
		profile("acceptance", nil, false, testRef("thermal-soak")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	for _, name := range []string{"spark-a", "spark-b"} {
		node := h.node(name)
		if !node.Spec.Unschedulable {
			t.Fatalf("setup: %s was never cordoned, so there is no declaration owed", name)
		}
		assertDeclaresHeat(t, node, "uid-run1", h.nowVal)
	}
}

// A declaration this operator wrote must survive a round trip through a real
// annotation value. The apiserver stores it as an opaque string, so the only
// thing that can break the loop is the format itself.
func TestHeat_TheOperatorCanReadBackEveryValueItWrites(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "glimmer-burnin-system",
			Name:      "acceptance-2026-08-17",
			UID:       types.UID("3f2a1b0c-4d5e-6f70-8192-a3b4c5d6e7f8"),
		},
	}
	for _, expiry := range []time.Time{
		time.Unix(0, 0).UTC(),
		time.Unix(1750000000, 0).UTC(),
		time.Unix(1750000000, 0).In(time.FixedZone("IST", 5*60*60+30*60)),
		time.Date(2099, 12, 31, 23, 59, 59, 0, time.UTC),
	} {
		value := heatExpectedID(run, expiry)
		marker, ok := parseHeatExpected(value)
		if !ok {
			t.Fatalf("%q does not parse back", value)
		}
		if marker.namespace != run.Namespace || marker.name != run.Name || marker.uid != string(run.UID) {
			t.Errorf("%q parsed to %s/%s/%s, want %s/%s/%s", value,
				marker.namespace, marker.name, marker.uid, run.Namespace, run.Name, run.UID)
		}
		if !marker.expiry.Equal(expiry) {
			t.Errorf("%q parsed to expiry %s, want %s", value, marker.expiry, expiry)
		}
	}
}

// ─── Scope: only kinds that hold the part under load ──────────────────────────

// The declaration says a burn-in is deliberately holding this part under load
// RIGHT NOW. That is a claim about what the runner does, so it is the kind's
// answer — and a kind that takes a reading and stops must not make it.
//
// Both directions cost something real. Declaring for a passive kind suppresses a
// thermal response on a node nothing is loading, which is a safety action given
// away for nothing. Not declaring for a kind that holds a load is #280 again for
// that kind: the drain evicts it and the hardware comes back unjudged.
//
// The kinds here are the Node-scope ones a two-line fixture can express. WHICH
// kinds hold a load is settled exhaustively, and for every kind this project
// ships, by TestDrivesSustainedLoad in pkg/contract; this is about the wiring —
// that the answer reaches the node at all.
func TestHeat_OnlyAKindThatHoldsALoadDeclaresHeat(t *testing.T) {
	for _, tc := range []struct {
		kind    burninv1alpha1.TestKind
		declare bool
	}{
		{burninv1alpha1.KindThermalSoak, true},
		{burninv1alpha1.KindGPUBurn, true},
		{burninv1alpha1.KindClockProbe, true},
		{burninv1alpha1.KindMemoryStress, true},
		{burninv1alpha1.KindDCGMDiag, true},

		{burninv1alpha1.KindHostHealth, false},
		{burninv1alpha1.KindComputeSmoke, false},
		{burninv1alpha1.KindMemoryBW, false},
		{burninv1alpha1.KindCustom, false},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			test := loadTest("t", tc.kind, 300)
			if tc.kind == burninv1alpha1.KindCustom {
				// A custom kind needs an image; nothing else about it changes.
				test.Spec.Runner = &burninv1alpha1.RunnerSpec{Image: "example.invalid/whatever:v1"}
			}
			h := newHarness(t,
				gb10Node("spark-a"),
				test,
				profile("acceptance", nil, false, testRef("t")),
				newRun("run1", "acceptance", "spark-a"),
			)
			h.reconcile("run1")
			h.reconcile("run1")

			node := h.node("spark-a")
			if !node.Spec.Unschedulable {
				t.Fatalf("setup: %s never cordoned its target", tc.kind)
			}
			_, declared := heatOn(t, node)
			if declared != tc.declare {
				if tc.declare {
					t.Errorf("%s holds the part under load and declared no heat — the watchdog drains "+
						"the node and this kind comes back unjudged", tc.kind)
				} else {
					t.Errorf("%s declared heat, but it takes a reading rather than holding a load — a "+
						"thermal response is suppressed on a node nothing is heating", tc.kind)
				}
			}
		})
	}
}

// A run holds a node ACROSS its tests, so the same node passes from a soak to a
// passive read without being released in between. The soak's declaration has to
// be withdrawn when it does, or the drain stays held over a test that is not
// heating anything — for a whole pod window, bought by a test that never asked.
func TestHeat_TheDeclarationIsWithdrawnWhenTheWaveTurnsToAPassiveTest(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		soakTest("soak", 900, 900),
		loadTest("health", burninv1alpha1.KindHostHealth, 300),
		profile("acceptance", nil, false, testRef("soak"), testRef("health")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	assertDeclaresHeat(t, h.node("spark-a"), "uid-run1", h.nowVal)

	// Settle the soak and drive on until the second test has a pod.
	h.burnSegment("run1", 1, 0, cleanSegment())
	for i := 0; i < 12; i++ {
		h.reconcile("run1")
		if res := resultFor(h.run("run1"), "health", "spark-a"); res != nil && res.StartedAt != nil {
			break
		}
		if pod := h.pods("run1")["spark-a"]; pod != nil && pod.Labels[labelTest] == "health" {
			break
		}
	}

	if _, declared := heatOn(t, h.node("spark-a")); declared {
		t.Errorf("the soak's heat declaration outlived it onto a host-health read — the thermal drain "+
			"stays held over a test that heats nothing. node: %v", h.node("spark-a").Annotations)
	}
	if !h.node("spark-a").Spec.Unschedulable {
		t.Error("withdrawing the declaration also released the cordon; they are not the same thing")
	}
}

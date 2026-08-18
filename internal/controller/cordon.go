package controller

import (
	"context"
	"sort"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Cordon safety, controller side. The invariants and the annotation values live
// in api/v1alpha1/cordon.go so that the controller and its tests agree on them
// byte for byte; what follows is the machinery that honours them.
//
// The shape of the problem: burn-in saturates a node on purpose, so anything
// scheduled beside it is both starved and a corrupting influence on the
// measurement. The run therefore holds a node out of the scheduler for as long
// as it is putting load on it. That is a destructive edit to an object the
// operator does not own, and there are exactly two ways to get it wrong —
// uncordoning somebody else's node, and never uncordoning your own. Every
// function here is written against one of those two.
//
// Note what the cordon does NOT do: it does not keep the burn-in's own pod off
// the node. An unschedulable node repels pods that do not tolerate
// node.kubernetes.io/unschedulable, and the runner pod is the one workload that
// SHOULD land there, so it carries that toleration (see podForTest). Without it
// the run cordons its target and then waits forever for a pod the scheduler
// will never place — a deadlock the operator builds for itself.

// cordonOwnerID is the value written into AnnotationCordonOwner.
//
// It carries the UID and not just namespace/name because names are reusable: a
// deleted-and-recreated run of the same name is a different run, and must not
// inherit authority over a node the previous incarnation cordoned.
func cordonOwnerID(run *burninv1alpha1.BurnInRun) string {
	return run.Namespace + "/" + run.Name + "/" + string(run.UID)
}

// cordonOwner is a parsed AnnotationCordonOwner value.
type cordonOwner struct {
	namespace string
	name      string
	uid       string
}

// parseCordonOwner reads a stamp back into the run it names.
//
// It is STRICT — exactly three non-empty segments — and that is a safety
// property, not tidiness. The only consumer is the reaper, whose whole job is to
// remove a stamp on the strength of its owner being provably absent, and it can
// only prove absence of a run it can name exactly. A value that does not parse
// is left alone: it was not written by this operator, and "I cannot read this"
// is not evidence that the node is free.
func parseCordonOwner(value string) (cordonOwner, bool) {
	parts := strings.Split(value, "/")
	if len(parts) != 3 {
		return cordonOwner{}, false
	}
	for _, p := range parts {
		if p == "" {
			return cordonOwner{}, false
		}
	}
	return cordonOwner{namespace: parts[0], name: parts[1], uid: parts[2]}, true
}

// ─── The heat declaration ─────────────────────────────────────────────────────
//
// A burn-in drives a part past the temperature at which the Glimmer agent's
// thermal watchdog drains the node — by design, because that is what a soak is.
// The watchdog cannot tell that from a cooling failure on its own, so the
// operator says so on the node: AnnotationHeatExpected is present for as long as
// this run is loading it, and carries the instant after which the statement
// stops being true. See issue #280 and the annotation's own documentation.
//
// It rides the cordon's lifecycle exactly — same write, same release, same reap,
// same ownership rule — and that is the whole reason it is safe. The cordon
// lifecycle has already been hardened against a manager that dies mid-sequence
// (#59, #84), a stale informer (#245), a run deleted mid-flight (#60) and a
// release that outran its own load (#246). A second lifecycle would have to
// re-earn all of it.

// heatReleaseMargin is how much longer than its pod a heat declaration stands.
//
// The marker is taken off by the reconcile pass that finds the pod gone, so the
// window it must cover is the pod's own life PLUS one pass, and
// settledPollInterval is the longest this operator waits between passes while
// work is outstanding. Sized from that constant rather than restated, so a
// change to the poll cannot silently make every marker expire mid-run — which
// would drain a node during a soak, the exact failure this is here to stop.
const heatReleaseMargin = settledPollInterval

// heatExpiry is the absolute instant a declaration written now stops being true.
//
// It is derived from the run's OWN window and not from a fixed constant, because
// a 60-second smoke test and a seven-day segmented soak are the same code path
// and need very different answers. podWindow is the operator's own outer bound
// on one pod's life — the point at which podOverdue gives up and reaps it — so a
// marker sized from it cannot outlast a pod by more than the margin above, and
// cannot expire under a pod that is still within its budget.
func heatExpiry(now time.Time, spec *burninv1alpha1.BurnInTestSpec) time.Time {
	return now.Add(podWindow(spec) + heatReleaseMargin)
}

// heatExpectedID is the value written into AnnotationHeatExpected: the ownership
// triple cordonOwnerID produces, plus the expiry, as
// "<namespace>/<name>/<uid>/<RFC3339 expiry>".
//
// UTC, always. The reader is a separate program on the node comparing this
// against its own clock, and a local offset here is a difference of opinion
// about what the timestamp means that nothing in either half would notice.
func heatExpectedID(run *burninv1alpha1.BurnInRun, expiry time.Time) string {
	return cordonOwnerID(run) + "/" + expiry.UTC().Format(time.RFC3339)
}

// heatMarker is a parsed AnnotationHeatExpected value.
type heatMarker struct {
	cordonOwner
	expiry time.Time
}

// parseHeatExpected reads a declaration back into the run that made it and the
// instant it stops being true.
//
// STRICT, for parseCordonOwner's reason: the only consumers remove the marker on
// the strength of owning it, and a value that does not parse cannot be proved to
// belong to anyone. "I cannot read this" is not evidence that the heat is over.
//
// The expiry must parse too, and that is not tidiness either — an expiry the
// operator cannot read is one the AGENT cannot read, and the agent's rule is to
// refuse a marker it cannot read. A value that reaches here malformed is already
// doing nothing on the node; treating it as ours and rewriting it would be this
// operator laundering something it did not write into something that looks like
// a live declaration.
func parseHeatExpected(value string) (heatMarker, bool) {
	parts := strings.Split(value, "/")
	if len(parts) != 4 {
		return heatMarker{}, false
	}
	owner, ok := parseCordonOwner(strings.Join(parts[:3], "/"))
	if !ok {
		return heatMarker{}, false
	}
	expiry, err := time.Parse(time.RFC3339, parts[3])
	if err != nil {
		return heatMarker{}, false
	}
	return heatMarker{cordonOwner: owner, expiry: expiry}, true
}

// stampHeatExpected puts this run's declaration on a node it is holding, or
// pushes an existing one out to a later expiry, and reports whether it changed
// anything. The caller writes the node.
//
// EXTENDING IS THE POINT, not an optimisation. A run outlives one pod window
// whenever it is segmented, repeated or retried, and a declaration that covered
// only the first pod would expire under a soak that is still running — the
// watchdog would drain the node mid-measurement, which is the bug being fixed,
// arriving late instead of immediately. The expiry only ever moves FORWARD, so
// a marker never grants more than the window the run can currently justify and
// a re-stamp can never shorten one already in force.
//
// It will not overwrite a declaration it cannot prove is this run's. A marker
// naming another run is that run's thermal protection and removing it — which
// is what rewriting it amounts to — would hand that run's soak to the watchdog.
// One this operator cannot parse says nothing about anything, and is left as
// found for parseCordonOwner's reason. Both cases are logged, because the
// consequence is a soak of this run's that gets drained, and the node is the
// only place that would otherwise say why.
func stampHeatExpected(ctx context.Context, node *corev1.Node, run *burninv1alpha1.BurnInRun, expiry time.Time) bool {
	if raw, present := node.Annotations[burninv1alpha1.AnnotationHeatExpected]; present && raw != "" {
		existing, ok := parseHeatExpected(raw)
		switch {
		case !ok:
			log.FromContext(ctx).Info("node carries a heat declaration this operator cannot parse; leaving it alone",
				"node", node.Name, "marker", raw)
			return false
		case existing.uid != string(run.UID):
			log.FromContext(ctx).Info("node carries another run's heat declaration; leaving it alone",
				"node", node.Name, "marker", raw)
			return false
		case !existing.expiry.Before(expiry):
			// Already covers the window this pod needs.
			return false
		}
	}
	if node.Annotations == nil {
		node.Annotations = map[string]string{}
	}
	node.Annotations[burninv1alpha1.AnnotationHeatExpected] = heatExpectedID(run, expiry)
	return true
}

// dropHeatExpected removes a heat declaration belonging to `uid` from the node
// in memory, so the caller's update carries it away with the cordon.
//
// TOGETHER OR NOT AT ALL. A node handed back with the declaration still on it is
// a node the scheduler is filling with ordinary workload while its thermal
// watchdog is switched off, and that is worse than the failure the marker exists
// to fix: a soak dying to a drain is a lost measurement, a cooling failure
// nothing acts on is lost hardware.
//
// The exception is the same one everywhere else in this file: authority is by
// UID. A marker naming a different run is that run's protection and is left
// alone, and one that does not parse is left alone because it is not evidence.
// It reports whether it removed anything, so a caller whose only reason to write
// the node is the withdrawal can skip the update.
func dropHeatExpected(ctx context.Context, node *corev1.Node, uid string) bool {
	raw, present := node.Annotations[burninv1alpha1.AnnotationHeatExpected]
	if !present {
		return false
	}
	marker, ok := parseHeatExpected(raw)
	switch {
	case !ok:
		log.FromContext(ctx).Info("node carries a heat declaration this operator cannot parse; leaving it alone",
			"node", node.Name, "marker", raw)
	case marker.uid != uid:
		log.FromContext(ctx).Info("node carries another run's heat declaration; leaving it alone",
			"node", node.Name, "marker", raw)
	default:
		delete(node.Annotations, burninv1alpha1.AnnotationHeatExpected)
		return true
	}
	return false
}

// capturePriorSchedulability records how the run FOUND its targets, before it
// has touched any of them. It is called once, from markRunning, and its result
// is written to status.priorUnschedulable and never recomputed.
//
// This is the whole fix for the stranded cordon. "Was this node already out of
// the scheduler?" is a question about the moment the run arrived, and the only
// place it can be answered correctly is before the run has a footprint of its
// own. Asked later — on the second wave, or by a manager that has just restarted
// — the reading includes the run's own cordon, or somebody's cordon placed over
// the top of it, and there is no way to tell the two apart from the node alone.
// A run that answers "yes" then honours that answer at teardown and leaves the
// node unschedulable forever, which is the exact failure the restore-on-terminal
// finalizer exists to prevent.
//
// A node that does not exist yet is simply absent from the map; cordonNode falls
// back to reading it, which is what the operator did everywhere before.
//
// IT READS UNCACHED, AND THAT IS NOT AN OPTIMISATION. Capturing at the right
// MOMENT — before this run's first cordon — is only half the requirement; the
// other half is reading the node as it actually is at that moment. An informer
// still holding the version a run five seconds ago cordoned answers "already
// unschedulable" about a node that has since been released, the run records
// that as pre-existing, and teardown faithfully makes it permanent. The
// contamination the comment in markRunning warns about therefore does not only
// come from this run's own footprint: it comes from the PREVIOUS run's, through
// the cache. See issue #84.
func (r *BurnInRunReconciler) capturePriorSchedulability(ctx context.Context, targets []string) map[string]bool {
	prior := map[string]bool{}
	for _, name := range targets {
		var node corev1.Node
		if err := r.uncached().Get(ctx, types.NamespacedName{Name: name}, &node); err != nil {
			continue
		}
		prior[name] = preBurnInSchedulability(&node)
	}
	if len(prior) == 0 {
		return nil
	}
	return prior
}

// preBurnInSchedulability answers "what was this node before ANY burn-in
// touched it", which is a different and stricter question than "is it
// unschedulable now".
//
// A NODE CARRYING A CORDON STAMP IS NOT IN ITS PRIOR STATE, BY DEFINITION. The
// stamp says a burn-in put it where it is, and it comes with that run's own
// record of how the node was found. That record — not the live spec — is the
// node's pre-burn-in state, and copying it is the only reading that cannot
// launder somebody's transient cordon into a permanent one.
//
// This is issue #80, and it is worth stating what it looked like, because the
// bug is not in any of the code it passed through. Three runs targeted the same
// two nodes. The first found spark-85a9 schedulable and cordoned it. The second
// and third captured their prior state while that cordon was held, read
// `unschedulable=true`, and recorded it as the node's PRE-EXISTING state. The
// release path then did exactly what its bookkeeping told it — it logged
// `restoredUnschedulable: true` — and re-asserted a cordon whose real owner had
// already let go. The state is self-perpetuating: every overlapping run
// re-observes the cordon and re-records prior=true, and the last one to release
// also clears its own annotations, so the node ends unschedulable with NO stamp
// at all. That end state is strictly worse than an orphaned stamp, because there
// is nothing left for a reaper to key on and nothing in the operator's telemetry
// that says the node is stranded. Half a two-node fleet, gone, silently.
//
// Admission (see admission.go) now stops overlapping runs from existing in the
// first place, and this is the second line: spec.force deliberately allows the
// overlap, and a forced run must still not be able to make another run's cordon
// permanent.
//
// THE NEGATIVE MATTERS AS MUCH AS THE POSITIVE, and it is the reason this reads
// the stamp rather than simply reporting `false` for any cordoned node. A node
// an administrator drained, or that another controller cordoned, carries NO
// stamp — and it must still be recorded as prior=true and left cordoned on the
// way out. Returning a node under maintenance to service because a burn-in
// finished is the failure this whole file's first invariant exists to prevent.
//
// A stamp with no accompanying prior record is treated as prior=false. The two
// annotations are written in a single update and cannot come apart on their own,
// so this means the pair was edited out of band — and the one thing the stamp
// still attests is that a BURN-IN cordoned the node, never an administrator.
// Burn-in cordons are always temporary, so the node is owed back.
func preBurnInSchedulability(node *corev1.Node) bool {
	if _, stamped := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; stamped {
		return node.Annotations[burninv1alpha1.AnnotationPriorUnschedulable] == burninv1alpha1.PriorUnschedulableTrue
	}
	return node.Spec.Unschedulable
}

// priorUnschedulable answers, for one node, what the run must restore it to.
//
// The run's own start-time record wins whenever it has one, because it is the
// only reading taken before the run could contaminate it. `observed` is the
// node's live spec.unschedulable and is used only when there is no record —
// a target that did not exist at start, or a run that began under an operator
// version that did not capture one.
//
// The deliberate consequence, stated so it is not mistaken for an oversight: a
// cordon that appears on a target AFTER the run started is not adopted. If an
// operator cordons a node the burn-in is already holding, the burn-in still
// restores the node at the end and the cordon has to be re-applied. That trade
// is the right way round — a cordon undone is visible and one command to redo,
// while a node the fleet silently loses has nothing left in the cluster that
// knows it was taken — and it is also the only self-consistent answer, since a
// cordon placed on an already-cordoned node is a no-op the operator cannot see.
func priorUnschedulable(run *burninv1alpha1.BurnInRun, name string, observed bool) bool {
	if recorded, ok := run.Status.PriorUnschedulable[name]; ok {
		return recorded
	}
	return observed
}

// restoredUnschedulable is the schedulability a release must put a node back to:
// the run's start-time record, falling back to the value stamped on the node.
//
// The two agree in the ordinary case, since the stamp is written from the record.
// They disagree exactly where it matters — a node stamped by an operator version
// that derived prior state per wave, or a stamp edited out of band — and there
// the run's own record is the one taken before the run had a footprint, so it
// wins. The node-side stamp stays authoritative when there is no record, because
// it is the only account a run started under an older operator ever had.
func restoredUnschedulable(run *burninv1alpha1.BurnInRun, node *corev1.Node) bool {
	stamped := node.Annotations[burninv1alpha1.AnnotationPriorUnschedulable] == burninv1alpha1.PriorUnschedulableTrue
	return priorUnschedulable(run, node.Name, stamped)
}

// cordonNode holds ONE node out of the scheduler and records the ownership that
// makes the release safe.
//
// THE UNIT OF CORDONING IS THE WAVE, NOT THE TARGET LIST. A node is cordoned
// immediately before the run puts load on it and released once it is no longer
// holding any, so the number of nodes a run has out of service tracks
// MaxConcurrentNodes rather than the size of the target list. An earlier version
// cordoned every target at start; on a two-node cluster that took the entire
// cluster out of scheduling for the whole run, and it hollowed out the
// interlock, whose entire premise is that only N nodes are occupied at once.
//
// The trade this makes, stated plainly: a node the run will come back to for a
// later test is released in between, so there is a window — one reconcile pass
// — in which the scheduler could place a workload on it. That is narrower than
// it sounds (it requires a pending unschedulable pod at that exact moment) and
// it is the price of not holding a fleet hostage to a long sweep. The cordon
// always goes on BEFORE the pod is created, so nothing lands beside a burn-in
// that is already running.
//
// The write order is the important part and is not negotiable: the ownership
// stamp and the prior-state record go on FIRST, in their own update, and only
// then is the node made unschedulable. A crash between the two leaves a stamped
// node that is still schedulable — harmless, and cleanable, because the stamp
// is what cleanup searches on. The other order leaves a cordoned node with no
// owner, which nothing in the system is able to release.
//
// It also declares the heat: the node is about to be driven at full power on
// purpose, and the agent's thermal watchdog has to be told so it does not read a
// soak as a cooling failure and drain the node out from under it. `spec` is the
// test whose pod is about to be created, and it is what sizes the declaration's
// expiry — see heatExpiry. It travels with the node name rather than being
// derived here because a run is many tests long and their windows differ by
// orders of magnitude.
//
// It mutates run.Status.CordonedNodes and reports whether it changed anything;
// the caller persists the status.
func (r *BurnInRunReconciler) cordonNode(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	name string,
	spec *burninv1alpha1.BurnInTestSpec,
) (bool, error) {
	logger := log.FromContext(ctx)
	owner := cordonOwnerID(run)
	expiry := heatExpiry(r.now(), spec)

	// WHETHER THE HEAT IS DECLARED AT ALL IS THE KIND'S ANSWER, NOT THIS
	// FUNCTION'S. The declaration says a burn-in is deliberately holding this
	// part under load right now, and that is only true of kinds whose runner
	// holds a load — see TestKind.DrivesSustainedLoad. A host-health read or a
	// millisecond GEMM would be claiming heat it does not produce, and it would
	// suppress a real thermal response on a node that was not being loaded.
	declareHeat := spec.Kind.DrivesSustainedLoad()
	held := map[string]bool{}
	for _, n := range run.Status.CordonedNodes {
		held[n] = true
	}

	// UNCACHED, for the same reason as releaseNode and with a worse consequence.
	// Two of the three arms below conclude from this read and DECIDE NOT TO
	// ACT — "the node does not exist" and "somebody else holds it" — and neither
	// writes anything, so no resourceVersion check stands behind either. But
	// unlike the release path, the run does not stop when cordoning declines:
	// the wave has already admitted this node and the pod is created next. An
	// informer that has not yet observed a freshly created node answers NotFound,
	// and one holding a version from before another owner released it answers
	// with a stamp that is gone. Either way the run puts a full-power runner pod
	// on a node it never took out of the scheduler, which is the interlock's
	// whole purpose — and having recorded no hold, it never gives the node back
	// either.
	var node corev1.Node
	if err := r.uncached().Get(ctx, types.NamespacedName{Name: name}, &node); err != nil {
		if apierrors.IsNotFound(err) {
			// Nothing to hold, and nothing owed on the way out. The test
			// targeting this node will fail to schedule and be recorded as an
			// Error, which is the honest report of a target that is not there.
			logger.Info("target node does not exist; not cordoning", "node", name)
			return false, nil
		}
		return false, err
	}

	switch existing := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; {
	case existing == owner:
		// Ours already. Re-assert the cordon in case a crash landed between the
		// stamp and the cordon, push the heat declaration out to cover the pod
		// about to be created, and make sure we still owe the release.
		//
		// This is the arm every segment of a soak after the first lands in, and
		// therefore the one that keeps a multi-day run declared. One update
		// carries both, so the node never goes through a state where it is
		// cordoned for a run whose declaration has lapsed.
		//
		// It is also where a declaration is WITHDRAWN. A run holds a node across
		// its tests, so the same node passes from a soak to a host-health read
		// without being released in between; leaving the soak's declaration
		// standing would keep the drain held over a test that is not heating
		// anything. Expiry would clear it eventually, but "eventually" is a
		// whole pod window of suppressed thermal response bought by a test that
		// never asked for it.
		reasserted := false
		if declareHeat {
			reasserted = stampHeatExpected(ctx, &node, run, expiry)
		} else if dropHeatExpected(ctx, &node, string(run.UID)) {
			reasserted = true
			logger.Info("this test does not hold the part under load; withdrawing the heat declaration",
				"node", name, "kind", spec.Kind)
		}
		if !node.Spec.Unschedulable {
			node.Spec.Unschedulable = true
			reasserted = true
		}
		if reasserted {
			if err := r.Update(ctx, &node); err != nil {
				return false, err
			}
		}
		held[name] = true

	case existing != "":
		// Somebody else's — another run, or an administrator's tooling that
		// uses the same stamp. Do not take it, do not overwrite the stamp, and
		// above all do not record it as ours: recording it would make this run
		// release a hold it never placed. The node is already out of the
		// scheduler, which is the condition the test needs anyway.
		logger.Info("node is already cordoned by another owner; leaving it alone",
			"node", name, "owner", existing)

	default:
		// Unowned. Stamp first, cordon second.
		//
		// The prior-state value comes from the RUN's start-time record, not from
		// what the node says right now. A node is cordoned and released once per
		// wave, so this branch runs repeatedly across a run, and re-deriving the
		// answer here would let the run's own cordon — or anything laid over it
		// while the run held the node — be recorded as pre-existing and then made
		// permanent at teardown. See priorUnschedulable.
		if node.Annotations == nil {
			node.Annotations = map[string]string{}
		}
		node.Annotations[burninv1alpha1.AnnotationCordonOwner] = owner
		node.Annotations[burninv1alpha1.AnnotationPriorUnschedulable] =
			strconv.FormatBool(priorUnschedulable(run, name, node.Spec.Unschedulable))
		if declareHeat {
			stampHeatExpected(ctx, &node, run, expiry)
		} else {
			// A node this run has not held before can still carry a stale
			// declaration of ours — a manager that died between the release's
			// two writes, say. Take it with the same update that claims the
			// node, so a cordon of ours never coexists with a declaration this
			// test cannot justify.
			dropHeatExpected(ctx, &node, string(run.UID))
		}
		if err := r.Update(ctx, &node); err != nil {
			return false, err
		}
		if !node.Spec.Unschedulable {
			node.Spec.Unschedulable = true
			if err := r.Update(ctx, &node); err != nil {
				return false, err
			}
		}
		held[name] = true
	}

	updated := sortedNames(held)
	if equalNames(run.Status.CordonedNodes, updated) {
		return false, nil
	}
	run.Status.CordonedNodes = updated
	return true, nil
}

// cordonWave holds every node of one execution unit — a single node at Node
// scope, both nodes of a pair at Pair scope — and reports whether anything
// changed.
//
// A pair is cordoned as a UNIT and before either pod is created, because a pair
// under test occupies both of its nodes for the whole test: releasing or
// admitting half of one would be a state the test cannot be run in.
func (r *BurnInRunReconciler) cordonWave(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	nodes []string,
	spec *burninv1alpha1.BurnInTestSpec,
) (bool, error) {
	changed := false
	for _, n := range nodes {
		c, err := r.cordonNode(ctx, run, n, spec)
		if err != nil {
			return changed, err
		}
		changed = changed || c
	}
	return changed, nil
}

// releaseHeldNodes gives back every node this run has cordoned that is no longer
// under its load, and reports whether it changed anything.
//
// held is the run's current wave: the nodes that have a live pod or have just
// been admitted under the interlock. Anything else this run is holding is
// capacity it is sitting on for no reason, and the fleet should have it back
// immediately rather than at run teardown — which for a multi-hour sweep is
// hours later, and for a wedged run is never.
//
// It mutates run.Status.CordonedNodes; the caller persists it.
func (r *BurnInRunReconciler) releaseHeldNodes(ctx context.Context, run *burninv1alpha1.BurnInRun, held map[string]bool) (bool, error) {
	changed := false
	for _, name := range append([]string(nil), run.Status.CordonedNodes...) {
		if held[name] {
			continue
		}
		c, err := r.releaseNode(ctx, run, name)
		if err != nil {
			return changed, err
		}
		changed = changed || c
	}
	return changed, nil
}

// releaseNode restores ONE node this run owns and drops it from the run's
// record. It is the single-node form of releaseCordons and obeys the same rule:
// restore the PRIOR state, never blindly uncordon, and never touch a node
// stamped by somebody else.
func (r *BurnInRunReconciler) releaseNode(ctx context.Context, run *burninv1alpha1.BurnInRun, name string) (bool, error) {
	logger := log.FromContext(ctx)
	owner := cordonOwnerID(run)

	// UNCACHED, because only ONE of the three arms below is an Update. That is
	// the rule that decides which reads go uncached at all: a stale read whose
	// only consequence is an Update of the SAME object cannot become a stale
	// write, since the apiserver refuses it on resourceVersion and the next pass
	// reads again — but a stale read used to decide NOT to act has no such check
	// standing behind it, and neither does one whose answer is RECORDED
	// somewhere else. This read is both. The two non-default arms write nothing
	// to the node and still drop it from status.CordonedNodes, so an informer
	// holding the version from before this run stamped the node — or one that
	// has not observed the node at all, which answers NotFound — makes the run
	// disown a node it is still holding with its own stamp on it. Nothing owes
	// that node back until the run cordons it again or reaches teardown, which
	// for a single-test profile or a long soak is the rest of the run, and
	// status.CordonedNodes says the fleet has it. Same class as
	// capturePriorSchedulability and releaseCordons' scan; see issue #245.
	var node corev1.Node
	err := r.uncached().Get(ctx, types.NamespacedName{Name: name}, &node)
	switch {
	case apierrors.IsNotFound(err):
		// The node is gone; there is nothing to restore and nothing to owe.
	case err != nil:
		return false, err
	case node.Annotations[burninv1alpha1.AnnotationCordonOwner] != owner:
		// Not ours (any more). Drop the claim rather than the node's state.
	default:
		prior := restoredUnschedulable(run, &node)
		node.Spec.Unschedulable = prior
		delete(node.Annotations, burninv1alpha1.AnnotationCordonOwner)
		delete(node.Annotations, burninv1alpha1.AnnotationPriorUnschedulable)
		dropHeatExpected(ctx, &node, string(run.UID))
		if updErr := r.Update(ctx, &node); updErr != nil && !apierrors.IsNotFound(updErr) {
			return false, updErr
		}
		logger.Info("released burn-in cordon", "node", name, "restoredUnschedulable", prior)
	}

	kept := make([]string, 0, len(run.Status.CordonedNodes))
	for _, n := range run.Status.CordonedNodes {
		if n != name {
			kept = append(kept, n)
		}
	}
	if len(kept) == len(run.Status.CordonedNodes) {
		return false, nil
	}
	if len(kept) == 0 {
		kept = nil
	}
	run.Status.CordonedNodes = kept
	return true, nil
}

func equalNames(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// releaseCordons returns every node this run is holding, and reports whether it
// changed anything.
//
// It searches by ANNOTATION over the live node list rather than by walking
// status.CordonedNodes, and that is the whole recovery story. The stamp is
// written before the cordon and before the status, so a manager that died
// anywhere in that sequence left evidence on the node and possibly none in the
// run. A cleanup driven by status alone would strand exactly those nodes —
// permanently, since nothing else in the cluster knows they are held. Scanning
// nodes makes the release level-based: whatever the controller crashed in the
// middle of, the next pass finds it.
//
// The release RESTORES rather than uncordons. A node that was already
// unschedulable when the run found it was somebody's deliberate act — a drain
// for a firmware update, most likely — and returning it to service because a
// burn-in finished is how a node under maintenance starts receiving production
// traffic mid-update.
//
// The scan is UNCACHED. Scanning at all is the recovery story; scanning a cache
// is a recovery story with a hole in it, because the informer can be holding a
// node version from before this run stamped it. The release would then find
// nothing to give back, drop its finalizer, and leave a cordoned node with a
// live owner annotation and no object left in the cluster that owes it a
// release — the precise strand this function exists to prevent. See #59 and #84.
//
// It mutates run.Status.CordonedNodes; the caller persists it.
func (r *BurnInRunReconciler) releaseCordons(ctx context.Context, run *burninv1alpha1.BurnInRun) (bool, error) {
	logger := log.FromContext(ctx)
	owner := cordonOwnerID(run)

	var nodes corev1.NodeList
	if err := r.uncached().List(ctx, &nodes); err != nil {
		return false, err
	}

	changed := false
	for i := range nodes.Items {
		node := &nodes.Items[i]
		if node.Annotations[burninv1alpha1.AnnotationCordonOwner] != owner {
			continue
		}
		prior := restoredUnschedulable(run, node)

		// One update carries the restored schedulability, the removal of the
		// stamp and the removal of the heat declaration. Splitting them would
		// open a window where the node is unstamped and still cordoned, which
		// is the unrecoverable state this whole scheme exists to avoid — or one
		// where it is back in the scheduler with its thermal watchdog still
		// told to stand down, which is worse.
		node.Spec.Unschedulable = prior
		delete(node.Annotations, burninv1alpha1.AnnotationCordonOwner)
		delete(node.Annotations, burninv1alpha1.AnnotationPriorUnschedulable)
		dropHeatExpected(ctx, node, string(run.UID))
		if err := r.Update(ctx, node); err != nil {
			if apierrors.IsNotFound(err) {
				// The node was deleted under us. There is nothing left to
				// restore, and refusing to move on would leave the run holding
				// a finalizer for a node that does not exist — trading a stuck
				// node for a stuck object.
				changed = true
				continue
			}
			return changed, err
		}
		logger.Info("released burn-in cordon", "node", node.Name, "restoredUnschedulable", prior)
		changed = true
	}

	if len(run.Status.CordonedNodes) > 0 {
		run.Status.CordonedNodes = nil
		changed = true
	}
	return changed, nil
}

func sortedNames(set map[string]bool) []string {
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for n := range set {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

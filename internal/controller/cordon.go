package controller

import (
	"context"
	"sort"
	"strconv"

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
		prior[name] = node.Spec.Unschedulable
	}
	if len(prior) == 0 {
		return nil
	}
	return prior
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
// It mutates run.Status.CordonedNodes and reports whether it changed anything;
// the caller persists the status.
func (r *BurnInRunReconciler) cordonNode(ctx context.Context, run *burninv1alpha1.BurnInRun, name string) (bool, error) {
	logger := log.FromContext(ctx)
	owner := cordonOwnerID(run)
	held := map[string]bool{}
	for _, n := range run.Status.CordonedNodes {
		held[n] = true
	}

	var node corev1.Node
	if err := r.Get(ctx, types.NamespacedName{Name: name}, &node); err != nil {
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
		// stamp and the cordon, and make sure we still owe the release.
		if !node.Spec.Unschedulable {
			node.Spec.Unschedulable = true
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
func (r *BurnInRunReconciler) cordonWave(ctx context.Context, run *burninv1alpha1.BurnInRun, nodes []string) (bool, error) {
	changed := false
	for _, n := range nodes {
		c, err := r.cordonNode(ctx, run, n)
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

	// Cached, deliberately, and the reason is the rule that decides which reads
	// go uncached at all. What follows this read is an Update of the SAME
	// object, so a stale reading cannot become a stale write: the apiserver
	// refuses it on resourceVersion and the next pass reads again. The reads
	// that had to move off the cache are the ones whose answer is RECORDED
	// somewhere else (capturePriorSchedulability, into the run's status) or used
	// to decide NOT to act (releaseCordons' scan, which finds nothing and
	// concludes there is nothing to give back). Neither of those has an
	// optimistic-concurrency check standing behind it.
	var node corev1.Node
	err := r.Get(ctx, types.NamespacedName{Name: name}, &node)
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

		// One update carries both the restored schedulability and the removal
		// of the stamp. Splitting them would open a window where the node is
		// unstamped and still cordoned, which is the unrecoverable state this
		// whole scheme exists to avoid.
		node.Spec.Unschedulable = prior
		delete(node.Annotations, burninv1alpha1.AnnotationCordonOwner)
		delete(node.Annotations, burninv1alpha1.AnnotationPriorUnschedulable)
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

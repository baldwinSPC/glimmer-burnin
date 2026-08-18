package controller

import (
	"context"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The orphaned-cordon reaper. Issue #60.
//
// The ownership rule is deliberately strict: never uncordon a node this run did
// not cordon. That strictness is right, and it has one consequence nothing else
// in the system covers — a stamp naming a BurnInRun that no longer exists takes
// its node out of every future run PERMANENTLY. The node stays unschedulable,
// and every subsequent run correctly refuses to touch it, because none of them
// owns it. It is invisible: the node looks cordoned-by-something and every
// burn-in politely leaves it alone. On a two-node cluster that is half the
// fleet, and the only recovery is a human noticing and running `kubectl
// uncordon`.
//
// WHY IT LIVES ON THE NODE SIDE. The obvious home is the BurnInRun reconciler,
// and it is the wrong one: that reconciler only runs when some run reconciles,
// so a cluster with no runs at all — exactly the state a stranded node is left
// in, and exactly when nobody is looking — would never sweep. The NodeFingerprint
// reconciler already watches Nodes, so the sweep costs nothing extra to trigger
// and fires on every node the manager sees at startup, which is the other moment
// orphans surface.

// cordonReapInterval is how often a node carrying a live burn-in stamp is
// re-examined.
//
// A stamp becomes an orphan when its run disappears, and a run disappearing
// produces no Node event at all — so nothing but a timer will ever notice. It
// is deliberately slow: an orphaned cordon is a capacity problem measured in
// hours of a human not noticing, not a race, and re-reading the apiserver for
// every cordoned node in a large fleet is a cost worth keeping small. Only nodes
// that actually carry a stamp are polled; the rest requeue not at all.
const cordonReapInterval = 5 * time.Minute

// reapOrphanedCordon removes a burn-in cordon stamp whose owning run is provably
// gone, restoring the node to the schedulability that stamp records.
//
// It reports whether a stamp is still present and live after the pass, which is
// what tells the caller to come back and look again.
//
// THE ABSENCE CHECK IS AN UNCACHED READ OF THE SPECIFIC RUN, and nothing weaker
// is acceptable. A cache MISS is not absence — an informer that has not yet
// observed a freshly created run would report exactly that, and reaping on it
// would hand the node's cordon to whoever reaps next while a live run is still
// driving the hardware. That is the precise failure the ownership rule exists to
// prevent, reintroduced by the code meant to repair it.
//
// The match is on name AND UID, because the annotation encodes both and names
// are reusable. A run deleted and recreated under the same name is a DIFFERENT
// run; it never cordoned this node, it owes it nothing, and letting it inherit
// authority over the node would be the same violation by another route.
func reapOrphanedCordon(
	ctx context.Context,
	c client.Client,
	uncached client.Reader,
	node *corev1.Node,
) (stampLive bool, err error) {
	logger := log.FromContext(ctx)

	stamp, ok := node.Annotations[burninv1alpha1.AnnotationCordonOwner]
	if !ok {
		return false, nil
	}
	owner, parsed := parseCordonOwner(stamp)
	if !parsed {
		// Not a value this operator wrote. Leaving it is the only safe move: an
		// unreadable stamp is not evidence that the node is free, and something
		// else in the cluster may be relying on it.
		logger.Info("node carries a cordon stamp this operator cannot parse; leaving it alone",
			"node", node.Name, "stamp", stamp)
		return false, nil
	}

	var run burninv1alpha1.BurnInRun
	getErr := uncached.Get(ctx, types.NamespacedName{Namespace: owner.namespace, Name: owner.name}, &run)
	switch {
	case getErr == nil && string(run.UID) == owner.uid:
		// The owner is alive. It will release the node itself, through its own
		// finalizer, and this must not help.
		return true, nil
	case getErr == nil:
		logger.Info("cordon stamp names a run that has been recreated under a new UID; reaping",
			"node", node.Name, "run", owner.namespace+"/"+owner.name,
			"stampedUID", owner.uid, "liveUID", string(run.UID))
	case apierrors.IsNotFound(getErr):
		logger.Info("cordon stamp names a BurnInRun that no longer exists; reaping",
			"node", node.Name, "run", owner.namespace+"/"+owner.name, "uid", owner.uid)
	default:
		// An inconclusive read is not an absence. Retry rather than guess.
		return true, getErr
	}

	// The stamp's own record of how the run found the node is what the release
	// restores to — the same rule the owning run would have followed, so a reaped
	// node ends up in the state it would have had if the run had lived to release
	// it. A node an administrator had already drained stays drained.
	prior := node.Annotations[burninv1alpha1.AnnotationPriorUnschedulable] == burninv1alpha1.PriorUnschedulableTrue

	// One update carries the restored schedulability AND the removal of all
	// three annotations, exactly as releaseCordons does. Splitting them would
	// leave a window in which the node is cordoned and unstamped, which is the
	// one state nothing in this system can recover from.
	//
	// The heat declaration goes with them, under the same UID check: the run
	// that made it is the run just proved absent, so it will never be back to
	// take it off, and a node returned to the scheduler still declaring heat is
	// a node the thermal watchdog has been told to ignore while ordinary
	// workload lands on it.
	//
	// A declaration with no cordon stamp beside it is not reached here at all —
	// the stamp is what this function keys on. That pair can only come apart by
	// an out-of-band edit, and the expiry is what covers it: an abandoned
	// marker stops being honoured on its own, which is the property it carries
	// an expiry for.
	node.Spec.Unschedulable = prior
	delete(node.Annotations, burninv1alpha1.AnnotationCordonOwner)
	delete(node.Annotations, burninv1alpha1.AnnotationPriorUnschedulable)
	dropHeatExpected(ctx, node, owner.uid)
	if updErr := c.Update(ctx, node); updErr != nil {
		if apierrors.IsNotFound(updErr) {
			return false, nil
		}
		// A conflict here is the healthy outcome, not a problem: it means the
		// node moved under us — most likely a live run stamping it — and the
		// next pass re-reads before deciding anything.
		return true, updErr
	}
	logger.Info("reaped an orphaned burn-in cordon", "node", node.Name,
		"orphanedOwner", stamp, "restoredUnschedulable", prior)
	return false, nil
}

// reapCordon is the NodeFingerprint reconciler's hook into the reaper.
//
// The node object it is handed comes from the informer cache, and that is safe
// here for the reason stated in releaseNode: what follows a positive decision is
// an Update of the SAME object, so a stale reading cannot become a stale write —
// the apiserver refuses it on resourceVersion and the next pass reads again. The
// read that had to be uncached is the one that decides, which is the read of the
// owning RUN, and that one is.
func (r *NodeFingerprintReconciler) reapCordon(ctx context.Context, node *corev1.Node) (ctrl.Result, error) {
	live, err := reapOrphanedCordon(ctx, r.Client, r.uncached(), node)
	if err != nil {
		return ctrl.Result{}, err
	}
	if live {
		// A stamp whose owner is alive today is an orphan tomorrow if that run is
		// force-deleted, and a run disappearing raises no Node event. The timer is
		// the only thing that will ever come back to look.
		return ctrl.Result{RequeueAfter: cordonReapInterval}, nil
	}
	return ctrl.Result{}, nil
}

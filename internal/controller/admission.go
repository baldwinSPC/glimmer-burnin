package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Admission: the cross-run interlock.
//
// MaxConcurrentNodes bounds concurrency WITHIN a run. Until this file existed
// nothing bounded it ACROSS runs, and the two are not the same guarantee. The
// cap is a facility interlock — N nodes at full power is N nodes' worth of heat
// and current in a real room — and two runs each honouring a cap of 1 on the
// SAME node defeat it entirely while each believes it is compliant.
//
// The cordon cannot mediate this and never could. Runner pods must tolerate
// node.kubernetes.io/unschedulable, because the operator cordons the node it is
// about to test and without the toleration the run would deadlock waiting for a
// pod the scheduler will never place. That same toleration means a second run's
// pods land happily on the first run's cordoned node. The cordon keeps OTHER
// PEOPLE's workloads off a node under test; it says nothing about a second
// burn-in. So the guard has to be admission, before any pod exists.
//
// Observed on a two-node GB10 cluster (issue #81): three BurnInRuns created
// seconds apart against the same two nodes, every one of them failing
// thermal-soak with exit 137 — the kubelet killing sustained full-load soaks
// that were competing for one machine. The operator reported Error, correctly,
// and an operator reading it had no way to tell that the cause was another run.
//
// REFUSE, DO NOT QUEUE. A burn-in is scheduled maintenance against hardware
// somebody is waiting on, and a run that silently waits behind another has an
// unpredictable duration and an unpredictable end time — which is precisely
// what a maintenance window cannot tolerate. A refusal is immediate, terminal
// and legible, and re-running is one `kubectl create` away. If queueing is ever
// wanted it belongs behind an explicit policy field, never as the default.

// admissionOrder decides which of two runs contending for the same node wins.
//
// It exists for the dead heat, and the dead heat is the common case: a control
// plane dispatching runs, or `kubectl apply -f dir/`, creates them within the
// same second, so both are Pending and neither has cordoned anything when the
// other looks. Without a total order both would look at the other, see a run
// that has not started, admit itself, and the interlock would hold only when it
// was not needed.
//
// A run that has STARTED always wins, whatever the clock says — it already has
// pods on hardware, and there is no reading of "first" under which the run
// holding the node is not it. Otherwise the older creation timestamp wins, and
// UID breaks the tie because creation timestamps have one-second granularity
// and two runs created in the same second must still order deterministically.
// It must be a strict total order: were it ever to report that neither of two
// runs precedes the other, both would admit.
func admissionOrder(a, b *burninv1alpha1.BurnInRun) bool {
	aStarted, bStarted := a.Status.StartedAt != nil, b.Status.StartedAt != nil
	if aStarted != bStarted {
		return aStarted
	}
	if !a.CreationTimestamp.Time.Equal(b.CreationTimestamp.Time) {
		return a.CreationTimestamp.Time.Before(b.CreationTimestamp.Time)
	}
	return string(a.UID) < string(b.UID)
}

// runHolds reports whether a run is in a position to be occupying nodes.
//
// The finalizer is the authority, not the phase, and the difference is not
// academic. The cordon finalizer IS the standing record that a run still owes
// the cluster something — a node to give back, a pod to stop — and it is
// removed only after both are done. A run that is terminal, or being deleted,
// but still carrying it may still have a runner at full power on the node; a
// deletion is not instantaneous, and `kubectl delete burninrun` on a
// misbehaving run is precisely the moment somebody launches a replacement. A
// phase-only check would admit that replacement onto hardware the first run has
// not let go of yet.
//
// A run that has not started has no finalizer and is caught by the phase
// instead: it has not taken anything, but it is next in line for these nodes and
// the admission order is what decides between the two.
//
// The cost of being this conservative is that a run wedged in deletion — a
// finalizer that cannot complete because a node is unreachable — blocks new runs
// on its targets. That is the honest answer while it is genuinely still holding
// them, it is visible (the object is right there, stuck), and spec.force is the
// documented way past it.
func runHolds(run *burninv1alpha1.BurnInRun) bool {
	if controllerutil.ContainsFinalizer(run, burninv1alpha1.FinalizerCordonCleanup) {
		return true
	}
	return !isTerminal(run.Status.Phase)
}

// conflict is one other run that already holds targets this run wants.
type conflict struct {
	run   *burninv1alpha1.BurnInRun
	nodes []string
}

func (c *conflict) String() string {
	return fmt.Sprintf("%s/%s (holding %s)", c.run.Namespace, c.run.Name, strings.Join(c.nodes, ", "))
}

// findConflict looks for an already-admitted run whose targets overlap this
// one's, and returns the first such run in admission order.
//
// THE LIST IS UNCACHED, and that is the whole point rather than an optimisation.
// The reproduction is runs created seconds apart; an informer that has not yet
// observed the run created two seconds ago answers "nothing is running", both
// runs admit, and the guard has failed in exactly the scenario it was written
// for. This is the third read in this controller that goes to the apiserver, and
// it obeys the same rule as the other two: a read goes uncached when a stale
// answer costs the fleet something it cannot get back — here, two full-power
// soaks on one machine.
//
// Runs are listed across ALL namespaces because Nodes are cluster-scoped. Two
// runs in different namespaces contend for the same hardware just as thoroughly
// as two in one, and a per-namespace guard would be a guard with a hole in it
// that nothing in the cluster announces.
func (r *BurnInRunReconciler) findConflict(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	targets []string,
) (*conflict, error) {
	wanted := make(map[string]bool, len(targets))
	for _, n := range targets {
		wanted[n] = true
	}

	var runs burninv1alpha1.BurnInRunList
	if err := r.uncached().List(ctx, &runs); err != nil {
		return nil, fmt.Errorf("list runs for admission: %w", err)
	}

	var found *conflict
	for i := range runs.Items {
		other := &runs.Items[i]
		if other.UID == run.UID {
			continue
		}
		if !runHolds(other) || !admissionOrder(other, run) {
			continue
		}
		overlap := r.overlappingTargets(ctx, other, wanted)
		if len(overlap) == 0 {
			continue
		}
		// Report the run that would win against every other contender, not
		// merely the first one the list happened to yield: the message names a
		// run for a human to go and look at, and naming a different one on each
		// pass would be worse than naming none.
		if found == nil || admissionOrder(other, found.run) {
			found = &conflict{run: other, nodes: overlap}
		}
	}
	return found, nil
}

// overlappingTargets is which of `wanted` another run is targeting.
//
// The other run's PINNED plan is used whenever it has one, because that is the
// node list it will actually execute against and it cannot change afterwards. A
// run that has not pinned yet has its selector re-resolved; a selector that
// resolves to nothing yields no overlap, which is correct — a run that can never
// find a target cannot be holding one, and refusing a legitimate run on the
// strength of a broken one would be the guard causing the outage.
//
// That selector re-resolution reads NODES through the cache, unlike the run list
// above, and deliberately so: a run resolves its own targets from the same cache
// in start(), so using a different source here would let admission believe the
// other run is testing a node that run will never touch. Consistency between the
// two readings is what matters, and it is the same reading either way.
func (r *BurnInRunReconciler) overlappingTargets(
	ctx context.Context,
	other *burninv1alpha1.BurnInRun,
	wanted map[string]bool,
) []string {
	var targets []string
	if p, pinned, err := loadPlan(other); err == nil && pinned {
		targets = p.Targets
	} else {
		resolved, resolveErr := r.resolveTargets(ctx, other.Spec.Target)
		if resolveErr != nil {
			return nil
		}
		targets = resolved
	}

	seen := map[string]bool{}
	var overlap []string
	for _, n := range targets {
		if wanted[n] && !seen[n] {
			seen[n] = true
			overlap = append(overlap, n)
		}
	}
	sort.Strings(overlap)
	return overlap
}

// admit is the gate. It returns false when the run must not proceed, having
// already driven it to a terminal refusal.
//
// It runs ONCE, at start, before the plan is pinned and therefore before the
// run has a finalizer, a cordon or a pod. A refused run has touched nothing,
// which is what makes refusing safe to do this bluntly.
func (r *BurnInRunReconciler) admit(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	targets []string,
	sinks []string,
) (bool, ctrl.Result, error) {
	found, err := r.findConflict(ctx, run, targets)
	if err != nil {
		// An admission check that could not be performed is not an admission.
		// Retrying costs a reconcile; guessing costs a node.
		return false, ctrl.Result{}, err
	}
	if found == nil {
		r.setAdmitted(run, metav1.ConditionTrue, burninv1alpha1.ReasonRunAdmitted,
			"no other active run holds these targets")
		return true, ctrl.Result{}, nil
	}

	if forced(run) {
		// Say it loudly and record it. A verdict measured while another run was
		// driving the same node is not comparable to one measured alone, and six
		// months later the only thing that will still say so is this condition.
		log.FromContext(ctx).Info("spec.force admitted a run onto nodes another run is already testing",
			"conflictingRun", found.run.Namespace+"/"+found.run.Name, "nodes", found.nodes)
		r.setAdmitted(run, metav1.ConditionTrue, burninv1alpha1.ReasonRunForced,
			fmt.Sprintf("spec.force admitted this run while %s — results measured under "+
				"contention are not comparable to results measured alone", found))
		return true, ctrl.Result{}, nil
	}

	message := fmt.Sprintf(
		"refused: BurnInRun %s/%s is already testing %s. Two burn-ins on one node are two full-power "+
			"loads on one machine — the facility interlock (maxConcurrentNodes) bounds only ONE run's "+
			"fan-out, so neither run can see the other's load — and the measurements contend for the "+
			"very resources they are measuring. No hardware was tested here. Re-create this run once "+
			"%s finishes, or set spec.force to accept the contention deliberately",
		found.run.Namespace, found.run.Name, strings.Join(found.nodes, ", "), found.run.Name)

	r.setAdmitted(run, metav1.ConditionFalse, burninv1alpha1.ReasonTargetsBusy, message)
	res, err := r.refuse(ctx, run, sinks, message)
	return false, res, err
}

// refuse drives a run to a terminal, non-verdict Error with the reason recorded
// as a result.
//
// It mirrors finalizeError rather than reusing it because the two say different
// things: finalizeError means the run's own spec could not be resolved, this
// means the run is fine and the fleet is busy. The result is synthetic — no
// Kind, which is the marker (TestResult.IsSynthetic), no Scope, no Nodes.
func (r *BurnInRunReconciler) refuse(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	sinks []string,
	message string,
) (ctrl.Result, error) {
	log.FromContext(ctx).Info("refusing to admit run", "run", run.Name, "reason", message)
	now := metav1.NewTime(r.now())
	run.Status.Results = append(run.Status.Results, burninv1alpha1.TestResult{
		Name:       burninv1alpha1.SyntheticResultAdmission,
		Phase:      burninv1alpha1.RunError,
		FinishedAt: &now,
		Message:    message,
	})
	return r.terminate(ctx, run, sinks, burninv1alpha1.RunError)
}

// setAdmitted records the admission decision on the run.
func (r *BurnInRunReconciler) setAdmitted(run *burninv1alpha1.BurnInRun, status metav1.ConditionStatus, reason, message string) {
	apimeta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               burninv1alpha1.ConditionRunAdmitted,
		Status:             status,
		Reason:             reason,
		Message:            clampConditionMessage(message),
		ObservedGeneration: run.Generation,
		LastTransitionTime: metav1.NewTime(r.now()),
	})
}

// forced resolves spec.force.
func forced(run *burninv1alpha1.BurnInRun) bool {
	return run.Spec.Force != nil && *run.Spec.Force
}

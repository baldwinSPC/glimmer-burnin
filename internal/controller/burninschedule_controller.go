package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	// The manager image is gcr.io/distroless/static, which carries no
	// /usr/share/zoneinfo. Without the embedded database time.LoadLocation
	// fails for every IANA name, so a schedule naming spec.timeZone would be
	// rejected as invalid and silently never fire — the exact failure the
	// TimeZone field exists to prevent, since a maintenance window is agreed
	// in local time and the staff who can answer a power event are local too.
	// Embedding it costs a few hundred KB in the binary and removes a class of
	// works-on-my-cluster/fails-in-production divergence.
	_ "time/tzdata"

	"github.com/robfig/cron/v3"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// labelSchedule attributes a BurnInRun to the BurnInSchedule that created it.
// The owner reference is the authority — it is UID-bearing and drives both the
// Owns() watch and history GC — but a label is what an operator can select on
// (`kubectl get burninruns -l burnin.glimmer.ai/schedule=weekly-soak`), and
// reading a schedule's history back is the whole point of keeping it.
const labelSchedule = "burnin.glimmer.ai/schedule"

// annotationScheduledAt records the cron tick a run was created for, in RFC3339.
//
// It is NOT the same as the run's creationTimestamp and the difference is the
// useful part: it says which nominal window the run belongs to, so a run that
// started eleven minutes late (slow watch, controller restart, busy apiserver)
// is still attributable to the 03:00 Sunday tick rather than looking like an
// unscheduled ad-hoc run.
const annotationScheduledAt = "burnin.glimmer.ai/scheduled-at"

// conditionScheduleValid is False when spec.schedule or spec.timeZone does not
// parse. A schedule that cannot be parsed creates nothing, so without a
// surfaced condition it is indistinguishable from one that is working and
// simply has not reached its next tick — which for a weekly soak means a week
// of believing the fleet is being re-verified when it is not.
const conditionScheduleValid = "ScheduleValid"

const (
	// Defaults matching the kubebuilder markers on BurnInScheduleSpec. They
	// are applied here as well because the reconciler must not depend on
	// apiserver defaulting having happened: an object that predates the CRD
	// defaults, or one built in a test, still has to be handled with the same
	// retention policy the field documentation promises.
	defaultSuccessfulRunsHistoryLimit int32 = 3
	defaultFailedRunsHistoryLimit     int32 = 10
)

// scheduleMaxTickScan bounds how many elapsed ticks a single reconcile will
// enumerate. A schedule whose cursor is far in the past ("@every 1m", cluster
// down for a week) would otherwise walk hundreds of thousands of ticks on
// every pass. Beyond this the schedule is resynchronised to now instead: it is
// too far behind for "the most recent missed tick" to mean anything, and a
// schedule that ticks that often has its next real tick moments away.
const scheduleMaxTickScan = 1000

// scheduleMinRequeue floors the requeue delay. controller-runtime treats a
// non-positive RequeueAfter as "do not requeue", so a tick less than a
// scheduler tick away must not round down to zero and strand the schedule.
const scheduleMinRequeue = time.Second

// BurnInScheduleReconciler turns a BurnInSchedule into BurnInRuns on a cron
// schedule.
//
// It deliberately does NOT catch up. A missed tick is recorded and dropped,
// never replayed: every run cordons its targets and drives them to their power
// and thermal limits, so replaying a weekend of missed ticks would launch a
// full-power sweep at whatever hour the controller happened to recover, far
// outside the window the schedule was agreed for. At most one run is created
// per reconcile, whatever the backlog.
type BurnInScheduleReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// Now is the clock, swappable in tests. Nil means time.Now.
	Now func() time.Time
}

func (r *BurnInScheduleReconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

// The schedule itself is only ever read and status-updated: this controller
// must never create or delete a BurnInSchedule. Runs are created and deleted
// (history GC), which is stated here rather than inherited from the run
// reconciler's markers so the grant is visible where it is exercised.
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninschedules,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninschedules/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninschedules/finalizers,verbs=update
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns,verbs=get;list;watch;create;delete

// Reconcile is level-based: every pass re-derives the schedule's standing from
// the live BurnInRuns it owns rather than from its own status, so a stale or
// partially written status can never let a second run start beside an active
// one.
func (r *BurnInScheduleReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var sched burninv1alpha1.BurnInSchedule
	if err := r.Get(ctx, req.NamespacedName, &sched); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Snapshot the status so the pass writes only when something actually
	// changed. Owned runs update their own status many times while executing,
	// and every one of those wakes this reconciler; without the guard each
	// would provoke a schedule status write, which wakes it again.
	before := *sched.Status.DeepCopy()

	active, finished, err := r.ownedRuns(ctx, &sched)
	if err != nil {
		return ctrl.Result{}, err
	}

	r.observe(&sched, active, finished)

	if err := r.collectHistory(ctx, &sched, finished); err != nil {
		return ctrl.Result{}, err
	}

	cronSchedule, err := parseSchedule(&sched)
	if err != nil {
		// A spec that does not parse is not a transient failure and retrying
		// it on a timer would only burn reconciles. Surface it and wait for
		// the edit, which arrives as its own watch event.
		apimeta.SetStatusCondition(&sched.Status.Conditions, metav1.Condition{
			Type:               conditionScheduleValid,
			Status:             metav1.ConditionFalse,
			Reason:             "InvalidSchedule",
			Message:            err.Error(),
			ObservedGeneration: sched.Generation,
			LastTransitionTime: metav1.NewTime(r.now()),
		})
		log.FromContext(ctx).Error(err, "BurnInSchedule does not parse; no runs will be created")
		return ctrl.Result{}, r.writeStatus(ctx, &sched, &before)
	}
	apimeta.SetStatusCondition(&sched.Status.Conditions, metav1.Condition{
		Type:               conditionScheduleValid,
		Status:             metav1.ConditionTrue,
		Reason:             "Parsed",
		Message:            "schedule parses and is being evaluated",
		ObservedGeneration: sched.Generation,
		LastTransitionTime: metav1.NewTime(r.now()),
	})

	now := r.now()

	// Suspend stops NEW runs only. Status above still tracked the in-flight
	// ones, and history was still collected: suspending is a pause on
	// creation, not an instruction to stop observing or to retain differently.
	if boolOrDefault(sched.Spec.Suspend, false) {
		// No timed requeue. There is nothing to wake up for until either the
		// suspend flag is cleared or an owned run finishes, and both arrive as
		// watch events.
		return ctrl.Result{}, r.writeStatus(ctx, &sched, &before)
	}

	tick, missed, resync := dueTick(cronSchedule, scheduleWindowStart(&sched, now), now, sched.Spec.StartingDeadlineSeconds)
	sched.Status.MissedSchedules = addMissed(sched.Status.MissedSchedules, missed)

	switch {
	case resync:
		// So far behind that enumeration was abandoned. Move the cursor to now
		// so the next pass evaluates a sane window, and fire nothing.
		sched.Status.LastScheduleTime = &metav1.Time{Time: now}

	case tick.IsZero():
		// Nothing due.

	case concurrencyPolicy(&sched) == burninv1alpha1.ConcurrencyForbid && len(active) > 0:
		// THE NODE-BUSY GUARD. Two runs from one schedule would contend for
		// the very accelerators, memory bandwidth and fabric each is trying to
		// measure, so both report degraded numbers and a healthy node earns a
		// failing verdict; worse, they disagree about the node's cordon and
		// one can uncordon a node the other is still driving at full power.
		//
		// The cursor still advances. Leaving it behind would keep this tick
		// due forever, so the moment the active run finished the schedule
		// would fire a run for a window that has already passed.
		sched.Status.LastScheduleTime = &metav1.Time{Time: tick}
		sched.Status.MissedSchedules = addMissed(sched.Status.MissedSchedules, 1)
		log.FromContext(ctx).Info("skipping tick: a run from this schedule is still active",
			"tick", tick, "activeRuns", len(active), "policy", burninv1alpha1.ConcurrencyForbid)

	default:
		if err := r.fire(ctx, &sched, &before, tick); err != nil {
			return ctrl.Result{}, err
		}
	}

	if err := r.writeStatus(ctx, &sched, &before); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: untilNext(cronSchedule, r.now())}, nil
}

// fire creates the run for a due tick.
//
// ORDER IS LOAD-BEARING: the cursor is persisted BEFORE the run is created.
// The two ways this can be interrupted are not equally bad. Persist-then-create
// loses a tick if the process dies in between — the next tick is soon enough.
// Create-then-persist duplicates the tick, which means a second full-power
// sweep launched across the fleet, against nodes a live run already holds
// cordoned. One is a gap in the record; the other is a facility event.
func (r *BurnInScheduleReconciler) fire(
	ctx context.Context,
	sched *burninv1alpha1.BurnInSchedule,
	before *burninv1alpha1.BurnInScheduleStatus,
	tick time.Time,
) error {
	sched.Status.LastScheduleTime = &metav1.Time{Time: tick}
	if err := r.writeStatus(ctx, sched, before); err != nil {
		return err
	}

	run, err := r.runForTick(sched, tick)
	if err != nil {
		return err
	}
	if err := r.Create(ctx, run); err != nil {
		// The cursor has already moved, so this tick is spent rather than
		// retried. That is the deliberate trade above; make it loud instead of
		// silent, and let the error requeue us so the NEXT tick is not also
		// lost to whatever is wrong.
		log.FromContext(ctx).Error(err, "tick was consumed but its BurnInRun could not be created", "tick", tick)
		return err
	}

	sched.Status.LastRunName = run.Name
	sched.Status.LastRunPhase = runPhaseOrPending(run.Status.Phase)
	sched.Status.Active = append(sched.Status.Active, runReference(run))
	return nil
}

// runPhaseOrPending reports a not-yet-reconciled run as Pending.
//
// A run is created with no phase at all and stays that way until its own
// reconciler first touches it. Reporting that gap verbatim would blank out the
// schedule's Last Phase column, and — because this same value is recomputed on
// every pass — would make the status flap between "" and Pending, so each
// write would wake the watch and provoke the next write.
func runPhaseOrPending(p burninv1alpha1.RunPhase) burninv1alpha1.RunPhase {
	if p == "" {
		return burninv1alpha1.RunPending
	}
	return p
}

// runForTick stamps the template onto a new BurnInRun.
func (r *BurnInScheduleReconciler) runForTick(sched *burninv1alpha1.BurnInSchedule, tick time.Time) (*burninv1alpha1.BurnInRun, error) {
	tpl := sched.Spec.RunTemplate

	// MaxConcurrentNodes and RetryOnErrorLimit are written explicitly rather
	// than left nil for the BurnInRun CRD to default. The facility interlock
	// and the no-retry-on-failure posture should be readable on the run object
	// itself, because that object is the audit record a verdict is read
	// against months later.
	maxConcurrent := int32OrDefault(tpl.MaxConcurrentNodes, 1)
	retryOnError := int32OrDefault(tpl.RetryOnErrorLimit, 0)

	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: sched.Namespace,
			// generateName, not a deterministic name: a run is an immutable
			// piece of evidence about one tick, and reusing a name would
			// invite a later tick to overwrite an earlier tick's verdict.
			GenerateName: truncate(sched.Name, 200) + "-",
			Labels:       map[string]string{labelSchedule: sched.Name},
			Annotations:  map[string]string{annotationScheduledAt: tick.UTC().Format(time.RFC3339)},
		},
		Spec: burninv1alpha1.BurnInRunSpec{
			ProfileRef: sched.Spec.ProfileRef,
			// Deep-copied: the schedule came out of the shared informer cache,
			// and handing its slices and maps to another object would let a
			// later mutation reach back into everyone else's view of it.
			Target:                  *sched.Spec.Target.DeepCopy(),
			MaxConcurrentNodes:      &maxConcurrent,
			RetryOnErrorLimit:       &retryOnError,
			DeadlineSeconds:         copyInt32(tpl.DeadlineSeconds),
			TTLSecondsAfterFinished: copyInt32(tpl.TTLSecondsAfterFinished),
		},
	}
	// The owner reference is what makes the Owns() watch, history GC and
	// cascading delete work, and it is UID-bearing, so a deleted-and-recreated
	// schedule of the same name does not inherit the previous one's runs.
	if err := controllerutil.SetControllerReference(sched, run, r.Scheme); err != nil {
		return nil, fmt.Errorf("set owner reference on run for tick %s: %w", tick.UTC().Format(time.RFC3339), err)
	}
	return run, nil
}

// ownedRuns splits the runs this schedule controls into active and finished.
//
// Ownership is matched by controller UID, never by name or label: a label is
// writable by anyone and a name is reusable, and either would let this
// schedule's Forbid guard be fooled — or let history GC delete a run belonging
// to something else.
func (r *BurnInScheduleReconciler) ownedRuns(
	ctx context.Context,
	sched *burninv1alpha1.BurnInSchedule,
) (active, finished []burninv1alpha1.BurnInRun, err error) {
	var list burninv1alpha1.BurnInRunList
	if err := r.List(ctx, &list, client.InNamespace(sched.Namespace)); err != nil {
		return nil, nil, fmt.Errorf("list BurnInRuns in %s: %w", sched.Namespace, err)
	}
	for i := range list.Items {
		run := list.Items[i]
		if !metav1.IsControlledBy(&run, sched) {
			continue
		}
		// A run whose phase is still empty has been created but not yet
		// reconciled. It counts as ACTIVE. Treating it as anything else would
		// break the Forbid guard in its most likely window — the moments right
		// after creation, when the next reconcile is already queued.
		if run.Status.Phase.IsTerminal() {
			finished = append(finished, run)
		} else {
			active = append(active, run)
		}
	}
	sortRunsNewestFirst(active)
	sortRunsNewestFirst(finished)
	return active, finished, nil
}

// observe refreshes the purely descriptive parts of status from live runs.
func (r *BurnInScheduleReconciler) observe(sched *burninv1alpha1.BurnInSchedule, active, finished []burninv1alpha1.BurnInRun) {
	sched.Status.ObservedGeneration = sched.Generation

	refs := make([]corev1.ObjectReference, 0, len(active))
	for i := range active {
		refs = append(refs, runReference(&active[i]))
	}
	sched.Status.Active = refs

	// Newest of everything the schedule owns, active or not.
	all := append(append([]burninv1alpha1.BurnInRun{}, active...), finished...)
	sortRunsNewestFirst(all)
	if len(all) > 0 {
		sched.Status.LastRunName = all[0].Name
		// Passed, Failed, Error and Cancelled are reported as themselves. They
		// are four different situations — a hardware verdict, no verdict at
		// all, and a human decision — and summarising them as "not passing"
		// would hide which response is called for.
		sched.Status.LastRunPhase = runPhaseOrPending(all[0].Status.Phase)
	}

	// Only ever advanced, never recomputed downwards: history GC will
	// eventually delete the passing run this was derived from, and the fact
	// that the schedule once succeeded must not be erased by retention.
	for i := range finished {
		if finished[i].Status.Phase != burninv1alpha1.RunPassed {
			continue
		}
		at := finished[i].Status.FinishedAt
		if at == nil {
			at = &metav1.Time{Time: finished[i].CreationTimestamp.Time}
		}
		if sched.Status.LastSuccessfulTime == nil || sched.Status.LastSuccessfulTime.Time.Before(at.Time) {
			sched.Status.LastSuccessfulTime = at.DeepCopy()
		}
	}
}

// collectHistory enforces the two retention limits.
//
// Only TERMINAL runs are ever considered. Deleting an active run would kill an
// in-flight burn-in and, worse, tear down the object that owes its targets an
// uncordon.
func (r *BurnInScheduleReconciler) collectHistory(
	ctx context.Context,
	sched *burninv1alpha1.BurnInSchedule,
	finished []burninv1alpha1.BurnInRun,
) error {
	var passed, notPassed []burninv1alpha1.BurnInRun
	for i := range finished {
		if finished[i].Status.Phase == burninv1alpha1.RunPassed {
			passed = append(passed, finished[i])
		} else {
			// Failed, Error and Cancelled share one budget, per the field docs.
			notPassed = append(notPassed, finished[i])
		}
	}

	// Counted separately and with a higher failure budget on purpose: the
	// passing runs are the boring ones, and diagnosing degradation means
	// comparing a failure against the failures before it.
	if err := r.deleteBeyond(ctx, passed, int32OrDefault(sched.Spec.SuccessfulRunsHistoryLimit, defaultSuccessfulRunsHistoryLimit)); err != nil {
		return err
	}
	return r.deleteBeyond(ctx, notPassed, int32OrDefault(sched.Spec.FailedRunsHistoryLimit, defaultFailedRunsHistoryLimit))
}

// deleteBeyond removes all but the newest limit runs. runs must be sorted
// newest-first.
func (r *BurnInScheduleReconciler) deleteBeyond(ctx context.Context, runs []burninv1alpha1.BurnInRun, limit int32) error {
	sortRunsNewestFirst(runs)
	for i := int(limit); i < len(runs); i++ {
		victim := runs[i]
		if err := r.Delete(ctx, &victim); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("garbage-collect BurnInRun %s/%s: %w", victim.Namespace, victim.Name, err)
		}
	}
	return nil
}

// writeStatus persists status only when it differs from the snapshot, then
// rebases the snapshot so a second call in the same pass compares correctly.
func (r *BurnInScheduleReconciler) writeStatus(
	ctx context.Context,
	sched *burninv1alpha1.BurnInSchedule,
	before *burninv1alpha1.BurnInScheduleStatus,
) error {
	if equality.Semantic.DeepEqual(*before, sched.Status) {
		return nil
	}
	if err := r.Status().Update(ctx, sched); err != nil {
		return err
	}
	*before = *sched.Status.DeepCopy()
	return nil
}

// parseSchedule resolves spec.schedule and spec.timeZone into a cron schedule.
func parseSchedule(sched *burninv1alpha1.BurnInSchedule) (cron.Schedule, error) {
	spec := strings.TrimSpace(sched.Spec.Schedule)
	if spec == "" {
		return nil, fmt.Errorf("spec.schedule is empty")
	}

	// Reject an inline timezone prefix rather than passing it through.
	//
	// Two reasons, and the second is the serious one. First, two places to say
	// the same thing is one place too many, and spec.timeZone is the declared
	// field. Second, the upstream parser PANICS on a prefix with no following
	// space ("TZ=x"): it slices on strings.Index(spec, " ") without checking
	// for -1. spec.schedule is attacker-reachable — anyone who can create a
	// BurnInSchedule could take down the manager process, and with it every
	// other reconciler in it, including the runs that owe nodes an uncordon.
	if strings.HasPrefix(spec, "TZ=") || strings.HasPrefix(spec, "CRON_TZ=") {
		return nil, fmt.Errorf("spec.schedule must not carry a TZ=/CRON_TZ= prefix; use spec.timeZone instead")
	}

	// UTC, not time.Local. The parser's own default is the process's local
	// zone, which would make a schedule's meaning depend on the manager pod's
	// environment — the same object firing at different wall-clock times on
	// two clusters.
	loc := time.UTC
	if tz := sched.Spec.TimeZone; tz != nil && *tz != "" {
		l, err := time.LoadLocation(*tz)
		if err != nil {
			return nil, fmt.Errorf("invalid spec.timeZone %q: %w", *tz, err)
		}
		loc = l
	}

	cronSchedule, err := cron.ParseStandard("CRON_TZ=" + loc.String() + " " + spec)
	if err != nil {
		return nil, fmt.Errorf("invalid spec.schedule %q: %w", spec, err)
	}
	return cronSchedule, nil
}

// scheduleWindowStart is the cursor a pass evaluates forward from: the last
// tick acted on, or the schedule's creation for one that has never fired.
func scheduleWindowStart(sched *burninv1alpha1.BurnInSchedule, now time.Time) time.Time {
	if sched.Status.LastScheduleTime != nil {
		return sched.Status.LastScheduleTime.Time
	}
	if !sched.CreationTimestamp.IsZero() {
		return sched.CreationTimestamp.Time
	}
	return now
}

// dueTick decides which single tick, if any, to run now.
//
// It returns the MOST RECENT due tick and counts the rest as missed. It never
// returns a list, because the caller must never be able to launch a burst:
// catching up on burn-in means putting a fleet under full electrical and
// thermal load at an unplanned hour, and the value of a tick is mostly in
// running at the agreed time.
//
// resync reports that the backlog exceeded scheduleMaxTickScan, so the caller
// should move the cursor to now and fire nothing.
func dueTick(cronSchedule cron.Schedule, from, now time.Time, startingDeadlineSeconds *int32) (tick time.Time, missed int, resync bool) {
	// Ticks older than the starting deadline are not eligible to run at all.
	var deadline time.Time
	if startingDeadlineSeconds != nil {
		deadline = now.Add(-time.Duration(*startingDeadlineSeconds) * time.Second)
	}

	var latest time.Time
	scanned := 0
	for t := cronSchedule.Next(from); !t.IsZero() && !t.After(now); t = cronSchedule.Next(t) {
		scanned++
		if scanned > scheduleMaxTickScan {
			return time.Time{}, scanned, true
		}
		if !deadline.IsZero() && t.Before(deadline) {
			// Too late to honour. Counted, not run: firing it now would put a
			// full-power sweep outside its agreed window.
			missed++
			continue
		}
		if !latest.IsZero() {
			// Superseded by a newer eligible tick.
			missed++
		}
		latest = t
	}
	return latest, missed, false
}

// untilNext is the delay to the next tick, so the controller sleeps until it
// is actually needed instead of polling.
func untilNext(cronSchedule cron.Schedule, now time.Time) time.Duration {
	next := cronSchedule.Next(now)
	if next.IsZero() {
		// A spec with no future occurrence at all (e.g. "0 0 30 2 *"). Nothing
		// to wait for; a spec edit re-triggers via the watch.
		return 0
	}
	if d := next.Sub(now); d > scheduleMinRequeue {
		return d
	}
	return scheduleMinRequeue
}

// concurrencyPolicy resolves the policy, defaulting to the node-busy guard.
// An unset or unrecognised value must never fall through to Allow.
func concurrencyPolicy(sched *burninv1alpha1.BurnInSchedule) burninv1alpha1.ScheduleConcurrencyPolicy {
	if sched.Spec.ConcurrencyPolicy == burninv1alpha1.ConcurrencyAllow {
		return burninv1alpha1.ConcurrencyAllow
	}
	return burninv1alpha1.ConcurrencyForbid
}

// runOrderKey is a run's position in the schedule's history.
//
// The scheduled-at tick is preferred over creationTimestamp because it is the
// run's LOGICAL position and is exact. creationTimestamp is serialised at
// one-second granularity and is stamped by the apiserver rather than by this
// controller, so it ties easily and can disagree with our clock — and a tie
// decides which evidence history GC destroys.
func runOrderKey(run *burninv1alpha1.BurnInRun) time.Time {
	if v := run.Annotations[annotationScheduledAt]; v != "" {
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			return t
		}
	}
	return run.CreationTimestamp.Time
}

// sortRunsNewestFirst orders runs newest first, breaking ties by name so that
// GC is deterministic even for runs that share an ordering key.
func sortRunsNewestFirst(runs []burninv1alpha1.BurnInRun) {
	sort.SliceStable(runs, func(i, j int) bool {
		a, b := runOrderKey(&runs[i]), runOrderKey(&runs[j])
		if a.Equal(b) {
			return runs[i].Name > runs[j].Name
		}
		return a.After(b)
	})
}

func runReference(run *burninv1alpha1.BurnInRun) corev1.ObjectReference {
	return corev1.ObjectReference{
		APIVersion: burninv1alpha1.GroupVersion.String(),
		Kind:       "BurnInRun",
		Namespace:  run.Namespace,
		Name:       run.Name,
		UID:        run.UID,
	}
}

// addMissed increments the missed counter without wrapping past int32.
func addMissed(current int32, delta int) int32 {
	if delta <= 0 {
		return current
	}
	sum := int64(current) + int64(delta)
	if sum > int64(int32(1<<31-1)) {
		return int32(1<<31 - 1)
	}
	return int32(sum)
}

func int32OrDefault(p *int32, def int32) int32 {
	if p == nil {
		return def
	}
	return *p
}

func boolOrDefault(p *bool, def bool) bool {
	if p == nil {
		return def
	}
	return *p
}

func copyInt32(p *int32) *int32 {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

// SetupWithManager wires the reconciler to BurnInSchedule events and to the
// runs it owns, so a run reaching a terminal phase immediately re-evaluates the
// Forbid guard and the history limits instead of waiting for the next tick.
func (r *BurnInScheduleReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&burninv1alpha1.BurnInSchedule{}).
		Owns(&burninv1alpha1.BurnInRun{}).
		Named("burninschedule").
		Complete(r)
}

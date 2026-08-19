// Package controller holds the reconcilers for the Glimmer Burn-In Operator.
package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/metrics"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// BurnInRunReconciler drives a BurnInRun from Pending to a terminal verdict:
// resolve the profile ONCE into a pinned plan, capture the target fingerprint,
// hold the target nodes out of the scheduler, execute each test's pod on each
// target node under the run's concurrency interlock, evaluate thresholds
// against parsed metrics, and export the verdict to the plan's sinks.
//
// v1 executes Node-scope tests. Pair/Group tests are recorded as Error rather
// than silently skipped: a required acceptance test the operator cannot run
// must never let hardware pass by omission.
//
// The reconciler is vendor-neutral by construction. There is no accelerator,
// no vendor and no product name in any branch here, and there must not be one:
// whether a test applies to a given part is the RUNNER's answer, delivered as
// exit code 2, because the runner is the only thing in the system that has
// actually looked at the hardware. A controller-side "this kind does not apply
// to that GPU" is a claim made from a label, it goes stale the moment a driver
// or a firmware revision changes the answer, and it produces a Skipped verdict
// for hardware nobody tested.
type BurnInRunReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// APIReader reads straight from the apiserver, bypassing the informer
	// cache. The manager wires it from mgr.GetAPIReader(); nil falls back to
	// the cached client, which is what a test that does not care gets.
	//
	// THE RULE, stated once here for anyone adding a read: a read goes uncached
	// ONLY when a stale answer would cost the fleet something a later pass
	// cannot get back — a node permanently out of the scheduler, or hardware
	// driven past what it was budgeted for.
	//
	// The test that decides it: does the answer get RECORDED somewhere else, or
	// used to decide NOT TO ACT? Neither has an optimistic-concurrency check
	// standing behind it. A read whose only consequence is an Update of the SAME
	// object does — the apiserver refuses a stale write on resourceVersion and
	// the next pass reads again — and stays cached.
	//
	// The five that qualify, all of them the cordon interlock:
	//
	//	capturePriorSchedulability   recorded into the run's status
	//	cordonNode's node read       two arms decline to cordon and write nothing
	//	releaseNode's node read      two arms drop the node from status, writing nothing
	//	releaseCordons' scan         finds nothing, concludes nothing is owed
	//	the admission scan           decides this run may take hardware another holds
	//
	// Everything else — pods, the run itself, profiles, tests — stays cached,
	// because a stale answer there costs a wasted pass and is corrected by the
	// next one.
	//
	// The failure this exists to prevent (issue #84): a run that cordons a node
	// and releases it seconds later leaves the informer holding the cordoned
	// version. The NEXT run's capture then reads Unschedulable=true, records it
	// as pre-existing, and honours that record at teardown — leaving a node
	// permanently out of the scheduler that nothing in the cluster knows is
	// held. It is intermittent by construction: it needs a recent cordon on the
	// same node, which is exactly what a back-to-back schedule produces.
	//
	// The admission read (issue #81) has the same shape and a heavier cost: the
	// runs that contend are created seconds apart, so an informer that has not
	// yet observed the first one lets the second admit itself onto hardware the
	// first is already saturating.
	APIReader client.Reader

	// PodLogs fetches a pod's stdout — the runner's metrics channel. It is
	// called both when a pod terminates and, for checkpointing, while it is
	// still running; the parser's last-occurrence-wins rule is what makes a
	// partial log a valid snapshot rather than a corrupt one.
	//
	// controller-runtime's client cannot read subresource logs, so the manager
	// wires this from a plain clientset; tests stub it.
	PodLogs func(ctx context.Context, namespace, name string) (string, error)

	// Now is the clock, swappable in tests. Nil means time.Now.
	Now func() time.Time

	// FingerprintNamespace is where NodeFingerprints live, so a run can read the
	// accelerator vendor of a node it is about to test and pick that vendor's
	// runner image.
	//
	// The same value the NodeFingerprint controller writes them to, resolved
	// from the downward API in main rather than guessed here. Empty means the
	// vendor is never read: every test then resolves its image from an explicit
	// pin or the kind's default, which is exactly the behaviour that existed
	// before imagesByVendor and is correct for a single-vendor fleet.
	FingerprintNamespace string

	// Cluster identifies the cluster in every delivered envelope. Nil emits no
	// cluster fields at all, which is a valid envelope: an operator that has not
	// been told its cluster's name must not guess one, because a verdict
	// attributed to the wrong fleet is worse than one attributed to none.
	Cluster *contract.ClusterRef

	// ForgetMetrics drops a run's exported Prometheus series. Nil uses the
	// package-level exporter. The series are a view of a live object, so they
	// are dropped when the object goes away and NOT when it merely finishes —
	// a finished run's verdict is exactly what a dashboard is there to show.
	ForgetMetrics func(namespace, name string)
}

func (r *BurnInRunReconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

// uncached is the reader for the two node reads that must not be served from a
// cache. It degrades to the cached client when no APIReader is wired, so a
// reconciler built by hand still works — but the manager always wires one.
func (r *BurnInRunReconciler) uncached() client.Reader {
	if r.APIReader != nil {
		return r.APIReader
	}
	return r.Client
}

func (r *BurnInRunReconciler) forgetMetrics(namespace, name string) {
	if r.ForgetMetrics != nil {
		r.ForgetMetrics(namespace, name)
		return
	}
	metrics.Default.Forget(namespace, name)
}

// resolvedTest is one entry of a profile with its spec materialised.
type resolvedTest struct {
	name     string
	spec     burninv1alpha1.BurnInTestSpec
	required bool
	// axes are the variant labels this execution came from, or nil for a test
	// with no variants. They are carried, echoed and never interpreted.
	axes map[string]string
	// parent is the profile entry this cell was expanded from, or empty when the
	// entry was not expanded at all.
	//
	// Recorded rather than recovered from the name. Splitting "<test>-<variant>"
	// on the last hyphen works until a test is called "gemm-sweep" or a variant
	// "fp8-dense", and then it is wrong silently — which is the same reason
	// TestResult.VariantAxes exists instead of asking a consumer to parse names.
	parent string
}

// resolveGracePeriod is how long a young run tolerates NotFound on its
// profile/tests before concluding they are genuinely missing. `kubectl apply
// -f dir/` creates the run and its profile together, and the run's watch
// event can arrive before the profile lands in the informer cache — a valid
// run must not be terminally errored by that race.
const resolveGracePeriod = 2 * time.Minute

// schedulingGracePeriod is how long a pod may sit unstarted before the run
// gives up on it. activeDeadlineSeconds cannot cover this: the kubelet
// enforces it relative to the pod's StartTime, which an unschedulable pod
// never gets — without this guard a typo'd node name wedges the run in
// Running forever.
const schedulingGracePeriod = 5 * time.Minute

// waitingPollInterval is the liveness backstop while pods are in flight.
// Pod events drive reconciles in the common case; this catches the pod that
// never emits one (stuck Pending on an unschedulable selector).
const waitingPollInterval = 30 * time.Second

// settledPollInterval is the backstop once every in-flight pod has started and
// no checkpoint is due sooner.
//
// It is the difference between a seven-day soak reconciling 20,000 times and
// reconciling 2,000. Nothing is lost by it: pod events still drive every state
// change, the run deadline still clamps the wake, and a checkpoint that is due
// sooner still wins. What it removes is the poll that exists only to re-ask a
// question nothing has changed the answer to.
const settledPollInterval = 5 * time.Minute

// terminalDeliveryRetryInterval paces re-sends of a terminal envelope that a
// sink refused. The terminal transition has no later transition to piggyback
// on, so it gets its own retry loop.
const terminalDeliveryRetryInterval = 5 * time.Minute

// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns/finalizers,verbs=update
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninprofiles;burnintests;burninsinks;nodefingerprints,verbs=get;list;watch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninsinks/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=nodefingerprints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=pods/log,verbs=get
// Pair-scope tests rendezvous through a headless Service, which is the only
// thing that lets the client name the server. See pair.go.
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;delete
// Nodes are updated, not just read: a run holds its targets out of the
// scheduler for its duration and restores them afterwards. See cordon.go.
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch;update;patch
// The kube-system namespace UID is the closest thing Kubernetes has to a stable
// cluster identity, and it rides in every envelope so a stored verdict is
// portable evidence. Read once at manager start; an RBAC that does not grant it
// is a legitimate deployment and the envelope simply carries less.
// +kubebuilder:rbac:groups="",resources=namespaces,verbs=get
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get

// Reconcile is the entrypoint: level-based, so every pass re-derives where the
// run stands from its pinned plan, its status, its pods and the cluster's
// nodes. Nothing is remembered in process memory, which is what makes the run
// survivable across a manager restart at any point in its lifecycle.
func (r *BurnInRunReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var run burninv1alpha1.BurnInRun
	if err := r.Get(ctx, req.NamespacedName, &run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Deletion outranks every other consideration. The run is going away; the
	// only thing that still matters is that the nodes it took out of the
	// scheduler are given back, which is what the finalizer is holding
	// deletion open for.
	if !run.DeletionTimestamp.IsZero() {
		return r.reconcileDeleted(ctx, &run)
	}

	if isTerminal(run.Status.Phase) {
		return r.reconcileTerminal(ctx, &run)
	}

	p, pinned, err := loadPlan(&run)
	if err != nil {
		// A pinned plan that no longer decodes is unrecoverable corruption of
		// our own annotation; the honest verdict is Error.
		return r.finalizeError(ctx, &run, nil, err)
	}
	if !pinned {
		return r.start(ctx, &run)
	}
	if run.Status.Phase == "" || run.Status.Phase == burninv1alpha1.RunPending {
		// Crash landed between pinning the plan and writing Running status:
		// finish the transition.
		return r.markRunning(ctx, &run, p)
	}
	return r.step(ctx, &run, p)
}

// ─── Start: resolve once, pin the plan ────────────────────────────────────────

// start resolves the profile and pins the plan. Resolution errors are
// classified: transient ones are retried by the controller's backoff, young
// NotFounds wait out the apply race, and only confirmed-permanent
// misconfiguration finalizes the run.
func (r *BurnInRunReconciler) start(ctx context.Context, run *burninv1alpha1.BurnInRun) (ctrl.Result, error) {
	profile, tests, resolveErr := r.resolveProfile(ctx, run)
	var targets []string
	if resolveErr == nil {
		targets, resolveErr = r.resolveTargets(ctx, run.Spec.Target)
	}
	if resolveErr != nil {
		if !isConfigError(resolveErr) {
			// Transient (API throttling, cache warming): let controller-runtime
			// retry with backoff rather than latching a permanent Error.
			return ctrl.Result{}, resolveErr
		}
		if r.now().Sub(run.CreationTimestamp.Time) < resolveGracePeriod {
			return ctrl.Result{RequeueAfter: 15 * time.Second}, nil
		}
		return r.finalizeError(ctx, run, profileSinks(profile), resolveErr)
	}

	// ADMISSION, and it is here for a reason: after the targets are known and
	// before the plan is pinned, which is the last moment at which the run still
	// holds nothing — no finalizer, no cordon, no pod. A refusal from this point
	// costs the fleet nothing to undo, and one taken any later would not.
	//
	// It answers the question maxConcurrentNodes cannot: not "how many nodes is
	// THIS run driving" but "is anybody else already driving these". See
	// admission.go.
	admitted, res, err := r.admit(ctx, run, targets, profileSinks(profile))
	if err != nil || !admitted {
		return res, err
	}

	p, err := buildPlan(profile, tests, targets, maxConcurrentNodes(run), isBaseline(run))
	if err != nil {
		return r.finalizeError(ctx, run, profileSinks(profile), err)
	}
	if err := pinPlan(run, p); err != nil {
		return r.finalizeError(ctx, run, p.Sinks, err)
	}
	// The finalizer goes on in the SAME write as the plan, which is the last
	// write before anything can cordon a node. It must never be possible to
	// hold a node without holding the finalizer that guarantees its release.
	controllerutil.AddFinalizer(run, burninv1alpha1.FinalizerCordonCleanup)
	// updateKeepingStatus, not a bare Update: the admission decision above lives
	// in run.Status.Conditions and has not been persisted yet, and Update
	// refreshes the object in place from the server's response — which still
	// holds the empty pre-start status. A bare write here would silently discard
	// the record of why this run was allowed to take its targets.
	if err := r.updateKeepingStatus(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	return r.markRunning(ctx, run, p)
}

func (r *BurnInRunReconciler) markRunning(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	// Recovery path: a crash between pinning the plan and here can land on a
	// run that never got the finalizer written.
	if controllerutil.AddFinalizer(run, burninv1alpha1.FinalizerCordonCleanup) {
		if err := r.updateKeepingStatus(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}
	// Nothing is cordoned here. Nodes are held one wave at a time, immediately
	// before load lands on them, so that a run's footprint on the fleet tracks
	// MaxConcurrentNodes instead of the length of its target list. See
	// cordonNode.
	//
	// But how the targets were FOUND is recorded here, and only here, because
	// this is the last moment at which the answer is uncontaminated: after this
	// the run starts cordoning, and a reading taken later cannot tell the run's
	// own hold from a hold that was there first. Written once and never
	// overwritten, so re-entering this path after a failed status write — or a
	// manager restart — cannot change what the run believes it must restore.
	if run.Status.PriorUnschedulable == nil {
		run.Status.PriorUnschedulable = r.capturePriorSchedulability(ctx, p.Targets)
	}

	run.Status.Phase = burninv1alpha1.RunRunning
	started := metav1.NewTime(r.now())
	run.Status.StartedAt = &started
	run.Status.Fingerprint = r.captureFingerprint(ctx, p.Targets)
	run.Status.ObservedGeneration = run.Generation
	r.applyThresholdsSoundCondition(ctx, run, p)
	r.applyPlanExpandedCondition(ctx, run, p)

	// Deliver before writing: a crash in between redelivers with the same
	// derived DeliveryID, which receivers dedupe. The reverse order would
	// drop the transition entirely.
	r.deliverPhase(ctx, run, p.Sinks, burninv1alpha1.RunRunning)
	if err := r.Status().Update(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
}

// applyThresholdsSoundCondition records what the threshold linter made of the
// gates this run is about to apply.
//
// This is the surface `pkg/verdict.ValidateThresholds` was written for and never
// had. It runs once, at the transition into Running, against the PINNED plan —
// early enough that an operator watching the run sees the note before any
// verdict exists, and late enough that it describes exactly the thresholds that
// will decide it.
//
// It never changes the phase, and it never fails the write it rides on: an
// advisory that could wedge a run would be a worse bug than the one it warns
// about. Malformed thresholds are not here at all — they refused the run at plan
// time, before this point is reachable.
func (r *BurnInRunReconciler) applyThresholdsSoundCondition(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) {
	advice := p.unsoundThresholds()

	cond := metav1.Condition{
		Type:               burninv1alpha1.ConditionThresholdsSound,
		Status:             metav1.ConditionTrue,
		Reason:             burninv1alpha1.ReasonThresholdsReviewed,
		ObservedGeneration: run.Generation,
		LastTransitionTime: metav1.NewTime(r.now()),
		Message: "every threshold in this run's pinned plan is a gate this operator can vouch for as a form of comparison. " +
			"That is not a claim about its NUMBER: whether a bound is calibrated for the hardware is measured, not linted.",
	}
	if len(advice) > 0 {
		cond.Status = metav1.ConditionFalse
		cond.Reason = burninv1alpha1.ReasonUnsoundThresholds
		cond.Message = clampConditionMessage(summariseThresholdAdvice(advice))
		log.FromContext(ctx).Info("unsound thresholds in pinned plan", "run", run.Name, "count", len(advice), "detail", cond.Message)
	}
	meta.SetStatusCondition(&run.Status.Conditions, cond)
}

// applyPlanExpandedCondition says how much work this run actually is.
//
// Only set when variants expanded something: a run whose profile means what it
// says needs no note, and a condition that always appears is a condition nobody
// reads. Like the threshold advisory it never changes the phase and never fails
// the write it rides on.
func (r *BurnInRunReconciler) applyPlanExpandedCondition(
	ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan,
) {
	sum := p.variantSummary()
	if len(sum.Expanded) == 0 {
		return
	}
	perNode := sum.Executions * len(p.Targets)
	msg := fmt.Sprintf(
		"variants expanded %d profile entries into %d executions (%s). Across %d target "+
			"nodes that is %d node-executions; at maxConcurrentNodes %d they are not all "+
			"concurrent. This arithmetic is not visible in the profile, which is why it "+
			"is stated here rather than discovered by waiting.",
		sum.Entries, sum.Executions, strings.Join(sum.Expanded, ", "),
		len(p.Targets), perNode, maxConcurrentNodes(run))

	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               burninv1alpha1.ConditionPlanExpanded,
		Status:             metav1.ConditionTrue,
		Reason:             burninv1alpha1.ReasonVariantsExpanded,
		ObservedGeneration: run.Generation,
		LastTransitionTime: metav1.NewTime(r.now()),
		Message:            clampConditionMessage(msg),
	})
	log.FromContext(ctx).Info("variants expanded the plan",
		"run", run.Name, "entries", sum.Entries, "executions", sum.Executions)
}

// isConfigError reports whether a resolution failure is the spec's fault
// (missing referent, invalid combination) rather than the cluster's.
func isConfigError(err error) bool {
	if apierrors.IsNotFound(err) {
		return true
	}
	// Structural errors from our own validation carry no API status at all.
	return !apierrors.IsInternalError(err) && apierrors.ReasonForError(err) == metav1.StatusReasonUnknown && !apierrors.IsTimeout(err) &&
		!isTransportError(err)
}

// isTransportError catches the errors a live client surfaces that carry no
// structured API reason (connection refused, context deadline).
func isTransportError(err error) bool {
	s := err.Error()
	return strings.Contains(s, "connection refused") ||
		strings.Contains(s, "context deadline exceeded") ||
		strings.Contains(s, "TLS handshake timeout") ||
		strings.Contains(s, "unavailable")
}

func profileSinks(profile *burninv1alpha1.BurnInProfile) []string {
	if profile == nil {
		return nil
	}
	return profile.Spec.Sinks
}

func (r *BurnInRunReconciler) resolveProfile(ctx context.Context, run *burninv1alpha1.BurnInRun) (*burninv1alpha1.BurnInProfile, []resolvedTest, error) {
	var profile burninv1alpha1.BurnInProfile
	if err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: run.Spec.ProfileRef}, &profile); err != nil {
		return nil, nil, fmt.Errorf("profile %q: %w", run.Spec.ProfileRef, err)
	}

	tests := make([]resolvedTest, 0, len(profile.Spec.Tests))
	for i, pt := range profile.Spec.Tests {
		required := pt.Required == nil || *pt.Required
		switch {
		case pt.TestRef != "" && pt.Inline != nil:
			return &profile, nil, fmt.Errorf("profile test %d sets both testRef and inline", i)
		case pt.TestRef != "":
			var t burninv1alpha1.BurnInTest
			if err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: pt.TestRef}, &t); err != nil {
				return &profile, nil, fmt.Errorf("testRef %q: %w", pt.TestRef, err)
			}
			tests = append(tests, expandVariants(t.Name, t.Spec, required, pt.Variants)...)
		case pt.Inline != nil:
			tests = append(tests, expandVariants(
				fmt.Sprintf("inline-%d-%s", i, pt.Inline.Kind), *pt.Inline, required, pt.Variants)...)
		default:
			return &profile, nil, fmt.Errorf("profile test %d names no testRef and no inline spec", i)
		}
	}
	return &profile, tests, nil
}

// expandVariants turns one profile entry into one resolvedTest per cell.
//
// This is the whole of the matrix feature, and it is deliberately the whole of
// it. Expansion happens ONCE, here, before buildPlan — so every stage
// downstream (the duplicate-name check, Pair and Group topology validation, pod
// naming, result identity, verdict, delivery keys, TestResult.Nodes) works
// unchanged, because none of them ever learns variants exist. That is why this
// is a small change rather than a reconciler rewrite, and it is the property to
// protect if this is ever extended.
//
// With no variants it returns exactly the one test it was given, so a profile
// written before variants existed plans identically.
func expandVariants(
	name string,
	spec burninv1alpha1.BurnInTestSpec,
	required bool,
	variants []burninv1alpha1.TestVariant,
) []resolvedTest {
	if len(variants) == 0 {
		return []resolvedTest{{name: name, spec: spec, required: required}}
	}

	out := make([]resolvedTest, 0, len(variants))
	for _, v := range variants {
		// A deep copy per cell: the overlays below must not reach back into the
		// parent spec, or the second variant would inherit the first's edits and
		// a sweep would silently measure the wrong thing in every cell after the
		// first.
		cell := *spec.DeepCopy()

		if v.DurationSeconds != nil {
			cell.DurationSeconds = *v.DurationSeconds
		}
		// A VARIANT THAT OVERLAYS A RUNNER FIELD IS ASKING FOR A RUNNER BLOCK,
		// and the test it overlays usually has none: a built-in kind resolves its
		// image from pkg/runnerimages, so there is nothing for its author to put
		// in `runner:`. That is the CANONICAL use of this feature — five cells
		// differing by one environment variable — and writing through a nil
		// pointer panicked the reconciler on the first profile anybody wrote.
		//
		// A nil dereference here is not one failed run. controller-runtime
		// recovers the panic and requeues, so it crash-loops every pass and the
		// operator stops making progress for every OTHER run in the cluster too.
		//
		// Creating an empty block is safe for image resolution: Resolve treats an
		// empty Image and an empty ImagesByVendor exactly as it treats a nil
		// RunnerSpec, so a cell that gains one still lands on the kind's default.
		if v.Args != nil || v.Env != nil {
			if cell.Runner == nil {
				cell.Runner = &burninv1alpha1.RunnerSpec{}
			}
		}
		if v.Args != nil {
			cell.Runner.Args = append([]string(nil), v.Args...)
		}
		if v.Env != nil {
			cell.Runner.Env = append([]corev1.EnvVar(nil), v.Env...)
		}
		if v.RepeatCount != nil {
			cell.RepeatCount = v.RepeatCount
		}
		if v.Thresholds != nil {
			// REPLACE, never merge. See TestVariant.Thresholds: merging by
			// metric name would silently retain a gate the author believed
			// replaced, and a node failed against a threshold nobody can find
			// in the profile has a verdict with nothing to read that explains
			// it. An empty non-nil list therefore means "no thresholds", which
			// is a thing a variant may legitimately want.
			cell.Thresholds = append([]burninv1alpha1.Threshold(nil), v.Thresholds...)
		}

		out = append(out, resolvedTest{
			name:     name + "-" + v.Name,
			spec:     cell,
			required: required,
			axes:     copyAxes(v.Axes),
			parent:   name,
		})
	}
	return out
}

func copyAxes(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func (r *BurnInRunReconciler) resolveTargets(ctx context.Context, sel burninv1alpha1.TargetSelector) ([]string, error) {
	if len(sel.NodeNames) > 0 {
		return sel.NodeNames, nil
	}
	if len(sel.NodeSelector) == 0 {
		return nil, fmt.Errorf("target selects no nodes: set nodeNames or nodeSelector")
	}
	var nodes corev1.NodeList
	if err := r.List(ctx, &nodes, client.MatchingLabels(sel.NodeSelector)); err != nil {
		return nil, fmt.Errorf("list nodes: %w", err)
	}
	if len(nodes.Items) == 0 {
		return nil, fmt.Errorf("nodeSelector %v matches no nodes", sel.NodeSelector)
	}
	names := make([]string, 0, len(nodes.Items))
	for _, n := range nodes.Items {
		names = append(names, n.Name)
	}
	return names, nil
}

// captureFingerprint records what hardware this verdict applies to. v1 reads
// the salient identity from the Node objects; the richer NodeFingerprint CRD
// flow can replace this without changing the envelope shape.
//
// This is data capture, not dispatch: nothing in this controller branches on
// what it finds here. The fingerprint travels with the verdict so a consumer
// can tell which part was judged, and a vendor label appearing in it must never
// become a reason to run a different test.
func (r *BurnInRunReconciler) captureFingerprint(ctx context.Context, targets []string) map[string]string {
	fp := map[string]string{}
	for _, name := range targets {
		var node corev1.Node
		if err := r.Get(ctx, types.NamespacedName{Name: name}, &node); err != nil {
			fp[name] = "node object unavailable at run start"
			continue
		}
		parts := []string{
			"kernel=" + node.Status.NodeInfo.KernelVersion,
			"os=" + node.Status.NodeInfo.OSImage,
			"arch=" + node.Status.NodeInfo.Architecture,
		}
		// Accelerator identity labels if present (device-plugin / GFD / glimmer).
		for _, k := range []string{"nvidia.com/gpu.product", "glimmer.ai/gpu-arch", "glimmer.ai/hw-class"} {
			if v, ok := node.Labels[k]; ok {
				parts = append(parts, k+"="+v)
			}
		}
		fp[name] = strings.Join(parts, " ")
	}
	return fp
}

// ─── Step: execute the pinned plan ────────────────────────────────────────────

// step advances a Running run.
//
// The order of the gates is a policy statement. Cancel dominates everything,
// because a run that could not be stopped would be unstoppable without deleting
// the object. The whole-run deadline comes next, because a suspended run still
// burns the wall clock it was budgeted. Suspend comes last, and only stops new
// work.
func (r *BurnInRunReconciler) step(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	if cancelRequested(run) || run.Annotations[cancellingAnnotation] != "" {
		return r.cancel(ctx, run, p)
	}
	if r.deadlineExceeded(run) {
		return r.expire(ctx, run, p)
	}

	pass, err := r.execute(ctx, run, p, !suspended(run))
	if err != nil {
		return ctrl.Result{}, err
	}

	switch {
	case pass.running || pass.pending:
		return r.persistPass(ctx, run, p, pass)
	case pass.harvested:
		// A test finished this pass: persist before moving on, so the delivery
		// and the recorded result stay adjacent in time.
		if err := r.Status().Update(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
		// THE RECORD IS DURABLE. Only now may a pod this pass condemned be
		// destroyed — issue #247, and the same rule terminate states at length.
		// A failure here leaves an overdue pod running for one more pass, which
		// the next pass's own overdue check settles; the opposite order left a
		// destroyed pod with no record that it ever ran.
		if err := r.killPods(ctx, pass.kill); err != nil {
			return ctrl.Result{}, err
		}
		if pass.checkpointed {
			r.deliverCheckpoint(ctx, run, p.Sinks, r.checkpointSequence(run, p))
		}
		return ctrl.Result{Requeue: true}, nil
	case suspended(run):
		// Every planned execution has reported, but the run is suspended and a
		// suspended run reaches no terminal phase and produces no verdict. It
		// waits, holding its cordons, because that is what the operator asked
		// for; the way to actually end a run is spec.cancel, which is one-way
		// and says so in the phase.
		// A zero pass, deliberately: nothing is in flight, so nothing votes for
		// a faster cadence. A suspended run holding its cordons and waiting for
		// a human gets the settled backstop, which is what it should have — the
		// thing it is waiting for is a spec edit, and that arrives as an event.
		return ctrl.Result{RequeueAfter: r.nextWake(run, passResult{})}, nil
	}
	return r.finalize(ctx, run, p)
}

// passResult is what one sweep of the plan achieved.
type passResult struct {
	// running means at least one execution has a live pod on a node.
	running bool
	// pending means work remains that this pass did not launch — blocked by
	// the concurrency interlock, by suspension, or by a cancel drain.
	pending bool
	// harvested means at least one attempt completed and was recorded.
	harvested bool
	// checkpointed means at least one in-flight result had its evidence
	// refreshed and a Checkpoint envelope is owed.
	checkpointed bool
	// kill are pods to destroy ONCE the status recording why is durable. See
	// advanceEffect.kill for the failure this ordering prevents.
	kill []*corev1.Pod
	// checkpointEvery is the shortest checkpoint interval among the tests
	// actually IN FLIGHT this pass, or 0 when none of them checkpoints.
	//
	// This is the whole of #158. The wake used to come from the plan-wide
	// minimum, so one test with a 60-second interval set the cadence for a
	// profile whose other tests check in hourly — INCLUDING while those other
	// tests were the only ones running. The information was already on the
	// pass; it was simply not threaded through.
	checkpointEvery time.Duration
	// awaitingStart means some in-flight pod has not started yet.
	//
	// That is the window where scheduling problems appear — unschedulable,
	// ImagePullBackOff, a node that went away — and where a fast reaction is
	// worth the apiserver traffic. Once every pod is running, the run is doing
	// exactly what it was asked to do and there is nothing to learn by asking
	// again in thirty seconds.
	awaitingStart bool
	// dirty means status was mutated and has to be written back. It is tracked
	// separately from harvested because merely OPENING a result — recording
	// that an execution has started on a node — is a status change with no
	// completion attached, and losing it would leave a long soak with no
	// StartedAt and nothing on a dashboard until the day it finishes.
	dirty bool
	// rendezvous means a Pair execution is between its two pods: the server is
	// up and the client is gated on it becoming Ready. It only affects how soon
	// the next pass is scheduled — the window is dead time on hardware the run
	// is already holding, so it is worth a tighter backstop than the general
	// waiting poll.
	rendezvous bool
}

// advanceEffect is the side effect one execution had on the run's status.
type advanceEffect struct {
	dirty        bool
	checkpointed bool
	rendezvous   bool
	// kill are pods this pass decided to destroy, DEFERRED until the status
	// recording that decision is durable.
	//
	// This is the #84 rule applied to the timeout paths, which is issue #247:
	// terminate had it right and these five sites had it inverted. They deleted
	// the pod, then recorded the Error in memory, and left the write to step —
	// so a lost resourceVersion race, or an error from any LATER target in the
	// same pass, discarded the record while the pod stayed destroyed. The
	// attempt then never happened as far as the apiserver was concerned,
	// ErrorRetries was never incremented, and the run recreated the same pod
	// under the same attempt number: a loop that spends no retry budget and
	// records nothing, which is exactly what a rolling update of the manager
	// produces.
	//
	// Carrying the pods here instead of deleting in place keeps ONE place
	// responsible for the ordering, the same way completeAttempt is the one
	// place a verdict is decided.
	kill []*corev1.Pod
}

// execute sweeps the plan once. launch=false harvests and checkpoints exactly
// as usual but starts nothing new, which is what both suspension and a graceful
// cancel need: neither is a reason to throw away an execution already paid for.
func (r *BurnInRunReconciler) execute(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, launch bool) (passResult, error) {
	var out passResult

	busy, awaitingStart, err := r.busyNodes(ctx, run)
	if err != nil {
		return out, err
	}
	// A WAVE CAN BE MANY PODS LONG, so the nodes this run HOLDS are not only the
	// ones with a pod alive this instant.
	//
	// busyNodes counts live pods, which was the whole truth while every execution
	// was one pod. A segmented soak is not: between segment N and segment N+1
	// there is a moment with no pod at all, and a repeat and an error-retry have
	// the same gap. Reading the interlock from live pods alone made that gap look
	// like released capacity, so a second target took the slot and
	// releaseHeldNodes gave the first node's cordon back — halfway through a
	// multi-day acceptance soak.
	//
	// Both halves of that are severe. The soak STOPS BEING A SOAK: at cap 1 over
	// two targets each node runs one window then idles for one, and a part
	// allowed to cool for as long as it was loaded never reaches the thermal
	// steady state the test exists to hold it at, so throttling and
	// heat-dependent faults are precisely what that schedule cannot find. And the
	// cordon is the only thing keeping FOREIGN workload off a node under burn-in
	// — runner pods tolerate node.kubernetes.io/unschedulable, so the cordon
	// never protected against this operator, only against everything else — which
	// means ordinary fleet work could land beside a measurement in progress.
	//
	// Seeding rather than replacing is deliberate: this can only ever make the
	// run hold MORE, never less, so the live-pod count stays the floor and a
	// controller restart still cannot lose track of what is already running.
	for node := range holdingNodes(run) {
		busy[node] = true
	}
	// A pod that exists and has not started keeps the fast cadence: that is the
	// window where scheduling problems appear — unschedulable, ImagePullBackOff,
	// a node that went away — and where a fast reaction is worth the traffic.
	out.awaitingStart = awaitingStart
	capNodes := maxConcurrentNodes(run)

	for i := range p.Tests {
		t := p.Tests[i]
		if p.FailFast && hasRequiredFailure(run, p) {
			// Remaining tests are never scheduled; termination cleans up any
			// pods already in flight. Their absence from Results is the record
			// that they did not run.
			break
		}

		// Tests the operator can settle without scheduling anything.
		if settled := r.settleWithoutPod(ctx, run, p, t); settled {
			continue
		}

		var test passResult
		apply := func(state advanceState, effect advanceEffect) {
			test.checkpointed = test.checkpointed || effect.checkpointed
			test.dirty = test.dirty || effect.dirty
			test.rendezvous = test.rendezvous || effect.rendezvous
			test.kill = append(test.kill, effect.kill...)
			switch state {
			case advanceRunning:
				test.running = true
				// This test is in flight, so ITS checkpoint interval is one the
				// wake has to respect. A test that is not running does not get
				// a vote, which is the fix.
				if cp := checkpointInterval(&t.Spec); cp > 0 &&
					(test.checkpointEvery == 0 || cp < test.checkpointEvery) {
					test.checkpointEvery = cp
				}
			case advancePending:
				test.pending = true
			case advanceHarvested:
				test.harvested = true
			}
		}

		switch t.Spec.Scope {
		case burninv1alpha1.ScopePair:
			// One execution over two nodes, not two executions. advancePair
			// owns the whole unit — both pods, the rendezvous Service, the
			// ready gate and the single combined verdict.
			state, effect, err := r.advancePair(ctx, run, p, i, t, launch, busy, capNodes)
			if err != nil {
				return out, err
			}
			apply(state, effect)
		case burninv1alpha1.ScopeGroup:
			// One execution over EVERY target node. advanceGroup owns the whole
			// unit — every rank's pod, the rendezvous Service, the root ready
			// gate and the single combined verdict.
			state, effect, err := r.advanceGroup(ctx, run, p, i, t, launch, busy, capNodes)
			if err != nil {
				return out, err
			}
			apply(state, effect)
		default:
			for _, node := range p.Targets {
				state, effect, err := r.advance(ctx, run, p, i, t, node, launch, busy, capNodes)
				if err != nil {
					return out, err
				}
				apply(state, effect)
			}
		}

		out.running = out.running || test.running
		out.pending = out.pending || test.pending
		out.harvested = out.harvested || test.harvested
		out.checkpointed = out.checkpointed || test.checkpointed
		out.dirty = out.dirty || test.dirty
		out.rendezvous = out.rendezvous || test.rendezvous
		out.kill = append(out.kill, test.kill...)
		if test.checkpointEvery > 0 &&
			(out.checkpointEvery == 0 || test.checkpointEvery < out.checkpointEvery) {
			out.checkpointEvery = test.checkpointEvery
		}

		// Tests run in plan order, one at a time: the next test must not start
		// on a node while this one still owes it a result there. Two tests
		// overlapping on one node would also break the concurrency interlock's
		// meaning, since the cap counts nodes, not pods.
		if test.running || test.pending || test.harvested {
			break
		}
	}

	// Give back everything this run is holding that it is no longer using.
	// busy is the wave — nodes with a live pod, plus the ones admitted during
	// this pass — so anything cordoned and not in it is capacity the run is
	// sitting on for nothing.
	released, err := r.releaseHeldNodes(ctx, run, busy)
	if err != nil {
		return out, err
	}
	out.dirty = out.dirty || released
	return out, nil
}

// persistPass writes whatever the pass produced and schedules the next wake-up.
func (r *BurnInRunReconciler) persistPass(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, pass passResult) (ctrl.Result, error) {
	if pass.dirty || pass.harvested || pass.checkpointed {
		if err := r.Status().Update(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}
	if pass.checkpointed {
		// Delivered after the status write, unlike a phase change: a checkpoint
		// is evidence rather than a transition, so the cheap failure is a
		// consumer seeing a snapshot the object already holds. The expensive
		// one would be re-deriving a delivery from status that never landed.
		r.deliverCheckpoint(ctx, run, p.Sinks, r.checkpointSequence(run, p))
	}
	wake := r.nextWake(run, pass)
	if pass.rendezvous && wake > rendezvousPollInterval {
		wake = rendezvousPollInterval
	}
	return ctrl.Result{RequeueAfter: wake}, nil
}

// nextWake is the backstop poll interval while work is outstanding.
//
// Pod events drive the common case; this covers the pod that never emits one,
// the checkpoint that is due before the next event, and the run deadline that
// would otherwise be noticed up to a poll interval late.
//
// # Why this is not a constant — issue #158
//
// It used to floor at 30 seconds for the whole of a run, so a seven-day soak
// reconciled roughly 20,000 times, and every pass listed pods to recompute the
// busy set, possibly fetched logs, and possibly wrote status. For a run that is
// BY DESIGN doing nothing observable for hours at a stretch, that is a lot of
// apiserver traffic to learn nothing.
//
// Two things decide the cadence now, and both come from the pass rather than
// from the plan:
//
//   - the checkpoint intervals of the tests actually IN FLIGHT. The plan-wide
//     minimum meant one 60-second test set the cadence for a profile whose
//     other tests check in hourly, including while those other tests were the
//     only ones running.
//   - whether every in-flight pod has STARTED. Before they start is the window
//     where scheduling problems appear and a fast reaction is worth the
//     traffic; after, the run is doing exactly what it was asked to and there
//     is nothing to learn by asking again in thirty seconds.
//
// A day-long run goes from roughly 2,880 wakes to roughly 288.
//
// checkpointSequence is deliberately NOT changed by any of this: it stays
// derived from the pinned plan-wide minimum, because it is what makes a
// checkpoint's DeliveryID stable across retries, and changing its basis
// mid-flight would break idempotency for every consumer.
func (r *BurnInRunReconciler) nextWake(run *burninv1alpha1.BurnInRun, pass passResult) time.Duration {
	wake := settledPollInterval
	if pass.awaitingStart {
		wake = waitingPollInterval
	}
	if cp := pass.checkpointEvery; cp > 0 && cp < wake {
		wake = cp
	}
	if d := runDeadline(run); d > 0 && run.Status.StartedAt != nil {
		if remaining := run.Status.StartedAt.Add(d).Sub(r.now()); remaining > 0 && remaining < wake {
			wake = remaining
		}
	}
	// No zero guard, deliberately. Every assignment above is positive-guarded —
	// the backstop constants are positive, checkpointEvery is taken only when
	// `> 0`, and the deadline remainder only when `> 0` — so a non-positive wake
	// is unreachable. A guard for it was written first and removed when mutating
	// it away changed nothing: an unreachable branch in the shape of a safety
	// net is worse than none, because it invites the reader to believe something
	// is being defended. TestTheWakeIsAlwaysPositive pins the property itself.
	return wake
}

// advanceState is what one (test, node) execution did this pass.
type advanceState int

const (
	// advanceDone: the execution has a terminal result and owes nothing.
	advanceDone advanceState = iota
	// advanceRunning: a pod exists and has not terminated.
	advanceRunning
	// advancePending: work remains but nothing was launched for it.
	advancePending
	// advanceHarvested: an attempt completed and was recorded this pass.
	advanceHarvested
)

// advance moves one (test, node) execution forward by exactly one step.
func (r *BurnInRunReconciler) advance(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	index int,
	t plannedTest,
	node string,
	launch bool,
	busy map[string]bool,
	capNodes int,
) (advanceState, advanceEffect, error) {
	var none advanceEffect
	nodes := []string{node}

	res := resultFor(run, t.Name, node)
	if res != nil && res.Phase.IsTerminal() {
		return advanceDone, none, nil
	}

	attempt, _ := nextAttempt(res)
	name := podName(run, index, node, attempt)

	var pod corev1.Pod
	err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, &pod)
	switch {
	case apierrors.IsNotFound(err):
		if !launch {
			return advancePending, none, nil
		}
		// The facility interlock. This is the single place a run can add load
		// to the room, and the cap is enforced here rather than by slicing the
		// target list up front so that it is honoured across tests, across
		// repeats, and across a controller restart — the count comes from the
		// pods that actually exist, not from a wave index nobody persisted.
		if !busy[node] && len(busy) >= capNodes {
			return advancePending, none, nil
		}
		newPod, buildErr := podForTest(run, index, attempt, t.Name, &t.Spec, t.Axes, node, r.vendorFor(ctx, node), run.Spec.Target, nil)
		if buildErr != nil {
			// Unbuildable pod (no image for the kind): machinery error, and
			// asking again cannot fix it.
			r.completeAttempt(ctx, run, p, t, nodes, attempt, "", nil,
				runner.Result{Verdict: runner.VerdictError, Message: buildErr.Error()})
			return advanceHarvested, advanceEffect{dirty: true}, nil
		}
		// Hold the node BEFORE the pod exists, so nothing else can be scheduled
		// beside a burn-in that is about to saturate it.
		cordoned, cordonErr := r.cordonNode(ctx, run, node, &t.Spec)
		if cordonErr != nil {
			return advancePending, none, cordonErr
		}
		if ownErr := controllerutil.SetControllerReference(run, newPod, r.Scheme); ownErr != nil {
			return advancePending, advanceEffect{dirty: cordoned}, ownErr
		}
		if createErr := r.Create(ctx, newPod); createErr != nil && !apierrors.IsAlreadyExists(createErr) {
			return advancePending, advanceEffect{dirty: cordoned}, createErr
		}
		busy[node] = true
		return advanceRunning, advanceEffect{dirty: cordoned}, nil
	case err != nil:
		return advancePending, none, err
	}

	if podLive(&pod) {
		busy[node] = true
	}

	_, terminated, _, _ := podOutcome(&pod)
	if !terminated {
		// activeDeadlineSeconds only starts once the pod is bound to a node;
		// an unschedulable pod (typo'd node name, missing toleration) has no
		// deadline at all and would wedge the run in Running forever.
		if r.podOverdue(&pod, &t.Spec) {
			// Recorded now, destroyed by step once the record is durable.
			overdue := pod
			r.completeAttempt(ctx, run, p, t, nodes, attempt, pod.Name, &pod, runner.Result{
				Verdict: runner.VerdictError,
				Message: fmt.Sprintf("pod never completed within its window (last phase %q) — unschedulable target or stuck image pull; no verdict%s",
					pod.Status.Phase, pendingDetail(&pod)),
			})
			return advanceHarvested, advanceEffect{dirty: true, kill: []*corev1.Pod{&overdue}}, nil
		}
		if !podStarted(&pod) {
			// Scheduled but not executing. Nothing has been tested yet, so no
			// result is opened and StartedAt stays unset.
			return advanceRunning, none, nil
		}
		_, opened := r.beginAttempt(run, t, nodes, attempt, &pod)
		checkpointed := r.checkpoint(ctx, run, t, node, &pod)
		return advanceRunning, advanceEffect{dirty: opened || checkpointed, checkpointed: checkpointed}, nil
	}

	// Terminated. Open the result if we never observed the pod running — the
	// execution plainly happened and must not be recorded as having no start.
	r.beginAttempt(run, t, nodes, attempt, &pod)

	parsed, logErr := r.harvestPod(ctx, t, &pod)
	if logErr != nil && parsed.Verdict == runner.VerdictPass && len(t.Spec.Thresholds) > 0 &&
		missingThresholdMetric(parsed, t.Spec.Thresholds) {
		// The exit code says pass but the metrics are gone and thresholds
		// exist. Fail-closed evaluation would blame the hardware for a log
		// fetch failure; the honest verdict is Error, machinery's fault.
		parsed.Verdict = runner.VerdictError
		parsed.Message = fmt.Sprintf("runner logs unavailable (%v) — thresholds cannot be evaluated; no verdict", logErr)
	}
	r.completeAttempt(ctx, run, p, t, nodes, attempt, pod.Name, &pod, parsed)
	// After completeAttempt, which is what guarantees the result exists — and
	// before the pod is deleted, because a container's log is the only copy of
	// its stdout and the kubelet reclaims it with the pod.
	r.harvestArtifacts(ctx, run, t.Name, nodes, pod.Name, parsed.Artifacts)
	return advanceHarvested, advanceEffect{dirty: true}, nil
}

// harvestPod turns a terminated pod into a parsed runner result, plus whatever
// went wrong reading its log.
//
// It deliberately stops short of the log-fetch rule ("a pass we cannot verify is
// an Error, not a Fail"), because that rule is about an EXECUTION and a Pair
// execution has two pods. Applying it per pod would error a pair whose client
// reported every number a threshold asks for merely because the server's log
// was unreadable. The callers apply it once, to the unit.
func (r *BurnInRunReconciler) harvestPod(ctx context.Context, t plannedTest, pod *corev1.Pod) (runner.Result, error) {
	exitCode, _, reason, detail := podOutcome(pod)
	stdout, logErr := r.fetchLogs(ctx, pod)
	parsed := runner.Parse(string(t.Spec.Kind), stdout, exitCode)
	if parsed.Verdict == runner.VerdictError {
		// THE KUBELET'S OWN DETAIL, and it is the whole point of carrying it
		// (#52). A container that never started produces no stdout at all, so
		// this is the ONLY thing that can say why: for a StartError the message
		// is the createContainer hook failure, and for an ImagePullBackOff it
		// names the image and the registry error. Without it every unstartable
		// container recorded the same sentence, and a fleet-wide outage caused
		// by a broken CDI hook was indistinguishable from a typo in
		// spec.runner.image.
		account := fmt.Sprintf("runner terminated abnormally (exit %d, reason %q)", exitCode, reason)
		if d := strings.TrimSpace(detail); d != "" {
			account += ": " + clampPodDetail(d)
		}
		switch {
		case parsed.Message == "":
			parsed.Message = account
		case !runnerChoseExitCode(exitCode):
			// The runner said something AND the exit code is not one it can
			// produce, so the process was ended by something outside it and the
			// kubelet is the only witness to what. Both are kept: the runner's
			// last words say how far it got, the kubelet's account says what
			// stopped it.
			//
			// This branch used to be missing, and the omission cost exactly the
			// evidence #280 is short of. A pod SIGKILLed after printing its
			// marker parses as Error (137 is not a contract code) with a
			// non-empty Message, so it took the FIRST branch's place and the
			// kubelet's reason was dropped — including "OOMKilled", which is the
			// one field that distinguishes an out-of-memory kill from every
			// other 137. The stored result read as though the runner had had its
			// say and ended itself.
			parsed.Message += " [" + account + "]"
		}
	}
	if reason == "DeadlineExceeded" {
		// The kubelet killed the pod at its deadline. Whatever the container
		// exited with — including 0 from a SIGTERM-trapping entrypoint — the
		// test did not run to completion, so there is no verdict.
		parsed.Verdict = runner.VerdictError
		parsed.Message = "test exceeded its deadline and was killed — no verdict"
	}
	return parsed, logErr
}

// settleWithoutPod records results for tests that must not schedule anything.
//
// The list is deliberately short, and it is about what THIS OPERATOR cannot
// execute — never about what a given piece of hardware supports. "This test
// does not apply to this part" is the runner's judgement, returned as exit 2,
// and it is made after looking at the device rather than at a label.
//
// Node, Pair and Group all execute. What is left here is an UNRECOGNISED scope,
// which is reachable because TestScope is an open string on the API: a run
// created against a newer CRD, or simply a typo, must land as Error rather than
// be quietly dropped — a required acceptance test the operator cannot run must
// never let hardware pass by omission.
func (r *BurnInRunReconciler) settleWithoutPod(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, t plannedTest) bool {
	switch t.Spec.Scope {
	case "", burninv1alpha1.ScopeNode, burninv1alpha1.ScopePair, burninv1alpha1.ScopeGroup:
		return false
	}
	if resultFor(run, t.Name, "") == nil {
		now := metav1.NewTime(r.now())
		r.appendSettled(ctx, run, p, burninv1alpha1.TestResult{
			Name:       t.Name,
			Kind:       t.Spec.Kind,
			Scope:      t.Spec.Scope,
			Phase:      burninv1alpha1.RunError,
			FinishedAt: &now,
			Message: fmt.Sprintf("scope %q is not a scope this operator version recognises (it runs Node, Pair and Group scope) — "+
				"the test was NOT run; treat this hardware as unjudged for it", t.Spec.Scope),
		})
	}
	return true
}

func (r *BurnInRunReconciler) fetchLogs(ctx context.Context, pod *corev1.Pod) (string, error) {
	if r.PodLogs == nil {
		return "", nil
	}
	out, err := r.PodLogs(ctx, pod.Namespace, pod.Name)
	if err != nil {
		log.FromContext(ctx).Error(err, "could not read runner logs", "pod", pod.Name)
		return "", err
	}
	return out, nil
}

// podWindow is the whole life this operator allows ONE pod of a test: its
// execution duration, plus the deadline grace the kubelet gets for start-up,
// plus the scheduling grace it gets for being placed at all.
//
// One SEGMENT's window for a segmented soak; podForTest sizes the pod's own
// deadline from the same executionDurationSeconds, and the two must not
// disagree about the same pod. It is a function rather than an expression
// because the heat declaration's expiry is sized from it too (see heatExpiry),
// and a marker that could expire before the operator itself gave up on the pod
// would let a soak be drained mid-measurement.
func podWindow(spec *burninv1alpha1.BurnInTestSpec) time.Duration {
	return time.Duration(executionDurationSeconds(spec)+deadlineGraceSeconds)*time.Second + schedulingGracePeriod
}

// podOverdue reports whether a non-terminated pod has outlived its whole
// window: duration + deadline grace + scheduling grace since creation.
func (r *BurnInRunReconciler) podOverdue(pod *corev1.Pod, spec *burninv1alpha1.BurnInTestSpec) bool {
	// A zero CreationTimestamp means the server never stamped the object
	// (only possible in tests); "overdue since year 1" is not a judgment.
	if pod.CreationTimestamp.IsZero() {
		return false
	}
	return r.now().Sub(pod.CreationTimestamp.Time) > podWindow(spec)
}

// ─── Attempts ─────────────────────────────────────────────────────────────────

// beginAttempt opens the result for (test, node) if it does not exist and
// records that an execution has started. The second return says whether it
// changed anything, so the caller knows a status write is owed.
//
// It is idempotent, and it is the ONLY place TestResult.StartedAt is set. The
// distinction it encodes: a pod that exists is scheduling, a pod the kubelet
// has started is testing. Only the second is time the hardware was occupied,
// which is why StartedAt is left nil for a pod that never started at all — an
// unschedulable pod reaped after its grace period has an Error and a
// FinishedAt, and honestly no start.
func (r *BurnInRunReconciler) beginAttempt(
	run *burninv1alpha1.BurnInRun,
	t plannedTest,
	nodes []string,
	attempt int32,
	pod *corev1.Pod,
) (*burninv1alpha1.TestResult, bool) {
	before := len(run.Status.Results)
	res := ensureResult(run, t, nodes)
	changed := len(run.Status.Results) != before

	var started *metav1.Time
	if pod != nil && podStarted(pod) {
		s := attemptStart(pod, r.now())
		started = &s
		if res.StartedAt == nil {
			res.StartedAt = &s
			changed = true
		}
	}

	if n, inProgress := nextAttempt(res); inProgress && n == attempt {
		a := &res.Attempts[len(res.Attempts)-1]
		if a.StartedAt == nil && started != nil {
			a.StartedAt = started
			changed = true
		}
		return res, changed
	}
	res.Attempts = append(res.Attempts, burninv1alpha1.TestAttempt{
		Attempt: attempt,
		Trigger: triggerFor(res),
		// The rank-0 / initiating node, which at Node scope is the only one and
		// at Pair scope is the server. TestResult.Nodes carries the full set.
		Node:      nodes[0],
		PodName:   podNameOf(pod),
		Phase:     burninv1alpha1.RunRunning,
		StartedAt: started,
	})
	return res, true
}

// completeAttempt records the outcome of one execution and decides what it
// means for the test as a whole.
//
// The two rules that live here are the ones the whole design hangs off:
//
//   - REPEATS ARE AN AND. Every execution demanded by RepeatCount must pass.
//     The first Failed attempt settles the test as Failed and no further pass
//     can retract it, because the point of a repeat is to catch the fault that
//     does not reproduce every time.
//
//   - A RETRY ONLY EVER FOLLOWS AN ERROR. An Error means the machinery
//     malfunctioned and the hardware was never judged, so running it again is
//     the correct response. A Failed attempt is a measurement, and re-running a
//     measurement until it comes out clean launders a hardware fault into an
//     acceptance — marginal hardware is precisely the hardware that passes on
//     the second try.
//
//   - A SEGMENT IS NEITHER. When the test is a segmented soak
//     (TestResult.SegmentsRequired > 0), the exit code alone decides what one
//     window means and THE VERDICT IS RENDERED ONCE, over the accumulated
//     aggregate, on the last segment. That separation is the whole of issue
//     #157: a threshold evaluated per segment would gate fifteen minutes of a
//     week and fail a soak on a window, and it would also make the run's own
//     answer depend on how finely it happened to be sliced.
//
// The switch below is exhaustive over the attempt phases and only the Error arm
// consults the retry budget. That is not an accident of structure; it is the
// rule, and TestAttempt.Trigger records enough history to audit it after the
// fact — including for segments, which is why AttemptSegment exists rather than
// being reported as a repeat.
//
// This stays the ONLY place a verdict is decided, for Node, Pair and Group scope
// and for segmented and unsegmented tests alike. attemptOutcome and gateOutcome
// are called from here and from nowhere else; a second decision point would be
// two answers to one question, and the answer that matters is which hardware
// gets shipped.
func (r *BurnInRunReconciler) completeAttempt(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	t plannedTest,
	nodes []string,
	attempt int32,
	podName string,
	pod *corev1.Pod,
	parsed runner.Result,
) {
	res, _ := r.beginAttempt(run, t, nodes, attempt, pod)
	now := metav1.NewTime(r.now())
	phase, message, ev := attemptOutcome(t, parsed)

	a := &res.Attempts[len(res.Attempts)-1]
	exit := int32(parsed.ExitCode)
	a.Phase = phase
	a.ExitCode = &exit
	a.FinishedAt = &now
	a.Metrics = parsed.Metrics
	a.Message = message
	if podName != "" {
		a.PodName = podName
	}

	// Only ever grows the history it is given, so it must run after the last
	// write through `a` above and before anything can return.
	truncateAttempts(res)

	// The in-progress result carries the latest evidence; on settle it is
	// overwritten by the metrics of the attempt that decided the verdict.
	if len(parsed.Metrics) > 0 {
		res.Metrics = parsed.Metrics
	}

	// settle takes its evidence as arguments rather than closing over the
	// attempt's, because a segmented soak's verdict is read from the AGGREGATE
	// and not from the window that happened to be last. The three fields still
	// travel together and are still written unconditionally, which is the rule
	// that matters: a settle that is not a threshold failure passes a zero
	// attemptEvidence, which CLEARS any violations left by an earlier errored
	// attempt — a retry that then passes must not keep the gates its predecessor
	// missed.
	settle := func(final burninv1alpha1.RunPhase, msg string, metrics map[string]string, ev attemptEvidence) {
		res.Phase = final
		res.FinishedAt = &now
		res.Message = msg
		res.Metrics = metrics
		res.Violations = ev.violations
		res.NotEvaluated = ev.notEvaluated
		res.Unmeasurable = ev.unmeasurable
		recount(run)
		// One completion per execution unit: the nodes are part of the event
		// identity, or a second node's completion would dedupe away against
		// the first's. A pair is one unit and gets one key naming both ends.
		r.deliverTestCompleted(ctx, run, p.Sinks, res.Name+"/"+strings.Join(nodes, "+"))
	}

	switch phase {
	case burninv1alpha1.RunPassed:
		if res.SegmentsRequired > 0 {
			foldSegment(res, parsed)
			if done, final, msg, aggEv := segmentVerdict(t, res, parsed); done {
				if final == burninv1alpha1.RunPassed {
					// The one repeat a segmented test owes is the soak itself.
					res.RepeatsCompleted = res.RepeatsRequired
				}
				settle(final, msg, res.AggregatedMetrics, aggEv)
				return
			}
			res.Message = fmt.Sprintf("segment %d/%d complete", res.SegmentsCompleted, res.SegmentsRequired)
			break
		}
		res.RepeatsCompleted++
		if res.RepeatsCompleted >= res.RepeatsRequired {
			settle(burninv1alpha1.RunPassed, message, parsed.Metrics, ev)
			return
		}
		// More repeats owed. The test stays open and the next pass starts the
		// next execution; repeats are sequential because two concurrent copies
		// on one node would contend for the very resource under test.
		res.Message = fmt.Sprintf("attempt %d/%d passed", res.RepeatsCompleted, res.RepeatsRequired)

	case burninv1alpha1.RunFailed:
		// A SEGMENT THAT EXITS 1 SETTLES THE SOAK, at whatever hour it happened.
		// A fault observed at hour six is a fault; continuing to burn is not
		// evidence-gathering, it is hoping. Nothing folds it into the aggregate
		// either — the verdict is the runner's own assertion about the hardware,
		// and the metrics that explain it are this window's.
		settle(burninv1alpha1.RunFailed, message, parsed.Metrics, ev)

	case burninv1alpha1.RunSkipped:
		// The test does not apply to this hardware. Repeating it will not make
		// it start applying, retrying it will not either, and neither will
		// burning another 287 windows of it.
		settle(burninv1alpha1.RunSkipped, message, parsed.Metrics, ev)

	default:
		if res.ErrorRetries < retryOnErrorLimit(run) {
			res.ErrorRetries++
			// An errored attempt measured nothing, so it consumes neither a
			// repeat nor a segment: RepeatsCompleted and SegmentsCompleted are
			// both deliberately untouched here, which is what makes the retry
			// re-run THE SAME segment index and contribute nothing to the
			// aggregate.
			res.Message = fmt.Sprintf("attempt %d errored (%s); retrying", attempt, message)
			return
		}
		// AN ERROR AT THE END DOES NOT DISCARD THE SOAK.
		//
		// settle overwrites the result's evidence with the deciding attempt's,
		// which is right for a Pass and for a Fail: both are assertions about the
		// hardware, and the window that made the assertion is what explains it.
		// An Error asserts nothing about hardware. It says this attempt produced
		// no measurement — which is not a reason to throw away the attempts that
		// did.
		//
		// The loss is worst exactly where segmentation is most valuable. An
		// evicted or unschedulable pod prints nothing at all, so parsed.Metrics is
		// empty, and a six-day soak that errored in its last hour reported Error
		// with no metrics whatsoever. AggregatedMetrics still held the evidence,
		// but res.Metrics is the field consumers read and the envelope carries, so
		// six days of measurement was unreachable outside the object.
		//
		// The folded Unmeasurable set goes the same way and for the same reason: a
		// runner positively established that this part has no ECC to read, and an
		// eviction two windows later is not new information about that. Violations
		// and NotEvaluated are deliberately NOT carried across — a segmented soak
		// gates only at the end, so there are none to carry, and synthesising some
		// here would report gate outcomes for a test that never reached its gate.
		if res.SegmentsRequired > 0 && len(res.AggregatedMetrics) > 0 {
			settle(burninv1alpha1.RunError, message, res.AggregatedMetrics,
				attemptEvidence{unmeasurable: res.Unmeasurable})
			return
		}
		settle(burninv1alpha1.RunError, message, parsed.Metrics, ev)
	}
	recount(run)
}

// foldSegment records one CLEAN segment against the soak's running aggregate.
//
// Only a segment that exited 0 reaches here, and that is the rule: an errored
// segment measured nothing and a failed one is the verdict itself, so neither
// contributes. The aggregate is persisted on the result, which is what makes
// attempt truncation and a controller restart mid-soak safe.
func foldSegment(res *burninv1alpha1.TestResult, parsed runner.Result) {
	res.SegmentsCompleted++
	res.AggregatedMetrics = foldMetrics(res.AggregatedMetrics, parsed.Metrics)
	res.Unmeasurable = foldUnmeasurable(res.Unmeasurable, parsed.Unmeasurable, res.AggregatedMetrics)
	// A reader mid-soak wants the accumulated numbers, not the last window's:
	// the soak is the thing being measured, and the window is an implementation
	// detail of how it survives an eviction.
	if len(res.AggregatedMetrics) > 0 {
		res.Metrics = res.AggregatedMetrics
	}
}

// segmentVerdict says whether a segmented soak is over, and with what.
//
// It is the only caller of gateOutcome on the aggregate, and it is reached only
// from completeAttempt. Two ways a soak ends here and they are not the same
// thing: the LAST segment, where every gate is applied once to everything that
// was measured; and AbortEarly, where the aggregate already violates a gate no
// later segment could retract. Everything else keeps the test open.
func segmentVerdict(
	t plannedTest,
	res *burninv1alpha1.TestResult,
	parsed runner.Result,
) (bool, burninv1alpha1.RunPhase, string, attemptEvidence) {
	declared := unmeasurableSet(res.Unmeasurable)

	if res.SegmentsCompleted >= res.SegmentsRequired {
		phase, message, ev := gateOutcome(res.AggregatedMetrics, declared, t.Spec.Thresholds, parsed.Message)
		ev.unmeasurable = append([]string(nil), res.Unmeasurable...)
		// How much was actually burned in leads the message, because it is the
		// first thing anyone reading a soak's verdict wants and the only place
		// it is stated in words.
		prefix := fmt.Sprintf("%d segment(s) of %ds completed",
			res.SegmentsCompleted, segmentSeconds(&t.Spec))
		if rest := strings.TrimSpace(message); rest != "" {
			prefix += "; " + rest
		}
		return true, phase, prefix, ev
	}

	if !abortEarly(&t.Spec) {
		return false, "", "", attemptEvidence{}
	}
	out := verdict.Evaluate(res.AggregatedMetrics, declared, t.Spec.Thresholds)
	provable := provableViolations(res.AggregatedMetrics, out, t.Spec.Thresholds)
	if len(provable) == 0 {
		return false, "", "", attemptEvidence{}
	}

	ev := attemptEvidence{
		// The PROVABLE ones only. A violation a later segment could still
		// retract did not end this soak and must not be recorded as though it
		// had — a report naming a gate that was still moving would send an
		// engineer after a part the soak never actually condemned.
		violations:   violationsFor(verdict.Outcome{Violations: provable}),
		unmeasurable: append([]string(nil), res.Unmeasurable...),
	}
	message := fmt.Sprintf(
		"soak aborted after segment %d of %d: %s. The aggregate already violates a gate no later segment could "+
			"retract, so the remaining %d segment(s) were not burned",
		res.SegmentsCompleted, res.SegmentsRequired, provable[0].Reason,
		res.SegmentsRequired-res.SegmentsCompleted)
	if more := (verdict.Outcome{Violations: provable}).ViolationSummary(); more != "" {
		message = strings.TrimSpace(message + " [" + more + "]")
	}
	return true, burninv1alpha1.RunFailed, message, ev
}

// emptyHarvestMessage is what an execution that produced no runner output at
// all is recorded as. It says what actually happened — the runner measured
// nothing — rather than naming whichever threshold happened to be evaluated
// first, which is what made this failure mode so misleading to read.
// undeclaredSkipMessage explains an Error that arrived wearing the skip code.
//
// pkg/runner sets Result.UndeclaredSkip when a runner exits 2 without printing a
// skip declaration. The phase is already right — Error, hardware unjudged,
// retryable — and this is only about the record, which is kept for years after
// the Event that accompanied it has expired.
//
// Without it the stored message is the runner's last non-key=value line, and for
// a Go panic that is a stack frame: "/src/main.go:132 +0x1d". Someone reading the
// status a month later sees an Error explained by a file path and an instruction
// offset, with nothing saying the runner exited 2, never declared a skip, or
// most likely crashed. The flag was added for exactly this and nothing read it.
const undeclaredSkipMessage = "runner exited 2 (the skip code) without declaring a skip — it never " +
	"reported that this test does not apply to this hardware. An unrecovered Go panic exits 2, as does " +
	"every Go runtime fatal error (out of memory, concurrent map writes, stack exhaustion), so the " +
	"likeliest cause is a crashed runner: the hardware is UNJUDGED, not out of scope"

// undeclaredSkipBrief is the same statement compressed for a Pair summary.
//
// The Pair path does NOT go through attemptOutcome — pairSide.summary() builds
// its own line and pairMessage composes TWO of them into one verdict about the
// link — so the explanation has to be applied there separately, and at a length
// that survives being said twice in one message. It already carries the role,
// node, verdict and exit code, so this states only what those cannot.
const undeclaredSkipBrief = "exited 2 without declaring a skip — likely a crashed runner, hardware unjudged"

// explainUndeclaredSkip leads with the explanation and DEMOTES the runner's own
// last line to context rather than dropping it.
//
// For a panic that line is the tail of the trace, which is the only clue to what
// actually died and is worth keeping — it is simply not an explanation on its
// own. The bracketed form matches how this function already reports contract-
// rejected metric names, so a reader meets one shape rather than two.
func explainUndeclaredSkip(runnerSaid string) string {
	if s := strings.TrimSpace(runnerSaid); s != "" {
		return undeclaredSkipMessage + " [runner said: " + s + "]"
	}
	return undeclaredSkipMessage
}

const emptyHarvestMessage = "runner exited 0 but reported no metrics at all — it measured nothing, so the test's %d threshold(s) could not be evaluated; no verdict (a runner killed or evicted before it printed looks like this)"

// attemptOutcome converts one runner execution into a phase.
//
// Thresholds are evaluated HERE, once, against a completed execution — never
// against a checkpoint. A mid-run sample that dips below a bar is not a
// failure, because the run is not over.
//
// EXCEPT FOR A SEGMENTED SOAK, where they are not evaluated here at all. One
// segment is one window of a test, not a test, and gating a window would fail a
// week-long soak on fifteen minutes — while also making the answer depend on how
// finely the soak happened to be sliced, which is a scheduling decision and not
// a property of the hardware. completeAttempt applies them once, to the
// aggregate, through gateOutcome. Note that the EMPTY-HARVEST rule below still
// reads the test's real thresholds and still fires: a segment that exits 0
// having printed nothing measured nothing, and counting it as a completed
// segment would quietly shorten the soak by every interruption it suffered.
//
// This is the single choke point where a runner.Result becomes a phase, for
// Node and Pair scope alike, which is why the empty-harvest rule below lives
// here rather than beside the log-fetch rule in each caller. It is also why the
// returned violations are identical in shape for both scopes: a link measured
// from one end is not measured, and a link reported from one end is not
// reported either.
//
// The violations are non-empty only when thresholds decided the phase. A
// runner's own exit-1 is its assertion about the hardware, not a threshold
// verdict, so it carries none — there is no gate to point at.
func attemptOutcome(t plannedTest, parsed runner.Result) (burninv1alpha1.RunPhase, string, attemptEvidence) {
	var phase burninv1alpha1.RunPhase
	var ev attemptEvidence
	message := parsed.Message

	// The runner's own declarations survive whatever the thresholds decide.
	// "This part cannot produce this measurement" is a claim about the HARDWARE
	// and it is true of a passing run as much as a failing one, so it is recorded
	// on every phase rather than only where a gate happened to notice it.
	for name := range parsed.Unmeasurable {
		ev.unmeasurable = append(ev.unmeasurable, name)
	}
	sort.Strings(ev.unmeasurable)

	switch parsed.Verdict {
	case runner.VerdictPass:
		switch {
		case len(t.Spec.Thresholds) > 0 && parsed.ReportedNothing():
			// Exit 0 and not one key=value line, with an acceptance bar to
			// clear. The runner did not measure this hardware — a SIGTERM, an
			// eviction or a controller replacement mid-run all land exactly
			// here — so there is nothing for the thresholds to judge. Handing
			// an empty map to fail-closed evaluation manufactures a hardware
			// verdict out of an absence, and condemns a healthy node for the
			// machinery's fault. Error means "we do not know", and we do not.
			//
			// THIS DOES NOT RELAX FAIL-CLOSED, and it must never be widened
			// until it does. A runner that reported SOME metrics and omitted a
			// gated one still Fails below: it looked at the hardware, and a
			// measurement it owed and did not produce is exactly the silence
			// that must never satisfy acceptance. Only the TOTAL absence of
			// output qualifies, because that is not evidence of anything.
			//
			// Error is also the retryable phase, so retryOnErrorLimit now
			// re-runs such an execution instead of settling a hardware verdict
			// on the first interrupted attempt. That is the desired behaviour:
			// an interrupted test is precisely the thing worth running again,
			// and a genuine hardware fault will still be there next time.
			phase = burninv1alpha1.RunError
			message = fmt.Sprintf(emptyHarvestMessage, len(t.Spec.Thresholds))
			if parsed.Message != "" {
				// The last non-metric line, when there was one. A runner that
				// died saying "CUDA driver version is insufficient" has told us
				// the actual cause, and it must not be dropped on the floor.
				message = strings.TrimSpace(message + " [runner said: " + parsed.Message + "]")
			}

		default:
			// Exit 0 says the runner is content; the thresholds are the
			// profile's own acceptance bar. A segmented soak passes none here —
			// gateOutcome is called once, later, on the aggregate.
			gates := t.Spec.Thresholds
			if segmentSeconds(&t.Spec) > 0 {
				gates = nil
			}
			var gateEv attemptEvidence
			phase, message, gateEv = gateOutcome(parsed.Metrics, parsed.Unmeasurable, gates, parsed.Message)
			ev.violations = gateEv.violations
			ev.notEvaluated = gateEv.notEvaluated
		}

	case runner.VerdictFail:
		// Exit 1 is the runner's OWN assertion that this hardware failed, and
		// it is honoured whether or not any metric came with it. The empty
		// harvest above is a guard against a Fail this operator invented; it is
		// not licence to overturn one the runner reported. A runner that
		// detects a fault and dies before printing has still detected a fault,
		// and erasing that into a retryable Error would lose it.
		phase = burninv1alpha1.RunFailed
	case runner.VerdictSkip:
		// Exit 2 is likewise the runner's own declaration, made after looking
		// at the device: this test does not apply here. Zero metrics is the
		// NORMAL shape of a skip — there was nothing to measure — and no
		// threshold is evaluated on this path, so there is no invented Fail to
		// prevent. Applying the empty-harvest rule here would turn every clean
		// skip into a retried Error, which is the "skip must stay distinct"
		// invariant broken in a new direction.
		phase = burninv1alpha1.RunSkipped
	default:
		phase = burninv1alpha1.RunError
		if parsed.UndeclaredSkip {
			message = explainUndeclaredSkip(message)
		}
	}

	if len(parsed.InvalidNames) > 0 {
		// The COUNT is exact and the LIST is bounded. A Group execution
		// concatenates every rank's names, and pkg/runner adds one per offending
		// line — so an unbounded join grows as ranks x samples and lands in a
		// status written once per attempt per retry. Naming a handful is enough
		// to go and fix the runner; naming two thousand loses the verdict.
		shown := parsed.InvalidNames
		suffix := ""
		if len(shown) > maxNamedInvalidMetrics {
			shown = shown[:maxNamedInvalidMetrics]
			suffix = fmt.Sprintf(", and %d more", len(parsed.InvalidNames)-maxNamedInvalidMetrics)
		}
		message = strings.TrimSpace(message + fmt.Sprintf(
			" [runner emitted %d metric name(s) the contract rejects: %s%s]",
			len(parsed.InvalidNames), strings.Join(shown, ", "), suffix))
	}
	return phase, message, ev
}

// gateOutcome applies a set of thresholds to a set of metrics and returns the
// phase they decide, the message that explains it, and the structured evidence.
//
// It is the ONE place a threshold becomes a phase, and it has exactly two
// callers, both of which arrive through completeAttempt: attemptOutcome, with
// one execution's own metrics, and segmentVerdict, with a soak's accumulated
// aggregate. Keeping them on the same function is what guarantees that a
// segmented soak and an unsegmented one report a shortfall in the same words,
// with the same violations, the same not-evaluated notes and the same
// truncation — a consumer must not be able to tell how a test was scheduled
// from how its verdict reads.
//
// Evaluation is fail-closed: a threshold naming a metric the runner never
// emitted is a failure, not a pass. runnerSaid is the runner's own last line,
// kept as context rather than replaced.
func gateOutcome(
	metrics map[string]string,
	unmeasurable map[string]bool,
	thresholds []burninv1alpha1.Threshold,
	runnerSaid string,
) (burninv1alpha1.RunPhase, string, attemptEvidence) {
	var ev attemptEvidence
	phase := burninv1alpha1.RunPassed
	message := runnerSaid

	out := verdict.Evaluate(metrics, unmeasurable, thresholds)
	if !out.Passed {
		phase = burninv1alpha1.RunFailed
		// THE RUNNER'S OWN MESSAGE IS KEPT, not replaced. Overwriting it was
		// invisible at Node scope, where runnerSaid is one runner's last line —
		// but a Pair or Group result's message is CONSTRUCTED by this operator
		// and carries things nothing else says: the clause stating that the
		// verdict is about the LINK or the COLLECTIVE rather than about any one
		// node, each end's or each rank's own report, and the note that ranks
		// disagreed about a metric. An 8-rank group failing a bandwidth gate lost
		// all of it and stored a bare threshold line beside eight node names,
		// which is exactly the misattribution the lead clause exists to prevent.
		message = out.Message
		if prior := strings.TrimSpace(runnerSaid); prior != "" {
			message = strings.TrimSpace(message + " [" + prior + "]")
		}
		ev.violations = violationsFor(out)
		// Message names the first gate only, and that field is frozen. Without
		// this tail a node that missed three gates reads as a node that missed
		// one, and an engineer replaces one part per burn-in cycle. The summary
		// names the CAUSE of each, because a broken threshold and a thermal
		// ceiling arrive here as the same Failed test and only one is a reason to
		// walk to a rack.
		if more := out.ViolationSummary(); more != "" {
			message = strings.TrimSpace(message + " [" + more + "]")
		}
	}
	// A RequiredIfMeasurable gate the hardware cannot measure was not applied,
	// and the report has to say so. Appending it to the message puts it on the
	// TestResult and therefore in the delivered envelope: a Passed test whose ECC
	// gate never ran must never be indistinguishable from one whose ECC gate ran
	// and was satisfied.
	for _, n := range out.NotEvaluated {
		ev.notEvaluated = append(ev.notEvaluated, burninv1alpha1.NotEvaluated{
			Metric: n.Metric, Reason: n.Reason,
		})
	}
	if why := out.NotEvaluatedMessage(); why != "" {
		message = strings.TrimSpace(message + " [" + why + "]")
	}
	return phase, message, ev
}

// attemptEvidence is the structured half of an outcome — everything a consumer
// would otherwise have to parse out of Message, which names only the first
// violation and is frozen there.
type attemptEvidence struct {
	violations   []burninv1alpha1.Violation
	notEvaluated []burninv1alpha1.NotEvaluated
	unmeasurable []string
}

// violationsFor maps an evaluated outcome onto the API's own violation shape.
//
// The two types are mirrored rather than shared because pkg/verdict imports the
// API package, so importing it back would be a cycle. This function is the ONE
// place that crossing happens; keep it that way, or the API and the evaluator
// will drift and a violation will exist that an operator cannot see.
func violationsFor(out verdict.Outcome) []burninv1alpha1.Violation {
	if len(out.Violations) == 0 {
		return nil
	}
	violations := make([]burninv1alpha1.Violation, 0, len(out.Violations))
	for _, v := range out.Violations {
		violations = append(violations, burninv1alpha1.Violation{
			Index:  int32(v.Index),
			Metric: v.Metric,
			Cause:  string(v.Cause),
			Kind:   string(v.Kind),
			Reason: v.Reason,
		})
	}
	return violations
}

// triggerFor says why the next execution of an already-open result is
// happening. It is derived from the previous attempt's phase rather than
// tracked separately, so the recorded history cannot disagree with what
// actually caused the attempt.
func triggerFor(res *burninv1alpha1.TestResult) burninv1alpha1.AttemptTrigger {
	if len(res.Attempts) == 0 {
		return burninv1alpha1.AttemptInitial
	}
	if res.Attempts[len(res.Attempts)-1].Phase == burninv1alpha1.RunError {
		return burninv1alpha1.AttemptErrorRetry
	}
	// A soak's next window, not a repeat. The distinction is the point of the
	// field: the previous execution succeeded AND folded into the aggregate this
	// test's one verdict will be read from, where a repeat is the whole test run
	// again for a verdict of its own. Derived from SegmentsRequired, which is
	// pinned onto the result, so the history cannot disagree with the plan.
	if res.SegmentsRequired > 0 {
		return burninv1alpha1.AttemptSegment
	}
	return burninv1alpha1.AttemptRepeat
}

// nextAttempt is the attempt number the (test, node) execution is currently on,
// and whether that attempt has already been opened.
func nextAttempt(res *burninv1alpha1.TestResult) (int32, bool) {
	if res == nil || len(res.Attempts) == 0 {
		return 1, false
	}
	last := res.Attempts[len(res.Attempts)-1]
	if !last.Phase.IsTerminal() {
		return last.Attempt, true
	}
	return last.Attempt + 1, false
}

func podNameOf(pod *corev1.Pod) string {
	if pod == nil {
		return ""
	}
	return pod.Name
}

// ─── Checkpoints ──────────────────────────────────────────────────────────────

// checkpoint refreshes an in-flight execution's evidence from the runner's
// stdout so far, and reports whether it wrote anything.
//
// A CHECKPOINT IS EVIDENCE, NEVER A VERDICT. It does not look at an exit code,
// it does not evaluate a threshold, and it cannot move a phase. Its whole job
// is that a multi-hour soak which is cancelled, evicted or lost to a node
// reboot at minute 200 still says what the temperatures and clocks were doing,
// instead of reporting nothing at all because the metrics only reach the status
// when the pod terminates.
//
// It is safe to read a running pod's log precisely because the parser is
// last-occurrence-wins: a runner reporting progressively emits the same keys
// repeatedly, so a truncated log is a valid earlier snapshot rather than a
// corrupt record. The values overwrite in place and never accumulate, so a long
// soak cannot grow its status object without bound.
//
// A log read that fails is not a failure of anything: the next checkpoint, or
// the terminal harvest, carries the same cumulative state.
//
// IT MUST NEVER FOLD INTO TestResult.AggregatedMetrics, and the temptation is
// real because it already writes TestResult.Metrics. A checkpoint is a partial
// window: folding its elapsedS into a Sum would count the same seconds again
// when the segment finishes, and a soak would certify a duration it never ran.
// Only a segment that EXITED CLEANLY contributes, which is completeAttempt's
// business and not this function's.
func (r *BurnInRunReconciler) checkpoint(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	t plannedTest,
	node string,
	pod *corev1.Pod,
) bool {
	interval := checkpointInterval(&t.Spec)
	if interval <= 0 {
		return false
	}
	res := resultFor(run, t.Name, node)
	if res == nil || res.Phase.IsTerminal() {
		return false
	}

	// Due only once a full interval has passed since the last sample — or,
	// for the first one, since the execution actually started.
	since := res.LastCheckpointAt
	if since == nil {
		since = res.StartedAt
	}
	if since != nil && r.now().Sub(since.Time) < interval {
		return false
	}

	stdout, err := r.fetchLogs(ctx, pod)
	if err != nil {
		return false
	}
	// The exit code is deliberately not consulted: the pod has not exited, and
	// only the parsed metrics are wanted. Passing 0 here selects no behaviour
	// — attemptOutcome, which is what turns a verdict into a phase, is not on
	// this path at all.
	parsed := runner.Parse(string(t.Spec.Kind), stdout, 0)
	if len(parsed.Metrics) == 0 {
		// Nothing to publish yet. Not stamping LastCheckpointAt means the next
		// pass tries again rather than waiting out another whole interval for
		// a runner that is simply slow to emit its first line.
		return false
	}

	now := metav1.NewTime(r.now())

	// A CHECKPOINT MUST NOT PUBLISH OVER THE FOLD.
	//
	// foldSegment maintains res.Metrics = res.AggregatedMetrics after every clean
	// segment, because a reader mid-soak wants the accumulated numbers rather
	// than the last window's. Overwriting that with ONE partial window erased the
	// soak so far from the only metric field anything outside this package reads:
	// the delivery envelope carries res.Metrics and not AggregatedMetrics, and so
	// do all four renderers and the Prometheus exposition.
	//
	// The damage was silent and in the certifying direction. A soak that had
	// peaked at 89 C over 1800 seconds, checkpointed 300 seconds into a cooler
	// window, published 70 C and 300 s — under a `segments` block still saying
	// two windows had completed, which docs/sinks.md tells consumers to read as a
	// fold. The progress record that is the whole reason to checkpoint a
	// multi-day soak sawtoothed instead of advancing.
	//
	// So the published view is the fold COMBINED WITH the window in flight, which
	// is the honest answer to "how is this soak going": foldMetrics applies each
	// metric's own rule, so elapsedS sums and gpuTempC takes the peak.
	// AggregatedMetrics itself is deliberately NOT touched — a checkpoint is not
	// a completed segment, and folding it in would count the same seconds again
	// when the segment finishes, which TestSoak_ACheckpointNeverFoldsIntoTheAggregate
	// pins.
	published := parsed.Metrics
	if res.SegmentsRequired > 0 && len(res.AggregatedMetrics) > 0 {
		published = foldMetrics(res.AggregatedMetrics, parsed.Metrics)
	}
	res.Metrics = published
	res.LastCheckpointAt = &now
	if n := len(res.Attempts); n > 0 && !res.Attempts[n-1].Phase.IsTerminal() {
		// The ATTEMPT record keeps this window's own reading. An attempt is one
		// pod, and the fold is not a property of it.
		res.Attempts[n-1].Metrics = parsed.Metrics
		res.Attempts[n-1].LastCheckpointAt = &now
	}
	return true
}

// checkpointSequence numbers a run's checkpoints.
//
// It is the index of the interval window the checkpoint falls in, counted from
// the run's start, and it is DERIVED rather than counted so that nothing has to
// persist a counter. That matters because the sequence feeds the envelope's
// DeliveryID: a key that moved on every attempt would mint a new identity per
// retry and defeat the receiver's dedupe, turning a flaky endpoint into a flood
// of near-identical records.
func (r *BurnInRunReconciler) checkpointSequence(run *burninv1alpha1.BurnInRun, p *plan) int {
	interval := p.checkpointInterval()
	if interval <= 0 || run.Status.StartedAt == nil {
		return 0
	}
	elapsed := r.now().Sub(run.Status.StartedAt.Time)
	if elapsed <= 0 {
		return 0
	}
	return int(elapsed / interval)
}

// ─── Suspension, cancellation, deadline ───────────────────────────────────────

// cancel stops the run for good and drives it to Cancelled.
//
// Cancelled is terminal and is NOT a verdict. Tests that finished before the
// cancel keep their real results; tests that had not run are settled as
// Cancelled rather than Failed, because nobody measured them and a consumer
// must be able to tell "I stopped this" from "this hardware is bad".
//
// The Graceful policy drains: no new executions start, and the ones already in
// flight are allowed to finish and record. An in-flight burn-in is expensive
// evidence — a soak four hours in has already cost the fleet those four hours,
// and discarding it means paying them again. Immediate deletes the pods now,
// which is what a facility power or cooling event calls for, since there the
// load itself is the reason for stopping.
func (r *BurnInRunReconciler) cancel(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	// Cancellation is one-way. Record that it was observed before doing
	// anything irreversible, so clearing spec.cancel mid-drain cannot resume a
	// run that has already begun releasing what it was holding.
	if run.Annotations[cancellingAnnotation] == "" {
		reason := run.Spec.CancelReason
		if reason == "" {
			reason = "cancelled with no reason given"
		}
		if run.Annotations == nil {
			run.Annotations = map[string]string{}
		}
		run.Annotations[cancellingAnnotation] = reason
		if err := r.updateKeepingStatus(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}

	message := "run cancelled: " + run.Annotations[cancellingAnnotation]

	// Harvest under BOTH policies before doing anything else. An execution
	// whose pod has already terminated is not "in flight" under any reading of
	// the word — its exit code and its metrics are sitting there — and throwing
	// that away because the cancel arrived a moment later would discard evidence
	// the fleet has already paid for. Immediate is about not waiting; it is not
	// about refusing to read.
	pass, err := r.execute(ctx, run, p, false)
	if err != nil {
		return ctrl.Result{}, err
	}
	if run.Spec.CancelPolicy != burninv1alpha1.CancelImmediate {
		if pass.running {
			// Still draining. Persist what was harvested and come back; the
			// run stays Running until the last in-flight execution reports.
			return r.persistPass(ctx, run, p, pass)
		}
		if pass.harvested || pass.dirty {
			if err := r.Status().Update(ctx, run); err != nil {
				return ctrl.Result{}, err
			}
			if pass.checkpointed {
				r.deliverCheckpoint(ctx, run, p.Sinks, r.checkpointSequence(run, p))
			}
			return ctrl.Result{Requeue: true}, nil
		}
	}

	r.settleRemaining(run, p, burninv1alpha1.RunCancelled, message)
	return r.terminate(ctx, run, p.Sinks, burninv1alpha1.RunCancelled)
}

// deadlineExceeded reports whether the run has outlived spec.deadlineSeconds,
// measured from status.startedAt across every test, node, repeat and retry.
func (r *BurnInRunReconciler) deadlineExceeded(run *burninv1alpha1.BurnInRun) bool {
	d := runDeadline(run)
	if d <= 0 || run.Status.StartedAt == nil {
		return false
	}
	return r.now().Sub(run.Status.StartedAt.Time) > d
}

// expire ends a run that ran out of time.
//
// The terminal phase is Error, and the two things it is not are the point.
// It is not Failed, because a run that ran out of time did not judge the
// hardware it never reached, and reporting that as a hardware verdict condemns
// parts nobody tested. It is not Cancelled, because nobody asked — Cancelled is
// reserved for an explicit request, so a consumer can tell an operator's
// decision from a bound being hit.
func (r *BurnInRunReconciler) expire(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	message := fmt.Sprintf("run deadline of %ds expired before this execution completed — the hardware was not judged",
		*run.Spec.DeadlineSeconds)
	r.settleRemaining(run, p, burninv1alpha1.RunError, message)
	return r.terminate(ctx, run, p.Sinks, burninv1alpha1.RunError)
}

// settleRemaining gives every execution that never reached a verdict an
// explicit one, so the status says which hardware was left unjudged instead of
// leaving a reader to infer it from an absence.
//
// It never touches a settled result: a deadline or a cancel stops a run, it
// does not retract evidence already gathered.
//
// These results are not delivered individually. They are not completions, and
// the terminal envelope that follows carries all of them at once.
func (r *BurnInRunReconciler) settleRemaining(
	run *burninv1alpha1.BurnInRun,
	p *plan,
	phase burninv1alpha1.RunPhase,
	message string,
) {
	now := metav1.NewTime(r.now())
	for i := range p.Tests {
		t := p.Tests[i]
		// An unexecutable scope is settled once for the whole test, with no
		// node attached; do not then add a per-node result beside it.
		if existing := resultFor(run, t.Name, ""); existing != nil && len(existing.Nodes) == 0 {
			continue
		}
		for _, nodes := range executionUnits(p, t) {
			res := ensureResult(run, t, nodes)
			if res.Phase.IsTerminal() {
				continue
			}
			res.Phase = phase
			res.FinishedAt = &now
			res.Message = message
			if n := len(res.Attempts); n > 0 && !res.Attempts[n-1].Phase.IsTerminal() {
				res.Attempts[n-1].Phase = phase
				res.Attempts[n-1].FinishedAt = &now
				res.Attempts[n-1].Message = message
			}
		}
	}
	recount(run)
}

// ─── Finalize and terminal handling ───────────────────────────────────────────

// finalize derives the run's terminal phase from its results.
//
// Precedence: Failed beats Error beats Passed. A required test that FAILED is
// a hardware verdict and wins outright; a required test that ERRORED leaves
// the hardware unjudged, which must not be called Passed. Skips count toward
// neither — a test the hardware cannot take is not evidence either way.
func (r *BurnInRunReconciler) finalize(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	phase := burninv1alpha1.RunPassed
	if hasRequiredError(run, p) {
		phase = burninv1alpha1.RunError
	}
	if hasRequiredFailure(run, p) {
		phase = burninv1alpha1.RunFailed
	}
	return r.terminate(ctx, run, p.Sinks, phase)
}

// finalizeError terminates a run the operator cannot execute at all.
func (r *BurnInRunReconciler) finalizeError(ctx context.Context, run *burninv1alpha1.BurnInRun, sinks []string, cause error) (ctrl.Result, error) {
	log.FromContext(ctx).Error(cause, "run cannot be executed")
	// Kind is required on every real BurnInTest, so an empty Kind marks this
	// result as synthetic — a real test named "resolve" cannot shadow it.
	now := metav1.NewTime(r.now())
	run.Status.Results = append(run.Status.Results, burninv1alpha1.TestResult{
		Name:       "resolve",
		Phase:      burninv1alpha1.RunError,
		FinishedAt: &now,
		Message:    cause.Error(),
	})
	return r.terminate(ctx, run, sinks, burninv1alpha1.RunError)
}

// terminate writes the terminal phase, stops the hardware being burned,
// releases every node the run was holding, and delivers the terminal envelope
// with its own retry loop.
func (r *BurnInRunReconciler) terminate(ctx context.Context, run *burninv1alpha1.BurnInRun, sinks []string, phase burninv1alpha1.RunPhase) (ctrl.Result, error) {
	run.Status.Phase = phase
	finished := metav1.NewTime(r.now())
	run.Status.FinishedAt = &finished

	// An execution still open at termination judged nothing. Saying so is the
	// difference between a status that reads as unfinished forever and one
	// that says which hardware was left unmeasured.
	for i := range run.Status.Results {
		res := &run.Status.Results[i]
		if res.Phase.IsTerminal() {
			continue
		}
		res.Phase = burninv1alpha1.RunError
		res.FinishedAt = &finished
		res.Message = "the run reached a terminal phase before this execution completed — no verdict"
	}
	recount(run)

	if err := r.Status().Update(ctx, run); err != nil {
		return ctrl.Result{}, err
	}

	// THE VERDICT IS DURABLE. Only now is it safe to stop the hardware being
	// burned: fail-fast, a resolution error and an immediate cancel can all
	// leave pods in flight, and a terminal run must not keep loading a node.
	// Terminated pods are kept until the run's own TTL for post-mortem logs.
	//
	// THE ORDER IS THE FIX FOR ISSUE #84 AND MUST NOT BE PUT BACK. Killing the
	// pods first meant that when either write above lost the resourceVersion
	// race — which is the conflict that appears in the report alongside the
	// kills — the reconcile returned an error, the terminal status was
	// discarded, and the run went back to believing it was Running with an open
	// execution. Its next pass then found the pod IT had just killed, harvested
	// the SIGKILL, and recorded the hardware as `Error: runner terminated
	// abnormally (exit 137)` — the operator condemning a healthy node for its
	// own act, 30 s into a 100 s soak. With the write first, a lost race costs
	// one wasted pass and nothing else, because nothing has been destroyed.
	//
	// A failure here leaves live pods behind a terminal run; reconcileTerminal
	// sweeps them, so the cleanup stays level-based rather than depending on
	// this one call succeeding.
	if err := r.deleteLivePods(ctx, run); err != nil {
		return ctrl.Result{}, err
	}

	// THE LOAD IS OFF. Only now is it safe to hand the nodes back, and that
	// order is the fix for issue #246.
	//
	// Releasing first put a node running a full-power burn-in back into the
	// scheduler with this run's own stamp already removed, so nothing in the
	// cluster said it was loaded — for the length of a terminal delivery, up to
	// the 15s delivery timeout against a slow sink. Production pods could be
	// placed onto a machine at maximum thermal and electrical load, and because
	// releaseCordons also drops AnnotationCordonOwner, a SECOND BurnInRun could
	// admit, cordon and load the same node inside the window, defeating
	// maxConcurrentNodes as a facility interlock.
	//
	// It was worst on the path the feature exists for: CancelImmediate is
	// documented as the tool for a facility power or cooling event, where the
	// load itself is the reason for stopping — and it returned the nodes to
	// service before stopping the load.
	//
	// Going last costs nothing. The release is level-based: reconcileTerminal
	// repeats it on every pass while the finalizer is on, so a deleteLivePods
	// failure returning early above leaves the cordon HELD, which is the safe
	// direction and converges on the next pass. reconcileTerminal has always
	// used this order; terminate was the one place inverted.
	released, err := r.releaseCordons(ctx, run)
	if err != nil {
		return ctrl.Result{}, err
	}
	if released {
		// releaseCordons clears status.cordonedNodes, and that has to reach the
		// apiserver or an operator watching the run still believes nodes are
		// held. Previously the release rode along on the status write above;
		// now that it happens after, it gets its own — the same shape
		// reconcileTerminal uses.
		if err := r.Status().Update(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}

	// Delivery LAST of the three, because it is network I/O to somebody else's
	// endpoint and nothing about the fleet's safety may wait on it.
	delivered := r.deliverPhase(ctx, run, sinks, phase)
	if !delivered {
		// The terminal envelope is the one delivery with no later transition
		// to carry it; mark it pending so the terminal loop keeps retrying.
		if run.Annotations == nil {
			run.Annotations = map[string]string{}
		}
		run.Annotations[pendingDeliveryAnnotation] = string(phase)
		if err := r.updateKeepingStatus(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}

	// The finalizer comes off LAST, because it is the standing record that this
	// run still owes the cluster something — a cordon to give back, and a pod to
	// stop. Removing it above, alongside the pending-delivery annotation, meant
	// that a failed pod deletion left a runner burning a node behind a finished
	// run with nothing left to reap it: reconcileTerminal's sweep is gated on
	// this finalizer, so dropping it early is dropping the only thing that would
	// have come back.
	if controllerutil.RemoveFinalizer(run, burninv1alpha1.FinalizerCordonCleanup) {
		if err := r.updateKeepingStatus(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}
	if !delivered {
		return ctrl.Result{RequeueAfter: terminalDeliveryRetryInterval}, nil
	}
	return r.handleTTL(ctx, run)
}

// updateKeepingStatus writes an object's METADATA without letting the write
// clobber the status the caller has just computed.
//
// Update() refreshes the object in place from the server's response, and the
// server still holds the PRE-terminal status — so a bare Update silently
// reverts everything the caller wrote into run.Status and the subsequent
// Status().Update persists the reversion. This exists so that trap is written
// down once instead of being re-derived at each of the four call sites.
func (r *BurnInRunReconciler) updateKeepingStatus(ctx context.Context, run *burninv1alpha1.BurnInRun) error {
	saved := run.Status
	if err := r.Update(ctx, run); err != nil {
		return err
	}
	run.Status = saved
	return nil
}

// reconcileTerminal serves an already-finished run.
//
// A terminal phase is FINAL: nothing here writes status.phase, so a cancel
// request, a spec edit or a late pod event arriving after the verdict cannot
// change it. What it does do is converge the side effects — release any cordon
// still held (the recovery path when a manager died between the terminal write
// and the release), retry a pending terminal delivery, and let the TTL reap it.
func (r *BurnInRunReconciler) reconcileTerminal(ctx context.Context, run *burninv1alpha1.BurnInRun) (ctrl.Result, error) {
	// The finalizer is removed in the same write that follows a successful
	// release, so its absence is proof there is nothing left to give back.
	// Checking it here keeps a terminal run that is merely waiting out its TTL
	// from listing every node in the cluster on each poll.
	if controllerutil.ContainsFinalizer(run, burninv1alpha1.FinalizerCordonCleanup) {
		// A terminal run must not still be loading a node. terminate deletes
		// its live pods AFTER the verdict is durable, so a conflict, a restart
		// or an apiserver blip between those two steps can leave one behind;
		// converging here rather than trusting that one call is what keeps the
		// cleanup level-based. It costs nothing on the ordinary path, where
		// there is nothing live to find.
		if err := r.deleteLivePods(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
		released, err := r.releaseCordons(ctx, run)
		if err != nil {
			return ctrl.Result{}, err
		}
		if released {
			if err := r.Status().Update(ctx, run); err != nil {
				return ctrl.Result{}, err
			}
		}
		controllerutil.RemoveFinalizer(run, burninv1alpha1.FinalizerCordonCleanup)
		if err := r.updateKeepingStatus(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
	}

	if phase, pending := run.Annotations[pendingDeliveryAnnotation]; pending {
		p, pinned, err := loadPlan(run)
		if err != nil || !pinned {
			// No plan means no sinks were ever known; nothing to deliver.
			delete(run.Annotations, pendingDeliveryAnnotation)
			if updErr := r.Update(ctx, run); updErr != nil {
				return ctrl.Result{}, updErr
			}
			return r.handleTTL(ctx, run)
		}
		if r.deliverPhase(ctx, run, p.Sinks, burninv1alpha1.RunPhase(phase)) {
			delete(run.Annotations, pendingDeliveryAnnotation)
			if err := r.Update(ctx, run); err != nil {
				return ctrl.Result{}, err
			}
			return r.handleTTL(ctx, run)
		}
		return ctrl.Result{RequeueAfter: terminalDeliveryRetryInterval}, nil
	}
	return r.handleTTL(ctx, run)
}

// reconcileDeleted releases what a deleted run was holding and then lets the
// deletion proceed.
//
// `kubectl delete burninrun` is the most natural reaction to a run behaving
// badly, and therefore the likeliest moment to strand a node: without the
// finalizer, deleting the run removes the only object that knows which nodes
// are being held, and the fleet quietly loses them. Deletion blocks here until
// the last owned cordon is released.
func (r *BurnInRunReconciler) reconcileDeleted(ctx context.Context, run *burninv1alpha1.BurnInRun) (ctrl.Result, error) {
	if !controllerutil.ContainsFinalizer(run, burninv1alpha1.FinalizerCordonCleanup) {
		return ctrl.Result{}, nil
	}
	if err := r.deleteLivePods(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	if _, err := r.releaseCordons(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	// The exported series are a view of a live object and must not outlive it.
	r.forgetMetrics(run.Namespace, run.Name)

	controllerutil.RemoveFinalizer(run, burninv1alpha1.FinalizerCordonCleanup)
	return ctrl.Result{}, client.IgnoreNotFound(r.Update(ctx, run))
}

// holdingNodes are the nodes this run is holding BETWEEN pods.
//
// A result that has not settled owes more work on its nodes, whether the next
// pod is the next segment of a soak, the next repeat, or a retry after an error.
// The node is not free in the gap and must not be offered to another target: the
// run is mid-measurement on it, and the pause is an artefact of how the
// measurement is scheduled rather than a statement that the hardware is idle.
//
// Every node of the result, not just the first: a Pair holds both of its nodes
// for the whole test and a Group holds all of its ranks, which is the same rule
// admission already applies when the unit starts.
//
// Read from status rather than from pods on purpose. Status is what survives a
// controller restart, and it is the same record `recount` and the deadline sweep
// read, so the interlock cannot disagree with them about which executions are
// still open.
func holdingNodes(run *burninv1alpha1.BurnInRun) map[string]bool {
	held := map[string]bool{}
	for i := range run.Status.Results {
		res := &run.Status.Results[i]
		if res.Phase.IsTerminal() {
			continue
		}
		for _, n := range res.Nodes {
			if n != "" {
				held[n] = true
			}
		}
	}
	return held
}

// busyNodes is the set of nodes this run currently holds under test load. It is
// the live input to the MaxConcurrentNodes interlock, counted from the pods
// that actually exist so that a controller restart cannot lose track of what is
// already running and fan out on top of it.
func (r *BurnInRunReconciler) busyNodes(ctx context.Context, run *burninv1alpha1.BurnInRun) (map[string]bool, bool, error) {
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(run.Namespace), client.MatchingLabels{labelRun: run.Name}); err != nil {
		return nil, false, err
	}
	busy := map[string]bool{}
	// awaitingStart is derived HERE because this is the one place that already
	// has every live pod in hand — the sweep below sees them one execution at a
	// time and would have to be told, at each of a dozen return sites, whether
	// the pod it holds has started. Deriving it once from the same list costs
	// nothing and cannot drift from what the sweep does.
	awaitingStart := false
	for i := range pods.Items {
		pod := &pods.Items[i]
		if !ownedBy(pod, run) || !podLive(pod) {
			continue
		}
		busy[pod.Labels[labelNode]] = true
		if !podStarted(pod) {
			awaitingStart = true
		}
	}
	return busy, awaitingStart, nil
}

// deleteLivePods removes every pod of this run that has not terminated.
//
// Terminated pods are left alone: their logs are the post-mortem, and they are
// reaped with the run at its TTL. A live pod behind a stopped run is different
// in kind — it is load on a node the run no longer accounts for, on hardware it
// may already have handed back to the scheduler.
func (r *BurnInRunReconciler) deleteLivePods(ctx context.Context, run *burninv1alpha1.BurnInRun) error {
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(run.Namespace), client.MatchingLabels{labelRun: run.Name}); err != nil {
		return err
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if !ownedBy(pod, run) || !podLive(pod) {
			continue
		}
		if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

// handleTTL garbage-collects a finished run after TTLSecondsAfterFinished.
// Zero or unset keeps the run (and its verdict) indefinitely.
func (r *BurnInRunReconciler) handleTTL(ctx context.Context, run *burninv1alpha1.BurnInRun) (ctrl.Result, error) {
	ttl := run.Spec.TTLSecondsAfterFinished
	if ttl == nil || *ttl <= 0 || run.Status.FinishedAt == nil {
		return ctrl.Result{}, nil
	}
	expiry := run.Status.FinishedAt.Add(time.Duration(*ttl) * time.Second)
	if remaining := expiry.Sub(r.now()); remaining > 0 {
		return ctrl.Result{RequeueAfter: remaining}, nil
	}
	// The run is about to stop existing, so its series must stop existing too.
	r.forgetMetrics(run.Namespace, run.Name)
	// Owned pods go with the run via ownerRefs.
	return ctrl.Result{}, client.IgnoreNotFound(r.Delete(ctx, run))
}

// ─── Result bookkeeping ───────────────────────────────────────────────────────

// resultFor returns the result for (test, node), or nil.
//
// An empty node matches any result with that test name, which is how the
// node-less results (an unsupported scope, a resolution error) are found.
func resultFor(run *burninv1alpha1.BurnInRun, testName, node string) *burninv1alpha1.TestResult {
	for i := range run.Status.Results {
		res := &run.Status.Results[i]
		if res.Name != testName {
			continue
		}
		if node == "" {
			return res
		}
		for _, n := range res.Nodes {
			if n == node {
				return res
			}
		}
	}
	return nil
}

// executionUnits enumerates the independent executions a test owes.
//
// At Node scope that is one per target node. At Pair scope it is exactly ONE
// unit covering both nodes, because a point-to-point measurement is a property
// of the link between them and cannot be attributed to either end. At Group
// scope it is likewise ONE unit, covering every target: a collective is a
// property of the whole group, and splitting it per node would produce N claims
// no single node can support.
//
// Everything that walks a test's work — the pass sweep, the deadline settle, the
// cancel settle — goes through here so none of them can disagree about how many
// verdicts a test owes.
func executionUnits(p *plan, t plannedTest) [][]string {
	switch t.Spec.Scope {
	case burninv1alpha1.ScopePair, burninv1alpha1.ScopeGroup:
		return [][]string{append([]string(nil), p.Targets...)}
	}
	out := make([][]string, 0, len(p.Targets))
	for _, node := range p.Targets {
		out = append(out, []string{node})
	}
	return out
}

// ensureResult returns the open result for one execution unit, creating it if
// needed. It is looked up by the unit's first node, which is the whole node at
// Node scope and the server at Pair scope.
//
// RepeatsRequired is pinned onto the result at creation from the plan's copy of
// the spec, so a verdict stays readable — "3 of 3 executions passed" — after
// the BurnInTest it came from has been edited or deleted.
func ensureResult(run *burninv1alpha1.BurnInRun, t plannedTest, nodes []string) *burninv1alpha1.TestResult {
	if res := resultFor(run, t.Name, nodes[0]); res != nil {
		return res
	}
	run.Status.Results = append(run.Status.Results, burninv1alpha1.TestResult{
		Name:            t.Name,
		Kind:            t.Spec.Kind,
		Scope:           t.Spec.Scope,
		Phase:           burninv1alpha1.RunRunning,
		Nodes:           append([]string(nil), nodes...),
		VariantAxes:     copyAxes(t.Axes),
		RepeatsRequired: repeatsRequired(&t.Spec),
		// Pinned from the plan's copy of the spec for the same reason as
		// RepeatsRequired, and additionally because it is what the reconciler
		// branches on: a positive value here is what makes this result's verdict
		// a statement about the aggregate rather than about one attempt. Zero
		// for every test that does not set spec.soak, which is the behaviour
		// every result had before segments existed.
		SegmentsRequired: segmentsRequired(&t.Spec),
		SegmentSeconds:   segmentSeconds(&t.Spec),
	})
	return &run.Status.Results[len(run.Status.Results)-1]
}

// appendSettled records a result that is complete the moment it is created and
// delivers its completion.
func (r *BurnInRunReconciler) appendSettled(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, result burninv1alpha1.TestResult) {
	run.Status.Results = append(run.Status.Results, result)
	recount(run)
	key := result.Name
	if len(result.Nodes) > 0 {
		key = result.Name + "/" + strings.Join(result.Nodes, "+")
	}
	r.deliverTestCompleted(ctx, run, p.Sinks, key)
}

// recount tallies per-(test,node) executions by phase.
//
// The four counters are kept apart on purpose. Errored must never be folded
// into Failed — a summary reading "0 failed" beside eight tests that never ran
// is a clean sweep of hardware nothing measured — and Skipped must never be
// either, since a node that cannot take a test has not failed it. Results still
// in flight count towards nothing.
func recount(run *burninv1alpha1.BurnInRun) {
	var passed, failed, errored, skipped int32
	for _, res := range run.Status.Results {
		switch res.Phase {
		case burninv1alpha1.RunPassed:
			passed++
		case burninv1alpha1.RunFailed:
			failed++
		case burninv1alpha1.RunError:
			errored++
		case burninv1alpha1.RunSkipped:
			skipped++
		}
	}
	run.Status.Passed, run.Status.Failed = passed, failed
	run.Status.Errored, run.Status.Skipped = errored, skipped
}

func hasRequiredFailure(run *burninv1alpha1.BurnInRun, p *plan) bool {
	req := p.requiredByName()
	for _, res := range run.Status.Results {
		if res.Phase == burninv1alpha1.RunFailed && req[res.Name] {
			return true
		}
	}
	return false
}

func hasRequiredError(run *burninv1alpha1.BurnInRun, p *plan) bool {
	req := p.requiredByName()
	for _, res := range run.Status.Results {
		if res.Phase != burninv1alpha1.RunError {
			continue
		}
		// The synthetic resolve result is marked by its empty Kind — every
		// real BurnInTest has one (the field is required) — so a real test
		// that happens to be named "resolve" gates on its own required flag.
		if res.Name == "resolve" && res.Kind == "" {
			return true
		}
		if req[res.Name] {
			return true
		}
	}
	return false
}

// isTerminal reports whether a RUN phase is final. It deliberately omits
// Skipped, which a run never takes; RunPhase.IsTerminal covers the TestResult
// case where it does.
func isTerminal(p burninv1alpha1.RunPhase) bool {
	switch p {
	case burninv1alpha1.RunPassed, burninv1alpha1.RunFailed, burninv1alpha1.RunError, burninv1alpha1.RunCancelled:
		return true
	}
	return false
}

// SetupWithManager wires the reconciler to BurnInRun events and to the pods
// it owns, so a test pod terminating re-triggers the run's reconcile.
func (r *BurnInRunReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&burninv1alpha1.BurnInRun{}).
		Owns(&corev1.Pod{}).
		Named("burninrun").
		Complete(r)
}

// maxPodDetail bounds the kubelet message this operator copies into a
// TestResult. The kubelet does not bound Terminated.Message, and a status write
// the apiserver refuses would lose the whole verdict rather than the tail of one
// sentence — so it is truncated visibly rather than trusted.
const maxPodDetail = 512

// maxNamedInvalidMetrics bounds how many rejected metric names a message spells
// out. The count is always exact.
const maxNamedInvalidMetrics = 8

// runnerChoseExitCode reports whether an exit code is one the runner contract
// defines, i.e. whether the runner picked it.
//
// 0 pass, 1 fail, 2 not applicable, 3 the runner's own error — see pkg/runner's
// VerdictFor, which maps everything else onto Error. Anything outside that range
// came from somewhere other than the runner: a signal (137, 143), a container
// that never started (128), a wrapped tool's own code (dcgmi's 226), or -1 for a
// pod with no terminated runner container at all. In those cases the runner did
// not report an outcome, whatever it may have printed first.
func runnerChoseExitCode(exitCode int) bool {
	return exitCode >= 0 && exitCode <= 3
}

func clampPodDetail(s string) string {
	// truncateAtRune, not a byte slice: the kubelet's message is whatever the
	// runtime produced and may be any encoding at all, and this value goes
	// straight into a stored status and a delivered envelope.
	return truncateAtRune(strings.Join(strings.Fields(s), " "), maxPodDetail, "… (truncated)")
}

// killPods destroys pods whose fate is already recorded durably.
//
// A TERMINATED pod is left alone: it is drawing no power, and its logs are the
// post-mortem. That rule came from deletePods and deletePairPods, which this
// replaced — losing it would delete the evidence for the very Error just
// recorded.
//
// NotFound is success: the pod may have finished on its own between the
// decision and the write, and that is the ordinary case rather than an error.
func (r *BurnInRunReconciler) killPods(ctx context.Context, pods []*corev1.Pod) error {
	for _, pod := range pods {
		if pod == nil || !podLive(pod) {
			continue
		}
		if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

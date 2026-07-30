// Package controller holds the reconcilers for the Glimmer Burn-In Operator.
package controller

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// BurnInRunReconciler drives a BurnInRun from Pending to a terminal verdict:
// resolve the profile ONCE into a pinned plan, capture the target fingerprint,
// execute each test's pod on each target node, evaluate thresholds against
// parsed metrics, and export the verdict to the plan's sinks.
//
// v1 executes Node-scope tests. Pair/Group tests are recorded as Error rather
// than silently skipped: a required acceptance test the operator cannot run
// must never let hardware pass by omission.
type BurnInRunReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// PodLogs fetches a completed pod's stdout — the runner's metrics channel.
	// controller-runtime's client cannot read subresource logs, so the manager
	// wires this from a plain clientset; tests stub it.
	PodLogs func(ctx context.Context, namespace, name string) (string, error)

	// Now is the clock, swappable in tests. Nil means time.Now.
	Now func() time.Time
}

func (r *BurnInRunReconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

// resolvedTest is one entry of a profile with its spec materialised.
type resolvedTest struct {
	name     string
	spec     burninv1alpha1.BurnInTestSpec
	required bool
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
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get

// Reconcile is the entrypoint: level-based, so every pass re-derives where the
// run stands from its pinned plan, its status, and its pods.
func (r *BurnInRunReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	var run burninv1alpha1.BurnInRun
	if err := r.Get(ctx, req.NamespacedName, &run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
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

	p, err := buildPlan(profile, tests, targets)
	if err != nil {
		return r.finalizeError(ctx, run, profileSinks(profile), err)
	}
	if err := pinPlan(run, p); err != nil {
		return r.finalizeError(ctx, run, p.Sinks, err)
	}
	if err := r.Update(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	return r.markRunning(ctx, run, p)
}

func (r *BurnInRunReconciler) markRunning(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	run.Status.Phase = burninv1alpha1.RunRunning
	started := metav1.NewTime(r.now())
	run.Status.StartedAt = &started
	run.Status.Fingerprint = r.captureFingerprint(ctx, p.Targets)
	run.Status.ObservedGeneration = run.Generation

	// Deliver before writing: a crash in between redelivers with the same
	// derived DeliveryID, which receivers dedupe. The reverse order would
	// drop the transition entirely.
	r.deliver(ctx, run, p.Sinks, contract.ReasonPhaseChanged, string(burninv1alpha1.RunRunning))
	if err := r.Status().Update(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{Requeue: true}, nil
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
			tests = append(tests, resolvedTest{name: t.Name, spec: t.Spec, required: required})
		case pt.Inline != nil:
			tests = append(tests, resolvedTest{name: fmt.Sprintf("inline-%d-%s", i, pt.Inline.Kind), spec: *pt.Inline, required: required})
		default:
			return &profile, nil, fmt.Errorf("profile test %d names no testRef and no inline spec", i)
		}
	}
	return &profile, tests, nil
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
		// GPU identity labels if present (GFD / glimmer labels).
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

// step advances a Running run: settle or execute each planned test in order,
// harvest terminated pods, and finalize when every test has its results.
func (r *BurnInRunReconciler) step(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan) (ctrl.Result, error) {
	for i, t := range p.Tests {
		if p.FailFast && hasRequiredFailure(run, p) {
			// Remaining tests are never scheduled; finalize cleans up any
			// pods already in flight. Their absence from Results is the
			// record that they did not run.
			break
		}

		// Tests the operator can settle without scheduling anything.
		if settled := r.settleWithoutPod(ctx, run, p, t); settled {
			continue
		}

		waiting := false
		harvested := false
		for _, node := range p.Targets {
			if hasResult(run, t.Name, node) {
				continue
			}
			outcome, ready, err := r.harvestOrCreate(ctx, run, i, t, node)
			if err != nil {
				return ctrl.Result{}, err
			}
			if !ready {
				waiting = true
				continue
			}
			r.recordResult(ctx, run, p, t, node, outcome)
			harvested = true
		}
		if waiting {
			// Pod events drive the next reconcile; the poll interval is the
			// backstop for pods that never emit one. Persist anything
			// harvested this pass so a restart cannot lose it.
			if harvested {
				return ctrl.Result{RequeueAfter: waitingPollInterval}, r.Status().Update(ctx, run)
			}
			return ctrl.Result{RequeueAfter: waitingPollInterval}, nil
		}
		if harvested {
			// Test finished this pass: persist before moving on, so the
			// delivery and the recorded result stay adjacent in time.
			return ctrl.Result{Requeue: true}, r.Status().Update(ctx, run)
		}
	}

	return r.finalize(ctx, run, p)
}

// settleWithoutPod records results for tests that must not schedule anything:
// unsupported scopes, and hardware the test does not apply to.
func (r *BurnInRunReconciler) settleWithoutPod(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, t plannedTest) bool {
	if t.Spec.Scope != "" && t.Spec.Scope != burninv1alpha1.ScopeNode {
		if !hasResult(run, t.Name, "") {
			r.appendResult(ctx, run, p, burninv1alpha1.TestResult{
				Name:  t.Name,
				Kind:  t.Spec.Kind,
				Scope: t.Spec.Scope,
				Phase: burninv1alpha1.RunError,
				Message: fmt.Sprintf("scope %q is not executed by this operator version (v1 runs Node scope) — "+
					"the test was NOT run; treat this hardware as unjudged for it", t.Spec.Scope),
			})
		}
		return true
	}

	// gpudirect-rdma cannot produce a meaningful verdict on GB10: NVIDIA
	// states unified memory does not support GPUDirect RDMA at all. Skip is
	// the honest verdict — the hardware has not failed a test it cannot take.
	if t.Spec.Kind == burninv1alpha1.KindGPUDirect && allTargetsGB10(run, p.Targets) {
		if !hasResult(run, t.Name, "") {
			r.appendResult(ctx, run, p, burninv1alpha1.TestResult{
				Name:    t.Name,
				Kind:    t.Spec.Kind,
				Scope:   t.Spec.Scope,
				Phase:   burninv1alpha1.RunSkipped,
				Nodes:   p.Targets,
				Message: "not applicable — GB10 unified memory does not support GPUDirect RDMA (per NVIDIA); RDMA serves the GPU via host memory",
			})
		}
		return true
	}
	return false
}

// allTargetsGB10 reports whether every target's captured fingerprint marks it
// as sm_121. Unknown fingerprints return false: skipping a test on hardware we
// cannot identify would be a verdict from ignorance.
func allTargetsGB10(run *burninv1alpha1.BurnInRun, targets []string) bool {
	for _, node := range targets {
		if !strings.Contains(run.Status.Fingerprint[node], "sm_121") {
			return false
		}
	}
	return len(targets) > 0
}

// harvestOrCreate ensures the pod for (test, node) exists, and returns its
// parsed outcome once it has terminated. ready is false while it runs.
func (r *BurnInRunReconciler) harvestOrCreate(ctx context.Context, run *burninv1alpha1.BurnInRun, index int, t plannedTest, node string) (runner.Result, bool, error) {
	name := podName(run, index, node)
	var pod corev1.Pod
	err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, &pod)
	switch {
	case apierrors.IsNotFound(err):
		newPod, buildErr := podForTest(run, index, t.Name, &t.Spec, node, run.Spec.Target)
		if buildErr != nil {
			// Unbuildable pod (no image for the kind): machinery error, and
			// asking again cannot fix it.
			return runner.Result{Verdict: runner.VerdictError, Message: buildErr.Error()}, true, nil
		}
		if ownErr := controllerutil.SetControllerReference(run, newPod, r.Scheme); ownErr != nil {
			return runner.Result{}, false, ownErr
		}
		if createErr := r.Create(ctx, newPod); createErr != nil && !apierrors.IsAlreadyExists(createErr) {
			return runner.Result{}, false, createErr
		}
		return runner.Result{}, false, nil
	case err != nil:
		return runner.Result{}, false, err
	}

	exitCode, terminated, reason := podOutcome(&pod)
	if !terminated {
		// activeDeadlineSeconds only starts once the pod is bound to a node;
		// an unschedulable pod (typo'd node name, missing toleration) has no
		// deadline at all and would wedge the run in Running forever.
		if r.podOverdue(&pod, &t.Spec) {
			if delErr := r.Delete(ctx, &pod); delErr != nil && !apierrors.IsNotFound(delErr) {
				return runner.Result{}, false, delErr
			}
			return runner.Result{
				Verdict: runner.VerdictError,
				Message: fmt.Sprintf("pod never completed within its window (last phase %q) — unschedulable target or stuck image pull; no verdict", pod.Status.Phase),
			}, true, nil
		}
		return runner.Result{}, false, nil
	}

	stdout, logErr := r.fetchLogs(ctx, &pod)

	res := runner.Parse(string(t.Spec.Kind), stdout, exitCode)
	if res.Verdict == runner.VerdictError && res.Message == "" {
		res.Message = fmt.Sprintf("runner terminated abnormally (exit %d, reason %q)", exitCode, reason)
	}
	if reason == "DeadlineExceeded" {
		// The kubelet killed the pod at its deadline. Whatever the container
		// exited with — including 0 from a SIGTERM-trapping entrypoint — the
		// test did not run to completion, so there is no verdict.
		res.Verdict = runner.VerdictError
		res.Message = "test exceeded its deadline and was killed — no verdict"
	}
	if logErr != nil && res.Verdict == runner.VerdictPass && len(t.Spec.Thresholds) > 0 {
		// The exit code says pass but the metrics are gone and thresholds
		// exist. Fail-closed evaluation would blame the hardware for a log
		// fetch failure; the honest verdict is Error, machinery's fault.
		res.Verdict = runner.VerdictError
		res.Message = fmt.Sprintf("runner logs unavailable (%v) — thresholds cannot be evaluated; no verdict", logErr)
	}
	return res, true, nil
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

// podOverdue reports whether a non-terminated pod has outlived its whole
// window: duration + deadline grace + scheduling grace since creation.
func (r *BurnInRunReconciler) podOverdue(pod *corev1.Pod, spec *burninv1alpha1.BurnInTestSpec) bool {
	// A zero CreationTimestamp means the server never stamped the object
	// (only possible in tests); "overdue since year 1" is not a judgment.
	if pod.CreationTimestamp.IsZero() {
		return false
	}
	duration := spec.DurationSeconds
	if duration <= 0 {
		duration = defaultDurationSeconds
	}
	window := time.Duration(duration+deadlineGraceSeconds)*time.Second + schedulingGracePeriod
	return r.now().Sub(pod.CreationTimestamp.Time) > window
}

// recordResult converts a runner outcome into a TestResult, applies
// thresholds, and appends it with a TestCompleted delivery.
func (r *BurnInRunReconciler) recordResult(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, t plannedTest, node string, res runner.Result) {
	now := metav1.NewTime(r.now())
	result := burninv1alpha1.TestResult{
		Name:       t.Name,
		Kind:       t.Spec.Kind,
		Scope:      t.Spec.Scope,
		Nodes:      []string{node},
		FinishedAt: &now,
		Metrics:    res.Metrics,
		Message:    res.Message,
	}

	switch res.Verdict {
	case runner.VerdictPass:
		// Exit 0 says the runner is content; the thresholds are the profile's
		// own acceptance bar, evaluated fail-closed.
		if ok, why := verdict.Evaluate(res.Metrics, t.Spec.Thresholds); ok {
			result.Phase = burninv1alpha1.RunPassed
		} else {
			result.Phase = burninv1alpha1.RunFailed
			result.Message = why
		}
	case runner.VerdictFail:
		result.Phase = burninv1alpha1.RunFailed
	case runner.VerdictSkip:
		result.Phase = burninv1alpha1.RunSkipped
	default:
		result.Phase = burninv1alpha1.RunError
	}
	if len(res.InvalidNames) > 0 {
		result.Message = strings.TrimSpace(result.Message + fmt.Sprintf(
			" [runner emitted %d metric name(s) the contract rejects: %s]",
			len(res.InvalidNames), strings.Join(res.InvalidNames, ", ")))
	}

	r.appendResult(ctx, run, p, result)
}

func (r *BurnInRunReconciler) appendResult(ctx context.Context, run *burninv1alpha1.BurnInRun, p *plan, result burninv1alpha1.TestResult) {
	run.Status.Results = append(run.Status.Results, result)
	recount(run)
	// One completion per (test, node) execution: the node is part of the
	// event identity, or a second node's completion would dedupe away against
	// the first's. Settled results (skips, unsupported scope) are one event
	// for the whole test.
	key := result.Name
	if len(result.Nodes) == 1 {
		key = result.Name + "/" + result.Nodes[0]
	}
	r.deliver(ctx, run, p.Sinks, contract.ReasonTestCompleted, key)
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
	run.Status.Results = append(run.Status.Results, burninv1alpha1.TestResult{
		Name:    "resolve",
		Phase:   burninv1alpha1.RunError,
		Message: cause.Error(),
	})
	return r.terminate(ctx, run, sinks, burninv1alpha1.RunError)
}

// terminate writes the terminal phase, cleans up leftover pods, and delivers
// the terminal envelope with its own retry loop.
func (r *BurnInRunReconciler) terminate(ctx context.Context, run *burninv1alpha1.BurnInRun, sinks []string, phase burninv1alpha1.RunPhase) (ctrl.Result, error) {
	run.Status.Phase = phase
	finished := metav1.NewTime(r.now())
	run.Status.FinishedAt = &finished
	recount(run)

	// FailFast (and finalizeError) can leave pods of never-harvested tests
	// running; a terminal run must not keep burning the hardware.
	r.deleteUnharvestedPods(ctx, run)

	delivered := r.deliver(ctx, run, sinks, contract.ReasonPhaseChanged, string(phase))
	if !delivered {
		// The terminal envelope is the one delivery with no later transition
		// to carry it; mark it pending so the terminal loop keeps retrying.
		if run.Annotations == nil {
			run.Annotations = map[string]string{}
		}
		run.Annotations[pendingDeliveryAnnotation] = string(phase)
		results := run.Status.Results
		if err := r.Update(ctx, run); err != nil {
			return ctrl.Result{}, err
		}
		// Update() refreshes the object from the server, which still holds the
		// PRE-terminal status — silently reverting the phase we are about to
		// write. Re-apply the terminal fields after the metadata write.
		run.Status.Phase = phase
		run.Status.FinishedAt = &finished
		run.Status.Results = results
		recount(run)
	}
	if err := r.Status().Update(ctx, run); err != nil {
		return ctrl.Result{}, err
	}
	if !delivered {
		return ctrl.Result{RequeueAfter: terminalDeliveryRetryInterval}, nil
	}
	return r.handleTTL(ctx, run)
}

// reconcileTerminal serves an already-finished run: retry a pending terminal
// delivery, then let the TTL reap it.
func (r *BurnInRunReconciler) reconcileTerminal(ctx context.Context, run *burninv1alpha1.BurnInRun) (ctrl.Result, error) {
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
		if r.deliver(ctx, run, p.Sinks, contract.ReasonPhaseChanged, phase) {
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

// deleteUnharvestedPods removes the run's pods that never produced a recorded
// result — the in-flight casualties of FailFast or a resolution error.
// Harvested pods are kept until the run's own TTL for post-mortem logs.
func (r *BurnInRunReconciler) deleteUnharvestedPods(ctx context.Context, run *burninv1alpha1.BurnInRun) {
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(run.Namespace), client.MatchingLabels{labelRun: run.Name}); err != nil {
		log.FromContext(ctx).Error(err, "could not list run pods for cleanup")
		return
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if hasResult(run, pod.Labels[labelTest], pod.Labels[labelNode]) {
			continue
		}
		if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			log.FromContext(ctx).Error(err, "could not delete unharvested pod", "pod", pod.Name)
		}
	}
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
	// Owned pods go with the run via ownerRefs.
	return ctrl.Result{}, client.IgnoreNotFound(r.Delete(ctx, run))
}

// ─── Result bookkeeping ───────────────────────────────────────────────────────

// hasResult reports whether a result exists for the test; with a non-empty
// node it must cover that node.
func hasResult(run *burninv1alpha1.BurnInRun, testName, node string) bool {
	for _, res := range run.Status.Results {
		if res.Name != testName {
			continue
		}
		if node == "" {
			return true
		}
		for _, n := range res.Nodes {
			if n == node {
				return true
			}
		}
	}
	return false
}

// recount tallies per-(test,node) executions. The envelope documents the same
// unit, so a 2-test × 3-node run legitimately reports passed=6.
func recount(run *burninv1alpha1.BurnInRun) {
	var passed, failed int32
	for _, res := range run.Status.Results {
		switch res.Phase {
		case burninv1alpha1.RunPassed:
			passed++
		case burninv1alpha1.RunFailed:
			failed++
		}
	}
	run.Status.Passed, run.Status.Failed = passed, failed
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

// isTerminal reports whether a phase is final.
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

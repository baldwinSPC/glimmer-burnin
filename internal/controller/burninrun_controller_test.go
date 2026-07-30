package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Verbatim stdout of the shipped compute-smoke runner on a real GB10.
const fp4Stdout = `gpu_name=NVIDIA GB10
compute_cap=12.1
max_rel_error=0.00195695
nonfinite_count=0
elapsed_ms=0.0206
tflops=104.37
FP4_GEMM_PASS
`

type harness struct {
	t      *testing.T
	r      *BurnInRunReconciler
	c      client.Client
	logs   map[string]string // pod name -> stdout
	nowVal time.Time
}

func newHarness(t *testing.T, objs ...client.Object) *harness {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(&burninv1alpha1.BurnInRun{}, &burninv1alpha1.BurnInSink{}).
		Build()

	h := &harness{
		t:      t,
		c:      c,
		logs:   map[string]string{},
		nowVal: time.Unix(1750000000, 0).UTC(),
	}
	h.r = &BurnInRunReconciler{
		Client: c,
		Scheme: scheme,
		PodLogs: func(_ context.Context, _, name string) (string, error) {
			return h.logs[name], nil
		},
		Now: func() time.Time { return h.nowVal },
	}
	return h
}

func (h *harness) reconcile(name string) ctrl.Result {
	h.t.Helper()
	res, err := h.r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "burnin", Name: name}})
	if err != nil {
		h.t.Fatalf("Reconcile: %v", err)
	}
	return res
}

// reconcileUntilSettled drives the loop like the manager would, bounded so a
// livelock fails the test instead of hanging it.
func (h *harness) reconcileUntilSettled(name string) {
	h.t.Helper()
	for i := 0; i < 25; i++ {
		res := h.reconcile(name)
		if !res.Requeue && res.RequeueAfter == 0 {
			return
		}
	}
	h.t.Fatal("run did not settle within 25 reconciles")
}

func (h *harness) run(name string) *burninv1alpha1.BurnInRun {
	h.t.Helper()
	var run burninv1alpha1.BurnInRun
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: name}, &run); err != nil {
		h.t.Fatalf("get run: %v", err)
	}
	return &run
}

// pods returns the run's pods, keyed by target node.
func (h *harness) pods(runName string) map[string]*corev1.Pod {
	h.t.Helper()
	var list corev1.PodList
	if err := h.c.List(context.Background(), &list, client.InNamespace("burnin"), client.MatchingLabels{labelRun: runName}); err != nil {
		h.t.Fatalf("list pods: %v", err)
	}
	out := map[string]*corev1.Pod{}
	for i := range list.Items {
		out[list.Items[i].Labels[labelNode]] = &list.Items[i]
	}
	return out
}

// finishPod marks a pod terminated with the given exit code and stdout.
func (h *harness) finishPod(pod *corev1.Pod, exitCode int, stdout, reason string) {
	h.t.Helper()
	h.logs[pod.Name] = stdout
	phase := corev1.PodSucceeded
	if exitCode != 0 {
		phase = corev1.PodFailed
	}
	pod.Status = corev1.PodStatus{
		Phase: phase,
		ContainerStatuses: []corev1.ContainerStatus{{
			Name: "runner",
			State: corev1.ContainerState{
				Terminated: &corev1.ContainerStateTerminated{ExitCode: int32(exitCode), Reason: reason},
			},
		}},
	}
	if err := h.c.Status().Update(context.Background(), pod); err != nil {
		h.t.Fatalf("update pod status: %v", err)
	}
}

// ─── Fixtures ─────────────────────────────────────────────────────────────────

func gb10Node(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"glimmer.ai/gpu-arch": "sm_121", corev1.LabelHostname: name},
		},
		Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{
			KernelVersion: "6.11.0-1014-nvidia", OSImage: "Ubuntu 24.04", Architecture: "arm64",
		}},
	}
}

func smokeTest(name string, thresholds ...burninv1alpha1.Threshold) *burninv1alpha1.BurnInTest {
	return &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:       burninv1alpha1.KindComputeSmoke,
			Scope:      burninv1alpha1.ScopeNode,
			Thresholds: thresholds,
		},
	}
}

func profile(name string, sinks []string, failFast bool, tests ...burninv1alpha1.ProfileTest) *burninv1alpha1.BurnInProfile {
	return &burninv1alpha1.BurnInProfile{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec:       burninv1alpha1.BurnInProfileSpec{Tests: tests, Sinks: sinks, FailFast: failFast},
	}
}

func testRef(name string) burninv1alpha1.ProfileTest {
	return burninv1alpha1.ProfileTest{TestRef: name}
}

func newRun(name, profileRef string, nodes ...string) *burninv1alpha1.BurnInRun {
	return &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "burnin", Name: name, UID: types.UID("uid-" + name),
			// Well past resolveGracePeriod relative to the harness clock, so
			// config errors finalize instead of waiting out the apply race.
			CreationTimestamp: metav1.NewTime(time.Unix(1750000000, 0).UTC().Add(-10 * time.Minute)),
		},
		Spec: burninv1alpha1.BurnInRunSpec{
			ProfileRef: profileRef,
			Target:     burninv1alpha1.TargetSelector{NodeNames: nodes},
		},
	}
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

func TestRun_PassesEndToEnd(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4", burninv1alpha1.Threshold{Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)

	// Pending → Running, then the pod is created.
	h.reconcile("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("phase after start = %q, want Running", got)
	}
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 1 {
		t.Fatalf("expected 1 pod, got %d", len(pods))
	}
	pod := pods["spark-a"]
	if pod.Spec.NodeSelector[corev1.LabelHostname] != "spark-a" {
		t.Errorf("pod not pinned to target node: %v", pod.Spec.NodeSelector)
	}
	if pod.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Errorf("RestartPolicy = %q; a restarted runner overwrites the evidence", pod.Spec.RestartPolicy)
	}
	if ors := pod.OwnerReferences; len(ors) != 1 || ors[0].Name != "run1" {
		t.Errorf("pod not owned by the run: %+v", ors)
	}

	// The run must sit in Running while the pod executes.
	h.reconcile("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("phase while pod runs = %q, want Running", got)
	}

	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("final phase = %q, want Passed (results: %+v)", run.Status.Phase, run.Status.Results)
	}
	if run.Status.Passed != 1 || run.Status.Failed != 0 {
		t.Errorf("counts = %d/%d, want 1/0", run.Status.Passed, run.Status.Failed)
	}
	res := run.Status.Results[0]
	if res.Metrics["throughputTflops"] != "104.37" {
		t.Errorf("canonical metrics not recorded: %v", res.Metrics)
	}
	if res.Metrics["nonfiniteCount"] != "0" {
		t.Errorf("snake_case metric not canonicalised: %v", res.Metrics)
	}
	if !strings.Contains(run.Status.Fingerprint["spark-a"], "sm_121") {
		t.Errorf("fingerprint not captured: %q", run.Status.Fingerprint["spark-a"])
	}
}

func TestRun_ThresholdFailureFailsTheRun(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		// The runner exits 0, but the profile demands the impossible.
		smokeTest("fp4", burninv1alpha1.Threshold{Metric: "maxRelError", Comparison: burninv1alpha1.LTE, Value: "0.000001"}),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed — exit 0 must not outrank the profile's threshold", run.Status.Phase)
	}
	if msg := run.Status.Results[0].Message; !strings.Contains(msg, "maxRelError") {
		t.Errorf("failure message does not name the failing threshold: %q", msg)
	}
}

// Exit 2 is Skip: the hardware has not failed a test it cannot take, and a
// run of skips still passes.
func TestRun_SkipDoesNotFail(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 2, "FP4_GEMM_SKIP\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunSkipped {
		t.Fatalf("result phase = %q, want Skipped", run.Status.Results[0].Phase)
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run phase = %q, want Passed — Skip must never count as Fail", run.Status.Phase)
	}
	if run.Status.Passed != 0 && run.Status.Failed != 0 {
		t.Errorf("a skip was counted as evidence: %d/%d", run.Status.Passed, run.Status.Failed)
	}
}

// A runner crash is Error — the hardware is unjudged, and unjudged required
// hardware must not be called Passed.
func TestRun_RunnerCrashIsErrorNotFail(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 137, "", "OOMKilled")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("result = %q, want Error", run.Status.Results[0].Phase)
	}
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("run phase = %q, want Error", run.Status.Phase)
	}
	if run.Status.Failed != 0 {
		t.Errorf("an Error was counted as Failed — those are different claims about the hardware")
	}
}

func TestRun_OptionalFailureStillPasses(t *testing.T) {
	notRequired := false
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false,
			burninv1alpha1.ProfileTest{TestRef: "fp4", Required: &notRequired}),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run phase = %q, want Passed — the failing test is marked Required=false", run.Status.Phase)
	}
	if run.Status.Failed != 1 {
		t.Errorf("the informational failure should still be counted: failed=%d", run.Status.Failed)
	}
}

func TestRun_FailFastSkipsRemainingTests(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("first"),
		smokeTest("second"),
		profile("acceptance", nil, true, testRef("first"), testRef("second")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", run.Status.Phase)
	}
	if len(run.Status.Results) != 1 {
		t.Errorf("failFast ran %d tests, want 1: %+v", len(run.Status.Results), run.Status.Results)
	}
	if len(h.pods("run1")) != 1 {
		t.Errorf("failFast scheduled the second test's pod anyway")
	}
}

// A Pair-scope test is beyond this operator version. It must land as Error —
// silently skipping a required acceptance test would pass hardware by omission.
func TestRun_UnsupportedScopeIsErrorNotSkip(t *testing.T) {
	pair := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "nccl-pair"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:  burninv1alpha1.KindNCCL,
			Scope: burninv1alpha1.ScopePair,
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		pair,
		profile("acceptance", nil, false, testRef("nccl-pair")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("unsupported scope = %q, want Error: %q", run.Status.Results[0].Phase, run.Status.Results[0].Message)
	}
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("run phase = %q, want Error", run.Status.Phase)
	}
	if len(h.pods("run1")) != 0 {
		t.Errorf("a pod was scheduled for a scope the operator cannot run")
	}
}

// gpudirect-rdma on GB10 must skip without a pod: NVIDIA states unified
// memory does not support GPUDirect RDMA, so there is no verdict to seek.
func TestRun_GPUDirectSkipsOnGB10(t *testing.T) {
	gd := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "gd"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:  burninv1alpha1.KindGPUDirect,
			Scope: burninv1alpha1.ScopeNode,
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		gd,
		profile("acceptance", nil, false, testRef("gd")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunSkipped {
		t.Fatalf("gpudirect on GB10 = %q, want Skipped: %q", run.Status.Results[0].Phase, run.Status.Results[0].Message)
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run phase = %q, want Passed", run.Status.Phase)
	}
	if len(h.pods("run1")) != 0 {
		t.Errorf("scheduled a GPUDirect pod on hardware that cannot take the test")
	}
}

func TestRun_MissingProfileIsTerminalError(t *testing.T) {
	h := newHarness(t, newRun("run1", "no-such-profile", "spark-a"))
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	if !strings.Contains(run.Status.Results[0].Message, "no-such-profile") {
		t.Errorf("error does not name the missing profile: %q", run.Status.Results[0].Message)
	}
}

// A kind with no default image and no explicit runner cannot be scheduled;
// asking again cannot fix it, so it must settle as Error, not requeue forever.
func TestRun_KindWithoutImageIsError(t *testing.T) {
	soak := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "soak"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:  burninv1alpha1.KindThermalSoak,
			Scope: burninv1alpha1.ScopeNode,
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		soak,
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	if !strings.Contains(run.Status.Results[0].Message, "no default runner image") {
		t.Errorf("message does not explain the missing image: %q", run.Status.Results[0].Message)
	}
}

func TestRun_MultiNodeNeedsEveryNodeToPass(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a", "spark-b"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("expected a pod per node, got %d", len(pods))
	}
	h.finishPod(pods["spark-a"], 0, fp4Stdout, "Completed")
	h.finishPod(pods["spark-b"], 1, "FP4_GEMM_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed — one bad node fails the batch", run.Status.Phase)
	}
	if run.Status.Passed != 1 || run.Status.Failed != 1 {
		t.Errorf("counts = %d/%d, want 1/1", run.Status.Passed, run.Status.Failed)
	}
}

func TestRun_TTLDeletesFinishedRun(t *testing.T) {
	ttl := int32(3600)
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.TTLSecondsAfterFinished = &ttl

	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")

	// Finalizing must schedule the TTL wake-up.
	var res ctrl.Result
	for i := 0; i < 25; i++ {
		res = h.reconcile("run1")
		if res.RequeueAfter > 0 {
			break
		}
	}
	if res.RequeueAfter != time.Hour {
		t.Fatalf("RequeueAfter = %v, want 1h", res.RequeueAfter)
	}

	// After the TTL the run is deleted.
	h.nowVal = h.nowVal.Add(2 * time.Hour)
	h.reconcile("run1")
	var gone burninv1alpha1.BurnInRun
	err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "run1"}, &gone)
	if err == nil {
		t.Error("run still exists after its TTL")
	}
}

// The delivery path, end to end against the real sink code: envelopes for the
// Running transition, each test completion, and the terminal phase must land
// in the ConfigMap sink, deduplicated by derived DeliveryID.
func TestRun_DeliversToConfigMapSink(t *testing.T) {
	cmSink := &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "results"},
		Spec: burninv1alpha1.BurnInSinkSpec{
			Type:      burninv1alpha1.SinkConfigMap,
			ConfigMap: &burninv1alpha1.ConfigMapSink{Name: "burnin-results"},
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		cmSink,
		profile("acceptance", []string{"results"}, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	var cm corev1.ConfigMap
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "burnin-results"}, &cm); err != nil {
		t.Fatalf("sink ConfigMap was never written: %v", err)
	}
	// Running + TestCompleted + Passed = exactly 3 distinct deliveries.
	if len(cm.Data) != 3 {
		keys := make([]string, 0, len(cm.Data))
		for k := range cm.Data {
			keys = append(keys, k)
		}
		t.Fatalf("deliveries = %d, want 3 (Running, TestCompleted, Passed): %v", len(cm.Data), keys)
	}
	reasons := map[contract.Reason]int{}
	for _, raw := range cm.Data {
		var env contract.Envelope
		if err := json.Unmarshal([]byte(raw), &env); err != nil {
			t.Fatalf("stored envelope does not decode: %v", err)
		}
		if err := env.Validate(); err != nil {
			t.Errorf("stored envelope is invalid: %v", err)
		}
		reasons[env.Reason]++
	}
	if reasons[contract.ReasonPhaseChanged] != 2 || reasons[contract.ReasonTestCompleted] != 1 {
		t.Errorf("delivery reasons = %v, want 2 phase changes + 1 test completion", reasons)
	}

	// Sink status must record the successful delivery.
	var s burninv1alpha1.BurnInSink
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "results"}, &s); err != nil {
		t.Fatal(err)
	}
	if s.Status.LastDelivery == nil {
		t.Error("sink status does not record the delivery")
	}
	if s.Status.LastError != "" {
		t.Errorf("sink status carries an error after successful deliveries: %q", s.Status.LastError)
	}
}

// A broken sink must not block the verdict: delivery is best-effort per
// transition and the run still terminates.
func TestRun_BrokenSinkDoesNotWedgeTheRun(t *testing.T) {
	badSink := &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "bad"},
		Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkWebhook}, // no webhook config
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		badSink,
		profile("acceptance", []string{"bad"}, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")

	// The run must reach its verdict; the failed terminal delivery stays
	// pending and is retried on a timer rather than lost.
	var res ctrl.Result
	for i := 0; i < 25; i++ {
		res = h.reconcile("run1")
		if h.run("run1").Status.Phase == burninv1alpha1.RunPassed {
			break
		}
	}
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed despite the broken sink", got)
	}
	if res.RequeueAfter != terminalDeliveryRetryInterval {
		t.Errorf("RequeueAfter = %v, want the terminal-delivery retry interval", res.RequeueAfter)
	}
	if _, pending := h.run("run1").Annotations[pendingDeliveryAnnotation]; !pending {
		t.Error("failed terminal delivery is not marked pending — the verdict envelope would be lost")
	}
	var s burninv1alpha1.BurnInSink
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "bad"}, &s); err != nil {
		t.Fatal(err)
	}
	if s.Status.LastError == "" {
		t.Error("the misconfigured sink's status does not surface the delivery failure")
	}
}

// Reconciling the same states again must not duplicate pods or results —
// the loop is level-based and every step idempotent.
func TestRun_ReconcileIsIdempotent(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}
	if pods := h.pods("run1"); len(pods) != 1 {
		t.Fatalf("repeated reconciles created %d pods, want 1", len(pods))
	}
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}
	run := h.run("run1")
	if len(run.Status.Results) != 1 {
		t.Fatalf("repeated reconciles recorded %d results, want 1", len(run.Status.Results))
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("phase = %q, want Passed", run.Status.Phase)
	}
}

// ─── Regressions from the adversarial review ──────────────────────────────────

// A SIGTERM-trapping runner exits 0 when the kubelet kills its pod at the
// deadline. The pod-level DeadlineExceeded reason must outrank the container's
// clean exit — otherwise a test that never completed records Pass.
func TestRun_DeadlineKillOutranksCleanContainerExit(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"), // no thresholds: exit 0 alone would pass
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	pod := h.pods("run1")["spark-a"]
	h.logs[pod.Name] = "FP4_GEMM_PASS\n"
	pod.Status = corev1.PodStatus{
		Phase:  corev1.PodFailed,
		Reason: "DeadlineExceeded",
		ContainerStatuses: []corev1.ContainerStatus{{
			Name: "runner",
			State: corev1.ContainerState{
				// The trap handler exited cleanly.
				Terminated: &corev1.ContainerStateTerminated{ExitCode: 0, Reason: "Completed"},
			},
		}},
	}
	if err := h.c.Status().Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("deadline-killed test = %q, want Error — exit 0 from a SIGTERM trap is not a completed test", run.Status.Results[0].Phase)
	}
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("run phase = %q, want Error", run.Status.Phase)
	}
}

// A log-fetch failure is the machinery's fault. With thresholds pending it
// must be Error, never a threshold Fail blaming healthy hardware.
func TestRun_LogFetchFailureIsErrorNotThresholdFail(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4", burninv1alpha1.Threshold{Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.r.PodLogs = func(context.Context, string, string) (string, error) {
		return "", fmt.Errorf("kubelet log rotation ate it")
	}
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, "", "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("result = %q, want Error: %q", run.Status.Results[0].Phase, run.Status.Results[0].Message)
	}
	if run.Status.Failed != 0 {
		t.Errorf("a log-fetch failure was counted as a hardware failure")
	}
}

// The plan is pinned at start: editing the profile mid-run must not change
// what the run executes or which tests count as required.
func TestRun_PlanIsHermeticAgainstProfileEdits(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1") // pins the plan

	// Sabotage: empty the profile and delete the test mid-run.
	var prof burninv1alpha1.BurnInProfile
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "acceptance"}, &prof); err != nil {
		t.Fatal(err)
	}
	prof.Spec.Tests = []burninv1alpha1.ProfileTest{{TestRef: "no-such-test"}}
	if err := h.c.Update(context.Background(), &prof); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — the run must execute its pinned plan, not the edited profile", run.Status.Phase)
	}
	if len(run.Status.Results) != 1 || run.Status.Results[0].Name != "fp4" {
		t.Errorf("results = %+v, want the originally planned fp4 test", run.Status.Results)
	}
}

// An unschedulable pod never gets a StartTime, so activeDeadlineSeconds never
// fires. The controller's own window must reap it or the run wedges forever.
func TestRun_UnschedulablePodTimesOutAsError(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-typo"), // no such node
	)
	h.reconcile("run1")
	h.reconcile("run1")

	// Stamp the pod's creation into the past, then advance past the window.
	pod := h.pods("run1")["spark-typo"]
	pod.CreationTimestamp = metav1.NewTime(h.nowVal)
	if err := h.c.Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	h.nowVal = h.nowVal.Add(time.Hour)
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error — an unschedulable pod must not wedge the run", run.Status.Phase)
	}
	if !strings.Contains(run.Status.Results[0].Message, "never completed") {
		t.Errorf("message does not explain the timeout: %q", run.Status.Results[0].Message)
	}
	if len(h.pods("run1")) != 0 {
		t.Errorf("the stuck pod was not cleaned up")
	}
}

// FailFast abandons the remaining tests; any in-flight pods must be deleted at
// finalize, not left burning the hardware behind a terminal run.
func TestRun_FailFastCleansUpInFlightPods(t *testing.T) {
	twoNode := newRun("run1", "acceptance", "spark-a", "spark-b")
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, true, testRef("fp4")),
		twoNode,
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("expected 2 pods, got %d", len(pods))
	}
	// One node fails; the other is still running when FailFast trips.
	h.finishPod(pods["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", run.Status.Phase)
	}
	remaining := h.pods("run1")
	if _, still := remaining["spark-b"]; still {
		t.Error("failFast left spark-b's pod running behind a terminal run")
	}
	if _, kept := remaining["spark-a"]; !kept {
		t.Error("the harvested pod should be kept for post-mortem logs")
	}
}

// Delete-and-recreate of a same-name run must not adopt the previous run's
// pods: pod identity includes the run UID.
func TestRun_RecreatedRunDoesNotAdoptOldPods(t *testing.T) {
	first := newRun("run1", "acceptance", "spark-a")
	second := newRun("run1", "acceptance", "spark-a")
	second.UID = types.UID("uid-run1-second")

	if podName(first, 0, "spark-a") == podName(second, 0, "spark-a") {
		t.Fatal("pod identity ignores the run UID — a recreated run would harvest the old run's evidence as its own")
	}
}

// A young run whose profile has not landed in the cache yet must wait out the
// apply race, not be terminally errored.
func TestRun_YoungNotFoundGetsGraceNotError(t *testing.T) {
	young := newRun("run1", "not-applied-yet", "spark-a")
	young.CreationTimestamp = metav1.NewTime(time.Unix(1750000000, 0).UTC()) // = harness now
	h := newHarness(t, gb10Node("spark-a"), young)

	res := h.reconcile("run1")
	if res.RequeueAfter == 0 {
		t.Fatal("young run with a missing profile was not given the apply-race grace period")
	}
	if got := h.run("run1").Status.Phase; isTerminal(got) {
		t.Fatalf("phase = %q — terminally errored inside the grace period", got)
	}

	// Past the grace period the missing profile is a real config error.
	h.nowVal = h.nowVal.Add(resolveGracePeriod + time.Minute)
	h.reconcileUntilSettled("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error once the grace period expires", got)
	}
}

// A pending terminal delivery is retried until the sink takes it, then cleared.
func TestRun_PendingTerminalDeliveryEventuallyClears(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		// The sink does not exist yet: every delivery fails.
		profile("acceptance", []string{"late-sink"}, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	for i := 0; i < 10 && h.run("run1").Status.Phase != burninv1alpha1.RunPassed; i++ {
		h.reconcile("run1")
	}
	if _, pending := h.run("run1").Annotations[pendingDeliveryAnnotation]; !pending {
		t.Fatal("terminal delivery failure was not marked pending")
	}

	// The sink comes online late.
	lateSink := &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "late-sink"},
		Spec: burninv1alpha1.BurnInSinkSpec{
			Type:      burninv1alpha1.SinkConfigMap,
			ConfigMap: &burninv1alpha1.ConfigMapSink{Name: "late-results"},
		},
	}
	if err := h.c.Create(context.Background(), lateSink); err != nil {
		t.Fatal(err)
	}
	h.reconcile("run1")

	if _, pending := h.run("run1").Annotations[pendingDeliveryAnnotation]; pending {
		t.Error("pending marker not cleared after the sink accepted the terminal envelope")
	}
	var cm corev1.ConfigMap
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "late-results"}, &cm); err != nil {
		t.Fatalf("terminal envelope never reached the late sink: %v", err)
	}
	found := false
	for _, raw := range cm.Data {
		var env contract.Envelope
		if json.Unmarshal([]byte(raw), &env) == nil && env.Reason == contract.ReasonPhaseChanged && env.Phase == "Passed" {
			found = true
		}
	}
	if !found {
		t.Error("the delivered envelope is not the terminal Passed phase change")
	}
}

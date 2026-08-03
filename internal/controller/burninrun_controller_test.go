package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
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
	"github.com/baldwinSPC/glimmer-burnin/internal/metrics"
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
	t         *testing.T
	r         *BurnInRunReconciler
	c         client.Client
	scheme    *runtime.Scheme
	logs      map[string]string // pod name -> stdout
	nowVal    time.Time
	forgotten []string
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
		scheme: scheme,
		logs:   map[string]string{},
		nowVal: time.Unix(1750000000, 0).UTC(),
	}
	h.r = h.newReconciler()
	return h
}

// newReconciler builds a reconciler over the harness's client. Calling it again
// mid-test is a manager restart: the reconciler holds no state of its own, so
// everything the new instance knows it has to re-derive from the cluster.
func (h *harness) newReconciler() *BurnInRunReconciler {
	return &BurnInRunReconciler{
		Client: h.c,
		Scheme: h.scheme,
		PodLogs: func(_ context.Context, _, name string) (string, error) {
			return h.logs[name], nil
		},
		Now:           func() time.Time { return h.nowVal },
		ForgetMetrics: func(_, name string) { h.forgotten = append(h.forgotten, name) },
	}
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
	for i := 0; i < 60; i++ {
		res := h.reconcile(name)
		if !res.Requeue && res.RequeueAfter == 0 {
			return
		}
	}
	h.t.Fatal("run did not settle within 60 reconciles")
}

func (h *harness) run(name string) *burninv1alpha1.BurnInRun {
	h.t.Helper()
	var run burninv1alpha1.BurnInRun
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: name}, &run); err != nil {
		h.t.Fatalf("get run: %v", err)
	}
	return &run
}

func (h *harness) node(name string) *corev1.Node {
	h.t.Helper()
	var n corev1.Node
	if err := h.c.Get(context.Background(), types.NamespacedName{Name: name}, &n); err != nil {
		h.t.Fatalf("get node %s: %v", name, err)
	}
	return &n
}

func (h *harness) allPods(runName string) []*corev1.Pod {
	h.t.Helper()
	var list corev1.PodList
	if err := h.c.List(context.Background(), &list, client.InNamespace("burnin"), client.MatchingLabels{labelRun: runName}); err != nil {
		h.t.Fatalf("list pods: %v", err)
	}
	out := make([]*corev1.Pod, 0, len(list.Items))
	for i := range list.Items {
		out = append(out, &list.Items[i])
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// pods returns the run's CURRENT pod per target node — the highest attempt
// index seen, since repeats and retries leave earlier attempts' pods behind.
func (h *harness) pods(runName string) map[string]*corev1.Pod {
	h.t.Helper()
	out := map[string]*corev1.Pod{}
	best := map[string]string{}
	for _, pod := range h.allPods(runName) {
		node := pod.Labels[labelNode]
		if prev, ok := best[node]; !ok || pod.Labels[labelAttempt] > prev {
			best[node] = pod.Labels[labelAttempt]
			out[node] = pod
		}
	}
	return out
}

// livePods are the pods currently drawing power on a node — the unit the
// facility interlock is expressed in.
func (h *harness) livePods(runName string) []*corev1.Pod {
	h.t.Helper()
	var out []*corev1.Pod
	for _, pod := range h.allPods(runName) {
		if podLive(pod) {
			out = append(out, pod)
		}
	}
	return out
}

// startPod marks a pod as running on its node, which is what makes an
// execution's StartedAt meaningful.
func (h *harness) startPod(pod *corev1.Pod) {
	h.t.Helper()
	st := metav1.NewTime(h.nowVal)
	pod.Status.Phase = corev1.PodRunning
	pod.Status.StartTime = &st
	if err := h.c.Status().Update(context.Background(), pod); err != nil {
		h.t.Fatalf("update pod status: %v", err)
	}
}

// readyPod marks a pod Running AND Ready. Ready is the gate a Pair client waits
// on, and it is deliberately a different signal from "started": a running
// container has not necessarily bound its socket.
func (h *harness) readyPod(pod *corev1.Pod) {
	h.t.Helper()
	st := metav1.NewTime(h.nowVal)
	pod.Status.Phase = corev1.PodRunning
	pod.Status.StartTime = &st
	pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}
	if err := h.c.Status().Update(context.Background(), pod); err != nil {
		h.t.Fatalf("update pod status: %v", err)
	}
}

// pairPod returns the run's current pod for one end of a Pair execution, or nil
// if that end has not been created yet.
func (h *harness) pairPod(runName, role string) *corev1.Pod {
	h.t.Helper()
	var out *corev1.Pod
	for _, pod := range h.allPods(runName) {
		if pod.Labels[labelPairRole] != role {
			continue
		}
		if out == nil || pod.Labels[labelAttempt] > out.Labels[labelAttempt] {
			out = pod
		}
	}
	return out
}

// rendezvousServices are the headless Services a run has created.
func (h *harness) rendezvousServices(runName string) []*corev1.Service {
	h.t.Helper()
	var list corev1.ServiceList
	if err := h.c.List(context.Background(), &list, client.InNamespace("burnin"), client.MatchingLabels{labelRun: runName}); err != nil {
		h.t.Fatalf("list services: %v", err)
	}
	out := make([]*corev1.Service, 0, len(list.Items))
	for i := range list.Items {
		out = append(out, &list.Items[i])
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// cordonedNodes is what the fleet sees: how many nodes this operator currently
// has out of the scheduler, counted from the nodes themselves rather than from
// the run's own bookkeeping.
func (h *harness) cordonedNodes() []string {
	h.t.Helper()
	var nodes corev1.NodeList
	if err := h.c.List(context.Background(), &nodes); err != nil {
		h.t.Fatalf("list nodes: %v", err)
	}
	var out []string
	for i := range nodes.Items {
		if _, owned := nodes.Items[i].Annotations[burninv1alpha1.AnnotationCordonOwner]; owned {
			out = append(out, nodes.Items[i].Name)
		}
	}
	sort.Strings(out)
	return out
}

func envOf(pod *corev1.Pod, name string) string {
	for _, e := range pod.Spec.Containers[0].Env {
		if e.Name == name {
			return e.Value
		}
	}
	return ""
}

func tolerates(pod *corev1.Pod, key string, effect corev1.TaintEffect) bool {
	for _, tol := range pod.Spec.Tolerations {
		if tol.Key == key && tol.Effect == effect {
			return true
		}
	}
	return false
}

// finishPod marks a pod terminated with the given exit code and stdout.
func (h *harness) finishPod(pod *corev1.Pod, exitCode int, stdout, reason string) {
	h.t.Helper()
	h.logs[pod.Name] = stdout
	phase := corev1.PodSucceeded
	if exitCode != 0 {
		phase = corev1.PodFailed
	}
	started := pod.Status.StartTime
	pod.Status = corev1.PodStatus{
		Phase:     phase,
		StartTime: started,
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

// envelopes decodes every delivery stored in a ConfigMap sink.
func (h *harness) envelopes(cmName string) []contract.Envelope {
	h.t.Helper()
	var cm corev1.ConfigMap
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: cmName}, &cm); err != nil {
		return nil
	}
	out := make([]contract.Envelope, 0, len(cm.Data))
	for _, raw := range cm.Data {
		var env contract.Envelope
		if err := json.Unmarshal([]byte(raw), &env); err != nil {
			h.t.Fatalf("stored envelope does not decode: %v", err)
		}
		out = append(out, env)
	}
	return out
}

func (h *harness) envelopesWithReason(cmName string, reason contract.Reason) []contract.Envelope {
	h.t.Helper()
	var out []contract.Envelope
	for _, env := range h.envelopes(cmName) {
		if env.Reason == reason {
			out = append(out, env)
		}
	}
	return out
}

// assertNoStrandedCordons is the invariant a fleet actually cares about: after
// a run is over, by whatever route, nothing is left holding a node.
func (h *harness) assertNoStrandedCordons() {
	h.t.Helper()
	var nodes corev1.NodeList
	if err := h.c.List(context.Background(), &nodes); err != nil {
		h.t.Fatalf("list nodes: %v", err)
	}
	for i := range nodes.Items {
		n := &nodes.Items[i]
		if owner, ok := n.Annotations[burninv1alpha1.AnnotationCordonOwner]; ok {
			h.t.Errorf("node %s is still stamped as cordoned by %q — the fleet has silently lost it", n.Name, owner)
		}
		if _, ok := n.Annotations[burninv1alpha1.AnnotationPriorUnschedulable]; ok {
			h.t.Errorf("node %s still carries the prior-unschedulable record", n.Name)
		}
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

// healthTest is a host-health test, the kind whose counters a part may be
// physically unable to produce.
func healthTest(name string, thresholds ...burninv1alpha1.Threshold) *burninv1alpha1.BurnInTest {
	return &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:       burninv1alpha1.KindHostHealth,
			Scope:      burninv1alpha1.ScopeNode,
			Thresholds: thresholds,
		},
	}
}

// gb10HostHealthStdout is what the host-health runner prints on a healthy GB10:
// it exits 0, and it DECLARES the two counters the hardware cannot produce
// rather than omitting them or reporting a zero it never measured.
const gb10HostHealthStdout = `nvml_status=ok
gpu_name=NVIDIA GB10
ecc_mode=unsupported
xid_count=0
ecc_errors=n/a
rows_remapped=n/a
pcie_replay_count=0
nic_link_down=0
node_ready=true
HOST_HEALTH_OK
`

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

// withNodeCap raises the facility interlock. Tests that want more than one
// node under load at a time have to say so explicitly, exactly as an operator
// does — which is the point of the default being 1.
func withNodeCap(run *burninv1alpha1.BurnInRun, n int32) *burninv1alpha1.BurnInRun {
	run.Spec.MaxConcurrentNodes = &n
	return run
}

// pairTest is a point-to-point fabric test. It carries an explicit image
// because no ib-write-bw or nccl runner ships in this repo yet, which is a
// packaging gap and not a scheduling one.
func pairTest(name string, thresholds ...burninv1alpha1.Threshold) *burninv1alpha1.BurnInTest {
	return &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:       burninv1alpha1.KindIBWriteBW,
			Scope:      burninv1alpha1.ScopePair,
			Runner:     &burninv1alpha1.RunnerSpec{Image: "example.invalid/ib-write-bw:v1"},
			Thresholds: thresholds,
		},
	}
}

// pairRun is a run over exactly two nodes with the interlock raised to admit
// them together — which a Pair test requires, because it holds both at once.
func pairRun(name, profileRef, a, b string) *burninv1alpha1.BurnInRun {
	return withNodeCap(newRun(name, profileRef, a, b), 2)
}

// ibClientStdout is what perftest's client prints: the measurement lives on the
// client, which is why the client is the deciding side of a pair. The numbers
// are the real measured RoCE ceiling between two GB10 Sparks.
const ibClientStdout = `bw_average=99.63
bw_peak=99.71
IB_WRITE_BW_PASS
`

// ibServerStdout is what the server prints: it served, and it measured nothing.
const ibServerStdout = `IB_WRITE_BW_SERVER_DONE
`

func cmSink(name, cmName string) *burninv1alpha1.BurnInSink {
	return &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInSinkSpec{
			Type:      burninv1alpha1.SinkConfigMap,
			ConfigMap: &burninv1alpha1.ConfigMapSink{Name: cmName},
		},
	}
}

func int32p(v int32) *int32 { return &v }
func boolp(v bool) *bool    { return &v }

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
	if len(res.Attempts) != 1 || res.Attempts[0].Trigger != burninv1alpha1.AttemptInitial {
		t.Errorf("attempt history = %+v, want one Initial attempt", res.Attempts)
	}
	if res.Attempts[0].ExitCode == nil || *res.Attempts[0].ExitCode != 0 {
		t.Errorf("attempt did not record the raw exit code: %+v", res.Attempts[0])
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

// A RequiredIfMeasurable gate on hardware that cannot measure the metric is NOT
// EVALUATED — and that must never look like a gate that was satisfied. The test
// passes on its other thresholds, and the result says which gate did not run and
// why, all the way out to the delivered envelope.
func TestRun_UnevaluatedGateIsReportedAndNeverLooksSatisfied(t *testing.T) {
	eccIfMeasurable := burninv1alpha1.Threshold{
		Metric: "eccErrors", Comparison: burninv1alpha1.EQ, Value: "0",
		Applicability: burninv1alpha1.RequiredIfMeasurable,
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		healthTest("host-fault-counters",
			burninv1alpha1.Threshold{Metric: "xidEvents", Comparison: burninv1alpha1.EQ, Value: "0"},
			eccIfMeasurable,
		),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("host-fault-counters")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, gb10HostHealthStdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — a healthy GB10 has no ECC to gate on (results: %+v)",
			run.Status.Phase, run.Status.Results)
	}
	res := run.Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("result phase = %q, want Passed", res.Phase)
	}

	// The gate that did not run has to be visible, by name and with its reason.
	for _, want := range []string{"not evaluated", "eccErrors", "unmeasurable on this hardware"} {
		if !strings.Contains(res.Message, want) {
			t.Errorf("result message %q does not mention %q; an un-evaluated gate that shows up nowhere reads as a satisfied one",
				res.Message, want)
		}
	}
	// The threshold that DID run is still a real pass, and the unmeasurable
	// metric never becomes a value anyone can chart or compare against.
	if res.Metrics["xidEvents"] != "0" {
		t.Errorf("metrics = %v, want the measured counters recorded", res.Metrics)
	}
	if v, ok := res.Metrics["eccErrors"]; ok {
		t.Errorf("eccErrors = %q was recorded as a metric; the runner declared it unmeasurable", v)
	}

	// And it reaches the consumer, which is the only place acceptance is
	// actually decided.
	envs := h.envelopesWithReason("burnin-results", contract.ReasonTestCompleted)
	if len(envs) != 1 {
		t.Fatalf("test-completion deliveries = %d, want 1", len(envs))
	}
	if len(envs[0].Results) == 0 || !strings.Contains(envs[0].Results[0].Message, "not evaluated") {
		t.Errorf("delivered envelope hides the un-evaluated gate: %+v", envs[0].Results)
	}
}

// The same runner output against the DEFAULT applicability fails: an unmeasured
// counter has not been shown to be within limits. This is what keeps the gate
// meaningful on hardware that does have ECC.
func TestRun_UnmeasurableUnderRequiredFailsTheTest(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		healthTest("host-fault-counters",
			// No applicability set at all: Required is the default.
			burninv1alpha1.Threshold{Metric: "eccErrors", Comparison: burninv1alpha1.EQ, Value: "0"},
		),
		profile("acceptance", nil, false, testRef("host-fault-counters")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, gb10HostHealthStdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed — a Required gate on an unmeasurable metric must fail closed", run.Status.Phase)
	}
	msg := run.Status.Results[0].Message
	if !strings.Contains(msg, "eccErrors") || !strings.Contains(msg, "unmeasurable") {
		t.Errorf("message = %q, want it to name the metric and say the hardware cannot measure it", msg)
	}
	if strings.Contains(msg, "not evaluated") {
		t.Errorf("message = %q; a Required gate was evaluated and failed, not skipped", msg)
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
	if run.Status.Passed != 0 || run.Status.Failed != 0 {
		t.Errorf("a skip was counted as evidence: %d/%d", run.Status.Passed, run.Status.Failed)
	}
	if run.Status.Skipped != 1 {
		t.Errorf("skipped counter = %d, want 1", run.Status.Skipped)
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
	if run.Status.Errored != 1 {
		t.Errorf("errored counter = %d, want 1; an Error that shows up nowhere reads as a clean sweep", run.Status.Errored)
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
	if len(h.allPods("run1")) != 1 {
		t.Errorf("failFast scheduled the second test's pod anyway")
	}
}

// Group scope is still beyond this operator version. It must land as Error —
// silently skipping a required acceptance test would pass hardware by omission.
// Pair is now executed and has its own section further down; this is the test
// that keeps the honest failure honest for what is left.
func TestRun_UnsupportedScopeIsErrorNotSkip(t *testing.T) {
	group := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "nccl-group"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:   burninv1alpha1.KindNCCL,
			Scope:  burninv1alpha1.ScopeGroup,
			Runner: &burninv1alpha1.RunnerSpec{Image: "example.invalid/nccl:v1"},
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"), gb10Node("spark-c"),
		group,
		profile("acceptance", nil, false, testRef("nccl-group")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b", "spark-c"), 3),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("unsupported scope = %q, want Error: %q", run.Status.Results[0].Phase, run.Status.Results[0].Message)
	}
	if !strings.Contains(run.Status.Results[0].Message, "Group") {
		t.Errorf("the message does not name the scope that was refused: %q", run.Status.Results[0].Message)
	}
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("run phase = %q, want Error", run.Status.Phase)
	}
	if len(h.allPods("run1")) != 0 {
		t.Errorf("a pod was scheduled for a scope the operator cannot run")
	}
	h.assertNoStrandedCordons()
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
	if run.Status.Results[0].StartedAt != nil {
		t.Errorf("a test whose pod could not even be built recorded a StartedAt — nothing was ever executed")
	}
}

func TestRun_MultiNodeNeedsEveryNodeToPass(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
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
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.TTLSecondsAfterFinished = int32p(3600)

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
		if res.RequeueAfter == time.Hour {
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
	if len(h.forgotten) != 1 || h.forgotten[0] != "run1" {
		t.Errorf("exported series were not dropped with the run: %v — a gauge for an object that no longer exists never goes away", h.forgotten)
	}
	h.assertNoStrandedCordons()
}

// The delivery path, end to end against the real sink code: envelopes for the
// Running transition, each test completion, and the terminal phase must land
// in the ConfigMap sink, deduplicated by derived DeliveryID.
func TestRun_DeliversToConfigMapSink(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	envs := h.envelopes("burnin-results")
	// Running + TestCompleted + Passed = exactly 3 distinct deliveries.
	if len(envs) != 3 {
		t.Fatalf("deliveries = %d, want 3 (Running, TestCompleted, Passed)", len(envs))
	}
	reasons := map[contract.Reason]int{}
	for _, env := range envs {
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
	// The nodes must come back even when the export never lands. A run whose
	// consumer is unreachable is still a run that is over.
	h.assertNoStrandedCordons()
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
	if pods := h.allPods("run1"); len(pods) != 1 {
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
	if len(run.Status.Results[0].Attempts) != 1 {
		t.Errorf("repeated reconciles recorded %d attempts, want 1", len(run.Status.Results[0].Attempts))
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("phase = %q, want Passed", run.Status.Phase)
	}
}

// ─── Vendor neutrality ────────────────────────────────────────────────────────

// The reconciler must special-case NO accelerator, vendor or product. Whether a
// test applies to a part is the runner's answer (exit 2), made after looking at
// the device; a controller-side exemption keyed off a node label is a verdict
// from a string, and it goes stale the moment a driver or firmware revision
// changes the answer.
//
// This test pins that down for every kind the API defines, on a node whose
// fingerprint is exactly the one an earlier version of this controller used to
// exempt. Every kind must get a pod.
func TestRun_ControllerSpecialCasesNoVendor(t *testing.T) {
	kinds := []burninv1alpha1.TestKind{
		burninv1alpha1.KindGPUBurn,
		burninv1alpha1.KindComputeSmoke,
		burninv1alpha1.KindDCGMDiag,
		burninv1alpha1.KindThermalSoak,
		burninv1alpha1.KindIBWriteBW,
		burninv1alpha1.KindGPUDirect,
		burninv1alpha1.KindMemoryBW,
		burninv1alpha1.KindHostHealth,
		burninv1alpha1.KindClockProbe,
		burninv1alpha1.KindMemoryStress,
		burninv1alpha1.KindCustom,
	}

	for _, kind := range kinds {
		t.Run(string(kind), func(t *testing.T) {
			bt := &burninv1alpha1.BurnInTest{
				ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "t"},
				Spec: burninv1alpha1.BurnInTestSpec{
					Kind:  kind,
					Scope: burninv1alpha1.ScopeNode,
					// An explicit image, so a missing default cannot be
					// mistaken for a vendor exemption.
					Runner: &burninv1alpha1.RunnerSpec{Image: "example.invalid/runner:v1"},
				},
			}
			h := newHarness(t,
				gb10Node("spark-a"), // sm_121 / unified memory: the old exemption
				bt,
				profile("acceptance", nil, false, testRef("t")),
				newRun("run1", "acceptance", "spark-a"),
			)
			h.reconcile("run1")
			h.reconcile("run1")

			if len(h.allPods("run1")) != 1 {
				t.Fatalf("kind %q was settled without a pod — the controller is judging applicability from a label instead of letting the runner answer", kind)
			}
			if run := h.run("run1"); len(run.Status.Results) != 0 {
				t.Fatalf("kind %q got a controller-side verdict before anything ran: %+v", kind, run.Status.Results)
			}
		})
	}
}

// The corollary: a test that genuinely does not apply says so with exit 2, and
// THAT is what produces Skipped — after the pod ran on the real hardware.
func TestRun_RunnerExitTwoIsTheOnlySkip(t *testing.T) {
	gd := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "gd"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:   burninv1alpha1.KindGPUDirect,
			Scope:  burninv1alpha1.ScopeNode,
			Runner: &burninv1alpha1.RunnerSpec{Image: "example.invalid/gpudirect:v1"},
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		gd,
		profile("acceptance", nil, false, testRef("gd")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 2, "unified memory does not support GPUDirect RDMA\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Results[0].Phase != burninv1alpha1.RunSkipped {
		t.Fatalf("exit 2 = %q, want Skipped", run.Status.Results[0].Phase)
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run phase = %q, want Passed — a skip is not a failure", run.Status.Phase)
	}
}

// ─── Facility interlock: maxConcurrentNodes ───────────────────────────────────

// The default is 1, and it is a safety default rather than a performance one:
// an unset field must never fan a full-power sweep across a rack.
func TestRun_DefaultConcurrencyIsOneNode(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"), gb10Node("spark-c"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a", "spark-b", "spark-c"),
	)
	if got := maxConcurrentNodes(h.run("run1")); got != 1 {
		t.Fatalf("resolved cap for an unset maxConcurrentNodes = %d, want 1", got)
	}

	h.reconcile("run1")
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
		if n := len(h.livePods("run1")); n > 1 {
			t.Fatalf("%d nodes under load at once with the default cap — that is a rack-level power event nobody asked for", n)
		}
	}
	if n := len(h.livePods("run1")); n != 1 {
		t.Fatalf("live pods = %d, want exactly 1", n)
	}
}

// The cap is honoured continuously, not just on the first pass: as nodes free
// up the run refills to the cap and never past it.
func TestRun_ConcurrencyCapNeverExceeded(t *testing.T) {
	nodes := []string{"n1", "n2", "n3", "n4", "n5"}
	objs := []client.Object{
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", nodes...), 2),
	}
	for _, n := range nodes {
		objs = append(objs, gb10Node(n))
	}
	h := newHarness(t, objs...)

	seenAtCap := false
	for i := 0; i < 40; i++ {
		res := h.reconcile("run1")
		live := h.livePods("run1")
		if len(live) > 2 {
			t.Fatalf("pass %d had %d nodes under load, cap is 2 — the interlock is not holding", i, len(live))
		}
		if len(live) == 2 {
			seenAtCap = true
		}
		for _, pod := range live {
			h.finishPod(pod, 0, fp4Stdout, "Completed")
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %+v", run.Status.Phase, run.Status.Results)
	}
	if !seenAtCap {
		t.Error("the run never used its full allowance; the test would pass trivially at a cap of 1")
	}
	if run.Status.Passed != 5 {
		t.Errorf("passed = %d, want one execution per node (5)", run.Status.Passed)
	}
}

// Lowering the cap mid-run has to bite immediately. The knob exists for a room
// that is in trouble right now, and a value that only applied to the next run
// would be useless in exactly the situation it was added for.
func TestRun_LoweringTheCapAppliesToWorkNotYetLaunched(t *testing.T) {
	h := newHarness(t,
		gb10Node("n1"), gb10Node("n2"), gb10Node("n3"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "n1", "n2", "n3"), 3),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	if n := len(h.livePods("run1")); n != 3 {
		t.Fatalf("live pods = %d, want 3 at a cap of 3", n)
	}
	for _, pod := range h.livePods("run1") {
		h.finishPod(pod, 0, fp4Stdout, "Completed")
	}

	// The operator pulls the cap down for a second test in the same run.
	run := h.run("run1")
	run.Spec.MaxConcurrentNodes = int32p(1)
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	if got := maxConcurrentNodes(h.run("run1")); got != 1 {
		t.Fatalf("lowered cap not observed: %d", got)
	}
}

// ─── Repeats ──────────────────────────────────────────────────────────────────

// RepeatCount is an AND, not a best-of: three clean executions are required and
// each one is its own pod, its own exit code and its own recorded attempt.
func TestRun_RepeatCountRequiresEveryExecutionToPass(t *testing.T) {
	bt := smokeTest("fp4")
	bt.Spec.RepeatCount = int32p(3)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)

	podNames := map[string]bool{}
	for i := 0; i < 30; i++ {
		res := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			podNames[pod.Name] = true
			h.finishPod(pod, 0, fp4Stdout, "Completed")
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %+v", run.Status.Phase, run.Status.Results)
	}
	res := run.Status.Results[0]
	if res.RepeatsRequired != 3 || res.RepeatsCompleted != 3 {
		t.Fatalf("repeats = %d/%d, want 3/3", res.RepeatsCompleted, res.RepeatsRequired)
	}
	if len(res.Attempts) != 3 {
		t.Fatalf("attempts = %d, want 3: %+v", len(res.Attempts), res.Attempts)
	}
	if len(podNames) != 3 {
		t.Errorf("repeats reused %d pod(s) for 3 executions — a re-harvested pod is not a second execution", len(podNames))
	}
	wantTriggers := []burninv1alpha1.AttemptTrigger{
		burninv1alpha1.AttemptInitial, burninv1alpha1.AttemptRepeat, burninv1alpha1.AttemptRepeat,
	}
	for i, want := range wantTriggers {
		if res.Attempts[i].Trigger != want {
			t.Errorf("attempt %d trigger = %q, want %q", i+1, res.Attempts[i].Trigger, want)
		}
	}
}

// The whole point of a repeat is the fault that does not reproduce first time.
// A late failure must fail the test, and it must not be recoverable.
func TestRun_RepeatFailsIfAnyExecutionFails(t *testing.T) {
	bt := smokeTest("fp4")
	bt.Spec.RepeatCount = int32p(3)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)

	execution := 0
	for i := 0; i < 30; i++ {
		res := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			execution++
			if execution == 3 {
				h.finishPod(pod, 1, "FP4_GEMM_FAIL\n", "Error")
			} else {
				h.finishPod(pod, 0, fp4Stdout, "Completed")
			}
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed — two clean passes do not outvote a failure", run.Status.Phase)
	}
	res := run.Status.Results[0]
	if res.RepeatsCompleted != 2 {
		t.Errorf("repeatsCompleted = %d, want 2 — the failing execution must not count as completed", res.RepeatsCompleted)
	}
	if len(res.Attempts) != 3 {
		t.Fatalf("attempts = %d, want 3", len(res.Attempts))
	}
	if got := res.Attempts[2].Phase; got != burninv1alpha1.RunFailed {
		t.Errorf("third attempt phase = %q, want Failed", got)
	}
	if execution != 3 {
		t.Errorf("%d executions ran; a failure must stop the sequence, not be retried into a pass", execution)
	}
}

// ─── Error retries ────────────────────────────────────────────────────────────

// A retry re-runs a test that produced NO verdict. Machinery malfunctioned;
// the hardware was never judged; running it again is the correct response.
func TestRun_ErrorRetryReRunsAnErroredExecution(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(2)
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	execution := 0
	for i := 0; i < 30; i++ {
		res := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			execution++
			if execution == 1 {
				h.finishPod(pod, 137, "", "OOMKilled") // machinery, not hardware
			} else {
				h.finishPod(pod, 0, fp4Stdout, "Completed")
			}
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed after the retry: %+v", got.Status.Phase, got.Status.Results)
	}
	res := got.Status.Results[0]
	if res.ErrorRetries != 1 {
		t.Errorf("errorRetries = %d, want 1", res.ErrorRetries)
	}
	if len(res.Attempts) != 2 {
		t.Fatalf("attempts = %d, want 2: %+v", len(res.Attempts), res.Attempts)
	}
	if res.Attempts[0].Phase != burninv1alpha1.RunError {
		t.Errorf("first attempt = %q, want Error", res.Attempts[0].Phase)
	}
	if res.Attempts[1].Trigger != burninv1alpha1.AttemptErrorRetry {
		t.Errorf("second attempt trigger = %q, want ErrorRetry", res.Attempts[1].Trigger)
	}
	// An errored attempt measured nothing, so it cannot have satisfied a repeat.
	if res.RepeatsCompleted != 1 {
		t.Errorf("repeatsCompleted = %d, want 1 — an errored attempt performed no measurement", res.RepeatsCompleted)
	}
}

// The rule the whole retry feature is bounded by. Re-running a FAILED test
// until it passes launders a hardware fault into an acceptance, and marginal
// hardware is exactly the hardware that passes on the second try.
func TestRun_FailedExecutionIsNeverRetried(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(5) // a generous budget, and irrelevant
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	executions := 0
	for i := 0; i < 30; i++ {
		res := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			executions++
			h.finishPod(pod, 1, "FP4_GEMM_FAIL\n", "Error")
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", got.Status.Phase)
	}
	if executions != 1 {
		t.Fatalf("a Failed execution was re-run %d times — the retry budget must apply to Error only", executions)
	}
	res := got.Status.Results[0]
	if res.ErrorRetries != 0 {
		t.Errorf("errorRetries = %d, want 0", res.ErrorRetries)
	}
	assertNoRetryFollowsANonError(t, res)
}

// Skipped is likewise never retried: a test that does not apply to this
// hardware will not start applying on the next attempt.
func TestRun_SkippedExecutionIsNeverRetried(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(5)
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	executions := 0
	for i := 0; i < 30; i++ {
		res := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			executions++
			h.finishPod(pod, 2, "FP4_GEMM_SKIP\n", "Error")
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}
	if executions != 1 {
		t.Fatalf("a Skipped execution was re-run %d times", executions)
	}
	if got := h.run("run1").Status.Results[0].Phase; got != burninv1alpha1.RunSkipped {
		t.Errorf("result = %q, want Skipped", got)
	}
}

// The retry budget is finite: once spent, the Error stands as the verdict.
func TestRun_ErrorRetryBudgetIsBounded(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(2)
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	executions := 0
	for i := 0; i < 30; i++ {
		res := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			executions++
			h.finishPod(pod, 137, "", "OOMKilled")
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	got := h.run("run1")
	if executions != 3 {
		t.Fatalf("executions = %d, want 3 (initial + 2 retries)", executions)
	}
	if got.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("phase = %q, want Error once the budget is spent", got.Status.Phase)
	}
	if got.Status.Failed != 0 {
		t.Errorf("an exhausted retry budget was reported as a hardware failure")
	}
	res := got.Status.Results[0]
	// The books have to match what happened: two retries granted, three
	// executions recorded, and no repeat satisfied by an attempt that measured
	// nothing.
	if res.ErrorRetries != 2 {
		t.Errorf("errorRetries = %d, want 2 — the budget was spent exactly", res.ErrorRetries)
	}
	if res.RepeatsCompleted != 0 {
		t.Errorf("repeatsCompleted = %d, want 0 — an errored attempt performed no measurement", res.RepeatsCompleted)
	}
	assertAttemptHistory(t, res,
		attemptWas(burninv1alpha1.AttemptInitial, burninv1alpha1.RunError),
		attemptWas(burninv1alpha1.AttemptErrorRetry, burninv1alpha1.RunError),
		attemptWas(burninv1alpha1.AttemptErrorRetry, burninv1alpha1.RunError),
	)
	assertNoRetryFollowsANonError(t, res)
}

// The half of the invariant whose regression is silent. A retry is granted after
// an Error, the machinery works this time, and the hardware turns out to be bad
// — the Fail must settle the test THERE, with retry budget still unspent. If
// anything ever re-ran it, a failing part would get further chances until one
// came out clean, and a clean fourth attempt is exactly how marginal hardware
// gets a certificate.
func TestRun_ErrorRetryThatThenFailsSettlesAsFailedAndStops(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(5) // deliberately generous, and mostly unspent
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	executions := 0
	for i := 0; i < 30; i++ {
		result := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			executions++
			if executions == 1 {
				h.finishPod(pod, 137, "", "OOMKilled") // machinery: no verdict
			} else {
				h.finishPod(pod, 1, "FP4_GEMM_FAIL\n", "Error") // hardware: a verdict
			}
		}
		if !result.Requeue && result.RequeueAfter == 0 {
			break
		}
	}

	got := h.run("run1")
	if executions != 2 {
		t.Fatalf("executions = %d, want 2 — the Fail that followed the retry was re-run", executions)
	}
	if pods := h.allPods("run1"); len(pods) != 2 {
		t.Errorf("%d pods exist for 2 executions — a further attempt was minted after the Fail", len(pods))
	}
	if got.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed — a hardware verdict outranks the machinery that preceded it", got.Status.Phase)
	}
	if got.Status.Failed != 1 {
		t.Errorf("failed count = %d, want 1", got.Status.Failed)
	}
	res := got.Status.Results[0]
	if res.Phase != burninv1alpha1.RunFailed {
		t.Errorf("result phase = %q, want Failed", res.Phase)
	}
	// Budget remaining is the point: the test stopped because the verdict was a
	// Fail, not because it ran out of chances.
	if res.ErrorRetries != 1 {
		t.Errorf("errorRetries = %d, want 1 — only the Error consumed budget, and 4 were still available", res.ErrorRetries)
	}
	if res.RepeatsCompleted != 0 {
		t.Errorf("repeatsCompleted = %d, want 0 — neither an Error nor a Fail completes a repeat", res.RepeatsCompleted)
	}
	assertAttemptHistory(t, res,
		attemptWas(burninv1alpha1.AttemptInitial, burninv1alpha1.RunError),
		attemptWas(burninv1alpha1.AttemptErrorRetry, burninv1alpha1.RunFailed),
	)
	assertNoRetryFollowsANonError(t, res)
}

// A Fail is not only "exit 1". A runner that exits 0 while missing the profile's
// bar is judged Failed by the threshold evaluation, and that verdict reaches
// completeAttempt by a different route than the exit code does. Both routes must
// be equally terminal, or the retry budget would launder exactly the failures
// the thresholds exist to catch.
func TestRun_ThresholdFailureIsNeverRetried(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(5)
	h := newHarness(t,
		gb10Node("spark-a"),
		// The runner is content; the profile demands the impossible.
		smokeTest("fp4", burninv1alpha1.Threshold{
			Metric: "maxRelError", Comparison: burninv1alpha1.LTE, Value: "0.000001",
		}),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	executions := 0
	for i := 0; i < 30; i++ {
		result := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			executions++
			h.finishPod(pod, 0, fp4Stdout, "Completed") // exit 0, below the bar
		}
		if !result.Requeue && result.RequeueAfter == 0 {
			break
		}
	}

	got := h.run("run1")
	if executions != 1 {
		t.Fatalf("a threshold failure was re-run %d times; retries apply to Error only", executions)
	}
	if got.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", got.Status.Phase)
	}
	res := got.Status.Results[0]
	if res.ErrorRetries != 0 {
		t.Errorf("errorRetries = %d, want 0 — nothing errored", res.ErrorRetries)
	}
	if !strings.Contains(res.Message, "maxRelError") {
		t.Errorf("message = %q, want it to name the threshold that decided the verdict", res.Message)
	}
	assertAttemptHistory(t, res, attemptWas(burninv1alpha1.AttemptInitial, burninv1alpha1.RunFailed))
	assertNoRetryFollowsANonError(t, res)
}

// Repeats and retries are different budgets counting different things, and the
// way to prove they are not confused is to spend one of each in the same test.
// A pass completes a repeat; an error consumes a retry and completes no repeat;
// the fail that follows ends it while both budgets still have room.
func TestRun_RepeatsAndRetriesKeepSeparateBooks(t *testing.T) {
	bt := smokeTest("fp4")
	bt.Spec.RepeatCount = int32p(3)
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.RetryOnErrorLimit = int32p(2)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)

	executions := 0
	for i := 0; i < 30; i++ {
		result := h.reconcile("run1")
		for _, pod := range h.livePods("run1") {
			executions++
			switch executions {
			case 1:
				h.finishPod(pod, 0, fp4Stdout, "Completed") // repeat 1 of 3
			case 2:
				h.finishPod(pod, 137, "", "OOMKilled") // no verdict; costs a retry
			default:
				h.finishPod(pod, 1, "FP4_GEMM_FAIL\n", "Error") // a verdict, and it is final
			}
		}
		if !result.Requeue && result.RequeueAfter == 0 {
			break
		}
	}

	got := h.run("run1")
	if executions != 3 {
		t.Fatalf("executions = %d, want 3 — the Fail must end the sequence with both budgets unspent", executions)
	}
	if got.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", got.Status.Phase)
	}
	res := got.Status.Results[0]
	if res.RepeatsRequired != 3 {
		t.Errorf("repeatsRequired = %d, want 3", res.RepeatsRequired)
	}
	if res.RepeatsCompleted != 1 {
		t.Errorf("repeatsCompleted = %d, want 1 — only the passing attempt completed a repeat", res.RepeatsCompleted)
	}
	if res.ErrorRetries != 1 {
		t.Errorf("errorRetries = %d, want 1 — only the errored attempt consumed a retry", res.ErrorRetries)
	}
	assertAttemptHistory(t, res,
		attemptWas(burninv1alpha1.AttemptInitial, burninv1alpha1.RunPassed),
		attemptWas(burninv1alpha1.AttemptRepeat, burninv1alpha1.RunError),
		attemptWas(burninv1alpha1.AttemptErrorRetry, burninv1alpha1.RunFailed),
	)
	assertNoRetryFollowsANonError(t, res)
}

// expectedAttempt is one row of the history a test asserts against.
type expectedAttempt struct {
	trigger burninv1alpha1.AttemptTrigger
	phase   burninv1alpha1.RunPhase
}

func attemptWas(trigger burninv1alpha1.AttemptTrigger, phase burninv1alpha1.RunPhase) expectedAttempt {
	return expectedAttempt{trigger: trigger, phase: phase}
}

// assertAttemptHistory checks the recorded attempts row by row. The history is
// the audit trail for the retry rule, so a test that cares about the rule has to
// assert on the record, not only on the count of pods it saw.
func assertAttemptHistory(t *testing.T, res burninv1alpha1.TestResult, want ...expectedAttempt) {
	t.Helper()
	if len(res.Attempts) != len(want) {
		t.Fatalf("attempts = %d, want %d: %+v", len(res.Attempts), len(want), res.Attempts)
	}
	for i, w := range want {
		got := res.Attempts[i]
		if got.Attempt != int32(i+1) {
			t.Errorf("attempt %d is indexed %d; the sequence must be 1-based and gapless", i+1, got.Attempt)
		}
		if got.Trigger != w.trigger {
			t.Errorf("attempt %d trigger = %q, want %q", i+1, got.Trigger, w.trigger)
		}
		if got.Phase != w.phase {
			t.Errorf("attempt %d phase = %q, want %q", i+1, got.Phase, w.phase)
		}
		if got.FinishedAt == nil {
			t.Errorf("attempt %d has no FinishedAt; a recorded attempt that never ended is not evidence", i+1)
		}
	}
}

// assertNoRetryFollowsANonError audits the recorded history for the one
// transition that must never appear: a retry granted after anything other than
// an Error. TestAttempt.Trigger exists so this is checkable after the fact.
func assertNoRetryFollowsANonError(t *testing.T, res burninv1alpha1.TestResult) {
	t.Helper()
	for i, a := range res.Attempts {
		if a.Trigger != burninv1alpha1.AttemptErrorRetry {
			continue
		}
		if i == 0 {
			t.Errorf("attempt 1 is marked ErrorRetry with nothing before it")
			continue
		}
		if prev := res.Attempts[i-1].Phase; prev != burninv1alpha1.RunError {
			t.Errorf("attempt %d is an ErrorRetry but follows a %q attempt — a hardware verdict has been laundered", i+1, prev)
		}
	}
}

// ─── Checkpoints ──────────────────────────────────────────────────────────────

func checkpointingSoak(name string, intervalSeconds int32, thresholds ...burninv1alpha1.Threshold) *burninv1alpha1.BurnInTest {
	bt := smokeTest(name, thresholds...)
	bt.Spec.CheckpointIntervalSeconds = int32p(intervalSeconds)
	bt.Spec.DurationSeconds = 36000
	return bt
}

// A long soak must say something while it soaks. Without checkpoints a consumer
// sees the Running transition and then nothing until the verdict, and cannot
// tell a run that is working from one that is wedged.
func TestRun_CheckpointDeliversMidRunEvidence(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		checkpointingSoak("soak", 60),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	h.reconcile("run1")

	// Partway through, the runner has printed a partial set of metrics.
	h.logs[pod.Name] = "tflops=51.2\nmax_rel_error=0.001\n"
	h.nowVal = h.nowVal.Add(90 * time.Second)
	res := h.reconcile("run1")

	if res.RequeueAfter <= 0 {
		t.Fatalf("a checkpointing run did not schedule its next wake-up: %+v", res)
	}
	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("phase = %q; a checkpoint is evidence and must not move the run", run.Status.Phase)
	}
	if len(run.Status.Results) != 1 {
		t.Fatalf("no in-progress result was recorded: %+v", run.Status.Results)
	}
	result := run.Status.Results[0]
	if result.Phase != burninv1alpha1.RunRunning {
		t.Errorf("in-progress result phase = %q, want Running", result.Phase)
	}
	if result.Metrics["throughputTflops"] != "51.2" {
		t.Errorf("checkpoint did not publish the partial metrics: %v", result.Metrics)
	}
	if result.LastCheckpointAt == nil {
		t.Error("lastCheckpointAt is nil — nothing marks these numbers as a mid-flight sample")
	}
	if result.FinishedAt != nil {
		t.Error("a checkpointed result must not carry a FinishedAt; the execution is still running")
	}

	checkpoints := h.envelopesWithReason("burnin-results", contract.ReasonCheckpoint)
	if len(checkpoints) != 1 {
		t.Fatalf("checkpoint deliveries = %d, want 1", len(checkpoints))
	}
	if checkpoints[0].Phase != string(burninv1alpha1.RunRunning) {
		t.Errorf("checkpoint envelope phase = %q, want Running", checkpoints[0].Phase)
	}
	if err := checkpoints[0].Validate(); err != nil {
		t.Errorf("checkpoint envelope is invalid: %v", err)
	}
	if len(checkpoints[0].Results) != 1 || checkpoints[0].Results[0].Metrics["throughputTflops"] != "51.2" {
		t.Errorf("checkpoint envelope does not carry the in-flight evidence: %+v", checkpoints[0].Results)
	}

	// A second window produces a second, distinct delivery rather than
	// overwriting the first.
	h.logs[pod.Name] = "tflops=49.8\nmax_rel_error=0.001\n"
	h.nowVal = h.nowVal.Add(90 * time.Second)
	h.reconcile("run1")
	if got := len(h.envelopesWithReason("burnin-results", contract.ReasonCheckpoint)); got != 2 {
		t.Errorf("checkpoint deliveries after a second window = %d, want 2", got)
	}
	if got := h.run("run1").Status.Results[0].Metrics["throughputTflops"]; got != "49.8" {
		t.Errorf("checkpoint metrics = %q, want the latest sample; checkpoints overwrite in place", got)
	}
}

// A checkpoint is EVIDENCE, never a verdict. A mid-run sample that dips below a
// threshold is not a failure, because the run is not over.
func TestRun_CheckpointNeverEvaluatesThresholds(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		checkpointingSoak("soak", 60,
			burninv1alpha1.Threshold{Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	h.reconcile("run1")

	// A sample that would fail the threshold outright.
	h.logs[pod.Name] = "nonfinite_count=17\ntflops=3.0\n"
	h.nowVal = h.nowVal.Add(90 * time.Second)
	h.reconcile("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("phase = %q; a violating mid-run sample must not fail the run", run.Status.Phase)
	}
	if got := run.Status.Results[0].Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("result phase = %q, want Running", got)
	}
	if run.Status.Failed != 0 {
		t.Errorf("a checkpoint was counted as a failure")
	}

	// The completed execution is clean, and that is what gets judged.
	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — the verdict comes from the completed execution", got)
	}
}

// Nothing is checkpointed once the run is over. A terminal run does no work,
// reads no logs, and delivers no further evidence.
func TestRun_NoCheckpointsAfterTerminal(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		checkpointingSoak("soak", 60),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("soak")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	h.logs[pod.Name] = "tflops=51.2\n"
	h.nowVal = h.nowVal.Add(90 * time.Second)
	h.reconcile("run1")

	before := len(h.envelopesWithReason("burnin-results", contract.ReasonCheckpoint))
	if before == 0 {
		t.Fatal("no checkpoint was delivered while the run was in flight")
	}

	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed", got)
	}

	// Keep reconciling a terminal run across several checkpoint windows.
	for i := 0; i < 5; i++ {
		h.nowVal = h.nowVal.Add(5 * time.Minute)
		h.reconcile("run1")
	}
	if got := len(h.envelopesWithReason("burnin-results", contract.ReasonCheckpoint)); got != before {
		t.Errorf("checkpoint deliveries went from %d to %d after the run terminated", before, got)
	}
	if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
		t.Error("a terminal run changed phase on a later reconcile")
	}
}

// Checkpointing is opt-in. A test that does not ask for it must not have its
// logs read on a timer, and must deliver nothing extra.
func TestRun_NoCheckpointsWithoutAnInterval(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.startPod(h.pods("run1")["spark-a"])
	for i := 0; i < 5; i++ {
		h.nowVal = h.nowVal.Add(10 * time.Minute)
		h.reconcile("run1")
	}
	if got := len(h.envelopesWithReason("burnin-results", contract.ReasonCheckpoint)); got != 0 {
		t.Errorf("checkpoint deliveries = %d for a test that never asked for them", got)
	}
}

// ─── StartedAt ────────────────────────────────────────────────────────────────

// StartedAt means "the hardware began being tested", never "we asked for a
// pod". Counting scheduling time as test time turns a stuck run into what looks
// like a slow one.
func TestRun_StartedAtMarksExecutionNotScheduling(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	// The pod exists but the kubelet has not started it.
	h.nowVal = h.nowVal.Add(time.Minute)
	h.reconcile("run1")
	if results := h.run("run1").Status.Results; len(results) != 0 {
		t.Fatalf("a pending pod opened a result with StartedAt %v — scheduling is not execution", results[0].StartedAt)
	}

	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	startedAt := pod.Status.StartTime.Time
	h.reconcile("run1")

	run := h.run("run1")
	if len(run.Status.Results) != 1 {
		t.Fatalf("a started execution recorded no result")
	}
	res := run.Status.Results[0]
	if res.StartedAt == nil {
		t.Fatal("StartedAt is nil for an execution the kubelet has started")
	}
	if !res.StartedAt.Time.Equal(startedAt) {
		t.Errorf("StartedAt = %v, want the pod's own StartTime %v", res.StartedAt.Time, startedAt)
	}
	if res.Attempts[0].StartedAt == nil {
		t.Error("the attempt did not record its own start")
	}

	h.nowVal = h.nowVal.Add(10 * time.Minute)
	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	final := h.run("run1").Status.Results[0]
	if !final.StartedAt.Time.Equal(startedAt) {
		t.Errorf("StartedAt moved on completion: %v", final.StartedAt.Time)
	}
	if final.FinishedAt == nil || !final.FinishedAt.Time.After(final.StartedAt.Time) {
		t.Errorf("StartedAt/FinishedAt do not span the execution: %v .. %v", final.StartedAt, final.FinishedAt)
	}
}

// ─── Suspend ──────────────────────────────────────────────────────────────────

// Suspend is reversible and non-terminal: it stops new work, produces no
// verdict, and resumes where it left off.
func TestRun_SuspendLaunchesNothingAndStaysNonTerminal(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.Suspend = boolp(true)
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		run,
	)
	h.reconcile("run1")
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}
	if n := len(h.allPods("run1")); n != 0 {
		t.Fatalf("a suspended run launched %d pod(s)", n)
	}
	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunRunning {
		t.Fatalf("phase = %q; suspending must not produce a verdict or a terminal phase", got.Status.Phase)
	}

	// Resuming picks the run up where it stopped.
	got.Spec.Suspend = boolp(false)
	if err := h.c.Update(context.Background(), got); err != nil {
		t.Fatal(err)
	}
	h.reconcile("run1")
	if n := len(h.allPods("run1")); n != 1 {
		t.Fatalf("resumed run launched %d pod(s), want 1", n)
	}
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	if p := h.run("run1").Status.Phase; p != burninv1alpha1.RunPassed {
		t.Errorf("phase = %q, want Passed", p)
	}
}

// A suspended in-flight execution is left alone to finish. Suspending is not a
// reason to discard evidence already being paid for.
func TestRun_SuspendLetsInFlightWorkFinish(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)

	got := h.run("run1")
	got.Spec.Suspend = boolp(true)
	if err := h.c.Update(context.Background(), got); err != nil {
		t.Fatal(err)
	}
	h.reconcile("run1")
	if len(h.livePods("run1")) != 1 {
		t.Fatal("suspending killed an in-flight execution")
	}

	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcile("run1")
	h.reconcile("run1")
	final := h.run("run1")
	if len(final.Status.Results) != 1 || final.Status.Results[0].Phase != burninv1alpha1.RunPassed {
		t.Fatalf("the in-flight execution's result was not recorded: %+v", final.Status.Results)
	}
	if final.Status.Phase != burninv1alpha1.RunRunning {
		t.Errorf("phase = %q; a suspended run must not terminate on its own", final.Status.Phase)
	}
}

// ─── Cancellation ─────────────────────────────────────────────────────────────

// Immediate is for a facility event, where the load itself is the reason for
// stopping. Pods go now, un-run work is settled as Cancelled — never Failed —
// and the terminal envelope goes out.
func TestRun_CancelImmediateStopsAndSettles(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("first"),
		smokeTest("second"),
		cmSink("results", "burnin-results"),
		profile("acceptance", []string{"results"}, false, testRef("first"), testRef("second")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.startPod(h.pods("run1")["spark-a"])

	run := h.run("run1")
	run.Spec.Cancel = boolp(true)
	run.Spec.CancelPolicy = burninv1alpha1.CancelImmediate
	run.Spec.CancelReason = "facility power event"
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.reconcileUntilSettled("run1")

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunCancelled {
		t.Fatalf("phase = %q, want Cancelled", got.Status.Phase)
	}
	if got.Status.Failed != 0 {
		t.Errorf("a cancelled run reported %d failures — nobody measured that hardware", got.Status.Failed)
	}
	if n := len(h.livePods("run1")); n != 0 {
		t.Errorf("%d pod(s) still burning hardware behind a cancelled run", n)
	}
	if len(got.Status.Results) != 2 {
		t.Fatalf("results = %d, want one per planned test: %+v", len(got.Status.Results), got.Status.Results)
	}
	for _, res := range got.Status.Results {
		if res.Phase != burninv1alpha1.RunCancelled {
			t.Errorf("result %q = %q, want Cancelled", res.Name, res.Phase)
		}
		if !strings.Contains(res.Message, "facility power event") {
			t.Errorf("result %q does not carry the cancel reason: %q", res.Name, res.Message)
		}
	}

	terminal := false
	for _, env := range h.envelopesWithReason("burnin-results", contract.ReasonPhaseChanged) {
		if env.Phase == string(burninv1alpha1.RunCancelled) {
			terminal = true
		}
	}
	if !terminal {
		t.Error("the terminal Cancelled envelope was never delivered")
	}
	h.assertNoStrandedCordons()
}

// Graceful is the default because an in-flight burn-in is expensive evidence:
// a soak four hours in has already cost the fleet those four hours.
func TestRun_CancelGracefulLetsInFlightFinish(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("first"),
		smokeTest("second"),
		profile("acceptance", nil, false, testRef("first"), testRef("second")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)

	run := h.run("run1")
	run.Spec.Cancel = boolp(true) // CancelPolicy defaults to Graceful
	run.Spec.CancelReason = "superseded by a newer run"
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1")
	if len(h.livePods("run1")) != 1 {
		t.Fatal("a graceful cancel killed the in-flight execution")
	}
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("phase = %q while draining, want Running", got)
	}

	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunCancelled {
		t.Fatalf("phase = %q, want Cancelled", got.Status.Phase)
	}
	first := resultFor(got, "first", "spark-a")
	if first == nil || first.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("the completed execution's real result was discarded: %+v", got.Status.Results)
	}
	if got.Status.Passed != 1 {
		t.Errorf("passed = %d, want 1 — evidence gathered before the cancel is kept", got.Status.Passed)
	}
	second := resultFor(got, "second", "spark-a")
	if second == nil || second.Phase != burninv1alpha1.RunCancelled {
		t.Errorf("the un-run test = %+v, want Cancelled", second)
	}
	// A second pod for the never-started test must never have been created.
	if n := len(h.allPods("run1")); n != 1 {
		t.Errorf("a cancelled run launched %d pods; it must launch nothing new", n)
	}
}

// Cancellation is ONE-WAY. A run that could un-cancel would race its own
// cleanup and could resume after its cordons had already been released.
func TestRun_CancelIsOneWay(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)

	run := h.run("run1")
	run.Spec.Cancel = boolp(true)
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.reconcile("run1") // observes the cancel and starts draining

	if h.run("run1").Annotations[cancellingAnnotation] == "" {
		t.Fatal("the cancel was not recorded durably; clearing the flag would resume the run")
	}

	// Somebody clears the flag mid-drain.
	run = h.run("run1")
	run.Spec.Cancel = boolp(false)
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.finishPod(pod, 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunCancelled {
		t.Fatalf("phase = %q, want Cancelled — clearing spec.cancel must not resurrect a run", got)
	}
}

// Terminal is terminal. A cancel arriving after the verdict changes nothing:
// a Passed run that could be rewritten as Cancelled would make every stored
// verdict provisional.
func TestRun_TerminalRunStaysTerminalWhenCancelled(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunPassed {
		t.Fatalf("setup: phase = %q, want Passed", got)
	}

	run := h.run("run1")
	run.Spec.Cancel = boolp(true)
	run.Spec.Suspend = boolp(true)
	run.Spec.DeadlineSeconds = int32p(1)
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.nowVal = h.nowVal.Add(time.Hour)
	h.reconcileUntilSettled("run1")

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("phase = %q, want Passed — a terminal phase is final", got.Status.Phase)
	}
	if got.Status.Passed != 1 {
		t.Errorf("the stored verdict was rewritten: passed=%d", got.Status.Passed)
	}
}

// ─── Run deadline ─────────────────────────────────────────────────────────────

// A run that runs out of time did not judge the hardware it never reached.
// Error, not Failed (which would condemn untested parts) and not Cancelled
// (which nobody asked for).
func TestRun_DeadlineExpiryIsErrorNotFailedOrCancelled(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.DeadlineSeconds = int32p(300)
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("first"),
		smokeTest("second"),
		profile("acceptance", nil, false, testRef("first"), testRef("second")),
		run,
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcile("run1")
	h.reconcile("run1") // "first" settles, "second" gets a pod

	h.nowVal = h.nowVal.Add(10 * time.Minute)
	h.reconcileUntilSettled("run1")

	got := h.run("run1")
	if got.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", got.Status.Phase)
	}
	if got.Status.Failed != 0 {
		t.Errorf("expiry condemned %d execution(s) as hardware failures", got.Status.Failed)
	}
	first := resultFor(got, "first", "spark-a")
	if first == nil || first.Phase != burninv1alpha1.RunPassed {
		t.Errorf("a deadline retracted evidence already gathered: %+v", got.Status.Results)
	}
	second := resultFor(got, "second", "spark-a")
	if second == nil || second.Phase != burninv1alpha1.RunError {
		t.Fatalf("the unreached test = %+v, want Error", second)
	}
	if !strings.Contains(second.Message, "deadline") {
		t.Errorf("the unjudged result does not explain itself: %q", second.Message)
	}
	if n := len(h.livePods("run1")); n != 0 {
		t.Errorf("%d pod(s) still running past the run's deadline", n)
	}
	h.assertNoStrandedCordons()
}

// ─── Cordon safety ────────────────────────────────────────────────────────────

func TestRun_CordonsTargetsAndRestoresOnSuccess(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	// Two passes: the first pins the plan and marks the run Running, the second
	// admits the node under the interlock — which is when it is cordoned, since
	// a node is held for as long as it is under load and not a moment longer.
	h.reconcile("run1")
	h.reconcile("run1")

	node := h.node("spark-a")
	if !node.Spec.Unschedulable {
		t.Fatal("the target node was not held out of the scheduler; workloads would land beside the burn-in and corrupt the measurement")
	}
	wantOwner := "burnin/run1/uid-run1"
	if got := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; got != wantOwner {
		t.Fatalf("cordon owner = %q, want %q", got, wantOwner)
	}
	if got := node.Annotations[burninv1alpha1.AnnotationPriorUnschedulable]; got != burninv1alpha1.PriorUnschedulableFalse {
		t.Errorf("prior-unschedulable = %q, want %q", got, burninv1alpha1.PriorUnschedulableFalse)
	}
	if got := h.run("run1").Status.CordonedNodes; len(got) != 1 || got[0] != "spark-a" {
		t.Errorf("status.cordonedNodes = %v, want [spark-a] — without the record nothing knows what to give back", got)
	}

	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	if h.node("spark-a").Spec.Unschedulable {
		t.Error("the node was never returned to the scheduler")
	}
	if got := h.run("run1").Status.CordonedNodes; len(got) != 0 {
		t.Errorf("status.cordonedNodes = %v after release, want empty", got)
	}
	h.assertNoStrandedCordons()
}

func TestRun_CordonRestoredOnFailure(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	if h.run("run1").Status.Phase != burninv1alpha1.RunFailed {
		t.Fatal("setup: run did not fail")
	}
	if h.node("spark-a").Spec.Unschedulable {
		t.Error("a failing run kept the node out of the scheduler; a bad verdict is not a reason to hold capacity")
	}
	h.assertNoStrandedCordons()
}

// "Uncordon" is not the inverse of "cordon" when the node was already cordoned.
// An administratively drained node must come out of the run exactly as it went
// in, or a node under maintenance receives production traffic mid-update.
func TestRun_PriorUnschedulableIsRestoredExactly(t *testing.T) {
	drained := gb10Node("spark-a")
	drained.Spec.Unschedulable = true
	h := newHarness(t,
		drained,
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	if got := h.node("spark-a").Annotations[burninv1alpha1.AnnotationPriorUnschedulable]; got != burninv1alpha1.PriorUnschedulableTrue {
		t.Fatalf("prior-unschedulable = %q, want %q", got, burninv1alpha1.PriorUnschedulableTrue)
	}

	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	node := h.node("spark-a")
	if !node.Spec.Unschedulable {
		t.Error("a node that was already drained was returned to service by a burn-in finishing")
	}
	if _, ok := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; ok {
		t.Error("the ownership stamp outlived the run")
	}
}

// A node held by somebody else is never taken and never released.
func TestRun_NeverUncordonsANodeItDidNotCordon(t *testing.T) {
	other := gb10Node("spark-a")
	other.Spec.Unschedulable = true
	other.Annotations = map[string]string{
		burninv1alpha1.AnnotationCordonOwner:        "burnin/other-run/uid-other",
		burninv1alpha1.AnnotationPriorUnschedulable: burninv1alpha1.PriorUnschedulableFalse,
	}
	h := newHarness(t,
		other,
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")

	if got := h.run("run1").Status.CordonedNodes; len(got) != 0 {
		t.Fatalf("status.cordonedNodes = %v — the run claimed a hold it never placed", got)
	}
	if got := h.node("spark-a").Annotations[burninv1alpha1.AnnotationCordonOwner]; got != "burnin/other-run/uid-other" {
		t.Fatalf("cordon owner = %q; the stamp was overwritten and the real owner can no longer release it", got)
	}

	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	node := h.node("spark-a")
	if !node.Spec.Unschedulable {
		t.Error("another owner's cordon was released; a node under maintenance is now taking production traffic")
	}
	if got := node.Annotations[burninv1alpha1.AnnotationCordonOwner]; got != "burnin/other-run/uid-other" {
		t.Errorf("cordon owner = %q after the run, want the original owner", got)
	}
}

// A recreated run of the same name is a DIFFERENT run and has no authority over
// the previous incarnation's node.
func TestRun_CordonOwnershipIsUIDScoped(t *testing.T) {
	first := newRun("run1", "acceptance", "spark-a")
	second := newRun("run1", "acceptance", "spark-a")
	second.UID = types.UID("uid-run1-second")
	if cordonOwnerID(first) == cordonOwnerID(second) {
		t.Fatal("cordon ownership ignores the run UID — a recreated run would release a hold it never placed")
	}
}

// `kubectl delete burninrun` is the likeliest moment to strand a node, and the
// finalizer is what makes cleanup part of deletion rather than a thing that
// happens beforehand if you remember.
func TestRun_DeletedRunReleasesCordonsBeforeItGoes(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	h.startPod(h.pods("run1")["spark-a"])

	run := h.run("run1")
	hasFinalizer := false
	for _, f := range run.Finalizers {
		if f == burninv1alpha1.FinalizerCordonCleanup {
			hasFinalizer = true
		}
	}
	if !hasFinalizer {
		t.Fatal("the run holds a cordon with no finalizer; deleting it would strand the node")
	}

	if err := h.c.Delete(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	// Deletion is blocked by the finalizer until cleanup runs.
	var blocked burninv1alpha1.BurnInRun
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "run1"}, &blocked); err != nil {
		t.Fatalf("the run vanished before its cordons were released: %v", err)
	}
	if h.node("spark-a").Spec.Unschedulable != true {
		t.Fatal("setup: node was not cordoned")
	}

	h.reconcile("run1")

	if h.node("spark-a").Spec.Unschedulable {
		t.Error("deleting the run stranded its cordon; the fleet has silently lost the node")
	}
	var gone burninv1alpha1.BurnInRun
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "run1"}, &gone); err == nil {
		t.Error("the finalizer was not removed; the run is now undeletable")
	}
	if n := len(h.livePods("run1")); n != 0 {
		t.Errorf("%d pod(s) left burning hardware for a deleted run", n)
	}
	if len(h.forgotten) == 0 {
		t.Error("the deleted run's exported series were not dropped")
	}
	h.assertNoStrandedCordons()
}

// The recovery story. The ownership stamp goes on the node BEFORE the cordon
// and before the status write, so a manager that died anywhere in that sequence
// left evidence on the node and possibly none on the run. Cleanup searches
// nodes by annotation, so whatever the crash interrupted, the next pass finds.
func TestRun_ManagerRestartLeavesNoStrandedCordons(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	if !h.node("spark-a").Spec.Unschedulable || !h.node("spark-b").Spec.Unschedulable {
		t.Fatal("setup: targets were not cordoned")
	}

	// Simulate a crash between cordoning the nodes and writing the run's own
	// record of it: the status forgets, the nodes do not.
	run := h.run("run1")
	run.Status.CordonedNodes = nil
	if err := h.c.Status().Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	if got := h.run("run1").Status.CordonedNodes; len(got) != 0 {
		t.Fatalf("setup: cordonedNodes = %v, want empty", got)
	}

	// The manager comes back as a fresh process holding no state at all.
	h.r = h.newReconciler()

	// The run is abandoned rather than completed.
	run = h.run("run1")
	run.Spec.Cancel = boolp(true)
	run.Spec.CancelPolicy = burninv1alpha1.CancelImmediate
	run.Spec.CancelReason = "aborted after a controller restart"
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.reconcileUntilSettled("run1")

	if got := h.run("run1").Status.Phase; got != burninv1alpha1.RunCancelled {
		t.Fatalf("phase = %q, want Cancelled", got)
	}
	for _, name := range []string{"spark-a", "spark-b"} {
		if h.node(name).Spec.Unschedulable {
			t.Errorf("node %s is still cordoned after the run was abandoned — status alone was never enough to find it", name)
		}
	}
	h.assertNoStrandedCordons()
}

// ─── Prometheus exposition ────────────────────────────────────────────────────

// The exporter is fed by the SAME envelopes every other sink receives, so the
// gauges cannot disagree with the delivered verdict. This checks the wiring at
// both ends of a run's life: series appear while it runs, and go away with the
// object rather than outliving it.
func TestRun_PrometheusSeriesFollowTheRunsLifecycle(t *testing.T) {
	promSink := &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "scrape"},
		Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkPrometheus},
	}
	run := newRun("run1", "acceptance", "spark-a")
	run.Spec.TTLSecondsAfterFinished = int32p(60)

	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		promSink,
		profile("acceptance", []string{"scrape"}, false, testRef("fp4")),
		run,
	)
	// This test drives the package-level exporter, which is what the sink
	// builds into; leave it as it was found.
	metrics.Default.Forget("burnin", "run1")
	t.Cleanup(func() { metrics.Default.Forget("burnin", "run1") })
	// The real exporter, not the harness stub: Forget is the behaviour under test.
	h.r.ForgetMetrics = nil

	h.reconcile("run1")
	if !tracks(metrics.Default, "burnin", "run1") {
		t.Fatal("the Running transition was not exposed; a dashboard would show nothing until the run finished")
	}

	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	// A TTL keeps the run requeueing, so drive to the verdict rather than to
	// a quiescent result.
	for i := 0; i < 25 && h.run("run1").Status.Phase != burninv1alpha1.RunPassed; i++ {
		h.reconcile("run1")
	}
	if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
		t.Fatal("setup: run did not pass")
	}
	if !tracks(metrics.Default, "burnin", "run1") {
		t.Error("a finished run's series were dropped; the verdict is exactly what a dashboard is for")
	}

	h.nowVal = h.nowVal.Add(time.Hour)
	h.reconcile("run1")
	if tracks(metrics.Default, "burnin", "run1") {
		t.Error("series outlived the object they describe")
	}
}

func tracks(e *metrics.Exporter, namespace, name string) bool {
	for _, id := range e.Tracked() {
		if id.Namespace == namespace && id.Name == name {
			return true
		}
	}
	return false
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

// A repeat count is pinned too: raising it mid-run must not move the bar the
// executions already recorded were measured against.
func TestRun_RepeatCountIsPinnedAtStart(t *testing.T) {
	bt := smokeTest("fp4")
	bt.Spec.RepeatCount = int32p(1)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")

	var live burninv1alpha1.BurnInTest
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "fp4"}, &live); err != nil {
		t.Fatal(err)
	}
	live.Spec.RepeatCount = int32p(9)
	if err := h.c.Update(context.Background(), &live); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1")
	h.finishPod(h.pods("run1")["spark-a"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed", run.Status.Phase)
	}
	if got := run.Status.Results[0].RepeatsRequired; got != 1 {
		t.Errorf("repeatsRequired = %d, want the value pinned at start (1)", got)
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
	if run.Status.Results[0].StartedAt != nil {
		t.Errorf("a pod that never started recorded a StartedAt of %v — the wait would read as test time", run.Status.Results[0].StartedAt)
	}
	if len(h.allPods("run1")) != 0 {
		t.Errorf("the stuck pod was not cleaned up")
	}
}

// FailFast abandons the remaining tests; any in-flight pods must be deleted at
// finalize, not left burning the hardware behind a terminal run.
func TestRun_FailFastCleansUpInFlightPods(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, true, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
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
	remaining := map[string]bool{}
	for _, pod := range h.allPods("run1") {
		remaining[pod.Labels[labelNode]] = true
	}
	if remaining["spark-b"] {
		t.Error("failFast left spark-b's pod running behind a terminal run")
	}
	if !remaining["spark-a"] {
		t.Error("the harvested pod should be kept for post-mortem logs")
	}
}

// Delete-and-recreate of a same-name run must not adopt the previous run's
// pods: pod identity includes the run UID.
func TestRun_RecreatedRunDoesNotAdoptOldPods(t *testing.T) {
	first := newRun("run1", "acceptance", "spark-a")
	second := newRun("run1", "acceptance", "spark-a")
	second.UID = types.UID("uid-run1-second")

	if podName(first, 0, "spark-a", 1) == podName(second, 0, "spark-a", 1) {
		t.Fatal("pod identity ignores the run UID — a recreated run would harvest the old run's evidence as its own")
	}
}

// Repeats and retries run the same test on the same node again. Each execution
// needs its own pod, or the second attempt would find the first's terminated
// pod and "confirm" a result the hardware never produced twice.
func TestRun_EachAttemptGetsItsOwnPodIdentity(t *testing.T) {
	run := newRun("run1", "acceptance", "spark-a")
	if podName(run, 0, "spark-a", 1) == podName(run, 0, "spark-a", 2) {
		t.Fatal("pod identity ignores the attempt — a repeat would re-harvest the previous execution")
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
	if err := h.c.Create(context.Background(), cmSink("late-sink", "late-results")); err != nil {
		t.Fatal(err)
	}
	h.reconcile("run1")

	if _, pending := h.run("run1").Annotations[pendingDeliveryAnnotation]; pending {
		t.Error("pending marker not cleared after the sink accepted the terminal envelope")
	}
	found := false
	for _, env := range h.envelopes("late-results") {
		if env.Reason == contract.ReasonPhaseChanged && env.Phase == "Passed" {
			found = true
		}
	}
	if !found {
		t.Error("the delivered envelope is not the terminal Passed phase change")
	}
}

// ─── Pair scope: the two-pod rendezvous ───────────────────────────────────────
//
// A Pair test is ONE execution across TWO nodes. Everything below is written
// against the two ways that can go wrong: launching the client before the
// server can answer it (which manufactures a hardware failure out of a race),
// and attributing a link's verdict to one of its endpoints (which sends an
// engineer to replace the wrong part).

// runPairToStart drives a fresh pair run up to the point where the server pod
// exists and is Ready, and returns the two pods.
func runPairToStart(h *harness) (server, client *corev1.Pod) {
	h.t.Helper()
	h.reconcile("run1") // pin the plan, mark Running
	h.reconcile("run1") // admit the pair, create the Service and the server
	server = h.pairPod("run1", pairRoleServer)
	if server == nil {
		h.t.Fatal("no server pod was created for the pair")
	}
	h.readyPod(server)
	h.reconcile("run1") // the ready gate opens; the client starts
	client = h.pairPod("run1", pairRoleClient)
	if client == nil {
		h.t.Fatal("the client pod was never created after the server became Ready")
	}
	return server, client
}

func newPairHarness(t *testing.T, thresholds ...burninv1alpha1.Threshold) *harness {
	t.Helper()
	return newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		pairTest("ib", thresholds...),
		profile("fabric", nil, false, testRef("ib")),
		pairRun("run1", "fabric", "spark-a", "spark-b"),
	)
}

// Two pods, one per node, each pinned to its own node, wired to each other
// through a headless Service. The env contract is what a fabric runner image
// depends on, so it is asserted literally.
func TestPair_TwoPodsPinnedAndWiredToEachOther(t *testing.T) {
	h := newPairHarness(t)
	server, client := runPairToStart(h)

	if got := server.Spec.NodeSelector[corev1.LabelHostname]; got != "spark-a" {
		t.Errorf("server pinned to %q, want spark-a — the first target is rank 0", got)
	}
	if got := client.Spec.NodeSelector[corev1.LabelHostname]; got != "spark-b" {
		t.Errorf("client pinned to %q, want spark-b", got)
	}
	if server.Name == client.Name {
		t.Fatal("both ends of the pair share a pod name")
	}
	if !strings.Contains(server.Name, "server") || !strings.Contains(client.Name, "client") {
		t.Errorf("pod names do not say which end they are: %q / %q", server.Name, client.Name)
	}

	svcs := h.rendezvousServices("run1")
	if len(svcs) != 1 {
		t.Fatalf("rendezvous services = %d, want exactly 1", len(svcs))
	}
	svc := svcs[0]
	if svc.Spec.ClusterIP != corev1.ClusterIPNone {
		t.Errorf("the rendezvous Service is not headless (clusterIP=%q); per-pod DNS is the whole point", svc.Spec.ClusterIP)
	}

	for _, tc := range []struct {
		pod              *corev1.Pod
		role, hostname   string
		wantPeer         string
		wantPeerNodeName string
	}{
		{server, pairRoleServer, pairRoleServer, "client." + svc.Name + ".burnin.svc", "spark-b"},
		{client, pairRoleClient, pairRoleClient, "server." + svc.Name + ".burnin.svc", "spark-a"},
	} {
		if got := envOf(tc.pod, "BURNIN_ROLE"); got != tc.role {
			t.Errorf("%s BURNIN_ROLE = %q, want %q", tc.role, got, tc.role)
		}
		if got := envOf(tc.pod, "BURNIN_PEER_HOST"); got != tc.wantPeer {
			t.Errorf("%s BURNIN_PEER_HOST = %q, want %q", tc.role, got, tc.wantPeer)
		}
		if got := envOf(tc.pod, "BURNIN_PEER_NODE"); got != tc.wantPeerNodeName {
			t.Errorf("%s BURNIN_PEER_NODE = %q, want %q", tc.role, got, tc.wantPeerNodeName)
		}
		if tc.pod.Spec.Hostname != tc.hostname || tc.pod.Spec.Subdomain != svc.Name {
			t.Errorf("%s has hostname/subdomain %q/%q, want %q/%q — without both there is no A record to resolve",
				tc.role, tc.pod.Spec.Hostname, tc.pod.Spec.Subdomain, tc.hostname, svc.Name)
		}
		if got := envOf(tc.pod, "BURNIN_DURATION_SECONDS"); got == "" {
			t.Errorf("%s lost BURNIN_DURATION_SECONDS; the pair contract is additive", tc.role)
		}
	}
}

// The gate. A client that starts before its server is listening dies with a
// connection error, and a connection error on a fabric test is indistinguishable
// from a broken fabric — so the race must not be possible in the first place.
func TestPair_ClientDoesNotStartUntilTheServerIsReady(t *testing.T) {
	h := newPairHarness(t)
	h.reconcile("run1")
	h.reconcile("run1")

	server := h.pairPod("run1", pairRoleServer)
	if server == nil {
		t.Fatal("no server pod")
	}
	if h.pairPod("run1", pairRoleClient) != nil {
		t.Fatal("the client was created alongside the server; it will race the server's listener")
	}

	// Running but NOT Ready is exactly the window the gate exists for.
	h.startPod(server)
	for i := 0; i < 3; i++ {
		res := h.reconcile("run1")
		if h.pairPod("run1", pairRoleClient) != nil {
			t.Fatal("the client started against a server that is running but not Ready")
		}
		if res.RequeueAfter == 0 && !res.Requeue {
			t.Fatal("the run stopped requeueing mid-rendezvous; nothing would ever start the client")
		}
		if res.RequeueAfter > waitingPollInterval {
			t.Errorf("mid-rendezvous requeue is %v; the pair is holding two nodes idle", res.RequeueAfter)
		}
	}

	h.readyPod(server)
	h.reconcile("run1")
	if h.pairPod("run1", pairRoleClient) == nil {
		t.Fatal("the client never started after the server became Ready")
	}
}

// One verdict, both nodes, and the numbers come from the side that measured
// them. This is the shape a consumer reads a link acceptance out of.
func TestPair_ProducesOneResultCarryingBothNodes(t *testing.T) {
	h := newPairHarness(t, burninv1alpha1.Threshold{
		Metric: "bandwidthGbps", Comparison: burninv1alpha1.GTE, Value: "89",
	})
	server, client := runPairToStart(h)
	h.finishPod(client, 0, ibClientStdout, "Completed")
	h.finishPod(server, 0, ibServerStdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if len(run.Status.Results) != 1 {
		t.Fatalf("a pair produced %d results, want exactly 1: %+v", len(run.Status.Results), run.Status.Results)
	}
	res := run.Status.Results[0]
	if len(res.Nodes) != 2 || res.Nodes[0] != "spark-a" || res.Nodes[1] != "spark-b" {
		t.Errorf("result nodes = %v, want both endpoints of the link", res.Nodes)
	}
	if res.Scope != burninv1alpha1.ScopePair {
		t.Errorf("result scope = %q, want Pair", res.Scope)
	}
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", res.Phase, res.Message)
	}
	if got := res.Metrics["bandwidthGbps"]; got != "99.63" {
		t.Errorf("bandwidthGbps = %q, want the client's measurement 99.63 — metrics come from the side that reports them", got)
	}
	if run.Status.Passed != 1 {
		t.Errorf("passed = %d, want 1 — a pair is one verdict, not one per node", run.Status.Passed)
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run phase = %q, want Passed", run.Status.Phase)
	}
}

// The threshold is evaluated ONCE against the merged metrics, and a shortfall
// indicts the LINK. Nothing in the result may single out an endpoint.
func TestPair_ThresholdFailureIndictsTheLinkNotAnEndpoint(t *testing.T) {
	h := newPairHarness(t, burninv1alpha1.Threshold{
		Metric: "bandwidthGbps", Comparison: burninv1alpha1.GTE, Value: "89",
	})
	server, client := runPairToStart(h)
	h.finishPod(client, 0, "bw_average=41.2\nIB_WRITE_BW_PASS\n", "Completed")
	h.finishPod(server, 0, ibServerStdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if len(run.Status.Results) != 1 {
		t.Fatalf("results = %d, want 1", len(run.Status.Results))
	}
	res := run.Status.Results[0]
	if res.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed: %s", res.Phase, res.Message)
	}
	if len(res.Nodes) != 2 {
		t.Errorf("a failing pair recorded %v — the verdict must name the whole link", res.Nodes)
	}
	if run.Status.Failed != 1 {
		t.Errorf("failed = %d, want 1 — one link, one failure, not one per endpoint", run.Status.Failed)
	}
	for _, other := range run.Status.Results {
		if len(other.Nodes) == 1 {
			t.Errorf("a per-node result appeared beside the pair result: %+v", other)
		}
	}
}

// A one-sided MACHINERY failure is an Error, never a Fail. The client's
// "connection refused" here is an artifact of its peer dying, not evidence
// about the fabric — and Fail is the one phase this operator never retries.
func TestPair_MachineryFailureOnOneSideIsErrorNotFail(t *testing.T) {
	h := newPairHarness(t)
	server, client := runPairToStart(h)
	// The server dies out of contract; the client then cannot connect and
	// reports a plain failure.
	h.finishPod(server, 137, "OOMKilled\n", "OOMKilled")
	h.finishPod(client, 1, "Unable to connect the QPs\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	res := run.Status.Results[0]
	if res.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error — machinery broke on one end, so the link was never measured: %s", res.Phase, res.Message)
	}
	if run.Status.Failed != 0 {
		t.Errorf("failed = %d; a rendezvous failure was recorded as a hardware verdict", run.Status.Failed)
	}
	if !strings.Contains(res.Message, "spark-a") || !strings.Contains(res.Message, "spark-b") {
		t.Errorf("the message does not report both ends: %q", res.Message)
	}
	if !strings.Contains(res.Message, "LINK") {
		t.Errorf("the message does not say what was judged: %q", res.Message)
	}
}

// A real link failure still fails. The precedence rule must not turn every
// disagreement into an Error, or nothing would ever fail a fabric.
func TestPair_ClientFailureWithAHealthyServerFailsTheLink(t *testing.T) {
	h := newPairHarness(t)
	server, client := runPairToStart(h)
	h.finishPod(server, 0, ibServerStdout, "Completed")
	h.finishPod(client, 1, "bw_average=0.4\nIB_WRITE_BW_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if got := run.Status.Results[0].Phase; got != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed: %s", got, run.Status.Results[0].Message)
	}
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Errorf("run phase = %q, want Failed", run.Status.Phase)
	}
}

// An endpoint that positively declares the test inapplicable (exit 2) makes the
// PAIR inapplicable. The client is never started, because there is nothing to
// connect to, and Skipped is not a failure.
func TestPair_ServerSkipSettlesThePairAsSkipped(t *testing.T) {
	h := newPairHarness(t)
	h.reconcile("run1")
	h.reconcile("run1")
	server := h.pairPod("run1", pairRoleServer)
	h.finishPod(server, 2, "no RDMA device on this host\n", "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if got := run.Status.Results[0].Phase; got != burninv1alpha1.RunSkipped {
		t.Fatalf("phase = %q, want Skipped: %s", got, run.Status.Results[0].Message)
	}
	if h.pairPod("run1", pairRoleClient) != nil {
		t.Error("a client was launched against a server that had already skipped")
	}
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Errorf("run phase = %q, want Passed — a skip is not a failure", run.Status.Phase)
	}
}

// The subtle one. A server that exits 0 before the client ever ran has measured
// NOTHING: no traffic crossed the link. Reporting that as a pass would certify
// a fabric nobody tested, so it is an Error.
func TestPair_ServerExitingCleanBeforeTheClientIsNotAPass(t *testing.T) {
	h := newPairHarness(t)
	h.reconcile("run1")
	h.reconcile("run1")
	server := h.pairPod("run1", pairRoleServer)
	h.finishPod(server, 0, ibServerStdout, "Completed")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	res := run.Status.Results[0]
	if res.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error — the link was never exercised: %s", res.Phase, res.Message)
	}
	if run.Status.Passed != 0 {
		t.Errorf("passed = %d; hardware was certified without a measurement", run.Status.Passed)
	}
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Errorf("run phase = %q, want Error", run.Status.Phase)
	}
}

// A pair needs exactly two distinct endpoints, and the refusal has to say what
// it actually got — a run that quietly measured something else would be worse
// than one that refused.
func TestPair_WrongTargetCardinalityIsRefusedWithTheCount(t *testing.T) {
	cases := []struct {
		name    string
		nodes   []string
		wantMsg string
	}{
		{"one node", []string{"spark-a"}, "resolved to 1"},
		{"three nodes", []string{"spark-a", "spark-b", "spark-c"}, "resolved to 3"},
		{"the same node twice", []string{"spark-a", "spark-a"}, "DISTINCT"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t,
				gb10Node("spark-a"), gb10Node("spark-b"), gb10Node("spark-c"),
				pairTest("ib"),
				profile("fabric", nil, false, testRef("ib")),
				withNodeCap(newRun("run1", "fabric", tc.nodes...), 3),
			)
			h.reconcileUntilSettled("run1")

			run := h.run("run1")
			if run.Status.Phase != burninv1alpha1.RunError {
				t.Fatalf("phase = %q, want Error", run.Status.Phase)
			}
			msg := run.Status.Results[0].Message
			if !strings.Contains(msg, tc.wantMsg) {
				t.Errorf("message = %q, want it to explain %q", msg, tc.wantMsg)
			}
			if !strings.Contains(msg, "ib") {
				t.Errorf("message does not name the offending test: %q", msg)
			}
			if len(h.allPods("run1")) != 0 {
				t.Error("a pod was scheduled for a pair that cannot be formed")
			}
			h.assertNoStrandedCordons()
		})
	}
}

// A pair holds BOTH of its nodes at once, so it needs two of the interlock's
// slots. At the default cap of 1 it can never start, and the run must say so at
// start rather than wait out its deadline discovering it.
func TestPair_RefusedAtTheDefaultConcurrencyCap(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		pairTest("ib"),
		profile("fabric", nil, false, testRef("ib")),
		newRun("run1", "fabric", "spark-a", "spark-b"), // cap defaults to 1
	)
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	msg := run.Status.Results[0].Message
	if !strings.Contains(msg, "maxConcurrentNodes") {
		t.Errorf("message = %q, want it to name the field to change", msg)
	}
	if len(h.allPods("run1")) != 0 {
		t.Error("a pair pod was scheduled under a cap that cannot admit the pair")
	}
	h.assertNoStrandedCordons()
}

// Both nodes of a pair are cordoned together and both are given back, whatever
// route the run takes out.
func TestPair_CordonsBothNodesAndRestoresThem(t *testing.T) {
	t.Run("on success", func(t *testing.T) {
		h := newPairHarness(t)
		server, client := runPairToStart(h)
		if got := h.cordonedNodes(); len(got) != 2 {
			t.Fatalf("cordoned = %v, want both endpoints — a pair under test occupies both", got)
		}
		h.finishPod(client, 0, ibClientStdout, "Completed")
		h.finishPod(server, 0, ibServerStdout, "Completed")
		h.reconcileUntilSettled("run1")

		if h.run("run1").Status.Phase != burninv1alpha1.RunPassed {
			t.Fatal("setup: the pair did not pass")
		}
		h.assertNoStrandedCordons()
	})

	t.Run("on failure", func(t *testing.T) {
		h := newPairHarness(t)
		server, client := runPairToStart(h)
		h.finishPod(server, 0, ibServerStdout, "Completed")
		h.finishPod(client, 1, "IB_WRITE_BW_FAIL\n", "Error")
		h.reconcileUntilSettled("run1")

		if h.run("run1").Status.Phase != burninv1alpha1.RunFailed {
			t.Fatal("setup: the pair did not fail")
		}
		for _, n := range []string{"spark-a", "spark-b"} {
			if h.node(n).Spec.Unschedulable {
				t.Errorf("node %s stayed cordoned after a failing pair; a bad verdict is not a reason to hold capacity", n)
			}
		}
		h.assertNoStrandedCordons()
	})

	t.Run("on abort", func(t *testing.T) {
		h := newPairHarness(t)
		runPairToStart(h)
		if got := h.cordonedNodes(); len(got) != 2 {
			t.Fatalf("setup: cordoned = %v, want 2", got)
		}

		run := h.run("run1")
		if err := h.c.Delete(context.Background(), run); err != nil {
			t.Fatal(err)
		}
		h.reconcile("run1")

		for _, n := range []string{"spark-a", "spark-b"} {
			if h.node(n).Spec.Unschedulable {
				t.Errorf("deleting the run stranded node %s — half a pair is as lost as a whole one", n)
			}
		}
		if n := len(h.livePods("run1")); n != 0 {
			t.Errorf("%d pod(s) still burning hardware for a deleted run", n)
		}
		h.assertNoStrandedCordons()
	})
}

// Reconciling repeatedly must converge, not accumulate. The rendezvous adds two
// new objects per attempt — a second pod and a Service — and both have to be
// found rather than recreated.
func TestPair_ReconcileIsIdempotent(t *testing.T) {
	h := newPairHarness(t)
	server, client := runPairToStart(h)

	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}
	if n := len(h.allPods("run1")); n != 2 {
		t.Fatalf("pods = %d after repeated reconciles, want exactly 2", n)
	}
	if n := len(h.rendezvousServices("run1")); n != 1 {
		t.Fatalf("rendezvous services = %d, want exactly 1", n)
	}

	// A manager restart re-derives everything from the cluster; it must adopt
	// the pods in flight rather than start the test again on hardware that is
	// already being burned.
	h.r = h.newReconciler()
	h.reconcile("run1")
	if n := len(h.allPods("run1")); n != 2 {
		t.Fatalf("pods = %d after a manager restart, want 2", n)
	}
	if got := h.pairPod("run1", pairRoleServer); got.Name != server.Name {
		t.Errorf("server pod identity changed across a restart: %q -> %q", server.Name, got.Name)
	}
	if got := h.pairPod("run1", pairRoleClient); got.Name != client.Name {
		t.Errorf("client pod identity changed across a restart: %q -> %q", client.Name, got.Name)
	}
}

// A recreated run of the same name must not adopt the previous incarnation's
// rendezvous, or it would report the old execution's evidence as its own.
func TestPair_RecreatedRunGetsAFreshRendezvous(t *testing.T) {
	first := newRun("run1", "fabric", "spark-a", "spark-b")
	second := newRun("run1", "fabric", "spark-a", "spark-b")
	second.UID = types.UID("uid-run1-second")

	if pairServiceName(first, 0, 1) == pairServiceName(second, 0, 1) {
		t.Error("the rendezvous Service name ignores the run UID; a recreated run would inherit the old one's endpoints")
	}
	if podNameForRole(first, 0, "spark-a", 1, pairRoleServer) == podNameForRole(second, 0, "spark-a", 1, pairRoleServer) {
		t.Error("pair pod identity ignores the run UID")
	}
	if podNameForRole(first, 0, "spark-a", 1, pairRoleServer) == podNameForRole(first, 0, "spark-a", 1, pairRoleClient) {
		t.Error("the two ends of one pair share a pod name")
	}
	if pairServiceName(first, 0, 1) == pairServiceName(first, 0, 2) {
		t.Error("a retry reuses the previous attempt's Service; its selector could still resolve a dead server")
	}
}

// ─── The cordon must not lock out the burn-in itself ──────────────────────────

// Observed on a live two-node GB10 cluster: the run cordoned its target and the
// scheduler then refused the runner pod ("0/2 nodes are available: 2 node(s)
// were unschedulable"), so the run waited out its deadline and reported Error —
// every test, every node, always. A burn-in runner is precisely the workload
// that should run on a node this operator cordoned.
//
// The fake client never runs the scheduler, so only an explicit assertion
// catches this.
func TestRun_RunnerPodToleratesTheOperatorsOwnCordon(t *testing.T) {
	t.Run("node scope", func(t *testing.T) {
		h := newHarness(t,
			gb10Node("spark-a"),
			smokeTest("fp4"),
			profile("acceptance", nil, false, testRef("fp4")),
			newRun("run1", "acceptance", "spark-a"),
		)
		h.reconcile("run1")
		h.reconcile("run1")
		pod := h.pods("run1")["spark-a"]
		if pod == nil {
			t.Fatal("no pod was created")
		}
		if !tolerates(pod, corev1.TaintNodeUnschedulable, corev1.TaintEffectNoSchedule) {
			t.Fatalf("the runner pod does not tolerate %s:NoSchedule — it can never be scheduled onto the node this run just cordoned: %+v",
				corev1.TaintNodeUnschedulable, pod.Spec.Tolerations)
		}
	})

	t.Run("pair scope", func(t *testing.T) {
		h := newPairHarness(t)
		server, client := runPairToStart(h)
		for _, pod := range []*corev1.Pod{server, client} {
			if !tolerates(pod, corev1.TaintNodeUnschedulable, corev1.TaintEffectNoSchedule) {
				t.Errorf("pair pod %s does not tolerate the operator's own cordon: %+v", pod.Name, pod.Spec.Tolerations)
			}
		}
	})

	t.Run("the target's own tolerations survive", func(t *testing.T) {
		run := newRun("run1", "acceptance", "spark-a")
		run.Spec.Target.Tolerations = []corev1.Toleration{{
			Key: "nvidia.com/gpu", Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule,
		}}
		h := newHarness(t,
			gb10Node("spark-a"),
			smokeTest("fp4"),
			profile("acceptance", nil, false, testRef("fp4")),
			run,
		)
		h.reconcile("run1")
		h.reconcile("run1")
		pod := h.pods("run1")["spark-a"]
		if !tolerates(pod, "nvidia.com/gpu", corev1.TaintEffectNoSchedule) {
			t.Error("the target's own toleration was dropped")
		}
		if !tolerates(pod, corev1.TaintNodeUnschedulable, corev1.TaintEffectNoSchedule) {
			t.Error("the cordon toleration was dropped")
		}
	})
}

// ─── The cordon footprint is the wave, not the target list ────────────────────

// maxConcurrentNodes is a facility interlock: only N nodes are supposed to be
// occupied at once. Cordoning every target up front made the run's footprint the
// size of its target list instead — which on the two-node cluster this was found
// on took the entire cluster out of scheduling for the whole run.
func TestRun_CordonsOnlyTheCurrentWave(t *testing.T) {
	h := newHarness(t,
		gb10Node("n1"), gb10Node("n2"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "n1", "n2"), // cap defaults to 1
	)

	sawOne := false
	for i := 0; i < 20; i++ {
		res := h.reconcile("run1")
		switch n := len(h.cordonedNodes()); {
		case n > 1:
			t.Fatalf("pass %d holds %d nodes cordoned at maxConcurrentNodes=1 (%v) — the interlock says one node at a time, and on a two-node cluster this is the whole cluster",
				i, n, h.cordonedNodes())
		case n == 1:
			sawOne = true
		}
		for _, pod := range h.livePods("run1") {
			h.finishPod(pod, 0, fp4Stdout, "Completed")
		}
		if !res.Requeue && res.RequeueAfter == 0 {
			break
		}
	}

	if !sawOne {
		t.Error("no node was ever cordoned; the run never held the hardware it was measuring")
	}
	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunPassed || run.Status.Passed != 2 {
		t.Fatalf("phase = %q, passed = %d, want Passed/2: %+v", run.Status.Phase, run.Status.Passed, run.Status.Results)
	}
	h.assertNoStrandedCordons()
}

// A node the run has finished with goes back to the fleet immediately, not at
// run teardown — which for a long sweep is hours later.
func TestRun_ReleasesANodeAsSoonAsItsWorkIsDone(t *testing.T) {
	h := newHarness(t,
		gb10Node("n1"), gb10Node("n2"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "n1", "n2"), 2),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	if got := h.cordonedNodes(); len(got) != 2 {
		t.Fatalf("cordoned = %v, want both nodes under load at a cap of 2", got)
	}

	// One node finishes; the other is still burning.
	h.finishPod(h.pods("run1")["n1"], 0, fp4Stdout, "Completed")
	h.reconcile("run1")
	h.reconcile("run1")

	if h.node("n1").Spec.Unschedulable {
		t.Error("n1 finished its work and is still held out of the scheduler")
	}
	if !h.node("n2").Spec.Unschedulable {
		t.Error("n2 is still under test and was released early; a workload can now land beside the burn-in")
	}
	if run := h.run("run1"); run.Status.Phase != burninv1alpha1.RunRunning {
		t.Errorf("phase = %q, want Running — the run is not over", run.Status.Phase)
	}

	h.finishPod(h.pods("run1")["n2"], 0, fp4Stdout, "Completed")
	h.reconcileUntilSettled("run1")
	h.assertNoStrandedCordons()
}

// ─── Host mounts ──────────────────────────────────────────────────────────────
//
// Measured on a live two-node GB10 cluster: `privileged: true` plus
// `hostNetwork: true` does NOT put the host's RDMA device nodes in the pod. A
// privileged hostNetwork pod saw only `rdma_cm umad0..3` under /dev/infiniband
// while the host also had `uverbs0..3`, and uverbs is what ibv_create_cq opens —
// so ib_write_bw died with "Couldn't create CQ" and nccl failed the same way.
// The same gap kept host-health from reading /dev/kmsg, which made it report
// xid_source=none and omit xidEvents, so any gate on that metric failed closed
// and certified nothing.
//
// RunnerSpec.HostPaths is the fix, and these are its load-bearing properties:
// the mount reaches the pod, it reaches BOTH ends of a pair, read-only means
// read-only, the spec is pinned so a mid-run edit cannot change it, and asking
// for nothing grants nothing.

// hostMount is one declared host mount, written the way a profile author writes
// the common case: two paths, and no opinion about read-only.
func hostMount(path, mountPath string) burninv1alpha1.HostPathMount {
	return burninv1alpha1.HostPathMount{Path: path, MountPath: mountPath}
}

// withHostPaths declares host mounts on a test, creating the runner block if the
// fixture has none — which is what an author does to a test whose image they are
// not overriding.
func withHostPaths(bt *burninv1alpha1.BurnInTest, mounts ...burninv1alpha1.HostPathMount) *burninv1alpha1.BurnInTest {
	if bt.Spec.Runner == nil {
		bt.Spec.Runner = &burninv1alpha1.RunnerSpec{}
	}
	bt.Spec.Runner.HostPaths = mounts
	return bt
}

// mountOf resolves one container mount path to the volumeMount that provides it
// AND the pod volume that volumeMount names.
//
// Both halves are asserted together on purpose: a volumeMount pointing at a
// volume that does not exist is an invalid pod, and a volume nothing mounts
// grants the runner nothing at all. Either half alone would let a broken pod
// pass this test.
func mountOf(t *testing.T, pod *corev1.Pod, mountPath string) (corev1.VolumeMount, corev1.Volume) {
	t.Helper()
	var vm corev1.VolumeMount
	found := false
	for _, m := range pod.Spec.Containers[0].VolumeMounts {
		if m.MountPath == mountPath {
			vm, found = m, true
			break
		}
	}
	if !found {
		t.Fatalf("pod %s has no mount at %q — the runner cannot see the host path: %+v",
			pod.Name, mountPath, pod.Spec.Containers[0].VolumeMounts)
	}
	for _, v := range pod.Spec.Volumes {
		if v.Name == vm.Name {
			return vm, v
		}
	}
	t.Fatalf("pod %s mounts volume %q at %q but declares no such volume: %+v",
		pod.Name, vm.Name, mountPath, pod.Spec.Volumes)
	return vm, corev1.Volume{}
}

// A declared host path reaches the runner container as a hostPath volume, with
// the host path, the mount point and the asserted type all intact.
func TestRun_HostPathsReachTheRunnerPod(t *testing.T) {
	charDev := corev1.HostPathCharDev
	bt := withHostPaths(healthTest("host-health"), burninv1alpha1.HostPathMount{
		Path: "/dev/kmsg", MountPath: "/dev/kmsg", Type: &charDev,
	})
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("host-health")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	pod := h.pods("run1")["spark-a"]
	if pod == nil {
		t.Fatal("no pod was created")
	}
	vm, vol := mountOf(t, pod, "/dev/kmsg")
	if vol.HostPath == nil {
		t.Fatalf("volume %q is not a hostPath volume: %+v", vol.Name, vol.VolumeSource)
	}
	if vol.HostPath.Path != "/dev/kmsg" {
		t.Errorf("host path = %q, want /dev/kmsg — the pod is reading the wrong thing off the node", vol.HostPath.Path)
	}
	if vol.HostPath.Type == nil || *vol.HostPath.Type != corev1.HostPathCharDev {
		t.Errorf("hostPath type = %v, want CharDevice — without the assertion the runtime may create the path on the host", vol.HostPath.Type)
	}
	if vm.Name != vol.Name {
		t.Errorf("mount names volume %q, pod declares %q", vm.Name, vol.Name)
	}
}

// A pair is one measurement across two pods, so both ends need the devices. A
// mount that reached only the server would leave the client — the side that
// actually measures — without the verbs nodes, which is the exact failure this
// field was added for.
func TestPair_HostPathsReachBothEndsOfTheLink(t *testing.T) {
	dir := corev1.HostPathDirectory
	writable := false
	bt := pairTest("ib")
	bt.Spec.Runner.HostPaths = []burninv1alpha1.HostPathMount{{
		Path: "/dev/infiniband", MountPath: "/dev/infiniband", ReadOnly: &writable, Type: &dir,
	}}
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		bt,
		profile("fabric", nil, false, testRef("ib")),
		pairRun("run1", "fabric", "spark-a", "spark-b"),
	)
	server, client := runPairToStart(h)

	for _, pod := range []*corev1.Pod{server, client} {
		role := pod.Labels[labelPairRole]
		vm, vol := mountOf(t, pod, "/dev/infiniband")
		if vol.HostPath == nil || vol.HostPath.Path != "/dev/infiniband" {
			t.Errorf("%s end: hostPath = %+v, want /dev/infiniband — this end cannot open uverbs and the link is not measured",
				role, vol.VolumeSource)
		}
		if vm.ReadOnly {
			t.Errorf("%s end: /dev/infiniband mounted read-only — ibv_open_device opens the verbs nodes read-write, so this fails in the same place as no mount at all",
				role)
		}
	}
}

// readOnly defaults to TRUE, and a Go-built object must get that default too:
// the reconciler cannot assume apiserver defaulting has happened, and a nil that
// fell through as "false" would hand out write access to a host path because
// nobody mentioned it.
func TestRun_HostPathReadOnlyDefaultsToTrueAndFalseIsHonoured(t *testing.T) {
	writable := false
	readOnly := true
	bt := withHostPaths(healthTest("host-health"),
		hostMount("/dev/kmsg", "/dev/kmsg"), // unset: must come out read-only
		burninv1alpha1.HostPathMount{Path: "/dev/infiniband", MountPath: "/dev/infiniband", ReadOnly: &writable},
		burninv1alpha1.HostPathMount{Path: "/var/log", MountPath: "/host/var/log", ReadOnly: &readOnly},
	)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("host-health")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	pod := h.pods("run1")["spark-a"]
	for _, tc := range []struct {
		mountPath string
		want      bool
		why       string
	}{
		{"/dev/kmsg", true, "readOnly was left unset, and the default for a host mount has to fall towards the harmless form"},
		{"/dev/infiniband", false, "readOnly: false was declared explicitly because the verbs nodes are opened read-write"},
		{"/host/var/log", true, "readOnly: true was declared explicitly"},
	} {
		vm, _ := mountOf(t, pod, tc.mountPath)
		if vm.ReadOnly != tc.want {
			t.Errorf("%s: readOnly = %v, want %v — %s", tc.mountPath, vm.ReadOnly, tc.want, tc.why)
		}
	}
}

// The mount spec is part of the pinned plan, so a mid-run edit cannot change
// what a running test takes off the host.
//
// This is the same hermeticity the rest of the spec has, and it matters more
// here than anywhere else: without it, editing a BurnInTest while a run is in
// flight would silently widen the host access of the very next attempt — the
// escalation would arrive with no new object, no new run, and no audit trail.
func TestRun_HostPathsArePinnedAgainstAMidRunEdit(t *testing.T) {
	bt := withHostPaths(healthTest("host-health"), hostMount("/dev/kmsg", "/dev/kmsg"))
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("host-health")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1") // pins the plan; no pod yet

	// Sabotage: widen the mount to the whole host filesystem, writable.
	var live burninv1alpha1.BurnInTest
	if err := h.c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "host-health"}, &live); err != nil {
		t.Fatal(err)
	}
	writable := false
	live.Spec.Runner.HostPaths = []burninv1alpha1.HostPathMount{
		{Path: "/", MountPath: "/host", ReadOnly: &writable},
	}
	if err := h.c.Update(context.Background(), &live); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1") // creates the pod, from the PINNED spec

	pod := h.pods("run1")["spark-a"]
	if pod == nil {
		t.Fatal("no pod was created")
	}
	if len(pod.Spec.Volumes) != 1 {
		t.Fatalf("pod has %d volumes, want the 1 that was pinned at start: %+v", len(pod.Spec.Volumes), pod.Spec.Volumes)
	}
	if _, vol := mountOf(t, pod, "/dev/kmsg"); vol.HostPath.Path != "/dev/kmsg" {
		t.Errorf("host path = %q, want the pinned /dev/kmsg", vol.HostPath.Path)
	}
	for _, m := range pod.Spec.Containers[0].VolumeMounts {
		if m.MountPath == "/host" {
			t.Fatal("the mid-run edit reached a running test: the pod mounts the host root, which nobody authorised for this run")
		}
	}
}

// A test that declares no host access gets a pod with NO volumes. There is no
// implicit /dev, no convenience /sys, nothing: a host mount nobody wrote down is
// a host mount nobody reviewed, and this is the assertion that keeps it that way
// as pod construction grows.
func TestRun_NoHostPathsDeclaredGrantsNoHostAccess(t *testing.T) {
	for _, tc := range []struct {
		name string
		test *burninv1alpha1.BurnInTest
	}{
		{"no runner block at all", smokeTest("fp4")},
		{"a runner block that declares no host paths", func() *burninv1alpha1.BurnInTest {
			bt := smokeTest("fp4")
			bt.Spec.Runner = &burninv1alpha1.RunnerSpec{Privileged: true}
			return bt
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t,
				gb10Node("spark-a"),
				tc.test,
				profile("acceptance", nil, false, testRef("fp4")),
				newRun("run1", "acceptance", "spark-a"),
			)
			h.reconcile("run1")
			h.reconcile("run1")

			pod := h.pods("run1")["spark-a"]
			if pod == nil {
				t.Fatal("no pod was created")
			}
			if len(pod.Spec.Volumes) != 0 {
				t.Errorf("pod has %d volumes without asking for any: %+v", len(pod.Spec.Volumes), pod.Spec.Volumes)
			}
			if len(pod.Spec.Containers[0].VolumeMounts) != 0 {
				t.Errorf("pod has %d volume mounts without asking for any: %+v",
					len(pod.Spec.Containers[0].VolumeMounts), pod.Spec.Containers[0].VolumeMounts)
			}
		})
	}
}

// Two mounts at one container path is an invalid pod spec, which the apiserver
// refuses on every Create — so the run would retry the same rejection forever
// while holding a cordon. It is refused at START instead, with the offending
// path named.
func TestRun_DuplicateHostMountPathIsRefusedAtStart(t *testing.T) {
	bt := withHostPaths(healthTest("host-health"),
		hostMount("/dev/kmsg", "/host/log"),
		hostMount("/var/log/kern.log", "/host/log"),
	)
	h := newHarness(t,
		gb10Node("spark-a"),
		bt,
		profile("acceptance", nil, false, testRef("host-health")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	msg := run.Status.Results[0].Message
	if !strings.Contains(msg, "/host/log") {
		t.Errorf("message = %q, want it to name the duplicated mount point", msg)
	}
	if len(h.allPods("run1")) != 0 {
		t.Error("a pod was created for a test whose mounts cannot be admitted")
	}
	h.assertNoStrandedCordons()
}

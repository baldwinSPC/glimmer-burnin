package controller

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// artifactStdout wraps a payload in the fence a runner would print.
func artifactStdout(name, mediaType, payload string) string {
	return "-----BEGIN BURNIN ARTIFACT " + name + " " + mediaType + "-----\n" +
		payload + "\n-----END BURNIN ARTIFACT-----\n"
}

func (h *harness) artifactConfigMap(runName string) *corev1.ConfigMap {
	h.t.Helper()
	cm := &corev1.ConfigMap{}
	err := h.c.Get(context.Background(),
		types.NamespacedName{Namespace: "burnin", Name: runName + artifactConfigMapSuffix}, cm)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		h.t.Fatal(err)
	}
	return cm
}

// Evidence a runner returned reaches a consumer, by reference, and the payload
// is durable in a ConfigMap the run owns — issue #143.
func TestArtifacts_ReachTheResultAndTheConfigMap(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	const payload = `{"health": {"overall": "Pass"}}`

	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	h.finishPod(pod, 0, "nonfiniteCount=0\n"+artifactStdout("dcgmi.json", "application/json", payload)+"DONE\n", "")
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", res.Phase, res.Message)
	}
	if len(res.Artifacts) != 1 {
		t.Fatalf("got %d artifact refs, want 1: %+v", len(res.Artifacts), res.Artifacts)
	}
	ref := res.Artifacts[0]
	if ref.Dropped != "" {
		t.Fatalf("the artifact was dropped: %s", ref.Dropped)
	}
	if ref.Name != "dcgmi.json" || ref.MediaType != "application/json" {
		t.Errorf("ref = %+v, want the runner's own name and media type", ref)
	}
	if !strings.HasPrefix(ref.Digest, "sha256:") {
		t.Errorf("ref has no digest, so a consumer cannot verify what it fetched: %+v", ref)
	}
	if ref.ConfigMap != "run1"+artifactConfigMapSuffix || ref.Key == "" {
		t.Fatalf("ref does not point anywhere: %+v", ref)
	}
	// The pod name is in the key, which is what keeps a repeat from overwriting
	// the attempt before it and a Pair's two ends apart.
	if !strings.Contains(ref.Key, pod.Name) {
		t.Errorf("key %q does not name the pod that produced it", ref.Key)
	}

	cm := h.artifactConfigMap("run1")
	if cm == nil {
		t.Fatal("no artifact ConfigMap was written — the ref points at nothing")
	}
	if got := cm.Data[ref.Key]; !strings.Contains(got, `"overall": "Pass"`) {
		t.Errorf("ConfigMap[%q] = %q, want the payload", ref.Key, got)
	}
	// Owned by the run, so the evidence is collected exactly when the verdict
	// it explains is.
	if ors := cm.OwnerReferences; len(ors) != 1 || ors[0].Name != "run1" || ors[0].Kind != "BurnInRun" {
		t.Errorf("artifact ConfigMap is not owned by the run: %+v", ors)
	}
}

// The metric a threshold gates on must survive a payload that spells it
// differently — the runner-tier rule, asserted here through the RECONCILER,
// because that is where it turns into a verdict about a node.
func TestArtifacts_PayloadCannotRewriteAVerdict(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4", burninv1alpha1.Threshold{
			Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	// The runner measured zero. Its own diagnostic dump says 41, and that
	// number must never become the one the gate is applied to.
	h.finishPod(pod, 0, "nonfiniteCount=0\n"+
		artifactStdout("dump.txt", "text/plain", "nonfiniteCount=41\neccErrors=99")+"DONE\n", "")
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — a payload line failed a healthy node: %s",
			res.Phase, res.Message)
	}
	if res.Metrics["nonfiniteCount"] != "0" {
		t.Errorf("nonfiniteCount = %q, want 0 — the artifact payload overwrote the measurement",
			res.Metrics["nonfiniteCount"])
	}
	if v, ok := res.Metrics["eccErrors"]; ok {
		t.Errorf("eccErrors=%q came out of an artifact payload", v)
	}
}

// An artifact name that cannot be a ConfigMap key is REFUSED, and refusing it
// must not cost the other artifacts in the same write.
//
// The name comes from a runner's stdout, which is the least trusted input this
// operator has. A '/' in a key makes the apiserver reject the whole object, so
// without this one runner's malformed name would silently discard another
// test's evidence.
func TestArtifacts_AnUnusableNameIsRefusedWithoutLosingTheRest(t *testing.T) {
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
	h.finishPod(pod, 0, "nonfiniteCount=0\n"+
		artifactStdout("../../etc/passwd", "text/plain", "escaped")+
		artifactStdout("good.json", "application/json", `{"kept": true}`)+"DONE\n", "")
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("a malformed artifact name changed the verdict to %q: %s", res.Phase, res.Message)
	}
	if len(res.Artifacts) != 2 {
		t.Fatalf("got %d refs, want 2 (one refused, one kept): %+v", len(res.Artifacts), res.Artifacts)
	}

	bad, good := res.Artifacts[0], res.Artifacts[1]
	if bad.Dropped == "" || bad.Key != "" {
		t.Errorf("a name with '/' in it was accepted as a ConfigMap key: %+v", bad)
	}
	if good.Dropped != "" || good.Key == "" {
		t.Fatalf("a well-named artifact was lost alongside the bad one: %+v", good)
	}

	cm := h.artifactConfigMap("run1")
	if cm == nil {
		t.Fatal("the whole write was lost")
	}
	for k := range cm.Data {
		if strings.Contains(k, "/") || strings.Contains(k, "..") {
			t.Errorf("ConfigMap key %q escapes the key grammar", k)
		}
	}
	if cm.Data[good.Key] == "" {
		t.Errorf("the good artifact is not in the ConfigMap")
	}
}

// A failure to STORE evidence must never change a verdict. An artifact is
// evidence about a verdict and never part of one.
func TestArtifacts_AStorageFailureDoesNotTouchTheVerdict(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4", burninv1alpha1.Threshold{
			Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"}),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	// Every write to the artifact ConfigMap fails, as it would under an
	// exhausted quota or a namespace being torn down.
	h.r.Client = refusingArtifactWrites{h.c}

	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	h.finishPod(pod, 0, "nonfiniteCount=0\n"+
		artifactStdout("dcgmi.json", "application/json", `{"a":1}`)+"DONE\n", "")
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — a ConfigMap write failure became a hardware "+
			"verdict: %s", res.Phase, res.Message)
	}
	if len(res.Artifacts) != 1 {
		t.Fatalf("got %d refs, want 1 recording the failure: %+v", len(res.Artifacts), res.Artifacts)
	}
	ref := res.Artifacts[0]
	if ref.Dropped == "" {
		t.Fatal("the write failed but the ref claims the payload was stored — a reader " +
			"sent to that key finds nothing, which reads as data loss")
	}
	if ref.ConfigMap != "" || ref.Key != "" || ref.Digest != "" {
		t.Errorf("a dropped ref still points somewhere: %+v", ref)
	}
	if ref.Name != "dcgmi.json" {
		t.Errorf("the refusal does not say WHAT was lost: %+v", ref)
	}
}

// The same pod is harvested on every reconcile until it is deleted. Its
// artifacts must be recorded once.
func TestArtifacts_ReHarvestDoesNotDuplicateRefs(t *testing.T) {
	res := &burninv1alpha1.TestResult{Name: "fp4"}
	refs := []burninv1alpha1.ArtifactRef{
		{Name: "a.json", Key: "fp4.pod.a.json", ConfigMap: "run1-artifacts"},
		{Name: "b.bin", Dropped: "too large"},
	}
	for i := 0; i < 5; i++ {
		recordArtifacts(res, refs)
	}
	if len(res.Artifacts) != 2 {
		t.Fatalf("after 5 harvests the result carries %d refs, want 2 — a lingering pod "+
			"grows the status without bound: %+v", len(res.Artifacts), res.Artifacts)
	}
}

// Refs ACCUMULATE across attempts, unlike Metrics. The payload of an earlier
// attempt is already in the ConfigMap and stays there; dropping its ref would
// not reclaim a byte, it would only make stored evidence unreachable.
func TestArtifacts_RefsFromEarlierAttemptsSurvive(t *testing.T) {
	res := &burninv1alpha1.TestResult{Name: "fp4"}
	recordArtifacts(res, []burninv1alpha1.ArtifactRef{
		{Name: "dump.txt", Key: "fp4.pod-attempt-1.dump.txt", ConfigMap: "run1-artifacts"}})
	recordArtifacts(res, []burninv1alpha1.ArtifactRef{
		{Name: "dump.txt", Key: "fp4.pod-attempt-2.dump.txt", ConfigMap: "run1-artifacts"}})

	if len(res.Artifacts) != 2 {
		t.Fatalf("got %d refs, want both attempts: %+v", len(res.Artifacts), res.Artifacts)
	}
	if res.Artifacts[0].Key == res.Artifacts[1].Key {
		t.Error("the two attempts collided onto one key")
	}
}

// The run's budget is enforced, and exhausting it is RECORDED. A report that
// quietly omits evidence is indistinguishable from a runner that produced none.
func TestArtifacts_RunBudgetIsEnforcedAndSaidOutLoud(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	// Pre-fill the run's ConfigMap to just under the budget.
	if err := h.c.Create(context.Background(), &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1" + artifactConfigMapSuffix},
		Data:       map[string]string{"filler": strings.Repeat("x", maxRunArtifactBytes-100)},
	}); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	h.startPod(pod)
	h.finishPod(pod, 0, "nonfiniteCount=0\n"+
		artifactStdout("late.json", "application/json", strings.Repeat("y", 4096))+"DONE\n", "")
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("an exhausted artifact budget became a verdict: %q %s", res.Phase, res.Message)
	}
	if len(res.Artifacts) != 1 {
		t.Fatalf("the refusal was not recorded at all: %+v", res.Artifacts)
	}
	ref := res.Artifacts[0]
	if ref.Dropped == "" {
		t.Fatal("the artifact was stored past the run's budget")
	}
	if !strings.Contains(ref.Dropped, "budget") {
		t.Errorf("Dropped = %q, want it to name the budget", ref.Dropped)
	}
	if ref.SizeBytes < 4096 {
		t.Errorf("SizeBytes = %d, want the true size — 'how much did we lose' is the "+
			"next question a reader asks", ref.SizeBytes)
	}

	cm := h.artifactConfigMap("run1")
	if _, over := cm.Data[ref.Key]; over && ref.Key != "" {
		t.Error("the payload was written despite the budget")
	}
	if configMapBytes(cm) > maxRunArtifactBytes {
		t.Errorf("the ConfigMap is %d bytes, over the %d budget — the apiserver would "+
			"reject it and the whole run's evidence would be lost",
			configMapBytes(cm), maxRunArtifactBytes)
	}
}

// refusingArtifactWrites fails every write to an artifact ConfigMap and passes
// everything else through, so a test can prove that losing the evidence leaves
// the verdict alone.
type refusingArtifactWrites struct{ client.Client }

func (c refusingArtifactWrites) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	if isArtifactCM(obj) {
		return apierrors.NewForbidden(
			corev1.Resource("configmaps"), obj.GetName(), errQuota)
	}
	return c.Client.Create(ctx, obj, opts...)
}

func (c refusingArtifactWrites) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	if isArtifactCM(obj) {
		return apierrors.NewForbidden(
			corev1.Resource("configmaps"), obj.GetName(), errQuota)
	}
	return c.Client.Update(ctx, obj, opts...)
}

func isArtifactCM(obj client.Object) bool {
	_, ok := obj.(*corev1.ConfigMap)
	return ok && strings.HasSuffix(obj.GetName(), artifactConfigMapSuffix)
}

var errQuota = &quotaError{}

type quotaError struct{}

func (*quotaError) Error() string { return "exceeded quota: object-counts" }

// Each node's evidence lands on ITS OWN result — issue #231.
//
// harvestArtifacts looked the result up with resultFor(run, testName, ""), and
// the empty-node form of resultFor returns the FIRST result whose name matches —
// a form its own doc says exists for node-LESS results (an unsupported scope, a
// resolution error). At Node scope every target's result shares the test name,
// so all N resolved to Results[0].
//
// The payloads were stored correctly the whole time; artifactKey embeds the pod
// name. Only the attribution was wrong, which is the worse failure: an engineer
// reading node B's verdict saw "this test produced no evidence" while node B's
// dcgmi JSON sat in the run's ConfigMap under a key nothing pointed at — and
// node A's verdict carried evidence about hardware it does not describe.
//
// Every existing test in this file used a SINGLE-node run and read Results[0],
// so all of them passed with the bug present and would pass identically with it
// fixed. That is why this one asserts per node.
func TestEachNodesEvidenceLandsOnItsOwnResult(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a", "spark-b"),
	)
	// Both nodes at once, so the two harvests happen in one run.
	run := h.run("run1")
	run.Spec.MaxConcurrentNodes = int32p(2)
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("expected a pod per node, got %d", len(pods))
	}
	for node, pod := range pods {
		h.startPod(pod)
		h.finishPod(pod, 0, "nonfiniteCount=0\n"+
			artifactStdout("dcgmi.json", "application/json",
				`{"node":"`+node+`"}`)+"DONE\n", "")
	}
	h.reconcileUntilSettled("run1")

	byNode := map[string]burninv1alpha1.TestResult{}
	for _, res := range h.run("run1").Status.Results {
		if len(res.Nodes) == 1 {
			byNode[res.Nodes[0]] = res
		}
	}
	if len(byNode) != 2 {
		t.Fatalf("expected one result per node, got %d: %+v", len(byNode), byNode)
	}

	for _, node := range []string{"spark-a", "spark-b"} {
		res := byNode[node]
		if len(res.Artifacts) != 1 {
			t.Errorf("%s's result carries %d artifact refs, want exactly 1. Its evidence "+
				"is in the run's ConfigMap either way — but a ref on the wrong result "+
				"tells an engineer the wrong node produced it, and tells this node's "+
				"reader there was none.", node, len(res.Artifacts))
			continue
		}
		// And the ref must point at THIS node's payload, not merely be present.
		cm := h.artifactConfigMap("run1")
		if cm == nil {
			t.Fatal("no artifact ConfigMap")
		}
		if got := cm.Data[res.Artifacts[0].Key]; !strings.Contains(got, `"`+node+`"`) {
			t.Errorf("%s's ref points at %q, whose payload is %q — the evidence is "+
				"attributed to the wrong node", node, res.Artifacts[0].Key, got)
		}
	}
}

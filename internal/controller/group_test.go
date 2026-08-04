package controller

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// Group-scope execution. The invariants under test are the ones a collective
// cannot be measured without, and each of them has a way of failing that looks
// like a hardware verdict if it is got wrong.

func groupTest(name string, thresholds ...burninv1alpha1.Threshold) *burninv1alpha1.BurnInTest {
	return &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: name},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:       burninv1alpha1.KindNCCL,
			Scope:      burninv1alpha1.ScopeGroup,
			Runner:     &burninv1alpha1.RunnerSpec{Image: "example.invalid/nccl:v1"},
			Thresholds: thresholds,
		},
	}
}

// groupRun is a run over n nodes with the interlock raised to admit them all —
// which a Group test requires, because it holds every one of them at once.
func groupRun(name, profileRef string, nodes ...string) *burninv1alpha1.BurnInRun {
	return withNodeCap(newRun(name, profileRef, nodes...), int32(len(nodes)))
}

// rankPod finds the pod for one rank of the latest attempt.
func (h *harness) rankPod(runName string, rank int) *corev1.Pod {
	h.t.Helper()
	var out *corev1.Pod
	want := strconv.Itoa(rank)
	for _, pod := range h.allPods(runName) {
		if pod.Labels[labelRank] != want {
			continue
		}
		if out == nil || pod.Labels[labelAttempt] > out.Labels[labelAttempt] {
			out = pod
		}
	}
	return out
}

// agePods stamps every pod of a run with the harness clock's current time, so
// that advancing the clock afterwards makes them overdue.
//
// It exists because of what the fake client is: a map with a type checker. A
// real apiserver stamps creationTimestamp on every object it admits, and
// podOverdue reads it — deliberately treating a ZERO timestamp as "not overdue",
// since "overdue since year 1" is not a judgment anyone should act on. Under the
// fake client every pod therefore has a zero timestamp and the whole deadline
// path is unreachable, which is why nothing in this package exercised it before.
func (h *harness) agePods(runName string) {
	h.t.Helper()
	for _, pod := range h.allPods(runName) {
		pod.CreationTimestamp = metav1.NewTime(h.nowVal)
		if err := h.c.Update(context.Background(), pod); err != nil {
			h.t.Fatalf("stamp pod %s: %v", pod.Name, err)
		}
	}
}

// startGroup drives a group to the point where every rank has a running pod.
func (h *harness) startGroup(runName string, nranks int) {
	h.t.Helper()
	h.reconcile(runName)
	h.reconcile(runName)
	root := h.rankPod(runName, groupRootRank)
	if root == nil {
		h.t.Fatalf("rank 0 was never created")
	}
	h.startPod(root)
	h.readyPod(root)
	h.reconcile(runName)
	for rank := 1; rank < nranks; rank++ {
		pod := h.rankPod(runName, rank)
		if pod == nil {
			h.t.Fatalf("rank %d was never created after the root went Ready", rank)
		}
		h.startPod(pod)
	}
	h.reconcile(runName)
}

// ncclRootStdout is what the rank that reports a collective prints.
const ncclRootStdout = `algbw=11.98
busbw=15.97
wrong_count=0
NCCL_PASS: 4-rank all-reduce
`

// ncclWorkerStdout is what a non-reporting rank prints: it took part, and the
// numbers live on the root.
const ncclWorkerStdout = `NCCL_PASS: rank joined the collective
`

// ── the shape ────────────────────────────────────────────────────────────────

// One execution, N pods, ONE verdict naming every node. Splitting a collective
// per node would produce N claims no single node can support, and would send an
// engineer to replace a part that was only ever waiting on a peer.
func TestGroup_OneResultNamingEveryNode(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c", "spark-d"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]), gb10Node(nodes[3]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	if got := len(h.allPods("run1")); got != len(nodes) {
		t.Fatalf("group launched %d pods, want %d", got, len(nodes))
	}
	h.finishPod(h.rankPod("run1", 0), 0, ncclRootStdout, "Completed")
	for rank := 1; rank < len(nodes); rank++ {
		h.finishPod(h.rankPod("run1", rank), 0, ncclWorkerStdout, "Completed")
	}
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if len(run.Status.Results) != 1 {
		t.Fatalf("a collective produced %d results, want exactly 1: %+v", len(run.Status.Results), run.Status.Results)
	}
	res := run.Status.Results[0]
	if res.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", res.Phase, res.Message)
	}
	if len(res.Nodes) != len(nodes) {
		t.Fatalf("result names %v, want all %v", res.Nodes, nodes)
	}
	for i, want := range nodes {
		if res.Nodes[i] != want {
			t.Errorf("Nodes[%d] = %q, want %q — ranks come from the pinned target order", i, res.Nodes[i], want)
		}
	}
	// Rank 0 is the reporting rank; its numbers are the collective's.
	if res.Metrics["busBandwidthGBs"] != "15.97" {
		t.Errorf("busBandwidthGBs = %q, want the root's 15.97: %v", res.Metrics["busBandwidthGBs"], res.Metrics)
	}
	if res.Scope != burninv1alpha1.ScopeGroup {
		t.Errorf("scope = %q, want Group", res.Scope)
	}
	h.assertNoStrandedCordons()
}

// The rendezvous contract, which is the whole of what a Group runner is told.
func TestGroup_RendezvousEnvironment(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	svcs := h.rendezvousServices("run1")
	if len(svcs) != 1 {
		t.Fatalf("group created %d rendezvous Services, want 1", len(svcs))
	}
	svc := svcs[0].Name

	for rank := 0; rank < len(nodes); rank++ {
		pod := h.rankPod("run1", rank)
		env := map[string]string{}
		for _, e := range pod.Spec.Containers[0].Env {
			env[e.Name] = e.Value
		}

		if env["BURNIN_RANK"] != strconv.Itoa(rank) {
			t.Errorf("rank %d: BURNIN_RANK = %q", rank, env["BURNIN_RANK"])
		}
		if env["BURNIN_NRANKS"] != strconv.Itoa(len(nodes)) {
			t.Errorf("rank %d: BURNIN_NRANKS = %q, want %d", rank, env["BURNIN_NRANKS"], len(nodes))
		}
		// EVERY rank, including the root itself, is told where the root answers.
		// A root that cannot name itself cannot bind the address its peers will
		// resolve.
		wantRoot := "rank-0." + svc + ".burnin.svc"
		if env["BURNIN_ROOT_HOST"] != wantRoot {
			t.Errorf("rank %d: BURNIN_ROOT_HOST = %q, want %q", rank, env["BURNIN_ROOT_HOST"], wantRoot)
		}
		if env["BURNIN_ROOT_NODE"] != nodes[groupRootRank] {
			t.Errorf("rank %d: BURNIN_ROOT_NODE = %q, want %q", rank, env["BURNIN_ROOT_NODE"], nodes[groupRootRank])
		}
		// A runner that keys off server/client must not silently treat rank 4 as
		// a client. Absence turns a wrong assumption into a loud one.
		if v, set := env["BURNIN_ROLE"]; set {
			t.Errorf("rank %d: BURNIN_ROLE is set to %q at Group scope", rank, v)
		}

		// hostname + subdomain are what give the pod its own A record. Without
		// both, the Service resolves to the endpoint set and no rank can name
		// another.
		if pod.Spec.Hostname != "rank-"+strconv.Itoa(rank) {
			t.Errorf("rank %d: hostname = %q", rank, pod.Spec.Hostname)
		}
		if pod.Spec.Subdomain != svc {
			t.Errorf("rank %d: subdomain = %q, want %q", rank, pod.Spec.Subdomain, svc)
		}
		if pod.Spec.NodeSelector[corev1.LabelHostname] != nodes[rank] {
			t.Errorf("rank %d landed on %q, want %q", rank, pod.Spec.NodeSelector[corev1.LabelHostname], nodes[rank])
		}
	}

	// The peer host is deliberately NOT qualified to cluster.local: a cluster's
	// DNS domain is configurable, and hard-coding the default breaks the
	// rendezvous as a connection error that reads like a bad fabric.
	if strings.Contains(h.rankPod("run1", 1).Spec.Containers[0].Env[3].Value, "cluster.local") {
		t.Error("BURNIN_ROOT_HOST is hard-coded to the default cluster domain")
	}
}

// ── the ready gate ───────────────────────────────────────────────────────────

// No worker rank exists until the root is Ready. A rank that tries to fetch the
// bootstrap handle before the root has bound its socket dies with a connection
// error, and a connection error on a fabric test is indistinguishable, in a
// report, from a fabric that is genuinely broken.
func TestGroup_WorkersWaitForTheRoot(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	if got := len(h.allPods("run1")); got != 1 {
		t.Fatalf("%d pods exist before the root is Ready, want only the root", got)
	}
	root := h.rankPod("run1", groupRootRank)
	if root == nil {
		t.Fatal("the root pod was not created")
	}

	// Started but NOT Ready: still no workers.
	h.startPod(root)
	h.reconcile("run1")
	if got := len(h.allPods("run1")); got != 1 {
		t.Fatalf("%d pods exist while the root is merely Running, want only the root", got)
	}

	h.readyPod(root)
	h.reconcile("run1")
	if got := len(h.allPods("run1")); got != len(nodes) {
		t.Fatalf("%d pods exist after the root went Ready, want all %d", got, len(nodes))
	}
	// All workers together: a collective makes no progress until every rank has
	// joined, so staggering them only adds latency on cordoned, idle nodes.
	for rank := 1; rank < len(nodes); rank++ {
		if h.rankPod("run1", rank) == nil {
			t.Errorf("rank %d was not created in the same pass as the other workers", rank)
		}
	}
}

// ── the verdict ──────────────────────────────────────────────────────────────

// Error outranks Fail. If any rank's machinery broke the collective was never
// formed, and every other rank's failure is then an artifact of the missing
// member rather than evidence about the fabric. Recording that as Fail would
// permanently indict a fleet over an unpullable image, on the one phase this
// operator never retries.
func TestGroup_ErrorOutranksFail(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	// The root reports a hardware failure; a worker's machinery broke.
	h.finishPod(h.rankPod("run1", 0), 1, "NCCL_FAIL: bus bandwidth below floor\n", "Error")
	h.finishPod(h.rankPod("run1", 1), 0, ncclWorkerStdout, "Completed")
	h.finishPod(h.rankPod("run1", 2), 3, "NCCL_ERROR: could not open device\n", "Error")
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error — a rank whose machinery broke means the collective was never formed: %s",
			res.Phase, res.Message)
	}
	// The message must name the ranks that dissent, so an engineer is sent to
	// the right node.
	if !strings.Contains(res.Message, "rank 2") {
		t.Errorf("message does not name the errored rank: %q", res.Message)
	}
	if !strings.Contains(res.Message, "COLLECTIVE") {
		t.Errorf("message does not say the verdict is about the collective: %q", res.Message)
	}
}

// A rank that never reported is not a rank that passed. A collective is only
// measured if every rank took part, so passing on a subset would certify a
// group of N on evidence from fewer — the same class of false negative as an
// empty harvest satisfying a threshold.
func TestGroup_SilentRankIsNotAPass(t *testing.T) {
	pass := func() *runner.Result {
		return &runner.Result{Verdict: runner.VerdictPass, Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}
	}
	members := []groupMember{
		{rank: 0, node: "spark-a", result: pass()},
		{rank: 1, node: "spark-b", result: pass()},
		{rank: 2, node: "spark-c"}, // never reported
	}
	out := combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{})
	if out.Verdict != runner.VerdictError {
		t.Fatalf("verdict = %q, want Error: a collective with a silent rank was not measured", out.Verdict)
	}
	if !strings.Contains(out.Message, "rank 2") || !strings.Contains(out.Message, "did not report") {
		t.Errorf("message does not name the silent rank: %q", out.Message)
	}
}

// A hang is attributed to the ranks that actually hung. On a collective every
// OTHER rank blocks waiting for the one that never arrived, so a message naming
// all of them equally sends an engineer to N racks instead of one.
func TestGroup_DeadlineNamesTheRanksThatHung(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	// Ranks 0 and 1 finish; rank 2 never does, and the window expires.
	h.finishPod(h.rankPod("run1", 0), 0, ncclRootStdout, "Completed")
	h.finishPod(h.rankPod("run1", 1), 0, ncclWorkerStdout, "Completed")
	// Past the whole window: duration + deadline grace + scheduling grace.
	h.agePods("run1")
	h.nowVal = h.nowVal.Add(time.Duration(defaultDurationSeconds+deadlineGraceSeconds)*time.Second +
		schedulingGracePeriod + time.Minute)
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	if res.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error: %s", res.Phase, res.Message)
	}
	if !strings.Contains(res.Message, "rank 2 (spark-c)") {
		t.Errorf("message does not name the rank that hung: %q", res.Message)
	}
	if strings.Contains(res.Message, "rank 0 (spark-a)") {
		t.Errorf("message blames a rank that finished cleanly: %q", res.Message)
	}
	if len(h.livePods("run1")) != 0 {
		t.Errorf("a hung rank's pod was left running on cordoned hardware")
	}
	h.assertNoStrandedCordons()
}

// ── the interlock ────────────────────────────────────────────────────────────

// A group costs one slot per node and is admitted as a unit. Under a cap smaller
// than the group it could never start, so the run is refused at start with the
// reason rather than hanging until its deadline and reporting a timeout.
func TestGroup_RefusedWhenTheCapCannotHoldIt(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"), gb10Node("spark-c"),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b", "spark-c"), 2),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	msg := run.Status.Results[0].Message
	for _, want := range []string{"maxConcurrentNodes", "3", "never start"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal does not mention %q: %q", want, msg)
		}
	}
	if len(h.allPods("run1")) != 0 {
		t.Errorf("a pod was scheduled for a group that can never be admitted")
	}
	h.assertNoStrandedCordons()
}

// Fewer than two nodes is not a collective.
func TestGroup_RefusedBelowTwoNodes(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	if !strings.Contains(run.Status.Results[0].Message, "at least two") {
		t.Errorf("refusal does not explain the minimum: %q", run.Status.Results[0].Message)
	}
}

// Two ranks on one node would contend for the same hardware, and the collective
// would measure that contention instead of the fabric.
func TestGroup_RefusedOnADuplicateNode(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", "spark-a", "spark-b", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error", run.Status.Phase)
	}
	if !strings.Contains(run.Status.Results[0].Message, "DISTINCT") {
		t.Errorf("refusal does not explain the distinctness rule: %q", run.Status.Results[0].Message)
	}
}

// ── cordoning ────────────────────────────────────────────────────────────────

// Every node is held before ANY pod exists. Cordoning only the root would leave
// each worker node accepting workload right up to the moment a rank lands on it,
// and a competing workload on one rank perturbs the measurement for all of them.
func TestGroup_EveryNodeIsCordonedBeforeAnyPodExists(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	if got := len(h.allPods("run1")); got != 1 {
		t.Fatalf("%d pods exist, want only the root at this point", got)
	}
	cordoned := h.cordonedNodes()
	if len(cordoned) != len(nodes) {
		t.Fatalf("cordoned %v, want every node held before the workers land", cordoned)
	}

	// And they are all given back.
	h.startPod(h.rankPod("run1", 0))
	h.readyPod(h.rankPod("run1", 0))
	h.reconcile("run1")
	for rank := 0; rank < len(nodes); rank++ {
		h.startPod(h.rankPod("run1", rank))
	}
	h.finishPod(h.rankPod("run1", 0), 0, ncclRootStdout, "Completed")
	for rank := 1; rank < len(nodes); rank++ {
		h.finishPod(h.rankPod("run1", rank), 0, ncclWorkerStdout, "Completed")
	}
	h.reconcileUntilSettled("run1")
	h.assertNoStrandedCordons()
}

// ── thresholds ───────────────────────────────────────────────────────────────

// Thresholds are evaluated ONCE against the merged metrics, by the same
// fail-closed evaluator every Node-scope test goes through. A gate on a metric
// only the root reports must be satisfied by the root's number.
func TestGroup_ThresholdsEvaluateAgainstTheMergedMetrics(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group", burninv1alpha1.Threshold{
			Metric:     "busBandwidthGBs",
			Comparison: burninv1alpha1.GTE,
			Value:      "40",
		}),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	h.finishPod(h.rankPod("run1", 0), 0, ncclRootStdout, "Completed")
	for rank := 1; rank < len(nodes); rank++ {
		h.finishPod(h.rankPod("run1", rank), 0, ncclWorkerStdout, "Completed")
	}
	h.reconcileUntilSettled("run1")

	res := h.run("run1").Status.Results[0]
	// 15.97 < 40, so this is a real threshold failure — a hardware verdict about
	// the collective, reached on a clean exit.
	if res.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed: %s", res.Phase, res.Message)
	}
	if len(res.Violations) != 1 || res.Violations[0].Metric != "busBandwidthGBs" {
		t.Errorf("violations = %+v, want one naming busBandwidthGBs", res.Violations)
	}
}

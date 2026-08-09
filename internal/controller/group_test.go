package controller

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

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

// A Skip from one rank is not a Skip for the group unless every rank reported.
//
// Reachable, and the shape matters: if the root's runner declares the test
// inapplicable (exit 2 with a _SKIP marker) BEFORE the workers are created —
// which is exactly what gateWorkersOnRoot does when the root terminates early —
// then N-1 nodes never had a pod at all. Honouring the root's declaration for
// the whole group would record "acceptance does not apply to this hardware" for
// nodes nothing looked at, and a run settles Passed around a Skip.
//
// This is the same rule the all-Pass case already has, applied to the verdict
// that is just as capable of certifying unmeasured hardware. A runner may only
// declare what it positively established, and rank 0's declaration is about
// rank 0.
func TestGroup_SkipNeedsEveryRankToHaveReported(t *testing.T) {
	skip := func() *runner.Result {
		return &runner.Result{Verdict: runner.VerdictSkip, ExitCode: 2,
			Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}
	}

	t.Run("the root skips before the workers exist", func(t *testing.T) {
		out := combineGroup([]groupMember{
			{rank: 0, node: "spark-a", result: skip()},
			{rank: 1, node: "spark-b"},
			{rank: 2, node: "spark-c"},
		}, nil, &burninv1alpha1.BurnInTestSpec{})

		if out.Verdict == runner.VerdictSkip {
			t.Fatalf("a group where 2 of 3 ranks never ran was recorded Skipped — " +
				"that certifies hardware nothing looked at as out of scope")
		}
		if out.Verdict != runner.VerdictError {
			t.Fatalf("verdict = %q, want Error: the group was not measured", out.Verdict)
		}
	})

	t.Run("every rank declares the skip", func(t *testing.T) {
		// The legitimate case is unaffected: when every rank looked at its own
		// hardware and said the test does not apply, the group really is out of
		// scope, and reporting Error would send someone to investigate a fleet
		// that answered the question correctly.
		out := combineGroup([]groupMember{
			{rank: 0, node: "spark-a", result: skip()},
			{rank: 1, node: "spark-b", result: skip()},
			{rank: 2, node: "spark-c", result: skip()},
		}, nil, &burninv1alpha1.BurnInTestSpec{})

		if out.Verdict != runner.VerdictSkip {
			t.Errorf("verdict = %q, want Skip: every rank positively declared it", out.Verdict)
		}
	})

	t.Run("a Fail still outranks a missing rank", func(t *testing.T) {
		// The guard must not swallow a measured fault. Fail outranks Skip, and a
		// rank that positively failed has established something about the
		// hardware whatever the others did.
		fail := &runner.Result{Verdict: runner.VerdictFail, ExitCode: 1,
			Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}
		out := combineGroup([]groupMember{
			{rank: 0, node: "spark-a", result: fail},
			{rank: 1, node: "spark-b"},
			{rank: 2, node: "spark-c"},
		}, nil, &burninv1alpha1.BurnInTestSpec{})

		if out.Verdict != runner.VerdictFail {
			t.Errorf("verdict = %q, want Fail: a measured fault is not erased by a silent peer", out.Verdict)
		}
	})
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

// A group that fails uniformly must still say WHY.
//
// The summariser collapses ranks that share a verdict so a large group cannot
// produce a status message the apiserver refuses. Collapsing them to a bare
// count discarded the runner's own words in the one case where every rank had
// the same thing to say — which is the shape of every real infrastructure
// fault: a missing host mount, an unpullable image, a driver skew. The stored
// record then explained nothing, and the pod logs that did explain it are gone
// at the run's TTL. That is the defect #114 fixed for a single runner, arriving
// again through the group summariser.
func TestGroup_UniformFailureStillCarriesTheReason(t *testing.T) {
	const reason = "NCCL_ERROR: could not open device /dev/infiniband/uverbs0"

	for _, nranks := range []int{3, 5, 12} {
		members := make([]groupMember, 0, nranks)
		for i := 0; i < nranks; i++ {
			members = append(members, groupMember{
				rank: i, node: fmt.Sprintf("spark-%d", i),
				result: &runner.Result{Verdict: runner.VerdictError, ExitCode: 3, Message: reason,
					Metrics: map[string]string{}, Unmeasurable: map[string]bool{}},
			})
		}
		out := combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{})

		if !strings.Contains(out.Message, reason) {
			t.Errorf("a %d-rank group where every rank reported %q recorded a message that does not contain it:\n  %s",
				nranks, reason, out.Message)
		}
		// Still bounded: the whole point of collapsing is that a large group
		// cannot write a status the apiserver refuses.
		if len(out.Message) > 1024 {
			t.Errorf("a %d-rank group produced a %d-character message; the summariser is not bounding it",
				nranks, len(out.Message))
		}
	}
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

// No rank is killed for waiting out the rendezvous — issue #122.
//
// A collective makes no progress until the LAST rank joins, so every rank sits
// idle while the others are scheduled and pull their image. The root waits
// longest — it is created first and the workers are not created until it reports
// Ready — but it is a difference of degree, not of kind, and every second of it
// used to be charged against an activeDeadlineSeconds sized for ONE pod's start.
func TestGroup_EveryRanksDeadlineCoversTheRendezvous(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	// The extra is schedulingGracePeriod and nothing invented: it is already how
	// long this operator waits for a pod to start before giving up on it.
	want := int64(defaultDurationSeconds+deadlineGraceSeconds) + int64(schedulingGracePeriod/time.Second)
	for rank := 0; rank < len(nodes); rank++ {
		pod := h.rankPod("run1", rank)
		if pod.Spec.ActiveDeadlineSeconds == nil {
			t.Fatalf("rank %d has no deadline", rank)
		}
		if got := *pod.Spec.ActiveDeadlineSeconds; got != want {
			t.Errorf("rank %d deadline = %ds, want %ds — it pays for the rendezvous out of its own "+
				"test budget", rank, got, want)
		}
	}

	// The ordering that makes the extra useful: the OPERATOR must give up at or
	// before the kubelet, so the recorded message names the ranks that did not
	// report rather than saying only that a deadline passed. podOverdue measures
	// from CreationTimestamp and activeDeadlineSeconds from StartTime, which is
	// never earlier, so equal windows put the operator first.
	operatorWindow := int64(defaultDurationSeconds+deadlineGraceSeconds) + int64(schedulingGracePeriod/time.Second)
	if want > operatorWindow {
		t.Errorf("a rank's deadline %ds outlives the operator's own patience %ds — the kubelet would "+
			"never be the one to kill it, and a wedged rank would linger", want, operatorWindow)
	}
}

// Node and Pair pods are untouched. A Pair server has the same shape with ONE
// peer and 120s has been sufficient for it on real hardware, so there is
// evidence it does not need this and none that it does.
func TestNodeAndPairDeadlinesAreUnchanged(t *testing.T) {
	spec := &burninv1alpha1.BurnInTestSpec{
		Kind:            burninv1alpha1.KindComputeSmoke,
		DurationSeconds: 300,
		Runner:          &burninv1alpha1.RunnerSpec{Image: "example.invalid/x:v1"},
	}
	run := newRun("run1", "p", "spark-a")
	want := int64(300 + deadlineGraceSeconds)

	node, err := podForTest(run, 0, 1, "t", spec, nil, "spark-a", run.Spec.Target, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := *node.Spec.ActiveDeadlineSeconds; got != want {
		t.Errorf("Node-scope deadline = %d, want %d", got, want)
	}

	for _, role := range []string{pairRoleServer, pairRoleClient} {
		pair, err := podForTest(run, 0, 1, "t", spec, nil, "spark-a", run.Spec.Target, &rendezvous{
			scope: burninv1alpha1.ScopePair, role: role, service: "svc", peerRole: "other",
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := *pair.Spec.ActiveDeadlineSeconds; got != want {
			t.Errorf("Pair %s deadline = %d, want %d", role, got, want)
		}
	}
}

// A Group test must not fall back to a runner image that cannot do Group — #118.
//
// This is the one that would have shipped a false negative. A fabric runner
// branches on BURNIN_ROLE and reads its absence as "this is a Node-scope run,
// and a link test with one node has no meaning": exit 2, with a declared _SKIP
// marker. At Group scope the operator sets BURNIN_RANK and BURNIN_NRANKS and
// deliberately NOT BURNIN_ROLE — so the result is recorded Skip, and the run
// settles Passed around a collective that never ran.
//
// The refusal is per KIND, because whether the shipped image speaks the Group
// contract is a property of that image. nccl's does; ib-write-bw's and
// gpudirect-rdma's do not and never will, because a point-to-point RDMA write
// has no N-rank form.
func TestGroup_RefusedWhenTheDefaultImageCannotDoGroup(t *testing.T) {
	bare := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "fabric-group"},
		Spec: burninv1alpha1.BurnInTestSpec{
			// A link runner. Its image will never speak the Group rendezvous.
			Kind:  burninv1alpha1.KindIBWriteBW,
			Scope: burninv1alpha1.ScopeGroup,
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"), gb10Node("spark-c"),
		bare,
		profile("acceptance", nil, false, testRef("fabric-group")),
		groupRun("run1", "acceptance", "spark-a", "spark-b", "spark-c"),
	)
	h.reconcile("run1")
	h.reconcileUntilSettled("run1")

	run := h.run("run1")
	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error — that image would SKIP, and a skipped run settles "+
			"Passed around hardware nobody measured", run.Status.Phase)
	}
	msg := run.Status.Results[0].Message
	for _, want := range []string{"spec.runner.image", "BURNIN_RANK", "#118"} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal does not mention %q: %q", want, msg)
		}
	}
	if len(h.allPods("run1")) != 0 {
		t.Errorf("a pod was scheduled for a Group test whose image cannot do Group")
	}
	h.assertNoStrandedCordons()
}

// nccl's shipped runner DOES speak the contract, so it needs no explicit image.
// This is the half that would rot silently if the allow-list and the runner ever
// drifted apart: the operator would refuse a test the image can run.
func TestGroup_TheNCCLDefaultImageIsAllowed(t *testing.T) {
	bare := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "nccl-group"},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:  burninv1alpha1.KindNCCL,
			Scope: burninv1alpha1.ScopeGroup,
		},
	}
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		bare,
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	if got := len(h.allPods("run1")); got != len(nodes) {
		t.Fatalf("group launched %d pods, want %d — the shipped nccl image speaks the Group "+
			"rendezvous and must not be refused", got, len(nodes))
	}
	if img := h.rankPod("run1", 0).Spec.Containers[0].Image; !strings.Contains(img, "nccl") {
		t.Errorf("rank 0 image = %q, want the nccl default", img)
	}
}

// An explicit image is the author saying they have a runner that speaks the
// contract, and the operator has no business second-guessing that. It is also
// the documented way to run Group scope today.
func TestGroup_AnExplicitRunnerImageIsAccepted(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"), // groupTest sets Runner.Image explicitly
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))
	if got := len(h.allPods("run1")); got != len(nodes) {
		t.Fatalf("group launched %d pods, want %d — an explicit image must be honoured", got, len(nodes))
	}
}

// Ranks disagreeing about a metric is recorded, not silently resolved — #121.
//
// combineGroup merges with rank 0 winning any shared key. That is right for a
// collective measurand (busBandwidthGBs is a property of the group) and wrong
// for a per-node one (gpuTempC belongs to whichever node emitted it), and
// nothing in the metric distinguishes them. Choosing a winner differently needs
// a metric's polarity, which pkg/contract does not carry, or a
// refuse-on-disagreement rule, which would fail closed on every healthy
// collective whose ranks each report their own figure. Both need a measurement
// of what a real N-rank runner emits.
//
// So the value is unchanged and the disagreement is on the record — which is
// the difference between a result an engineer can act on and one they cannot.
func TestGroup_RankDisagreementIsRecorded(t *testing.T) {
	mk := func(rank int, temp string) groupMember {
		return groupMember{
			rank: rank, node: fmt.Sprintf("spark-%d", rank),
			result: &runner.Result{
				Verdict: runner.VerdictPass, ExitCode: 0,
				Metrics:      map[string]string{"gpuTempC": temp, "busBandwidthGBs": "15.97"},
				Unmeasurable: map[string]bool{},
			},
		}
	}
	out := combineGroup([]groupMember{mk(0, "70"), mk(1, "72"), mk(2, "95")},
		nil, &burninv1alpha1.BurnInTestSpec{})

	// The stored value is still rank 0's — the merge rule did not change.
	if got := out.Metrics["gpuTempC"]; got != "70" {
		t.Errorf("gpuTempC = %q, want rank 0's 70", got)
	}
	// But the fact that it was contested is now visible.
	if !strings.Contains(out.Message, "DISAGREED") {
		t.Errorf("the message does not record the disagreement: %q", out.Message)
	}
	for _, want := range []string{"gpuTempC", "rank 2=95", "#121"} {
		if !strings.Contains(out.Message, want) {
			t.Errorf("the message does not mention %q: %q", want, out.Message)
		}
	}
	// A key every rank agreed on must NOT be reported as contested — otherwise
	// the note fires on every healthy collective and gets ignored.
	if strings.Contains(out.Message, "busBandwidthGBs recorded as") {
		t.Errorf("an agreed metric was reported as a disagreement: %q", out.Message)
	}
}

// A group whose ranks agree says nothing, which is what keeps the note worth
// reading when it does appear.
func TestGroup_AgreementIsSilent(t *testing.T) {
	mk := func(rank int) groupMember {
		return groupMember{
			rank: rank, node: fmt.Sprintf("spark-%d", rank),
			result: &runner.Result{
				Verdict: runner.VerdictPass,
				Metrics: map[string]string{"busBandwidthGBs": "15.97"}, Unmeasurable: map[string]bool{},
			},
		}
	}
	out := combineGroup([]groupMember{mk(0), mk(1), mk(2)}, nil, &burninv1alpha1.BurnInTestSpec{})
	if strings.Contains(out.Message, "DISAGREED") {
		t.Errorf("a unanimous group reported a disagreement: %q", out.Message)
	}
}

// Bounded: a large group disagreeing about many keys must not write a status the
// apiserver refuses, and the counts stay exact even where the detail is elided.
func TestGroup_DisagreementNoteIsBounded(t *testing.T) {
	var members []groupMember
	for rank := 0; rank < 40; rank++ {
		m := map[string]string{}
		for k := 0; k < 12; k++ {
			m[fmt.Sprintf("metric%dPct", k)] = fmt.Sprintf("%d", rank)
		}
		members = append(members, groupMember{
			rank: rank, node: fmt.Sprintf("spark-%d", rank),
			result: &runner.Result{Verdict: runner.VerdictPass, Metrics: m, Unmeasurable: map[string]bool{}},
		})
	}
	out := combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{})
	if len(out.Message) > 2048 {
		t.Errorf("a 40-rank group disagreeing about 12 metrics produced a %d-character message",
			len(out.Message))
	}
	if !strings.Contains(out.Message, "12 metric(s)") {
		t.Errorf("the exact count was lost when the detail was elided: %q", out.Message)
	}
}

// A rank declaring a metric unmeasurable must not be erased by another rank's
// number — audit round 3.
//
// The Pair rule is "if either end measured the quantity, it was measurable on
// this link", and at Pair scope that is right: two ends, one link, one quantity.
// At Group scope it certifies. CLAUDE.md's own worked example is the trigger:
// ranks 0 and 1 have ECC and report 0, rank 2 is a GB10 whose runner correctly
// emits eccErrors=n/a — and the group stored eccErrors=0 with an EMPTY
// Unmeasurable set, so an `eccErrors Equal 0` gate passed and all three nodes
// were certified including the one whose gate never ran. The same node under
// Node scope fails that threshold closed.
func TestGroup_AnUnmeasurableDeclarationIsNotSilentlyErased(t *testing.T) {
	measured := func(rank int) groupMember {
		return groupMember{rank: rank, node: fmt.Sprintf("spark-%d", rank),
			result: &runner.Result{Verdict: runner.VerdictPass,
				Metrics: map[string]string{"eccErrors": "0"}, Unmeasurable: map[string]bool{}}}
	}
	gb10 := groupMember{rank: 2, node: "spark-2",
		result: &runner.Result{Verdict: runner.VerdictPass,
			Metrics: map[string]string{}, Unmeasurable: map[string]bool{"eccErrors": true}}}

	out := combineGroup([]groupMember{measured(0), measured(1), gb10}, nil,
		&burninv1alpha1.BurnInTestSpec{})

	if !strings.Contains(out.Message, "eccErrors") {
		t.Errorf("rank 2 declared eccErrors unmeasurable and the group message does not mention "+
			"it at all — the node is certified on the other ranks' zeros: %q", out.Message)
	}
	if !strings.Contains(out.Message, "rank 2=") {
		t.Errorf("the declaration is not attributed to the rank that made it: %q", out.Message)
	}
}

// The note must name the rank that produced the STORED value, and rank 0 is not
// always that rank.
//
// An earlier version said "the value stored is rank 0's" unconditionally, which
// is false whenever rank 0 omitted the metric — and it then sent an engineer to
// the one node that had nothing to do with the reading.
func TestGroup_DisagreementNamesTheRankThatProducedTheStoredValue(t *testing.T) {
	// Rank 0 does not report gpuTempC at all.
	root := groupMember{rank: 0, node: "spark-0",
		result: &runner.Result{Verdict: runner.VerdictPass,
			Metrics: map[string]string{"busBandwidthGBs": "15.97"}, Unmeasurable: map[string]bool{}}}
	warm := groupMember{rank: 1, node: "spark-1",
		result: &runner.Result{Verdict: runner.VerdictPass,
			Metrics: map[string]string{"gpuTempC": "72"}, Unmeasurable: map[string]bool{}}}
	hot := groupMember{rank: 2, node: "spark-2",
		result: &runner.Result{Verdict: runner.VerdictPass,
			Metrics: map[string]string{"gpuTempC": "95"}, Unmeasurable: map[string]bool{}}}

	out := combineGroup([]groupMember{root, warm, hot}, nil, &burninv1alpha1.BurnInTestSpec{})

	if out.Metrics["gpuTempC"] != "72" {
		t.Fatalf("gpuTempC = %q, want rank 1's 72 (the lowest rank that reported it)",
			out.Metrics["gpuTempC"])
	}
	if strings.Contains(out.Message, "rank 0's") {
		t.Errorf("the note credits rank 0 for a value rank 0 never produced: %q", out.Message)
	}
	if !strings.Contains(out.Message, "(rank 1)") {
		t.Errorf("the note does not name the rank that produced the stored value: %q", out.Message)
	}
}

// A threshold failure must not erase the constructed group message.
//
// At Node scope parsed.Message is one runner's last line and replacing it costs
// nothing. A Group message is built by this operator and carries the clause
// saying the verdict is about the COLLECTIVE, each rank's own report, and the
// disagreement note — an 8-rank group failing a bandwidth gate stored a bare
// threshold line beside eight node names, which is the misattribution that lead
// clause exists to prevent.
func TestGroup_AThresholdFailureKeepsTheCollectiveContext(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	h := newHarness(t,
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group", burninv1alpha1.Threshold{
			Metric: "busBandwidthGBs", Comparison: burninv1alpha1.GTE, Value: "40",
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
	if res.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", res.Phase)
	}
	if !strings.Contains(res.Message, "busBandwidthGBs") {
		t.Errorf("the threshold that decided the verdict is missing: %q", res.Message)
	}
	if !strings.Contains(res.Message, "COLLECTIVE") {
		t.Errorf("a Failed group lost the clause saying the verdict is about the collective, so a "+
			"reader with three node names in front of them will pick one: %q", res.Message)
	}
}

// A long message must lose the RANK SPEW, never the clauses this operator wrote.
//
// The previous revision clamped the whole message from the end, and the
// disagreement note was appended last — so on exactly the long messages the
// clamp exists for, the note went and the runner spew stayed. Since the merge
// deliberately leaves out.Unmeasurable empty when another rank measured the key,
// that note is the ONLY record that a rank declared the metric unmeasurable, and
// truncating it restored the certification it was written to prevent.
//
// The old guard asserted only length, so it stayed green throughout. This one
// asserts what the message is FOR.
func TestGroup_ClampingKeepsWhatTheOperatorChoseToSay(t *testing.T) {
	spew := strings.Repeat("spark:1:1 [0] NCCL WARN Cuda failure 'unhandled cuda error' net_ib.cc:1043 ", 40)

	// The verdicts are spread across all four classes DELIBERATELY. groupMessage
	// collapses ranks that agree, so a uniform group produces one short exemplar
	// and never reaches the clamp at all — an earlier version of this test made
	// that mistake and passed against the broken ordering. Four classes, each
	// naming up to groupMaxNamedRanks ranks with their own clamped line, is what
	// actually pushes the summaries past maxGroupMessage.
	verdicts := []runner.Verdict{runner.VerdictPass, runner.VerdictFail, runner.VerdictSkip, runner.VerdictError}
	var members []groupMember
	for rank := 0; rank < 20; rank++ {
		m := map[string]string{"eccErrors": "0", "gpuTempC": "70"}
		u := map[string]bool{}
		switch rank {
		case 2:
			m["gpuTempC"] = "95" // the node that ran hot
		case 3:
			delete(m, "eccErrors") // a GB10: it cannot produce the measurement
			u["eccErrors"] = true
		}
		members = append(members, groupMember{rank: rank, node: fmt.Sprintf("spark-%02d", rank),
			result: &runner.Result{Verdict: verdicts[rank%len(verdicts)], ExitCode: rank % 4,
				Message: spew, Metrics: m, Unmeasurable: u}})
	}
	out := combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{})
	if len(out.Message) < maxGroupMessage/2 {
		t.Fatalf("the fixture produced a %d-character message and never approaches the clamp, so "+
			"it proves nothing about ordering", len(out.Message))
	}

	if len(out.Message) > maxGroupMessage+512 {
		t.Errorf("message is %d characters; a status grows per attempt and an unwritable one "+
			"loses the verdict entirely", len(out.Message))
	}
	// The whole point: the deliberate clauses survive the truncation.
	for _, want := range []string{"DISAGREED", "eccErrors", "rank 3=n/a", "gpuTempC", "rank 2=95"} {
		if !strings.Contains(out.Message, want) {
			t.Errorf("truncation removed %q — the note is the ONLY record that rank 3's gate never "+
				"ran, and the only thing naming the node that ran hot:\n%s", want, out.Message)
		}
	}
	if !strings.Contains(out.Message, "COLLECTIVE") {
		t.Errorf("truncation removed the clause saying the verdict is about the collective")
	}
	// And a rune boundary was respected: these messages carry em dashes.
	if !utf8.ValidString(out.Message) {
		t.Error("clamping cut a multi-byte sequence in half")
	}
}

// One loud rank must not crowd the DISSENTING ranks out of the summary.
//
// Ranks sharing a verdict are collapsed to an exemplar on purpose, so this uses
// ranks that differ: rank 0 errors verbosely and the rest pass. The ones that
// differ are the ones worth reading, and a whole-message cap alone would have
// let rank 0's spew consume the budget before they were named.
func TestGroup_OneLoudRankDoesNotHideTheDissenters(t *testing.T) {
	var members []groupMember
	for rank := 0; rank < 4; rank++ {
		m := groupMember{rank: rank, node: fmt.Sprintf("spark-%d", rank),
			result: &runner.Result{Verdict: runner.VerdictPass, ExitCode: 0, Message: "clean",
				Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}}
		if rank == 0 {
			m.result.Verdict, m.result.ExitCode = runner.VerdictError, 3
			m.result.Message = strings.Repeat("an extremely verbose rank-0 failure. ", 400)
		}
		members = append(members, m)
	}
	out := combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{})

	// Rank 0's own words are clamped, not omitted.
	if !strings.Contains(out.Message, "verbose rank-0 failure") {
		t.Errorf("rank 0's explanation was dropped entirely: %s", out.Message)
	}
	if len(out.Message) > maxGroupMessage+512 {
		t.Errorf("message is %d characters", len(out.Message))
	}
	// And the ranks that disagreed with it are still there.
	for rank := 1; rank < 4; rank++ {
		if !strings.Contains(out.Message, fmt.Sprintf("spark-%d", rank)) {
			t.Errorf("rank %d is missing because rank 0 was loud:\n%s", rank, out.Message)
		}
	}
}

// The rejected-name list is deduped and bounded, with the count kept exact.
func TestGroup_InvalidNamesAreDedupedAndBounded(t *testing.T) {
	var members []groupMember
	for rank := 0; rank < 40; rank++ {
		// pkg/runner appends one entry per offending LINE, and progressive
		// reporting is expected — so one bad key becomes ranks x samples entries.
		var names []string
		for sample := 0; sample < 20; sample++ {
			names = append(names, "nccl.busbw")
		}
		members = append(members, groupMember{rank: rank, node: fmt.Sprintf("spark-%d", rank),
			result: &runner.Result{Verdict: runner.VerdictPass, InvalidNames: names,
				Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}})
	}
	out := combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{})
	if len(out.InvalidNames) != 1 {
		t.Errorf("InvalidNames has %d entries for ONE distinct rejected name; the set is the "+
			"finding, the number of times it was printed is not", len(out.InvalidNames))
	}
}

// Truncation must never cut a rune in half, and must never back off more than a
// rune to avoid doing so — audit round 5.
//
// Both halves were wrong, in the same commit that claimed to have fixed the
// first. clampRunnerLine sliced at a raw byte offset, and it is the function
// that cuts RUNNER text — the text least under this project's control, and full
// of em dashes because this project's own runners join their prose with them.
// clampRankSummaries did back off, but looped on utf8.ValidString(s[:cut]),
// which inspects the ENTIRE prefix: it is false for every cut at or beyond the
// first invalid byte anywhere, so the loop walked back to just before that byte
// instead of at most three.
//
// The previous guard asserted utf8.ValidString on a fixture built from
// strings.Repeat of pure ASCII, so it could not have failed.
func TestTruncateAtRuneNeverSplitsARune(t *testing.T) {
	// An em dash is three bytes. Sweeping the pad walks it across the boundary.
	for pad := maxRankLine - 4; pad <= maxRankLine+1; pad++ {
		in := strings.Repeat("a", pad) + "—" + strings.Repeat("b", 100)
		got := clampRunnerLine(in)
		if !utf8.ValidString(got) {
			t.Errorf("pad=%d produced invalid UTF-8: %q", pad, got[len(got)-12:])
		}
		// And it backed off at most a rune: the result must still be near the cap.
		if len(got) < maxRankLine-4 {
			t.Errorf("pad=%d backed off to %d bytes, far short of the %d cap", pad, len(got), maxRankLine)
		}
	}
}

// The back-off must be bounded by a rune, not by where the first invalid byte
// happens to be. One bad byte early in a long message used to cost thousands of
// bytes of budget and nineteen ranks out of twenty.
func TestTruncationDoesNotCollapseOnAnEarlyBadByte(t *testing.T) {
	// A lone 0xff early on: a runner's raw container-log bytes can be anything.
	in := "HEAD-CLAUSE " + strings.Repeat("a", 100) + "\xff" + strings.Repeat("b", 8000)
	got := clampRankSummaries(in)

	if len(got) < maxGroupMessage-4 {
		t.Errorf("an invalid byte at offset ~112 collapsed the result to %d bytes of a %d budget — "+
			"the back-off is scanning the prefix instead of the tail", len(got), maxGroupMessage)
	}
	if !strings.HasPrefix(got, "HEAD-CLAUSE") {
		t.Error("the head clause was lost")
	}
}

// End to end: one em dash in one rank's line must not cost the other ranks their
// place in the stored result.
func TestGroup_AnEmDashDoesNotCostTheOtherRanks(t *testing.T) {
	build := func(dash string) string {
		verdicts := []runner.Verdict{runner.VerdictPass, runner.VerdictFail, runner.VerdictSkip, runner.VerdictError}
		var members []groupMember
		for rank := 0; rank < 20; rank++ {
			// 299, not 297. At 297 the em dash's three bytes sit ENTIRELY inside a
			// 300-byte cut and nothing invalid is produced — which is how the
			// first version of this test passed against the very code it was
			// written to catch. The rune must STRADDLE the boundary.
			msg := strings.Repeat("x", maxRankLine-1) + dash +
				strings.Repeat(" NCCL WARN cuda failure net_ib.cc:1043", 30)
			members = append(members, groupMember{rank: rank, node: fmt.Sprintf("spark-%02d", rank),
				result: &runner.Result{Verdict: verdicts[rank%4], ExitCode: rank % 4, Message: msg,
					Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}})
		}
		return combineGroup(members, nil, &burninv1alpha1.BurnInTestSpec{}).Message
	}
	named := func(msg string) int {
		n := 0
		for rank := 0; rank < 20; rank++ {
			if strings.Contains(msg, fmt.Sprintf("spark-%02d", rank)) {
				n++
			}
		}
		return n
	}

	withDash, withHyphen := build("—"), build("-")
	if !utf8.ValidString(withDash) {
		t.Error("the stored message carries a half-written UTF-8 sequence")
	}
	if d, h := named(withDash), named(withHyphen); d < h {
		t.Errorf("one em dash cost the message %d of the %d ranks the byte-identical ASCII version "+
			"names — the stored result is the only durable account once the pods' TTL expires", h-d, h)
	}
}

// The undeclared-skip branch is clamped too, and it is the branch that needs it
// most.
//
// summary() has two paths that embed a rank's own words. The ordinary one was
// clamped; the UndeclaredSkip one returned early and was not — so the single
// branch carrying the LEAST bounded text was the one branch that did not bound
// it. An undeclared exit 2 is most often a crashed runner (a Go panic exits 2),
// which means the message is the tail of a stack trace rather than a sentence
// anybody wrote.
func TestGroup_UndeclaredSkipSummaryIsClampedToo(t *testing.T) {
	trace := strings.Repeat("main.(*probe).scan(0xc000123456, 0x1d) /src/main.go:132 +0x1d ", 60)

	clamped := func(undeclared bool) int {
		m := groupMember{rank: 0, node: "spark-a", result: &runner.Result{
			Verdict: runner.VerdictError, ExitCode: 2, Message: trace,
			UndeclaredSkip: undeclared,
			Metrics:        map[string]string{}, Unmeasurable: map[string]bool{},
		}}
		return len(m.summary())
	}

	ordinary, undeclaredSkip := clamped(false), clamped(true)
	if undeclaredSkip > ordinary+len(undeclaredSkipBrief)+64 {
		t.Errorf("an undeclared-skip summary is %d bytes against %d for the same message on the "+
			"ordinary path — the branch most likely to carry a raw stack trace is the one not "+
			"being clamped", undeclaredSkip, ordinary)
	}
}

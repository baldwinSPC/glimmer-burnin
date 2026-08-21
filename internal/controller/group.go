package controller

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	burningroup "github.com/baldwinSPC/glimmer-burnin/pkg/group"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// Group-scope execution: an operator-native N-rank rendezvous.
//
// # Shape
//
// A Group test is ONE execution across ALL of the run's target nodes. Target i
// is rank i, each pinned by hostname nodeSelector exactly as a Node-scope pod
// is, and all of them addressable through one headless Service that gives each
// pod its own DNS name ("rank-0", "rank-1", …). Rank 0 is the ROOT: it starts
// first, and the other ranks are not created until it reports Ready. The group
// produces a SINGLE TestResult naming every node.
//
// # Why not JobSet, and why not OpenMPI
//
// The issue that asked for this scope (#36) specified both, and the Pair work
// that had to land first is what showed neither is needed.
//
// JobSet exists to solve gang SCHEDULING: N pods that must be placed together or
// not at all, competing for scarce capacity. This operator has already solved
// that problem before any pod exists, and solved it harder. Every rank is pinned
// by hostname to exactly one candidate node; that node has been admitted against
// maxConcurrentNodes, checked against every other run holding hardware
// (admission.go), and CORDONED so nothing else can land beside the burn-in.
// There is no placement decision left for a gang scheduler to make. What JobSet
// would add is a controller every cluster installing this operator must also
// install, and a second owner of pod lifecycle — which this project cannot
// afford, because "a pod may only be destroyed once the status justifying its
// destruction is readable from the apiserver" is an invariant asserted against a
// real apiserver in test/envtest/invariants_test.go, and it is only assertable
// while this reconciler is the thing doing the destroying.
//
// OpenMPI is refused for the reason nccl_pair.cu already records in its own
// header: mpirun needs a launcher able to start a process on the OTHER node,
// which on Kubernetes means shipping an sshd in a burn-in image and giving it a
// key — a standing remote-execution service on every accelerator node in the
// fleet, to run a bandwidth test. That argument does not weaken at N > 2. It
// gets stronger, because the number of nodes carrying the sshd goes up.
//
// What is left is the rendezvous the operator already owns, generalised from two
// members to N. A collective's bootstrap is not a launcher problem: one rank
// publishes a handle and the others fetch it, which is exactly what this
// project's own nccl runner does over one TCP connection today.
//
// # The verdict rule
//
// One result, Nodes = every rank's node in rank order, judged by the same
// precedence Pair uses:
//
//	Error > Fail > Skip > Pass
//
// Error outranks Fail for the reason it does at Pair scope: if any rank's
// machinery broke, the collective was never formed, and the other ranks'
// failures are then artifacts of the missing member rather than evidence about
// the fabric. Recording that as Fail would permanently indict a fleet over an
// unpullable image, and Fail is the one phase this operator never retries.
//
// # Why every rank is waited for, unlike a Pair
//
// A Pair settles when its CLIENT terminates, because a server may legitimately
// linger — it is a listener, and waiting for it would turn every successful pair
// into a deadline Error.
//
// A collective is different in kind and this rule follows the difference rather
// than the symmetry: every rank is a PARTICIPANT in a synchronous operation, so
// they finish together, and a rank that has not finished is a rank the collective
// is still waiting on. Settling on rank 0 alone would therefore discard the
// reports of ranks that were milliseconds from writing them, and — worse — would
// hide the one rank that hung, which is the single most useful thing a group
// failure can tell an engineer. So the group settles when EVERY rank has
// terminated, and podOverdue converts a genuine hang into an Error that NAMES
// the ranks still running.
//
// # Metrics
//
// Merged in rank order, so a LOWER rank wins any key several ranks report and
// rank 0's numbers stand. Rank 0 is the root and the conventional reporting rank
// for every collective benchmark; the other ranks' keys are kept where they do
// not collide, because a rank that measured something nobody else did is
// evidence worth storing.

// groupMember is one rank's report. A nil result means that rank never produced
// one, which is a different statement from "it reported nothing useful".
type groupMember struct {
	rank   int
	node   string
	pod    string
	result *runner.Result
}

func (m groupMember) summary() string {
	if m.result == nil {
		return fmt.Sprintf("rank %d (%s) did not report", m.rank, m.node)
	}
	out := fmt.Sprintf("rank %d (%s) %s (exit %d)",
		m.rank, m.node, strings.ToLower(string(m.result.Verdict)), m.result.ExitCode)
	// Same treatment as pairSide.summary: an undeclared exit 2 is explained
	// rather than quoted, because a stack frame standing in for the reason a
	// collective went unmeasured is the record that outlives every Event about it.
	if m.result.UndeclaredSkip {
		out += ": " + undeclaredSkipBrief
		// Clamped on THIS path too, and it is the path that needs it most: an
		// undeclared exit 2 is most often a crashed runner, so the message here
		// is the tail of a Go panic trace rather than a sentence anyone wrote.
		// The early return used to skip the clamp below, which made the one
		// branch carrying the least bounded text the one branch that did not
		// bound it.
		if s := strings.TrimSpace(m.result.Message); s != "" {
			out += " [runner said: " + clampRunnerLine(s) + "]"
		}
		return out
	}
	if s := strings.TrimSpace(m.result.Message); s != "" {
		out += ": " + clampRunnerLine(s)
	}
	return out
}

// maxRankLine bounds ONE rank's own words inside a group summary.
//
// A rank's message is its last non-key=value stdout line, which nothing caps —
// an NCCL or CUTLASS failure runs to kilobytes. Clamping per rank rather than
// only in total is what stops one loud rank from crowding every other rank out
// of the summary: with a whole-message cap alone, rank 0 spewing 8 KB means
// ranks 1..N are never named at all, and on a collective the ranks that DIFFER
// are the ones worth reading.
const maxRankLine = 300

func clampRunnerLine(s string) string {
	return truncateAtRune(strings.Join(strings.Fields(s), " "), maxRankLine, "…")
}

// advanceGroup moves one Group-scope execution forward by exactly one step. It
// is the Group counterpart of advance and advancePair, and it returns the same
// states so the pass bookkeeping in execute does not care which produced them.
func (r *BurnInRunReconciler) advanceGroup(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	index int,
	t plannedTest,
	launch bool,
	busy map[string]bool,
	capNodes int,
) (advanceState, advanceEffect, error) {
	var none advanceEffect

	// Ranks come from the PINNED plan's target order, so they are stable across
	// reconciles and across a manager restart. Re-deriving them from a live
	// selector would renumber the collective mid-test and orphan every pod.
	nodes := append([]string(nil), p.Targets...)

	res := resultFor(run, t.Name, nodes[groupRootRank])
	if res != nil && res.Phase.IsTerminal() {
		return advanceDone, none, nil
	}
	attempt, _ := nextAttempt(res)

	pods, err := r.groupPods(ctx, run, index, attempt, nodes)
	if err != nil {
		return advancePending, none, err
	}

	// A group in flight holds EVERY one of its nodes, including in the window
	// where only the root has a pod. The workers' nodes are committed from the
	// moment the root starts: they must not be handed to another unit, and they
	// must not have their cordons released out from under a rendezvous that is
	// about to land pods on them.
	for _, pod := range pods {
		if pod != nil && podLive(pod) {
			for _, n := range nodes {
				busy[n] = true
			}
			break
		}
	}

	switch {
	case pods[groupRootRank] == nil:
		return r.startGroup(ctx, run, p, index, attempt, t, nodes, launch, busy, capNodes)
	case anyNil(pods[1:]):
		return r.gateWorkersOnRoot(ctx, run, p, index, attempt, t, nodes, pods, launch)
	default:
		return r.harvestGroup(ctx, run, p, t, nodes, attempt, pods)
	}
}

func anyNil(pods []*corev1.Pod) bool {
	for _, pod := range pods {
		if pod == nil {
			return true
		}
	}
	return false
}

// startGroup admits the group and launches rank 0.
//
// THE INTERLOCK IS CHECKED ONCE, HERE, FOR EVERY NODE. A group is an indivisible
// unit of load: the test is the collective across N nodes, so it holds all N for
// its whole duration, and maxConcurrentNodes counts nodes. Admitting the unit
// therefore costs N slots and the group waits until all N are free — which also
// means a Group test can never run under a cap smaller than its target list, and
// buildPlan refuses the run at start with that explanation instead of letting it
// hang until its deadline.
//
// The workers are NOT re-checked against the cap when they are created later.
// Once the root is up the unit has been admitted and the load is committed;
// refusing the rest would leave the root burning its clock against a collective
// that can never form, and report an Error about hardware that is perfectly
// fine. spec.cancel with CancelImmediate is the tool for getting load off the
// floor right now.
func (r *BurnInRunReconciler) startGroup(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	index int,
	attempt int32,
	t plannedTest,
	nodes []string,
	launch bool,
	busy map[string]bool,
	capNodes int,
) (advanceState, advanceEffect, error) {
	var none advanceEffect

	if !launch {
		return advancePending, none, nil
	}

	need := 0
	for _, n := range nodes {
		if !busy[n] {
			need++
		}
	}
	if len(busy)+need > capNodes {
		return advancePending, none, nil
	}

	svc := headlessServiceForRendezvous(run, index, attempt, t.Name)
	if err := controllerutil.SetControllerReference(run, svc, r.Scheme); err != nil {
		return advancePending, none, err
	}
	if err := r.Create(ctx, svc); err != nil && !apierrors.IsAlreadyExists(err) {
		return advancePending, none, err
	}

	pod, buildErr := podForTest(run, index, attempt, t.Name, &t.Spec, t.Axes, nodes[groupRootRank], r.vendorFor(ctx, nodes[groupRootRank]), run.Spec.Target,
		groupRendezvous(groupRootRank, nodes, svc.Name))
	if buildErr != nil {
		// Unbuildable pod (no image for the kind). Asking again cannot fix it,
		// and it is machinery, not hardware.
		r.completeAttempt(ctx, run, p, t, nodes, attempt, "", nil,
			runner.Result{Verdict: runner.VerdictError, Message: buildErr.Error()})
		return advanceHarvested, advanceEffect{dirty: true}, nil
	}

	// EVERY node is held before ANY pod exists. A group is admitted as a unit, so
	// it is cordoned as a unit: cordoning only the root would leave every worker
	// node accepting workload right up to the moment a rank lands on it, and a
	// competing workload on one rank perturbs the measurement for all of them.
	cordoned, cordonErr := r.cordonWave(ctx, run, nodes, &t.Spec)
	if cordonErr != nil {
		return advancePending, advanceEffect{dirty: cordoned}, cordonErr
	}
	if err := controllerutil.SetControllerReference(run, pod, r.Scheme); err != nil {
		return advancePending, advanceEffect{dirty: cordoned}, err
	}
	if err := r.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
		return advancePending, advanceEffect{dirty: cordoned}, err
	}
	for _, n := range nodes {
		busy[n] = true
	}
	return advanceRunning, advanceEffect{dirty: cordoned, rendezvous: true}, nil
}

// gateWorkersOnRoot is the ready gate: ranks 1..N-1 are created only once rank 0
// reports Ready.
//
// Same rule as Pair's, and load-bearing for the same reason: a rank that tries
// to fetch the collective's bootstrap handle before the root has bound its
// socket dies with a connection error, and a connection error on a fabric test
// is indistinguishable, in a report, from a fabric that is genuinely broken.
//
// The workers are created TOGETHER rather than one per pass. A collective makes
// no progress until every rank has joined, so staggering them would only add
// reconcile latency to a window in which N nodes are already cordoned and idle.
func (r *BurnInRunReconciler) gateWorkersOnRoot(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	index int,
	attempt int32,
	t plannedTest,
	nodes []string,
	pods []*corev1.Pod,
	launch bool,
) (advanceState, advanceEffect, error) {
	var none advanceEffect
	root := pods[groupRootRank]

	if _, terminated, _, _ := podOutcome(root); terminated {
		// The root finished before the collective existed, so nothing was
		// measured. Settle from the reports there are; combineGroup knows that a
		// clean root exit with no workers is not a pass.
		members := r.harvestMembers(ctx, t, nodes, pods)
		logErr := members.logErr
		r.completeAttempt(ctx, run, p, t, nodes, attempt, root.Name, root,
			combineGroup(members.members, logErr, &t.Spec))
		// EVERY rank, deferred until the record is durable. A collective is one
		// unit: losing the record here destroys N live ranks and restarts the
		// whole thing having spent no retry budget.
		return advanceHarvested, advanceEffect{dirty: true, kill: pods}, nil
	}

	if r.podOverdue(root, &t.Spec) {
		r.completeAttempt(ctx, run, p, t, nodes, attempt, root.Name, root, runner.Result{
			Verdict: runner.VerdictError,
			Message: fmt.Sprintf(
				"rank %d (the root) on %s never became ready within its window (last phase %q) — "+
					"the other %d rank(s) were never started and the collective was not measured; no verdict%s",
				groupRootRank, nodes[groupRootRank], root.Status.Phase, len(nodes)-1, pendingDetail(root)),
		})
		return advanceHarvested, advanceEffect{dirty: true, kill: pods}, nil
	}

	if !podStarted(root) {
		// Scheduled, not executing. Nothing has been tested, so no result is
		// opened and StartedAt stays unset.
		return advanceRunning, advanceEffect{rendezvous: true}, nil
	}

	// The group's occupancy of the hardware begins when the root starts.
	_, opened := r.beginAttempt(run, t, nodes, attempt, root)

	if !podReady(root) {
		return advanceRunning, advanceEffect{dirty: opened, rendezvous: true}, nil
	}
	if !launch {
		// Suspended, or draining a cancel. The root keeps running and will be
		// harvested or cleaned up; starting the rest of a unit that has been told
		// to stop would be new load.
		return advancePending, advanceEffect{dirty: opened}, nil
	}

	service := rendezvousServiceName(run, index, attempt)
	for rank := 1; rank < len(nodes); rank++ {
		if pods[rank] != nil {
			continue
		}
		pod, buildErr := podForTest(run, index, attempt, t.Name, &t.Spec, t.Axes, nodes[rank], r.vendorFor(ctx, nodes[rank]), run.Spec.Target,
			groupRendezvous(rank, nodes, service))
		if buildErr != nil {
			r.completeAttempt(ctx, run, p, t, nodes, attempt, root.Name, root,
				runner.Result{Verdict: runner.VerdictError, Message: buildErr.Error()})
			return advanceHarvested, advanceEffect{dirty: true}, nil
		}
		if err := controllerutil.SetControllerReference(run, pod, r.Scheme); err != nil {
			return advancePending, none, err
		}
		if err := r.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
			return advancePending, none, err
		}
	}
	return advanceRunning, advanceEffect{dirty: opened}, nil
}

// harvestGroup waits for every rank and then settles the collective.
//
// See the package comment for why this waits for ALL of them rather than for one
// deciding rank the way Pair does.
func (r *BurnInRunReconciler) harvestGroup(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	t plannedTest,
	nodes []string,
	attempt int32,
	pods []*corev1.Pod,
) (advanceState, advanceEffect, error) {
	root := pods[groupRootRank]

	// A rank that never STARTED and a rank that started and hung are different
	// findings and send an engineer to different places: the first is scheduling
	// — an unschedulable node, a stuck image pull — and the second is the node
	// under test. Reporting both as "never finished" buries the distinction in
	// the one message somebody reads off a failed collective.
	var stillRunning []string
	for rank, pod := range pods {
		if _, done, _, _ := podOutcome(pod); done {
			continue
		}
		what := "never finished"
		if pod == nil || !podStarted(pod) {
			what = "never started" + pendingDetail(pod)
		}
		stillRunning = append(stillRunning, fmt.Sprintf("rank %d (%s) %s", rank, nodes[rank], what))
	}

	if len(stillRunning) > 0 {
		// A hang is attributed to the ranks that are actually hung, which is the
		// most useful sentence a group failure can produce: on a collective,
		// every OTHER rank blocks waiting for the one that never arrived, so
		// without this the report would name all of them equally.
		if r.anyOverdue(pods, &t.Spec) {
			r.completeAttempt(ctx, run, p, t, nodes, attempt, root.Name, root, runner.Result{
				Verdict: runner.VerdictError,
				Message: fmt.Sprintf(
					"the collective did not complete within its window — %d of %d rank(s) did not report (%s); "+
						"every other rank blocks on a rank that never arrives, so this names the ones that did not; no verdict",
					len(stillRunning), len(nodes), strings.Join(stillRunning, ", ")),
			})
			return advanceHarvested, advanceEffect{dirty: true, kill: pods}, nil
		}
		_, opened := r.beginAttempt(run, t, nodes, attempt, root)
		// Checkpoint from the root, because that is the conventional reporting
		// rank. resultFor matches any node in the result's Nodes, so this
		// refreshes the one shared group result.
		checkpointed := r.checkpoint(ctx, run, t, nodes[groupRootRank], root)
		return advanceRunning, advanceEffect{dirty: opened || checkpointed, checkpointed: checkpointed}, nil
	}

	// Every rank has reported. Open the result if the root was never observed
	// running — the execution plainly happened.
	r.beginAttempt(run, t, nodes, attempt, root)

	harvested := r.harvestMembers(ctx, t, nodes, pods)
	combined := combineGroup(harvested.members, harvested.logErr, &t.Spec)
	r.completeAttempt(ctx, run, p, t, nodes, attempt, root.Name, root, combined)
	// EVERY rank. A collective's evidence is per-rank by nature — the whole
	// point of a Group verdict is that one rank's report does not describe the
	// others — so a ref list naming only the root would hide the ranks a reader
	// most needs when the collective hung.
	for _, m := range harvested.members {
		if m.result != nil {
			r.harvestArtifacts(ctx, run, t.Name, nodes, m.pod, m.result.Artifacts)
		}
	}
	return advanceHarvested, advanceEffect{dirty: true}, nil
}

// anyOverdue reports whether any rank has outlived its whole window. One rank
// past its deadline condemns the unit, because a collective missing a member is
// not a collective.
func (r *BurnInRunReconciler) anyOverdue(pods []*corev1.Pod, spec *burninv1alpha1.BurnInTestSpec) bool {
	for _, pod := range pods {
		if pod != nil && r.podOverdue(pod, spec) {
			return true
		}
	}
	return false
}

type harvestedGroup struct {
	members []groupMember
	logErr  error
}

// harvestMembers parses every rank that terminated. A rank that has not
// terminated contributes a nil result, which combineGroup reads as "did not
// report" rather than as silence to be ignored.
func (r *BurnInRunReconciler) harvestMembers(
	ctx context.Context,
	t plannedTest,
	nodes []string,
	pods []*corev1.Pod,
) harvestedGroup {
	out := harvestedGroup{members: make([]groupMember, 0, len(nodes))}
	for rank, pod := range pods {
		m := groupMember{rank: rank, node: nodes[rank]}
		if pod != nil {
			m.pod = pod.Name
			if _, done, _, _ := podOutcome(pod); done {
				parsed, logErr := r.harvestPod(ctx, t, pod)
				m.result = &parsed
				if out.logErr == nil {
					out.logErr = logErr
				}
			}
		}
		out.members = append(out.members, m)
	}
	return out
}

// groupPods fetches every rank's pod, indexed by rank. A missing pod is nil: not
// existing yet is the normal state of a rendezvous in progress, not an error.
func (r *BurnInRunReconciler) groupPods(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	index int,
	attempt int32,
	nodes []string,
) ([]*corev1.Pod, error) {
	pods := make([]*corev1.Pod, len(nodes))
	for rank, node := range nodes {
		var pod corev1.Pod
		name := podNameForRole(run, index, node, attempt, groupRole(rank))
		err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, &pod)
		switch {
		case apierrors.IsNotFound(err):
			continue
		case err != nil:
			return nil, err
		}
		pods[rank] = pod.DeepCopy()
	}
	return pods, nil
}

// groupRendezvous builds one rank's half of the rendezvous contract.
func groupRendezvous(rank int, nodes []string, service string) *rendezvous {
	return &rendezvous{
		scope:    burninv1alpha1.ScopeGroup,
		role:     groupRole(rank),
		service:  service,
		rank:     rank,
		nranks:   len(nodes),
		rootNode: nodes[groupRootRank],
	}
}

// combineGroup folds every rank's report into the single result the collective
// is judged on. See the package comment for the precedence rule.
func combineGroup(members []groupMember, logErr error, spec *burninv1alpha1.BurnInTestSpec) runner.Result {
	// The FOLD is pkg/group's, so cmd/burnin elects the same verdict and the
	// same metric values from the same eight rank reports. What stays here is
	// the MESSAGE — this side has a pod name to quote and the bare-metal side
	// has a file path, and neither is a decision.
	c := burningroup.Combine(rankReports(members))

	out := runner.Result{
		Metrics:      c.Metrics,
		Unmeasurable: c.Unmeasurable,
		InvalidNames: c.InvalidNames,
		Verdict:      c.Verdict,
		ExitCode:     c.ExitCode,
	}

	// THE ORDER HERE IS THE POINT. groupMessage is the unbounded half — it embeds
	// each rank's own last stdout line, which nothing caps — so IT is clamped,
	// and the operator-constructed note is appended AFTER and never truncated.
	//
	// The reverse order is what an earlier revision did, and it defeated the fix
	// it shipped alongside: the note was appended to the tail and the clamp cut
	// the tail, so on exactly the long messages the clamp exists for, the note
	// went and the runner spew stayed. Since the merge deliberately leaves
	// out.Unmeasurable empty when another rank measured the key, that note is the
	// ONLY record that a rank declared the metric unmeasurable — so truncating it
	// restored the certification it was written to prevent, silently, in the case
	// hardest to notice.
	//
	// Deliberate clauses outlive verbose ones. Everything appended below this
	// line is something this operator chose to say.
	out.Message = clampRankSummaries(groupMessage(members))
	if note := disagreementNote(readingsToRankValues(c.Readings), out.Metrics); note != "" {
		out.Message = strings.TrimSpace(out.Message + " [" + note + "]")
	}

	// The log-fetch rule, applied ONCE to the combined result rather than per
	// rank. A fetch failure only invalidates the verdict if it actually cost a
	// number a threshold needs: when rank 0's log came through with every
	// thresholded metric in it, an unreadable rank-7 log changes nothing, and
	// erroring on it would throw away a good measurement.
	if logErr != nil && out.Verdict == runner.VerdictPass && len(spec.Thresholds) > 0 && missingThresholdMetric(out, spec.Thresholds) {
		out.Verdict = runner.VerdictError
		out.Message = fmt.Sprintf("runner logs unavailable (%v) — thresholds cannot be evaluated; no verdict [%s]", logErr, out.Message)
	}
	return out
}

// rankReports adapts this package's members to the shared fold's input.
func rankReports(members []groupMember) []burningroup.Rank {
	out := make([]burningroup.Rank, 0, len(members))
	for _, m := range members {
		out = append(out, burningroup.Rank{Rank: m.rank, Node: m.node, Result: m.result})
	}
	return out
}

// readingsToRankValues adapts the shared fold's readings to what
// disagreementNote already takes. The note is presentation and stays here; WHO
// SAID WHAT is a fact about the run and comes from the fold.
func readingsToRankValues(in map[string][]burningroup.Reading) map[string][]rankValue {
	out := make(map[string][]rankValue, len(in))
	for k, vals := range in {
		for _, v := range vals {
			out[k] = append(out[k], rankValue{rank: v.Rank, value: v.Value})
		}
	}
	return out
}

// groupMessage states the verdict's subject before it states the evidence.
//
// The lead clause is not decoration, for the same reason pairMessage has one: a
// reader who sees a failure next to eleven node names will pick one of them, and
// on a collective that is nearly always the wrong one — every rank except the
// faulty one reports a timeout waiting for it.
//
// Ranks that agree with the verdict are summarised by COUNT rather than listed.
// A twelve-node group whose every rank says the same thing produces a status
// message the apiserver would refuse, and a list that lied about its own
// truncation would be worse than no list; the ranks that DISSENT are always
// named in full, because those are the ones worth walking to a rack for.
func groupMessage(members []groupMember) string {
	head := fmt.Sprintf("group of %d (this verdict is about the COLLECTIVE, not about any one rank)", len(members))

	byVerdict := map[string][]string{}
	var order []string
	for _, m := range members {
		key := "did not report"
		if m.result != nil {
			key = strings.ToLower(string(m.result.Verdict))
		}
		if _, seen := byVerdict[key]; !seen {
			order = append(order, key)
		}
		byVerdict[key] = append(byVerdict[key], m.summary())
	}
	sort.Strings(order)

	parts := make([]string, 0, len(order))
	for _, key := range order {
		lines := byVerdict[key]
		// The whole group agreeing is one clause — but it still carries ONE
		// rank's words as the exemplar, and that is not decoration.
		//
		// A bare "all 12 ranks error" is the shape every real infrastructure
		// fault takes (a missing host mount, an unpullable image, a driver
		// skew), so collapsing it to a count discarded the runner's own
		// explanation in precisely the case where every rank had the same one.
		// The stored result then explained nothing, and the pod logs that did
		// are gone at the run's TTL. Keeping one line costs a bounded amount and
		// is the difference between a record somebody can act on and a record
		// somebody has to reproduce.
		if len(lines) == len(members) && len(members) > groupMaxNamedRanks {
			parts = append(parts, fmt.Sprintf("all %d ranks %s, e.g. %s", len(members), key, lines[0]))
			continue
		}
		if len(lines) > groupMaxNamedRanks {
			parts = append(parts, fmt.Sprintf("%d ranks %s (%s, and %d more)",
				len(lines), key, strings.Join(lines[:groupMaxNamedRanks], "; "), len(lines)-groupMaxNamedRanks))
			continue
		}
		parts = append(parts, strings.Join(lines, "; "))
	}
	return head + ": " + strings.Join(parts, "; ")
}

// groupMaxNamedRanks bounds how many ranks sharing one verdict are spelled out.
// The count is always exact even where the detail is elided — a status write the
// apiserver refuses would lose the verdict entirely, which is a worse outcome
// than a summarised one.
const groupMaxNamedRanks = 4

// rankValue is one rank's answer for one metric key.
type rankValue struct {
	rank  int
	value string
}

// disagreementNote reports the metric keys several ranks answered differently.
//
// It is appended to the result's message rather than emitted as a metric,
// because a metric would need a registered name and a value, and the value here
// is "these ranks said other things" — which is prose. The message is what a
// human reads next to the node list, and what the delivered envelope carries, so
// it outlives the pods.
//
// Sorted and bounded: a group of sixty-four disagreeing about a dozen keys must
// not write a status the apiserver refuses, and the counts stay exact even where
// the detail is elided.
func disagreementNote(reported map[string][]rankValue, kept map[string]string) string {
	var keys []string
	for k, vals := range reported {
		for _, rv := range vals {
			if rv.value != kept[k] {
				keys = append(keys, k)
				break
			}
		}
	}
	if len(keys) == 0 {
		return ""
	}
	sort.Strings(keys)

	shown := keys
	if len(shown) > groupMaxNamedRanks {
		shown = shown[:groupMaxNamedRanks]
	}
	parts := make([]string, 0, len(shown)+1)
	for _, k := range shown {
		var others []string
		for _, rv := range reported[k] {
			if rv.value != kept[k] {
				others = append(others, fmt.Sprintf("rank %d=%s", rv.rank, rv.value))
			}
		}
		sort.Strings(others)
		total := len(others)
		if total > groupMaxNamedRanks {
			others = append(append([]string{}, others[:groupMaxNamedRanks]...),
				fmt.Sprintf("and %d more", total-groupMaxNamedRanks))
		}
		// HOW the value was reached, not merely which rank held it. Since #121 a
		// combined key may hold a value NO rank reported — a Sum is nobody's
		// reading — so naming a rank as its source would be false. The rule that
		// produced it is the honest attribution, and for the keys that are still
		// elected rather than computed the rank is still named.
		if v, stored := kept[k]; stored {
			switch c := contract.CombinationFor(k); c {
			case contract.CombineSum, contract.CombineMax, contract.CombineMin:
				// EVERY rank, not only the dissenters. A combined value may be
				// one no rank reported, and for a Max the rank that set it is
				// precisely the one worth naming — listing only the others would
				// hide the hot node in a note about the hot node.
				// The rank(s) holding the elected value FIRST, then the rest.
				// Truncation cuts the tail, and on a Max the rank that set the
				// value is the node an engineer has to go and look at — sorting
				// the list plainly put "rank 10" ahead of "rank 2" and cut the
				// hot one.
				var lead, rest []string
				for _, rv := range reported[k] {
					entry := fmt.Sprintf("rank %d=%s", rv.rank, rv.value)
					if rv.value == v {
						lead = append(lead, entry)
					} else {
						rest = append(rest, entry)
					}
				}
				sort.Strings(lead)
				sort.Strings(rest)
				all := append(lead, rest...)
				if len(all) > groupMaxNamedRanks {
					all = append(append([]string{}, all[:groupMaxNamedRanks]...),
						fmt.Sprintf("and %d more", len(all)-groupMaxNamedRanks))
				}
				parts = append(parts, fmt.Sprintf("%s combined to %s by %s across %s",
					k, v, c, strings.Join(all, ", ")))
			default:
				keptBy := -1
				for _, rv := range reported[k] {
					if rv.value == v && (keptBy < 0 || rv.rank < keptBy) {
						keptBy = rv.rank
					}
				}
				parts = append(parts, fmt.Sprintf("%s recorded as %s (rank %d) but %s",
					k, v, keptBy, strings.Join(others, ", ")))
			}
			continue
		}
		// Dropped: no rule says how this metric combines across ranks, so no
		// single value may stand for the group and a threshold on it fails closed.
		// Terse on purpose: a status grows per attempt, and the reasoning behind
		// each of these belongs in this file rather than in every stored record.
		why := "no across-rank rule"
		for _, rv := range reported[k] {
			if rv.value == runner.Unmeasurable {
				why = "a rank could not measure it"
				break
			}
		}
		// The rank that could not measure it leads, for the same reason the hot
		// rank leads above: it is the node this note exists to name, and the
		// tail is what truncation takes.
		var lead, rest []string
		for _, rv := range reported[k] {
			entry := fmt.Sprintf("rank %d=%s", rv.rank, rv.value)
			if rv.value == runner.Unmeasurable {
				lead = append(lead, entry)
			} else {
				rest = append(rest, entry)
			}
		}
		sort.Strings(lead)
		sort.Strings(rest)
		shownRanks := append(lead, rest...)
		if len(shownRanks) > groupMaxNamedRanks {
			shownRanks = append(append([]string{}, shownRanks[:groupMaxNamedRanks]...),
				fmt.Sprintf("and %d more", len(shownRanks)-groupMaxNamedRanks))
		}
		parts = append(parts, fmt.Sprintf("%s DROPPED as unmeasurable (%s; ranks reported %s)",
			k, why, strings.Join(shownRanks, ", ")))
	}
	if elided := len(keys) - len(shown); elided > 0 {
		parts = append(parts, fmt.Sprintf("and %d more metric(s)", elided))
	}
	return fmt.Sprintf("ranks DISAGREED about %d metric(s), combined by each metric's "+
		"registered rule: %s. A per-node reading cannot be attributed to a group — see issue #121",
		len(keys), strings.Join(parts, "; "))
}

// maxGroupMessage bounds the per-rank SUMMARIES of one group execution — the
// half that carries text this operator did not write.
//
// Generous, because every clause in it is there to stop a misattribution and
// truncating early would trade one failure for another — but finite, because an
// unwritable status loses the verdict entirely rather than the tail of a
// sentence. The truncation says so, so a reader never trusts a sentence that was
// cut.
const maxGroupMessage = 4096

func clampRankSummaries(s string) string {
	if len(s) <= maxGroupMessage {
		return s
	}
	return truncateAtRune(s, maxGroupMessage,
		"… (rank detail truncated; see the pods' logs while the run's TTL holds them)")
}

// truncateAtRune cuts s to at most max BYTES, never mid-rune, and is the ONE
// place this package truncates anything.
//
// It exists because the same mistake was made twice in one commit and once
// before that. Three functions each sliced at a raw byte offset — and these
// messages are full of em dashes, because this project's own runners join their
// prose with them — so a status could carry a half-written UTF-8 sequence into
// an apiserver and a JSON envelope, where it becomes U+FFFD and stops being what
// the runner said.
//
// THE BACK-OFF SCANS THE TAIL, NOT THE PREFIX, and that distinction is the whole
// of the second bug. An earlier version looped on `!utf8.ValidString(s[:cut])`,
// which inspects the ENTIRE prefix — so it is false for every cut at or beyond
// the first invalid byte anywhere in the string, and the loop walked all the way
// back to just before that byte instead of at most three. One em dash near the
// start of a 20-rank summary cost 3,673 bytes of a 4,096-byte budget and left
// the stored result naming ONE rank out of twenty. utf8.RuneStart moves the cut
// by at most 3, which is the property wanted; it is also the idiom already in
// runners/dcgm-diag/report.go, written for this exact reason.
func truncateAtRune(s string, max int, suffix string) string {
	if len(s) <= max {
		return s
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + suffix
}

// dedupeStrings keeps first-seen order and drops repeats.
func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	return out
}

// combineRanks and combineNumeric moved to pkg/group, where cmd/burnin can
// reach them: two dispatchers electing DIFFERENT values from the same rank
// reports, under one metric name, is a fleet dashboard nobody can read.

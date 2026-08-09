package controller

import (
	"context"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// Pair-scope execution: an operator-native two-pod rendezvous.
//
// # Shape
//
// A Pair test is ONE execution across TWO nodes. The first target runs the
// server, the second runs the client, both pinned by hostname nodeSelector
// exactly as a Node-scope pod is, and both addressable through one headless
// Service that gives each pod its own DNS name. The client is not created until
// the server pod is Ready. The pair produces a SINGLE TestResult naming both
// nodes.
//
// # Why not JobSet
//
// At n=2 gang scheduling buys nothing that two pinned pods and a headless
// Service do not already provide, and it would put a controller dependency
// between this operator and every cluster that installs it. JobSet earns its
// place at Group scope, where rank assignment and gang semantics are real
// problems; here it would be a dependency taken for symmetry. This is a
// deliberate decision recorded in GEP-0178, not an omission.
//
// # The verdict rule, and why it is not per-node
//
// A point-to-point measurement is a property of the LINK. Splitting it into two
// per-node verdicts would produce two claims neither endpoint can support on its
// own, and would send an engineer to swap a NIC when the fault was a cable, a
// transceiver, a switch port, or the peer. So: one result, Nodes = [server,
// client], and the message names what each end reported without ever electing
// one of them the culprit.
//
// # Precedence when the two ends disagree
//
//	Error > Fail > Skip > Pass
//
// Error outranks Fail on purpose, and this is the one ordering worth arguing
// about. If either end's machinery broke — image pull, log fetch, deadline kill,
// an exit code outside the contract — the link was never measured, and the
// client's "connection refused" is then an artifact of its peer rather than
// evidence about the fabric. Recording that as Fail would permanently indict a
// link because of an unpullable image, and Fail is the one phase this operator
// never retries; Error is retryable under RetryOnErrorLimit, which is exactly
// the right response to machinery that malfunctioned. Nothing passes by
// omission either way, because a required Error still drives the run to Error.
//
// Skip outranks Pass but not Fail. An end that positively declares the test
// inapplicable (exit 2, after looking at its own hardware) makes the pair
// inapplicable; but a Skip must never erase a Fail, because that would launder a
// measured fault into "not applicable".
//
// # Metrics
//
// Merged, with the CLIENT winning any key both ends report. The client is the
// measuring side for every kind that runs here — ib_write_bw prints its table on
// the client, nccl-tests reports bus bandwidth from the rank that drives the
// collective — and a server-side echo of the same key is at best a duplicate.
// Thresholds are then evaluated ONCE, against the merged metrics, by the same
// fail-closed evaluator every Node-scope test goes through.
const (
	pairRoleServer = "server"
	pairRoleClient = "client"
)

// rendezvousPollInterval is the backstop wake-up while a unit is mid-rendezvous:
// a Pair whose server exists and whose client has not started, or a Group whose
// root is up and whose workers have not been created.
//
// It is ONE constant for both scopes, reached through advanceEffect.rendezvous
// rather than through either scope's own code. Group briefly had an alias of its
// own — declared, documented, and never referenced, because the shared wake-up
// path in reconcileRunning was already clamping to this value. A second name for
// one number is a number that will eventually be two.
//
// Pod events drive the common case — the controller Owns pods, so the server
// going Ready re-triggers a reconcile immediately — and this only covers the
// event that never arrives. It is shorter than waitingPollInterval because the
// window it covers is dead time on hardware the run is already holding, and
// longer than a spin because the reconciler must not poll the API server in a
// loop waiting for a container to bind a socket.
const rendezvousPollInterval = 5 * time.Second

// pairSide is one endpoint's report. A nil result means that end never produced
// one, which is a different statement from "it reported nothing useful" and is
// treated differently below.
type pairSide struct {
	role   string
	node   string
	pod    string
	result *runner.Result
}

func (s pairSide) summary() string {
	if s.result == nil {
		return fmt.Sprintf("%s %s did not report", s.role, s.node)
	}
	out := fmt.Sprintf("%s %s %s (exit %d)", s.role, s.node, strings.ToLower(string(s.result.Verdict)), s.result.ExitCode)
	// An undeclared exit 2 is explained rather than quoted. Without this the
	// line reads "server spark-a error (exit 2): /src/main.go:132 +0x1d" — a
	// stack frame standing in for the reason a link went unmeasured, in the one
	// record that outlives every Event about it. The runner's own last line is
	// kept as context because for a panic it is the only clue to what died.
	if s.result.UndeclaredSkip {
		out += ": " + undeclaredSkipBrief
		if m := strings.TrimSpace(s.result.Message); m != "" {
			out += " [runner said: " + m + "]"
		}
		return out
	}
	if m := strings.TrimSpace(s.result.Message); m != "" {
		out += ": " + m
	}
	return out
}

// advancePair moves one Pair-scope execution forward by exactly one step. It is
// the Pair counterpart of advance, and it returns the same states so the pass
// bookkeeping in execute does not care which one produced them.
func (r *BurnInRunReconciler) advancePair(
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

	// Roles come from the PINNED plan's target order, so they are stable across
	// reconciles and across a manager restart. If they were re-derived from a
	// live selector, a node list that reordered would swap server and client
	// mid-test and orphan both pods.
	serverNode, clientNode := p.Targets[0], p.Targets[1]
	nodes := []string{serverNode, clientNode}

	res := resultFor(run, t.Name, serverNode)
	if res != nil && res.Phase.IsTerminal() {
		return advanceDone, none, nil
	}
	attempt, _ := nextAttempt(res)

	serverPod, err := r.pairPod(ctx, run, index, attempt, serverNode, pairRoleServer)
	if err != nil {
		return advancePending, none, err
	}
	clientPod, err := r.pairPod(ctx, run, index, attempt, clientNode, pairRoleClient)
	if err != nil {
		return advancePending, none, err
	}

	// A pair in flight holds BOTH of its nodes, even in the window where only
	// one pod exists. That is not bookkeeping generosity: the client's node is
	// committed from the moment the server starts, it must not be handed to
	// another unit, and it must not have its cordon released out from under a
	// rendezvous that is about to land a pod on it.
	if (serverPod != nil && podLive(serverPod)) || (clientPod != nil && podLive(clientPod)) {
		busy[serverNode], busy[clientNode] = true, true
	}

	switch {
	case serverPod == nil:
		return r.startPair(ctx, run, p, index, attempt, t, nodes, launch, busy, capNodes)
	case clientPod == nil:
		return r.gateClientOnServer(ctx, run, p, index, attempt, t, nodes, serverPod, launch)
	default:
		return r.harvestPair(ctx, run, p, t, nodes, attempt, serverPod, clientPod)
	}
}

// startPair admits the pair and launches the server.
//
// THE INTERLOCK IS CHECKED ONCE, HERE, FOR BOTH NODES. A pair is an indivisible
// unit of load: the test is the traffic between two nodes, so it holds both for
// its whole duration, and maxConcurrentNodes counts nodes. Admitting the unit
// therefore costs two slots and the pair waits until both are free — which also
// means that at the default cap of 1 a Pair test cannot run at all, and the run
// says so at start (see buildPlan) instead of hanging until its deadline.
//
// The client is NOT re-checked against the cap when it is created later. Once
// the server is up the unit has been admitted and the load is committed;
// refusing the second half would leave the server burning its clock against a
// peer that will never arrive, and report an Error about hardware that is
// perfectly fine. An operator who needs load off the floor right now has
// spec.cancel with CancelImmediate, which is the tool built for that.
func (r *BurnInRunReconciler) startPair(
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
	serverNode, clientNode := nodes[0], nodes[1]

	if !launch {
		return advancePending, none, nil
	}

	need := 0
	if !busy[serverNode] {
		need++
	}
	if !busy[clientNode] {
		need++
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

	pod, buildErr := podForTest(run, index, attempt, t.Name, &t.Spec, t.Axes, serverNode, run.Spec.Target, &rendezvous{
		scope:    burninv1alpha1.ScopePair,
		role:     pairRoleServer,
		service:  svc.Name,
		peerRole: pairRoleClient,
		peerNode: clientNode,
	})
	if buildErr != nil {
		// Unbuildable pod (no image for the kind). Asking again cannot fix it,
		// and it is machinery, not hardware.
		r.completeAttempt(ctx, run, p, t, nodes, attempt, "", nil,
			runner.Result{Verdict: runner.VerdictError, Message: buildErr.Error()})
		return advanceHarvested, advanceEffect{dirty: true}, nil
	}

	// BOTH nodes are held before either pod exists. A pair is admitted as a
	// unit, so it is cordoned as a unit: cordoning only the server would leave
	// the client's node accepting workload right up to the moment the client
	// lands on it, and the traffic it measures is exactly what a competing
	// workload would perturb.
	cordoned, cordonErr := r.cordonWave(ctx, run, nodes)
	if cordonErr != nil {
		return advancePending, advanceEffect{dirty: cordoned}, cordonErr
	}
	if err := controllerutil.SetControllerReference(run, pod, r.Scheme); err != nil {
		return advancePending, advanceEffect{dirty: cordoned}, err
	}
	if err := r.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
		return advancePending, advanceEffect{dirty: cordoned}, err
	}
	// Both nodes are spoken for from this moment, even though only one has a
	// pod: the client's slot was admitted with the server's.
	busy[serverNode], busy[clientNode] = true, true
	return advanceRunning, advanceEffect{dirty: cordoned, rendezvous: true}, nil
}

// gateClientOnServer is the ready gate: the client pod is created only once the
// server pod reports Ready.
//
// This is the single most load-bearing step in Pair execution. An ib_write_bw or
// nccl client that starts before its server is listening dies with a connection
// error, and a connection error on a fabric test is indistinguishable, in a
// report, from a fabric that is genuinely broken. Racing the two pods would
// therefore manufacture hardware failures.
func (r *BurnInRunReconciler) gateClientOnServer(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	index int,
	attempt int32,
	t plannedTest,
	nodes []string,
	serverPod *corev1.Pod,
	launch bool,
) (advanceState, advanceEffect, error) {
	var none advanceEffect
	serverNode, clientNode := nodes[0], nodes[1]

	if _, terminated, _, _ := podOutcome(serverPod); terminated {
		// The server finished before the client existed, so the link was never
		// exercised. Settle from the one report there is; combinePair knows
		// that a clean server exit with no client is not a pass.
		parsed, logErr := r.harvestPod(ctx, t, serverPod)
		combined := combinePair(
			pairSide{role: pairRoleServer, node: serverNode, pod: serverPod.Name, result: &parsed},
			pairSide{role: pairRoleClient, node: clientNode},
			logErr, &t.Spec,
		)
		r.completeAttempt(ctx, run, p, t, nodes, attempt, serverPod.Name, serverPod, combined)
		r.harvestArtifacts(ctx, run, t.Name, serverPod.Name, parsed.Artifacts)
		return advanceHarvested, advanceEffect{dirty: true}, nil
	}

	if r.podOverdue(serverPod, &t.Spec) {
		if err := r.Delete(ctx, serverPod); err != nil && !apierrors.IsNotFound(err) {
			return advancePending, none, err
		}
		r.completeAttempt(ctx, run, p, t, nodes, attempt, serverPod.Name, serverPod, runner.Result{
			Verdict: runner.VerdictError,
			Message: fmt.Sprintf(
				"the %s endpoint on %s never became ready within its window (last phase %q) — the client was never started and the link was not measured; no verdict",
				pairRoleServer, serverNode, serverPod.Status.Phase),
		})
		return advanceHarvested, advanceEffect{dirty: true}, nil
	}

	if !podStarted(serverPod) {
		// Scheduled, not executing. Nothing has been tested, so no result is
		// opened and StartedAt stays unset.
		return advanceRunning, advanceEffect{rendezvous: true}, nil
	}

	// The pair's occupancy of the hardware begins when the server starts: from
	// here on both nodes are held and one of them is running a process.
	_, opened := r.beginAttempt(run, t, nodes, attempt, serverPod)

	if !podReady(serverPod) {
		return advanceRunning, advanceEffect{dirty: opened, rendezvous: true}, nil
	}
	if !launch {
		// Suspended or draining a cancel. The server keeps running and will be
		// harvested or cleaned up; starting the second half of a unit that has
		// been told to stop would be new load.
		return advancePending, advanceEffect{dirty: opened}, nil
	}

	pod, buildErr := podForTest(run, index, attempt, t.Name, &t.Spec, t.Axes, clientNode, run.Spec.Target, &rendezvous{
		scope:    burninv1alpha1.ScopePair,
		role:     pairRoleClient,
		service:  rendezvousServiceName(run, index, attempt),
		peerRole: pairRoleServer,
		peerNode: serverNode,
	})
	if buildErr != nil {
		r.completeAttempt(ctx, run, p, t, nodes, attempt, serverPod.Name, serverPod,
			runner.Result{Verdict: runner.VerdictError, Message: buildErr.Error()})
		return advanceHarvested, advanceEffect{dirty: true}, nil
	}
	if err := controllerutil.SetControllerReference(run, pod, r.Scheme); err != nil {
		return advancePending, none, err
	}
	if err := r.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
		return advancePending, none, err
	}
	return advanceRunning, advanceEffect{dirty: opened}, nil
}

// harvestPair waits for the client and then settles the pair.
//
// THE CLIENT IS THE DECIDING SIDE. It is the end that measures — perftest and
// nccl-tests both report from the client — and it is also the end that
// terminates when the test is over. A server may legitimately linger after its
// peer has finished and be killed by its own activeDeadlineSeconds; waiting for
// it would turn every successful pair into a deadline Error. So the pair settles
// when the CLIENT reports: the server's result is folded in if it also
// terminated, and otherwise the server pod is stopped and recorded as not having
// reported.
func (r *BurnInRunReconciler) harvestPair(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	p *plan,
	t plannedTest,
	nodes []string,
	attempt int32,
	serverPod, clientPod *corev1.Pod,
) (advanceState, advanceEffect, error) {
	var none advanceEffect
	serverNode, clientNode := nodes[0], nodes[1]

	_, clientDone, _, _ := podOutcome(clientPod)
	if !clientDone {
		if r.podOverdue(clientPod, &t.Spec) {
			if err := r.deletePairPods(ctx, serverPod, clientPod); err != nil {
				return advancePending, none, err
			}
			r.completeAttempt(ctx, run, p, t, nodes, attempt, serverPod.Name, serverPod, runner.Result{
				Verdict: runner.VerdictError,
				Message: fmt.Sprintf(
					"the %s endpoint on %s never completed within its window (last phase %q) — the link was not measured; no verdict",
					pairRoleClient, clientNode, clientPod.Status.Phase),
			})
			return advanceHarvested, advanceEffect{dirty: true}, nil
		}
		_, opened := r.beginAttempt(run, t, nodes, attempt, serverPod)
		// Checkpoint from the client, because that is where the numbers are.
		// resultFor matches any node in the result's Nodes, so this refreshes
		// the one shared pair result.
		checkpointed := r.checkpoint(ctx, run, t, clientNode, clientPod)
		return advanceRunning, advanceEffect{dirty: opened || checkpointed, checkpointed: checkpointed}, nil
	}

	// The client has reported. Open the result if the server was never observed
	// running — the execution plainly happened.
	r.beginAttempt(run, t, nodes, attempt, serverPod)

	clientParsed, clientLogErr := r.harvestPod(ctx, t, clientPod)
	client := pairSide{role: pairRoleClient, node: clientNode, pod: clientPod.Name, result: &clientParsed}
	server := pairSide{role: pairRoleServer, node: serverNode, pod: serverPod.Name}

	logErr := clientLogErr
	if _, serverDone, _, _ := podOutcome(serverPod); serverDone {
		serverParsed, serverLogErr := r.harvestPod(ctx, t, serverPod)
		server.result = &serverParsed
		if logErr == nil {
			logErr = serverLogErr
		}
	} else if err := r.Delete(ctx, serverPod); err != nil && !apierrors.IsNotFound(err) {
		return advancePending, none, err
	}

	combined := combinePair(server, client, logErr, &t.Spec)
	r.completeAttempt(ctx, run, p, t, nodes, attempt, serverPod.Name, serverPod, combined)
	// BOTH ends. A pair is one verdict about a link, but the two pods are two
	// separate stdouts, and evidence from the server end is exactly what
	// explains a client-side "connection refused".
	r.harvestArtifacts(ctx, run, t.Name, clientPod.Name, clientParsed.Artifacts)
	if server.result != nil {
		r.harvestArtifacts(ctx, run, t.Name, serverPod.Name, server.result.Artifacts)
	}
	return advanceHarvested, advanceEffect{dirty: true}, nil
}

func (r *BurnInRunReconciler) deletePairPods(ctx context.Context, pods ...*corev1.Pod) error {
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

// pairPod fetches one endpoint's pod. A missing pod is (nil, nil): not existing
// yet is the normal state of half a rendezvous, not an error.
func (r *BurnInRunReconciler) pairPod(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	index int,
	attempt int32,
	node, role string,
) (*corev1.Pod, error) {
	var pod corev1.Pod
	name := podNameForRole(run, index, node, attempt, role)
	err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, &pod)
	switch {
	case apierrors.IsNotFound(err):
		return nil, nil
	case err != nil:
		return nil, err
	}
	return &pod, nil
}

// combinePair folds the two endpoints' reports into the single result the pair
// is judged on. See the package comment above for the precedence rule and why
// Error outranks Fail.
func combinePair(server, client pairSide, logErr error, spec *burninv1alpha1.BurnInTestSpec) runner.Result {
	out := runner.Result{Metrics: map[string]string{}, Unmeasurable: map[string]bool{}}
	var invalid []string

	// Server first, client second, so the client wins any key both report.
	for _, s := range []pairSide{server, client} {
		if s.result == nil {
			continue
		}
		for k, v := range s.result.Metrics {
			out.Metrics[k] = v
		}
		invalid = append(invalid, s.result.InvalidNames...)
	}
	// A declaration of unmeasurability never overwrites a real measurement: if
	// one end measured the quantity, it was measurable on this link.
	for _, s := range []pairSide{server, client} {
		if s.result == nil {
			continue
		}
		for k := range s.result.Unmeasurable {
			if _, measured := out.Metrics[k]; !measured {
				out.Unmeasurable[k] = true
			}
		}
	}

	// Deduped for the same reason the Group merge is: pkg/runner appends one
	// entry per OFFENDING LINE and progressive reporting is expected there, so a
	// runner emitting one bad key per sample gives one entry per sample, twice
	// over at Pair scope. attemptOutcome joins the list into a message stored
	// once per attempt per retry. The SET is the finding; how many times it was
	// printed is not.
	out.InvalidNames = dedupeStrings(invalid)

	verdict, exit := pairVerdict(server, client)
	out.Verdict = verdict
	out.ExitCode = exit
	out.Message = pairMessage(server, client)

	// The log-fetch rule, applied ONCE to the combined result rather than per
	// side. A fetch failure only invalidates the verdict if it actually cost us
	// a number a threshold needs: when the client's log came through with every
	// thresholded metric in it, a missing server log changes nothing, and
	// erroring on it would throw away a good measurement.
	if logErr != nil && out.Verdict == runner.VerdictPass && len(spec.Thresholds) > 0 && missingThresholdMetric(out, spec.Thresholds) {
		out.Verdict = runner.VerdictError
		out.Message = fmt.Sprintf("runner logs unavailable (%v) — thresholds cannot be evaluated; no verdict [%s]", logErr, out.Message)
	}
	return out
}

// pairVerdict applies the precedence rule and returns the deciding side's exit
// code alongside it.
func pairVerdict(server, client pairSide) (runner.Verdict, int) {
	switch {
	case server.result == nil && client.result == nil:
		// Neither end reported. Reachable only through a build failure, which
		// supplies its own result; kept exhaustive rather than assumed.
		return runner.VerdictError, -1

	case client.result == nil:
		// The server finished before the client was ever started.
		switch server.result.Verdict {
		case runner.VerdictPass:
			// A clean exit that measured nothing. This must NOT become a pass:
			// no traffic crossed the link, so there is no evidence about it.
			return runner.VerdictError, server.result.ExitCode
		default:
			return server.result.Verdict, server.result.ExitCode
		}

	case server.result == nil:
		// The client decided while the server was still running; the client is
		// the measuring side, so its verdict stands on its own.
		return client.result.Verdict, client.result.ExitCode
	}

	// Both reported: Error > Fail > Skip > Pass.
	for _, v := range []runner.Verdict{runner.VerdictError, runner.VerdictFail, runner.VerdictSkip} {
		if client.result.Verdict == v {
			return v, client.result.ExitCode
		}
		if server.result.Verdict == v {
			return v, server.result.ExitCode
		}
	}
	return runner.VerdictPass, client.result.ExitCode
}

// pairMessage states the verdict's subject before it states the evidence.
//
// The lead clause is not decoration. A reader who sees a failure next to two
// node names will pick one of them, and picking one is the mistake this whole
// design exists to prevent — so the message says out loud that the link is what
// was judged, and then reports each end's own words without ranking them.
func pairMessage(server, client pairSide) string {
	return fmt.Sprintf("pair %s <-> %s (this verdict is about the LINK, not about either endpoint): %s; %s",
		server.node, client.node, server.summary(), client.summary())
}

// missingThresholdMetric reports whether any thresholded metric is absent from
// the result entirely. A metric the runner explicitly declared unmeasurable is
// NOT missing: that is a positive statement about the hardware, and it is
// pkg/verdict's job to decide what it means, not this function's.
func missingThresholdMetric(res runner.Result, thresholds []burninv1alpha1.Threshold) bool {
	for _, th := range thresholds {
		if _, ok := res.Metrics[th.Metric]; ok {
			continue
		}
		if res.Unmeasurable[th.Metric] {
			continue
		}
		return true
	}
	return false
}

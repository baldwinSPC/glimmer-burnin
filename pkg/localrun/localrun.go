// Package localrun runs a burn-in profile on a machine that is not a cluster
// member.
//
// # Why this is a package and not a command
//
// GEP-0178 is a dual-path design: burn-in at enrollment time on a bare host, and
// the comprehensive suite in-cluster. It matters most for exactly the hardware
// that needs it, because a node whose defect prevents it from joining a cluster
// cannot be tested by an operator that requires it to have joined one.
//
// The obvious way to build the bare-metal path is a second implementation. That
// would reproduce, deliberately, the defect this solution already has on record:
// a second dispatcher running a DIFFERENT test under the same TestKind name,
// with no way to reconcile the two verdicts. So the sequencing lives here, in a
// public package, and every dispatcher imports it rather than restating it.
//
// # What was already free
//
// Almost everything below the sequencing. The runners are standard-library Go
// and CUDA with no Kubernetes assumptions: a peer address can be a plain IP, a
// Pair server learns its peer from the connection rather than from DNS, and
// RLIMIT_MEMLOCK is easier to set under `docker run` than under a PodSpec.
// pkg/runner parses, pkg/verdict judges, pkg/contract emits, pkg/runnerimages
// says which image implements a kind. This package is the part that decides what
// runs, in what order, with what retries, and what the verdict is.
//
// # The rule that governs the whole file
//
// The decision table below MUST match internal/controller's completeAttempt and
// attemptOutcome. Not approximately — exactly, because the promise of the design
// is that two dispatchers reach the same verdict from the same evidence, and a
// divergence would show up as a node that passes on one path and fails on the
// other. conformance_test.go encodes the table as data and names the controller
// code each row mirrors. Extracting a shared state machine both consume is the
// right long-term answer and is deliberately deferred; the table is the cheap
// thing that catches the drift now.
package localrun

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/plan"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// Plan is what to run. It is resolved by the caller from profile and test
// definitions, so this package needs no opinion about where those came from.
type Plan struct {
	// Node is the machine's name, used to attribute results.
	Node string
	// Vendor is this host's accelerator vendor — nvidia, amd, intel,
	// tenstorrent — as pkg/hostinfo derives it from PCI IDs. Empty where nothing
	// established one, which is not a mismatch: absence is not a declaration, and
	// an unknown vendor resolves exactly as it did before the field existed.
	//
	// It is here because IMAGE SELECTION DEPENDS ON IT. The operator reads the
	// same fact from a NodeFingerprint; this path reads it from the PCI bus. Two
	// sources for one question, and the same ladder consuming them, so the two
	// dispatchers cannot resolve different images for the same test on the same
	// hardware.
	Vendor string
	// HostIP is this machine's primary address, as pkg/hostinfo establishes it
	// from the routing table. Empty where none could be found.
	//
	// It is here for the same reason Vendor is: a BurnInTest may ask for it,
	// and the two dispatchers must answer from equally authoritative sources.
	// In a cluster `valueFrom.fieldRef: status.hostIP` is filled by the kubelet;
	// here it is filled from this field, which the caller probes. Resolved by
	// the CALLER rather than by Translate so that translation stays a pure
	// function of the plan — the same reason Node and Vendor are not probed
	// here either.
	HostIP string
	// Tests run in order. A profile is an ordered list, and the order is part of
	// what it means: a smoke test that gates a soak has to run first.
	Tests []PlannedTest
	// FailFast stops the run at the first required test that does not pass.
	FailFast bool
	// RetryOnErrorLimit is how many times an Error may be retried, per test.
	// Zero means no retries, which is the operator's default too.
	RetryOnErrorLimit int32
	// Rendezvous is set when this host is ONE END of a multi-host test. Nil for
	// an ordinary single-machine run, and a Pair-scope test then runs with
	// BURNIN_ROLE unset — which the runners already treat as "not applicable"
	// and skip cleanly, rather than half-running a link test against nobody.
	Rendezvous *Rendezvous
}

// Rendezvous is where the other end of a multi-host test is.
//
// In a cluster the operator supplies a Service DNS name because it has one; the
// runners never needed it. Their control channel is their own: a server takes
// conn.RemoteAddr() and never resolves anything, and a client dials whatever
// address it is handed. So a bare-metal pair is the same protocol with a plain
// IP substituted for the DNS name, which is why this is a few fields and not a
// second rendezvous design.
type Rendezvous struct {
	// Role is "server" or "client". The CLIENT IS THE DECIDING SIDE, here as at
	// Pair scope in the operator: it is where perftest and nccl-tests report.
	Role string
	// PeerHost is the address the client dials. Empty for a server, which
	// learns its peer when the peer connects.
	PeerHost string
	// PeerNode is the peer's name FOR MESSAGES ONLY. Never for addressing —
	// a node name is not a route, and treating it as one turns a naming
	// mismatch into what reads as a fabric fault.
	PeerNode string

	// The Group variables. Plumbed through the same path even though multi-host
	// Group orchestration is not wired up yet: the cost is these few lines, and
	// it keeps ONE env contract across both dispatchers instead of two that
	// drift until someone notices.
	Rank     *int32
	NRanks   int32
	RootHost string
	RootNode string
}

// Roles.
const (
	RoleServer = "server"
	RoleClient = "client"
)

// pairNodes returns the nodes a link result names, in target order.
//
// Target order, not local-first: the operator emits [server, client] and a
// result that reorders them by which host happened to render it would not
// compare with the same link measured in a cluster.
func (r *Rendezvous) pairNodes(local string) []string {
	if r == nil || r.PeerNode == "" {
		return []string{local}
	}
	if r.Role == RoleClient {
		return []string{r.PeerNode, local}
	}
	return []string{local, r.PeerNode}
}

// groupNodes returns the nodes ONE RANK can honestly name.
//
// Just itself — and that is the whole reason `burnin merge` exists.
//
// A Group result names EVERY rank's node, because the verdict is about the
// collective. But no single rank knows the others' node names: the rendezvous
// contract carries BURNIN_ROOT_NODE and DELIBERATELY NOT A RANK LIST, so a rank
// that invented the roster would be inventing hardware it never heard from.
// The operator knows the roster because it planned the run; here the roster is
// only assembled when the per-rank records are merged.
//
// So a rank's own record names one node, truthfully, and the merged result
// names all N. A rank that padded its list with the root's name would produce
// a record claiming to be about two machines when it is about one.
func (r *Rendezvous) groupNodes(local string) []string {
	return []string{local}
}

// PlannedTest is one test, with its spec pinned.
//
// An ALIAS for plan.Test, which is the same type the operator materialises its
// own plan into. It is the same type on purpose: variant expansion produces
// these, and expansion is shared (see pkg/plan) precisely because this
// dispatcher used to drop variants on the floor. A parallel struct here would
// have made "the two dispatchers plan the same executions" a claim rather than
// a fact.
//
// Required tests participate in FailFast and in the run's own verdict; an
// optional test that fails is recorded and does not condemn the run. Axes and
// Parent are the variant labels the cell came from, carried and echoed and
// never interpreted.
type PlannedTest = plan.Test

// Hooks let an embedding agent observe a run as it happens.
//
// OnTestComplete exists because the control plane's dispatch has a hard
// execution-time ceiling and needs to publish results as each test finishes
// rather than accumulating them until the end. Every hook is optional.
type Hooks struct {
	// OnTestStart fires before the first attempt of a test.
	OnTestStart func(test string)
	// OnAttempt fires after each attempt, before the decision is made. Useful
	// for progress; not a decision point.
	OnAttempt func(test string, attempt int32, res runner.Result)
	// OnTestComplete fires once per test, when it settles.
	OnTestComplete func(TestResult)
	// OnCheckpoint fires while a test is still running, every
	// spec.checkpointIntervalSeconds, with the metrics the runner has emitted
	// so far.
	//
	// A CHECKPOINT IS EVIDENCE, NEVER A VERDICT. Nothing in this package reads
	// what it publishes; thresholds are evaluated once, at the end, against the
	// completed execution. A mid-run sample that dips below a floor is not a
	// failure, because the run is not over — and a consumer that treated one as
	// a verdict would condemn hardware for a moment it was told to expect.
	OnCheckpoint func(Checkpoint)
}

// Checkpoint is a test's progress, published while it is still running.
//
// It exists for the case a long soak is most likely to meet: a multi-hour
// thermal soak that is cancelled, killed at its deadline, or lost to a reboot
// at minute 200 otherwise reports NOTHING AT ALL, because a runner's metrics
// only reach the report when the container exits. The operator has published
// these since checkpointIntervalSeconds was added; this dispatcher buffered
// stdout until exit and so could not, which is the drift the parity ledger
// pins.
type Checkpoint struct {
	// Test is the execution's name — a variant cell's own name, where the entry
	// was expanded.
	Test string
	// Sequence orders checkpoints within one run. Derived from elapsed time
	// rather than counted, exactly as the operator derives it, so it is stable
	// across a retry and a receiver's dedupe still works.
	Sequence int
	// Attempt is which execution of this test is being sampled.
	Attempt int32
	// Metrics is what the runner has emitted so far, parsed by the same
	// pkg/runner both dispatchers use.
	Metrics map[string]string
	At      time.Time
}

func (h Hooks) checkpoint(c Checkpoint) {
	if h.OnCheckpoint != nil {
		h.OnCheckpoint(c)
	}
}

func (h Hooks) testStart(name string) {
	if h.OnTestStart != nil {
		h.OnTestStart(name)
	}
}

func (h Hooks) attempt(name string, n int32, res runner.Result) {
	if h.OnAttempt != nil {
		h.OnAttempt(name, n, res)
	}
}

func (h Hooks) testComplete(r TestResult) {
	if h.OnTestComplete != nil {
		h.OnTestComplete(r)
	}
}

// Report is the run's outcome.
type Report struct {
	Node       string
	Phase      api.RunPhase
	StartedAt  time.Time
	FinishedAt time.Time
	Results    []TestResult
	// Summary counts executions, matching the envelope's units.
	Summary Summary
}

// Summary counts results by phase.
type Summary struct{ Passed, Failed, Errored, Skipped int32 }

// TestResult is one test's outcome, shaped to match what the operator records so
// a caller can build an identical envelope from either dispatcher.
type TestResult struct {
	Name string
	Kind string
	// Scope travels with the result because a consumer needs it to read the
	// verdict at all: a Pair result is a statement about the LINK between its
	// two nodes, and without it that is indistinguishable from two single-node
	// claims. The operator has always delivered it; this dispatcher EXECUTED
	// Pair scope while delivering Scope:"" — the same-contract-two-assemblers
	// drift the parity guard in cmd/burnin now pins (found while writing it).
	Scope      api.TestScope
	Phase      api.RunPhase
	Nodes      []string
	StartedAt  time.Time
	FinishedAt time.Time
	Metrics    map[string]string
	Message    string

	// VariantAxes are the labels of the variant cell this execution came from,
	// or nil for a test with no variants. Delivered in the envelope so a report
	// can group a sweep's cells without parsing them back out of names — see
	// contract.TestResult.VariantAxes, which is the field this feeds.
	VariantAxes map[string]string

	Violations   []api.Violation
	NotEvaluated []api.NotEvaluated
	Applied      []api.AppliedGate
	Unmeasurable []string

	RepeatsRequired  int32
	RepeatsCompleted int32
	ErrorRetries     int32
	Attempts         []Attempt
}

// scopeOf mirrors the CRD's defaulting: an unset scope means Node.
func scopeOf(spec api.BurnInTestSpec) api.TestScope {
	if spec.Scope == "" {
		return api.ScopeNode
	}
	return spec.Scope
}

// Attempt is one execution.
type Attempt struct {
	Attempt    int32
	Trigger    api.AttemptTrigger
	Phase      api.RunPhase
	ExitCode   int
	StartedAt  time.Time
	FinishedAt time.Time
	Metrics    map[string]string
	Message    string
}

// Clock is injectable so tests are not timing-dependent.
type Clock func() time.Time

// Run executes a plan.
//
// It returns an error only when the run could not proceed at all — a runtime
// that cannot be reached, a context that was cancelled. A test that fails, errors
// or is skipped is a RESULT, not an error: collapsing those into Go errors would
// be the same category mistake as collapsing Error into Fail.
func Run(ctx context.Context, p Plan, rt ContainerRuntime, h Hooks) (Report, error) {
	return runWithClock(ctx, p, rt, h, time.Now)
}

func runWithClock(ctx context.Context, p Plan, rt ContainerRuntime, h Hooks, now Clock) (Report, error) {
	rep := Report{Node: p.Node, StartedAt: now()}

	for _, t := range p.Tests {
		if err := ctx.Err(); err != nil {
			rep.FinishedAt = now()
			rep.Phase = finalPhase(rep.Results)
			return rep, err
		}

		h.testStart(t.Name)
		res, err := runTest(ctx, p, t, rt, h, now, rep.StartedAt)
		if err != nil {
			return rep, err
		}
		rep.Results = append(rep.Results, res)
		h.testComplete(res)

		if p.FailFast && t.Required && res.Phase != api.RunPassed {
			// The remaining tests are not run and are not recorded. A test that
			// did not execute has no verdict, and inventing a phase for it —
			// even Skipped — would claim something about hardware nobody looked
			// at.
			break
		}
	}

	rep.FinishedAt = now()
	rep.Summary = tally(rep.Results)
	rep.Phase = finalPhase(rep.Results)
	return rep, nil
}

// runTest executes one test to a verdict, applying the repeat and retry rules.
//
// This is the loop that mirrors completeAttempt. Every branch below names the
// controller behaviour it reproduces.
func runTest(ctx context.Context, p Plan, t PlannedTest, rt ContainerRuntime, h Hooks, now Clock, runStarted time.Time) (TestResult, error) {
	res := TestResult{
		Name: t.Name,
		Kind: string(t.Spec.Kind),
		// Defaulted here because no apiserver ever saw this spec: the CRD
		// defaults an unset scope to Node, and a dispatcher that reads the
		// YAML verbatim must apply the same default or the two disagree about
		// the same document.
		Scope: scopeOf(t.Spec),
		// COPIED, not shared: the result outlives the plan and a consumer must
		// not be able to reach back into it. Echoed and never interpreted —
		// pkg/contract tells a consumer to group a sweep's cells BY these, and
		// a result that carried the name but not the labels would make a
		// four-cell precision sweep four results nothing can group.
		VariantAxes:     plan.CopyAxes(t.Axes),
		Nodes:           NodesFor(p, t),
		StartedAt:       now(),
		RepeatsRequired: repeatCount(t.Spec),
	}

	for attempt := int32(1); ; attempt++ {
		if err := ctx.Err(); err != nil {
			res.Phase = api.RunError
			res.Message = "run cancelled before this test reached a verdict"
			res.FinishedAt = now()
			return res, err
		}

		spec, err := Translate(p, t)
		if err != nil {
			// A test that cannot be turned into a container invocation is a
			// configuration Error, not a hardware verdict — the same phase the
			// operator gives an unresolvable image.
			res.Phase = api.RunError
			res.Message = err.Error()
			res.FinishedAt = now()
			return res, nil
		}
		spec.Env["BURNIN_ATTEMPT"] = fmt.Sprint(attempt)

		started := now()
		exec, err := runOnce(ctx, rt, spec, t, attempt, started, runStarted, p, h, now)
		if err != nil {
			// The runtime itself failed. Recorded as an Error attempt so the
			// retry budget applies, rather than aborting the whole run: a
			// transient docker hiccup is exactly what retries are for.
			exec = Execution{ExitCode: -1, Stderr: err.Error()}
		}

		parsed := runner.Parse(string(t.Spec.Kind), exec.Stdout, exec.ExitCode)
		h.attempt(t.Name, attempt, parsed)

		phase, message, ev := outcome(t, parsed)
		if exec.Stderr != "" && phase == api.RunError && parsed.Message == "" {
			// Nothing on stdout explained an Error. The runtime's own words are
			// the only account of what happened, and dropping them leaves a
			// verdict nobody can act on.
			message = strings.TrimSpace(message + " [" + firstLine(exec.Stderr) + "]")
		}

		res.Attempts = append(res.Attempts, Attempt{
			Attempt:    attempt,
			Trigger:    triggerFor(res, attempt),
			Phase:      phase,
			ExitCode:   exec.ExitCode,
			StartedAt:  started,
			FinishedAt: now(),
			Metrics:    parsed.Metrics,
			Message:    message,
		})
		if len(parsed.Metrics) > 0 {
			res.Metrics = parsed.Metrics
		}

		settle := func(final api.RunPhase) TestResult {
			res.Phase = final
			res.FinishedAt = now()
			res.Message = message
			res.Metrics = parsed.Metrics
			// Assigned unconditionally and from the SAME attempt as Metrics, so
			// a retry that passes cannot keep its predecessor's evidence.
			res.Violations = ev.violations
			res.NotEvaluated = ev.notEvaluated
			res.Applied = ev.applied
			res.Unmeasurable = ev.unmeasurable
			return res
		}

		switch phase {
		case api.RunPassed:
			res.RepeatsCompleted++
			if res.RepeatsCompleted >= res.RepeatsRequired {
				return settle(api.RunPassed), nil
			}
			// More repeats owed. Sequential, because two concurrent copies on
			// one machine would contend for the very resource under test.
			res.Message = fmt.Sprintf("attempt %d/%d passed", res.RepeatsCompleted, res.RepeatsRequired)

		case api.RunFailed:
			// Settles immediately with the retry budget UNSPENT. Re-running a
			// measurement until it comes out clean launders a hardware fault
			// into an acceptance.
			return settle(api.RunFailed), nil

		case api.RunSkipped:
			// The test does not apply to this hardware. Repeating will not make
			// it start applying, and retrying will not either.
			return settle(api.RunSkipped), nil

		default:
			if res.ErrorRetries < p.RetryOnErrorLimit {
				res.ErrorRetries++
				// An errored attempt measured nothing, so it does not consume a
				// repeat: RepeatsCompleted is deliberately untouched.
				res.Message = fmt.Sprintf("attempt %d errored (%s); retrying", attempt, message)
				continue
			}
			return settle(api.RunError), nil
		}
	}
}

// runOnce executes one attempt, publishing checkpoints along the way when the
// test asked for them and the runtime can stream.
//
// The timing decision lives HERE and not in the runtime, because the interval
// is a property of the test — which is in the plan, which the engine owns. A
// runtime that decided when to sample would need to be told the plan, and every
// implementation of the interface would have to reproduce the rule.
//
// A checkpoint publishes nothing back into the attempt: the returned Execution
// is exactly what the runtime produced. That is the "evidence, never a verdict"
// rule expressed as a signature — there is no path from here into the phase.
func runOnce(
	ctx context.Context,
	rt ContainerRuntime,
	spec RunSpec,
	t PlannedTest,
	attempt int32,
	started time.Time,
	runStarted time.Time,
	p Plan,
	h Hooks,
	now Clock,
) (Execution, error) {
	interval := checkpointInterval(t.Spec)
	stream, canStream := rt.(StreamingRuntime)
	if interval <= 0 || !canStream || h.OnCheckpoint == nil {
		// Nothing to publish, nobody to publish to, or a runtime that cannot
		// stream. The last case degrades the EVIDENCE and never the verdict,
		// which is why it is silent rather than an error.
		return rt.Run(ctx, spec)
	}

	// last is read and written only from the runtime's write goroutine, which
	// calls onOutput serially — liveBuffer notifies outside its lock but never
	// concurrently with itself.
	last := started
	return stream.RunStreaming(ctx, spec, func(stdoutSoFar string) {
		at := now()
		if at.Sub(last) < interval {
			return
		}

		// The exit code is deliberately not consulted: the container has not
		// exited, and only the parsed metrics are wanted. Passing 0 selects no
		// behaviour — the phase logic is not on this path at all, which is the
		// same choice the operator's own checkpoint makes.
		parsed := runner.Parse(string(t.Spec.Kind), stdoutSoFar, 0)
		if len(parsed.Metrics) == 0 {
			// Nothing to publish yet. `last` is NOT advanced, so the next write
			// tries again rather than waiting out another whole interval for a
			// runner that is simply slow to emit its first line.
			return
		}
		last = at

		h.checkpoint(Checkpoint{
			Test:     t.Name,
			Sequence: checkpointSequence(p, runStarted, at),
			Attempt:  attempt,
			Metrics:  parsed.Metrics,
			At:       at,
		})
	})
}

// checkpointInterval is how often one test publishes its in-progress metrics.
// Zero or nil disables it, matching the operator's own reading of the field.
func checkpointInterval(spec api.BurnInTestSpec) time.Duration {
	if spec.CheckpointIntervalSeconds == nil || *spec.CheckpointIntervalSeconds <= 0 {
		return 0
	}
	return time.Duration(*spec.CheckpointIntervalSeconds) * time.Second
}

// checkpointSequence numbers a run's checkpoints.
//
// DERIVED from elapsed time rather than counted, which is the operator's own
// choice and for its reason: the sequence feeds the envelope's DeliveryID, and
// a key that moved on every attempt would mint a new identity per retry and
// defeat the receiver's dedupe — turning a flaky endpoint into a flood of
// near-identical records.
//
// The interval used is the RUN's — the shortest positive one any test asked
// for — so two tests with different cadences still produce one monotonic
// sequence for the run, which is what a consumer orders by.
func checkpointSequence(p Plan, started, at time.Time) int {
	interval := runCheckpointInterval(p)
	if interval <= 0 {
		return 0
	}
	elapsed := at.Sub(started)
	if elapsed <= 0 {
		return 0
	}
	return int(elapsed / interval)
}

// runCheckpointInterval is the shortest positive interval any test declares.
func runCheckpointInterval(p Plan) time.Duration {
	var shortest time.Duration
	for i := range p.Tests {
		if d := checkpointInterval(p.Tests[i].Spec); d > 0 && (shortest == 0 || d < shortest) {
			shortest = d
		}
	}
	return shortest
}

// evidence is what an attempt established beyond its phase.
type evidence struct {
	violations   []api.Violation
	notEvaluated []api.NotEvaluated
	applied      []api.AppliedGate
	unmeasurable []string
}

// outcome decides one attempt's phase, mirroring attemptOutcome.
func outcome(t PlannedTest, parsed runner.Result) (api.RunPhase, string, evidence) {
	var phase api.RunPhase
	var ev evidence
	message := parsed.Message

	// The runner's own declarations survive whatever the thresholds decide.
	// "This part cannot produce this measurement" is a claim about the HARDWARE,
	// true of a passing run as much as a failing one.
	for name := range parsed.Unmeasurable {
		ev.unmeasurable = append(ev.unmeasurable, name)
	}
	sort.Strings(ev.unmeasurable)

	switch parsed.Verdict {
	case runner.VerdictPass:
		switch {
		case len(t.Spec.Thresholds) > 0 && parsed.ReportedNothing():
			// Exit 0, not one key=value line, and an acceptance bar to clear.
			// The runner did not measure this hardware, so there is nothing for
			// the thresholds to judge, and handing an empty map to fail-closed
			// evaluation manufactures a hardware verdict out of an absence.
			//
			// THIS DOES NOT RELAX FAIL-CLOSED. A runner that reported SOME
			// metrics and omitted a gated one still Fails below: it looked at
			// the hardware, and a measurement it owed and did not produce is
			// exactly the silence that must never satisfy acceptance. Only the
			// TOTAL absence of output qualifies.
			phase = api.RunError
			message = fmt.Sprintf(
				"runner exited 0 but reported nothing, with %d threshold(s) to evaluate — "+
					"the hardware was not measured and is UNJUDGED", len(t.Spec.Thresholds))
			if parsed.Message != "" {
				message = strings.TrimSpace(message + " [runner said: " + parsed.Message + "]")
			}
		default:
			out := verdict.Evaluate(parsed.Metrics, parsed.Unmeasurable, t.Spec.Thresholds)
			if out.Passed {
				phase = api.RunPassed
			} else {
				phase = api.RunFailed
				// The runner's own message is KEPT, not replaced.
				message = out.Message
				if prior := strings.TrimSpace(parsed.Message); prior != "" {
					message = strings.TrimSpace(message + " [" + prior + "]")
				}
				ev.violations = violationsFor(out)
				if more := out.ViolationSummary(); more != "" {
					message = strings.TrimSpace(message + " [" + more + "]")
				}
			}
			for _, n := range out.NotEvaluated {
				ev.notEvaluated = append(ev.notEvaluated, api.NotEvaluated{Metric: n.Metric, Reason: n.Reason})
			}
			if why := out.NotEvaluatedMessage(); why != "" {
				message = strings.TrimSpace(message + " [" + why + "]")
			}
			// Unconditional like NotEvaluated above and unlike violations: a
			// Failed attempt still cleared some OTHER gates (#262).
			ev.applied = appliedFor(out)
		}

	case runner.VerdictFail:
		// Exit 1 is the runner's OWN assertion that this hardware failed, and it
		// is honoured whether or not any metric came with it. A runner that
		// detects a fault and dies before printing has still detected a fault.
		phase = api.RunFailed

	case runner.VerdictSkip:
		// Zero metrics is the NORMAL shape of a skip, and no threshold is
		// evaluated on this path, so the empty-harvest rule must not apply here.
		phase = api.RunSkipped

	default:
		phase = api.RunError
		if parsed.UndeclaredSkip {
			message = strings.TrimSpace(
				"runner exited 2 (the skip code) without declaring a skip — an unrecovered Go panic " +
					"exits 2, as does every Go runtime fatal error, so the likeliest cause is a crashed " +
					"runner: the hardware is UNJUDGED, not out of scope" + bracket(message))
		}
	}

	if len(parsed.InvalidNames) > 0 {
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

// maxNamedInvalidMetrics bounds the list in a message. The count stays exact.
const maxNamedInvalidMetrics = 5

func bracket(s string) string {
	if strings.TrimSpace(s) == "" {
		return ""
	}
	return " [runner said: " + strings.TrimSpace(s) + "]"
}

func firstLine(s string) string {
	line, _, _ := strings.Cut(strings.TrimSpace(s), "\n")
	return strings.TrimSpace(line)
}

func violationsFor(out verdict.Outcome) []api.Violation {
	if len(out.Violations) == 0 {
		return nil
	}
	vs := make([]api.Violation, 0, len(out.Violations))
	for _, v := range out.Violations {
		vs = append(vs, api.Violation{
			Index:  int32(v.Index),
			Metric: v.Metric,
			Cause:  string(v.Cause),
			Kind:   string(v.Kind),
			Reason: v.Reason,
		})
	}
	return vs
}

func appliedFor(out verdict.Outcome) []api.AppliedGate {
	if len(out.Applied) == 0 {
		return nil
	}
	ag := make([]api.AppliedGate, 0, len(out.Applied))
	for _, a := range out.Applied {
		ag = append(ag, api.AppliedGate{
			Index:      int32(a.Index),
			Metric:     a.Metric,
			Comparison: string(a.Comparison),
			Value:      a.Value,
		})
	}
	return ag
}

// triggerFor records why an attempt happened, so the rule is auditable from a
// stored result long after the run.
func triggerFor(res TestResult, attempt int32) api.AttemptTrigger {
	if attempt == 1 {
		return api.AttemptInitial
	}
	if len(res.Attempts) > 0 && res.Attempts[len(res.Attempts)-1].Phase == api.RunError {
		return api.AttemptErrorRetry
	}
	return api.AttemptRepeat
}

func repeatCount(spec api.BurnInTestSpec) int32 {
	if spec.RepeatCount != nil && *spec.RepeatCount > 0 {
		return *spec.RepeatCount
	}
	return 1
}

func tally(results []TestResult) Summary {
	var s Summary
	for _, r := range results {
		switch r.Phase {
		case api.RunPassed:
			s.Passed++
		case api.RunFailed:
			s.Failed++
		case api.RunSkipped:
			s.Skipped++
		default:
			s.Errored++
		}
	}
	return s
}

// finalPhase decides the run's verdict.
//
// Failed outranks Error, which outranks everything else — the same precedence
// the controller's finalize uses. Skips count toward neither: a run whose tests
// all skipped judged nothing, and calling that Passed would certify hardware the
// suite never measured.
func finalPhase(results []TestResult) api.RunPhase {
	var failed, errored, passed bool
	for _, r := range results {
		switch r.Phase {
		case api.RunFailed:
			failed = true
		case api.RunError:
			errored = true
		case api.RunPassed:
			passed = true
		}
	}
	switch {
	case failed:
		return api.RunFailed
	case errored:
		return api.RunError
	case passed:
		return api.RunPassed
	default:
		return api.RunSkipped
	}
}

// NodesFor attributes a result to the machines it is about.
//
// Exported because a dispatcher building a result outside this package must
// attribute it the same way; there is one rule and it lives here.
//
// A Pair verdict is about the LINK, so its result names BOTH nodes — the same
// single result the operator produces, never one per endpoint. Attributing a
// point-to-point measurement to one machine sends an engineer to replace the
// wrong part.
func NodesFor(p Plan, t PlannedTest) []string {
	if p.Rendezvous == nil {
		return []string{p.Node}
	}
	switch t.Spec.Scope {
	case api.ScopePair:
		return p.Rendezvous.pairNodes(p.Node)
	case api.ScopeGroup:
		return p.Rendezvous.groupNodes(p.Node)
	}
	return []string{p.Node}
}

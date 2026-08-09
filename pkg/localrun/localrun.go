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
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// Plan is what to run. It is resolved by the caller from profile and test
// definitions, so this package needs no opinion about where those came from.
type Plan struct {
	// Node is the machine's name, used to attribute results.
	Node string
	// Tests run in order. A profile is an ordered list, and the order is part of
	// what it means: a smoke test that gates a soak has to run first.
	Tests []PlannedTest
	// FailFast stops the run at the first required test that does not pass.
	FailFast bool
	// RetryOnErrorLimit is how many times an Error may be retried, per test.
	// Zero means no retries, which is the operator's default too.
	RetryOnErrorLimit int32
}

// PlannedTest is one test, with its spec pinned.
type PlannedTest struct {
	Name string
	Spec api.BurnInTestSpec
	// Required tests participate in FailFast and in the run's own verdict. An
	// optional test that fails is recorded and does not condemn the run.
	Required bool
}

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
	Name       string
	Kind       string
	Phase      api.RunPhase
	Nodes      []string
	StartedAt  time.Time
	FinishedAt time.Time
	Metrics    map[string]string
	Message    string

	Violations   []api.Violation
	NotEvaluated []api.NotEvaluated
	Unmeasurable []string

	RepeatsRequired  int32
	RepeatsCompleted int32
	ErrorRetries     int32
	Attempts         []Attempt
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
		res, err := runTest(ctx, p, t, rt, h, now)
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
func runTest(ctx context.Context, p Plan, t PlannedTest, rt ContainerRuntime, h Hooks, now Clock) (TestResult, error) {
	res := TestResult{
		Name:            t.Name,
		Kind:            string(t.Spec.Kind),
		Nodes:           []string{p.Node},
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
		exec, err := rt.Run(ctx, spec)
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

// evidence is what an attempt established beyond its phase.
type evidence struct {
	violations   []api.Violation
	notEvaluated []api.NotEvaluated
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

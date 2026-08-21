package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/sink"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
	"github.com/baldwinSPC/glimmer-burnin/pkg/plan"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// Exit codes.
//
// The shape mirrors the runner contract and preserves Error != Fail at the
// process boundary, so a CI job can tell "this hardware is bad" from "this run
// never happened" — which are the two things that must not be collapsed, and
// the two a single non-zero exit would collapse.
const (
	exitPass = 0
	// exitFail is a required test that FAILED. The hardware was measured and
	// fell short.
	exitFail = 1
	// exitNothingJudged is every test skipping. Not a pass: nothing was
	// measured, and a pipeline that treats it as one certifies hardware the
	// suite never looked at.
	exitNothingJudged = 2
	// exitError is machinery — a config error, an unreachable runtime, a
	// runner that never reported. Retryable; not a verdict about hardware.
	exitError = 3
)

const runUsage = `burnin run — execute a profile on this machine

USAGE
  burnin run -f suite.yaml [--profile NAME] [flags]

FLAGS
  -f, --file        suite YAML (repeatable). BurnInProfile and BurnInTest
                    documents, exactly as a cluster would take them
  --profile         which profile, when the suite declares more than one
  --node            this machine's name in the results (default: hostname)
  --results-dir     where to write envelopes and raw runner output
  --runtime         auto|docker|podman|nerdctl (default auto)
  --retry-on-error  how many times an Error may be retried, per test (default 0)
  --dry-run         resolve and print what would run, then stop
  --sink-url        POST the envelope to this URL when the run finishes
  --sink-token-file file holding the bearer token — a file, never a flag, so the
                    credential is not in every process listing on the box

VARIANTS

  A profile entry's variants: block expands here exactly as it does in a cluster —
  one execution per cell, each with the cell's own name, thresholds and
  BURNIN_VARIANT_<AXIS> variables. --dry-run lists the cells and their axes.

PROGRESS

  A test that sets checkpointIntervalSeconds publishes the metrics it has
  emitted so far, every interval, to the terminal and to --sink-url. A soak
  killed at minute 200 otherwise reports nothing at all.

  A checkpoint is EVIDENCE, never a verdict: thresholds are evaluated once, at
  the end, against the completed execution.

TWO MACHINES (Pair-scope tests)

  A link test needs both ends. Neither has to be in a cluster.

    hostA$ burnin run -f suite.yaml --role server --node spark-a --results-dir r/
    hostB$ burnin run -f suite.yaml --role client --peer 10.0.0.11 \
                      --peer-node spark-a --node spark-b --results-dir r/

  --role server     no --peer: the server learns the client's address when the
                    client connects
  --role client     --peer <ip|host> is the server's address
  --peer-node       the peer's name, for messages only — never for addressing

  Start ordering is yours. Start the server first if you can; a client that
  starts first retries into success rather than reporting a fabric fault. If the
  test declares a tcpSocket readinessProbe, the server end prints a line when
  its listener opens.

  THE CLIENT DECIDES. The client's results directory holds the one envelope for
  the link, naming both nodes; the server writes sidecar/server-record.json,
  which is a record of that end and not a second verdict.

N MACHINES (Group-scope tests)

  A collective needs N machines. There is no --role: Group scope has no roles,
  and BURNIN_ROLE is deliberately never set, so a runner cannot mistake rank 4
  for a client.

    spark-a$ burnin run -f suite.yaml --rank 0 --nranks 3 --node spark-a --results-dir r/
    spark-b$ burnin run -f suite.yaml --rank 1 --nranks 3 --root 10.0.0.11 \
                        --node spark-b --results-dir r/

  --rank i          this machine's 0-based rank
  --nranks n        how many ranks the collective has
  --root <ip|host>  rank 0's address; required on every rank but 0

  START RANK 0 FIRST, wait for its listener, then start the rest. There is no
  scheduler here to gate on readiness, so the ordering is yours.

  NO RANK WRITES THE VERDICT. Each writes ranks/rank-NN.json; burnin merge
  folds them, because "did every rank take part" is a fact no rank can see.
  A partial collective is refused, naming the ranks that never reported.

EXIT CODES
  0  every required test passed
  1  a required test FAILED — measured, and fell short
  2  nothing was judged (everything skipped)
  3  machinery: configuration, runtime, or a runner that never reported
`

type runFlags struct {
	files      multiFlag
	profile    string
	node       string
	resultsDir string
	runtime    string
	retries    int
	dryRun     bool
	pair       pairFlags
	group      groupFlags
	sink       sinkFlags
}

type multiFlag []string

func (m *multiFlag) String() string     { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error { *m = append(*m, v); return nil }

func runRun(args []string) error {
	var f runFlags
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprint(os.Stderr, runUsage) }

	fs.Var(&f.files, "f", "suite YAML (repeatable)")
	fs.Var(&f.files, "file", "suite YAML (repeatable)")
	fs.StringVar(&f.profile, "profile", "", "which profile to run")
	fs.StringVar(&f.node, "node", "", "this machine's name in the results")
	fs.StringVar(&f.resultsDir, "results-dir", "", "where to write envelopes and raw output")
	fs.StringVar(&f.runtime, "runtime", "auto", "auto|docker|podman|nerdctl")
	fs.IntVar(&f.retries, "retry-on-error", 0, "retries per test, Error only")
	fs.BoolVar(&f.dryRun, "dry-run", false, "print the resolved plan and stop")
	fs.StringVar(&f.pair.role, "role", "", "server|client, for a link test across two machines")
	fs.StringVar(&f.pair.peer, "peer", "", "the server's address (client only)")
	fs.StringVar(&f.pair.peerNode, "peer-node", "", "the peer's name, for messages only")
	fs.IntVar(&f.group.rank, "rank", 0, "this machine's 0-based rank, for a Group-scope collective")
	fs.IntVar(&f.group.nranks, "nranks", 0, "how many ranks the collective has")
	fs.StringVar(&f.group.root, "root", "", "rank 0's address, which every other rank dials")
	fs.StringVar(&f.sink.url, "sink-url", "", "POST the envelope here when the run finishes")
	fs.StringVar(&f.sink.tokenFile, "sink-token-file", "", "file holding the bearer token")
	fs.BoolVar(&f.sink.insecure, "sink-insecure-skip-tls-verify", false, "development only")
	fs.DurationVar(&f.sink.timeout, "sink-timeout", 30*time.Second, "per-attempt timeout")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if len(f.files) == 0 {
		fs.Usage()
		return exitWith(exitError, fmt.Errorf("no suite: pass -f suite.yaml"))
	}

	node := f.node
	if node == "" {
		h, err := os.Hostname()
		if err != nil {
			return exitWith(exitError, fmt.Errorf("no --node and the hostname could not be read: %w", err))
		}
		node = h
	}

	// --rank is only "given" if it appears; rank 0 is a legal value and its
	// zero value is indistinguishable from absence, which would make every
	// machine that forgot the flag believe it was the root.
	fs.Visit(func(fl *flag.Flag) {
		if fl.Name == "rank" {
			f.group.rankSet = true
		}
	})

	rz, err := f.rendezvous(node)
	if err != nil {
		return exitWith(exitError, err)
	}

	// The sink is built BEFORE anything runs. A missing token file or a
	// malformed URL is a mistake worth catching in the first second, not after
	// a multi-hour soak has produced the result it would have carried.
	sender, err := f.sink.deliverer()
	if err != nil {
		return exitWith(exitError, err)
	}

	s, err := loadSuite(f.files)
	if err != nil {
		return exitWith(exitError, err)
	}
	resolved, warnings, err := s.buildPlan(f.profile, node, int32(f.retries), rz)
	if err != nil {
		return exitWith(exitError, err)
	}

	// A rendezvous with nothing to rendezvous is refused, not ignored. Someone
	// who passed --role believes a link is about to be measured, and running a
	// profile of Node-scope tests instead would answer a question they did not
	// ask while looking exactly like it answered theirs.
	if rz != nil && !pairScoped(resolved) {
		flag := "--role"
		if f.group.isGroup() {
			flag = "--rank"
		}
		return exitWith(exitError, fmt.Errorf(
			"%s was given but no test in this profile is Pair- or Group-scope; nothing here measures a "+
				"link or a collective", flag))
	}
	for _, w := range warnings {
		warn("%s", w)
	}

	// Threshold linting before anything runs, so an unsatisfiable gate is
	// reported while its author is still here rather than as a verdict on
	// hardware hours later.
	lintPlan(resolved)

	if f.dryRun {
		printPlan(resolved)
		return nil
	}

	rt, err := resolveRuntime(f.runtime)
	if err != nil {
		return exitWith(exitError, err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "burnin: %d test(s) on %s via %s\n", len(resolved.Tests), node, rt.Name())
	if rz != nil && rz.Role == localrun.RoleServer {
		announceServerReady(ctx, os.Stderr, resolved)
	}

	// ONE identity for the whole run, minted before anything executes. Every
	// checkpoint and the final verdict carry it, so a consumer can join a
	// soak's progress to its outcome — which is the entire point of publishing
	// progress at all.
	id, err := NewRunIdentity(node)
	if err != nil {
		return exitWith(exitError, err)
	}

	hooks := Hooks(os.Stderr)
	hooks.OnCheckpoint = checkpointHook(ctx, os.Stderr, id, resolved, sender, rz)

	report, runErr := localrun.Run(ctx, resolved, rt, hooks)
	if runErr != nil && ctx.Err() == nil {
		return exitWith(exitError, runErr)
	}

	if f.resultsDir != "" {
		if err := writeResults(f.resultsDir, report, rz, resolved); err != nil {
			// The run happened and its verdict stands; failing to file it is
			// worth saying loudly and is not a reason to discard the verdict.
			warn("results were not written: %v", err)
		}
	}

	if sender != nil {
		if !deciding(rz) {
			// A server end has no verdict to deliver, and posting one would put
			// a second document about one link into a consumer's history.
			fmt.Fprintf(os.Stderr, "burnin: not delivering from the server end — the client sends the link's envelope\n")
		} else if env, err := id.Final(report); err != nil {
			warn("no envelope to deliver: %v", err)
		} else {
			deliver(ctx, os.Stderr, sender, env)
		}
	}

	printSummary(report)
	if !deciding(rz) {
		// Said every time, because a server-side exit code is the thing most
		// likely to be mistaken for the link's verdict by whatever script is
		// reading it.
		fmt.Printf("\nThis is the SERVER end. The client's result is the link's verdict;\n" +
			"what is written here is a record of this end, not a second opinion.\n")
	}
	if ctx.Err() != nil {
		return exitWith(exitError, fmt.Errorf("interrupted before every test reached a verdict"))
	}
	return exitWith(codeFor(report, resolved), nil)
}

// codeFor turns a report into a process exit code.
func codeFor(rep localrun.Report, p localrun.Plan) int {
	required := map[string]bool{}
	for _, t := range p.Tests {
		required[t.Name] = t.Required
	}

	var sawFail, sawError, sawPass bool
	for _, r := range rep.Results {
		if !required[r.Name] {
			// An optional test is recorded and does not decide the run.
			continue
		}
		switch r.Phase {
		case api.RunFailed:
			sawFail = true
		case api.RunError:
			sawError = true
		case api.RunPassed:
			sawPass = true
		}
	}
	switch {
	case sawFail:
		return exitFail
	case sawError:
		return exitError
	case sawPass:
		return exitPass
	default:
		return exitNothingJudged
	}
}

// exitErr carries a code out through main.
type exitErr struct {
	code int
	err  error
}

func (e *exitErr) Error() string {
	if e.err == nil {
		return ""
	}
	return e.err.Error()
}

func exitWith(code int, err error) error {
	if code == exitPass && err == nil {
		return nil
	}
	return &exitErr{code: code, err: err}
}

// Hooks writes progress as it happens.
//
// To stderr, so a document or a summary on stdout stays clean and pipeable.
func Hooks(w *os.File) localrun.Hooks {
	return localrun.Hooks{
		OnTestStart: func(name string) {
			fmt.Fprintf(w, "  %-28s ", name)
		},
		OnTestComplete: func(r localrun.TestResult) {
			fmt.Fprintf(w, "%-8s %s\n", r.Phase, firstLineOf(r.Message))
		},
	}
}

// checkpointHook publishes a test's progress while it is still running.
//
// Two things happen, and only one of them may fail loudly. The progress line
// goes to the terminal always, because it is the thing a person watching a
// twelve-hour soak actually wants. The DELIVERY is best-effort: a checkpoint is
// evidence about a run that has not finished, and a sink that rejects one must
// never stop the run or change its verdict. So a failure is warned about and
// the soak continues — the opposite posture to the final envelope, which is the
// verdict and is worth saying loudly about.
//
// Nothing is delivered from a Pair server end, for the same reason its final
// envelope is not: the client decides the link, and a second document about one
// link in a consumer's history is worse than none.
func checkpointHook(
	ctx context.Context,
	w *os.File,
	id *RunIdentity,
	p localrun.Plan,
	sender *sink.Sender,
	rz *localrun.Rendezvous,
) func(localrun.Checkpoint) {
	return func(c localrun.Checkpoint) {
		fmt.Fprintf(w, "\n  %-28s checkpoint %d: %s\n", c.Test, c.Sequence, summariseMetrics(c.Metrics))

		if sender == nil || !deciding(rz) {
			return
		}
		t, ok := testNamed(p, c.Test)
		if !ok {
			return
		}
		env, err := id.Checkpoint(c, string(t.Spec.Kind), t.Spec.Scope, localrun.NodesFor(p, t))
		if err != nil {
			warn("checkpoint %d not delivered: %v", c.Sequence, err)
			return
		}
		if err := sender.Send(ctx, &env); err != nil {
			warn("checkpoint %d not delivered: %v", c.Sequence, err)
		}
	}
}

func testNamed(p localrun.Plan, name string) (localrun.PlannedTest, bool) {
	for _, t := range p.Tests {
		if t.Name == name {
			return t, true
		}
	}
	return localrun.PlannedTest{}, false
}

// summariseMetrics renders a checkpoint for a human watching a long soak.
//
// Sorted, because Go randomises map iteration and a progress line whose fields
// move between samples is one nobody can read down a column.
func summariseMetrics(m map[string]string) string {
	if len(m) == 0 {
		return "(no metrics yet)"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, k+"="+m[k])
	}
	line := strings.Join(parts, " ")
	if len(line) > 120 {
		line = line[:117] + "..."
	}
	return line
}

func firstLineOf(s string) string {
	line, _, _ := strings.Cut(strings.TrimSpace(s), "\n")
	if len(line) > 96 {
		line = line[:93] + "..."
	}
	return line
}

// resolveRuntime picks a container runtime.
func resolveRuntime(name string) (localrun.ContainerRuntime, error) {
	if name == "" || name == "auto" {
		rt, err := localrun.DetectRuntime()
		if err != nil {
			return nil, fmt.Errorf("%w — install one, or pass --runtime", err)
		}
		return rt, nil
	}
	return localrun.NewRuntime(name)
}

// lintPlan reports thresholds that can never do what their author intended.
//
// Advisory, exactly as in the operator: it changes nothing about evaluation,
// which still fails closed. The point is to say it at authoring time, while
// someone is standing here, rather than have it surface as a hardware verdict.
func lintPlan(p localrun.Plan) {
	for _, t := range p.Tests {
		for _, problem := range verdict.ValidateThresholdsForKind(t.Spec.Kind, t.Spec.Thresholds) {
			warn("%s threshold %d (%s): %s", t.Name, problem.Index, problem.Metric, problem.Reason)
		}
	}
}

func printPlan(p localrun.Plan) {
	fmt.Printf("node: %s\n", p.Node)
	fmt.Printf("failFast: %v   retryOnError: %d\n", p.FailFast, p.RetryOnErrorLimit)
	if rz := p.Rendezvous; rz != nil {
		// Printed before the tests, because it is the thing most worth checking
		// before walking to the other machine.
		fmt.Printf("role: %s", rz.Role)
		if rz.PeerHost != "" {
			fmt.Printf("   peer: %s", rz.PeerHost)
		}
		if rz.PeerNode != "" {
			fmt.Printf("   peer node: %s", rz.PeerNode)
		}
		if deciding(rz) {
			fmt.Printf("\n(this end decides the link's verdict)")
		} else {
			fmt.Printf("\n(the client end decides; this end writes a record)")
		}
		fmt.Println()
	}
	fmt.Println()
	for i, t := range p.Tests {
		spec, err := localrun.Translate(p, t)
		image := spec.Image
		if err != nil {
			image = "UNRESOLVED: " + err.Error()
		}
		req := "optional"
		if t.Required {
			req = "required"
		}
		fmt.Printf("%d. %-24s %-9s %s\n", i+1, t.Name, req, image)
		// The cell's axes, where this line is one cell of an expanded entry.
		// Printed because the arithmetic is otherwise invisible: one profile
		// entry with four precision variants is four numbered lines here, and
		// without the labels a reader has only the names to tell them apart —
		// which is exactly what TestResult.VariantAxes exists so that a
		// consumer does not have to do.
		if len(t.Axes) > 0 {
			pairs := make([]string, 0, len(t.Axes))
			for _, k := range plan.SortedAxisKeys(t.Axes) {
				pairs = append(pairs, fmt.Sprintf("%s=%s", k, t.Axes[k]))
			}
			fmt.Printf("     variant of %s   %s\n", t.Parent, strings.Join(pairs, " "))
		}
		if len(t.Spec.Thresholds) > 0 {
			for _, th := range t.Spec.Thresholds {
				fmt.Printf("     gate %s %s %s\n", th.Metric, th.Comparison, th.Value)
			}
		}
	}
}

func printSummary(rep localrun.Report) {
	fmt.Printf("\n%s — %d passed, %d failed, %d errored, %d skipped\n",
		rep.Phase, rep.Summary.Passed, rep.Summary.Failed, rep.Summary.Errored, rep.Summary.Skipped)

	for _, r := range rep.Results {
		if r.Phase == api.RunPassed || r.Phase == api.RunSkipped {
			continue
		}
		fmt.Printf("\n%s: %s\n", r.Name, r.Phase)
		if r.Message != "" {
			fmt.Printf("  %s\n", r.Message)
		}
		for _, v := range r.Violations {
			// The cause first: it is the difference between replacing a part
			// and fixing a profile.
			fmt.Printf("  [%s] %s — %s\n", v.Cause, v.Metric, v.Reason)
		}
	}
}

// writeResults lays out a results directory.
//
// The same shape `burnin report --results-dir` reads, so the two halves of the
// CLI fit together without a format anyone has to remember.
// requiredNames is which tests decide the run's exit code, for a rank record
// that will be merged on another machine where no plan exists.
func requiredNames(p localrun.Plan) map[string]bool {
	out := map[string]bool{}
	for _, t := range p.Tests {
		out[t.Name] = t.Required
	}
	return out
}

func writeResults(dir string, rep localrun.Report, rz *localrun.Rendezvous, p localrun.Plan) error {
	// raw/ always; the verdict directory only where a verdict is produced. An
	// empty envelopes/ on a server end reads as a run that lost its results.
	if err := os.MkdirAll(filepath.Join(dir, "raw"), 0o755); err != nil {
		return err
	}

	runFile := map[string]any{
		"node":       rep.Node,
		"phase":      string(rep.Phase),
		"startedAt":  rep.StartedAt.UTC().Format(time.RFC3339),
		"finishedAt": rep.FinishedAt.UTC().Format(time.RFC3339),
	}
	if err := writeJSON(filepath.Join(dir, "run.json"), runFile); err != nil {
		return err
	}

	// EVERY RANK of a collective writes a record, and none writes an envelope.
	//
	// Not because a rank has nothing to say, but because no rank can say the
	// thing that matters: "did every rank take part" is invisible from inside
	// one of them. The verdict is rendered by `burnin merge`, which is the only
	// thing that can count to N — see pkg/group.Verdict, which refuses a
	// partial turnout for Pass and for Skip alike.
	if rz != nil && rz.Rank != nil {
		rec := rankRecord{
			Rank:        int(*rz.Rank),
			NRanks:      int(rz.NRanks),
			Node:        rep.Node,
			Phase:       string(rep.Phase),
			StartedAt:   rep.StartedAt.UTC(),
			FinishedAt:  rep.FinishedAt.UTC(),
			Results:     rep.Results,
			Required:    requiredNames(p),
			Fingerprint: fingerprint(rep.Node),
			Note: "One rank of a collective. NOT a verdict: a group verdict is about the " +
				"COLLECTIVE, and no single rank knows whether every other rank took part. " +
				"Run `burnin merge` over the directory holding every rank's record.",
		}
		return writeJSON(filepath.Join(dir, "ranks", rankRecordName(rec.Rank)), rec)
	}

	// The SERVER end of a link writes a sidecar, not an envelope.
	//
	// Into its own directory, so `burnin report --results-dir <dir>/envelopes`
	// cannot pick it up: a Pair verdict is about the LINK and there is exactly
	// one of them. Two envelopes for one measurement would render as two
	// results, and an engineer comparing them would be comparing a measurement
	// against its own echo.
	if !deciding(rz) {
		rec := map[string]any{
			"role":     rz.Role,
			"node":     rep.Node,
			"peerNode": rz.PeerNode,
			"phase":    string(rep.Phase),
			"note": "The server end of a link test. Not a verdict: the client's " +
				"envelope is the measurement. Kept because a server that failed " +
				"or never started explains a client result that otherwise looks " +
				"like a fabric fault.",
			"results": rep.Results,
		}
		return writeJSON(filepath.Join(dir, "sidecar", "server-record.json"), rec)
	}

	env, err := EnvelopeFor(rep)
	if err != nil {
		return err
	}
	return writeJSON(filepath.Join(dir, "envelopes", "001-RunPhaseChanged.json"), env)
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

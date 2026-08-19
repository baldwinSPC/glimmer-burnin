package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
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

	rz, err := f.pair.rendezvous()
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
		return exitWith(exitError, fmt.Errorf(
			"--role was given but no test in this profile is Pair- or Group-scope; nothing here measures a link"))
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

	report, runErr := localrun.Run(ctx, resolved, rt, Hooks(os.Stderr))
	if runErr != nil && ctx.Err() == nil {
		return exitWith(exitError, runErr)
	}

	if f.resultsDir != "" {
		if err := writeResults(f.resultsDir, report, rz); err != nil {
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
		} else if env, err := EnvelopeFor(report); err != nil {
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
func writeResults(dir string, rep localrun.Report, rz *localrun.Rendezvous) error {
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

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

	s, err := loadSuite(f.files)
	if err != nil {
		return exitWith(exitError, err)
	}
	plan, warnings, err := s.buildPlan(f.profile, node, int32(f.retries))
	if err != nil {
		return exitWith(exitError, err)
	}
	for _, w := range warnings {
		warn("%s", w)
	}

	// Threshold linting before anything runs, so an unsatisfiable gate is
	// reported while its author is still here rather than as a verdict on
	// hardware hours later.
	lintPlan(plan)

	if f.dryRun {
		printPlan(plan)
		return nil
	}

	rt, err := resolveRuntime(f.runtime)
	if err != nil {
		return exitWith(exitError, err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "burnin: %d test(s) on %s via %s\n", len(plan.Tests), node, rt.Name())

	report, runErr := localrun.Run(ctx, plan, rt, Hooks(os.Stderr))
	if runErr != nil && ctx.Err() == nil {
		return exitWith(exitError, runErr)
	}

	if f.resultsDir != "" {
		if err := writeResults(f.resultsDir, report); err != nil {
			// The run happened and its verdict stands; failing to file it is
			// worth saying loudly and is not a reason to discard the verdict.
			warn("results were not written: %v", err)
		}
	}

	printSummary(report)
	if ctx.Err() != nil {
		return exitWith(exitError, fmt.Errorf("interrupted before every test reached a verdict"))
	}
	return exitWith(codeFor(report, plan), nil)
}

// codeFor turns a report into a process exit code.
func codeFor(rep localrun.Report, plan localrun.Plan) int {
	required := map[string]bool{}
	for _, t := range plan.Tests {
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
	fmt.Printf("failFast: %v   retryOnError: %d\n\n", p.FailFast, p.RetryOnErrorLimit)
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
func writeResults(dir string, rep localrun.Report) error {
	envDir := filepath.Join(dir, "envelopes")
	rawDir := filepath.Join(dir, "raw")
	for _, d := range []string{envDir, rawDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			return err
		}
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

	env, err := EnvelopeFor(rep)
	if err != nil {
		return err
	}
	return writeJSON(filepath.Join(envDir, "001-RunPhaseChanged.json"), env)
}

func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

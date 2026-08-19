package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

func writeSuite(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "suite.yaml")
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

const oneProfile = `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: smoke
spec:
  kind: compute-smoke
  thresholds:
    - metric: sustainedClockPct
      comparison: GreaterThanOrEqual
      value: "70"
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: acceptance
spec:
  tests:
    - testRef: smoke
`

func TestASuiteIsTheSameYAMLAClusterTakes(t *testing.T) {
	// The whole reason for the apimachinery dependency: a profile is
	// copy-paste identical between kubectl and a bare host.
	s, err := loadSuite([]string{writeSuite(t, oneProfile)})
	if err != nil {
		t.Fatalf("loadSuite: %v", err)
	}
	plan, _, err := s.buildPlan("", "n1", 0, nil)
	if err != nil {
		t.Fatalf("buildPlan: %v", err)
	}
	if len(plan.Tests) != 1 || plan.Tests[0].Name != "smoke" {
		t.Fatalf("plan = %+v, want the referenced test", plan.Tests)
	}
	if len(plan.Tests[0].Spec.Thresholds) != 1 {
		t.Error("the test's thresholds did not survive loading")
	}
}

func TestClusterOnlyDocumentsAreIgnoredNotRejected(t *testing.T) {
	// One file should serve both paths. A sink or a schedule beside a profile
	// means something in a cluster and nothing here, and refusing the file
	// would force sites to maintain two copies — which is what this design
	// exists to avoid.
	body := oneProfile + `
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInSink
metadata:
  name: results
spec:
  type: ConfigMap
  configMap:
    name: burnin-results
`
	if _, err := loadSuite([]string{writeSuite(t, body)}); err != nil {
		t.Fatalf("a BurnInSink beside a profile should be ignored, got %v", err)
	}
}

func TestAnAmbiguousProfileIsRefusedRatherThanGuessed(t *testing.T) {
	// Picking the first silently means a user gets a run they did not ask for,
	// against hardware they care about.
	body := oneProfile + `
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: soak
spec:
  tests:
    - testRef: smoke
`
	s, err := loadSuite([]string{writeSuite(t, body)})
	if err != nil {
		t.Fatalf("loadSuite: %v", err)
	}

	if _, _, err := s.buildPlan("", "n1", 0, nil); err == nil {
		t.Fatal("two profiles and no --profile was accepted")
	} else if !strings.Contains(err.Error(), "--profile") {
		t.Errorf("the error should say how to choose: %v", err)
	}

	// Naming one works, and the error for a wrong name lists what exists.
	if _, _, err := s.buildPlan("soak", "n1", 0, nil); err != nil {
		t.Errorf("naming a profile should work: %v", err)
	}
	_, _, err = s.buildPlan("nope", "n1", 0, nil)
	if err == nil || !strings.Contains(err.Error(), "acceptance") {
		t.Errorf("an unknown profile should list the ones that exist: %v", err)
	}
}

func TestAMissingTestReferenceSaysWhatToDo(t *testing.T) {
	body := `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: acceptance
spec:
  tests:
    - testRef: not-here
`
	s, err := loadSuite([]string{writeSuite(t, body)})
	if err != nil {
		t.Fatalf("loadSuite: %v", err)
	}
	_, _, err = s.buildPlan("", "n1", 0, nil)
	if err == nil {
		t.Fatal("a dangling testRef was accepted")
	}
	if !strings.Contains(err.Error(), "pass the file") {
		t.Errorf("the error should suggest the likely fix: %v", err)
	}
}

func TestAPairScopeTestWarnsRatherThanFailingSilently(t *testing.T) {
	// A user whose profile has a Pair test has a belief about what will run.
	// Letting it pass without a word leaves the belief intact and wrong.
	body := `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: p
spec:
  tests:
    - inline:
        kind: ib-write-bw
        scope: Pair
`
	s, _ := loadSuite([]string{writeSuite(t, body)})
	_, warnings, err := s.buildPlan("", "n1", 0, nil)
	if err != nil {
		t.Fatalf("buildPlan: %v", err)
	}
	if len(warnings) == 0 || !strings.Contains(strings.Join(warnings, " "), "peer") {
		t.Errorf("a Pair-scope test should warn that it needs a peer: %v", warnings)
	}
}

// ─── Exit codes ──────────────────────────────────────────────────────────────

func TestExitCodesKeepErrorDistinctFromFail(t *testing.T) {
	// The reason these are separate codes at all: a CI job has to tell "this
	// hardware is bad" from "this run never happened". They lead to different
	// actions, and only one is worth retrying.
	plan := localrun.Plan{Tests: []localrun.PlannedTest{
		{Name: "a", Required: true},
		{Name: "b", Required: true},
	}}

	rows := []struct {
		name    string
		results []localrun.TestResult
		want    int
	}{
		{"all passed", []localrun.TestResult{{Name: "a", Phase: api.RunPassed}, {Name: "b", Phase: api.RunPassed}}, exitPass},
		{"a failure", []localrun.TestResult{{Name: "a", Phase: api.RunPassed}, {Name: "b", Phase: api.RunFailed}}, exitFail},
		{"an error", []localrun.TestResult{{Name: "a", Phase: api.RunPassed}, {Name: "b", Phase: api.RunError}}, exitError},
		{"a failure outranks an error", []localrun.TestResult{{Name: "a", Phase: api.RunError}, {Name: "b", Phase: api.RunFailed}}, exitFail},
		{"everything skipped is not a pass", []localrun.TestResult{{Name: "a", Phase: api.RunSkipped}, {Name: "b", Phase: api.RunSkipped}}, exitNothingJudged},
	}
	for _, r := range rows {
		if got := codeFor(localrun.Report{Results: r.results}, plan); got != r.want {
			t.Errorf("%s: exit = %d, want %d", r.name, got, r.want)
		}
	}
}

func TestAnOptionalTestDoesNotDecideTheRun(t *testing.T) {
	plan := localrun.Plan{Tests: []localrun.PlannedTest{
		{Name: "required", Required: true},
		{Name: "optional", Required: false},
	}}
	rep := localrun.Report{Results: []localrun.TestResult{
		{Name: "required", Phase: api.RunPassed},
		{Name: "optional", Phase: api.RunFailed},
	}}
	if got := codeFor(rep, plan); got != exitPass {
		t.Errorf("exit = %d, want %d — an optional test is recorded and does not condemn the run", got, exitPass)
	}
}

// ─── Envelope and the round trip ─────────────────────────────────────────────

func sampleReport() localrun.Report {
	start := time.Date(2026, 8, 6, 10, 0, 0, 0, time.UTC)
	return localrun.Report{
		Node:       "spark-a",
		Phase:      api.RunFailed,
		StartedAt:  start,
		FinishedAt: start.Add(5 * time.Minute),
		Summary:    localrun.Summary{Passed: 1, Failed: 1},
		Results: []localrun.TestResult{
			{
				Name: "smoke", Kind: "compute-smoke", Phase: api.RunPassed, Nodes: []string{"spark-a"},
				StartedAt: start, FinishedAt: start.Add(time.Minute),
				Metrics: map[string]string{"throughputTflops": "101.99"},
			},
			{
				Name: "clock", Kind: "clockprobe", Phase: api.RunFailed, Nodes: []string{"spark-a"},
				StartedAt: start, FinishedAt: start.Add(2 * time.Minute),
				Message:      "sustainedClockPct 61.2 below floor 70",
				Metrics:      map[string]string{"sustainedClockPct": "61.2"},
				Violations:   []api.Violation{{Metric: "sustainedClockPct", Cause: "Measurement", Kind: "Unsatisfied", Reason: "61.2 < 70"}},
				NotEvaluated: []api.NotEvaluated{{Metric: "eccErrors", Reason: "unmeasurable on this hardware"}},
				Unmeasurable: []string{"eccErrors"},
			},
		},
	}
}

func TestTheEnvelopeIsTheSameDocumentTheOperatorDelivers(t *testing.T) {
	// What makes the dual path pay off downstream: one document shape, so
	// `burnin report` renders it and an ingest endpoint accepts it unchanged.
	env, err := EnvelopeFor(sampleReport())
	if err != nil {
		t.Fatalf("EnvelopeFor: %v", err)
	}
	// Validate is the contract's own, so this cannot drift into producing
	// something the rest of the system rejects.
	if err := env.Validate(); err != nil {
		t.Fatalf("the built envelope is invalid: %v", err)
	}

	if env.Run.Namespace != "local" {
		t.Errorf("namespace = %q — it should say plainly that no apiserver knew about this run", env.Run.Namespace)
	}
	if env.Run.UID == "" || env.DeliveryID == "" {
		t.Error("a run needs an identity and a delivery key for a receiver to deduplicate on")
	}
	if len(env.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(env.Results))
	}

	var clock = env.Results[1]
	if len(clock.Violations) != 1 || clock.Violations[0].Cause != "Measurement" {
		t.Errorf("violations did not survive with their cause: %+v", clock.Violations)
	}
	if len(clock.NotEvaluated) != 1 || len(clock.Unmeasurable) != 1 {
		t.Error("an unevaluated gate and an unmeasurable metric must reach the envelope")
	}
	if env.Summary.Failed != 1 || env.Summary.Passed != 1 {
		t.Errorf("summary = %+v", env.Summary)
	}
}

func TestTwoRunsOnOneMachineAreNotOneRun(t *testing.T) {
	// A derived UID would make the second run dedupe away as a replay of the
	// first at any receiver keyed on the delivery ID.
	a, err := EnvelopeFor(sampleReport())
	if err != nil {
		t.Fatal(err)
	}
	b, err := EnvelopeFor(sampleReport())
	if err != nil {
		t.Fatal(err)
	}
	if a.Run.UID == b.Run.UID {
		t.Error("two runs share a UID — the second would be discarded as a duplicate")
	}
	if a.DeliveryID == b.DeliveryID {
		t.Error("two runs share a delivery ID")
	}
}

func TestTheTwoHalvesOfTheCLIFitTogether(t *testing.T) {
	// `burnin run --results-dir` writes what `burnin report --results-dir`
	// reads. If that ever stops being true the CLI has two halves and no
	// feature, so it is asserted rather than assumed.
	dir := t.TempDir()
	if err := writeResults(dir, sampleReport(), nil); err != nil {
		t.Fatalf("writeResults: %v", err)
	}

	out := t.TempDir()
	if err := runReport([]string{"--results-dir", filepath.Join(dir, "envelopes"), "-o", "markdown", "--out", out}); err != nil {
		t.Fatalf("report could not read what run wrote: %v", err)
	}

	entries, _ := os.ReadDir(out)
	if len(entries) == 0 {
		t.Fatal("no document rendered from the results directory")
	}
	body, _ := os.ReadFile(filepath.Join(out, entries[0].Name()))
	if !strings.Contains(string(body), "clock") {
		t.Error("the rendered report does not mention the failing test")
	}
}

func TestResultsDirectoryLayoutIsStable(t *testing.T) {
	dir := t.TempDir()
	if err := writeResults(dir, sampleReport(), nil); err != nil {
		t.Fatalf("writeResults: %v", err)
	}
	for _, want := range []string{"run.json", "envelopes", "raw"} {
		if _, err := os.Stat(filepath.Join(dir, want)); err != nil {
			t.Errorf("missing %s: %v", want, err)
		}
	}
}

// ─── Variants ────────────────────────────────────────────────────────────────

// THE BUG THIS PINS: buildPlan read spec.tests[].variants, the CRD schema
// validated them, and then they were silently discarded. A four-cell precision
// sweep — the entire point of the profile — ran as ONE execution of the parent
// test, under the parent's name, with none of the cell's thresholds applied.
// Nothing in the output said a cell was missing, so the only way to notice was
// to run the same profile in a cluster and count the results.
func TestVariantsExpandIntoOneExecutionPerCell(t *testing.T) {
	body := `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: sweep
spec:
  tests:
    - testRef: gemm
      variants:
        - name: fp4
          axes: {precision: fp4}
          thresholds:
            - metric: achievedTflops
              comparison: GreaterThanOrEqual
              value: "700"
        - name: bf16
          axes: {precision: bf16}
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: gemm
spec:
  kind: compute-smoke
  thresholds:
    - metric: nonfiniteCount
      comparison: Equal
      value: "0"
`
	s, err := loadSuite([]string{writeSuite(t, body)})
	if err != nil {
		t.Fatalf("loadSuite: %v", err)
	}
	p, _, err := s.buildPlan("", "n1", 0, nil)
	if err != nil {
		t.Fatalf("buildPlan: %v", err)
	}

	if len(p.Tests) != 2 {
		t.Fatalf("got %d executions, want 2 — the profile's variants were dropped", len(p.Tests))
	}
	if p.Tests[0].Name != "gemm-fp4" || p.Tests[1].Name != "gemm-bf16" {
		t.Fatalf("names = %q/%q, want gemm-fp4/gemm-bf16 — a cell's name is its result identity",
			p.Tests[0].Name, p.Tests[1].Name)
	}
	if got := p.Tests[0].Axes["precision"]; got != "fp4" {
		t.Errorf("fp4 cell axes = %v, want precision=fp4 carried onto the execution", p.Tests[0].Axes)
	}
	if p.Tests[0].Parent != "gemm" {
		t.Errorf("Parent = %q, want gemm", p.Tests[0].Parent)
	}
	// The overlay REPLACES: the fp4 cell must not still carry the parent's gate.
	if th := p.Tests[0].Spec.Thresholds; len(th) != 1 || th[0].Metric != "achievedTflops" {
		t.Errorf("fp4 thresholds = %+v, want only its own — a retained parent gate is invisible in the profile", th)
	}
	// A cell with no threshold overlay INHERITS.
	if th := p.Tests[1].Spec.Thresholds; len(th) != 1 || th[0].Metric != "nonfiniteCount" {
		t.Errorf("bf16 thresholds = %+v, want the parent's inherited", th)
	}
}

// The axes must reach the RUNNER, not merely the plan. A cell that planned
// correctly but ran without BURNIN_VARIANT_PRECISION would run the default
// configuration and be reported as the fp4 cell — the same wrong answer the
// unreachable-axis refusal exists to prevent, reached by forgetting to inject.
func TestAVariantCellsAxesReachTheRunnersEnvironment(t *testing.T) {
	body := `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: sweep
spec:
  tests:
    - testRef: t
      variants:
        - name: fp4
          axes: {precision: fp4, class: smoke}
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: t
spec:
  kind: custom
  runner:
    image: example.invalid/runner:test
`
	s, _ := loadSuite([]string{writeSuite(t, body)})
	p, _, err := s.buildPlan("", "n1", 0, nil)
	if err != nil {
		t.Fatalf("buildPlan: %v", err)
	}
	spec, err := localrun.Translate(p, p.Tests[0])
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if got := spec.Env["BURNIN_VARIANT_PRECISION"]; got != "fp4" {
		t.Errorf("BURNIN_VARIANT_PRECISION = %q, want fp4; env was %v", got, spec.Env)
	}
	if got := spec.Env["BURNIN_VARIANT_CLASS"]; got != "smoke" {
		t.Errorf("BURNIN_VARIANT_CLASS = %q, want smoke", got)
	}
}

// #296, on this dispatcher: an axis that cannot become an environment variable
// is refused before a container starts, rather than skipped — the cell would
// otherwise run the DEFAULT configuration while the run reported it under a
// distinct label, which is a confident wrong answer dressed as evidence.
func TestAnAxisThatCannotReachTheRunnerIsRefused(t *testing.T) {
	body := `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: sweep
spec:
  tests:
    - testRef: t
      variants:
        - name: big
          axes: {message-bytes: "8"}
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: t
spec:
  kind: custom
  runner:
    image: example.invalid/runner:test
`
	s, _ := loadSuite([]string{writeSuite(t, body)})
	_, _, err := s.buildPlan("", "n1", 0, nil)
	if err == nil {
		t.Fatal("buildPlan accepted an axis the runner could never receive")
	}
	if !strings.Contains(err.Error(), "message-bytes") || !strings.Contains(err.Error(), "t-big") {
		t.Errorf("the refusal must name both the axis and the cell: %v", err)
	}
}

// Two variants of one test sharing a name collide exactly as two tests sharing
// one do: the second would never run and its verdict would silently overwrite
// the first's.
func TestTwoVariantsWithTheSameNameAreRefused(t *testing.T) {
	body := `
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: sweep
spec:
  tests:
    - testRef: t
      variants:
        - name: a
        - name: a
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: t
spec:
  kind: custom
  runner:
    image: example.invalid/runner:test
`
	s, _ := loadSuite([]string{writeSuite(t, body)})
	if _, _, err := s.buildPlan("", "n1", 0, nil); err == nil {
		t.Fatal("two variants named the same were accepted")
	}
}

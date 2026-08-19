package localrun

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runnerimages"
)

// The conformance table.
//
// This file is the reason pkg/localrun is allowed to exist as a second
// dispatcher. The promise of the design is that both reach the SAME verdict from
// the SAME evidence; a divergence would surface as a node that passes on one
// path and fails on the other, and nobody would know which to believe.
//
// Every row below names the controller code it mirrors, so a reviewer changing
// one can find the other. The controller's own tests assert the same table
// against internal/controller/burninrun_controller.go — completeAttempt for the
// repeat/retry rules and attemptOutcome for the phase rules.
//
// Extracting a state machine both dispatchers consume is the right long-term
// answer and is deliberately deferred. This is the cheap thing that catches the
// drift today, and a cheap thing that runs beats an elegant thing that is
// planned.

// fakeRuntime replays scripted executions, so the engine can be exercised with
// no container runtime present.
type fakeRuntime struct {
	steps []Execution
	calls int
	specs []RunSpec
}

func (f *fakeRuntime) Name() string { return "fake" }

func (f *fakeRuntime) Run(_ context.Context, spec RunSpec) (Execution, error) {
	f.specs = append(f.specs, spec)
	if f.calls >= len(f.steps) {
		return Execution{}, fmt.Errorf("fake runtime: unexpected call %d", f.calls+1)
	}
	e := f.steps[f.calls]
	f.calls++
	return e, nil
}

func exits(code int, stdout string) Execution {
	return Execution{ExitCode: code, Stdout: stdout}
}

// testSpec builds a spec whose image resolves without touching the real table.
func testSpec(thresholds ...api.Threshold) api.BurnInTestSpec {
	return api.BurnInTestSpec{
		Kind:       api.KindCustom,
		Runner:     &api.RunnerSpec{Image: "example.invalid/runner:test"},
		Thresholds: thresholds,
	}
}

func onePlan(t PlannedTest, retries int32) Plan {
	return Plan{Node: "n1", Tests: []PlannedTest{t}, RetryOnErrorLimit: retries}
}

func runOne(t *testing.T, p Plan, rt *fakeRuntime) TestResult {
	t.Helper()
	clock := time.Date(2026, 8, 6, 12, 0, 0, 0, time.UTC)
	rep, err := runWithClock(context.Background(), p, rt, Hooks{}, func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rep.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(rep.Results))
	}
	return rep.Results[0]
}

// ─── The phase rules (mirrors attemptOutcome) ────────────────────────────────

func TestConformance_ExitCodeToPhase(t *testing.T) {
	// runner.VerdictFor: 0 pass, 1 fail, 2 skip, anything else Error.
	// A skip additionally requires a declared marker; see the next test.
	rows := []struct {
		name string
		exec Execution
		want api.RunPhase
		why  string
	}{
		{"exit 0 with no thresholds passes", exits(0, "tflops=101\n"), api.RunPassed,
			"exit 0 is the runner saying it is content, and with no gates there is nothing to overturn it"},
		{"exit 1 fails", exits(1, "MARKER_FAIL\n"), api.RunFailed,
			"exit 1 is the runner's OWN assertion that the hardware failed, honoured with or without metrics"},
		{"exit 2 with a marker skips", exits(2, "CUSTOM_SKIP not applicable\n"), api.RunSkipped,
			"a declared skip is the runner saying the test does not apply to this part"},
		{"exit 3 errors", exits(3, "something broke\n"), api.RunError,
			"anything outside 0/1/2 is machinery, not a hardware verdict"},
		{"exit 137 errors", exits(137, ""), api.RunError,
			"an OOM kill is not a hardware verdict"},
	}

	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 0),
				&fakeRuntime{steps: []Execution{r.exec}})
			if got.Phase != r.want {
				t.Errorf("phase = %s, want %s — %s", got.Phase, r.want, r.why)
			}
		})
	}
}

func TestConformance_AnUndeclaredSkipIsAnError(t *testing.T) {
	// pkg/runner sets UndeclaredSkip when a runner exits 2 without printing a
	// marker. A Go panic exits 2 and prints a stack trace, which is byte-for-byte
	// the shape of a legitimate skip — whose normal form is no metrics at all.
	//
	// Mirrors: attemptOutcome's default arm plus explainUndeclaredSkip.
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 0),
		&fakeRuntime{steps: []Execution{exits(2, "panic: nil map\n/src/main.go:132 +0x1d\n")}})

	if got.Phase != api.RunError {
		t.Fatalf("phase = %s, want Error — exit 2 with no declaration is a crash, not a skip", got.Phase)
	}
	if !strings.Contains(got.Message, "UNJUDGED") {
		t.Errorf("the message must say the hardware is unjudged rather than out of scope: %q", got.Message)
	}
}

func TestConformance_AnEmptyHarvestWithThresholdsIsAnError(t *testing.T) {
	// Exit 0, no key=value line, and gates to clear. Handing an empty map to
	// fail-closed evaluation would manufacture a hardware verdict out of an
	// absence — an evicted or SIGTERMed runner lands exactly here.
	//
	// Mirrors: attemptOutcome's ReportedNothing() branch.
	th := api.Threshold{Metric: "tflops", Comparison: api.GTE, Value: "100"}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(th), Required: true}, 0),
		&fakeRuntime{steps: []Execution{exits(0, "")}})

	if got.Phase != api.RunError {
		t.Fatalf("phase = %s, want Error", got.Phase)
	}
	if got.Phase == api.RunFailed {
		t.Fatal("an unmeasured node was condemned — fail-closed was applied to an absence")
	}
}

func TestConformance_SomeMetricsButAGatedOneMissingStillFails(t *testing.T) {
	// The empty-harvest rule must NOT be widened. A runner that looked at the
	// hardware and omitted a measurement it owed is exactly the silence that
	// must never satisfy acceptance.
	//
	// Mirrors: the "THIS DOES NOT RELAX FAIL-CLOSED" comment in attemptOutcome.
	th := api.Threshold{Metric: "eccErrors", Comparison: api.EQ, Value: "0"}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(th), Required: true}, 0),
		&fakeRuntime{steps: []Execution{exits(0, "tflops=101\n")}})

	if got.Phase != api.RunFailed {
		t.Fatalf("phase = %s, want Failed — a gated metric the runner did not emit must fail closed", got.Phase)
	}
}

func TestConformance_ASkipIsNotSubjectToTheEmptyHarvestRule(t *testing.T) {
	// Zero metrics is the NORMAL shape of a skip, and no threshold is evaluated
	// on that path. Applying the empty-harvest rule here would turn every clean
	// skip into a retried Error.
	//
	// Mirrors: attemptOutcome's VerdictSkip arm.
	th := api.Threshold{Metric: "tflops", Comparison: api.GTE, Value: "100"}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(th), Required: true}, 0),
		&fakeRuntime{steps: []Execution{exits(2, "CUSTOM_SKIP no accelerator visible\n")}})

	if got.Phase != api.RunSkipped {
		t.Fatalf("phase = %s, want Skipped", got.Phase)
	}
}

func TestConformance_AFailKeepsTheRunnersOwnMessage(t *testing.T) {
	// The runner's words are kept, not replaced. At Node scope that is one
	// line; the same code path at Pair or Group carries the clause saying the
	// verdict is about the LINK rather than any one node.
	//
	// Mirrors: "THE RUNNER'S OWN MESSAGE IS KEPT" in attemptOutcome.
	th := api.Threshold{Metric: "tflops", Comparison: api.GTE, Value: "100"}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(th), Required: true}, 0),
		&fakeRuntime{steps: []Execution{exits(0, "tflops=50\nthermal throttling detected\n")}})

	if got.Phase != api.RunFailed {
		t.Fatalf("phase = %s, want Failed", got.Phase)
	}
	if !strings.Contains(got.Message, "thermal throttling detected") {
		t.Errorf("the runner's own message was dropped: %q", got.Message)
	}
	if len(got.Violations) == 0 {
		t.Error("no violations recorded — the cause of the failure is the decision-relevant part")
	}
}

func TestConformance_UnmeasurableIsRecordedOnEveryPhase(t *testing.T) {
	// "This part cannot produce this measurement" is a claim about the HARDWARE
	// and is true of a passing run as much as a failing one.
	//
	// Mirrors: the loop above the switch in attemptOutcome.
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 0),
		&fakeRuntime{steps: []Execution{exits(0, "eccErrors=n/a\ntflops=101\n")}})

	if got.Phase != api.RunPassed {
		t.Fatalf("phase = %s, want Passed", got.Phase)
	}
	if len(got.Unmeasurable) != 1 || got.Unmeasurable[0] != "eccErrors" {
		t.Errorf("unmeasurable = %v, want it recorded on a passing run too", got.Unmeasurable)
	}
}

// ─── The repeat and retry rules (mirrors completeAttempt) ────────────────────

func TestConformance_OnlyErrorIsRetried(t *testing.T) {
	// The rule with teeth. retryOnErrorLimit re-runs an Error and nothing else:
	// a Fail settles where it happened with the budget UNSPENT, because
	// re-running a measurement until it comes out clean launders a hardware
	// fault into an acceptance.
	//
	// Mirrors: completeAttempt's switch — the RunFailed arm settles, and only
	// the default (Error) arm consults retryOnErrorLimit.
	rows := []struct {
		name      string
		first     Execution
		wantCalls int
		wantPhase api.RunPhase
		why       string
	}{
		{"an Error is retried", exits(3, "broke"), 2, api.RunPassed,
			"an errored attempt measured nothing, so running it again is not laundering anything"},
		{"a Fail is NOT retried", exits(1, "MARKER_FAIL"), 1, api.RunFailed,
			"re-running a measurement until it comes out clean launders a hardware fault into an acceptance"},
		{"a Skip is NOT retried", exits(2, "CUSTOM_SKIP"), 1, api.RunSkipped,
			"retrying will not make an inapplicable test start applying"},
	}

	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			rt := &fakeRuntime{steps: []Execution{r.first, exits(0, "ok=1\n")}}
			got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 3), rt)

			if rt.calls != r.wantCalls {
				t.Errorf("ran %d attempt(s), want %d — %s", rt.calls, r.wantCalls, r.why)
			}
			if got.Phase != r.wantPhase {
				t.Errorf("phase = %s, want %s", got.Phase, r.wantPhase)
			}
		})
	}
}

func TestConformance_AnErroredAttemptDoesNotConsumeARepeat(t *testing.T) {
	// RepeatsCompleted is deliberately untouched on the Error path: an attempt
	// that measured nothing owes its repeat again.
	//
	// Mirrors: "RepeatsCompleted is deliberately untouched here" in completeAttempt.
	three := int32(3)
	spec := testSpec()
	spec.RepeatCount = &three

	rt := &fakeRuntime{steps: []Execution{
		exits(0, "ok=1\n"), // repeat 1
		exits(3, "broke"),  // error, retried, no repeat consumed
		exits(0, "ok=1\n"), // repeat 2
		exits(0, "ok=1\n"), // repeat 3
	}}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: spec, Required: true}, 1), rt)

	if got.Phase != api.RunPassed {
		t.Fatalf("phase = %s, want Passed", got.Phase)
	}
	if got.RepeatsCompleted != 3 {
		t.Errorf("RepeatsCompleted = %d, want 3 — the errored attempt must not have counted", got.RepeatsCompleted)
	}
	if got.ErrorRetries != 1 {
		t.Errorf("ErrorRetries = %d, want 1", got.ErrorRetries)
	}
	if rt.calls != 4 {
		t.Errorf("ran %d attempts, want 4", rt.calls)
	}
}

func TestConformance_RepeatsAreANDNotOR(t *testing.T) {
	// Every repeat must pass. One failure settles the test, however many
	// passes preceded it.
	//
	// Mirrors: completeAttempt's RunPassed arm requiring
	// RepeatsCompleted >= RepeatsRequired before settling.
	three := int32(3)
	spec := testSpec()
	spec.RepeatCount = &three

	rt := &fakeRuntime{steps: []Execution{
		exits(0, "ok=1\n"),
		exits(1, "MARKER_FAIL second time"),
		exits(0, "ok=1\n"), // must never run
	}}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: spec, Required: true}, 0), rt)

	if got.Phase != api.RunFailed {
		t.Fatalf("phase = %s, want Failed — repeats are AND", got.Phase)
	}
	if rt.calls != 2 {
		t.Errorf("ran %d attempts, want 2 — the run should stop at the failure", rt.calls)
	}
}

func TestConformance_TheRetryBudgetIsPerTestAndDoesNotReset(t *testing.T) {
	// Mirrors: ErrorRetries lives on the TestResult and is only ever
	// incremented.
	rt := &fakeRuntime{steps: []Execution{
		exits(3, "broke"), exits(3, "broke"), exits(3, "broke"),
	}}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 2), rt)

	if got.Phase != api.RunError {
		t.Fatalf("phase = %s, want Error once the budget is spent", got.Phase)
	}
	if rt.calls != 3 {
		t.Errorf("ran %d attempts, want 3 (initial + 2 retries)", rt.calls)
	}
	if got.ErrorRetries != 2 {
		t.Errorf("ErrorRetries = %d, want 2", got.ErrorRetries)
	}
}

func TestConformance_ARetryThatPassesClearsItsPredecessorsEvidence(t *testing.T) {
	// Violations, NotEvaluated and Unmeasurable are assigned unconditionally on
	// settle, from the SAME attempt as Metrics, so a retry that passes cannot
	// keep the gates its predecessor missed.
	//
	// Mirrors: the settle closure in completeAttempt.
	th := api.Threshold{Metric: "tflops", Comparison: api.GTE, Value: "100"}
	rt := &fakeRuntime{steps: []Execution{
		exits(3, "broke"),        // Error, retried
		exits(0, "tflops=101\n"), // passes cleanly
	}}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(th), Required: true}, 1), rt)

	if got.Phase != api.RunPassed {
		t.Fatalf("phase = %s, want Passed", got.Phase)
	}
	if len(got.Violations) != 0 {
		t.Errorf("a passing retry kept violations from an earlier attempt: %+v", got.Violations)
	}
	if got.Metrics["tflops"] != "101" {
		t.Errorf("metrics are not from the deciding attempt: %v", got.Metrics)
	}
}

func TestConformance_TriggersRecordWhyEachAttemptHappened(t *testing.T) {
	// So the retry rule is auditable from a stored result long after the run.
	//
	// Mirrors: triggerFor in the controller.
	rt := &fakeRuntime{steps: []Execution{exits(3, "broke"), exits(0, "ok=1\n")}}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 1), rt)

	if len(got.Attempts) != 2 {
		t.Fatalf("got %d attempts, want 2", len(got.Attempts))
	}
	if got.Attempts[0].Trigger != api.AttemptInitial {
		t.Errorf("first attempt trigger = %s, want Initial", got.Attempts[0].Trigger)
	}
	if got.Attempts[1].Trigger != api.AttemptErrorRetry {
		t.Errorf("second attempt trigger = %s, want ErrorRetry", got.Attempts[1].Trigger)
	}
}

// ─── Run-level rules ─────────────────────────────────────────────────────────

func TestConformance_FailFastStopsAndDoesNotInventPhases(t *testing.T) {
	// A test that did not execute has no verdict. Recording it as Skipped would
	// claim something about hardware nobody looked at.
	rt := &fakeRuntime{steps: []Execution{exits(1, "MARKER_FAIL")}}
	p := Plan{
		Node:     "n1",
		FailFast: true,
		Tests: []PlannedTest{
			{Name: "first", Spec: testSpec(), Required: true},
			{Name: "second", Spec: testSpec(), Required: true},
		},
	}
	clock := time.Now()
	rep, err := runWithClock(context.Background(), p, rt, Hooks{}, func() time.Time { return clock })
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	if len(rep.Results) != 1 {
		t.Fatalf("got %d results, want only the test that ran", len(rep.Results))
	}
	if rt.calls != 1 {
		t.Errorf("ran %d containers, want 1", rt.calls)
	}
}

func TestConformance_TheRunPhasePrefersFailedThenError(t *testing.T) {
	// Failed outranks Error, which outranks Passed. Skips count toward neither:
	// a run whose tests all skipped judged nothing, and calling that Passed
	// would certify hardware the suite never measured.
	//
	// Mirrors: the controller's finalize precedence.
	rows := []struct {
		name    string
		results []TestResult
		want    api.RunPhase
	}{
		{"a failure decides", []TestResult{{Phase: api.RunPassed}, {Phase: api.RunError}, {Phase: api.RunFailed}}, api.RunFailed},
		{"an error outranks a pass", []TestResult{{Phase: api.RunPassed}, {Phase: api.RunError}}, api.RunError},
		{"all passed", []TestResult{{Phase: api.RunPassed}}, api.RunPassed},
		{"all skipped is not a pass", []TestResult{{Phase: api.RunSkipped}, {Phase: api.RunSkipped}}, api.RunSkipped},
	}
	for _, r := range rows {
		if got := finalPhase(r.results); got != r.want {
			t.Errorf("%s: finalPhase = %s, want %s", r.name, got, r.want)
		}
	}
}

func TestConformance_AnUnresolvableImageIsAConfigurationError(t *testing.T) {
	// The same phase the operator gives a kind with no default image: a
	// configuration fault, not a hardware verdict.
	spec := api.BurnInTestSpec{Kind: api.TestKind("nobody-registered-this")}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: spec, Required: true}, 0), &fakeRuntime{})

	if got.Phase != api.RunError {
		t.Fatalf("phase = %s, want Error", got.Phase)
	}
	if !strings.Contains(got.Message, "spec.runner.image") {
		t.Errorf("the message should say what to set: %q", got.Message)
	}
}

func TestHooksFireInOrder(t *testing.T) {
	// OnTestComplete is what lets an embedding agent publish results as they
	// happen rather than accumulating them, which is what keeps a long profile
	// inside a dispatch ceiling.
	var events []string
	rt := &fakeRuntime{steps: []Execution{exits(0, "ok=1\n"), exits(0, "ok=1\n")}}
	p := Plan{Node: "n1", Tests: []PlannedTest{
		{Name: "a", Spec: testSpec(), Required: true},
		{Name: "b", Spec: testSpec(), Required: true},
	}}
	clock := time.Now()
	_, err := runWithClock(context.Background(), p, rt, Hooks{
		OnTestStart:    func(name string) { events = append(events, "start:"+name) },
		OnTestComplete: func(r TestResult) { events = append(events, "done:"+r.Name+"="+string(r.Phase)) },
	}, func() time.Time { return clock })
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	want := []string{"start:a", "done:a=Passed", "start:b", "done:b=Passed"}
	if strings.Join(events, ",") != strings.Join(want, ",") {
		t.Errorf("hook order = %v, want %v", events, want)
	}
}

// THE TWO DISPATCHERS RESOLVE THE SAME IMAGE, and this is the guard that makes
// the drift impossible to land again rather than merely unlikely.
//
// pkg/runnerimages exists because a bare-metal path runs these same images on
// hosts that are not cluster members, and its own package comment records that
// this solution ALREADY has one instance of two implementations running a
// different test under the same TestKind name. `imagesByVendor` reintroduced it:
// the operator learned to resolve per vendor and this package's resolveImage was
// left on the old three-line ladder, so a host the operator would have sent to a
// ROCm image ran the NVIDIA default instead — silently, and reported under the
// same kind.
//
// Asserting "both call Resolve" would be a test of the source text. This asserts
// the ANSWERS agree, over the cases where they could differ.
func TestConformance_ImageResolutionAgreesWithTheOperator(t *testing.T) {
	rocm := api.VendorImage{Vendor: "amd", Image: "example.invalid/rocm:v1"}

	for _, tc := range []struct {
		name   string
		spec   api.BurnInTestSpec
		vendor string
	}{
		{name: "explicit image", vendor: "amd", spec: api.BurnInTestSpec{
			Kind: api.KindMemoryBW, Runner: &api.RunnerSpec{Image: "example.invalid/pinned:v9"}}},
		{name: "the vendor's own entry", vendor: "amd", spec: api.BurnInTestSpec{
			Kind: api.KindMemoryBW, Runner: &api.RunnerSpec{ImagesByVendor: []api.VendorImage{rocm}}}},
		{name: "an unlisted vendor the default can serve", vendor: "nvidia", spec: api.BurnInTestSpec{
			Kind: api.KindMemoryBW, Runner: &api.RunnerSpec{ImagesByVendor: []api.VendorImage{rocm}}}},
		{name: "an unlisted vendor the default cannot serve", vendor: "amd", spec: api.BurnInTestSpec{
			Kind: api.KindMemoryBW, Runner: &api.RunnerSpec{ImagesByVendor: []api.VendorImage{
				{Vendor: "intel", Image: "example.invalid/xe:v1"}}}}},
		{name: "no map at all on an amd node", vendor: "amd", spec: api.BurnInTestSpec{
			Kind: api.KindMemoryBW}},
		{name: "a vendor-neutral kind", vendor: "amd", spec: api.BurnInTestSpec{
			Kind: api.KindIBWriteBW}},
		{name: "an unknown vendor", vendor: "", spec: api.BurnInTestSpec{
			Kind: api.KindMemoryBW}},
		{name: "a kind with no default", vendor: "nvidia", spec: api.BurnInTestSpec{
			Kind: api.KindCustom}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The operator's answer, through the same function podForTest calls.
			wantImg, wantErr := runnerimages.Resolve(tc.spec.Kind, tc.spec.Runner, tc.vendor)

			// This dispatcher's answer, through Translate — the real path, not a
			// helper, so a Translate that stopped consulting the vendor would be
			// caught here even if resolveImage itself were right.
			got, err := Translate(
				Plan{Node: "host-1", Vendor: tc.vendor},
				PlannedTest{Name: "t", Spec: tc.spec},
			)

			if (wantErr == nil) != (err == nil) {
				t.Fatalf("operator err=%v, bare-metal err=%v — the two dispatchers disagree about "+
					"whether this test can run at all", wantErr, err)
			}
			if wantErr != nil {
				return
			}
			if got.Image != wantImg {
				t.Errorf("bare-metal resolved %q, operator resolved %q — the same TestKind on the "+
					"same hardware is running two different images", got.Image, wantImg)
			}
		})
	}
}

// ─── Variants (mirrors expandVariants + variantEnv, both now pkg/plan) ───────
//
// The row: A VARIANT CELL IS AN ORDINARY EXECUTION THAT CARRIES LABELS. Its
// axes reach the runner as BURNIN_VARIANT_<AXIS> and are echoed onto the
// result; nothing in this engine interprets them, and no verdict depends on
// them. That is the same statement the controller makes — ensureResult copies
// t.Axes onto TestResult.VariantAxes and podForTest injects variantEnv — and
// it has to hold on both sides, because pkg/contract tells a consumer to group
// a sweep's cells BY these labels.
//
// This dispatcher used to fail the row completely: cmd/burnin discarded
// spec.tests[].variants before ever building a Plan, so a four-cell sweep ran
// once with no labels at all.
func TestConformance_AVariantCellsAxesAreCarriedAndNeverInterpreted(t *testing.T) {
	rt := &fakeRuntime{steps: []Execution{exits(0, "achievedTflops=812.5\n")}}
	cell := PlannedTest{
		Name:     "gemm-fp4",
		Spec:     testSpec(),
		Required: true,
		Axes:     map[string]string{"precision": "fp4", "class": "smoke"},
		Parent:   "gemm",
	}
	got := runOne(t, onePlan(cell, 0), rt)

	// Echoed onto the result, for the consumer that groups the sweep.
	if got.VariantAxes["precision"] != "fp4" || got.VariantAxes["class"] != "smoke" {
		t.Errorf("VariantAxes = %v, want the cell's own labels; a result carrying the cell's NAME but not "+
			"its labels makes a four-cell sweep four results nothing can group", got.VariantAxes)
	}
	// Delivered to the runner, which is the half that changes what is measured.
	if len(rt.specs) != 1 {
		t.Fatalf("got %d executions, want 1", len(rt.specs))
	}
	if v := rt.specs[0].Env["BURNIN_VARIANT_PRECISION"]; v != "fp4" {
		t.Errorf("BURNIN_VARIANT_PRECISION = %q, want fp4 — a cell that never receives its axis runs the "+
			"DEFAULT configuration while being reported as the fp4 cell", v)
	}

	// And interpreted by nothing: the verdict is the exit code's, exactly as it
	// would be for the same test with no variants at all.
	if got.Phase != api.RunPassed {
		t.Errorf("phase = %v, want Passed — an axis must never reach a verdict", got.Phase)
	}
}

// The result's axes are a COPY. A consumer holding a Report must not be able to
// reach back into the plan the run was executed from.
func TestConformance_AResultsAxesDoNotAliasThePlan(t *testing.T) {
	rt := &fakeRuntime{steps: []Execution{exits(0, "")}}
	axes := map[string]string{"precision": "fp4"}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true, Axes: axes}, 0), rt)

	got.VariantAxes["precision"] = "mutated"
	if axes["precision"] != "fp4" {
		t.Error("TestResult.VariantAxes aliases the plan's map")
	}
}

// A test with no variants carries NO axes — not an empty map. An empty map
// delivers as an empty object in the envelope, which reads as "this cell had
// labels and they were blank" rather than "this test was never expanded".
func TestConformance_ATestWithNoVariantsCarriesNoAxes(t *testing.T) {
	rt := &fakeRuntime{steps: []Execution{exits(0, "")}}
	got := runOne(t, onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 0), rt)

	if got.VariantAxes != nil {
		t.Errorf("VariantAxes = %v, want nil", got.VariantAxes)
	}
	for k := range rt.specs[0].Env {
		if strings.HasPrefix(k, "BURNIN_VARIANT_") {
			t.Errorf("an unexpanded test received %s", k)
		}
	}
}

// ─── valueFrom environment (mirrors podForTest's verbatim env passthrough) ───
//
// The row: A VARIABLE THIS DISPATCHER CANNOT RESOLVE IS UNSET, NEVER EMPTY.
//
// In a cluster, spec.runner.env survives podForTest verbatim and the kubelet
// resolves valueFrom. Here there is no kubelet, and this engine copied e.Value
// — which on a valueFrom entry is the EMPTY STRING. So the variable was set,
// and set to nothing: a runner testing `if [ -n "$HOST_IP" ]` took the wrong
// branch and one that used it built ":9000". The same BurnInTest worked
// in-cluster, which is exactly the drift the parity ledger exists to catch.
//
// This is the same rule the runners emit metrics under, applied one layer up:
// absence is not a declaration, and a value nobody established must never be
// presented as one.
func TestConformance_AnUnresolvableValueFromIsUnsetNeverEmpty(t *testing.T) {
	spec := testSpec()
	spec.Runner.Env = []corev1.EnvVar{
		{Name: "FROM_SECRET", ValueFrom: &corev1.EnvVarSource{
			SecretKeyRef: &corev1.SecretKeySelector{
				LocalObjectReference: corev1.LocalObjectReference{Name: "creds"}, Key: "token"},
		}},
		{Name: "FROM_POD_FIELD", ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"},
		}},
		{Name: "LITERAL", Value: "kept"},
	}

	p := Plan{Node: "n1", HostIP: "10.0.0.5", Tests: []PlannedTest{{Name: "t", Spec: spec, Required: true}}}
	got, err := Translate(p, p.Tests[0])
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}

	for _, name := range []string{"FROM_SECRET", "FROM_POD_FIELD"} {
		if v, present := got.Env[name]; present {
			t.Errorf("%s is present with value %q; it must be ABSENT, so a runner can tell "+
				"\"nobody could answer this\" from \"the cluster says it is blank\"", name, v)
		}
	}
	if got.Env["LITERAL"] != "kept" {
		t.Errorf("a plain value must still be passed through, got %q", got.Env["LITERAL"])
	}

	// And the omission is ANNOUNCED, not discovered by a runner.
	warned := UnresolvableEnv(p, spec)
	if len(warned) != 2 {
		t.Fatalf("UnresolvableEnv = %v, want both unresolvable variables named", warned)
	}
	if !strings.Contains(strings.Join(warned, " "), "secretKeyRef creds/token") {
		t.Errorf("the warning must name what the profile wrote: %v", warned)
	}
}

// The two field paths that name the MACHINE are answered, because this
// dispatcher is standing on the machine. status.hostIP is the documented way a
// runner learns the address its peers reach it on.
func TestConformance_TheMachinesOwnFieldRefsAreResolved(t *testing.T) {
	spec := testSpec()
	spec.Runner.Env = []corev1.EnvVar{
		{Name: "HOST_IP", ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.hostIP"}}},
		{Name: "NODE", ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"}}},
	}

	p := Plan{Node: "spark-a", HostIP: "10.0.0.5", Tests: []PlannedTest{{Name: "t", Spec: spec, Required: true}}}
	got, err := Translate(p, p.Tests[0])
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if got.Env["HOST_IP"] != "10.0.0.5" {
		t.Errorf("HOST_IP = %q, want the host's primary address", got.Env["HOST_IP"])
	}
	if got.Env["NODE"] != "spark-a" {
		t.Errorf("NODE = %q, want the run's node name", got.Env["NODE"])
	}
	if w := UnresolvableEnv(p, spec); len(w) != 0 {
		t.Errorf("nothing should be warned about here, got %v", w)
	}
}

// A test that asks for status.hostIP on a host with no routable address is
// REFUSED, not silently left unset. The dispatcher is standing on the machine;
// if it cannot answer, the test cannot do what it was written to do, and that
// is a plan-time refusal naming the variable rather than a hole discovered by a
// runner hours later.
func TestConformance_AskingForHostIPWithNoAddressIsRefused(t *testing.T) {
	spec := testSpec()
	spec.Runner.Env = []corev1.EnvVar{{Name: "HOST_IP", ValueFrom: &corev1.EnvVarSource{
		FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.hostIP"}}}}

	p := Plan{Node: "n1", Tests: []PlannedTest{{Name: "t", Spec: spec, Required: true}}}
	_, err := Translate(p, p.Tests[0])
	if err == nil {
		t.Fatal("translation succeeded with no host address to answer status.hostIP with")
	}
	if !strings.Contains(err.Error(), "HOST_IP") || !strings.Contains(err.Error(), "status.hostIP") {
		t.Errorf("the refusal must name the variable and the field path: %v", err)
	}
	// A refusal is not a warning: it is raised, not listed.
	if w := UnresolvableEnv(p, spec); len(w) != 0 {
		t.Errorf("a refusal must not also be warned about, got %v", w)
	}
}

// A test cannot smuggle a contract variable in through valueFrom either — the
// reserved check comes first, exactly as it does for a literal value.
func TestConformance_ReservedVariablesAreRefusedThroughValueFromToo(t *testing.T) {
	spec := testSpec()
	spec.Scope = api.ScopePair
	spec.Runner.Env = []corev1.EnvVar{{Name: "BURNIN_ROLE", ValueFrom: &corev1.EnvVarSource{
		FieldRef: &corev1.ObjectFieldSelector{FieldPath: "spec.nodeName"}}}}

	p := Plan{
		Node: "n1", HostIP: "10.0.0.5",
		Rendezvous: &Rendezvous{Role: RoleServer},
		Tests:      []PlannedTest{{Name: "t", Spec: spec, Required: true}},
	}
	got, err := Translate(p, p.Tests[0])
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if got.Env["BURNIN_ROLE"] != RoleServer {
		t.Errorf("BURNIN_ROLE = %q, want %q — a profile that could set it could make both ends the client",
			got.Env["BURNIN_ROLE"], RoleServer)
	}
}

// ─── Checkpoints (mirrors the reconciler's checkpoint) ───────────────────────
//
// The row: A CHECKPOINT IS EVIDENCE, NEVER A VERDICT.
//
// A long soak that is cancelled, killed at its deadline, or lost to a reboot at
// minute 200 otherwise reports NOTHING AT ALL, because a runner's metrics only
// reach the report when the container exits. The operator has published these
// since checkpointIntervalSeconds existed; this engine buffered stdout until
// exit and so could not.
//
// What must NOT follow is a checkpoint reaching the verdict. Thresholds are
// evaluated once, at the end, against the completed execution — a mid-run
// sample that dips below a floor is not a failure, because the run is not over,
// and a dispatcher that judged one would condemn hardware for a moment the test
// was written to expect.

// streamingFake is a fakeRuntime that emits its stdout in pieces, so the
// engine's checkpoint path is exercised without a container runtime.
type streamingFake struct {
	fakeRuntime
	chunks []string
}

func (f *streamingFake) RunStreaming(ctx context.Context, spec RunSpec, onOutput func(string)) (Execution, error) {
	f.specs = append(f.specs, spec)
	var sofar strings.Builder
	for _, c := range f.chunks {
		sofar.WriteString(c)
		if onOutput != nil {
			onOutput(sofar.String())
		}
	}
	if f.calls >= len(f.steps) {
		return Execution{}, fmt.Errorf("streaming fake: unexpected call %d", f.calls+1)
	}
	e := f.steps[f.calls]
	f.calls++
	return e, nil
}

func checkpointed(spec api.BurnInTestSpec, seconds int32) api.BurnInTestSpec {
	spec.CheckpointIntervalSeconds = &seconds
	return spec
}

func TestConformance_ACheckpointIsEvidenceAndNeverReachesTheVerdict(t *testing.T) {
	// A floor the MID-RUN sample violates and the FINAL metrics satisfy. If a
	// checkpoint could reach the verdict, this would settle Failed.
	th := api.Threshold{Metric: "bandwidthGbps", Comparison: api.GTE, Value: "90"}
	spec := checkpointed(testSpec(th), 1)

	rt := &streamingFake{
		fakeRuntime: fakeRuntime{steps: []Execution{exits(0, "bandwidthGbps=97.2\n")}},
		chunks:      []string{"bandwidthGbps=12.0\n", "bandwidthGbps=97.2\n"},
	}
	var seen []Checkpoint
	p := onePlan(PlannedTest{Name: "t", Spec: spec, Required: true}, 0)

	clock := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	rep, err := runWithClock(context.Background(), p, rt, Hooks{
		OnCheckpoint: func(c Checkpoint) { seen = append(seen, c) },
	}, func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := rep.Results[0]
	if got.Phase != api.RunPassed {
		t.Fatalf("phase = %v (%s), want Passed — the FINAL metrics satisfy the floor, and a mid-run "+
			"sample that dips below it is not a failure because the run is not over", got.Phase, got.Message)
	}
	if len(got.Violations) != 0 {
		t.Errorf("violations = %+v, want none", got.Violations)
	}
	if got.Metrics["bandwidthGbps"] != "97.2" {
		t.Errorf("the result's metrics = %v, want the final execution's, not a checkpoint's", got.Metrics)
	}

	// And the evidence was published along the way.
	if len(seen) == 0 {
		t.Fatal("no checkpoint was published; a soak killed mid-run would report nothing at all")
	}
	if seen[0].Metrics["bandwidthGbps"] != "12.0" {
		t.Errorf("first checkpoint = %v, want the metrics as they stood at that moment", seen[0].Metrics)
	}
	if seen[0].Test != "t" {
		t.Errorf("checkpoint names test %q, want t", seen[0].Test)
	}
}

// No interval means no checkpoints, so a profile written before the field
// existed behaves exactly as it did.
func TestConformance_NoCheckpointIntervalPublishesNothing(t *testing.T) {
	rt := &streamingFake{
		fakeRuntime: fakeRuntime{steps: []Execution{exits(0, "bandwidthGbps=97.2\n")}},
		chunks:      []string{"bandwidthGbps=12.0\n", "bandwidthGbps=97.2\n"},
	}
	var seen []Checkpoint
	p := onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 0)
	clock := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	rep, err := runWithClock(context.Background(), p, rt, Hooks{
		OnCheckpoint: func(c Checkpoint) { seen = append(seen, c) },
	}, func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Results[0].Phase != api.RunPassed {
		t.Errorf("phase = %v, want Passed", rep.Results[0].Phase)
	}
	if len(seen) != 0 {
		t.Errorf("got %d checkpoints for a test that asked for none: %+v", len(seen), seen)
	}
}

// A runtime that cannot stream degrades the EVIDENCE and never the verdict: the
// test runs exactly as before and simply publishes nothing.
func TestConformance_ANonStreamingRuntimeStillProducesTheSameVerdict(t *testing.T) {
	spec := checkpointed(testSpec(), 1)
	rt := &fakeRuntime{steps: []Execution{exits(0, "bandwidthGbps=97.2\n")}}

	published := 0
	p := onePlan(PlannedTest{Name: "t", Spec: spec, Required: true}, 0)
	clock := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	rep, err := runWithClock(context.Background(), p, rt, Hooks{
		OnCheckpoint: func(Checkpoint) { published++ },
	}, func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if rep.Results[0].Phase != api.RunPassed {
		t.Errorf("phase = %v, want Passed", rep.Results[0].Phase)
	}
	if published != 0 {
		t.Errorf("a runtime that cannot stream published %d checkpoints", published)
	}
}

// A checkpoint with no metrics yet is NOT published — an envelope carrying an
// empty metric set is not evidence, and it would reset the receiver's view of a
// soak that had already reported real numbers.
func TestConformance_ACheckpointWithNoMetricsIsNotPublished(t *testing.T) {
	spec := checkpointed(testSpec(), 1)
	rt := &streamingFake{
		fakeRuntime: fakeRuntime{steps: []Execution{exits(0, "starting\nbandwidthGbps=97.2\n")}},
		chunks:      []string{"starting up, no metrics yet\n", "bandwidthGbps=97.2\n"},
	}
	var seen []Checkpoint
	p := onePlan(PlannedTest{Name: "t", Spec: spec, Required: true}, 0)
	clock := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	if _, err := runWithClock(context.Background(), p, rt, Hooks{
		OnCheckpoint: func(c Checkpoint) { seen = append(seen, c) },
	}, func() time.Time {
		clock = clock.Add(time.Second)
		return clock
	}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	for _, c := range seen {
		if len(c.Metrics) == 0 {
			t.Error("a checkpoint was published with no metrics")
		}
	}
	if len(seen) != 1 || seen[0].Metrics["bandwidthGbps"] != "97.2" {
		t.Errorf("checkpoints = %+v, want exactly the one that had something to say", seen)
	}
}

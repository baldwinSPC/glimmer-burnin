package controller

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Variants: one profile entry, many executions — issue #155.
//
// The whole feature is that expansion happens ONCE, before buildPlan, so nothing
// downstream ever learns variants exist. These tests are mostly about that: they
// check the expansion, and then they check that the rest of the reconciler is
// unchanged by it.

func variantSpec(kind burninv1alpha1.TestKind) burninv1alpha1.BurnInTestSpec {
	return burninv1alpha1.BurnInTestSpec{
		Kind:            kind,
		Scope:           burninv1alpha1.ScopeNode,
		DurationSeconds: 60,
		Runner:          &burninv1alpha1.RunnerSpec{Image: "example.com/gemm:v1"},
		Thresholds: []burninv1alpha1.Threshold{
			{Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"},
		},
	}
}

// Four precision variants plan four tests with distinct names, distinct
// thresholds, and their own environment.
func TestFourVariantsPlanFourDistinctExecutions(t *testing.T) {
	spec := variantSpec(burninv1alpha1.KindComputeSmoke)
	i64 := func(v int32) *int32 { return &v }

	got := expandVariants("gemm", spec, true, []burninv1alpha1.TestVariant{
		{Name: "fp4", Axes: map[string]string{"precision": "fp4"},
			Thresholds: []burninv1alpha1.Threshold{
				{Metric: "achievedTflops", Comparison: burninv1alpha1.GTE, Value: "700"}}},
		{Name: "fp8", Axes: map[string]string{"precision": "fp8"},
			Thresholds: []burninv1alpha1.Threshold{
				{Metric: "achievedTflops", Comparison: burninv1alpha1.GTE, Value: "350"}}},
		{Name: "bf16", Axes: map[string]string{"precision": "bf16"}},
		{Name: "fp64", Axes: map[string]string{"precision": "fp64"},
			DurationSeconds: i64(300)},
	})

	if len(got) != 4 {
		t.Fatalf("got %d executions from 4 variants, want 4", len(got))
	}
	names := map[string]bool{}
	for _, g := range got {
		if names[g.name] {
			t.Errorf("duplicate planned name %q — test names are result identity", g.name)
		}
		names[g.name] = true
	}
	for _, want := range []string{"gemm-fp4", "gemm-fp8", "gemm-bf16", "gemm-fp64"} {
		if !names[want] {
			t.Errorf("no execution named %q; got %v", want, names)
		}
	}

	// Thresholds REPLACE. The parent's nonfiniteCount gate is gone from a
	// variant that set its own — merging would silently retain a gate the
	// author believed replaced, and a node failed against a threshold nobody
	// can find in the profile has a verdict with nothing to explain it.
	fp4 := byName(t, got, "gemm-fp4")
	if len(fp4.spec.Thresholds) != 1 || fp4.spec.Thresholds[0].Metric != "achievedTflops" {
		t.Errorf("fp4 thresholds = %+v; an overlay REPLACES, it does not merge",
			fp4.spec.Thresholds)
	}
	if fp4.spec.Thresholds[0].Value != "700" {
		t.Errorf("fp4 floor = %q, want 700", fp4.spec.Thresholds[0].Value)
	}
	// A different cell's floor is a different number about different silicon.
	if v := byName(t, got, "gemm-fp8").spec.Thresholds[0].Value; v != "350" {
		t.Errorf("fp8 floor = %q, want 350 — the cells share a threshold list", v)
	}

	// A variant that set NO thresholds inherits the parent's.
	if bf := byName(t, got, "gemm-bf16"); len(bf.spec.Thresholds) != 1 ||
		bf.spec.Thresholds[0].Metric != "nonfiniteCount" {
		t.Errorf("bf16 thresholds = %+v, want the parent's inherited", bf.spec.Thresholds)
	}

	// Overlays apply, and do not leak between cells.
	if d := byName(t, got, "gemm-fp64").spec.DurationSeconds; d != 300 {
		t.Errorf("fp64 duration = %d, want 300", d)
	}
	if d := byName(t, got, "gemm-fp4").spec.DurationSeconds; d != 60 {
		t.Errorf("fp4 duration = %d, want the parent's 60 — an overlay leaked between "+
			"cells, so every cell after the first measures the wrong thing", d)
	}
	// And the parent spec itself is untouched.
	if spec.DurationSeconds != 60 || len(spec.Thresholds) != 1 {
		t.Errorf("expandVariants mutated the spec it was given: %+v", spec)
	}
}

// An empty (non-nil) threshold list means NO thresholds, which is a thing a
// variant may legitimately want.
func TestAnEmptyThresholdOverlayClearsTheParentGates(t *testing.T) {
	got := expandVariants("gemm", variantSpec(burninv1alpha1.KindComputeSmoke), true,
		[]burninv1alpha1.TestVariant{
			{Name: "explore", Thresholds: []burninv1alpha1.Threshold{}},
		})
	if n := len(got[0].spec.Thresholds); n != 0 {
		t.Errorf("an empty overlay left %d thresholds; nil inherits and empty clears, "+
			"and a variant that means to gate nothing has no other way to say so", n)
	}
}

// A profile with no variants plans exactly as it did before variants existed.
func TestNoVariantsPlansIdentically(t *testing.T) {
	spec := variantSpec(burninv1alpha1.KindComputeSmoke)
	got := expandVariants("gemm", spec, true, nil)

	if len(got) != 1 {
		t.Fatalf("got %d executions with no variants, want 1", len(got))
	}
	if got[0].name != "gemm" {
		t.Errorf("name = %q, want the bare test name — a suffix would change result "+
			"identity for every profile written before variants existed", got[0].name)
	}
	if got[0].axes != nil {
		t.Errorf("axes = %v, want nil", got[0].axes)
	}
	if !got[0].required {
		t.Error("required was lost")
	}
}

// Two variants with the same name are refused, by the same check that refuses
// two tests with the same name — the failure is identical.
func TestDuplicateVariantNamesAreRefusedAtPlanTime(t *testing.T) {
	tests := expandVariants("gemm", variantSpec(burninv1alpha1.KindComputeSmoke), true,
		[]burninv1alpha1.TestVariant{{Name: "fp4"}, {Name: "fp4"}})

	profile := &burninv1alpha1.BurnInProfile{}
	_, err := buildPlan(profile, tests, []string{"spark-a"}, 1, false)
	if err == nil {
		t.Fatal("two variants named the same planned cleanly. The second would never " +
			"run and its verdict would silently overwrite the first's.")
	}
	if !strings.Contains(err.Error(), "gemm-fp4") {
		t.Errorf("error = %v, want it to name the colliding execution", err)
	}
}

// The axes reach the runner as BURNIN_VARIANT_<AXIS>, and nothing else reads
// them.
func TestAxesReachTheRunnerAndNothingInterpretsThem(t *testing.T) {
	spec := variantSpec(burninv1alpha1.KindComputeSmoke)
	pod, err := podForTest(newRun("r", "p", "spark-a"), 0, 1, "gemm-fp4", &spec,
		map[string]string{"precision": "fp4", "layout": "nt"},
		"spark-a", burninv1alpha1.TargetSelector{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	env := map[string]string{}
	for _, e := range pod.Spec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["BURNIN_VARIANT_PRECISION"] != "fp4" {
		t.Errorf("BURNIN_VARIANT_PRECISION = %q, want fp4 (env: %v)",
			env["BURNIN_VARIANT_PRECISION"], env)
	}
	if env["BURNIN_VARIANT_LAYOUT"] != "nt" {
		t.Errorf("BURNIN_VARIANT_LAYOUT = %q, want nt", env["BURNIN_VARIANT_LAYOUT"])
	}
	// The rest of the contract is untouched.
	if env["BURNIN_DURATION_SECONDS"] == "" || env["BURNIN_ATTEMPT"] == "" {
		t.Error("variant injection displaced the existing environment contract")
	}
}

// The pod spec is deterministic: two identical plans produce identical pods.
//
// An unsorted map would make a diff of a pod against itself a diff nobody can
// act on.
func TestVariantEnvIsDeterministic(t *testing.T) {
	axes := map[string]string{"zebra": "1", "alpha": "2", "middle": "3"}
	first := variantEnv(axes)
	for i := 0; i < 20; i++ {
		got := variantEnv(axes)
		for j := range got {
			if got[j] != first[j] {
				t.Fatalf("variantEnv is not deterministic: %v vs %v", first, got)
			}
		}
	}
	if first[0].Name != "BURNIN_VARIANT_ALPHA" {
		t.Errorf("not sorted by axis name: %v", first)
	}
}

// An axis name that is not usable as an environment variable is SKIPPED, never
// mangled.
//
// Mangling would collide two axes onto one variable, and the runner would read
// one cell's value while reporting the other's — the same class of error as
// rewriting an artifact name into a colliding ConfigMap key.
func TestAnUnusableAxisNameIsSkippedNotMangled(t *testing.T) {
	got := variantEnv(map[string]string{
		"message.size": "8",   // a dot would have to become an underscore
		"message-size": "16",  // so would a hyphen — and they would collide
		"9lives":       "yes", // cannot lead with a digit
		"good":         "ok",
	})
	if len(got) != 1 || got[0].Name != "BURNIN_VARIANT_GOOD" {
		t.Fatalf("got %+v, want only the usable axis. A mangled name collides two axes "+
			"onto one variable and the runner reads the wrong cell's value.", got)
	}
}

// The axes are echoed onto the result, so a report can group a sweep's cells
// without parsing them back out of names.
func TestAxesAreEchoedOntoTheResult(t *testing.T) {
	run := newRun("r", "p", "spark-a")
	tst := plannedTest{
		Name: "gemm-fp4", Required: true,
		Spec: variantSpec(burninv1alpha1.KindComputeSmoke),
		Axes: map[string]string{"precision": "fp4"},
	}
	res := ensureResult(run, tst, []string{"spark-a"})
	if res.VariantAxes["precision"] != "fp4" {
		t.Fatalf("VariantAxes = %v; a consumer would have to recover the precision by "+
			"splitting the name on a hyphen, which works until a variant is called "+
			"\"fp8-dense\"", res.VariantAxes)
	}
}

// Editing the profile mid-run changes nothing in flight: the axes ride in the
// PINNED plan like the rest of the spec.
func TestAxesRideInThePinnedPlan(t *testing.T) {
	tests := expandVariants("gemm", variantSpec(burninv1alpha1.KindComputeSmoke), true,
		[]burninv1alpha1.TestVariant{{Name: "fp4", Axes: map[string]string{"precision": "fp4"}}})
	p, err := buildPlan(&burninv1alpha1.BurnInProfile{}, tests, []string{"spark-a"}, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if p.Tests[0].Axes["precision"] != "fp4" {
		t.Fatalf("the plan does not carry the axes: %+v", p.Tests[0])
	}
	// Mutating the source afterwards cannot reach the plan.
	tests[0].axes["precision"] = "fp64"
	if p.Tests[0].Axes["precision"] != "fp4" {
		t.Error("the plan shares an axes map with its input, so editing the profile " +
			"mid-run would change what an in-flight cell reports it was")
	}
}

// The expanded count is stated, because the arithmetic is invisible in the
// profile and the usual way to find it out is to wait.
func TestTheExpandedCountIsReported(t *testing.T) {
	tests := expandVariants("gemm", variantSpec(burninv1alpha1.KindComputeSmoke), true,
		[]burninv1alpha1.TestVariant{{Name: "fp4"}, {Name: "fp8"}, {Name: "bf16"}, {Name: "fp64"}})
	profile := &burninv1alpha1.BurnInProfile{
		Spec: burninv1alpha1.BurnInProfileSpec{
			Tests: []burninv1alpha1.ProfileTest{{TestRef: "gemm"}},
		},
	}
	p, err := buildPlan(profile, tests, []string{"a", "b", "c", "d", "e", "f", "g", "h"}, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	sum := p.variantSummary()
	if sum.Entries != 1 || sum.Executions != 4 {
		t.Fatalf("summary = %+v, want 1 entry expanding to 4 executions", sum)
	}
	if len(sum.Expanded) != 1 || !strings.Contains(sum.Expanded[0], "4 cells") {
		t.Errorf("Expanded = %v, want it to name gemm and its cell count", sum.Expanded)
	}
	// A run with no variants says nothing at all: a condition that always
	// appears is a condition nobody reads.
	plain, err := buildPlan(profile,
		expandVariants("gemm", variantSpec(burninv1alpha1.KindComputeSmoke), true, nil),
		[]string{"a"}, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(plain.variantSummary().Expanded) != 0 {
		t.Error("a run with no variants produced an expansion advisory")
	}
}

// Env overlays replace rather than append, like every other overlay.
func TestAnEnvOverlayReplaces(t *testing.T) {
	spec := variantSpec(burninv1alpha1.KindComputeSmoke)
	spec.Runner.Env = []corev1.EnvVar{{Name: "PARENT", Value: "1"}}
	got := expandVariants("gemm", spec, true, []burninv1alpha1.TestVariant{
		{Name: "cell", Env: []corev1.EnvVar{{Name: "CELL", Value: "2"}}},
	})
	env := got[0].spec.Runner.Env
	if len(env) != 1 || env[0].Name != "CELL" {
		t.Errorf("env = %+v; an overlay REPLACES, so the parent's PARENT is gone", env)
	}
	if spec.Runner.Env[0].Name != "PARENT" {
		t.Error("the parent spec's env was mutated")
	}
}

func byName(t *testing.T, in []resolvedTest, name string) resolvedTest {
	t.Helper()
	for _, r := range in {
		if r.name == name {
			return r
		}
	}
	t.Fatalf("no execution named %q", name)
	return resolvedTest{}
}

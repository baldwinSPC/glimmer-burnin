// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package plan

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

func ptr[T any](v T) *T { return &v }

// The property that made this package worth extracting: a profile entry with
// no variants plans EXACTLY as it did before variants existed. Both
// dispatchers call this, so a regression here changes what every profile
// written before the feature means.
func TestExpandVariants_NoVariantsIsOneTestUnchanged(t *testing.T) {
	spec := api.BurnInTestSpec{Kind: api.KindComputeSmoke, DurationSeconds: 120}
	got := ExpandVariants("gemm", spec, true, nil)

	if len(got) != 1 {
		t.Fatalf("got %d tests, want exactly 1", len(got))
	}
	if got[0].Name != "gemm" {
		t.Errorf("name = %q, want the entry's own name unsuffixed — a test that gained a suffix would "+
			"change result identity for every profile written before variants existed", got[0].Name)
	}
	if got[0].Axes != nil {
		t.Errorf("Axes = %v, want nil: a test with no variants has no labels to group by, and an empty "+
			"map would deliver as one", got[0].Axes)
	}
	if got[0].Parent != "" {
		t.Errorf("Parent = %q, want empty — nothing expanded this entry", got[0].Parent)
	}
	if !got[0].Required {
		t.Error("Required did not survive")
	}
}

// An unset scope becomes Node HERE, before expansion, so every cell inherits
// it. The apiserver defaults it in a cluster; a suite file read off a laptop
// never went near an apiserver, and the two dispatchers must still agree about
// what the same document means.
func TestExpandVariants_DefaultsScopeBeforeExpanding(t *testing.T) {
	cells := ExpandVariants("t", api.BurnInTestSpec{Kind: api.KindCustom}, true,
		[]api.TestVariant{{Name: "a"}, {Name: "b"}})

	for _, c := range cells {
		if c.Spec.Scope != api.ScopeNode {
			t.Errorf("%s: scope = %q, want %q", c.Name, c.Spec.Scope, api.ScopeNode)
		}
	}
	if got := ExpandVariants("t", api.BurnInTestSpec{Kind: api.KindCustom}, true, nil); got[0].Spec.Scope != api.ScopeNode {
		t.Errorf("unexpanded scope = %q, want %q", got[0].Spec.Scope, api.ScopeNode)
	}
}

// Thresholds REPLACE, never merge — see TestVariant.Thresholds. A merged gate
// the author believed replaced is the worst kind of verdict this project can
// produce, because there is nothing in the profile to read that explains it.
func TestExpandVariants_ThresholdsReplaceAndNilInherits(t *testing.T) {
	parent := api.BurnInTestSpec{
		Kind:       api.KindComputeSmoke,
		Thresholds: []api.Threshold{{Metric: "nonfiniteCount", Comparison: api.EQ, Value: "0"}},
	}
	cells := ExpandVariants("gemm", parent, true, []api.TestVariant{
		{Name: "fp4", Thresholds: []api.Threshold{{Metric: "achievedTflops", Comparison: api.GTE, Value: "700"}}},
		{Name: "bf16"},
		{Name: "none", Thresholds: []api.Threshold{}},
	})

	if len(cells) != 3 {
		t.Fatalf("got %d cells, want 3", len(cells))
	}
	if m := cells[0].Spec.Thresholds; len(m) != 1 || m[0].Metric != "achievedTflops" {
		t.Errorf("fp4 thresholds = %+v, want ONLY its own — a retained parent gate is invisible in the profile", m)
	}
	if m := cells[1].Spec.Thresholds; len(m) != 1 || m[0].Metric != "nonfiniteCount" {
		t.Errorf("bf16 thresholds = %+v, want the parent's inherited (nil overlay inherits)", m)
	}
	if m := cells[2].Spec.Thresholds; len(m) != 0 {
		t.Errorf("an empty non-nil overlay means NO thresholds, got %+v", m)
	}
}

// The nil-Runner case that panicked the reconciler on the first profile
// anybody wrote: a built-in kind resolves its image from pkg/runnerimages, so
// its author has nothing to put in `runner:` — and overlaying env or args is
// the canonical use of the feature.
func TestExpandVariants_OverlayingEnvOnATestWithNoRunnerBlock(t *testing.T) {
	cells := ExpandVariants("t", api.BurnInTestSpec{Kind: api.KindComputeSmoke}, true, []api.TestVariant{
		{Name: "one", Env: []corev1.EnvVar{{Name: "X", Value: "1"}}},
		{Name: "two", Args: []string{"--flag"}},
	})

	if cells[0].Spec.Runner == nil || len(cells[0].Spec.Runner.Env) != 1 {
		t.Fatalf("env overlay did not create a runner block: %+v", cells[0].Spec.Runner)
	}
	if cells[1].Spec.Runner == nil || len(cells[1].Spec.Runner.Args) != 1 {
		t.Fatalf("args overlay did not create a runner block: %+v", cells[1].Spec.Runner)
	}
}

// Each cell is deep-copied, so an overlay cannot reach back into the parent
// and the SECOND cell cannot inherit the FIRST's edits — which would make a
// sweep silently measure the wrong thing in every cell after the first.
func TestExpandVariants_CellsDoNotShareState(t *testing.T) {
	parent := api.BurnInTestSpec{
		Kind:   api.KindComputeSmoke,
		Runner: &api.RunnerSpec{Env: []corev1.EnvVar{{Name: "SHARED", Value: "parent"}}},
	}
	cells := ExpandVariants("t", parent, true, []api.TestVariant{
		{Name: "a", DurationSeconds: ptr(int32(60))},
		{Name: "b"},
	})

	if cells[1].Spec.DurationSeconds == 60 {
		t.Error("cell b inherited cell a's duration overlay — the cells share state")
	}
	cells[0].Spec.Runner.Env[0].Value = "mutated"
	if parent.Runner.Env[0].Value != "parent" {
		t.Error("mutating a cell reached back into the parent spec")
	}
	if cells[1].Spec.Runner.Env[0].Value != "parent" {
		t.Error("mutating cell a reached into cell b")
	}
}

// Axes are COPIED onto the cell, never shared with the variant they came from.
func TestExpandVariants_AxesAreCopiedNotShared(t *testing.T) {
	v := api.TestVariant{Name: "fp4", Axes: map[string]string{"precision": "fp4"}}
	cells := ExpandVariants("gemm", api.BurnInTestSpec{Kind: api.KindComputeSmoke}, true, []api.TestVariant{v})

	cells[0].Axes["precision"] = "mutated"
	if v.Axes["precision"] != "fp4" {
		t.Error("the cell's axes alias the variant's map; a pinned plan must not point back at its source")
	}
	if cells[0].Parent != "gemm" {
		t.Errorf("Parent = %q, want the entry it expanded from — recorded, not recovered by splitting the "+
			"name, which is wrong silently for a test called \"gemm-sweep\"", cells[0].Parent)
	}
}

func TestEnvSafeAxis(t *testing.T) {
	for _, c := range []struct {
		axis string
		want bool
	}{
		{"precision", true},
		{"messageBytes", true},
		{"class_2", true},
		{"_leading", true},
		{"", false},
		{"has-hyphen", false},
		{"has.dot", false},
		{"9leading", false},
		{"has space", false},
	} {
		if got := EnvSafeAxis(c.axis); got != c.want {
			t.Errorf("EnvSafeAxis(%q) = %v, want %v", c.axis, got, c.want)
		}
	}
}

// The env rendering is the half a RUNNER sees, and it must be deterministic:
// an unsorted map would make two identical plans produce two different
// container specs, and a diff of a pod against itself is one nobody can act on.
func TestVariantEnv(t *testing.T) {
	got := VariantEnv(map[string]string{"precision": "fp4", "class": "smoke", "bad-name": "x"})

	want := []EnvVar{
		{Name: "BURNIN_VARIANT_CLASS", Value: "smoke"},
		{Name: "BURNIN_VARIANT_PRECISION", Value: "fp4"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %+v, want %+v — an axis whose name cannot become a variable is SKIPPED, never mangled", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] = %+v, want %+v (sorted by axis name)", i, got[i], want[i])
		}
	}
	if VariantEnv(nil) != nil {
		t.Error("no axes must render no variables, not an empty slice a caller might iterate into a set")
	}
}

// nccl's `collective` axis is a real, shipped consumer of this name. Spelled
// out here so a change to the prefix fails beside the runner that reads it.
func TestVariantEnvName(t *testing.T) {
	if got := VariantEnvName("collective"); got != "BURNIN_VARIANT_COLLECTIVE" {
		t.Errorf("VariantEnvName(\"collective\") = %q — runners/nccl reads BURNIN_VARIANT_COLLECTIVE", got)
	}
}

// #296: an axis that cannot reach the runner is refused at plan time, not
// silently skipped, because the cell would run the DEFAULT configuration while
// the run reported it under a distinct label — a confident wrong answer dressed
// as evidence, and worse than a crash.
func TestRefuseUnreachableAxes(t *testing.T) {
	for _, c := range []struct{ name, axis string }{
		{"a hyphen is not legal in an environment variable", "message-bytes"},
		{"nor is a dot", "message.bytes"},
		{"nor may it lead with a digit", "8bytes"},
		{"nor may it be empty", ""},
	} {
		err := RefuseUnreachableAxes([]Test{{Name: "sweep-cell", Axes: map[string]string{c.axis: "v"}}})
		if err == nil {
			t.Errorf("%s: accepted axis %q", c.name, c.axis)
			continue
		}
		if !strings.Contains(err.Error(), "sweep-cell") {
			t.Errorf("%s: the refusal does not name the test: %v", c.name, err)
		}
	}

	ok := []Test{{Name: "t", Axes: map[string]string{"precision": "fp4", "class": "smoke", "measurand": "bandwidth"}}}
	if err := RefuseUnreachableAxes(ok); err != nil {
		t.Errorf("refused the axis names the shipped samples use: %v", err)
	}
}

// The refusal names ONE axis, and the same one every time. Go randomises map
// iteration, and a plan that blames a different axis on each attempt is one
// nobody can act on.
func TestRefuseUnreachableAxes_IsDeterministic(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 30; i++ {
		err := RefuseUnreachableAxes([]Test{{
			Name: "cell",
			Axes: map[string]string{"bad-one": "a", "bad.two": "b", "9bad": "c"},
		}})
		if err == nil {
			t.Fatal("accepted three illegal axes")
		}
		seen[err.Error()] = true
	}
	if len(seen) != 1 {
		t.Errorf("got %d different refusals across 30 runs, want 1: %v", len(seen), seen)
	}
}

func TestCopyAxes(t *testing.T) {
	if CopyAxes(nil) != nil || CopyAxes(map[string]string{}) != nil {
		t.Error("an empty map copies to nil, so an unexpanded test delivers no axes rather than an empty object")
	}
	src := map[string]string{"a": "1"}
	dst := CopyAxes(src)
	dst["a"] = "2"
	if src["a"] != "1" {
		t.Error("CopyAxes returned an alias")
	}
}

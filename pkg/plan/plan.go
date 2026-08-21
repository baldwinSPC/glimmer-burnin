// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Package plan turns a profile entry into the executions a dispatcher runs.
//
// ONE BRAIN, TWO DISPATCHERS — the same argument pkg/verdict and pkg/runner
// are public for, applied one stage earlier. The operator resolves a profile
// into pods; cmd/burnin resolves the SAME profile into containers on a bare
// host. Everything downstream of this package (result identity, topology
// validation, the verdict, the delivery envelope) already agrees between the
// two, because both read the same runner output through the same parser and
// judge it with the same evaluator.
//
// Variant expansion did not agree. It lived unexported in the reconciler, so
// `burnin run` SILENTLY DROPPED spec.tests[].variants: a profile whose four
// FP4/FP8/BF16/TF32 cells are the entire point of the acceptance ran as one
// execution of the parent test, and nothing said so. The cell's thresholds
// were not applied, its BURNIN_VARIANT_* variables never reached the runner,
// and the report named one result where the profile asked for four. A user
// comparing a bare-metal run against the same profile in-cluster would have
// found a different number of results and no explanation for it.
//
// So expansion moved HERE, where both dispatchers call it, rather than being
// reimplemented on the CLI side. A second implementation would have been the
// same category of mistake this package exists to end: two answers to one
// question, drifting, with nothing comparing them.
//
// WHAT THIS PACKAGE DOES NOT DO. It does not interpret an axis, ever. Axes are
// opaque labels that mean something to a runner and nothing to a dispatcher —
// see TestVariant's own doc comment, which is the canonical statement of that
// rule. This package upper-cases a name, refuses one that cannot become an
// environment variable, and passes the value through.
package plan

import (
	"fmt"
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Test is one execution: a test name, the spec pinned as it was when the plan
// was built, and the variant labels it came from.
//
// The JSON tags are load-bearing on the operator's side — this is the shape
// stored in the run's pinned plan, and an in-flight run reads back what an
// older manager wrote. They are inert on the CLI's side, which never
// serialises a plan. Do not rename one without the other in mind.
type Test struct {
	Name     string             `json:"name"`
	Required bool               `json:"required"`
	Spec     api.BurnInTestSpec `json:"spec"`

	// Axes are the variant labels this execution came from, or nil for a test
	// with no variants. They are carried, echoed and never interpreted.
	//
	// Pinned like the rest of the spec, so editing the profile mid-run cannot
	// change what an in-flight cell reports it was.
	Axes map[string]string `json:"axes,omitempty"`

	// Parent is the profile entry this cell was expanded from, or empty when
	// the entry was not expanded at all.
	//
	// Recorded rather than recovered from Name. Splitting "<test>-<variant>"
	// on the last hyphen works until a test is called "gemm-sweep" or a variant
	// "fp8-dense", and then it is wrong silently — which is the same reason
	// TestResult.VariantAxes exists instead of asking a consumer to parse names.
	Parent string `json:"parent,omitempty"`
}

// ExpandVariants turns one profile entry into one Test per cell.
//
// This is the whole of the matrix feature, and it is deliberately the whole of
// it. Expansion happens ONCE, here, before either dispatcher builds its plan —
// so every stage downstream (the duplicate-name check, Pair and Group topology
// validation, pod or container naming, result identity, verdict, delivery keys,
// TestResult.Nodes) works unchanged, because none of them ever learns variants
// exist. That is why this is a small function rather than a reconciler rewrite,
// and it is the property to protect if this is ever extended.
//
// With no variants it returns exactly the one test it was given, so a profile
// written before variants existed plans identically on both dispatchers.
func ExpandVariants(
	name string,
	spec api.BurnInTestSpec,
	required bool,
	variants []api.TestVariant,
) []Test {
	// An empty scope is Node. The apiserver defaults it, but the plan must not
	// depend on that having happened: a suite file read off a laptop never went
	// near an apiserver at all, and the operator's settleWithoutPod and
	// pkg/localrun already read "" as Node. A real TestResult must carry the
	// scope it ran at now that the schema no longer refuses one without (see
	// TestResult.Scope). One place, before variants are expanded, so every cell
	// inherits it.
	if spec.Scope == "" {
		spec.Scope = api.ScopeNode
	}
	if len(variants) == 0 {
		return []Test{{Name: name, Spec: spec, Required: required}}
	}

	out := make([]Test, 0, len(variants))
	for _, v := range variants {
		// A deep copy per cell: the overlays below must not reach back into the
		// parent spec, or the second variant would inherit the first's edits and
		// a sweep would silently measure the wrong thing in every cell after the
		// first.
		cell := *spec.DeepCopy()

		if v.DurationSeconds != nil {
			cell.DurationSeconds = *v.DurationSeconds
		}
		// A VARIANT THAT OVERLAYS A RUNNER FIELD IS ASKING FOR A RUNNER BLOCK,
		// and the test it overlays usually has none: a built-in kind resolves its
		// image from pkg/runnerimages, so there is nothing for its author to put
		// in `runner:`. That is the CANONICAL use of this feature — five cells
		// differing by one environment variable — and writing through a nil
		// pointer panicked the reconciler on the first profile anybody wrote.
		//
		// A nil dereference here is not one failed run. controller-runtime
		// recovers the panic and requeues, so it crash-loops every pass and the
		// operator stops making progress for every OTHER run in the cluster too.
		// On the CLI side it is a stack trace instead of an acceptance report.
		//
		// Creating an empty block is safe for image resolution: Resolve treats an
		// empty Image and an empty ImagesByVendor exactly as it treats a nil
		// RunnerSpec, so a cell that gains one still lands on the kind's default.
		if v.Args != nil || v.Env != nil {
			if cell.Runner == nil {
				cell.Runner = &api.RunnerSpec{}
			}
		}
		if v.Args != nil {
			cell.Runner.Args = append([]string(nil), v.Args...)
		}
		if v.Env != nil {
			cell.Runner.Env = append([]corev1.EnvVar(nil), v.Env...)
		}
		if v.RepeatCount != nil {
			cell.RepeatCount = v.RepeatCount
		}
		if v.Thresholds != nil {
			// REPLACE, never merge. See TestVariant.Thresholds: merging by
			// metric name would silently retain a gate the author believed
			// replaced, and a node failed against a threshold nobody can find
			// in the profile has a verdict with nothing to read that explains
			// it. An empty non-nil list therefore means "no thresholds", which
			// is a thing a variant may legitimately want.
			cell.Thresholds = append([]api.Threshold(nil), v.Thresholds...)
		}

		out = append(out, Test{
			Name:     name + "-" + v.Name,
			Spec:     cell,
			Required: required,
			Axes:     CopyAxes(v.Axes),
			Parent:   name,
		})
	}
	return out
}

// CopyAxes returns an independent copy, or nil for an empty map.
//
// Callers pin a plan precisely so that editing the profile mid-run changes
// nothing in flight, and a map handed over by reference would leave a live
// pointer back into the thing the pinning exists to isolate it from.
func CopyAxes(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

// EnvVar is one name/value pair, in neither dispatcher's shape.
//
// Deliberately not corev1.EnvVar: the operator turns these into a container's
// env list and the CLI turns them into a map for a `docker run -e`, and a
// shared decision that arrived pre-shaped for one of them would make the other
// unpack it again. What is shared is WHICH axes reach the runner and under
// WHAT NAME — that is the part that must not differ.
type EnvVar struct {
	Name  string
	Value string
}

// VariantEnv renders a cell's axes as BURNIN_VARIANT_<AXIS>=<value>.
//
// Sorted by axis name so the output is deterministic: an unsorted map would
// make two identical plans produce two different container specs, and a diff
// of a pod against itself is a diff nobody can act on.
//
// An axis whose name is not usable as an environment variable is SKIPPED
// rather than mangled. A mangled name would collide two axes onto one variable,
// and the runner would read one cell's value while reporting the other's — the
// same class of error as rewriting an artifact name into a colliding ConfigMap
// key. RefuseUnreachableAxes makes that skip unreachable at plan time; this
// stays as the backstop.
func VariantEnv(axes map[string]string) []EnvVar {
	if len(axes) == 0 {
		return nil
	}
	names := make([]string, 0, len(axes))
	for k := range axes {
		if EnvSafeAxis(k) {
			names = append(names, k)
		}
	}
	sort.Strings(names)

	out := make([]EnvVar, 0, len(names))
	for _, k := range names {
		out = append(out, EnvVar{Name: VariantEnvName(k), Value: axes[k]})
	}
	return out
}

// VariantEnvName is the environment variable one axis becomes.
//
// Exported because it is the contract a runner reads — `nccl` looks for
// BURNIN_VARIANT_COLLECTIVE — and a test asserting a runner receives its axis
// should spell the name the same way the dispatcher does.
func VariantEnvName(axis string) string {
	return "BURNIN_VARIANT_" + strings.ToUpper(axis)
}

// EnvSafeAxis is whether an axis name maps to an environment variable without
// rewriting. Letters, digits and underscore only, and not leading with a digit.
func EnvSafeAxis(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r == '_':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// RefuseUnreachableAxes rejects a plan whose axis would never reach the runner.
//
// VariantEnv skips such an axis rather than mangling it, which is the right
// instinct — a mangled name would silently configure the WRONG thing — but the
// axis is still copied into TestResult.VariantAxes and delivered in the
// envelope. So the cell runs the runner's default configuration while the run
// reports it under a distinct axis label (#296).
//
// That is a confident wrong answer dressed as evidence, and worse than a
// crash. pkg/contract tells a consumer to group a sweep's cells BY these
// labels, so a precision sweep or a message-size sweep reports N cells that all
// measured the same thing — and if the cells carry per-variant thresholds, as
// config/samples/variant-sweep.yaml demonstrates, those gates are applied to a
// run that was never told which measurand to take. It is the sample's own
// warning ("a sweep certified against measurements nobody took") reached
// through the axis name rather than through the kind.
//
// Refused at plan time — before any node is cordoned in-cluster, before any
// container starts on bare metal — for the reason every other rule here is: a
// run that quietly produces a meaningless sweep is a much worse report than one
// that refuses at start and says which axis and why.
func RefuseUnreachableAxes(tests []Test) error {
	for _, t := range tests {
		for _, k := range SortedAxisKeys(t.Axes) {
			if EnvSafeAxis(k) {
				continue
			}
			return fmt.Errorf(
				"test %q declares variant axis %q, which cannot become an environment variable, so the runner "+
					"would never receive it: the cell would run the default configuration while the run reported "+
					"it under this axis label, and any threshold on the cell would gate a measurement nobody asked "+
					"for. An axis name must be letters, digits and underscores, and must not start with a digit "+
					"(it becomes %s)",
				t.Name, k, VariantEnvName("<name>"))
		}
	}
	return nil
}

// SortedAxisKeys makes a refusal deterministic: Go randomises map iteration,
// and a run that refuses naming a different axis on each attempt is a run
// nobody can act on.
func SortedAxisKeys(axes map[string]string) []string {
	if len(axes) == 0 {
		return nil
	}
	out := make([]string, 0, len(axes))
	for k := range axes {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

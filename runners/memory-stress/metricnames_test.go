// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// This file is the seam between this runner and the operator, and it is why the
// runner's source lives in the repository's root module rather than beside its
// Dockerfile as an island.
//
// The runner prints its OWN key names; pkg/runner maps them to the canonical
// names a threshold is written against, and pkg/contract owns the grammar those
// names must obey. Nothing at runtime reconciles a disagreement: a key that
// normalises to an illegal name, or onto another key's name, is silently
// unthresholdable, and the symptom is a node nobody can accept for reasons
// nobody can see. Running the real output through the real parser is what keeps
// the three in step.
//
// These imports are test-only. The image build strips *_test.go, which is what
// lets the runner's own sources stay standard-library-only.

// canonicalNames parses the runner's stdout the way the operator does.
func canonicalNames(t *testing.T, stdout string, exitCode int) runner.Result {
	t.Helper()
	return runner.Parse("memory-stress", stdout, exitCode)
}

func TestEmittedMetricNamesObeyTheContract(t *testing.T) {
	code, reason, stdout, _ := runExecute(t, cleanRun, 0, nil)
	if code != exitPass {
		t.Fatalf("exit = %d (%s)", code, reason)
	}
	res := canonicalNames(t, stdout, code)

	if len(res.InvalidNames) != 0 {
		t.Errorf("runner emitted keys that fail the metric grammar: %v", res.InvalidNames)
	}
	for name := range res.Metrics {
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("canonical name from this runner is invalid: %v", err)
		}
	}

	// The names the acceptance thresholds for this kind are written against.
	// A rename on either side of the seam drops the measurement silently.
	for _, name := range []string{
		"memoryErrors", "miscompares", "sdcDetections",
		"readBandwidthMBs", "writeBandwidthMBs", "elapsedS",
	} {
		if _, ok := res.Metrics[name]; !ok {
			t.Errorf("canonical metric %q was not produced; parsed set was %v", name, res.Metrics)
		}
	}
	if res.Message != "STRESSAPPTEST_PASS" {
		t.Errorf("message = %q, want the pass marker", res.Message)
	}
}

// A dimensional measurement whose name declares no unit is stored as a bare
// number that charts happily and compares against nothing. The MB/s and
// percentage keys are the ones at risk, because the tool's own casing ("mbs",
// "pct") does not survive generic normalisation.
func TestDimensionalMetricsKeepTheirUnits(t *testing.T) {
	_, _, stdout, _ := runExecute(t, cleanRun, 0, nil)
	res := canonicalNames(t, stdout, exitPass)

	want := map[string]contract.Unit{
		"readBandwidthMBs":   contract.UnitMegabytesPerSecond,
		"writeBandwidthMBs":  contract.UnitMegabytesPerSecond,
		"elapsedS":           contract.UnitSeconds,
		"satRunS":            contract.UnitSeconds,
		"durationRequestedS": contract.UnitSeconds,
		"memoryCoveredPct":   contract.UnitPercent,
	}
	for name, unit := range want {
		if _, ok := res.Metrics[name]; !ok {
			t.Errorf("%q was not produced", name)
			continue
		}
		if got := contract.UnitOf(name); got != unit {
			t.Errorf("%q declares unit %q, want %q", name, got, unit)
		}
	}
	// The counters must NOT carry a unit suffix: rule 2 of the grammar.
	for _, name := range []string{"memoryErrors", "miscompares", "sdcDetections", "threads"} {
		if got := contract.UnitOf(name); got != contract.UnitNone {
			t.Errorf("counter %q declares unit %q, want none", name, got)
		}
	}
}

// Parsing is last-occurrence-wins, so two raw keys landing on one canonical
// name would silently discard one of two real measurements.
func TestNoTwoEmittedKeysCollide(t *testing.T) {
	_, _, stdout, _ := runExecute(t, cleanRun, 0, nil)

	sources := map[string][]string{}
	for _, line := range strings.Split(stdout, "\n") {
		key, _, ok := strings.Cut(line, "=")
		if !ok || strings.ContainsAny(key, " \t") {
			continue
		}
		res := runner.Parse("memory-stress", line, 0)
		for name := range res.Metrics {
			sources[name] = append(sources[name], key)
		}
	}
	if len(sources) == 0 {
		t.Fatalf("no metrics parsed from:\n%s", stdout)
	}
	for name, keys := range sources {
		if len(keys) > 1 {
			t.Errorf("keys %v all normalise to %q; one of those measurements would be discarded", keys, name)
		}
	}
}

// The exit codes are the runner contract. Exit 3 in particular must reach the
// operator as an Error — hardware unjudged — and never as a Fail.
func TestExitCodesMapToTheIntendedVerdicts(t *testing.T) {
	cases := []struct {
		code int
		want runner.Verdict
	}{
		{exitPass, runner.VerdictPass},
		{exitFail, runner.VerdictFail},
		{exitSkip, runner.VerdictSkip},
		{exitError, runner.VerdictError},
	}
	for _, c := range cases {
		if got := runner.VerdictFor(c.code); got != c.want {
			t.Errorf("exit %d maps to %s, want %s", c.code, got, c.want)
		}
	}
}

// Metrics this runner reports must be ones a profile is allowed to gate on,
// or the profile author finds out at verdict time instead of at admission.
func TestAcceptanceMetricsAreThresholdable(t *testing.T) {
	for _, name := range []string{"memoryErrors", "miscompares", "sdcDetections", "readBandwidthMBs", "elapsedS"} {
		if !contract.SafeToThresholdOn(name) {
			t.Errorf("%q is registered as unsafe to threshold on, but this runner's acceptance depends on it", name)
		}
	}
}

package main

import (
	"testing"
	"time"
)

func TestResolveLevel_Explicit(t *testing.T) {
	cases := []struct {
		in       string
		duration time.Duration
		want     int
	}{
		{"1", time.Hour, 1},
		{"4", 10 * time.Second, 4},
		{"short", 0, 1},
		{"MEDIUM", 0, 2},
		{" long ", 0, 3},
		{"xlong", 0, 4},
	}
	for _, tc := range cases {
		got, source, err := resolveLevel(tc.in, tc.duration)
		if err != nil {
			t.Fatalf("resolveLevel(%q) errored: %v", tc.in, err)
		}
		if got != tc.want {
			t.Errorf("resolveLevel(%q) = %d, want %d", tc.in, got, tc.want)
		}
		if source != "explicit" {
			t.Errorf("resolveLevel(%q) source = %q, want explicit", tc.in, source)
		}
	}
}

// An explicit level must survive a duration too short for it. Silently
// downgrading would turn a level-4 acceptance gate into a level-1 one with
// nothing in the output admitting the substitution — the node would be
// certified against a test it never ran.
func TestResolveLevel_ExplicitIsNotDowngradedToFit(t *testing.T) {
	got, _, err := resolveLevel("4", 30*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != 4 {
		t.Errorf("level = %d, want 4: an explicit level must not be quietly weakened", got)
	}
}

func TestResolveLevel_Invalid(t *testing.T) {
	for _, in := range []string{"0", "5", "-1", "extreme", "3.5", "one"} {
		if _, _, err := resolveLevel(in, time.Hour); err == nil {
			t.Errorf("resolveLevel(%q) accepted an invalid level", in)
		}
	}
}

// The derived level must fit inside the time the test was given, with the whole
// budget to spare. A level that overruns is killed at the pod deadline, which
// reports Error — no verdict, and a burn-in slot spent for nothing.
func TestDeriveLevel_ChoosesTheDeepestLevelThatFits(t *testing.T) {
	cases := []struct {
		duration time.Duration
		want     int
	}{
		{0, 1},
		{30 * time.Second, 1},
		{59 * time.Second, 1},
		{time.Minute, 1},
		{119 * time.Second, 1},
		{2 * time.Minute, 2},
		{10 * time.Minute, 2},
		{15 * time.Minute, 3},
		{59 * time.Minute, 3},
		{time.Hour, 4},
		{6 * time.Hour, 4},
	}
	for _, tc := range cases {
		if got := deriveLevel(tc.duration); got != tc.want {
			t.Errorf("deriveLevel(%s) = %d, want %d", tc.duration, got, tc.want)
		}
	}
}

func TestDeriveLevel_NeverExceedsItsBudget(t *testing.T) {
	for _, d := range []time.Duration{0, time.Second, 90 * time.Second, 14 * time.Minute, 90 * time.Minute} {
		lvl := deriveLevel(d)
		if lvl > 1 && levelBudgets[lvl] > d {
			t.Errorf("deriveLevel(%s) = %d, whose budget %s does not fit", d, lvl, levelBudgets[lvl])
		}
	}
}

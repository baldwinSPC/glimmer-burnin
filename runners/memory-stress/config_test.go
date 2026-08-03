// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
)

// env builds a getenvFunc over a map, so a test never touches the process
// environment and cases stay independent.
func env(pairs map[string]string) getenvFunc {
	return func(name string) (string, bool) {
		v, ok := pairs[name]
		return v, ok
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatalf("loadConfig with an empty environment: %v", err)
	}
	// A runner started by hand must behave like one started by a BurnInRun,
	// whose controller defaults to the same 600s.
	if cfg.durationSeconds != defaultDurationSeconds {
		t.Errorf("durationSeconds = %d, want %d", cfg.durationSeconds, defaultDurationSeconds)
	}
	if cfg.fraction != defaultFraction || cfg.minMB != defaultMinMB {
		t.Errorf("fraction=%v minMB=%d, want %v and %d", cfg.fraction, cfg.minMB, defaultFraction, defaultMinMB)
	}
	if cfg.memoryMB != 0 || cfg.threads != 0 {
		t.Errorf("memoryMB=%d threads=%d, want 0 (auto) for both", cfg.memoryMB, cfg.threads)
	}
	if cfg.stressfulCopy {
		t.Error("stressfulCopy defaulted on; the tool's own default is off")
	}
}

func TestLoadConfigHonoursDuration(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_DURATION_SECONDS": "3600"}))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.durationSeconds != 3600 {
		t.Errorf("durationSeconds = %d, want 3600", cfg.durationSeconds)
	}
}

// A malformed value is refused rather than replaced by a default. Silently
// falling back would accept a whole fleet on a workload nobody chose —
// BURNIN_MEMORY_FRACTION=70 meaning "70%" is the obvious way to write it wrong.
func TestLoadConfigRefusesMalformedValues(t *testing.T) {
	cases := []struct{ name, value string }{
		{"BURNIN_DURATION_SECONDS", "ten minutes"},
		{"BURNIN_DURATION_SECONDS", "0"},
		{"BURNIN_MEMORY_FRACTION", "70"},
		{"BURNIN_MEMORY_FRACTION", "0"},
		{"BURNIN_MEMORY_MB", "-1"},
		{"BURNIN_MEMORY_VERBOSITY", "99"},
		{"BURNIN_MEMORY_STRESSFUL_COPY", "ture"},
	}
	for _, c := range cases {
		if _, err := loadConfig(env(map[string]string{c.name: c.value})); err == nil {
			t.Errorf("%s=%q was accepted; it should be refused", c.name, c.value)
		} else if !strings.Contains(err.Error(), c.name) {
			t.Errorf("%s=%q: error %q does not name the offending variable", c.name, c.value, err)
		}
	}
}

func TestLoadConfigEmptyValueUsesDefault(t *testing.T) {
	// Kubernetes env vars are frequently present but empty; that is "unset".
	cfg, err := loadConfig(env(map[string]string{"BURNIN_MEMORY_MB": "", "BURNIN_DURATION_SECONDS": " "}))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if cfg.memoryMB != 0 || cfg.durationSeconds != defaultDurationSeconds {
		t.Errorf("memoryMB=%d durationSeconds=%d, want 0 and %d", cfg.memoryMB, cfg.durationSeconds, defaultDurationSeconds)
	}
}

// The -s the tool is given is deliberately smaller than the budget: allocation
// and pattern fill happen before its clock starts, and overrunning the pod's
// deadline turns a whole run into Errors.
func TestSatSecondsReservesSetupTime(t *testing.T) {
	cases := []struct{ duration, want int }{
		{600, 540},     // 10% reserve
		{60, 45},       // floored at the 15s minimum reserve
		{20, 10},       // reserve would leave 5s; the minimum run wins
		{36000, 35940}, // capped at the 60s maximum reserve
	}
	for _, c := range cases {
		if got := satSeconds(c.duration); got != c.want {
			t.Errorf("satSeconds(%d) = %d, want %d", c.duration, got, c.want)
		}
		if got := satSeconds(c.duration); got > c.duration {
			t.Errorf("satSeconds(%d) = %d, which exceeds the budget", c.duration, got)
		}
	}
}

func TestPlanArgs(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	p := plan{testMB: 4096, threads: 8, satSeconds: 540}

	got := strings.Join(p.args(cfg), " ")
	want := "-M 4096 -s 540 -m 8 -v 8"
	if got != want {
		t.Errorf("args = %q, want %q", got, want)
	}
	// Flags for thread classes that are switched off are not passed at all: a
	// flag we never send is a flag that cannot be mis-parsed by a future
	// stressapptest.
	if strings.Contains(got, "-i") || strings.Contains(got, "-C") || strings.Contains(got, "-W") {
		t.Errorf("args = %q; disabled thread classes must not appear", got)
	}

	cfg, err = loadConfig(env(map[string]string{
		"BURNIN_MEMORY_INVERT_THREADS": "2",
		"BURNIN_MEMORY_CHECK_THREADS":  "2",
		"BURNIN_MEMORY_CPU_THREADS":    "1",
		"BURNIN_MEMORY_STRESSFUL_COPY": "true",
		"BURNIN_MEMORY_EXTRA_ARGS":     "--max_errors 4",
	}))
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	got = strings.Join(p.args(cfg), " ")
	want = "-M 4096 -s 540 -m 8 -v 8 -i 2 -c 2 -C 1 -W --max_errors 4"
	if got != want {
		t.Errorf("args = %q, want %q", got, want)
	}
}

func TestMarkerLines(t *testing.T) {
	// The pass marker carries no suffix: it is the exact string the operator's
	// parser test pins as this kind's Message.
	if got := marker(exitPass, ""); got != "STRESSAPPTEST_PASS" {
		t.Errorf("marker(pass) = %q", got)
	}
	for _, c := range []struct {
		code   int
		prefix string
	}{
		{exitFail, "STRESSAPPTEST_FAIL: "},
		{exitSkip, "STRESSAPPTEST_SKIP: "},
		{exitError, "STRESSAPPTEST_ERROR: "},
	} {
		if got := marker(c.code, "why"); got != c.prefix+"why" {
			t.Errorf("marker(%d) = %q, want %q", c.code, got, c.prefix+"why")
		}
	}
}

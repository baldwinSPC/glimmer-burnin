// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"bytes"
	"strings"
	"testing"
)

func metricsOf(out string) map[string]string {
	m := map[string]string{}
	for _, line := range strings.Split(out, "\n") {
		key, value, ok := strings.Cut(line, "=")
		if !ok || strings.ContainsAny(key, " \t") {
			continue
		}
		m[key] = value
	}
	return m
}

func TestExecuteEndToEndCleanRunPasses(t *testing.T) {
	// The real floor (minHoldSeconds) is 30s per pattern by design; shrunk here
	// so this test exercises the real end-to-end path in about a second
	// instead of a minute, the same reason memory-stress's own tests shrink
	// its hardCapGrace var.
	old := minHoldSeconds
	minHoldSeconds = 1
	defer func() { minHoldSeconds = old }()

	root := fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G})
	vars := map[string]string{
		"BURNIN_DURATION_SECONDS":       "60",
		"BURNIN_RETENTION_MB":           "1",
		"BURNIN_RETENTION_HOLD_SECONDS": "1", // real, but short — two patterns is ~2s
	}
	var out, errOut bytes.Buffer
	code, reason := execute(&out, &errOut, env(vars), root)
	if code != exitPass {
		t.Fatalf("execute = %d (%q), want exitPass\nstderr:\n%s", code, reason, errOut.String())
	}
	m := metricsOf(out.String())
	if m["region_mb"] != "1" {
		t.Errorf("region_mb = %q, want \"1\"", m["region_mb"])
	}
	if m["retention_patterns_completed"] != "2" {
		t.Errorf("retention_patterns_completed = %q, want \"2\"", m["retention_patterns_completed"])
	}
	if m["retention_bit_flips"] != "0" {
		t.Errorf("retention_bit_flips = %q, want \"0\" on a clean run", m["retention_bit_flips"])
	}
	if _, present := m["retention_first_flip_offset"]; present {
		t.Error("first_flip_offset was reported on a clean run — a metric with nothing behind it")
	}
}

func TestExecuteSkipsATooSmallMachine(t *testing.T) {
	root := fakeRoot(t, map[string]string{"proc/meminfo": "MemTotal: 10240 kB\nMemAvailable: 8192 kB\n"})
	code, reason := execute(&bytes.Buffer{}, &bytes.Buffer{}, env(nil), root)
	if code != exitSkip {
		t.Fatalf("execute on a genuinely tiny machine = %d (%q), want exitSkip", code, reason)
	}
}

func TestExecuteErrorsOnAConfigMistake(t *testing.T) {
	root := fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G})
	code, reason := execute(&bytes.Buffer{}, &bytes.Buffer{}, env(map[string]string{
		"BURNIN_RETENTION_FRACTION": "not a number",
	}), root)
	if code != exitError {
		t.Fatalf("execute with a malformed env var = %d (%q), want exitError", code, reason)
	}
}

func TestExecuteErrorsWhenMeminfoIsUnreadable(t *testing.T) {
	root := t.TempDir() // no proc/meminfo at all
	code, reason := execute(&bytes.Buffer{}, &bytes.Buffer{}, env(nil), root)
	if code != exitError {
		t.Fatalf("execute with no /proc/meminfo = %d (%q), want exitError — the node was not judged, not skipped", code, reason)
	}
}

func TestMarkerFormatsEveryExitCode(t *testing.T) {
	cases := []struct {
		code int
		want string
	}{
		{exitPass, markerPass},
		{exitFail, markerFail + ": why"},
		{exitSkip, markerSkip + ": why"},
		{exitError, markerError + ": why"},
		{99, markerError + ": why"}, // anything unrecognised is an Error, never a silent pass
	}
	for _, c := range cases {
		reason := "why"
		if c.code == exitPass {
			reason = ""
		}
		if got := marker(c.code, reason); got != c.want {
			t.Errorf("marker(%d, %q) = %q, want %q", c.code, reason, got, c.want)
		}
	}
}

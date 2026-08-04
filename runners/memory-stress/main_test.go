// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fakeSat writes a stand-in stressapptest that prints the given capture and
// exits with the given code. The wrapper's whole value is what it does with
// that output, so the tool itself is the part worth faking.
func fakeSat(t *testing.T, output string, exitCode int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "stressapptest")
	script := "#!/bin/sh\ncat <<'SATEOF'\n" + output + "\nSATEOF\nexit " + strconv.Itoa(exitCode) + "\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

// metrics turns the runner's stdout into the map the operator's parser would
// build from it, minus the canonical-name mapping.
func metrics(t *testing.T, out string) map[string]string {
	t.Helper()
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

func runExecute(t *testing.T, satOutput string, satExit int, extraEnv map[string]string) (code int, reason, stdout, stderr string) {
	t.Helper()
	root := fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G})
	vars := map[string]string{
		"BURNIN_DURATION_SECONDS":  "60",
		"BURNIN_STRESSAPPTEST_BIN": fakeSat(t, satOutput, satExit),
	}
	for k, v := range extraEnv {
		vars[k] = v
	}
	var out, errOut bytes.Buffer
	code, reason = execute(&out, &errOut, env(vars), root)
	// main() appends the marker to stdout; the tests assert against the whole
	// stream the operator would read, so it is appended the same way here.
	out.WriteString(marker(code, reason) + "\n")
	return code, reason, out.String(), errOut.String()
}

// The keys asserted here are the ones pkg/runner's memory-stress alias table
// maps to canonical metric names. If a key is renamed on either side the
// measurement stops being recorded, silently — so it is pinned from this end.
func TestExecuteEmitsTheContractedMetricKeys(t *testing.T) {
	code, reason, stdout, stderr := runExecute(t, cleanRun, 0, nil)
	if code != exitPass {
		t.Fatalf("exit = %d (%s)\nstdout:\n%s\nstderr:\n%s", code, reason, stdout, stderr)
	}

	m := metrics(t, stdout)
	for _, key := range []string{
		"tool", "tool_version", "duration_requested_s", "threads",
		"memory_covered_pct", "elapsed_s", "sat_run_s", "sdc_count",
		"hardware_incidents", "miscompares",
		"read_bandwidth_mbs", "write_bandwidth_mbs", "sat_status",
	} {
		if _, ok := m[key]; !ok {
			t.Errorf("metric %q missing from stdout:\n%s", key, stdout)
		}
	}
	if m["tool"] != "stressapptest" {
		t.Errorf("tool = %q", m["tool"])
	}
	if m["hardware_incidents"] != "0" || m["miscompares"] != "0" || m["sdc_count"] != "0" {
		t.Errorf("clean run reported incidents=%q miscompares=%q sdc=%q", m["hardware_incidents"], m["miscompares"], m["sdc_count"])
	}
	if m["read_bandwidth_mbs"] != "8737.85" || m["write_bandwidth_mbs"] != "8737.85" {
		t.Errorf("read=%q write=%q, want 8737.85 for a copy-only run", m["read_bandwidth_mbs"], m["write_bandwidth_mbs"])
	}
	if m["duration_requested_s"] != "60" {
		t.Errorf("duration_requested_s = %q, want the BURNIN_DURATION_SECONDS it was given", m["duration_requested_s"])
	}

	// The tool's own output reaches the log for a human, but PREFIXED: the
	// operator parses metrics out of a stream that merges stdout and stderr,
	// and a raw tool line could be read as a measurement.
	if !strings.Contains(stderr, "sat| ") {
		t.Errorf("stressapptest output was not forwarded to the log:\n%s", stderr)
	}
	if strings.Contains(stdout, "Stats:") {
		t.Errorf("raw tool output leaked into the metrics stream:\n%s", stdout)
	}
}

// Evidence first, decision second: a failing run must still report what it
// measured, or the operator gets a verdict with nothing behind it.
func TestExecuteEmitsMetricsBeforeFailing(t *testing.T) {
	const failing = `2026/08/02-11:00:31(7) Hardware Error: miscompare on CPU 3(0x8) at 0xffff8801(0x0:DIMM Unknown): read:0x0, reread:0x0 expected:0xdeadbeef
2026/08/02-11:01:00(7) Stats: Found 1 hardware incidents
2026/08/02-11:01:00(7) Stats: Completed: 61440.00M in 45.01s 1023.83MB/s, with 1 hardware incidents, 4 errors
2026/08/02-11:01:00(7) Stats: Memory Copy: 61440.00M at 1023.83MB/s
2026/08/02-11:01:00(7) Status: FAIL - test discovered HW problems`

	code, reason, stdout, _ := runExecute(t, failing, 1, nil)
	if code != exitFail {
		t.Fatalf("exit = %d (%s), want exitFail\n%s", code, reason, stdout)
	}
	m := metrics(t, stdout)
	if m["hardware_incidents"] != "1" || m["miscompares"] != "4" || m["sdc_count"] != "1" {
		t.Errorf("incidents=%q miscompares=%q sdc=%q, want 1, 4 and 1", m["hardware_incidents"], m["miscompares"], m["sdc_count"])
	}
	if m["read_bandwidth_mbs"] == "" {
		t.Error("a failing run reported no bandwidth; the evidence must survive the verdict")
	}
}

// stressapptest exits 1 for a procedural failure too. Forwarding that exit code
// would condemn memory that was never successfully tested.
func TestExecuteReportsProceduralFailureAsError(t *testing.T) {
	const procedural = `2026/08/02-11:00:00(7) Process Error: failed to allocate memory
2026/08/02-11:00:00(7) Stats: Completed: 0.00M in 0.01s 0.00MB/s, with 0 hardware incidents, 0 errors
2026/08/02-11:00:00(7) Status: FAIL - test encountered procedural errors`

	code, reason, _, _ := runExecute(t, procedural, 1, nil)
	if code == exitFail {
		t.Fatalf("procedural failure reported as a hardware Fail: %s", reason)
	}
	if code != exitError {
		t.Errorf("exit = %d (%s), want exitError", code, reason)
	}
}

// A missing tool is the runner's own problem, not the node's.
func TestExecuteReportsAMissingToolAsError(t *testing.T) {
	root := fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G})
	var out, errOut bytes.Buffer
	code, reason := execute(&out, &errOut, env(map[string]string{
		"BURNIN_DURATION_SECONDS":  "60",
		"BURNIN_STRESSAPPTEST_BIN": filepath.Join(t.TempDir(), "not-installed"),
	}), root)
	if code != exitError {
		t.Errorf("exit = %d (%s), want exitError", code, reason)
	}
	// Even this path names the tool and the duration it was asked for.
	if m := metrics(t, out.String()); m["tool"] != "stressapptest" {
		t.Errorf("no tool metric on the error path:\n%s", out.String())
	}
}

// A run the wrapper had to kill is unjudged, even when the log it managed to
// print looks like a clean pass. The wrapper cannot tell "finished, then hung"
// from "printed part of a summary, then hung", and only one of those is a
// verdict — so neither is allowed to be one.
func TestExecuteKillsAWedgedRunAndReportsError(t *testing.T) {
	defer func(original time.Duration) { hardCapGrace = original }(hardCapGrace)
	hardCapGrace = 0 // the cap becomes BURNIN_DURATION_SECONDS itself

	path := filepath.Join(t.TempDir(), "stressapptest")
	script := "#!/bin/sh\ncat <<'SATEOF'\n" + cleanRun + "\nSATEOF\nsleep 30\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	root := fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G})
	var out, errOut bytes.Buffer
	start := time.Now()
	code, reason := execute(&out, &errOut, env(map[string]string{
		"BURNIN_DURATION_SECONDS":  "1",
		"BURNIN_STRESSAPPTEST_BIN": path,
	}), root)

	if code != exitError {
		t.Fatalf("exit = %d (%s), want exitError", code, reason)
	}
	if elapsed := time.Since(start); elapsed > 20*time.Second {
		t.Errorf("the wrapper took %s to end a wedged run; the cap did not fire", elapsed)
	}
	if !strings.Contains(reason, "not judged") {
		t.Errorf("reason = %q, should say the node was not judged", reason)
	}
	// The evidence it did gather still reaches the log.
	if m := metrics(t, out.String()); m["elapsed_s"] == "" {
		t.Errorf("a killed run reported no elapsed time:\n%s", out.String())
	}
}

// The production root is the one input every other test in this package injects
// around: they all pass a temp dir, so an empty root — which filepath.Join turns
// into the RELATIVE "proc/meminfo" — kept reading the right files purely because
// the image sets no WORKDIR and the process therefore starts in /. Anything that
// moved the working directory (a WORKDIR, a command:/workingDir: in the pod
// spec, a wrapper entrypoint) would have made every node exit 3 at once, which
// is an unjudged fleet rather than a visible crash.
//
// The decoy is what makes this a real test rather than a Linux-only one: it is
// a /proc a relative root WOULD read, planted where the bug would look. A node's
// verdict must never be sized from whatever the working directory happens to
// hold, on any host.
func TestProductionRootDoesNotDependOnTheWorkingDirectory(t *testing.T) {
	const decoyTotalKB = 999999999
	t.Chdir(fakeRoot(t, map[string]string{
		"proc/meminfo": fmt.Sprintf("MemTotal: %d kB\nMemAvailable: %d kB\n", decoyTotalKB, decoyTotalKB),
	}))

	if got := filepath.Join(productionRoot, "proc", "meminfo"); got != "/proc/meminfo" {
		t.Fatalf("the production root resolves meminfo to %q, want %q — sizing inputs must not be read relative to the working directory", got, "/proc/meminfo")
	}
	if sys, err := readSysInfo(productionRoot); err == nil && sys.memTotalBytes == decoyTotalKB*1024 {
		t.Fatalf("readSysInfo sized this node from the decoy /proc in the working directory (MemTotal %d kB)", decoyTotalKB)
	}

	// The production call itself, sized explicitly so the outcome turns on
	// whether the sizing inputs were found and not on the memory of whatever
	// machine is running the test.
	var out, errOut bytes.Buffer
	code, reason := execute(&out, &errOut, env(map[string]string{
		"BURNIN_DURATION_SECONDS":  "60",
		"BURNIN_MEMORY_MB":         "1",
		"BURNIN_STRESSAPPTEST_BIN": fakeSat(t, cleanRun, 0),
	}), productionRoot)

	// On Linux the real /proc/meminfo is there and the run passes. On a host
	// without one it is an Error — also correct, and its message has to name the
	// ABSOLUTE path, which is what proves where it looked. The one outcome that
	// must never happen is a verdict sized from the decoy.
	if code != exitPass && !strings.Contains(reason, "/proc/meminfo") {
		t.Fatalf("exit = %d (%s) from a working directory other than /; the runner did not read its sizing inputs from an absolute path\nstdout:\n%s\nstderr:\n%s",
			code, reason, out.String(), errOut.String())
	}
}

// A node too small to test is a Skip, and the marker says so in the exact shape
// the parser records as the result's Message.
func TestExecuteSkipsAnUnsizableNode(t *testing.T) {
	root := fakeRoot(t, map[string]string{
		"proc/meminfo": "MemTotal: 262144 kB\nMemAvailable: 204800 kB\n",
	})
	var out, errOut bytes.Buffer
	code, reason := execute(&out, &errOut, env(map[string]string{
		"BURNIN_DURATION_SECONDS":  "60",
		"BURNIN_STRESSAPPTEST_BIN": fakeSat(t, cleanRun, 0),
	}), root)
	if code != exitSkip {
		t.Fatalf("exit = %d (%s), want exitSkip", code, reason)
	}
	if got := marker(code, reason); !strings.HasPrefix(got, "STRESSAPPTEST_SKIP: ") {
		t.Errorf("marker = %q", got)
	}
}

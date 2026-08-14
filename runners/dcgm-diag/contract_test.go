package main

import (
	"bytes"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// These tests import the operator's own contract and parser packages, which is
// why the Dockerfile deletes *_test.go before building: the publish workflow
// pins the build context to this directory, so those packages are not there.
// They run from the repository root, in CI, where they belong — the point is to
// check the RUNNER against the PARSER, and a check that lived in only one of
// them would not notice the two drifting apart.

// The name a runner prints and the name a threshold is written against are
// joined by pkg/runner's alias table, and nothing reconciles a disagreement. A
// key that normalises to a name no profile references does not fail to parse;
// it parses to something no threshold matches, and verdict evaluation then
// fails closed on a healthy node.
func TestEmittedKeysSurviveTheParser(t *testing.T) {
	rep := newReport()
	for _, key := range emittedKeys {
		rep.set(key, "0")
	}
	var out bytes.Buffer
	rep.writeTo(&out)

	res := runner.Parse("dcgm-diag", out.String(), 0)

	if len(res.InvalidNames) > 0 {
		t.Fatalf("the parser rejected these names: %v", res.InvalidNames)
	}
	if len(res.Metrics) != len(emittedKeys) {
		t.Fatalf("emitted %d keys but the parser kept %d — two keys collided on one canonical name, and last-occurrence-wins would have discarded a real measurement",
			len(emittedKeys), len(res.Metrics))
	}
	for name := range res.Metrics {
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("canonical name %q breaks the metric grammar: %v", name, err)
		}
	}
}

// The four names the project owns for this TestKind. If one of these mappings
// changes, every profile thresholding on it silently stops matching.
func TestOwnedMetricsLandOnTheirRegisteredNames(t *testing.T) {
	want := map[string]string{
		keyTestsFailed:     "diagTestsFailed",
		keyPCIeReplayCount: "pcieReplayErrors",
		keyGPUTempC:        "gpuTempC",
		keyLastXidCode:     "lastXidCode",
		keyRowsRemapped:    "remappedRows",
		keyElapsedS:        "elapsedS",
	}

	rep := newReport()
	for key := range want {
		rep.set(key, "1")
	}
	var out bytes.Buffer
	rep.writeTo(&out)
	res := runner.Parse("dcgm-diag", out.String(), 0)

	for key, canonical := range want {
		if _, ok := res.Metrics[canonical]; !ok {
			t.Errorf("%q did not reach the canonical name %q; got %v", key, canonical, res.Metrics)
			continue
		}
		if _, registered := contract.Lookup(canonical); !registered {
			t.Errorf("%q is not registered in pkg/contract", canonical)
		}
	}
}

// DCGM reports correctable (SBE) and uncorrectable (DBE) ECC counts separately
// and they mean different things: a handful of SBEs is a working part, one DBE
// is a failing one. This runner keeps them as two numbers rather than emitting
// the registered eccErrors, because a single collapsed figure would either fail
// a healthy part on its correctable count or hide an uncorrectable one.
//
// The consequence is deliberate and belongs in a test rather than only in a
// comment: a profile that thresholds eccErrors against THIS TestKind gets a
// fails-closed FAILURE, because the metric is genuinely absent. Threshold
// eccDbeTotal (and eccSbeTotal) instead.
func TestECCStaysTwoNumbers(t *testing.T) {
	rep := newReport()
	rep.set(keyECCSbeTotal, "3")
	rep.set(keyECCDbeTotal, "0")
	var out bytes.Buffer
	rep.writeTo(&out)

	res := runner.Parse("dcgm-diag", out.String(), 0)

	if res.Metrics["eccSbeTotal"] != "3" || res.Metrics["eccDbeTotal"] != "0" {
		t.Fatalf("ECC counts did not survive as two metrics: %v", res.Metrics)
	}
	if v, present := res.Metrics["eccErrors"]; present {
		t.Errorf("eccErrors = %q: the correctable and uncorrectable counts were collapsed into one number", v)
	}
}

// The marker line carries prose, and prose can contain an "=". It must never be
// mistaken for a metric, or a failure's explanation would be recorded as a
// measurement and the run would lose its Message.
func TestMarkerLineIsNeverParsedAsAMetric(t *testing.T) {
	cases := []struct {
		marker, reason string
		wantCode       int
	}{
		{markerPass, "", exitPass},
		{markerFail, "1 of 4 DCGM subtests failed: memory — status=Fail, ecc=1", exitFail},
		{markerSkip, "DCGM will not run against this part: unsupported GPU", exitSkip},
		{markerError, "dcgmi exited 125 and produced no readable JSON result", exitError},
	}
	for _, tc := range cases {
		rep := newReport()
		rep.set(keyDiagLevel, "3")
		var out bytes.Buffer
		code := finish(&out, rep, tc.wantCode, tc.marker, tc.reason)

		if code != tc.wantCode {
			t.Errorf("finish returned %d, want %d", code, tc.wantCode)
		}
		res := runner.Parse("dcgm-diag", out.String(), code)
		if !strings.HasPrefix(res.Message, tc.marker) {
			t.Errorf("Message = %q, want it to start with %q", res.Message, tc.marker)
		}
		if len(res.Metrics) != 1 || res.Metrics["diagLevel"] != "3" {
			t.Errorf("marker text leaked into the metrics: %v", res.Metrics)
		}
	}
}

// Metrics come first on every path, including the failing ones — a run that
// ends without evidence says nothing about the node it just condemned.
func TestFinishPrintsMetricsBeforeTheVerdict(t *testing.T) {
	rep := newReport()
	rep.set(keyDiagLevel, "3")
	rep.set(keyTestsFailed, "2")
	rep.set(keyGPUTempC, "88")

	var out bytes.Buffer
	finish(&out, rep, exitFail, markerFail, "2 subtests failed")

	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 4 {
		t.Fatalf("got %d lines, want 3 metrics then the marker: %q", len(lines), out.String())
	}
	for _, line := range lines[:3] {
		if !strings.Contains(line, "=") {
			t.Errorf("line %q is not a metric", line)
		}
	}
	if !strings.HasPrefix(lines[3], markerFail) {
		t.Errorf("last line = %q, want the marker", lines[3])
	}
	// Insertion order, so two runs of the same test are diffable.
	if !strings.HasPrefix(lines[0], keyDiagLevel+"=") {
		t.Errorf("first line = %q, want %s", lines[0], keyDiagLevel)
	}
}

// A value containing a newline would be read by the parser as two lines, and
// the fragment would become the run's Message — losing the marker.
func TestReportValuesStayOnOneLine(t *testing.T) {
	rep := newReport()
	rep.set(keyReason, "DCGM said:\nUnable to run diagnostic\r\non unsupported GPU")

	var out bytes.Buffer
	finish(&out, rep, exitSkip, markerSkip, "unsupported")

	res := runner.Parse("dcgm-diag", out.String(), exitSkip)
	if res.Message != markerSkip+": unsupported" {
		t.Errorf("Message = %q; a multi-line value displaced the marker", res.Message)
	}
	if strings.Contains(res.Metrics["reason"], "\n") {
		t.Errorf("reason still contains a newline: %q", res.Metrics["reason"])
	}
}

// A truncated reason must still be valid UTF-8. The reasons this runner builds
// join with an em dash, so a byte-wise cut can land inside a multi-byte rune and
// put a replacement character into the one field a human reads to find out why a
// node failed.
func TestSanitizeTruncatesOnARuneBoundary(t *testing.T) {
	// Lands the 300-byte cut inside the em dash, wherever it happens to fall.
	for pad := 295; pad < 303; pad++ {
		in := strings.Repeat("a", pad) + "—" + strings.Repeat("b", 40)
		got := sanitize(in)
		if !utf8.ValidString(got) {
			t.Errorf("pad=%d: sanitize produced invalid UTF-8: %q", pad, got)
		}
	}
}

// Exit codes must map to the verdicts this runner documents.
func TestExitCodesMeanWhatTheContractSays(t *testing.T) {
	cases := map[int]runner.Verdict{
		exitPass:  runner.VerdictPass,
		exitFail:  runner.VerdictFail,
		exitSkip:  runner.VerdictSkip,
		exitError: runner.VerdictError,
	}
	for code, want := range cases {
		if got := runner.VerdictFor(code); got != want {
			t.Errorf("exit %d = %v, want %v", code, got, want)
		}
	}
}

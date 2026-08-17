// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
)

// The hardware half of #307, as a MATCHED PAIR.
//
// testdata/gb10-level2-plugins-{gated,allowed}.json were captured on 2026-08-16
// from spark-043a (NVIDIA GB10, sm_121, driver 580.82.09, DCGM 4.5.2) by running
// `dcgmi diag -r 2 -j` minutes apart against the GPU operator's host engine.
// dcgmi exited 226 on both. The ONLY difference between the two invocations is
// `-p memory.is_allowed=true;pcie.is_allowed=true`, which is what
// BURNIN_DCGM_ALLOW=memory,pcie expands to.
//
// Why these exist alongside #361's testdata/gb10-level2-partial-skip.json,
// which is the same shape: the value here is the CONTROLLED COMPARISON, and a
// comparison needs both halves from one session. #361's capture is spark-85a9
// on DCGM 4.2.3; pairing a 4.5.2 "allowed" document against it would leave the
// node and the DCGM version varying alongside the parameter, which is the one
// thing this pair exists to hold still. They also pin that 4.5.2 emits the same
// document shape 4.2.3 does — the hybrid tree #361 wrote down.
const (
	gb10PluginsGated  = "gb10-level2-plugins-gated.json"
	gb10PluginsAllowd = "gb10-level2-plugins-allowed.json"
)

// The measurement #307 asked for: without the parameter this part had no memory
// test and no PCIe test run against it; with it, both ran and passed.
func TestGB10PluginsRunOnlyWhenAllowed(t *testing.T) {
	statusOf := func(file, test string) status {
		t.Helper()
		st, ok := mustParse(t, testdataDoc(t, file)).tests[test]
		if !ok {
			t.Fatalf("%s: %q is not in the document at all", file, test)
		}
		return st
	}
	for _, test := range []string{"memory", "pcie"} {
		if got := statusOf(gb10PluginsGated, test); got != statusSkip {
			t.Errorf("without -p, %s = %v; DCGM's per-SKU allowlist should have gated it off", test, got)
		}
		if got := statusOf(gb10PluginsAllowd, test); got != statusPass {
			t.Errorf("with is_allowed=true, %s = %v, want Pass — the parameter did not enable it", test, got)
		}
	}
}

// Both halves are the same part on the same DCGM, so a difference in the
// provenance would mean the pair is not the controlled comparison it claims.
func TestGB10PluginPairIsOneConfiguration(t *testing.T) {
	gated := mustParse(t, testdataDoc(t, gb10PluginsGated))
	allowed := mustParse(t, testdataDoc(t, gb10PluginsAllowd))

	if gated.Version != allowed.Version {
		t.Errorf("DCGM version differs across the pair (%q vs %q), so the parameter is "+
			"not the only variable", gated.Version, allowed.Version)
	}
	if gated.DriverVersion != allowed.DriverVersion {
		t.Errorf("driver version differs across the pair (%q vs %q)",
			gated.DriverVersion, allowed.DriverVersion)
	}
	if got, want := gated.Version, "4.5.2"; got != want {
		t.Errorf("Version = %q, want %q — #361's capture is 4.2.3, and this pair is "+
			"here partly to cover the newer build", got, want)
	}
}

// A gated GB10 run must never be a pass: the node had no memory test and no
// PCIe test run against it.
func TestGB10GatedRunIsNeverAPass(t *testing.T) {
	doc := testdataDoc(t, gb10PluginsGated)
	code, out := runVerdict(t, doc, 226, nil, doc)
	if code == exitPass {
		t.Fatalf("a GB10 with memory and pcie skipped PASSED the DCGM gate\n%s", out)
	}
	if code != exitError {
		t.Fatalf("exit = %d, want %d (unjudged, never a hardware verdict)\n%s", code, exitError, out)
	}
}

// The plugins that never ran are NAMED on the result, whichever verdict branch
// wins.
//
// On this document the excused-NotRun branch wins, not the partial-skip branch:
// persistence mode is disabled on these machines, so the software subtest fails
// and every finding against it is a node setting (#304). The skipped names used
// to be recorded only inside the partial-skip branch, so the stored result for a
// real GB10 said "fix persistence mode" and carried no trace of the two plugins
// the allowlist had gated off.
func TestGB10GatedRunNamesTheSkippedPlugins(t *testing.T) {
	doc := testdataDoc(t, gb10PluginsGated)
	_, out := runVerdict(t, doc, 226, nil, doc)

	if !strings.Contains(out, keySkippedSubtests+"=") {
		t.Fatalf("the skipped plugins were not recorded at all:\n%s", out)
	}
	for _, plugin := range []string{"memory", "pcie"} {
		if !strings.Contains(out, plugin) {
			t.Errorf("%q is not named anywhere in the result:\n%s", plugin, out)
		}
	}
}

// DCGM skips these with NO reason — no warning, no info, just "status":"Skip".
// So the runner cannot quote DCGM here, and the allowlist has to be explained by
// the runner's own message or an operator is left with an unexplained Error.
func TestGB10SkipsCarryNoReasonFromDCGM(t *testing.T) {
	res := mustParse(t, testdataDoc(t, gb10PluginsGated))
	if len(res.skipReasons) != 0 {
		t.Fatalf("DCGM supplied skip reasons after all: %v — the runner's message could "+
			"quote them instead of explaining the allowlist itself", res.skipReasons)
	}
}

// Enabling the plugins does not manufacture a pass. The software subtest still
// fails on persistence mode, so the run is still unjudged — the parameter buys
// coverage, not a verdict.
func TestGB10AllowedRunStillReportsTheNodeSetting(t *testing.T) {
	doc := testdataDoc(t, gb10PluginsAllowd)
	code, out := runVerdict(t, doc, 226, nil, doc)
	if code != exitError {
		t.Fatalf("exit = %d, want %d\n%s", code, exitError, out)
	}
	if !strings.Contains(out, "Persistence mode") {
		t.Errorf("the node setting an operator has to fix is not reported:\n%s", out)
	}
}

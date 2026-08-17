// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// The parser and the classifier, exercised against REAL dcgmi documents.
//
// testdata/gb10-*.json was captured on 2026-08-17 from a GB10 DGX Spark
// (spark-85a9, driver 580.82.09, DCGM 4.2.3) by running
// `dcgmi diag -r <level> -j` against the host's own DCGM installation, during
// the hardware pass for #330. Every fixture in this package until now was
// hand-written, and that is exactly the gap a capture closes: a hand-typed
// sample proves the parser agrees with whoever typed it.
//
// It closed a real one immediately. Both hand-written DCGM 4.x fixtures in
// diagjson_test.go use the shape where "tests" is hoisted to the top level.
// What DCGM 4.2.3 actually emits on this fleet is a HYBRID nobody had written
// down — the 3.x-style "test_categories" tree, keyed "DCGM Diagnostic" rather
// than "DCGM GPU Diagnostic", holding 4.x-style entity results
// (entity_group/entity_id, not gpu_ids). The parser handles it, and nothing
// asserted that it did.
func testdataDoc(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	return string(b)
}

// TestParse_RealGB10Level1 pins the shape of a real level-1 document and the
// classification of the persistence-mode finding it carries.
//
// The persistence finding is the worked example behind #321: DCGM states in its
// own numeric vocabulary that this is a node setting, and the run must report
// the node UNJUDGED rather than condemn it for a configuration it can fix.
func TestParse_RealGB10Level1(t *testing.T) {
	res, err := parseDiagJSON(testdataDoc(t, "gb10-level1.json"))
	if err != nil {
		t.Fatalf("parseDiagJSON: %v", err)
	}

	if !res.SawResultStructure {
		t.Error("SawResultStructure is false on a document that plainly has results; " +
			"a Skip from here would take a node out of scope on our own blindness (#322)")
	}
	if got, want := res.Version, "4.2.3"; got != want {
		t.Errorf("Version = %q, want %q", got, want)
	}
	if got, want := res.DriverVersion, "580.82.09"; got != want {
		t.Errorf("DriverVersion = %q, want %q", got, want)
	}

	counts := res.Counts()
	if counts.Total != 1 {
		t.Fatalf("Total = %d, want 1 (the software subtest)", counts.Total)
	}
	if len(counts.ExcusedNames) != 1 || counts.ExcusedNames[0] != "software" {
		t.Errorf("ExcusedNames = %v, want [software]: the only finding is a node "+
			"setting, so the subtest must be excused rather than failed", counts.ExcusedNames)
	}
	if counts.Failed != 0 {
		t.Errorf("Failed = %d, want 0: a persistence-mode setting is not a hardware "+
			"verdict, and reporting it as one is the fail-open #321 fixed", counts.Failed)
	}
	if len(counts.ConfigFindings) == 0 {
		t.Error("ConfigFindings is empty; the persistence finding must be REPORTED, " +
			"not silently swallowed — an excusal nobody can see is indistinguishable " +
			"from a probe that never ran")
	}
}

// TestParse_RealGB10Level2PartialSkip pins the partial-skip document.
//
// Level 2 on GB10 gates two of its three plugins off. No fixture in tree had
// such a document, and it is the one #306/#307 turn on: a run where most of the
// suite never executed must NOT report as a pass over the part.
func TestParse_RealGB10Level2PartialSkip(t *testing.T) {
	res, err := parseDiagJSON(testdataDoc(t, "gb10-level2-partial-skip.json"))
	if err != nil {
		t.Fatalf("parseDiagJSON: %v", err)
	}

	counts := res.Counts()
	if counts.Total != 3 {
		t.Fatalf("Total = %d, want 3 (software, memory, pcie)", counts.Total)
	}
	if counts.Skipped != 2 {
		t.Errorf("Skipped = %d, want 2", counts.Skipped)
	}
	// Executed is 1, NOT 0, and that is the point rather than an oversight.
	// The software subtest was excused, and an excused subtest still counts as
	// executed (#304) — otherwise this document would reach Executed == 0 and
	// be reported SKIPPED, "not applicable to this part", which is a worse
	// claim than the Fail the excusal exists to avoid because it looks benign
	// while verifying nothing. A real capture is the right place to pin that,
	// since this is the shape of document where the two answers diverge.
	if counts.Executed != 1 {
		t.Errorf("Executed = %d, want 1 (the excused software subtest): at 0 this "+
			"document would report as a clean Skip over a part nothing measured",
			counts.Executed)
	}
	if len(counts.ExcusedNames) != 1 || counts.ExcusedNames[0] != "software" {
		t.Errorf("ExcusedNames = %v, want [software]", counts.ExcusedNames)
	}
	if counts.Passed != 0 {
		t.Errorf("Passed = %d, want 0: nothing in this document judged the hardware, "+
			"so no part of it may read as a pass", counts.Passed)
	}
	if len(counts.SkippedNames) != 2 {
		t.Errorf("SkippedNames = %v, want two names: %q does not tell an operator "+
			"WHICH plugin never ran", counts.SkippedNames, "2 skipped")
	}
}

// TestParse_RealGB10CarriesNumericClassificationFields is the regression guard
// for the assumption #321 rests on.
//
// The classifier prefers DCGM's numeric error_severity / error_category /
// error_id over prose matching. #330 asked whether those fields actually arrive
// as JSON numbers on a real build, or as strings — because if they were strings,
// intField's string case would be load-bearing rather than defensive. Measured
// on DCGM 4.2.3: they are numbers. This test fails if a future capture drifts,
// which is the moment to re-read that decision rather than the moment a fleet
// gets a wrong verdict.
//
// THE SLICE IS SEARCHED, NEVER INDEXED, and that is not a style preference. The
// software subtest carries TWO findings on this document, arriving by the two
// different routes findings.go describes: the persistence one is an object in a
// `warnings` array under `results` and carries the enums, the unreadable-SRAM
// one is a bare string in `test_summary.info` and carries nothing. They are
// appended by separate record() calls, and walk() reaches those two branches in
// Go's randomised map order — so `findings[0]` was the metadata-less one in
// roughly 1 run in 15, and the test failed claiming the numeric fields "were not
// read" on a document that plainly carries them. Assert what the fixture MEANS —
// exactly one finding bears metadata, and it is the persistence one.
func TestParse_RealGB10CarriesNumericClassificationFields(t *testing.T) {
	res, err := parseDiagJSON(testdataDoc(t, "gb10-level1.json"))
	if err != nil {
		t.Fatalf("parseDiagJSON: %v", err)
	}
	findings := res.findings["software"]
	if len(findings) != 2 {
		t.Fatalf("recorded %d findings for the software subtest, want 2 "+
			"(persistence mode with metadata, unreadable SRAM count without): %+v",
			len(findings), findings)
	}

	var withMeta []finding
	for _, f := range findings {
		if f.HasMeta {
			withMeta = append(withMeta, f)
		}
	}
	if len(withMeta) != 1 {
		t.Fatalf("%d of %d findings carry metadata, want exactly 1: %+v; "+
			"at 0 the numeric classification fields were not read and classification "+
			"silently fell back to matching English prose, and above 1 the bare-string "+
			"finding acquired enums DCGM never attached to it",
			len(withMeta), len(findings), findings)
	}

	f := withMeta[0]
	if f.Severity != 5 {
		t.Errorf("Severity = %d, want 5 (CONFIG)", f.Severity)
	}
	if f.Category != 8 {
		t.Errorf("Category = %d, want 8 (SOFTWARE_OTHER — note NOT SOFTWARE_CONFIG, "+
			"which is why the excusal turns on the error_id)", f.Category)
	}
	if f.ErrorID != 29 {
		t.Errorf("ErrorID = %d, want 29 (DCGM_FR_PERSISTENCE_MODE)", f.ErrorID)
	}
	if got := classify(f); got != findingConfiguration {
		t.Errorf("classify() = %v, want configuration", got)
	}
}

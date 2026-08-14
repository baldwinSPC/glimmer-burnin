// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// Every test in this file failed against the first draft of the metadata path,
// and each one describes a document DCGM really emits. They are grouped here
// rather than spread through findings_test.go because they share a single
// subject: the metadata may only ever make a verdict MORE conservative than the
// prose would have been. A change that reads DCGM's enums as a licence to
// excuse is the failure mode, and it is not hypothetical — the first draft
// excused a test that had been aborted before it measured anything, and a
// subtest DCGM had declined to run.

// THE INVARIANT, and the test that makes the numeric path load-bearing.
//
// Deleting the whole `if f.HasMeta` block from classify() leaves the rest of
// this package's suite green, because every other metadata row picks prose the
// text path already classifies identically. These rows do not: the text says
// one thing and DCGM's category says another, so they can only pass while the
// metadata is actually consulted.
func TestMetadataConstrainsTheProseAndNeverExcusesIt(t *testing.T) {
	rows := []struct {
		name string
		f    finding
		want findingClass
	}{
		// Prose that the text path excuses, on a category DCGM calls hardware.
		// This is the whole reason to read the enums at all.
		{"persistence-mode prose, HARDWARE_MEMORY category",
			finding{Text: "Persistence mode for GPU 0 is disabled", Severity: 5, Category: 10, HasMeta: true},
			findingHardware},
		{"unreadable-counter prose, HARDWARE_PCIE category",
			finding{Text: "Error getting the SRAM Threshold Count for GPU 0", Severity: 5, Category: 13, HasMeta: true},
			findingHardware},
		{"denylist prose, HARDWARE_THERMAL category",
			finding{Text: "GPU 0 is on the denylist", Severity: 5, Category: 9, HasMeta: true},
			findingHardware},
	}
	for _, r := range rows {
		if got := classify(r.f); got != r.want {
			t.Errorf("%s: class = %v, want %v", r.name, got, r.want)
		}
	}
}

// (severity, category) IS NOT A UNIQUE KEY, so it cannot be the whole rule.
//
// DCGM_FR_ABORTED carries the identical (5, 8) pair as the GB10 persistence
// finding this feature was built for. Excusing on that pair alone therefore
// excuses "Test was aborted early" — a test that measured NOTHING — as if it
// were a node setting, which is the precise shape of laundering a non-result
// into an acceptance.
func TestTheAmbiguousConfigPairsAreNotExcusedOnSeverityAlone(t *testing.T) {
	rows := []struct {
		name string
		f    finding
		want findingClass
	}{
		{"DCGM_FR_PERSISTENCE_MODE (id 29) is the verified node setting",
			finding{Text: "Persistence mode for GPU 0 is disabled", Severity: 5, Category: 8, ErrorID: 29, HasMeta: true},
			findingConfiguration},
		{"DCGM_FR_ABORTED (id 47) shares that pair and measured nothing",
			finding{Text: "Test was aborted early due to user signal", Severity: 5, Category: 8, ErrorID: 47, HasMeta: true},
			findingHardware},

		// DCGM's skipped-test entry, error_id 40 (CONFIG, SOFTWARE_CONFIG). Excusing this as a
		// node setting is the acceptance #306 refused: a subtest DCGM declined
		// to run is not a subtest that passed.
		{"DCGM's skipped-test entry is not a node setting",
			finding{Text: "the subtest did not run", Severity: 5, Category: 3, ErrorID: 40, HasMeta: true},
			findingHardware},

		// DCGM_FR_GPU_OP_MODE (MONITOR, SOFTWARE_CONFIG). The category alone
		// used to excuse it, through a severity this package says outright is
		// not excusable.
		{"MONITOR severity is not excused by a SOFTWARE_CONFIG category",
			finding{Text: "Skipping plugin due to GPU Operating Mode: LOW_DP", Severity: 1, Category: 3, ErrorID: 74, HasMeta: true},
			findingHardware},

		// The remaining severities, for completeness. NVIDIA files no
		// SOFTWARE_CONFIG entry at ISOLATE or RESET today, so these are
		// forward-looking rather than observed — which is the point: the rule
		// must not depend on the table staying as it is.
		{"ISOLATE severity is not excused by a SOFTWARE_CONFIG category",
			finding{Text: "unrecognised", Severity: 2, Category: 3, HasMeta: true},
			findingHardware},
		{"RESET severity is not excused by a SOFTWARE_CONFIG category",
			finding{Text: "unrecognised", Severity: 6, Category: 3, HasMeta: true},
			findingHardware},

		// An XID never reaches the excusal branch by severity — NVIDIA files
		// DCGM_FR_XID_ERROR under TRIAGE and SXID/UNCONTAINED under ISOLATE.
		// Pinned so nobody "fixes" the classifier towards excusing one.
		{"TRIAGE severity, as NVIDIA really files an XID",
			finding{Text: "XID 79: GPU has fallen off the bus", Severity: 4, Category: 5, HasMeta: true},
			findingHardware},

		// An unrecognised CONFIG-severity finding keeps its Fail. Having the
		// right severity is not the same as being a known node setting.
		{"CONFIG severity with prose nobody recognises stays hardware",
			finding{Text: "some future check nobody has seen", Severity: 5, Category: 8, ErrorID: 9999, HasMeta: true},
			findingHardware},
	}
	for _, r := range rows {
		if got := classify(r.f); got != r.want {
			t.Errorf("%s: class = %v, want %v", r.name, got, r.want)
		}
	}
}

// NOTHING MAY BE DROPPED ON THE WAY TO THE VERDICT.
//
// findings is the verdict's only input, so a shape findingsFrom does not
// understand is not a cosmetic gap — it removes the hardware finding from the
// evidence and lets the remaining, excusable findings carry the subtest.
func TestNoDocumentShapeLosesItsHardwareFinding(t *testing.T) {
	const hw = "Inforom is corrupted for GPU 0"
	docs := map[string]string{
		// Bare strings inside warnings[] — the pre-3.1 DCGM shape.
		"bare strings in warnings[]": `{"tests":[{"name":"software","results":[{"status":"Fail",
			"info":["Persistence Mode: Persistence mode for GPU 0 is disabled"],
			"warnings":["` + hw + `"]}]}]}`,

		// warnings is a single object rather than an array.
		"warnings as a lone object": `{"tests":[{"name":"software","results":[{"status":"Fail",
			"info":["Persistence Mode: Persistence mode for GPU 0 is disabled"],
			"warnings":{"warning":"` + hw + `"}}]}]}`,

		// warnings is a bare string.
		"warnings as a bare string": `{"tests":[{"name":"software","results":[{"status":"Fail",
			"info":["Persistence Mode: Persistence mode for GPU 0 is disabled"],
			"warnings":"` + hw + `"}]}]}`,

		// Prose nested one level deeper than the collector used to reach.
		"prose nested in an object": `{"tests":[{"name":"software","results":[{"status":"Fail",
			"info":{"outer":{"inner":"` + hw + `"}}}]}]}`,
	}
	for name, doc := range docs {
		res, err := parseDiagJSON(doc)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var texts []string
		for _, f := range res.findings["software"] {
			texts = append(texts, f.Text)
		}
		if !strings.Contains(strings.Join(texts, "|"), hw) {
			t.Errorf("%s: hardware finding was dropped; kept %v", name, texts)
		}
		if c := res.Counts(); c.Failed != 1 {
			t.Errorf("%s: Failed = %d, want 1 (excused = %v)", name, c.Failed, c.ExcusedNames)
		}
	}
}

// A finding whose PROSE is unreadable must not take its metadata down with it.
//
// The classification for these lives in the enums, not the sentence, so
// discarding the entry throws away exactly the ISOLATE / HARDWARE_MEMORY answer
// the metadata path exists to start trusting.
func TestAnUnreadableMessageDoesNotDiscardItsMetadata(t *testing.T) {
	const doc = `{"tests":[{"name":"software","results":[{"status":"Fail",
		"info":["Persistence Mode: Persistence mode for GPU 0 is disabled"],
		"warnings":[{"error_severity":2,"error_category":10,"error_id":91,
			"detail":"Uncorrectable ECC error detected on GPU 0"}]}]}]}`
	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	if c := res.Counts(); c.Failed != 1 {
		t.Fatalf("Failed = %d, want 1 (excused = %v)", c.Failed, c.ExcusedNames)
	}

	// And the same object with NO readable prose at all still classifies, by
	// its enums, and says so legibly rather than vanishing.
	const noProse = `{"tests":[{"name":"software","results":[{"status":"Fail",
		"warnings":[{"error_severity":2,"error_category":10,"error_id":91}]}]}]}`
	res2, err := parseDiagJSON(noProse)
	if err != nil {
		t.Fatal(err)
	}
	got := res2.findings["software"]
	if len(got) != 1 || !got[0].HasMeta || classify(got[0]) != findingHardware {
		t.Fatalf("metadata-only finding = %+v, want one HARDWARE finding", got)
	}
	if !strings.Contains(got[0].Text, "error_id 91") {
		t.Errorf("metadata-only finding text = %q, want it to name the error_id", got[0].Text)
	}
}

// Quoted enums must not silently disarm the hardware veto.
//
// A string-typed enum used to clear HasMeta, which sent the finding to the text
// path — where "Could not read ..." is excused as unreadable, even though DCGM
// had labelled it ISOLATE / HARDWARE_MEMORY.
func TestQuotedEnumsStillCarryTheHardwareVeto(t *testing.T) {
	const doc = `{"tests":[{"name":"memory","results":[{"status":"Fail",
		"warnings":[{"error_severity":"2","error_category":"10","error_id":"91",
			"warning":"Could not read the row remapping state for GPU 0"}]}]}]}`
	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	got := res.findings["memory"]
	if len(got) != 1 || !got[0].HasMeta {
		t.Fatalf("findings = %+v, want one finding carrying metadata", got)
	}
	if c := res.Counts(); c.Failed != 1 {
		t.Errorf("Failed = %d, want 1 (excused = %v)", c.Failed, c.ExcusedNames)
	}
}

// HasMeta must be computed as `sevOK && catOK`, THROUGH the parser.
//
// Existing coverage hand-builds a finding with HasMeta already set, which tests
// classify() rather than the rule that decides it — relaxing the `&&` to `||`
// left the whole suite green. The rule is what protects us from a pre-3.3 DCGM,
// whose severity enum is numbered differently and which has no category enum at
// all: without both fields we must not read the numbers as if they meant what
// the 4.2.3 table says.
func TestPartialMetadataIsNotAuthoritativeThroughTheParser(t *testing.T) {
	rows := []struct {
		name, warning string
		wantMeta      bool
	}{
		{"both enums present", `"error_severity":5,"error_category":8,"error_id":29`, true},
		{"severity only — the DCGM 3.2 shape", `"error_severity":1`, false},
		{"category only", `"error_category":10`, false},
		{"neither", `"error_id":29`, false},
		{"nulls", `"error_severity":null,"error_category":null`, false},
	}
	for _, r := range rows {
		doc := `{"tests":[{"name":"software","results":[{"status":"Fail",
			"warnings":[{"warning":"something happened",` + r.warning + `}]}]}]}`
		res, err := parseDiagJSON(doc)
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		got := res.findings["software"]
		if len(got) != 1 {
			t.Fatalf("%s: findings = %+v, want exactly one", r.name, got)
		}
		if got[0].HasMeta != r.wantMeta {
			t.Errorf("%s: HasMeta = %v, want %v", r.name, got[0].HasMeta, r.wantMeta)
		}
	}
}

// The display cap must not reach the verdict.
//
// Eight excusable findings followed by the hardware one is not a contrived
// document — it is an 8-GPU node with persistence mode off and one bad inforom.
func TestTheNinthFindingStillCondemnsTheNode(t *testing.T) {
	var infos []string
	for i := 0; i < 8; i++ {
		infos = append(infos, fmt.Sprintf(`"Persistence Mode: Persistence mode for GPU %d is disabled"`, i))
	}
	infos = append(infos, `"Inforom is corrupted for GPU 3"`)
	doc := `{"tests":[{"name":"software","results":[{"status":"Fail","info":[` +
		strings.Join(infos, ",") + `]}]}]}`

	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(res.findings["software"]); n != 9 {
		t.Errorf("kept %d of 9 findings", n)
	}
	if c := res.Counts(); c.Failed != 1 {
		t.Errorf("Failed = %d, want 1 (excused = %v)", c.Failed, c.ExcusedNames)
	}
}

// The finding named as the blocking one must not vary between runs.
//
// Go randomises map iteration, and excusedFindings returns on the first
// hardware finding it meets — so an unordered walk made the reported cause of a
// node's rejection differ on byte-identical input.
func TestTheBlockingFindingIsDeterministic(t *testing.T) {
	const doc = `{"tests":[{"name":"software","results":[{"status":"Fail",
		"info":{"a":"Inforom is corrupted for GPU 0","b":"Page retirement: GPU 0 has 64 retired pages"}}]}]}`
	seen := map[string]bool{}
	for i := 0; i < 40; i++ {
		res, err := parseDiagJSON(doc)
		if err != nil {
			t.Fatal(err)
		}
		seen[res.Counts().BlockingFinding] = true
	}
	if len(seen) != 1 {
		t.Errorf("blocking finding varied across runs: %v", seen)
	}
}

// A DOCUMENT WE CANNOT PARSE IS NOT A PART DCGM REFUSED TO TEST.
//
// Both statements used to arrive as Total == 0, and the runner then scanned the
// whole of stdout plus stderr for unsupported-part prose. So a document that
// reported real failures, but whose subtest statuses this runner could not
// read, became exit 2 SKIP — "not applicable to this hardware" — on the
// strength of a sentence that was never about the verdict. Skip is the one
// outcome nothing retries and nothing condemns (#322).
func TestAnUnreadableDocumentIsUnjudgedRatherThanNotApplicable(t *testing.T) {
	// DCGM reported subtests. The status key is spelled the way a future
	// release might, so nothing is recorded — and the output also carries a
	// perfectly ordinary sentence about one subsystem being unsupported.
	const doc = `{"version":"9.9.9","tests":[
		{"name":"memory","results":[{"gpu_id":0,"state":"Fail",
		  "warnings":[{"warning":"Uncorrectable ECC error detected on GPU 0",
		    "error_severity":2,"error_category":10}]}]}]}`
	const stderr = "NvLink is not supported by dcgm on this GPU\n"

	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	if c := res.Counts(); c.Total != 0 {
		t.Fatalf("Total = %d; this test is only meaningful when nothing was recorded", c.Total)
	}
	if !res.SawResultStructure {
		t.Fatal("SawResultStructure = false, but the document carries a non-empty tests[] array")
	}

	var out strings.Builder
	code := verdict(&out, newReport(), res, 226, nil, doc, stderr, time.Minute)
	if code != exitError {
		t.Errorf("exit = %d, want exitError (%d): an unreadable document must leave the hardware "+
			"unjudged, never mark it out of scope", code, exitError)
	}
	if strings.Contains(out.String(), markerSkip) {
		t.Errorf("output declares %s; a parser failure is not a hardware exemption:\n%s",
			markerSkip, out.String())
	}
}

// The genuine refusal still skips, which is the behaviour that must survive.
//
// DCGM produced no subtest structure at all and said why, so there is nothing
// to read and nothing to judge — the one case where "not applicable" is a claim
// about the hardware rather than about this runner.
func TestAGenuineUnsupportedPartStillSkips(t *testing.T) {
	const doc = `{"version":"4.2.3"}`
	const stderr = "Error: Unable to run diagnostic on unsupported GPU 0.\n"

	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	if res.SawResultStructure {
		t.Fatal("SawResultStructure = true on a document with no tests[] at all")
	}

	var out strings.Builder
	code := verdict(&out, newReport(), res, 1, nil, doc, stderr, time.Minute)
	if code != exitSkip {
		t.Errorf("exit = %d, want exitSkip (%d): a part DCGM refuses to test is out of scope",
			code, exitSkip)
	}
}

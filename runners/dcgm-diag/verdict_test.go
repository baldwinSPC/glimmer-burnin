package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"
)

func mustParse(t *testing.T, doc string) *diagResults {
	t.Helper()
	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatalf("fixture is not parseable: %v", err)
	}
	return res
}

func runVerdict(t *testing.T, doc string, code int, ctxErr error, text string) (int, string) {
	t.Helper()
	var out bytes.Buffer
	rc := verdict(&out, newReport(), mustParse(t, doc), code, ctxErr, text, "", 10*time.Minute)
	return rc, out.String()
}

func TestVerdict_AllPassing(t *testing.T) {
	code, out := runVerdict(t, dcgm3Pass, 0, nil, dcgm3Pass)
	if code != exitPass {
		t.Fatalf("exit = %d, want %d\n%s", code, exitPass, out)
	}
	if !strings.Contains(out, markerPass) {
		t.Errorf("output does not carry the pass marker:\n%s", out)
	}
}

// A failed subtest is a verdict about the hardware: exit 1, not 3.
func TestVerdict_FailedSubtestIsFailNotError(t *testing.T) {
	code, out := runVerdict(t, dcgm4Fail, 1, nil, dcgm4Fail)
	if code != exitFail {
		t.Fatalf("exit = %d, want %d\n%s", code, exitFail, out)
	}
	if !strings.Contains(out, "memory") {
		t.Errorf("the failing subtest is not named in the message:\n%s", out)
	}
}

// A suite that could only run some of its tests has not cleared the node. It
// reports Error — unjudged — rather than a pass that overstates the coverage.
func TestVerdict_PartialSuiteIsErrorNotPass(t *testing.T) {
	const doc = `{"tests":[
	  {"name":"software","results":[{"status":"pass"}]},
	  {"name":"targeted stress","results":[{"status":"Not Run"}]}]}`
	code, out := runVerdict(t, doc, 0, nil, doc)
	if code != exitError {
		t.Fatalf("exit = %d, want %d (unjudged, not passed)\n%s", code, exitError, out)
	}
}

// ...but a real failure alongside an incomplete one is still a failure. Partial
// coverage does not erase an observed fault.
func TestVerdict_FailOutranksNotRun(t *testing.T) {
	const doc = `{"tests":[
	  {"name":"memory","results":[{"status":"Fail"}]},
	  {"name":"targeted stress","results":[{"status":"Not Run"}]}]}`
	code, _ := runVerdict(t, doc, 1, nil, doc)
	if code != exitFail {
		t.Fatalf("exit = %d, want %d", code, exitFail)
	}
}

// DCGM ran and skipped everything: nothing was established about the part, so
// it is not applicable, not passed. This is the GB10 path.
func TestVerdict_EverythingSkippedIsSkip(t *testing.T) {
	const doc = `{"tests":[
	  {"name":"memory","results":[{"status":"Skip","info":["not supported on this device"]}]},
	  {"name":"pcie","results":[{"status":"Skip"}]}]}`
	code, out := runVerdict(t, doc, 0, nil, doc)
	if code != exitSkip {
		t.Fatalf("exit = %d, want %d\n%s", code, exitSkip, out)
	}
	if !strings.Contains(out, "not supported on this device") {
		t.Errorf("DCGM's own explanation was dropped:\n%s", out)
	}
}

// A document with no subtests plus DCGM's unsupported-part wording is the other
// GB10 shape: dcgmi refuses before it runs anything.
func TestVerdict_UnsupportedPartIsSkip(t *testing.T) {
	const doc = `{"DCGM GPU Diagnostic":{"version":"4.2.3"}}`
	code, out := runVerdict(t, doc, 1, nil,
		"Error: Unable to run diagnostic on unsupported GPU 0.\n"+doc)
	if code != exitSkip {
		t.Fatalf("exit = %d, want %d\n%s", code, exitSkip, out)
	}
	if !strings.Contains(out, "unsupported GPU") {
		t.Errorf("the skip does not quote DCGM's reason:\n%s", out)
	}
}

// No subtests and no explanation is NOT a skip. Excusing a node from acceptance
// requires a reason; without one the honest answer is that we do not know.
func TestVerdict_NoResultsWithoutAReasonIsError(t *testing.T) {
	const doc = `{"DCGM GPU Diagnostic":{"version":"4.2.3"}}`
	code, _ := runVerdict(t, doc, 1, nil, doc)
	if code != exitError {
		t.Fatalf("exit = %d, want %d", code, exitError)
	}
}

// A diagnostic cut short cannot be read as a pass, however many subtests had
// passed by the time the clock ran out.
func TestVerdict_TimeoutIsErrorEvenWhenEverythingSoFarPassed(t *testing.T) {
	code, out := runVerdict(t, dcgm3Pass, 0, context.DeadlineExceeded, dcgm3Pass)
	if code != exitError {
		t.Fatalf("exit = %d, want %d\n%s", code, exitError, out)
	}
	if !strings.Contains(out, "unjudged") {
		t.Errorf("the message does not say the hardware is unjudged:\n%s", out)
	}
}

// dcgmi exiting non-zero over an all-pass document is a contradiction, and a
// contradiction is not evidence of health.
func TestVerdict_NonZeroExitWithAllPassIsError(t *testing.T) {
	code, out := runVerdict(t, dcgm3Pass, 4, nil, dcgm3Pass)
	if code != exitError {
		t.Fatalf("exit = %d, want %d\n%s", code, exitError, out)
	}
}

// A DCGM warning is an observation, not a failure. It is counted and left for
// the profile's thresholds; promoting it here would take the decision away from
// the person who wrote the acceptance criteria.
func TestVerdict_WarningPassesAndIsCounted(t *testing.T) {
	const doc = `{"tests":[
	  {"name":"pcie","results":[{"status":"Warn","warnings":["bandwidth below expected"]}]},
	  {"name":"software","results":[{"status":"Pass"}]}]}`
	var out bytes.Buffer
	rep := newReport()
	res := mustParse(t, doc)
	reportDiag(rep, res)
	code := verdict(&out, rep, res, 0, nil, doc, "", time.Minute)

	if code != exitPass {
		t.Fatalf("exit = %d, want %d\n%s", code, exitPass, out.String())
	}
	if rep.vals[keyTestsWarned] != "1" {
		t.Errorf("%s = %q, want 1", keyTestsWarned, rep.vals[keyTestsWarned])
	}
}

// dcgmi exits 226, which is not a runner-contract code at all, and the parsed
// document says exactly one determinate thing — its `software` subtest failed.
//
// THIS TEST USED TO REQUIRE exit 1, on the argument that reporting Error would
// "convert a completely determinate diagnostic result into we-don't-know". That
// argument is right about determinacy and wrong about what was determined:
// running this on a healthy GB10 (issue #304) showed the two findings behind
// that failure are persistence mode being off and a field DCGM could not read.
// The result IS determinate — determinately a node that is not configured — and
// the node's silicon was never judged either way.
//
// So the verdict is Error, and the distinction is not academic. Fail is the one
// phase never retried and it settles the test permanently: at exit 1 this
// suite condemns 100% of a stock Spark fleet for something `nvidia-smi -pm 1`
// fixes. Error costs a retry that will not help and then reports honestly,
// which is the cheaper mistake by a wide margin.
//
// What must NOT happen is the message becoming vague. It names the setting, and
// this test still checks that.
func TestVerdict_GB10PersistenceModeIsUnjudgedRatherThanAHardwareFail(t *testing.T) {
	code, out := runVerdict(t, dcgm4GB10, 226, nil, dcgm4GB10)
	if code != exitError {
		t.Fatalf("exit = %d, want %d — a node setting is not a hardware verdict (#304)\n%s",
			code, exitError, out)
	}
	if !strings.Contains(out, "Persistence mode") {
		t.Errorf("the message does not say what the operator has to fix:\n%s", out)
	}
	if !strings.Contains(out, "UNJUDGED") {
		t.Errorf("the message must say the hardware was not judged, not merely that something failed:\n%s", out)
	}
	// The findings still reach the result — excusing them from the verdict must
	// not delete them from the report.
	if !strings.Contains(out, "diag_config_findings=") {
		t.Errorf("the config finding was dropped from the metrics:\n%s", out)
	}
}

// #371: on a real GB10 the excused-NotRun branch above wins EVERY TIME
// (persistence mode is off on these machines), which used to mean a reader
// never learned that DCGM's per-SKU allowlist separately skipped the memory
// and PCIe plugins outright — the message said "fix persistence mode" and
// stayed silent about two subtests never having run at all. This is the exact
// document (testdata/gb10-level2-plugins-gated.json, DCGM 4.5.2, spark-043a)
// the issue was filed against: composed, not reordered, so the persistence
// finding is still the primary reason and the skipped subtests are named
// alongside it rather than replacing it.
func TestVerdict_ExcusedNotRunAlsoNamesOutrightSkips(t *testing.T) {
	doc := testdataDoc(t, gb10PluginsGated)
	code, out := runVerdict(t, doc, 226, nil, doc)
	if code != exitError {
		t.Fatalf("exit = %d, want %d\n%s", code, exitError, out)
	}
	if !strings.Contains(out, "Persistence mode") {
		t.Errorf("the excused finding — still the primary reason — was dropped from the message:\n%s", out)
	}
	if !strings.Contains(out, "UNJUDGED") {
		t.Errorf("the message must say the hardware was not judged:\n%s", out)
	}
	for _, want := range []string{"SKIPPED", "memory", "pcie", "BURNIN_DCGM_ALLOW"} {
		if !strings.Contains(out, want) {
			t.Errorf("message does not mention %q — a reader would not learn that DCGM's per-SKU "+
				"allowlist skipped subtests outright, only that persistence mode needs fixing:\n%s", want, out)
		}
	}
}

// dcgmi parses argv[1] as the subsystem name. Putting --host first is rejected
// with "ERROR: Invalid subsystem" and exit 255 — no JSON, no results, and an
// Error verdict on every node the runner is ever scheduled on. Verified against
// DCGM 4.2.3; this test is what keeps the ordering from being "tidied".
func TestDcgmiArgs_SubsystemComesFirst(t *testing.T) {
	for _, sub := range []string{"diag", "discovery", "dmon"} {
		args := dcgmiArgs(sub, "127.0.0.1:5555", "-c", "1")
		if len(args) < 3 || args[0] != sub {
			t.Fatalf("%s: args = %v, want the subsystem first", sub, args)
		}
		if args[1] != "--host" || args[2] != "127.0.0.1:5555" {
			t.Errorf("%s: args = %v, want --host and the address after the subsystem", sub, args)
		}
		if args[len(args)-2] != "-c" || args[len(args)-1] != "1" {
			t.Errorf("%s: args = %v, want the caller's arguments preserved at the end", sub, args)
		}
	}
}

// tests_run counts what actually judged hardware; tests_failed counts subtests,
// not verdicts. Reporting a non-zero runner exit with zero failed subtests is
// how a caller tells "the suite could not run" from "the part is bad".
func TestReportDiag_Counts(t *testing.T) {
	rep := newReport()
	reportDiag(rep, mustParse(t, dcgm4Fail))

	want := map[string]string{
		keyTestsRun:     "3",
		keyTestsFailed:  "1",
		keyTestsWarned:  "0",
		keyTestsNotRun:  "0",
		keyTestsSkipped: "0",
		keyDcgmVersion:  "4.2.3",
	}
	for k, v := range want {
		if rep.vals[k] != v {
			t.Errorf("%s = %q, want %q", k, rep.vals[k], v)
		}
	}
}

// partialSkipDoc is the shape a DCGM level-2+ run takes on a part that is not
// on DCGM's per-SKU plugin allowlist: the deployment check runs and passes, and
// every CUDA-backed plugin is skipped. Measured on GB10 (issue #304 follow-up).
const partialSkipDoc = `{
 "DCGM Diagnostic": {
  "test_categories": [
   {"category": "Deployment", "tests": [
     {"name": "software", "results": [{"status": "Pass"}]}
   ]},
   {"category": "Hardware", "tests": [
     {"name": "pcie",   "results": [{"status": "Skip", "info": ["Plugin is not allowed on this SKU"]}]},
     {"name": "memory", "results": [{"status": "Skip", "info": ["Plugin is not allowed on this SKU"]}]}
   ]},
   {"category": "Stress", "tests": [
     {"name": "targeted_stress", "results": [{"status": "Skip"}]},
     {"name": "sm_stress",       "results": [{"status": "Skip"}]}
   ]}
  ]
 }
}`

// A NODE DCGM DECLINED TO TEST MUST NOT PASS THE DCGM GATE.
//
// The verdict checked Executed==0, Failed and NotRun and then fell through to
// Pass, with no rule for Skipped. So this document — one subtest executed and
// passed, four skipped — reported exit 0 on a node that had no memory test, no
// PCIe test and no stress test run against it.
//
// That is the false negative the exit-2 skip path exists to prevent, arriving
// through the pass path instead, and it is the ordinary case on any part
// missing from DCGM's plugin allowlist rather than an exotic one.
func TestVerdict_APartlySkippedSuiteIsNotAPass(t *testing.T) {
	code, out := runVerdict(t, partialSkipDoc, 0, nil, partialSkipDoc)
	if code == exitPass {
		t.Fatalf("a node with 4 of 5 subtests skipped PASSED the DCGM gate\n%s", out)
	}
	// Error, not Fail: DCGM declining to test something says nothing about the
	// silicon, so it must never condemn the part.
	if code != exitError {
		t.Fatalf("exit = %d, want %d (unjudged, not a hardware verdict)\n%s", code, exitError, out)
	}
	for _, want := range []string{"SKIPPED", "partly checked", "is_allowed"} {
		if !strings.Contains(out, want) {
			t.Errorf("the message should mention %q so an operator can act on it:\n%s", want, out)
		}
	}
	// The skipped subtests are NAMED. "4 skipped" does not tell anyone whether
	// the memory test or the PCIe test is the one that never ran.
	if !strings.Contains(out, "diag_skipped_subtests=") {
		t.Errorf("the skipped subtests were not recorded:\n%s", out)
	}
}

// The all-skipped case must stay a SKIP, not become an Error.
//
// "This kind does not apply to this part" is a different statement from "this
// part is half-checked", and collapsing them would make a genuinely
// inapplicable suite look like a machinery fault on every node in a fleet.
func TestVerdict_AnAllSkippedSuiteIsStillNotApplicable(t *testing.T) {
	doc := `{"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [
	  {"name": "pcie", "results": [{"status": "Skip"}]}]}]}}`
	code, out := runVerdict(t, doc, 0, nil, doc)
	if code != exitSkip {
		t.Fatalf("exit = %d, want %d — a suite that applies to nothing here is a Skip, "+
			"not a partial check\n%s", code, exitSkip, out)
	}
}

// A clean run still passes. The new rule must not make every node unjudged.
func TestVerdict_AFullyExecutedCleanRunStillPasses(t *testing.T) {
	doc := `{"DCGM Diagnostic": {"test_categories": [{"category": "Deployment", "tests": [
	  {"name": "software", "results": [{"status": "Pass"}]},
	  {"name": "pcie",     "results": [{"status": "Pass"}]}]}]}}`
	code, out := runVerdict(t, doc, 0, nil, doc)
	if code != exitPass {
		t.Fatalf("exit = %d, want %d — nothing skipped, nothing failed\n%s", code, exitPass, out)
	}
}

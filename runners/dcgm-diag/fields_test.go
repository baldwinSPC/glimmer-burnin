package main

import (
	"strings"
	"testing"
)

func TestParseDmonSample(t *testing.T) {
	const want = 7 // len(sampledFields)

	gpu, vals, ok := parseDmonSample("GPU 0      61      0      0      0      0      0      0", want)
	if !ok {
		t.Fatal("a well-formed dmon row was refused")
	}
	if gpu != "0" {
		t.Errorf("gpu = %q, want 0", gpu)
	}
	if len(vals) != want {
		t.Fatalf("got %d values, want %d", len(vals), want)
	}
	if !vals[0].ok || vals[0].value != 61 {
		t.Errorf("first column = %+v, want 61", vals[0])
	}
}

// "N/A" is DCGM saying the part does not expose the field. It must land as a
// missing measurement, never as a zero: a zero is a claim, and verdict
// evaluation fails closed on an absent metric precisely so that an unmeasured
// counter cannot satisfy a threshold.
func TestParseDmonSample_NotAvailableIsNotZero(t *testing.T) {
	_, vals, ok := parseDmonSample("GPU 0   N/A   0   0   0   0   0   0", 7)
	if !ok {
		t.Fatal("row refused")
	}
	if vals[0].ok {
		t.Errorf("N/A parsed as the value %v; it is a missing measurement", vals[0].value)
	}
	if !vals[1].ok {
		t.Error("a real value after an N/A column was lost")
	}
}

// A row with the wrong number of columns is dropped rather than read
// positionally. A single shifted column would report the PCIe replay counter as
// a temperature — both are small integers, and nothing downstream could catch
// the substitution.
func TestParseDmonSample_RefusesMisalignedRows(t *testing.T) {
	cases := []string{
		"GPU 0   61   0   0",
		"GPU 0   61   0   0   0   0   0   0   0",
		"#Entity  TMPTR  XIDER",
		"     Id",
		"",
		"Error: Host engine connection invalid/disconnected",
	}
	for _, line := range cases {
		if _, _, ok := parseDmonSample(line, 7); ok {
			t.Errorf("parseDmonSample(%q) accepted a row it should have dropped", line)
		}
	}
}

const dmonTwoGPUs = `#Entity   TMPTR   XIDER   PCIRP   ECCSB   ECCDB   RRUNC   RRCOR
     Id
GPU 0        58       0       4       2       0       0       0
GPU 1        61       0       7       3       0       1       0
`

func TestSampler_MaxAcrossTheWindowAndDeltaAcrossTheTest(t *testing.T) {
	s := newSampler(config{})
	s.observeAll(dmonTwoGPUs)
	// A hotter, busier second reading: temperature peaks at 74, and the
	// counters move by 3 replays, 1 SBE and 1 remapped row in total.
	s.observeAll(`GPU 0        74       0       6       2       0       0       0
GPU 1        66       1       8       4       0       1       1
`)

	rep := newReport()
	s.report(rep)

	if got := rep.vals[keyGPUTempC]; got != "74" {
		t.Errorf("%s = %q, want 74 (the peak, not the last reading)", keyGPUTempC, got)
	}
	if got := rep.vals[keyPCIeReplayCount]; got != "3" {
		t.Errorf("%s = %q, want 3 (movement during the test, not the lifetime total)",
			keyPCIeReplayCount, got)
	}
	if got := rep.vals[keyECCSbeTotal]; got != "1" {
		t.Errorf("%s = %q, want 1", keyECCSbeTotal, got)
	}
	if got := rep.vals[keyLastXidCode]; got != "1" {
		t.Errorf("%s = %q, want 1", keyLastXidCode, got)
	}
	// Correctable and uncorrectable remapped rows are one quantity, summed.
	if got := rep.vals[keyRowsRemapped]; got != "1" {
		t.Errorf("%s = %q, want 1", keyRowsRemapped, got)
	}
	if got := rep.vals[keyGPUCount]; got != "2" {
		t.Errorf("%s = %q, want 2", keyGPUCount, got)
	}
}

// One reading cannot establish that a counter did not move. Reporting "0" from
// a single sample would be a fabricated measurement that satisfies a
// `pcieReplayErrors <= 0` threshold without anything having been observed.
func TestSampler_SingleSampleEmitsNoDeltas(t *testing.T) {
	s := newSampler(config{})
	s.observeAll(dmonTwoGPUs)

	rep := newReport()
	s.report(rep)

	for _, key := range []string{keyPCIeReplayCount, keyECCSbeTotal, keyECCDbeTotal, keyRowsRemapped} {
		if v, present := rep.vals[key]; present {
			t.Errorf("%s = %q from a single sample; a delta needs two readings", key, v)
		}
	}
	// The peak temperature IS establishable from one reading, so it stays.
	if _, present := rep.vals[keyGPUTempC]; !present {
		t.Errorf("%s should still be reported from a single sample", keyGPUTempC)
	}
	// And so is the Xid CODE, which is the point of reading field 230 as a
	// state rather than a counter. One reading answers "which Xid does this
	// device report" completely; it was only the subtraction that needed two,
	// and the subtraction was never a count of anything (#311).
	if _, present := rep.vals[keyLastXidCode]; !present {
		t.Errorf("%s should be reported from a single sample: it is a state, not a delta", keyLastXidCode)
	}
}

// FIELD 230 IS A CODE, AND MUST NOT BE SUBTRACTED.
//
// The two rows below are the case that made this a false PASS rather than a
// cosmetic naming problem: a GPU sitting at Xid 79 for the whole window. Read
// as a delta it reported 79-79 = 0 and satisfied `xidEvents Equal 0`; read as
// a state it reports 79, and nothing gates on it.
func TestSampler_APersistentXidIsReportedNotSubtractedAway(t *testing.T) {
	s := newSampler(config{})
	s.observeAll("GPU 0 45 79 100 12 4 8 2\n")
	s.observeAll("GPU 0 46 79 100 12 4 8 2\n")

	rep := newReport()
	s.report(rep)

	if got := rep.vals[keyLastXidCode]; got != "79" {
		t.Errorf("%s = %q, want \"79\" — the code the device actually reports", keyLastXidCode, got)
	}
	// And the name that used to carry the subtraction is gone, so no profile
	// can gate a count on this runner's output by accident.
	if v, present := rep.vals["xid_errors"]; present {
		t.Errorf("xid_errors = %q; this runner must not publish a count it cannot derive", v)
	}
}

// A lifetime counter that goes backwards means the driver was reloaded or the
// device was reset mid-test. Report no movement and say so, rather than a
// negative count nothing can threshold.
func TestSampler_CounterResetIsFlaggedNotNegative(t *testing.T) {
	s := newSampler(config{})
	s.observeAll("GPU 0   58   0   9   0   0   0   0\n")
	s.observeAll("GPU 0   58   0   1   0   0   0   0\n")

	rep := newReport()
	s.report(rep)

	if got := rep.vals[keyPCIeReplayCount]; got != "0" {
		t.Errorf("%s = %q, want 0", keyPCIeReplayCount, got)
	}
	if rep.vals[keyCounterReset] != "true" {
		t.Errorf("%s was not set; a silent clamp hides that the device reset", keyCounterReset)
	}
}

// Every sampled field's id is asserted against DCGM's own dcgm_fields.h at
// image build time. This guards the machinery that assertion depends on: the
// symbols must be present, non-empty and unique, or the shell loop in the
// Dockerfile silently checks nothing.
func TestPrintFieldIDs_IsAssertableAgainstTheDCGMHeader(t *testing.T) {
	var sb strings.Builder
	printFieldIDs(&sb)

	lines := strings.Split(strings.TrimSpace(sb.String()), "\n")
	if len(lines) != len(sampledFields) {
		t.Fatalf("printed %d lines for %d fields", len(lines), len(sampledFields))
	}
	seen := map[string]bool{}
	for _, line := range lines {
		sym, id, found := strings.Cut(line, "=")
		if !found || sym == "" || id == "" {
			t.Fatalf("line %q is not SYMBOL=ID", line)
		}
		if !strings.HasPrefix(sym, "DCGM_FI_") {
			t.Errorf("%q is not a DCGM field symbol; the build assertion greps dcgm_fields.h for it", sym)
		}
		if seen[sym] {
			t.Errorf("%s is listed twice", sym)
		}
		seen[sym] = true
	}
}

func TestSampledFields_KeysAreDeclared(t *testing.T) {
	declared := map[string]bool{}
	for _, k := range emittedKeys {
		declared[k] = true
	}
	for _, f := range sampledFields {
		if !declared[f.key] {
			t.Errorf("field %s reports under %q, which is not in emittedKeys, so no test checks that it survives the parser", f.sym, f.key)
		}
	}
}

// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
)

// Every signature below is a message this project has SEEN on hardware, and the
// classification is what decides whether a node gets condemned. Verified on a
// GB10 DGX Spark, 2026-08-10, DCGM 4.2.3 — issue #304.
func TestFindingsAreClassifiedByWhatTheyClaim(t *testing.T) {
	rows := []struct {
		name string
		text string
		want findingClass
	}{
		{"persistence mode, measured on GB10",
			`Persistence Mode: Persistence mode for GPU 0 is disabled. Enable persistence mode by running "nvidia-smi -i <gpuId> -pm 1 " as root.`,
			findingConfiguration},
		{"nvvs cannot write its scratch file, measured on GB10",
			"Permissions and OS Blocks: No permission to create a file in directory '/'",
			findingConfiguration},
		{"a field DCGM could not read, measured on GB10",
			"Error getting the SRAM Threshold Count for GPU 0. Status = 0. Value = 9223372036854775794.",
			findingUnreadable},

		// The ones that MUST keep their verdict. A page-retirement or inforom
		// finding lives in the same `software` subtest as persistence mode, so
		// excusing by subtest name rather than by finding would have silently
		// swallowed these.
		{"page retirement is the hardware",
			"Page retirement: GPU 0 has 64 retired pages, which exceeds the threshold",
			findingHardware},
		{"inforom corruption is the hardware",
			"Inforom is corrupted for GPU 0",
			findingHardware},
		{"an unfamiliar message fails closed",
			"Some future DCGM check nobody here has seen",
			findingHardware},
	}
	for _, r := range rows {
		if got := classifyFinding(r.text); got != r.want {
			t.Errorf("%s: class = %v, want %v", r.name, got, r.want)
		}
	}
}

func TestOneHardwareFindingKeepsTheWholeSubtestFailing(t *testing.T) {
	// The rule that makes excusal safe. A subtest that failed for two reasons,
	// one of them the hardware, has failed — and the blocking finding is named
	// so an operator is not left guessing which of them decided it.
	_, _, blocking, excused := excusedFindings([]finding{
		{Text: "Persistence Mode: disabled"},
		{Text: "Inforom is corrupted for GPU 0"},
	})
	if excused {
		t.Fatal("a subtest with a hardware finding was excused")
	}
	if !strings.Contains(blocking, "Inforom") {
		t.Errorf("blocking = %q, want the inforom finding named", blocking)
	}
}

func TestAFailureWithNoStatedReasonIsNeverExcused(t *testing.T) {
	// Absence is not a declaration. DCGM failing something without saying why
	// is the one case where there is nothing to classify, and guessing would be
	// the fail-open this whole mechanism exists to avoid.
	_, _, _, excused := excusedFindings(nil)
	if excused {
		t.Error("a reasonless failure was excused")
	}
}

// The root-cause fix, not the excusal: nvvs needs somewhere writable, and
// without --home-dir it reports a FAILED subtest on a healthy GPU because it
// could not create a file in "/". The image runs as uid 65532.
func TestTheHostEngineIsGivenAWritableHome(t *testing.T) {
	c, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	joined := strings.Join(c.hostEngineArgs, " ")
	if !strings.Contains(joined, "--home-dir") {
		t.Errorf("nv-hostengine args %q lack --home-dir; in a container nvvs cannot write to \"/\" "+
			"and DCGM reports that as a hardware failure (#304)", joined)
	}
}

// DCGM'S OWN CLASSIFICATION OUTRANKS OUR READING OF A SENTENCE.
//
// Values are from dcgm_errors.h. The persistence case is the one measured on
// GB10 — severity 5 (CONFIG), category 8 (SOFTWARE_OTHER), error_id 29 — which
// is DCGM stating in its own vocabulary what #304 had to infer from wording.
func TestDCGMsOwnSeverityDecidesWhereItIsGiven(t *testing.T) {
	rows := []struct {
		name string
		f    finding
		want findingClass
	}{
		{"CONFIG severity, measured on GB10",
			finding{Text: "Persistence Mode: ... disabled", Severity: 5, Category: 8, ErrorID: 29, HasMeta: true},
			findingConfiguration},
		{"ISOLATE severity is the part",
			finding{Text: "anything at all", Severity: 2, Category: 10, HasMeta: true},
			findingHardware},
		{"RESET severity is the part",
			finding{Text: "anything at all", Severity: 6, Category: 15, HasMeta: true},
			findingHardware},

		// The composition that matters. A severity that would excuse, on a
		// category DCGM says is hardware, must NOT be excused — otherwise a
		// memory fault marked configurable walks straight through.
		{"CONFIG severity cannot excuse a HARDWARE_MEMORY category",
			finding{Text: "plausible-sounding config message", Severity: 5, Category: 10, HasMeta: true},
			findingHardware},
		{"CONFIG severity cannot excuse a HARDWARE_POWER category",
			finding{Text: "plausible-sounding config message", Severity: 5, Category: 14, HasMeta: true},
			findingHardware},

		// UNKNOWN means DCGM does not recognise the code. That is the
		// definition of something we must not wave through.
		{"UNKNOWN severity fails closed",
			finding{Text: "some future check", Severity: 3, Category: 16, HasMeta: true},
			findingHardware},
		// MONITOR is an observation about the part, not a node setting.
		{"MONITOR severity is not excused",
			finding{Text: "watch this", Severity: 1, Category: 2, HasMeta: true},
			findingHardware},
	}
	for _, r := range rows {
		if got := classify(r.f); got != r.want {
			t.Errorf("%s: class = %v, want %v", r.name, got, r.want)
		}
	}
}

// Where DCGM gives no enums, the text rules still apply — and they must, because
// `test_summary.info` entries are bare strings. The unreadable-SRAM finding
// measured on GB10 arrives exactly that way.
func TestTextRulesStillApplyWhereDCGMGivesNoMetadata(t *testing.T) {
	f := finding{Text: "Error getting the SRAM Threshold Count for GPU 0. Status = 0. Value = 9223372036854775794."}
	if got := classify(f); got != findingUnreadable {
		t.Errorf("class = %v, want findingUnreadable — this finding has no enums attached", got)
	}
	if got := classify(finding{Text: "Inforom is corrupted"}); got != findingHardware {
		t.Error("an unrecognised message with no metadata must fail closed to hardware")
	}
}

// Half a classification is worse than none, because it looks authoritative.
func TestPartialMetadataIsNotTreatedAsAuthoritative(t *testing.T) {
	// Severity present, category absent: HasMeta must be false, so the text
	// path decides rather than a severity that cannot be composed with the
	// hardware-category veto.
	f := finding{Text: "Inforom is corrupted for GPU 0", Severity: 5, HasMeta: false}
	if got := classify(f); got != findingHardware {
		t.Errorf("class = %v, want findingHardware — a lone severity must not excuse", got)
	}
}

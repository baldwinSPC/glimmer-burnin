// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"strings"
	"testing"
)

// feedAll parses a whole capture the way runStressapptest does, line by line.
func feedAll(t *testing.T, out string) satResult {
	t.Helper()
	var res satResult
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue
		}
		res.feed(line)
	}
	return res
}

// A clean run on a large node. Timestamps and pids are present because
// stressapptest always prints them, and nothing in the parser may be anchored
// to the start of a line.
const cleanRun = `2026/08/02-11:00:00(7) Log: Commandline - stressapptest -M 90112 -s 540 -m 20 -v 8
2026/08/02-11:00:00(7) Stats: SAT revision 1.0.11, 64 bit binary
2026/08/02-11:00:00(7) Log: Total 131072 MB. Free 128000 MB. Hugepages 0 MB. Targeting 90112 MB
2026/08/02-11:09:00(7) Stats: Found 0 hardware incidents
2026/08/02-11:09:00(7) Stats: Completed: 4718592.00M in 540.02s 8737.85MB/s, with 0 hardware incidents, 0 errors
2026/08/02-11:09:00(7) Stats: Memory Copy: 4718592.00M at 8737.85MB/s
2026/08/02-11:09:00(7) Stats: File Copy: 0.00M at 0.00MB/s
2026/08/02-11:09:00(7) Stats: Net Copy: 0.00M at 0.00MB/s
2026/08/02-11:09:00(7) Stats: Data Check: 0.00M at 0.00MB/s
2026/08/02-11:09:00(7) Stats: Invert Data: 0.00M at 0.00MB/s
2026/08/02-11:09:00(7) Stats: Disk: 0.00M at 0.00MB/s
2026/08/02-11:09:00(7) Status: PASS - please verify no corrected errors`

func TestParseCleanRun(t *testing.T) {
	res := feedAll(t, cleanRun)

	if !res.completed {
		t.Fatal("no completion summary parsed from a complete run")
	}
	if res.status != statusPass {
		t.Errorf("status = %q, want %q", res.status, statusPass)
	}
	if res.hardwareIncidents != 0 || res.miscompares != 0 || res.sdcDetections != 0 {
		t.Errorf("clean run reported incidents=%d miscompares=%d sdc=%d, want all zero",
			res.hardwareIncidents, res.miscompares, res.sdcDetections)
	}
	if res.toolSeconds != 540.02 {
		t.Errorf("toolSeconds = %v, want 540.02", res.toolSeconds)
	}
	if res.copyMBs != 8737.85 {
		t.Errorf("copyMBs = %v, want 8737.85", res.copyMBs)
	}
	// Copy threads only: a memcpy moves the same bytes in both directions.
	if res.readMBs() != 8737.85 || res.writeMBs() != 8737.85 {
		t.Errorf("read=%v write=%v, want both 8737.85", res.readMBs(), res.writeMBs())
	}
	if code, reason := decide(res, nil); code != exitPass {
		t.Errorf("decide = %d (%s), want exitPass", code, reason)
	}
}

func TestParseHardwareFailure(t *testing.T) {
	const out = `2026/08/02-11:00:00(7) Log: Commandline - stressapptest -M 4096 -s 60 -m 4 -v 8
2026/08/02-11:00:31(7) Hardware Error: miscompare on CPU 3(0x8) at 0xffff8801(0x0:DIMM Unknown): read:0x0000000000000000, reread:0x0000000000000000 expected:0xdeadbeefdeadbeef
2026/08/02-11:00:44(7) Hardware Error: miscompare on CPU 5(0x20) at 0xffff9002(0x0:DIMM Unknown): read:0x0000000000000000, reread:0x0000000000000000 expected:0xdeadbeefdeadbeef
2026/08/02-11:01:00(7) Stats: Found 2 hardware incidents
2026/08/02-11:01:00(7) Stats: Completed: 61440.00M in 60.01s 1023.83MB/s, with 2 hardware incidents, 7 errors
2026/08/02-11:01:00(7) Stats: Memory Copy: 61440.00M at 1023.83MB/s
2026/08/02-11:01:00(7) Status: FAIL - test discovered HW problems`

	res := feedAll(t, out)
	if res.hardwareIncidents != 2 || res.miscompares != 7 {
		t.Errorf("incidents=%d miscompares=%d, want 2 and 7", res.hardwareIncidents, res.miscompares)
	}
	// Two logged incidents, seven mismatched values: the two counts are
	// different measurements and must not be normalised into one.
	if res.sdcDetections != 2 {
		t.Errorf("sdcDetections = %d, want 2", res.sdcDetections)
	}
	if res.status != statusFailHardware {
		t.Errorf("status = %q, want %q", res.status, statusFailHardware)
	}
	// stressapptest exits 1 on a hardware failure, so the wait error is present
	// on this path too and must not turn the verdict into an Error.
	code, reason := decide(res, errors.New("stressapptest exited with exit status 1"))
	if code != exitFail {
		t.Errorf("decide = %d (%s), want exitFail", code, reason)
	}
}

// The distinction the whole runner exists to preserve: stressapptest exits 1
// for BOTH "the memory is bad" and "I could not run", and only the first is a
// verdict about the hardware.
func TestProceduralFailureIsErrorNotFail(t *testing.T) {
	const out = `2026/08/02-11:00:00(7) Log: Commandline - stressapptest -M 900000 -s 60 -m 4 -v 8
2026/08/02-11:00:00(7) Process Error: failed to allocate memory
2026/08/02-11:00:00(7) Stats: Completed: 0.00M in 0.01s 0.00MB/s, with 0 hardware incidents, 0 errors
2026/08/02-11:00:00(7) Status: FAIL - test encountered procedural errors`

	res := feedAll(t, out)
	if res.status != statusFailProcedural {
		t.Fatalf("status = %q, want %q", res.status, statusFailProcedural)
	}
	code, reason := decide(res, errors.New("stressapptest exited with exit status 1"))
	if code == exitFail {
		t.Fatalf("a procedural failure was reported as a hardware Fail (%s) — memory nobody could test must never be condemned", reason)
	}
	if code != exitError {
		t.Errorf("decide = %d (%s), want exitError", code, reason)
	}
}

func TestIncompleteRunIsError(t *testing.T) {
	const out = `2026/08/02-11:00:00(7) Log: Commandline - stressapptest -M 4096 -s 600 -m 4 -v 8
2026/08/02-11:00:03(7) Log: Region 0: 4096 MB`

	res := feedAll(t, out)
	if res.completed {
		t.Fatal("a truncated capture parsed as a completed run")
	}
	code, reason := decide(res, nil)
	if code != exitError {
		t.Errorf("decide = %d (%s), want exitError", code, reason)
	}
}

// A run that was killed still reports corruption it already saw: memory that
// returned a wrong value returned a wrong value, whatever happened afterwards.
func TestCorruptionOutranksInterruption(t *testing.T) {
	res := satResult{sdcDetections: 1}
	code, reason := decide(res, errors.New("the runner received terminated before the test finished"))
	if code != exitFail {
		t.Errorf("decide = %d (%s), want exitFail", code, reason)
	}
}

// The mirror image: an interruption with no evidence is not a verdict.
func TestInterruptionWithoutEvidenceIsError(t *testing.T) {
	res := feedAll(t, cleanRun)
	res.completed = false
	res.status = statusUnknown
	code, reason := decide(res, errors.New("the runner received terminated before the test finished"))
	if code != exitError {
		t.Errorf("decide = %d (%s), want exitError", code, reason)
	}
}

// Nothing reaches exit 0 without both a completion summary and the tool's own
// PASS. A missing measurement must never look like a clean run.
func TestNoPassWithoutCompletionSummary(t *testing.T) {
	res := satResult{status: statusPass} // PASS line seen, summary never printed
	if code, _ := decide(res, nil); code != exitError {
		t.Errorf("decide = %d, want exitError for a PASS with no completion summary", code)
	}

	res = satResult{completed: true} // summary seen, no final status
	if code, _ := decide(res, nil); code != exitError {
		t.Errorf("decide = %d, want exitError for a completed run with no status", code)
	}
}

// With invert and check threads the two directions genuinely differ, which is
// why they are two metrics rather than one.
func TestReadWriteDerivation(t *testing.T) {
	const out = `2026/08/02-11:09:00(7) Stats: Completed: 100.00M in 60.00s 500.00MB/s, with 0 hardware incidents, 0 errors
2026/08/02-11:09:00(7) Stats: Memory Copy: 60000.00M at 1000.00MB/s
2026/08/02-11:09:00(7) Stats: Data Check: 18000.00M at 300.00MB/s
2026/08/02-11:09:00(7) Stats: Invert Data: 12000.00M at 200.00MB/s
2026/08/02-11:09:00(7) Status: PASS - please verify no corrected errors`

	res := feedAll(t, out)
	if got, want := res.readMBs(), 1500.0; got != want {
		t.Errorf("readMBs = %v, want %v (copy + check + invert)", got, want)
	}
	if got, want := res.writeMBs(), 1200.0; got != want {
		t.Errorf("writeMBs = %v, want %v (copy + invert; check only reads)", got, want)
	}
}

// "Stats: Found N hardware incidents" and the Completed line state the same
// count independently; an incident reported by either one happened.
func TestHardwareIncidentsTakeTheLargerReport(t *testing.T) {
	const out = `2026/08/02-11:09:00(7) Stats: Found 3 hardware incidents
2026/08/02-11:09:00(7) Stats: Completed: 100.00M in 60.00s 1.66MB/s, with 0 hardware incidents, 0 errors
2026/08/02-11:09:00(7) Status: PASS - please verify no corrected errors`

	res := feedAll(t, out)
	if res.hardwareIncidents != 3 {
		t.Errorf("hardwareIncidents = %d, want 3", res.hardwareIncidents)
	}
	if code, reason := decide(res, nil); code != exitFail {
		t.Errorf("decide = %d (%s), want exitFail — a PASS status must not outrank a reported incident", code, reason)
	}
}

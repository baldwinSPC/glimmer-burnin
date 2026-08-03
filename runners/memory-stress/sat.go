// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// satStatus is stressapptest's own final verdict, which is a three-way answer
// even though the tool spells two of them "FAIL".
type satStatus string

const (
	statusUnknown satStatus = "unknown"
	// statusPass is "Status: PASS - please verify no corrected errors".
	statusPass satStatus = "pass"
	// statusFailHardware is "Status: FAIL - test discovered HW problems": the
	// tool ran and the memory is bad. This is a verdict about the node.
	statusFailHardware satStatus = "fail-hardware"
	// statusFailProcedural is "Status: FAIL - test encountered procedural
	// errors": the tool could not do its job — it could not allocate the
	// region, or a thread failed for a reason that is not a miscompare. The
	// memory was NOT judged, so this is an Error and never a Fail. stressapptest
	// exits 1 for both kinds of FAIL, which is exactly why the exit code alone
	// cannot be forwarded as this runner's verdict.
	statusFailProcedural satStatus = "fail-procedural"
)

// satResult is one stressapptest run, as read off its output.
type satResult struct {
	// completed records that a "Stats: Completed:" summary was seen. Without
	// it the counters below are not zero, they are unknown — the distinction
	// the whole Error-vs-Fail rule rests on.
	completed         bool
	status            satStatus
	hardwareIncidents int64
	miscompares       int64
	// sdcDetections counts the individual "Hardware Error:" incidents logged
	// during the run. It is a distinct measurement from miscompares, not a
	// restatement: one corrupted region produces many mismatched values and a
	// smaller number of logged incidents, and the tool caps its own logging, so
	// this can under-count where miscompares cannot.
	sdcDetections int64
	// toolSeconds is stressapptest's own run time, which excludes the
	// allocation and pattern fill that precede it. elapsedS covers both.
	toolSeconds float64
	totalMBs    float64
	copyMBs     float64
	checkMBs    float64
	invertMBs   float64
}

// readMBs and writeMBs derive the two directions from the thread mix.
//
// stressapptest reports throughput per thread class, not per direction, so the
// directions are derived from what each class does to memory:
//
//   - a copy thread reads its source and writes its destination at the same
//     rate, so it contributes equally to both;
//   - a check thread only reads and compares, so it contributes to reads alone;
//   - an invert thread reads a block and writes it back, contributing to both.
//
// With the default thread mix (copy threads only) the two figures are therefore
// EQUAL by construction. That is a property of the workload, not a bug in the
// parser: a memcpy moves the same number of bytes in each direction. They are
// still reported separately because the invert and check threads a profile can
// enable do pull them apart, and because a consumer charting them wants the
// same two series from every node regardless of thread mix.
func (r satResult) readMBs() float64  { return r.copyMBs + r.checkMBs + r.invertMBs }
func (r satResult) writeMBs() float64 { return r.copyMBs + r.invertMBs }

// stressapptest prefixes every log line with a timestamp and a pid, so nothing
// here is anchored to the start of the line.
var (
	// "Stats: Completed: 1234.00M in 60.02s 2056.94MB/s, with 0 hardware incidents, 0 errors"
	completedRe = regexp.MustCompile(`Stats: Completed:\s+([0-9.]+)M in ([0-9.]+)s ([0-9.]+)MB/s, with ([0-9]+) hardware incidents, ([0-9]+) errors`)
	// "Stats: Found 0 hardware incidents"
	foundRe = regexp.MustCompile(`Stats: Found ([0-9]+) hardware incidents`)

	copyRe   = rateRe("Memory Copy")
	checkRe  = rateRe("Data Check")
	invertRe = rateRe("Invert Data")
)

// rateRe matches one of stressapptest's per-class throughput lines, e.g.
// "Stats: Memory Copy: 123456.00M at 2056.94MB/s".
func rateRe(label string) *regexp.Regexp {
	return regexp.MustCompile(`Stats: ` + regexp.QuoteMeta(label) + `:\s+([0-9.]+)M at ([0-9.]+)MB/s`)
}

// feed folds one line of stressapptest output into the result.
func (r *satResult) feed(line string) {
	if strings.Contains(line, "Hardware Error:") {
		r.sdcDetections++
		return
	}
	if m := completedRe.FindStringSubmatch(line); m != nil {
		r.completed = true
		r.toolSeconds = parseFloat(m[2])
		r.totalMBs = parseFloat(m[3])
		r.hardwareIncidents = max(r.hardwareIncidents, parseInt(m[4]))
		r.miscompares = max(r.miscompares, parseInt(m[5]))
		return
	}
	if m := foundRe.FindStringSubmatch(line); m != nil {
		// A second, independently printed statement of the same count. Take
		// the larger: an incident reported by either line happened.
		r.hardwareIncidents = max(r.hardwareIncidents, parseInt(m[1]))
		return
	}
	if m := copyRe.FindStringSubmatch(line); m != nil {
		r.copyMBs = parseFloat(m[2])
		return
	}
	if m := checkRe.FindStringSubmatch(line); m != nil {
		r.checkMBs = parseFloat(m[2])
		return
	}
	if m := invertRe.FindStringSubmatch(line); m != nil {
		r.invertMBs = parseFloat(m[2])
		return
	}
	if strings.Contains(line, "Status: PASS") {
		r.status = statusPass
		return
	}
	if strings.Contains(line, "Status: FAIL") {
		if strings.Contains(line, "HW problems") {
			r.status = statusFailHardware
		} else {
			// "procedural errors", and anything the tool learns to say later:
			// a FAIL we cannot attribute to hardware is not a hardware verdict.
			r.status = statusFailProcedural
		}
	}
}

func parseFloat(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}

func parseInt(s string) int64 {
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

// decide turns the parsed run into this runner's exit code.
//
// The order of the clauses is the whole design:
//
//  1. Positive evidence of corruption is a verdict about the hardware even if
//     the run was later interrupted or the tool died. Memory that returned a
//     wrong value returned a wrong value.
//  2. Everything else that is not a clean, completed, PASS run is an Error —
//     unjudged — and never a Fail. A runner that cannot measure has not found
//     bad hardware; it has found nothing.
//
// Nothing reaches exit 0 without a completed summary AND the tool's own PASS.
func decide(res satResult, runErr error) (int, string) {
	if res.hardwareIncidents > 0 || res.miscompares > 0 || res.sdcDetections > 0 {
		return exitFail, fmt.Sprintf(
			"memory errors detected: %d hardware incidents, %d miscompares, %d logged corruption events",
			res.hardwareIncidents, res.miscompares, res.sdcDetections)
	}
	if res.status == statusFailHardware {
		// The tool says the hardware is bad but reported no counters we could
		// read. Believe it: it is the component that looked at the memory.
		return exitFail, "stressapptest reported hardware problems"
	}
	if runErr != nil {
		return exitError, runErr.Error() + " — this node's memory was not judged"
	}
	if !res.completed {
		return exitError, "stressapptest printed no completion summary — the test did not run to completion, so this node's memory was not judged"
	}
	if res.status == statusFailProcedural {
		return exitError, "stressapptest failed procedurally (it could not run the test) — this node's memory was not judged"
	}
	if res.status != statusPass {
		return exitError, "stressapptest printed no final status — this node's memory was not judged"
	}
	return exitPass, ""
}

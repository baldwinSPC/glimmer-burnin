// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"
	"time"
)

func TestFillPatternSetsEveryByte(t *testing.T) {
	buf := make([]byte, 1000) // deliberately not a power of two
	fillPattern(buf, 0xAB)
	for i, b := range buf {
		if b != 0xAB {
			t.Fatalf("buf[%d] = 0x%02X, want 0xAB", i, b)
		}
	}
}

func TestFillPatternHandlesEmptyAndSingleByte(t *testing.T) {
	fillPattern(nil, 0xFF) // must not panic
	buf := make([]byte, 1)
	fillPattern(buf, 0x42)
	if buf[0] != 0x42 {
		t.Errorf("buf[0] = 0x%02X, want 0x42", buf[0])
	}
}

func TestVerifyPatternCleanBufferFindsNothing(t *testing.T) {
	buf := make([]byte, 4096)
	fillPattern(buf, 0x00)
	r := verifyPattern(buf, 0x00)
	if r.bitFlips != 0 || r.bytesFlipped != 0 || r.firstOffsetValid {
		t.Errorf("verifyPattern on a clean buffer = %+v, want all zero", r)
	}
}

func TestVerifyPatternCountsBitsNotBytes(t *testing.T) {
	buf := make([]byte, 16)
	fillPattern(buf, 0x00)
	// Flip two bits in one byte, one bit in another — three bit flips, two
	// flipped bytes. If this were counting bytes instead of bits, the two
	// numbers would collapse to the same value and the test below would not
	// distinguish a single-bit upset from a two-bit one, which is exactly the
	// distinction this metric exists to preserve.
	buf[3] = 0b00000011 // 2 bits wrong against expected 0x00
	buf[9] = 0b00000001 // 1 bit wrong
	r := verifyPattern(buf, 0x00)
	if r.bitFlips != 3 {
		t.Errorf("bitFlips = %d, want 3", r.bitFlips)
	}
	if r.bytesFlipped != 2 {
		t.Errorf("bytesFlipped = %d, want 2", r.bytesFlipped)
	}
	if !r.firstOffsetValid || r.firstOffset != 3 {
		t.Errorf("firstOffset = %d (valid=%v), want 3 (valid=true) — the EARLIER corrupted byte", r.firstOffset, r.firstOffsetValid)
	}
}

func TestVerifyPatternAgainstComplement(t *testing.T) {
	// The whole point of the two-pattern design: a cell stuck at 1 reads back
	// correctly against a 0xFF fill (every bit is already 1) and is invisible
	// until verified against 0xFF's complement, 0x00, where a stuck-1 bit now
	// disagrees with every expected 0.
	buf := make([]byte, 8)
	fillPattern(buf, 0xFF)
	// Everything transitions to 0x00 correctly EXCEPT one stuck bit, which
	// the real complement fill would have cleared to 0 and did not.
	fillPattern(buf, 0x00)
	buf[2] = 0x01
	rComplement := verifyPattern(buf, 0x00)
	if rComplement.bitFlips != 1 {
		t.Errorf("against the complement, bitFlips = %d, want 1 (the stuck bit)", rComplement.bitFlips)
	}
}

func TestRunTestCleanRunCompletesBothPatterns(t *testing.T) {
	buf := make([]byte, 64)
	p := plan{holdSeconds: 0} // hold instantly resolves since deadline <= now on first check
	never := func() (string, bool) { return "", false }
	var logs []string
	res, err := runTest(buf, p, never, &emitter{w: discard{}}, func(f string, a ...any) { logs = append(logs, f) })
	if err != nil {
		t.Fatalf("runTest: %v", err)
	}
	if res.patternsCompleted != len(patternBytes) {
		t.Errorf("patternsCompleted = %d, want %d", res.patternsCompleted, len(patternBytes))
	}
	if res.bitFlips != 0 {
		t.Errorf("bitFlips = %d on a buffer only this test touched, want 0", res.bitFlips)
	}
	// The buffer should have been left holding the LAST pattern
	// (patternBytes' final entry) — confirms the loop actually ran fill for
	// every pattern in order rather than stopping after the first.
	want := patternBytes[len(patternBytes)-1]
	for i, b := range buf {
		if b != want {
			t.Fatalf("buf[%d] = 0x%02X after the run, want the last pattern 0x%02X", i, b, want)
		}
	}
}

func TestRunTestDetectsARealFlipDuringTheHold(t *testing.T) {
	// A genuine integration test, not a mock: a background goroutine mutates
	// the buffer partway through a real (short) hold, exactly as a real
	// hardware fault would — flipping a bit AFTER the fill and BEFORE the
	// verify — and this asserts runTest's own verify pass actually catches
	// it and reports it unfiltered, rather than trusting verifyPattern's own
	// unit tests to stand in for the integration.
	buf := make([]byte, 64)
	p := plan{holdSeconds: 1} // real wall-clock seconds; short enough to keep the test fast
	never := func() (string, bool) { return "", false }

	go func() {
		time.Sleep(300 * time.Millisecond) // inside pattern 1's (0xFF) hold
		buf[10] &^= 0x01                   // clear one bit runTest's fill just set to 1
	}()

	res, err := runTest(buf, p, never, &emitter{w: discard{}}, func(string, ...any) {})
	if err != nil {
		t.Fatalf("runTest: %v", err)
	}
	if res.bitFlips != 1 {
		t.Fatalf("bitFlips = %d, want 1 (the bit cleared during the hold)", res.bitFlips)
	}
	if !res.firstOffsetValid || res.firstOffset != 10 {
		t.Errorf("firstOffset = %d (valid=%v), want 10 (valid=true)", res.firstOffset, res.firstOffsetValid)
	}
	if res.firstOffsetPattern != 0xFF {
		t.Errorf("firstOffsetPattern = 0x%02X, want 0xFF (pattern 1, where the flip happened)", res.firstOffsetPattern)
	}
	// Pattern 2 (0x00) then fills over the corrupted byte and finds it clean
	// again, since nothing corrupted it a second time — patternsCompleted
	// should still reach the full count.
	if res.patternsCompleted != len(patternBytes) {
		t.Errorf("patternsCompleted = %d, want %d", res.patternsCompleted, len(patternBytes))
	}
}

func TestRunTestStopsBeforeTheFirstPatternReportsNothingCompleted(t *testing.T) {
	buf := make([]byte, 64)
	p := plan{holdSeconds: 3600} // would hang the test if stop were not honoured
	stop := func() (string, bool) { return "pod deadline", true }
	res, err := runTest(buf, p, stop, &emitter{w: discard{}}, func(string, ...any) {})
	if err == nil {
		t.Fatal("runTest returned no error despite stop firing immediately")
	}
	if res.patternsCompleted != 0 {
		t.Errorf("patternsCompleted = %d, want 0", res.patternsCompleted)
	}
}

func TestRunTestStopsDuringTheHoldOfTheSecondPattern(t *testing.T) {
	buf := make([]byte, 64)
	// A real, short hold: waitOrStop calls stop() once per pattern's hold (a
	// 1s hold resolves within its single tick), plus once for each pattern's
	// pre-loop check — call 1 (pattern 1 pre-check), call 2 (pattern 1's
	// hold tick), call 3 (pattern 2 pre-check). Firing on call 3 lets
	// pattern 1 genuinely finish (fill, hold, verify) before pattern 2 is
	// ever started.
	p := plan{holdSeconds: 1}
	calls := 0
	stop := func() (string, bool) {
		calls++
		return "pod deadline", calls > 2
	}
	res, err := runTest(buf, p, stop, &emitter{w: discard{}}, func(string, ...any) {})
	if err == nil {
		t.Fatal("runTest returned no error despite stop firing")
	}
	if res.patternsCompleted != 1 {
		t.Errorf("patternsCompleted = %d, want 1 (only the first pattern had time to finish)", res.patternsCompleted)
	}
}

func TestDecidePassesOnlyWhenEveryPatternCompletedClean(t *testing.T) {
	code, _ := decide(execResult{patternsCompleted: len(patternBytes)}, nil)
	if code != exitPass {
		t.Errorf("decide on a clean, complete run = %d, want exitPass", code)
	}
}

func TestDecideFailsOnAnyBitFlipEvenIfInterrupted(t *testing.T) {
	// A real finding must never be lost to an interruption — the same
	// record-before-kill principle this project's soak engine applies to a
	// miscompare, applied here.
	code, reason := decide(execResult{patternsCompleted: 1, bitFlips: 3, bytesFlipped: 2}, stopErr("pod deadline"))
	if code != exitFail {
		t.Errorf("decide with bitFlips>0 and an interruption = %d, want exitFail; reason=%q", code, reason)
	}
}

func TestDecideErrorsOnInterruptionWithNothingFoundYet(t *testing.T) {
	// Absence of a flip after only PART of the plan ran is not the same fact
	// as a clean full run — it is unjudged, not clean.
	code, _ := decide(execResult{patternsCompleted: 1}, stopErr("pod deadline"))
	if code != exitError {
		t.Errorf("decide on an interrupted, otherwise-clean partial run = %d, want exitError", code)
	}
}

// discard is a minimal io.Writer that keeps stdout noise out of test output.
type discard struct{}

func (discard) Write(p []byte) (int, error) { return len(p), nil }

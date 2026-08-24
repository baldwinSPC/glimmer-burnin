// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math/bits"
	"time"
)

// tickInterval is how often the hold loop wakes to check for a stop request.
// Coarse enough to cost nothing, fine enough that a signal or a deadline is
// noticed promptly rather than riding out to the next multi-second boundary.
const tickInterval = time.Second

// execResult is what runTest accumulated, across however many patterns
// actually completed before it returned.
type execResult struct {
	patternsCompleted  int
	bitFlips           int64
	bytesFlipped       int64
	firstOffsetValid   bool
	firstOffset        int64
	firstOffsetPattern byte
}

// runTest runs planBytes' patterns in order against buf: fill, hold for
// p.holdSeconds while polling stop, then verify. It returns whatever it
// accumulated and, if stop fired before every pattern completed, an error
// naming why — the caller decides what that combination means (decide()),
// since "stopped after finding damage" and "stopped having found nothing yet"
// are different verdicts, not different plumbing.
//
// stop is polled rather than selected on: the hold is already a loop that
// must wake at least once a second to remain responsive, so a tick-and-check
// costs nothing extra and needs no goroutine of its own — which also makes it
// trivial to fake in a test (a stub returning true on whichever call number a
// test wants to simulate an interruption at).
func runTest(buf []byte, p plan, stop func() (string, bool), em *emitter, logf func(string, ...any)) (execResult, error) {
	var res execResult

	for _, want := range patternBytes {
		if reason, stopped := stop(); stopped {
			return res, stopErr(reason)
		}

		fillStart := time.Now()
		fillPattern(buf, want)
		logf("pattern %d/%d: filled %d MiB with 0x%02X in %s",
			res.patternsCompleted+1, len(patternBytes), len(buf)/int(mib), want, time.Since(fillStart))

		if !waitOrStop(time.Duration(p.holdSeconds)*time.Second, stop) {
			reason, _ := stop()
			return res, stopErr(reason)
		}

		verifyStart := time.Now()
		vr := verifyPattern(buf, want)
		logf("pattern %d/%d: verified against 0x%02X in %s — %d bit flip(s) in %d byte(s)",
			res.patternsCompleted+1, len(patternBytes), want, time.Since(verifyStart), vr.bitFlips, vr.bytesFlipped)

		res.bitFlips += vr.bitFlips
		res.bytesFlipped += vr.bytesFlipped
		if vr.firstOffsetValid && !res.firstOffsetValid {
			res.firstOffsetValid = true
			res.firstOffset = vr.firstOffset
			res.firstOffsetPattern = want
		}
		res.patternsCompleted++
		em.int("retention_patterns_completed", int64(res.patternsCompleted))
	}
	return res, nil
}

// waitOrStop holds for d, waking every tickInterval to poll stop. It returns
// false the instant stop reports true, without waiting out the rest of d —
// the hold exists to test the hardware, not to guarantee a fixed wall-clock
// runtime once it is already known there is nothing left to wait for.
func waitOrStop(d time.Duration, stop func() (string, bool)) bool {
	deadline := time.Now().Add(d)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return true
		}
		wait := tickInterval
		if remaining < wait {
			wait = remaining
		}
		time.Sleep(wait)
		if _, stopped := stop(); stopped {
			return false
		}
	}
}

func stopErr(reason string) error { return &stopError{reason: reason} }

type stopError struct{ reason string }

func (e *stopError) Error() string { return e.reason }

// fillPattern writes value into every byte of buf.
//
// A naive `for i := range buf { buf[i] = value }` is a per-byte store, and on
// a multi-GiB region that is seconds taken out of the hold budget for no
// reason. Doubling copy turns the fill into O(log n) calls to the runtime's
// own memmove: after the first byte is set, each copy doubles how much of buf
// is already correct, so the last copy — the expensive one — is also the
// only one anywhere near buf's full size.
func fillPattern(buf []byte, value byte) {
	if len(buf) == 0 {
		return
	}
	buf[0] = value
	for filled := 1; filled < len(buf); filled *= 2 {
		copy(buf[filled:], buf[:filled])
	}
}

// verifyResult is what one verify pass over the region found.
type verifyResult struct {
	// bitFlips is the total number of bits that read back wrong across the
	// whole region — a byte with two wrong bits counts as two, not one, so
	// this answers "how much damage", not merely "how many bytes".
	bitFlips int64
	// bytesFlipped is how many BYTES contained at least one wrong bit — a
	// coarser, easier-to-reason-about companion to bitFlips (one corrupted
	// byte can carry up to eight flipped bits, and the two numbers answer
	// different questions the way miscompares and sdcDetections do elsewhere
	// in this project's soak engine).
	bytesFlipped int64
	// firstOffsetValid is false when nothing was found, in which case
	// firstOffset is meaningless and must not be reported — a metric printed
	// from a zero value that was never actually a finding is exactly the
	// fabricated-zero this project's runners are built to avoid.
	firstOffsetValid bool
	firstOffset      int64
}

// verifyPattern reads every byte of buf and compares it against want.
func verifyPattern(buf []byte, want byte) verifyResult {
	var r verifyResult
	for i, b := range buf {
		if b == want {
			continue
		}
		diff := b ^ want
		n := bits.OnesCount8(diff)
		r.bitFlips += int64(n)
		r.bytesFlipped++
		if !r.firstOffsetValid {
			r.firstOffsetValid = true
			r.firstOffset = int64(i)
		}
	}
	return r
}

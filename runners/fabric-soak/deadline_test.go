// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"
	"time"
)

// The grace exists at all, and is exactly the sum of the two things it has
// to cover — not just "some positive number" that could silently shrink.
func TestServerDeadline_GraceIsWindowPlusHangPlusSkewBudget(t *testing.T) {
	start := time.Date(2026, 8, 23, 16, 40, 44, 0, time.UTC)
	const durationSeconds = 600
	const windowSeconds = 20

	got := serverDeadline(start, durationSeconds, windowSeconds)

	wantGrace := startSkewBudgetSeconds + windowSeconds + windowGraceSeconds
	want := start.Add(time.Duration(durationSeconds+wantGrace) * time.Second)
	if !got.Equal(want) {
		t.Fatalf("serverDeadline = %v, want %v (grace=%ds)", got, want, wantGrace)
	}
	if grace := got.Sub(start.Add(durationSeconds * time.Second)); grace <= 0 {
		t.Fatalf("grace = %v, want strictly positive — a server with no grace is the bug #459 found", grace)
	}
}

// This is the actual scenario #459 measured: server and client each compute
// start+duration independently, the client starts a few seconds after the
// server (the operator waits for the server to be Ready), and the client's
// LAST window can start just before its own deadline and then run for up to
// windowSeconds+windowGraceSeconds before it is known to be finished.
//
// Without the fix, the server's deadline is strictly before this finish time
// for ANY positive skew — that is the bug, not an edge case of it. This test
// fails against the pre-fix behaviour (serverDeadline = start+duration, no
// grace) for the exact skew #459 measured, which is the "verify it fails
// without the fix" bar this project holds threshold and timing logic to.
func TestServerDeadline_OutlastsTheClientsLastWindow(t *testing.T) {
	const durationSeconds = 600
	const windowSeconds = 20

	cases := []struct {
		name        string
		skewSeconds int // how much later the client pod starts than the server
	}{
		{"the skew #459 actually measured", 2},
		{"a slower schedule under contention", 45},
		{"right at the edge of the skew budget", startSkewBudgetSeconds - 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			serverStart := time.Date(2026, 8, 23, 16, 40, 44, 0, time.UTC)
			clientStart := serverStart.Add(time.Duration(tc.skewSeconds) * time.Second)

			// The client's absolute worst case: a window starts in the last
			// instant before its own deadline, then runs the full
			// windowSeconds+windowGraceSeconds before the client can know
			// whether it succeeded.
			clientDeadline := clientStart.Add(durationSeconds * time.Second)
			clientLastWindowKnownBy := clientDeadline.Add(time.Duration(windowSeconds+windowGraceSeconds) * time.Second)

			srvDeadline := serverDeadline(serverStart, durationSeconds, windowSeconds)
			if !srvDeadline.After(clientLastWindowKnownBy) {
				t.Errorf("server deadline %v does not outlast the client's last possible window (done by %v) "+
					"at %ds of skew — this is exactly #459's race", srvDeadline, clientLastWindowKnownBy, tc.skewSeconds)
			}
		})
	}
}

// The budget is finite. A server that lingers forever defeats the point of a
// deadline at all, so the grace should be a small, bounded multiple of the
// soak's own duration for realistic durations, not something that grows
// without bound.
func TestServerDeadline_GraceIsBoundedNotUnlimited(t *testing.T) {
	start := time.Now()
	got := serverDeadline(start, 600, 20)
	grace := got.Sub(start) - 600*time.Second
	if grace > 5*time.Minute {
		t.Errorf("grace = %v, want well under 5 minutes for a 10-minute soak — "+
			"the fix for #459 is a bounded skew allowance, not unbounded lingering", grace)
	}
}

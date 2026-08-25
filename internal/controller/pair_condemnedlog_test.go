package controller

import (
	"strings"
	"testing"
)

// TestPair_ALingeringServersLogIsCapturedBeforeItIsDestroyed is the regression
// test for #490: a server condemned while still live used to be deleted with
// no attempt ever made to read what it had written. harvestPair now fetches
// its log best-effort before killPods can destroy it, and the tail rides in
// the stored result's message so a human debugging a hung server afterward has
// something to read.
//
// This must NOT change the verdict. The client is still the deciding side for
// a lingering server (pairVerdict's server.result == nil branch), and this
// test asserts that alongside the new message content — a regression that
// quietly moved the server into the both-reported precedence path would be
// worse than the bug #490 fixes.
func TestPair_ALingeringServersLogIsCapturedBeforeItIsDestroyed(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"), gb10Node("spark-b"),
		pairTest("link"),
		profile("acceptance", nil, false, testRef("link")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	server := h.pairPod("run1", pairRoleServer)
	if server == nil {
		t.Fatal("no server pod")
	}
	h.readyPod(server)
	h.reconcile("run1")
	clientPod := h.pairPod("run1", pairRoleClient)
	if clientPod == nil {
		t.Fatal("no client pod")
	}
	h.startPod(clientPod)

	// The server never terminates — it is still Running when the client
	// finishes — but it did write something before being condemned. finishPod
	// is deliberately not called on it: this is the "hung, not crashed" case
	// #490 is about, and finishPod would give it a terminated container status
	// this scenario must not have.
	const serverLog = "binding rendezvous port\nwaiting for a peer connection\n"
	h.logs[server.Name] = serverLog

	h.finishPod(clientPod, 0, "bandwidthGbps=99.6\n", "Completed")
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	run := h.run("run1")
	if len(run.Status.Results) == 0 {
		t.Fatal("no results recorded")
	}
	res := run.Status.Results[0]

	if res.Phase != "Passed" {
		t.Fatalf("phase = %q, want Passed (the client decides for a lingering server) — message: %q", res.Phase, res.Message)
	}
	if !strings.Contains(res.Message, "did not report") {
		t.Errorf("message dropped the existing lingering-server wording: %q", res.Message)
	}
	if !strings.Contains(res.Message, "binding rendezvous port") || !strings.Contains(res.Message, "waiting for a peer connection") {
		t.Errorf("message does not carry the server's own log, captured before it was condemned: %q", res.Message)
	}
}

// TestClampCondemnedLog_KeepsBothEndsAndBoundsLength guards the knobs #490's
// diagnostic capture depends on: a chatty server's account is cut to its
// first and last lines rather than an arbitrary byte offset or the tail
// alone, and the result never grows past maxCondemnedLog regardless of how
// much the server wrote.
//
// Head AND tail, not tail alone, is the point being guarded — #490 shipped
// tail-only, and hardware verification on fabric-soak (a runner that fails to
// START, restarting a banner-printing subprocess every window) showed a
// tail-only capture reliably grabbing the late-run banner noise while the one
// line explaining WHY the server never listened sat at line 2, before any of
// it. This test's "the first line" assertion is that regression, pinned.
func TestClampCondemnedLog_KeepsBothEndsAndBoundsLength(t *testing.T) {
	var lines []string
	lines = append(lines, "the first line")
	for i := 0; i < condemnedLogHeadLines+condemnedLogTailLines+10; i++ {
		lines = append(lines, "line")
	}
	lines[len(lines)-1] = "the last line"
	got := clampCondemnedLog(strings.Join(lines, "\n"))
	if !strings.Contains(got, "the first line") {
		t.Errorf("dropped the earliest line — this is the regression #490's tail-only capture had: %q", got)
	}
	if !strings.Contains(got, "the last line") {
		t.Errorf("dropped the most recent line: %q", got)
	}
	if strings.Contains(got, "line line line line line line line line line line line line line line line line line line line line") {
		t.Errorf("kept a long unbroken run from the middle, meaning nothing was actually elided: %q", got)
	}
	if !strings.Contains(got, "omitted") {
		t.Errorf("no elision marker for a log that overflowed both windows: %q", got)
	}

	huge := strings.Repeat("x", maxCondemnedLog*4)
	if got := clampCondemnedLog(huge); len(got) > maxCondemnedLog+len("… (truncated)") {
		t.Errorf("clampCondemnedLog did not bound its output: got %d bytes", len(got))
	}
}

// TestClampCondemnedLog_ShortLogNeedsNoElision asserts the no-op case:
// nothing is dropped, and no elision marker appears, when the whole log
// already fits inside head+tail's combined budget.
func TestClampCondemnedLog_ShortLogNeedsNoElision(t *testing.T) {
	lines := []string{"first", "second", "third"}
	got := clampCondemnedLog(strings.Join(lines, "\n"))
	for _, l := range lines {
		if !strings.Contains(got, l) {
			t.Errorf("dropped %q from a log short enough to need no elision: %q", l, got)
		}
	}
	if strings.Contains(got, "omitted") {
		t.Errorf("elision marker present for a log that fit entirely within budget: %q", got)
	}
}

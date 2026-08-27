package main

import "testing"

// TestConnectRefusedRecognisesPerftestsOwnWording pins the two spellings #496
// was diagnosed from.
//
// These strings are perftest's, not ours, so they are the part of this fix most
// likely to rot: a perftest bump that reworded them turns the retry off
// silently, and the symptom is the ORIGINAL bug — a healthy link reported as
// having "carried traffic and stopped carrying it". Captured verbatim from the
// run in #496.
func TestConnectRefusedRecognisesPerftestsOwnWording(t *testing.T) {
	captured := "Couldn't connect to server.bp-i496-run-1-t0-a1-fc37e2c5.glimmer-burnin-system.svc:18515\n" +
		"Unable to open file descriptor for socket connection Unable to init the socket connection\n"
	if !connectRefused(captured) {
		t.Fatal("the captured #496 refusal is not recognised — the retry will not fire and the " +
			"refusal will be reported as a link failure, which is the bug this fixes")
	}

	for _, each := range []string{
		"Couldn't connect to server.example.svc:18515",
		"Unable to init the socket connection",
	} {
		if !connectRefused(each) {
			t.Errorf("perftest prints these together but either has been seen leading; %q alone must match", each)
		}
	}
}

// TestConnectRefusedIgnoresAFabricFault is the half that protects the verdict.
//
// Every string here is a window that REACHED the fabric and failed there. If any
// of them matched, the runner would retry a real fault until it happened to pass
// and report a bad link as a good one — laundering a hardware verdict into an
// acceptance, which is the one direction this project never tolerates.
func TestConnectRefusedIgnoresAFabricFault(t *testing.T) {
	for _, each := range []string{
		"Couldn't create CQ",
		"Failed to modify QP to RTR",
		"ethernet_read_keys: Couldn't read remote address",
		"Completion with error at client",
		"",
		"1048576    116220    0.00    97.49    0.011622",
	} {
		if connectRefused(each) {
			t.Errorf("%q was taken for a refused control connection; a window that reached the fabric "+
				"must never be retried", each)
		}
	}
}

// TestAConnectFailureIsNotALinkFailure asserts the accounting split.
//
// This is the whole verdict argument in three lines: a refusal happens before
// any traffic, so it must not reach failures(), which is what decides Fail and
// therefore what permanently indicts a link the operator will never retry.
func TestAConnectFailureIsNotALinkFailure(t *testing.T) {
	var s soak
	s.recordConnectRetries(3)
	s.recordConnectFailure()
	s.recordConnectFailure()
	s.record(97.5, true)

	if got := s.failures(); got != 0 {
		t.Errorf("failures() = %d, want 0: a refused control connection is not a window in which the "+
			"link carried traffic and stopped", got)
	}
	if got := s.iterations(); got != 1 {
		t.Errorf("iterations() = %d, want 1: only windows that actually ran are windows", got)
	}
	if s.connectFailures != 2 || s.connectRetries != 3 {
		t.Errorf("connect accounting = %d failures / %d retries, want 2/3 — it must still be REPORTED, "+
			"or the fix hides the race instead of closing it", s.connectFailures, s.connectRetries)
	}
}

// TestEveryWindowRefusedLeavesNothingMeasured is the failing-closed case.
//
// If the server never comes back, every window is refused, windows[] stays empty
// and stats() reports nothing — which is what routes runClient to its "no window
// completed ... the link was not soaked" Error. Unjudged, not passed: the danger
// of moving refusals out of failures() would be a run that refuses everything and
// still exits 0.
func TestEveryWindowRefusedLeavesNothingMeasured(t *testing.T) {
	var s soak
	for i := 0; i < 12; i++ {
		s.recordConnectFailure()
	}
	if _, ok := s.stats(); ok {
		t.Fatal("stats() reported a measurement after every window was refused — a soak that never " +
			"opened a connection must reach the Error path, never a pass")
	}
	if s.failures() != 0 || s.iterations() != 0 {
		t.Errorf("iterations/failures = %d/%d, want 0/0", s.iterations(), s.failures())
	}
}

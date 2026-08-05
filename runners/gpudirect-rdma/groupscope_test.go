package main

import (
	"os"
	"testing"
)

// A Pair-shaped runner must ERROR at Group scope, never Skip — issue #118.
//
// The operator sets BURNIN_RANK/BURNIN_NRANKS for a Group execution and
// deliberately does NOT set BURNIN_ROLE, on the stated guarantee that a runner
// keying off server/client "fails loudly rather than treating rank 4 of eleven
// as a client". It did not fail loudly: the empty-role branch read a collective
// as a single node and declared a SKIP, so the operator recorded "acceptance
// does not apply to this hardware" and the run settled Passed around a
// collective that never ran.
//
// This is the guard for the guarantee. Skip is the one verdict that certifies,
// and it is never retried.
func TestGroupScopeIsAnErrorNotASkip(t *testing.T) {
	t.Setenv("BURNIN_RANK", "3")
	t.Setenv("BURNIN_NRANKS", "8")
	os.Unsetenv("BURNIN_ROLE")

	if got := run(); got != exitError {
		t.Fatalf("exit = %d, want %d (Error, hardware unjudged). exit %d would be a SKIP, which "+
			"certifies acceptance as inapplicable to a fleet nobody measured", got, exitError, exitSkip)
	}
}

// Node scope is unchanged: no rendezvous variables at all is still a clean,
// honest Skip — a link test with one endpoint really is inapplicable.
func TestNodeScopeStillSkips(t *testing.T) {
	os.Unsetenv("BURNIN_ROLE")
	os.Unsetenv("BURNIN_RANK")
	os.Unsetenv("BURNIN_NRANKS")

	if got := run(); got != exitSkip {
		t.Errorf("exit = %d, want %d — a Node-scope run of a link test is inapplicable, not an error",
			got, exitSkip)
	}
}

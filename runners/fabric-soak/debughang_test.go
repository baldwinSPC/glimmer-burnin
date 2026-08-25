// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"
	"time"
)

// TestDebugNeverExit_UnsetReturnsImmediately guards the default, unset case:
// this must be a true no-op on every ordinary run, since it sits directly in
// the server's normal return path.
func TestDebugNeverExit_UnsetReturnsImmediately(t *testing.T) {
	done := make(chan struct{})
	go func() {
		debugNeverExit(0)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("debugNeverExit blocked with FABRIC_SOAK_DEBUG_NEVER_EXIT unset")
	}
}

// TestDebugNeverExit_SetBlocksForever is the regression test for #498: this
// is the whole point of the function, so "does it actually block" is the one
// thing worth asserting about it. The spawned goroutine deliberately never
// returns and leaks for the rest of this test binary's life — that is
// debugNeverExit's real contract, not a test artifact.
func TestDebugNeverExit_SetBlocksForever(t *testing.T) {
	t.Setenv("FABRIC_SOAK_DEBUG_NEVER_EXIT", "1")
	done := make(chan struct{})
	go func() {
		debugNeverExit(0)
		close(done)
	}()
	select {
	case <-done:
		t.Fatal("debugNeverExit returned with FABRIC_SOAK_DEBUG_NEVER_EXIT set — it must block forever")
	case <-time.After(200 * time.Millisecond):
		// Still blocked, as required.
	}
}

// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestTheRandomPassReadsTheFileItWasGiven(t *testing.T) {
	buffered(t)
	path := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(path, make([]byte, 64*randBlock), 0o600); err != nil {
		t.Fatal(err)
	}
	r, err := randReadPass(path, 64*randBlock, 4, time.Now().Add(300*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}
	if !r.reached || r.ops == 0 || r.elapsed <= 0 || r.iops() <= 0 {
		t.Errorf("result = ops %d elapsed %s reached %v", r.ops, r.elapsed, r.reached)
	}
	if _, ok := r.p99Us(); !ok {
		t.Error("no p99 from a pass that counted reads")
	}
}

// A file too small to hold one block measured nothing, and says so.
func TestAFileWithNoWholeBlockIsRefused(t *testing.T) {
	buffered(t)
	if _, err := randReadPass(filepath.Join(t.TempDir(), "f"), randBlock-1, 4, time.Now().Add(time.Second)); err == nil {
		t.Error("a sub-block file was read")
	}
}

// An open that fails never reached the device: reached stays false, which is
// what makes the caller report Error rather than Fail.
func TestAnUnopenableFileHasNotReachedTheDevice(t *testing.T) {
	buffered(t)
	r, err := randReadPass(filepath.Join(t.TempDir(), "missing"), 64*randBlock, 2, time.Now().Add(100*time.Millisecond))
	if err == nil || r.reached {
		t.Errorf("err=%v reached=%v", err, r.reached)
	}
}

func TestP99FromTheHistogram(t *testing.T) {
	r := randResult{hist: make([]uint64, randHistMaxUs+1)}
	r.hist[80] = 98
	r.hist[85] = 1
	r.hist[900] = 1
	if p, ok := r.p99Us(); !ok || p != 85 {
		t.Errorf("p99 = %v %v, want 85", p, ok)
	}
	r.overflow = 50
	if _, ok := r.p99Us(); ok {
		t.Error("a tail inside the overflow bucket was reported as a number")
	}
	if _, ok := (randResult{hist: make([]uint64, 1)}).p99Us(); ok {
		t.Error("an empty histogram produced a p99")
	}
}

func TestTheSequentialReadCannotTakeTheRandomPassesShare(t *testing.T) {
	now := time.Now()
	dl := now.Add(40 * time.Second)
	if got := splitReadWindow(now, dl).Sub(now); got != 20*time.Second {
		t.Errorf("sequential read gets %s of 40s, want 20s", got)
	}
	if got := splitReadWindow(now, now.Add(-time.Second)); !got.Equal(now) {
		t.Errorf("a past deadline produced %v", got)
	}
}

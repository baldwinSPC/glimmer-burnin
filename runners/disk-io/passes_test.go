// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// buffered substitutes for openDirect so the loops below can run on a machine
// with no O_DIRECT.
//
// What this does NOT test, and cannot: whether O_DIRECT really bypasses the
// page cache. That is a kernel property and needs a Linux node — see the
// hardware-verification issue. What it DOES test is everything else, all of
// which would be just as wrong on real hardware: byte accounting, the deadline,
// the read-back, and the O_EXCL refusal.
func buffered(t *testing.T) {
	t.Helper()
	original := openFile
	openFile = func(path string, flags int, perm os.FileMode) (*os.File, error) {
		return os.OpenFile(path, flags, perm)
	}
	t.Cleanup(func() { openFile = original })
}

func TestAWritePassWritesExactlyWhatItWasAskedFor(t *testing.T) {
	buffered(t)
	path := filepath.Join(t.TempDir(), fileName)

	const total = 4 << 20 // 4 MiB
	const block = 1 << 20

	res, err := writePass(path, total, block, time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("writePass: %v", err)
	}
	if res.bytes != total {
		t.Errorf("wrote %d bytes, want %d", res.bytes, total)
	}
	if res.errors != 0 {
		t.Errorf("errors = %d", res.errors)
	}
	// One sample per block plus one for the fsync — the flush is inside the
	// measured window on purpose, because a failing controller is exactly one
	// whose flush is slow.
	if want := total/block + 1; len(res.lat) != want {
		t.Errorf("latency samples = %d, want %d", len(res.lat), want)
	}

	st, err := os.Stat(path)
	if err != nil || st.Size() != total {
		t.Errorf("file on disk is %v (err %v), want %d bytes", st, err, total)
	}
}

func TestTheDeadlineStopsAWriteShortRatherThanOverrunning(t *testing.T) {
	// BURNIN_DURATION_SECONDS is a promise about how long a node is held. A
	// slow device must write less, not run longer.
	buffered(t)
	path := filepath.Join(t.TempDir(), fileName)

	res, err := writePass(path, 1<<30, 1<<20, time.Now().Add(-time.Second)) // already expired
	if err != nil {
		t.Fatalf("writePass: %v", err)
	}
	if res.bytes != 0 {
		t.Errorf("wrote %d bytes past an expired deadline", res.bytes)
	}
}

func TestAReadPassReadsBackWhatWasWritten(t *testing.T) {
	buffered(t)
	path := filepath.Join(t.TempDir(), fileName)

	const total = 4 << 20
	const block = 1 << 20
	if _, err := writePass(path, total, block, time.Now().Add(time.Minute)); err != nil {
		t.Fatal(err)
	}

	res, err := readPass(path, block, time.Now().Add(time.Minute))
	if err != nil {
		t.Fatalf("readPass: %v", err)
	}
	if res.bytes != total {
		t.Errorf("read %d bytes, want %d — the read measures the file this runner wrote, never site data",
			res.bytes, total)
	}
	if res.errors != 0 {
		t.Errorf("errors = %d", res.errors)
	}
}

func TestAnExistingFileIsRefusedAndNeverTruncated(t *testing.T) {
	// Safety rule 2. A leftover from a crashed run must become a refusal, not
	// an overwrite of something that might not be ours at all.
	buffered(t)
	dir := t.TempDir()
	path := filepath.Join(dir, fileName)

	const precious = "this is somebody's data"
	if err := os.WriteFile(path, []byte(precious), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := writePass(path, 1<<20, 1<<20, time.Now().Add(time.Minute)); err == nil {
		t.Fatal("writePass opened a file that already existed")
	}

	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != precious {
		t.Errorf("the existing file was modified: %q", body)
	}
}

func TestTheWrittenPatternIsNotAllZeroes(t *testing.T) {
	// All-zeroes would let a compressing or deduplicating layer turn the write
	// into almost no device traffic, which is the storage equivalent of
	// measuring the cache.
	buffered(t)
	path := filepath.Join(t.TempDir(), fileName)
	if _, err := writePass(path, 1<<20, 1<<20, time.Now().Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var nonZero int
	for _, b := range body {
		if b != 0 {
			nonZero++
		}
	}
	if nonZero != len(body) {
		t.Errorf("%d of %d bytes were zero", len(body)-nonZero, len(body))
	}
}

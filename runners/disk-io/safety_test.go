// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAnAlmostFullFilesystemIsRefusedRatherThanUnderflowed(t *testing.T) {
	// The case the obvious arithmetic gets wrong. Written as
	// `free < write + reserve` rather than `free - write < reserve` because the
	// second underflows on unsigned when free is already below the reserve —
	// turning the single most dangerous input into a pass.
	const gib = uint64(1) << 30

	rows := []struct {
		name  string
		free  uint64
		write uint64
		ok    bool
	}{
		{"plenty of room", 100 * gib, 1 * gib, true},
		{"exactly the reserve remains", reserveBytes + 1*gib, 1 * gib, true},
		{"one byte short of the reserve", reserveBytes + 1*gib - 1, 1 * gib, false},
		{"less free than the reserve already", 1 * gib, 1 * gib, false},
		{"nothing free at all", 0, 1 * gib, false},
		{"a write larger than the device", 10 * gib, 100 * gib, false},
		{"a zero-length write", 100 * gib, 0, false},
	}
	for _, r := range rows {
		err := spaceCheck(r.free, r.write)
		if r.ok && err != nil {
			t.Errorf("%s: refused, want allowed: %v", r.name, err)
		}
		if !r.ok && err == nil {
			t.Errorf("%s: ALLOWED, want refused — free=%d write=%d", r.name, r.free, r.write)
		}
	}
}

func TestTheRefusalSaysWhatToDoAboutIt(t *testing.T) {
	err := spaceCheck(1<<30, 1<<30)
	if err == nil {
		t.Fatal("accepted")
	}
	for _, want := range []string{"disk pressure", "scratch volume", "DISK_IO_SIZE_MB"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not mention %q: %v", want, err)
		}
	}
}

func TestTheRunnerWritesExactlyOnePathAndItIsInsideTheDeclaredDirectory(t *testing.T) {
	got, err := targetFile("/mnt/scratch")
	if err != nil {
		t.Fatal(err)
	}
	if got != "/mnt/scratch/"+fileName {
		t.Errorf("target = %q", got)
	}

	// A relative path is resolved against the container's working directory,
	// which is not a place on the node — so it is refused rather than guessed.
	if _, err := targetFile("scratch"); err == nil {
		t.Error("a relative DISK_IO_PATH was accepted")
	}
	if _, err := targetFile(""); err == nil {
		t.Error("an empty DISK_IO_PATH was accepted")
	}

	// Traversal in the declared directory cannot reach outside it, because the
	// file name is a constant and the join is the only construction.
	got, err = targetFile("/mnt/scratch/../scratch2")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "/mnt/scratch2/") {
		t.Errorf("a cleaned path did not stay inside its directory: %q", got)
	}
}

func TestHumanBytesReadsLikeSomethingAPersonWrote(t *testing.T) {
	rows := []struct {
		in   uint64
		want string
	}{
		{2 << 30, "2.0 GiB"},
		{1536 << 20, "1.5 GiB"},
		{512 << 20, "512.0 MiB"},
		{512, "512 B"},
	}
	for _, r := range rows {
		if got := humanBytes(r.in); got != r.want {
			t.Errorf("humanBytes(%d) = %q, want %q", r.in, got, r.want)
		}
	}
}

// A LEFTOVER FILE MUST SURVIVE THE REFUSAL. Found on real hardware (#242).
//
// The runner refuses to open a file it did not create — O_EXCL, rule 2 — and
// says so in a message that promises the file has been left alone: "a leftover
// from a crashed run is a refusal rather than an overwrite. Remove it once you
// are satisfied it is not something else."
//
// The cleanup `defer` was armed BEFORE that refusal could fire, so the runner
// deleted the very file the refusal existed to protect, and then advised the
// operator to inspect it. That is worse than having no check at all: without the
// check the data is overwritten and the operator knows the runner touched it;
// with it, the data is gone and the operator has been told it is safe.
//
// Verified against a real ext4 volume on a GB10: exit 3 with the right message,
// and the file removed byte-for-byte.
func TestALeftoverFileSurvivesTheRefusal(t *testing.T) {
	if !directSupported {
		t.Skip("no O_DIRECT on this platform; this test is about the Linux write path (CI runs it)")
	}
	dir := t.TempDir()
	target := filepath.Join(dir, fileName)
	const irreplaceable = "IRREPLACEABLE-DATA-FROM-A-CRASHED-RUN\n"
	if err := os.WriteFile(target, []byte(irreplaceable), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("DISK_IO_PATH", dir)
	t.Setenv("DISK_IO_SIZE_MB", "1")
	t.Setenv("DISK_IO_BLOCK_KB", "64")
	t.Setenv("BURNIN_DURATION_SECONDS", "5")
	code := run()

	if code != exitError {
		t.Errorf("exit = %d, want %d (Error): refusing to open a file we did not create is machinery, "+
			"not a verdict about the disk", code, exitError)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("the leftover was DELETED by the run that refused to touch it: %v — the refusal "+
			"exists to protect exactly this file, and its message tells the operator it is safe", err)
	}
	if string(got) != irreplaceable {
		t.Errorf("the leftover was modified: %q", got)
	}
}

// The other half of the same block: an open that fails for any reason OTHER than
// "it already exists" is machinery, not a hardware verdict.
//
// A read-only mount, a wrongly-owned directory or a missing hostPath all surface
// as an open failure, and every one of them was reported as exit 1 — a FAILED
// disk, never retried, with the retry budget unspent. Observed on real hardware:
// a directory not writable by the runner's uid produced
// `DISK_IO_FAIL: write failed after 0 B: ... permission denied`.
//
// This is the compute-smoke v0.1.0 mistake in a new runner: everything that
// stopped the measurement from happening is exit 3.
func TestATargetTheRunnerCannotCreateIsAnErrorAndNotAFailedDisk(t *testing.T) {
	if !directSupported {
		t.Skip("no O_DIRECT on this platform; this test is about the Linux write path (CI runs it)")
	}
	// A path that is a FILE rather than a directory, so the create fails with
	// ENOTDIR. Chosen over an unwritable directory because mode bits do not deny
	// root, and CI may run either way — this failure mode is not bypassable.
	dir := t.TempDir()
	notADir := filepath.Join(dir, "actually-a-file")
	if err := os.WriteFile(notADir, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("DISK_IO_PATH", notADir)
	t.Setenv("DISK_IO_SIZE_MB", "1")
	t.Setenv("DISK_IO_BLOCK_KB", "64")
	t.Setenv("BURNIN_DURATION_SECONDS", "5")
	code := run()

	if code == exitFail {
		t.Errorf("exit = %d (Fail): a target the runner cannot create is a configuration fault, and "+
			"Fail is the one phase that is never retried — this permanently indicts a healthy disk on "+
			"every node sharing the mistake, with the retry budget unspent", code)
	}
	if code != exitError {
		t.Errorf("exit = %d, want %d (Error)", code, exitError)
	}
}

// A SLOW DEVICE MUST NOT BE CONDEMNED THROUGH A STARVED READ (#298).
//
// Both passes shared one deadline, and writePass stops CLEANLY when it arrives
// rather than erroring. So a device that merely wrote slower than the window
// allowed handed the read a deadline already in the past: the read moved zero
// bytes and the run reported exit 1 — a permanent hardware verdict, never
// retried, on a working disk, with no readBandwidthMBs to explain it.
func TestTheReadKeepsAShareOfTheWindowAWriteCannotConsume(t *testing.T) {
	now := time.Now()

	for _, window := range []time.Duration{time.Second, 30 * time.Second, 10 * time.Minute} {
		wd, dl := passDeadlines(now, window)

		if !wd.Before(dl) {
			t.Errorf("window %s: the write deadline (%v) is not before the run deadline (%v), so a "+
				"write that uses its whole budget leaves the read none", window, wd, dl)
		}
		if got := dl.Sub(now); got != window {
			t.Errorf("window %s: run deadline is %s away, want the full window", window, got)
		}
		// The reserved share must be a real amount of time, not a rounding
		// artefact: this is the whole point of the split.
		if reserved := dl.Sub(wd); reserved < window/5 {
			t.Errorf("window %s: only %s reserved for the read", window, reserved)
		}
	}

	// A degenerate window must not produce a deadline in the future by
	// arithmetic. Nothing is measurable in zero time, and the run says so
	// through the zero-bytes path rather than by running unbounded.
	wd, dl := passDeadlines(now, 0)
	if wd.After(now) || dl.After(now) {
		t.Errorf("a zero window produced future deadlines: write=%v run=%v", wd, dl)
	}
}

// The read half of the asymmetry the write half already observes.
//
// A read that never reached the device measured nothing about it. The write
// path has said so since it shipped ("A FAILURE TO OPEN IS NOT A FAILING
// DISK"); the read path returned exit 1 for the same class of cause.
func TestAReadThatNeverReachedTheDeviceIsAnErrorNotAFailedDisk(t *testing.T) {
	if !directSupported {
		t.Skip("no O_DIRECT on this platform; run() skips before reaching the passes (CI runs it)")
	}
	buffered(t)
	dir := t.TempDir()

	// Let the write succeed, then make the read's open fail: openFile is the
	// only seam, so refuse exactly the read-only open.
	original := openFile
	openFile = func(path string, flags int, perm os.FileMode) (*os.File, error) {
		if flags&os.O_WRONLY == 0 && flags&os.O_RDWR == 0 {
			return nil, errors.New("simulated: the mount went away before the read")
		}
		return os.OpenFile(path, flags, perm)
	}
	t.Cleanup(func() { openFile = original })

	t.Setenv("DISK_IO_PATH", dir)
	t.Setenv("DISK_IO_SIZE_MB", "1")
	t.Setenv("DISK_IO_BLOCK_KB", "64")
	t.Setenv("BURNIN_DURATION_SECONDS", "30")

	code := run()

	if code == exitFail {
		t.Errorf("exit = %d (Fail): the read never opened the file, so nothing about the device was "+
			"measured. Fail is the one phase never retried", code)
	}
	if code != exitError {
		t.Errorf("exit = %d, want %d (Error)", code, exitError)
	}
}

// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
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

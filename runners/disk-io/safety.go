// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

// What this test writes, and where.
//
// Settled here rather than discovered later, because a storage test is the one
// runner in this suite that can destroy the thing it is measuring. Four rules,
// and the default of every one of them falls towards "cannot lose data":
//
//  1. NO DECLARED PATH, NO TEST. There is no default directory — not /tmp, not
//     /var, nothing. A test that declares no hostPath gets a pod with no volumes
//     at all, and this runner behaves the same way: it SKIPS. Inventing a path
//     would mean writing to whatever filesystem the container image happens to
//     have, which measures the overlay and not the node's NVMe.
//
//  2. IT WRITES ONE FILE, WHICH IT CREATED. O_EXCL, so an existing file is
//     never opened and never truncated. The name is this runner's own, and the
//     file is removed on every exit path including the failures.
//
//  3. IT LEAVES HEADROOM. A write sized to fill the filesystem is destructive
//     even though it destroys nothing directly: a node whose root or scratch
//     volume is full stops scheduling, and a burn-in that causes that has taken
//     the fleet down while measuring it. The run is refused unless the write
//     fits with a reserve to spare.
//
//  4. IT READS BACK WHAT IT WROTE. The read measurement never touches site
//     data, so pointing this at a directory holding a dataset measures the
//     device without reading a byte of the dataset.
//
// The one thing deliberately NOT offered is a "read-only mode" over existing
// files. It sounds safer and is worse: it would make the measurement depend on
// what happens to be lying in the directory — file sizes, extents, page-cache
// residency — so two nodes with identical hardware would report different
// numbers, and a slow node could not be told from an unlucky layout.

// fileName is the name of the file this runner creates. Fixed and recognisable
// rather than random: an operator who finds one after a crash should be able to
// tell what wrote it, and O_EXCL turns a leftover into a refusal rather than a
// silent overwrite.
const fileName = ".burnin-disk-io.tmp"

// reserveBytes is the free space that must remain AFTER the test file is
// written. 2 GiB, which is not a tuning parameter so much as an assertion that
// a burn-in has no business being the reason a kubelet starts evicting for disk
// pressure.
const reserveBytes = 2 << 30

// spaceCheck is the arithmetic of rule 3, split out so it is testable without a
// filesystem.
//
// Deliberately takes free space rather than reading statfs itself: the awkward
// cases here are arithmetic ones — a device smaller than the reserve, a write
// that fits exactly, an unsigned underflow when free is already below the
// reserve — and none of them need a disk to get wrong.
func spaceCheck(freeBytes, writeBytes uint64) error {
	if writeBytes == 0 {
		return fmt.Errorf("a zero-length write measures nothing")
	}
	// Written as an addition rather than `free - write < reserve`, because that
	// subtraction underflows on an almost-full filesystem and turns the most
	// dangerous case into a pass.
	if freeBytes < writeBytes+reserveBytes {
		return fmt.Errorf(
			"writing %s would leave less than the %s reserve free (%s available): a burn-in must not be "+
				"the reason a node starts evicting for disk pressure. Point --path at a scratch volume, or "+
				"lower DISK_IO_SIZE_MB",
			humanBytes(writeBytes), humanBytes(reserveBytes), humanBytes(freeBytes))
	}
	return nil
}

// targetFile turns a declared directory into the one path this runner may write.
//
// The join is the guard: a value containing .. or an absolute path cannot
// escape, because the file name is a constant and the directory is the only
// input. It is checked anyway — a defence that costs one comparison and would
// catch a future refactor that made the name configurable.
func targetFile(dir string) (string, error) {
	if strings.TrimSpace(dir) == "" {
		return "", fmt.Errorf("no directory")
	}
	if !filepath.IsAbs(dir) {
		return "", fmt.Errorf("DISK_IO_PATH must be absolute, got %q — a relative path is resolved against "+
			"the container's working directory, which is not a place on the node", dir)
	}
	target := filepath.Join(dir, fileName)
	if filepath.Dir(target) != filepath.Clean(dir) {
		return "", fmt.Errorf("refusing to write outside %s", dir)
	}
	return target, nil
}

func humanBytes(n uint64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/(1<<20))
	}
	return fmt.Sprintf("%d B", n)
}

// passDeadlines splits the test's window so the read cannot be starved.
//
// Both passes used the SAME deadline, and writePass stops CLEANLY when it
// arrives rather than returning an error. So a device that merely wrote slower
// than the window allowed handed the read a deadline already in the past: the
// read moved zero bytes, and the run reported exit 1 — a permanent hardware
// verdict, never retried, on a working disk, with no readBandwidthMBs in the
// report to explain it (#298).
//
// A quarter is reserved rather than a half because reads are the faster
// direction on every device this runs on, and the write is the pass carrying a
// byte target it is trying to hit. A write that finishes early simply leaves
// the remainder to the read, which is the ordinary case — the reservation only
// binds when the write would otherwise have consumed everything.
//
// Pure, and takes `now`, so the split is testable on any platform. The paths
// that consume it need O_DIRECT and therefore Linux.
func passDeadlines(now time.Time, window time.Duration) (writeDeadline, deadline time.Time) {
	if window <= 0 {
		return now, now
	}
	return now.Add(window - window/4), now.Add(window)
}

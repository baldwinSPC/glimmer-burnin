// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

//go:build linux

package main

import "syscall"

// oDirect bypasses the page cache.
//
// It is the whole reason this runner can claim to measure a device. Without it
// a 1 GiB write to a node with 128 GiB of RAM measures memcpy: the numbers come
// back enormous, they look like a healthy NVMe, and a failing drive is
// certified. There is deliberately no fallback — see openDirect.
const oDirect = syscall.O_DIRECT

// directSupported says this platform has the flag at all.
const directSupported = true

// freeBytes is space available to an unprivileged writer.
//
// Bavail, not Bfree: the difference is the root-reserved portion, and this
// runner does not run as root. Using Bfree would let the space check pass on a
// filesystem where the write then fails with ENOSPC — turning a careful refusal
// into a hardware-looking failure.
func freeBytes(path string) (uint64, error) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, err
	}
	return st.Bavail * uint64(st.Bsize), nil
}

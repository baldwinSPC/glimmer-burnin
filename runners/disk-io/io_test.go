// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"
	"unsafe"
)

func TestBuffersAreAlignedBecauseODirectRefusesAnythingElse(t *testing.T) {
	// A misaligned buffer fails the write with EINVAL, which would surface as
	// an I/O error and read as a bad drive. Go offers no way to ask for aligned
	// memory, so this arithmetic is load-bearing.
	for _, size := range []int{4096, 1 << 20, 4 << 20, 4096 * 3} {
		buf := alignedBuffer(size)
		if len(buf) != size {
			t.Errorf("size %d: got a buffer of %d", size, len(buf))
		}
		if addr := uintptr(unsafe.Pointer(&buf[0])); addr%alignment != 0 {
			t.Errorf("size %d: buffer starts at %#x, which is not %d-aligned", size, addr, alignment)
		}
		// Writable through its whole length — a slicing error would show up
		// here rather than as a corrupted measurement.
		buf[0], buf[len(buf)-1] = 1, 2
	}
}

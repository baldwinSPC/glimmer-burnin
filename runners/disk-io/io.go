// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"time"
	"unsafe"
)

// alignment is the buffer and offset alignment O_DIRECT requires.
//
// 4096 rather than a queried logical block size: every device this suite will
// meet has a logical block of 512 or 4096, and a buffer aligned to 4096 is
// aligned for both. Querying it would be one more ioctl that can fail on the
// path where failing means measuring the page cache.
const alignment = 4096

// alignedBuffer returns a slice whose first byte is at a 4096-byte boundary.
//
// O_DIRECT reads and writes go from the device to THIS memory with no kernel
// copy, so the kernel requires the address, the length and the file offset all
// to be aligned; a misaligned buffer fails the write with EINVAL. Go gives no
// way to ask for aligned memory, so the trick is to over-allocate and slice
// forward to the first aligned address.
//
// The unsafe.Pointer is used only to READ the address for that arithmetic — the
// slice returned is an ordinary Go slice into an ordinary Go allocation, which
// keeps the whole buffer reachable and the GC out of the question.
func alignedBuffer(size int) []byte {
	raw := make([]byte, size+alignment)
	offset := int(uintptr(unsafe.Pointer(&raw[0])) % alignment)
	if offset != 0 {
		offset = alignment - offset
	}
	return raw[offset : offset+size]
}

// openDirect opens a file with the page cache bypassed.
//
// If O_DIRECT is not available — a filesystem that refuses it, or a platform
// without the flag — this returns an error and the caller SKIPS. There is
// deliberately no fallback to buffered I/O: a buffered measurement on a machine
// with more RAM than the test file measures memcpy, reports figures a real NVMe
// cannot reach, and certifies a failing drive. Refusing to answer is the only
// honest response, and it is a skip rather than a failure because "this
// filesystem does not support direct I/O" is a fact about the target, not a
// fault in it.
// openFile is the seam the tests substitute.
//
// A variable rather than a parameter threaded through every call: the ONLY
// reason it exists is that O_DIRECT cannot be exercised on a developer's
// machine, and the byte accounting, the deadline handling and the O_EXCL
// refusal below are all worth testing without it. Production never reassigns
// it, so the shipped path is openDirect and nothing else.
var openFile = openDirect

func openDirect(path string, flags int, perm os.FileMode) (*os.File, error) {
	if !directSupported {
		return nil, fmt.Errorf("this platform has no O_DIRECT, so nothing here can measure a device " +
			"rather than the page cache")
	}
	f, err := os.OpenFile(path, flags|oDirect, perm)
	if err != nil {
		return nil, err
	}
	return f, nil
}

// result is one direction's measurement.
type result struct {
	bytes   uint64
	elapsed time.Duration
	lat     latencies
	errors  int
}

// writePass creates the file and writes it once, sequentially.
//
// O_EXCL is rule 2 of safety.go: an existing file is never opened, so a
// leftover from a crashed run becomes a refusal instead of a silent overwrite
// of something that might not be ours at all.
func writePass(path string, totalBytes uint64, block int, deadline time.Time) (result, error) {
	f, err := openFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return result{}, err
	}
	defer f.Close()

	buf := alignedBuffer(block)
	// A repeating but non-zero pattern. All-zeroes would let a
	// compressing or deduplicating layer turn the write into almost no device
	// traffic at all, which is the storage equivalent of measuring the cache.
	for i := range buf {
		buf[i] = byte(i%251) + 1
	}

	var res result
	start := time.Now()
	for res.bytes < totalBytes {
		if time.Now().After(deadline) {
			break
		}
		opStart := time.Now()
		n, err := f.Write(buf)
		res.lat = append(res.lat, time.Since(opStart))
		if err != nil {
			res.errors++
			return res, fmt.Errorf("writing at offset %d: %w", res.bytes, err)
		}
		res.bytes += uint64(n)
	}

	// fsync BEFORE stopping the clock. Without it the measurement ends when the
	// last write returns, which under O_DIRECT is close to honest but still
	// leaves device-side caching outside the window — and on the drives most
	// worth catching, a controller that is failing is precisely one whose flush
	// is slow.
	syncStart := time.Now()
	if err := f.Sync(); err != nil {
		res.errors++
		return res, fmt.Errorf("flushing: %w", err)
	}
	res.lat = append(res.lat, time.Since(syncStart))
	res.elapsed = time.Since(start)
	return res, nil
}

// readPass reads back the file this runner wrote.
//
// Never site data — see safety.go rule 4. Pointing this at a directory holding
// a dataset measures the device without reading a byte of the dataset.
func readPass(path string, block int, deadline time.Time) (result, error) {
	f, err := openFile(path, os.O_RDONLY, 0)
	if err != nil {
		return result{}, err
	}
	defer f.Close()

	buf := alignedBuffer(block)
	var res result
	start := time.Now()
	for {
		if time.Now().After(deadline) {
			break
		}
		opStart := time.Now()
		n, err := f.Read(buf)
		if n > 0 {
			res.lat = append(res.lat, time.Since(opStart))
			res.bytes += uint64(n)
		}
		if err != nil {
			// The ordinary end of the file, not an error. Compared with
			// errors.Is rather than by message: os.File wraps its errors, and a
			// string comparison here would silently start counting every EOF as
			// an I/O error the day that wrapping changed.
			if errors.Is(err, io.EOF) {
				break
			}
			res.errors++
			return res, fmt.Errorf("reading at offset %d: %w", res.bytes, err)
		}
		if n == 0 {
			break
		}
	}
	res.elapsed = time.Since(start)
	return res, nil
}

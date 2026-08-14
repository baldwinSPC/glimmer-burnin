// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Command disk-io measures a node's storage: sequential throughput and
// operation latency against a site-declared path, with the page cache bypassed.
//
// Storage is the subsystem this suite otherwise never touches, and on an AI node
// it is not a minor one. Checkpoint writes, dataset streaming and shared-
// filesystem reads are where a training run stalls when the NVMe underneath it
// is degrading, and a drive with a thermal-throttling controller or rising media
// errors passes every other test this operator runs.
//
// It is a STANDARD LIBRARY runner, and that is a licensing decision rather than
// an aesthetic one: fio, IOR and elbencho are all GPL, and a copyleft dependency
// would make this project unpublishable. There is no third-party import here at
// all, and metricnames_test.go asserts it.
//
// Contract:
//
//	exit 0                             the device was measured
//	exit 1                             an I/O error, or a threshold-free failure
//	exit 2  DISK_IO_SKIP: <reason>     does not apply to this node
//	exit 3  DISK_IO_ERROR: <reason>    machinery, including a refused write
//
// What it writes and where is settled in safety.go. Read that before changing
// anything here: this is the one runner in the suite that can destroy the thing
// it is measuring.
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	exitPass  = 0
	exitFail  = 1
	exitSkip  = 2
	exitError = 3
)

func main() { os.Exit(run()) }

func logf(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }

func metric(key, value string) { fmt.Printf("%s=%s\n", key, value) }

// fin prints the declared marker for a non-pass outcome.
//
// The marker is required, not decorative: an exit 2 WITHOUT a recognised *_SKIP
// marker is an Error by contract, so a runner that crashes into a 2 cannot be
// mistaken for one that declared itself inapplicable.
func fin(code int, format string, args ...any) int {
	msg := fmt.Sprintf(format, args...)
	switch code {
	case exitSkip:
		fmt.Printf("DISK_IO_SKIP: %s\n", msg)
	case exitError:
		fmt.Printf("DISK_IO_ERROR: %s\n", msg)
	case exitFail:
		fmt.Printf("DISK_IO_FAIL: %s\n", msg)
	}
	logf("%s", msg)
	return code
}

func run() int {
	dir := strings.TrimSpace(os.Getenv("DISK_IO_PATH"))
	if dir == "" {
		// Rule 1: no declared path, no test. Inventing one would measure the
		// container's overlay filesystem and report it as the node's NVMe.
		return fin(exitSkip, "DISK_IO_PATH is not set, so there is no declared place to measure. "+
			"Point it at a directory this pod mounts through spec.runner.hostPaths — there is deliberately "+
			"no default, because a default would measure the container's own filesystem and call it the node's disk")
	}
	if !directSupported {
		return fin(exitSkip, "this platform has no O_DIRECT, so a measurement here would be of the page "+
			"cache rather than the device")
	}

	target, err := targetFile(dir)
	if err != nil {
		return fin(exitError, "%v", err)
	}

	sizeMB := envInt("DISK_IO_SIZE_MB", 1024)
	blockKB := envInt("DISK_IO_BLOCK_KB", 1024)
	duration := envInt("BURNIN_DURATION_SECONDS", 120)

	block := blockKB * 1024
	if block%alignment != 0 {
		return fin(exitError, "DISK_IO_BLOCK_KB=%d gives a %d-byte block, which is not a multiple of the "+
			"%d-byte alignment O_DIRECT requires", blockKB, block, alignment)
	}
	totalBytes := uint64(sizeMB) * 1024 * 1024

	free, err := freeBytes(dir)
	if err != nil {
		return fin(exitError, "could not read free space on %s: %v — the write is refused rather than "+
			"attempted blind", dir, err)
	}
	if err := spaceCheck(free, totalBytes); err != nil {
		return fin(exitError, "%v", err)
	}

	metric("diskIoPath", dir)
	logf("disk-io: %s, %d MiB in %d KiB blocks, %s free", target, sizeMB, blockKB, humanBytes(free))

	writeDeadline, deadline := passDeadlines(time.Now(), time.Duration(duration)*time.Second)

	write, werr := writePass(target, totalBytes, block, writeDeadline)

	// CLEANUP IS ARMED ONLY FOR A FILE THIS RUN CREATED, and that is why it is
	// registered HERE rather than before the write.
	//
	// It used to be armed before writePass, with the reasoning that a burn-in
	// leaving a gigabyte behind on each of a thousand nodes has done damage of
	// its own kind. True, and still handled — but it also fired on the ONE path
	// where the file is not ours: the O_EXCL refusal below. So the runner
	// declined to open a leftover from a crashed run, told the operator it had
	// been left alone and to inspect it, and then deleted it. That is worse than
	// having no check at all: without it the data is overwritten and the
	// operator knows the runner touched it; with it, the data is gone and the
	// operator has been told it is safe. Observed on a real ext4 volume (#242).
	//
	// An open that failed for any other reason created nothing, so removing is a
	// no-op there and the ordering costs nothing.
	if !isExistErr(werr) {
		defer func() {
			if err := os.Remove(target); err != nil && !os.IsNotExist(err) {
				logf("disk-io: WARNING could not remove %s: %v", target, err)
			}
		}()
	}

	if werr != nil {
		if isExistErr(werr) {
			return fin(exitError, "%s already exists. Refusing to open it: this runner only ever writes a "+
				"file it created, so a leftover from a crashed run is a refusal rather than an overwrite. "+
				"Remove it once you are satisfied it is not something else", target)
		}
		// A FAILURE TO OPEN IS NOT A FAILING DISK. Exit 1 is a hardware verdict
		// and is never retried, so it belongs only to something the run actually
		// measured about the device. A read-only mount, a directory the runner's
		// uid cannot write, a hostPath that resolves to a file — every one of
		// those stopped the measurement from happening and is exit 3.
		//
		// This shipped reporting `permission denied` as a FAILED disk, which
		// would have condemned every node sharing one bad mount with the retry
		// budget unspent — the compute-smoke v0.1.0 mistake in a new runner.
		if write.bytes == 0 {
			report(write, result{})
			return fin(exitError, "could not write to %s: %v — the measurement never started, so this "+
				"says nothing about the device", target, werr)
		}
		report(write, result{})
		// Past the first byte the device was reached and it failed, which IS a
		// verdict about the part.
		return fin(exitFail, "write failed after %s: %v", humanBytes(write.bytes), werr)
	}

	read, rerr := readPass(target, block, deadline)
	if rerr != nil {
		report(write, read)
		// The same asymmetry the write path above already observes, and for the
		// same reason. A read that never reached the device — the file could
		// not be opened, O_DIRECT was refused, the mount went away — measured
		// nothing, so it cannot be a verdict about the part.
		if read.bytes == 0 {
			return fin(exitError, "could not read back from %s: %v — the read never reached the device, "+
				"so this says nothing about it", target, rerr)
		}
		// Past the first byte the device was reached and it failed, which IS a
		// verdict about the part.
		return fin(exitFail, "read failed after %s: %v", humanBytes(read.bytes), rerr)
	}

	report(write, read)

	// NEITHER DIRECTION MOVING DATA IS AN ERROR, NOT A FAIL, and the two are
	// reported apart because "the run completed without moving data (wrote
	// 1073741824, read 0 bytes)" contradicted itself in the one message an
	// operator had to work from.
	//
	// Exit 1 is a permanent hardware verdict that is never retried. It belongs
	// only to something the run actually measured about the device, and a pass
	// that moved no bytes measured nothing — this is the compute-smoke v0.1.0
	// mistake, which this file already cites by name a few lines above.
	if write.bytes == 0 {
		return fin(exitError, "the write pass moved no data, so nothing was measured and the device is "+
			"unjudged. Check the size and duration this test was given")
	}
	if read.bytes == 0 {
		return fin(exitError, "the write moved %s but the read moved nothing, so no read was measured "+
			"and readBandwidthMBs is absent. The device is unjudged rather than failing",
			humanBytes(write.bytes))
	}
	logf("disk-io: write %.1f MB/s, read %.1f MB/s",
		throughputMBs(write.bytes, write.elapsed), throughputMBs(read.bytes, read.elapsed))
	return exitPass
}

// report emits the metrics.
//
// A direction that moved no bytes emits NOTHING for itself rather than a zero.
// A zero MB/s is a measurement nobody took, and a floor gate on it would fail a
// node for a run that never started — the same rule host-health follows when it
// cannot read /dev/kmsg.
func report(write, read result) {
	if write.bytes > 0 && write.elapsed > 0 {
		metric("writeBandwidthMBs", trim(throughputMBs(write.bytes, write.elapsed)))
	}
	if read.bytes > 0 && read.elapsed > 0 {
		metric("readBandwidthMBs", trim(throughputMBs(read.bytes, read.elapsed)))
	}

	// Latency is reported over BOTH directions together, because the metric
	// registry has one ioLatencyUs and a drive that stalls does not care which
	// way the traffic was going.
	all := append(append(latencies{}, write.lat...), read.lat...)
	if len(all) > 0 {
		metric("ioLatencyUs", trim(all.micros(all.mean())))
		metric("p99LatencyUs", trim(all.micros(all.percentile(99))))
	}

	// Always emitted, including zero. Unlike bandwidth, a zero here IS a
	// measurement: the run reached the device and counted no errors, which is
	// exactly what a gate of `ioErrors Equal 0` needs to see to pass a healthy
	// node.
	metric("ioErrors", strconv.Itoa(write.errors+read.errors))
}

// isExistErr reports whether an open failed because the file was already there.
//
// Checked in two ways because the runner's own openFile seam may wrap it: the
// distinction between "this file is not mine" and every other open failure
// decides both the exit code and whether cleanup is armed, so it must not
// depend on which layer produced the error.
func isExistErr(err error) bool {
	return err != nil && (os.IsExist(err) || strings.Contains(err.Error(), "file exists"))
}

func trim(v float64) string { return strconv.FormatFloat(v, 'f', 3, 64) }

func envInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		logf("disk-io: %s=%q is not a positive integer; using %d", name, v, def)
		return def
	}
	return n
}

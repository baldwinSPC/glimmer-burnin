// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// sysInfo is what the node will admit about its own memory and CPU.
type sysInfo struct {
	memTotalBytes     int64
	memAvailableBytes int64
	// limitBytes is the cgroup memory limit in force, or 0 when there is none.
	// It is the difference between "this node is too small" (a property of the
	// hardware, a Skip) and "this pod was given too little" (a property of the
	// profile, an Error), so it is read even though the sizing arithmetic could
	// get by with MemAvailable alone.
	limitBytes int64
	cpus       int
}

// readSysInfo probes /proc and /sys beneath root. root is productionRoot ("/")
// in production and a temporary directory in tests — the numbers this returns
// decide a node's verdict, so the arithmetic that consumes them has to be
// testable without a container. It must stay ABSOLUTE in production: an empty
// root joins to relative paths, which read the right files only while the
// process happens to be running in /.
func readSysInfo(root string) (sysInfo, error) {
	total, available, err := readMeminfo(filepath.Join(root, "proc", "meminfo"))
	if err != nil {
		return sysInfo{}, err
	}
	return sysInfo{
		memTotalBytes:     total,
		memAvailableBytes: available,
		limitBytes:        readMemoryLimit(root, total),
		cpus:              usableCPUs(root),
	}, nil
}

// readMeminfo returns MemTotal and MemAvailable in bytes.
func readMeminfo(path string) (total, available int64, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, fmt.Errorf("cannot read %s (%v) — the runner could not size the workload, so this node's memory was not judged", path, err)
	}
	var free int64
	for _, line := range strings.Split(string(b), "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		kb, perr := strconv.ParseInt(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(value), "kB")), 10, 64)
		if perr != nil {
			continue
		}
		switch strings.TrimSpace(key) {
		case "MemTotal":
			total = kb * 1024
		case "MemAvailable":
			available = kb * 1024
		case "MemFree":
			free = kb * 1024
		}
	}
	if total <= 0 {
		return 0, 0, fmt.Errorf("%s reported no MemTotal — the runner could not size the workload, so this node's memory was not judged", path)
	}
	if available <= 0 {
		// MemAvailable predates nothing we care about, but a kernel without it
		// is not a reason to give up: MemFree is a stricter answer, never a
		// looser one, so falling back can only make the workload smaller.
		available = free
	}
	if available <= 0 {
		return 0, 0, fmt.Errorf("%s reported neither MemAvailable nor MemFree — the runner could not size the workload, so this node's memory was not judged", path)
	}
	return total, available, nil
}

// readMemoryLimit returns the cgroup memory limit in bytes, or 0 when no limit
// is in force. Both cgroup layouts are read: a runner image outlives the
// distributions it runs on.
func readMemoryLimit(root string, memTotal int64) int64 {
	for _, rel := range []string{
		"sys/fs/cgroup/memory.max",                   // cgroup v2
		"sys/fs/cgroup/memory/memory.limit_in_bytes", // cgroup v1
	} {
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			continue
		}
		raw := strings.TrimSpace(string(b))
		if raw == "max" {
			return 0
		}
		v, err := strconv.ParseInt(raw, 10, 64)
		if err != nil || v <= 0 {
			continue
		}
		// cgroup v1 spells "unlimited" as a huge sentinel, and a limit at or
		// above the machine's own RAM is not a limit in any useful sense —
		// treating either as a real limit would turn "no limit" into the Error
		// branch of tooSmall and blame a profile that set nothing.
		if memTotal > 0 && v >= memTotal {
			return 0
		}
		return v
	}
	return 0
}

// usableCPUs is how many CPUs this container may actually run on.
//
// It matters for the bandwidth metrics rather than for correctness: threads
// beyond the cgroup's CPU quota do not add memory traffic, they add scheduling
// latency, so a run with more copy threads than quota reports a lower
// readBandwidthMBs for a reason that has nothing to do with the DIMMs. Two
// nodes are only comparable if the thread count was chosen the same way.
func usableCPUs(root string) int {
	n := runtime.NumCPU()
	if quota := cgroupCPUQuota(root); quota > 0 && quota < n {
		n = quota
	}
	if n < 1 {
		n = 1
	}
	return n
}

func cgroupCPUQuota(root string) int {
	if b, err := os.ReadFile(filepath.Join(root, "sys/fs/cgroup/cpu.max")); err == nil {
		fields := strings.Fields(string(b))
		if len(fields) == 2 && fields[0] != "max" {
			quota, err1 := strconv.ParseInt(fields[0], 10, 64)
			period, err2 := strconv.ParseInt(fields[1], 10, 64)
			if err1 == nil && err2 == nil && quota > 0 && period > 0 {
				return int((quota + period - 1) / period)
			}
		}
		return 0
	}
	quota, err1 := readInt(filepath.Join(root, "sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
	period, err2 := readInt(filepath.Join(root, "sys/fs/cgroup/cpu/cpu.cfs_period_us"))
	if err1 == nil && err2 == nil && quota > 0 && period > 0 {
		return int((quota + period - 1) / period)
	}
	return 0
}

func readInt(path string) (int64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
}

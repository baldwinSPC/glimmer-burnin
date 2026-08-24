// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// sysInfo is what the node will admit about its own memory. Adapted from
// runners/memory-stress's own sysInfo — same files, same v1/v2 cgroup
// handling, same MemAvailable-falls-back-to-MemFree reasoning — trimmed to
// what a single-threaded fill/hold/verify test needs: no CPU quota, since
// this runner never sizes a thread count.
type sysInfo struct {
	memTotalBytes     int64
	memAvailableBytes int64
	// limitBytes is the cgroup memory limit in force, or 0 when there is none.
	// Read for the same reason memory-stress reads it: it is what separates
	// "this node is too small" (a property of the hardware, a Skip) from "this
	// pod was given too little" (a property of the profile, an Error).
	limitBytes int64
}

// readSysInfo probes /proc and /sys beneath root. root is productionRoot ("/")
// in production and a temporary directory in tests, for the same reason
// memory-stress's own readSysInfo takes it: the arithmetic that decides a
// node's verdict has to be testable without a container, and it must stay
// ABSOLUTE in production — an empty root joins to a relative path that only
// resolves correctly while the process happens to start in /.
func readSysInfo(root string) (sysInfo, error) {
	total, available, err := readMeminfo(filepath.Join(root, "proc", "meminfo"))
	if err != nil {
		return sysInfo{}, err
	}
	return sysInfo{
		memTotalBytes:     total,
		memAvailableBytes: available,
		limitBytes:        readMemoryLimit(root, total),
	}, nil
}

// readMeminfo returns MemTotal and MemAvailable in bytes.
func readMeminfo(path string) (total, available int64, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, 0, fmt.Errorf("cannot read %s (%v) — the runner could not size the test region, so this node's memory was not judged", path, err)
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
		return 0, 0, fmt.Errorf("%s reported no MemTotal — the runner could not size the test region, so this node's memory was not judged", path)
	}
	if available <= 0 {
		// MemAvailable predates nothing we care about, but a kernel without it
		// is not a reason to give up: MemFree is a stricter answer, never a
		// looser one, so falling back can only make the test region smaller.
		available = free
	}
	if available <= 0 {
		return 0, 0, fmt.Errorf("%s reported neither MemAvailable nor MemFree — the runner could not size the test region, so this node's memory was not judged", path)
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
		// treating either as real would turn "no limit" into the Error branch
		// of tooSmall and blame a profile that set nothing.
		if memTotal > 0 && v >= memTotal {
			return 0
		}
		return v
	}
	return 0
}

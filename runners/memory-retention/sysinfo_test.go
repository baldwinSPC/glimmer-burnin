// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// fakeRoot builds a throwaway /proc and /sys tree, mirroring
// runners/memory-stress's own helper of the same name and for the same
// reason: the sizing arithmetic decides whether a node is skipped, errored or
// tested, so it has to be exercised without a real container.
func fakeRoot(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, content := range files {
		path := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

const meminfo128G = `MemTotal:       134217728 kB
MemFree:         8388608 kB
MemAvailable:  125829120 kB
Buffers:          123456 kB
`

func TestReadSysInfoFromMeminfo(t *testing.T) {
	root := fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if got, want := sys.memTotalBytes/mib, int64(131072); got != want {
		t.Errorf("memTotalBytes = %d MiB, want %d", got, want)
	}
	if got, want := sys.memAvailableBytes/mib, int64(122880); got != want {
		t.Errorf("memAvailableBytes = %d MiB, want %d", got, want)
	}
	if sys.limitBytes != 0 {
		t.Errorf("limitBytes = %d, want 0 when no cgroup limit file exists", sys.limitBytes)
	}
}

func TestReadSysInfoFallsBackToMemFree(t *testing.T) {
	root := fakeRoot(t, map[string]string{"proc/meminfo": "MemTotal: 1048576 kB\nMemFree: 524288 kB\n"})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if got, want := sys.memAvailableBytes/1024, int64(524288); got != want {
		t.Errorf("memAvailableBytes fell back to %d kB, want %d", got, want)
	}
}

func TestReadSysInfoMissingMeminfoIsAnError(t *testing.T) {
	root := t.TempDir()
	if _, err := readSysInfo(root); err == nil {
		t.Fatal("readSysInfo accepted a root with no /proc/meminfo at all")
	}
}

func TestReadSysInfoNoTotalIsAnError(t *testing.T) {
	root := fakeRoot(t, map[string]string{"proc/meminfo": "MemFree: 1024 kB\n"})
	if _, err := readSysInfo(root); err == nil {
		t.Fatal("readSysInfo accepted a meminfo with no MemTotal line")
	}
}

func TestReadMemoryLimitCgroupV2(t *testing.T) {
	root := fakeRoot(t, map[string]string{
		"proc/meminfo":             meminfo128G,
		"sys/fs/cgroup/memory.max": "1073741824", // 1 GiB
	})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if got, want := sys.limitBytes, int64(1073741824); got != want {
		t.Errorf("limitBytes = %d, want %d", got, want)
	}
}

func TestReadMemoryLimitCgroupV2MaxMeansNoLimit(t *testing.T) {
	root := fakeRoot(t, map[string]string{
		"proc/meminfo":             meminfo128G,
		"sys/fs/cgroup/memory.max": "max\n",
	})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if sys.limitBytes != 0 {
		t.Errorf("limitBytes = %d, want 0 for a literal \"max\"", sys.limitBytes)
	}
}

func TestReadMemoryLimitCgroupV1(t *testing.T) {
	root := fakeRoot(t, map[string]string{
		"proc/meminfo": meminfo128G,
		"sys/fs/cgroup/memory/memory.limit_in_bytes": "2147483648", // 2 GiB
	})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if got, want := sys.limitBytes, int64(2147483648); got != want {
		t.Errorf("limitBytes = %d, want %d", got, want)
	}
}

func TestReadMemoryLimitAtOrAboveMemTotalIsNotALimit(t *testing.T) {
	// cgroup v1's "unlimited" sentinel, and any value >= the machine's own
	// RAM: neither is a real limit, and treating either as one would blame a
	// profile that set nothing.
	root := fakeRoot(t, map[string]string{
		"proc/meminfo":             meminfo128G,
		"sys/fs/cgroup/memory.max": "999999999999999",
	})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if sys.limitBytes != 0 {
		t.Errorf("limitBytes = %d, want 0 for a sentinel at or above MemTotal", sys.limitBytes)
	}
}

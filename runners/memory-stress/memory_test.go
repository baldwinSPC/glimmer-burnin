// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakeRoot builds a throwaway /proc and /sys tree. The sizing arithmetic
// decides whether a node is skipped, errored or tested, so it is exercised
// here rather than only inside a container.
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
	// MemFree is the stricter of the two, so falling back can only shrink the
	// workload — never oversize it into an OOM kill.
	if got, want := sys.memAvailableBytes/mib, int64(512); got != want {
		t.Errorf("memAvailableBytes = %d MiB, want %d", got, want)
	}
}

func TestReadSysInfoWithoutMeminfoIsAnError(t *testing.T) {
	if _, err := readSysInfo(t.TempDir()); err == nil {
		t.Fatal("readSysInfo succeeded with no /proc/meminfo; an unsizable workload must not be silently sized")
	}
}

func TestCgroupLimits(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		want  int64 // MiB, 0 = no limit in force
	}{
		{
			name:  "v2 limit",
			files: map[string]string{"proc/meminfo": meminfo128G, "sys/fs/cgroup/memory.max": "8589934592\n"},
			want:  8192,
		},
		{
			name:  "v2 max means unlimited",
			files: map[string]string{"proc/meminfo": meminfo128G, "sys/fs/cgroup/memory.max": "max\n"},
			want:  0,
		},
		{
			name: "v1 limit",
			files: map[string]string{
				"proc/meminfo": meminfo128G,
				"sys/fs/cgroup/memory/memory.limit_in_bytes": "4294967296\n",
			},
			want: 4096,
		},
		{
			// cgroup v1's "unlimited" sentinel. Read literally it is a limit
			// larger than the machine, which would send an undersized node
			// down the Error branch and blame a profile that set nothing.
			name: "v1 unlimited sentinel",
			files: map[string]string{
				"proc/meminfo": meminfo128G,
				"sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
			},
			want: 0,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			sys, err := readSysInfo(fakeRoot(t, c.files))
			if err != nil {
				t.Fatalf("readSysInfo: %v", err)
			}
			if got := sys.limitBytes / mib; got != c.want {
				t.Errorf("limitBytes = %d MiB, want %d", got, c.want)
			}
		})
	}
}

func TestCPUQuotaClampsThreadCount(t *testing.T) {
	root := fakeRoot(t, map[string]string{
		"proc/meminfo":             meminfo128G,
		"sys/fs/cgroup/cpu.max":    "100000 100000\n", // one CPU
		"sys/fs/cgroup/memory.max": "max\n",
	})
	sys, err := readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if sys.cpus != 1 {
		t.Errorf("cpus = %d, want 1 — threads beyond the quota add scheduling latency, not memory traffic", sys.cpus)
	}

	root = fakeRoot(t, map[string]string{"proc/meminfo": meminfo128G, "sys/fs/cgroup/cpu.max": "max 100000\n"})
	sys, err = readSysInfo(root)
	if err != nil {
		t.Fatalf("readSysInfo: %v", err)
	}
	if sys.cpus != runtime.NumCPU() {
		t.Errorf("cpus = %d, want %d when no quota is set", sys.cpus, runtime.NumCPU())
	}
}

func TestPlanAutoSizesFromTheSmallerOfLimitAndAvailable(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{
		memTotalBytes:     128 * 1024 * mib,
		memAvailableBytes: 120 * 1024 * mib,
		cpus:              20,
	}

	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	availableMB := float64(120 * 1024)
	if want := int64(availableMB * defaultFraction); p.testMB != want {
		t.Errorf("testMB = %d, want %d (70%% of what is available)", p.testMB, want)
	}
	if p.threads != 20 {
		t.Errorf("threads = %d, want 20 (one per usable CPU)", p.threads)
	}

	// A cgroup limit below MemAvailable is the binding constraint: sizing off
	// the host's free memory inside a limited pod is an OOM kill, which reports
	// as an infrastructure Error and judges nothing.
	sys.limitBytes = 16 * 1024 * mib
	p, err = planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	limitMB := float64(16 * 1024)
	if want := int64(limitMB * defaultFraction); p.testMB != want {
		t.Errorf("testMB = %d, want %d (70%% of the cgroup limit)", p.testMB, want)
	}
	// Coverage is measured against the node's whole RAM, which is the number a
	// reader needs to know how much a pass is worth.
	if p.coveredPct > 10 {
		t.Errorf("coveredPct = %.2f; 11 GiB of a 128 GiB node is not 10%% of it", p.coveredPct)
	}
}

// A pod that was given too little memory is a configuration problem: the node
// was never measured, so the answer is Error. Reporting it as a Skip would
// quietly excuse a misconfigured profile from ever testing memory.
func TestTooSmallCgroupLimitIsErrorNotSkip(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{
		memTotalBytes:     128 * 1024 * mib,
		memAvailableBytes: 120 * 1024 * mib,
		limitBytes:        64 * mib,
		cpus:              4,
	}
	_, err = planRun(cfg, sys)
	var re *runnerError
	if !errors.As(err, &re) {
		t.Fatalf("planRun error = %v, want a runnerError", err)
	}
	if re.code != exitError {
		t.Fatalf("exit code = %d, want exitError (%d)", re.code, exitError)
	}
	if !strings.Contains(re.msg, "memory limit") || !strings.Contains(re.msg, "not judged") {
		t.Errorf("message %q should name the knob to turn and say the node was not judged", re.msg)
	}
}

// A machine that is genuinely this small is what exit 2 is for: the test does
// not apply to this hardware, and nobody misconfigured anything.
func TestTooSmallNodeWithoutALimitSkips(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{
		memTotalBytes:     256 * mib,
		memAvailableBytes: 200 * mib,
		cpus:              2,
	}
	_, err = planRun(cfg, sys)
	var re *runnerError
	if !errors.As(err, &re) {
		t.Fatalf("planRun error = %v, want a runnerError", err)
	}
	if re.code != exitSkip {
		t.Errorf("exit code = %d, want exitSkip (%d) — a small machine has not failed anything", re.code, exitSkip)
	}
}

// An explicit size is an instruction, not a hint: it bypasses the auto-sizing
// floor, because the operator asking for a 128 MiB region knows what they want.
func TestExplicitSizeBypassesTheFloor(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_MEMORY_MB": "128"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 256 * mib, memAvailableBytes: 200 * mib, cpus: 2}

	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun with an explicit size: %v", err)
	}
	if p.testMB != 128 || !p.explicitSize {
		t.Errorf("testMB = %d explicit = %v, want 128 and true", p.testMB, p.explicitSize)
	}
}

func TestThreadOverrideWins(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_MEMORY_THREADS": "3"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 128 * 1024 * mib, memAvailableBytes: 120 * 1024 * mib, cpus: 20}
	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	if p.threads != 3 {
		t.Errorf("threads = %d, want 3", p.threads)
	}
}

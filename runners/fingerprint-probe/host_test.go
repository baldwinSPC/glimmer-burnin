// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// gb10Sysfs builds the shape measured on a DGX Spark (spark-043a, 2026-09-27):
// cpus 0-9 at capacity 718-731, 10-19 at 997-1024, one NUMA node, one Samsung
// drive in slot 0004:01:00.0 with 8001573552 sectors.
func gb10Sysfs(t *testing.T) string {
	root := t.TempDir()
	cpu := filepath.Join(root, "devices", "system", "cpu")
	writeFile(t, filepath.Join(cpu, "present"), "0-19\n")
	for i := 0; i < 20; i++ {
		c := 718 + i%10
		if i >= 10 {
			c = 997 + (i-10)*3
		}
		writeFile(t, filepath.Join(cpu, "cpu"+strconv.Itoa(i), "cpu_capacity"), strconv.Itoa(c)+"\n")
	}
	writeFile(t, filepath.Join(root, "devices", "system", "node", "node0", "meminfo"),
		"Node 0 MemTotal:       125489732 kB\nNode 0 MemFree:        1 kB\n")

	pci := filepath.Join(root, "devices", "pci0004:00", "0004:00:00.0", "0004:01:00.0")
	ctrl := filepath.Join(root, "class", "nvme", "nvme0")
	writeFile(t, filepath.Join(ctrl, "model"), "SAMSUNG MZALC4T0HBL1-00B07              \n")
	writeFile(t, filepath.Join(ctrl, "nvme0n1", "size"), "8001573552\n")
	if err := os.MkdirAll(pci, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(pci, filepath.Join(ctrl, "device")); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestHostIdentityOnAGB10(t *testing.T) {
	h := scanHost(gb10Sysfs(t))
	if !h.cpuCountKnown || h.cpuCount != 20 {
		t.Errorf("cpuCount = %d (known %v), want 20", h.cpuCount, h.cpuCountKnown)
	}
	if h.perfCoresState != "known" || h.perfCores != 10 {
		t.Errorf("performance cores = %d (%q), want 10: the X925 cluster", h.perfCores, h.perfCoresState)
	}
	if !h.memTotalKnown || h.memTotalKiB != 125489732 {
		t.Errorf("memTotalKiB = %d (known %v)", h.memTotalKiB, h.memTotalKnown)
	}
	if len(h.nvme) != 1 {
		t.Fatalf("nvme = %+v, want one controller", h.nvme)
	}
	n := h.nvme[0]
	if n.model != "SAMSUNG MZALC4T0HBL1-00B07" || n.pciAddress != "0004:01:00.0" || n.bytes != 8001573552*512 {
		t.Errorf("nvme0 = %+v", n)
	}
}

// A lost performance cluster is the fault this field exists for: the node still
// has CPUs, passes every load test, and runs slower. Only the class count moves.
func TestALostPerformanceClusterChangesOnlyTheClassCount(t *testing.T) {
	root := gb10Sysfs(t)
	for i := 15; i < 20; i++ {
		writeFile(t, filepath.Join(root, "devices", "system", "cpu", "cpu"+strconv.Itoa(i), "cpu_capacity"), "720\n")
	}
	if h := scanHost(root); h.perfCores != 5 {
		t.Errorf("performance cores = %d, want 5", h.perfCores)
	}
}

func TestAUniformPartDeclaresNoCapacityClasses(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "devices", "system", "cpu", "present"), "0-3\n")
	if h := scanHost(root); h.perfCoresState != "n/a" || h.cpuCount != 4 {
		t.Errorf("got %+v, want 4 CPUs and n/a performance cores", h)
	}
}

// Some cores publishing a capacity and some not is not a declaration of
// anything, so the count is omitted rather than computed over the ones present.
func TestPartialCapacityDataIsOmitted(t *testing.T) {
	root := gb10Sysfs(t)
	if err := os.Remove(filepath.Join(root, "devices", "system", "cpu", "cpu19", "cpu_capacity")); err != nil {
		t.Fatal(err)
	}
	if h := scanHost(root); h.perfCoresState != "" {
		t.Errorf("perfCoresState = %q, want omitted", h.perfCoresState)
	}
}

func TestAnUnreadableNUMANodeOmitsTheMemoryTotal(t *testing.T) {
	root := gb10Sysfs(t)
	writeFile(t, filepath.Join(root, "devices", "system", "node", "node1", "meminfo"), "garbage\n")
	if h := scanHost(root); h.memTotalKnown {
		t.Errorf("memory total reported as %d KiB from a partial read", h.memTotalKiB)
	}
}

func TestParseCPUList(t *testing.T) {
	for in, want := range map[string]int{"0": 1, "0-19": 20, "0-3,8,10-11": 7} {
		got, err := parseCPUList(in)
		if err != nil || len(got) != want {
			t.Errorf("parseCPUList(%q) = %v, %v; want %d cpus", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "a", "3-1", "0-"} {
		if _, err := parseCPUList(bad); err == nil {
			t.Errorf("parseCPUList(%q) accepted", bad)
		}
	}
}

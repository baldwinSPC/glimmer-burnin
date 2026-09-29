//go:build unix

package localrun

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// A group-only device the container is not in is named with the gid to add,
// and adding that gid is what silences it — the same field both dispatchers
// read (#535).
func TestAGroupOnlyDeviceIsNamedUntilItsGroupIsAdded(t *testing.T) {
	dir := t.TempDir()
	kfd := filepath.Join(dir, "kfd")
	open := filepath.Join(dir, "open")
	for p, mode := range map[string]os.FileMode{kfd: 0o660, open: 0o666} {
		if err := os.WriteFile(p, nil, mode); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(p, mode); err != nil { // umask
			t.Fatal(err)
		}
	}
	orig := amdDeviceNodes
	amdDeviceNodes = func() []string { return []string{kfd, open} }
	defer func() { amdDeviceNodes = orig }()

	gid := int64(os.Getgid())
	gaps := DeviceGroupGaps(RunSpec{GPUAccess: GPUAMD})
	if len(gaps) != 1 || !strings.Contains(gaps[0], kfd) || strings.Contains(gaps[0], open) ||
		!strings.Contains(gaps[0], "supplementalGroups: ["+strconv.FormatInt(gid, 10)+"]") {
		t.Fatalf("gaps = %q", gaps)
	}
	if gaps := DeviceGroupGaps(RunSpec{GPUAccess: GPUAMD, GroupAdd: []int64{gid}}); len(gaps) != 0 {
		t.Errorf("group added and still reported: %q", gaps)
	}
	// A CPU-only test is handed no AMD nodes, so it has nothing to report.
	if gaps := DeviceGroupGaps(RunSpec{}); len(gaps) != 0 {
		t.Errorf("no device requested and still reported: %q", gaps)
	}
}

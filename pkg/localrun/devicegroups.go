//go:build unix

package localrun

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
)

// amdDeviceNodes are the nodes GPUAMD hands a container, as they exist on
// THIS host. A variable so a test can point it at files it controls.
var amdDeviceNodes = func() []string {
	nodes := []string{"/dev/kfd"}
	render, _ := filepath.Glob("/dev/dri/renderD*")
	return append(nodes, render...)
}

// DeviceGroupGaps names the device nodes a container will be handed and will
// not be able to open, because the node grants read-write to a GROUP the
// container's process is not in.
//
// This is the bare-metal half of #535. /dev/kfd on a Strix Halo is 0660 root:render
// with `other::---`, and every runner image runs as a non-root uid with no
// supplemental groups: the device is passed, visible, and unopenable, and an
// AMD runner then reports — honestly — that no accelerator is visible and
// SKIPS. Five runners reading "not applicable" on hardware nobody measured
// is the outcome, so the gap is named before the run, with the gid to add.
//
// A warning and never a refusal, and never an automatic --group-add: the
// operator cannot see a node's device modes, so granting the group here would
// make one profile open the device on bare metal and skip in-cluster, and two
// dispatchers must not disagree about whether a device is visible. The fix is
// the same field in both: spec.runner.supplementalGroups.
//
// It cannot see an image that runs as root, or an ACL that grants the image's
// uid directly, so it says "unless".
func DeviceGroupGaps(spec RunSpec) []string {
	paths := append([]string(nil), spec.Devices...)
	if spec.GPUAccess == GPUAMD {
		paths = append(paths, amdDeviceNodes()...)
	}
	added := make(map[uint32]bool, len(spec.GroupAdd))
	for _, g := range spec.GroupAdd {
		added[uint32(g)] = true
	}

	byGID := map[uint32][]string{}
	for _, p := range paths {
		fi, err := os.Stat(p)
		if err != nil {
			continue // a missing path is PreflightMounts' business
		}
		st, ok := fi.Sys().(*syscall.Stat_t)
		if !ok {
			continue
		}
		perm := fi.Mode().Perm()
		if perm&0o006 == 0o006 || perm&0o060 != 0o060 || added[st.Gid] {
			continue
		}
		byGID[st.Gid] = append(byGID[st.Gid], p)
	}

	gids := make([]uint32, 0, len(byGID))
	for g := range byGID {
		gids = append(gids, g)
	}
	sort.Slice(gids, func(i, j int) bool { return gids[i] < gids[j] })
	out := make([]string, 0, len(gids))
	for _, g := range gids {
		out = append(out, fmt.Sprintf(
			"%s can be opened only by group %d, which this test does not add: unless the image runs as "+
				"root, the runner will see the device and fail to open it (an AMD runner reports no "+
				"accelerator visible and skips) — add `supplementalGroups: [%d]` to spec.runner",
			strings.Join(byGID[g], ", "), g, g))
	}
	return out
}

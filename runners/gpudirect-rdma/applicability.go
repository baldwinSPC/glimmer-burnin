// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// THE APPLICABILITY DECISION LIVES HERE, IN THE RUNNER.
//
// GPUDirect RDMA lets an HCA DMA straight into accelerator memory, bypassing a
// bounce buffer in host RAM. It is not a property of the fabric and not a
// property of the GPU alone: it requires a PEER-MEMORY PROVIDER registered with
// the RDMA subsystem, which is a driver-stack fact about the node.
//
// # Why this is not a controller branch
//
// The reconciler used to carry an `allTargetsGB10` vendor check that skipped
// this test on Sparks. That was the wrong place for it, twice over: it put a
// vendor's part number in a vendor-neutral controller, and it made the answer a
// property of the node's LABELS rather than of the node's DRIVERS — so a Spark
// that somehow did have a provider would still be skipped, and a
// differently-labelled part without one would still be attempted. The controller
// branch was deleted so this decision could be made by the only component that
// can actually look: the runner, on the node, at the moment of the test.
//
// # What is looked for
//
// A peer-memory provider registers under /sys/kernel/mm/memory_peers/<name>,
// which is a GENERIC kernel interface — nvidia_peermem uses it, and so does any
// other vendor's provider. The NVIDIA module is additionally looked for by name
// so the message can say which specific thing is missing. Nothing here asks the
// GPU what model it is: the model is not the reason, the missing provider is,
// and a check that keyed off "GB10" would go stale the moment a driver release
// changed the answer for that part.
//
// # Why GB10 lands here, specifically
//
// On GB10 (DGX Spark) the CPU and GPU share one physical LPDDR5x pool over
// NVLink-C2C. There is no separate device memory for an HCA to be pointed at, so
// there is no peer-memory provider to register and NVIDIA has confirmed the
// absence is architectural rather than a packaging gap. The generic check above
// therefore reports it correctly without knowing anything about GB10 — which is
// the test of whether the check is the right one.
//
// # Skip, never Fail
//
// Every negative answer here is exit 2. A node that cannot do GPUDirect RDMA has
// not failed a hardware test; it is not a candidate for one. That asymmetry is
// deliberate and is also the safe direction for the one case this check can get
// wrong: a stack that supports GPUDirect only through dma-buf
// (ibv_reg_dmabuf_mr) registers no peer-memory provider, and would be reported
// here as inapplicable. The cost of that is a test not run, which is visible;
// the cost of the opposite mistake would be a healthy link recorded as broken,
// which is not.

// memoryPeersDir is the kernel's generic registration point for peer-memory
// providers. Its ABSENCE is the strongest single signal that no provider exists.
const memoryPeersDir = "/sys/kernel/mm/memory_peers"

// nvidiaGPUDir is where the NVIDIA driver publishes one directory per GPU it has
// bound. It is present in a container that was given a GPU by the NVIDIA
// Container Toolkit.
const nvidiaGPUDir = "/proc/driver/nvidia/gpus"

// applicability is the result of looking, in enough detail that the SKIP message
// can say what was looked at rather than merely what was concluded.
type applicability struct {
	GPUPresent bool
	// GPUs are the driver's own identifiers for what it found, for the log.
	GPUs []string
	// Providers are the peer-memory providers registered with the kernel.
	Providers []string
	// PeermemModule is true when the nvidia_peermem module is loaded, which is
	// the usual way a provider comes to exist on an NVIDIA node.
	PeermemModule bool
	// Evidence records every path that was consulted and what it said, so a
	// skip is auditable from the pod log alone.
	Evidence []string
}

// Applicable reports whether GPUDirect RDMA can be exercised on this node at
// all, and why not when it cannot.
func (a applicability) Applicable() (bool, string) {
	if !a.GPUPresent {
		return false, "no NVIDIA accelerator is visible to this pod " +
			"(neither /proc/driver/nvidia/gpus nor /dev/nvidia* is populated) — " +
			"GPUDirect RDMA needs an accelerator to DMA into; the pure-fabric equivalent of this " +
			"test is the ib-write-bw TestKind, which needs no GPU"
	}
	if len(a.Providers) == 0 && !a.PeermemModule {
		return false, fmt.Sprintf(
			"this node exposes NO GPUDirect RDMA peer-memory provider: %s does not exist and "+
				"nvidia_peermem is not loaded, so no HCA on this node can be given a handle to "+
				"accelerator memory. On NVIDIA GB10 (DGX Spark) this is ARCHITECTURAL and not a "+
				"packaging gap — the CPU and GPU share one physical memory pool over NVLink-C2C, so "+
				"there is no separate device memory for a peer-memory provider to publish. "+
				"Not applicable, not failed: the fabric itself is measured by the ib-write-bw TestKind",
			memoryPeersDir)
	}
	return true, ""
}

// inspect looks. It never errors: an unreadable path is a "no" with the reason
// recorded, because a runner that could not look must still reach a decision and
// "not applicable" is the safe one.
func inspect() applicability {
	var a applicability

	// ── the accelerator ──────────────────────────────────────────────────────
	if entries, err := os.ReadDir(nvidiaGPUDir); err == nil && len(entries) > 0 {
		a.GPUPresent = true
		for _, e := range entries {
			a.GPUs = append(a.GPUs, e.Name()+" ("+gpuModel(filepath.Join(nvidiaGPUDir, e.Name(), "information"))+")")
		}
		a.Evidence = append(a.Evidence, fmt.Sprintf("%s: %d GPU(s) — %s", nvidiaGPUDir, len(entries), strings.Join(a.GPUs, ", ")))
	} else {
		a.Evidence = append(a.Evidence, nvidiaGPUDir+": absent or empty")
	}
	if !a.GPUPresent {
		if devs, _ := filepath.Glob("/dev/nvidia[0-9]*"); len(devs) > 0 {
			a.GPUPresent = true
			a.Evidence = append(a.Evidence, "/dev/nvidia*: "+strings.Join(devs, ", "))
		} else {
			a.Evidence = append(a.Evidence, "/dev/nvidia*: absent")
		}
	}

	// ── the peer-memory provider ─────────────────────────────────────────────
	if entries, err := os.ReadDir(memoryPeersDir); err == nil {
		for _, e := range entries {
			a.Providers = append(a.Providers, e.Name())
		}
		sort.Strings(a.Providers)
		if len(a.Providers) == 0 {
			a.Evidence = append(a.Evidence, memoryPeersDir+": exists but no provider is registered")
		} else {
			a.Evidence = append(a.Evidence, memoryPeersDir+": "+strings.Join(a.Providers, ", "))
		}
	} else {
		a.Evidence = append(a.Evidence, memoryPeersDir+": absent (no provider has ever registered)")
	}

	a.PeermemModule = moduleLoaded("nvidia_peermem") || moduleLoaded("nv_peer_mem")
	if a.PeermemModule {
		a.Evidence = append(a.Evidence, "nvidia_peermem: loaded")
	} else {
		a.Evidence = append(a.Evidence, "nvidia_peermem: not loaded")
	}

	return a
}

// gpuModel reads the driver's own model string. GB10 reports "Unknown" here,
// which is itself worth printing: it is the driver's answer, not a failure to
// read, and a reader who sees it should not go looking for a parse bug.
func gpuModel(informationPath string) string {
	b, err := os.ReadFile(informationPath)
	if err != nil {
		return "model unreadable"
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "Model:") {
			m := strings.TrimSpace(strings.TrimPrefix(line, "Model:"))
			if m == "" {
				return "model empty"
			}
			return "model " + m
		}
	}
	return "model not reported"
}

// moduleLoaded reports whether a kernel module is loaded. /proc/modules is read
// rather than /sys/module because a builtin module has no /sys/module entry on
// some kernels, and both are checked for the same reason.
func moduleLoaded(name string) bool {
	if _, err := os.Stat("/sys/module/" + name); err == nil {
		return true
	}
	b, err := os.ReadFile("/proc/modules")
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(b), "\n") {
		if f := strings.Fields(line); len(f) > 0 && f[0] == name {
			return true
		}
	}
	return false
}

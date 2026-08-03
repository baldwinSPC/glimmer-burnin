// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
)

// The decision this runner exists to own. Each case is a state a real node can
// be in, and the assertion is on the DIRECTION of the answer: everything that is
// not a measurement must be Skip, never Fail.
func TestApplicable(t *testing.T) {
	for _, tc := range []struct {
		name string
		app  applicability
		want bool
		// mentions is a phrase the SKIP reason must contain, because the reason
		// is what an engineer reads before deciding whether to go and look at
		// the node. "not applicable" with no subject sends them anyway.
		mentions string
	}{
		{
			// The GB10 / DGX Spark case, and the one verified on real silicon:
			// the accelerator is there, the fabric is there, and no peer-memory
			// provider exists or can.
			name:     "gpu present, no peer-memory provider (GB10)",
			app:      applicability{GPUPresent: true},
			want:     false,
			mentions: "GPUDirect RDMA peer-memory provider",
		},
		{
			name:     "no accelerator at all",
			app:      applicability{},
			want:     false,
			mentions: "no NVIDIA accelerator is visible",
		},
		{
			// A provider registered under the generic kernel interface, whoever
			// owns it. The check must not require it to be NVIDIA's.
			name: "generic peer-memory provider registered",
			app:  applicability{GPUPresent: true, Providers: []string{"some_vendor_mem"}},
			want: true,
		},
		{
			// nvidia_peermem loaded but /sys/kernel/mm/memory_peers not yet
			// readable (it is not mounted in every container). The module is
			// enough: refusing here would skip a node that can do the test.
			name: "nvidia_peermem loaded without a readable memory_peers",
			app:  applicability{GPUPresent: true, PeermemModule: true},
			want: true,
		},
		{
			// A provider without a GPU is not a runnable state, and the GPU
			// check must come first so the message says the useful thing.
			name:     "provider but no accelerator",
			app:      applicability{Providers: []string{"nv_mem"}},
			want:     false,
			mentions: "no NVIDIA accelerator is visible",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, why := tc.app.Applicable()
			if got != tc.want {
				t.Fatalf("Applicable() = %v (%q), want %v", got, why, tc.want)
			}
			if tc.want {
				if why != "" {
					t.Errorf("an applicable node should carry no reason, got %q", why)
				}
				return
			}
			if !strings.Contains(why, tc.mentions) {
				t.Errorf("reason %q does not mention %q", why, tc.mentions)
			}
		})
	}
}

// The GB10 skip must say WHY it is architectural. Without that, the next
// engineer's first move is to try to install nvidia_peermem on a part that
// cannot use it, and their second is to file a bug against this runner.
func TestGB10SkipExplainsItself(t *testing.T) {
	_, why := applicability{GPUPresent: true}.Applicable()
	for _, want := range []string{"GB10", "NVLink-C2C", "ARCHITECTURAL", "ib-write-bw"} {
		if !strings.Contains(why, want) {
			t.Errorf("the skip reason should mention %q so it is actionable; got:\n%s", want, why)
		}
	}
}

func TestGPUModel(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) string {
		p := dir + "/" + name
		if err := osWriteFile(p, content); err != nil {
			t.Fatal(err)
		}
		return p
	}
	// GB10's driver genuinely reports "Unknown" here. That is the driver's
	// answer, not a parse failure, and it must be passed through rather than
	// turned into an error a reader would chase.
	if got := gpuModel(write("gb10", "Model: \t\t Unknown\nIRQ: \t\t 490\n")); got != "model Unknown" {
		t.Errorf("gpuModel = %q, want %q", got, "model Unknown")
	}
	if got := gpuModel(write("named", "Model: \t\t NVIDIA H100 80GB HBM3\n")); got != "model NVIDIA H100 80GB HBM3" {
		t.Errorf("gpuModel = %q", got)
	}
	if got := gpuModel(dir + "/absent"); got != "model unreadable" {
		t.Errorf("gpuModel on a missing file = %q", got)
	}
}

// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// The seam between this runner and the operator; see the same file in
// runners/ib-write-bw for the full rationale. These imports are test-only and
// the image build strips *_test.go, which is what lets the runner's own sources
// stay standard-library-only.

const kind = "gpudirect-rdma"

func osWriteFile(path, content string) error { return os.WriteFile(path, []byte(content), 0o600) }

const clientStdout = `[burnin-gpudirect-rdma] applicability: /sys/kernel/mm/memory_peers: nv_mem
| bw |  1048576    71244            0.00               88.10  		   0.010500
bw_average=88.10
t_avg_usec=2.41
[burnin-gpudirect-rdma] 1048576 bytes x 71244 iterations device-to-device over roceP2p1s0f1 to 10.252.26.239
GPUDIRECT_RDMA_PASS: measured 88.10 Gb/s average into device memory over roceP2p1s0f1 to 10.252.26.239 with 4 QPs at 1048576-byte messages; acceptance is the profile's thresholds to decide
`

func TestEmittedKeysReachTheirCanonicalNames(t *testing.T) {
	res := runner.Parse(kind, clientStdout, 0)
	if res.Verdict != runner.VerdictPass {
		t.Fatalf("verdict = %v, want Pass", res.Verdict)
	}
	for name, want := range map[string]string{
		"bandwidthGbps": "88.10",
		"latencyUs":     "2.41",
	} {
		got, ok := res.Metrics[name]
		if !ok {
			t.Errorf("metric %q missing; the operator got %v", name, res.Metrics)
			continue
		}
		if got != want {
			t.Errorf("metric %q = %q, want %q", name, got, want)
		}
	}
	if len(res.InvalidNames) != 0 {
		t.Errorf("invalid metric names: %v", res.InvalidNames)
	}
	if err := contract.ValidateMetrics(res.Metrics); err != nil {
		t.Errorf("ValidateMetrics: %v", err)
	}
}

// bandwidthGbps is deliberately shared with the ib-write-bw runner: same
// measurand, same link. If this ever needs a name of its own, the reason must be
// that it stopped being the same quantity — not that sharing felt ambiguous.
func TestBandwidthSharesTheCanonicalNameWithIbWriteBw(t *testing.T) {
	m, ok := contract.Lookup("bandwidthGbps")
	if !ok {
		t.Fatal("bandwidthGbps is not registered")
	}
	if !m.SafeToThresholdOn() {
		t.Error("bandwidthGbps must be safe to threshold on: it is this runner's acceptance number")
	}
}

// The whole point of this runner on GB10. Exit 2 must arrive as Skip, because a
// node that cannot do GPUDirect RDMA has not failed anything.
func TestSkipOnGB10IsSkipNotFail(t *testing.T) {
	_, why := applicability{GPUPresent: true}.Applicable()
	stdout := "GPUDIRECT_RDMA_SKIP: " + why + "\n"
	res := runner.Parse(kind, stdout, exitSkip)
	if res.Verdict != runner.VerdictSkip {
		t.Fatalf("verdict = %v, want Skip — a Fail here would permanently indict a healthy link", res.Verdict)
	}
	if len(res.Metrics) != 0 {
		t.Errorf("a skipped run must report no measurements, got %v", res.Metrics)
	}
	// And it must not be laundered into an "unmeasurable" declaration either:
	// the n/a sentinel says "this hardware cannot produce THIS METRIC", whereas
	// this runner is saying "this hardware cannot run this TEST".
	if len(res.Unmeasurable) != 0 {
		t.Errorf("a skipped run must declare nothing unmeasurable, got %v", res.Unmeasurable)
	}
	if !strings.Contains(res.Message, "GPUDIRECT_RDMA_SKIP") {
		t.Errorf("Message = %q, want the skip marker", res.Message)
	}
}

func TestCudaFlagIsOneBased(t *testing.T) {
	// perftest numbers CUDA devices from 1. cudaDevice is stored zero-based, so
	// device 0 must become --use_cuda=1; an off-by-one would measure the wrong
	// accelerator on a multi-GPU node and never say so.
	if got := cudaFlag(plan{cudaDevice: 0}); got != "--use_cuda=1" {
		t.Errorf("cudaFlag(0) = %q, want --use_cuda=1", got)
	}
	if got := cudaFlag(plan{cudaDevice: 3}); got != "--use_cuda=4" {
		t.Errorf("cudaFlag(3) = %q, want --use_cuda=4", got)
	}
}

// The CUDA flag is what distinguishes this runner from ib-write-bw. Losing it
// would turn this into a duplicate host-memory test reporting under a name that
// claims device memory.
func TestPerftestArgsAlwaysUseCuda(t *testing.T) {
	p := rdmaPort{Device: "roceP2p1s0f1", Port: 1, LinkLayer: linkLayerEthernet, GIDIndex: 3}
	cfg := plan{perftestPort: 18525, messageBytes: 1 << 20, qps: 4, bwSeconds: 40, latIters: 5000}
	for _, args := range [][]string{bwArgs(p, cfg, "10.0.0.1"), latArgs(p, cfg, "10.0.0.1")} {
		joined := strings.Join(args, " ")
		if !strings.Contains(joined, "--use_cuda=") {
			t.Errorf("missing --use_cuda in %q", joined)
		}
	}
	if !strings.Contains(strings.Join(bwArgs(p, cfg, ""), " "), "--report_gbits") {
		t.Error("bwArgs must always pass --report_gbits")
	}
}

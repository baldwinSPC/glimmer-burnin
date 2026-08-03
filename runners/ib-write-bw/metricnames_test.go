// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// This file is the seam between this runner and the operator.
//
// The runner prints its OWN key names; pkg/runner maps them to the canonical
// names a threshold is written against, and pkg/contract owns the grammar those
// names must obey. Nothing at runtime reconciles a disagreement: a key that
// normalises to an illegal name, or onto another key's name, is silently
// unthresholdable, and the symptom is a link nobody can accept for reasons
// nobody can see.
//
// These imports are test-only. The image build strips *_test.go, which is what
// lets the runner's own sources stay standard-library-only.

const kind = "ib-write-bw"

// clientStdout is what runClient prints on a good run, in order. Metrics first,
// decision last — the ordering is part of the contract, and the Message the
// operator records is the LAST non-metric line.
const clientStdout = `[burnin-ib-write-bw] role=client peer=server.svc.ns.svc (node spark-85a9) duration=60s
| bw | 1048576    71244            0.00               99.61  		   0.011874
bw_average=99.61
latency_us=1.87
[burnin-ib-write-bw] 1048576 bytes x 71244 iterations over roceP2p1s0f1 to 10.252.161.209
IB_WRITE_BW_PASS: measured 99.61 Gb/s average over roceP2p1s0f1 to 10.252.161.209 with 4 QPs at 1048576-byte messages; acceptance is the profile's thresholds to decide
`

func TestEmittedKeysReachTheirCanonicalNames(t *testing.T) {
	res := runner.Parse(kind, clientStdout, 0)

	if res.Verdict != runner.VerdictPass {
		t.Fatalf("verdict = %v, want Pass", res.Verdict)
	}
	for name, want := range map[string]string{
		"bandwidthGbps": "99.61",
		"latencyUs":     "1.87",
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
	// Every metric emitted must obey the grammar, registered or not.
	if err := contract.ValidateMetrics(res.Metrics); err != nil {
		t.Errorf("ValidateMetrics: %v", err)
	}
}

// The runner's log lines must not be mistaken for metrics. perftest's own
// output is echoed with a "| " prefix precisely so that a stray "=" in a banner
// cannot become a key.
func TestEchoedPerftestOutputIsNotParsedAsMetrics(t *testing.T) {
	res := runner.Parse(kind, "| bw |  Max inline data : 0[B]\n| bw |  key=value inside perftest output\nbw_average=99.61\n", 0)
	if len(res.Metrics) != 1 {
		t.Fatalf("expected exactly one metric, got %v", res.Metrics)
	}
	if _, ok := res.Metrics["bandwidthGbps"]; !ok {
		t.Errorf("bandwidthGbps missing, got %v", res.Metrics)
	}
}

// The message the operator records is the last non-metric line, which is why
// the decision marker is printed after the metrics rather than before them.
func TestDecisionIsTheRecordedMessage(t *testing.T) {
	res := runner.Parse(kind, clientStdout, 0)
	if got := res.Message; len(got) < len("IB_WRITE_BW_PASS") || got[:len("IB_WRITE_BW_PASS")] != "IB_WRITE_BW_PASS" {
		t.Errorf("Message = %q, want it to start with the decision marker", got)
	}
}

// Both canonical names must be safe to gate a fleet on: a profile author writing
// `bandwidthGbps >= 89` should not be rejected at admission.
func TestEmittedMetricsAreThresholdable(t *testing.T) {
	for _, name := range []string{"bandwidthGbps", "latencyUs"} {
		if !contract.SafeToThresholdOn(name) {
			t.Errorf("%s is not safe to threshold on, but it is this runner's acceptance number", name)
		}
	}
}

// peakBandwidthGbps is registered and this runner deliberately never emits it,
// because duration mode does not measure a peak. The test records that as an
// intentional omission so a future edit does not "fix" it by emitting the 0.00
// perftest prints.
func TestPeakBandwidthIsNeverEmitted(t *testing.T) {
	res := runner.Parse(kind, clientStdout, 0)
	if _, ok := res.Metrics["peakBandwidthGbps"]; ok {
		t.Error("peakBandwidthGbps was emitted; duration mode does not measure a peak and perftest prints 0.00 there")
	}
	if res.Unmeasurable["peakBandwidthGbps"] {
		t.Error("peakBandwidthGbps was declared n/a; the sentinel is for hardware that cannot produce a measurement, " +
			"not for a measurement this runner's chosen mode does not take")
	}
}

// The skip path is the one a node with no fabric takes, and it must arrive as
// Skip rather than as Fail.
func TestSkipExitCodeIsSkip(t *testing.T) {
	res := runner.Parse(kind, "IB_WRITE_BW_SKIP: this node has no RDMA device\n", exitSkip)
	if res.Verdict != runner.VerdictSkip {
		t.Errorf("verdict = %v, want Skip", res.Verdict)
	}
}

func TestErrorExitCodeIsError(t *testing.T) {
	res := runner.Parse(kind, "IB_WRITE_BW_ERROR: no client connected\n", exitError)
	if res.Verdict != runner.VerdictError {
		t.Errorf("verdict = %v, want Error", res.Verdict)
	}
}

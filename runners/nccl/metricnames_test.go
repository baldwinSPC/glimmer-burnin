// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// The seam between this runner and the operator; see the same file in
// runners/ib-write-bw for the full rationale. These imports are test-only and
// the image build strips *_test.go, which is what lets the runner's own sources
// stay standard-library-only.

const kind = "nccl"

// What runClient prints on a good run: the whole sweep as log lines, then
// exactly one value per metric, then the decision.
const clientStdout = `[burnin-nccl] NCCL v2.27.7-1; role=client peer=server.svc.ns.svc (node spark-85a9) duration=60s gpus=1
| RESULT size_bytes=8 time_us=41.180 algbw_gbs=0.0002 busbw_gbs=0.0002
| RESULT size_bytes=268435456 time_us=22400.000 algbw_gbs=11.9838 busbw_gbs=11.9838
[burnin-nccl]        8 B       41.2 us  alg    0.00 GB/s  bus    0.00 GB/s
[burnin-nccl]   256 MiB    22400.0 us  alg   11.98 GB/s  bus   11.98 GB/s
busbw=11.98
algbw=11.98
latency_us=41.18
wrong_count=0
NCCL_PASS: peak bus bandwidth 11.98 GB/s at 256 MiB over 2 ranks with 10.252.26.239; acceptance is the profile's thresholds to decide
`

func TestEmittedKeysReachTheirCanonicalNames(t *testing.T) {
	res := runner.Parse(kind, clientStdout, 0)
	if res.Verdict != runner.VerdictPass {
		t.Fatalf("verdict = %v, want Pass", res.Verdict)
	}
	for name, want := range map[string]string{
		"busBandwidthGBs": "11.98",
		"algBandwidthGBs": "11.98",
		"latencyUs":       "41.18",
		"miscompares":     "0",
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

// THE REASON results.go EXISTS, asserted through the REAL parser.
//
// pkg/runner is last-occurrence-wins. If this runner emitted busbw once per
// message size, the operator would record only the last one — here the 8-byte
// sample, whose bus bandwidth is 0.00 GB/s. That would fail every threshold on
// every healthy link, and nothing in the pipeline would say why.
func TestOneValuePerMetricSurvivesLastOccurrenceWins(t *testing.T) {
	perSize := `busbw=11.98
algbw=11.98
busbw=0.0002
algbw=0.0002
`
	bad := runner.Parse(kind, perSize, 0)
	if bad.Metrics["busBandwidthGBs"] != "0.0002" {
		t.Fatalf("precondition failed: the parser is no longer last-occurrence-wins (got %q)",
			bad.Metrics["busBandwidthGBs"])
	}

	// The real output emits each name once, so the value the operator records is
	// the one the runner chose rather than the one that happened to print last.
	good := runner.Parse(kind, clientStdout, 0)
	if got := good.Metrics["busBandwidthGBs"]; got != "11.98" {
		t.Errorf("busBandwidthGBs = %q, want the peak 11.98", got)
	}
	if strings.Count(clientStdout, "\nbusbw=") != 1 {
		t.Errorf("busbw must be printed exactly once; the sweep belongs in the log, not in metrics")
	}
	if strings.Count(clientStdout, "\nalgbw=") != 1 {
		t.Errorf("algbw must be printed exactly once")
	}
}

// The sweep lines are evidence, not metrics. They must not be parsed as
// key=value pairs, or "size_bytes" and friends would land in the envelope as
// invalid names.
func TestSweepLinesAreNotParsedAsMetrics(t *testing.T) {
	res := runner.Parse(kind, clientStdout, 0)
	for _, forbidden := range []string{"sizeBytes", "timeUs", "algbwGbs", "busbwGbs"} {
		if _, ok := res.Metrics[forbidden]; ok {
			t.Errorf("%q leaked into the metrics from an echoed RESULT line", forbidden)
		}
	}
	if len(res.Metrics) != 4 {
		t.Errorf("expected exactly 4 metrics, got %v", res.Metrics)
	}
}

func TestEmittedMetricsAreThresholdable(t *testing.T) {
	for _, name := range []string{"busBandwidthGBs", "algBandwidthGBs", "latencyUs", "miscompares"} {
		if !contract.SafeToThresholdOn(name) {
			t.Errorf("%s is not safe to threshold on", name)
		}
	}
}

// The Node-scope skip on a one-GPU part. This is the DGX Spark case and it must
// be Skip, not Fail: a node with one accelerator has not failed a collective it
// cannot form.
func TestNodeScopeSkipIsSkip(t *testing.T) {
	res := runner.Parse(kind, "NCCL_SKIP: this node has 1 accelerator(s) and an all-reduce needs at least 2 ranks\n", exitSkip)
	if res.Verdict != runner.VerdictSkip {
		t.Fatalf("verdict = %v, want Skip", res.Verdict)
	}
	if len(res.Metrics) != 0 {
		t.Errorf("a skipped run must report no measurements, got %v", res.Metrics)
	}
}

// Node scope's own key set — busbw/algbw/latency/wrong_count unchanged, plus
// ranks and (best-effort) nccl_transport. CONSTRUCTED, not captured: this
// project has no multi-GPU node, but every key on this line is one main.go
// itself prints (metric() calls this file's own code makes), unlike the NCCL
// transport-line SCANNING in classifyTransport, which genuinely does need a
// real capture and is tested separately, with its own fixtures marked as
// constructed, in nodescope_test.go.
const nodeScopeStdout = `[burnin-nccl] NCCL v2.27.7-1; Node scope, collective=allreduce, gpus=8
| nccl_pair: NCCL runtime version 2.27.7; collective=allreduce
| RESULT size_bytes=8 time_us=12.400 algbw_gbs=0.0006 busbw_gbs=0.0011
| RESULT size_bytes=268435456 time_us=9800.000 algbw_gbs=27.3934 busbw_gbs=47.9384
ranks=8
nccl_transport=nvlink
busbw=47.94
algbw=27.39
latency_us=12.40
wrong_count=0
NCCL_PASS: peak bus bandwidth 47.94 GB/s at 256 MiB over 8 ranks (allreduce) with 8 devices on this node; acceptance is the profile's thresholds to decide
`

func TestNodeScopeEmittedKeysReachTheirCanonicalNames(t *testing.T) {
	res := runner.Parse(kind, nodeScopeStdout, 0)
	if res.Verdict != runner.VerdictPass {
		t.Fatalf("verdict = %v, want Pass", res.Verdict)
	}
	for name, want := range map[string]string{
		"busBandwidthGBs": "47.94",
		"algBandwidthGBs": "27.39",
		"latencyUs":       "12.40",
		"miscompares":     "0",
		"ranks":           "8",
		"ncclTransport":   "nvlink",
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

// ranks and ncclTransport must both be Evidence, never Acceptance: ranks is a
// bookkeeping fact and ncclTransport is a label whose parsing is best-effort
// by design (see classifyTransport). Neither should ever be able to gate a
// fleet the way busBandwidthGBs does.
func TestNodeScopeMetricsAreEvidenceNotAcceptance(t *testing.T) {
	for _, name := range []string{"ranks", "ncclTransport"} {
		if contract.SafeToThresholdOn(name) == false {
			continue // Evidence, as expected — SafeToThresholdOn false is correct here
		}
		t.Errorf("%s: registry does not mark this Evidence (or leaves it unregistered), "+
			"so a threshold on it would pass every authoring-time check while gating on a label", name)
	}
}

// A wrong answer from the collective is a FAIL, not an Error: the hardware ran
// the test and returned bad data, which is a hardware verdict.
func TestMiscompareIsFail(t *testing.T) {
	res := runner.Parse(kind, "wrong_count=17\nNCCL_FAIL: the all-reduce returned 17 incorrect element(s)\n", exitFail)
	if res.Verdict != runner.VerdictFail {
		t.Fatalf("verdict = %v, want Fail", res.Verdict)
	}
	if res.Metrics["miscompares"] != "17" {
		t.Errorf("miscompares = %q, want 17", res.Metrics["miscompares"])
	}
}

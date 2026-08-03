// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"strings"
	"testing"
)

// A real nccl_pair sweep, shaped as the harness prints it.
const realSweep = `nccl_pair: rank 1 of 2 on NVIDIA GB10 (sm_121)
RESULT size_bytes=8 time_us=41.180 algbw_gbs=0.0002 busbw_gbs=0.0002
RESULT size_bytes=1048576 time_us=228.500 algbw_gbs=4.5889 busbw_gbs=4.5889
RESULT size_bytes=8388608 time_us=800.100 algbw_gbs=10.4844 busbw_gbs=10.4844
RESULT size_bytes=67108864 time_us=5700.000 algbw_gbs=11.7735 busbw_gbs=11.7735
RESULT size_bytes=268435456 time_us=22400.000 algbw_gbs=11.9838 busbw_gbs=11.9838
RESULT_WRONG 0
`

func TestParseSweep(t *testing.T) {
	s, err := parseSweep(realSweep)
	if err != nil {
		t.Fatalf("parseSweep: %v", err)
	}
	if len(s.Samples) != 5 {
		t.Fatalf("got %d samples, want 5", len(s.Samples))
	}
	if !s.WrongReported || s.Wrong != 0 {
		t.Errorf("wrong = %d (reported %v), want 0 reported", s.Wrong, s.WrongReported)
	}
}

// THE CONSTRAINT THE WHOLE FILE EXISTS FOR. The parser downstream is
// last-occurrence-wins, so the runner must promote exactly one sample. It must
// be the PEAK, not the last row printed, or the reported bandwidth would depend
// on the order the sweep happened to be listed in.
func TestPeakIsTheBestSampleNotTheLast(t *testing.T) {
	// Deliberately descending, so "last" and "peak" differ.
	out := `RESULT size_bytes=268435456 time_us=22400.000 algbw_gbs=11.9838 busbw_gbs=11.9838
RESULT size_bytes=1048576 time_us=228.500 algbw_gbs=4.5889 busbw_gbs=4.5889
RESULT_WRONG 0
`
	s, err := parseSweep(out)
	if err != nil {
		t.Fatalf("parseSweep: %v", err)
	}
	peak, ok := s.Peak()
	if !ok {
		t.Fatal("no peak")
	}
	if peak.Bytes != 268435456 {
		t.Errorf("peak came from the %d-byte sample, want the 268435456-byte one — "+
			"a last-row-wins reduction would report %v GB/s instead of %v",
			peak.Bytes, 4.5889, 11.9838)
	}
}

// algbw and busbw must come from the SAME sample. Reporting the peak of each
// independently could pair two different message sizes and produce a ratio that
// exists nowhere in the data.
func TestPeakIsCoherentAcrossBothBandwidths(t *testing.T) {
	s, err := parseSweep(realSweep)
	if err != nil {
		t.Fatalf("parseSweep: %v", err)
	}
	peak, _ := s.Peak()
	if peak.AlgGBs != 11.9838 || peak.BusGBs != 11.9838 {
		t.Errorf("peak = alg %v / bus %v, want both from the 256 MiB sample", peak.AlgGBs, peak.BusGBs)
	}
}

// Latency must come from the SMALLEST message, where the number is a latency.
// Taking it from the peak-bandwidth sample would report 22 ms as this link's
// latency.
func TestSmallestIsUsedForLatency(t *testing.T) {
	s, err := parseSweep(realSweep)
	if err != nil {
		t.Fatalf("parseSweep: %v", err)
	}
	small, ok := s.Smallest()
	if !ok {
		t.Fatal("no smallest sample")
	}
	if small.Bytes != 8 {
		t.Errorf("smallest = %d bytes, want 8", small.Bytes)
	}
	if small.TimeUs != 41.180 {
		t.Errorf("latency = %v us, want 41.180", small.TimeUs)
	}
}

// At two ranks the bus/alg factor is exactly 1. This is asserted so that nobody
// later "fixes" an apparent duplicate by dropping one of the two metrics: they
// diverge the moment the rank count does.
func TestBusBandwidthFactor(t *testing.T) {
	if got := busBandwidthFactor(2); math.Abs(got-1.0) > 1e-12 {
		t.Errorf("factor at n=2 = %v, want exactly 1", got)
	}
	if got := busBandwidthFactor(4); math.Abs(got-1.5) > 1e-12 {
		t.Errorf("factor at n=4 = %v, want 1.5", got)
	}
	if got := busBandwidthFactor(8); math.Abs(got-1.75) > 1e-12 {
		t.Errorf("factor at n=8 = %v, want 1.75", got)
	}
}

// A run that never reached the correctness check must not report a zero
// miscompare count: a crashed collective would otherwise look clean.
func TestMiscompareCountIsNotInventedWhenAbsent(t *testing.T) {
	s, err := parseSweep("RESULT size_bytes=8 time_us=41.180 algbw_gbs=0.0002 busbw_gbs=0.0002\n")
	if err != nil {
		t.Fatalf("parseSweep: %v", err)
	}
	if s.WrongReported {
		t.Error("WrongReported = true for output with no RESULT_WRONG line")
	}
}

func TestParseSweepRejectsEmptyOutput(t *testing.T) {
	if _, err := parseSweep("nccl error: unhandled system error\n"); err == nil {
		t.Fatal("parseSweep accepted output with no results")
	}
}

func TestParseSweepRejectsAMalformedRow(t *testing.T) {
	if _, err := parseSweep("RESULT size_bytes=notanumber time_us=1 algbw_gbs=1 busbw_gbs=1\n"); err == nil {
		t.Fatal("parseSweep accepted a malformed row")
	}
}

func TestHumanBytes(t *testing.T) {
	for in, want := range map[int64]string{
		8: "8 B", 1024: "1 KiB", 1 << 20: "1 MiB", 268435456: "256 MiB", 1500: "1500 B",
	} {
		if got := humanBytes(in); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", in, got, want)
		}
	}
}

// The harness command line is the contract between the wrapper and nccl_pair.
func TestHarnessArgs(t *testing.T) {
	cfg := plan{bootstrapPort: 18535, warmup: 5, iters: 20}
	server := strings.Join(harnessArgs(0, "", cfg), " ")
	if strings.Contains(server, "--peer") {
		t.Errorf("rank 0 must not be given a peer: %q", server)
	}
	client := strings.Join(harnessArgs(1, "10.252.26.239", cfg), " ")
	if !strings.Contains(client, "--peer 10.252.26.239") || !strings.Contains(client, "--rank 1") {
		t.Errorf("rank 1 args wrong: %q", client)
	}
	if !strings.Contains(client, "--port 18535") {
		t.Errorf("both ranks must agree on the bootstrap port: %q", client)
	}
}

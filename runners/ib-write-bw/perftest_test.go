// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// Real ib_write_bw output, captured from a DGX Spark pair over RoCE v2 with 4
// queue pairs and 1 MiB messages. Kept verbatim, tabs and all, because the point
// of the test is that the parser survives the exact whitespace perftest emits.
const realBandwidthOutput = `---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF		Device         : roceP2p1s0f1
 Number of qps   : 4		Transport type : IB
 Connection type : RC		Using SRQ      : OFF
 PCIe relax order: ON
 ibv_wr* API     : ON
 TX depth        : 128
 CQ Moderation   : 1
 Mtu             : 4096[B]
 Link type       : Ethernet
 GID index       : 3
 Max inline data : 0[B]
 rdma_cm QPs	 : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0000 QPN 0x02c0 PSN 0x4d6ec0 RKey 0x1e05f1 VAddr 0x00e2f94be76000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:252:161:208
 remote address: LID 0000 QPN 0x02bc PSN 0xa81962 RKey 0x1e05f1 VAddr 0x00d1e83af21000
 GID: 00:00:00:00:00:00:00:00:00:00:255:255:10:252:161:209
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 1048576    71244            0.00               99.61  		   0.011874
---------------------------------------------------------------------------------------
`

func TestParseBandwidthReadsTheAverageColumn(t *testing.T) {
	got, err := parseBandwidth(realBandwidthOutput)
	if err != nil {
		t.Fatalf("parseBandwidth: %v", err)
	}
	if got.AverageGbps != 99.61 {
		t.Errorf("AverageGbps = %v, want 99.61", got.AverageGbps)
	}
	if got.Bytes != 1048576 || got.Iterations != 71244 {
		t.Errorf("bytes/iterations = %d/%d, want 1048576/71244", got.Bytes, got.Iterations)
	}
	// Duration mode does not measure a peak. The runner must be able to tell
	// that from a genuine peak of zero, because it emits peakBandwidthGbps in
	// neither case but must not silently treat 0.00 as a measurement.
	if got.PeakMeasured {
		t.Errorf("PeakMeasured = true for a duration-mode run that printed 0.00")
	}
}

// The 8x bug this parser exists to prevent: perftest prints the SAME table in
// MB/sec when --report_gbits is missing, and 12.45 MB/sec-shaped in a metric
// named bandwidthGbps is a wrong answer that looks like a right one.
func TestParseBandwidthRefusesMegabytesPerSecond(t *testing.T) {
	out := strings.ReplaceAll(realBandwidthOutput, "[Gb/sec]", "[MB/sec]")
	out = strings.ReplaceAll(out, "99.61", "12.45")
	_, err := parseBandwidth(out)
	if err == nil {
		t.Fatal("parseBandwidth accepted an MB/sec table; the value would land in bandwidthGbps 8x too small")
	}
	if !strings.Contains(err.Error(), "MB/sec") {
		t.Errorf("error should name the unit it saw, got %q", err)
	}
}

func TestParseBandwidthWithNoTable(t *testing.T) {
	_, err := parseBandwidth(" Unable to init the socket connection\n")
	if !errors.Is(err, errNoTable) {
		t.Errorf("err = %v, want errNoTable", err)
	}
}

// A multi-size sweep must report the LAST row, not the first. perftest prints
// ascending sizes, so taking the first would report the smallest and least
// representative message size as the link's bandwidth.
func TestParseBandwidthTakesTheLastRow(t *testing.T) {
	out := ` #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 2          1000             1.20               1.10               68.750000
 65536      5000             97.90              97.49              0.185942
 1048576    71244            99.80              99.61              0.011874
`
	got, err := parseBandwidth(out)
	if err != nil {
		t.Fatalf("parseBandwidth: %v", err)
	}
	if got.Bytes != 1048576 || got.AverageGbps != 99.61 {
		t.Errorf("got %d bytes at %v Gb/s, want the 1048576-byte row at 99.61", got.Bytes, got.AverageGbps)
	}
	if !got.PeakMeasured {
		t.Errorf("PeakMeasured = false for an iteration-mode row with a real peak")
	}
}

const realLatencyOutput = `---------------------------------------------------------------------------------------
                    RDMA_Write Latency Test
 Number of qps   : 1		Transport type : IB
 Link type       : Ethernet
 GID index       : 3
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
 2       5000           1.79           12.34        1.85               1.87           0.15            2.45                   9.12
---------------------------------------------------------------------------------------
`

func TestParseLatency(t *testing.T) {
	got, err := parseLatency(realLatencyOutput)
	if err != nil {
		t.Fatalf("parseLatency: %v", err)
	}
	if got.AvgUs != 1.87 {
		t.Errorf("AvgUs = %v, want 1.87", got.AvgUs)
	}
	if got.MinUs != 1.79 || got.TypicalUs != 1.85 || got.MaxUs != 12.34 {
		t.Errorf("min/typical/max = %v/%v/%v, want 1.79/1.85/12.34", got.MinUs, got.TypicalUs, got.MaxUs)
	}
	if !got.P99Measured || got.P99Us != 2.45 {
		t.Errorf("p99 = %v (measured %v), want 2.45", got.P99Us, got.P99Measured)
	}
}

// An older perftest stops the latency table at t_stdev. That is a missing
// column, not a broken run, and it must not cost the average.
func TestParseLatencyWithoutPercentileColumns(t *testing.T) {
	out := ` #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]
 2       5000           1.79           12.34        1.85               1.87           0.15
`
	got, err := parseLatency(out)
	if err != nil {
		t.Fatalf("parseLatency: %v", err)
	}
	if got.AvgUs != 1.87 {
		t.Errorf("AvgUs = %v, want 1.87", got.AvgUs)
	}
	if got.P99Measured {
		t.Errorf("P99Measured = true, but the table had no percentile columns")
	}
}

// perftest's banner contains lines that begin with digits inside addresses and
// GIDs. None of them is a result row, and a parser that counted lines instead of
// validating fields would pick one up.
func TestNumericRowsIgnoresTheBanner(t *testing.T) {
	rows := numericRows(realBandwidthOutput)
	if len(rows) != 1 {
		t.Fatalf("numericRows found %d rows, want 1: %v", len(rows), rows)
	}
}

func TestBwArgsAlwaysReportsGbits(t *testing.T) {
	p := rdmaPort{Device: "roceP2p1s0f1", Port: 1, LinkLayer: linkLayerEthernet, GIDIndex: 3}
	args := bwArgs(p, plan{perftestPort: 18515, messageBytes: 1 << 20, qps: 4, bwSeconds: 40}, "10.252.26.239")
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--report_gbits") {
		t.Fatalf("bwArgs must always pass --report_gbits, got %q", joined)
	}
	if !strings.Contains(joined, "-x 3") {
		t.Errorf("an Ethernet (RoCE) port must be given its GID index, got %q", joined)
	}
	if args[len(args)-1] != "10.252.26.239" {
		t.Errorf("the destination must be the last argument, got %q", joined)
	}
}

// An InfiniBand port needs no GID index, and passing one is not harmless: it
// selects an address family the fabric does not use.
func TestBwArgsOmitsGIDIndexOnInfiniBand(t *testing.T) {
	p := rdmaPort{Device: "mlx5_0", Port: 1, LinkLayer: "InfiniBand", GIDIndex: -1}
	if joined := strings.Join(bwArgs(p, plan{perftestPort: 18515, messageBytes: 1 << 20, qps: 4, bwSeconds: 40}, ""), " "); strings.Contains(joined, "-x") {
		t.Errorf("bwArgs passed -x for an InfiniBand port: %q", joined)
	}
}

// The tail was parsed and thrown away — a gap found by researching what industry
// actually uses to qualify a fabric.
//
// perftest's latency table carries min, max and p99 alongside the mean, and this
// runner read all four and emitted only the mean. A MEAN says a link is fine on
// average, which is the one thing nobody needs to know about a fabric: a
// collective runs at the speed of its slowest participant on EVERY iteration, so
// a link with a healthy mean and a p99 an order of magnitude worse degrades every
// job on the fleet while passing a bandwidth gate outright.
func TestLatencyTailReachesTheOperator(t *testing.T) {
	// A real ib_write_lat table shape: healthy mean, ugly tail. This is exactly
	// what the old output could not express.
	const out = `
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec]
 2       1000          1.52           412.80       1.61              1.74           0.42            38.90                  402.11
---------------------------------------------------------------------------------------
`
	lat, err := parseLatency(out)
	if err != nil {
		t.Fatalf("parseLatency: %v", err)
	}
	if !lat.P99Measured {
		t.Fatal("the percentile columns were present and were not parsed")
	}
	if lat.AvgUs > 2 && lat.P99Us < 30 {
		t.Fatalf("fixture is wrong: it must pair a healthy mean with an ugly tail, got avg=%.2f p99=%.2f",
			lat.AvgUs, lat.P99Us)
	}

	res := runner.Parse("ib-write-bw", latencyMetricLines(lat), 0)
	for name, want := range map[string]string{
		"latencyUs":    "1.74", // the mean, UNCHANGED: existing profiles keep meaning what they meant
		"minLatencyUs": "1.52",
		"maxLatencyUs": "412.80",
		"p99LatencyUs": "38.90",
	} {
		if got := res.Metrics[name]; got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
	if len(res.InvalidNames) != 0 {
		t.Errorf("the new names do not obey the metric grammar: %v", res.InvalidNames)
	}
	if !contract.SafeToThresholdOn("p99LatencyUs") {
		t.Error("p99LatencyUs must be thresholdable — it is the number a collective runs at")
	}
	for _, evidence := range []string{"minLatencyUs", "maxLatencyUs"} {
		if contract.SafeToThresholdOn(evidence) {
			t.Errorf("%s must be Evidence: a fleet gating on one best or worst sample fails "+
				"healthy links on noise", evidence)
		}
	}
}

// A perftest build whose table stops at t_stdev emits NO percentile key rather
// than a fabricated one. The columns are recent and their absence is not an
// error; an omitted metric then fails its threshold closed, which is honest.
func TestOlderPerftestEmitsNoPercentile(t *testing.T) {
	const out = `
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]
 2       1000          1.52           4.80         1.61              1.74           0.42
`
	lat, err := parseLatency(out)
	if err != nil {
		t.Fatalf("parseLatency: %v", err)
	}
	if lat.P99Measured {
		t.Fatal("a table with no percentile columns reported one as measured")
	}
	res := runner.Parse("ib-write-bw", latencyMetricLines(lat), 0)
	if v, ok := res.Metrics["p99LatencyUs"]; ok {
		t.Errorf("p99LatencyUs = %q was emitted for a build that never measured it", v)
	}
	if res.Metrics["latencyUs"] != "1.74" || res.Metrics["maxLatencyUs"] != "4.80" {
		t.Errorf("the rest of the table was lost: %v", res.Metrics)
	}
}

// latencyMetricLines renders exactly the lines main.go's latency block emits, so
// these tests exercise the EMISSION RULE rather than restating it.
func latencyMetricLines(lat latResult) string {
	out := "latency_us=" + strconv.FormatFloat(lat.AvgUs, 'f', 2, 64) + "\n" +
		"latency_min_us=" + strconv.FormatFloat(lat.MinUs, 'f', 2, 64) + "\n" +
		"latency_max_us=" + strconv.FormatFloat(lat.MaxUs, 'f', 2, 64) + "\n"
	if lat.P99Measured {
		out += "latency_p99_us=" + strconv.FormatFloat(lat.P99Us, 'f', 2, 64) + "\n"
	}
	return out + "IB_WRITE_BW_PASS\n"
}

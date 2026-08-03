// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Parsing of linux-rdma/perftest result tables.
//
// perftest prints a fixed-width table with a header naming the unit of every
// column, and the unit is the whole reason this parser reads the header instead
// of counting columns and hoping:
//
//	 #bytes  #iterations  BW peak[Gb/sec]  BW average[Gb/sec]  MsgRate[Mpps]
//	 1048576 71244        0.00             99.61               0.011874
//
// WITHOUT --report_gbits the identical table is printed in MB/sec. Those two
// numbers differ by a factor of eight, both are plausible on a modern fabric,
// and nothing downstream can tell them apart once the value has been written to
// a metric named bandwidthGbps. A node with a 100 Gb/s link would report 12.45
// and fail a >= 89 gate; a 12.45 GB/s link reported as 99.6 would pass one it
// should not. So the header is ASSERTED, and a table in the wrong unit is a
// runner Error rather than a measurement.

// errNoTable is returned when the process produced no result row at all — the
// usual shape of "the client could not reach the server yet".
var errNoTable = errors.New("no perftest result table in output")

// bwResult is one bandwidth run.
type bwResult struct {
	Bytes      int64
	Iterations int64
	// PeakGbps is only meaningful in ITERATION mode. In duration mode (-D)
	// perftest prints 0.00 here and says so on stderr, and this runner does not
	// emit it — see main.go.
	PeakGbps     float64
	AverageGbps  float64
	MsgRateMpps  float64
	PeakMeasured bool
}

// latResult is one latency run.
type latResult struct {
	Bytes      int64
	Iterations int64
	MinUs      float64
	MaxUs      float64
	TypicalUs  float64
	AvgUs      float64
	P99Us      float64
	// P99Measured is false for a perftest build whose latency table stops at
	// t_stdev. The percentile columns are recent, and their absence is not an
	// error.
	P99Measured bool
}

// parseBandwidth reads the last result row of an ib_write_bw / ib_send_bw run.
//
// The LAST row is taken deliberately: a run given several message sizes prints
// one row per size, and the runner only ever asks for one size (see main.go)
// precisely so that "the last row" and "the row" are the same thing. Taking the
// last one keeps a multi-size invocation from silently reporting the first,
// smallest, and least representative number.
func parseBandwidth(out string) (bwResult, error) {
	unit, hdrOK := bwHeaderUnit(out)
	if !hdrOK {
		return bwResult{}, errNoTable
	}
	if unit != "Gb/sec" {
		return bwResult{}, fmt.Errorf(
			"perftest reported bandwidth in %q, not Gb/sec — --report_gbits did not take effect, "+
				"and a value 8x off in a metric named bandwidthGbps is worse than no value", unit)
	}

	var got bwResult
	var found bool
	for _, f := range numericRows(out) {
		// bytes, iterations, BW peak, BW average, MsgRate
		if len(f) < 5 {
			continue
		}
		b, err1 := strconv.ParseInt(f[0], 10, 64)
		it, err2 := strconv.ParseInt(f[1], 10, 64)
		peak, err3 := strconv.ParseFloat(f[2], 64)
		avg, err4 := strconv.ParseFloat(f[3], 64)
		rate, err5 := strconv.ParseFloat(f[4], 64)
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil || err5 != nil {
			continue
		}
		got = bwResult{
			Bytes:        b,
			Iterations:   it,
			PeakGbps:     peak,
			AverageGbps:  avg,
			MsgRateMpps:  rate,
			PeakMeasured: peak > 0,
		}
		found = true
	}
	if !found {
		return bwResult{}, errNoTable
	}
	return got, nil
}

// parseLatency reads the last result row of an ib_write_lat run.
func parseLatency(out string) (latResult, error) {
	if !strings.Contains(out, "t_avg[usec]") && !strings.Contains(out, "t_typical[usec]") {
		return latResult{}, errNoTable
	}
	var got latResult
	var found bool
	for _, f := range numericRows(out) {
		// bytes, iterations, t_min, t_max, t_typical, t_avg, t_stdev[, 99%, 99.9%]
		if len(f) < 7 {
			continue
		}
		vals := make([]float64, 0, len(f))
		bad := false
		for _, s := range f {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				bad = true
				break
			}
			vals = append(vals, v)
		}
		if bad {
			continue
		}
		got = latResult{
			Bytes:      int64(vals[0]),
			Iterations: int64(vals[1]),
			MinUs:      vals[2],
			MaxUs:      vals[3],
			TypicalUs:  vals[4],
			AvgUs:      vals[5],
		}
		if len(vals) >= 8 {
			got.P99Us, got.P99Measured = vals[7], true
		}
		found = true
	}
	if !found {
		return latResult{}, errNoTable
	}
	return got, nil
}

// bwHeaderUnit finds the bandwidth table header and returns the unit it names.
func bwHeaderUnit(out string) (string, bool) {
	for _, line := range strings.Split(out, "\n") {
		i := strings.Index(line, "BW average[")
		if i < 0 {
			continue
		}
		rest := line[i+len("BW average["):]
		j := strings.Index(rest, "]")
		if j < 0 {
			return "", false
		}
		return rest[:j], true
	}
	return "", false
}

// numericRows returns the whitespace-split fields of every line that looks like
// a data row: it starts with a digit and every field parses as a number.
// perftest's decorations (banners, "---" rules, address dumps) all fail one of
// those tests, which is what keeps the parser from depending on line offsets
// that move between releases.
func numericRows(out string) [][]string {
	var rows [][]string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line[0] < '0' || line[0] > '9' {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		ok := true
		for _, f := range fields {
			if _, err := strconv.ParseFloat(f, 64); err != nil {
				ok = false
				break
			}
		}
		if ok {
			rows = append(rows, fields)
		}
	}
	return rows
}

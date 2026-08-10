// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// InfiniBand/RoCE port error counters, read as a DELTA over the soak window.
//
// The delta is the whole point. These counters are lifetime totals for the
// port: a NIC that has been up for two hundred days will show a non-zero
// symbol_error_counter that says nothing about the last four hours. Reporting
// the raw value would make every long-lived node look faulty and every freshly
// booted one look clean, which is the opposite of a measurement.
//
// A counter that moved DURING the run is evidence about this run. That is what
// makes `linkErrorEvents Equal 0` a gate worth writing, and it is why the
// baseline is taken before the first iteration rather than at process start.

// countersOfInterest are the port counters worth differencing, and why.
//
// Deliberately a short list. Every one of these moves on a link that is
// physically marginal — a bad optic, a loose seat, a transceiver failing at
// temperature — which is what a soak is looking for. Counters that move for
// ordinary congestion are excluded: they would fire on a healthy fabric under
// load and teach an operator to ignore the metric.
var countersOfInterest = []string{
	"symbol_error_counter", // physical-layer symbol errors
	"link_error_recovery",  // the link renegotiated after an error
	"link_downed",          // the link dropped entirely
	"port_rcv_errors",      // malformed or damaged packets received
	"local_link_integrity_errors",
	"excessive_buffer_overrun_errors",
}

// portCounters is one port's readings, keyed by counter name.
type portCounters map[string]int64

// readPortCounters reads the counters for one device/port under a sysfs root.
//
// A counter that cannot be read is OMITTED rather than recorded as zero. That
// distinction carries all the way to the verdict: a zero delta says "this
// counter did not move", while an absence says "this counter was never read",
// and only the first is evidence a link is healthy. Reporting an unread counter
// as zero is how a fabric gets certified by a file nobody opened.
func readPortCounters(sysfsRoot, device, port string) portCounters {
	dir := filepath.Join(sysfsRoot, "class", "infiniband", device, "ports", port, "counters")
	out := portCounters{}
	for _, name := range countersOfInterest {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		v, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
		if err != nil {
			continue
		}
		out[name] = v
	}
	return out
}

// delta subtracts a baseline from a final reading.
//
// Only counters present in BOTH are differenced. One that appeared or vanished
// mid-run was not measured across the window, and a delta computed against a
// missing endpoint would be the raw lifetime value wearing a delta's name.
//
// A NEGATIVE delta means the counter was reset — a port bounce, or a driver
// reload — which is not "minus four errors". It is reported as unmeasurable
// rather than clamped to zero, because clamping would silently turn a link that
// bounced mid-soak into a clean run.
func delta(before, after portCounters) (portCounters, bool) {
	out := portCounters{}
	reset := false
	for name, start := range before {
		end, ok := after[name]
		if !ok {
			continue
		}
		if end < start {
			reset = true
			continue
		}
		out[name] = end - start
	}
	return out, !reset
}

// total sums a delta into the single number a gate is written against.
//
// One counter would be more precise; one NUMBER is what a threshold can hold,
// and the individual deltas are emitted alongside as evidence so the precise
// answer is still in the result. A fabric where any of these moved is a fabric
// worth looking at, which makes the sum the right shape for the gate.
func (c portCounters) total() int64 {
	var n int64
	for _, v := range c {
		n += v
	}
	return n
}

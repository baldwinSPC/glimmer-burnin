// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"sort"
)

// The accumulator, and what a fabric soak is actually for.
//
// ib-write-bw answers "does this link work right now". It runs for about a
// minute and reports an average, and it passes on a link with a marginal optic,
// a cable that is seated badly, or a transceiver that fails once it is warm.
// Those are the failures that hurt most in production, because they present as
// a slow training job or an occasional NCCL timeout rather than as a link that
// is down — and they are found by running the fabric for hours.
//
// So the figure that matters here is NOT the peak, and not even the mean. It is
// the WORST window and the SPREAD: a soak that averaged fine and spent ninety
// seconds at a third of line rate has found something, and an average hides it
// completely.

// window is one iteration's measurement.
type window struct {
	gbps float64
	// ok is false for an iteration that failed or timed out. A failed window
	// contributes to the failure count and to NOTHING else — averaging a zero
	// into the bandwidth would report a fabric slower than it is, and a
	// fabricated zero is exactly what this project forbids elsewhere.
	ok bool
}

// soak accumulates windows over the run.
type soak struct {
	windows []window

	// connectRetries and connectFailures count the CONTROL-CONNECTION race,
	// which is not a property of the fabric and is deliberately kept out of
	// windows[] so it can never reach failures() and indict a link.
	//
	// perftest's server exits when its client disconnects — that is the normal
	// end of every window — and runServer then re-execs it. Between the old
	// process exiting and the new one binding the control port, the port is
	// unbound, and a client starting its next window in that gap is refused
	// (#496: 19 refusals in 359 windows, with the link moving data at 97.50
	// Gb/s either side of every one of them and every port counter delta zero).
	// A refusal happens BEFORE any traffic, so it is evidence about this
	// runner's own restart timing and about nothing else.
	connectRetries  int
	connectFailures int
}

// recordConnectRetry counts one refused attempt that a later attempt recovered.
func (s *soak) recordConnectRetries(n int) { s.connectRetries += n }

// recordConnectFailure counts a window whose retries were all refused.
//
// It deliberately does NOT append to windows[]: a window that never opened its
// control connection did not soak the link, so counting it as a failed window
// would report "the link carried traffic and stopped carrying it" about a
// window in which no traffic was ever attempted. Where EVERY window is refused,
// windows[] stays empty and runClient's existing "no window completed" path
// reports Error — unjudged — which is the correct verdict for a link nobody
// measured.
func (s *soak) recordConnectFailure() { s.connectFailures++ }

func (s *soak) record(gbps float64, ok bool) {
	s.windows = append(s.windows, window{gbps: gbps, ok: ok})
}

// iterations is every window attempted, good or not.
func (s *soak) iterations() int { return len(s.windows) }

// failures is the count of windows that did not complete.
//
// The metric this feeds is a COUNTER, so a healthy soak reports exactly zero
// and `Equal 0` is a safe gate from day one — unlike the bandwidth figures,
// which need a measured baseline first.
func (s *soak) failures() int {
	n := 0
	for _, w := range s.windows {
		if !w.ok {
			n++
		}
	}
	return n
}

// measured returns the bandwidths of the windows that completed.
func (s *soak) measured() []float64 {
	out := make([]float64, 0, len(s.windows))
	for _, w := range s.windows {
		if w.ok {
			out = append(out, w.gbps)
		}
	}
	return out
}

// stats is what the run reports about the completed windows.
type stats struct {
	count int
	min   float64
	max   float64
	mean  float64
	// stddev is the population standard deviation, in Gb/s.
	//
	// Reported alongside the minimum rather than instead of it: a link that
	// flaps produces a low minimum AND a wide spread, while a link that is
	// simply slow produces a low minimum and a narrow one. Distinguishing those
	// is the difference between replacing an optic and re-checking a cable
	// budget.
	stddev float64
	// p1 is the 1st percentile, nearest-rank — a value a window actually
	// reached. On a long soak it is a better floor than the single minimum,
	// which one scheduling hiccup can drag down.
	p1 float64
}

func (s *soak) stats() (stats, bool) {
	v := s.measured()
	if len(v) == 0 {
		return stats{}, false
	}

	out := stats{count: len(v), min: v[0], max: v[0]}
	var total float64
	for _, x := range v {
		total += x
		if x < out.min {
			out.min = x
		}
		if x > out.max {
			out.max = x
		}
	}
	out.mean = total / float64(len(v))

	var sq float64
	for _, x := range v {
		d := x - out.mean
		sq += d * d
	}
	out.stddev = math.Sqrt(sq / float64(len(v)))

	sorted := append([]float64(nil), v...)
	sort.Float64s(sorted)
	rank := int(math.Ceil(0.01 * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	out.p1 = sorted[rank-1]

	return out, true
}

// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"sort"
	"time"
)

// latencies accumulates per-operation timings.
//
// Every sample is kept rather than a running summary, because a percentile is
// the point: a drive whose mean latency is fine and whose p99 is 40x the mean
// is a drive that stalls a training step, and a mean alone hides exactly that.
// A 60-second run at 4 MiB blocks is a few thousand samples — small enough that
// the honest thing is also the cheap one.
type latencies []time.Duration

func (l latencies) mean() time.Duration {
	if len(l) == 0 {
		return 0
	}
	var total time.Duration
	for _, d := range l {
		total += d
	}
	return total / time.Duration(len(l))
}

// percentile returns the p'th percentile, 0 < p < 100.
//
// Nearest-rank on a sorted copy: no interpolation, so the value reported is one
// a real operation actually took. An interpolated p99 is a number that never
// happened, which is a poor thing to send an engineer to a datacentre over.
//
// It sorts a COPY — the caller may still want its samples, and silently
// reordering an argument is the kind of thing that is fine until the day
// somebody adds a second percentile.
func (l latencies) percentile(p float64) time.Duration {
	if len(l) == 0 {
		return 0
	}
	sorted := make(latencies, len(l))
	copy(sorted, l)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	rank := int(math.Ceil(p / 100 * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

func (l latencies) micros(d time.Duration) float64 {
	return float64(d) / float64(time.Microsecond)
}

// throughputMBs is bytes over elapsed, in MB/s.
//
// MB, decimal, because that is what the registered metric name means
// (readBandwidthMBs) and what every drive datasheet quotes. Mixing MiB into a
// metric whose name says MB is how two dispatchers end up 4.9% apart and
// nobody can say which is right.
func throughputMBs(bytes uint64, elapsed time.Duration) float64 {
	if elapsed <= 0 {
		return 0
	}
	return (float64(bytes) / 1e6) / elapsed.Seconds()
}

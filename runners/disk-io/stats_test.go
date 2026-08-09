// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"
	"time"
)

func TestThePercentileIsAValueThatActuallyHappened(t *testing.T) {
	// Nearest-rank, no interpolation. An interpolated p99 is a number no
	// operation ever took, which is a poor thing to send an engineer to a
	// datacentre over.
	var l latencies
	for i := 1; i <= 100; i++ {
		l = append(l, time.Duration(i)*time.Millisecond)
	}

	rows := []struct {
		p    float64
		want time.Duration
	}{
		{50, 50 * time.Millisecond},
		{99, 99 * time.Millisecond},
		{100, 100 * time.Millisecond},
		{1, 1 * time.Millisecond},
	}
	for _, r := range rows {
		if got := l.percentile(r.p); got != r.want {
			t.Errorf("p%v = %v, want %v", r.p, got, r.want)
		}
	}
}

func TestThePercentileFindsTheTailAMeanHides(t *testing.T) {
	// The reason percentiles are reported at all: this drive's mean looks fine
	// and its p99 is the thing that stalls a training step.
	//
	// TWO outliers in a hundred, not one. p99 is the value at the 99th of 100
	// ranks, so a single outlier in a hundred samples sits at rank 100 and p99
	// correctly reports a fast operation — the tail has to be at least 1% of
	// operations before p99 is the metric that shows it. Getting this wrong in
	// the other direction is worse than the bug it would hide: it would mean
	// reporting a percentile that overstates the tail on every healthy drive.
	var l latencies
	for i := 0; i < 98; i++ {
		l = append(l, 100*time.Microsecond)
	}
	l = append(l, 250*time.Millisecond, 250*time.Millisecond)

	// Stated as a RATIO rather than an absolute: the claim is that the mean
	// understates the tail, and by how much is the whole reason both are
	// reported. An absolute bound here would just be a number I picked.
	if mean, p99 := l.mean(), l.percentile(99); p99 < 20*mean {
		t.Errorf("mean %v, p99 %v — this case is meant to show a tail the mean flattens", mean, p99)
	}
	if p99 := l.percentile(99); p99 < 200*time.Millisecond {
		t.Errorf("p99 = %v, want the 250ms outlier — a mean that hides it is why this metric exists", p99)
	}
	// And the boundary the comment above describes, asserted so nobody
	// "fixes" percentile() into interpolating: one outlier in a hundred is a
	// p100 event, not a p99 one.
	var one latencies
	for i := 0; i < 99; i++ {
		one = append(one, 100*time.Microsecond)
	}
	one = append(one, 250*time.Millisecond)
	if got := one.percentile(99); got != 100*time.Microsecond {
		t.Errorf("p99 of 99 fast + 1 slow = %v, want the fast value: a lone outlier is rank 100", got)
	}
}

func TestPercentileDoesNotReorderItsInput(t *testing.T) {
	// It sorts a copy. Silently reordering an argument is fine until the day
	// somebody asks for a second percentile.
	l := latencies{5 * time.Millisecond, 1 * time.Millisecond, 3 * time.Millisecond}
	_ = l.percentile(50)
	if l[0] != 5*time.Millisecond {
		t.Errorf("the caller's samples were reordered: %v", l)
	}
}

func TestEmptySamplesAreZeroAndNotAPanic(t *testing.T) {
	var l latencies
	if l.mean() != 0 || l.percentile(99) != 0 {
		t.Error("empty samples should be zero")
	}
}

func TestThroughputIsDecimalMBBecauseTheMetricNameSaysMB(t *testing.T) {
	// Mixing MiB into a metric called readBandwidthMBs is how two dispatchers
	// end up 4.9% apart with nobody able to say which is right.
	got := throughputMBs(1_000_000_000, time.Second)
	if got != 1000 {
		t.Errorf("1 GB in 1s = %v MB/s, want 1000", got)
	}
	if throughputMBs(1000, 0) != 0 {
		t.Error("a zero-length window should be zero, not an infinity")
	}
}

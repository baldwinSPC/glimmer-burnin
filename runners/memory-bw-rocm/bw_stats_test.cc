// bw_stats_test.cc — unit tests for bw_stats.h.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Plain C++17, no framework, no HIP, no hardware — compiled and run by
// runners/cxxtests_test.go.
//
// The cases that matter here are the ones where a bandwidth number would be
// FABRICATED rather than measured: a zero-length interval, an empty sample set,
// a single sample pretending to have a spread. Each of those produces a figure
// that passes any floor a profile could write, which is the shape of a gate
// that cannot fail.

#include "bw_stats.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace burnin;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		failures++;
	}
}

void testBandwidthGBs() {
	double gbs = 0;
	// 1 GB in 1 second is 1 GB/s, decimal.
	check(bandwidthGBs(1000000000ULL, 1.0, &gbs) && gbs == 1.0, "1 GB in 1 s is 1 GB/s");
	check(bandwidthGBs(2000000000ULL, 0.5, &gbs) && gbs == 4.0, "2 GB in 0.5 s is 4 GB/s");

	// THE TRAP: a duration under the clock's resolution must be refused, not
	// divided by. +inf satisfies every floor a profile could write.
	gbs = -1;
	check(!bandwidthGBs(1000000000ULL, 0.0, &gbs), "a zero interval is refused");
	check(!bandwidthGBs(1000000000ULL, 1e-9, &gbs), "an interval under the floor is refused");
	check(gbs == -1, "a refused conversion writes nothing to the output");

	// Zero bytes is equally not a measurement.
	check(!bandwidthGBs(0, 1.0, &gbs), "zero bytes is refused");

	// And the boundary is inclusive of the floor.
	check(bandwidthGBs(1000ULL, kMinTimedSeconds, &gbs), "exactly the floor is accepted");
	check(std::isfinite(gbs), "the accepted boundary produces a finite figure");
}

void testSamples() {
	Samples s;
	check(!s.measured(), "an empty sample set reports itself unmeasured");
	check(s.mean() == 0, "an empty set's mean is not a division by zero");
	check(s.spreadPct() == 0, "an empty set has no spread");

	s.add(100.0);
	check(s.measured(), "one sample counts as measured");
	check(s.min == 100.0 && s.max == 100.0, "one sample is both min and max");
	// A single measurement has no spread, and reporting one would invent a
	// stability claim from a sample size of one.
	check(s.spreadPct() == 0, "a single sample has no spread");

	s.add(50.0);
	s.add(75.0);
	check(s.n == 3, "three samples counted");
	check(s.min == 50.0, "min is the worst pass — the acceptance figure");
	check(s.max == 100.0, "max is the best pass — evidence");
	check(s.mean() == 75.0, "mean is the mean");
	check(s.spreadPct() == 50.0, "spread is how far the worst fell below the best");
}

void testTriadBytes() {
	// 1000 floats, 4 bytes each, 3 arrays, 10 iterations.
	check(triadBytes(1000, 4, 10) == 120000ULL, "triad counts three arrays per element");
	// Zero or negative iterations is not a measurement.
	check(triadBytes(1000, 4, 0) == 0, "zero iterations moves no bytes");
	check(triadBytes(1000, 4, -1) == 0, "negative iterations moves no bytes");
	// And that zero feeds bandwidthGBs' zero-byte refusal, so the two compose
	// into "no iterations completed means no figure".
	double gbs = 0;
	check(!bandwidthGBs(triadBytes(1000, 4, 0), 1.0, &gbs),
	      "no completed iterations yields no bandwidth figure");
}

void testTriadExpected() {
	check(triadExpected(2.0f, 3.0f, 4.0f) == 14.0f, "a = b + scalar * c");
	check(triadExpected(0.0f, 0.0f, 3.0f) == 0.0f, "zeros stay zero");
}

void testFirstMismatch() {
	const float want[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	const float same[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	check(firstMismatch(same, want, 4, 1e-6) == -1, "identical buffers report no mismatch");

	const float off[4] = {1.0f, 2.0f, 3.5f, 4.0f};
	check(firstMismatch(off, want, 4, 1e-6) == 2, "the first differing index is reported");
	check(firstMismatch(off, want, 4, 1.0) == -1, "a difference inside tolerance is not a mismatch");

	// A NaN must be caught. It is, because the comparison is written as "not
	// within tolerance" rather than "outside tolerance" — every comparison
	// against NaN is false, so the naive form would report a match.
	const float nan_ = std::nan("");
	const float withNan[4] = {1.0f, nan_, 3.0f, 4.0f};
	check(firstMismatch(withNan, want, 4, 1e-6) == 1, "a NaN is a mismatch, not a match");

	// An all-zero buffer against a non-zero reference is the signature of a
	// copy that never happened, and it must fail rather than pass.
	const float zeros[4] = {0, 0, 0, 0};
	check(firstMismatch(zeros, want, 4, 1e-6) == 0, "an unwritten buffer mismatches at once");
}

}  // namespace

int main() {
	testBandwidthGBs();
	testSamples();
	testTriadBytes();
	testTriadExpected();
	testFirstMismatch();
	if (failures > 0) {
		std::fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("bw_stats_test: all checks passed\n");
	return 0;
}

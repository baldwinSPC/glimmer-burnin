// bw_stats.h — bandwidth arithmetic and sample aggregation for memory-bw-rocm.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only and free of HIP, so the arithmetic that decides a node's verdict is
// unit-tested on any machine (see bw_stats_test.cc, which runners/cxxtests_test.go
// compiles and runs under `make test`). The HIP calls live in the .cc.
//
// The one rule this file exists to enforce: A MEASUREMENT THAT DID NOT HAPPEN
// MUST NOT PRODUCE A NUMBER. Every path that could divide by a zero duration,
// or aggregate an empty sample set, answers "unmeasurable" instead — because
// the alternative is an infinity or a zero that sails through a floor gate and
// certifies hardware nobody measured.

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace burnin {

// kMinTimedSeconds is the shortest interval this runner will convert into a
// bandwidth.
//
// A copy that appears to take zero time is not infinitely fast; it is a copy
// whose duration fell under the clock's resolution, or one the runtime elided.
// Dividing by it yields +inf, which satisfies EVERY floor a profile could
// write — the exact shape of a gate that cannot fail. The floor is
// deliberately far above any real clock's resolution (a microsecond) so that a
// suspiciously fast result is refused rather than rounded into plausibility.
constexpr double kMinTimedSeconds = 1e-6;

// bandwidthGBs converts bytes moved over an interval into GB/s, decimal GB to
// match every vendor bandwidth figure a reader will compare it against.
//
// Returns false when the interval is too short to divide by; the caller must
// then declare the figure unmeasurable rather than emit anything.
inline bool bandwidthGBs(std::uint64_t bytes, double seconds, double *out) {
	if (seconds < kMinTimedSeconds || bytes == 0) return false;
	*out = static_cast<double>(bytes) / seconds / 1e9;
	return true;
}

// Samples accumulates per-iteration bandwidth figures.
//
// MIN IS THE ACCEPTANCE FIGURE and max is evidence, matching the NVIDIA
// memory-bw runner's convention for the same reason: a path is as good as its
// worst pass, and a mean hides the one slow iteration that a thermal or
// power-management fault produces. Their distance is what makes an intermittent
// fault visible as a spread rather than as a slightly lower average.
struct Samples {
	long n = 0;
	double min = std::numeric_limits<double>::infinity();
	double max = 0;
	double sum = 0;

	void add(double gbs) {
		n++;
		if (gbs < min) min = gbs;
		if (gbs > max) max = gbs;
		sum += gbs;
	}

	bool measured() const { return n > 0; }
	double mean() const { return n > 0 ? sum / static_cast<double>(n) : 0; }

	// spreadPct is how far the worst pass fell below the best, as a percentage
	// of the best. Zero when a single sample was taken — one measurement has no
	// spread, and reporting one would invent a stability claim.
	double spreadPct() const {
		if (n < 2 || max <= 0) return 0;
		return 100.0 * (max - min) / max;
	}
};

// STREAM triad moves three arrays per element: two read, one written.
//
// The write counts ONCE even though a write-allocate cache policy also reads
// the line, which is the convention STREAM itself uses and therefore the one a
// reader comparing against any published STREAM figure expects. Counting the
// read-for-ownership would inflate this runner's number by a third against
// every number anyone would compare it to.
constexpr int kTriadArrays = 3;

inline std::uint64_t triadBytes(std::size_t elements, std::size_t elementSize,
                                long iterations) {
	if (iterations <= 0) return 0;
	return static_cast<std::uint64_t>(elements) * elementSize * kTriadArrays *
	       static_cast<std::uint64_t>(iterations);
}

// triadExpected is the reference value for a[i] = b[i] + scalar * c[i], used to
// verify the kernel actually computed rather than merely ran.
inline float triadExpected(float b, float c, float scalar) { return b + scalar * c; }

// firstMismatch returns the index of the first element that differs by more
// than tolerance, or -1 when every element matches.
//
// Data verification is what separates this runner's one FAIL condition from its
// errors: a copy that moved bytes at full speed and corrupted them is a hardware
// verdict, while a copy that could not run at all is an infrastructure fault.
inline long firstMismatch(const float *got, const float *want, std::size_t n, double tolerance) {
	for (std::size_t i = 0; i < n; i++) {
		double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
		if (d < 0) d = -d;
		// A NaN fails every comparison, so it is caught by the same branch
		// rather than needing its own — but only because the test is written
		// as "not within tolerance" instead of "outside tolerance".
		if (!(d <= tolerance)) return static_cast<long>(i);
	}
	return -1;
}

}  // namespace burnin

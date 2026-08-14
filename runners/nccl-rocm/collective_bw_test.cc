// collective_bw_test.cc — unit tests for collective_bw.h.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Plain C++17, no framework, no HIP, no RCCL, no hardware.
//
// The scaling factor is the thing worth testing hardest, because it is the one
// number that must agree across TWO runners and TWO vendors: a Spark pair and a
// Halo pair report busBandwidthGBs under the same TestKind onto the same
// dashboard axis, and a factor that differed would make that axis meaningless
// while every individual run looked fine.

#include "collective_bw.h"

#include <cstdio>

using namespace burnin;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		failures++;
	}
}

void testBusBandwidthFactor() {
	// The value that matters most here: a Pair. Two ranks means factor 1, so
	// busbw == algbw, and this is the rank count every Halo pair will run.
	check(busBandwidthFactor(2) == 1.0, "n=2 gives factor exactly 1");
	check(busBandwidth(6.4, 2) == 6.4, "at n=2 bus bandwidth equals algorithm bandwidth");

	// 2*(n-1)/n for the group cases.
	check(busBandwidthFactor(4) == 1.5, "n=4 gives 1.5");
	check(busBandwidthFactor(8) == 1.75, "n=8 gives 1.75");

	// This is rccl-tests' formula verbatim; spot-check it against the source
	// arithmetic rather than trusting the shape.
	for (int n = 2; n <= 16; n++) {
		const double want = (2.0 * (n - 1)) / static_cast<double>(n);
		check(busBandwidthFactor(n) == want, "matches 2*(n-1)/n for every rank count");
	}

	// THE ONE-RANK TRAP. The factor is genuinely zero, so a one-rank run would
	// report busbw=0.00 on perfectly healthy hardware and fail any floor gate
	// forever. The runner refuses nranks < 2; this pins the arithmetic that
	// makes that refusal necessary.
	check(busBandwidthFactor(1) == 0.0, "n=1 gives zero — a collective of one crosses nothing");
	check(busBandwidth(999.0, 1) == 0.0, "no algorithm bandwidth rescues a one-rank bus figure");
	check(busBandwidthFactor(0) == 0.0, "zero ranks is zero, not a division by zero");
	check(busBandwidthFactor(-1) == 0.0, "a negative rank count does not produce a figure");
}

void testAlgBandwidth() {
	double gbs = 0;
	check(algBandwidthGBs(1000000000ULL, 1.0, &gbs) && gbs == 1.0, "1 GB in 1 s is 1 GB/s");
	check(algBandwidthGBs(134217728ULL, 0.021, &gbs), "a realistic 128 MiB timing converts");
	// ~6.4 GB/s, the order of magnitude AMD's own two-node gfx1151 runs report.
	check(gbs > 6.0 && gbs < 6.6, "128 MiB in 21 ms is about 6.4 GB/s");

	gbs = -1;
	check(!algBandwidthGBs(1000000ULL, 0.0, &gbs), "a zero interval is refused");
	check(!algBandwidthGBs(0, 1.0, &gbs), "zero bytes is refused");
	check(gbs == -1, "a refused conversion writes nothing");
}

void testSweep() {
	Sweep s;
	check(!s.any, "an empty sweep reports nothing observed");

	// A sweep is latency-bound at small sizes and bandwidth-bound at large
	// ones, so the PEAK is the link's figure — and it is selected here because
	// the operator's parser is last-occurrence-wins, which would otherwise
	// report whichever size happened to come last.
	s.observe(1048576, 5.07, 5.07);
	s.observe(16777216, 6.35, 6.35);
	s.observe(134217728, 6.38, 6.38);
	check(s.any, "observations register");
	check(s.peakBusBw == 6.38, "the peak bus bandwidth is reported");
	check(s.peakSizeBytes == 134217728, "the size that achieved the peak is recorded");

	// A later, SLOWER size must not displace the peak — the exact failure a
	// last-occurrence-wins emission would produce.
	s.observe(268435456, 4.10, 4.10);
	check(s.peakBusBw == 6.38, "a slower later size does not displace the peak");
	check(s.peakSizeBytes == 134217728, "nor does it displace the recorded size");

	// The alg figure reported alongside must be the one from the SAME row as
	// the peak, not the peak of algbw independently — they are one measurement.
	Sweep t;
	t.observe(1024, 9.9, 0.1);
	t.observe(1048576, 2.0, 8.0);
	check(t.peakBusBw == 8.0 && t.peakAlgBw == 2.0,
	      "the alg figure comes from the peak's own row, not from a separate maximum");
}

}  // namespace

int main() {
	testBusBandwidthFactor();
	testAlgBandwidth();
	testSweep();
	if (failures > 0) {
		std::fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("collective_bw_test: all checks passed\n");
	return 0;
}

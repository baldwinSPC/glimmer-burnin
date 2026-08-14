// collective_bw.h — the all-reduce bandwidth arithmetic and sweep selection
// for nccl-rocm.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only and free of HIP and RCCL, so the arithmetic a verdict hangs off is
// unit-tested on any machine (see collective_bw_test.cc, which
// runners/cxxtests_test.go compiles and runs under `make test`).
//
// The formulas here MUST match the ones runners/nccl/nccl_pair.cu and
// runners/nccl/results.go use, and for a sharper reason than tidiness: both
// vendors' images report `busBandwidthGBs` under the SAME TestKind, so a fleet
// dashboard puts a Spark pair and a Halo pair on one axis. Two different
// scaling factors would make that axis a lie. The NVIDIA runner has a test
// asserting its harness and its parser agree; this is the AMD half of the same
// invariant.

#pragma once

#include <cstddef>

namespace burnin {

// busBandwidthFactor is the all-reduce scaling factor, 2*(n-1)/n.
//
// It converts ALGORITHM bandwidth (bytes the caller moved / time) into BUS
// bandwidth (bytes actually crossing the interconnect), which is the figure
// that is comparable across rank counts and the one nccl-tests and rccl-tests
// both report as `busbw`. rccl-tests computes it identically — verified
// against projects/rccl-tests/src/all_reduce.cu, which reads
//
//     double factor = ((double)(2*(nranks - 1)))/((double)nranks);
//
// AT n=2 IT IS EXACTLY 1, so busBandwidthGBs equals algBandwidthGBs for a Pair.
// That is not a bug and not a coincidence to paper over: a two-rank all-reduce
// moves each byte across the link exactly once in each direction.
//
// AT n=1 IT IS EXACTLY 0. A single-rank all-reduce crosses no interconnect at
// all, so its bus bandwidth is genuinely zero however healthy the hardware —
// which makes a busbw threshold on a one-rank run a gate that fails on every
// node forever. The runner refuses nranks < 2 rather than reporting that zero.
inline double busBandwidthFactor(int nranks) {
	if (nranks < 2) return 0.0;
	return (2.0 * (nranks - 1)) / static_cast<double>(nranks);
}

inline double busBandwidth(double algbw, int nranks) {
	return algbw * busBandwidthFactor(nranks);
}

// algBandwidthGBs is bytes moved over elapsed time, in decimal GB/s to match
// every published collective figure a reader would compare against.
//
// Returns false when the interval is too short to divide by — the same rule
// memory-bw-rocm applies, and for the same reason: +inf satisfies every floor a
// profile could write.
constexpr double kMinTimedSeconds = 1e-9;

inline bool algBandwidthGBs(std::size_t bytes, double seconds, double *out) {
	if (seconds < kMinTimedSeconds || bytes == 0) return false;
	*out = static_cast<double>(bytes) / seconds / 1e9;
	return true;
}

// Sweep tracks the best result across the message-size sweep.
//
// THE PEAK IS THE REPORTED FIGURE, matching results.go: a sweep exists because
// small messages are latency-bound and large ones are bandwidth-bound, and the
// question this test answers is "what can this link actually do". The small
// sizes are evidence about latency, not candidates for the bandwidth number.
//
// A last-occurrence-wins parser is why the peak is selected HERE rather than by
// emitting one line per size: the operator's parser keeps the last value for a
// repeated key, so printing `busbw_gbs=` once per size would silently report
// whichever size happened to be last in the sweep.
struct Sweep {
	bool any = false;
	double peakAlgBw = 0;
	double peakBusBw = 0;
	std::size_t peakSizeBytes = 0;
	long long wrong = 0;

	void observe(std::size_t sizeBytes, double algbw, double busbw) {
		any = true;
		if (busbw > peakBusBw) {
			peakBusBw = busbw;
			peakAlgBw = algbw;
			peakSizeBytes = sizeBytes;
		}
	}
};

}  // namespace burnin

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
#include <string>

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

// ── Node scope: intra-node, ncclCommInitAll ─────────────────────────────────
//
// Everything below ports runners/nccl/collective/collective.h's Node-scope
// design to RCCL. It is a SEPARATE file from that one — each runner is its
// own Docker build context and COPY cannot reach outside one, so shared logic
// is physically duplicated across this project — but the ARITHMETIC must
// agree exactly, for the reason busBandwidthFactor above already states: both
// vendors' images report busBandwidthGBs under the same TestKind onto the
// same fleet dashboard. collective_bw_test.cc pins the same magic numbers
// (2*(n-1)/n, (n-1)/n) this header's NVIDIA counterpart is tested against.
//
// It exists because DGX BasePOD validation runs the 8-GPU intra-node
// all-reduce BEFORE any multi-node test, and AMD's own CVF gates intra-node
// all-reduce bandwidth the same way — RCCL's half of that requirement is this
// file's reason to exist as much as NCCL's is.

// Collective is which operation the intra-node path runs. AllReduce is the
// default and the one with a load-bearing external requirement (CVF); the
// other three exist because `collective` is a variant axis a profile can set.
enum class Collective { AllReduce, AllGather, ReduceScatter, AllToAll };

// ParseCollective reads BURNIN_VARIANT_COLLECTIVE's value. Empty is
// AllReduce. Any other unrecognised value is refused (false) rather than
// silently measured as AllReduce — the SAME vocabulary
// runners/nccl/collective/collective.h accepts, so a profile that names a
// mixed NVIDIA/AMD fleet measures the same collective on both vendors.
inline bool ParseCollective(const std::string &raw, Collective *out) {
	if (raw.empty() || raw == "allreduce") {
		*out = Collective::AllReduce;
		return true;
	}
	if (raw == "allgather") {
		*out = Collective::AllGather;
		return true;
	}
	if (raw == "reducescatter") {
		*out = Collective::ReduceScatter;
		return true;
	}
	if (raw == "alltoall") {
		*out = Collective::AllToAll;
		return true;
	}
	return false;
}

inline const char *CollectiveName(Collective c) {
	switch (c) {
	case Collective::AllReduce:
		return "allreduce";
	case Collective::AllGather:
		return "allgather";
	case Collective::ReduceScatter:
		return "reducescatter";
	case Collective::AllToAll:
		return "alltoall";
	}
	return "allreduce";
}

// BusBandwidthScale generalises busBandwidthFactor (AllReduce's 2*(n-1)/n) to
// AllGather, ReduceScatter and AllToAll, each of which moves every byte
// across the interconnect exactly ONCE per rank's share and so scales by
// (n-1)/n — rccl-tests computes all three identically for that reason, and it
// is the same generalisation runners/nccl/collective/collective.h makes for
// NCCL. AllReduce here calls busBandwidthFactor directly rather than
// reimplementing it, so the two can never drift against each other.
inline double BusBandwidthScale(Collective c, int nranks) {
	if (c == Collective::AllReduce) return busBandwidthFactor(nranks);
	if (nranks < 2) return 0.0;
	return (static_cast<double>(nranks) - 1.0) / static_cast<double>(nranks);
}

inline double BusBandwidthFor(Collective c, double algbw, int nranks) {
	return algbw * BusBandwidthScale(c, nranks);
}

// BufferPlan and PlanBuffers: identical semantics to
// runners/nccl/collective/collective.h's — see that file for the full
// rationale (NCCL's and RCCL's ncclAllGather/ncclReduceScatter/ncclSend/
// ncclRecv share the same buffer-size contract, since RCCL declares the same
// C API) AND for the caller obligation that matters most: recompute this plan
// per swept size and use that SAME plan for seeding, launching and reading
// back one size's data. AllToAll's ChunkCount scales with totalCount, so
// mixing a plan seeded at the largest size with one launched at a smaller
// size reads every peer's chunk from the wrong offset — a real bug caught in
// this project's own first draft of the intra-node path, in both languages.
struct BufferPlan {
	std::size_t SendCount;
	std::size_t RecvCount;
	std::size_t ChunkCount;  // AllToAll only; zero otherwise
	std::size_t BufElements;
};

inline BufferPlan PlanBuffers(Collective c, std::size_t totalCount, int nranks) {
	if (totalCount == 0) totalCount = 1;
	const std::size_t n = static_cast<std::size_t>(nranks > 0 ? nranks : 1);
	switch (c) {
	case Collective::AllGather: {
		const std::size_t per = totalCount / n == 0 ? 1 : totalCount / n;
		return BufferPlan{per, per * n, 0, per * n};
	}
	case Collective::ReduceScatter: {
		const std::size_t per = totalCount / n == 0 ? 1 : totalCount / n;
		return BufferPlan{per * n, per, 0, per * n};
	}
	case Collective::AllToAll: {
		const std::size_t chunk = totalCount / n == 0 ? 1 : totalCount / n;
		const std::size_t total = chunk * n;
		return BufferPlan{total, total, chunk, total};
	}
	case Collective::AllReduce:
	default:
		return BufferPlan{totalCount, totalCount, 0, totalCount};
	}
}

// Seed/expect pairs: identical semantics to collective.h's. AllReduce and
// ReduceScatter seed 1.0f everywhere and expect `nranks` back (a sum);
// AllGather and AllToAll seed a per-rank identity and expect exactly what was
// seeded, so a chunk delivered to the wrong peer or from the wrong sender is
// caught, not merely a chunk with a wrong VALUE.
inline float SeedForAllGather(int rank) { return static_cast<float>(rank + 1); }
inline float ExpectedAllGatherSegment(int segment) { return static_cast<float>(segment + 1); }
inline float SeedForAllToAll(int senderRank, int peerRank) {
	return static_cast<float>(senderRank) * 1000.0f + static_cast<float>(peerRank) + 1.0f;
}
inline float ExpectedAllToAllChunk(int senderRank, int receiverRank) {
	return SeedForAllToAll(senderRank, receiverRank);
}

// NodeScopeDecision is the ENTIRE Node-scope capability check for RCCL, pure
// so it is unit-tested with no GPU (collective_bw_test.cc) — this project has
// no multi-GPU node. It answers exactly one question, from exactly one input:
// can this node run an intra-node collective at all, given how many devices
// HIP can see. That devCount is the ONLY input is deliberate: what decides
// whether a Node-scope pod may run a collective is a POSITIVE fact about the
// hardware, never the mere ABSENCE of BURNIN_ROLE/BURNIN_NRANKS — reading an
// absence as permission is the exact failure this project's Group-scope guard
// exists to catch (#118), restated here for its Node-scope mirror.
struct NodeScopeDecision {
	bool Skip;
	std::string Reason;
};

inline NodeScopeDecision DecideNodeScope(int devCount) {
	if (devCount < 2) {
		return NodeScopeDecision{
		    true, "no BURNIN_ROLE and no BURNIN_NRANKS: this is a Node-scope run, and this node has " +
		              std::to_string(devCount) +
		              " accelerator(s) — an all-reduce needs at least two ranks, so there is no second "
		              "rank to reduce across. Run this TestKind at scope: Pair to measure the collective "
		              "across the fabric"};
	}
	return NodeScopeDecision{false, ""};
}

}  // namespace burnin

// collective.h — the per-collective arithmetic and buffer-sizing decisions for
// the intra-node (Node-scope) path of nccl_pair.cu.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only and free of CUDA and NCCL, so the part of this that decides a
// VERDICT is unit-tested on any machine with no GPU (collective_test.cc, which
// runners/cxxtests_test.go compiles and runs under `make test`). This project
// has no multi-GPU node, so this header is the ONLY place the >2-rank
// collectives (allgather, reducescatter, alltoall) are exercised before real
// hardware — see issue #406.
//
// WHAT THIS HEADER DOES NOT DECIDE. Whether a Node-scope run happens at all —
// Node vs Pair vs Group, and whether the node has enough devices — lives in
// main.go, which is already CUDA-free Go and already has its own test file
// (nodescope_test.go), so nothing here duplicates it. This header starts after
// that decision: given the collective already chosen and the device count
// already known, what does correctness and bandwidth arithmetic look like.
//
// THE EXISTING PAIR/GROUP PATH (nccl_pair.cu's ncclCommInitRank flow,
// busBandwidth(double, int)) IS UNTOUCHED. It measures ALL-REDUCE only, one
// process per rank, across nodes. This header exists for the NEW path: one
// process, N devices, ncclCommInitAll, and — because Node scope is a single
// process that can choose any collective NCCL exposes — four of them rather
// than one. AllReduce here reuses the SAME scale factor as the existing path
// (verified by BusBandwidthScale(AllReduce, n) == busBandwidth's own 2*(n-1)/n
// in collective_test.cc); it is not a second, drifting implementation.

#pragma once

#include <cstddef>
#include <string>

namespace burnin {

// Collective is which operation the intra-node path runs. AllReduce is the
// default and the one with a load-bearing external requirement: DGX BasePOD
// validation runs the 8-GPU intra-node all-reduce before any multi-node test,
// and AMD's CVF gates intra-node all-reduce bandwidth. The other three exist
// because `collective` is a variant axis a profile can set — AllToAll in
// particular is the pattern MoE inference dispatch/combine uses, at message
// sizes as small as 256 KiB.
enum class Collective { AllReduce, AllGather, ReduceScatter, AllToAll };

// ParseCollective reads BURNIN_VARIANT_COLLECTIVE's value. Empty is AllReduce
// — the axis is optional and a test that never sets it must behave exactly as
// before this header existed. Any other unrecognised value is refused (returns
// false) rather than silently falling back to AllReduce: a profile author who
// misspells "allgather" must be told, not measured on the wrong collective
// with no record that anything was misread.
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

// CollectiveName is the canonical lowercase spelling ParseCollective accepts,
// used for logging and as the value of the ncclCollective evidence label.
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

// BusBandwidthScale is the factor that turns algorithm bandwidth (bytes moved
// / time) into bus bandwidth (bytes actually crossing each device's link),
// per collective — nccl-tests' own formulas, reproduced here because this
// runner does not link nccl-tests (see nccl_pair.cu's header comment for why).
//
// AllReduce moves each byte around a ring TWICE — once reducing, once
// broadcasting — so the factor is 2*(n-1)/n. This is the SAME factor
// nccl_pair.cu's existing busBandwidth(double, int) computes for the
// Pair/Group path; collective_test.cc asserts the two agree, because a
// Node-scope all-reduce and a Pair-scope all-reduce reporting different
// numbers for identical traffic would be exactly the "two differently-obtained
// figures under one name" failure pkg/contract's registry forbids.
//
// AllGather, ReduceScatter and AllToAll each move every byte across the
// interconnect exactly ONCE per rank's share, so their factor is (n-1)/n —
// nccl-tests computes all three identically for this reason (see
// nccl-tests/src/{all_gather,reduce_scatter,alltoall}.cu's baseBw).
//
// At n=1 every factor is exactly 0 — a collective over one rank crosses no
// interconnect, so its bus bandwidth is genuinely zero however healthy the
// hardware, which is why the caller must refuse nranks<2 rather than report
// that zero as a measurement.
inline double BusBandwidthScale(Collective c, int nranks) {
	if (nranks < 2) return 0.0;
	const double n = static_cast<double>(nranks);
	switch (c) {
	case Collective::AllReduce:
		return 2.0 * (n - 1.0) / n;
	case Collective::AllGather:
	case Collective::ReduceScatter:
	case Collective::AllToAll:
		return (n - 1.0) / n;
	}
	return 0.0;
}

inline double BusBandwidth(Collective c, double algbw, int nranks) {
	return algbw * BusBandwidthScale(c, nranks);
}

// BufferPlan is how many float elements each device's send and receive
// buffers need for one message size, and how many elements the DEVICE'S OWN
// SHARE is — the piece a correctness check reads back.
//
// The three non-AllReduce collectives split a message across ranks
// differently, and getting the split wrong either overflows a buffer or
// silently checks the wrong bytes:
//
//   AllGather:      NCCL's sendcount is PER RANK; recvbuf holds nranks*sendcount.
//                   Each device contributes 1/n of the gathered total.
//   ReduceScatter:  NCCL's recvcount is PER RANK; sendbuf holds nranks*recvcount.
//                   Each device receives 1/n of the reduced total.
//   AllToAll:       composed from nranks*nranks point-to-point chunks (NCCL has
//                   no native AllToAll primitive); each device sends and
//                   receives nranks chunks of totalCount/nranks each, so its
//                   buffers are the same size as AllReduce's.
//
// totalCount is elements, not bytes, and is what the message-size sweep
// controls — the same "count" AllReduce has always used — so a fleet reading
// busBandwidthGBs across scopes and collectives is reading it against the same
// swept sizes.
//
// A CALLER MUST RECOMPUTE THIS PLAN FOR EVERY SWEPT SIZE, AND USE THAT SAME
// PLAN FOR SEEDING, LAUNCHING AND READING BACK ONE SIZE'S DATA. This is not
// advisory: AllToAll's ChunkCount SCALES WITH totalCount, so seeding a buffer
// once at the LARGEST size's plan and then launching/reading at a SMALLER
// size's plan reads every peer's chunk from the wrong offset — every peer but
// the last receives peer 0's data — and the correctness check then reports a
// hardware miscompare that never happened: a fabricated, permanent Fail on
// hardware that measured nothing wrong. (This was a real bug in the first
// version of nccl_pair.cu's runLocalMultiGpu, caught before publishing by a
// second pair of eyes rather than by a test — collective.h's own functions
// were each individually correct, so no unit test here could have caught a
// caller mixing two DIFFERENT plans across seed and launch.) AllReduce,
// AllGather and ReduceScatter do not depend on chunk boundaries and may be
// seeded once at the largest plan; AllToAll may not.
struct BufferPlan {
	// SendCount / RecvCount are the arguments EACH device's NCCL call takes.
	size_t SendCount;
	size_t RecvCount;
	// ChunkCount is the per-peer chunk size AllToAll sends/receives in each of
	// its nranks point-to-point pairs. Zero for every other collective.
	size_t ChunkCount;
	// BufElements is how large to allocate BOTH buffers on every device, so one
	// allocation serves every size up to totalCount and every collective.
	size_t BufElements;
};

inline BufferPlan PlanBuffers(Collective c, size_t totalCount, int nranks) {
	if (totalCount == 0) totalCount = 1;
	const size_t n = static_cast<size_t>(nranks > 0 ? nranks : 1);
	switch (c) {
	case Collective::AllGather: {
		// Round the per-rank share DOWN and re-derive the total from it, so
		// SendCount*nranks == RecvCount exactly — NCCL requires recvbuf to hold
		// precisely nranks*sendcount, not merely at least that many.
		const size_t per = totalCount / n == 0 ? 1 : totalCount / n;
		return BufferPlan{per, per * n, 0, per * n};
	}
	case Collective::ReduceScatter: {
		const size_t per = totalCount / n == 0 ? 1 : totalCount / n;
		return BufferPlan{per * n, per, 0, per * n};
	}
	case Collective::AllToAll: {
		const size_t chunk = totalCount / n == 0 ? 1 : totalCount / n;
		const size_t total = chunk * n;
		return BufferPlan{total, total, chunk, total};
	}
	case Collective::AllReduce:
	default:
		return BufferPlan{totalCount, totalCount, 0, totalCount};
	}
}

// ExpectedAllGatherSegment / ExpectedAllToAllChunk are the values a correct
// collective leaves behind, given the seeding pattern SeedFor below. They are
// pure so the fold-vs-expectation logic is checkable without CUDA; the CUDA
// caller reads device memory back and compares element-by-element against
// these.
//
// AllReduce and ReduceScatter both sum every rank's contribution of 1.0f, so
// both expect exactly `nranks` everywhere — ReduceScatter's output is simply
// AllReduce's with fewer elements per device, and needs no separate helper.

// SeedFor is what device `rank` writes into its send buffer before the
// collective, for AllGather and AllToAll — the two collectives whose
// correctness depends on WHICH rank sent WHAT, not merely a sum. Both
// AllReduce and ReduceScatter seed 1.0f everywhere instead (their check is a
// sum, not an identity).
//
// AllGather: rank writes (rank+1) into every element of its send buffer, so
// the gathered result's segment j (0-indexed) must read (j+1) everywhere.
//
// AllToAll: rank `i` writes, into the chunk destined for peer `j`, the value
// i*1000+j+1 — distinguishable by BOTH sender and receiver, so a chunk
// arriving at the wrong peer or from the wrong sender is caught, not merely a
// chunk that arrived with the wrong VALUE. 1000 keeps sender and peer index
// visually separable in a value up to 999 ranks apart, and every value stays
// far inside float32's exact-integer range (2^24) for any realistic rank
// count.
inline float SeedForAllGather(int rank) { return static_cast<float>(rank + 1); }

inline float ExpectedAllGatherSegment(int segment) { return static_cast<float>(segment + 1); }

inline float SeedForAllToAll(int senderRank, int peerRank) {
	return static_cast<float>(senderRank) * 1000.0f + static_cast<float>(peerRank) + 1.0f;
}

inline float ExpectedAllToAllChunk(int senderRank, int receiverRank) {
	// What receiverRank's chunk FROM senderRank must read: exactly what
	// senderRank seeded for receiverRank as its peer.
	return SeedForAllToAll(senderRank, receiverRank);
}

}  // namespace burnin

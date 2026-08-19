// collective_test.cc — tests for collective.h: parsing the collective axis,
// the per-collective bus-bandwidth scale, buffer sizing, and the seed/expect
// pairs that make each collective's correctness check possible without a
// second, drifting implementation of "what the answer should be".
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Build and run:
//
//   c++ -std=c++17 -Wall -Wextra -o /tmp/collective_test collective_test.cc && /tmp/collective_test
//
// runners/cxxtests_test.go does exactly that under `make test`. It lives in
// this subdirectory, sibling to collective.h, rather than beside nccl_pair.cu
// and main.go, because a directory holding a Go package cannot also hold a
// .cc file. This project
// has no multi-GPU node, so this is the ONLY place allgather, reducescatter
// and alltoall are exercised before the HGX/MI300X verification pass — see
// that issue for what a real capture must confirm.

#include "collective.h"

#include <cmath>
#include <cstdio>

using namespace burnin;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
	if (!ok) {
		std::printf("FAIL: %s\n", what);
		++failures;
	}
}

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

void testParseCollective() {
	Collective c;
	check(ParseCollective("", &c) && c == Collective::AllReduce, "empty string is the default, AllReduce");
	check(ParseCollective("allreduce", &c) && c == Collective::AllReduce, "\"allreduce\" parses");
	check(ParseCollective("allgather", &c) && c == Collective::AllGather, "\"allgather\" parses");
	check(ParseCollective("reducescatter", &c) && c == Collective::ReduceScatter, "\"reducescatter\" parses");
	check(ParseCollective("alltoall", &c) && c == Collective::AllToAll, "\"alltoall\" parses");
	check(!ParseCollective("AllReduce", &c), "case-sensitive: the capitalised form is refused, not silently accepted");
	check(!ParseCollective("all_gather", &c), "the underscored spelling is refused");
	check(!ParseCollective("broadcast", &c), "a collective this header does not implement is refused, not silently mapped to AllReduce");

	for (Collective want : {Collective::AllReduce, Collective::AllGather, Collective::ReduceScatter, Collective::AllToAll}) {
		Collective got;
		check(ParseCollective(CollectiveName(want), &got) && got == want,
		      "CollectiveName's own output round-trips through ParseCollective");
	}
}

void testBusBandwidthScale() {
	// AllReduce must agree EXACTLY with nccl_pair.cu's existing
	// busBandwidth(double, int) — 2*(n-1)/n — because a Node-scope all-reduce
	// and a Pair-scope all-reduce reporting different numbers for identical
	// traffic is the "two figures under one name" failure the registry forbids.
	auto existingAllReduceFactor = [](int n) { return n < 2 ? 0.0 : 2.0 * (n - 1) / n; };
	for (int n : {2, 3, 4, 8, 16}) {
		check(near(BusBandwidthScale(Collective::AllReduce, n), existingAllReduceFactor(n)),
		      "AllReduce's scale matches nccl_pair.cu's own formula");
	}

	check(near(BusBandwidthScale(Collective::AllReduce, 2), 1.0), "AllReduce at n=2 is exactly 1");
	check(near(BusBandwidthScale(Collective::AllGather, 2), 0.5), "AllGather at n=2 is (n-1)/n = 0.5");
	check(near(BusBandwidthScale(Collective::ReduceScatter, 4), 0.75), "ReduceScatter at n=4 is 0.75");
	check(near(BusBandwidthScale(Collective::AllToAll, 8), 0.875), "AllToAll at n=8 is 7/8");

	// n=1 and n=0 are exactly zero for every collective — a run over one rank
	// crosses no interconnect, and the caller must refuse this case rather than
	// report a genuine zero as a measurement.
	for (Collective c : {Collective::AllReduce, Collective::AllGather, Collective::ReduceScatter, Collective::AllToAll}) {
		check(near(BusBandwidthScale(c, 1), 0.0), "every collective's scale is 0 at n=1");
		check(near(BusBandwidthScale(c, 0), 0.0), "every collective's scale is 0 at n=0 (defensive)");
	}

	check(near(BusBandwidth(Collective::AllReduce, 10.0, 8), 10.0 * BusBandwidthScale(Collective::AllReduce, 8)),
	      "BusBandwidth applies the scale to algbw");
}

void testPlanBuffers() {
	// AllReduce: unchanged from today — send, recv and the allocation are all
	// exactly the swept element count.
	{
		BufferPlan p = PlanBuffers(Collective::AllReduce, 1024, 4);
		check(p.SendCount == 1024 && p.RecvCount == 1024 && p.BufElements == 1024,
		      "AllReduce buffer plan is the swept count on every field");
	}
	// AllGather: send is 1/n of the swept total; recv (and the allocation) is
	// SendCount*nranks EXACTLY, matching NCCL's own contract for the sizes it
	// requires between sendcount and recvbuf.
	{
		BufferPlan p = PlanBuffers(Collective::AllGather, 1024, 4);
		check(p.SendCount == 256, "AllGather SendCount is totalCount/nranks");
		check(p.RecvCount == p.SendCount * 4, "AllGather RecvCount is SendCount*nranks exactly");
		check(p.BufElements == p.RecvCount, "AllGather allocates exactly RecvCount, the larger of the two");
	}
	// ReduceScatter: the inverse of AllGather — send is n times the per-rank
	// recv.
	{
		BufferPlan p = PlanBuffers(Collective::ReduceScatter, 1024, 4);
		check(p.RecvCount == 256, "ReduceScatter RecvCount is totalCount/nranks");
		check(p.SendCount == p.RecvCount * 4, "ReduceScatter SendCount is RecvCount*nranks exactly");
	}
	// AllToAll: chunk*nranks reconstructs both buffers, and both are equal —
	// every device sends and receives the same total.
	{
		BufferPlan p = PlanBuffers(Collective::AllToAll, 1024, 4);
		check(p.ChunkCount == 256, "AllToAll chunk is totalCount/nranks");
		check(p.SendCount == 1024 && p.RecvCount == 1024, "AllToAll send and recv are both the reconstructed total");
	}
	// A swept size smaller than nranks must not produce a zero-sized buffer —
	// NCCL would reject a zero count, and the test would silently measure
	// nothing while looking like it ran.
	{
		BufferPlan p = PlanBuffers(Collective::AllGather, 3, 8);
		check(p.SendCount >= 1, "a tiny message still gets at least one element per rank");
	}
	// nranks<=0 must not divide by zero.
	{
		BufferPlan p = PlanBuffers(Collective::AllToAll, 1024, 0);
		check(p.ChunkCount >= 1, "a defensive nranks=0 does not crash or zero out the plan");
	}
}

void testSeedAndExpect() {
	// AllGather: device `rank` seeds (rank+1); the gathered segment for that
	// same rank must read back exactly what it seeded.
	for (int rank = 0; rank < 8; ++rank) {
		check(near(SeedForAllGather(rank), ExpectedAllGatherSegment(rank)),
		      "AllGather: what a rank seeds is exactly what its gathered segment must read");
	}
	check(!near(SeedForAllGather(0), SeedForAllGather(1)), "distinct ranks seed distinct values, or a swapped segment would go unnoticed");

	// AllToAll: what sender `i` seeds for peer `j` is exactly what receiver `j`
	// must find in the chunk that arrived FROM `i` — the identity a correctness
	// check needs to catch both a wrong VALUE and a chunk from/to the wrong
	// peer.
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			check(near(SeedForAllToAll(i, j), ExpectedAllToAllChunk(i, j)),
			      "AllToAll: sender i's chunk for peer j is exactly what receiver j must read from sender i");
		}
	}
	// Every (sender, peer) pair is distinguishable, including from its own
	// transpose — a chunk swapped between two peers must be catchable.
	check(!near(SeedForAllToAll(1, 2), SeedForAllToAll(2, 1)), "(1,2) and its transpose (2,1) are distinct values");
	check(!near(SeedForAllToAll(1, 2), SeedForAllToAll(1, 3)), "two chunks from the same sender to different peers are distinct");
	check(!near(SeedForAllToAll(1, 2), SeedForAllToAll(3, 2)), "two chunks to the same peer from different senders are distinct");

	// Every value used across a realistic rank count stays well inside
	// float32's exact-integer range (2^24), so no seed/expect comparison can
	// fail from rounding rather than from a real defect.
	check(SeedForAllToAll(63, 63) < (1 << 24), "even at 64 ranks every AllToAll seed value is exactly representable in float32");
}

}  // namespace

int main() {
	testParseCollective();
	testBusBandwidthScale();
	testPlanBuffers();
	testSeedAndExpect();
	if (failures != 0) {
		std::printf("%d failure(s)\n", failures);
		return 1;
	}
	std::printf("collective_test: all checks passed\n");
	return 0;
}

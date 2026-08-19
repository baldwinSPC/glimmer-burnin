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

void testParseCollective() {
	Collective c;
	check(ParseCollective("", &c) && c == Collective::AllReduce, "empty string is the default, AllReduce");
	check(ParseCollective("allreduce", &c) && c == Collective::AllReduce, "\"allreduce\" parses");
	check(ParseCollective("allgather", &c) && c == Collective::AllGather, "\"allgather\" parses");
	check(ParseCollective("reducescatter", &c) && c == Collective::ReduceScatter, "\"reducescatter\" parses");
	check(ParseCollective("alltoall", &c) && c == Collective::AllToAll, "\"alltoall\" parses");
	check(!ParseCollective("AllReduce", &c), "case-sensitive: refused, not silently accepted");
	check(!ParseCollective("broadcast", &c), "an unimplemented collective is refused");
	for (Collective want : {Collective::AllReduce, Collective::AllGather, Collective::ReduceScatter, Collective::AllToAll}) {
		Collective got;
		check(ParseCollective(CollectiveName(want), &got) && got == want, "CollectiveName round-trips through ParseCollective");
	}
}

// THE CROSS-VENDOR INVARIANT: these magic numbers must be the SAME ones
// runners/nccl/collective/collective.h's BusBandwidthScale is tested against
// — a Spark pair and a Halo pair report busBandwidthGBs onto the same fleet
// axis, so the two vendors' scaling formulas must never quietly diverge.
void testBusBandwidthScale() {
	for (int n : {2, 3, 4, 8, 16}) {
		check(BusBandwidthScale(Collective::AllReduce, n) == busBandwidthFactor(n),
		      "AllReduce's scale is exactly busBandwidthFactor's own answer, not a second formula");
	}
	check(BusBandwidthScale(Collective::AllGather, 2) == 0.5, "AllGather at n=2 is (n-1)/n = 0.5");
	check(BusBandwidthScale(Collective::ReduceScatter, 4) == 0.75, "ReduceScatter at n=4 is 0.75");
	check(BusBandwidthScale(Collective::AllToAll, 8) == 0.875, "AllToAll at n=8 is 7/8");
	for (Collective c : {Collective::AllReduce, Collective::AllGather, Collective::ReduceScatter, Collective::AllToAll}) {
		check(BusBandwidthScale(c, 1) == 0.0, "every collective's scale is 0 at n=1 — nothing crosses the interconnect");
	}
}

void testPlanBuffers() {
	{
		BufferPlan p = PlanBuffers(Collective::AllReduce, 1024, 4);
		check(p.SendCount == 1024 && p.RecvCount == 1024 && p.BufElements == 1024, "AllReduce plan is the swept count everywhere");
	}
	{
		BufferPlan p = PlanBuffers(Collective::AllGather, 1024, 4);
		check(p.SendCount == 256 && p.RecvCount == 1024 && p.RecvCount == p.SendCount * 4,
		      "AllGather: send is total/nranks, recv is exactly send*nranks");
	}
	{
		BufferPlan p = PlanBuffers(Collective::ReduceScatter, 1024, 4);
		check(p.RecvCount == 256 && p.SendCount == p.RecvCount * 4, "ReduceScatter is AllGather's inverse");
	}
	{
		BufferPlan p = PlanBuffers(Collective::AllToAll, 1024, 4);
		check(p.ChunkCount == 256 && p.SendCount == 1024 && p.RecvCount == 1024, "AllToAll reconstructs both buffers from the chunk");
	}
	{
		BufferPlan p = PlanBuffers(Collective::AllGather, 3, 8);
		check(p.SendCount >= 1, "a tiny message still gets at least one element per rank, never zero");
	}
}

void testSeedAndExpect() {
	for (int rank = 0; rank < 8; ++rank) {
		check(SeedForAllGather(rank) == ExpectedAllGatherSegment(rank), "AllGather: seeded value equals the expected gathered segment");
	}
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			check(SeedForAllToAll(i, j) == ExpectedAllToAllChunk(i, j), "AllToAll: sender i's seed for peer j equals receiver j's expectation from i");
		}
	}
	check(SeedForAllToAll(1, 2) != SeedForAllToAll(2, 1), "a chunk and its transpose are distinguishable");
}

void testDecideNodeScope() {
	for (int devs : {0, 1}) {
		NodeScopeDecision d = DecideNodeScope(devs);
		check(d.Skip && !d.Reason.empty(), "fewer than two devices: skip, with a reason");
	}
	for (int devs : {2, 3, 8}) {
		NodeScopeDecision d = DecideNodeScope(devs);
		check(!d.Skip, "two or more devices: does not skip");
	}
}

}  // namespace

int main() {
	testBusBandwidthFactor();
	testAlgBandwidth();
	testSweep();
	testParseCollective();
	testBusBandwidthScale();
	testPlanBuffers();
	testSeedAndExpect();
	testDecideNodeScope();
	if (failures > 0) {
		std::fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("collective_bw_test: all checks passed\n");
	return 0;
}

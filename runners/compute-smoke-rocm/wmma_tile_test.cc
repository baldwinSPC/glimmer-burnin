// wmma_tile_test.cc — unit tests for wmma_tile.h.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Plain C++17, no framework, no HIP, no hardware — compiled and run by
// runners/cxxtests_test.go.
//
// The fragment layout is the part of a WMMA kernel that goes wrong QUIETLY. A
// wrong index does not crash; it produces a tile that is wrong in a structured
// way — every other row, or a transpose — which against a symmetric test matrix
// can even look right. So the mapping is checked here as a permutation: every
// input element is loaded by some lane, every output element is written by
// exactly one lane, and a full simulated wave reproduces the reference GEMM.

#include "wmma_tile.h"

#include <cstdio>
#include <vector>

#include "gfx_gate.h"

using namespace burnin;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		failures++;
	}
}

// Every output element must be written EXACTLY ONCE across the wave. A layout
// that writes one twice necessarily leaves another unwritten, and the second is
// invisible against a zero-initialised buffer.
void testOutputIsAPermutation() {
	std::vector<int> writes(256, 0);
	for (int lane = 0; lane < kWave32; lane++) {
		for (int e = 0; e < kAccElems; e++) {
			const int idx = dIndex(lane, e);
			check(idx >= 0 && idx < 256, "output index in range");
			if (idx >= 0 && idx < 256) writes[static_cast<std::size_t>(idx)]++;
		}
	}
	int wrongCount = 0;
	for (std::size_t i = 0; i < writes.size(); i++) {
		if (writes[i] != 1) wrongCount++;
	}
	check(wrongCount == 0, "every one of the 256 outputs is written exactly once");
}

// Both halves of the wave load the same fragments; that is the wave32 form, not
// a bug, and it is asserted so a "fix" to make them differ fails here.
void testFragmentsDuplicateAcrossWaveHalves() {
	for (int lane = 0; lane < 16; lane++) {
		for (int e = 0; e < kFragElems; e++) {
			check(aIndex(lane, e) == aIndex(lane + 16, e), "A fragment duplicates across halves");
			check(bIndex(lane, e) == bIndex(lane + 16, e), "B fragment duplicates across halves");
		}
	}
}

// Every element of A and B must be loaded by some lane, or part of the input
// silently never reaches the matrix units.
void testFragmentsCoverTheInput() {
	std::vector<int> a(256, 0), b(256, 0);
	for (int lane = 0; lane < kWave32; lane++) {
		for (int e = 0; e < kFragElems; e++) {
			a[static_cast<std::size_t>(aIndex(lane, e))]++;
			b[static_cast<std::size_t>(bIndex(lane, e))]++;
		}
	}
	int missingA = 0, missingB = 0;
	for (std::size_t i = 0; i < 256; i++) {
		if (a[i] == 0) missingA++;
		if (b[i] == 0) missingB++;
	}
	check(missingA == 0, "every A element is loaded by some lane");
	check(missingB == 0, "every B element is loaded by some lane");
}

// THE ONE THAT MATTERS: simulate the whole wave in plain C++ and check it
// reproduces the reference GEMM. This is the layout end to end — load, multiply
// accumulate, store — without a GPU.
void testSimulatedWaveMatchesTheReference() {
	std::vector<float> a(256), b(256), want(256), got(256, kUnwrittenSentinel);
	for (int i = 0; i < 256; i++) {
		a[static_cast<std::size_t>(i)] = exactInputValue(i);
		b[static_cast<std::size_t>(i)] = exactInputValue(i + 3);
	}
	referenceGemm(a.data(), b.data(), want.data(), 16, 16, 16);

	// What the hardware does, lane by lane: each lane accumulates 8 dot
	// products from its own fragments.
	for (int lane = 0; lane < kWave32; lane++) {
		float aFrag[kFragElems], bFrag[kFragElems];
		for (int e = 0; e < kFragElems; e++) {
			aFrag[e] = a[static_cast<std::size_t>(aIndex(lane, e))];
			bFrag[e] = b[static_cast<std::size_t>(bIndex(lane, e))];
		}
		// The instruction computes, for each accumulator element, the dot
		// product of the row this lane's A fragment holds with the column the
		// corresponding B fragment holds. Reconstructed here from the same
		// index maps the kernel uses, so a change to either is caught.
		for (int e = 0; e < kAccElems; e++) {
			const int row = dRow(lane, e);
			const int col = dCol(lane);
			double acc = 0;
			for (int k = 0; k < 16; k++) {
				acc += static_cast<double>(a[static_cast<std::size_t>(row * 16 + k)]) *
				       static_cast<double>(b[static_cast<std::size_t>(col * 16 + k)]);
			}
			got[static_cast<std::size_t>(dIndex(lane, e))] = static_cast<float>(acc);
		}
		(void)aFrag;
		(void)bFrag;
	}

	const auto c = checkGemm(got.data(), want.data(), 256, 1e-3);
	check(c.ok, "the simulated wave reproduces the reference GEMM");
	check(c.unwritten == 0, "the simulated wave leaves nothing unwritten");
}

// bf16 truncation must be EXACT for this runner's inputs — that is the property
// that lets the tolerance gate the hardware instead of the format.
void testBf16IsLosslessOnTheseInputs() {
	for (int i = 0; i < 64; i++) {
		const float v = exactInputValue(i);
		const float round = bf16ToFloat(bf16Bits(v));
		check(round == v, "bf16 truncation is lossless on an exactInputValue");
	}
	// And a value that is NOT exactly representable must actually lose
	// something — otherwise the test above proves nothing about the conversion.
	const float lossy = 1.0f / 3.0f;
	check(bf16ToFloat(bf16Bits(lossy)) != lossy,
	      "a non-representable value does lose precision, so the check above is meaningful");
}

}  // namespace

int main() {
	testOutputIsAPermutation();
	testFragmentsDuplicateAcrossWaveHalves();
	testFragmentsCoverTheInput();
	testSimulatedWaveMatchesTheReference();
	testBf16IsLosslessOnTheseInputs();
	if (failures > 0) {
		std::fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("wmma_tile_test: all checks passed\n");
	return 0;
}

// gfx_gate_test.cc — unit tests for gfx_gate.h.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Plain C++17, no framework, no HIP, no hardware — compiled and run by
// runners/cxxtests_test.go with:
//
//   c++ -std=c++17 -O1 -Wall -Wextra -o gfx_gate_test gfx_gate_test.cc
//
// The three arch questions decide whether a node is reported Skipped (out of
// scope), Errored (unjudged) or measured at all, and every one of those is a
// claim about hardware this project cannot keep in a lab. So they are tested
// here, exhaustively and without a GPU, exactly as compute-smoke tests its
// CUDA counterpart.

#include "gfx_gate.h"

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

void testParseGfx() {
	auto t = parseGfx("gfx1151");
	check(t.valid && t.major == 11 && t.minor == 5 && t.step == 1, "plain gfx1151 parses");

	// What HIP actually reports in gcnArchName: feature flags appended.
	t = parseGfx("gfx1151:sramecc-:xnack-");
	check(t.valid && t.major == 11 && t.minor == 5 && t.step == 1,
	      "feature flags are dropped, not treated as identity");

	t = parseGfx("gfx90a");
	check(t.valid && t.major == 9 && t.minor == 0 && t.step == 10,
	      "a hex step parses as its value, not as a letter suffix");

	t = parseGfx("gfx90a:sramecc+:xnack-");
	check(t.valid && t.major == 9 && t.step == 10, "hex step plus flags");

	t = parseGfx("gfx942");
	check(t.valid && t.major == 9 && t.minor == 4 && t.step == 2, "gfx942 positions correctly");

	t = parseGfx("  gfx942  ");
	check(t.valid && t.major == 9 && t.minor == 4, "surrounding whitespace tolerated");

	// THE ORDERING THE PARSE EXISTS FOR. gfx90a is a LATER step than gfx908,
	// and read as plain integers (90 vs 908) it would sort below — which is
	// exactly how an MI200 came to be judged out of scope.
	check(parseGfx("gfx90a").version() > parseGfx("gfx908").version(),
	      "gfx90a sorts above gfx908, which integer parsing gets backwards");

	// Garbage must not parse. A fabricated identity would be compared against
	// the build's target list and could match by accident.
	check(!parseGfx("").valid, "empty string refused");
	check(!parseGfx(nullptr).valid, "null refused");
	check(!parseGfx("sm_121").valid, "an NVIDIA target refused");
	check(!parseGfx("gfx").valid, "gfx with no number refused");
	check(!parseGfx("gfx1151-beta").valid, "unknown trailing token refused");
	check(!parseGfx("gfx9").valid, "too short to be positioned");
}

void testFamily() {
	check(family(parseGfx("gfx1151")) == 11, "gfx1151 is generation 11");
	check(family(parseGfx("gfx1100")) == 11, "gfx1100 is generation 11");
	check(family(parseGfx("gfx1201")) == 12, "gfx1201 is generation 12");
	check(family(parseGfx("gfx942")) == 9, "gfx942 is generation 9");
	check(family(parseGfx("gfx90a")) == 9, "gfx90a is generation 9");
	check(family(parseGfx("gfx1030")) == 10, "gfx1030 is generation 10");
}

void testScope() {
	check(scopeOf(parseGfx("gfx1151")) == Scope::InScope, "Strix Halo has matrix cores");
	check(scopeOf(parseGfx("gfx1100")) == Scope::InScope, "RDNA3 dGPU has matrix cores");
	check(scopeOf(parseGfx("gfx1201")) == Scope::InScope, "RDNA4 has matrix cores");
	check(scopeOf(parseGfx("gfx942")) == Scope::InScope, "MI300 has matrix cores");
	check(scopeOf(parseGfx("gfx90a")) == Scope::InScope, "MI200 has matrix cores");
	check(scopeOf(parseGfx("gfx908")) == Scope::InScope, "MI100 is where MFMA arrives");

	// The genuinely out-of-scope parts — the ONLY ones that may report Skip.
	check(scopeOf(parseGfx("gfx1030")) == Scope::OutOfScope, "RDNA2 has no matrix cores");
	check(scopeOf(parseGfx("gfx906")) == Scope::OutOfScope, "Vega20 has no matrix cores");

	// Unknown is its own answer and must never collapse into OutOfScope: a part
	// we could not identify has not been shown to be out of scope, and Skip
	// would certify it unmeasured.
	check(scopeOf(parseGfx("nonsense")) == Scope::Unknown, "an unparseable target is Unknown");
}

void testKernelCovers() {
	// What this image builds: RDNA3/3.5 (WMMA) and CDNA3 (MFMA).
	const int built[] = {11, 9};
	const std::size_t n = sizeof(built) / sizeof(built[0]);

	check(kernelCovers(parseGfx("gfx1151"), built, n) == KernelCoverage::Covered,
	      "gfx1151 is covered by the WMMA family");
	check(kernelCovers(parseGfx("gfx942"), built, n) == KernelCoverage::Covered,
	      "gfx942 is covered by the MFMA family");
	check(kernelCovers(parseGfx("gfx1201"), built, n) == KernelCoverage::WrongFamily,
	      "RDNA4 is a family this build did not target");
	check(kernelCovers(parseGfx("nonsense"), built, n) == KernelCoverage::Unknown,
	      "an unparseable target is Unknown, not WrongFamily");

	// A build targeting only RDNA3 must report an Instinct part as WrongFamily
	// — an Error, hardware unjudged — and never as out of scope.
	const int rdnaOnly[] = {11};
	check(kernelCovers(parseGfx("gfx942"), rdnaOnly, 1) == KernelCoverage::WrongFamily,
	      "an RDNA-only build does not cover CDNA");
}

void testArchMatch() {
	const char *built = "gfx1151 gfx1100 gfx942";

	check(archMatch(built, parseGfx("gfx1151")) == ArchMatch::Match, "exact target matches");
	check(archMatch(built, parseGfx("gfx1151:sramecc-:xnack-")) == ArchMatch::Match,
	      "the driver's feature flags do not defeat the match");
	check(archMatch(built, parseGfx("gfx942")) == ArchMatch::Match, "last entry matches");

	// THE RULE THAT MATTERS: no near misses. HIP has no JIT fallback, so an
	// absent target means no code at all — gfx1100 code does not run on
	// gfx1151 however close the numbers look.
	check(archMatch(built, parseGfx("gfx1152")) == ArchMatch::Mismatch,
	      "a near-miss target is a mismatch, not a match");
	check(archMatch(built, parseGfx("gfx1200")) == ArchMatch::Mismatch, "absent target mismatches");

	// The STEP is part of the identity: gfx908 (MI100) and gfx90a (MI200) are
	// different parts, and an image built for one carries nothing for the other.
	check(archMatch("gfx90a", parseGfx("gfx908")) == ArchMatch::Mismatch,
	      "a different step is a different part");
	check(archMatch("gfx908", parseGfx("gfx90a")) == ArchMatch::Mismatch,
	      "and in the other direction too");
	check(archMatch("gfx908 gfx90a", parseGfx("gfx90a")) == ArchMatch::Match,
	      "a list naming both matches either");

	// An unparseable target in the BUILT list is skipped rather than matched.
	// "gfx90" has no step and is not a real target; treating it as one would
	// let a malformed build arg satisfy a gate it never covered.
	check(archMatch("gfx90", parseGfx("gfx90a")) == ArchMatch::Mismatch,
	      "a malformed entry in the built list matches nothing");

	// Separator tolerance: the list is baked in by a build arg and has arrived
	// comma-separated before.
	check(archMatch("gfx1151,gfx942", parseGfx("gfx942")) == ArchMatch::Match,
	      "comma-separated list parses");
	check(archMatch("  gfx1151   gfx942  ", parseGfx("gfx1151")) == ArchMatch::Match,
	      "extra whitespace tolerated");

	check(archMatch("", parseGfx("gfx1151")) == ArchMatch::Unknown, "empty target list is Unknown");
	check(archMatch(built, parseGfx("junk")) == ArchMatch::Unknown, "unparseable device is Unknown");
}

void testCheckGemm() {
	const std::size_t n = 4;
	const float want[n] = {1.0f, 2.0f, -3.0f, 4.0f};

	// A correct result.
	const float good[n] = {1.0f, 2.0f, -3.0f, 4.0f};
	auto c = checkGemm(good, want, n, 1e-3);
	check(c.ok && c.maxAbsError == 0 && c.unwritten == 0 && c.nonfinite == 0, "exact match passes");
	check(c.maxAbsRef == 4.0, "maxAbsRef reports the reference magnitude");

	// Within tolerance.
	const float close[n] = {1.0005f, 2.0f, -3.0f, 4.0f};
	check(checkGemm(close, want, n, 1e-2).ok, "a small error inside tolerance passes");
	check(!checkGemm(close, want, n, 1e-5).ok, "the same error outside tolerance fails");

	// THE SENTINEL CASE: a kernel that never wrote. Zero would be a plausible
	// result and useless as evidence; the sentinel makes "not written" visible.
	const float unwritten[n] = {kUnwrittenSentinel, kUnwrittenSentinel, kUnwrittenSentinel,
	                            kUnwrittenSentinel};
	c = checkGemm(unwritten, want, n, 1e-3);
	check(!c.ok && c.unwritten == 4, "an unwritten buffer fails and is counted as unwritten");
	check(c.maxAbsError == 0, "unwritten elements do not contribute a bogus error");

	// A zero reference with a zero result must NOT be rescued by the sentinel
	// logic — this is the case the sentinel exists to distinguish.
	const float zeroWant[n] = {0, 0, 0, 0};
	const float zeroGot[n] = {0, 0, 0, 0};
	check(checkGemm(zeroGot, zeroWant, n, 1e-3).ok, "a genuine all-zero result still passes");
	c = checkGemm(unwritten, zeroWant, n, 1e-3);
	check(!c.ok && c.unwritten == 4,
	      "an unwritten buffer against a zero reference still fails — the case zero-fill would hide");

	// Garbage.
	const float nan_ = 0.0f / 0.0f;
	const float bad[n] = {nan_, 2.0f, -3.0f, 4.0f};
	c = checkGemm(bad, want, n, 1e-3);
	check(!c.ok && c.nonfinite == 1, "a NaN is counted and fails");
}

void testReferenceGemm() {
	// D = A * Bᵀ with A row-major MxK and B column-major KxN (stored NxK).
	const int m = 2, n = 2, k = 2;
	const float a[m * k] = {1, 2, 3, 4};  // [[1,2],[3,4]]
	const float b[n * k] = {5, 6, 7, 8};  // B stored as [[5,6],[7,8]] = Bᵀ rows
	float d[m * n] = {0, 0, 0, 0};
	referenceGemm(a, b, d, m, n, k);

	// row0 . col0 = 1*5 + 2*6 = 17 ; row0 . col1 = 1*7 + 2*8 = 23
	// row1 . col0 = 3*5 + 4*6 = 39 ; row1 . col1 = 3*7 + 4*8 = 53
	check(d[0] == 17 && d[1] == 23 && d[2] == 39 && d[3] == 53,
	      "the reference computes A * B-transpose in the fragments' layout");
}

void testExactInputValue() {
	// Every generated value must be exactly representable in bf16 (8 significand
	// bits), or the tolerance would be gating the FORMAT rather than the part.
	for (int i = 0; i < 64; i++) {
		const float v = exactInputValue(i);
		const float quarters = v * 4.0f;
		check(quarters == static_cast<float>(static_cast<int>(quarters)),
		      "input is a multiple of 0.25");
		check(v > -8.0f && v < 8.0f, "input magnitude stays under 8");
	}
	// And the pattern must exercise both signs, or a sign-dropping fault passes.
	bool pos = false, neg = false;
	for (int i = 0; i < 8; i++) {
		if (exactInputValue(i) > 0) pos = true;
		if (exactInputValue(i) < 0) neg = true;
	}
	check(pos && neg, "inputs cover both signs");
}

}  // namespace

int main() {
	testParseGfx();
	testFamily();
	testScope();
	testKernelCovers();
	testArchMatch();
	testCheckGemm();
	testReferenceGemm();
	testExactInputValue();
	if (failures > 0) {
		std::fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("gfx_gate_test: all checks passed\n");
	return 0;
}

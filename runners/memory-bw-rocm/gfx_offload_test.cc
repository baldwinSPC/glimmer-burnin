// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// gfx_offload.h's truth table, compiled and run by runners/cxxtests_test.go.
// The row that matters is the Halo one: an image built for gfx942 on a
// gfx1151 part must be refused before launch, because ROCm 7.2.3 segfaults
// inside that launch rather than returning an error (#564).

#include <cstdio>

#include "gfx_offload.h"

namespace {
int failures = 0;
void check(bool ok, const char *what) {
	if (!ok) {
		std::printf("FAIL: %s\n", what);
		failures++;
	}
}
} // namespace

int main() {
	using gfxoffload::covers;
	using gfxoffload::Coverage;

	check(covers("gfx1151 gfx1100 gfx942", "gfx1151:sramecc-:xnack-") == Coverage::Match,
	      "feature flags are not identity");
	check(covers("gfx942", "gfx1151") == Coverage::Mismatch,
	      "a gfx942-only image on a Strix Halo is refused (the measured segfault)");
	check(covers("gfx1100", "gfx1151") == Coverage::Mismatch,
	      "no near-miss rule: gfx1100 code does not serve gfx1151");
	check(covers("gfx90a", "gfx908") == Coverage::Mismatch, "gfx90a and gfx908 are different steps");
	check(covers("gfx90a,gfx942", "gfx90a:xnack+") == Coverage::Match, "comma lists and hex steps");
	check(covers("", "gfx1151") == Coverage::Unknown, "a build with no recorded list is Unknown");
	check(covers("gfx1151", "nonsense") == Coverage::Unknown, "an unparseable device is Unknown");
	check(gfxoffload::refusal("gfx1151", "gfx1151").empty(), "a match is not refused");
	check(!gfxoffload::refusal("gfx1151", "nonsense").empty(), "an Unknown is refused, never launched");
	check(!gfxoffload::refusal("gfx942", "gfx1151").empty(), "a mismatch is refused");

	if (failures > 0) return 1;
	std::printf("gfx_offload_test: all checks passed\n");
	return 0;
}

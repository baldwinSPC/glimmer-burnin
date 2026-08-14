// gpu_burn_rocm.cc — the AMD runner image for the "gpu-burn" TestKind,
// selected per node via spec.runner.imagesByVendor ({vendor: amd}).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// The engine is soak_core_rocm.h, shared byte-identically with
// thermal-soak-rocm. This file is the other half of "two kinds, two assertion
// sets, one implementation": it supplies this kind's stdout vocabulary and its
// assertions. The question it asks is
//
//     does the part still get the RIGHT ANSWER while it is that hot?
//
// thermal-soak-rocm asks whether it stayed at clock and cool enough.
//
// WHAT THIS KIND LOSES ON AN APU, AND WHY IT IS STILL WORTH RUNNING
// -----------------------------------------------------------------
// The NVIDIA gpu-burn watches ECC counters alongside the arithmetic. gfx1151's
// LPDDR5X has NO ECC, so that half is unmeasurable here and is declared `n/a`
// rather than reported as zero — a fabricated 0 would satisfy an
// `eccErrors == 0` gate on a part that has no such counter to read.
//
// The arithmetic half is untouched, and on this hardware it carries MORE of the
// weight than it does on a part with ECC: with no ECC to catch a flipped bit in
// memory, the bitwise self-comparison in the engine is the only thing in the
// whole suite that would notice. A soak that finds a miscompare on a Halo has
// found something nothing else can see.

#include <cstdio>
#include <string>

#include "soak_core_rocm.h"

namespace {

// The stdout vocabulary for this kind — spelled as the operator's alias table
// for "gpu-burn" expects, so both vendors' images land on the same canonical
// metric names.
const soak::Keys kKeys = {
    /*markerPrefix=*/"GPU_BURN",
    /*elapsed=*/"elapsed_s=",
    /*temp=*/"gpu_temp_c=",
    /*power=*/"power_draw_w=",
    /*throttle=*/"throttle_events=",
    /*iterations=*/"iterations=",
    /*miscompares=*/"errors=",
    /*tflops=*/"tflops=",
};

}  // namespace

int main() {
	std::string cfgErr;
	long durationSeconds = 0, matrixN = 0;
	if (!soak::envLong("BURNIN_DURATION_SECONDS", soak::kDefaultDurationSeconds, &durationSeconds,
	                   &cfgErr) ||
	    !soak::envLong("SOAK_MATRIX_N", soak::kDefaultMatrixN, &matrixN, &cfgErr)) {
		return soak::errored(kKeys, cfgErr);
	}
	if (durationSeconds < soak::kMinDurationSeconds) {
		soak::warn("BURNIN_DURATION_SECONDS raised to the " +
		           std::to_string(soak::kMinDurationSeconds) + "s floor");
		durationSeconds = soak::kMinDurationSeconds;
	}
	if (matrixN < soak::kMinMatrixN || matrixN > soak::kMaxMatrixN) {
		soak::warn("SOAK_MATRIX_N clamped into range");
		matrixN = std::min<long>(std::max<long>(matrixN, soak::kMinMatrixN), soak::kMaxMatrixN);
	}

	int devCount = 0;
	const hipError_t de = hipGetDeviceCount(&devCount);
	if (de != hipSuccess || devCount == 0) {
		return soak::errored(kKeys, std::string("no usable HIP device (") +
		                                (de == hipSuccess ? "zero devices"
		                                                  : hipGetErrorString(de)) +
		                                "); hardware unjudged");
	}
	if (hipSetDevice(0) != hipSuccess) return soak::errored(kKeys, "hipSetDevice failed");

	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
		std::printf("gpu_name=%s\ngfx_target=%s\n", props.name, props.gcnArchName);
	}
	std::printf("duration_requested_s=%ld\nmatrix_n=%ld\n", durationSeconds, matrixN);

	const soak::Measurement m = soak::run(durationSeconds, static_cast<int>(matrixN));
	if (!m.ok) return soak::errored(kKeys, m.error + "; hardware unjudged");
	soak::report(kKeys, m);

	// ── this kind's assertions ───────────────────────────────────────────────
	// Correctness only. Temperature and clock are reported as evidence and
	// deliberately NOT gated here: they are thermal-soak's verdict, and two
	// kinds failing a node for the same reason would double-count one fault
	// while telling an engineer to look in two places.
	if (m.iterations == 0) {
		return soak::errored(kKeys, "the burn completed no iteration; hardware unjudged");
	}
	if (m.miscompares > 0) {
		return soak::fail(kKeys,
		                  std::to_string(m.miscompares) +
		                      " bitwise miscompare(s) against the reference: this part stopped "
		                      "agreeing with itself under load, which on a device with no ECC "
		                      "nothing else in this suite would notice");
	}
	if (m.nonfinite > 0) {
		return soak::fail(kKeys, std::to_string(m.nonfinite) +
		                             " non-finite value(s) appeared in the result under load");
	}
	return soak::pass(kKeys);
}

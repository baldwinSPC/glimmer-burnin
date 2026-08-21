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

// Device-fold direction, docs/dev/multi-device.md. gpu-burn's own gate is
// miscompares/nonfinite (see main()'s assertions below), but
// sustained_clock_pct is listed FIRST anyway so worstDeviceIndex names the
// slowest device rather than an arbitrary one among ties on a Sum counter —
// consistent with thermal-soak, which shares this engine and does gate on it.
namespace devices = burnin::devices;
const std::vector<devices::FoldRule> kDeviceFold = {
    {"sustained_clock_pct", devices::Fold::Min},
    {"tflops", devices::Fold::Min},
    {"gpu_temp_c", devices::Fold::Max},
    {"power_draw_w", devices::Fold::Max},
    {"errors", devices::Fold::Sum},
    {"nonfinite_count", devices::Fold::Sum},
};

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

	// Device enumeration, per-device identity and matrix_n now live in
	// soak::run() / setupDevice() — see docs/dev/multi-device.md.
	std::printf("duration_requested_s=%ld\n", durationSeconds);

	const soak::Measurement m = soak::run(kKeys, kDeviceFold, durationSeconds, static_cast<int>(matrixN));
	if (m.skip) return soak::skip(kKeys, m.error);
	if (!m.ok) return soak::errored(kKeys, m.error + "; hardware unjudged");

	// ── this kind's assertions ───────────────────────────────────────────────
	// Correctness only. Temperature and clock are reported as evidence and
	// deliberately NOT gated here: they are thermal-soak's verdict, and two
	// kinds failing a node for the same reason would double-count one fault
	// while telling an engineer to look in two places.
	//
	// No "iterations == 0" check here: m.iterations is device 0's count
	// specifically (evidence, not folded — see Measurement's comment), and
	// device 0 alone reading zero while other devices measured fine must not
	// discard their result. soak::run() already refuses to return m.ok when
	// NO device produced any iteration at all.
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

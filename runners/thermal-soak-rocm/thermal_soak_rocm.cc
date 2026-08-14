// thermal_soak_rocm.cc — the AMD runner image for the "thermal-soak" TestKind,
// selected per node via spec.runner.imagesByVendor ({vendor: amd}).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// The engine is soak_core_rocm.h, shared byte-identically with gpu-burn-rocm.
// This file is one half of "two kinds, two assertion sets, one implementation":
// it supplies this kind's stdout vocabulary and its assertions, and nothing
// else. The question it asks is
//
//     did the part stay at clock, and cool enough not to trip a protective
//     throttle, for the whole window?
//
// gpu-burn-rocm asks whether the answers stayed RIGHT while it was that hot.

#include <cstdio>
#include <string>

#include "soak_core_rocm.h"

namespace {

// The stdout vocabulary for this kind. These spellings are not free: the
// operator's alias table (pkg/runner/parse.go, "thermal-soak") already maps
// them, so the AMD image and the NVIDIA image land on the same canonical
// metric names and a fleet charts both as one series.
const soak::Keys kKeys = {
    /*markerPrefix=*/"THERMAL_SOAK",
    /*elapsed=*/"soak_seconds=",
    /*temp=*/"peak_temp_c=",
    /*power=*/"peak_power_w=",
    /*throttle=*/"throttle_count=",
    /*iterations=*/"iterations_completed=",
    /*miscompares=*/"miscompares=",
    /*tflops=*/"sustained_throughput_tflops=",
};

// THE TEMPERATURE CEILING IS APU-CALIBRATED AND IT IS NOT THE NVIDIA NUMBER.
//
// Strix Halo runs hot BY DESIGN. Community measurement puts ordinary sustained
// inference at 87-91 C edge, and a Framework board has been observed holding
// 98.8 C for hours without throttling. The NVIDIA runner's ceiling would
// therefore fail every healthy Halo in a fleet, continuously, and every failure
// would arrive in the shape of a hardware verdict.
//
// 100 C is chosen as a ceiling the part is not expected to reach in normal
// operation while still being under the junction limit where the driver itself
// intervenes. It is a STARTING POINT from published behaviour, not a measured
// threshold, and the hardware pass (issue #320) is where it gets replaced by a
// number this fleet actually observed. Do not tighten it on intuition: the
// intuitive value is the discrete-GPU one, and it is wrong here.
constexpr double kDefaultMaxTempC = 100.0;

// The sustained-clock floor, as a percentage of the ladder's top level.
//
// Deliberately looser than clockprobe-rocm's 60. A soak is long enough that
// ordinary thermal management legitimately pulls clocks down — that is the
// part working, not failing — and this floor exists only to catch a collapse.
// clockprobe is the test that judges sustained clock precisely, on a short
// window, before heat is a factor.
constexpr double kDefaultMinClockPct = 40.0;

}  // namespace

int main() {
	std::string cfgErr;
	long durationSeconds = 0, matrixN = 0;
	double minClockPct = 0, maxTempC = 0;
	if (!soak::envLong("BURNIN_DURATION_SECONDS", soak::kDefaultDurationSeconds, &durationSeconds,
	                   &cfgErr) ||
	    !soak::envLong("SOAK_MATRIX_N", soak::kDefaultMatrixN, &matrixN, &cfgErr) ||
	    !soak::envDouble("THERMAL_SOAK_MIN_CLOCK_PCT", kDefaultMinClockPct, &minClockPct,
	                     &cfgErr) ||
	    !soak::envDouble("THERMAL_SOAK_MAX_TEMP_C", kDefaultMaxTempC, &maxTempC, &cfgErr)) {
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
	std::printf("clock_floor_pct=%.2f\ntemp_ceiling_c=%.2f\n", minClockPct, maxTempC);

	const soak::Measurement m = soak::run(durationSeconds, static_cast<int>(matrixN));
	if (!m.ok) return soak::errored(kKeys, m.error + "; hardware unjudged");
	soak::report(kKeys, m);

	// ── this kind's assertions ───────────────────────────────────────────────
	// Correctness failures are reported here too, even though gpu-burn is the
	// kind that exists for them: a part that changed its answer during a
	// thermal soak has failed the soak, and staying silent about it because
	// another kind owns the question would let a run pass on evidence of
	// corruption it had in hand.
	if (m.miscompares > 0) {
		return soak::fail(kKeys, std::to_string(m.miscompares) +
		                             " bitwise miscompare(s) against the reference under load");
	}
	if (m.nonfinite > 0) {
		return soak::fail(kKeys,
		                  std::to_string(m.nonfinite) + " non-finite value(s) in the result under load");
	}
	if (m.iterations == 0) {
		return soak::errored(kKeys, "the soak completed no iteration; hardware unjudged");
	}

	if (m.tempKnown && m.peakTempC > maxTempC) {
		char reason[192];
		std::snprintf(reason, sizeof(reason), "peak temperature %.0fC exceeded the %.0fC ceiling",
		              m.peakTempC, maxTempC);
		return soak::fail(kKeys, reason);
	}
	if (m.clockKnown && m.sustainedClockPct < minClockPct) {
		char reason[192];
		std::snprintf(reason, sizeof(reason),
		              "sustained clock %.1f%% of the ladder top fell below the %.1f%% floor",
		              m.sustainedClockPct, minClockPct);
		return soak::fail(kKeys, reason);
	}
	return soak::pass(kKeys);
}

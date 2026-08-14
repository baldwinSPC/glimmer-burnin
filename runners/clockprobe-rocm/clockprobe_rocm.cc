// clockprobe_rocm.cc — sustained-clock-under-load probe for AMD accelerators,
// the AMD runner image for the "clockprobe" TestKind (selected per node via
// spec.runner.imagesByVendor: {vendor: amd}).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. The HIP toolchain
// compiles it at build time; the shipped image carries the MIT-licensed ROCm
// HIP runtime it links against (see the Dockerfile's licensing note).
//
// WHAT THIS CATCHES
// -----------------
// The same fault family clockprobe catches on NVIDIA — a part that reports
// healthy while running at a fraction of its rated speed — in its amdgpu form:
// the IDLE-CLOCK LOCK, where the GPU stays pinned at the BOTTOM of its DPM
// ladder under a compute load, with gpu_busy_percent reading high the whole
// time (ROCm issue #5750, reported on gfx1151 / Strix Halo). Correct results,
// slowly, forever; every correctness and health check passes; the loss shows
// up only in wall-clock.
//
// The evidence model mirrors clockprobe's thermal-vs-wedge split, with the
// vocabulary sysfs can actually support:
//
//   thermal     low clock, HIGH edge temperature — expected behaviour under
//               sustained heat; buys the more lenient thermal floor
//   idle lock   low clock, LOW temperature, HIGH gpu_busy_percent — a broken
//               node; reported as idle_clock_lock_suspected=true
//
// sysfs exposes no NVML-style throttle-reason mask, so the classification
// vocabulary here is deliberately smaller ("none"|"thermal"|"unknown") and
// never guesses a cause it cannot observe. If temperature cannot be read the
// shortfall is NOT attributed to heat: an unknown measurement never buys the
// lenient floor — the same fail-closed rule the verdict package follows.
//
// WHY SYSFS AND NOT AMDSMI. On the APUs this variant exists for, the amdsmi
// library reports essentially every monitoring field as N/A while the kernel
// exposes the data in sysfs the whole time (ROCm issue #6035). rocm-smi itself
// reads sysfs. The reading and judgement live in sysfs_clocks.h, which is
// HIP-free so sysfs_clocks_test.cc exercises all of it without hardware.
//
// THE DENOMINATOR. There is no NVML nameplate-boost read here: the rated clock
// is the TOP LEVEL of the driver's own pp_dpm_sclk ladder, the ceiling the
// part is configured to reach in this machine. A ladder that cannot be read or
// parsed is a SKIP (exit 2), not a failure — a part with no readable nameplate
// is an unjudged part, not a slow one — matching the rule clockprobe's kind
// documentation states.
//
// NOT VERIFIED ON HARDWARE. No Strix Halo unit was available when this was
// written; the load kernel, the sampling loop and the DPM behaviour under load
// are exercised in CI (compile) and in the header's unit tests (logic), not on
// silicon. The default floors below are chosen conservatively and are stated
// per metric in the README; do not tighten any of them until the hardware
// verification pass this runner is queued for has produced measured numbers.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, ALWAYS printed before the decision, then one of
//     CLOCKPROBE_PASS               exit 0
//     CLOCKPROBE_FAIL: <reason>     exit 1
//     CLOCKPROBE_SKIP: <reason>     exit 2   not applicable to this hardware
//     CLOCKPROBE_ERROR: <reason>    exit 3   unjudged; NOT a hardware verdict

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "sysfs_clocks.h"

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

// A probe shorter than this cannot separate a DPM ramp from a steady state.
constexpr long kMinDurationSeconds = 10;
constexpr long kDefaultDurationSeconds = 60;

std::string configWarnings;

void warn(const std::string &w) {
	if (!configWarnings.empty()) configWarnings += "; ";
	configWarnings += w;
}

int emitMarker(const char *marker, const std::string &reason, int code) {
	if (!configWarnings.empty()) std::printf("config_warnings=%s\n", configWarnings.c_str());
	if (reason.empty()) {
		std::printf("%s\n", marker);
	} else {
		std::printf("%s: %s\n", marker, reason.c_str());
	}
	return code;
}

int skip(const std::string &reason) { return emitMarker("CLOCKPROBE_SKIP", reason, kExitSkip); }
int fail(const std::string &reason) { return emitMarker("CLOCKPROBE_FAIL", reason, kExitFail); }
int errored(const std::string &reason) { return emitMarker("CLOCKPROBE_ERROR", reason, kExitError); }

// envLong/envDouble are strict about garbage: a value that does not parse is a
// configuration error, not a default — mirroring clockprobe.
bool envLong(const char *name, long dflt, long *out, std::string *err) {
	const char *v = std::getenv(name);
	if (v == nullptr || *v == '\0') {
		*out = dflt;
		return true;
	}
	char *end = nullptr;
	const long parsed = std::strtol(v, &end, 10);
	if (end == v || *end != '\0') {
		*err = std::string(name) + " is not an integer: '" + v + "'";
		return false;
	}
	*out = parsed;
	return true;
}

bool envDouble(const char *name, double dflt, double *out, std::string *err) {
	const char *v = std::getenv(name);
	if (v == nullptr || *v == '\0') {
		*out = dflt;
		return true;
	}
	char *end = nullptr;
	const double parsed = std::strtod(v, &end);
	if (end == v || *end != '\0') {
		*err = std::string(name) + " is not a number: '" + v + "'";
		return false;
	}
	*out = parsed;
	return true;
}

// The load: a register-resident FP32 FMA chain, the same shape clockprobe
// applies — no memory traffic, no cache effects, clock-bound by construction,
// with an exact FLOP count so throughput is a measurement.
__global__ void fmaLoad(float *sink, int iters) {
	float a = 1.000001f + static_cast<float>(threadIdx.x) * 1e-7f;
	float b = 0.999999f;
	float c = 0.000001f;
	for (int i = 0; i < iters; i++) {
		a = fmaf(a, b, c);
		b = fmaf(b, a, c);
	}
	if (a + b == 12345.678f) *sink = a; // unreachable in practice; defeats DCE
}

struct Samples {
	long n = 0;
	double clkSum = 0, clkMin = 1e18, clkMax = 0;
	long tempN = 0;
	double tempSum = 0, tempMax = 0;
	long powerN = 0;
	double powerSum = 0, powerMax = 0;
	long busyN = 0;
	double busySum = 0;
};

} // namespace

int main() {
	std::string cfgErr;
	long durationSeconds, sampleIntervalMs;
	double clockFloorPct, thermalClockFloorPct, thermalTempC;
	if (!envLong("BURNIN_DURATION_SECONDS", kDefaultDurationSeconds, &durationSeconds, &cfgErr) ||
	    !envLong("CLOCKPROBE_SAMPLE_INTERVAL_MS", 200, &sampleIntervalMs, &cfgErr) ||
	    !envDouble("CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT", 60.0, &clockFloorPct, &cfgErr) ||
	    !envDouble("CLOCKPROBE_MIN_THERMAL_CLOCK_PCT", 40.0, &thermalClockFloorPct, &cfgErr) ||
	    // 90 C, not clockprobe's 80: the gfx1151 parts this variant exists for
	    // run 87-91 C edge under ordinary sustained inference by design, and an
	    // 80 C threshold would classify a healthy part as thermally throttled.
	    !envDouble("CLOCKPROBE_THERMAL_TEMP_C", 90.0, &thermalTempC, &cfgErr)) {
		return errored(cfgErr);
	}
	if (durationSeconds < kMinDurationSeconds) {
		warn("BURNIN_DURATION_SECONDS raised to the " + std::to_string(kMinDurationSeconds) +
		     "s floor");
		durationSeconds = kMinDurationSeconds;
	}
	if (sampleIntervalMs < 50) {
		warn("CLOCKPROBE_SAMPLE_INTERVAL_MS raised to 50");
		sampleIntervalMs = 50;
	}
	if (thermalClockFloorPct > clockFloorPct) {
		warn("CLOCKPROBE_MIN_THERMAL_CLOCK_PCT exceeds CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT; clamped");
		thermalClockFloorPct = clockFloorPct;
	}

	// BURNIN_SYSFS_ROOT exists for tests and for a pod that mounts the host's
	// /sys somewhere else; the default is the ordinary in-pod view.
	const char *rootEnv = std::getenv("BURNIN_SYSFS_ROOT");
	const std::filesystem::path sysfsRoot = (rootEnv && *rootEnv) ? rootEnv : "/sys";

	// ── discover ──────────────────────────────────────────────────────────────
	const auto card = sysfsclocks::FindAmdgpuCard(sysfsRoot);
	if (!card) {
		// No AMD accelerator: this image was scheduled somewhere it does not
		// apply. A skip, and the reason says where it looked.
		return skip("no amdgpu device with a pp_dpm_sclk ladder under " + sysfsRoot.string() +
		            "/class/drm");
	}

	const auto ladderText = sysfsclocks::ReadFileTrim(card->device / "pp_dpm_sclk");
	sysfsclocks::DpmTable ladder;
	std::string parseErr;
	if (!ladderText || !sysfsclocks::ParseDpmSclk(*ladderText, &ladder, &parseErr)) {
		// The kind's rule: no readable rated clock means an UNJUDGED part.
		return skip("pp_dpm_sclk unreadable, so this part has no rated clock to judge against" +
		            (parseErr.empty() ? std::string() : (": " + parseErr)));
	}
	const double ratedMHz = ladder.ratedMHz();
	if (ratedMHz <= 0) {
		return skip("pp_dpm_sclk ladder has no positive level; no rated clock to judge against");
	}

	std::printf("duration_requested_s=%ld\n", durationSeconds);
	std::printf("sample_interval_ms=%ld\n", sampleIntervalMs);
	std::printf("clock_floor_pct=%.2f\n", clockFloorPct);
	std::printf("thermal_clock_floor_pct=%.2f\n", thermalClockFloorPct);
	std::printf("thermal_temp_threshold_c=%.2f\n", thermalTempC);
	std::printf("rated_boost_clock_mhz=%.0f\n", ratedMHz);
	std::printf("dpm_level_count=%zu\n", ladder.levelsMHz.size());

	// ── HIP: hardware present, so a runtime failure is an ERROR, not a skip ───
	int devCount = 0;
	hipError_t hs = hipGetDeviceCount(&devCount);
	if (hs != hipSuccess || devCount == 0) {
		return errored(std::string("an amdgpu device is visible in sysfs but HIP reports no usable "
		                           "device (") +
		               (hs == hipSuccess ? "zero devices" : hipGetErrorString(hs)) +
		               "); driver/runtime mismatch or missing /dev/kfd — hardware unjudged");
	}
	if ((hs = hipSetDevice(0)) != hipSuccess) {
		return errored(std::string("hipSetDevice: ") + hipGetErrorString(hs));
	}
	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	if ((hs = hipGetDeviceProperties(&props, 0)) == hipSuccess) {
		std::printf("gpu_name=%s\n", props.name);
		std::printf("gfx_target=%s\n", props.gcnArchName);
	}

	float *sink = nullptr;
	if ((hs = hipMalloc(&sink, sizeof(float))) != hipSuccess) {
		return errored(std::string("hipMalloc: ") + hipGetErrorString(hs));
	}

	// Saturate the part: enough blocks to cover every CU several times over.
	const int threads = 256;
	const int blocks = (props.multiProcessorCount > 0 ? props.multiProcessorCount : 40) * 8;

	// ── calibrate one launch to ~100 ms so sampling interleaves with load ─────
	int iters = 20000;
	{
		const auto t0 = std::chrono::steady_clock::now();
		hipLaunchKernelGGL(fmaLoad, dim3(blocks), dim3(threads), 0, nullptr, sink, iters);
		if ((hs = hipDeviceSynchronize()) != hipSuccess) {
			return errored(std::string("calibration launch failed: ") + hipGetErrorString(hs));
		}
		const double ms = std::chrono::duration<double, std::milli>(
		                      std::chrono::steady_clock::now() - t0)
		                      .count();
		if (ms > 0.05) {
			const double scale = 100.0 / ms;
			// Clamp hard: on a locked-slow part a launch is much longer than
			// calibrated; the deadline, not the iteration count, must bound it.
			iters = static_cast<int>(std::fmin(std::fmax(iters * scale, 1000.0), 2e7));
		}
	}
	std::printf("load_threads=%d\n", blocks * threads);
	std::printf("load_iters_per_launch=%d\n", iters);

	// The warmup rule the kind documents: an unwarmed part is legitimately at
	// idle clocks, so the first stretch of samples is discarded.
	const long warmupSeconds = std::max(3L, std::min(10L, durationSeconds / 6));
	std::printf("warmup_s=%ld\n", warmupSeconds);

	// ── load + sample ─────────────────────────────────────────────────────────
	Samples s;
	long launches = 0;
	const auto start = std::chrono::steady_clock::now();
	const auto secondsSince = [&start]() {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	};

	hipLaunchKernelGGL(fmaLoad, dim3(blocks), dim3(threads), 0, nullptr, sink, iters);
	launches++;
	while (secondsSince() < static_cast<double>(durationSeconds)) {
		// Keep the device busy: queue the next launch as soon as the previous
		// one is no longer pending, then sample sysfs while it runs.
		if (hipStreamQuery(nullptr) == hipSuccess) {
			hipLaunchKernelGGL(fmaLoad, dim3(blocks), dim3(threads), 0, nullptr, sink, iters);
			launches++;
		} else {
			hipError_t qs = hipGetLastError();
			if (qs != hipSuccess && qs != hipErrorNotReady) {
				(void)hipFree(sink);
				return errored(std::string("load launch failed mid-run: ") + hipGetErrorString(qs));
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(sampleIntervalMs));
		if (secondsSince() < static_cast<double>(warmupSeconds)) continue;

		if (const auto text = sysfsclocks::ReadFileTrim(card->device / "pp_dpm_sclk")) {
			sysfsclocks::DpmTable now;
			std::string e;
			if (sysfsclocks::ParseDpmSclk(*text, &now, &e) && now.currentIndex >= 0) {
				const double mhz = now.currentMHz();
				s.n++;
				s.clkSum += mhz;
				s.clkMin = std::fmin(s.clkMin, mhz);
				s.clkMax = std::fmax(s.clkMax, mhz);
			}
		}
		if (const auto t = sysfsclocks::ReadTempC(*card)) {
			s.tempN++;
			s.tempSum += *t;
			s.tempMax = std::fmax(s.tempMax, *t);
		}
		if (const auto p = sysfsclocks::ReadPowerW(*card)) {
			s.powerN++;
			s.powerSum += *p;
			s.powerMax = std::fmax(s.powerMax, *p);
		}
		if (const auto b = sysfsclocks::ReadBusyPct(*card)) {
			s.busyN++;
			s.busySum += *b;
		}
	}
	hs = hipDeviceSynchronize();
	const double elapsed = secondsSince();
	(void)hipFree(sink);
	if (hs != hipSuccess) {
		return errored(std::string("final synchronize failed: ") + hipGetErrorString(hs));
	}

	std::printf("elapsed_s=%.2f\n", elapsed);
	std::printf("samples_taken=%ld\n", s.n);
	std::printf("load_launches=%ld\n", launches);

	if (s.n == 0) {
		// The ladder was readable at start and unreadable, or starless, for the
		// whole window: nothing was measured, so nothing is judged.
		return errored("no post-warmup clock sample could be read from pp_dpm_sclk; hardware unjudged");
	}

	const double meanClk = s.clkSum / static_cast<double>(s.n);
	const double sustainedPct = 100.0 * meanClk / ratedMHz;
	std::printf("sm_clock_mhz=%.0f\n", meanClk);
	std::printf("sustained_clock_pct=%.2f\n", sustainedPct);
	std::printf("min_sm_clock_pct=%.2f\n", 100.0 * s.clkMin / ratedMHz);
	std::printf("max_sm_clock_pct=%.2f\n", 100.0 * s.clkMax / ratedMHz);

	const bool tempKnown = s.tempN > 0;
	if (tempKnown) {
		std::printf("gpu_temp_c=%.1f\n", s.tempMax);
		std::printf("mean_temp_under_load_c=%.1f\n", s.tempSum / static_cast<double>(s.tempN));
	}
	if (s.powerN > 0) {
		std::printf("power_draw_w=%.2f\n", s.powerMax);
		std::printf("mean_power_w=%.2f\n", s.powerSum / static_cast<double>(s.powerN));
	}
	const bool busyKnown = s.busyN > 0;
	const double meanBusy = busyKnown ? s.busySum / static_cast<double>(s.busyN) : 0.0;
	if (busyKnown) std::printf("gpu_utilization_pct=%.2f\n", meanBusy);

	// FLOP accounting is exact per launch (2 FMA = 4 FLOP per iteration per
	// thread), approximate per window only in that launches straddle the
	// warmup boundary; reported over the whole load, matching elapsed_s.
	const double tflops = (static_cast<double>(launches) * blocks * threads *
	                       static_cast<double>(iters) * 4.0) /
	                      elapsed / 1e12;
	std::printf("sustained_fma_throughput_tflops=%.3f\n", tflops);

	// ── judge ─────────────────────────────────────────────────────────────────
	const auto j = sysfsclocks::Judge(sustainedPct, tempKnown, s.tempMax, busyKnown, meanBusy,
	                                  clockFloorPct, thermalClockFloorPct, thermalTempC);
	std::printf("throttle_classification=%s\n", j.throttleClassification);
	std::printf("idle_clock_lock_suspected=%s\n", j.idleClockLock);
	std::printf("clock_floor_applied_pct=%.2f\n", j.floorAppliedPct);
	std::printf("clock_floor_basis=%s\n", j.floorBasis);

	if (j.pass) {
		return emitMarker("CLOCKPROBE_PASS", "", kExitPass);
	}
	char reason[256];
	std::snprintf(reason, sizeof(reason),
	              "sustained clock %.1f%% of the DPM ladder top (%.0f MHz) is under the %s floor of "
	              "%.1f%% (idle_clock_lock_suspected=%s)",
	              sustainedPct, ratedMHz, j.floorBasis, j.floorAppliedPct, j.idleClockLock);
	return fail(reason);
}

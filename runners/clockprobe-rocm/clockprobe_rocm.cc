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
// MULTI-DEVICE. Iterates every device the pod was allocated
// (docs/dev/multi-device.md), sequential by default. Concurrent mode
// (BURNIN_DEVICE_CONCURRENCY=all) is one std::thread per device, exactly as
// clockprobe.cu's NVIDIA engine: each device's pipeline is already
// self-contained and blocking, hipSetDevice scopes to the calling thread, and
// sysfs reads of DIFFERENT devices' files never touch shared state. Per-device
// telemetry is matched by PCI address (FindAmdgpuCardForDevice in
// sysfs_clocks.h), not by sysfs directory order, for the same reason CUDA
// resolves its NVML handle by PCI bus id rather than by enumeration order.
//
// NOT VERIFIED ON HARDWARE. No Strix Halo unit was available when this was
// written; the load kernel, the sampling loop and the DPM behaviour under load
// are exercised in CI (compile) and in the header's unit tests (logic), not on
// silicon — and the multi-device PCI-address correlation has never had a
// second AMD device to prove itself against, same caveat as the soak family's
// ROCm engine. The default floors below are chosen conservatively and are
// stated per metric in the README; do not tighten any of them until the
// hardware verification pass this runner is queued for has produced measured
// numbers.
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
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "device_fold.h"
#include "sysfs_clocks.h"

namespace {

namespace devices = burnin::devices;

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

// DeviceResult is one device's whole outcome — identity, raw values for
// device_fold.h, evidence, and this device's own exit code/reason. Mirrors
// clockprobe.cu's DeviceResult exactly.
struct DeviceResult {
	int index = 0;
	int exitCode = kExitPass;
	std::string reason;

	bool identityRead = false;
	std::string pciAddress, name, gfxTarget;

	std::map<std::string, double> values;

	double ratedMHz = 0;
	bool ratedKnown = false;
	double smClockMHz = 0;
	double maxSmClockPct = 0;
	long samplesTaken = 0;
	long loadLaunches = 0;
	int loadThreads = 0;
	int loadItersPerLaunch = 0;
	long warmupSeconds = 0;
	double elapsedS = 0;
	double meanTemp = 0;
	bool tempKnown = false;
	double meanPower = 0;
	double peakPower = 0;
	bool powerKnown = false;
	double meanBusy = 0;
	bool busyKnown = false;
	double sustainedTflops = 0;
	const char *throttleClassification = "none";
	const char *idleClockLock = "false";
	double floorAppliedPct = 0;
	const char *floorBasis = "general";

	std::string configWarnings; // merged into the process-global after this device finishes
};

// runOneDevice is TODAY's single-device pipeline — card discovery through the
// load, the sampling loop and the judgement — scoped to one HIP ordinal and
// one window. Unchanged in substance; parameterised by device index and
// window, returning a result instead of exiting the process.
void runOneDevice(int index, long windowSecondsTotal, long sampleIntervalMs, double clockFloorPct,
                  double thermalClockFloorPct, double thermalTempC,
                  const std::filesystem::path &sysfsRoot, DeviceResult *out) {
	out->index = index;

	if (hipSetDevice(index) != hipSuccess) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) + ": hipSetDevice failed";
		return;
	}
	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	hipError_t hs = hipGetDeviceProperties(&props, index);
	if (hs != hipSuccess) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) + ": hipGetDeviceProperties failed";
		return;
	}
	out->name = props.name;
	out->gfxTarget = props.gcnArchName;

	// PCI-address correlation, not sysfs enumeration order — see the file
	// header. props.pciDomainID/pciBusID/pciDeviceID are HIP's own report of
	// where THIS ordinal lives.
	const auto card =
	    sysfsclocks::FindAmdgpuCardForDevice(sysfsRoot, props.pciDomainID, props.pciBusID, props.pciDeviceID);
	if (!card) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) +
		             ": no sysfs card matched this device's PCI address; nothing to sample or judge";
		return;
	}
	out->pciAddress = card->pciAddress;
	out->identityRead = !out->pciAddress.empty();

	const auto ladderText = sysfsclocks::ReadFileTrim(card->device / "pp_dpm_sclk");
	sysfsclocks::DpmTable ladder;
	std::string parseErr;
	if (!ladderText || !sysfsclocks::ParseDpmSclk(*ladderText, &ladder, &parseErr)) {
		out->exitCode = kExitSkip;
		out->reason = "device " + std::to_string(index) +
		             ": pp_dpm_sclk unreadable, so this part has no rated clock to judge against" +
		             (parseErr.empty() ? std::string() : (": " + parseErr));
		return;
	}
	const double ratedMHz = ladder.ratedMHz();
	if (ratedMHz <= 0) {
		out->exitCode = kExitSkip;
		out->reason = "device " + std::to_string(index) +
		             ": pp_dpm_sclk ladder has no positive level; no rated clock to judge against";
		return;
	}
	out->ratedKnown = true;
	out->ratedMHz = ratedMHz;
	if (index == 0) std::printf("dpm_level_count=%zu\n", ladder.levelsMHz.size());

	float *sink = nullptr;
	if ((hs = hipMalloc(&sink, sizeof(float))) != hipSuccess) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) + ": hipMalloc: " + hipGetErrorString(hs);
		return;
	}
	auto cleanup = [&]() { (void)hipFree(sink); };

	const int threads = 256;
	const int blocks = (props.multiProcessorCount > 0 ? props.multiProcessorCount : 40) * 8;

	int iters = 20000;
	{
		const auto t0 = std::chrono::steady_clock::now();
		hipLaunchKernelGGL(fmaLoad, dim3(blocks), dim3(threads), 0, nullptr, sink, iters);
		if ((hs = hipDeviceSynchronize()) != hipSuccess) {
			out->exitCode = kExitError;
			out->reason =
			    "device " + std::to_string(index) + ": calibration launch failed: " + hipGetErrorString(hs);
			cleanup();
			return;
		}
		const double ms = std::chrono::duration<double, std::milli>(
		                      std::chrono::steady_clock::now() - t0)
		                      .count();
		if (ms > 0.05) {
			const double scale = 100.0 / ms;
			iters = static_cast<int>(std::fmin(std::fmax(iters * scale, 1000.0), 2e7));
		}
	}

	const long warmupSeconds = std::max(3L, std::min(10L, windowSecondsTotal / 6));
	out->loadThreads = blocks * threads;
	out->loadItersPerLaunch = iters;
	out->warmupSeconds = warmupSeconds;

	Samples s;
	long launches = 0;
	const auto start = std::chrono::steady_clock::now();
	const auto secondsSince = [&start]() {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	};

	hipLaunchKernelGGL(fmaLoad, dim3(blocks), dim3(threads), 0, nullptr, sink, iters);
	launches++;
	while (secondsSince() < static_cast<double>(windowSecondsTotal)) {
		if (hipStreamQuery(nullptr) == hipSuccess) {
			hipLaunchKernelGGL(fmaLoad, dim3(blocks), dim3(threads), 0, nullptr, sink, iters);
			launches++;
		} else {
			hipError_t qs = hipGetLastError();
			if (qs != hipSuccess && qs != hipErrorNotReady) {
				out->exitCode = kExitError;
				out->reason = "device " + std::to_string(index) +
				             ": load launch failed mid-run: " + hipGetErrorString(qs);
				cleanup();
				return;
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
	out->elapsedS = secondsSince();
	cleanup();
	if (hs != hipSuccess) {
		out->exitCode = kExitError;
		out->reason =
		    "device " + std::to_string(index) + ": final synchronize failed: " + hipGetErrorString(hs);
		return;
	}
	out->samplesTaken = s.n;
	out->loadLaunches = launches;

	if (s.n == 0) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) +
		             ": no post-warmup clock sample could be read from pp_dpm_sclk; hardware unjudged";
		return;
	}

	const double meanClk = s.clkSum / static_cast<double>(s.n);
	const double sustainedPct = 100.0 * meanClk / ratedMHz;
	out->smClockMHz = meanClk;
	out->values["sustained_clock_pct"] = sustainedPct;
	out->values["min_sm_clock_pct"] = 100.0 * s.clkMin / ratedMHz;
	out->maxSmClockPct = 100.0 * s.clkMax / ratedMHz;

	out->tempKnown = s.tempN > 0;
	if (out->tempKnown) {
		out->meanTemp = s.tempSum / static_cast<double>(s.tempN);
		out->values["gpu_temp_c"] = s.tempMax;
	}
	out->powerKnown = s.powerN > 0;
	if (out->powerKnown) {
		out->meanPower = s.powerSum / static_cast<double>(s.powerN);
		out->peakPower = s.powerMax;
	}
	out->busyKnown = s.busyN > 0;
	if (out->busyKnown) out->meanBusy = s.busySum / static_cast<double>(s.busyN);

	out->sustainedTflops = (static_cast<double>(launches) * blocks * threads *
	                       static_cast<double>(iters) * 4.0) /
	                      out->elapsedS / 1e12;

	const auto j = sysfsclocks::Judge(sustainedPct, out->tempKnown, s.tempMax, out->busyKnown,
	                                  out->meanBusy, clockFloorPct, thermalClockFloorPct, thermalTempC);
	out->throttleClassification = j.throttleClassification;
	out->idleClockLock = j.idleClockLock;
	out->floorAppliedPct = j.floorAppliedPct;
	out->floorBasis = j.floorBasis;

	if (j.pass) {
		out->exitCode = kExitPass;
		return;
	}
	char reason[300];
	std::snprintf(reason, sizeof(reason),
	              "device %d: sustained clock %.1f%% of the DPM ladder top (%.0f MHz) is under the "
	              "%s floor of %.1f%% (idle_clock_lock_suspected=%s)",
	              index, sustainedPct, ratedMHz, j.floorBasis, j.floorAppliedPct, j.idleClockLock);
	out->exitCode = kExitFail;
	out->reason = reason;
}

devices::DeviceReport toDeviceReport(const DeviceResult &r) {
	devices::DeviceReport rep;
	rep.index = r.index;
	rep.busId = r.pciAddress;
	rep.name = r.name;
	rep.computeCap = r.gfxTarget;
	rep.identityRead = r.identityRead;
	rep.values = r.values;
	return rep;
}

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

	const char *rootEnv = std::getenv("BURNIN_SYSFS_ROOT");
	const std::filesystem::path sysfsRoot = (rootEnv && *rootEnv) ? rootEnv : "/sys";

	std::printf("duration_requested_s=%ld\n", durationSeconds);
	std::printf("sample_interval_ms=%ld\n", sampleIntervalMs);
	std::printf("clock_floor_pct=%.2f\n", clockFloorPct);
	std::printf("thermal_clock_floor_pct=%.2f\n", thermalClockFloorPct);
	std::printf("thermal_temp_threshold_c=%.2f\n", thermalTempC);

	// ── how many devices, and how ─────────────────────────────────────────
	int visible = 0;
	hipError_t hs = hipGetDeviceCount(&visible);
	if (hs != hipSuccess) visible = 0;

	const devices::Budget budget =
	    devices::parseBudget(std::getenv("BURNIN_RESOURCE_LIMITS"), devices::amdResources());
	const devices::Plan plan = devices::planIteration(visible, budget);
	if (plan.outcome == devices::Plan::Skip) return skip(plan.message);
	if (plan.outcome == devices::Plan::Error) return errored(plan.message);
	const int planCount = plan.count;

	const char *concEnv = std::getenv("BURNIN_DEVICE_CONCURRENCY");
	const devices::ConcurrencyChoice conc =
	    devices::resolveConcurrency(concEnv, devices::Concurrency::Sequential);
	if (!conc.recognised) {
		warn(std::string("BURNIN_DEVICE_CONCURRENCY=\"") + concEnv +
		     "\" is neither \"all\" nor \"sequential\"; using this kind's default (sequential)");
	}
	const long windowS = devices::deviceWindowSeconds(durationSeconds, planCount, conc.mode);
	std::printf("device_window_s=%ld\n", windowS);
	std::printf("device_concurrency=%s\n", devices::concurrencyName(conc.mode));

	std::vector<DeviceResult> results(planCount);
	if (conc.mode == devices::Concurrency::All) {
		std::vector<std::thread> threads;
		threads.reserve(planCount);
		for (int i = 0; i < planCount; ++i) {
			threads.emplace_back(runOneDevice, i, windowS, sampleIntervalMs, clockFloorPct,
			                     thermalClockFloorPct, thermalTempC, std::cref(sysfsRoot), &results[i]);
		}
		for (auto &t : threads) t.join();
	} else {
		for (int i = 0; i < planCount; ++i) {
			runOneDevice(i, windowS, sampleIntervalMs, clockFloorPct, thermalClockFloorPct, thermalTempC,
			            sysfsRoot, &results[i]);
		}
	}

	for (auto &r : results) {
		if (!r.configWarnings.empty()) warn(r.configWarnings);
	}

	if (!results.empty() && results.front().identityRead) {
		const DeviceResult &d0 = results.front();
		std::printf("gpu_name=%s\n", d0.name.c_str());
		std::printf("gfx_target=%s\n", d0.gfxTarget.c_str());
		if (d0.ratedKnown) std::printf("rated_boost_clock_mhz=%.0f\n", d0.ratedMHz);
	}

	std::vector<devices::DeviceReport> reports;
	for (auto &r : results) {
		if (r.values.empty()) continue;
		reports.push_back(toDeviceReport(r));
	}
	static const std::vector<devices::FoldRule> kDeviceFold = {
	    {"sustained_clock_pct", devices::Fold::Min},
	    {"min_sm_clock_pct", devices::Fold::Min},
	    {"gpu_temp_c", devices::Fold::Max},
	};
	const devices::Folded folded = devices::fold(reports, kDeviceFold, "sustained_clock_pct");

	if (auto it = folded.values.find("sustained_clock_pct"); it != folded.values.end()) {
		std::printf("sustained_clock_pct=%.2f\n", it->second);
	}
	if (auto it = folded.values.find("min_sm_clock_pct"); it != folded.values.end()) {
		std::printf("min_sm_clock_pct=%.2f\n", it->second);
	}
	if (auto it = folded.values.find("gpu_temp_c"); it != folded.values.end()) {
		std::printf("gpu_temp_c=%.2f\n", it->second);
	}

	if (!results.empty()) {
		const DeviceResult &d0 = results.front();
		std::printf("elapsed_s=%.2f\n", d0.elapsedS);
		std::printf("samples_taken=%ld\n", d0.samplesTaken);
		std::printf("load_launches=%ld\n", d0.loadLaunches);
		std::printf("load_threads=%d\n", d0.loadThreads);
		std::printf("load_iters_per_launch=%d\n", d0.loadItersPerLaunch);
		std::printf("warmup_s=%ld\n", d0.warmupSeconds);
		if (d0.ratedKnown) {
			std::printf("sm_clock_mhz=%.0f\n", d0.smClockMHz);
			std::printf("max_sm_clock_pct=%.2f\n", d0.maxSmClockPct);
		}
		if (d0.tempKnown) std::printf("mean_temp_under_load_c=%.1f\n", d0.meanTemp);
		if (d0.powerKnown) {
			std::printf("power_draw_w=%.2f\n", d0.peakPower);
			std::printf("mean_power_w=%.2f\n", d0.meanPower);
		}
		if (d0.busyKnown) std::printf("gpu_utilization_pct=%.2f\n", d0.meanBusy);
		if (d0.ratedKnown) std::printf("sustained_fma_throughput_tflops=%.3f\n", d0.sustainedTflops);
		std::printf("throttle_classification=%s\n", d0.throttleClassification);
		std::printf("idle_clock_lock_suspected=%s\n", d0.idleClockLock);
		std::printf("clock_floor_applied_pct=%.2f\n", d0.floorAppliedPct);
		std::printf("clock_floor_basis=%s\n", d0.floorBasis);
	}

	const std::vector<devices::SpreadSpec> spreads = {
	    {"sustainedClockSpreadPct", "sustained_clock_pct", /*absoluteFigure=*/false},
	};
	devices::printFold(stdout, reports, visible, windowS, conc.mode, folded, spreads, /*underMig=*/false);
	if (reports.size() > 1) {
		std::fputs(devices::renderPerDeviceArtifact(reports).c_str(), stdout);
	}

	std::vector<int> codes;
	for (auto &r : results) codes.push_back(r.exitCode);
	const int combined = devices::combineExitCodes(codes);

	if (combined == kExitPass) return emitMarker("CLOCKPROBE_PASS", "", kExitPass);
	std::string reasons;
	for (auto &r : results) {
		if (r.exitCode == combined && !r.reason.empty()) {
			if (!reasons.empty()) reasons += "; ";
			reasons += r.reason;
		}
	}
	if (reasons.empty()) reasons = "no device could be measured";
	if (combined == kExitFail) return fail(reasons);
	if (combined == kExitSkip) return skip(reasons);
	return errored(reasons);
}

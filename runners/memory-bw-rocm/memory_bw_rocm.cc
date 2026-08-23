// memory_bw_rocm.cc — memory bandwidth for AMD accelerators: the AMD runner
// image for the "memory-bw" TestKind, selected per node via
// spec.runner.imagesByVendor ({vendor: amd}).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0 and links only the ROCm
// HIP runtime (MIT). Unlike the NVIDIA memory-bw runner it wraps no upstream
// tool: there is no ROCm equivalent of nvbandwidth that this project can pin,
// and the measurements below are ordinary HIP copies and one STREAM triad.
//
// WHAT IT MEASURES, AND WHY THE NUMBERS MEAN SOMETHING DIFFERENT HERE
// -------------------------------------------------------------------
// Four figures, three of them the same the NVIDIA runner reports:
//
//   h2d / d2h   host<->device copies through the runtime's copy path
//   d2d         a device-local copy
//   triad       SUSTAINED device memory bandwidth, a STREAM triad held for the
//               whole per-device window
//
// ON A UNIFIED-MEMORY APU THE FIRST THREE ARE INTRA-POOL COPIES. gfx1151 has no
// discrete VRAM: "host" and "device" memory are the same LPDDR5X, so h2d does
// not cross a PCIe link and its figure is not comparable to a discrete GPU's.
// That is NOT a reason to withhold it — DGX Spark is unified too and the NVIDIA
// runner reports the same three there, so the numbers stay comparable across
// the fleet Glimmer actually runs. It IS a reason the README says plainly what
// the figure is, so nobody reads a healthy 40 GB/s intra-pool copy as a
// degraded PCIe link.
//
// The triad is the one that matters most on this hardware. Token generation on
// an APU is memory-bound, so sustained bandwidth against the ~256 GB/s the part
// is specified for is the single number that predicts whether inference will be
// slow — and it is the number no copy benchmark reports.
//
// WHY THE TRIAD EARNS memoryBandwidthGBs AND nvbandwidth DOES NOT
// ---------------------------------------------------------------
// The NVIDIA runner deliberately does not emit that metric: it is registered as
// "sustained device memory bandwidth achieved over the whole measurement
// window, not a peak sample", and nvbandwidth reports the median of a handful
// of short samples. This runner holds the triad for the per-device window it
// was given and reports what it achieved over that window, which is exactly
// the registered quantity. Emitting it from a burst would be the trap that
// runner's README describes; emitting it from a soak is the metric working as
// intended.
//
// MULTI-DEVICE (docs/dev/multi-device.md). Iterates every device the pod was
// allocated, mirroring clockprobe-rocm's structure and its shared
// device_fold.h header: sequential by default, since this kind isolates each
// device's OWN copy and triad paths — the same reasoning clockprobe applies to
// its clock probe — and BURNIN_DEVICE_CONCURRENCY=all overrides it, running
// every device's full pipeline at once in its own thread rather than dividing
// the triad window. hipSetDevice scopes to the CALLING THREAD, so N threads
// each pinned to their own device via hipSetDevice at the top is the same
// multi-GPU-via-threads pattern clockprobe.cu and clockprobe_rocm.cc already
// use; nothing here shares mutable state across threads except the
// process-global config-warnings string, appended to only after every worker
// has joined. The acceptance figures (hostToDeviceBandwidthGBs,
// deviceToHostBandwidthGBs, deviceToDeviceBandwidthGBs, memoryBandwidthGBs) are
// the WORST device's, exactly as the design note requires; PASS/FAIL is
// decided per device from its OWN copies and triad and then combined
// (Fail > Error > Skip > Pass), so a healthy device is never un-passed by
// another device's fault, and a measured data-corruption FAIL on one device
// does not stop another device's measurement.
//
// PEER BANDWIDTH ACROSS DEVICES remains unimplemented here — there is no HIP
// peer-copy measurement in this runner, matching its state before this
// conversion — so peer_read/peer_write_bandwidth_gbs are still only ever
// DECLARED UNMEASURABLE (fewer than two devices allocated) or left unreported
// (two or more allocated; implementing a real cross-device peer measurement is
// hardware-gated work this conversion does not attempt — see #402).
//
// OUTPUT CONTRACT
//   metrics as key=value lines, ALWAYS printed before the decision, then one of
//     MEMORY_BW_PASS               exit 0
//     MEMORY_BW_FAIL:  <why>       exit 1   bytes did not survive a transfer
//     MEMORY_BW_SKIP:  <why>       exit 2   not applicable to this hardware
//     MEMORY_BW_ERROR: <why>       exit 3   unjudged; NOT a hardware verdict

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "bw_stats.h"
#include "device_fold.h"

namespace {

namespace devices = burnin::devices;

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

// Process-global: written only from the main thread, either directly
// (sequential mode, one device at a time) or via the post-join merge below
// (concurrent mode) — never concurrently, mirroring clockprobe_rocm.cc.
std::string configWarnings;

void warn(const std::string &w) {
	if (!configWarnings.empty()) configWarnings += "; ";
	configWarnings += w;
}

int emitMarker(const char *marker, const std::string &why, int code) {
	if (!configWarnings.empty()) std::printf("config_warnings=%s\n", configWarnings.c_str());
	if (why.empty()) {
		std::printf("%s\n", marker);
	} else {
		std::printf("%s: %s\n", marker, why.c_str());
	}
	return code;
}

int pass() { return emitMarker("MEMORY_BW_PASS", "", kExitPass); }
int fail(const std::string &w) { return emitMarker("MEMORY_BW_FAIL", w, kExitFail); }
int skip(const std::string &w) { return emitMarker("MEMORY_BW_SKIP", w, kExitSkip); }
int errored(const std::string &w) { return emitMarker("MEMORY_BW_ERROR", w, kExitError); }

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

// triad is STREAM's a[i] = b[i] + scalar * c[i], the standard sustained
// memory-bandwidth kernel. Grid-stride so one launch covers any buffer size and
// the occupancy does not depend on the element count.
__global__ void triad(float *a, const float *b, const float *c, float scalar, std::size_t n) {
	const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
	for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
	     i += stride) {
		a[i] = b[i] + scalar * c[i];
	}
}

// Case is one copy path measured over several passes on ONE device.
struct Case {
	const char *label;
	const char *minKey;  // acceptance: the WORST pass, folded Min across devices
	const char *maxKey;  // evidence: the best pass of device 0, not folded
};

const Case kCopyCases[] = {
    {"h2d", "h2d_bandwidth_gbs", "hostToDeviceBandwidthMaxGBs"},
    {"d2h", "d2h_bandwidth_gbs", "deviceToHostBandwidthMaxGBs"},
    {"d2d", "d2d_bandwidth_gbs", "deviceToDeviceBandwidthMaxGBs"},
};

// DeviceResult is one device's whole outcome — identity, raw values for
// device_fold.h, evidence, and this device's own exit code/reason. Mirrors
// clockprobe_rocm.cc's DeviceResult.
struct DeviceResult {
	int index = 0;
	int exitCode = kExitPass;
	std::string reason;

	bool identityRead = false;
	std::string pciAddress, name, gfxTarget;

	// Raw values for device_fold.h, keyed exactly as printed.
	std::map<std::string, double> values;

	unsigned long long deviceMemoryTotalBytes = 0;

	// Evidence carried through for device 0's own "representative" printing —
	// mirrors clockprobe_rocm.cc's DeviceResult evidence fields, none of which
	// are folded across devices.
	burnin::Samples h2d, d2h, d2d;
	long passesBelowTimerResolution = 0;
	long triadIterations = 0;
	double triadElapsedS = 0;
	double triadGBs = 0;
	bool triadMeasured = false;

	std::string configWarnings;  // merged into the process-global after this device finishes
};

// processDevice is TODAY's single-device pipeline — device discovery,
// allocation, the copy passes and the sustained triad — scoped to one HIP
// ordinal and one triad window. Unchanged in substance from the
// pre-multi-device engine; only parameterised by device index and window, and
// recording a result instead of exiting the process. Mirrors
// clockprobe_rocm.cc's runOneDevice.
void processDevice(int index, long windowSecondsTotal, long bufferMiB, long copyPasses,
                   DeviceResult *out) {
	out->index = index;

	hipError_t e = hipSetDevice(index);
	if (e != hipSuccess) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) + ": hipSetDevice: " + hipGetErrorString(e);
		return;
	}
	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	if ((e = hipGetDeviceProperties(&props, index)) != hipSuccess) {
		out->exitCode = kExitError;
		out->reason =
		    "device " + std::to_string(index) + ": hipGetDeviceProperties: " + hipGetErrorString(e);
		return;
	}
	out->name = props.name;
	out->gfxTarget = props.gcnArchName;
	// A PCI address string built directly from HIP's own report of where this
	// ordinal lives — unlike clockprobe_rocm.cc, this runner reads no sysfs
	// telemetry, so there is no separate sysfs card to correlate against.
	char pciBuf[32];
	std::snprintf(pciBuf, sizeof(pciBuf), "%04x:%02x:%02x.0", props.pciDomainID, props.pciBusID,
	             props.pciDeviceID);
	out->pciAddress = pciBuf;
	out->identityRead = true;
	// Effective GPU-visible memory. On an APU this is the GTT allocation the
	// kernel was configured for, not a VRAM size.
	out->deviceMemoryTotalBytes = static_cast<unsigned long long>(props.totalGlobalMem);

	const std::size_t bytes = static_cast<std::size_t>(bufferMiB) * 1024 * 1024;
	const std::size_t elems = bytes / sizeof(float);

	// ── allocate ─────────────────────────────────────────────────────────────
	float *hostSrc = nullptr, *hostDst = nullptr;
	float *dA = nullptr, *dB = nullptr, *dC = nullptr;
	auto cleanup = [&]() {
		if (hostSrc) (void)hipHostFree(hostSrc);
		if (hostDst) (void)hipHostFree(hostDst);
		if (dA) (void)hipFree(dA);
		if (dB) (void)hipFree(dB);
		if (dC) (void)hipFree(dC);
	};
	// setError records this DEVICE's own outcome and returns — it never exits
	// the process, so a fault on one device does not stop another device's
	// measurement (docs/dev/multi-device.md: a device is a PART).
	auto setError = [&](const char *where, hipError_t err) {
		cleanup();
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) + ": " + where + ": " + hipGetErrorString(err) +
		             "; hardware unjudged";
	};

	// PINNED host memory. Pageable host memory measures the runtime's staging
	// path rather than the link, and the difference is large enough that a
	// pageable figure compared against a pinned one reads as a fault.
	if ((e = hipHostMalloc(&hostSrc, bytes)) != hipSuccess) { setError("hipHostMalloc src", e); return; }
	if ((e = hipHostMalloc(&hostDst, bytes)) != hipSuccess) { setError("hipHostMalloc dst", e); return; }
	if ((e = hipMalloc(&dA, bytes)) != hipSuccess) { setError("hipMalloc a", e); return; }
	if ((e = hipMalloc(&dB, bytes)) != hipSuccess) { setError("hipMalloc b", e); return; }
	if ((e = hipMalloc(&dC, bytes)) != hipSuccess) { setError("hipMalloc c", e); return; }

	// A recognisable, non-uniform pattern: a buffer of zeros or of one repeated
	// value cannot distinguish a copy that worked from one that never ran.
	for (std::size_t i = 0; i < elems; i++) {
		hostSrc[i] = static_cast<float>((i % 251) + 1) * 0.5f;
	}

	const auto now = []() { return std::chrono::steady_clock::now(); };
	const auto secondsSince = [](std::chrono::steady_clock::time_point t0) {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	};

	// ── copy paths ───────────────────────────────────────────────────────────
	long refused = 0;
	for (long p = 0; p < copyPasses; p++) {
		double gbs = 0;

		auto t0 = now();
		if ((e = hipMemcpy(dA, hostSrc, bytes, hipMemcpyHostToDevice)) != hipSuccess) {
			setError("hipMemcpy h2d", e);
			return;
		}
		if ((e = hipDeviceSynchronize()) != hipSuccess) { setError("sync after h2d", e); return; }
		if (burnin::bandwidthGBs(bytes, secondsSince(t0), &gbs)) out->h2d.add(gbs);
		else refused++;

		std::memset(hostDst, 0, bytes);
		t0 = now();
		if ((e = hipMemcpy(hostDst, dA, bytes, hipMemcpyDeviceToHost)) != hipSuccess) {
			setError("hipMemcpy d2h", e);
			return;
		}
		if ((e = hipDeviceSynchronize()) != hipSuccess) { setError("sync after d2h", e); return; }
		if (burnin::bandwidthGBs(bytes, secondsSince(t0), &gbs)) out->d2h.add(gbs);
		else refused++;

		// THE ONE FAIL CONDITION for this device. Bytes that did not survive a
		// round trip are a hardware verdict about THIS device; everything else
		// in this pipeline is an Error. Recorded on this device's own result
		// rather than exiting the process — see docs/dev/multi-device.md's
		// Fail > Error > Skip > Pass precedence across devices.
		const long bad = burnin::firstMismatch(hostDst, hostSrc, elems, 0.0);
		if (bad >= 0) {
			cleanup();
			char why[256];
			std::snprintf(why, sizeof(why),
			              "device %d: data corruption: element %ld of the host->device->host round "
			              "trip came back as %g, expected %g",
			              index, bad, static_cast<double>(hostDst[bad]), static_cast<double>(hostSrc[bad]));
			out->exitCode = kExitFail;
			out->reason = why;
			return;
		}

		t0 = now();
		if ((e = hipMemcpy(dB, dA, bytes, hipMemcpyDeviceToDevice)) != hipSuccess) {
			setError("hipMemcpy d2d", e);
			return;
		}
		if ((e = hipDeviceSynchronize()) != hipSuccess) { setError("sync after d2d", e); return; }
		if (burnin::bandwidthGBs(bytes, secondsSince(t0), &gbs)) out->d2d.add(gbs);
		else refused++;
	}
	out->passesBelowTimerResolution = refused;

	// Feed device_fold.h only what this device actually measured: a case this
	// device could not time is simply absent from this device's contribution
	// (never a fabricated zero), which is what lets another device's own
	// measurement of the same path still decide the fold.
	if (out->h2d.measured()) out->values["h2d_bandwidth_gbs"] = out->h2d.min;
	if (out->d2h.measured()) out->values["d2h_bandwidth_gbs"] = out->d2h.min;
	if (out->d2d.measured()) out->values["d2d_bandwidth_gbs"] = out->d2d.min;

	// ── sustained triad ──────────────────────────────────────────────────────
	if ((e = hipMemcpy(dB, hostSrc, bytes, hipMemcpyHostToDevice)) != hipSuccess) {
		setError("hipMemcpy triad b", e);
		return;
	}
	if ((e = hipMemcpy(dC, hostSrc, bytes, hipMemcpyHostToDevice)) != hipSuccess) {
		setError("hipMemcpy triad c", e);
		return;
	}

	const float scalar = 3.0f;
	const int threads = 256;
	const int blocks = (props.multiProcessorCount > 0 ? props.multiProcessorCount : 40) * 8;

	// Warm up so the measured window excludes first-launch and page-in costs,
	// which on a GTT-backed pool are real and would understate a healthy part.
	hipLaunchKernelGGL(triad, dim3(blocks), dim3(threads), 0, nullptr, dA, dB, dC, scalar, elems);
	if ((e = hipDeviceSynchronize()) != hipSuccess) { setError("triad warmup", e); return; }

	long iterations = 0;
	const auto triadStart = now();
	while (secondsSince(triadStart) < static_cast<double>(windowSecondsTotal)) {
		hipLaunchKernelGGL(triad, dim3(blocks), dim3(threads), 0, nullptr, dA, dB, dC, scalar, elems);
		if ((e = hipGetLastError()) != hipSuccess) { setError("triad launch", e); return; }
		if ((e = hipDeviceSynchronize()) != hipSuccess) { setError("triad", e); return; }
		iterations++;
	}
	const double triadSeconds = secondsSince(triadStart);

	// Verify the triad COMPUTED, not merely ran. A kernel that was elided, or
	// one whose writes never landed, would otherwise report full bandwidth.
	if ((e = hipMemcpy(hostDst, dA, bytes, hipMemcpyDeviceToHost)) != hipSuccess) {
		setError("hipMemcpy triad result", e);
		return;
	}
	std::vector<float> want(elems);
	for (std::size_t i = 0; i < elems; i++) {
		want[i] = burnin::triadExpected(hostSrc[i], hostSrc[i], scalar);
	}
	// Exact: the inputs and the arithmetic are fp32 on both sides, and a
	// tolerance here would only mask a fault.
	const long badTriad = burnin::firstMismatch(hostDst, want.data(), elems, 0.0);
	if (badTriad >= 0) {
		cleanup();
		char why[256];
		std::snprintf(why, sizeof(why),
		              "device %d: data corruption: triad element %ld is %g, expected %g", index, badTriad,
		              static_cast<double>(hostDst[badTriad]), static_cast<double>(want[badTriad]));
		out->exitCode = kExitFail;
		out->reason = why;
		return;
	}

	out->triadIterations = iterations;
	out->triadElapsedS = triadSeconds;
	double triadGBs = 0;
	if (burnin::bandwidthGBs(burnin::triadBytes(elems, sizeof(float), iterations), triadSeconds,
	                         &triadGBs)) {
		out->triadMeasured = true;
		out->triadGBs = triadGBs;
		out->values["memory_bandwidth_gbs"] = triadGBs;
	}

	cleanup();

	// A device that measured nothing at all is not a pass on that device.
	// Every path here can legitimately be unmeasured on some hardware, but all
	// of them being unmeasured means this device produced no evidence.
	if (!out->h2d.measured() && !out->d2h.measured() && !out->d2d.measured() && !out->triadMeasured) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) +
		             ": no bandwidth figure could be measured: every interval fell under the timer's "
		             "resolution; hardware unjudged";
		return;
	}
	if (iterations == 0) {
		out->exitCode = kExitError;
		out->reason = "device " + std::to_string(index) +
		             ": the triad completed no iteration inside its window; hardware unjudged";
		return;
	}
	out->exitCode = kExitPass;
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

}  // namespace

int main() {
	std::string cfgErr;
	long durationSeconds, bufferMiB, copyPasses;
	if (!envLong("BURNIN_DURATION_SECONDS", 60, &durationSeconds, &cfgErr) ||
	    !envLong("MEMORY_BW_BUFFER_MIB", 512, &bufferMiB, &cfgErr) ||
	    !envLong("MEMORY_BW_COPY_PASSES", 10, &copyPasses, &cfgErr)) {
		return errored(cfgErr);
	}
	if (durationSeconds < 5) {
		warn("BURNIN_DURATION_SECONDS raised to the 5s floor");
		durationSeconds = 5;
	}
	if (bufferMiB < 16) {
		warn("MEMORY_BW_BUFFER_MIB raised to 16");
		bufferMiB = 16;
	}
	if (copyPasses < 2) {
		// Below two passes there is no spread, and the min/max convention that
		// makes an intermittent fault visible degenerates to a single number.
		warn("MEMORY_BW_COPY_PASSES raised to 2");
		copyPasses = 2;
	}

	const auto runStart = std::chrono::steady_clock::now();
	const auto secondsSinceRun = [&runStart]() {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - runStart).count();
	};

	std::printf("duration_requested_s=%ld\nbuffer_mib=%ld\ncopy_passes=%ld\n", durationSeconds, bufferMiB,
	            copyPasses);
	std::printf("transfer_size_bytes=%llu\n",
	            static_cast<unsigned long long>(static_cast<std::size_t>(bufferMiB) * 1024 * 1024));

	// ── how many devices, and how ─────────────────────────────────────────────
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

	// ── run every device ─────────────────────────────────────────────────────
	std::vector<DeviceResult> results(planCount);
	if (conc.mode == devices::Concurrency::All) {
		std::vector<std::thread> threads;
		threads.reserve(planCount);
		for (int i = 0; i < planCount; ++i) {
			threads.emplace_back(processDevice, i, windowS, bufferMiB, copyPasses, &results[i]);
		}
		for (auto &t : threads) t.join();
	} else {
		for (int i = 0; i < planCount; ++i) {
			processDevice(i, windowS, bufferMiB, copyPasses, &results[i]);
		}
	}

	// Merge every device's own warnings into the process global,
	// single-threaded (every worker has already joined).
	for (auto &r : results) {
		if (!r.configWarnings.empty()) warn(r.configWarnings);
	}

	// Device 0's identity, printed once — see device_fold.h's design note on
	// why identity keys keep the first device's meaning.
	if (!results.empty() && results.front().identityRead) {
		const DeviceResult &d0 = results.front();
		std::printf("gpu_name=%s\n", d0.name.c_str());
		std::printf("gfx_target=%s\n", d0.gfxTarget.c_str());
		std::printf("device_memory_total_bytes=%llu\n", d0.deviceMemoryTotalBytes);
	}

	// The peer cases the NVIDIA runner reports across a multi-GPU node. A
	// single-device ALLOCATION has no peer, so they are DECLARED UNMEASURABLE
	// rather than omitted — see the file header. Two or more devices remain an
	// unimplemented gap this conversion does not attempt to close (#402).
	if (planCount < 2) {
		std::printf("peer_read_bandwidth_gbs=n/a\npeerReadBandwidthMaxGBs=n/a\n");
		std::printf("peer_write_bandwidth_gbs=n/a\npeerWriteBandwidthMaxGBs=n/a\n");
	}

	// ── fold and report ──────────────────────────────────────────────────────
	std::vector<devices::DeviceReport> reports;
	for (auto &r : results) {
		if (r.values.empty()) continue;  // never measured (setup failed before sampling)
		reports.push_back(toDeviceReport(r));
	}
	static const std::vector<devices::FoldRule> kDeviceFold = {
	    {"h2d_bandwidth_gbs", devices::Fold::Min},
	    {"d2h_bandwidth_gbs", devices::Fold::Min},
	    {"d2d_bandwidth_gbs", devices::Fold::Min},
	    {"memory_bandwidth_gbs", devices::Fold::Min},
	};
	const devices::Folded folded = devices::fold(reports, kDeviceFold, "memory_bandwidth_gbs");

	// The three copy-path acceptance figures, folded Min across every device
	// that measured them. A path NO device could time stays exactly the n/a
	// declaration a single device made before this conversion — "the runner
	// looked and found nothing to report" — rather than an omission, which is
	// what preserves this runner's meaning on the single-device path this
	// conversion must leave unchanged.
	for (const Case &c : kCopyCases) {
		if (auto it = folded.values.find(c.minKey); it != folded.values.end()) {
			std::printf("%s=%.2f\n", c.minKey, it->second);
		} else {
			std::printf("%s=n/a\n", c.minKey);
		}
	}

	// Device 0's own max/spread evidence per copy path — not folded, mirroring
	// how clockprobe.cu treats maxSmClockPct and the rest of its per-device
	// evidence as "Once, device 0's" rather than a cross-device fold.
	if (!results.empty()) {
		const DeviceResult &d0 = results.front();
		const burnin::Samples *sets[] = {&d0.h2d, &d0.d2h, &d0.d2d};
		for (int i = 0; i < 3; i++) {
			const burnin::Samples *s = sets[i];
			if (!s->measured()) {
				std::printf("%s=n/a\n", kCopyCases[i].maxKey);
				continue;
			}
			std::printf("%s=%.2f\n%s_spread_pct=%.2f\n", kCopyCases[i].maxKey, s->max,
			            kCopyCases[i].label, s->spreadPct());
		}
		if (d0.passesBelowTimerResolution > 0) {
			// Reported rather than silently dropped: a pass whose interval was
			// under the timer's floor is a measurement that did not happen, and
			// the count is how a reader tells a thin sample set from a full one.
			std::printf("passes_below_timer_resolution=%ld\n", d0.passesBelowTimerResolution);
		}
		std::printf("triad_iterations=%ld\ntriad_elapsed_s=%.2f\n", d0.triadIterations, d0.triadElapsedS);
	}

	// The registered sustained-bandwidth metric, folded Min across devices —
	// see the file header on why it is earned by a window rather than a burst.
	// It has no Max/spread companion, so unlike kCopyCases it is printed here
	// on its own.
	if (auto it = folded.values.find("memory_bandwidth_gbs"); it != folded.values.end()) {
		std::printf("memory_bandwidth_gbs=%.2f\n", it->second);
	} else {
		std::printf("memory_bandwidth_gbs=n/a\n");
	}

	// The WHOLE run, not just one device's triad window: elapsedS is what an
	// `elapsedS >= 0.95 * durationSeconds` gate reads to decide whether a test
	// actually ran for as long as it was asked to.
	std::printf("elapsed_s=%.1f\n", secondsSinceRun());

	const std::vector<devices::SpreadSpec> spreads = {
	    {"hostToDeviceBandwidthSpreadPct", "h2d_bandwidth_gbs", /*absoluteFigure=*/true},
	};
	devices::printFold(stdout, reports, visible, windowS, conc.mode, folded, spreads, /*underMig=*/false);
	if (reports.size() > 1) {
		std::fputs(devices::renderPerDeviceArtifact(reports).c_str(), stdout);
	}

	// ── combine per-device outcomes into the machinery/verdict exit code ─────
	std::vector<int> codes;
	for (auto &r : results) codes.push_back(r.exitCode);
	const int combined = devices::combineExitCodes(codes);

	if (combined == kExitPass) return pass();

	// The message names the device(s) that decided it — a fold over several
	// devices is not a verdict about a board nobody finished measuring, and a
	// single culprit should read as one, not as "device 3: ...".
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

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
//               whole duration budget
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
// of short samples. This runner holds the triad for the duration budget it was
// given and reports what it achieved over that window, which is exactly the
// registered quantity. Emitting it from a burst would be the trap that runner's
// README describes; emitting it from a soak is the metric working as intended.
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
#include <string>
#include <vector>

#include "bw_stats.h"

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

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

// Case is one copy path measured over several passes.
struct Case {
	const char *label;
	const char *minKey;  // acceptance: the WORST pass
	const char *maxKey;  // evidence: the best pass, so a spread is visible
};

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

	int devCount = 0;
	hipError_t e = hipGetDeviceCount(&devCount);
	if (e != hipSuccess || devCount == 0) {
		return errored(std::string("no usable HIP device (") +
		               (e == hipSuccess ? "zero devices" : hipGetErrorString(e)) +
		               "); hardware unjudged");
	}
	if ((e = hipSetDevice(0)) != hipSuccess) {
		return errored(std::string("hipSetDevice: ") + hipGetErrorString(e));
	}
	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	if ((e = hipGetDeviceProperties(&props, 0)) != hipSuccess) {
		return errored(std::string("hipGetDeviceProperties: ") + hipGetErrorString(e));
	}

	std::printf("gpu_name=%s\ngfx_target=%s\ndevice_count=%d\n", props.name, props.gcnArchName,
	            devCount);
	// Effective GPU-visible memory. On an APU this is the GTT allocation the
	// kernel was configured for, not a VRAM size — it moves with
	// amdgpu.gttsize and ttm.pages_limit, so it is reported as evidence beside
	// the bandwidth figures rather than assumed from the SKU.
	std::printf("device_memory_total_bytes=%llu\n",
	            static_cast<unsigned long long>(props.totalGlobalMem));
	std::printf("duration_requested_s=%ld\nbuffer_mib=%ld\ncopy_passes=%ld\n", durationSeconds,
	            bufferMiB, copyPasses);

	const std::size_t bytes = static_cast<std::size_t>(bufferMiB) * 1024 * 1024;
	const std::size_t elems = bytes / sizeof(float);
	std::printf("transfer_size_bytes=%llu\n", static_cast<unsigned long long>(bytes));

	// The peer cases the NVIDIA runner reports across a multi-GPU node. A
	// single-accelerator node has no peer, so they are DECLARED UNMEASURABLE
	// rather than omitted: omitting the keys makes a gate on them fail closed
	// and condemn every single-GPU node, while exiting 2 would throw away the
	// four figures this run does take. A profile pairs them with
	// applicability: RequiredIfMeasurable and gets NOT EVALUATED instead.
	if (devCount < 2) {
		std::printf("peer_read_bandwidth_gbs=n/a\npeerReadBandwidthMaxGBs=n/a\n");
		std::printf("peer_write_bandwidth_gbs=n/a\npeerWriteBandwidthMaxGBs=n/a\n");
	}

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
	auto hipErr = [&](const char *where, hipError_t err) {
		cleanup();
		return errored(std::string(where) + ": " + hipGetErrorString(err) + "; hardware unjudged");
	};

	// PINNED host memory. Pageable host memory measures the runtime's staging
	// path rather than the link, and the difference is large enough that a
	// pageable figure compared against a pinned one reads as a fault.
	if ((e = hipHostMalloc(&hostSrc, bytes)) != hipSuccess) return hipErr("hipHostMalloc src", e);
	if ((e = hipHostMalloc(&hostDst, bytes)) != hipSuccess) return hipErr("hipHostMalloc dst", e);
	if ((e = hipMalloc(&dA, bytes)) != hipSuccess) return hipErr("hipMalloc a", e);
	if ((e = hipMalloc(&dB, bytes)) != hipSuccess) return hipErr("hipMalloc b", e);
	if ((e = hipMalloc(&dC, bytes)) != hipSuccess) return hipErr("hipMalloc c", e);

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
	burnin::Samples h2d, d2h, d2d;
	long refused = 0;
	for (long p = 0; p < copyPasses; p++) {
		double gbs = 0;

		auto t0 = now();
		if ((e = hipMemcpy(dA, hostSrc, bytes, hipMemcpyHostToDevice)) != hipSuccess)
			return hipErr("hipMemcpy h2d", e);
		if ((e = hipDeviceSynchronize()) != hipSuccess) return hipErr("sync after h2d", e);
		if (burnin::bandwidthGBs(bytes, secondsSince(t0), &gbs)) h2d.add(gbs);
		else refused++;

		std::memset(hostDst, 0, bytes);
		t0 = now();
		if ((e = hipMemcpy(hostDst, dA, bytes, hipMemcpyDeviceToHost)) != hipSuccess)
			return hipErr("hipMemcpy d2h", e);
		if ((e = hipDeviceSynchronize()) != hipSuccess) return hipErr("sync after d2h", e);
		if (burnin::bandwidthGBs(bytes, secondsSince(t0), &gbs)) d2h.add(gbs);
		else refused++;

		// THE ONE FAIL CONDITION. Bytes that did not survive a round trip are a
		// hardware verdict; everything else in this runner is an Error.
		const long bad = burnin::firstMismatch(hostDst, hostSrc, elems, 0.0);
		if (bad >= 0) {
			cleanup();
			char why[256];
			std::snprintf(why, sizeof(why),
			              "data corruption: element %ld of the host->device->host round trip came "
			              "back as %g, expected %g",
			              bad, static_cast<double>(hostDst[bad]), static_cast<double>(hostSrc[bad]));
			return fail(why);
		}

		t0 = now();
		if ((e = hipMemcpy(dB, dA, bytes, hipMemcpyDeviceToDevice)) != hipSuccess)
			return hipErr("hipMemcpy d2d", e);
		if ((e = hipDeviceSynchronize()) != hipSuccess) return hipErr("sync after d2d", e);
		if (burnin::bandwidthGBs(bytes, secondsSince(t0), &gbs)) d2d.add(gbs);
		else refused++;
	}
	if (refused > 0) {
		// Reported rather than silently dropped: a pass whose interval was
		// under the timer's floor is a measurement that did not happen, and the
		// count is how a reader tells a thin sample set from a full one.
		std::printf("passes_below_timer_resolution=%ld\n", refused);
	}

	const Case cases[] = {
	    {"h2d", "h2d_bandwidth_gbs", "hostToDeviceBandwidthMaxGBs"},
	    {"d2h", "d2h_bandwidth_gbs", "deviceToHostBandwidthMaxGBs"},
	    {"d2d", "d2d_bandwidth_gbs", "deviceToDeviceBandwidthMaxGBs"},
	};
	const burnin::Samples *sets[] = {&h2d, &d2h, &d2d};
	for (int i = 0; i < 3; i++) {
		const burnin::Samples *s = sets[i];
		if (!s->measured()) {
			// Declared unmeasurable, not omitted — same rule as the peer cases.
			std::printf("%s=n/a\n%s=n/a\n", cases[i].minKey, cases[i].maxKey);
			continue;
		}
		std::printf("%s=%.2f\n%s=%.2f\n%s_spread_pct=%.2f\n", cases[i].minKey, s->min,
		            cases[i].maxKey, s->max, cases[i].label, s->spreadPct());
	}

	// ── sustained triad ──────────────────────────────────────────────────────
	if ((e = hipMemcpy(dB, hostSrc, bytes, hipMemcpyHostToDevice)) != hipSuccess)
		return hipErr("hipMemcpy triad b", e);
	if ((e = hipMemcpy(dC, hostSrc, bytes, hipMemcpyHostToDevice)) != hipSuccess)
		return hipErr("hipMemcpy triad c", e);

	const float scalar = 3.0f;
	const int threads = 256;
	const int blocks = (props.multiProcessorCount > 0 ? props.multiProcessorCount : 40) * 8;

	// Warm up so the measured window excludes first-launch and page-in costs,
	// which on a GTT-backed pool are real and would understate a healthy part.
	hipLaunchKernelGGL(triad, dim3(blocks), dim3(threads), 0, nullptr, dA, dB, dC, scalar, elems);
	if ((e = hipDeviceSynchronize()) != hipSuccess) return hipErr("triad warmup", e);

	long iterations = 0;
	const auto triadStart = now();
	while (secondsSince(triadStart) < static_cast<double>(durationSeconds)) {
		hipLaunchKernelGGL(triad, dim3(blocks), dim3(threads), 0, nullptr, dA, dB, dC, scalar,
		                   elems);
		if ((e = hipGetLastError()) != hipSuccess) return hipErr("triad launch", e);
		if ((e = hipDeviceSynchronize()) != hipSuccess) return hipErr("triad", e);
		iterations++;
	}
	const double triadSeconds = secondsSince(triadStart);

	// Verify the triad COMPUTED, not merely ran. A kernel that was elided, or
	// one whose writes never landed, would otherwise report full bandwidth.
	if ((e = hipMemcpy(hostDst, dA, bytes, hipMemcpyDeviceToHost)) != hipSuccess)
		return hipErr("hipMemcpy triad result", e);
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
		              "data corruption: triad element %ld is %g, expected %g", badTriad,
		              static_cast<double>(hostDst[badTriad]), static_cast<double>(want[badTriad]));
		return fail(why);
	}

	std::printf("triad_iterations=%ld\ntriad_elapsed_s=%.2f\n", iterations, triadSeconds);
	double triadGBs = 0;
	if (burnin::bandwidthGBs(burnin::triadBytes(elems, sizeof(float), iterations), triadSeconds,
	                         &triadGBs)) {
		// The registered sustained-bandwidth metric, earned by a window rather
		// than a burst — see the header comment.
		std::printf("memory_bandwidth_gbs=%.2f\n", triadGBs);
	} else {
		std::printf("memory_bandwidth_gbs=n/a\n");
	}
	// The WHOLE run, not just the triad window: elapsedS is what an
	// `elapsedS >= 0.95 * durationSeconds` gate reads to decide whether a test
	// actually ran for as long as it was asked to, and a figure covering only
	// one phase would answer that question about the wrong interval.
	std::printf("elapsed_s=%.1f\n", secondsSince(runStart));

	cleanup();

	// A run that measured nothing at all is not a pass. Every path here can
	// legitimately be n/a on some hardware, but all of them being n/a means the
	// run produced no evidence and the node is unjudged.
	if (!h2d.measured() && !d2h.measured() && !d2d.measured() && triadGBs == 0) {
		return errored("no bandwidth figure could be measured: every interval fell under the "
		               "timer's resolution; hardware unjudged");
	}
	if (iterations == 0) {
		return errored("the triad completed no iteration inside its window; hardware unjudged");
	}
	return pass();
}

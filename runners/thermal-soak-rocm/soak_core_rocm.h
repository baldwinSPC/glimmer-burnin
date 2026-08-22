// soak_core_rocm.h — the shared soak engine for AMD accelerators.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// SHARED FILE. This header is byte-identical in runners/thermal-soak-rocm/ and
// runners/gpu-burn-rocm/, and runners/sharedsource_test.go fails if the two
// copies ever drift. It is physically duplicated because each runner is its own
// Docker build context and COPY cannot reach outside it; duplication that
// cannot be removed is duplication that has to be checked.
//
// TWO KINDS, TWO ASSERTION SETS, ONE IMPLEMENTATION
// -------------------------------------------------
// Ported from runners/thermal-soak/soak_core.cuh and keeping its central
// design, because the reasoning is vendor-independent: thermal-soak and
// gpu-burn ask different questions of the SAME event — hold a heavy,
// correctness-checked load on the part for a long time.
//
//   thermal-soak   does it stay at clock, and cool enough not to trip a
//                  protective throttle, for the whole window?
//   gpu-burn       does it still get the RIGHT ANSWER while it is that hot?
//
// Two verdicts over one experiment, not two experiments. Running them from one
// engine is what makes them comparable: the same GEMM, the same tile shape, the
// same sampler, so both kinds' sustainedThroughputTflops is the same quantity
// and a fleet can chart AMD and NVIDIA nodes as one series.
//
// MULTI-DEVICE, AND THE ONE PIECE THAT HAS NO NVIDIA PRECEDENT
// --------------------------------------------------------------
// The engine iterates every device the pod was allocated
// (docs/dev/multi-device.md), concurrent by default, mirroring the NVIDIA
// engine's DeviceCtx / lockstep-round architecture exactly (HIP's stream,
// event and query API is the same shape as CUDA's). The one piece with no
// precedent to port: the NVIDIA engine resolves its NVML handle for device N
// from CUDA's own `cudaDeviceGetPCIBusId(N)`, so the sampled device and the
// loaded device are PROVABLY the same part. sysfs has no HIP-provided
// equivalent lookup — `findCard()` originally just took "the first
// vendor-0x1002 card in /sys/class/drm", which is silently wrong the moment a
// second AMD device exists (HIP's device ordering and sysfs's directory
// ordering have no documented relationship). `findCardForDevice` below closes
// that gap the same way CUDA does: read the PCI domain/bus/device HIP reports
// for this ordinal (`hipDeviceProp_t.pciDomainID/pciBusID/pciDeviceID`),
// resolve each sysfs card's OWN PCI address by reading its `device` symlink's
// target directory name (`0000:c1:00.0`), and match on that string — never on
// enumeration order. NOT VERIFIED AGAINST REAL MULTI-GPU AMD HARDWARE: this
// project's ROCm fleet is Strix Halo, one iGPU per box, so the correlation has
// never had a second device to prove itself against. See #402.
//
// THE CORRECTNESS CHECK IS BITWISE, AND A TOLERANCE WOULD BE WRONG
// ----------------------------------------------------------------
// The first GEMM's output is kept as the reference; every later GEMM is
// compared against it BITWISE on the device, and every differing element is
// counted. The kernel's accumulation order is fixed by its schedule, so a
// healthy part recomputes the identical bit pattern every time; any difference
// at all is the hardware changing its answer between two runs of the same
// arithmetic. That is silent data corruption, invisible to every other test
// here — the part reports success, the driver logs nothing, the clocks look
// fine, and the wrong number flows into a training run. A tolerance would hide
// exactly the single-bit flips this exists to catch.
//
// The reference is computed on the part under test, which bounds what this can
// prove: it detects a part that STOPS agreeing with itself, not one that was
// wrong from the first instruction. compute-smoke-rocm is the test that asserts
// the arithmetic is right at all.
//
// WHAT AMD CANNOT MEASURE, AND WHY IT IS DECLARED RATHER THAN ZEROED
// ------------------------------------------------------------------
// Two of the NVIDIA engine's signals have no equivalent here, and both are
// emitted as the `n/a` sentinel rather than as a number:
//
//   throttleEvents  NVML publishes a throttle-REASON MASK. amdgpu's sysfs has
//                   no equivalent: it exposes clocks and temperature, from
//                   which throttling can be inferred but not enumerated. A
//                   fabricated throttle_count=0 would satisfy a
//                   `throttleEvents == 0` threshold on a node whose throttle
//                   state was never read.
//   eccErrors       gfx1151's LPDDR5X has no ECC at all. This is the case the
//                   contract's RequiredIfMeasurable applicability exists for:
//                   the hardware positively declares it has no such counter,
//                   which is different from a counter that could not be read.
//
// Everything else is emitted only when sysfs actually answered. Omitting a key
// makes a threshold on it fail closed, which is the correct direction.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, emitted PERIODICALLY during the run and always
//   before the decision, then one of
//     <KIND>_PASS               exit 0
//     <KIND>_FAIL: <reason>     exit 1
//     <KIND>_SKIP: <reason>     exit 2
//     <KIND>_ERROR: <reason>    exit 3

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "device_fold.h"
#include "kmsg/kmsg_watch.h"

namespace soak {

// The watch lives in burnin::kmsg (its own header, shared with the NVIDIA
// engines); aliased here so the uses below read the same as every other
// engine-local name. Without this alias nothing in this namespace could name
// it at all — caught in review before any of the four images ever built.
namespace kmsg = burnin::kmsg;
// Same reason, same fix, for the device-fold header.
namespace devices = burnin::devices;

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

// A soak shorter than this cannot separate a thermal ramp from a steady state.
constexpr long kMinDurationSeconds = 15;
constexpr long kDefaultDurationSeconds = 900;

// GEMM tile geometry: 256 threads compute a 64x64 output tile, 4x4 per thread.
// A GEMM rather than clockprobe's FMA chain because a soak has a different job
// — an FMA chain heats the ALUs and nothing else, while a GEMM also drives
// shared memory (LDS), the register file and the memory path, which is what
// makes it a whole-part soak rather than an ALU soak.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadTile = 4;
constexpr int kBlockDim = 16;  // 16x16 = 256 threads

constexpr int kDefaultMatrixN = 8192;
constexpr int kMinMatrixN = 512;
constexpr int kMaxMatrixN = 32768;

constexpr long kPollMillis = 2;

// Keys is a kind's stdout vocabulary. The two kinds spell the same quantities
// differently because the operator's alias tables in pkg/runner already pin
// those spellings per kind; a key outside those tables falls through to generic
// snake_case -> lowerCamelCase normalisation.
struct Keys {
	const char *markerPrefix;  // "THERMAL_SOAK" | "GPU_BURN"
	const char *elapsed;
	const char *temp;
	const char *power;
	const char *throttle;
	const char *iterations;
	const char *miscompares;
	const char *tflops;
};

// ── configuration plumbing ──────────────────────────────────────────────────

inline std::string configWarnings;

inline void warn(const std::string &s) {
	if (!configWarnings.empty()) configWarnings += "; ";
	configWarnings += s;
}

// A list of sysfs/HIP reads that could not be attributed, deduplicated — see
// the NVIDIA engine's noteUnsupported for why deduplication matters once
// there is more than one device to discover a shared gap on.
inline std::string unsupportedReads;

inline void noteUnsupported(const char *what) {
	const std::string item(what);
	size_t pos = 0;
	while (pos < unsupportedReads.size()) {
		size_t comma = unsupportedReads.find(',', pos);
		if (comma == std::string::npos) comma = unsupportedReads.size();
		if (unsupportedReads.compare(pos, comma - pos, item) == 0) return;
		pos = comma + 1;
	}
	if (!unsupportedReads.empty()) unsupportedReads += ",";
	unsupportedReads += item;
}

inline bool envLong(const char *name, long dflt, long *out, std::string *err) {
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

inline bool envDouble(const char *name, double dflt, double *out, std::string *err) {
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

inline double nowSeconds() {
	using namespace std::chrono;
	return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline void sleepMillis(long ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// ── sysfs telemetry ─────────────────────────────────────────────────────────
//
// sysfs rather than amd-smi, for the reason recorded across this whole AMD
// family: on gfx1151 the amdsmi library reports essentially every monitoring
// field as N/A while the kernel has the data the entire time (ROCm issue
// #6035). rocm-smi itself reads these files.

inline std::string sysfsRoot() {
	const char *r = std::getenv("BURNIN_SYSFS_ROOT");
	return (r != nullptr && *r != '\0') ? std::string(r) : std::string("/sys");
}

inline std::optional<std::string> readTrim(const std::filesystem::path &p) {
	std::ifstream f(p);
	if (!f) return std::nullopt;
	std::stringstream ss;
	ss << f.rdbuf();
	std::string s = ss.str();
	while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
	return s;
}

inline std::optional<double> readNumber(const std::filesystem::path &p) {
	const auto s = readTrim(p);
	if (!s || s->empty()) return std::nullopt;
	try {
		std::size_t pos = 0;
		const double v = std::stod(*s, &pos);
		if (pos == 0) return std::nullopt;
		return v;
	} catch (...) {
		return std::nullopt;
	}
}

// Card is one amdgpu sysfs card directory, resolved to a specific PCI address
// so it can be matched against a specific HIP device rather than assumed to be
// "the only one".
struct Card {
	std::filesystem::path device;
	std::filesystem::path hwmon;
	std::string pciAddress;  // "0000:c1:00.0", from the device symlink's target
	bool found = false;
};

// allCards enumerates every vendor-0x1002 card under /sys/class/drm, in name
// order for determinism — the same discovery clockprobe-rocm uses — but
// returns ALL of them rather than the first, so the caller can match by PCI
// address instead of assuming card0 is device 0.
inline std::vector<Card> allCards() {
	std::vector<Card> out;
	const auto drm = std::filesystem::path(sysfsRoot()) / "class" / "drm";
	std::error_code ec;
	std::vector<std::filesystem::path> entries;
	for (const auto &e : std::filesystem::directory_iterator(drm, ec)) entries.push_back(e.path());
	if (ec) return out;
	std::sort(entries.begin(), entries.end());
	for (const auto &p : entries) {
		const std::string name = p.filename().string();
		if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
		const auto dev = p / "device";
		const auto vendor = readTrim(dev / "vendor");
		if (!vendor || *vendor != "0x1002") continue;
		Card c;
		c.device = dev;
		c.found = true;
		// The `device` entry is a symlink into /sys/devices/.../<domain:bus:dev.fn>;
		// its own filename (not its target path) is that PCI address.
		std::error_code linkEc;
		const auto resolved = std::filesystem::canonical(dev, linkEc);
		if (!linkEc) c.pciAddress = resolved.filename().string();
		std::vector<std::filesystem::path> hwmons;
		for (const auto &h : std::filesystem::directory_iterator(dev / "hwmon", ec))
			hwmons.push_back(h.path());
		if (!hwmons.empty()) {
			std::sort(hwmons.begin(), hwmons.end());
			c.hwmon = hwmons.front();
		}
		out.push_back(c);
	}
	return out;
}

// findCardForDevice matches a HIP device's own reported PCI address against
// every card sysfs shows, the same cross-check CUDA gets from
// cudaDeviceGetPCIBusId + nvmlDeviceGetHandleByPciBusId. Returns an unfound
// Card if HIP could not report an address or nothing in sysfs matches it —
// callers treat that as "no sysfs telemetry for this device" rather than
// silently sampling the wrong one.
inline Card findCardForDevice(int domain, int bus, int device) {
	char want[16];
	std::snprintf(want, sizeof(want), "%04x:%02x:%02x.0", domain, bus, device);
	for (auto &c : allCards()) {
		if (c.pciAddress == want) return c;
	}
	return Card{};
}

inline std::optional<double> readTempC(const Card &c) {
	if (c.hwmon.empty()) return std::nullopt;
	const auto v = readNumber(c.hwmon / "temp1_input");
	if (!v) return std::nullopt;
	return *v / 1000.0;  // millidegrees
}

inline std::optional<double> readPowerW(const Card &c) {
	if (c.hwmon.empty()) return std::nullopt;
	if (const auto v = readNumber(c.hwmon / "power1_average")) return *v / 1e6;
	if (const auto v = readNumber(c.hwmon / "power1_input")) return *v / 1e6;
	return std::nullopt;
}

// readSclkMHz parses pp_dpm_sclk's active level ("1: 2900Mhz *").
//
// The ladder's TOP level is this part's rated clock — there is no NVML-style
// nameplate boost on an APU, and the driver's own declared ceiling is the
// honest denominator: a BIOS or power-mode change that lowers the ladder lowers
// the denominator with it, rather than failing the part against a ceiling it
// was never permitted to reach.
struct Sclk {
	double currentMHz = 0;
	double ratedMHz = 0;
	bool ok = false;
};

inline Sclk readSclk(const Card &c) {
	Sclk s;
	const auto text = readTrim(c.device / "pp_dpm_sclk");
	if (!text) return s;
	std::istringstream in(*text);
	std::string line;
	while (std::getline(in, line)) {
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
		if (line.empty()) continue;
		const bool active = !line.empty() && line.back() == '*';
		const auto colon = line.find(':');
		if (colon == std::string::npos) continue;
		double mhz = 0;
		try {
			mhz = std::stod(line.substr(colon + 1));
		} catch (...) {
			continue;
		}
		s.ratedMHz = std::max(s.ratedMHz, mhz);
		if (active) s.currentMHz = mhz;
		s.ok = true;
	}
	return s;
}

// ── the load ────────────────────────────────────────────────────────────────

// Deterministic fill into [0.5, 1.5): a row's products sum to something of
// order N — inside FP32's range, no denormals, no cancellation — so the
// reference is stable for reasons that have nothing to do with luck.
__global__ void fillKernel(float *m, std::size_t total, unsigned int seed) {
	const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
	for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	     i < total; i += stride) {
		unsigned int h = static_cast<unsigned int>(i) * 2654435761u + seed;
		h ^= h >> 15;
		h *= 2246822519u;
		h ^= h >> 13;
		m[i] = 0.5f + static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
	}
}

__global__ __launch_bounds__(kBlockDim *kBlockDim) void sgemmKernel(const float *__restrict__ a,
                                                                    const float *__restrict__ b,
                                                                    float *__restrict__ c, int n) {
	__shared__ float as[kTileK][kTileM];
	__shared__ float bs[kTileK][kTileN];

	const int tx = threadIdx.x, ty = threadIdx.y;
	const int row0 = blockIdx.y * kTileM + ty * kThreadTile;
	const int col0 = blockIdx.x * kTileN + tx * kThreadTile;

	float acc[kThreadTile][kThreadTile] = {};

	for (int k0 = 0; k0 < n; k0 += kTileK) {
		for (int i = 0; i < kThreadTile; i++) {
			const int r = ty * kThreadTile + i;
			const int kk = tx;
			if (kk < kTileK) {
				const int gr = blockIdx.y * kTileM + r;
				as[kk][r] = (gr < n && k0 + kk < n) ? a[gr * n + k0 + kk] : 0.0f;
			}
			const int cidx = tx * kThreadTile + i;
			const int kb = ty;
			if (kb < kTileK) {
				const int gc = blockIdx.x * kTileN + cidx;
				bs[kb][cidx] = (gc < n && k0 + kb < n) ? b[(k0 + kb) * n + gc] : 0.0f;
			}
		}
		__syncthreads();
		for (int kk = 0; kk < kTileK; kk++) {
			for (int i = 0; i < kThreadTile; i++) {
				for (int j = 0; j < kThreadTile; j++) {
					acc[i][j] += as[kk][ty * kThreadTile + i] * bs[kk][tx * kThreadTile + j];
				}
			}
		}
		__syncthreads();
	}
	for (int i = 0; i < kThreadTile; i++) {
		for (int j = 0; j < kThreadTile; j++) {
			const int r = row0 + i, cc = col0 + j;
			if (r < n && cc < n) c[r * n + cc] = acc[i][j];
		}
	}
}

// compareKernel counts BITWISE differences against the reference. Comparing the
// bit patterns rather than the floats also makes a NaN that reappears
// identically a match, which is correct: the question is whether the part
// changed its answer, not whether the answer is a number.
__global__ void compareKernel(const float *__restrict__ got, const float *__restrict__ want,
                              std::size_t total, unsigned long long *diffs,
                              unsigned long long *nonfinite) {
	const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
	for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	     i < total; i += stride) {
		unsigned int g, w;
		__builtin_memcpy(&g, &got[i], sizeof(g));
		__builtin_memcpy(&w, &want[i], sizeof(w));
		if (g != w) atomicAdd(diffs, 1ULL);
		if (!isfinite(got[i])) atomicAdd(nonfinite, 1ULL);
	}
}

// ── what the engine hands back ──────────────────────────────────────────────
//
// The fold across devices: every field is the WORST device's reading for a
// gated metric, or the SUM for a counter, or device 0's for an identity/
// wall-clock field — see kDeviceFold in thermal_soak_rocm.cc / gpu_burn_rocm.cc
// and docs/dev/multi-device.md.
struct Measurement {
	bool ok = false;
	// skip distinguishes "no accelerator visible to this container" from
	// every other reason the hardware went unjudged — the same distinction
	// device_fold.h's Plan::Skip draws and the NVIDIA engine's run() already
	// makes. The original single-device engine had no Skip path at all (every
	// failure, including zero devices, was Error); this is the honest
	// classification the shared header now makes available, not a behaviour
	// change anyone asked to keep.
	bool skip = false;
	std::string error;

	double elapsedSeconds = 0;
	long long iterations = 0;
	unsigned long long miscompares = 0;
	unsigned long long nonfinite = 0;
	bool tflopsKnown = false;
	double tflops = 0;

	bool tempKnown = false;
	double peakTempC = 0;
	double meanTempC = 0;  // device 0's; evidence, not folded

	bool powerKnown = false;
	double peakPowerW = 0;

	bool clockKnown = false;
	double ratedMHz = 0;      // device 0's; evidence, not folded
	double meanSclkMHz = 0;   // device 0's; evidence, not folded
	double minSclkPct = 100.0;  // device 0's; evidence, not folded
	double sustainedClockPct = 0;

	// xidAvailable mirrors the NVIDIA engine's own field of the same name: true
	// only when /dev/kmsg could be opened and positioned at this run's start.
	// One watch for the whole pod, not per device — see kmsg/kmsg_watch.h.
	bool xidAvailable = false;
	bool xidDropped = false;
	long xidCount = 0;
	std::string xidUnavailableReason;
};

// amdFaultPatterns is what THIS engine counts as a driver-logged GPU fault on
// AMD hardware, via kmsg::MatchesAll — every substring in one inner list must
// appear in a kernel-log line, case-insensitively, for it to count.
//
// Deliberately NARROWER than runners/host-health's own amdgpu heuristic
// (`(?i)amdgpu.*(gpu reset|ras)`), and that difference is a decision, not an
// oversight. host-health's match feeds kernelHwErrors, which is registered
// Evidence-only and never gates a node — so a false positive there costs
// nothing but a human's attention. This match feeds xidEvents, which IS
// registered Acceptance and is exactly the metric a profile is expected to
// write `xidEvents Equal 0` against. host-health's bare "ras" substring would
// match plenty of ordinary, non-fault amdgpu log lines that merely mention RAS
// being present or configured, and turning that into a hardware verdict would
// be the false-Fail this project's whole verdict philosophy exists to prevent
// ("Fail is never retried"). "reset" and "uncorrectable" are what actually
// distinguish a fault line from an informational one in AMD's own driver
// vocabulary — a GPU reset is unambiguously a recovery event, and an
// uncorrectable RAS error is unambiguously damage, whereas "RAS" alone can
// appear in a line that is merely reporting the feature is enabled.
//
// NOT VERIFIED AGAINST REAL AMD HARDWARE. This project's ROCm fleet
// (gfx1151/Strix Halo) has never produced a GPU reset or an uncorrectable RAS
// event to capture the real message shape from; these patterns are built from
// the AMDGPU kernel driver's own published log-message vocabulary rather than
// a capture. See issue references in the ROCm README's "Not yet verified"
// section.
inline const std::vector<std::vector<std::string>> &amdFaultPatterns() {
	static const std::vector<std::vector<std::string>> patterns = {
	    {"amdgpu", "reset"},
	    {"amdgpu", "uncorrectable"},
	};
	return patterns;
}

// ── per-device state ────────────────────────────────────────────────────────

// DeviceCtx is everything ONE device's run needs, from setup through teardown.
// Mirrors the NVIDIA engine's DeviceCtx; see soak_core.cuh for the fuller
// rationale of each field's presence.
struct DeviceCtx {
	int index = 0;
	bool active = false;
	int exitCode = kExitPass;
	std::string detail;

	hipDeviceProp_t props{};
	std::string gcnArchName;
	bool identityRead = false;

	Card card;  // may be `found == false`; see findCardForDevice

	int n = 0;
	std::size_t elems = 0;
	std::size_t bytes = 0;

	float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
	unsigned long long *dDiffs = nullptr, *dNonfinite = nullptr;
	hipStream_t stream = nullptr;
	hipEvent_t evStart = nullptr, evStop = nullptr;

	// Sysfs samples, taken on the sampleIntervalMs cadence.
	double tempSum = 0, tempMax = 0;
	long tempN = 0;
	double powerSum = 0, powerMax = 0;
	long powerN = 0;
	double sclkSum = 0, ratedMHz = 0, minSclkPct = 100.0;
	long sclkN = 0;
	double lastSampleAt = 0;

	unsigned long long hostDiffs = 0, hostNonfinite = 0;
	double totalGemmSeconds = 0.0;
	long long iterations = 0;

	double started = 0, deadline = 0;
	bool launched = false;
};

inline void releaseDevice(DeviceCtx *d) {
	hipSetDevice(d->index);
	if (d->dA) hipFree(d->dA);
	if (d->dB) hipFree(d->dB);
	if (d->dC) hipFree(d->dC);
	if (d->dRef) hipFree(d->dRef);
	if (d->dDiffs) hipFree(d->dDiffs);
	if (d->dNonfinite) hipFree(d->dNonfinite);
	if (d->evStart) hipEventDestroy(d->evStart);
	if (d->evStop) hipEventDestroy(d->evStop);
	if (d->stream) hipStreamDestroy(d->stream);
}

// setupDevice resolves the device's identity, its sysfs card (by PCI address —
// see the header comment), and allocates and seeds its buffers, computing and
// self-verifying the reference GEMM. On any failure it sets
// d->exitCode/detail and returns; the caller excludes this device from the
// load loop.
inline void setupDevice(DeviceCtx *d, int matrixN) {
	if (hipSetDevice(d->index) != hipSuccess) {
		d->exitCode = kExitError;
		d->detail = "hipSetDevice failed";
		return;
	}
	std::memset(&d->props, 0, sizeof(d->props));
	if (hipGetDeviceProperties(&d->props, d->index) != hipSuccess) {
		d->exitCode = kExitError;
		d->detail = "could not read HIP device properties";
		return;
	}
	d->gcnArchName = d->props.gcnArchName;
	d->identityRead = true;
	// Identity keys keep TODAY's meaning — the FIRST device's — exactly as the
	// NVIDIA engine's setupDevice does, and for the same reason: a bracket-
	// indexed pseudo-key would not match the registered metric-name grammar,
	// and every device's identity already rides in the per-device.json
	// artifact.
	if (d->index == 0) {
		std::printf("gpu_name=%s\ngfx_target=%s\n", d->props.name, d->props.gcnArchName);
	}

	d->card = findCardForDevice(d->props.pciDomainID, d->props.pciBusID, d->props.pciDeviceID);
	if (!d->card.found) {
		noteUnsupported("sysfsCard");
	}

	d->n = matrixN;
	d->elems = static_cast<std::size_t>(d->n) * static_cast<std::size_t>(d->n);
	d->bytes = d->elems * sizeof(float);
	if (d->index == 0) std::printf("matrix_n=%d\n", d->n);

	auto fail = [&](const char *where, hipError_t e) {
		d->exitCode = kExitError;
		d->detail = std::string(where) + ": " + hipGetErrorString(e);
		releaseDevice(d);
	};

	hipError_t e;
	if ((e = hipMalloc(&d->dA, d->bytes)) != hipSuccess) return fail("hipMalloc a", e);
	if ((e = hipMalloc(&d->dB, d->bytes)) != hipSuccess) return fail("hipMalloc b", e);
	if ((e = hipMalloc(&d->dC, d->bytes)) != hipSuccess) return fail("hipMalloc c", e);
	if ((e = hipMalloc(&d->dRef, d->bytes)) != hipSuccess) return fail("hipMalloc ref", e);
	if ((e = hipMalloc(&d->dDiffs, sizeof(unsigned long long))) != hipSuccess)
		return fail("hipMalloc diffs", e);
	if ((e = hipMalloc(&d->dNonfinite, sizeof(unsigned long long))) != hipSuccess)
		return fail("hipMalloc nonfinite", e);
	if ((e = hipMemset(d->dDiffs, 0, sizeof(unsigned long long))) != hipSuccess)
		return fail("hipMemset diffs", e);
	if ((e = hipMemset(d->dNonfinite, 0, sizeof(unsigned long long))) != hipSuccess)
		return fail("hipMemset nonfinite", e);
	if ((e = hipStreamCreate(&d->stream)) != hipSuccess) return fail("hipStreamCreate", e);
	if ((e = hipEventCreate(&d->evStart)) != hipSuccess) return fail("hipEventCreate start", e);
	if ((e = hipEventCreate(&d->evStop)) != hipSuccess) return fail("hipEventCreate stop", e);

	hipLaunchKernelGGL(fillKernel, dim3(256), dim3(256), 0, d->stream, d->dA, d->elems, 0x9E3779B9u);
	hipLaunchKernelGGL(fillKernel, dim3(256), dim3(256), 0, d->stream, d->dB, d->elems, 0x85EBCA6Bu);
	if ((e = hipStreamSynchronize(d->stream)) != hipSuccess) return fail("fill", e);

	const dim3 block(kBlockDim, kBlockDim);
	const dim3 grid((d->n + kTileN - 1) / kTileN, (d->n + kTileM - 1) / kTileM);
	hipLaunchKernelGGL(sgemmKernel, grid, block, 0, d->stream, d->dA, d->dB, d->dRef, d->n);
	if ((e = hipStreamSynchronize(d->stream)) != hipSuccess) return fail("reference gemm", e);

	d->active = true;
}

// takeSample reads sysfs for one device. Every read is independent — a
// missing hwmon file disables just that field, exactly as the NVIDIA engine's
// takeSample degrades per capability rather than all-or-nothing.
inline void takeSample(DeviceCtx *d) {
	if (!d->card.found) return;
	if (const auto t = readTempC(d->card)) {
		d->tempN++;
		d->tempSum += *t;
		d->tempMax = std::max(d->tempMax, *t);
	}
	if (const auto p = readPowerW(d->card)) {
		d->powerN++;
		d->powerSum += *p;
		d->powerMax = std::max(d->powerMax, *p);
	}
	const Sclk s = readSclk(d->card);
	if (s.ok && s.ratedMHz > 0 && s.currentMHz > 0) {
		d->sclkN++;
		d->sclkSum += s.currentMHz;
		d->ratedMHz = s.ratedMHz;
		d->minSclkPct = std::min(d->minSclkPct, 100.0 * s.currentMHz / s.ratedMHz);
	}
}

// buildDeviceReport mirrors the NVIDIA engine's; see soak_core.cuh.
inline devices::DeviceReport buildDeviceReport(DeviceCtx &d, const Keys &k) {
	devices::DeviceReport r;
	r.index = d.index;
	r.busId = d.card.pciAddress;
	r.name = d.props.name;
	r.computeCap = d.gcnArchName;
	r.identityRead = d.identityRead;

	auto stripEq = [](const char *s) {
		std::string x(s);
		if (!x.empty() && x.back() == '=') x.pop_back();
		return x;
	};

	if (d.iterations > 0 && d.totalGemmSeconds > 0.0) {
		const double flop =
		    2.0 * static_cast<double>(d.n) * d.n * d.n * static_cast<double>(d.iterations);
		r.values[stripEq(k.tflops)] = flop / d.totalGemmSeconds / 1e12;
	}
	if (d.sclkN > 0 && d.ratedMHz > 0) {
		r.values["sustained_clock_pct"] = 100.0 * (d.sclkSum / d.sclkN) / d.ratedMHz;
	}
	if (d.tempN > 0) r.values[stripEq(k.temp)] = d.tempMax;
	if (d.powerN > 0) r.values[stripEq(k.power)] = d.powerMax;
	r.values[stripEq(k.miscompares)] = static_cast<double>(d.hostDiffs);
	r.values["nonfinite_count"] = static_cast<double>(d.hostNonfinite);
	return r;
}

// fillMeasurementFromFold copies a fold across devices into the Measurement
// fields thermal_soak_rocm.cc / gpu_burn_rocm.cc already read — the whole
// point of folding before returning.
inline void fillMeasurementFromFold(Measurement *m, const devices::Folded &f) {
	auto get = [&](const char *key, double *out) {
		auto it = f.values.find(key);
		if (it == f.values.end()) return false;
		*out = it->second;
		return true;
	};
	double v = 0.0;
	m->tflopsKnown = get("sustained_throughput_tflops", &v) || get("tflops", &v);
	if (m->tflopsKnown) m->tflops = v;
	m->clockKnown = get("sustained_clock_pct", &v);
	if (m->clockKnown) m->sustainedClockPct = v;
	m->tempKnown = get("peak_temp_c", &v) || get("gpu_temp_c", &v);
	if (m->tempKnown) m->peakTempC = v;
	m->powerKnown = get("peak_power_w", &v) || get("power_draw_w", &v);
	if (m->powerKnown) m->peakPowerW = v;
	if (get("miscompares", &v) || get("errors", &v)) m->miscompares = static_cast<unsigned long long>(v);
	if (get("nonfinite_count", &v)) m->nonfinite = static_cast<unsigned long long>(v);
}

// ── reporting ───────────────────────────────────────────────────────────────

inline int emitMarker(const Keys &k, const char *suffix, const std::string &reason, int code) {
	if (!unsupportedReads.empty()) std::printf("nvml_unsupported=%s\n", unsupportedReads.c_str());
	if (!configWarnings.empty()) std::printf("config_warnings=%s\n", configWarnings.c_str());
	if (reason.empty()) {
		std::printf("%s%s\n", k.markerPrefix, suffix);
	} else {
		std::printf("%s%s: %s\n", k.markerPrefix, suffix, reason.c_str());
	}
	return code;
}

inline int pass(const Keys &k) { return emitMarker(k, "_PASS", "", kExitPass); }
inline int fail(const Keys &k, const std::string &w) { return emitMarker(k, "_FAIL", w, kExitFail); }
inline int skip(const Keys &k, const std::string &w) { return emitMarker(k, "_SKIP", w, kExitSkip); }
inline int errored(const Keys &k, const std::string &w) {
	return emitMarker(k, "_ERROR", w, kExitError);
}

// report prints every metric the fold actually measured, plus the
// multi-device bookkeeping and (when more than one device reported) the
// per-device.json artifact — the same shape as the NVIDIA engine's
// emitMeasurement + printFold, adapted to this engine's single-shot (not
// periodic) reporting: HIP's synchronous hipStreamSynchronize gives this
// engine no natural "still running" moment to snapshot from mid-soak the way
// CUDA's async stream query does, so unlike thermal-soak/gpu-burn this AMD
// engine reports once, at the end, exactly as it always has.
inline void report(const Keys &k, const std::vector<DeviceCtx *> &reportable,
                   const std::vector<devices::FoldRule> &foldRules, double elapsedS, int visible,
                   devices::Concurrency mode, kmsg::Watch &xidWatch, kmsg::Tally &xidTally) {
	std::vector<devices::DeviceReport> reports;
	for (DeviceCtx *d : reportable) {
		if (d->iterations == 0 && d->exitCode != kExitPass) continue;
		reports.push_back(buildDeviceReport(*d, k));
	}
	const devices::Folded folded =
	    devices::fold(reports, foldRules, foldRules.empty() ? nullptr : foldRules.front().key);

	Measurement m;
	fillMeasurementFromFold(&m, folded);
	m.elapsedSeconds = elapsedS;
	if (!reportable.empty()) {
		DeviceCtx *first = reportable.front();
		m.iterations = first->iterations;
		if (first->tempN > 0) m.meanTempC = first->tempSum / first->tempN;
		if (first->sclkN > 0) {
			m.meanSclkMHz = first->sclkSum / first->sclkN;
			m.ratedMHz = first->ratedMHz;
			m.minSclkPct = first->minSclkPct;
		}
	}

	std::printf("%s%.2f\n", k.elapsed, m.elapsedSeconds);
	std::printf("%s%lld\n", k.iterations, m.iterations);
	std::printf("%s%llu\n", k.miscompares, m.miscompares);
	std::printf("nonfinite_count=%llu\n", m.nonfinite);
	if (m.tflopsKnown) std::printf("%s%.3f\n", k.tflops, m.tflops);

	if (m.tempKnown) {
		std::printf("%s%.1f\n", k.temp, m.peakTempC);
		std::printf("mean_temp_under_load_c=%.1f\n", m.meanTempC);
	}
	if (m.powerKnown) std::printf("%s%.2f\n", k.power, m.peakPowerW);
	if (m.clockKnown) {
		std::printf("sustained_clock_pct=%.2f\n", m.sustainedClockPct);
		std::printf("sm_clock_mhz=%.0f\n", m.meanSclkMHz);
		std::printf("rated_boost_clock_mhz=%.0f\n", m.ratedMHz);
		std::printf("min_sm_clock_pct=%.2f\n", m.minSclkPct);
	}

	// amdgpu publishes no throttle-reason mask; a zero here would satisfy a
	// `throttleEvents == 0` gate on a node whose throttle state was never read.
	std::printf("%sn/a\n", k.throttle);
	// gfx1151's LPDDR5X has no ECC. The hardware positively declares the
	// absence, which is exactly what RequiredIfMeasurable is for.
	std::printf("ecc_errors=n/a\n");

	// xid_source is printed unconditionally, mirroring the NVIDIA engine
	// (soak_core.cuh) and host-health: it is the label a reader checks FIRST
	// when xid_count is absent, to tell "nothing happened" from "nothing was
	// watched". Found on real hardware alongside the NVIDIA engine's own
	// disconnection: this function previously had no access to xidWatch/
	// xidTally at all, so xid_source/xid_count/xid_windows_watched were never
	// printed here — the fields existed and were fully populated on run()'s
	// own Measurement, but nothing read them into what report() actually
	// prints. No last_xid_code here: Tally::ObserveGeneric never extracts one
	// for AMD (see kmsg/kmsg_watch.h) — amdgpu's kernel messages carry no
	// single canonical numeric field the way NVIDIA's "Xid: NN" does.
	m.xidAvailable = xidWatch.Available();
	m.xidDropped = xidWatch.Dropped();
	m.xidUnavailableReason = xidWatch.Why();
	m.xidCount = xidTally.xidCount;
	std::printf("xid_source=%s\n", m.xidAvailable ? "kmsg" : "none");
	const bool xidClean = m.xidAvailable && !m.xidDropped;
	std::printf("xid_windows_watched=%d\n", xidClean ? 1 : 0);
	if (xidClean) {
		std::printf("xid_count=%ld\n", m.xidCount);
	} else if (m.xidAvailable) {
		std::printf("xid_log_dropped=1\n");
	} else if (!m.xidUnavailableReason.empty()) {
		std::printf("xid_source_detail=%s\n", m.xidUnavailableReason.c_str());
	}

	const std::vector<devices::SpreadSpec> spreads = {
	    {"sustainedClockSpreadPct", "sustained_clock_pct", /*absoluteFigure=*/false},
	};
	devices::printFold(stdout, reports, visible, /*windowS=*/static_cast<long>(elapsedS), mode, folded,
	                   spreads, /*underMig=*/false);
	if (reports.size() > 1) {
		std::fputs(devices::renderPerDeviceArtifact(reports).c_str(), stdout);
	}
}

// ── the interleaved load loop ────────────────────────────────────────────
//
// Mirrors the NVIDIA engine's runActiveDevices exactly — see soak_core.cuh's
// comment on the same function for the full reasoning. HIP's stream/event/
// query API is the same shape as CUDA's, so the lockstep is identical:
// launch every still-running device's next iteration, poll without blocking,
// harvest a finished iteration's counters and relaunch until that device's
// own deadline.
inline void runActiveDevices(std::vector<DeviceCtx *> &active, long sampleIntervalMs) {
	const double sampleIntervalS = static_cast<double>(sampleIntervalMs) / 1000.0;

	auto launchOne = [&](DeviceCtx *d) {
		hipSetDevice(d->index);
		hipEventRecord(d->evStart, d->stream);
		const dim3 block(kBlockDim, kBlockDim);
		const dim3 grid((d->n + kTileN - 1) / kTileN, (d->n + kTileM - 1) / kTileM);
		hipLaunchKernelGGL(sgemmKernel, grid, block, 0, d->stream, d->dA, d->dB, d->dC, d->n);
		hipEventRecord(d->evStop, d->stream);
		hipLaunchKernelGGL(compareKernel, dim3(256), dim3(256), 0, d->stream, d->dC, d->dRef, d->elems,
		                   d->dDiffs, d->dNonfinite);
		const hipError_t launchErr = hipGetLastError();
		if (launchErr != hipSuccess) {
			d->exitCode = kExitError;
			d->detail = std::string("gemm launch failed: ") + hipGetErrorString(launchErr);
			d->active = false;
			return;
		}
		d->launched = true;
	};

	for (DeviceCtx *d : active) launchOne(d);

	for (;;) {
		bool anyActive = false;
		for (DeviceCtx *d : active) {
			if (!d->active) continue;
			anyActive = true;
			hipSetDevice(d->index);
			if (d->launched) {
				const hipError_t q = hipStreamQuery(d->stream);
				if (q == hipSuccess) {
					d->launched = false;
					d->iterations++;
					float ms = 0.0f;
					if (hipEventElapsedTime(&ms, d->evStart, d->evStop) == hipSuccess && ms > 0.0f) {
						d->totalGemmSeconds += ms / 1000.0;
					}
					unsigned long long diffs = 0, nonfinite = 0;
					if (hipMemcpy(&diffs, d->dDiffs, sizeof(diffs), hipMemcpyDeviceToHost) != hipSuccess ||
					    hipMemcpy(&nonfinite, d->dNonfinite, sizeof(nonfinite), hipMemcpyDeviceToHost) !=
					        hipSuccess) {
						d->exitCode = kExitError;
						d->detail = "could not read the device counter block";
						d->active = false;
						continue;
					}
					d->hostDiffs = diffs;
					d->hostNonfinite = nonfinite;
					if (nowSeconds() < d->deadline) {
						launchOne(d);
					} else {
						d->active = false;
					}
				} else if (q != hipErrorNotReady) {
					d->exitCode = kExitError;
					d->detail = std::string("gemm kernel failed: ") + hipGetErrorString(q);
					d->active = false;
				}
			}
			const double now = nowSeconds();
			if (now - d->lastSampleAt >= sampleIntervalS) {
				takeSample(d);
				d->lastSampleAt = now;
			}
		}
		if (!anyActive) break;
		sleepMillis(kPollMillis);
	}
}

// ── the engine ──────────────────────────────────────────────────────────────

// run drives the whole soak — across every device the pod was allocated — and
// returns the fold. `ok == false` means the marker is already printed and the
// caller returns the code unchanged; anything else means the caller may now
// judge the hardware from the folded fields.
inline Measurement run(const Keys &k, const std::vector<devices::FoldRule> &foldRules,
                       long durationSeconds, int matrixNStart) {
	// One watch for the whole pod, not per device — see the Measurement
	// comment. Opened before anything else, for the same reason the NVIDIA
	// engine opens its own copy first: every hipMalloc/GEMM below takes real
	// time, and the window this soak reports faults over should start as
	// close to process start as this runner can make it.
	kmsg::Watch xidWatch;
	kmsg::Tally xidTally;

	Measurement m;
	m.xidAvailable = xidWatch.Available();
	m.xidUnavailableReason = xidWatch.Why();

	int visible = 0;
	const hipError_t countErr = hipGetDeviceCount(&visible);
	if (countErr != hipSuccess && countErr != hipErrorNoDevice) {
		m.error = std::string("hipGetDeviceCount: ") + hipGetErrorString(countErr);
		return m;
	}
	if (countErr == hipErrorNoDevice) visible = 0;

	const devices::Budget budget =
	    devices::parseBudget(std::getenv("BURNIN_RESOURCE_LIMITS"), devices::amdResources());
	const devices::Plan plan = devices::planIteration(visible, budget);
	if (plan.outcome == devices::Plan::Skip) {
		m.skip = true;
		m.error = plan.message;
		return m;
	}
	if (plan.outcome == devices::Plan::Error) {
		m.error = plan.message;
		return m;
	}
	const int planCount = plan.count;

	const char *concEnv = std::getenv("BURNIN_DEVICE_CONCURRENCY");
	const devices::ConcurrencyChoice conc =
	    devices::resolveConcurrency(concEnv, devices::Concurrency::All);
	if (!conc.recognised) {
		warn(std::string("BURNIN_DEVICE_CONCURRENCY=\"") + concEnv +
		     "\" is neither \"all\" nor \"sequential\"; using this kind's default (all)");
	}
	const long windowS = devices::deviceWindowSeconds(durationSeconds, planCount, conc.mode);

	long sampleIntervalMs = 250;  // fixed; this engine has no env override today

	std::vector<DeviceCtx> ctxs(planCount);
	for (int i = 0; i < planCount; ++i) {
		ctxs[i].index = i;
		setupDevice(&ctxs[i], matrixNStart);
	}

	const double runStart = nowSeconds();

	if (conc.mode == devices::Concurrency::All) {
		std::vector<DeviceCtx *> active;
		for (auto &d : ctxs) {
			if (!d.active) continue;
			d.started = nowSeconds();
			d.deadline = d.started + windowS;
			active.push_back(&d);
		}
		if (!active.empty()) runActiveDevices(active, sampleIntervalMs);
	} else {
		for (auto &d : ctxs) {
			if (!d.active) continue;
			std::vector<DeviceCtx *> one = {&d};
			d.started = nowSeconds();
			d.deadline = d.started + windowS;
			runActiveDevices(one, sampleIntervalMs);
		}
	}

	const double finished = nowSeconds();
	xidWatch.Collect([&](const std::string &line) { xidTally.ObserveGeneric(line, amdFaultPatterns()); });
	m.xidAvailable = xidWatch.Available();
	m.xidDropped = xidWatch.Dropped();
	m.xidUnavailableReason = xidWatch.Why();
	m.xidCount = xidTally.xidCount;

	std::vector<DeviceCtx *> allPtrs;
	for (auto &d : ctxs) allPtrs.push_back(&d);
	report(k, allPtrs, foldRules, finished - runStart, visible, conc.mode, xidWatch, xidTally);

	std::vector<devices::DeviceReport> finalReports;
	for (auto &d : ctxs) {
		if (d.iterations == 0 && d.exitCode != kExitPass) continue;
		finalReports.push_back(buildDeviceReport(d, k));
	}
	const devices::Folded finalFold =
	    devices::fold(finalReports, foldRules, foldRules.empty() ? nullptr : foldRules.front().key);
	fillMeasurementFromFold(&m, finalFold);
	m.elapsedSeconds = finished - runStart;
	if (!ctxs.empty()) m.iterations = ctxs.front().iterations;

	std::vector<int> codes;
	for (auto &d : ctxs) codes.push_back(d.exitCode);
	const int combined = devices::combineExitCodes(codes);

	for (auto &d : ctxs) releaseDevice(&d);

	if (combined == kExitError) {
		std::string reasons;
		for (auto &d : ctxs) {
			if (d.exitCode == kExitError && !d.detail.empty()) {
				if (!reasons.empty()) reasons += "; ";
				reasons += "device " + std::to_string(d.index) + ": " + d.detail;
			}
		}
		m.error = reasons.empty() ? "no device could be measured" : reasons;
		return m;
	}
	if (finalReports.empty()) {
		m.error = "the soak completed no iterations on any device; nothing was measured";
		return m;
	}
	m.ok = true;
	return m;
}

}  // namespace soak

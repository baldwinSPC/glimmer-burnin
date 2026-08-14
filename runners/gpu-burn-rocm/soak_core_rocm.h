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

namespace soak {

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

// Card locates the first AMD device with sensors. Discovery mirrors
// clockprobe-rocm's: PCI vendor 0x1002, visited in name order for determinism.
struct Card {
	std::filesystem::path device;
	std::filesystem::path hwmon;
	bool found = false;
};

inline Card findCard() {
	Card c;
	const auto drm = std::filesystem::path(sysfsRoot()) / "class" / "drm";
	std::error_code ec;
	std::vector<std::filesystem::path> entries;
	for (const auto &e : std::filesystem::directory_iterator(drm, ec)) entries.push_back(e.path());
	if (ec) return c;
	std::sort(entries.begin(), entries.end());
	for (const auto &p : entries) {
		const std::string name = p.filename().string();
		if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
		const auto dev = p / "device";
		const auto vendor = readTrim(dev / "vendor");
		if (!vendor || *vendor != "0x1002") continue;
		c.device = dev;
		c.found = true;
		std::vector<std::filesystem::path> hwmons;
		for (const auto &h : std::filesystem::directory_iterator(dev / "hwmon", ec))
			hwmons.push_back(h.path());
		if (!hwmons.empty()) {
			std::sort(hwmons.begin(), hwmons.end());
			c.hwmon = hwmons.front();
		}
		return c;
	}
	return c;
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

// ── the measurement ─────────────────────────────────────────────────────────

struct Measurement {
	bool ok = false;
	std::string error;

	double elapsedSeconds = 0;
	long long iterations = 0;
	unsigned long long miscompares = 0;
	unsigned long long nonfinite = 0;
	double tflops = 0;

	bool tempKnown = false;
	double peakTempC = 0;
	double meanTempC = 0;

	bool powerKnown = false;
	double peakPowerW = 0;

	bool clockKnown = false;
	double ratedMHz = 0;
	double meanSclkMHz = 0;
	double minSclkPct = 100.0;
	double sustainedClockPct = 0;
};

// run holds the load for durationSeconds, sampling sysfs between iterations.
inline Measurement run(long durationSeconds, int matrixN) {
	Measurement m;
	const std::size_t total = static_cast<std::size_t>(matrixN) * matrixN;
	const std::size_t bytes = total * sizeof(float);

	float *a = nullptr, *b = nullptr, *c = nullptr, *ref = nullptr;
	unsigned long long *dDiffs = nullptr, *dNonfinite = nullptr;
	auto cleanup = [&]() {
		if (a) (void)hipFree(a);
		if (b) (void)hipFree(b);
		if (c) (void)hipFree(c);
		if (ref) (void)hipFree(ref);
		if (dDiffs) (void)hipFree(dDiffs);
		if (dNonfinite) (void)hipFree(dNonfinite);
	};
	auto bail = [&](const char *where, hipError_t e) {
		m.error = std::string(where) + ": " + hipGetErrorString(e);
		cleanup();
		return m;
	};

	hipError_t e;
	if ((e = hipMalloc(&a, bytes)) != hipSuccess) return bail("hipMalloc a", e);
	if ((e = hipMalloc(&b, bytes)) != hipSuccess) return bail("hipMalloc b", e);
	if ((e = hipMalloc(&c, bytes)) != hipSuccess) return bail("hipMalloc c", e);
	if ((e = hipMalloc(&ref, bytes)) != hipSuccess) return bail("hipMalloc ref", e);
	if ((e = hipMalloc(&dDiffs, sizeof(unsigned long long))) != hipSuccess)
		return bail("hipMalloc diffs", e);
	if ((e = hipMalloc(&dNonfinite, sizeof(unsigned long long))) != hipSuccess)
		return bail("hipMalloc nonfinite", e);
	if ((e = hipMemset(dDiffs, 0, sizeof(unsigned long long))) != hipSuccess)
		return bail("hipMemset diffs", e);
	if ((e = hipMemset(dNonfinite, 0, sizeof(unsigned long long))) != hipSuccess)
		return bail("hipMemset nonfinite", e);

	hipLaunchKernelGGL(fillKernel, dim3(256), dim3(256), 0, nullptr, a, total, 1u);
	hipLaunchKernelGGL(fillKernel, dim3(256), dim3(256), 0, nullptr, b, total, 2u);
	if ((e = hipDeviceSynchronize()) != hipSuccess) return bail("fill", e);

	const dim3 block(kBlockDim, kBlockDim);
	const dim3 grid((matrixN + kTileN - 1) / kTileN, (matrixN + kTileM - 1) / kTileM);

	// The reference pass.
	hipLaunchKernelGGL(sgemmKernel, grid, block, 0, nullptr, a, b, ref, matrixN);
	if ((e = hipDeviceSynchronize()) != hipSuccess) return bail("reference gemm", e);

	const Card card = findCard();
	double tempSum = 0, sclkSum = 0;
	long tempN = 0, sclkN = 0;

	const double start = nowSeconds();
	while (nowSeconds() - start < static_cast<double>(durationSeconds)) {
		hipLaunchKernelGGL(sgemmKernel, grid, block, 0, nullptr, a, b, c, matrixN);
		if ((e = hipGetLastError()) != hipSuccess) return bail("gemm launch", e);
		if ((e = hipDeviceSynchronize()) != hipSuccess) return bail("gemm", e);
		m.iterations++;

		hipLaunchKernelGGL(compareKernel, dim3(256), dim3(256), 0, nullptr, c, ref, total, dDiffs,
		                   dNonfinite);
		if ((e = hipDeviceSynchronize()) != hipSuccess) return bail("compare", e);

		if (card.found) {
			if (const auto t = readTempC(card)) {
				m.tempKnown = true;
				m.peakTempC = std::max(m.peakTempC, *t);
				tempSum += *t;
				tempN++;
			}
			if (const auto p = readPowerW(card)) {
				m.powerKnown = true;
				m.peakPowerW = std::max(m.peakPowerW, *p);
			}
			const Sclk s = readSclk(card);
			if (s.ok && s.ratedMHz > 0 && s.currentMHz > 0) {
				m.clockKnown = true;
				m.ratedMHz = s.ratedMHz;
				sclkSum += s.currentMHz;
				sclkN++;
				m.minSclkPct = std::min(m.minSclkPct, 100.0 * s.currentMHz / s.ratedMHz);
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(kPollMillis));
	}
	m.elapsedSeconds = nowSeconds() - start;

	unsigned long long diffs = 0, nonfinite = 0;
	if ((e = hipMemcpy(&diffs, dDiffs, sizeof(diffs), hipMemcpyDeviceToHost)) != hipSuccess)
		return bail("read diffs", e);
	if ((e = hipMemcpy(&nonfinite, dNonfinite, sizeof(nonfinite), hipMemcpyDeviceToHost)) !=
	    hipSuccess)
		return bail("read nonfinite", e);
	m.miscompares = diffs;
	m.nonfinite = nonfinite;

	if (tempN > 0) m.meanTempC = tempSum / static_cast<double>(tempN);
	if (sclkN > 0) {
		m.meanSclkMHz = sclkSum / static_cast<double>(sclkN);
		if (m.ratedMHz > 0) m.sustainedClockPct = 100.0 * m.meanSclkMHz / m.ratedMHz;
	}
	// 2*N^3 FLOP per GEMM.
	const double flop = 2.0 * static_cast<double>(matrixN) * matrixN * matrixN *
	                    static_cast<double>(m.iterations);
	if (m.elapsedSeconds > 0) m.tflops = flop / m.elapsedSeconds / 1e12;

	cleanup();
	m.ok = true;
	return m;
}

// ── reporting ───────────────────────────────────────────────────────────────

inline int emitMarker(const Keys &k, const char *suffix, const std::string &reason, int code) {
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

// report prints every metric the run actually measured.
//
// A key is emitted ONLY when its source answered, except the two AMD cannot
// measure at all, which are declared `n/a` — see the header comment. Omitting a
// key makes a threshold on it fail closed; fabricating a zero would let it pass.
inline void report(const Keys &k, const Measurement &m) {
	std::printf("%s%.2f\n", k.elapsed, m.elapsedSeconds);
	std::printf("%s%lld\n", k.iterations, m.iterations);
	std::printf("%s%llu\n", k.miscompares, m.miscompares);
	std::printf("nonfinite_count=%llu\n", m.nonfinite);
	std::printf("%s%.3f\n", k.tflops, m.tflops);

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
}

}  // namespace soak

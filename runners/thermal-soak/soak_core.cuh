// soak_core.cuh — the load generator, the NVML sampler and the reporting engine
// shared by the "thermal-soak" and "gpu-burn" TestKinds.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// SHARED FILE. This header is byte-identical in runners/thermal-soak/ and
// runners/gpu-burn/, and TestSharedSoakSourcesAreIdentical (in
// runners/thermal-soak/soak_contract_test.go) fails if the two copies drift. It
// is duplicated rather than referenced because the publish workflow builds each
// runner with its own directory as the Docker build context, and COPY cannot
// reach outside a build context. Edit one, copy it to the other.
//
// TWO KINDS, TWO ASSERTION SETS, ONE IMPLEMENTATION
// -------------------------------------------------
// thermal-soak and gpu-burn ask different questions of the same event: hold a
// heavy, correctness-checked load on the part for a long time.
//
//   thermal-soak   does it stay at clock, and does it stay cool enough not to
//                  trip a protective throttle, for the whole window?
//   gpu-burn       does it still get the RIGHT ANSWER while it is that hot?
//
// Those are two verdicts over one experiment, not two experiments. Running them
// from one engine is what makes them comparable: the same GEMM, the same tile
// shape, the same sampler, so gpu-burn's sustainedThroughputTflops and
// thermal-soak's are the same quantity and a fleet can chart them as one series.
// Two independently written heaters would have produced two numbers under one
// name, which is the exact failure pkg/contract's naming rules exist to prevent.
//
// Each kind supplies a Keys table (its own stdout vocabulary, since the
// operator's alias tables in pkg/runner already pin different spellings for the
// two kinds) and applies its own assertions to the Measurement this engine
// returns. Everything between those two ends is here, once.
//
// THE LOAD
// --------
// A square FP32 SGEMM, C = A x B, tiled 64x64 with a 4x4 register micro-tile,
// run back to back until the deadline. It is chosen over the register-resident
// FMA chain that runners/clockprobe uses because a soak has a different job: an
// FMA chain heats the ALUs and nothing else, while a GEMM also drives shared
// memory, the register file, the L2 and the memory controller — which is where
// a marginal part actually fails. It also has a natural correctness check.
//
// THE CORRECTNESS CHECK
// ---------------------
// A and B are filled deterministically on the device. The first GEMM's output is
// kept as the reference. Every later GEMM is compared against it BITWISE, on the
// GPU, and every element that differs is counted.
//
// Bitwise is the right comparison and a tolerance would be wrong. The kernel's
// accumulation order is fixed by its schedule, so a healthy part recomputes the
// identical bit pattern every time; any difference at all is the hardware
// changing its answer between two runs of the same arithmetic. That is silent
// data corruption, and it is invisible to every other test in the suite: the
// part reports success, the driver logs nothing, the clocks look fine, and the
// wrong number flows into a training run. A tolerance would hide exactly the
// single-bit flips this exists to catch.
//
// The reference is computed on the part under test, not on the host, which
// bounds what this can prove: it detects a part that STOPS agreeing with itself,
// not one that was wrong from the first instruction. That is deliberate — a
// host-computed FP32 reference would differ in the last bits for reasons that
// have nothing to do with hardware health, and every node in the fleet would
// fail. compute-smoke is the test that asserts the arithmetic is right at all.
//
// WHY THE COUNTERS ARE ONLY EVER EMITTED WHEN THEY WERE MEASURED
// --------------------------------------------------------------
// Every metric below is printed only if the driver actually answered. A
// fabricated throttle_count=0 would satisfy a `throttleEvents == 0` threshold on
// a node whose throttle state was never read — a missing measurement silently
// passing acceptance, which is what the fails-closed rule forbids. Omitting the
// key makes such a threshold fail instead. The single exception is ECC, where
// the hardware can positively declare it has no such counter; see EccSupport in
// nvml_dynamic.h.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, emitted PERIODICALLY during the run and always
//   before the decision, then one of
//     <KIND>_PASS               exit 0
//     <KIND>_FAIL: <reason>     exit 1
//     <KIND>_SKIP: <reason>     exit 2   not applicable to this hardware
//     <KIND>_ERROR: <reason>    exit 3   unjudged; NOT a hardware verdict

#ifndef GLIMMER_BURNIN_SOAK_CORE_CUH
#define GLIMMER_BURNIN_SOAK_CORE_CUH

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include "kmsg/kmsg_watch.h"
#include "nvml_dynamic.h"

namespace soak {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
// Any exit code that is not 0/1/2 is Error to the operator. 3 is chosen so the
// code is stable and greppable rather than whatever errno-ish value leaks out.
constexpr int kExitError = 3;

// A soak shorter than this proves nothing: the part is still coming up to clock
// and to temperature, so neither the clock floor nor the thermal question has an
// answer yet. A requested duration below the floor is raised to it and the
// adjustment is reported in config_warnings.
constexpr long kMinDurationSeconds = 15;
constexpr long kDefaultDurationSeconds = 900;

// GEMM tile geometry. 256 threads compute a 64x64 output tile, 4x4 per thread,
// over 16-deep K slices. Changing any of these changes the accumulation order,
// which changes the reference bit pattern — harmless within one run, but it
// means a runner image rebuilt with different tiling is not bit-comparable with
// an older one. That is fine: the comparison is always within a single run.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadTile = 4;
constexpr int kBlockDim = 16; // 16x16 = 256 threads

// Default problem size. Large enough that the working set (4 x N^2 floats)
// leaves cache and drives the memory system, small enough to fit any accelerator
// this suite targets; shrunk automatically if the device has less free memory.
constexpr int kDefaultMatrixN = 8192;
constexpr int kMinMatrixN = 512;
constexpr int kMaxMatrixN = 32768;

// How often the host checks whether the queued work finished. Short on purpose:
// the GPU is idle between the moment a launch completes and the moment the host
// notices, and a soak that idles a third of its wall time is not a soak. NVML
// sampling happens on its own, much slower, cadence.
constexpr long kPollMillis = 2;

// If the load is not actually executing on the device, the clock and the
// temperature we sampled are not attributable to it and the run is unjudged
// rather than failed.
constexpr double kMinCredibleUtilizationPct = 50.0;

// ── the kind-specific stdout vocabulary ─────────────────────────────────────
//
// Every key ends in '=' and is printed with "%s". That is not a style quirk: it
// means EVERY metric key either runner can emit appears somewhere in its source
// as a `key=` string literal, and soak_contract_test.go greps for exactly that
// to run the whole emitted vocabulary through the operator's real parser. Two
// keys that normalise onto one canonical name would silently discard a
// measurement (parsing is last-occurrence-wins), and that test is what catches
// it before a fleet is gated on the survivor.
struct Keys {
  const char *markerPrefix; // "THERMAL_SOAK" | "GPU_BURN"
  const char *elapsed;
  const char *temp;
  const char *power;
  const char *throttle;
  const char *iterations;
  const char *miscompares;
  const char *tflops;
};

// ── the load ────────────────────────────────────────────────────────────────

// Deterministic fill. The values land in [0.5, 1.5) so that a row's worth of
// products sums to something of order N — comfortably inside FP32's range, with
// no denormals and no cancellation, so the reference is stable for reasons that
// have nothing to do with luck.
__global__ void fillKernel(float *m, size_t total, unsigned int seed) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += stride) {
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
  // +1 pad: without it the transposed A-tile store has every fourth thread on
  // the same bank.
  __shared__ float as[kTileK][kTileM + 1];
  __shared__ float bs[kTileK][kTileN + 1];

  const int tid = threadIdx.y * kBlockDim + threadIdx.x; // 0..255
  const int blockRow = blockIdx.y * kTileM;
  const int blockCol = blockIdx.x * kTileN;

  // A tile is 64 rows x 16 columns; each thread moves one float4.
  const int aRow = tid >> 2;
  const int aCol = (tid & 3) * 4;
  // B tile is 16 rows x 64 columns; each thread moves one float4.
  const int bRow = tid >> 4;
  const int bCol = (tid & 15) * 4;

  float acc[kThreadTile][kThreadTile] = {};

  for (int k0 = 0; k0 < n; k0 += kTileK) {
    const float4 av =
        *reinterpret_cast<const float4 *>(&a[static_cast<size_t>(blockRow + aRow) * n + k0 + aCol]);
    as[aCol + 0][aRow] = av.x;
    as[aCol + 1][aRow] = av.y;
    as[aCol + 2][aRow] = av.z;
    as[aCol + 3][aRow] = av.w;

    const float4 bv =
        *reinterpret_cast<const float4 *>(&b[static_cast<size_t>(k0 + bRow) * n + blockCol + bCol]);
    bs[bRow][bCol + 0] = bv.x;
    bs[bRow][bCol + 1] = bv.y;
    bs[bRow][bCol + 2] = bv.z;
    bs[bRow][bCol + 3] = bv.w;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTileK; ++k) {
      float ar[kThreadTile];
      float br[kThreadTile];
#pragma unroll
      for (int i = 0; i < kThreadTile; ++i) ar[i] = as[k][threadIdx.y * kThreadTile + i];
#pragma unroll
      for (int j = 0; j < kThreadTile; ++j) br[j] = bs[k][threadIdx.x * kThreadTile + j];
#pragma unroll
      for (int i = 0; i < kThreadTile; ++i) {
#pragma unroll
        for (int j = 0; j < kThreadTile; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kThreadTile; ++i) {
    const int row = blockRow + threadIdx.y * kThreadTile + i;
    const float4 out = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    *reinterpret_cast<float4 *>(
        &c[static_cast<size_t>(row) * n + blockCol + threadIdx.x * kThreadTile]) = out;
  }
}

// injectKernel corrupts `count` elements of the reference, spread evenly across
// it, by flipping the lowest mantissa bit.
//
// This exists so the DETECTOR can be proven on healthy hardware. A silent-data-
// corruption check that has never once fired is an assertion nobody has tested:
// if the compare kernel were mis-launched, or the counter never read back, the
// runner would report miscompares=0 for the rest of its life and every node in
// the fleet would pass a test that was measuring nothing. Running a soak with
// BURNIN_SOAK_INJECT_MISCOMPARES=N on a known-good part and watching it fail
// with exactly N wrong elements per iteration is the only way to establish that
// the instrument works, and it costs one kernel launch that never runs in
// production.
//
// A run with injection enabled is NOT a hardware verdict, and it says so twice:
// faults_injected is emitted on every run (0 when the knob is unset, so its
// absence from a stored envelope is itself informative), and an injected run
// also carries a config_warnings entry. A verdict recorded from such a run is
// self-labelling rather than silently indistinguishable from a real fault.
__global__ void injectKernel(float *ref, size_t total, long count) {
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
    const size_t idx = (total / static_cast<size_t>(count)) * static_cast<size_t>(i);
    if (idx >= total) continue;
    ref[idx] = __uint_as_float(__float_as_uint(ref[idx]) ^ 1u);
  }
}

// Counter slots in the device-side counter block.
enum : int {
  kCounterMiscompares = 0,
  kCounterNonfinite = 1,
  kCounterFirstBadIndex = 2,
  kCounterSlots = 3,
};

// verifyKernel runs on the GPU so the comparison itself is part of the load
// rather than a host round trip that would idle the part between iterations.
//
// The atomics are on the rare path only: a healthy part takes neither branch, so
// the kernel costs one streaming read of two buffers.
__global__ void verifyKernel(const float *__restrict__ c, const float *__restrict__ ref,
                             size_t total, unsigned long long *counters) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += stride) {
    const float v = c[i];
    if (!isfinite(v)) atomicAdd(&counters[kCounterNonfinite], 1ULL);
    if (__float_as_uint(v) != __float_as_uint(ref[i])) {
      atomicAdd(&counters[kCounterMiscompares], 1ULL);
      atomicMin(&counters[kCounterFirstBadIndex], static_cast<unsigned long long>(i));
    }
  }
}

// ── small helpers ───────────────────────────────────────────────────────────

inline double nowSeconds() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

inline void sleepMillis(long ms) {
  timespec ts{ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, nullptr);
}

inline std::string configWarnings;

inline void warn(const std::string &s) {
  if (!configWarnings.empty()) configWarnings += "; ";
  configWarnings += s;
}

// A list of NVML reads the driver refused. Recorded rather than substituted:
// emitting a sentinel for an unsupported read would be a fabricated number that
// a threshold could not tell from a measured one.
inline std::string unsupportedReads;

inline void noteUnsupported(const char *what) {
  if (!unsupportedReads.empty()) unsupportedReads += ",";
  unsupportedReads += what;
}

// envLong is strict about garbage: a value we cannot parse is a configuration
// error, and guessing a default in its place would run a different test than the
// one that was asked for while reporting success.
inline bool envLong(const char *name, long dflt, long *out, std::string *err) {
  const char *v = std::getenv(name);
  *out = dflt;
  if (v == nullptr || *v == '\0') return true;
  char *end = nullptr;
  const long n = std::strtol(v, &end, 10);
  if (end == v || *end != '\0') {
    *err = std::string(name) + "=\"" + v + "\" is not an integer";
    return false;
  }
  *out = n;
  return true;
}

inline bool envDouble(const char *name, double dflt, double *out, std::string *err) {
  const char *v = std::getenv(name);
  *out = dflt;
  if (v == nullptr || *v == '\0') return true;
  char *end = nullptr;
  const double n = std::strtod(v, &end);
  if (end == v || *end != '\0' || !std::isfinite(n)) {
    *err = std::string(name) + "=\"" + v + "\" is not a number";
    return false;
  }
  *out = n;
  return true;
}

// ── throttle bookkeeping ────────────────────────────────────────────────────

struct ReasonBit {
  unsigned long long bit;
  const char *key;   // ends in '=', see Keys
  const char *label; // for the human-readable reason list
};

inline const ReasonBit kReasonBits[] = {
    {nvmlrt::kReasonGpuIdle, "gpu_idle_samples=", "gpuIdle"},
    {nvmlrt::kReasonApplicationsClocksSetting, "applications_clocks_setting_samples=",
     "applicationsClocksSetting"},
    {nvmlrt::kReasonSwPowerCap, "sw_power_cap_samples=", "swPowerCap"},
    {nvmlrt::kReasonHwSlowdown, "hw_slowdown_samples=", "hwSlowdown"},
    {nvmlrt::kReasonSyncBoost, "sync_boost_samples=", "syncBoost"},
    {nvmlrt::kReasonSwThermalSlowdown, "sw_thermal_slowdown_samples=", "swThermalSlowdown"},
    {nvmlrt::kReasonHwThermalSlowdown, "hw_thermal_slowdown_samples=", "hwThermalSlowdown"},
    {nvmlrt::kReasonHwPowerBrakeSlowdown, "hw_power_brake_samples=", "hwPowerBrake"},
    {nvmlrt::kReasonDisplayClockSetting, "display_clock_setting_samples=", "displayClockSetting"},
};
constexpr int kNumReasonBits = static_cast<int>(sizeof(kReasonBits) / sizeof(kReasonBits[0]));

// PROTECTIVE reasons: the part pulling its own clocks down to survive. These are
// what "the part throttled or tripped" means, and they are what throttleEvents
// counts — the metric a soak profile writes `throttleEvents Equal 0` against.
//
// swPowerCap is deliberately NOT in this set, and that distinction is the whole
// reason the two counters exist separately. Sitting at the board power limit
// under a heavy GEMM is what a healthy accelerator DOES; it is the power
// management working, not the part protecting itself. Counting it here would
// make `throttleEvents Equal 0` fail on every healthy node under real load,
// which is the same class of mistake as gating GB10 on a 90% sustained clock.
// It is still reported, as power_cap_events and sw_power_cap_samples, because a
// part that spends its whole soak power-capped is worth seeing.
constexpr unsigned long long kProtectiveReasons =
    nvmlrt::kReasonHwSlowdown | nvmlrt::kReasonSwThermalSlowdown |
    nvmlrt::kReasonHwThermalSlowdown | nvmlrt::kReasonHwPowerBrakeSlowdown;

constexpr unsigned long long kThermalReasons = nvmlrt::kReasonSwThermalSlowdown |
                                               nvmlrt::kReasonHwThermalSlowdown |
                                               nvmlrt::kReasonHwSlowdown;

constexpr unsigned long long kPowerCapReasons = nvmlrt::kReasonSwPowerCap;

struct Samples {
  long n = 0;

  double smSum = 0.0;
  unsigned int smMin = 0xFFFFFFFFu;
  unsigned int smMax = 0;

  long memN = 0;
  double memSum = 0.0;

  long tempN = 0;
  double tempSum = 0.0;
  double tempMax = 0.0;

  long powerN = 0;
  double powerSum = 0.0;
  double powerMax = 0.0;

  long utilN = 0;
  double utilSum = 0.0;
  double memUtilSum = 0.0;

  long reasonsN = 0;
  unsigned long long reasonMask = 0;
  long reasonCount[kNumReasonBits] = {0};
  long protectedSamples = 0;
  long throttleEvents = 0; // transitions into a protective state, not sample counts
  long powerCapEvents = 0;
  bool prevProtected = false;
  bool prevPowerCapped = false;

  double mean(double sum, long count) const { return count > 0 ? sum / count : 0.0; }
};

// Which optional NVML reads are still believed to work. A read the driver
// refuses disables that field for the rest of the run rather than being retried
// every sample; NOT_SUPPORTED does not become supported mid-soak.
struct SamplerState {
  bool mem = true;
  bool temp = true;
  bool power = true;
  bool util = true;
  bool reasons = true;
};

inline void takeSample(const nvmlrt::Library &nvml, nvmlrt::Device dev, Samples *s,
                       SamplerState *ok) {
  unsigned int sm = 0;
  if (nvml.deviceGetClockInfo(dev, nvmlrt::kClockSM, &sm) != nvmlrt::kSuccess) return;

  s->n++;
  s->smSum += sm;
  s->smMax = std::max(s->smMax, sm);
  s->smMin = std::min(s->smMin, sm);

  if (ok->temp && nvml.deviceGetTemperature != nullptr) {
    unsigned int t = 0;
    if (nvml.deviceGetTemperature(dev, nvmlrt::kTemperatureGpu, &t) == nvmlrt::kSuccess) {
      s->tempN++;
      s->tempSum += t;
      s->tempMax = std::max(s->tempMax, static_cast<double>(t));
    } else {
      ok->temp = false;
      noteUnsupported("temperature");
    }
  }

  if (ok->mem) {
    unsigned int mem = 0;
    if (nvml.deviceGetClockInfo(dev, nvmlrt::kClockMem, &mem) == nvmlrt::kSuccess) {
      s->memN++;
      s->memSum += mem;
    } else {
      ok->mem = false;
      noteUnsupported("memClock");
    }
  }

  if (ok->power && nvml.deviceGetPowerUsage != nullptr) {
    unsigned int mw = 0;
    if (nvml.deviceGetPowerUsage(dev, &mw) == nvmlrt::kSuccess) {
      const double w = mw / 1000.0;
      s->powerN++;
      s->powerSum += w;
      s->powerMax = std::max(s->powerMax, w);
    } else {
      ok->power = false;
      noteUnsupported("powerDraw");
    }
  }

  if (ok->util && nvml.deviceGetUtilizationRates != nullptr) {
    nvmlrt::Utilization u{0, 0};
    if (nvml.deviceGetUtilizationRates(dev, &u) == nvmlrt::kSuccess) {
      s->utilN++;
      s->utilSum += u.gpu;
      s->memUtilSum += u.memory;
    } else {
      ok->util = false;
      noteUnsupported("utilization");
    }
  }

  if (ok->reasons && nvml.deviceGetCurrentClocksThrottleReasons != nullptr) {
    unsigned long long mask = 0;
    if (nvml.deviceGetCurrentClocksThrottleReasons(dev, &mask) == nvmlrt::kSuccess) {
      s->reasonsN++;
      s->reasonMask |= mask;
      for (int i = 0; i < kNumReasonBits; ++i) {
        if ((mask & kReasonBits[i].bit) != 0) s->reasonCount[i]++;
      }
      const bool prot = (mask & kProtectiveReasons) != 0;
      if (prot) s->protectedSamples++;
      if (prot && !s->prevProtected) s->throttleEvents++;
      s->prevProtected = prot;

      const bool capped = (mask & kPowerCapReasons) != 0;
      if (capped && !s->prevPowerCapped) s->powerCapEvents++;
      s->prevPowerCapped = capped;
    } else {
      ok->reasons = false;
      noteUnsupported("throttleReasons");
    }
  }
}

// ── what the engine hands back ──────────────────────────────────────────────

struct Measurement {
  double elapsedS = 0.0;
  long iterations = 0;

  unsigned long long miscompares = 0;
  unsigned long long nonfinite = 0;
  // Iterations in which at least one element differed. One corrupted region
  // produces many miscompares and a single incident, so the two numbers answer
  // different questions: "how much of the result was wrong" and "how often did
  // the part go wrong".
  long sdcDetections = 0;

  bool clockKnown = false;
  double sustainedClockPct = 0.0;

  bool tempKnown = false;
  double peakTempC = 0.0;
  double meanTempC = 0.0;

  bool powerKnown = false;
  double peakPowerW = 0.0;

  bool throttleKnown = false;
  long throttleEvents = 0;
  long powerCapEvents = 0;
  unsigned long long reasonMask = 0;

  bool utilKnown = false;
  double meanUtilPct = 0.0;

  nvmlrt::EccSupport eccSupport = nvmlrt::kEccUnknown;
  bool eccKnown = false; // a real delta was computed
  unsigned long long eccErrors = 0;

  // xidAvailable mirrors host-health's xid_source: true only when /dev/kmsg
  // could be opened and positioned. xidCount and haveLastXidCode are then a
  // real measurement (possibly zero); when false, nothing below is a
  // measurement at all and must not be printed — see kmsg/kmsg_watch.h.
  bool xidAvailable = false;
  bool xidDropped = false;
  long xidCount = 0;
  bool haveLastXidCode = false;
  long lastXidCode = 0;
  std::string xidUnavailableReason;

  double sustainedTflops = 0.0;
};

// ── terminal output ─────────────────────────────────────────────────────────
//
// Markers are printed LAST, after every metric, so that a failing or erroring
// run still yields its evidence. pkg/runner takes the final non-key=value line
// as the message, which is exactly the marker.

inline int emitMarker(const Keys &k, const char *suffix, const std::string &reason, int code) {
  if (reason.empty()) {
    std::printf("%s%s\n", k.markerPrefix, suffix);
  } else {
    std::printf("%s%s: %s\n", k.markerPrefix, suffix, reason.c_str());
  }
  std::fflush(stdout);
  return code;
}

inline int pass(const Keys &k) { return emitMarker(k, "_PASS", "", kExitPass); }
inline int fail(const Keys &k, const std::string &r) { return emitMarker(k, "_FAIL", r, kExitFail); }
inline int skip(const Keys &k, const std::string &r) { return emitMarker(k, "_SKIP", r, kExitSkip); }
inline int errored(const Keys &k, const std::string &r) {
  return emitMarker(k, "_ERROR", r, kExitError);
}

inline void emitCommon() {
  if (!unsupportedReads.empty()) {
    std::printf("nvml_unsupported=%s\n", unsupportedReads.c_str());
    // Counted as well as listed, so a profile can gate on "the driver answered
    // every question we asked" without parsing a string.
    long count = 1;
    for (char ch : unsupportedReads) {
      if (ch == ',') count++;
    }
    std::printf("unsupported_reads=%ld\n", count);
  }
  if (!configWarnings.empty()) std::printf("config_warnings=%s\n", configWarnings.c_str());
  std::fflush(stdout);
}

// emitMeasurement prints the whole acceptance-relevant metric block.
//
// It is called PERIODICALLY during the soak as well as at the end, and that is
// load-bearing rather than cosmetic. Parsing is last-occurrence-wins, so a log
// truncated at any point — a pod killed at its deadline, a mid-run checkpoint
// delivery, an engineer tailing stdout — still parses into a complete, coherent
// snapshot of the soak so far instead of into nothing at all. A soak that only
// reports at the end has no evidence for the case where it never reaches the
// end, which is precisely the case worth having evidence for.
inline void emitMeasurement(const Keys &k, const Measurement &m, long seq) {
  std::printf("snapshot_seq=%ld\n", seq);
  std::printf("%s%.2f\n", k.elapsed, m.elapsedS);
  std::printf("%s%ld\n", k.iterations, m.iterations);
  std::printf("%s%llu\n", k.miscompares, m.miscompares);
  std::printf("sdc_detections=%ld\n", m.sdcDetections);
  std::printf("nonfinite_count=%llu\n", m.nonfinite);
  if (m.sustainedTflops > 0.0) std::printf("%s%.3f\n", k.tflops, m.sustainedTflops);
  if (m.clockKnown) std::printf("sustained_clock_pct=%.2f\n", m.sustainedClockPct);
  if (m.tempKnown) {
    std::printf("%s%.1f\n", k.temp, m.peakTempC);
    std::printf("mean_temp_c=%.1f\n", m.meanTempC);
  }
  if (m.powerKnown) std::printf("%s%.2f\n", k.power, m.peakPowerW);
  if (m.throttleKnown) {
    std::printf("%s%ld\n", k.throttle, m.throttleEvents);
    std::printf("power_cap_events=%ld\n", m.powerCapEvents);
  }
  // ECC is the one counter a runner may declare UNMEASURABLE rather than omit,
  // and only when the part itself said it has no ECC subsystem. See EccSupport.
  if (m.eccSupport == nvmlrt::kEccAbsent) {
    std::printf("ecc_errors=n/a\n");
  } else if (m.eccKnown) {
    std::printf("ecc_errors=%llu\n", m.eccErrors);
  }
  // xid_source is printed unconditionally, exactly as host-health does: it is
  // the label a reader checks FIRST when xid_count is absent, to tell "nothing
  // happened" from "nothing was watched". xid_count and last_xid_code are only
  // ever printed when xid_source=kmsg — see kmsg/kmsg_watch.h's header comment
  // for why a failed probe must never fall back to printing a zero.
  std::printf("xid_source=%s\n", m.xidAvailable ? "kmsg" : "none");
  if (m.xidAvailable) {
    // Both are genuine, POSITIVE measurements even at zero — this probe
    // watched the whole window and nothing NVRM:Xid-shaped appeared in it,
    // which is exactly as reportable as throttle_count=0 is when the driver
    // answered and had nothing to report. last_xid_code=0 matches the
    // "0 when none is reported" convention dcgm-diag's own lastXidCode
    // already established for this metric name.
    std::printf("xid_count=%ld\n", m.xidCount);
    std::printf("last_xid_code=%ld\n", m.lastXidCode);
    if (m.xidDropped) {
      // The ring buffer wrapped faster than a drain could keep up, or a drain
      // was cut off at its read bound. xid_count above is then a FLOOR, not a
      // complete count — worth knowing before trusting it, same as
      // host-health's xid_log_dropped.
      std::printf("xid_log_dropped=1\n");
    }
  } else if (!m.xidUnavailableReason.empty()) {
    std::printf("xid_source_detail=%s\n", m.xidUnavailableReason.c_str());
  }
  std::fflush(stdout);
}

// ── the engine ──────────────────────────────────────────────────────────────

// readEccTotal sums the volatile corrected and uncorrected ECC counters.
// Returns false if either half could not be read: a total assembled from one
// half would understate the damage while looking like a complete answer.
inline bool readEccTotal(const nvmlrt::Library &nvml, nvmlrt::Device dev,
                         unsigned long long *out) {
  if (nvml.deviceGetTotalEccErrors == nullptr) return false;
  unsigned long long corrected = 0, uncorrected = 0;
  if (nvml.deviceGetTotalEccErrors(dev, nvmlrt::kMemoryErrorTypeCorrected, nvmlrt::kVolatileEcc,
                                   &corrected) != nvmlrt::kSuccess) {
    return false;
  }
  if (nvml.deviceGetTotalEccErrors(dev, nvmlrt::kMemoryErrorTypeUncorrected, nvmlrt::kVolatileEcc,
                                   &uncorrected) != nvmlrt::kSuccess) {
    return false;
  }
  *out = corrected + uncorrected;
  return true;
}

// Run drives the whole soak and fills *m.
//
// It returns 0 when the machinery worked and the caller may now judge the
// hardware. Any other return is a process exit code and the marker has ALREADY
// been printed — the caller returns it unchanged.
inline int run(const Keys &k, Measurement *m) {
  // ── /dev/kmsg, opened before anything else in this function ────────────────
  //
  // Positioning happens here, deliberately before config parsing and every
  // CUDA/NVML call below: the window this soak reports Xids over should start
  // as close to process start as this runner can make it, and none of the
  // setup between here and the load loop is instant. See kmsg/kmsg_watch.h's
  // header comment for the full design — this is opt-in evidence, never a
  // hardware verdict on its own, and every field below stays at its "probe
  // never ran" default if the mount was not granted.
  kmsg::Watch xidWatch;
  kmsg::Tally xidTally;
  m->xidAvailable = xidWatch.Available();
  m->xidUnavailableReason = xidWatch.Why();

  // ── configuration ─────────────────────────────────────────────────────────
  std::string cfgErr;
  long durationSeconds = 0, sampleIntervalMs = 0, progressIntervalS = 0, matrixN = 0;
  long injectMiscompares = 0;
  if (!envLong("BURNIN_DURATION_SECONDS", kDefaultDurationSeconds, &durationSeconds, &cfgErr) ||
      !envLong("BURNIN_SOAK_SAMPLE_INTERVAL_MS", 250, &sampleIntervalMs, &cfgErr) ||
      !envLong("BURNIN_SOAK_PROGRESS_INTERVAL_S", 15, &progressIntervalS, &cfgErr) ||
      !envLong("BURNIN_SOAK_MATRIX_N", kDefaultMatrixN, &matrixN, &cfgErr) ||
      !envLong("BURNIN_SOAK_INJECT_MISCOMPARES", 0, &injectMiscompares, &cfgErr)) {
    return errored(k, cfgErr);
  }
  if (injectMiscompares < 0) injectMiscompares = 0;
  const long durationRequested = durationSeconds;
  if (durationSeconds < kMinDurationSeconds) {
    warn("BURNIN_DURATION_SECONDS raised to the " + std::to_string(kMinDurationSeconds) +
         "s floor; a shorter soak cannot reach a steady thermal or clock state");
    durationSeconds = kMinDurationSeconds;
  }
  sampleIntervalMs = std::max(10L, std::min(sampleIntervalMs, 5000L));
  progressIntervalS = std::max(5L, std::min(progressIntervalS, 3600L));

  // Warm-up: an unwarmed part sits at idle clock and idle temperature
  // legitimately, so the measurement window must not begin until the load has
  // had time to bring both up. Correctness IS checked during warm-up — a wrong
  // answer is a wrong answer whatever the part's temperature.
  long warmupSeconds = std::max(5L, std::min(30L, durationSeconds / 10));
  if (warmupSeconds >= durationSeconds) warmupSeconds = durationSeconds / 3;

  // ── device ────────────────────────────────────────────────────────────────
  int deviceCount = 0;
  const cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
  if (countErr == cudaErrorNoDevice || (countErr == cudaSuccess && deviceCount == 0)) {
    // Not applicable rather than broken: a node with no accelerator has nothing
    // to soak.
    return skip(k, "no accelerator visible to this container");
  }
  if (countErr != cudaSuccess) {
    return errored(k, std::string("cudaGetDeviceCount: ") + cudaGetErrorString(countErr));
  }

  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    return errored(k, "could not read CUDA device properties");
  }
  char busId[32] = {0};
  const bool busIdOK = cudaDeviceGetPCIBusId(busId, sizeof(busId), device) == cudaSuccess;

  std::printf("gpu_name=%s\n", props.name);
  std::printf("compute_cap=%d.%d\n", props.major, props.minor);
  if (busIdOK) std::printf("pci_bus_id=%s\n", busId);
  std::printf("duration_requested_s=%ld\n", durationRequested);
  std::printf("warmup_s=%ld\n", warmupSeconds);
  std::printf("sample_interval_ms=%ld\n", sampleIntervalMs);
  std::printf("progress_interval_s=%ld\n", progressIntervalS);

  // ── problem size ──────────────────────────────────────────────────────────
  // Rounded DOWN to the tile size: a partial tile would read past the end of a
  // row. Shrunk to fit free device memory, because a soak that cannot allocate
  // is not a statement about the hardware's health.
  if (matrixN < kMinMatrixN || matrixN > kMaxMatrixN) {
    warn("BURNIN_SOAK_MATRIX_N clamped into [" + std::to_string(kMinMatrixN) + "," +
         std::to_string(kMaxMatrixN) + "]");
    matrixN = std::max(static_cast<long>(kMinMatrixN), std::min(matrixN, static_cast<long>(kMaxMatrixN)));
  }
  matrixN -= matrixN % kTileM;

  size_t freeBytes = 0, totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
    // Four N x N float buffers: A, B, C and the reference.
    while (matrixN > kMinMatrixN &&
           4.0 * static_cast<double>(matrixN) * matrixN * sizeof(float) > 0.8 * freeBytes) {
      matrixN /= 2;
      matrixN -= matrixN % kTileM;
    }
  }
  const int n = static_cast<int>(matrixN);
  const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
  const size_t bytes = elems * sizeof(float);
  std::printf("matrix_n=%d\n", n);

  // ── NVML ──────────────────────────────────────────────────────────────────
  // An accelerator IS present at this point. If its management library is
  // missing, the container was built or run without the "utility" driver
  // capability — a misconfiguration, not a property of the hardware. Skipping
  // here would quietly report "not applicable" for every node in a fleet whose
  // toolkit was set up wrong, so this is an Error: unjudged, and visible.
  nvmlrt::Library nvml;
  std::string nvmlErr;
  if (!nvml.open(&nvmlErr)) {
    emitCommon();
    return errored(k, nvmlErr +
                          " (an accelerator is visible, so this is a container/driver "
                          "misconfiguration: NVIDIA_DRIVER_CAPABILITIES must include 'utility')");
  }
  nvmlrt::Return rc = nvml.init();
  if (rc != nvmlrt::kSuccess) {
    emitCommon();
    nvml.close();
    return errored(k, std::string("nvmlInit: ") + nvml.errorString(rc));
  }

  nvmlrt::Device nvdev = nullptr;
  bool haveHandle = false;
  // Prefer the PCI bus id: NVML index order and CUDA ordinal order are not
  // guaranteed to agree, and sampling a different GPU than the one under load
  // would produce a confident, wrong verdict.
  if (busIdOK && nvml.deviceGetHandleByPciBusId != nullptr) {
    haveHandle = nvml.deviceGetHandleByPciBusId(busId, &nvdev) == nvmlrt::kSuccess;
  }
  if (!haveHandle) {
    warn("NVML handle resolved by index, not PCI bus id");
    rc = nvml.deviceGetHandleByIndex(0, &nvdev);
    if (rc != nvmlrt::kSuccess) {
      emitCommon();
      nvml.close();
      return errored(k, std::string("nvmlDeviceGetHandleByIndex: ") + nvml.errorString(rc));
    }
  }

  if (nvml.systemGetDriverVersion != nullptr) {
    char driver[96] = {0};
    if (nvml.systemGetDriverVersion(driver, sizeof(driver)) == nvmlrt::kSuccess) {
      std::printf("driver_version=%s\n", driver);
    }
  }

  // The denominator for sustainedClockPct. Without a rated clock there is no
  // portable percentage; the soak still runs and still judges correctness and
  // throttling, and the clock metric is simply omitted so a threshold on it
  // fails closed.
  unsigned int ratedBoostMHz = 0;
  bool ratedKnown = false;
  if (nvml.deviceGetMaxClockInfo(nvdev, nvmlrt::kClockSM, &ratedBoostMHz) == nvmlrt::kSuccess &&
      ratedBoostMHz > 0) {
    ratedKnown = true;
    // Emitted in lowerCamelCase, not as rated_boost_clock_mhz. pkg/runner passes
    // an already-camelCase key through unchanged, whereas "..._mhz" would
    // normalise to "...Mhz" — which is NOT the registered "MHz" unit suffix, so
    // the clock would be stored as a dimensionless number. Only the clockprobe
    // kind has alias-table entries for the snake_case spelling.
    std::printf("ratedBoostClockMHz=%u\n", ratedBoostMHz);
  } else {
    noteUnsupported("ratedBoostClock");
  }

  // MIG: clocks and temperature are whole-device properties, so nothing sampled
  // inside a MIG instance can be attributed to this instance's load.
  if (nvml.deviceGetMigMode != nullptr) {
    unsigned int current = 0, pending = 0;
    if (nvml.deviceGetMigMode(nvdev, &current, &pending) == nvmlrt::kSuccess) {
      std::printf("mig_mode=%s\n", current == nvmlrt::kMigEnable ? "enabled" : "disabled");
      if (current == nvmlrt::kMigEnable) {
        emitCommon();
        nvml.close();
        return skip(k, "MIG is enabled; clock, temperature and power are device-wide properties "
                       "and cannot be attributed to one instance's load");
      }
    }
  }

  // ── ECC support, and the start reading ────────────────────────────────────
  unsigned long long eccStart = 0;
  bool eccStartKnown = false;
  if (nvml.deviceGetEccMode != nullptr) {
    unsigned int current = 0, pending = 0;
    const nvmlrt::Return ercc = nvml.deviceGetEccMode(nvdev, &current, &pending);
    if (ercc == nvmlrt::kSuccess) {
      m->eccSupport = nvmlrt::kEccPresent;
      std::printf("ecc_mode=%s\n", current == nvmlrt::kFeatureEnabled ? "enabled" : "disabled");
      eccStartKnown = readEccTotal(nvml, nvdev, &eccStart);
      if (!eccStartKnown) noteUnsupported("eccCounters");
    } else if (ercc == nvmlrt::kErrorNotSupported) {
      // The part says it has no ECC subsystem. This — and only this — is the
      // state in which the counter may be DECLARED unmeasurable.
      m->eccSupport = nvmlrt::kEccAbsent;
      std::printf("ecc_mode=unsupported\n");
    } else {
      m->eccSupport = nvmlrt::kEccUnknown;
      noteUnsupported("eccMode");
    }
  } else {
    noteUnsupported("eccMode");
  }

  // ── allocate and seed ─────────────────────────────────────────────────────
  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
  unsigned long long *dCounters = nullptr;
  auto releaseAll = [&]() {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dRef);
    cudaFree(dCounters);
  };
  if (cudaMalloc(&dA, bytes) != cudaSuccess || cudaMalloc(&dB, bytes) != cudaSuccess ||
      cudaMalloc(&dC, bytes) != cudaSuccess || cudaMalloc(&dRef, bytes) != cudaSuccess ||
      cudaMalloc(&dCounters, kCounterSlots * sizeof(unsigned long long)) != cudaSuccess) {
    releaseAll();
    emitCommon();
    nvml.close();
    return errored(k, "could not allocate " + std::to_string(4 * (bytes >> 20)) +
                          " MiB of device memory for the soak buffers");
  }

  unsigned long long hostCounters[kCounterSlots] = {0, 0, ~0ULL};
  if (cudaMemcpy(dCounters, hostCounters, sizeof(hostCounters), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    releaseAll();
    emitCommon();
    nvml.close();
    return errored(k, "could not initialise the device counter block");
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t evStart = nullptr, evStop = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess || cudaEventCreate(&evStart) != cudaSuccess ||
      cudaEventCreate(&evStop) != cudaSuccess) {
    releaseAll();
    emitCommon();
    nvml.close();
    return errored(k, "could not create CUDA stream/events");
  }

  const int fillBlocks = std::min(65535, std::max(1, props.multiProcessorCount * 32));
  fillKernel<<<fillBlocks, 256, 0, stream>>>(dA, elems, 0x9E3779B9u);
  fillKernel<<<fillBlocks, 256, 0, stream>>>(dB, elems, 0x85EBCA6Bu);
  cudaError_t setupErr = cudaGetLastError();
  if (setupErr == cudaSuccess) setupErr = cudaStreamSynchronize(stream);
  if (setupErr != cudaSuccess) {
    releaseAll();
    emitCommon();
    nvml.close();
    return errored(k, std::string("operand fill failed: ") + cudaGetErrorString(setupErr));
  }

  const dim3 gemmGrid(n / kTileN, n / kTileM);
  const dim3 gemmBlock(kBlockDim, kBlockDim);
  const int verifyBlocks = std::min(65535, std::max(1, props.multiProcessorCount * 32));
  const double flopsPerGemm = 2.0 * static_cast<double>(n) * n * n;

  // The reference. This first launch also pays for CUDA context creation and
  // module load, so it is deliberately outside the timed window.
  sgemmKernel<<<gemmGrid, gemmBlock, 0, stream>>>(dA, dB, dRef, n);
  setupErr = cudaGetLastError();
  if (setupErr == cudaSuccess) setupErr = cudaStreamSynchronize(stream);
  if (setupErr != cudaSuccess) {
    releaseAll();
    emitCommon();
    nvml.close();
    return errored(k, std::string("reference GEMM failed: ") + cudaGetErrorString(setupErr));
  }
  // The reference is checked against itself: miscompares must be zero by
  // construction, but a non-finite value in it is a real result about the
  // hardware and is counted like any other.
  verifyKernel<<<verifyBlocks, 256, 0, stream>>>(dRef, dRef, elems, dCounters);
  setupErr = cudaGetLastError();
  if (setupErr == cudaSuccess) setupErr = cudaStreamSynchronize(stream);
  if (setupErr != cudaSuccess) {
    releaseAll();
    emitCommon();
    nvml.close();
    return errored(k, std::string("reference verify failed: ") + cudaGetErrorString(setupErr));
  }

  // Fault injection, if it was asked for. Deliberately AFTER the reference has
  // been verified against itself, so the injected damage is attributed to the
  // comparison and not to the reference GEMM.
  if (injectMiscompares > static_cast<long>(elems)) injectMiscompares = static_cast<long>(elems);
  std::printf("faults_injected=%ld\n", injectMiscompares);
  if (injectMiscompares > 0) {
    warn("BURNIN_SOAK_INJECT_MISCOMPARES=" + std::to_string(injectMiscompares) +
         " corrupted the reference on purpose; this run is a self-test of the detector and is "
         "NOT a verdict about this hardware");
    injectKernel<<<256, 256, 0, stream>>>(dRef, elems, injectMiscompares);
    setupErr = cudaGetLastError();
    if (setupErr == cudaSuccess) setupErr = cudaStreamSynchronize(stream);
    if (setupErr != cudaSuccess) {
      releaseAll();
      emitCommon();
      nvml.close();
      return errored(k, std::string("fault injection failed: ") + cudaGetErrorString(setupErr));
    }
  }

  // ── the soak ──────────────────────────────────────────────────────────────
  Samples s;
  SamplerState ok;
  double totalFlops = 0.0, totalGemmSeconds = 0.0;
  long seq = 0;
  long sdcDetections = 0;
  unsigned long long prevMiscompares = 0;
  std::string runErr;

  const double started = nowSeconds();
  const double warmupUntil = started + warmupSeconds;
  const double deadline = started + durationSeconds;
  double lastSample = 0.0;
  double lastProgress = started;

  auto snapshot = [&](Measurement *out, double atSeconds) {
    out->elapsedS = atSeconds - started;
    out->iterations = m->iterations;
    out->miscompares = hostCounters[kCounterMiscompares];
    out->nonfinite = hostCounters[kCounterNonfinite];
    out->sdcDetections = sdcDetections;
    out->eccSupport = m->eccSupport;
    out->eccKnown = false;
    out->eccErrors = 0;
    if (m->eccSupport == nvmlrt::kEccPresent && eccStartKnown) {
      unsigned long long now = 0;
      if (readEccTotal(nvml, nvdev, &now)) {
        out->eccKnown = true;
        out->eccErrors = now > eccStart ? now - eccStart : 0;
      }
    }
    if (s.n > 0 && ratedKnown) {
      out->clockKnown = true;
      out->sustainedClockPct = 100.0 * (s.smSum / s.n) / ratedBoostMHz;
    }
    if (s.tempN > 0) {
      out->tempKnown = true;
      out->peakTempC = s.tempMax;
      out->meanTempC = s.mean(s.tempSum, s.tempN);
    }
    if (s.powerN > 0) {
      out->powerKnown = true;
      out->peakPowerW = s.powerMax;
    }
    if (s.reasonsN > 0) {
      out->throttleKnown = true;
      out->throttleEvents = s.throttleEvents;
      out->powerCapEvents = s.powerCapEvents;
      out->reasonMask = s.reasonMask;
    }
    if (s.utilN > 0) {
      out->utilKnown = true;
      out->meanUtilPct = s.mean(s.utilSum, s.utilN);
    }
    if (totalGemmSeconds > 0.0) out->sustainedTflops = totalFlops / totalGemmSeconds / 1e12;

    // xidWatch/xidTally are shared, cumulative, run-lifetime state — see their
    // construction at the top of this function. `out` may be a fresh, local
    // Measurement (the periodic path constructs one per call), so every field
    // below is copied in rather than assumed to already be set on `out`.
    out->xidAvailable = xidWatch.Available();
    out->xidDropped = xidWatch.Dropped();
    out->xidUnavailableReason = xidWatch.Why();
    out->xidCount = xidTally.xidCount;
    out->haveLastXidCode = xidTally.haveLastXidCode;
    out->lastXidCode = xidTally.lastXidCode;
  };

  while (nowSeconds() < deadline) {
    const bool measuring = nowSeconds() >= warmupUntil;

    cudaEventRecord(evStart, stream);
    sgemmKernel<<<gemmGrid, gemmBlock, 0, stream>>>(dA, dB, dC, n);
    cudaEventRecord(evStop, stream);
    verifyKernel<<<verifyBlocks, 256, 0, stream>>>(dC, dRef, elems, dCounters);
    // A launch-configuration error is reported synchronously by
    // cudaGetLastError, not by the later synchronise. Checking only the
    // synchronise would let a kernel that never ran look like a completed
    // iteration, and the idle clock that followed would be blamed on the part.
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
      runErr = std::string("soak launch failed: ") + cudaGetErrorString(launchErr);
      break;
    }

    for (;;) {
      const cudaError_t q = cudaStreamQuery(stream);
      if (q == cudaSuccess) break;
      if (q != cudaErrorNotReady) {
        runErr = std::string("soak kernel failed: ") + cudaGetErrorString(q);
        break;
      }
      const double now = nowSeconds();
      if (measuring && (now - lastSample) * 1000.0 >= sampleIntervalMs) {
        takeSample(nvml, nvdev, &s, &ok);
        lastSample = now;
      }
      sleepMillis(kPollMillis);
    }
    if (!runErr.empty()) break;

    m->iterations++;
    if (measuring) {
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, evStart, evStop) == cudaSuccess && ms > 0.0f) {
        totalFlops += flopsPerGemm;
        totalGemmSeconds += ms / 1000.0;
      }
    }
    if (cudaMemcpy(hostCounters, dCounters, sizeof(hostCounters), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      runErr = "could not read the device counter block";
      break;
    }
    if (hostCounters[kCounterMiscompares] > prevMiscompares) {
      sdcDetections++;
      prevMiscompares = hostCounters[kCounterMiscompares];
    }

    const double now = nowSeconds();
    if (now - lastProgress >= progressIntervalS) {
      // Drained on the SAME cadence as every other periodic metric, and for
      // the same reason: a pod killed between two progress prints still has an
      // xidCount current as of the last one it completed, rather than nothing
      // at all. xidTally ACCUMULATES across calls — see kmsg/kmsg_watch.h —
      // so this adds only what arrived since the previous drain.
      xidWatch.Collect([&](const std::string &line) { xidTally.ObserveNvidia(line); });
      Measurement snap;
      snapshot(&snap, now);
      emitMeasurement(k, snap, ++seq);
      lastProgress = now;
    }
  }

  const double finished = nowSeconds();
  xidWatch.Collect([&](const std::string &line) { xidTally.ObserveNvidia(line); });
  snapshot(m, finished);

  // ── report, then let the caller judge ─────────────────────────────────────
  std::printf("samples_taken=%ld\n", s.n);
  std::printf("gemm_active_s=%.2f\n", totalGemmSeconds);
  if (s.n > 0) {
    if (ratedKnown) {
      std::printf("smClockMHz=%.0f\n", s.smSum / s.n);
      std::printf("min_sm_clock_pct=%.2f\n", 100.0 * s.smMin / ratedBoostMHz);
      std::printf("max_sm_clock_pct=%.2f\n", 100.0 * s.smMax / ratedBoostMHz);
    }
    if (s.memN > 0) std::printf("memClockMHz=%.0f\n", s.memSum / s.memN);
    if (s.powerN > 0) std::printf("mean_power_w=%.2f\n", s.mean(s.powerSum, s.powerN));
    if (s.utilN > 0) {
      std::printf("gpu_utilization_pct=%.2f\n", s.mean(s.utilSum, s.utilN));
      std::printf("mem_utilization_pct=%.2f\n", s.mean(s.memUtilSum, s.utilN));
    }
    if (s.reasonsN > 0) {
      std::printf("throttled_samples=%ld\n", s.protectedSamples);
      std::printf("throttle_reasons_mask=%llu\n", s.reasonMask);
      std::string labels;
      for (int i = 0; i < kNumReasonBits; ++i) {
        std::printf("%s%ld\n", kReasonBits[i].key, s.reasonCount[i]);
        if ((s.reasonMask & kReasonBits[i].bit) != 0) {
          if (!labels.empty()) labels += ",";
          labels += kReasonBits[i].label;
        }
      }
      std::printf("throttle_reasons=%s\n", labels.empty() ? "none" : labels.c_str());
      std::printf("thermal_throttle_latched=%s\n",
                  (s.reasonMask & kThermalReasons) != 0 ? "true" : "false");
    }
  }
  if (hostCounters[kCounterFirstBadIndex] != ~0ULL) {
    std::printf("first_miscompare_index=%llu\n", hostCounters[kCounterFirstBadIndex]);
  }
  emitMeasurement(k, *m, ++seq);
  emitCommon();

  releaseAll();
  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);
  cudaStreamDestroy(stream);
  nvml.close();

  if (!runErr.empty()) return errored(k, runErr);
  if (m->iterations == 0) {
    return errored(k, "the soak completed no iterations; nothing was measured");
  }
  // A load that did not land makes every sample unattributable. Refusing to
  // judge is the honest answer; calling it a hardware failure would condemn a
  // node for the runner's own problem.
  if (m->utilKnown && m->meanUtilPct < kMinCredibleUtilizationPct) {
    return errored(k, "device utilization averaged " +
                          std::to_string(static_cast<int>(m->meanUtilPct)) +
                          "% under load; the sampled clock and temperature are not attributable "
                          "to this test");
  }
  return 0;
}

} // namespace soak

#endif // GLIMMER_BURNIN_SOAK_CORE_CUH

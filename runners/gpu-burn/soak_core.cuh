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
// two kinds) and a device-fold table (docs/dev/multi-device.md) and applies its
// own assertions to the Measurement this engine returns. Everything between
// those two ends is here, once.
//
// MULTI-DEVICE
// ------------
// The engine iterates every device the pod was allocated
// (docs/dev/multi-device.md), not device 0. The soak family's default is
// CONCURRENT: every device runs the load at once for the whole window, because
// the measurement IS the board — a part that holds clock alone but throttles
// beside seven busy neighbours will throttle in production, and a soak sliced
// into eight consecutive duration/8 windows is not a soak. The gated Measurement
// fields below (sustainedClockPct, peakTempC, throttleEvents, miscompares,
// nonfinite, eccErrors) are the FOLD across devices — the worst reading, per the
// registry's Aggregation for each name — so gpu_burn.cu and thermal_soak.cu's
// assertions did not have to change at all: they already read m.miscompares,
// m.sustainedClockPct and so on, and those fields now mean "across every device"
// rather than "on device 0" without their call sites knowing the difference.
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
// nvml_dynamic.h. Multi-device keeps the same discipline one level up: a device
// that never reported a key is simply absent from that key's fold (device_fold.h
// treats an absent contribution as an omission, never a zero) and a device
// excluded entirely (setup failed, or it errored mid-run) contributes to no fold
// at all — see combineExitCodes below for what its exclusion does to the exit
// code.
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
#include <vector>

#include "device_fold.h"
#include "kmsg/kmsg_watch.h"
#include "nvml_dynamic.h"

namespace soak {

// The watch lives in burnin::kmsg (its own header, shared with the ROCm
// engines); aliased here so the uses below read the same as every other
// engine-local name. Without this alias nothing in this namespace could name
// it at all — caught in review before any of the four images ever built.
namespace kmsg = burnin::kmsg;
// Same reason, same fix, for the device-fold header: an unaliased `devices::`
// below would be undeclared inside this namespace and every one of the four
// soak images would fail to build.
namespace devices = burnin::devices;

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

// A list of NVML reads the driver refused, deduplicated. Recorded rather than
// substituted: emitting a sentinel for an unsupported read would be a
// fabricated number that a threshold could not tell from a measured one. On a
// multi-device node every device usually shares one driver, so the same gap is
// discovered once per device; deduplication is what keeps this a capability
// statement about the driver rather than a repeated line per device.
inline std::string unsupportedReads;

inline void noteUnsupported(const char *what) {
  const std::string item(what);
  // A cheap linear scan: this list holds at most a handful of NVML capability
  // names, never one per device.
  size_t pos = 0;
  while (pos < unsupportedReads.size()) {
    size_t comma = unsupportedReads.find(',', pos);
    if (comma == std::string::npos) comma = unsupportedReads.size();
    if (unsupportedReads.compare(pos, comma - pos, item) == 0) return; // already noted
    pos = comma + 1;
  }
  if (!unsupportedReads.empty()) unsupportedReads += ",";
  unsupportedReads += item;
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
//
// The fold across devices: every field below is the WORST device's reading for
// a gated metric, or the SUM for a windowed counter, or the first device's for
// an identity/wall-clock field — see the FoldRule direction each kind declares
// (kDeviceFold in thermal_soak.cu / gpu_burn.cu) and docs/dev/multi-device.md.
// A single-device node folds a list of one, so every field means exactly what
// it always has on such a node.
struct Measurement {
  double elapsedS = 0.0;
  long iterations = 0;

  unsigned long long miscompares = 0;
  unsigned long long nonfinite = 0;
  // Iterations in which at least one element differed. One corrupted region
  // produces many miscompares and a single incident, so the two numbers answer
  // different questions: "how much of the result was wrong" and "how often did
  // the part go wrong". Device 0's count (Once): not part of the registry's
  // gated vocabulary, so it is evidence about one representative device rather
  // than a fold.
  long sdcDetections = 0;

  bool clockKnown = false;
  double sustainedClockPct = 0.0;

  bool tempKnown = false;
  double peakTempC = 0.0;
  double meanTempC = 0.0; // device 0's mean; evidence, not folded

  bool powerKnown = false;
  double peakPowerW = 0.0;

  bool throttleKnown = false;
  long throttleEvents = 0;
  long powerCapEvents = 0; // device 0's; evidence, not folded
  unsigned long long reasonMask = 0; // device 0's; evidence, not folded

  bool utilKnown = false;
  double meanUtilPct = 0.0; // device 0's; evidence, not folded

  nvmlrt::EccSupport eccSupport = nvmlrt::kEccUnknown;
  bool eccKnown = false; // a real delta was computed on at least one device
  unsigned long long eccErrors = 0;

  bool sustainedTflopsKnown = false;
  double sustainedTflops = 0.0;

  // xidAvailable mirrors host-health's xid_source: true only when /dev/kmsg
  // could be opened and positioned. xidCount and haveLastXidCode are then a
  // real measurement (possibly zero); when false, nothing below is a
  // measurement at all and must not be printed — see kmsg/kmsg_watch.h. This
  // watch is process-wide, not per device: /dev/kmsg is one kernel log for the
  // whole node, so it is opened once regardless of deviceCount.
  bool xidAvailable = false;
  bool xidDropped = false;
  long xidCount = 0;
  bool haveLastXidCode = false;
  long lastXidCode = 0;
  std::string xidUnavailableReason;
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
  if (m.sustainedTflopsKnown) std::printf("%s%.3f\n", k.tflops, m.sustainedTflops);
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
  // happened" from "nothing was watched".
  std::printf("xid_source=%s\n", m.xidAvailable ? "kmsg" : "none");
  // xid_windows_watched is the coverage self-report: 1 when this window was
  // watched START TO FINISH, 0 otherwise — always printed, because both values
  // are positively established facts about the PROBE (like faults_injected),
  // not measurements of the hardware. It exists for the segmented case:
  // foldMetrics keeps a metric if ANY segment reported it, so a soak whose
  // segment 300 lost its watch still sums the other segments' honest zeros
  // into an xidEvents the gate would accept — while this key, Sum-aggregated,
  // then reads 671 of 672, and a profile pairing `xidEvents Equal 0` with
  // `xidWindowsWatched GreaterThanOrEqual <segments>` certifies only a soak
  // every window of which was actually observed.
  const bool xidClean = m.xidAvailable && !m.xidDropped;
  std::printf("xid_windows_watched=%d\n", xidClean ? 1 : 0);
  if (xidClean) {
    // A genuine, POSITIVE measurement even at zero — this probe watched the
    // whole window and nothing NVRM:Xid-shaped appeared in it, which is
    // exactly as reportable as throttle_count=0 is when the driver answered
    // and had nothing to report.
    std::printf("xid_count=%ld\n", m.xidCount);
    // The code only when a code was actually EXTRACTED. dcgm-diag prints 0
    // for "none" because its device register genuinely reads 0; this runner
    // has no register, so a printed 0 would be a value nobody measured —
    // both for a quiet window (nothing to say) and for the nastier case of a
    // counted Xid whose line shape defeated extraction, where 0 would
    // positively claim "none" about an event the same result counts.
    if (m.haveLastXidCode) std::printf("last_xid_code=%ld\n", m.lastXidCode);
  } else if (m.xidAvailable) {
    // The watch ran but a drain wrapped (EPIPE), errored, or was cut at its
    // bound: the tally is a FLOOR over a window with unread gaps, which is
    // not a measurement a threshold could honestly be written against —
    // host-health discards and omits in exactly this case, and so does this.
    // What was positively established still goes on the record: the drop
    // itself, and any code that was extracted before it.
    std::printf("xid_log_dropped=1\n");
    if (m.haveLastXidCode) std::printf("last_xid_code=%ld\n", m.lastXidCode);
  } else if (!m.xidUnavailableReason.empty()) {
    std::printf("xid_source_detail=%s\n", m.xidUnavailableReason.c_str());
  }
  std::fflush(stdout);
}

// ── per-device state ────────────────────────────────────────────────────────

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

// DeviceCtx is everything ONE device's run needs, from setup through teardown.
// One is constructed per index the plan says to iterate; a device that fails
// setup stays `active = false` for the rest of the run but keeps whatever it
// managed to read, so a partial identity can still feed homogeneity and the
// per-device artifact.
struct DeviceCtx {
  int index = 0;
  bool active = false;   // still iterating; false once finished, skipped, or errored
  int exitCode = kExitPass; // this device's own outcome: 0 ran cleanly, 2 skip, 3 error
  std::string detail;    // why, for combineExitCodes' caller to explain

  cudaDeviceProp props{};
  std::string busId;
  bool busIdOk = false;
  std::string computeCap;
  bool identityRead = false;

  nvmlrt::Device nvdev = nullptr;
  bool haveHandle = false;

  bool ratedKnown = false;
  unsigned int ratedBoostMHz = 0;

  nvmlrt::EccSupport eccSupport = nvmlrt::kEccUnknown;
  bool eccStartKnown = false;
  unsigned long long eccStart = 0;

  int n = 0;
  size_t elems = 0;
  size_t bytes = 0;
  double flopsPerGemm = 0.0;

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
  unsigned long long *dCounters = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t evStart = nullptr, evStop = nullptr;

  Samples s;
  SamplerState samplerOk;

  double totalFlops = 0.0;
  double totalGemmSeconds = 0.0;
  long iterations = 0;
  long sdcDetections = 0;
  unsigned long long prevMiscompares = 0;
  unsigned long long hostCounters[kCounterSlots] = {0, 0, ~0ULL};

  double started = 0, warmupUntil = 0, deadline = 0, lastSampleAt = 0;
  bool launched = false;        // a kernel is currently in flight for this device
  bool measuringThisLaunch = false;
};

// releaseDevice frees whatever setupDevice allocated, tolerating partial setup
// (any pointer may still be null).
inline void releaseDevice(DeviceCtx *d) {
  cudaSetDevice(d->index);
  cudaFree(d->dA);
  cudaFree(d->dB);
  cudaFree(d->dC);
  cudaFree(d->dRef);
  cudaFree(d->dCounters);
  if (d->evStart != nullptr) cudaEventDestroy(d->evStart);
  if (d->evStop != nullptr) cudaEventDestroy(d->evStop);
  if (d->stream != nullptr) cudaStreamDestroy(d->stream);
}

// setupDevice resolves the device's identity, its NVML handle, its ECC start
// reading, and allocates and seeds its buffers, computing and self-verifying
// the reference GEMM. On any failure it sets d->exitCode/detail and returns —
// the caller excludes this device from the load loop but keeps whatever was
// read. This is the single-device engine's old top half, unchanged in what it
// checks, now scoped to one device among several instead of the whole process.
inline void setupDevice(DeviceCtx *d, int matrixNStart, long injectMiscompares, const nvmlrt::Library &nvml) {
  if (cudaSetDevice(d->index) != cudaSuccess ||
      cudaGetDeviceProperties(&d->props, d->index) != cudaSuccess) {
    d->exitCode = kExitError;
    d->detail = "could not read CUDA device properties";
    return;
  }
  char busId[32] = {0};
  d->busIdOk = cudaDeviceGetPCIBusId(busId, sizeof(busId), d->index) == cudaSuccess;
  if (d->busIdOk) d->busId = busId;
  char cc[16];
  std::snprintf(cc, sizeof(cc), "%d.%d", d->props.major, d->props.minor);
  d->computeCap = cc;
  d->identityRead = d->busIdOk; // gpu_name is always readable from props; bus id is the gate

  // Identity keys keep TODAY's meaning — the FIRST device's — per
  // docs/dev/multi-device.md's verdict semantics: a bracket-indexed pseudo-key
  // would not match the registered `gpu_name`/`compute_cap`/`pci_bus_id`
  // names at all (the metric-name grammar has no room for one), and every
  // device's identity already rides in the per-device.json artifact. Device 0
  // alone gets the wire-compatible, unindexed keys a single-device fleet has
  // always parsed.
  if (d->index == 0) {
    std::printf("gpu_name=%s\n", d->props.name);
    std::printf("compute_cap=%s\n", d->computeCap.c_str());
    if (d->busIdOk) std::printf("pci_bus_id=%s\n", d->busId.c_str());
  }

  // Resolve the NVML handle for THIS device. Prefer the PCI bus id: NVML index
  // order and CUDA ordinal order are not guaranteed to agree, and sampling a
  // different GPU than the one under load would produce a confident, wrong
  // verdict for both of them.
  if (d->busIdOk && nvml.deviceGetHandleByPciBusId != nullptr) {
    d->haveHandle = nvml.deviceGetHandleByPciBusId(d->busId.c_str(), &d->nvdev) == nvmlrt::kSuccess;
  }
  if (!d->haveHandle) {
    warn("device " + std::to_string(d->index) + ": NVML handle resolved by index, not PCI bus id");
    d->haveHandle = nvml.deviceGetHandleByIndex(static_cast<unsigned int>(d->index), &d->nvdev) ==
                    nvmlrt::kSuccess;
  }
  if (!d->haveHandle) {
    d->exitCode = kExitError;
    d->detail = "could not resolve an NVML handle for this device";
    return;
  }

  // The denominator for sustainedClockPct. Without a rated clock there is no
  // portable percentage; the device still runs and still judges correctness and
  // throttling, and the clock metric is simply omitted for it.
  if (nvml.deviceGetMaxClockInfo(d->nvdev, nvmlrt::kClockSM, &d->ratedBoostMHz) == nvmlrt::kSuccess &&
      d->ratedBoostMHz > 0) {
    d->ratedKnown = true;
    // Emitted in lowerCamelCase, not rated_boost_clock_mhz — see the
    // single-device engine's original comment: "..._mhz" would normalise to
    // "...Mhz", not the registered "MHz" suffix. Device 0 only; see above.
    if (d->index == 0) std::printf("ratedBoostClockMHz=%u\n", d->ratedBoostMHz);
  } else {
    noteUnsupported("ratedBoostClock");
  }

  // MIG: clocks and temperature are whole-device properties, so nothing sampled
  // inside a MIG instance can be attributed to this instance's load. This
  // device is Skip, not the whole test — a board mixing MIG-enabled and
  // MIG-disabled parts (unusual, but not impossible) still measures the ones it
  // can.
  if (nvml.deviceGetMigMode != nullptr) {
    unsigned int current = 0, pending = 0;
    if (nvml.deviceGetMigMode(d->nvdev, &current, &pending) == nvmlrt::kSuccess) {
      // The registered `migMode` label, device 0 only — see the identity-key
      // comment above. The SKIP decision below still applies per device
      // regardless of which device's label made it to stdout.
      if (d->index == 0) {
        std::printf("mig_mode=%s\n", current == nvmlrt::kMigEnable ? "enabled" : "disabled");
      }
      if (current == nvmlrt::kMigEnable) {
        d->exitCode = kExitSkip;
        d->detail = "MIG is enabled; clock, temperature and power are device-wide properties and "
                    "cannot be attributed to one instance's load";
        return;
      }
    }
  }

  // ECC support, and the start reading.
  if (nvml.deviceGetEccMode != nullptr) {
    unsigned int current = 0, pending = 0;
    const nvmlrt::Return ercc = nvml.deviceGetEccMode(d->nvdev, &current, &pending);
    if (ercc == nvmlrt::kSuccess) {
      d->eccSupport = nvmlrt::kEccPresent;
      if (d->index == 0) {
        std::printf("ecc_mode=%s\n", current == nvmlrt::kFeatureEnabled ? "enabled" : "disabled");
      }
      d->eccStartKnown = readEccTotal(nvml, d->nvdev, &d->eccStart);
      if (!d->eccStartKnown) noteUnsupported("eccCounters");
    } else if (ercc == nvmlrt::kErrorNotSupported) {
      // The part says it has no ECC subsystem. This — and only this — is the
      // state in which the counter may be DECLARED unmeasurable.
      d->eccSupport = nvmlrt::kEccAbsent;
      if (d->index == 0) std::printf("ecc_mode=unsupported\n");
    } else {
      d->eccSupport = nvmlrt::kEccUnknown;
      noteUnsupported("eccMode");
    }
  } else {
    noteUnsupported("eccMode");
  }

  // Problem size, clamped to free memory on THIS device — a heterogeneous board
  // may see different N per device, which is fine: TFLOPS already normalises
  // for problem size and the correctness check needs no cross-device agreement.
  long matrixN = matrixNStart;
  matrixN -= matrixN % kTileM;
  size_t freeBytes = 0, totalBytes = 0;
  if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) {
    while (matrixN > kMinMatrixN &&
           4.0 * static_cast<double>(matrixN) * matrixN * sizeof(float) > 0.8 * freeBytes) {
      matrixN /= 2;
      matrixN -= matrixN % kTileM;
    }
  }
  d->n = static_cast<int>(matrixN);
  d->elems = static_cast<size_t>(d->n) * static_cast<size_t>(d->n);
  d->bytes = d->elems * sizeof(float);
  if (d->index == 0) std::printf("matrix_n=%d\n", d->n); // informational; device 0 only

  auto fail = [&](const char *msg) {
    d->exitCode = kExitError;
    d->detail = msg;
    releaseDevice(d);
  };

  if (cudaMalloc(&d->dA, d->bytes) != cudaSuccess || cudaMalloc(&d->dB, d->bytes) != cudaSuccess ||
      cudaMalloc(&d->dC, d->bytes) != cudaSuccess || cudaMalloc(&d->dRef, d->bytes) != cudaSuccess ||
      cudaMalloc(&d->dCounters, kCounterSlots * sizeof(unsigned long long)) != cudaSuccess) {
    fail("could not allocate device memory for the soak buffers");
    return;
  }
  if (cudaMemcpy(d->dCounters, d->hostCounters, sizeof(d->hostCounters), cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    fail("could not initialise the device counter block");
    return;
  }
  if (cudaStreamCreate(&d->stream) != cudaSuccess || cudaEventCreate(&d->evStart) != cudaSuccess ||
      cudaEventCreate(&d->evStop) != cudaSuccess) {
    fail("could not create CUDA stream/events");
    return;
  }

  const int fillBlocks = std::min(65535, std::max(1, d->props.multiProcessorCount * 32));
  fillKernel<<<fillBlocks, 256, 0, d->stream>>>(d->dA, d->elems, 0x9E3779B9u);
  fillKernel<<<fillBlocks, 256, 0, d->stream>>>(d->dB, d->elems, 0x85EBCA6Bu);
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(d->stream);
  if (e != cudaSuccess) {
    fail((std::string("operand fill failed: ") + cudaGetErrorString(e)).c_str());
    return;
  }

  const dim3 gemmGrid(d->n / kTileN, d->n / kTileM);
  const dim3 gemmBlock(kBlockDim, kBlockDim);
  const int verifyBlocks = std::min(65535, std::max(1, d->props.multiProcessorCount * 32));
  d->flopsPerGemm = 2.0 * static_cast<double>(d->n) * d->n * d->n;

  // The reference. This first launch also pays for CUDA context creation and
  // module load, so it is deliberately outside the timed window.
  sgemmKernel<<<gemmGrid, gemmBlock, 0, d->stream>>>(d->dA, d->dB, d->dRef, d->n);
  e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(d->stream);
  if (e != cudaSuccess) {
    fail((std::string("reference GEMM failed: ") + cudaGetErrorString(e)).c_str());
    return;
  }
  // The reference is checked against itself: miscompares must be zero by
  // construction, but a non-finite value in it is a real result about the
  // hardware and is counted like any other.
  verifyKernel<<<verifyBlocks, 256, 0, d->stream>>>(d->dRef, d->dRef, d->elems, d->dCounters);
  e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaStreamSynchronize(d->stream);
  if (e != cudaSuccess) {
    fail((std::string("reference verify failed: ") + cudaGetErrorString(e)).c_str());
    return;
  }

  // Fault injection, if it was asked for. Deliberately AFTER the reference has
  // been verified against itself, so the injected damage is attributed to the
  // comparison and not to the reference GEMM. Clamped to THIS device's elems —
  // a heterogeneous board, or free memory shrinking one device's matrix, can
  // give devices different element counts.
  long inject = injectMiscompares;
  if (inject > static_cast<long>(d->elems)) inject = static_cast<long>(d->elems);
  if (inject > 0) {
    injectKernel<<<256, 256, 0, d->stream>>>(d->dRef, d->elems, inject);
    e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaStreamSynchronize(d->stream);
    if (e != cudaSuccess) {
      fail((std::string("fault injection failed: ") + cudaGetErrorString(e)).c_str());
      return;
    }
  }

  d->active = true;
}

// buildDeviceReport turns one device's cumulative state into the raw
// key/value map the fold reads. Keys are stripped of the trailing '=' every
// Keys field carries, so what is folded is exactly what emitDeviceMeasurement
// prints for that device — the same discipline the single-device engine
// always had, now applied per device instead of once.
inline devices::DeviceReport buildDeviceReport(DeviceCtx &d, const Keys &k, const nvmlrt::Library &nvml) {
  devices::DeviceReport r;
  r.index = d.index;
  r.busId = d.busId;
  r.name = d.props.name;
  r.computeCap = d.computeCap;
  r.identityRead = d.identityRead;

  auto stripEq = [](const char *s) {
    std::string x(s);
    if (!x.empty() && x.back() == '=') x.pop_back();
    return x;
  };

  if (d.totalGemmSeconds > 0.0) r.values[stripEq(k.tflops)] = d.totalFlops / d.totalGemmSeconds / 1e12;
  if (d.s.n > 0 && d.ratedKnown) {
    r.values["sustained_clock_pct"] = 100.0 * (d.s.smSum / d.s.n) / d.ratedBoostMHz;
  }
  if (d.s.tempN > 0) r.values[stripEq(k.temp)] = d.s.tempMax;
  if (d.s.powerN > 0) r.values[stripEq(k.power)] = d.s.powerMax;
  if (d.s.reasonsN > 0) r.values[stripEq(k.throttle)] = static_cast<double>(d.s.throttleEvents);
  r.values[stripEq(k.miscompares)] = static_cast<double>(d.hostCounters[kCounterMiscompares]);
  r.values["nonfinite_count"] = static_cast<double>(d.hostCounters[kCounterNonfinite]);
  if (d.eccSupport == nvmlrt::kEccPresent && d.eccStartKnown) {
    unsigned long long now = 0;
    if (readEccTotal(nvml, d.nvdev, &now)) {
      r.values["ecc_errors"] = static_cast<double>(now > d.eccStart ? now - d.eccStart : 0);
    }
  }
  return r;
}

// fillMeasurementFromFold copies a fold across devices into the Measurement
// fields gpu_burn.cu and thermal_soak.cu already read — this is the whole
// point of folding before returning: their assertions did not have to learn
// devices exist.
inline void fillMeasurementFromFold(Measurement *m, const devices::Folded &f) {
  auto get = [&](const char *key, double *out) {
    auto it = f.values.find(key);
    if (it == f.values.end()) return false;
    *out = it->second;
    return true;
  };
  double v = 0.0;
  m->sustainedTflopsKnown = get("sustained_throughput_tflops", &v) || get("tflops", &v);
  if (m->sustainedTflopsKnown) m->sustainedTflops = v;
  m->clockKnown = get("sustained_clock_pct", &v);
  if (m->clockKnown) m->sustainedClockPct = v;
  m->tempKnown = get("peak_temp_c", &v) || get("gpu_temp_c", &v);
  if (m->tempKnown) m->peakTempC = v;
  m->powerKnown = get("peak_power_w", &v) || get("power_draw_w", &v);
  if (m->powerKnown) m->peakPowerW = v;
  m->throttleKnown = get("throttle_count", &v) || get("throttle_events", &v);
  if (m->throttleKnown) m->throttleEvents = static_cast<long>(v);
  if (get("miscompares", &v) || get("errors", &v)) m->miscompares = static_cast<unsigned long long>(v);
  if (get("nonfinite_count", &v)) m->nonfinite = static_cast<unsigned long long>(v);
  if (get("ecc_errors", &v)) {
    m->eccKnown = true;
    m->eccErrors = static_cast<unsigned long long>(v);
  }
}

// emitDeviceMeasurementImpl folds `reportable` and prints the same
// acceptance block a single-device run always has, PLUS the multi-device
// bookkeeping (device_count, worst device, homogeneity, spreads) — then, on
// the FINAL call only, the per-device.json artifact.
//
// `reportable` is deliberately NOT "every device the plan admitted": in
// sequential mode a device that has not started yet has an empty values map,
// and folding it in would make `spreadOf` see fewer reporting devices than
// total and declare every spread Omitted ("a probe failed") for a device that
// simply has not run yet — a false alarm on every periodic snapshot of a
// sequential soak. The caller passes exactly the devices that have produced
// at least one iteration so far, in mode-appropriate order.
inline void emitDeviceMeasurementImpl(const Keys &k, const std::vector<DeviceCtx *> &reportable,
                                      const std::vector<devices::FoldRule> &foldRules,
                                      const nvmlrt::Library &nvml, long *seq, double elapsedS,
                                      int visible, long windowS, devices::Concurrency mode,
                                      bool underMig, bool final) {
  std::vector<devices::DeviceReport> reports;
  reports.reserve(reportable.size());
  for (DeviceCtx *d : reportable) {
    if (d->iterations == 0 && d->exitCode != kExitPass) continue;
    reports.push_back(buildDeviceReport(*d, k, nvml));
  }
  const devices::Folded folded = devices::fold(reports, foldRules, foldRules.empty() ? nullptr : foldRules.front().key);

  Measurement m;
  fillMeasurementFromFold(&m, folded);
  m.elapsedS = elapsedS;
  // iterations/sdcDetections/mean* fields are evidence about one representative
  // device (the first that reported), not folded — see Measurement's comments.
  if (!reportable.empty()) {
    DeviceCtx *first = reportable.front();
    m.iterations = first->iterations;
    m.sdcDetections = first->sdcDetections;
    m.eccSupport = first->eccSupport;
    if (first->s.tempN > 0) m.meanTempC = first->s.mean(first->s.tempSum, first->s.tempN);
    if (first->s.reasonsN > 0) {
      m.powerCapEvents = first->s.powerCapEvents;
      m.reasonMask = first->s.reasonMask;
    }
    if (first->s.utilN > 0) {
      m.utilKnown = true;
      m.meanUtilPct = first->s.mean(first->s.utilSum, first->s.utilN);
    }
  }

  emitMeasurement(k, m, ++(*seq));

  const std::vector<devices::SpreadSpec> spreads = {
      {"sustainedClockSpreadPct", "sustained_clock_pct", /*absoluteFigure=*/false},
  };
  devices::printFold(stdout, reports, visible, windowS, mode, folded, spreads, underMig);
  if (final && reports.size() > 1) {
    std::fputs(devices::renderPerDeviceArtifact(reports).c_str(), stdout);
  }
  std::fflush(stdout);
}

// ── the interleaved load loop, shared by both iteration modes ──────────────
//
// Sequential mode calls this once per device, with `active` holding exactly
// one DeviceCtx and that device's own (shorter) window; concurrent mode calls
// it once with every device's context and the full window. Either way this is
// the SAME lockstep: launch every still-running device's next iteration
// (already-async CUDA calls on distinct streams genuinely execute at once on
// the device side), then poll each stream without blocking, harvesting a
// finished iteration's counters and relaunching until that device's own
// deadline, sampling NVML on its own configured cadence regardless of which
// devices happen to finish an iteration on a given pass. A device that errors
// stops being polled but does not stop the others — see device_fold.h's
// combineExitCodes for what that means for the test's own exit code.
//
// `reportable` is what periodic snapshots fold — see emitDeviceMeasurementImpl
// for why it is not simply `active`: the caller passes `active` itself for
// concurrent mode (every device is running from the start) and `active` PLUS
// whatever already finished for sequential mode (built by the caller before
// each call, since it does not change during any single call).
inline void runActiveDevices(std::vector<DeviceCtx *> &active, std::vector<DeviceCtx *> &reportable,
                             const nvmlrt::Library &nvml, long sampleIntervalMs, long progressIntervalS,
                             const Keys &k, const std::vector<devices::FoldRule> &foldRules,
                             kmsg::Watch &xidWatch, kmsg::Tally &xidTally, long *seq, int visible,
                             long windowS, devices::Concurrency mode, double runStart) {
  const double sampleIntervalS = static_cast<double>(sampleIntervalMs) / 1000.0;
  double lastProgress = nowSeconds();

  auto launchOne = [&](DeviceCtx *d) {
    cudaSetDevice(d->index);
    d->measuringThisLaunch = nowSeconds() >= d->warmupUntil;
    cudaEventRecord(d->evStart, d->stream);
    sgemmKernel<<<dim3(d->n / kTileN, d->n / kTileM), dim3(kBlockDim, kBlockDim), 0, d->stream>>>(
        d->dA, d->dB, d->dC, d->n);
    cudaEventRecord(d->evStop, d->stream);
    const int verifyBlocks = std::min(65535, std::max(1, d->props.multiProcessorCount * 32));
    verifyKernel<<<verifyBlocks, 256, 0, d->stream>>>(d->dC, d->dRef, d->elems, d->dCounters);
    // A launch-configuration error is reported synchronously by
    // cudaGetLastError, not by the later synchronise/query. Checking only the
    // query would let a kernel that never ran look like a completed iteration.
    const cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
      d->exitCode = kExitError;
      d->detail = std::string("soak launch failed: ") + cudaGetErrorString(launchErr);
      d->active = false;
      return;
    }
    d->launched = true;
  };

  // started/warmupUntil/deadline are the caller's to set, before this runs —
  // sequential mode gives each device its own later start time than the last,
  // and overwriting them here would collapse every device's warm-up window to
  // zero the instant its first kernel launched.
  for (DeviceCtx *d : active) launchOne(d);

  for (;;) {
    bool anyActive = false;
    for (DeviceCtx *d : active) {
      if (!d->active) continue;
      anyActive = true;
      cudaSetDevice(d->index);
      if (d->launched) {
        const cudaError_t q = cudaStreamQuery(d->stream);
        if (q == cudaSuccess) {
          d->launched = false;
          d->iterations++;
          if (d->measuringThisLaunch) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, d->evStart, d->evStop) == cudaSuccess && ms > 0.0f) {
              d->totalFlops += d->flopsPerGemm;
              d->totalGemmSeconds += ms / 1000.0;
            }
          }
          if (cudaMemcpy(d->hostCounters, d->dCounters, sizeof(d->hostCounters),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            d->exitCode = kExitError;
            d->detail = "could not read the device counter block";
            d->active = false;
            continue;
          }
          if (d->hostCounters[kCounterMiscompares] > d->prevMiscompares) {
            d->sdcDetections++;
            d->prevMiscompares = d->hostCounters[kCounterMiscompares];
          }
          if (nowSeconds() < d->deadline) {
            launchOne(d);
          } else {
            d->active = false; // this device's window is done
          }
        } else if (q != cudaErrorNotReady) {
          d->exitCode = kExitError;
          d->detail = std::string("soak kernel failed: ") + cudaGetErrorString(q);
          d->active = false;
        }
      }
      const double now = nowSeconds();
      if (now - d->lastSampleAt >= sampleIntervalS) {
        takeSample(nvml, d->nvdev, &d->s, &d->samplerOk);
        d->lastSampleAt = now;
      }
    }
    if (!anyActive) break;

    const double now = nowSeconds();
    if (now - lastProgress >= static_cast<double>(progressIntervalS)) {
      // Drained on the same cadence as every other periodic signal, for the
      // same reason a single-device run does: a pod killed between two
      // progress prints still has an xidCount current as of the last drain.
      // The watch is process-wide, not per device.
      xidWatch.Collect([&](const std::string &line) { xidTally.ObserveNvidia(line); });
      emitDeviceMeasurementImpl(k, reportable, foldRules, nvml, seq, nowSeconds() - runStart, visible,
                               windowS, mode, /*underMig=*/false, /*final=*/false);
      lastProgress = now;
    }
    sleepMillis(kPollMillis);
  }
}

// ── the engine ──────────────────────────────────────────────────────────────

// Run drives the whole soak — across every device the pod was allocated — and
// fills *m with the fold. It returns 0 when at least one device measured
// cleanly and the caller may now judge the hardware from the folded
// Measurement; any other return is a process exit code and the marker has
// ALREADY been printed.
inline int run(const Keys &k, const std::vector<devices::FoldRule> &foldRules, Measurement *m) {
  // ── /dev/kmsg, opened before anything else in this function ────────────────
  //
  // One watch for the whole pod, not per device — see the Measurement comment.
  // Positioning happens here, deliberately before config parsing and every
  // CUDA/NVML call below: the window this soak reports Xids over should start
  // as close to process start as this runner can make it.
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
  if (matrixN < kMinMatrixN || matrixN > kMaxMatrixN) {
    warn("BURNIN_SOAK_MATRIX_N clamped into [" + std::to_string(kMinMatrixN) + "," +
         std::to_string(kMaxMatrixN) + "]");
    matrixN = std::max(static_cast<long>(kMinMatrixN), std::min(matrixN, static_cast<long>(kMaxMatrixN)));
  }
  std::printf("duration_requested_s=%ld\n", durationRequested);
  std::printf("sample_interval_ms=%ld\n", sampleIntervalMs);
  std::printf("progress_interval_s=%ld\n", progressIntervalS);

  // ── how many devices, and how ────────────────────────────────────────────
  int visible = 0;
  const cudaError_t countErr = cudaGetDeviceCount(&visible);
  if (countErr != cudaSuccess && countErr != cudaErrorNoDevice) {
    return errored(k, std::string("cudaGetDeviceCount: ") + cudaGetErrorString(countErr));
  }
  if (countErr == cudaErrorNoDevice) visible = 0;

  const devices::Budget budget = devices::parseBudget(std::getenv("BURNIN_RESOURCE_LIMITS"), devices::nvidiaResources());
  const devices::Plan plan = devices::planIteration(visible, budget);
  if (plan.outcome == devices::Plan::Skip) return skip(k, plan.message);
  if (plan.outcome == devices::Plan::Error) return errored(k, plan.message);
  const int planCount = plan.count;

  const char *concEnv = std::getenv("BURNIN_DEVICE_CONCURRENCY");
  const devices::ConcurrencyChoice conc = devices::resolveConcurrency(concEnv, devices::Concurrency::All);
  if (!conc.recognised) {
    warn(std::string("BURNIN_DEVICE_CONCURRENCY=\"") + concEnv +
         "\" is neither \"all\" nor \"sequential\"; using this kind's default (all)");
  }
  const long windowS = devices::deviceWindowSeconds(durationSeconds, planCount, conc.mode);
  // Mirrors the single-device engine's own warm-up derivation exactly, applied
  // to the device's own window rather than the whole test's duration.
  long warmupSeconds = std::max(5L, std::min(30L, windowS / 10));
  if (warmupSeconds >= windowS) warmupSeconds = windowS / 3;
  const long effectiveWarmup = warmupSeconds;

  std::printf("device_window_s=%ld\n", windowS);
  std::printf("warmup_s=%ld\n", effectiveWarmup);

  // Documented as always emitted (README: "Proving the detector works"), so
  // the key stays exactly `faults_injected` — the requested count. Each
  // device clamps it to its OWN elems (which can differ across a
  // heterogeneous board, or when free memory shrinks the matrix) before
  // actually injecting, in setupDevice; this line is the config as asked for.
  std::printf("faults_injected=%ld\n", injectMiscompares);
  if (injectMiscompares > 0) {
    warn("BURNIN_SOAK_INJECT_MISCOMPARES=" + std::to_string(injectMiscompares) +
         " corrupted the reference on purpose; this run is a self-test of the detector and is "
         "NOT a verdict about this hardware");
  }

  // ── NVML, opened ONCE for the whole process ─────────────────────────────
  // An accelerator IS present at this point (Skip already returned otherwise).
  // If its management library is missing, the container was built or run
  // without the "utility" driver capability — a misconfiguration, not a
  // property of the hardware.
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
  if (nvml.systemGetDriverVersion != nullptr) {
    char driver[96] = {0};
    if (nvml.systemGetDriverVersion(driver, sizeof(driver)) == nvmlrt::kSuccess) {
      std::printf("driver_version=%s\n", driver);
    }
  }

  // ── set up every device the plan admits ─────────────────────────────────
  std::vector<DeviceCtx> ctxs(planCount);
  for (int i = 0; i < planCount; ++i) {
    ctxs[i].index = i;
    setupDevice(&ctxs[i], static_cast<int>(matrixN), injectMiscompares, nvml);
  }

  std::vector<DeviceCtx *> allPtrs;
  for (auto &d : ctxs) allPtrs.push_back(&d);

  const double runStart = nowSeconds();
  long seq = 0;

  if (conc.mode == devices::Concurrency::All) {
    // Every device starts together, so the reportable set is the active set
    // itself for the whole call — nothing outside it will ever start later.
    std::vector<DeviceCtx *> active;
    for (auto &d : ctxs) {
      if (!d.active) continue;
      d.started = nowSeconds();
      d.warmupUntil = d.started + effectiveWarmup;
      d.deadline = d.started + windowS;
      active.push_back(&d);
    }
    if (!active.empty()) {
      runActiveDevices(active, active, nvml, sampleIntervalMs, progressIntervalS, k, foldRules, xidWatch,
                       xidTally, &seq, visible, windowS, conc.mode, runStart);
    }
  } else {
    // One device at a time. `reportable` for device i's call is every earlier
    // device (already finished, real data) plus device i itself — never a
    // device that has not started, which is what keeps sequential-mode
    // periodic snapshots from reading as a probe failure on the devices still
    // waiting their turn.
    std::vector<DeviceCtx *> finishedSoFar;
    for (auto &d : ctxs) {
      if (!d.active) continue;
      std::vector<DeviceCtx *> one = {&d};
      std::vector<DeviceCtx *> reportable = finishedSoFar;
      reportable.push_back(&d);
      d.started = nowSeconds();
      d.warmupUntil = d.started + effectiveWarmup;
      d.deadline = d.started + windowS;
      runActiveDevices(one, reportable, nvml, sampleIntervalMs, progressIntervalS, k, foldRules, xidWatch,
                       xidTally, &seq, visible, windowS, conc.mode, runStart);
      finishedSoFar.push_back(&d);
    }
  }

  const double finished = nowSeconds();
  xidWatch.Collect([&](const std::string &line) { xidTally.ObserveNvidia(line); });
  m->xidAvailable = xidWatch.Available();
  m->xidDropped = xidWatch.Dropped();
  m->xidUnavailableReason = xidWatch.Why();
  m->xidCount = xidTally.xidCount;
  m->haveLastXidCode = xidTally.haveLastXidCode;
  m->lastXidCode = xidTally.lastXidCode;

  // ── final report ─────────────────────────────────────────────────────────
  bool underMig = false;
  for (auto &d : ctxs) {
    if (d.exitCode == kExitSkip) underMig = true;
  }
  emitDeviceMeasurementImpl(k, allPtrs, foldRules, nvml, &seq, finished - runStart, visible, windowS, conc.mode,
                           underMig, /*final=*/true);

  // Re-fold once more into *m for the caller's own assertions, from exactly
  // the same reports the final emission just printed.
  std::vector<devices::DeviceReport> finalReports;
  for (auto &d : ctxs) {
    if (d.iterations == 0 && d.exitCode != kExitPass) continue;
    finalReports.push_back(buildDeviceReport(d, k, nvml));
  }
  const devices::Folded finalFold =
      devices::fold(finalReports, foldRules, foldRules.empty() ? nullptr : foldRules.front().key);
  fillMeasurementFromFold(m, finalFold);
  m->elapsedS = finished - runStart;
  if (!finalReports.empty()) {
    m->iterations = ctxs.front().iterations;
    m->sdcDetections = ctxs.front().sdcDetections;
  }
  for (auto &d : ctxs) {
    if (d.eccSupport == nvmlrt::kEccPresent || d.eccSupport == nvmlrt::kEccAbsent) {
      m->eccSupport = d.eccSupport;
      break;
    }
  }
  if (!ctxs.empty() && ctxs.front().s.reasonsN > 0) {
    m->throttleKnown = m->throttleKnown || ctxs.front().s.reasonsN > 0;
    m->reasonMask = ctxs.front().s.reasonMask;
    m->powerCapEvents = ctxs.front().s.powerCapEvents;
  }
  if (!ctxs.empty() && ctxs.front().s.utilN > 0) {
    m->utilKnown = true;
    m->meanUtilPct = ctxs.front().s.mean(ctxs.front().s.utilSum, ctxs.front().s.utilN);
  }
  if (!ctxs.empty() && ctxs.front().s.tempN > 0) {
    m->tempKnown = m->tempKnown || ctxs.front().s.tempN > 0;
    m->meanTempC = ctxs.front().s.mean(ctxs.front().s.tempSum, ctxs.front().s.tempN);
  }

  // The sample-level summary. Device 0 only, matching every other "evidence,
  // not folded" field above — the per-device.json artifact is the sound
  // cross-device attribution, and this is the same quick-read single-device
  // fleets have always had.
  if (!ctxs.empty()) {
    const DeviceCtx &d0 = ctxs.front();
    std::printf("samples_taken=%ld\n", d0.s.n);
    std::printf("gemm_active_s=%.2f\n", d0.totalGemmSeconds);
    if (d0.s.n > 0) {
      if (d0.ratedKnown) {
        std::printf("smClockMHz=%.0f\n", d0.s.smSum / d0.s.n);
        std::printf("min_sm_clock_pct=%.2f\n", 100.0 * d0.s.smMin / d0.ratedBoostMHz);
        std::printf("max_sm_clock_pct=%.2f\n", 100.0 * d0.s.smMax / d0.ratedBoostMHz);
      }
      if (d0.s.memN > 0) std::printf("memClockMHz=%.0f\n", d0.s.memSum / d0.s.memN);
      if (d0.s.powerN > 0) std::printf("mean_power_w=%.2f\n", d0.s.mean(d0.s.powerSum, d0.s.powerN));
      if (d0.s.utilN > 0) {
        std::printf("gpu_utilization_pct=%.2f\n", d0.s.mean(d0.s.utilSum, d0.s.utilN));
        std::printf("mem_utilization_pct=%.2f\n", d0.s.mean(d0.s.memUtilSum, d0.s.utilN));
      }
      if (d0.s.reasonsN > 0) {
        std::printf("throttled_samples=%ld\n", d0.s.protectedSamples);
        std::printf("throttle_reasons_mask=%llu\n", d0.s.reasonMask);
        std::string labels;
        for (int i = 0; i < kNumReasonBits; ++i) {
          std::printf("%s%ld\n", kReasonBits[i].key, d0.s.reasonCount[i]);
          if ((d0.s.reasonMask & kReasonBits[i].bit) != 0) {
            if (!labels.empty()) labels += ",";
            labels += kReasonBits[i].label;
          }
        }
        std::printf("throttle_reasons=%s\n", labels.empty() ? "none" : labels.c_str());
        std::printf("thermal_throttle_latched=%s\n",
                    (d0.s.reasonMask & kThermalReasons) != 0 ? "true" : "false");
      }
    }
    if (d0.hostCounters[kCounterFirstBadIndex] != ~0ULL) {
      std::printf("first_miscompare_index=%llu\n", d0.hostCounters[kCounterFirstBadIndex]);
    }
  }

  emitCommon();

  // ── combine per-device outcomes into the machinery exit code ────────────
  std::vector<int> codes;
  for (auto &d : ctxs) codes.push_back(d.exitCode);
  const int combined = devices::combineExitCodes(codes);

  for (auto &d : ctxs) releaseDevice(&d);
  nvml.close();

  if (combined == kExitSkip) {
    // Every device that could not be measured was MIG; nothing was skipped
    // silently — the per-device mig_mode lines above already said which.
    return skip(k, "every allocated device is MIG-enabled; clock, temperature and power cannot be "
                   "attributed to one instance's load");
  }
  if (combined == kExitError) {
    std::string reasons;
    for (auto &d : ctxs) {
      if (d.exitCode == kExitError && !d.detail.empty()) {
        if (!reasons.empty()) reasons += "; ";
        reasons += "device " + std::to_string(d.index) + ": " + d.detail;
      }
    }
    return errored(k, reasons.empty() ? "no device could be measured" : reasons);
  }
  if (finalReports.empty()) {
    return errored(k, "the soak completed no iterations on any device; nothing was measured");
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

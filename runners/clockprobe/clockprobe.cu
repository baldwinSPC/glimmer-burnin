// clockprobe.cu — sustained-clock-under-load probe for the "clockprobe" TestKind.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It uses no third-party
// source. The CUDA toolchain compiles it at build time only, and NVML is
// resolved at runtime by dlopen (see nvml_dynamic.h) so the published image
// carries no NVIDIA redistributable library.
//
// WHAT THIS CATCHES
// -----------------
// A GPU pinned in a low P-state that reports perfectly healthy utilization. The
// node looks busy, nvidia-smi shows 100% utilization, every liveness and health
// check passes, and the part simply runs at a fraction of its rated speed. On
// GB10 (DGX Spark) this is the USB-C Power-Delivery failure mode: an under-spec
// supply, a degraded cable, or a PD contract that negotiated a lower wattage
// silently caps the board power budget and the part clocks down and stays down.
//
// Most of the suite is blind to it. compute-smoke passes (the arithmetic is
// still correct, just slower). dcgm-diag passes (nothing is faulty). memory-bw
// passes (the memory path is not what was capped). Every one of those asserts
// correctness or health; none of them asserts SPEED. A fleet with this fault
// delivers correct results, slowly, forever, and the loss shows up only in the
// training-job wall-clock — usually months later.
//
// thermal-soak is the EXCEPTION and this runner does not claim otherwise: it
// applies its own sustained-clock floor (THERMAL_SOAK_MIN_CLOCK_PCT, default 60%
// of rated boost) and fails below it. That floor sits ten points under the
// MEASURED 69.9% asymptote of a healthy part rather than being calibrated
// against a wedge, but this project's ESTIMATE of where a wedge lands — near 20%
// of rated boost, see the NOT VERIFIED note below — is far enough under it that
// a wedged part put through a soak should fail the soak.
//
// What this probe adds is COST and ATTRIBUTION, not exclusive coverage. It
// defaults to 60 seconds against the soak's 900, which is the difference between
// a gate affordable at every enrollment and one run overnight; and the soak's
// clock floor is explicitly a backstop rather than its verdict, so a wedge
// surfaces there as "the soak did not pass" while this runner reports it as a
// wedge — pd_wedge_suspected, throttle_classification, power_limit_ratio_pct —
// pointing at a cable or a supply.
//
// NOT VERIFIED: that thermal-soak fires on a genuinely wedged part is inferred
// from reading thermal_soak.cu, not observed. No wedged Spark was available to
// run either test against, and WHERE A WEDGE LANDS HAS NEVER BEEN MEASURED HERE.
// The 20% figure is an estimate: a researched pin point of ~611 MHz over the
// 3003 MHz rated boost this fleet reports (611/3003 = 20.3%), agreeing in order
// of magnitude with GEP-0178's independent "~4x slow at 96% reported
// utilization". Do not tighten any threshold on the strength of it. Tracked as
// issue #61.
//
// HOW IT WORKS
// ------------
// Apply a known, steady, clock-bound compute load; sample the achieved SM clock,
// memory clock, temperature, board power, utilization and the driver's throttle
// reason mask while that load is running; and compare the achieved SM clock
// against the part's own rated boost clock read back from the driver.
//
// The load is a register-resident FP32 FMA chain: no memory traffic, no cache
// effects, no dependence on problem size. That matters twice over. It is
// clock-bound, so achieved throughput moves in lockstep with achieved clock and
// gives an independent cross-check on the clock the driver reports; and its FLOP
// count is exact, so sustainedThroughputTflops is a real measurement rather than
// a figure derived from a nameplate constant.
//
// THERMAL THROTTLE vs POWER-DELIVERY WEDGE
// ----------------------------------------
// Both present as a low clock, and treating them alike would make this test
// useless: thermal throttling under sustained heat is expected behaviour and is
// thermal-soak's business, while a PD wedge is a broken node. They are separated
// by evidence the runner emits, not by guesswork:
//
//   thermal   low clock, HIGH temperature, a thermal reason latched in the mask
//             (swThermalSlowdown / hwThermalSlowdown / hwSlowdown)
//   PD wedge  low clock, LOW temperature, typically swPowerCap or hwPowerBrake
//             latched, and frequently an enforced power limit well below the
//             board's default limit — the negotiated contract, visible
//
// If temperature cannot be read, the shortfall is NOT attributed to heat. An
// unknown measurement must never buy the more lenient floor; that is the same
// fail-closed rule the verdict package follows.
//
// MULTI-DEVICE
// ------------
// Iterates every device the pod was allocated (docs/dev/multi-device.md),
// sequential by default: this probe isolates ONE device's clock behaviour, and
// that is a measurement kind's job, not a soak's — see the design note's
// "sequential vs concurrent" reasoning. BURNIN_DEVICE_CONCURRENCY=all overrides
// it, and unlike the soak family's single-host-thread lockstep, concurrent mode
// here is genuinely one std::thread per device: this probe's per-device pipeline
// is already fully self-contained and blocking (cudaStreamQuery-poll-and-sample
// is a per-device pattern, not a cross-device one), and `cudaSetDevice` scopes
// to the CALLING THREAD, so N threads each pinned to their own device via
// cudaSetDevice at the top is the standard multi-GPU-via-threads pattern —
// no interleaving machinery to build. NVML's query calls are assumed
// thread-safe (documented NVIDIA behaviour and the basis of every concurrent
// GPU-monitoring tool built on it); nothing here calls an NVML mutator.
// Per-thread config-warning/unsupported-reads strings avoid a data race on the
// two process-global ones and are merged into them after every thread joins —
// see mergeDeviceNotes.
//
// The gated metric (sustainedClockPct) is the WORST device, exactly as the
// design note requires — but this file has no separate "fold the aggregate,
// then decide" step the way the soak family does: DECIDING is per device (the
// same classification/floor logic below, run once per device against ITS OWN
// samples), and the OVERALL test result is combineExitCodes across devices.
// Folding via device_fold.h is what turns "8 devices' passes" into the single
// FOLDED metric line a consumer reads (Min of sustained_clock_pct, worst
// device named), not what decides the verdict — a device that clears its own
// floor is not un-passed by another device's number.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, ALWAYS printed before the decision, then one of
//     CLOCKPROBE_PASS               exit 0
//     CLOCKPROBE_FAIL: <reason>     exit 1
//     CLOCKPROBE_SKIP: <reason>     exit 2   not applicable to this hardware
//     CLOCKPROBE_ERROR: <reason>    exit 3   unjudged; NOT a hardware verdict

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "device_fold.h"
#include "nvml_dynamic.h"

namespace {

namespace devices = burnin::devices;

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
// Any exit code that is not 0/1/2 is Error to the operator. 3 is chosen so the
// code is stable and greppable rather than whatever errno-ish value leaks out.
constexpr int kExitError = 3;

// A probe shorter than this cannot separate a ramp from a steady state: the part
// is still coming up to clock, so an unwarmed sample is legitimately low and
// would fail a healthy node. A requested duration below the floor is raised to
// it and the adjustment is reported in configWarnings.
constexpr long kMinDurationSeconds = 10;
constexpr long kDefaultDurationSeconds = 60;

// Target wall time for one measurement window. The window is the unit over
// which throughput is computed, and it is long enough to swamp launch overhead
// while short enough that several windows fit in a short probe.
constexpr double kWindowSeconds = 1.0;
// Target wall time for a single kernel launch. Short enough that the host can
// poll and sample at a steady cadence, long enough that launch latency is noise.
constexpr double kLaunchSeconds = 0.05;

// FMA chains per thread per loop iteration, and the FLOP count that follows.
// Eight independent chains keep the SM's pipelines fed without spilling.
constexpr int kChains = 8;
constexpr double kFlopsPerIterPerThread = kChains * 2.0; // fused multiply-add = 2 flops
constexpr int kThreadsPerBlock = 256;
constexpr int kBlocksPerSM = 8;

// If the load is not actually executing on the device, the clock we sampled is
// not attributable to it and the run is unjudged rather than failed. Utilization
// well below this while we believe we are hammering the part means the
// measurement, not the hardware, is broken.
constexpr double kMinCredibleUtilizationPct = 50.0;

// ── the load ────────────────────────────────────────────────────────────────
// Register-resident FMA chains. No loads, no stores, no shared memory: the only
// thing that can make this slower is a lower clock.
__global__ void fmaLoadKernel(float *sink, int iters) {
  const float b = 0.99999994f;
  const float c = 1.0000001f;
  float x0 = threadIdx.x * 1.0e-4f + 1.0f;
  float x1 = x0 + 0.125f, x2 = x0 + 0.250f, x3 = x0 + 0.375f;
  float x4 = x0 + 0.500f, x5 = x0 + 0.625f, x6 = x0 + 0.750f, x7 = x0 + 0.875f;
  for (int i = 0; i < iters; ++i) {
    x0 = fmaf(x0, b, c);
    x1 = fmaf(x1, b, c);
    x2 = fmaf(x2, b, c);
    x3 = fmaf(x3, b, c);
    x4 = fmaf(x4, b, c);
    x5 = fmaf(x5, b, c);
    x6 = fmaf(x6, b, c);
    x7 = fmaf(x7, b, c);
  }
  // The recurrence converges to c/(1-b) — finite and positive, so s can never
  // be -inf. The compiler cannot prove that, so the chains stay live and the
  // loop is not optimised away. The null check is belt and braces: every launch
  // in this runner passes sink=nullptr.
  const float s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
  if (sink != nullptr && s == -__int_as_float(0x7f800000)) sink[threadIdx.x] = s;
}

// ── small helpers ───────────────────────────────────────────────────────────

double nowSeconds() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

void sleepMillis(long ms) {
  timespec ts{ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, nullptr);
}

// Process-global: written only from the main thread, either directly
// (sequential mode, one device at a time) or via mergeDeviceNotes after every
// worker thread has joined (concurrent mode) — never concurrently.
std::string configWarnings;
void warn(const std::string &s) {
  if (!configWarnings.empty()) configWarnings += "; ";
  configWarnings += s;
}

std::string unsupportedReads;
void noteUnsupported(const char *what) {
  if (!unsupportedReads.empty()) unsupportedReads += ",";
  unsupportedReads += what;
}

// envLong is strict about garbage: a value we cannot parse is a configuration
// error, and guessing a default in its place would run a different test than the
// one that was asked for while reporting success.
bool envLong(const char *name, long dflt, long *out, std::string *err) {
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

bool envDouble(const char *name, double dflt, double *out, std::string *err) {
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

// ── sampling ────────────────────────────────────────────────────────────────

struct ReasonBit {
  unsigned long long bit;
  const char *key;   // runner key; pkg/runner normalises it to lowerCamelCase
  const char *label; // for the human-readable reason list
};

const ReasonBit kReasonBits[] = {
    {nvmlrt::kReasonGpuIdle, "gpu_idle_samples", "gpuIdle"},
    {nvmlrt::kReasonApplicationsClocksSetting, "applications_clocks_setting_samples",
     "applicationsClocksSetting"},
    {nvmlrt::kReasonSwPowerCap, "sw_power_cap_samples", "swPowerCap"},
    {nvmlrt::kReasonHwSlowdown, "hw_slowdown_samples", "hwSlowdown"},
    {nvmlrt::kReasonSyncBoost, "sync_boost_samples", "syncBoost"},
    {nvmlrt::kReasonSwThermalSlowdown, "sw_thermal_slowdown_samples", "swThermalSlowdown"},
    {nvmlrt::kReasonHwThermalSlowdown, "hw_thermal_slowdown_samples", "hwThermalSlowdown"},
    {nvmlrt::kReasonHwPowerBrakeSlowdown, "hw_power_brake_samples", "hwPowerBrake"},
    {nvmlrt::kReasonDisplayClockSetting, "display_clock_setting_samples", "displayClockSetting"},
};
constexpr int kNumReasonBits = static_cast<int>(sizeof(kReasonBits) / sizeof(kReasonBits[0]));

// Everything that makes a clock reading capping rather than idle. gpuIdle is
// excluded on purpose: an idle GPU is not throttled, it is unloaded, and
// counting it as a throttle would turn a broken load into a hardware verdict.
constexpr unsigned long long kCappingReasons =
    nvmlrt::kReasonApplicationsClocksSetting | nvmlrt::kReasonSwPowerCap |
    nvmlrt::kReasonHwSlowdown | nvmlrt::kReasonSyncBoost | nvmlrt::kReasonSwThermalSlowdown |
    nvmlrt::kReasonHwThermalSlowdown | nvmlrt::kReasonHwPowerBrakeSlowdown |
    nvmlrt::kReasonDisplayClockSetting;

constexpr unsigned long long kThermalReasons = nvmlrt::kReasonSwThermalSlowdown |
                                               nvmlrt::kReasonHwThermalSlowdown |
                                               nvmlrt::kReasonHwSlowdown;

constexpr unsigned long long kPowerReasons =
    nvmlrt::kReasonSwPowerCap | nvmlrt::kReasonHwPowerBrakeSlowdown;

struct Samples {
  long n = 0;

  double smSum = 0.0;
  unsigned smMin = 0xFFFFFFFFu;
  unsigned smMax = 0;

  long memN = 0;
  double memSum = 0.0;

  long tempN = 0;
  double tempSum = 0.0;
  double tempMax = 0.0;
  double tempAtMinSm = 0.0;
  bool tempAtMinSmKnown = false;

  long powerN = 0;
  double powerSum = 0.0;
  double powerMax = 0.0;

  long utilN = 0;
  double utilSum = 0.0;
  double memUtilSum = 0.0;

  long reasonsN = 0;
  unsigned long long reasonMask = 0;
  long reasonCount[kNumReasonBits] = {0};
  long throttledSamples = 0;
  long throttleEvents = 0; // transitions into a capped state, not sample counts
  bool prevThrottled = false;

  double mean(double sum, long count) const { return count > 0 ? sum / count : 0.0; }
};

// SamplerState is which optional NVML reads are still believed to work, per
// device — a read the driver refuses on one device says nothing about another.
struct SamplerState {
  bool mem = true, temp = true, power = true, util = true, reasons = true;
};

// takeSample reads one instant of the device's state. notes carries this
// device's OWN unsupported-reads list (merged into the process-global one
// after every device is done), so two devices probing NVML concurrently never
// touch the shared string.
void takeSample(const nvmlrt::Library &nvml, nvmlrt::Device dev, Samples *s, SamplerState *ok,
               std::string *notes) {
  auto note = [&](const char *what) {
    if (!notes->empty()) *notes += ",";
    *notes += what;
  };
  unsigned int sm = 0;
  if (nvml.deviceGetClockInfo(dev, nvmlrt::kClockSM, &sm) != nvmlrt::kSuccess) return;

  s->n++;
  s->smSum += sm;
  s->smMax = std::max(s->smMax, sm);

  double temp = 0.0;
  bool tempKnown = false;
  if (ok->temp && nvml.deviceGetTemperature != nullptr) {
    unsigned int t = 0;
    if (nvml.deviceGetTemperature(dev, nvmlrt::kTemperatureGpu, &t) == nvmlrt::kSuccess) {
      temp = t;
      tempKnown = true;
      s->tempN++;
      s->tempSum += temp;
      s->tempMax = std::max(s->tempMax, temp);
    } else {
      ok->temp = false;
      note("temperature");
    }
  }

  // The temperature AT the slowest observed clock is the single most useful
  // number for telling the two failure modes apart, so it is captured with the
  // minimum rather than reconstructed from averages afterwards.
  if (sm < s->smMin) {
    s->smMin = sm;
    s->tempAtMinSmKnown = tempKnown;
    s->tempAtMinSm = temp;
  }

  if (ok->mem) {
    unsigned int mem = 0;
    if (nvml.deviceGetClockInfo(dev, nvmlrt::kClockMem, &mem) == nvmlrt::kSuccess) {
      s->memN++;
      s->memSum += mem;
    } else {
      ok->mem = false;
      note("memClock");
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
      note("powerDraw");
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
      note("utilization");
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
      const bool capped = (mask & kCappingReasons) != 0;
      if (capped) s->throttledSamples++;
      if (capped && !s->prevThrottled) s->throttleEvents++;
      s->prevThrottled = capped;
    } else {
      ok->reasons = false;
      note("throttleReasons");
    }
  }
}

// ── terminal output ─────────────────────────────────────────────────────────
//
// Markers are printed LAST, after every metric, so that a failing or erroring
// run still yields its evidence. pkg/runner takes the final non-key=value line
// as the message, which is exactly the marker.

int emitMarker(const char *marker, const std::string &reason, int code) {
  if (reason.empty()) {
    std::printf("%s\n", marker);
  } else {
    std::printf("%s: %s\n", marker, reason.c_str());
  }
  std::fflush(stdout);
  return code;
}

int skip(const std::string &reason) { return emitMarker("CLOCKPROBE_SKIP", reason, kExitSkip); }
int fail(const std::string &reason) { return emitMarker("CLOCKPROBE_FAIL", reason, kExitFail); }
int errored(const std::string &reason) { return emitMarker("CLOCKPROBE_ERROR", reason, kExitError); }

void emitCommon() {
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
}

// ── per-device state and pipeline ───────────────────────────────────────────

struct DeviceResult {
  int index = 0;
  int exitCode = kExitPass; // 0 pass, 1 fail, 2 skip, 3 error — THIS device's own outcome
  std::string reason;       // why, for the terminal marker if this device decides it

  bool identityRead = false;
  std::string busId, name, computeCap;

  // Raw values for device_fold.h, keyed exactly as printed.
  std::map<std::string, double> values;

  // Evidence carried through for the artifact / device-0 "representative"
  // printing, mirroring the soak family's "Once" fields.
  double meanTemp = 0.0;
  bool tempKnown = false;
  double meanPower = 0.0;
  double peakPower = 0.0;
  bool powerKnown = false;
  double meanUtil = 0.0;
  double meanMemUtil = 0.0;
  bool utilKnown = false;
  unsigned long long reasonMask = 0;
  long reasonCount[kNumReasonBits] = {0};
  bool reasonsKnown = false;
  double smClockMHz = 0.0;
  double maxSmClockPct = 0.0;
  double memClockMHz = 0.0;
  bool memClockKnown = false;
  unsigned int ratedBoostMHz = 0;
  bool ratedKnown = false;
  double sustainedTflops = 0.0;
  double peakTflops = 0.0;
  bool tflopsKnown = false;
  long samplesTaken = 0;
  long loadLaunches = 0;
  double loadThreads = 0;
  int loadItersPerLaunch = 0;
  long warmupSeconds = 0;
  long sampleWindowSeconds = 0;
  double elapsedS = 0.0;
  double tempAtMinSm = 0.0;
  bool tempAtMinSmKnown = false;
  double enforcedLimitW = 0.0, defaultLimitW = 0.0;
  bool haveEnforcedLimit = false, haveDefaultLimit = false;
  const char *classification = "none";
  const char *pdWedge = "false";
  double floorApplied = 0.0;
  const char *floorBasis = "general";

  std::string configWarnings;   // merged into the process-global after this device finishes
  std::string unsupportedReads; // ditto
};

// runOneDevice is TODAY's single-device pipeline — NVML handle resolution
// through calibration, warm-up, the windowed load, classification and the
// per-device pass/fail decision — scoped to one CUDA ordinal and one window.
// Unchanged in substance from the pre-multi-device engine; only parameterised
// by device index and window, and returning a result instead of exiting the
// process.
void runOneDevice(int index, long windowSecondsTotal, double clockFloorPct,
                  double thermalClockFloorPct, double thermalTempC, long sampleIntervalMs,
                  const nvmlrt::Library &nvml, DeviceResult *out) {
  out->index = index;

  if (cudaSetDevice(index) != cudaSuccess) {
    out->exitCode = kExitError;
    out->reason = "cudaSetDevice failed";
    return;
  }
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, index) != cudaSuccess) {
    out->exitCode = kExitError;
    out->reason = "could not read CUDA device properties";
    return;
  }
  char busIdBuf[32] = {0};
  const bool busIdOK = cudaDeviceGetPCIBusId(busIdBuf, sizeof(busIdBuf), index) == cudaSuccess;
  out->busId = busIdOK ? busIdBuf : "";
  out->name = props.name;
  char cc[16];
  std::snprintf(cc, sizeof(cc), "%d.%d", props.major, props.minor);
  out->computeCap = cc;
  out->identityRead = busIdOK;

  nvmlrt::Device nvdev = nullptr;
  bool haveHandle = false;
  // Prefer the PCI bus id: NVML index order and CUDA ordinal order are not
  // guaranteed to agree, and probing the clock of a different GPU than the one
  // under load would produce a confident, wrong verdict.
  if (busIdOK && nvml.deviceGetHandleByPciBusId != nullptr) {
    haveHandle = nvml.deviceGetHandleByPciBusId(out->busId.c_str(), &nvdev) == nvmlrt::kSuccess;
  }
  if (!haveHandle) {
    out->configWarnings += (out->configWarnings.empty() ? "" : "; ") +
                           ("device " + std::to_string(index) +
                            ": NVML handle resolved by index, not PCI bus id");
    if (nvml.deviceGetHandleByIndex(static_cast<unsigned int>(index), &nvdev) != nvmlrt::kSuccess) {
      out->exitCode = kExitError;
      out->reason = "could not resolve an NVML handle for this device";
      return;
    }
  }

  // MIG: clocks are a whole-device property, so a clock sampled from inside a
  // MIG instance cannot be attributed to this instance's load. This device is
  // Skip, not the whole test.
  if (nvml.deviceGetMigMode != nullptr) {
    unsigned int current = 0, pending = 0;
    if (nvml.deviceGetMigMode(nvdev, &current, &pending) == nvmlrt::kSuccess && index == 0) {
      std::printf("mig_mode=%s\n", current == nvmlrt::kMigEnable ? "enabled" : "disabled");
    }
    if (nvml.deviceGetMigMode(nvdev, &current, &pending) == nvmlrt::kSuccess &&
        current == nvmlrt::kMigEnable) {
      out->exitCode = kExitSkip;
      out->reason = "MIG is enabled; SM clock is a device-wide property and cannot be "
                    "attributed to one instance's load";
      return;
    }
  }

  // The denominator. Without a rated clock there is no portable percentage, and
  // a missing nameplate is an UNJUDGED part rather than a slow one — Skip.
  unsigned int ratedBoostMHz = 0;
  nvmlrt::Return rc = nvml.deviceGetMaxClockInfo(nvdev, nvmlrt::kClockSM, &ratedBoostMHz);
  if (rc != nvmlrt::kSuccess || ratedBoostMHz == 0) {
    out->exitCode = kExitSkip;
    out->reason = std::string("rated SM boost clock unavailable (") + nvml.errorString(rc) +
                 "); without a nameplate denominator the part is unjudged, not slow";
    return;
  }
  out->ratedKnown = true;
  out->ratedBoostMHz = ratedBoostMHz;

  double enforcedLimitW = 0.0, defaultLimitW = 0.0;
  bool haveEnforced = false, haveDefault = false;
  if (nvml.deviceGetEnforcedPowerLimit != nullptr) {
    unsigned int mw = 0;
    if (nvml.deviceGetEnforcedPowerLimit(nvdev, &mw) == nvmlrt::kSuccess) {
      enforcedLimitW = mw / 1000.0;
      haveEnforced = true;
    } else {
      out->unsupportedReads += (out->unsupportedReads.empty() ? "" : ",") + std::string("enforcedPowerLimit");
    }
  }
  if (nvml.deviceGetPowerManagementDefaultLimit != nullptr) {
    unsigned int mw = 0;
    if (nvml.deviceGetPowerManagementDefaultLimit(nvdev, &mw) == nvmlrt::kSuccess) {
      defaultLimitW = mw / 1000.0;
      haveDefault = true;
    } else {
      out->unsupportedReads += (out->unsupportedReads.empty() ? "" : ",") + std::string("defaultPowerLimit");
    }
  }
  const bool limitBelowDefault =
      haveEnforced && haveDefault && defaultLimitW > 0.0 && enforcedLimitW < defaultLimitW * 0.95;
  out->haveEnforcedLimit = haveEnforced;
  out->enforcedLimitW = enforcedLimitW;
  out->haveDefaultLimit = haveDefault;
  out->defaultLimitW = defaultLimitW;

  // ── build the load ──────────────────────────────────────────────────────
  const int blocks = std::max(1, props.multiProcessorCount * kBlocksPerSM);
  const double totalThreads = static_cast<double>(blocks) * kThreadsPerBlock;

  cudaStream_t stream = nullptr;
  cudaEvent_t evStart = nullptr, evStop = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess || cudaEventCreate(&evStart) != cudaSuccess ||
      cudaEventCreate(&evStop) != cudaSuccess) {
    out->exitCode = kExitError;
    out->reason = "could not create CUDA stream/events";
    return;
  }
  auto cleanup = [&]() {
    if (evStart) cudaEventDestroy(evStart);
    if (evStop) cudaEventDestroy(evStop);
    if (stream) cudaStreamDestroy(stream);
  };

  // Calibrate: how many loop iterations make one launch take ~kLaunchSeconds?
  // Done on the live part so the load is sized for the clock it actually runs
  // at, including a clock that is already wedged low.
  int iters = 4096;
  {
    fmaLoadKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(nullptr, 1024);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr == cudaSuccess) launchErr = cudaStreamSynchronize(stream);
    if (launchErr != cudaSuccess) {
      out->exitCode = kExitError;
      out->reason = std::string("initial launch failed: ") + cudaGetErrorString(launchErr);
      cleanup();
      return;
    }
    for (int round = 0; round < 2; ++round) {
      cudaEventRecord(evStart, stream);
      fmaLoadKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(nullptr, iters);
      cudaEventRecord(evStop, stream);
      cudaError_t e = cudaGetLastError();
      if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
      if (e != cudaSuccess) {
        out->exitCode = kExitError;
        out->reason = std::string("calibration launch failed: ") + cudaGetErrorString(e);
        cleanup();
        return;
      }
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, evStart, evStop) != cudaSuccess || ms <= 0.0f) break;
      const double scaled = iters * (kLaunchSeconds * 1000.0) / ms;
      iters = static_cast<int>(std::max(1024.0, std::min(scaled, 2.0e6)));
    }
  }
  const double flopsPerLaunch = totalThreads * iters * kFlopsPerIterPerThread;
  const int launchesPerWindow = std::max(1, static_cast<int>(kWindowSeconds / kLaunchSeconds));
  out->loadThreads = totalThreads;
  out->loadItersPerLaunch = iters;

  // Warm-up: an unwarmed part sits at idle clocks legitimately, so sampling
  // must not begin until the load has had time to bring it up. Derived from
  // THIS device's own window, matching the single-device engine's original
  // proportions.
  long warmupSeconds = std::max(2L, std::min(10L, windowSecondsTotal / 5));
  if (warmupSeconds >= windowSecondsTotal) warmupSeconds = windowSecondsTotal / 2;
  out->warmupSeconds = warmupSeconds;
  out->sampleWindowSeconds = windowSecondsTotal - warmupSeconds;

  SamplerState ok;
  double totalFlops = 0.0, totalWindowSeconds = 0.0, peakWindowTflops = 0.0;
  long launches = 0;
  std::string runErr;

  auto runPhase = [&](double deadline, Samples *samples) -> bool {
    while (nowSeconds() < deadline) {
      cudaEventRecord(evStart, stream);
      for (int i = 0; i < launchesPerWindow; ++i) {
        fmaLoadKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(nullptr, iters);
      }
      cudaEventRecord(evStop, stream);
      const cudaError_t launchErr = cudaGetLastError();
      if (launchErr != cudaSuccess) {
        runErr = std::string("load launch failed: ") + cudaGetErrorString(launchErr);
        return false;
      }
      for (;;) {
        const cudaError_t q = cudaStreamQuery(stream);
        if (q == cudaSuccess) break;
        if (q != cudaErrorNotReady) {
          runErr = std::string("load kernel failed: ") + cudaGetErrorString(q);
          return false;
        }
        if (samples != nullptr) takeSample(nvml, nvdev, samples, &ok, &out->unsupportedReads);
        sleepMillis(sampleIntervalMs);
      }
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, evStart, evStop) != cudaSuccess || ms <= 0.0f) continue;
      if (samples != nullptr) {
        const double windowFlops = flopsPerLaunch * launchesPerWindow;
        const double seconds = ms / 1000.0;
        totalFlops += windowFlops;
        totalWindowSeconds += seconds;
        launches += launchesPerWindow;
        peakWindowTflops = std::max(peakWindowTflops, windowFlops / seconds / 1e12);
      }
    }
    return true;
  };

  const double started = nowSeconds();
  if (!runPhase(started + warmupSeconds, nullptr)) {
    out->exitCode = kExitError;
    out->reason = runErr;
    cleanup();
    return;
  }

  Samples s;
  const bool loadOK = runPhase(started + windowSecondsTotal, &s);
  out->elapsedS = nowSeconds() - started;
  out->samplesTaken = s.n;
  out->loadLaunches = launches;

  if (!loadOK) {
    out->exitCode = kExitError;
    out->reason = runErr;
    cleanup();
    return;
  }
  if (s.n == 0) {
    out->exitCode = kExitError;
    out->reason = "no clock samples were taken; the probe measured nothing";
    cleanup();
    return;
  }

  const double meanSm = s.smSum / s.n;
  const double sustainedClockPct = 100.0 * meanSm / ratedBoostMHz;
  out->smClockMHz = meanSm;
  out->values["sustained_clock_pct"] = sustainedClockPct;
  out->values["min_sm_clock_pct"] = 100.0 * s.smMin / ratedBoostMHz;
  out->maxSmClockPct = 100.0 * s.smMax / ratedBoostMHz;
  out->tempAtMinSmKnown = s.tempAtMinSmKnown;
  out->tempAtMinSm = s.tempAtMinSm;
  if (s.memN > 0) {
    out->memClockKnown = true;
    out->memClockMHz = s.memSum / s.memN;
  }

  out->tempKnown = s.tempN > 0;
  const double meanTemp = s.mean(s.tempSum, s.tempN);
  out->meanTemp = meanTemp;
  if (out->tempKnown) out->values["gpu_temp_c"] = s.tempMax;

  out->powerKnown = s.powerN > 0;
  if (out->powerKnown) {
    out->meanPower = s.mean(s.powerSum, s.powerN);
    out->peakPower = s.powerMax;
  }

  out->utilKnown = s.utilN > 0;
  const double meanUtil = s.mean(s.utilSum, s.utilN);
  out->meanUtil = meanUtil;
  out->meanMemUtil = s.mean(s.memUtilSum, s.utilN);

  if (totalWindowSeconds > 0.0) {
    out->tflopsKnown = true;
    out->sustainedTflops = totalFlops / totalWindowSeconds / 1e12;
    out->peakTflops = peakWindowTflops;
  }

  out->reasonsKnown = s.reasonsN > 0;
  if (out->reasonsKnown) {
    out->values["throttle_events"] = static_cast<double>(s.throttleEvents);
    out->values["throttled_samples"] = static_cast<double>(s.throttledSamples);
    out->reasonMask = s.reasonMask;
    for (int i = 0; i < kNumReasonBits; ++i) out->reasonCount[i] = s.reasonCount[i];
  }

  // ── classify (per device) ────────────────────────────────────────────────
  const bool shortfall = sustainedClockPct < clockFloorPct;
  const bool thermalLatched = out->reasonsKnown && (s.reasonMask & kThermalReasons) != 0;
  const bool powerLatched = out->reasonsKnown && (s.reasonMask & kPowerReasons) != 0;
  const bool appClocksLatched =
      out->reasonsKnown && (s.reasonMask & nvmlrt::kReasonApplicationsClocksSetting) != 0;
  const bool hot = out->tempKnown && meanTemp >= thermalTempC;

  if (shortfall) {
    if (hot && thermalLatched) {
      out->classification = "thermal";
    } else if (powerLatched || limitBelowDefault) {
      out->classification = "powerCap";
    } else if (appClocksLatched) {
      out->classification = "applicationClocks";
    } else {
      out->classification = "unknown";
    }
  }

  const bool pdWedgeSuspected = shortfall && out->tempKnown && !hot && !thermalLatched;
  if (shortfall && !out->tempKnown) {
    out->pdWedge = "unknown";
  } else if (pdWedgeSuspected) {
    out->pdWedge = "true";
  }

  out->floorApplied = clockFloorPct;
  if (shortfall && hot && thermalLatched) {
    out->floorApplied = std::min(clockFloorPct, thermalClockFloorPct);
    out->floorBasis = "thermal";
  }

  cleanup();

  // ── decide (per device) ──────────────────────────────────────────────────
  if (out->utilKnown && meanUtil < kMinCredibleUtilizationPct) {
    out->exitCode = kExitError;
    out->reason = "device utilization averaged " + std::to_string(static_cast<int>(meanUtil)) +
                 "% under load; the sampled clock is not attributable to this test";
    return;
  }
  if (sustainedClockPct >= out->floorApplied) {
    out->exitCode = kExitPass;
    return;
  }
  char reason[512];
  if (std::strcmp(out->floorBasis, "thermal") == 0) {
    std::snprintf(reason, sizeof(reason),
                  "device %d: sustained SM clock %.1f%% of rated boost, below the %.1f%% thermal "
                  "floor; attributed to heat (mean %.1fC, thermal throttle latched)",
                  index, sustainedClockPct, out->floorApplied, meanTemp);
  } else if (pdWedgeSuspected) {
    std::snprintf(reason, sizeof(reason),
                  "device %d: sustained SM clock %.1f%% of rated boost at only %.1fC with no "
                  "thermal throttle — power-delivery wedge suspected (classification=%s)",
                  index, sustainedClockPct, meanTemp, out->classification);
  } else {
    std::snprintf(reason, sizeof(reason),
                  "device %d: sustained SM clock %.1f%% of rated boost, below the %.1f%% floor "
                  "(classification=%s)",
                  index, sustainedClockPct, out->floorApplied, out->classification);
  }
  out->exitCode = kExitFail;
  out->reason = reason;
}

devices::DeviceReport toDeviceReport(const DeviceResult &r) {
  devices::DeviceReport rep;
  rep.index = r.index;
  rep.busId = r.busId;
  rep.name = r.name;
  rep.computeCap = r.computeCap;
  rep.identityRead = r.identityRead;
  rep.values = r.values;
  return rep;
}

} // namespace

int main() {
  // ── configuration ─────────────────────────────────────────────────────────
  std::string cfgErr;
  long durationSeconds = 0;
  long sampleIntervalMs = 0;
  double clockFloorPct = 0.0;
  double thermalClockFloorPct = 0.0;
  double thermalTempC = 0.0;
  if (!envLong("BURNIN_DURATION_SECONDS", kDefaultDurationSeconds, &durationSeconds, &cfgErr) ||
      !envLong("CLOCKPROBE_SAMPLE_INTERVAL_MS", 100, &sampleIntervalMs, &cfgErr) ||
      !envDouble("CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT", 70.0, &clockFloorPct, &cfgErr) ||
      !envDouble("CLOCKPROBE_MIN_THERMAL_CLOCK_PCT", 50.0, &thermalClockFloorPct, &cfgErr) ||
      !envDouble("CLOCKPROBE_THERMAL_TEMP_C", 80.0, &thermalTempC, &cfgErr)) {
    return errored(cfgErr);
  }
  const long durationRequested = durationSeconds;
  if (durationSeconds < kMinDurationSeconds) {
    warn("BURNIN_DURATION_SECONDS raised to the " + std::to_string(kMinDurationSeconds) +
         "s floor; a shorter probe cannot separate clock ramp from steady state");
    durationSeconds = kMinDurationSeconds;
  }
  sampleIntervalMs = std::max(10L, std::min(sampleIntervalMs, 5000L));
  if (thermalClockFloorPct > clockFloorPct) {
    warn("CLOCKPROBE_MIN_THERMAL_CLOCK_PCT exceeds CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT; clamped");
    thermalClockFloorPct = clockFloorPct;
  }

  std::printf("duration_requested_s=%ld\n", durationRequested);
  std::printf("clock_floor_pct=%.2f\n", clockFloorPct);
  std::printf("thermal_clock_floor_pct=%.2f\n", thermalClockFloorPct);
  std::printf("thermal_temp_threshold_c=%.2f\n", thermalTempC);

  // ── how many devices, and how ────────────────────────────────────────────
  int visible = 0;
  const cudaError_t countErr = cudaGetDeviceCount(&visible);
  if (countErr != cudaSuccess && countErr != cudaErrorNoDevice) {
    return errored(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(countErr));
  }
  if (countErr == cudaErrorNoDevice) visible = 0;

  const devices::Budget budget =
      devices::parseBudget(std::getenv("BURNIN_RESOURCE_LIMITS"), devices::nvidiaResources());
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

  // ── NVML, opened ONCE for the whole process ─────────────────────────────
  nvmlrt::Library nvml;
  std::string nvmlErr;
  if (!nvml.open(&nvmlErr)) {
    emitCommon();
    return errored(nvmlErr + " (an accelerator is visible, so this is a container/driver "
                             "misconfiguration: NVIDIA_DRIVER_CAPABILITIES must include 'utility')");
  }
  nvmlrt::Return rc = nvml.init();
  if (rc != nvmlrt::kSuccess) {
    emitCommon();
    nvml.close();
    return errored(std::string("nvmlInit: ") + nvml.errorString(rc));
  }
  if (nvml.systemGetDriverVersion != nullptr) {
    char driver[96] = {0};
    if (nvml.systemGetDriverVersion(driver, sizeof(driver)) == nvmlrt::kSuccess) {
      std::printf("driver_version=%s\n", driver);
    }
  }

  // ── run every device ─────────────────────────────────────────────────────
  std::vector<DeviceResult> results(planCount);
  if (conc.mode == devices::Concurrency::All) {
    std::vector<std::thread> threads;
    threads.reserve(planCount);
    for (int i = 0; i < planCount; ++i) {
      threads.emplace_back(runOneDevice, i, windowS, clockFloorPct, thermalClockFloorPct,
                           thermalTempC, sampleIntervalMs, std::cref(nvml), &results[i]);
    }
    for (auto &t : threads) t.join();
  } else {
    for (int i = 0; i < planCount; ++i) {
      runOneDevice(i, windowS, clockFloorPct, thermalClockFloorPct, thermalTempC, sampleIntervalMs,
                  nvml, &results[i]);
    }
  }
  nvml.close();

  // Merge every device's own warnings/unsupported-reads into the process
  // globals, single-threaded (every worker has already joined).
  for (auto &r : results) {
    if (!r.configWarnings.empty()) warn(r.configWarnings);
    for (size_t pos = 0; pos < r.unsupportedReads.size();) {
      size_t comma = r.unsupportedReads.find(',', pos);
      if (comma == std::string::npos) comma = r.unsupportedReads.size();
      noteUnsupported(r.unsupportedReads.substr(pos, comma - pos).c_str());
      pos = comma + 1;
    }
  }

  // Device 0's identity, printed once — see device_fold.h's design note on
  // why identity keys keep the first device's meaning.
  if (!results.empty() && results.front().identityRead) {
    const DeviceResult &d0 = results.front();
    std::printf("gpu_name=%s\n", d0.name.c_str());
    std::printf("compute_cap=%s\n", d0.computeCap.c_str());
    if (!d0.busId.empty()) std::printf("pci_bus_id=%s\n", d0.busId.c_str());
    if (d0.ratedKnown) std::printf("rated_boost_clock_mhz=%u\n", d0.ratedBoostMHz);
  }

  // ── fold and report ──────────────────────────────────────────────────────
  std::vector<devices::DeviceReport> reports;
  for (auto &r : results) {
    if (r.values.empty()) continue; // never measured (setup failed before sampling)
    reports.push_back(toDeviceReport(r));
  }
  static const std::vector<devices::FoldRule> kDeviceFold = {
      {"sustained_clock_pct", devices::Fold::Min},
      {"min_sm_clock_pct", devices::Fold::Min},
      {"gpu_temp_c", devices::Fold::Max},
      {"throttled_samples", devices::Fold::Sum},
      {"throttle_events", devices::Fold::Sum},
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
  if (auto it = folded.values.find("throttled_samples"); it != folded.values.end()) {
    std::printf("throttled_samples=%.0f\n", it->second);
  }
  if (auto it = folded.values.find("throttle_events"); it != folded.values.end()) {
    std::printf("throttle_events=%.0f\n", it->second);
  }

  // Evidence, device 0's — mirrors the soak family's "Once" fields, and every
  // one of these was already unconditional single-device output.
  if (!results.empty()) {
    const DeviceResult &d0 = results.front();
    std::printf("elapsed_s=%.2f\n", d0.elapsedS);
    std::printf("samples_taken=%ld\n", d0.samplesTaken);
    std::printf("load_launches=%ld\n", d0.loadLaunches);
    std::printf("load_threads=%.0f\n", d0.loadThreads);
    std::printf("load_iters_per_launch=%d\n", d0.loadItersPerLaunch);
    std::printf("warmup_s=%ld\n", d0.warmupSeconds);
    std::printf("sample_window_s=%ld\n", d0.sampleWindowSeconds);
    if (d0.ratedKnown) {
      std::printf("sm_clock_mhz=%.0f\n", d0.smClockMHz);
      std::printf("max_sm_clock_pct=%.2f\n", d0.maxSmClockPct);
    }
    if (d0.memClockKnown) std::printf("mem_clock_mhz=%.0f\n", d0.memClockMHz);
    if (d0.tempKnown) {
      std::printf("mean_temp_under_load_c=%.1f\n", d0.meanTemp);
      if (d0.tempAtMinSmKnown) std::printf("temp_at_min_clock_c=%.1f\n", d0.tempAtMinSm);
    }
    if (d0.haveEnforcedLimit) std::printf("enforced_power_limit_w=%.2f\n", d0.enforcedLimitW);
    if (d0.haveDefaultLimit) std::printf("default_power_limit_w=%.2f\n", d0.defaultLimitW);
    if (d0.haveEnforcedLimit && d0.haveDefaultLimit && d0.defaultLimitW > 0.0) {
      std::printf("power_limit_ratio_pct=%.2f\n", 100.0 * d0.enforcedLimitW / d0.defaultLimitW);
    }
    if (d0.powerKnown) {
      std::printf("power_draw_w=%.2f\n", d0.peakPower);
      std::printf("mean_power_w=%.2f\n", d0.meanPower);
    }
    if (d0.utilKnown) {
      std::printf("gpu_utilization_pct=%.2f\n", d0.meanUtil);
      std::printf("mem_utilization_pct=%.2f\n", d0.meanMemUtil);
    }
    if (d0.tflopsKnown) {
      std::printf("sustained_fma_throughput_tflops=%.3f\n", d0.sustainedTflops);
      std::printf("peak_fma_throughput_tflops=%.3f\n", d0.peakTflops);
      if (d0.peakTflops > 0.0) {
        std::printf("throughput_consistency_pct=%.2f\n", 100.0 * d0.sustainedTflops / d0.peakTflops);
      }
    }
    if (d0.reasonsKnown) {
      std::printf("throttle_reasons_mask=%llu\n", d0.reasonMask);
      std::string labels;
      for (int i = 0; i < kNumReasonBits; ++i) {
        std::printf("%s=%ld\n", kReasonBits[i].key, d0.reasonCount[i]);
        if ((d0.reasonMask & kReasonBits[i].bit) != 0) {
          if (!labels.empty()) labels += ",";
          labels += kReasonBits[i].label;
        }
      }
      std::printf("throttle_reasons=%s\n", labels.empty() ? "none" : labels.c_str());
    }
    std::printf("throttle_classification=%s\n", d0.classification);
    std::printf("pd_wedge_suspected=%s\n", d0.pdWedge);
    std::printf("clock_floor_applied_pct=%.2f\n", d0.floorApplied);
    std::printf("clock_floor_basis=%s\n", d0.floorBasis);
  }

  const std::vector<devices::SpreadSpec> spreads = {
      {"sustainedClockSpreadPct", "sustained_clock_pct", /*absoluteFigure=*/false},
      {"fmaThroughputSpreadPct", "sustained_fma_throughput_tflops", /*absoluteFigure=*/true},
  };
  devices::printFold(stdout, reports, visible, windowS, conc.mode, folded, spreads, /*underMig=*/false);
  if (reports.size() > 1) {
    std::fputs(devices::renderPerDeviceArtifact(reports).c_str(), stdout);
  }

  emitCommon();

  // ── combine per-device outcomes into the machinery/verdict exit code ─────
  std::vector<int> codes;
  for (auto &r : results) codes.push_back(r.exitCode);
  const int combined = devices::combineExitCodes(codes);

  if (combined == kExitPass) return emitMarker("CLOCKPROBE_PASS", "", kExitPass);

  // The message names the device(s) that decided it — a fold over eight
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

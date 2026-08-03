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
// Nothing else in the suite sees it. compute-smoke passes (the arithmetic is
// still correct, just slower). thermal-soak passes (a slow part runs cool, so
// no thermal trip). dcgm-diag passes (nothing is faulty). Every one of those
// asserts correctness or health; none of them asserts SPEED. A fleet with this
// fault delivers correct results, slowly, forever, and the loss shows up only
// in the training-job wall-clock — usually months later.
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
#include <string>

#include "nvml_dynamic.h"

namespace {

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

std::string configWarnings;

void warn(const std::string &s) {
  if (!configWarnings.empty()) configWarnings += "; ";
  configWarnings += s;
}

// A list of NVML reads the driver refused. Recorded rather than substituted:
// emitting a sentinel for an unsupported read would be a fabricated number that
// a threshold could not tell from a measured one.
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

// takeSample reads one instant of the device's state. A read the driver refuses
// disables that field for the rest of the run rather than being retried every
// sample; NOT_SUPPORTED does not become supported mid-probe.
void takeSample(const nvmlrt::Library &nvml, nvmlrt::Device dev, Samples *s, bool *memOK,
                bool *tempOK, bool *powerOK, bool *utilOK, bool *reasonsOK) {
  unsigned int sm = 0;
  if (nvml.deviceGetClockInfo(dev, nvmlrt::kClockSM, &sm) != nvmlrt::kSuccess) return;

  s->n++;
  s->smSum += sm;
  s->smMax = std::max(s->smMax, sm);

  double temp = 0.0;
  bool tempKnown = false;
  if (*tempOK && nvml.deviceGetTemperature != nullptr) {
    unsigned int t = 0;
    if (nvml.deviceGetTemperature(dev, nvmlrt::kTemperatureGpu, &t) == nvmlrt::kSuccess) {
      temp = t;
      tempKnown = true;
      s->tempN++;
      s->tempSum += temp;
      s->tempMax = std::max(s->tempMax, temp);
    } else {
      *tempOK = false;
      noteUnsupported("temperature");
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

  if (*memOK) {
    unsigned int mem = 0;
    if (nvml.deviceGetClockInfo(dev, nvmlrt::kClockMem, &mem) == nvmlrt::kSuccess) {
      s->memN++;
      s->memSum += mem;
    } else {
      *memOK = false;
      noteUnsupported("memClock");
    }
  }

  if (*powerOK && nvml.deviceGetPowerUsage != nullptr) {
    unsigned int mw = 0;
    if (nvml.deviceGetPowerUsage(dev, &mw) == nvmlrt::kSuccess) {
      const double w = mw / 1000.0;
      s->powerN++;
      s->powerSum += w;
      s->powerMax = std::max(s->powerMax, w);
    } else {
      *powerOK = false;
      noteUnsupported("powerDraw");
    }
  }

  if (*utilOK && nvml.deviceGetUtilizationRates != nullptr) {
    nvmlrt::Utilization u{0, 0};
    if (nvml.deviceGetUtilizationRates(dev, &u) == nvmlrt::kSuccess) {
      s->utilN++;
      s->utilSum += u.gpu;
      s->memUtilSum += u.memory;
    } else {
      *utilOK = false;
      noteUnsupported("utilization");
    }
  }

  if (*reasonsOK && nvml.deviceGetCurrentClocksThrottleReasons != nullptr) {
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
      *reasonsOK = false;
      noteUnsupported("throttleReasons");
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
  // A thermal allowance must never be STRICTER than the general floor: it exists
  // to excuse an expected shortfall, not to invent a second way to fail.
  if (thermalClockFloorPct > clockFloorPct) {
    warn("CLOCKPROBE_MIN_THERMAL_CLOCK_PCT exceeds CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT; clamped");
    thermalClockFloorPct = clockFloorPct;
  }

  // Warm-up: an unwarmed part sits at idle clocks legitimately, so sampling
  // must not begin until the load has had time to bring it up.
  long warmupSeconds = std::max(2L, std::min(10L, durationSeconds / 5));
  if (warmupSeconds >= durationSeconds) warmupSeconds = durationSeconds / 2;
  const long windowSeconds = durationSeconds - warmupSeconds;

  // ── device ────────────────────────────────────────────────────────────────
  int deviceCount = 0;
  const cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
  if (countErr == cudaErrorNoDevice || (countErr == cudaSuccess && deviceCount == 0)) {
    // Not applicable rather than broken: a node with no accelerator has nothing
    // whose clock could be probed.
    return skip("no accelerator visible to this container");
  }
  if (countErr != cudaSuccess) {
    return errored(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(countErr));
  }

  int device = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    return errored("could not read CUDA device properties");
  }
  char busId[32] = {0};
  const bool busIdOK = cudaDeviceGetPCIBusId(busId, sizeof(busId), device) == cudaSuccess;

  std::printf("gpu_name=%s\n", props.name);
  std::printf("compute_cap=%d.%d\n", props.major, props.minor);
  if (busIdOK) std::printf("pci_bus_id=%s\n", busId);
  std::printf("duration_requested_s=%ld\n", durationRequested);
  std::printf("warmup_s=%ld\n", warmupSeconds);
  std::printf("sample_window_s=%ld\n", windowSeconds);
  std::printf("clock_floor_pct=%.2f\n", clockFloorPct);
  std::printf("thermal_clock_floor_pct=%.2f\n", thermalClockFloorPct);
  std::printf("thermal_temp_threshold_c=%.2f\n", thermalTempC);

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
    return errored(nvmlErr + " (an accelerator is visible, so this is a container/driver "
                             "misconfiguration: NVIDIA_DRIVER_CAPABILITIES must include 'utility')");
  }
  nvmlrt::Return rc = nvml.init();
  if (rc != nvmlrt::kSuccess) {
    emitCommon();
    nvml.close();
    return errored(std::string("nvmlInit: ") + nvml.errorString(rc));
  }

  nvmlrt::Device nvdev = nullptr;
  bool haveHandle = false;
  // Prefer the PCI bus id: NVML index order and CUDA ordinal order are not
  // guaranteed to agree, and probing the clock of a different GPU than the one
  // under load would produce a confident, wrong verdict.
  if (busIdOK && nvml.deviceGetHandleByPciBusId != nullptr) {
    haveHandle = nvml.deviceGetHandleByPciBusId(busId, &nvdev) == nvmlrt::kSuccess;
  }
  if (!haveHandle) {
    warn("NVML handle resolved by index, not PCI bus id");
    rc = nvml.deviceGetHandleByIndex(0, &nvdev);
    if (rc != nvmlrt::kSuccess) {
      emitCommon();
      nvml.close();
      return errored(std::string("nvmlDeviceGetHandleByIndex: ") + nvml.errorString(rc));
    }
  }

  if (nvml.systemGetDriverVersion != nullptr) {
    char driver[96] = {0};
    if (nvml.systemGetDriverVersion(driver, sizeof(driver)) == nvmlrt::kSuccess) {
      std::printf("driver_version=%s\n", driver);
    }
  }

  // MIG: clocks are a whole-device property, so a clock sampled from inside a
  // MIG instance cannot be attributed to this instance's load. Unjudgeable on
  // this hardware configuration, which is a Skip.
  if (nvml.deviceGetMigMode != nullptr) {
    unsigned int current = 0, pending = 0;
    if (nvml.deviceGetMigMode(nvdev, &current, &pending) == nvmlrt::kSuccess) {
      std::printf("mig_mode=%s\n", current == nvmlrt::kMigEnable ? "enabled" : "disabled");
      if (current == nvmlrt::kMigEnable) {
        emitCommon();
        nvml.close();
        return skip("MIG is enabled; SM clock is a device-wide property and cannot be "
                    "attributed to one instance's load");
      }
    }
  }

  // The denominator. Without a rated clock there is no portable percentage, and
  // a missing nameplate is an UNJUDGED part rather than a slow one — so this is
  // a Skip, exactly as the TestKind's documentation requires.
  unsigned int ratedBoostMHz = 0;
  rc = nvml.deviceGetMaxClockInfo(nvdev, nvmlrt::kClockSM, &ratedBoostMHz);
  if (rc != nvmlrt::kSuccess || ratedBoostMHz == 0) {
    emitCommon();
    nvml.close();
    return skip(std::string("rated SM boost clock unavailable (") + nvml.errorString(rc) +
                "); without a nameplate denominator the part is unjudged, not slow");
  }
  std::printf("rated_boost_clock_mhz=%u\n", ratedBoostMHz);

  // Power limits are the PD contract made visible: on a wedged GB10 the enforced
  // limit sits well below the board's default limit.
  double enforcedLimitW = 0.0, defaultLimitW = 0.0;
  bool haveEnforced = false, haveDefault = false;
  if (nvml.deviceGetEnforcedPowerLimit != nullptr) {
    unsigned int mw = 0;
    if (nvml.deviceGetEnforcedPowerLimit(nvdev, &mw) == nvmlrt::kSuccess) {
      enforcedLimitW = mw / 1000.0;
      haveEnforced = true;
      std::printf("enforced_power_limit_w=%.2f\n", enforcedLimitW);
    } else {
      noteUnsupported("enforcedPowerLimit");
    }
  }
  if (nvml.deviceGetPowerManagementDefaultLimit != nullptr) {
    unsigned int mw = 0;
    if (nvml.deviceGetPowerManagementDefaultLimit(nvdev, &mw) == nvmlrt::kSuccess) {
      defaultLimitW = mw / 1000.0;
      haveDefault = true;
      std::printf("default_power_limit_w=%.2f\n", defaultLimitW);
    } else {
      noteUnsupported("defaultPowerLimit");
    }
  }
  const bool limitBelowDefault =
      haveEnforced && haveDefault && defaultLimitW > 0.0 && enforcedLimitW < defaultLimitW * 0.95;
  if (haveEnforced && haveDefault && defaultLimitW > 0.0) {
    std::printf("power_limit_ratio_pct=%.2f\n", 100.0 * enforcedLimitW / defaultLimitW);
  }

  // ── build the load ────────────────────────────────────────────────────────
  const int blocks = std::max(1, props.multiProcessorCount * kBlocksPerSM);
  const double totalThreads = static_cast<double>(blocks) * kThreadsPerBlock;

  cudaStream_t stream = nullptr;
  cudaEvent_t evStart = nullptr, evStop = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess || cudaEventCreate(&evStart) != cudaSuccess ||
      cudaEventCreate(&evStop) != cudaSuccess) {
    emitCommon();
    nvml.close();
    return errored("could not create CUDA stream/events");
  }

  // Calibrate: how many loop iterations make one launch take ~kLaunchSeconds?
  // Done on the live part so the load is sized for the clock it actually runs
  // at, including a clock that is already wedged low.
  //
  // The first launch of the process pays for CUDA context creation and module
  // load, which is tens to hundreds of milliseconds and has nothing to do with
  // the kernel. Timing it would size the load far too small, the windows would
  // be short, and the probe would take few samples. So: one throwaway launch to
  // pay that cost, then measure, then refine once against the measured rate.
  int iters = 4096;
  {
    // A launch-configuration error is reported synchronously by
    // cudaGetLastError, not by the later synchronise. Checking only the
    // synchronise would let a kernel that never ran look like a completed load,
    // and the idle clock that follows would be blamed on the hardware.
    fmaLoadKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(nullptr, 1024);
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr == cudaSuccess) launchErr = cudaStreamSynchronize(stream);
    if (launchErr != cudaSuccess) {
      emitCommon();
      nvml.close();
      return errored(std::string("initial launch failed: ") + cudaGetErrorString(launchErr));
    }
    for (int round = 0; round < 2; ++round) {
      cudaEventRecord(evStart, stream);
      fmaLoadKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(nullptr, iters);
      cudaEventRecord(evStop, stream);
      cudaError_t e = cudaGetLastError();
      if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
      if (e != cudaSuccess) {
        emitCommon();
        nvml.close();
        return errored(std::string("calibration launch failed: ") + cudaGetErrorString(e));
      }
      float ms = 0.0f;
      if (cudaEventElapsedTime(&ms, evStart, evStop) != cudaSuccess || ms <= 0.0f) break;
      const double scaled = iters * (kLaunchSeconds * 1000.0) / ms;
      // Bounded on both sides: too few iterations makes launch overhead the
      // measurement, too many makes a single launch outrun the pod's deadline.
      iters = static_cast<int>(std::max(1024.0, std::min(scaled, 2.0e6)));
    }
  }
  const double flopsPerLaunch = totalThreads * iters * kFlopsPerIterPerThread;
  const int launchesPerWindow =
      std::max(1, static_cast<int>(kWindowSeconds / kLaunchSeconds));

  std::printf("load_threads=%.0f\n", totalThreads);
  std::printf("load_iters_per_launch=%d\n", iters);

  // runPhase drives the load until deadline, sampling only while the device is
  // demonstrably busy. samples==nullptr is the warm-up: same load, no evidence.
  bool memOK = true, tempOK = true, powerOK = true, utilOK = true, reasonsOK = true;
  double totalFlops = 0.0;
  double totalWindowSeconds = 0.0;
  double peakWindowTflops = 0.0;
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
        if (samples != nullptr) takeSample(nvml, nvdev, samples, &memOK, &tempOK, &powerOK, &utilOK,
                                           &reasonsOK);
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
    emitCommon();
    nvml.close();
    return errored(runErr);
  }

  Samples s;
  const bool loadOK = runPhase(started + durationSeconds, &s);
  const double elapsed = nowSeconds() - started;

  // ── derive ────────────────────────────────────────────────────────────────
  std::printf("elapsed_s=%.2f\n", elapsed);
  std::printf("samples_taken=%ld\n", s.n);
  std::printf("load_launches=%ld\n", launches);

  if (!loadOK) {
    emitCommon();
    nvml.close();
    return errored(runErr);
  }
  if (s.n == 0) {
    emitCommon();
    nvml.close();
    return errored("no clock samples were taken; the probe measured nothing");
  }

  const double meanSm = s.smSum / s.n;
  const double sustainedClockPct = 100.0 * meanSm / ratedBoostMHz;
  std::printf("sm_clock_mhz=%.0f\n", meanSm);
  std::printf("sustained_clock_pct=%.2f\n", sustainedClockPct);
  // Reported as percentages, not MHz: only the three aliased *_mhz keys are
  // mapped to the MHz unit by pkg/runner, and an unaliased "*_mhz" key would
  // normalise to a name that declares no unit — a clock stored as a bare number.
  std::printf("min_sm_clock_pct=%.2f\n", 100.0 * s.smMin / ratedBoostMHz);
  std::printf("max_sm_clock_pct=%.2f\n", 100.0 * s.smMax / ratedBoostMHz);
  if (s.memN > 0) std::printf("mem_clock_mhz=%.0f\n", s.memSum / s.memN);

  const bool tempKnown = s.tempN > 0;
  const double meanTemp = s.mean(s.tempSum, s.tempN);
  if (tempKnown) {
    std::printf("gpu_temp_c=%.1f\n", s.tempMax);
    std::printf("mean_temp_under_load_c=%.1f\n", meanTemp);
    if (s.tempAtMinSmKnown) std::printf("temp_at_min_clock_c=%.1f\n", s.tempAtMinSm);
  }
  if (s.powerN > 0) {
    std::printf("power_draw_w=%.2f\n", s.powerMax);
    std::printf("mean_power_w=%.2f\n", s.mean(s.powerSum, s.powerN));
  }

  const bool utilKnown = s.utilN > 0;
  const double meanUtil = s.mean(s.utilSum, s.utilN);
  if (utilKnown) {
    // The headline symptom of the wedge: this reads healthy while the clock does
    // not. Emitting both side by side is what makes the fault legible.
    std::printf("gpu_utilization_pct=%.2f\n", meanUtil);
    std::printf("mem_utilization_pct=%.2f\n", s.mean(s.memUtilSum, s.utilN));
  }

  // NOT "sustained_throughput_tflops". That canonical name belongs to gpu-burn's
  // heavy GEMM burn, and this is a synthetic register-resident FMA chain with no
  // memory traffic — the two differ by a large factor on the same healthy part.
  // pkg/contract's own rule is that sharing a name across two differently-obtained
  // figures is the failure the convention exists to prevent: a profile author
  // whose threshold was calibrated on gpu-burn would silently condemn every node
  // this runner touched, and the stored series would interleave two quantities.
  double sustainedTflops = 0.0;
  if (totalWindowSeconds > 0.0) {
    sustainedTflops = totalFlops / totalWindowSeconds / 1e12;
    std::printf("sustained_fma_throughput_tflops=%.3f\n", sustainedTflops);
    std::printf("peak_fma_throughput_tflops=%.3f\n", peakWindowTflops);
    if (peakWindowTflops > 0.0) {
      // Sustained against best window. A part that starts fast and collapses
      // scores low here even when its mean clock still clears the floor.
      std::printf("throughput_consistency_pct=%.2f\n", 100.0 * sustainedTflops / peakWindowTflops);
    }
  }

  // Throttle counters are emitted ONLY when the driver actually answered. A
  // fabricated throttleEvents=0 would satisfy a `throttleEvents == 0` threshold
  // on a node whose throttle state was never read — a missing measurement
  // silently passing acceptance, which is exactly what the fails-closed rule
  // forbids. Omitting the metric makes such a threshold fail instead.
  const bool reasonsKnown = s.reasonsN > 0;
  if (reasonsKnown) {
    std::printf("throttle_events=%ld\n", s.throttleEvents);
    std::printf("throttled_samples=%ld\n", s.throttledSamples);
    std::printf("throttle_reasons_mask=%llu\n", s.reasonMask);
    std::string labels;
    for (int i = 0; i < kNumReasonBits; ++i) {
      std::printf("%s=%ld\n", kReasonBits[i].key, s.reasonCount[i]);
      if ((s.reasonMask & kReasonBits[i].bit) != 0) {
        if (!labels.empty()) labels += ",";
        labels += kReasonBits[i].label;
      }
    }
    std::printf("throttle_reasons=%s\n", labels.empty() ? "none" : labels.c_str());
  }

  // ── classify ──────────────────────────────────────────────────────────────
  const bool shortfall = sustainedClockPct < clockFloorPct;
  const bool thermalLatched = reasonsKnown && (s.reasonMask & kThermalReasons) != 0;
  const bool powerLatched = reasonsKnown && (s.reasonMask & kPowerReasons) != 0;
  const bool appClocksLatched = reasonsKnown && (s.reasonMask & nvmlrt::kReasonApplicationsClocksSetting) != 0;
  // Unknown temperature is NOT hot. Fails closed: an unread sensor must not buy
  // the lenient thermal floor.
  const bool hot = tempKnown && meanTemp >= thermalTempC;

  const char *classification = "none";
  if (shortfall) {
    if (hot && thermalLatched) {
      classification = "thermal";
    } else if (powerLatched || limitBelowDefault) {
      classification = "powerCap";
    } else if (appClocksLatched) {
      classification = "applicationClocks";
    } else {
      classification = "unknown";
    }
  }
  std::printf("throttle_classification=%s\n", classification);

  // The wedge signature: slow, cool, and not thermally throttled.
  //
  // Tri-state, not a bool. "false" has to mean "we looked and it is not this",
  // which is a different claim from "we could not tell" — and without a
  // temperature we genuinely cannot tell a PD wedge from a thermal throttle.
  // Collapsing the two would let an unread sensor read as an all-clear.
  const bool pdWedgeSuspected = shortfall && tempKnown && !hot && !thermalLatched;
  const char *pdWedge = "false";
  if (shortfall && !tempKnown) {
    pdWedge = "unknown";
  } else if (pdWedgeSuspected) {
    pdWedge = "true";
  }
  std::printf("pd_wedge_suspected=%s\n", pdWedge);

  double floorApplied = clockFloorPct;
  const char *floorBasis = "general";
  if (shortfall && hot && thermalLatched) {
    floorApplied = std::min(clockFloorPct, thermalClockFloorPct);
    floorBasis = "thermal";
  }
  std::printf("clock_floor_applied_pct=%.2f\n", floorApplied);
  std::printf("clock_floor_basis=%s\n", floorBasis);

  emitCommon();
  nvml.close();

  // ── decide ────────────────────────────────────────────────────────────────
  // Everything above is already on stdout, so every branch from here leaves a
  // complete record behind it.

  // A load that did not land makes the clock unattributable. Refusing to judge
  // is the honest answer; calling it a hardware failure would condemn a node
  // for the runner's own problem.
  if (utilKnown && meanUtil < kMinCredibleUtilizationPct) {
    return errored("device utilization averaged " + std::to_string(static_cast<int>(meanUtil)) +
                   "% under load; the sampled clock is not attributable to this test");
  }

  char reason[512];
  if (sustainedClockPct >= floorApplied) {
    return emitMarker("CLOCKPROBE_PASS", "", kExitPass);
  }
  if (std::strcmp(floorBasis, "thermal") == 0) {
    std::snprintf(reason, sizeof(reason),
                  "sustained SM clock %.1f%% of rated boost, below the %.1f%% thermal floor; "
                  "attributed to heat (mean %.1fC, thermal throttle latched)",
                  sustainedClockPct, floorApplied, meanTemp);
    return fail(reason);
  }
  if (pdWedgeSuspected) {
    std::snprintf(reason, sizeof(reason),
                  "sustained SM clock %.1f%% of rated boost at only %.1fC with no thermal "
                  "throttle — power-delivery wedge suspected (classification=%s)",
                  sustainedClockPct, meanTemp, classification);
    return fail(reason);
  }
  std::snprintf(reason, sizeof(reason),
                "sustained SM clock %.1f%% of rated boost, below the %.1f%% floor "
                "(classification=%s)",
                sustainedClockPct, floorApplied, classification);
  return fail(reason);
}

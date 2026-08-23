// nvml_dynamic.h — the NVML surface the soak runners need, resolved at RUNTIME.
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
// It is a superset of runners/clockprobe/nvml_dynamic.h: same design, same ABI
// constants, plus the two ECC entry points the soak runners need to tell "this
// part has no ECC" from "ECC is switched off" (see EccSupport below).
//
// Why dlopen instead of -lnvidia-ml
// ---------------------------------
// The runner-image rule in CLAUDE.md is that a published image ships NO
// NVIDIA-licensed redistributable library, and the Dockerfile enforces it by
// failing the build if `ldd` finds libcuda/libcudart/libnv in the final binary.
// NVML has no static form: link it and the binary carries a DT_NEEDED entry for
// libnvidia-ml.so.1, the assertion fires, and the image cannot be built. The
// driver-provided libnvidia-ml.so.1 IS present at runtime — the NVIDIA Container
// Toolkit injects it from the host when NVIDIA_DRIVER_CAPABILITIES includes
// "utility" — so the library is available exactly when we need it and never
// baked into a layer we redistribute. Resolving it with dlopen/dlsym is what
// reconciles those two facts.
//
// A second consequence: this header declares the handful of NVML entry points
// itself rather than including <nvml.h>. nvml.h ships under the NVIDIA CUDA
// toolkit's own licence, and while a build-time-only header would be no worse
// than nvcc itself, hand-declaring the ABI keeps every file we redistribute
// unambiguously Apache-2.0 and removes a build dependency on the header being
// present in whatever CUDA image is pinned. The constants below are NVML's
// published ABI values, not values this project is free to choose; changing one
// silently mis-reads the driver.
//
// Everything here is host code. It is included from a .cu only because the
// runner is a single translation unit compiled by nvcc.

#ifndef GLIMMER_BURNIN_NVML_DYNAMIC_H
#define GLIMMER_BURNIN_NVML_DYNAMIC_H

#include <dlfcn.h>

#include <string>

namespace nvmlrt {

// nvmlDevice_t is an opaque handle; nvmlReturn_t is an int-sized enum.
typedef void *Device;
typedef int Return;

// nvmlReturn_t values used here. Only kSuccess and kErrorNotSupported drive
// control flow; the rest are reported through errorString().
enum : Return {
  kSuccess = 0,
  kErrorNotSupported = 3,
  kErrorFunctionNotFound = 13,
};

// nvmlClockType_t. kClockSM is the SM (shader) domain — the one that collapses
// when a part is pinned in a low P-state. kClockGraphics is a different domain
// and is deliberately not substituted for it.
enum : int {
  kClockGraphics = 0,
  kClockSM = 1,
  kClockMem = 2,
  kClockVideo = 3,
};

// nvmlTemperatureSensors_t.
enum : int { kTemperatureGpu = 0 };

// nvmlDeviceGetMigMode() current-mode values.
enum : unsigned { kMigDisable = 0, kMigEnable = 1 };

// nvmlEnableState_t, as returned by nvmlDeviceGetEccMode.
enum : unsigned { kFeatureDisabled = 0, kFeatureEnabled = 1 };

// nvmlMemoryErrorType_t and nvmlEccCounterType_t.
//
// Volatile counters reset when the driver reloads; aggregate counters persist
// for the life of the part. The soak runners read the VOLATILE counters at the
// start and again at the end and report the difference, because the metric they
// owe is "ECC errors observed DURING the test" — an aggregate total would
// re-report a fault the fleet retired months ago on every single run.
enum : int { kMemoryErrorTypeCorrected = 0, kMemoryErrorTypeUncorrected = 1 };
enum : int { kVolatileEcc = 0, kAggregateEcc = 1 };

// nvmlClocksThrottleReason_* bits, as published by NVIDIA.
enum : unsigned long long {
  kReasonGpuIdle = 0x0000000000000001ULL,
  kReasonApplicationsClocksSetting = 0x0000000000000002ULL,
  kReasonSwPowerCap = 0x0000000000000004ULL,
  kReasonHwSlowdown = 0x0000000000000008ULL,
  kReasonSyncBoost = 0x0000000000000010ULL,
  kReasonSwThermalSlowdown = 0x0000000000000020ULL,
  kReasonHwThermalSlowdown = 0x0000000000000040ULL,
  kReasonHwPowerBrakeSlowdown = 0x0000000000000080ULL,
  kReasonDisplayClockSetting = 0x0000000000000100ULL,
};

// nvmlUtilization_t. Field order and types are ABI.
struct Utilization {
  unsigned int gpu;
  unsigned int memory;
};

// Library is the resolved NVML entry-point table.
//
// Members are split into REQUIRED and OPTIONAL. A missing required symbol means
// we cannot measure the thing the test exists to measure, and the runner must
// not pretend otherwise. A missing optional symbol costs us a piece of evidence
// (temperature, power, throttle reasons, ECC) and the runner reports which one
// is absent rather than substituting a sentinel value — a fabricated number
// would be indistinguishable from a measured one and could satisfy a threshold.
struct Library {
  void *handle = nullptr;

  // Required.
  Return (*init)() = nullptr;
  Return (*shutdown)() = nullptr;
  Return (*deviceGetHandleByIndex)(unsigned int, Device *) = nullptr;
  Return (*deviceGetClockInfo)(Device, int, unsigned int *) = nullptr;
  Return (*deviceGetMaxClockInfo)(Device, int, unsigned int *) = nullptr;

  // Optional.
  Return (*deviceGetHandleByPciBusId)(const char *, Device *) = nullptr;
  Return (*systemGetDriverVersion)(char *, unsigned int) = nullptr;
  Return (*deviceGetTemperature)(Device, int, unsigned int *) = nullptr;
  Return (*deviceGetPowerUsage)(Device, unsigned int *) = nullptr;
  Return (*deviceGetEnforcedPowerLimit)(Device, unsigned int *) = nullptr;
  Return (*deviceGetPowerManagementDefaultLimit)(Device, unsigned int *) = nullptr;
  Return (*deviceGetCurrentClocksThrottleReasons)(Device, unsigned long long *) = nullptr;
  Return (*deviceGetUtilizationRates)(Device, Utilization *) = nullptr;
  Return (*deviceGetMigMode)(Device, unsigned int *, unsigned int *) = nullptr;
  Return (*deviceGetEccMode)(Device, unsigned int *, unsigned int *) = nullptr;
  Return (*deviceGetTotalEccErrors)(Device, int, int, unsigned long long *) = nullptr;
  const char *(*errorStringFn)(Return) = nullptr;

  // errorString never returns null, so a caller may always print it.
  const char *errorString(Return r) const {
    if (errorStringFn != nullptr) {
      const char *s = errorStringFn(r);
      if (s != nullptr) return s;
    }
    return r == kSuccess ? "success" : "NVML error";
  }

  // open resolves libnvidia-ml and every symbol above. It returns false only
  // when the library or a REQUIRED symbol is missing, and writes a human
  // reason into err.
  bool open(std::string *err);

  void close() {
    if (handle != nullptr) {
      if (shutdown != nullptr) shutdown();
      dlclose(handle);
      handle = nullptr;
    }
  }
};

// EccSupport is whether the part has an ECC subsystem AT ALL, which is a
// different question from whether ECC is switched on, and a different question
// again from whether its counters could be read.
//
// This mirrors runners/host-health/nvidia.go, which reaches the same three
// states through nvidia-smi's ecc.mode.current column, and the reasoning is
// identical: only kEccAbsent may be reported to the operator as the "n/a"
// unmeasurable sentinel. On a part that HAS ECC, an unreadable counter means
// either that ECC was switched off or that our read failed — both leave the
// node unproven, so the counter is OMITTED and any threshold on it fails
// closed. Collapsing the two would hand every fleet a way to pass an ECC gate
// by turning ECC off.
enum EccSupport {
  // The mode itself could not be read: we could not look.
  kEccUnknown = 0,
  // The part reported a mode (enabled or disabled), so ECC hardware exists.
  kEccPresent = 1,
  // The driver answered NOT_SUPPORTED for the mode: there is no ECC subsystem
  // here to have counters. This is GB10 — its unified LPDDR5X has on-die ECC
  // that NVML cannot see — and it is the ONLY state in which a runner may
  // declare the counter unmeasurable.
  kEccAbsent = 2,
};

namespace detail {

// dlsym returns void*; converting that to a function pointer is not strictly
// conforming ISO C++ but is guaranteed by POSIX, which is the platform this
// image targets. The two-step cast keeps compilers from warning about it.
template <typename Fn>
inline Fn symbol(void *handle, const char *name, const char *fallback = nullptr) {
  void *p = dlsym(handle, name);
  if (p == nullptr && fallback != nullptr) p = dlsym(handle, fallback);
  Fn fn = nullptr;
  __builtin_memcpy(&fn, &p, sizeof(fn));
  return fn;
}

} // namespace detail

inline bool Library::open(std::string *err) {
  // The SONAME first: that is what the container toolkit injects. The bare
  // name is a fallback for a hand-built image with a dev symlink.
  static const char *const kCandidates[] = {"libnvidia-ml.so.1", "libnvidia-ml.so"};
  for (const char *name : kCandidates) {
    handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) break;
  }
  if (handle == nullptr) {
    if (err != nullptr) {
      const char *e = dlerror();
      *err = std::string("could not load libnvidia-ml.so.1: ") + (e != nullptr ? e : "unknown");
    }
    return false;
  }

  // Versioned symbol names, oldest-compatible fallback second. nvmlInit_v2 has
  // been the real entry point since NVML 5; nvmlInit remains as a compat stub.
  init = detail::symbol<Return (*)()>(handle, "nvmlInit_v2", "nvmlInit");
  shutdown = detail::symbol<Return (*)()>(handle, "nvmlShutdown");
  deviceGetHandleByIndex = detail::symbol<Return (*)(unsigned int, Device *)>(
      handle, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
  deviceGetClockInfo =
      detail::symbol<Return (*)(Device, int, unsigned int *)>(handle, "nvmlDeviceGetClockInfo");
  deviceGetMaxClockInfo =
      detail::symbol<Return (*)(Device, int, unsigned int *)>(handle, "nvmlDeviceGetMaxClockInfo");

  deviceGetHandleByPciBusId = detail::symbol<Return (*)(const char *, Device *)>(
      handle, "nvmlDeviceGetHandleByPciBusId_v2", "nvmlDeviceGetHandleByPciBusId");
  systemGetDriverVersion =
      detail::symbol<Return (*)(char *, unsigned int)>(handle, "nvmlSystemGetDriverVersion");
  deviceGetTemperature =
      detail::symbol<Return (*)(Device, int, unsigned int *)>(handle, "nvmlDeviceGetTemperature");
  deviceGetPowerUsage =
      detail::symbol<Return (*)(Device, unsigned int *)>(handle, "nvmlDeviceGetPowerUsage");
  deviceGetEnforcedPowerLimit =
      detail::symbol<Return (*)(Device, unsigned int *)>(handle, "nvmlDeviceGetEnforcedPowerLimit");
  deviceGetPowerManagementDefaultLimit = detail::symbol<Return (*)(Device, unsigned int *)>(
      handle, "nvmlDeviceGetPowerManagementDefaultLimit");
  // Renamed in NVML 12; the old name is retained as an alias, so try the new
  // one first and fall back rather than losing the mask on either vintage.
  deviceGetCurrentClocksThrottleReasons = detail::symbol<Return (*)(Device, unsigned long long *)>(
      handle, "nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons");
  deviceGetUtilizationRates =
      detail::symbol<Return (*)(Device, Utilization *)>(handle, "nvmlDeviceGetUtilizationRates");
  deviceGetMigMode = detail::symbol<Return (*)(Device, unsigned int *, unsigned int *)>(
      handle, "nvmlDeviceGetMigMode_v2", "nvmlDeviceGetMigMode");
  deviceGetEccMode = detail::symbol<Return (*)(Device, unsigned int *, unsigned int *)>(
      handle, "nvmlDeviceGetEccMode_v2", "nvmlDeviceGetEccMode");
  deviceGetTotalEccErrors = detail::symbol<Return (*)(Device, int, int, unsigned long long *)>(
      handle, "nvmlDeviceGetTotalEccErrors_v2", "nvmlDeviceGetTotalEccErrors");
  errorStringFn = detail::symbol<const char *(*)(Return)>(handle, "nvmlErrorString");

  const char *missing = nullptr;
  if (init == nullptr) missing = "nvmlInit_v2";
  else if (deviceGetHandleByIndex == nullptr) missing = "nvmlDeviceGetHandleByIndex_v2";
  else if (deviceGetClockInfo == nullptr) missing = "nvmlDeviceGetClockInfo";
  else if (deviceGetMaxClockInfo == nullptr) missing = "nvmlDeviceGetMaxClockInfo";
  if (missing != nullptr) {
    if (err != nullptr) {
      *err = std::string("libnvidia-ml is loaded but lacks required symbol ") + missing;
    }
    dlclose(handle);
    handle = nullptr;
    return false;
  }
  return true;
}

} // namespace nvmlrt

#endif // GLIMMER_BURNIN_NVML_DYNAMIC_H

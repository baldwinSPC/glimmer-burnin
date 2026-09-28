// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// load_envelope.h — the clock, temperature, power and throttle state a GEMM
// figure was measured under (#544). CUDA-free and NVML-free: the sampler in
// gemm_sweep.cu feeds readings in, and load_envelope_test.cc tests the
// arithmetic under `make test`.
//
// Sampled DURING the timed loop only, never across setup: a mean clock that
// included the host reference computation would drift toward idle and describe
// nothing. The names are the soak family's, with the same meaning — this is
// the part under this runner's load — and the fold across devices follows the
// registry (smClockMHz and enforcedPowerLimitW are floors, gpuTempC and
// powerDrawW ceilings), so a gate on any of them judges the WORST device.
#ifndef GLIMMER_BURNIN_LOAD_ENVELOPE_H
#define GLIMMER_BURNIN_LOAD_ENVELOPE_H

#include <algorithm>
#include <map>
#include <string>

namespace loadenv {

struct Envelope {
  long clockN = 0;
  double clockSum = 0;
  bool haveTemp = false;
  double tempMax = 0;
  bool havePower = false;
  double powerMax = 0;
  bool haveLimit = false;
  double limitMin = 0;
  bool haveMask = false;
  unsigned long long mask = 0;

  void addClock(unsigned int mhz) {
    ++clockN;
    clockSum += mhz;
  }
  void addTemp(double c) {
    tempMax = haveTemp ? std::max(tempMax, c) : c;
    haveTemp = true;
  }
  void addPower(double w) {
    powerMax = havePower ? std::max(powerMax, w) : w;
    havePower = true;
  }
  void addLimit(double w) {
    limitMin = haveLimit ? std::min(limitMin, w) : w;
    haveLimit = true;
  }
  void addMask(unsigned long long m) {
    mask |= m;
    haveMask = true;
  }

  // store writes only what was read. A field the driver never returned is
  // absent, never a zero nobody measured.
  void store(std::map<std::string, double> *values) const {
    if (clockN > 0) (*values)["sm_clock_mhz"] = clockSum / clockN;
    if (haveTemp) (*values)["gpu_temp_c"] = tempMax;
    if (havePower) (*values)["power_draw_w"] = powerMax;
    if (haveLimit) (*values)["enforced_power_limit_w"] = limitMin;
  }
};

// The NVML clocks-event reason bits and the soak family's labels for them, so
// throttleReasons reads the same whichever runner reported it.
struct Reason {
  unsigned long long bit;
  const char *label;
};
inline const Reason kReasons[] = {
    {0x1ULL, "gpuIdle"},           {0x2ULL, "applicationsClocksSetting"},
    {0x4ULL, "swPowerCap"},        {0x8ULL, "hwSlowdown"},
    {0x10ULL, "syncBoost"},        {0x20ULL, "swThermalSlowdown"},
    {0x40ULL, "hwThermalSlowdown"}, {0x80ULL, "hwPowerBrake"},
    {0x100ULL, "displayClockSetting"},
};

inline std::string reasonLabels(unsigned long long mask) {
  std::string s;
  for (const Reason &r : kReasons) {
    if ((mask & r.bit) == 0) continue;
    if (!s.empty()) s += ",";
    s += r.label;
  }
  return s.empty() ? "none" : s;
}

}  // namespace loadenv

#endif  // GLIMMER_BURNIN_LOAD_ENVELOPE_H

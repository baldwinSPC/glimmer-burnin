// thermal_soak.cu — the "thermal-soak" TestKind.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0 and uses no third-party
// source. The load generator, the NVML sampler and the reporting engine live in
// soak_core.cuh, shared byte-for-byte with runners/gpu-burn; this file is only
// the kind's stdout vocabulary and its assertions.
//
// WHAT THIS CATCHES
// -----------------
// A part that is fine for thirty seconds and not fine for thirty minutes. Heat
// takes time to move: the die reaches its steady state in seconds, the heatsink
// in minutes, and the chassis in tens of minutes, so a cooling fault — a dry
// thermal interface, a blocked intake, a fan that spins but does not move air,
// a chassis installed too close to its neighbour — is invisible to every short
// test in the suite and shows up only once the whole thermal mass has caught up.
// The node then throttles under real work, months later, and looks like a
// scheduling problem.
//
// WHY THE CORRECTNESS CHECK IS NOT OPTIONAL HERE
// ----------------------------------------------
// A soak that stops checking results is just a heater. The interesting failure
// is not "it got hot", it is "it got hot AND started returning different
// answers" — marginal silicon that computes correctly at 40C and not at 85C is
// exactly what a burn-in exists to find, and it is invisible to a test that
// measures only temperature. So every iteration is bitwise-compared against the
// reference for the whole soak, and a miscompare fails the node even if it never
// came close to a thermal limit. That check is the same code gpu-burn asserts
// on; the difference between the two kinds is which question decides the exit
// code, not what is measured.
//
// WHAT IT DOES NOT ASSERT BY DEFAULT
// ----------------------------------
// A temperature CEILING. There is no portable "too hot": the safe steady-state
// temperature of a part depends on its SKU, its cooling class, its firmware fan
// curve and the inlet temperature of the room it is in, and the driver already
// enforces the only limit that is a property of the silicon. A hard-coded
// ceiling would either fail healthy dense nodes or pass nothing at all in a warm
// aisle. What IS portable is "did the part have to protect itself" —
// throttleEvents — and that is what this runner decides on. Set
// THERMAL_SOAK_MAX_TEMP_C once you have a baseline for one hardware class in one
// room; see README.md.

#include "soak_core.cuh"

namespace {

// Device-fold direction, docs/dev/multi-device.md. The GATED metric
// (sustained_clock_pct — see main()'s clock-floor check below) is listed
// FIRST: soak::run() names the primaryKey passed to devices::fold() from this
// table's first entry, which is what worstDeviceIndex/worstDevicePciBusId end
// up naming. Aliased with the plain `devices` name devicefold_test.go's
// regex recognises (`(?:burnin::)?(?:devices::)?Fold::`) — `soak::devices`
// would not match it.
namespace devices = burnin::devices;
const std::vector<devices::FoldRule> kDeviceFold = {
    {"sustained_clock_pct", devices::Fold::Min},
    {"sustained_throughput_tflops", devices::Fold::Min},
    {"peak_temp_c", devices::Fold::Max},
    {"peak_power_w", devices::Fold::Max},
    {"throttle_count", devices::Fold::Sum},
    {"miscompares", devices::Fold::Sum},
    {"nonfinite_count", devices::Fold::Sum},
    // ecc_errors is Last across windows in the registry (an NVML lifetime
    // total, not a per-window measurement) but Sum across DEVICES — each
    // device's own counter since its own reset, per device_fold.h's Last
    // allowlist. devicefold_test.go skips checking a registry-Last key's
    // fold direction for exactly this reason.
    {"ecc_errors", devices::Fold::Sum},
};

// The stdout vocabulary for this kind. These spellings are not free: the
// operator's alias table (pkg/runner/parse.go, "thermal-soak") already maps
// them, and a key not in that table falls through to generic snake_case ->
// lowerCamelCase normalisation.
const soak::Keys kKeys = {
    /*markerPrefix=*/"THERMAL_SOAK",
    /*elapsed=*/"soak_seconds=",
    /*temp=*/"peak_temp_c=",
    /*power=*/"peak_power_w=",
    /*throttle=*/"throttle_count=",
    /*iterations=*/"iterations_completed=",
    /*miscompares=*/"miscompares=",
    /*tflops=*/"sustained_throughput_tflops=",
};

// The default sustained-clock floor, as a percentage of the part's own rated
// boost clock.
//
// 60 — and the road to that number is the point, because every intuitive answer
// is wrong. Measured on a healthy GB10 (DGX Spark, rated boost 3003 MHz read
// back from the driver) under this exact load, 2026-08-03:
//
//     25 s   82.2%     120 s  72.3%
//     45 s   80.5%     240 s  70.7%
//     60 s   78.1%     600 s  69.9%   (min sample 69.3%, peak 83C)
//
// Three things fall out of that table and each one kills a plausible threshold:
//
//   1. The widely cited 90% floor fails a healthy part instantly. Rated boost is
//      a short-burst number; nothing sustains it under a dense GEMM.
//   2. The sustained clock is a function of SOAK DURATION, not just of the part.
//      A floor calibrated on a 60-second run condemns every node in a 30-minute
//      run. This is not drift or noise — it is thermal mass reaching steady
//      state, and it is asymptotic, settling near 69.9%.
//   3. It happens with ZERO throttle events. Across the whole 600 s the reason
//      mask stayed 0: no thermal slowdown, no hardware slowdown, not even
//      swPowerCap. On this part the clock is managed down to hold ~83C and
//      ~86W without anything being latched for NVML to report.
//
// Point 3 is why this floor is a BACKSTOP and not the thermal verdict. The
// verdict is throttleEvents; the floor exists to catch a part wedged far below
// its class, not to grade performance.
//
// 60 is derived from the MEASURED side ALONE: ten points of headroom under the
// 69.9% asymptote in the table above, which is what this fleet actually
// produced. The wedge is NOT a calibration input for it, because no wedged part
// has ever been measured — see issue #61. This project ESTIMATES that a
// PD-wedged GB10 sits near 20% of rated boost, from a researched pin point of
// ~611 MHz against the 3003 MHz rated boost this fleet reports (611/3003 =
// 20.3%), which agrees in order of magnitude with GEP-0178's independent "~4x
// slow while reporting 96% utilization". An earlier revision of this comment
// said 30-50% and called 60 "ten points above the wedge ceiling"; that figure
// had no derivation behind it and the arithmetic it justified was never real.
// All the estimate supports is the weak claim that the floor clears a wedge
// with room to spare — at 20% a part misses this floor by forty points, and it
// would still miss it under the discarded 30-50% figure. Do not tighten this
// number towards a wedge estimate; measure one first.
//
// A gate that fires on everything is worse than no gate, because it gets
// disabled — and then it is not watching when something really does wedge.
// Recalibrate per hardware class AND per soak duration from your own fleet's
// distribution rather than inheriting this number for a part it was not
// measured on.
constexpr double kDefaultMinClockPct = 60.0;

} // namespace

int main() {
  std::string cfgErr;
  double minClockPct = 0.0, maxTempC = 0.0;
  if (!soak::envDouble("THERMAL_SOAK_MIN_CLOCK_PCT", kDefaultMinClockPct, &minClockPct, &cfgErr) ||
      !soak::envDouble("THERMAL_SOAK_MAX_TEMP_C", 0.0, &maxTempC, &cfgErr)) {
    return soak::errored(kKeys, cfgErr);
  }
  // Emitted before the soak so that even a run killed at its pod deadline
  // records what it was going to be judged against.
  std::printf("clock_floor_pct=%.2f\n", minClockPct);
  std::printf("temp_ceiling_c=%.2f\n", maxTempC);

  soak::Measurement m;
  const int rc = soak::run(kKeys, kDeviceFold, &m);
  if (rc != 0) return rc; // Skip or Error; the marker is already printed.

  // Everything above is already on stdout, so every branch from here leaves a
  // complete record behind it.
  char reason[512];

  // Correctness first. A part that got the wrong answer has failed regardless of
  // how cool it stayed, and reporting the thermal verdict first would bury it.
  if (m.miscompares > 0) {
    std::snprintf(reason, sizeof(reason),
                  "%llu element(s) of the result changed under load after %.0fs at up to %.0fC — "
                  "silent data corruption, not a thermal fault",
                  m.miscompares, m.elapsedS, m.tempKnown ? m.peakTempC : 0.0);
    return soak::fail(kKeys, reason);
  }
  if (m.nonfinite > 0) {
    std::snprintf(reason, sizeof(reason), "%llu non-finite value(s) in the result under load",
                  m.nonfinite);
    return soak::fail(kKeys, reason);
  }

  // The thermal verdict proper: did the part have to protect itself?
  if (m.throttleKnown && m.throttleEvents > 0) {
    std::snprintf(reason, sizeof(reason),
                  "the part throttled to protect itself %ld time(s) during the soak "
                  "(peak %.0fC, mask %llu)",
                  m.throttleEvents, m.tempKnown ? m.peakTempC : 0.0, m.reasonMask);
    return soak::fail(kKeys, reason);
  }

  if (m.clockKnown && m.sustainedClockPct < minClockPct) {
    std::snprintf(reason, sizeof(reason),
                  "sustained SM clock %.1f%% of rated boost over the soak, below the %.1f%% floor "
                  "(peak %.0fC)",
                  m.sustainedClockPct, minClockPct, m.tempKnown ? m.peakTempC : 0.0);
    return soak::fail(kKeys, reason);
  }

  // Off by default: a ceiling is only meaningful against a fleet baseline.
  if (maxTempC > 0.0 && m.tempKnown && m.peakTempC > maxTempC) {
    std::snprintf(reason, sizeof(reason), "peak temperature %.0fC exceeded the %.0fC ceiling",
                  m.peakTempC, maxTempC);
    return soak::fail(kKeys, reason);
  }

  return soak::pass(kKeys);
}

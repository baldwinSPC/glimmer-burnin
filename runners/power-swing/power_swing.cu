// power_swing.cu — the "power-swing" TestKind.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0 and uses no third-party
// source. The load generator, the NVML sampler and the reporting engine live in
// soak_core.cuh, shared byte-for-byte with runners/thermal-soak and
// runners/gpu-burn; this file is only the kind's stdout vocabulary, its duty
// cycle and its assertions.
//
// WHAT THIS CATCHES THAT thermal-soak DOES NOT
// ----------------------------------------------
// thermal-soak holds a SUSTAINED load and catches a cooling fault: the die, the
// heatsink and the chassis all reach their steady state and the part either
// holds its clock and stays out of protective throttling or it does not. What
// that tells you nothing about is a power-delivery TRANSIENT — a VRM that
// cannot slew fast enough to keep up with a sudden load step, or a PSU that
// sags for a few hundred milliseconds when several GPUs ramp together. A rail
// that droops for 200ms and recovers never shows up in a metric averaged, or
// even sampled at its worst, over a multi-minute sustained window; it shows up
// only in the instant right after the load changes.
//
// So this kind does the opposite of holding load: it ALTERNATES the load on a
// period (BURNIN_POWER_SWING_ON_SECONDS / _OFF_SECONDS) and watches what
// happens in the seconds right after each OFF -> ON transition
// (BURNIN_POWER_SWING_RAMP_WINDOW_S) — the worst instantaneous SM clock seen in
// that window, the peak instantaneous power, and any throttle-reason bit that
// appears during the ramp that was not already latched during the steady OFF
// phase before it. A VRM or PSU that cannot keep up shows up as a clock dip or
// a fresh throttle reason concentrated in that narrow post-ramp window, which a
// sustained soak's own sampling cadence is not aimed at catching.
//
// WHAT THIS DOES NOT ASSERT — DELIBERATELY THRESHOLDLESS ON THE SWING EVIDENCE
// ------------------------------------------------------------------------------
// This kind gates on CORRECTNESS ONLY: a miscompare or a non-finite value in
// the GEMM this engine still runs (identical to thermal-soak's and gpu-burn's)
// is a real hardware verdict regardless of how the load was scheduled, and
// fails the node exactly as those two kinds do. It does NOT fail on
// throttleEvents, and it does NOT fail on any swing_* field — those are
// evidence, always reported, never gating. The issue this kind was filed
// against says so explicitly: "Thresholdless until measured on the fleet."
// Nobody has yet watched a real VRM/PSU transient on this project's own
// hardware, so there is no calibrated floor for swingWorstPostRampClockPct or
// ceiling for swingPeakRampPowerW to gate on — publishing an uncalibrated one
// would fail healthy nodes on noise or pass a genuine transient on a floor set
// too low, which is exactly the trap CLAUDE.md's threshold-authoring rules
// exist to prevent. Once a fleet has run this and a real transient has been
// observed and characterised, a future revision can propose a threshold from
// measured data — see thermal_soak.cu's own kDefaultMinClockPct comment for
// what that derivation should look like.

#include "soak_core.cuh"

namespace {

// Device-fold direction, docs/dev/multi-device.md. This kind still runs the
// same GEMM+verify load thermal-soak and gpu-burn do, so the correctness and
// thermal/clock fold entries are carried over unchanged; the four swing_*
// entries are new. sustained_clock_pct is listed FIRST for the same reason
// thermal-soak lists it first: soak::run() names the primaryKey passed to
// devices::fold() from this table's first entry, which is what
// worstDeviceIndex/worstDevicePciBusId end up naming.
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
    // total) but Sum across DEVICES — see thermal_soak.cu's identical comment.
    {"ecc_errors", devices::Fold::Sum},
    // The duty-cycle evidence. swing_transitions is identical across devices
    // by construction (every device follows the same wall-clock schedule), so
    // it folds Once; the other three follow pkg/contract's Aggregation for
    // their registered names — Min for a floor, Max for a ceiling, Sum for a
    // per-window count.
    {"swing_transitions", devices::Fold::Once},
    {"swing_worst_post_ramp_clock_pct", devices::Fold::Min},
    {"swing_peak_ramp_power_w", devices::Fold::Max},
    {"swing_new_throttle_events", devices::Fold::Sum},
};

// The stdout vocabulary for this kind. Deliberately the same spellings
// thermal-soak uses for the fields the two kinds share (same engine, same
// measurands) — pkg/runner/parse.go's "power-swing" alias table maps them the
// same way "thermal-soak"'s does, kept as a SEPARATE table because the two are
// separate TestKinds and parsing is scoped per kind.
const soak::Keys kKeys = {
    /*markerPrefix=*/"POWER_SWING",
    /*elapsed=*/"soak_seconds=",
    /*temp=*/"peak_temp_c=",
    /*power=*/"peak_power_w=",
    /*throttle=*/"throttle_count=",
    /*iterations=*/"iterations_completed=",
    /*miscompares=*/"miscompares=",
    /*tflops=*/"sustained_throughput_tflops=",
};

} // namespace

int main() {
  std::string cfgErr;

  // ── duty-cycle configuration ────────────────────────────────────────────
  long onSeconds = 0, offSeconds = 0, rampWindowSeconds = 0;
  if (!soak::envLong("BURNIN_POWER_SWING_ON_SECONDS", 10, &onSeconds, &cfgErr) ||
      !soak::envLong("BURNIN_POWER_SWING_OFF_SECONDS", 10, &offSeconds, &cfgErr) ||
      !soak::envLong("BURNIN_POWER_SWING_RAMP_WINDOW_S", 3, &rampWindowSeconds, &cfgErr)) {
    return soak::errored(kKeys, cfgErr);
  }

  // Sane-bounds clamps, in the raise-and-warn style soak_core.cuh's own
  // BURNIN_SOAK_MATRIX_N clamp uses: a value outside these bounds is a
  // configuration mistake, not a reason to refuse a run outright.
  if (onSeconds < 1) {
    soak::warn("BURNIN_POWER_SWING_ON_SECONDS raised to a 1s floor");
    onSeconds = 1;
  }
  if (offSeconds < 1) {
    soak::warn("BURNIN_POWER_SWING_OFF_SECONDS raised to a 1s floor");
    offSeconds = 1;
  }
  if (rampWindowSeconds < 1) {
    soak::warn("BURNIN_POWER_SWING_RAMP_WINDOW_S raised to a 1s floor");
    rampWindowSeconds = 1;
  }
  if (rampWindowSeconds > onSeconds) {
    soak::warn("BURNIN_POWER_SWING_RAMP_WINDOW_S clamped to the on-phase length (" +
               std::to_string(onSeconds) +
               "s); a ramp window longer than the on-phase itself is nonsensical");
    rampWindowSeconds = onSeconds;
  }

  // ── the duration-vs-cycle refusal ───────────────────────────────────────
  //
  // A run shorter than two full on+off cycles never completes even one clean
  // ramp-to-ramp comparison, so it cannot answer the question this kind
  // exists to ask. thermal-soak's own duration floor (soak::kMinDurationSeconds)
  // instead RAISES a too-short request and warns, because that floor is an
  // intrinsic property of the single duration value against itself — it
  // cannot be pushed far from what was asked.
  //
  // This is a different shape of problem and it gets a different answer:
  // REFUSE rather than raise. Two independently configured knobs are involved
  // here (BURNIN_DURATION_SECONDS and the on/off cycle length), and unlike
  // thermal-soak's small, bounded floor (15s), the gap between what was asked
  // and what the configured cycle needs can be large — a fleet's default
  // BURNIN_DURATION_SECONDS left unchanged against a deliberately long cycle
  // could need several times the requested duration. Silently running that
  // much longer than requested risks running past the pod deadline the
  // operator computed from the ORIGINAL, unraised duration — turning a
  // configuration mismatch into a pod eviction partway through, which reports
  // as an Error with a confusing kubelet-provided reason rather than as the
  // clear, immediate configuration refusal this is. Read BURNIN_DURATION_SECONDS
  // ourselves, before soak::run() gets to it, precisely so this refusal can
  // happen first.
  long durationSeconds = 0;
  std::string durErr;
  if (!soak::envLong("BURNIN_DURATION_SECONDS", soak::kDefaultDurationSeconds, &durationSeconds, &durErr)) {
    return soak::errored(kKeys, durErr);
  }
  const long cycleSeconds = onSeconds + offSeconds;
  const long minDurationForTwoCycles = 2 * cycleSeconds;
  if (durationSeconds < minDurationForTwoCycles) {
    char reason[320];
    std::snprintf(reason, sizeof(reason),
                  "BURNIN_DURATION_SECONDS=%ld cannot complete two full on+off cycles at "
                  "%lds on + %lds off (%lds per cycle); request at least %lds, or shorten "
                  "BURNIN_POWER_SWING_ON_SECONDS/_OFF_SECONDS",
                  durationSeconds, onSeconds, offSeconds, cycleSeconds, minDurationForTwoCycles);
    return soak::errored(kKeys, reason);
  }

  // Echoed before the soak so that even a run killed at its pod deadline
  // records what it was going to be judged against — the same reason
  // thermal_soak.cu echoes clock_floor_pct/temp_ceiling_c before its own run.
  std::printf("swing_on_seconds=%ld\n", onSeconds);
  std::printf("swing_off_seconds=%ld\n", offSeconds);
  std::printf("swing_ramp_window_s=%ld\n", rampWindowSeconds);

  soak::DutyCycle duty;
  duty.onSeconds = onSeconds;
  duty.offSeconds = offSeconds;
  duty.rampWindowSeconds = rampWindowSeconds;

  soak::Measurement m;
  const int rc = soak::run(kKeys, kDeviceFold, &m, &duty);
  if (rc != 0) return rc; // Skip or Error; the marker is already printed.

  // Correctness only — see the header comment above. No throttleEvents check,
  // no swing_* gate: those are evidence, reported unconditionally by the
  // engine above, and this kind is explicitly thresholdless on them until a
  // real transient has been observed on this project's own fleet.
  char reason[512];
  if (m.miscompares > 0) {
    std::snprintf(reason, sizeof(reason),
                  "%llu element(s) of the result changed under load across %ld duty-cycle "
                  "transition(s) — silent data corruption, not a power-delivery finding",
                  m.miscompares, m.swingTransitions);
    return soak::fail(kKeys, reason);
  }
  if (m.nonfinite > 0) {
    std::snprintf(reason, sizeof(reason), "%llu non-finite value(s) in the result under load",
                  m.nonfinite);
    return soak::fail(kKeys, reason);
  }

  return soak::pass(kKeys);
}

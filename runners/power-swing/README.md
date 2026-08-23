# `power-swing` runner — the transient a sustained soak cannot see

Runner image for the `power-swing` [`TestKind`](../../pkg/contract/vocabulary.go):
alternate a heavy, correctness-checked compute load between "on" and "off" on a
period, and report what happens in the seconds right after each ramp.

Implements TestKind `power-swing`. Shares its entire engine with
[`../thermal-soak`](../thermal-soak) and [`../gpu-burn`](../gpu-burn) — same
GEMM, same NVML sampler, same reporting engine (`soak_core.cuh`) — and differs
only in its duty cycle, its stdout vocabulary, and in gating on correctness
alone. See [Shares its engine with thermal-soak](#shares-its-engine-with-thermal-soak-and-gpu-burn).

> **Hardware verification status: NONE.** This kind has not been compiled with
> a real `nvcc` toolchain and has not run on any GPU. See
> [What is not verified](#what-is-not-verified) before relying on anything
> below.

---

## Why this test exists

[`thermal-soak`](../thermal-soak) holds a **sustained** load and catches a
cooling fault: the die reaches its steady state in seconds, the heatsink in
minutes, the chassis in tens of minutes, and a part that cannot hold its clock
or stay out of protective throttling once all three have caught up is unwell.

That tells you nothing about a **power-delivery transient** — a VRM that
cannot slew fast enough to keep up with a sudden load step, or a PSU that sags
for a few hundred milliseconds when several GPUs ramp together. Neither shows
up in a figure averaged, or even sampled at its worst, over a multi-minute
sustained hold: both show up only in the instant right after the load
changes, which a sustained soak's own sampling cadence is not aimed at
catching.

So this kind does the opposite of holding load. It alternates the load on a
period (`BURNIN_POWER_SWING_ON_SECONDS` / `_OFF_SECONDS`) and watches the
seconds right after each OFF → ON transition
(`BURNIN_POWER_SWING_RAMP_WINDOW_S`) for:

- the worst instantaneous SM clock seen in that window, as a percentage of
  rated boost (`swingWorstPostRampClockPct`) — a VRM that cannot keep up shows
  up as a clock dip concentrated in the ramp;
- the peak instantaneous board power seen in that window
  (`swingPeakRampPowerW`);
- any throttle-reason bit that appears during the ramp that was **not**
  already latched during the steady OFF phase immediately before it
  (`swingNewThrottleEvents`, `swingNewThrottleReasons`) — a reason that shows
  up specifically at the moment of the ramp, rather than one already present
  beforehand.

## Shares its engine with thermal-soak and gpu-burn

`soak_core.cuh`, `nvml_dynamic.h` and `kmsg/kmsg_watch.h` are **byte-identical
copies** across all three directories (plus their `-rocm` counterparts for the
two that have one), because the publish workflow builds each runner with its
own directory as the Docker build context and `COPY` cannot reach outside a
build context.
[`../thermal-soak/soak_contract_test.go`](../thermal-soak/soak_contract_test.go)
fails if any copy drifts. Edit the canonical copy in `runners/thermal-soak/`
and copy it out to the others.

The duty cycle itself lives in the shared engine, threaded through as an
**additive, defaulted** `DutyCycle*` parameter: `thermal_soak.cu` and
`gpu_burn.cu` call `soak::run(kKeys, kDeviceFold, &m)` exactly as before —
unchanged, unaffected, and covered by the shared contract test — while
`power_swing.cu` is the one caller that passes `&duty`.

---

## Output contract

| Result | stdout | Exit |
|---|---|---|
| Pass | `POWER_SWING_PASS` | 0 |
| Fail | `POWER_SWING_FAIL: <reason>` | 1 |
| Not applicable | `POWER_SWING_SKIP: <reason>` | 2 |
| Unjudged | `POWER_SWING_ERROR: <reason>` | 3 |

Metrics are `key=value` lines on stdout, printed **before** the decision and
**re-printed periodically** during the run (every
`BURNIN_SOAK_PROGRESS_INTERVAL_S`, default 15 s) — the identical discipline
`thermal-soak` documents, for the identical reason: a log truncated at any
point still parses into a complete, coherent snapshot.

### Skip (exit 2) / Error (exit 3)

Identical causes to [`thermal-soak`'s own list](../thermal-soak/README.md#skip-exit-2--the-test-does-not-apply):
no accelerator visible or MIG enabled (Skip); the CUDA driver unusable from
the container, `libnvidia-ml.so.1` absent while an accelerator is visible,
allocation or launch failures, no iterations completed, mean utilization
under load below 50%, or a malformed environment variable (Error).

**One error is specific to this kind**: `BURNIN_DURATION_SECONDS` too short to
complete two full on+off cycles. See
[The duration-vs-cycle refusal](#the-duration-vs-cycle-refusal) below.

### Fail (exit 1) — correctness only

Checked in this order:

1. `miscompares > 0` — silent data corruption.
2. `nonfiniteCount > 0` — a NaN or Inf in a result whose operands are all
   finite.

**That is the whole list.** Unlike `thermal-soak`, there is no throttle check
and no clock-floor check here — see the next section.

---

## What is deliberately not asserted — thresholdless by design

**This kind gates on correctness only.** It does not fail on
`throttleEvents`, and it does not fail on any `swing_*` field — those are
evidence, always reported, never gating. The issue this kind was filed
against says so explicitly: *"Thresholdless until measured on the fleet."*

Nobody has yet watched a real VRM/PSU transient on this project's own
hardware, so there is no calibrated floor for `swingWorstPostRampClockPct` or
ceiling for `swingPeakRampPowerW` to gate on. Publishing an uncalibrated one
would fail healthy nodes on noise, or pass a genuine transient on a floor set
too low — the exact trap `thermal_soak.cu`'s own `kDefaultMinClockPct` comment
documents deriving 60% (not the intuitive 90%) from **measured** fleet data
before it was safe to ship. Once a fleet has run this kind and a real
transient has been observed and characterised, a future revision can propose
a threshold from that data.

### The duration-vs-cycle refusal

A run shorter than two full on+off cycles never completes even one clean
ramp-to-ramp comparison, so `power_swing.cu` refuses (`POWER_SWING_ERROR`)
rather than running a truncated, meaningless duty cycle.

This is a **refusal**, not a raise-and-warn — a deliberate difference from
`thermal-soak`'s own duration floor, which silently raises a too-short
request to 15 s. That floor compares ONE value (`BURNIN_DURATION_SECONDS`)
against its own small, bounded minimum. Here TWO independently configured
knobs are involved (the requested duration, and the on/off cycle length a
site chooses separately), and the gap between what was asked for and what
the configured cycle needs can be large — silently running several times
longer than requested risks running past the pod deadline the operator
computed from the original, unraised duration, turning a configuration
mismatch into a pod eviction partway through rather than an immediate,
legible refusal.

---

## Configuration

| env | default | meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 900 | total wall time; must be at least `2 * (BURNIN_POWER_SWING_ON_SECONDS + BURNIN_POWER_SWING_OFF_SECONDS)` or the run refuses (see above) |
| `BURNIN_POWER_SWING_ON_SECONDS` | 10 | how long the load is held on, per cycle. Raised to a 1 s floor if configured lower |
| `BURNIN_POWER_SWING_OFF_SECONDS` | 10 | how long the device idles (no kernel in flight), per cycle. Raised to a 1 s floor if configured lower |
| `BURNIN_POWER_SWING_RAMP_WINDOW_S` | 3 | how long after a ramp starts counts as "post-ramp". Raised to a 1 s floor, and clamped down to the on-phase length — a ramp window longer than the on-phase itself is nonsensical |

Every other engine knob (`BURNIN_SOAK_MATRIX_N`, `BURNIN_SOAK_SAMPLE_INTERVAL_MS`,
`BURNIN_SOAK_PROGRESS_INTERVAL_S`, `BURNIN_SOAK_INJECT_MISCOMPARES`) is the
same shared engine and behaves identically to
[`thermal-soak`'s own table](../thermal-soak/README.md#configuration).

A value that cannot be parsed is a configuration Error, not a default:
guessing would run a different test than the one that was asked for while
reporting success.

**`BURNIN_DEVICE_CONCURRENCY=sequential` defeats this kind's whole point.**
This engine's shared `BURNIN_DEVICE_CONCURRENCY` knob (default `all`) decides
whether a multi-device node's devices ramp together or one at a time. Under
`sequential`, each device duty-cycles on its own schedule while every other
device sits untouched — so no two devices are ever ramping at the same
wall-clock moment, and a board-wide PSU sag when *several* GPUs ramp together
(the failure this kind exists to catch) can never be observed. It still
measures something real per device, just not that; the engine warns at
runtime if you select it anyway. Leave this at the default on a multi-device
node.

---

## Metrics

### Correctness (the acceptance gate)

| raw key | canonical | |
|---|---|---|
| `miscompares` | `miscompares` | count of computed values that differed from the reference |
| `nonfinite_count` | `nonfiniteCount` | count of NaN/Inf values |

### Duty-cycle evidence — never gated on

| raw key | canonical | |
|---|---|---|
| `swing_transitions` | `swingTransitions` | count of OFF→ON transitions (ramps) completed |
| `swing_worst_post_ramp_clock_pct` | `swingWorstPostRampClockPct` | lowest instantaneous SM clock, % of rated boost, seen inside any post-ramp window. Omitted if no ramp sample was taken or the rated clock could not be read |
| `swing_peak_ramp_power_w` | `swingPeakRampPowerW` | highest instantaneous board power seen inside any post-ramp window. Omitted if no ramp sample with a working power read was taken |
| `swing_new_throttle_events` | `swingNewThrottleEvents` | count of post-ramp windows during which a throttle-reason bit appeared that was not already latched during the steady OFF phase before it. Omitted on a device whose throttle reasons could never be read at all |
| `swing_new_throttle_reasons` | `swingNewThrottleReasons` | human-readable, comma-separated names of those bits ("none" when none appeared) |
| `swing_on_seconds`, `swing_off_seconds`, `swing_ramp_window_s` | (unregistered; generic normalisation) | the resolved (post-clamp) duty-cycle configuration, echoed before the run so a pod killed at its deadline still records what it was going to be judged against |

### Load, thermal, power and identity evidence

Everything [`thermal-soak` reports](../thermal-soak/README.md#load-thermal-and-power-evidence)
from the shared engine — `sustainedClockPct`, `smClockMHz`, `gpuTempC`,
`powerDrawW`, `throttleEvents`, `throttleReasons`, `eccErrors` (or `n/a` on a
part with no ECC subsystem, e.g. GB10), `xidEvents`, `sustainedThroughputTflops`
— is reported here too, from the identical GEMM+verify load. None of it gates
this kind; correctness is the only acceptance decision it makes.

### Multi-device

This runner measures **every device the pod was allocated**, not device 0
(`docs/dev/multi-device.md`), through the same shared `device_fold.h` every
converted runner in this repository carries. The correctness and thermal/power
figures fold exactly as `thermal-soak`'s do (worst device for a floor, sum for
a windowed count). The four `swing_*` figures fold the same way:
`swingTransitions` is identical across devices by construction — every device
follows the same wall-clock schedule — so it folds `Once`;
`swingWorstPostRampClockPct` is `Min` (a floor), `swingPeakRampPowerW` is `Max`
(a ceiling), `swingNewThrottleEvents` is `Sum` (a windowed count).

---

## Build

```sh
docker build -t power-swing runners/power-swing
docker run --rm --gpus all \
  -e BURNIN_DURATION_SECONDS=60 \
  -e BURNIN_POWER_SWING_ON_SECONDS=10 \
  -e BURNIN_POWER_SWING_OFF_SECONDS=10 \
  power-swing
```

Multi-stage: the CUDA toolchain is used at build time only, `nvcc` links no
NVIDIA-owned redistributable library (`-lnvidia-ml` never appears; NVML is
`dlopen`ed at runtime), and the build **fails** if `ldd` finds
`libcuda`/`libcudart`/`libcublas`/`libnv` in the shipped binary — the same
assertion `thermal-soak`'s Dockerfile makes, over the same reasoning. See its
[Licensing section](../thermal-soak/README.md#licensing) for the full
argument; nothing about it differs here.

Arch targets are the same fleet-wide list `thermal-soak` ships (`sm_80`,
`sm_90`, `sm_100`, `sm_120`, `sm_121`, plus PTX), for the same reason: this
kind's claim — the part held its clock and kept agreeing with itself, this
time through a duty cycle rather than a sustained hold — does not depend on
which real instruction path ran, so a PTX-JIT fallback cannot fake it.

---

## What is not verified

**Nothing about this runner has been verified on real hardware, and its CUDA
source has not even been compiled.** This session had no CUDA toolchain (no
`nvcc`) and no GPU available. The precedent for how to say this plainly is
[`../clockprobe`](../clockprobe)'s own #301 change: `go build`/`go vet`/`gofmt`
and the Go-side unit and contract tests are clean, but `power_swing.cu` and
its additions to the shared `soak_core.cuh` engine were checked by careful
manual review against the existing, working `thermal-soak`/`gpu-burn` code
they were modelled on — never compiled, never run.

Specifically unverified:

- that `power_swing.cu` and the modified `soak_core.cuh` compile at all under
  `nvcc`;
- that the duty cycle actually produces the alternating on/off load pattern
  it is designed to (the launch-gating and idle-relaunch logic in
  `runActiveDevices`);
- every `swing_*` metric's value, on any hardware;
- that this kind's whole premise — a VRM or PSU transient is detectable this
  way at all — holds on any real board. No power-delivery transient has ever
  been observed by this project; this kind exists to let someone start
  looking.

Before publishing a tag: build the image, run it against real hardware with
a short duty cycle, and confirm the alternating load is visible in the
reported `swing_*` evidence and (ideally) in an external power meter or the
node's own telemetry. File the hardware-verification issue this playbook
calls for (the shape of #237, #242, #265: what to run, what to capture, what
would falsify the design) before publishing.

---

## Licensing

Identical to [`thermal-soak`'s own table](../thermal-soak/README.md#licensing):
`power_swing.cu`, `soak_core.cuh` and `nvml_dynamic.h` are Apache-2.0
(this project), no third-party source is shipped, the CUDA toolchain is a
build-time-only dependency, and `libcuda.so.1`/`libnvidia-ml.so.1` are
injected at runtime by the NVIDIA Container Toolkit rather than baked in.

---

## Requirements on the node

Identical to [`thermal-soak`'s](../thermal-soak/README.md#requirements-on-the-node):
an NVIDIA accelerator, the NVIDIA Container Toolkit with driver injection
declared (this Dockerfile sets `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`), and — for the `xidEvents`
window-scoped watch — `/dev/kmsg` reachable via `spec.runner.hostPaths`, which
degrades honestly (`xid_source=none`, `xidEvents` omitted) rather than
fabricating a zero when it is not mounted.

# `clockprobe` runner — sustained clock under load

Runner image for the `clockprobe` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
hold a known, steady, clock-bound load and assert the accelerator actually
**sustains** its rated clocks.

## Why this test exists

The fault it catches is a GPU pinned in a low P-state that reports perfectly
healthy utilization. The node looks busy, `nvidia-smi` shows 100% utilization,
every liveness and readiness check passes, and the part simply runs at a
fraction of its rated clock. On GB10 (DGX Spark) this is the **USB-C
Power-Delivery wedge**: an under-spec supply, a degraded cable, or a PD contract
that negotiated a lower wattage silently caps the board power budget, and the
part clocks down and stays down.

A fleet with this fault delivers correct results, slowly, forever. The loss
shows up only in training-job wall-clock, usually months later and attributed to
the model rather than the hardware.

Most of the suite really is blind to it — every one of these asserts correctness
or health, and none of them asserts speed:

| test | why it passes anyway |
|------|----------------------|
| `compute-smoke` | the arithmetic is still exactly right, just slower |
| `dcgm-diag` | nothing is faulty; no counter moves |
| `memory-bw` | the memory path is not what was capped |
| `host-health` | it counts faults, and a wedge latches none; it reports `powerDrawW` as evidence but reads no clock |

**`thermal-soak` is the exception, and this runner does not claim otherwise.**
It applies its own sustained-clock floor — `THERMAL_SOAK_MIN_CLOCK_PCT`, default
**60 %** of rated boost — and returns `THERMAL_SOAK_FAIL` below it
([`thermal_soak.cu`](../thermal-soak/thermal_soak.cu)). That floor is set ten
points under the **measured** 69.9 % sustained-clock asymptote of a healthy
Spark, not against a wedge — but this project's **estimate** of where a wedge
lands, near **20 %** of rated boost, is far enough under 60 that a wedged part
put through a soak should fail the soak. Estimate, not measurement; see
[what is not verified](#what-is-not-verified) below.

So `clockprobe` earns its place on **cost and attribution**, not on exclusive
coverage:

- **Cost.** This probe defaults to **60 s** (10 s floor) against `thermal-soak`'s
  **900 s**, and the sample acceptance profile soaks for 1800 s. That is the
  difference between a gate you can afford to run on every node at enrollment
  and on every reboot, and one you run overnight. A wedge that only a 15-minute
  soak would have caught is a wedge that ships.
- **Attribution.** `thermal-soak`'s clock floor is explicitly a **backstop**, not
  its verdict — a soak failure says "this part was slow *or* hot *or* throttling"
  and hands a human a soak log to disentangle. This runner isolates the clock
  signal: it holds a clock-bound load with no memory traffic and no thermal
  agenda, then reports `pdWedgeSuspected`, `throttleClassification`,
  `powerLimitRatioPct` and `tempAtMinClockC`. A wedge comes back named as a
  wedge, pointing at a cable or a supply, rather than as a soak that did not
  pass.

### What is not verified

That `thermal-soak` actually fires on a genuinely wedged part is **inferred from
reading its source**, not observed. Its floor logic and its default were read out
of `thermal_soak.cu`; no wedged Spark was available to run either test against.
Three things follow that a reader should not have to guess at:

- The soak's floor is only applied when the clock could be read at all
  (`m.clockKnown`). A part whose clock NVML will not report is not judged against
  the floor by that runner.
- **Where a wedge actually lands has never been measured by this project.** The
  20 % figure is an ESTIMATE: a researched pin point of ~611 MHz over the 3003 MHz
  rated boost this fleet reports (611/3003 = 20.3 %), corroborated only in order
  of magnitude by GEP-0178's independent "~4× slow while reporting 96 %
  utilization". Neither this runner nor `thermal-soak` has produced a number from
  wedged silicon. A 30–50 % figure appeared in earlier revisions of this file and
  of `thermal_soak.cu`, each citing the other and neither citing a measurement;
  it has been dropped rather than reconciled, because it had no derivation to
  reconcile.
- The 60 % floor is calibrated per hardware class **and per soak duration**, and
  from healthy parts only. On other parts, or at a duration the floor was not
  calibrated for, the margin between a wedge and a healthy soak is not
  established.

Confirming both runners against a deliberately under-powered Spark is the
outstanding piece of work here, and is tracked as
[#61](https://github.com/baldwinSPC/glimmer-burnin/issues/61). Until that lands,
no threshold in this repository should be tightened on the strength of the 20 %
estimate.

## Workload

A register-resident FP32 FMA chain: eight independent `fmaf` chains per thread,
looped, with no loads, no stores and no shared memory. `blocks = SMs × 8`,
256 threads per block.

That shape is chosen for two properties:

- **Clock-bound.** Nothing else can make it slower. Achieved throughput moves in
  lockstep with achieved clock, which gives an independent cross-check on the
  clock the driver reports — if `sustainedClockPct` and the throughput ratio
  disagree, one of them is lying.
- **Exactly countable.** FLOPs are `threads × iterations × 8 × 2`, so
  `sustainedThroughputTflops` is a measurement, not a figure derived from a
  nameplate constant.

The iteration count is **calibrated on the live part** so one launch takes about
50 ms, after a throwaway launch that pays for CUDA context creation. Work is
issued in ~1-second windows; NVML is sampled (default every 100 ms) only while
the stream is demonstrably busy, so no idle sample can dilute the mean.

`BURNIN_DURATION_SECONDS` is honoured: the first `warmup_s` seconds run the same
load with sampling **disabled**, and the remainder is the measurement window. The
warm-up is not optional — an unwarmed part sits at idle clocks legitimately, and
sampling through the ramp would fail healthy nodes.

## Telling a thermal throttle from a PD wedge

Both present as a low clock. Treating them alike would make the test useless:
thermal throttling under sustained heat is expected behaviour and is
`thermal-soak`'s business, while a PD wedge is a broken node.

| | sustained clock | temperature | throttle reason mask | power limits |
|---|---|---|---|---|
| **thermal** | low | **high** | `swThermalSlowdown` / `hwThermalSlowdown` / `hwSlowdown` | enforced ≈ default |
| **PD wedge** | low | **low** | `swPowerCap` / `hwPowerBrake`, or nothing at all | enforced **well below** default |

The runner emits all four columns, plus `temp_at_min_clock_c` — the temperature
at the *slowest observed sample*, which is the single most legible number for a
human triaging the difference.

**If temperature cannot be read, the shortfall is not attributed to heat.** An
unknown measurement must never buy the more lenient floor; that is the same
fail-closed rule `pkg/verdict` follows.

## Output contract

| Result | stdout | Exit |
|--------|--------|------|
| Pass | `CLOCKPROBE_PASS` | 0 |
| Fail | `CLOCKPROBE_FAIL: <reason>` | 1 |
| Not applicable | `CLOCKPROBE_SKIP: <reason>` | 2 |
| Unjudged | `CLOCKPROBE_ERROR: <reason>` | 3 |

Metrics are `key=value` lines on stdout and are **always printed before the
decision**, so a failing, skipping or erroring run still leaves its full
evidence behind.

### Skip (exit 2) — the test does not apply

- **No accelerator visible to the container.** Nothing whose clock could be
  probed.
- **MIG is enabled.** SM clock is a device-wide property; a clock sampled from
  inside one MIG instance cannot be attributed to that instance's load.
- **No rated SM boost clock.** Without a nameplate denominator there is no
  portable percentage. A missing nameplate is an *unjudged* part, not a slow
  one — this is stated explicitly in the `KindClockProbe` documentation and is
  the rule this runner implements.

### Error (exit 3) — we do not know, and will not guess

`Error` is not `Fail`. These paths mean the runner could not produce a verdict;
none of them condemns the hardware.

- An accelerator is visible but **`libnvidia-ml.so.1` cannot be loaded**. That
  is a container misconfiguration (`NVIDIA_DRIVER_CAPABILITIES` must include
  `utility`), not a property of the node. Skipping here would quietly report
  "not applicable" for an entire fleet whose toolkit was set up wrong.
- `nvmlInit` fails, the device handle cannot be resolved, or the load kernel
  fails to launch or run.
- **No clock samples were taken.** The probe measured nothing.
- **Mean GPU utilization under load is below 50%.** The load did not land, so
  the sampled clock is not attributable to this test. Reporting a low clock here
  would condemn a node for the runner's own problem.
- A `CLOCKPROBE_*` environment variable or `BURNIN_DURATION_SECONDS` is not
  parseable as a number. Guessing a default would silently run a different test
  than the one that was asked for.

An `Error` run still prints everything it established — identity, configuration,
power limits, `elapsedS`, `samplesTaken`, `loadLaunches` — but deliberately does
**not** publish a derived `sustainedClockPct`. A percentage computed over a
window that was cut short is not the measurement that metric name promises, and
charting it beside real ones would make a broken run look like a slow node.

### Fail (exit 1)

`sustainedClockPct` below the applied floor. Two floors exist, and which one was
applied is reported as `clockFloorBasis`:

- `general` — `CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT`, default **70**.
- `thermal` — `CLOCKPROBE_MIN_THERMAL_CLOCK_PCT`, default **50**, applied only
  when the shortfall is attributable to heat: mean temperature at or above
  `CLOCKPROBE_THERMAL_TEMP_C` (default 80) **and** a thermal reason latched in
  the mask. It is clamped so it can never be stricter than the general floor —
  a thermal allowance exists to excuse an expected shortfall, not to invent a
  second way to fail.

The runner's own floors are a backstop for a probe run with no thresholds at
all. A fleet gate belongs in the `BurnInTest`, on `sustainedClockPct`, where the
number is visible and reviewable.

## Metrics

Keys below are what the runner prints; the second column is the canonical name
[`pkg/runner`](../../pkg/runner/parse.go) normalises it to and a threshold is
written against.

### Identity and configuration

| key | canonical |
|-----|-----------|
| `gpu_name`, `compute_cap`, `pci_bus_id`, `driver_version`, `mig_mode` | `gpuName`, `computeCap`, `pciBusId`, `driverVersion`, `migMode` |
| `duration_requested_s`, `warmup_s`, `sample_window_s`, `elapsed_s` | `durationRequestedS`, `warmupS`, `sampleWindowS`, `elapsedS` |
| `clock_floor_pct`, `thermal_clock_floor_pct`, `thermal_temp_threshold_c` | `clockFloorPct`, `thermalClockFloorPct`, `thermalTempThresholdC` |
| `load_threads`, `load_iters_per_launch`, `load_launches`, `samples_taken` | `loadThreads`, `loadItersPerLaunch`, `loadLaunches`, `samplesTaken` |

### The measurement

| key | canonical | notes |
|-----|-----------|-------|
| `sustained_clock_pct` | `sustainedClockPct` | **the acceptance metric.** Mean SM clock over the window as a percentage of rated boost |
| `sm_clock_mhz` | `smClockMHz` | mean SM clock, absolute; SKU-specific, so a fleet gate belongs on the percentage |
| `mem_clock_mhz` | `memClockMHz` | mean memory clock |
| `rated_boost_clock_mhz` | `ratedBoostClockMHz` | the denominator, read back from the driver so the ratio can be audited |
| `min_sm_clock_pct`, `max_sm_clock_pct` | `minSmClockPct`, `maxSmClockPct` | the spread; a part that dips and recovers is different from one pinned flat |
| `sustained_fma_throughput_tflops` | `sustainedFmaThroughputTflops` | FP32 FMA, whole window. Deliberately **not** `sustainedThroughputTflops` — see below |
| `peak_fma_throughput_tflops` | `peakFmaThroughputTflops` | best single window |
| `throughput_consistency_pct` | `throughputConsistencyPct` | sustained ÷ peak. Catches a part that starts fast and collapses even when the mean still clears the floor |
| `gpu_utilization_pct`, `mem_utilization_pct` | `gpuUtilizationPct`, `memUtilizationPct` | the headline symptom: this reads healthy while the clock does not |

### Thermal and power

| key | canonical |
|-----|-----------|
| `gpu_temp_c` (peak), `mean_temp_under_load_c`, `temp_at_min_clock_c` | `gpuTempC`, `meanTempUnderLoadC`, `tempAtMinClockC` |
| `power_draw_w` (peak), `mean_power_w` | `powerDrawW`, `meanPowerW` |
| `enforced_power_limit_w`, `default_power_limit_w`, `power_limit_ratio_pct` | `enforcedPowerLimitW`, `defaultPowerLimitW`, `powerLimitRatioPct` |

`powerLimitRatioPct` is the PD contract made visible: on a wedged GB10 the
enforced limit sits well below the board's default.

### Throttle evidence

| key | canonical |
|-----|-----------|
| `throttle_events` | `throttleEvents` — *transitions into* a capped state, not a sample count |
| `throttled_samples` | `throttledSamples` — samples with any capping reason latched |
| `throttle_reasons_mask`, `throttle_reasons` | `throttleReasonsMask` (decimal OR across samples), `throttleReasons` (labels) |
| `gpu_idle_samples`, `applications_clocks_setting_samples`, `sw_power_cap_samples`, `hw_slowdown_samples`, `sync_boost_samples`, `sw_thermal_slowdown_samples`, `hw_thermal_slowdown_samples`, `hw_power_brake_samples`, `display_clock_setting_samples` | per-reason sample counts |
| `throttle_classification` | `throttleClassification` — `none` \| `thermal` \| `powerCap` \| `applicationClocks` \| `unknown` |
| `pd_wedge_suspected` | `pdWedgeSuspected` — `true` \| `false` \| `unknown`. `true` when slow **and** cool **and** not thermally throttled; `unknown` when the part is slow but no temperature could be read, since without one a wedge and a thermal throttle are indistinguishable |
| `clock_floor_applied_pct`, `clock_floor_basis` | `clockFloorAppliedPct`, `clockFloorBasis` |

`gpuIdle` is deliberately excluded from what counts as a throttle: an idle GPU is
not throttled, it is unloaded, and counting it would turn a broken load into a
hardware verdict.

### Metrics that are absent rather than zero

A read the driver refuses is **omitted**, listed in `nvml_unsupported`
(`nvmlUnsupported`) and counted in `unsupported_reads` (`unsupportedReads`).

This matters more than it looks. Emitting `throttle_events=0` for a node whose
throttle state was never read would satisfy a `throttleEvents == 0` threshold —
a missing measurement silently passing acceptance, the exact failure
[`pkg/verdict`](../../pkg/verdict/verdict.go) exists to prevent. Omitting
the metric makes that threshold fail instead.

### Why the throughput metric has its own name

`sustainedThroughputTflops` is registered in `pkg/contract` and marked safe to
threshold on, and reusing it here would have been the obvious move. It is the
wrong move. That name belongs to `gpu-burn`'s heavy GEMM burn; this runner's
load is a register-resident FMA chain with no memory traffic, and on the same
healthy part the two figures differ by a large factor.

`pkg/contract`'s own rule is that sharing a name across two differently-obtained
figures is precisely the failure the convention exists to prevent. A profile
author whose threshold was calibrated against `gpu-burn` would silently condemn
every node this runner touched, and a fleet's stored history would interleave
two different quantities under one series.

So the name is `sustainedFmaThroughputTflops`: unregistered, which the registry
explicitly permits ("the registry exists to stop the names we DO own from
drifting apart, not to forbid new measurements"), still thresholdable, and
honestly distinguishable. Registering it — with a description saying what load
produced it — is a reasonable follow-up.

### A note for anyone adding a metric here

Only three `*_mhz` keys are aliased in `pkg/runner`'s `clockprobe` table
(`sm_clock_mhz`, `mem_clock_mhz`, `rated_boost_clock_mhz`). Generic
snake_case→lowerCamelCase folds `mhz` to `Mhz`, which is **not** the registered
`MHz` suffix, so any new unaliased `*_mhz` key lands as a name that declares no
unit — a clock recorded as a bare number that charts happily and compares
against nothing. That is why `min_sm_clock_pct` / `max_sm_clock_pct` are
percentages here rather than the MHz they were measured in. Add the alias first,
or use a unit whose lowercase spelling survives the fold.

## Configuration

| env | default | meaning |
|-----|---------|---------|
| `BURNIN_DURATION_SECONDS` | 60 | total probe wall time; set by the operator. Raised to a 10 s floor, with a `configWarnings` note, because a shorter probe cannot separate clock ramp from steady state |
| `CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT` | 70 | general floor |
| `CLOCKPROBE_MIN_THERMAL_CLOCK_PCT` | 50 | floor when the shortfall is attributable to heat |
| `CLOCKPROBE_THERMAL_TEMP_C` | 80 | mean temperature at or above which heat is a candidate explanation |
| `CLOCKPROBE_SAMPLE_INTERVAL_MS` | 100 | NVML sampling cadence, clamped to [10, 5000] |

An unparseable value is an `Error`, never a silent fallback to the default.

## Build

```sh
docker build -t glimmer-burnin-clockprobe:dev .
```

Build args: `CUDA_IMAGE`, `CUDA_ARCH_FLAGS`.

The default `CUDA_ARCH_FLAGS` builds real cubins for sm_80, sm_90, sm_120 and
sm_121 plus a compute_121 PTX fallback. That is a **deliberate** contrast with
`compute-smoke`, which pins `sm_121a` with no fallback, and it is not a
relaxation of the "do not widen an arch target" rule:

- `compute-smoke` claims *"the real block-scaled FP4 instruction path executed"*.
  A JIT or emulation fallback could produce a passing result without ever
  touching the FP4 units, so the narrow target **is** the proof.
- `clockprobe` makes no claim about which instructions ran. Its measurands are
  achieved clock and the throughput of an ordinary FP32 FMA chain, and PTX JIT
  compiles FFMA one-to-one — a JIT'd loop runs on the same SMs at the same clock
  and can fake neither number. Meanwhile the fault it detects is a fleet-wide
  power-delivery problem, so refusing to run anywhere but GB10 would be the
  actual failure.

## Licensing

Everything in this directory is **Apache-2.0** original work
(`clockprobe.cu`, `nvml_dynamic.h`). This runner links **no third-party library
at all** — the load kernel is plain FP32 FMA and needs no math library, and
`nvml_dynamic.h` declares the ~15 NVML entry points it uses rather than
including the vendor's `nvml.h`. There is no GPL, LGPL, MPL or otherwise
copyleft component, and nothing here is dual-licensed.

The build is multi-stage specifically so the NVIDIA CUDA toolchain is used at
**build time only**. The published image contains our binary on a distroless
base and **no NVIDIA-licensed redistributable libraries**. The build asserts it:
it fails if `ldd` finds any `libcuda*` / `libcudart*` / `libnv*` dependency.

NVML is resolved at runtime by `dlopen`, never linked. That is what makes the
assertion satisfiable — `libnvidia-ml` has no static form, so `-lnvidia-ml`
would put a `DT_NEEDED` entry for an NVIDIA redistributable into the shipped
binary. The driver's own `libnvidia-ml.so.1` and `libcuda.so.1` are injected
into the container at runtime by the NVIDIA Container Toolkit.

## Requirements on the node

- NVIDIA Container Toolkit **≥ 1.17** with the `nvidia` runtime registered
  (`nvidia-ctk runtime configure --runtime=docker`).
- `NVIDIA_DRIVER_CAPABILITIES` must include **`utility`** as well as `compute`.
  The image sets both; a pod that overrides the variable and drops `utility`
  gets no NVML, and this runner exits `Error` rather than pretending the
  hardware is fine.
- The pod must request a GPU (`nvidia.com/gpu`) — a `clockprobe` pod without one
  skips.

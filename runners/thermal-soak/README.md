# `thermal-soak` runner — sustained load, sustained correctness

Holds a heavy, correctness-checked compute load on the accelerator for the whole
requested duration and asserts that the part neither **throttles to protect
itself** nor **starts returning different answers**.

Implements TestKind `thermal-soak`. Shares its entire implementation with
[`../gpu-burn`](../gpu-burn) — same load, same sampler, same reporting engine —
and differs only in its stdout vocabulary and in which question decides the exit
code. See [Two kinds, one implementation](#two-kinds-one-implementation).

---

## Why this test exists

Heat takes time to move. The die reaches steady state in seconds, the heatsink in
minutes, and the chassis in tens of minutes — so a cooling fault is invisible to
every short test in the suite:

| fault | why the short tests miss it |
|---|---|
| dry or misapplied thermal interface | the die is fine for the first 30 s |
| blocked intake, or a chassis racked too close to its neighbour | the air is still cool when a smoke test finishes |
| a fan that spins but moves no air | fan RPM reads healthy; nothing reads airflow |
| marginal silicon that computes correctly cold and not hot | every cold test passes, by construction |

The node then throttles under real work, months later, and looks like a
scheduling problem.

The last row is the one that matters most, and it is why this runner keeps
checking results the whole way through. **A soak that stops checking results is
just a heater.** The interesting failure is not "it got hot", it is "it got hot
*and* started returning different answers".

---

## Two kinds, one implementation

`thermal-soak` and `gpu-burn` are two verdicts over ONE experiment, not two
experiments:

| | asks |
|---|---|
| `thermal-soak` | did it stay at clock, and stay out of protective throttling, for the whole window? |
| `gpu-burn` | did it still get the RIGHT ANSWER while it was that hot? |

Both run the identical GEMM through the identical NVML sampler, so
`sustainedThroughputTflops` from one is the same quantity as from the other and
a fleet can chart them as a single series. Two independently written heaters
would have produced two different numbers under one metric name, which is the
exact failure `pkg/contract`'s naming rules exist to prevent.

`soak_core.cuh` and `nvml_dynamic.h` are **byte-identical copies** in both runner
directories, because the publish workflow builds each runner with its own
directory as the Docker build context and `COPY` cannot reach outside a build
context. `TestSharedSoakSourcesAreIdentical` in
[`soak_contract_test.go`](soak_contract_test.go) fails if the copies drift.
Edit one, copy it to the other.

---

## Workload

A square FP32 SGEMM, `C = A × B`, tiled 64×64 with a 4×4 register micro-tile over
16-deep K slices, run back to back until the deadline. Default `N` is 8192
(four 256 MiB buffers), shrunk automatically if the device has less free memory.

Chosen over the register-resident FMA chain that [`../clockprobe`](../clockprobe)
uses because a soak has a different job: an FMA chain heats the ALUs and nothing
else, while a GEMM also drives shared memory, the register file, the L2 and the
memory controller — which is where a marginal part actually fails.

### The correctness check

The first GEMM's output is kept as a reference. Every later GEMM is compared
against it **bitwise, on the GPU**, and every element that differs is counted.

Bitwise is the right comparison and a tolerance would be wrong. The kernel's
accumulation order is fixed by its schedule, so a healthy part recomputes the
identical bit pattern every time; any difference at all is the hardware changing
its answer between two runs of the same arithmetic. A tolerance would hide
exactly the single-bit flips this exists to catch.

The reference is computed **on the part under test**, which bounds what this can
prove: it detects a part that stops agreeing with *itself*, not one that was
wrong from the first instruction. That is deliberate — a host-computed FP32
reference would differ in the last bits for reasons that have nothing to do with
hardware health, and every node in the fleet would fail.
[`../compute-smoke`](../compute-smoke) is the test that asserts the arithmetic is
right at all.

---

## Output contract

| Result | stdout | Exit |
|---|---|---|
| Pass | `THERMAL_SOAK_PASS` | 0 |
| Fail | `THERMAL_SOAK_FAIL: <reason>` | 1 |
| Not applicable | `THERMAL_SOAK_SKIP: <reason>` | 2 |
| Unjudged | `THERMAL_SOAK_ERROR: <reason>` | 3 |

Metrics are `key=value` lines on stdout, printed **before** the decision, so a
failing or erroring run still yields its evidence.

They are also **re-printed periodically during the run** (every
`BURNIN_SOAK_PROGRESS_INTERVAL_S`, default 15 s). Parsing is
last-occurrence-wins, so a log truncated at any point — a pod killed at its
deadline, a mid-run checkpoint delivery, an engineer tailing stdout — still
parses into a complete, coherent snapshot of the soak so far instead of into
nothing at all. A soak that only reports at the end has no evidence for the case
where it never reaches the end, which is precisely the case worth having
evidence for.

### Skip (exit 2) — the test does not apply

- no accelerator is visible to the container
- MIG is enabled: clock, temperature and power are device-wide properties and
  cannot be attributed to one instance's load

### Error (exit 3) — we do not know, and will not guess

- the CUDA driver is not usable from this container at all — no `--gpus`, or a
  broken container toolkit. The message is
  `cudaGetDeviceCount: CUDA driver version is insufficient for CUDA runtime
  version`. This is Error and **not** Skip on purpose, and it is the same
  behaviour as [`../clockprobe`](../clockprobe): the operator only schedules this
  pod onto a node that advertises an accelerator, so "the driver is missing" is a
  fleet-wide misconfiguration. Reporting Skip would let that read as "not
  applicable" on every node at once, silently. Only `cudaErrorNoDevice` — the
  driver answering that there is no accelerator — is a Skip.
- `libnvidia-ml.so.1` is absent while an accelerator IS visible. That is a
  container/driver misconfiguration (`NVIDIA_DRIVER_CAPABILITIES` must include
  `utility`), not a property of the hardware. Skipping here would quietly report
  "not applicable" for every node in a fleet whose toolkit was set up wrong.
- device memory could not be allocated, or a kernel launch failed
- the soak completed no iterations
- mean GPU utilization under load came out below 50 %. The load did not land, so
  the sampled clock and temperature are not attributable to this test. Calling
  that a hardware failure would condemn a node for the runner's own problem.
- a malformed environment variable

### Fail (exit 1) — a verdict about the hardware

Checked in this order, so the most serious finding is the one reported:

1. `miscompares > 0` — silent data corruption. Reported first because a part
   that got the wrong answer has failed regardless of how cool it stayed.
2. `nonfiniteCount > 0` — a NaN or Inf in a result whose operands are all finite.
3. `throttleEvents > 0` — the part had to protect itself. **This is the thermal
   verdict proper.**
4. `sustainedClockPct` below `THERMAL_SOAK_MIN_CLOCK_PCT`.
5. peak temperature above `THERMAL_SOAK_MAX_TEMP_C`, **only if that variable is
   set** — see [What is deliberately not asserted](#what-is-deliberately-not-asserted).

---

## What is deliberately not asserted

**A temperature ceiling, by default.** There is no portable "too hot". The safe
steady-state temperature of a part depends on its SKU, its cooling class, its
firmware fan curve and the inlet temperature of the room, and the driver already
enforces the only limit that is a property of the silicon. A hard-coded ceiling
would either fail healthy dense nodes or pass nothing at all in a warm aisle.

What IS portable is *did the part have to protect itself* — `throttleEvents` —
and that is what this runner decides on. Set `THERMAL_SOAK_MAX_TEMP_C` (and the
matching `gpuTempC` threshold) only once you have a baseline for **one hardware
class in one room**, and re-measure it for any other class.

**`swPowerCap` is not counted as a throttle.** `throttleEvents` counts
transitions into a *protective* state only — `hwSlowdown`, `swThermalSlowdown`,
`hwThermalSlowdown`, `hwPowerBrake`. Sitting at the board power limit under a
heavy GEMM is what a healthy accelerator *does*; it is power management working,
not the part protecting itself. Counting it would make `throttleEvents Equal 0`
fail on every healthy node under real load, which is the same class of mistake as
gating GB10 on a 90 % sustained clock. It is still reported, as
`power_cap_events` and `sw_power_cap_samples`, because a part that spends its
whole soak power-capped is worth seeing.

---

## Metrics

Runner keys on the left, the canonical name `pkg/runner` maps them to on the
right. The mapping for this kind lives in `pkg/runner/parse.go` under
`"thermal-soak"`.

### The acceptance metrics

| key | canonical | notes |
|---|---|---|
| `throttle_count` | `throttleEvents` | **the thermal verdict.** Transitions into a protective throttle, not a sample count |
| `miscompares` | `miscompares` | elements whose bit pattern changed between two runs of identical arithmetic |
| `sdc_detections` | `sdcDetections` | iterations in which at least one element differed. One corrupted region gives many miscompares and one detection |
| `nonfinite_count` | `nonfiniteCount` | NaN/Inf in the result |
| `sustained_clock_pct` | `sustainedClockPct` | mean SM clock over the window, as a percentage of rated boost |
| `peak_temp_c` | `gpuTempC` | peak GPU temperature during the soak |
| `soak_seconds` | `elapsedS` | wall time the soak body ran |
| `iterations_completed` | `iterationsCompleted` | GEMM+verify iterations, including warm-up |
| `ecc_errors` | `eccErrors` | delta over the soak window, **or the `n/a` sentinel** — see [ECC](#ecc-and-the-unmeasurable-sentinel) |

### Load, thermal and power evidence

| key | canonical |
|---|---|
| `peak_power_w`, `mean_power_w` | `powerDrawW` (peak), `meanPowerW` |
| `mean_temp_c` | `meanTempC` |
| `sustained_throughput_tflops` | `sustainedThroughputTflops` — whole-window average; the same measurand gpu-burn reports as `tflops` |
| `gemm_active_s` | `gemmActiveS` — time the timed GEMM was actually executing |
| `gpu_utilization_pct`, `mem_utilization_pct` | `gpuUtilizationPct`, `memUtilizationPct` |
| `smClockMHz`, `memClockMHz`, `ratedBoostClockMHz` | passed through unchanged |
| `min_sm_clock_pct`, `max_sm_clock_pct` | `minSmClockPct`, `maxSmClockPct` — the spread; a part that dips and recovers differs from one pinned flat |

The three clock keys are emitted in **lowerCamelCase**, not `..._mhz`. That is
not a style inconsistency: `pkg/runner` passes an already-camelCase key through
unchanged, whereas `rated_boost_clock_mhz` would normalise to
`ratedBoostClockMhz` — and `Mhz` is not the registered `MHz` suffix, so the clock
would be stored as a dimensionless number that charts happily and compares
against nothing. Only the `clockprobe` kind has alias-table entries for the
snake_case spelling.

### Throttle evidence

| key | canonical |
|---|---|
| `throttled_samples` | `throttledSamples` — samples with a protective reason latched |
| `throttle_reasons_mask`, `throttle_reasons` | `throttleReasonsMask` (decimal OR across samples), `throttleReasons` (labels) |
| `power_cap_events`, `sw_power_cap_samples` | `powerCapEvents`, `swPowerCapSamples` — reported, but not counted as throttling |
| `gpu_idle_samples`, `applications_clocks_setting_samples`, `hw_slowdown_samples`, `sync_boost_samples`, `sw_thermal_slowdown_samples`, `hw_thermal_slowdown_samples`, `hw_power_brake_samples`, `display_clock_setting_samples` | per-reason sample counts |
| `thermal_throttle_latched` | `thermalThrottleLatched` — `true` \| `false` |

### Identity, configuration and self-audit

| key | canonical |
|---|---|
| `gpu_name`, `compute_cap`, `pci_bus_id`, `driver_version`, `mig_mode`, `ecc_mode` | `gpuName`, `computeCap`, `pciBusId`, `driverVersion`, `migMode`, `eccMode` |
| `duration_requested_s`, `warmup_s`, `sample_interval_ms`, `progress_interval_s`, `matrix_n` | `durationRequestedS`, `warmupS`, `sampleIntervalMs`, `progressIntervalS`, `matrixN` |
| `clock_floor_pct`, `temp_ceiling_c` | `clockFloorPct`, `tempCeilingC` — emitted **before** the soak, so a run killed at its pod deadline still records what it was going to be judged against |
| `samples_taken`, `snapshot_seq` | `samplesTaken`, `snapshotSeq` |
| `first_miscompare_index` | `firstMiscompareIndex` — only when there was one |
| `faults_injected` | `faultsInjected` — always emitted; see [Proving the detector works](#proving-the-detector-works) |
| `nvml_unsupported`, `unsupported_reads`, `config_warnings` | `nvmlUnsupported`, `unsupportedReads`, `configWarnings` |

### Metrics that are absent rather than zero

Every counter above is printed **only if the driver actually answered**. A
fabricated `throttle_count=0` would satisfy a `throttleEvents Equal 0` threshold
on a node whose throttle state was never read — a missing measurement silently
passing acceptance. Omitting the key makes such a threshold fail instead, which
is the fail-closed rule `pkg/verdict` is built on.

On GB10 the memory clock is one of these: NVML answers `NOT_SUPPORTED` for the
memory domain on a unified-memory part, so `memClockMHz` is absent and
`nvml_unsupported=memClock` records why.

### ECC and the unmeasurable sentinel

ECC is the one counter a runner may **declare unmeasurable** rather than omit,
and only when the part itself said it has no ECC subsystem. The runner reads
`nvmlDeviceGetEccMode` first and follows the same three-state logic as
[`../host-health`](../host-health):

| `nvmlDeviceGetEccMode` | meaning | what is emitted |
|---|---|---|
| `NOT_SUPPORTED` | the part has no ECC subsystem to have counters | `ecc_errors=n/a` — the sentinel, which lands in `Result.Unmeasurable` and never in `Result.Metrics` |
| success (`enabled` or `disabled`) | ECC hardware exists | the volatile corrected+uncorrected **delta over the soak window** — or, if the counters could not be read, the key is **omitted** so the gate fails closed |
| any other error | we could not look | the key is omitted |

The middle row is the important one: on a part that HAS ECC, an unreadable
counter means either that ECC was switched off or that our read failed. Both
leave the node unproven. Collapsing that into the sentinel would hand every fleet
a way to pass an ECC gate by turning ECC off.

**Verified on GB10:** `ecc_mode=unsupported`, and the runner emits
`ecc_errors=n/a`. GB10's unified LPDDR5X has on-die ECC that NVML cannot see, so
a bare `eccErrors Equal 0` fails every healthy Spark — pair it with
`applicability: RequiredIfMeasurable` (below).

---

## Recommended thresholds

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: thermal-soak
spec:
  kind: thermal-soak
  scope: Node
  durationSeconds: 1800
  thresholds:
    # The thermal verdict. Exact, because it is a dimensionless counter and a
    # tolerance around zero protective throttles is a tolerance for throttling.
    - metric: throttleEvents
      comparison: Equal
      value: "0"

    # Correctness under heat. A soak that stops checking results is just a
    # heater, and these are the two metrics that make it a test.
    - metric: nonfiniteCount
      comparison: Equal
      value: "0"
    - metric: miscompares
      comparison: Equal
      value: "0"

    # The clock floor. A BACKSTOP against a wedged part, not a performance
    # gate — and the number depends on how long the soak runs. 60 is right for
    # a 30-minute soak on GB10; see below before reusing it.
    - metric: sustainedClockPct
      comparison: GreaterThanOrEqual
      value: "60"

    # A soak that exited early has not proven what its duration claims. Set this
    # slightly below spec.durationSeconds.
    - metric: elapsedS
      comparison: GreaterThanOrEqual
      value: "1750"

    # RequiredIfMeasurable, not Required: GB10 declares ECC unmeasurable, and
    # under Required that declaration fails the node exactly like a missing
    # metric. This value relaxes the gate ONLY for the runner's explicit
    # declaration — a metric that merely goes missing still fails closed.
    - metric: eccErrors
      comparison: Equal
      value: "0"
      applicability: RequiredIfMeasurable
```

### The clock floor: 60, not 90 — and it depends on how long you soak

Measured on a DGX Spark (GB10, rated boost **3003 MHz** read back from the
driver, driver 580.82.09) running this exact load, 2026-08-03. Every row is the
same healthy part:

| elapsed into the soak | sustained | peak temp | peak power | throttle events |
|---|---|---|---|---|
| 25 s | 82.21 % | 72 °C | 89.98 W | 0 |
| 45 s | 80.50 % | 80 °C | 87.31 W | 0 |
| 60 s | 78.13 % | 82 °C | 86.17 W | 0 |
| 120 s | 72.32 % | 82 °C | 86.17 W | 0 |
| 240 s | 70.66 % | 82 °C | 86.17 W | 0 |
| **600 s** | **69.90 %** | 83 °C | 86.17 W | **0** |

The asymptote is not a fuzzy average — it is a **discrete P-state**. The minimum
sampled clock in every long run is 69.26 % of rated boost, which is 2080 MHz
exactly, and the 600 s mean of 69.90 % is that state plus the ramp it took to get
there. A confirmation soak started from an already-hot part reported 69.46 % at
two minutes and 69.48 % at ten, with `smClockMHz=2087`: the part finds 2080 MHz
and stays there.

Three things fall out of that table, and each one kills a plausible threshold.

**1. `>= 90` fails a healthy part instantly.** Rated boost is a short-burst
number; nothing sustains it under a dense GEMM. A gate at 90 fires on everything,
and a gate that fires on everything gets disabled — and then it is not watching
when something really does wedge.

**2. The sustained clock is a function of soak DURATION, not just of the part.**
A floor calibrated on a 60-second run (78 %) condemns every node in a 30-minute
run (70 %). This is not drift or noise: it is thermal mass reaching steady state,
asymptotic, settling near 69.9 % from about the four-minute mark. **Calibrate
against a soak of the duration you intend to run.** It is also load-dependent —
[`../clockprobe`](../clockprobe)'s register-resident FMA chain holds ~83 % on the
same part because it never touches memory — so a floor calibrated against one
runner does not transfer to the other either.

**3. It happens with ZERO throttle events.** Across the whole 600 s the reason
mask stayed `0`: no thermal slowdown, no hardware slowdown, not even
`swPowerCap`. The part is managed down to hold ~83 °C and ~86 W without anything
being latched for NVML to report.

Point 3 is the one to carry away: **on this hardware `sustainedClockPct` is a
backstop, not the thermal verdict.** The verdict is `throttleEvents`. The floor
exists to catch a part wedged far below its class, not to grade performance.

**60 is derived from the measured side alone:** ten points of headroom under the
69.9 % asymptote in the table above. The wedge is *not* an input to it. No
power-delivery-wedged part has ever been put through this runner
([#61](https://github.com/baldwinSPC/glimmer-burnin/issues/61)) — this project
**estimates** a wedged GB10 sits near **20 %** of rated boost, from a researched
pin point of ~611 MHz against the 3003 MHz rated boost this fleet reports
(611/3003 = 20.3 %), which agrees in order of magnitude with GEP-0178's
independent "~4× slow while reporting 96 % utilization". Earlier revisions of
this file said 30–50 % and described 60 as "ten points above the wedge ceiling";
that figure had no derivation behind it, so the gap it appeared to split was not
real. What the estimate does support is only the weak claim that the floor
clears a wedge comfortably: a part at 20 % misses this floor by forty points,
and would have missed it under the discarded figure too. Do not tighten the
floor towards a wedge estimate — measure one first.

This was not theoretical. The default shipped in the first draft of this runner
was 75, chosen from the 25-second measurement, and the 600-second calibration run
failed a perfectly healthy Spark with
`THERMAL_SOAK_FAIL: sustained SM clock 69.9% of rated boost … below the 75.0%
floor`. **Recalibrate per hardware class and per duration from your own fleet's
distribution rather than inheriting this number for a part it was not measured
on.**

---

## Configuration

| env | default | meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 900 | total soak wall time; set by the operator from `spec.durationSeconds`. Raised to a 15 s floor, with a `config_warnings` note, because a shorter soak cannot reach a steady thermal or clock state |
| `THERMAL_SOAK_MIN_CLOCK_PCT` | 60 | sustained-clock floor, as a percentage of the part's own rated boost clock. A wedge backstop, not a performance gate — and duration-dependent; see above |
| `THERMAL_SOAK_MAX_TEMP_C` | 0 (off) | peak-temperature ceiling. Off by default; only meaningful against a per-class fleet baseline |
| `BURNIN_SOAK_MATRIX_N` | 8192 | GEMM dimension. Rounded down to a multiple of 64 and shrunk to fit free device memory |
| `BURNIN_SOAK_SAMPLE_INTERVAL_MS` | 250 | NVML sampling cadence, clamped to [10, 5000] |
| `BURNIN_SOAK_PROGRESS_INTERVAL_S` | 15 | how often the full metric block is re-emitted, clamped to [5, 3600] |
| `BURNIN_SOAK_INJECT_MISCOMPARES` | 0 | **self-test only** — see below |

A value that cannot be parsed is a configuration Error, not a default: guessing
would run a different test than the one that was asked for while reporting
success.

Warm-up is `clamp(duration/10, 5 s, 30 s)` and is excluded from the clock,
temperature and throughput windows — an unwarmed part sits at idle clock
legitimately. Correctness **is** checked during warm-up.

### Proving the detector works

`BURNIN_SOAK_INJECT_MISCOMPARES=N` corrupts N elements of the reference on
purpose, so the compare path fires on known-good hardware.

A silent-data-corruption check that has never once fired is an assertion nobody
has tested: if the compare kernel were mis-launched or the counter never read
back, the runner would report `miscompares=0` for the rest of its life and every
node in the fleet would pass a test that was measuring nothing.

A run with injection enabled is **not a hardware verdict**, and it says so twice
— `faults_injected` is emitted on every run (0 when the knob is unset, so its
absence from a stored envelope is itself informative), and an injected run also
carries a `config_warnings` entry. Never set it in a profile.

---

## Build

```sh
docker build -t ghcr.io/baldwinspc/glimmer-burnin-thermal-soak:vX.Y.Z runners/thermal-soak
```

Build args: `CUDA_IMAGE` (default `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04`)
and `CUDA_ARCH_FLAGS`.

**Host architecture is a separate axis from GPU architecture.** The image is
published as a manifest list for `linux/amd64` **and** `linux/arm64`; the gencode
list below is the same on both, because an `sm_90` cubin is the same device code
whether the host is x86 or Grace.

The default arch list ships real cubins for sm_80 / sm_90 / **sm_100** / sm_120 /
sm_121 plus a PTX fallback. With CUDA's minor-version binary compatibility (a
cubin for CC X.Y runs on CC X.Z where Z ≥ Y) that covers A100/A10/L40S (`sm_80`),
H100/H200/GH200 (`sm_90`), B200/B300/GB200/GB300 (`sm_100`), RTX PRO 6000
Blackwell (`sm_120`) and GB10 (`sm_121`). `sm_100` was the one gap that mattered
for x86 adopters: `compute_121` PTX cannot JIT **down** onto CC 10.x, so before
it a B200 had no code in this image at all, and a soak that cannot launch on the
part it was sent to is an `Error` on every node of a Blackwell fleet.

The list is deliberately wider than `compute-smoke`'s single `sm_121a`
target. `compute-smoke` pins one arch with no fallback because its claim is "the
real block-scaled FP4 instruction path executed", so the narrow target IS the
proof. A soak claims something else: that the part held its clock and kept
agreeing with itself under ordinary FP32 FMA. PTX JIT compiles FFMA one-to-one,
so a JIT-compiled GEMM runs on the same SMs at the same clock and produces the
same bit patterns — it cannot fake any of the three measurands. Meanwhile the
faults this catches are fleet-wide, so an image that refuses to run on anything
but GB10 would be the actual failure.

The `publish-runner` workflow's `CUDA_ARCH` input is deliberately **not**
consumed by this Dockerfile, for that reason. Override `CUDA_ARCH_FLAGS` to trim
or extend the list for your fleet.

### Prerequisites for publishing

Both were outside this directory and are **now done**; they are kept here as the
record of what a new runner still owes.

1. `thermal-soak` is in the `runner` choice list in
   `.github/workflows/publish-runner.yml`. Without it the image cannot be built
   by the workflow at all, and `runners/pins_test.go` fails a runner that is
   missing from that list.
2. `internal/controller/pods.go` pins a `defaultRunnerImages` entry. Before a tag
   existed the kind deliberately had none, so it failed fast at plan time asking
   for an explicit `spec.runner.image` rather than pull-failing on every node —
   which is still the correct state for a kind whose runner has not been
   published.

---

## Verified on hardware

Built and run on a DGX Spark (`spark-85a9`, aarch64, NVIDIA GB10, compute
capability 12.1, driver 580.82.09, Docker 28.3.3, native arm64 build) on
2026-08-03, via `docker run --gpus all`:

**Pass.** A 45 s soak and a 600 s soak both reported `THERMAL_SOAK_PASS`, exit 0.
The 600 s run: 3984 iterations, `miscompares=0`, `sdc_detections=0`,
`nonfinite_count=0`, `throttle_count=0`, `throttle_reasons=none`,
`sustained_clock_pct=69.48`, `smClockMHz=2087`, peak 83 °C, `ecc_errors=n/a`.

**Fail.** Exercised with a deliberately impossible floor
(`THERMAL_SOAK_MIN_CLOCK_PCT=99`):
`THERMAL_SOAK_FAIL: sustained SM clock 78.5% of rated boost over the soak, below
the 99.0% floor (peak 82C)`, exit 1, with every metric still on stdout. The
miscompare path was proven separately with `BURNIN_SOAK_INJECT_MISCOMPARES` —
see [`../gpu-burn`](../gpu-burn#verified-on-hardware) for that transcript; it is
the same shared code.

**Error.** Started without `--gpus`: `THERMAL_SOAK_ERROR: cudaGetDeviceCount:
CUDA driver version is insufficient for CUDA runtime version`, exit 3, with the
pre-soak configuration lines already on stdout.

**What could not be exercised on healthy hardware**: a real protective throttle,
a real miscompare, a real non-finite result, a real ECC event (GB10 has none to
report), a temperature-ceiling failure, and the MIG skip path. The clock-floor
and miscompare decision paths were proven by forcing them; the throttle branch
has never fired on this fleet, because nothing in it has ever throttled.

---

## Licensing

| component | licence | shipped? |
|---|---|---|
| `thermal_soak.cu`, `soak_core.cuh`, `nvml_dynamic.h` | Apache-2.0 (this project) | yes |
| third-party source | **none** | — |
| CUDA toolchain | NVIDIA CUDA EULA | **no** — build stage only, multi-stage |
| `libcuda.so.1`, `libnvidia-ml.so.1` | NVIDIA driver | **no** — injected at runtime by the NVIDIA Container Toolkit |

The load is a hand-written SGEMM, so unlike a cuBLAS-based burn there is no
NVIDIA toolkit library to ship and no site mount to arrange. See
[`../gpu-burn`](../gpu-burn) for the full reasoning; it is the same decision.

The build **fails** rather than shipping an image that drags in an NVIDIA `.so`:

```dockerfile
RUN ldd /out/thermal_soak | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libcublas|libnv' /out/ldd.txt
```

Verified: the shipped binary links `libc.so.6` and nothing else.

---

## Requirements on the node

- an NVIDIA accelerator, and the NVIDIA Container Toolkit
- `NVIDIA_DRIVER_CAPABILITIES` must include **`compute`** (mounts
  `libcuda.so.1`) and **`utility`** (mounts `libnvidia-ml.so.1`). The image sets
  both. Without `utility` there is no temperature, clock or throttle telemetry
  and the runner exits Error rather than pretending the hardware is fine.
- device memory for four `N × N` float buffers (1 GiB at the default `N=8192`),
  shrunk automatically if less is free
- runs as non-root (`65532:65532`); no privileges, no host mounts

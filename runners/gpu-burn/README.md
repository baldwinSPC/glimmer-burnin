# `gpu-burn` runner — wrong answers under sustained load

Holds a heavy compute load on the accelerator and asserts that it keeps returning
the **same answer to identical arithmetic**. The measurand is correctness under
load, not heat.

Implements TestKind `gpu-burn`. Shares its entire implementation with
[`../thermal-soak`](../thermal-soak) — same load, same sampler, same reporting
engine — and differs only in its stdout vocabulary and in which question decides
the exit code.

---

## Why this test exists

Every other test in the suite asserts that something *reports* healthy. This one
asserts that the arithmetic is still **right**.

The fault it catches is a part that returns the wrong answer and reports success.
Not a crash, not an Xid, not an ECC event — a bit that is different this time
from last time, on hardware whose every status register says it is fine. The
wrong number then flows into a training run and nothing anywhere records that it
happened.

On a part with visible ECC that fault is usually caught and counted. On a part
**without** — GB10's unified LPDDR5X has on-die ECC that NVML cannot see, so
`eccErrors` is unmeasurable there — a recompute-and-compare burn is the only
instrument the fleet has for it. That is exactly the hardware this project's
first fleet runs.

---

## Why this is not `wilicc/gpu-burn`

The obvious implementation is Ville Timonen's
[gpu-burn](https://github.com/wilicc/gpu-burn), and this TestKind is named after
it. **Its licence is not the problem.** `LICENSE` in that repository is
**BSD-2-Clause** (verified 2026-08-03), which is on this project's permissive
allow-list.

What it **links** is the problem. gpu-burn's numeric core is cuBLAS
(`cublas_v2.h`, `cublasSgemm`, `cublasDgemm`, `cublasSetMathMode`), and cuBLAS is
covered by the NVIDIA CUDA EULA. Critically, **cuBLAS is a toolkit library, not a
driver library**: the NVIDIA Container Toolkit injects `libcuda.so.1` and
`libnvidia-ml.so.1` from the host *driver* at runtime, and does not inject
cuBLAS.

Verified on the hardware rather than assumed, on a GB10 Spark with driver
580.82.09:

```
$ ldd cublas_check | grep -iE 'cublas'
        libcublas.so.13 => /usr/local/cuda/lib64/libcublas.so.13
        libcublasLt.so.13 => /usr/local/cuda/lib64/libcublasLt.so.13
$ ./cublas_check
cublas_version=130002 sgemm_status=0 sync=no error
device=NVIDIA GB10 cc=12.1
```

…and, inside a container started with `--gpus all`, the toolkit-injected library
directory contains the driver libraries and **zero** cuBLAS libraries:

```
$ ls /usr/lib/aarch64-linux-gnu/ | grep -E '^libcuda\.so|^libnvidia-ml\.so'
libcuda.so.1
libcuda.so.580.82.09
libnvidia-ml.so.1
libnvidia-ml.so.580.82.09
$ ls /usr/lib/aarch64-linux-gnu/ | grep -ci cublas
0
$ ls /usr/local/cuda/lib64/ | grep -c cublas      # from the IMAGE, not injected
8
```

So **cuBLAS 13 does support sm_121 and does run on GB10 — the blocker is
redistribution, not capability.** A working cuBLAS-based image has to ship
libcublas itself, which is precisely the redistribution this project's runner
images may not do, or adopt the posture [`../dcgm-diag`](../dcgm-diag) was forced
into, where the **site** mounts the NVIDIA component and the runner reports Error
rather than a verdict on any node where that mount is missing.

That posture is a real cost, and it is worth paying only when the upstream tool
does something we cannot. Here it does not. gpu-burn's method is three lines of
idea: run a big GEMM repeatedly, compare each result against the first, count the
elements that differ. The value is in the method, not in cuBLAS. A self-contained
hand-written SGEMM:

- keeps the image redistributable, with **no** site mount and no Error-on-missing-mount path;
- makes this runner's `sustainedThroughputTflops` the *same measurement* as
  thermal-soak's, because it is literally the same kernel, rather than two
  different numbers reported under one metric name.

cuBLAS would reach a higher absolute TFLOPS. It would not detect one more wrong
answer.

---

## Workload

Identical to [`../thermal-soak`](../thermal-soak): a square FP32 SGEMM tiled
64×64 with a 4×4 register micro-tile, default `N=8192`, run back to back until
the deadline; the first result is kept as a reference and every later result is
compared against it **bitwise, on the GPU**.

Bitwise is the right comparison and a tolerance would be wrong. The kernel's
accumulation order is fixed by its schedule, so a healthy part recomputes the
identical bit pattern every time; any difference at all is the hardware changing
its answer. A tolerance would hide exactly the single-bit flips this exists to
catch.

The reference is computed **on the part under test**, so this detects a part that
stops agreeing with *itself*, not one that was wrong from the first instruction
— a host-computed FP32 reference would differ in the last bits for reasons that
have nothing to do with hardware health.
[`../compute-smoke`](../compute-smoke) is the test that asserts the arithmetic is
right at all.

---

## Output contract

| Result | stdout | Exit |
|---|---|---|
| Pass | `GPU_BURN_PASS` | 0 |
| Fail | `GPU_BURN_FAIL: <reason>` | 1 |
| Not applicable | `GPU_BURN_SKIP: <reason>` | 2 |
| Unjudged | `GPU_BURN_ERROR: <reason>` | 3 |

Metrics are `key=value` lines printed **before** the decision, and re-printed
every `BURNIN_SOAK_PROGRESS_INTERVAL_S` (default 15 s) during the burn. Parsing
is last-occurrence-wins, so a truncated log still parses into a complete snapshot
of the burn so far.

**Fail (exit 1)**, in order: `miscompares > 0`, then `nonfiniteCount > 0`, then
`eccErrors > 0` — the last only when a real ECC delta could be computed.

**Skip (exit 2)**: no accelerator visible; MIG enabled.

**Error (exit 3)**: the CUDA driver is not usable from this container at all (no
`--gpus`, or a broken container toolkit — reported as `cudaGetDeviceCount: CUDA
driver version is insufficient for CUDA runtime version`); NVML absent while an
accelerator IS visible; allocation or launch failure; no iterations completed;
mean GPU utilization below 50 % under load, which makes the run unattributable.

The first of those is Error and **not** Skip on purpose, matching
[`../clockprobe`](../clockprobe): the operator only schedules this pod onto a
node that advertises an accelerator, so a missing driver is a fleet-wide
misconfiguration, and reporting Skip would let it read as "not applicable" on
every node at once. Only `cudaErrorNoDevice` — the driver answering that there is
no accelerator — is a Skip.

Temperature and clock are **reported but never judged here** — those are
thermal-soak's verdict, over the identical load. Two kinds, two assertion sets,
one implementation.

---

## Metrics

Runner keys on the left, canonical names on the right. The mapping lives in
`pkg/runner/parse.go` under `"gpu-burn"`, and it deliberately uses upstream
gpu_burn's own vocabulary so that a site running the upstream tool and a site
running this image produce the same canonical metrics.

| key | canonical | notes |
|---|---|---|
| `errors` | `miscompares` | gpu_burn's word for a wrong answer. Kept distinct from `eccErrors`: **a miscompare with no ECC event is the silent-corruption case** |
| `sdc_detections` | `sdcDetections` | iterations in which at least one element differed; one corrupted region gives many miscompares and one detection |
| `nonfinite_count` | `nonfiniteCount` | NaN/Inf in a result whose operands are all finite |
| `ecc_errors` | `eccErrors` | delta over the burn window, **or the `n/a` sentinel** — see below |
| `iterations` | `iterationsCompleted` | GEMM+verify iterations, including warm-up |
| `elapsed_s` | `elapsedS` | wall time the burn body ran |
| `tflops` | `sustainedThroughputTflops` | **not** `throughputTflops` — see below |
| `gpu_temp_c`, `mean_temp_c` | `gpuTempC` (peak), `meanTempC` | reported, not judged |
| `power_draw_w`, `mean_power_w` | `powerDrawW` (peak), `meanPowerW` | reported, not judged |
| `throttle_events`, `power_cap_events` | `throttleEvents`, `powerCapEvents` | reported, not judged |
| `sustained_clock_pct`, `min_sm_clock_pct`, `max_sm_clock_pct` | `sustainedClockPct`, `minSmClockPct`, `maxSmClockPct` | reported, not judged |
| `smClockMHz`, `memClockMHz`, `ratedBoostClockMHz` | passed through unchanged | lowerCamelCase on purpose — `..._mhz` would normalise to `Mhz`, which is not the registered `MHz` suffix, and the clock would be stored as a dimensionless number |
| `first_miscompare_index` | `firstMiscompareIndex` | only when there was one |
| `faults_injected` | `faultsInjected` | always emitted; see [Proving the detector works](#proving-the-detector-works) |
| `gpu_name`, `compute_cap`, `pci_bus_id`, `driver_version`, `mig_mode`, `ecc_mode` | identity evidence | |
| `nvml_unsupported`, `unsupported_reads`, `config_warnings` | self-audit | |

`tflops` maps to `sustainedThroughputTflops`, not to `throughputTflops`. The
latter is compute-smoke's single unwarmed launch and `pkg/contract` marks it
unsafe to threshold on; sharing the name would silently make this benchmark
ungateable, and would interleave two different quantities in one stored series.

Every counter is printed **only if the driver actually answered**. A fabricated
zero would satisfy a threshold on a node whose state was never read.

### ECC and the unmeasurable sentinel

`gpu-burn` follows the same three-state logic as
[`../host-health`](../host-health), reading `nvmlDeviceGetEccMode` first:

| `nvmlDeviceGetEccMode` | meaning | emitted |
|---|---|---|
| `NOT_SUPPORTED` | the part has no ECC subsystem to have counters | `ecc_errors=n/a` — the reserved sentinel |
| success (`enabled`/`disabled`) | ECC hardware exists | the volatile corrected+uncorrected delta over the burn — or, if unreadable, the key is **omitted** so the gate fails closed |
| any other error | we could not look | omitted |

The middle row is the one that matters: on a part that HAS ECC, an unreadable
counter means either ECC was switched off or our read failed. Both leave the node
unproven, and collapsing that into the sentinel would hand every fleet a way to
pass an ECC gate by turning ECC off.

**Verified on GB10:** `ecc_mode=unsupported`, and the runner emits
`ecc_errors=n/a` on every snapshot. That declaration is what lets
`applicability: RequiredIfMeasurable` report the gate as *not evaluated* instead
of failing a healthy Spark — and it is also why `miscompares` carries the whole
weight of this test on that hardware.

---

## Recommended thresholds

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: gpu-burn
spec:
  kind: gpu-burn
  scope: Node
  durationSeconds: 1800
  thresholds:
    # The whole point of the test. Exact, because it is a dimensionless counter
    # and a tolerance around zero wrong answers is a tolerance for wrong answers.
    - metric: miscompares
      comparison: Equal
      value: "0"
    - metric: nonfiniteCount
      comparison: Equal
      value: "0"

    # RequiredIfMeasurable, not Required: GB10 declares ECC unmeasurable, and
    # under Required that declaration fails the node exactly like a missing
    # metric. This relaxes the gate ONLY for the runner's explicit declaration —
    # a metric that merely goes missing still fails closed.
    - metric: eccErrors
      comparison: Equal
      value: "0"
      applicability: RequiredIfMeasurable

    # A burn that exited early did less work than its duration claims. Set both
    # slightly below what the run should achieve. Measured on GB10 at N=8192:
    # ~6.6 iterations/s once the part is at its steady-state clock, so a
    # 1800 s burn does ~12000 — 1000 is a "did it do any work at all" floor,
    # not a performance gate.
    - metric: elapsedS
      comparison: GreaterThanOrEqual
      value: "1750"
    - metric: iterationsCompleted
      comparison: GreaterThanOrEqual
      value: "1000"
```

Do **not** add a `sustainedThroughputTflops` floor without measuring your own
fleet first: the figure is specific to this kernel, this `N`, and this part, and
it moves with the clock. Measured on GB10 at `N=8192`, ~7.7 TFLOPS on a 45 s
burn and lower once the part settles into its steady-state clock — see
[`../thermal-soak`](../thermal-soak#the-clock-floor-60-not-90--and-it-depends-on-how-long-you-soak)
for that curve. It is a real benchmark number (unlike compute-smoke's) and is
safe to threshold on, but only against a distribution you have actually
observed, at the duration you actually run.

Leave temperature and clock gates to `thermal-soak`, which runs the identical
load and owns that verdict. Two tests gating the same measurand from the same
load is one place for them to disagree.

---

## Configuration

| env | default | meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 900 | total burn wall time; set by the operator. Raised to a 15 s floor with a `config_warnings` note |
| `BURNIN_SOAK_MATRIX_N` | 8192 | GEMM dimension. Rounded down to a multiple of 64 and shrunk to fit free device memory |
| `BURNIN_SOAK_SAMPLE_INTERVAL_MS` | 250 | NVML sampling cadence, clamped to [10, 5000] |
| `BURNIN_SOAK_PROGRESS_INTERVAL_S` | 15 | how often the full metric block is re-emitted, clamped to [5, 3600] |
| `BURNIN_SOAK_INJECT_MISCOMPARES` | 0 | **self-test only** — see below |

A value that cannot be parsed is a configuration Error, not a default.

### Proving the detector works

`BURNIN_SOAK_INJECT_MISCOMPARES=N` corrupts N elements of the reference on
purpose, so the compare path fires on known-good hardware.

This matters more here than anywhere else in the suite: this runner's entire
verdict is one counter, and a silent-data-corruption check that has never once
fired is an assertion nobody has tested. If the compare kernel were mis-launched
or the counter never read back, the runner would report `errors=0` for the rest
of its life and every node in the fleet would pass a test that was measuring
nothing.

A run with injection enabled is **not a hardware verdict**, and it says so twice
— `faults_injected` is emitted on every run (0 when the knob is unset, so its
absence from a stored envelope is itself informative), and an injected run also
carries a `config_warnings` entry. Never set it in a profile.

---

## Build

```sh
docker build -t ghcr.io/baldwinspc/glimmer-burnin-gpu-burn:vX.Y.Z runners/gpu-burn
```

Build args: `CUDA_IMAGE` (default `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04`)
and `CUDA_ARCH_FLAGS`. The default list ships real cubins for sm_80 / sm_90 /
**sm_100** / sm_120 / sm_121 plus a PTX fallback — deliberately wider than
compute-smoke's single `sm_121a`, because silent data corruption is a fleet-wide
concern and PTX JIT compiles FFMA one-to-one, so a JIT-compiled GEMM produces
the same bit patterns and cannot fake a miscompare count either way. The
`publish-runner` workflow's `CUDA_ARCH` input is deliberately not consumed here.

**Host architecture is a separate axis from GPU architecture.** The image is
published as a manifest list for `linux/amd64` **and** `linux/arm64`; the
gencode list above is the same on both, because an `sm_90` cubin is the same
device code whether the host is x86 or Grace. With CUDA's minor-version binary
compatibility (a cubin for CC X.Y runs on CC X.Z where Z ≥ Y) the list covers
A100/A10/L40S (`sm_80`), H100/H200/GH200 (`sm_90`), B200/B300/GB200/GB300
(`sm_100`), RTX PRO 6000 Blackwell (`sm_120`) and GB10 (`sm_121`). `sm_100` was
the one gap that mattered for x86 adopters: `compute_121` PTX cannot JIT **down**
onto CC 10.x, so before it a Blackwell datacentre part had no code in this image
at all — and silent data corruption is exactly what such a fleet wants this
runner to look for.

`soak_core.cuh` and `nvml_dynamic.h` are **byte-identical copies** of
[`../thermal-soak`](../thermal-soak)'s, because the publish workflow builds each
runner with its own directory as the Docker build context and `COPY` cannot reach
outside a build context. `TestSharedSoakSourcesAreIdentical` in
`../thermal-soak/soak_contract_test.go` fails if they drift. Edit one, copy it to
the other.

### Prerequisites for publishing

Both are outside this directory:

1. `gpu-burn` must be added to the `runner` choice list in
   `.github/workflows/publish-runner.yml`, or the image cannot be built by the
   workflow at all.
2. A `defaultRunnerImages` entry in `internal/controller/pods.go`, once a tag
   exists. Until then the kind deliberately has none, so it fails fast at plan
   time asking for an explicit `spec.runner.image`.

---

## Verified on hardware

Built and run on a DGX Spark (`spark-85a9`, aarch64, NVIDIA GB10, compute
capability 12.1, driver 580.82.09, Docker 28.3.3, native arm64 build) on
2026-08-03, via `docker run --gpus all`:

**Pass.** A 45 s burn: `GPU_BURN_PASS`, exit 0 — 299 iterations, `errors=0`,
`sdc_detections=0`, `nonfinite_count=0`, `ecc_errors=n/a`, `faults_injected=0`,
7.472 TFLOPS sustained, peak 82 °C, peak 87.44 W, `throttle_events=0`,
`throttle_reasons=none`, `ecc_mode=unsupported`.

**The detector was proven, not assumed.** A 30 s burn with
`BURNIN_SOAK_INJECT_MISCOMPARES=3` found exactly the injected damage and nothing
else:

```
faults_injected=3
first_miscompare_index=0
iterations=204
errors=612          # 3 corrupted elements x 204 iterations
sdc_detections=204  # every iteration saw the same corrupted region
nonfinite_count=0
config_warnings=BURNIN_SOAK_INJECT_MISCOMPARES=3 corrupted the reference on purpose; this run is a self-test of the detector and is NOT a verdict about this hardware
GPU_BURN_FAIL: 612 wrong element(s) across 204 incident(s) in 204 iterations at up to 81C — the part returned a different answer to identical arithmetic
EXIT=1
```

**What could not be exercised on healthy hardware**: a *real* miscompare, a real
non-finite result, and a real ECC event (GB10 has no ECC to report at all). The
compare path itself is proven by the injection above; the ECC branch that reports
a number rather than the sentinel has never run on this fleet, because no part in
it has visible ECC.

---

## Licensing

| component | licence | shipped? |
|---|---|---|
| `gpu_burn.cu`, `soak_core.cuh`, `nvml_dynamic.h` | Apache-2.0 (this project) | yes |
| upstream `wilicc/gpu-burn` source | BSD-2-Clause | **no** — not used; see above |
| cuBLAS / cuBLASLt | NVIDIA CUDA EULA | **no** — not linked, not shipped, no site mount needed |
| CUDA toolchain | NVIDIA CUDA EULA | **no** — build stage only, multi-stage |
| `libcuda.so.1`, `libnvidia-ml.so.1` | NVIDIA driver | **no** — injected at runtime by the NVIDIA Container Toolkit |

The build **fails** rather than shipping an image that drags in an NVIDIA `.so`.
`libcublas` is in the pattern on purpose: it is the one library a future edit of
this runner is most likely to reach for, and this line is what stops that edit
from quietly turning a redistributable image into one that is not.

```dockerfile
RUN ldd /out/gpu_burn | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libcublas|libnv' /out/ldd.txt
```

Verified: the shipped binary links `libc.so.6` and nothing else.

---

## Requirements on the node

- an NVIDIA accelerator, and the NVIDIA Container Toolkit
- `NVIDIA_DRIVER_CAPABILITIES` must include **`compute`** and **`utility`**. The
  image sets both. Without `utility` the runner cannot read the ECC state it must
  either report or declare unmeasurable, so it exits Error rather than pretending
  the hardware is fine.
- device memory for four `N × N` float buffers (1 GiB at the default `N=8192`),
  shrunk automatically if less is free
- runs as non-root (`65532:65532`); no privileges, no host mounts

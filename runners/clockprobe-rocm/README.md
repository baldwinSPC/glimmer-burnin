# clockprobe-rocm — sustained-clock probe for AMD accelerators

The AMD runner image for the **`clockprobe`** TestKind. It is not a kind of
its own: a profile selects it per node with `imagesByVendor`, so one profile
serves a mixed fleet and the verdict vocabulary stays identical across
vendors.

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: clockprobe
spec:
  kind: clockprobe
  scope: Node
  durationSeconds: 60
  runner:
    imagesByVendor:
      # No default is registered for this image yet: publication is
      # hardware-gated and it has not run on silicon. Pin it explicitly once
      # it is published.
      - vendor: amd
        image: <registry>/glimmer-burnin-clockprobe-rocm:<version>
  resources:
    limits:
      amd.com/gpu: 1
  thresholds:
    - metric: sustainedClockPct
      comparison: GreaterThanOrEqual
      value: "60"
```

## What it catches

The amdgpu **idle-clock lock** ([ROCm issue #5750](https://github.com/ROCm/ROCm/issues/5750),
reported on gfx1151 / Strix Halo): under a compute load the GPU stays pinned
at the **bottom** of its DPM ladder while `gpu_busy_percent` reads high. The
node looks busy, every health check passes, and the part runs at a fraction of
its speed — the same fault family as clockprobe's GB10 USB-C PD wedge, in its
AMD form. This is the platform's "reports healthy while running slow" failure
mode, and nothing else in the suite asserts speed.

## How it reads the hardware

**sysfs, not amdsmi.** On the APUs this variant exists for, AMD's management
library reports essentially every monitoring field as N/A while the kernel
exposes clocks, temperature, power and utilization through documented sysfs
the whole time ([ROCm issue #6035](https://github.com/ROCm/ROCm/issues/6035)).
The probe reads:

| File | Meaning |
|---|---|
| `pp_dpm_sclk` | the DPM clock ladder; the `*`-marked level is the current clock, the **top level is the rated clock** (the judgement denominator) |
| `gpu_busy_percent` | utilization — what makes "busy while slow" observable |
| `hwmon/*/temp1_input` | edge temperature, millidegree C |
| `hwmon/*/power1_average` (fallback `power1_input`) | power, microwatt |

There is no NVML-style nameplate boost on an APU; the ladder top is the
driver's own declared ceiling for the part as configured, which keeps the
denominator honest when a BIOS or power-mode change lowers the ladder.

The load is a register-resident FP32 FMA chain — clock-bound, no memory
traffic, exact FLOP count — launched continuously while the sampler reads
sysfs between launches.

## Judgement

Same rules as clockprobe, with the vocabulary sysfs supports:

- `sustainedClockPct` (mean post-warmup clock / ladder top) must clear
  `CLOCKPROBE_MIN_SUSTAINED_CLOCK_PCT` (default **60**).
- **Thermal evidence buys the lenient floor**: at or above
  `CLOCKPROBE_THERMAL_TEMP_C` (default **90** — gfx1151 parts run 87–91 °C
  edge under ordinary sustained load by design, so clockprobe's 80 would
  misclassify healthy parts) the floor drops to
  `CLOCKPROBE_MIN_THERMAL_CLOCK_PCT` (default **40**). An unreadable
  temperature never buys leniency.
- `idle_clock_lock_suspected` is **tri-state** (`true|false|unknown`): `true`
  only for slow + cool + busy (≥80% mean utilization); `unknown` whenever the
  temperature or utilization needed to establish the signature could not be
  read. Unknown never reads as all-clear.
- No readable `pp_dpm_sclk` ladder → **skip** (exit 2): a part with no rated
  clock is unjudged, not slow. An amdgpu device visible in sysfs that HIP
  cannot use → **error** (exit 3): hardware present, unjudged.

## Status — read before pinning

- **NOT VERIFIED ON HARDWARE.** No Strix Halo unit was available when this was
  written. The sysfs parsing and the judgement truth table are unit-tested
  (`sysfs_clocks_test.cc`, run by CI without hardware); the HIP load path is
  compile-verified only. The floor defaults are conservative choices, not
  measured numbers — a fleet declares its own thresholds, and nothing here
  should be tightened until the hardware pass produces measurements.
- **No published image, no registered default.** Publication is hardware-gated
  by this repository's policy, and `pkg/runnerimages` lists nothing before it
  is published.
- **linux/amd64 only** — the ROCm toolchain ships for amd64 alone, and so does
  every AMD accelerator host. The CI matrix excludes this runner's arm64 leg;
  `publish-runner.yml` still asserts two platforms and must learn this
  exception before a publish is attempted.
- **Runtime prerequisites**: `/dev/kfd` and `/dev/dri` from the device plugin
  (`amd.com/gpu: 1`). The image runs as uid 65532; hosts whose `/dev/kfd` is
  group-restricted may need `runAsUser: 0` or a supplemental group until the
  hardware pass settles the least privilege that works — record what is
  measured, then narrow.

## Build note: the distro shadow

Ubuntu noble/universe ships its own `hipcc` package (5.7.1-3, installed to
`/usr/bin`). On this image's first CI build every ROCm package resolved from
`repo.radeon.com` at 6.4.4 **except** `hipcc`, which apt took from
`archive.ubuntu.com`; the compile then failed with `/opt/rocm/bin/hipcc: not
found`. The Dockerfile now pins the `repo.radeon.com` origin at priority 1000
and asserts the compiler identifies as ROCm/AMD **before** compiling with it.

The pin matters beyond that one package: a missing path fails loudly, but a
distro-shadowed `hsa-rocr` or `comgr` would build fine and then misbehave
against a 6.4.4 runtime — the same shape as the container-toolkit version-skew
faults this project already tracks. If you re-pin `ROCM_VERSION`, keep both the
pin and the assertion.

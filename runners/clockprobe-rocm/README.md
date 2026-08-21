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

## Multi-device

This runner measures **every device the pod was allocated**, not device 0
(docs/dev/multi-device.md), mirroring clockprobe's NVIDIA engine — see its
README's fuller write-up. **The one piece with no NVIDIA precedent**: sysfs
has no HIP-provided device<->telemetry correlation, so `sysfs_clocks.h`'s
`FindAmdgpuCardForDevice` matches each HIP device's own reported PCI
domain/bus/device against each sysfs card's resolved PCI address — never
sysfs enumeration order, which has no documented relationship to HIP's device
ordering. **NOT VERIFIED AGAINST REAL MULTI-GPU AMD HARDWARE** for the same
reason as the rest of this runner: no second AMD device to prove the
correlation against.

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

## Build notes

**Build from AMD's devel image, not from apt.** The build stage is
`rocm/dev-ubuntu-24.04:6.4.4-complete`, matching how every NVIDIA runner here
builds against `nvcr.io/nvidia/cuda:*-devel`. The first two CI builds of this
image assembled the toolchain from apt instead, and both failed on package
archaeology rather than on anything about this runner:

1. Ubuntu noble/universe ships its **own** `hipcc` (5.7.1-3, `/usr/bin`), and
   apt preferred it while every other ROCm package came from
   `repo.radeon.com` at 6.4.4 — the compile died on `/opt/rocm/bin/hipcc: not
   found`.
2. With the origin pinned, the hand-picked package set still lacked what
   `amdgcn-link` needs, so device-code linking failed.

**The origin pin still matters in the runtime stage**, which does install from
apt: a distro-shadowed `hsa-rocr` or `comgr` would install quietly and then
misbehave against a 6.4.4-compiled binary — subtler than a missing path, and
the same shape as the container-toolkit version-skew faults this project
already tracks. If you re-pin `ROCM_VERSION`, keep the pin, and keep both
`ldd` assertions: one proves the built binary links the HIP runtime, the other
proves the runtime stage actually resolves it.

**Why 6.4.x and not 7.x**: 6.4.4 is the first release train listing gfx1151 as
supported, and the community has reported ROCm 7.x performance regressions on
this part. Revisit when the hardware pass can measure both.

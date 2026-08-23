# memory-bw-rocm — memory bandwidth for AMD accelerators

The AMD runner image for the **`memory-bw`** TestKind. It is not a kind of its
own: a profile selects it per node with `imagesByVendor`, so one profile serves
a mixed fleet and the metric names mean the same thing on every vendor. This is
the case the `imagesByVendor` field's own documentation uses as its example.

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: memory-bw
spec:
  kind: memory-bw
  scope: Node
  durationSeconds: 60          # the triad's sustained window
  runner:
    imagesByVendor:
      - vendor: nvidia
        image: ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.6.0
      # No default is registered for the AMD image yet: publication is
      # hardware-gated and it has not run on silicon.
      - vendor: amd
        image: <registry>/glimmer-burnin-memory-bw-rocm:<version>
  resources:
    limits:
      amd.com/gpu: 1
```

## What it measures

| Figure | Metric | What it is |
|---|---|---|
| host→device copy | `hostToDeviceBandwidthGBs` | worst of N passes (acceptance) |
| device→host copy | `deviceToHostBandwidthGBs` | worst of N passes |
| device-local copy | `deviceToDeviceBandwidthGBs` | worst of N passes |
| **STREAM triad** | **`memoryBandwidthGBs`** | **sustained** over the duration window |

Each copy path also reports a `*MaxGBs` companion (device 0's best pass) and a
`*_spread_pct` (device 0's own pass-to-pass spread). **Min is the acceptance
figure and max is evidence**, following the NVIDIA runner for the same reason:
a path is as good as its worst pass, and a mean hides the one slow iteration a
thermal or power-management fault produces. Their distance is what makes an
intermittent fault visible as a spread rather than as a slightly lower
average.

## On a unified-memory APU, three of these are intra-pool copies

gfx1151 has no discrete VRAM. "Host" and "device" memory are the same LPDDR5X,
so **h2d does not cross a PCIe link** and its figure is not comparable to a
discrete GPU's.

That is not a reason to withhold it — DGX Spark is unified too, and the NVIDIA
runner reports the same three figures there, so the numbers stay comparable
across a fleet mixing both. It *is* a reason to say plainly what the figure is,
so nobody reads a healthy intra-pool copy rate as a degraded link.
Set fleet thresholds from measured baselines per hardware class, never from a
datasheet PCIe number.

## Why this runner emits `memoryBandwidthGBs` when the NVIDIA one does not

That metric is registered as *"sustained device memory bandwidth achieved over
the whole measurement window, not a peak sample"*. The NVIDIA runner withholds
it deliberately, because nvbandwidth reports the median of a handful of short
samples and emitting it would let a profile author write a sustained-bandwidth
gate and have it evaluated against a burst.

This runner holds a STREAM triad for the duration budget it was given and
reports what it achieved over that window — which is exactly the registered
quantity. Emitting it from a burst would be the trap; emitting it from a soak is
the metric working as intended.

**On this hardware it is the number that matters most.** Token generation on an
APU is memory-bound, so sustained bandwidth against the ~256 GB/s the part is
specified for predicts whether inference will be slow — and no copy benchmark
reports it.

## A measurement that did not happen produces no number

`bw_stats.h` holds the arithmetic, free of HIP so it is unit-tested without a
GPU, and it enforces one rule: **never divide by an interval that did not
elapse.** A copy that appears to take zero time is not infinitely fast — it is
one whose duration fell under the clock's resolution, and `+inf` satisfies every
floor a profile could write. Such a pass is refused, counted in
`passes_below_timer_resolution`, and the figure is declared `n/a` if no pass
survived.

The same rule covers the peer cases (`peer_read_bandwidth_gbs`,
`peer_write_bandwidth_gbs`): an allocation of fewer than two devices has no
peer, so they are declared **`n/a` — not omitted, and not a skip of the whole
test**. Omitting the keys would make a gate on them fail closed and condemn
every single-GPU node; exiting 2 would throw away the figures the run did
take. Pair such a gate with `applicability: RequiredIfMeasurable` and it is
reported NOT EVALUATED. Two or more allocated devices leaves the peer keys
unreported rather than measured — this runner has no HIP peer-copy path, and
implementing one is hardware-gated work this conversion does not attempt
(#402).

The same "no number without a measurement" rule now also applies ACROSS
devices: `hostToDeviceBandwidthGBs`/`deviceToHostBandwidthGBs`/
`deviceToDeviceBandwidthGBs`/`memoryBandwidthGBs` are folded as the Min across
every device that measured that path (see Multi-device below); a path NO
device could time is still declared `n/a`, exactly the single-device
declaration this conversion must leave unchanged.

## Verdicts

Decided **per device**, then combined across the board
(`Fail > Error > Skip > Pass` — see Multi-device below).

**Pass (0)** — on that device, the copy paths ran, at least one figure was
measured, the triad completed at least one iteration, and every transfer's
data survived intact.

**Fail (1)** — **data corruption only, on that device.** Bytes that did not
survive a host→device→host round trip, or a triad result that does not match
the host reference, are a hardware verdict about that device. This is the sole
path to exit 1, and a corruption fault on one device does not stop another
device's own measurement.

**Error (3)** — any HIP failure on that device, or a device where every
interval fell under the timer's resolution. Hardware unjudged, retry
available.

**Skip (2)** — fires once, for the whole pod, only when **no accelerator is
visible to the container at all** (`device_fold.h`'s shared iteration
mechanism — see Multi-device). A device that IS visible but whose allocation
cannot be established, or that HIP cannot otherwise use, is an Error: memory
bandwidth applies to every accelerator this pod was actually handed, so a
usable device reporting nothing is a fault, not an inapplicable test.

## Thresholds

**Ships thresholdless**, pending fleet baselines — the same posture as the
NVIDIA runner. Absolute bandwidth is SKU-, kernel- and BIOS-specific here: the
GTT split and `ttm.pages_limit` change what the pool even is. Collect figures
across the fleet first, then set floors per hardware class in a `BurnInProfile`.

`device_memory_total_bytes` is reported as evidence for exactly that reason —
effective GPU-visible memory on an APU is a node-configuration output, not a SKU
constant.

## Multi-device

This runner measures **every device the pod was allocated**, not device 0
(docs/dev/multi-device.md), mirroring clockprobe-rocm's structure and its
shared `device_fold.h` header. Sequential iteration is the default — this kind
isolates each device's OWN copy and triad paths, the same reasoning clockprobe
applies to its clock probe — dividing the per-device triad window
(`deviceWindowS`) so the whole board's measurement still fits inside
`BURNIN_DURATION_SECONDS`; `BURNIN_DEVICE_CONCURRENCY=all` overrides it,
running every device's full pipeline (copy passes and triad) at once in its
own thread rather than dividing the window. `hostToDeviceBandwidthGBs`,
`deviceToHostBandwidthGBs`, `deviceToDeviceBandwidthGBs` and
`memoryBandwidthGBs` are the fold across devices (worst of each path);
PASS/FAIL/ERROR is decided per device from its OWN copies and triad and then
combined `Fail > Error > Skip > Pass` — a measured corruption fault on one
device is a fact about that device that another device's own clean result does
not erase, which is the same precedence memory-bw's NVIDIA runner already uses
across nvbandwidth's peer matrix, generalised here to real per-device
iteration. New keys: `deviceCount`, `devicesVisible`, `deviceWindowS`,
`deviceConcurrency`, `worstDeviceIndex`/`worstDevicePciBusId`, and
`hostToDeviceBandwidthSpreadPct` (n/a on a single-device node). Unlike the
NVIDIA memory-bw runner, this one DOES claim `deviceCount` — it folds devices
itself, and only a runner that does may claim the gate
(docs/dev/multi-device.md). When more than one device reported, a
`per-device.json` artifact carries every device's own reading.

**NOT VERIFIED AGAINST REAL MULTI-GPU AMD HARDWARE**, same caveat as the rest
of this runner: no second AMD device was available to prove the per-device
iteration, the fold, or the artifact against.

## Status — read before pinning

- **NOT VERIFIED ON HARDWARE.** The bandwidth arithmetic, aggregation and
  corruption checks are unit-tested without a GPU (`bw_stats_test.cc`); the HIP
  path is compile-verified only. Tracked in issue #320.
- **No published image, no registered default.**
- **linux/amd64 only** (issue #319).
- Covers gfx1151, gfx1100 and gfx942 — unlike compute-smoke-rocm this runner has
  no matrix-instruction dependency, so one image genuinely serves CDNA too.
- **Runtime prerequisites**: `/dev/kfd` and `/dev/dri` from the device plugin
  (`amd.com/gpu: 1`). Runs as uid 65532.

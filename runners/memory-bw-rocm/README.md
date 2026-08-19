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

Each copy path also reports a `*MaxGBs` companion (the best pass) and a
`*_spread_pct`. **Min is the acceptance figure and max is evidence**, following
the NVIDIA runner for the same reason: a path is as good as its worst pass, and
a mean hides the one slow iteration a thermal or power-management fault
produces. Their distance is what makes an intermittent fault visible as a
spread rather than as a slightly lower average.

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
`peer_write_bandwidth_gbs`): a single-accelerator node has no peer, so they are
declared **`n/a` — not omitted, and not a skip of the whole test**. Omitting the
keys would make a gate on them fail closed and condemn every single-GPU node;
exiting 2 would throw away the four figures the run did take. Pair such a gate
with `applicability: RequiredIfMeasurable` and it is reported NOT EVALUATED.

## Verdicts

**Pass (0)** — the copy paths ran, at least one figure was measured, the triad
completed at least one iteration, and every transfer's data survived intact.

**Fail (1)** — **data corruption only.** Bytes that did not survive a
host→device→host round trip, or a triad result that does not match the host
reference, are a hardware verdict. This is the sole path to exit 1.

**Error (3)** — any HIP failure, or a run where every interval fell under the
timer's resolution. Hardware unjudged, retry available.

**Skip (2)** — no path currently skips. A part with no HIP device at all is an
Error rather than a Skip: memory bandwidth applies to every accelerator, so an
absent device is a fault, not an inapplicable test.

## Thresholds

**Ships thresholdless**, pending fleet baselines — the same posture as the
NVIDIA runner. Absolute bandwidth is SKU-, kernel- and BIOS-specific here: the
GTT split and `ttm.pages_limit` change what the pool even is. Collect figures
across the fleet first, then set floors per hardware class in a `BurnInProfile`.

`device_memory_total_bytes` is reported as evidence for exactly that reason —
effective GPU-visible memory on an APU is a node-configuration output, not a SKU
constant.

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

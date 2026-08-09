# Intel

**No Intel-specific runner image is published or planned beyond a spike.** This
page is deliberately short, and honest about that.

## Coverage

| Kind | Measures | State |
|---|---|---|
| `memory-stress` | host DIMM stress | **shipped** (vendor-free) |
| `ib-write-bw` | RDMA write bandwidth over verbs | **shipped** (vendor-free) |
| `tcp-baseline` | plain TCP throughput and retransmits | in tree (vendor-free) |
| `disk-io` | storage throughput and latency | in tree (vendor-free) |
| `xpum-diag` | XPU Manager diagnostics | **spike only** — scoped, not started |
| everything else | — | no image, none planned |

## What works today

The four vendor-neutral kinds, unmodified. On an Intel accelerator node that
covers host memory, the fabric, the TCP path and storage. It does not cover the
accelerators themselves.

## The fingerprint knows Intel already

Worth knowing, because it is the part people assume is missing: Intel nodes are
fingerprinted correctly with no work. The vendor derivation takes the
**registrable** domain — the last two labels — so `gpu.intel.com/i915` and
`gpu.intel.com/xe` both yield `intel`. Taking the first label of the whole
domain would have produced a vendor called `gpu`, which resolves against
nothing; there is a test pinning that behaviour.

So `imagesByVendor` with an `intel` entry works today. What is missing is an
image to put in it.

## Running Intel accelerators in the meantime

Point a kind at your own image:

```yaml
spec:
  kind: custom
  runner:
    image: registry.example.com/our-xpum:v1
```

XPU Manager is MIT-licensed and emits JSON (`xpumcli diag -l 1|2|3 -j`), which
makes it a good candidate for a wrapper — that is what the `xpum-diag` spike
would be. **`hl_qual` (Habana) is proprietary** and cannot be redistributed
here; a site that has licensed it wraps it the same way.

## Thresholds

As everywhere: measured, never ported. See the
[overview](README.md#thresholds-do-not-port).

# AMD

**Six ROCm runners exist in this repository and none of their images is
published yet.** That is the honest summary: the source is real and building,
and until an image is pushed with a pinned tag nobody outside the org can pull
one (#531). This page says what works today and how to run AMD hardware now.

**For the measurement methodology, the tooling landscape and what a gfx1151 part
actually does under load, see
[amd-performance-testing.md](amd-performance-testing.md).** That page carries
the measured numbers; this one is the coverage map.

## Coverage

| Kind | Measures | State |
|---|---|---|
| `memory-stress` | host DIMM stress | **shipped** (vendor-free) |
| `ib-write-bw` | RDMA write bandwidth over verbs | **shipped** (vendor-free) |
| `tcp-baseline` | plain TCP throughput and retransmits | in tree (vendor-free) |
| `disk-io` | storage throughput and latency | in tree (vendor-free) |
| `host-health` | amdgpu RAS counters from sysfs | **in tree**, unverified on an Instinct (#259) |
| `clockprobe` | sustained clock vs the DPM ladder | **in tree** (`clockprobe-rocm`), hardware-verified on gfx1151 — and #534 is open against its verdict model |
| `compute-smoke` | HIP BF16/FP16 GEMM, gfx11 WMMA | **in tree** (`compute-smoke-rocm`); reports a CDNA part as Error (#331) |
| `memory-bw` | device bandwidth | **in tree** (`memory-bw-rocm`) |
| `nccl` | collective bandwidth | **in tree** (`nccl-rocm`); multi-node RoCE is #345 |
| `gpu-burn`, `thermal-soak` | compute and thermal soak | **in tree** (`-rocm`) |
| `rvs-diag` | ROCm Validation Suite wrapper | undecided — hardware-gated investigation, #318 |
| `gpudirect-rdma`, `dcgm-diag` | — | not applicable; NVIDIA-specific |

**None of the `-rocm` images is pullable.** A profile that needs one must build
and push it, or name its own. See #531.

## What works today

The four vendor-neutral kinds run on AMD nodes **now**, unmodified. That covers
host memory, the RDMA fabric, the TCP path and storage — and a node failing any
of those is worth pulling from service regardless of what its accelerators are.

`ib-write-bw` is the one worth emphasising: it measures the wire through verbs
and never opens a vendor context, so an AMD node's fabric is as measurable today
as an NVIDIA node's.

## Running AMD accelerators in the meantime

Point a kind at your own image. The runner contract is four exit codes and
`key=value` lines on stdout; anything obeying it works, and nothing about this
operator is NVIDIA-specific.

```yaml
spec:
  kind: memory-bw
  runner:
    imagesByVendor:
      - vendor: nvidia
        image: ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.7.1
      - vendor: amd
        image: registry.example.com/our-transferbench:v1
  resources:
    limits:
      amd.com/gpu: 1
```

**AGFHC is proprietary** and cannot be redistributed here. A site that has
licensed it wraps it in an image and points a kind at it — the same pattern
`dcgm-diag` uses for site-supplied DCGM.

The upstreams the planned images will be built from are all permissively
licensed, and a site can use them directly today:

| Tool | Licence | Would become |
|---|---|---|
| ROCm Validation Suite (RVS) | MIT | `rvs-diag` |
| rccl-tests | BSD-3-Clause | the `nccl` AMD image |
| TransferBench | MIT | the `memory-bw` AMD image |

## Node requirements

- A device plugin advertising `amd.com/gpu`. The fingerprint derives the vendor
  from that resource's DNS domain even on a node with no labels at all.
- `/dev/kfd` and `/dev/dri` reachable by the pod. Declare them through
  `spec.runner.hostPaths` — that is the only way a runner pod reaches the node's
  filesystem, and a test that declares nothing gets no volumes.
- **Group access to `/dev/kfd`, which is not automatic.** Measured on a Strix
  Halo unit: `root:render`, mode 0660, `other::---`, gid 992. The runner images
  run as uid 65532 and **cannot open it**. The least-privilege fix is a
  supplemental group for the node's own `render` gid, which `RunnerSpec` cannot
  yet express (#535) — so today the expressible option is `runAsUser: 0`.
  `privileged: true` is not a substitute.

## One profile, two very different AMD parts

A Strix Halo APU and an Instinct MI300X are both `amd`, and almost nothing about
them is shared: WMMA against MFMA, unified GTT memory against HBM3, no RAS block
against ECC with row remapping, and a ~120 W package shared with sixteen CPU
cores against a 750 W GPU-only board.

`spec.runner.imagesByVendor` therefore takes an optional **`arch`**, matching the
node fingerprint's `status.gpus[].arch` exactly:

```yaml
runner:
  imagesByVendor:
    - {vendor: amd, arch: gfx1151, image: …-compute-smoke-rocm-gfx1151:v1}
    - {vendor: amd, arch: gfx942,  image: …-compute-smoke-rocm-cdna:v1}
    - {vendor: amd,                image: …-compute-smoke-rocm:v1}   # fallback
```

Most specific wins, regardless of the order the entries are written in. A vendor
listed *only* with architectures refuses a node whose arch is absent rather than
falling through to a default the author did not choose.

Worked example: [`config/samples/profile-amd-halo.yaml`](../../config/samples/profile-amd-halo.yaml).

## Thresholds

Do not port NVIDIA numbers. A bandwidth floor that is right for one vendor's
part is meaningless for the other's, and the failure mode is a healthy node
condemned by a threshold nobody measured. Run thresholdless, look at the fleet,
then gate.

Counters are the exception and are safe from day one: `ioErrors`,
`tcpRetransmits` and `eccErrors` are zero on a healthy part regardless of who
made it.

**One APU-specific trap, measured rather than predicted.** On a Strix Halo the
GPU and the CPU cores share one socket power budget, so all-core CPU load drops
the graphics clock from 100% of its rated ceiling to 41% — while the part stays
busy and gets *cooler*. A `sustainedClockPct` gate will fail healthy nodes
whenever anything else is running on the box, and the thermal-leniency escape
hatch does not help because the part is cool. **Sequence CPU-side and GPU-side
tests; never overlap them.** See #534 and
[amd-performance-testing.md](amd-performance-testing.md).

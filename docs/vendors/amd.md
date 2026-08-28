# AMD

**No ROCm runner image is published by this project yet.** This page says what
works today, what is planned, and how to run AMD hardware in the meantime — which
is more than nothing, and considerably more than the coverage table alone
suggests.

## Coverage

| Kind | Measures | State |
|---|---|---|
| `memory-stress` | host DIMM stress | **shipped** (vendor-free) |
| `ib-write-bw` | RDMA write bandwidth over verbs | **shipped** (vendor-free) |
| `tcp-baseline` | plain TCP throughput and retransmits | in tree (vendor-free) |
| `disk-io` | storage throughput and latency | in tree (vendor-free) |
| `memory-bw` | device bandwidth | planned — TransferBench image |
| `nccl` | collective bandwidth | planned — rccl-tests image |
| `host-health` | amdgpu RAS counters | planned |
| `rvs-diag` | ROCm Validation Suite wrapper | planned |
| `gpu-burn`, `thermal-soak`, `clockprobe` | compute and thermal soak | later — RVS `gst`/`iet` wrappers |
| `compute-smoke`, `gpudirect-rdma`, `dcgm-diag` | — | not applicable; NVIDIA-specific |

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

## Thresholds

Do not port NVIDIA numbers. A bandwidth floor that is right for one vendor's
part is meaningless for the other's, and the failure mode is a healthy node
condemned by a threshold nobody measured. Run thresholdless, look at the fleet,
then gate.

Counters are the exception and are safe from day one: `ioErrors`,
`tcpRetransmits` and `eccErrors` are zero on a healthy part regardless of who
made it.

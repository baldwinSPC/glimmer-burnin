# `gpudirect-rdma` runner — RDMA straight into accelerator memory

Runner image for the `gpudirect-rdma` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
`ib_write_bw --use_cuda` between **two nodes**, so the HCA DMAs to and from
device memory without bouncing through host RAM.

[`ib-write-bw`](../ib-write-bw) measures the wire with a host buffer. This
measures the same link with a **device** buffer. The difference between the two
numbers is the cost of the bounce, and on a node where GPUDirect is
*misconfigured* rather than *absent* it is large.

Pair scope only. A point-to-point measurement is a property of a link.

## The headline: on GB10 this runner exits 2, by design

**On NVIDIA GB10 (DGX Spark) this test is not applicable and the runner says so
positively, with exit 2.** Verified on real hardware — see *Verified on* below.

GPUDirect RDMA needs a **peer-memory provider** registered with the RDMA
subsystem, so that an HCA can be handed a mapping of accelerator memory. On GB10
there is none and there cannot be one: CPU and GPU share a single physical
LPDDR5x pool over NVLink-C2C, so there is no separate device memory for a
provider to publish. NVIDIA has confirmed the absence is architectural rather
than a packaging gap.

That is a **Skip**, never a Fail. A node that cannot do GPUDirect RDMA has not
failed a hardware test; it is not a candidate for one.

### Why the decision lives here and not in the reconciler

The controller used to carry an `allTargetsGB10` vendor branch that skipped this
test on Sparks. It was deleted, and this runner owns the decision instead. The
branch was wrong twice over:

- it put a vendor's part number in a **vendor-neutral** controller, which is the
  thing `CLAUDE.md` says belongs in runner images; and
- it made the answer a property of the node's **labels** rather than of the
  node's **drivers** — so a Spark that somehow did have a provider would still
  have been skipped, and a differently-labelled part without one would still have
  been attempted.

Only a process on the node can look at the node's drivers. So it looks.

### What is actually checked

`applicability.go` reports every path it consulted, so a skip is auditable from
the pod log alone. On a real Spark:

```
applicability: /proc/driver/nvidia/gpus: 1 GPU(s) — 000f:01:00.0 (model Unknown)
applicability: /sys/kernel/mm/memory_peers: absent (no provider has ever registered)
applicability: nvidia_peermem: not loaded
```

`/sys/kernel/mm/memory_peers/<name>` is the kernel's **generic** registration
point — `nvidia_peermem` uses it, and so would any other vendor's provider — so
the check is not NVIDIA-specific. The NVIDIA module is looked for by name only so
the message can say which particular thing is missing. Nothing here asks the GPU
what model it is; a check keyed off "GB10" would go stale the moment a driver
release changed the answer for that part.

(`model Unknown` is the driver's own answer on GB10, not a parse failure. It is
printed verbatim so nobody goes looking for a bug in this runner.)

### The one case this check gets wrong, and why that direction is chosen

A stack that supports GPUDirect **only through dma-buf** (`ibv_reg_dmabuf_mr`,
perftest's `--use_cuda_dmabuf`) registers no peer-memory provider, and would be
reported here as inapplicable. That is a known limitation.

It is the safe direction. The cost of this mistake is a test that did not run,
which is visible in the run's results; the cost of the opposite mistake would be
a healthy link recorded as broken, which is not. If your fleet is dma-buf-only,
raise it — the fix is to widen the check, not to make the failure a Fail.

## Output contract

| Result | stdout | Exit |
|--------|--------|------|
| Pass | `GPUDIRECT_RDMA_PASS: <detail>` | 0 |
| Fail | `GPUDIRECT_RDMA_FAIL: <reason>` | 1 |
| Not applicable to this node | `GPUDIRECT_RDMA_SKIP: <reason>` | 2 |
| Unjudged (runner or configuration fault) | `GPUDIRECT_RDMA_ERROR: <reason>` | 3 |

| Runner key | Canonical metric | Meaning |
|---|---|---|
| `bw_average` | `bandwidthGbps` | average RDMA write throughput into device memory |
| `t_avg_usec` | `latencyUs` | `ib_write_lat --use_cuda` `t_avg` |

`bandwidthGbps` is deliberately the **same** canonical name `ib-write-bw`
reports: same measurand, same link. Minting a second name would split a link's
history in two the day a fleet changed which test it gates on. The two are told
apart by the `TestKind`, which the delivery envelope carries.

`peakBandwidthGbps` is never emitted, for the same reason as in `ib-write-bw`:
duration mode does not measure a peak, and perftest prints `0.00` there.

`--report_gbits` is forced and the table header is asserted — see
[`../ib-write-bw/README.md`](../ib-write-bw/README.md#--report_gbits-is-not-optional)
for the 8x bug that guard exists to prevent.

## How a Skip settles the pair

The applicability gate runs **before** the rendezvous, deliberately: a node that
cannot do this test must not hold its peer's node hostage waiting for a handshake
it will refuse anyway.

So on a GB10 pair the server pod exits 2 before the client is ever created.
[`internal/controller/pair.go`](../../internal/controller/pair.go) settles that
correctly: with no client result, a server verdict of Pass would become an Error
(no traffic crossed the link), but a server **Skip** propagates as a Skip. The
pair reports one result naming both nodes, and neither node is blamed.

## Pod requirements

Everything [`ib-write-bw`](../ib-write-bw/README.md#pod-requirements) needs —
`hostNetwork`, the `/dev/infiniband` mount, unlimited `memlock`, non-root —
**plus** a GPU:

```yaml
spec:
  hostNetwork: true
  resources:
    limits:
      nvidia.com/gpu: 1
  runner:
    readinessProbe:
      tcpSocket: { port: 18520 }
    hostPaths:
      - path: /dev/infiniband
        mountPath: /dev/infiniband
        readOnly: false
        type: Directory
```

The `hostPaths` entry is not optional and `privileged` will not stand in for it:
a privileged `hostNetwork` pod on these nodes was measured to be missing every
`uverbs*` device, which is what `ibv_create_cq` opens. See
[`../ib-write-bw/README.md`](../ib-write-bw/README.md#pod-requirements) for the
observed pod-versus-host listing.

The control/readiness port defaults to **18520** (perftest uses 18525/18526), so
this runner and `ib-write-bw` can be scheduled without colliding under
`hostNetwork`. As there, **do not** probe perftest's own port: it listens with a
backlog of one and would accept the probe as its peer.

## Tuning

| Variable | Default | Meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 60 | total budget; the bandwidth phase gets this minus 20s |
| `BURNIN_HEALTH_PORT` | 18520 | control/readiness port |
| `BURNIN_PERFTEST_PORT` | 18525 | perftest's bandwidth port; latency uses +1 |
| `BURNIN_MESSAGE_BYTES` | 1048576 | message size |
| `BURNIN_QPS` | 4 | queue pairs |
| `BURNIN_CUDA_DEVICE` | 1 | perftest's own 1-based CUDA device number |
| `BURNIN_LATENCY_ITERATIONS` | 5000 | `ib_write_lat` iterations |

## Licences

perftest is dual-licensed **GPL-2.0-only OR BSD-2-Clause** and is consumed here
**under the BSD-2-Clause option only**, identically to
[`ib-write-bw`](../ib-write-bw/README.md#licences) — including the same pin at
`v4.4-0.37` and the same `assert-no-copyleft.sh` guard against the pciutils
(GPL-2.0-or-later) requirement introduced in the v4.5 series. Read that section
before bumping anything here.

| Component | Licence |
|---|---|
| this runner's wrapper (`*.go`) | Apache-2.0 |
| linux-rdma/perftest | BSD-2-Clause (option taken) |
| rdma-core — `libibverbs`, `librdmacm`, `ibverbs-providers` | dual GPL-2.0 OR BSD, taken under the BSD option |
| `libnl-3`, `libnl-route-3` | LGPL-2.1-or-later, dynamically linked |
| Go standard library | BSD-3-Clause |
| glibc (base image) | LGPL-2.1-or-later, dynamically linked |

### The NVIDIA rule, and the distinction this image turns on

No **NVIDIA-licensed redistributable library** ships here. `libcuda.so.1` is not
one: it is the **driver**, it belongs to the host, and the NVIDIA Container
Toolkit injects it at runtime — the same arrangement
[`runners/compute-smoke`](../compute-smoke) already relies on. So this image
**links** against `libcuda.so.1` while **containing** no NVIDIA library at all.

That is why the build has two different guards rather than one:

- an `ldd` check on the binary, which forbids `libcudart` (a redistributable that
  nothing injects) and *requires* `libcuda.so` (proof the CUDA path was
  compiled in); and
- a `find` over the finished filesystem, which forbids any NVIDIA `.so` from
  actually being present.

`ldd` alone cannot tell "links a library the host will provide" from "ships one",
and for this runner those two questions have different answers.

The build also asserts that the compiled `ib_write_bw` really has `--use_cuda`. A
perftest that silently configured without CUDA would run, measure **host**
memory, and report it under a metric name claiming device memory — a wrong answer
that looks exactly like a right one.

## Pinned versions

| Thing | Pin |
|---|---|
| perftest | `v4.4-0.37` (`5fb4f10…`) |
| CUDA toolchain (build only) | `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04` |
| runtime base | `ubuntu:24.04` |

CUDA **13** cannot be used to build this perftest: CUDA 13 changed
`cuCtxCreate` to the four-argument `cuCtxCreate_v4`, and `v4.4-0.37`'s call no
longer compiles. 12.9 is the newest toolchain that both builds perftest and
supports `sm_121`. (perftest uses only the CUDA driver API and launches no
kernel, so no gencode is involved and a binary built against 12.9 headers runs
fine on the 580.x driver.)

## Verified on

Two DGX Sparks (NVIDIA GB10, aarch64, driver 580.82.09) with ConnectX-7 over RoCE
v2. **The exit-2 path was exercised on real silicon** — the accelerator is
present, the fabric is up, and no peer-memory provider exists — which is the
outcome this runner is expected to produce on that hardware.

The **measuring** path (exit 0, with real bandwidth) has **not** been exercised,
because no hardware with a peer-memory provider was available. The parsing and
argument construction it depends on are shared with `ib-write-bw` and are covered
by unit tests and by that runner's live run, but the `--use_cuda` execution path
itself is unproven.

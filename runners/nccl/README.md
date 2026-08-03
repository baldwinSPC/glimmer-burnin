# `nccl` runner — a two-rank all-reduce across a link

Runner image for the `nccl` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
an NCCL `AllReduce` between **two nodes**, one rank each, over the fabric.

[`ib-write-bw`](../ib-write-bw) measures the wire. This measures what a
**collective** achieves over that wire — NCCL's transport selection, its protocol
and algorithm choices, and the copies it needs on a node without GPUDirect RDMA.
A link can be perfect and the collective still slow because NCCL fell back to a
socket transport; that is exactly the failure this test exists to catch.

Pair scope. See *Node scope* below for what happens on a one-GPU part.

## Output contract

| Result | stdout | Exit |
|--------|--------|------|
| Pass | `NCCL_PASS: <detail>` | 0 |
| Fail | `NCCL_FAIL: <reason>` | 1 |
| Not applicable to this node | `NCCL_SKIP: <reason>` | 2 |
| Unjudged (runner or configuration fault) | `NCCL_ERROR: <reason>` | 3 |

| Runner key | Canonical metric | Meaning |
|---|---|---|
| `busbw` | `busBandwidthGBs` | **peak** bus bandwidth across the sweep |
| `algbw` | `algBandwidthGBs` | algorithm bandwidth **at the same message size** as the peak |
| `latency_us` | `latencyUs` | per-collective time at the **smallest** message size |
| `wrong_count` | `miscompares` | elements whose all-reduce result was wrong, summed over the sweep |

Metrics are printed **before** the decision line.

## Exactly one value per metric name — and why that matters here

`pkg/runner`'s parser is **last-occurrence-wins**. A runner that swept message
sizes and printed `busbw=` once per size would have only its **last** size
recorded — silently, with a perfectly plausible-looking number, and whether that
number was the best or worst of the sweep would depend on the order the sizes
happened to be listed in. With this runner's ascending sweep the last size is the
largest, but with an 8-byte sample anywhere at the end the recorded bus bandwidth
would be `0.0002 GB/s` and every healthy link would fail its gate.

Two ways out were available:

- **(a)** emit one metric per size under distinct names — `busBandwidth1MiBGBs`,
  `busBandwidth256MiBGBs`, … ; or
- **(b)** sweep for evidence, but emit exactly **one** value per metric name.

**This runner does (b).** (a) makes the metric name carry a benchmark parameter:
a profile author gating a fleet would have to know which sizes this image happens
to sweep, every threshold would break the day the sweep changed, and two runner
versions would report a link's bandwidth under two different names — precisely
the drift `pkg/contract`'s registry exists to prevent.

So the full sweep goes to the **log** as evidence, and one figure is promoted to
each metric:

- `busBandwidthGBs` / `algBandwidthGBs` — the **peak**, and both from the **same
  sample**, so the pair is coherent. The peak is the right summary because the
  small sizes in the sweep are latency-bound by construction; averaging them in
  produces a number that is not the link's bandwidth and that moves whenever the
  sweep changes.
- `latencyUs` — the **smallest** message, where the per-collective time is a
  latency figure rather than a bandwidth one.

`results_test.go` asserts this against the real parser, including the negative
case that the last row must *not* win.

### `busBandwidthGBs == algBandwidthGBs` at two ranks, and that is correct

Bus bandwidth is algorithm bandwidth scaled by `2*(n-1)/n`, the factor
nccl-tests uses for an all-reduce (a ring moves each byte around twice, once
reducing and once broadcasting). **At n=2 that factor is exactly 1.**

They are still reported as two metrics, because they are two different quantities
that merely coincide at this rank count — they diverge the moment a Group-scope
test uses more ranks, and collapsing them now would make a Pair's history
incomparable with a Group's. The wrapper re-derives the factor independently of
the harness and logs a warning if the two disagree, so a units or formula change
in either file becomes visible instead of silently rescaling a fleet metric.

## Thresholds: ship none, and here is the arithmetic

**This runner ships no default threshold, and the obvious one is impossible.**

A 2-rank all-reduce cannot exceed the link. On the DGX Spark pair this was
developed against, `ib_write_bw` plateaus at **99.61 Gb/s** because the
ConnectX-7 attaches at PCIe Gen5 x4 — the wire is 200G and the host can only feed
half of it. 99.61 Gb/s is **12.45 GB/s**, so:

> `busBandwidthGBs >= 20` is **arithmetically unreachable** on this hardware.
> 20 GB/s is 160 Gb/s. Such a threshold would not be strict; it would condemn
> every node in the fleet for a property of the part.

Set the gate from the measurement your fleet actually produces. **Measured on
that pair** (rank 0 `spark-85a9`, rank 1 `spark-043a`, RoCE v2, two runs):

| Message | time | alg = bus bandwidth |
|---|---|---|
| 8 B | 29–44 µs | ~0.0002 GB/s |
| 1 MiB | 184–225 µs | 4.65–5.69 GB/s |
| 8 MiB | 759–772 µs | 10.87–11.05 GB/s |
| 64 MiB | 5.91–5.98 ms | 11.21–11.35 GB/s |
| 256 MiB | 22.36–22.37 ms | **12.00–12.01 GB/s** |

`busBandwidthGBs = 12.0 GB/s` is 96 Gb/s — **96% of the 99.61 Gb/s that
`ib_write_bw` measures on the same link**, which is what a healthy collective
over a saturated link looks like. So a defensible gate for this fleet is roughly
90% of the measured figure:

```yaml
thresholds:
  - metric: busBandwidthGBs
    operator: GreaterThanOrEqual
    value: "10.8"        # ~90% of the measured 12.0
  - metric: miscompares
    operator: Equal
    value: "0"
```

`miscompares == 0` is the threshold worth having unconditionally: it is the one
that catches a link returning wrong data rather than slow data.

## Node scope exits 2 on a one-GPU part

An all-reduce needs at least two ranks. At Node scope both would have to be GPUs
in the same box — where this would be a free NVLink test and well worth running.
**Each DGX Spark has one GPU**, so there is no second rank and the test is
inapplicable, not failed: exit 2, with the GPU count in the message.

Intra-node (multi-GPU, single-node) collectives are **not implemented** by this
runner even where the GPUs exist; it is a Pair-scope fabric test. A node with two
or more accelerators also exits 2, saying so.

## Why this is not `nccl-tests`

NVIDIA's [nccl-tests](https://github.com/NVIDIA/nccl-tests) is the obvious tool
and it is BSD-3-Clause, so licensing is not the reason. **The launcher is.**

nccl-tests spans more than one process only when built with MPI, and must then be
started by `mpirun`, which needs a launcher able to start a process on the *other*
node — ssh, or a resource manager. A Kubernetes Pair is two independently
scheduled pods with no such launcher between them, and adding one would mean
shipping an **sshd in a burn-in image and giving it a key**: a standing
remote-execution service on every accelerator node in the fleet, in order to run
a bandwidth test.

The operator already provides the rendezvous that is missing. So
[`nccl_pair.cu`](nccl_pair.cu) takes its rank from the wrapper, exchanges NCCL's
bootstrap handle over one TCP connection, and runs the same collective with the
**same bandwidth formulas** nccl-tests uses.

**What is lost:** nccl-tests' breadth of collectives (this does `AllReduce` only)
and its multi-node topology reporting. **What is gained:** two pods are enough.

### The bootstrap handle is exchanged inside the harness, not by the wrapper

`ncclGetUniqueId()` does not merely mint a token: it opens a listening socket **in
the calling process** and records that address inside the ID. The process that
generates the ID must therefore be the same process that later joins the
communicator as rank 0. A wrapper that generated the ID in a separate invocation
and passed it in would hand over the address of a socket that had already closed,
and the symptom would be a hang inside `ncclCommInitRank` that looks exactly like
a dead fabric.

## NCCL is pinned to the fabric, not left to choose

Left to itself NCCL enumerates every interface it can see and picks by its own
ranking. On a burn-in node that set includes the **wireless** interface, a CNI
overlay and `docker0` — and NCCL choosing one produces a number that is real,
reproducible, and about the wrong link entirely.

So the wrapper sets, from the **same route-based discovery `ib-write-bw` uses**
(so the two tests measure the same physical path and their numbers are
comparable):

| Variable | Set to |
|---|---|
| `NCCL_IB_HCA` | `<device>:<port>` of the HCA that routes to the peer |
| `NCCL_IB_GID_INDEX` | the RoCE v2 GID whose IPv4 address is the local route address |
| `NCCL_SOCKET_IFNAME` | the netdev that HCA fronts, so NCCL's own bootstrap ring does not go over Wi-Fi |

These are appended to the environment the operator already populated from
`spec.runner.env`, so an operator's explicit setting wins.

Note that port numbering is not symmetric across a cable — on the reference pair
the *same* link is `roceP2p1s0f1` at one end and `roceP2p1s0f0` at the other — so
each side computes its own device. See
[`../ib-write-bw/README.md`](../ib-write-bw/README.md#how-the-two-ends-find-each-other).

## Pod requirements

```yaml
spec:
  hostNetwork: true
  resources:
    limits:
      nvidia.com/gpu: 1
  runner:
    readinessProbe:
      tcpSocket: { port: 18530 }
```

Also needs `/dev/infiniband` and an unlimited `memlock`. Runs **non-root** (uid
65532). The control/readiness port is **18530** and rank 0's NCCL bootstrap
handle is published on **18535**; both differ from the other two fabric runners'
ports so they do not collide under `hostNetwork`.

The readiness probe targets the wrapper's own control port, which the client
genuinely uses — the server binds it, then starts rank 0, then releases the
client only once `/proc/net/tcp` shows the bootstrap socket is actually
**LISTENING**. "The process started" is not the same claim as "the port is
bound", and the operator's gate is only as good as what it gates on.

## Tuning

| Variable | Default | Meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 60 | bounds the rendezvous wait (min 2 min) |
| `BURNIN_HEALTH_PORT` | 18530 | control/readiness port |
| `BURNIN_BOOTSTRAP_PORT` | 18535 | where rank 0 publishes the `ncclUniqueId` |
| `BURNIN_NCCL_ITERS` | 20 | timed iterations per size |
| `BURNIN_NCCL_WARMUP_ITERS` | 5 | warmup iterations per size |
| `BURNIN_NCCL_SIZES` | `8,1048576,8388608,67108864,268435456` | the sweep, in bytes |

NCCL's own variables (`NCCL_DEBUG`, `NCCL_ALGO`, …) pass through
`spec.runner.env` untouched. `NCCL_DEBUG=INFO` is the first thing to reach for
when a collective is slow; the harness's stderr is echoed into the pod log.

## Correctness is checked, at every size

Every rank contributes `1.0`, so a correct `SUM` all-reduce leaves `nranks` in
every element. That makes the check a real one rather than a tautology: a
collective that silently dropped a rank's contribution leaves `1.0` and is
caught, and one that corrupted data produces neither value.

It runs at **every** message size, not once, because NCCL changes algorithm and
protocol with message size and a fault that only appears above a switchover
threshold would be invisible to a single-size check.

A non-zero `miscompares` is the one condition that makes this runner exit **1**:
the hardware ran the collective and returned wrong data, which is a hardware
verdict. If the run never reached the check, `wrong_count` is **not emitted** —
a miscompare count that was never established must not be reported as `0`, and it
is not `n/a` either, since that sentinel is for hardware which *cannot* produce
the measurement.

## Licences

### NCCL is BSD-3-Clause — this is worth stating plainly

NCCL is often assumed to be an NVIDIA-licensed redistributable of the kind this
project may not ship. **It is not.**
[NVIDIA/nccl](https://github.com/NVIDIA/nccl) is **BSD-3-Clause**, the same
permissive licence as CUTLASS. The build asserts it by reading the `LICENSE.txt`
of the exact commit it fetched, so the claim is a build-time fact rather than a
line in a README nobody re-reads when the dependency moves.

**NOTICE-ready text:**

```
NVIDIA NCCL
  Copyright (c) 2015-2024, NVIDIA CORPORATION. All rights reserved.
  Copyright (c) 2015, Lawrence Berkeley National Laboratory.
  Licensed under BSD-3-Clause.
  Built from source and statically linked into runners/nccl's nccl_pair binary.
  Version: v2.27.7-1 (593de54e52679b51428571c13271e2ea9f91b1b1)
```

| Component | Licence |
|---|---|
| this runner's wrapper (`*.go`) and harness (`nccl_pair.cu`) | Apache-2.0 |
| NVIDIA NCCL | BSD-3-Clause, built from source, statically linked |
| CUDA runtime (`libcudart`, statically linked via `-cudart static`) | NVIDIA CUDA EULA, which permits redistribution of the runtime linked into an application — the same arrangement [`runners/compute-smoke`](../compute-smoke) already ships |
| rdma-core — `libibverbs`, `librdmacm`, `ibverbs-providers` | dual GPL-2.0 OR BSD, taken under the BSD option |
| `libnl-3`, `libnl-route-3`, glibc | LGPL-2.1-or-later, dynamically linked |
| Go standard library | BSD-3-Clause |

`ibverbs-providers` is not optional: **NCCL `dlopen()`s `libibverbs`** for its
IB/RoCE transport, and without the provider plugins the devices are invisible to
it and NCCL falls back to a TCP socket transport — a result that is real,
reproducible, and about the wrong path.

No NVIDIA **shared** object ships: NCCL and the CUDA runtime are linked
statically, and the build asserts with a `find` over the finished filesystem that
none is present. `libcuda.so.1` is the driver and is injected at runtime by the
NVIDIA Container Toolkit.

### Why NCCL is built from source rather than installed

The prebuilt `libnccl` from NVIDIA's apt repository is **411 MB**, because it
embeds device code for every architecture NVIDIA ships. A runner image is
executed by a node's readiness gate, so its pull cost is paid by every node on
every gate. Building from source with one gencode is a fraction of that, and
static linking then leaves no NVIDIA `.so` at all — which lets the shipping guard
be an unconditional assertion instead of an exception list.

## Pinned versions

| Thing | Pin | Note |
|---|---|---|
| NCCL | `v2.27.7-1` (`593de54…`) | licence asserted at build time |
| gencode | `-gencode=arch=compute_121,code=sm_121` | **GB10 only** |
| CUDA toolchain (build only) | `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04` | |
| runtime base | `ubuntu:24.04` | |

**The arch target is deliberate.** `sm_121` is GB10, the hardware this runner was
verified on. Each extra architecture multiplies NCCL's device-code build time and
every node's image pull, so a fleet on different silicon should **rebuild** with
its own gencode (`--build-arg NCCL_GENCODE=...`) rather than have this default to
"everything". Guessing on a fleet's behalf is the thing `CLAUDE.md` warns against.

The harness itself is built `-arch=sm_121` without the `a` suffix, unlike
`compute-smoke`: it launches no kernel of its own — every kernel it runs is
NCCL's — so an `a` target would constrain the binary without proving anything
about the instruction path.

## Verified on

Two DGX Sparks (NVIDIA GB10, aarch64, driver 580.82.09), ConnectX-7 at PCIe Gen5
x4, RoCE v2, MTU 4096. Rank 0 on `spark-85a9`, rank 1 on `spark-043a`. The
measured figures are in the run recorded with this runner's introduction; use
them, not a spec sheet, to set the gate.

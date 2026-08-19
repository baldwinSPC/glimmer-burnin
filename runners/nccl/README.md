# `nccl` runner — a collective, over a link or within one box

Runner image for the `nccl` [`TestKind`](../../api/v1alpha1/burnintest_types.go).
At **Pair** and **Group** scope it is an NCCL `AllReduce` between nodes, one rank
each, over the fabric. At **Node** scope — a box with two or more GPUs — it is
one process, every device the pod was allocated, joined via `ncclCommInitAll`
into a collective that never leaves the node: no bootstrap handle to exchange,
because there is no second pod.

[`ib-write-bw`](../ib-write-bw) measures the wire. This measures what a
**collective** achieves over it (or over NVLink/PCIe P2P/shared memory, at Node
scope) — NCCL's transport selection, its protocol and algorithm choices, and the
copies it needs on a node without GPUDirect RDMA. A link can be perfect and the
collective still slow because NCCL fell back to a socket transport; that is
exactly the failure this test exists to catch, at every scope.

**Why Node scope exists.** DGX BasePOD validation runs the 8-GPU intra-node
all-reduce *before* any multi-node test, and it has no Pair or Group form — there
is no fabric between two GPUs in one box for a Pair test to measure.

## Group scope (N ranks)

One rank per target node. Rank 0 mints the `ncclUniqueId` and serves it to the
other N-1 over TCP; every rank then joins the same communicator. The operator
supplies the whole rendezvous:

| Variable | Meaning |
|---|---|
| `BURNIN_RANK` | this pod's 0-based rank |
| `BURNIN_NRANKS` | how many ranks the collective has |
| `BURNIN_ROOT_HOST` | rank 0's DNS name, `rank-0.<service>.<ns>.svc` |
| `BURNIN_ROOT_NODE` | rank 0's node name (messages only) |

`BURNIN_ROLE` is **absent** at Group scope, and this runner does not look for it
there. That is deliberate on both sides: a runner keying off `server`/`client`
must fail loudly rather than treat rank 4 of eleven as a client.

**Only rank 0 reports numbers.** The others exit 0 having taken part. The
operator merges a group's metrics with rank 0 winning any shared key, and
`pkg/runner` is last-occurrence-wins — so N ranks printing the same keys would
record whichever pod happened to be harvested last, which is a number nobody
chose.

`busBandwidthGBs` and `algBandwidthGBs` **diverge** above two ranks. The factor
is `2(n-1)/n`, which is exactly 1 at n=2 and 1.33 at n=3 — so a Pair run's two
figures coincide and a Group run's do not. That is arithmetic, not a fault.

> **NOT VERIFIED ON HARDWARE.** Every line of the N-rank path compiles and its
> argument construction, contract parsing and host resolution are unit-tested,
> but **no 3-rank collective has ever run through it** — this fleet has two GPU
> nodes. The acceptance is a real ≥3-node run whose reported bus bandwidth is
> checked by hand against the algorithm bandwidth and the rank count, once. See
> issue #118.

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

## Node scope: intra-node, `ncclCommInitAll`

`BURNIN_ROLE` unset with no `BURNIN_RANK`/`BURNIN_NRANKS` either means Node
scope. What decides whether this pod may run a collective is a **positive** fact
about the hardware — how many devices the driver reports — never the mere
absence of those variables: reading an absence as permission is the exact
failure the Group-scope `groupCapableKinds` guard exists to catch (#118), and
this is its Node-scope mirror.

- **Fewer than two GPUs — `DGX Spark` among them, which has one — exits 2.** An
  all-reduce needs at least two ranks; there is no second one in the box, so the
  test is inapplicable, not failed. The GPU count is in the message.
- **Two or more GPUs runs the collective.** `nccl_pair`'s `--local-multi-gpu`
  mode enumerates every visible device, joins them with `ncclCommInitAll`, and
  runs the collective named by the `collective` variant axis
  (`BURNIN_VARIANT_COLLECTIVE`) — `allreduce` by default, or `allgather`,
  `reducescatter`, `alltoall`. An unrecognised value is refused as a config
  error rather than silently measured as `allreduce`.

**`busBandwidthGBs`/`algBandwidthGBs` keep their names at Node scope**, for
every collective, deliberately — see the field's own doc comment in
`pkg/contract/metrics.go` for the full reasoning: the *same* `busBandwidth`
formula computes the figure regardless of scope or collective, `Scope` on the
stored result already tells a Node reading from a Pair one apart, and a
threshold is written per `BurnInTest`, never globally by metric name — so
nothing ever compares a Node-scope number against a Pair-scope gate by
accident. `AllGather`/`ReduceScatter`/`AllToAll` scale by `(n-1)/n` rather than
`AllReduce`'s `2(n-1)/n` — see `collective/collective.h`.

**Two additional metrics, both Evidence, never gates:**

| Runner key | Canonical metric | Meaning |
|---|---|---|
| `ranks` | `ranks` | how many devices actually joined — the same count that decided whether to run at all |
| `nccl_transport` | `ncclTransport` | best-effort: `nvlink`\|`p2p`\|`shm`\|`net`, read from `NCCL_DEBUG=INFO`'s own log text. **Absent, never wrong**, if NCCL's wording ever moves — this is the one place this runner reads NCCL's diagnostic output at all, and it can never become a hardware verdict |

**`RLIMIT_MEMLOCK` is checked but not enforced at Node scope.** An intra-node
collective over NVLink, PCIe P2P or shared memory registers no pinned buffers
the way NCCL's `net` transport does, so a small limit does not stop it here —
unlike Pair and Group, where it is fatal before the rendezvous even starts. It
only matters if NCCL falls back to `net` transport *within* the box, and if that
happens the harness's own failure is still recognised and gets the same advice.

> **NOT VERIFIED ON HARDWARE.** This project has no multi-GPU node. The
> `>=2`-device decision and every collective's arithmetic are unit-tested in a
> CUDA-free header (`collective/collective_test.cc`); the `<2`-device skip is
> verified on a real DGX Spark. `AllGather`, `ReduceScatter` and `AllToAll` in
> particular have never run on real silicon. See issue #406 for exactly what a
> real run must capture before any of this is published.

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

## RLIMIT_MEMLOCK — this runner needs the cluster's help, and says so

NCCL's IB/RoCE transport registers **pinned** memory, capped by
`RLIMIT_MEMLOCK`. A containerd-started pod inherits containerd's own
`LimitMEMLOCK`, **8 MiB** by systemd default, and NCCL cannot work within it:

```
NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory
```

**Unlike [`ib-write-bw`](../ib-write-bw), this runner cannot size itself to
fit.** That runner owns every byte it registers, so it shrinks its plan and still
measures the link. NCCL does not expose the same lever — **measured**, at 8 MiB
*all* of these still fail in `ibv_reg_mr_iova2`:

| setting | result |
|---|---|
| default | FAIL |
| `NCCL_BUFFSIZE` = 2 MiB / 1 MiB / 512 KiB | FAIL |
| `NCCL_BUFFSIZE` + `NCCL_MAX_NCHANNELS=2` | FAIL |
| an **8-byte** all-reduce, `NCCL_BUFFSIZE=256 KiB` | FAIL |

An eight-byte collective failing rules out the user buffers and the channel
buffers alike: what does not fit is NCCL's own internal registration. The only
alternative to refusing would be `NCCL_IB_DISABLE=1` — a silent fallback to a TCP
socket transport, which is the exact failure this test exists to catch, so it
must never be how this test passes.

So the runner **checks the limit up front** (before the rendezvous, so it does
not hold its peer's node hostage) and exits 3 with both remedies. **Error, not
Fail**: the link was never measured, so there is no hardware verdict, and Error
is the retryable phase — right for an environment somebody is about to fix.

### The two remedies, and the trap between them

What actually matters is **`CAP_IPC_LOCK`, which bypasses `RLIMIT_MEMLOCK`
entirely** — the identical NCCL run passes as uid 0 and fails as uid 65532 at the
very same 8 MiB limit. Measured on this cluster:

1. **Raise the container runtime's limit.** For containerd under systemd:
   ```
   # /etc/systemd/system/containerd.service.d/10-memlock.conf
   [Service]
   LimitMEMLOCK=infinity
   ```
   then `systemctl daemon-reload && systemctl restart containerd`. This is what
   every RDMA/NCCL guide means by `--ulimit memlock=-1`. **This is what the
   reference cluster now does.**
2. **Run as uid 0 in a privileged pod**, so `CAP_IPC_LOCK` applies.

**The trap:** `securityContext.capabilities.add: [IPC_LOCK]` does **not** work
for a non-root container — Kubernetes has no ambient-capabilities field, so runc
clears the permitted set when it drops to the UID. Verified: `CapPrm` and
`CapEff` are both `0000000000000000` in a privileged pod running as 65532, with
and without `IPC_LOCK` added.

The required floor is **64 MiB**, and it is a *conservative* floor rather than a
measured minimum — the 8 MiB default is measured to fail, but NCCL's true
requirement could not be measured, because the only ways to give a container more
(CAP_IPC_LOCK, or raising the runtime limit) remove the ceiling rather than
raising it to a number. The Error always reports the **observed** limit and both
remedies, so it stays actionable regardless.

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
    hostPaths:
      - path: /dev/infiniband
        mountPath: /dev/infiniband
        readOnly: false
        type: Directory
```

**The `/dev/infiniband` mount is mandatory.** `privileged: true` plus
`hostNetwork: true` does *not* supply the `uverbs*` user-verbs nodes — measured,
see [`../ib-write-bw/README.md`](../ib-write-bw/README.md#pod-requirements) — and
neither does an `rdma/…` resource request. Without them NCCL's `net_ib` plugin
fails to create its resources exactly as `ib_write_bw` does, and the failure
mode here is the worse one: a plugin that cannot initialise falls back to TCP and
still reports a bus bandwidth, so the run yields a plausible number for a
transport nobody was qualifying. The mount is applied to both ranks of the pair.

Also needs an unlimited `memlock`. Runs **non-root** (uid 65532). The control/readiness port is **18530** and rank 0's NCCL bootstrap
handle is published on **18535**; both differ from the other two fabric runners'
ports so they do not collide under `hostNetwork`.

The readiness probe targets the wrapper's own control port, which the client
genuinely uses — the server binds it, then starts rank 0, then releases the
client only once `/proc/net/tcp` shows the bootstrap socket is actually
**LISTENING**. "The process started" is not the same claim as "the port is
bound", and the operator's gate is only as good as what it gates on.

**At Node scope, none of the above applies.** There is one pod, no rendezvous,
no `hostNetwork`, no `/dev/infiniband` mount and no readiness probe to wait
for a peer that does not exist. `spec.resources.limits` requests the SKU's
whole board (`nvidia.com/gpu: 8`, not `1`) — see
[docs/dev/multi-device.md](../../docs/dev/multi-device.md) for why the pod asks
for the count rather than `all`.

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
| gencode | *per host platform*, see below | one architecture, deliberately |
| CUDA toolchain (build only) | `nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04` | |
| runtime base | `ubuntu:24.04` | |

## Host architecture, GPU architecture, and the one build arg you may need

The image is published as a manifest list for **`linux/amd64` and
`linux/arm64`** — that is the CPU axis, and it says nothing about which GPU the
image can talk to. `NCCL_GENCODE` is the GPU axis, and it is where this runner
asks something of you.

**The arch target is deliberate and it is a single architecture.** Each extra
architecture multiplies NCCL's device-code build time and every node's image
pull; the prebuilt `libnccl` that carries them all is 411 MB, paid by every node
on every readiness gate. So the default is one part per host architecture,
chosen as the part that architecture is actually deployed with:

| host | `NCCL_GENCODE` default | harness `CUDA_ARCH` | parts |
|---|---|---|---|
| `linux/arm64` | `-gencode=arch=compute_121,code=sm_121` | `sm_121` | GB10 / DGX Spark — the hardware this runner was verified on |
| `linux/amd64` | `-gencode=arch=compute_90,code=sm_90` | `sm_90` | H100, H200 — the dominant x86 multi-node NCCL parts |

**This is the runner where "usable on x86" means "usable with a gencode that
matches your GPU".** A single `code=sm_XX` target carries no PTX, so an x86 fleet
on B200, L40S or A100 gets an image with no kernels for its part and must
rebuild:

```sh
docker build \
  --build-arg NCCL_GENCODE=-gencode=arch=compute_100,code=sm_100 \
  --build-arg CUDA_ARCH=sm_100 \
  runners/nccl        # B200 / GB200
```

Override **both** or neither: a harness compiled for a part NCCL has no kernels
for is an image that cannot run anywhere. Guessing "everything" on a fleet's
behalf is the alternative, and it costs every node in every fleet a 400 MB pull
to avoid one build argument.

The harness is built without the `a` suffix, unlike `compute-smoke`: it launches
no kernel of its own — every kernel it runs is NCCL's — so an `a` target would
constrain the binary without proving anything about the instruction path.

## Verified on

Two DGX Sparks (NVIDIA GB10, aarch64, driver 580.82.09), ConnectX-7 at PCIe Gen5
x4, RoCE v2, MTU 4096. Rank 0 on `spark-85a9`, rank 1 on `spark-043a`. The
measured figures are in the run recorded with this runner's introduction; use
them, not a spec sheet, to set the gate.

**Node scope: only the `<2`-device skip**, on `spark-043a` (one GB10). The
`>=2`-device path — every collective, `ncclCommInitAll`, the buffer sizing and
correctness checks — has never run on real silicon; this project has no
multi-GPU node. See issue #406.

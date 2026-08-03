# `ib-write-bw` runner — RDMA write bandwidth and latency across a link

Runner image for the `ib-write-bw` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
it measures RDMA write bandwidth and latency **between two nodes**, using
[`ib_write_bw`](https://github.com/linux-rdma/perftest) and `ib_write_lat` from
linux-rdma/perftest.

This is a **Pair-scope** runner and only a Pair-scope runner. There is no such
thing as the RDMA write bandwidth of a single node: the quantity is a property
of a link, and the verdict names both endpoints. Run at Node scope it exits 2.

## Output contract

| Result | stdout | Exit |
|--------|--------|------|
| Pass | `IB_WRITE_BW_PASS: <detail>` | 0 |
| Fail | `IB_WRITE_BW_FAIL: <reason>` | 1 |
| Not applicable to this node | `IB_WRITE_BW_SKIP: <reason>` | 2 |
| Unjudged (runner or configuration fault) | `IB_WRITE_BW_ERROR: <reason>` | 3 |

Metrics are `key=value` lines on stdout and are **always printed before** the
decision line, so a verdict never arrives without the numbers behind it.

| Runner key | Canonical metric | Meaning |
|---|---|---|
| `bw_average` | `bandwidthGbps` | average RDMA write throughput over the whole duration |
| `latency_us` | `latencyUs` | `ib_write_lat` `t_avg`, a 2-byte write round trip |

### `--report_gbits` is not optional

Without it, perftest prints **MB/sec into the same column**. The value would land
in a metric named `bandwidthGbps` eight times too small: a 100 Gb/s link would
report `12.45` and fail a `>= 89` gate, and the reverse mistake would pass a link
that should not. The flag is set in `bwArgs`, **and** `perftest.go` asserts the
unit in the table header, so the flag going missing is caught even if someone
edits the argument list.

### `peakBandwidthGbps` is deliberately never emitted

perftest only measures a peak in **iteration** mode. This runner uses **duration**
mode (`-D`), because that is what honours `BURNIN_DURATION_SECONDS`, and in
duration mode perftest prints `0.00` in the peak column and warns that it was not
measured.

Emitting that zero would put a number nobody measured into an acceptance metric.
It is not `n/a` either: the `n/a` sentinel declares that *the hardware* cannot
produce a measurement, and this is a property of the mode this runner chose. So
the key is simply absent, and **a threshold on `peakBandwidthGbps` will fail
closed** — which is correct. Do not write one.

## What the runner decides, and what the profile decides

The acceptance number belongs in the `BurnInTest`'s thresholds, not in the image:
a 200G fabric and a 100G fabric are both healthy, and a runner with a hard-coded
floor would condemn one of them. A completed measurement therefore exits 0.

Exit 1 is reserved for two states that are hardware facts on their own terms:

- **no ACTIVE port on any RDMA device** — the node has fabric hardware and no
  usable link;
- **the test completed carrying no traffic** (`bandwidthGbps` of 0).

Exit 2 (skip) means the test does not apply: no RDMA device in
`/sys/class/infiniband` at all, or no `BURNIN_ROLE` (a Node-scope run).

Everything else — a peer that never appeared, a perftest that would not start, a
table in the wrong unit — is exit 3, **Error**, not Fail. On a Pair that
distinction is load-bearing: `Error` outranks `Fail` in
[`internal/controller/pair.go`](../../internal/controller/pair.go) precisely so a
machinery fault on one end cannot permanently indict a link, and `Fail` is the
one phase the operator never retries.

## Suggested thresholds — and the number that is wrong

**Measured on two DGX Sparks (GB10) over RoCE v2, 4 QPs, 1 MiB messages:**

```
bandwidthGbps  99.61
latencyUs       1.71
```

The Ethernet link negotiates 200000 Mb/s, but the measurement plateaus just under
100 Gb/s and is **identical at 4 and 8 queue pairs**, which is what makes it a
real ceiling rather than a queue-depth artefact. The cause is the host, not the
fabric: the ConnectX-7 attaches at **PCIe Gen5 x4**, where `current_link_*`
equals `max_link_*` — nothing is degraded — and Gen5 x4 is ~15.75 GB/s raw,
about 100 Gb/s after overhead. The wire is 200G; the host can feed half of it.
On DGX Spark this is architectural.

So for that fleet:

```yaml
thresholds:
  - metric: bandwidthGbps
    operator: GreaterThanOrEqual
    value: "89"          # ~90% of the measured 99.61
```

**A threshold of 180 is wrong by about 2x** and would fail every healthy Spark
pair. It comes from reading the link's negotiated rate off a spec sheet rather
than measuring the host behind it. Derive the gate from a measurement on the
hardware you have.

## How the two ends find each other

The operator supplies the whole rendezvous contract — `BURNIN_ROLE`,
`BURNIN_PEER_HOST`, `BURNIN_PEER_NODE` — and starts the client only once the
server pod is Ready. On top of that the runner opens **one TCP control
connection** between the two halves (`rendezvous.go`). It exists for two reasons
that the operator's contract deliberately does not cover:

1. **Which local RDMA device to use.** Port numbering is not symmetric across a
   cable. On the two Sparks this was developed against, the *same* link is
   `roceP2p1s0f1` at one end and `roceP2p1s0f0` at the other. A hard-coded device
   measures the right link from one node and a dead port from the other, and
   reports the difference as a fabric fault. Each side therefore picks the device
   that **routes to its peer** — and the server cannot resolve its peer's DNS
   name before the client exists, so it learns the peer's address from the
   accepted connection itself.
2. **Agreement on parameters.** perftest requires matching message size, queue
   pairs and mode at both ends. Mismatch it and the run dies with
   `Failed to complete run_iter_bw function successfully`, which reads exactly
   like a broken fabric. The server sends the parameters, so a mismatch is
   unreachable rather than merely unlikely.

The GID index is derived the same way: on RoCE there is no safe default (indices
0 and 1 are the link-local IPv6 GIDs), so the runner picks the **RoCE v2 GID
whose embedded IPv4 address is the local address the route chose**. On the test
fleet that resolved to index 11, not the index 3 a manual invocation would have
used — which is exactly why it is derived rather than configured.

### The readiness probe, and the port you must *not* probe

Declare a probe on the runner's **control port** (default `18510`):

```yaml
runner:
  readinessProbe:
    tcpSocket:
      port: 18510
    periodSeconds: 2
```

The server binds it before starting perftest, and the client actually uses it, so
a successful probe is a true statement that this end is ready to be
rendezvous'd with.

**Do not point a probe at perftest's own port (18515).** perftest listens with a
**backlog of one** and treats the next connection as its client: a kubelet probe
would be accepted as the peer, fed nothing, and break the test it was meant to
guard. Probe connections that reach the control port are read with a short
deadline and dropped, which is why the client identifies itself with a hello.

## RLIMIT_MEMLOCK — the limit that decides whether this runs at all

Every RDMA buffer is **pinned** memory, capped by `RLIMIT_MEMLOCK`. Exceed it and
`ibv_create_cq` fails with `ENOMEM`, which perftest reports as:

```
Couldn't create CQ
Failed to create CQs
 Couldn't create IB resources
```

That message names neither the limit nor the remedy and reads exactly like a
broken HCA.

**This runner needs nothing from the cluster: it sizes itself to fit.** It reads
the limit, budgets half of it (the data buffers are not the only pinned pages),
and reduces the plan if needed — surrendering **message size before queue
pairs**, because that order is measured, not preferred:

| change | cost |
|---|---|
| 4 queue pairs → 1 | 99.6 → 97.5 Gb/s |
| 1 MiB → 64 KiB at 4 QPs | 99.61 → 99.40 Gb/s |

Queue pairs are what saturate the link; message size mostly buys headroom above
saturation. **Verified in a pod at the 8 MiB container default:**

```
RLIMIT_MEMLOCK = 8.0 MiB; this runner will register at most 4.0 MiB of RDMA buffers
REDUCED TO FIT RLIMIT_MEMLOCK: message size 1048576 -> 524288 bytes
plan: 524288-byte messages across 4 queue pairs = 4.0 MiB registered (budget 4.0 MiB)
IB_WRITE_BW_PASS: measured 99.59 Gb/s ... with 4 QPs at 524288-byte messages
```

99.59 vs 99.61 Gb/s unconstrained — the gate at 89 is unaffected. Both ends
negotiate: the client reports its limit in the hello and the **server plans for
the smaller of the two**, because the two pods can be given different limits and
a plan that fits only the side that chose it fails on the other.

If the plan cannot be fitted even at the floor, the runner reports an **Error**
naming `RLIMIT_MEMLOCK` rather than measuring something meaningless — a 4 KiB
single-QP "bandwidth" says nothing about a link but would fail a correctly-set
threshold as if the hardware were at fault.

### Why the limit differs between your laptop and the cluster

The host allows ~15 GB; a containerd-started pod inherits containerd's own
`LimitMEMLOCK`, which is **8 MiB** by systemd default. The same binary with the
same arguments therefore gets 8 MiB in a pod and 15 GB on the node.

**`docker run --ulimit memlock=-1` hides this completely.** That flag is what
every NVIDIA container guide tells you to pass, and it is what this runner was
first verified with — which is precisely why the failure only appeared once it
ran as a pod. Verify fabric runners **as pods**.

Note also that neither `privileged: true` nor
`securityContext.capabilities.add: [IPC_LOCK]` raises the limit for a
**non-root** container: Kubernetes has no ambient-capabilities field, so the
capability sets are empty even when privileged (`CapEff: 0000000000000000`), and
the process cannot raise its own hard limit. Sizing the work to fit is the only
portable answer, which is what this runner does.

## Pod requirements

```yaml
spec:
  hostNetwork: true          # the fabric addresses live on the host's NICs
  runner:
    readinessProbe:
      tcpSocket: { port: 18510 }
    hostPaths:
      - path: /dev/infiniband
        mountPath: /dev/infiniband
        readOnly: false      # ibv_open_device opens the verbs nodes read-write
        type: Directory
```

**The `/dev/infiniband` mount is mandatory, and `privileged` is not a substitute
for it.** Measured on this project's reference nodes, inside a pod that had both
`privileged: true` and `hostNetwork: true`:

```
pod:  /dev/infiniband -> rdma_cm umad0 umad1 umad2 umad3
host: /dev/infiniband -> by-ibdev by-path rdma_cm umad0-3 uverbs0-3
```

Every `uverbs*` node — the user-verbs devices `ibv_create_cq` opens — was
missing. Without them this runner dies at `Couldn't create CQ / Failed to create
CQs / Couldn't create IB resources`, which reads like a broken link and is
nothing of the kind. Adding the `hostPaths` entry above makes `uverbs0-3` appear
in the pod; that was verified directly. The mount is applied to **both** pods of
the pair, which is what it has to be: a link measured from one end is not
measured.

The pod also needs an unlimited `memlock`. On this project's reference nodes
`/dev/infiniband/uverbs*` is mode `0666`, so the container runs as **non-root**
(uid 65532) with no added capability — RDMA verbs need neither `CAP_SYS_ADMIN`
nor privileged mode, and a privileged fabric pod on every node in a fleet is a
far larger grant than this test needs. Mounting one device directory is the
smaller grant, and it is the one that actually works.

`hostNetwork` is required rather than merely convenient: without it the pod's
address is a CNI overlay address, the route to the peer does not land on an RDMA
netdev, and device selection falls back to "the first ACTIVE port" — which on a
crossed topology is the wrong one.

## Tuning

| Variable | Default | Meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 60 | total budget; the bandwidth phase gets this minus 20s |
| `BURNIN_HEALTH_PORT` | 18510 | the control/readiness port |
| `BURNIN_PERFTEST_PORT` | 18515 | perftest's port for the bandwidth phase; latency uses +1 |
| `BURNIN_MESSAGE_BYTES` | 1048576 | message size |
| `BURNIN_QPS` | 4 | queue pairs |
| `BURNIN_LATENCY_ITERATIONS` | 5000 | `ib_write_lat` iterations |

## Licences

### perftest is dual-licensed, and this project takes the BSD option

[linux-rdma/perftest](https://github.com/linux-rdma/perftest) is offered under
**GPL-2.0-only OR BSD-2-Clause**. glimmer-burnin consumes and redistributes it
**under the BSD-2-Clause option only** — under the GPL option this image could
not be published at all. The build asserts that the source it fetched still
offers the BSD grant, and copies the licence plus an explicit statement of the
choice into `/licenses` inside the image.

**NOTICE-ready text** (this is the entry for issue #3):

```
linux-rdma/perftest
  Copyright (c) 2005-2023 Mellanox Technologies Ltd. and contributors.
  Dual-licensed: GPL-2.0-only OR BSD-2-Clause (the "OpenIB.org BSD license").
  glimmer-burnin consumes and redistributes perftest under the BSD-2-Clause
  option ONLY. No part of perftest is taken under the GPL option.
  Used by: runners/ib-write-bw, runners/gpudirect-rdma.
  Version: v4.4-0.37 (5fb4f10a7e7827ed15e53c25810a10be279d6e23)
```

### A copyleft finding worth acting on

**From the v4.5 series onward, perftest's `configure` hard-requires pciutils
(`libpci`, GPL-2.0-or-later) on every non-FreeBSD platform, and there is no
`--without-pciutils`:**

```
if [test $IS_FREEBSD = no]; then
    AC_CHECK_HEADERS([pci/pci.h],,[AC_MSG_ERROR([pciutils header files not found...])])
    AC_CHECK_LIB([pci], [pci_init], [LIBPCI=-lpci], AC_MSG_ERROR([libpci not found]))
fi
```

A GPL library may not ship in an image this project publishes, so **v4.5+ is not
usable here as-is**. The pinned `v4.4-0.37` predates the requirement and
references `libpci` nowhere, and `assert-no-copyleft.sh` fails the build if a
future bump changes that. Do not resolve such a failure by installing
`pciutils-dev`.

The dependency is narrow enough to be worth raising upstream: the only code that
uses `libpci` is a PCI-bus scan for **one** Intel root-complex erratum (vendor
`0x8086`, devices `0x2f01` and `0x6f01`–`0x6f0e`), which cannot match on any
aarch64 node this runner targets. A `--without-pciutils` would remove the
blocker. If one is ever needed locally, patch out the *scan* while keeping
`HAVE_RO`: dropping `HAVE_RO` is the easier patch and the wrong one, because it
also drops `IBV_ACCESS_RELAXED_ORDERING` from the memory region and so changes
the number being measured.

### Everything else in the image

| Component | Licence |
|---|---|
| this runner's wrapper (`*.go`) | Apache-2.0 |
| linux-rdma/perftest | BSD-2-Clause (option taken; see above) |
| rdma-core — `libibverbs`, `librdmacm`, `ibverbs-providers` | dual GPL-2.0 OR BSD-2-Clause / OpenIB BSD, taken under the BSD option |
| `libnl-3`, `libnl-route-3` | LGPL-2.1-or-later, dynamically linked and unmodified |
| Go standard library | BSD-3-Clause |
| glibc (base image) | LGPL-2.1-or-later, dynamically linked |

**No NVIDIA library is present**, and the build asserts it. This runner never
touches the accelerator, which is what lets it run on a node whose GPU is busy or
absent. The GPUDirect variant — perftest built with `--use_cuda` — is a separate
image, [`runners/gpudirect-rdma`](../gpudirect-rdma).

### Why the runtime base is `debian-slim` and not distroless

`libibverbs` does not link its hardware providers; it `dlopen()`s them, guided by
the `.driver` files in `/etc/libibverbs.d` and found on a path compiled into the
library. Hand-copying those into a distroless base ties the runner to two
absolute paths the distribution owns, and the failure mode when one moves is
**"no RDMA device found"** — which this runner would report as a clean SKIP on
perfectly healthy hardware, silently excusing a whole fleet from its fabric test.
A fabric runner must not be able to fail that way.

## Pinned versions

| Thing | Pin | Why this one |
|---|---|---|
| perftest | `v4.4-0.37` (`5fb4f10…`) | see the copyleft finding above; **also** the v4.5 series does not compile against stock rdma-core (50.0 on Ubuntu 24.04, 56.1 on Debian trixie) — it needs the AES-XTS crypto mkey API from `mlx5dv` that only MLNX_OFED/DOCA ships. Both constraints were verified by building every v4.5 tag on both distributions. |
| base | `debian:trixie-slim` | build and runtime are the same release, so the glibc perftest was linked against is the one it runs on |

## Verified on

Two DGX Sparks (NVIDIA GB10, aarch64, driver 580.82.09), ConnectX-7 at PCIe Gen5
x4, RoCE v2 over a 200G Ethernet link, MTU 4096. Server on `spark-85a9`
(`roceP2p1s0f1`), client on `spark-043a` (`roceP2p1s0f0`) — the crossed device
pair was resolved automatically by route.

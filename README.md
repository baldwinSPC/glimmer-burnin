# Glimmer Burn-In Operator

A vendor-neutral, Kubernetes-native **hardware acceptance-testing** controller for
GPU/accelerator fleets. Define a burn-in profile once; run it against a node — or a
**pair/group of nodes** for interconnect validation — and get a structured pass/fail
verdict you can gate provisioning on.

> **Standalone by design.** This operator does **not** import any control plane. It
> exports verdicts through a generic `BurnInSink` (webhook / ConfigMap / Prometheus),
> so it integrates with [Glimmer](https://github.com/baldwinSPC/glimmer) — or your own
> system — with zero code dependency. CI enforces the no-import rule.

## Scope: what runs today

`BurnInTest.spec.scope` is `Node`, `Pair` or `Group`. **`Node` and `Pair` are
executed; `Group` is not.** A `Group` test is recorded as **`Error`** —
explicitly *not run*, hardware *not judged* — rather than skipped. That is the
fail-closed rule at work: a required acceptance test the operator cannot run
must never let hardware pass by omission.

Interconnect acceptance is the design's reason for existing. Single-node burn-in
(gpu-burn, DCGM, thermal soak) is table stakes; the value most fleets are missing
is **link acceptance** — NCCL bus bandwidth over RoCE/IB, `ib_write_bw`,
GPUDirect RDMA — which is only meaningful **across at least two nodes**.

**Pair scope** runs a server pod on the first target node and a client pod on the
second, each pinned to its node, rendezvous'd through a headless `Service`. The
client is not started until the server pod is **Ready** — a client that connects
a moment early dies with a connection error, and a connection error on a fabric
test reads as a bad link. Runners receive `BURNIN_ROLE=server|client`,
`BURNIN_PEER_HOST` (the peer's DNS name) and `BURNIN_PEER_NODE`.

A pair produces **one** `TestResult` naming **both** nodes, because a
point-to-point measurement is a property of the link and cannot be attributed to
either endpoint. Two consequences worth knowing before you run one:

- **`maxConcurrentNodes` must be at least 2.** A pair holds both of its nodes for
  the whole test, so it costs two of the interlock's slots. At the default of 1
  the run is refused at start with that explanation rather than hanging.
- **Exactly two distinct target nodes are required**, and a run whose target
  resolves to anything else is refused at start naming the count it got.

`Group` (N≥2 collectives) needs gang scheduling and rank assignment and is not
implemented.

## Custom Resources

| Kind | Purpose |
|------|---------|
| `BurnInTest` | One reusable test, with `scope: Node\|Pair\|Group`, thresholds, and a pluggable runner image. |
| `BurnInProfile` | An ordered suite of tests + verdict policy (e.g. `acceptance`, `pair-network`). |
| `BurnInRun` | One execution of a profile against a target; status carries per-test metrics + the overall verdict. |
| `BurnInSchedule` | Runs a profile against a target on a cron schedule. Acceptance at install time only proves the hardware was good on the day it arrived. |
| `NodeFingerprint` | The captured hardware/network identity (GPUs, NICs/RDMA) a verdict is bound to — and the drift detector between runs. |
| `BurnInSink` | Where results are exported. The **only** integration seam. |

## Test kinds

**All eleven kinds ship a runner image, and every one is published and public**
under `ghcr.io/baldwinspc/glimmer-burnin-<kind>:v0.2.0`. A test that names no
`spec.runner.image` gets the built-in default for its kind; `spec.runner.image`
overrides it, which is the seam new hardware arrives through.

| Kind | Scope | What it gates | Built on |
|------|-------|---------------|----------|
| `compute-smoke` | Node | Arch-correct FP4 block-scaled GEMM against a host reference; `nonfiniteCount` | NVIDIA CUTLASS (BSD-3, header-only) |
| `clockprobe` | Node | Sustained clock under load — catches a part pinned in a low P-state that every health check calls healthy | ours |
| `thermal-soak` | Node | Clock and temperature over a long duration, sampled throughout | ours |
| `gpu-burn` | Node | Correctness under sustained load: same arithmetic, same answer, for hours | ours (see below) |
| `memory-bw` | Node | Host-to-device, device-to-host and on-device copy bandwidth | NVIDIA nvbandwidth (Apache-2.0) |
| `memory-stress` | Node | Host DIMM stress | stressapptest (Apache-2.0) |
| `host-health` | Node | Passive host/driver fault counters over the window; ECC, Xid, PCIe replays | ours (Go stdlib) |
| `dcgm-diag` | Node | NVIDIA DCGM diagnostics | wrapper only — **DCGM is not shipped**, see below |
| `ib-write-bw` | **Pair** | RDMA write bandwidth and latency across the link | linux-rdma/perftest (**BSD-2 option**) |
| `nccl` | **Pair** | Collective bus bandwidth and miscompares | nccl-tests + NCCL (BSD-3) |
| `gpudirect-rdma` | **Pair** | NIC-to-GPU peer-memory path | linux-rdma/perftest (**BSD-2 option**) |

`custom` is any image honouring the runner contract, with no built-in parsing.

Two of these carry a caveat worth reading before you rely on them:

- **`dcgm-diag` ships no DCGM.** Every DCGM *binary* package carries the NVIDIA
  DCGM License — §2(c) forbids distribution — even though the source is
  Apache-2.0. The image is our wrapper alone; the site mounts DCGM at
  `/usr/local/dcgm`. Without that mount the runner reports **Error**, not Fail:
  the hardware was never judged.
- **`gpu-burn` contains no code from [wilicc/gpu-burn](https://github.com/wilicc/gpu-burn).**
  Its licence is fine (BSD-2); what it *links* is not. Its numeric core is
  cuBLAS, and cuBLAS is a CUDA **toolkit** library, so the Container Toolkit
  does not inject it the way it injects `libcuda.so.1` — a working image would
  have to redistribute `libcublas`. The method is the value here (run a large
  GEMM repeatedly, compare each result against the first, count elements that
  differ), so this is a self-contained SGEMM instead. It shares its kernel with
  `thermal-soak`, which makes the two kinds' throughput figures the same
  measurement rather than two different numbers under one metric name.

Full provenance and per-image licence assertions are in [NOTICE](./NOTICE);
each runner's Dockerfile fails the build if a shipped binary references an
NVIDIA redistributable.

## Heterogeneity

A new accelerator or NIC ships a **runner image**, not a controller change — the
`runner` field on a test overrides the built-in image/command. The controller stays
vendor-neutral; vendor specifics live in images (mirrors how Glimmer keeps vendor
logic behind a single seam).

## Quick start

```sh
make install          # install CRDs into the current cluster
make run              # run the manager locally against it
kubectl apply -f config/samples/
kubectl get burninruns -w
```

## Cluster prerequisite for the fabric tests

**Node-scope tests need nothing beyond a working accelerator. The Pair-scope
fabric tests (`ib-write-bw`, `nccl`, `gpudirect-rdma`) need one node-level
setting, and without it they fail in a way that does not name its own cause.**

RDMA buffers are pinned memory, and containerd's systemd unit ships
`LimitMEMLOCK=8388608` — 8 MiB — which every pod inherits. perftest registers
`message size × queue pairs × 2`, so the ordinary `1 MiB × 4 QPs` lands exactly
on that ceiling and `ibv_create_cq` returns ENOMEM. What you see is:

```
Couldn't create CQ / Failed to create CQs / Couldn't create IB resources
```

and from NCCL, `ibv_reg_mr_iova2 failed with error Cannot allocate memory`.
Neither mentions a limit. Raise it on every node that will run fabric tests:

```sh
sudo mkdir -p /etc/systemd/system/containerd.service.d
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee /etc/systemd/system/containerd.service.d/10-memlock.conf
sudo systemctl daemon-reload && sudo systemctl restart containerd
```

Two things not to reach for instead. `securityContext.capabilities.add:
[IPC_LOCK]` does **not** work for a non-root container: Kubernetes has no
ambient-capabilities field, so runc clears the permitted set and `CapPrm`
reads as all zeros. And `docker run --ulimit memlock=-1` masks the problem
entirely — a runner verified under Docker can still fail as a pod, which is
exactly how this reached us.

`ib-write-bw` and `gpudirect-rdma` degrade rather than die: they read the limit
and shrink message size to fit, so a constrained node still measures (99.59
Gb/s at a forced 8 MiB, against 99.61 unconstrained). `nccl` cannot — it fails
even an 8-byte all-reduce at 8 MiB — so it reports **Error, not Fail**, because
the link was never measured. It will not fall back to `NCCL_IB_DISABLE=1`,
which would silently benchmark TCP sockets and report a plausible number for a
path nobody was qualifying.

## Runner contract

A runner is any image that exits **0 = pass, 1 = fail, 2 = skip** (not applicable
to this hardware) and prints `key=value` metrics on stdout. Any other exit code
is an **Error** — the runner malfunctioned and the hardware is unjudged, which is
never folded into a failure. Parsing is last-occurrence-wins, so a runner may
report progressively. That is the whole contract, in any language, which is what
keeps the vendor seam at the image boundary rather than in the reconciler.

Metric names are governed by [`pkg/contract`](./pkg/contract/metrics.go):
lowerCamelCase, and a dimensional metric ends in a registered unit suffix
(`busBandwidthGBs`, `latencyUs`, `gpuTempC`). A threshold naming a metric the
runner did not emit is a **failure**, never a pass.

## Thresholds compare exactly

There is no epsilon anywhere in this API, which is what makes each comparison
right for exactly one kind of metric:

- `GreaterThanOrEqual` / `LessThanOrEqual` gate a **measurement** — a floor, a
  ceiling, or both together for a band. Anything with a unit belongs here.
- `Equal` / `NotEqual` are **exact**, and exist for **dimensionless counters**:
  `eccErrors == 0`, `throttleEvents == 0`, `miscompares == 0`. On a counter
  exactness is the point — a tolerance around zero ECC errors is a tolerance for
  ECC errors.

Using an exact comparison on a continuous metric is the trap.
`sustainedClockPct Equal 83.22` asks an averaged sample to reproduce a decimal
string, so the gate fails on every healthy node forever and each failure is
reported in the same shape as a hardware verdict.
[`verdict.ValidateThresholds`](./pkg/verdict/lint.go) reports that — along with
gates on metrics the registry marks as evidence rather than acceptance, and
thresholds that can never be evaluated at all — at **authoring time**, while the
author is still there to fix it. It is advisory: evaluation is unchanged and
still fails closed.

## Status

Working end to end for **Node-scope and Pair-scope** tests: CRDs, manager, run
reconciler (pinned plan, wave cordoning, concurrency interlock, repeats,
error-retry, checkpointing, two-pod Pair rendezvous), schedule and fingerprint
reconcilers, threshold evaluation, and sink delivery over webhook / ConfigMap /
Prometheus. All eleven runner images are published and public.

### Qualified on hardware, not only in CI

Two NVIDIA DGX Sparks (GB10, `sm_121`), Kubernetes v1.32.0, ConnectX-7 over
200G RoCE. Node-scope: **10/10 passed** across both nodes. Pair-scope:
`ib-write-bw` 99.61 Gb/s at 1.56 µs, `nccl` 12.02 GB/s bus bandwidth with zero
miscompares, `gpudirect-rdma` correctly **Skipped** (GB10 exposes no
peer-memory provider).

That exercise is also where most of this design's sharp edges came from. **Every
acceptance threshold that had been derived from a spec sheet rather than
measured was wrong**, and most of them would have failed healthy hardware — a
`busBandwidthGBs >= 20` gate is 160 Gb/s, arithmetically impossible over a
~100 Gb/s link. The convention that came out of it, and the one to copy: **ship
thresholdless, gather fleet baselines, then pin.** The sample profiles carry
their measurements in comments next to each number so a reader can see what the
gate is made of.

### Known limitations

- **Group scope is not implemented** and settles as an honest `Error` — never a
  silent pass. See [Scope](#scope-what-runs-today).
- **The `v0.2.0` runner tags are `linux/arm64` only.** Published tags are
  immutable, so multi-arch begins at the next tag; until then an x86 fleet must
  build its own and set `spec.runner.image`.
- **Fabric tests need a node-level `RLIMIT_MEMLOCK` raise.** containerd ships an
  8 MiB limit that RDMA registration lands exactly on. See
  [the prerequisite section](#cluster-prerequisite-for-the-fabric-tests) — the
  failure does not name its own cause.
- **`dcgm-diag` needs the site to mount DCGM**, for the licensing reason above.
- **`thermal-soak`'s power-delivery-wedge detection is inferred from source, not
  observed** — no wedged part was available to test against. Tracked in
  [#61](https://github.com/baldwinSPC/glimmer-burnin/issues/61).

## License

[Apache-2.0](./LICENSE). See [NOTICE](./NOTICE). Contributions: [CONTRIBUTING.md](./CONTRIBUTING.md).

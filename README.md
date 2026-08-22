# Glimmer Burn-In Operator

A vendor-neutral, Kubernetes-native **hardware acceptance-testing** controller for
GPU/accelerator fleets. Define a burn-in profile once; run it against a node — or a
**pair/group of nodes** for interconnect validation — and get a structured pass/fail
verdict you can gate provisioning on.

> **Standalone by design.** This operator does **not** import any control plane. It
> exports verdicts through a generic `BurnInSink` (webhook / ConfigMap / Prometheus),
> so it integrates with a control plane of your choice — or your own system — with
> zero code dependency. CI enforces the no-import rule.

## Scope: what runs today

`BurnInTest.spec.scope` is `Node`, `Pair` or `Group`, and **all three are
executed**. A scope this operator version does not recognise is recorded as
**`Error`** — explicitly *not run*, hardware *not judged* — rather than skipped.
That is the fail-closed rule at work: a required acceptance test the operator
cannot run must never let hardware pass by omission.

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

**Group scope** runs a collective across **every** target node, one rank per
node. Target *i* is rank *i*; rank 0 is the **root**, it starts first, and no
other rank is created until it reports **Ready** — the same gate Pair uses, for
the same reason. Runners receive `BURNIN_RANK`, `BURNIN_NRANKS`,
`BURNIN_ROOT_HOST` (the root's DNS name) and `BURNIN_ROOT_NODE`. There is
deliberately no rank list: every collective bootstrap in practice has one rank
publish a handle the rest fetch, and a list would be a topology the operator has
to keep correct rather than a name the runner resolves.

A group produces **one** `TestResult` naming **every** node. That matters more
here than at Pair scope, not less: when one rank is faulty, every healthy rank
blocks waiting for it and reports the same timeout, so a per-node verdict would
indict the whole group for one node's fault. The report names the ranks that
**dissent** and summarises the rest by count.

- **`maxConcurrentNodes` must be at least the number of target nodes.** A group
  holds every one of its nodes for the whole test, so it costs one slot per rank.
  At any smaller cap the run is refused at start with that explanation.
- **At least two distinct target nodes are required.**
- **A `Skip` needs every rank to have reported**, as `Pass` does. `Error` and
  `Fail` are honoured from any rank, but neither `Skip` nor `Pass` may be
  concluded from a subset — both of them let the run settle `Passed`, so
  concluding either from one rank would certify nodes that never had a pod.
- Every rank is waited for, unlike a Pair — a collective is synchronous, so its
  ranks finish together, and a rank that has not finished is one the collective
  is still waiting on. A genuine hang becomes an `Error` naming the ranks that
  did not finish.

**No JobSet and no OpenMPI**, deliberately. Gang scheduling solves partial
placement under contention; this operator has already pinned every rank by
hostname to a node it admitted and cordoned, so there is no placement decision
left to make — and JobSet would be a controller every cluster must install plus a
second owner of pod lifecycle. OpenMPI would mean shipping an `sshd` and a key on
every accelerator node in the fleet to run a bandwidth test. The reasoning is in
`internal/controller/group.go`.

## Custom Resources

| Kind | Purpose |
|------|---------|
| `BurnInTest` | One reusable test, with `scope: Node\|Pair\|Group`, thresholds, and a pluggable runner image. |
| `BurnInProfile` | An ordered suite of tests + verdict policy (e.g. `acceptance`, `pair-network`). |
| `BurnInRun` | One execution of a profile against a target; status carries per-test metrics + the overall verdict. |
| `BurnInSchedule` | Runs a profile against a target on a cron schedule. Acceptance at install time only proves the hardware was good on the day it arrived. |
| `NodeFingerprint` | The captured hardware/network identity (GPUs, NICs/RDMA) a verdict is bound to — and the drift detector between runs. |
| `BurnInSink` | Where results are exported. The **only** integration seam. |

## Two runs never share a node

`maxConcurrentNodes` is a **facility interlock** — it bounds how many nodes one
run drives to their power and thermal limits at once, and it defaults to 1. It
bounds nothing *across* runs, and the cordon cannot: runner pods must tolerate
`node.kubernetes.io/unschedulable`, because the operator cordons the node it is
about to test, so a second run's pods schedule onto the first run's cordoned node
without complaint. Two runs each honouring a cap of 1 on the same node are two
full-power soaks on one machine, and each believes it is compliant.

So a run whose targets overlap an already-active run is **refused at start**: a
terminal `Error` (no hardware was judged — never a `Failed`), an `Admitted=False`
condition with reason `TargetsBusy`, and a message naming the run that holds the
node. Nothing is queued — a burn-in is scheduled maintenance, and silently
waiting behind another run makes its duration unpredictable. Re-create the run
when the first finishes, or set **`spec.force: true`** to accept the contention
deliberately; a forced run records `Admitted=True` with reason `AdmittedByForce`,
so a verdict measured under contention stays identifiable as one.

A cordon stamp naming a `BurnInRun` that no longer exists is **reaped** from the
node side, restoring the schedulability the stamp records. The check is an
uncached read matching name *and* UID — a recreated run of the same name is a
different run — so a live run's node is never released out from under it.

## Test kinds

**All eleven kinds ship a runner image, and every one is published and public**
under `ghcr.io/baldwinspc/glimmer-burnin-<kind>:<version>`; the tags the operator
defaults to live in `pkg/runnerimages/images.go`, which is the source of truth
and states what each release was and was not verified against. A test that names no
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
| `nccl` | **Pair, Group** | Collective bus bandwidth and miscompares | NCCL (BSD-3) |
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
vendor-neutral; vendor specifics live in images, kept behind a single seam rather
than scattered through the reconciler.

### Host architecture is not GPU architecture

Two independent axes, and conflating them is how an operator ends up unusable on
half the fleets it was written for:

| axis | values | selected by |
|---|---|---|
| **host (CPU)** | `linux/amd64`, `linux/arm64` | the image's manifest list |
| **GPU** | `sm_80`, `sm_90`, `sm_100`, `sm_120`, `sm_121`, … | the runner's gencode build arg |

The operator image and **every** runner image are built for `linux/amd64` and
`linux/arm64`. A node pulls only its own platform's layers, so one tag serves an
x86 rack and a Grace rack alike.

The GPU axis is per runner and is documented in each runner's README:

| runner | GPU coverage |
|---|---|
| `clockprobe`, `thermal-soak`, `gpu-burn` | cubins for sm_80 / sm_90 / sm_100 / sm_120 / sm_121 + PTX — covers A100 through GB300 out of the box |
| `memory-bw` | nvbandwidth's own arch list; copy-engine testcases launch no kernel, so the gencode does not affect any reported number |
| `compute-smoke` | **CC 12.x kernel only** — the NVFP4 kernel is CUTLASS `arch::Sm120`. Defaults `sm_121a` on arm64, `sm_120a` on amd64. Never a `Fail`, and the two ways it declines are **not** the same: an H100/A100/L40S `Skip`s (exit 2 — no block-scaled FP4 on those parts at all), while a **B200/GB200/B300/GB300 is an `Error`** (exit 3, hardware **unjudged**) — those parts *do* have NVFP4, by the `tcgen05.mma` path this image does not implement, so the test applies and was not run. SM10x needs its own runner (#10) |
| `nccl` | **one gencode, and you may need to change it.** Defaults `sm_121` on arm64, `sm_90` on amd64; a B200 or L40S fleet must rebuild with `--build-arg NCCL_GENCODE=…` |
| `ib-write-bw`, `gpudirect-rdma`, `memory-stress`, `host-health`, `dcgm-diag` | no gencode — nothing to choose |

> **The `v0.1.0` and `v0.2.x` runner tags are `linux/arm64` only.** Published
> tags are immutable, so they stay that way; multi-arch begins at `v0.3.0`. The
> operator pins `v0.6.0` for every runner — `pkg/runnerimages/images.go` is the
> source of truth, not this table.

## Documentation

| | |
|---|---|
| [`docs/concepts.md`](docs/concepts.md) | the CRDs, the three scopes, and how a run actually proceeds |
| [`docs/runner-contract.md`](docs/runner-contract.md) | **the extension point** — exit codes, metrics, the `BURNIN_*` environment, host access |
| [`docs/thresholds.md`](docs/thresholds.md) | authoring gates, applicability, the linter, and measure-then-pin |
| [`docs/sinks.md`](docs/sinks.md) | the delivery envelope, idempotency, the three sinks |
| [`docs/reports.md`](docs/reports.md) | JUnit, HTML, markdown and the NVVS-compatible document |
| [`docs/soaks.md`](docs/soaks.md) | running a soak, and the capacity it costs |
| [`docs/bare-metal.md`](docs/bare-metal.md) | `burnin run` on a host that is not a cluster member, and exactly what differs |
| [`docs/vendors/`](docs/vendors/README.md) | running a mixed fleet: what works on which vendor, and what to bring yourself |
| [`docs/verifying-images.md`](docs/verifying-images.md) | verifying the cosign signature on what you pulled |
| [`docs/dev/invariants.md`](docs/dev/invariants.md) | **the design rationale** — the rules and the failures each prevents |
| [`docs/dev/new-testkind-playbook.md`](docs/dev/new-testkind-playbook.md) | adding a TestKind, step by step, with the guard for each |

## Install

```sh
helm install burnin oci://ghcr.io/baldwinspc/charts/glimmer-burnin \
  --namespace glimmer-burnin-system --create-namespace
```

Or without Helm, from a clone:

```sh
make deploy
```

Chart values, the CRD-upgrade caveat and the two things to know before pointing
this at hardware are in
[`deploy/charts/glimmer-burnin/README.md`](deploy/charts/glimmer-burnin/README.md).

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

**A skip must be declared, not merely exited.** Exit 2 counts as `Skip` only if
stdout also carries a marker — an upper-case token ending in `_SKIP` at the start
of a line, like `IB_WRITE_BW_SKIP: this node has no RDMA device`. Exit 2 without
one is an `Error`.

That is not pedantry. An unrecovered Go panic exits **2**, and so does every Go
runtime fatal error — out of memory, concurrent map writes, stack exhaustion —
none of which a runner can recover from. A container log merges stdout and
stderr, so a crashed runner emits a stack trace and no `key=value` line at all —
which is exactly the shape of an honest skip, whose normal form is "nothing to
measure, nothing reported". Landing that on `Skip` is the worst available
outcome: `Skip` is never retried, leaves the retry budget unspent, and does not
affect the run's verdict, so a crashed runner reported a node as *out of scope*
inside a run that settled **Passed**.

It fails towards `Error` on purpose. A runner that skips honestly but forgets the
marker is reported unjudged and retried — visible and cheap. The opposite mistake
certifies a fleet nobody looked at.

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

Working end to end for **Node-, Pair- and Group-scope** tests: CRDs, manager, run
reconciler (pinned plan, wave cordoning, concurrency interlock, repeats,
error-retry, checkpointing, two-pod Pair rendezvous, segmented soaks), schedule
and fingerprint reconcilers, threshold evaluation, and sink delivery over
webhook / ConfigMap / Prometheus. All eleven runner images are published and
public.

A long soak should set **`spec.soak.segmentSeconds`**, which runs it as a
sequence of shorter pods so that an eviction, a drain or a reboot costs one
segment rather than the whole run. The verdict is still rendered once, over the
aggregate of the clean segments, using the per-metric combining rule each metric
declares in `pkg/contract`. See [docs/soaks.md](./docs/soaks.md) and
[config/samples/segmented-soak.yaml](./config/samples/segmented-soak.yaml).

### Qualified on hardware — and which tags that covers

Two NVIDIA DGX Sparks (GB10, `sm_121`), Kubernetes v1.32.0, ConnectX-7 over
200G RoCE. Node-scope: **10/10 passed** across both nodes. Pair-scope:
`ib-write-bw` 99.61 Gb/s at 1.56 µs, `nccl` 12.02 GB/s bus bandwidth with zero
miscompares, `gpudirect-rdma` correctly **Skipped** (GB10 exposes no
peer-memory provider).

> **That exercise measured the `v0.2.0` build of these sources, published as the
> `v0.3.0` tags.** Later tags are not covered by it. What stands behind a tag is
> recorded next to the pins in `pkg/runnerimages/images.go` rather than here,
> because the pins are what a fleet actually runs — and a release cut without a
> GPU available says so there in as many words. Do not read this section as a
> claim about whichever tag is newest.

**The multi-device conversion (`docs/dev/multi-device.md`) was verified on the
same two-node fleet.** `clockprobe`, `compute-smoke`, `gpu-burn`, `thermal-soak`
and `gemm-sweep` — every kind converted to iterate every allocated device rather
than device 0 alone — passed cleanly measuring their one real device each:
`deviceCount`/`devicesVisible` correctly report `1`, the soak family defaults to
`deviceConcurrency=all` and the measurement kinds default to `sequential`
exactly as designed, and `gemm-sweep` passed all five precision cells (fp4, fp8,
bf16, tf32, fp64) on both nodes. `fingerprint-probe`'s #380 fix was confirmed
too: `acceleratorCount=1` on a GB10, where it previously reported `0`. A longer
(5-minute) `thermal-soak`/`gpu-burn` pass on both nodes found no recurrence of
the SIGKILL pattern #280 investigated. **This is still a single-device
qualification** — the fleet has one GPU per node, so the N-device fold itself
(N > 1) remains unverified on real hardware, tracked by #402 the same as before.
Built and run from locally-built images matching this exact source tree; whether
new runner tags get published from it is a separate decision from the pins in
`pkg/runnerimages/images.go`.

That exercise is also where most of this design's sharp edges came from. **Every
acceptance threshold that had been derived from a spec sheet rather than
measured was wrong**, and most of them would have failed healthy hardware — a
`busBandwidthGBs >= 20` gate is 160 Gb/s, arithmetically impossible over a
~100 Gb/s link. The convention that came out of it, and the one to copy: **ship
thresholdless, gather fleet baselines, then pin.** The sample profiles carry
their measurements in comments next to each number so a reader can see what the
gate is made of.

### Known limitations

- **No 3-rank collective has ever executed on hardware.** The `nccl` runner does
  speak the Group rendezvous as of `v0.6.0` — it reads `BURNIN_RANK`,
  `BURNIN_NRANKS` and `BURNIN_ROOT_HOST`, and it is the one kind the operator
  will dispatch by default for a Group test — but no GPU cluster has run it.
  Verifying an N-rank collective needs three or more GPU nodes and this fleet has
  two; the bus-bandwidth scaling factor `2(n-1)/n` is exactly 1 at two ranks, so
  n=3 is the first configuration in which a wrong factor is visible at all
  ([#118](https://github.com/baldwinSPC/glimmer-burnin/issues/118)).

  **Do not work around this by pinning an older `nccl` tag as
  `spec.runner.image`.** Naming any image is the one thing that disables the
  operator's Group check, and `v0.4.0` and earlier have no `BURNIN_RANK` handling
  — they read the unset `BURNIN_ROLE` as a Node-scope run and exit 2 with a
  declared `NCCL_SKIP`, so the run settles `Passed` around a collective that
  never happened. Published tags are immutable and will do that forever.
- **The `v0.1.0` and `v0.2.x` runner tags are `linux/arm64` only.** Published
  tags are immutable, so multi-arch begins at `v0.3.0`. An x86 fleet pinning an
  older tag must repin or build its own; from `v0.3.0` onward one tag serves both
  architectures and no `spec.runner.image` is needed.
- **Fabric tests need a node-level `RLIMIT_MEMLOCK` raise.** containerd ships an
  8 MiB limit that RDMA registration lands exactly on. See
  [the prerequisite section](#cluster-prerequisite-for-the-fabric-tests) — the
  failure does not name its own cause.
- **`dcgm-diag` needs the site to mount DCGM**, for the licensing reason above.
- **`thermal-soak`'s power-delivery-wedge detection is inferred from source, not
  observed** — no wedged part was available to test against. Tracked in
  [#61](https://github.com/baldwinSPC/glimmer-burnin/issues/61).
- **A runner image published before the multi-device conversion measured one
  device on a multi-GPU node, even when asked for the whole board.** The
  in-cluster operator still gives an accelerator runner exactly what
  `resources.limits` requests, so this never applied there — it is a `burnin
  run` (bare-metal) story specifically. `pkg/localrun` already passes `--gpus
  all` / `--device nvidia.com/gpu=all` unconditionally (there is no flag to
  request fewer), so on a multi-GPU box every device was always VISIBLE inside
  the container; a device-0-only runner silently measured one of them and
  certified the board on that reading. `gpu-burn`, `thermal-soak`, `clockprobe`
  and `compute-smoke` (plus their ROCm siblings) and `gemm-sweep` now iterate
  every allocated device instead
  (`docs/dev/multi-device.md`), which is exactly why the absent-
  `BURNIN_RESOURCE_LIMITS` case in `device_fold.h`'s budget check means "iterate
  every visible device": bare metal never sets that variable, and on bare metal
  `all` genuinely IS the allocation. Published tags from before the conversion
  landed keep the old, one-device behaviour forever; repin to get the fix.

## Where this is going

[docs/roadmap.md](./docs/roadmap.md) — the test matrix as it stands, what each
milestone is for, and links to the issues that carry the work.

To add a test kind, start at
[docs/dev/new-testkind-playbook.md](./docs/dev/new-testkind-playbook.md): the
ordered checklist, and which guard test catches each mistake.

## License

[Apache-2.0](./LICENSE). See [NOTICE](./NOTICE). Contributions: [CONTRIBUTING.md](./CONTRIBUTING.md).

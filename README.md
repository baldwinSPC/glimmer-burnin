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

Node scope, with a runner in this repo:

| Kind | What it gates | Runner |
|------|---------------|--------|
| `compute-smoke` | Arch-correct FP4 kernel; `nonfiniteCount` | [`runners/compute-smoke`](./runners/compute-smoke) |
| `clockprobe` | Sustained clocks under load — catches a part pinned in a low P-state that every health check calls healthy | [`runners/clockprobe`](./runners/clockprobe) |
| `dcgm-diag` | NVIDIA DCGM diagnostics; XIDs, remapped rows, PCIe replays | [`runners/dcgm-diag`](./runners/dcgm-diag) |
| `host-health` | Passive host/driver fault counters over the window | [`runners/host-health`](./runners/host-health) |
| `memory-bw` | Device, H2D, D2H and D2D bandwidth | [`runners/memory-bw`](./runners/memory-bw) |
| `memory-stress` | Host DIMM stress via stressapptest (Apache-2.0) | [`runners/memory-stress`](./runners/memory-stress) |

Kinds the parser understands but that ship **no** runner here — `gpu-burn`,
`thermal-soak`, `nccl`, `ib-write-bw`, `gpudirect-rdma` — require an explicit
`spec.runner.image`, and say so at plan time rather than pull-failing per node.
`custom` is any image honouring the runner contract, with no built-in parsing.

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
Prometheus.

Not done: Group scope is recorded as `Error` rather than executed (see
[Scope](#scope-what-runs-today)). No `nccl` or `ib-write-bw` runner image ships
in this repo yet, so a Pair test must name its own `spec.runner.image`.

**Runner images: only `compute-smoke:v0.1.0` is published.** The other five
runners have source, Dockerfiles and tests in this repo, and
`internal/controller/pods.go` names the tags they *will* be published under, but
those tags are not in the registry yet. Until they are pushed by the manual
`publish-runner` workflow, use an explicit `spec.runner.image` — otherwise the
pod pull-fails and the run reports an infrastructure Error for that node.

## License

[Apache-2.0](./LICENSE). See [NOTICE](./NOTICE). Contributions: [CONTRIBUTING.md](./CONTRIBUTING.md).

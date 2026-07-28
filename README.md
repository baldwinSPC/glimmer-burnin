# Glimmer Burn-In Operator

A vendor-neutral, Kubernetes-native **hardware acceptance-testing** controller for
GPU/accelerator fleets. Define a burn-in profile once; run it against a node — or a
**pair/group of nodes** for interconnect validation — and get a structured pass/fail
verdict you can gate provisioning on.

> **Standalone by design.** This operator does **not** import any control plane. It
> exports verdicts through a generic `BurnInSink` (webhook / ConfigMap / Prometheus),
> so it integrates with [Glimmer](https://github.com/baldwinSPC/glimmer) — or your own
> system — with zero code dependency. CI enforces the no-import rule.

## Why pairs are first-class

Single-node burn-in (gpu-burn, DCGM, thermal soak) is table stakes. The value most
fleets are missing is **interconnect acceptance** — NCCL bus bandwidth over RoCE/IB,
`ib_write_bw`, GPUDirect RDMA — which is only meaningful **across at least two nodes**.
`BurnInTest.spec.scope: Pair` (and `Group`) makes the operator schedule and correlate
multi-node tests, not just per-node ones. This is a v1 requirement, not a roadmap item.

## Custom Resources

| Kind | Purpose |
|------|---------|
| `BurnInTest` | One reusable test (`gpu-burn`, `nccl`, `ib-write-bw`, `dcgm-diag`, `thermal-soak`, `custom`), with `scope: Node\|Pair\|Group`, thresholds, and a pluggable runner image. |
| `BurnInProfile` | An ordered suite of tests + verdict policy (e.g. `acceptance`, `pair-network`). |
| `BurnInRun` | One execution of a profile against a target; status carries per-test metrics + the overall verdict. |
| `NodeFingerprint` | The captured hardware/network identity (GPUs, NICs/RDMA) a verdict is bound to — and the drift detector between runs. |
| `BurnInSink` | Where results are exported. The **only** integration seam. |

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

## Status

Early scaffold: CRDs + manager + reconcile loop + RBAC are in place. Per-kind runner
scheduling and result parsing land in follow-up PRs.

First runner image: [`runners/compute-smoke`](./runners/compute-smoke) — NVFP4
block-scaled GEMM for NVIDIA Blackwell SM120/SM121 (GB10 / DGX Spark), verified on
real hardware.

## License

[Apache-2.0](./LICENSE). See [NOTICE](./NOTICE). Contributions: [CONTRIBUTING.md](./CONTRIBUTING.md).

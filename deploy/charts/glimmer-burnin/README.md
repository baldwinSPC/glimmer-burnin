# glimmer-burnin

Vendor-neutral, Kubernetes-native hardware acceptance testing for accelerator
fleets.

```sh
helm install burnin oci://ghcr.io/baldwinspc/charts/glimmer-burnin \
  --namespace glimmer-burnin-system --create-namespace
```

Or from a clone:

```sh
helm install burnin ./deploy/charts/glimmer-burnin \
  --namespace glimmer-burnin-system --create-namespace
```

## Before you point this at hardware

**This operator cordons the nodes it tests.** It cordons immediately before it
puts load on a node and releases once it is no longer holding any, so its
footprint tracks `maxConcurrentNodes` rather than the size of your target list.
The capacity arithmetic for long runs is in [`docs/soaks.md`](../../../docs/soaks.md).

**Runner images execute against real hardware**, and the defaults are
NVIDIA-oriented. Set `spec.runner.image` per test for anything else.

## No cert-manager

The operator has **no webhooks**, so this chart has no cert-manager dependency
and no certificate plumbing. That is deliberate and is enforced by a test —
please do not add one.

## Values worth knowing

| key | default | why |
|---|---|---|
| `replicaCount` | `1` | with `leaderElection` on. One replica plus a lease is not redundancy — it is what makes a **rolling update** safe, because `maxSurge` starts the second manager before the first is gone |
| `leaderElection` | `true` | turning it off with one replica makes every upgrade a window where two managers reconcile the same run |
| `tolerations` | control-plane + unschedulable | the operator cordons nodes; on a small cluster the manager may be scheduled on one, and without the unschedulable toleration it becomes unschedulable exactly when it is needed |
| `image.tag` | `""` | empty means the chart's `appVersion`, so "which chart" and "which operator" stay one question |
| `resources.limits` | memory only | no CPU limit: the manager is bursty at run start, and CFS throttling there delays cordons and pod creation on hardware the run is already holding |
| `metrics.serviceMonitor.enabled` | `false` | needs the Prometheus Operator's CRD, so the chart installs without it |

## CRDs and upgrades

CRDs live in `crds/`, which is Helm's convention and carries Helm's caveat:
**Helm installs them on first install and never upgrades or deletes them.**

That is stated rather than worked around. To upgrade the CRDs:

```sh
kubectl apply -f https://raw.githubusercontent.com/baldwinSPC/glimmer-burnin/v0.6.0/config/crd/
```

`helm uninstall` leaves the CRDs, and therefore every `BurnInRun`, in place. That
is the right default: a verdict is worth more than a tidy uninstall. Removing
them deletes every stored result in the cluster.

## Without Helm

```sh
make deploy     # kubectl apply of config/crd, config/rbac, config/manager
make undeploy   # removes the manager and its RBAC; leaves the CRDs alone
```

`config/` is applied by CI on every e2e run, so it is the path that is exercised
continuously. The chart's RBAC and CRDs are checked against it by
`hack/chart` — two install paths for one operator is two things that can drift.

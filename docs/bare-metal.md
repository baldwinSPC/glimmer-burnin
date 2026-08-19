# Burn-in on a bare host

`burnin run` executes the same acceptance profile a cluster would, on a machine
that is not a cluster member.

That matters most for exactly the hardware that needs it: a node whose defect
prevents it from joining a cluster cannot be tested by an operator that requires
it to have joined one.

```sh
burnin run -f suite.yaml --node spark-a --results-dir ./results
```

The suite file is **multi-document YAML of the same objects the CRDs use** —
`BurnInProfile` and `BurnInTest`, copy-paste identical between `kubectl apply`
and a bare host. Anything else in the file (sinks, schedules, runs) is ignored,
so one file can serve both paths. That is deliberate: a slimmer CLI-native
schema would be a third contract that can drift, which is the disease this whole
design exists to cure.

---

## One brain, two dispatchers

The promise is that both paths reach the **same verdict from the same
evidence**. A divergence would surface as a node that passes on one path and
fails on the other, and nobody would know which to believe.

So the shared parts are shared as code, not as prose:

| | package |
|---|---|
| what a runner's output means | `pkg/runner` |
| whether it satisfies a threshold | `pkg/verdict` |
| which image implements a kind | `pkg/runnerimages` |
| what a profile entry expands into | `pkg/plan` |
| the delivery envelope | `pkg/contract` |
| the sequencing, repeats and retries | `pkg/localrun` |

`pkg/localrun/conformance_test.go` is a decision table asserting the engine
matches the reconciler row by row, and every row names the controller code it
mirrors.

---

## What the CLI does not deliver, and why

`cmd/burnin/envelope_parity_test.go` holds a **ledger** — the exact list of
contract fields the local dispatcher does not populate. It fails in both
directions: a field that leaves the ledger must really be delivered, and a
field that starts being delivered must leave the ledger. The bound only
shrinks.

| Field | Why not |
|---|---|
| `Envelope.Cluster` | there is no apiserver; inventing a cluster ref would claim one that never existed |
| `Envelope.Baseline` | a run-lifecycle mode the local engine does not have |
| `Envelope.CancelReason` | a local run is one shot, driven by a person, cancelled by `^C` |
| `TestResult.Segments` | see below — this one is a decision, not a gap |
| `TestResult.Artifacts` | the artifact channel rides in a run-owned ConfigMap |

### Segmented soaks run as ONE local execution, on purpose

`spec.soak.segmentSeconds` divides a long test into a sequence of shorter pods.
It exists to **mitigate Kubernetes evictions**: a seven-day soak as a single
604,800-second pod loses the whole week to one eviction or one node reboot,
and with `retryOnErrorLimit` set the retry starts the week over from zero.
Segmenting costs one segment instead.

A local run has no evictions to mitigate. There is no scheduler, no
cluster-autoscaler, and no rolling manager update; the container runs until it
exits or the person running it stops it. So `pkg/localrun` runs a segmented soak
as **one execution for the whole duration** — asserted by
`TestASegmentedSoakRunsAsOneLocalExecution`.

Slicing it here would buy nothing and cost something real. The segmented path's
verdict comes from `AggregatedMetrics`, folded per metric by the `Aggregation`
rule each name declares in `pkg/contract` — `Sum` for a windowed counter, `Last`
for an NVML lifetime aggregate, `Min` for a floor, `Max` for a ceiling. Getting
that wrong in the certifying direction is how a node with four remapped rows
reads as twelve. A **second implementation of that fold** is exactly the drift
the two-dispatcher design spends its tests preventing, and it would be written
to serve a failure mode this path does not have.

A soak still soaks. `durationSeconds` is honoured in full; only the *pod
lifecycle* machinery is absent.

---

## Progress: checkpoints

A test may set `spec.checkpointIntervalSeconds`. The runner's metrics so far are
then published every interval, as a `Checkpoint` delivery carrying the same run
identity as the final verdict.

This is the case a long soak is most likely to meet. A multi-hour thermal soak
cancelled, killed at its deadline, or lost to a reboot at minute 200 otherwise
reports **nothing at all**, because a runner's metrics only reach the report
when the container exits.

**A checkpoint is evidence, never a verdict.** Thresholds are evaluated once, at
the end, against the completed execution. A mid-run sample that dips below a
floor is not a failure, because the run is not over — and a consumer that
treated one as a verdict would condemn hardware for a moment the test was
written to expect. The envelope says so in the only ways the contract has:
`reason: Checkpoint`, `phase: Running`, and a result with no violations.

Checkpoint delivery is **best-effort**. A sink that rejects one is warned about
and the soak continues; the verdict is what must be delivered loudly, and it
still is.

Requires a container runtime the engine can stream from. `docker`, `podman` and
`nerdctl` all qualify; a `ContainerRuntime` supplied by an embedding agent may
not, in which case the run behaves exactly as it did before — no checkpoints,
same verdict. That is a degradation of *evidence*, never of the verdict.

---

## Environment a runner receives

Everything the operator injects, plus the test's own `spec.runner.env`.

`valueFrom` is resolved where this dispatcher can honestly answer:

| `valueFrom` | on a bare host |
|---|---|
| `fieldRef: status.hostIP` | the host's primary address, from the routing table |
| `fieldRef: spec.nodeName` | `--node` |
| `secretKeyRef`, `configMapKeyRef`, `resourceFieldRef`, any pod-scoped `fieldRef` | **left unset**, and warned about by name |

Never set to the empty string. A runner meeting an unset variable can fail
loudly or skip honestly; one meeting an empty variable cannot tell "nobody could
answer this" from "the cluster says it is blank". A test that asks for
`status.hostIP` on a host with no routable address is **refused at plan time**
rather than run without it.

---

## Two machines: Pair scope

A link test needs both ends, and neither has to be in a cluster.

```sh
hostA$ burnin run -f suite.yaml --role server --node spark-a --results-dir r/
hostB$ burnin run -f suite.yaml --role client --peer 10.0.0.11 \
                  --peer-node spark-a --node spark-b --results-dir r/
```

**The client decides.** A point-to-point measurement is a property of the
*link*, so the client's results directory holds the one envelope for it, naming
both nodes. The server writes `sidecar/server-record.json`, which is a record of
that end and not a second opinion.

`--peer-node` is for **messages only, never for addressing**. A node name is not
a route, and treating it as one turns a naming mismatch into what reads as a
fabric fault.

Start ordering is yours — there is no scheduler to gate on readiness. Start the
server first if you can; a client that starts first retries into success rather
than reporting a fabric fault. If the test declares a `tcpSocket`
`readinessProbe`, the server end prints a line when its listener opens.

---

## Exit codes

The runner contract's shape, preserved at the process boundary, because a CI job
has to tell "this hardware is bad" from "this run never happened".

| Code | Meaning |
|---|---|
| 0 | every required test passed |
| 1 | a required test **failed** — measured, and fell short |
| 2 | nothing was judged (everything skipped) — **not a pass** |
| 3 | machinery: configuration, runtime, or a runner that never reported |

Only 3 is worth retrying.

---

## See also

- [concepts.md](concepts.md) — profiles, variants, scopes, the run lifecycle
- [soaks.md](soaks.md) — segmented soaks in a cluster, and the capacity arithmetic
- [thresholds.md](thresholds.md) — what a gate means and how it fails closed
- [runner-contract.md](runner-contract.md) — exit codes, metrics, skip markers
- [sinks.md](sinks.md) — the delivery envelope and how a consumer reads it

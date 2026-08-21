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

## N machines: Group scope

A collective needs N machines, and none of them has to be in a cluster.

```sh
spark-a$ burnin run -f suite.yaml --rank 0 --nranks 3 --node spark-a --results-dir r/
spark-b$ burnin run -f suite.yaml --rank 1 --nranks 3 --root 10.0.0.11 --node spark-b --results-dir r/
spark-c$ burnin run -f suite.yaml --rank 2 --nranks 3 --root 10.0.0.11 --node spark-c --results-dir r/

# then, with every rank's ranks/rank-NN.json copied into one directory:
$ burnin merge --results-dir merged/ --nranks 3
```

**There is no `--role`, and that is not an omission.** Group scope has no roles.
The contract is `BURNIN_RANK`, `BURNIN_NRANKS`, `BURNIN_ROOT_HOST` and
`BURNIN_ROOT_NODE`, and `BURNIN_ROLE` is *deliberately absent* — every fabric
runner branches on it, and a rank handed one would behave as a client. Passing
`--rank` and `--role` together is refused.

**Start ordering is yours: rank 0 first.** In a cluster the operator starts rank
0 and creates no other rank until it is Ready. Here there is no controller
watching pods, so you are the operator: start rank 0, wait for its listener,
then start the rest. That is documented rather than faked — a gate this command
pretended to enforce would be worse than one it plainly does not have. The
runners' connect-with-retry is the real gate, exactly as in a cluster.

### No rank writes the verdict

Every rank writes `ranks/rank-NN.json`. **None writes an envelope.**

This is the one place Group differs from Pair in kind rather than degree. At
Pair scope the client holds both halves of the measurement, so it can render the
link's verdict alone. A collective's verdict is not held by any rank: rank 0 has
the bandwidth figure, but *"did every rank take part"* is a fact invisible from
inside any one of them — and it is the fact that decides whether there is a
verdict at all.

So `burnin merge` renders it, because it is the only thing that can count to N.

### A partial collective has no verdict

If fewer records are found than `--nranks`, the merge **refuses**, naming the
ranks that never reported:

```
only 2 of 3 ranks reported — rank 2 never did. A collective is only MEASURED if
every rank took part, so there is no verdict here: a fold over the ranks that
happened to report would certify machines nobody looked at.
```

Exit code 3 — machinery, not a hardware verdict.

This holds for `Skip` as much as for `Pass`, and Skip is the more dangerous of
the two to get wrong: a Skip does not fail the run, it records "acceptance does
not apply to this hardware" and the run settles `Passed` around it. Honouring
rank 0's declaration for a group whose other members never ran would certify
every one of them on evidence from one node.

`Error` and `Fail` are honoured however many ranks reported — a rank that
positively failed established something, and a silent peer does not erase it.

### How metrics are elected

By what the metric **says it is**, through `pkg/contract`'s `Combination` — the
same election the operator makes, in the same code (`pkg/group`).

| Combination | across ranks |
|---|---|
| `Collective` | rank 0's value — it is the reporting rank for a property of the group |
| `Max` / `Min` / `Sum` | computed across every rank |
| unclassified | kept only if the ranks **agree**; a disagreement is dropped and declared unmeasurable, so a threshold fails closed |

`elapsedS` is the worked example of why `Aggregation` cannot stand in for
`Combination`: it is `Sum` across a soak's windows and `Max` across ranks —
eight ranks running 300 s took 300 s, not 2400 s.

A rank that declared a metric **unmeasurable** blocks the election entirely. If
ranks 0 and 1 report `eccErrors=0` and rank 2 is a part with no ECC to report,
electing `0` would certify the node that said it could not tell.

Disagreements are **recorded, not resolved**: the merged message names the keys
the ranks answered differently. It changes no verdict, which is the point — a
stored result saying "ranks disagreed about `gpuTempC`" is one an engineer can
act on, where a silent `70` next to eight node names is not.

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

# Getting results out

This page is for whoever is building the thing on the other end.

A **BurnInSink** is where a run's verdicts go. It is the *only* integration seam
this operator has: nothing downstream imports this project's code, and this
project imports nothing of the consumer's. What crosses the boundary is a
versioned JSON document — the **envelope**, defined in
[`pkg/contract/envelope.go`](../pkg/contract/envelope.go) — over a webhook, a
ConfigMap, or the operator's own `/metrics` endpoint.

`pkg/contract` is deliberately free of Kubernetes types. A consumer decoding a
verdict should not have to pull in client-go to do it.

---

## What a receiver must do, in five lines

1. **Reject an envelope whose `version` you do not recognise.** Do not guess at
   its shape.
2. **Dedupe on `deliveryId`.** Delivery is at-least-once. The key is derived, not
   random, so a retry carries the same one.
3. **Do not expect the body to be byte-identical across retries.** `sentAt`
   moves, and a re-sent terminal envelope may carry more results than the first
   attempt did.
4. **Gate acceptance on `reason: RunPhaseChanged` with a terminal `phase`** —
   `Passed`, `Failed`, `Error`, `Cancelled` (and `Skipped`, which a *result*
   takes and a run is not expected to). A `Checkpoint` is progress, not a
   verdict.
5. **Check `baseline`.** A `true` there means the run gated nothing. See
   [below](#baseline--a-passed-that-certifies-nothing).

---

## A worked envelope

A terminal delivery for a three-test acceptance run on one node, with one gate
missed, one gate that never ran, and one artifact refused. Annotations follow
the block.

```json
{
  "version": "burnin.glimmer.ai/v1alpha1",
  "deliveryId": "9f2c1ab0d5e34786b1c2a4f60e7d8931",
  "reason": "RunPhaseChanged",
  "sentAt": "2026-08-08T04:11:52.318Z",
  "phase": "Failed",
  "run": {
    "namespace": "burnin",
    "name": "node-acceptance-h9x2",
    "uid": "6f0e6c62-0e6e-4d9a-9a5b-2c1f9a6d1a77",
    "profile": "node-acceptance"
  },
  "cluster": {
    "name": "spark-lab-1",
    "uid": "1b7cf0a2-32b4-4a51-8d3c-0f9e4a71c505"
  },
  "fingerprint": {
    "spark-01": "kernel=6.11.0-1013-nvidia os=Ubuntu 24.04.2 LTS arch=arm64 nvidia.com/gpu.product=NVIDIA GB10"
  },
  "results": [
    {
      "name": "sustained-clock",
      "kind": "clockprobe",
      "scope": "Node",
      "phase": "Failed",
      "nodes": ["spark-01"],
      "startedAt": "2026-08-08T03:52:04Z",
      "finishedAt": "2026-08-08T04:07:31Z",
      "metrics": {
        "sustainedClockPct": "41.2",
        "ratedBoostClockMHz": "1657",
        "throttleClassification": "thermal",
        "throttledSamples": "812"
      },
      "message": "sustainedClockPct: got 41.2, need GreaterThanOrEqual 80",
      "violations": [
        {
          "index": 0,
          "metric": "sustainedClockPct",
          "cause": "Measurement",
          "kind": "Unsatisfied",
          "reason": "sustainedClockPct: got 41.2, need GreaterThanOrEqual 80"
        }
      ]
    },
    {
      "name": "host-fault-counters",
      "kind": "host-health",
      "scope": "Node",
      "phase": "Passed",
      "nodes": ["spark-01"],
      "startedAt": "2026-08-08T04:07:35Z",
      "finishedAt": "2026-08-08T04:08:02Z",
      "metrics": { "xidEvents": "0", "pcieReplayErrors": "0" },
      "message": "ECC is not exposed by NVML on this part [not evaluated: eccErrors is unmeasurable on this hardware]",
      "notEvaluated": [
        { "metric": "eccErrors", "reason": "unmeasurable on this hardware" }
      ],
      "unmeasurable": ["eccErrors"]
    },
    {
      "name": "dcgm-diagnostics",
      "kind": "dcgm-diag",
      "scope": "Node",
      "phase": "Passed",
      "nodes": ["spark-01"],
      "startedAt": "2026-08-08T04:08:06Z",
      "finishedAt": "2026-08-08T04:11:48Z",
      "metrics": { "diagTestsFailed": "0" },
      "artifacts": [
        {
          "name": "dcgm.json",
          "mediaType": "application/json",
          "sizeBytes": 48213,
          "digest": "sha256:3f1ab8…",
          "configMap": "node-acceptance-h9x2-artifacts",
          "key": "dcgm-diagnostics.node-acceptance-h9x2-t2-a1-7c04b1e9.dcgm.json"
        },
        {
          "name": "nvidia-smi-q.txt",
          "sizeBytes": 62914,
          "dropped": "the run's 921600-byte artifact budget is exhausted (901207 used); evidence not kept"
        }
      ]
    }
  ],
  "summary": { "passed": 2, "failed": 1, "errored": 0, "skipped": 0 }
}
```

What is worth reading closely:

- **`sustained-clock` is the only result that says anything about the
  hardware.** Its violation carries `cause: Measurement` — a real number that
  missed a real bar. Had it been `Evidence` or `Authoring`, no node would be
  implicated. Read `cause` before you dispatch an engineer.
- **`message` and `violations` are not the same statement.** `message` names
  only the first violation and is frozen at that behaviour; `violations` is the
  complete list. A test that missed three different gates delivers one sentence
  about the first. Never parse `message`.
- **`host-fault-counters` passed with a gate that never ran.** `eccErrors` is in
  `unmeasurable` because the runner positively declared it cannot measure ECC on
  this part, so the profile's `RequiredIfMeasurable` gate on it appears in
  `notEvaluated` instead of being applied. That is *not* the same as a passing
  ECC gate, and a consumer that treats it as one will eventually certify a part
  whose ECC nobody looked at.
- **The refused artifact is still delivered.** A ref with `dropped` set is a
  fact about the run: "the runner produced no evidence" and "the evidence was
  too large to keep" are different observations and only one sends somebody to a
  node. `configMap` and `key` are empty exactly when `dropped` is set.
- **`summary` counts entries in `results`, not tests and not nodes.** Three
  results, three counted. See [Summary](#summary--exhaustive-and-not-about-tests).

---

## The envelope, field by field

### Top level

| Field | What it means to a consumer |
|---|---|
| `version` | The schema. Currently `burnin.glimmer.ai/v1alpha1`. Reject what you do not recognise rather than guessing. |
| `deliveryId` | The idempotency key. Stable across retries of the same delivery — see [Idempotency](#deliveryid-and-idempotency). |
| `reason` | Why this was sent: `RunPhaseChanged`, `TestCompleted` or `Checkpoint`. Lets you tell a progress update from a verdict without diffing state. |
| `sentAt` | When *this attempt* was built, UTC. It moves on every retry; it is not the run's clock and must not be used to order anything. |
| `phase` | The run's phase at send time: `Pending`, `Running`, `Passed`, `Failed`, `Error`, `Cancelled`. All but the first two are terminal, as is `Skipped` — which a *result* takes and a run is not expected to, though `pkg/report` and the exporter both treat it as a run phase so a consumer should too. |
| `baseline` | `true` means the run applied **no thresholds at all**. See [below](#baseline--a-passed-that-certifies-nothing). Omitted when false. |
| `cancelReason` | Present only when `phase` is `Cancelled`. Distinguishes an operator-requested stop from a deadline expiry — opposite meanings about the hardware. |
| `checkpointSequence` | Present only when `reason` is `Checkpoint`. The ordering key for a run's progress record; checkpoints can arrive out of order and this is what puts them back. |
| `run` | Which run this describes. See below. |
| `cluster` | Which cluster it came from. Optional; see below. |
| `fingerprint` | Node name → a summary of what hardware the verdict applies to, captured **once, at run start**. A verdict without it is not portable evidence. |
| `results` | Every execution recorded so far, cumulative. Present on checkpoints too, mid-flight results included. |
| `summary` | The tally. Exhaustive over the four terminal outcomes; see below. |

`fingerprint` values are a flat space-separated string, currently
`kernel=… os=… arch=…` plus whichever of `nvidia.com/gpu.product`,
`glimmer.ai/gpu-arch` and `glimmer.ai/hw-class` the Node carries. Treat it as
opaque evidence, not as a parseable structure — the shape is data capture and
nothing in the operator branches on it.

**`run`** (`RunRef`):

| Field | Meaning |
|---|---|
| `namespace`, `name` | How a human finds the run with `kubectl`. |
| `uid` | The stable identity. **Key your records on this**, not on name: a name can be reused after a run is garbage-collected. |
| `profile` | The BurnInProfile this run executed. |

**`cluster`** (`ClusterRef`, optional): `name` comes only from an explicit
`--cluster-name` flag on the manager and is **never guessed** — a wrong cluster
identity attributes a verdict to the wrong fleet, which is worse than no
identity. `uid` is the `kube-system` namespace UID, read once at manager start.
An operator with neither configured emits neither field. Ingest is usually
per-cluster and authenticated per-cluster, so this looks redundant right up
until anybody aggregates, replays or archives envelopes.

### `results[]` — one entry per execution unit

| Field | What it means to a consumer |
|---|---|
| `name` | The test name from the profile. Together with `nodes` it is result identity. |
| `kind` | The `TestKind` — `clockprobe`, `nccl`, `dcgm-diag`, … |
| `scope` | `Node`, `Pair` or `Group`. |
| `phase` | `Running`, `Passed`, `Failed`, `Error`, `Skipped`, `Cancelled`. |
| `nodes` | Every node this result covers. **One entry for a Node test; two for a Pair; all of them for a Group** — a Pair verdict is about the *link* and a Group verdict is about the *collective*, and splitting either per node sends an engineer to the wrong part. |
| `startedAt` / `finishedAt` | The first attempt's start and the last attempt's end, so the pair spans the test's real occupancy of the hardware — errored attempts and the gaps between repeats included. `startedAt` is when execution actually began, not when the pod was created. |
| `metrics` | The parsed `key=value` results, **as strings and not all numeric** — `throttleClassification` above is one of five words, and `computeCap` is a `major.minor` version. With repeats and retries these are the metrics of the attempt that *decided* the verdict, so they always explain the phase beside them. |
| `message` | Human-readable outcome. Names only the **first** violation, or the underlying error when `phase` is `Error`. Frozen. Do not parse it. |
| `violations` | Every threshold this test failed, in spec order. Empty unless `phase` is `Failed`. |
| `notEvaluated` | Thresholds that were never applied. **Not passes** — and [incomplete on a `Failed` result](#notevaluated--a-gate-that-did-not-run). |
| `unmeasurable` | Metric names the runner positively declared it cannot measure on this hardware. A claim about the **part**. |
| `artifacts` | Non-metric evidence, **by reference**. The payload is not in the envelope. |
| `variantAxes` | The matrix labels this cell came from — `{"precision": "fp4"}`. Group a sweep by these rather than by splitting test names on a hyphen, which works until a variant is called `fp8-dense` and then works wrongly and silently. |
| `segments` | Present **only for a segmented soak**, absent for an ordinary execution. See below. |

### `results[].segments` — a soak, and how much of it burned

```json
"segments": { "completed": 40, "required": 288, "segmentSeconds": 900, "truncatedAttempts": 23 }
```

A soak is divided into a sequence of shorter pods so that an eviction costs one
window instead of the week. That is a **scheduling** decision and it deliberately
does not change the verdict — thresholds are applied once, at the end, to the
fold — so `phase` and `metrics` look identical whichever way a test was run.
This block is the only thing that tells them apart.

| Field | Meaning |
|---|---|
| `completed` / `required` | **`40 of 288` and `288 of 288` are different statements about a node.** `completed` counts only segments that finished *cleanly* and contributed to the fold, so it does not advance for an interruption — `completed < required` on a terminal result means the soak was cut short. |
| `segmentSeconds` | One window's length. `completed` is a count, and a count is not a duration; the envelope does not carry the run's spec, so without this there is no way to turn `40` into ten hours. |
| `truncatedAttempts` | Passing segments dropped from the attempt history to keep the status writable. A consumer counting delivered attempts undercounts by exactly this much, and would never learn that it did. The verdict is unaffected — it is read from the persisted fold, never from the attempt list. |

Two things follow for a consumer. **`metrics` on such a result is a FOLD**, not a
reading: `gpuTempC` is the peak across every window and `elapsedS` is the sum of
them. How each key combined is declared beside its name in the registry and
answered by `contract.AggregationFor` — query that rather than assuming, and note
that an *unregistered* name folds `Last`. And **a segmented soak that settles
`Error` still reports the aggregate**, because the errored window measured
nothing; read `elapsedS` against the declared duration to see how far it got.

What is **not** in a result, and where to get it: per-attempt history
(`attempts`, including `AttemptTrigger`, which records why each attempt
happened) lives on the BurnInRun object only. So do the run's conditions —
including `ThresholdsSound`, the advisory the linter writes. A consumer that
needs either reads the object; the envelope does not carry them.

### `results[].violations[]`

| Field | Meaning |
|---|---|
| `index` | Position in the test's `spec.thresholds`. |
| `metric` | The metric name, as written in the profile. |
| `cause` | **Read this first.** Who should act. |
| `kind` | The specific route, finer-grained than `cause`. |
| `reason` | Human-readable detail. |

`cause` has three values and they route to three different people:

| `cause` | What happened | Who acts |
|---|---|---|
| `Measurement` | A real measurement fell short of a bar that was applied to it. | Hardware. This is the only cause that is evidence about the part. |
| `Evidence` | The runner's report could not support a judgement — the metric is missing, unparseable, non-finite, self-contradictory, or declared unmeasurable under a gate that requires it. | Nobody, yet. **The node is unjudged, not condemned** (the threshold still fails closed). |
| `Authoring` | The threshold itself is broken — a `value` that does not parse, or is NaN/±Inf. | The profile author. No node should be touched. |

`kind` currently takes `Unsatisfied`, `NotReported`, `NonNumeric`,
`NonFinite`, `Contradictory`, `UnmeasurableRequired`,
`ThresholdValueNonNumeric`, `ThresholdValueNonFinite`. It is finer-grained
precisely so new routes can be added without reclassifying the ones already
there — so **switch on `cause`, and treat an unfamiliar `kind` as its stated
`cause`**. A single `Failed` test can mix causes.

### `results[].notEvaluated[]`

`{"metric": …, "reason": …}`. Today the only reason is
`unmeasurable on this hardware`.

### `results[].artifacts[]`

| Field | Meaning |
|---|---|
| `name` | The name the runner announced. |
| `mediaType` | As announced by the runner. May be absent. |
| `sizeBytes` | The payload's **true** size — including on a ref that was dropped for exceeding a cap, which is the case where it matters most, because "how much evidence did we lose" is the next question. |
| `digest` | `sha256:<hex>` over the stored payload. Verify what you fetch against it. |
| `configMap`, `key` | Where the payload lives, in the run's namespace. The key is `<test>.<pod>.<artifact name>` — the pod name carries the attempt, which is what keeps a repeat from overwriting the evidence of the attempt before it and keeps a Pair's two endpoints apart. **Both empty when `dropped` is set.** |
| `dropped` | Why the evidence was not kept. Empty when it was. |

The payload is deliberately not in the envelope: a consumer that only wants the
verdict should not have to carry a megabyte of dcgmi JSON to get it. Two things
to know before you write the fetch:

- The artifact ConfigMap is **owned by the run** and is garbage-collected with
  it. If the run sets `spec.ttlSecondsAfterFinished`, fetch before it expires.
- A payload that is not valid UTF-8 is stored in the ConfigMap's `binaryData`,
  not `data`, and is therefore base64 on the wire. Handle both. (The apiserver
  does not reject invalid UTF-8 in `data` — it stores it and mangles each bad
  byte to U+FFFD on the way out to any JSON client, kubectl included, so the
  evidence looks present and only the digest reveals otherwise.)

---

## Delivery reasons

Three, and `pkg/contract.Reasons` is the complete list.

| `reason` | Sent when | What it is for |
|---|---|---|
| `RunPhaseChanged` | The run enters a new phase, including the terminal one. | **This is the delivery a consumer gates acceptance on.** |
| `TestCompleted` | One execution unit reaches a verdict. | Per-test streaming. The event key is `<test>/<node>` (nodes joined by `+` for a Pair or Group), so two nodes finishing the same test cannot dedupe away against each other. |
| `Checkpoint` | Periodically, while the run is still going. | A multi-day soak would otherwise deliver nothing between start and finish, leaving a consumer unable to tell "soaking" from "wedged". |

Checkpoints are opt-in and cadence-driven: a test asks for them with
`spec.checkpointIntervalSeconds`, and the run's cadence is the **shortest**
positive interval any test in the pinned plan asked for. There is one cadence
for the whole run because a checkpoint is a cumulative snapshot of the run, not
of the test that happened to be due.

**A checkpoint carries evidence, never a verdict.** Its `results` include
in-flight executions with their latest parsed metrics, and **no threshold has
been applied to any of it**. Its `summary` counts only results that have
settled, so it will not add up to the number of results while work is
outstanding.

Two ordering facts that follow from the code and are worth designing around:

- A **phase-change** envelope is sent *before* the operator writes the new
  status to the apiserver. A receiver can therefore briefly see a transition the
  BurnInRun object does not show yet. That order is deliberate: a crash in
  between redelivers with the same `deliveryId`, whereas the reverse order would
  drop the transition entirely.
- A **checkpoint** is sent *after* the status write, for the mirror-image
  reason: the cheap failure is a consumer seeing a snapshot the object already
  holds.

### What happens when your endpoint is down

| | |
|---|---|
| **In-process retry** | 2 attempts, ~1 s apart, bounded by a 15 s timeout that covers the *whole fan-out across every sink named by the profile*. Size your handler accordingly. |
| **Non-terminal deliveries** | Not re-scheduled. A missed `Running` phase change or a missed `TestCompleted` is not replayed; the next checkpoint and the terminal envelope carry the cumulative state anyway. |
| **Checkpoints** | Deliberately not retried. Each carries the whole cumulative state, so a retry would only queue a stale snapshot behind a fresher one. |
| **The terminal envelope** | **Is** retried, because it is the one delivery with no later transition to carry it. The run records it as pending on itself and re-sends every 5 minutes until a sink accepts. |
| **A failed delivery never fails the run.** | A broken webhook must not stop hardware acceptance. The error lands on the BurnInSink's `status.lastError`, and `status.lastDelivery` records the last success — that is where an operator looks for delivery health, not the controller log. |

---

## `deliveryId` and idempotency

The key is `sha256(runUID ‖ reason ‖ eventKey)`, truncated to 128 bits and hex
encoded. It is a **pure function of what the delivery describes** — never of the
wall clock, never of a random source.

| `reason` | event key |
|---|---|
| `RunPhaseChanged` | the target phase (a run enters each phase once) |
| `TestCompleted` | `<test>/<node>[+<node>…]`, or the bare test name for a result that covers no node at all (a resolution error, say) |
| `Checkpoint` | the sequence number |

The checkpoint sequence is *derived* — it is the index of the interval window
the checkpoint falls in, counted from the run's start — rather than counted from
a persisted counter. That is the whole reason it is a number and not a
timestamp: a timestamped key would mint a new identity on every attempt and
defeat deduplication, turning a flaky endpoint into a flood of near-identical
records.

**What a receiver must do:** treat delivery as at-least-once, keep a dedupe
table keyed on `deliveryId`, and make applying an envelope idempotent. For a
webhook the key also arrives as the **`Idempotency-Key` HTTP header**, so a
gateway or ingest front-end can drop a repeat without parsing JSON.

**What a receiver must not do:** assume duplicate bodies are identical. `sentAt`
differs, and a terminal envelope re-sent five minutes later may carry results
that were still in flight on the first attempt. Last-write-wins on the key is
the safe rule.

---

## The three sinks

A profile names sinks by name in `spec.sinks`; they must be BurnInSink objects
**in the same namespace as the run**. The list is pinned into the run's plan at
start, so editing a profile's sinks mid-run does not change where an in-flight
run delivers. A run whose profile names no sinks produces no deliveries at all —
its verdict lives only on the BurnInRun object.

| Type | Direction | Use it for | Hard limit |
|---|---|---|---|
| `Webhook` | push, over HTTPS | The real integration. An external control plane, an ingest service, anything that must not lose a verdict. | Your endpoint must answer within the 15 s fan-out budget. |
| `ConfigMap` | write, in-cluster | Fleets with no egress. | ~900 KiB for the **whole ConfigMap**, across every run delivering to it. |
| `Prometheus` | pull, scraped | Dashboards and alerts. | Not durable. Never gate acceptance on it. |

### Webhook

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInSink
metadata:
  name: results-webhook
spec:
  type: Webhook
  webhook:
    url: https://results.example.com/api/v1/burnin
    tokenSecretRef:
      name: burnin-webhook-token
      key: token
```

A `POST` with `Content-Type: application/json`, the `Idempotency-Key` header,
and — if `tokenSecretRef` is set — `Authorization: Bearer <token>`. The token is
read from the Secret in the sink's namespace, held only in memory, and never
written to status, logs or the sink's description.

How the operator reads your response status:

| Status | Treated as |
|---|---|
| 2xx | delivered |
| 408, 429 | **retryable** — an explicit "try again" or "slow down", despite being 4xx |
| other 4xx | **permanent**. Retrying sends identical bytes and gets an identical answer while hiding the cause behind attempt counts. Bad credentials land here, which is exactly the case worth surfacing on attempt one rather than attempt six. |
| 5xx, transport failure (DNS, refused, timeout) | retryable |

Up to 2048 bytes of your response body are quoted back in the operator's error
message and onto the sink's `status.lastError`, so a useful error string is
worth returning.

`insecureSkipVerify: true` disables TLS verification and exists for development.
It is a field on the CRD rather than a hidden flag so that using it is visible
in the object anybody can read.

**A misconfigured sink fails immediately, not slowly.** A `Webhook` type with no
`webhook` block, a Secret that does not exist, a missing key, or an empty value
is a permanent error at build time, reported on the sink's `status.lastError` on
the first attempt: a sink pointing at a Secret that does not exist will not start
working because the operator asked again.

### ConfigMap

```yaml
spec:
  type: ConfigMap
  configMap:
    name: burnin-results
```

Each envelope is stored as one entry keyed by its `deliveryId`, in the sink's
namespace. That makes it idempotent by construction — a redelivery overwrites
the same key rather than appending a duplicate.

The limit is the one to plan for. etcd caps an object at about 1 MiB, and this
sink refuses **permanently** once the ConfigMap would exceed ~900 KiB, with an
error telling you to rotate or shard it. It fails loudly rather than writing an
object the apiserver will reject or silently dropping a verdict to make room.
The ConfigMap is not owned by any run, so it accumulates across runs until
something rotates it. Point long-lived or high-volume profiles at separate
sinks.

### Prometheus

```yaml
spec:
  type: Prometheus
```

Prometheus pulls, so a "Prometheus sink" cannot be a destination the way a
webhook is. What it is instead is a **selector**: naming it in a profile marks
that profile's runs for exposition on the manager's existing `/metrics`
endpoint. Every envelope the run would have delivered is reflected into the
in-process exporter. There is no network step, so delivery always succeeds and
nothing is queued or retried — and the sink's `status.lastDelivery` records it
like any other, because it did take the result.

It carries no configuration, which is why the CR has no `prometheus` block.
Exposition is opt-in per profile rather than a global switch, so a cluster
running a thousand throwaway smoke runs does not have to publish all of them.

The series (all gauges):

| Metric | Labels |
|---|---|
| `burnin_run_phase` | `namespace`, `run`, `profile`, `phase` — 1 for the phase the run is in, 0 for the others |
| `burnin_run_tests` | `namespace`, `run`, `phase` — recorded results by phase |
| `burnin_run_start_time_seconds` | `namespace`, `run` |
| `burnin_run_finish_time_seconds` | `namespace`, `run` — published only once the run is terminal |
| `burnin_test_phase` | `namespace`, `run`, `test`, `kind`, `scope`, `node`, `phase` — always 1 |
| `burnin_test_metric` | `namespace`, `run`, `test`, `kind`, `scope`, `node`, `metric`, `unit`, `threshold_use` |

Plus `burnin_exporter_runs_tracked`, `burnin_exporter_runs_evicted_total` and
`burnin_exporter_metrics_dropped_total{reason=…}` for the exporter's own health.

Four things that will surprise you if nobody says them:

- **Exposition is not durable delivery.** A scrape that does not happen before
  the operator restarts loses that sample. The system of record is the BurnInRun
  object and the envelope delivered to a Webhook or ConfigMap sink. **Pair a
  Prometheus sink with a durable one; a fleet that gates acceptance on a scrape
  has gated it on a cache.**
- **Only metrics registered in `pkg/contract` become series.** An unregistered
  name is counted in `burnin_exporter_metrics_dropped_total{reason="unregistered"}`
  and not published, because an unregistered name is unbounded label input and
  that is the one thing that can actually take a Prometheus server down. The
  filter is registration, not thresholdability: `threshold_use=Evidence` metrics
  *are* charted, with the registry's judgement carried in the label so nobody
  writes a page-at-3am alert on one without seeing it.
- **Non-numeric and non-finite values are dropped**, under
  `reason="non_numeric"` and `reason="non_finite"`. NaN is a legal Prometheus
  sample but as a measurement it means the runner emitted garbage, and
  publishing it makes every average downstream non-finite too.
- **A metric belongs to a node only when exactly one node produced it.** A
  Group-scope figure — NCCL bus bandwidth across eight ranks — is a property of
  the set, so its `node` label is empty. Copying it onto each member would let a
  per-node dashboard sum one measurement eight times.

Series are per-run and runs are created forever, so the exporter holds at most
`DefaultMaxRuns` (128) runs and evicts the least recently observed beyond that.
Eviction silently drops a dashboard's data, which is only tolerable because the
verdict lives elsewhere; `burnin_exporter_runs_evicted_total` is what makes it
visible rather than mysterious.

---

## Versioning: add fields, never repurpose them

`version` is `burnin.glimmer.ai/v1alpha1`. The rule the whole contract rests on:

> Changing a field's meaning changes it for every consumer at once. **The
> envelope is additive: add fields, do not repurpose them.**

One field in the worked example shows what that rule looks like in practice.
`results[].message` still names only the first violation, character for
character, even though every threshold is now evaluated. It is marked FROZEN in
the source and the complete picture was added *beside* it, as `violations`,
rather than by widening it — because the fields it derives from are public API
with **two dispatchers** (this operator, and Glimmer's pre-Kubernetes burn-in
path, which runs the same runner images), and silently widening a field they
both read would change acceptance reporting for both.

**What a consumer should do with a field it does not recognise: ignore it.** A
new field is not a schema break and does not warrant rejecting the envelope.
Reject on `version` alone.

**What a consumer should do with a field it expected and did not get:** treat
absence as absence, not as zero. Every optional field here is `omitempty`, and
several of them mean something precise by being absent — `cancelReason` outside
a `Cancelled` phase, `checkpointSequence` outside a checkpoint, `cluster` on an
operator with no cluster identity configured.

The operator validates before it sends: an envelope missing `version`,
`deliveryId`, `reason`, `run.uid` or `sentAt`, or carrying a metric name that
breaks the naming grammar, fails locally rather than landing on your ingest
endpoint. Shipping a malformed document to somebody else's service is worse than
failing where the error is visible.

---

## The five things consumers get wrong

### `summary` — exhaustive, and not about tests

```json
"summary": { "passed": 2, "failed": 1, "errored": 0, "skipped": 0 }
```

The counters tally **entries in `results`**, one per execution unit — which for
Node scope means one per (test, node), and for Pair and Group means one per test
covering all of its nodes. A 2-test profile against 3 nodes at Node scope
finishes with up to 6 counted, so **`passed == number of tests` is not a check
that works**.

All four terminal outcomes are counted. `errored` and `skipped` are kept as
separate counters and must never be folded into `failed`: an errored execution
measured nothing and a skipped one did not apply, and neither is a hardware
verdict. A summary reading "0 failed" beside eight executions that never ran
would be a clean sweep of hardware nobody looked at. On a checkpoint, results
still in flight count towards nothing, so the four will not add up to
`len(results)`.

### `violations` versus `message`

`message` is one sentence naming the first violation, and it is frozen there.
Before `violations` existed, a run that failed three gates for three different
reasons delivered as one sentence about the first, and a consumer had to parse
English to learn whether a node was implicated. Read `violations`; read
`message` only to show a human.

### `notEvaluated` — a gate that did not run

Not a pass, and not a failure. A `Passed` test whose ECC gate never applied must
never be indistinguishable from one whose ECC gate applied and was satisfied. If
your acceptance policy requires a particular gate to have been *applied*, check
`notEvaluated` for it — the phase alone will not tell you.

**On a `Failed` result this list is deliberately incomplete.** It carries the
un-applied gates found *before* the first violation, which is what the field
reported back when evaluation stopped there; a gate sitting after the first
violation went un-applied too and is absent. The truncation was kept rather than
quietly widened, for the same reason `message` was — see
[Versioning](#versioning-add-fields-never-repurpose-them). So: read it together
with `violations`, and on a `Failed` result never treat a metric's absence from
`notEvaluated` as proof that its gate ran.

### `unmeasurable` — a claim about the part

These are metric names the runner **positively declared** it cannot measure on
this hardware, by emitting the reserved value `n/a`. "This part has no ECC" and
"this part reported zero ECC errors" are different statements about different
hardware.

The rule that makes this safe, and that a consumer can rely on: **absence is not
a declaration.** A metric that merely never appears is *not* in `unmeasurable`
and its gate still fails closed, under every applicability, because a crashed
probe must not become an acceptance. Only a runner that looked and found nothing
to measure may declare.

### `baseline` — a `Passed` that certifies nothing

`"baseline": true` means the run carried **no gates at all**. Running
thresholdless to gather baselines has always worked; what did not work was
telling the two apart afterwards, because a baseline sweep and an acceptance run
both deliver `phase: Passed`. A consumer gating admission on "the last run
passed" would certify hardware against a run that gated nothing.

It is a statement about what the verdict is *worth*, not a hint. The operator
**refuses** a baseline run whose profile carries a threshold — it is a refusal
and not a suppression, because a flag that turned a failing acceptance run into
a passing measurement run would be a threshold-laundering switch. So `true` here
is a reliable claim that nothing was gated.

---

## See also

- [docs/concepts.md](concepts.md) — the CRDs, the scopes, how a run proceeds
- [docs/thresholds.md](thresholds.md) — authoring gates, applicability, the linter
- [docs/reports.md](reports.md) — turning stored envelopes into a report
- [docs/runner-contract.md](runner-contract.md) — where the metrics, the
  `n/a` sentinel and the artifact fences come from
- [docs/dev/invariants.md](dev/invariants.md) — the rules this page describes the
  consequences of
- [`pkg/contract`](../pkg/contract) — the Go types, importable without client-go

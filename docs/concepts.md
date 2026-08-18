# Concepts

How the operator actually works, for somebody who has installed it and now has
to use it.

The shape of the system in one paragraph: a **BurnInTest** is one measurement, a
**BurnInProfile** is an ordered suite of them, and a **BurnInRun** executes a
profile against named nodes. The run resolves the profile *once* into a pinned
plan, cordons nodes a wave at a time, runs one pod per execution, parses the
runner's stdout into metrics, evaluates the profile's thresholds against them,
and settles on a terminal phase which it exports to a **BurnInSink**. A
**BurnInSchedule** does that on a cron, and a **NodeFingerprint** records what
hardware each verdict was actually about.

Related pages: [the runner contract](runner-contract.md) for what an image must
do; [thresholds](thresholds.md) for authoring gates; [sinks](sinks.md) for the
delivery envelope; [reports](reports.md) for the output formats;
[running a soak](soaks.md) for the capacity arithmetic; and
[the project invariants](dev/invariants.md) for the rules none of this may
break.

---

## The six objects

| Kind | Short name | What it is for |
|---|---|---|
| `BurnInTest` | `bit` | One reusable measurement: a kind, a scope, a duration, thresholds, and what the runner pod is allowed to touch. Holds no execution state. |
| `BurnInProfile` | `bip` | An ordered suite of tests, plus `failFast` and the sink list. This is what an operator thinks of as "acceptance" or "quick-smoke". |
| `BurnInRun` | `bir` | One execution of a profile against a target. All execution state, every result, and the verdict live here. |
| `BurnInSchedule` | `bisc` | Creates BurnInRuns on a cron expression, with concurrency and history policy. |
| `BurnInSink` | `bis` | Where a verdict is exported: a webhook, a ConfigMap, or Prometheus. |
| `NodeFingerprint` | `nfp` | The captured hardware identity of a node, and the drift signal against it. Created by the operator, not by you. |

### BurnInTest

The unit of measurement. `spec.kind` selects the default runner image and the
result parser; `spec.scope` says whether the measurement is about a part, a link
or a collective; `spec.thresholds` are the gates.

Two fields deserve reading before you use them. `spec.runner.hostPaths` is the
**only** way a runner pod reaches the node's filesystem — a test that declares
nothing gets a pod with no volumes at all, and `privileged: true` is not a
substitute (measured on this project's own nodes, a privileged `hostNetwork` pod
was missing every `/dev/infiniband/uverbs*` device the host had, which is
exactly what `ibv_create_cq` opens). And `spec.repeatCount` is an **AND**, not a
best-of: every execution must pass, because the faults that matter in burn-in
are the ones that do not reproduce on the first try.

A BurnInTest's status is deliberately almost empty. The definition is reusable;
execution state belongs to the run.

### BurnInProfile

An ordered list of entries, each of which either names a BurnInTest
(`testRef`) or inlines a spec (`inline`). `required` defaults to true and is
what decides whether an entry's failure fails the whole run — informational
tests can be marked `required: false`.

Order is honoured. Tests execute **one at a time, in plan order**: the sweep
breaks out of the plan as soon as one test has work in flight, so the next test
does not start on a node while this one still owes it a result there. Two tests
overlapping on one node would also break the concurrency interlock's meaning,
since the cap counts nodes rather than pods.

`failFast` stops scheduling further tests once a *required* test has failed.
The tests that never ran are simply absent from `status.results`, and that
absence is the record that they did not run.

### BurnInRun

One execution. This is where every knob with a facility consequence lives:
`maxConcurrentNodes`, `force`, `retryOnErrorLimit`, `deadlineSeconds`,
`suspend`, `cancel`, `baseline`, `ttlSecondsAfterFinished`.

`spec.target` selects the nodes, by `nodeNames` or by `nodeSelector`, and
carries `tolerations` that are applied to every test pod so it can land on
tainted accelerator nodes.

`spec.baseline` is worth calling out because of what it *refuses* to do. It
declares that a run MEASURES rather than certifies, and it rides into every
delivered envelope so a consumer gating admission on "the last run passed"
cannot certify hardware against a run that gated nothing. It does **not**
suppress thresholds: a baseline run whose profile carries a gate is refused at
start, naming the test. Suppression would be a threshold-laundering switch — a
way to turn a failing acceptance run into a passing measurement run by flipping
a boolean.

### BurnInSchedule

Acceptance at install time only proves the hardware was good on the day it
arrived. Fans clog, thermal paste degrades, PD supplies and cables age, links
start retraining — and the clock-wedge failure mode in particular is invisible
to every liveness check a cluster already runs, so nothing but a scheduled
re-test will ever find it.

Two deliberate departures from `CronJob`:

- **There is no `Replace` concurrency policy.** Replace means "kill the
  in-flight one and start fresh", which for burn-in destroys expensive evidence
  (a soak four hours in has already cost the fleet those four hours) and kills a
  run mid-cordon, which is the moment cordon cleanup is most likely to be
  interrupted. The choices are `Forbid` (the default) and `Allow`.
- **`failedRunsHistoryLimit` defaults higher than the successful one** (10
  versus 3). The successful runs are the boring ones; diagnosing degradation
  means comparing a failure against the ones before it, and that comparison is
  impossible if the evidence was garbage-collected down to the latest failure.

### BurnInSink

The only integration seam with an external control plane. The operator never
imports its consumer; it posts a versioned envelope. Three types — `Webhook`,
`ConfigMap`, `Prometheus`. See [sinks](sinks.md).

### NodeFingerprint

Created and maintained by the operator's own reconciler, one per Node. A
BurnInRun says "this node passed"; the fingerprint says what "this node" was
made of when it did — GPUs, fabric-relevant NICs, OS image, kernel, a
hardware-identity subset of the node's labels, and a digest over the salient
parts.

Two properties are load-bearing:

- **Drift is reported, never acted on.** The reconciler writes a `Drifted`
  condition and an Event and stops. Launching runs on drift is policy, and
  policy that saturates accelerators across a fleet is not something a hash
  comparison gets to trigger on its own.
- **The `Drifted` condition is sticky.** It is not cleared when a later
  reconcile finds the digest stable, because the fingerprint is updated to
  describe the node as it is *now* — so one pass after the change the new state
  IS the baseline, and a self-clearing condition would erase the only record
  that anything happened, usually within seconds and always before a human saw
  it.

The digest is written as `<scheme>:<hex>`. Digests of different schemes are not
comparable and an incomparable pair is never reported as drift, so changing what
the operator considers salient does not fire a false alarm on every node at
once.

Note that `BurnInRun.status.fingerprint` is a separate, simpler capture taken
directly from the target Node objects at run start (kernel, OS image,
architecture, plus accelerator identity labels where present). It travels with
the verdict; nothing in the reconciler branches on it.

---

## The three scopes, and what a verdict MEANS at each

| | `Node` | `Pair` | `Group` |
|---|---|---|---|
| Executions per test | one per target node | exactly **one**, over both | exactly **one**, over all N |
| Target rule | any number | exactly 2, distinct | ≥ 2, all distinct |
| `TestResult`s produced | one per node | one, naming **both** nodes | one, naming **every** node |
| Slots it costs | 1 | **2** | **N** |
| The verdict is about | the part | the **link** | the **collective** |
| Rendezvous env | none | `BURNIN_ROLE`, `BURNIN_PEER_HOST`, `BURNIN_PEER_NODE` | `BURNIN_RANK`, `BURNIN_NRANKS`, `BURNIN_ROOT_HOST`, `BURNIN_ROOT_NODE` |

Every scope additionally gets `BURNIN_DURATION_SECONDS`, `BURNIN_ATTEMPT`, and
one `BURNIN_VARIANT_<AXIS>` per variant axis.

**Never split a Pair or Group result per node.** A point-to-point measurement is
a property of the link, and attributing it to one endpoint sends an engineer to
replace the wrong part. On a collective it is worse: every healthy rank blocks
waiting for the faulty one and would report the same timeout, so N per-node
results would be N claims no single node can support.

At Pair scope the **client is the deciding side** — it is where perftest and
nccl-tests report — and verdict precedence across the two ends is
`Error > Fail > Skip > Pass`. A machinery failure on either end means the link
was never measured, so a client's "connection refused" is an artifact of its
peer rather than evidence about the fabric; recording that as `Fail` would
permanently indict a link over an unpullable image. A **server that exits 0
before the client ever started is an `Error`**, never a pass: no traffic crossed
the link.

At Group scope, target *i* is rank *i*, pinned by hostname. Rank 0 is the root:
it starts first and no worker is created until it reports Ready. `BURNIN_ROLE`
is deliberately **absent**, so a runner keying off server/client fails loudly
rather than treating rank 4 as a client. A rank that never reported is not a
pass — a collective is only measured if every rank took part.

Group scope needs an image that speaks the N-rank rendezvous. A test at Group
scope that names no `spec.runner.image` and whose kind's shipped runner does not
speak it is **refused at plan time**. The reason is that the failure it prevents
is silent: every fabric runner branches on `BURNIN_ROLE` and reads its absence
as a Node-scope run, so it would exit 2 with a skip marker, the run would record
"acceptance does not apply to this hardware", and it would settle `Passed`
around a collective that never executed. Published tags are immutable, so images
already in the field will behave that way forever; the operator declining to
dispatch them is the only thing that protects a fleet running those tags. Today
`nccl` is the one kind whose default image speaks the contract. `ib-write-bw`
and `gpudirect-rdma` refuse Group scope and always will — a point-to-point RDMA
write and a GPUDirect peer-memory check have no N-rank form.

A scope this operator version does not recognise (the enum is open at the API
level) is recorded as a terminal `Error` for that test, never skipped: a
required acceptance test the operator cannot run must never let hardware pass by
omission.

---

## Phases

`RunPhase` is used both for the run and for each `TestResult`.

| Phase | Terminal | Meaning |
|---|---|---|
| `Pending` | no | The run has not started. |
| `Running` | no | Executing, or draining a graceful cancel. |
| `Passed` | yes | Every required test produced a verdict and met it. |
| `Failed` | yes | A required test was **measured** and fell short. |
| `Error` | yes | The machinery malfunctioned or refused; the hardware is **unjudged**. |
| `Skipped` | yes | *TestResult only.* The runner declared the test does not apply to this hardware. A run never takes this phase. |
| `Cancelled` | yes | A human or a control plane stopped it on purpose. Not a verdict. |

**`Error` is not `Fail`, and the difference has teeth.** `retryOnErrorLimit`
re-runs an `Error` and nothing else, so a `Fail` settles the test where it
happened with the retry budget unspent. Re-running a measurement until it comes
out clean launders a hardware fault into an acceptance, and marginal hardware is
precisely the hardware that passes on the second try.

`Cancelled` is kept distinct from `Error` for the opposite reason: `Error` says
our machinery broke, `Cancelled` says it did exactly what it was told. A fleet
operator has to be able to tell "I stopped this" from "this stopped itself".

A refused run — admission conflict, unresolvable profile, malformed threshold —
is terminal with phase `Error` and does **not** get a phase of its own. A new
terminal phase would have to be understood by every consumer of the delivery
envelope before any of them could tell it from a pass, and a consumer that does
not recognise a phase is a consumer that might treat it as one. The cause is
made machine-readable by a condition instead.

### How the run's phase is derived

Precedence, over **required** tests only: `Failed` beats `Error` beats `Passed`.
A required failure is a hardware verdict and wins outright; a required error
leaves the hardware unjudged, which must not be called `Passed`. Skips count
towards neither — a test the hardware cannot take is not evidence either way.

`status.passed`, `status.failed`, `status.errored` and `status.skipped` are four
separate counters and none may be folded into another. A summary showing
"0 failed" while eight tests never ran reads as a clean sweep of hardware
nothing actually measured.

---

## How a run proceeds

### 1. Resolve

The profile is fetched, each `testRef` is fetched, `inline` entries are taken as
written, and variants are expanded (below). The target selector is resolved:
`nodeNames` if given, otherwise `nodeSelector` against the node list.

A resolution failure is classified before it is acted on. Transient failures
(API throttling, a cold cache) are retried with backoff. A `NotFound` on a young
run waits out a two-minute grace period, because `kubectl apply -f dir/` creates
the run and its profile together and the run's watch event can arrive first.
Only a confirmed-permanent misconfiguration finalises the run as `Error`, with a
synthetic result named `resolve`.

### 2. Admit

Between resolving the targets and pinning the plan — **the last moment at which
the run still holds nothing: no finalizer, no cordon, no pod** — the operator
checks whether any other active run is already holding one of these nodes.

This answers the question `maxConcurrentNodes` cannot. That cap bounds one run's
fan-out; nothing about it bounds how many *runs* drive the same node, and two
runs each honouring a cap of 1 on one machine are two full-power soaks on one
machine, each believing it is compliant. Observed on a two-node GB10 cluster:
three BurnInRuns created seconds apart against the same two nodes, every one of
them failing thermal-soak with exit 137 — the kubelet killing sustained
full-load soaks that were competing for one machine, reported as `Error` with
nothing to say the cause was another run.

The check **refuses rather than queues**. A burn-in is scheduled maintenance
against hardware somebody is waiting on, and a run that silently waits behind
another has an unpredictable end time, which is exactly what a maintenance
window cannot tolerate. `spec.force` admits anyway and is recorded as
`Admitted=True` with reason `AdmittedByForce`, so a verdict produced under
deliberate contention is identifiable as such months later.

### 3. Plan and lint

`buildPlan` materialises the profile against the resolved targets and refuses
anything the rest of the controller depends on being true:

- duplicate test names (names are result identity — the second would never run
  and its `required` flag could be silently overwritten by the first's);
- Pair topology (exactly two distinct targets; `maxConcurrentNodes` ≥ 2);
- Group topology (≥ 2 distinct targets; `maxConcurrentNodes` ≥ N; a
  Group-capable image);
- host mounts that would produce a pod the apiserver rejects;
- a baseline run whose profile carries thresholds;
- **malformed thresholds** — gates nothing could ever satisfy.

Every one of these refuses at **start**, as a config `Error`, before a node is
cordoned. A malformed gate would otherwise fail on every node forever and be
reported in the shape of a hardware verdict, which is the exact confusion
`Error` is kept distinct from `Fail` to prevent.

*Unsound* thresholds — gates that do evaluate but may not mean what their author
intended — do not block. They are recorded on the `ThresholdsSound` condition
and the run proceeds; refusing on suspicion would let this operator veto
profiles it merely does not understand. See [thresholds](thresholds.md).

### 4. Pin

The resolved plan is serialised into the annotation
`burnin.glimmer.ai/plan`, **in the same write that adds the cordon-cleanup
finalizer**. It must never be possible to hold a node without holding the
finalizer that guarantees its release.

### 5. Mark running

`status.priorUnschedulable` is captured here, **once, before the first cordon,
and never overwritten** — see [the cordon rules](#cordoning-follows-the-wave)
below. The fingerprint is captured, the `ThresholdsSound` and (if variants
expanded anything) `PlanExpanded` conditions are set, and a `Running` envelope
is delivered *before* the status write, so a crash in between redelivers with
the same derived delivery ID rather than dropping the transition entirely.

### 6. The execution loop

Each pass sweeps the plan in order and, for the first test with outstanding
work:

1. **Compute the busy set** — every node with a live pod belonging to this run.
2. **Admit under the cap.** A Node execution needs one free slot, a Pair needs
   two, a Group needs N. If they are not free, the execution is *pending* and
   nothing is launched.
3. **Cordon the wave.** The node (or both nodes of a pair, or every node of a
   group) is cordoned *before any pod exists*, and the ownership stamp goes on
   before the cordon.
4. **Create the pod(s).** At Pair scope the server goes first and the client is
   not created until the server pod reports `Ready`; at Group scope rank 0 goes
   first and the workers are created together once it is Ready.
5. **Wait.** Pod events drive the common case. The backstop poll is 30 s while
   any in-flight pod has not started — that is where scheduling problems appear
   — and 5 minutes once everything is running, because a run doing exactly what
   it was asked to do teaches nothing by being asked again in thirty seconds.
   A pair or group between its pods polls at 5 s, since that window is dead time
   on hardware already held.
6. **Checkpoint**, if `checkpointIntervalSeconds` is set: read the running pod's
   stdout so far and refresh the result's metrics. A checkpoint is **evidence,
   never a verdict** — no exit code is consulted and no threshold is applied.
   Reading a live log is safe because the parser is last-occurrence-wins, so a
   truncated log is a valid earlier snapshot rather than a corrupt record.
7. **Harvest.** On termination the pod's exit code and stdout become a parsed
   result, thresholds are evaluated once against the completed execution, the
   attempt is recorded, and artifacts are lifted into a run-owned ConfigMap.
8. **Release.** Any node this run holds that is no longer in the wave is
   uncordoned immediately, not at teardown.

### 7. Settle

When every planned execution has a terminal result, `terminate` runs in this
order, and the order is a bug fix that must not be undone:

1. write the terminal phase and mark any still-open execution `Error`;
2. **release every cordon** (idempotent and level-based, so it is safe early);
3. deliver the terminal envelope;
4. **write status**;
5. **delete live pods**;
6. remove the finalizer;
7. hand over to the TTL.

Killing the pods before the status write was the defect. When either write lost
a `resourceVersion` race the reconcile returned an error, the terminal status
was discarded, and the run went back to believing it was `Running` with an open
execution — then found the pod *it* had just killed, harvested the SIGKILL, and
recorded the hardware as `Error: runner terminated abnormally (exit 137)`. The
operator condemning a healthy node for its own act, 30 s into a 100 s soak. With
the write first, a lost race costs one wasted pass and nothing else, because
nothing has been destroyed.

The finalizer comes off last because it is the standing record that the run
still owes the cluster something. A terminal run that still carries it is swept
again on every pass, so a failed pod deletion does not leave a runner burning a
node behind a finished run.

---

## Cordoning follows the wave

A node is cordoned immediately before the run puts load on it and released once
it is no longer holding any, so a run's footprint on the fleet tracks
`maxConcurrentNodes` instead of the size of its target list. Cordoning every
target up front took a two-node cluster entirely out of scheduling and hollowed
out the interlock, whose whole premise is that only N nodes are occupied at
once.

The trade, stated plainly: a node the run will come back to for a later test is
released in between, so there is a one-pass window in which the scheduler could
place something on it. That is the price of not holding a fleet hostage to a
long sweep. The cordon always goes on before the pod is created, so nothing
lands beside a burn-in already running.

Three rules keep this safe, and each exists because of a specific way it went
wrong:

- **A run only uncordons what it stamped.** The node carries
  `burnin.glimmer.ai/cordon-owner` = `<namespace>/<name>/<uid>`. The UID is
  there because names are reusable: a deleted-and-recreated run of the same name
  is a different run and must not inherit authority over the previous
  incarnation's cordon.
- **Release restores, it does not uncordon.** A node that was already
  unschedulable when the run found it was somebody's deliberate act — a drain
  for a firmware update, most likely — and returning it to service because a
  burn-in finished is how a node under maintenance receives production traffic
  mid-update.
- **What the run FOUND is captured once, at start** (`status.priorUnschedulable`),
  and is what the release restores. Re-deriving it per wave read the run's own
  footprint back: three overlapping runs each observed the first run's cordon,
  recorded it as pre-existing, and re-asserted it at teardown after its real
  owner had let go — and the last one to release also cleared the annotations,
  leaving the node unschedulable with **no stamp at all** and nothing in the
  cluster that knew it was held. Half a two-node fleet, gone, silently.

The deliberate consequence: a cordon placed on a target *after* the run started
is not adopted, and is undone at teardown. That is the right way round — a
cordon undone is visible and one command to redo, and a node the fleet silently
loses has nothing left in the cluster that knows it was taken.

Runner pods carry a toleration for `node.kubernetes.io/unschedulable:NoSchedule`
unconditionally, scoped to that one taint by key. Without it the run cordons its
target and then waits forever for a pod the scheduler will never place — a
deadlock the operator builds for itself.

---

## Declaring the heat

A node held for a test that **holds the part under load** also carries
`burnin.glimmer.ai/heat-expected` = `<namespace>/<name>/<uid>/<RFC3339 expiry>`,
written, refreshed and removed in the same updates as the cordon stamp and under
the same ownership rule.

Which kinds those are is the kind's own answer — `TestKind.DrivesSustainedLoad`,
beside `BurstOnly`, because the fact belongs to the runner and a controller
deciding it from a name would go stale the moment a runner changed. Today:
`thermal-soak`, `gpu-burn`, `clockprobe`, `fabric-soak`, `memory-stress` and
`dcgm-diag` (whose levels 3 and 4 run `targeted_stress` for ~15 minutes, and the
level is a runner env var this operator must not read, so it is declared by its
worst case). A `host-health` read, a `compute-smoke` GEMM or a `memory-bw`
measurement declares nothing — claiming heat it does not produce would suppress
a real thermal response on a node nobody is loading.

`custom` declares nothing either. The whole point of it is an image this project
knows nothing about, and this answer suppresses a safety action on somebody
else's behalf; a site running a custom soak asserts the hold on the node itself.

A run holds a node across its tests, so the declaration is **withdrawn** when
the same node passes from a soak to a passive read without being released in
between. Expiry would clear it eventually, but eventually is a whole pod window
of suppressed thermal response bought by a test that never asked for it.

It exists because two safety systems were fighting each other. The Glimmer agent
runs a thermal watchdog that drains a node when the part passes its trip point,
and a thermal soak drives the part past that point **by design** — so every soak
on a fleet running both was drained by the watchdog, its runner pods killed with
SIGKILL, and the hardware recorded as `Error — hardware unjudged`. The watchdog
was not wrong; it had no way to tell a soak from a cooling failure. This
annotation is that way. The operator declares the heat; the reader gates its
drain on the declaration.

**The value carries an absolute expiry, and that is what makes it safe.** The
reader is on the node and knows nothing about whether the run is still alive. A
bare marker left by an operator that crashed, lost its lease or was deleted
mid-flight would disable thermal protection on that node indefinitely, silently,
until the part cooked. The expiry bounds that to one pod's window: a reader must
refuse a declaration whose expiry has passed, so a marker nothing ever removes
stops counting on its own.

The window is derived from the run's own per-pod duration — one **segment** for
a segmented soak — plus the grace the operator already allows that pod and one
reconcile pass to take the marker off. A sixty-second smoke test and a seven-day
soak therefore get very different windows, and a run that outlives one pod
re-stamps a later expiry as it goes. The expiry only ever moves forward.

Two rules mirror the cordon's exactly, and for the same reasons: a declaration
naming a **different** run is that run's thermal protection and is never removed
(authority is by UID), and one this operator **cannot parse** is left as found —
"I cannot read this" is not evidence that the heat is over.

The key is a cross-repository contract. A mismatch between the writer here and
the reader on the node fails **open** — the drain still happens, the pod still
dies — and looks exactly like the bug it fixes.

---

## The concurrency interlock

**`maxConcurrentNodes` counts NODES, and it is a facility safety interlock, not
a performance knob.** Burn-in deliberately drives hardware to its power and
thermal limits. Raising this multiplies real electrical load and real heat in a
real room: ten nodes at full power draw is a rack-level power event, and either
that or the cooling load can trip a breaker or push inlet temperatures past what
the room can carry — landing the consequences on hardware that was never under
test. There is a correctness reason on top of the facility one: a node measured
beside nine others under full load is not measured under the same conditions as
one measured alone.

The default is 1, and it is deliberately inconvenient. It must never be raised
automatically, defaulted higher "for throughput", or inferred from cluster size.

| Scope | Slots | When it is checked |
|---|---|---|
| `Node` | 1 | at each pod launch |
| `Pair` | **2** | **once**, when the unit starts |
| `Group` | **N** | once, when the unit starts |

**A Pair costs two slots and holds both nodes for the whole test**, so it is
admitted only when two are free — and never at the default cap of 1, which the
run refuses at plan time with that explanation rather than hanging until its
deadline. The client is **not** re-checked against the cap when it is created
later: once the server is up the load is committed, and refusing the second half
would leave the server burning its clock against a peer that will never arrive,
reporting an `Error` about hardware that is perfectly fine. If load has to come
off the floor right now, `spec.cancel` with `CancelImmediate` is the tool built
for that.

**A Group needs `maxConcurrentNodes` ≥ its target count**, refused at start
otherwise, for the same reason multiplied: a group is one indivisible unit that
holds every one of its nodes for the whole test.

The cap is read **live** from the spec on every pass rather than pinned into the
plan. It is not part of what a verdict describes — it is a standing instruction
about the room — and an operator lowering it during a power event needs it to
take effect on the next node the run would have started, not on the next run.

The wall-clock arithmetic this implies is in [running a soak](soaks.md).

---

## Repeats versus retries

Both re-run a test; they are opposites and must never be confused.

| | `spec.repeatCount` (on the test) | `spec.retryOnErrorLimit` (on the run) |
|---|---|---|
| Re-runs a test that | already produced a verdict | produced **no** verdict at all |
| Effect on the gate | makes it **stricter** | restores a measurement that never happened |
| Which phases | every execution must `Pass` | **`Error` only** |
| Can turn `Passed` into `Failed`? | yes — that is the point | no |
| Can turn `Failed` into `Passed`? | **never** | **never** |
| Recorded as | `Trigger: Repeat` | `Trigger: ErrorRetry` |

Repeats are the intermittent-fault gate: a marginal solder joint, a link that
retrains under thermal cycling, a DIMM that miscompares once an hour. A single
clean pass does not distinguish a good part from one that fails one run in five.
They are sequential, never parallel — the point is to cycle the hardware, and
two concurrent copies on one node would contend for the very resource under
test.

Retries exist because an image pull failure, an eviction, a node disappearing or
a runner crash all mean the machinery malfunctioned and the hardware was never
judged. A `Fail` — whether from a non-zero exit or from a threshold violation on
a clean exit — settles the test where it happened, with retry budget left
unspent. `Skipped` is likewise never retried: a test that does not apply to this
hardware will not start applying on the next attempt.

An errored attempt measured nothing, so it does **not** consume a repeat:
`errorRetries` is tracked separately from `repeatsCompleted` on the result.

Every attempt is recorded individually in `status.results[].attempts`, with its
own exit code, metrics, pod name and trigger. A node that passed first time and
a node that passed on the fourth try after three errors are both `Passed` at the
test level and are very different parts — and the `Trigger` field is what makes
the retry rule auditable from a stored result long after the run.

---

## The pinned plan

At start the fully-resolved plan is written into the run's
`burnin.glimmer.ai/plan` annotation and **everything downstream reads that copy,
never a live re-read of the profile or the cluster**: pod identity, result
bookkeeping, which tests count as required, which nodes results are owed for,
sink routing, and the threshold linter's advisory.

What it pins:

| Pinned | Why |
|---|---|
| the resolved target list | a node dropping out of a label selector mid-run still owes its results |
| every test's full spec, in order | editing a BurnInTest mid-run cannot widen what an in-flight attempt takes off the host, or change how long it runs |
| each entry's `required` flag | a required failure cannot be forgotten by an edit |
| the sink list and `failFast` | a verdict is delivered where the run was launched to deliver it |
| `baseline` | a consumer reading a stored run months later cannot be misled by a spec since edited |
| variant axes and the parent entry | a report groups a sweep's cells without parsing names back out of strings |

A run must be hermetic. Re-resolving on every pass allowed all of the above to
silently rewrite history, up to and including a required failure being forgotten
because its node left the selector.

Mechanics worth knowing: annotations share a 256 KiB budget per object, so the
plan is capped at 128 KiB. A small plan is stored as raw JSON so `kubectl get -o
yaml` shows something a human can read; only a plan that would not otherwise fit
is gzipped and base64-encoded, which works well because variant cells are nearly
identical copies. The format is sniffed rather than versioned, so a plan pinned
by an older controller is read unchanged — the reverse (a plan pinned by a newer
controller and read by an older one) fails to parse and finalises the run as
`Error`, which is a downgrade hazard worth writing down but is the right way
round to fail.

`maxConcurrentNodes`, `retryOnErrorLimit`, `deadlineSeconds`, `suspend` and
`cancel` are deliberately **not** pinned; they are read live.

---

## Variants: one test, many cells

`compute-smoke` runs an NVFP4 GEMM. That is one cell of a matrix: the same
kernel at FP8, BF16, TF32 and FP64 answers different questions about the same
silicon, and before variants the only way to ask all five was to author five
BurnInTest objects differing by one environment variable, each with its own
name, its own thresholds, and its own copy of every other field. The same shape
recurs on every axis worth sweeping — message size for a fabric test, buffer
size for a bandwidth test, a duration class for a soak.

A variant lives on the **profile entry**, not on the test:

```yaml
tests:
  - testRef: fp4-compute-smoke
    variants:
      - name: fp4
        axes: {precision: fp4}
      - name: fp8
        axes: {precision: fp8}
```

- Each cell is planned as an ordinary execution named `<test>-<name>`. **The
  name is result identity**, so two variants of one test may not share one; the
  run refuses at plan time if they do, by the same check that refuses two tests
  with the same name.
- The overlays are `durationSeconds`, `args`, `env`, `thresholds` and
  `repeatCount`. Nil **inherits**; non-nil **replaces**.
- **`thresholds` replace, they do not merge.** An FP4 throughput floor is not an
  edited FP8 floor. Merging by metric name would silently retain a gate the
  author believed they had replaced — and a node failed against a threshold
  nobody can find in the profile is the worst kind of verdict this project can
  produce, because there is nothing to read that explains it. A cell that means
  to apply *no* thresholds sets an empty (non-nil) list.
- **`axes` are opaque labels.** Each becomes `BURNIN_VARIANT_<AXIS>` in the
  runner's environment, upper-cased, and is echoed onto `TestResult.variantAxes`
  and into the envelope. The controller never interprets them: a precision means
  something to a runner and nothing to a reconciler, and a controller that
  learned to branch on `precision: fp4` would be the vendor branch this project
  refuses, wearing a different hat.

Expansion happens once, before the plan is built, so every stage downstream —
the duplicate-name check, topology validation, pod naming, result identity,
verdict, delivery keys — works unchanged because none of them ever learns
variants exist. A profile with no variants plans identically to one written
before variants existed.

**The arithmetic is invisible in the profile, so the run states it.** One entry
with four variants across eight nodes at a cap of 1 is thirty-two sequential
node-executions. The `PlanExpanded` condition says so at start — a condition
rather than an Event, because an Event expires in an hour and a multi-day sweep
does not.

---

## Conditions on a run

| Condition | Set when | What it tells you |
|---|---|---|
| `Admitted` | always, at start | `TargetsAvailable` (free), `AdmittedByForce` (contended, admitted anyway — results are not comparable to results measured alone), or `TargetsBusy` on a refusal, naming the run that holds the node |
| `PlanExpanded` | only when variants expanded something | how many executions the plan holds, and the node-execution count |
| `ThresholdsSound` | always, at start, against the **pinned** plan | `Reviewed` when the linter found nothing to say; `UnsoundThresholds` with the offending gates named. **Advisory** — it never changes the phase |

`ThresholdsSound=True` is not a claim that the thresholds are correctly
*calibrated*. No linter can judge that, and the gate this project got most wrong
was one it would have passed clean.

---

## Stopping a run

| Mechanism | Terminal | What happens to work in flight |
|---|---|---|
| `spec.suspend` | **no** — reversible | Nothing new launches; in-flight executions finish and record. The run holds its cordons and reaches no terminal phase. Clearing the flag resumes it with its results intact. |
| `spec.cancel` + `Graceful` (default) | yes → `Cancelled` | Nothing new launches; in-flight executions are allowed to finish and record. |
| `spec.cancel` + `Immediate` | yes → `Cancelled` | In-flight pods are deleted now. Terminated-but-unharvested pods are still read first — "immediate" is about not waiting, not about refusing to read. |
| `spec.deadlineSeconds` | yes → **`Error`** | Whatever completed keeps its real results; the rest are settled unjudged. |

**Cancel is one-way.** Once the controller has observed it and begun cancelling,
clearing the flag does not resurrect the run — a run that could un-cancel would
race its own cleanup and could resume after its cordons had been released,
putting test load back onto nodes the scheduler has already been given back. To
run again, create a new BurnInRun, which is also what keeps the audit trail
honest.

Cancel dominates suspend, because a run that is both suspended and cancelled and
could not be cancelled would be unstoppable without deleting the object.

A deadline drives the run to `Error` and deliberately not to `Failed` (a run
that ran out of time did not judge the hardware it never reached) and not to
`Cancelled` (nobody asked — `Cancelled` is reserved for an explicit request, so
a consumer can tell an operator's decision from a bound being hit).
`spec.cancelReason` is free-form text carried into the envelope: a cancellation
with no reason is the one result nobody can act on.

Deleting the run is also safe — the cordon-cleanup finalizer holds deletion open
until every node it took out of the scheduler is given back. `kubectl delete
burninrun` is the most natural reaction to a run behaving badly and therefore
the likeliest moment for a fleet to lose capacity silently, which is exactly
what the finalizer prevents.

---

## A worked walkthrough

Three nodes, `spark-a`, `spark-b`, `spark-c`. A profile with two tests. Cap of
1.

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: gemm
spec:
  kind: compute-smoke        # burst-only: it ignores durationSeconds as a budget
  scope: Node
  thresholds:
    - metric: nonfiniteCount
      comparison: Equal
      value: "0"
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: soak
spec:
  kind: thermal-soak
  scope: Node
  durationSeconds: 900
  checkpointIntervalSeconds: 60
  repeatCount: 2
  thresholds:
    - metric: throttleEvents
      comparison: Equal
      value: "0"
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInProfile
metadata:
  name: acceptance
spec:
  failFast: false
  tests:
    - testRef: gemm
    - testRef: soak
      required: false
  sinks: [glimmer]
---
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInRun
metadata:
  name: acceptance-2026-08
spec:
  profileRef: acceptance
  target:
    nodeNames: [spark-a, spark-b, spark-c]
  maxConcurrentNodes: 1
  retryOnErrorLimit: 1
  ttlSecondsAfterFinished: 86400
```

*(Both gates above are exact comparisons on dimensionless counters, which is
what `Equal` is for. Continuous measurements take `GreaterThanOrEqual` /
`LessThanOrEqual`, and their numbers should come from a baseline sweep of your
own fleet rather than from a spec sheet — see [thresholds](thresholds.md).)*

**What happens, in order.**

1. **Resolve.** Both BurnInTests are fetched; targets resolve to the three names
   as written. No variants, so the plan holds two executions per node.
2. **Admit.** No other active run holds `spark-a`, `spark-b` or `spark-c`.
   `Admitted=True`, reason `TargetsAvailable`.
3. **Plan.** Names are unique; both tests are Node scope so no topology rule
   applies; the threshold lints clean. The plan is pinned into
   `burnin.glimmer.ai/plan` in the same write that adds the finalizer.
4. **Mark running.** `status.priorUnschedulable` records all three nodes as
   schedulable — **this is the record the release will restore, and it is never
   recomputed**. `status.fingerprint` captures each node's kernel, OS image,
   architecture and accelerator labels. `ThresholdsSound=True`. A `Running`
   envelope goes to the `glimmer` sink.
5. **Wave 1.** `gemm` is the first test with work. `spark-a` is free, the cap is
   1, so: stamp `spark-a` with the cordon owner, cordon it, create the pod
   (`acceptance-2026-08-t0-a1-<hash>` — deterministic, so a crashed controller
   finds the pod it already made instead of starting the test twice, and the
   attempt is part of the identity so a repeat cannot re-harvest its
   predecessor's exit code). `spark-b` and `spark-c` are *pending* and untouched — they
   are still in the scheduler and still carrying production workloads.
6. `gemm` exits 0 in a few seconds with `nonfinite_count=0` on stdout, which the
   parser normalises to `nonfiniteCount`. The threshold is evaluated once
   against the completed execution: pass. A
   `TestCompleted` envelope is delivered keyed `gemm/spark-a`. `spark-a` is no
   longer in the wave, so it is uncordoned **immediately** and handed back.
7. **Waves 2 and 3.** The same for `spark-b`, then `spark-c`. `gemm` now owes
   nothing, so the sweep moves to `soak`.
8. **`soak` on `spark-a`.** `repeatCount: 2`, so `repeatsRequired` is 2. The pod
   runs for 900 s; every 60 s the operator reads its stdout so far and refreshes
   `status.results[].metrics` and delivers a `Checkpoint` envelope. Those numbers
   are evidence — no threshold is applied to them, and a mid-run temperature
   spike is not a failure because the run is not over.
9. Attempt 1 exits 0. `repeatsCompleted` becomes 1, the result stays open, and
   the next pass starts attempt 2 on the same node — sequentially, because two
   concurrent soaks on one node would contend for the resource under test.
10. Attempt 2's pod is killed at minute 12 — a node drain, say — and terminates
    with exit 137. That is outside the runner contract, so it is an `Error`, not
    a `Failed`: the hardware was not judged. The attempt stays in
    `status.results[].attempts` with its raw exit code, so an out-of-contract
    code remains visible as the contract violation it is instead of being
    flattened away. `retryOnErrorLimit: 1` grants attempt 3, recorded with
    `trigger: ErrorRetry`. `repeatsCompleted` stays at 1, because an errored
    attempt measured nothing and still owes its repeat.
11. Attempt 3 passes. `repeatsCompleted` reaches 2, and the result settles
    `Passed` with three attempts on record: `Initial`, `Repeat`, `ErrorRetry`.
    A node that only passes after a retry is visible as such, not
    indistinguishable from one that passed first time.
12. **`soak` on `spark-b`** fails a throttle gate. `soak` is `required: false`,
    so this does not decide the run — but the result is `Failed`, its
    `violations` list names every gate the deciding attempt missed with a
    `cause` on each, and **no retry is granted**: a `Fail` is a measurement.
13. **`soak` on `spark-c`** passes.
14. **Settle.** Every planned execution has a terminal result. No *required*
    test failed or errored, so the run is `Passed` — `status.failed` is 1 and
    says so plainly beside it. Cordons are released (all three nodes back to
    schedulable, matching `priorUnschedulable`), the terminal envelope is
    delivered, status is written, any live pod is deleted, the finalizer comes
    off, and the run waits out its 24-hour TTL with its evidence intact.

**What to read afterwards.** `kubectl get bir` shows the phase and the counters.
`status.results[]` carries each execution's metrics, message, violations,
`notEvaluated` gates and artifact references. `status.conditions` carries the
admission decision and the threshold advisory. See [reports](reports.md) for the
rendered forms.

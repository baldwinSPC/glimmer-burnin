# Running a soak

A soak holds real load on real hardware for hours or days. That makes it the
most useful test in the suite and the most expensive, and the expense is not
compute — it is **fleet capacity, removed from scheduling for the duration**.

This page is about that arithmetic and about surviving the week.

---

## The capacity arithmetic, before you start

The operator cordons a node immediately before it puts load on it and releases
it once it is no longer holding any, so a run's footprint tracks
`maxConcurrentNodes` rather than the size of its target list. That is the good
news. The bad news is the multiplication:

```
wall-clock ≈ ceil(targets / maxConcurrentNodes) × testsPerNode × duration
```

A 24-hour soak across 8 nodes at `maxConcurrentNodes: 2` is **four waves of 24
hours — four days**, with two nodes out of the fleet for the whole of it. At
`maxConcurrentNodes: 1` it is eight days.

Variants multiply the middle term. Four precision cells of a 24-hour soak across
8 nodes at a cap of 2 is sixteen days. The run says so at start — the
`PlanExpanded` condition states the expansion and the node-execution count — but
it is cheaper to know before applying it than after.

**A Pair costs two slots** and holds both nodes for the whole test, so a pair
soak is admitted only when two slots are free, and never at the default cap of 1
(the run refuses at start with that explanation). **A Group needs
`maxConcurrentNodes >= ` its target count**, refused at start otherwise.

### Getting capacity back now

`spec.cancel` with `CancelImmediate` is the tool. It is not graceful and is not
meant to be: it exists for the case where somebody needs the hardware back this
minute. A cancelled run's results are kept and its verdict is `Cancelled`, which
is terminal and is never a hardware finding.

---

## Surviving the week

Over a multi-day run, the cluster's own housekeeping is the most likely thing to
end it — not the hardware, and not the test.

### What the operator does for you

**Every runner pod carries
`cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`, unconditionally.** A
burn-in pod is never safely evictable: it is holding a node the operator cordoned
precisely so nothing else lands there, and moving it discards a measurement in
progress — which is then recorded as an `Error`, spending retry budget on a fault
the cluster caused rather than the hardware. There is no knob, because there is
no case where the honest answer is "yes, move this".

**Runner pods tolerate the operator's own cordon** and nothing else. The
toleration is scoped to `node.kubernetes.io/unschedulable:NoSchedule` by key; it
is emphatically not a blanket toleration.

### What you have to do

**Set `spec.runner.priorityClassName` on contended hardware.** The annotation
handles consolidation; it does nothing about preemption, which is a different
mechanism and needs a different answer. Point it at a PriorityClass your cluster
already manages — the operator never creates one, and its RBAC deliberately does
not permit it. A cluster-scoped object minted by a test-runner is a footgun.

**Drains still win.** A `kubectl drain` evicts the pod regardless. If a
maintenance window overlaps a soak, the soak loses — but segmenting decides
whether it loses fifteen minutes or a week.

---

## Segmenting: an eviction should cost a segment, not the week

Without `spec.soak`, a seven-day soak is **one pod with
`activeDeadlineSeconds: 604920`**. Everything that normally happens to a pod over
a week is fatal to that: a kubelet restart, a drain, an eviction, an image-GC
pass, a reboot. Each ends the run as an `Error`, and with `retryOnErrorLimit` set
the retry starts the week over from zero.

```yaml
spec:
  kind: thermal-soak
  durationSeconds: 604800     # seven days
  soak:
    segmentSeconds: 900       # fifteen-minute windows; minimum 300
  thresholds:
    # elapsedS SUMS across segments, so this gates whether the soak soaked.
    - metric: elapsedS
      comparison: GreaterThanOrEqual
      value: "574560"         # 0.95 × 604800
    - metric: xidEvents
      comparison: Equal
      value: "0"
```

Each segment is its own pod, with `BURNIN_DURATION_SECONDS` and
`activeDeadlineSeconds` sized to the segment. An eviction costs that one segment;
the operator relaunches it and the soak carries on.

**It is opt-in per test, and it has to be.** Whether a duration means anything is
the *runner's* property: `host-health` clamps its window to 30 seconds and
`compute-smoke` is declared burst-only, so segmenting either would run the same
short measurement over and over and report it as a week. A burst kind is refused
at plan time; the rest is your judgement about your runner.

### The verdict is rendered once, over the aggregate

This is the part to understand before writing a threshold. **Per segment only the
exit code counts**: `0` folds that window's metrics into
`status.results[].aggregatedMetrics` and the test stays open, `1` settles the test
`Failed` on the spot, `2` settles it `Skipped`, and anything else is an `Error`
that spends `retryOnErrorLimit`, re-runs *the same segment index*, and contributes
nothing. **Thresholds are applied once, on the last segment, to the aggregate.**

How each metric combines is declared beside its name in `pkg/contract`, not by
the profile:

| Rule | Meaning | Examples |
|---|---|---|
| `Sum` | a per-window count | `elapsedS`, `xidEvents`, `pcieReplayErrors`, `throttleEvents` |
| `Min` | a floor: the WORST window is the verdict | `sustainedClockPct`, `busBandwidthGBs` |
| `Max` | a ceiling: the peak across the whole soak | `gpuTempC`, `powerDrawW`, `p99LatencyUs` |
| `Last` | a nameplate, or a lifetime total the runner already reports absolutely | `eccErrors`, `remappedRows`, `driverVersion` |

The `Last` row is the one that bites. `eccErrors` and `remappedRows` are NVML
aggregates *since reset*: summing them would multiply them by the number of
windows, and a node with four remapped rows would read as 1,152 after a
288-segment soak. An **unregistered** metric combines `Last`, because inventing
`Sum` semantics for somebody else's measurement would be this operator deciding
what it means.

### `abortEarly`

`soak.abortEarly: true` ends the soak `Failed` the moment the aggregate
**provably** violates a gate — provably meaning no later segment could retract it,
which is true in exactly four places:

- a `Sum` under `LessThanOrEqual` (a count only grows);
- a `Sum` under `Equal`, once the sum has passed the value;
- a `Min` under a `GreaterThanOrEqual` floor (a minimum only falls);
- a `Max` under a `LessThanOrEqual` ceiling (a maximum only rises).

Everywhere else it stays silent. A `Min` under a *ceiling* can be pulled back
under it by the next window, and ending a week of burn-in on a reading that
fifteen minutes would have overturned is the wrong trade in both directions. It
is evaluated at segment boundaries only, from harvested metrics — never from a
checkpoint's parse of a bounded log tail — and a metric no segment has reported
yet never aborts anything.

### Reading a segmented soak

`status.results[]` gains `segmentsRequired`, `segmentsCompleted`,
`aggregatedMetrics` and `truncatedAttempts`. `segmentsCompleted` counts only
*clean* segments, so it does not advance for an interruption. Attempts are capped
— the first, every non-passing one, and the last 16 passing — because 288 attempt
records with a metrics map each is a status nobody can write; `truncatedAttempts`
says how many uneventful windows are not shown. That is safe precisely because
the verdict is read from `aggregatedMetrics`, which is persisted, and never from
the attempt list.

Every segment is a **full** window, including the last, so
`segmentsRequired = ceil(durationSeconds / segmentSeconds)` and a soak may overrun
its duration by less than one segment. A short remainder segment would be below
the floor and — worse — a counter summed over a ten-second window is not
comparable to the same counter summed over a fifteen-minute one.

---

## Reading a soak that ended badly

The distinction that matters most is the one the whole operator is built around:

| | |
|---|---|
| **`Failed`** | the hardware was measured and fell short. **Never retried** — re-running a measurement until it comes out clean launders a fault into an acceptance. |
| **`Error`** | the measurement did not happen. The hardware is **unjudged**, not condemned. This is the retryable phase, and it is where an eviction, a drain, a reboot or an image-pull failure lands. |

A soak that ends `Error` at hour eighty tells you nothing about the part — but
it does tell you what the eighty hours before it measured. **A segmented soak
that settles `Error` reports the AGGREGATE, not the window that errored.** The
errored window measured nothing, and an evicted pod usually printed nothing at
all; overwriting the fold with it would discard every clean segment at the last
step. So `metrics` on such a result is the same fold a completed soak would
carry, and it is the honest answer to "how much of this actually burned" — read
it together with `elapsedS`, which says how far the soak got.

An unsegmented test is unchanged: it has no fold, and its `Error` reports the
readings the errored attempt itself printed, which are what explain the error.

Read `TestAttempt.Trigger` to see why each attempt happened, and the run's
conditions for anything the operator noticed at plan time.

**`retryOnErrorLimit` re-runs an `Error` and nothing else.** On an unsegmented
soak that means the retry starts the duration over from zero, which is worth
knowing before you set it high. On a segmented one it re-runs the segment that
errored and nothing before it, which is the whole reason to segment.

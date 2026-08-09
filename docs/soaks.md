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
maintenance window overlaps a soak, the soak loses; schedule around it, or accept
the `Error` and the retry.

---

## Reading a soak that ended badly

The distinction that matters most is the one the whole operator is built around:

| | |
|---|---|
| **`Failed`** | the hardware was measured and fell short. **Never retried** — re-running a measurement until it comes out clean launders a fault into an acceptance. |
| **`Error`** | the measurement did not happen. The hardware is **unjudged**, not condemned. This is the retryable phase, and it is where an eviction, a drain, a reboot or an image-pull failure lands. |

A soak that ends `Error` at hour eighty tells you nothing about the part. Read
`TestAttempt.Trigger` to see why each attempt happened, and the run's conditions
for anything the operator noticed at plan time.

**`retryOnErrorLimit` re-runs an `Error` and nothing else.** On an unsegmented
soak that means the retry starts the duration over from zero, which is worth
knowing before you set it high.

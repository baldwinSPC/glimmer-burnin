# Invariants

This project decides whether a piece of hardware is fit to accept work. That is
the whole of what it does, and it means the expensive mistakes are not crashes —
they are **confident wrong answers**: a healthy node condemned, or a fleet
certified clean that nobody measured.

The rules below are what stand between the operator and those two outcomes. Each
one records a failure that actually happened. They are listed here as the
project's canonical statement of them, for anyone deciding whether to adopt this
thing or to contribute to it; the procedure for adding a test kind is in
[the playbook](new-testkind-playbook.md), and the runner-side contract is in
[docs/runner-contract.md](../runner-contract.md).

**The sentence the rest of this page is a case of:**

> A runner may only declare what it positively established, and the operator may
> only report what a runner declared.

Absence is never a declaration. Almost every rule here is that sentence applied
to a specific place where it was once got wrong.

---

## 1. The standalone rule

**This repository must never import its control-plane consumer.**

CI greps every `*.go` file for the consumer's module path and fails the build if
it finds one. There is no exception — not "just for a type", not "just in a
test".

The dependency direction is one-way. A consuming control plane may depend on
this project; this project may never depend on one. Integration happens through
the `BurnInSink` export contract — a versioned delivery envelope over a webhook,
described in [docs/sinks.md](../sinks.md) — and not through shared code.

The reason matters more than the rule. This project's first consumer is
private. If this repository imported it, it could not be published, and it
could not be adopted by anyone whose fleet has nothing to do with that consumer.
That is the entire premise of the project. **So when something is needed from
the control plane, the answer is to widen the contract, never to add an
import** — a widened envelope is something every adopter gets, and an import is
something only one consumer can use.

---

## 2. Permissive licences only, enforced rather than described

The allow-list is [`hack/licenses-allowed.txt`](../../hack/licenses-allowed.txt),
`go run ./hack/licensecheck` walks the whole Go module graph against it, and CI
blocks on the result. Prose describes the policy; that file decides it.

| Allowed | Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, and the public-domain equivalents 0BSD, Unlicense, CC0-1.0 |
|---|---|
| **Never** | GPL, AGPL, LGPL, SSPL, MPL, Sleepycat, CDDL, EPL — no exception process |
| **Also fails** | an UNCLASSIFIABLE licence, exactly as a copyleft one does |

A copyleft dependency anywhere in the graph — including a transitive one, which
is the case nobody sees — would impose terms this project cannot meet as source
under Apache-2.0 and as container images. It would make the project
unpublishable, which is to say it would end it. If the dependency you need is
copyleft, the answer is a different dependency, or an interface this project
defines and the consumer implements.

An unidentified licence must never default to acceptable, for the obvious reason:
it is the one most likely to be a problem.

**The incident.** ISC is on the list because the check, the first time it ran,
found the project had *already been shipping it* — `go-spew` arrives through
`k8s.io/apimachinery/pkg/util/diff` and is linked into the operator binary, not
test-only as its name suggests. The written policy at the time listed four
licences and had never described what the project actually distributed. That is
the shape of the problem a prose-only policy has.

Two further duties: where an upstream is dual-licensed, `NOTICE` must state which
option we consume it under (`linux-rdma/perftest` is GPL/BSD and is taken under
**BSD**); and runner images ship **no NVIDIA-licensed redistributable
libraries** — builds are multi-stage, and the Dockerfile fails if `ldd` finds
`libcuda`/`libcudart`/`libnv` in the final binary. `libcuda.so.1` is injected at
runtime from the host driver.

---

## 3. `Error` is not `Fail`, and the retry rule is what gives that teeth

An infrastructure error — an image pull, a scheduling failure, an eviction — must
stay distinguishable from a hardware verdict. Collapsing them is the single most
damaging simplification available in this codebase.

| phase | means | retried? |
|---|---|---|
| `Passed` | measured and within the declared thresholds | n/a |
| `Failed` | **measured and fell short** — a hardware verdict | **never** |
| `Skipped` | the test does not apply to this hardware | never — repeating will not make it start applying |
| `Error` | the measurement did not happen; the hardware is **unjudged** | yes, up to `spec.retryOnErrorLimit` |

`retryOnErrorLimit` re-runs an `Error` and nothing else. A `Fail` — whether it
arrived by a non-zero exit code or by a threshold violation on a clean exit,
which reach the decision by different routes — settles the test where it
happened, with the retry budget left unspent. **Re-running a measurement until it
comes out clean launders a hardware fault into an acceptance.**

There is exactly one place that decision is made, `completeAttempt` in
`internal/controller/burninrun_controller.go`, shared by every scope. Keep it
that way. `TestAttempt.Trigger` records why each attempt happened, so the rule
stays auditable from a stored result long after the run.

The consequence for runner authors is [in the runner
contract](../runner-contract.md): **exit 1 is the expensive code**, and it is the
one runners get wrong. `compute-smoke` shipped `v0.1.0` reporting "no usable CUDA
device", a wrong-arch image and every CUDA runtime error as exit 1 — three
failures to measure, each reported as a hardware failure verdict against the
node, none of them retried. `v0.2.0` was the first build where those are exit 3.
The `v0.1.0` tag is public and immutable and stays exactly as it is, which is
rule 12.

---

## 4. Verdict logic fails closed

A threshold naming a metric the runner did not emit is a **failure**, not a pass.
A missing measurement must never silently satisfy acceptance.

`pkg/verdict` performs no I/O, so this is
testable in isolation and so the bare-metal dispatcher reaches the same verdict
from the same output. It fails closed on all of:

- a metric that was never reported;
- a value that does not parse as `float64`;
- a **non-finite** value: `NaN` compares false against everything, which would
  otherwise turn `NotEqual` into a gate that always passes;
- a metric the runner reported *and* declared unmeasurable. The runner said two
  incompatible things about one measurement; which is true is unknowable from
  here, so the gate fails rather than picking the convenient reading.

---

## 5. Unmeasurable is a third state, and only a runner may declare it

Some hardware cannot produce a measurement at all. GB10 exposes no ECC to NVML,
so a perfectly ordinary `eccErrors Equal 0` gate failed **every healthy node in
the fleet** — a hardware verdict on hardware that was fine.

The fix is a reserved metric value. A runner that looked and found nothing to
measure emits `n/a` (case-insensitive): `eccErrors=n/a`. `pkg/runner` puts that
name in `Result.Unmeasurable`, never in `Result.Metrics`, and the two maps are
disjoint — storing a value would invent the measurement the sentinel exists to
deny. A threshold with `applicability: RequiredIfMeasurable` is then reported as
**not evaluated**, in `TestResult.Message` and in the delivered envelope, and
never as a pass.

Three sub-rules make that safe, and none may be relaxed:

| | |
|---|---|
| **Absence is not a declaration** | A metric that simply never appears still fails, under *every* applicability. A crashed probe must not become an acceptance. |
| **`Required` is the default** | It is the fail-closed behaviour, and it still fails closed on an unmeasurable metric. `RequiredIfMeasurable` is not a way to make a gate optional. |
| **Only a positive finding is declarable** | "We could not look" stays an omission. Emit `n/a` where you asked the hardware and it had nothing to report; omit the key where the probe itself failed. |

`host-health`'s ECC path is the worked example: it reads `ecc.mode.current` to
tell "this part has no ECC" from "ECC was switched off", and only the first is
declarable.

And never emit a `0` you did not measure. Without `/dev/kmsg`, `host-health`
reports `xid_source=none` and *omits* `xidEvents` — so a gate on it fails the
node rather than certifying it. A node condemned for a reading nobody took is
recoverable; a fleet certified clean by a fabricated zero is not.

---

## 6. A skip must be declared, not merely exited

Exit 2 is honoured as `Skip` only when stdout also carries an upper-case token
ending in `_SKIP` at the start of a line — `FP4_GEMM_SKIP:`, `NCCL_SKIP:`,
`IB_WRITE_BW_SKIP:`. **Exit 2 with no marker is an `Error`.**

An unrecovered Go panic exits 2, as does every Go runtime fatal error: out of
memory, concurrent map writes, stack exhaustion — none of which a runner can
recover from however carefully it is written. The camouflage was otherwise
perfect. A container log merges stdout and stderr, so a crashed runner produces a
stack trace with no `key=value` line at all, and "no metrics" is the *normal*
shape of a skip. That landed a crash on the one phase that is never retried,
never affects the run's verdict, and reports the node as one the test did not
apply to — so a run settled `Passed` around hardware nobody measured.

It fails towards `Error` deliberately. A runner that skips honestly but forgets
the marker is reported unjudged and retried, which is visible and cheap. The
opposite mistake certifies a fleet nobody looked at.

---

## 7. Thresholds compare exactly, and there is no epsilon

`GreaterThanOrEqual` and `LessThanOrEqual` gate a measurement; use both together
for a band. `Equal` and `NotEqual` are **exact**, and they exist for dimensionless
counters: `eccErrors Equal 0` means exactly zero, because a tolerance around zero
ECC errors is a tolerance for ECC errors.

They are the wrong tool for a continuous metric. A metric name carrying a unit
suffix cannot reproduce a decimal string, so `sustainedClockPct Equal 83.22`
fails on every healthy node forever — and the failure reads as a hardware
verdict.

**An epsilon was proposed for that and refused**, for three reasons that are
worth keeping written down:

1. it would silently reinterpret every counter gate already written, including
   the zero-tolerance ones that are the whole point of `Equal`;
2. it would not rescue the continuous case anyway — the tolerance such an author
   needs is domain knowledge, and the API can already express it as GTE + LTE;
3. being invisible in the spec, it would make a profile's meaning depend on which
   dispatcher evaluated it. This project has two, and they must agree about the
   same hardware.

The guidance is made discoverable at **authoring** time instead. Authoring
detail belongs to [docs/thresholds.md](../thresholds.md); the invariant is the
split between the severities, which is easy to erode and hard to restore:

| severity | what happens | why |
|---|---|---|
| **`Malformed`** | **refuses the run**, as a config `Error` | A gate nothing can satisfy fails every node forever. Reporting that in the shape of a hardware verdict is exactly the confusion `Error` is kept distinct from `Fail` to prevent. |
| **`Unsound`** | does **not** block; recorded on the run's `ThresholdsSound` condition and the run proceeds | The gate does evaluate. This operator must not veto profiles it merely does not understand. |

`Unsound` is a **condition** and not an Event on purpose: an Event expires in an
hour, and the verdict it qualifies is kept for years.

Do not promote advice to a refusal, and do not demote a refusal to advice.

The cheapest surface says what a regex can say: the CRD carries `Pattern` markers
on `Threshold.Metric` (lowerCamelCase) and on `Threshold.Value` (the finite
decimal forms — so `NaN`, `Inf` and `twenty` are refused by the apiserver, while
the author is still holding the file).

---

## 8. Scope decides what a verdict is *about*

A verdict names a thing. Getting that thing wrong sends an engineer to replace
the wrong part, which is worse than saying nothing. Scopes and the run lifecycle
are described in [docs/concepts.md](../concepts.md); the invariants are these.

### A Pair verdict is about the LINK

Pair scope runs two pods — a server on the first target, a client on the second,
rendezvous'd through one headless Service — and produces exactly **one**
`TestResult` naming **both** nodes. Never split it per node: a point-to-point
measurement is a property of the link, and attributing it to one endpoint sends
an engineer to replace the wrong part — a cable, a transceiver, a switch port or
either NIC can produce the same number, and the verdict names none of them.

Three rules hold it together:

- **The client is the deciding side.** It is where perftest and nccl-tests
  report, so its metrics win any key both ends emit, and the pair settles when
  the client terminates rather than waiting for a server that may legitimately
  linger.
- **Verdict precedence is `Error` > `Fail` > `Skip` > `Pass`.** A machinery
  failure on either end means the link was never measured, and a client's
  "connection refused" is then an artefact of its peer rather than evidence about
  the fabric. Recording that as `Fail` would permanently indict a link over an
  unpullable image — and `Fail` is the one phase that is never retried.
- **A server that exits 0 before the client ever started is an `Error`**, never a
  pass. No traffic crossed the link.

### A Group verdict is about the COLLECTIVE

Group scope runs one rank per target node — target *i* is rank *i*, pinned in the
plan — all rendezvous'd through one headless Service, producing exactly one
`TestResult` naming **every** node. Rank 0 is the root and starts first; no other
rank is created until it is Ready.

Three rules differ from Pair, and each follows a real difference:

- **Every rank is waited for.** A collective is synchronous, so its ranks finish
  together, and one that has not finished is one the collective is still waiting
  on.
- **A rank that never reported is not a pass.** A collective is only measured if
  every rank took part.
- **A deadline names the ranks that actually hung**, because every healthy rank
  blocks on the faulty one and would otherwise be indicted equally.

Group needs neither JobSet nor OpenMPI, and both were refused on evidence. Gang
scheduling solves partial placement under contention, and every rank here is
pinned by hostname to a node this operator has already admitted and cordoned, so
there is no placement decision left; JobSet would additionally be a controller
every adopter must install and a second owner of pod lifecycle, which would make
rule 10 unassertable. OpenMPI needs a launcher able to start a process on another
node, which on Kubernetes means an sshd and a key on every accelerator node in
the fleet.

### A runner that speaks only Pair must refuse Group

Every fabric runner branches on `BURNIN_ROLE`, and Group deliberately does not
set it — so a Pair-only runner asked to join a collective would read the absence
as a Node-scope run, print a `_SKIP` marker and exit 2, and the run would settle
`Passed` around a collective that never executed. That is the one verdict that
certifies, awarded for a measurement nobody took.

Both halves are closed. The runners exit 3 *before any hardware inspection*,
because "this image does not speak the Group rendezvous" is true whatever is in
the box. And the operator refuses the same case at plan time (issue #118), which
is the half that protects a fleet running already-published tags — those are
immutable and would skip forever.

`groupCapableKinds` in `internal/controller/plan.go` currently holds `nccl`
alone: rank 0 serves the ncclUniqueId to the other N−1 ranks. `ib-write-bw` and
`gpudirect-rdma` refuse and always will, because a point-to-point RDMA write and
a GPUDirect peer-memory check have no N-rank form.

**Reading a variable in order to refuse is not implementing the contract.** The
guard for this list is keyed on `BURNIN_ROOT_HOST`; a first attempt keyed on
`BURNIN_RANK`/`BURNIN_NRANKS` failed immediately, because the refusing runners
read exactly those in order to say no.

---

## 9. The run's footprint on the fleet

**Cordoning follows the wave, not the target list.** A node is cordoned
immediately before the run puts load on it and released once it is no longer
holding any, so a run's footprint tracks `spec.maxConcurrentNodes` rather than
the size of its target list. Both nodes of a pair are cordoned together, before
either pod exists. Cordoning every target up front took a two-node cluster
entirely out of scheduling and hollowed out the interlock, whose whole premise is
that only N nodes are occupied at once.

**A wave can be many pods long, and "holding" is read from the RESULT and not
from a live pod.** A segmented soak, a repeat and an error-retry each have a
moment between pods where nothing is running, and reading the interlock from live
pods alone made that gap look like released capacity: a second target took the
slot and the first node's cordon came off halfway through a multi-day acceptance.
Both halves of that are severe — the soak stops being a soak (at cap 1 over two
targets each node runs one window then idles for one, and a part allowed to cool
for as long as it was loaded never reaches the thermal steady state the test
exists to hold it at), and the cordon is the only thing keeping *foreign*
workload off a node under burn-in, since runner pods tolerate
`node.kubernetes.io/unschedulable` and it therefore never protected against this
operator to begin with. `holdingNodes` seeds the busy set from every non-terminal
result's nodes — every node of the unit, so a pair holds both ends and a group
holds every rank — and it *seeds* rather than replaces, so the live-pod count
stays the floor and a controller restart still cannot lose track of what is
already running.

**A pair is one unit of load and costs two slots.** `maxConcurrentNodes` counts
nodes, and a pair holds both of its nodes for the whole test, so it is admitted
only when two slots are free — and never at the default cap of 1, which the run
refuses at start with that explanation. A Group needs the cap to be at least its
target count, likewise refused at start. Admission is checked once, when the unit
starts; the client is part of an already-committed unit and is not re-checked, or
a lowered cap would strand a running server against a peer that never arrives.
The arithmetic that implies for a long run is in [docs/soaks.md](../soaks.md).

**What the run found is captured once, at start, and never re-derived.** Because
a node is cordoned and released per wave, "was this node already cordoned when I
got here?" would otherwise be re-asked several times per run and answered from
whatever `spec.unschedulable` said at that instant — *after* the run already had
a footprint. Anything the run could not attribute to itself — its own hold seen
through an annotation it no longer recognises after a manager restart, or an
administrator cordoning a node the burn-in was already holding, which on an
unschedulable node is an invisible no-op — was then recorded as pre-existing and
made **permanent** at teardown. A stranded cordon, signed off as intentional.

`status.priorUnschedulable` is that record: written before the first cordon,
never overwritten, and it — not the node's annotation — decides what the release
restores. The node annotation remains as the account a human reads off the node,
and as the fallback for a run that started under an older operator. The
deliberate consequence is that a cordon placed on a target *after* the run
started is not adopted and is undone at teardown. That is the right way round: a
cordon undone is visible and one command to redo, while a node the fleet silently
loses has nothing left in the cluster that knows it was taken.

**A runner pod is never safely evictable, and that is not a knob.** Every runner
pod carries `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`,
unconditionally, at every scope. It is holding a node this operator cordoned
precisely so that nothing else lands there, and moving it discards a measurement
in progress — recorded as an `Error`, spending retry budget on a fault the
cluster caused rather than the hardware. Over a multi-day soak the cluster's own
housekeeping is the most likely thing to end a run.
`spec.runner.priorityClassName` is a passthrough for the different mechanism
(preemption under contention); the operator never creates a PriorityClass,
because a cluster-scoped object minted by a test-runner is a footgun and a
permission it should not hold.

**Runner pods must tolerate the operator's own cordon.** The node controller
expresses a cordon as `node.kubernetes.io/unschedulable:NoSchedule`, so a pod
that does not tolerate it can never be scheduled onto the node the cordon was
placed for — the run would hold a node out of service and then wait forever. The
toleration is added unconditionally and is scoped to that one taint by key. It is
emphatically not a blanket toleration, and removing it deadlocks every test on
every node.

---

## 10. A pod may only be destroyed once the status justifying its destruction is readable from the apiserver

This is an ordering rule inside `terminate`, and it is the one that is easiest to
undo by accident while tidying.

**The incident (#84).** The pods were killed first. When the terminal status
write then lost a `resourceVersion` race, the reconcile returned an error, the
terminal status was discarded, and the run went back to believing it was
`Running` with an open execution. Its next pass found the pod *it had just
killed*, harvested the SIGKILL, and recorded the hardware as
`Error: runner terminated abnormally (exit 137)` — the operator condemning a
healthy node for its own act, thirty seconds into a hundred-second soak.

With the write first, a lost race costs one wasted reconcile pass and nothing
else, because nothing has been destroyed. The finalizer comes off last, as the
standing record that the run still owes the cluster something — a cordon to give
back and a pod to stop — and `reconcileTerminal` sweeps anything left, so the
cleanup stays level-based rather than depending on one call succeeding.

It is now **asserted rather than argued**:
`test/envtest/invariants_test.go` →
`TestNoPodIsDestroyedBeforeTheStatusJustifyingItIsDurable` runs the scenario
against a real apiserver and makes the terminal status write lose a *real*
`resourceVersion` race — the competing write is performed and the apiserver
returns the 409 — then checks, at every pod deletion, that the run's durable
status already justified it. Reproducing that needs a real apiserver, which is
why the assertion did not exist while the only harness was controller-runtime's
fake client. The condition it depends on is the shipped Deployment's own shape:
`replicas: 1` under the default RollingUpdate has `maxSurge` 1, so an apply
starts the second manager before the first is gone.

Read that test before reordering any write in `terminate` or `reconcileTerminal`.

---

## 11. Host access is named, narrow, and never implicit

`spec.runner.hostPaths` is the **only** way a runner pod reaches the node's
filesystem. A test that declares nothing gets a pod with no volumes at all: no
default `/dev`, no convenience `/sys`, nothing.

The shape is a curated `HostPathMount{path, mountPath, readOnly, type}` rather
than `[]corev1.Volume` + `[]corev1.VolumeMount`, because every mount a burn-in
runner has ever needed is a host device or a host log path, and the curated form
states that intent where a general volume list would bury it. It also refuses
what the general form cannot.

| rule | why |
|---|---|
| `readOnly` **defaults to true**, and is a pointer | A privilege grant's default has to fall towards the harmless form. The pointer is what makes "unset" distinguishable from "false", including for objects the apiserver never defaulted. |
| `type` admits only the **assert-only** hostPath kinds (`Directory`, `File`, `Socket`, `CharDevice`, `BlockDevice`) | `DirectoryOrCreate` and `FileOrCreate` are deliberately absent: an operator sent to measure a fleet must never mutate it. Setting `type` also converts "the runner saw an empty directory and reported nothing" into a pod that refuses to start — an `Error`, which says the hardware was never judged, where an empty mount looks like a measurement. |
| Mounts are built in `podForTest`, the one place a runner pod is constructed | Node scope and **both** pods of a Pair get identical host access by construction. A link measured from one end is not measured. |
| The mount spec rides in the **pinned plan** | Editing a BurnInTest mid-run cannot widen what an in-flight attempt takes off the host. |

**`privileged: true` is not a substitute.** Measured on this project's own nodes,
a privileged `hostNetwork` pod was missing every `/dev/infiniband/uverbs*` device
the host had — which is exactly what `ibv_create_cq` opens.

Widen to the general volume form only for a case the narrow one genuinely cannot
express, and say which.

The other half of the rule is on the runner: **a runner without its mount must
degrade honestly.** Without `/dev/infiniband` the fabric runners fail loudly at
`Couldn't create CQ` rather than reporting a number. Never paper over a missing
mount with a fabricated measurement.

---

## 12. Published tags are immutable

A runner image is executed by a node's readiness gate, against hardware a site is
deciding whether to accept. That gives images two properties this project treats
as non-negotiable:

- **Publication is manual**, via the `publish-runner` workflow's
  `workflow_dispatch`, never automatic — a bad tag degrades a whole fleet at once
  — and not before the kernel has been verified on real hardware. "It builds" is
  not verification.
- **A tag is never republished.** Not for a bug fix, not for a security fix. Cut
  a new version. Silently changing a tag changes every node's verdict with no
  audit trail — and every profile that pins it, retroactively.

`pkg/runnerimages/images.go` is the single table mapping a `TestKind` to its
default image, and it is public precisely because the operator is not the only
thing that runs these images: a bare-metal dispatcher reads the same table, and
a second copy would drift until a node passed under one and failed under the
other. That exact failure is on record in this solution — a second
implementation running a *different* test under the same `TestKind` name.

A kind with **no** entry is not an oversight. It fails fast and legibly at plan
time asking for an explicit `spec.runner.image`, which is far better than
scheduling a pod with no image and getting an `ImagePullBackOff` that reads as an
infrastructure fault on every node in turn.

The same logic pins upstreams: every `<UPSTREAM>_REF` in a Dockerfile is
accompanied by an `<UPSTREAM>_SHA` and the build refuses to proceed when the
clone is not that commit. A tag is a mutable pointer, and because our published
tag is immutable, a moved upstream tag would change what a fleet is measured by
with nothing in this repository recording it.

Verifying that a pulled image really is one of ours is
[docs/verifying-images.md](../verifying-images.md).

---

## 13. Arch targets are deliberate

**CPU architecture and GPU architecture are orthogonal.** `linux/amd64` versus
`linux/arm64` is the *host* the container runs on; `sm_121a` / `sm_90` / `sm_100`
is the GPU gencode. An x86 host with an H100 needs `linux/amd64` **and** `sm_90`;
an amd64 image containing only `sm_121a` device code helps nobody, because CC
12.1 is GB10, GB10 is a Grace part, and there is no x86 host with one.

Do not widen a gencode target to make a build succeed. `sm_121a` emits a cubin
for GB10 only, with no PTX fallback, so a pass proves the *real* instruction path
ran rather than an emulated one — which is the entire claim `compute-smoke`
makes. Where a runner's claim does not depend on which instructions ran (the soak
family) it compiles a list of real cubins instead, and that list has to cover the
parts x86 fleets actually run. Note PTX only JITs **upward**: `compute_121` PTX
cannot rescue a CC 10.x part, which is why an explicit `sm_100` cubin is needed.

**A wrong arch does not reliably announce itself**, which is why this is an
invariant rather than a build preference. `cudaErrorNoKernelImageForDevice` fires
only when the loader refuses outright. The other direction is not refused at all:
SM121 is binary-compatible with SM120, so an `sm_120a` image *loads* on a CC 12.1
GB10 and trips a device-side assert inside the kernel, reaching the operator as a
bare "device-side assert triggered" under several hundred lines of CUTLASS
template text. `compute-smoke` therefore establishes the mismatch from the
`BURNIN_CUDA_ARCH` string and the reported compute capability and refuses
**before** launching — `cudaErrorAssert` is sticky and poisons the context, and
`pkg/runner.Parse` takes `Result.Message` from the *last* non-`key=value` line,
so kernel spew arriving after the diagnosis would overwrite it in the stored
result.

That refusal stays an `Error` and never a `Skip`, and an arch string it cannot
parse is Unknown rather than a mismatch. Which is rule zero again: a runner may
only declare what it positively established.

---

## 14. A Node verdict covers every accelerator on the node

**A Node verdict describes EVERY accelerator on the node, gated on the WORST;
a single-device measurement on a multi-device node is a bug.**

Every CUDA/HIP runner used to take device 0, and every shipped sample requested
one GPU. On an eight-GPU node one arbitrary card was certified as the node and
nothing in the stored result said so — a confident wrong answer of exactly the
kind this page exists to prevent. The full design, reviewed before code, is
[multi-device.md](multi-device.md); the rules that hold it up:

- **The runner iterates over the devices it was ALLOCATED**, not the devices it
  can see. The operator injects the pod's own resource limits verbatim as
  `BURNIN_RESOURCE_LIMITS`, interpreting nothing (which name is the accelerator
  is vendor knowledge, and it stays in the image); the runner sums its own
  vendor's count-shaped names into a budget. **`budget ≠ visible` is exit 3.**
  A budget is a count and not an identity: on a host where the legacy runtime
  injects from the image's baked `NVIDIA_VISIBLE_DEVICES=all`, a pod handed one
  card sees the whole board, and capping the iteration at one would still
  iterate device 0 — which may be another pod's.
- **The pod requests the SKU's board** — one profile and one run per SKU —
  and a wrong count is an unschedulable pod whose `Error` carries the
  scheduler's own reason (`Insufficient nvidia.com/gpu`).
- **`durationSeconds` is the test's budget and does not grow with the board.**
  A soak kind iterates concurrently (a soak sliced into `duration/8` windows is
  not a soak); a measurement kind divides the window, and `deviceWindowS`
  says what each device got.
- **The gated metric keeps its name and is the worst device.** The direction
  is the metric's, read off the registry's `Aggregation`; identity labels keep
  the first device's meaning; `worstDeviceIndex` and `worstDevicePciBusId`
  name the device behind the gated figure of one window, and the per-device
  table is an **artifact**, never a suffix on a metric name.
- **`deviceCount` is an acceptance gate, claimed only by a runner that folded
  devices.** A fleet writes `deviceCount Equal 8`, so a pod handed one card of
  eight fails instead of certifying that card. A runner that merely SAW
  devices — a node-wide read-only probe, or one not yet converted — reports
  `devicesVisible` (what `gpu_count` became; not aliased) and does not claim
  the gate.
- **Spreads are `n/a` on one device, under MIG and on a mixed board** — a
  positive claim, "nothing to spread across" — so a threshold on one must be
  `RequiredIfMeasurable`, and the threshold linter reports a `Required` gate on
  a spread as `Unsound`.
- **Across DEVICES the precedence is `Fail > Error > Skip > Pass`, and that
  inverts Pair's `Error > Fail` deliberately.** A device is a *part*: a
  measured miscompare on device 3 is a fact about device 3 that device 6's
  enumeration failure does not erase. A Pair endpoint is *half a link*: a
  machinery failure at either end means the link was never measured. Both
  orders are invariants; the discriminator is what the reporters are.

The fold lives in one CUDA-free header, `device_fold.h`, byte-identical across
the runners that carry it and unit-tested under `make test`;
`runners/devicefold_test.go` holds every runner directory to one of CONVERTED,
EXEMPT with a reason, or PENDING with the delivery step, so a new accelerator
runner cannot regress to device 0 silently.

---

## Where these are enforced

A rule with no guard is a rule that will be gone in a year. The tiers differ in
what they can see, which is why the same property is sometimes asserted twice.

| tier | runs | can see |
|---|---|---|
| unit (`internal/…`, `pkg/…`) | every PR, seconds | pure logic, result bookkeeping, parsing, verdicts |
| **envtest** (`test/envtest/`) | every PR | RBAC enforcement, CRD schema and defaulting, the status subresource, **real `resourceVersion` conflicts** |
| **e2e** (`test/e2e/`, build tag `e2e`) | PR (core) / main and weekly (chaos) | a real scheduler and kubelet, the shipped manifests, DNS, manager restarts |

The unit suite runs against controller-runtime's fake client, which is a map with
a type checker in front of it. Five bugs shipped through a green fake-client
suite because of what that client does not do; the two tiers above it exist for
exactly that reason. Write a new test at the **lowest** tier that can actually
observe the property — and note that when a tier cannot observe it, moving up is
not optional. `test/envtest` skips itself when the control-plane binaries are
absent so `go test ./...` stays green on a laptop; CI sets
`BURNIN_ENVTEST=required`, which turns that skip into a failure. A suite that
silently skips in CI is worse than no suite.

---

## See also

- [docs/runner-contract.md](../runner-contract.md) — the contract a runner image
  must satisfy
- [docs/dev/new-testkind-playbook.md](new-testkind-playbook.md) — the ordered
  procedure for adding one here
- [docs/concepts.md](../concepts.md) — the CRDs, the scopes, how a run proceeds
- [docs/thresholds.md](../thresholds.md) — authoring thresholds, applicability,
  the linter
- [docs/sinks.md](../sinks.md) — the delivery envelope, idempotency, the three
  sinks
- [docs/soaks.md](../soaks.md) — capacity arithmetic for a multi-day run
- [docs/verifying-images.md](../verifying-images.md) — signature verification

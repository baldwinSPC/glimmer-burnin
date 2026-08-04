# CLAUDE.md

Guidance for AI assistants (Claude and others) working on **glimmer-burnin**.

---

## What this is

A vendor-neutral, Kubernetes-native **burn-in operator**: it runs acceptance
tests against accelerator nodes (compute smoke, thermal soak, NCCL, RDMA
bandwidth, DCGM diagnostics), evaluates the results against declared
thresholds, and exports verdicts to an external consumer.

It is a standalone project, published under **Apache-2.0**. It is designed to be
useful to anyone running GPU fleets, not only to its first consumer.

---

## The standalone invariant (the most important rule here)

**This repository must never import the Glimmer control plane.**

CI fails any PR adding an import of `github.com/baldwinSPC/glimmer/…`. There is
no exception, including "just for a type" or "just in a test".

- Dependency direction is one-way: Glimmer MAY depend on this project; this
  project MUST NEVER depend on Glimmer.
- Integration happens through the `BurnInSink` export contract — a delivery
  envelope over a webhook — not through shared code.
- This is what allows the repo to be open-sourced and adopted independently. If
  you need something from Glimmer, the answer is to widen the contract, not to
  add an import.

If a task appears to require reaching into Glimmer, stop and raise it rather
than working around the guard.

---

## Cross-repo context

The Glimmer solution spans several repositories. The canonical registry and the
cross-repo rules live in the `baldwinSPC/glimmer` repo's own `CLAUDE.md`, which
is **private** — so the rules that matter here are restated below rather than
linked, since public contributors cannot read the original.

| Repo | Visibility | Role |
|------|------------|------|
| `baldwinSPC/glimmer` | private | Control plane, and the planning home for the whole solution |
| `baldwinSPC/glimmer-burnin` | this repo | The burn-in operator |
| `baldwinSPC/glimmer-releases` | public | Release artifacts only; never source |

Rules that apply to work in this repo:

- **Design proposals (GEPs) live in `glimmer`**, at `docs/geps/NNNN-name.md`,
  even when the code lands here. This project's design is GEP-0178. The GEP
  states which repo owns which piece.
- **Implementation issues are filed here**, in the repo where the code lands,
  and linked to the workstream's tracking issue in `glimmer`.
- The dependency and license rules apply to every repo in the solution, and
  most strictly to this one, because it is the one that gets published.

---

## Licensing rules (strict)

- **Permissive only: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause.**
- **Never** add a dependency under a copyleft license — GPL, AGPL, LGPL, SSPL,
  MPL, or Sleepycat. This is not a preference; a copyleft dependency would make
  the project unpublishable.
- Verify a new dependency's license **in the PR description**. If you cannot
  verify it, ask before touching `go.mod`.
- Where an upstream project is dual-licensed, state explicitly which option we
  consume it under, in `NOTICE` (e.g. `linux-rdma/perftest` is GPL/BSD
  dual-licensed and must be consumed under the **BSD** option).
- Do not reproduce substantial blocks of third-party source. Header-only
  libraries compiled in (e.g. CUTLASS) are fine and are recorded in `NOTICE`;
  redistributing their source is not.

Runner images have an additional constraint: they must ship **no
NVIDIA-licensed redistributable libraries**. Builds are multi-stage so the CUDA
toolchain is used at build time only, and the Dockerfile fails the build if
`ldd` finds `libcuda`/`libcudart`/`libnv` in the final binary. `libcuda.so.1`
is injected at runtime from the host driver by the NVIDIA Container Toolkit.

---

## Build, test, lint

```sh
make generate        # deepcopy methods (controller-gen)
make manifests       # CRDs + RBAC
make test            # unit tests + envtest (envtest skips without its binaries)
make lint            # gofmt + go vet
make build           # binary
make all             # generate manifests fmt vet test build

make envtest-assets  # download the envtest kube-apiserver + etcd
make test-envtest    # controller invariants against a REAL apiserver
make e2e             # kind cluster + apply config/ + the e2e suite
make e2e-clean       # delete the kind cluster
```

After changing anything in `api/v1alpha1`, run `make generate manifests` and
**commit the results** (`zz_generated.deepcopy.go`, `config/crd/*`,
`config/rbac/*`). CI checks for drift.

CI (`.github/workflows/ci.yml`) runs build, vet, gofmt, test, the
no-glimmer-import guard, and the CRD drift check. All of them are **blocking** —
nothing there is informational.

### Three tiers of test, and what each one can see

The unit suite runs against controller-runtime's **fake client**, which is a map
with a type checker in front of it. Five bugs shipped through a green fake-client
suite because of what that client does not do, so two more tiers exist above it.
Write a new test at the LOWEST tier that can actually observe the property:

| Tier | Runs | Sees |
|---|---|---|
| unit (`internal/…`, `pkg/…`) | every PR, seconds | pure logic, result bookkeeping, parsing, verdicts |
| **envtest** (`test/envtest/`) | every PR, ~12s | RBAC enforcement, CRD schema + defaulting, the status subresource, **real resourceVersion conflicts** |
| **e2e** (`test/e2e/`, tag `e2e`) | PR (core) / main + weekly (chaos) | a real **scheduler** and **kubelet**, the shipped manifests, DNS, manager restarts |

- **`test/envtest`** needs the control-plane binaries and **skips itself** when
  they are absent, so `go test ./...` stays green on a laptop. CI sets
  `BURNIN_ENVTEST=required`, which turns that skip into a failure. Never remove
  that: a suite that silently skips in CI is worse than no suite.
- **`test/e2e`** is behind the `e2e` build tag so `go test ./...` can never pick
  it up. Its most valuable part is not a test — it is the CI step that applies
  `config/crd`, `config/rbac` and `config/manager` to a cluster, which is the
  only thing that could have caught a release shipping with no deployment
  manifest at all. That step runs first, before the image is even built.
- The e2e runners are `sh -c` one-liners on **busybox**, deliberately. Hosted CI
  has no GPU, and every bug this tier exists for is orchestration: scheduling,
  cordoning, RBAC, rendezvous, delivery, recovery.
- **What the PR tier does not guard**: the chaos e2e (manager restart,
  rolling-update overlap, delete-mid-flight) runs on pushes to main, on the
  weekly sweep and on demand, not on pull requests, because each case waits out a
  leader-election lease. The same invariants are asserted deterministically on
  every PR by `test/envtest/invariants_test.go`, which injects the
  resourceVersion conflict rather than racing a rollout for it. What is genuinely
  unguarded between merges is the interaction with a real kubelet — a pod
  SIGTERMed by a departing manager, and the lease handover itself.

It also builds runner IMAGES, which for a long time it did not, and that is why
two runners once shipped in a PR having never been built. The matrix is derived
from the filesystem, never from a hand-written list:

- on a PR or push: every runner whose **directory the change touches** (a
  runner's Docker build context is its own directory, so nothing outside it can
  affect the image), plus the runners with no emulated build stage, as a
  standing smoke test;
- **weekly**, and on `workflow_dispatch`: all of them, which is what catches a
  base image or an apt package moving under a runner nobody edited.

Build-only; publishing stays manual in `publish-runner.yml`. The split exists
because this repo is private, so arm64 builds run under QEMU and are billed by
the minute. When the repo goes public, set the repository variable
`ARM64_RUNNER` to `ubuntu-24.04-arm`: the QEMU step skips itself, the builds
become native, and the honest thing is then to build every runner on every PR.

Several guards need no Docker at all and so run in `make test`:
`runners/pins_test.go` (every upstream pinned to a commit the build asserts;
every `FROM`'s ARG declared before the first `FROM`; every runner offered by the
publish workflow; every runner either reading `BURNIN_DURATION_SECONDS` or its
kind declaring itself burst-only; no Dockerfile defining a fault-injection
macro), `api/v1alpha1/samples_test.go` (every sample decodes strictly, every
sample threshold survives the linter, and no sample asks a burst-only kind for a
duration), the shared-source drift guards described below, and
`runners/cxxtests_test.go`, which compiles and runs every `*_test.cc` in the
tree. That last one is how a C++/CUDA runner gets unit tests at all: the part
worth testing is never the part needing a GPU, so it is kept in a CUDA-free
header (`compute-smoke/arch_match.h`) with a plain C++ test beside it. Note a
runner directory containing a `.cc` file cannot also hold a Go file — `go build`
rejects C++ sources outside cgo — which is why the sweep lives in `runners/`.

---

## Layout

```
api/v1alpha1/        CRD types: BurnInTest, BurnInProfile, BurnInRun,
                     BurnInSchedule, BurnInSink, NodeFingerprint
internal/controller/ the reconcilers: BurnInRun (run core, cordon, plan,
                     pods, Pair rendezvous, delivery), BurnInSchedule,
                     NodeFingerprint
internal/sink/       sink delivery engine: webhook, ConfigMap, Prometheus
                     selection, retry and idempotency
internal/metrics/    Prometheus exposition of run state, registered on
                     controller-runtime's registry
runners/             per-TestKind runner images, one directory each
config/crd|rbac|manager|samples/   generated manifests, the operator's own
                     Deployment, and examples
test/envtest/        controller invariants against a real apiserver + etcd
test/e2e/            the shipped manifests on a real kind cluster (tag `e2e`)
```

Public packages — importable by other projects, free of Kubernetes types:

```
pkg/verdict/         pure threshold evaluation (no k8s, no I/O)
pkg/runner/          runner exit-code + key=value stdout parsing
pkg/contract/        versioned delivery envelope + metric-name registry
```

`pkg/verdict` and `pkg/runner` are public because Glimmer's pre-Kubernetes
burn-in path runs the same runner images: if the two dispatchers derived
different metrics or different verdicts from identical runner output, they would
disagree about the same hardware. One brain, two dispatchers.

---

## Conventions

- **Vendor neutrality lives in the reconciler.** Accelerator- and
  vendor-specific behaviour belongs in *runner images*, never in controller
  branches. Adding support for a vendor should mean adding a runner, not adding
  an `if nvidia {}`.
- **Every new `TestKind`** ships with a result parser (an alias table entry in
  `pkg/runner/parse.go` where the runner's keys are not merely a different
  spelling of the canonical names) and a unit test for that parser. It gets a
  `defaultRunnerImages` entry in `internal/controller/pods.go` once a runner
  image exists for it; a kind with no runner source in this repo deliberately
  has none, so it fails fast at plan time asking for an explicit
  `spec.runner.image` instead of pull-failing per node.
- **Verdict logic fails closed.** A threshold naming a metric the runner did not
  emit is a FAILURE, not a pass — a missing measurement must never silently
  satisfy acceptance. Keep it that way; `pkg/verdict` is deliberately pure so
  this is testable in isolation.
- **Unmeasurable is a third state, and only a runner may declare it.** Some
  hardware cannot produce a measurement at all: GB10 exposes no ECC to NVML, so
  `eccErrors Equal 0` used to fail every healthy node in the fleet. A runner that
  looked and found nothing to measure emits the reserved value `n/a`
  (`eccErrors=n/a`), which `pkg/runner` puts in `Result.Unmeasurable`, never in
  `Metrics`. A threshold with `applicability: RequiredIfMeasurable` is then NOT
  EVALUATED — reported as such in `TestResult.Message` and the envelope, never
  as a pass. Three rules make this safe and none of them may be relaxed:
  ABSENCE IS NOT A DECLARATION (a metric that simply never appears still fails,
  under every applicability, because a crashed probe must not become an
  acceptance); `Required` is the default and still fails closed on an
  unmeasurable metric; and a runner may only declare what it positively
  established — "we could not look" stays an omission. host-health's ECC path is
  the worked example: it uses `ecc.mode.current` to tell "this part has no ECC"
  from "ECC was switched off", and only the first is declarable.
- **Thresholds compare exactly, and there is no epsilon.** `GreaterThanOrEqual`
  and `LessThanOrEqual` gate a measurement (both together for a band). `Equal`
  and `NotEqual` are EXACT and exist for dimensionless counters: `eccErrors
  Equal 0` means exactly zero, because a tolerance around zero ECC errors is a
  tolerance for ECC errors. They are the wrong tool for a continuous metric — a
  name carrying a unit suffix cannot reproduce a decimal string, so
  `sustainedClockPct Equal 83.22` fails on every healthy node forever and the
  failure reads as a hardware verdict. Do not "fix" that with an epsilon: it
  would silently reinterpret every counter gate already written, it would not
  rescue the continuous case anyway (the tolerance such an author needs is
  domain knowledge the API can already express as GTE+LTE), and being invisible
  in the spec it would make a profile's meaning depend on which of the two
  dispatchers evaluated it. The guidance is made discoverable at AUTHORING time
  instead: `pkg/verdict.ValidateThresholds` reports exact comparisons on
  continuous metrics, gates on metrics `pkg/contract` marks `ThresholdUse:
  Evidence`, and thresholds that can never be evaluated at all. It is advisory
  and changes nothing about evaluation, which still fails closed — including on
  a non-finite value, since `NaN` compares false against everything and would
  otherwise make `NotEqual` a gate that always passes.
- **The threshold linter has three surfaces, and the severities land in
  different places.** A warning nobody runs is not discoverability, so
  `ValidateThresholds` is wired in rather than merely available. The CRD's
  `Pattern` markers on `Threshold.Metric` and `Threshold.Value` refuse the
  malformed spellings at the apiserver, which is the earliest and cheapest place
  any of it can be said; `buildPlan` lints the resolved profile before a node is
  cordoned; and `TestSamplesThresholdsLintClean` holds `config/samples/` to the
  same bar in CI. Then: a **`Malformed`** problem REFUSES the run as a config
  `Error` — a gate nothing can satisfy fails every node forever and would
  otherwise be reported in the shape of a hardware verdict, which is exactly the
  confusion `Error` is kept distinct from `Fail` to prevent. An **`Unsound`** one
  does NOT block: it is recorded on the run's `ThresholdsSound` condition and the
  run proceeds, because the gate does evaluate and this operator must not veto
  profiles it merely does not understand. It is a condition rather than an Event
  on purpose — an Event expires in an hour and the verdict it qualifies is kept
  for years. Keep the split; do not promote advice to a refusal, and do not
  demote a refusal to advice.
- **Unregistered is legal, except on a runner this project ships.** The registry
  is an OPEN world — a lowerCamelCase name nothing has registered is valid, so a
  third-party runner can report a new measurement without a release of
  `pkg/contract` — and that stays true. But the same silence hid a gate on a
  `clockprobe` metric whose values are `true`/`false`/`unknown`, which fails
  closed on every node forever. So `verdict.ValidateThresholdsForKind` asks the
  question only where the answer is knowable: for a kind in
  `v1alpha1.BuiltInKinds` this project owns the runner, the parser and the name,
  so an unregistered gated metric is either a missing registry entry or a gate
  nothing satisfies, and it is reported as `Unsound`. For `KindCustom` — and for
  any unrecognised kind, since `TestKind` is an open enum — nothing is said, and
  the kind-agnostic `ValidateThresholds` keeps exactly that behaviour. **Add a
  new kind to `BuiltInKinds`** when you add its constant, or the check silently
  stops applying to it. Writing a threshold is what promotes a name from
  incidental evidence to acceptance-deciding, which is why registration is owed
  at that point and not before.
- **Metric names are a contract.** `pkg/contract/metrics.go` holds the registry
  and the grammar: lowerCamelCase, a dimensional metric ends in a registered
  unit suffix (`Gbps`, `GBs`, `MBs`, `Us`, `Ms`, `S`, `C`, `W`, `Pct`,
  `Tflops`, `MHz`), a dimensionless counter does not. A runner's own key is
  mapped to the canonical name by the alias table in `pkg/runner/parse.go`;
  parsing is last-occurrence-wins, so two keys must never alias to the same name.
- **A first-party metric whose value is a LABEL must be registered, as
  `Evidence`.** The registry is an open world and that is right for third-party
  runners, but it has one consequence that is a bug for our own: an unregistered
  name is assumed thresholdable. `pdWedgeSuspected` is `true|false|unknown`,
  `throttleClassification` is one of five words, `computeCap` is `12.1` — a
  threshold is compared as a `float64`, so a gate on any of them passed every
  authoring-time check and then failed closed on EVERY node forever, reading as
  a hardware verdict on healthy hardware. Registering them turns
  `SafeToThresholdOn` to false, so the gate is reported as `Unsound` on every
  surface the bullets above describe. `computeCap` is the sly one: it parses, so
  the gate silently compares a `major.minor` version as a decimal. The identity
  metrics (`gpuName`, `pciBusId`, `driverVersion`, `builtCudaArch`) are the same
  class. This is a stronger duty than the unregistered rule above and does not
  overlap it: that rule can only say "nothing has registered this name", which is
  advice about a name; this one makes the registry state what the values ARE, so
  the author is told what to gate on instead. Diagnostic INTEGERS may stay
  unregistered — clockprobe's nine per-reason counters deliberately do — because
  no gate on one fails closed; the unregistered rule still asks for registration
  the moment somebody gates on one, which is the right moment. Asserted against
  real captured runner output in
  `pkg/runner.TestParse_RealClockProbeOutputIsRegistered`, because an audit of
  the source missed nine metrics and a capture cannot.
- **A kind that ignores `BURNIN_DURATION_SECONDS` says so, in code.** Every
  runner here reads it except `compute-smoke`, whose GEMM finishes in
  milliseconds however long it is asked for — while a shipped sample requested
  120 s, so "node acceptance" burned nothing in and nothing said so.
  `TestKind.BurstOnly()` is that declaration, and two guards keep behaviour, kind
  doc and sample from drifting apart again: `runners/pins_test.go` refuses a
  non-burst kind whose runner source never reads the variable AND a burst kind
  whose source does, and `api/v1alpha1/samples_test.go` refuses a sample that
  sets `durationSeconds` on a burst kind. The resolution is deliberately "stop
  pretending" rather than "add a loop": a burst proves the arch-correct
  instruction path executed and got the right answer, which a longer run does not
  strengthen, and looping it would duplicate `thermal-soak`, `gpu-burn` and
  `clockprobe` while costing the fleet its one cheap correctness gate. One kind,
  one job. `durationSeconds` still bounds the POD deadline for a burst kind; it
  just does not decide how long the test runs.
- **`Error` is not `Fail`.** An infrastructure error (image pull, scheduling)
  must stay distinguishable from a hardware verdict. Do not collapse them. The
  consequence with teeth is the retry rule: `retryOnErrorLimit` re-runs an
  `Error` and nothing else. A `Fail` — whether from a non-zero exit or from a
  threshold violation on a clean exit, which reach the decision by different
  routes — settles the test where it happened, with retry budget left unspent,
  because re-running a measurement until it comes out clean launders a hardware
  fault into an acceptance. There is exactly ONE place that decision is made
  (`completeAttempt`), shared by Node and Pair scope; keep it that way.
  `TestAttempt.Trigger` records why every attempt happened, so the rule is
  auditable from a stored result long after the run.
- **A Pair verdict is about the LINK.** Pair scope runs two pods — a server on
  the first target, a client on the second, rendezvous'd through a headless
  Service (`internal/controller/pair.go`) — and produces exactly ONE `TestResult`
  naming BOTH nodes. Never split it per node: a point-to-point measurement is a
  property of the link, and attributing it to one endpoint sends an engineer to
  replace the wrong part. Three rules hold it together and none may be relaxed:
  the **client is the deciding side** (it is where perftest and nccl-tests
  report, so its metrics win any key both ends emit, and the pair settles when
  it terminates rather than waiting for a server that may legitimately linger);
  verdict precedence is **`Error` > `Fail` > `Skip` > `Pass`**, because a
  machinery failure on either end means the link was never measured and a
  client's "connection refused" is then an artifact of its peer, not evidence
  about the fabric — recording that as `Fail` would permanently indict a link
  over an unpullable image, and `Fail` is the one phase that is never retried;
  and a **server that exits 0 before the client ever started is an `Error`**,
  never a pass, because no traffic crossed the link.
- **A pair is one unit of load, and it costs two slots.** `maxConcurrentNodes`
  counts NODES, and a pair holds both of its nodes for the whole test, so it is
  admitted only when two slots are free — and never at the default cap of 1,
  which the run refuses at start with that explanation. Admission is checked
  once, when the unit starts: the client is part of an already-committed unit
  and is not re-checked, or a lowered cap would strand a running server against
  a peer that never arrives. `spec.cancel` with `CancelImmediate` is the tool
  for getting load off the floor now.
- **Cordoning follows the wave, not the target list.** A node is cordoned
  immediately before the run puts load on it and released once it is no longer
  holding any, so a run's footprint on the fleet tracks `maxConcurrentNodes`
  instead of the size of its target list. Both nodes of a pair are cordoned
  together, before either pod exists. Cordoning every target up front took a
  two-node cluster entirely out of scheduling and hollowed out the interlock,
  whose whole premise is that only N nodes are occupied at once.
- **What the run FOUND is captured once, at start, and never re-derived.**
  Because a node is cordoned and released per wave, "was this node already
  cordoned when I got here?" would otherwise be re-asked several times per run
  and answered from whatever `spec.unschedulable` said at that instant — after
  the run already had a footprint. Anything the run could not attribute to
  itself (its own hold seen through an annotation it no longer recognises after
  a manager restart; an operator cordoning a node the burn-in was already
  holding, which on an unschedulable node is an invisible no-op) was then
  recorded as pre-existing and made PERMANENT at teardown: a stranded cordon
  signed off as intentional. `status.priorUnschedulable` is that record, written
  before the first cordon and never overwritten, and it — not the node's
  annotation — decides what the release restores. The node annotation stays as
  the account a human reads off the node and as the fallback for a run that
  started under an older operator. The deliberate consequence: a cordon placed
  on a target after the run started is not adopted and is undone at teardown.
  That is the right way round, since a cordon undone is visible and one command
  to redo, and a node the fleet silently loses has nothing left in the cluster
  that knows it was taken.
- **A pod may only be destroyed once the status justifying its destruction is
  readable from the apiserver.** The rule and its consequence in `terminate` are
  documented at that function; what matters here is that it is now ASSERTED
  rather than argued. `test/envtest/invariants_test.go` runs the scenario
  against a real apiserver and makes the terminal status write lose a REAL
  resourceVersion race — the competing write is performed and the apiserver
  returns the 409 — then checks, at every pod deletion, that the run's durable
  status already justified it. Reproducing that needs a real apiserver, which is
  why the assertion did not exist while the only harness was the fake client.
  Read it before reordering any write in `terminate` or `reconcileTerminal`.
  The condition it depends on is the shipped Deployment's own shape: `replicas:
  1` under the default RollingUpdate has `maxSurge` 1, so an apply starts the
  second manager before the first is gone.
- **Runner pods must tolerate the operator's own cordon.** The run cordons its
  target, which the node controller expresses as
  `node.kubernetes.io/unschedulable:NoSchedule`; a pod that does not tolerate it
  can never be scheduled onto the node the cordon was placed for. `podForTest`
  adds that toleration unconditionally — it is scoped to that one taint by key
  and is emphatically not a blanket toleration. Removing it deadlocks every test
  on every node.
- **Host access is named, narrow, and never implicit.** `spec.runner.hostPaths`
  is the ONLY way a runner pod reaches the node's filesystem: a test that
  declares nothing gets a pod with no volumes at all, and there is no default
  `/dev`, no convenience `/sys`, nothing. The shape is deliberately a curated
  `HostPathMount{path, mountPath, readOnly, type}` rather than
  `[]corev1.Volume` + `[]corev1.VolumeMount`, because every mount a burn-in
  runner has ever needed is a host device or a host log path, and the curated
  form states that intent where a general volume list would bury it. It also
  refuses what the general form cannot: `type` admits only the assert-only
  hostPath kinds, so a BurnInTest can never ask the kubelet to CREATE a path —
  an operator sent to measure a fleet must not mutate it. Widen to the general
  form only for a case the narrow one genuinely cannot express, and say which.
  Four rules hold it up: `readOnly` **defaults to true** and is a pointer so
  that "unset" is distinguishable from "false" (a privilege grant's default has
  to fall towards the harmless form, including for objects the apiserver never
  defaulted); the mounts are built in `podForTest`, which is the one place a
  runner pod is constructed, so Node scope and BOTH pods of a Pair get identical
  host access by construction — a link measured from one end is not measured;
  the mount spec rides in the **pinned plan** like the rest of the test spec, so
  editing a BurnInTest mid-run cannot widen what an in-flight attempt takes off
  the host; and `privileged: true` is NOT a substitute — measured on this
  project's own nodes, a privileged `hostNetwork` pod was missing every
  `/dev/infiniband/uverbs*` device the host had, which is exactly what
  `ibv_create_cq` opens.
- **A runner without its mount degrades honestly; keep it that way.** The two
  cases that motivated `hostPaths` are the worked examples. Without
  `/dev/infiniband` the fabric runners fail loudly at `Couldn't create CQ`
  rather than reporting a number, and without `/dev/kmsg` host-health reports
  `xid_source=none` and OMITS `xidEvents` rather than printing a zero it never
  measured — so a gate on it fails the node instead of certifying it. A runner
  must never paper over a missing mount with a fabricated measurement; a node
  condemned for a reading nobody took is recoverable, and a fleet certified
  clean by a fabricated zero is not.
- Small, focused PRs, with tests. CI must be green.

### Runner images

- A runner's contract is its **exit code plus `key=value` metrics on stdout**.
  Exit 0 = pass, 1 = fail, 2 = skip (not applicable to this hardware), **3 =
  error** (the runner could not measure; the hardware is unjudged). Anything
  else is also an error — `pkg/runner.VerdictFor` maps every unrecognised code
  there — but 3 is the code a runner author writes deliberately.
  **Exit 1 is the expensive one, and it is the one runners get wrong.** It is a
  hardware verdict, and the operator never retries a `Fail`, so it permanently
  indicts a node with the retry budget unspent. It belongs ONLY to something the
  run actually measured about the part. Everything that stopped the measurement
  from happening — no device visible, an image with no kernel for this part, a
  driver or runtime error, a failure to set up the comparison — is exit 3.
  `compute-smoke` shipped `v0.1.0` reporting all three of those as exit 1; that
  is the shape of the bug to look for. The skip path matters for the same
  reason: a node that cannot run a test must skip cleanly, not report a failure.
- **A branch no available hardware can reach still gets exercised — three ways,
  and the strongest one is on real silicon.** `compute-smoke`'s exit 2 fires only
  on a part that is not CC 12.0/12.1, this project has none, and the assumed
  route (DCGM being unsupported on GB10) turned out not to exist. So: the
  decision is lifted into a CUDA-free header and unit-tested exhaustively
  (`arch_match.h`'s `scopeOf`, driven by `arch_match_test.cc` under `make test`);
  the sentinel is asserted end to end in Go from CAPTURED output, through the
  parser and through the reconciler with a threshold attached, so a skipped
  run's missing metric is proven not to fail closed into a hardware verdict; and
  a **compile-time** fault-injection macro builds the same source with the same
  toolchain and tells it the device reports a capability it does not have, which
  is how `FP4_GEMM_SKIP` / exit 2 was finally observed on a real GB10.
  Fault injection has three rules and none may be relaxed: it is COMPILE-TIME so
  no environment variable can reach it; no Dockerfile may define it, which
  `runners/pins_test.go` fails the build over, because an image built with one
  would report a fabricated "acceptance does not apply" for a whole fleet with
  the retry budget unspent; and such a build must ANNOUNCE itself in a
  `key=value` line (`forced_compute_cap=`), so the marker rides into the stored
  result and the delivered envelope and the verdict stays self-identifying for as
  long as it exists.
- **A compile-time `#else` fallback is compiled by the build, not by hope.**
  `compute-smoke`'s block-scaled-MMA guard had an `#else` no build had ever
  compiled, so an error in it would have surfaced first on a fleet. The arch flag
  cannot reach it — measured on CUTLASS v4.6.1 / CUDA 13.0.1, both
  `CUTLASS_ARCH_MMA_SM120_SUPPORTED` and `..._SM121_SUPPORTED` are defined even
  for `-arch=sm_80` — so the Dockerfile forces it with a macro and asserts the
  message is in the artefact. The second half is the one with teeth: the SHIPPED
  binary must NOT contain that message, which is the proof the real kernel is in
  there. Without it a CUTLASS bump that stopped defining those macros would
  publish an immutable tag reporting every node `Error`.
- **Environment the operator injects.** `BURNIN_DURATION_SECONDS` and
  `BURNIN_ATTEMPT` at every scope; a **Pair**-scope pod additionally gets:

  | Variable | Value |
  |---|---|
  | `BURNIN_ROLE` | `server` or `client` |
  | `BURNIN_PEER_HOST` | the peer's DNS name, `<peer-role>.<service>.<ns>.svc` |
  | `BURNIN_PEER_NODE` | the peer's node name (for messages, never for addressing) |

  That is the whole rendezvous contract: which end of the link this is, and
  where the other end answers. A fabric runner image needs nothing else — one
  image is the server on one node and the client on the other, and it never
  learns anything about Kubernetes. The peer host is deliberately **not**
  qualified to `cluster.local`, because a cluster's DNS domain is configurable
  and hard-coding the default would break the rendezvous as a connection error
  that looks like a bad link.
- **A Pair server should declare `spec.runner.readinessProbe`.** The operator
  will not start the client until the server pod is Ready, but without a probe
  "Ready" only means the container started — and an `ib_write_bw` or `nccl`
  client that connects before its server has bound its socket dies with a
  connection error that reads as a fabric fault. A `tcpSocket` probe on the
  runner's own port converts the gate into a real statement about a listener.
  **But the probe SPENDS a connection, and that is not free.** TCP admits no
  way to prove a listener is up other than connecting to it, so the probe is a
  real client: the server accepts it, the probe closes it, and a server that
  accepts a bounded number of connections has served the probe instead of the
  client. The gate meant to guarantee the listener is up is then the thing that
  consumes it, and the client arrives at a closed port a moment after the
  operator was told the server was ready. `ib_write_bw` and the nccl-tests
  server both accept a bounded count. Probe a port the measurement does not use,
  or make the server re-accept; the e2e's own `nc -l` server had exactly this
  bug and the operator correctly settled the run `Error` — server exit 0 with no
  traffic across the link is never a pass.
- **`n/a` is a reserved metric value** (case-insensitive), and the only way a
  runner declares a counter unmeasurable on the hardware in front of it. Emit it
  where you asked and the hardware has nothing to report; omit the key where the
  probe itself failed. Never emit a `0` you did not measure.
- Runner images are published **manually** via the `publish-runner` workflow
  (`workflow_dispatch`), never automatically. A runner image is executed by a
  node's readiness gate, so a bad tag degrades a whole fleet at once. Publish
  only after the kernel has been verified on real hardware.
- Published tags are **immutable**. Never republish a tag a gate pins; cut a new
  version. Silently changing a tag changes every node's verdict with no audit
  trail.
- **Every upstream is pinned to a COMMIT, and the build asserts it.** An
  `<UPSTREAM>_REF` (a tag) is always accompanied by an `<UPSTREAM>_SHA`, and the
  Dockerfile refuses to build when the clone is not that commit. A tag is a
  mutable pointer; because our published image tag is immutable, a moved
  upstream tag would change what a fleet is measured by with nothing in this
  repo recording it. Resolve a new value with `git ls-remote <repo>
  'refs/tags/<ref>' 'refs/tags/<ref>^{}'`, taking the peeled `^{}` value for an
  annotated tag (DCGM's are annotated) — that is what a shallow clone leaves at
  HEAD. `runners/pins_test.go` fails a `_REF` with no asserted `_SHA`.
- **Sources duplicated across runners are guarded, not merely copied.** Each
  runner is its own Docker build context and `COPY` cannot reach outside one, so
  shared code is physically duplicated. Two contract tests make drift impossible
  to land silently: `runners/thermal-soak/soak_contract_test.go` for the soak
  pair (`soak_core.cuh`, `nvml_dynamic.h`, and clockprobe's subset of that
  header), and `runners/ib-write-bw/fabric_contract_test.go` for the fabric trio.
  The fabric table is TOTAL — every filename present in two or more of
  ib-write-bw/nccl/gpudirect-rdma must be declared either shared (byte-identical)
  or forked (with the reason), and a new duplicate fails until it is. nccl's
  `memlock.go` and `rendezvous.go` are deliberate forks; a copy-paste that
  resynced them would also fail, because a declared fork must really differ.
- Arch targets are deliberate. `sm_121a` emits a cubin for GB10 only, with no
  PTX fallback, so a pass proves the real instruction path ran rather than an
  emulated one. Do not "helpfully" widen an arch target to make a build succeed.
- **A wrong arch does not reliably announce itself, so do not wait for the
  launch to tell you.** `cudaErrorNoKernelImageForDevice` fires only when the
  loader refuses outright — an `sm_121a` cubin on a CC 12.0 part. The other
  direction is not refused at all: SM121 is binary-compatible with SM120, so an
  `sm_120a` image LOADS on a CC 12.1 GB10 and trips a device-side assert inside
  the kernel, reaching the operator as a bare "device-side assert triggered"
  under several hundred lines of CUTLASS template text. `compute-smoke`
  therefore establishes the mismatch from the `BURNIN_CUDA_ARCH` string and the
  reported compute capability, and refuses BEFORE launching. Doing it before
  matters twice over: `cudaErrorAssert` is sticky and poisons the context, and a
  container log is stdout and stderr MERGED while `pkg/runner.Parse` takes
  `Result.Message` from the LAST non-`key=value` line — so kernel spew arriving
  after the diagnosis overwrites it in the stored result. It stays an `Error`
  and never a `Skip`, and an arch string it cannot parse is Unknown rather than
  a mismatch: a runner may only declare what it positively established.

---

## Tracking work

Follow Kubernetes community conventions. Do not leave findings only in chat or
as bare TODOs in code:

- Unfinished or deferred work → a GitHub issue **in this repo**, linked to the
  GEP tracking issue in `glimmer`.
- Bugs, security issues, untested paths → a GitHub issue with a minimal
  reproduction, written so someone without the session's context can act on it.
- New workstreams or significant design changes → a GEP in `glimmer`
  (`docs/geps/`), not a design document here.

---

## Updating this file

Keep it current. When a convention, invariant, or tool changes, update the
relevant section so the next session starts with accurate context.

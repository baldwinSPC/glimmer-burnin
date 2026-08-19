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

**This repository must never import its control-plane consumer.**

CI fails any PR adding an import of a control plane's module path. There is
no exception, including "just for a type" or "just in a test".

- Dependency direction is one-way: a consuming control plane MAY depend on this
  project; this project MUST NEVER depend on one.
- Integration happens through the `BurnInSink` export contract — a delivery
  envelope over a webhook — not through shared code.
- This is what allows the repo to be open-sourced and adopted independently. If
  you need something from a control plane that consumes this project, the
  answer is to widen the contract, not to add an import.

If a task appears to require reaching into a consumer's own codebase, stop and
raise it rather than working around the guard.

---

## Design tracking

This project is developed alongside a private planning repository that is not
readable by public contributors, so the rules that matter for working in this
repo are stated here directly rather than by reference:

- **Implementation issues are filed here**, in the repo where the code lands,
  and linked to the workstream's tracking issue in the planning repo when one
  exists.
- **Significant design changes are written up as a design proposal (a GEP,
  `docs/geps/NNNN-name.md`) and tracked outside this repo**, even when the code
  lands here — not as a design document in this repo. This project's own design
  is GEP-0178.
- The dependency and license rules in this file apply most strictly to this
  repo, because it is the one that gets published.

---

## Licensing rules (strict)

- **The allow-list is `hack/licenses-allowed.txt`, and it is ENFORCED, not
  described.** `go run ./hack/licensecheck` walks the whole Go module graph and
  CI blocks on it. This paragraph is for a human; that file is what decides.
- Permissive only: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, and the
  public-domain equivalents (0BSD, Unlicense, CC0-1.0). ISC is on the list
  because the check found the project had ALREADY been shipping it — go-spew
  arrives through `k8s.io/apimachinery/pkg/util/diff` and is linked into the
  operator binary — while this file listed four licences and had never described
  what the project actually distributes. That is the shape of the problem a
  prose-only policy has.
- **Never** add a dependency under a copyleft license — GPL, AGPL, LGPL, SSPL,
  MPL, Sleepycat, CDDL or EPL. This is not a preference; a copyleft dependency
  would make the project unpublishable. There is no exception process: if the
  dependency you need is copyleft, the answer is a different dependency, or an
  interface this project defines and the consumer implements.
- Verify a new dependency's license **in the PR description**. If you cannot
  verify it, ask before touching `go.mod`. An UNCLASSIFIABLE licence fails the
  check exactly like a copyleft one — an unidentified licence is the one most
  likely to be a problem, so it must never default to acceptable.
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

make licenses        # the licence policy, over the whole module graph
make vulncheck       # govulncheck: only vulnerabilities in reachable code
make supply-chain    # both of the above
```

`golangci-lint` is configured in `.golangci.yml` and runs in CI. The starting set
is deliberately small — `govet`, `ineffassign`, `unused`, `staticcheck` (SA only)
— because a linter enabled with everything gets disabled in anger and then
protects nothing. Turning another one on is a small PR that fixes what it finds;
`.golangci.yml` records which are next and why each is not on yet.

**The Go toolchain is pinned in `go.mod`, and the `go` and `toolchain` directives
say different things on purpose.** `go` is the language version a CONSUMER of
`pkg/verdict`, `pkg/runner`, `pkg/contract` or `pkg/report` must have, and
raising it forces every one of them to upgrade. `toolchain` is what this project
BUILDS with, and it is what `govulncheck`'s verdict is about — when the check was
first added it found ten reachable standard-library vulnerabilities, including an
auth bypass in `crypto/x509` that `internal/sink`'s webhook TLS path reaches,
purely because the toolchain was stale. Bump `toolchain` freely; bump `go` only
with a reason.

After changing anything in `api/v1alpha1`, run `make generate manifests` and
**commit the results** (`zz_generated.deepcopy.go`, `config/crd/*`,
`config/rbac/*`). CI checks for drift.

CI (`.github/workflows/ci.yml`) runs build, vet, gofmt, test, the
standalone-import guard, the CRD drift check, and the supply-chain job (licence
policy, `govulncheck`, `golangci-lint`). All of them are **blocking** — nothing
there is informational.

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
                     pods, Pair + Group rendezvous, segmented soaks, delivery),
                     BurnInSchedule, NodeFingerprint
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

Public packages — importable by other projects:

```
pkg/contract/        versioned delivery envelope + metric-name registry
                     NO Kubernetes dependency. Measured: 0 k8s modules.
pkg/runner/          runner exit-code + key=value stdout parsing
                     NO Kubernetes dependency. Measured: 0 k8s modules.
pkg/verdict/         threshold evaluation, no I/O
                     NO Kubernetes dependency. Measured: 0 k8s modules.
pkg/group/           folds N ranks' reports into the ONE result a collective is
                     judged on: the verdict precedence and the per-metric
                     Combination election
                     NO Kubernetes dependency. Measured: 0 k8s modules.
```

Shared by both dispatchers, but Kubernetes-coupled (see the ledger below):

```
pkg/plan/            profile entry -> executions: variant expansion, the axis
                     vocabulary, and the refusal of an axis a runner could
                     never receive
```

**`pkg/verdict` was not Kubernetes-free until #274 was fixed, and this file has
said so in both directions.** `Evaluate` took `[]burninv1alpha1.Threshold`, so
the CRD package was in its signature and travelled with it — a fresh module
importing each package alone measured 0, 0 and 9 k8s modules.

The fix was to move the ACCEPTANCE VOCABULARY — `TestKind`, `Comparison`,
`Applicability`, `Threshold` — into `pkg/contract`, with `api/v1alpha1` keeping
every name as a type ALIAS. A Go alias is the same type, so the CRD wire format
is byte-identical (the regenerated manifests showed zero drift), controller-gen
reads the kubebuilder markers where they now live, and no caller had to change.

It was cheap only because it was done early: the only external consumer at the
time imported `pkg/contract` alone, so there were zero external callers of
`Evaluate` to break. That window would have closed the moment Path A adopted
`pkg/localrun`.

`pkg/localrun`, `pkg/runnerimages` and `pkg/plan` are STILL coupled, through the
EXECUTION vocabulary rather than the acceptance one — `BurnInTestSpec`,
`RunnerSpec`, `VendorImage`, `TestVariant`. Freeing them would make the CRD a
thin Kubernetes wrapper over a neutral core, which is a larger decision left to
the GEP-0178 amendment and to the moment Path A can say what it actually needs.
`pkg/localrun.PlannedTest` is an ALIAS for `pkg/plan.Test`, so those two entries
describe one coupling and must leave the ledger together.

**`pkg/plan` is profile resolution, and it exists because the two dispatchers
disagreed about it.** Variant expansion lived unexported in the reconciler, so
`burnin run` SILENTLY DROPPED `spec.tests[].variants` — a four-cell precision
sweep ran as one execution of the parent test, with none of the cell's
thresholds applied and no `BURNIN_VARIANT_*` reaching the runner, and nothing in
the output said a cell was missing. Both dispatchers now call
`plan.ExpandVariants`, and `plan.RefuseUnreachableAxes` / `plan.VariantEnv` are
shared for the same reason: WHICH axes reach a runner and UNDER WHAT NAME is one
decision, and a second implementation of it is the drift this package ends. Each
dispatcher still renders the result in its own shape (a `corev1.EnvVar` list
in-cluster, a map for `docker run -e` on bare metal), which is the part that
genuinely differs.

The claim is no longer prose: `hack/invariants/kubernetesfree_test.go` is a
two-sided ledger — a clean package that acquires a Kubernetes dependency fails,
and a coupled one that becomes clean fails until its entry is deleted. Nothing
was enforcing this when the original wrong claim survived a release.

`pkg/verdict` and `pkg/runner` are public because a separate, pre-Kubernetes
burn-in path runs the same runner images: if the two dispatchers derived
different metrics or different verdicts from identical runner output, they would
disagree about the same hardware. One brain, two dispatchers.

---

## Conventions

> **The invariants below are also published, for humans, at
> [`docs/dev/invariants.md`](docs/dev/invariants.md).** That page is the
> canonical PUBLIC text: a stranger evaluating this project should not have to
> read a file addressed to AI assistants to find the design rationale.
>
> What is here is the same set, written as instructions to a session. When one
> changes, change both — `hack/invariants` fails if the public page loses an
> invariant this file still asserts, because two copies that drift are two
> different rules and the one somebody reads is whichever they found first.


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
- **A metric declares HOW IT COMBINES, because a soak runs in segments.**
  `pkg/contract.Metric.Aggregation` is `Sum`, `Min`, `Max` or `Last`,
  it is required on every registered metric (the registry self-consistency test
  refuses `Unspecified`, same discipline as `ThresholdUse`), and
  `internal/controller.foldMetrics` is what consumes it, once per clean segment.
  It exists so the answer lives beside the NAME rather than in a
  per-metric switch inside the reconciler, which is exactly the contract-shaped
  knowledge this file keeps out of it. The rule that is wrong in a way nothing
  else catches is LIFETIME versus WINDOWED, and host-health has both: `xidEvents`
  and `pcieReplayErrors` are differenced over the window and `Sum`, while
  `eccErrors` and `remappedRows` are NVML aggregates since reset and are `Last` —
  summing those multiplies them by the number of windows, so a node with four
  remapped rows reads as twelve after a three-segment soak and is condemned for
  damage it does not have. A floor takes `Min` because the WORST window describes
  the part (a soak holding 83% of rated clock for eleven hours and 40% for one is
  a part that dropped to 40%; anything else certifies the drop away), and a
  ceiling takes `Max` for the same reason inverted. An UNREGISTERED name
  aggregates `Last`: inventing `Sum` semantics for a name nothing declared would
  be this operator deciding what somebody else's measurement means, silently.
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
- **A runner that needs the host driver DECLARES it, and a runner that speaks
  only Pair REFUSES Group.** Two rules, one class of failure: a runner answering
  a question it was not asked, with the one verdict that certifies. (a) The GB10
  cluster's CDI spec declares a hook its host toolkit cannot implement, so every
  pod injected through CDI dies at createContainer — and the burn-in runners
  survive only because containerd falls back to the legacy runtime, which injects
  from `NVIDIA_VISIBLE_DEVICES`/`NVIDIA_DRIVER_CAPABILITIES` in the IMAGE
  environment. That is a property of these images, not of the host, so the fleet
  is one Dockerfile away from a cluster-wide outage; `runners/pins_test.go`
  asserts the correspondence, and the two runners that touch no accelerator
  (`ib-write-bw` verbs, `memory-stress` host RAM) are exempt BY NAME with their
  reason. (b) Every fabric runner branches on `BURNIN_ROLE` and reads its absence
  as Node scope — but Group scope sets `BURNIN_RANK`/`BURNIN_NRANKS` and
  deliberately NOT `BURNIN_ROLE`, so all three would have declared a `_SKIP` for
  a collective that never ran. They now exit 3, BEFORE any hardware inspection,
  because "this image does not speak the Group rendezvous" is true whatever is in
  the box. The operator refuses the same case at plan time, which is the half
  that protects a fleet running already-published tags: those are immutable and
  will skip forever. `nccl` NOW SPEAKS the contract — rank 0 serves the
  ncclUniqueId to the other N-1 — so it is on `groupCapableKinds` in
  `internal/controller/plan.go` and a Group test may use its default image.
  `ib-write-bw` and `gpudirect-rdma` refuse and always will, because a
  point-to-point RDMA write and a GPUDirect peer-memory check have no N-rank
  form. `runners/pins_test.go` fails if that list and the runner sources drift
  EITHER way, keyed on `BURNIN_ROOT_HOST` — a first attempt keyed on
  `BURNIN_RANK`/`BURNIN_NRANKS` failed immediately, because the refusing runners
  read exactly those in order to say no. READING A VARIABLE TO REFUSE IS NOT
  IMPLEMENTING THE CONTRACT.
- **An artifact is evidence ABOUT a verdict, never part of one.** A runner
  returns non-metric evidence — dcgmi's own JSON, a captured topology — in a
  fenced block on stdout (`-----BEGIN BURNIN ARTIFACT <name> <mediaType>-----`),
  and `pkg/runner` lifts those lines out BEFORE the metric scanner sees them.
  That ordering is the whole feature: a dcgmi document is full of `"key": value`
  lines, several of which parse, so without it a runner returning evidence would
  silently rewrite its own measurements. Three consequences follow and none may
  be relaxed. A line that OPENS a fence opens one even when its header does not
  parse — the artifact is refused and its payload consumed, because treating a
  three-field header (`has space.json`) as ordinary text handed the entire
  payload it was announcing to the scanner, which is the defect the fence exists
  to prevent, reached by the route nobody was watching. A failure to STORE an
  artifact changes no phase: the payload goes to a run-owned ConfigMap, and a
  full quota, a deleted namespace or an unusable name must never become a
  hardware verdict — every such failure is recorded as a dropped `ArtifactRef`,
  which is also why a refusal is REPORTED rather than swallowed ("produced no
  evidence" and "the evidence was too large" are different facts, and only one
  sends an engineer to a node). And an artifact NAME is the least trusted input
  this operator has: it is refused, not sanitised, when it is not a legal
  ConfigMap key, because a '/' makes the apiserver reject the whole object —
  one runner's malformed name discarding every other test's evidence — while
  rewriting `a/b` to `a_b` would collide two names onto one key and point a ref
  at the wrong payload under the right digest. Payloads that are not valid UTF-8
  go to `BinaryData`, and the reason is measured rather than assumed: the
  apiserver does NOT reject them in `Data`, it stores them and mangles them to
  U+FFFD on the way out to any JSON consumer — kubectl included — so the
  evidence looks present and only the ref's digest reveals otherwise.
  `test/envtest/artifacts_test.go` asserts that through a JSON client, because
  the fake client stores whatever bytes it is handed and reports green.
- **A container that never started has no stdout, so the KUBELET'S message is
  the only evidence there is.** `podOutcome` returns it as `detail` alongside the
  reason and `harvestPod` appends it to the stored message. Dropping it made a
  cluster-wide outage undiagnosable (#52): every GPU pod on the fleet was dying
  at `createContainer` because the device plugin's CDI spec declared a hook the
  host toolkit could not implement, the kubelet put `No help topic for
  'disable-device-node-modification'` in `Terminated.Message` — naming the broken
  hook and therefore the fix — and the operator recorded `runner terminated
  abnormally (exit 128, reason "StartError")`, which is equally true of a typo in
  `spec.runner.image`. It is DETAIL and not a reason: the short machine token
  stays, because that is what the phase logic keys off and what a reader scans
  for. It is clamped and newline-flattened, because the kubelet bounds neither
  and `pkg/runner` takes `Message` from the LAST non-`key=value` line.
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
- **A SEGMENTED SOAK IS ONE VERDICT OVER MANY WINDOWS, and the threshold moves
  to the end.** `spec.soak.segmentSeconds` divides a long test into a sequence of
  shorter pods, each sized to the segment, so an eviction or a reboot costs one
  segment instead of the week — a seven-day soak used to be a single 604,920-second
  pod, and with `retryOnErrorLimit` set the retry started the week over from zero.
  The crux is not the pods, it is that thresholds are NO LONGER evaluated per
  attempt for such a test. `attemptOutcome` applies none of them; the exit code
  alone decides what one window means, and `completeAttempt` applies them ONCE, on
  the last segment, to the persisted `TestResult.AggregatedMetrics`, through
  `gateOutcome` — which is the same function the unsegmented path uses, so a
  consumer cannot tell from a verdict how the test was scheduled. Gating a window
  would fail a week on fifteen minutes AND make the answer depend on how finely
  the soak was sliced, which is a scheduling decision and not a property of the
  part. Five rules hold it up and none may be relaxed. An ERRORED segment
  contributes nothing and does not advance `SegmentsCompleted`, so the retry
  re-runs THE SAME index — a counter that advanced on an error would silently
  shorten the soak by every interruption it suffered, which is the failure this
  feature exists to prevent, reintroduced as bookkeeping. A segment exiting 1
  settles the test `Failed` immediately, because a fault observed at hour six is a
  fault and continuing to burn is hoping. A metric no segment reported stays
  ABSENT from the aggregate and fails closed at the end. `AbortEarly` may fire only
  where the aggregation is monotone IN THE GATED DIRECTION (`Sum` under `LTE`, or
  under `EQ` once the sum has passed the value; `Min` under a `GTE` floor; `Max`
  under an `LTE` ceiling) and only on a `Measurement` violation — a `Min` under a
  ceiling can be pulled back under it by the next window, and a metric nothing has
  reported yet may be reported next window. And it is OPT-IN PER TEST, refused at
  plan time on a burst-only kind, because whether a duration means anything is the
  runner's property: `host-health` clamps its window to 30 seconds, so segmenting
  it would run the same short measurement 672 times and call it a week. Attempt
  history is capped for a segmented result (first + every non-passing + last 16
  passing, remainder in `TruncatedAttempts`), which is safe only because the
  verdict reads the persisted aggregate and never the attempt list — and the LAST
  attempt is never dropped, since `nextAttempt` reads it to decide which segment
  comes next. `docs/soaks.md` is the operator-facing version.
- **A Group verdict is about the COLLECTIVE, and Group needs neither JobSet nor
  OpenMPI.** Group scope runs one rank per target node — target *i* is rank *i*,
  pinned in the plan — all rendezvous'd through one headless Service, producing
  exactly ONE `TestResult` naming EVERY node. Rank 0 is the root: it starts
  first and no other rank is created until it is Ready, which is Pair's gate for
  Pair's reason. The env contract is `BURNIN_RANK`, `BURNIN_NRANKS`,
  `BURNIN_ROOT_HOST`, `BURNIN_ROOT_NODE` and deliberately no rank list;
  `BURNIN_ROLE` is ABSENT so a runner keying off server/client fails loudly
  rather than treating rank 4 as a client. The dependencies were specified in the
  issue and are refused on evidence: gang scheduling solves partial placement
  under contention, and every rank here is pinned by hostname to a node this
  operator has ALREADY admitted and cordoned, so there is no placement decision
  left — while JobSet would be a controller every adopter must install and a
  second owner of pod lifecycle, which would make the "a pod may only be
  destroyed once the status justifying it is readable" invariant unassertable.
  OpenMPI needs a launcher able to start a process on another node, which on
  Kubernetes means an sshd and a key on every accelerator node in the fleet;
  `nccl_pair.cu` already records that argument and it only gets stronger at N>2.
  Three rules differ from Pair and each follows a real difference: EVERY rank is
  waited for (a collective is synchronous, so its ranks finish together, and one
  that has not finished is one the collective is still waiting on — whereas a
  Pair server may legitimately linger); a rank that never reported is NOT a pass,
  because a collective is only measured if every rank took part; and a deadline
  names the ranks that actually hung, because every healthy rank blocks on the
  faulty one and would otherwise be indicted equally. `maxConcurrentNodes` must
  be >= the target count, refused at start.
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
- **A runner pod is never safely evictable, and that is not a knob.** Every
  runner pod carries `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`
  unconditionally, at every scope. It is holding a node this operator CORDONED
  precisely so that nothing else lands there, and moving it discards a
  measurement in progress — which is then recorded as an `Error`, spending retry
  budget on a fault the cluster caused rather than the hardware. Over a
  multi-day soak the cluster's own housekeeping is the most likely thing to end
  a run. `spec.runner.priorityClassName` is a PASSTHROUGH for the different
  mechanism (preemption under contention); the operator never creates a
  PriorityClass, because a cluster-scoped object minted by a test-runner is a
  footgun and a permission it should not hold. The capacity arithmetic a long
  soak implies is in `docs/soaks.md` rather than here.
- **Cordoning follows the wave, not the target list.** A node is cordoned
  immediately before the run puts load on it and released once it is no longer
  holding any, so a run's footprint on the fleet tracks `maxConcurrentNodes`
  instead of the size of its target list. Both nodes of a pair are cordoned
  together, before either pod exists. Cordoning every target up front took a
  two-node cluster entirely out of scheduling and hollowed out the interlock,
  whose whole premise is that only N nodes are occupied at once. **A WAVE CAN BE
  MANY PODS LONG, and "holding" is read from the RESULT and not from a live
  pod.** A segmented soak, a repeat and an error-retry each have a moment between
  pods where nothing is running, and reading the interlock from live pods alone
  made that gap look like released capacity: a second target took the slot and
  the first node's cordon came off halfway through a multi-day acceptance. Both
  halves of that are severe — the soak stops being a soak (at cap 1 over two
  targets each node runs one window then idles for one, and a part allowed to
  cool for as long as it was loaded never reaches the thermal steady state the
  test exists to hold it at), and the cordon is the only thing keeping FOREIGN
  workload off a node under burn-in, since runner pods tolerate
  `node.kubernetes.io/unschedulable` and it therefore never protected against
  this operator to begin with. `holdingNodes` seeds the busy set from every
  non-terminal result's nodes — every node of the unit, so a pair holds both ends
  and a group holds every rank — and it SEEDS rather than replaces, so the
  live-pod count stays the floor and a controller restart still cannot lose track
  of what is already running.
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

**Adding a kind? Read `docs/dev/new-testkind-playbook.md` first.** It is the
ordered checklist — every step names the file to touch and the guard that catches
the mistake — and it is what the `new-runner` skill loads. The rules below are
the invariants; the playbook is the procedure, and it does not restate them.

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
- **A SKIP MUST BE DECLARED, NOT MERELY EXITED.** Exit 2 is honoured as `Skip`
  only when stdout also carries a marker — an upper-case token ending in `_SKIP`
  at the start of a line (`FP4_GEMM_SKIP:`, `NCCL_SKIP:`, `IB_WRITE_BW_SKIP:`).
  Exit 2 with no marker is an **`Error`**. The reason is that an unrecovered Go
  panic exits 2, as does every Go runtime fatal error — out of memory, concurrent
  map writes, stack exhaustion — which no runner can recover from however
  carefully it is written. The camouflage was otherwise perfect: a container log
  merges stdout and stderr, so a crashed runner produces a stack trace with no
  `key=value` line at all, and "no metrics" is the NORMAL shape of a skip. That
  landed a crash on the one phase that is never retried, never affects the run's
  verdict, and reports the node as one the test did not apply to — so a run
  settled `Passed` around hardware nobody measured. This is the same rule as the
  `n/a` sentinel and it is the same sentence: a runner may only declare what it
  positively established, and ABSENCE IS NOT A DECLARATION. It fails towards
  `Error` deliberately — a runner that skips honestly but forgets the marker is
  reported unjudged and retried, which is visible and cheap, while the opposite
  mistake certifies a fleet nobody looked at. Every runner here that can skip
  already prints a marker, and `runners/pins_test.go` fails if one is renamed
  into a form `pkg/runner.DeclaresSkip` no longer recognises. **A marker is not
  always one literal**, and the guard resolves both halves: the soak family
  (`thermal-soak`, `gpu-burn`) prints `markerPrefix + "_SKIP"` from the shared
  `soak_core.cuh`, so the only literal in their sources is the bare suffix and
  neither directory contributed anything to the sweep until it composed them —
  the same blindness that pattern's own comment warns about, reached by a
  different route, on two runners that really do skip (no accelerator visible,
  MIG enabled). A suffix found with no `markerPrefix` declared in the directory
  is itself a failure: a marker nothing in the repository can predict is a
  marker nothing can check.
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

  A **Group**-scope pod gets a different set, and no `BURNIN_ROLE`:

  | Variable | Value |
  |---|---|
  | `BURNIN_RANK` | this pod's 0-based rank |
  | `BURNIN_NRANKS` | how many ranks the collective has |
  | `BURNIN_ROOT_HOST` | rank 0's DNS name, `rank-0.<service>.<ns>.svc` |
  | `BURNIN_ROOT_NODE` | rank 0's node name (for messages, never for addressing) |

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
- **CPU architecture and GPU architecture are ORTHOGONAL, and conflating them
  produces images nobody can use.** `linux/amd64` vs `linux/arm64` is the HOST
  the container runs on; `sm_121a` / `sm_90` / `sm_100` is the GPU gencode. An
  x86 host with an H100 needs `linux/amd64` AND `sm_90`; an amd64 image
  containing only `sm_121a` device code helps nobody, because CC 12.1 is GB10,
  GB10 is a Grace part, and there is no x86 host with one. Every runner image
  and the operator image are built `linux/amd64,linux/arm64` in both `ci.yml`
  and `publish-runner.yml`; amd64 is NATIVE on GitHub's hosted runners and is
  the cheap half of that matrix, arm64 is the QEMU half. Where the gencode
  cannot be one value for both hosts it defaults PER TARGET PLATFORM inside the
  Dockerfile (an empty `CUDA_ARCH` selects that default; an explicit value
  applies to both platforms) — `compute-smoke` arm64 `sm_121a` / amd64
  `sm_120a`, `nccl` arm64 `sm_121` / amd64 `sm_90`.
- Arch targets are deliberate. `sm_121a` emits a cubin for GB10 only, with no
  PTX fallback, so a pass proves the real instruction path ran rather than an
  emulated one. Do not "helpfully" widen an arch target to make a build succeed.
  The amd64 default for that runner is `sm_120a` — a DIFFERENT arch, also with
  no PTX fallback, not a widened one. Where a runner's claim does not depend on
  which instructions ran (the soak family: `clockprobe`, `thermal-soak`,
  `gpu-burn`) it compiles a list of real cubins instead, and that list must
  cover the parts x86 fleets actually run: `sm_80` (A100 through L40S, by
  minor-version binary compatibility), `sm_90` (H100/H200), `sm_100`
  (B200/B300/GB200), `sm_120`, `sm_121`, plus PTX. Note that PTX only JITs
  UPWARD — `compute_121` PTX cannot rescue a CC 10.x part, which is why
  `sm_100` has to be an explicit cubin.
- **A runner may honestly require a gencode parameter, and saying so is better
  than guessing.** `nccl` builds ONE gencode on purpose (the prebuilt libnccl
  carrying them all is 411 MB, paid by every node on every readiness gate), so
  an x86 B200 fleet must rebuild with `--build-arg NCCL_GENCODE=…`. Make that
  parameter obvious in the Dockerfile, the README and the workflow input rather
  than papering over it with a default that silently measures nothing.
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
  workstream's tracking issue in the design-tracking repo when one exists.
- Bugs, security issues, untested paths → a GitHub issue with a minimal
  reproduction, written so someone without the session's context can act on it.
- New workstreams or significant design changes → a GEP (`docs/geps/`) in the
  design-tracking repo, not a design document here.

---

## Updating this file

Keep it current. When a convention, invariant, or tool changes, update the
relevant section so the next session starts with accurate context.

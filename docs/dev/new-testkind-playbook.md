# Adding a TestKind

The ordered checklist for adding a test to the suite, with the guard that
catches each mistake.

Most of what follows is enforced mechanically. That is deliberate: this document
tells you what the build will reject, but the build is the authority. Where a
rule is *not* enforced by a test, it says so — those are the ones to read
twice.

> Status: this page is the first draft, extracted from the design notes and the
> existing runners. It is validated by using it. If a step here turns out to be
> wrong or incomplete while you are following it, fix the page in the same pull
> request as the runner — a checklist nobody corrects is a checklist nobody
> trusts.

---

## 0. Before you write anything

**Decide whether the kind is vendor-neutral or vendor-named.**

A kind named after a measurand (`memory-bw`, `nccl`, `host-health`) is
vendor-neutral: it can have one image per vendor, all emitting the same metric
names, because the thing being measured is the same thing. A kind named after a
vendor suite (`dcgm-diag`, `rvs-diag`) is vendor-named, because the result is
that vendor's opinion about their own hardware and pretending otherwise would
misrepresent whose judgement it is.

Get this wrong and you will either fragment a measurand across three kinds or
imply that two vendors' diagnostic suites are interchangeable.

**Pick the closest existing runner and read it.** They are the real
documentation:

| Shape | Read |
|---|---|
| A standard-library probe reading sysfs or the kernel log | `runners/host-health` |
| A wrapper around an upstream tool | `runners/memory-bw` |
| A long-running load generator | `runners/gpu-burn`, `runners/thermal-soak` |
| A Pair-scope fabric test | `runners/ib-write-bw` |
| A Group-scope collective | `runners/nccl` |
| A vendor suite with its own exit codes | `runners/dcgm-diag` |

---

## 1. The output contract

A runner's entire interface is **an exit code and `key=value` lines on stdout**.

| Exit | Meaning |
|---|---|
| 0 | Pass |
| 1 | Fail — a measurement was taken and it did not meet the bar |
| 2 | Skip — **and only with a declared marker, see below** |
| anything else | Error — no verdict was reached |

Three rules, each of which exists because its absence caused a real incident.

**A skip must be declared.** Exit 2 must be accompanied by a marker matching
`^[A-Z][A-Z0-9_]*_SKIP\b` at the start of a line — `NCCL_SKIP`,
`MEMORY_BW_SKIP`, and so on. Exit 2 *without* one is recorded as an `Error`,
not a skip.

The reason is that exit 2 alone cannot be trusted: a panicking Go runner exits 2
and prints a stack trace, and because a container log merges stdout and stderr,
that is byte-for-byte the shape of a legitimate skip — whose normal form is "no
metrics at all". A skip is a claim about the hardware ("acceptance does not
apply to this node") and a runner may only claim what it positively established.

*Guard:* `TestEverySkipMarkerIsOneTheParserRecognises` in `runners/pins_test.go`.
Note that two runners *compose* their marker from a prefix constant rather than
writing the literal — if you do the same, make sure the guard can still see it.

**`n/a` is a reserved value, and it is a claim about the hardware.** Emitting
`eccErrors=n/a` means "I looked, and this part has no ECC subsystem to report
from". It is matched case-insensitively after trimming, and the parser puts it
in `Result.Unmeasurable` rather than `Result.Metrics`, where a threshold with
`applicability: RequiredIfMeasurable` will report it as not evaluated — never as
a pass.

Absence means something different. A metric that simply never appears still
fails its gate, under every applicability, because a crashed probe must not
become an acceptance. So: emit `n/a` where you asked and the hardware had
nothing to say; **omit the key entirely** where the probe itself failed.

**Never emit a zero you did not measure.** A fabricated zero is the worst
possible output: it certifies a node against a reading nobody took. A node
condemned by a bad reading is recoverable; a fleet certified clean by a
fabricated one is not.

The last line that is not a `key=value` pair becomes the result's message, so
print your marker and any explanation *after* the metrics.

---

## 2. Duration semantics

Declare one of these, and be honest about which:

- **Honours the duration.** Read `BURNIN_DURATION_SECONDS` and run for about
  that long. Apply a floor if a very short run is meaningless, and say so in the
  output rather than silently substituting.
- **Treats it as a budget.** The work takes as long as it takes; the duration
  bounds it. `memory-bw` does this.
- **Burst-only.** The test is a single measurement and duration is meaningless
  to it. `compute-smoke` is declared this way, and the API refuses a
  `durationSeconds` on a burst-only kind.

*Guard:* `TestDurationIsHonouredOrDeclaredBurstOnly` refuses a kind that claims
to honour a duration whose runner never reads the variable.

If your runner can be killed by its deadline, emit progress periodically so a
killed run still leaves evidence — cumulative re-emission, relying on
last-occurrence-wins, is the established pattern.

---

## 3. Metrics

**The grammar.** A metric name is lowerCamelCase, starts with a lower-case
letter, and is alphanumeric. A dimensional metric ends in a registered unit
suffix; a dimensionless counter does not.

Registered suffixes: `Gbps`, `GBs`, `MBs`, `Us`, `Ms`, `S`, `C`, `W`, `Pct`,
`Tflops`, `MHz`.

**The casing trap.** Suffix matching is case-sensitive and the list is ordered
longest-first. A runner key `foo_bandwidth_gbs` normalises to `…Gbs`, which is
**not** the registered `GBs`, so it will not be recognised as carrying a unit.
This has bitten before; `memory-bw` works around it by printing some values
already-canonical. When in doubt, write a test that asserts the canonical name
your key produces.

**Register the name** in `pkg/contract/metrics.go` with its unit, a description,
and a `ThresholdUse`:

- `Acceptance` — a number a threshold can meaningfully gate on.
- `Evidence` — a number that identifies or explains but must not be gated. A
  nameplate constant (`ratedBoostClockMHz`) is evidence: gating on it fails a
  heterogeneous fleet for no hardware reason. So is any label-valued metric
  (`"true"`, `"unknown"`, a driver version): a gate on one fails closed on every
  node forever, and the failure reads as a hardware verdict. Registering it as
  `Evidence` is what lets the linter refuse the gate while its author is still
  there to fix it.

**Add the alias entry** in `pkg/runner/parse.go` if your runner's key is not
already the canonical name, and **write the parser test**. Aliases are needed
when the key names no measurand (`tflops`), when the unit casing differs, or
when the tool's vocabulary differs from ours (`errors` → `miscompares`).

Two keys must never alias to the same canonical name — parsing is
last-occurrence-wins and one would silently overwrite the other.

*Guards:* `TestAliasTargetsAreRegistered`, `TestAliasTargetsAreUniqueWithinAKind`,
`TestAliasEntriesAreNecessary`, `TestParse_UnitCasingTrapsAreAliased`.

---

## 4. Environment

The operator injects these and nothing else:

| Variable | Scope | Value |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | all | the resolved duration |
| `BURNIN_ATTEMPT` | all | 1-based attempt number |
| `BURNIN_ROLE` | Pair | `server` or `client` |
| `BURNIN_PEER_HOST` | Pair | the peer's DNS name |
| `BURNIN_PEER_NODE` | Pair | the peer's node name — **messages only, never addressing** |
| `BURNIN_RANK` | Group | this pod's 0-based rank |
| `BURNIN_NRANKS` | Group | how many ranks the collective has |
| `BURNIN_ROOT_HOST` | Group | rank 0's DNS name |
| `BURNIN_ROOT_NODE` | Group | rank 0's node name — messages only |

`BURNIN_ROLE` is deliberately **absent** at Group scope, so a Pair-shaped runner
fails loudly rather than treating rank 4 of eleven as a client. A runner that
supports both scopes must branch on which set is present, and refuse when
neither is.

Peer and root hostnames are deliberately not qualified with a cluster DNS
domain, because that domain is configurable and hard-coding the default would
break the rendezvous as a connection error that looks like a bad link.

A Pair server should declare `spec.runner.readinessProbe` — without one,
"Ready" only means the container started, and a client that connects before its
server has bound its socket dies with an error that reads as a fabric fault.

*Guard:* `TestGroupCapableRunnersReallyReadTheGroupContract`.

---

## 5. Host access

`spec.runner.hostPaths` is the only way a runner reaches the node's filesystem.
A test that declares nothing gets a pod with no volumes at all — there is no
default `/dev`, no convenience `/sys`.

`readOnly` defaults to **true**. The `type` field admits only assert-only kinds
(`Directory`, `File`, `Socket`, `CharDevice`, `BlockDevice`) — a BurnInTest can
never ask the kubelet to *create* a path, because an operator sent to measure a
fleet must not mutate it.

**Degrade honestly when a mount is missing.** This is the rule most likely to be
got wrong under time pressure. Without `/dev/infiniband` the fabric runners fail
loudly at connection setup rather than reporting a number. Without `/dev/kmsg`,
`host-health` reports its source as unavailable and **omits** the affected
counter rather than printing a zero — so a gate on it fails the node instead of
certifying it.

`privileged: true` is not a substitute for a mount, and has been measured not to
be: a privileged host-network pod was missing every `/dev/infiniband/uverbs*`
device the host had.

---

## 6. The Dockerfile

Patterns every runner follows:

- **Multi-stage**, so a toolchain is used at build time only.
- **Pin every upstream to a commit SHA**, not a tag. *Guards:*
  `TestEveryUpstreamRefIsPinnedToACommit`, `TestEveryCloneNamesAPinnedRef`.
- **Assert the licence at build time** — fetch the upstream's LICENSE and grep
  it. Permissive only: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause. A copyleft
  dependency would make the project unpublishable, so this is not a preference.
  Where an upstream is dual-licensed, state which option we consume it under, in
  the Dockerfile and in `NOTICE`.
- **Guard against shipping vendor-redistributable libraries.** The established
  form asserts that `ldd` finds no `libcuda`/`libcudart`/`libnv` in the final
  binary; one runner uses `objdump` on `DT_NEEDED` with an explicit allow-list
  instead, and two add a `find` over the finished filesystem because `ldd`
  cannot distinguish "links a host-injected library" from "ships one". Copy the
  strictest one that fits.
- **Ship `/licenses`** where the upstream requires attribution.
- **State the architecture deliberately.** Do not widen a gencode list to make a
  build succeed: a narrow target is what makes a pass prove the real instruction
  path ran rather than an emulated one. *Guard:*
  `TestNoRunnerCompilesAgainstAnUnresolvedArch`.
- Run as a non-root user, and prefer a distroless final stage unless the runtime
  genuinely needs a fuller image (one runner documents exactly why it cannot).

*Guards also worth knowing:* `TestNoDockerfileDefinesAFaultInjectionMacro`,
`TestFaultInjectionMacrosAnnounceThemselves`,
`TestFromArgsAreDeclaredBeforeTheFirstFrom`,
`TestNoRunnerBinaryCanBeCommittedAtTheRepositoryRoot`.

---

## 7. Wiring it up

- **`NOTICE`** — an entry for every third-party component, its licence, and
  whether it ships. This file is the published licence inventory; a stale entry
  is a real problem, not a documentation nit.
- **`runners/<kind>/README.md`** — what it measures, every metric it emits, its
  skip conditions, and its environment variables.
- **`config/samples/`** — a worked example, thresholdless unless you have
  measured numbers.
- **`pkg/runnerimages`** — a default image entry, **once a published image
  exists**. A kind with no runner source in this repository deliberately has no
  default, so it fails fast at plan time asking for an explicit
  `spec.runner.image` rather than pull-failing on every node.
- **The publish workflow** must offer the new runner. *Guard:*
  `TestPublishWorkflowOffersEveryRunner`.

---

## 8. Publishing

Publishing is a human gate and stays one.

- Images publish by manual dispatch only, never automatically. A runner image is
  executed by a node's readiness gate, so a bad tag degrades a whole fleet at
  once.
- **Publish only after the kernel has been verified on the hardware it
  targets.** Not after CI passes — after it has run on the silicon.
- **Published tags are immutable.** Never re-push a tag a gate pins; cut a new
  version. Silently changing a tag changes every node's verdict with no audit
  trail.

---

## Working with an assistant

Most of this checklist is mechanical, which makes runner work a good fit for a
smaller model — but only once the checklist and the guards exist, which is why
they are prerequisites rather than conveniences.

What has been found to work:

- **A capable general coding model for the runner itself.** A runner is real
  engineering: CUDA or Go, a multi-stage Dockerfile, licence assertions, and
  parser tests. It is not template-filling.
- **A smaller, cheaper model for the strictly-templated parts** — the alias
  table entry and its test, the README metric table, the sample YAML. These have
  a worked example two directories away and a guard that fails if they are
  wrong.
- **Escalate anything touching shared semantics.** `pkg/verdict`, the single
  place attempt outcomes are decided, Pair and Group verdict rules, and the
  metric registry's threshold classes are load-bearing for every test in the
  suite, not just yours. A mistake there is not caught by a runner's own tests.
- **A review pass before merge, always.** The guards catch contract violations.
  They do not catch a kernel that measures the wrong thing, a skip condition
  that is too broad, or a threshold that would fail healthy hardware.

The last point generalises past assistants: the guards enforce the contract, not
the science. Whether the test measures what it claims to measure is a human
judgement, and it is the one that matters most.

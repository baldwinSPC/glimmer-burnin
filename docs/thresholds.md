# Authoring thresholds

A threshold is where a site says what "acceptable" means. Everything else in this
operator is machinery for getting a measurement off a node and a verdict back to
a consumer; the threshold is the only part that carries a judgement, and it is
the part this project has got wrong most expensively.

The mistakes have not been subtle bugs. They were numbers copied off a spec
sheet, exact comparisons against continuous measurements, and gates on values
that are not numbers at all. Each of them failed every healthy node in the fleet,
forever, and reported it in the shape of a hardware verdict — so the first
suspicion fell on the hardware, which is the one place there was nothing to find.

Thresholds live on `spec.thresholds` of a `BurnInTest` (or of a variant, which
**replaces** the parent's list rather than merging into it — see
[concepts](./concepts.md)). A test with no thresholds passes on a clean exit:
the runner's exit code is then the only gate.

---

## The rule everything else follows: gates fail closed

**A threshold naming a metric the runner did not emit is a FAILURE, not a pass.**

This is not a defensive default that could be relaxed with better tooling. A
missing measurement must never silently satisfy acceptance, because the cases
that produce one are exactly the cases where nothing was measured: a probe that
crashed, a mount that was not declared, a key somebody renamed. host-health
without its `/dev/kmsg` mount reports `xid_source=none` and **omits**
`xidEvents` rather than printing a zero it never took — so `xidEvents Equal 0`
fails the node instead of certifying it. That is the honest outcome, and it is
recoverable; a fleet certified clean by a fabricated zero is not.

`pkg/verdict.Evaluate` is the whole of it, and it is deliberately pure — no
Kubernetes, no I/O — so this logic is testable in isolation and reachable by a
second dispatcher that runs the same runner images.

| the runner's report | the outcome |
|---|---|
| metric present, satisfies the comparison | pass |
| metric present, violates the comparison | **failure** |
| metric absent from both metrics and the unmeasurable set | **failure**, under every applicability |
| metric present but not parseable as a float64 | **failure** |
| metric parses as `NaN` or `±Inf` | **failure** |
| metric both reported *and* declared unmeasurable | **failure** — the runner's output is self-contradictory |
| metric declared unmeasurable, applicability `Required` | **failure** |
| metric declared unmeasurable, applicability `RequiredIfMeasurable` | **not evaluated** — never a pass |

The non-finite row is the one that looks like fussiness and is not. `strconv`
accepts `"NaN"`, and `NaN` compares false against everything — so it fails a
floor closed, but it *satisfies* `NotEqual` against every value on earth. That is
the single route by which this comparison could have handed out an acceptance
nobody measured.

Every threshold is evaluated, not just up to the first failure. `Outcome.Message`
still names only the first violated gate (a frozen field, so a consumer that
predates the change sees no difference), while `status.results[].violations`
carries all of them. Without that, a node that missed three gates read as a node
that missed one, and an engineer replaced one part per burn-in cycle.

---

## Choosing the comparison

There are four, they are exact, and there is **no epsilon anywhere in this API**.

| comparison | for | notes |
|---|---|---|
| `GreaterThanOrEqual` | a floor on a measurement | the common case for bandwidth, clocks, throughput |
| `LessThanOrEqual` | a ceiling on a measurement | temperature, latency, power |
| both, on one metric | a **band** | this is how a continuous quantity gets a tolerance |
| `Equal` | a dimensionless counter that must be exactly some value | `eccErrors Equal 0` |
| `NotEqual` | a dimensionless counter that must not be some value | rare; the same exactness applies |

**`Equal` and `NotEqual` exist for dimensionless counters.** `eccErrors`,
`throttleEvents`, `miscompares`, `xidEvents`, `nonfiniteCount`: integers where
exactly zero is the only acceptable answer, and where every value a runner can
emit round-trips through `float64` without loss. On those, exact equality is not
a hazard — it is the only correct comparison. **A tolerance around zero ECC
errors is a tolerance for ECC errors.**

**They are the wrong tool for a continuous measurement.**
`sustainedClockPct Equal 83.22` asks a sampled average to reproduce one decimal
string exactly. It will not, so the gate fails on every healthy node forever, and
each failure is reported in the same shape as a hardware verdict. A metric whose
name ends in a registered unit suffix (`Gbps`, `GBs`, `MBs`, `Us`, `Ms`, `S`,
`C`, `W`, `Pct`, `Tflops`, `MHz`) is continuous by construction — gate it with
`GreaterThanOrEqual` and/or `LessThanOrEqual`. The linter says so at authoring
time; see below.

### Why an epsilon was refused

The obvious fix for the paragraph above is a tolerance inside the comparison. It
was considered and rejected for three reasons, and none of them has weakened:

1. **It would silently reinterpret every counter gate already written.**
   `eccErrors Equal 0` is the archetypal gate of this project and it means
   exactly zero. An epsilon changes what it means without changing what it says.
2. **It would not rescue the case that motivates it.** No sampled clock average
   lands within `1e-9` of `83.22` either. The tolerance such an author actually
   needs — half a percentage point, say — is domain knowledge only they have,
   and the API can already express it as `GreaterThanOrEqual` plus
   `LessThanOrEqual`.
3. **It would be invisible in the spec.** Two dispatchers evaluate these
   thresholds (this operator, and a pre-Kubernetes path that runs the same runner
   images). An epsilon lives in the evaluator, so a profile's meaning would depend
   on *which evaluator ran it* rather than on what the profile says — which is the
   exact disagreement `pkg/verdict` exists to prevent.

The guidance is delivered at **authoring time** instead, which is the trade: no
tolerance during evaluation, a linter that tells you before a run rather than
after a node has been condemned.

---

## Applicability: the one legitimate way a gate goes unapplied

`spec.thresholds[].applicability` has two values and defaults to `Required`.

| value | on hardware that cannot produce the metric |
|---|---|
| `Required` (default) | **failure**. An unmeasured quantity has not been shown to be within limits. |
| `RequiredIfMeasurable` | **not evaluated**, and reported as such — in `TestResult.Message`, in `status.results[].notEvaluated`, and in the delivered envelope. |

Three properties make this safe, and none may be relaxed:

- **Absence is not a declaration.** `RequiredIfMeasurable` relaxes *only* the
  case where the runner explicitly emitted the reserved `n/a` sentinel for that
  metric (see the [runner contract](./runner-contract.md)). A runner that simply
  omitted the key — crashed before emitting, probe timed out, key renamed — still
  fails the gate exactly as under `Required`. A profile author cannot invoke this
  relaxation; only a runner that looked at the hardware can.
- **Not evaluated is not a pass.** A `Passed` test whose ECC gate never ran must
  never be indistinguishable from one whose ECC gate ran and was satisfied, which
  is why the fact rides on the result and into the envelope rather than being
  dropped.
- **`Required` remains the default**, so the relaxation has to be asked for. An
  unset or unrecognised applicability is treated as `Required`.

**The worked example is ECC on GB10.** A DGX Spark's unified LPDDR5X has on-die
ECC only and exposes nothing to NVML, so `nvidia-smi` returns `[N/A]` for every
ECC and row-remap field on a perfectly healthy part. `eccErrors Equal 0` used to
fail every node in the fleet. host-health now emits `eccErrors=n/a` — it asked,
and the hardware said there is nothing to count — and the gate is reported as not
evaluated. On an ECC-capable part nothing is relaxed: real counters are reported,
so a non-zero count still fails, and a GPU whose ECC has been *switched off*
reports no counter at all and therefore fails closed. Turning ECC off is not a
way to pass this gate. `config/samples/node-acceptance.yaml` carries this exact
pair of gates with the reasoning in comments.

---

## The linter, and where each severity lands

`pkg/verdict.ValidateThresholds` (and its kind-aware form,
`ValidateThresholdsForKind`) reviews gates for problems that would otherwise
surface at verdict time as an ordinary-looking test outcome. It is **advisory
about evaluation** — it never filters, rewrites or relaxes a threshold, and
`Evaluate` still fails closed regardless — but it is not merely available: three
surfaces run it, because a warning nobody executes is not discoverability.

| surface | when | what it does |
|---|---|---|
| CRD `Pattern` markers on `Threshold.Metric` and `Threshold.Value` | at `kubectl apply`, in the apiserver | refuses a name that is not lowerCamelCase, and a value that is not a finite decimal (`NaN`, `Inf`, `twenty`) — the earliest and cheapest place any of this can be said |
| `buildPlan`, over the resolved profile and then the pinned plan | at run start, **before any node is cordoned** | refuses `Malformed` gates as a config `Error`; records `Unsound` ones on the run's `ThresholdsSound` condition |
| `TestSamplesThresholdsLintClean` | CI, over `config/samples/` | zero tolerance at **both** severities — an unsound gate may be defensible in somebody's profile, but never as an example handed to a newcomer |

### The two severities, and why the split is load-bearing

**`Malformed` refuses the run.** A gate that cannot be evaluated at all fails
closed on every node forever, and a fleet-wide permanent failure reported as a
hardware verdict is precisely the confusion `Error` is kept distinct from `Fail`
to prevent. The refusal happens in `buildPlan`, before a node is cordoned, and it
names every offending threshold across every test rather than the first —
rediscovering the second typo on the next run costs another cordon, another
schedule and another wait.

**`Unsound` does not block.** It is recorded on the `ThresholdsSound` condition
and the run proceeds. The gate does evaluate, and it may well be doing what its
author wanted; refusing on suspicion would let this operator veto profiles it
merely does not understand. It is a **condition rather than an Event** on
purpose: an Event expires in an hour, and the verdict it qualifies is kept for
years. An acceptance record with no surviving note that one of its gates was
unsound is exactly the situation the condition exists to prevent.

Do not promote advice to a refusal, and do not demote a refusal to advice.

### What each check finds

| check | severity |
|---|---|
| metric name fails the contract grammar (`contract.ValidateMetricName`) — the parser drops such a metric, so the gate can never be satisfied | `Malformed` |
| threshold value does not parse as `float64` | `Malformed` |
| threshold value is `NaN` or `±Inf` — it expresses no bound, and `NaN` turns `NotEqual` into a gate that always passes | `Malformed` |
| comparison is not one of the four (an unrecognised comparison fails closed) | `Malformed` |
| `Equal`/`NotEqual` on a metric whose name carries a unit suffix | `Unsound` |
| `Equal`/`NotEqual` against a value that is not a whole number | `Unsound` |
| the metric is registered with `ThresholdUse: Evidence` | `Unsound` |
| the metric is **unregistered** and the test's kind is one this project ships | `Unsound` |

The last row is asked only where the answer is knowable. `pkg/contract`'s
registry is an **open world** on purpose — an unregistered lowerCamelCase name is
legal, which is what lets a third-party runner report a new measurement without a
release of this project — so for `KindCustom`, and for any kind this project does
not ship, nothing is said. For a kind in `v1alpha1.BuiltInKinds` the same silence
is a trap: this project owns the runner, the parser and the name, so an
unregistered gated name is either a registry entry somebody forgot or a gate no
runner satisfies. That is the failure issue #65 found — a `clockprobe` gate on a
metric whose values are `true`/`false`/`unknown`, which passed every check the
package then had and failed closed on every node.

**What the linter cannot check is the number.** A `clockprobe` gate at
`sustainedClockPct >= 90` failed a healthy part by seven points and lints
perfectly clean, because `GreaterThanOrEqual` is a sound comparison and
calibration is measured, not linted. The `ThresholdsSound` condition says
so in as many words when it is `True`.

---

## Not every metric is a number you may decide on

A threshold value is compared as a `float64`. Some of what a runner reports is a
**label**, and a gate on one of those parses (or fails to) in ways that look like
arithmetic and are not:

| metric | its actual values | why a gate on it is wrong |
|---|---|---|
| `pdWedgeSuspected` | `true` \| `false` \| `unknown` | tri-state on purpose; a threshold cannot express three states and fails closed on all of them |
| `throttleClassification` | `none` \| `thermal` \| `powerCap` \| `applicationClocks` \| `unknown` | an attribution, meant to send an engineer to a cable rather than a die |
| `computeCap` | `12.1` | the sly one: it **parses**, so the gate silently compares a `major.minor` version as a decimal, where `12.1` and `12.10` are the same value |
| `gpuName`, `pciBusId`, `driverVersion`, `builtCudaArch` | identity strings | identity is a fleet-inventory question; target a heterogeneous fleet with node selectors, not with a gate |

These are registered with `ThresholdUse: Evidence`, which makes
`contract.SafeToThresholdOn` answer false and the linter report the gate as
`Unsound` on every surface above. The registry entry's `Description` is the text
the linter prints, so it names the values the metric actually takes and says what
to gate on instead — for a clock shortfall, gate `sustainedClockPct` and *read*
`throttleClassification` to find out why.

The same classification catches numbers that are real measurements but poor
acceptance gates: `throughputTflops` from compute-smoke is a single unwarmed
kernel launch dominated by launch overhead, and `minLatencyUs` is the best sample
any run happened to catch. Both are `Evidence`. See the
[new-TestKind playbook](./dev/new-testkind-playbook.md) for the duty this puts on
a runner author adding a metric.

---

## Measure, then pin

Every threshold in this repository that has ever been wrong was wrong because
somebody derived it instead of measuring it. Both directions of the mistake are
represented, and the second is worse.

**Too tight — the fabric ceiling.** Two GB10 Sparks are joined by an Ethernet
link that genuinely negotiates 200000 Mb/s, so "about 90% of line rate" gives
`bandwidthGbps >= 180`. Measured on 2026-08-02, `ib_write_bw` between those two
nodes plateaued at **99.63 Gb/s** — identical at 4 and at 8 queue pairs with 1
MiB messages, so a real ceiling and not a single-QP artifact. The limit is the
ConnectX-7's host attachment: PCIe Gen5 x4, with `current_link_*` equal to
`max_link_*` (nothing is degraded), which is ~15.75 GB/s raw and ~100 Gb/s after
overhead. **The wire is 200G; the host can only feed half of it.** The spec-sheet
number is wrong by about 2x and would fail every healthy Spark pair in the fleet,
permanently, with each failure reading as a bad cable. The shipped sample gates
at `89` — about 90% of the *measured* ceiling — and says to re-measure before
reusing that number on any other SKU.

The companion number is worth reading beside it. The NCCL pair gate was once
`busBandwidthGBs >= 20`, justified as a "CX-7 200G RoCE expectation"; 20 GB/s is
160 Gb/s over a link whose measured ceiling is ~99.6 Gb/s, and for a 2-rank
all-reduce bus bandwidth equals algorithm bandwidth, so no healthy Spark pair
could ever have passed it. The measured peak was 12.00 GB/s and the gate is now
`10.8`.

**Too loose — the memory-bandwidth gates.** The same sample once carried
`hostToDeviceBandwidthGBs >= 20` against measured values around 59 GB/s. That
gate accepts a node running at a third of its expected bandwidth, which is
exactly the fault `memory-bw` exists to detect. **A too-permissive gate is worse
than no gate, because it reports a verdict.** Those gates are now pinned near 90%
of the measured fleet median.

**Duration and load change the answer.** Sustained clock on a healthy Spark is
83.22% of its 3003 MHz rated boost at 25 s and decays to 69.90% by 600 s as it
settles on a discrete P-state — with *zero* throttle reasons reported throughout.
A gate of 90 fails a perfectly good node at 300 s. Measure at the duration and
under the load your profile actually runs.

The practical procedure:

1. Run the profile **thresholdless** across known-good hardware, several nodes
   and several repeats. Set `spec.baseline: true` on the run so the result is
   marked as a measurement rather than a certification.
2. Read the distribution, not one sample. Note the minimum as well as the median.
3. Pin the gate against the measurement with headroom — roughly 90% of a
   measured median has been the working rule here for floors.
4. Record **in a comment beside the threshold** what was measured, when, on what
   hardware, and what the number is 90% of. Every gate in `config/samples/`
   carries that note, and it is what lets the next person tell a calibrated
   number from a guess.
5. Re-measure before reusing any of it on a different SKU.

### Baseline runs are refused, not laundered

`spec.baseline` marks a run as measuring rather than certifying, and it rides
into every delivered envelope so a consumer gating admission on "the last run
passed" cannot mistake a sweep for a certification. **It does not suppress
thresholds.** A baseline run whose profile carries a gate is refused at start, as
a config `Error` naming the test. Suppression would be a threshold-laundering
switch — a way to turn a failing acceptance run into a passing measurement run by
flipping a boolean, with the profile unchanged — and this operator's premise is
that a verdict cannot be edited into existence. If a profile has gates it is not
a baseline; point the run at a thresholdless profile.

---

## Reading a failed gate: who should act

All three of these arrive as the same `Failed` test, and only one of them is a
reason to touch a node. `status.results[].violations[].cause` is the field to
read first.

| `Cause` | means | who acts |
|---|---|---|
| `Measurement` | a real measurement was compared against a bar and fell short | the hardware. This is the only cause that is evidence about the part. |
| `Evidence` | the runner's report cannot support a judgement at all — metric missing, unparseable, non-finite, self-contradictory, or declared unmeasurable under a gate that requires it | whoever owns the runner, the mount, or the image. The node is **unjudged**, not condemned — though the gate still fails closed. |
| `Authoring` | the threshold itself is broken (a value that is not a finite number) | the profile. No hardware is implicated and no node should be touched. |

`Kind` is the finer-grained route — `Unsatisfied`, `NotReported`, `NonNumeric`,
`NonFinite`, `Contradictory`, `UnmeasurableRequired`, `ThresholdValueNonNumeric`,
`ThresholdValueNonFinite` — and maps onto `Cause` in exactly one place, so the
two can never disagree. An unrecognised kind answers `Evidence`: the conservative
reading is that nothing was established, never that the hardware is at fault.

The human-readable message names the first violated gate and then, when there are
more, appends each remaining metric with its cause in plain words (`hardware`,
`not measured`, `profile error`) — because that is what a reader needs to decide
whether to walk to a rack.

Two things worth remembering while reading one:

- **A threshold violation on a clean exit is a `Fail`, and a `Fail` is never
  retried.** `retryOnErrorLimit` re-runs an `Error` and nothing else, so a gate
  that fires settles the test where it happened with the retry budget unspent.
  See [invariants](./dev/invariants.md).
- **Thresholds are evaluated once, against the final metrics of a completed
  execution.** A checkpoint published mid-run by
  `spec.checkpointIntervalSeconds` is evidence, never a verdict: a sample that
  dips below a floor at minute 200 is not a failure, because the run is not over.

---

## A note on segmented runs

`pkg/contract.Metric.Aggregation` declares how a metric combines when the same
test runs more than once and one verdict must be rendered across all of it —
`Sum`, `Min`, `Max` or `Last`. It is required on every registered metric.
**Nothing consumes it yet**, so it does not affect any verdict today; it exists
so that the answer lives beside the name rather than in a per-metric switch
inside the reconciler.

It is worth knowing about while authoring, because the rule that is wrong in a
way nothing else catches is lifetime versus windowed, and host-health has both.
`xidEvents` and `pcieReplayErrors` are differenced over the window and `Sum`;
`eccErrors` and `remappedRows` are NVML aggregates since reset and are `Last`.
Summing a lifetime total across windows multiplies it by the number of windows —
a node with four remapped rows would read as twelve after a three-segment soak
and be condemned for damage it does not have. A floor takes `Min`, because the
worst window describes the part; a ceiling takes `Max` for the same reason
inverted. An unregistered name aggregates `Last`, because inventing `Sum`
semantics for a name nothing declared would be this operator deciding what
somebody else's measurement means.

---

## See also

- [The runner contract](./runner-contract.md) — exit codes, `key=value` output,
  the `n/a` sentinel a `RequiredIfMeasurable` gate depends on
- [Concepts](./concepts.md) — the CRDs, scopes, and how a run proceeds
- [Reports](./reports.md) — how violations and not-evaluated gates are rendered
- [Sinks](./sinks.md) — what a delivered envelope carries about a verdict
- [Invariants](./dev/invariants.md) — `Error` is not `Fail`, and the rest
- [Running a soak](./soaks.md) — the capacity arithmetic behind a long gate
- [Adding a TestKind](./dev/new-testkind-playbook.md) — registering a metric so
  the linter can say something useful about a gate on it

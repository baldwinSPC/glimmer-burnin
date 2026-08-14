# Adding a TestKind

Everything you need to add a new kind of acceptance test, in the order you need
it, with the guard that catches each mistake.

**Every rule here exists because something went wrong.** Where a step names an
incident, that is not colour — it is the reason the rule is not negotiable.

The guards are listed so you learn which mistakes are caught **mechanically** and
which are not. The ones that are not are the ones to be careful about.

---

## Before you start: is it a kind at all?

Three questions, in this order.

**Is it a new measurement, or a new cell of an existing one?** The same kernel at
FP8 and FP64 is one kind with a `precision` variant axis, not two kinds. So is
bandwidth versus latency on one link. Use `variants` on the profile entry; the
runner reads `BURNIN_VARIANT_<AXIS>`. A new kind is for a genuinely different
measurement, not a different setting of the same one.

**Is it vendor-neutral, or is it the vendor's artifact?** `nccl` and `memory-bw`
are neutral names — an AMD fleet should be able to satisfy them with a different
image. `dcgm-diag` is vendor-named because the thing being run *is* NVIDIA's own
suite, and wrapping AMD's would be a different kind (`rvs-diag`), not a different
image for this one.

**Does it need a scope other than Node?** Pair produces one verdict about a
**link**; Group produces one verdict about a **collective**. If your measurement
is a property of a relationship rather than of a part, say so now — it changes
the environment contract and the runner's structure.

---

## 1. The runner contract

A runner's entire interface is **an exit code and `key=value` lines on stdout**.
That is what makes it trivially testable, trivially portable, and unable to reach
anything it was not given.

| exit | meaning |
|---|---|
| 0 | pass |
| 1 | **fail — a hardware verdict** |
| 2 | skip, **only with a declared marker** |
| 3 | error — could not measure; the hardware is unjudged |
| anything else | error |

### Exit 1 is the expensive one, and it is the one runners get wrong

The operator **never retries a `Fail`**. It permanently indicts a node with the
retry budget unspent. It belongs *only* to something the run actually measured
about the part.

Everything that stopped the measurement from happening — no device visible, an
image with no kernel for this part, a driver error, a failure to set up the
comparison — is **exit 3**. `compute-smoke` shipped `v0.1.0` reporting all three
of those as exit 1; that is the shape of the bug to look for in your own.

### A skip must be DECLARED, not merely exited

Exit 2 is honoured as `Skip` only when stdout also carries an upper-case token
ending in `_SKIP` at the start of a line: `FP4_GEMM_SKIP:`, `NCCL_SKIP:`,
`IB_WRITE_BW_SKIP:`.

**Exit 2 with no marker is an `Error`.** An unrecovered Go panic exits 2, as does
every Go runtime fatal error — out of memory, concurrent map writes, stack
exhaustion — which no runner can recover from however carefully written. The
camouflage is otherwise perfect: a container log merges stdout and stderr, so a
crashed runner produces a stack trace with no `key=value` line at all, and "no
metrics" is the *normal* shape of a skip. That landed a crash on the one phase
that is never retried, never affects the run's verdict, and reports the node as
one the test did not apply to — so a run settled `Passed` around hardware nobody
measured.

> **Guard:** `runners/pins_test.go` → `TestEverySkipMarkerIsOneTheParserRecognises`.
> It fails if a marker is renamed into a form `pkg/runner.DeclaresSkip` no longer
> recognises. A marker is not always one literal — the soak family composes
> `markerPrefix + "_SKIP"` — and a suffix found with no `markerPrefix` declared in
> the directory is itself a failure: a marker nothing in the repository can
> predict is a marker nothing can check.

### `n/a` is reserved, and omission is different

Emit `n/a` (case-insensitive) where you **asked the hardware and it has nothing
to report**. GB10 exposes no ECC to NVML, so `eccErrors=n/a` is a positive
declaration about the part, and a threshold marked
`applicability: RequiredIfMeasurable` is then reported as NOT EVALUATED rather
than failing a healthy node.

**Omit the key** where the probe itself failed. Absence is not a declaration: a
metric that simply never appears still fails under every applicability, because a
crashed probe must not become an acceptance.

**Never emit a `0` you did not measure.** Without `/dev/kmsg`, `host-health`
reports `xid_source=none` and omits `xidEvents` rather than printing a zero — so a
gate on it fails the node instead of certifying it. A node condemned for a
reading nobody took is recoverable; a fleet certified clean by a fabricated zero
is not.

### Last-occurrence-wins

`pkg/runner.Parse` takes the last value for a repeated key, and takes
`Result.Message` from the **last non-`key=value` line**. Two consequences:

- Two keys must never alias to the same canonical name — one measurement would
  silently discard the other.
- Diagnose *before* you launch anything that can spew. `compute-smoke`
  establishes an arch mismatch before launching, because `cudaErrorAssert` is
  sticky and several hundred lines of CUTLASS template text arriving after your
  diagnosis would overwrite it in the stored result.

---

## 2. Duration: honour it, or declare the kind burst-only

Read `BURNIN_DURATION_SECONDS`, **or** declare the kind burst-only via
`TestKind.BurstOnly()`.

`compute-smoke` is burst-only: its GEMM finishes in milliseconds however long it
is asked for, and a shipped sample requested 120 s — so "node acceptance" burned
nothing in and nothing said so. The resolution was to stop pretending, not to add
a loop: a burst proves the arch-correct instruction path executed and got the
right answer, which a longer run does not strengthen, and looping it would
duplicate `thermal-soak` while costing the fleet its one cheap correctness gate.

`durationSeconds` still bounds the **pod deadline** for a burst kind; it just
does not decide how long the test runs.

> **Guards:** `TestDurationIsHonouredOrDeclaredBurstOnly` refuses a non-burst kind
> whose source never reads the variable AND a burst kind whose source does.
> `api/v1alpha1/samples_test.go` → `TestNoSampleRequestsADurationARunnerIgnores`
> refuses a sample that sets `durationSeconds` on a burst kind.

---

## 3. Metrics

### The grammar

lowerCamelCase. A **dimensional** metric ends in a registered unit suffix; a
**dimensionless counter** does not.

`Gbps` `GBs` `MBs` `Us` `Ms` `S` `C` `W` `Pct` `Tflops` `MHz`

The casing is a trap and it is the quiet one. `gbs` folds to `Gbs`, which is not
the registered `GBs` — so `h2d_bandwidth_gbs` normalises to a name `UnitOf()`
reads as **dimensionless**, and a bandwidth is recorded as a bare number. That is
why nearly every bandwidth runner needs an alias entry.

### Register it in `pkg/contract/metrics.go`

Each entry declares `Unit`, `Description`, `Aggregation` and `ThresholdUse`.

**`Aggregation` says how the metric COMBINES when a soak runs in segments**, and
the rule that is wrong in a way nothing else catches is LIFETIME versus WINDOWED:

| | |
|---|---|
| `Sum` | a windowed count — `xidEvents`, `pcieReplayErrors` differenced over the window |
| `Last` | a **lifetime** aggregate since reset — `eccErrors`, `remappedRows` from NVML. Summing these multiplies them by the number of windows, so a node with four remapped rows reads as twelve after a three-segment soak and is condemned for damage it does not have |
| `Min` | a **floor**: the worst window describes the part. A soak holding 83% of rated clock for eleven hours and 40% for one is a part that dropped to 40% |
| `Max` | a **ceiling**, for the same reason inverted |

**`ThresholdUse: Evidence` for anything whose value is a LABEL.** A threshold is
compared as a `float64`, so a gate on `true|false|unknown`, or on a word, or on
`12.1` read as a version, fails closed on **every node forever** while reading as
a hardware verdict. `computeCap` is the sly one: it parses, so the gate silently
compares a `major.minor` version as a decimal. Registering these turns
`SafeToThresholdOn` to false so the linter reports the gate as `Unsound`.

Diagnostic **integers** may stay unregistered — clockprobe's nine per-reason
counters deliberately do — because no gate on one fails closed. The unregistered
rule still asks for registration the moment somebody gates on one, which is the
right moment.

### Alias, only if you need one

The generic `snake_case → lowerCamelCase` handles nearly everything. An entry in
`pkg/runner/parse.go` is needed only where the runner's key is not merely a
different spelling:

1. the key names no measurand (`tflops` is a bare unit);
2. the key's unit differs in **case** from the registered suffix (see above);
3. the tool's vocabulary differs from ours (`gpu_burn`'s "errors" are
   miscompares).

### Write the parser test

Required for every new kind. Assert against **captured real output** where you
have it: an audit of `clockprobe`'s source missed nine metrics and a capture
could not.

Where the runner has never executed — a kernel awaiting its first hardware
session — there is nothing to capture, and the parser test **waits for the
capture**. Do not write it against stdout you invented: a test asserting output
you imagined is "a test that passes against broken code" wearing a different
hat, and it manufactures confidence in exactly the layer CI cannot check. File
the hardware-verification issue instead (the shape of #237, #242, #265: what to
run, what to capture, what would falsify the design), write the parser test
against those fixtures, and only then publish a tag.

> **Guards:** the registry self-consistency test refuses `Unspecified` for either
> `Aggregation` or `ThresholdUse`; `TestAliasTargetsAreRegistered` refuses an
> alias pointing at an unregistered name;
> `TestRegistryContainsTheNamesConsumersDependOn` makes you add each name
> deliberately, because every canonical name is a promise to a consumer.

---

## 4. The environment the operator injects

Every scope gets `BURNIN_DURATION_SECONDS` and `BURNIN_ATTEMPT`, plus
`BURNIN_VARIANT_<AXIS>` for each variant axis.

**Pair** additionally:

| | |
|---|---|
| `BURNIN_ROLE` | `server` or `client` |
| `BURNIN_PEER_HOST` | the peer's DNS name, `<peer-role>.<service>.<ns>.svc` |
| `BURNIN_PEER_NODE` | the peer's node name — for messages, **never** for addressing |

**Group** gets a different set, and deliberately **no `BURNIN_ROLE`**:

| | |
|---|---|
| `BURNIN_RANK` | this pod's 0-based rank |
| `BURNIN_NRANKS` | how many ranks the collective has |
| `BURNIN_ROOT_HOST` | rank 0's DNS name, `rank-0.<service>.<ns>.svc` |
| `BURNIN_ROOT_NODE` | rank 0's node name — for messages, never for addressing |

That is the whole rendezvous contract: which end this is, and where the other end
answers. A fabric runner needs nothing else and never learns anything about
Kubernetes.

`BURNIN_PEER_HOST` is deliberately **not** qualified to `cluster.local`: a
cluster's DNS domain is configurable, and hard-coding the default would break the
rendezvous as a connection error that looks like a bad link.

### If you write a Pair runner

**A Pair verdict is about the LINK.** One `TestResult` naming both nodes, never
split per node — attributing a point-to-point measurement to one endpoint sends
an engineer to replace the wrong part.

The **client is the deciding side**. Verdict precedence is
`Error > Fail > Skip > Pass`, because a machinery failure on either end means the
link was never measured and a client's "connection refused" is then an artifact
of its peer. A **server that exits 0 before the client ever started is an
`Error`**, never a pass: no traffic crossed the link.

Declare `spec.runner.readinessProbe`, but know that **the probe spends a
connection**. TCP admits no way to prove a listener is up other than connecting
to it, so the probe is a real client — and a server that accepts a bounded number
of connections has served the probe instead of the client. `ib_write_bw` and the
nccl-tests server both accept a bounded count. Probe a port the measurement does
not use, or make the server re-accept.

### If your runner speaks only Pair

**Refuse Group explicitly, at exit 3, before any hardware inspection.** Every
fabric runner branches on `BURNIN_ROLE`, and Group deliberately does not set it —
so all three would have declared a `_SKIP` for a collective that never ran, which
is the one verdict that certifies. "This image does not speak the Group
rendezvous" is true whatever is in the box.

> **Guard:** `TestGroupCapableRunnersReallyReadTheGroupContract`, keyed on
> `BURNIN_ROOT_HOST`. A first attempt keyed on `BURNIN_RANK`/`BURNIN_NRANKS`
> failed immediately, because the refusing runners read exactly those in order to
> say no. **Reading a variable to refuse is not implementing the contract.**

---

## 5. Host access

`spec.runner.hostPaths` is the **only** way a runner pod reaches the node's
filesystem. A test that declares nothing gets a pod with no volumes at all —
there is no default `/dev`, no convenience `/sys`, nothing.

`readOnly` **defaults to true** and is a pointer, so "unset" is distinguishable
from "false": a privilege grant's default has to fall towards the harmless form.
`type` admits only the **assert-only** hostPath kinds, so a test can never ask
the kubelet to *create* a path — an operator sent to measure a fleet must not
mutate it.

**`privileged: true` is not a substitute.** Measured on this project's own nodes,
a privileged `hostNetwork` pod was missing every `/dev/infiniband/uverbs*` device
the host had — which is exactly what `ibv_create_cq` opens.

**Degrade honestly when a mount is missing.** Without `/dev/infiniband` the
fabric runners fail loudly at `Couldn't create CQ` rather than reporting a
number. Never paper over a missing mount with a fabricated measurement.

---

## 6. The Dockerfile

Multi-stage. The CUDA toolchain is used at **build time only**, and the build
fails if `ldd` finds `libcuda`/`libcudart`/`libnv` in the final binary —
`libcuda.so.1` is injected at runtime from the host driver by the NVIDIA
Container Toolkit. **Runner images ship no NVIDIA-licensed redistributable
libraries**, which is also why cuBLAS is not an option however much simpler it
would make a kernel.

**Every upstream is pinned to a COMMIT and the build asserts it.** An
`<UPSTREAM>_REF` (a tag) is always accompanied by an `<UPSTREAM>_SHA`. A tag is a
mutable pointer, and because our published image tag is immutable, a moved
upstream tag would change what a fleet is measured by with nothing in this repo
recording it. Resolve with:

```sh
git ls-remote <repo> 'refs/tags/<ref>' 'refs/tags/<ref>^{}'
```

taking the peeled `^{}` value for an annotated tag — that is what a shallow clone
leaves at HEAD.

**Declare driver injection** unless your runner touches no accelerator. The GB10
cluster's CDI spec declares a hook its host toolkit cannot implement, so every
pod injected through CDI dies at `createContainer`; the burn-in runners survive
only because containerd falls back to the legacy runtime, which injects from
`NVIDIA_VISIBLE_DEVICES`/`NVIDIA_DRIVER_CAPABILITIES` **in the image
environment**. That is a property of these images, not of the host — the fleet is
one Dockerfile away from a cluster-wide outage.

**Arch targets are deliberate.** CPU architecture and GPU architecture are
orthogonal: `linux/amd64` vs `linux/arm64` is the host; `sm_121a` is the GPU
gencode. An amd64 image containing only `sm_121a` helps nobody, because CC 12.1
is GB10 and GB10 is a Grace part. Do not widen a gencode list to make a build
succeed. Note PTX only JITs **upward** — `compute_121` PTX cannot rescue a CC
10.x part.

> **Guards:** `TestEveryRunnerDirectoryHasADockerfile`,
> `TestEveryUpstreamRefIsPinnedToACommit`, `TestEveryCloneNamesAPinnedRef`,
> `TestFromArgsAreDeclaredBeforeTheFirstFrom`,
> `TestNoRunnerCompilesAgainstAnUnresolvedArch`,
> `TestDriverInjectionIsDeclaredWhereItIsNeeded`,
> `TestNoDockerfileDefinesAFaultInjectionMacro`,
> `TestNoRunnerBinaryCanBeCommittedAtTheRepositoryRoot`.

---

## 7. Test the part that does not need hardware

**The part of a GPU runner worth unit-testing is never the part that needs a
GPU.** Lift every decision that can produce a wrong *verdict* into a CUDA-free
header and test it exhaustively with plain C++ — which precision was asked for,
whether this part supports it, what the skip message says.

`compute-smoke/arch_match.h` is the worked example, driven by
`arch_match_test.cc`. Note a runner directory containing a `.cc` file cannot also
hold a Go file — `go build` rejects C++ sources outside cgo — which is why the
sweep lives in `runners/`.

For a branch no available hardware can reach, add a **compile-time**
fault-injection macro. It must be compile-time so no environment variable can
reach it; no Dockerfile may define one; and such a build must announce itself in
a `key=value` line, so the marker rides into the stored result and the verdict
stays self-identifying.

> **Guards:** `runners/cxxtests_test.go` → `TestCxxUnitTests` compiles and runs
> every `*_test.cc` in the tree. `TestFaultInjectionMacrosAnnounceThemselves`.

---

## 8. Wire it up

| file | what |
|---|---|
| `api/v1alpha1/burnintest_types.go` | the `Kind*` constant, **and add it to `BuiltInKinds`** — otherwise `ValidateThresholdsForKind` silently stops applying to it |
| `pkg/contract/metrics.go` | every metric, with `Aggregation` and `ThresholdUse` |
| `pkg/runner/parse.go` | the alias entry, if needed |
| `pkg/runnerimages/images.go` | the default image, **once a tag exists**. A kind with no runner source deliberately has none, so it fails fast at plan time asking for an explicit `spec.runner.image` instead of pull-failing per node |
| `NOTICE` | every upstream. Where dual-licensed, say which option we consume it under — `linux-rdma/perftest` is GPL/BSD and must be taken under **BSD** |
| `runners/<kind>/README.md` | what a pass claims, what each exit code means here, what it needs on the host |
| `config/samples/` | a sample that decodes strictly and lints clean |
| `.github/workflows/publish-runner.yml` | the `runner` choice list, or the image cannot be built by the workflow at all |

If you duplicate source across runners — each is its own Docker build context and
`COPY` cannot reach outside one — declare it in the relevant contract test as
**shared** (byte-identical) or **forked** (with the reason). The fabric table is
total: a new duplicate fails until it is declared, and a declared fork must
really differ.

---

## 9. Publishing

**Manual only**, via `workflow_dispatch`. A runner image is executed by a node's
readiness gate, so a bad tag degrades a whole fleet at once.

**Published tags are immutable.** Never republish a tag a gate pins — not for a
bug fix, not for a security fix. Cut a new version. Silently changing a tag
changes every node's verdict with no audit trail.

**Not before the kernel has been verified on real hardware.** "It builds" is not
verification: an `sm_120a` image *loads* on a CC 12.1 GB10 and trips a
device-side assert inside the kernel.

---

## Which runner to copy

| shape | model |
|---|---|
| stdlib probe, no accelerator SDK | `host-health` |
| wrapping somebody else's tool | `memory-bw` |
| sustained load with a fault watch | `gpu-burn` |
| Pair, two endpoints over a link | `ib-write-bw` |
| Group, N ranks in a collective | `nccl` |
| CUDA correctness kernel | `compute-smoke` |

---

## The shortest possible summary

A runner may only declare **what it positively established**. Everything above is
that sentence applied to a specific place where it was once got wrong.


## The shared helper stamp

Every Go runner carries `zz_runnerlib.go`, stamped byte-identical from
`hack/runnerlib/runnerlib.go.src` — `logf`, `metric` (sanitizing), `envInt`,
`envOr`, `sanitize`. The image build is a throwaway module and cannot import a
repository package, so this is how helpers are shared. For a new Go runner:

```sh
go run ./hack/runnerlib
```

and do not declare any of those five locally — `runners/runnerlib_test.go`
fails CI on a missing or drifted stamp and on a local re-declaration alike.

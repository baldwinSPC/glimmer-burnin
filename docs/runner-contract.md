# The runner contract

A **runner** is a container image that measures one thing about one piece of
hardware. Its entire interface to this operator is:

- an **exit code**, and
- **`key=value` lines on stdout**.

That is all of it. There is no sidecar, no API to call, no library to link, no
credential to hold. A runner can be a Go binary, a CUDA program, a shell script
wrapping somebody else's tool — the operator cannot tell and does not care. The
smallness is the design: it is what keeps vendor- and accelerator-specific
behaviour at the image boundary instead of inside the reconciler, and it is what
lets a runner be tested by piping a captured log into a parser.

**This page is the contract. [`dev/new-testkind-playbook.md`](dev/new-testkind-playbook.md)
is the checklist** for wiring a new kind into *this* repository — which file to
edit, which guard catches which mistake. If you are writing a runner that lives
somewhere else, you need this page and not that one. Neither restates the other.

Two facts to keep in mind throughout, because most of the rules below follow
from them:

1. **The operator never retries a `Fail`.** A hardware verdict settles the test
   where it happened, with the retry budget unspent. Only an `Error` is re-run
   (`spec.retryOnErrorLimit`). See [`concepts.md`](concepts.md).
2. **A container log is stdout and stderr merged.** Whatever your runner writes
   to stderr is interleaved into the same stream the parser reads.

---

## 1. Exit codes, and what each one claims

| exit | verdict | the claim it makes |
|---|---|---|
| `0` | `Pass` | the test ran, and the runner is content. Thresholds then decide acceptance |
| `1` | **`Fail`** | **the hardware was measured and fell short.** A verdict about the part |
| `2` | `Skip`, **only with a declared marker** | acceptance does not apply to this hardware |
| `3` | `Error` | the runner could not measure. The hardware is **unjudged** |
| anything else | `Error` | as above — `pkg/runner.VerdictFor` maps every unrecognised code here |

`3` is the code a runner author writes deliberately; the "anything else" clause
is a safety net, not a licence to be vague.

### Exit 1 is the expensive one, and it is the one runners get wrong

Exit 1 permanently indicts a node. It belongs **only** to something the run
actually measured about the part.

Everything that stopped the measurement from happening belongs to exit 3:

| situation | code |
|---|---|
| no accelerator visible / no device to open | 3 |
| the image contains no kernel for this part | 3 |
| a driver or runtime error | 3 |
| the peer never came up, so no traffic crossed the link | 3 |
| a variant axis carried a value the runner does not recognise | 3 |
| the measurement completed and the number is bad | **1** |

`compute-smoke` shipped `v0.1.0` reporting the first three of those as exit 1 —
a hardware FAILURE verdict against every node it could not run on. That tag is
public and immutable and still behaves that way; `v0.2.0` was the first build
where they are exit 3. That is the shape of the bug to look for in your own.

`ib-write-bw` is the worked example of the discipline. It exits 0 on any
completed measurement and lets the profile's thresholds decide, because a 200G
fabric and a 100G fabric are both healthy and a runner that hard-coded a floor
would condemn one of them. It reserves exit 1 for the two states that are
hardware facts on their own terms: RDMA hardware present with no usable link,
and a link that completed the test carrying no traffic at all.

### A skip must be DECLARED, not merely exited

Exit 2 is honoured as `Skip` **only when stdout also carries a marker**. The
marker is matched by `pkg/runner.DeclaresSkip` with this regular expression:

```
(?m)^[A-Z][A-Z0-9_]*_SKIP\b
```

which means, precisely:

- at the **start of a line** — leading whitespace disqualifies it;
- upper-case letters, digits and underscores only;
- at least one character before the `_SKIP`, so a bare `SKIP` does not count;
- a word boundary after it, so `NCCL_SKIPPED` does not count. `NCCL_SKIP:` does.

Shipped examples: `FP4_GEMM_SKIP:`, `NCCL_SKIP:`, `IB_WRITE_BW_SKIP:`,
`GPUDIRECT_RDMA_SKIP:`, `STRESSAPPTEST_SKIP:`, `DCGM_DIAG_SKIP:`,
`MEMORY_BW_SKIP:`, `CLOCKPROBE_SKIP:`.

**Exit 2 with no marker is an `Error`,** and `Result.UndeclaredSkip` records that
that is what happened.

The reason is that **an unrecovered Go panic exits 2** — as does every Go runtime
fatal error: out of memory, concurrent map writes, stack exhaustion, none of
which any runner can recover from however carefully it is written. The
camouflage was otherwise perfect. Because a container log merges stdout and
stderr, a crashed runner produces a stack trace with **no `key=value` line at
all** — and "no metrics" is the *normal* shape of a legitimate skip. So a crash
landed on the one phase that is never retried, never affects the run's verdict,
and reports the node as one the test did not apply to. A run settled `Passed`
around hardware nobody measured (issue #103, found on `host-health`, the one
runner that claims never to skip).

It fails towards `Error` deliberately. A runner that skips honestly and forgets
the marker is reported unjudged and retried, which is visible and cheap. The
opposite mistake certifies a fleet nobody looked at.

If your runner is written in a language where a crash exits 2, this rule is
protecting you specifically. If it is not, the rule costs you one printed line.

### A runner that cannot skip should say so and never print a marker

`host-health` applies to every node, so exit 2 is unreachable from its `run()`
by design and it prints no marker anywhere — which is exactly what makes a crash
of it legible as an `Error`. Do not add a marker "for completeness"; it is the
one thing that would make its crashes invisible again.

---

## 2. Metrics: `key=value` on stdout

One metric per line. `pkg/runner.Parse` splits on the **first** `=`, so a value
may itself contain one. Keys and values are whitespace-trimmed.

A line is **not** a metric — and becomes the run's `Message` instead — when:

| line | why |
|---|---|
| it contains no `=` | plain prose |
| it starts with `=` | there is no key |
| the key contains a space or tab | `some prose = with an equals` is not a metric |

`Result.Message` is the **last** such line. Blank lines are ignored.

### `Message` is the last non-metric line, and that has teeth

Anything your runner prints *after* its diagnosis overwrites the diagnosis in
the stored result. Since the container log merges stderr, "anything" includes
several hundred lines of CUTLASS template text from a failed kernel launch.

`compute-smoke` therefore establishes an architecture mismatch from the arch
string its Dockerfile bakes into the binary (`-DBURNIN_CUDA_ARCH`) and the
device's reported compute capability, and **refuses before launching**, rather
than letting the launch tell it. Doing it beforehand
matters twice over: `cudaErrorAssert` is sticky and poisons the context, and the
spew would bury the explanation.

`host-health` does the same thing from the other end: it prints the stack trace
of a recovered panic **before** its `HOST_HEALTH_ERROR:` line, so the marker is
last and the stack does not overwrite it.

### Last-occurrence-wins

A repeated key keeps its **last** value. Two consequences follow.

- **Two keys must never map to the same canonical name.** One real measurement
  would silently discard the other, and nothing would say so.
- **Progressive reporting works, and is useful.** `host-health` streams a
  `host_health_stage=<stage>` breadcrumb the instant each stage is entered,
  straight to stdout rather than into its output buffer. That is the only thing
  that survives an OOM kill or a runtime fatal error, and it lands in the stored
  result as the furthest stage reached. Issue #112 was undiagnosable precisely
  because every metric was buffered and written in one block at the end, so a
  runner that died produced completely empty stdout.

Last-occurrence-wins also applies **across** the metric map and the unmeasurable
set, keeping the two disjoint: a key declared `n/a` and later given a real number
has been measured, and the reverse retracts the number.

### `n/a`, omission, and the zero you must never print

Three states, three different claims:

| what you print | what it means | what a threshold does |
|---|---|---|
| `eccErrors=3` | measured | evaluated normally |
| `eccErrors=n/a` | **this hardware cannot produce this measurement at all** | a `RequiredIfMeasurable` threshold is reported NOT EVALUATED; a `Required` one still fails |
| *(the key is absent)* | you could not look | **fails closed**, under every applicability |
| `eccErrors=0` when you did not measure | a lie | silently certifies the node |

`n/a` is reserved, matched case-insensitively after trimming, and is the **only**
spelling that counts — `unknown`, `none` and an empty value are ordinary values
a runner might mean something else by, and a gate must not relax on a guess. It
lands in `Result.Unmeasurable`, never in `Result.Metrics`, so nothing can compare
against it and `Numeric()` cleanly reports "not a number".

The rule behind all three rows: **a runner may only declare what it positively
established.** GB10 exposes no ECC to NVML at all, so `eccErrors=n/a` is a real
claim about the part — and `host-health` establishes it from `ecc.mode.current`,
which is how it tells "this part has no ECC" from "ECC was switched off". Only
the first is declarable. A counter it could not read for any other reason — no
driver, a timeout, a missing column, some GPUs answering and others not — is
**omitted**.

Worked examples of the omission side, both deliberate:

- Without `/dev/kmsg`, `host-health` reports `xid_source=none` and **omits**
  `xidEvents`. A gate on it then fails the node rather than certifying it.
- `ib-write-bw` never emits `bw_peak`. perftest measures a peak only in
  iteration mode; in duration mode — which is the mode a burn-in runs in — it
  prints `0.00` and warns that the peak was not measured. Emitting that zero
  would put a number nobody took into `peakBandwidthGbps`. Nor is it `n/a`: the
  sentinel is a claim about the *hardware*, and this is a property of the mode
  the runner chose. A threshold on it fails closed, which is correct.

### Exit 0 with no output at all is not a pass

If a runner exits 0 and the parse finds **no `key=value` line whatsoever** — no
metric, no `n/a`, not even a name the grammar rejected — and the test carries
thresholds, the operator records **`Error`**, not `Pass` and not a fail-closed
`Fail`. A runner that printed nothing measured nothing: it was killed, evicted,
or died before its first line, and handing an empty map to fail-closed
evaluation would manufacture a hardware verdict out of an absence.

This does **not** relax fail-closed and must never be widened until it does. A
runner that reported forty numbers and omitted the forty-first still fails: it
looked at the hardware, and a measurement it owed and did not produce is exactly
the silence that must never satisfy acceptance.

---

## 3. Metric names

Canonical names cross a repository boundary — you emit them, this operator
gates on them, an external consumer stores and charts them, and nothing
reconciles a disagreement. **The name is the contract.**

### The grammar

1. **lowerCamelCase.** Must start with a lowercase `a`–`z`; alphanumeric
   throughout. A canonical name contains no underscores or dashes.
2. A metric carrying a **physical unit** must end in a registered unit suffix.
   `bandwidth` is not a name; `bandwidthGbps` is. A bare unit is not a name
   either — `tflops` says nothing about what was measured, so the canonical form
   is `throughputTflops`.
3. A **dimensionless counter** must not carry a unit suffix.

The registered suffixes, from `pkg/contract/metrics.go`:

| suffix | unit |
|---|---|
| `Gbps` | gigabits per second — wire throughput, as interconnect vendors quote it |
| `GBs` | gigabytes per second — memory and collective bandwidth, as NCCL/DCGM report it |
| `MBs` | megabytes per second |
| `Us` | microseconds |
| `Ms` | milliseconds |
| `S` | seconds |
| `C` | degrees Celsius |
| `W` | watts |
| `Pct` | percent |
| `Tflops` | teraflops |
| `MHz` | megahertz |

Longer suffixes are checked first, so `busBandwidthGBs` is not read as a name
ending in `S`.

`bandwidthGbps` and `busBandwidthGBs` coexist rather than one being renamed to
match the other, and that is the convention working: they are not the same
quantity. `ib_write_bw` reports raw link throughput; NCCL bus bandwidth is a
derived figure scaled by the collective's communication pattern. Normalising
them to one unit would make two different measurements look interchangeable.

### The casing trap

This is the quiet one, and it is why nearly every bandwidth runner needs an
alias entry.

Generic normalisation turns `snake_case` into `lowerCamelCase` by lowercasing
each part and capitalising its first letter. So `h2d_bandwidth_gbs` becomes
`h2dBandwidthGbs` — and **`Gbs` is not the registered `GBs`**. `UnitOf()` reads
the result as *dimensionless*, and a bandwidth is stored as a bare number with
no unit anywhere in its name. Same for `mhz` → `Mhz` (not `MHz`) and
`soak_seconds` → `soakSeconds` (whose lowercase `s` is not the `S` suffix, so a
duration lands dimensionless).

**The portable fix, and the one available to a runner outside this repository:
print the canonical name yourself.** A key already in lowerCamelCase passes
through normalisation unchanged, so a runner may emit `hostToDeviceBandwidthGBs=…`
directly and needs no alias entry at all. This is the *only* option for
`kind: custom` and for any kind this project has no alias table for.

### Aliases (first-party kinds only)

`pkg/runner/parse.go` holds a per-kind alias table, keyed on the runner's
**literal printed key**, for the three cases generic normalisation cannot reach:

1. the key names no measurand — `tflops` is a bare unit;
2. the key's unit differs in **case** from the registered suffix (above);
3. the tool's own vocabulary differs from ours — `gpu_burn`'s "errors" are
   miscompares; `stressapptest`'s "hardware incidents" are memory errors.

An unknown or custom kind still gets the generic scan — that scan **is** the
runner contract, and a third-party runner honouring it is not punished — but
gets no kind-specific mapping, since only the kind's author knows what its keys
mean.

### A name the grammar rejects is reported, not dropped

A canonical name failing `ValidateMetricName` lands in `Result.InvalidNames`
rather than being silently discarded. A runner emitting an unusable name should
be visible. (The common way to get there is printing `Foo=1`: it has no `_`, so
it passes through unchanged and then fails the lowercase-first-letter rule.)

### Registered, unregistered, and label-valued

The registry is an **open world**: a lowerCamelCase name nothing has registered
is legal, so a third-party runner can report a new measurement without a release
of `pkg/contract`. That stays true.

Two consequences a runner author should know:

- An **unregistered** name is assumed thresholdable and aggregates `Last`. For a
  kind this project ships a runner for, gating on an unregistered name is
  reported as `Unsound` by the threshold linter — see [`thresholds.md`](thresholds.md).
- **If your metric's value is a word, a version or a tri-state, it is not
  thresholdable**, and a gate on it fails closed on every node forever while
  reading as a hardware verdict. A threshold is evaluated by parsing the value
  as a `float64`. `pdWedgeSuspected` is `true|false|unknown`;
  `throttleClassification` is one of five words; `computeCap` is `12.1`, which is
  the sly one because it *parses* — the gate silently compares a `major.minor`
  version as a decimal. First-party metrics of this class are registered as
  `ThresholdUse: Evidence` so the linter refuses the gate. A third-party runner
  emitting one is outside that machinery; document it, and do not gate on it.

Registered metrics also declare an `Aggregation` (`Sum`, `Min`, `Max`, `Last`)
saying how the number combines when a test runs in segments. Nothing consumes it
yet. The distinction that bites is lifetime versus windowed: `xidEvents` is
differenced over the window and sums, while `eccErrors` and `remappedRows` are
NVML aggregates since reset and take `Last` — summing those would report a node
with four remapped rows as having twelve after a three-segment soak.

---

## 4. The environment the operator injects

### Every scope

| variable | value |
|---|---|
| `BURNIN_DURATION_SECONDS` | how long the test should run |
| `BURNIN_ATTEMPT` | which attempt this is (1-based). A runner may use it for seeding or logging; nothing in the contract depends on it |
| `BURNIN_VARIANT_<AXIS>` | one per variant axis, axis name upper-cased, value passed through uninterpreted |

The operator does not read the variant values back. Interpreting a precision or
a message size is the runner's job — a reconciler branching on `precision: fp4`
would be exactly the vendor-specific behaviour this project keeps out of the
control plane.

**Refuse an axis value you do not recognise.** `ib-write-bw` exits 3 on an
unknown `BURNIN_VARIANT_MEASURAND` rather than defaulting to "both": defaulting
would silently take a measurement the author did not ask for, gated by thresholds
written for the one they did.

### Pair scope, additionally

| variable | value |
|---|---|
| `BURNIN_ROLE` | `server` or `client` |
| `BURNIN_PEER_HOST` | the peer's DNS name, `<peer-role>.<service>.<ns>.svc` |
| `BURNIN_PEER_NODE` | the peer's node name — for messages, **never** for addressing |

### Group scope, and deliberately no `BURNIN_ROLE`

| variable | value |
|---|---|
| `BURNIN_RANK` | this pod's 0-based rank |
| `BURNIN_NRANKS` | how many ranks the collective has |
| `BURNIN_ROOT_HOST` | rank 0's DNS name, `rank-0.<service>.<ns>.svc` |
| `BURNIN_ROOT_NODE` | rank 0's node name — for messages, never for addressing |

That is the whole rendezvous contract: which member of the unit this is, and
where the other end answers. There is deliberately **no rank list** — every
collective bootstrap in practice has one rank publish a handle the rest fetch,
and an N-entry list would be a topology the operator has to keep correct rather
than a name the runner resolves. A fabric runner needs nothing else and never
learns anything about Kubernetes.

`BURNIN_PEER_HOST` and `BURNIN_ROOT_HOST` are deliberately **not** qualified to
`cluster.local`. A cluster's DNS domain is configurable, and hard-coding the
default would break the rendezvous silently — as a connection error that looks
like a bad link. `<pod>.<service>.<namespace>.svc` resolves through the pod's own
search path under any cluster domain.

### Precedence and the pod's shape

Env is assembled in this order, and **the last writer wins**: duration and
attempt, then variant axes, then the rendezvous variables, then anything in
`spec.runner.env`. So an operator's explicit `env` entry overrides an injected
default.

Other pod facts a runner can rely on:

| | |
|---|---|
| restart policy | `Never`. **The exit code IS the verdict**, and a restarted container would overwrite the evidence |
| `activeDeadlineSeconds` | `durationSeconds` + 120 s of grace for image pull and container start, plus a further 300 s at Group scope for time spent waiting on ranks the operator has not created yet. An unset `durationSeconds` defaults to 600 |
| eviction | every runner pod carries `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`, unconditionally, at every scope |
| cordon | the pod tolerates `node.kubernetes.io/unschedulable:NoSchedule`, and only that taint |
| hostname / subdomain | at Pair and Group scope, the pod's hostname is its role (`server`, `client`, `rank-N`) and its subdomain is the headless Service. That is what gives it its own A record |

The rendezvous Service is headless and has **no ports**: the operator does not
know, and must not need to know, which port your runner listens on. Ports are the
runner's business; naming is the operator's.

---

## 5. Writing a Pair runner

**A Pair verdict is about the LINK.** Two pods — a server on the first target, a
client on the second — produce exactly **one** result naming **both** nodes. It
is never split per node: attributing a point-to-point measurement to one endpoint
sends an engineer to replace the wrong part.

Three rules the operator applies, which shape what your runner should do:

| rule | consequence for the runner |
|---|---|
| **the client is the deciding side** | it is where perftest and nccl-tests report. Its metrics win any key both ends emit, and the pair settles when it terminates rather than waiting for a server that may legitimately linger |
| verdict precedence is **`Error` > `Fail` > `Skip` > `Pass`** | a machinery failure on either end means the link was never measured. A client's "connection refused" is an artifact of its peer, not evidence about the fabric — recording it as `Fail` would permanently indict a link over an unpullable image |
| **a server that exits 0 before the client ever started is an `Error`** | never a pass. No traffic crossed the link |

So: on the server side, a clean exit is *not* a claim that the link is good.
`ib-write-bw`'s server says so in its own final message — the client's
measurement is the pair's verdict.

### The readiness probe SPENDS a connection

The operator will not create the client until the server pod is Ready. Without a
`spec.runner.readinessProbe`, "Ready" only means the container started — and a
client that connects before its server has bound its socket dies with a
connection error that reads as a fabric fault. A `tcpSocket` probe converts that
gate into a real statement about a listener.

**But TCP admits no way to prove a listener is up other than connecting to it**,
so the probe *is* a real client: the server accepts it, the probe closes it, and
a server that accepts a bounded number of connections has served the probe
instead of the client. The gate meant to guarantee the listener is up is then the
thing that consumes it. `ib_write_bw` and the nccl-tests server both accept a
bounded count.

Two ways out, and take one of them:

- **probe a port the measurement does not use.** `ib-write-bw` listens on its own
  control/readiness port (default 18510), separate from perftest's port (18515);
  the README says so explicitly and the probe is pointed at the former.
- **make the server re-accept.**

The e2e suite's own `nc -l` server had exactly this bug, and the operator
correctly settled the run `Error`.

### If your runner speaks only Pair, refuse Group — before touching the hardware

Every fabric runner branches on `BURNIN_ROLE` and reads its absence as Node
scope. Group scope sets `BURNIN_RANK`/`BURNIN_NRANKS` and deliberately **not**
`BURNIN_ROLE` — so all three of this project's fabric runners would have declared
a `_SKIP` for a collective that never ran, and the run would have settled
`Passed` around it (issue #118).

They now exit **3**, with a message saying the image does not speak the Group
rendezvous, **before any hardware inspection**. The ordering is the point: "this
image cannot do Group scope" is true whatever is in the box, so letting a "no
accelerator visible" skip answer first would reply to a question nobody asked —
and reply with the one verdict that certifies.

Note that **reading `BURNIN_RANK` in order to refuse is not implementing the
contract.** The guard in `runners/pins_test.go` keys on `BURNIN_ROOT_HOST` for
exactly that reason: a first attempt keyed on `BURNIN_RANK`/`BURNIN_NRANKS`
failed immediately, because the refusing runners read precisely those to say no.

---

## 6. Host access

`spec.runner.hostPaths` is the **only** way a runner pod reaches the node's
filesystem. A test that declares nothing gets a pod with **no volumes at all** —
there is no default `/dev`, no convenience `/sys`, nothing.

Each entry is `{path, mountPath, readOnly, type}`. `readOnly` **defaults to
true**. `type` admits only the assert-only hostPath kinds (`Directory`, `File`,
`Socket`, `CharDevice`, `BlockDevice`), so a test can never ask the kubelet to
*create* a path — an operator sent to measure a fleet must not mutate it.

Two measured facts, not inferences:

- **`privileged: true` is not a substitute.** On this project's own nodes, a
  privileged `hostNetwork` pod's `/dev/infiniband` held only `rdma_cm` and
  `umad*`, while the host also had `uverbs*`. `uverbs` is what `ibv_create_cq`
  opens, so `ib_write_bw` dies at "Couldn't create CQ". Mounting the host's
  `/dev/infiniband` is what makes those nodes appear.
- **`/dev/infiniband` must be mounted writable** (`readOnly: false`): the
  user-verbs device nodes are opened read-write by `ibv_open_device`, so a
  read-only mount fails in much the same place as no mount at all.
  `/dev/kmsg` is read-only — a burn-in runner has no business writing to the
  kernel log.

Both pods of a Pair, and every rank of a Group, get identical host access by
construction: the mounts are built in the one function that constructs a runner
pod. A link measured from one end is not measured.

### Degrade honestly

**A runner without its mount must fail loudly, not fabricate a number.** Without
`/dev/infiniband` the fabric runners die at `Couldn't create CQ`. Without
`/dev/kmsg`, `host-health` reports `xid_source=none` and omits `xidEvents`.

A node condemned for a reading nobody took is recoverable. A fleet certified
clean by a fabricated zero is not.

---

## 7. Artifacts: returning evidence that is not a number

A runner returns non-metric evidence — dcgmi's own JSON, a captured topology, a
failing kernel's disassembly — in a fenced block on stdout:

```
-----BEGIN BURNIN ARTIFACT <name> <mediaType>-----
<payload>
-----END BURNIN ARTIFACT-----
```

Both header fields are required and neither may contain a space: a nameless
artifact cannot be referenced and an untyped one cannot be rendered, and
inventing a default for either would put the parser in the business of guessing.

`pkg/runner` lifts fenced blocks out **before** the metric scanner sees a single
line, and that ordering is the whole feature. A dcgmi document is full of
`"key": value` lines, several of which parse — so without it, a runner returning
evidence would silently rewrite its own measurements.

**An artifact is evidence ABOUT a verdict, never part of one.** Nothing here can
change a metric, a verdict or an exit code, and no failure to store one changes
a phase. Every refusal is *recorded* as a dropped reference, because "the runner
produced no evidence" and "the evidence was too large" are different facts and
only one of them sends an engineer to a node.

### The rules, each one a way this could corrupt a verdict

| case | what happens |
|---|---|
| a line inside a fence | it is **payload**, never a metric |
| a BEGIN line whose header does not parse | it **still opens a fence**. The artifact is refused and its payload is consumed. A runner naming an artifact `has space.json` produced a three-field header that once fell through to the metric scanner along with the entire payload it was announcing |
| an unterminated fence | dropped whole, and it swallows the rest of stdout — those lines were the payload, whatever the runner intended. A truncated JSON document fails at report time, far from the runner that produced it |
| a BEGIN marker inside a payload | the outer artifact is refused, and scanning **restarts from that marker** as a fresh BEGIN. The likely cause is two artifacts with a missing END between them |
| a payload over 256 KiB (`MaxArtifactBytes`) | dropped and recorded as dropped, with its true size |

### What the operator does with an accepted artifact

The payload goes into a run-owned ConfigMap, garbage-collected with the run.

- The **name** is refused, not sanitised, if it is not a legal ConfigMap key
  (alphanumerics, `-`, `_`, `.`; not `.` or `..`). A `/` would make the apiserver
  reject the whole object — one runner's malformed name discarding every other
  test's evidence — while rewriting `a/b` to `a_b` would collide two names onto
  one key and point a reference at the wrong payload under the right digest.
- The stored key is `<test>.<pod>.<artifact>` and must be ≤ 253 bytes. The pod
  name carries the attempt, which is what keeps a repeat from overwriting the
  evidence of the attempt before it.
- A run's whole artifact budget is 900 KiB; overflow is recorded as dropped.
- The reference carries a `sha256:` digest over the payload.

### Binary payloads

The channel is **line-oriented** — it is stdout — so a payload is a sequence of
lines and always ends in a newline. **Encode genuinely binary evidence** (base64,
and say so in the media type): a raw blob containing a `0x0a` is
indistinguishable from a line break here, and nothing downstream could tell them
apart either.

A payload that is not valid UTF-8 is stored in the ConfigMap's `BinaryData`
rather than `Data`, and the reason was measured rather than assumed: the
apiserver does **not** reject invalid UTF-8 in `Data`. It stores it and mangles
each bad byte to U+FFFD on the way out to any JSON consumer, kubectl included —
so the evidence looks present, and only the digest reveals otherwise.

---

## 8. Duration, and declaring a kind burst-only

**Read `BURNIN_DURATION_SECONDS`.** Every runner in this repository does, with
one deliberate exception.

`compute-smoke`'s GEMM finishes in milliseconds however long it is asked for,
while a shipped sample requested 120 s — so "node acceptance" burned nothing in
and nothing said so (issue #25). The resolution was to stop pretending rather
than to add a loop: a burst proves the arch-correct instruction path executed and
got the right answer, which a longer run does not strengthen, and looping it
would duplicate `thermal-soak`, `gpu-burn` and `clockprobe` while costing the
fleet its one cheap correctness gate. One kind, one job.

Such a kind declares itself with `TestKind.BurstOnly()`, and `durationSeconds`
then bounds the **pod deadline** only — it does not decide how long the test
runs.

**A duration is an upper bound the caller asked for, not a floor.** `host-health`
is passive, so it clamps the requested window to 30 s (defaulting to 15): a
profile that asked for a ten-minute compute soak did not thereby ask anyone to
stare at sysfs for ten minutes. It also holds its own internal deadline well
inside the pod's, so that a wedged probe — an unresponsive driver makes
`nvidia-smi` hang, which is itself a symptom — cannot become a pod killed at its
`activeDeadlineSeconds`, which would destroy the evidence and be reported as an
infrastructure `Error`.

`ib-write-bw` does the same arithmetic outward: it carves a fixed budget out of
the duration for the latency phase and the two rendezvous handshakes, so the
whole runner finishes inside the pod deadline rather than being killed at the end.

**Finish inside your budget and print your evidence before your verdict.** Both
runners print metrics before the pass/fail line, so a failing run still yields
the numbers it was reached from.

---

## 9. Two worked examples

| runner | shape | what to take from it |
|---|---|---|
| [`runners/host-health`](../runners/host-health/) | stdlib-only probe, no accelerator SDK, Node scope | the three-state rule (`n/a` / omit / never zero) applied per probe; stage breadcrumbs streamed so a hard kill still says where it died; a runner that never skips and therefore prints no marker; degrade per probe, not as a whole |
| [`runners/ib-write-bw`](../runners/ib-write-bw/) | wraps somebody else's tool, Pair scope | why a runner almost never returns 1; refusing Group before touching hardware; a readiness port separate from the measurement port; omitting a metric the mode did not measure; refusing an unknown variant value |

---

## 10. Before you publish

A runner image is executed by a node's readiness gate, against hardware a site is
deciding whether to accept. A bad tag degrades a whole fleet at once.

- **Published tags are immutable.** Never republish a tag a gate pins — not for a
  bug fix, not for a security fix. Cut a new version. Silently changing a tag
  changes every node's verdict with no audit trail.
- **"It builds" is not verification.** An `sm_120a` image *loads* on a CC 12.1
  GB10 and trips a device-side assert inside the kernel; the loader only refuses
  outright in the other direction. Verify the kernel on real hardware.
- Images published by this project are signed; see
  [`verifying-images.md`](verifying-images.md).

---

## Where to go next

| | |
|---|---|
| [`dev/new-testkind-playbook.md`](dev/new-testkind-playbook.md) | the ordered checklist for adding a kind **to this repository**, with the guard that catches each mistake |
| [`concepts.md`](concepts.md) | the CRDs, the scopes, and how a run proceeds |
| [`thresholds.md`](thresholds.md) | authoring thresholds, applicability, and the linter that reads your metric names |
| [`sinks.md`](sinks.md) | the delivery envelope your metrics end up in |
| [`soaks.md`](soaks.md) | what a long-running test costs a fleet |

---

## The shortest possible summary

**A runner may only declare what it positively established.** Every rule above is
that sentence applied to a specific place where it was once got wrong.

# Reports

A sink delivers a verdict. A report turns one into a document somebody reads, or
a pipeline consumes.

The two are deliberately different jobs. [Sinks](./sinks.md) are **transport** —
JSON to a webhook, a ConfigMap or Prometheus — and for a long time that was all
this repository offered. The consequence was that every consumer wrote its own
renderer, and they disagreed about the same run; an operator running this project
standalone got nothing at all, and the only report anybody could produce was a
closed one. `pkg/report` exists to end that: one assembled input, one shared view
model, four renderers that cannot contradict each other because they all read the
same derived facts.

---

## Producing one

```sh
burnin report --from-file   ./delivery.json      -o html      --out ./out
burnin report --results-dir ./results            -o junit     --out -
burnin report --run burnin/acceptance-2026-08-08 -o nvvs-json --out ./out
```

Exactly one input, and passing two is refused rather than prioritised: a user who
passes both `--from-file` and `--run` has a mistaken belief about which is being
read, and silently picking a winner leaves them holding a report about the wrong
run.

`--out -` writes a single document to stdout and **refuses** when the renderer
produced several, naming them. Concatenating N documents into one stream produces
a file that is not valid in its own format, and the NVVS renderer emits one
document per node precisely because that schema is single-host.

`burnin` is its own binary rather than a subcommand of the manager: the manager is
a controller that runs in a cluster under a service account, and this is a thing a
person runs on a laptop against files. Installed as `kubectl-burnin` on `PATH`,
`kubectl burnin report --run ns/name` works with no further plumbing.

### `--run` reads the record, not the status

`--run` reads the `BurnInRun`, then the **ConfigMap sinks** its profile named, and
then a `NodeFingerprint` per node the run touched. The envelopes are the point: the
operator already assembled them, already derived their idempotency keys, and
already decided what a verdict was. Reconstructing a verdict from CRD status would
be a second opinion, and two opinions about one run is the failure this package
exists to prevent.

So a run with no ConfigMap sink cannot be rendered from the cluster, and the
command says so explicitly rather than producing a thinner document. A missing
`NodeFingerprint` is best-effort — the report then says less about the hardware,
which is honest — but a ConfigMap key that will not decode is a hard error, because
a report describing three of a run's four deliveries with nothing saying so is the
one failure mode this package must not have.

### Assembling a pile of envelopes

What a reader actually holds is a pile: several deliveries about one run, in no
order, some superseded. `report.Assemble` turns that into a coherent input.

| Rule | Why |
|---|---|
| The terminal `RunPhaseChanged` delivery is the **verdict of record** and sorts last | It is the run's own account of itself. Earlier `TestCompleted` deliveries describe the same tests and would duplicate every one of them |
| Checkpoints sort by `checkpointSequence` | A long soak's progress is only readable in order; see [soaks](./soaks.md) |
| Deliveries are deduplicated by `deliveryId` | Delivery is at-least-once, so a results directory legitimately holds the same delivery twice, and counting a retry as a second checkpoint shows a soak progressing twice as fast as it did |
| Two different `run.uid` values are refused | Rendering two runs into one document produces a verdict about hardware that was never tested together |
| Envelope JSON is decoded with unknown fields **refused** | A consumer rendering a document from a newer operator must be told it does not understand the whole record. A `baseline` flag it never learned about would turn a measurement sweep into a certification, silently |
| A non-JSON file in a results directory is skipped; a JSON file that is not an envelope is an **error** | A results directory legitimately holds artifacts, logs and a README. It does not legitimately hold an envelope nobody can read |

An unfinished run still renders — that is what a checkpoint is for — but whether
the run was final is carried through to every format, and no renderer may present
a checkpoint as a verdict.

---

## A report never fabricates

This is the rule the package is built around and every renderer inherits it.

**A value that was not captured is ABSENT.** Not the empty string, not `unknown`,
not `N/A`, not zero. The distinction is the same one the verdict engine spends its
whole existence preserving (see [thresholds](./thresholds.md) and
[invariants](./dev/invariants.md)): a measurement nobody took and a measurement
that came back clean are different facts, and a report that renders them
identically converts the first into the second. An RMA conversation opened on a
fabricated serial number is worse than one opened with a gap in it.

The typed shapes express it where they can — a field the caller could not populate
stays at its zero value and renderers omit it — and each format has its own version
of the same discipline:

| Situation | What the document does | What it must never do |
|---|---|---|
| No `startedAt`/`finishedAt` on a result | JUnit omits the `time` attribute | Write `time="0.000"`, which reads as a test that ran instantly |
| A GPU serial was never read | Omitted from the JUnit properties, from `entities[].serial_num`, from the HTML inventory | Emit an empty serial, which is a claim the part has none |
| No inventory was collected for a node | Says "no inventory was collected"; markdown adds that an absent field was not measured | Render a blank row, which reads as hardware that reported nothing |
| A test did not run on a node | An em dash in the HTML matrix, titled "not run on this node" | Show a status, because an absent test is not a passing one |
| A threshold was not evaluated | Stated as such: `NOT EVALUATED: <metric> — <reason>` | Omit it, because an invisible gate reads as a gate that passed |
| A run had no timestamps at all | Start and finish are empty | Substitute `sentAt`, which is when the delivery was sent and not when anything was measured |
| Evidence was too large to store | Surfaced as a warning naming the artifact and the reason | Say nothing, which is indistinguishable from a clean run |

The run window itself is derived rather than invented: the envelope carries no
run-level start or finish, so the window is the earliest execution's start to the
latest execution's finish, which is the run's real occupancy of the hardware.

Two more facts must appear on every format because they change what a phase
**means**:

- **Baseline.** A thresholdless sweep and a certification both end `Passed`, and
  the `baseline` flag is the only thing between them. A consumer gating admission
  on "the last run passed" would otherwise certify hardware against a run that
  gated nothing. JUnit puts it in the suite name, because a CI dashboard shows the
  name and nothing else.
- **Incomplete.** A run that had not finished is labelled as progress, in the
  banner, on the same line as the phase — a reader who takes only the first line
  must not take away a verdict the run never reached.

---

## The four formats

| Format | `-o` | Files | Content type | Reader |
|---|---|---|---|---|
| JUnit XML | `junit` | `junit.xml` | `application/xml` | CI: a provisioning gate, a nightly dashboard |
| HTML | `html` (default) | `burn-in-report.html` | `text/html; charset=utf-8` | A person accepting a delivery, or assembling a warranty claim |
| Markdown | `markdown` | `burnin-report.md` | `text/markdown; charset=utf-8` | An issue comment, a CI post, a terminal over SSH |
| NVVS-schema JSON | `nvvs-json` | `diag-<node>.json` per node, plus `diag-unassigned.json` when needed | `application/json` | Vendor-shaped ingest, and an engineer who already reads `dcgmi diag -j` |

All four are pure functions of the assembled input and none of them performs I/O.
A renderer that could read a file or call an API would produce documents that
depend on when they were rendered, and a burn-in report's whole value is that it
describes one moment that has already passed. A renderer also must not mutate its
input: callers render the same run through several formats, and one that normalised
something in place would change what the next one saw — two formats of one run
disagreeing being exactly what this package was created to end.

### The shared view model

`report.BuildView` derives the model every renderer reads. Its most important
property is structural rather than conventional: **a Pair or Group result lives in
`View.Links` and is unreachable through `View.Nodes`.** A point-to-point
measurement is a property of the link, and attributing it to one endpoint sends an
engineer to replace the wrong part — so no renderer can split one by accident,
because there is no path to do it. See [concepts](./concepts.md) for what Pair and
Group scope mean.

A result that names no node at all — an unresolvable test, a config error — still
belongs to the report, under the empty node name. A required acceptance test that
never ran must never vanish from the document that certifies the fleet.

---

## JUnit

JUnit is how a burn-in result enters a pipeline. The mapping is the whole format:

| Phase | Element |
|---|---|
| `Error` | `<error>` |
| `Failed` | `<failure>` |
| `Skipped` | `<skipped>` |
| `Passed` | nothing; the case carries only its metrics |
| anything else | `<error>`, typed `Unknown` |

**`Error` becomes `<error>` and never `<failure>`.** Collapsing them would erase
the distinction the verdict engine exists to preserve, on the surface most likely
to be automated against: an unpullable image and a degraded GPU would look
identical on a dashboard, and only one of them is a reason for anybody to touch
hardware. JUnit has had a separate `<error>` element since Ant, precisely for "the
test could not run", so nothing is being invented — the lossy mapping is the choice,
not the honest one.

A phase this build does not recognise — an older or newer operator — is an
`<error>` too, never a silent pass. A result nobody can classify has certified
nothing, and reporting it green is the one outcome that lets unmeasured hardware
into a fleet.

Beyond the mapping:

- **A link is one test case**, in a dedicated `links` suite, with every endpoint in
  the case name so a red dashboard row is self-describing.
- **The `<failure>` body spells out every gate that was missed**, with its cause.
  The result's `message` names only the first violation and that field is frozen,
  so a run that failed three gates for three different reasons would otherwise
  reach a dashboard as one sentence about one of them.
- **The cause is translated into who should act**, because a CI log is read by
  whoever the build woke up: `hardware fell short` (Measurement), `unjudged: the
  runner's report could not support a judgement` (Evidence), `the threshold itself
  is broken; no node should be touched` (Authoring).
- **The `<error>` body says the hardware was not judged**, in words, for the same
  reason.
- `system-out` carries what the run could not measure and what raw evidence exists,
  including artifacts that were **not kept** and why.
- Suite properties carry only captured node identity. `<properties/>` is omitted
  entirely when nothing was captured, because an empty element suggests a test that
  measured nothing on purpose.

---

## HTML and markdown

### One file, no network. Forever.

The HTML report is a single self-contained file: CSS inline, no JavaScript, no
fonts to fetch, no images to miss. A report that fetches a stylesheet is useless in
a secure facility, useless as an email attachment, and useless in eighteen months
when the CDN path has moved and the document that was meant to be evidence renders
as unstyled text. `TestTheDocumentMakesNoExternalRequests` asserts it against the
rendered bytes rather than trusting it, and it also rejects `<script>` — a document
that is evidence should not depend on a JS engine behaving.

There is a print stylesheet, with `break-inside: avoid` on the cards, because this
document gets printed and attached to claims.

### Error and Fail are distinguished in TEXT, not only in colour

Every status appears as a word — `PASS`, `FAIL`, `ERROR`, `SKIP`, `CANCELLED` — and
`Error` and `Failed` get **different hues, not two shades of red**. Both halves are
load-bearing and both are pinned by tests: the words survive greyscale printing and
colour-blind readers, and the separate hues stop a glance conflating two states
that lead to different actions. One says the hardware fell short; the other says it
was never judged, and a reader who cannot tell them apart cannot tell whether to
send an engineer.

The violation cause is spelled out as a sentence rather than left as a one-word
enum a reader has to look up, because it is the most decision-relevant thing in the
document — the difference between walking to a rack and fixing a profile.

### Markdown

Markdown shares `build()` with the HTML renderer, and that sharing is the point
rather than a convenience. The two formats describe the same run to the same people
and would eventually word "who should act" differently if they derived it
separately — a divergence no code review catches, because it appears months later,
in one format only, in front of somebody deciding whether to accept a delivery.
Every phrase that carries meaning comes from the shared page; the markdown file
only decides layout.

It stays plain — no HTML fallbacks, nothing that renders as literal angle brackets
in a terminal — and it escapes `|` in every value that reaches it from a runner. A
pipe in a test name, a metric name or a threshold reason silently splits the row it
is in, so a reader sees a shifted table rather than the values that were measured.
Backticks do not protect it: a table row is split into cells before code spans are
parsed, which is why the escape is applied even to values already wrapped in
backticks. That looks redundant and is not — it was written the redundant-looking
way first, and was wrong.

An unrecorded field renders as an em dash, not an empty cell: "not recorded" and
"recorded as nothing" must not look the same.

---

## NVVS-schema JSON

This is the format that needs the most care, so read this section before using it.

### Why borrow someone else's schema

An engineer qualifying accelerators already knows what `dcgmi diag -j` output looks
like. Their tooling parses it, their runbooks reference its fields, and an RMA
conversation goes faster when the attachment has the shape the vendor's own tool
produces. This operator measures strictly more than DCGM does for the tests it runs
— per-threshold violations classified by cause, link-scoped verdicts, gates that
were measured rather than assumed — and delivering all of that in a shape nobody
reads is a self-inflicted wound.

### What compatibility does and does not claim

**These documents are schema-compatible with DCGM's diagnostic JSON. They are not
produced by NVIDIA software, are not endorsed by NVIDIA, and are not a claim to
have run NVIDIA's tests.**

| Claimed | Not claimed |
|---|---|
| The field names are the ones NVIDIA's own `nvvs/include/NvvsJsonStrings.h` (Apache-2.0) defines, kept as constants so a rename is a single edit and a typo is a compile error | That any NVIDIA code ran, or that a vendor diagnosed anything |
| The document tree matches a real capture: `metadata`, `entity_groups` and `Overall Result` are root siblings of `DCGM Diagnostic`, which nests only `test_categories` | That the tree is right in the parts the capture does not exercise |
| The categories (`Deployment`, `Integration`, `Hardware`, `Stress`) are the vendor's taxonomy | That the tests inside them are the vendor's tests |
| A consumer that iterates `test_categories` generically will read the document | That a consumer keying on a specific NVIDIA plugin name will find one |

The tree was originally **inferred, and inferred wrong** — `metadata`,
`entity_groups` and `Overall Result` were nested inside `DCGM Diagnostic`, where no
real consumer looks. The failure was silent and total: every key name was already
correct, so nothing looked wrong, and a consumer simply got a document with no
driver version, no serials and no entity inventory. The capture that settled it was
already in this repository — `dcgm4GB10` in `runners/dcgm-diag/diagjson_test.go`, a
genuine `dcgmi diag -j` document from this project's own GB10 fleet under DCGM
4.2.3 — and `TestTheDocumentTreeMatchesARealCapture` reads it out of that file
rather than copying it, so the tree cannot be re-inferred. That guard asserts
*placement*, never presence: requiring a key to be present would put it in direct
conflict with the never-fabricate rule, and one of the two would lose.

### Our names stay ours

A synthesised entry is **never** labelled with an NVIDIA plugin name — not
`sm_stress`, not `targeted_stress`, not `memtest`, not `diagnostic`. A consumer
must not be able to mistake our reconstruction for that plugin's own result.
`TestNoNVIDIAPluginNameIsEverEmitted` checks the rendered bytes against a list of
them. A link result is additionally suffixed ` (link)` so the category alone cannot
mislead.

### No vendor error code is ever invented

NVVS carries NVIDIA's own `DCGM_FR_*` codes in `error_id`. **This renderer never
emits `error_id` and never synthesises a `DCGM_FR_*` string** — doing either would
let a consumer look up a vendor diagnosis for a condition the vendor never
diagnosed. `TestNoErrorIDIsInvented` greps the rendered document for both.

What goes in a `warning` instead is ours and is strictly additive:
`error_category` carries the violation's `Cause` verbatim — the field that says
whether a human should walk to a rack or fix a profile, and one NVVS has no
equivalent for — and `error_severity` is a coarse ranking derived from it.
`Evidence` and `Authoring` are deliberately **not** `error` severity: neither
implicates hardware, and a consumer triaging by severity should not be sent to a
rack over a broken threshold.

The numeric `entity_group_id` is omitted for the same class of reason: its values
come from a DCGM enum this project does not vendor, and a wrong number is worse
than a missing one — a consumer keying on it would attach our results to the wrong
kind of device.

### Provenance appears in four places

A schema-compatible report with no provenance is indistinguishable from vendor
output once it has been forwarded twice. So the document says what it is four times
over, and each placement is meant to survive a different hop:

| Where | Value | The hop it survives |
|---|---|---|
| `"DCGM Diagnostic".version` | `NVVS-compatible` | A consumer that reads only the fields dcgmi itself defines. It looks where a DCGM version number would be and finds a string that plainly is not one |
| `aux_data.schema` | `NVVS-compatible` | A machine that indexes fields: one token to key on, with no prose to parse |
| `aux_data.generator` | `glimmer-burnin <version>` | Attribution after forwarding — which tool, which build |
| `aux_data.notice` | "Schema-compatible with the DCGM diagnostic JSON format. Generated by glimmer-burnin; NOT produced by, or endorsed by, NVIDIA. Test names are this project's own and do not correspond to NVIDIA plugins." | A screenshot, a paste into a ticket, a quoted fragment — anywhere the field names are lost and only the sentence survives |

`aux_data` also carries `run_uid`, `run_name`, `profile`, `delivery_id`, `cluster`,
`generated_at`, `node`, the `baseline` flag, and any warnings about evidence that
was not stored — so a document that has travelled can still be traced back to the
delivery it came from. `TestTheNoticeSurvivesEncoding` checks the disclaimer is in
the encoded bytes rather than merely in a struct a renderer might forget to fill.

`metadata.version` — DCGM's own version — is **never emitted**. This project ships
no DCGM, and inventing a version would be the exact fabrication the package
refuses.

### One document per node

NVVS is single-host by construction: dcgmi runs on one machine, and the schema has
no way to say "these results are from eight machines". So a multi-node run renders
one document per node rather than a single document that misrepresents its own
shape. Three sources decide the set, and each was a silent loss on its own:

1. nodes named in results;
2. nodes the caller supplied inventory for — a node that was targeted and produced
   nothing is exactly the case a reader most needs to see, and omitting it made the
   report read as a fleet that was fully measured;
3. `diag-unassigned.json`, whenever a result cannot be attributed to a named node.
   Those used to be dropped once a real node existed to enumerate instead. It is
   emitted only when something lands in it, so a healthy run does not grow an empty
   file nobody can explain.

A **link** result appears identically in both endpoints' documents, never split,
with `link=`, `peer=` and a note in `info[]` saying the verdict is about the link
and not about this endpoint. A result that names no node appears in the unassigned
document and in **no** other — it used to appear in every one, which inflated a
single config error into one finding per node in the fleet.

### Status mapping, and the one that matters

| Phase | NVVS status |
|---|---|
| `Passed` | `Pass` |
| `Failed` | `Fail` |
| `Skipped` | `Skip` |
| `Error` | `Not Run` |
| `Pending`, `Running`, `Cancelled` | `Not Run` |
| unrecognised | `Not Run` |

`Error` maps to **`Not Run`, never `Fail`**. NVVS has its own vocabulary for
"unjudged", so the distinction survives the schema crossing intact rather than
being approximated — and this is the boundary where losing it would be
unrecoverable, because nothing downstream could tell the two apart again.

A **baseline** run's results are all `Skip`, applied in exactly one place
(`statusUnder`). A baseline run applied no thresholds, so no result met a
criterion; rendering one as `Pass` is how a measurement sweep becomes
indistinguishable from a certification to anything reading the status field.

`Overall Result` is an **admission gate rather than a diagnosis** — both `Fail` and
`Not Run` answer "do not admit this node", and the diagnosis lives in each result's
own status. It is computed per document, so a node that passed does not report
`Fail` because a different node did. Its precedence has one hard-won rule:

> **`Fail` beats `Not Run`.**
>
> This used to rank `Error` first, justified as being pessimistic. The engine ranks
> the other way — a required test that failed is a hardware verdict and wins
> outright — so a run the operator settled `Failed` rendered the node's verdict as
> `Not Run`, which in real dcgmi means a plugin that was not selected and which
> consumers filter as infrastructure noise. A GPU that missed a sustained-clock
> gate was queued for a re-run instead of an RMA, because an unrelated image pull
> failed on the same node. It is the Error/Fail collapse this package exists to
> prevent, running in the direction nobody checks.
>
> The `Error`-beats-`Fail` precedence that *does* exist in this project is the Pair
> rule, and it answers a different question: it combines the two ends of one
> measurement. Aggregating independent tests on a host is not that case.

### A node verdict binds to no device when there is more than one

On a node with exactly one GPU, the result binds to that device. On a node with
several, it binds to **none of them** — one `results[]` entry, entity-less, with a
prose line naming the devices present.

It used to broadcast the node's verdict to every accelerator, with an `info[]` line
saying the per-device attribution had not been measured. The line was honest and it
did not reach the consumer that matters: NVVS consumers **count** `results[]`
objects, which is the normal reason to parse this format at all. One failing
thermal-soak on an eight-GPU node was eight objects with status `Fail`, and every
tallying tool reported eight failed GPUs — an engineer replacing seven healthy
parts, or a fleet manager quarantining eight devices. Once the document has been
forwarded or re-serialised, eight objects are eight results and nothing carries the
fact that they were one.

It was also more wrong than "over-attribution" suggests: `clockprobe`,
`thermal-soak` and `gpu-burn` run on one CUDA device and report for that device
alone, so the verdict was never about all eight — it was about one the ordinal
cannot identify, since MIG and `CUDA_VISIBLE_DEVICES` both remap it. The device
that was actually measured is already in `info[]`, because every metric is emitted
there as `key=value` and a runner that reported `pciBusId` has said which part was
under load. An entity-less `results[]` entry is a shape DCGM itself produces — it is
exactly what `test_summary` is — so binding to nothing costs no compatibility.

`GPU Device IDs` and `GPU Device Serials` are **deliberately never populated**, for
a related reason: they are positional arrays built by appending non-empty values,
so with GPU 0's serial uncaptured and GPU 1's captured the array is `["SN-GPU1"]`
and a positional reader attributes it to GPU 0. That names the wrong part in the
one document somebody attaches to an RMA. The shape cannot express a partial set at
all, which is why DCGM's live form is a per-entity `serial_num` — self-describing,
and simply absent for a device that has none. Neither key appears in the real 4.2.3
capture either.

### A partial record is not a confident document

`contract.Summary` is an independent count, produced by the operator from its own
status rather than derived from the results carried in the envelope. When the
summary accounts for more executions than the record contains, what we hold is a
subset — a webhook that dropped a delivery, a ConfigMap that was truncated, a
results directory copied mid-run — and **every** `Overall Result` in that render
becomes `Not Run`, with the discrepancy stated in the document's `Warning`. A
confident document over an inconsistent record is how a fleet gets signed off
against the results that happened to be delivered.

The check is deliberately one-directional. More results than the summary counts is
normal — repeats and retries legitimately produce several results per execution —
so that direction is not flagged.

### A genuine vendor document is evidence in its own right

`report.Input.Artifacts` carries raw evidence a runner returned — a dcgmi JSON
document, a captured topology. The rule attached to it is that **a renderer must
never re-synthesise something it was handed for real**: a genuine vendor document
and our reconstruction of one are different evidence, and only the first is worth
attaching to an RMA. Merging a real `dcgmi diag -j` document into a synthesised
NVVS-shaped one would destroy exactly that difference — the result would carry
vendor-diagnosed findings and our own in one tree with nothing in the schema to
separate them, and the provenance table above would then be describing a document
that is only partly ours. A real vendor document belongs beside the rendered
report, as its own file, under its own digest.

Being precise about the current state: **no shipped renderer inlines an artifact
payload, and neither the CLI's `--run` path nor its file-loading paths populate
`Input.Artifacts` today.** What every format does carry is the artifact by
*reference* — its name, media type, size, and the ConfigMap and key it was stored
under, or the reason it was **not kept**. That last one is not cosmetic: "produced
no evidence" and "the evidence was too large" are different facts and only one
sends an engineer to a node, so a drop is reported on every surface (JUnit
`system-out`, HTML and markdown warnings, NVVS `info[]` and `aux_data.warnings`)
rather than swallowed. Fetching payloads and writing them out beside the rendered
documents is the natural shape for this to take; it is not implemented, and this
page does not claim it is.

### What is not established

Stated plainly, because a confident wrong sentence here is worse than a gap:

- **`entity_group_id` values** are unverified, which is why the field is omitted
  rather than guessed.
- **`Overall Result` at the root** is our placement. The key name is the vendor's,
  from `NvvsJsonStrings.h`, but the pinned capture does not contain that key at
  all, so the tree guard cannot check where a real one would sit.
- **The `entity_groups` inventory only ever declares a `GPU` group**, built from
  the caller's supplied inventory. A host-scoped kind (`host-health`,
  `memory-stress`, `disk-io`, `tcp-baseline`) attaches its result to
  `entity_group: "CPU"`, which the document's own `entity_groups` does not
  enumerate.
- **`metadata` is emitted only when a driver version was captured**, since that is
  the only field of it this renderer ever populates.
- The **category placement** of each `TestKind` is a judgement call, not a lookup
  of anything authoritative: load generators are `Stress`, passive probes and
  single-shot correctness checks are `Hardware`, anything measuring a path
  *between* things is `Integration`, and an unrecognised kind lands in `Hardware`.

---

## `pkg/report` holds no Kubernetes types, and that is a promise to consumers

`pkg/report` and its renderer subpackages depend on `pkg/contract` and the standard
library. **Nothing else** — not `api/v1alpha1`, and not `pkg/verdict`, which takes
that dependency (and apimachinery with it) because it must speak `Threshold`.

The audience is CI jobs, a CLI and third-party ingest, which is exactly the
audience that should not be pulling in client-go to format a table. A consumer can
depend on `pkg/report` to render a stored envelope without inheriting the
Kubernetes API machinery, its version constraints, or its upgrade cadence.

`TestNoKubernetesDependency` asserts it against the **real build graph** — `go list
-deps ./...` — rather than by reading the imports at the top of a file, because the
dependency that lands is never the one you can see: a helper added later to "just
reuse the CRD types" would drag the whole tree in behind it. The guard also refuses
to pass on an implausibly small graph, and requires `pkg/contract` to be present,
so it cannot succeed by reading nothing.

The consequence you will meet in the code is `report.NodeInfo`, `GPUInfo` and
`NICInfo`: plain-Go **mirrors** of the `NodeFingerprint` status, populated by the
caller. The conversion lives in the CLI, which is allowed Kubernetes types; the
rendering package is not.

The terminal run phases are pinned in `report.TerminalPhases` as literal strings
for the same reason — they are what a stored envelope actually contains, and an
envelope written years ago must still render, so pinning the wire strings is more
honest than importing constants that could be renamed without the format changing.
The drift risk is closed from the side that can see both: `internal/sink`'s
`TestReportKnowsEveryTerminalPhase` fails if the API grows a terminal phase this
list misses, which would otherwise make those runs render as unfinished forever —
a terminal `Failed` run reported as "still running".

---

## What a report does not claim

Every HTML and markdown document ends with the sentence that bounds all of this:

> Thresholds in this report are authored by the site that ran it. A pass is a
> statement about the gates that were applied, not a warranty.

A report describes a run. Whether that run was a good acceptance test is a question
about the profile, and that is [thresholds](./thresholds.md)' subject.

---

## See also

- [Concepts](./concepts.md) — the CRDs, the scopes, how a run proceeds
- [Sinks](./sinks.md) — the delivery envelope a report is built from
- [Thresholds](./thresholds.md) — what a violation's `Cause` means and where it comes from
- [The runner contract](./runner-contract.md) — where metrics, artifacts and the `Error`/`Fail` split originate
- [Invariants](./dev/invariants.md) — the project rules the renderers inherit

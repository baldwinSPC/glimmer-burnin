# Roadmap

Where this project is, where it is going, and why in that order.

This document is a map, not a promise. Milestones here correspond to GitHub
milestones, and every item links to an issue that can be read on its own. When
something lands, this page should change; when something is deferred, the reason
belongs in the issue rather than here.

---

## What this is

A vendor-neutral, Kubernetes-native burn-in operator: it runs acceptance tests
against accelerator nodes, evaluates the results against declared thresholds,
and exports verdicts to an external consumer.

Three properties are worth stating up front, because they explain most of the
decisions below.

**One brain, two dispatchers.** The evaluation logic — how a runner's output is
parsed, how a threshold is judged, what a verdict means — lives in public
packages (`pkg/runner`, `pkg/verdict`, `pkg/contract`) that hold no Kubernetes
types. The operator is one caller. A second caller that runs the same images on
a bare host reaches the same verdict on the same evidence, which is the point:
two dispatchers that disagree about the same hardware are worse than one.

**Verdicts fail closed, and `Error` is not `Fail`.** A threshold naming a metric
the runner did not emit is a failure, not a pass — a missing measurement must
never quietly satisfy acceptance. Separately, an infrastructure error (an image
that would not pull, a pod that was evicted) stays distinguishable from a
hardware verdict, because only one of those is a reason to touch a node, and
only one of them is safe to retry.

**Thresholds are measured, not assumed.** This project has a documented instance
of a threshold derived from a specification sheet that would have failed every
healthy node in the fleet: a fabric whose wire speed is 200 Gb/s, attached to a
host that cannot exceed roughly 100 Gb/s. The gate read as a fabric fault. The
rule that came out of it is to ship thresholdless, measure the fleet, then pin.

---

## Where it is today

Node, Pair and Group scopes all execute. Eleven runner images and the operator
publish for `linux/amd64` and `linux/arm64`. Verdicts carry every violated
threshold classified by cause, so a reader can tell a hardware shortfall from an
unusable report from a broken gate. Thresholds are linted at three surfaces:
apiserver patterns, a plan-time refusal for gates that can never be satisfied,
and an advisory condition for gates that are satisfiable but do not mean what
their author intended. There is an envtest suite against a real apiserver, a
kind-based end-to-end suite including the Group rendezvous, and chaos tests.

The runner contract has three rules that arrived from real incidents and are
worth knowing before reading anything below:

- **A skip must be declared.** Exit 2 with no `*_SKIP` marker on stdout is an
  `Error`, not a skip. A panicking Go runner exits 2 and prints a stack trace,
  which is byte-for-byte the shape of a legitimate skip — whose normal form is
  "no metrics at all".
- **`n/a` is a claim about the hardware**, and only a runner may make it. It
  means "I looked and this part has nothing to report". Absence means something
  different and must not be confused with it: a metric that simply never appears
  still fails its gate, under every applicability, because a crashed probe must
  not become an acceptance.
- **A scan that failed is not a sample.** A counter whose source could not be
  read is omitted, not reported as zero.

---

## The test matrix

Duration classes: **smoke** (≤5 min) · **standard** (~30 min) · **extended**
(2–8 h) · **soak** (24–72 h) · **marathon** (1–2 weeks). Anything past
*extended* depends on the segmented-soak engine ([#157]).

"Shipped" means an image is published and has been run on real hardware of at
least one supported kind. Unverified is tracked as its own state, deliberately.

| TestKind | Measures | Scope | NVIDIA | AMD | Intel | Status |
|---|---|---|---|---|---|---|
| `compute-smoke` | FP4 block-scaled GEMM, exact instruction path | Node | shipped | — | — | shipped (burst-only) |
| `gemm-sweep` | GEMM across FP64→FP4/INT8 | Node | planned | later | — | [#160] |
| `gpu-burn` | Sustained compute, SDC detection | Node | shipped | later | — | shipped |
| `thermal-soak` | Power and thermal behaviour, throttling | Node | shipped | later | — | shipped |
| `clockprobe` | Sustained clock under load | Node | shipped | later | — | shipped |
| `power-swing` | Load transients, VRM/PSU behaviour | Node | planned | — | — | [#166] |
| `memory-bw` | Host↔device and device-local bandwidth | Node | shipped | planned | — | shipped · [#170] |
| *(p2p cases)* | All-pairs peer bandwidth | Node | planned | planned | — | [#161] |
| `memory-stress` | Host DIMM stress | Node | shipped | shipped | shipped | shipped (vendor-free) |
| `host-health` | Xid, ECC, PCIe AER, NIC counters | Node | shipped | planned | — | shipped · [#171] |
| `dcgm-diag` | Vendor diagnostic suite | Node | shipped | — | — | shipped (site-supplied DCGM) |
| `rvs-diag` | AMD vendor diagnostic suite | Node | — | planned | — | [#169] |
| `xpum-diag` | Intel vendor diagnostic suite | Node | — | — | spike | [#172] |
| `ib-write-bw` | RDMA write bandwidth (and latency) | Pair | shipped | shipped | shipped | shipped · [#163] |
| `gpudirect-rdma` | GPU↔NIC peer-memory path | Pair | shipped | — | — | shipped |
| `nccl` | Collective bandwidth | Pair, Group | shipped | planned | later | shipped · [#118] · [#170] |
| `fabric-soak` | Iterated collectives over hours | Pair, Group | planned | planned | — | [#162] |
| `tcp-baseline` | Plain TCP throughput and retransmits | Pair | planned | planned | planned | [#164] |
| `disk-io` | NVMe throughput and latency | Node | planned | planned | planned | [#165] |

Beyond the kinds themselves, the matrix has axes: **variants** ([#155]) express
one test across precisions, message sizes or duration classes; **baseline mode**
([#142]) runs a profile thresholdless to gather the numbers a gate should be
pinned to.

---

## Milestones

Status lives in the issues, not on this page — GitHub already tracks what is
open. What follows is the reasoning and the ordering, which is the part worth
writing down. Items are marked landed here only where leaving them unmarked
would make the surrounding prose wrong.

### R1 — Contract and reporting

The delivery envelope carried less than the operator knew: per-threshold
violations, classified by cause, existed on the CRD status and reached an
external consumer only as one sentence of prose. Everything downstream of the
operator is built on the envelope, which is what makes this the critical path.

Additions here must stay additive, and that is not only good manners: consumers
import the contract package directly and may be pinned to an older version than
the operator, so a newer envelope has to decode cleanly against an older struct.

The widening itself has since landed: violations with their causes,
not-evaluated thresholds and unmeasurable metrics now cross the boundary as data
([#139]); an envelope names the cluster it came from and its summary accounts for
all four phases rather than two ([#140]); and every metric declares how it
combines across windows ([#141]), which is what lets a segmented soak render one
verdict over many windows.

- [#142] A thresholdless sweep is indistinguishable from a certification
- [#143] A runner has nowhere to put evidence that is not a metric
- [#144] `pkg/report`: a stored verdict has no renderer
- [#145] Render a run in NVIDIA's own diagnostic schema
- [#146] One HTML file an engineer can open offline
- [#147] JUnit, and a command that reaches both dispatchers

### R2 — Bare-metal CLI

Burn-in at provisioning time, before a machine is a cluster member. This matters
most for exactly the hardware that needs it: a node whose defect prevents it
from joining a cluster cannot be tested by something that requires it to have
joined one.

Most of the work is sequencing. The runners already have no Kubernetes
assumptions — a peer address can be a plain IP, a Pair server learns its peer
from the connection rather than from DNS, and memory locking is easier to
arrange under `docker run` than under a PodSpec.

- [#149] The default image table is locked inside `internal/`
- [#150] Nothing can describe a host that is not a cluster member
- [#151] Path A does not exist in the open
- [#152] `cmd/burnin`: the command a provisioning script can call
- [#153] Measure a link between two machines that are not in a cluster
- [#154] Deliver results anywhere, and ship a signed binary

### R3 — Test matrix and soak engine

A seven-day soak is currently one pod. Any eviction, drain or kubelet restart
ends the week, and a retry starts it over.

- [#155] One test, many cells: the matrix has no word in the API
- [#156] The pinned plan cannot hold a matrix
- [#157] A week-long soak is one pod, and one eviction ends the week
- [#158] A quiet run still wakes every 30 seconds for a week
- [#159] A multi-day run is one autoscaler decision away from Error

### R5 — Suite expansion

- [#160] One precision is not a compute verdict: sweep the GEMM
- [#161] `memory-bw` never measures the links between devices
- [#162] A minute of clean traffic does not find a flapping optic
- [#163] `ib-write-bw` reports a latency it cannot be asked for — the tail
  (p99, min, max, message rate) now reaches the result; asking for latency as its
  own gated execution still waits on variants ([#155])
- [#164] When a fabric test fails, nothing answers whether plain TCP works
- [#165] Storage is the one subsystem the suite never touches
- [#166] Steady load never tests the power path's transients

### R6 — Heterogeneous fleets

Vendor support is data — a fingerprint and a runner image — never a controller
branch. Adding an accelerator should mean adding a runner, not adding an
`if nvidia {}`.

- [#167] One profile cannot serve a fleet with two vendors in it
- [#168] A node with no labels has no vendor, and no vendor has no image
- [#169] AMD has a vendor diagnostic suite and no wrapper
- [#170] `nccl` and `memory-bw` are vendor-neutral kinds with NVIDIA-only images
- [#171] `host-health` barely measures an AMD node
- [#172] Intel is a claimed vendor with no code path
- [#173] A mixed fleet has no documentation to follow

### R7 — Open-source launch

This repository is already public. The items below are the ones that would
normally have happened first, which makes a couple of them live rather than
preparatory — most obviously the absence of any non-public way to report a
vulnerability in software that runs on production hardware.

- [#176] A stranger cannot report a vulnerability — **the live one**
- [#174] There is still no supported way to install the operator
- [#175] Nothing this project publishes is signed
- [#177] The licence policy is enforced by prose, not by CI
- [#178] The design lives in a file addressed to robots
- [#179] The launch checklist that never happened, including the module-path
  decision — now urgent, because an external consumer already imports these
  packages and Go module paths do not follow redirects

### R-loop — Test factory

Adding a TestKind is currently a research project, because the rules that govern
one are spread across a design document, eleven runner READMEs, and a set of
guard tests. Writing them down once is what makes a new runner a routine change.

- [#180] Adding a TestKind is a research project every time
- [#181] A runner proposal has no shape

---

## On report formats

One of the planned renderers emits the schema NVIDIA's `dcgmi diag -j` uses.
That deserves a precise statement of what is and is not being claimed.

**What it is:** documents whose structure and key names match the NVVS schema,
so tooling that already parses DCGM diagnostic output can parse these too, and
an engineer who reads those reports does not have to learn a second layout.
Every document carries provenance in `aux_data` naming this project as the
generator and stating plainly that it was not produced by NVIDIA software.

**What it is not:** NVIDIA output, an NVIDIA-endorsed format, or a claim to
have run NVIDIA's tests. Our test names stay ours — a synthesised entry is never
labelled with an NVIDIA plugin name, so it cannot be mistaken for one. Fields we
did not measure are absent rather than filled in: a serial number that was not
captured is omitted, never guessed and never empty-stringed. Where a real
`dcgmi -j` document exists as an artifact, it passes through verbatim instead of
being re-synthesised.

The same discipline applies to every other format. A report that quietly omits
evidence is worse than one that says the evidence was too large.

---

## How new tests get added

The intent is that adding a TestKind becomes routine rather than exceptional.
Three things make that possible, and they are prerequisites rather than nice to
have:

1. **A written checklist** — [docs/dev/new-testkind-playbook.md] — so the rules
   are in one place rather than distributed across the codebase's memory.
2. **Mechanical enforcement.** Most of the rules are already guarded by tests
   that fail the build: every upstream reference pinned to a commit, every skip
   marker recognised by the parser, every kind either honouring a duration or
   declaring itself burst-only, no runner compiling against an unresolved
   architecture, no binary committed at the repository root. The checklist tells
   you what the guards will catch; it is not a substitute for them.
3. **A proposal template** ([#181]) whose fields are the specification an
   implementation needs, so the answers are captured once rather than
   reconstructed later.

Verification on real hardware remains a human gate. Runner images publish
manually and only after the kernel has been verified on the silicon it targets,
because a runner image is executed by a node's readiness gate and a bad tag
degrades a whole fleet at once. Published tags are immutable; a fix is a new
version, never a re-push.

[#118]: https://github.com/baldwinSPC/glimmer-burnin/issues/118
[#139]: https://github.com/baldwinSPC/glimmer-burnin/issues/139
[#140]: https://github.com/baldwinSPC/glimmer-burnin/issues/140
[#141]: https://github.com/baldwinSPC/glimmer-burnin/issues/141
[#142]: https://github.com/baldwinSPC/glimmer-burnin/issues/142
[#143]: https://github.com/baldwinSPC/glimmer-burnin/issues/143
[#144]: https://github.com/baldwinSPC/glimmer-burnin/issues/144
[#145]: https://github.com/baldwinSPC/glimmer-burnin/issues/145
[#146]: https://github.com/baldwinSPC/glimmer-burnin/issues/146
[#147]: https://github.com/baldwinSPC/glimmer-burnin/issues/147
[#149]: https://github.com/baldwinSPC/glimmer-burnin/issues/149
[#150]: https://github.com/baldwinSPC/glimmer-burnin/issues/150
[#151]: https://github.com/baldwinSPC/glimmer-burnin/issues/151
[#152]: https://github.com/baldwinSPC/glimmer-burnin/issues/152
[#153]: https://github.com/baldwinSPC/glimmer-burnin/issues/153
[#154]: https://github.com/baldwinSPC/glimmer-burnin/issues/154
[#155]: https://github.com/baldwinSPC/glimmer-burnin/issues/155
[#156]: https://github.com/baldwinSPC/glimmer-burnin/issues/156
[#157]: https://github.com/baldwinSPC/glimmer-burnin/issues/157
[#158]: https://github.com/baldwinSPC/glimmer-burnin/issues/158
[#159]: https://github.com/baldwinSPC/glimmer-burnin/issues/159
[#160]: https://github.com/baldwinSPC/glimmer-burnin/issues/160
[#161]: https://github.com/baldwinSPC/glimmer-burnin/issues/161
[#162]: https://github.com/baldwinSPC/glimmer-burnin/issues/162
[#163]: https://github.com/baldwinSPC/glimmer-burnin/issues/163
[#164]: https://github.com/baldwinSPC/glimmer-burnin/issues/164
[#165]: https://github.com/baldwinSPC/glimmer-burnin/issues/165
[#166]: https://github.com/baldwinSPC/glimmer-burnin/issues/166
[#167]: https://github.com/baldwinSPC/glimmer-burnin/issues/167
[#168]: https://github.com/baldwinSPC/glimmer-burnin/issues/168
[#169]: https://github.com/baldwinSPC/glimmer-burnin/issues/169
[#170]: https://github.com/baldwinSPC/glimmer-burnin/issues/170
[#171]: https://github.com/baldwinSPC/glimmer-burnin/issues/171
[#172]: https://github.com/baldwinSPC/glimmer-burnin/issues/172
[#173]: https://github.com/baldwinSPC/glimmer-burnin/issues/173
[#174]: https://github.com/baldwinSPC/glimmer-burnin/issues/174
[#175]: https://github.com/baldwinSPC/glimmer-burnin/issues/175
[#176]: https://github.com/baldwinSPC/glimmer-burnin/issues/176
[#177]: https://github.com/baldwinSPC/glimmer-burnin/issues/177
[#178]: https://github.com/baldwinSPC/glimmer-burnin/issues/178
[#179]: https://github.com/baldwinSPC/glimmer-burnin/issues/179
[#180]: https://github.com/baldwinSPC/glimmer-burnin/issues/180
[#181]: https://github.com/baldwinSPC/glimmer-burnin/issues/181
[docs/dev/new-testkind-playbook.md]: dev/new-testkind-playbook.md

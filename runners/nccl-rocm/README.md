# nccl-rocm — all-reduce bus bandwidth for AMD accelerators

The AMD runner image for the **`nccl`** TestKind (Pair and Group scope). Not a
kind of its own: a profile selects it per node with `imagesByVendor`.

## "The kind is called nccl but AMD uses RCCL"

Correct, and deliberate. The TestKind names a **measurand** — *all-reduce
bus-bandwidth over the fabric* — not a library.

RCCL is AMD's port of NCCL. It declares the same C API (`ncclGetUniqueId`,
`ncclCommInitRank`, `ncclAllReduce`, …), versions itself against NCCL releases
("Compatibility with NCCL 2.30.4"), and reads the same `NCCL_*` environment
variables. `rccl-tests` computes bus bandwidth with the identical `2*(n-1)/n`
factor. So both vendors' images measure the same quantity in the same units and
emit the same metric name, and one profile serves a mixed Spark+Halo fleet.

A separate `rccl` kind would fragment that: every mixed-fleet profile would need
two entries, and a fleet dashboard would need to know which axis to read. The
vocabulary is inherited and slightly awkward on AMD; the alternative is worse.
(A vendor-neutral rename of the kind itself — `collective-bw` — would be cleaner
still, but it is a breaking contract change across every existing profile.)

## It needs ROCm 7.12+, and 7.2.x will not do

The other `-rocm` runners pin 6.4.4. This one cannot.

RCCL did not support gfx1151 until [rocm-systems PR #3415](https://github.com/ROCm/rocm-systems/pull/3415)
(merged 2026-02-26), first shipped in **ROCm 7.12**. The `rocm-7.2.x` line is a
maintenance branch that **never received the fix**, so "ROCm 7.x" is not a
sufficient floor — a 7.2.4 image would carry an RCCL with no kernels for the
target. The Dockerfile asserts this at build time — but it asserts **the property, not
the version**. It greps `librccl.so` for each target's own name, because the
bundled code objects carry their arch strings and a target with no kernels
leaves no mark. A version floor would only be a proxy, and a misleading one:
AMD backports across lines that do not sort the way a comparison assumes, since
`rocm-7.2.x` is newer than 7.0 and still lacks the fix entirely.

(The first version of the check did compare `/opt/rocm/.info/version` against a
7.12 floor. It failed closed on the 7.14 image because that file does not exist
there — right behaviour, wrong question.)

Seven separate defects had to be fixed for gfx1151, which is why an unpatched
build cannot be coaxed into working: the target missing from `DEFAULT_GPUS`,
absent `__gfx1151__` guards, no tuning-table entry, a link failure on
`ncclSymkMask`, a symbol typo, a CMake typo — and a **segfault in the very code
path meant to reject an unsupported architecture**.

Before that, the answer was explicit: AMD, on rccl#2026 (Nov 2025), *"we do not
have plans to support Strix Halo for RCCL."* Any guide still telling you to ship
a patched `librccl.so` is describing the pre-7.12 world.

## AMD's apt repository cannot supply this runtime

Every other `-rocm` runner installs its ROCm runtime from `repo.radeon.com`.
This one cannot, and the reason compounds the version floor above:
**the apt repository tops out at 7.2.4.** There is no 7.12, 7.13 or 7.14
published there at all, and `apt/latest` *is* 7.2.4 — the exact maintenance line
that never received the gfx1151 fix. AMD's container images are a dozen minor
versions ahead of their package repository.

So the runtime libraries are copied from the devel image that compiled the
binary, which also guarantees the runtime is the exact build the link resolved
against. The NVIDIA runner reaches the same destination by a different road: it
builds NCCL from source and links it statically, so no vendor `.so` ships at
all. HIP has no comparable static path, so the `ldd` closure is copied instead —
plus the known dlopen'd companions, since `ldd` cannot see those.

## Transport: TCP/Ethernet, deliberately

This runner is socket-only. That is what AMD themselves support on this
hardware — the ROCm 7.13 notes describe multi-node collectives for Ryzen AI Max
300 *"connected over Ethernet"* across up to 4 nodes, with no RDMA requirement.

The NVIDIA image additionally raises memlock limits and configures `NCCL_IB_*`
for an RDMA fabric. **A RoCE path for AMD is a follow-up, not a silent
omission** — the community E810 setup on Framework Desktops is real and roughly
13× better on latency, and it deserves its own work rather than an untested
branch here.

## A collective of one measures nothing

The bus-bandwidth factor `2*(n-1)/n` is **exactly zero at one rank**. A
single-rank run would print `busbw=0.00` on perfectly healthy hardware and fail
every floor gate forever, so the runner **refuses `nranks < 2`** as a config
error rather than reporting that zero.

This matters on Strix Halo specifically: each box has **one** iGPU, so a
meaningful figure requires a **pair of nodes** — on this SKU the Node-scope path
below always finds one device and skips, which is correct for this hardware, not
a gap. (RCCL does have an AMD-only escape hatch,
`NCCL_MULTI_RANK_GPU_ENABLE=1`, to put two ranks on one device. It is not used
here: it would measure a loopback, not a fabric, while reporting under a metric
name that means fabric.)

## Node scope: intra-node, `ncclCommInitAll`

No `BURNIN_ROLE` and no `BURNIN_NRANKS` means Node scope. What decides whether
this pod may run a collective is a **positive** fact about the hardware — how
many devices HIP reports — never the mere absence of those variables.

- **Fewer than two devices skips**, with the count in the message. On Strix
  Halo (one iGPU per box) this is the ONLY path this runner's shipped hardware
  ever takes.
- **Two or more devices runs the collective**, joined via `ncclCommInitAll`,
  over the collective named by the `collective` variant axis
  (`BURNIN_VARIANT_COLLECTIVE`) — `allreduce` by default, or `allgather`,
  `reducescatter`, `alltoall`.

This is aimed at a **different** AMD SKU than the rest of this file: a
multi-GPU-per-box part (MI300X-class), which is what AMD's CVF gates intra-node
all-reduce bandwidth on. This repository has no such hardware. `ranks` is
reported (Evidence); `ncclTransport` is **not** — unlike NVIDIA's `nccl`, this
runner has no wrapper process to capture `RCCL_DEBUG=INFO` from, and adding a
same-process stderr-capture trick was judged not worth the risk for a label
that never gates anything.

Same naming decision as NVIDIA's runner: `busBandwidthGBs`/`algBandwidthGBs`
keep their names at every scope and for every collective — see
`pkg/contract/metrics.go`'s doc comment.

**`BURNIN_DURATION_SECONDS` bounds this sweep, and NVIDIA's `nccl` does not
follow the same shape.** This file's cross-node path has always swept sizes
geometrically until the duration budget ran out, so the new Node-scope path
follows that pre-existing convention; `nccl_pair.cu`'s Node-scope path instead
follows NCCL's own pre-existing convention of a fixed 5-size list, unbounded by
duration, matching how its Pair/Group paths have always worked (there,
`BURNIN_DURATION_SECONDS` bounds only the rendezvous wait). Neither vendor's
Node-scope path invented this difference — each inherited its own file's
established sweep shape — but it means a profile setting `durationSeconds` on
`nccl` gets a wall-clock-bounded sweep on AMD and a fixed one on NVIDIA. This
is accepted rather than papered over: harmonising the two would mean rewriting
one vendor's PRE-EXISTING cross-node sweep, which is out of scope for the
Node-scope addition that motivated this note.

At **n=2 the factor is exactly 1**, so `busBandwidthGBs == algBandwidthGBs` for
a Pair. That is not a rounding artifact — a two-rank all-reduce moves each byte
across the link once in each direction.

## Rendezvous: a connection is not a rank

Ported from the NVIDIA harness unchanged, including the part that was expensive
to learn. A kubelet `tcpSocket` probe is connect-then-close, and a `send()` into
the socket buffer succeeds even after the peer has sent FIN — so **a probe was
indistinguishable from a rank that received the bootstrap handle**. Rank 0 would
close its listener believing every peer was served while a real rank got
ECONNREFUSED, and the collective blocked until something killed it.

Two defences are kept: peers must send a 4-byte hello before being counted, and
the readiness listener is on a **different port** from the bootstrap listener.

## Verdicts

**Pass (0)** — the collective ran across every rank and every element equalled
the rank count.

**Fail (1)** — **incorrect data only.** An all-reduce that ran fast and summed
wrong is the one hardware verdict this runner returns.

**Error (3)** — bootstrap failure, HIP/RCCL API failure, fewer than two ranks,
or no size producing a timed result. Hardware unjudged, retry available.

**Skip (2)** — a Node-scope invocation with fewer than two devices — every
Strix Halo box, one iGPU each.

## Status — read before pinning

- **NOT VERIFIED ON HARDWARE.** The bandwidth arithmetic and sweep selection are
  unit-tested without a GPU (`collective_bw_test.cc`); the HIP/RCCL path is
  compile-verified only. Tracked in issue #320. The Node-scope `>=2`-device
  path additionally has no MI300X-class hardware to run on at all — see issue
  #406.
- **AMD's own multi-node validation on gfx1151 is thin.** PR #3415's test plan
  states testing was on *a single gfx1151 GPU only; multi-GPU and multi-node
  cluster validation is still pending.* The only two-node evidence is a
  community PR (11 of 12 rccl-tests passing, ~6.4 GB/s busbw at 128 MiB). Treat
  that figure as an order-of-magnitude reference, not a baseline.
- **No published image, no registered default**; amd64-only (issue #319).
- Licences: RCCL and rccl-tests are BSD-3-Clause; the HIP runtime is MIT.
